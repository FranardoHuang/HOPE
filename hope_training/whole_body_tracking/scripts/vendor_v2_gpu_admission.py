#!/usr/bin/env python3
"""Fail-closed VendorV2 physical-GPU admission and receipt mechanics."""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence


MAX_VENDOR_V2_COMPUTE_PIDS = 2
MIN_VENDOR_V2_FREE_MEMORY_MIB = 8 * 1024
GPU_RESERVATION_FILENAME = "vendor_v2_gpu_slot_reservation.json"
GPU_NAMESPACE_RECEIPT_FILENAME = "vendor_v2_gpu_namespace_receipt.json"
GPU_NAMESPACE_RECEIPT_ENV = "HOPE_VENDOR_V2_GPU_NAMESPACE_RECEIPT"
GPU_NAMESPACE_RECEIPT_SHA_ENV = "HOPE_VENDOR_V2_GPU_NAMESPACE_RECEIPT_SHA256"


class VendorV2GPUAdmission:
    """Own lock, /proc, nvidia-smi, receipt, and slot-admission mechanics."""

    def __init__(
        self,
        *,
        base: Any,
        schema_version: int,
        claim_kind: str,
        experiment_name: str,
        colocation_spec_key: str,
        physical_ball_semantics: str,
        runtime_source_paths: Sequence[tuple[str, str]],
        launcher_source: str,
        admission_source: str,
        exact_group_source: str,
        exact_group: Any,
        canonical_sha256: Any,
        exact_dict: Any,
        validate_spec: Any,
        output_contract: Any,
        training_argv: Any,
    ) -> None:
        self._B = base
        self.LaunchRefused = base.LaunchRefused
        self.schema_version = schema_version
        self.claim_kind = claim_kind
        self.experiment_name = experiment_name
        self.colocation_spec_key = colocation_spec_key
        self.physical_ball_semantics = physical_ball_semantics
        self.runtime_source_paths = runtime_source_paths
        self.launcher_source = launcher_source
        self.admission_source = admission_source
        self.exact_group_source = exact_group_source
        self._exact_group = exact_group
        self._sleep = time.sleep
        self.canonical_sha256 = canonical_sha256
        self._exact_dict = exact_dict
        self._validate_spec = validate_spec
        self._output_contract = output_contract
        self._training_argv = training_argv

    def _open_gpu_shared_lock(self, lock_path: Path) -> int:
        """Hold the physical-GPU flock shared for one VendorV2 trainer lifetime.

        Legacy launchers take the same file exclusively, so they remain mutually
        exclusive with this opt-in path.  VendorV2 launchers use a short POSIX byte
        lock below to serialize the count-and-reserve transaction while retaining
        this shared flock for their complete trainer lifetime.
        """

        try:
            before = lock_path.lstat()
        except OSError as exc:
            raise self.LaunchRefused(
                "GPU lifetime lock must already exist: %s: %s" % (lock_path, exc)
            ) from exc
        if not stat.S_ISREG(before.st_mode):
            raise self.LaunchRefused("GPU lifetime lock must be a regular file")
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags)
        except OSError as exc:
            raise self.LaunchRefused("cannot open GPU lifetime lock: %s" % exc) from exc
        try:
            opened = os.fstat(descriptor)
            after = lock_path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(after.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise self.LaunchRefused("GPU lock pathname identity changed")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise self.LaunchRefused(
                    "GPU lifetime lock conflicts with another launcher"
                ) from exc
            os.set_inheritable(descriptor, True)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _lock_gpu_admission(self, descriptor: int) -> None:
        try:
            fcntl.lockf(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
                1,
                0,
                os.SEEK_SET,
            )
        except BlockingIOError as exc:
            raise self.LaunchRefused(
                "VendorV2 GPU admission transaction is already running"
            ) from exc

    def _unlock_gpu_admission(self, descriptor: int) -> None:
        fcntl.lockf(descriptor, fcntl.LOCK_UN, 1, 0, os.SEEK_SET)

    def _proc_starttime(self, pid: int, *, proc_root: Path = Path("/proc")) -> int:
        try:
            raw = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise self.LaunchRefused(
                "cannot read process identity for pid=%d" % pid
            ) from exc
        close = raw.rfind(")")
        fields = raw[close + 2 :].split() if close >= 0 else []
        if len(fields) <= 19 or not fields[19].isdigit():
            raise self.LaunchRefused("unparseable /proc stat for pid=%d" % pid)
        value = int(fields[19])
        if value <= 0:
            raise self.LaunchRefused("invalid process starttime for pid=%d" % pid)
        return value

    def _proc_environment(
        self, pid: int, *, proc_root: Path = Path("/proc")
    ) -> dict[str, str]:
        try:
            raw = (proc_root / str(pid) / "environ").read_bytes()
        except OSError as exc:
            raise self.LaunchRefused(
                "cannot read process environment for pid=%d" % pid
            ) from exc
        result: dict[str, str] = {}
        try:
            entries = raw.split(b"\0")
            for entry in entries:
                if not entry:
                    continue
                key, separator, value = entry.partition(b"=")
                if not separator:
                    raise ValueError
                name = key.decode("ascii")
                text = value.decode("utf-8")
                if name in result:
                    raise ValueError
                result[name] = text
        except (UnicodeError, ValueError) as exc:
            raise self.LaunchRefused(
                "unparseable process environment for pid=%d" % pid
            ) from exc
        return result

    def _proc_executable(self, pid: int, *, proc_root: Path = Path("/proc")) -> Path:
        try:
            return (proc_root / str(pid) / "exe").resolve(strict=True)
        except OSError as exc:
            raise self.LaunchRefused(
                "cannot read process executable for pid=%d" % pid
            ) from exc

    def _proc_cmdline(self, pid: int, *, proc_root: Path = Path("/proc")) -> list[str]:
        try:
            raw = (proc_root / str(pid) / "cmdline").read_bytes()
        except OSError as exc:
            raise self.LaunchRefused(
                "cannot read process cmdline for pid=%d" % pid
            ) from exc
        if not raw or not raw.endswith(b"\0"):
            raise self.LaunchRefused("unparseable process cmdline for pid=%d" % pid)
        try:
            result = [item.decode("utf-8") for item in raw[:-1].split(b"\0")]
        except UnicodeError as exc:
            raise self.LaunchRefused(
                "unparseable process cmdline for pid=%d" % pid
            ) from exc
        if not result or any(not item for item in result):
            raise self.LaunchRefused("unparseable process cmdline for pid=%d" % pid)
        return result

    def _stable_regular_bytes(self, path: Path, *, name: str) -> bytes:
        before = self._B._stable_regular_file(path, name=name)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                chunks = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                raw = b"".join(chunks)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            final = path.lstat()
        except OSError as exc:
            raise self.LaunchRefused("%s cannot be read stably" % name) from exc
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_nlink,
        )
        if (
            identity
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_nlink,
            )
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_nlink,
            )
            or identity
            != (
                final.st_dev,
                final.st_ino,
                final.st_size,
                final.st_mtime_ns,
                final.st_nlink,
            )
            or before.st_nlink != 1
        ):
            raise self.LaunchRefused("%s changed while reading" % name)
        return raw

    def _stable_canonical_json(
        self, path: Path, *, name: str
    ) -> tuple[dict[str, Any], str]:
        raw = self._stable_regular_bytes(path, name=name)
        document = self._B._strict_json_bytes(raw, name=name)
        if raw != self._B._canonical_bytes(document) + b"\n":
            raise self.LaunchRefused("%s must be canonical JSON plus newline" % name)
        if type(document) is not dict:
            raise self.LaunchRefused("%s must be a JSON object" % name)
        return document, hashlib.sha256(raw).hexdigest()

    def _owned_launch_group(self, namespace: Path, state_path: Path) -> dict[str, Any]:
        if state_path.parent != namespace or state_path.name != "run.log.launch":
            raise self.LaunchRefused(
                "post-boot cleanup state is outside the current namespace"
            )
        try:
            text = self._stable_regular_bytes(
                state_path, name="current launch state"
            ).decode("utf-8")
        except UnicodeError as exc:
            raise self.LaunchRefused("current launch state is not UTF-8") from exc
        rows: dict[str, str] = {}
        for line in text.splitlines():
            key, separator, value = line.partition("=")
            if not separator or not key or key in rows:
                raise self.LaunchRefused("current launch state is not canonical")
            rows[key] = value
        required = (
            "pid",
            "pgid",
            "leader_starttime_ticks",
            "leader_identity_evidence",
            "ready_utc",
        )
        if any(not rows.get(key) for key in required):
            raise self.LaunchRefused("current launch state lacks ready group identity")
        try:
            pid = int(rows["pid"])
            pgid = int(rows["pgid"])
            starttime = int(rows["leader_starttime_ticks"])
        except ValueError as exc:
            raise self.LaunchRefused(
                "current launch group identity is not numeric"
            ) from exc
        leader_path = Path(rows["leader_identity_evidence"])
        expected_leader_path = Path(str(state_path) + ".leader.json")
        if (
            pid <= 0
            or pid != pgid
            or starttime <= 0
            or leader_path != expected_leader_path
        ):
            raise self.LaunchRefused("current launch group identity differs")
        leader_document, leader_sha = self._stable_canonical_json(
            leader_path, name="current launch leader identity"
        )
        expected_leader = {
            "schema_version": 1,
            "kind": "leader_identity",
            "leader": {
                "pid": pid,
                "pgid": pgid,
                "starttime_ticks": starttime,
            },
        }
        if leader_document != expected_leader:
            raise self.LaunchRefused("current launch leader evidence differs")
        return {
            "pid": pid,
            "pgid": pgid,
            "proc_starttime_ticks": starttime,
            "leader_identity": {"path": str(leader_path), "sha256": leader_sha},
        }

    def _cleanup_post_boot_admission_failure(
        self,
        namespace: Path,
        state_path: Path,
        claim_sha: str,
        admission_error: str,
        *,
        proc_root: Path = Path("/proc"),
    ) -> dict[str, Any]:
        """Terminate only this launch's exact group and publish the failed closure."""

        cleanup: dict[str, Any] = {
            "attempted": True,
            "completed": False,
            "leader": None,
            "term_member_pids": [],
            "term_identity": None,
            "kill_identity": None,
            "residual_member_count": None,
            "error": None,
        }
        try:
            leader = self._owned_launch_group(namespace, state_path)
            cleanup["leader"] = leader
            leader_path = Path(leader["leader_identity"]["path"])
            term_path = Path(str(state_path) + ".pre_term.json")
            kill_path = Path(str(state_path) + ".pre_kill.json")
            term_document = self._exact_group.term_group(
                proc_root, leader_path, term_path
            )
            cleanup["term_member_pids"] = sorted(
                int(item["pid"]) for item in term_document["members"]
            )
            _term_receipt, term_sha = self._stable_canonical_json(
                term_path, name="post-boot TERM identity"
            )
            cleanup["term_identity"] = {
                "path": str(term_path),
                "sha256": term_sha,
            }
            residual = []
            for _ in range(5):
                residual = self._exact_group.verify_residual(proc_root, term_path)
                if not residual:
                    break
                self._sleep(1)
            if residual:
                self._exact_group.kill_residual(proc_root, term_path, kill_path)
                _kill_receipt, kill_sha = self._stable_canonical_json(
                    kill_path, name="post-boot KILL identity"
                )
                cleanup["kill_identity"] = {
                    "path": str(kill_path),
                    "sha256": kill_sha,
                }
                for _ in range(5):
                    residual = self._exact_group.verify_residual(proc_root, term_path)
                    if not residual:
                        break
                    self._sleep(1)
            cleanup["residual_member_count"] = len(residual)
            if residual:
                raise self.LaunchRefused(
                    "current launch group remains after exact TERM/KILL"
                )
            cleanup["completed"] = True
        except Exception as exc:
            cleanup["error"] = "%s: %s" % (type(exc).__name__, exc)

        document = {
            "schema_version": 1,
            "kind": "measured_vendor_v2_post_boot_admission_failure_v1",
            "launch_claim_sha256": claim_sha,
            "namespace": str(namespace),
            "phase": "post_boot",
            "admission_error": admission_error,
            "cleanup": cleanup,
            "diagnostic_unauthorized": True,
            "accepted": False,
        }
        path = namespace / "post_boot_admission_failure.json"
        self._B._write_exclusive_json(path, document)
        return {
            "path": str(path),
            "sha256": self._B.sha256_file(path),
            "cleanup": cleanup,
        }

    def _validate_namespace_claim(
        self,
        namespace: Path,
        claim_sha: str,
        *,
        checkout: Path,
        commit: str,
        gpu_index: int,
        gpu_uuid: str,
        require_colocation_opt_in: bool,
        observed_argv: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        claim, _file_sha = self._stable_canonical_json(
            namespace / "launch_claim.json", name="co-resident launch claim"
        )
        claim = self._exact_dict(
            claim,
            ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
            name="co-resident launch claim",
        )
        payload = claim["canonical_payload"]
        if (
            claim["schema_version"] != self.schema_version
            or claim["kind"] != self.claim_kind
            or claim["launch_claim_sha256"] != claim_sha
            or self.canonical_sha256(payload) != claim_sha
            or type(payload) is not dict
            or type(payload.get("spec")) is not dict
        ):
            raise self.LaunchRefused("co-resident launch claim identity differs")
        payload = self._exact_dict(
            payload,
            (
                "schema_version",
                "kind",
                "diagnostic_unauthorized",
                "formal_evidence_prohibited",
                "promotion_prohibited",
                "resume_prohibited",
                "export_prohibited",
                "deployment_prohibited",
                "hardware_prohibited",
                "single_gpu",
                "max_compute_pids_on_physical_gpu",
                "minimum_free_memory_mib",
                "gpu_default_empty",
                "vendor_v2_colocation_opt_in",
                "fresh_only",
                "reward_materialization_only",
                "policy_recipe_materialization_only",
                "teacher_qdes_oracle_only",
                "ppo_updates_authorized",
                "control_step_action_delay",
                "reset_inverse_solve",
                "physical_ball_semantics",
                "spec_file_sha256",
                "spec",
                "source",
                "runtime_sources",
                "runtime_assets",
                "bundle",
                "materialization_inputs",
                "output_contract",
                "boot_marker",
                "training_argv",
            ),
            name="co-resident canonical payload",
        )
        spec = self._validate_spec(payload["spec"], claimed=True)
        if spec != payload["spec"]:
            raise self.LaunchRefused("co-resident claim spec is not normalized")
        source = spec["source"]
        gpu = spec["gpu"]
        if (
            spec.get("namespace") != str(namespace)
            or source.get("checkout") != str(checkout)
            or source.get("commit_sha") != commit
            or gpu.get("index") != gpu_index
            or gpu.get("uuid") != gpu_uuid
        ):
            raise self.LaunchRefused(
                "co-resident claim does not bind this checkout/GPU"
            )
        if require_colocation_opt_in and spec.get(self.colocation_spec_key) is not True:
            raise self.LaunchRefused(
                "co-resident claim did not opt in to VendorV2 colocation"
            )
        output_contract = self._output_contract(spec)
        fixed_true = (
            "diagnostic_unauthorized",
            "formal_evidence_prohibited",
            "promotion_prohibited",
            "resume_prohibited",
            "export_prohibited",
            "deployment_prohibited",
            "hardware_prohibited",
            "single_gpu",
            "fresh_only",
        )
        if (
            payload["schema_version"] != self.schema_version
            or payload["kind"] != self.claim_kind
            or any(payload[key] is not True for key in fixed_true)
            or payload["max_compute_pids_on_physical_gpu"] != MAX_VENDOR_V2_COMPUTE_PIDS
            or payload["minimum_free_memory_mib"] != MIN_VENDOR_V2_FREE_MEMORY_MIB
            or payload["gpu_default_empty"] is not (not spec[self.colocation_spec_key])
            or payload["vendor_v2_colocation_opt_in"]
            is not spec[self.colocation_spec_key]
            or payload["reward_materialization_only"]
            is not (spec["stage"] == "materialize")
            or payload["policy_recipe_materialization_only"]
            is not (spec["stage"] == "recipe")
            or payload["teacher_qdes_oracle_only"]
            is not (spec["stage"] in ("oracle2", "oracle32"))
            or payload["ppo_updates_authorized"] != output_contract["ppo_update_count"]
            or payload["control_step_action_delay"] != 0
            or payload["reset_inverse_solve"] is not False
            or payload["physical_ball_semantics"] != self.physical_ball_semantics
            or payload["output_contract"] != output_contract
            or payload["boot_marker"] != output_contract["boot_marker"]
        ):
            raise self.LaunchRefused("co-resident claim safety semantics differ")
        self._B._sha256(payload["spec_file_sha256"], name="co-resident spec file SHA")
        expected_source = {
            "checkout": str(checkout),
            "commit_sha": commit,
            "clean": True,
        }
        if payload["source"] != expected_source:
            raise self.LaunchRefused("co-resident clean source claim differs")
        runtime_sources = self._exact_dict(
            payload["runtime_sources"],
            tuple(name for _path, name in self.runtime_source_paths),
            name="co-resident runtime sources",
        )
        for expected_path, name in self.runtime_source_paths:
            pin = self._exact_dict(
                runtime_sources[name],
                ("path", "sha256"),
                name="runtime source %s" % name,
            )
            if pin["path"] != expected_path:
                raise self.LaunchRefused(
                    "co-resident runtime source path differs: %s" % name
                )
            self._B._sha256(pin["sha256"], name="runtime source %s SHA" % name)
        critical_sources = (
            ("VendorV2 N1 launcher", self.launcher_source),
            ("VendorV2 GPU admission", self.admission_source),
            ("exact process-group helper", self.exact_group_source),
        )
        for source_name, source_path in critical_sources:
            critical_pin = runtime_sources[source_name]
            critical_path = checkout / source_path
            self._B._stable_regular_file(
                critical_path, name="co-resident %s" % source_name
            )
            if self._B.sha256_file(critical_path) != critical_pin["sha256"]:
                raise self.LaunchRefused(
                    "co-resident %s bytes differ from claim" % source_name
                )
        if type(payload["bundle"]) is not dict:
            raise self.LaunchRefused("co-resident claim bundle must be an object")
        expected_argv = self._training_argv(spec, payload["bundle"])
        if payload["training_argv"] != expected_argv:
            raise self.LaunchRefused(
                "co-resident training argv differs from exact recipe"
            )
        if observed_argv is not None and list(observed_argv) != expected_argv:
            raise self.LaunchRefused(
                "co-resident /proc cmdline differs from exact training argv"
            )
        if expected_argv[1] != str(
            checkout / self._B.WBT_RELATIVE / "scripts/train.py"
        ):
            raise self.LaunchRefused("co-resident training entrypoint differs")
        return {"spec": spec, "training_argv": expected_argv}

    def _validate_runtime_gpu_process(
        self,
        process: Mapping[str, Any],
        *,
        checkout: Path,
        commit: str,
        gpu_index: int,
        gpu_uuid: str,
        current_namespace: Path | None,
        proc_root: Path = Path("/proc"),
    ) -> dict[str, Any]:
        pid = process["pid"]
        start_before = self._proc_starttime(pid, proc_root=proc_root)
        environment = self._proc_environment(pid, proc_root=proc_root)
        receipt_text = environment.get(GPU_NAMESPACE_RECEIPT_ENV)
        receipt_sha = environment.get(GPU_NAMESPACE_RECEIPT_SHA_ENV)
        claim_sha = environment.get("HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256")
        if receipt_text is None or receipt_sha is None or claim_sha is None:
            raise self.LaunchRefused(
                "unknown GPU co-resident pid=%d: VendorV2 receipt environment is absent"
                % pid
            )
        receipt_path = self._B._absolute_path(
            receipt_text, name="co-resident namespace receipt", must_exist=True
        )
        receipt, observed_receipt_sha = self._stable_canonical_json(
            receipt_path, name="co-resident namespace receipt"
        )
        if observed_receipt_sha != self._B._sha256(
            receipt_sha, name="co-resident namespace receipt SHA"
        ):
            raise self.LaunchRefused("co-resident namespace receipt SHA differs")
        receipt = self._exact_dict(
            receipt,
            (
                "schema_version",
                "kind",
                "pid",
                "proc_starttime_ticks",
                "gpu_index",
                "gpu_uuid",
                "namespace",
                "checkout",
                "commit_sha",
                "wbt_cwd",
                "launch_claim_sha256",
                "max_compute_pids",
                "minimum_free_memory_mib",
                "allow_vendor_v2_colocation",
            ),
            name="co-resident namespace receipt",
        )
        namespace = Path(receipt["namespace"])
        expected_receipt = namespace / GPU_NAMESPACE_RECEIPT_FILENAME
        if (
            receipt["schema_version"] != 1
            or receipt["kind"] != "measured_vendor_v2_gpu_namespace_receipt_v1"
            or receipt["pid"] != pid
            or receipt["proc_starttime_ticks"] != start_before
            or receipt["gpu_index"] != gpu_index
            or receipt["gpu_uuid"] != gpu_uuid
            or receipt["checkout"] != str(checkout)
            or receipt["commit_sha"] != commit
            or receipt["wbt_cwd"] != str(checkout / self._B.WBT_RELATIVE)
            or receipt["launch_claim_sha256"] != claim_sha
            or receipt["max_compute_pids"] != MAX_VENDOR_V2_COMPUTE_PIDS
            or receipt["minimum_free_memory_mib"] != MIN_VENDOR_V2_FREE_MEMORY_MIB
            or receipt_path != expected_receipt
            or namespace.parent.name != self.experiment_name
        ):
            raise self.LaunchRefused("co-resident namespace receipt identity differs")
        require_opt_in = current_namespace is None or namespace != current_namespace
        if require_opt_in and receipt["allow_vendor_v2_colocation"] is not True:
            raise self.LaunchRefused("co-resident namespace receipt did not opt in")
        try:
            cwd = (proc_root / str(pid) / "cwd").resolve(strict=True)
        except OSError as exc:
            raise self.LaunchRefused(
                "cannot verify co-resident cwd for pid=%d" % pid
            ) from exc
        executable = self._proc_executable(pid, proc_root=proc_root)
        cmdline = self._proc_cmdline(pid, proc_root=proc_root)
        claim_info = self._validate_namespace_claim(
            namespace,
            claim_sha,
            checkout=checkout,
            commit=commit,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            require_colocation_opt_in=require_opt_in,
            observed_argv=cmdline,
        )
        expected_executable = Path(
            claim_info["spec"]["source"]["isaac_python"]
        ).resolve(strict=True)
        start_after = self._proc_starttime(pid, proc_root=proc_root)
        if (
            start_after != start_before
            or cwd != checkout / self._B.WBT_RELATIVE
            or executable != expected_executable
        ):
            raise self.LaunchRefused(
                "co-resident process identity/cwd/executable drifted"
            )
        return {
            "pid": pid,
            "process_name": process["process_name"],
            "gpu_uuid": gpu_uuid,
            "used_gpu_memory_mib": process["used_gpu_memory_mib"],
            "namespace": str(namespace),
            "namespace_receipt": {
                "path": str(receipt_path),
                "sha256": observed_receipt_sha,
            },
            "launch_claim_sha256": claim_sha,
            "proc_starttime_ticks": start_before,
            "executable": str(executable),
            "cmdline_sha256": hashlib.sha256(
                b"\0".join(item.encode() for item in cmdline)
            ).hexdigest(),
        }

    def _query_gpu_processes(self, index: int, uuid: str) -> dict[str, Any]:
        nvidia_smi, binary_sha = self._B._trusted_nvidia_smi()
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        identity = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if identity.returncode != 0:
            raise self.LaunchRefused(
                "nvidia-smi identity query failed: %s" % identity.stderr.strip()
            )
        observed: dict[int, dict[str, Any]] = {}
        for line in identity.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if (
                len(parts) != 4
                or not parts[0].isdigit()
                or not parts[2].isdigit()
                or not parts[3].isdigit()
            ):
                raise self.LaunchRefused("unparseable GPU identity row: %r" % line)
            total = int(parts[2])
            free = int(parts[3])
            if total <= 0 or free < 0 or free > total:
                raise self.LaunchRefused("invalid GPU memory row: %r" % line)
            observed[int(parts[0])] = {
                "uuid": parts[1],
                "total_memory_mib": total,
                "free_memory_mib": free,
            }
        selected = observed.get(index)
        if selected is None or selected["uuid"] != uuid:
            raise self.LaunchRefused(
                "GPU %d UUID differs: expected=%s, actual=%r"
                % (index, uuid, None if selected is None else selected["uuid"])
            )
        occupancy = subprocess.run(
            [
                nvidia_smi,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if occupancy.returncode != 0:
            raise self.LaunchRefused(
                "nvidia-smi compute query failed: %s" % occupancy.stderr.strip()
            )
        processes = []
        seen_pids: set[int] = set()
        for line in occupancy.stdout.splitlines():
            if not line.strip():
                continue
            parts = [item.strip() for item in line.rsplit(",", 1)]
            prefix = (
                [item.strip() for item in parts[0].split(",", 2)]
                if len(parts) == 2
                else []
            )
            if (
                len(parts) != 2
                or len(prefix) != 3
                or not prefix[0].startswith("GPU-")
                or not prefix[1].isdigit()
                or not parts[1].isdigit()
            ):
                raise self.LaunchRefused("unparseable GPU compute row: %r" % line)
            if prefix[0] != uuid:
                continue
            pid = int(prefix[1])
            if pid <= 0 or pid in seen_pids:
                raise self.LaunchRefused("duplicate/invalid GPU compute pid=%d" % pid)
            seen_pids.add(pid)
            processes.append(
                {
                    "pid": pid,
                    "process_name": prefix[2],
                    "used_gpu_memory_mib": int(parts[1]),
                }
            )
        return {
            "index": index,
            "uuid": uuid,
            "nvidia_smi_path": nvidia_smi,
            "nvidia_smi_sha256": binary_sha,
            "total_memory_mib": selected["total_memory_mib"],
            "free_memory_mib": selected["free_memory_mib"],
            "processes": processes,
        }

    def _live_runtime_handoff(
        self,
        namespace: Path,
        *,
        checkout: Path,
        commit: str,
        gpu_index: int,
        gpu_uuid: str,
        claim_sha: str,
        proc_root: Path,
    ) -> dict[str, int] | None:
        """Keep a reserved slot live if the outer launcher dies after handoff."""

        path = namespace / GPU_NAMESPACE_RECEIPT_FILENAME
        if not path.exists():
            return None
        receipt, _receipt_sha = self._stable_canonical_json(
            path, name="VendorV2 runtime handoff receipt"
        )
        pid = receipt.get("pid")
        expected_start = receipt.get("proc_starttime_ticks")
        if (
            type(pid) is not int
            or isinstance(pid, bool)
            or pid <= 0
            or type(expected_start) is not int
            or isinstance(expected_start, bool)
            or expected_start <= 0
        ):
            raise self.LaunchRefused(
                "VendorV2 runtime handoff process identity is invalid"
            )
        try:
            start = self._proc_starttime(pid, proc_root=proc_root)
        except self.LaunchRefused:
            return None
        if start != expected_start:
            return None
        receipt = self._exact_dict(
            receipt,
            (
                "schema_version",
                "kind",
                "pid",
                "proc_starttime_ticks",
                "gpu_index",
                "gpu_uuid",
                "namespace",
                "checkout",
                "commit_sha",
                "wbt_cwd",
                "launch_claim_sha256",
                "max_compute_pids",
                "minimum_free_memory_mib",
                "allow_vendor_v2_colocation",
            ),
            name="VendorV2 runtime handoff receipt",
        )
        if (
            receipt["schema_version"] != 1
            or receipt["kind"] != "measured_vendor_v2_gpu_namespace_receipt_v1"
            or receipt["proc_starttime_ticks"] != start
            or receipt["gpu_index"] != gpu_index
            or receipt["gpu_uuid"] != gpu_uuid
            or receipt["namespace"] != str(namespace)
            or receipt["checkout"] != str(checkout)
            or receipt["commit_sha"] != commit
            or receipt["wbt_cwd"] != str(checkout / self._B.WBT_RELATIVE)
            or receipt["launch_claim_sha256"] != claim_sha
            or receipt["max_compute_pids"] != MAX_VENDOR_V2_COMPUTE_PIDS
            or receipt["minimum_free_memory_mib"] != MIN_VENDOR_V2_FREE_MEMORY_MIB
        ):
            raise self.LaunchRefused("live VendorV2 runtime handoff identity differs")
        try:
            cwd = (proc_root / str(pid) / "cwd").resolve(strict=True)
        except OSError as exc:
            raise self.LaunchRefused(
                "cannot verify VendorV2 runtime handoff cwd"
            ) from exc
        if cwd != checkout / self._B.WBT_RELATIVE:
            raise self.LaunchRefused("VendorV2 runtime handoff cwd differs")
        self._validate_namespace_claim(
            namespace,
            claim_sha,
            checkout=checkout,
            commit=commit,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            require_colocation_opt_in=False,
        )
        return {"pid": pid, "proc_starttime_ticks": start}

    def _live_reservations(
        self,
        experiment_root: Path,
        *,
        checkout: Path,
        commit: str,
        gpu_index: int,
        gpu_uuid: str,
        proc_root: Path = Path("/proc"),
    ) -> list[dict[str, Any]]:
        result = []
        try:
            children = tuple(experiment_root.iterdir())
        except OSError as exc:
            raise self.LaunchRefused("cannot scan VendorV2 namespace root") from exc
        for namespace in children:
            receipt_path = namespace / GPU_RESERVATION_FILENAME
            if not receipt_path.exists():
                continue
            receipt, receipt_sha = self._stable_canonical_json(
                receipt_path, name="VendorV2 GPU reservation"
            )
            receipt_gpu_index = receipt.get("gpu_index")
            receipt_gpu_uuid = receipt.get("gpu_uuid")
            if (
                type(receipt_gpu_index) is not int
                or isinstance(receipt_gpu_index, bool)
                or receipt_gpu_index < 0
                or type(receipt_gpu_uuid) is not str
                or not receipt_gpu_uuid.startswith("GPU-")
            ):
                raise self.LaunchRefused(
                    "VendorV2 GPU reservation physical-GPU identity is invalid"
                )
            if receipt_gpu_index != gpu_index or receipt_gpu_uuid != gpu_uuid:
                continue
            pid = receipt.get("owner_pid")
            expected_start = receipt.get("owner_proc_starttime_ticks")
            if (
                type(pid) is not int
                or isinstance(pid, bool)
                or pid <= 0
                or type(expected_start) is not int
                or isinstance(expected_start, bool)
                or expected_start <= 0
            ):
                raise self.LaunchRefused(
                    "VendorV2 GPU reservation process identity is invalid"
                )
            try:
                live_start = self._proc_starttime(pid, proc_root=proc_root)
            except self.LaunchRefused:
                live_start = None
            owner_kind = "outer_launcher"
            if expected_start != live_start:
                handoff = self._live_runtime_handoff(
                    namespace,
                    checkout=checkout,
                    commit=commit,
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuid,
                    claim_sha=receipt.get("launch_claim_sha256"),
                    proc_root=proc_root,
                )
                if handoff is None:
                    continue
                pid = handoff["pid"]
                live_start = handoff["proc_starttime_ticks"]
                owner_kind = "runtime_handoff"
            receipt = self._exact_dict(
                receipt,
                (
                    "schema_version",
                    "kind",
                    "owner_pid",
                    "owner_proc_starttime_ticks",
                    "gpu_index",
                    "gpu_uuid",
                    "namespace",
                    "checkout",
                    "commit_sha",
                    "launch_claim_sha256",
                    "max_compute_pids",
                    "minimum_free_memory_mib",
                    "allow_vendor_v2_colocation",
                ),
                name="VendorV2 GPU reservation",
            )
            if (
                receipt["schema_version"] != 1
                or receipt["kind"] != "measured_vendor_v2_gpu_slot_reservation_v1"
                or receipt["gpu_index"] != gpu_index
                or receipt["gpu_uuid"] != gpu_uuid
                or receipt["namespace"] != str(namespace)
                or receipt["checkout"] != str(checkout)
                or receipt["commit_sha"] != commit
                or receipt["max_compute_pids"] != MAX_VENDOR_V2_COMPUTE_PIDS
                or receipt["minimum_free_memory_mib"] != MIN_VENDOR_V2_FREE_MEMORY_MIB
            ):
                raise self.LaunchRefused(
                    "live VendorV2 GPU reservation identity differs"
                )
            self._validate_namespace_claim(
                namespace,
                receipt["launch_claim_sha256"],
                checkout=checkout,
                commit=commit,
                gpu_index=gpu_index,
                gpu_uuid=gpu_uuid,
                require_colocation_opt_in=False,
            )
            result.append(
                {
                    "owner_pid": pid,
                    "owner_proc_starttime_ticks": live_start,
                    "reservation_owner_kind": owner_kind,
                    "namespace": str(namespace),
                    "reservation_receipt": {
                        "path": str(receipt_path),
                        "sha256": receipt_sha,
                    },
                    "allow_vendor_v2_colocation": receipt["allow_vendor_v2_colocation"],
                }
            )
        return result

    def _verify_gpu_admission(
        self,
        spec: Mapping[str, Any],
        *,
        phase: str,
        current_namespace: Path | None,
        require_current_compute: bool = False,
        proc_root: Path = Path("/proc"),
        query_gpu_processes: Any = None,
        validate_runtime_gpu_process: Any = None,
        live_reservations: Any = None,
    ) -> dict[str, Any]:
        checkout = Path(spec["source"]["checkout"])
        commit = spec["source"]["commit_sha"]
        gpu = spec["gpu"]
        query = (
            self._query_gpu_processes
            if query_gpu_processes is None
            else query_gpu_processes
        )
        validate_process = (
            self._validate_runtime_gpu_process
            if validate_runtime_gpu_process is None
            else validate_runtime_gpu_process
        )
        reservations_query = (
            self._live_reservations if live_reservations is None else live_reservations
        )
        queried = query(gpu["index"], gpu["uuid"])
        if queried["free_memory_mib"] < MIN_VENDOR_V2_FREE_MEMORY_MIB:
            raise self.LaunchRefused(
                "GPU %d free memory %d MiB is below conservative headroom %d MiB"
                % (
                    gpu["index"],
                    queried["free_memory_mib"],
                    MIN_VENDOR_V2_FREE_MEMORY_MIB,
                )
            )
        if len(queried["processes"]) > MAX_VENDOR_V2_COMPUTE_PIDS:
            raise self.LaunchRefused(
                "GPU %d has %d compute PIDs; max is %d"
                % (gpu["index"], len(queried["processes"]), MAX_VENDOR_V2_COMPUTE_PIDS)
            )
        verified = [
            validate_process(
                process,
                checkout=checkout,
                commit=commit,
                gpu_index=gpu["index"],
                gpu_uuid=gpu["uuid"],
                current_namespace=current_namespace,
                proc_root=proc_root,
            )
            for process in queried["processes"]
        ]
        process_namespaces = {row["namespace"] for row in verified}
        if len(process_namespaces) != len(verified):
            raise self.LaunchRefused(
                "one VendorV2 namespace owns multiple compute PIDs"
            )
        reservations = reservations_query(
            Path(spec["namespace"]).parent,
            checkout=checkout,
            commit=commit,
            gpu_index=gpu["index"],
            gpu_uuid=gpu["uuid"],
            proc_root=proc_root,
        )
        reserved_namespaces = {row["namespace"] for row in reservations}
        if len(reserved_namespaces) != len(reservations):
            raise self.LaunchRefused("duplicate live VendorV2 namespace reservation")
        active_namespaces = process_namespaces | reserved_namespaces
        current_text = None if current_namespace is None else str(current_namespace)
        other_namespaces = active_namespaces - (
            {current_text} if current_text else set()
        )
        allow_colocation = spec[self.colocation_spec_key]
        if other_namespaces and not allow_colocation:
            raise self.LaunchRefused(
                "GPU is not empty and this exact claim did not opt in"
            )
        for reservation in reservations:
            if (
                reservation["namespace"] in other_namespaces
                and reservation["allow_vendor_v2_colocation"] is not True
            ):
                raise self.LaunchRefused(
                    "existing VendorV2 reservation did not opt in to colocation"
                )
        if len(other_namespaces) >= MAX_VENDOR_V2_COMPUTE_PIDS:
            raise self.LaunchRefused(
                "VendorV2 GPU admission has no free compute-PID slot"
            )
        if (
            current_namespace is None
            and len(active_namespaces) >= MAX_VENDOR_V2_COMPUTE_PIDS
        ):
            raise self.LaunchRefused(
                "VendorV2 GPU admission would exceed two compute PIDs"
            )
        if current_namespace is not None and current_text not in reserved_namespaces:
            raise self.LaunchRefused(
                "current VendorV2 namespace has no live slot reservation"
            )
        if require_current_compute and current_text not in process_namespaces:
            raise self.LaunchRefused(
                "current VendorV2 namespace has no verified compute PID"
            )
        return {
            "schema_version": 1,
            "kind": "measured_vendor_v2_gpu_admission_snapshot_v1",
            "phase": phase,
            "gpu_index": gpu["index"],
            "gpu_uuid": gpu["uuid"],
            "allow_vendor_v2_colocation": allow_colocation,
            "max_compute_pids": MAX_VENDOR_V2_COMPUTE_PIDS,
            "minimum_free_memory_mib": MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "total_memory_mib": queried["total_memory_mib"],
            "free_memory_mib": queried["free_memory_mib"],
            "free_memory_headroom_mib": (
                queried["free_memory_mib"] - MIN_VENDOR_V2_FREE_MEMORY_MIB
            ),
            "compute_process_count": len(verified),
            "compute_processes": verified,
            "live_reservation_count": len(reservations),
            "live_reservations": reservations,
            "nvidia_smi_path": queried["nvidia_smi_path"],
            "nvidia_smi_sha256": queried["nvidia_smi_sha256"],
        }

    def _reservation_document(
        self, spec: Mapping[str, Any], claim_sha: str
    ) -> dict[str, Any]:
        pid = os.getpid()
        return {
            "schema_version": 1,
            "kind": "measured_vendor_v2_gpu_slot_reservation_v1",
            "owner_pid": pid,
            "owner_proc_starttime_ticks": self._proc_starttime(pid),
            "gpu_index": spec["gpu"]["index"],
            "gpu_uuid": spec["gpu"]["uuid"],
            "namespace": spec["namespace"],
            "checkout": spec["source"]["checkout"],
            "commit_sha": spec["source"]["commit_sha"],
            "launch_claim_sha256": claim_sha,
            "max_compute_pids": MAX_VENDOR_V2_COMPUTE_PIDS,
            "minimum_free_memory_mib": MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "allow_vendor_v2_colocation": spec[self.colocation_spec_key],
        }

    def _runtime_namespace_receipt(
        self, spec: Mapping[str, Any], claim_sha: str
    ) -> tuple[Path, str]:
        pid = os.getpid()
        path = Path(spec["namespace"]) / GPU_NAMESPACE_RECEIPT_FILENAME
        document = {
            "schema_version": 1,
            "kind": "measured_vendor_v2_gpu_namespace_receipt_v1",
            "pid": pid,
            "proc_starttime_ticks": self._proc_starttime(pid),
            "gpu_index": spec["gpu"]["index"],
            "gpu_uuid": spec["gpu"]["uuid"],
            "namespace": spec["namespace"],
            "checkout": spec["source"]["checkout"],
            "commit_sha": spec["source"]["commit_sha"],
            "wbt_cwd": str(Path(spec["source"]["checkout"]) / self._B.WBT_RELATIVE),
            "launch_claim_sha256": claim_sha,
            "max_compute_pids": MAX_VENDOR_V2_COMPUTE_PIDS,
            "minimum_free_memory_mib": MIN_VENDOR_V2_FREE_MEMORY_MIB,
            "allow_vendor_v2_colocation": spec[self.colocation_spec_key],
        }
        self._B._write_exclusive_json(path, document)
        return path, self._B.sha256_file(path)
