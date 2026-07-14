from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_SCRIPT = ROOT / "scripts/validate_motion_schema2_fk_consume_activation.py"
RUNNER_SCRIPT = ROOT / "scripts/run_motion_schema2_fk_consume_once.py"
RECEIPT = ROOT / "configs/motion_backhand_loop_bc_schema2_fk_runtime_inspection_receipt_20260714.json"
ACTIVATION = ROOT / "configs/motion_backhand_loop_bc_schema2_fk_consume_activation_20260714.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V = _load_module("validate_motion_schema2_fk_consume_activation", VALIDATOR_SCRIPT)
R = _load_module("run_motion_schema2_fk_consume_once_test", RUNNER_SCRIPT)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated() -> tuple[dict, dict]:
    receipt = _json(RECEIPT)
    activation = _json(ACTIVATION)
    V.validate_receipt(receipt)
    V.validate_activation(
        activation,
        receipt,
        historical_binding_commit=V.ACTIVATION_CONTRACT_COMMIT,
    )
    return receipt, activation


def _capture(stdout: str = "", stderr: str = "", returncode: int = 0) -> dict:
    out = stdout.encode("utf-8")
    err = stderr.encode("utf-8")
    return {
        "returncode": returncode,
        "stdout": stdout,
        "stdout_bytes": len(out),
        "stdout_sha256": hashlib.sha256(out).hexdigest(),
        "stderr": stderr,
        "stderr_bytes": len(err),
        "stderr_sha256": hashlib.sha256(err).hexdigest(),
    }


def test_native_consume_loader_does_not_adopt_the_new_runner():
    receipt, activation = _validated()
    with pytest.raises(V.ActivationContractError, match="binding changed"):
        V.validate_activation(activation, receipt)
    with pytest.raises(V.ActivationContractError, match="binding changed"):
        R._load_contract(ACTIVATION, V.sha256_file(ACTIVATION))


def test_portable_historical_commit_is_exact_not_caller_selected():
    receipt, activation = _validated()
    with pytest.raises(
        V.ActivationContractError,
        match="not the frozen contract commit",
    ):
        V.validate_activation(
            activation,
            receipt,
            historical_binding_commit="0" * 40,
        )


def test_portable_loader_preserves_frozen_activation_bytes(monkeypatch):
    expected_activation = _json(ACTIVATION)
    expected_receipt = _json(RECEIPT)
    context = {
        "current_checkout": {"path": str(ROOT)},
        "current_runner": {"path": "runner", "bytes": 1, "sha256": "0" * 64},
        "current_source_gate_validator": {
            "path": "validator",
            "bytes": 1,
            "sha256": "0" * 64,
        },
        "runtime_body_order": {"path": "body", "bytes": 1, "sha256": "0" * 64},
        "recorded_source_checkout": expected_activation["source_checkout"],
    }
    monkeypatch.setattr(
        R,
        "_validate_portable_source_context",
        lambda activation, value: (Path("body"), {"path": "runner"}, {"path": "validator"}),
    )
    monkeypatch.setattr(R.gate, "validate_receipt", lambda receipt, *, repo_root: None)
    calls = []

    def capture_activation(activation, receipt, **kwargs):
        calls.append((copy.deepcopy(activation), copy.deepcopy(receipt), kwargs))

    monkeypatch.setattr(R.gate, "validate_activation", capture_activation)
    activation, receipt, _meta = R.load_validated_contract_portably(
        ACTIVATION,
        V.sha256_file(ACTIVATION),
        context,
    )
    assert activation == expected_activation
    assert receipt == expected_receipt
    assert calls == [
        (
            expected_activation,
            expected_receipt,
            {
                "repo_root": ROOT,
                "historical_binding_commit": V.ACTIVATION_CONTRACT_COMMIT,
            },
        )
    ]


def _paths(tmp_path: Path) -> R.AttemptPaths:
    root = tmp_path / "control"
    return R.AttemptPaths(
        control_root=root,
        lock=root / "bc.shared.lock",
        claim=root / "asset.claim.json",
        failure=root / "asset.failure.json",
        success=root / "asset.success.json",
    )


def _failure_builder(claim_binding, claim, phase, error, child):
    return {
        "status": "failed_attempt_consumed_permanently",
        "claim": dict(claim_binding),
        "attempt_id": claim["attempt_id"],
        "phase": phase,
        "error": error,
        "child": child,
    }


def _success_builder(claim_binding, claim, child, output):
    return {
        "status": "complete",
        "claim": dict(claim_binding),
        "attempt_id": claim["attempt_id"],
        "child": dict(child),
        "output": dict(output),
    }


def _flock_worker(lock: str, start, ready, records, asset: str) -> None:
    ready.put(asset)
    start.wait(timeout=5)
    with R.ExclusiveFlock(Path(lock)):
        entered = time.monotonic_ns()
        time.sleep(0.15)
        exited = time.monotonic_ns()
    records.put((asset, entered, exited))


def _synthetic_contract(tmp_path: Path, asset: str = "franco_backhand_loop_b"):
    receipt = _json(RECEIPT)
    activation = copy.deepcopy(_json(ACTIVATION))
    source = tmp_path / "source"
    body_path = source / R.BODY_ORDER_RELATIVE_PATH
    body_path.parent.mkdir(parents=True)
    body_path.write_bytes((ROOT / R.BODY_ORDER_RELATIVE_PATH).read_bytes())
    activation["source_checkout"]["path"] = str(source)
    output = tmp_path / "output"
    control = tmp_path / "control"
    row = activation["assets"][asset]
    row["output_root"] = str(output)
    row["claim_path"] = str(control / f"{asset}.claim.json")
    row["failure_ledger_path"] = str(control / f"{asset}.failure.json")
    row["success_ledger_path"] = str(control / f"{asset}.success.json")
    activation["control"]["root"] = str(control)
    activation["control"]["shared_flock_path"] = str(control / "bc.shared.lock")
    meta = {
        "path": str(tmp_path / "activation.json"),
        "bytes": 123,
        "sha256": "a" * 64,
    }
    return receipt, activation, meta, asset


def _recorded_preflight(activation: dict, receipt: dict, asset: str, meta: dict) -> dict:
    source = Path(activation["source_checkout"]["path"])
    identity = {
        "sys_executable": activation["runtime"]["executable"],
        "resolved_executable": activation["runtime"]["resolved_executable"],
        "python_version": activation["runtime"]["python_version"],
        "prefix": activation["runtime"]["prefix"],
        "base_prefix": activation["runtime"]["base_prefix"],
        "packages": activation["runtime"]["packages"],
        "module_origins": activation["runtime"]["module_origins"],
    }
    tracked = {
        name: {
            "path": str(source / row["path"]),
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for name, row in receipt["inspection_checkout"]["tracked_files"].items()
    }
    row = activation["assets"][asset]
    def current_binding(value: dict) -> dict:
        return {
            "path": str(ROOT / value["path"]),
            "bytes": value["bytes"],
            "sha256": value["sha256"],
        }

    checkout = {
        "path": str(source),
        "commit": activation["source_checkout"]["commit"],
        "detached": True,
        "clean": True,
    }
    return {
        "asset_id": asset,
        "checkout_before": checkout,
        "checkout_after": checkout,
        "tracked_files": tracked,
        "private_files": {
            "source_motion": row["source_motion"],
            "source_materialization_report": row["source_materialization_report"],
            "donor_onnx": {
                key: activation["donor_onnx"][key] for key in ("path", "bytes", "sha256")
            },
        },
        "runtime": {
            "identity": identity,
            "executable": {
                "path": activation["runtime"]["resolved_executable"],
                "bytes": activation["runtime"]["executable_bytes"],
                "sha256": activation["runtime"]["executable_sha256"],
            },
            "probe": _capture(json.dumps(identity, sort_keys=True) + "\n"),
        },
        "inspect": _capture(activation["commands"][asset]["expected_inspect_stdout"] + "\n"),
        "current_contract_files": {
            "activation": meta,
            "inspection_receipt": current_binding(activation["inspection_receipt"]),
            "runner": current_binding(activation["runner"]),
            "source_gate_validator": current_binding(activation["source_gate_validator"]),
        },
        "output_root_absent": True,
        "dynamics_steps": 0,
        "writes": 0,
    }


def _write_npz(activation: dict, asset: str, *, exact: bool = True) -> Path:
    row = activation["assets"][asset]
    output = Path(row["output_root"])
    output.mkdir(parents=True)
    path = output / row["output_motion_filename"]
    if not exact:
        np.savez(path, fps=np.array([50], dtype=np.int64))
        return path
    frames = R.EXPECTED_OUTPUT_FRAMES[asset]
    names = np.asarray(
        [line for line in (ROOT / R.BODY_ORDER_RELATIVE_PATH).read_text().splitlines() if line]
    )
    quaternions = np.zeros((frames, 32, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    np.savez(
        path,
        fps=np.array([50], dtype=np.int64),
        joint_pos=np.zeros((frames, 31), dtype=np.float32),
        joint_vel=np.zeros((frames, 31), dtype=np.float32),
        body_pos_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_quat_w=quaternions,
        body_lin_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=names,
    )
    return path


def _write_report(activation: dict, asset: str, motion: Path) -> Path:
    row = activation["assets"][asset]
    report = {
        "schema_version": 1,
        "status": "complete_exact_schema2_fk_materialization_certificate_blocked",
        "completed_utc": "2026-07-14T00:00:00Z",
        "scope": (
            "exact schema-2 MuJoCo FK materialization only; no L0/L1, table/net, dynamics, "
            "simulator, training, formal-motion or hardware claim"
        ),
        "asset_id": asset,
        "preregistration": {
            "path": activation["commands"][asset]["child_argv"][3],
            "sha256": row["preregistration"]["sha256"],
        },
        "shared_runtime": {
            "path": "configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json",
            "bytes": 5503,
            "sha256": "3d32b146e72029960ebf9cb2777f484804dafc87097e9cd3d0513dc277eed6e8",
        },
        "source_motion": row["source_motion"],
        "source_materialization_report": row["source_materialization_report"],
        "donor": {
            "path": activation["donor_onnx"]["path"],
            "bytes": activation["donor_onnx"]["bytes"],
            "sha256": activation["donor_onnx"]["sha256"],
            "required_metadata_subset_exact": True,
        },
        "vendor_mjcf_closure": {
            "algorithm": "sha256(canonical-json(sorted[{path,bytes,sha256}]))-v1",
            "file_count": 75,
            "total_bytes": 14127373,
            "manifest_sha256": "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de",
            "xml_file_count": 1,
            "include_reference_count": 0,
            "external_file_reference_count": 74,
            "unique_external_file_count": 74,
            "mesh_reference_count": 74,
        },
        "output_motion": R.binding(motion),
        "structure": {
            "input_frames": 91 if asset.endswith("_b") else 98,
            "input_fps": 30,
            "output_frames": R.EXPECTED_OUTPUT_FRAMES[asset],
            "output_fps": 50,
            "hope_frame": "off",
            "kinematics_schema_version": 2,
            "body_pos_point": "link_origin",
            "body_lin_vel_point": "center_of_mass",
            "joint_count": 31,
            "body_count": 32,
            "finite": True,
        },
        "authorization": {
            "schema2_materialized": True,
            "l0_authorized": True,
            "vendor_l1_authorized": False,
            "table_net_authorized": False,
            "dynamics_authorized": False,
            "simulator_authorized": False,
            "training_authorized": False,
            "formal_motion_authorized": False,
            "hardware_authorized": False,
        },
        "next_gate": "independent_L0_static_schema2_audit_then_vendor_L1_self_collision",
    }
    path = Path(row["output_root"]) / row["report_filename"]
    path.write_bytes(R.json_bytes(report))
    return path


def _publish_synthetic_ledgers(receipt: dict, activation: dict, meta: dict, asset: str) -> None:
    paths = R._attempt_paths(activation, asset)
    paths.control_root.mkdir(parents=True, exist_ok=True)
    preflight = _recorded_preflight(activation, receipt, asset, meta)
    claim = R._claim_payload(activation, receipt, asset, meta, preflight)
    claim_binding = R.atomic_write_once(paths.claim, R.json_bytes(claim))
    output = R.validate_materialized_output(activation, asset)
    report_path = Path(activation["assets"][asset]["output_root"]) / activation["assets"][asset]["report_filename"]
    child = _capture(f"[schema2-fk] PASS consume report={report_path}\n")
    success = R._success_payload(activation, asset, claim_binding, claim, child, output)
    R.atomic_write_once(paths.success, R.json_bytes(success))


def test_tracked_v2_contract_and_static_cli_pass():
    receipt, activation = _validated()
    assert activation["schema_version"] == 2
    assert activation["control"]["asset_parallelism"] == 1
    assert activation["authorization"]["direct_materializer_consume_authorized"] is False
    run = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_SCRIPT),
            "--activation",
            str(ACTIVATION),
            "--expected-activation-sha256",
            V.sha256_file(ACTIVATION),
            "static",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "runner_exact=true" in run.stdout
    assert "pod_run=false" in run.stdout
    assert receipt["inspection_checkout"]["commit"] == V.INSPECTION_COMMIT


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["runner"].__setitem__("sha256", "0" * 64),
        lambda value: value["source_gate_validator"].__setitem__("bytes", 1),
        lambda value: value["runtime"]["packages"].__setitem__("mujoco", "3.9.0"),
        lambda value: value["runtime"]["module_origins"].__setitem__("numpy", "/tmp/numpy.py"),
        lambda value: value["assets"]["franco_backhand_loop_b"].__setitem__("attempts_authorized", 2),
        lambda value: value["assets"]["franco_backhand_loop_b"].__setitem__("attempts_started", 1),
        lambda value: value["control"].__setitem__("shared_flock_path", "/tmp/foreign.lock"),
        lambda value: value["execution_contract"].__setitem__("automatic_retry", True),
        lambda value: value["execution_contract"].__setitem__("failure_cleanup_grants_retry", True),
        lambda value: value["commands"]["franco_backhand_loop_c"]["child_argv"].__setitem__(-1, "inspect"),
        lambda value: value["authorization"].__setitem__("direct_materializer_consume_authorized", True),
        lambda value: value["authorization"].__setitem__("training_authorized", True),
    ],
)
def test_v2_activation_mutations_fail_closed(mutate):
    receipt, activation = _validated()
    changed = copy.deepcopy(activation)
    mutate(changed)
    with pytest.raises(V.ActivationContractError):
        V.validate_activation(changed, receipt)


def test_duplicate_json_wrong_outer_sha_and_direct_consume_cli_fail_closed(tmp_path: Path):
    with pytest.raises(V.ActivationContractError, match="duplicate JSON key"):
        V.strict_json_bytes(b'{"schema_version":1,"schema_version":1}', "duplicate")
    copied = tmp_path / "activation.json"
    copied.write_bytes(ACTIVATION.read_bytes())
    with pytest.raises(V.ActivationContractError, match="activation SHA"):
        V.read_bound_json(copied, "0" * 64, "consume activation")
    run = subprocess.run(
        [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--activation",
            str(ACTIVATION),
            "--expected-activation-sha256",
            V.sha256_file(ACTIVATION),
            "--asset",
            "franco_backhand_loop_b",
            "consume",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode != 0
    assert "invalid choice" in run.stderr


def test_atomic_claim_is_no_replace_and_preserves_first_bytes(tmp_path: Path):
    root = tmp_path / "control"
    root.mkdir()
    path = root / "claim.json"
    R.atomic_write_once(path, b"first\n")
    with pytest.raises(R.OneShotRunnerError, match="already exists"):
        R.atomic_write_once(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_child_wrapper_is_synchronous_setsid_and_captures_both_streams(tmp_path: Path):
    program = (
        "import json,os,sys; "
        "print(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),'sid':os.getsid(0)})); "
        "print('diagnostic', file=sys.stderr); sys.exit(7)"
    )
    run, record = R._run_captured(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        environment={},
        start_new_session=True,
    )
    assert run.returncode == 7
    identity = json.loads(record["stdout"])
    assert identity["pid"] == identity["pgid"] == identity["sid"]
    assert record["stderr"] == "diagnostic\n"
    R._validate_capture(record, "captured child")


def test_preflight_failure_does_not_claim_or_start_child(tmp_path: Path):
    paths = _paths(tmp_path)
    child_calls = []
    with pytest.raises(RuntimeError, match="preflight drift"):
        R.execute_once(
            paths=paths,
            build_preflight=lambda: (_ for _ in ()).throw(RuntimeError("preflight drift")),
            build_claim=lambda _: {"attempt_id": "one", "asset_id": "asset"},
            run_child=lambda: child_calls.append(True) or {"returncode": 0},
            validate_output=lambda: {},
            build_failure=_failure_builder,
            build_success=_success_builder,
        )
    assert not paths.claim.exists()
    assert not paths.failure.exists()
    assert not child_calls


def test_claim_precedes_child_and_success_is_completion_last(tmp_path: Path):
    paths = _paths(tmp_path)

    def child():
        assert paths.claim.is_file()
        assert not paths.success.exists()
        return {"returncode": 0}

    result = R.execute_once(
        paths=paths,
        build_preflight=lambda: {"ok": True},
        build_claim=lambda _: {"attempt_id": "one", "asset_id": "asset"},
        run_child=child,
        validate_output=lambda: {"finite": True},
        build_failure=_failure_builder,
        build_success=_success_builder,
    )
    assert result["status"] == "complete"
    assert paths.claim.is_file() and paths.success.is_file()
    assert not paths.failure.exists()


def test_child_failure_spends_attempt_even_if_failure_ledger_is_deleted(tmp_path: Path):
    paths = _paths(tmp_path)
    child_calls = []

    def child():
        child_calls.append(True)
        return {"returncode": 17, "stdout": "kept", "stderr": "root cause"}

    with pytest.raises(R.OneShotRunnerError, match="permanently spent"):
        R.execute_once(
            paths=paths,
            build_preflight=lambda: {"ok": True},
            build_claim=lambda _: {"attempt_id": "one", "asset_id": "asset"},
            run_child=child,
            validate_output=lambda: pytest.fail("must not validate failed child"),
            build_failure=_failure_builder,
            build_success=_success_builder,
        )
    assert paths.claim.is_file() and paths.failure.is_file() and not paths.success.exists()
    failure = json.loads(paths.failure.read_text())
    assert failure["child"] == {"returncode": 17, "stdout": "kept", "stderr": "root cause"}
    paths.failure.unlink()  # hostile cleanup must not restore the spent attempt
    with pytest.raises(R.OneShotRunnerError, match="claim path already exists"):
        R.execute_once(
            paths=paths,
            build_preflight=lambda: {"ok": True},
            build_claim=lambda _: {"attempt_id": "two", "asset_id": "asset"},
            run_child=child,
            validate_output=lambda: {},
            build_failure=_failure_builder,
            build_success=_success_builder,
        )
    assert len(child_calls) == 1


def test_post_child_validation_failure_spends_attempt_and_preserves_output(tmp_path: Path):
    paths = _paths(tmp_path)
    output = tmp_path / "output.bin"

    def child():
        output.write_bytes(b"evidence")
        return {"returncode": 0}

    with pytest.raises(R.OneShotRunnerError, match="permanently spent"):
        R.execute_once(
            paths=paths,
            build_preflight=lambda: {"ok": True},
            build_claim=lambda _: {"attempt_id": "one", "asset_id": "asset"},
            run_child=child,
            validate_output=lambda: (_ for _ in ()).throw(RuntimeError("bad lineage")),
            build_failure=_failure_builder,
            build_success=_success_builder,
        )
    assert output.read_bytes() == b"evidence"
    assert json.loads(paths.failure.read_text())["phase"] == "post_child_validation"


def test_b_and_c_share_one_exclusive_flock(tmp_path: Path):
    context = multiprocessing.get_context("fork")
    start = context.Event()
    ready = context.Queue()
    records = context.Queue()
    lock = tmp_path / "control" / "bc.shared.lock"
    lock.parent.mkdir()
    processes = [
        context.Process(target=_flock_worker, args=(str(lock), start, ready, records, asset))
        for asset in ("B", "C")
    ]
    for process in processes:
        process.start()
    assert {ready.get(timeout=3), ready.get(timeout=3)} == {"B", "C"}
    start.set()
    intervals = [records.get(timeout=5), records.get(timeout=5)]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    intervals.sort(key=lambda item: item[1])
    assert intervals[1][1] >= intervals[0][2]


def test_runtime_module_origin_drift_fails_closed():
    activation = _json(ACTIVATION)
    expected = activation["runtime"]
    observed = {
        "sys_executable": expected["executable"],
        "resolved_executable": expected["resolved_executable"],
        "python_version": expected["python_version"],
        "prefix": expected["prefix"],
        "base_prefix": expected["base_prefix"],
        "packages": expected["packages"],
        "module_origins": copy.deepcopy(expected["module_origins"]),
    }
    R.validate_runtime_probe(expected, observed)
    observed["module_origins"]["mujoco"] = "/tmp/shadow/mujoco.py"
    with pytest.raises(R.OneShotRunnerError, match="origin drift"):
        R.validate_runtime_probe(expected, observed)


def test_checkout_must_be_exact_detached_and_clean(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "tracked").write_text("one\n")
    subprocess.run(["git", "add", "tracked"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "checkout", "--detach", "-q", commit], cwd=repo, check=True)
    R.validate_detached_clean_checkout(repo, commit)
    (repo / "untracked").write_text("drift\n")
    with pytest.raises(R.OneShotRunnerError, match="not clean"):
        R.validate_detached_clean_checkout(repo, commit)
    (repo / "untracked").unlink()
    subprocess.run(["git", "switch", "-q", "-c", "attached"], cwd=repo, check=True)
    with pytest.raises(R.OneShotRunnerError, match="attached"):
        R.validate_detached_clean_checkout(repo, commit)


def test_direct_materializer_output_without_claim_or_success_is_rejected(tmp_path: Path):
    receipt, activation, meta, asset = _synthetic_contract(tmp_path)
    motion = _write_npz(activation, asset)
    _write_report(activation, asset, motion)
    with pytest.raises(R.OneShotRunnerError, match="missing irreversible claim"):
        R.validate_formal_result(activation, receipt, asset, meta)


def test_report_hash_cannot_hide_npz_missing_schema2_fields(tmp_path: Path):
    _receipt, activation, _meta, asset = _synthetic_contract(tmp_path)
    motion = _write_npz(activation, asset, exact=False)
    _write_report(activation, asset, motion)
    with pytest.raises(R.OneShotRunnerError, match="members are missing"):
        R.validate_materialized_output(activation, asset)


def test_formal_result_rejects_success_ledger_when_npz_is_missing(tmp_path: Path):
    receipt, activation, meta, asset = _synthetic_contract(tmp_path)
    output = Path(activation["assets"][asset]["output_root"])
    output.mkdir()
    paths = R._attempt_paths(activation, asset)
    paths.control_root.mkdir()
    preflight = _recorded_preflight(activation, receipt, asset, meta)
    claim = R._claim_payload(activation, receipt, asset, meta, preflight)
    claim_binding = R.atomic_write_once(paths.claim, R.json_bytes(claim))
    report_path = output / activation["assets"][asset]["report_filename"]
    child = _capture(f"[schema2-fk] PASS consume report={report_path}\n")
    success = R._success_payload(activation, asset, claim_binding, claim, child, {"forged": True})
    R.atomic_write_once(paths.success, R.json_bytes(success))
    with pytest.raises(R.OneShotRunnerError, match="missing or unexpected entries"):
        R.validate_formal_result(activation, receipt, asset, meta)


def test_exact_synthetic_npz_claim_and_completion_last_lineage_pass(tmp_path: Path):
    receipt, activation, meta, asset = _synthetic_contract(tmp_path)
    motion = _write_npz(activation, asset)
    _write_report(activation, asset, motion)
    _publish_synthetic_ledgers(receipt, activation, meta, asset)
    evidence = R.validate_formal_result(activation, receipt, asset, meta)
    assert evidence["output"]["npz"]["kinematics_schema_version"] == 2
    assert evidence["output"]["npz"]["finite"] is True


def test_native_same_root_body_order_validation_is_unchanged(tmp_path: Path):
    _receipt, activation, _meta, _asset = _synthetic_contract(tmp_path)
    expected = tuple(
        line
        for line in (ROOT / R.BODY_ORDER_RELATIVE_PATH).read_text().splitlines()
        if line
    )
    assert R._expected_body_names(activation) == expected


def test_claim_preflight_sha_and_child_capture_are_fail_closed(tmp_path: Path):
    receipt, activation, meta, asset = _synthetic_contract(tmp_path)
    motion = _write_npz(activation, asset)
    _write_report(activation, asset, motion)
    _publish_synthetic_ledgers(receipt, activation, meta, asset)
    paths = R._attempt_paths(activation, asset)
    claim = json.loads(paths.claim.read_text())
    success = json.loads(paths.success.read_text())
    paths.claim.unlink()
    paths.success.unlink()
    claim["runtime_preflight_sha256"] = "0" * 64
    claim_binding = R.atomic_write_once(paths.claim, R.json_bytes(claim))
    success["claim"] = claim_binding
    success["child"]["stdout"] += "forged"
    R.atomic_write_once(paths.success, R.json_bytes(success))
    with pytest.raises(R.OneShotRunnerError, match="preflight SHA"):
        R.validate_formal_result(activation, receipt, asset, meta)
