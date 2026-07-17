from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_ready_to_strike_join_ladder_stage2.py"
SPEC = importlib.util.spec_from_file_location("ready_stage2_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
REAL_COLLECT_PRIOR_V2_INPUTS = runner._collect_prior_v2_inputs
REAL_RUN_GIT_READONLY = runner._run_git_readonly
REAL_INSPECT_TOPP_RUNTIME = runner._inspect_topp_runtime
REAL_OBSERVE_MJCF_RUNTIME_PREFLIGHT = runner._observe_mjcf_runtime_preflight
REAL_PREFLIGHT_MJCF_RUNTIME = runner._preflight_mjcf_runtime
REAL_COLLECT_DYNAMIC_DEPENDENCY_CLOSURE = runner._collect_dynamic_dependency_closure


def _canonical(document: object) -> bytes:
    return (json.dumps(document, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    queue = json.loads((REPO / "configs/ready_to_strike_join_ladder_20260717.yaml").read_text())
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    generator_payload = b"attested-stage1-generator\n"
    _write(stage1 / "build_ready_to_strike_motion.py", generator_payload)
    queue["runtime"]["generator_sha256"] = _sha(generator_payload)
    runtime = tmp_path / "runtime"
    queue["runtime"]["checkout_path"] = str(runtime)
    runtime_payloads: dict[str, bytes] = {}
    for index, (key, relative) in enumerate(runner.RUNTIME_RELATIVE_PATHS.items()):
        payload = f"runtime-{key}-{index}\n".encode()
        runtime_payloads[str(relative)] = payload
        queue["runtime"][key] = _sha(payload)
    for index, relative in enumerate(runner.TOPP_CLOSURE_PATHS):
        runtime_payloads.setdefault(str(relative), f"closure-{index}-{relative}\n".encode())
    for relative, payload in runtime_payloads.items():
        _write(runtime / relative, payload)

    for name, asset in queue["assets"].items():
        asset_path = tmp_path / f"{name}.npz"
        payload = f"asset-{name}\n".encode()
        _write(asset_path, payload)
        asset["path"] = str(asset_path)
        asset["sha256"] = _sha(payload)

    queue_path = tmp_path / "queue.json"
    queue_payload = _canonical(queue)
    _write(queue_path, queue_payload)
    queue_sha = _sha(queue_payload)
    monkeypatch.setattr(runner, "EXPECTED_QUEUE_SHA256", queue_sha)

    stage2 = tmp_path / "stage2"
    observations = []
    for cell_id, values in runner.EXPECTED_OBSERVATIONS.items():
        action, ready, delta, candidate_sha, cert_sha, timing = values
        observations.append({
            "cell_id": cell_id,
            "action": action,
            "ready_source": ready,
            "delta": delta,
            "candidate_sha256": candidate_sha,
            "topp_certificate_sha256": cert_sha,
            "start_to_contact_s": timing,
        })

    launch_snapshot = {"mode": "execute", "argv": ["attest"]}
    observation_by_id = {row["cell_id"]: row for row in observations}
    receipt_cells = []
    for cell_id, (action, ready, delta) in runner.EXPECTED_STAGE1_CELLS.items():
        observation = observation_by_id[cell_id]
        timing = observation["start_to_contact_s"]
        receipt_cells.append({
            "cell_id": cell_id,
            "action": action,
            "ready_source": ready,
            "delta": delta,
            "join_frame": queue["assets"][action]["contact_frame"] - delta,
            "blend_intervals": 22 - delta,
            "candidate_sha256": observation["candidate_sha256"],
            "generator_contract_sha256": "1" * 64,
            "output_sha256": "2" * 64,
            "certificate_sha256": observation["topp_certificate_sha256"],
            "candidate_start_to_contact_s": timing,
            "within_0p5_s": timing <= 0.5,
            "doses": {"cop": 0.01, "friction": 0.02, "torque": 0.0},
        })
    receipt = {
        "schema_version": 1,
        "artifact_kind": "ready_to_strike_stage1_historical_attestation",
        "status": "historical_evidence_attested_no_runtime_authority",
        "experiment_id": runner.EXPECTED_EXPERIMENT_ID,
        "inputs": {
            "root": str(stage1),
            "queue_path": str(queue_path),
            "queue_sha256": queue_sha,
            "generator_sha256": queue["runtime"]["generator_sha256"],
            "summary_sha256": "3" * 64,
            "runtime_source_commit": queue["runtime"]["checkout_commit"],
        },
        "cells": receipt_cells,
        "attestor": {
            "source_sha256": "4" * 64,
            "launch_snapshot": launch_snapshot,
            "launch_snapshot_sha256": _sha(_canonical(launch_snapshot)),
            "source_and_inputs_unchanged_before_publish": True,
        },
        "formal_claims": {
            "physics_replay_exact": False,
            "source_closure_exact": False,
            "mjcf_closure_exact": False,
            "screening_activation_evidence_only": True,
        },
        "runtime_authority": {
            "read_only_historical": True,
            "ssh": False,
            "process_signal": False,
            "automatic_retry": False,
            "simulator": False,
            "trainer": False,
            "deployment": False,
            "robot_command": False,
        },
    }
    receipt_path = stage1 / "stage1_historical_attestation.json"
    receipt_payload = _canonical(receipt)
    _write(receipt_path, receipt_payload)

    prior_root = tmp_path / "prior-stage2"
    prior_root.mkdir()
    prior_runner_sha = "5" * 64
    prior_activation_sha = "6" * 64
    prior_v1_summary_sha = "a" * 64
    fixture_prior_candidates = {
        cell_id: (format(index + 1, "x") * 64, format(index + 9, "x") * 64)
        for index, cell_id in enumerate(runner.EXPECTED_STAGE2_CELLS)
    }
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_V1_SUMMARY_SHA256", prior_v1_summary_sha)
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_CANDIDATES", fixture_prior_candidates)
    prior_rows = []
    for cell_id, (action, ready, delta) in runner.EXPECTED_STAGE2_CELLS.items():
        candidate_sha, contract_sha = fixture_prior_candidates[cell_id]
        prior_rows.append({
            "cell_id": cell_id, "action": action, "ready_source": ready,
            "delta": delta,
            "join_frame": queue["assets"][action]["contact_frame"] - 12,
            "blend_intervals": 10, "generator_rc": 0,
            "candidate_sha256": candidate_sha,
            "generator_contract_sha256": contract_sha,
            "frames": 40, "phase": 25.0 / 39.0,
            "joint_path_l2": 0.0, "joint_curvature_l2": 0.0,
            "max_joint_step_rad": 0.0, "topp_rc": 1,
        })
    prior_summary = {
        "schema_version": 1,
        "artifact_kind": "ready_to_strike_join_ladder_stage2_screening_result",
        "status": "stage2_terminal_failure_no_retry",
        "activation_sha256": prior_activation_sha,
        "queue_sha256": queue_sha,
        "stage1_receipt_sha256": _sha(receipt_payload),
        "runner_sha256": prior_runner_sha,
        "prior_failed_attempt_summary_sha256": prior_v1_summary_sha,
        "runtime_snapshot_shas": {
            relative: _sha(payload) for relative, payload in runtime_payloads.items()
        },
        "asset_snapshot_shas": {
            name: asset["sha256"] for name, asset in queue["assets"].items()
        },
        "rows": prior_rows,
        "screening_acceptance": {
            "at_or_below_0p5_cells": [], "timing_by_cell_s": {},
            "shared_ready_two_side_at_or_below_0p5": {
                "backhand": False, "forehand": False,
            },
            "any_shared_ready_pass": False,
        },
        "input_stability_errors": [],
        "formal_claims": {
            "physics_replay_exact": False, "source_closure_exact": False,
            "mjcf_closure_exact": False, "screening_evidence_only": True,
            "strict_global_minimum_proven": False,
        },
        "runtime_authority": {
            "cpu_only": True, "automatic_retry": False, "trainer_signal": False,
            "robot_command": False, "training_authorized": False,
            "deployment_authorized": False,
        },
        "automatic_retry": False, "reviewed_child_timeout_s": 3600,
        "trainer_or_robot_signals": [],
    }
    prior_summary_path = prior_root / "stage2_summary.json"
    prior_summary_payload = _canonical(prior_summary)
    _write(prior_summary_path, prior_summary_payload)
    prior_binding = {
        "namespace": str(prior_root),
        "summary_path": str(prior_summary_path),
        "summary_sha256": _sha(prior_summary_payload),
        "runner_sha256": prior_runner_sha,
        "activation_sha256": prior_activation_sha,
        "failure_class": "prior_v2_topp_rc1_no_timing",
        "automatic_retry": False,
    }
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_ATTEMPT", prior_binding)
    monkeypatch.setattr(runner, "EXPECTED_STAGE2_NAMESPACE", str(stage2))

    source = tmp_path / "runner.py"
    _write(source, b"runner-source\n")
    activation = {
        "schema_version": 1,
        "activation_id": runner.EXPECTED_ACTIVATION_ID,
        "created_utc": "2026-07-17T00:00:00Z",
        "parent_queue": {
            "path": "configs/ready_to_strike_join_ladder_20260717.yaml",
            "sha256": queue_sha,
            "main_commit": runner.EXPECTED_PREREG_COMMIT,
        },
        "stage1_namespace": str(stage1),
        "stage2_namespace": str(stage2),
        "observations": observations,
        "decision": {
            "ready_by_side_crossover": True,
            "forehand_prefers": "forehand",
            "backhand_prefers": "backhand",
            "delta17_strictly_worse_than_delta6_all_four_ready_by_side_pairs": True,
            "activate_both_ready_sources_at_midpoint": True,
            "shared_ready_not_yet_selected": True,
        },
        "evidence_status": runner.EXPECTED_EVIDENCE_STATUS,
        "required_attestation_receipt": str(receipt_path),
        "required_attestation_receipt_sha256": _sha(receipt_payload),
        "stage2_runner": {
            "path": "scripts/run_ready_to_strike_join_ladder_stage2.py",
            "sha256": _sha(source.read_bytes()),
        },
        "launch_authorized": True,
        "authorized_stage2_cells": [
            {"cell_id": cell_id, "action": values[0], "ready_source": values[1],
             "delta": values[2]}
            for cell_id, values in runner.EXPECTED_STAGE2_CELLS.items()
        ],
        "prior_failed_attempt": prior_binding,
        "topp_runtime": runner.EXPECTED_TOPP_RUNTIME,
        "runtime_authority": {
            "cpu_only": True,
            "automatic_retry": False,
            "trainer_signal": False,
            "robot_command": False,
            "training_authorized": False,
            "deployment_authorized": False,
        },
    }
    activation_path = tmp_path / "activation.json"
    _write(activation_path, _canonical(activation))

    def fake_mjcf_closure(**_kwargs):
        return ({
            "checkout_commit": queue["runtime"]["checkout_commit"],
            "model_root_git_tree_oid": "b" * 40,
            "mjcf_relative_path": str(runner.RUNTIME_RELATIVE_PATHS["mjcf_sha256"]),
            "compiler_meshdir": "meshes", "file_count": 2, "mesh_count": 1,
            "total_bytes": 2, "mesh_manifest_sha256": "c" * 64,
            "files": [], "git_blob_manifest_sha256": "d" * 64, "git_blobs": [],
        }, {})

    monkeypatch.setattr(runner, "_collect_mjcf_mesh_closure", fake_mjcf_closure)

    fake_runtime_receipt = {
        "interpreter": {
            "path": runner.EXPECTED_TOPP_RUNTIME["interpreter"]["path"],
            "observed_symlink_target": "python3.12",
            "canonical_realpath": runner.EXPECTED_TOPP_RUNTIME["interpreter"]["canonical_realpath"],
            "binary_sha256": runner.EXPECTED_TOPP_RUNTIME["interpreter"]["binary_sha256"],
            "python_version": runner.EXPECTED_TOPP_RUNTIME["interpreter"]["python_version"],
            "venv_prefix": runner.EXPECTED_TOPP_RUNTIME["interpreter"]["venv_prefix"],
            "symlink_identity": {"device": 1, "inode": 2, "size": 10,
                                 "mtime_ns": 3, "ctime_ns": 4},
            "binary_identity": {"device": 1, "inode": 3, "size": 20,
                                "mtime_ns": 5, "ctime_ns": 6},
        },
        "packages": {}, "probe_argv": ["fixture-runtime-probe"],
        "probe_rc": 0, "probe_stdout_sha256": "e" * 64,
        "pythonpath_removed": True, "pythonhome_removed": True,
    }
    monkeypatch.setattr(
        runner, "_inspect_topp_runtime",
        lambda _contract: (fake_runtime_receipt, {}),
    )
    monkeypatch.setattr(
        runner, "_preflight_mjcf_runtime",
        lambda **_kwargs: {
            "loader": "mujoco.MjModel.from_xml_path",
            "argv": [runner.EXPECTED_TOPP_RUNTIME["interpreter"]["path"], "fixture"],
            "returncode": 0,
            "dimensions": {"nq": 38, "nv": 37, "nbody": 33, "ngeom": 79, "nmesh": 74},
            "stdout_sha256": "f" * 64, "stderr_sha256": _sha(b""),
            "snapshot_tree_before_sha256": "1" * 64,
            "snapshot_tree_after_sha256": "1" * 64,
            "output_files_created": [],
        },
    )

    def fake_prior_inputs(**_kwargs):
        body_names_path = runtime / runner.RUNTIME_RELATIVE_PATHS["body_order_sha256"]
        body_names = tuple(line.strip() for line in body_names_path.read_text().splitlines()
                           if line.strip())
        fixture_root = tmp_path / "prior-v2-inputs"
        fixture_root.mkdir(exist_ok=True)
        records = {}
        snapshots = {}
        for cell_id in runner.EXPECTED_STAGE2_CELLS:
            q = np.zeros((40, 31), dtype=np.float32)
            quat = np.zeros((40, len(body_names), 4), dtype=np.float32)
            quat[..., 0] = 1.0
            candidate_path = fixture_root / f"{cell_id}.npz"
            np.savez(
                candidate_path,
                fps=np.array([50], dtype=np.int64), joint_pos=q,
                joint_vel=np.gradient(q, 1.0 / 50.0, axis=0).astype(np.float32),
                body_pos_w=np.zeros((40, len(body_names), 3), dtype=np.float32),
                body_quat_w=quat,
                body_lin_vel_w=np.zeros((40, len(body_names), 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((40, len(body_names), 3), dtype=np.float32),
                kinematics_schema_version=np.array([2], dtype=np.int64),
                body_pos_point=np.array("link_origin"),
                body_lin_vel_point=np.array("center_of_mass"),
                body_names=np.asarray(body_names),
            )
            contract_path = fixture_root / f"{cell_id}.contract.json"
            contract_path.write_bytes(b"fixture-contract\n")
            candidate = runner._read_snapshot(candidate_path, "fixture prior candidate")
            contract = runner._read_snapshot(contract_path, "fixture prior contract")
            info = {
                "candidate_sha256": candidate.sha256,
                "generator_contract_sha256": contract.sha256,
                "frames": 40, "phase": 25.0 / 39.0,
                "joint_path_l2": 0.0, "joint_curvature_l2": 0.0,
                "max_joint_step_rad": 0.0,
            }
            records[cell_id] = {
                "candidate": candidate, "contract": contract, "info": info,
            }
            snapshots[f"prior:candidate:{cell_id}"] = candidate
            snapshots[f"prior:contract:{cell_id}"] = contract
        for name in ("forehand", "backhand"):
            asset = runner._read_snapshot(
                Path(queue["assets"][name]["path"]), f"fixture prior {name} asset"
            )
            snapshots[f"prior:asset:{name}"] = asset
        return records, snapshots

    monkeypatch.setattr(runner, "_collect_prior_v2_inputs", fake_prior_inputs)
    return {
        "queue": queue_path,
        "activation": activation_path,
        "receipt": receipt_path,
        "root": stage2,
        "source": source,
    }


def _plan(paths: dict[str, Path]) -> dict[str, object]:
    return runner.plan_stage2(
        activation_path=paths["activation"],
        queue_path=paths["queue"],
        root=paths["root"],
        runner_source=paths["source"],
    )


def _prepare_real_prior_v2_preflight_prefix(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], dict[str, str], dict[str, str], Path]:
    activation = json.loads(paths["activation"].read_text())
    queue = json.loads(paths["queue"].read_text())
    prior_root = Path(activation["prior_failed_attempt"]["namespace"])
    controls = {
        "runner": b"prior-v2-runner\n", "activation": b"prior-v2-activation\n",
        "queue": paths["queue"].read_bytes(), "receipt": b"prior-v2-receipt\n",
        "generator": (Path(activation["stage1_namespace"])
                      / "build_ready_to_strike_motion.py").read_bytes(),
        "v1_summary": b"prior-v1-summary\n",
    }
    control_paths = {
        "runner": prior_root / "snapshots/run_ready_to_strike_join_ladder_stage2.py",
        "activation": prior_root / "snapshots/activation.json",
        "queue": prior_root / "snapshots/queue.json",
        "receipt": prior_root / "snapshots/stage1_historical_attestation.json",
        "generator": prior_root / "snapshots/build_ready_to_strike_motion.py",
        "v1_summary": prior_root / "snapshots/prior_stage2_failure_summary.json",
    }
    for name, payload in controls.items():
        _write(control_paths[name], payload)
    for name in ("forehand", "backhand"):
        _write(prior_root / f"snapshots/assets/{name}.npz",
               Path(queue["assets"][name]["path"]).read_bytes())
    binding = dict(activation["prior_failed_attempt"])
    binding["runner_sha256"] = _sha(controls["runner"])
    binding["activation_sha256"] = _sha(controls["activation"])
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_ATTEMPT", binding)
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_V1_SUMMARY_SHA256",
                        _sha(controls["v1_summary"]))
    candidate_shas: dict[str, str] = {}
    contract_shas: dict[str, str] = {}
    for cell_id in runner.EXPECTED_STAGE2_CELLS:
        cell_root = prior_root / cell_id
        candidate_payload = f"prior-v2-candidate-{cell_id}\n".encode()
        contract_payload = f"prior-v2-contract-{cell_id}\n".encode()
        _write(cell_root / "candidate.npz", candidate_payload)
        _write(cell_root / "candidate.contract.json", contract_payload)
        _write(cell_root / "topp/run.log", f"diagnostic-{cell_id}\n".encode())
        candidate_shas[cell_id] = _sha(candidate_payload)
        contract_shas[cell_id] = _sha(contract_payload)
    return activation, queue, candidate_shas, contract_shas, prior_root


@pytest.mark.parametrize("tamper", ["candidate", "contract"])
def test_real_prior_v2_preflight_rejects_tampered_scientific_input_before_runtime(
    tamper: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    activation, queue, candidate_shas, contract_shas, _prior_root = (
        _prepare_real_prior_v2_preflight_prefix(paths, monkeypatch)
    )
    cell_id = next(iter(runner.EXPECTED_STAGE2_CELLS))
    candidates = {
        name: (candidate_shas[name], contract_shas[name])
        for name in runner.EXPECTED_STAGE2_CELLS
    }
    candidates[cell_id] = (
        "0" * 64 if tamper == "candidate" else candidate_shas[cell_id],
        "0" * 64 if tamper == "contract" else contract_shas[cell_id],
    )
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_CANDIDATES", candidates)
    receipt_sha = _sha(b"prior-v2-receipt\n")

    with pytest.raises(runner.Stage2Error, match="candidate or contract SHA"):
        REAL_COLLECT_PRIOR_V2_INPUTS(
            activation=activation, queue=queue, body_order=("body",),
        )


def test_real_prior_v2_collector_ignores_diagnostic_and_obsolete_control_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    activation, queue, candidate_shas, contract_shas, prior_root = (
        _prepare_real_prior_v2_preflight_prefix(paths, monkeypatch)
    )
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_CANDIDATES", {
        name: (candidate_shas[name], contract_shas[name])
        for name in runner.EXPECTED_STAGE2_CELLS
    })

    def accept_candidate(candidate, contract, **_kwargs):
        return {
            "candidate_sha256": candidate.sha256,
            "generator_contract_sha256": contract.sha256,
            "frames": 40, "phase": 25.0 / 39.0,
            "joint_path_l2": 0.0, "joint_curvature_l2": 0.0,
            "max_joint_step_rad": 0.0,
        }

    monkeypatch.setattr(runner, "_validate_candidate", accept_candidate)
    (prior_root / "snapshots/build_ready_to_strike_motion.py").unlink()
    (prior_root / "snapshots/prior_stage2_failure_summary.json").unlink()
    for index, cell_id in enumerate(runner.EXPECTED_STAGE2_CELLS):
        log = prior_root / cell_id / "topp/run.log"
        if index % 2:
            log.unlink()
        else:
            log.write_bytes(f"arbitrary changed diagnostic {index}\n".encode())

    records, snapshots = REAL_COLLECT_PRIOR_V2_INPUTS(
        activation=activation, queue=queue, body_order=("body",),
    )
    assert set(records) == set(runner.EXPECTED_STAGE2_CELLS)
    assert set(snapshots) == {
        *(f"prior:asset:{name}" for name in ("forehand", "backhand")),
        *(f"prior:candidate:{cell}" for cell in runner.EXPECTED_STAGE2_CELLS),
        *(f"prior:contract:{cell}" for cell in runner.EXPECTED_STAGE2_CELLS),
    }


def test_v2_activation_bytes_remain_immutable() -> None:
    payload = (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v2_20260717.json").read_bytes()
    assert _sha(payload) == "8742aadff796218f170fede3f6e386e54314e086740f4ad82b9242f52667ab10"


def test_v3_preflight_activation_bytes_remain_immutable() -> None:
    payload = (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v3_20260717.json").read_bytes()
    assert _sha(payload) == "5b42fa8dd43cd29e5fc858a2ef895a5e92bc547c52fedd2d9e3dfaf7e68ab488"


def test_v4_preflight_activation_bytes_remain_immutable() -> None:
    payload = (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v4_20260717.json").read_bytes()
    assert _sha(payload) == "818f23d97b07cc52b2d8677d9d1e9d7670ab13cbcb85589b23e369450d3ef969"


def test_v5_preflight_activation_bytes_remain_immutable() -> None:
    payload = (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v5_20260717.json").read_bytes()
    assert _sha(payload) == "4541044946f3359632369330e02037c824b87ee4b92da8f4133964711b81bdf2"


def test_v6_activation_bytes_remain_immutable() -> None:
    payload = (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v6_20260717.json").read_bytes()
    assert _sha(payload) == "e32135c05e676cfda7902cc0acc3cb30bbc8239cb126f222d96d454c3a3ce3da"


def test_v7_activation_bytes_remain_immutable() -> None:
    payload = (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v7_20260717.json").read_bytes()
    assert _sha(payload) == "87598e0a39fe06ed7827c4fcd1809b05481021ca80defc6b24ff7743d8cd99c2"


def test_v7_excludes_diagnostic_logs_from_scientific_input_contract() -> None:
    source = SCRIPT.read_bytes().lower()
    assert b"no such file or directory" not in source
    assert b"expected_prior_topp_logs" not in source
    collect_source = source[source.index(b"def _collect_prior_v2_inputs"):]
    collect_source = collect_source[:collect_source.index(b"def _validate_inputs")]
    assert b"topp/run.log" not in collect_source
    assert b"prior:topp_log" not in collect_source


def _synthetic_topp_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, Path], dict[str, object]]:
    venv = tmp_path / "hope_mjeval_venv"
    interpreter = venv / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    target = tmp_path / "usr" / "bin" / "python3.12"
    _write(target, b"\x7fELF-python-3.12.3\n")
    os.symlink(str(target), interpreter)
    site = venv / "lib" / "python3.12" / "site-packages"
    files: dict[str, Path] = {"target": target}
    packages: dict[str, object] = {}
    probe_packages: dict[str, object] = {}
    for name, version in (("numpy", "2.5.0"), ("mujoco", "3.10.0")):
        package_dir = site / name
        dist = site / f"{name}-{version}.dist-info"
        paths = {
            "module": package_dir / "__init__.py",
            "metadata": dist / "METADATA",
            "record": dist / "RECORD",
            "wheel": dist / "WHEEL",
            "native": package_dir / f"_{name}_native.so",
            "pyc": package_dir / "__pycache__" / "cached.pyc",
        }
        if name == "mujoco":
            paths["libmujoco"] = package_dir / "lib" / "libmujoco.so.3.10.0"
            paths["optional"] = (
                package_dir / "experimental" / "studio"
                / "native_viewer_cc.cpython-312-x86_64-linux-gnu.so"
            )
        for label, path in paths.items():
            if label == "record":
                continue
            payload = (b"\x7fELF" + f"-{name}-{version}-{label}\n".encode()
                       if label in {"native", "libmujoco"}
                       else f"{name}-{version}-{label}\n".encode())
            _write(path, payload)
            files[f"{name}:{label}"] = path
        record_rows = []
        recorded_labels = ["module", "metadata", "wheel", "native"]
        if name == "mujoco":
            recorded_labels.extend(["libmujoco", "optional"])
        for label in recorded_labels:
            payload = paths[label].read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            record_rows.append(
                f"{paths[label].relative_to(site).as_posix()},sha256={digest},{len(payload)}"
            )
        record_rows.extend([
            f"{paths['pyc'].relative_to(site).as_posix()},,",
            f"{paths['record'].relative_to(site).as_posix()},,",
        ])
        _write(paths["record"], ("\n".join(record_rows) + "\n").encode())
        files[f"{name}:record"] = paths["record"]
        record_snapshot = runner._read_snapshot(paths["record"], f"fixture {name} RECORD")
        record_closure, _native = runner._verify_distribution_record(
            package_name=name, version=version, site_packages=site,
            venv_root=venv, record_snapshot=record_snapshot,
        )
        packages[name] = {
            "version": version,
            **{f"{label}_sha256": _sha(paths[label].read_bytes())
               for label in ("module", "metadata", "record", "wheel")},
            "record_file_count": record_closure["file_count"],
            "record_total_bytes": record_closure["total_bytes"],
            "record_manifest_sha256": record_closure["verified_manifest_sha256"],
            "record_native_elf_count": record_closure["native_elf_count"],
            "record_unhashed_row_count": (
                record_closure["explicitly_bound_unhashed_row_count"]
            ),
        }
        probe_packages[name] = {
            "version": version,
            **{label: str(paths[label]) for label in ("module", "metadata", "record", "wheel")},
        }
    ldd = tmp_path / "usr" / "bin" / "ldd"
    _write(ldd, b"reviewed-ldd\n")
    readelf = tmp_path / "usr" / "bin" / "readelf"
    _write(readelf, b"reviewed-readelf\n")
    dynamic_contract = {
        "ldd_path": str(ldd), "ldd_sha256": _sha(ldd.read_bytes()),
        "readelf_path": str(readelf), "readelf_sha256": _sha(readelf.read_bytes()),
        "allowed_virtual_dependencies": ["linux-vdso.so.1"],
        "elf_input_count": 2, "resolved_file_count": 1, "edge_count": 2,
        "manifest_sha256": "d" * 64,
    }
    contract: dict[str, object] = {
        "interpreter": {
            "path": str(interpreter), "canonical_realpath": str(target),
            "binary_sha256": _sha(target.read_bytes()), "python_version": "3.12.3",
            "venv_prefix": str(venv),
        },
        "packages": packages,
        "dynamic_dependencies": dynamic_contract,
        "mjcf_model": {
            "loader": "mujoco.MjModel.from_xml_path",
            "nq": 38, "nv": 37, "nbody": 33, "ngeom": 79, "nmesh": 74,
        },
    }
    probe: dict[str, object] = {
        "python_version": "3.12.3", "executable": str(interpreter),
        "prefix": str(venv), "packages": probe_packages,
    }

    def fake_probe(command, *, cwd, env):
        assert command[:4] == [str(interpreter), "-I", "-B", "-c"]
        assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
        return subprocess.CompletedProcess(command, 0, json.dumps(probe), "")

    monkeypatch.setattr(runner, "EXPECTED_TOPP_RUNTIME", contract)
    monkeypatch.setattr(runner, "_run_runtime_command", fake_probe)
    monkeypatch.setattr(
        runner, "_collect_dynamic_dependency_closure",
        lambda **_kwargs: {
            "ldd": {"path": str(ldd), "sha256": _sha(ldd.read_bytes())},
            "readelf": {"path": str(readelf), "sha256": _sha(readelf.read_bytes())},
            "elf_input_count": 2, "resolved_file_count": 1, "edge_count": 2,
            "manifest_sha256": "d" * 64,
            "allowed_virtual_dependencies": ["linux-vdso.so.1"],
        },
    )
    return contract, files, probe


@pytest.mark.parametrize(
    "tamper", ["target", "numpy:module", "numpy:metadata", "numpy:record",
               "numpy:wheel", "numpy:native", "numpy:pyc",
               "mujoco:module", "mujoco:metadata", "mujoco:record",
               "mujoco:wheel", "mujoco:native", "mujoco:libmujoco",
               "mujoco:optional", "mujoco:pyc"],
)
def test_v7_runtime_rejects_real_file_tamper(
    tamper: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, files, _probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    files[tamper].write_bytes(files[tamper].read_bytes() + b"tamper")
    with pytest.raises(runner.Stage2Error, match="(?:SHA|size|bytes|manifest) changed"):
        REAL_INSPECT_TOPP_RUNTIME(contract)


def test_v8_runtime_accepts_relative_or_absolute_symlink_to_same_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _files, _probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    interpreter = Path(contract["interpreter"]["path"])
    target = Path(contract["interpreter"]["canonical_realpath"])
    REAL_INSPECT_TOPP_RUNTIME(contract)
    interpreter.unlink()
    os.symlink(os.path.relpath(target, interpreter.parent), interpreter)
    receipt, _snapshots = REAL_INSPECT_TOPP_RUNTIME(contract)
    assert receipt["interpreter"]["canonical_realpath"] == str(target)


def test_v8_runtime_rejects_different_binary_even_with_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _files, _probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    interpreter = Path(contract["interpreter"]["path"])
    target = Path(contract["interpreter"]["canonical_realpath"])
    wrong = tmp_path / "wrong-python"
    _write(wrong, target.read_bytes())
    interpreter.unlink()
    os.symlink(str(wrong), interpreter)
    with pytest.raises(runner.Stage2Error, match="canonical realpath changed"):
        REAL_INSPECT_TOPP_RUNTIME(contract)


@pytest.mark.parametrize("race", ["symlink", "binary"])
def test_v8_runtime_rejects_interpreter_inspection_race(
    race: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _files, probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    interpreter = Path(contract["interpreter"]["path"])
    target = Path(contract["interpreter"]["canonical_realpath"])

    def racing_probe(command, *, cwd, env):
        if race == "symlink":
            interpreter.unlink()
            os.symlink(os.path.relpath(target, interpreter.parent), interpreter)
        else:
            target.write_bytes(target.read_bytes() + b"race")
        return subprocess.CompletedProcess(command, 0, json.dumps(probe), "")

    monkeypatch.setattr(runner, "_run_runtime_command", racing_probe)
    with pytest.raises(runner.Stage2Error, match="changed while inspecting"):
        REAL_INSPECT_TOPP_RUNTIME(contract)


def test_v7_runtime_rejects_wrong_package_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _files, probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    probe["packages"]["numpy"]["version"] = "2.5.1"
    with pytest.raises(runner.Stage2Error, match="numpy version changed"):
        REAL_INSPECT_TOPP_RUNTIME(contract)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("absolute", "absolute or empty"),
        ("escape", "escapes the fixed venv"),
        ("duplicate", "duplicate path"),
        ("canonical_duplicate", "duplicate canonical destination"),
        ("noncanonical_path", "noncanonical path"),
        ("partial_hash", "partial hash or size"),
        ("missing", "component is missing"),
        ("symlink", "contains symlink"),
    ],
)
def test_v7_record_parser_rejects_unsafe_or_incomplete_closure(
    mutation: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, files, _probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    venv = Path(contract["interpreter"]["path"]).parent.parent
    site = venv / "lib" / "python3.12" / "site-packages"
    record_path = files["numpy:record"]
    rows = record_path.read_text().splitlines()
    if mutation == "absolute":
        rows[0] = "/escape," + rows[0].split(",", 1)[1]
    elif mutation == "escape":
        rows[0] = "../../../../../../escape," + rows[0].split(",", 1)[1]
    elif mutation == "duplicate":
        rows.insert(1, rows[0])
    elif mutation == "canonical_duplicate":
        rows.insert(1, "numpy/sub/../__init__.py," + rows[0].split(",", 1)[1])
    elif mutation == "noncanonical_path":
        rows[0] = "numpy//__init__.py," + rows[0].split(",", 1)[1]
    elif mutation == "partial_hash":
        fields = rows[0].split(",")
        rows[0] = f"{fields[0]},,{fields[2]}"
    elif mutation == "missing":
        files["numpy:module"].unlink()
    elif mutation == "symlink":
        module = files["numpy:module"]
        payload = module.read_bytes()
        module.unlink()
        replacement = tmp_path / "replacement.py"
        _write(replacement, payload)
        os.symlink(str(replacement), module)
    if mutation not in {"missing", "symlink"}:
        record_path.write_text("\n".join(rows) + "\n")
    record = runner._read_snapshot(record_path, "mutated RECORD")
    with pytest.raises(runner.Stage2Error, match=message):
        runner._verify_distribution_record(
            package_name="numpy", version="2.5.0", site_packages=site,
            venv_root=venv, record_snapshot=record,
        )


def test_v7_record_parser_allows_bound_console_script_inside_fixed_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, files, _probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    venv = Path(contract["interpreter"]["path"]).parent.parent
    site = venv / "lib" / "python3.12" / "site-packages"
    script = venv / "bin" / "f2py"
    _write(script, b"#!/bin/sh\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(script.read_bytes()).digest()).rstrip(b"=").decode()
    record_path = files["numpy:record"]
    rows = record_path.read_text().splitlines()
    rows.insert(-1, f"../../../bin/f2py,sha256={digest},{len(script.read_bytes())}")
    record_path.write_text("\n".join(rows) + "\n")
    record = runner._read_snapshot(record_path, "console-script RECORD")
    receipt, _elfs = runner._verify_distribution_record(
        package_name="numpy", version="2.5.0", site_packages=site,
        venv_root=venv, record_snapshot=record,
    )
    assert receipt["file_count"] == 7


def _synthetic_dynamic_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, Path], dict[str, runner.Snapshot]]:
    ldd = tmp_path / "usr" / "bin" / "ldd"
    _write(ldd, b"reviewed ldd\n")
    readelf = tmp_path / "usr" / "bin" / "readelf"
    _write(readelf, b"reviewed readelf\n")
    sources = {}
    source_paths = []
    for name in ("python3.12", "native.so"):
        path = tmp_path / "elf" / name
        _write(path, b"\x7fELF-" + name.encode())
        snapshot = runner._read_snapshot(path, f"fixture ELF {name}")
        sources[str(path)] = snapshot
        source_paths.append(path)
    dependencies = []
    for name in ("libfixture.so", "ld-linux-x86-64.so.2"):
        path = tmp_path / "lib" / name
        _write(path, b"\x7fELF-lib-" + name.encode())
        dependencies.append(path)

    def fake_ldd(command, *, cwd, env):
        assert command[0] == str(ldd)
        assert not ({"LD_AUDIT", "LD_DEBUG", "LD_LIBRARY_PATH", "LD_PRELOAD"} & set(env))
        output = (
            "linux-vdso.so.1 (0x00007fff)\n"
            f"libfixture.so => {dependencies[0]} (0x00007fff)\n"
            f"{dependencies[1]} (0x00007fff)\n"
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(runner, "_run_ldd_command", fake_ldd)
    edges = [
        {"soname": "ld-linux-x86-64.so.2", "resolved_path": str(dependencies[1]),
         "resolution_kind": "ldd_absolute"},
        {"soname": "libfixture.so", "resolved_path": str(dependencies[0]),
         "resolution_kind": "ldd_absolute"},
    ]
    source_rows = [{
        "path": str(path), "bytes": len(sources[str(path)].payload),
        "sha256": sources[str(path)].sha256, "dependencies": edges,
        "virtual_dependencies": ["linux-vdso.so.1"],
        "linkage_kind": "dynamic", "static_verification": None,
    } for path in sorted(source_paths)]
    resolved_rows = sorted(({
        "path": str(path), "bytes": len(path.read_bytes()),
        "sha256": _sha(path.read_bytes()),
    } for path in dependencies), key=lambda row: row["path"])
    manifest = {
        "ldd": {"path": str(ldd), "sha256": _sha(ldd.read_bytes())},
        "readelf": {"path": str(readelf), "sha256": _sha(readelf.read_bytes())},
        "sources": source_rows, "resolved_files": resolved_rows,
    }
    contract = {
        "ldd_path": str(ldd), "ldd_sha256": _sha(ldd.read_bytes()),
        "readelf_path": str(readelf), "readelf_sha256": _sha(readelf.read_bytes()),
        "allowed_virtual_dependencies": ["linux-vdso.so.1"],
        "elf_input_count": 2, "resolved_file_count": 2, "edge_count": 4,
        "manifest_sha256": _sha(_canonical(manifest)),
    }
    return contract, {
        "ldd": ldd, "readelf": readelf, "dependency": dependencies[0],
    }, sources


def test_v7_dynamic_dependency_closure_is_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _files, sources = _synthetic_dynamic_closure(tmp_path, monkeypatch)
    receipt = REAL_COLLECT_DYNAMIC_DEPENDENCY_CLOSURE(
        elf_inputs=sources, contract=contract)
    assert receipt["manifest_sha256"] == contract["manifest_sha256"]
    assert receipt["elf_input_count"] == 2
    assert receipt["resolved_file_count"] == 2
    assert receipt["edge_count"] == 4


def test_v7_readelf_symlink_alias_is_rejected_but_canonical_target_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, files, sources = _synthetic_dynamic_closure(tmp_path, monkeypatch)
    receipt = REAL_COLLECT_DYNAMIC_DEPENDENCY_CLOSURE(
        elf_inputs=sources, contract=contract)
    assert receipt["readelf"]["path"] == str(files["readelf"])
    alias = files["readelf"].with_name("readelf-alias")
    os.symlink(str(files["readelf"]), alias)
    with pytest.raises(runner.Stage2Error, match="reviewed readelf tool contains symlink"):
        runner._observe_dynamic_dependency_closure(
            elf_inputs=sources, ldd_path=contract["ldd_path"],
            readelf_path=alias,
            allowed_virtual_dependencies=["linux-vdso.so.1"],
        )


@pytest.mark.parametrize("tamper", ["ldd", "readelf", "dependency"])
def test_v7_dynamic_dependency_tamper_fails_closed(
    tamper: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, files, sources = _synthetic_dynamic_closure(tmp_path, monkeypatch)
    files[tamper].write_bytes(files[tamper].read_bytes() + b"tamper")
    with pytest.raises(
        runner.Stage2Error,
        match="(?:(?:ldd|readelf) tool SHA|manifest) changed",
    ):
        REAL_COLLECT_DYNAMIC_DEPENDENCY_CLOSURE(
            elf_inputs=sources, contract=contract)


def test_v7_loaded_elf_tamper_fails_before_ldd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _files, sources = _synthetic_dynamic_closure(tmp_path, monkeypatch)
    source = Path(next(iter(sources)))
    payload = source.read_bytes()
    source.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(runner.Stage2Error, match="loaded ELF changed before ldd"):
        REAL_COLLECT_DYNAMIC_DEPENDENCY_CLOSURE(
            elf_inputs=sources, contract=contract)


def test_v7_loaded_elf_missing_dependency_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _files, sources = _synthetic_dynamic_closure(tmp_path, monkeypatch)

    def missing_dependency(command, *, cwd, env):
        return subprocess.CompletedProcess(
            command, 0,
            "linux-vdso.so.1 (0x0)\nlibmissing.so => not found\n", "",
        )

    monkeypatch.setattr(runner, "_run_ldd_command", missing_dependency)
    with pytest.raises(
        runner.Stage2Error,
        match="libmissing.so.*does not uniquely match an actual loaded ELF",
    ):
        runner._observe_dynamic_dependency_closure(
            elf_inputs=sources,
            ldd_path=contract["ldd_path"],
            readelf_path=contract["readelf_path"],
            allowed_virtual_dependencies=["linux-vdso.so.1"],
        )


def test_v7_unresolved_soname_resolves_only_to_unique_actual_loaded_elf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ldd = tmp_path / "usr" / "bin" / "ldd"
    readelf = tmp_path / "usr" / "bin" / "readelf"
    plugin = tmp_path / "venv" / "mujoco" / "plugin" / "libactuator.so"
    libmujoco = tmp_path / "venv" / "mujoco" / "lib" / "libmujoco.so.3.10.0"
    dependency = tmp_path / "lib" / "libc.so.6"
    for path, payload in (
        (ldd, b"reviewed ldd"), (readelf, b"reviewed readelf"),
        (plugin, b"\x7fELF-plugin"), (libmujoco, b"\x7fELF-libmujoco"),
        (dependency, b"\x7fELF-libc"),
    ):
        _write(path, payload)
    sources = {
        str(path): runner._read_snapshot(path, f"loaded {path.name}")
        for path in (plugin, libmujoco)
    }

    def fake_ldd(command, *, cwd, env):
        if command[1] == str(plugin):
            stdout = "libmujoco.so.3.10.0 => not found\n"
        else:
            stdout = f"libc.so.6 => {dependency} (0x0)\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(runner, "_run_ldd_command", fake_ldd)
    receipt, manifest = runner._observe_dynamic_dependency_closure(
        elf_inputs=sources, ldd_path=ldd, readelf_path=readelf,
        allowed_virtual_dependencies=["linux-vdso.so.1"],
    )
    assert receipt["edge_count"] == 2
    plugin_row = next(row for row in manifest["sources"] if row["path"] == str(plugin))
    assert plugin_row["dependencies"] == [{
        "soname": "libmujoco.so.3.10.0",
        "resolved_path": str(libmujoco),
        "resolution_kind": "actual_loaded_unique_soname",
    }]


def test_v7_unresolved_soname_rejects_multiple_loaded_basename_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ldd = tmp_path / "usr" / "bin" / "ldd"
    readelf = tmp_path / "usr" / "bin" / "readelf"
    plugin = tmp_path / "plugin.so"
    duplicate_a = tmp_path / "a" / "libduplicate.so"
    duplicate_b = tmp_path / "b" / "libduplicate.so"
    dependency = tmp_path / "libc.so.6"
    for path, payload in (
        (ldd, b"reviewed ldd"), (readelf, b"reviewed readelf"),
        (plugin, b"\x7fELF-plugin"), (duplicate_a, b"\x7fELF-a"),
        (duplicate_b, b"\x7fELF-b"), (dependency, b"\x7fELF-libc"),
    ):
        _write(path, payload)
    sources = {
        str(path): runner._read_snapshot(path, f"loaded {path}")
        for path in (plugin, duplicate_a, duplicate_b)
    }

    def fake_ldd(command, *, cwd, env):
        stdout = ("libduplicate.so => not found\n" if command[1] == str(plugin)
                  else f"libc.so.6 => {dependency} (0x0)\n")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(runner, "_run_ldd_command", fake_ldd)
    with pytest.raises(
        runner.Stage2Error,
        match="libduplicate.so.*does not uniquely match an actual loaded ELF",
    ):
        runner._observe_dynamic_dependency_closure(
            elf_inputs=sources, ldd_path=ldd, readelf_path=readelf,
            allowed_virtual_dependencies=["linux-vdso.so.1"],
        )


def test_v7_exact_static_ldd_is_verified_by_readelf_and_manifested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "elf" / "static-extension.so"
    ldd = tmp_path / "usr" / "bin" / "ldd"
    readelf = tmp_path / "usr" / "bin" / "readelf"
    _write(source, b"\x7fELF-static-extension")
    _write(ldd, b"reviewed ldd")
    _write(readelf, b"reviewed readelf")
    snapshot = runner._read_snapshot(source, "static fixture")

    monkeypatch.setattr(
        runner, "_run_ldd_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "\tstatically linked\n", ""),
    )
    readelf_stdout = (
        "\nDynamic section at offset 0x10 contains 2 entries:\n"
        "  Tag        Type                         Name/Value\n"
        " 0x000000000000000e (SONAME)             Library soname: [fixture.so]\n"
        " 0x0000000000000000 (NULL)               0x0\n"
    )
    monkeypatch.setattr(
        runner, "_run_readelf_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, readelf_stdout, ""),
    )
    receipt, manifest = runner._observe_dynamic_dependency_closure(
        elf_inputs={str(source): snapshot}, ldd_path=ldd,
        readelf_path=readelf,
        allowed_virtual_dependencies=["linux-vdso.so.1"],
    )
    assert receipt["elf_input_count"] == 1
    assert receipt["edge_count"] == 0
    assert manifest["sources"][0]["linkage_kind"] == "static_no_dependencies"
    assert manifest["sources"][0]["dependencies"] == []
    assert manifest["sources"][0]["static_verification"]["needed_count"] == 0


def test_v7_static_ldd_claim_with_needed_entry_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "elf" / "fake-static.so"
    ldd = tmp_path / "usr" / "bin" / "ldd"
    readelf = tmp_path / "usr" / "bin" / "readelf"
    _write(source, b"\x7fELF-fake-static")
    _write(ldd, b"reviewed ldd")
    _write(readelf, b"reviewed readelf")
    snapshot = runner._read_snapshot(source, "fake-static fixture")
    monkeypatch.setattr(
        runner, "_run_ldd_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "\tstatically linked\n", ""),
    )
    readelf_stdout = (
        "Dynamic section at offset 0x10 contains 2 entries:\n"
        "  Tag        Type                         Name/Value\n"
        " 0x0000000000000001 (NEEDED)             Shared library: [libbad.so]\n"
        " 0x0000000000000000 (NULL)               0x0\n"
    )
    monkeypatch.setattr(
        runner, "_run_readelf_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, readelf_stdout, ""),
    )
    with pytest.raises(runner.Stage2Error, match="ldd claimed static but readelf found NEEDED"):
        runner._observe_dynamic_dependency_closure(
            elf_inputs={str(source): snapshot}, ldd_path=ldd,
            readelf_path=readelf,
            allowed_virtual_dependencies=["linux-vdso.so.1"],
        )


def _synthetic_mjcf_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot" / "runtime"
    mjcf = root / runner.RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    _write(mjcf, b"<mujoco/>\n")
    _write(mjcf.parent / "meshes" / "one.stl", b"mesh\n")
    return root


def test_v7_unloaded_optional_record_elf_is_verified_but_not_sent_to_ldd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, files, _probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    root = _synthetic_mjcf_snapshot(tmp_path)
    loaded_paths = sorted({
        str(files["target"].resolve()),
        str(files["numpy:native"].resolve()),
        str(files["mujoco:native"].resolve()),
        str(files["mujoco:libmujoco"].resolve()),
    })
    optional = str(files["mujoco:optional"].resolve())

    def fake_preflight(command, *, cwd, env):
        result = {
            "nq": 38, "nv": 37, "nbody": 33, "ngeom": 79, "nmesh": 74,
            "loaded_elf_paths": loaded_paths,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    monkeypatch.setattr(runner, "_run_runtime_command", fake_preflight)
    receipt, loaded = REAL_OBSERVE_MJCF_RUNTIME_PREFLIGHT(
        runtime_snapshot_root=root, contract=contract)
    assert receipt["loaded_elf_count"] == 4
    assert optional not in loaded

    dependency = tmp_path / "lib" / "libfixture.so"
    _write(dependency, b"\x7fELF-fixture-dependency")
    ldd_sources: list[str] = []

    def fake_ldd(command, *, cwd, env):
        ldd_sources.append(command[1])
        output = (
            "linux-vdso.so.1 (0x0)\n"
            f"libfixture.so => {dependency} (0x0)\n"
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(runner, "_run_ldd_command", fake_ldd)
    dynamic, _manifest = runner._observe_dynamic_dependency_closure(
        elf_inputs=loaded,
        ldd_path=contract["dynamic_dependencies"]["ldd_path"],
        readelf_path=contract["dynamic_dependencies"]["readelf_path"],
        allowed_virtual_dependencies=["linux-vdso.so.1"],
    )
    assert dynamic["elf_input_count"] == 4
    assert ldd_sources == loaded_paths
    assert optional not in ldd_sources


@pytest.mark.parametrize("failure", ["dimensions", "returncode", "output_file"])
def test_v7_mjcf_preflight_fails_closed(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, files, _probe = _synthetic_topp_runtime(tmp_path, monkeypatch)
    root = _synthetic_mjcf_snapshot(tmp_path)
    loaded_paths = sorted({
        str(files["target"].resolve()),
        str(files["numpy:native"].resolve()),
        str(files["mujoco:native"].resolve()),
        str(files["mujoco:libmujoco"].resolve()),
    })

    def fake_preflight(command, *, cwd, env):
        assert command[0] == contract["interpreter"]["path"]
        assert command[1:4] == ["-I", "-B", "-c"]
        assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
        dimensions = {"nq": 38, "nv": 37, "nbody": 33, "ngeom": 79, "nmesh": 74}
        dimensions["loaded_elf_paths"] = loaded_paths
        if failure == "dimensions":
            dimensions["nq"] = 39
        if failure == "output_file":
            (root / "unreviewed.out").write_bytes(b"side effect\n")
        return subprocess.CompletedProcess(
            command, 9 if failure == "returncode" else 0,
            json.dumps(dimensions), "preflight failed" if failure == "returncode" else "",
        )

    monkeypatch.setattr(runner, "_run_runtime_command", fake_preflight)
    match = {
        "dimensions": "dimensions changed",
        "returncode": "preflight failed rc=9",
        "output_file": "created or changed snapshot files",
    }[failure]
    with pytest.raises(runner.Stage2Error, match=match):
        REAL_PREFLIGHT_MJCF_RUNTIME(runtime_snapshot_root=root, contract=contract)


def test_v4_binds_the_observed_v2_contract_lineage_not_v1_contracts() -> None:
    assert runner.EXPECTED_PRIOR_CANDIDATES == {
        "fh_rf_d12": (
            "a6c181f1b29b7e683a2efa70414f908c0896d110b21721c39565e3641a4eeb17",
            "7c8e1f3a5184829d66e48f33e2ed93dbe93c044b2b4feea1dd921f2dddd9fb1a",
        ),
        "fh_rb_d12": (
            "ac3089ed72492eb92a4bdb63c218070af9303fa7fb4ec6df909f7e406ea13c6a",
            "9970770e897b9464f258888e645bd45f6de8cebdfc640816e194ad713a20a535",
        ),
        "bh_rf_d12": (
            "c892336ee0363e0867535be9fc892a071c49ae3af338412bcc090f06d66c6c64",
            "f7686ef8dad9709eecf9009d276b90b2a2d04ae72836c93585aa95d4ad2afbfb",
        ),
        "bh_rb_d12": (
            "d9ce654c861d343be8fd6ed81ac40a15fda9b95d6bf2969bacdb936697e68643",
            "e504637a42bf1c26d6100d5a682974a5e950c0a18aeeb10c120754a87cce1790",
        ),
    }


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_schema2_gradient_contract_separates_generator_from_topp_workspace() -> None:
    rng = np.random.default_rng(20260717)
    q = (rng.normal(size=(40, 31)).astype(np.float32) * np.float32(1.0e-3))
    qd32 = np.gradient(q, 1.0 / 50.0, axis=0).astype(np.float32)
    qd64 = np.gradient(q.astype(np.float64), 1.0 / 50.0, axis=0).astype(np.float32)
    assert not np.array_equal(qd32, qd64)
    quat = np.zeros((40, 1, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    arrays = {
        "fps": np.array([50], dtype=np.int64),
        "joint_pos": q,
        "joint_vel": qd32,
        "body_pos_w": np.zeros((40, 1, 3), dtype=np.float32),
        "body_quat_w": quat,
        "body_lin_vel_w": np.zeros((40, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((40, 1, 3), dtype=np.float32),
        "kinematics_schema_version": np.array([2], dtype=np.int64),
        "body_pos_point": np.array("link_origin"),
        "body_lin_vel_point": np.array("center_of_mass"),
        "body_names": np.array(["body"]),
    }

    assert runner._validate_schema2(
        arrays, label="generator candidate", body_order=("body",),
        allow_migration=False, gradient_contract="float32_producer",
    ) == 40
    with pytest.raises(runner.Stage2Error, match="canonical position gradient"):
        runner._validate_schema2(
            arrays, label="TOPP output", body_order=("body",),
            allow_migration=False, gradient_contract="float64_workspace",
        )
    arrays["joint_vel"] = qd64
    assert runner._validate_schema2(
        arrays, label="TOPP output", body_order=("body",),
        allow_migration=False, gradient_contract="float64_workspace",
    ) == 40


def test_real_mjcf_mesh_closure_is_git_bound_and_complete() -> None:
    mjcf_relative = runner.RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    snapshot = runner._read_snapshot(REPO / mjcf_relative, "real MJCF fixture")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()

    closure, snapshots = runner._collect_mjcf_mesh_closure(
        runtime_root=REPO, checkout_commit=head,
        mjcf_relative=mjcf_relative, mjcf_snapshot=snapshot,
    )

    assert closure["file_count"] == 75
    assert closure["mesh_count"] == 74
    assert closure["total_bytes"] == 14127373
    assert closure["mesh_manifest_sha256"] == runner.EXPECTED_MJCF_CLOSURE_MANIFEST_SHA256
    assert closure["model_root_git_tree_oid"] == runner.EXPECTED_MJCF_MODEL_TREE_OID
    assert len(snapshots) == 74


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.replace(
            b'file="pelvis_link.STL"', b'file="../pelvis_link.STL"', 1), "traversal"),
        (lambda payload: payload.replace(
            b'file="waist_yaw_Link.STL"', b'file="pelvis_link.STL"', 1), "duplicate"),
        (lambda payload: payload.replace(
            b"<asset>", b'<include file="external.xml" />\n  <asset>', 1),
         "unsupported external"),
        (lambda payload: b"<!DOCTYPE mujoco [<!ENTITY x 'y'>]>\n" + payload,
         "entity declarations"),
    ],
)
def test_mjcf_closure_rejects_unreviewed_external_or_ambiguous_references(
    mutator, message: str
) -> None:
    mjcf_relative = runner.RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    original = runner._read_snapshot(REPO / mjcf_relative, "real MJCF fixture")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    changed = runner.Snapshot(original.path, mutator(original.payload), original.mode)

    with pytest.raises(runner.Stage2Error, match=message):
        runner._collect_mjcf_mesh_closure(
            runtime_root=REPO, checkout_commit=head,
            mjcf_relative=mjcf_relative, mjcf_snapshot=changed,
        )


def test_mjcf_closure_rejects_wrong_frozen_tree_before_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mjcf_relative = runner.RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    snapshot = runner._read_snapshot(REPO / mjcf_relative, "real MJCF fixture")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()

    def wrong_tree(runtime_root, arguments, label):
        if label == "MJCF model-root tree binding":
            return ("0" * 40 + "\n").encode()
        return REAL_RUN_GIT_READONLY(runtime_root, arguments, label)

    monkeypatch.setattr(runner, "_run_git_readonly", wrong_tree)
    with pytest.raises(runner.Stage2Error, match="model-root Git tree"):
        runner._collect_mjcf_mesh_closure(
            runtime_root=REPO, checkout_commit=head,
            mjcf_relative=mjcf_relative, mjcf_snapshot=snapshot,
        )


@pytest.mark.parametrize("mutation", ["blob", "mode"])
def test_mjcf_closure_rejects_wrong_git_blob_or_mode(
    mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mjcf_relative = runner.RUNTIME_RELATIVE_PATHS["mjcf_sha256"]
    snapshot = runner._read_snapshot(REPO / mjcf_relative, "real MJCF fixture")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()

    def corrupt_git(runtime_root, arguments, label):
        payload = REAL_RUN_GIT_READONLY(runtime_root, arguments, label)
        if mutation == "blob" and arguments[:2] == ["cat-file", "blob"]:
            return b"corrupt-object-bytes"
        if mutation == "mode" and label == "MJCF mesh tree binding":
            return payload.replace(b"100644 blob", b"100755 blob", 1)
        return payload

    monkeypatch.setattr(runner, "_run_git_readonly", corrupt_git)
    message = "corrupt MJCF blob" if mutation == "blob" else "unsupported Git entry"
    with pytest.raises(runner.Stage2Error, match=message):
        runner._collect_mjcf_mesh_closure(
            runtime_root=REPO, checkout_commit=head,
            mjcf_relative=mjcf_relative, mjcf_snapshot=snapshot,
        )


def _fake_executor(paths: dict[str, Path], calls: list[list[str]], *,
                   fail_generator_cell: str | None = None,
                   fail_topp_cell: str | None = None,
                   budget_scale: float = 1.5,
                   reported_runup_s: float = 0.6,
                   output_contact_frame: int | None = None):
    queue = json.loads(paths["queue"].read_text())
    body_order_path = Path(queue["runtime"]["checkout_path"]) / runner.RUNTIME_RELATIVE_PATHS["body_order_sha256"]
    body_names = tuple(line.strip() for line in body_order_path.read_text().splitlines() if line.strip())

    def arrays(frames: int = 40) -> dict[str, np.ndarray]:
        q = np.zeros((frames, 31), dtype=np.float32)
        quat = np.zeros((frames, len(body_names), 4), dtype=np.float32)
        quat[..., 0] = 1.0
        return {
            "fps": np.array([50], dtype=np.int64),
            "joint_pos": q,
            "joint_vel": np.gradient(q.astype(np.float64), 1.0 / 50.0, axis=0).astype(np.float32),
            "body_pos_w": np.zeros((frames, len(body_names), 3), dtype=np.float32),
            "body_quat_w": quat,
            "body_lin_vel_w": np.zeros((frames, len(body_names), 3), dtype=np.float32),
            "body_ang_vel_w": np.zeros((frames, len(body_names), 3), dtype=np.float32),
            "kinematics_schema_version": np.array([2], dtype=np.int64),
            "body_pos_point": np.array("link_origin"),
            "body_lin_vel_point": np.array("center_of_mass"),
            "body_names": np.asarray(body_names),
        }

    def execute(command, *, cwd, env):
        command = list(command)
        calls.append(command)
        if "--output-contract" in command:
            output = Path(_flag(command, "--output-npz"))
            contract_path = Path(_flag(command, "--output-contract"))
            cell_id = output.parent.name
            if cell_id == fail_generator_cell:
                return subprocess.CompletedProcess(command, 7, "registered generator failure")
            np.savez(output, **arrays())
            output_payload = output.read_bytes()
            source_path = _flag(command, "--source")
            ready_path = _flag(command, "--ready-source")
            action = Path(source_path).stem
            ready = Path(ready_path).stem
            assert action in {"forehand", "backhand"}
            assert ready in {"forehand", "backhand"}
            contract = {
                "schema_version": 1,
                "artifact_kind": "host_only_ready_to_strike_motion_candidate",
                "status": "candidate_only_all_runtime_and_safety_gates_open",
                "inputs": {
                    "source_schema2_npz": {
                        "path": source_path, "bytes": 1,
                        "sha256": queue["assets"][action]["sha256"],
                        "device": 1, "inode": 1, "mtime_ns": 1, "ctime_ns": 1,
                    },
                    "shared_ready_schema2_npz": {
                        "path": ready_path, "bytes": 1,
                        "sha256": queue["assets"][ready]["sha256"],
                        "device": 1, "inode": 1, "mtime_ns": 1, "ctime_ns": 1,
                    },
                    "shared_ready_frame": 0,
                },
                "tool": {"path": command[2], "sha256": queue["runtime"]["generator_sha256"]},
                "request": {
                    "source_contact_frame": int(_flag(command, "--contact-frame")),
                    "source_join_frame": int(_flag(command, "--join-frame")),
                    "ready_hold_frames": 4, "quintic_blend_intervals": 10,
                    "protected_precontact_seconds": 0.1,
                },
                "synthesis": {},
                "proof": {
                    "fps": 50,
                    "source_contact_frame": int(_flag(command, "--contact-frame")),
                    "source_join_frame": int(_flag(command, "--join-frame")),
                    "output_contact_frame": 25, "protected_frames_before_contact": 5,
                    "protected_window_bitwise_equal": True,
                    "pose_and_body_velocity_source_suffix_bitwise_equal": True,
                    "frame0_shared_ready_pose_bitwise_equal": True,
                    "ready_source_velocity_channels_ignored": True,
                    "ready_velocity_definition": "explicit_bitwise_zero",
                    "initial_zero_velocity_frames": 3,
                    "joint_position_continuous_quintic_endpoint_c2": True,
                    "finite": True, "contact_time_from_frame0_s": 0.5,
                },
                "output": {"npz": {"sha256": _sha(output_payload)}},
                "authorization": {
                    "training_authorized": False, "deployment_authorized": False,
                    "hardware_authorized": False,
                },
                "required_next_gates": [], "explicit_non_claims": [],
            }
            contract_path.write_bytes(_canonical(contract))
            return subprocess.CompletedProcess(command, 0, "generated")

        input_path = Path(_flag(command, "--input"))
        if input_path.parent.name == fail_topp_cell:
            return subprocess.CompletedProcess(command, 7, "registered TOPP failure")
        output = Path(_flag(command, "--output"))
        report = Path(_flag(command, "--report"))
        markdown = Path(_flag(command, "--md"))
        with np.load(input_path, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files
                    if not key.startswith("kinematics_migration_")}
        np.savez(output, **data)
        output_sha = _sha(output.read_bytes())
        contact_out = (round(reported_runup_s * 50)
                       if output_contact_frame is None else output_contact_frame)
        certificate = {key: {} for key in runner.TOPP_CERTIFICATE_KEYS}
        certificate.update({
            "tool": runner.TOPP_TOOL, "algorithm_scope": runner.TOPP_ALGORITHM_SCOPE,
            "search_objective": "runup",
            "generated_utc": "2026-07-17T00:00:00Z", "verdict": "PASS",
            "direction": "slowed", "chosen_scale": 1.0, "feasible_reason": "fixture",
            "files": {
                "input": {"sha256": _sha(input_path.read_bytes())},
                "output": {"sha256": output_sha},
                "report_path": str(report), "markdown_path": str(markdown),
            },
            "acceptance": {
                "cop_dose_final": 0.01, "fric_dose_final": 0.01,
                "tau_dose_final": 0.01, "within_budget": True,
                "kin_out_window_clean": True, "kin_lock_window_clean": True,
                "kinematic_hard_limits_clean": True,
            },
            "source": {
                "frames": 40, "fps": 50, "contact_frame": 25,
                "phase": 25.0 / 39.0, "runup_s": 0.5, "duration_s": 0.78,
                "clean_blade_speed_mps": 1.0, "mean_abs_acc": 0.0,
            },
            "output": {
                "frames": 40, "fps": 50.0, "contact_frame": contact_out,
                "phase_out": contact_out / 39.0, "body_mode": "fk",
                "runup_s": reported_runup_s,
                "duration_s": 0.78, "runup_change_x": 1.2,
                "duration_change_x": 1.0, "wait_s": 0.0, "mean_abs_acc": 0.0,
            },
            "fidelity": {
                "contact_row_bitwise": True, "blade_speed_clean_out_mps": 1.0,
                "blade_speed_dev_frac": 0.0, "face_normal_diff_deg": 0.0,
                "first_frame_max_joint_vel": 0.0,
            },
            "timing_bound": {
                "candidate_start_to_contact_s": reported_runup_s,
                "bound_semantics": "feasible upper bound within this searched family",
                "strict_global_minimum_proven": False,
            },
            "runtime_provenance": {
                "mjcf": {"sha256": queue["runtime"]["mjcf_sha256"]},
                "urdf": {"sha256": queue["runtime"]["urdf_sha256"]},
                "body_order": {"sha256": queue["runtime"]["body_order_sha256"]},
                "tool": {"topp_mintime": {"sha256": queue["runtime"]["topp_sha256"]}},
            },
            "budget_provenance": {
                "clips": [
                    {"sha256": queue["assets"]["forehand"]["sha256"]},
                    {"sha256": queue["assets"]["backhand"]["sha256"]},
                ],
                "scale": budget_scale,
                "envelope": [1.0] * 31,
            },
            "budget": {
                "cop_gate": 0.1, "fric_gate": 0.05, "tau_gate": 0.02,
                "vel_limit_frac": 0.85, "kin_vel_target": 0.95,
                "kin_acc_target": 0.95, "note": "fixture",
            },
        })
        report.write_bytes(_canonical(certificate))
        markdown.write_text("fixture\n")
        return subprocess.CompletedProcess(command, 0, "retimed")

    return execute


def test_dry_run_derives_exact_four_midpoint_cells_without_creating_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    result = _plan(paths)

    assert result["status"] == "dry_run_passed_no_namespace_created"
    assert not paths["root"].exists()
    rows = {row["cell_id"]: row for row in result["cells"]}
    assert set(rows) == set(runner.EXPECTED_STAGE2_CELLS)
    assert rows["fh_rf_d12"]["join_frame"] == 54
    assert rows["bh_rb_d12"]["join_frame"] == 33
    assert {row["blend_intervals"] for row in rows.values()} == {10}
    assert {row["output_contact_frame"] for row in rows.values()} == {25}


def test_missing_receipt_fails_before_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["receipt"].unlink()

    with pytest.raises(runner.Stage2Error, match="attestation receipt"):
        _plan(paths)
    assert not paths["root"].exists()


def test_missing_attested_stage1_generator_copy_is_not_a_v6_runtime_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    stage1 = Path(json.loads(paths["activation"].read_text())["stage1_namespace"])
    (stage1 / "build_ready_to_strike_motion.py").unlink()

    assert _plan(paths)["status"] == "dry_run_passed_no_namespace_created"
    assert not paths["root"].exists()


def test_tampered_attested_stage1_generator_copy_is_not_a_v6_runtime_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    stage1 = Path(json.loads(paths["activation"].read_text())["stage1_namespace"])
    (stage1 / "build_ready_to_strike_motion.py").write_bytes(b"tampered\n")

    assert _plan(paths)["status"] == "dry_run_passed_no_namespace_created"
    assert not paths["root"].exists()


def test_tampered_receipt_fails_sha_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["receipt"].write_bytes(paths["receipt"].read_bytes() + b" \n")

    with pytest.raises(runner.Stage2Error, match="receipt bytes"):
        _plan(paths)
    assert not paths["root"].exists()


def test_missing_prior_failure_summary_blocks_v2_before_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    activation = json.loads(paths["activation"].read_text())
    Path(activation["prior_failed_attempt"]["summary_path"]).unlink()

    with pytest.raises(runner.Stage2Error, match="prior Stage2 failure summary"):
        _plan(paths)
    assert not paths["root"].exists()


def test_tampered_prior_failure_summary_blocks_v2_before_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    activation = json.loads(paths["activation"].read_text())
    prior = Path(activation["prior_failed_attempt"]["summary_path"])
    prior.write_bytes(prior.read_bytes() + b" ")

    with pytest.raises(runner.Stage2Error, match="summary bytes changed"):
        _plan(paths)
    assert not paths["root"].exists()


def _rewrite_prior_summary(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch, mutate,
) -> None:
    activation = json.loads(paths["activation"].read_text())
    prior_path = Path(activation["prior_failed_attempt"]["summary_path"])
    prior = json.loads(prior_path.read_text())
    mutate(prior)
    payload = _canonical(prior)
    prior_path.write_bytes(payload)
    binding = dict(runner.EXPECTED_PRIOR_ATTEMPT)
    binding["summary_sha256"] = _sha(payload)
    activation["prior_failed_attempt"] = binding
    paths["activation"].write_bytes(_canonical(activation))
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_ATTEMPT", binding)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("runtime_snapshot_shas", "runtime inputs differ"),
        ("asset_snapshot_shas", "assets differ"),
    ],
)
def test_prior_summary_scientific_map_drift_fails_before_namespace(
    field: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    def mutate(prior):
        first = next(iter(prior[field]))
        prior[field][first] = "0" * 64

    _rewrite_prior_summary(paths, monkeypatch, mutate)
    with pytest.raises(runner.Stage2Error, match=message):
        _plan(paths)
    assert not paths["root"].exists()


@pytest.mark.parametrize("drift", ["topp_rc", "timing"])
def test_prior_v2_cannot_invent_success_or_timing(
    drift: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    def mutate(prior):
        if drift == "topp_rc":
            prior["rows"][0]["topp_rc"] = 0
        else:
            prior["screening_acceptance"]["timing_by_cell_s"] = {
                prior["rows"][0]["cell_id"]: 0.5
            }

    _rewrite_prior_summary(paths, monkeypatch, mutate)
    match = "did not fail in TOPP" if drift == "topp_rc" else "screening outcome changed"
    with pytest.raises(runner.Stage2Error, match=match):
        _plan(paths)
    assert not paths["root"].exists()


def test_receipt_cannot_upgrade_screening_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = json.loads(paths["receipt"].read_text())
    receipt["formal_claims"]["physics_replay_exact"] = True
    receipt_payload = _canonical(receipt)
    paths["receipt"].write_bytes(receipt_payload)
    activation = json.loads(paths["activation"].read_text())
    activation["required_attestation_receipt_sha256"] = _sha(receipt_payload)
    paths["activation"].write_bytes(_canonical(activation))

    with pytest.raises(runner.Stage2Error, match="formal claims changed"):
        _plan(paths)
    assert not paths["root"].exists()


def test_unknown_activation_field_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    activation = json.loads(paths["activation"].read_text())
    activation["unreviewed_override"] = True
    paths["activation"].write_bytes(_canonical(activation))

    with pytest.raises(runner.Stage2Error, match="activation keys changed"):
        _plan(paths)
    assert not paths["root"].exists()


def test_existing_namespace_is_no_clobber_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["root"].mkdir()
    marker = paths["root"] / "keep"
    marker.write_text("do not replace")

    with pytest.raises(runner.Stage2Error, match="already exists"):
        _plan(paths)
    assert marker.read_text() == "do not replace"


def test_execute_existing_namespace_does_not_write_a_failure_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["root"].mkdir()
    marker = paths["root"] / "keep"
    marker.write_bytes(b"preexisting-bytes")

    with pytest.raises(runner.Stage2Error, match="already exists"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"],
        )

    assert list(paths["root"].iterdir()) == [marker]
    assert marker.read_bytes() == b"preexisting-bytes"


def test_losing_atomic_mkdir_race_does_not_touch_winner_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    original_mkdir = Path.mkdir

    def racing_mkdir(self: Path, *args, **kwargs):
        if self == paths["root"]:
            original_mkdir(self, *args, **kwargs)
            (self / "winner-owned").write_bytes(b"winner")
            raise FileExistsError("simulated concurrent winner")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    with pytest.raises(runner.Stage2Error, match="unexpected Stage2 failure"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"],
        )

    assert [path.name for path in paths["root"].iterdir()] == ["winner-owned"]
    assert (paths["root"] / "winner-owned").read_bytes() == b"winner"


def test_root_must_equal_the_single_activation_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    with pytest.raises(runner.Stage2Error, match="one-shot Stage2 activation namespace"):
        runner.plan_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=tmp_path / "alternate-stage2", runner_source=paths["source"],
        )
    assert not (tmp_path / "alternate-stage2").exists()


def test_execute_requires_exact_confirmation_token_before_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    with pytest.raises(runner.Stage2Error, match=runner.CONFIRM_TOKEN):
        runner.run_stage2(
            activation_path=paths["activation"],
            queue_path=paths["queue"],
            root=paths["root"],
            execute=True,
            confirm="WRONG",
            runner_source=paths["source"],
        )
    assert not paths["root"].exists()


def test_execute_runs_each_registered_cell_once_and_separates_timing_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    result = runner.run_stage2(
        activation_path=paths["activation"], queue_path=paths["queue"],
        root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
        runner_source=paths["source"], command_runner=_fake_executor(paths, calls),
    )

    assert result["status"] == "stage2_execution_complete_no_retry"
    assert len(calls) == 4
    assert all(command[:3] == [
        runner.EXPECTED_TOPP_RUNTIME["interpreter"]["path"], "-I", "-B",
    ] for command in calls)
    assert sum("--output-contract" in command for command in calls) == 0
    assert sum("--report" in command for command in calls) == 4
    assert result["generator_commands_executed"] == 0
    assert result["prior_diagnostic_logs_consumed"] is False
    assert result["prior_v2_timing_available"] is False
    assert set(result["prior_v2_snapshot_shas"]) == {
        *(f"asset:{name}" for name in ("forehand", "backhand")),
        *(f"candidate:{cell}" for cell in runner.EXPECTED_STAGE2_CELLS),
        *(f"contract:{cell}" for cell in runner.EXPECTED_STAGE2_CELLS),
    }
    topp_calls = [command for command in calls if "--report" in command]
    assert all(Path(_flag(command, "--input")).parent.parent == paths["root"]
               for command in topp_calls)
    assert all(
        Path(command[command.index("--budget-clips") + 1]).parent
        == paths["root"] / "snapshots" / "assets"
        and Path(command[command.index("--budget-clips") + 2]).parent
        == paths["root"] / "snapshots" / "assets"
        for command in topp_calls
    )
    assert result["screening_acceptance"]["at_or_below_0p5_cells"] == []
    assert result["screening_acceptance"]["any_shared_ready_pass"] is False
    assert result["asset_snapshot_shas"] == {
        "forehand": json.loads(paths["queue"].read_text())["assets"]["forehand"]["sha256"],
        "backhand": json.loads(paths["queue"].read_text())["assets"]["backhand"]["sha256"],
    }
    assert result["mjcf_runtime_postflight"] == result["mjcf_runtime_preflight"]
    assert result["execution_snapshot_stability"]["postflight_matches_preflight"] is True
    assert result["execution_snapshot_stability"]["runtime"]["file_count"] > 0
    assert result["execution_snapshot_stability"]["assets"]["file_count"] == 2
    assert result["formal_claims"]["mjcf_closure_exact"] is True
    assert (paths["root"] / "snapshots" / "mjcf_runtime_postflight.json").is_file()
    assert (paths["root"] / "stage2_summary.json").is_file()


def test_execute_rejects_child_tamper_of_materialized_mjcf_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    base_executor = _fake_executor(paths, calls)
    lock = threading.Lock()
    tampered = False

    def tampering_executor(command, *, cwd, env):
        nonlocal tampered
        completed = base_executor(command, cwd=cwd, env=env)
        with lock:
            if not tampered:
                mjcf = Path(_flag(list(command), "--mjcf"))
                os.chmod(mjcf, 0o600)
                mjcf.write_bytes(mjcf.read_bytes() + b"child-tamper")
                tampered = True
        return completed

    with pytest.raises(runner.Stage2Error, match="no retry attempted"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"], command_runner=tampering_executor,
        )

    summary = json.loads((paths["root"] / "stage2_summary.json").read_text())
    assert len(calls) == 4
    assert summary["status"] == "stage2_terminal_failure_no_retry"
    assert any("materialized runtime snapshot changed during Stage2" in error
               for error in summary["input_stability_errors"])
    assert summary["execution_snapshot_stability"]["runtime"] is None
    assert summary["formal_claims"]["mjcf_closure_exact"] is False


def test_execute_rejects_system_dynamic_closure_drift_at_postflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    stable_preflight = runner._preflight_mjcf_runtime
    preflight_calls = 0

    def drifting_preflight(**kwargs):
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls == 1:
            return stable_preflight(**kwargs)
        raise runner.Stage2Error("TOPP dynamic dependency manifest changed")

    monkeypatch.setattr(runner, "_preflight_mjcf_runtime", drifting_preflight)
    with pytest.raises(runner.Stage2Error, match="no retry attempted"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"],
            command_runner=_fake_executor(paths, calls),
        )

    summary = json.loads((paths["root"] / "stage2_summary.json").read_text())
    assert preflight_calls == 2
    assert len(calls) == 4
    assert summary["status"] == "stage2_terminal_failure_no_retry"
    assert "TOPP dynamic dependency manifest changed" in summary["input_stability_errors"]
    assert summary["mjcf_runtime_postflight"] is None
    assert summary["execution_snapshot_stability"]["postflight_matches_preflight"] is False
    assert summary["formal_claims"]["mjcf_closure_exact"] is False
    assert not (paths["root"] / "snapshots" / "mjcf_runtime_postflight.json").exists()


def test_execute_rejects_any_postflight_receipt_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    stable_preflight = runner._preflight_mjcf_runtime
    preflight_calls = 0

    def differing_postflight(**kwargs):
        nonlocal preflight_calls
        preflight_calls += 1
        receipt = dict(stable_preflight(**kwargs))
        if preflight_calls == 2:
            receipt["stdout_sha256"] = "0" * 64
        return receipt

    monkeypatch.setattr(runner, "_preflight_mjcf_runtime", differing_postflight)
    with pytest.raises(runner.Stage2Error, match="no retry attempted"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"],
            command_runner=_fake_executor(paths, calls),
        )
    summary = json.loads((paths["root"] / "stage2_summary.json").read_text())
    assert preflight_calls == 2
    assert any("actual-loaded dynamic closure changed" in error
               for error in summary["input_stability_errors"])
    assert summary["mjcf_runtime_postflight"]["stdout_sha256"] == "0" * 64
    assert summary["execution_snapshot_stability"]["postflight_matches_preflight"] is False
    assert summary["formal_claims"]["mjcf_closure_exact"] is False


def test_execute_failure_is_terminal_summary_and_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    with pytest.raises(runner.Stage2Error, match="no retry attempted"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"],
            command_runner=_fake_executor(paths, calls, fail_topp_cell="fh_rf_d12"),
        )

    summary = json.loads((paths["root"] / "stage2_summary.json").read_text())
    assert summary["status"] == "stage2_terminal_failure_no_retry"
    assert len(calls) == 4  # four prior candidates, each TOPP attempted exactly once
    failed = next(row for row in summary["rows"] if row["cell_id"] == "fh_rf_d12")
    assert failed["generator_rc"] == 0
    assert failed["topp_rc"] == 7
    assert summary["generator_commands_executed"] == 0
    assert summary["execution_snapshot_stability"]["postflight_matches_preflight"] is True
    assert summary["formal_claims"]["mjcf_closure_exact"] is False


def test_execute_rejects_nonregistered_topp_budget_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    with pytest.raises(runner.Stage2Error, match="no retry attempted"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"],
            command_runner=_fake_executor(paths, calls, budget_scale=1.0),
        )

    summary = json.loads((paths["root"] / "stage2_summary.json").read_text())
    assert summary["status"] == "stage2_terminal_failure_no_retry"
    assert all("TOPP budget scale changed" in row["terminal_error"]
               for row in summary["rows"])


def test_execute_rejects_timing_that_disagrees_with_contact_frame_and_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    with pytest.raises(runner.Stage2Error, match="no retry attempted"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"],
            command_runner=_fake_executor(
                paths, calls, reported_runup_s=0.6, output_contact_frame=25
            ),
        )

    summary = json.loads((paths["root"] / "stage2_summary.json").read_text())
    assert all("runup disagrees with contact frame/fps" in row["terminal_error"]
               for row in summary["rows"])


def test_unexpected_post_namespace_failure_gets_terminal_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def explode(*args, **kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(runner, "_validate_topp", explode)
    with pytest.raises(runner.Stage2Error, match="unexpected Stage2 failure"):
        runner.run_stage2(
            activation_path=paths["activation"], queue_path=paths["queue"],
            root=paths["root"], execute=True, confirm=runner.CONFIRM_TOKEN,
            runner_source=paths["source"], command_runner=_fake_executor(paths, calls),
        )

    terminal = json.loads(
        (paths["root"] / "stage2_unexpected_terminal_failure.json").read_text()
    )
    assert terminal["status"] == "stage2_unexpected_terminal_failure_no_retry"
    assert terminal["error_type"] == "RuntimeError"
    assert terminal["trainer_or_robot_signals"] == []


def test_current_repo_activation_exactly_binds_the_tracked_runner() -> None:
    activation = json.loads(
        (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v8_20260717.json").read_text()
    )
    queue = runner._validate_queue(json.loads(
        (REPO / "configs/ready_to_strike_join_ladder_20260717.yaml").read_text()
    ))

    validated, cells, receipt_path, receipt_sha = runner._validate_activation(
        activation, queue, runner_sha256=_sha(SCRIPT.read_bytes())
    )
    assert validated["launch_authorized"] is True
    assert validated["topp_runtime"] == runner.EXPECTED_TOPP_RUNTIME
    assert len(cells) == 4
    assert receipt_path == Path(validated["required_attestation_receipt"])
    assert receipt_sha == validated["required_attestation_receipt_sha256"]
