from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
        "runtime_snapshot_shas": {"runtime.py": "7" * 64},
        "asset_snapshot_shas": {"forehand": "8" * 64, "backhand": "9" * 64},
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
        "failure_class": (
            "mjcf_snapshot_omitted_referenced_mesh_assets"
        ),
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
            log_path = fixture_root / f"{cell_id}.log"
            log_path.write_bytes(b"missing mesh.STL: No such file or directory\n")
            candidate = runner._read_snapshot(candidate_path, "fixture prior candidate")
            contract = runner._read_snapshot(contract_path, "fixture prior contract")
            log = runner._read_snapshot(log_path, "fixture prior log")
            info = {
                "candidate_sha256": candidate.sha256,
                "generator_contract_sha256": contract.sha256,
                "frames": 40, "phase": 25.0 / 39.0,
                "joint_path_l2": 0.0, "joint_curvature_l2": 0.0,
                "max_joint_step_rad": 0.0,
            }
            records[cell_id] = {
                "candidate": candidate, "contract": contract, "log": log, "info": info,
            }
            snapshots[f"prior:candidate:{cell_id}"] = candidate
            snapshots[f"prior:contract:{cell_id}"] = contract
            snapshots[f"prior:topp_log:{cell_id}"] = log
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
) -> tuple[dict[str, object], dict[str, object], str, str, str]:
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
    cell_id = next(iter(runner.EXPECTED_STAGE2_CELLS))
    cell_root = prior_root / cell_id
    candidate_payload = b"prior-v2-candidate\n"
    contract_payload = b"prior-v2-contract\n"
    log_payload = b"missing mesh.STL: No such file or directory\n"
    _write(cell_root / "candidate.npz", candidate_payload)
    _write(cell_root / "candidate.contract.json", contract_payload)
    _write(cell_root / "topp/run.log", log_payload)
    return activation, queue, _sha(candidate_payload), _sha(contract_payload), _sha(log_payload)


@pytest.mark.parametrize("tamper", ["candidate", "log"])
def test_real_prior_v2_preflight_rejects_tampered_candidate_or_log_before_runtime(
    tamper: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    activation, queue, candidate_sha, contract_sha, log_sha = (
        _prepare_real_prior_v2_preflight_prefix(paths, monkeypatch)
    )
    cell_id = next(iter(runner.EXPECTED_STAGE2_CELLS))
    candidates = dict(runner.EXPECTED_PRIOR_CANDIDATES)
    candidates[cell_id] = (
        "0" * 64 if tamper == "candidate" else candidate_sha,
        contract_sha,
    )
    logs = dict(runner.EXPECTED_PRIOR_TOPP_LOGS)
    logs[cell_id] = "0" * 64 if tamper == "log" else log_sha
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_CANDIDATES", candidates)
    monkeypatch.setattr(runner, "EXPECTED_PRIOR_TOPP_LOGS", logs)
    receipt_sha = _sha(b"prior-v2-receipt\n")
    message = "candidate or contract SHA" if tamper == "candidate" else "TOPP log SHA"

    with pytest.raises(runner.Stage2Error, match=message):
        REAL_COLLECT_PRIOR_V2_INPUTS(
            activation=activation, queue=queue, body_order=("body",),
            expected_receipt_sha=receipt_sha,
        )


def test_v2_activation_bytes_remain_immutable() -> None:
    payload = (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v2_20260717.json").read_bytes()
    assert _sha(payload) == "8742aadff796218f170fede3f6e386e54314e086740f4ad82b9242f52667ab10"


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


def test_missing_attested_stage1_generator_copy_fails_before_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    stage1 = Path(json.loads(paths["activation"].read_text())["stage1_namespace"])
    (stage1 / "build_ready_to_strike_motion.py").unlink()

    with pytest.raises(runner.Stage2Error, match="Stage1 generator copy"):
        _plan(paths)
    assert not paths["root"].exists()


def test_tampered_attested_stage1_generator_copy_fails_before_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    stage1 = Path(json.loads(paths["activation"].read_text())["stage1_namespace"])
    (stage1 / "build_ready_to_strike_motion.py").write_bytes(b"tampered\n")

    with pytest.raises(runner.Stage2Error, match="generator copy SHA changed"):
        _plan(paths)
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
    assert sum("--output-contract" in command for command in calls) == 0
    assert sum("--report" in command for command in calls) == 4
    assert result["generator_commands_executed"] == 0
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
    assert (paths["root"] / "stage2_summary.json").is_file()


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
        (REPO / "configs/ready_to_strike_join_ladder_stage2_activation_v3_20260717.json").read_text()
    )
    queue = runner._validate_queue(json.loads(
        (REPO / "configs/ready_to_strike_join_ladder_20260717.yaml").read_text()
    ))

    validated, cells, receipt_path, receipt_sha = runner._validate_activation(
        activation, queue, runner_sha256=_sha(SCRIPT.read_bytes())
    )
    assert validated["launch_authorized"] is True
    assert len(cells) == 4
    assert receipt_path == Path(validated["required_attestation_receipt"])
    assert receipt_sha == validated["required_attestation_receipt_sha256"]
