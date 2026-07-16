from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/attest_ready_to_strike_ladder_stage1.py"
QUEUE_TEMPLATE = REPO / "configs/ready_to_strike_join_ladder_20260717.yaml"


def _load_module(path: Path = SCRIPT):
    spec = importlib.util.spec_from_file_location(f"stage1_attestor_{path.stat().st_ino}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _evidence(path: Path, *, generator_style: bool = False) -> dict:
    info = path.stat()
    result = {"path": str(path), "bytes": info.st_size, "sha256": _sha(path)}
    if generator_style:
        result.update(
            device=info.st_dev,
            inode=info.st_ino,
            mtime_ns=info.st_mtime_ns,
            ctime_ns=info.st_ctime_ns,
        )
    return result


def _write_npz(path: Path, *, frames: int = 40, candidate: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(frames, dtype=np.float32)[:, None]
    basis = np.concatenate((0.01 * t, -0.02 * t, 0.003 * t * t), axis=1)
    q = np.tile(basis, (1, 11))[:, :31].astype(np.float32)
    q[:4] = q[0]
    qv = np.gradient(q, 1.0 / 50.0, axis=0).astype(np.float32)
    lin = np.zeros((frames, 2, 3), dtype=np.float32)
    ang = np.zeros((frames, 2, 3), dtype=np.float32)
    pos = np.zeros((frames, 2, 3), dtype=np.float32)
    quat = np.zeros((frames, 2, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    with path.open("wb") as stream:
        np.savez(
            stream,
            fps=np.array([50], dtype=np.int64),
            joint_pos=q,
            joint_vel=qv,
            body_pos_w=pos,
            body_quat_w=quat,
            body_lin_vel_w=lin,
            body_ang_vel_w=ang,
            kinematics_schema_version=np.array([2], dtype=np.int64),
            body_pos_point=np.array("link_origin"),
            body_lin_vel_point=np.array("center_of_mass"),
            body_names=np.array(["body0", "body1"]),
        )


def _write_candidate_npz(
    path: Path, *, action_path: Path, ready_path: Path, contact_frame: int, frames: int = 40
) -> None:
    _write_npz(path, frames=frames, candidate=True)
    with np.load(path, allow_pickle=False) as archive:
        candidate = {key: np.asarray(archive[key]).copy() for key in archive.files}
    with np.load(action_path, allow_pickle=False) as archive:
        action = {key: np.asarray(archive[key]).copy() for key in archive.files}
    with np.load(ready_path, allow_pickle=False) as archive:
        ready = {key: np.asarray(archive[key]).copy() for key in archive.files}
    for key in ("joint_pos", "body_pos_w", "body_quat_w"):
        candidate[key][0] = ready[key][0]
    candidate["joint_pos"][19:27] = action["joint_pos"][contact_frame - 6:contact_frame + 2]
    candidate["joint_vel"] = np.gradient(
        candidate["joint_pos"], 1.0 / 50.0, axis=0
    ).astype(np.float32)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        candidate[key][20:26] = action[key][contact_frame - 5:contact_frame + 1]
    with path.open("wb") as stream:
        np.savez(stream, **candidate)


def _write_topp_output(path: Path, *, candidate_path: Path, frames: int = 51) -> None:
    _write_npz(path, frames=frames)
    with np.load(path, allow_pickle=False) as archive:
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
    with np.load(candidate_path, allow_pickle=False) as archive:
        contact = np.asarray(archive["joint_pos"])[25].copy()
    output["joint_pos"][30] = contact
    output["joint_vel"] = np.gradient(
        output["joint_pos"].astype(np.float64), 1.0 / 50.0, axis=0
    ).astype(np.float32)
    with path.open("wb") as stream:
        np.savez(stream, **output)


def _generator_contract(
    *, queue: dict, cell: dict, candidate: Path, runtime_generator: Path
) -> dict:
    action = queue["assets"][cell["action"]]
    ready = queue["assets"][cell["ready_source"]]
    contact = action["contact_frame"]
    join = contact - cell["delta"]
    blend = 22 - cell["delta"]
    return {
        "schema_version": 1,
        "artifact_kind": "host_only_ready_to_strike_motion_candidate",
        "status": "candidate_only_all_runtime_and_safety_gates_open",
        "inputs": {
            "source_schema2_npz": _evidence(Path(action["path"]), generator_style=True),
            "shared_ready_schema2_npz": _evidence(
                Path(ready["path"]), generator_style=True
            ),
            "shared_ready_frame": 0,
        },
        "tool": {
            **_evidence(runtime_generator, generator_style=True),
            "binding_semantics": "source_file_snapshot_at_main_entry_unchanged_before_publish",
        },
        "request": {
            "source_contact_frame": contact,
            "source_join_frame": join,
            "ready_hold_frames": 4,
            "quintic_blend_intervals": blend,
            "protected_precontact_seconds": 0.1,
        },
        "synthesis": {},
        "proof": {
            "fps": 50,
            "source_join_frame": join,
            "source_contact_frame": contact,
            "output_contact_frame": 25,
            "protected_frames_before_contact": 5,
            "protected_window_bitwise_equal": True,
            "protected_window_sha256": "a" * 64,
            "pose_and_body_velocity_source_suffix_bitwise_equal": True,
            "frame0_shared_ready_pose_bitwise_equal": True,
            "ready_source_velocity_channels_ignored": True,
            "ready_velocity_definition": "explicit_bitwise_zero",
            "initial_zero_velocity_frames": 3,
            "joint_position_continuous_quintic_endpoint_c2": True,
            "finite": True,
            "contact_time_from_frame0_s": 0.5,
            "quaternion_max_norm_error": 0.0,
            "producer_gradient_join_velocity_error_rad_s": 0.0,
        },
        "output": {
            "npz": _evidence(candidate),
            "contract_binding": "JSON binds exact NPZ SHA-256; publication is no-clobber",
        },
        "authorization": {
            "host_candidate_materialized": True,
            "topp_runup_0p5_pass": False,
            "l0_static_pass": False,
            "vendor_l1_pass": False,
            "self_hit_pass": False,
            "table_net_clearance_5mm_pass": False,
            "dynamics_pass": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "required_next_gates": [],
        "explicit_non_claims": [],
    }


def _certificate(
    *, queue: dict, candidate: Path, output: Path, report: Path, markdown: Path,
    runtime_paths: dict[str, Path], frames: int,
) -> dict:
    dependencies = {
        name: _evidence(Path(queue["runtime"]["checkout_path"]) / relative)
        for name, relative in M.TOPP_DEPENDENCIES.items()
    }
    acceptance = {
        "cop_dose_final": 0.02,
        "fric_dose_final": 0.01,
        "tau_dose_final": 0.01,
        "within_budget": True,
        "kin_out_window_clean": True,
        "kin_lock_window_clean": True,
        "kinematic_hard_limits_clean": True,
    }
    timing = {
        "candidate_start_to_contact_s": 0.6,
        "bound_semantics": "feasible upper bound within this searched family",
        "strict_global_minimum_proven": False,
    }
    fidelity = {
        "contact_row_bitwise": True,
        "blade_speed_clean_out_mps": 2.0,
        "blade_speed_dev_frac": 0.01,
        "face_normal_diff_deg": 0.0,
        "first_frame_max_joint_vel": 0.0,
    }
    return {
        "tool": M.TOPP_TOOL,
        "algorithm_scope": M.TOPP_ALGORITHM_SCOPE,
        "search_objective": "runup",
        "generated_utc": "2026-07-16 00:00:00Z",
        "verdict": "PASS",
        "direction": "slowed",
        "chosen_scale": 1.0,
        "feasible_reason": "fixture",
        "files": {
            "input": _evidence(candidate),
            "output": _evidence(output),
            "report_path": str(report),
            "markdown_path": str(markdown),
        },
        "budget_provenance": {
            "clips": [
                _evidence(Path(queue["assets"]["forehand"]["path"])),
                _evidence(Path(queue["assets"]["backhand"]["path"])),
            ],
            "scale": 1.0,
            "envelope": [1.0, 2.0, 3.0],
        },
        "runtime_provenance": {
            "mjcf": _evidence(runtime_paths["mjcf_sha256"]),
            "urdf": _evidence(runtime_paths["urdf_sha256"]),
            "body_order": _evidence(runtime_paths["body_order_sha256"]),
            "tool": {
                "topp_mintime": _evidence(runtime_paths["topp_sha256"]),
                "dependencies": dependencies,
            },
        },
        "source": {
            "frames": frames,
            "fps": 50,
            "contact_frame": 25,
            "phase": 25.0 / (frames - 1),
            "runup_s": 0.5,
            "duration_s": (frames - 1) / 50.0,
            "clean_blade_speed_mps": 2.0,
            "mean_abs_acc": 1.0,
        },
        "output": {
            "frames": 51,
            "fps": 50.0,
            "contact_frame": 30,
            "phase_out": 0.6,
            "body_mode": "fk",
            "runup_s": 0.6,
            "duration_s": 1.0,
            "runup_change_x": 1.2,
            "duration_change_x": 1.0,
            "wait_s": 0.0,
            "mean_abs_acc": 1.0,
        },
        "acceptance": acceptance,
        "budget": {
            "cop_gate": 0.1,
            "fric_gate": 0.05,
            "tau_gate": 0.02,
            "vel_limit_frac": 0.85,
            "kin_vel_target": 0.98,
            "kin_acc_target": 0.98,
            "note": "fixture",
        },
        "fidelity": fidelity,
        "timing_bound": timing,
        "durations": {},
        "oracle_before": {},
        "oracle_after": {},
        "kin": {},
        "answer": {},
        "baseline_law": {},
        "stretch": {},
        "outer_trace": [
            {
                "gamma": 1.0,
                "feasible": True,
                "reason": "fixture",
                "iters": 1,
                "T_out": 51,
                "duration_s": 1.0,
                "runup_s": 0.6,
                "cop": 0.02,
                "fric": 0.01,
                "tau": 0.01,
            }
        ],
        "inner_trace_best": [{"iter": 0}],
    }


@pytest.fixture()
def complete_stage1(tmp_path: Path) -> dict:
    runtime = tmp_path / "runtime"
    runtime_paths = {
        key: runtime / relative for key, relative in M.RUNTIME_RELATIVE_PATHS.items()
    }
    for index, path in enumerate(runtime_paths.values()):
        payload = b"body0\nbody1\n" if path == runtime_paths["body_order_sha256"] else f"runtime-{index}\n".encode()
        _write(path, payload)
    for index, relative in enumerate(M.TOPP_DEPENDENCIES.values()):
        _write(runtime / relative, f"dependency-{index}\n".encode())
    assets = tmp_path / "assets"
    forehand = assets / "forehand.npz"
    backhand = assets / "backhand.npz"
    _write_npz(forehand, frames=80)
    _write_npz(backhand, frames=60)

    queue = json.loads(QUEUE_TEMPLATE.read_text(encoding="utf-8"))
    queue["runtime"]["checkout_path"] = str(runtime)
    for key, path in runtime_paths.items():
        queue["runtime"][key] = _sha(path)
    queue["assets"]["forehand"]["path"] = str(forehand)
    queue["assets"]["forehand"]["sha256"] = _sha(forehand)
    queue["assets"]["backhand"]["path"] = str(backhand)
    queue["assets"]["backhand"]["sha256"] = _sha(backhand)
    queue_path = tmp_path / "queue.json"
    queue_bytes = (json.dumps(queue, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _write(queue_path, queue_bytes)

    root = tmp_path / "stage1"
    root.mkdir()
    _write(root / "queue.yaml", queue_bytes)
    _write(root / "build_ready_to_strike_motion.py", runtime_paths["generator_sha256"].read_bytes())
    generator_copy = root / "build_ready_to_strike_motion.py"
    # Production used a source-pinned copy from generator_source_commit; the older
    # runtime checkout is not required to contain that later generator file.
    runtime_paths["generator_sha256"].unlink()
    rows = []
    for cell in queue["staged_cells"]["stage1_endpoint_factorial"]:
        cell_root = root / cell["cell_id"]
        candidate = cell_root / "candidate.npz"
        contract = cell_root / "candidate.contract.json"
        topp = cell_root / "topp"
        output = topp / "motion.npz"
        report = topp / "certificate.json"
        markdown = topp / "certificate.md"
        action_path = Path(queue["assets"][cell["action"]]["path"])
        ready_path = Path(queue["assets"][cell["ready_source"]]["path"])
        contact = queue["assets"][cell["action"]]["contact_frame"]
        _write_candidate_npz(
            candidate,
            action_path=action_path,
            ready_path=ready_path,
            contact_frame=contact,
        )
        contract_doc = _generator_contract(
            queue=queue, cell=cell, candidate=candidate,
            runtime_generator=generator_copy,
        )
        _write(contract, (json.dumps(contract_doc, sort_keys=True) + "\n").encode())
        _write_topp_output(output, candidate_path=candidate)
        _write(markdown, b"# TOPP certificate\n")
        certificate = _certificate(
            queue=queue, candidate=candidate, output=output, report=report,
            markdown=markdown, runtime_paths=runtime_paths, frames=40,
        )
        _write(report, (json.dumps(certificate, sort_keys=True) + "\n").encode())
        with np.load(candidate, allow_pickle=False) as archive:
            q = np.asarray(archive["joint_pos"], dtype=np.float64)
        segment = np.diff(q[:26], axis=0)
        second = np.diff(q[:26], n=2, axis=0)
        rows.append(
            {
                **cell,
                "join_frame": contact - cell["delta"],
                "blend_intervals": 22 - cell["delta"],
                "generator_rc": 0,
                "candidate": str(candidate),
                "candidate_sha256": _sha(candidate),
                "contract_sha256": _sha(contract),
                "frames": 40,
                "phase": 25.0 / 39.0,
                "joint_path_l2": float(np.linalg.norm(segment, axis=1).sum()),
                "joint_curvature_l2": float(np.linalg.norm(second, axis=1).sum()),
                "max_joint_step_rad": float(np.abs(segment).max()),
                "topp_rc": 0,
                "topp_certificate_sha256": _sha(report),
                "topp_acceptance": certificate["acceptance"],
                "topp_timing_bound": certificate["timing_bound"],
                "topp_fidelity": certificate["fidelity"],
            }
        )
    summary = {
        "schema_version": 1,
        "status": "stage1_complete_no_retry",
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "generator_sha256": queue["runtime"]["generator_sha256"],
        "main_prereg_commit": M.EXPECTED_PREREG_COMMIT,
        "runtime_source_commit": queue["runtime"]["checkout_commit"],
        "rows": rows,
        "trainer_or_robot_signals": [],
        "automatic_retry": False,
    }
    _write(root / "stage1_summary.json", (json.dumps(summary, sort_keys=True) + "\n").encode())
    return {
        "root": root,
        "queue": queue_path,
        "runtime_paths": runtime_paths,
        "summary": root / "stage1_summary.json",
    }


def _attest(fixture: dict, **kwargs):
    return M.attest_stage1(root=fixture["root"], queue_path=fixture["queue"], **kwargs)


def test_positive_dry_run_and_execute_no_clobber(complete_stage1: dict) -> None:
    dry = _attest(complete_stage1)
    assert len(dry["cells"]) == 6
    assert all(cell["within_0p5_s"] is False for cell in dry["cells"])
    assert dry["formal_claims"] == {
        "physics_replay_exact": False,
        "source_closure_exact": False,
        "mjcf_closure_exact": False,
        "screening_activation_evidence_only": True,
    }
    receipt = complete_stage1["root"] / "stage1_historical_attestation.json"
    assert not receipt.exists()
    executed = _attest(
        complete_stage1, execute=True, confirm=M.CONFIRM_TOKEN,
        launch_argv=["--root", str(complete_stage1["root"]), "--execute"],
    )
    assert json.loads(receipt.read_text()) == executed
    with pytest.raises(M.AttestationError, match="receipt already exists"):
        _attest(complete_stage1, execute=True, confirm=M.CONFIRM_TOKEN)


def test_candidate_mutation_fails_closed(complete_stage1: dict) -> None:
    candidate = complete_stage1["root"] / "fh_rf_d17/candidate.npz"
    candidate.write_bytes(candidate.read_bytes() + b"mutated")
    with pytest.raises(M.AttestationError, match="candidate SHA"):
        _attest(complete_stage1)


def test_certificate_input_binding_mutation_fails_closed(complete_stage1: dict) -> None:
    report = complete_stage1["root"] / "fh_rf_d17/topp/certificate.json"
    doc = json.loads(report.read_text())
    doc["files"]["input"]["sha256"] = "0" * 64
    report.write_text(json.dumps(doc, sort_keys=True) + "\n")
    with pytest.raises(M.AttestationError):
        _attest(complete_stage1)


def test_runtime_tool_mutation_fails_closed(complete_stage1: dict) -> None:
    topp = complete_stage1["runtime_paths"]["topp_sha256"]
    topp.write_bytes(topp.read_bytes() + b"changed")
    with pytest.raises(M.AttestationError, match="runtime topp_sha256 SHA changed"):
        _attest(complete_stage1)


def test_certificate_field_mutation_fails_closed(complete_stage1: dict) -> None:
    report = complete_stage1["root"] / "fh_rf_d17/topp/certificate.json"
    doc = json.loads(report.read_text())
    doc["search_objective"] = "total"
    report.write_text(json.dumps(doc, sort_keys=True) + "\n")
    summary = json.loads(complete_stage1["summary"].read_text())
    summary["rows"][0]["topp_certificate_sha256"] = _sha(report)
    complete_stage1["summary"].write_text(json.dumps(summary, sort_keys=True) + "\n")
    with pytest.raises(M.AttestationError, match="objective"):
        _attest(complete_stage1)


def test_selected_outer_trace_must_bind_published_answer(complete_stage1: dict) -> None:
    report = complete_stage1["root"] / "fh_rf_d17/topp/certificate.json"
    doc = json.loads(report.read_text())
    doc["chosen_scale"] = 0.5
    report.write_text(json.dumps(doc, sort_keys=True) + "\n")
    summary = json.loads(complete_stage1["summary"].read_text())
    summary["rows"][0]["topp_certificate_sha256"] = _sha(report)
    complete_stage1["summary"].write_text(json.dumps(summary, sort_keys=True) + "\n")
    with pytest.raises(M.AttestationError, match="selected trace row gamma"):
        _attest(complete_stage1)


def test_consistently_rebound_candidate_cannot_change_protected_window(
    complete_stage1: dict,
) -> None:
    root = complete_stage1["root"] / "fh_rf_d17"
    candidate = root / "candidate.npz"
    with np.load(candidate, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["joint_pos"][20, 0] += np.float32(0.05)
    arrays["joint_vel"] = np.gradient(
        arrays["joint_pos"], 1.0 / 50.0, axis=0
    ).astype(np.float32)
    with candidate.open("wb") as stream:
        np.savez(stream, **arrays)

    contract = root / "candidate.contract.json"
    contract_doc = json.loads(contract.read_text())
    contract_doc["output"]["npz"] = _evidence(candidate)
    contract.write_text(json.dumps(contract_doc, sort_keys=True) + "\n")

    report = root / "topp/certificate.json"
    report_doc = json.loads(report.read_text())
    report_doc["files"]["input"] = _evidence(candidate)
    report.write_text(json.dumps(report_doc, sort_keys=True) + "\n")

    summary = json.loads(complete_stage1["summary"].read_text())
    row = summary["rows"][0]
    row["candidate_sha256"] = _sha(candidate)
    row["contract_sha256"] = _sha(contract)
    row["topp_certificate_sha256"] = _sha(report)
    complete_stage1["summary"].write_text(json.dumps(summary, sort_keys=True) + "\n")
    with pytest.raises(M.AttestationError, match="protected .* window"):
        _attest(complete_stage1)


def test_symlink_candidate_is_rejected(complete_stage1: dict) -> None:
    candidate = complete_stage1["root"] / "fh_rf_d17/candidate.npz"
    target = candidate.with_name("candidate.real.npz")
    candidate.rename(target)
    candidate.symlink_to(target)
    with pytest.raises(M.AttestationError, match="symlink"):
        _attest(complete_stage1)


def test_attestor_source_mutation_before_publish_fails_closed(
    complete_stage1: dict, tmp_path: Path
) -> None:
    source = tmp_path / "attestor_copy.py"
    source.write_bytes(SCRIPT.read_bytes())

    def mutate() -> None:
        source.write_bytes(source.read_bytes() + b"\n# mutation\n")

    with pytest.raises(M.AttestationError, match="changed"):
        _attest(
            complete_stage1,
            execute=True,
            confirm=M.CONFIRM_TOKEN,
            attestor_source=source,
            before_publish_hook=mutate,
        )
    assert not (complete_stage1["root"] / "stage1_historical_attestation.json").exists()
