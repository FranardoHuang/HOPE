from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_ready_to_strike_motion.py"
LADDER = ROOT / "configs/ready_to_strike_join_ladder_20260717.yaml"
STAGE2_ACTIVATION_V1 = ROOT / "configs/ready_to_strike_join_ladder_stage2_activation_20260717.json"
STAGE2_ACTIVATION = ROOT / "configs/ready_to_strike_join_ladder_stage2_activation_v3_20260717.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_ready_to_strike_motion_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_module()


def test_join_ladder_stage1_is_complete_nonduplicative_and_keeps_contact_tick_25() -> None:
    queue = json.loads(LADDER.read_text(encoding="utf-8"))
    assert queue["fixed_contract"]["output_contact_frame"] == 25
    assert queue["fixed_contract"]["delta_plus_blend_intervals"] == 22
    observed = {row["cell_id"] for row in queue["observed_baseline_not_to_rerun"]}
    assert observed == {"fh_shared_d06", "bh_shared_d06"}
    cells: set[tuple[str, str, int]] = set()
    cell_ids: set[str] = set()
    for cell in queue["staged_cells"]["stage1_endpoint_factorial"]:
        assert cell["cell_id"] not in cell_ids
        cell_ids.add(cell["cell_id"])
        action = cell["action"]
        contact = queue["assets"][action]["contact_frame"]
        delta = cell["delta"]
        identity = (action, cell["ready_source"], delta)
        assert identity not in cells
        cells.add(identity)
        join = contact - delta
        blend = 22 - delta
        assert join + delta == contact
        assert blend >= 5
        assert 4 + (blend - 1) + delta == 25
    assert len(cells) == 6
    assert ("forehand", "forehand", 6) not in cells
    assert ("backhand", "forehand", 6) not in cells
    assert ("backhand", "backhand", 6) in cells
    assert cells == {
        ("forehand", "forehand", 17),
        ("forehand", "backhand", 6),
        ("forehand", "backhand", 17),
        ("backhand", "forehand", 17),
        ("backhand", "backhand", 6),
        ("backhand", "backhand", 17),
    }
    assert queue["staged_cells"]["stage2_midpoint_rule"]["delta"] == 12
    assert queue["staged_cells"]["stage3_refinement_rule"]["left_refinement_delta"] == 9
    assert queue["staged_cells"]["stage3_refinement_rule"]["right_refinement_delta"] == 14


def test_join_ladder_binds_exact_sources_and_keeps_all_runtime_authority_false() -> None:
    queue = json.loads(LADDER.read_text(encoding="utf-8"))
    for key in (
        "generator_sha256",
        "topp_sha256",
        "mjcf_sha256",
        "urdf_sha256",
        "body_order_sha256",
    ):
        value = queue["runtime"][key]
        assert len(value) == 64 and set(value) <= set("0123456789abcdef")
    for asset in queue["assets"].values():
        value = asset["sha256"]
        assert len(value) == 64 and set(value) <= set("0123456789abcdef")
    fixed = queue["fixed_contract"]
    assert fixed["automatic_retry"] is False
    assert fixed["gpu_or_trainer_signals"] is False
    assert fixed["training_authorized"] is False
    assert fixed["deployment_authorized"] is False
    assert fixed["hardware_authorized"] is False
    assert queue["acceptance"]["candidate_start_to_contact_s_max"] == 0.5


def test_stage2_activation_is_exactly_the_registered_crossover_branch() -> None:
    queue = json.loads(LADDER.read_text(encoding="utf-8"))
    activation = json.loads(STAGE2_ACTIVATION.read_text(encoding="utf-8"))
    assert activation["parent_queue"]["sha256"] == hashlib.sha256(
        LADDER.read_bytes()
    ).hexdigest()
    observations = {row["cell_id"]: row for row in activation["observations"]}
    assert len(observations) == 8
    assert all(row["start_to_contact_s"] > 0.5 for row in observations.values())
    assert observations["fh_rf_d06"]["start_to_contact_s"] < observations["fh_rb_d06"][
        "start_to_contact_s"
    ]
    assert observations["bh_rb_d06"]["start_to_contact_s"] < observations["bh_rf_d06"][
        "start_to_contact_s"
    ]
    assert activation["decision"]["ready_by_side_crossover"] is True
    assert activation["decision"]["activate_both_ready_sources_at_midpoint"] is True
    assert activation["evidence_status"] == "historical_stage1_attested_screening_only"
    assert activation["launch_authorized"] is True
    assert activation["required_attestation_receipt"] == (
        activation["stage1_namespace"] + "/stage1_historical_attestation.json"
    )
    assert activation["required_attestation_receipt_sha256"] == (
        "7cf1c7c9613eb4a319dc8038934d3b439b8d3f948fab6b2650872e71f54c377f"
    )
    assert activation["stage2_runner"] == {
        "path": "scripts/run_ready_to_strike_join_ladder_stage2.py",
        "sha256": hashlib.sha256(
            (ROOT / "scripts/run_ready_to_strike_join_ladder_stage2.py").read_bytes()
        ).hexdigest(),
    }
    assert activation["stage2_namespace"].endswith(
        "/join_ladder_stage2_d12_v3_mjcf_closure"
    )
    assert activation["prior_failed_attempt"]["summary_sha256"] == (
        "6910db2826654123c576afa67b9c2e873c4785c2bd095b2f61abb26d5f1f1476"
    )
    assert activation["prior_failed_attempt"]["failure_class"] == (
        "mjcf_snapshot_omitted_referenced_mesh_assets"
    )
    assert activation["prior_failed_attempt"]["automatic_retry"] is False
    cells = activation["authorized_stage2_cells"]
    assert {(cell["action"], cell["ready_source"], cell["delta"]) for cell in cells} == {
        ("forehand", "forehand", 12),
        ("forehand", "backhand", 12),
        ("backhand", "forehand", 12),
        ("backhand", "backhand", 12),
    }
    assert queue["staged_cells"]["stage2_midpoint_rule"]["delta"] == 12
    assert activation["runtime_authority"] == {
        "cpu_only": True,
        "automatic_retry": False,
        "trainer_signal": False,
        "robot_command": False,
        "training_authorized": False,
        "deployment_authorized": False,
    }


def test_stage2_v1_activation_remains_an_immutable_failed_attempt_record() -> None:
    activation = json.loads(STAGE2_ACTIVATION_V1.read_text(encoding="utf-8"))
    assert activation["activation_id"] == "ready_to_strike_join_ladder_stage2_20260717"
    assert activation["stage2_runner"]["sha256"] == (
        "835cb56f9aac7c5b85791368e1de26745140a5de1af00144b0167fc2c26cf9f4"
    )
    assert activation["stage2_namespace"].endswith("/join_ladder_stage2_d12_8d74025e")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _motion_arrays(
    *, frames: int, ready: bool, migration_provenance: bool = False
) -> dict[str, np.ndarray]:
    fps = 50
    dt = 1.0 / fps
    joints = 3
    bodies = 2
    time = np.arange(frames, dtype=np.float64) * dt
    if ready:
        joint_pos = np.repeat(
            np.array([[-0.12, 0.08, 0.03]], dtype=np.float32), frames, axis=0
        )
        body_pos = np.zeros((frames, bodies, 3), dtype=np.float32)
        body_pos[:, 0, 2] = 1.0
        body_pos[:, 1, 0] = 0.25
        body_pos[:, 1, 2] = 1.2
    else:
        joint_pos = np.stack(
            [
                -0.05 + 0.18 * time + 0.03 * time**2,
                0.02 - 0.10 * time + 0.02 * time**2,
                0.04 + 0.07 * time,
            ],
            axis=1,
        ).astype(np.float32)
        body_pos = np.zeros((frames, bodies, 3), dtype=np.float32)
        body_pos[:, 0, 0] = (0.02 * time).astype(np.float32)
        body_pos[:, 0, 2] = 1.0
        body_pos[:, 1, 0] = (0.25 + 0.10 * time).astype(np.float32)
        body_pos[:, 1, 1] = (-0.03 * time**2).astype(np.float32)
        body_pos[:, 1, 2] = 1.2
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_quat = np.zeros((frames, bodies, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang = np.zeros((frames, bodies, 3), dtype=np.float32)
    arrays = {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "kinematics_schema_version": np.array([2], dtype=np.int64),
        "body_pos_point": np.array("link_origin"),
        "body_lin_vel_point": np.array("center_of_mass"),
        "body_names": np.array(["pelvis_link", "racket_link"]),
    }
    if migration_provenance:
        arrays.update(
            kinematics_migration_source_sha256=np.array("a" * 64),
            kinematics_migration_source_point=np.array("link_origin"),
            kinematics_migration_tool=np.array("migrate_motion_kinematics.py/v2"),
        )
    return arrays


def _write_motion(path: Path, *, frames: int = 40, ready: bool = False) -> dict[str, np.ndarray]:
    arrays = _motion_arrays(frames=frames, ready=ready)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return arrays


def _command(source: Path, ready: Path, output: Path, contract: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--ready-source",
        str(ready),
        "--ready-frame",
        "0",
        "--contact-frame",
        "30",
        "--join-frame",
        "16",
        "--hold-frames",
        "4",
        "--blend-intervals",
        "8",
        "--output-npz",
        str(output),
        "--output-contract",
        str(contract),
    ]


def test_quintic_boundary_conditions_are_c2_exact() -> None:
    p0 = np.array([0.1, -0.2])
    v0 = np.array([0.0, 0.0])
    a0 = np.array([0.0, 0.0])
    p1 = np.array([0.7, 0.3])
    v1 = np.array([0.4, -0.1])
    a1 = np.array([-0.2, 0.5])
    p, v, a = M.quintic_hermite(p0, v0, a0, p1, v1, a1, 0.2, [0.0, 0.2])
    np.testing.assert_allclose(p[0], p0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(v[0], v0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(a[0], a0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(p[-1], p1, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(v[-1], v1, rtol=0.0, atol=1.0e-11)
    np.testing.assert_allclose(a[-1], a1, rtol=0.0, atol=1.0e-10)


def test_canonical_migration_provenance_is_preserved_and_bound(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    source = _motion_arrays(frames=40, ready=False, migration_provenance=True)
    ready = _motion_arrays(frames=12, ready=True, migration_provenance=True)
    ready["kinematics_migration_source_sha256"] = np.array("b" * 64)
    ready["kinematics_migration_source_point"] = np.array("center_of_mass")
    with source_path.open("wb") as stream:
        np.savez(stream, **source)
    with ready_path.open("wb") as stream:
        np.savez(stream, **ready)

    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as archive:
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
    for key in M.SCHEMA2_MIGRATION_PROVENANCE_KEYS:
        assert np.array_equal(output[key], source[key])
        assert output[key].dtype == source[key].dtype
        assert output[key].shape == source[key].shape
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    proof = contract["proof"]
    assert proof["source_migration_provenance"] == {
        "kinematics_migration_source_sha256": "a" * 64,
        "kinematics_migration_source_point": "link_origin",
        "kinematics_migration_tool": "migrate_motion_kinematics.py/v2",
    }
    assert proof["ready_source_migration_provenance"][
        "kinematics_migration_source_sha256"
    ] == "b" * 64
    assert proof["source_migration_provenance_preserved"] is True
    assert proof["migration_provenance_validation"] == {
        "canonical_syntax_and_verbatim_lineage_only": True,
        "legacy_ancestor_bytes_rehashed": False,
    }
    assert "migration_legacy_ancestor_bytes_not_rehashed" in contract["explicit_non_claims"]


def test_ready_only_migration_provenance_does_not_become_output_lineage(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    source = _motion_arrays(frames=40, ready=False)
    ready = _motion_arrays(frames=12, ready=True, migration_provenance=True)
    with source_path.open("wb") as stream:
        np.savez(stream, **source)
    with ready_path.open("wb") as stream:
        np.savez(stream, **ready)

    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as archive:
        assert not (set(archive.files) & set(M.SCHEMA2_MIGRATION_PROVENANCE_KEYS))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["proof"]["source_migration_provenance"] is None
    assert contract["proof"]["ready_source_migration_provenance"] is not None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_sha", "partial migration provenance"),
        ("missing_point", "partial migration provenance"),
        ("missing_tool", "partial migration provenance"),
        ("unknown", "unexpected"),
        ("uppercase_sha", "lowercase SHA-256"),
        ("nonscalar_sha", "canonical unicode scalar string"),
        ("bytes_sha", "canonical unicode scalar string"),
        ("object_sha", "Object arrays cannot be loaded"),
        ("integer_sha", "canonical unicode scalar string"),
        ("bad_point", "must be link_origin or center_of_mass"),
        ("bad_tool", "kinematics_migration_tool must be"),
    ],
)
def test_noncanonical_migration_provenance_fails_closed_without_outputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    source = _motion_arrays(frames=40, ready=False, migration_provenance=True)
    if mutation == "missing_sha":
        del source["kinematics_migration_source_sha256"]
    elif mutation == "missing_point":
        del source["kinematics_migration_source_point"]
    elif mutation == "missing_tool":
        del source["kinematics_migration_tool"]
    elif mutation == "unknown":
        source["unexpected_provenance"] = np.array("not allowed")
    elif mutation == "uppercase_sha":
        source["kinematics_migration_source_sha256"] = np.array("A" * 64)
    elif mutation == "nonscalar_sha":
        source["kinematics_migration_source_sha256"] = np.array(["a" * 64])
    elif mutation == "bytes_sha":
        source["kinematics_migration_source_sha256"] = np.array(b"a" * 64)
    elif mutation == "object_sha":
        source["kinematics_migration_source_sha256"] = np.array("a" * 64, dtype=object)
    elif mutation == "integer_sha":
        source["kinematics_migration_source_sha256"] = np.array(7, dtype=np.int64)
    elif mutation == "bad_point":
        source["kinematics_migration_source_point"] = np.array("racket_origin")
    elif mutation == "bad_tool":
        source["kinematics_migration_tool"] = np.array("migrate_motion_kinematics.py/v3")
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    with source_path.open("wb") as stream:
        np.savez(stream, **source)
    _write_motion(ready_path, frames=12, ready=True)

    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert message in completed.stderr
    assert not output_path.exists() and not contract_path.exists()


def test_build_preserves_ready_contact_window_and_velocity_continuity(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.contract.json"
    source = _write_motion(source_path)
    ready = _write_motion(ready_path, frames=12, ready=True)

    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["training_authorized"] is False
    assert summary["contact_time_from_frame0_s"] == 0.5

    with np.load(output_path, allow_pickle=False) as archive:
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
    assert output["joint_pos"].shape[0] == 35
    assert np.array_equal(output["joint_pos"][0], ready["joint_pos"][0])
    assert np.array_equal(output["body_pos_w"][0], ready["body_pos_w"][0])
    assert np.array_equal(output["body_quat_w"][0], ready["body_quat_w"][0])
    assert np.array_equal(output["joint_vel"][:2], np.zeros_like(output["joint_vel"][:2]))
    assert np.array_equal(
        output["body_lin_vel_w"][:2], np.zeros_like(output["body_lin_vel_w"][:2])
    )

    # Source join frame 16 maps to output frame 11.  The quintic endpoint
    # matches source position/velocity, so there is no pose jump and only the
    # expected finite-grid derivative error at the discrete splice.
    output_join = 11
    assert np.array_equal(output["joint_pos"][output_join], source["joint_pos"][16])
    assert float(np.max(np.abs(output["joint_vel"][output_join] - source["joint_vel"][16]))) < 0.05
    assert float(
        np.max(np.abs(output["joint_pos"][output_join] - output["joint_pos"][output_join - 1]))
    ) < 0.02
    assert np.isfinite(output["joint_vel"][output_join - 1 : output_join + 2]).all()

    # Five 50 Hz frames before contact plus the contact row are byte-identical
    # for every schema-2 time channel.  No cropped or synthesized contact data
    # can pass this assertion.
    for key in M.SCHEMA2_TIME_KEYS:
        assert np.array_equal(output[key][20:26], source[key][25:31]), key
        assert np.array_equal(output[key][25], source[key][30]), key
    assert np.isfinite(output["joint_pos"]).all()
    assert np.isfinite(output["body_quat_w"]).all()
    expected_joint_vel = np.gradient(output["joint_pos"], 1.0 / 50.0, axis=0).astype(np.float32)
    assert np.array_equal(output["joint_vel"], expected_joint_vel)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["tool"]["sha256"] == _sha(SCRIPT)
    assert contract["tool"]["binding_semantics"] == (
        "source_file_snapshot_at_main_entry_unchanged_before_publish"
    )
    assert contract["output"]["npz"]["sha256"] == _sha(output_path)
    assert contract["proof"]["protected_window_bitwise_equal"] is True
    assert contract["proof"]["output_contact_frame"] == 25
    assert contract["synthesis"]["simple_crop"] is False
    assert contract["synthesis"]["production_fk_rebuild_required"] is True
    assert contract["authorization"] == {
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
    }


def test_publication_is_no_clobber_and_preserves_first_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    ready = tmp_path / "ready.npz"
    output = tmp_path / "candidate.npz"
    contract = tmp_path / "candidate.json"
    _write_motion(source)
    _write_motion(ready, frames=12, ready=True)
    command = _command(source, ready, output, contract)
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    before = output.read_bytes(), contract.read_bytes()
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert second.returncode == 2
    assert "no-clobber" in second.stderr
    assert (output.read_bytes(), contract.read_bytes()) == before


def test_symlinked_input_and_output_parent_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    ready = tmp_path / "ready.npz"
    _write_motion(source)
    _write_motion(ready, frames=12, ready=True)
    source_link = tmp_path / "source-link.npz"
    source_link.symlink_to(source)
    output = tmp_path / "candidate.npz"
    contract = tmp_path / "candidate.json"
    linked_input = subprocess.run(
        _command(source_link, ready, output, contract),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked_input.returncode == 2
    assert "symlink" in linked_input.stderr
    assert not output.exists() and not contract.exists()

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    linked_parent = subprocess.run(
        _command(source, ready, alias / "candidate.npz", alias / "candidate.json"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked_parent.returncode == 2
    assert "symlink" in linked_parent.stderr
    assert not (real_parent / "candidate.npz").exists()


def test_nonfinite_and_late_join_are_rejected_without_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    ready = tmp_path / "ready.npz"
    output = tmp_path / "candidate.npz"
    contract = tmp_path / "candidate.json"
    arrays = _motion_arrays(frames=40, ready=False)
    arrays["joint_pos"][4, 1] = np.nan
    arrays["joint_vel"] = np.gradient(arrays["joint_pos"], 1.0 / 50.0, axis=0).astype(np.float32)
    with source.open("wb") as stream:
        np.savez(stream, **arrays)
    _write_motion(ready, frames=12, ready=True)
    nonfinite = subprocess.run(
        _command(source, ready, output, contract),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert nonfinite.returncode == 2
    assert "NaN/Inf" in nonfinite.stderr
    assert not output.exists() and not contract.exists()

    _write_motion(source)
    command = _command(source, ready, output, contract)
    command[command.index("--join-frame") + 1] = "26"
    late = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert late.returncode == 2
    assert "protected window" in late.stderr
    assert not output.exists() and not contract.exists()


def test_ready_uses_frame0_pose_and_explicitly_zeros_source_velocity(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "moving-ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    _write_motion(source_path)
    _write_motion(ready_path, frames=12, ready=False)
    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as archive:
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
    ready = _motion_arrays(frames=12, ready=False)
    assert np.array_equal(output["joint_pos"][0], ready["joint_pos"][0])
    assert np.array_equal(output["joint_vel"][:3], np.zeros_like(output["joint_vel"][:3]))
    assert np.array_equal(
        output["body_lin_vel_w"][:4], np.zeros_like(output["body_lin_vel_w"][:4])
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["proof"]["ready_source_velocity_channels_ignored"] is True
    assert contract["proof"]["ready_velocity_definition"] == "explicit_bitwise_zero"


def test_nonzero_ready_frame_is_rejected(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    _write_motion(source_path)
    _write_motion(ready_path, frames=12, ready=True)
    command = _command(source_path, ready_path, output_path, contract_path)
    command[command.index("--ready-frame") + 1] = "1"
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 2
    assert "frame 0" in completed.stderr
    assert not output_path.exists() and not contract_path.exists()


def test_quaternion_sign_at_join_matches_bitwise_source_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    ready_path = tmp_path / "ready.npz"
    output_path = tmp_path / "candidate.npz"
    contract_path = tmp_path / "candidate.json"
    source = _motion_arrays(frames=40, ready=False)
    source["body_quat_w"][16:] *= -1.0
    with source_path.open("wb") as stream:
        np.savez(stream, **source)
    _write_motion(ready_path, frames=12, ready=True)

    completed = subprocess.run(
        _command(source_path, ready_path, output_path, contract_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as archive:
        output_quat = np.asarray(archive["body_quat_w"]).copy()
    output_join = 11
    assert np.array_equal(output_quat[output_join], source["body_quat_w"][16])
    assert np.all(np.sum(output_quat[output_join - 1] * output_quat[output_join], axis=-1) > 0.0)
