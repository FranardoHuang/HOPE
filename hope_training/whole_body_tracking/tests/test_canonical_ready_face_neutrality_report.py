"""Focused tests for the independent grounded-ready face report producer."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training/whole_body_tracking/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_grounded_ready as grounded  # noqa: E402
import canonical_motion_recipe as recipe  # noqa: E402
import canonical_ready_face_neutrality_report as face_report  # noqa: E402


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _savez(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    np.savez(stream, **values)
    path.write_bytes(stream.getvalue())


def _fixture(tmp_path: Path) -> dict[str, Any]:
    scripts = tmp_path / "hope_training/whole_body_tracking/scripts"
    scripts.mkdir(parents=True)
    producer = scripts / "canonical_ready_face_neutrality_report.py"
    producer.write_bytes(b"# independent exact vendor MuJoCo FK fixture\n")

    joint_names = tuple(grounded.RUNTIME_JOINT_NAMES)
    right_ids = np.asarray(
        [joint_names.index(name) for name in face_report.RIGHT_ARM_NAMES],
        dtype=np.int64,
    )
    joint_pos = np.linspace(-0.4, 0.4, 31, dtype=np.float64)
    root_pos = np.asarray([0.1, -0.2, 0.9], dtype=np.float64)
    root_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    ground = {
        "schema_version": 1,
        "artifact_class": "diagnostic_stationary_grounded_ready_candidate",
        "trust_scope": {
            "value_class": (
                "UNTRUSTED_DIAGNOSTIC_UNTIL_CONSTRUCTION_BOUND_PUBLICATION"
            )
        },
        "candidate_id": "G1",
        "verdict": "PASS_STATIC_GROUNDED_READY_CANDIDATE",
        "candidate": {
            "joint_pos": joint_pos.tolist(),
            "joint_pos_sha256": face_report._array_sha256(joint_pos),
            "joint_vel": np.zeros(31, np.float64).tolist(),
            "root_pos_w": root_pos.tolist(),
            "root_quat_wxyz": root_quat.tolist(),
            "root_lin_vel_w": np.zeros(3, np.float64).tolist(),
            "root_ang_vel_w": np.zeros(3, np.float64).tolist(),
            "state_sha256": "1" * 64,
            "zero_velocity_emitted": True,
        },
        "source": {
            "mode": "G1_donor_root_flat_feet_leg12_continuation",
            "root_bitwise_preserved": True,
            "nonleg_joint_values_bitwise_preserved": True,
            "upper_overlay": {
                "applied": True,
                "joint_names": list(face_report.RIGHT_ARM_NAMES),
                "joint_indices": right_ids.tolist(),
                "input_joint_pos_sha256": face_report._array_sha256(joint_pos),
                "copied_values_sha256": face_report._array_sha256(
                    joint_pos[right_ids]
                ),
                "root_preserved": True,
                "lower_preserved": True,
            },
        },
        "exact_model": {
            "exact_mujoco_backend": True,
            "status": "PASS_EXACT_MUJOCO",
            "mjcf_sha256": "a" * 64,
            "compiled_model_sha256": "b" * 64,
            "path_model_binding_sha256": "c" * 64,
            "ground_model_binding_sha256": "d" * 64,
            "joint_order": list(joint_names),
        },
        "foot_targets": {},
        "static_geometry": {},
        "static_ground_dynamics": {},
        "gates": {
            name: "PASS"
            for name in (
                "exact_model_identity",
                "joint_limits",
                "foot_pose",
                "leg_to_foot_jacobian",
                "double_support",
                "sole_floor",
                "collision",
                "support_margin",
                "static_ground_dynamics",
            )
        },
        "authorization": dict(face_report.FALSE_AUTHORIZATION),
        "selection": {
            "selected_as_canonical_ready": False,
            "automatic_G1_or_G2_adoption": False,
            "requires_outer_comparison_across_all_five_motions": True,
        },
        "non_claims": ["fixture is not authorized"],
        "config": {},
    }
    unsigned_ground = dict(ground)
    receipt_payload_sha = hashlib.sha256(_canonical(unsigned_ground)).hexdigest()
    ground["receipt_payload_sha256"] = receipt_payload_sha

    candidate = tmp_path / "input/grounded_ready_candidate_v1.npz"
    _savez(
        candidate,
        joint_pos=joint_pos,
        joint_vel=np.zeros(31, np.float64),
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=np.zeros(3, np.float64),
        root_ang_vel_w=np.zeros(3, np.float64),
        candidate_id=np.asarray("G1"),
        receipt_sha256=np.asarray(receipt_payload_sha),
        training_authorized=np.asarray(False),
        hardware_authorized=np.asarray(False),
    )
    ground["publication"] = {
        "candidate_filename": candidate.name,
        "candidate_npz_sha256": _sha(candidate),
        "receipt_filename": "RECEIPT.json",
        "completion_semantics": "exclusive_directory_and_receipt_written_last",
    }
    publication_unsigned = dict(ground)
    ground["publication_payload_sha256"] = hashlib.sha256(
        _canonical(publication_unsigned)
    ).hexdigest()
    ground_path = tmp_path / "input/RECEIPT.json"
    _write_json(ground_path, ground)

    ready = tmp_path / "input/canonical_ready_v2_g1_neutral_arm.npz"
    _savez(
        ready,
        joint_pos=joint_pos,
        joint_vel=np.zeros(31, np.float64),
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        source_segment=np.asarray("grounded_ready_v2_g1_neutral_arm"),
        source_npz=np.asarray("input/grounded_ready_candidate_v1.npz"),
        source_frame=np.asarray(0, np.int64),
        striking_joint_ids=right_ids,
        note=np.asarray("fixture"),
    )

    recipe_path = tmp_path / "configs/recipe.json"
    phase_path = tmp_path / "configs/phase.json"
    _write_json(recipe_path, {"fixture": "recipe"})
    _write_json(phase_path, {"fixture": "phase"})
    labels = [
        "{}:{}:{}".format(scope, phase, face_name)
        for scope in face_report.SCOPES
        for phase in face_report.PHASES
        for face_name in face_report.FACES
    ]
    proof_rows = []
    for index, label in enumerate(labels):
        normal = (
            np.asarray([1.0, 0.0, 0.0], np.float64)
            if label.endswith(":bh")
            else np.asarray([-1.0, 0.0, 0.0], np.float64)
        )
        proof_rows.append(
            {
                "label": label,
                "source_frame_index": index,
                "pose_content_sha256": "{:064x}".format(index + 1),
                "pair_contract_sha256": "{:064x}".format(index + 101),
                "joint_pos_sha256": "{:064x}".format(index + 201),
                "root_pos_w_sha256": "{:064x}".format(index + 301),
                "root_quat_w_sha256": "{:064x}".format(index + 401),
                "site_pos_w_sha256": "{:064x}".format(index + 501),
                "site_rotation_w_sha256": "{:064x}".format(index + 601),
                "signed_face_normal_w_sha256": (
                    face_report._neutral_array_sha256(normal)
                ),
            }
        )
    lineage = {
        "schema_version": 1,
        "builder_contract": "canonical_block_lineage_pose_reconstruction_v2",
        "file_bindings": [
            {
                "role": "recipe",
                "path": str(recipe_path),
                "sha256": _sha(recipe_path),
            },
            {
                "role": "phase_authority",
                "path": str(phase_path),
                "sha256": _sha(phase_path),
            },
        ],
        "contact_matrix_sha256": "e" * 64,
        "contact_row_count": 16,
        "phase_source_frames": {},
        "scope_contract": {},
        "face_contract": {},
        "model_binding": {
            "mjcf_sha256": "a" * 64,
            "compiled_model_sha256": "b" * 64,
            "backend_limits_sha256": "f" * 64,
            "backend_model_contract_sha256": "9" * 64,
            "urdf_sha256": "8" * 64,
            "normal_convention": face_report.FACE_NORMAL_CONVENTION,
            "racket_site_name": face_report.RACKET_SITE,
        },
        "rows": proof_rows,
    }
    lineage_path = tmp_path / "input/EXACT_16_ROW_LINEAGE_POSE_RECEIPT.json"
    _write_json(lineage_path, lineage)
    target_rows = []
    for index, label in enumerate(labels):
        normal = (
            np.asarray([1.0, 0.0, 0.0], np.float64)
            if label.endswith(":bh")
            else np.asarray([-1.0, 0.0, 0.0], np.float64)
        )
        target_rows.append(
            {
                "label": label,
                "signed_face_normal_w": normal.tolist(),
                "source_frame_index": index,
            }
        )
    required_pass = (
        "all_global_targets_exact_ik",
        "exact_model",
        "exact_site_normal_ik",
        "finite_global_optimizer_locus",
        "fixed_joints",
        "global_angular_minimax_bound",
        "input_contact_exact_fk",
        "joint_limits",
        "neutrality_threshold",
        "paired_face_and_site_content_contract",
        "source_ready_hash",
        "upstream_source_pose_reconstruction",
    )
    challenger = {
        "schema_version": 1,
        "artifact_class": "diagnostic_face_neutral_ready_candidate",
        "verdict": "INCOMPLETE_FAIL_CLOSED",
        "authorization": {
            "training_authorized": False,
            "deploy_authorized": False,
            "hardware_authorized": False,
        },
        "gates": {key: "PASS" for key in required_pass},
        "joint_contract": {
            "active_joint_indices": right_ids.tolist(),
            "active_joint_names": list(face_report.RIGHT_ARM_NAMES),
            "all_joint_names": list(joint_names),
        },
        "candidate": {
            "joint_pos": joint_pos.tolist(),
            "joint_pos_sha256": face_report._neutral_array_sha256(joint_pos),
            "root_pos_w": root_pos.tolist(),
            "root_quat_w": root_quat.tolist(),
        },
        "model": {
            "mjcf_sha256": "a" * 64,
            "compiled_model_sha256": "b" * 64,
            "backend_limits_sha256": "f" * 64,
            "backend_model_contract_sha256": "9" * 64,
        },
        "contact_matrix": {
            "input_sha256": "e" * 64,
            "row_count": 16,
            "rows": target_rows,
        },
    }
    challenger_path = (
        tmp_path / "input/UNPUBLISHED_RIGHT_ARM_CHALLENGER_RECEIPT.json"
    )
    _write_json(challenger_path, challenger)
    return {
        "tmp_path": tmp_path,
        "producer": producer,
        "ready": ready,
        "candidate": candidate,
        "ground": ground_path,
        "lineage": lineage_path,
        "challenger": challenger_path,
        "recipe": recipe_path,
        "phase": phase_path,
        "joint_pos": joint_pos,
        "right_ids": right_ids,
        "labels": labels,
        "proof_rows": proof_rows,
    }


def _validate(files: dict[str, Any]) -> face_report.ValidatedInputs:
    return face_report.validate_inputs(
        repo_root=files["tmp_path"],
        ready_path=files["ready"],
        candidate_path=files["candidate"],
        ground_receipt_path=files["ground"],
        lineage_receipt_path=files["lineage"],
        challenger_receipt_path=files["challenger"],
        recipe_path=files["recipe"],
        phase_authority_path=files["phase"],
    )


def _evaluation(files: dict[str, Any]) -> face_report.ExactEvaluation:
    rows = []
    for index, label in enumerate(files["labels"]):
        scope, phase, face_name = label.split(":")
        normal = (
            np.asarray([1.0, 0.0, 0.0], np.float64)
            if face_name == "bh"
            else np.asarray([-1.0, 0.0, 0.0], np.float64)
        )
        rows.append(
            face_report.TargetRow(
                scope=scope,
                phase=phase,
                face=face_name,
                normal_w=normal,
                target_sha256=face_report._neutral_array_sha256(normal),
                source_frame_index=index,
                pose_content_sha256=files["proof_rows"][index][
                    "pose_content_sha256"
                ],
            )
        )
    return face_report.ExactEvaluation(
        mjcf_sha256="a" * 64,
        compiled_model_sha256="b" * 64,
        ready_normal_w=np.asarray([0.0, 0.0, 1.0], np.float64),
        rows=tuple(rows),
    )


def test_matching_chain_publishes_recipe_compatible_report_last(
    tmp_path, monkeypatch
):
    files = _fixture(tmp_path)
    validated = _validate(files)
    (tmp_path / "evidence").mkdir()
    writes = []
    original_write = face_report._exclusive_write_at

    def record_write(directory_fd, filename, payload):
        writes.append(filename)
        original_write(directory_fd, filename, payload)

    monkeypatch.setattr(face_report, "_exclusive_write_at", record_write)
    result = face_report.publish(
        validated,
        _evaluation(files),
        tmp_path / "evidence/output",
        producer_path=files["producer"],
    )

    assert sorted(path.name for path in result.directory.iterdir()) == [
        face_report.REPORT_FILENAME,
        face_report.TARGET_SET_FILENAME,
    ]
    assert writes == [
        face_report.TARGET_SET_FILENAME,
        face_report.REPORT_FILENAME,
    ]
    raw = json.loads(result.report.read_text("ascii"))
    seal = raw.pop("report_payload_sha256")
    assert seal == hashlib.sha256(_canonical(raw)).hexdigest()
    assert raw["verdict"] == "PASS_FACE_NEUTRAL_READY"
    assert raw["authorization"] == face_report.FALSE_AUTHORIZATION
    assert raw["evaluation"]["maximum_pair_asymmetry_rad"] == pytest.approx(0.0)
    assert raw["evaluation"]["maximum_allowed_pair_asymmetry_rad"] == pytest.approx(
        np.deg2rad(5.0)
    )

    ready = recipe._load_ready(files["ready"], _sha(files["ready"]))
    recipe._validate_face_neutrality_report(
        json.loads(result.report.read_text("ascii")),
        repo_root=tmp_path,
        ready=ready,
        grounded_exact_model={
            "mjcf_sha256": "a" * 64,
            "compiled_model_sha256": "b" * 64,
        },
    )


def test_exact_replay_uses_fresh_backend_rows_not_old_distances(
    tmp_path, monkeypatch
):
    files = _fixture(tmp_path)
    validated = _validate(files)

    import canonical_face_manifold as manifold
    import canonical_mujoco_dynamics_gate as dynamics
    import canonical_neutral_ready_cli as adapter

    mjcf = tmp_path / "agi/model.xml"
    urdf = tmp_path / "agi/robot.urdf"
    mjcf.parent.mkdir(parents=True)
    mjcf.write_bytes(b"<mujoco/>")
    urdf.write_bytes(b"<robot/>")
    fake_recipe = SimpleNamespace(
        model_paths={"mjcf": mjcf, "urdf": urdf}
    )

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.model = object()

        def site_pose(self, joint_pos, root_pos, root_quat):
            assert np.array_equal(joint_pos, validated.state.joint_pos)
            assert np.array_equal(root_pos, validated.state.root_pos_w)
            assert np.array_equal(root_quat, validated.state.root_quat_wxyz)
            rotation = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                np.float64,
            )
            return np.zeros(3, np.float64), rotation

    contacts = []
    for index, label in enumerate(files["labels"]):
        scope, phase, face_name = label.split(":")
        normal = (
            np.asarray([1.0, 0.0, 0.0], np.float64)
            if face_name == "bh"
            else np.asarray([-1.0, 0.0, 0.0], np.float64)
        )
        contacts.append(
            SimpleNamespace(
                scope=scope,
                phase=phase,
                face_name=face_name,
                signed_face_normal_w=normal,
                source_frame_index=index,
                pose_content_sha256=files["proof_rows"][index][
                    "pose_content_sha256"
                ],
            )
        )
    loaded = SimpleNamespace(
        contacts=tuple(contacts),
        contact_source_proof=SimpleNamespace(
            receipt={"rows": files["proof_rows"]}
        ),
    )
    monkeypatch.setattr(manifold, "MujocoRightRacketBackend", FakeBackend)
    monkeypatch.setattr(
        dynamics, "compiled_model_signature", lambda model: "b" * 64
    )
    monkeypatch.setattr(
        adapter,
        "_snapshot_recipe_inputs",
        lambda *args, **kwargs: (fake_recipe, {}),
    )
    monkeypatch.setattr(
        adapter,
        "load_block_phase_map_binding",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        adapter,
        "load_real_neutral_ready_inputs",
        lambda *args, **kwargs: loaded,
    )

    exact = face_report.recompute_exact_evidence(validated)
    assert np.array_equal(exact.ready_normal_w, [0.0, 0.0, 1.0])
    assert len(exact.rows) == 16
    assert exact.rows[0].target_sha256 == files["proof_rows"][0][
        "signed_face_normal_w_sha256"
    ]


def test_grounded_overlay_must_match_supplied_challenger_exactly(tmp_path):
    files = _fixture(tmp_path)
    challenger = json.loads(files["challenger"].read_text("ascii"))
    challenger["candidate"]["joint_pos"][files["right_ids"][0]] += 0.01
    changed = np.asarray(challenger["candidate"]["joint_pos"], np.float64)
    challenger["candidate"]["joint_pos_sha256"] = (
        face_report._neutral_array_sha256(changed)
    )
    _write_json(files["challenger"], challenger)

    with pytest.raises(
        face_report.FaceNeutralityError,
        match="overlay input digest|right-arm differs",
    ):
        _validate(files)


def test_nonleg_or_root_mutation_is_rejected(tmp_path):
    files = _fixture(tmp_path)
    validated = _validate(files)
    challenger = dict(validated.challenger)
    challenger["candidate"] = dict(challenger["candidate"])
    challenger["candidate"]["joint_pos"] = list(
        challenger["candidate"]["joint_pos"]
    )
    # Index 2 is waist_yaw: non-leg and outside the seven-joint arm.
    challenger["candidate"]["joint_pos"][2] += 0.01
    changed = np.asarray(challenger["candidate"]["joint_pos"], np.float64)
    challenger["candidate"]["joint_pos_sha256"] = (
        face_report._neutral_array_sha256(changed)
    )
    ground = dict(validated.ground)
    ground["source"] = dict(ground["source"])
    ground["source"]["upper_overlay"] = dict(
        ground["source"]["upper_overlay"]
    )
    ground["source"]["upper_overlay"]["input_joint_pos_sha256"] = (
        face_report._array_sha256(changed)
    )

    with pytest.raises(
        face_report.FaceNeutralityError, match="non-leg joint"
    ):
        face_report._critical_challenger_contract(
            challenger, ground, validated.state
        )


def test_pair_asymmetry_over_five_degrees_fails_before_publication(tmp_path):
    files = _fixture(tmp_path)
    validated = _validate(files)
    evaluation = _evaluation(files)
    rows = list(evaluation.rows)
    first = rows[1]
    rows[1] = face_report.TargetRow(
        scope=first.scope,
        phase=first.phase,
        face=first.face,
        normal_w=np.asarray([0.0, 0.0, -1.0], np.float64),
        target_sha256=face_report._neutral_array_sha256(
            np.asarray([0.0, 0.0, -1.0], np.float64)
        ),
        source_frame_index=first.source_frame_index,
        pose_content_sha256=first.pose_content_sha256,
    )
    bad = face_report.ExactEvaluation(
        mjcf_sha256=evaluation.mjcf_sha256,
        compiled_model_sha256=evaluation.compiled_model_sha256,
        ready_normal_w=evaluation.ready_normal_w,
        rows=tuple(rows),
    )
    (tmp_path / "evidence").mkdir()
    output = tmp_path / "evidence/bad"
    with pytest.raises(
        face_report.FaceNeutralityError, match="exceeds 5 degrees"
    ):
        face_report.publish(
            validated, bad, output, producer_path=files["producer"]
        )
    assert not output.exists()


def test_no_clobber_preserves_existing_directory(tmp_path):
    files = _fixture(tmp_path)
    validated = _validate(files)
    output = tmp_path / "evidence/existing"
    output.mkdir(parents=True)
    sentinel = output / "SENTINEL"
    sentinel.write_bytes(b"do-not-touch")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        face_report.publish(
            validated,
            _evaluation(files),
            output,
            producer_path=files["producer"],
        )
    assert sentinel.read_bytes() == b"do-not-touch"
    assert sorted(path.name for path in output.iterdir()) == ["SENTINEL"]


def test_target_digest_or_authorization_tamper_is_rejected(tmp_path):
    files = _fixture(tmp_path)
    validated = _validate(files)
    evaluation = _evaluation(files)
    rows = list(evaluation.rows)
    row = rows[0]
    rows[0] = face_report.TargetRow(
        scope=row.scope,
        phase=row.phase,
        face=row.face,
        normal_w=row.normal_w,
        target_sha256="0" * 64,
        source_frame_index=row.source_frame_index,
        pose_content_sha256=row.pose_content_sha256,
    )
    with pytest.raises(
        face_report.FaceNeutralityError, match="row contract"
    ):
        face_report._validate_evaluation(
            validated,
            face_report.ExactEvaluation(
                evaluation.mjcf_sha256,
                evaluation.compiled_model_sha256,
                evaluation.ready_normal_w,
                tuple(rows),
            ),
        )

    challenger = json.loads(files["challenger"].read_text("ascii"))
    challenger["authorization"]["training_authorized"] = True
    _write_json(files["challenger"], challenger)
    with pytest.raises(
        face_report.FaceNeutralityError, match="must deny training"
    ):
        _validate(files)
