"""Host-only tests for the diagnostic canonical-ready sidecar mint."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_motion_recipe as recipe  # noqa: E402
import canonical_ready_sidecar_mint as mint  # noqa: E402


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash_array(label: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return (
        label.encode("utf-8")
        + str(array.dtype).encode("ascii")
        + np.asarray(array.shape, np.int64).tobytes()
        + array.tobytes()
    )


def _array_sha(value: np.ndarray) -> str:
    return _sha256(_hash_array("array", value))


def _state_sha(q: np.ndarray, root: np.ndarray, quat: np.ndarray) -> str:
    return _sha256(
        _hash_array("joint_pos", q)
        + _hash_array("root_pos_w", root)
        + _hash_array("root_quat_wxyz", quat)
    )


def _base_receipt(
    q: np.ndarray,
    root: np.ndarray,
    quat: np.ndarray,
) -> dict[str, object]:
    zeros31 = np.zeros(31, np.float64)
    zeros3 = np.zeros(3, np.float64)
    strike_ids = np.asarray(
        [mint.RUNTIME_JOINT_NAMES.index(name) for name in mint.RIGHT_STRIKE_CHAIN],
        dtype=np.int64,
    )
    digest = "a" * 64
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_class": "diagnostic_stationary_grounded_ready_candidate",
        "trust_scope": copy.deepcopy(mint._TRUST_SCOPE),
        "candidate_id": "G1",
        "verdict": "PASS_STATIC_GROUNDED_READY_CANDIDATE",
        "candidate": {
            "state_sha256": _state_sha(q, root, quat),
            "joint_pos_sha256": _array_sha(q),
            "joint_pos": q.tolist(),
            "joint_vel": zeros31.tolist(),
            "root_pos_w": root.tolist(),
            "root_quat_wxyz": quat.tolist(),
            "root_lin_vel_w": zeros3.tolist(),
            "root_ang_vel_w": zeros3.tolist(),
            "zero_velocity_emitted": True,
        },
        "source": {
            "mode": mint._EXPECTED_SOURCE_MODE,
            "donor_state_sha256": digest,
            "seed_state_sha256": "b" * 64,
            "upper_overlay": {
                "applied": True,
                "joint_names": list(mint.RIGHT_STRIKE_CHAIN),
                "joint_indices": strike_ids.tolist(),
                "input_joint_pos_sha256": "c" * 64,
                "copied_values_sha256": _array_sha(q[strike_ids]),
                "root_preserved": True,
                "lower_preserved": True,
            },
            "root_bitwise_preserved": True,
            "nonleg_joint_values_bitwise_preserved": True,
            "target_semantics": "fixture",
            "target_contact_preload_m": 0.0005,
            "solver_trace": [],
        },
        "exact_model": {
            "backend_type": "canonical_grounded_ready.MujocoGroundedReadyBackend",
            "compiled_model_sha256": digest,
            "exact_mujoco_backend": True,
            "ground_model_binding_sha256": "b" * 64,
            "joint_order": list(mint.RUNTIME_JOINT_NAMES),
            "joint_position_lower_sha256": "c" * 64,
            "joint_position_upper_sha256": "d" * 64,
            "mjcf_path": "/fixture/a3.xml",
            "mjcf_sha256": "e" * 64,
            "path_model_binding_sha256": "f" * 64,
            "status": "PASS_EXACT_MUJOCO",
            "xml_model_name": "fixture",
        },
        "foot_targets": {"fixture": True},
        "static_geometry": {"passed": True},
        "static_ground_dynamics": {
            "feasible": True,
            "status": "PASS_STATIC_GROUND_CONTACT_LP",
        },
        "gates": {key: "PASS" for key in mint._GATE_KEYS},
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "selection": {
            "selected_as_canonical_ready": False,
            "automatic_G1_or_G2_adoption": False,
            "requires_outer_comparison_across_all_five_motions": True,
        },
        "non_claims": ["fixture remains diagnostic"],
        "config": {"fixture": True},
    }
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    return receipt


def _publish_fixture(
    repo: Path,
    *,
    mutation=None,
    candidate_mutation=None,
) -> tuple[Path, str, Path, str]:
    source = repo / "input"
    source.mkdir(parents=True)
    q = np.linspace(-0.7, 0.8, 31, dtype=np.float64)
    root = np.asarray([0.1, -0.2, 0.9], dtype=np.float64)
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    receipt = _base_receipt(q, root, quat)
    if mutation is not None:
        mutation(receipt)
        receipt.pop("receipt_payload_sha256", None)
        receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    payload_seal = str(receipt["receipt_payload_sha256"])

    candidate_values = {
        "joint_pos": q,
        "joint_vel": np.zeros(31, np.float64),
        "root_pos_w": root,
        "root_quat_w": quat,
        "root_lin_vel_w": np.zeros(3, np.float64),
        "root_ang_vel_w": np.zeros(3, np.float64),
        "candidate_id": np.asarray("G1"),
        "receipt_sha256": np.asarray(payload_seal),
        "training_authorized": np.asarray(False, np.bool_),
        "hardware_authorized": np.asarray(False, np.bool_),
    }
    if candidate_mutation is not None:
        candidate_mutation(candidate_values)
    buffer = io.BytesIO()
    np.savez(buffer, **candidate_values)
    candidate_bytes = buffer.getvalue()
    candidate = source / "grounded_ready_candidate_v1.npz"
    candidate.write_bytes(candidate_bytes)

    receipt["publication"] = {
        "candidate_filename": candidate.name,
        "candidate_npz_sha256": _sha256(candidate_bytes),
        "receipt_filename": "RECEIPT.json",
        "completion_semantics": "exclusive_directory_and_receipt_written_last",
    }
    receipt["publication_payload_sha256"] = _sha256(_canonical(receipt))
    receipt_path = source / "RECEIPT.json"
    receipt_bytes = _canonical(receipt)
    receipt_path.write_bytes(receipt_bytes)
    return (
        candidate,
        _sha256(candidate_bytes),
        receipt_path,
        _sha256(receipt_bytes),
    )


def _validate(repo: Path, fixture):
    candidate, candidate_sha, receipt, receipt_sha = fixture
    return mint.validate_ready_candidate_bundle(
        repo_root=repo,
        candidate_path=candidate,
        expected_candidate_sha256=candidate_sha,
        receipt_path=receipt,
        expected_receipt_sha256=receipt_sha,
    )


def _write_sealed_json(path: Path, payload: dict, seal_field: str) -> tuple[str, str]:
    value = copy.deepcopy(payload)
    value.pop(seal_field, None)
    payload_sha = _sha256(_canonical(value))
    value[seal_field] = payload_sha
    encoded = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return _sha256(encoded), payload_sha


def _grounded_ready_recipe_fixture(tmp_path: Path):
    fixture = _publish_fixture(tmp_path)
    validated = _validate(tmp_path, fixture)
    published = mint.mint_ready_sidecar(validated, tmp_path / "published")
    ready_repo_path = published.ready_npz.relative_to(tmp_path).as_posix()
    ready_state_sha = mint._state_sha256(
        validated.joint_pos,
        validated.root_pos_w,
        validated.root_quat_wxyz,
    )

    face_tool = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "independent_face_ready_audit.py"
    )
    face_tool.parent.mkdir(parents=True)
    face_tool.write_bytes(b"independent exact-FK face audit fixture\n")
    grounded_model = validated.receipt["exact_model"]
    upper_bh, upper_fh = 1.2, 1.21
    full_bh, full_fh = 1.1, 1.109
    upper_asymmetry = abs(upper_bh - upper_fh)
    full_asymmetry = abs(full_bh - full_fh)
    target_set = tmp_path / "evidence" / "FACE_TARGET_SET.json"
    target_set.parent.mkdir(parents=True)
    target_set.write_bytes(b'{"fixture":"16 exact FK target identities"}')
    face_report = {
        "schema_version": 1,
        "report_type": "canonical-ready-face-neutrality-v1",
        "artifact_class": "independent_exact_fk_face_neutrality_evidence",
        "producer": {
            "tool_path": face_tool.relative_to(tmp_path).as_posix(),
            "tool_sha256": _sha256(face_tool.read_bytes()),
            "independent_from_ready_minter": True,
            "backend": "exact_vendor_mujoco_fk",
        },
        "ready": {
            "path": ready_repo_path,
            "sha256": published.ready_npz_sha256,
            "state_sha256": ready_state_sha,
        },
        "model": {
            "mjcf_sha256": grounded_model["mjcf_sha256"],
            "compiled_model_sha256": grounded_model["compiled_model_sha256"],
            "racket_site": "right_racket",
            "face_normal_convention": (
                "right_racket_site_local_plus_y_world_signed_face_normal_v1"
            ),
        },
        "evaluation": {
            "scopes": ["upper", "full"],
            "phases": list(recipe._READY_FACE_PHASES),
            "faces": ["bh", "fh"],
            "target_set_path": target_set.relative_to(tmp_path).as_posix(),
            "target_set_sha256": _sha256(target_set.read_bytes()),
            "rows": [
                {
                    "scope": scope,
                    "phase": phase,
                    "bh_target_sha256": _sha256(f"{scope}:{phase}:bh".encode("ascii")),
                    "fh_target_sha256": _sha256(f"{scope}:{phase}:fh".encode("ascii")),
                    "bh_distance_rad": (upper_bh if scope == "upper" else full_bh),
                    "fh_distance_rad": (upper_fh if scope == "upper" else full_fh),
                    "absolute_asymmetry_rad": (
                        upper_asymmetry if scope == "upper" else full_asymmetry
                    ),
                }
                for scope in ("upper", "full")
                for phase in recipe._READY_FACE_PHASES
            ],
            "maximum_pair_asymmetry_rad": max(upper_asymmetry, full_asymmetry),
            "maximum_allowed_pair_asymmetry_rad": np.deg2rad(5.0).item(),
            "all_rows_exact_fk": True,
        },
        "verdict": "PASS_FACE_NEUTRAL_READY",
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": ["not training, deployment, or hardware authorization"],
    }
    face_path = tmp_path / "evidence" / "FACE_NEUTRALITY_REPORT.json"
    face_file_sha, face_payload_sha = _write_sealed_json(
        face_path, face_report, "report_payload_sha256"
    )

    minter_report = json.loads(published.identity_report_json.read_text("ascii"))
    minter_payload_sha = str(minter_report["report_payload_sha256"])
    adoption = {
        "schema_version": 1,
        "evidence_type": "canonical-ready-human-adoption-v1",
        "ready": {
            "path": ready_repo_path,
            "sha256": published.ready_npz_sha256,
            "state_sha256": ready_state_sha,
        },
        "evidence_bindings": {
            "candidate_sha256": validated.candidate_sha256,
            "grounded_receipt_sha256": validated.receipt_file_sha256,
            "grounded_receipt_payload_sha256": (validated.receipt_payload_sha256),
            "minter_identity_report_sha256": (published.identity_report_json_sha256),
            "minter_identity_report_payload_sha256": minter_payload_sha,
            "face_neutrality_report_sha256": face_file_sha,
            "face_neutrality_report_payload_sha256": face_payload_sha,
        },
        "decision": {
            "selected_as_canonical_ready": True,
            "decision_scope": ("canonical_ready_identity_for_compiler_candidate_only"),
            "decision_maker_kind": "human",
            "decision_maker": "Franco",
            "decision_recorded_at_utc": "2026-07-29T12:00:00Z",
            "rationale": (
                "Select the content-bound neutral grounded state for compiler "
                "candidate generation only."
            ),
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "selection is not training, deployment, or hardware authorization"
        ],
    }
    adoption_path = tmp_path / "evidence" / "HUMAN_ADOPTION.json"
    adoption_file_sha, adoption_payload_sha = _write_sealed_json(
        adoption_path, adoption, "evidence_payload_sha256"
    )

    candidate, candidate_sha, receipt, receipt_sha = fixture
    contract = {
        "path": ready_repo_path,
        "sha256": published.ready_npz_sha256,
        "provenance_mode": "selected_static_grounded_ready_identity_v1",
        "candidate": {
            "path": candidate.relative_to(tmp_path).as_posix(),
            "sha256": candidate_sha,
        },
        "grounded_receipt": {
            "path": receipt.relative_to(tmp_path).as_posix(),
            "sha256": receipt_sha,
            "payload_sha256": validated.receipt_payload_sha256,
        },
        "minter_identity_report": {
            "path": published.identity_report_json.relative_to(tmp_path).as_posix(),
            "sha256": published.identity_report_json_sha256,
            "payload_sha256": minter_payload_sha,
        },
        "face_neutrality_report": {
            "path": face_path.relative_to(tmp_path).as_posix(),
            "sha256": face_file_sha,
            "payload_sha256": face_payload_sha,
        },
        "human_adoption_evidence": {
            "path": adoption_path.relative_to(tmp_path).as_posix(),
            "sha256": adoption_file_sha,
            "payload_sha256": adoption_payload_sha,
        },
        "endpoint_velocity_policy": ("all_joint_root_body_velocities_exact_zero"),
    }
    return contract, {
        "validated": validated,
        "published": published,
        "face_path": face_path,
        "adoption_path": adoption_path,
    }


def test_mints_strict_nine_key_sidecar_and_separate_identity_reports(tmp_path):
    validated = _validate(tmp_path, _publish_fixture(tmp_path))
    published = mint.mint_ready_sidecar(validated, tmp_path / "published")

    assert published.ready_npz_sha256 == _sha256(published.ready_npz.read_bytes())
    loaded = recipe._load_ready(published.ready_npz, published.ready_npz_sha256)
    assert loaded.source_segment == mint.SOURCE_SEGMENT
    assert loaded.source_frame == 0
    assert np.array_equal(loaded.joint_pos, validated.joint_pos)
    assert np.array_equal(loaded.joint_vel, np.zeros(31, np.float64))
    with np.load(published.ready_npz, allow_pickle=False) as ready:
        assert frozenset(ready.files) == mint._READY_KEYS
        assert (
            str(ready["source_npz"].item()) == "input/grounded_ready_candidate_v1.npz"
        )
        assert "not donor-frame exact" in str(ready["note"].item())

    report = json.loads(published.identity_report_json.read_text("ascii"))
    seal = report.pop("report_payload_sha256")
    assert seal == _sha256(_canonical(report))
    assert report["ground_identity"]["status"] == (
        "PASS_BOUND_UPSTREAM_G1_STATIC_GROUND_RECEIPT"
    )
    assert report["ground_identity"]["physics_rerun_by_this_tool"] is False
    assert report["face_identity"] == {
        "status": "NOT_PROVEN_BY_GROUNDED_READY_RECEIPT",
        "face_neutrality_proven": False,
        "external_face_identity_report_required": True,
        "claim_scope": (
            "right-arm overlay bytes are identified; face FK/neutrality is not"
        ),
    }
    assert report["upstream_selection"]["selected_as_canonical_ready"] is False
    assert report["recipe_compatibility"]["legacy_donor_frame_exact_contract"] is False
    assert report["authorization"]["training_authorized"] is False


def test_rejects_explicit_file_hash_mismatch(tmp_path):
    fixture = _publish_fixture(tmp_path)
    candidate, _, receipt, receipt_sha = fixture
    with pytest.raises(mint.ReadySidecarMintError, match="candidate SHA-256 mismatch"):
        mint.validate_ready_candidate_bundle(
            repo_root=tmp_path,
            candidate_path=candidate,
            expected_candidate_sha256="0" * 64,
            receipt_path=receipt,
            expected_receipt_sha256=receipt_sha,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda receipt: receipt["gates"].__setitem__("collision", "FAIL_CLOSED"),
            "every exact G1 static-ground gate",
        ),
        (
            lambda receipt: receipt["authorization"].__setitem__(
                "training_authorized", True
            ),
            "deny training",
        ),
        (
            lambda receipt: receipt["selection"].__setitem__(
                "selected_as_canonical_ready", True
            ),
            "original non-adoption",
        ),
        (
            lambda receipt: receipt["source"]["upper_overlay"].__setitem__(
                "applied", False
            ),
            "expected G1 plus seven-joint",
        ),
    ],
)
def test_rejects_resealed_semantic_forgery(tmp_path, mutation, message):
    with pytest.raises(mint.ReadySidecarMintError, match=message):
        _validate(tmp_path, _publish_fixture(tmp_path, mutation=mutation))


def test_rejects_candidate_receipt_cross_sha_mismatch(tmp_path):
    def mutate(values):
        values["receipt_sha256"] = np.asarray("f" * 64)

    with pytest.raises(mint.ReadySidecarMintError, match="embedded receipt SHA"):
        _validate(
            tmp_path,
            _publish_fixture(tmp_path, candidate_mutation=mutate),
        )


def test_rejects_nonzero_candidate_velocity(tmp_path):
    def mutate(values):
        velocity = np.zeros(31, np.float64)
        velocity[7] = 1.0e-12
        values["joint_vel"] = velocity

    with pytest.raises(
        mint.ReadySidecarMintError, match="joint_vel must be exact zero"
    ):
        _validate(
            tmp_path,
            _publish_fixture(tmp_path, candidate_mutation=mutate),
        )


def test_no_clobber_leaves_first_bundle_unchanged(tmp_path):
    validated = _validate(tmp_path, _publish_fixture(tmp_path))
    output = tmp_path / "published"
    first = mint.mint_ready_sidecar(validated, output)
    ready_before = first.ready_npz.read_bytes()
    report_before = first.identity_report_json.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        mint.mint_ready_sidecar(validated, output)
    assert first.ready_npz.read_bytes() == ready_before
    assert first.identity_report_json.read_bytes() == report_before


def test_cli_is_fail_closed_and_prints_diagnostic_non_authorization(tmp_path, capsys):
    candidate, candidate_sha, receipt, receipt_sha = _publish_fixture(tmp_path)
    output = tmp_path / "published"
    status = mint.main(
        [
            "--repo-root",
            str(tmp_path),
            "--candidate",
            str(candidate),
            "--candidate-sha256",
            candidate_sha,
            "--receipt",
            str(receipt),
            "--receipt-sha256",
            receipt_sha,
            "--output-dir",
            str(output),
        ]
    )
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["verdict"] == "PASS_DIAGNOSTIC_IDENTITY_MINT"
    assert summary["training_authorized"] is False
    assert summary["hardware_authorized"] is False

    status = mint.main(
        [
            "--repo-root",
            str(tmp_path),
            "--candidate",
            str(candidate),
            "--candidate-sha256",
            candidate_sha,
            "--receipt",
            str(receipt),
            "--receipt-sha256",
            receipt_sha,
            "--output-dir",
            str(output),
        ]
    )
    assert status == 2
    assert "FAIL_CLOSED" in capsys.readouterr().err


def test_recipe_loads_selected_grounded_ready_evidence_chain(tmp_path):
    contract, fixture = _grounded_ready_recipe_fixture(tmp_path)
    ready, provenance, normalized = recipe._load_canonical_ready_contract(
        tmp_path, contract
    )
    assert normalized == contract
    assert ready.sha256 == fixture["published"].ready_npz_sha256
    assert ready.source_segment == mint.SOURCE_SEGMENT
    assert provenance.mode == "selected_static_grounded_ready_identity_v1"
    assert provenance.candidate_sha256 == fixture["validated"].candidate_sha256
    assert provenance.grounded_mjcf_sha256 == (
        fixture["validated"].receipt["exact_model"]["mjcf_sha256"]
    )
    assert set(contract) == recipe._GROUNDED_READY_RECIPE_KEYS
    assert set(contract["candidate"]) == recipe._BOUND_FILE_KEYS
    for key in (
        "grounded_receipt",
        "minter_identity_report",
        "face_neutrality_report",
        "human_adoption_evidence",
    ):
        assert set(contract[key]) == recipe._BOUND_PAYLOAD_FILE_KEYS


def test_recipe_preserves_legacy_donor_contract_shape(tmp_path):
    contract, fixture = _grounded_ready_recipe_fixture(tmp_path)
    legacy = {
        "path": contract["path"],
        "sha256": contract["sha256"],
        "donor_motion_id": "fixture_motion",
        "donor_source_frame": 0,
        "donor_source_sha256": "1" * 64,
        "endpoint_velocity_policy": contract["endpoint_velocity_policy"],
    }
    ready, provenance, normalized = recipe._load_canonical_ready_contract(
        tmp_path, legacy
    )
    assert ready.path == fixture["published"].ready_npz
    assert provenance.mode == "legacy_donor_frame_exact_v1"
    assert normalized == legacy


def test_recipe_new_mode_rejects_legacy_donor_laundering(tmp_path):
    contract, _ = _grounded_ready_recipe_fixture(tmp_path)
    contract["donor_motion_id"] = "bh_loop_c"
    with pytest.raises(recipe.MotionRecipeError, match="keys changed"):
        recipe._load_canonical_ready_contract(tmp_path, contract)


def _rewrite_contract_report(
    tmp_path: Path,
    contract: dict,
    binding_name: str,
    seal_field: str,
    mutation,
) -> None:
    binding = contract[binding_name]
    path = tmp_path / binding["path"]
    payload = json.loads(path.read_text("ascii"))
    payload.pop(seal_field, None)
    mutation(payload)
    file_sha, payload_sha = _write_sealed_json(path, payload, seal_field)
    binding["sha256"] = file_sha
    binding["payload_sha256"] = payload_sha


@pytest.mark.parametrize(
    ("binding_name", "seal_field", "mutation", "message"),
    [
        (
            "minter_identity_report",
            "report_payload_sha256",
            lambda row: row["face_identity"].__setitem__(
                "face_neutrality_proven", True
            ),
            "may not claim its own face-neutrality",
        ),
        (
            "face_neutrality_report",
            "report_payload_sha256",
            lambda row: row.__setitem__("verdict", "INCOMPLETE_FAIL_CLOSED"),
            "schema/verdict",
        ),
        (
            "face_neutrality_report",
            "report_payload_sha256",
            lambda row: row["producer"].__setitem__(
                "independent_from_ready_minter", False
            ),
            "must be independent",
        ),
        (
            "face_neutrality_report",
            "report_payload_sha256",
            lambda row: row["evaluation"].__setitem__(
                "maximum_allowed_pair_asymmetry_rad", 0.5
            ),
            "must be <=",
        ),
        (
            "human_adoption_evidence",
            "evidence_payload_sha256",
            lambda row: row["decision"].__setitem__("decision_maker_kind", "agent"),
            "explicit named human",
        ),
        (
            "human_adoption_evidence",
            "evidence_payload_sha256",
            lambda row: row["decision"].__setitem__(
                "selected_as_canonical_ready", False
            ),
            "explicit named human",
        ),
        (
            "human_adoption_evidence",
            "evidence_payload_sha256",
            lambda row: row["authorization"].__setitem__("training_authorized", True),
            "must deny training",
        ),
    ],
)
def test_recipe_grounded_ready_evidence_fails_closed(
    tmp_path,
    binding_name,
    seal_field,
    mutation,
    message,
):
    contract, _ = _grounded_ready_recipe_fixture(tmp_path)
    _rewrite_contract_report(
        tmp_path,
        contract,
        binding_name,
        seal_field,
        mutation,
    )
    with pytest.raises(recipe.MotionRecipeError, match=message):
        recipe._load_canonical_ready_contract(tmp_path, contract)


def test_recipe_rejects_unbound_candidate_and_receipt_payload(tmp_path):
    contract, _ = _grounded_ready_recipe_fixture(tmp_path)
    contract["candidate"]["sha256"] = "0" * 64
    with pytest.raises(recipe.MotionRecipeError, match="SHA-256 mismatch"):
        recipe._load_canonical_ready_contract(tmp_path, contract)

    second_root = tmp_path / "second"
    second_root.mkdir()
    contract, _ = _grounded_ready_recipe_fixture(second_root)
    contract["grounded_receipt"]["payload_sha256"] = "0" * 64
    with pytest.raises(recipe.MotionRecipeError, match="payload SHA-256"):
        recipe._load_canonical_ready_contract(second_root, contract)


def test_recipe_rejects_face_report_produced_by_minter_itself(tmp_path):
    contract, _ = _grounded_ready_recipe_fixture(tmp_path)
    minter_tool = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "canonical_ready_sidecar_mint.py"
    )
    minter_tool.parent.mkdir(parents=True, exist_ok=True)
    minter_tool.write_bytes((SCRIPTS / minter_tool.name).read_bytes())

    def forge_producer(report):
        report["producer"]["tool_path"] = minter_tool.relative_to(tmp_path).as_posix()
        report["producer"]["tool_sha256"] = _sha256(minter_tool.read_bytes())

    _rewrite_contract_report(
        tmp_path,
        contract,
        "face_neutrality_report",
        "report_payload_sha256",
        forge_producer,
    )
    with pytest.raises(recipe.MotionRecipeError, match="must be independent"):
        recipe._load_canonical_ready_contract(tmp_path, contract)
