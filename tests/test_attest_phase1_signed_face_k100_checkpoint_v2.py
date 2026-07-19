from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/attest_phase1_signed_face_k100_checkpoint_v2.py"
MANIFEST = ROOT / "configs/phase1_signed_face_k100_checkpoint_attestor_v2_20260714.json"
SCHEDULE_MODULE = ROOT / "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = _load(SCRIPT, "signed_k100_checkpoint_attestor_v2_under_test")
S = _load(SCHEDULE_MODULE, "signed_k100_checkpoint_schedule_under_test")


def _tracked_manifest_document():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _manifest():
    """Load a current-source fixture without rewriting the frozen manifest."""
    document = _tracked_manifest_document()
    for source in document["source_bindings"].values():
        source["sha256"] = A.sha256_file(ROOT / source["path"])
    original_read_json = A.read_json

    def read_json(path, label):
        if Path(path).resolve() == MANIFEST.resolve():
            return copy.deepcopy(document)
        return original_read_json(path, label)

    with mock.patch.object(A, "read_json", side_effect=read_json):
        return A.load_manifest(MANIFEST, repo_root=ROOT)


def _fingerprint(packages):
    return {
        "implementation": "CPython",
        "version": "3.10.test",
        "executable": "/runtime/python",
        "packages": {name: "1.0" for name in packages},
    }


def _request(manifest, tmp_path: Path):
    checkpoint = tmp_path / "run" / "model_1000.pt"
    hard = checkpoint.parent / "params" / "training_contract.json"
    claim = tmp_path / "claims" / "launch_claim.json"
    checkpoint_sha = "1" * 64
    return {
        "schema_version": 1,
        "request_id": "signed-k100-test-1000",
        "status": A.REQUEST_STATUS,
        "human_owner": "Franco",
        "executor": "Codex",
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "source_checkout": {
            "path": str(ROOT.resolve()),
            "commit": "2" * 40,
            "tree": "3" * 40,
            "clean_required": True,
        },
        "isaac_asset_bundle": {
            "source_root": str(A.TRAINING_ASSET_ROOT),
            "destination_root": str(ROOT.resolve() / A.ASSET_RELATIVE_PATH),
            "inventory": {
                "schema_version": 1,
                "algorithm": A.ASSET_INVENTORY_ALGORITHM,
                "file_count": 50,
                "total_bytes": 1000,
                "canonical_sha256": "a" * 64,
            },
            "required_urdf": A._required_urdf_spec(),
            "hydration_mode": "verify_existing",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": 123,
            "sha256": checkpoint_sha,
            "filename_iteration": 1000,
            "embedded_iteration": 1000,
            "training_contract_schema_version": 3,
            "training_contract_sha256": "4" * 64,
            "training_contract_lineage_exact": 1,
            "producer_claim": {
                "path": str(claim),
                "file_sha256": "5" * 64,
                "canonical_sha256": "6" * 64,
            },
        },
        "adjacent_hard_contract": {
            "path": str(hard),
            "bytes": 456,
            "sha256": "4" * 64,
            "plant_contract_sha256": "7" * 64,
        },
        "runtime": {
            "checkpoint_python": {
                "path": "/runtime/checkpoint-python",
                "resolved_path": "/runtime/python",
                "resolved_sha256": "8" * 64,
                "fingerprint": _fingerprint(["torch"]),
            },
            "evaluator_python": {
                "path": "/runtime/evaluator-python",
                "resolved_path": "/runtime/python",
                "resolved_sha256": "8" * 64,
                "fingerprint": _fingerprint(["mujoco", "numpy", "onnx", "onnxruntime"]),
            },
        },
        "mjcf": {
            "path": "/runtime/vendor/a3_pingpong.xml",
            "bytes": 789,
            "sha256": "9" * 64,
        },
        "output": {"root": str(Path(manifest["output"]["root"]) / checkpoint_sha)},
        "authorization": copy.deepcopy(A.ATTESTATION_AUTHORIZATION),
    }


def _write_request(tmp_path: Path, request):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def _write_manifest(tmp_path: Path, manifest):
    path = tmp_path / "attestor.manifest.json"
    document = {key: value for key, value in manifest.items() if not key.startswith("_")}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _plant_contract():
    value = {key: [0.0] * 31 for key in A.PLANT_KEYS}
    value.update(
        {
            "schema_version": 3,
            "actor_obs_contract": "deploy_parity_face179",
            "actor_obs_total_dim": 179,
            "face_command_pairing": "shared_plus_y",
            "mount_normal_sign_per_clip": [1.0, -1.0],
            "motion_kinematics_exact": True,
            "motion_allow_legacy_link_origin_velocity": False,
            "joint_names": [f"joint_{index}" for index in range(31)],
            "action_joint_ids": list(range(31)),
            "articulation_joint_names": [f"joint_{index}" for index in range(31)],
            "joint_actuator_types": ["implicit"] * 31,
            "joint_friction_backend": "physx",
            "joint_friction_semantics": "load_dependent_spatial_force_coefficient",
            "joint_friction_units": "dimensionless",
            "action_use_default_offset": True,
            "qdes_clamp": True,
            "physics_step_dt_s": 0.005,
            "policy_step_dt_s": 0.02,
            "control_decimation": 4,
        }
    )
    return value


def _paper_fixture(tmp_path: Path, manifest):
    question_ids = tuple(
        tuple(A.canonical_sha256({"clip": clip, "row": row}) for row in range(60))
        for clip in range(2)
    )
    artifact = S.materialize_balanced_bank_exam_schedule(
        bank_sha256=manifest["paper"]["bank"]["sha256"],
        clip_names=("forehand", "backhand"),
        question_ids=question_ids,
        per_clip_quota=50,
        schedule_seed=0,
        hold_range=(0, 100),
    )
    schedule_path = tmp_path / "paper" / "signed.schedule.json"
    schedule_path.parent.mkdir()
    schedule_path.write_bytes(S.canonical_json_bytes(S.artifact_document(artifact)) + b"\n")
    order = [item.question_id for item in artifact.items]
    receipt = {
        "path": str(schedule_path.resolve()),
        "bytes": schedule_path.stat().st_size,
        "file_sha256": A.sha256_file(schedule_path),
        "semantic_sha256": artifact.schedule_sha256,
        "question_id_order_sha256": A.canonical_sha256(order),
        "question_id_order": order,
        "schedule_k": 100,
        "selected_per_side": {"forehand": 50, "backhand": 50},
    }
    content = {
        "status": "paper_materialized_not_started",
        "bank": {"sha256": manifest["paper"]["bank"]["sha256"]},
        "schedule": receipt,
        "signed_face_contract": copy.deepcopy(A.EXPECTED_FACE_CONTRACT),
        "scoring_denominator": copy.deepcopy(manifest["paper"]["denominator"]),
        "all_scheduled_attempts_in_denominator": True,
        "authorization": copy.deepcopy(A.PAPER_AUTHORIZATION),
    }
    activation = A.content_document(A.ACTIVATION_KIND, content)
    activation_path = schedule_path.parent / "signed.activation.json"
    activation_path.write_bytes(A.canonical_bytes(activation) + b"\n")
    manifest["paper"]["schedule"] = {
        "path": str(schedule_path.resolve()),
        "bytes": schedule_path.stat().st_size,
        "file_sha256": A.sha256_file(schedule_path),
        "semantic_sha256": artifact.schedule_sha256,
        "question_id_order_sha256": A.canonical_sha256(order),
        "scheduled_attempts": 100,
        "unique_question_ids": 100,
        "selected_per_side": {"forehand": 50, "backhand": 50},
    }
    manifest["paper"]["activation"] = {
        "path": str(activation_path.resolve()),
        "bytes": activation_path.stat().st_size,
        "file_sha256": A.sha256_file(activation_path),
        "content_sha256": activation["content_sha256"],
        "artifact_kind": A.ACTIVATION_KIND,
    }
    return schedule_path, activation_path


def test_tracked_manifest_fails_closed_after_bound_source_changes():
    with pytest.raises(A.ContractError, match="source binding play_exporter bytes changed"):
        A.load_manifest(MANIFEST, repo_root=ROOT)


def test_manifest_semantics_are_attestation_only_and_correction_preserves_original():
    manifest = _manifest()
    assert manifest["authorization"] == A.ATTESTATION_AUTHORIZATION
    assert manifest["paper"]["signed_face_contract"]["mount_normal_sign_per_clip"] == [1.0, -1.0]
    assert all(type(value) is float for value in manifest["paper"]["signed_face_contract"]["mount_normal_sign_per_clip"])
    correction = json.loads(
        (ROOT / manifest["receipt_correction"]["path"]).read_text(encoding="utf-8")
    )
    assert correction["original_receipt"]["preserved_unchanged"] is True
    assert correction["recorded_summary_value"] == [1, -1]
    assert all(type(value) is int for value in correction["recorded_summary_value"])
    assert all(type(value) is float for value in correction["correction"]["correct_exact_value"])


def test_duplicate_json_key_and_nonfinite_constant_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(A.ContractError, match="duplicate JSON key"):
        A.read_json(duplicate, "duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}')
    with pytest.raises(A.ContractError, match="non-finite"):
        A.read_json(nonfinite, "nonfinite")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["checkpoint"].__setitem__("training_contract_lineage_exact", True), "fresh lineage"),
        (lambda value: value["checkpoint"].__setitem__("embedded_iteration", 999), "iterations differ"),
        (lambda value: value["checkpoint"].__setitem__("path", value["checkpoint"]["path"].replace("model_1000.pt", "model_*.pt")), "glob or wildcard"),
        (lambda value: value["runtime"]["evaluator_python"].__setitem__("path", "/runtime/python?"), "glob or wildcard"),
        (lambda value: value["mjcf"].__setitem__("path", "/runtime/vendor/../vendor/a3_pingpong.xml"), "lexical-canonical"),
        (lambda value: value["adjacent_hard_contract"].__setitem__("path", "/tmp/other.json"), "not adjacent"),
        (lambda value: value["isaac_asset_bundle"].__setitem__("source_root", "/tmp/foreign-assets"), "training-time"),
        (lambda value: value["isaac_asset_bundle"].__setitem__("destination_root", "/tmp/foreign-destination"), "exact eval checkout"),
        (lambda value: value["isaac_asset_bundle"].__setitem__("hydration_mode", "copy-anyway"), "hydration_mode"),
        (lambda value: value["output"].__setitem__("root", "/tmp/alternate-output"), "unique checkpoint-SHA"),
        (lambda value: value["authorization"].__setitem__("judge_started", True), "authorization"),
    ),
)
def test_request_mutations_fail_closed(tmp_path, mutation, message):
    manifest = _manifest()
    request = _request(manifest, tmp_path)
    mutation(request)
    path = _write_request(tmp_path, request)
    with pytest.raises(A.ContractError, match=message):
        A.load_request(path, manifest, runtime=False)


def test_plan_accepts_one_explicit_exact_request_without_runtime_reads(tmp_path):
    manifest = _manifest()
    request = _request(manifest, tmp_path)
    path = _write_request(tmp_path, request)
    loaded = A.load_request(path, manifest, runtime=False)
    assert loaded["checkpoint"]["sha256"] == "1" * 64
    assert loaded["output"]["root"].endswith("/" + "1" * 64)


def test_request_symlink_and_duplicate_key_are_rejected(tmp_path):
    manifest = _manifest()
    request = _request(manifest, tmp_path)
    request_path = _write_request(tmp_path, request)
    symlink = tmp_path / "request-link.json"
    symlink.symlink_to(request_path)
    with pytest.raises(A.ContractError, match="non-symlink"):
        A.load_request(symlink, manifest, runtime=False)

    duplicate = tmp_path / "request-duplicate.json"
    payload = json.dumps(request)
    payload = payload.replace('"request_id":', '"request_id":"shadow","request_id":', 1)
    duplicate.write_text(payload, encoding="utf-8")
    with pytest.raises(A.ContractError, match="duplicate JSON key"):
        A.load_request(duplicate, manifest, runtime=False)


def test_runtime_file_binding_rejects_symlink_ancestry(tmp_path):
    real = tmp_path / "real" / "model_1000.pt"
    real.parent.mkdir()
    real.write_bytes(b"checkpoint")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real.parent, target_is_directory=True)
    linked = linked_parent / real.name
    with pytest.raises(A.ContractError, match="symlink ancestry"):
        A.validate_file_binding(
            linked,
            size=real.stat().st_size,
            digest=A.sha256_file(real),
            label="exact checkpoint",
        )


def test_hard_contract_requires_exact_float_face_and_bound_plant_hash(tmp_path):
    manifest = _manifest()
    request = _request(manifest, tmp_path)
    hard = _plant_contract()
    plant = {key: hard[key] for key in A.PLANT_KEYS}
    request["adjacent_hard_contract"]["plant_contract_sha256"] = A.canonical_sha256(plant)
    assert A.validate_hard_contract(hard, request, manifest) == A.canonical_sha256(plant)
    hard["mount_normal_sign_per_clip"] = [1, -1]
    with pytest.raises(A.ContractError, match="bool/int/float"):
        A.validate_hard_contract(hard, request, manifest)
    hard["mount_normal_sign_per_clip"] = [1.0, -1.0]
    hard["joint_armature"][0] = 1.0
    with pytest.raises(A.ContractError, match="plant semantic SHA"):
        A.validate_hard_contract(hard, request, manifest)


def test_actual_activation_is_direct_authority_and_int_summary_cannot_pass(tmp_path):
    manifest = copy.deepcopy(_manifest())
    _, activation_path = _paper_fixture(tmp_path, manifest)
    result = A.validate_actual_paper(manifest)
    assert result["actual_signed_face_contract"]["mount_normal_sign_per_clip"] == [1.0, -1.0]

    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    activation["content"]["signed_face_contract"]["mount_normal_sign_per_clip"] = [1, -1]
    activation["content_sha256"] = A.canonical_sha256(activation["content"])
    activation_path.write_bytes(A.canonical_bytes(activation) + b"\n")
    manifest["paper"]["activation"]["bytes"] = activation_path.stat().st_size
    manifest["paper"]["activation"]["file_sha256"] = A.sha256_file(activation_path)
    manifest["paper"]["activation"]["content_sha256"] = activation["content_sha256"]
    with pytest.raises(A.ContractError, match="bool/int/float"):
        A.validate_actual_paper(manifest)


def test_checkpoint_audit_finite_lineage_claim_and_runtime_bindings(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    request = _request(manifest, tmp_path)
    checkpoint_path = Path(request["checkpoint"]["path"])
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    request["checkpoint"]["bytes"] = checkpoint_path.stat().st_size
    request["checkpoint"]["sha256"] = A.sha256_file(checkpoint_path)
    manifest["output"]["root"] = str(tmp_path / "outputs")
    request["output"]["root"] = str(Path(manifest["output"]["root"]) / request["checkpoint"]["sha256"])

    hard_path = Path(request["adjacent_hard_contract"]["path"])
    hard_path.parent.mkdir()
    hard = _plant_contract()
    hard_path.write_text(json.dumps(hard), encoding="utf-8")
    request["adjacent_hard_contract"]["bytes"] = hard_path.stat().st_size
    request["adjacent_hard_contract"]["sha256"] = A.sha256_file(hard_path)
    request["checkpoint"]["training_contract_sha256"] = A.sha256_file(hard_path)
    request["adjacent_hard_contract"]["plant_contract_sha256"] = A.canonical_sha256(
        {key: hard[key] for key in A.PLANT_KEYS}
    )

    claim_path = Path(request["checkpoint"]["producer_claim"]["path"])
    claim_path.parent.mkdir()
    producer_claim = {"claim": "exact", "gpu": 0}
    claim_path.write_text(json.dumps(producer_claim), encoding="utf-8")
    request["checkpoint"]["producer_claim"]["file_sha256"] = A.sha256_file(claim_path)
    request["checkpoint"]["producer_claim"]["canonical_sha256"] = A.canonical_sha256(producer_claim)

    mjcf_path = tmp_path / "vendor.xml"
    mjcf_path.write_text("<mujoco/>", encoding="utf-8")
    request["mjcf"] = {
        "path": str(mjcf_path),
        "bytes": mjcf_path.stat().st_size,
        "sha256": A.sha256_file(mjcf_path),
    }
    _paper_fixture(tmp_path, manifest)
    monkeypatch.setattr(A, "validate_source_checkout", lambda *args, **kwargs: {"clean": True})
    monkeypatch.setattr(A, "_runtime_fingerprint", lambda spec, names, label: spec["fingerprint"])
    monkeypatch.setattr(
        A,
        "validate_asset_bundle",
        lambda *args, **kwargs: {
            "inventory": request["isaac_asset_bundle"]["inventory"],
            "destination_verified_exact": True,
        },
    )
    monkeypatch.setattr(
        A, "validate_libglu_presence", lambda: {"soname": A.LIBGLU_SONAME, "loadable": True}
    )
    audit = {
        "iter": 1000,
        "training_contract_schema_version": 3,
        "training_contract_sha256": A.sha256_file(hard_path),
        "training_contract_lineage_exact": 1,
        "training_launch_claim_sha256": A.canonical_sha256(producer_claim),
        "floating_tensor_count": 2,
        "floating_elements": 100,
        "nonfinite_floating_elements": 0,
    }
    monkeypatch.setattr(A, "checkpoint_audit", lambda *args: dict(audit))
    result = A.validate_runtime_request(request, manifest)
    assert result["checkpoint_audit"]["nonfinite_floating_elements"] == 0
    audit["nonfinite_floating_elements"] = 1
    with pytest.raises(A.ContractError, match="finite"):
        A.validate_runtime_request(request, manifest)

    audit["nonfinite_floating_elements"] = 0

    def replace_checkpoint(*args):
        checkpoint_path.unlink()
        checkpoint_path.write_bytes(b"replacement checkpoint bytes")
        return dict(audit)

    monkeypatch.setattr(A, "checkpoint_audit", replace_checkpoint)
    with pytest.raises(A.ContractError, match="replaced or changed|bytes/SHA changed"):
        A.validate_runtime_request(request, manifest)


def test_attest_writes_evidence_then_claim_once_and_never_decides(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    manifest["output"]["root"] = str(tmp_path / "outputs")
    request = _request(manifest, tmp_path)
    request["output"]["root"] = str(Path(manifest["output"]["root"]) / request["checkpoint"]["sha256"])
    request_path = _write_request(tmp_path, request)
    manifest_path = _write_manifest(tmp_path, manifest)
    runtime = {
        "source_checkout": {"clean": True},
        "checkpoint_audit": {"nonfinite_floating_elements": 0},
        "producer_claim": request["checkpoint"]["producer_claim"],
        "checkpoint_python": request["runtime"]["checkpoint_python"]["fingerprint"],
        "evaluator_python": request["runtime"]["evaluator_python"]["fingerprint"],
        "mjcf": request["mjcf"],
        "plant_contract_sha256": request["adjacent_hard_contract"]["plant_contract_sha256"],
        "isaac_asset_bundle": {
            "inventory": request["isaac_asset_bundle"]["inventory"],
            "destination_root": request["isaac_asset_bundle"]["destination_root"],
            "destination_verified_exact": True,
        },
        "libglu": {"soname": A.LIBGLU_SONAME, "loadable": True},
        "paper": {
            "schedule": manifest["paper"]["schedule"],
            "activation": manifest["paper"]["activation"],
            "actual_signed_face_contract": A.EXPECTED_FACE_CONTRACT,
        },
    }
    monkeypatch.setattr(A, "validate_runtime_request", lambda *args, **kwargs: runtime)
    result = A.attest(request_path, request, manifest_path, manifest)
    assert result["judge_started"] is False
    assert result["stop_or_promote_authorized"] is False
    claim = A.read_json(Path(result["claim_path"]), "claim")
    assert claim["content"]["status"] == "attested_not_executed_no_decision"
    assert claim["content"]["judge_started"] is False
    with pytest.raises(A.ContractError, match="namespace exists"):
        A.attest(request_path, request, manifest_path, manifest)


def test_dangling_symlink_namespace_is_not_treated_as_absent(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    manifest["output"]["root"] = str(tmp_path / "outputs")
    request = _request(manifest, tmp_path)
    request["output"]["root"] = str(Path(manifest["output"]["root"]) / request["checkpoint"]["sha256"])
    request_path = _write_request(tmp_path, request)
    manifest_path = _write_manifest(tmp_path, manifest)
    output = Path(request["output"]["root"])
    output.parent.mkdir()
    output.symlink_to(tmp_path / "absent-target", target_is_directory=True)
    monkeypatch.setattr(A, "validate_runtime_request", lambda *args, **kwargs: pytest.fail("runtime must not run"))
    with pytest.raises(A.ContractError, match="namespace exists"):
        A.attest(request_path, request, manifest_path, manifest)


def test_partial_evidence_failure_permanently_consumes_namespace(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    manifest["output"]["root"] = str(tmp_path / "outputs")
    request = _request(manifest, tmp_path)
    request["output"]["root"] = str(Path(manifest["output"]["root"]) / request["checkpoint"]["sha256"])
    request_path = _write_request(tmp_path, request)
    manifest_path = _write_manifest(tmp_path, manifest)
    runtime = {
        "source_checkout": {"clean": True},
        "checkpoint_audit": {"nonfinite_floating_elements": 0},
        "producer_claim": request["checkpoint"]["producer_claim"],
        "checkpoint_python": request["runtime"]["checkpoint_python"]["fingerprint"],
        "evaluator_python": request["runtime"]["evaluator_python"]["fingerprint"],
        "mjcf": request["mjcf"],
        "plant_contract_sha256": request["adjacent_hard_contract"]["plant_contract_sha256"],
        "isaac_asset_bundle": {
            "inventory": request["isaac_asset_bundle"]["inventory"],
            "destination_root": request["isaac_asset_bundle"]["destination_root"],
            "destination_verified_exact": True,
        },
        "libglu": {"soname": A.LIBGLU_SONAME, "loadable": True},
        "paper": {
            "schedule": manifest["paper"]["schedule"],
            "activation": manifest["paper"]["activation"],
            "actual_signed_face_contract": A.EXPECTED_FACE_CONTRACT,
        },
    }
    monkeypatch.setattr(A, "validate_runtime_request", lambda *args, **kwargs: runtime)
    original_write = A.write_exclusive

    def fail_claim(path, document):
        if document["artifact_kind"] == A.CLAIM_KIND:
            raise OSError("injected claim write failure")
        original_write(path, document)

    monkeypatch.setattr(A, "write_exclusive", fail_claim)
    with pytest.raises(OSError, match="injected"):
        A.attest(request_path, request, manifest_path, manifest)
    output = Path(request["output"]["root"])
    assert (output / "execution_evidence.json").is_file()
    assert not (output / "execution_claim.json").exists()
    with pytest.raises(A.ContractError, match="namespace exists"):
        A.attest(request_path, request, manifest_path, manifest)


def test_request_replacement_during_runtime_never_writes(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    manifest["output"]["root"] = str(tmp_path / "outputs")
    request = _request(manifest, tmp_path)
    request["output"]["root"] = str(Path(manifest["output"]["root"]) / request["checkpoint"]["sha256"])
    request_path = _write_request(tmp_path, request)
    manifest_path = _write_manifest(tmp_path, manifest)

    def replace_request(*args, **kwargs):
        changed = copy.deepcopy(request)
        changed["request_id"] = "replacement-request"
        request_path.write_text(json.dumps(changed), encoding="utf-8")
        return {}

    monkeypatch.setattr(A, "validate_runtime_request", replace_request)
    with pytest.raises(A.ContractError, match="replaced or changed|loaded content"):
        A.attest(request_path, request, manifest_path, manifest)
    assert not Path(request["output"]["root"]).exists()


def test_asset_inventory_hydrates_once_and_rejects_symlinks(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    source = tmp_path / "training" / "assets" / "agibot_a3"
    urdf = source / "urdf" / "model.urdf"
    mesh = source / "meshes" / "body.STL"
    urdf.parent.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    urdf.write_bytes(b"<robot name='fixture'/>")
    mesh.write_bytes(b"mesh-fixture")
    monkeypatch.setattr(A, "REQUIRED_URDF_BYTES", urdf.stat().st_size)
    monkeypatch.setattr(A, "REQUIRED_URDF_SHA256", A.sha256_file(urdf))
    manifest["execution_semantics"]["isaac_asset_hydration"]["source_root"] = str(source)
    eval_root = tmp_path / "eval"
    destination_parent = eval_root / A.ASSET_RELATIVE_PATH.parent
    destination_parent.mkdir(parents=True)
    request = _request(manifest, tmp_path)
    request["source_checkout"]["path"] = str(eval_root)
    request["isaac_asset_bundle"] = {
        "source_root": str(source),
        "destination_root": str(eval_root / A.ASSET_RELATIVE_PATH),
        "inventory": A.asset_inventory_summary(source),
        "required_urdf": A._required_urdf_spec(),
        "hydration_mode": "hydrate_absent",
    }
    before = A.validate_asset_bundle(request, manifest, require_destination=False)
    assert before["destination_verified_exact"] is False
    A.hydrate_asset_bundle(request, manifest)
    after = A.validate_asset_bundle(request, manifest, require_destination=True)
    assert after["destination_verified_exact"] is True
    assert (Path(after["destination_root"]) / "meshes" / "body.STL").read_bytes() == b"mesh-fixture"
    with pytest.raises(A.ContractError, match="destination must be absent"):
        A.hydrate_asset_bundle(request, manifest)

    linked = tmp_path / "linked-assets"
    linked.mkdir()
    (linked / "payload").symlink_to(mesh)
    with pytest.raises(A.ContractError, match="symlink"):
        A.asset_inventory_summary(linked)


def test_asset_hydration_publish_never_overwrites_concurrent_sentinel(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    source = tmp_path / "training" / "assets" / "agibot_a3"
    urdf = source / "urdf" / "model.urdf"
    mesh = source / "meshes" / "body.STL"
    urdf.parent.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    urdf.write_bytes(b"<robot name='fixture'/>")
    mesh.write_bytes(b"trusted-mesh")
    monkeypatch.setattr(A, "REQUIRED_URDF_BYTES", urdf.stat().st_size)
    monkeypatch.setattr(A, "REQUIRED_URDF_SHA256", A.sha256_file(urdf))
    manifest["execution_semantics"]["isaac_asset_hydration"]["source_root"] = str(source)
    eval_root = tmp_path / "eval"
    (eval_root / A.ASSET_RELATIVE_PATH.parent).mkdir(parents=True)
    request = _request(manifest, tmp_path)
    destination = eval_root / A.ASSET_RELATIVE_PATH
    request["source_checkout"]["path"] = str(eval_root)
    request["isaac_asset_bundle"] = {
        "source_root": str(source),
        "destination_root": str(destination),
        "inventory": A.asset_inventory_summary(source),
        "required_urdf": A._required_urdf_spec(),
        "hydration_mode": "hydrate_absent",
    }

    real_link = A.os.link
    sentinel = {"path": None}

    def inject_sentinel_then_link(staged_file, target_file, *, follow_symlinks):
        target = Path(target_file)
        if sentinel["path"] is None:
            target.write_bytes(b"concurrent-sentinel")
            sentinel["path"] = target
        return real_link(staged_file, target_file, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(A.os, "link", inject_sentinel_then_link)
    with pytest.raises(A.ContractError, match="no-replace: concurrent asset file appeared"):
        A.hydrate_asset_bundle(request, manifest)

    assert sentinel["path"] is not None
    assert sentinel["path"].read_bytes() == b"concurrent-sentinel"
    stage = destination.parent / f".{destination.name}.hydrate-{request['request_id']}"
    assert stage.is_dir()
    assert (stage / sentinel["path"].relative_to(destination)).is_file()


def test_libglu_preflight_is_existence_only_and_fails_closed(monkeypatch):
    loaded = []
    monkeypatch.setattr(A.ctypes, "CDLL", lambda soname: loaded.append(soname))
    assert A.validate_libglu_presence() == {"soname": "libGLU.so.1", "loadable": True}
    assert loaded == ["libGLU.so.1"]

    def missing(_soname):
        raise OSError("fixture missing")

    monkeypatch.setattr(A.ctypes, "CDLL", missing)
    with pytest.raises(A.ContractError, match="not loadable"):
        A.validate_libglu_presence()


def test_consumer_has_no_judge_launch_signal_ssh_or_robot_command_surface():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"asset-plan"' in source
    assert '"attest"' in source
    assert "subprocess.Popen" not in source
    assert "os.kill" not in source
    assert "pkill" not in source
    assert "ssh " not in source
    assert "real_robot_authorized\": False" in source
