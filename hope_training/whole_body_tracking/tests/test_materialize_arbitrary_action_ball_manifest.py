"""CPU contracts for the arbitrary-N final ActionBall materializer."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


TESTS = Path(__file__).resolve().parent
WHOLE_BODY_ROOT = TESTS.parent
SCRIPTS = WHOLE_BODY_ROOT / "scripts"
REPO_ROOT = WHOLE_BODY_ROOT.parents[1]
for directory in (SCRIPTS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import canonical_motion_arbitrary_bank as arbitrary  # noqa: E402
import canonical_motion_compiler as compiler  # noqa: E402
import canonical_motion_registry as registry_module  # noqa: E402
import materialize_arbitrary_action_ball_manifest as materializer  # noqa: E402
import test_action_ball_manifest as action_manifest_tests  # noqa: E402
import test_canonical_motion_arbitrary_bank as arbitrary_tests  # noqa: E402
import test_canonical_motion_compiler as compiler_tests  # noqa: E402
import test_canonical_motion_registry as registry_tests  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return _sha(path)


def _canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _source_action_manifest(fixture) -> dict:
    actions = []
    for index, source in enumerate(fixture.capsule["actions"]):
        profile = deepcopy(action_manifest_tests._profile(index))
        center = list(source["base_spawn_center_w_xy_m"])
        profile["base_spawn_center_w_xy_m"] = center
        profile["base_spawn_min_w_xy_m"] = [
            center[0] - 0.5,
            center[1] - 0.5,
        ]
        profile["base_spawn_max_w_xy_m"] = [
            center[0] + 0.5,
            center[1] + 0.5,
        ]
        profile["time_to_contact_center_s"] = 0.70
        profile["time_to_contact_min_s"] = 0.55
        profile["time_to_contact_max_s"] = 1.00
        action_id = source["action_id"]
        family = source["family"]
        motion_sha = source["motion_sha256"]
        actions.append(
            {
                "action_id": action_id,
                "action_uid": (
                    action_manifest_tests.M.derive_action_ball_action_uid(
                        action_id, family, motion_sha
                    )
                ),
                "motion_path": (
                    fixture.capsule_directory.relative_to(fixture.root)
                    / source["motion_path"]
                ).as_posix(),
                "motion_sha256": motion_sha,
                "strike_phase": (
                    source["hit_frame_50"] / (source["T"] - 1)
                ),
                "reference_t_hit_s": source["reference_t_hit_s"],
                "reference_t_cycle_s": source["reference_t_cycle_s"],
                "reference_racket_site_speed_mps": 2.0,
                "reaction_margin_s": 0.05,
                "teacher_rate_min": 0.8,
                "teacher_rate_max": 1.2,
                "family": family,
                "mount_normal_sign": -1,
                "ball_profile": profile,
            }
        )
    document = action_manifest_tests._document(len(actions))
    document["manifest_id"] = "source_action_ball_n73"
    document["action_order"] = [row["action_id"] for row in actions]
    document["actions"] = actions
    # The old source profile is intentionally pre-formal.  The materializer
    # may validate it only after lifting the final heldout floor to >=768.
    document["holdout"]["samples_per_action"] = 512
    return document


def _copy_runtime_contract_files(root: Path) -> None:
    relative = Path(materializer.MDP_DIR_REL)
    source = REPO_ROOT / relative
    target = root / relative
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "action_ball_manifest.py",
        "stroke_prototypes_torch.py",
        "racket_contact_geometry.py",
        "counter_rally.py",
    ):
        shutil.copy2(source / name, target / name)


def _build_synthetic_compiled_bank(
    *,
    loaded,
    compiled: Path,
):
    """Create the exact N x 2 post-compiler file/manifest contract."""

    compiled.mkdir(parents=True)
    body_names = registry_module.RUNTIME_BODY_NAMES
    wrist = body_names.index("right_wrist_yaw_Link")
    ready_joint = np.asarray(
        loaded.canonical_recipe.ready.joint_pos, dtype=np.float32
    )
    outputs = []
    for index, (motion_id, scope) in enumerate(
        (motion_id, scope)
        for motion_id in loaded.motion_ids
        for scope in ("upper", "full")
    ):
        fps = 50.0
        frames = 16
        joint_pos = np.repeat(ready_joint[None, :], frames, axis=0)
        joint_vel = np.zeros_like(joint_pos)
        body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
        body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        time = np.arange(frames, dtype=np.float32)
        body_pos[:, wrist, 0] = np.float32(0.04) * np.sin(
            np.float32(2.0 * np.pi) * time / np.float32(frames - 1)
        )
        body_pos[-1] = body_pos[0]
        body_lin = np.zeros_like(body_pos)
        body_lin[:, wrist, 0] = np.gradient(
            body_pos[:, wrist, 0], np.float32(1.0 / fps)
        )
        body_lin[0] = 0.0
        body_lin[-1] = 0.0
        body_ang = np.zeros_like(body_pos)
        filename = f"{motion_id}_{scope}_canonical_v2.npz"
        path = compiled / filename
        np.savez(
            path,
            fps=np.asarray([fps], dtype=np.float64),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=body_lin,
            body_ang_vel_w=body_ang,
            kinematics_schema_version=np.asarray([2], dtype=np.int64),
            body_pos_point=np.asarray("link_origin"),
            body_lin_vel_point=np.asarray("center_of_mass"),
            body_names=np.asarray(body_names),
        )
        _write_json(path.with_name(path.name + ".manifest.json"), {})
        _write_json(path.with_name(path.name + ".report.json"), {})
        outputs.append(
            {
                "motion_id": motion_id,
                "scope": scope,
                "filename": filename,
                "output_npz_sha256": _sha(path),
                "entry_frame": 3,
                "exit_frame": 5,
                "duration_s": 0.30,
                "contact_window_start_s": 0.10,
                "contact_window_end_s": 0.18,
                "source_anchor_time_s": 0.14,
                "scaled_l2_total_variation": float(index),
                "search": {
                    "contact_opportunity": {
                        "marker_only": True,
                        "acceleration_allowed_through_window_end": True,
                    }
                },
                "scope_preprocessing": {},
                "face_manifold": None,
                "geometry": {},
                "retiming": {},
                "schema2_manifest": {},
                "schema2_report": {},
            }
        )
    build_document = {
        "schema_version": 1,
        "library_id": loaded.raw["bank_id"],
        "publication_class": arbitrary.PUBLICATION_CLASS,
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "training_authorized": False,
        "hardware_authorized": False,
        "recipe": {
            "path": str(loaded.path),
            "sha256": loaded.sha256,
        },
        "output_matrix": {
            "motion_ids": list(loaded.motion_ids),
            "scopes": ["upper", "full"],
            "candidate_count": 2 * len(loaded.motion_ids),
        },
        "outputs": outputs,
    }
    _write_json(compiled / compiler.BUILD_MANIFEST_NAME, build_document)
    arbitrary.validate_arbitrary_build_manifest(build_document, loaded)
    return build_document


def _build_registry(
    *,
    root: Path,
    loaded,
    compiled: Path,
    build_document: dict,
    scope: str,
) -> tuple[Path, object]:
    outputs = {
        (row["motion_id"], row["scope"]): row
        for row in build_document["outputs"]
    }
    first_path = (
        compiled / outputs[(loaded.motion_ids[0], scope)]["filename"]
    )
    with np.load(first_path, allow_pickle=False) as data:
        body_pos = np.asarray(data["body_pos_w"])[0]
        body_quat = np.asarray(data["body_quat_w"])[0]
    ready_path = loaded.canonical_recipe.ready.path
    ready_sha = loaded.canonical_recipe.ready.sha256
    ready_fk_path = root / "registry" / "canonical_ready_fk.npz"
    ready_fk_sha = registry_tests._write_ready_fk(
        ready_fk_path,
        canonical_ready_sha256=ready_sha,
        body_pos=body_pos,
        body_quat=body_quat,
    )

    entries = []
    for index, motion_id in enumerate(loaded.motion_ids):
        output = outputs[(motion_id, scope)]
        npz_path = compiled / output["filename"]
        with np.load(npz_path, allow_pickle=False) as data:
            frames = int(np.asarray(data["joint_pos"]).shape[0])
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        start = int(round(float(output["contact_window_start_s"]) * fps))
        end = int(round(float(output["contact_window_end_s"]) * fps))
        marker = int(round(float(output["source_anchor_time_s"]) * fps))
        source_path = (
            root / "registry" / "source" / f"{motion_id}.json"
        )
        source_sha = _write_json(
            source_path,
            {"motion_id": motion_id, "source_index": index},
        )
        applicability_path = (
            root / "registry" / "applicability" / f"{motion_id}.json"
        )
        applicability_sha = _write_json(
            applicability_path,
            {
                "schema_version": 1,
                "motion_id": motion_id,
                "domain": "synthetic-arbitrary-n73-materializer",
                "scope": scope,
                "variant": "compiled_fixture_v1",
                "npz_sha256": output["output_npz_sha256"],
            },
        )
        evidence_path, evidence_sha = (
            registry_tests._write_evidence_bundle(
                root,
                motion_id=motion_id,
                scope=scope,
                variant="compiled_fixture_v1",
                npz_sha256=output["output_npz_sha256"],
                level="E0",
            )
        )
        build_receipt = (
            root
            / "registry"
            / "build"
            / f"{motion_id}.{scope}.json"
        )
        build_receipt_sha = _write_json(
            build_receipt,
            {
                "hashes": {
                    "output_npz_sha256": output["output_npz_sha256"],
                    "ready_sha256": ready_sha,
                },
                "publication_class": "compiler_candidate",
                "training_authorized": False,
            },
        )
        entries.append(
            {
                "motion_id": motion_id,
                "scope": scope,
                "variant": "compiled_fixture_v1",
                "npz_path": npz_path.relative_to(root).as_posix(),
                "npz_sha256": output["output_npz_sha256"],
                "frames": frames,
                "fps": fps,
                "family": "backhand",
                "strike_marker_frame": marker,
                "contact_opportunity_frames": [start, end],
                "mount_normal_sign": -1.0,
                "canonical_ready_sha256": ready_sha,
                "source_manifest_path": (
                    source_path.relative_to(root).as_posix()
                ),
                "source_manifest_sha256": source_sha,
                "build_manifest_path": (
                    build_receipt.relative_to(root).as_posix()
                ),
                "build_manifest_sha256": build_receipt_sha,
                "applicability_manifest_path": (
                    applicability_path.relative_to(root).as_posix()
                ),
                "applicability_manifest_sha256": applicability_sha,
                "evidence_level": "E0",
                "evidence_manifest_path": (
                    evidence_path.relative_to(root).as_posix()
                ),
                "evidence_manifest_sha256": evidence_sha,
                "question_bank_path": None,
                "question_bank_sha256": None,
                "question_bank_schema_version": None,
                "training_config_path": None,
                "training_config_sha256": None,
                "training_config_schema_version": None,
                "onnx_model_path": None,
                "onnx_model_sha256": None,
                "onnx_model_schema_version": None,
                "onnx_metadata_path": None,
                "onnx_metadata_sha256": None,
                "onnx_metadata_schema_version": None,
                "adoption_manifest_path": None,
                "adoption_manifest_sha256": None,
                "publication_class": "compiler_candidate",
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            }
        )
    registry_document = {
        "schema_version": 2,
        "bank_id": loaded.raw["bank_id"],
        "scope": scope,
        "motion_ids": list(loaded.motion_ids),
        "canonical_ready_path": ready_path.relative_to(root).as_posix(),
        "canonical_ready_sha256": ready_sha,
        "canonical_ready_fk_path": (
            ready_fk_path.relative_to(root).as_posix()
        ),
        "canonical_ready_fk_sha256": ready_fk_sha,
        "entries": entries,
    }
    registry_path = root / "registry" / f"{scope}_registry.json"
    registry_sha = _write_json(registry_path, registry_document)
    loaded_registry = registry_module.load_canonical_motion_bank_registry(
        registry_path,
        repo_root=root,
        expected_registry_sha256=registry_sha,
    )
    return registry_path, loaded_registry


def _build_prototype(
    *,
    root: Path,
    registry,
    compiled: Path,
    build_document: dict,
) -> tuple[Path, dict]:
    outputs = {
        (row["motion_id"], row["scope"]): row
        for row in build_document["outputs"]
    }
    rows = []
    for index, entry in enumerate(registry.entries):
        output = outputs[(entry.motion_id, registry.scope)]
        contact = entry.strike_marker_frame
        rows.append(
            {
                "motion_id": entry.motion_id,
                "scope": registry.scope,
                "clip_index": index,
                "npz_sha256": entry.npz_sha256,
                "family": entry.family,
                "face_sign": entry.mount_normal_sign,
                "frames": entry.frames,
                "strike_phase": contact / (entry.frames - 1),
                "t_prepare_s": contact / entry.fps,
                "t_prepare_min_s": 0.05,
                "t_prepare_max_s": 2.0,
                "band_b_x": [0.1, 0.2],
                "band_b_y": [-0.2, 0.2],
                "band_z_w": [0.8, 1.2],
                "slack_b_xy_m": 0.1,
                "slack_z_w_m": 0.1,
                "p_contact_b": [0.2, 0.0, 1.0],
                "n_hat_b": [1.0, 0.0, 0.0],
                "priority": index,
                "enabled": True,
                "contact_frame": contact,
                "contact_window_frames": list(
                    entry.contact_opportunity_frames
                ),
                "racket_face_center_velocity_hat_b": [1.0, 0.0, 0.0],
                "racket_face_center_elevation_deg": 0.0,
                "racket_face_center_window_dir_cone_deg": 2.0,
                "racket_face_center_speed_nominal_mps": 2.0,
                "racket_face_center_speed_max_mps": 3.0,
                "racket_face_center_speed_min_mps": 1.0,
                "racket_face_center_v_star_cap_mps": 3.0,
                "racket_face_center_v_dir_tol_deg": 10.0,
                "racket_face_center_cos_normal_velocity": 0.0,
            }
        )
        assert (compiled / output["filename"]).is_file()
    scopes = {registry.scope: rows}
    document = {
        "schema_version": 2,
        "prototype_set_id": "synthetic_arbitrary_n73_v2",
        "velocity_contract": {
            "direction_and_speed_point": "selected_rubber_face_center",
            "policy_control_point": "official_racket_site",
            "mapping": (
                "v_face_center=v_site+omega_world_cross_"
                "r_face_center_from_site_world"
            ),
            "site_velocity_authority": (
                "centered_position_fd_half_window_2_clamped_per_clip"
            ),
            "angular_velocity_authority": (
                "npz_body_ang_vel_w_at_right_wrist_yaw_Link"
            ),
            "direction_frame_authority": (
                "canonical_ready_root_yaw_at_frame_0"
            ),
            "geometry_source_sha256": (
                action_manifest_tests.M._exact_face_geometry_source_sha256()
            ),
        },
        "contact_rule": {},
        "provenance": {},
        "scopes": scopes,
        "derived_sha256": hashlib.sha256(
            json.dumps(
                scopes,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    path = root / "configs" / "stroke_prototypes_n73_v2.json"
    _write_json(path, document)
    return path, document


def _build_profile_pins(root: Path) -> Path:
    physics_payload = {
        "kind": "synthetic_physics_fixture",
        "schema_version": 1,
    }
    physics_sha = _canonical_sha(physics_payload)
    solver_payload = {
        "kind": "synthetic_solver_fixture",
        "schema_version": 1,
        "physics_profile_sha256": physics_sha,
    }
    solver_sha = _canonical_sha(solver_payload)
    document = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball.profile_pins",
        "source_authority": {
            "schema_version": 1,
            "authority": "external_exact_commit_subset_blob_map_v1",
            "commit_binding": (
                "external_preexec_immutable_launch_capsule_v1"
            ),
            "embedded_commit": False,
            "source_blob_map_sha256": _canonical_sha(
                {"fixture.py": "a" * 64}
            ),
        },
        "cfg": {},
        "geometry": {},
        "venue_yaml": "configs/ball_physics_venue.yaml",
        "venue_yaml_sha256": "b" * 64,
        "planes": {},
        "solver_implementation_source_sha256": {
            "fixture.py": "a" * 64
        },
        "contact_geometry": {},
        "physics_profile_sha256": physics_sha,
        "solver_profile_sha256": solver_sha,
        "physics_payload": physics_payload,
        "solver_payload": solver_payload,
    }
    path = root / "configs" / "profile_pins.json"
    _write_json(path, document)
    return path


def _build_report(
    *,
    root: Path,
    loaded,
    registry,
    compiled: Path,
    build_document: dict,
) -> tuple[Path, dict]:
    binding = registry_module.bank_promotion_binding(
        registry, authorization_purpose="training"
    )
    report = registry_tests._complete_bank_gate_report(
        binding,
        root,
        loaded.canonical_recipe.ready.path,
        report_schema_version=2,
    )
    build_path = compiled / compiler.BUILD_MANIFEST_NAME
    report["manifest"] = {
        "path": build_path.relative_to(root).as_posix(),
        "sha256": _sha(build_path),
    }
    report["bank_dir"] = compiled.relative_to(root).as_posix()
    report["bound_inputs"]["recipe"] = {
        "path": loaded.path.relative_to(root).as_posix(),
        "sha256": loaded.sha256,
    }
    generic_target = (
        root
        / "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_generic_bank_gate.py"
    )
    generic_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPTS / generic_target.name, generic_target)
    report["bound_inputs"]["verifier_tools"]["bank_gate"] = {
        "path": generic_target.relative_to(root).as_posix(),
        "sha256": _sha(generic_target),
    }
    output_by_key = {
        (row["motion_id"], row["scope"]): row
        for row in build_document["outputs"]
    }
    for report_row in report["clips"]:
        output = output_by_key[
            (report_row["motion_id"], report_row["scope"])
        ]
        npz_path = compiled / output["filename"]
        with np.load(npz_path, allow_pickle=False) as data:
            report_row["frames"] = int(data["joint_pos"].shape[0])
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        report_row["fps"] = fps
        report_row["duration_s"] = (report_row["frames"] - 1) / fps
        report_row["filename"] = output["filename"]
        report_row["sha256"] = output["output_npz_sha256"]
    path = root / "reports" / "generic_bank_gate_v2.json"
    _write_json(path, report)
    return path, report


class N73Fixture:
    pass


@pytest.fixture(scope="module")
def n73_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("arbitrary_action_ball_n73")
    patcher = pytest.MonkeyPatch()
    fixture = arbitrary_tests.BankFixture(root, 73, patcher)
    source_document = _source_action_manifest(fixture)
    source_path = fixture.capsule_directory / "configs" / "source.json"
    source_sha = _write_json(source_path, source_document)
    fixture.capsule["inputs"] = {
        "action_manifest": {
            "path": source_path.relative_to(
                fixture.capsule_directory
            ).as_posix(),
            "sha256": source_sha,
        }
    }
    fixture.write_capsule()
    loaded = fixture.load()
    template = arbitrary.load_canonical_motion_recipe(None)
    valid_ready_path = root / "ready" / "canonical_ready.npz"
    valid_ready_sha = registry_tests._write_ready(valid_ready_path)
    with np.load(valid_ready_path, allow_pickle=False) as ready_data:
        valid_ready = type(template.ready)(
            path=valid_ready_path,
            sha256=valid_ready_sha,
            joint_pos=np.asarray(ready_data["joint_pos"]),
            joint_vel=np.asarray(ready_data["joint_vel"]),
            root_pos_w=np.asarray(ready_data["root_pos_w"]),
            root_quat_wxyz=np.asarray(ready_data["root_quat_w"]),
            source_segment=str(ready_data["source_segment"].item()),
            source_frame=int(ready_data["source_frame"].item()),
        )
    template = replace(template, ready=valid_ready)
    patcher.setattr(
        arbitrary,
        "load_canonical_motion_recipe",
        lambda *args, **kwargs: template,
    )
    fixture.recipe["shared_ready"]["canonical_ready"] = fixture._binding(
        valid_ready_path
    )
    fixture.write_recipe()
    loaded = fixture.load()

    compiled = root / "compiled"
    build_document = _build_synthetic_compiled_bank(
        loaded=loaded,
        compiled=compiled,
    )
    registry_path, registry = _build_registry(
        root=root,
        loaded=loaded,
        compiled=compiled,
        build_document=build_document,
        scope="upper",
    )
    prototype_path, prototype_document = _build_prototype(
        root=root,
        registry=registry,
        compiled=compiled,
        build_document=build_document,
    )
    profile_pins_path = _build_profile_pins(root)
    report_path, report_document = _build_report(
        root=root,
        loaded=loaded,
        registry=registry,
        compiled=compiled,
        build_document=build_document,
    )
    _copy_runtime_contract_files(root)
    producer = (
        root
        / "hope_training/whole_body_tracking/scripts/"
        "materialize_arbitrary_action_ball_manifest.py"
    )
    producer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(materializer.__file__), producer)
    patcher.setattr(materializer, "__file__", str(producer))
    result = N73Fixture()
    result.root = root
    result.fixture = fixture
    result.loaded = loaded
    result.compiled = compiled
    result.registry_path = registry_path
    result.registry = registry
    result.prototype_path = prototype_path
    result.prototype_document = prototype_document
    result.profile_pins_path = profile_pins_path
    result.report_path = report_path
    result.report_document = report_document
    result.source_document = source_document
    try:
        yield result
    finally:
        patcher.undo()


def _kwargs(fixture: N73Fixture, *, tag: str) -> dict:
    return {
        "repo_root": fixture.root,
        "bank_report_path": fixture.report_path,
        "expected_bank_report_sha256": _sha(fixture.report_path),
        "registry_path": fixture.registry_path,
        "expected_registry_sha256": _sha(fixture.registry_path),
        "scope": "upper",
        "profile_pins_path": fixture.profile_pins_path,
        "expected_profile_pins_sha256": _sha(
            fixture.profile_pins_path
        ),
        "prototype_path": fixture.prototype_path,
        "expected_prototype_sha256": _sha(fixture.prototype_path),
        "manifest_id": f"synthetic_n73_{tag}",
        "output_path": fixture.root / "outputs" / f"{tag}.json",
        "receipt_output_path": (
            fixture.root / "outputs" / f"{tag}.receipt.json"
        ),
    }


def test_synthetic_n73_materializes_exact_compiled_order_and_heldout(
    n73_fixture,
):
    args = _kwargs(n73_fixture, tag="success")
    receipt = materializer.materialize(**args)
    manifest = json.loads(Path(args["output_path"]).read_text())

    assert manifest["schema_version"] == 3
    assert manifest["action_order"] == list(n73_fixture.loaded.motion_ids)
    assert len(manifest["actions"]) == 73
    assert manifest["holdout"]["samples_per_action"] == 768
    assert receipt["inputs"]["source_action_manifest"]["sha256"] == _sha(
        n73_fixture.fixture.capsule_directory
        / "configs"
        / "source.json"
    )
    for index in (0, 36, 72):
        output = manifest["actions"][index]
        source = n73_fixture.source_document["actions"][index]
        registry_row = n73_fixture.registry.entries[index]
        assert output["action_id"] == n73_fixture.loaded.motion_ids[index]
        assert output["motion_path"] == registry_row.npz_path_text
        assert output["motion_sha256"] == registry_row.npz_sha256
        assert output["motion_sha256"] != source["motion_sha256"]
        assert output["action_uid"] != source["action_uid"]
        assert (
            output["ball_profile"]["incoming_speed_center_mps"]
            == source["ball_profile"]["incoming_speed_center_mps"]
        )


@pytest.mark.parametrize("index", [0, 36, 72])
def test_first_middle_last_report_identity_drift_fails_closed(
    n73_fixture,
    index,
):
    report = deepcopy(n73_fixture.report_document)
    report["selected_registry_binding"]["motion_ids"][index] = (
        f"drifted_{index:03d}"
    )
    report_path = (
        n73_fixture.root / "reports" / f"drift_{index:03d}.json"
    )
    _write_json(report_path, report)
    args = _kwargs(n73_fixture, tag=f"drift_{index:03d}")
    args["bank_report_path"] = report_path
    args["expected_bank_report_sha256"] = _sha(report_path)

    with pytest.raises(
        materializer.MaterializationError,
        match="selected_registry_binding",
    ):
        materializer.materialize(**args)
    assert not Path(args["output_path"]).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document, fixture: document["scopes"]["upper"].__setitem__(
                slice(None), document["scopes"]["upper"][:5]
            ),
            "prototype",
        ),
        (
            lambda document, fixture: document["scopes"]["upper"][0].__setitem__(
                "npz_sha256",
                fixture.source_document["actions"][0]["motion_sha256"],
            ),
            "NPZ SHA",
        ),
        (
            lambda document, fixture: document["scopes"]["upper"].__setitem__(
                slice(0, 2),
                list(reversed(document["scopes"]["upper"][:2])),
            ),
            "clips",
        ),
        (
            lambda document, fixture: document["scopes"]["upper"].append(
                {
                    **document["scopes"]["upper"][-1],
                    "clip_index": 73,
                }
            ),
            "prototype",
        ),
    ],
)
def test_prototype_rejects_five_rows_raw_sha_order_and_extra(
    n73_fixture,
    mutation,
    message,
):
    document = deepcopy(n73_fixture.prototype_document)
    mutation(document, n73_fixture)
    document["derived_sha256"] = hashlib.sha256(
        json.dumps(
            document["scopes"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    tag = f"prototype_bad_{abs(hash(message + str(len(document['scopes']['upper']))))}"
    path = n73_fixture.root / "configs" / f"{tag}.json"
    _write_json(path, document)
    args = _kwargs(n73_fixture, tag=tag)
    args["prototype_path"] = path
    args["expected_prototype_sha256"] = _sha(path)

    with pytest.raises(
        materializer.MaterializationError,
        match=message,
    ):
        materializer.materialize(**args)
    assert not Path(args["output_path"]).exists()


def test_raw_source_path_or_sha_is_rejected_directly(tmp_path):
    source = tmp_path / "raw.npz"
    compiled = tmp_path / "compiled.npz"
    source.write_bytes(b"raw")
    compiled.write_bytes(b"compiled")

    with pytest.raises(
        materializer.MaterializationError,
        match="raw source motion",
    ):
        materializer._assert_compiled_identity(
            source_path=source,
            source_manifest_path="raw.npz",
            source_sha256="a" * 64,
            compiled_path=source,
            compiled_manifest_path="compiled.npz",
            compiled_sha256="b" * 64,
            label="path",
        )
    with pytest.raises(
        materializer.MaterializationError,
        match="raw source motion",
    ):
        materializer._assert_compiled_identity(
            source_path=source,
            source_manifest_path="raw.npz",
            source_sha256="a" * 64,
            compiled_path=compiled,
            compiled_manifest_path="compiled.npz",
            compiled_sha256="a" * 64,
            label="sha",
        )


def test_manifest_receipt_pair_is_no_clobber(tmp_path):
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    materializer._publish_pair_no_clobber(
        manifest_path=manifest,
        manifest_bytes=b"manifest",
        receipt_path=receipt,
        receipt_bytes=b"receipt",
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materializer._publish_pair_no_clobber(
            manifest_path=manifest,
            manifest_bytes=b"new",
            receipt_path=tmp_path / "new-receipt.json",
            receipt_bytes=b"new",
        )
    assert manifest.read_bytes() == b"manifest"
    assert receipt.read_bytes() == b"receipt"
