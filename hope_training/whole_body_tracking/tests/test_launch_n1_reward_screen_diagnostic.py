from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_n1_reward_screen_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("n1_diag_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write(path: Path, value) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else _canonical(value)
    path.write_bytes(raw)
    return {
        "path": "",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _converter_config(urdf_path: Path) -> dict:
    """A UrdfConverterCfg dump shaped like the one IsaacLab leaves in a bundle."""

    return {
        "asset_path": str(urdf_path),
        "usd_dir": None,
        "usd_file_name": None,
        "force_usd_conversion": False,
        "make_instanceable": True,
        "fix_base": False,
        "link_density": 0.0,
        "merge_fixed_joints": True,
        "collider_type": "convex_hull",
        "self_collision": False,
        "replace_cylinders_with_capsules": True,
        "activate_contact_sensors": True,
    }


def _isaaclab_asset_hash(config: dict, urdf_bytes: bytes) -> str:
    """The upstream ``.asset_hash`` recipe, spelled out once for the fixtures.

    isaaclab/sim/converters/asset_converter_base.py::_config_to_hash — MD5 over
    ``json.dumps`` of the converter config minus the three path keys, then over
    the source asset bytes.
    """

    payload = dict(config)
    for key in ("asset_path", "usd_dir", "usd_file_name"):
        payload.pop(key, None)
    digest = hashlib.md5()
    digest.update(json.dumps(payload).encode())
    digest.update(urdf_bytes)
    return digest.hexdigest()


def _write_usd_bundle(
    bundle_root: Path, *, urdf_path: Path, urdf_bytes: bytes, config: dict | None = None
) -> dict[str, str]:
    """Write a plausible pre-converted USD bundle and return its six SHAs.

    Only ``config.yaml`` and ``.asset_hash`` carry meaning; the USD payloads are
    opaque bytes, exactly as the gate treats them.
    """

    config = _converter_config(urdf_path) if config is None else config
    contents = {
        "config.yaml": yaml.safe_dump(config, sort_keys=False).encode("utf-8"),
        ".asset_hash": _isaaclab_asset_hash(config, urdf_bytes).encode("utf-8"),
        "model.usd": b"fixture root usd\n",
        "configuration/model_base.usd": b"fixture base usd\n",
        "configuration/model_physics.usd": b"fixture physics usd\n",
        "configuration/model_sensor.usd": b"fixture sensor usd\n",
    }
    hashes: dict[str, str] = {}
    for relative, raw in contents.items():
        path = bundle_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    return hashes


def _plant_asset_relative(asset_root_name: str) -> str:
    return (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        f"whole_body_tracking/assets/{asset_root_name}"
    )


def _robot_source_text(asset_root_name: str) -> str:
    return (
        "# exact legacy robot fixture\n"
        f'AGIBOT_A3_ASSET_ROOT = f"{{ASSET_DIR}}/{asset_root_name}"\n'
        'AGIBOT_A3_URDF_PATH = f"{AGIBOT_A3_ASSET_ROOT}/urdf/model.urdf"\n'
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def exact_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    launcher_path = repo / L.LAUNCHER_SOURCE
    train_path = repo / L.TRAIN_SOURCE
    task_path = repo / L.TASK_SOURCE
    robot_path = repo / L.LEGACY_ROBOT_SOURCE
    kit_path = repo / L.KIT_LAUNCHER_SOURCE
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    train_path.write_text("# train\n", encoding="utf-8")
    task_path.write_text("name: test\n", encoding="utf-8")
    robot_path.write_text(
        _robot_source_text(L.A3_PLANT_ASSET_ROOT_NAME), encoding="utf-8"
    )
    monkeypatch.setattr(
        L,
        "LEGACY_ROBOT_SOURCE_SHA256",
        hashlib.sha256(robot_path.read_bytes()).hexdigest(),
    )
    kit_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher_path.chmod(0o755)
    kit_path.chmod(0o755)

    # The plant closure the runtime USD cache has to belong to: the URDF the
    # live spawner names, and the checkout's receipt for it.
    plant_asset_relative = _plant_asset_relative(L.A3_PLANT_ASSET_ROOT_NAME)
    plant_urdf_relative = "urdf/model.urdf"
    plant_urdf = repo / plant_asset_relative / plant_urdf_relative
    plant_urdf.parent.mkdir(parents=True, exist_ok=True)
    plant_urdf_bytes = b"<robot name='fixture_a3p_p1_0807'/>\n"
    plant_urdf.write_bytes(plant_urdf_bytes)
    plant_urdf_sha = hashlib.sha256(plant_urdf_bytes).hexdigest()
    monkeypatch.setattr(L, "A3_PLANT_SOURCE_URDF_SHA256", plant_urdf_sha)
    plant_receipt_path = repo / L.A3_PLANT_RECEIPT_RELATIVE
    plant_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    plant_receipt = {
        "schema_version": 1,
        "manifest_type": L.A3_PLANT_RECEIPT_MANIFEST_TYPE,
        "isaac": {
            "asset_path": plant_asset_relative,
            "urdf_path": plant_urdf_relative,
            "urdf_sha256": plant_urdf_sha,
        },
    }
    plant_receipt_path.write_bytes(_canonical(plant_receipt))

    runtime_root = tmp_path / "runtime_assets"
    usd_root = runtime_root / "a3_preconverted_usd"
    usd_hashes = _write_usd_bundle(
        usd_root, urdf_path=plant_urdf, urdf_bytes=plant_urdf_bytes
    )
    assert set(usd_hashes) == set(L.A3_RUNTIME_USD_BUNDLE_SHA256)
    opengl_root = runtime_root / "private_opengl"
    opengl_root.mkdir(parents=True)
    opengl_library = opengl_root / L.PRIVATE_OPENGL_LIBRARY
    opengl_library.write_bytes(b"fixture private OpenGL\n")
    (opengl_root / L.PRIVATE_OPENGL_SONAME).symlink_to(
        L.PRIVATE_OPENGL_LIBRARY
    )
    glu_root = runtime_root / "private_glu"
    glu_root.mkdir(parents=True)
    glu_library = glu_root / L.PRIVATE_GLU_LIBRARY
    glu_library.write_bytes(b"fixture private GLU\n")
    (glu_root / L.PRIVATE_GLU_SONAME).symlink_to(L.PRIVATE_GLU_LIBRARY)
    monkeypatch.setattr(L, "A3_RUNTIME_USD_BUNDLE_SHA256", usd_hashes)
    monkeypatch.setattr(L, "PRIVATE_OPENGL_DIRECTORY", str(opengl_root))
    monkeypatch.setattr(
        L,
        "PRIVATE_OPENGL_SHA256",
        hashlib.sha256(opengl_library.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(L, "PRIVATE_GLU_DIRECTORY", str(glu_root))
    monkeypatch.setattr(
        L,
        "PRIVATE_GLU_SHA256",
        hashlib.sha256(glu_library.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("HOPE_URDF_IMPORTER_NO_UI", "1")
    monkeypatch.setenv(
        "HOPE_AGIBOT_A3_USD_PATH", str(usd_root / "model.usd")
    )
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", f"{opengl_root}{L.os.pathsep}{glu_root}"
    )

    def add_json(relative: str, value):
        path = repo / relative
        pin = _write(path, value)
        pin["path"] = relative
        return pin

    source_manifest = add_json(
        "configs/source_n5.json", {"schema_version": 3, "source": "fixture"}
    )
    source_hashes = {}
    for filename in L.SOLVER_IMPLEMENTATION_SOURCES:
        relative = f"{L.MDP_RELATIVE}/{filename}"
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"# exact fixture source: {filename}\n".encode("utf-8")
        path.write_bytes(raw)
        source_hashes[filename] = hashlib.sha256(raw).hexdigest()
    motion_relative = "motions/bh_loop_c_upper.npz"
    motion_path = repo / motion_relative
    motion = _write(motion_path, b"exact-schema2-motion")
    motion["path"] = motion_relative
    dynamic_ready_document = {
        "schema_version": 1,
        "kind": L.DYNAMIC_READY_KIND,
        "action_id": "bh_loop_c",
        "robot": {
            "family": "AgiBot A3",
            "joint_names": [f"joint_{index}" for index in range(31)],
        },
        "sources": {
            "stable_motion": {
                "path": str(motion_path),
                "sha256": motion["sha256"],
                "frame_index": 0,
            }
        },
        "required_next_gate": {
            "kind": L.NOMINAL_HOLD_RECEIPT_KIND,
            "zero_terminal_required": [
                "joint_qdes_forbidden",
                "joint_actual_forbidden",
                "robot_hit_table",
                "base_fell_tilt",
                "base_too_low",
            ],
        },
    }
    dynamic_ready_document["content_sha256"] = (
        L._canonical_ascii_sha256(dynamic_ready_document)
    )
    dynamic_ready = add_json(
        "configs/bh_loop_c.dynamic_ready.v1.json",
        dynamic_ready_document,
    )
    nominal_hold_document = {
        "schema_version": 1,
        "kind": L.NOMINAL_HOLD_RECEIPT_KIND,
        "verdict": "PASS",
        "action_id": "bh_loop_c",
        "artifact": {
            "path": str(repo / dynamic_ready["path"]),
            "sha256": dynamic_ready["sha256"],
            "content_sha256": dynamic_ready_document[
                "content_sha256"
            ],
        },
        "motion_sha256": motion["sha256"],
        "plant_contract_match": True,
        "active_terminations": [
            "time_out",
            "base_fell_tilt",
            "base_too_low",
            "robot_hit_table",
            "joint_qdes_forbidden",
            "joint_actual_forbidden",
        ],
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
    }
    nominal_hold_document["content_sha256"] = L.canonical_sha256(
        nominal_hold_document
    )
    nominal_hold = add_json(
        "configs/bh_loop_c.nominal_hold.v1.json",
        nominal_hold_document,
    )

    contact_geometry_payload = {
        "schema_version": 2,
        "kind": "exact_face_contact_v2",
        "fixture": True,
    }
    contact_geometry_sha = hashlib.sha256(
        json.dumps(
            contact_geometry_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    contact_geometry = {
        "payload": contact_geometry_payload,
        "sha256": contact_geometry_sha,
    }
    geometry = {
        "path": L.CONTACT_GEOMETRY_SOURCE,
        "sha256": source_hashes["racket_contact_geometry.py"],
        "payload_sha256": contact_geometry_sha,
        "kind": "exact_face_contact_v2",
    }
    # Solver profile v3: the payload seals a per-symbol semantic surface and the
    # document publishes the same surface SHA next to the byte map.
    semantic_surface = {
        "kind": "whole_body_tracking.action_ball.solver_semantic_surface",
        "schema_version": 1,
        "sha256": hashlib.sha256(b"fixture-semantic-surface").hexdigest(),
        "covered_symbol_count": 3,
        "pinned_sources": ["hope_commands.py"],
    }
    solver_payload = {
        "solver": "fixture",
        "semantic_surface": semantic_surface,
        "contact_geometry": contact_geometry,
    }
    physics_payload = {"physics": "fixture"}
    table_geometry_payload = {"geometry": "fixture"}
    solver_profile_sha = hashlib.sha256(
        json.dumps(
            solver_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    physics_profile_sha = hashlib.sha256(
        json.dumps(
            physics_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    profile_document = {
        "schema_version": 1,
        "solver_payload": solver_payload,
        "physics_payload": physics_payload,
        "geometry": table_geometry_payload,
        "contact_geometry": contact_geometry,
        "solver_implementation_source_sha256": source_hashes,
        "solver_semantic_surface": {"sha256": semantic_surface["sha256"]},
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
    }
    profile = add_json("configs/profile_pins.json", profile_document)
    profile.update(
        {
            "solver_profile_sha256": solver_profile_sha,
            "physics_profile_sha256": physics_profile_sha,
            "geometry_payload_sha256": contact_geometry_sha,
        }
    )
    prototype_document = {
        "schema_version": 2,
        "scopes": {"upper": [{"motion_id": "bh_loop_c"}]},
    }
    prototype = add_json(
        "configs/bh_loop_c.upper.prototype.v2.json",
        prototype_document,
    )
    prototype.update({"schema_version": 2, "scope": "upper"})

    action_uid = 1722317591841513
    contact_center = [0.70, -0.03, 0.40]
    manifest_document = {
        "schema_version": 3,
        "manifest_id": "fixture_n1",
        "mobility_mode": "no_move",
        "action_order": ["bh_loop_c"],
        "prototype": {
            "path": prototype["path"],
            "sha256": prototype["sha256"],
            "scope": "upper",
        },
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
        "actions": [
            {
                "action_id": "bh_loop_c",
                "action_uid": action_uid,
                "motion_path": motion["path"],
                "motion_sha256": motion["sha256"],
                "reference_t_hit_s": 0.62,
                "reference_t_cycle_s": 1.4,
                "ball_profile": {
                    "contact_offset_center_b_yaw_m": contact_center
                },
            }
        ],
        "holdout": {
            "seed": 20260729,
            "samples_per_action": 768,
            "split_id": "fixture",
        },
        "counter_rally_objective": {"mode": "counter_rally_v1"},
    }
    manifest = add_json(
        "configs/bh_loop_c.manifest.v3.json", manifest_document
    )
    manifest.update(
        {"schema_version": 3, "action_order": ["bh_loop_c"]}
    )

    claims = {
        "selector_executed": False,
        "action_identity_frozen_before_ball_sampling": True,
        "contact_alignment_claim": True,
        "landing_claim": False,
        "post_bounce_claim": False,
        "baseline_crossing_claim": False,
        "deployment_claim": False,
    }
    contact_document = {
        "schema_version": 1,
        "artifact_type": L.CONTACT_KIND,
        "status": "PASS",
        "action_id": "bh_loop_c",
        "action_uid": action_uid,
        "scope": "upper",
        "source_manifest": source_manifest,
        "motion": motion,
        "profile_pins": profile,
        "geometry": geometry,
        "timing": {
            "fps_hz": 50.0,
            "frame_count": 71,
            "contact_frame": 31,
            "manifest_t_hit_s": 0.62,
            "motion_t_hit_s": 0.62,
            "manifest_t_cycle_s": 1.4,
            "motion_t_cycle_s": 1.4,
            "t_hit_abs_error_s": 0.0,
            "t_cycle_abs_error_s": 0.0,
        },
        "frames": {
            "task_contact_frame": "B_yaw_relative_to_actual_spawn_goal",
            "teacher_reference_frame": "B_yaw_at_frame0",
            "world_z_origin": "floor",
        },
        "alignment": {
            "threshold_m": 0.03,
            "ready_root_z_w_m": 0.90,
            "legacy_absolute_contact_z_w_m": 1.30,
            "corrected_contact_offset_z_b_yaw_m": 0.40,
            "task_contact_offset_center_b_yaw_m": contact_center,
            "teacher_racket_site_b_yaw_m": [0.60, -0.03, 0.40],
            "teacher_selected_face_center_b_yaw_m": [
                0.70,
                -0.03,
                0.41,
            ],
            "task_to_teacher_site_distance_m": 0.10,
            "task_to_teacher_face_center_distance_m": 0.01,
            "center_gate_point": "selected_rubber_face_center",
            "center_gate_distance_m": 0.01,
            "center_within_threshold": True,
        },
        "claims": claims,
    }
    contact = add_json(
        "configs/bh_loop_c.contact_alignment.v1.json",
        contact_document,
    )
    contact.update({"schema_version": 1, "status": "PASS"})
    bundle_document = {
        "schema_version": 2,
        "artifact_type": L.BUNDLE_KIND,
        "action_id": "bh_loop_c",
        "action_uid": action_uid,
        "scope": "upper",
        "source_manifest": source_manifest,
        "motion": motion,
        "profile_pins": profile,
        "prototype": prototype,
        "manifest": manifest,
        "contact_alignment": contact,
        "dynamic_ready": {
            "artifact": dynamic_ready,
            "nominal_hold_receipt": nominal_hold,
        },
        "geometry": geometry,
        "claims": claims,
    }
    bundle = add_json(
        "configs/bh_loop_c.bundle.v2.json", bundle_document
    )

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(L, "__file__", str(launcher_path))

    run_root = tmp_path / "runs"
    run_root.mkdir()
    spec_path = tmp_path / "spec.json"

    def make_spec(
        *,
        profile_name: str = "current_low",
        stage: str = "smoke",
    ):
        namespace = run_root / f"bh-loop-{profile_name}-{stage}"
        if stage == "smoke":
            num_envs, iterations, save = 1, 2, 1
        elif stage == "probe":
            num_envs, iterations, save = 4096, 5, 1
        elif stage == "milestone1000":
            num_envs, iterations, save = 4096, 1001, 100
        elif stage == "long":
            num_envs, iterations, save = 4096, 20_001, 100
        else:
            num_envs, iterations, save = 64, 100, 20
        spec = {
            "schema_version": 1,
            "kind": L.SPEC_KIND,
            "source": {
                "checkout": str(repo),
                "commit_sha": commit,
                "isaac_python": str(Path(sys.executable).resolve()),
            },
            "action_id": "bh_loop_c",
            "scope": "upper",
            "bundle": bundle,
            "policy_contract_sha256": "4" * 64,
            "reward_profile": profile_name,
            "expected_effective_reward_recipe_sha256": "5" * 64,
            "seed": 7,
            "stage": stage,
            "num_envs": num_envs,
            "max_iterations": iterations,
            "save_interval": save,
            "gpu": {
                "index": 2,
                "uuid": "GPU-fixture",
                "owner": "Franco",
                "lock_path": "/tmp/hope_lean_queue_gpu2.lock",
                "require_empty": True,
            },
            "namespace": str(namespace),
            "log_path": str(namespace / "run.log"),
        }
        spec_path.write_bytes(_canonical(spec))
        return spec, spec_path

    return {
        "repo": repo,
        "commit": commit,
        "bundle": bundle,
        "bundle_document": bundle_document,
        "manifest_document": manifest_document,
        "contact_document": contact_document,
        "contact_path": repo / contact["path"],
        "dynamic_ready_path": repo / dynamic_ready["path"],
        "nominal_hold_path": repo / nominal_hold["path"],
        "opengl_root": opengl_root,
        "opengl_library": opengl_library,
        "glu_root": glu_root,
        "glu_library": glu_library,
        "make_spec": make_spec,
        "runtime_root": runtime_root,
        "usd_root": usd_root,
        "robot_path": robot_path,
        "plant_urdf": plant_urdf,
        "plant_urdf_bytes": plant_urdf_bytes,
        "plant_urdf_sha256": plant_urdf_sha,
        "plant_asset_relative": plant_asset_relative,
        "plant_urdf_relative": plant_urdf_relative,
        "plant_receipt_path": plant_receipt_path,
        "plant_receipt": plant_receipt,
    }


def _convert_fixture_to_full(exact_repo, spec: dict) -> Path:
    repo = exact_repo["repo"]
    bundle_path = repo / spec["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text())

    prototype_path = repo / bundle["prototype"]["path"]
    prototype = json.loads(prototype_path.read_text())
    prototype["scopes"] = {"full": prototype["scopes"].pop("upper")}
    prototype["provenance"] = {
        "full_solver_admission_preflight": {
            "schema_version": 1,
            "kind": "full_fixed_action_exact_solver_admission_preflight_v1",
            "proposal_count": 512,
            "admitted_count": 511,
            "rejected_count": 1,
            "admit_rate": 511 / 512,
            "diagnostic_gate": {
                "status": "PASS",
                "environment_count": 4096,
                "runtime_per_birth_redraw_replay": False,
                "zero_admission_canary_group_count": 0,
            },
        }
    }
    prototype_path.write_bytes(_canonical(prototype))
    prototype_sha = hashlib.sha256(prototype_path.read_bytes()).hexdigest()
    bundle["prototype"]["scope"] = "full"
    bundle["prototype"]["sha256"] = prototype_sha

    manifest_path = repo / bundle["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["prototype"]["scope"] = "full"
    manifest["prototype"]["sha256"] = prototype_sha
    manifest_path.write_bytes(_canonical(manifest))
    bundle["manifest"]["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    full_claims = dict(bundle["claims"])
    full_claims.update(
        {"diagnostic_only": True, "training_authorized": False}
    )
    contact_path = repo / bundle["contact_alignment"]["path"]
    contact = json.loads(contact_path.read_text())
    contact["scope"] = "full"
    contact["claims"] = full_claims
    alignment = contact["alignment"]
    alignment.pop("legacy_absolute_contact_z_w_m")
    alignment.pop("corrected_contact_offset_z_b_yaw_m")
    alignment.update(
        {
            "contact_center_authority": (
                "full_motion_selected_rubber_face_center_at_explicit_strike_frame"
            ),
            "retargeted_contact_center_z_w_m": (
                alignment["ready_root_z_w_m"]
                + alignment["task_contact_offset_center_b_yaw_m"][2]
            ),
            "upper_contact_center_preserved": False,
        }
    )
    contact_path.write_bytes(_canonical(contact))
    bundle["contact_alignment"]["sha256"] = hashlib.sha256(
        contact_path.read_bytes()
    ).hexdigest()

    bundle["scope"] = "full"
    bundle["claims"] = full_claims
    bundle_path.write_bytes(_canonical(bundle))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "full diagnostic fixture")

    spec["scope"] = "full"
    spec["bundle"]["sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()
    spec["source"]["commit_sha"] = _git(repo, "rev-parse", "HEAD")
    spec_path = repo.parent / "full-spec.json"
    spec_path.write_bytes(_canonical(spec))
    return spec_path


def _convert_fixture_to_legacy_v1(exact_repo, spec: dict) -> Path:
    repo = exact_repo["repo"]
    bundle_path = repo / spec["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text())
    bundle["schema_version"] = 1
    bundle["artifact_type"] = L.BUNDLE_KIND_V1
    bundle.pop("dynamic_ready")
    bundle_path.write_bytes(_canonical(bundle))
    spec["bundle"]["sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "legacy bundle fixture")
    spec["source"]["commit_sha"] = _git(repo, "rev-parse", "HEAD")
    spec_path = repo.parent / "legacy-spec.json"
    spec_path.write_bytes(_canonical(spec))
    return spec_path


def test_old_launcher_refuses_a_clean_commit_with_new_robot_plant(
    exact_repo,
) -> None:
    spec, spec_path = exact_repo["make_spec"]()
    repo = exact_repo["repo"]
    robot_path = repo / L.LEGACY_ROBOT_SOURCE
    robot_path.write_text("# new vendor robot plant\n", encoding="utf-8")
    _git(repo, "add", L.LEGACY_ROBOT_SOURCE)
    _git(repo, "commit", "-qm", "switch global robot plant")
    spec["source"]["commit_sha"] = _git(repo, "rev-parse", "HEAD")
    spec_path.write_bytes(_canonical(spec))

    with pytest.raises(L.LaunchRefused, match="SHA differs"):
        L.build_plan(spec_path)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("current_low", ("4.0", "0.5", "0.5", "1.0")),
        ("mimic_x2", ("4.0", "0.5", "0.5", "2.0")),
        ("task_strong_x4", ("16.0", "2.0", "2.0", "1.0")),
    ],
)
def test_plan_binds_exact_three_reward_profiles_and_no_override_seam(
    exact_repo, profile, expected
):
    _spec, path = exact_repo["make_spec"](profile_name=profile)
    plan = L.build_plan(path)
    payload = plan["canonical_payload"]
    argv = payload["training_argv"]
    assert payload["diagnostic_unauthorized"] is True
    assert payload["formal_evidence_prohibited"] is True
    assert payload["curriculum_promotion_prohibited"] is True
    assert "task.racket.action_ball_diagnostic_unauthorized=true" in argv
    assert "task=HOPEPingPongActionBall" in argv
    assert "task=HOPEPingPongActionBallA3VendorV1" not in argv
    assert "+task.racket.reference_guard_mode=metrics_only" in argv
    assert (
        "task.actor_obs_contract="
        "action_ball_table_pose_twist_heading_task_teacher_start_v2"
        in argv
    )
    assert "algo.policy.init_noise_std=0.02" in argv
    assert "action_ball_dynamic_ready_bootstrap=true" in argv
    assert "+task.domain_rand.stable_ready_plant=true" in argv
    assert not any(token.startswith("task.push.") for token in argv)
    assert not any(token.startswith("+task.push.") for token in argv)
    assert "action_ball_shared_ready_bootstrap=true" not in argv
    dynamic_ready = payload["bundle"]["dynamic_ready"]
    checkout = Path(payload["spec"]["source"]["checkout"])
    assert (
        "action_ball_dynamic_ready_artifact_path="
        f"{checkout / dynamic_ready['artifact']['path']}"
    ) in argv
    assert (
        "action_ball_dynamic_ready_artifact_sha256="
        f"{dynamic_ready['artifact']['sha256']}"
    ) in argv
    assert (
        "action_ball_dynamic_ready_nominal_receipt_path="
        f"{checkout / dynamic_ready['nominal_hold_receipt']['path']}"
    ) in argv
    assert (
        "action_ball_dynamic_ready_nominal_receipt_sha256="
        f"{dynamic_ready['nominal_hold_receipt']['sha256']}"
    ) in argv
    assert "algo.policy.init_noise_std=0.15" not in argv
    assert "algo.policy.init_noise_std=1.0" not in argv
    assert "task.rewards.full_body_mimic=false" in argv
    assert f"+task.rewards.motion_scale={expected[3]}" in argv
    assert f"task.rewards.racket_position_weight={expected[0]}" in argv
    assert f"task.rewards.racket_velocity_weight={expected[1]}" in argv
    assert f"task.rewards.racket_normal_weight={expected[2]}" in argv
    assert payload["reward_weights"] == {
        "racket_position_weight": float(expected[0]),
        "racket_velocity_weight": float(expected[1]),
        "racket_normal_weight": float(expected[2]),
        "motion_scale": float(expected[3]),
    }
    assert not any("override" in token for token in argv)
    assert not Path(payload["spec"]["namespace"]).exists()
    assert (
        plan["launch_claim_sha256"]
        == L.canonical_sha256(plan["canonical_payload"])
    )


def test_plan_binds_complete_external_a3_runtime_asset_closure(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    payload = L.build_plan(path)["canonical_payload"]
    runtime = payload["runtime_assets"]

    assert runtime["schema_version"] == 2
    assert runtime["kind"] == "n1_a3_runtime_asset_pins_v2"
    assert runtime["integrity_model"] == L.RUNTIME_ASSET_INTEGRITY_MODEL
    assert runtime["loader_library_path"] == (
        f"{exact_repo['opengl_root']}{L.os.pathsep}{exact_repo['glu_root']}"
    )
    assert (
        runtime["private_opengl"]["soname_target"]
        == L.PRIVATE_OPENGL_LIBRARY
    )
    assert runtime["private_opengl"]["sha256"] == L.PRIVATE_OPENGL_SHA256
    assert runtime["urdf_importer_no_ui"] == "1"
    assert runtime["a3_preconverted_usd"]["path"].endswith("/model.usd")
    assert set(runtime["a3_preconverted_usd"]["files"]) == set(
        L.A3_RUNTIME_USD_BUNDLE_SHA256
    )
    assert runtime["private_glu"]["soname_target"] == L.PRIVATE_GLU_LIBRARY
    assert runtime["private_glu"]["sha256"] == L.PRIVATE_GLU_SHA256

    model = Path(runtime["a3_preconverted_usd"]["path"])
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(L.LaunchRefused, match="model.usd SHA differs"):
        L.build_plan(path)


def test_plan_receipt_states_which_plant_the_usd_cache_belongs_to(exact_repo):
    """The admitted case, and the receipt has to say what it compared."""

    _spec, path = exact_repo["make_spec"]()
    payload = L.build_plan(path)["canonical_payload"]
    identity = payload["runtime_assets"]["a3_preconverted_usd"]["plant_identity"]

    assert identity["kind"] == L.A3_PLANT_IDENTITY_KIND
    assert identity["compared"] == [
        "live_spawner_asset_root_vs_pin",
        "plant_receipt_manifest_type",
        "plant_receipt_asset_root_vs_live_spawner",
        "plant_receipt_urdf_sha256_vs_pin",
        "worktree_urdf_sha256_vs_plant_receipt",
        "bundle_config_asset_path_vs_plant_receipt",
        "bundle_isaaclab_asset_hash_vs_rederived_from_worktree_urdf",
    ]
    assert identity["checkout"] == str(exact_repo["repo"])
    assert identity["robot_source"] == L.LEGACY_ROBOT_SOURCE
    assert identity["live_spawner_asset_root"] == L.A3_PLANT_ASSET_ROOT_NAME
    assert identity["plant_receipt"] == L.A3_PLANT_RECEIPT_RELATIVE
    assert identity["source_urdf_relative"] == (
        f"{exact_repo['plant_asset_relative']}/{exact_repo['plant_urdf_relative']}"
    )
    assert identity["source_urdf_sha256"] == exact_repo["plant_urdf_sha256"]
    assert identity["bundle_config_asset_path"] == str(exact_repo["plant_urdf"])
    assert (
        identity["isaaclab_asset_hash"]
        == identity["isaaclab_asset_hash_rederived"]
    )
    assert (
        identity["isaaclab_asset_hash"]
        == (exact_repo["usd_root"] / ".asset_hash").read_text().strip()
    )


def test_usd_cache_of_the_retired_plant_is_refused_even_when_restamped(
    exact_repo, monkeypatch
):
    """The exact way this gate was fooled: convert the OLD robot, re-stamp all six.

    Two flavours, because only the second one shows the byte pin is not doing
    the work.  In (a) the bundle honestly records the retired asset package.  In
    (b) the recorded path is doctored to name the current package, so every
    string in the bundle agrees with the checkout — and it is still refused,
    because the cache's own IsaacLab digest cannot be re-derived from this
    plant's URDF.
    """

    # The retired plant lives outside the launched checkout, exactly as the real
    # one did: the bundle was cut in a checkout that is long gone, and the gate
    # never opens the path config.yaml records.
    retired_urdf_bytes = b"<robot name='fixture_retired_0409'/>\n"
    retired_urdf = (
        exact_repo["runtime_root"]
        / "retired_checkout"
        / _plant_asset_relative("agibot_a3")
        / exact_repo["plant_urdf_relative"]
    )
    retired_urdf.parent.mkdir(parents=True, exist_ok=True)
    retired_urdf.write_bytes(retired_urdf_bytes)

    # (a) honest bundle for the retired robot, six hashes freshly stamped.
    honest_root = exact_repo["runtime_root"] / "retired_usd_honest"
    honest_hashes = _write_usd_bundle(
        honest_root, urdf_path=retired_urdf, urdf_bytes=retired_urdf_bytes
    )
    monkeypatch.setattr(L, "A3_RUNTIME_USD_BUNDLE_SHA256", honest_hashes)
    monkeypatch.setenv(
        "HOPE_AGIBOT_A3_USD_PATH", str(honest_root / "model.usd")
    )
    _spec, path = exact_repo["make_spec"]()
    with pytest.raises(
        L.LaunchRefused, match="converted from a different robot"
    ):
        L.build_plan(path)

    # (b) same retired cache, but config.yaml now claims the current plant.
    doctored_config = _converter_config(exact_repo["plant_urdf"])
    doctored_root = exact_repo["runtime_root"] / "retired_usd_doctored"
    doctored_hashes = _write_usd_bundle(
        doctored_root,
        urdf_path=exact_repo["plant_urdf"],
        urdf_bytes=retired_urdf_bytes,
        config=doctored_config,
    )
    monkeypatch.setattr(L, "A3_RUNTIME_USD_BUNDLE_SHA256", doctored_hashes)
    monkeypatch.setenv(
        "HOPE_AGIBOT_A3_USD_PATH", str(doctored_root / "model.usd")
    )
    _spec, path = exact_repo["make_spec"]()
    with pytest.raises(
        L.LaunchRefused, match="was not converted from this plant"
    ):
        L.build_plan(path)


def test_usd_cache_byte_tamper_is_still_refused_beside_the_identity_check(
    exact_repo,
):
    """The old strength has to survive the new one: bytes still get re-hashed."""

    _spec, path = exact_repo["make_spec"]()
    L.build_plan(path)

    base = exact_repo["usd_root"] / "configuration/model_base.usd"
    base.write_bytes(base.read_bytes() + b"tamper")
    with pytest.raises(
        L.LaunchRefused, match="model_base.usd SHA differs"
    ):
        L.build_plan(path)


def test_plant_pointer_moved_without_recutting_the_cache_is_refused(
    exact_repo, monkeypatch
):
    """Move the plant in the live spawner only; the cache is now stale."""

    _spec, path = exact_repo["make_spec"]()
    L.build_plan(path)

    exact_repo["robot_path"].write_text(
        _robot_source_text("agibot_a3p_p1_0901_v9"), encoding="utf-8"
    )
    # Let the unrelated historical-robot byte pin follow the edit, so this test
    # reaches the plant check instead of stopping one gate earlier.
    monkeypatch.setattr(
        L,
        "LEGACY_ROBOT_SOURCE_SHA256",
        hashlib.sha256(exact_repo["robot_path"].read_bytes()).hexdigest(),
    )
    _git(exact_repo["repo"], "add", L.LEGACY_ROBOT_SOURCE)
    _git(exact_repo["repo"], "commit", "-qm", "move the plant pointer")
    spec, path = exact_repo["make_spec"]()
    spec["source"]["commit_sha"] = _git(exact_repo["repo"], "rev-parse", "HEAD")
    path.write_bytes(_canonical(spec))

    with pytest.raises(
        L.LaunchRefused, match="plant pointer moved without re-cutting"
    ):
        L.build_plan(path)


def test_plant_urdf_that_drifts_from_its_own_receipt_is_refused(exact_repo):
    """The receipt is not taken on its word; the URDF on disk is re-hashed."""

    _spec, path = exact_repo["make_spec"]()
    L.build_plan(path)

    exact_repo["plant_urdf"].write_bytes(b"<robot name='silently_edited'/>\n")
    _git(exact_repo["repo"], "add", ".")
    _git(exact_repo["repo"], "commit", "-qm", "edit the plant URDF in place")
    spec, path = exact_repo["make_spec"]()
    spec["source"]["commit_sha"] = _git(exact_repo["repo"], "rev-parse", "HEAD")
    path.write_bytes(_canonical(spec))

    with pytest.raises(
        L.LaunchRefused, match="differs from its own receipt"
    ):
        L.build_plan(path)


def test_plant_receipt_that_is_not_the_reviewed_model_set_is_refused(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    L.build_plan(path)

    receipt = copy.deepcopy(exact_repo["plant_receipt"])
    receipt["manifest_type"] = "some_other_model_set_v9"
    exact_repo["plant_receipt_path"].write_bytes(_canonical(receipt))
    _git(exact_repo["repo"], "add", ".")
    _git(exact_repo["repo"], "commit", "-qm", "swap the plant receipt kind")
    spec, path = exact_repo["make_spec"]()
    spec["source"]["commit_sha"] = _git(exact_repo["repo"], "rev-parse", "HEAD")
    path.write_bytes(_canonical(spec))

    with pytest.raises(
        L.LaunchRefused, match="not the reviewed dual-engine model set"
    ):
        L.build_plan(path)


def test_plant_receipt_urdf_digest_must_match_the_pin_the_cache_was_cut_against(
    exact_repo,
):
    """A new plant with a new URDF cannot ride the old cache's pin."""

    _spec, path = exact_repo["make_spec"]()
    L.build_plan(path)

    successor_bytes = b"<robot name='fixture_a3p_p1_next'/>\n"
    exact_repo["plant_urdf"].write_bytes(successor_bytes)
    receipt = copy.deepcopy(exact_repo["plant_receipt"])
    receipt["isaac"]["urdf_sha256"] = hashlib.sha256(successor_bytes).hexdigest()
    exact_repo["plant_receipt_path"].write_bytes(_canonical(receipt))
    _git(exact_repo["repo"], "add", ".")
    _git(exact_repo["repo"], "commit", "-qm", "land a successor plant")
    spec, path = exact_repo["make_spec"]()
    spec["source"]["commit_sha"] = _git(exact_repo["repo"], "rev-parse", "HEAD")
    path.write_bytes(_canonical(spec))

    with pytest.raises(
        L.LaunchRefused, match="moved without re-converting the USD cache"
    ):
        L.build_plan(path)


def test_isaaclab_asset_hash_recipe_matches_the_upstream_three_dropped_keys():
    """The derivation proof only works if it mirrors IsaacLab exactly."""

    assert L.A3_ASSET_HASH_EXCLUDED_CONFIG_KEYS == (
        "asset_path",
        "usd_dir",
        "usd_file_name",
    )
    urdf_bytes = b"<robot name='x'/>\n"
    config = _converter_config(Path("/somewhere/urdf/model.urdf"))
    expected = _isaaclab_asset_hash(config, urdf_bytes)
    # The two path keys that are dropped must not move the digest, and the
    # payload bytes must.
    moved = dict(config)
    moved["asset_path"] = "/elsewhere/urdf/model.urdf"
    assert _isaaclab_asset_hash(moved, urdf_bytes) == expected
    assert _isaaclab_asset_hash(config, urdf_bytes + b" ") != expected
    coarser = dict(config)
    coarser["merge_fixed_joints"] = not config["merge_fixed_joints"]
    assert _isaaclab_asset_hash(coarser, urdf_bytes) != expected


def test_runtime_asset_open_gl_tamper_and_soname_drift_fail_closed(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    runtime = L.build_plan(path)["canonical_payload"]["runtime_assets"]

    exact_repo["opengl_library"].write_bytes(b"tampered OpenGL\n")
    with pytest.raises(L.LaunchRefused, match="OpenGL library SHA differs"):
        L._validate_runtime_asset_claim(runtime, checkout=exact_repo["repo"])

    exact_repo["opengl_library"].write_bytes(b"fixture private OpenGL\n")
    soname = exact_repo["opengl_root"] / L.PRIVATE_OPENGL_SONAME
    soname.unlink()
    (exact_repo["opengl_root"] / "wrong-library.so").write_bytes(b"wrong\n")
    soname.symlink_to("wrong-library.so")
    with pytest.raises(L.LaunchRefused, match="OpenGL soname"):
        L._validate_runtime_asset_claim(runtime, checkout=exact_repo["repo"])


@pytest.mark.parametrize("mode", ("missing", "reversed", "tail"))
def test_runtime_asset_loader_path_is_exact_not_inherited(
    exact_repo, monkeypatch, mode
):
    _spec, path = exact_repo["make_spec"]()
    opengl = str(exact_repo["opengl_root"])
    glu = str(exact_repo["glu_root"])
    values = {
        "missing": None,
        "reversed": f"{glu}{L.os.pathsep}{opengl}",
        "tail": f"{opengl}{L.os.pathsep}{glu}{L.os.pathsep}/usr/lib",
    }
    value = values[mode]
    if value is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", value)

    with pytest.raises(L.LaunchRefused, match="must equal"):
        L.build_plan(path)
    assert not Path(json.loads(path.read_text())["namespace"]).exists()


def test_runtime_asset_old_v1_or_missing_opengl_claim_is_refused(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    runtime = L.build_plan(path)["canonical_payload"]["runtime_assets"]
    old = copy.deepcopy(runtime)
    old["schema_version"] = 1
    old["kind"] = "n1_a3_runtime_asset_pins_v1"
    old.pop("private_opengl")
    old.pop("loader_library_path")

    with pytest.raises(L.LaunchRefused, match="malformed|schema-v2"):
        L._validate_runtime_asset_claim(old, checkout=exact_repo["repo"])

    missing_integrity = copy.deepcopy(runtime)
    missing_integrity.pop("integrity_model")
    with pytest.raises(L.LaunchRefused, match="malformed"):
        L._validate_runtime_asset_claim(
            missing_integrity, checkout=exact_repo["repo"]
        )


def test_runtime_asset_exec_environment_is_exactly_claim_owned(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    runtime = L.build_plan(path)["canonical_payload"]["runtime_assets"]

    assert L._runtime_asset_exec_environment(
        runtime, checkout=exact_repo["repo"]
    ) == {
        "HOPE_URDF_IMPORTER_NO_UI": "1",
        "HOPE_AGIBOT_A3_USD_PATH": runtime["a3_preconverted_usd"]["path"],
        "LD_LIBRARY_PATH": runtime["loader_library_path"],
    }


def test_old_spec_normalizes_diagnostic_update_profile_false(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    plan = L.build_plan(path)
    normalized = plan["canonical_payload"]["spec"]

    assert normalized["diagnostic_update_profile"] is False
    assert L._diagnostic_update_profile_environment(normalized) == {}
    assert (
        plan["launch_claim_sha256"]
        == L.canonical_sha256(plan["canonical_payload"])
    )


def test_new_spec_claim_pins_diagnostic_update_profile_true(exact_repo):
    spec, path = exact_repo["make_spec"]()
    spec["diagnostic_update_profile"] = True
    path.write_bytes(_canonical(spec))
    plan = L.build_plan(path)
    normalized = plan["canonical_payload"]["spec"]

    assert normalized["diagnostic_update_profile"] is True
    assert L._diagnostic_update_profile_environment(normalized) == {
        "HOPE_ACTION_BALL_UPDATE_PROFILE": "1"
    }
    assert (
        plan["launch_claim_sha256"]
        == L.canonical_sha256(plan["canonical_payload"])
    )


@pytest.mark.parametrize("value", (None, 0, 1, "1", [], {}))
def test_rejects_non_boolean_diagnostic_update_profile(exact_repo, value):
    spec, path = exact_repo["make_spec"]()
    spec["diagnostic_update_profile"] = value
    path.write_bytes(_canonical(spec))

    with pytest.raises(
        L.LaunchRefused,
        match="diagnostic_update_profile must be a boolean",
    ):
        L.build_plan(path)


def test_runtime_asset_environment_is_required_before_namespace_claim(
    exact_repo, monkeypatch
):
    _spec, path = exact_repo["make_spec"]()
    monkeypatch.delenv("HOPE_URDF_IMPORTER_NO_UI")
    with pytest.raises(
        L.LaunchRefused, match="HOPE_URDF_IMPORTER_NO_UI must equal 1"
    ):
        L.build_plan(path)


def test_full_scope_is_diagnostic_only_and_enables_full_body_mimic(exact_repo):
    spec, _path = exact_repo["make_spec"]()
    path = _convert_fixture_to_full(exact_repo, spec)
    payload = L.build_plan(path)["canonical_payload"]

    assert payload["spec"]["scope"] == "full"
    assert payload["bundle"]["scope"] == "full"
    assert payload["bundle"]["prototype"]["scope"] == "full"
    assert payload["diagnostic_unauthorized"] is True
    assert payload["formal_evidence_prohibited"] is True
    assert payload["bundle"]["contact_alignment"]["status"] == "PASS"
    assert payload["bundle"]["full_solver_admission_preflight"] == {
        "schema_version": 1,
        "kind": "full_fixed_action_exact_solver_admission_preflight_v1",
        "proposal_count": 512,
        "admitted_count": 511,
        "rejected_count": 1,
        "admit_rate": 511 / 512,
        "diagnostic_status": "PASS",
    }
    assert "task.rewards.full_body_mimic=true" in payload["training_argv"]
    assert "task.rewards.full_body_mimic=false" not in payload["training_argv"]


def test_legacy_v1_bundle_is_read_compatible_but_not_dynamic_launchable(
    exact_repo,
):
    spec, _path = exact_repo["make_spec"]()
    path = _convert_fixture_to_legacy_v1(exact_repo, spec)
    validated = L._validate_bundle(
        exact_repo["repo"],
        spec["source"]["commit_sha"],
        spec["bundle"],
        expected_action="bh_loop_c",
        expected_scope="upper",
    )
    assert "dynamic_ready" not in validated
    with pytest.raises(
        L.LaunchRefused, match="schema-v1 remains read-compatible only"
    ):
        L.build_plan(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda receipt: receipt.__setitem__("verdict", "FAIL"),
            "does not prove",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "terminal_reasons", ["joint_actual_forbidden"]
            ),
            "does not prove",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "plant_contract_match", False
            ),
            "does not prove",
        ),
    ),
)
def test_rejects_dynamic_ready_receipt_semantic_drift(
    exact_repo, mutation, match
):
    spec, path = exact_repo["make_spec"]()
    repo = exact_repo["repo"]
    receipt_path = exact_repo["nominal_hold_path"]
    receipt = json.loads(receipt_path.read_text())
    mutation(receipt)
    unsigned = dict(receipt)
    unsigned.pop("content_sha256")
    receipt["content_sha256"] = L.canonical_sha256(unsigned)
    receipt_path.write_bytes(_canonical(receipt))
    bundle_path = repo / spec["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text())
    bundle["dynamic_ready"]["nominal_hold_receipt"][
        "sha256"
    ] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    bundle_path.write_bytes(_canonical(bundle))
    spec["bundle"]["sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "dynamic-ready receipt tamper")
    spec["source"]["commit_sha"] = _git(repo, "rev-parse", "HEAD")
    path.write_bytes(_canonical(spec))
    with pytest.raises(L.LaunchRefused, match=match):
        L.build_plan(path)


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("bundle_scope", "bundle scope differs"),
        ("prototype_scope", "prototype must be schema 2"),
        ("manifest_scope", "manifest prototype scope differs"),
        ("contact_scope", "contact receipt scope differs"),
        ("diagnostic_only", "frozen-action contact alignment only"),
        ("training_authorized", "frozen-action contact alignment only"),
        ("preflight_missing", "missing exact solver admission provenance"),
        ("preflight_status", "diagnostic gate is not PASS"),
    ],
)
def test_full_scope_rejects_scope_or_diagnostic_claim_drift(
    exact_repo, tamper, match
):
    spec, _path = exact_repo["make_spec"]()
    path = _convert_fixture_to_full(exact_repo, spec)
    repo = exact_repo["repo"]
    bundle_path = repo / spec["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text())

    if tamper == "bundle_scope":
        bundle["scope"] = "upper"
    elif tamper == "prototype_scope":
        bundle["prototype"]["scope"] = "upper"
    elif tamper == "manifest_scope":
        artifact_path = repo / bundle["manifest"]["path"]
        artifact = json.loads(artifact_path.read_text())
        artifact["prototype"]["scope"] = "upper"
        artifact_path.write_bytes(_canonical(artifact))
        bundle["manifest"]["sha256"] = hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
    elif tamper == "contact_scope":
        artifact_path = repo / bundle["contact_alignment"]["path"]
        artifact = json.loads(artifact_path.read_text())
        artifact["scope"] = "upper"
        artifact_path.write_bytes(_canonical(artifact))
        bundle["contact_alignment"]["sha256"] = hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
    elif tamper in {"preflight_missing", "preflight_status"}:
        artifact_path = repo / bundle["prototype"]["path"]
        artifact = json.loads(artifact_path.read_text())
        if tamper == "preflight_missing":
            artifact.pop("provenance")
        else:
            artifact["provenance"]["full_solver_admission_preflight"][
                "diagnostic_gate"
            ]["status"] = "FAIL"
        artifact_path.write_bytes(_canonical(artifact))
        prototype_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        bundle["prototype"]["sha256"] = prototype_sha
        manifest_path = repo / bundle["manifest"]["path"]
        manifest = json.loads(manifest_path.read_text())
        manifest["prototype"]["sha256"] = prototype_sha
        manifest_path.write_bytes(_canonical(manifest))
        bundle["manifest"]["sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    else:
        bundle["claims"][tamper] = tamper != "diagnostic_only"

    bundle_path.write_bytes(_canonical(bundle))
    spec["bundle"]["sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", f"tamper {tamper}")
    spec["source"]["commit_sha"] = _git(repo, "rev-parse", "HEAD")
    path.write_bytes(_canonical(spec))

    with pytest.raises(L.LaunchRefused, match=match):
        L.build_plan(path)


@pytest.mark.parametrize("profile", ("user_override", "task_mid_x2"))
def test_rejects_unknown_or_retired_profile_and_canary_budget(
    exact_repo, profile
):
    spec, path = exact_repo["make_spec"]()
    spec["reward_profile"] = profile
    path.write_bytes(_canonical(spec))
    with pytest.raises(L.LaunchRefused, match="reward_profile"):
        L.build_plan(path)

    spec, path = exact_repo["make_spec"](stage="canary")
    spec["max_iterations"] = 2001
    path.write_bytes(_canonical(spec))
    with pytest.raises(L.LaunchRefused, match="\\[10,2000\\]"):
        L.build_plan(path)



def test_accepts_only_the_exact_long_budget(exact_repo):
    spec, path = exact_repo["make_spec"](stage="long")
    payload = L.build_plan(path)["canonical_payload"]
    assert payload["long_stage_prohibited"] is False
    assert payload["formal_evidence_prohibited"] is True
    assert payload["curriculum_promotion_prohibited"] is True
    assert payload["diagnostic_unauthorized"] is True
    assert payload["spec"]["num_envs"] == 4096
    assert payload["spec"]["max_iterations"] == 20_001
    assert payload["spec"]["save_interval"] == 100


def test_accepts_only_the_exact_probe_budget(exact_repo):
    spec, path = exact_repo["make_spec"](stage="probe")
    payload = L.build_plan(path)["canonical_payload"]
    assert payload["long_stage_prohibited"] is True
    assert payload["formal_evidence_prohibited"] is True
    assert payload["curriculum_promotion_prohibited"] is True
    assert payload["diagnostic_unauthorized"] is True
    assert payload["spec"]["num_envs"] == 4096
    assert payload["spec"]["max_iterations"] == 5
    assert payload["spec"]["save_interval"] == 1
    assert "num_envs=4096" in payload["training_argv"]
    assert "max_iterations=5" in payload["training_argv"]
    assert "algo.runner.save_interval=1" in payload["training_argv"]


def test_accepts_only_the_exact_milestone1000_budget(exact_repo):
    spec, path = exact_repo["make_spec"](stage="milestone1000")
    payload = L.build_plan(path)["canonical_payload"]
    assert payload["long_stage_prohibited"] is True
    assert payload["formal_evidence_prohibited"] is True
    assert payload["curriculum_promotion_prohibited"] is True
    assert payload["diagnostic_unauthorized"] is True
    assert payload["spec"]["num_envs"] == 4096
    assert payload["spec"]["max_iterations"] == 1001
    assert payload["spec"]["save_interval"] == 100
    assert "num_envs=4096" in payload["training_argv"]
    assert "max_iterations=1001" in payload["training_argv"]
    assert "algo.runner.save_interval=100" in payload["training_argv"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", 4095),
        ("num_envs", 4097),
        ("max_iterations", 1000),
        ("max_iterations", 1002),
        ("save_interval", 99),
        ("save_interval", 101),
    ],
)
def test_rejects_any_milestone1000_budget_drift(
    exact_repo, field, value
):
    spec, path = exact_repo["make_spec"](stage="milestone1000")
    spec[field] = value
    path.write_bytes(_canonical(spec))
    with pytest.raises(L.LaunchRefused, match="milestone1000 is exactly"):
        L.build_plan(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", 1024),
        ("num_envs", 4095),
        ("max_iterations", 4),
        ("max_iterations", 6),
        ("save_interval", 2),
    ],
)
def test_rejects_any_probe_budget_drift(exact_repo, field, value):
    spec, path = exact_repo["make_spec"](stage="probe")
    spec[field] = value
    path.write_bytes(_canonical(spec))
    with pytest.raises(L.LaunchRefused, match="probe is exactly"):
        L.build_plan(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", 4095),
        ("num_envs", 4097),
        ("max_iterations", 20_000),
        ("max_iterations", 20_002),
        ("save_interval", 99),
        ("save_interval", 101),
    ],
)
def test_rejects_any_long_budget_drift(exact_repo, field, value):
    spec, path = exact_repo["make_spec"](stage="long")
    spec[field] = value
    path.write_bytes(_canonical(spec))
    with pytest.raises(L.LaunchRefused, match="long is exactly"):
        L.build_plan(path)


def test_rejects_dirty_checkout_before_namespace_claim(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    (exact_repo["repo"] / "untracked.txt").write_text("dirty\n")
    with pytest.raises(L.LaunchRefused, match="source checkout is dirty"):
        L.build_plan(path)


def test_rejects_edited_contact_receipt_before_semantic_use(exact_repo):
    _spec, path = exact_repo["make_spec"]()
    contact_path = exact_repo["contact_path"]
    contact = json.loads(contact_path.read_text())
    contact["claims"]["landing_claim"] = True
    contact_path.write_bytes(_canonical(contact))
    # This fails at clean-source binding even before semantic validation: a
    # contact claim cannot be edited under a reviewed commit and bundle pin.
    with pytest.raises(L.LaunchRefused, match="source checkout is dirty"):
        L.build_plan(path)


def test_semantic_gate_rejects_landing_claim(exact_repo):
    contact = copy.deepcopy(exact_repo["contact_document"])
    contact["claims"]["landing_claim"] = True
    with pytest.raises(L.LaunchRefused, match="contact alignment only"):
        L._validate_contact_receipt(
            contact,
            bundle=exact_repo["bundle_document"],
            manifest=exact_repo["manifest_document"],
        )


def test_semantic_gate_accepts_stable_upper_retargeted_contact(exact_repo):
    contact = copy.deepcopy(exact_repo["contact_document"])
    alignment = contact["alignment"]
    alignment.pop("legacy_absolute_contact_z_w_m")
    alignment.pop("corrected_contact_offset_z_b_yaw_m")
    alignment.update(
        {
            "contact_center_authority": (
                "a3_stable_upper_selected_rubber_face_center_at_pinned_"
                "strike_frame"
            ),
            "retargeted_contact_center_z_w_m": (
                alignment["ready_root_z_w_m"]
                + alignment["task_contact_offset_center_b_yaw_m"][2]
            ),
            "upper_contact_center_preserved": False,
        }
    )
    result = L._validate_contact_receipt(
        contact,
        bundle=exact_repo["bundle_document"],
        manifest=exact_repo["manifest_document"],
    )
    assert result["status"] == "PASS"

    alignment["contact_center_authority"] = (
        "full_motion_selected_rubber_face_center_at_explicit_strike_frame"
    )
    with pytest.raises(L.LaunchRefused, match="authority"):
        L._validate_contact_receipt(
            contact,
            bundle=exact_repo["bundle_document"],
            manifest=exact_repo["manifest_document"],
        )


def test_rejects_absolute_world_contact_disguised_as_relative(exact_repo):
    spec, _path = exact_repo["make_spec"]()
    # Rebuild the exact artifact chain and commit the malicious alternative so
    # this reaches the semantic gate instead of stopping at the byte pin.
    repo = exact_repo["repo"]
    contact_path = exact_repo["contact_path"]
    contact = json.loads(contact_path.read_text())
    contact["alignment"]["corrected_contact_offset_z_b_yaw_m"] = 1.30
    contact["alignment"]["task_contact_offset_center_b_yaw_m"][2] = 1.30
    contact_path.write_bytes(_canonical(contact))
    contact_sha = hashlib.sha256(contact_path.read_bytes()).hexdigest()
    bundle_path = repo / spec["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text())
    bundle["contact_alignment"]["sha256"] = contact_sha
    bundle_path.write_bytes(_canonical(bundle))
    spec["bundle"]["sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "malicious absolute contact fixture")
    spec["source"]["commit_sha"] = _git(repo, "rev-parse", "HEAD")
    spec_path = bundle_path.parents[1].parent / "malicious-spec.json"
    # Put the spec outside the checkout so source remains clean.
    spec_path = repo.parent / "malicious-spec.json"
    spec_path.write_bytes(_canonical(spec))
    with pytest.raises(L.LaunchRefused, match="corrected z is not"):
        L.build_plan(spec_path)


def test_source_contains_lifetime_lock_double_gpu_check_and_no_shell():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pass_fds=(lock_fd,)" in source
    assert source.count("_verify_gpu_empty(") >= 3
    assert "os.execve(argv[0], argv, environment)" in source
    assert "**_runtime_asset_exec_environment(runtime_assets)" in source
    assert (
        '"HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256": expected_sha'
        in source
    )
    assert source.count(
        "**_diagnostic_update_profile_environment(spec)"
    ) == 2
    assert '"KIT_BOOT_TIMEOUT_S": "2700"' in source
    assert '"KIT_BOOT_STALE_TIMEOUT_S": "1800"' in source
    assert "shell=True" not in source
    assert "subprocess.Popen" not in source
    assert "long_stage_prohibited" in source
    assert L.ALLOWED_STAGES == frozenset(
        ("smoke", "probe", "canary", "milestone1000", "long")
    )
