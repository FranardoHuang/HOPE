from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


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
    kit_path = repo / L.KIT_LAUNCHER_SOURCE
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    train_path.write_text("# train\n", encoding="utf-8")
    task_path.write_text("name: test\n", encoding="utf-8")
    kit_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher_path.chmod(0o755)
    kit_path.chmod(0o755)

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
    solver_payload = {
        "solver": "fixture",
        "implementation_source_sha256": source_hashes,
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
        "schema_version": 1,
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
        "geometry": geometry,
        "claims": claims,
    }
    bundle = add_json(
        "configs/bh_loop_c.bundle.v1.json", bundle_document
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
        elif stage == "long":
            num_envs, iterations, save = 4096, 300_000_000_000, 100
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
        "make_spec": make_spec,
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
    assert "+task.racket.reference_guard_mode=metrics_only" in argv
    assert "task.actor_obs_contract=action_ball_n1" in argv
    assert "algo.policy.init_noise_std=0.02" in argv
    assert "action_ball_shared_ready_bootstrap=true" in argv
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
    assert payload["spec"]["max_iterations"] == 300_000_000_000
    assert payload["spec"]["save_interval"] == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", 4095),
        ("num_envs", 4097),
        ("max_iterations", 299_999_999_999),
        ("max_iterations", 300_000_000_001),
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
    assert "shell=True" not in source
    assert "subprocess.Popen" not in source
    assert "long_stage_prohibited" in source
    assert L.ALLOWED_STAGES == frozenset(("smoke", "canary", "long"))
