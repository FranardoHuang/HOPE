"""Host-only fail-closed tests for the formal Isaac table-smoke producer.

These tests deliberately do not import Isaac Lab or launch Kit.  They cover the
strict input/receipt closure and prove that a host process with no live runtime
origin cannot mint ``PASS``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_table_obstacle_scene.py"
)
SPEC = importlib.util.spec_from_file_location(
    "check_table_obstacle_scene_producer_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
P = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P
SPEC.loader.exec_module(P)

ADMISSION_SCRIPT = SCRIPT.with_name("canonical_motion_admission.py")
ADMISSION_SPEC = importlib.util.spec_from_file_location(
    "canonical_motion_admission_isaac_producer_roundtrip_test",
    ADMISSION_SCRIPT,
)
assert ADMISSION_SPEC is not None and ADMISSION_SPEC.loader is not None
ADMISSION = importlib.util.module_from_spec(ADMISSION_SPEC)
sys.modules[ADMISSION_SPEC.name] = ADMISSION
ADMISSION_SPEC.loader.exec_module(ADMISSION)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _nominal_hold_fixture(tmp_path: Path):
    joint_names = [f"a3_joint_{index:02d}" for index in range(31)]
    source_rows = {}
    for key, payload in (
        ("stable_motion", b"stable-motion-npz-fixture"),
        ("stable_receipt", b'{"stable":true}'),
        ("mujoco_model", b"<mujoco model='a3'/>"),
    ):
        path = tmp_path / f"{key}.bin"
        path.write_bytes(payload)
        source_rows[key] = {
            "path": str(path.resolve()),
            "sha256": _sha(payload),
        }
    runtime_contract = {
        "target_mode": "action_ball",
        "joint_names": joint_names,
        "joint_stiffness": [100.0] * 31,
        "joint_damping": [4.0] * 31,
        "joint_effort_limits": [40.0] * 31,
        "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in range(31)],
        "default_joint_pos": [0.0] * 31,
        "action_scale": [0.25] * 31,
        "joint_armature": [0.01] * 31,
        "joint_friction_coefficients": [0.02] * 31,
        "finite_projection_soft_envelope_inset_fraction": 0.05,
        "physics_step_dt_s": 0.005,
        "policy_step_dt_s": 0.02,
        "control_decimation": 4,
    }
    runtime_payload = P._canonical_json_bytes(runtime_contract)
    runtime_path = tmp_path / "runtime_training_contract.json"
    runtime_path.write_bytes(runtime_payload)
    source_rows["runtime_training_contract"] = {
        "path": str(runtime_path.resolve()),
        "sha256": _sha(runtime_payload),
    }
    document = {
        "schema_version": 1,
        "kind": P.NOMINAL_HOLD_ARTIFACT_KIND,
        "action_id": "bh_block",
        "robot": {
            "family": "AgiBot A3",
            "joint_names": joint_names,
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "sources": source_rows,
        "physical_ready": {
            "root_pos_w_m": [0.0, 0.0, 1.0],
            "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_pos_rad": [0.0] * 31,
            "joint_vel_radps": [0.0] * 31,
        },
        "runtime_plant": {
            "joint_stiffness": [100.0] * 31,
            "joint_damping": [4.0] * 31,
            "joint_effort_limits": [40.0] * 31,
            "joint_armature": [0.01] * 31,
            "joint_friction_coefficients": [0.02] * 31,
            "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in range(31)],
            "finite_projection_soft_envelope_inset_fraction": 0.05,
            "executed_qdes_lower_rad": [-0.9] * 31,
            "executed_qdes_upper_rad": [0.9] * 31,
            "default_joint_pos_rad": [0.0] * 31,
            "action_scale_rad": [0.25] * 31,
            "physics_step_dt_s": 0.005,
            "policy_step_dt_s": 0.02,
            "control_decimation": 4,
        },
        "hold_candidate": {
            "hold_qdes_joint_pos_rad": [0.0] * 31,
            "normalized_actor_action": [0.0] * 31,
        },
        "required_next_gate": {
            "kind": P.NOMINAL_HOLD_RECEIPT_KIND,
        },
    }
    document["content_sha256"] = _sha(P._canonical_json_bytes(document))
    artifact_path = tmp_path / "dynamic_ready.json"
    artifact_path.write_bytes(P._canonical_json_bytes(document))
    return artifact_path, document, runtime_contract


def _fixture_tree(tmp_path: Path, action_count: int = 5):
    source = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "check_table_obstacle_scene.py"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# committed producer fixture\n")
    solver_source_dir = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
    )
    solver_source_dir.mkdir(parents=True)
    solver_source_sha = {}
    for name in P._ACTION_BALL_SOLVER_SOURCE_NAMES:
        path = solver_source_dir / name
        path.write_bytes(f"# exact solver fixture: {name}\n".encode("ascii"))
        solver_source_sha[name] = _sha(path.read_bytes())
    contact_geometry_payload = {
        "schema_version": 2,
        "semantics": "fixture canonical exact-face geometry",
    }
    contact_geometry = {
        "payload": contact_geometry_payload,
        "sha256": _sha(P._canonical_json_bytes(contact_geometry_payload)),
    }
    solver_payload = {
        "kind": "fixture.frozen_ball_to_task_solver",
        "implementation_source_sha256": solver_source_sha,
        "contact_geometry": contact_geometry,
    }
    physics_payload = {
        "kind": "fixture.action_ball_physics",
        "geometry_and_grading": {
            "table_surface_z_m": 0.76,
            "ball_center_net_top_z_m": 0.9325,
            "net_x_m": 1.87,
            "opponent_near_x_m": 0.5,
            "opponent_far_x_m": 3.24,
            "minimum_landing_depth_m": 0.3,
            "table_half_width_m": 0.7625,
        },
    }
    solver_profile_sha = _sha(P._canonical_json_bytes(solver_payload))
    physics_profile_sha = _sha(P._canonical_json_bytes(physics_payload))
    profile_pins = {
        "solver_payload": solver_payload,
        "physics_payload": physics_payload,
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
        "solver_implementation_source_sha256": solver_source_sha,
        "contact_geometry": contact_geometry,
    }
    profile_path = tmp_path / "profile_pins.json"
    profile_path.write_text(
        json.dumps(
            profile_pins,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    geometry_source = solver_source_dir / "racket_contact_geometry.py"
    geometry_contract = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": geometry_source.relative_to(tmp_path).as_posix(),
        "source_sha256": _sha(geometry_source.read_bytes()),
        "geometry_source_sha256": contact_geometry["sha256"],
    }
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    action_ids = (
        P.FRESH_N5_ACTION_IDS
        if action_count == 5
        else tuple(f"fixture_action_{index:03d}" for index in range(action_count))
    )
    families = (
        (
            "backhand",
            "forehand",
            "backhand",
            "backhand",
            "forehand",
        )
        if action_count == 5
        else tuple(
            "backhand" if index % 2 == 0 else "forehand"
            for index in range(action_count)
        )
    )
    signs = tuple(
        -1 if family == "backhand" else 1 for family in families
    )
    actions = []
    for index, (motion_id, family, sign) in enumerate(
        zip(action_ids, families, signs)
    ):
        path = motion_dir / f"{motion_id}.npz"
        payload = f"exact-motion-{index}-{motion_id}".encode("ascii")
        path.write_bytes(payload)
        actions.append(
            {
                "action_id": motion_id,
                "action_uid": P._derive_action_uid(
                    motion_id, family, _sha(payload)
                ),
                "family": family,
                "motion_path": path.relative_to(tmp_path).as_posix(),
                "motion_sha256": _sha(payload),
                "strike_phase": 0.5,
                "reference_t_hit_s": 0.5,
                "reference_t_cycle_s": 1.0,
                "reference_racket_site_speed_mps": 2.0,
                "reaction_margin_s": 0.1,
                "teacher_rate_min": 0.5,
                "teacher_rate_max": 1.0,
                "mount_normal_sign": sign,
                "ball_profile": {
                    "time_to_contact_center_s": 1.2,
                },
            }
        )
    manifest = {
        "schema_version": 3,
        "manifest_id": f"producer-fixture-n{action_count}",
        "mobility_mode": "no_move",
        "action_order": list(action_ids),
        "prototype": {
            "scope": "full" if action_count == 73 else "upper"
        },
        "solver_profile_sha256": solver_profile_sha,
        "physics_profile_sha256": physics_profile_sha,
        "actions": actions,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest_path, manifest


def _load_fixture_inputs(
    source: Path,
    manifest_path: Path,
    *,
    repo_root: Path,
):
    profile_path = manifest_path.with_name("profile_pins.json")
    return _load_formal_fixture(
        manifest_path.relative_to(repo_root).as_posix(),
        profile_pins_value=profile_path.relative_to(repo_root).as_posix(),
        expected_profile_pins_sha256=_sha(profile_path.read_bytes()),
        repo_root=repo_root,
        source_path=source,
    )


def _fixture_action_set(repo_root: Path, manifest_path: Path) -> dict:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {
        row["action_id"]: row
        for row in document["actions"]
    }
    if len(document["actions"]) == 5:
        ids = list(P.FRESH_N5_ACTION_IDS)
    else:
        ids = [row["action_id"] for row in document["actions"]]
    uids = [by_id[action_id]["action_uid"] for action_id in ids]
    action_count = len(ids)
    profile_id = (
        "fresh_upper_nomove_n5_v3"
        if action_count == 5
        else f"fixture_n{action_count}"
    )
    row = {
        "profile_id": profile_id,
        "expected_n": action_count,
        "scope": "full" if action_count == 73 else "upper",
        "mobility_mode": "no_move",
        "ordered_action_ids": ids,
        "ordered_action_uids": uids,
        "order_uid_digest_sha256": P.action_set_contract.order_uid_digest(
            ids, uids
        ),
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "manifest_sha256": _sha(manifest_path.read_bytes()),
        "experiment_name": f"fixture_n{action_count}",
    }
    return P.action_set_contract.validate_contract(
        row, profile_id=profile_id, profile_policies={}
    )


def _load_formal_fixture(
    manifest_value,
    *,
    profile_pins_value,
    expected_profile_pins_sha256,
    repo_root,
    source_path,
):
    manifest_path = repo_root / manifest_value
    trusted = _fixture_action_set(repo_root, manifest_path)
    original = P._load_trusted_action_set
    P._load_trusted_action_set = lambda profile_id: trusted
    try:
        return P._load_formal_inputs(
            manifest_value,
            action_set_profile=trusted["profile_id"],
            profile_pins_value=profile_pins_value,
            expected_profile_pins_sha256=expected_profile_pins_sha256,
            repo_root=repo_root,
            source_path=source_path,
        )
    finally:
        P._load_trusted_action_set = original


def _commit_fixture(repo_root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "producer-test@example.invalid"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Producer Test"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"], cwd=repo_root, check=True
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _valid_evidence(inputs):
    token = object()
    P._ISAAC_RUNTIME_ORIGIN = token
    action_rows = tuple(
        P._RuntimeActionEvidence(
            motion_id=row.motion_id,
            action_uid=row.action_uid,
            motion_sha256=row.file.sha256,
            frame_count=3,
            physics_steps=12,
            complete_cycle=True,
            table_contact_count=0,
            fall_count=0,
            hard_limit_count=0,
            unsafe_count=0,
        )
        for row in inputs.motions
    )
    return P._RuntimeEvidence(
        origin=token,
        source_commit_sha="1" * 40,
        isaac_version="isaaclab=fixture",
        python_executable=sys.executable,
        gpu_identity={
            "physical_index": 2,
            "logical_index": 0,
            "cuda_visible_devices": "2",
            "gpu_uuid": "GPU-fixture",
            "gpu_name": "Fixture GPU",
            "driver_version": "fixture-driver",
            "nvml_verified": True,
        },
        physics_steps=sum(row.physics_steps for row in action_rows),
        actions=action_rows,
        real_physx_contacts=True,
        full_action_ball_assembly=True,
        all_32_body_pair_filters=True,
        all_five_obstacles=True,
        all_four_substeps=True,
        positive_control_pass=True,
        negative_control_pass=True,
        zero_reset_leakage=True,
    )


def test_host_import_does_not_launch_kit_or_create_runtime_origin():
    assert P._app is None
    assert P._ISAAC_RUNTIME_ORIGIN is None
    assert P.gym is None
    assert P.torch is None


def test_real_task_id_is_default_and_retired_fake_id_is_rejected():
    args = P._parse([])
    assert args.task == "HOPE-PingPong-ActionBall-AgibotA3-v0"
    P._validate_cli_mode(args)

    fake = P._parse(
        ["--task", "Tracking-Flat-AgibotA3-Hope-ActionBall-v0"]
    )
    with pytest.raises(P.TableSmokeReceiptError, match="retired fake task id"):
        P._validate_cli_mode(fake)


def test_nominal_hold_cli_is_opt_in_one_env_and_pinned():
    args = P._parse(
        [
            "--num-envs",
            "1",
            "--device",
            "cuda:1",
            "--nominal-hold",
            "/tmp/ready.json",
            "--nominal-hold-sha256",
            "1" * 64,
            "--nominal-hold-receipt-out",
            "/tmp/hold.json",
        ]
    )
    P._validate_cli_mode(args)
    with pytest.raises(P.TableSmokeReceiptError, match="requires one explicit cuda:N"):
        P._validate_cli_mode(
            P._parse(
                [
                    "--num-envs",
                    "2",
                    "--nominal-hold",
                    "/tmp/ready.json",
                    "--nominal-hold-sha256",
                    "1" * 64,
                    "--nominal-hold-receipt-out",
                    "/tmp/hold.json",
                ]
            )
        )


def test_nominal_hold_artifact_pins_a3_motion_and_core_plant(tmp_path):
    path, document, _contract = _nominal_hold_fixture(tmp_path)
    loaded = P._load_nominal_hold_input(
        path, expected_sha256=_sha(path.read_bytes())
    )
    assert loaded.action_id == "bh_block"
    assert loaded.joint_names == tuple(document["robot"]["joint_names"])
    assert loaded.motion_sha256 == document["sources"]["stable_motion"]["sha256"]
    assert loaded.expected_plant["control_decimation"] == 4

    document["hold_candidate"]["hold_qdes_joint_pos_rad"][0] = "nan"
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    document["content_sha256"] = _sha(P._canonical_json_bytes(unsigned))
    path.write_bytes(P._canonical_json_bytes(document))
    with pytest.raises(P.TableSmokeReceiptError, match="hold q_des"):
        P._load_nominal_hold_input(
            path, expected_sha256=_sha(path.read_bytes())
        )


def test_nominal_hold_outputs_are_no_clobber(tmp_path):
    output = P._fresh_nominal_path(tmp_path / "hold.json", "receipt")
    P._exclusive_publish_nominal_hold_receipt(
        output,
        {"kind": P.NOMINAL_HOLD_RECEIPT_KIND, "verdict": "FAIL"},
    )
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        P._fresh_nominal_path(output, "receipt")


def test_nominal_hold_captures_raw_reset_before_dynamic_ready_write():
    events = []
    unwrapped = SimpleNamespace(
        sim=SimpleNamespace(forward=lambda: events.append("forward")),
        scene=SimpleNamespace(
            update=lambda dt: events.append(("scene.update", dt))
        ),
    )
    P._refresh_nominal_hold_derived_state(unwrapped)
    assert events == ["forward", ("scene.update", 0.0)]

    source = inspect.getsource(P.nominal_hold_probe)
    reset = source.index("env.reset()")
    reset_refresh = source.index(
        "_refresh_nominal_hold_derived_state(unwrapped)", reset
    )
    raw_frame = source.index('save_frame("raw_env_reset", 0, last_png)')
    artifact_write = source.index("motion_command.clip_id[env_ids] = 0")
    simulator_write = source.index("robot.write_root_state_to_sim(")
    ready_refresh = source.index(
        "_refresh_nominal_hold_derived_state(unwrapped)", simulator_write
    )
    ready_frame = source.index(
        'save_frame("physical_ready_after_reset_write", 0, last_png)'
    )
    assert (
        reset
        < reset_refresh
        < raw_frame
        < artifact_write
        < simulator_write
        < ready_refresh
        < ready_frame
    )


def test_formal_cli_has_no_boolean_pass_claims_and_requires_pod_shape():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "--real-physx-contacts",
        "--all-32-body-pair-filters",
        "--positive-control-pass",
        "--negative-control-pass",
        "--zero-reset-leakage",
    ):
        assert forbidden not in source

    args = P._parse(
        [
            "--receipt-out",
            "receipt.json",
            "--action-set-profile",
            "fixture_n5",
            "--manifest",
            "manifest.json",
        ]
    )
    with pytest.raises(P.TableSmokeReceiptError, match="num-envs"):
        P._validate_cli_mode(args)


def test_manifest_snapshots_exact_fresh_n5_order_and_motion_bytes(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    assert tuple(row.motion_id for row in inputs.motions) == P.FRESH_N5_ACTION_IDS
    assert len({row.action_uid for row in inputs.motions}) == 5
    assert inputs.manifest.sha256 == _sha(manifest_path.read_bytes())
    assert len({row.file.sha256 for row in inputs.motions}) == 5
    P._assert_formal_inputs_unchanged(inputs)

    inputs.motions[0].file.path.write_bytes(b"changed-after-snapshot")
    with pytest.raises(P.TableSmokeReceiptError, match="inode or bytes changed"):
        P._assert_formal_inputs_unchanged(inputs)


@pytest.mark.parametrize("action_count", (1, 5, 73))
def test_formal_table_inputs_and_receipt_keep_every_action_and_32xn_rows(
    tmp_path, action_count: int
):
    root = tmp_path / f"n{action_count}"
    source, manifest_path, _ = _fixture_tree(
        root, action_count=action_count
    )
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=root
    )
    assert len(inputs.motions) == action_count
    assert inputs.action_set_contract["expected_n"] == action_count
    receipt = P._build_formal_receipt(
        inputs, _valid_evidence(inputs)
    )
    assert len(receipt["actions"]) == action_count
    assert all(
        row["body_pair_filter_count"] == 32
        for row in receipt["actions"]
    )
    assert (
        receipt["runtime_contract"]["action_body_pair_filter_rows"]
        == 32 * action_count
    )
    assert receipt["schema_version"] == 2


@pytest.mark.parametrize("action_count", (1, 5, 73))
@pytest.mark.parametrize(
    "tamper",
    ("count", "order", "uid", "scope", "mobility", "manifest_sha"),
)
def test_formal_table_rejects_action_set_or_manifest_tamper(
    tmp_path, action_count: int, tamper: str
):
    root = tmp_path / f"n{action_count}_{tamper}"
    source, manifest_path, manifest = _fixture_tree(
        root, action_count=action_count
    )
    trusted = _fixture_action_set(root, manifest_path)
    if tamper == "count":
        trusted = dict(trusted)
        trusted["expected_n"] += 1
    elif tamper == "order":
        manifest["action_order"] = (
            list(reversed(manifest["action_order"]))
            if action_count > 1
            else ["wrong_action"]
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "uid":
        manifest["actions"][0]["action_uid"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "scope":
        manifest["prototype"]["scope"] = (
            "upper" if trusted["scope"] == "full" else "full"
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "mobility":
        manifest["mobility_mode"] = "move"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        trusted = dict(trusted)
        trusted["manifest_sha256"] = "0" * 64
    original = P._load_trusted_action_set
    P._load_trusted_action_set = lambda profile_id: trusted
    try:
        with pytest.raises(P.TableSmokeReceiptError):
            P._load_formal_inputs(
                manifest_path.relative_to(root).as_posix(),
                action_set_profile=trusted["profile_id"],
                profile_pins_value="profile_pins.json",
                expected_profile_pins_sha256=_sha(
                    (root / "profile_pins.json").read_bytes()
                ),
                repo_root=root,
                source_path=source,
            )
    finally:
        P._load_trusted_action_set = original


def test_profile_pins_and_solver_geometry_bytes_are_fail_closed(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    profile_path = tmp_path / "profile_pins.json"
    expected_profile_sha = _sha(profile_path.read_bytes())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["physics_payload"]["kind"] = "tampered"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(
        P.TableSmokeReceiptError, match="preregistered SHA-256"
    ):
        _load_formal_fixture(
            manifest_path.name,
            profile_pins_value=profile_path.name,
            expected_profile_pins_sha256=expected_profile_sha,
            repo_root=tmp_path,
            source_path=source,
        )

    source, manifest_path, _ = _fixture_tree(tmp_path / "source")
    source_root = tmp_path / "source"
    solver_path = (
        source_root
        / "hope_training/whole_body_tracking/source/"
        "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
        "racket_contact_geometry.py"
    )
    solver_path.write_text("# drifted geometry source\n", encoding="utf-8")
    profile_path = source_root / "profile_pins.json"
    with pytest.raises(
        P.TableSmokeReceiptError, match="bytes differ from profile pins"
    ):
        _load_formal_fixture(
            manifest_path.name,
            profile_pins_value=profile_path.name,
            expected_profile_pins_sha256=_sha(
                profile_path.read_bytes()
            ),
            repo_root=source_root,
            source_path=source,
        )

    semantic_root = tmp_path / "semantic"
    source, manifest_path, manifest = _fixture_tree(semantic_root)
    manifest["racket_geometry_contract"] = geometry_contract = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": (
            "hope_training/whole_body_tracking/source/"
            "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
            "racket_contact_geometry.py"
        ),
        "source_sha256": "1" * 64,
        "geometry_source_sha256": "2" * 64,
    }
    assert geometry_contract["schema_version"] == 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    profile_path = semantic_root / "profile_pins.json"
    with pytest.raises(
        P.TableSmokeReceiptError,
        match="strict training manifest contains gate-only",
    ):
        _load_formal_fixture(
            manifest_path.name,
            profile_pins_value=profile_path.name,
            expected_profile_pins_sha256=_sha(profile_path.read_bytes()),
            repo_root=semantic_root,
            source_path=source,
        )


def test_source_identity_rejects_nonignored_untracked_checkout_files(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    expected_commit = _commit_fixture(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    assert P._committed_source_identity(inputs) == expected_commit

    (tmp_path / "untracked_override.py").write_text(
        "raise RuntimeError('must not be ignored')\n", encoding="utf-8"
    )
    with pytest.raises(P.TableSmokeReceiptError, match="exact clean checkout"):
        P._committed_source_identity(inputs)


def test_runtime_module_closure_rejects_other_checkout_and_byte_drift(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    module_path = tmp_path / "package" / "runtime_fixture.py"
    module_path.parent.mkdir()
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    expected_commit = _commit_fixture(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    module_name = "whole_body_tracking.runtime_fixture"
    module = ModuleType(module_name)
    module.__file__ = str(module_path)
    # Other host-only tests install synthetic ``whole_body_tracking`` namespace
    # parents during collection.  They intentionally have no source file and
    # are not part of this fixture checkout.  Isolate the closure fixture
    # instead of weakening the formal producer to accept source-less runtime
    # modules.
    displaced = {
        name: loaded
        for name, loaded in tuple(sys.modules.items())
        if name == "whole_body_tracking"
        or name.startswith("whole_body_tracking.")
    }
    for name in displaced:
        sys.modules.pop(name, None)
    sys.modules[module_name] = module
    try:
        baseline = P._assert_runtime_source_closure(
            inputs,
            expected_commit,
            required_modules=(module_name,),
        )
        assert baseline[module_name].sha256 == _sha(b"VALUE = 1\n")

        module_path.write_text("VALUE = 2\n", encoding="utf-8")
        with pytest.raises(
            P.TableSmokeReceiptError, match="bytes differ from source commit"
        ):
            P._assert_runtime_source_closure(
                inputs,
                expected_commit,
                baseline=baseline,
                required_modules=(module_name,),
            )

        outside = tmp_path.parent / f"{tmp_path.name}-other-checkout.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")
        module.__file__ = str(outside)
        with pytest.raises(
            P.TableSmokeReceiptError, match="must resolve inside repository root"
        ):
            P._assert_runtime_source_closure(
                inputs,
                expected_commit,
                required_modules=(module_name,),
            )
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.update(displaced)


def test_formal_repo_root_is_derived_from_exact_tracked_producer_path(
    tmp_path,
):
    source = (
        tmp_path
        / "hope_training"
        / "whole_body_tracking"
        / "scripts"
        / "check_table_obstacle_scene.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("# producer\n", encoding="utf-8")
    assert P._repository_root_from_producer(source) == tmp_path

    relocated = tmp_path / "scripts" / "check_table_obstacle_scene.py"
    relocated.parent.mkdir()
    relocated.write_text("# producer\n", encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="exact tracked"):
        P._repository_root_from_producer(relocated)


def test_manifest_rejects_reorder_retired_id_hash_drift_and_traversal(tmp_path):
    source, manifest_path, manifest = _fixture_tree(tmp_path)

    manifest["action_order"] = list(reversed(P.FRESH_N5_ACTION_IDS))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="action order"):
        _load_fixture_inputs(
            source, manifest_path, repo_root=tmp_path
        )

    hash_root = tmp_path / "hash"
    hash_source, manifest_path, manifest = _fixture_tree(hash_root)
    manifest["actions"][0]["motion_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="differ from manifest"):
        _load_fixture_inputs(
            hash_source, manifest_path, repo_root=hash_root
        )

    traversal_root = tmp_path / "traversal"
    traversal_source, manifest_path, manifest = _fixture_tree(
        traversal_root
    )
    manifest["actions"][0]["motion_path"] = "../escape.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="path traversal"):
        _load_fixture_inputs(
            traversal_source,
            manifest_path,
            repo_root=traversal_root,
        )

    uid_root = tmp_path / "uid"
    uid_source, manifest_path, manifest = _fixture_tree(uid_root)
    manifest["actions"][0]["action_uid"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P.TableSmokeReceiptError, match="canonical action"):
        _load_fixture_inputs(
            uid_source, manifest_path, repo_root=uid_root
        )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers():
    with pytest.raises(P.TableSmokeReceiptError, match="duplicate JSON key"):
        P._strict_json_object(b'{"schema_version":3,"schema_version":3}', "x")
    with pytest.raises(P.TableSmokeReceiptError, match="forbidden JSON constant"):
        P._strict_json_object(b'{"value":NaN}', "x")


def test_symlinked_motion_and_symlinked_output_parent_are_rejected(tmp_path):
    source, manifest_path, manifest = _fixture_tree(tmp_path)
    real = tmp_path / "motions" / "bh_loop_c.npz"
    target = tmp_path / "real-motion.npz"
    target.write_bytes(real.read_bytes())
    real.unlink()
    real.symlink_to(target)
    with pytest.raises(P.TableSmokeReceiptError, match="symlink component"):
        _load_fixture_inputs(
            source, manifest_path, repo_root=tmp_path
        )

    output_real = tmp_path / "output-real"
    output_real.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output_real, target_is_directory=True)
    with pytest.raises(P.TableSmokeReceiptError, match="symlink component"):
        P._prepare_output_path(
            "output-link/receipt.json", repo_root=tmp_path
        )


def test_without_live_isaac_origin_no_pass_receipt_can_be_built(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    evidence = _valid_evidence(inputs)
    P._ISAAC_RUNTIME_ORIGIN = None
    with pytest.raises(P.TableSmokeReceiptError, match="live Isaac runtime"):
        P._build_formal_receipt(inputs, evidence)


def test_exact_receipt_schema_seal_and_exclusive_readback(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    receipt = P._build_formal_receipt(inputs, _valid_evidence(inputs))
    assert receipt["task_id"] == P.ACTION_BALL_TASK_ID
    assert receipt["manifest"] == {
        "path": "manifest.json",
        "sha256": _sha(manifest_path.read_bytes()),
    }
    assert receipt["runtime_contract"]["runtime_source"] == {
        "path": (
            "hope_training/whole_body_tracking/scripts/"
            "check_table_obstacle_scene.py"
        ),
        "sha256": _sha(source.read_bytes()),
    }
    assert receipt["runtime_contract"]["gpu_identity"]["nvml_verified"] is True
    assert [row["action_uid"] for row in receipt["actions"]] == [
        row.action_uid for row in inputs.motions
    ]
    P._validate_formal_receipt_document(receipt, inputs=inputs)

    output, _ = P._prepare_output_path(
        "receipt.json", repo_root=tmp_path
    )
    previous_umask = os.umask(0o077)
    try:
        file_sha = P._exclusive_publish_receipt(output, receipt)
    finally:
        os.umask(previous_umask)
    payload = output.read_bytes()
    assert file_sha == _sha(payload)
    assert payload == P._canonical_json_bytes(json.loads(payload))
    assert (os.stat(output).st_mode & 0o777) == 0o444
    with pytest.raises(FileExistsError):
        P._exclusive_publish_receipt(output, receipt)

    forged = dict(receipt)
    forged["task_id"] = "Tracking-Flat-AgibotA3-Hope-ActionBall-v0"
    with pytest.raises(P.TableSmokeReceiptError, match="identity is not exact"):
        P._validate_formal_receipt_document(forged)


def test_real_producer_receipt_roundtrips_through_canonical_admission(
    tmp_path,
):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    receipt = P._build_formal_receipt(
        inputs, _valid_evidence(inputs)
    )
    output, relative = P._prepare_output_path(
        "isaac_table_smoke.json", repo_root=tmp_path
    )
    receipt_sha = P._exclusive_publish_receipt(output, receipt)
    binding = SimpleNamespace(
        isaac_table_filtered_smoke_receipt_sha256=receipt_sha,
        motion_ids=tuple(P.FRESH_N5_ACTION_IDS),
        npz_sha256=tuple(row.file.sha256 for row in inputs.motions),
    )

    ADMISSION._validate_fresh_n5_isaac_table_smoke_receipt(
        {"path": relative, "sha256": receipt_sha},
        binding=binding,
        repo_root=tmp_path,
    )


def test_false_runtime_boolean_or_unsafe_action_cannot_seal_pass(tmp_path):
    source, manifest_path, _ = _fixture_tree(tmp_path)
    inputs = _load_fixture_inputs(
        source, manifest_path, repo_root=tmp_path
    )
    good = _valid_evidence(inputs)
    bad_boolean = P._RuntimeEvidence(
        **{**good.__dict__, "positive_control_pass": False}
    )
    with pytest.raises(P.TableSmokeReceiptError, match="required table-smoke"):
        P._build_formal_receipt(inputs, bad_boolean)

    first = good.actions[0]
    unsafe = P._RuntimeActionEvidence(
        **{**first.__dict__, "table_contact_count": 1, "unsafe_count": 1}
    )
    bad_action = P._RuntimeEvidence(
        **{**good.__dict__, "actions": (unsafe, *good.actions[1:])}
    )
    with pytest.raises(P.TableSmokeReceiptError, match="not zero"):
        P._build_formal_receipt(inputs, bad_action)
