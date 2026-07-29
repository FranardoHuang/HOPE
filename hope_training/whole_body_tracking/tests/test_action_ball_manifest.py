"""Host-only tests for the action-conditioned ball-first manifest."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "action_ball_manifest_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

ACTION_CATALOG_PATH = (
    ROOT.parent.parent
    / "hope_ws"
    / "src"
    / "hope_planner"
    / "hope_planner"
    / "action_catalog.py"
)
CATALOG_SPEC = importlib.util.spec_from_file_location(
    "action_catalog_for_action_ball_test", ACTION_CATALOG_PATH
)
assert CATALOG_SPEC is not None and CATALOG_SPEC.loader is not None
A = importlib.util.module_from_spec(CATALOG_SPEC)
sys.modules[CATALOG_SPEC.name] = A
CATALOG_SPEC.loader.exec_module(A)


def _curriculum(target=0.10, half_width=0.025):
    return {
        "min_proposals": 256,
        "min_safe_closed": 256,
        "target_failure_rate": target,
        "failure_band_half_width": half_width,
        "min_solver_admit_rate": 0.95,
        "min_install_rate": 0.95,
        "min_start_rate": 0.95,
        "min_close_rate": 0.95,
        "max_other_unsafe_rate": 0.02,
        "confidence_z": 1.96,
        "max_center_failures": 8,
    }


def _profile(index):
    return {
        "contact_offset_center_b_yaw_m": [
            0.55,
            -0.20 + index * 0.001,
            1.02,
        ],
        "contact_offset_std_lower_initial_m": [0.005, 0.01, 0.01],
        "contact_offset_std_lower_max_m": [0.05, 0.30, 0.20],
        "contact_offset_std_upper_initial_m": [0.004, 0.02, 0.01],
        "contact_offset_std_upper_max_m": [0.04, 0.25, 0.20],
        "contact_offset_min_b_yaw_m": [0.40, -0.60, 0.78],
        "contact_offset_max_b_yaw_m": [0.70, 0.60, 1.30],
        "time_to_contact_center_s": 1.20,
        "time_to_contact_std_lower_initial_s": 0.01,
        "time_to_contact_std_lower_max_s": 0.15,
        "time_to_contact_std_upper_initial_s": 0.02,
        "time_to_contact_std_upper_max_s": 0.30,
        "time_to_contact_min_s": 1.05,
        "time_to_contact_max_s": 1.60,
        "incoming_direction_center_b_yaw": [-1.0, 0.0, 0.0],
        "incoming_direction_tangent_u_b_yaw": [0.0, 1.0, 0.0],
        "incoming_direction_tangent_v_b_yaw": [0.0, 0.0, -1.0],
        "incoming_direction_tangent_u_neg_initial_deg": 0.5,
        "incoming_direction_tangent_u_neg_max_deg": 10.0,
        "incoming_direction_tangent_u_pos_initial_deg": 0.6,
        "incoming_direction_tangent_u_pos_max_deg": 8.0,
        "incoming_direction_tangent_v_neg_initial_deg": 0.7,
        "incoming_direction_tangent_v_neg_max_deg": 6.0,
        "incoming_direction_tangent_v_pos_initial_deg": 0.8,
        "incoming_direction_tangent_v_pos_max_deg": 7.0,
        "incoming_inbound_axis_b_yaw": [-1.0, 0.0, 0.0],
        "incoming_inbound_min_cosine": 0.20,
        "incoming_speed_center_mps": 3.0,
        "incoming_speed_std_lower_initial_mps": 0.05,
        "incoming_speed_std_lower_max_mps": 0.75,
        "incoming_speed_std_upper_initial_mps": 0.06,
        "incoming_speed_std_upper_max_mps": 0.50,
        "incoming_speed_min_mps": 1.2,
        "incoming_speed_max_mps": 5.0,
        "spin_direction_center_b_yaw": [0.0, 1.0, 0.0],
        "spin_direction_tangent_u_b_yaw": [0.0, 0.0, 1.0],
        "spin_direction_tangent_v_b_yaw": [1.0, 0.0, 0.0],
        "spin_direction_tangent_u_neg_initial_deg": 0.0,
        "spin_direction_tangent_u_neg_max_deg": 45.0,
        "spin_direction_tangent_u_pos_initial_deg": 0.0,
        "spin_direction_tangent_u_pos_max_deg": 40.0,
        "spin_direction_tangent_v_neg_initial_deg": 0.0,
        "spin_direction_tangent_v_neg_max_deg": 35.0,
        "spin_direction_tangent_v_pos_initial_deg": 0.0,
        "spin_direction_tangent_v_pos_max_deg": 30.0,
        "spin_magnitude_center_radps": 20.0,
        "spin_magnitude_std_lower_initial_radps": 1.0,
        "spin_magnitude_std_lower_max_radps": 15.0,
        "spin_magnitude_std_upper_initial_radps": 2.0,
        "spin_magnitude_std_upper_max_radps": 20.0,
        "spin_magnitude_min_radps": 0.0,
        "spin_magnitude_max_radps": 100.0,
        "base_spawn_center_w_xy_m": [0.0, 0.0],
        "base_spawn_std_lower_initial_m": [0.01, 0.01],
        "base_spawn_std_lower_max_m": [0.15, 0.25],
        "base_spawn_std_upper_initial_m": [0.02, 0.01],
        "base_spawn_std_upper_max_m": [0.10, 0.20],
        "base_spawn_min_w_xy_m": [-0.30, -0.40],
        "base_spawn_max_w_xy_m": [0.30, 0.40],
        "base_travel_center_b_yaw_xy_m": [0.0, 0.0],
        "base_travel_std_lower_initial_m": [0.0, 0.0],
        "base_travel_std_lower_max_m": [0.25, 0.20],
        "base_travel_std_upper_initial_m": [0.01, 0.02],
        "base_travel_std_upper_max_m": [0.30, 0.25],
        "base_travel_min_b_yaw_xy_m": [-0.50, -0.40],
        "base_travel_max_b_yaw_xy_m": [0.50, 0.40],
    }


def _action(index):
    action_id = f"action_{index:03d}"
    family = "forehand" if index % 2 == 0 else "backhand"
    motion_sha256 = hashlib.sha256(f"motion-{index}".encode()).hexdigest()
    return {
        "action_id": action_id,
        "action_uid": M.derive_action_ball_action_uid(
            action_id, family, motion_sha256
        ),
        "motion_path": f"motions/action_{index:03d}.npz",
        "motion_sha256": motion_sha256,
        "strike_phase": 0.50,
        "reference_t_hit_s": 0.80,
        "reference_t_cycle_s": 1.60,
        "reference_racket_site_speed_mps": 6.0,
        "reaction_margin_s": 0.05,
        "teacher_rate_min": 0.80,
        "teacher_rate_max": 1.20,
        "family": family,
        "mount_normal_sign": -1,
        "ball_profile": _profile(index),
    }


def _document(
    action_count=5,
    *,
    mobility_mode="no_move",
    target=0.10,
    half_width=0.025,
):
    actions = [_action(index) for index in range(action_count)]
    return {
        "schema_version": 3,
        "manifest_id": f"action_ball_n{action_count}_v3",
        "mobility_mode": mobility_mode,
        "action_order": [action["action_id"] for action in actions],
        "prototype": {
            "path": "configs/stroke_prototypes_v2.json",
            "sha256": hashlib.sha256(b"prototype").hexdigest(),
            "scope": "full",
        },
        "solver_profile_sha256": hashlib.sha256(b"solver").hexdigest(),
        "physics_profile_sha256": hashlib.sha256(b"physics").hexdigest(),
        "landing_aim": {
            "center_w_xy_m": [2.55, 0.0],
            "std_lower_initial_m": [0.01, 0.01],
            "std_lower_max_m": [0.25, 0.45],
            "std_upper_initial_m": [0.02, 0.01],
            "std_upper_max_m": [0.20, 0.40],
            "min_w_xy_m": [2.10, -0.60],
            "max_w_xy_m": [3.00, 0.60],
        },
        "actions": actions,
        "curriculum": _curriculum(target, half_width),
        "holdout": {
            "seed": 20260727,
            "samples_per_action": 768,
            "split_id": "heldout_ball_v1",
        },
        "notes": "Host-only schema fixture.",
    }


def _counter_rally_objective():
    return M._counter_rally_objective_profile_type()().to_mapping()


def _prototype_bytes(document):
    scope = document["prototype"]["scope"]
    rows = []
    for index, action in enumerate(document["actions"]):
        rows.append(
            {
                "motion_id": action["action_id"],
                "scope": scope,
                "clip_index": index,
                "npz_sha256": action["motion_sha256"],
                "family": action["family"],
                "face_sign": action["mount_normal_sign"],
                "frames": 5,
                "strike_phase": 0.5,
                "t_prepare_s": action["reference_t_hit_s"],
                "t_prepare_min_s": 0.1,
                "t_prepare_max_s": 2.0,
                "band_b_x": [0.1, 0.2],
                "band_b_y": [-0.2, 0.2],
                "band_z_w": [0.8, 1.2],
                "slack_b_xy_m": 0.1,
                "slack_z_w_m": 0.1,
                "p_contact_b": [0.2, 0.0, 1.0],
                "n_hat_b": [1.0, 0.0, 0.0],
                "priority": 0,
                "enabled": True,
                "contact_frame": 2,
                "contact_window_frames": [1, 3],
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
    scopes = {scope: rows}
    prototype = {
        "schema_version": 2,
        "prototype_set_id": "test_face_center_v2",
        "velocity_contract": {
            "direction_and_speed_point": (
                "selected_rubber_face_center"
            ),
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
                M._exact_face_geometry_source_sha256()
            ),
        },
        "contact_rule": {},
        "provenance": {},
        "scopes": scopes,
        "derived_sha256": M._prototype_canonical_sha256(scopes),
    }
    return (
        json.dumps(
            prototype,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _materialize_referenced_assets(repo_root, document):
    repo_root.mkdir(parents=True, exist_ok=True)
    prototype_path = repo_root / document["prototype"]["path"]
    prototype_path.parent.mkdir(parents=True, exist_ok=True)
    prototype_bytes = _prototype_bytes(document)
    prototype_path.write_bytes(prototype_bytes)
    document["prototype"]["sha256"] = hashlib.sha256(
        prototype_bytes
    ).hexdigest()
    for index, action in enumerate(document["actions"]):
        motion_path = repo_root / action["motion_path"]
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        motion_path.write_bytes(f"motion-{index}".encode())


def _write(tmp_path, document, *, name="manifest.json", **json_kwargs):
    path = tmp_path / name
    path.write_text(
        json.dumps(document, allow_nan=True, **json_kwargs),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_arbitrary_n_preserves_order_identity_and_profiles(
    tmp_path, action_count
):
    path = _write(tmp_path, _document(action_count))
    loaded = M.load_action_ball_manifest(path)
    manifest = loaded.manifest

    assert len(manifest.actions) == action_count
    assert manifest.action_order == tuple(
        f"action_{index:03d}" for index in range(action_count)
    )
    assert tuple(a.action_id for a in manifest.actions) == manifest.action_order
    assert manifest.mobility_mode == "no_move"
    assert manifest.landing_aim.center_w_xy_m == (2.55, 0.0)
    assert (
        manifest.actions[0].ball_profile.contact_offset_center_b_yaw_m
        == (0.55, -0.20, 1.02)
    )
    assert loaded.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.canonical_sha256 == M.canonical_manifest_sha256(manifest)

    with pytest.raises(FrozenInstanceError):
        manifest.manifest_id = "mutated"
    with pytest.raises(FrozenInstanceError):
        manifest.actions[
            0
        ].ball_profile.incoming_direction_tangent_u_neg_max_deg = 10.0


def test_file_sha_is_exact_startup_pin_and_canonical_sha_ignores_format(
    tmp_path,
):
    document = _document()
    compact = _write(
        tmp_path,
        document,
        name="compact.json",
        sort_keys=True,
        separators=(",", ":"),
    )
    pretty = _write(
        tmp_path,
        dict(reversed(tuple(document.items()))),
        name="pretty.json",
        indent=2,
    )

    compact_loaded = M.load_action_ball_manifest(compact)
    pretty_loaded = M.load_action_ball_manifest(pretty)
    assert compact_loaded.file_sha256 != pretty_loaded.file_sha256
    assert compact_loaded.canonical_sha256 == pretty_loaded.canonical_sha256
    assert (
        compact_loaded.manifest.to_mapping()
        == pretty_loaded.manifest.to_mapping()
    )

    assert (
        M.load_action_ball_manifest(
            compact, expected_sha256=compact_loaded.file_sha256
        ).file_sha256
        == compact_loaded.file_sha256
    )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        M.load_action_ball_manifest(compact, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="64 lowercase"):
        M.load_action_ball_manifest(compact, expected_sha256="A" * 64)


@pytest.mark.parametrize("old_version", [1, 2])
def test_old_schema_is_not_silently_interpreted_as_v3(
    tmp_path, old_version
):
    document = _document()
    document["schema_version"] = old_version
    with pytest.raises(ValueError, match=r"schema_version.*\[3, 3\]"):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_formal_holdout_rejects_512_and_accepts_768(tmp_path):
    document = _document()
    document["holdout"]["samples_per_action"] = 512
    with pytest.raises(
        ValueError,
        match=r"holdout\.samples_per_action.*at least 768",
    ):
        M.load_action_ball_manifest(_write(tmp_path, document))

    document = _document()
    document["holdout"]["samples_per_action"] = 768
    loaded = M.load_action_ball_manifest(_write(tmp_path, document))
    assert loaded.manifest.holdout.samples_per_action == 768


def test_holdout_must_cover_larger_manifest_curriculum_window(tmp_path):
    document = _document()
    document["curriculum"]["min_proposals"] = 769
    document["holdout"]["samples_per_action"] = 768
    with pytest.raises(
        ValueError,
        match=r"holdout\.samples_per_action.*at least 769",
    ):
        M.load_action_ball_manifest(_write(tmp_path, document))

    document = _document()
    document["curriculum"]["min_safe_closed"] = 769
    with pytest.raises(
        ValueError,
        match=r"holdout\.samples_per_action.*at least 769",
    ):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_referenced_asset_verification_hashes_prototype_and_every_motion(
    tmp_path,
):
    document = _document(5)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    path = _write(tmp_path, document)

    # Metadata-only review deliberately does not require local assets.
    review = M.load_action_ball_manifest(path)
    assert review.referenced_assets is None

    loaded = M.load_action_ball_manifest(
        path,
        verify_referenced_assets=True,
        repo_root=repo_root,
    )
    assets = loaded.referenced_assets
    assert assets is not None
    assert assets.repo_root == repo_root.resolve()
    assert assets.prototype.relative_path == document["prototype"]["path"]
    assert assets.prototype.sha256 == document["prototype"]["sha256"]
    assert tuple(asset.relative_path for asset in assets.motions) == tuple(
        action["motion_path"] for action in document["actions"]
    )
    assert tuple(asset.sha256 for asset in assets.motions) == tuple(
        action["motion_sha256"] for action in document["actions"]
    )
    assert all(asset.resolved_path.is_file() for asset in assets.motions)

    explicit = M.verify_action_ball_referenced_assets(
        review.manifest, repo_root=repo_root
    )
    assert explicit == assets
    with pytest.raises(
        M.ActionBallManifestAdmissionError,
        match="referenced-byte verification is not code-rooted",
    ):
        M.load_action_ball_manifest(
            path,
            verify_referenced_assets=True,
            repo_root=repo_root,
            require_formal_admission=True,
        )


def test_prototype_motion_sha_must_equal_manifest_motion_sha(tmp_path):
    document = _document(2)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    prototype_path = repo_root / document["prototype"]["path"]
    prototype = json.loads(prototype_path.read_text(encoding="utf-8"))
    prototype["scopes"]["full"][1]["npz_sha256"] = "0" * 64
    prototype["derived_sha256"] = M._prototype_canonical_sha256(
        prototype["scopes"]
    )
    raw = (
        json.dumps(
            prototype,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    prototype_path.write_bytes(raw)
    document["prototype"]["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(
        ValueError,
        match="NPZ SHA differs from manifest motion_sha256",
    ):
        M.load_action_ball_manifest(
            _write(tmp_path, document),
            verify_referenced_assets=True,
            repo_root=repo_root,
        )


def test_legacy_site_velocity_prototype_is_not_action_ball_admissible(
    tmp_path,
):
    document = _document(1)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    prototype_path = repo_root / document["prototype"]["path"]
    prototype = json.loads(prototype_path.read_text(encoding="utf-8"))
    prototype["schema_version"] = 1
    raw = (
        json.dumps(
            prototype,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    prototype_path.write_bytes(raw)
    document["prototype"]["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(
        ValueError,
        match="legacy site-velocity prototype rows are not admissible",
    ):
        M.load_action_ball_manifest(
            _write(tmp_path, document),
            verify_referenced_assets=True,
            repo_root=repo_root,
        )


def test_prototype_geometry_source_sha_must_match_runtime(tmp_path):
    document = _document(1)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    prototype_path = repo_root / document["prototype"]["path"]
    prototype = json.loads(prototype_path.read_text(encoding="utf-8"))
    prototype["velocity_contract"]["geometry_source_sha256"] = "0" * 64
    raw = (
        json.dumps(
            prototype,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    prototype_path.write_bytes(raw)
    document["prototype"]["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(
        ValueError,
        match="current selected-rubber face-centre geometry",
    ):
        M.load_action_ball_manifest(
            _write(tmp_path, document),
            verify_referenced_assets=True,
            repo_root=repo_root,
        )


def test_asset_verification_flag_and_repo_root_are_explicit(tmp_path):
    path = _write(tmp_path, _document())
    with pytest.raises(ValueError, match="requires repo_root"):
        M.load_action_ball_manifest(
            path, verify_referenced_assets=True
        )
    with pytest.raises(ValueError, match="only accepted"):
        M.load_action_ball_manifest(path, repo_root=tmp_path)
    with pytest.raises(ValueError, match="must be a bool"):
        M.load_action_ball_manifest(
            path,
            verify_referenced_assets=1,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="existing directory"):
        M.verify_action_ball_referenced_assets(
            M.load_action_ball_manifest(path).manifest,
            repo_root=tmp_path / "missing",
        )


@pytest.mark.parametrize("which", ["prototype", "first_motion", "last_motion"])
def test_sha_drifted_referenced_asset_fails_closed(
    tmp_path, which
):
    document = _document(5)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    if which == "prototype":
        path_to_change = repo_root / document["prototype"]["path"]
        expected_label = "prototype"
    elif which == "first_motion":
        path_to_change = repo_root / document["actions"][0]["motion_path"]
        expected_label = r"motion\[action_000\]"
    else:
        path_to_change = repo_root / document["actions"][-1]["motion_path"]
        expected_label = r"motion\[action_004\]"
    path_to_change.write_bytes(b"drifted bytes")
    with pytest.raises(
        ValueError,
        match=expected_label + " referenced asset SHA-256 mismatch",
    ):
        M.load_action_ball_manifest(
            _write(tmp_path, document),
            verify_referenced_assets=True,
            repo_root=repo_root,
        )


def test_missing_referenced_asset_fails_closed(tmp_path):
    document = _document(5)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    missing = repo_root / document["actions"][-1]["motion_path"]
    missing.unlink()
    with pytest.raises(
        ValueError,
        match=r"motion\[action_004\] referenced asset does not resolve",
    ):
        M.load_action_ball_manifest(
            _write(tmp_path, document),
            verify_referenced_assets=True,
            repo_root=repo_root,
        )


def test_referenced_asset_must_resolve_to_regular_file(tmp_path):
    document = _document(1)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    prototype = repo_root / document["prototype"]["path"]
    prototype.unlink()
    prototype.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        M.load_action_ball_manifest(
            _write(tmp_path, document),
            verify_referenced_assets=True,
            repo_root=repo_root,
        )


def test_referenced_asset_symlink_cannot_escape_repo_root(tmp_path):
    document = _document(1)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    prototype = repo_root / document["prototype"]["path"]
    prototype.unlink()
    outside = tmp_path / "outside-prototype.json"
    outside.write_bytes(b"prototype")
    prototype.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes repo_root through a symlink"):
        M.load_action_ball_manifest(
            _write(tmp_path, document),
            verify_referenced_assets=True,
            repo_root=repo_root,
        )


def test_internal_symlink_is_allowed_but_still_hashed(tmp_path):
    document = _document(1)
    repo_root = tmp_path / "repo"
    _materialize_referenced_assets(repo_root, document)
    prototype = repo_root / document["prototype"]["path"]
    prototype_bytes = prototype.read_bytes()
    prototype.unlink()
    stored = repo_root / "objects" / "prototype.json"
    stored.parent.mkdir()
    stored.write_bytes(prototype_bytes)
    prototype.symlink_to(stored)

    loaded = M.load_action_ball_manifest(
        _write(tmp_path, document),
        verify_referenced_assets=True,
        repo_root=repo_root,
    )
    assert loaded.referenced_assets is not None
    assert loaded.referenced_assets.prototype.resolved_path == stored.resolve()


def test_schema_has_no_self_authorization_and_formal_launch_fails_closed(
    tmp_path,
):
    path = _write(tmp_path, _document())
    with pytest.raises(
        M.ActionBallManifestAdmissionError,
        match="executable launch boundary.*opaque motion admission",
    ):
        M.load_action_ball_manifest(
            path, require_formal_admission=True
        )
    with pytest.raises(ValueError, match="must be a bool"):
        M.load_action_ball_manifest(
            path, require_formal_admission=1
        )


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("top", "extra"),
        ("prototype", "extra"),
        ("landing_aim", "extra"),
        ("action", "extra"),
        ("profile", "extra"),
        ("curriculum", "extra"),
        ("holdout", "extra"),
    ],
)
def test_unknown_keys_are_rejected_at_every_level(
    tmp_path, scope, field
):
    document = _document()
    targets = {
        "top": document,
        "prototype": document["prototype"],
        "landing_aim": document["landing_aim"],
        "action": document["actions"][0],
        "profile": document["actions"][0]["ball_profile"],
        "curriculum": document["curriculum"],
        "holdout": document["holdout"],
    }
    targets[scope][field] = 1
    with pytest.raises(ValueError, match="invalid keys"):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("top", "notes"),
        ("prototype", "scope"),
        ("landing_aim", "max_w_xy_m"),
        ("action", "strike_phase"),
        ("profile", "incoming_speed_center_mps"),
        ("curriculum", "target_failure_rate"),
        ("holdout", "split_id"),
    ],
)
def test_missing_keys_are_rejected_at_every_level(
    tmp_path, scope, field
):
    document = _document()
    targets = {
        "top": document,
        "prototype": document["prototype"],
        "landing_aim": document["landing_aim"],
        "action": document["actions"][0],
        "profile": document["actions"][0]["ball_profile"],
        "curriculum": document["curriculum"],
        "holdout": document["holdout"],
    }
    del targets[scope][field]
    with pytest.raises(ValueError, match="invalid keys"):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (("schema_version",), True, "plain integer"),
        (("actions", 0, "action_uid"), True, "plain integer"),
        (("actions", 0, "strike_phase"), True, "plain finite"),
        (
            ("actions", 0, "ball_profile", "incoming_speed_center_mps"),
            True,
            "plain finite",
        ),
        (
            (
                "actions",
                0,
                "ball_profile",
                "contact_offset_std_lower_max_m",
                0,
            ),
            True,
            "plain finite",
        ),
        (
            ("landing_aim", "std_upper_max_m", 0),
            True,
            "plain finite",
        ),
        (("curriculum", "min_proposals"), True, "plain integer"),
        (("holdout", "seed"), True, "plain integer"),
        (("notes",), False, "must be a string"),
    ],
)
def test_bool_is_never_accepted_as_number_or_text(
    tmp_path, location, value, message
):
    document = _document()
    target = document
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize("constant", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_numbers_are_rejected(tmp_path, constant):
    document = _document()
    document["actions"][0]["ball_profile"][
        "incoming_speed_center_mps"
    ] = constant
    with pytest.raises(ValueError, match="JSON constant"):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_duplicate_json_keys_are_rejected(tmp_path):
    encoded = json.dumps(_document())
    duplicate = encoded[:-1] + ',"notes":"duplicate"}'
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        M.load_action_ball_manifest(path)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/abs/motion.npz",
        "../motion.npz",
        "motions/../../motion.npz",
        r"C:\motions\motion.npz",
        r"motions\..\motion.npz",
        "./motions/motion.npz",
        "motions//motion.npz",
        "motions/motion.npz/",
        ".",
    ],
)
@pytest.mark.parametrize("field", ["motion_path", "prototype.path"])
def test_paths_are_normalized_relative_posix_and_cannot_escape(
    tmp_path, invalid_path, field
):
    document = _document()
    if field == "motion_path":
        document["actions"][0]["motion_path"] = invalid_path
    else:
        document["prototype"]["path"] = invalid_path
    with pytest.raises(ValueError, match="path"):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("action_id", "family", "motion_sha256", "expected_uid"),
    [
        (
            "fh_loop_high",
            "forehand",
            "7d045fcb036ffa668dede4607cfcc82e789a0db7ab86fd8df9dd52cfd5ac4153",
            3383780318617471,
        ),
        (
            "bh_loop_c",
            "backhand",
            "0d49cdd54c8aeffcc98cb9b4b22dff476535323251d8ea191205756542941617",
            3539572639101871,
        ),
    ],
)
def test_uid_derivation_is_byte_compatible_with_planner_catalog(
    action_id, family, motion_sha256, expected_uid
):
    uid = M.derive_action_ball_action_uid(
        action_id, family, motion_sha256
    )
    assert uid == expected_uid
    assert uid == A.derive_action_uid(action_id, family, motion_sha256)
    assert 1 <= uid <= M.MAX_ACTION_UID


@pytest.mark.parametrize("drift", ["action_id", "family", "motion_sha256"])
def test_action_uid_rejects_identity_drift(tmp_path, drift):
    document = _document()
    action = document["actions"][0]
    if drift == "action_id":
        action["action_id"] = "renamed_action"
        document["action_order"][0] = action["action_id"]
    elif drift == "family":
        action["family"] = "backhand"
    else:
        action["motion_sha256"] = hashlib.sha256(
            b"replacement"
        ).hexdigest()
    with pytest.raises(ValueError, match="canonical action identity"):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_order_duplicates_empty_bank_and_uid_range_fail_closed(tmp_path):
    swapped = _document()
    swapped["actions"][0], swapped["actions"][1] = (
        swapped["actions"][1],
        swapped["actions"][0],
    )
    with pytest.raises(ValueError, match="same IDs and order"):
        M.load_action_ball_manifest(
            _write(tmp_path, swapped, name="swapped.json")
        )

    duplicate = _document()
    duplicate["action_order"][1] = duplicate["action_order"][0]
    with pytest.raises(ValueError, match="duplicate action IDs"):
        M.load_action_ball_manifest(
            _write(tmp_path, duplicate, name="duplicate-id.json")
        )

    empty = _document(1)
    empty["action_order"] = []
    empty["actions"] = []
    with pytest.raises(ValueError, match="at least one"):
        M.load_action_ball_manifest(
            _write(tmp_path, empty, name="empty.json")
        )

    for uid in (0, M.MAX_ACTION_UID + 1):
        invalid = _document()
        invalid["actions"][0]["action_uid"] = uid
        with pytest.raises(ValueError, match="action_uid"):
            M.load_action_ball_manifest(
                _write(tmp_path, invalid, name=f"uid-{uid}.json")
            )


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (
            ("contact_offset_std_lower_initial_m", 1),
            0.31,
            r"initial\[1\].*<= max",
        ),
        (
            ("contact_offset_std_lower_max_m", 0),
            0.11,
            r"must be <= 0.1",
        ),
        (
            ("contact_offset_std_lower_max_m",),
            [0.05, 0.04, 0.20],
            "x std must not exceed y",
        ),
        (
            ("contact_offset_center_b_yaw_m", 2),
            1.31,
            "center must lie inside bounds",
        ),
        (
            ("incoming_direction_center_b_yaw",),
            [-2.0, 0.0, 0.0],
            "unit length",
        ),
        (
            ("incoming_direction_tangent_u_neg_max_deg",),
            181.0,
            "must be <= 180",
        ),
        (
            ("incoming_speed_center_mps",),
            0.0,
            "must be > 0",
        ),
        (
            ("incoming_speed_min_mps",),
            5.0,
            "max strictly above min",
        ),
        (
            ("incoming_speed_center_mps",),
            6.0,
            "inside the speed range",
        ),
        (
            ("incoming_speed_std_lower_initial_mps",),
            0.76,
            "initial.*<=.*u_neg_max|lower.*initial.*<=.*max",
        ),
        (
            ("spin_direction_center_b_yaw",),
            [0.0, 0.0, 0.0],
            "unit length",
        ),
        (
            ("spin_direction_tangent_u_neg_initial_deg",),
            46.0,
            "u_neg_initial.*<=.*u_neg_max",
        ),
        (
            ("spin_direction_tangent_u_neg_max_deg",),
            181.0,
            "must be <= 180",
        ),
        (
            ("spin_magnitude_center_radps",),
            101.0,
            "inside the spin range",
        ),
        (
            ("spin_magnitude_std_lower_initial_radps",),
            16.0,
            "lower.*initial.*<=.*max",
        ),
        (
            ("base_spawn_std_lower_initial_m", 0),
            0.16,
            r"initial\[0\].*<= max",
        ),
        (
            ("base_spawn_center_w_xy_m", 0),
            0.31,
            "center must lie inside bounds",
        ),
        (
            ("base_travel_std_lower_initial_m", 0),
            0.26,
            r"initial\[0\].*<= max",
        ),
        (
            ("base_travel_center_b_yaw_xy_m", 0),
            0.51,
            "center must lie inside bounds",
        ),
    ],
)
def test_ball_domain_cross_field_constraints(
    tmp_path, location, value, message
):
    document = _document()
    profile = document["actions"][0]["ball_profile"]
    target = profile
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_schema_v3_exposes_exactly_32_independent_curriculum_arms(tmp_path):
    manifest = M.load_action_ball_manifest(
        _write(tmp_path, _document(1))
    ).manifest
    ball = manifest.actions[0].ball_profile
    aim = manifest.landing_aim

    euclidean_side_dimensions = (
        1  # time_to_contact
        + 3  # contact offset
        + 1  # incoming speed
        + 1  # spin magnitude
        + 2  # base spawn
        + 2  # base travel
        + 2  # landing aim
    )
    direction_side_dimensions = 4 + 4  # u-/u+/v-/v+ twice
    assert 2 * euclidean_side_dimensions + direction_side_dimensions == 32

    # Distinct values must survive strict parse/round-trip; no side may be
    # reconstructed from its opposite side.
    assert ball.time_to_contact_std_lower_initial_s == 0.01
    assert ball.time_to_contact_std_upper_initial_s == 0.02
    assert ball.contact_offset_std_lower_max_m != (
        ball.contact_offset_std_upper_max_m
    )
    assert ball.incoming_speed_std_lower_max_mps == 0.75
    assert ball.incoming_speed_std_upper_max_mps == 0.50
    assert ball.spin_magnitude_std_lower_max_radps == 15.0
    assert ball.spin_magnitude_std_upper_max_radps == 20.0
    assert ball.base_spawn_std_lower_max_m != (
        ball.base_spawn_std_upper_max_m
    )
    assert ball.base_travel_std_lower_max_m != (
        ball.base_travel_std_upper_max_m
    )
    assert aim.std_lower_max_m != aim.std_upper_max_m
    assert manifest.to_mapping() == _document(1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "time_to_contact_min_s",
            1.049,
            r"reference_t_hit_s / teacher_rate_min",
        ),
        (
            "time_to_contact_max_s",
            1.667,
            r"maximum pre_swing_wait.*<= 1.0",
        ),
    ],
)
def test_time_to_contact_covers_every_teacher_rate_without_retiming_motion(
    tmp_path, field, value, message
):
    document = _document(1)
    document["actions"][0]["ball_profile"][field] = value
    if field == "time_to_contact_max_s":
        document["actions"][0]["ball_profile"][
            "time_to_contact_std_upper_max_s"
        ] = 0.30
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_t_hit_s", 0.0, "reference_t_hit_s must be > 0"),
        (
            "reference_t_cycle_s",
            0.8,
            "reference_t_cycle_s must be > reference_t_hit_s",
        ),
        (
            "reference_racket_site_speed_mps",
            0.0,
            "reference_racket_site_speed_mps must be > 0",
        ),
        ("teacher_rate_min", 0.0, "teacher_rate_min must be > 0"),
        (
            "teacher_rate_min",
            1.01,
            "rate range must contain.*1.0",
        ),
        (
            "teacher_rate_max",
            0.99,
            "rate range must contain.*1.0",
        ),
    ],
)
def test_reference_teacher_metadata_is_strict_but_not_self_authorizing(
    tmp_path, field, value, message
):
    document = _document(1)
    document["actions"][0][field] = value
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_incoming_speed_floor_is_exactly_point_four_of_center(tmp_path):
    for invalid_min in (1.19, 1.21):
        document = _document(1)
        document["actions"][0]["ball_profile"][
            "incoming_speed_min_mps"
        ] = invalid_min
        with pytest.raises(ValueError, match="exactly 0.4 times"):
            M.load_action_ball_manifest(
                _write(tmp_path, document, name=f"speed-{invalid_min}.json")
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "incoming_direction_tangent_u_b_yaw",
            [-1.0, 0.0, 0.0],
            "tangent_u must be orthogonal",
        ),
        (
            "incoming_direction_tangent_v_b_yaw",
            [0.0, 0.0, 1.0],
            r"cross\(u,v\)=center",
        ),
        (
            "incoming_direction_tangent_u_neg_max_deg",
            80.0,
            "violates the inbound cone contract",
        ),
        (
            "incoming_inbound_min_cosine",
            1.0,
            "incoming_inbound_min_cosine must be < 1",
        ),
    ],
)
def test_fixed_tangent_frame_and_full_support_are_inbound(
    tmp_path, field, value, message
):
    document = _document(1)
    document["actions"][0]["ball_profile"][field] = value
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (
            ("std_lower_initial_m", 0),
            0.26,
            r"initial\[0\].*<= max",
        ),
        (("center_w_xy_m", 0), 3.01, "center must lie inside bounds"),
        (
            ("min_w_xy_m", 1),
            0.70,
            "upper bound must be >= lower bound",
        ),
    ],
)
def test_landing_aim_is_a_bounded_curriculum_domain(
    tmp_path, location, value, message
):
    document = _document()
    target = document["landing_aim"]
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_degenerate_spin_range_requires_zero_asymmetric_std(tmp_path):
    valid = _document()
    profile = valid["actions"][0]["ball_profile"]
    profile["spin_magnitude_center_radps"] = 0.0
    profile["spin_magnitude_min_radps"] = 0.0
    profile["spin_magnitude_max_radps"] = 0.0
    profile["spin_magnitude_std_lower_initial_radps"] = 0.0
    profile["spin_magnitude_std_lower_max_radps"] = 0.0
    profile["spin_magnitude_std_upper_initial_radps"] = 0.0
    profile["spin_magnitude_std_upper_max_radps"] = 0.0
    loaded = M.load_action_ball_manifest(
        _write(tmp_path, valid, name="zero-spin.json")
    )
    assert (
        loaded.manifest.actions[
            0
        ].ball_profile.spin_magnitude_std_upper_max_radps
        == 0.0
    )

    invalid = deepcopy(valid)
    invalid["actions"][0]["ball_profile"][
        "spin_magnitude_std_upper_max_radps"
    ] = 1.0
    with pytest.raises(ValueError, match="exceeds center-to-max support"):
        M.load_action_ball_manifest(
            _write(tmp_path, invalid, name="bad-zero-spin.json")
        )


def test_no_move_and_move_semantics_are_structural(tmp_path):
    no_move_doc = _document(1, mobility_mode="no_move")
    move_doc = _document(1, mobility_mode="move")
    no_move = M.load_action_ball_manifest(
        _write(tmp_path, no_move_doc, name="no-move.json")
    )
    move = M.load_action_ball_manifest(
        _write(tmp_path, move_doc, name="move.json")
    )
    assert no_move.manifest.mobility_mode == "no_move"
    assert move.manifest.mobility_mode == "move"
    assert no_move.canonical_sha256 != move.canonical_sha256
    assert (
        no_move.manifest.actions[0].ball_profile.to_mapping()
        == move.manifest.actions[0].ball_profile.to_mapping()
    )
    # Both modes bind the same non-zero latent travel tape.  no_move ignores
    # its realization at runtime; it does not alter the distribution.
    assert no_move.manifest.actions[
        0
    ].ball_profile.base_travel_std_lower_max_m == (0.25, 0.20)

    per_action_override = _document(1)
    per_action_override["actions"][0]["ball_profile"][
        "mobility_mode"
    ] = "move"
    with pytest.raises(ValueError, match="invalid keys.*mobility_mode"):
        M.load_action_ball_manifest(
            _write(
                tmp_path,
                per_action_override,
                name="per-action-mode-override.json",
            )
        )

    invalid_mode = _document(1)
    invalid_mode["mobility_mode"] = "maybe"
    with pytest.raises(ValueError, match="no_move.*move"):
        M.load_action_ball_manifest(
            _write(tmp_path, invalid_mode, name="bad-mode.json")
        )


def test_default_is_ten_percent_and_twenty_percent_is_distinct_manifest(
    tmp_path,
):
    ten = M.load_action_ball_manifest(
        _write(tmp_path, _document(), name="ten.json")
    )
    twenty = M.load_action_ball_manifest(
        _write(
            tmp_path,
            _document(target=0.20, half_width=0.05),
            name="twenty.json",
        )
    )
    assert ten.manifest.curriculum.failure_band == pytest.approx(
        (0.075, 0.125)
    )
    assert twenty.manifest.curriculum.failure_band == pytest.approx(
        (0.15, 0.25)
    )
    assert ten.canonical_sha256 != twenty.canonical_sha256
    defaults = M.ActionBallCurriculumConfig()
    assert defaults.target_failure_rate == 0.10
    assert defaults.failure_band_half_width == 0.025


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (("target_failure_rate",), 1.1, "must be <= 1"),
        (
            ("failure_band_half_width",),
            0.11,
            "band must lie inside",
        ),
        (("min_safe_closed",), 0, "must be >= 1"),
        (("min_solver_admit_rate",), -0.01, "must be >= 0"),
        (("min_install_rate",), 1.01, "must be <= 1"),
        (("min_start_rate",), -0.01, "must be >= 0"),
        (("min_close_rate",), 1.01, "must be <= 1"),
        (("max_other_unsafe_rate",), 1.01, "must be <= 1"),
        (("max_center_failures",), 0, "must be >= 1"),
    ],
)
def test_curriculum_bounds_fail_closed(
    tmp_path, location, value, message
):
    document = _document()
    target = document["curriculum"]
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", -1, "holdout.seed"),
        ("seed", 1 << 63, "holdout.seed"),
        ("samples_per_action", 0, "samples_per_action"),
        ("split_id", "", "non-empty"),
    ],
)
def test_holdout_is_pinned_and_bounded(
    tmp_path, field, value, message
):
    document = _document()
    document["holdout"][field] = value
    with pytest.raises(ValueError, match=message):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prototype.sha256", "A" * 64),
        ("solver_profile_sha256", "0" * 63),
        ("physics_profile_sha256", True),
    ],
)
def test_all_global_sha_bindings_are_strict_lowercase(
    tmp_path, field, value
):
    document = _document()
    if field == "prototype.sha256":
        document["prototype"]["sha256"] = value
    else:
        document[field] = value
    with pytest.raises(ValueError, match="64 lowercase"):
        M.load_action_ball_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    "action_id",
    (
        "fore\u0000hand",
        "fore\u200bhand",
        "fore\u0301hand",
    ),
)
def test_action_identity_rejects_nonportable_unicode(
    tmp_path, action_id
):
    document = _document()
    document["action_order"][0] = action_id
    document["actions"][0]["action_id"] = action_id
    with pytest.raises(ValueError):
        M.load_action_ball_manifest(_write(tmp_path, document))


def test_canonical_helper_rejects_unvalidated_objects():
    with pytest.raises(TypeError, match="ActionBallManifest"):
        M.canonical_manifest_bytes(_document())


def test_counter_rally_objective_is_exact_n1_and_legacy_bytes_stay_unchanged(
    tmp_path,
):
    legacy_document = _document(action_count=1)
    legacy = M.load_action_ball_manifest(
        _write(tmp_path, legacy_document, name="legacy.json")
    ).manifest
    assert legacy.counter_rally_objective is None
    assert legacy.to_mapping() == legacy_document

    objective_document = deepcopy(legacy_document)
    objective_document["counter_rally_objective"] = (
        _counter_rally_objective()
    )
    objective = M.load_action_ball_manifest(
        _write(tmp_path, objective_document, name="objective.json")
    ).manifest
    assert objective.counter_rally_objective.mode == "counter_rally_v1"
    assert objective.to_mapping() == objective_document

    invalid_n5 = _document(action_count=5)
    invalid_n5["counter_rally_objective"] = _counter_rally_objective()
    with pytest.raises(ValueError, match="exact N=1"):
        M.load_action_ball_manifest(
            _write(tmp_path, invalid_n5, name="invalid_n5.json")
        )


def test_counter_rally_objective_rejects_unknown_or_unreviewed_fields(
    tmp_path,
):
    document = _document(action_count=1)
    document["counter_rally_objective"] = _counter_rally_objective()
    document["counter_rally_objective"]["unknown"] = 1
    with pytest.raises(ValueError, match="keys mismatch"):
        M.load_action_ball_manifest(
            _write(tmp_path, document, name="unknown_objective.json")
        )


def _n5_with_counter_rally_candidates():
    document = _document(action_count=5)
    for index, action_id in enumerate(("bh_loop_c", "bh_block")):
        action = document["actions"][index]
        action["action_id"] = action_id
        action["family"] = "backhand"
        action["action_uid"] = M.derive_action_ball_action_uid(
            action_id, action["family"], action["motion_sha256"]
        )
        document["action_order"][index] = action_id
    return document


def test_pure_n1_subset_producer_preserves_exact_selected_action_rows(
    tmp_path,
):
    document = _n5_with_counter_rally_candidates()
    source_path = _write(tmp_path, document, name="source_n5.json")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = M.load_action_ball_manifest(
        source_path, expected_sha256=source_sha
    )
    profile = M._counter_rally_objective_profile_type()()
    for action_id in ("bh_loop_c", "bh_block"):
        subset = M.build_counter_rally_n1_subset(
            source,
            expected_source_file_sha256=source_sha,
            action_id=action_id,
            counter_rally_objective=profile,
        )
        source_row = next(
            row for row in document["actions"]
            if row["action_id"] == action_id
        )
        assert subset.action_order == (action_id,)
        assert subset.actions[0].to_mapping() == source_row
        assert subset.counter_rally_objective.sha256 == profile.sha256
        assert subset.counter_rally_objective.inactive_curriculum_arms == (
            "landing_aim_y_lower",
            "landing_aim_y_upper",
        )
        assert len(M.canonical_manifest_sha256(subset)) == 64


def test_no_clobber_n1_writer_emits_two_separate_strict_manifests(
    tmp_path,
):
    document = _n5_with_counter_rally_candidates()
    source_path = _write(tmp_path, document, name="source_n5.json")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    profile = M._counter_rally_objective_profile_type()()
    outputs = []
    for action_id in ("bh_loop_c", "bh_block"):
        destination = tmp_path / f"{action_id}.counter_rally.json"
        loaded = M.write_counter_rally_n1_subset_no_clobber(
            source_path,
            expected_source_file_sha256=source_sha,
            action_id=action_id,
            counter_rally_objective=profile,
            output_path=destination,
        )
        outputs.append(loaded)
        assert loaded.manifest.action_order == (action_id,)
        assert len(loaded.manifest.actions) == 1
        with pytest.raises(FileExistsError):
            M.write_counter_rally_n1_subset_no_clobber(
                source_path,
                expected_source_file_sha256=source_sha,
                action_id=action_id,
                counter_rally_objective=profile,
                output_path=destination,
            )
    assert outputs[0].file_sha256 != outputs[1].file_sha256
    assert outputs[0].manifest.action_order != outputs[1].manifest.action_order


def test_n1_subset_producer_rejects_n2_unknown_action_and_wrong_source_sha(
    tmp_path,
):
    document = _n5_with_counter_rally_candidates()
    source_path = _write(tmp_path, document, name="source_n5.json")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = M.load_action_ball_manifest(source_path)
    profile = M._counter_rally_objective_profile_type()()
    with pytest.raises(ValueError, match="exact-byte"):
        M.build_counter_rally_n1_subset(
            source,
            expected_source_file_sha256="0" * 64,
            action_id="bh_loop_c",
            counter_rally_objective=profile,
        )
    with pytest.raises(ValueError, match="bh_loop_c or bh_block"):
        M.build_counter_rally_n1_subset(
            source,
            expected_source_file_sha256=source_sha,
            action_id="action_002",
            counter_rally_objective=profile,
        )
    n2_path = _write(
        tmp_path, _document(action_count=2), name="source_n2.json"
    )
    n2 = M.load_action_ball_manifest(n2_path)
    with pytest.raises(ValueError, match="exact N=5"):
        M.build_counter_rally_n1_subset(
            n2,
            expected_source_file_sha256=n2.file_sha256,
            action_id="bh_loop_c",
            counter_rally_objective=profile,
        )


def test_n1_subset_cli_reports_output_and_bound_hashes(tmp_path, capsys):
    source_path = _write(
        tmp_path,
        _n5_with_counter_rally_candidates(),
        name="source_cli_n5.json",
    )
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_path = tmp_path / "bh_loop_c.cli.json"
    assert (
        M._counter_rally_subset_cli(
            (
                "--source",
                str(source_path),
                "--source-sha256",
                source_sha,
                "--action-id",
                "bh_loop_c",
                "--output",
                str(output_path),
            )
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["output_path"] == str(output_path)
    assert report["action_order"] == ["bh_loop_c"]
    assert report["inactive_curriculum_arms"] == [
        "landing_aim_y_lower",
        "landing_aim_y_upper",
    ]
    assert report["file_sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert len(report["canonical_sha256"]) == 64
    assert len(report["objective_profile_sha256"]) == 64
