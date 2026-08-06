"""Focused CPU proofs for the contact-only N=1 bundle producer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
SCRIPT = WBT_ROOT / "scripts" / "materialize_n1_contact_training_bundle.py"
PINNER = WBT_ROOT / "scripts" / "pin_action_ball_profile_contracts.py"
TABLE_GEOMETRY_RELATIVE = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/geometry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_n1_contact_training_bundle_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = B
SPEC.loader.exec_module(B)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = B._json_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _make_repo(tmp_path: Path, action_id: str) -> dict[str, object]:
    root = tmp_path / "repo"
    mdp_source = REPO_ROOT / B.MDP_RELATIVE_DIR
    mdp_target = root / B.MDP_RELATIVE_DIR
    source_names = (
        "hope_commands.py",
        "continuous_questions.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "strike_spec_torch.py",
        "stroke_prototypes_torch.py",
        "virtual_ball.py",
        "counter_rally.py",
        "counter_rally_torch.py",
        "action_ball_curriculum.py",
        "action_ball_sampling.py",
        "action_ball_profile_adapter.py",
        # Solver profile v3: the pinner reads the per-symbol semantic surface.
        "action_ball_solver_semantic_surface.py",
    )
    for name in (*source_names, "action_ball_manifest.py"):
        _copy(mdp_source / name, mdp_target / name)
    _copy(
        REPO_ROOT / B.SOURCE_MANIFEST_RELATIVE_PATH,
        root / B.SOURCE_MANIFEST_RELATIVE_PATH,
    )
    facts = B.SUPPORTED_ACTIONS[action_id]
    motion_relative = Path(facts["motion_path"])
    _copy(REPO_ROOT / motion_relative, root / motion_relative)
    venue_relative = Path("configs/ball_physics_venue.yaml")
    _copy(REPO_ROOT / venue_relative, root / venue_relative)
    _copy(
        REPO_ROOT / TABLE_GEOMETRY_RELATIVE,
        root / TABLE_GEOMETRY_RELATIVE,
    )

    counter = B._load_module(
        f"fixture_counter_{action_id}",
        mdp_target / "counter_rally.py",
    )
    objective_sha = counter.CounterRallyObjectiveProfile().sha256
    pins_path = root / "configs/profile_pins_counter_n1.json"
    pin_result = subprocess.run(
        [
            sys.executable,
            str(PINNER),
            "--repo-root",
            str(root),
            "--out",
            str(pins_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert pin_result.returncode == 0, pin_result.stderr
    pins = json.loads(pins_path.read_text())
    pins_sha = B._sha256_file(pins_path)
    solver_sha = pins["solver_profile_sha256"]
    physics_sha = pins["physics_profile_sha256"]
    source_path = root / B.SOURCE_MANIFEST_RELATIVE_PATH
    dynamic_ready_path = (
        root
        / "configs/a3_dynamic_ready"
        / f"{action_id}.dynamic_ready.v1.json"
    )
    dynamic_ready = {
        "schema_version": 2,
        "kind": B.DYNAMIC_READY_KIND,
        "action_id": action_id,
        "robot": {
            "family": "AgiBot A3",
            "joint_names": [f"joint_{index}" for index in range(31)],
        },
        "sources": {
            "stable_motion": {
                "path": f"/workspace/fixture/{motion_relative.as_posix()}",
                "sha256": facts["motion_sha256"],
                "frame_index": 0,
            }
        },
        "required_next_gate": {
            "kind": B.NOMINAL_HOLD_RECEIPT_KIND,
            "zero_terminal_required": [
                "joint_qdes_forbidden",
                "joint_actual_forbidden",
                "robot_hit_table",
                "base_fell_tilt",
                "base_too_low",
            ],
        },
    }
    dynamic_ready["content_sha256"] = B._canonical_sha256(
        dynamic_ready
    )
    dynamic_ready_sha = _write_json(dynamic_ready_path, dynamic_ready)
    nominal_hold_path = (
        root
        / "configs/a3_dynamic_ready"
        / f"{action_id}.nominal_hold.v1.json"
    )
    nominal_hold = {
        "schema_version": 1,
        "kind": B.NOMINAL_HOLD_RECEIPT_KIND,
        "verdict": "PASS",
        "action_id": action_id,
        "artifact": {
            "path": (
                f"/workspace/fixture/{action_id}.dynamic_ready.v1.json"
            ),
            "sha256": dynamic_ready_sha,
            "content_sha256": dynamic_ready["content_sha256"],
        },
        "motion_sha256": facts["motion_sha256"],
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
    nominal_hold["content_sha256"] = B._canonical_utf8_sha256(
        nominal_hold
    )
    nominal_hold_sha = _write_json(nominal_hold_path, nominal_hold)
    return {
        "root": root,
        "source_path": source_path,
        "source_sha": B._sha256_file(source_path),
        "profile_path": pins_path,
        "profile_sha": pins_sha,
        "solver_sha": solver_sha,
        "physics_sha": physics_sha,
        "objective_sha": objective_sha,
        "motion_path": root / motion_relative,
        "dynamic_ready_path": dynamic_ready_path,
        "dynamic_ready_sha": dynamic_ready_sha,
        "nominal_hold_path": nominal_hold_path,
        "nominal_hold_sha": nominal_hold_sha,
    }


def _materialize(
    fixture: dict[str, object],
    action_id: str,
    *,
    scope: str = B.SCOPE,
    strike_frame: int | None = None,
):
    return B.materialize_n1_contact_bundle(
        repo_root=fixture["root"],
        action_id=action_id,
        source_manifest=fixture["source_path"],
        expected_source_manifest_sha256=fixture["source_sha"],
        profile_pins=fixture["profile_path"],
        expected_profile_pins_sha256=fixture["profile_sha"],
        dynamic_ready_artifact=fixture["dynamic_ready_path"],
        expected_dynamic_ready_artifact_sha256=fixture[
            "dynamic_ready_sha"
        ],
        nominal_hold_receipt=fixture["nominal_hold_path"],
        expected_nominal_hold_receipt_sha256=fixture[
            "nominal_hold_sha"
        ],
        output_dir=Path(fixture["root"]) / "configs/n1_contact",
        require_git_tracked_motion=False,
        scope=scope,
        strike_frame=strike_frame,
    )


def _rebind_dynamic_ready_motion(
    fixture: dict[str, object], motion_sha256: str
) -> None:
    artifact_path = Path(fixture["dynamic_ready_path"])
    artifact = json.loads(artifact_path.read_text())
    artifact["sources"]["stable_motion"]["sha256"] = motion_sha256
    unsigned_artifact = dict(artifact)
    unsigned_artifact.pop("content_sha256")
    artifact["content_sha256"] = B._canonical_sha256(
        unsigned_artifact
    )
    fixture["dynamic_ready_sha"] = _write_json(
        artifact_path, artifact
    )
    receipt_path = Path(fixture["nominal_hold_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"]["sha256"] = fixture["dynamic_ready_sha"]
    receipt["artifact"]["content_sha256"] = artifact[
        "content_sha256"
    ]
    receipt["motion_sha256"] = motion_sha256
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("content_sha256")
    receipt["content_sha256"] = B._canonical_utf8_sha256(
        unsigned_receipt
    )
    fixture["nominal_hold_sha"] = _write_json(receipt_path, receipt)


def _install_full_motion_fixture(
    fixture: dict[str, object],
    action_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], int]:
    """Use one tracked schema-v2 motion as a scope-routing fixture."""

    source_action = next(
        row
        for row in json.loads(Path(fixture["source_path"]).read_text())[
            "actions"
        ]
        if row["action_id"] == action_id
    )
    upper_facts = B.SUPPORTED_ACTIONS[action_id]
    relative = Path("motions/test_fixtures") / (
        f"{action_id}_full_scope_fixture.npz"
    )
    target = Path(fixture["root"]) / relative
    _copy(REPO_ROOT / upper_facts["motion_path"], target)
    full_facts = {
        "motion_path": relative.as_posix(),
        "motion_sha256": B._sha256_file(target),
        "reference_t_hit_s": source_action["reference_t_hit_s"],
        "reference_t_cycle_s": source_action["reference_t_cycle_s"],
    }
    _rebind_dynamic_ready_motion(fixture, full_facts["motion_sha256"])
    monkeypatch.setitem(B.FULL_SUPPORTED_ACTIONS, action_id, full_facts)
    strike_frame = round(
        source_action["strike_phase"]
        * (
            source_action["reference_t_cycle_s"] * 50.0
        )
    )
    return full_facts, strike_frame


@pytest.mark.parametrize("action_id", ("bh_loop_c", "bh_block"))
def test_materializes_strict_contact_only_bundle(tmp_path: Path, action_id: str):
    fixture = _make_repo(tmp_path, action_id)
    result = _materialize(fixture, action_id)
    root = Path(fixture["root"])
    bundle = json.loads((root / result["bundle_path"]).read_text())
    manifest = json.loads((root / result["manifest_path"]).read_text())
    prototype = json.loads((root / result["prototype_path"]).read_text())
    receipt = json.loads(
        (root / result["contact_alignment_path"]).read_text()
    )
    source = json.loads(Path(fixture["source_path"]).read_text())
    source_action = next(
        row for row in source["actions"] if row["action_id"] == action_id
    )
    action = manifest["actions"][0]

    assert set(bundle) == {
        "schema_version",
        "artifact_type",
        "action_id",
        "action_uid",
        "scope",
        "source_manifest",
        "profile_pins",
        "motion",
        "prototype",
        "manifest",
        "contact_alignment",
        "dynamic_ready",
        "geometry",
        "claims",
    }
    assert bundle["schema_version"] == 2
    assert bundle["artifact_type"] == "n1_contact_training_bundle_v2"
    assert bundle["dynamic_ready"] == {
        "artifact": {
            "path": Path(fixture["dynamic_ready_path"])
            .relative_to(root)
            .as_posix(),
            "sha256": fixture["dynamic_ready_sha"],
        },
        "nominal_hold_receipt": {
            "path": Path(fixture["nominal_hold_path"])
            .relative_to(root)
            .as_posix(),
            "sha256": fixture["nominal_hold_sha"],
        },
    }
    assert bundle["geometry"] == receipt["geometry"]
    assert set(bundle["geometry"]) == {
        "path",
        "sha256",
        "payload_sha256",
        "kind",
    }
    assert bundle["geometry"]["path"].endswith(
        "/racket_contact_geometry.py"
    )
    assert bundle["claims"] == B.CLAIMS
    assert bundle["claims"]["landing_claim"] is False
    assert bundle["claims"]["post_bounce_claim"] is False
    assert bundle["claims"]["baseline_crossing_claim"] is False
    assert manifest["action_order"] == [action_id]
    assert manifest["mobility_mode"] == "no_move"
    assert manifest["holdout"]["samples_per_action"] >= 768
    assert (
        manifest["solver_profile_sha256"] == fixture["solver_sha"]
    )
    assert (
        manifest["physics_profile_sha256"] == fixture["physics_sha"]
    )
    assert (
        manifest["counter_rally_objective"]["mode"]
        == "counter_rally_v1"
    )
    assert (
        result["counter_rally_objective_profile_sha256"]
        == fixture["objective_sha"]
    )
    replacement = B.SUPPORTED_ACTIONS[action_id]
    assert action["action_uid"] == (
        B._load_module(
            "uid_contract_for_test",
            root
            / B.MDP_RELATIVE_DIR
            / "action_ball_manifest.py",
        ).derive_action_ball_action_uid(
            action_id,
            action["family"],
            replacement["motion_sha256"],
        )
    )
    assert action["action_uid"] != source_action["action_uid"]
    assert action["motion_path"] == replacement["motion_path"]
    assert action["motion_sha256"] == replacement["motion_sha256"]
    assert action["reference_t_hit_s"] == source_action["reference_t_hit_s"]
    assert (
        action["reference_t_cycle_s"]
        == source_action["reference_t_cycle_s"]
    )
    source_profile = source_action["ball_profile"]
    output_profile = action["ball_profile"]
    for key in source_profile:
        if key not in {
            "contact_offset_center_b_yaw_m",
            "contact_offset_min_b_yaw_m",
            "contact_offset_max_b_yaw_m",
        }:
            assert output_profile[key] == source_profile[key]
    source_center = source_profile["contact_offset_center_b_yaw_m"]
    output_center = output_profile["contact_offset_center_b_yaw_m"]
    for bound_key in (
        "contact_offset_min_b_yaw_m",
        "contact_offset_max_b_yaw_m",
    ):
        assert [
            bound - center
            for bound, center in zip(
                output_profile[bound_key], output_center
            )
        ] == pytest.approx(
            [
                bound - center
                for bound, center in zip(
                    source_profile[bound_key], source_center
                )
            ],
            abs=1.0e-12,
        )
    assert output_center == pytest.approx(
        receipt["alignment"]["teacher_selected_face_center_b_yaw_m"],
        abs=1.0e-12,
    )
    assert prototype["schema_version"] == 2
    assert tuple(prototype["scopes"]) == ("upper",)
    assert prototype["scopes"]["upper"][0]["motion_id"] == action_id
    assert prototype["scopes"]["upper"][0]["clip_index"] == 0
    assert receipt["status"] == "PASS"
    assert receipt["claims"] == B.CLAIMS
    assert receipt["timing"]["t_hit_abs_error_s"] <= 1.0e-12
    assert receipt["timing"]["t_cycle_abs_error_s"] <= 1.0e-12
    assert receipt["alignment"]["center_within_threshold"] is True
    assert (
        receipt["alignment"]["center_gate_distance_m"]
        <= B.CENTER_ALIGNMENT_THRESHOLD_M
    )
    assert receipt["alignment"]["center_gate_distance_m"] <= 1.0e-12
    assert (
        receipt["alignment"]["contact_center_authority"]
        == (
            "a3_stable_upper_selected_rubber_face_center_at_pinned_"
            "strike_frame"
        )
    )
    assert receipt["alignment"]["upper_contact_center_preserved"] is False
def test_scope_cli_defaults_to_upper():
    arguments = B._build_parser().parse_args(
        [
            "--action-id",
            "bh_loop_c",
            "--profile-pins",
            "pins.json",
            "--expected-profile-pins-sha256",
            "0" * 64,
            "--dynamic-ready-artifact",
            "ready.json",
            "--expected-dynamic-ready-artifact-sha256",
            "1" * 64,
            "--nominal-hold-receipt",
            "hold.json",
            "--expected-nominal-hold-receipt-sha256",
            "2" * 64,
            "--output-dir",
            "out",
        ]
    )
    assert arguments.scope == "upper"
    assert arguments.strike_frame is None


def test_full_scope_pins_exact_motion_facts():
    assert B.FULL_SUPPORTED_ACTIONS == {
        "bh_loop_c": {
            "motion_path": (
                "motions/fivebind_n5_20260728/"
                "bh_loop_c_full_full_fivebind.npz"
            ),
            "motion_sha256": (
                "010740965573863c6dbcb48f4efa3318eea51d1d005da0e458824c837a43c8b0"
            ),
            "reference_t_hit_s": 0.76,
            "reference_t_cycle_s": 1.6,
        },
        "bh_block": {
            "motion_path": (
                "motions/fivebind_n5_20260728/"
                "bh_block_full_full_fivebind.npz"
            ),
            "motion_sha256": (
                "12a6c5b7914dc2d023bbd0447fab41ccc80de7d1be0bb4a8018a98e453dceefa"
            ),
            "reference_t_hit_s": 0.52,
            "reference_t_cycle_s": 1.08,
        },
    }


@pytest.mark.parametrize("action_id", ("bh_loop_c", "bh_block"))
def test_full_scope_retargets_contact_box_and_preserves_incoming_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action_id: str,
):
    fixture = _make_repo(tmp_path, action_id)
    full_facts, strike_frame = _install_full_motion_fixture(
        fixture, action_id, monkeypatch
    )
    # A negative tolerance makes the upper stationary-root checks impossible;
    # full must bypass both because root motion belongs to full-body tracking.
    monkeypatch.setattr(B, "ROOT_STATIONARY_TOLERANCE_M", -1.0)
    monkeypatch.setattr(B, "ROOT_YAW_TOLERANCE_RAD", -1.0)
    result = _materialize(
        fixture,
        action_id,
        scope="full",
        strike_frame=strike_frame,
    )
    root = Path(fixture["root"])
    bundle = json.loads((root / result["bundle_path"]).read_text())
    manifest = json.loads((root / result["manifest_path"]).read_text())
    prototype = json.loads((root / result["prototype_path"]).read_text())
    receipt = json.loads(
        (root / result["contact_alignment_path"]).read_text()
    )
    source = json.loads(Path(fixture["source_path"]).read_text())
    source_action = next(
        row for row in source["actions"] if row["action_id"] == action_id
    )
    action = manifest["actions"][0]

    assert bundle["scope"] == "full"
    assert bundle["prototype"]["scope"] == "full"
    assert bundle["claims"]["diagnostic_only"] is True
    assert bundle["claims"]["training_authorized"] is False
    assert receipt["scope"] == "full"
    assert receipt["claims"] == bundle["claims"]
    assert tuple(prototype["scopes"]) == ("full",)
    assert prototype["scopes"]["full"][0]["scope"] == "full"
    assert action["motion_path"] == full_facts["motion_path"]
    assert action["motion_sha256"] == full_facts["motion_sha256"]
    assert action["reference_t_hit_s"] == full_facts[
        "reference_t_hit_s"
    ]
    assert action["reference_t_cycle_s"] == full_facts[
        "reference_t_cycle_s"
    ]
    manifest_module = B._load_module(
        f"test_manifest_{action_id}",
        root / B.MDP_RELATIVE_DIR / "action_ball_manifest.py",
    )
    assert action["action_uid"] == (
        manifest_module.derive_action_ball_action_uid(
            action_id,
            action["family"],
            action["motion_sha256"],
        )
    )
    for key in (
        "bundle_path",
        "manifest_path",
        "prototype_path",
        "contact_alignment_path",
    ):
        assert f"{action_id}.full." in Path(result[key]).name
    for path_key, sha_key in (
        ("bundle_path", "bundle_sha256"),
        ("manifest_path", "manifest_sha256"),
        ("prototype_path", "prototype_sha256"),
        ("contact_alignment_path", "contact_alignment_sha256"),
    ):
        assert result[sha_key][:12] in Path(result[path_key]).name

    source_profile = source_action["ball_profile"]
    output_profile = action["ball_profile"]
    contact_keys = {
        "contact_offset_center_b_yaw_m",
        "contact_offset_min_b_yaw_m",
        "contact_offset_max_b_yaw_m",
    }
    timing_keys = {
        "time_to_contact_center_s",
        "time_to_contact_std_lower_max_s",
        "time_to_contact_std_upper_max_s",
        "time_to_contact_min_s",
        "time_to_contact_max_s",
    }
    for key, value in source_profile.items():
        if key not in contact_keys | timing_keys:
            assert output_profile[key] == value
    source_center = source_profile[
        "contact_offset_center_b_yaw_m"
    ]
    output_center = output_profile[
        "contact_offset_center_b_yaw_m"
    ]
    for bound_key in (
        "contact_offset_min_b_yaw_m",
        "contact_offset_max_b_yaw_m",
    ):
        assert [
            bound - center
            for bound, center in zip(
                output_profile[bound_key], output_center
            )
        ] == pytest.approx(
            [
                bound - center
                for bound, center in zip(
                    source_profile[bound_key], source_center
                )
            ],
            abs=1.0e-12,
        )
    assert output_center == pytest.approx(
        receipt["alignment"][
            "teacher_ball_contact_center_b_yaw_m"
        ],
        abs=1.0e-12,
    )
    assert receipt["alignment"]["center_gate_distance_m"] <= 1.0e-12
    assert (
        receipt["alignment"]["contact_center_authority"]
        == (
            "full_motion_ball_center_at_selected_rubber_contact_at_"
            "explicit_strike_frame"
        )
    )
    assert (
        receipt["alignment"]["upper_contact_center_preserved"] is False
    )
    minimum = (
        action["reference_t_hit_s"] / action["teacher_rate_min"]
        + action["reaction_margin_s"]
    )
    maximum = (
        action["reference_t_hit_s"] / action["teacher_rate_max"]
        + 1.0
    )
    assert output_profile["time_to_contact_min_s"] == pytest.approx(
        minimum, abs=1.0e-12
    )
    assert output_profile["time_to_contact_max_s"] == pytest.approx(
        maximum, abs=1.0e-12
    )
    assert (
        manifest["counter_rally_objective"]["mode"]
        == "counter_rally_v1"
    )
    assert manifest["solver_profile_sha256"] == fixture["solver_sha"]
    assert manifest["physics_profile_sha256"] == fixture["physics_sha"]


def test_full_scope_requires_matching_explicit_strike_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _make_repo(tmp_path, "bh_loop_c")
    _, strike_frame = _install_full_motion_fixture(
        fixture, "bh_loop_c", monkeypatch
    )
    with pytest.raises(
        B.N1ContactBundleError,
        match="requires one explicit integer strike frame",
    ):
        _materialize(fixture, "bh_loop_c", scope="full")
    with pytest.raises(
        B.N1ContactBundleError,
        match="t_hit/t_cycle disagree",
    ):
        _materialize(
            fixture,
            "bh_loop_c",
            scope="full",
            strike_frame=strike_frame + 1,
        )


@pytest.mark.parametrize(
    (
        "action_id,strike_frame,t_hit_s,t_cycle_s,"
        "expected_floor_mps,expected_formal"
    ),
    (
        (
            "bh_loop_c",
            38,
            0.76,
            1.6,
            1.4711791276931763,
            "CANARY_THRESHOLD_PASS",
        ),
        (
            "bh_block",
            26,
            0.52,
            1.08,
            0.8788235783576965,
            "CANARY_THRESHOLD_FAIL",
        ),
    ),
)
def test_full_exact_asset_when_available(
    tmp_path: Path,
    action_id: str,
    strike_frame: int,
    t_hit_s: float,
    t_cycle_s: float,
    expected_floor_mps: float,
    expected_formal: str,
):
    facts = B.FULL_SUPPORTED_ACTIONS[action_id]
    source_motion = REPO_ROOT / facts["motion_path"]
    if not source_motion.is_file():
        pytest.skip(f"exact {action_id} full asset is not restored locally")
    if B._sha256_file(source_motion) != facts["motion_sha256"]:
        pytest.skip(
            f"local {action_id} full asset is not the pinned exact bytes"
        )
    fixture = _make_repo(tmp_path, action_id)
    _copy(
        source_motion,
        Path(fixture["root"]) / facts["motion_path"],
    )
    _rebind_dynamic_ready_motion(fixture, facts["motion_sha256"])
    result = _materialize(
        fixture,
        action_id,
        scope="full",
        strike_frame=strike_frame,
    )
    root = Path(fixture["root"])
    manifest = json.loads((root / result["manifest_path"]).read_text())
    prototype = json.loads((root / result["prototype_path"]).read_text())
    receipt = json.loads(
        (root / result["contact_alignment_path"]).read_text()
    )
    action = manifest["actions"][0]
    assert action["reference_t_hit_s"] == t_hit_s
    assert action["reference_t_cycle_s"] == t_cycle_s
    assert action["strike_phase"] == pytest.approx(
        strike_frame / (receipt["timing"]["frame_count"] - 1),
        abs=1.0e-12,
    )
    assert action["mount_normal_sign"] == -1
    assert receipt["timing"]["contact_frame"] == strike_frame
    assert receipt["alignment"]["center_gate_distance_m"] <= 1.0e-12
    row = prototype["scopes"]["full"][0]
    preflight = prototype["provenance"][
        "full_solver_admission_preflight"
    ]
    floor = preflight["speed_floor_proof"]
    assert row["racket_face_center_speed_min_mps"] == pytest.approx(
        expected_floor_mps, abs=1.0e-12
    )
    assert floor["selected_float32_floor_mps"] == pytest.approx(
        expected_floor_mps, abs=1.0e-12
    )
    assert floor["selected_float32_floor_mps"] >= floor[
        "analytical_floor_mps"
    ]
    assert floor["added_mapping_margin_mps"] > 0.0
    assert preflight["proposal_count"] == 512
    assert len(preflight["execution"]["proposal_corpus_sha256"]) == 64
    assert (
        sum(preflight["rejection_reasons"].values())
        + preflight["admitted_count"]
        == preflight["proposal_count"]
    )
    assert set(preflight["rejection_reasons"]) <= {"resid_gt_tol"}
    if action_id == "bh_loop_c":
        assert preflight["admitted_count"] == 512
        assert preflight["rejection_reasons"] == {}
    else:
        assert preflight["rejected_count"] > 0
        assert preflight["rejection_reasons"] == {
            "resid_gt_tol": preflight["rejected_count"]
        }
    assert preflight["numerical_reproducibility"] == {
        "proposal_identity": (
            "execution_local_canonical_sample_id_corpus_sha256"
        ),
        "same_execution_proposal_replay_verified": True,
        "cross_python_exact_proposal_corpus_claim": False,
        "solver_result_scope": (
            "exact_only_with_matching_torch_build_cpu_backend_and_"
            "implementation_sources"
        ),
        "cross_backend_exact_admitted_count_claim": False,
        "same_execution_repeatability_required": True,
        "decision_authority": (
            "diagnostic_and_formal_threshold_status_not_exact_count"
        ),
        "reason": (
            "python_normaldist_can_change_the_corpus_and_float32_"
            "batched_lm_can_change_branches_across_cpu_backends"
        ),
    }
    assert len(preflight["execution"]["torch_build_config_sha256"]) == 64
    assert preflight["diagnostic_gate"]["status"] == "PASS"
    assert (
        preflight["diagnostic_gate"][
            "zero_admission_canary_group_count"
        ]
        == 0
    )
    assert (
        preflight["formal_rate_threshold"]["threshold_status"]
        == expected_formal
    )
    assert (
        preflight["formal_rate_threshold"]["formal_evidence_status"]
        == "NOT_EVALUATED"
    )
    assert (
        preflight["formal_rate_threshold"]["claim"]
        == "fixed_seed_512_canary_only_not_formal_heldout_evidence"
    )
    assert preflight["episode_horizon"] == {
        "checked": True,
        "episode_length_s": 10.0,
        "attempt_close_margin_s": 0.02,
    }


def test_default_upper_matches_explicit_and_never_clobbers_existing_outputs(
    tmp_path: Path,
):
    first = _make_repo(tmp_path / "first", "bh_loop_c")
    second = _make_repo(tmp_path / "second", "bh_loop_c")
    first_result = _materialize(first, "bh_loop_c")
    second_result = _materialize(
        second,
        "bh_loop_c",
        scope="upper",
    )
    assert first_result == second_result
    first_root = Path(first["root"])
    prototype = json.loads(
        (first_root / first_result["prototype_path"]).read_text()
    )
    assert (
        prototype["provenance"]["producer_source_sha256"]
        == B._sha256_file(SCRIPT)
    )
    output_dir = first_root / "configs/n1_contact"
    before = {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="no-clobber"):
        _materialize(first, "bh_loop_c", scope="upper")
    after = {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda document: next(
                row
                for row in document["actions"]
                if row["action_id"] == "bh_loop_c"
            ).__setitem__("reference_t_hit_s", 0.64),
            "reference_t_hit_s changed",
        ),
    ),
)
def test_rejects_timing_or_contact_centre_drift(
    tmp_path: Path, mutation, match: str
):
    fixture = _make_repo(tmp_path, "bh_loop_c")
    source_path = Path(fixture["source_path"])
    document = json.loads(source_path.read_text())
    mutation(document)
    fixture["source_sha"] = _write_json(source_path, document)
    with pytest.raises(B.N1ContactBundleError, match=match):
        _materialize(fixture, "bh_loop_c")


def test_rejects_motion_byte_drift(tmp_path: Path):
    fixture = _make_repo(tmp_path, "bh_block")
    motion = Path(fixture["motion_path"])
    motion.write_bytes(motion.read_bytes() + b"tamper")
    with pytest.raises(B.N1ContactBundleError, match="motion SHA"):
        _materialize(fixture, "bh_block")


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
def test_rejects_dynamic_ready_nominal_hold_drift(
    tmp_path: Path, mutation, match: str
):
    fixture = _make_repo(tmp_path, "bh_loop_c")
    receipt_path = Path(fixture["nominal_hold_path"])
    receipt = json.loads(receipt_path.read_text())
    mutation(receipt)
    unsigned = dict(receipt)
    unsigned.pop("content_sha256")
    receipt["content_sha256"] = B._canonical_utf8_sha256(unsigned)
    fixture["nominal_hold_sha"] = _write_json(receipt_path, receipt)
    with pytest.raises(B.N1ContactBundleError, match=match):
        _materialize(fixture, "bh_loop_c")


def test_rejects_dynamic_ready_content_seal_or_motion_drift(
    tmp_path: Path,
):
    fixture = _make_repo(tmp_path / "seal", "bh_loop_c")
    artifact_path = Path(fixture["dynamic_ready_path"])
    artifact = json.loads(artifact_path.read_text())
    artifact["content_sha256"] = "f" * 64
    fixture["dynamic_ready_sha"] = _write_json(artifact_path, artifact)
    with pytest.raises(B.N1ContactBundleError, match="does not seal"):
        _materialize(fixture, "bh_loop_c")

    fixture = _make_repo(tmp_path / "motion", "bh_loop_c")
    artifact_path = Path(fixture["dynamic_ready_path"])
    artifact = json.loads(artifact_path.read_text())
    artifact["sources"]["stable_motion"]["sha256"] = "e" * 64
    unsigned = dict(artifact)
    unsigned.pop("content_sha256")
    artifact["content_sha256"] = B._canonical_sha256(unsigned)
    fixture["dynamic_ready_sha"] = _write_json(artifact_path, artifact)
    with pytest.raises(B.N1ContactBundleError, match="exact A3 N=1"):
        _materialize(fixture, "bh_loop_c")


def test_rejects_non_counter_rally_profile_pins(tmp_path: Path):
    fixture = _make_repo(tmp_path, "bh_loop_c")
    pins_path = Path(fixture["profile_path"])
    pins = json.loads(pins_path.read_text())
    del pins["solver_payload"]["counter_rally"]
    pins["solver_profile_sha256"] = B._canonical_sha256(
        pins["solver_payload"]
    )
    fixture["profile_sha"] = _write_json(pins_path, pins)
    with pytest.raises(
        B.N1ContactBundleError,
        match="not the exact canonical N=1 counter-rally",
    ):
        _materialize(fixture, "bh_loop_c")
