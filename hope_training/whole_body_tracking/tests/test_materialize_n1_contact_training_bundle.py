"""Focused CPU proofs for the contact-only N=1 bundle producer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
SCRIPT = WBT_ROOT / "scripts" / "materialize_n1_contact_training_bundle.py"
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
        "virtual_ball.py",
        "counter_rally.py",
        "counter_rally_torch.py",
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

    geometry = B._load_module(
        f"fixture_geometry_{action_id}",
        mdp_target / "racket_contact_geometry.py",
    )
    counter = B._load_module(
        f"fixture_counter_{action_id}",
        mdp_target / "counter_rally.py",
    )
    objective_sha = counter.CounterRallyObjectiveProfile().sha256
    source_hashes = {
        name: B._sha256_file(mdp_target / name)
        for name in source_names
    }
    venue_sha = B._sha256_file(root / venue_relative)
    physics_payload = {
        "schema_version": 1,
        "kind": "fixture.action_ball.physics",
        "venue_source": {
            "path": venue_relative.as_posix(),
            "file_sha256": venue_sha,
        },
    }
    physics_sha = B._canonical_sha256(physics_payload)
    geometry_contract = {
        "payload": geometry.GEOMETRY_SOURCE_PAYLOAD,
        "sha256": geometry.GEOMETRY_SOURCE_SHA256,
    }
    solver_payload = {
        "schema_version": 1,
        "kind": "fixture.action_ball.counter_rally_solver",
        "implementation_source_sha256": source_hashes,
        "physics_profile_sha256": physics_sha,
        "contact_geometry": geometry_contract,
        "counter_rally": {
            "mode": "exact_n1_fixed_action_reverse_ray",
            "objective_profile_sha256": objective_sha,
            "venue_physics_sha256": B._canonical_sha256(
                {"kind": "fixture.counter_rally.venue_physics"}
            ),
            "precheck_before_ordinary_solver": True,
            "selector_or_action_switching": False,
        },
    }
    solver_sha = B._canonical_sha256(solver_payload)
    pins = {
        "schema_version": 1,
        "kind": "fixture.action_ball.profile_pins",
        "solver_implementation_source_sha256": source_hashes,
        "contact_geometry": geometry_contract,
        "physics_payload": physics_payload,
        "physics_profile_sha256": physics_sha,
        "solver_payload": solver_payload,
        "solver_profile_sha256": solver_sha,
    }
    pins_path = root / "configs/profile_pins_counter_n1.json"
    pins_sha = _write_json(pins_path, pins)
    source_path = root / B.SOURCE_MANIFEST_RELATIVE_PATH
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
    }


def _materialize(fixture: dict[str, object], action_id: str):
    return B.materialize_n1_contact_bundle(
        repo_root=fixture["root"],
        action_id=action_id,
        source_manifest=fixture["source_path"],
        expected_source_manifest_sha256=fixture["source_sha"],
        profile_pins=fixture["profile_path"],
        expected_profile_pins_sha256=fixture["profile_sha"],
        output_dir=Path(fixture["root"]) / "configs/n1_contact",
        require_git_tracked_motion=False,
    )


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
        "geometry",
        "claims",
    }
    assert bundle["artifact_type"] == "n1_contact_training_bundle_v1"
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
    assert action["action_uid"] == source_action["action_uid"]
    assert action["motion_path"] == source_action["motion_path"]
    assert action["motion_sha256"] == source_action["motion_sha256"]
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
    root_z = receipt["alignment"]["ready_root_z_w_m"]
    for key in (
        "contact_offset_center_b_yaw_m",
        "contact_offset_min_b_yaw_m",
        "contact_offset_max_b_yaw_m",
    ):
        assert output_profile[key][:2] == source_profile[key][:2]
        assert output_profile[key][2] == pytest.approx(
            source_profile[key][2] - root_z,
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
    assert (
        receipt["alignment"]["legacy_absolute_contact_z_w_m"]
        == pytest.approx(
            source_profile["contact_offset_center_b_yaw_m"][2],
            abs=1.0e-12,
        )
    )


def test_outputs_are_deterministic_and_never_clobbered(tmp_path: Path):
    first = _make_repo(tmp_path / "first", "bh_loop_c")
    second = _make_repo(tmp_path / "second", "bh_loop_c")
    first_result = _materialize(first, "bh_loop_c")
    second_result = _materialize(second, "bh_loop_c")
    for key in (
        "bundle_sha256",
        "manifest_sha256",
        "prototype_sha256",
        "contact_alignment_sha256",
    ):
        assert first_result[key] == second_result[key]
    with pytest.raises(FileExistsError, match="no-clobber"):
        _materialize(first, "bh_loop_c")


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
        (
            lambda document: [
                vector.__setitem__(0, vector[0] + 0.20)
                for vector in (
                    next(
                        row
                        for row in document["actions"]
                        if row["action_id"] == "bh_loop_c"
                    )["ball_profile"][key]
                    for key in (
                        "contact_offset_center_b_yaw_m",
                        "contact_offset_min_b_yaw_m",
                        "contact_offset_max_b_yaw_m",
                    )
                )
            ],
            "above 0.03 m",
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
