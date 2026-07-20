from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_motion_role_catalog.py"
SPEC = importlib.util.spec_from_file_location("validate_motion_role_catalog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CATALOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CATALOG)

CATALOG_PATH = REPO_ROOT / "configs" / "motion_role_catalog.json"
SOURCE_MANIFESTS = (
    "configs/phase1_fresh_v3_asset_manifest_20260711.json",
    "configs/motion_video_intake_20260711.json",
    "configs/motion_video_intake_20260713.json",
)
INTAKE_11 = SOURCE_MANIFESTS[1]
INTAKE_13 = SOURCE_MANIFESTS[2]


@pytest.fixture()
def repo_copy(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    for relpath in SOURCE_MANIFESTS:
        shutil.copyfile(REPO_ROOT / relpath, tmp_path / relpath)
    return tmp_path


@pytest.fixture()
def catalog_dict() -> dict:
    return copy.deepcopy(CATALOG.load_strict_json(CATALOG_PATH))


def _entry(catalog: dict, asset_id: str) -> dict:
    for entry in catalog["entries"]:
        if entry["asset_id"] == asset_id:
            return entry
    raise AssertionError(f"missing catalog entry {asset_id!r}")


def _git_head_bytes(relpath: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{relpath}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_real_catalog_passes(catalog_dict):
    counts = CATALOG.validate_catalog(catalog_dict, REPO_ROOT)
    assert counts == {
        "entries": 19,
        "stationary_strike": 15,
        "shared_lateral_footwork_module": 4,
    }


def test_cli_main_passes_on_repo(capsys):
    assert CATALOG.main([]) == 0
    assert "PASS: 19 entries" in capsys.readouterr().out


def test_exact_cover_catalog_to_intake_direction(repo_copy, catalog_dict):
    _entry(catalog_dict, "franco_forehand_loop")["asset_id"] = "franco_ghost_loop"
    with pytest.raises(CATALOG.CatalogError, match="not present in"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_exact_cover_intake_to_catalog_direction(repo_copy, catalog_dict):
    intake_path = repo_copy / INTAKE_11
    intake = json.loads(intake_path.read_text())
    extra = copy.deepcopy(intake["assets"][0])
    extra["id"] = "franco_extra_clip"
    extra["sha256"] = "e" * 64
    intake["assets"].append(extra)
    intake_path.write_text(json.dumps(intake))
    catalog_dict["source_manifests"][INTAKE_11]["sha256"] = hashlib.sha256(
        intake_path.read_bytes()
    ).hexdigest()
    with pytest.raises(CATALOG.CatalogError, match="exactly 20 entries"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_missing_catalog_entry_fails_count(repo_copy, catalog_dict):
    catalog_dict["entries"] = [
        entry
        for entry in catalog_dict["entries"]
        if entry["asset_id"] != "v12_forehand_block"
    ]
    with pytest.raises(CATALOG.CatalogError, match="exactly 19 entries"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_tampered_video_sha_detected(repo_copy, catalog_dict):
    _entry(catalog_dict, "v6_forehand_block")["sha256"] = "f" * 64
    with pytest.raises(CATALOG.CatalogError, match="!= intake value"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_tampered_manifest_bytes_detected(repo_copy, catalog_dict):
    intake_path = repo_copy / INTAKE_13
    intake_path.write_bytes(intake_path.read_bytes() + b"\n")
    with pytest.raises(CATALOG.CatalogError, match="does not match recorded"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_recorded_manifest_sha_format_enforced(repo_copy, catalog_dict):
    catalog_dict["source_manifests"][INTAKE_13]["sha256"] = "not-a-sha"
    with pytest.raises(CATALOG.CatalogError, match="64 lowercase hex"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_v4rg_sha_must_match_asset_manifest(repo_copy, catalog_dict):
    _entry(catalog_dict, "hope_forehand_v4rg_cal")["sha256"] = "a" * 64
    with pytest.raises(CATALOG.CatalogError, match="v4rg manifest"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_v4rg_asset_id_must_match_basename_stem(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "hope_backhand_v4rg_cal")
    forehand = _entry(catalog_dict, "hope_forehand_v4rg_cal")
    entry["manifest_motion_key"] = "forehand"
    entry["source_basename"] = forehand["source_basename"]
    entry["asset_id"] = "hope_forehand_v4rg_cal_bis"
    with pytest.raises(CATALOG.CatalogError, match="asset_id must equal"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_motion_clip_must_be_shared_footwork(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "lateral_step_left_1")
    entry["motion_role"] = "stationary_strike"
    with pytest.raises(CATALOG.CatalogError, match="every motion/ clip"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_stroke_video_must_be_stationary(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "franco_forehand_block")
    entry["motion_role"] = "shared_lateral_footwork_module"
    with pytest.raises(CATALOG.CatalogError, match="every non-motion/ strike video"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_standalone_stop_teacher_role_rejected(repo_copy, catalog_dict):
    _entry(catalog_dict, "lateral_step_right_1")["motion_role"] = "stop_teacher"
    with pytest.raises(CATALOG.CatalogError, match="unsupported motion_role"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_slot_bound_footwork_rejected(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "lateral_step_left_2")
    entry["shared_across_action_slots"] = False
    with pytest.raises(CATALOG.CatalogError, match="standalone locomotion"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_footwork_requires_strike_intent_scope(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "lateral_step_right_2")
    entry["activation_scope"] = "own_action_slot"
    with pytest.raises(CATALOG.CatalogError, match="strike_intent_triggered_only"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_video_training_authorization_rejected(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "v12_backhand_block")
    entry["training_authorized"] = True
    with pytest.raises(CATALOG.CatalogError, match="grandfathered formal runtime pair"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)
    # Even a claimed vendor dynamic pass does not authorize a raw video.
    entry["vendor_mujoco_dynamic_pass"] = True
    with pytest.raises(CATALOG.CatalogError, match="never training_authorized"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_grandfather_flag_on_video_rejected(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "static_backhand_high_press")
    entry["grandfathered_formal_runtime_pair"] = True
    with pytest.raises(CATALOG.CatalogError, match="only the v4rg runtime pair"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_v4rg_without_grandfather_flag_rejected(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "hope_backhand_v4rg_cal")
    entry["grandfathered_formal_runtime_pair"] = False
    with pytest.raises(CATALOG.CatalogError, match="grandfathered formal runtime pair"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_grandfather_set_must_equal_declared_pair(repo_copy, catalog_dict):
    catalog_dict["formal_runtime_pair"]["asset_ids"] = [
        "hope_forehand_v4rg_cal",
        "franco_forehand_block",
    ]
    with pytest.raises(CATALOG.CatalogError, match="formal_runtime_pair.asset_ids"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_dang_direction_must_match_filename(repo_copy, catalog_dict):
    _entry(catalog_dict, "lateral_step_left_1")["movement_direction"] = "right"
    with pytest.raises(CATALOG.CatalogError, match="contradicts source filename"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_footwork_direction_null_rejected(repo_copy, catalog_dict):
    _entry(catalog_dict, "lateral_step_right_1")["movement_direction"] = None
    with pytest.raises(CATALOG.CatalogError, match="movement_direction in"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_stationary_direction_must_be_null(repo_copy, catalog_dict):
    _entry(catalog_dict, "franco_backhand_block")["movement_direction"] = "left"
    with pytest.raises(CATALOG.CatalogError, match="movement_direction=null"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_footwork_input_gate_status_preserved(repo_copy, catalog_dict):
    entry = _entry(catalog_dict, "lateral_step_left_1")
    entry["input_gate_status"] = "accepted"
    with pytest.raises(CATALOG.CatalogError, match="overwrite the safety fact"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)
    del entry["input_gate_status"]
    with pytest.raises(CATALOG.CatalogError, match="missing required keys"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_duplicate_catalog_asset_id_rejected(repo_copy, catalog_dict):
    catalog_dict["entries"][6]["asset_id"] = catalog_dict["entries"][5]["asset_id"]
    with pytest.raises(CATALOG.CatalogError, match="duplicate catalog asset_id"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_duplicate_json_keys_rejected(tmp_path):
    bad = tmp_path / "duplicate.json"
    bad.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(CATALOG.CatalogError, match="duplicate JSON key"):
        CATALOG.load_strict_json(bad)


def test_nonfinite_json_rejected(tmp_path):
    bad = tmp_path / "nonfinite.json"
    bad.write_text('{"schema_version": NaN}')
    with pytest.raises(CATALOG.CatalogError, match="non-finite|cannot read"):
        CATALOG.load_strict_json(bad)


def test_supersedes_must_declare_intake13(repo_copy, catalog_dict):
    catalog_dict["semantic_authority"]["supersedes"] = [
        {"manifest": INTAKE_11, "fields": ["role"], "note": "wrong target"}
    ]
    with pytest.raises(CATALOG.CatalogError, match="supersedes must declare"):
        CATALOG.validate_catalog(catalog_dict, repo_copy)


def test_intake_20260711_unmodified_versus_git_head():
    head = _git_head_bytes(INTAKE_11)
    assert (REPO_ROOT / INTAKE_11).read_bytes() == head
    recorded = CATALOG.load_strict_json(CATALOG_PATH)["source_manifests"][INTAKE_11]
    assert recorded["sha256"] == hashlib.sha256(head).hexdigest()


def test_intake_20260713_unmodified_versus_git_head():
    head = _git_head_bytes(INTAKE_13)
    assert (REPO_ROOT / INTAKE_13).read_bytes() == head
    recorded = CATALOG.load_strict_json(CATALOG_PATH)["source_manifests"][INTAKE_13]
    assert recorded["sha256"] == hashlib.sha256(head).hexdigest()


def test_cli_fails_closed_on_tampered_catalog(tmp_path, repo_copy, capsys):
    catalog = copy.deepcopy(CATALOG.load_strict_json(CATALOG_PATH))
    _entry(catalog, "v7_forehand_block")["sha256"] = "9" * 64
    bad_path = tmp_path / "bad_catalog.json"
    bad_path.write_text(json.dumps(catalog, ensure_ascii=False))
    assert (
        CATALOG.main(["--catalog", str(bad_path), "--repo-root", str(repo_copy)]) == 1
    )
    assert "[FAIL]" in capsys.readouterr().err
