from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rebind_stage1_question_bank_physics_contract.py"
CONFIG = ROOT / "configs/phase1_signed_face_bank_rebind_prereg_20260713.json"


def load_module():
    spec = importlib.util.spec_from_file_location("stage1_bank_rebind", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R = load_module()


def checked_manifest(tmp_path: Path | None = None):
    value = R.load_manifest(CONFIG)
    if tmp_path is not None:
        value["source_repo"] = str(ROOT)
    return value


def test_checked_manifest_is_sim_only_no_legacy_and_no_clobber():
    value = checked_manifest()
    assert value["legacy_load_forbidden"] is True
    assert value["real_robot_commands_forbidden"] is True
    assert value["output"]["root_must_not_exist"] is True
    assert value["output"]["completion_report_written_last"] is True
    assert value["physics_contract"]["only_changed_file"].endswith("virtual_ball.py")


def test_real_git_pair_proves_one_additive_function(monkeypatch):
    value = checked_manifest()
    value["source_repo"] = str(ROOT)
    monkeypatch.setattr(R, "git_text", lambda repo, args: (
        value["target_commit"] if args == ["rev-parse", "HEAD"] else ""
    ))
    proof = R.prove_additive_source_change(value)
    assert proof["only_changed_file"].endswith("virtual_ball.py")
    assert proof["added_top_level_function"] == "signed_face_hemisphere"
    assert proof["preexisting_executable_ast_equal"] is True
    assert proof["target_physics_contract_sha256"] == (
        "09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95"
    )


def test_metadata_rebind_changes_only_four_preregistered_leaves(tmp_path):
    value = checked_manifest()
    value["source_bank"] = copy.deepcopy(value["source_bank"])
    value["source_bank"]["path"] = str(tmp_path / "old.npz")
    old_files = {
        path: R.sha256_bytes(R.git_show(ROOT, value["base_commit"], path))
        for path in R.EXPECTED_PHYSICS_FILES
    }
    old_contract = R.physics_contract_from_files(
        old_files, value["physics_contract"]["contract_name"]
    )
    family = {
        "contract": "stage1-source-family-v2",
        "physics_contract_sha256": R.canonical_sha256(old_contract),
    }
    clips = {}
    arrays = {}
    for clip in value["source_bank"]["clip_order"]:
        count = value["source_bank"]["question_counts"][clip]
        clips[clip] = {
            "motion_sha256": value["source_bank"]["motion_sha256"][clip],
            "question_count": count,
        }
        arrays[f"{clip}/contact_pos_env"] = np.zeros(3, dtype=np.float64)
        for suffix in ("incoming_vel", "incoming_spin", "demanded_vel", "demanded_normal"):
            arrays[f"{clip}/{suffix}"] = np.zeros((count, 3), dtype=np.float64)
        arrays[f"{clip}/difficulty_deg"] = np.zeros(count, dtype=np.float64)
    meta = {
        "schema_version": 3,
        "split": "train",
        "clip_order": value["source_bank"]["clip_order"],
        "clips": clips,
        "physics_contract": old_contract,
        "physics_contract_sha256": R.canonical_sha256(old_contract),
        "source_family_contract": family,
        "source_family_sha256": R.canonical_sha256(family),
    }
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(value["source_bank"]["path"], **arrays)
    value["source_bank"]["sha256"] = R.sha256_file(Path(value["source_bank"]["path"]))
    proof = {
        "base_file_sha256": old_files,
        "target_physics_contract": {
            "contract": value["physics_contract"]["contract_name"],
            "files": {
                **old_files,
                value["physics_contract"]["only_changed_file"]: value["physics_contract"][
                    "target_changed_file_sha256"
                ],
            },
        },
        "target_physics_contract_sha256": value["physics_contract"][
            "target_physics_contract_sha256"
        ],
    }
    synthetic_new_family = copy.deepcopy(family)
    synthetic_new_family["physics_contract_sha256"] = proof[
        "target_physics_contract_sha256"
    ]
    value["target_source_family_sha256"] = R.canonical_sha256(synthetic_new_family)
    for clip in value["source_bank"]["clip_order"]:
        motion = tmp_path / f"{clip}.npz"
        motion.write_bytes(clip.encode("utf-8"))
        value["source_bank"]["motion_runtime"][clip]["path"] = str(motion)
        value["source_bank"]["motion_sha256"][clip] = R.sha256_file(motion)
        meta["clips"][clip]["motion_sha256"] = value["source_bank"]["motion_sha256"][clip]
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(value["source_bank"]["path"], **arrays)
    value["source_bank"]["sha256"] = R.sha256_file(Path(value["source_bank"]["path"]))
    rebound, evidence = R.prepare_rebound_arrays(value, proof)
    assert evidence["allowed_metadata_leaf_changes"] == sorted(
        value["allowed_metadata_leaf_changes"]
    )
    for key in arrays:
        if key != "meta_json":
            assert R.array_fingerprint(arrays[key]) == R.array_fingerprint(rebound[key])


def test_exclusive_writers_never_overwrite(tmp_path):
    target = tmp_path / "record.json"
    R.write_json_exclusive(target, {"first": True})
    with pytest.raises(FileExistsError):
        R.write_json_exclusive(target, {"second": True})
    assert json.loads(target.read_text()) == {"first": True}

    bank = tmp_path / "bank.npz"
    R.write_npz_exclusive(bank, {"x": np.asarray([1.0])})
    with pytest.raises(FileExistsError):
        R.write_npz_exclusive(bank, {"x": np.asarray([2.0])})
    with np.load(bank, allow_pickle=False) as loaded:
        assert loaded["x"].tolist() == [1.0]


def test_manifest_rejects_weakened_legacy_or_extra_physics_file(tmp_path):
    value = json.loads(CONFIG.read_text())
    value["legacy_load_forbidden"] = False
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(R.RebindError, match="legacy"):
        R.load_manifest(path)

    value = json.loads(CONFIG.read_text())
    value["physics_contract"]["files"].append("other.py")
    path.write_text(json.dumps(value))
    with pytest.raises(R.RebindError, match="file set"):
        R.load_manifest(path)
