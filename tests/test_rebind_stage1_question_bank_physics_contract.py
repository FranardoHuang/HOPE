from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rebind_stage1_question_bank_physics_contract.py"
TRAIN_CONFIG = ROOT / "configs/phase1_signed_face_bank_rebind_prereg_20260713.json"
EXAM_CONFIG = ROOT / "configs/phase1_signed_face_exam_bank_rebind_prereg_20260713.json"
TRAIN_CONFIG_SHA256 = "5b22a6dd3c41ba1abd44e631e408ed73ada2ac66fc7ff86dc62d48f69ff2ad29"


def load_module():
    spec = importlib.util.spec_from_file_location("stage1_bank_rebind", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R = load_module()


def checked_manifest(config: Path = TRAIN_CONFIG):
    return R.load_manifest(config)


def test_checked_manifest_is_sim_only_no_legacy_and_no_clobber():
    value = checked_manifest()
    assert value["legacy_load_forbidden"] is True
    assert value["real_robot_commands_forbidden"] is True
    assert value["output"]["root_must_not_exist"] is True
    assert value["output"]["completion_report_written_last"] is True
    assert value["physics_contract"]["only_changed_file"].endswith("virtual_ball.py")


def test_train_v2_manifest_remains_byte_exact_and_profile_compatible():
    assert R.sha256_file(TRAIN_CONFIG) == TRAIN_CONFIG_SHA256
    value = checked_manifest(TRAIN_CONFIG)
    assert value["manifest_id"] == R.TRAIN_V2_MANIFEST_ID
    assert value["source_bank"]["split"] == "train"
    assert value["source_bank"]["question_counts"] == {
        "forehand": 757,
        "backhand": 724,
    }
    assert "bytes" not in value["source_bank"]


def test_exam_v1_manifest_binds_old_exam_and_independent_publication():
    value = checked_manifest(EXAM_CONFIG)
    train = checked_manifest(TRAIN_CONFIG)
    assert value["manifest_id"] == R.EXAM_V1_MANIFEST_ID
    assert value["source_bank"] == R.MANIFEST_PROFILES[R.EXAM_V1_MANIFEST_ID][
        "source_bank"
    ]
    assert value["source_bank"]["split"] == "exam"
    assert value["source_bank"]["bytes"] == 63_968
    assert value["source_bank"]["question_counts"] == {
        "forehand": 183,
        "backhand": 188,
    }
    assert value["target_source_family_sha256"] == (
        "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db"
    )
    assert value["target_source_family_sha256"] == train["target_source_family_sha256"]
    assert value["physics_contract"]["target_physics_contract_sha256"] == (
        train["physics_contract"]["target_physics_contract_sha256"]
    )
    assert value["output"] == R.MANIFEST_PROFILES[R.EXAM_V1_MANIFEST_ID]["output"]
    assert value["output"]["root"] != train["output"]["root"]


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


@pytest.mark.parametrize(
    "config",
    (TRAIN_CONFIG, EXAM_CONFIG),
    ids=("frozen-train-v2", "frozen-exam-v1"),
)
def test_metadata_rebind_changes_only_four_preregistered_leaves(tmp_path, config):
    value = checked_manifest(config)
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
        "split": value["source_bank"]["split"],
        "clip_order": value["source_bank"]["clip_order"],
        "clips": clips,
        "physics_contract": old_contract,
        "physics_contract_sha256": R.canonical_sha256(old_contract),
        "source_family_contract": family,
        "source_family_sha256": R.canonical_sha256(family),
    }
    if "source_family_sha256" in value["source_bank"]:
        value["source_bank"]["source_family_sha256"] = meta["source_family_sha256"]
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(value["source_bank"]["path"], **arrays)
    value["source_bank"]["sha256"] = R.sha256_file(Path(value["source_bank"]["path"]))
    if "bytes" in value["source_bank"]:
        value["source_bank"]["bytes"] = Path(value["source_bank"]["path"]).stat().st_size
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
    if "bytes" in value["source_bank"]:
        value["source_bank"]["bytes"] = Path(value["source_bank"]["path"]).stat().st_size
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


def test_source_bank_receipt_rejects_wrong_frozen_byte_count(tmp_path):
    bank = tmp_path / "bank.npz"
    np.savez(bank, x=np.asarray([1.0], dtype=np.float64))
    digest = R.sha256_file(bank)
    with pytest.raises(R.RebindError, match="byte count mismatch"):
        R.load_npz_stable(bank, digest, bank.stat().st_size + 1)
    arrays, receipt = R.load_npz_stable(bank, digest, bank.stat().st_size)
    assert arrays["x"].tolist() == [1.0]
    assert receipt["st_size"] == bank.stat().st_size


def test_manifest_rejects_weakened_legacy_or_extra_physics_file(tmp_path):
    value = json.loads(TRAIN_CONFIG.read_text())
    value["legacy_load_forbidden"] = False
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(R.RebindError, match="legacy"):
        R.load_manifest(path)

    value = json.loads(TRAIN_CONFIG.read_text())
    value["physics_contract"]["files"].append("other.py")
    path.write_text(json.dumps(value))
    with pytest.raises(R.RebindError, match="file set"):
        R.load_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda value: value["source_bank"].__setitem__("split", "train"), "source_bank"),
        (lambda value: value["source_bank"].__setitem__("bytes", 63_967), "source_bank"),
        (
            lambda value: value["source_bank"].__setitem__("path", "/tmp/exam.npz"),
            "source_bank",
        ),
        (
            lambda value: value["source_bank"]["question_counts"].__setitem__(
                "forehand", 184
            ),
            "source_bank",
        ),
        (
            lambda value: value["source_bank"].__setitem__("sha256", "0" * 64),
            "source_bank",
        ),
        (
            lambda value: value["source_bank"].__setitem__(
                "source_family_sha256", "0" * 64
            ),
            "source_bank",
        ),
        (
            lambda value: value["output"].__setitem__(
                "root",
                R.MANIFEST_PROFILES[R.TRAIN_V2_MANIFEST_ID]["output"]["root"],
            ),
            "output",
        ),
        (
            lambda value: value.__setitem__("target_source_family_sha256", "0" * 64),
            "target_source_family_sha256",
        ),
        (
            lambda value: value["physics_contract"].__setitem__(
                "target_physics_contract_sha256", "0" * 64
            ),
            "physics_contract",
        ),
    ),
)
def test_exam_manifest_profile_rejects_contract_mutations(tmp_path, mutation, match):
    value = json.loads(EXAM_CONFIG.read_text())
    mutation(value)
    path = tmp_path / "mutated-exam.json"
    path.write_text(json.dumps(value))
    with pytest.raises(R.RebindError, match=match):
        R.load_manifest(path)
