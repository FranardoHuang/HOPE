from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "subset_stage1_question_bank.py"
SPEC = importlib.util.spec_from_file_location("subset_stage1_question_bank", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


CLIPS = ["fh_loop", "bh_loop_c", "bh_block", "s0_highpress"]
SELECTED = ["bh_loop_c", "bh_block", "s0_highpress"]


def _source_meta():
    family = {
        "contract": "stage1-source-family-v2",
        "clip_order": list(CLIPS),
        "clips": {
            name: {"motion_sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in CLIPS
        },
    }
    return {
        "schema_version": 3,
        "split": "train",
        "clip_order": list(CLIPS),
        "clips": {
            name: {
                "motion_sha256": family["clips"][name]["motion_sha256"],
                "question_count": index + 2,
            }
            for index, name in enumerate(CLIPS)
        },
        "grip_applied_per_clip": {name: True for name in CLIPS},
        "rally_yaw_applied_per_clip": {name: True for name in CLIPS},
        "source_family_contract": family,
        "source_family_sha256": S._canonical_sha256(family),
    }


def _write_source(path: Path) -> str:
    arrays = {}
    for index, name in enumerate(CLIPS):
        arrays[f"{name}/incoming_vel"] = np.arange(
            (index + 2) * 3, dtype=np.float64
        ).reshape(index + 2, 3)
        arrays[f"{name}/demanded_vel"] = np.full(
            (index + 2, 3), index + 0.25, dtype=np.float32
        )
    arrays["meta_json"] = np.frombuffer(
        json.dumps(_source_meta(), sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accept_runtime(path: Path, clips, split):
    assert path.is_file()
    assert list(clips) == SELECTED
    assert split == "train"


def test_build_subset_meta_preserves_source_order_and_rehashes_family():
    source = _source_meta()
    projected = S.build_subset_meta(
        source, SELECTED, source_bank_sha256="a" * 64
    )
    assert source["clip_order"] == CLIPS
    assert projected["clip_order"] == SELECTED
    assert list(projected["clips"]) == SELECTED
    assert list(projected["source_family_contract"]["clips"]) == SELECTED
    assert projected["source_family_sha256"] == S._canonical_sha256(
        projected["source_family_contract"]
    )
    assert projected["clip_subset"]["dropped_clips"] == ["fh_loop"]
    assert projected["clip_subset"]["question_arrays_bitwise_preserved"] is True


@pytest.mark.parametrize(
    "selection",
    [[], ["bh_loop_c", "bh_loop_c"], ["s0_highpress", "bh_loop_c"], ["missing"]],
)
def test_build_subset_meta_rejects_empty_duplicate_reordered_and_unknown(selection):
    with pytest.raises(S.SubsetBankError):
        S.build_subset_meta(_source_meta(), selection, source_bank_sha256="b" * 64)


def test_subset_bank_is_deterministic_and_preserves_selected_arrays(tmp_path: Path):
    source = tmp_path / "source.npz"
    source_sha = _write_source(source)
    outputs = []
    for suffix in ("a", "b"):
        out = tmp_path / f"subset-{suffix}.npz"
        receipt = tmp_path / f"subset-{suffix}.json"
        report = S.subset_bank(
            source,
            SELECTED,
            out,
            receipt,
            expected_source_sha256=source_sha,
            runtime_validator=_accept_runtime,
        )
        outputs.append(out.read_bytes())
        assert report["content"]["output_bank"]["clip_order"] == SELECTED
        assert report["content"]["dropped_clips"] == ["fh_loop"]
        with np.load(out, allow_pickle=False) as bank:
            assert not any(key.startswith("fh_loop/") for key in bank.files)
            assert list(
                json.loads(bank["meta_json"].tobytes().decode())["clip_order"]
            ) == SELECTED
            with np.load(source, allow_pickle=False) as original:
                for key in bank.files:
                    if key != "meta_json":
                        assert bank[key].dtype == original[key].dtype
                        assert bank[key].shape == original[key].shape
                        assert bank[key].tobytes() == original[key].tobytes()
    assert outputs[0] == outputs[1]


def test_subset_bank_rejects_source_sha_and_no_clobber(tmp_path: Path):
    source = tmp_path / "source.npz"
    source_sha = _write_source(source)
    out = tmp_path / "subset.npz"
    receipt = tmp_path / "subset.json"
    with pytest.raises(S.SubsetBankError, match="SHA-256 mismatch"):
        S.subset_bank(
            source,
            SELECTED,
            out,
            receipt,
            expected_source_sha256="0" * 64,
            runtime_validator=_accept_runtime,
        )
    out.write_bytes(b"occupied")
    with pytest.raises(S.SubsetBankError, match="overwrite"):
        S.subset_bank(
            source,
            SELECTED,
            out,
            receipt,
            expected_source_sha256=source_sha,
            runtime_validator=_accept_runtime,
        )
