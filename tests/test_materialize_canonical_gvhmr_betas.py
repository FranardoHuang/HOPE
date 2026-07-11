from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_canonical_gvhmr_betas.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_canonical_gvhmr_betas", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATERIALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATERIALIZE)


@pytest.mark.parametrize("shape", [(10,), (1, 10), (5, 10)])
def test_shape_preserving_replacement_repeats_canonical_vector(shape):
    source_betas = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    payload = {
        "smpl_params_global": {
            "betas": source_betas,
            "body_pose": np.arange(5 * 63, dtype=np.float32).reshape(5, 63),
        },
        "camera": {"static": True, "name": "front"},
    }
    original_non_beta_sha, original_leaves = MATERIALIZE.semantic_digest(payload, None)
    original_betas = source_betas.copy()
    canonical = np.linspace(-0.25, 0.25, 10, dtype=np.float32)

    output = MATERIALIZE.replace_betas(payload, canonical, None)
    matrix, metadata = MATERIALIZE.beta_array_and_metadata(output, 5, None)

    assert tuple(output["smpl_params_global"]["betas"].shape) == shape
    assert output["smpl_params_global"]["betas"].dtype == np.float32
    assert np.array_equal(matrix, np.broadcast_to(canonical, matrix.shape))
    assert np.array_equal(payload["smpl_params_global"]["betas"], original_betas)
    assert MATERIALIZE.semantic_digest(output, None) == (
        original_non_beta_sha,
        original_leaves,
    )
    MATERIALIZE.verify_materialized_betas(output, canonical, 5, metadata, None)


def test_equal_video_weighting_is_not_frame_pooled_weighting():
    long_clip = np.zeros((101, 10), dtype=np.float32)
    short_clip = np.full((3, 10), 10.0, dtype=np.float32)

    canonical, per_video, deviations = MATERIALIZE.compute_canonical_vector(
        [long_clip, short_clip]
    )

    assert np.array_equal(per_video[0], np.zeros(10))
    assert np.array_equal(per_video[1], np.full(10, 10.0))
    assert np.array_equal(canonical, np.full(10, 5.0))
    assert deviations == [0.0, 0.0]
    assert np.median(np.concatenate([long_clip, short_clip]), axis=0)[0] == 0.0


def test_non_beta_digest_is_bit_sensitive_but_excludes_only_target():
    payload = {
        "smpl_params_global": {
            "betas": np.zeros((2, 10), dtype=np.float32),
            "transl": np.array([[0.0, -0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        }
    }
    digest, leaves = MATERIALIZE.semantic_digest(payload, None)
    beta_changed = {
        "smpl_params_global": {
            "betas": np.ones((2, 10), dtype=np.float32),
            "transl": payload["smpl_params_global"]["transl"].copy(),
        }
    }
    assert MATERIALIZE.semantic_digest(beta_changed, None) == (digest, leaves)
    beta_changed["smpl_params_global"]["transl"][0, 1] = 0.0
    assert MATERIALIZE.semantic_digest(beta_changed, None)[0] != digest


def test_beta_validation_rejects_bad_shape_nonfinite_and_integer():
    payload = {"smpl_params_global": {"betas": np.zeros((4, 10), dtype=np.float32)}}
    with pytest.raises(MATERIALIZE.MaterializationError, match="expected"):
        MATERIALIZE.beta_array_and_metadata(payload, 5, None)
    payload["smpl_params_global"]["betas"] = np.zeros((5, 10), dtype=np.int32)
    with pytest.raises(MATERIALIZE.MaterializationError, match="floating"):
        MATERIALIZE.beta_array_and_metadata(payload, 5, None)
    payload["smpl_params_global"]["betas"] = np.zeros((5, 10), dtype=np.float32)
    payload["smpl_params_global"]["betas"][0, 0] = np.nan
    with pytest.raises(MATERIALIZE.MaterializationError, match="non-finite"):
        MATERIALIZE.beta_array_and_metadata(payload, 5, None)


def test_publish_is_no_clobber_and_completion_manifest_is_last(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "clip.pt").write_bytes(b"clip")
    (staging / "materialization_manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"

    MATERIALIZE._publish_staged(staging, output, "materialization_manifest.json")
    assert (output / "clip.pt").read_bytes() == b"clip"
    assert (output / "materialization_manifest.json").is_file()

    second_stage = tmp_path / "second-stage"
    second_stage.mkdir()
    (second_stage / "materialization_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MATERIALIZE.MaterializationError, match="already exists"):
        MATERIALIZE._publish_staged(
            second_stage, output, "materialization_manifest.json"
        )
    assert json.loads((output / "materialization_manifest.json").read_text()) == {}


def test_save_reload_verifier_exercises_non_beta_and_beta_contract(tmp_path):
    class PickleTorchFixture:
        @staticmethod
        def is_tensor(value):
            return False

        @staticmethod
        def save(payload, handle):
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        @staticmethod
        def load(path, **_kwargs):
            with Path(path).open("rb") as handle:
                return pickle.load(handle)

    torch_fixture = PickleTorchFixture()
    payload = {
        "smpl_params_global": {
            "betas": np.zeros((5, 10), dtype=np.float32),
            "body_pose": np.arange(315, dtype=np.float32).reshape(5, 63),
        },
        "tag": "fixture",
    }
    metadata = MATERIALIZE.beta_array_and_metadata(payload, 5, torch_fixture)[1]
    non_beta_sha, leaf_count = MATERIALIZE.semantic_digest(payload, torch_fixture)
    canonical = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    output = MATERIALIZE.replace_betas(payload, canonical, torch_fixture)
    saved = tmp_path / "canonical.pt"

    output_sha, output_leaves, file_sha, file_bytes = (
        MATERIALIZE.save_and_reload_verified(
            output,
            non_beta_sha,
            canonical,
            5,
            metadata,
            saved,
            torch_fixture,
        )
    )

    assert output_sha == non_beta_sha
    assert output_leaves == leaf_count
    assert file_sha == MATERIALIZE.sha256_file(saved)
    assert file_bytes == saved.stat().st_size > 0


def test_torch_round_trip_when_torch_is_available(tmp_path):
    torch = pytest.importorskip("torch")
    payload = {
        "smpl_params_global": {
            "betas": torch.arange(50, dtype=torch.float32).reshape(5, 10),
            "body_pose": torch.arange(315, dtype=torch.float32).reshape(5, 63),
            "global_orient": torch.zeros((5, 3), dtype=torch.float32),
            "transl": torch.zeros((5, 3), dtype=torch.float32),
        },
        "tag": "fixture",
    }
    metadata = MATERIALIZE.beta_array_and_metadata(payload, 5, torch)[1]
    non_beta_sha, leaf_count = MATERIALIZE.semantic_digest(payload, torch)
    canonical = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    output = MATERIALIZE.replace_betas(payload, canonical, torch)
    saved = tmp_path / "canonical.pt"

    output_sha, output_leaves, file_sha, file_bytes = (
        MATERIALIZE.save_and_reload_verified(
            output, non_beta_sha, canonical, 5, metadata, saved, torch
        )
    )

    assert output_sha == non_beta_sha
    assert output_leaves == leaf_count
    assert file_sha == MATERIALIZE.sha256_file(saved)
    assert file_bytes == saved.stat().st_size > 0
