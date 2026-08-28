from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest


LANE = Path(__file__).resolve().parents[1]


def _load():
    name = "mujoco_full_mdp_plant_contract_direct_test"
    path = LANE / "mujoco_full_mdp_plant_contract.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_base_mjb_and_augmented_runtime_are_separate_path_free_layers():
    module = _load()
    expected = module.expected_plant_model_identity()
    assert set(expected) == {"source_plant", "runtime_attach"}
    assert expected["source_plant"]["model_scope"] == (
        "pre_registered_vendor_base"
    )
    assert expected["source_plant"]["source_member_count"] == 93
    assert expected["source_plant"]["compiled_mjb_size_bytes"] == 113759317
    assert expected["runtime_attach"]["model_scope"] == (
        "mjlab_augmented_court_ball_runtime"
    )
    assert expected["runtime_attach"]["contract_type"].endswith("_v2")
    assert expected["runtime_attach"]["policy_clock"] == {
        "decimation": 20, "step_dt": 0.02,
    }
    assert expected["runtime_attach"]["warp_capacity"] == {
        "njmax_per_world": 572,
        "nconmax_per_world": 128,
    }
    assert set(expected["runtime_attach"]["final_augmented_mjb"]) == {
        "relative_locator", "sha256", "size_bytes",
    }
    assert expected["runtime_attach"]["final_augmented_mjb"][
        "relative_locator"
    ] == "runtime.mjb"
    assert "verification_receipt_sha256" not in expected["source_plant"]
    assert "owner_local_frame_sha256" not in expected["runtime_attach"]
    assert all("path" not in key for layer in expected.values() for key in layer)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["source_plant"].__setitem__(
            "portable_identity_sha256", "0" * 64
        ),
        lambda value: value["source_plant"].__setitem__(
            "verification_receipt_sha256", "short"
        ),
        lambda value: value["runtime_attach"].__setitem__(
            "owner_local_frame_sha256", "short"
        ),
        lambda value: value["runtime_attach"]["policy_clock"].__setitem__(
            "decimation", False
        ),
        lambda value: value["runtime_attach"]["final_augmented_mjb"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value["runtime_attach"]["warp_capacity"].__setitem__(
            "njmax_per_world", 571
        ),
    ),
)
def test_verified_identity_clone_is_strict_and_isolated(mutation):
    module = _load()
    original = module.verified_plant_model_identity(
        verification_receipt_sha256="c" * 64,
        owner_local_frame_sha256="d" * 64,
        final_augmented_mjb=module.expected_plant_model_identity()[
            "runtime_attach"
        ]["final_augmented_mjb"],
    )
    clone = module.clone_plant_model_identity(original)
    mutation(original)
    assert module.plant_model_identity_is_exact(original) is False
    assert module.plant_model_identity_is_exact(clone) is True


def test_verified_identity_requires_keyword_only_nonexchangeable_receipts():
    module = _load()
    with pytest.raises(TypeError):
        module.verified_plant_model_identity("c" * 64, "d" * 64)
    with pytest.raises(module.PlantContractError, match="owner-local"):
        module.verified_plant_model_identity(
            verification_receipt_sha256="c" * 64,
            owner_local_frame_sha256="short",
            final_augmented_mjb=module.expected_plant_model_identity()[
                "runtime_attach"
            ]["final_augmented_mjb"],
        )
    bad_mjb = dict(
        module.expected_plant_model_identity()["runtime_attach"][
            "final_augmented_mjb"
        ]
    )
    bad_mjb["sha256"] = "0" * 64
    with pytest.raises(module.PlantContractError, match="runtime MJB receipt"):
        module.verified_plant_model_identity(
            verification_receipt_sha256="c" * 64,
            owner_local_frame_sha256="d" * 64,
            final_augmented_mjb=bad_mjb,
        )


def test_augmented_mjb_serializer_hashes_the_actual_saved_bytes(tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    payload = b"one deterministic augmented model"

    class FakeMujoco:
        @staticmethod
        def mj_saveModel(_model, filename):
            Path(filename).write_bytes(payload)

    receipt = module.serialize_augmented_mjb_identity(FakeMujoco, object())
    assert receipt == {
        "relative_locator": "runtime.mjb",
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    assert list(tmp_path.iterdir()) == []


def _bind_expected_runtime_mjb(module, monkeypatch, receipt):
    expected = module.expected_plant_model_identity()
    expected["runtime_attach"]["final_augmented_mjb"] = dict(receipt)
    monkeypatch.setattr(
        module,
        "expected_plant_model_identity",
        lambda: copy.deepcopy(expected),
    )


def test_augmented_mjb_persistence_is_run_relative_and_no_clobber(
    tmp_path, monkeypatch,
):
    module = _load()
    payload = b"the live model bytes"
    expected_receipt = {
        "relative_locator": "runtime.mjb",
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    _bind_expected_runtime_mjb(module, monkeypatch, expected_receipt)

    class FakeMujoco:
        @staticmethod
        def mj_saveModel(_model, filename, *_args):
            Path(filename).write_bytes(payload)

    receipt = module.persist_augmented_runtime_mjb(
        FakeMujoco, object(), tmp_path,
    )
    assert receipt == expected_receipt
    target = tmp_path / "runtime.mjb"
    assert target.read_bytes() == payload
    assert target.stat().st_nlink == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["runtime.mjb"]

    with pytest.raises(module.PlantContractError, match="already exists"):
        module.persist_augmented_runtime_mjb(FakeMujoco, object(), tmp_path)
    assert target.read_bytes() == payload
    assert sorted(path.name for path in tmp_path.iterdir()) == ["runtime.mjb"]


def test_augmented_mjb_drift_is_rejected_before_publication(
    tmp_path, monkeypatch,
):
    module = _load()
    expected_payload = b"the expected augmented model"
    drift_payload = b"X" + expected_payload[1:]
    assert len(drift_payload) == len(expected_payload)
    _bind_expected_runtime_mjb(
        module,
        monkeypatch,
        {
            "relative_locator": "runtime.mjb",
            "sha256": __import__("hashlib").sha256(expected_payload).hexdigest(),
            "size_bytes": len(expected_payload),
        },
    )

    class FakeMujoco:
        @staticmethod
        def mj_saveModel(_model, filename, *_args):
            Path(filename).write_bytes(drift_payload)

    with pytest.raises(module.PlantContractError, match="runtime MJB receipt"):
        module.persist_augmented_runtime_mjb(FakeMujoco, object(), tmp_path)
    assert not (tmp_path / "runtime.mjb").exists()
    assert list(tmp_path.iterdir()) == []


def test_geometry_source_is_exact_checkout_only(tmp_path):
    module = _load()
    expected = module.expected_geometry_source_path()
    assert module.verify_geometry_source(expected) == (
        module.TRUSTED_GEOMETRY_SOURCE_SHA256
    )
    hardlink = tmp_path / "geometry.py"
    os.link(expected, hardlink)
    with pytest.raises(module.PlantContractError, match="path differs"):
        module.verify_geometry_source(hardlink)


def test_manifest_hardlink_matches_canonical_policy_but_symlink_and_drift_fail(
    tmp_path,
):
    module = _load()
    source = module.expected_manifest_path()
    hardlink = tmp_path / source.name
    os.link(source, hardlink)
    assert module.load_pinned_manifest(hardlink)["root_filename"] == (
        "a3p_pingpong_0807.xml"
    )

    symlink = tmp_path / "manifest-link.json"
    symlink.symlink_to(source)
    with pytest.raises(module.PlantContractError):
        module.load_pinned_manifest(symlink)

    drift = tmp_path / "manifest-drift.json"
    drift.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(module.PlantContractError):
        module.load_pinned_manifest(drift)
