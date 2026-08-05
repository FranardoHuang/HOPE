from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/scripts/"
    "materialize_action_ball_n1_fixed_tape_variants.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_materialize_action_ball_n1_fixed_tape_variants_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

PREPARED_RELATIVE = (
    "configs/action_ball_n1_measured_20260803/"
    "fresh_core_seed0_20260803_take061_robust20n_r7/"
    "take_061_unit04_bh.measured_prepared_core.v1.b22f07369caf.json"
)
ARTIFACT_RELATIVE = (
    "configs/action_ball_n1_measured_20260803/"
    "evidence_holdpass_robust20n_20260803/"
    "take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json"
)
RECEIPT_RELATIVE = (
    "configs/action_ball_n1_measured_20260803/"
    "evidence_holdpass_robust20n_20260803/"
    "take061.robust20n.nominal_hold.v1.json"
)
GEOMETRY_RELATIVE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py"
)


def _geometry():
    return M._load_module(
        "_fixed_tape_variant_geometry_test",
        REPO_ROOT / GEOMETRY_RELATIVE,
    )


def _profile_modules():
    mdp = REPO_ROOT.joinpath(*M.MDP_RELATIVE.parts)
    package_name = "_fixed_tape_variant_profile_test_mdp"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(mdp)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    loaded = {}
    for name in (
        "action_ball_curriculum",
        "action_ball_sampling",
        "action_ball_manifest",
        "action_ball_profile_adapter",
    ):
        loaded[name] = M._load_module(
            "%s.%s" % (package_name, name), mdp / (name + ".py")
        )
    return loaded


def _real_inputs(root: Path = REPO_ROOT):
    prepared = M._strict_json(root / PREPARED_RELATIVE, label="prepared fixture")
    core = M._strict_json(
        root / prepared["core_contact_bundle"]["path"], label="core fixture"
    )
    motion_path = root / M.MOTION_PATH
    manifest = M._strict_json(
        root / core["manifest"]["path"], label="manifest fixture"
    )
    state = M._motion_state(
        motion_path, manifest["actions"][0]["strike_phase"], _geometry()
    )
    return prepared, core, motion_path, state


def _dynamic_ready(root: Path = REPO_ROOT):
    prepared, core, motion_path, state = _real_inputs(root)
    return M._dynamic_ready_source(
        root,
        prepared=prepared,
        core=core,
        motion_path=motion_path,
        motion_state=state,
    )


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_sealed(path: Path, document: dict) -> tuple[str, str]:
    unsigned = deepcopy(document)
    unsigned.pop("content_sha256", None)
    content_sha = hashlib.sha256(_canonical(unsigned)).hexdigest()
    unsigned["content_sha256"] = content_sha
    raw = _canonical(unsigned) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), content_sha


def _copy_fixture(tmp_path: Path):
    root = tmp_path / "repo"
    for relative in (
        M.TRAINING_CONTRACT_RELATIVE.as_posix(),
        M.MOTION_PATH,
        ARTIFACT_RELATIVE,
        RECEIPT_RELATIVE,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    prepared = {"claims": {"dynamic_ready_status": "PASS"}}
    core = {
        "dynamic_ready": {
            "artifact": {
                "path": ARTIFACT_RELATIVE,
                "sha256": M._sha256_file(root / ARTIFACT_RELATIVE),
            },
            "nominal_hold_receipt": {
                "path": RECEIPT_RELATIVE,
                "sha256": M._sha256_file(root / RECEIPT_RELATIVE),
            },
        }
    }
    motion_path = root / M.MOTION_PATH
    state = M._motion_state(
        motion_path, 48.0 / 56.0, _geometry()
    )
    return root, prepared, core, motion_path, state


def test_uses_physical_birth_pose_and_separate_teacher_contact_reference():
    source = _dynamic_ready()
    assert source["ready_root_z"] == pytest.approx(1.0684000253677368)
    assert source["contact_reference_root_z"] == pytest.approx(
        0.8918390870094299
    )

    physical_quat = source["source_contract"]["physical_ready"][
        "root_quat_wxyz"
    ]
    expected_yaw = M._yaw_from_quat(physical_quat)
    expected_projection = (
        math.cos(0.5 * expected_yaw),
        0.0,
        0.0,
        math.sin(0.5 * expected_yaw),
    )
    assert source["base_yaw"] == pytest.approx(expected_yaw, abs=1.0e-15)
    assert source["base_quat"] == pytest.approx(
        expected_projection, abs=1.0e-15
    )

    contract = source["source_contract"]
    assert contract["artifact"] == {
        "path": ARTIFACT_RELATIVE,
        "file_sha256": "ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069",
        "content_sha256": "fff10533205afb4a47db17cdd829b956c8834dada1f9ae872af91fd8fa7a2b91",
    }
    assert contract["nominal_hold_receipt"] == {
        "path": RECEIPT_RELATIVE,
        "file_sha256": "c8b92a28203cbf9b9a4f6dee784d6cc08f3f279672d8a9fc886aa6d92b5bb19b",
        "content_sha256": "21149c53185641cc605be148db7d8d93db0e5d94c29126cc224d075d823b342a",
    }


def test_adapter_preserves_world_contact_z_while_spawning_at_physical_z():
    prepared, core, _, _ = _real_inputs()
    modules = _profile_modules()
    loaded = modules["action_ball_manifest"].load_action_ball_manifest(
        REPO_ROOT / core["manifest"]["path"],
        expected_sha256=core["manifest"]["sha256"],
        verify_referenced_assets=True,
        repo_root=REPO_ROOT,
    )
    source = _dynamic_ready()
    adapted = M._adapt_manifest_for_dynamic_ready(
        modules["action_ball_profile_adapter"], loaded.manifest, source
    )
    profile = adapted.profiles[0]
    original_offset_z = loaded.manifest.actions[
        0
    ].ball_profile.contact_offset_center_b_yaw_m[2]

    assert prepared["claims"]["dynamic_ready_status"] == "PASS"
    assert profile.base_spawn_center_w_m[2] == pytest.approx(
        source["ready_root_z"]
    )
    assert (
        source["ready_root_z"] + profile.contact_offset_center_b_yaw_m[2]
    ) == pytest.approx(
        source["contact_reference_root_z"] + original_offset_z,
        abs=1.0e-12,
    )


def test_recipe_producer_contract_binds_dynamic_source_file_and_content_shas():
    source = _dynamic_ready()["source_contract"]
    contract = M._producer_contract(
        recipe="current_lm",
        algorithm_id="test",
        parameters={},
        source_sha256={"producer": "a" * 64},
        prepared_sha256="b" * 64,
        base_question_sha256="c" * 64,
        dynamic_ready_source=source,
    )
    assert contract["payload"]["dynamic_ready_source"] == source
    assert contract["payload"]["dynamic_ready_source"]["artifact"][
        "file_sha256"
    ] == M._sha256_file(REPO_ROOT / ARTIFACT_RELATIVE)
    assert contract["payload"]["dynamic_ready_source"]["artifact"][
        "content_sha256"
    ] == json.loads((REPO_ROOT / ARTIFACT_RELATIVE).read_text())[
        "content_sha256"
    ]


def test_rejects_resealed_teacher_reference_that_differs_from_motion(
    tmp_path: Path,
):
    root, prepared, core, motion_path, state = _copy_fixture(tmp_path)
    artifact_path = root / ARTIFACT_RELATIVE
    artifact = json.loads(artifact_path.read_text())
    artifact["teacher_reference"]["root_pos_w_m"][2] += 0.01
    artifact_file_sha, artifact_content_sha = _write_sealed(
        artifact_path, artifact
    )
    core["dynamic_ready"]["artifact"]["sha256"] = artifact_file_sha

    receipt_path = root / RECEIPT_RELATIVE
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"]["sha256"] = artifact_file_sha
    receipt["artifact"]["content_sha256"] = artifact_content_sha
    receipt_file_sha, _ = _write_sealed(receipt_path, receipt)
    core["dynamic_ready"]["nominal_hold_receipt"][
        "sha256"
    ] = receipt_file_sha

    with pytest.raises(
        M.ProducerError,
        match="teacher reference differs from motion frame0",
    ):
        M._dynamic_ready_source(
            root,
            prepared=prepared,
            core=core,
            motion_path=motion_path,
            motion_state=state,
        )


def test_rejects_receipt_without_explicit_teacher_physical_separation(
    tmp_path: Path,
):
    root, prepared, core, motion_path, state = _copy_fixture(tmp_path)
    receipt_path = root / RECEIPT_RELATIVE
    receipt = json.loads(receipt_path.read_text())
    receipt["teacher_physical_birth_separated"] = False
    receipt_file_sha, _ = _write_sealed(receipt_path, receipt)
    core["dynamic_ready"]["nominal_hold_receipt"][
        "sha256"
    ] = receipt_file_sha

    with pytest.raises(
        M.ProducerError,
        match="does not prove physical/teacher separation",
    ):
        M._dynamic_ready_source(
            root,
            prepared=prepared,
            core=core,
            motion_path=motion_path,
            motion_state=state,
        )


def test_rejects_prepared_core_without_dynamic_ready_pass():
    prepared, core, motion_path, state = _real_inputs()
    prepared = deepcopy(prepared)
    prepared["claims"]["dynamic_ready_status"] = "BLOCKED_EXTERNAL_EVIDENCE"
    with pytest.raises(M.ProducerError, match="lacks dynamic-ready"):
        M._dynamic_ready_source(
            REPO_ROOT,
            prepared=prepared,
            core=core,
            motion_path=motion_path,
            motion_state=state,
        )


# --- initial-center single-question collapse, end to end -------------------

TRACKED_CORE_RELATIVE = (
    "configs/action_ball_n1_measured_20260803/"
    "fresh_core_seed0_20260803_take061_robust20n_r8_splitready/"
    "take_061_unit04_bh.measured_prepared_core.v1.c5212ce9f41b.json"
)
TRACKED_CORE_SHA256 = (
    "c5212ce9f41b23a4932f470859c6cb6627d245eab8caca3215e993702db60370"
)


def _link_or_copy(source, destination):
    try:
        import os

        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _mirror_repo(tmp_path: Path) -> Path:
    """Mirror the tracked trees ``produce`` reads, without touching the repo.

    ``produce`` resolves every input through ``_repo_path``, which refuses any
    path that escapes the supplied root, so a symlink farm is not usable and
    the output directory must live inside that same root.  Hard links keep this
    close to free while still giving the producer a private root to write into.
    """

    root = tmp_path / "repo"
    for name in ("assets", "configs", "hope_training"):
        shutil.copytree(
            REPO_ROOT / name,
            root / name,
            copy_function=_link_or_copy,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    return root


def test_produce_draws_the_collapsed_center_question_not_the_quota_slot(
    tmp_path,
):
    pytest.importorskip("torch")
    root = _mirror_repo(tmp_path)
    args = types.SimpleNamespace(
        repo_root=str(root),
        prepared_core_bundle=TRACKED_CORE_RELATIVE,
        expected_prepared_core_bundle_sha256=TRACKED_CORE_SHA256,
        seed=0,
        output_dir="out_center",
    )
    report = M.produce(args)
    assert report["status"] == "PASS_DIAGNOSTIC_ONLY"

    runtime = M._load_mdp_modules(root)["action_ball_runtime"]
    receipt_paths = sorted(
        (root / "out_center").glob("*.task_receipt.v5.*.json")
    )
    assert receipt_paths, "producer wrote no task receipts"
    for path in receipt_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        # The sampler ran at level zero with initial_center_single_question on,
        # so the quota slot ("interior" at birth index 0) is not the law; the
        # only legal plan is the literal centre point, and the receipt has to
        # say so or the runtime gate would reject it.
        assert row["initial_center_single_question"] is True
        assert row["sampling_stratum"] == "center"
        assert row["birth_sampling_stratum"] == "center"
        assert row["frontier_arm"] is None
        assert row["birth_frontier_arm"] is None
        assert row["sampling_mixture"]["schedule"][0] == "interior"
        assert row["time_to_contact_tick"] == 91
        assert row["time_to_contact_s"] == 1.82
        assert all(row["domain_levels"][arm] == 0.0 for arm in runtime.ARM_KEYS)
        assert all(
            row["sampling_levels"][arm] == 0.0 for arm in runtime.ARM_KEYS
        )
        # Round-tripping proves the runtime receipt gate admits the collapse.
        parsed = runtime.ActionBallTaskReceipt.from_dict(row)
        assert parsed.initial_center_single_question is True
        assert parsed.to_dict() == row
