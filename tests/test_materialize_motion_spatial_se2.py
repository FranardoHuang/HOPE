from __future__ import annotations

import copy
import importlib.util
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_motion_spatial_se2.py"
SPEC = importlib.util.spec_from_file_location("materialize_motion_spatial_se2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

B_PLAN = ROOT / "configs" / "motion_backhand_loop_b_se2_materialization_prereg_20260714.json"
C_PLAN = ROOT / "configs" / "motion_backhand_loop_c_se2_materialization_prereg_20260714.json"
B_SHA = "e016ca742dfebbd9726b03df1ad3cd7e75f19a07557e5e458e57e00088751aee"
C_SHA = "27f938cd6016fcadada8c6ea806329279c379ccb77b5db99b8902275ebd9d454"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def payload(frames: int = 5, dtype=np.float64, *, velocities: bool = True) -> dict:
    source = {
        "fps": 30,
        "root_pos": np.array(
            [[0.1 * i, (-1) ** i * 0.03 * i, 0.88 + 0.001 * i] for i in range(frames)],
            dtype=dtype,
        ),
        "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=dtype), (frames, 1)),
        "dof_pos": np.arange(frames * 31, dtype=dtype).reshape(frames, 31) / 100.0,
        "local_body_pos": None,
        "link_body_list": None,
    }
    if velocities:
        source["root_lin_vel_world"] = np.tile(
            np.array([1.0, 0.25, 0.0], dtype=dtype), (frames, 1)
        )
        source["root_ang_vel_world"] = np.tile(
            np.array([0.0, 0.0, 0.5], dtype=dtype), (frames, 1)
        )
    return source


@pytest.mark.parametrize(
    ("path", "sha", "asset", "candidate"),
    [
        (B_PLAN, B_SHA, "franco_backhand_loop_b", "98e7b883b29d"),
        (C_PLAN, C_SHA, "franco_backhand_loop_c", "aa0c86fd3509"),
    ],
)
def test_tracked_preregs_are_exact_independent_primary_only_contracts(path, sha, asset, candidate):
    assert M.sha256_file(path) == sha
    plan, actual = M.validate_plan(path, sha)
    assert actual == sha
    assert plan["asset_id"] == asset
    assert plan["candidate_id"].startswith(candidate)
    assert plan["fallback_policy"]["automatic_fallback"] is False
    assert plan["formal_eligible"] is False
    assert plan["training_authorized"] is False
    assert plan["hardware_authorized"] is False
    assert plan["schema2_materialized"] is False


def test_transform_is_proper_left_action_and_preserves_non_spatial_fields():
    source = payload()
    translation = [0.2, -0.1, 0.0]
    output = M.transform_payload(source, translation_w_m=translation, yaw_deg=90.0)
    np.testing.assert_allclose(output["root_pos"][0, :2], [0.2, -0.1], atol=1e-15)
    np.testing.assert_allclose(output["root_pos"][1, :2], [0.23, 0.0], atol=1e-15)
    np.testing.assert_array_equal(output["root_pos"][:, 2], source["root_pos"][:, 2])
    expected_yaw = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    np.testing.assert_allclose(output["root_rot"], np.tile(expected_yaw, (5, 1)), atol=1e-15)
    np.testing.assert_allclose(output["root_lin_vel_world"][0], [-0.25, 1.0, 0.0], atol=1e-15)
    np.testing.assert_array_equal(output["root_ang_vel_world"], source["root_ang_vel_world"])
    np.testing.assert_array_equal(output["dof_pos"], source["dof_pos"])
    assert output["fps"] is source["fps"]
    assert output["local_body_pos"] is None and output["link_body_list"] is None
    report = M.verify_transform(
        source,
        output,
        translation_w_m=translation,
        yaw_deg=90.0,
        tolerance=1e-12,
    )
    assert report["proper_rotation_determinant"] == pytest.approx(1.0)
    assert report["root_z_bit_exact"] is True
    assert report["world_vector_fields_rotated"] == [
        "root_ang_vel_world",
        "root_lin_vel_world",
    ]


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 1e-12), (np.float32, 2e-7)])
def test_inverse_and_rigid_distance_survive_pickle_round_trip(dtype, tolerance):
    source = payload(dtype=dtype)
    output = M.transform_payload(
        source,
        translation_w_m=[0.05035998433, -0.109155849041, 0.0],
        yaw_deg=-5.0,
    )
    reloaded = M.restricted_pickle_loads(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL))
    result = M.verify_transform(
        source,
        reloaded,
        translation_w_m=[0.05035998433, -0.109155849041, 0.0],
        yaw_deg=-5.0,
        tolerance=tolerance,
    )
    assert result["root_pairwise_distance_max_abs_error_m"] <= tolerance
    assert result["root_position_inverse_max_abs_error"] <= tolerance
    assert result["root_quaternion_inverse_max_abs_error"] <= tolerance


def test_unknown_field_non_null_local_and_unsafe_pickle_fail_closed():
    source = payload(velocities=False)
    source["mystery_world_thing"] = np.zeros((5, 3))
    with pytest.raises(M.MaterializationError, match="unknown"):
        M.validate_payload(source, frames=5)

    source = payload(velocities=False)
    source["local_body_pos"] = np.zeros((5, 2, 3))
    with pytest.raises(M.MaterializationError, match="requires local_body_pos"):
        M.validate_payload(source, frames=5)

    class Unsafe:
        def __reduce__(self):
            return (os.system, ("false",))

    with pytest.raises(M.MaterializationError, match="not allowlisted"):
        M.restricted_pickle_loads(pickle.dumps({"bad": Unsafe()}))


def test_grounding_and_mirror_guards_fail_closed():
    source = payload(velocities=False)
    with pytest.raises(M.MaterializationError, match="ground-preserving"):
        M.transform_payload(source, translation_w_m=[0.0, 0.0, 0.001], yaw_deg=0.0)

    output = M.transform_payload(source, translation_w_m=[0.1, 0.2, 0.0], yaw_deg=-10.0)
    output["root_pos"][2, 2] += 1e-6
    with pytest.raises(M.MaterializationError, match="root z changed"):
        M.verify_transform(
            source,
            output,
            translation_w_m=[0.1, 0.2, 0.0],
            yaw_deg=-10.0,
            tolerance=1e-12,
        )

    original = M.yaw_rotation
    try:
        M.yaw_rotation = lambda _yaw: (np.diag([-1.0, 1.0, 1.0]), np.array([0, 0, 0, 1]))
        with pytest.raises(M.MaterializationError, match="improper/mirrored"):
            M.transform_payload(source, translation_w_m=[0.0, 0.0, 0.0], yaw_deg=0.0)
    finally:
        M.yaw_rotation = original


def runtime_plan(tmp_path: Path) -> tuple[dict, Path]:
    plan = copy.deepcopy(json.loads(B_PLAN.read_text(encoding="utf-8")))
    source_path = tmp_path / "source.pkl"
    source_path.write_bytes(pickle.dumps(payload(frames=5), protocol=pickle.HIGHEST_PROTOCOL))
    plan["source_motion"] = {
        "path": str(source_path),
        "bytes": source_path.stat().st_size,
        "sha256": M.sha256_file(source_path),
    }
    plan["payload_contract"]["frames"] = 5
    plan["output_contract"]["output_root"] = str(tmp_path / "published")
    return plan, source_path


def test_inspect_consume_report_last_and_no_clobber(tmp_path, monkeypatch):
    plan, _source_path = runtime_plan(tmp_path)
    evidence = M.inspect_inputs(plan)
    assert evidence["structure"]["frames"] == 5
    assert not Path(plan["output_contract"]["output_root"]).exists()

    link_order: list[str] = []
    real_link = M.os.link

    def recording_link(source, destination):
        link_order.append(Path(destination).name)
        return real_link(source, destination)

    monkeypatch.setattr(M.os, "link", recording_link)
    report_path = M.consume(plan, tmp_path / "fixture-prereg.json", "3" * 64, evidence)
    assert link_order == [
        plan["output_contract"]["motion_filename"],
        plan["output_contract"]["report_filename"],
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == M.RESULT_STATUS
    assert report["candidate_id"] == plan["candidate_id"]
    assert report["fallback_advanced"] is False
    assert report["schema2_materialized"] is False
    assert report["invariants"]["root_z_bit_exact"] is True
    output_motion = Path(report["output_motion"]["path"])
    assert M.sha256_file(output_motion) == report["output_motion"]["sha256"]
    with pytest.raises(M.MaterializationError, match="already exists"):
        M.consume(plan, tmp_path / "fixture-prereg.json", "3" * 64, evidence)


def test_report_publication_failure_never_deletes_concurrent_foreign_file(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "motion.pkl").write_bytes(b"motion")
    (staging / "report.json").write_bytes(b"report")
    output = tmp_path / "output"
    real_link = M.os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_link(source, destination)
        (output / "foreign.keep").write_bytes(b"do-not-delete")
        raise OSError("injected report-link failure")

    monkeypatch.setattr(M.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected"):
        M.publish_report_last(staging, output, "motion.pkl", "report.json")
    assert (output / "foreign.keep").read_bytes() == b"do-not-delete"
    assert not (output / "motion.pkl").exists()


def test_primary_or_fallback_mutation_is_rejected(tmp_path):
    plan = json.loads(B_PLAN.read_text(encoding="utf-8"))
    plan["candidate_id"] = "0" * 64
    path = tmp_path / "bad.json"
    write_json(path, plan)
    with pytest.raises(M.MaterializationError, match="candidate"):
        M.validate_plan(path, M.sha256_file(path))

    plan = json.loads(B_PLAN.read_text(encoding="utf-8"))
    plan["fallback_policy"]["automatic_fallback"] = True
    write_json(path, plan)
    with pytest.raises(M.MaterializationError, match="fallback"):
        M.validate_plan(path, M.sha256_file(path))

    plan = json.loads(B_PLAN.read_text(encoding="utf-8"))
    plan["output_contract"]["output_root"] = "/tmp/unbound-output"
    write_json(path, plan)
    with pytest.raises(M.MaterializationError, match="output root differs"):
        M.validate_plan(path, M.sha256_file(path))

    plan = json.loads(B_PLAN.read_text(encoding="utf-8"))
    plan["unexpected_claim"] = True
    write_json(path, plan)
    with pytest.raises(M.MaterializationError, match="plan keys changed"):
        M.validate_plan(path, M.sha256_file(path))
