from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MJCF_SHA256 = (
    "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"
)
CURRENT_MJCF_SHA256 = (
    "70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a"
)
SCRIPT = ROOT / "scripts" / "run_motion_video_canonical_gmr_ground_queue.py"
SPEC = importlib.util.spec_from_file_location("run_motion_video_canonical_gmr_ground_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUEUE
SPEC.loader.exec_module(QUEUE)


def _tracked_plan() -> tuple[Path, dict]:
    path = ROOT / "configs" / "motion_video_canonical_gmr_ground_prereg_v2_20260711.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_tracked_plan_binds_tools_source_inputs_mjcf_and_collision_contract():
    plan_path, _ = _tracked_plan()
    plan = QUEUE.validate_plan(plan_path, QUEUE.sha256_file(plan_path))
    canonical_result_path = ROOT / "configs" / "motion_video_canonical_gmr_results_20260711.json"
    canonical_result = json.loads(canonical_result_path.read_text(encoding="utf-8"))
    ground_tool = ROOT / "scripts" / "ground_gmr_pkl.py"
    mjcf = (
        ROOT
        / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
    )

    assert plan["queue_tool"]["sha256"] == QUEUE.sha256_file(SCRIPT)
    assert plan["queue_tool"]["bytes"] == SCRIPT.stat().st_size
    assert plan["grounding_tool"]["sha256"] == QUEUE.sha256_file(ground_tool)
    assert plan["grounding_tool"]["bytes"] == ground_tool.stat().st_size
    assert plan["canonical_gmr_result"]["sha256"] == QUEUE.sha256_file(canonical_result_path)
    assert plan["canonical_gmr_result"]["bytes"] == canonical_result_path.stat().st_size
    assert plan["mjcf"]["sha256"] == HISTORICAL_MJCF_SHA256
    assert plan["mjcf"]["bytes"] == 49107
    assert QUEUE.sha256_file(mjcf) == CURRENT_MJCF_SHA256
    assert CURRENT_MJCF_SHA256 != HISTORICAL_MJCF_SHA256
    assert mjcf.stat().st_size == 49134 != plan["mjcf"]["bytes"]
    assert plan["compiled_collision_contract"]["expected_sha256"] == (
        "18e7f6ffbefba9dbd988f7c3cb9fb92b250777862fc25fa3d4a0b2ca0f8386e5"
    )
    assert len(plan["compiled_collision_contract"]["enabled_robot_geom_ids"]) == 37

    by_id = {row["asset_id"]: row for row in canonical_result["results"]}
    assert plan["processing_order"] == [row["asset_id"] for row in plan["inputs"]]
    assert len(plan["inputs"]) == len(by_id) == 10
    for row in plan["inputs"]:
        source = by_id[row["asset_id"]]
        assert row["frames"] == source["frames"]
        assert row["input"] == {
            "path": source["output_path"],
            "bytes": source["output_bytes"],
            "sha256": source["output_sha256"],
        }


def test_plan_hash_body_shape_and_collision_semantics_fail_closed(tmp_path: Path):
    tracked_path, plan = _tracked_plan()
    with pytest.raises(QUEUE.QueueError, match="plan sha256"):
        QUEUE.validate_plan(tracked_path, "0" * 64)

    plan["body_shape_contract"] = "diagnostic_video_betas"
    path = tmp_path / "bad-shape.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(QUEUE.QueueError, match="body_shape_contract"):
        QUEUE.validate_plan(path, QUEUE.sha256_file(path))

    _, plan = _tracked_plan()
    plan["compiled_collision_contract"]["enabled_robot_geom_ids"][0] = 78
    path = tmp_path / "bad-collision.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(QUEUE.QueueError, match="unique sorted geom ids"):
        QUEUE.validate_plan(path, QUEUE.sha256_file(path))


def test_runtime_verifier_preserves_venv_launcher_symlink(tmp_path: Path):
    python_link = tmp_path / "venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(Path(sys.executable).resolve())
    version_result = __import__("subprocess").run(
        [str(python_link), "--version"], capture_output=True, text=True, check=True
    )
    version = (version_result.stdout or version_result.stderr).strip()
    mjcf = tmp_path / "model.xml"
    mjcf.write_text("<mujoco/>", encoding="utf-8")
    ground = ROOT / "scripts" / "ground_gmr_pkl.py"
    plan = {
        "queue_tool": {
            "path": SCRIPT.name,
            "bytes": SCRIPT.stat().st_size,
            "sha256": QUEUE.sha256_file(SCRIPT),
        },
        "grounding_tool": {
            "path": ground.name,
            "bytes": ground.stat().st_size,
            "sha256": QUEUE.sha256_file(ground),
        },
        "mjcf": {
            "path": str(mjcf),
            "bytes": mjcf.stat().st_size,
            "sha256": QUEUE.sha256_file(mjcf),
        },
        "python_environment": {"executable": str(python_link), "version": version},
    }
    _, _, verified_python = QUEUE.verify_tools_and_runtime(plan)
    assert verified_python == python_link.absolute()
    assert verified_python != python_link.resolve()


def test_source_manifest_verifier_binds_all_physical_inputs(tmp_path: Path):
    rows = []
    results = []
    for index in range(10):
        asset_id = f"asset{index}"
        source = tmp_path / f"{asset_id}.pkl"
        source.write_bytes(f"payload-{index}".encode())
        binding = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": QUEUE.sha256_file(source),
        }
        rows.append({"asset_id": asset_id, "frames": index + 2, "input": binding})
        results.append(
            {
                "asset_id": asset_id,
                "frames": index + 2,
                "output_path": str(source),
                "output_bytes": source.stat().st_size,
                "output_sha256": QUEUE.sha256_file(source),
            }
        )
    manifest = {
        "status": "complete_diagnostic_canonical_gmr",
        "body_shape_contract": QUEUE.BODY_SHAPE_CONTRACT,
        "formal_eligible": False,
        "results": results,
    }
    manifest_path = tmp_path / "result.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan = {
        "canonical_gmr_result": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": QUEUE.sha256_file(manifest_path),
        },
        "inputs": rows,
    }
    verified = QUEUE.verify_source_manifest(plan)
    assert [row["asset_id"] for row in verified] == [f"asset{i}" for i in range(10)]

    manifest["results"][5]["output_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan["canonical_gmr_result"].update(
        bytes=manifest_path.stat().st_size,
        sha256=QUEUE.sha256_file(manifest_path),
    )
    with pytest.raises(QUEUE.QueueError, match="asset5.output_sha256 mismatch"):
        QUEUE.verify_source_manifest(plan)


def test_validate_report_requires_constant_shift_exact_invariants_and_bound_collision(tmp_path: Path):
    plan_path, plan = _tracked_plan()
    QUEUE.validate_plan(plan_path, QUEUE.sha256_file(plan_path))
    output = tmp_path / "grounded.pkl"
    output.write_bytes(b"grounded")
    ground_path = tmp_path / "ground_gmr_pkl.py"
    ground_path.write_bytes(b"tool")
    mjcf = tmp_path / "model.xml"
    mjcf.write_bytes(b"mjcf")
    row = plan["inputs"][0]
    report = {
        "status": "pass",
        "formal_eligible": False,
        "input": row["input"],
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": QUEUE.sha256_file(output),
        },
        "mjcf": {
            "path": str(mjcf),
            "bytes": plan["mjcf"]["bytes"],
            "sha256": plan["mjcf"]["sha256"],
            "compiled_kinematic_collision_sha256": plan["compiled_collision_contract"]["expected_sha256"],
        },
        "tool": {"path": str(ground_path), "sha256": plan["grounding_tool"]["sha256"]},
        "collision_contract": {
            field: plan["compiled_collision_contract"][field]
            for field in (
                "robot_root_body_id",
                "enabled_robot_geom_count",
                "enabled_robot_geom_ids",
                "surface_method",
                "visual_only_geoms_excluded",
            )
        },
        "structure": {"frames": row["frames"], "fps": 30.0},
        "invariants": {
            "root_xy_exact": True,
            "root_rotation_exact": True,
            "dof_position_exact": True,
            "root_pos_dtype_preserved": True,
            "all_other_payload_fields_shallow_preserved": True,
            "root_z_relative_trajectory_max_error_m": 0.0,
        },
        "grounding": {
            "applied_root_z_shift_min_m": 0.08,
            "applied_root_z_shift_max_m": 0.08,
            "after": {"minimum_clearance_m": 1e-5},
        },
    }
    QUEUE.validate_report(
        report,
        row=row,
        output=output,
        ground_path=ground_path.resolve(),
        mjcf=mjcf.resolve(),
        plan=plan,
    )

    report["grounding"]["applied_root_z_shift_max_m"] = 0.081
    with pytest.raises(QUEUE.QueueError, match="shift is not constant"):
        QUEUE.validate_report(
            report,
            row=row,
            output=output,
            ground_path=ground_path.resolve(),
            mjcf=mjcf.resolve(),
            plan=plan,
        )
