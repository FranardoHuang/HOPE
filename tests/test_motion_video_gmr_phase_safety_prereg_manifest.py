from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "configs/motion_video_gmr_phase_safety_prereg_20260711.json"
GROUND = REPO / "configs/motion_video_canonical_gmr_ground_results_20260711.json"
SCREEN = REPO / "scripts/screen_motion_gmr_phase_safety.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_screen():
    spec = importlib.util.spec_from_file_location("screen_motion_gmr_phase_safety_contract", SCREEN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ready_prereg_is_content_addressed_and_crosschecks_canonical_grounding():
    module = _load_screen()
    expected_plan_sha = "232cd9ef1a72381895b54c75cc87c82e991d9c605ea169e86605b3afb9e64e15"
    assert _sha(PLAN) == expected_plan_sha
    plan = module.validate_manifest(
        PLAN,
        expected_plan_sha,
        require_ready=True,
        verify_files=False,
    )
    assert plan["execution_ready"] is True
    assert plan["cpu_only"] is True
    assert plan["CUDA_VISIBLE_DEVICES"] == ""
    assert plan["real_robot_commands_authorized"] is False
    assert plan["contact_phase_truth"] is None
    assert plan["formal_eligible"] is False
    assert plan["frame_contract"]["returnability_enabled"] is False
    assert plan["frame_contract"]["gmr_world_to_hope_matrix_4x4"] is None
    assert plan["dense_safety_contract"]["substeps_per_source_interval"] == 8
    assert plan["dense_safety_contract"]["effective_sampling_hz"] == 240
    assert plan["question_schedule"]["count_per_side"] == 32
    assert (
        plan["question_schedule"]["expected_semantic_sha256"]
        == "4dfa0548d898fa6456d09261216e26ff9547c1a9dca2da221835c3cfa25332c7"
    )

    ground = json.loads(GROUND.read_text(encoding="utf-8"))
    assert plan["canonical_grounding_result_manifest"]["sha256"] == _sha(GROUND)
    assert plan["canonical_grounding_result_manifest"]["bytes"] == GROUND.stat().st_size
    source = {row["asset_id"]: row for row in ground["results"]}
    assert list(source) == plan["expected_asset_ids"]
    for row in plan["inputs"]:
        accepted = source[row["asset_id"]]
        assert row["input"] == accepted["output"]
        assert row["grounding_report"] == accepted["report"]
        assert row["frames"] == accepted["structure"]["frames"]


def test_prereg_binds_exact_source_tools_and_two_vs_four_papers():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    # The v4 preregistration is immutable runtime history.  Its screen tool is
    # intentionally the accepted v4 bytes; the live tool now contains the
    # separately preregistered per-asset counterfactual-frame path.
    assert plan["tool_contract"]["screen"] == {
        "path": "/workspace/codexschema/motion_video_intake_20260711/phase_safety_control_v4/screen_motion_gmr_phase_safety.py",
        "bytes": 69769,
        "sha256": "3244c3ff395ad10809d478b9469cb867555be4cf397ace606834dc0de9f3e302",
    }
    local_tools = {
        "grounding_dependency": REPO / "scripts/ground_gmr_pkl.py",
        "self_collision_dependency": (
            REPO / "hope_training/whole_body_tracking/scripts/audit_self_collision.py"
        ),
        "motion_audit_dependency": (
            REPO / "hope_training/whole_body_tracking/scripts/audit_motion_npz.py"
        ),
        "virtual_return_dependency": (
            REPO / "hope_training/whole_body_tracking/scripts/virtual_return_scorer.py"
        ),
    }
    for name, path in local_tools.items():
        assert plan["tool_contract"][name]["sha256"] == _sha(path)
        assert plan["tool_contract"][name]["bytes"] == path.stat().st_size

    assert plan["libraries"]["franco_two_blocks"] == [
        "franco_forehand_block",
        "franco_backhand_block",
    ]
    for suffix in ("a", "b", "c"):
        members = plan["libraries"][f"franco_four_loop_{suffix}"]
        assert len(members) == 4
        assert "franco_forehand_loop" in members
        assert f"franco_backhand_loop_{suffix}" in members
    assert len(plan["libraries"]["v6_two_blocks"]) == 2
    assert len(plan["libraries"]["v7_two_blocks"]) == 2
