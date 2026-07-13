from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "screen_motion_spatial_retarget.py"
MANIFEST_SHA256 = "69bdeabc9b5a934143c52ec6a7fe28ab0a0be6573b2f14f0748e49063c69eb62"
SPEC = importlib.util.spec_from_file_location("screen_motion_spatial_retarget", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _manifest() -> tuple[Path, str]:
    path = ROOT / "configs" / "motion_video_spatial_retarget_prereg_20260712.json"
    assert mod.sha256_file(path) == MANIFEST_SHA256
    return path, MANIFEST_SHA256


def test_tracked_preregistration_is_valid_and_blocks_promotion() -> None:
    path, digest = _manifest()
    plan = mod.validate_manifest(path, digest)
    assert tuple(plan["asset_ids"]) == mod.EXPECTED_ASSETS
    assert plan["capture_table_pose_observed"] is False
    assert plan["virtual_return_contract"]["scorer_dependency"]["sha256"] == (
        "9d01da15a6f24166d4d185ede26a2bd29c9c61d02d15942beadb35b335e0f5ec"
    )
    assert plan["virtual_return_contract"]["capture_radius_m"] == 0.095
    assert plan["promotion_contract"]["topp"] is False
    assert plan["promotion_contract"]["final_arbiter"].endswith("gate3_gate3b_no_reset")


def test_manifest_rejects_asset_skipping_and_capture_extrinsic(tmp_path: Path) -> None:
    source, _ = _manifest()
    plan = json.loads(source.read_text())
    plan["asset_ids"] = plan["asset_ids"][:2]
    bad = tmp_path / "bad-assets.json"
    bad.write_text(json.dumps(plan))
    with pytest.raises(mod.RetargetError, match="all ten motions"):
        mod.validate_manifest(bad, mod.sha256_file(bad))

    plan = json.loads(source.read_text())
    plan["capture_table_pose_observed"] = True
    bad = tmp_path / "bad-extrinsic.json"
    bad.write_text(json.dumps(plan))
    with pytest.raises(mod.RetargetError, match="unobserved"):
        mod.validate_manifest(bad, mod.sha256_file(bad))


def test_atomic_se2_aligns_xy_but_preserves_z_scale_and_chirality() -> None:
    pos = np.asarray([0.2, -0.1, 0.91])
    vel = np.asarray([1.2, 0.4, -0.2])
    normal = np.asarray([0.0, 1.0, 0.0])
    question = np.asarray([0.55, 0.12, 0.87])
    mapped, mapped_vel, mapped_normal, translation, pos_err = mod.solve_atomic_se2(
        pos, vel, normal, question, 10.0
    )
    rotation = mod.rotation_z(10.0)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert mapped[:2] == pytest.approx(question[:2])
    assert mapped[2] == pytest.approx(pos[2])
    assert translation[2] == 0.0
    assert pos_err == pytest.approx(abs(pos[2] - question[2]))
    assert np.linalg.norm(mapped_vel) == pytest.approx(np.linalg.norm(vel))
    assert np.linalg.norm(mapped_normal) == pytest.approx(1.0)


class _AlwaysReturnScorer:
    def __init__(self) -> None:
        self.calls = []
        self.spec = SimpleNamespace(
            capture_radius=0.095,
            net_x=1.87,
            far_x=3.24,
            half_width=0.7625,
        )
        self.net_clear_center_z = 0.9325

    def score(self, **kwargs):
        self.calls.append(kwargs)
        contacted = kwargs["pos_err"] < self.spec.capture_radius
        return SimpleNamespace(
            landed_ok=contacted,
            landing_xy=np.asarray([2.4, 0.0]),
            net_z=1.10,
        )


def _search_plan() -> dict:
    path, digest = _manifest()
    return mod.validate_manifest(path, digest)


def _asset(*, side: str = "backhand", safe: bool = True, eligible: bool = True) -> dict:
    return {
        "asset_id": "franco_backhand_loop_b",
        "input": {"path": "/ignored/motion.pkl", "bytes": 10, "sha256": "c" * 64},
        "side": side,
        "effective_side_after_verified_mirror": side,
        "per_source_frame": [
            {
                "frame": 9,
                "phase": 0.5,
                "hard_safe": safe,
                "candidate_eligible": eligible,
                "racket_site_pos_w_m": [1.75, 0.0, 0.90],
                "racket_site_vel_w_mps": [2.0, 0.0, 0.2],
                "racket_face_normal_w": [1.0, 0.0, 0.0],
                "dense_racket_body_clearance_m": 0.04,
            }
        ],
    }


def _question(*, side: str = "backhand", x: float = 1.90, y: float = 0.10) -> object:
    return mod.Question(
        question_id="a" * 64,
        side=side,
        ball_pos_w=np.asarray([x, y, 0.88]),
        ball_vel_w=np.asarray([-2.0, 0.0, -0.4]),
        ball_spin_w=np.zeros(3),
    )


def test_search_is_all_frame_safe_bounded_and_side_specific() -> None:
    plan = _search_plan()
    scorer = _AlwaysReturnScorer()
    proposals = mod.search_motion_question(_asset(), _question(), "R0", plan, scorer)
    assert len(proposals) == 1
    assert proposals[0]["translation_w_m"] == pytest.approx([0.15, 0.10, 0.0])
    assert proposals[0]["capture_extrinsic_claim"] is False
    assert proposals[0]["transform_semantics"].startswith("atomic_whole_motion")
    assert scorer.calls[0]["clip_id"] == 1
    np.testing.assert_array_equal(scorer.calls[0]["target_normal_raw_a"], [-1.0, 0.0, 0.0])
    assert "racket_normal_raw_a" in scorer.calls[0] and "racket_normal" not in scorer.calls[0]
    changed_source = _asset()
    changed_source["input"] = {**changed_source["input"], "sha256": "e" * 64}
    changed = mod.search_motion_question(changed_source, _question(), "R0", plan, scorer)
    assert changed[0]["candidate_id"] != proposals[0]["candidate_id"]

    assert mod.search_motion_question(_asset(safe=False), _question(), "R0", plan, scorer) == []
    assert mod.search_motion_question(_asset(eligible=False), _question(), "R0", plan, scorer) == []
    assert mod.search_motion_question(_asset(), _question(side="forehand"), "R0", plan, scorer) == []
    # 0.4 m x translation breaches both the x and norm bounds.
    assert mod.search_motion_question(_asset(), _question(x=2.15, y=0.0), "R0", plan, scorer) == []


def _report_binding() -> dict:
    return {"path": "/ignored/report.json", "bytes": 10, "sha256": "b" * 64}


def _certificate(candidate: dict, *, l1: str = "PASS", clearance: float = 0.006) -> dict:
    gates = {
        name: {"verdict": "PASS", "report": _report_binding()}
        for name in mod.REQUIRED_CERTIFICATE_GATES
    }
    gates["l1_vendor_mjcf_self_collision"]["verdict"] = l1
    gates["table_net_swept_clearance"].update(
        {"zero_hard_failures": True, "minimum_clearance_m": clearance}
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "atomic_transform_applied_to_entire_motion": True,
        "capture_extrinsic_claim": False,
        "gates": gates,
    }


def test_candidate_promotion_is_fail_closed_on_all_four_exact_gates() -> None:
    candidate = mod.search_motion_question(
        _asset(), _question(), "R0", _search_plan(), _AlwaysReturnScorer()
    )[0]
    assert mod.validate_certificate(candidate, None) == (False, "missing_candidate_certificate")
    assert mod.validate_certificate(candidate, _certificate(candidate, l1="WARN"))[0] is False
    assert mod.validate_certificate(candidate, _certificate(candidate, clearance=0.0049))[0] is False
    assert mod.validate_certificate(candidate, _certificate(candidate))[0] is True


def _full_result(plan: dict) -> dict:
    questions = [
        {
            "question_id": f"{index:064x}",
            "side": "forehand" if index < 32 else "backhand",
            "ball_pos_w_m": [1.8, 0.0, 0.9],
            "ball_vel_w_mps": [-2.0, 0.0, -0.2],
            "ball_spin_w_radps": [0.0, 0.0, 0.0],
        }
        for index in range(64)
    ]
    assets = []
    for asset_id in mod.EXPECTED_ASSETS:
        side = "forehand" if "forehand" in asset_id else "backhand"
        assets.append(
            {
                "asset_id": asset_id,
                "side": side,
                "frames": 1,
                "selection_status": "no_nonzero_exact_reference_coverage",
                "input": {
                    "path": f"/ignored/{asset_id}.pkl",
                    "bytes": 10,
                    "sha256": ("c" if side == "forehand" else "d") * 64,
                },
                "per_source_frame": [{}],
            }
        )
    return {
        "formal_eligible": False,
        "robot_approved": False,
        "contact_phase_truth": None,
        "frame_contract_evidence": {
            "sha256": plan["predecessor"]["frame_evidence_sha256"],
            "capture_table_pose_observed": False,
        },
        "question_schedule": {
            "consumed_for_returnability": True,
            "semantic_sha256": plan["predecessor"]["question_semantic_sha256"],
            "questions": questions,
        },
        "assets": assets,
    }


def test_predecessor_rejects_table_pose_or_incomplete_asset_paper() -> None:
    plan = _search_plan()
    full = _full_result(plan)
    questions, assets = mod.validate_predecessor_result(full, plan)
    assert len(questions) == 64 and len(assets) == 10

    full["frame_contract_evidence"]["capture_table_pose_observed"] = True
    with pytest.raises(mod.RetargetError, match="capture-table"):
        mod.validate_predecessor_result(full, plan)

    full = _full_result(plan)
    full["assets"].pop()
    with pytest.raises(mod.RetargetError, match="all ten"):
        mod.validate_predecessor_result(full, plan)


def test_atomic_writer_refuses_clobber(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    mod.atomic_write_new(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    with pytest.raises(mod.RetargetError, match="overwrite"):
        mod.atomic_write_new(path, {"ok": False})
