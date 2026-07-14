from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_motion_schema2_table_net_clearance.py"
PLAN = ROOT / "configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json"
L1_PLAN = ROOT / "configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json"
L1_VALIDATOR = ROOT / "scripts/audit_motion_schema2_vendor_l1_safety.py"
GEOMETRY = ROOT / (
    "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
    "tasks/table_tennis/geometry.py"
)
COMMAND = ROOT / (
    "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
    "tasks/tracking/mdp/hope_commands.py"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TABLE_NET = _load(SCRIPT, "motion_table_net_test")


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_static_gate_binds_l1_npz_mjcf_closure_and_frame_sources():
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prereg",
            str(PLAN),
            "--expected-prereg-sha256",
            _sha(PLAN),
            "static",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "source_exact=true" in run.stdout
    assert "runtime_audit=false" in run.stdout
    assert "continuous_time_claim=false" in run.stdout
    plan, digest, l1_plan = TABLE_NET.validate_plan(PLAN, _sha(PLAN))
    assert digest == _sha(PLAN)
    assert plan["frozen_vendor_l1"]["certificate"]["sha256"] == (
        "6840df34a6aa6e5636192c705a8ecaa563f751658fe538df428bc317c858db60"
    )
    assert plan["frozen_vendor_l1"]["preregistration"]["sha256"] == _sha(L1_PLAN)
    assert plan["frozen_vendor_l1"]["validator"]["sha256"] == _sha(L1_VALIDATOR)
    assert plan["exact_runtime_input"]["sha256"] == (
        "e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc"
    )
    assert l1_plan["a3_model"]["derived_closure"]["manifest_sha256"] == (
        "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de"
    )
    assert plan["frame_sources"]["table_geometry"]["sha256"] == _sha(GEOMETRY)
    assert plan["frame_sources"]["tracking_command"]["sha256"] == _sha(COMMAND)


def test_hope_to_mjcf_transform_and_all_obstacles_are_exact():
    plan = _plan()
    frame = TABLE_NET.validate_frame_and_geometry_sources(plan)
    assert frame["hope_to_schema2_mjcf"]["translation_m"] == [0.5, 0.7625, 0.76]
    assert frame["hope_to_schema2_mjcf"]["rotation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    obstacles = frame["obstacles"]
    assert [row["name"] for row in obstacles] == [
        "motion_table_top",
        "motion_net",
        "motion_net_post_left",
        "motion_net_post_right",
    ]
    assert obstacles[0]["center_mjcf_world_m"] == [1.87, 0.0, 0.735]
    assert obstacles[1]["full_extents_m"] == [0.01, 1.825, 0.1525]
    assert obstacles[2]["center_mjcf_world_m"][1] == pytest.approx(0.9125)
    assert obstacles[3]["center_mjcf_world_m"][1] == pytest.approx(-0.9125)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("rotate", "frame contract changed"),
        ("translate", "frame contract changed"),
        ("drop_post", "obstacle geometry changed"),
        ("thin_table", "obstacle geometry changed"),
    ],
)
def test_frame_or_obstacle_drift_fails_closed(tmp_path, mutation, match):
    plan = _plan()
    if mutation == "rotate":
        plan["frame_contract"]["hope_to_schema2_mjcf"]["rotation_wxyz"] = [0, 0, 0, 1]
    elif mutation == "translate":
        plan["frame_contract"]["hope_to_schema2_mjcf"]["translation_m"][1] = 0.0
    elif mutation == "drop_post":
        plan["obstacle_geometry"]["net_posts"].pop()
    else:
        plan["obstacle_geometry"]["table_top"]["full_extents_m"][2] = 0.01
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(TABLE_NET.TableNetError, match=match):
        TABLE_NET.validate_plan(path, _sha(path))


def test_model_augmentation_appends_four_boxes_without_reordering_robot_worldbody():
    xml = "<mujoco><worldbody><geom name='floor'/><body name='robot'/></worldbody></mujoco>"
    result = TABLE_NET.augment_mjcf_xml(xml.encode(), _plan()["obstacle_geometry"])
    text = result.decode()
    assert text.index("name=\"robot\"") < text.index("name=\"motion_table_top\"")
    for name in (
        "motion_table_top",
        "motion_net",
        "motion_net_post_left",
        "motion_net_post_right",
    ):
        assert text.count(f'name="{name}"') == 1
    assert "<include" not in text


def test_model_augmentation_rejects_duplicate_bound_obstacle_name():
    xml = "<mujoco><worldbody><geom name='motion_net'/></worldbody></mujoco>"
    with pytest.raises(TABLE_NET.TableNetError, match="already exists"):
        TABLE_NET.augment_mjcf_xml(xml.encode(), _plan()["obstacle_geometry"])


class _DistanceHelper:
    def __init__(self, distances: dict[tuple[int, int], float]):
        self.distances = distances

    def _distance(self, a: int, b: int) -> float:
        if (a, b) in self.distances:
            return self.distances[(a, b)]
        return self.distances[(b, a)]

    def _far(self, model, data, a: int, b: int, threshold: float) -> bool:
        return self._distance(a, b) >= threshold

    def geom_clearance(self, model, data, a: int, b: int, *, distmax: float, tol: float):
        assert distmax == 0.1
        assert tol == 1.0e-6
        return self._distance(a, b), False


@pytest.mark.parametrize(
    ("distance_m", "expected_hard"),
    [(0.00499, True), (0.00500, False), (0.00501, False)],
)
def test_five_mm_boundary_is_exact_saturation_predicate(distance_m, expected_hard):
    helper = _DistanceHelper({(1, 10): distance_m, (2, 10): 0.1})
    result = TABLE_NET.evaluate_robot_obstacle_pairs(
        helper,
        object(),
        object(),
        robot_ids=(1, 2),
        racket_ids=(1,),
        obstacle_ids={"motion_table_top": 10},
        hard_threshold_m=0.005,
        warning_threshold_m=0.02,
        reporting_tolerance_m=1.0e-6,
        geom_name=lambda geom_id: f"geom{geom_id}",
    )
    assert result["hard_failure"] is expected_hard
    assert result["racket_or_handle_hard_failure"] is expected_hard


def test_non_racket_robot_geom_is_in_scope_and_noncompensable():
    helper = _DistanceHelper({(1, 10): 0.1, (2, 10): 0.00499})
    result = TABLE_NET.evaluate_robot_obstacle_pairs(
        helper,
        object(),
        object(),
        robot_ids=(1, 2),
        racket_ids=(1,),
        obstacle_ids={"motion_net": 10},
        hard_threshold_m=0.005,
        warning_threshold_m=0.02,
        reporting_tolerance_m=1.0e-6,
        geom_name=lambda geom_id: f"geom{geom_id}",
    )
    assert result["hard_failure"] is True
    assert result["racket_or_handle_hard_failure"] is False
    with pytest.raises(TABLE_NET.TableNetError, match="table/net clearance hard failure"):
        TABLE_NET.summarize_dense_failures(
            np.array([False, True, False]),
            np.array([False, False, False]),
            np.array([0.0, 0.5, 1.0]),
            source_frames=2,
            unsafe_source_mask_fn=lambda count, times, mask: np.array([True, True]),
        )


def _synthetic_l1_certificate(plan: dict) -> dict:
    l1 = plan["frozen_vendor_l1"]
    canonical = dict(plan["a3_model"]["canonical_mjcf"])
    canonical["path"] = "/workspace/codexschema/nohope/" + canonical["path"]
    return {
        "schema_version": 1,
        "status": l1["required_certificate_status"],
        "completed_utc": "2026-07-14T21:00:00Z",
        "scope": "bound upstream scope",
        "asset_id": "franco_backhand_loop_b",
        "preregistration": {
            "path": l1["preregistration"]["path"],
            "sha256": l1["preregistration"]["sha256"],
        },
        "validator": l1["validator"],
        "runtime": {"synthetic": True},
        "lineage": {
            "motion_npz": {
                "path": plan["exact_runtime_input"]["path"],
                "bytes": 123,
                "sha256": plan["exact_runtime_input"]["sha256"],
            },
            "canonical_mjcf": canonical,
            "derived_mjcf_closure": plan["a3_model"]["derived_closure"],
            "compiled_collision_sha256": plan["a3_model"]["compiled_collision_contract_sha256"],
            "l0_certificate": {"synthetic": True},
        },
        "audit": {
            "sampling": {
                "dense_frames": 1201,
                "effective_sampling_hz": 400,
                "continuous_time_certificate": False,
            },
            "self_collision": {"dangerous_dense_samples": 0},
            "racket_body_clearance": {"dangerous_dense_samples": 0},
            "hard_gate": {"dangerous_dense_samples": 0},
            "mj_step_calls": 0,
        },
        "authorization": {
            "l0_static_complete": True,
            **l1["required_authorization"],
        },
        "explicit_non_claims": ["synthetic"],
        "next_gate": "synthetic",
    }


def test_l1_certificate_must_authorize_table_net(monkeypatch):
    plan = _plan()
    cert = _synthetic_l1_certificate(plan)
    monkeypatch.setattr(TABLE_NET, "_verify_absolute_sha", lambda *args: Path("/synthetic"))
    monkeypatch.setattr(TABLE_NET, "read_json", lambda *args: cert)
    binding = TABLE_NET.validate_vendor_l1_certificate(plan)
    assert binding["authorization"]["table_net_authorized"] is True
    cert["authorization"]["table_net_authorized"] = False
    with pytest.raises(TABLE_NET.TableNetError, match="authorization changed"):
        TABLE_NET.validate_vendor_l1_certificate(plan)


def test_contract_cannot_claim_continuous_time_or_weaken_hard_gate(tmp_path):
    plan = _plan()
    plan["audit_contract"]["dense_sampling_is_continuous_time_certificate"] = True
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(TABLE_NET.TableNetError, match="audit contract changed"):
        TABLE_NET.validate_plan(path, _sha(path))
    plan = _plan()
    plan["audit_contract"]["hard_clearance_m"] = 0.004
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(TABLE_NET.TableNetError, match="audit contract changed"):
        TABLE_NET.validate_plan(path, _sha(path))


def test_dry_run_never_writes_and_output_preflight_precedes_runtime(monkeypatch, tmp_path):
    plan = _plan()
    output = tmp_path / "certificate.json"
    plan["output_contract"]["certificate_path"] = str(output)
    calls: list[tuple] = []
    monkeypatch.setattr(TABLE_NET, "validate_plan", lambda *args: (plan, "a" * 64, {"l1": True}))
    monkeypatch.setattr(
        TABLE_NET,
        "build_certificate",
        lambda *args: calls.append(args) or {"status": "synthetic-pass"},
    )
    monkeypatch.setattr(
        TABLE_NET,
        "write_exclusive",
        lambda *args: pytest.fail("dry-run attempted certificate publication"),
    )
    assert TABLE_NET.main(
        ["--prereg", str(PLAN), "--expected-prereg-sha256", "a" * 64, "dry-run"]
    ) == 0
    assert len(calls) == 1
    assert not output.exists()


def test_no_clobber_certificate_write(tmp_path):
    output = tmp_path / "certificate.json"
    TABLE_NET.write_exclusive(output, {"status": "first"})
    original = output.read_bytes()
    with pytest.raises(TABLE_NET.TableNetError, match="already exists"):
        TABLE_NET.write_exclusive(output, {"status": "second"})
    assert output.read_bytes() == original
