from __future__ import annotations

import hashlib
import io
import importlib.util
import ast
import builtins
import json
import math
import os
import subprocess
import sys
import types
import zipfile
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
RUNTIME_JOINT_ORDER = ROOT / "configs/a3_runtime_articulation_joint_order.txt"
L0_V1_PLAN = ROOT / "configs/motion_backhand_loop_b_l0_static_prereg_20260714.json"
REJECTED_V3_RESULT = ROOT / (
    "configs/motion_backhand_loop_b_table_net_v3_rejected_result_20260715.json"
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
GROUND = _load(ROOT / "scripts/ground_gmr_pkl.py", "motion_table_net_ground_test")


def _tracked_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _plan() -> dict:
    """Rebind mutable source snapshots in memory, preserving the frozen plan."""
    plan = _tracked_plan()
    for source in plan["frame_sources"].values():
        path = ROOT / source["path"]
        source["bytes"] = path.stat().st_size
        source["sha256"] = _sha(path)
    return plan


def _write_plan_fixture(tmp_path: Path, plan: dict | None = None) -> Path:
    path = tmp_path / "current-source-table-net-plan.json"
    path.write_text(json.dumps(_plan() if plan is None else plan), encoding="utf-8")
    return path


def test_static_gate_binds_l1_npz_mjcf_closure_and_frame_sources(tmp_path):
    historical = subprocess.run(
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
    assert historical.returncode == 2
    assert "tracking command source bytes" in historical.stderr

    plan_path = _write_plan_fixture(tmp_path)
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prereg",
            str(plan_path),
            "--expected-prereg-sha256",
            _sha(plan_path),
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
    plan, digest, l1_plan = TABLE_NET.validate_plan(plan_path, _sha(plan_path))
    assert digest == _sha(plan_path)
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
    rejected = plan["rejected_prior_certificate"]
    assert rejected["sha256"] == (
        "39d6cc38941acfed2aa57e09add90660f946d16849ba3a8f02581fe646a79a19"
    )
    assert rejected["accepted_as_table_net_complete"] is False
    assert rejected["dynamics_authorized"] is False
    output = plan["output_contract"]["certificate_path"]
    assert "/table_net_primary_v4/" in output
    assert output.endswith(".table_net_clearance_v4_certificate.json")
    assert output != rejected["path"]


def test_rejected_v3_certificate_cannot_be_reauthorized_in_v4_plan(tmp_path):
    plan = _plan()
    plan["rejected_prior_certificate"]["accepted_as_table_net_complete"] = True
    path = tmp_path / "forged-v4-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(TABLE_NET.TableNetError, match="rejected prior.*binding changed"):
        TABLE_NET.validate_plan(path, _sha(path))


def test_rejected_v3_result_and_v4_successor_are_cross_bound():
    plan = _plan()
    result = json.loads(REJECTED_V3_RESULT.read_text(encoding="utf-8"))
    rejected = result["rejected_certificate"]
    assert {
        "path": rejected["path"],
        "bytes": rejected["bytes"],
        "sha256": rejected["sha256"],
    } == {
        key: plan["rejected_prior_certificate"][key]
        for key in ("path", "bytes", "sha256")
    }
    assert rejected["rejection"]["accepted_as_table_net_complete"] is False
    assert rejected["rejection"]["dynamics_authorized"] is False
    redteam = result["v4_premerge_redteam_rejection"]
    assert redteam["source_commit"] == "7241157e036a8892c80319457ba5c7cdf31d69b2"
    assert redteam["status"] == "no_merge_no_runtime"
    assert redteam["runtime_executed"] is False
    assert redteam["certificate_present"] is False
    assert redteam["dynamics_authorized"] is False
    successor = result["successor_v4"]
    assert successor["preregistration_sha256"] == _sha(PLAN)
    assert successor["validator_sha256"] == _sha(SCRIPT)
    assert successor["certificate_path"] == plan["output_contract"]["certificate_path"]
    assert successor["runtime_executed"] is False
    assert successor["certificate_present"] is False


def test_local_phase_kernels_are_ast_identical_to_exact_upstream():
    plan = _plan()
    validator = TABLE_NET.read_file_snapshot(SCRIPT, "table/net validator")
    binding = json.loads(L1_PLAN.read_text(encoding="utf-8"))["dependencies"][
        "dense_safety_tool"
    ]
    upstream = TABLE_NET.read_file_snapshot(
        ROOT / binding["path"],
        "dense safety reference",
        expected_bytes=binding["bytes"],
        expected_sha256=binding["sha256"],
    )
    hashes = TABLE_NET.validate_phase_pure_function_parity(validator, upstream)
    assert set(hashes) == set(TABLE_NET.PURE_PHASE_FUNCTIONS)
    assert all(len(value) == 64 for value in hashes.values())
    assert plan["validator"]["path"] == "scripts/audit_motion_schema2_table_net_clearance.py"


def test_bound_production_joint_order_skips_upstream_comment_header_exactly():
    l0_plan = json.loads(L0_V1_PLAN.read_text(encoding="utf-8"))
    binding = l0_plan["upstream_contracts"]["runtime_joint_order"]
    snapshot = TABLE_NET.read_file_snapshot(
        RUNTIME_JOINT_ORDER,
        "production runtime joint order",
        expected_bytes=binding["bytes"],
        expected_sha256=binding["sha256"],
    )
    assert snapshot.data.startswith(b"# Isaac/runtime articulation")
    names = TABLE_NET._read_names_snapshot(snapshot, 31, "runtime joint order")
    assert len(names) == len(set(names)) == 31
    assert names[0] == "left_hip_pitch_joint"
    assert names[-1] == "right_wrist_yaw_joint"
    assert all(not name.startswith("#") for name in names)


def test_name_snapshot_rejects_noncomment_metadata_as_a_32nd_joint(tmp_path):
    data = RUNTIME_JOINT_ORDER.read_text(encoding="utf-8") + "schema_version=1\n"
    path = tmp_path / "joint_order_with_unmarked_metadata.txt"
    path.write_text(data, encoding="utf-8")
    snapshot = TABLE_NET.read_file_snapshot(path, "mutated runtime joint order")
    with pytest.raises(TABLE_NET.TableNetError, match="exactly 31 unique names"):
        TABLE_NET._read_names_snapshot(snapshot, 31, "runtime joint order")


def test_name_snapshot_rejects_duplicate_joint_after_comment_filtering(tmp_path):
    lines = RUNTIME_JOINT_ORDER.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[1]
    lines.insert(1, "   # second human-readable comment")
    path = tmp_path / "joint_order_with_duplicate.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    snapshot = TABLE_NET.read_file_snapshot(path, "mutated runtime joint order")
    with pytest.raises(TABLE_NET.TableNetError, match="exactly 31 unique names"):
        TABLE_NET._read_names_snapshot(snapshot, 31, "runtime joint order")


def test_project_module_sys_modules_injection_is_never_consumed(monkeypatch, tmp_path):
    banned = {"ground_gmr_pkl", "virtual_return_scorer", "audit_motion_npz"}
    for name in banned:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in banned:
            imported.append(name)
            raise AssertionError(f"unbound project import attempted: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path_before = list(sys.path)
    modules_before = dict(sys.modules)
    plan_path = _write_plan_fixture(tmp_path)
    TABLE_NET.validate_plan(plan_path, _sha(plan_path))
    assert imported == []
    assert sys.path == path_before
    assert set(sys.modules) == set(modules_before)
    assert all(sys.modules[name] is module for name, module in modules_before.items())


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
    assert frame["net_post_source"]["full_extents_m"] == [0.02, 0.02, 0.1725]


def test_net_post_geometry_source_semantics_fail_closed(tmp_path):
    source = ROOT / (
        "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/"
        "tasks/table_tennis/table_tennis_env_cfg.py"
    )
    mutated = tmp_path / "table_tennis_env_cfg.py"
    mutated.write_text(
        source.read_text(encoding="utf-8").replace(
            "size=(0.02, 0.02, post_h)", "size=(0.01, 0.02, post_h)"
        ),
        encoding="utf-8",
    )
    snapshot = TABLE_NET.read_file_snapshot(mutated, "mutated table scene")
    with pytest.raises(TABLE_NET.TableNetError, match="post extents changed"):
        TABLE_NET.validate_net_post_geometry_source(snapshot)


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


def test_model_augmentation_appends_four_boxes_after_robot_xml_children():
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


class _FakeCompiledModel:
    pass


def _fake_compiled_models():
    canonical = _FakeCompiledModel()
    augmented = _FakeCompiledModel()
    canonical.ngeom = 38
    augmented.ngeom = 42
    for model in (canonical, augmented):
        model.nbody = 2
        model.njnt = 1
        model.nq = 1
        model.nv = 1
        model.nu = 0
        model.na = 0
        model.qpos0 = np.array([0.25], dtype=np.float64)
        model.body_parentid = np.array([0, 0], dtype=np.int32)
        model.body_pos = np.zeros((2, 3), dtype=np.float64)
        model.body_quat = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        model.jnt_type = np.array([0], dtype=np.int32)
        model.jnt_bodyid = np.array([1], dtype=np.int32)
        model.jnt_qposadr = np.array([0], dtype=np.int32)
        model.jnt_pos = np.zeros((1, 3), dtype=np.float64)
        model.jnt_axis = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        model.mesh_vertadr = np.zeros(0, dtype=np.int32)
        model.mesh_vertnum = np.zeros(0, dtype=np.int32)
        model.mesh_vert = np.zeros((0, 3), dtype=np.float64)

    canonical.names = ["floor", *[f"robot_collision_{index}" for index in range(37)]]
    augmented.names = [
        "floor",
        *TABLE_NET.OBSTACLE_NAMES,
        *canonical.names[1:],
    ]
    canonical.geom_type = np.array([0, *([6] * 37)], dtype=np.int32)
    canonical.geom_bodyid = np.array([0, *([1] * 37)], dtype=np.int32)
    canonical.geom_contype = np.array([1, *([1] * 37)], dtype=np.int32)
    canonical.geom_conaffinity = np.array([1, *([7] * 37)], dtype=np.int32)
    canonical.geom_dataid = np.full(38, -1, dtype=np.int32)
    canonical.geom_size = np.arange(38 * 3, dtype=np.float64).reshape(38, 3) / 1000.0
    canonical.geom_pos = np.arange(38 * 3, dtype=np.float64).reshape(38, 3) / 100.0
    canonical.geom_quat = np.tile([1.0, 0.0, 0.0, 0.0], (38, 1))
    for label in TABLE_NET.COLLISION_GEOM_ARRAYS:
        source = np.asarray(getattr(canonical, label))
        shape = (42, *source.shape[1:])
        value = np.zeros(shape, dtype=source.dtype)
        value[0] = source[0]
        value[5:] = source[1:]
        setattr(augmented, label, value)
    augmented.geom_type[1:5] = 6
    augmented.geom_bodyid[1:5] = 0
    augmented.geom_contype[1:5] = 0
    augmented.geom_conaffinity[1:5] = 0
    augmented.geom_dataid[1:5] = -1

    fields = {
        "root_joint_id": 0,
        "root_body_id": 1,
        "root_qpos_address": 0,
        "joint_ids": (0,),
        "joint_qpos_addresses": (0,),
        "ground_geom_id": 0,
    }
    canonical_binding = types.SimpleNamespace(
        model=canonical,
        collision_geom_ids=tuple(range(1, 38)),
        **fields,
    )
    augmented_binding = types.SimpleNamespace(
        model=augmented,
        collision_geom_ids=tuple(range(5, 42)),
        **fields,
    )

    class _FakeMujoco:
        class mjtObj:
            mjOBJ_GEOM = object()

        @staticmethod
        def mj_id2name(model, _kind, index):
            return model.names[index]

    collision_sha = GROUND._compiled_collision_contract_sha256(
        canonical, canonical_binding.collision_geom_ids
    )
    return (
        _FakeMujoco(),
        canonical_binding,
        augmented_binding,
        {name: index for index, name in enumerate(TABLE_NET.OBSTACLE_NAMES, start=1)},
        collision_sha,
    )


def test_worldbody_first_geom_index_shift_preserves_exact_robot_contract():
    mujoco, canonical, augmented, obstacles, expected_sha = _fake_compiled_models()
    evidence = TABLE_NET.validate_augmented_robot_identity(
        mujoco,
        GROUND,
        canonical,
        augmented,
        obstacles,
        expected_collision_sha256=expected_sha,
    )
    assert evidence["robot_geom_index_shift"] == 4
    assert evidence["augmented_robot_collision_geom_ids"] == list(range(5, 42))
    assert evidence["normalized_robot_collision_contract_sha256"] == expected_sha


def test_worldbody_index_normalization_rejects_robot_collision_row_drift():
    mujoco, canonical, augmented, obstacles, expected_sha = _fake_compiled_models()
    augmented.model.geom_size[9, 0] += 1.0e-9
    with pytest.raises(TABLE_NET.TableNetError, match="geom_size rows"):
        TABLE_NET.validate_augmented_robot_identity(
            mujoco,
            GROUND,
            canonical,
            augmented,
            obstacles,
            expected_collision_sha256=expected_sha,
        )


def test_worldbody_index_normalization_rejects_unexpected_obstacle_ids():
    mujoco, canonical, augmented, obstacles, expected_sha = _fake_compiled_models()
    obstacles = dict(obstacles)
    obstacles["motion_net"] = 4
    with pytest.raises(TABLE_NET.TableNetError, match="obstacle IDs 1..4"):
        TABLE_NET.validate_augmented_robot_identity(
            mujoco,
            GROUND,
            canonical,
            augmented,
            obstacles,
            expected_collision_sha256=expected_sha,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("topology_count", "topology counts"),
        ("qpos0", "qpos0"),
        ("joint_binding", "root/joint topology"),
        ("robot_name_order", "relative order/name"),
    ],
)
def test_worldbody_index_normalization_never_weakens_robot_identity_gates(
    mutation, match
):
    mujoco, canonical, augmented, obstacles, expected_sha = _fake_compiled_models()
    if mutation == "topology_count":
        augmented.model.nq += 1
    elif mutation == "qpos0":
        augmented.model.qpos0[0] += 1.0e-12
    elif mutation == "joint_binding":
        augmented.joint_ids = (1,)
    else:
        augmented.model.names[8], augmented.model.names[9] = (
            augmented.model.names[9],
            augmented.model.names[8],
        )
    with pytest.raises(TABLE_NET.TableNetError, match=match):
        TABLE_NET.validate_augmented_robot_identity(
            mujoco,
            GROUND,
            canonical,
            augmented,
            obstacles,
            expected_collision_sha256=expected_sha,
        )


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

    def mj_geomDistance(self, model, data, a: int, b: int, threshold: float, _fromto):
        distance = self._distance(a, b)
        if distance < 0.0:
            return distance
        return min(distance, threshold)

    def geom_clearance(self, model, data, a: int, b: int, *, distmax: float, tol: float):
        assert distmax == 0.1
        assert tol == 1.0e-6
        return self._distance(a, b), False


@pytest.mark.parametrize(
    ("distance_m", "expected_hard"),
    [(0.00499, True), (0.00500, False), (0.00501, False)],
)
def test_five_mm_boundary_is_exact_fail_closed_threshold(distance_m, expected_hard):
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


@pytest.mark.parametrize("threshold_m", [0.005, 0.02])
@pytest.mark.parametrize("probe", ["nextafter_below", "half_epsilon_below"])
def test_hard_and_warning_thresholds_do_not_inherit_reporting_epsilon(threshold_m, probe):
    if probe == "nextafter_below":
        distance_m = math.nextafter(threshold_m, -math.inf)
    else:
        distance_m = threshold_m - 0.5 * TABLE_NET.DISTANCE_SATURATION_EPSILON_M
    assert distance_m < threshold_m
    helper = _DistanceHelper({(1, 10): distance_m})
    assert TABLE_NET.geom_meets_clearance_threshold(
        helper, object(), object(), 1, 10, threshold_m
    ) is False
    assert TABLE_NET.geom_reporting_bound_is_saturated(
        helper, object(), object(), 1, 10, threshold_m
    ) is True
    result = TABLE_NET.evaluate_robot_obstacle_pairs(
        helper,
        object(),
        object(),
        robot_ids=(1,),
        racket_ids=(1,),
        obstacle_ids={"motion_net": 10},
        hard_threshold_m=0.005,
        warning_threshold_m=0.02,
        reporting_tolerance_m=1.0e-6,
        geom_name=lambda geom_id: f"geom{geom_id}",
    )
    assert result["hard_failure"] is (threshold_m == 0.005)
    assert result["racket_or_handle_hard_failure"] is (threshold_m == 0.005)
    assert result["warning"] is True


def test_twenty_mm_boundary_does_not_warn():
    helper = _DistanceHelper({(1, 10): 0.02})
    result = TABLE_NET.evaluate_robot_obstacle_pairs(
        helper,
        object(),
        object(),
        robot_ids=(1,),
        racket_ids=(1,),
        obstacle_ids={"motion_net": 10},
        hard_threshold_m=0.005,
        warning_threshold_m=0.02,
        reporting_tolerance_m=1.0e-6,
        geom_name=lambda geom_id: f"geom{geom_id}",
    )
    assert result["hard_failure"] is False
    assert result["warning"] is False


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


def test_clearance_reports_midpoint_and_actual_certified_lower_bracket():
    true_distance = 0.0371234
    helper = _DistanceHelper({(1, 10): true_distance})
    result = TABLE_NET.evaluate_robot_obstacle_pairs(
        helper,
        object(),
        object(),
        robot_ids=(1,),
        racket_ids=(1,),
        obstacle_ids={"motion_net": 10},
        hard_threshold_m=0.005,
        warning_threshold_m=0.02,
        reporting_tolerance_m=1.0e-6,
        reporting_cap_m=0.1,
        geom_name=lambda geom_id: f"geom{geom_id}",
    )
    midpoint = result["minimum_clearance_midpoint_estimate_m"]
    lower = result["minimum_clearance_lower_bound_m"]
    assert lower <= true_distance < lower + 1.0e-6
    assert abs(midpoint - true_distance) <= 0.5e-6
    assert midpoint != lower
    assert result["minimum_lower_bound_pair"] == ["geom1", "motion_net"]
    assert result["minimum_lower_bound_is_reporting_cap_saturation"] is False


def test_reporting_cap_minus_epsilon_boundary_is_the_saturated_certified_bound():
    cap = 0.1
    boundary = cap - TABLE_NET.DISTANCE_SATURATION_EPSILON_M
    helper = _DistanceHelper({(1, 10): boundary})
    result = TABLE_NET.evaluate_robot_obstacle_pairs(
        helper,
        object(),
        object(),
        robot_ids=(1,),
        racket_ids=(1,),
        obstacle_ids={"motion_net": 10},
        hard_threshold_m=0.005,
        warning_threshold_m=0.02,
        reporting_tolerance_m=1.0e-6,
        reporting_cap_m=cap,
        geom_name=lambda geom_id: f"geom{geom_id}",
    )
    assert result["hard_failure"] is False
    assert result["warning"] is False
    assert result["minimum_clearance_midpoint_estimate_m"] is None
    assert result["minimum_midpoint_pair"] is None
    assert result["minimum_clearance_lower_bound_m"] == boundary
    assert result["minimum_lower_bound_pair"] is None
    assert result["minimum_lower_bound_is_reporting_cap_saturation"] is True


@pytest.mark.parametrize("true_distance", [-0.002, 0.0371234, 0.2])
def test_local_distance_kernel_matches_exact_upstream_reference(true_distance):
    helper = _DistanceHelper({(1, 10): true_distance})
    source = ast.parse(
        (ROOT / "hope_training/whole_body_tracking/scripts/audit_self_collision.py").read_text(
            encoding="utf-8"
        )
    )
    selected = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_far", "geom_clearance"}
    ]
    namespace = {
        "mujoco": helper,
        "CLEARANCE_DISTMAX": 0.6,
        "CLEARANCE_TOL": 1.0e-4,
        "Tuple": tuple,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<exact-distance-reference>", "exec"), namespace)
    upstream, saturated = namespace["geom_clearance"](
        object(), object(), 1, 10, distmax=0.1, tol=1.0e-6
    )
    midpoint, lower, upper, local_saturated = TABLE_NET.geom_clearance_bracket(
        helper, object(), object(), 1, 10, distmax=0.1, tol=1.0e-6
    )
    assert midpoint == upstream
    assert local_saturated is saturated
    if upper is not None:
        assert lower <= true_distance <= upper
    else:
        assert lower == pytest.approx(0.1 - TABLE_NET.DISTANCE_SATURATION_EPSILON_M)


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


def _bind_synthetic_certificate(tmp_path: Path, plan: dict, cert: dict) -> Path:
    path = tmp_path / "vendor_l1_certificate.json"
    path.write_text(json.dumps(cert, sort_keys=True), encoding="utf-8")
    plan["frozen_vendor_l1"]["certificate"] = {"path": str(path), "sha256": _sha(path)}
    return path


def test_l1_certificate_must_authorize_table_net(tmp_path):
    plan = _plan()
    cert = _synthetic_l1_certificate(plan)
    _bind_synthetic_certificate(tmp_path, plan, cert)
    binding, snapshot = TABLE_NET.validate_vendor_l1_certificate(plan)
    assert snapshot.sha256 == plan["frozen_vendor_l1"]["certificate"]["sha256"]
    assert binding["authorization"]["table_net_authorized"] is True
    cert["authorization"]["table_net_authorized"] = False
    _bind_synthetic_certificate(tmp_path, plan, cert)
    with pytest.raises(TABLE_NET.TableNetError, match="authorization changed"):
        TABLE_NET.validate_vendor_l1_certificate(plan)


class _ModuleScopedOs:
    """A stand-in ``os`` that only one module under test can see.

    人话:有些测试要在"读文件"这一步插一手才能演出攻击。但绝不能把整个进程的
    ``os.read`` 换掉 —— 同一个 pytest 进程里别的测试、别的库读文件时也会撞进这个
    钩子,结果就会随执行顺序变(那正是本文件旧版本 flake 的病根之一)。这里只替换
    被测脚本自己模块命名空间里的 ``os`` 这个名字,别人看到的仍然是真 ``os``。
    """

    def __init__(self, real_os, **overrides) -> None:
        self.__dict__.update(overrides)
        self.__dict__["_real_os"] = real_os

    def __getattr__(self, name: str):  # only for names we did not override
        return getattr(self.__dict__["_real_os"], name)


def test_l1_certificate_path_swap_cannot_change_consumed_bytes(monkeypatch, tmp_path):
    """证书已经 open 之后再换路径下的文件,消费到的字节必须还是原来那个 inode 的。

    人话:验证器打开证书拿到 fd 之后、读第一个字节之前,把该路径改名藏起来,再在同名
    路径上摆一份被改过的证书。因为读走的是 fd 不是路径,消费到的字节、算出的 SHA、
    解析出的证书内容都必须是原始那一份。

    这条判定不依赖任何时间戳:交换发生在 fstat 基线之前,所以"元数据漂移"告警本来就
    不该响;唯一可判定的就是"消费到的字节"本身,而那正是这条护栏的名字。
    (旧版本断言的是 rename 顺带改了 st_ctime_ns 从而触发漂移告警 —— 但本机 /tmp 的
    内核文件时间戳只有 1 ms 刻度,而 write→rename 整个窗口只有约 43 µs,所以 200 次里
    有 196 次时间戳根本没变、告警不响,测试随机红。见 exp §9.2.13。)
    """

    plan = _plan()
    original = _synthetic_l1_certificate(plan)
    path = _bind_synthetic_certificate(tmp_path, plan, original)
    original_bytes = path.read_bytes()
    original_sha = plan["frozen_vendor_l1"]["certificate"]["sha256"]
    forged = json.loads(json.dumps(original))
    forged["runtime"] = {"forged_after_sha_check": True}
    forged_bytes = json.dumps(forged, sort_keys=True).encode("utf-8")
    assert forged_bytes != original_bytes
    real_open_fd_snapshot = TABLE_NET._read_open_fd_snapshot
    swapped: list[Path] = []

    def swap_path_then_read(fd: int, opened: Path, label: str, **kwargs):
        if not swapped and Path(opened) == path:
            path.rename(tmp_path / "held-original.json")
            path.write_bytes(forged_bytes)
            swapped.append(path)
        return real_open_fd_snapshot(fd, opened, label, **kwargs)

    monkeypatch.setattr(TABLE_NET, "_read_open_fd_snapshot", swap_path_then_read)
    cert, snapshot = TABLE_NET.validate_vendor_l1_certificate(plan)
    # 攻击确实落地了(否则这条测试是空的)
    assert swapped == [path]
    assert path.read_bytes() == forged_bytes
    # 但消费到的字节没有跟着路径走
    assert snapshot.data == original_bytes
    assert snapshot.sha256 == original_sha
    assert snapshot.size == len(original_bytes)
    assert snapshot.inode != path.stat().st_ino
    assert cert["runtime"] == {"synthetic": True}
    assert "forged_after_sha_check" not in json.dumps(cert, sort_keys=True)


def test_l1_certificate_inode_mutated_mid_read_fails_closed(monkeypatch, tmp_path):
    """读到一半有人往同一个 inode 上追加字节,必须当场失败,不许把变长的内容收下。

    人话:这条守的是另一半 —— 攻击者不换路径,直接改同一个文件。判定用"文件大小变了",
    大小是整数不是时间戳,所以是确定的,不会像旧版本那样被内核时间戳的 1 ms 刻度赌输赢。
    拦截只装在被测模块自己的 ``os`` 上(见 ``_ModuleScopedOs``),不碰全局 ``os.read``。
    """

    plan = _plan()
    original = _synthetic_l1_certificate(plan)
    path = _bind_synthetic_certificate(tmp_path, plan, original)
    original_inode = path.stat().st_ino
    original_size = path.stat().st_size
    real_read = os.read
    appended: list[int] = []

    def append_then_read(fd: int, count: int) -> bytes:
        if not appended and os.fstat(fd).st_ino == original_inode:
            with open(path, "ab") as handle:
                handle.write(b" ")
            appended.append(fd)
        return real_read(fd, count)

    monkeypatch.setattr(TABLE_NET, "os", _ModuleScopedOs(os, read=append_then_read))
    with pytest.raises(TABLE_NET.TableNetError, match="changed during immutable read"):
        TABLE_NET.validate_vendor_l1_certificate(plan)
    assert appended, "拦截没有生效,这条测试是空的"
    assert os.read is real_read, "全局 os.read 被换掉了 —— 会污染同进程的其它测试"
    assert path.stat().st_ino == original_inode
    assert path.stat().st_size == original_size + 1


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


def _valid_npz_bytes(body_names: tuple[str, ...], mutation: str | None = None) -> bytes:
    joint_pos = np.zeros((151, 31), dtype=np.float32)
    body_pos = np.zeros((151, 32, 3), dtype=np.float32)
    body_quat = np.zeros((151, 32, 4), dtype=np.float32)
    body_quat[..., 3] = 1.0
    if mutation == "nonfinite":
        joint_pos[0, 0] = np.nan
    if mutation == "wrong_dtype":
        joint_pos = joint_pos.astype(np.float64)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        fps=np.array([50], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=np.gradient(joint_pos, 1.0 / 50.0, axis=0).astype(np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((151, 32, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((151, 32, 3), dtype=np.float32),
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=np.asarray(body_names),
    )
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [("nonfinite", "NaN/Inf"), ("wrong_dtype", "shape/dtype")],
)
def test_npz_snapshot_rejects_nonfinite_or_wrong_dtype(tmp_path, mutation, match):
    body_names = tuple(f"body_{index}" for index in range(32))
    path = tmp_path / "motion.npz"
    path.write_bytes(_valid_npz_bytes(body_names, mutation))
    snapshot = TABLE_NET.read_file_snapshot(path, "synthetic NPZ")
    plan = {"l0_contract": {"fps": 50, "quaternion_norm_tolerance": 1.0e-5}}
    with pytest.raises(TABLE_NET.TableNetError, match=match):
        TABLE_NET.load_schema2_npz_snapshot(snapshot, plan, body_names)


def test_npz_snapshot_rejects_duplicate_zip_members(tmp_path):
    body_names = tuple(f"body_{index}" for index in range(32))
    path = tmp_path / "motion.npz"
    path.write_bytes(_valid_npz_bytes(body_names))
    with zipfile.ZipFile(path, "r") as archive:
        duplicate = archive.read("fps.npy")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("fps.npy", duplicate)
    snapshot = TABLE_NET.read_file_snapshot(path, "duplicate NPZ")
    plan = {"l0_contract": {"fps": 50, "quaternion_norm_tolerance": 1.0e-5}}
    with pytest.raises(TABLE_NET.TableNetError, match="duplicated"):
        TABLE_NET.load_schema2_npz_snapshot(snapshot, plan, body_names)


def test_bound_directory_fd_ignores_replacement_path(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "mesh.stl").write_bytes(b"original-mesh")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    root_fd = os.open(root, flags)
    try:
        held = tmp_path / "held-model"
        root.rename(held)
        root.mkdir()
        (root / "mesh.stl").write_bytes(b"forged-mesh")
        snapshot = TABLE_NET._read_relative_snapshot(
            root_fd,
            root,
            Path("mesh.stl"),
            "synthetic mesh",
            expected_sha256=hashlib.sha256(b"original-mesh").hexdigest(),
        )
    finally:
        os.close(root_fd)
    assert snapshot.data == b"original-mesh"
    assert (root / "mesh.stl").read_bytes() == b"forged-mesh"


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
    target = TABLE_NET.validate_output_preconditions(
        {"output_contract": {"certificate_path": str(output)}}
    )
    try:
        published = TABLE_NET.write_exclusive(target, {"status": "first"})
        original = output.read_bytes()
        assert published.data == original
        with pytest.raises(TABLE_NET.TableNetError, match="already exists"):
            TABLE_NET.write_exclusive(target, {"status": "second"})
        assert output.read_bytes() == original
    finally:
        target.close()


def test_output_parent_swap_fails_before_publication(tmp_path):
    parent = tmp_path / "bound-parent"
    parent.mkdir()
    output = parent / "certificate.json"
    target = TABLE_NET.validate_output_preconditions(
        {"output_contract": {"certificate_path": str(output)}}
    )
    held_parent = tmp_path / "held-parent"
    parent.rename(held_parent)
    parent.mkdir()
    try:
        with pytest.raises(TABLE_NET.TableNetError, match="bound directory inode"):
            TABLE_NET.write_exclusive(target, {"status": "must-not-publish"})
    finally:
        target.close()
    assert not (held_parent / "certificate.json").exists()
    assert not output.exists()
