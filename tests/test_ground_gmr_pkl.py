from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ground_gmr_pkl.py"
SPEC = importlib.util.spec_from_file_location("ground_gmr_pkl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GROUND = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GROUND
SPEC.loader.exec_module(GROUND)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(frames: int = 3, dtype=np.float64) -> dict:
    root_pos = np.zeros((frames, 3), dtype=dtype)
    root_pos[:, 0] = np.linspace(0.0, 0.02, frames)
    root_pos[:, 2] = np.linspace(-0.05, 0.0, frames)
    return {
        "fps": 30.0,
        "root_pos": root_pos,
        "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=dtype), (frames, 1)),
        "dof_pos": np.zeros((frames, 31), dtype=dtype),
        "opaque_metadata": {"keep": [1, 2, 3]},
    }


class _Obj:
    mjOBJ_JOINT = 0
    mjOBJ_BODY = 1
    mjOBJ_GEOM = 2


class _Joint:
    mjJNT_FREE = 0
    mjJNT_HINGE = 3


class _Geom:
    mjGEOM_PLANE = 0
    mjGEOM_HFIELD = 1
    mjGEOM_SPHERE = 2
    mjGEOM_CAPSULE = 3
    mjGEOM_ELLIPSOID = 4
    mjGEOM_CYLINDER = 5
    mjGEOM_BOX = 6
    mjGEOM_MESH = 7


class _FakeModel:
    def __init__(self) -> None:
        self.joint_names = ["pelvis_free_joint", *GROUND.A3_GMR_JOINT_NAMES]
        self.body_names = ["world", "pelvis_link"]
        self.geom_names = ["floor", "foot_box"]
        self.njnt = len(self.joint_names)
        self.nbody = len(self.body_names)
        self.ngeom = len(self.geom_names)
        self.nq = 7 + len(GROUND.A3_GMR_JOINT_NAMES)

        self.body_parentid = np.array([0, 0], dtype=np.int32)
        self.body_pos = np.zeros((2, 3), dtype=np.float64)
        self.body_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        self.jnt_type = np.array([_Joint.mjJNT_FREE] + [_Joint.mjJNT_HINGE] * 31)
        self.jnt_bodyid = np.array([1] * 32, dtype=np.int32)
        self.jnt_qposadr = np.array([0] + list(range(7, 38)), dtype=np.int32)
        self.jnt_pos = np.zeros((32, 3), dtype=np.float64)
        self.jnt_axis = np.tile(np.array([0.0, 0.0, 1.0]), (32, 1))
        self.jnt_limited = np.array([False] + [True] * 31)
        self.jnt_range = np.tile(np.array([-2.0, 2.0]), (32, 1))
        self.qpos0 = np.zeros(self.nq, dtype=np.float64)
        self.qpos0[3] = 1.0

        self.geom_type = np.array([_Geom.mjGEOM_PLANE, _Geom.mjGEOM_BOX], dtype=np.int32)
        self.geom_bodyid = np.array([0, 1], dtype=np.int32)
        self.geom_contype = np.array([1, 1], dtype=np.int32)
        self.geom_conaffinity = np.array([1, 1], dtype=np.int32)
        self.geom_dataid = np.array([-1, -1], dtype=np.int32)
        self.geom_size = np.array([[1.0, 1.0, 0.0], [0.1, 0.1, 0.1]])
        self.geom_pos = np.zeros((2, 3), dtype=np.float64)
        self.geom_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))

        self.mesh_vertadr = np.zeros(0, dtype=np.int32)
        self.mesh_vertnum = np.zeros(0, dtype=np.int32)
        self.mesh_vert = np.zeros((0, 3), dtype=np.float64)


class _FakeData:
    def __init__(self, model: _FakeModel) -> None:
        self.qpos = model.qpos0.copy()
        self.geom_xpos = np.zeros((model.ngeom, 3), dtype=np.float64)
        self.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (model.ngeom, 1))


class FakeMujoco:
    mjtObj = _Obj
    mjtJoint = _Joint
    mjtGeom = _Geom
    model = _FakeModel()

    class MjModel:
        @staticmethod
        def from_xml_path(_path: str):
            return FakeMujoco.model

    MjData = _FakeData

    @staticmethod
    def mj_name2id(model, obj_type, name):
        names = {
            _Obj.mjOBJ_JOINT: model.joint_names,
            _Obj.mjOBJ_BODY: model.body_names,
            _Obj.mjOBJ_GEOM: model.geom_names,
        }[obj_type]
        try:
            return names.index(name)
        except ValueError:
            return -1

    @staticmethod
    def mj_id2name(model, obj_type, obj_id):
        names = {
            _Obj.mjOBJ_JOINT: model.joint_names,
            _Obj.mjOBJ_BODY: model.body_names,
            _Obj.mjOBJ_GEOM: model.geom_names,
        }[obj_type]
        return names[obj_id]

    @staticmethod
    def mj_forward(model, data):
        data.geom_xpos[0] = 0.0
        data.geom_xmat[0] = np.eye(3).reshape(-1)
        data.geom_xpos[1] = data.qpos[:3]
        data.geom_xmat[1] = np.eye(3).reshape(-1)


@pytest.fixture(autouse=True)
def _fresh_fake_model():
    FakeMujoco.model = _FakeModel()


def _args(tmp_path: Path, input_path: Path, mjcf_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "input": str(input_path),
        "expected_input_sha256": _sha(input_path),
        "output": str(tmp_path / "grounded.pkl"),
        "report": str(tmp_path / "grounded.report.json"),
        "mjcf": str(mjcf_path),
        "expected_mjcf_sha256": _sha(mjcf_path),
        "ground_geom": "floor",
        "expected_frames": 3,
        "expected_fps": 30.0,
        "target_clearance_m": 1e-5,
        "max_grounded_clearance_m": 1e-3,
        "numerical_tolerance_m": 1e-7,
        "max_abs_shift_m": 0.25,
        "quaternion_norm_tolerance": 1e-6,
        "joint_range_tolerance_rad": 1e-5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_fixture(tmp_path: Path, payload: dict | None = None) -> tuple[Path, Path]:
    input_path = tmp_path / "raw.pkl"
    with input_path.open("wb") as handle:
        pickle.dump(payload if payload is not None else _payload(), handle)
    mjcf_path = tmp_path / "canonical.xml"
    mjcf_path.write_text("<mujoco model='bound-by-fake-test'/>", encoding="utf-8")
    return input_path, mjcf_path


def test_validate_payload_checks_shape_finite_quaternion_and_optional_joint_order():
    payload = _payload()
    result = GROUND.validate_payload(
        payload, expected_frames=3, expected_fps=30.0, quaternion_norm_tolerance=1e-6
    )
    assert result["frames"] == 3
    assert not result["joint_names_present_in_input"]

    payload["joint_names"] = np.asarray(GROUND.A3_GMR_JOINT_NAMES)
    result = GROUND.validate_payload(
        payload, expected_frames=3, expected_fps=30.0, quaternion_norm_tolerance=1e-6
    )
    assert result["joint_names_present_in_input"]

    payload["joint_names"] = np.asarray(GROUND.A3_GMR_JOINT_NAMES[::-1])
    with pytest.raises(GROUND.GroundingError, match="joint_names"):
        GROUND.validate_payload(
            payload, expected_frames=3, expected_fps=30.0, quaternion_norm_tolerance=1e-6
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["root_pos"].__setitem__((1, 2), np.nan), "non-finite"),
        (lambda p: p["root_rot"].__setitem__((1, slice(None)), 0.0), "norm error"),
        (lambda p: p.__setitem__("dof_pos", np.zeros((3, 30))), "dof_pos shape"),
        (lambda p: p.__setitem__("fps", 50.0), "expected exactly"),
    ],
)
def test_validate_payload_fails_loud(mutation, message):
    payload = _payload()
    mutation(payload)
    with pytest.raises(GROUND.GroundingError, match=message):
        GROUND.validate_payload(
            payload, expected_frames=3, expected_fps=30.0, quaternion_norm_tolerance=1e-6
        )


def test_primitive_support_functions_use_world_orientation():
    model = _FakeModel()
    data = _FakeData(model)
    data.geom_xpos[1] = np.array([0.0, 0.0, 0.5])
    model.geom_type[1] = _Geom.mjGEOM_BOX
    model.geom_size[1] = np.array([0.1, 0.2, 0.3])
    # Local x points along world z, local z points along world x.
    data.geom_xmat[1] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).reshape(-1)
    assert GROUND.geom_world_min_z(FakeMujoco, model, data, 1) == pytest.approx(0.4)

    model.geom_type[1] = _Geom.mjGEOM_CAPSULE
    model.geom_size[1] = np.array([0.05, 0.4, 0.0])
    # Capsule axis local z is horizontal, so only its spherical radius lowers z.
    assert GROUND.geom_world_min_z(FakeMujoco, model, data, 1) == pytest.approx(0.45)


def test_mesh_support_uses_compiled_vertices_in_geom_frame():
    model = _FakeModel()
    data = _FakeData(model)
    model.geom_type[1] = _Geom.mjGEOM_MESH
    model.geom_dataid[1] = 0
    model.mesh_vertadr = np.array([0])
    model.mesh_vertnum = np.array([3])
    model.mesh_vert = np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.3], [1.0, 0.0, 0.0]])
    data.geom_xpos[1] = np.array([0.0, 0.0, 1.0])
    data.geom_xmat[1] = np.eye(3).reshape(-1)
    assert GROUND.geom_world_min_z(FakeMujoco, model, data, 1) == pytest.approx(0.8)


def test_single_file_grounding_writes_new_bound_output_and_preserves_motion(tmp_path: Path):
    original = _payload()
    input_path, mjcf_path = _write_fixture(tmp_path, original)
    original_bytes = input_path.read_bytes()
    args = _args(tmp_path, input_path, mjcf_path)

    report = GROUND.run_grounding(args, mujoco_module=FakeMujoco)

    output_path = Path(args.output)
    report_path = Path(args.report)
    assert input_path.read_bytes() == original_bytes
    assert output_path.is_file() and report_path.is_file()
    with output_path.open("rb") as handle:
        grounded = pickle.load(handle)
    shift = report["grounding"]["requested_constant_root_z_shift_m"]
    assert shift == pytest.approx(0.15001)
    np.testing.assert_array_equal(grounded["root_pos"][:, :2], original["root_pos"][:, :2])
    np.testing.assert_allclose(grounded["root_pos"][:, 2], original["root_pos"][:, 2] + shift)
    np.testing.assert_array_equal(grounded["root_rot"], original["root_rot"])
    np.testing.assert_array_equal(grounded["dof_pos"], original["dof_pos"])
    assert grounded["opaque_metadata"] == original["opaque_metadata"]

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report == report
    assert report["input"]["sha256"] == _sha(input_path)
    assert report["output"]["sha256"] == _sha(output_path)
    assert report["mjcf"]["sha256"] == _sha(mjcf_path)
    assert report["tool"]["sha256"] == _sha(SCRIPT)
    assert report["grounding"]["before"]["minimum_clearance_m"] == pytest.approx(-0.15)
    assert report["grounding"]["after"]["minimum_clearance_m"] == pytest.approx(1e-5)
    assert report["grounding"]["after"]["minimum_clearance_m"] <= 1e-3
    assert not report["formal_eligible"]
    assert any("continuous-time ground clearance is not proven" in item for item in report["limitations"])


def test_report_install_failure_rolls_back_only_this_invocations_output(
    tmp_path: Path, monkeypatch
):
    input_path, mjcf_path = _write_fixture(tmp_path)
    args = _args(tmp_path, input_path, mjcf_path)
    install = GROUND._install_new_file
    calls = 0

    def fail_second_install(temporary: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected report install failure")
        install(temporary, target)

    monkeypatch.setattr(GROUND, "_install_new_file", fail_second_install)
    with pytest.raises(OSError, match="injected report install failure"):
        GROUND.run_grounding(args, mujoco_module=FakeMujoco)
    assert calls == 2
    assert not Path(args.output).exists()
    assert not Path(args.report).exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_refuses_wrong_hashes_existing_output_and_in_place_paths(tmp_path: Path):
    input_path, mjcf_path = _write_fixture(tmp_path)
    args = _args(tmp_path, input_path, mjcf_path, expected_input_sha256="0" * 64)
    with pytest.raises(GROUND.GroundingError, match="input SHA mismatch"):
        GROUND.run_grounding(args, mujoco_module=FakeMujoco)

    args = _args(tmp_path, input_path, mjcf_path, output=str(input_path))
    with pytest.raises(GROUND.GroundingError, match="three distinct"):
        GROUND.run_grounding(args, mujoco_module=FakeMujoco)

    existing = tmp_path / "occupied.pkl"
    existing.write_bytes(b"do-not-touch")
    args = _args(tmp_path, input_path, mjcf_path, output=str(existing))
    with pytest.raises(GROUND.GroundingError, match="refusing to overwrite"):
        GROUND.run_grounding(args, mujoco_module=FakeMujoco)
    assert existing.read_bytes() == b"do-not-touch"

    symlink = tmp_path / "output-symlink.pkl"
    symlink.symlink_to(tmp_path / "missing-target.pkl")
    args = _args(tmp_path, input_path, mjcf_path, output=str(symlink))
    with pytest.raises(GROUND.GroundingError, match="refusing to overwrite"):
        GROUND.run_grounding(args, mujoco_module=FakeMujoco)
    assert symlink.is_symlink()


def test_model_binding_rejects_wrong_joint_order_and_unsupported_collision_geom(tmp_path: Path):
    _input_path, mjcf_path = _write_fixture(tmp_path)
    FakeMujoco.model.joint_names[1], FakeMujoco.model.joint_names[2] = (
        FakeMujoco.model.joint_names[2],
        FakeMujoco.model.joint_names[1],
    )
    with pytest.raises(GROUND.GroundingError, match="hinge order"):
        GROUND.bind_model(FakeMujoco, mjcf_path, ground_geom_name="floor")

    FakeMujoco.model = _FakeModel()
    FakeMujoco.model.geom_type[1] = _Geom.mjGEOM_HFIELD
    with pytest.raises(GROUND.GroundingError, match="unsupported type"):
        GROUND.bind_model(FakeMujoco, mjcf_path, ground_geom_name="floor")


def test_joint_limit_and_shift_guards_fail_before_writing(tmp_path: Path):
    payload = _payload()
    payload["dof_pos"][1, 5] = 2.1
    input_path, mjcf_path = _write_fixture(tmp_path, payload)
    args = _args(tmp_path, input_path, mjcf_path)
    with pytest.raises(GROUND.GroundingError, match="exceeds MJCF range"):
        GROUND.run_grounding(args, mujoco_module=FakeMujoco)
    assert not Path(args.output).exists()

    input_path.unlink()
    with input_path.open("wb") as handle:
        pickle.dump(_payload(), handle)
    args = _args(tmp_path, input_path, mjcf_path, max_abs_shift_m=0.1)
    with pytest.raises(GROUND.GroundingError, match="exceeds max_abs_shift"):
        GROUND.run_grounding(args, mujoco_module=FakeMujoco)
    assert not Path(args.output).exists()


def test_float32_grounding_remains_one_constant_shift_within_tolerance(tmp_path: Path):
    input_path, mjcf_path = _write_fixture(tmp_path, _payload(dtype=np.float32))
    args = _args(tmp_path, input_path, mjcf_path, numerical_tolerance_m=2e-7)
    report = GROUND.run_grounding(args, mujoco_module=FakeMujoco)
    assert report["grounding"]["applied_root_z_shift_spread_m"] <= 2e-7
    assert report["invariants"]["root_pos_dtype_preserved"]


def test_canonical_mjcf_binding_when_mujoco_is_available():
    mujoco = pytest.importorskip("mujoco", exc_type=ImportError)
    mjcf = (
        Path(__file__).resolve().parents[1]
        / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
    )
    binding = GROUND.bind_model(mujoco, mjcf, ground_geom_name="floor")
    assert len(binding.joint_ids) == 31
    assert len(binding.collision_geom_ids) > 0
    assert binding.ground_z_m == pytest.approx(0.0, abs=1e-12)
    assert len(binding.collision_contract_sha256) == 64


def test_canonical_mjcf_full_grounding_when_mujoco_is_available(tmp_path: Path):
    mujoco = pytest.importorskip("mujoco", exc_type=ImportError)
    mjcf = (
        Path(__file__).resolve().parents[1]
        / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    free = [
        jid
        for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    ]
    assert len(free) == 1
    root_address = int(model.jnt_qposadr[free[0]])
    root_wxyz = model.qpos0[root_address + 3 : root_address + 7]
    root_xyzw = root_wxyz[[1, 2, 3, 0]]
    dof = np.asarray(
        [
            model.qpos0[
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            ]
            for name in GROUND.A3_GMR_JOINT_NAMES
        ],
        dtype=np.float64,
    )
    payload = {
        "fps": 30.0,
        "root_pos": np.tile(model.qpos0[root_address : root_address + 3], (2, 1)),
        "root_rot": np.tile(root_xyzw, (2, 1)),
        "dof_pos": np.tile(dof, (2, 1)),
        "joint_names": list(GROUND.A3_GMR_JOINT_NAMES),
    }
    input_path = tmp_path / "canonical_raw.pkl"
    with input_path.open("wb") as handle:
        pickle.dump(payload, handle)
    args = _args(
        tmp_path,
        input_path,
        mjcf,
        expected_frames=2,
        output=str(tmp_path / "canonical_grounded.pkl"),
        report=str(tmp_path / "canonical_grounded.json"),
    )
    report = GROUND.run_grounding(args, mujoco_module=mujoco)
    assert report["grounding"]["after"]["minimum_clearance_m"] >= 1e-5 - 1e-7
    assert report["grounding"]["after"]["minimum_clearance_m"] <= 1e-3
    assert report["collision_contract"]["enabled_robot_geom_count"] > 0
