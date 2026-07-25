"""Pure-CPU contract tests for canonical_schema2_builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "canonical_schema2_builder.py"
)
_SPEC = importlib.util.spec_from_file_location("canonical_schema2_builder", _SCRIPT)
builder = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)


def _quat_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class _Enum:
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_SLIDE = 2
    mjJNT_HINGE = 3


class _Obj:
    mjOBJ_JOINT = 0
    mjOBJ_BODY = 1


class _FakeModel:
    def __init__(self, joint_names: tuple[str, ...], body_names: tuple[str, ...]):
        self.joint_names = ("root_free",) + joint_names
        self.body_names = body_names
        self.njnt = 32
        self.nbody = 32
        self.nq = 38
        self.nv = 37
        self.jnt_type = np.array([_Enum.mjJNT_FREE] + [_Enum.mjJNT_HINGE] * 31)
        self.jnt_qposadr = np.array([0] + list(range(7, 38)))
        self.jnt_dofadr = np.array([0] + list(range(6, 37)))
        self.jnt_bodyid = np.array([0] + [min(i + 1, 31) for i in range(31)])
        self.jnt_limited = np.array([False] + [True] * 31)
        self.jnt_range = np.tile(np.array([-3.0, 3.0]), (32, 1))


class _FakeData:
    def __init__(self, model: _FakeModel):
        self.qpos = np.zeros(model.nq)
        self.qvel = np.zeros(model.nv)
        self.xpos = np.zeros((model.nbody, 3))
        self.xquat = np.zeros((model.nbody, 4))
        self.xipos = np.zeros((model.nbody, 3))


class _FakeMujoco:
    mjtJoint = _Enum
    mjtObj = _Obj
    joint_names = builder.RUNTIME_JOINT_NAMES
    body_names: tuple[str, ...] = ()

    class MjModel:
        @staticmethod
        def from_xml_path(_path: str) -> _FakeModel:
            return _FakeModel(_FakeMujoco.joint_names, _FakeMujoco.body_names)

    MjData = _FakeData

    @staticmethod
    def mj_name2id(model: _FakeModel, kind: int, name: str) -> int:
        names = model.joint_names if kind == _Obj.mjOBJ_JOINT else model.body_names
        try:
            return names.index(name)
        except ValueError:
            return -1

    @staticmethod
    def _offset(body_id: int, *, com: bool) -> np.ndarray:
        return np.array([0.01 * body_id, 0.005 if com else 0.0, 0.002 * body_id])

    @staticmethod
    def mj_forward(model: _FakeModel, data: _FakeData) -> None:
        root_pos = data.qpos[:3]
        root_quat = data.qpos[3:7]
        rotation = _quat_to_matrix(root_quat)
        for body_id in range(model.nbody):
            data.xpos[body_id] = root_pos + rotation @ _FakeMujoco._offset(
                body_id, com=False
            )
            data.xipos[body_id] = root_pos + rotation @ _FakeMujoco._offset(
                body_id, com=True
            )
            data.xquat[body_id] = root_quat

    @staticmethod
    def _jacobian(
        model: _FakeModel,
        data: _FakeData,
        jacp: np.ndarray,
        jacr: np.ndarray,
        body_id: int,
        *,
        com: bool,
    ) -> None:
        rotation = _quat_to_matrix(data.qpos[3:7])
        point = rotation @ _FakeMujoco._offset(body_id, com=com)
        jacp[:] = 0.0
        jacr[:] = 0.0
        jacp[:, :3] = np.eye(3)
        jacp[:, 3:6] = -_skew(point)
        jacr[:, 3:6] = np.eye(3)

    @staticmethod
    def mj_jacBody(
        model: _FakeModel,
        data: _FakeData,
        jacp: np.ndarray,
        jacr: np.ndarray,
        body_id: int,
    ) -> None:
        _FakeMujoco._jacobian(model, data, jacp, jacr, body_id, com=False)

    @staticmethod
    def mj_jacBodyCom(
        model: _FakeModel,
        data: _FakeData,
        jacp: np.ndarray,
        jacr: np.ndarray,
        body_id: int,
    ) -> None:
        _FakeMujoco._jacobian(model, data, jacp, jacr, body_id, com=True)


def _fixture(tmp_path: Path, frames: int = 3):
    body_names = tuple(f"body_{index}" for index in range(32))
    _FakeMujoco.body_names = body_names
    body_order = tmp_path / "body_order.txt"
    body_order.write_text("\n".join(body_names) + "\n", encoding="utf-8")
    mjcf = tmp_path / "vendor.xml"
    mjcf.write_text("<mujoco model='fake'/>\n", encoding="utf-8")

    q = np.zeros((frames, 31))
    q[1] = np.linspace(-0.2, 0.2, 31)
    qd = np.zeros_like(q)
    qd[1] = np.linspace(-1.0, 1.0, 31)
    root_pos = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 1.1], [0.0, 0.0, 1.0]])
    root_quat = np.stack((_quat_z(0.0), _quat_z(0.4), _quat_z(0.0)))
    root_lin = np.zeros((frames, 3))
    root_ang = np.zeros((frames, 3))
    root_lin[1] = [1.0, -0.5, 0.2]
    root_ang[1] = [0.0, 0.0, 2.0]
    return body_order, mjcf, q, qd, root_pos, root_quat, root_lin, root_ang


def _build(tmp_path: Path, **overrides):
    fixture = _fixture(tmp_path)
    body_order, mjcf, q, qd, root_pos, root_quat, root_lin, root_ang = fixture
    kwargs = {
        "joint_pos": q,
        "joint_vel": qd,
        "root_pos_w": root_pos,
        "root_quat_wxyz": root_quat,
        "root_lin_vel_w": root_lin,
        "root_ang_vel_w": root_ang,
        "fps": 50,
        "mjcf_path": mjcf,
        "body_order_path": body_order,
        "input_sha256": "1" * 64,
        "ready_sha256": "2" * 64,
        "_mujoco_module": _FakeMujoco,
    }
    kwargs.update(overrides)
    return builder.build_schema2_candidate(**kwargs)


def test_builds_exact_11_field_fk_and_jacobian_velocity_candidate(tmp_path: Path):
    candidate = _build(tmp_path)
    assert frozenset(candidate.arrays) == builder.ALLOWED_KEYSETS[0]
    assert candidate.arrays["joint_pos"].shape == (3, 31)
    assert candidate.arrays["body_pos_w"].shape == (3, 32, 3)
    assert candidate.arrays["body_quat_w"].shape == (3, 32, 4)
    assert candidate.arrays["body_names"].tolist() == list(_FakeMujoco.body_names)
    assert np.count_nonzero(candidate.arrays["joint_vel"][[0, -1]]) == 0
    assert np.count_nonzero(candidate.arrays["body_lin_vel_w"][[0, -1]]) == 0
    assert np.count_nonzero(candidate.arrays["body_ang_vel_w"][[0, -1]]) == 0

    rotation = _quat_to_matrix(_quat_z(0.4))
    body = 10
    com_offset_w = rotation @ _FakeMujoco._offset(body, com=True)
    expected = np.array([1.0, -0.5, 0.2]) + np.cross(
        np.array([0.0, 0.0, 2.0]), com_offset_w
    )
    np.testing.assert_allclose(
        candidate.arrays["body_lin_vel_w"][1, body], expected, atol=2e-7
    )
    np.testing.assert_allclose(
        candidate.arrays["body_ang_vel_w"][1, body], [0.0, 0.0, 2.0]
    )
    assert candidate.manifest["publication_class"] == "compiler_candidate"
    assert candidate.manifest["training_authorized"] is False
    assert candidate.report["checks"]["six_velocity_channels_zero_first_last"] is True
    assert candidate.manifest["kinematics"]["pose_finite_difference_used"] is False
    assert hashlib.sha256(candidate.npz_bytes).hexdigest() == candidate.output_sha256
    with np.load(__import__("io").BytesIO(candidate.npz_bytes), allow_pickle=False) as data:
        assert frozenset(data.files) == builder.ALLOWED_KEYSETS[0]
        np.testing.assert_array_equal(data["joint_vel"], candidate.arrays["joint_vel"])


def test_optional_migration_tuple_produces_exact_14_fields(tmp_path: Path):
    migration = builder.MigrationProvenance(
        source_sha256="a" * 64,
        source_point="legacy_link_origin",
        tool="migration_tool_v1",
    )
    candidate = _build(tmp_path, migration_provenance=migration)
    assert frozenset(candidate.arrays) == builder.ALLOWED_KEYSETS[1]
    assert str(candidate.arrays["kinematics_migration_source_sha256"]) == "a" * 64


def test_write_is_no_clobber_and_hash_bound(tmp_path: Path):
    candidate = _build(tmp_path)
    output = tmp_path / "out" / "candidate.npz"
    paths = builder.write_schema2_candidate(candidate, output)
    assert all(path.is_file() for path in paths)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == candidate.output_sha256
    manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    report = json.loads(paths[2].read_text(encoding="utf-8"))
    for record in (manifest, report):
        assert record["hashes"]["output_npz_sha256"] == candidate.output_sha256
        assert record["hashes"]["input_sha256"] == "1" * 64
        assert record["hashes"]["ready_sha256"] == "2" * 64
        assert len(record["hashes"]["mjcf_sha256"]) == 64
        assert len(record["hashes"]["body_order_sha256"]) == 64
        assert len(record["hashes"]["tool_sha256"]) == 64
    before = [path.read_bytes() for path in paths]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.write_schema2_candidate(candidate, output)
    assert [path.read_bytes() for path in paths] == before


def test_write_treats_broken_symlink_as_occupied(tmp_path: Path):
    candidate = _build(tmp_path)
    output = tmp_path / "candidate.npz"
    output.symlink_to(tmp_path / "missing-target.npz")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.write_schema2_candidate(candidate, output)
    assert output.is_symlink()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("joint_vel", "first/last rows must be exactly zero"),
        ("root_lin_vel_w", "first/last rows must be exactly zero"),
        ("root_ang_vel_w", "first/last rows must be exactly zero"),
    ],
)
def test_nonzero_velocity_endpoint_fails_closed(
    tmp_path: Path, field: str, message: str
):
    body_order, mjcf, q, qd, root_pos, root_quat, root_lin, root_ang = _fixture(
        tmp_path
    )
    values = {
        "joint_vel": qd,
        "root_lin_vel_w": root_lin,
        "root_ang_vel_w": root_ang,
    }
    values[field] = values[field].copy()
    values[field][0, 0] = 1e-12
    with pytest.raises(builder.Schema2BuildError, match=message):
        builder.build_schema2_candidate(
            joint_pos=q,
            joint_vel=values["joint_vel"],
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            root_lin_vel_w=values["root_lin_vel_w"],
            root_ang_vel_w=values["root_ang_vel_w"],
            fps=50,
            mjcf_path=mjcf,
            body_order_path=body_order,
            input_sha256="1" * 64,
            ready_sha256="2" * 64,
            _mujoco_module=_FakeMujoco,
        )


def test_ready_pose_hash_and_model_contract_fail_closed(tmp_path: Path):
    body_order, mjcf, q, qd, root_pos, root_quat, root_lin, root_ang = _fixture(
        tmp_path
    )
    q[-1, 0] = 0.1
    with pytest.raises(builder.Schema2BuildError, match="same canonical-ready"):
        builder.build_schema2_candidate(
            joint_pos=q,
            joint_vel=qd,
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            root_lin_vel_w=root_lin,
            root_ang_vel_w=root_ang,
            fps=50,
            mjcf_path=mjcf,
            body_order_path=body_order,
            input_sha256="not-a-hash",
            ready_sha256="2" * 64,
            _mujoco_module=_FakeMujoco,
        )

    q[-1, 0] = q[0, 0]
    with pytest.raises(builder.Schema2BuildError, match="input_sha256"):
        builder.build_schema2_candidate(
            joint_pos=q,
            joint_vel=qd,
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            root_lin_vel_w=root_lin,
            root_ang_vel_w=root_ang,
            fps=50,
            mjcf_path=mjcf,
            body_order_path=body_order,
            input_sha256="not-a-hash",
            ready_sha256="2" * 64,
            _mujoco_module=_FakeMujoco,
        )

    _FakeMujoco.body_names = tuple(f"wrong_{i}" for i in range(32))
    with pytest.raises(builder.Schema2BuildError, match="absent from MJCF"):
        builder.build_schema2_candidate(
            joint_pos=q,
            joint_vel=qd,
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            root_lin_vel_w=root_lin,
            root_ang_vel_w=root_ang,
            fps=50,
            mjcf_path=mjcf,
            body_order_path=body_order,
            input_sha256="1" * 64,
            ready_sha256="2" * 64,
            _mujoco_module=_FakeMujoco,
        )
