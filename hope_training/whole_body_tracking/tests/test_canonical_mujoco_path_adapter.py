"""Tests for the canonical-path to exact A3 MuJoCo nq/nv adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SCRIPT = _SCRIPTS / "canonical_mujoco_path_adapter.py"
_SPEC = importlib.util.spec_from_file_location(
    "canonical_mujoco_path_adapter", _SCRIPT
)
adapter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = adapter
_SPEC.loader.exec_module(adapter)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(left, dtype=np.float64)
    bw, bx, by, bz = np.asarray(right, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def _quat_log(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector_norm = float(np.linalg.norm(quat[1:]))
    if vector_norm < 1.0e-14:
        return 2.0 * quat[1:]
    angle = 2.0 * np.arctan2(vector_norm, float(quat[0]))
    return quat[1:] * (angle / vector_norm)


def _qz(angle: float) -> np.ndarray:
    return np.array(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        dtype=np.float64,
    )


class _ObjectTypes:
    mjOBJ_BODY = 1
    mjOBJ_JOINT = 3
    mjOBJ_GEOM = 5
    mjOBJ_ACTUATOR = 19


class _JointTypes:
    mjJNT_FREE = 0
    mjJNT_HINGE = 3


class _FakeModel:
    def __init__(
        self,
        *,
        physical_joint_names: tuple[str, ...] | None = None,
        body_names: tuple[str, ...] = ("world", adapter.ROOT_BODY_NAME),
    ):
        if physical_joint_names is None:
            permutation = np.random.default_rng(20260724).permutation(
                adapter.RUNTIME_JOINT_NAMES
            )
            physical_joint_names = (
                adapter.ROOT_JOINT_NAME,
                *(str(name) for name in permutation),
            )
        self.nq = 38
        self.nv = 37
        self.njnt = 32
        self.nu = 31
        self.nbody = len(body_names)
        self.ngeom = 0
        self.opt = SimpleNamespace(
            gravity=np.array([0.0, 0.0, -9.81], dtype=np.float64),
            timestep=0.001,
        )
        self.jnt_type = np.full(32, _JointTypes.mjJNT_HINGE, dtype=np.int32)
        self.jnt_type[0] = _JointTypes.mjJNT_FREE
        self.jnt_bodyid = np.ones(32, dtype=np.int32)
        self.jnt_qposadr = np.concatenate(
            (np.array([0], dtype=np.int32), np.arange(7, 38, dtype=np.int32))
        )
        self.jnt_dofadr = np.concatenate(
            (np.array([0], dtype=np.int32), np.arange(6, 37, dtype=np.int32))
        )
        self._names = {
            _ObjectTypes.mjOBJ_BODY: tuple(body_names),
            _ObjectTypes.mjOBJ_JOINT: tuple(physical_joint_names),
            _ObjectTypes.mjOBJ_GEOM: (),
            _ObjectTypes.mjOBJ_ACTUATOR: tuple(
                f"motor_{index}" for index in range(31)
            ),
        }


class _FakeMjModelLoader:
    model: _FakeModel | None = None

    @classmethod
    def from_xml_path(cls, unused_path: str) -> _FakeModel:
        del unused_path
        assert cls.model is not None
        return cls.model


class _FakeMujoco:
    mjtObj = _ObjectTypes
    mjtJoint = _JointTypes
    MjModel = _FakeMjModelLoader

    @staticmethod
    def mj_id2name(model: _FakeModel, object_type: int, index: int):
        rows = model._names.get(object_type, ())
        if index < 0 or index >= len(rows):
            return None
        return rows[index]

    @staticmethod
    def mj_name2id(model: _FakeModel, object_type: int, name: str) -> int:
        rows = model._names.get(object_type, ())
        try:
            return rows.index(name)
        except ValueError:
            return -1

    @staticmethod
    def mj_differentiatePos(
        model: _FakeModel,
        output: np.ndarray,
        dt: float,
        qpos1: np.ndarray,
        qpos2: np.ndarray,
    ) -> None:
        output[:] = 0.0
        output[:3] = (qpos2[:3] - qpos1[:3]) / dt
        relative = _quat_multiply(_quat_conjugate(qpos1[3:7]), qpos2[3:7])
        output[3:6] = _quat_log(relative) / dt
        for joint in range(int(model.njnt)):
            if int(model.jnt_type[joint]) != _JointTypes.mjJNT_HINGE:
                continue
            qpos_address = int(model.jnt_qposadr[joint])
            dof_address = int(model.jnt_dofadr[joint])
            output[dof_address] = (
                qpos2[qpos_address] - qpos1[qpos_address]
            ) / dt


def _write_mjcf(tmp_path: Path, *, model_name: str | None = None) -> Path:
    name = model_name or adapter.EXPECTED_MJCF_MODEL_NAME
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "a3_pingpong.xml"
    path.write_text(f'<mujoco model="{name}"></mujoco>', encoding="utf-8")
    return path


def _bind(
    tmp_path: Path,
    *,
    model: _FakeModel | None = None,
    model_name: str | None = None,
) -> tuple[_FakeModel, Path, adapter.ExactMujocoModelBinding]:
    exact_model = _FakeModel() if model is None else model
    mjcf = _write_mjcf(tmp_path, model_name=model_name)
    source_sha = hashlib.sha256(mjcf.read_bytes()).hexdigest()
    compiled_sha = adapter.compiled_model_signature(exact_model, _FakeMujoco)
    binding = adapter.bind_exact_mujoco_model(
        _FakeMujoco,
        exact_model,
        mjcf_path=mjcf,
        expected_mjcf_sha256=source_sha,
        expected_compiled_model_sha256=compiled_sha,
        expected_xml_model_name=(
            adapter.EXPECTED_MJCF_MODEL_NAME
            if model_name is None
            else model_name
        ),
    )
    return exact_model, mjcf, binding


def _empty_upper(samples: int = 4):
    return (
        np.zeros((samples, 31), dtype=np.float64),
        np.zeros((samples, 31), dtype=np.float64),
        np.zeros((samples, 31), dtype=np.float64),
    )


def _empty_full(samples: int = 4):
    return (
        np.zeros((samples, 37), dtype=np.float64),
        np.zeros((samples, 37), dtype=np.float64),
        np.zeros((samples, 37), dtype=np.float64),
    )


def _adapt(
    model: _FakeModel,
    binding: adapter.ExactMujocoModelBinding,
    *,
    scope: str,
    q: np.ndarray,
    q_s: np.ndarray,
    q_ss: np.ndarray,
    coordinate_names=None,
    ready_pos=np.array([0.12, -0.03, 0.91]),
    ready_quat=_qz(0.37),
):
    return adapter.adapt_canonical_path_jet(
        _FakeMujoco,
        model,
        binding,
        scope=scope,
        coordinate_names=(
            adapter.UPPER_COORDINATE_NAMES
            if coordinate_names is None and scope == "upper"
            else adapter.FULL_COORDINATE_NAMES
            if coordinate_names is None
            else coordinate_names
        ),
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        canonical_ready_root_pos_w=ready_pos,
        canonical_ready_root_quat_wxyz=ready_quat,
    )


def test_upper_name_maps_scrambled_model_addresses_and_is_bitwise_deterministic(
    tmp_path: Path,
):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    assert tuple(model._names[_ObjectTypes.mjOBJ_JOINT][1:]) != (
        adapter.RUNTIME_JOINT_NAMES
    )
    q = np.arange(4 * 31, dtype=np.float64).reshape(4, 31) / 100.0
    q_s = q + 0.25
    q_ss = q - 0.75
    first = _adapt(
        model, binding, scope="upper", q=q, q_s=q_s, q_ss=q_ss
    )
    second = _adapt(
        model, binding, scope="upper", q=q, q_s=q_s, q_ss=q_ss
    )

    np.testing.assert_array_equal(first.qpos, second.qpos)
    np.testing.assert_array_equal(first.q_s, second.q_s)
    np.testing.assert_array_equal(first.q_ss, second.q_ss)
    assert dict(first.report) == dict(second.report)
    np.testing.assert_array_equal(first.qpos[:, binding.joint_qpos_adrs], q)
    np.testing.assert_array_equal(first.q_s[:, binding.joint_dof_adrs], q_s)
    np.testing.assert_array_equal(first.q_ss[:, binding.joint_dof_adrs], q_ss)
    np.testing.assert_array_equal(
        first.qpos[:, :3], np.broadcast_to([0.12, -0.03, 0.91], (4, 3))
    )
    np.testing.assert_array_equal(
        first.qpos[:, 3:7], np.broadcast_to(_qz(0.37), (4, 4))
    )
    assert np.count_nonzero(first.q_s[:, :6]) == 0
    assert np.count_nonzero(first.q_ss[:, :6]) == 0
    assert first.qpos.flags.writeable is False
    assert first.q_s.flags.writeable is False
    assert first.q_ss.flags.writeable is False
    assert first.report["finite_difference_generation_used"] is False
    assert first.report["complete_compiled_mjb_bound"] is False
    assert first.report["training_authorized"] is False
    assert first.report["deployment_authorized"] is False
    assert first.report["hardware_authorized"] is False


def test_full_nonzero_root_uses_current_body_tangent_and_exact_addresses(
    tmp_path: Path,
):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    samples = 7
    s = np.linspace(0.0, 1.0, samples)
    q, q_s, q_ss = _empty_full(samples)
    q[:, :31] = np.arange(31, dtype=np.float64)[None, :] / 50.0
    q_s[:, :31] = 0.1
    q_ss[:, :31] = -0.2
    q[:, 31:34] = np.column_stack(
        (0.1 + 0.02 * s, -0.03 * s * s, 0.91 + 0.01 * s)
    )
    q_s[:, 31:34] = np.column_stack(
        (np.full(samples, 0.02), -0.06 * s, np.full(samples, 0.01))
    )
    q_ss[:, 31:34] = np.column_stack(
        (np.zeros(samples), np.full(samples, -0.06), np.zeros(samples))
    )
    q[:, 34:] = np.column_stack(
        (0.2 * s, -0.12 * s * s, 0.08 * s + 0.03 * s * s)
    )
    q_s[:, 34:] = np.column_stack(
        (np.full(samples, 0.2), -0.24 * s, 0.08 + 0.06 * s)
    )
    q_ss[:, 34:] = np.column_stack(
        (np.zeros(samples), np.full(samples, -0.24), np.full(samples, 0.06))
    )
    ready_quat = _qz(-0.48)

    result = _adapt(
        model,
        binding,
        scope="full",
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        ready_quat=ready_quat,
    )
    root_pos, root_quat = adapter.decode_root_pose(
        q[:, 31:],
        canonical_ready_root_quat_wxyz=ready_quat,
    )
    omega_world, alpha_world = (
        adapter.rotation_vector_derivatives_to_world_angular_kinematics(
            q[:, 34:],
            q_s[:, 34:],
            q_ss[:, 34:],
            canonical_ready_root_quat_wxyz=ready_quat,
        )
    )
    rotation = adapter._rotation_matrix_wxyz(root_quat)
    expected_omega_body = np.einsum("nji,nj->ni", rotation, omega_world)
    expected_alpha_body = np.einsum("nji,nj->ni", rotation, alpha_world)

    np.testing.assert_allclose(result.qpos[:, :3], root_pos, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        result.qpos[:, 3:7], root_quat, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(result.q_s[:, :3], q_s[:, 31:34])
    np.testing.assert_allclose(result.q_ss[:, :3], q_ss[:, 31:34])
    np.testing.assert_allclose(result.q_s[:, 3:6], expected_omega_body)
    np.testing.assert_allclose(result.q_ss[:, 3:6], expected_alpha_body)
    np.testing.assert_array_equal(
        result.qpos[:, binding.joint_qpos_adrs], q[:, :31]
    )
    assert result.qpos.shape == (samples, 38)
    assert result.q_s.shape == (samples, 37)
    assert result.q_ss.shape == (samples, 37)


def test_near_zero_root_rotation_is_finite_and_not_special_cased_to_zero(
    tmp_path: Path,
):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    q, q_s, q_ss = _empty_full(3)
    q[:, 31:34] = [0.0, 0.0, 0.9]
    q[:, 34:] = np.array(
        [[0.0, 0.0, 0.0], [1.0e-14, -2.0e-14, 3.0e-14], [0.0, 0.0, 0.0]]
    )
    q_s[:, 34:] = [0.13, -0.17, 0.19]
    q_ss[:, 34:] = [-0.07, 0.11, 0.05]
    result = _adapt(
        model, binding, scope="full", q=q, q_s=q_s, q_ss=q_ss
    )

    assert np.isfinite(result.qpos).all()
    assert np.isfinite(result.q_s).all()
    assert np.isfinite(result.q_ss).all()
    assert np.linalg.norm(result.q_s[:, 3:6]) > 0.0
    assert np.linalg.norm(result.q_ss[:, 3:6]) > 0.0


def test_branch_cut_fails_closed_before_mujoco_arrays_are_returned(tmp_path: Path):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    q, q_s, q_ss = _empty_full(3)
    q[:, 31:34] = [0.0, 0.0, 0.9]
    q[1, 34] = np.pi - adapter.PI_MARGIN_RAD / 2.0
    with pytest.raises(adapter.MujocoPathAdapterError, match="branch cut"):
        _adapt(model, binding, scope="full", q=q, q_s=q_s, q_ss=q_ss)


@pytest.mark.parametrize(
    "bad_names, message",
    [
        (
            (
                adapter.RUNTIME_JOINT_NAMES[0],
                adapter.RUNTIME_JOINT_NAMES[0],
                *adapter.RUNTIME_JOINT_NAMES[2:],
            ),
            "duplicates",
        ),
        (
            (
                adapter.RUNTIME_JOINT_NAMES[1],
                adapter.RUNTIME_JOINT_NAMES[0],
                *adapter.RUNTIME_JOINT_NAMES[2:],
            ),
            "order drifted",
        ),
        (adapter.RUNTIME_JOINT_NAMES[:-1], "order drifted"),
    ],
)
def test_input_coordinate_duplicate_reorder_or_missing_fails_closed(
    tmp_path: Path, bad_names, message: str
):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    q, q_s, q_ss = _empty_upper()
    with pytest.raises(adapter.MujocoPathAdapterError, match=message):
        _adapt(
            model,
            binding,
            scope="upper",
            q=q,
            q_s=q_s,
            q_ss=q_ss,
            coordinate_names=bad_names,
        )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda q, qs, qss: (q[:, :-1], qs, qss), "identical sampled shapes"),
        (
            lambda q, qs, qss: (
                np.where(np.indices(q.shape)[0] == 0, np.nan, q),
                qs,
                qss,
            ),
            "NaN/Inf",
        ),
        (
            lambda q, qs, qss: (
                q.astype(bool),
                qs,
                qss,
            ),
            "real numeric",
        ),
    ],
)
def test_bad_dimensions_nonfinite_and_bool_fail_closed(
    tmp_path: Path, mutator, message: str
):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    q, q_s, q_ss = _empty_upper()
    bad_q, bad_q_s, bad_q_ss = mutator(q, q_s, q_ss)
    with pytest.raises(adapter.MujocoPathAdapterError, match=message):
        _adapt(
            model,
            binding,
            scope="upper",
            q=bad_q,
            q_s=bad_q_s,
            q_ss=bad_q_ss,
        )


def test_source_and_compiled_model_drift_are_rechecked_on_every_adaptation(
    tmp_path: Path,
):
    model, mjcf, binding = _bind(tmp_path)
    q, q_s, q_ss = _empty_upper()
    model.jnt_qposadr[1] = 8
    with pytest.raises(
        adapter.MujocoPathAdapterError, match="compiled model signature mismatch"
    ):
        _adapt(model, binding, scope="upper", q=q, q_s=q_s, q_ss=q_ss)

    model2, mjcf2, binding2 = _bind(tmp_path / "second")
    mjcf2.write_text(
        f'<mujoco model="{adapter.EXPECTED_MJCF_MODEL_NAME}"> </mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(adapter.MujocoPathAdapterError, match="MJCF SHA mismatch"):
        _adapt(model2, binding2, scope="upper", q=q, q_s=q_s, q_ss=q_ss)
    assert mjcf.is_file()


def test_wrong_hash_model_name_and_missing_runtime_joint_fail_binding(
    tmp_path: Path,
):
    model = _FakeModel()
    mjcf = _write_mjcf(tmp_path)
    compiled = adapter.compiled_model_signature(model, _FakeMujoco)
    with pytest.raises(adapter.MujocoPathAdapterError, match="MJCF SHA mismatch"):
        adapter.bind_exact_mujoco_model(
            _FakeMujoco,
            model,
            mjcf_path=mjcf,
            expected_mjcf_sha256="0" * 64,
            expected_compiled_model_sha256=compiled,
        )
    source = hashlib.sha256(mjcf.read_bytes()).hexdigest()
    with pytest.raises(
        adapter.MujocoPathAdapterError, match="compiled model signature mismatch"
    ):
        adapter.bind_exact_mujoco_model(
            _FakeMujoco,
            model,
            mjcf_path=mjcf,
            expected_mjcf_sha256=source,
            expected_compiled_model_sha256="1" * 64,
        )
    with pytest.raises(adapter.MujocoPathAdapterError, match="model name mismatch"):
        adapter.bind_exact_mujoco_model(
            _FakeMujoco,
            model,
            mjcf_path=mjcf,
            expected_mjcf_sha256=source,
            expected_compiled_model_sha256=compiled,
            expected_xml_model_name="wrong_name",
        )

    missing_names = list(model._names[_ObjectTypes.mjOBJ_JOINT])
    missing_names[-1] = "not_a_runtime_joint"
    missing_model = _FakeModel(physical_joint_names=tuple(missing_names))
    missing_compiled = adapter.compiled_model_signature(
        missing_model, _FakeMujoco
    )
    with pytest.raises(
        adapter.MujocoPathAdapterError, match="runtime joint"
    ):
        adapter.bind_exact_mujoco_model(
            _FakeMujoco,
            missing_model,
            mjcf_path=mjcf,
            expected_mjcf_sha256=source,
            expected_compiled_model_sha256=missing_compiled,
        )


def test_loader_uses_same_bound_model_contract(tmp_path: Path):
    model = _FakeModel()
    mjcf = _write_mjcf(tmp_path)
    source = hashlib.sha256(mjcf.read_bytes()).hexdigest()
    compiled = adapter.compiled_model_signature(model, _FakeMujoco)
    _FakeMjModelLoader.model = model
    loaded, binding = adapter.load_exact_mujoco_model(
        mjcf_path=mjcf,
        expected_mjcf_sha256=source,
        expected_compiled_model_sha256=compiled,
        mujoco_module=_FakeMujoco,
    )
    assert loaded is model
    assert binding.mjcf_sha256 == source
    assert binding.compiled_model_sha256 == compiled


def test_compiled_signature_matches_existing_dynamics_gate_receipt(tmp_path: Path):
    del tmp_path
    import canonical_mujoco_dynamics_gate as dynamics_gate

    model = _FakeModel()
    previous = dynamics_gate.mujoco
    try:
        dynamics_gate.mujoco = _FakeMujoco
        assert adapter.compiled_model_signature(
            model, _FakeMujoco
        ) == dynamics_gate.compiled_model_signature(model)
    finally:
        dynamics_gate.mujoco = previous


def test_mj_differentiate_pos_validation_is_independent_and_fail_closed(
    tmp_path: Path,
):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    s = np.linspace(0.0, 1.0, 21)
    q, q_s, q_ss = _empty_full(len(s))
    q[:, 0] = 0.1 * s * s
    q_s[:, 0] = 0.2 * s
    q_ss[:, 0] = 0.2
    q[:, 31:34] = np.column_stack(
        (0.02 * s * s, -0.03 * s * s, 0.9 + 0.01 * s * s)
    )
    q_s[:, 31:34] = np.column_stack(
        (0.04 * s, -0.06 * s, 0.02 * s)
    )
    q_ss[:, 31:34] = [0.04, -0.06, 0.02]
    q[:, 36] = 0.3 * s * s
    q_s[:, 36] = 0.6 * s
    q_ss[:, 36] = 0.6
    path = _adapt(
        model, binding, scope="full", q=q, q_s=q_s, q_ss=q_ss
    )

    report = adapter.validate_with_mj_differentiate_pos(
        _FakeMujoco,
        model,
        binding,
        path,
        path_parameter=s,
        tangent_absolute_tolerance=2.0e-12,
        tangent_relative_tolerance=2.0e-12,
        curvature_absolute_tolerance=2.0e-11,
        curvature_relative_tolerance=2.0e-11,
    )
    assert report["status"] == "PASS_SAMPLED_MUJOCO_DIFFERENTIATE_POS"
    assert report["finite_difference_used_for_generation"] is False
    assert report["training_authorized"] is False
    assert report["deployment_authorized"] is False
    assert report["hardware_authorized"] is False

    bad_q_s = np.array(path.q_s, copy=True)
    bad_q_s[10, 6] += 0.05
    corrupted = adapter.MujocoPathJet(
        qpos=path.qpos,
        q_s=bad_q_s,
        q_ss=path.q_ss,
        report=path.report,
    )
    with pytest.raises(
        adapter.MujocoPathAdapterError, match="disagree"
    ) as caught:
        adapter.validate_with_mj_differentiate_pos(
            _FakeMujoco,
            model,
            binding,
            corrupted,
            path_parameter=s,
            tangent_absolute_tolerance=2.0e-12,
            tangent_relative_tolerance=2.0e-12,
            curvature_absolute_tolerance=2.0e-11,
            curvature_relative_tolerance=2.0e-11,
        )
    assert caught.value.report["status"] == "INCOMPLETE_FAIL_CLOSED"


def test_noncommuting_root_curve_matches_current_body_mujoco_tangent(
    tmp_path: Path,
):
    model, unused_mjcf, binding = _bind(tmp_path)
    del unused_mjcf
    s = np.linspace(0.0, 1.0, 501)
    q, q_s, q_ss = _empty_full(len(s))
    q[:, 31:34] = [0.0, 0.0, 0.9]
    q[:, 34] = 0.2 * s + 0.05 * s * s
    q[:, 35] = -0.12 * s * s + 0.03 * s * s * s
    q[:, 36] = 0.08 * s + 0.04 * s * s
    q_s[:, 34] = 0.2 + 0.1 * s
    q_s[:, 35] = -0.24 * s + 0.09 * s * s
    q_s[:, 36] = 0.08 + 0.08 * s
    q_ss[:, 34] = 0.1
    q_ss[:, 35] = -0.24 + 0.18 * s
    q_ss[:, 36] = 0.08
    ready_quat = np.array(
        [np.cos(0.31), np.sin(0.31), 0.0, 0.0], dtype=np.float64
    )
    path = _adapt(
        model,
        binding,
        scope="full",
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        ready_quat=ready_quat,
    )
    report = adapter.validate_with_mj_differentiate_pos(
        _FakeMujoco,
        model,
        binding,
        path,
        path_parameter=s,
        tangent_absolute_tolerance=2.0e-7,
        tangent_relative_tolerance=2.0e-7,
        curvature_absolute_tolerance=2.0e-6,
        curvature_relative_tolerance=2.0e-6,
    )
    assert report["status"] == "PASS_SAMPLED_MUJOCO_DIFFERENTIATE_POS"
    assert report["max_tangent_residual"] < 2.0e-7
    assert report["max_curvature_residual"] < 2.0e-6
