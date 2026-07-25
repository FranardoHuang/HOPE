"""Pure-CPU tests for the A3 kinematic motion player.

The production module deliberately delays MuJoCo imports.  These tests use a
small fake name/address API to prove that FK playback is bound by names rather
than MJCF declaration order; no MuJoCo, viewer, Isaac, or GPU is required.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "mujoco_motion_player.py"
)
_SPEC = importlib.util.spec_from_file_location("mujoco_motion_player", _SCRIPT)
player = importlib.util.module_from_spec(_SPEC)
sys.modules["mujoco_motion_player"] = player
_SPEC.loader.exec_module(player)


def make_schema_arrays(frames: int = 3) -> dict[str, np.ndarray]:
    joint_pos = np.linspace(
        -0.2, 0.3, frames * 31, dtype=np.float64
    ).reshape(frames, 31)
    body_pos = np.zeros((frames, 32, 3), dtype=np.float64)
    body_pos[:, 0, 2] = 1.0
    body_quat = np.zeros((frames, 32, 4), dtype=np.float64)
    body_quat[..., 0] = 1.0
    return {
        "fps": np.array([50], dtype=np.int64),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": np.gradient(joint_pos, 0.02, axis=0).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": np.zeros((frames, 32, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 32, 3), dtype=np.float32),
        player.SCHEMA_KEY: np.array([2], dtype=np.int64),
        player.POS_POINT_KEY: np.array(player.BODY_POS_POINT),
        player.LIN_VEL_POINT_KEY: np.array(player.BODY_LIN_VEL_POINT),
        player.BODY_NAMES_KEY: np.asarray(player.RUNTIME_BODY_NAMES),
    }


def save_motion(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    np.savez(path, **arrays)
    return path


class FakeMjtObj:
    mjOBJ_JOINT = 1
    mjOBJ_BODY = 2
    mjOBJ_SITE = 3


class FakeMjtJoint:
    mjJNT_FREE = 0
    mjJNT_HINGE = 3


class FakeModel:
    """Names, declaration ids and qpos addresses intentionally disagree."""

    def __init__(self):
        self.nq = 38
        self.body_name_to_id = {
            name: index + 1
            for index, name in enumerate(player.RUNTIME_BODY_NAMES)
        }
        declaration = list(reversed(player.RUNTIME_JOINT_NAMES))
        self.joint_name_to_id = {"pelvis_free_joint": 0}
        self.joint_name_to_id.update(
            {name: index + 1 for index, name in enumerate(declaration)}
        )
        self.qaddr_by_name = {
            name: 7 + ((runtime_index * 7) % 31)
            for runtime_index, name in enumerate(player.RUNTIME_JOINT_NAMES)
        }
        self.daddr_by_name = {
            name: 6 + ((runtime_index * 7) % 31)
            for runtime_index, name in enumerate(player.RUNTIME_JOINT_NAMES)
        }
        self.jnt_type = np.full(32, FakeMjtJoint.mjJNT_HINGE, dtype=np.int64)
        self.jnt_type[0] = FakeMjtJoint.mjJNT_FREE
        self.jnt_qposadr = np.zeros(32, dtype=np.int64)
        self.jnt_dofadr = np.zeros(32, dtype=np.int64)
        self.jnt_bodyid = np.zeros(32, dtype=np.int64)
        self.jnt_qposadr[0] = 0
        self.jnt_dofadr[0] = 0
        self.jnt_bodyid[0] = self.body_name_to_id["pelvis_link"]
        for name, joint_id in self.joint_name_to_id.items():
            if name == "pelvis_free_joint":
                continue
            self.jnt_qposadr[joint_id] = self.qaddr_by_name[name]
            self.jnt_dofadr[joint_id] = self.daddr_by_name[name]
            self.jnt_bodyid[joint_id] = 1
        self.nbody = 33
        self.nsite = 1
        self.site_names = [player.RACKET_SITE_NAME]
        self.site_name_to_id = {player.RACKET_SITE_NAME: 0}
        self.site_bodyid = np.array(
            [self.body_name_to_id[player.RACKET_SITE_BODY_NAME]],
            dtype=np.int64,
        )
        self.site_pos = np.asarray(
            [player.RACKET_SITE_OFFSET_WRIST_M], dtype=np.float64
        )
        self.site_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.nv = 37


class FakeData:
    def __init__(self, model: FakeModel):
        self.qpos = np.zeros(model.nq, dtype=np.float64)
        self.qvel = np.ones(37, dtype=np.float64)
        self.act = np.ones(1, dtype=np.float64)
        self.ctrl = np.ones(31, dtype=np.float64)
        self.xpos = np.zeros((model.nbody, 3), dtype=np.float64)
        self.xipos = np.zeros((model.nbody, 3), dtype=np.float64)
        self.xquat = np.zeros((model.nbody, 4), dtype=np.float64)
        self.xquat[:, 0] = 1.0
        self.site_xpos = np.zeros((model.nsite, 3), dtype=np.float64)
        self.site_xmat = np.tile(np.eye(3).reshape(1, 9), (model.nsite, 1))
        self.forward_calls = 0


def fake_expected_poses(
    joint_pos: np.ndarray, root_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weighted = float(
        np.dot(np.asarray(joint_pos, dtype=np.float64), np.arange(1.0, 32.0))
    )
    pos = np.zeros((32, 3), dtype=np.float64)
    quat = np.zeros((32, 4), dtype=np.float64)
    quat[:, 0] = 1.0
    pos[0] = root_pos
    for body_index in range(1, 32):
        pos[body_index] = root_pos + np.array(
            [
                body_index * 0.01 + weighted * 1.0e-4,
                body_index * -0.002 + weighted * 2.0e-5,
                body_index * 0.003,
            ]
        )
    return pos, quat


class FakeModelFactory:
    @staticmethod
    def from_xml_path(_path: str) -> FakeModel:
        return FakeModel()


class FakeMujoco:
    mjtObj = FakeMjtObj
    mjtJoint = FakeMjtJoint
    MjModel = FakeModelFactory
    MjData = FakeData

    @staticmethod
    def mj_name2id(
        model: FakeModel, object_type: int, name: str
    ) -> int:
        if object_type == FakeMjtObj.mjOBJ_JOINT:
            return model.joint_name_to_id.get(name, -1)
        if object_type == FakeMjtObj.mjOBJ_BODY:
            return model.body_name_to_id.get(name, -1)
        if object_type == FakeMjtObj.mjOBJ_SITE:
            return model.site_name_to_id.get(name, -1)
        return -1

    @staticmethod
    def mj_id2name(
        model: FakeModel, object_type: int, object_id: int
    ) -> str | None:
        if object_type == FakeMjtObj.mjOBJ_JOINT:
            return next(
                (
                    name
                    for name, value in model.joint_name_to_id.items()
                    if value == object_id
                ),
                None,
            )
        if object_type == FakeMjtObj.mjOBJ_BODY:
            return next(
                (
                    name
                    for name, value in model.body_name_to_id.items()
                    if value == object_id
                ),
                None,
            )
        if object_type == FakeMjtObj.mjOBJ_SITE:
            if 0 <= object_id < len(model.site_names):
                return model.site_names[object_id]
        return None

    @staticmethod
    def mj_resetData(model: FakeModel, data: FakeData) -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.act[:] = 0.0
        data.ctrl[:] = 0.0
        data.xpos[:] = 0.0
        data.xipos[:] = 0.0
        data.xquat[:] = 0.0
        data.xquat[:, 0] = 1.0
        data.site_xpos[:] = 0.0
        data.site_xmat[:] = np.eye(3).reshape(1, 9)

    @staticmethod
    def mj_forward(model: FakeModel, data: FakeData) -> None:
        joint_pos = np.asarray(
            [
                data.qpos[model.qaddr_by_name[name]]
                for name in player.RUNTIME_JOINT_NAMES
            ]
        )
        body_pos, body_quat = fake_expected_poses(joint_pos, data.qpos[:3])
        for column, name in enumerate(player.RUNTIME_BODY_NAMES):
            body_id = model.body_name_to_id[name]
            data.xpos[body_id] = body_pos[column]
            data.xipos[body_id] = body_pos[column]
            data.xquat[body_id] = body_quat[column]
        racket_body = model.site_bodyid[0]
        data.site_xpos[0] = data.xpos[racket_body] + model.site_pos[0]
        data.site_xmat[0] = np.eye(3).reshape(9)
        data.forward_calls += 1

    @staticmethod
    def _site_jacobian(model: FakeModel) -> np.ndarray:
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacp[:, :3] = np.eye(3)
        coefficient = np.array([1.0e-4, 2.0e-5, 0.0])
        for runtime_index, name in enumerate(player.RUNTIME_JOINT_NAMES):
            jacp[:, model.daddr_by_name[name]] = (
                coefficient * (runtime_index + 1.0)
            )
        return jacp

    @staticmethod
    def mj_jacBody(
        model: FakeModel,
        data: FakeData,
        jacp: np.ndarray,
        jacr: np.ndarray,
        body_id: int,
    ) -> None:
        del data
        jacp[:] = 0.0
        jacr[:] = 0.0
        jacp[:, :3] = np.eye(3)
        jacr[:, 3:6] = np.eye(3)
        if body_id != model.body_name_to_id["pelvis_link"]:
            jacp[:] = FakeMujoco._site_jacobian(model)

    @staticmethod
    def mj_jacSite(
        model: FakeModel,
        data: FakeData,
        jacp: np.ndarray,
        jacr: np.ndarray,
        site_id: int,
    ) -> None:
        del data, site_id
        jacp[:] = FakeMujoco._site_jacobian(model)
        jacr[:] = 0.0

    @staticmethod
    def mj_objectVelocity(
        model: FakeModel,
        data: FakeData,
        object_type: int,
        object_id: int,
        result: np.ndarray,
        local: int,
    ) -> None:
        assert object_type == FakeMjtObj.mjOBJ_SITE
        assert object_id == 0
        assert local == 0
        result[:3] = 0.0
        result[3:] = FakeMujoco._site_jacobian(model) @ data.qvel


def make_fake_fk_arrays(frames: int = 3) -> dict[str, np.ndarray]:
    arrays = make_schema_arrays(frames)
    roots = np.zeros((frames, 3), dtype=np.float64)
    for frame in range(frames):
        root = np.array([0.1 * frame, -0.02 * frame, 1.0])
        roots[frame] = root
        pos, quat = fake_expected_poses(arrays["joint_pos"][frame], root)
        arrays["body_pos_w"][frame] = pos
        arrays["body_quat_w"][frame] = quat
    root_velocity = np.gradient(roots, 0.02, axis=0)
    weights = np.arange(1.0, 32.0)
    weighted_joint_velocity = arrays["joint_vel"].astype(np.float64) @ weights
    coefficient = np.array([1.0e-4, 2.0e-5, 0.0])
    arrays["body_lin_vel_w"][:, 0] = root_velocity
    arrays["body_lin_vel_w"][:, 1:] = (
        root_velocity[:, None, :]
        + weighted_joint_velocity[:, None, None] * coefficient[None, None, :]
    )
    return arrays


def load_fake_clip(tmp_path: Path, frames: int = 3) -> player.MotionClip:
    return player.load_motion(
        save_motion(tmp_path / "motion.npz", make_fake_fk_arrays(frames))
    )


def save_racket_reference(
    path: Path,
    report: dict,
    *,
    normal_convention: str = player.RACKET_NORMAL_CONVENTION,
    linear_velocity_offset: float = 0.0,
) -> Path:
    rows = report["racket"]["per_frame"]
    site_lin = np.asarray(
        [row["site_lin_vel_w_m_s"] for row in rows], dtype=np.float64
    )
    site_lin[0, 0] += linear_velocity_offset
    np.savez(
        path,
        racket_reference_schema_version=np.array(
            [player.RACKET_REFERENCE_SCHEMA_VERSION], dtype=np.int64
        ),
        site_name=np.array(player.RACKET_SITE_NAME),
        normal_convention=np.array(normal_convention),
        site_pos_w=np.asarray(
            [row["site_pos_w_m"] for row in rows], dtype=np.float64
        ),
        site_normal_w=np.asarray(
            [row["site_local_plus_y_normal_w"] for row in rows],
            dtype=np.float64,
        ),
        site_lin_vel_w=site_lin,
        site_ang_vel_w=np.asarray(
            [row["site_ang_vel_w_rad_s"] for row in rows],
            dtype=np.float64,
        ),
    )
    return path


def test_contract_has_exact_31_joint_and_32_body_orders():
    assert len(player.RUNTIME_JOINT_NAMES) == 31
    assert len(set(player.RUNTIME_JOINT_NAMES)) == 31
    assert len(player.RUNTIME_BODY_NAMES) == 32
    assert len(set(player.RUNTIME_BODY_NAMES)) == 32
    assert player.RUNTIME_BODY_NAMES[0] == "pelvis_link"
    assert player.RUNTIME_JOINT_NAMES[26] == "right_wrist_roll_joint"


@pytest.mark.parametrize("with_migration", [False, True])
def test_loader_accepts_only_exact_schema2_11_or_14_fields(
    tmp_path: Path, with_migration: bool
):
    arrays = make_schema_arrays()
    if with_migration:
        arrays.update(
            {
                player.MIGRATION_SOURCE_SHA256_KEY: np.array("a" * 64),
                player.MIGRATION_SOURCE_POINT_KEY: np.array("link_origin"),
                player.MIGRATION_TOOL_KEY: np.array("migration-tool/v2"),
            }
        )
    clip = player.load_motion(save_motion(tmp_path / "valid.npz", arrays))
    assert clip.has_migration_provenance is with_migration
    assert clip.joint_pos.shape == (3, 31)
    assert clip.body_pos_w.shape == (3, 32, 3)


def test_loader_rejects_unknown_field_and_partial_migration(tmp_path: Path):
    unknown = make_schema_arrays()
    unknown["surprise"] = np.array([1])
    save_motion(tmp_path / "unknown.npz", unknown)
    with pytest.raises(ValueError, match="exact schema-2 11/14 fields"):
        player.load_motion(tmp_path / "unknown.npz")

    partial = make_schema_arrays()
    partial[player.MIGRATION_TOOL_KEY] = np.array("tool")
    save_motion(tmp_path / "partial.npz", partial)
    with pytest.raises(ValueError, match="exact schema-2 11/14 fields"):
        player.load_motion(tmp_path / "partial.npz")


def test_loader_rejects_wrong_body_order_shape_nan_and_quaternion(tmp_path: Path):
    wrong_order = make_schema_arrays()
    names = list(player.RUNTIME_BODY_NAMES)
    names[1], names[2] = names[2], names[1]
    wrong_order[player.BODY_NAMES_KEY] = np.asarray(names)
    save_motion(tmp_path / "order.npz", wrong_order)
    with pytest.raises(ValueError, match="exact 32-name"):
        player.load_motion(tmp_path / "order.npz")

    wrong_shape = make_schema_arrays()
    wrong_shape["joint_pos"] = wrong_shape["joint_pos"][:, :-1]
    save_motion(tmp_path / "shape.npz", wrong_shape)
    with pytest.raises(ValueError, match=r"joint_pos must have shape .*31"):
        player.load_motion(tmp_path / "shape.npz")

    nan_array = make_schema_arrays()
    nan_array["body_lin_vel_w"][0, 0, 0] = np.nan
    save_motion(tmp_path / "nan.npz", nan_array)
    with pytest.raises(ValueError, match="NaN/Inf"):
        player.load_motion(tmp_path / "nan.npz")

    bad_quat = make_schema_arrays()
    bad_quat["body_quat_w"][0, 0] = np.array([2.0, 0.0, 0.0, 0.0])
    save_motion(tmp_path / "quat.npz", bad_quat)
    with pytest.raises(ValueError, match="non-unit quaternions"):
        player.load_motion(tmp_path / "quat.npz")


def test_model_binding_and_frame_write_are_name_based(tmp_path: Path):
    clip = load_fake_clip(tmp_path)
    model = FakeModel()
    data = FakeData(model)
    binding = player.bind_model(FakeMujoco, model)
    player.reset_kinematic_state(FakeMujoco, model, data)
    player.apply_frame(
        FakeMujoco,
        model,
        data,
        binding,
        clip.body_pos_w[1, 0],
        clip.body_quat_w[1, 0],
        clip.joint_pos[1],
    )
    assert np.array_equal(
        binding.joint_qpos_adrs,
        np.asarray(
            [model.qaddr_by_name[name] for name in player.RUNTIME_JOINT_NAMES]
        ),
    )
    for column, name in enumerate(player.RUNTIME_JOINT_NAMES):
        assert data.qpos[model.qaddr_by_name[name]] == pytest.approx(
            clip.joint_pos[1, column]
        )
    assert np.array_equal(data.qpos[:3], clip.body_pos_w[1, 0])
    assert data.forward_calls == 1


def test_smoke_check_passes_exact_fk_and_declares_evidence_boundary(tmp_path: Path):
    clip = load_fake_clip(tmp_path, frames=4)
    model = FakeModel()
    data = FakeData(model)
    report = player.smoke_check(
        clip,
        FakeMujoco,
        model,
        data,
        player.bind_model(FakeMujoco, model),
        position_tol_m=1.0e-6,
        orientation_tol_rad=1.0e-6,
    )
    assert report["verdict"] == player.PASS
    assert report["evidence_boundary"]["mj_forward_calls"] == 8
    assert report["evidence_boundary"]["mj_step_calls"] == 0
    assert report["evidence_boundary"]["dynamic_certificate"] is False
    assert report["evidence_boundary"]["training_certificate"] is False
    assert report["evidence_boundary"]["deployment_certificate"] is False
    assert report["evidence_boundary"]["hardware_certificate"] is False
    assert report["authorization"] == {
        "training": False,
        "deployment": False,
        "hardware": False,
    }
    assert report["contract"]["racket_site"] == "right_racket"
    assert (
        report["contract"]["racket_normal_convention"]
        == player.RACKET_NORMAL_CONVENTION
    )
    assert len(report["racket"]["per_frame"]) == 4
    assert len(report["racket"]["trajectory_sha256"]) == 64
    assert all(
        receipt["shape"] == [4, 3] and len(receipt["sha256"]) == 64
        for receipt in report["racket"]["array_receipts"].values()
    )
    assert report["gates"]["racket_site_position_vs_schema"]["pass"] is True
    assert report["gates"]["racket_site_normal_vs_schema"]["pass"] is True
    assert (
        report["gates"]["racket_site_linear_velocity_vs_schema"]["pass"]
        is True
    )
    assert (
        report["gates"]["racket_site_jacobian_vs_object_velocity"]["pass"]
        is True
    )
    assert data.forward_calls == 8


def test_binding_rejects_missing_or_duplicate_right_racket_site():
    missing = FakeModel()
    missing.nsite = 0
    missing.site_names = []
    missing.site_name_to_id = {}
    missing.site_bodyid = np.zeros(0, dtype=np.int64)
    missing.site_pos = np.zeros((0, 3), dtype=np.float64)
    missing.site_quat = np.zeros((0, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="exactly one site 'right_racket'.*0"):
        player.bind_model(FakeMujoco, missing)

    duplicate = FakeModel()
    duplicate.nsite = 2
    duplicate.site_names = ["right_racket", "right_racket"]
    duplicate.site_bodyid = np.repeat(duplicate.site_bodyid, 2)
    duplicate.site_pos = np.repeat(duplicate.site_pos, 2, axis=0)
    duplicate.site_quat = np.repeat(duplicate.site_quat, 2, axis=0)
    with pytest.raises(ValueError, match="exactly one site 'right_racket'.*2"):
        player.bind_model(FakeMujoco, duplicate)


def test_binding_rejects_wrong_normal_convention_and_site_model_drift():
    rotated = FakeModel()
    rotated.site_quat[0] = np.array(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )
    with pytest.raises(ValueError, match="local orientation drifted"):
        player.bind_model(FakeMujoco, rotated)

    wrong_body = FakeModel()
    wrong_body.site_bodyid[0] = wrong_body.body_name_to_id[
        "left_wrist_yaw_Link"
    ]
    with pytest.raises(ValueError, match="must be attached"):
        player.bind_model(FakeMujoco, wrong_body)

    shifted = FakeModel()
    shifted.site_pos[0, 0] += 1.0e-5
    with pytest.raises(ValueError, match="local position drifted"):
        player.bind_model(FakeMujoco, shifted)


def test_schema_velocity_tamper_fails_racket_site_gate(tmp_path: Path):
    clip = load_fake_clip(tmp_path, frames=4)
    racket_body_column = player.RUNTIME_BODY_NAMES.index(
        player.RACKET_SITE_BODY_NAME
    )
    clip.body_lin_vel_w[1, racket_body_column, 0] += 0.05
    model = FakeModel()
    report = player.smoke_check(
        clip,
        FakeMujoco,
        model,
        FakeData(model),
        player.bind_model(FakeMujoco, model),
    )
    assert report["verdict"] == player.FAIL
    gate = report["gates"]["racket_site_linear_velocity_vs_schema"]
    assert gate["pass"] is False
    assert gate["worst_frame"] == 1
    assert gate["max_error_m_s"] == pytest.approx(0.05, abs=1.0e-5)


def test_optional_racket_reference_accepts_exact_and_rejects_tamper(
    tmp_path: Path,
):
    clip = load_fake_clip(tmp_path, frames=4)
    model = FakeModel()
    binding = player.bind_model(FakeMujoco, model)
    baseline = player.smoke_check(
        clip, FakeMujoco, model, FakeData(model), binding
    )
    exact_path = save_racket_reference(tmp_path / "exact.npz", baseline)
    exact = player.load_racket_reference(exact_path, clip.n_frames)
    exact_report = player.smoke_check(
        clip,
        FakeMujoco,
        model,
        FakeData(model),
        binding,
        racket_reference=exact,
    )
    assert exact_report["verdict"] == player.PASS
    assert exact_report["gates"]["racket_external_reference"]["enabled"] is True
    assert exact_report["gates"]["racket_external_reference"]["pass"] is True

    tampered_path = save_racket_reference(
        tmp_path / "tampered.npz",
        baseline,
        linear_velocity_offset=0.05,
    )
    tampered = player.load_racket_reference(tampered_path, clip.n_frames)
    tampered_report = player.smoke_check(
        clip,
        FakeMujoco,
        model,
        FakeData(model),
        binding,
        racket_reference=tampered,
    )
    assert tampered_report["verdict"] == player.FAIL
    assert (
        tampered_report["gates"]["racket_external_reference"]["pass"] is False
    )
    assert tampered_report["gates"]["racket_external_reference"][
        "max_errors"
    ]["linear_velocity_m_s"] == pytest.approx(0.05)


def test_racket_reference_rejects_wrong_normal_convention(tmp_path: Path):
    clip = load_fake_clip(tmp_path, frames=3)
    model = FakeModel()
    report = player.smoke_check(
        clip,
        FakeMujoco,
        model,
        FakeData(model),
        player.bind_model(FakeMujoco, model),
    )
    path = save_racket_reference(
        tmp_path / "wrong_axis.npz",
        report,
        normal_convention="right_racket_site_local_plus_x_world_v1",
    )
    with pytest.raises(ValueError, match="normal_convention"):
        player.load_racket_reference(path, clip.n_frames)


def test_smoke_check_rejects_programmatic_reference_shape_drift(
    tmp_path: Path,
):
    clip = load_fake_clip(tmp_path, frames=3)
    model = FakeModel()
    binding = player.bind_model(FakeMujoco, model)
    baseline = player.smoke_check(
        clip, FakeMujoco, model, FakeData(model), binding
    )
    path = save_racket_reference(tmp_path / "exact.npz", baseline)
    exact = player.load_racket_reference(path, clip.n_frames)
    malformed = player.RacketTrajectoryReference(
        path=exact.path,
        site_pos_w=exact.site_pos_w,
        site_normal_w=exact.site_normal_w,
        site_lin_vel_w=exact.site_lin_vel_w,
        site_ang_vel_w=exact.site_ang_vel_w[:-1],
    )
    with pytest.raises(ValueError, match=r"site_ang_vel_w must have shape"):
        player.smoke_check(
            clip,
            FakeMujoco,
            model,
            FakeData(model),
            binding,
            racket_reference=malformed,
        )


def test_smoke_check_fails_loudly_on_stored_pose_drift(tmp_path: Path):
    clip = load_fake_clip(tmp_path)
    clip.body_pos_w[2, 17, 0] += 0.02
    model = FakeModel()
    report = player.smoke_check(
        clip,
        FakeMujoco,
        model,
        FakeData(model),
        player.bind_model(FakeMujoco, model),
        position_tol_m=1.0e-4,
    )
    assert report["verdict"] == player.FAIL
    assert report["gates"]["position"]["worst_frame"] == 2
    assert (
        report["gates"]["position"]["worst_body"]
        == player.RUNTIME_BODY_NAMES[17]
    )
    assert report["gates"]["position"]["max_error_m"] == pytest.approx(0.02)


def test_quaternion_error_is_sign_invariant():
    quat = np.array([[0.70710678, 0.70710678, 0.0, 0.0]])
    assert player.quaternion_angle_error_rad(quat, -quat)[0] == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_report_is_no_clobber_and_dynamic_command_only_suggests(
    tmp_path: Path,
):
    output = tmp_path / "report.json"
    player.write_report_no_clobber(output, {"verdict": "PASS"})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        player.write_report_no_clobber(output, {"verdict": "FAIL"})
    assert output.read_bytes() == original

    command = player.dynamic_replay_command(
        tmp_path / "motion with spaces.npz",
        tmp_path / "vendor model.xml",
        tmp_path / "dynamic report.json",
    )
    tokens = shlex.split(command)
    assert tokens[1].endswith("motion_dynamic_replay.py")
    assert "--motion" in tokens
    assert "--mjcf" in tokens
    assert "--out" in tokens
    assert "mujoco" not in player.__dict__


def test_cli_writes_nothing_without_report_and_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    motion = save_motion(tmp_path / "clip.npz", make_fake_fk_arrays())
    mjcf = tmp_path / "a3_pingpong.xml"
    mjcf.write_text("<fake/>", encoding="utf-8")
    monkeypatch.setattr(player, "load_mujoco", lambda: FakeMujoco)

    before = {path.name for path in tmp_path.iterdir()}
    assert (
        player.main(["--motion", str(motion), "--mjcf", str(mjcf)])
        == 0
    )
    assert {path.name for path in tmp_path.iterdir()} == before

    report = tmp_path / "report.json"
    assert (
        player.main(
            [
                "--motion",
                str(motion),
                "--mjcf",
                str(mjcf),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    report_bytes = report.read_bytes()
    assert (
        player.main(
            [
                "--motion",
                str(motion),
                "--mjcf",
                str(mjcf),
                "--report",
                str(report),
            ]
        )
        == 2
    )
    assert report.read_bytes() == report_bytes
