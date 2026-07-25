"""Pure-fixture tests for pre-registered exact MuJoCo model identity."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "canonical_mujoco_identity.py"
)
_REPOSITORY_MANIFEST = (
    _REPO_ROOT / "configs" / "a3_mujoco_identity_v1_20260724.json"
)
_REPOSITORY_MANIFEST_SHA256 = (
    "1a7422da0ae38de22303702262185abc8a91a77fd234df086df2436afde202a5"
)
_REPOSITORY_A3_MJCF = (
    _REPO_ROOT
    / "agi"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
    / "a3_pingpong"
    / "a3_pingpong.xml"
)
_SPEC = importlib.util.spec_from_file_location(
    "canonical_mujoco_identity", _SCRIPT
)
identity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = identity
_SPEC.loader.exec_module(identity)


class _ObjectTypes:
    mjOBJ_BODY = 1
    mjOBJ_JOINT = 3
    mjOBJ_GEOM = 5
    mjOBJ_SITE = 6
    mjOBJ_ACTUATOR = 19


class _JointTypes:
    mjJNT_FREE = 0
    mjJNT_HINGE = 3


class _FakeModel:
    def __init__(self):
        self.nq = 9
        self.nv = 8
        self.njnt = 3
        self.nbody = 4
        self.ngeom = 2
        self.nsite = 1
        self.nu = 2
        self.nmesh = 2
        self.nplugin = 0
        self.jnt_type = np.array(
            [_JointTypes.mjJNT_FREE, _JointTypes.mjJNT_HINGE,
             _JointTypes.mjJNT_HINGE],
            dtype=np.int32,
        )
        self.jnt_bodyid = np.array([1, 2, 3], dtype=np.int32)
        self.jnt_qposadr = np.array([0, 7, 8], dtype=np.int32)
        self.jnt_dofadr = np.array([0, 6, 7], dtype=np.int32)
        self.body_parentid = np.array([0, 0, 1, 2], dtype=np.int32)
        self.site_bodyid = np.array([3], dtype=np.int32)
        self.actuator_trnid = np.array([[1, -1], [2, -1]], dtype=np.int32)
        self._names = {
            _ObjectTypes.mjOBJ_BODY: (
                "world",
                "pelvis_link",
                "link_a",
                "link_b",
            ),
            _ObjectTypes.mjOBJ_JOINT: (
                "pelvis_free_joint",
                "joint_a",
                "joint_b",
            ),
            _ObjectTypes.mjOBJ_GEOM: ("geom_a", "geom_b"),
            _ObjectTypes.mjOBJ_SITE: ("right_racket",),
            _ObjectTypes.mjOBJ_ACTUATOR: ("motor_a", "motor_b"),
        }
        self.mjb_payload = b"fixture-compiled-mjb-v1"


class _FakeMjModelLoader:
    owner = None

    @classmethod
    def from_xml_path(cls, path: str):
        owner = cls.owner
        assert owner is not None
        owner.compile_count += 1
        if owner.compile_callback is not None:
            owner.compile_callback(Path(path))
        return owner.model


class _FakeMujoco:
    mjtObj = _ObjectTypes
    mjtJoint = _JointTypes
    MjModel = _FakeMjModelLoader

    def __init__(self, package_root: Path):
        package_root.mkdir(parents=True)
        init = package_root / "__init__.py"
        init.write_text("# fake mujoco\n", encoding="utf-8")
        (package_root / "libmujoco.so.9.9.9").write_bytes(
            b"fixture-core-mujoco-library"
        )
        self.__file__ = str(init)
        self.__version__ = "9.9.9"
        self.model = _FakeModel()
        self.compile_count = 0
        self.compile_callback = None
        _FakeMjModelLoader.owner = self

    @staticmethod
    def mj_version() -> int:
        return 9_009_009

    @staticmethod
    def mj_versionString() -> str:
        return "9.9.9"

    @staticmethod
    def mj_id2name(model: _FakeModel, object_type: int, index: int):
        rows = model._names[object_type]
        return rows[index]

    @staticmethod
    def mj_saveModel(model: _FakeModel, filename: str, *unused) -> None:
        del unused
        Path(filename).write_bytes(model.mjb_payload)


def _write_source_tree(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source = tmp_path / "source"
    (source / "parts").mkdir(parents=True)
    (source / "mesh").mkdir()
    (source / "tex").mkdir()
    (source / "assets").mkdir()
    files = {
        "mesh_a": source / "mesh" / "a.stl",
        "mesh_b": source / "mesh" / "b.stl",
        "texture": source / "tex" / "ground.png",
        "hfield": source / "mesh" / "floor.bin",
        "skin": source / "mesh" / "robot.skn",
    }
    for name, path in files.items():
        path.write_bytes((name + "-payload").encode("ascii"))
    include = source / "parts" / "body.xml"
    include.write_text(
        """<mujocoinclude>
  <asset><mesh name="b" file="b.stl"/></asset>
</mujocoinclude>
""",
        encoding="utf-8",
    )
    root = source / "fixture.xml"
    root.write_text(
        """<mujoco model="FixtureModel">
  <compiler assetdir="assets" meshdir="mesh" texturedir="tex"/>
  <include file="parts/body.xml"/>
  <asset>
    <mesh name="a" file="a.stl"/>
    <texture name="ground" type="2d" file="ground.png"/>
    <hfield name="floor" file="floor.bin"/>
    <skin name="robot" file="robot.skn"/>
  </asset>
</mujoco>
""",
        encoding="utf-8",
    )
    files["include"] = include
    return root, files


def _portable_from_current(current) -> dict:
    return identity._portable_identity_payload(
        root_filename=current.source_closure.root_filename,
        root_mjcf_sha256=current.source_closure.root_mjcf_sha256,
        source_closure_sha256=current.source_closure.closure_sha256,
        source_member_count=current.source_closure.member_count,
        source_total_bytes=current.source_closure.total_bytes,
        compiled_mjb_sha256=current.compiled_mjb_sha256,
        compiled_mjb_size_bytes=current.compiled_mjb_size_bytes,
        compiler_toolchain_sha256=current.compiler_toolchain_sha256,
        model_contract_sha256=current.model_contract_sha256,
    )


def _manifest_from_current(current) -> dict:
    portable = _portable_from_current(current)
    return {
        "schema_version": identity.IDENTITY_SCHEMA_VERSION,
        "manifest_type": identity.EXPECTED_MANIFEST_TYPE,
        "identity_type": identity.IDENTITY_TYPE,
        "root_filename": current.source_closure.root_filename,
        "expected": {
            "root_mjcf_sha256": portable["root_mjcf_sha256"],
            "source_closure_sha256": portable["source_closure_sha256"],
            "source_member_count": portable["source_member_count"],
            "source_total_bytes": portable["source_total_bytes"],
            "compiled_mjb_sha256": portable["compiled_mjb_sha256"],
            "compiled_mjb_size_bytes": portable["compiled_mjb_size_bytes"],
            "compiler_toolchain_sha256": portable[
                "compiler_toolchain_sha256"
            ],
            "model_contract_sha256": portable["model_contract_sha256"],
            "portable_identity_sha256": hashlib.sha256(
                identity._canonical_json_bytes(portable)
            ).hexdigest(),
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_bytes(identity._canonical_json_bytes(payload))
    return path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_expected(
    tmp_path: Path,
    fake: _FakeMujoco,
    root: Path,
) -> tuple[Path, object]:
    current = identity._compute_current_identity(fake, root)
    manifest = _write_manifest(
        tmp_path / "expected_identity.json",
        _manifest_from_current(current),
    )
    return manifest, current


def _rewrite_expected(
    manifest: Path, field: str, value
) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["expected"][field] = value
    portable = {
        "schema_version": identity.IDENTITY_SCHEMA_VERSION,
        "identity_type": identity.IDENTITY_TYPE,
        "root_filename": payload["root_filename"],
        **{
            key: payload["expected"][key]
            for key in (
                "root_mjcf_sha256",
                "source_closure_sha256",
                "source_member_count",
                "source_total_bytes",
                "compiled_mjb_sha256",
                "compiled_mjb_size_bytes",
                "compiler_toolchain_sha256",
                "model_contract_sha256",
            )
        },
    }
    payload["expected"]["portable_identity_sha256"] = hashlib.sha256(
        identity._canonical_json_bytes(portable)
    ).hexdigest()
    _write_manifest(manifest, payload)
    return payload


def test_source_closure_covers_include_and_all_supported_external_types(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    first = identity.scan_mjcf_source_closure(root)
    second = identity.scan_mjcf_source_closure(root)

    assert first.closure_sha256 == second.closure_sha256
    assert first.receipt == second.receipt
    assert first.member_count == 7
    assert first.receipt["xml_model_name"] == "FixtureModel"
    assert first.receipt["compiler_directories"] == {
        "declared": {
            "assetdir": "assets",
            "meshdir": "mesh",
            "texturedir": "tex",
        },
        "effective": {
            "assetdir": "assets",
            "meshdir": "mesh",
            "texturedir": "tex",
        },
        "strippath": False,
    }
    assert [row["path"] for row in first.receipt["members"]] == [
        "fixture.xml",
        "mesh/a.stl",
        "mesh/b.stl",
        "mesh/floor.bin",
        "mesh/robot.skn",
        "parts/body.xml",
        "tex/ground.png",
    ]
    assert len(first.receipt["include_edges"]) == 1
    assert len(first.receipt["external_references"]) == 5


def test_repository_a3_manifest_exact_bytes_schema_and_source_closure():
    assert _file_sha256(_REPOSITORY_MANIFEST) == (
        _REPOSITORY_MANIFEST_SHA256
    )
    expected = identity.load_expected_identity_manifest(
        _REPOSITORY_MANIFEST,
        trusted_manifest_sha256=_REPOSITORY_MANIFEST_SHA256,
    )
    closure = identity.scan_mjcf_source_closure(_REPOSITORY_A3_MJCF)

    assert expected.manifest_sha256 == _REPOSITORY_MANIFEST_SHA256
    assert expected.root_filename == "a3_pingpong.xml"
    assert expected.root_mjcf_sha256 == closure.root_mjcf_sha256
    assert expected.source_closure_sha256 == closure.closure_sha256
    assert expected.source_member_count == closure.member_count == 75
    assert expected.source_total_bytes == closure.total_bytes == 14_127_373
    assert all(
        value is False
        for value in expected.raw["authorization"].values()
    )


def test_identity_is_independent_of_absolute_root_and_file_creation_order(
    tmp_path: Path,
):
    root_a, unused_a = _write_source_tree(tmp_path / "a")
    del unused_a
    root_b, unused_b = _write_source_tree(tmp_path / "b")
    del unused_b
    payloads = {
        path.relative_to(root_b.parent): path.read_bytes()
        for path in root_b.parent.rglob("*")
        if path.is_file()
    }
    for path in sorted(
        (path for path in root_b.parent.rglob("*") if path.is_file()),
        reverse=True,
    ):
        path.unlink()
    for relative_path in sorted(payloads, reverse=True):
        (root_b.parent / relative_path).write_bytes(
            payloads[relative_path]
        )

    closure_a = identity.scan_mjcf_source_closure(root_a)
    closure_b = identity.scan_mjcf_source_closure(root_b)
    assert closure_a.closure_sha256 == closure_b.closure_sha256
    assert closure_a.receipt == closure_b.receipt

    fake_a = _FakeMujoco(tmp_path / "a" / "fake_mujoco")
    current_a = identity._compute_current_identity(fake_a, root_a)
    fake_b = _FakeMujoco(tmp_path / "b" / "fake_mujoco")
    current_b = identity._compute_current_identity(fake_b, root_b)
    portable_a = _portable_from_current(current_a)
    portable_b = _portable_from_current(current_b)
    assert portable_a == portable_b
    assert (
        hashlib.sha256(identity._canonical_json_bytes(portable_a)).hexdigest()
        == hashlib.sha256(
            identity._canonical_json_bytes(portable_b)
        ).hexdigest()
    )


def test_nested_include_uses_main_directory_and_strippath_matches_compiler(
    tmp_path: Path,
):
    source = tmp_path / "source"
    (source / "parts").mkdir(parents=True)
    (source / "mesh").mkdir()
    (source / "mesh" / "payload.stl").write_bytes(b"mesh")
    (source / "leaf.xml").write_text(
        "<mujocoinclude><asset>"
        '<mesh name="m" file="ignored/subdir/payload.stl"/>'
        "</asset></mujocoinclude>",
        encoding="utf-8",
    )
    (source / "parts" / "middle.xml").write_text(
        '<mujocoinclude><include file="leaf.xml"/></mujocoinclude>',
        encoding="utf-8",
    )
    root = source / "fixture.xml"
    root.write_text(
        '<mujoco model="x"><compiler meshdir="mesh" strippath="true"/>'
        '<include file="parts/middle.xml"/></mujoco>',
        encoding="utf-8",
    )

    closure = identity.scan_mjcf_source_closure(root)

    assert [row["resolved_path"] for row in closure.receipt["include_edges"]] == [
        "parts/middle.xml",
        "leaf.xml",
    ]
    assert closure.receipt["compiler_directories"]["strippath"] is True
    assert list(closure.receipt["external_references"]) == [
        {
            "attribute": "file",
            "declaring_xml": "leaf.xml",
            "effective_file": "payload.stl",
            "element": "mesh",
            "raw_file": "ignored/subdir/payload.stl",
            "resolved_path": "mesh/payload.stl",
        }
    ]


def test_cube_texture_faces_are_all_in_closure_and_cannot_escape(
    tmp_path: Path,
):
    source = tmp_path / "source"
    texture_dir = source / "tex"
    texture_dir.mkdir(parents=True)
    attributes = (
        "fileright",
        "fileleft",
        "fileup",
        "filedown",
        "filefront",
        "fileback",
    )
    for attribute in attributes:
        (texture_dir / f"{attribute}.png").write_bytes(
            attribute.encode("ascii")
        )
    root = source / "fixture.xml"
    face_attributes = " ".join(
        f'{attribute}="{attribute}.png"' for attribute in attributes
    )
    root.write_text(
        '<mujoco model="x"><compiler texturedir="tex"/><asset>'
        f'<texture name="cube" type="cube" {face_attributes}/>'
        "</asset></mujoco>",
        encoding="utf-8",
    )

    closure = identity.scan_mjcf_source_closure(root)

    assert closure.member_count == 7
    assert {
        row["attribute"] for row in closure.receipt["external_references"]
    } == set(attributes)

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    root.write_text(
        '<mujoco model="x"><asset><texture name="cube" type="cube" '
        'fileright="../outside.png"/></asset></mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(identity.ExactMujocoIdentityError, match="escapes"):
        identity.scan_mjcf_source_closure(root)


@pytest.mark.parametrize("kind", ["include", "mesh"])
def test_dependency_path_escape_fails_closed(tmp_path: Path, kind: str):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    root = source / "fixture.xml"
    if kind == "include":
        root.write_text(
            '<mujoco model="x"><include file="../outside.bin"/></mujoco>',
            encoding="utf-8",
        )
    else:
        root.write_text(
            '<mujoco model="x"><asset>'
            '<mesh file="../outside.bin"/></asset></mujoco>',
            encoding="utf-8",
        )
    with pytest.raises(identity.ExactMujocoIdentityError, match="escapes"):
        identity.scan_mjcf_source_closure(root)


def test_symlink_root_dependency_and_manifest_fail_closed(tmp_path: Path):
    root, files = _write_source_tree(tmp_path)
    root_link = tmp_path / "root_link.xml"
    root_link.symlink_to(root)
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="symlink"
    ):
        identity.scan_mjcf_source_closure(root_link)

    original = files["mesh_a"]
    moved = original.with_name("real_a.stl")
    original.rename(moved)
    original.symlink_to(moved.name)
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="symlink"
    ):
        identity.scan_mjcf_source_closure(root)

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    manifest_link = tmp_path / "manifest_link.json"
    manifest_link.symlink_to(manifest)
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="symlink"
    ):
        identity.load_expected_identity_manifest(
            manifest_link,
            trusted_manifest_sha256=_file_sha256(manifest),
        )


def test_unknown_file_tag_include_cycle_and_included_compiler_fail_closed(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"x")
    unsupported = source / "unsupported.xml"
    unsupported.write_text(
        '<mujoco model="x"><plugin file="payload.bin"/></mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="unsupported file-bearing"
    ):
        identity.scan_mjcf_source_closure(unsupported)

    a = source / "a.xml"
    b = source / "b.xml"
    a.write_text(
        '<mujoco model="x"><include file="b.xml"/></mujoco>',
        encoding="utf-8",
    )
    b.write_text(
        '<mujocoinclude><include file="a.xml"/></mujocoinclude>',
        encoding="utf-8",
    )
    with pytest.raises(identity.ExactMujocoIdentityError, match="cycle"):
        identity.scan_mjcf_source_closure(a)

    b.write_text(
        '<mujocoinclude><compiler meshdir="."/></mujocoinclude>',
        encoding="utf-8",
    )
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="none in included XML"
    ):
        identity.scan_mjcf_source_closure(a)


def test_dtd_and_entity_declarations_fail_closed(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    root = source / "fixture.xml"
    root.write_text(
        '<!DOCTYPE mujoco [<!ENTITY x "bad">]>'
        '<mujoco model="x"></mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="forbidden DTD"
    ):
        identity.scan_mjcf_source_closure(root)


def test_valid_expected_manifest_and_verification_are_deterministic(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, current = _prepare_expected(tmp_path, fake, root)
    trusted_manifest_sha = _file_sha256(manifest)
    loaded = identity.load_expected_identity_manifest(
        manifest,
        trusted_manifest_sha256=trusted_manifest_sha,
    )
    compile_count_before = fake.compile_count

    first = identity._verify_exact_mujoco_identity_with_module(
        mjcf_path=root,
        expected_manifest_path=manifest,
        trusted_expected_manifest_sha256=trusted_manifest_sha,
        mujoco_module=fake,
    )
    second = identity._verify_exact_mujoco_identity_with_module(
        mjcf_path=root,
        expected_manifest_path=manifest,
        trusted_expected_manifest_sha256=trusted_manifest_sha,
        mujoco_module=fake,
    )

    assert fake.compile_count == compile_count_before + 2
    assert not hasattr(first, "model")
    assert first.consume_verified_model(
        lambda model: model is fake.model
    ) is True
    assert first.portable_identity_sha256 == (
        loaded.portable_identity_sha256
    )
    assert first.portable_identity_sha256 == (
        second.portable_identity_sha256
    )
    assert first.verification_receipt_sha256 == (
        second.verification_receipt_sha256
    )
    first.assert_model_unchanged()
    assert first.receipt["source_closure"] == (
        dict(current.source_closure.receipt)
    )
    assert first.receipt["compiled_mjb"] == {
        "sha256": hashlib.sha256(fake.model.mjb_payload).hexdigest(),
        "size_bytes": len(fake.model.mjb_payload),
        "stored_in_repository": False,
    }
    assert first.receipt["claims"] == {
        "pre_registered_expected_values_used": True,
        "runtime_values_promoted_to_expected": False,
        "external_manifest_sha256_matched": True,
        "source_closure_checked_before_and_after_compile": True,
        "model_contract_checked_before_and_after_mjb": True,
        "complete_compiled_mjb_bound": True,
        "live_model_identity_is_point_in_time": True,
        "live_model_exposed_directly": False,
        "live_model_recheck_method": "assert_model_unchanged",
        "live_model_consumer_method": "consume_verified_model",
        "trusted_process_and_filesystem_required": True,
        "hostile_concurrent_source_writer_resistant": False,
        "core_library_process_mapping_proven": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }
    with pytest.raises(TypeError):
        first.receipt["claims"]["training_authorized"] = True
    with pytest.raises(AttributeError):
        first.receipt["source_closure"]["members"].append({})
    with pytest.raises(TypeError):
        loaded.raw["authorization"]["training_authorized"] = True


def test_live_model_recheck_detects_post_verification_mutation(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, unused_current = _prepare_expected(tmp_path, fake, root)
    del unused_current
    verified = identity._verify_exact_mujoco_identity_with_module(
        mjcf_path=root,
        expected_manifest_path=manifest,
        trusted_expected_manifest_sha256=_file_sha256(manifest),
        mujoco_module=fake,
    )

    fake.model.jnt_qposadr[1] = 8

    with pytest.raises(
        identity.ExactMujocoIdentityError,
        match="changed after exact identity",
    ) as caught:
        verified.assert_model_unchanged()
    assert "model_contract_sha256" in caught.value.report["mismatches"]

    fake.model.jnt_qposadr[1] = 7
    verified.assert_model_unchanged()

    def mutate_inside_consumer(model) -> None:
        model.jnt_qposadr[1] = 8

    with pytest.raises(
        identity.ExactMujocoIdentityError,
        match="changed after exact identity",
    ):
        verified.consume_verified_model(mutate_inside_consumer)


def test_model_contract_is_rechecked_after_mjb_serialization(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")

    def save_then_mutate(model, filename: str, *unused_arguments) -> None:
        del unused_arguments
        Path(filename).write_bytes(model.mjb_payload)
        model.jnt_qposadr[1] = 8

    fake.mj_saveModel = save_then_mutate
    with pytest.raises(
        identity.ExactMujocoIdentityError,
        match="changed while its MJB was serialized",
    ):
        identity._compute_current_identity(fake, root)


def test_manifest_duplicate_unknown_authorization_and_digest_fail_closed(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, unused_current = _prepare_expected(tmp_path, fake, root)
    del unused_current
    valid = json.loads(manifest.read_text(encoding="utf-8"))

    manifest.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(identity.ExactMujocoIdentityError, match="duplicate"):
        identity.load_expected_identity_manifest(
            manifest,
            trusted_manifest_sha256=_file_sha256(manifest),
        )

    unknown = copy.deepcopy(valid)
    unknown["unexpected"] = 1
    _write_manifest(manifest, unknown)
    with pytest.raises(identity.ExactMujocoIdentityError, match="keys changed"):
        identity.load_expected_identity_manifest(
            manifest,
            trusted_manifest_sha256=_file_sha256(manifest),
        )

    authorized = copy.deepcopy(valid)
    authorized["authorization"]["training_authorized"] = True
    _write_manifest(manifest, authorized)
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="cannot authorize"
    ):
        identity.load_expected_identity_manifest(
            manifest,
            trusted_manifest_sha256=_file_sha256(manifest),
        )

    bad_digest = copy.deepcopy(valid)
    bad_digest["expected"]["portable_identity_sha256"] = "0" * 64
    _write_manifest(manifest, bad_digest)
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="does not match"
    ):
        identity.load_expected_identity_manifest(
            manifest,
            trusted_manifest_sha256=_file_sha256(manifest),
        )


def test_runtime_reminted_manifest_fails_old_external_anchor_before_compile(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, unused_current = _prepare_expected(tmp_path, fake, root)
    del unused_current
    trusted_sha = _file_sha256(manifest)

    fake.model.mjb_payload = b"runtime-reminted-different-mjb"
    reminted_current = identity._compute_current_identity(fake, root)
    _write_manifest(manifest, _manifest_from_current(reminted_current))
    compile_count = fake.compile_count

    with pytest.raises(
        identity.ExactMujocoIdentityError,
        match="external trust anchor",
    ):
        identity._verify_exact_mujoco_identity_with_module(
            mjcf_path=root,
            expected_manifest_path=manifest,
            trusted_expected_manifest_sha256=trusted_sha,
            mujoco_module=fake,
        )
    assert fake.compile_count == compile_count


def test_root_filename_mismatch_fails_before_compilation(tmp_path: Path):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, unused_current = _prepare_expected(tmp_path, fake, root)
    del unused_current
    wrong = root.with_name("wrong.xml")
    wrong.write_bytes(root.read_bytes())
    count = fake.compile_count
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="basename"
    ):
        identity._verify_exact_mujoco_identity_with_module(
            mjcf_path=wrong,
            expected_manifest_path=manifest,
            trusted_expected_manifest_sha256=_file_sha256(manifest),
            mujoco_module=fake,
        )
    assert fake.compile_count == count


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_mjcf_sha256", "1" * 64),
        ("source_closure_sha256", "2" * 64),
        ("source_member_count", 999),
        ("source_total_bytes", 999),
        ("compiled_mjb_sha256", "3" * 64),
        ("compiled_mjb_size_bytes", 999),
        ("compiler_toolchain_sha256", "4" * 64),
        ("model_contract_sha256", "5" * 64),
    ],
)
def test_every_pre_registered_component_mismatch_fails_closed(
    tmp_path: Path, field: str, value
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, unused_current = _prepare_expected(tmp_path, fake, root)
    del unused_current
    _rewrite_expected(manifest, field, value)
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="does not match"
    ) as caught:
        identity._verify_exact_mujoco_identity_with_module(
            mjcf_path=root,
            expected_manifest_path=manifest,
            trusted_expected_manifest_sha256=_file_sha256(manifest),
            mujoco_module=fake,
        )
    assert field in caught.value.report["mismatches"]
    assert caught.value.report["training_authorized"] is False


def test_source_change_during_compile_fails_before_identity_acceptance(
    tmp_path: Path,
):
    root, files = _write_source_tree(tmp_path)
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, unused_current = _prepare_expected(tmp_path, fake, root)
    del unused_current

    def mutate_during_compile(unused_root: Path) -> None:
        del unused_root
        files["mesh_a"].write_bytes(b"changed-during-compile")
        fake.compile_callback = None

    fake.compile_callback = mutate_during_compile
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="changed while"
    ):
        identity._verify_exact_mujoco_identity_with_module(
            mjcf_path=root,
            expected_manifest_path=manifest,
            trusted_expected_manifest_sha256=_file_sha256(manifest),
            mujoco_module=fake,
        )


def test_model_address_and_core_library_drift_fail_against_manifest(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    manifest, unused_current = _prepare_expected(tmp_path, fake, root)
    del unused_current

    fake.model.jnt_qposadr[1] = 8
    with pytest.raises(identity.ExactMujocoIdentityError) as address_error:
        identity._verify_exact_mujoco_identity_with_module(
            mjcf_path=root,
            expected_manifest_path=manifest,
            trusted_expected_manifest_sha256=_file_sha256(manifest),
            mujoco_module=fake,
        )
    assert "model_contract_sha256" in (
        address_error.value.report["mismatches"]
    )

    fake.model.jnt_qposadr[1] = 7
    core = Path(fake.__file__).parent / "libmujoco.so.9.9.9"
    core.write_bytes(b"drifted-core-library")
    with pytest.raises(identity.ExactMujocoIdentityError) as core_error:
        identity._verify_exact_mujoco_identity_with_module(
            mjcf_path=root,
            expected_manifest_path=manifest,
            trusted_expected_manifest_sha256=_file_sha256(manifest),
            mujoco_module=fake,
        )
    assert "compiler_toolchain_sha256" in (
        core_error.value.report["mismatches"]
    )


def test_active_plugin_model_fails_until_plugin_resolver_is_versioned(
    tmp_path: Path,
):
    root, unused = _write_source_tree(tmp_path)
    del unused
    fake = _FakeMujoco(tmp_path / "fake_mujoco")
    fake.model.nplugin = 1
    with pytest.raises(
        identity.ExactMujocoIdentityError, match="nplugin=0"
    ):
        identity._compute_current_identity(fake, root)


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="MuJoCo is not installed for the repository identity integration test",
)
def test_repository_a3_full_pre_registered_identity_when_mujoco_available():
    importlib.import_module("mujoco")
    result = identity.verify_exact_mujoco_identity(
        mjcf_path=_REPOSITORY_A3_MJCF,
        expected_manifest_path=_REPOSITORY_MANIFEST,
        trusted_expected_manifest_sha256=_REPOSITORY_MANIFEST_SHA256,
    )
    assert result.receipt["status"] == identity.VERIFICATION_STATUS
    assert result.receipt["claims"]["complete_compiled_mjb_bound"] is True
    assert result.receipt["claims"]["training_authorized"] is False
