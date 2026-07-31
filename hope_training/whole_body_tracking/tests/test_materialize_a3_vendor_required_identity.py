"""Host-only tests for atomic A3 vendor live-contract materialization."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from typing import Callable

import pytest


TEST_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = TEST_ROOT / "scripts" / "materialize_a3_vendor_required_identity.py"
AUTHORITY_SCRIPT = TEST_ROOT / "scripts" / "materialize_a3_vendor_runtime_authority.py"
ROBOT_SCRIPT = (
    TEST_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "robots"
    / "agibot_a3.py"
)
PRODUCER_RELATIVE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_a3_vendor_required_identity.py"
)
AUTHORITY_RELATIVE = (
    "hope_training/whole_body_tracking/scripts/"
    "materialize_a3_vendor_runtime_authority.py"
)
REGISTRY_RELATIVE = (
    "hope_training/whole_body_tracking/scripts/a3_vendor_action_registry.py"
)

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

SOURCE_PATHS = {
    "robot_config": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/robots/agibot_a3.py"
    ),
    "task_profile": (
        "hope_training/whole_body_tracking/cfg/task/"
        "HOPEPingPongActionBallA3VendorV1.yaml"
    ),
    "training_contract_builder": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/utils/training_contract.py"
    ),
    "training_entrypoint": "hope_training/whole_body_tracking/scripts/train.py",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(root: Path, relative: str, raw: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _registry_source(
    *,
    loop_contract: str = "configs/runtime_epoch/loop.training_contract.json",
    loop_identity: str = "configs/identity_epoch/loop.required_identity.json",
    block_contract: str = "configs/runtime_epoch/block.training_contract.json",
    block_identity: str = "configs/identity_epoch/block.required_identity.json",
    materialized: bool = False,
) -> bytes:
    pin = repr("f" * 64) if materialized else "None"
    return f'''from dataclasses import dataclass
from types import MappingProxyType

@dataclass(frozen=True)
class ArtifactPin:
    path: str
    sha256: object

@dataclass(frozen=True)
class VendorActionConfig:
    action_id: str
    stable_motion: ArtifactPin
    runtime_contract: ArtifactPin
    required_identity_manifest: ArtifactPin
    runtime_authority_receipt: ArtifactPin

class VendorActionRegistryError(ValueError):
    pass

_LOOP = VendorActionConfig(
    "bh_loop_c",
    ArtifactPin("assets/loop.npz", "{'a' * 64}"),
    ArtifactPin({loop_contract!r}, {pin}),
    ArtifactPin({loop_identity!r}, {pin}),
    ArtifactPin("configs/authority/loop.json", None),
)
_BLOCK = VendorActionConfig(
    "bh_block",
    ArtifactPin("assets/block.npz", "{'b' * 64}"),
    ArtifactPin({block_contract!r}, {pin}),
    ArtifactPin({block_identity!r}, {pin}),
    ArtifactPin("configs/authority/block.json", None),
)
ACTION_CONFIGS = MappingProxyType({{"bh_loop_c": _LOOP, "bh_block": _BLOCK}})
ALLOWED_ACTION_IDS = frozenset(ACTION_CONFIGS)
DEFAULT_ACTION_ID = "bh_loop_c"

def get_action_config(action_id):
    if type(action_id) is not str or action_id not in ACTION_CONFIGS:
        raise VendorActionRegistryError("vendor action_id must be one of: bh_block, bh_loop_c")
    return ACTION_CONFIGS[action_id]

def stable_pin(pin):
    if not pin.path or pin.sha256 is None:
        raise AssertionError("stable pin incomplete")
    return {{"path": pin.path, "sha256": pin.sha256}}
'''.encode("utf-8")


def _joint_values(name: str) -> tuple[float, float, float, float]:
    if name == "waist_yaw_joint":
        return 85.0, 3.0, 220.0, 0.06646569891
    if name == "waist_roll_joint":
        return 50.0, 2.0, 46.0, 0.01462087613
    if name == "waist_pitch_joint":
        return 50.0, 2.0, 118.0, 0.08820859156
    if name.startswith("head_"):
        return 40.0, 2.0, 6.0, 0.0008100893338
    if name.endswith(("_hip_pitch_joint", "_hip_yaw_joint")):
        return 80.0, 3.0, 220.0, 0.06646569891
    if name.endswith("_hip_roll_joint"):
        return 120.0, 4.0, 220.0, 0.06646569891
    if name.endswith("_knee_joint"):
        return 250.0, 8.0, 320.0, 0.1203404
    if name.endswith("_ankle_pitch_joint"):
        return 50.0, 2.0, 118.19999694824219, 0.06444060531
    if name.endswith("_ankle_roll_joint"):
        return 50.0, 2.0, 54.75, 0.02012630058
    if name.endswith(("_shoulder_pitch_joint", "_shoulder_roll_joint")):
        return 40.0, 3.0, 60.0, 0.01208336871
    if name.endswith(("_wrist_pitch_joint", "_wrist_yaw_joint")):
        return 20.0, 2.0, 6.0, 0.0008100893338
    return 30.0, 2.0, 24.0, 0.004967351303


def _contract(action_id: str) -> dict:
    motion_sha = "a" * 64 if action_id == "bh_loop_c" else "b" * 64
    rows = [_joint_values(name) for name in JOINT_NAMES]
    stiffness = [row[0] for row in rows]
    effort = [row[2] for row in rows]
    return {
        "schema_version": 3,
        "target_mode": "action_ball",
        "joint_names": list(JOINT_NAMES),
        "articulation_joint_names": list(JOINT_NAMES),
        "action_joint_ids": list(range(31)),
        "joint_stiffness": stiffness,
        "joint_damping": [row[1] for row in rows],
        "joint_effort_limits": effort,
        "joint_velocity_limits": [12.0] * 31,
        "joint_armature": [row[3] for row in rows],
        "default_joint_pos": [0.0] * 31,
        "action_scale": [0.25 * e / k for e, k in zip(effort, stiffness)],
        "qdes_joint_pos_limits": [[-1.0, 1.0] for _ in range(31)],
        "physics_step_dt_s": 0.005,
        "policy_step_dt_s": 0.02,
        "control_decimation": 4,
        "control_step_action_delay": {
            "schema_version": 1,
            "enabled": True,
            "semantic_unit": "policy_control_step",
            "sample_timing": "once_per_episode_reset",
            "distribution": "discrete_uniform_inclusive",
            "min_steps": 0,
            "max_steps": 2,
            "shared_across_all_31_joints": True,
            "history_fill": "safe_default_or_action_specific_hold",
        },
        "push_robot_event": {
            "schema_version": 2,
            "enabled": True,
            "semantics": "symmetric_6d_velocity_delta",
            "func": "push_by_setting_velocity",
            "mode": "interval",
            "interval_range_s": [5.0, 15.0],
            "velocity_range": {
                "x": [-0.25, 0.25],
                "y": [-0.25, 0.25],
                "z": [-0.1, 0.1],
                "roll": [-0.26, 0.26],
                "pitch": [-0.26, 0.26],
                "yaw": [-0.39, 0.39],
            },
        },
        "action_ball_training": {
            "preflight": {"action_order": [action_id]},
            "policy_bootstrap": {
                "schema_version": 1,
                "kind": "action_ball_shared_ready_actor_bootstrap_v1",
                "action_order": [action_id],
            },
            "motion_admission": {"motion_file_sha256": [motion_sha]},
        },
    }


def _pretty(document: object) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_module(root: Path):
    path = root / PRODUCER_RELATIVE
    name = f"fixture_required_identity_{id(root)}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    original = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original
    return module


def _fixture(
    tmp_path: Path,
    *,
    registry_bytes: bytes | None = None,
    action_id: str = "bh_loop_c",
) -> dict:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "fixture")
    _write(root, PRODUCER_RELATIVE, SOURCE_SCRIPT.read_bytes())
    _write(root, AUTHORITY_RELATIVE, AUTHORITY_SCRIPT.read_bytes())
    _write(root, REGISTRY_RELATIVE, registry_bytes or _registry_source())
    for role, relative in SOURCE_PATHS.items():
        payload = (
            ROBOT_SCRIPT.read_bytes()
            if role == "robot_config"
            else f"# {role} exact source\n".encode("utf-8")
        )
        _write(root, relative, payload)
    # Git cannot track empty output directories.  These sentinels make both
    # fixed registry parents exist in a clean clone without occupying targets.
    _write(root, "configs/runtime_epoch/.keep", b"")
    _write(root, "configs/identity_epoch/.keep", b"")
    _git(root, "add", "--", ".")
    _git(root, "commit", "-qm", "fixture source")
    commit = _git(root, "rev-parse", "HEAD")
    module = _load_module(root)
    live = tmp_path / f"{action_id}.live.json"
    live.write_bytes(_pretty(_contract(action_id)))
    config = module._REGISTRY.get_action_config(action_id)
    return {
        "root": root,
        "commit": commit,
        "module": module,
        "live": live,
        "config": config,
        "action_id": action_id,
    }


def _run(fx: dict) -> dict:
    return fx["module"].materialize_a3_vendor_required_identity(
        repo_root=fx["root"],
        source_commit=fx["commit"],
        action_id=fx["action_id"],
        live_training_contract=fx["live"],
    )


def _outputs(fx: dict) -> tuple[Path, Path]:
    config = fx["config"]
    return (
        fx["root"] / config.runtime_contract.path,
        fx["root"] / config.required_identity_manifest.path,
    )


def _group_for(identity: dict, joint: str) -> dict:
    return next(
        group
        for group in identity["robot_action_contract"]["groups"]
        if joint in group["joints"]
    )


@pytest.mark.parametrize("action_id", ["bh_loop_c", "bh_block"])
def test_materializes_fixed_action_isolated_outputs_byte_exact(
    tmp_path: Path, action_id: str
) -> None:
    fx = _fixture(tmp_path, action_id=action_id)
    live_bytes = fx["live"].read_bytes()
    result = _run(fx)
    contract_path, identity_path = _outputs(fx)

    assert contract_path.read_bytes() == live_bytes
    assert result["runtime_contract"] == {
        "path": fx["config"].runtime_contract.path,
        "sha256": _sha(live_bytes),
    }
    identity_bytes = identity_path.read_bytes()
    identity = json.loads(identity_bytes)
    assert identity_bytes == fx["module"]._canonical_identity_bytes(identity)
    assert result["required_identity"] == {
        "path": fx["config"].required_identity_manifest.path,
        "sha256": _sha(identity_bytes),
    }
    assert identity["runtime_materialization"]["required_dynamic_ready_actions"] == [
        action_id
    ]
    assert identity["runtime_materialization"]["training_contract_sha256"] == _sha(
        live_bytes
    )
    assert set(identity["sources"]) == set(SOURCE_PATHS)
    for role, relative in SOURCE_PATHS.items():
        blob = subprocess.run(
            ["git", "-C", str(fx["root"]), "show", f"{fx['commit']}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        assert identity["sources"][role] == {
            "path": relative,
            "sha256": _sha(blob),
        }
    assert stat_mode(contract_path) == 0o444
    assert stat_mode(identity_path) == 0o444


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_identity_uses_exact_deploy_nominal_and_stable_twelve_groups(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    _run(fx)
    identity = json.loads(_outputs(fx)[1].read_bytes())
    groups = identity["robot_action_contract"]["groups"]
    assert len(groups) == 12

    assert _group_for(identity, "waist_yaw_joint")["stiffness"] == 85.0
    assert _group_for(identity, "waist_pitch_joint")["effort_limit"] == 118.0
    for side in ("left", "right"):
        for axis in ("pitch", "yaw"):
            group = _group_for(identity, f"{side}_wrist_{axis}_joint")
            assert group["stiffness"] == 20.0
            assert group["effort_limit"] == 6.0
            assert group["armature"] == 0.0008100893338
    forbidden = [
        group
        for group in groups
        if group["stiffness"] == 30.0
        and any("wrist_pitch" in joint or "wrist_yaw" in joint for joint in group["joints"])
    ]
    assert forbidden == []
    assert _group_for(identity, "waist_yaw_joint")["stiffness"] != 80.0
    assert _group_for(identity, "waist_pitch_joint")["effort_limit"] != 115.0


def test_authority_tolerance_residue_cannot_create_a_thirteenth_group(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    document = _contract("bh_loop_c")
    index = JOINT_NAMES.index("left_wrist_pitch_joint")
    document["joint_armature"][index] += 5.0e-8
    fx["live"].write_bytes(_pretty(document))

    _run(fx)
    identity = json.loads(_outputs(fx)[1].read_bytes())
    assert len(identity["robot_action_contract"]["groups"]) == 12
    assert _group_for(identity, "left_wrist_pitch_joint")["armature"] == (
        0.0008100893338
    )
    assert _group_for(identity, "left_wrist_pitch_joint") is not None
    assert _group_for(identity, "left_wrist_pitch_joint")["joints"] == [
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]


def test_deterministic_across_two_clean_clones_of_same_commit(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "source")
    clones = []
    for index in (1, 2):
        root = tmp_path / f"clone{index}"
        _git(tmp_path, "clone", "-q", str(source["root"]), str(root))
        module = _load_module(root)
        live = tmp_path / f"clone{index}.live.json"
        live.write_bytes(source["live"].read_bytes())
        clones.append(
            {
                "root": root,
                "commit": source["commit"],
                "module": module,
                "live": live,
                "config": module._REGISTRY.get_action_config("bh_loop_c"),
                "action_id": "bh_loop_c",
            }
        )
    _run(clones[0])
    _run(clones[1])
    assert _outputs(clones[0])[0].read_bytes() == _outputs(clones[1])[0].read_bytes()
    assert _outputs(clones[0])[1].read_bytes() == _outputs(clones[1])[1].read_bytes()


@pytest.mark.parametrize("dirty_name", ["tracked", "untracked"])
def test_dirty_or_untracked_checkout_is_refused(
    tmp_path: Path, dirty_name: str
) -> None:
    fx = _fixture(tmp_path)
    if dirty_name == "tracked":
        (fx["root"] / SOURCE_PATHS["robot_config"]).write_text("dirty\n")
    else:
        (fx["root"] / "untracked.txt").write_text("dirty\n")
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="clean"):
        _run(fx)
    assert not any(path.exists() for path in _outputs(fx))


def test_head_drift_and_full_commit_are_refused(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    (fx["root"] / "next.txt").write_text("next\n")
    _git(fx["root"], "add", "next.txt")
    _git(fx["root"], "commit", "-qm", "next")
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="HEAD="):
        _run(fx)
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="full 40"):
        fx["module"].materialize_a3_vendor_required_identity(
            repo_root=fx["root"],
            source_commit=fx["commit"][:12],
            action_id=fx["action_id"],
            live_training_contract=fx["live"],
        )


@pytest.mark.parametrize("relative", [PRODUCER_RELATIVE, *SOURCE_PATHS.values()])
def test_assume_unchanged_source_blob_drift_is_still_refused(
    tmp_path: Path, relative: str
) -> None:
    fx = _fixture(tmp_path)
    _git(fx["root"], "update-index", "--assume-unchanged", "--", relative)
    path = fx["root"] / relative
    path.write_bytes(path.read_bytes() + b"# drift\n")
    assert _git(fx["root"], "status", "--porcelain=v1") == ""
    pattern = "required-identity producer|differs between source commit"
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match=pattern):
        _run(fx)
    assert not any(path.exists() for path in _outputs(fx))


def test_materialized_pins_and_unknown_action_are_refused(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, registry_bytes=_registry_source(materialized=True))
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="already"):
        _run(fx)
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="one of"):
        fx["module"].materialize_a3_vendor_required_identity(
            repo_root=fx["root"],
            source_commit=fx["commit"],
            action_id="bh_unknown",
            live_training_contract=fx["live"],
        )


@pytest.mark.parametrize(
    "case",
    ["relative", "path_dotdot", "symlink", "noncanonical", "duplicate", "nan"],
)
def test_invalid_live_file_form_is_refused(tmp_path: Path, case: str) -> None:
    fx = _fixture(tmp_path)
    live: str | Path = fx["live"]
    if case == "relative":
        live = fx["live"].name
    elif case == "path_dotdot":
        live = fx["live"].parent / "unused" / ".." / fx["live"].name
    elif case == "symlink":
        link = tmp_path / "live-link.json"
        link.symlink_to(fx["live"])
        live = link
    elif case == "noncanonical":
        fx["live"].write_bytes(json.dumps(_contract("bh_loop_c")).encode())
    elif case == "duplicate":
        fx["live"].write_bytes(b'{"schema_version":3,"schema_version":3}\n')
    elif case == "nan":
        fx["live"].write_bytes(b'{"schema_version":NaN}\n')
    with pytest.raises(fx["module"].VendorRequiredIdentityError):
        fx["module"].materialize_a3_vendor_required_identity(
            repo_root=fx["root"],
            source_commit=fx["commit"],
            action_id=fx["action_id"],
            live_training_contract=live,
        )
    assert not any(path.exists() for path in _outputs(fx))


def test_live_file_stat_mutation_is_refused(tmp_path: Path, monkeypatch) -> None:
    fx = _fixture(tmp_path)
    original = fx["module"].os.fstat
    calls = {"count": 0}

    def changing_fstat(descriptor: int):
        value = original(descriptor)
        calls["count"] += 1
        if calls["count"] == 2:
            return SimpleNamespace(
                st_dev=value.st_dev,
                st_ino=value.st_ino,
                st_size=value.st_size,
                st_mtime_ns=value.st_mtime_ns + 1,
                st_ctime_ns=value.st_ctime_ns,
                st_mode=value.st_mode,
            )
        return value

    monkeypatch.setattr(fx["module"].os, "fstat", changing_fstat)
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="changed"):
        _run(fx)
    assert not any(path.exists() for path in _outputs(fx))


def _mutate_schema(document: dict) -> None:
    document["schema_version"] = 2


def _mutate_target(document: dict) -> None:
    document["target_mode"] = "motion"


def _mutate_order(document: dict) -> None:
    document["joint_names"][0], document["joint_names"][1] = (
        document["joint_names"][1], document["joint_names"][0]
    )
    document["articulation_joint_names"] = list(document["joint_names"])


def _mutate_ids(document: dict) -> None:
    document["action_joint_ids"][-1] = 0


def _mutate_motion(document: dict) -> None:
    document["action_ball_training"]["motion_admission"]["motion_file_sha256"] = [
        "0" * 64
    ]


def _mutate_bootstrap(document: dict) -> None:
    document["action_ball_training"]["policy_bootstrap"]["action_order"] = [
        "bh_block"
    ]


def _mutate_delay(document: dict) -> None:
    document["control_step_action_delay"]["max_steps"] = 1


def _mutate_push(document: dict) -> None:
    document["push_robot_event"]["velocity_range"]["x"] = [-0.5, 0.5]


def _mutate_nominal(document: dict) -> None:
    document["joint_stiffness"][JOINT_NAMES.index("waist_yaw_joint")] = 80.0


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_schema,
        _mutate_target,
        _mutate_order,
        _mutate_ids,
        _mutate_motion,
        _mutate_bootstrap,
        _mutate_delay,
        _mutate_push,
        _mutate_nominal,
    ],
)
def test_wrong_runtime_semantics_are_refused(
    tmp_path: Path, mutation: Callable[[dict], None]
) -> None:
    fx = _fixture(tmp_path)
    document = deepcopy(_contract("bh_loop_c"))
    mutation(document)
    fx["live"].write_bytes(_pretty(document))
    with pytest.raises(
        fx["module"].VendorRequiredIdentityError,
        match="authority validation",
    ):
        _run(fx)
    assert not any(path.exists() for path in _outputs(fx))


@pytest.mark.parametrize("occupied", ["contract", "identity"])
def test_preoccupied_fixed_target_never_creates_sibling(
    tmp_path: Path, occupied: str
) -> None:
    fx = _fixture(tmp_path)
    contract_path, identity_path = _outputs(fx)
    chosen = contract_path if occupied == "contract" else identity_path
    chosen.write_bytes(b"spent\n")
    _git(fx["root"], "add", "--", str(chosen.relative_to(fx["root"])))
    _git(fx["root"], "commit", "-qm", f"occupy {occupied}")
    fx["commit"] = _git(fx["root"], "rev-parse", "HEAD")
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="must not exist"):
        _run(fx)
    sibling = identity_path if occupied == "contract" else contract_path
    assert not sibling.exists()
    assert chosen.read_bytes() == b"spent\n"


def test_second_reservation_failure_rolls_back_first(
    tmp_path: Path, monkeypatch
) -> None:
    fx = _fixture(tmp_path)
    targets = [
        fx["module"]._fixed_output_target(
            fx["root"], fx["config"].runtime_contract.path, name="a"
        ),
        fx["module"]._fixed_output_target(
            fx["root"], fx["config"].required_identity_manifest.path, name="b"
        ),
    ]
    real_open = fx["module"].os.open
    reserve_calls = {"count": 0}

    def fail_second(path, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            reserve_calls["count"] += 1
            if reserve_calls["count"] == 2:
                raise OSError("injected second reservation failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fx["module"].os, "open", fail_second)
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="reserve"):
        fx["module"]._reserve_outputs(targets)
    assert not any(path.exists() for path in _outputs(fx))


@pytest.mark.parametrize("failure", ["write", "fsync"])
def test_publication_failure_rolls_back_both_outputs(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    fx = _fixture(tmp_path)
    if failure == "write":
        real_write = fx["module"].os.write
        calls = {"count": 0}

        def fail_write(descriptor, payload):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("injected write failure")
            return real_write(descriptor, payload)

        monkeypatch.setattr(fx["module"].os, "write", fail_write)
    else:
        real_fsync = fx["module"].os.fsync
        calls = {"count": 0}

        def fail_fsync(descriptor):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError("injected fsync failure")
            return real_fsync(descriptor)

        monkeypatch.setattr(fx["module"].os, "fsync", fail_fsync)
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="rolled back"):
        _run(fx)
    assert not any(path.exists() for path in _outputs(fx))


@pytest.mark.parametrize(
    "registry_bytes",
    [
        _registry_source(loop_identity="configs/runtime_epoch/loop.training_contract.json"),
        _registry_source(loop_contract="configs/runtime_epoch/../escape.json"),
    ],
)
def test_same_target_or_path_traversal_is_refused(
    tmp_path: Path, registry_bytes: bytes
) -> None:
    fx = _fixture(tmp_path, registry_bytes=registry_bytes)
    with pytest.raises(fx["module"].VendorRequiredIdentityError):
        _run(fx)


def test_symlink_output_parent_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    registry = _registry_source(loop_contract="configs/link/loop.json")
    fx = _fixture(tmp_path, registry_bytes=registry)
    link = fx["root"] / "configs" / "link"
    link.symlink_to(outside, target_is_directory=True)
    _git(fx["root"], "add", "--", "configs/link")
    _git(fx["root"], "commit", "-qm", "tracked symlink parent")
    fx["commit"] = _git(fx["root"], "rev-parse", "HEAD")
    with pytest.raises(fx["module"].VendorRequiredIdentityError, match="parent"):
        _run(fx)
    assert not (outside / "loop.json").exists()
