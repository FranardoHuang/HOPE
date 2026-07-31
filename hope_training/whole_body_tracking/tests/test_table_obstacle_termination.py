"""``robot_hit_table`` — a body inside the table ends the episode, a legal pose does not.

人话:ActionBall 里拍子/身体进入保守桌体禁区 = 这局结束；legacy 仍要求桌碰力；
站着挥空拍、脚踩地板 = 不结束。

HOST NOTE: needs torch, so it does NOT run on the py3.8 host.  Run it on a pod checkout (which is
a COPY of this repo)::

    python -m pytest hope_training/whole_body_tracking/tests/test_table_obstacle_termination.py -q

isaaclab is STUBBED (the same stub the other mdp behaviour tests use), so this exercises the real
shipped ``terminations.robot_hit_table`` / ``rewards.terminated_by_term`` against a fake scene
rather than a re-derivation, without needing a GPU or Isaac Sim.  The complementary check that the
collider actually EXISTS at this pose in a constructed env is
``scripts/check_table_obstacle_scene.py`` (Isaac, GPU).
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
import hashlib
import json
import os
import pathlib
import sys
import types
import xml.etree.ElementTree as ET

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = pathlib.Path(HERE).resolve().parents[2]
_TT = (REPO / "hope_training/whole_body_tracking/source/whole_body_tracking"
       / "whole_body_tracking/tasks/table_tennis")

from test_reward_flags_mdp import _PKG, _load  # noqa: E402  (installs the isaaclab stub)

MDP_DIR = str(REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
              / "whole_body_tracking" / "tasks" / "tracking" / "mdp")
sys.modules[_PKG].__path__ = [MDP_DIR]

NEAR_X = 0.5
SURFACE_Z = 0.76
MARGIN = 0.02


def _install_table_tennis_pkg():
    """Make ``whole_body_tracking.tasks.table_tennis.{geometry,table_frame}`` importable.

    ``terminations.robot_hit_table`` imports ``table_frame`` lazily inside the function, so the
    package has to resolve at CALL time.  The stub only registers the tracking mdp tree.
    """
    import importlib.util

    for pkg in ("whole_body_tracking", "whole_body_tracking.tasks",
                "whole_body_tracking.tasks.table_tennis"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    for name in ("geometry", "table_frame"):
        dotted = f"whole_body_tracking.tasks.table_tennis.{name}"
        if dotted in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(dotted, _TT / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        setattr(sys.modules["whole_body_tracking.tasks.table_tennis"], name, mod)
    return sys.modules["whole_body_tracking.tasks.table_tennis.table_frame"]


@pytest.fixture(scope="module")
def term_mod():
    _install_table_tennis_pkg()
    return _load(f"{_PKG}.terminations", "terminations.py")


@pytest.fixture(scope="module")
def rew_mod():
    return _load(f"{_PKG}.rewards", "rewards.py")


@pytest.fixture(scope="module")
def frame():
    return _install_table_tennis_pkg()


# --------------------------------------------------------------------------- the pure kernel -- #
def test_kernel_needs_both_halves(term_mod, frame):
    """Inside the box with force -> done.  Inside without force, or force outside -> not done."""
    lo, hi = frame.table_top_aabb_env(NEAR_X, SURFACE_Z, margin=MARGIN)
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    origins = torch.zeros(4, 3)

    inside = [NEAR_X + 0.30, 0.0, SURFACE_Z - 0.02]        # in the middle of the slab
    outside = [NEAR_X - 0.40, 0.0, SURFACE_Z - 0.02]       # behind the near edge (the robot's side)
    pos = torch.tensor([[inside], [inside], [outside], [outside]], dtype=torch.float32)
    force = torch.tensor([[[0.0, 0.0, 50.0]], [[0.0, 0.0, 0.0]],
                          [[0.0, 0.0, 50.0]], [[0.0, 0.0, 0.0]]], dtype=torch.float32)
    got = term_mod.table_hit_mask(pos, force, origins, lo_t, hi_t, 1.0)
    assert got.tolist() == [True, False, False, False]


def test_kernel_is_per_env_origin_relative(term_mod, frame):
    """A second env's table is at that env's own origin, not at the world origin."""
    lo, hi = frame.table_top_aabb_env(NEAR_X, SURFACE_Z, margin=MARGIN)
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, -4.0, 0.0]])
    p_local = [NEAR_X + 0.3, 0.0, SURFACE_Z - 0.02]
    pos = torch.tensor(
        [[p_local], [[p_local[0] + 10.0, p_local[1] - 4.0, p_local[2]]]], dtype=torch.float32)
    force = torch.full((2, 1, 3), 30.0)
    assert term_mod.table_hit_mask(pos, force, origins, lo_t, hi_t, 1.0).tolist() == [True, True]
    # the same WORLD point without the origin shift is outside env 1's table
    pos_bad = torch.tensor([[p_local], [p_local]], dtype=torch.float32)
    assert term_mod.table_hit_mask(pos_bad, force, origins, lo_t, hi_t, 1.0).tolist() == [True, False]


def test_kernel_threshold_is_a_strict_inequality(term_mod, frame):
    lo, hi = frame.table_top_aabb_env(NEAR_X, SURFACE_Z, margin=MARGIN)
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    pos = torch.tensor([[[NEAR_X + 0.3, 0.0, SURFACE_Z - 0.02]]], dtype=torch.float32)
    for f, want in ((0.5, False), (1.0, False), (1.5, True)):
        force = torch.tensor([[[f, 0.0, 0.0]]], dtype=torch.float32)
        assert bool(term_mod.table_hit_mask(
            pos, force, torch.zeros(1, 3), lo_t, hi_t, 1.0)) is want


def test_filtered_kernel_nonfinite_force_fails_safe(term_mod):
    force_matrix = torch.tensor(
        [
            [[[0.0, 0.0, 0.0]]],
            [[[float("nan"), 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    assert term_mod.filtered_contact_hit_mask(force_matrix, 1.0).tolist() == [False, True]


def test_body_alignment_is_by_name_not_position(term_mod):
    """Sensor body order != articulation body order must not silently mis-pair."""
    sensor_names = ["torso_Link", "right_wrist_yaw_Link", "left_wrist_yaw_Link"]
    asset_names = ["left_wrist_yaw_Link", "torso_Link", "right_wrist_yaw_Link", "pelvis_link"]
    s_ids, a_ids = term_mod.align_body_ids(
        sensor_names, asset_names, [0, 1, 2], [0, 1, 2, 3])
    assert [sensor_names[i] for i in s_ids] == [asset_names[i] for i in a_ids]
    # a selection that names nothing in common is a configuration error, not a silent empty mask
    with pytest.raises(RuntimeError, match="do not overlap"):
        term_mod.align_body_ids(["a"], ["b"], [0], [0])


def test_full_table_alignment_reorders_both_views_to_reviewed_order(term_mod):
    """Backend enumeration order must not change proxy/racket/body semantics."""

    expected = ("pelvis_link", "left_elbow_Link", "right_wrist_yaw_Link")
    sensor_names = [
        "right_wrist_yaw_Link",
        "pelvis_link",
        "left_elbow_Link",
    ]
    asset_names = [
        "left_elbow_Link",
        "right_wrist_yaw_Link",
        "pelvis_link",
    ]
    sensor_ids, asset_ids = term_mod.align_body_ids_in_expected_order(
        sensor_names,
        asset_names,
        [0, 1, 2],
        [0, 1, 2],
        expected,
    )
    assert tuple(sensor_names[index] for index in sensor_ids) == expected
    assert tuple(asset_names[index] for index in asset_ids) == expected
    with pytest.raises(RuntimeError, match="exactly cover"):
        term_mod.align_body_ids_in_expected_order(
            sensor_names,
            asset_names,
            [0, 1],
            [0, 1, 2],
            expected,
        )


def test_axiswise_point_aabb_distance_is_bitwise_dense_parity(term_mod):
    """The allocation reduction preserves the old dense arithmetic exactly."""

    point_xyz = torch.tensor(
        [
            [
                [-0.500, -0.375, 0.000],
                [0.250, 0.500, 0.875],
                [1.125, -0.750, 0.625],
            ],
            [
                [0.375, 0.125, -0.250],
                [1.500, 0.750, 1.250],
                [-1.000, -0.875, 0.500],
            ],
        ],
        dtype=torch.float32,
    )
    aabb_lo = torch.tensor(
        [
            [-0.250, -0.250, -0.125],
            [0.500, -0.625, 0.375],
            [-0.750, 0.250, 0.750],
            [1.000, -1.000, -0.500],
            [0.000, 0.000, 0.000],
        ],
        dtype=torch.float32,
    )
    aabb_hi = aabb_lo + torch.tensor(
        [0.500, 0.375, 0.250], dtype=torch.float32
    )
    point_before = point_xyz.clone()
    lo_before = aabb_lo.clone()
    hi_before = aabb_hi.clone()
    below = aabb_lo[None, None, :, :] - point_xyz[:, :, None, :]
    above = point_xyz[:, :, None, :] - aabb_hi[None, None, :, :]
    outside = torch.maximum(
        torch.maximum(below, above), torch.zeros_like(below)
    )
    dense = torch.sum(outside * outside, dim=-1)
    axiswise = term_mod._squared_distance_to_aabbs(
        point_xyz, aabb_lo, aabb_hi
    )
    assert torch.equal(axiswise, dense)
    assert torch.equal(point_xyz, point_before)
    assert torch.equal(aabb_lo, lo_before)
    assert torch.equal(aabb_hi, hi_before)


def _dense_geometric_reference(
    term_mod,
    body_pos_w,
    body_quat_w,
    env_origins,
    component_indices,
    component_centers,
    component_half_axes,
    aabb_lo,
    aabb_hi,
    racket_index,
    blade_center_offset,
    blade_local_half_axes,
):
    """Pre-optimization dense kernel retained only as a parity oracle."""

    p_local = body_pos_w - env_origins[:, None, :]
    body_norm_sq = torch.sum(body_quat_w * body_quat_w, dim=-1, keepdim=True)
    safe_body_quat = body_quat_w / torch.sqrt(
        torch.clamp(body_norm_sq, min=torch.finfo(body_pos_w.dtype).tiny)
    )
    owner_quat = safe_body_quat[:, component_indices, :]
    center = p_local[:, component_indices, :] + term_mod._quat_rotate_wxyz(
        owner_quat,
        component_centers.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1),
    )
    rotated_axes = term_mod._quat_rotate_wxyz(
        owner_quat[:, :, None, :].expand(-1, -1, 3, -1),
        component_half_axes.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1, -1),
    )
    world_half = torch.sum(torch.abs(rotated_axes), dim=2)
    component_lo = center - world_half
    component_hi = center + world_half
    component_overlap = torch.all(
        (
            component_hi[:, :, None, :]
            >= aabb_lo[None, None, :, :]
        )
        & (
            component_lo[:, :, None, :]
            <= aabb_hi[None, None, :, :]
        ),
        dim=-1,
    )
    body_hit = torch.any(component_overlap, dim=(1, 2))

    safe_quat = safe_body_quat[:, racket_index, :]
    blade_offset_w = term_mod._quat_rotate_wxyz(
        safe_quat, blade_center_offset.expand_as(p_local[:, 0])
    )
    blade_center_local = p_local[:, racket_index, :] + blade_offset_w
    blade_quat = safe_quat[:, None, :].expand(-1, 3, -1)
    rotated_half_axes = term_mod._quat_rotate_wxyz(
        blade_quat,
        blade_local_half_axes.unsqueeze(0).expand(
            body_pos_w.shape[0], -1, -1
        ),
    )
    blade_world_aabb_half = torch.sum(
        torch.abs(rotated_half_axes), dim=1
    )
    blade_lo = blade_center_local - blade_world_aabb_half
    blade_hi = blade_center_local + blade_world_aabb_half
    blade_overlap = torch.any(
        torch.all(
            (blade_hi[:, None, :] >= aabb_lo[None, :, :])
            & (blade_lo[:, None, :] <= aabb_hi[None, :, :]),
            dim=-1,
        ),
        dim=1,
    )
    racket_hit = blade_overlap
    invalid_runtime = (
        ~torch.isfinite(body_pos_w).all(dim=(1, 2))
        | ~torch.isfinite(body_quat_w).all(dim=(1, 2))
        | ~torch.isfinite(env_origins).all(dim=1)
        | ~(body_norm_sq[..., 0] > 0.0).all(dim=1)
    )
    return body_hit | racket_hit | invalid_runtime


def test_geometric_kernel_boolean_parity_and_nonfinite_fail_safe(term_mod):
    """Dense→axiswise rewrite changes no bit, including broken runtime rows."""

    body_pos = torch.tensor(
        [
            [[0.25, 0.00, 0.75], [-1.00, 0.00, 1.00]],
            [[-1.00, 0.00, 1.00], [-1.00, 0.00, 1.00]],
            [[-0.30, 0.00, 0.75], [-1.00, 0.00, 1.00]],
            [[-1.00, 0.00, 1.00], [-1.00, 0.00, 1.00]],
            [[-1.00, 0.00, 1.00], [-1.00, 0.00, 1.00]],
            [[-1.00, 0.00, 1.00], [-1.00, 0.00, 1.00]],
            [[-1.00, 0.00, 1.00], [-1.00, 0.00, 1.00]],
        ],
        dtype=torch.float32,
    )
    body_quat = torch.zeros(7, 2, 4, dtype=torch.float32)
    body_quat[..., 0] = 1.0
    env_origins = torch.zeros(7, 3, dtype=torch.float32)
    # Four independent broken-runtime channels must each fail safe.
    body_pos[3, 1, 0] = float("nan")
    body_quat[4, 1, 2] = float("inf")
    env_origins[5, 2] = float("inf")
    body_quat[6, 0, :] = 0.0

    component_indices = torch.tensor([0, 1], dtype=torch.long)
    component_centers = torch.zeros(2, 3, dtype=torch.float32)
    component_axes = torch.stack(
        (
            torch.diag(torch.tensor([0.10, 0.10, 0.10])),
            torch.diag(torch.tensor([0.10, 0.10, 0.10])),
        ),
        dim=0,
    )
    aabb_lo = torch.tensor(
        [[0.0, -0.5, 0.5], [1.0, -0.1, 0.8]],
        dtype=torch.float32,
    )
    aabb_hi = torch.tensor(
        [[0.5, 0.5, 1.0], [1.2, 0.1, 1.2]],
        dtype=torch.float32,
    )
    blade_center = torch.tensor(
        [0.35, 0.0, 0.0], dtype=torch.float32
    )
    blade_axes = torch.diag(
        torch.tensor([0.10, 0.01, 0.10], dtype=torch.float32)
    )
    dense = _dense_geometric_reference(
        term_mod,
        body_pos,
        body_quat,
        env_origins,
        component_indices,
        component_centers,
        component_axes,
        aabb_lo,
        aabb_hi,
        0,
        blade_center,
        blade_axes,
    )
    axiswise = term_mod.geometric_table_contact_hit_mask(
        body_pos,
        body_quat,
        env_origins,
        component_indices,
        component_centers,
        component_axes,
        aabb_lo,
        aabb_hi,
        racket_body_index=0,
        racket_blade_center_offset_wrist_m=blade_center,
        racket_blade_local_half_axes_m=blade_axes,
    )
    assert torch.equal(axiswise, dense)
    assert axiswise.tolist() == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_geometric_component_obb_tracks_live_body_rotation(term_mod):
    body_pos = torch.zeros(2, 1, 3, dtype=torch.float32)
    body_quat = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]],
        ],
        dtype=torch.float32,
    )
    component_indices = torch.tensor([0], dtype=torch.long)
    component_center = torch.zeros(1, 3, dtype=torch.float32)
    component_axes = torch.diag(
        torch.tensor([0.20, 0.02, 0.02], dtype=torch.float32)
    ).unsqueeze(0)
    aabb_lo = torch.tensor([[0.15, -0.05, -0.05]], dtype=torch.float32)
    aabb_hi = torch.tensor([[0.25, 0.05, 0.05]], dtype=torch.float32)
    far_blade = torch.tensor([-10.0, 0.0, 0.0], dtype=torch.float32)
    blade_axes = torch.diag(torch.full((3,), 0.01))
    got = term_mod.geometric_table_contact_hit_mask(
        body_pos,
        body_quat,
        torch.zeros(2, 3),
        component_indices,
        component_center,
        component_axes,
        aabb_lo,
        aabb_hi,
        racket_body_index=0,
        racket_blade_center_offset_wrist_m=far_blade,
        racket_blade_local_half_axes_m=blade_axes,
    )
    assert got.tolist() == [True, False]


def test_configured_exact_pair_body_table_matches_shipped_urdf_rigid_order():
    """The whole-Robot filter contract covers root + every child, including both feet."""

    cfg_path = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking"
        / "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    )
    cfg_tree = ast.parse(
        cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path)
    )
    assignment = next(
        node
        for node in cfg_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "TABLE_CONTACT_BODY_NAMES"
            for target in node.targets
        )
    )
    configured = tuple(ast.literal_eval(assignment.value))

    urdf_path = REPO / "agi/URDF/a3_t2d5/urdf/model.urdf"
    urdf = ET.parse(urdf_path).getroot()
    child_links = {
        joint.find("child").attrib["link"] for joint in urdf.findall("joint")
    }
    root_names = [
        link.attrib["name"]
        for link in urdf.findall("link")
        if link.attrib["name"] not in child_links
    ]
    assert root_names == ["pelvis_link"]
    nonfixed_children = defaultdict(list)
    for joint in urdf.findall("joint"):
        if joint.attrib["type"] == "fixed":
            continue
        nonfixed_children[joint.find("parent").attrib["link"]].append(
            joint.find("child").attrib["link"]
        )
    # The importer/PhysX articulation table is a breadth-first traversal with sibling prim names
    # sorted lexically.  A tracked training contract from this same URDF has this exact order; this
    # independent derivation catches both omissions and accidental source-file-order assumptions.
    queue = deque(root_names)
    runtime_order = []
    while queue:
        body_name = queue.popleft()
        runtime_order.append(body_name)
        queue.extend(sorted(nonfixed_children[body_name]))
    assert configured == tuple(runtime_order)
    assert len(configured) == 32
    assert {"left_ankle_roll_Link", "right_ankle_roll_Link"} <= set(configured)
    assert configured.count("right_wrist_yaw_Link") == 1


def test_collision_proxy_artifact_binds_43_components_32_bodies_and_pinned_usd(
    term_mod,
):
    artifact = REPO / COLLISION_PROXY_PATH
    payload = artifact.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == COLLISION_PROXY_SHA256
    document = json.loads(payload)
    assert document["component_count"] == 43
    assert tuple(document["body_order"]) == tuple(BODIES)
    assert {
        component["owner_body_name"]
        for component in document["components"]
    } == set(BODIES)
    assert document["source_urdf"]["sha256"] == (
        "0d83529cf808e2e68036f8168bd8b7a1c9a97d9c536eb9a14981ea4105d6b9ae"
    )
    assert document["runtime_usd_bundle"]["bundle_tree_sha256"] == (
        "716487dfdf02a5973f78263f0ae8a09e4680c04159e57dbe20796b7825dbeb4d"
    )
    owners, centers, axes = term_mod._load_table_collision_proxy_artifact(
        str(artifact),
        COLLISION_PROXY_SHA256,
        tuple(BODIES),
    )
    assert len(owners) == len(centers) == len(axes) == 43
    assert set(owners) == set(range(32))
    assert sum(
        document["body_order"][owner] == "right_wrist_yaw_Link"
        for owner in owners
    ) >= 5


def test_collision_proxy_artifact_rejects_runtime_usd_pin_drift(
    term_mod, tmp_path
):
    document = json.loads((REPO / COLLISION_PROXY_PATH).read_text())
    document["runtime_usd_bundle"]["bundle_tree_sha256"] = "0" * 64
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    document["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    payload = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    artifact = tmp_path / "drifted_collision_proxy.json"
    artifact.write_bytes(payload)
    with pytest.raises(RuntimeError, match="six-file Pod runtime USD"):
        term_mod._load_table_collision_proxy_artifact(
            str(artifact),
            hashlib.sha256(payload).hexdigest(),
            tuple(BODIES),
        )


def test_live_runtime_usd_validator_binds_exact_tree_once(
    term_mod, tmp_path, monkeypatch
):
    root = tmp_path / "runtime_usd"
    payloads = {
        ".asset_hash": b"asset",
        "config.yaml": b"config",
        "configuration/model_base.usd": b"base",
        "configuration/model_physics.usd": b"physics",
        "configuration/model_sensor.usd": b"sensor",
        "model.usd": b"root",
    }
    expected = []
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected.append(
            (relative, hashlib.sha256(payload).hexdigest(), len(payload))
        )
    expected = tuple(sorted(expected))
    entries = [
        {"path": path, "sha256": sha256, "size": size}
        for path, sha256, size in expected
    ]
    tree_sha256 = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    monkeypatch.setattr(
        term_mod, "_A3_COLLISION_PROXY_RUNTIME_USD_FILES", expected
    )
    monkeypatch.setattr(
        term_mod, "_A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256", tree_sha256
    )
    term_mod._verify_loaded_runtime_usd_bundle.cache_clear()
    model_path = str((root / "model.usd").resolve())
    assert term_mod._verify_loaded_runtime_usd_bundle(model_path) == tree_sha256

    # The cached identity does not re-walk or re-hash a stable run asset.
    (root / "extra.usd").write_bytes(b"unexpected")
    assert term_mod._verify_loaded_runtime_usd_bundle(model_path) == tree_sha256
    term_mod._verify_loaded_runtime_usd_bundle.cache_clear()
    with pytest.raises(RuntimeError, match="exact six-file pin"):
        term_mod._verify_loaded_runtime_usd_bundle(model_path)
    term_mod._verify_loaded_runtime_usd_bundle.cache_clear()


def test_live_articulation_usd_path_rejects_split_identity(
    term_mod, tmp_path, monkeypatch
):
    asset_model = tmp_path / "asset" / "model.usd"
    environment_model = tmp_path / "environment" / "model.usd"
    asset_model.parent.mkdir()
    environment_model.parent.mkdir()
    asset_model.write_bytes(b"asset")
    environment_model.write_bytes(b"environment")
    asset = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            spawn=types.SimpleNamespace(usd_path=str(asset_model))
        )
    )
    env = types.SimpleNamespace(cfg=None)
    monkeypatch.setenv(
        "HOPE_AGIBOT_A3_USD_PATH", str(environment_model.resolve())
    )
    with pytest.raises(RuntimeError, match="launch environment pin"):
        term_mod._live_articulation_model_usd_path(env, asset)


def test_action_ball_reuses_whole_body_sensor_without_pair_filtered_views():
    cfg_path = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking"
        / "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    )
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    assignments = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "TABLE_CONTACT_SENSOR_NAME",
                "TABLE_FULL_CONTACT_SENSOR_NAMES",
                "TABLE_CONTACT_BODY_NAMES",
            }:
                assignments[target.id] = ast.literal_eval(node.value)
    wrist_sensor_name = assignments["TABLE_CONTACT_SENSOR_NAME"]
    sensor_names = tuple(assignments["TABLE_FULL_CONTACT_SENSOR_NAMES"])
    body_names = tuple(assignments["TABLE_CONTACT_BODY_NAMES"])
    assert len(body_names) == 32

    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "attach_table_contact_sensor"
    )

    class FakeContactSensorCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    namespace = {
        "ContactSensorCfg": FakeContactSensorCfg,
        "TABLE_CONTACT_SENSOR_NAME": wrist_sensor_name,
        "TABLE_CONTACT_SENSOR_PRIM": (
            "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link"
        ),
        "TABLE_FULL_CONTACT_SENSOR_NAMES": sensor_names,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(cfg_path),
            "exec",
        ),
        namespace,
    )
    five_obstacles = tuple(f"{{ENV_REGEX_NS}}/TablePart{index}" for index in range(5))
    namespace["TABLE_FULL_CONTACT_SENSOR_PRIMS"] = five_obstacles
    env_cfg = types.SimpleNamespace(
        table_robot_keepout=True,
        table_obstacle_prims=five_obstacles,
        table_pair_contact_sensor_names=(
            wrist_sensor_name,
            *sensor_names,
        ),
        scene=types.SimpleNamespace(
            **{
                name: object()
                for name in (wrist_sensor_name, *sensor_names)
            }
        ),
    )
    namespace["attach_table_contact_sensor"](env_cfg)
    assert env_cfg.table_pair_contact_sensor_names == ()
    for stale_name in (wrist_sensor_name, *sensor_names):
        assert getattr(env_cfg.scene, stale_name) is None

    # A late full→legacy override recreates only the historic one-body wrist source.
    env_cfg.table_robot_keepout = False
    env_cfg.table_obstacle_prims = five_obstacles[:1]
    namespace["attach_table_contact_sensor"](env_cfg)
    assert env_cfg.table_pair_contact_sensor_names == (wrist_sensor_name,)
    assert tuple(
        getattr(env_cfg.scene, wrist_sensor_name).filter_prim_paths_expr
    ) == five_obstacles[:1]
    for stale_name in set(sensor_names) - {wrist_sensor_name}:
        assert getattr(env_cfg.scene, stale_name) is None


def test_action_ball_table_parts_are_kinematic_and_own_no_contact_reporters():
    """Only Robot/contact_forces reports; table boxes create no extra GPU sensor views."""

    cfg_path = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking"
        / "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    )
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "attach_table_obstacle"
    )
    cuboid_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "CuboidCfg"
    ]
    assert len(cuboid_calls) == 4
    for call in cuboid_calls:
        rigid_keyword = next(
            (
                keyword
                for keyword in call.keywords
                if keyword.arg == "rigid_props"
            ),
            None,
        )
        assert rigid_keyword is not None
        assert isinstance(rigid_keyword.value, ast.Call)
        assert isinstance(rigid_keyword.value.func, ast.Name)
        assert rigid_keyword.value.func.id == "robot_proxy_rigid_props"
        reporter_keyword = next(
            (
                keyword
                for keyword in call.keywords
                if keyword.arg == "activate_contact_sensors"
            ),
            None,
        )
        assert reporter_keyword is not None
        assert isinstance(reporter_keyword.value, ast.Constant)
        assert reporter_keyword.value.value is False


# ------------------------------------------------------------------- the termination on an env - #
class _Data:
    def __init__(self, forces, pos, force_matrix=None, quat=None):
        self.net_forces_w = forces
        self.body_pos_w = pos
        self.body_quat_w = quat
        self.force_matrix_w = force_matrix


class _Sensor:
    def __init__(self, names, forces):
        self.body_names = list(names)
        self.data = _Data(forces, None)


class _FilteredSensor:
    def __init__(self, force_matrix, source_prim, filter_prims):
        self.body_names = [source_prim.rsplit("/", 1)[-1]]
        self.cfg = types.SimpleNamespace(
            prim_path=source_prim,
            filter_prim_paths_expr=list(filter_prims),
        )
        self.data = _Data(None, None, force_matrix)


class _Asset:
    def __init__(self, names, pos):
        self.body_names = list(names)
        quat = torch.zeros((*pos.shape[:2], 4), dtype=pos.dtype)
        quat[..., 0] = 1.0
        self.data = _Data(None, pos, quat=quat)
        runtime_usd_path = os.environ.get(
            "HOPE_AGIBOT_A3_USD_PATH",
            str(REPO / COLLISION_PROXY_PATH),
        )
        self.cfg = types.SimpleNamespace(
            spawn=types.SimpleNamespace(usd_path=runtime_usd_path)
        )


class _Scene:
    def __init__(self, sensor, filtered_sensors, asset, origins):
        self.sensors = {
            "contact_forces": sensor,
            **filtered_sensors,
        }
        self._assets = {"robot": asset}
        self.env_origins = origins

    def __getitem__(self, key):
        return self._assets[key]


class _Env:
    def __init__(self, scene):
        self.scene = scene
        self.num_envs = int(scene.env_origins.shape[0])


class _Cfg:
    def __init__(self, name, body_ids):
        self.name = name
        self.body_ids = body_ids


BODIES = [
    "pelvis_link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "waist_yaw_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "waist_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "head_yaw_Link",
    "left_shoulder_pitch_Link",
    "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "head_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
]
LOGICAL_BODIES = (
    "pelvis_link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
    "left_ankle_roll_Link",
)
WATCHED = [BODIES.index(name) for name in LOGICAL_BODIES[:3]]
COLLISION_PROXY_PATH = (
    "configs/a3_table_collision_proxy_20260731/"
    "a3_table_collision_components.v1.json"
)
COLLISION_PROXY_SHA256 = (
    "23e2f5b30bbba909f1123dc41f6c010354122b9837b4ef133a1c285a2cd78ca8"
)
EXACT_SENSOR_NAMES = [
    "table_top_robot_contact",
    "table_keepout_robot_contact",
    "table_net_robot_contact",
    "table_post_left_robot_contact",
    "table_post_right_robot_contact",
]
EXACT_SOURCE_PRIMS = tuple(
    f"{{ENV_REGEX_NS}}/TablePart{index}" for index in range(5)
)
EXACT_ROLES = ("top", "keepout", "net", "post_left", "post_right")
ROBOT_FILTER_PRIMS = tuple(
    f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in BODIES
)


def _env(
    pos,
    force,
    filtered_force=None,
    *,
    robot_filter_count=32,
    exact_role_forces=None,
):
    pos_tensor = torch.tensor(pos, dtype=torch.float32)
    force_tensor = torch.tensor(force, dtype=torch.float32)
    if pos_tensor.shape[1] > len(LOGICAL_BODIES):
        raise ValueError("test fixture supplied more than four logical A3 bodies")
    body_pos = torch.zeros(
        pos_tensor.shape[0], len(BODIES), 3, dtype=pos_tensor.dtype
    )
    body_pos[..., 2] = 1.5
    body_force = torch.zeros(
        force_tensor.shape[0], len(BODIES), 3, dtype=force_tensor.dtype
    )
    for logical_index in range(pos_tensor.shape[1]):
        body_index = BODIES.index(LOGICAL_BODIES[logical_index])
        body_pos[:, body_index, :] = pos_tensor[:, logical_index, :]
        body_force[:, body_index, :] = force_tensor[:, logical_index, :]
    pos_tensor = body_pos
    force_tensor = body_force
    sensor = _Sensor(BODIES, force_tensor)
    if filtered_force is None:
        filtered_force = torch.zeros(len(pos), 1, 1, 3)
    else:
        filtered_force = torch.as_tensor(filtered_force, dtype=torch.float32)
    exact_role_forces = exact_role_forces or {}
    filtered_sensors = {
        "racket_table_contact": _FilteredSensor(
            filtered_force,
            "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link",
            (EXACT_SOURCE_PRIMS[0],),
        )
    }
    for role, sensor_name, source_prim in zip(
        EXACT_ROLES, EXACT_SENSOR_NAMES, EXACT_SOURCE_PRIMS
    ):
        matrix = exact_role_forces.get(role)
        if matrix is None:
            matrix = torch.zeros(
                len(pos), 1, robot_filter_count, 3
            )
        filtered_sensors[sensor_name] = _FilteredSensor(
            torch.as_tensor(matrix, dtype=torch.float32),
            source_prim,
            ROBOT_FILTER_PRIMS,
        )
    asset = _Asset(BODIES, pos_tensor)
    return _Env(
        _Scene(
            sensor,
            filtered_sensors,
            asset,
            torch.zeros(len(pos), 3),
        )
    )


def _call(term_mod, env, **overrides):
    params = {
        "near_x": NEAR_X,
        "surface_z": SURFACE_Z,
        "force_threshold": 1.0,
        "margin": MARGIN,
    }
    params.update(overrides)
    if params.get("full_table_assembly"):
        params.setdefault("full_table_filtered_sensor_cfgs", ())
        params.setdefault(
            "expected_full_table_source_prim_paths",
            EXACT_SOURCE_PRIMS,
        )
        params.setdefault("expected_full_robot_body_names", tuple(BODIES))
        params.setdefault(
            "collision_proxy_artifact_path", COLLISION_PROXY_PATH
        )
        params.setdefault(
            "collision_proxy_artifact_sha256",
            COLLISION_PROXY_SHA256,
        )
        params.setdefault("racket_body_name", "right_wrist_yaw_Link")
        params.setdefault(
            "racket_blade_center_offset_wrist_m",
            (0.206194, 0.025474, 0.028020),
        )
        params.setdefault(
            "racket_blade_half_extents_m", (0.082, 0.008, 0.082)
        )
    ids = list(range(len(BODIES))) if params.get("full_table_assembly") else WATCHED
    original_verify = term_mod._verify_loaded_runtime_usd_bundle
    if params.get("full_table_assembly"):
        # Runtime USD bytes are launch/Pod evidence.  These host-only geometry
        # tests explicitly stub only that one-time receipt while retaining the
        # real artifact, body-name, dtype and pose validation.
        term_mod._verify_loaded_runtime_usd_bundle = (
            lambda _path: (
                "716487dfdf02a5973f78263f0ae8a09e4680c04159e57dbe20796b7825dbeb4d"
            )
        )
    try:
        return term_mod.robot_hit_table(
            env, _Cfg("contact_forces", ids), _Cfg("racket_table_contact", [0]),
            _Cfg("robot", ids),
            **params,
        )
    finally:
        term_mod._verify_loaded_runtime_usd_bundle = original_verify


def test_racket_inside_the_table_terminates(term_mod, frame):
    """The exact case that motivated this: a racket commanded to z ~ 0.65-0.69 over the table."""
    standing = [0.0, 0.0, 1.0]
    torso = [0.0, 0.0, 1.1]
    racket_in_table = [NEAR_X + 0.25, 0.10, 0.70]   # over the table, below the surface
    pos = [[standing, torso, racket_in_table, [0.0, 0.1, 0.05]]]
    force = [[[0, 0, 0], [0, 0, 0], [0.0, 0.0, 120.0], [0.0, 0.0, 400.0]]]
    assert bool(_call(term_mod, _env(pos, force))) is True


def test_filtered_racket_contact_terminates_with_wrist_origin_outside_table(term_mod):
    """The 21 cm racket offset may touch the near edge while the wrist origin is still outside."""
    wrist_before_near_edge = [NEAR_X - 0.10, 0.10, SURFACE_Z]
    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], wrist_before_near_edge,
            [0.0, 0.1, 0.05]]]
    # The broad stream sees the same contact but cannot attribute it to the table, and its wrist
    # origin correctly fails the table AABB.  The filtered pair supplies that missing identity.
    broad_force = [[[0, 0, 0], [0, 0, 0], [0.0, 0.0, 120.0], [0, 0, 0]]]
    filtered_force = [[[[0.0, 0.0, 120.0]]]]
    assert bool(_call(term_mod, _env(pos, broad_force, filtered_force))) is True


def test_wrist_origin_outside_without_filtered_contact_does_not_terminate(term_mod):
    wrist_before_near_edge = [NEAR_X - 0.10, 0.10, SURFACE_Z]
    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], wrist_before_near_edge,
            [0.0, 0.1, 0.05]]]
    broad_force = [[[0, 0, 0], [0, 0, 0], [0.0, 0.0, 120.0], [0, 0, 0]]]
    assert bool(_call(term_mod, _env(pos, broad_force))) is False


def test_a_legal_swing_does_not_terminate(term_mod, frame):
    """Racket above the table with the feet loaded on the floor: nothing fires."""
    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [NEAR_X + 0.25, 0.10, 1.05],
            [0.0, 0.1, 0.05]]]
    force = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0.0, 0.0, 400.0]]]
    assert bool(_call(term_mod, _env(pos, force))) is False


def test_falling_onto_the_floor_is_not_a_table_hit(term_mod, frame):
    """An arm slamming the FLOOR behind the near edge belongs to the fall guards, not to this one.

    This is the discrimination the geometric half exists for: the contact force is large and it is
    on a watched body, but the body is not in the table.
    """
    pos = [[[0.0, 0.0, 0.3], [0.1, 0.0, 0.35], [0.2, 0.4, 0.06], [0.0, 0.1, 0.05]]]
    force = [[[0, 0, 200.0], [0, 0, 150.0], [0.0, 0.0, 300.0], [0.0, 0.0, 50.0]]]
    assert bool(_call(term_mod, _env(pos, force))) is False


def test_under_the_slab_is_the_documented_gap(term_mod, frame):
    """Legacy top-only mode keeps its documented under-slab behavior."""
    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [NEAR_X + 0.25, 0.0, 0.60], [0.0, 0.1, 0.05]]]
    force = [[[0, 0, 0], [0, 0, 0], [0.0, 0.0, 120.0], [0.0, 0.0, 400.0]]]
    assert bool(_call(term_mod, _env(pos, force))) is False


def test_action_ball_keepout_catches_under_slab_contact(term_mod):
    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1],
            [NEAR_X + 0.25, 0.0, 0.60], [0.0, 0.1, 0.05]]]
    force = [[[0, 0, 0], [0, 0, 0], [0, 0, 120.0], [0, 0, 0]]]
    assert bool(
        _call(
            term_mod,
            _env(pos, force),
            full_table_assembly=True,
            keepout_floor_z=0.0,
        )
    ) is True


@pytest.mark.parametrize(
    ("role", "point"),
    [
        # Top and near-side edge.
        ("top", [NEAR_X + 0.30, 0.0, SURFACE_Z - 0.01]),
        ("top", [NEAR_X, 0.0, SURFACE_Z - 0.025]),
        # Net center.
        ("net", [NEAR_X + 1.37, 0.0, SURFACE_Z + 0.07]),
        # Left/right post centers.
        (
            "post_left",
            [NEAR_X + 1.37, 0.7625 + 0.15, SURFACE_Z + 0.08],
        ),
        (
            "post_right",
            [NEAR_X + 1.37, -(0.7625 + 0.15), SURFACE_Z + 0.08],
        ),
    ],
)
def test_full_assembly_geometric_channel_covers_top_edge_net_and_posts(
    term_mod, role, point
):
    pos = [[[0.0, 0.0, 1.0], point, [0.0, 0.0, 1.1], [0.0, 0.1, 0.05]]]
    broad_force = [[[0, 0, 0]] * 4]
    assert bool(
        _call(
            term_mod,
            _env(pos, broad_force),
            full_table_assembly=True,
        )
    ) is True


def test_elbow_proxy_catches_contact_with_origin_outside_table_aabb(
    term_mod,
):
    """The materialized elbow component covers its shipped hull past the body origin."""

    elbow_origin_far_from_table = [NEAR_X - 0.15, 0.0, SURFACE_Z]
    pos = [[
        [0.0, 0.0, 1.0],
        elbow_origin_far_from_table,
        [0.0, 0.0, 1.1],
        [0.0, 0.1, 0.05],
    ]]
    broad_force = [[[0, 0, 0]] * 4]
    assert bool(
        _call(
            term_mod,
            _env(pos, broad_force),
            full_table_assembly=True,
        )
    ) is True


def test_exact_elbow_component_avoids_old_18cm_sphere_false_positive(
    term_mod,
):
    """A 0.18 m origin sphere hit here; the shipped elbow hull still clears the table."""

    elbow_origin = [NEAR_X - 0.18, 0.0, SURFACE_Z]
    pos = [[
        [0.0, 0.0, 1.0],
        elbow_origin,
        [0.0, 0.0, 1.1],
        [0.0, 0.1, 0.05],
    ]]
    broad_force = [[[0, 0, 0]] * 4]
    assert bool(
        _call(
            term_mod,
            _env(pos, broad_force),
            full_table_assembly=True,
        )
    ) is False


def test_full_assembly_pose_overlap_does_not_require_contact_force(
    term_mod,
):
    pos = [[
        [0.0, 0.0, 1.0],
        [NEAR_X + 0.3, 0.0, SURFACE_Z],
        [0.0, 0.0, 1.1],
        [0.0, 0.1, 0.05],
    ]]
    broad_force = [[[0, 0, 0]] * 4]
    assert bool(
        _call(
            term_mod,
            _env(pos, broad_force),
            full_table_assembly=True,
        )
    ) is True


def test_full_assembly_explicit_robot_contract_includes_feet(term_mod):
    """A foot under the table is illegal, while a loaded foot at the stance is not."""

    pos = [[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.1],
        [0.0, 0.0, 1.1],
        [NEAR_X + 0.2, 0.0, 0.05],
    ]]
    broad_force = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 400.0]]]
    assert bool(
        _call(
            term_mod,
            _env(pos, broad_force),
            full_table_assembly=True,
        )
    ) is True


def test_full_assembly_loaded_feet_far_from_table_do_not_misreport(term_mod):
    pos = [[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.1],
        [0.0, 0.0, 1.1],
        [0.0, 0.1, 0.05],
    ]]
    broad_force = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 400.0]]]
    assert bool(
        _call(
            term_mod,
            _env(pos, broad_force),
            full_table_assembly=True,
        )
    ) is False


def test_full_assembly_needs_no_pair_filtered_sensor(term_mod):
    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    env = _env(pos, broad_force)

    class ForbiddenSensorRegistry:
        def __getitem__(self, _name):
            raise AssertionError(
                "full pose-only table guard touched ContactSensor registry"
            )

    env.scene.sensors = ForbiddenSensorRegistry()
    assert bool(_call(term_mod, env, full_table_assembly=True)) is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"collision_proxy_artifact_sha256": "not-a-sha"},
            "collision proxy SHA",
        ),
        (
            {
                "racket_blade_center_offset_wrist_m": (
                    float("inf"),
                    0.0,
                    0.0,
                )
            },
            "racket blade geometry",
        ),
        (
            {"racket_blade_half_extents_m": (0.082, 0.0, 0.082)},
            "racket blade geometry",
        ),
    ],
)
def test_full_assembly_static_geometry_fails_at_cache_construction(
    term_mod, overrides, message
):
    """Run-invariant geometry is rejected once, before the hot tensor kernel."""

    env = _env([[[0.0, 0.0, 1.0]] * 4], [[[0, 0, 0]] * 4])
    with pytest.raises(RuntimeError, match=message):
        _call(term_mod, env, full_table_assembly=True, **overrides)


def test_full_assembly_rejects_invalid_cached_table_boxes(
    term_mod, frame, monkeypatch
):
    env = _env([[[0.0, 0.0, 1.0]] * 4], [[[0, 0, 0]] * 4])
    invalid_boxes = (
        ((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0)),
    ) * 5
    monkeypatch.setattr(
        frame,
        "table_assembly_aabbs_env",
        lambda *_args, **_kwargs: invalid_boxes,
    )
    with pytest.raises(RuntimeError, match="five finite ordered boxes"):
        _call(term_mod, env, full_table_assembly=True)


def test_full_assembly_caches_component_geometry_and_blade_local_axes(term_mod):
    """Artifact parsing/tensor materialization occurs only on the first sample."""

    env = _env([[[0.0, 0.0, 1.0]] * 4], [[[0, 0, 0]] * 4])
    assert bool(_call(term_mod, env, full_table_assembly=True)) is False
    asset = env.scene["robot"]
    cached = asset._hope_table_geometric_guard_cache
    prepared = cached[1]
    component_indices = prepared._component_indices
    component_centers = prepared._component_centers
    component_local_axes = prepared._component_half_axes
    blade_local_axes = prepared._blade_local_half_axes
    expected_axes = torch.diag(
        torch.tensor([0.082, 0.008, 0.082], dtype=torch.float32)
    )
    assert component_indices.shape == (43,)
    assert component_centers.shape == (43, 3)
    assert component_local_axes.shape == (43, 3, 3)
    assert torch.equal(blade_local_axes, expected_axes)

    component_ptr = component_local_axes.data_ptr()
    axes_ptr = blade_local_axes.data_ptr()
    assert bool(_call(term_mod, env, full_table_assembly=True)) is False
    reused = asset._hope_table_geometric_guard_cache
    assert reused[1]._component_half_axes.data_ptr() == component_ptr
    assert reused[1]._blade_local_half_axes.data_ptr() == axes_ptr


@pytest.mark.parametrize(("field", "value"), [("pos", float("nan")), ("quat", float("inf"))])
def test_full_assembly_nonfinite_pose_fails_safe(term_mod, field, value):
    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    env = _env(pos, broad_force)
    if field == "pos":
        env.scene["robot"].data.body_pos_w[0, 1, 0] = value
    else:
        env.scene["robot"].data.body_quat_w[0, 1, 0] = value
    assert bool(
        _call(
            term_mod,
            env,
            full_table_assembly=True,
        )
    ) is True


def test_full_assembly_rejects_articulation_body_name_drift(term_mod):
    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    env = _env(pos, broad_force)
    env.scene["robot"].body_names[0] = "forged_body"
    with pytest.raises(RuntimeError, match="name-bijective"):
        _call(term_mod, env, full_table_assembly=True)


def test_full_assembly_live_body_order_is_name_mapped_not_traversal_order(
    term_mod,
):
    pos = [[
        [0.0, 0.0, 1.0],
        [NEAR_X + 0.3, 0.0, SURFACE_Z],
        [0.0, 0.0, 1.1],
        [0.0, 0.1, 0.05],
    ]]
    env = _env(pos, [[[0, 0, 0]] * 4])
    asset = env.scene["robot"]
    permutation = list(reversed(range(len(BODIES))))
    asset.body_names = [asset.body_names[index] for index in permutation]
    asset.data.body_pos_w = asset.data.body_pos_w[:, permutation, :]
    asset.data.body_quat_w = asset.data.body_quat_w[:, permutation, :]
    assert bool(_call(term_mod, env, full_table_assembly=True)) is True


def test_live_racket_blade_obb_catches_offset_contact(term_mod):
    """Blade touches the near edge while the wrist's joint-side component remains outside."""

    wrist = [NEAR_X - 0.19, 0.0, SURFACE_Z]
    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], wrist, [0.0, 0.1, 0.05]]]
    force = [[[0, 0, 0], [0, 0, 0], [0, 0, 120.0], [0, 0, 0]]]
    assert bool(
        _call(
            term_mod,
            _env(pos, force),
            full_table_assembly=True,
        )
    ) is True


def test_done_term_consumes_substep_latch_without_resampling(term_mod):
    env = _env([[[0.0, 0.0, 1.0]] * 4], [[[0, 0, 0]] * 4])
    env.num_envs = 1
    calls = []
    action = types.SimpleNamespace(
        finalize_table_contact_substep_readback=lambda: (
            calls.append("finalize") or torch.tensor([True])
        )
    )
    env.action_manager = types.SimpleNamespace(
        get_term=lambda name: action if name == "joint_pos" else None
    )
    assert bool(
        _call(
            term_mod,
            env,
            require_substep_latch=True,
            action_name="joint_pos",
        )
    ) is True
    assert calls == ["finalize"]


def test_missing_force_stream_fails_loud(term_mod):
    env = _env([[[0.0, 0.0, 1.0]] * 4], [[[0, 0, 0]] * 4])
    env.scene.sensors["contact_forces"].data.net_forces_w = None
    with pytest.raises(RuntimeError, match="net_forces_w"):
        _call(term_mod, env)


def test_missing_filtered_force_stream_fails_loud(term_mod):
    env = _env([[[0.0, 0.0, 1.0]] * 4], [[[0, 0, 0]] * 4])
    env.scene.sensors["racket_table_contact"].data.force_matrix_w = None
    with pytest.raises(RuntimeError, match="force_matrix_w"):
        _call(term_mod, env)


def test_apply_table_obstacle_only_binds_scene_entities_installed_in_each_mode():
    """Manager eagerly resolves every SceneEntityCfg, including compatibility-only params."""

    cfg_path = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking"
        / "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    )
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_table_obstacle"
    )

    class FakeSceneEntityCfg:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)

    sensor_names = tuple(EXACT_SENSOR_NAMES)
    namespace = {
        "TABLE_CONTACT_SENSOR_NAME": "racket_table_contact",
        "TABLE_CONTACT_BODY_NAMES": tuple(BODIES),
        "TABLE_FULL_CONTACT_SENSOR_NAMES": sensor_names,
        "A3_NON_FOOT_BODY_REGEX": "non_foot_regex",
        "SceneEntityCfg": FakeSceneEntityCfg,
        "attach_table_obstacle": lambda _cfg: None,
        "attach_table_contact_sensor": lambda _cfg: None,
        "table_hit_done_term": lambda: None,
        "table_hit_rew_term": lambda: None,
    }
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            str(cfg_path),
            "exec",
        ),
        namespace,
    )

    def make_cfg(*, full):
        term = types.SimpleNamespace(params={})
        return types.SimpleNamespace(
            table_obstacle=True,
            table_robot_keepout=full,
            table_obstacle_prims=EXACT_SOURCE_PRIMS if full else EXACT_SOURCE_PRIMS[:1],
            decimation=4,
            scene=types.SimpleNamespace(),
            terminations=types.SimpleNamespace(robot_hit_table=term),
            rewards=types.SimpleNamespace(table_hit_penalty=object()),
            commands=types.SimpleNamespace(
                racket_target=types.SimpleNamespace(
                    vb_table_near_x=NEAR_X,
                    vb_table_surface_z=SURFACE_Z,
                )
            ),
            actions=types.SimpleNamespace(
                joint_pos=types.SimpleNamespace()
            ),
        )

    full_cfg = make_cfg(full=True)
    namespace["apply_table_obstacle"](full_cfg)
    full_params = full_cfg.terminations.robot_hit_table.params
    assert full_params["sensor_cfg"].name == "contact_forces"
    assert tuple(full_params["sensor_cfg"].body_names) == tuple(BODIES)
    assert full_params["asset_cfg"].name == "robot"
    assert full_params["filtered_sensor_cfg"].name == "contact_forces"
    assert full_params["full_table_filtered_sensor_cfgs"] == ()
    assert (
        tuple(full_params["expected_full_robot_body_names"])
        == tuple(BODIES)
    )

    legacy_cfg = make_cfg(full=False)
    namespace["apply_table_obstacle"](legacy_cfg)
    legacy_params = legacy_cfg.terminations.robot_hit_table.params
    assert legacy_params["filtered_sensor_cfg"].name == "racket_table_contact"
    assert legacy_params["full_table_filtered_sensor_cfgs"] == ()
    assert legacy_params["expected_full_robot_body_names"] == ()


def test_table_disabled_removes_filtered_sensor_with_other_table_parts():
    """Execute the shipped off branch against a mock cfg; no stale sensor may survive the table."""
    cfg_path = (REPO / "hope_training/whole_body_tracking/source/whole_body_tracking"
                / "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py")
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_table_obstacle"
    )
    namespace = {"TABLE_CONTACT_SENSOR_NAME": "racket_table_contact"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(cfg_path), "exec"), namespace)

    scene = types.SimpleNamespace(
        table_obstacle=object(),
        table_obstacle_visual=object(),
        racket_table_contact=object(),
    )
    env_cfg = types.SimpleNamespace(
        table_obstacle=False,
        table_obstacle_prim="{ENV_REGEX_NS}/TableObstacle",
        scene=scene,
        terminations=types.SimpleNamespace(robot_hit_table=object()),
        rewards=types.SimpleNamespace(table_hit_penalty=object()),
    )
    namespace["apply_table_obstacle"](env_cfg)
    assert scene.table_obstacle is None
    assert scene.table_obstacle_visual is None
    assert scene.racket_table_contact is None
    assert env_cfg.terminations.robot_hit_table is None
    assert env_cfg.rewards.table_hit_penalty is None
    assert env_cfg.table_obstacle_prim == ""


# ------------------------------------------------------------------------------ the penalty --- #
class _TM:
    def __init__(self, terms):
        self._terms = terms
        self.active_terms = tuple(terms)

    def get_term(self, name):
        return self._terms[name]


def test_penalty_charges_only_its_own_termination(rew_mod):
    env = types.SimpleNamespace(termination_manager=_TM({
        "robot_hit_table": torch.tensor([True, False, False]),
        "base_fell_tilt": torch.tensor([False, True, False]),
    }))
    got = rew_mod.terminated_by_term(env, "robot_hit_table")
    assert got.tolist() == [1.0, 0.0, 0.0]
    assert got.dtype == torch.float32


def test_penalty_on_a_missing_termination_fails_loud(rew_mod):
    """A silently-zero penalty would be indistinguishable from "the robot never hits the table"."""
    env = types.SimpleNamespace(termination_manager=_TM({"base_fell_tilt": torch.tensor([False])}))
    with pytest.raises(RuntimeError, match="not active"):
        rew_mod.terminated_by_term(env, "robot_hit_table")
