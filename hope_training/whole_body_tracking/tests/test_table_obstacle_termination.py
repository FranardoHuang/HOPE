"""``robot_hit_table`` — a body inside the table ends the episode, a legal pose does not.

人话:拍子/胳膊进了桌子里且有接触力 = 这局结束;站着挥空拍、脚踩地板 = 不结束。

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


def test_configured_exact_pair_body_table_matches_shipped_urdf_rigid_order():
    """The one-body sensors cover root + every non-fixed child, including both feet."""

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

    urdf_path = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking"
        / "whole_body_tracking/assets/agibot_a3/urdf/model.urdf"
    )
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


def test_action_ball_attaches_one_robot_only_five_filter_sensor_per_body():
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
                "TABLE_CONTACT_BODY_NAMES",
                "TABLE_CONTACT_SENSOR_NAME",
            }:
                assignments[target.id] = ast.literal_eval(node.value)
    body_names = tuple(assignments["TABLE_CONTACT_BODY_NAMES"])
    wrist_sensor_name = assignments["TABLE_CONTACT_SENSOR_NAME"]
    sensor_names = tuple(
        wrist_sensor_name
        if body == "right_wrist_yaw_Link"
        else f"robot_table_contact_{index:02d}"
        for index, body in enumerate(body_names)
    )

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
        "TABLE_CONTACT_BODY_NAMES": body_names,
        "TABLE_CONTACT_SENSOR_NAME": wrist_sensor_name,
        "TABLE_CONTACT_SENSOR_PRIM": (
            "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link"
        ),
        "TABLE_ALL_BODY_CONTACT_SENSOR_NAMES": sensor_names,
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
    env_cfg = types.SimpleNamespace(
        table_robot_keepout=True,
        table_obstacle_prims=five_obstacles,
        scene=types.SimpleNamespace(),
    )
    namespace["attach_table_contact_sensor"](env_cfg)
    assert env_cfg.table_pair_contact_sensor_names == sensor_names
    for sensor_name, body_name in zip(sensor_names, body_names):
        sensor_cfg = getattr(env_cfg.scene, sensor_name)
        assert sensor_cfg.prim_path == f"{{ENV_REGEX_NS}}/Robot/{body_name}"
        assert tuple(sensor_cfg.filter_prim_paths_expr) == five_obstacles
        assert sensor_cfg.update_period == 0.0
        assert "Ball" not in sensor_cfg.prim_path

    # Late full→legacy override must retire every stale per-body pair sensor, not leave hidden
    # collision telemetry outside the declared contract.
    env_cfg.table_robot_keepout = False
    env_cfg.table_obstacle_prims = five_obstacles[:1]
    namespace["attach_table_contact_sensor"](env_cfg)
    assert env_cfg.table_pair_contact_sensor_names == (wrist_sensor_name,)
    assert tuple(
        getattr(env_cfg.scene, wrist_sensor_name).filter_prim_paths_expr
    ) == five_obstacles[:1]
    for stale_name in set(sensor_names) - {wrist_sensor_name}:
        assert getattr(env_cfg.scene, stale_name) is None


def test_action_ball_table_filter_targets_are_kinematic_rigid_bodies():
    """GPU filtered contacts require every static-looking target to be a rigid body."""

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
    helper = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "filter_target_rigid_props"
    )
    rigid_call = next(
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "RigidBodyPropertiesCfg"
    )
    rigid_keywords = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in rigid_call.keywords
    }
    assert rigid_keywords["kinematic_enabled"] is True
    assert rigid_keywords["disable_gravity"] is True

    class FakeRigidBodyPropertiesCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    helper_namespace = {
        "full_assembly": True,
        "sim_utils": types.SimpleNamespace(
            RigidBodyPropertiesCfg=FakeRigidBodyPropertiesCfg
        ),
    }
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            str(cfg_path),
            "exec",
        ),
        helper_namespace,
    )
    full_props = helper_namespace["filter_target_rigid_props"]()
    assert full_props.kinematic_enabled is True
    assert full_props.disable_gravity is True
    helper_namespace["full_assembly"] = False
    # Legacy top-only tasks keep the prior static-collider representation.
    assert helper_namespace["filter_target_rigid_props"]() is None

    cuboid_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "CuboidCfg"
    ]
    filtered_target_calls = []
    for call in cuboid_calls:
        rigid_keyword = next(
            (
                keyword
                for keyword in call.keywords
                if keyword.arg == "rigid_props"
            ),
            None,
        )
        if (
            rigid_keyword is not None
            and isinstance(rigid_keyword.value, ast.Call)
            and isinstance(rigid_keyword.value.func, ast.Name)
            and rigid_keyword.value.func.id == "filter_target_rigid_props"
        ):
            filtered_target_calls.append(call)
    # top + keep-out + net + one post template instantiated for left and right.
    assert len(filtered_target_calls) == 4


# ------------------------------------------------------------------- the termination on an env - #
class _Data:
    def __init__(self, forces, pos, force_matrix=None):
        self.net_forces_w = forces
        self.body_pos_w = pos
        self.force_matrix_w = force_matrix


class _Sensor:
    def __init__(self, names, forces):
        self.body_names = names
        self.data = _Data(forces, None)


class _FilteredSensor:
    def __init__(self, force_matrix, body_name="right_wrist_yaw_Link"):
        self.body_names = [body_name]
        self.cfg = types.SimpleNamespace(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
            filter_prim_paths_expr=list(EXACT_FILTER_PRIMS),
        )
        self.data = _Data(None, None, force_matrix)


class _Asset:
    def __init__(self, names, pos):
        self.body_names = names
        self.data = _Data(None, pos)


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


class _Cfg:
    def __init__(self, name, body_ids):
        self.name = name
        self.body_ids = body_ids


BODIES = ["pelvis_link", "right_elbow_Link", "right_wrist_yaw_Link", "left_ankle_roll_Link"]
WATCHED = [0, 1, 2]  # everything but the foot
EXACT_SENSOR_NAMES = [
    "robot_table_contact_00",
    "robot_table_contact_01",
    "racket_table_contact",
    "robot_table_contact_03",
]
EXACT_FILTER_PRIMS = tuple(
    f"{{ENV_REGEX_NS}}/TablePart{index}" for index in range(5)
)


def _env(
    pos,
    force,
    filtered_force=None,
    *,
    filter_count=1,
    exact_body_forces=None,
):
    sensor = _Sensor(BODIES, torch.tensor(force, dtype=torch.float32))
    if filtered_force is None:
        filtered_force = torch.zeros(len(pos), 1, filter_count, 3)
    else:
        filtered_force = torch.as_tensor(filtered_force, dtype=torch.float32)
    exact_body_forces = exact_body_forces or {}
    filtered_sensors = {}
    for sensor_name, body_name in zip(EXACT_SENSOR_NAMES, BODIES):
        matrix = exact_body_forces.get(body_name)
        if matrix is None:
            matrix = (
                filtered_force
                if body_name == "right_wrist_yaw_Link"
                else torch.zeros(len(pos), 1, filter_count, 3)
            )
        filtered_sensors[sensor_name] = _FilteredSensor(
            torch.as_tensor(matrix, dtype=torch.float32), body_name
        )
    asset = _Asset(BODIES, torch.tensor(pos, dtype=torch.float32))
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
        params.setdefault(
            "all_body_filtered_sensor_cfgs",
            tuple(_Cfg(name, [0]) for name in EXACT_SENSOR_NAMES),
        )
        params.setdefault(
            "expected_full_table_filter_prim_paths",
            EXACT_FILTER_PRIMS,
        )
    return term_mod.robot_hit_table(
        env, _Cfg("contact_forces", WATCHED), _Cfg("racket_table_contact", [0]),
        _Cfg("robot", WATCHED),
        **params,
    )


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
    force = [[[0, 0, 0]] * 4]
    wrist_pair_force = torch.zeros(1, 1, 5, 3)
    wrist_pair_force[0, 0, 1, 2] = 120.0
    assert bool(
        _call(
            term_mod,
            _env(pos, force, wrist_pair_force, filter_count=5),
            full_table_assembly=True,
            keepout_floor_z=0.0,
        )
    ) is True


@pytest.mark.parametrize(
    "point",
    [
        # Top and near-side edge.
        [NEAR_X + 0.30, 0.0, SURFACE_Z - 0.01],
        [NEAR_X, 0.0, SURFACE_Z - 0.025],
        # Net center.
        [NEAR_X + 1.37, 0.0, SURFACE_Z + 0.07],
        # Left/right post centers.
        [NEAR_X + 1.37, 0.7625 + 0.15, SURFACE_Z + 0.08],
        [NEAR_X + 1.37, -(0.7625 + 0.15), SURFACE_Z + 0.08],
    ],
)
def test_full_assembly_exact_pair_channel_covers_top_edge_net_and_posts(
    term_mod, point
):
    pos = [[[0.0, 0.0, 1.0], point, [0.0, 0.0, 1.1], [0.0, 0.1, 0.05]]]
    broad_force = [[[0, 0, 0]] * 4]
    torso_pair_force = torch.zeros(1, 1, 5, 3)
    torso_pair_force[0, 0, 0, 2] = 120.0
    assert bool(
        _call(
            term_mod,
            _env(
                pos,
                broad_force,
                filter_count=5,
                exact_body_forces={"right_elbow_Link": torso_pair_force},
            ),
            full_table_assembly=True,
        )
    ) is True


def test_elbow_mesh_contact_terminates_when_body_origin_is_outside_every_aabb(
    term_mod,
):
    """Regression for the P1: exact contact identity must not depend on elbow origin geometry."""

    elbow_origin_far_from_table = [NEAR_X - 0.40, 0.0, 1.20]
    pos = [[
        [0.0, 0.0, 1.0],
        elbow_origin_far_from_table,
        [0.0, 0.0, 1.1],
        [0.0, 0.1, 0.05],
    ]]
    # A broad net-force stream cannot say what the elbow mesh touched.  Its origin is outside the
    # table assembly, so the old origin-AABB heuristic returned false.  The pair-filter force says
    # this exact rigid body contacted the top collider and must terminate.
    broad_force = [[[0, 0, 0], [0.0, 0.0, 120.0], [0, 0, 0], [0, 0, 0]]]
    elbow_pair_force = torch.zeros(1, 1, 5, 3)
    elbow_pair_force[0, 0, 0, 2] = 120.0
    assert bool(
        _call(
            term_mod,
            _env(
                pos,
                broad_force,
                filter_count=5,
                exact_body_forces={"right_elbow_Link": elbow_pair_force},
            ),
            full_table_assembly=True,
        )
    ) is True


def test_full_assembly_does_not_use_unattributed_broad_force_or_body_origin(
    term_mod,
):
    """An origin in the table plus unrelated broad force is not pair-contact truth."""

    pos = [[
        [0.0, 0.0, 1.0],
        [NEAR_X + 0.3, 0.0, SURFACE_Z],
        [0.0, 0.0, 1.1],
        [0.0, 0.1, 0.05],
    ]]
    broad_force = [[[0, 0, 0], [0.0, 0.0, 120.0], [0, 0, 0], [0, 0, 0]]]
    assert bool(
        _call(
            term_mod,
            _env(pos, broad_force, filter_count=5),
            full_table_assembly=True,
        )
    ) is False


def test_full_assembly_includes_feet_in_exact_table_pair_coverage(term_mod):
    """Floor contact is legal; a foot contacting the table assembly is not."""

    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    foot_pair_force = torch.zeros(1, 1, 5, 3)
    foot_pair_force[0, 0, 1, 0] = 25.0
    assert bool(
        _call(
            term_mod,
            _env(
                pos,
                broad_force,
                filter_count=5,
                exact_body_forces={
                    "left_ankle_roll_Link": foot_pair_force
                },
            ),
            full_table_assembly=True,
        )
    ) is True


@pytest.mark.parametrize("filter_index", range(5))
def test_filtered_racket_channel_accepts_every_assembly_prim(
    term_mod, filter_index
):
    """One-to-many wrist sensor covers top, keepout, net and both posts."""

    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1],
            [NEAR_X - 0.30, 0.0, 1.1], [0.0, 0.1, 0.05]]]
    broad_force = [[[0, 0, 0]] * 4]
    filtered = torch.zeros(1, 1, 5, 3)
    filtered[0, 0, filter_index, 2] = 120.0
    assert bool(
            _call(
                term_mod,
                _env(pos, broad_force, filtered, filter_count=5),
                full_table_assembly=True,
            )
    ) is True


def test_full_assembly_rejects_nonexact_filtered_column_count(term_mod):
    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    with pytest.raises(RuntimeError, match=r"1, 5, 3"):
        _call(
            term_mod,
            _env(pos, broad_force, filter_count=4),
            full_table_assembly=True,
        )


def test_full_assembly_fails_closed_on_missing_or_misordered_body_sensor(term_mod):
    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    env = _env(pos, broad_force, filter_count=5)
    env.scene.sensors.pop(EXACT_SENSOR_NAMES[0])
    with pytest.raises(RuntimeError, match="missing exact per-body"):
        _call(term_mod, env, full_table_assembly=True)

    env = _env(pos, broad_force, filter_count=5)
    first = env.scene.sensors[EXACT_SENSOR_NAMES[0]]
    first.body_names = ["forged_body"]
    with pytest.raises(RuntimeError, match="expected rigid body"):
        _call(term_mod, env, full_table_assembly=True)


def test_full_assembly_fails_closed_on_sensor_source_or_filter_drift(term_mod):
    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    env = _env(pos, broad_force, filter_count=5)
    env.scene.sensors[EXACT_SENSOR_NAMES[1]].cfg.prim_path = (
        "{ENV_REGEX_NS}/Robot/wrong_body"
    )
    with pytest.raises(RuntimeError, match="source/filter binding drift"):
        _call(term_mod, env, full_table_assembly=True)

    env = _env(pos, broad_force, filter_count=5)
    env.scene.sensors[
        EXACT_SENSOR_NAMES[1]
    ].cfg.filter_prim_paths_expr[-1] = "{ENV_REGEX_NS}/Ball"
    with pytest.raises(RuntimeError, match="source/filter binding drift"):
        _call(term_mod, env, full_table_assembly=True)


def test_full_assembly_accepts_interactive_scene_expanded_prim_paths(term_mod):
    """InteractiveScene expands ``{ENV_REGEX_NS}`` in live sensor cfgs in place."""

    pos = [[[0.0, 0.0, 1.0]] * 4]
    broad_force = [[[0, 0, 0]] * 4]
    env = _env(pos, broad_force, filter_count=5)
    env.scene.env_regex_ns = "/World/envs/env_.*"
    for sensor_name in EXACT_SENSOR_NAMES:
        sensor_cfg = env.scene.sensors[sensor_name].cfg
        sensor_cfg.prim_path = sensor_cfg.prim_path.format(
            ENV_REGEX_NS=env.scene.env_regex_ns
        )
        sensor_cfg.filter_prim_paths_expr = [
            path.format(ENV_REGEX_NS=env.scene.env_regex_ns)
            for path in sensor_cfg.filter_prim_paths_expr
        ]

    elbow_pair_force = env.scene.sensors[
        EXACT_SENSOR_NAMES[1]
    ].data.force_matrix_w
    elbow_pair_force[0, 0, 0, 2] = 120.0
    assert bool(_call(term_mod, env, full_table_assembly=True)) is True


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
