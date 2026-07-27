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
import os
import pathlib
import sys
import types

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
    def __init__(self, force_matrix):
        self.data = _Data(None, None, force_matrix)


class _Asset:
    def __init__(self, names, pos):
        self.body_names = names
        self.data = _Data(None, pos)


class _Scene:
    def __init__(self, sensor, filtered_sensor, asset, origins):
        self.sensors = {
            "contact_forces": sensor,
            "racket_table_contact": filtered_sensor,
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


BODIES = ["pelvis_link", "torso_Link", "right_wrist_yaw_Link", "left_ankle_roll_Link"]
WATCHED = [0, 1, 2]  # everything but the foot


def _env(pos, force, filtered_force=None):
    sensor = _Sensor(BODIES, torch.tensor(force, dtype=torch.float32))
    if filtered_force is None:
        filtered_force = torch.zeros(len(pos), 1, 1, 3)
    else:
        filtered_force = torch.tensor(filtered_force, dtype=torch.float32)
    filtered_sensor = _FilteredSensor(filtered_force)
    asset = _Asset(BODIES, torch.tensor(pos, dtype=torch.float32))
    return _Env(_Scene(sensor, filtered_sensor, asset, torch.zeros(len(pos), 3)))


def _call(term_mod, env):
    return term_mod.robot_hit_table(
        env, _Cfg("contact_forces", WATCHED), _Cfg("racket_table_contact", [0]),
        _Cfg("robot", WATCHED),
        near_x=NEAR_X, surface_z=SURFACE_Z, force_threshold=1.0, margin=MARGIN,
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
    """Stated, not hidden: the collider is the top slab, so the legless underside is not covered."""
    pos = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [NEAR_X + 0.25, 0.0, 0.60], [0.0, 0.1, 0.05]]]
    force = [[[0, 0, 0], [0, 0, 0], [0.0, 0.0, 120.0], [0.0, 0.0, 400.0]]]
    assert bool(_call(term_mod, _env(pos, force))) is False


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
