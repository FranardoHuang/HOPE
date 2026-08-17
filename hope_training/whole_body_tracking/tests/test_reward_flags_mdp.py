"""reward-flags-0709 — mdp BEHAVIOR unit tests (CPU, isaaclab STUBBED).

The mdp modules import isaaclab at the top, so this file installs a minimal isaaclab stub into
sys.modules and then loads the REAL repo modules under their canonical dotted names
(commands.py / rewards.py / terminations.py / hope_commands.py / hope_rewards.py /
hope_observations.py). What is tested is therefore the actual shipped math, not a re-derivation:

* V2  rewards._apply_window_scale through the real motion_* funcs: in-window x k, out-window x 1,
      default params = byte-identical.
* 1c  RacketTargetCommand._compute_strike_timing: split pos/wide masks; None defaults alias the
      legacy strike_window; hope_rewards racket_position gates on the TIGHT window while
      racket_velocity/racket_normal gate on the WIDE one.
* C2a hope_rewards._pos_gate: sigmoid((r - pos_err)/0.05) values at pos_err = r (0.5) and far
      out of reach (~0); None = exactly 1.0.
* B2  hope_rewards.racket_guidance: min(dist, d_max) clamp + (pre_strike | strike_window) mask.
* R-b hope_rewards.tracking_envelope_violation == the union of the two removed termination
      expressions (anchor z / any-envelope-body z over threshold).
* R-a hope_observations.generated_commands_actor_leg_masked: exactly the 24 runtime-derived leg
      dims become default-stand pos + zero vel, every other dim byte-identical to
      cat([joint_pos, joint_vel]); a wrong find_joints count refuses loudly.
* H0  HitterPure's four body-imitation terms and three reference-relative terminations are
      swing-only; held envs receive no teacher-velocity income / hidden reference death while
      absolute safety remains a separate config concern.
* A1  target position/velocity/demanded-normal/swing-sign/TTS remain one atomic actor message
      through delay/dropout; source-timestamp compensation and its stale-TTS negative control are
      distinct, and a true reset clears/backfills every stateful sensor-defect buffer.
* P2.4 base_decel is zero during a frozen hold; debug raw/gated telemetry records the real mask.
* D6 qdot-limit hinge uses realized joint velocity divided by the actual 31 runtime-ordered
      articulation limits; reordered joints and zero/non-finite limits fail closed.
* R-c commands.MotionCommand on a synthetic 2-clip npz pair: rsi_skip_settle_frames offsets the
      swing-entry frame (multiseg + single-clip clamp, incl. the short-clip clamp) and
      rsi_hold_root_stand_z rewrites ONLY the held-RSI birth root z to the default stand height.
* A8 post-swing replay activation: true-reset counts distinguish buffer-not-ready, eligible,
      random-not-selected, selected and state-write-started paths; per-update consumption resets
      exactly, and the disabled mechanism stays all-zero.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py -q
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MDP_DIR = os.path.abspath(os.path.join(
    HERE, "..", "source", "whole_body_tracking", "whole_body_tracking", "tasks", "tracking", "mdp"))


# --------------------------------------------------------------------------------------------- #
# isaaclab stub — the minimal surface the mdp modules import at module scope
# --------------------------------------------------------------------------------------------- #
def _module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _sample_uniform(lo, hi, size, device=None):
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device)
    shape = tuple(size) if isinstance(size, (tuple, list, torch.Size)) else tuple(size.shape) if torch.is_tensor(size) else (int(size),)
    return lo_t + (hi_t - lo_t) * torch.rand(*shape, device=device)


def _install_isaaclab_stub():
    if "isaaclab" in sys.modules and getattr(sys.modules["isaaclab"], "_reward_flags_stub", False):
        return
    isaaclab = _module("isaaclab")
    isaaclab._reward_flags_stub = True

    assets = _module("isaaclab.assets")
    for cls_name in ("Articulation", "RigidObject", "ArticulationCfg", "AssetBaseCfg", "RigidObjectCfg"):
        setattr(assets, cls_name, type(cls_name, (), {}))
    isaaclab.assets = assets

    managers = _module("isaaclab.managers")

    class CommandTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.num_envs = env.num_envs
            self.device = env.device
            self.metrics = {}

    class _KwInit:
        def __init__(self, *args, **kwargs):
            for a, name in zip(args, ("name",)):
                setattr(self, name, a)
            for k, v in kwargs.items():
                setattr(self, k, v)

    managers.CommandTerm = CommandTerm
    managers.CommandTermCfg = type("CommandTermCfg", (), {"__init__": _KwInit.__init__})
    managers.SceneEntityCfg = type("SceneEntityCfg", (_KwInit,), {})
    for cfg_name in ("RewardTermCfg", "ObservationTermCfg", "TerminationTermCfg", "EventTermCfg",
                     "ObservationGroupCfg"):
        setattr(managers, cfg_name, type(cfg_name, (_KwInit,), {}))
    isaaclab.managers = managers

    markers = _module("isaaclab.markers")

    class VisualizationMarkers:
        def __init__(self, cfg):
            self.cfg = cfg

    class _MarkerCfg:
        def __init__(self):
            self.markers = {"frame": types.SimpleNamespace(scale=None)}

        def replace(self, **kwargs):
            return copy.deepcopy(self)

    markers.VisualizationMarkers = VisualizationMarkers
    markers.VisualizationMarkersCfg = _MarkerCfg
    markers_config = _module("isaaclab.markers.config")
    markers_config.FRAME_MARKER_CFG = _MarkerCfg()
    markers.config = markers_config
    isaaclab.markers = markers

    sensors = _module("isaaclab.sensors")
    sensors.ContactSensor = type("ContactSensor", (), {})
    sensors.ContactSensorCfg = type("ContactSensorCfg", (), {})
    isaaclab.sensors = sensors

    envs = _module("isaaclab.envs")
    envs.ManagerBasedRLEnv = type("ManagerBasedRLEnv", (), {})
    envs.ManagerBasedEnv = type("ManagerBasedEnv", (), {})
    isaaclab.envs = envs

    envs_mdp = _module("isaaclab.envs.mdp")
    envs_actions = _module("isaaclab.envs.mdp.actions")
    actions_cfg = _module("isaaclab.envs.mdp.actions.actions_cfg")
    joint_actions = _module("isaaclab.envs.mdp.actions.joint_actions")

    class JointPositionAction:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._asset = env.scene[cfg.asset_name]
            self.num_envs = env.num_envs
            self.device = env.device
            self._joint_ids = slice(None)
            self._joint_names = list(self._asset.data.joint_names)
            self._raw_actions = torch.zeros(
                self.num_envs, len(self._joint_names), device=self.device
            )
            self._processed_actions = torch.zeros_like(self._raw_actions)
            self._scale = float(getattr(cfg, "scale", 1.0))
            self._offset = (
                self._asset.data.default_joint_pos.clone()
                if getattr(cfg, "use_default_offset", False)
                else 0.0
            )

        @property
        def raw_actions(self):
            # real isaaclab JointAction exposes raw_actions; action_acc_l2 reads it
            return self._raw_actions

        @property
        def processed_actions(self):
            return self._processed_actions

        def process_actions(self, actions):
            self._raw_actions[:] = actions
            self._processed_actions = self._raw_actions * self._scale + self._offset

        def reset(self, env_ids=None):
            self._raw_actions[env_ids] = 0.0

    actions_cfg.JointPositionActionCfg = type("JointPositionActionCfg", (), {})
    joint_actions.JointPositionAction = JointPositionAction
    envs_actions.actions_cfg = actions_cfg
    envs_actions.joint_actions = joint_actions
    envs_mdp.actions = envs_actions
    envs.mdp = envs_mdp

    utils = _module("isaaclab.utils")

    def configclass(cls):
        def __init__(self, **kwargs):  # class attributes double as defaults
            for k, v in kwargs.items():
                setattr(self, k, v)

        cls.__init__ = __init__
        return cls

    utils.configclass = configclass

    math_mod = _module("isaaclab.utils.math")
    math_mod.sample_uniform = _sample_uniform
    math_mod.quat_error_magnitude = lambda a, b: torch.norm(a - b, dim=-1)
    math_mod.quat_apply = lambda q, v: v
    math_mod.quat_rotate_inverse = lambda q, v: v
    math_mod.quat_inv = lambda q: q
    math_mod.quat_mul = lambda a, b: b.clone() if torch.is_tensor(b) else b
    math_mod.yaw_quat = lambda q: q

    def quat_from_euler_xyz(r, p, y):
        q = torch.zeros(r.shape[0], 4)
        q[:, 0] = 1.0
        return q

    math_mod.quat_from_euler_xyz = quat_from_euler_xyz
    math_mod.matrix_from_quat = lambda q: torch.eye(3, device=q.device).expand(q.shape[0], 3, 3)
    math_mod.euler_xyz_from_quat = lambda q: (torch.zeros(q.shape[0]),) * 3
    utils.math = math_mod
    isaaclab.utils = utils


def _load(dotted, filename):
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, os.path.join(MDP_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


_ISAACLAB_PREFIX = "isaaclab."
_PREEXISTING_ISAACLAB_MODULES = {
    name: module
    for name, module in sys.modules.items()
    if name == "isaaclab" or name.startswith(_ISAACLAB_PREFIX)
}
try:
    _install_isaaclab_stub()
    _PKG = "whole_body_tracking.tasks.tracking.mdp"
    for _p in ("whole_body_tracking", "whole_body_tracking.tasks", "whole_body_tracking.tasks.tracking", _PKG):
        sys.modules.setdefault(_p, types.ModuleType(_p))
    _load(f"{_PKG}.event_timing", "event_timing.py")
    _load(f"{_PKG}.post_swing_teacher", "post_swing_teacher.py")
    planner_revision_mod = _load(f"{_PKG}.planner_revision", "planner_revision.py")
    _racket_contact_geometry_mod = _load(
        f"{_PKG}.racket_contact_geometry", "racket_contact_geometry.py"
    )
    # ``_load`` executes a real source file under its canonical dotted name, but unlike the
    # regular import machinery it does not attach the child to our synthetic parent package.
    # Production code uses ``from . import racket_contact_geometry`` lazily, so make that package
    # edge explicit in the shared isolated-module harness as well.
    setattr(
        sys.modules[_PKG], "racket_contact_geometry", _racket_contact_geometry_mod
    )
    # 同一条边:``hope_commands.py`` 走的是 ``from ...mdp import
    # action_ball_solver_semantic_surface``(d4e1e70c 起),而合成父包没有 ``__path__``,
    # 所以子模块必须像上面那样手工挂上去,否则整个共享夹具在 import 期就炸,
    # 依赖它的每个测试模块一起变成"收集失败",而不是失败的断言。
    _solver_semantic_surface_mod = _load(
        f"{_PKG}.action_ball_solver_semantic_surface",
        "action_ball_solver_semantic_surface.py",
    )
    setattr(
        sys.modules[_PKG],
        "action_ball_solver_semantic_surface",
        _solver_semantic_surface_mod,
    )
    commands_mod = _load(f"{_PKG}.commands", "commands.py")
    rewards_mod = _load(f"{_PKG}.rewards", "rewards.py")
    terminations_mod = _load(f"{_PKG}.terminations", "terminations.py")
    _load(f"{_PKG}.stage1_question_bank", "stage1_question_bank.py")
    hope_commands_mod = _load(f"{_PKG}.hope_commands", "hope_commands.py")
    hope_rewards_mod = _load(f"{_PKG}.hope_rewards", "hope_rewards.py")
    hope_observations_mod = _load(f"{_PKG}.hope_observations", "hope_observations.py")
    hope_actions_mod = _load(f"{_PKG}.hope_actions", "hope_actions.py")
finally:
    # This is a collection-time unit-test harness, not a process-wide Isaac Lab
    # replacement.  Leaving these doubles in ``sys.modules`` makes a later real
    # import (for example ``isaaclab.utils.noise``) resolve through a fake
    # non-package and turns otherwise independent tests into order-dependent
    # collection/execution failures.
    for _name in tuple(sys.modules):
        if _name == "isaaclab" or _name.startswith(_ISAACLAB_PREFIX):
            sys.modules.pop(_name)
    sys.modules.update(_PREEXISTING_ISAACLAB_MODULES)


# --------------------------------------------------------------------------------------------- #
# shared fakes
# --------------------------------------------------------------------------------------------- #
def _fake_env(**terms):
    return types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: terms[name]))


def _termination_env(active_terms, masks):
    return types.SimpleNamespace(
        termination_manager=types.SimpleNamespace(
            active_terms=tuple(active_terms),
            get_term=lambda name: masks[name],
        )
    )


def test_stage1_object_free_safety_union_or_and_fail_loud_contract():
    names = (
        "base_fell_tilt",
        "base_too_low",
        "joint_actual_forbidden",
        "joint_qdes_forbidden",
    )
    masks = {
        names[0]: torch.tensor([False, True, False]),
        names[1]: torch.tensor([False, False, True]),
        names[2]: torch.tensor([True, False, False]),
        names[3]: torch.tensor([False, False, False]),
    }
    env = _termination_env(names, masks)
    # The Stage-1 object-free union intentionally has no robot_hit_table input.
    actual = hope_rewards_mod.stage1_object_free_safety_terminated(env, names)
    torch.testing.assert_close(actual, torch.ones(3))

    missing = _termination_env(names[:-1], masks)
    with pytest.raises(RuntimeError, match="missing active termination terms"):
        hope_rewards_mod.stage1_object_free_safety_terminated(missing, names)

    with pytest.raises(RuntimeError, match="requires the exact ordered"):
        hope_rewards_mod.stage1_object_free_safety_terminated(env, names[::-1])

    malformed_masks = dict(masks)
    malformed_masks[names[2]] = torch.zeros(3, dtype=torch.float32)
    malformed = _termination_env(names, malformed_masks)
    with pytest.raises(RuntimeError, match="same-device.*bool mask"):
        hope_rewards_mod.stage1_object_free_safety_terminated(malformed, names)


#: The table guard owns this number; read it rather than keeping a copy that
#: silently disagrees the next time the plant changes.
_TABLE_GUARD_COMPONENTS = _load(
    f"{_PKG}.terminations", "terminations.py"
)._A3_COLLISION_PROXY_COMPONENT_COUNT


def test_table_guard_first_hit_ledger_conserves_cells_categories_and_phases():
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command.num_envs = 5
    command.device = "cpu"
    command._recover_from_clip = torch.tensor([-1, -1, -1, 0, -1])
    command.strike_window = torch.tensor([False, True, False, False, False])
    command.pre_strike = torch.tensor([True, True, False, True, True])
    component_ids = tuple(
        f"component:{index}" for index in range(_TABLE_GUARD_COMPONENTS)
    )
    owner_names = tuple(
        f"owner_{index}" for index in range(_TABLE_GUARD_COMPONENTS)
    )
    obstacle_roles = ("top", "keepout", "net", "post_left", "post_right")
    # Production creates and books the dense ledger on the inference-mode physics path, then
    # consumes it from the normal-mode PPO logger.  Preserve that cross-mode lifecycle here.
    with torch.inference_mode():
        command.configure_table_guard_attribution(
            component_ids=component_ids,
            component_owner_names=owner_names,
            obstacle_roles=obstacle_roles,
        )
    counts = command._table_guard_attribution_counts
    counts_data_ptr = counts.data_ptr()
    assert torch.is_inference(counts)
    assert counts.dtype == torch.long
    assert counts.device.type == "cpu"
    # Item rows: one per proxy component, plus the independent blade row and
    # the non-finite row.
    assert tuple(counts.shape) == (4, 3, _TABLE_GUARD_COMPONENTS + 2, 5)

    component_broad = torch.zeros(5, _TABLE_GUARD_COMPONENTS, 5, dtype=torch.bool)
    component_exact = torch.zeros_like(component_broad)
    blade_broad = torch.zeros(5, 5, dtype=torch.bool)
    blade_exact = torch.zeros_like(blade_broad)
    nonfinite = torch.zeros(5, dtype=torch.bool)
    component_broad[0, 2, 1] = True
    component_exact[0, 2, 1] = True
    component_broad[1, 1, 4] = True
    component_exact[1, 1, 4] = True
    blade_broad[2, 2] = True
    blade_exact[2, 2] = True
    blade_broad[3, 3] = True
    blade_exact[3, 3] = True
    nonfinite[4] = True
    attribution = types.SimpleNamespace(
        legacy_mask=torch.ones(5, dtype=torch.bool),
        component_conservative_overlap=component_broad,
        component_exact_overlap=component_exact,
        blade_conservative_overlap=blade_broad,
        blade_exact_overlap=blade_exact,
        nonfinite=nonfinite,
    )
    with torch.inference_mode():
        command.record_table_guard_first_hits(
            torch.ones(5, dtype=torch.bool), attribution
        )
    snapshot = command._consume_table_guard_attribution_counts()
    command._validate_table_guard_attribution_conservation(
        {"termination_reason_robot_hit_table_count": torch.tensor(5)},
        snapshot,
    )
    assert snapshot["table_guard_first_hit_total_count"] == 5
    assert snapshot["table_guard_first_hit_phase_pre_count"] == 2
    assert snapshot["table_guard_first_hit_phase_strike_count"] == 1
    assert snapshot["table_guard_first_hit_phase_post_count"] == 1
    assert snapshot["table_guard_first_hit_phase_recovery_count"] == 1
    assert snapshot["table_guard_first_hit_category_proxy_exact_overlap_count"] == 2
    assert snapshot["table_guard_first_hit_category_blade_exact_overlap_count"] == 2
    assert snapshot["table_guard_first_hit_category_nonfinite_count"] == 1
    cells = {
        key: value
        for key, value in snapshot.items()
        if key.startswith("table_guard_first_hit_cell_")
    }
    assert len(cells) == 5
    assert sum(cells.values()) == snapshot["table_guard_first_hit_total_count"]
    assert any("component_02_owner_2_keepout" in key for key in cells)
    assert any("independent_blade_net" in key for key in cells)
    assert any("nonfinite_pose_not_applicable" in key for key in cells)
    assert command._table_guard_attribution_counts is counts
    assert counts.data_ptr() == counts_data_ptr
    assert counts.sum().item() == 0
    second = command._consume_table_guard_attribution_counts()
    assert second["table_guard_first_hit_total_count"] == 0
    assert not any(
        key.startswith("table_guard_first_hit_cell_") for key in second
    )
    with torch.inference_mode():
        command.record_table_guard_first_hits(
            torch.ones(5, dtype=torch.bool), attribution
        )
    reused = command._consume_table_guard_attribution_counts()
    assert reused["table_guard_first_hit_total_count"] == 5
    assert command._table_guard_attribution_counts is counts
    assert counts.data_ptr() == counts_data_ptr
    assert counts.sum().item() == 0

    drifted = dict(snapshot)
    drifted["table_guard_first_hit_total_count"] = 4
    with pytest.raises(RuntimeError, match="do not conserve"):
        command._validate_table_guard_attribution_conservation(
            {"termination_reason_robot_hit_table_count": torch.tensor(5)},
            drifted,
        )


def test_table_guard_oracle_first_hit_export_is_sidecar_with_honest_gaps():
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command.num_envs = 1
    command.device = "cpu"
    command._recover_from_clip = torch.tensor([-1])
    command.strike_window = torch.tensor([True])
    command.pre_strike = torch.tensor([True])
    component_ids = tuple(
        f"component:{index}" for index in range(_TABLE_GUARD_COMPONENTS)
    )
    owner_names = tuple(
        f"owner_{index}" for index in range(_TABLE_GUARD_COMPONENTS)
    )
    obstacle_roles = ("top", "keepout", "net", "post_left", "post_right")
    command.cfg = types.SimpleNamespace(racket_body_name="racket")
    command.robot = types.SimpleNamespace(
        body_names=("owner_2", "racket"),
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        ),
    )
    command._motion = lambda: types.SimpleNamespace(time_steps=torch.tensor([17]))
    with torch.inference_mode():
        command.configure_table_guard_attribution(
            component_ids=component_ids,
            component_owner_names=owner_names,
            obstacle_roles=obstacle_roles,
        )
        command.configure_table_guard_oracle_first_hit_export()
        command.set_table_guard_oracle_first_hit_context(
            episode=3, control_step=5
        )
    component_broad = torch.zeros(1, _TABLE_GUARD_COMPONENTS, 5, dtype=torch.bool)
    component_exact = torch.zeros_like(component_broad)
    component_broad[0, 2, 1] = True
    component_exact[0, 2, 1] = True
    attribution = types.SimpleNamespace(
        legacy_mask=torch.tensor([True]),
        component_conservative_overlap=component_broad,
        component_exact_overlap=component_exact,
        blade_conservative_overlap=torch.zeros(1, 5, dtype=torch.bool),
        blade_exact_overlap=torch.zeros(1, 5, dtype=torch.bool),
        nonfinite=torch.tensor([False]),
    )
    with torch.inference_mode():
        command.record_table_guard_first_hits(torch.tensor([True]), attribution)
    rows = command.consume_table_guard_oracle_first_hit_rows()
    assert rows == [{
        "episode": 3,
        "control_step": 5,
        "physics_substep": None,
        "physics_substep_unavailable_reason": (
            "the existing action-to-command ledger interface does not expose a physics-substep ordinal"
        ),
        "motion_frame": 17,
        "motion_frame_unavailable_reason": None,
        "phase": "strike",
        "component_id": "component:2",
        "body_name": "owner_2",
        "obstacle": "keepout",
        "blade_or_proxy": "collision_proxy_component",
        "exact_vs_conservative": "proxy_exact_overlap",
        "actual_pose_w": {
            "position_w_m": [1.0, 2.0, 3.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "actual_pose_unavailable_reason": None,
    }]
    # Consuming the export is independent of the production aggregate ledger.
    assert command._consume_table_guard_attribution_counts()[
        "table_guard_first_hit_total_count"
    ] == 1
    assert command.consume_table_guard_oracle_first_hit_rows() == []


def test_table_guard_action_hook_preserves_terminal_and_books_only_new_hits():
    action = hope_actions_mod.ClampedJointPositionAction.__new__(
        hope_actions_mod.ClampedJointPositionAction
    )
    action._table_contact_latch = types.SimpleNamespace(
        hit=torch.tensor([False, True, False])
    )
    attribution = types.SimpleNamespace(
        legacy_mask=torch.tensor([True, True, False])
    )
    booked = []

    class Prepared:
        attribution_enabled = True

        def sample_with_attribution(self):
            return attribution

        def record_first_hits(self, mask, evidence):
            booked.append((mask.clone(), evidence))

    result = action._sample_prepared_table_pose_guard(Prepared())
    assert result is attribution.legacy_mask
    assert len(booked) == 1
    assert booked[0][0].tolist() == [True, False, False]
    assert booked[0][1] is attribution

    # The real latch reset clears per-env sticky bits.  The next episode must
    # be eligible for a fresh first hit rather than inheriting stale evidence.
    action._table_contact_latch.hit.zero_()
    action._sample_prepared_table_pose_guard(Prepared())
    assert booked[-1][0].tolist() == [True, True, False]


def test_table_guard_action_hook_default_off_is_exact_legacy_fast_path():
    action = hope_actions_mod.ClampedJointPositionAction.__new__(
        hope_actions_mod.ClampedJointPositionAction
    )
    action._table_contact_latch = types.SimpleNamespace(
        hit=torch.tensor([False])
    )
    terminal = torch.tensor([True])

    class Prepared:
        attribution_enabled = False

        def __call__(self):
            return terminal

        def sample_with_attribution(self):
            raise AssertionError("default-off path constructed SAT evidence")

        def record_first_hits(self, *_args):
            raise AssertionError("default-off path touched attribution ledger")

    assert action._sample_prepared_table_pose_guard(Prepared()) is terminal


def test_stand_start_yaw_degenerate_nonzero_range_is_applied_deterministically():
    fixed = commands_mod._stand_start_yaw_samples((0.35, 0.35), 4, "cpu")
    assert torch.equal(fixed, torch.full((4,), 0.35))
    # The legacy default remains a true no-op and consumes no random draw.
    assert commands_mod._stand_start_yaw_samples((0.0, 0.0), 4, "cpu") is None


def _yaw_quats(yaws):
    y = torch.as_tensor(yaws, dtype=torch.float32)
    q = torch.zeros(len(y), 4)
    q[:, 0] = torch.cos(0.5 * y)
    q[:, 3] = torch.sin(0.5 * y)
    return q


def _recovery_cmd(n):
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(root_quat_w=_yaw_quats([0.0] * n))
    )
    rt._hold_edge_pending = torch.zeros(n, dtype=torch.bool)
    rt._previous_in_hold = torch.zeros(n, dtype=torch.bool)
    rt._hold_start_yaw = torch.zeros(n)
    rt._heading_expiry_sum_acc = 0.0
    rt._heading_expiry_n_acc = 0.0
    rt._recovery_spawn_sum_acc = 0.0
    rt._recovery_expiry_sum_acc = 0.0
    rt._recovery_n_acc = 0.0
    return rt


def test_recovery_edges_condition_on_spawn_yaw_and_handle_negative_wrap_yaw():
    rt = _recovery_cmd(3)
    rt.robot.data.root_quat_w = _yaw_quats([0.5, 0.1, -0.6])
    rt._update_hold_recovery_metrics(torch.tensor([True, True, True]))
    assert torch.allclose(rt._hold_start_yaw, torch.tensor([0.5, 0.1, 0.6]), atol=1e-6)

    rt.robot.data.root_quat_w = _yaw_quats([0.2, 0.05, -0.1])
    rt._update_hold_recovery_metrics(torch.tensor([False, False, False]))
    assert rt._heading_expiry_n_acc == pytest.approx(3.0)
    assert rt._heading_expiry_sum_acc == pytest.approx(0.35, abs=1e-6)
    # Only spawn yaw > 0.30 rad belongs to the conditioned recovery ledger.
    assert rt._recovery_n_acc == pytest.approx(2.0)
    assert rt._recovery_spawn_sum_acc == pytest.approx(1.1, abs=1e-6)
    assert rt._recovery_expiry_sum_acc == pytest.approx(0.3, abs=1e-6)


def test_recovery_reset_while_held_restarts_edge_and_zero_length_hold_is_not_counted():
    rt = _recovery_cmd(1)
    rt.robot.data.root_quat_w = _yaw_quats([0.8])
    rt._update_hold_recovery_metrics(torch.tensor([True]))

    # A reset/wrap replaces the held state without a False edge. Pending must discard the old
    # 0.8-rad spawn and stamp the new 0.4-rad state, not book a false expiry.
    rt._hold_edge_pending[:] = True
    rt._previous_in_hold[:] = False
    rt._hold_start_yaw[:] = 0.0
    rt.robot.data.root_quat_w = _yaw_quats([0.4])
    rt._update_hold_recovery_metrics(torch.tensor([True]))
    assert rt._hold_start_yaw.item() == pytest.approx(0.4, abs=1e-6)
    assert rt._recovery_n_acc == 0.0

    rt.robot.data.root_quat_w = _yaw_quats([0.1])
    rt._update_hold_recovery_metrics(torch.tensor([False]))
    assert rt._recovery_n_acc == pytest.approx(1.0)
    assert rt._recovery_spawn_sum_acc == pytest.approx(0.4, abs=1e-6)
    assert rt._recovery_expiry_sum_acc == pytest.approx(0.1, abs=1e-6)

    before = (rt._heading_expiry_n_acc, rt._recovery_n_acc)
    rt._hold_edge_pending[:] = True
    rt._previous_in_hold[:] = False
    rt._update_hold_recovery_metrics(torch.tensor([False]))
    assert (rt._heading_expiry_n_acc, rt._recovery_n_acc) == before


def _fake_racket_cmd(n=4, window=None, window_pos=None, window_wide=None, pre_strike=None):
    window = torch.zeros(n, dtype=torch.bool) if window is None else window
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(face_command=False, debug_reward_logging=False),
        num_envs=n, device="cpu", metrics={},
        strike_window=window,
        strike_window_pos=window if window_pos is None else window_pos,
        strike_window_wide=window if window_wide is None else window_wide,
        pre_strike=torch.zeros(n, dtype=torch.bool) if pre_strike is None else pre_strike,
        time_to_strike=torch.zeros(n),
        racket_pos_w=torch.zeros(n, 3),
        racket_target_pos_w=torch.zeros(n, 3),
        racket_lin_vel_w=torch.zeros(n, 3),
        racket_target_vel_w=torch.zeros(n, 3),
        racket_normal_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).clone(),
        # raw (+Y/A-frame) twin — face_command 通道读它(hope_rewards._face_pair);无符号表的
        # 环境里 raw ≡ signed,fake 保持同值同语义。
        racket_normal_raw_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).clone(),
        racket_target_normal_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).clone(),
        target_normal_cmd=torch.zeros(n, 3),
    )


# --------------------------------------------------------------------------------------------- #
# V2 in-window imitation yield (rewards._apply_window_scale through the real funcs)
# --------------------------------------------------------------------------------------------- #
def _fake_motion_for_rewards(n=4, n_bodies=2):
    zeros_b = torch.zeros(n, n_bodies, 3)
    quat = torch.zeros(n, n_bodies, 4)
    quat[..., 0] = 1.0
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            body_names=["torso_Link", "right_wrist_yaw_Link"],
            v1_free_wrist_vel_mimic_activation=False,
            v2_motion_scale_in_window_activation=None,
        ),
        anchor_pos_w=torch.zeros(n, 3), robot_anchor_pos_w=torch.zeros(n, 3),
        anchor_quat_w=torch.zeros(n, 4), robot_anchor_quat_w=torch.zeros(n, 4),
        body_pos_relative_w=zeros_b.clone(), robot_body_pos_w=zeros_b.clone(),
        body_quat_relative_w=quat.clone(), robot_body_quat_w=quat.clone(),
        body_lin_vel_w=zeros_b.clone(), robot_body_lin_vel_w=zeros_b.clone(),
        body_ang_vel_w=zeros_b.clone(), robot_body_ang_vel_w=zeros_b.clone(),
        in_hold=torch.zeros(n, dtype=torch.bool),
        record_v1_velocity_mimic_activation=lambda *args, **kwargs: None,
        record_v2_strike_window_scale_activation=lambda *args, **kwargs: None,
    )


def _activation_motion_for_rewards(n=4, *, v1=False, v2_scale=None):
    motion = _fake_motion_for_rewards(n)
    motion.cfg.v1_free_wrist_vel_mimic_activation = v1
    motion.cfg.v2_motion_scale_in_window_activation = v2_scale
    motion._post_swing_activation_counters = {}
    motion._reward_activation_counters = {
        name: torch.zeros((), dtype=torch.long)
        for name in (
            "v1_velocity_mimic_eligible_sample_count",
            "v1_held_wrist_excluded_sample_count",
            "v2_strike_window_eligible_imitation_sample_count",
            "v2_quarter_scaled_strike_window_imitation_sample_count",
        )
    }
    motion.record_v1_velocity_mimic_activation = types.MethodType(
        commands_mod.MotionCommand.record_v1_velocity_mimic_activation, motion
    )
    motion.record_v2_strike_window_scale_activation = types.MethodType(
        commands_mod.MotionCommand.record_v2_strike_window_scale_activation,
        motion,
    )
    motion.consume_training_activation_counters = types.MethodType(
        commands_mod.MotionCommand.consume_training_activation_counters, motion
    )
    return motion


def test_v2_window_scale_scales_only_windowed_envs():
    n = 4
    win = torch.tensor([True, False, True, False])
    racket = _fake_racket_cmd(n, window=torch.zeros(n, dtype=torch.bool), window_wide=win)
    env = _fake_env(motion=_fake_motion_for_rewards(n), racket_target=racket)
    for func in (
        rewards_mod.motion_global_anchor_position_error_exp,
        rewards_mod.motion_global_anchor_orientation_error_exp,
    ):
        r = func(env, "motion", std=1.0, window_scale=0.25, window_command_name="racket_target")
        assert torch.allclose(r, torch.tensor([0.25, 1.0, 0.25, 1.0])), func.__name__
    for func in (
        rewards_mod.motion_relative_body_position_error_exp,
        rewards_mod.motion_relative_body_orientation_error_exp,
        rewards_mod.motion_global_body_linear_velocity_error_exp,
        rewards_mod.motion_global_body_angular_velocity_error_exp,
    ):
        r = func(env, "motion", std=1.0, body_names=None,
                 window_scale=0.25, window_command_name="racket_target")
        assert torch.allclose(r, torch.tensor([0.25, 1.0, 0.25, 1.0])), func.__name__


def test_v2_default_params_are_byte_identical():
    n = 3
    env = _fake_env(motion=_fake_motion_for_rewards(n))  # no racket cmd needed on the default path
    r_default = rewards_mod.motion_global_body_linear_velocity_error_exp(env, "motion", std=1.0)
    assert torch.allclose(r_default, torch.ones(n))
    # k=1.0 short-circuits before the command lookup too
    r_k1 = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, window_scale=1.0, window_command_name="racket_target")
    assert torch.equal(r_default, r_k1)


def test_v2_uses_wide_window_and_forwards_through_swing_only_wrappers():
    n = 2
    motion = _fake_motion_for_rewards(n)
    wide = torch.tensor([True, False])
    racket = _fake_racket_cmd(n, window=torch.zeros(n, dtype=torch.bool), window_wide=wide)
    env = _fake_env(motion=motion, racket_target=racket)
    r = hope_rewards_mod.motion_body_pos_swing_only(
        env, "motion", std=1.0, body_names=None, window_scale=0.0, window_command_name="racket_target")
    assert torch.allclose(r, torch.tensor([0.0, 1.0]))  # teacher fully silent in-window
    motion.in_hold = torch.tensor([False, True])  # hold gate still wins
    r = hope_rewards_mod.motion_body_ori_swing_only(
        env, "motion", std=1.0, body_names=None, window_scale=0.0, window_command_name="racket_target")
    assert torch.allclose(r, torch.tensor([0.0, 0.0]))


def test_v1_execution_counts_multi_env_samples_and_resolved_wrist_exclusion():
    motion = _activation_motion_for_rewards(3, v1=True)
    env = _fake_env(motion=motion)
    kept_bodies = ["torso_Link"]

    first = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, body_names=kept_bodies
    )
    second = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, body_names=kept_bodies
    )
    assert torch.equal(first, torch.ones(3))
    assert torch.equal(second, first)

    snapshot = motion.consume_training_activation_counters()
    assert snapshot["v1_velocity_mimic_eligible_sample_count"].item() == 6
    assert snapshot["v1_held_wrist_excluded_sample_count"].item() == 6
    assert snapshot["v1_velocity_mimic_eligible_sample_count"].dtype == torch.long
    assert all(value.item() == 0 for value in motion._reward_activation_counters.values())


def test_v1_counterexample_keeps_denominator_when_wrist_was_not_excluded():
    motion = _activation_motion_for_rewards(4, v1=True)
    env = _fake_env(motion=motion)
    rewards_mod.motion_global_body_linear_velocity_error_exp(
        env,
        "motion",
        std=1.0,
        body_names=["torso_Link", "right_wrist_yaw_Link"],
    )
    snapshot = motion.consume_training_activation_counters()
    assert snapshot["v1_velocity_mimic_eligible_sample_count"].item() == 4
    assert snapshot["v1_held_wrist_excluded_sample_count"].item() == 0


def test_v2_execution_counts_each_real_windowed_reward_application():
    motion = _activation_motion_for_rewards(4, v2_scale=0.25)
    wide = torch.tensor([True, False, True, False])
    racket = _fake_racket_cmd(
        4, window=torch.zeros(4, dtype=torch.bool), window_wide=wide
    )
    env = _fake_env(motion=motion, racket_target=racket)

    anchor = rewards_mod.motion_global_anchor_position_error_exp(
        env,
        "motion",
        std=1.0,
        window_scale=0.25,
        window_command_name="racket_target",
    )
    velocity = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env,
        "motion",
        std=1.0,
        body_names=["torso_Link"],
        window_scale=0.25,
        window_command_name="racket_target",
    )
    expected = torch.tensor([0.25, 1.0, 0.25, 1.0])
    assert torch.equal(anchor, expected)
    assert torch.equal(velocity, expected)

    snapshot = motion.consume_training_activation_counters()
    # Two imitation terms x two in-window env samples.  This deliberately measures reward
    # applications, not unique environment steps.
    assert snapshot["v2_strike_window_eligible_imitation_sample_count"].item() == 4
    assert snapshot[
        "v2_quarter_scaled_strike_window_imitation_sample_count"
    ].item() == 4


def test_v2_counterexample_records_window_but_not_wrong_scale_as_quarter():
    motion = _activation_motion_for_rewards(4, v2_scale=0.25)
    wide = torch.tensor([True, False, True, False])
    env = _fake_env(
        motion=motion,
        racket_target=_fake_racket_cmd(4, window_wide=wide),
    )
    result = rewards_mod.motion_global_anchor_position_error_exp(
        env,
        "motion",
        std=1.0,
        window_scale=0.5,
        window_command_name="racket_target",
    )
    assert torch.equal(result, torch.tensor([0.5, 1.0, 0.5, 1.0]))
    snapshot = motion.consume_training_activation_counters()
    assert snapshot["v2_strike_window_eligible_imitation_sample_count"].item() == 2
    assert snapshot[
        "v2_quarter_scaled_strike_window_imitation_sample_count"
    ].item() == 0


def test_v1_v2_disabled_ledgers_are_zero_without_rng_or_reward_change(monkeypatch):
    motion = _activation_motion_for_rewards(3, v1=False, v2_scale=None)
    env = _fake_env(motion=motion)

    def _unexpected_rng(*args, **kwargs):
        raise AssertionError("disabled V1/V2 reward instrumentation must not sample RNG")

    monkeypatch.setattr(torch, "rand", _unexpected_rng)
    result = rewards_mod.motion_global_body_linear_velocity_error_exp(
        env, "motion", std=1.0, body_names=["torso_Link"]
    )
    assert torch.equal(result, torch.ones(3))
    snapshot = motion.consume_training_activation_counters()
    assert all(value.item() == 0 for value in snapshot.values())


def test_hitter_pure_velocity_imitation_is_swing_only():
    motion = _fake_motion_for_rewards(3)
    motion.in_hold = torch.tensor([False, True, False])
    # Make env2 non-perfect so the test proves normal swing reward math is preserved rather
    # than all rows coincidentally returning one.
    motion.robot_body_lin_vel_w[2, :, 0] = 1.0
    motion.robot_body_ang_vel_w[2, :, 1] = 1.0
    env = _fake_env(motion=motion)
    lin = hope_rewards_mod.motion_body_lin_vel_swing_only(env, "motion", std=1.0)
    ang = hope_rewards_mod.motion_body_ang_vel_swing_only(env, "motion", std=1.0)
    expected = torch.tensor([1.0, 0.0, torch.exp(torch.tensor(-1.0))])
    assert torch.allclose(lin, expected)
    assert torch.allclose(ang, expected)


def test_split_ready_transition_keeps_body_and_zero_velocity_imitation_active():
    motion = _fake_motion_for_rewards(2)
    motion.in_hold = torch.tensor([True, True])
    motion.imitation_eligible = torch.tensor([True, True])
    motion.robot_body_pos_w[1, :, 0] = 1.0
    motion.robot_body_lin_vel_w[1, :, 0] = 1.0
    env = _fake_env(motion=motion)

    pos = hope_rewards_mod.motion_body_pos_swing_only(
        env, "motion", std=1.0
    )
    vel = hope_rewards_mod.motion_body_lin_vel_swing_only(
        env, "motion", std=1.0
    )
    expected = torch.tensor([1.0, torch.exp(torch.tensor(-1.0))])
    assert torch.allclose(pos, expected)
    assert torch.allclose(vel, expected)


def test_split_ready_single_stroke_latch_is_a_terminal_mask():
    command = types.SimpleNamespace(
        action_ball_diagnostic_split_ready_teacher=True,
        action_ball_single_stroke_complete=torch.tensor([False, True]),
    )
    env = types.SimpleNamespace(
        num_envs=2,
        command_manager=types.SimpleNamespace(
            get_term=lambda name: command if name == "motion" else None
        ),
    )
    result = terminations_mod.action_ball_diagnostic_single_stroke_complete(
        env, "motion"
    )
    assert torch.equal(result, torch.tensor([False, True]))


def test_motion_in_hold_keeps_the_final_post_decrement_step_held():
    cmd = commands_mod.MotionCommand.__new__(commands_mod.MotionCommand)
    cmd.hold_counter = torch.tensor([1, 0], dtype=torch.long)
    cmd.metrics = {"in_hold": torch.zeros(2)}
    held = cmd.hold_counter > 0
    cmd.hold_counter = torch.clamp(cmd.hold_counter - 1, min=0)
    cmd.metrics["in_hold"] = held.float()
    # Env0's clock was frozen on this step, so its teacher/termination contract must still be
    # held even though the remaining counter is now zero.
    assert cmd.in_hold.tolist() == [True, False]
    cmd.metrics["in_hold"].zero_()  # next control step releases normally
    assert cmd.in_hold.tolist() == [False, False]


def test_motion_teacher_start_wait_uses_future_post_decrement_steps_only():
    cmd = commands_mod.MotionCommand.__new__(commands_mod.MotionCommand)
    cmd.num_envs = 3
    cmd._env = types.SimpleNamespace(step_dt=0.02)
    cmd.time_steps_f = torch.zeros(3, dtype=torch.float32)
    cmd.hold_counter = torch.tensor([3, 1, 0], dtype=torch.long)
    cmd.metrics = {"in_hold": torch.zeros(3)}

    # Reset/pre-update: all counter steps are still in the future.
    assert cmd.teacher_start_wait_remaining_s.tolist() == pytest.approx(
        [0.06, 0.02, 0.0]
    )

    # Normal update order snapshots held and then decrements.  The final frozen step remains
    # marked in_hold for reward/termination accounting, but there are zero future wait steps for
    # the next action, so its countdown must not OR in that historical metric bit.
    held = cmd.hold_counter > 0
    cmd.hold_counter = torch.clamp(cmd.hold_counter - 1, min=0)
    cmd.metrics["in_hold"] = held.float()
    assert cmd.in_hold.tolist() == [True, True, False]
    assert cmd.teacher_start_wait_remaining_s.tolist() == pytest.approx(
        [0.04, 0.0, 0.0]
    )


def test_reference_envelope_terminations_ignore_hold_only_when_explicit():
    n = 3
    body_names = ["left_wrist_yaw_link"]
    motion = types.SimpleNamespace(
        cfg=types.SimpleNamespace(body_names=body_names),
        anchor_pos_w=torch.zeros(n, 3),
        robot_anchor_pos_w=torch.zeros(n, 3),
        body_pos_relative_w=torch.zeros(n, 1, 3),
        robot_body_pos_w=torch.zeros(n, 1, 3),
        in_hold=torch.tensor([True, False, True]),
    )
    # Every env violates both reference envelopes. The hold-aware wrappers must retain only
    # env1; ignore_hold=False must be exactly the original behavior.
    motion.anchor_pos_w[:, 2] = 0.5
    motion.body_pos_relative_w[:, 0, 2] = 0.5
    env = _fake_env(motion=motion)
    for wrapped, raw in (
        (hope_rewards_mod.bad_anchor_pos_z_only_hold_aware,
         terminations_mod.bad_anchor_pos_z_only),
        (hope_rewards_mod.bad_motion_body_pos_z_only_hold_aware,
         terminations_mod.bad_motion_body_pos_z_only),
    ):
        kwargs = {"body_names": body_names} if "body_pos" in wrapped.__name__ else {}
        assert wrapped(env, "motion", 0.25, ignore_hold=True, **kwargs).tolist() == [False, True, False]
        assert torch.equal(
            wrapped(env, "motion", 0.25, ignore_hold=False, **kwargs),
            raw(env, "motion", 0.25, **kwargs),
        )


def test_ignore_hold_contract_fails_loud_without_mask():
    motion = types.SimpleNamespace(
        cfg=types.SimpleNamespace(body_names=[]),
        anchor_pos_w=torch.zeros(1, 3), robot_anchor_pos_w=torch.zeros(1, 3),
    )
    with pytest.raises(RuntimeError, match="in_hold"):
        hope_rewards_mod.bad_anchor_pos_z_only_hold_aware(
            _fake_env(motion=motion), "motion", 0.25, ignore_hold=True
        )


def _torque_cmd(tau=None, limits=None, indices=(0, 1)):
    joint_count = max(
        2,
        0 if tau is None else tau.shape[1],
        0 if limits is None else limits.shape[1],
        0 if not indices else max(indices) + 1,
    )
    robot = types.SimpleNamespace(
        find_joints=lambda expr: (list(indices), None),
        data=types.SimpleNamespace(
            computed_torque=tau,
            joint_effort_limits=limits,
            joint_names=tuple(f"joint_{i}" for i in range(joint_count)),
        ),
        actuators={
            "test_explicit": types.SimpleNamespace(
                joint_indices=list(range(joint_count)),
                is_implicit_model=False,
            )
        },
    )
    return types.SimpleNamespace(
        robot=robot, num_envs=2, device="cpu", metrics={},
    )


def test_arm_torque_saturation_uses_heterogeneous_joint_limits():
    # Models the important A3 asymmetry directly: 24 Nm elbow/yaw vs 60 Nm shoulder pitch/roll.
    tau = torch.tensor([[24.0, 120.0], [48.0, 30.0]])
    limits = torch.tensor([[24.0, 60.0], [24.0, 60.0]])
    cmd = _torque_cmd(tau, limits)
    value = hope_rewards_mod.arm_torque_saturation(_fake_env(racket_target=cmd), "racket_target")
    assert torch.allclose(value, torch.tensor([0.5, 0.5]))
    assert torch.equal(cmd.metrics["arm_torque_sat_frac"], value)


@pytest.mark.parametrize(
    "cmd,match",
    [
        (_torque_cmd(None, None), "requires robot.data.computed_torque"),
        (_torque_cmd(torch.zeros(2, 2), torch.tensor([[24.0, float("nan")]] * 2)),
         "non-finite"),
        (_torque_cmd(torch.zeros(2, 2), torch.tensor([[24.0, 0.0]] * 2)),
         "strictly positive"),
        (_torque_cmd(torch.zeros(2, 2), torch.ones(2, 2), indices=()),
         "resolved zero"),
    ],
)
def test_arm_torque_saturation_fails_loud_when_mechanism_unavailable(cmd, match):
    with pytest.raises(RuntimeError, match=match):
        hope_rewards_mod.arm_torque_saturation(
            _fake_env(racket_target=cmd), "racket_target"
        )


# --------------------------------------------------------------------------------------------- #
# 1c split windows — timing masks + per-channel gating
# --------------------------------------------------------------------------------------------- #
def _timing_cmd(time_steps, pos_s=None, wide_s=None, window_s=0.12):
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.cfg = types.SimpleNamespace(
        strike_phase=0.5, strike_phase_per_clip=(), strike_window_s=window_s,
        strike_window_pos_s=pos_s, strike_window_wide_s=wide_s)
    rt._env = types.SimpleNamespace(step_dt=0.02)
    fake_motion = types.SimpleNamespace(
        _multiseg=False, motion=types.SimpleNamespace(time_step_total=101),
        retiming_active=False, time_steps=torch.tensor(time_steps))
    rt._motion = lambda: fake_motion
    rt._strike_phase_per_clip_t = None
    # 真 __init__ 无条件设置(hope_commands.py:633);_compute_strike_timing 末尾会读。
    rt.planner_revision_enabled = False
    rt._compute_strike_timing()
    return rt


def test_1c_split_masks():
    # strike step = round(0.5 * 100) = 50; tts = (50 - ts) * 0.02
    rt = _timing_cmd([50, 49, 45, 30], pos_s=0.02, wide_s=0.10)
    assert rt.strike_window_pos.tolist() == [True, True, False, False]     # |tts| <= 0.02
    assert rt.strike_window_wide.tolist() == [True, True, True, False]     # |tts| <= 0.10
    assert rt.strike_window.tolist() == [True, True, True, False]          # base 0.12 window


def test_1c_default_masks_alias_legacy_window():
    rt = _timing_cmd([50, 49, 45, 30])
    assert rt.strike_window_pos is rt.strike_window
    assert rt.strike_window_wide is rt.strike_window


def _timing_cmd_multiseg(spc):
    """Multiseg twin of _timing_cmd: 2 segments (20 + 15 frames), per-clip strike phase table."""
    rt = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    rt.cfg = types.SimpleNamespace(
        strike_phase=0.5, strike_phase_per_clip=spc, strike_window_s=0.12,
        strike_window_pos_s=None, strike_window_wide_s=None)
    rt._env = types.SimpleNamespace(step_dt=0.02)
    rt.device = "cpu"
    rt._strike_phase_per_clip_t = None
    rt.planner_revision_enabled = False  # 同 _timing_cmd:真 __init__ 无条件设置
    fake_ml = types.SimpleNamespace(
        num_segments=2, seg_start=torch.tensor([0, 20]), seg_len=torch.tensor([20, 15]))
    fake_motion = types.SimpleNamespace(
        _multiseg=True, motion=fake_ml, retiming_active=False,
        time_steps=torch.tensor([0, 25]), clip_id=torch.tensor([0, 1]))
    rt._motion = lambda: fake_motion
    return rt


def test_strike_phase_per_clip_length_mismatch_fails_loud():
    """人话:每 clip 击球点表和加载的 clip 数对不上,必须当场报错——以前会悄悄退回全局 strike_phase,
    每个 env 都在错误帧上找击球点还不吭声(fail-loud 文化,取证副产品 2026-07-09)。"""
    rt = _timing_cmd_multiseg((0.4, 0.5, 0.6))  # 3 entries, 2 loaded segments
    with pytest.raises(ValueError, match="strike_phase_per_clip"):
        rt._compute_strike_timing()
    with pytest.raises(ValueError, match="strike_phase_per_clip"):
        rt._strike_frame_for_clip(rt._motion().motion, 0)


def test_strike_phase_per_clip_empty_falls_back_and_matched_table_is_used():
    rt = _timing_cmd_multiseg(())  # () stays the documented "use global strike_phase" default
    rt._compute_strike_timing()
    assert torch.allclose(rt._strike_phase_per_clip_t, torch.tensor([0.5, 0.5]))
    step, phase, seg_start, seg_len = rt._strike_frame_for_clip(rt._motion().motion, 1)
    assert (step, phase, seg_start, seg_len) == (20 + round(0.5 * 14), 0.5, 20, 15)
    rt2 = _timing_cmd_multiseg((0.4, 0.6))  # matched table is taken verbatim
    rt2._compute_strike_timing()
    assert torch.allclose(rt2._strike_phase_per_clip_t, torch.tensor([0.4, 0.6]))
    step2, phase2, _, _ = rt2._strike_frame_for_clip(rt2._motion().motion, 0)
    assert (step2, phase2) == (round(0.4 * 19), 0.4)


def test_1c_rewards_gate_per_channel():
    n = 3
    pos_win = torch.tensor([True, False, False])
    wide_win = torch.tensor([True, True, False])
    cmd = _fake_racket_cmd(n, window=wide_win, window_pos=pos_win, window_wide=wide_win)
    env = _fake_env(racket_target=cmd)
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=0.2)
    rv = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    rn = hope_rewards_mod.racket_normal_tracking_exp(env, "racket_target", std=0.3)
    assert torch.allclose(rp, torch.tensor([1.0, 0.0, 0.0]))   # tight window
    assert torch.allclose(rv, torch.tensor([1.0, 1.0, 0.0]))   # wide window
    assert torch.allclose(rn, torch.tensor([1.0, 1.0, 0.0]))   # wide window
    # multiplicative success = the LEGACY window (R3b fix: the bonus channel must NOT be narrowed
    # to the window intersection when the split windows are active). Here strike_window == wide_win.
    rs = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs, torch.tensor([1.0, 1.0, 0.0]))


def test_racket_position_coarse_shares_swing_target_and_tight_window():
    pos_win = torch.tensor([True, True, True, False])
    cmd = _fake_racket_cmd(4, window=pos_win, window_pos=pos_win)
    cmd.racket_pos_w = torch.tensor(
        [[0.0, 0.0, 0.0], [0.075, 0.0, 0.0], [0.30, 0.0, 0.0], [0.30, 0.0, 0.0]]
    )
    env = _fake_env(racket_target=cmd)

    fine = hope_rewards_mod.racket_position_tracking_exp(
        env, "racket_target", std=0.075
    )
    coarse = hope_rewards_mod.racket_position_coarse_tracking_exp(
        env, "racket_target", std=0.30
    )

    assert torch.allclose(fine[:3], torch.exp(-torch.tensor([0.0, 1.0, 16.0])))
    assert torch.allclose(coarse[:3], torch.exp(-torch.tensor([0.0, 0.0625, 1.0])))
    assert fine[2] < 1.0e-6
    assert coarse[2] == pytest.approx(math.exp(-1.0))
    assert fine[3] == 0.0 and coarse[3] == 0.0


def test_1c_strike_success_keeps_legacy_window_not_intersection():
    """R3b zero-return regression (2026-07-08): with split windows on, racket_strike_success must pay
    across the FULL legacy strike window — not just the ±0.02 s tight intersection. Before the fix the
    expected value below read [1, 0, 0, 0] (support collapsed to the tight window)."""
    n = 4
    legacy = torch.tensor([True, True, True, False])     # strike_window_s (e.g. ±0.12 s)
    pos_win = torch.tensor([True, False, False, False])  # strike_window_pos_s = 0.02
    wide_win = torch.tensor([True, True, False, False])  # strike_window_wide_s = 0.10
    cmd = _fake_racket_cmd(n, window=legacy, window_pos=pos_win, window_wide=wide_win)
    env = _fake_env(racket_target=cmd)
    rs = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs, torch.tensor([1.0, 1.0, 1.0, 0.0]))
    # the additive channels keep their own (narrower) windows — the fix must not widen them
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=0.2)
    rv = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    assert torch.allclose(rp, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(rv, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_1c_strike_success_default_path_byte_identical_and_kernels_match():
    """No split windows (pos/wide alias the legacy window): the fixed product must equal the OLD
    semantics rp*rv*rn exactly, including nonzero kernel values and the face_command re-anchor."""
    n = 2
    win = torch.tensor([True, False])
    cmd = _fake_racket_cmd(n, window=win)
    # nonzero errors so the kernels are not trivially 1
    cmd.racket_pos_w = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    cmd.racket_lin_vel_w = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cmd.racket_normal_w = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cmd.racket_normal_raw_w = cmd.racket_normal_w.clone()  # 无符号表:raw ≡ signed
    cmd.racket_target_normal_w = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    env = _fake_env(racket_target=cmd)
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=0.2)
    rv = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    rn = hope_rewards_mod.racket_normal_tracking_exp(env, "racket_target", std=0.3)
    rs = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs, rp * rv * rn)
    assert rs[0].item() > 0.0 and rs[1].item() == 0.0
    # face_command=True: the product's normal factor re-anchors to target_normal_cmd
    cmd.cfg.face_command = True
    cmd.target_normal_cmd = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])  # aligned -> kernel 1
    rs_fc = hope_rewards_mod.racket_strike_success(env, "racket_target", 0.2, 1.0, 0.3)
    assert torch.allclose(rs_fc, rp * rv)  # normal factor == 1 when re-anchored and aligned


# --------------------------------------------------------------------------------------------- #
# C2a proximity power-gate
# --------------------------------------------------------------------------------------------- #
def test_pos_gate_sigmoid_values_and_default():
    n = 3
    win = torch.ones(n, dtype=torch.bool)
    cmd = _fake_racket_cmd(n, window=win)
    cmd.racket_pos_w = torch.tensor([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.65, 0.0, 0.0]])
    env = _fake_env(racket_target=cmd)
    r = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0,
                                                      pos_gate_radius=0.15)
    assert abs(r[0].item() - torch.sigmoid(torch.tensor(3.0)).item()) < 1e-6  # in reach: ~0.95
    assert abs(r[1].item() - 0.5) < 1e-6                                       # at the gate edge
    assert r[2].item() < 1e-4                                                  # far out of reach
    # default = exactly ungated
    r_off = hope_rewards_mod.racket_velocity_tracking_exp(env, "racket_target", std=1.0)
    assert torch.allclose(r_off, torch.ones(n))
    # normal channel gets the same gate; position never does
    rn = hope_rewards_mod.racket_normal_tracking_exp(env, "racket_target", std=0.3,
                                                     pos_gate_radius=0.15)
    assert abs(rn[1].item() - 0.5) < 1e-6
    rp = hope_rewards_mod.racket_position_tracking_exp(env, "racket_target", std=10.0)
    assert rp[2].item() > 0.99  # position channel keeps paying at distance (reach gradient)


# --------------------------------------------------------------------------------------------- #
# B2 constant guidance
# --------------------------------------------------------------------------------------------- #
def test_racket_guidance_clamp_and_mask():
    n = 4
    cmd = _fake_racket_cmd(
        n,
        window=torch.tensor([False, False, True, False]),
        pre_strike=torch.tensor([True, True, False, False]),
    )
    cmd.racket_pos_w = torch.tensor(
        [[0.3, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
    env = _fake_env(racket_target=cmd)
    r = hope_rewards_mod.racket_guidance(env, "racket_target", d_max=0.5)
    # env0 pre-strike: dist 0.3; env1 pre-strike far: clamped 0.5; env2 in-window: 0.2;
    # env3 post-strike/post-window: unpaid (follow-through untouched)
    assert torch.allclose(r, torch.tensor([0.3, 0.5, 0.2, 0.0]))


def test_racket_face_guidance_clamp_mask_and_target_selection():
    """Face-angle guidance (M3c 死区解药): min(angle, theta_max) × (pre_strike|window) mask;
    face_command=True 时目标取需求面(target_normal_cmd),False 时取 clip 参考面。"""
    import math
    n = 4
    cmd = _fake_racket_cmd(
        n,
        window=torch.tensor([False, False, True, False]),
        pre_strike=torch.tensor([True, True, False, False]),
    )
    # achieved normals: 0° / 180°(反面,应被 theta_max=pi/2 截断)/ 90° / 90°(mask 外不付钱)
    cmd.racket_normal_w = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cmd.racket_normal_raw_w = cmd.racket_normal_w.clone()  # 无符号表:raw ≡ signed(_face_pair 读 raw)
    cmd.target_normal_cmd = torch.tensor([[1.0, 0.0, 0.0]] * n)
    cmd.racket_target_normal_w = torch.tensor([[0.0, 0.0, 1.0]] * n)
    cmd.cfg.face_command = True
    env = _fake_env(racket_target=cmd)
    r = hope_rewards_mod.racket_face_guidance(env, "racket_target", theta_max=math.pi / 2)
    assert torch.allclose(
        r, torch.tensor([0.0, math.pi / 2, math.pi / 2, 0.0]), atol=1e-5), r
    # face_command=False → 换 clip 参考面为目标:env0 的 achieved [1,0,0] vs [0,0,1] = 90°
    cmd.cfg.face_command = False
    r2 = hope_rewards_mod.racket_face_guidance(env, "racket_target", theta_max=math.pi / 2)
    assert abs(float(r2[0]) - math.pi / 2) < 1e-5, r2


def test_conditional_face_guidance_has_readiness_preserving_fixed_budget():
    """The term is [0,1], window-only, with no face gradient outside either readiness margin."""
    import math

    n = 7
    win = torch.tensor([True, True, True, True, False, True, True])
    cmd = _fake_racket_cmd(n, window=win, window_wide=win)
    cmd.cfg.face_command = True
    cmd.target_normal_cmd = torch.tensor([[1.0, 0.0, 0.0]]).expand(n, 3).clone()
    angles = torch.tensor([
        math.pi,
        0.1,
        math.pi,
        math.pi,
        math.pi,
        (0.262 + math.pi) / 2.0,
        math.pi,
    ])
    cmd.racket_normal_raw_w = torch.stack(
        [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
    ).detach().requires_grad_()
    cmd.racket_normal_w = cmd.racket_normal_raw_w.clone()

    # env2 is exactly at the 9.5cm outer position margin; env5 halfway through both compact
    # readiness ramps; env6 proves position is measured against the swing-through target-now.
    cmd.racket_pos_w[2, 0] = 0.095
    cmd.racket_pos_w[5, 0] = 0.085
    cmd.racket_lin_vel_w[3, 0] = 1.0
    cmd.racket_lin_vel_w[5, 0] = 0.75
    cmd.time_to_strike[6] = 0.1
    cmd.racket_target_vel_w[6, 0] = 1.0
    cmd.racket_lin_vel_w[6, 0] = 1.0
    cmd.racket_pos_w[6, 0] = -0.1

    r = hope_rewards_mod.racket_face_conditional_guidance(_fake_env(racket_target=cmd), "racket_target")
    # Within the time window an unready state keeps the fixed cost 1 instead of escaping the face
    # penalty.  A fully ready state pays only face_fraction; the halfway-ready/half-face case is .875.
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.875, 1.0]), atol=1e-5), r
    assert torch.all((r >= 0.0) & (r <= 1.0))
    assert torch.allclose(
        cmd.metrics["face_conditional_guidance_gate"],
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.25, 1.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        cmd.metrics["face_conditional_guidance_error_fraction"],
        torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0]),
        atol=1e-5,
    )
    assert torch.allclose(cmd.metrics["face_conditional_guidance_cost_fraction"], r)
    r.sum().backward()
    assert torch.isfinite(cmd.racket_normal_raw_w.grad).all()
    # Below-angle-floor and every compact-gate-zero row have no face gradient to compete with
    # position/velocity acquisition.  Exact 180 degrees is finite (its turn axis is ambiguous).
    assert torch.equal(
        cmd.racket_normal_raw_w.grad[[1, 2, 3, 4]],
        torch.zeros_like(cmd.racket_normal_raw_w.grad[[1, 2, 3, 4]]),
    )


def test_conditional_face_guidance_never_rewards_abandoning_readiness():
    """The scalar reward, not merely autograd, must prefer .094m/.94mps over escaping the outer gate."""
    errors = torch.tensor([0.070, 0.075, 0.085, 0.094, 0.095, 0.096])

    for dimension in ("position", "velocity"):
        cmd = _fake_racket_cmd(len(errors), window=torch.ones(len(errors), dtype=torch.bool))
        cmd.cfg.face_command = True
        cmd.target_normal_cmd = torch.tensor([[1.0, 0.0, 0.0]]).expand(len(errors), 3).clone()
        angle = torch.full((len(errors),), math.pi / 2.0)
        cmd.racket_normal_raw_w = torch.stack(
            [torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1
        )
        cmd.racket_normal_w = cmd.racket_normal_raw_w.clone()
        if dimension == "position":
            cmd.racket_pos_w[:, 0] = errors
        else:
            # Map the same ordered fractions across the 0.5 -> 1.0 m/s velocity readiness band.
            cmd.racket_lin_vel_w[:, 0] = (errors - 0.075) / 0.02 * 0.5 + 0.5

        cost = hope_rewards_mod.racket_face_conditional_guidance(
            _fake_env(racket_target=cmd), "racket_target"
        )
        weighted_reward = -0.4 * cost
        # As either error worsens, cost may rise but must never fall; a negative weight therefore
        # never makes a deliberately less-ready state preferable.
        assert torch.all(cost[1:] >= cost[:-1] - 1e-7), (dimension, cost)
        assert torch.all(weighted_reward[:-1] >= weighted_reward[1:] - 1e-7), (
            dimension,
            weighted_reward,
        )
        assert weighted_reward[3] > weighted_reward[5], (dimension, weighted_reward)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"theta_free": math.pi}, "theta_free < theta_max"),
        ({"pos_full": 0.1, "pos_zero": 0.095}, "pos_full < pos_zero"),
        ({"vel_full": 1.0, "vel_zero": 1.0}, "vel_full < vel_zero"),
        ({"pos_zero": float("nan")}, "finite bounds"),
    ],
)
def test_conditional_face_guidance_rejects_invalid_source_bounds(kwargs, message):
    cmd = _fake_racket_cmd(1, window=torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError, match=message):
        hope_rewards_mod.racket_face_conditional_guidance(
            _fake_env(racket_target=cmd), "racket_target", **kwargs
        )


# --------------------------------------------------------------------------------------------- #
# R-b envelope violation indicator
# --------------------------------------------------------------------------------------------- #
def test_tracking_envelope_violation_matches_removed_terminations():
    n = 3
    body_names = ["left_ankle_roll_Link", "right_ankle_roll_Link",
                  "left_wrist_yaw_Link", "right_wrist_yaw_Link"]
    all_bodies = ["pelvis_link"] + body_names
    nb = len(all_bodies)
    motion = types.SimpleNamespace(
        cfg=types.SimpleNamespace(body_names=all_bodies),
        anchor_pos_w=torch.zeros(n, 3), robot_anchor_pos_w=torch.zeros(n, 3),
        body_pos_relative_w=torch.zeros(n, nb, 3), robot_body_pos_w=torch.zeros(n, nb, 3),
    )
    motion.anchor_pos_w[0, 2] = 0.30              # env0: anchor z blows the 0.25 envelope
    motion.body_pos_relative_w[1, 2, 2] = 0.30    # env1: one envelope body z blows it
    motion.body_pos_relative_w[2, 0, 2] = 0.30    # env2: pelvis is NOT in the envelope list
    env = _fake_env(motion=motion)
    viol = hope_rewards_mod.tracking_envelope_violation(env, "motion", 0.25, body_names)
    assert viol.tolist() == [1.0, 1.0, 0.0]
    # exact agreement with the termination funcs it replaces
    t = terminations_mod.bad_anchor_pos_z_only(env, "motion", 0.25) | \
        terminations_mod.bad_motion_body_pos_z_only(env, "motion", 0.25, body_names)
    assert torch.equal(viol, t.float())
    motion.in_hold = torch.tensor([True, False, True])
    hold_aware = hope_rewards_mod.tracking_envelope_violation(
        env, "motion", 0.25, body_names, ignore_hold=True
    )
    assert hold_aware.tolist() == [0.0, 1.0, 0.0]


# --------------------------------------------------------------------------------------------- #
# R-a actor leg-reference mask
# --------------------------------------------------------------------------------------------- #
_A3_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # 31 joints, A3-style interleaved order


class _MaskRobot:
    def __init__(self, num_envs, joint_names):
        self.joint_names = list(joint_names)
        gen = torch.Generator().manual_seed(7)
        self.data = types.SimpleNamespace(
            default_joint_pos=torch.randn(num_envs, len(joint_names), generator=gen))

    def find_joints(self, exprs):
        ids, names = [], []
        for i, n in enumerate(self.joint_names):
            if any(re.fullmatch(e, n) for e in exprs):
                ids.append(i)
                names.append(n)
        return ids, names


def _mask_cmd(n=3, joints=_A3_JOINTS):
    gen = torch.Generator().manual_seed(11)
    nj = len(joints)
    return types.SimpleNamespace(
        joint_pos=torch.randn(n, nj, generator=gen),
        joint_vel=torch.randn(n, nj, generator=gen),
        robot=_MaskRobot(n, joints),
        device="cpu",
    )


def test_leg_mask_masks_exactly_24_dims():
    n = 3
    cmd = _mask_cmd(n)
    env = _fake_env(motion=cmd)
    out = hope_observations_mod.generated_commands_actor_leg_masked(env, "motion")
    nj = len(_A3_JOINTS)
    assert out.shape == (n, 2 * nj) == (n, 62)
    leg_ids, _ = cmd.robot.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
    assert len(leg_ids) == 12
    raw = torch.cat([cmd.joint_pos, cmd.joint_vel], dim=1)
    for j in range(nj):
        if j in leg_ids:
            assert torch.equal(out[:, j], cmd.robot.data.default_joint_pos[:, j]), j
            assert torch.all(out[:, j + nj] == 0.0), j
        else:
            assert torch.equal(out[:, j], raw[:, j]), j          # non-leg pos byte-identical
            assert torch.equal(out[:, j + nj], raw[:, j + nj]), j  # non-leg vel byte-identical
    # second call hits the cache and agrees
    out2 = hope_observations_mod.generated_commands_actor_leg_masked(env, "motion")
    assert torch.equal(out, out2)
    assert hasattr(cmd, "_actor_leg_mask_ids")


def test_leg_mask_refuses_wrong_joint_count():
    joints = [j for j in _A3_JOINTS if j != "left_ankle_roll_joint"]  # 11 leg joints only
    cmd = _mask_cmd(3, joints=joints)
    env = _fake_env(motion=cmd)
    with pytest.raises(RuntimeError, match="expected 12 leg joints"):
        hope_observations_mod.generated_commands_actor_leg_masked(env, "motion")


# --------------------------------------------------------------------------------------------- #
# A1 atomic target-message contract + reward gate diagnostics
# --------------------------------------------------------------------------------------------- #
def _make_a1_cmd(
    delay_steps=0,
    dropout_prob=0.0,
    ar1_sigma=0.0,
    tts_mode="live",
    planner_revision=False,
):
    cmd = hope_commands_mod.RacketTargetCommand.__new__(hope_commands_mod.RacketTargetCommand)
    cmd.num_envs = 1
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(
        step_dt=0.02,
        termination_manager=types.SimpleNamespace(active_terms=()),
    )
    cmd.metrics = {"actor_time_to_strike_s": torch.zeros(1)}
    cmd._actor_view_active = True
    cmd._delay_steps = delay_steps
    # 2026-07-25 耦合传输改造后,__init__ 会派生 _actor_ring_steps(耦合模式=0,legacy=d)。
    # 本文件的 A1 用例钉的是 legacy 观测延迟环本身的原子性/记账,故按 legacy 口径造假体;
    # 耦合模式(修订+延迟)的行为合同见 test_coupled_transport_delay.py。
    cmd._coupled_transport = False
    cmd._actor_ring_steps = delay_steps
    cmd._delay_tts_mode = tts_mode
    cmd._delay_tts_active = tts_mode != "live"
    cmd.planner_revision_enabled = planner_revision
    # 家族台账(c102b9e3 family-aware):真 __init__ 固定两族表(hope_commands.py:443),
    # 稀疏奖励 eligibility ledger 懒初始化时会读它;family/timing-bucket 记账还会取
    # _motion()——给单段假 motion(_multiseg=False、无 just_resampled → 零掩码路径)。
    cmd._clip_names = {0: "forehand", 1: "backhand"}
    cmd._motion = lambda: types.SimpleNamespace(_multiseg=False)
    cmd._atomic_tts_active = cmd._delay_tts_active or planner_revision
    cmd._delay_ptr = 0
    cmd._jitter_pos = 0.0
    cmd._jitter_vel = 0.0
    cmd._mnoise_white = 0.0
    cmd._mnoise_ar1_sigma = ar1_sigma
    cmd._mnoise_ar1_rho = 0.9
    cmd._mnoise_ar1_state = torch.full((1, 3), 7.0)
    cmd._drop_prob = dropout_prob
    cmd._post_strike_drop_steps = 0
    cmd._bias_per_swing = 0.0
    cmd._a1v2_active = dropout_prob > 0.0
    cmd.time_to_strike = torch.tensor([0.5])
    cmd.pre_strike = torch.tensor([True])
    cmd.racket_target_pos_w = torch.tensor([[1.0, 2.0, 3.0]])
    cmd.racket_target_vel_w = torch.tensor([[4.0, 5.0, 6.0]])
    cmd.target_normal_cmd = torch.tensor([[0.0, 1.0, 0.0]])
    cmd.swing_sign = torch.tensor([1.0])
    cmd.delayed_racket_target_pos_w = cmd.racket_target_pos_w.clone()
    cmd.delayed_racket_target_vel_w = cmd.racket_target_vel_w.clone()
    cmd.delayed_target_normal_cmd = cmd.target_normal_cmd.clone()
    cmd.delayed_swing_sign = cmd.swing_sign.clone()
    if cmd._atomic_tts_active:
        cmd.delayed_time_to_strike = cmd.time_to_strike.clone()
    if planner_revision:
        cmd._planner_initial_tts_mixture = None
        cmd._planner_control_epoch = torch.tensor([7], dtype=torch.long)
        cmd._planner_task_id = torch.tensor([10], dtype=torch.long)
        cmd._planner_task_revision = torch.tensor([1], dtype=torch.long)
        cmd._planner_visible_pos = cmd.racket_target_pos_w.clone()
        cmd._planner_visible_vel = cmd.racket_target_vel_w.clone()
        cmd._planner_visible_normal = cmd.target_normal_cmd.clone()
        cmd._planner_visible_tts = cmd.time_to_strike.clone()
        cmd._planner_visible_last_precontact = torch.zeros(1, dtype=torch.bool)
        cmd._planner_actor_control_epoch = cmd._planner_control_epoch.clone()
        cmd._planner_actor_task_id = cmd._planner_task_id.clone()
        cmd._planner_actor_task_revision = cmd._planner_task_revision.clone()
    if delay_steps > 0:
        length = delay_steps + 1
        cmd._delay_buf_pos = cmd.racket_target_pos_w.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_vel = cmd.racket_target_vel_w.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_normal = cmd.target_normal_cmd.unsqueeze(0).repeat(length, 1, 1)
        cmd._delay_buf_sign = cmd.swing_sign.unsqueeze(0).repeat(length, 1)
        if cmd._atomic_tts_active:
            cmd._delay_buf_tts = cmd.time_to_strike.unsqueeze(0).repeat(length, 1)
        if planner_revision:
            cmd._delay_buf_planner_epoch = cmd._planner_control_epoch.unsqueeze(0).repeat(length, 1)
            cmd._delay_buf_planner_task = cmd._planner_task_id.unsqueeze(0).repeat(length, 1)
            cmd._delay_buf_planner_revision = cmd._planner_task_revision.unsqueeze(0).repeat(length, 1)
            cmd._delay_buf_planner_last_precontact = torch.zeros(
                length, 1, dtype=torch.bool
            )
    if cmd._a1v2_active:
        cmd._swing_bias = torch.zeros(1, 3)
        cmd._drop_cd = torch.zeros(1, dtype=torch.long)
        cmd._prev_pre_strike = torch.ones(1, dtype=torch.bool)
        cmd._held_pos = cmd.racket_target_pos_w.clone()
        cmd._held_vel = cmd.racket_target_vel_w.clone()
        cmd._held_normal = cmd.target_normal_cmd.clone()
        cmd._held_sign = cmd.swing_sign.clone()
        if cmd._atomic_tts_active:
            cmd._held_tts = cmd.time_to_strike.clone()
        if planner_revision:
            cmd._held_planner_epoch = cmd._planner_control_epoch.clone()
            cmd._held_planner_task = cmd._planner_task_id.clone()
            cmd._held_planner_revision = cmd._planner_task_revision.clone()
            cmd._held_planner_last_precontact = torch.zeros(1, dtype=torch.bool)
    return cmd


def _set_question_b(cmd):
    cmd.racket_target_pos_w[:] = torch.tensor([[10.0, 20.0, 30.0]])
    cmd.racket_target_vel_w[:] = torch.tensor([[40.0, 50.0, 60.0]])
    cmd.target_normal_cmd[:] = torch.tensor([[1.0, 0.0, 0.0]])
    cmd.swing_sign[:] = -1.0


def _assert_actor_question(cmd, which):
    if which == "a":
        assert torch.equal(cmd.delayed_racket_target_pos_w, torch.tensor([[1.0, 2.0, 3.0]]))
        assert torch.equal(cmd.delayed_racket_target_vel_w, torch.tensor([[4.0, 5.0, 6.0]]))
        assert torch.equal(cmd.delayed_target_normal_cmd, torch.tensor([[0.0, 1.0, 0.0]]))
        assert torch.equal(cmd.delayed_swing_sign, torch.tensor([1.0]))
    else:
        assert torch.equal(cmd.delayed_racket_target_pos_w, torch.tensor([[10.0, 20.0, 30.0]]))
        assert torch.equal(cmd.delayed_racket_target_vel_w, torch.tensor([[40.0, 50.0, 60.0]]))
        assert torch.equal(cmd.delayed_target_normal_cmd, torch.tensor([[1.0, 0.0, 0.0]]))
        assert torch.equal(cmd.delayed_swing_sign, torch.tensor([-1.0]))


def test_a1_delay_keeps_face_command_atomic_with_pos_vel_and_side():
    cmd = _make_a1_cmd(delay_steps=1)
    _set_question_b(cmd)
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    # The face observation must read the same delayed message, not the live question B normal.
    obs = hope_observations_mod.racket_target_normal_cmd(
        _fake_env(racket_target=cmd), "racket_target"
    )
    assert torch.equal(obs, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    cmd._push_actor_target()
    _assert_actor_question(cmd, "b")


@pytest.mark.parametrize(
    ("mode", "expected_first_tts", "expected_delayed_tts"),
    [
        ("source_timestamp_compensated", 0.46, 0.44),
        ("uncompensated", 0.50, 0.48),
    ],
)
def test_a1_delay_two_keeps_tts_atomic_and_distinguishes_timestamp_compensation(
    mode, expected_first_tts, expected_delayed_tts
):
    cmd = _make_a1_cmd(delay_steps=2, tts_mode=mode)
    _set_question_b(cmd)

    # The two backfilled source tuples both carry question A/TTS=0.50.  Compensation advances
    # that source TTS by the known 2*20 ms transport age; the negative control exposes 0.50 stale.
    cmd.time_to_strike[:] = 0.48
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    assert cmd.actor_time_to_strike().item() == pytest.approx(expected_first_tts)

    cmd.time_to_strike[:] = 0.46
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")

    # Third push reads the first question-B source tuple (source TTS=0.48) atomically with B.
    cmd.time_to_strike[:] = 0.44
    cmd._push_actor_target()
    _assert_actor_question(cmd, "b")
    assert cmd.actor_time_to_strike().item() == pytest.approx(expected_delayed_tts)
    assert cmd.metrics["actor_time_to_strike_s"].item() == pytest.approx(expected_delayed_tts)


def test_a1_default_live_tts_is_the_live_tensor_alias_even_when_targets_are_delayed():
    cmd = _make_a1_cmd(delay_steps=2, tts_mode="live")
    assert cmd.actor_time_to_strike() is cmd.time_to_strike
    cmd.time_to_strike[:] = 0.37
    _set_question_b(cmd)
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    assert cmd.actor_time_to_strike().item() == pytest.approx(0.37)
    obs = hope_observations_mod.actor_time_to_strike(
        _fake_env(racket_target=cmd), "racket_target"
    )
    assert obs.item() == pytest.approx(0.37)


@pytest.mark.parametrize("mode", ["source_timestamp_compensated", "uncompensated"])
def test_a1_atomic_tts_zero_delay_is_numerically_live(mode):
    cmd = _make_a1_cmd(delay_steps=0, tts_mode=mode)
    cmd.time_to_strike[:] = 0.31
    cmd._push_actor_target()
    assert cmd.actor_time_to_strike().item() == pytest.approx(0.31)


def test_planner_revision_actor_delivery_counts_only_after_atomic_ring_materializes():
    cmd = _make_a1_cmd(
        delay_steps=2,
        tts_mode="source_timestamp_compensated",
        planner_revision=True,
    )
    cmd._planner_task_revision[:] = 2
    cmd._planner_visible_pos[:] = torch.tensor([[10.0, 20.0, 30.0]])
    cmd._planner_visible_vel[:] = torch.tensor([[40.0, 50.0, 60.0]])
    cmd._planner_visible_normal[:] = torch.tensor([[1.0, 0.0, 0.0]])
    cmd._planner_visible_tts[:] = 0.02
    cmd._planner_visible_last_precontact[:] = True

    cmd._push_actor_target()
    cmd._push_actor_target()
    before_delivery = cmd.consume_exact_behavior_decision_counters()
    assert before_delivery["planner_revision_actor_visible_count"].item() == 0
    assert before_delivery["planner_revision_last_precontact_actor_visible_count"].item() == 0

    cmd._push_actor_target()
    delivered = cmd.consume_exact_behavior_decision_counters()
    assert delivered["planner_revision_actor_visible_count"].item() == 1
    assert delivered["planner_revision_last_precontact_actor_visible_count"].item() == 1
    assert cmd.actor_time_to_strike().item() == 0.0

    cmd._push_actor_target()
    duplicate = cmd.consume_exact_behavior_decision_counters()
    assert duplicate["planner_revision_actor_visible_count"].item() == 0
    assert duplicate["planner_revision_last_precontact_actor_visible_count"].item() == 0


def test_planner_revision_delay_zero_counts_same_step_and_clamps_postcontact_tts():
    cmd = _make_a1_cmd(
        delay_steps=0,
        tts_mode="source_timestamp_compensated",
        planner_revision=True,
    )
    cmd._planner_task_revision[:] = 2
    cmd._planner_visible_tts[:] = -0.04
    cmd._planner_visible_last_precontact[:] = True
    cmd._push_actor_target()
    delivered = cmd.consume_exact_behavior_decision_counters()
    assert delivered["planner_revision_actor_visible_count"].item() == 1
    assert delivered["planner_revision_last_precontact_actor_visible_count"].item() == 1
    assert cmd.actor_time_to_strike().item() == 0.0


def test_delayed_final_revision_arriving_after_contact_is_not_precontact_visible():
    cmd = _make_a1_cmd(
        delay_steps=2,
        tts_mode="source_timestamp_compensated",
        planner_revision=True,
    )
    cmd._planner_task_revision[:] = 2
    cmd._planner_visible_tts[:] = 0.02
    cmd._planner_visible_last_precontact[:] = True
    cmd._push_actor_target()
    cmd._push_actor_target()
    cmd.pre_strike[:] = False
    cmd._push_actor_target()
    delivered = cmd.consume_exact_behavior_decision_counters()
    assert delivered["planner_revision_actor_visible_count"].item() == 1
    assert delivered["planner_revision_last_precontact_actor_visible_count"].item() == 0


def test_a1_dropout_holds_the_entire_target_message():
    cmd = _make_a1_cmd(dropout_prob=1.0)
    _set_question_b(cmd)
    cmd._push_actor_target()
    _assert_actor_question(cmd, "a")
    cmd._drop_prob = 0.0
    cmd._push_actor_target()
    _assert_actor_question(cmd, "b")


def test_a1_true_reset_clears_state_and_backfills_all_message_fields():
    cmd = _make_a1_cmd(
        delay_steps=2,
        dropout_prob=1.0,
        ar1_sigma=0.2,
        tts_mode="source_timestamp_compensated",
    )
    _set_question_b(cmd)
    cmd.time_to_strike[:] = 0.63
    cmd._drop_cd[:] = 9
    cmd._swing_bias[:] = 3.0
    cmd._held_pos[:] = -3.0
    cmd._held_vel[:] = -4.0
    cmd._held_normal[:] = -5.0
    cmd._held_sign[:] = 0.0
    cmd._held_tts[:] = -6.0
    cmd.delayed_time_to_strike[:] = -7.0
    cmd._reset_actor_target_state(torch.tensor([0]))
    _assert_actor_question(cmd, "b")
    assert torch.all(cmd._delay_buf_pos == cmd.racket_target_pos_w.unsqueeze(0))
    assert torch.all(cmd._delay_buf_vel == cmd.racket_target_vel_w.unsqueeze(0))
    assert torch.all(cmd._delay_buf_normal == cmd.target_normal_cmd.unsqueeze(0))
    assert torch.all(cmd._delay_buf_sign == cmd.swing_sign.unsqueeze(0))
    assert torch.all(cmd._delay_buf_tts == cmd.time_to_strike.unsqueeze(0))
    assert torch.equal(cmd.delayed_time_to_strike, cmd.time_to_strike)
    assert torch.all(cmd._mnoise_ar1_state == 0.0)
    assert torch.all(cmd._swing_bias == 0.0)
    assert torch.all(cmd._drop_cd == 0)
    assert torch.equal(cmd._held_normal, cmd.target_normal_cmd)
    assert torch.equal(cmd._held_sign, cmd.swing_sign)
    assert torch.equal(cmd._held_tts, cmd.time_to_strike)


def test_debug_reward_log_records_real_gate_instead_of_identity():
    cmd = types.SimpleNamespace(
        cfg=types.SimpleNamespace(debug_reward_logging=True),
        metrics={"dbg_probe_raw": torch.zeros(2), "dbg_probe_gated": torch.zeros(2)},
    )
    raw = torch.tensor([0.25, 0.75])
    hope_rewards_mod._dbg_log(cmd, "probe", raw, torch.tensor([True, False]))
    assert torch.equal(cmd.metrics["dbg_probe_raw"], raw)
    assert torch.equal(cmd.metrics["dbg_probe_gated"], torch.tensor([0.25, 0.0]))


def test_base_decel_is_disabled_during_frozen_hold():
    motion = types.SimpleNamespace(in_hold=torch.tensor([True, False]))
    cmd = types.SimpleNamespace(
        racket_target_pos_w=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        racket_pos_w=torch.zeros(2, 3),
        robot=types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_w=torch.zeros(2, 3))),
        pre_strike=torch.tensor([True, True]),
        _motion=lambda: motion,
    )
    reward = hope_rewards_mod.base_decel_tracking(
        _fake_env(racket_target=cmd), "racket_target", v_gain=1.0, v_max=2.0, std=1.0
    )
    assert reward[0] == 0.0
    assert reward[1] > 0.0


def test_base_decel_observer_is_a_real_reward_term_not_a_command_stage_hook():
    # Isaac 2.1 executes reward -> reset -> command.  Both paired arms therefore must enter through
    # RewardManager; any command-stage observer would compare a next-state control against an
    # old-state treatment.  Keep this regression tied to the shipped class/config source.
    command_source = inspect.getsource(hope_commands_mod.RacketTargetCommand)
    assert "_observe_base_decel_activation" not in command_source
    assert "base_decel_activation_enabled" not in command_source

    cfg_path = os.path.abspath(
        os.path.join(MDP_DIR, "..", "config", "agibot_a3", "hope_env_cfg.py")
    )
    with open(cfg_path, encoding="utf-8") as handle:
        config_source = handle.read()
    probe_decl = config_source.index("base_decel_activation_probe = RewTerm(")
    treatment_decl = config_source.index("base_decel = RewTerm(", probe_decl)
    assert probe_decl < treatment_decl
    assert "func=mdp.base_decel_activation_probe, weight=0.0" in config_source[
        probe_decl:treatment_decl
    ]


# --------------------------------------------------------------------------------------------- #
# R-c MotionCommand birth fixes (real MotionCommand on synthetic clips)
# --------------------------------------------------------------------------------------------- #
_BODY_NAMES = list(
    commands_mod.MotionCommand._canonical_registry_module().RUNTIME_BODY_NAMES
)
_N_JOINTS = 31
_STAND_Z = 1.0684
_CROUCH_Z = 0.78


def _write_motion_npz(
    path,
    frames,
    root_z=_CROUCH_Z,
    *,
    fps=50,
    metadata_body_names=None,
):
    quat = np.zeros((frames, len(_BODY_NAMES), 4), dtype=np.float32)
    quat[..., 0] = 1.0
    body_pos = np.zeros((frames, len(_BODY_NAMES), 3), dtype=np.float32)
    body_pos[:, :, 2] = root_z
    np.savez(
        path,
        fps=fps,
        joint_pos=np.zeros((frames, _N_JOINTS), dtype=np.float32),
        joint_vel=np.zeros((frames, _N_JOINTS), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((frames, len(_BODY_NAMES), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, len(_BODY_NAMES), 3), dtype=np.float32),
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=np.asarray(metadata_body_names or _BODY_NAMES),
    )
    return path


def _write_canonical_ready_motion_npz(
    path,
    *,
    interior_scale=1.0,
    start_joint_offset=0.0,
    end_joint_offset=0.0,
    start_body_offset=0.0,
    end_body_offset=0.0,
    endpoint_joint_velocity=0.0,
):
    """Write a tiny schema-2 clip whose normal case starts/ends at one non-default ready."""

    frames = 5
    ready_joint = np.linspace(-0.3, 0.3, _N_JOINTS, dtype=np.float32)
    joint_pos = np.repeat(ready_joint[None, :], frames, axis=0)
    joint_pos[0, 0] += np.float32(start_joint_offset)
    joint_pos[-1, 0] += np.float32(end_joint_offset)
    joint_pos[1, 0] += np.float32(0.11 * interior_scale)
    joint_pos[2, 1] -= np.float32(0.17 * interior_scale)
    joint_pos[3, 2] += np.float32(0.09 * interior_scale)

    joint_vel = np.zeros_like(joint_pos)
    joint_vel[0, 0] = np.float32(endpoint_joint_velocity)
    joint_vel[-1, 0] = np.float32(endpoint_joint_velocity)
    joint_vel[1:-1, :3] = np.float32(0.25 * interior_scale)

    ready_body_pos = np.zeros((len(_BODY_NAMES), 3), dtype=np.float32)
    ready_body_pos[:, 2] = np.float32(0.93)
    ready_body_pos[0] = np.array([0.08, -0.03, 0.93], dtype=np.float32)
    ready_body_pos[_BODY_NAMES.index("torso_Link")] = np.array(
        [0.09, -0.02, 1.24], dtype=np.float32
    )
    body_pos = np.repeat(ready_body_pos[None, :, :], frames, axis=0)
    body_pos[0, 0, 0] += np.float32(start_body_offset)
    body_pos[-1, 0, 0] += np.float32(end_body_offset)
    body_pos[1, :, 0] += np.float32(0.04 * interior_scale)
    body_pos[2, :, 1] -= np.float32(0.03 * interior_scale)

    body_quat = np.zeros((frames, len(_BODY_NAMES), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    half_yaw = np.float32(0.08 * interior_scale)
    body_quat[1:-1, :, 0] = np.cos(half_yaw)
    body_quat[1:-1, :, 3] = np.sin(half_yaw)

    body_lin_vel = np.zeros((frames, len(_BODY_NAMES), 3), dtype=np.float32)
    body_ang_vel = np.zeros_like(body_lin_vel)
    body_lin_vel[1:-1, :, 0] = np.float32(0.4 * interior_scale)
    body_ang_vel[1:-1, :, 2] = np.float32(0.6 * interior_scale)

    np.savez(
        path,
        fps=np.array([50.0], dtype=np.float64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=np.asarray(_BODY_NAMES),
    )
    return path


_CANONICAL_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
_CANONICAL_FAMILIES = (
    "forehand",
    "backhand",
    "forehand",
    "backhand",
    "forehand",
)
_CANONICAL_FACE_SIGNS = (1.0, -1.0, 1.0, -1.0, 1.0)


def _file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _write_registry_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return _file_sha256(path)


def _write_runtime_receipt(repo_root: Path, relative: str, label: str):
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fixture:{label}\n".encode("utf-8"))
    return {
        "path": relative,
        "sha256": _file_sha256(path),
    }


def _complete_runtime_bank_gate_report(
    binding,
    repo_root: Path,
    ready_path: Path,
):
    digest = lambda label: hashlib.sha256(label.encode("utf-8")).hexdigest()
    manifest = _write_runtime_receipt(
        repo_root, "canonical_registry_contract/bank/BUILD_MANIFEST.json", "manifest"
    )
    recipe = _write_runtime_receipt(
        repo_root, "canonical_registry_contract/recipe.json", "recipe"
    )
    compiler = _write_runtime_receipt(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_compiler.py",
        "compiler",
    )
    geometry = _write_runtime_receipt(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_geometry.py",
        "geometry",
    )
    mjcf = _write_runtime_receipt(
        repo_root, "canonical_registry_contract/a3.xml", "mjcf"
    )
    urdf = _write_runtime_receipt(
        repo_root, "canonical_registry_contract/a3.urdf", "urdf"
    )
    body_order = _write_runtime_receipt(
        repo_root,
        "canonical_registry_contract/body_order.txt",
        "body-order",
    )
    gate_tool = _write_runtime_receipt(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_motion_bank_gate.py",
        "gate",
    )
    player_tool = _write_runtime_receipt(
        repo_root,
        "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py",
        "player",
    )
    dynamics_tool = _write_runtime_receipt(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_mujoco_dynamics_gate.py",
        "dynamics",
    )
    schema2_tool = _write_runtime_receipt(
        repo_root,
        "hope_training/whole_body_tracking/scripts/"
        "canonical_schema2_builder.py",
        "schema2-builder",
    )
    ready_receipt = {
        "path": ready_path.relative_to(repo_root).as_posix(),
        "sha256": _file_sha256(ready_path),
    }
    endpoint_zero = {
        "joint_start": True,
        "joint_end": True,
        "body_linear_start": True,
        "body_linear_end": True,
        "body_angular_start": True,
        "body_angular_end": True,
    }
    by_motion = dict(zip(binding.motion_ids, binding.npz_sha256))
    clips = [
        {
            "motion_id": motion_id,
            "scope": scope,
            "filename": f"{motion_id}_{scope}.npz",
            "sha256": (
                by_motion[motion_id]
                if scope == binding.scope
                else digest(f"{motion_id}:{scope}")
            ),
            "frames": 5,
            "fps": 50.0,
            "duration_s": 0.08,
            "schema2_receipts": {
                "input_sha256": digest(f"{motion_id}:{scope}:input"),
                "builder_tool_sha256": schema2_tool["sha256"],
                "manifest_sidecar": _write_runtime_receipt(
                    repo_root,
                    (
                        "canonical_registry_contract/bank/"
                        f"{motion_id}_{scope}.manifest.json"
                    ),
                    f"{motion_id}:{scope}:manifest",
                ),
                "report_sidecar": _write_runtime_receipt(
                    repo_root,
                    (
                        "canonical_registry_contract/bank/"
                        f"{motion_id}_{scope}.report.json"
                    ),
                    f"{motion_id}:{scope}:report",
                ),
            },
            "strict_schema2_and_ready": {
                "shared_joint_ready_exact": True,
                "shared_32_body_ready_exact": True,
                "six_velocity_classes_exact_zero": endpoint_zero,
            },
            "contact_opportunity": {
                "acceleration_allowed_through_window_end": True
            },
            "mujoco_fk": {"pass": True},
            "plant_specific_dynamics": {
                "verdict": "PASS",
                "screen_pass": True,
                "non_torque_screens_pass": True,
                "inverse_dynamics": {
                    "torque_interpretation": {"valid": True}
                },
            },
        }
        for motion_id in binding.motion_ids
        for scope in ("upper", "full")
    ]
    aggregate = {
        key: 0
        for key in commands_mod.MotionCommand._canonical_registry_module()
        .motion_admission._BANK_GATE_AGGREGATE_KEYS
    }
    aggregate.update(
        {
            key: 10
            for key in (
                "clip_count",
                "fk_pass_count",
                "velocity_consistency_pass_count",
                "joint_limit_pass_count",
                "geometry_pass_count",
                "non_torque_dynamics_pass_count",
                "complete_dynamics_pass_count",
                "torque_interpretation_valid_count",
            )
        }
    )
    return {
        "schema_version": 1,
        "verdict": "PASS",
        "bank_gate_pass": True,
        "candidate_integrity_pass": True,
        "grounded_trace_status": "COMPLETE_PASS",
        "publication_class": "post_build_diagnostic_only",
        "training_authorized": False,
        "hardware_authorized": False,
        "library_id": binding.bank_id,
        "manifest": manifest,
        "bank_dir": "bank",
        "bound_inputs": {
            "recipe": recipe,
            "compiler": compiler,
            "geometry_tool": geometry,
            "compiler_options_sha256": digest("options"),
            "ready": ready_receipt,
            "mjcf": mjcf,
            "urdf": urdf,
            "body_order": body_order,
            "plant": {
                "mjcf_sha256": mjcf["sha256"],
                "urdf_sha256": urdf["sha256"],
                "compiled_signature_sha256": digest("signature"),
                "identity_bound": True,
                "runtime_body_order": ["pelvis_link"],
            },
            "verifier_tools": {
                "bank_gate": gate_tool,
                "mujoco_motion_player": player_tool,
                "canonical_mujoco_dynamics_gate": {
                    **dynamics_tool,
                    "report_schema_version": 1,
                },
            },
        },
        "contracts": {
            "matrix": {
                "motion_ids": list(binding.motion_ids),
                "scopes": ["upper", "full"],
                "count": 10,
            },
            "shared_ready": True,
            "six_endpoint_velocity_classes_exact_zero": True,
            "contact_opportunity_is_marker_only": True,
            "acceleration_allowed_through_window_end": True,
            "nonnegative_scalar_acceleration_through_window_end": True,
            "adv2c3_role": "comparator_only_not_default",
            "grounded_inverse_dynamics": "complete",
            "grounded_trace_status": "COMPLETE_PASS",
        },
        "aggregate": aggregate,
        "clips": clips,
        "non_claims": [],
    }


def _canonical_registry_for_motion_files(motion_files):
    files = [os.path.abspath(os.fspath(path)) for path in motion_files]
    if len(files) == 1:
        files = files * 5
    elif len(files) == 2:
        files = [files[0], files[1], files[0], files[1], files[0]]
    elif len(files) != 5:
        raise AssertionError("canonical test banks need one, two, or five source paths")
    repo_root = os.path.commonpath([os.path.dirname(path) for path in files])
    contract_dir = os.path.join(repo_root, "canonical_registry_contract")
    os.makedirs(contract_dir, exist_ok=True)

    with np.load(files[0], allow_pickle=False) as data:
        ready_joint = np.asarray(data["joint_pos"][0], dtype=np.float64)
        ready_root_pos = np.asarray(data["body_pos_w"][0, 0], dtype=np.float64)
        ready_root_quat = np.asarray(data["body_quat_w"][0, 0], dtype=np.float64)
        ready_body_pos = np.asarray(data["body_pos_w"][0], dtype=np.float32)
        ready_body_quat = np.asarray(data["body_quat_w"][0], dtype=np.float32)
    ready_path = os.path.join(contract_dir, "canonical_ready_v1.npz")
    np.savez(
        ready_path,
        joint_pos=ready_joint,
        joint_vel=np.zeros(31, dtype=np.float64),
        root_pos_w=ready_root_pos,
        root_quat_w=ready_root_quat,
        source_segment=np.array("bh_loop_c"),
        source_npz=np.array(os.path.basename(files[0])),
        source_frame=np.array(0, dtype=np.int64),
        striking_joint_ids=np.arange(7, dtype=np.int64),
        note=np.array("canonical consumer unit-test ready"),
    )
    ready_sha = _file_sha256(ready_path)
    ready_fk_path = os.path.join(contract_dir, "canonical_ready_fk_v1.npz")
    np.savez(
        ready_fk_path,
        canonical_ready_sha256=np.array(ready_sha),
        body_names=np.asarray(_BODY_NAMES),
        body_pos_w=ready_body_pos,
        body_quat_w=ready_body_quat,
        kinematics_contract_version=np.array([1], dtype=np.int64),
    )
    ready_fk_sha = _file_sha256(ready_fk_path)

    entries = []
    strike_phases = []
    for index, (motion_id, motion_path) in enumerate(
        zip(_CANONICAL_MOTION_IDS, files)
    ):
        with np.load(motion_path, allow_pickle=False) as data:
            frames = int(data["joint_pos"].shape[0])
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        marker = frames // 2
        opportunity = [max(0, marker - 1), min(frames - 1, marker + 1)]
        strike_phases.append(float(marker) / float(frames - 1))
        npz_sha = _file_sha256(motion_path)
        source_path = os.path.join(contract_dir, f"{motion_id}.source.json")
        source_sha = _write_registry_json(
            source_path, {"motion_id": motion_id, "source": "unit-test"}
        )
        build_path = os.path.join(contract_dir, f"{motion_id}.build.json")
        build_sha = _write_registry_json(
            build_path,
            {
                "hashes": {
                    "output_npz_sha256": npz_sha,
                    "ready_sha256": ready_sha,
                },
                "publication_class": "training_adopted",
                "training_authorized": True,
            },
        )
        applicability_path = os.path.join(
            contract_dir, f"{motion_id}.applicability.json"
        )
        applicability_sha = _write_registry_json(
            applicability_path,
            {
                "schema_version": 1,
                "motion_id": motion_id,
                "scope": "upper",
                "variant": "unit_test",
                "npz_sha256": npz_sha,
                "domain": "canonical-consumer-unit-test",
            },
        )
        evidence_certificates = []
        for evidence_level in ("E1", "E2"):
            certificate_path = os.path.join(
                contract_dir,
                f"{motion_id}.{evidence_level.lower()}.certificate.json",
            )
            certificate_sha = _write_registry_json(
                certificate_path,
                {
                    "schema_version": 1,
                    "level": evidence_level,
                    "motion_id": motion_id,
                    "scope": "upper",
                    "variant": "unit_test",
                    "npz_sha256": npz_sha,
                    "status": "pass",
                },
            )
            evidence_certificates.append(
                {
                    "level": evidence_level,
                    "path": os.path.relpath(certificate_path, repo_root),
                    "sha256": certificate_sha,
                    "status": "pass",
                }
            )
        evidence_path = os.path.join(contract_dir, f"{motion_id}.evidence.json")
        evidence_sha = _write_registry_json(
            evidence_path,
            {
                "schema_version": 1,
                "motion_id": motion_id,
                "scope": "upper",
                "variant": "unit_test",
                "npz_sha256": npz_sha,
                "highest_evidence_level": "E2",
                "certificates": evidence_certificates,
            },
        )
        question_bank_path = os.path.join(
            contract_dir, f"{motion_id}.question_bank.npz"
        )
        question_meta = {
            "schema_version": 3,
            "split": "train",
            "clip_order": [motion_id],
            "clips": {
                motion_id: {
                    "motion_sha256": npz_sha,
                    "n_frames": frames,
                    "anchor_frame": marker,
                }
            },
        }
        question_vector = np.zeros((1, 3), dtype=np.float32)
        np.savez(
            question_bank_path,
            meta_json=np.frombuffer(
                json.dumps(
                    question_meta,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"),
                dtype=np.uint8,
            ),
            **{
                f"{motion_id}/contact_pos_env": np.zeros(3, dtype=np.float32),
                f"{motion_id}/incoming_vel": question_vector,
                f"{motion_id}/incoming_spin": question_vector,
                f"{motion_id}/demanded_vel": question_vector,
                f"{motion_id}/demanded_normal": question_vector,
            },
        )
        training_config_path = os.path.join(
            contract_dir, f"{motion_id}.training_config.json"
        )
        _write_registry_json(
            training_config_path,
            {
                "schema_version": 1,
                "contract": "canonical-motion-training-config-v1",
                "motion_id": motion_id,
                "scope": "upper",
                "variant": "unit_test",
                "npz_sha256": npz_sha,
            },
        )
        question_bank_sha = _file_sha256(question_bank_path)
        training_config_sha = _file_sha256(training_config_path)
        adoption_path = os.path.join(contract_dir, f"{motion_id}.adoption.json")
        adoption_sha = _write_registry_json(
            adoption_path,
            {
                "schema_version": 1,
                "motion_id": motion_id,
                "scope": "upper",
                "variant": "unit_test",
                "npz_sha256": npz_sha,
                "strike_marker_frame": marker,
                "contact_opportunity_frames": opportunity,
                "mount_normal_sign": _CANONICAL_FACE_SIGNS[index],
                "canonical_ready_sha256": ready_sha,
                "canonical_ready_fk_sha256": ready_fk_sha,
                "artifacts": {
                    "question_bank": {
                        "sha256": question_bank_sha,
                        "schema_version": 3,
                    },
                    "training_config": {
                        "sha256": training_config_sha,
                        "schema_version": 1,
                    },
                    "onnx_model": None,
                    "onnx_metadata": None,
                },
            },
        )
        entries.append(
            {
                "motion_id": motion_id,
                "scope": "upper",
                "variant": "unit_test",
                "npz_path": os.path.relpath(motion_path, repo_root),
                "npz_sha256": npz_sha,
                "frames": frames,
                "fps": fps,
                "family": _CANONICAL_FAMILIES[index],
                "strike_marker_frame": marker,
                "contact_opportunity_frames": opportunity,
                "mount_normal_sign": _CANONICAL_FACE_SIGNS[index],
                "canonical_ready_sha256": ready_sha,
                "source_manifest_path": os.path.relpath(source_path, repo_root),
                "source_manifest_sha256": source_sha,
                "build_manifest_path": os.path.relpath(build_path, repo_root),
                "build_manifest_sha256": build_sha,
                "applicability_manifest_path": os.path.relpath(
                    applicability_path, repo_root
                ),
                "applicability_manifest_sha256": applicability_sha,
                "evidence_level": "E2",
                "evidence_manifest_path": os.path.relpath(
                    evidence_path, repo_root
                ),
                "evidence_manifest_sha256": evidence_sha,
                "question_bank_path": os.path.relpath(
                    question_bank_path, repo_root
                ),
                "question_bank_sha256": question_bank_sha,
                "question_bank_schema_version": 3,
                "training_config_path": os.path.relpath(
                    training_config_path, repo_root
                ),
                "training_config_sha256": training_config_sha,
                "training_config_schema_version": 1,
                "onnx_model_path": None,
                "onnx_model_sha256": None,
                "onnx_model_schema_version": None,
                "onnx_metadata_path": None,
                "onnx_metadata_sha256": None,
                "onnx_metadata_schema_version": None,
                "adoption_manifest_path": os.path.relpath(
                    adoption_path, repo_root
                ),
                "adoption_manifest_sha256": adoption_sha,
                "publication_class": "training_adopted",
                "training_authorized": True,
                "deployment_authorized": False,
                "hardware_authorized": False,
            }
        )
    registry_path = os.path.join(contract_dir, "upper_bank.json")
    registry_sha = _write_registry_json(
        registry_path,
        {
            "schema_version": 1,
            "bank_id": "canonical_upper_unit_test",
            "scope": "upper",
            "canonical_ready_path": os.path.relpath(ready_path, repo_root),
            "canonical_ready_sha256": ready_sha,
            "canonical_ready_fk_path": os.path.relpath(
                ready_fk_path, repo_root
            ),
            "canonical_ready_fk_sha256": ready_fk_sha,
            "entries": entries,
        },
    )
    registry_module = commands_mod.MotionCommand._canonical_registry_module()
    loaded = registry_module.load_canonical_motion_bank_registry(
        registry_path,
        repo_root=repo_root,
        expected_registry_sha256=registry_sha,
    )
    tables = registry_module.adapt_registry_for_runtime(
        loaded, authorization_purpose=None
    )
    admission_binding = registry_module.bank_promotion_binding(
        loaded, authorization_purpose="training"
    )
    bank_gate_path = os.path.join(contract_dir, "upper_bank_gate.json")
    bank_gate_sha = _write_registry_json(
        bank_gate_path,
        _complete_runtime_bank_gate_report(
            admission_binding,
            Path(repo_root),
            Path(ready_path),
        ),
    )
    promotion_path = os.path.join(contract_dir, "upper_bank_promotion.json")
    promotion_sha = _write_registry_json(
        promotion_path,
        {
            "schema_version": 1,
            "certificate_type": "canonical-motion-bank-promotion-v1",
            **registry_module.motion_admission._binding_document(
                admission_binding
            ),
            "bank_gate_report": {
                "path": os.path.relpath(bank_gate_path, repo_root),
                "sha256": bank_gate_sha,
            },
        },
    )
    return {
        "motion_files": files,
        "repo_root": repo_root,
        "registry_path": registry_path,
        "registry_sha256": registry_sha,
        "alignment_sha256": tables.alignment_sha256,
        "ready_sha256": ready_sha,
        "ready_fk_sha256": ready_fk_sha,
        "promotion_certificate_path": promotion_path,
        "promotion_certificate_sha256": promotion_sha,
        "families": _CANONICAL_FAMILIES,
        "strike_phases": tuple(strike_phases),
        "face_signs": _CANONICAL_FACE_SIGNS,
    }


class _Scene:
    def __init__(self, robot, num_envs):
        self._robot = robot
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, key):
        assert key == "robot"
        return self._robot


class _CmdRobot:
    def __init__(self, num_envs):
        self.body_names = list(_BODY_NAMES)
        self.joint_names = list(_A3_JOINTS)
        default_root = torch.zeros(num_envs, 13)
        default_root[:, 2] = _STAND_Z
        default_root[:, 3] = 1.0
        body_pos = torch.zeros(num_envs, len(_BODY_NAMES), 3)
        body_quat = torch.zeros(num_envs, len(_BODY_NAMES), 4)
        body_quat[..., 0] = 1.0
        self.data = types.SimpleNamespace(
            joint_names=list(_A3_JOINTS),
            joint_pos=torch.zeros(num_envs, _N_JOINTS),
            joint_vel=torch.zeros(num_envs, _N_JOINTS),
            joint_vel_limits=torch.full((_N_JOINTS,), 5.0),
            root_state_w=default_root.clone(),
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=torch.zeros_like(body_pos),
            body_ang_vel_w=torch.zeros_like(body_pos),
            default_joint_pos=torch.zeros(num_envs, _N_JOINTS),
            default_joint_vel=torch.zeros(num_envs, _N_JOINTS),
            default_root_state=default_root,
            soft_joint_pos_limits=torch.stack(
                [-torch.ones(num_envs, _N_JOINTS) * 3.14, torch.ones(num_envs, _N_JOINTS) * 3.14], dim=-1),
        )
        self.calls = []

    def find_bodies(self, names, preserve_order=True):
        return [self.body_names.index(n) for n in names], list(names)

    def write_root_state_to_sim(self, root, env_ids=None):
        self.calls.append(("root", root.clone(), None if env_ids is None else env_ids.clone()))
        self.data.root_state_w[env_ids] = root

    def write_joint_state_to_sim(self, jp, jv, env_ids=None):
        self.calls.append(("joint", jp.clone(), jv.clone()))
        self.data.joint_pos[env_ids] = jp
        self.data.joint_vel[env_ids] = jv


def _make_motion_command(motion_files, num_envs=8, **cfg_overrides):
    # 运行时 motion_file 只会是 str 或 str 列表(Hydra CLI);裸 PosixPath 是测试自造形态,
    # 统一转 str 以继续走真实的标量/列表两条加载路径。
    motion_files = [str(item) for item in motion_files]
    trusted_promotion_sha = cfg_overrides.pop(
        "_test_trusted_promotion_sha", None
    )
    racket_strike_phases = cfg_overrides.pop(
        "_racket_strike_phase_per_clip", ()
    )
    racket_face_signs = cfg_overrides.pop(
        "_racket_mount_normal_sign_per_clip", ()
    )
    robot = _CmdRobot(num_envs)
    env = types.SimpleNamespace(
        num_envs=num_envs, device="cpu", step_dt=0.02,
        scene=_Scene(robot, num_envs),
        cfg=types.SimpleNamespace(
            decimation=4,
            sim=types.SimpleNamespace(dt=0.005),
            commands=types.SimpleNamespace(
                racket_target=types.SimpleNamespace(
                    strike_phase_per_clip=racket_strike_phases,
                    mount_normal_sign_per_clip=racket_face_signs,
                )
            ),
        ),
        termination_manager=types.SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
    )
    cfg_kwargs = dict(
        asset_name="robot",
        motion_file=motion_files if len(motion_files) > 1 else motion_files[0],
        anchor_body_name="torso_Link",
        body_names=list(_BODY_NAMES),
        pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0),
        stand_start_prob=0.0, post_swing_start_prob=0.0,
        hold_steps_range=(5, 5),  # every birth held -> deterministic held-RSI path
    )
    cfg_kwargs.update(cfg_overrides)
    cfg = commands_mod.MotionCommandCfg(**cfg_kwargs)
    # Default (non-canonical) motion_file path takes no trust-set entry: it
    # loads raw NPZ bytes directly. Only canonical_ready_mode needs the code
    # trust set, so inject a promotion digest solely when one is supplied.
    registry_module = commands_mod.MotionCommand._canonical_registry_module()
    admission_module = registry_module.motion_admission
    prior_promotion_trust = (
        admission_module.TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256
    )
    if trusted_promotion_sha is not None:
        admission_module.TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256 = (
            prior_promotion_trust | frozenset({trusted_promotion_sha})
        )
    try:
        return commands_mod.MotionCommand(cfg, env), robot
    finally:
        admission_module.TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256 = (
            prior_promotion_trust
        )


def _make_canonical_ready_command(motion_files, num_envs=8, **cfg_overrides):
    binding = _canonical_registry_for_motion_files(motion_files)
    canonical = dict(
        canonical_ready_mode=True,
        canonical_registry_path=binding["registry_path"],
        canonical_registry_repo_root=binding["repo_root"],
        canonical_registry_sha256=binding["registry_sha256"],
        canonical_registry_alignment_sha256=binding["alignment_sha256"],
        canonical_ready_sha256=binding["ready_sha256"],
        canonical_ready_fk_sha256=binding["ready_fk_sha256"],
        canonical_promotion_certificate_path=(
            binding["promotion_certificate_path"]
        ),
        stand_start_prob=1.0,
        post_swing_start_prob=0.0,
        wrap_teleport=False,
        rsi_skip_settle_frames=0,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        stand_start_yaw_range=(0.0, 0.0),
        hold_steps_range=(2, 2),
        stand_start_min_hold=2,
        clip_family_per_clip=binding["families"],
        _racket_strike_phase_per_clip=binding["strike_phases"],
        _racket_mount_normal_sign_per_clip=binding["face_signs"],
        _test_trusted_promotion_sha=(
            binding["promotion_certificate_sha256"]
        ),
    )
    canonical.update(cfg_overrides)
    return _make_motion_command(
        binding["motion_files"], num_envs=num_envs, **canonical
    )


@pytest.fixture(scope="module")
def clips():
    tmp = tempfile.mkdtemp(prefix="reward_flags_clips_")
    return (
        _write_motion_npz(os.path.join(tmp, "forehand.npz"), frames=20),
        _write_motion_npz(os.path.join(tmp, "backhand.npz"), frames=15),
        _write_motion_npz(os.path.join(tmp, "single.npz"), frames=30),
    )


def test_motion_loader_rejects_wrong_body_order_and_fps_contracts(tmp_path):
    wrong_order = _write_motion_npz(
        tmp_path / "wrong_order.npz",
        frames=12,
        metadata_body_names=list(reversed(_BODY_NAMES)),
    )
    with pytest.raises(ValueError, match="body_names/order"):
        _make_motion_command([wrong_order])

    clip_50 = _write_motion_npz(tmp_path / "clip_50.npz", frames=12, fps=50)
    clip_49 = _write_motion_npz(tmp_path / "clip_49.npz", frames=12, fps=49)
    with pytest.raises(ValueError, match="unequal fps"):
        _make_motion_command([clip_50, clip_49])

    clip_60 = _write_motion_npz(tmp_path / "clip_60.npz", frames=12, fps=60)
    with pytest.raises(ValueError, match="motion fps must equal the policy rate"):
        _make_motion_command([clip_60])


def test_motion_loader_rejects_non_scalar_or_nonfinite_fps(tmp_path):
    nonscalar = _write_motion_npz(
        tmp_path / "nonscalar_fps.npz", frames=12, fps=np.array([50.0, 50.0])
    )
    with pytest.raises(ValueError, match="fps must be scalar"):
        _make_motion_command([nonscalar])

    nonfinite = _write_motion_npz(tmp_path / "nonfinite_fps.npz", frames=12, fps=np.nan)
    with pytest.raises(ValueError, match="finite and positive"):
        _make_motion_command([nonfinite])


def test_default_motion_file_path_needs_no_trust_set_entry(tmp_path):
    # Regression: the default (non-canonical) motion_file channel loads a plain
    # NPZ with no code-owned trust set. Admission is scoped to canonical mode,
    # so a raw clip that is in neither trust set must still construct.
    clip = _write_motion_npz(tmp_path / "untrusted_raw.npz", frames=12)
    registry_module = commands_mod.MotionCommand._canonical_registry_module()
    admission_module = registry_module.motion_admission
    assert not hasattr(admission_module, "TRUSTED_LEGACY_RAW_MOTION_SHA256")
    assert admission_module.TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256 == frozenset()

    cmd, _ = _make_motion_command([clip])

    assert cmd.canonical_ready_mode is False
    assert cmd.motion.time_step_total == 12


def test_runtime_registry_loader_ignores_preloaded_file_spoof(
    monkeypatch,
):
    script = (
        Path(commands_mod.__file__).resolve().parents[6]
        / "scripts"
        / "canonical_motion_registry.py"
    )
    fake = types.ModuleType("_hope_canonical_motion_registry_runtime")
    fake.__file__ = str(script)
    monkeypatch.setattr(
        commands_mod, "_CANONICAL_REGISTRY_RUNTIME_MODULE", None
    )
    monkeypatch.setitem(
        sys.modules, "_hope_canonical_motion_registry_runtime", fake
    )

    loaded = commands_mod.MotionCommand._canonical_registry_module()

    assert loaded is not fake
    assert Path(loaded.__file__).resolve() == script


def test_canonical_ready_mode_is_default_off_and_does_not_validate_legacy_endpoints(
    tmp_path,
):
    mismatched = _write_canonical_ready_motion_npz(
        tmp_path / "legacy_endpoint_mismatch.npz", end_joint_offset=0.02
    )
    cmd, robot = _make_motion_command([mismatched])

    assert cmd.canonical_ready_mode is False
    cmd.hold_counter[:] = 1
    # Historical hold behavior remains default_joint_pos + clip-owned body pose.  Merely adding
    # the new flag must not alter a legacy training command or make old endpoint shapes unloadable.
    assert torch.equal(cmd.joint_pos, robot.data.default_joint_pos)
    assert torch.equal(cmd.body_pos_w[:, 0, 2], torch.full((cmd.num_envs,), 0.93))


def test_canonical_ready_mode_has_no_raw_motion_escape_hatch(tmp_path):
    clip = _write_canonical_ready_motion_npz(tmp_path / "unregistered.npz")
    with pytest.raises(ValueError, match="requires canonical_registry_path"):
        _make_motion_command(
            [clip],
            canonical_ready_mode=True,
            stand_start_prob=1.0,
            wrap_teleport=False,
            stand_start_yaw_range=(0.0, 0.0),
        )


def test_canonical_ready_mode_rejects_self_report_without_code_trust(
    tmp_path, monkeypatch
):
    clips = [
        _write_canonical_ready_motion_npz(
            tmp_path / f"candidate_{index}.npz",
            interior_scale=1.0 + 0.1 * index,
        )
        for index in range(5)
    ]
    binding = _canonical_registry_for_motion_files(clips)
    registry_module = commands_mod.MotionCommand._canonical_registry_module()
    monkeypatch.setattr(
        registry_module.motion_admission,
        "TRUSTED_BANK_PROMOTION_CERTIFICATE_SHA256",
        frozenset(),
    )

    with pytest.raises(
        ValueError, match="promotion certificate SHA-256 is absent"
    ):
        _make_motion_command(
            binding["motion_files"],
            canonical_ready_mode=True,
            canonical_registry_path=binding["registry_path"],
            canonical_registry_repo_root=binding["repo_root"],
            canonical_registry_sha256=binding["registry_sha256"],
            canonical_registry_alignment_sha256=binding["alignment_sha256"],
            canonical_ready_sha256=binding["ready_sha256"],
            canonical_ready_fk_sha256=binding["ready_fk_sha256"],
            canonical_promotion_certificate_path=(
                binding["promotion_certificate_path"]
            ),
            stand_start_prob=1.0,
            post_swing_start_prob=0.0,
            wrap_teleport=False,
            rsi_skip_settle_frames=0,
            pose_range={},
            velocity_range={},
            joint_position_range=(0.0, 0.0),
            stand_start_yaw_range=(0.0, 0.0),
            hold_steps_range=(2, 2),
            stand_start_min_hold=2,
            clip_family_per_clip=binding["families"],
            _racket_strike_phase_per_clip=binding["strike_phases"],
            _racket_mount_normal_sign_per_clip=binding["face_signs"],
        )


@pytest.mark.parametrize(
    ("override_factory", "message"),
    (
        (
            lambda binding: {"canonical_registry_sha256": "0" * 64},
            "registry SHA-256 mismatch",
        ),
        (
            lambda binding: {"canonical_registry_alignment_sha256": "0" * 64},
            "alignment SHA-256 mismatch",
        ),
        (
            lambda binding: {"canonical_ready_sha256": "0" * 64},
            "canonical ready SHA-256 mismatch",
        ),
        (
            lambda binding: {"canonical_ready_fk_sha256": "0" * 64},
            "canonical ready FK SHA-256 mismatch",
        ),
        (
            lambda binding: {"clip_family_per_clip": ("backhand",) * 5},
            "clip_family_per_clip must exactly equal",
        ),
        (
            lambda binding: {"_racket_strike_phase_per_clip": (0.1,) * 5},
            "strike_phase_per_clip differs",
        ),
        (
            lambda binding: {"_racket_mount_normal_sign_per_clip": (1.0,) * 5},
            "mount_normal_sign_per_clip differs",
        ),
        (
            lambda binding: {
                "motion_file": (
                    binding["motion_files"][1],
                    binding["motion_files"][0],
                    *binding["motion_files"][2:],
                )
            },
            "motion_file order differs",
        ),
    ),
)
def test_canonical_ready_mode_rejects_any_atomic_registry_table_drift(
    tmp_path, override_factory, message
):
    clips = [
        _write_canonical_ready_motion_npz(
            tmp_path / f"ready_{index}.npz", interior_scale=1.0 + 0.1 * index
        )
        for index in range(5)
    ]
    binding = _canonical_registry_for_motion_files(clips)
    canonical = dict(
        canonical_ready_mode=True,
        canonical_registry_path=binding["registry_path"],
        canonical_registry_repo_root=binding["repo_root"],
        canonical_registry_sha256=binding["registry_sha256"],
        canonical_registry_alignment_sha256=binding["alignment_sha256"],
        canonical_ready_sha256=binding["ready_sha256"],
        canonical_ready_fk_sha256=binding["ready_fk_sha256"],
        canonical_promotion_certificate_path=(
            binding["promotion_certificate_path"]
        ),
        stand_start_prob=1.0,
        wrap_teleport=False,
        stand_start_yaw_range=(0.0, 0.0),
        clip_family_per_clip=binding["families"],
        _racket_strike_phase_per_clip=binding["strike_phases"],
        _racket_mount_normal_sign_per_clip=binding["face_signs"],
        _test_trusted_promotion_sha=(
            binding["promotion_certificate_sha256"]
        ),
    )
    canonical.update(override_factory(binding))
    with pytest.raises(ValueError, match=message):
        _make_motion_command(binding["motion_files"], **canonical)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"end_joint_offset": 2.0e-5}, "first/last joint pose"),
        ({"start_body_offset": 2.0e-5}, "first/last body pose"),
        ({"endpoint_joint_velocity": 1.0e-5}, "first/last frames must be exactly zero"),
    ),
)
def test_canonical_ready_mode_rejects_any_mixed_or_moving_boundary(
    tmp_path, kwargs, message
):
    good = _write_canonical_ready_motion_npz(tmp_path / "good.npz")
    bad = _write_canonical_ready_motion_npz(tmp_path / "bad.npz", **kwargs)
    with pytest.raises(ValueError, match=message):
        _make_canonical_ready_command([good, bad])


@pytest.mark.parametrize(
    "overrides",
    (
        {"stand_start_prob": 0.5},
        {"post_swing_start_prob": 0.25},
        {"joint_position_range": (-0.1, 0.1)},
        {"pose_range": {"x": (-0.01, 0.01)}},
        {"wrap_teleport": True},
        {"rsi_skip_settle_frames": 1},
        {"clip_switch_prob": 0.1},
        {"event_timing_mode": "post_strike_t1"},
    ),
)
def test_canonical_ready_mode_fails_closed_on_alternate_reset_curricula(
    tmp_path, overrides
):
    clip = _write_canonical_ready_motion_npz(tmp_path / "ready.npz")
    with pytest.raises(
        ValueError, match="incompatible with RSI/post-swing/noised reset curricula"
    ):
        _make_canonical_ready_command([clip], **overrides)


def test_canonical_ready_hold_uses_one_clip_frame_for_every_reference_channel(
    tmp_path,
):
    clip_a = _write_canonical_ready_motion_npz(
        tmp_path / "a.npz", interior_scale=1.0
    )
    clip_b = _write_canonical_ready_motion_npz(
        tmp_path / "b.npz", interior_scale=1.7
    )
    cmd, robot = _make_canonical_ready_command(
        [clip_a, clip_b], num_envs=2
    )
    cmd.clip_id[:] = torch.tensor([0, 1])
    starts = cmd.motion.seg_start[cmd.clip_id]
    # Attack the old mixed-reference seam: leave each held environment's clock on a moving
    # interior frame.  The enabled contract must still source every pose from that clip's ready.
    cmd.time_steps[:] = starts + 1
    cmd.time_steps_f[:] = cmd.time_steps.float()
    cmd.hold_counter[:] = 1
    cmd.metrics["in_hold"].zero_()

    assert torch.equal(cmd.joint_pos, cmd.motion.joint_pos[starts])
    assert not torch.equal(cmd.joint_pos, robot.data.default_joint_pos)
    assert torch.equal(
        cmd.body_pos_w - cmd._env.scene.env_origins[:, None, :],
        cmd.motion.body_pos_w[starts],
    )
    assert torch.equal(cmd.body_quat_w, cmd.motion.body_quat_w[starts])
    assert torch.equal(
        cmd.anchor_pos_w - cmd._env.scene.env_origins,
        cmd.motion.body_pos_w[starts, cmd.motion_anchor_body_index],
    )
    assert torch.equal(
        cmd.anchor_quat_w,
        cmd.motion.body_quat_w[starts, cmd.motion_anchor_body_index],
    )
    for velocity in (
        cmd.joint_vel,
        cmd.body_lin_vel_w,
        cmd.body_ang_vel_w,
        cmd.anchor_lin_vel_w,
        cmd.anchor_ang_vel_w,
    ):
        assert torch.count_nonzero(velocity).item() == 0


def test_canonical_ready_true_reset_writes_root_and_31_joints_from_same_frame(
    tmp_path,
):
    clip_a = _write_canonical_ready_motion_npz(tmp_path / "a.npz")
    clip_b = _write_canonical_ready_motion_npz(
        tmp_path / "b.npz", interior_scale=1.4
    )
    cmd, robot = _make_canonical_ready_command(
        [clip_a, clip_b], num_envs=3
    )
    cmd._env.scene.env_origins[:] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, -2.0, 0.0]]
    )
    env_ids = torch.arange(3)
    cmd._resample_command(env_ids)

    assert [call[0] for call in robot.calls] == ["root", "joint"]
    root_state = robot.calls[0][1]
    joint_pos, joint_vel = robot.calls[1][1:3]
    ready_steps = cmd.motion.seg_start[cmd.clip_id]
    assert torch.equal(cmd.time_steps, ready_steps)
    assert torch.equal(
        root_state[:, :3],
        cmd.motion.body_pos_w[ready_steps, 0] + cmd._env.scene.env_origins,
    )
    assert torch.equal(root_state[:, 3:7], cmd.motion.body_quat_w[ready_steps, 0])
    assert torch.count_nonzero(root_state[:, 7:]).item() == 0
    assert joint_pos.shape == (3, _N_JOINTS)
    assert torch.equal(joint_pos, cmd.motion.joint_pos[ready_steps])
    assert torch.count_nonzero(joint_vel).item() == 0
    assert torch.equal(robot.data.root_state_w, root_state)
    assert torch.equal(robot.data.joint_pos, joint_pos)
    assert torch.count_nonzero(robot.data.joint_vel).item() == 0


def test_canonical_ready_release_has_one_final_hold_step_then_advances_frame_one(
    tmp_path,
):
    clip = _write_canonical_ready_motion_npz(tmp_path / "ready.npz")
    cmd, _ = _make_canonical_ready_command(
        [clip], num_envs=2, hold_steps_range=(1, 1), stand_start_min_hold=1
    )
    cmd._resample_command(torch.arange(2))
    ready_step = cmd.motion.seg_start[cmd.clip_id]

    cmd._update_command()
    assert torch.equal(cmd.time_steps, ready_step)
    assert torch.all(cmd.in_hold)
    assert torch.equal(cmd.joint_pos, cmd.motion.joint_pos[ready_step])
    assert torch.count_nonzero(cmd.joint_vel).item() == 0

    cmd._update_command()
    assert torch.equal(cmd.time_steps, ready_step + 1)
    assert not torch.any(cmd.in_hold)
    assert torch.equal(cmd.joint_pos, cmd.motion.joint_pos[ready_step + 1])
    assert torch.count_nonzero(cmd.joint_vel).item() > 0


def test_canonical_ready_runtime_clip_retarget_is_ready_boundary_only(tmp_path):
    clip = _write_canonical_ready_motion_npz(tmp_path / "ready.npz")
    cmd, _ = _make_canonical_ready_command([clip], num_envs=2)
    start = int(cmd.motion.seg_start[0].item())
    end = start + int(cmd.motion.seg_len[0].item()) - 1

    cmd.time_steps[0] = start + 1
    with pytest.raises(ValueError, match="cannot change canonical clip mid-stroke"):
        cmd.install_external_exam_timing(
            torch.tensor([0]), torch.tensor([1]), torch.tensor([2])
        )

    cmd.time_steps[0] = end
    cmd.install_external_exam_timing(
        torch.tensor([0]), torch.tensor([1]), torch.tensor([2])
    )
    assert int(cmd.clip_id[0].item()) == 1
    assert int(cmd.time_steps[0].item()) == int(cmd.motion.seg_start[1].item())
    assert int(cmd.hold_counter[0].item()) == 2


def test_rc_skip_settle_frames_multiseg(clips):
    cmd, _ = _make_motion_command([clips[0], clips[1]], rsi_skip_settle_frames=6)
    ids = torch.arange(cmd.num_envs)
    cmd._adaptive_sampling(ids)
    seg_start = cmd.motion.seg_start[cmd.clip_id]
    assert torch.all(cmd.time_steps == seg_start + 6)  # every swing entry starts at frame N
    # default 0 = byte-identical entry at seg_start
    cmd0, _ = _make_motion_command([clips[0], clips[1]])
    cmd0._adaptive_sampling(ids)
    assert torch.all(cmd0.time_steps == cmd0.motion.seg_start[cmd0.clip_id])


def test_rc_skip_settle_frames_clamps_short_clips(clips):
    cmd, _ = _make_motion_command([clips[0], clips[1]], rsi_skip_settle_frames=50)
    ids = torch.arange(cmd.num_envs)
    cmd._adaptive_sampling(ids)
    seg_start = cmd.motion.seg_start[cmd.clip_id]
    seg_last = seg_start + cmd.motion.seg_len[cmd.clip_id] - 1
    assert torch.all(cmd.time_steps == seg_last)  # clamped inside the segment, never out of range


def test_rc_skip_settle_frames_single_clip(clips):
    torch.manual_seed(0)
    cmd, _ = _make_motion_command([clips[2]], rsi_skip_settle_frames=6)
    ids = torch.arange(cmd.num_envs)
    cmd._adaptive_sampling(ids)
    assert torch.all(cmd.time_steps >= 6)
    assert torch.all(cmd.time_steps <= cmd.motion.time_step_total - 1)


def test_rc_hold_root_stand_z_rewrites_held_rsi_births(clips):
    torch.manual_seed(0)
    cmd, robot = _make_motion_command([clips[2]], rsi_hold_root_stand_z=True)
    cmd._resample_command(torch.arange(cmd.num_envs))
    roots = [c for c in robot.calls if c[0] == "root"]
    assert len(roots) == 1
    root = roots[0][1]
    assert torch.all(cmd.hold_counter[roots[0][2]] > 0)  # the births under test are HELD
    assert torch.allclose(root[:, 2], torch.full((root.shape[0],), _STAND_Z))  # stand height
    assert torch.allclose(root[:, 7:], torch.zeros_like(root[:, 7:]))  # hold-zeroed velocities


def test_rc_hold_root_stand_z_off_keeps_reference_z(clips):
    torch.manual_seed(0)
    cmd, robot = _make_motion_command([clips[2]])  # flag off (default)
    cmd._resample_command(torch.arange(cmd.num_envs))
    root = [c for c in robot.calls if c[0] == "root"][0][1]
    assert torch.allclose(root[:, 2], torch.full((root.shape[0],), _CROUCH_Z))  # crouch frame-0 z


def test_rc_hold_root_stand_z_leaves_unheld_births_alone(clips):
    torch.manual_seed(0)
    cmd, robot = _make_motion_command(
        [clips[2]], rsi_hold_root_stand_z=True, hold_steps_range=(0, 0))  # hold=0: never held
    cmd._resample_command(torch.arange(cmd.num_envs))
    root = [c for c in robot.calls if c[0] == "root"][0][1]
    assert torch.allclose(root[:, 2], torch.full((root.shape[0],), _CROUCH_Z))  # untouched


def _prime_post_swing_buffer(cmd):
    count = int(cmd.cfg.post_swing_min_fill)
    cmd._post_swing_count = count
    cmd._post_swing_ptr = 0
    cmd._post_swing_root = torch.zeros(count, 13)
    cmd._post_swing_root[:, 3] = 1.0
    cmd._post_swing_joint_pos = torch.zeros(count, _N_JOINTS)
    cmd._post_swing_joint_vel = torch.zeros(count, _N_JOINTS)


def _counter_values(snapshot):
    return {name: int(value.item()) for name, value in snapshot.items()}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_post_swing_teacher_receipt(tmp_path, motion_files, *, count=4):
    payload = tmp_path / "post_swing_teacher_states.npz"
    root = np.zeros((count, 13), dtype=np.float32)
    root[:, 2] = 1.0
    root[:, 3] = 1.0
    np.savez(
        payload,
        root_state_origin_relative=root,
        joint_pos=np.zeros((count, _N_JOINTS), dtype=np.float32),
        joint_vel=np.zeros((count, _N_JOINTS), dtype=np.float32),
    )
    receipt = {
        "schema_version": 2,
        "artifact_kind": "hope_post_swing_teacher_state_receipt",
        "capture_contract": {
            "event": "natural_clip_wrap",
            "wrap_teleport": False,
            "clip_switch_aborted_states_included": False,
            "root_position_frame": "environment_origin_relative",
            "root_state_layout": "pos3_quat_wxyz4_linear_velocity_com3_angular_velocity3",
            "joint_state_order": "runtime_articulation_joint_names",
        },
        "teacher": {
            "source_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "training_contract_sha256": "3" * 64,
            "training_contract_schema_version": 3,
            "fresh_lineage": True,
        },
        "motion_clips": [
            {"index": index, "sha256": _sha256(path)}
            for index, path in enumerate(motion_files)
        ],
        "states": {
            "relative_path": payload.name,
            "sha256": _sha256(payload),
            "count": count,
            "root_shape": [count, 13],
            "joint_pos_shape": [count, _N_JOINTS],
            "joint_vel_shape": [count, _N_JOINTS],
            "joint_names": list(_A3_JOINTS),
            "velocity_limits": {
                "root_linear_norm_max_mps": 2.0,
                "root_angular_norm_max_radps": 4.0,
                "joint_abs_max_radps": [5.0] * _N_JOINTS,
            },
        },
        "attestation": {
            "schema_version": 2,
            "artifact_kind": "hope_post_swing_teacher_capture_attestation",
            "capture_result_sha256": "4" * 64,
            "capture_result_relative_path": "natural_wrap_capture.json",
            "capture_claim_sha256": "a" * 64,
            "capture_claim_relative_path": "natural_wrap_capture.claim.json",
            "checkpoint": {
                "sha256": "2" * 64,
                "training_contract_schema_version": 3,
                "training_contract_sha256": "3" * 64,
                "training_contract_lineage_exact": True,
                "training_launch_claim_sha256": "5" * 64,
            },
            "hard_contract": {"sha256": "3" * 64, "schema_version": 3},
            "checkpoint_source": {
                "commit": "1" * 40,
                "launch_claim_content_sha256": "5" * 64,
            },
            "capture_source": {
                "commit": "6" * 40,
                "clean": True,
                "producer_source_sha256": "8" * 64,
            },
            "attestor_source": {
                "commit": "7" * 40,
                "clean": True,
                "attestor_source_sha256": "9" * 64,
            },
            "retry_authorization": {
                "authorization_id": "test-v3-attestor-attempt2",
                "file_sha256": "b" * 64,
                "v3_plan_file_sha256": "c" * 64,
            },
        },
    }
    capture_claim = {
        "schema_version": 1,
        "artifact_kind": "hope_post_swing_natural_wrap_exclusive_claim",
        "producer_source_sha256": "8" * 64,
        "runtime_hard_contract_sha256": "3" * 64,
        "target_count": count,
        "motion_clips": receipt["motion_clips"],
        "joint_names": list(_A3_JOINTS),
        "exclusive_create": True,
    }
    claim_path = tmp_path / "natural_wrap_capture.claim.json"
    claim_path.write_text(
        json.dumps(capture_claim, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_claim_sha256"] = _sha256(claim_path)
    capture_result = {
        "schema_version": 2,
        "artifact_kind": "hope_post_swing_natural_wrap_capture_result",
        "capture_contract": receipt["capture_contract"],
        "evidence": {
            "producer_source_sha256": "8" * 64,
            "runtime_hard_contract_sha256": "3" * 64,
            "exclusive_claim_sha256": _sha256(claim_path),
            "exclusive_claim_relative_path": "natural_wrap_capture.claim.json",
            "no_clobber": True,
        },
        "motion_clips": receipt["motion_clips"],
        "states": {
            key: value for key, value in receipt["states"].items() if key != "velocity_limits"
        },
    }
    capture_path = tmp_path / "natural_wrap_capture.json"
    capture_path.write_text(
        json.dumps(capture_result, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_result_sha256"] = _sha256(capture_path)
    # a1b0b1b0 起 attestor 授权强制收据规范名:必须是 capture 目录下的 teacher_receipt.json。
    receipt_path = tmp_path / "teacher_receipt.json"
    authorization = {
        "schema_version": 1,
        "artifact_kind": "hope_post_swing_teacher_attestor_retry_authorization",
        "authorization_id": "test-v3-attestor-attempt2",
        "v3_plan": {"plan_id": tmp_path.name, "file_sha256": "c" * 64},
        "capture": {
            "output_directory": str(tmp_path.absolute()),
            "output_receipt": str(receipt_path.absolute()),
            "capture_claim_sha256": _sha256(claim_path),
            "states_sha256": _sha256(payload),
            "result_sha256": _sha256(capture_path),
            "state_count": count,
        },
        "teacher": {
            "checkpoint_sha256": "2" * 64,
            "hard_contract_sha256": "3" * 64,
            "launch_claim_content_sha256": "5" * 64,
        },
        "capture_source": {
            "commit": "6" * 40,
            "producer_source_sha256": "8" * 64,
        },
        "attestor_source": {
            "commit": "7" * 40,
            "attestor_source_sha256": "9" * 64,
        },
        "decision": {
            "capture_retry_authorized": False,
            "attestor_attempt2_authorized": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
        },
    }
    authorization_path = tmp_path / "retry_authorization.json"
    authorization_path.write_text(
        json.dumps(authorization, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["retry_authorization"]["file_sha256"] = _sha256(
        authorization_path
    )
    receipt_path.write_text(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path, _sha256(receipt_path), authorization_path, _sha256(
        authorization_path
    )


def test_post_swing_activation_disabled_stays_zero_without_replay_adoption(clips):
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]], num_envs=4, post_swing_start_prob=0.0
    )
    _prime_post_swing_buffer(cmd)
    cmd._resample_command(torch.arange(4))

    assert _counter_values(cmd.consume_post_swing_activation_counters()) == {
        "post_swing_replay_buffer_not_ready_reset_count": 0,
        "post_swing_replay_eligible_reset_count": 0,
        "post_swing_replay_random_not_selected_reset_count": 0,
        "post_swing_replay_selected_reset_count": 0,
        "post_swing_replay_started_reset_count": 0,
    }
    # The legacy RSI path still writes exactly one joint/root batch; enabling instrumentation
    # alone must not turn a disabled post-swing mechanism into a replay state write.
    assert [call[0] for call in robot.calls] == ["joint", "root"]


def test_post_swing_activation_counts_buffer_not_ready_as_ineligible(clips):
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]], num_envs=5, post_swing_start_prob=0.5
    )
    assert cmd._post_swing_count < int(cmd.cfg.post_swing_min_fill)
    cmd._resample_command(torch.arange(5))

    assert _counter_values(cmd.consume_post_swing_activation_counters()) == {
        "post_swing_replay_buffer_not_ready_reset_count": 5,
        "post_swing_replay_eligible_reset_count": 0,
        "post_swing_replay_random_not_selected_reset_count": 0,
        "post_swing_replay_selected_reset_count": 0,
        "post_swing_replay_started_reset_count": 0,
    }


def test_post_swing_activation_multi_env_selection_and_per_update_reset(clips, monkeypatch):
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        stand_start_prob=0.25,
        post_swing_start_prob=0.5,
    )
    _prime_post_swing_buffer(cmd)
    real_rand = torch.rand

    def controlled_rand(*shape, **kwargs):
        if shape == (4,):
            # stand, replay, replay, RSI: the replay interval is [0.25, 0.75).
            return torch.tensor([0.10, 0.30, 0.70, 0.90], device=kwargs.get("device"))
        return real_rand(*shape, **kwargs)

    monkeypatch.setattr(torch, "rand", controlled_rand)
    cmd._resample_command(torch.arange(4))
    cmd._resample_command(torch.arange(4))
    assert _counter_values(cmd.consume_post_swing_activation_counters()) == {
        "post_swing_replay_buffer_not_ready_reset_count": 0,
        "post_swing_replay_eligible_reset_count": 8,
        "post_swing_replay_random_not_selected_reset_count": 4,
        "post_swing_replay_selected_reset_count": 4,
        "post_swing_replay_started_reset_count": 4,
    }
    # A second logger consumption is the next PPO update's empty window, not a cumulative total.
    assert all(
        value == 0
        for value in _counter_values(
            cmd.consume_post_swing_activation_counters()
        ).values()
    )


def test_post_swing_selected_is_not_started_until_state_write_returns(clips, monkeypatch):
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]], num_envs=3, post_swing_start_prob=1.0
    )
    _prime_post_swing_buffer(cmd)

    def fail_before_adoption(_env_ids):
        raise RuntimeError("synthetic replay state write failure")

    monkeypatch.setattr(cmd, "_write_post_swing_states", fail_before_adoption)
    with pytest.raises(RuntimeError, match="synthetic replay state write failure"):
        cmd._resample_command(torch.arange(3))
    snapshot = _counter_values(cmd.consume_post_swing_activation_counters())
    assert snapshot["post_swing_replay_eligible_reset_count"] == 3
    assert snapshot["post_swing_replay_selected_reset_count"] == 3
    assert snapshot["post_swing_replay_started_reset_count"] == 0
    assert snapshot["post_swing_replay_random_not_selected_reset_count"] == 0


def test_post_swing_teacher_cold_start_is_ready_and_activates_before_learning(
    clips, tmp_path, monkeypatch
):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=1.0,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
    )
    assert cmd._post_swing_count == 4
    assert cmd._post_swing_ptr == 4
    contract = cmd.post_swing_replay_hard_contract()
    assert contract["teacher_distribution"] == "immutable"
    assert contract["teacher_receipt"]["receipt_sha256"] == receipt_sha

    cmd._resample_command(torch.arange(4))
    snapshot = _counter_values(cmd.consume_training_activation_counters())
    assert snapshot["post_swing_replay_buffer_not_ready_reset_count"] == 0
    assert snapshot["post_swing_replay_eligible_reset_count"] == 4
    assert snapshot["post_swing_replay_selected_reset_count"] == 4
    assert snapshot["post_swing_replay_started_reset_count"] == 4
    assert [call[0] for call in robot.calls] == ["root", "joint"]

    # A frozen controlled-direct-effect buffer ignores later live captures.  The state count and
    # pointer remain bound to the immutable receipt instead of becoming treatment-dependent.
    cmd._capture_post_swing_states(torch.arange(4))
    assert cmd._post_swing_count == 4
    assert cmd._post_swing_ptr == 4


def test_post_swing_teacher_fail_fast_refuses_zero_activation_before_first_update(
    clips, tmp_path, monkeypatch
):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=0.25,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
    )
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.full(shape, 0.9))
    with pytest.raises(RuntimeError, match="adopted count below"):
        cmd._resample_command(torch.arange(4))


def test_post_swing_first_reset_checks_count_fraction_and_state_readback(clips, tmp_path):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=1.0,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
        post_swing_first_reset_min_adopted_count=4,
        post_swing_first_reset_min_adopted_fraction=1.0,
        post_swing_first_reset_selection_tolerance=0.0,
        post_swing_first_reset_require_readback=True,
    )
    cmd._resample_command(torch.arange(4))
    assert cmd._post_swing_first_reset_checked is True
    assert torch.allclose(robot.data.root_state_w[:, 2], torch.ones(4))


def test_post_swing_first_reset_readback_cannot_be_forged_by_selected_count(
    clips, tmp_path, monkeypatch
):
    receipt, receipt_sha, authorization, authorization_sha = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    cmd, robot = _make_motion_command(
        [clips[0], clips[1]],
        num_envs=4,
        post_swing_start_prob=1.0,
        post_swing_min_fill=4,
        post_swing_buffer_size=8,
        post_swing_teacher_receipt=str(receipt),
        post_swing_teacher_receipt_sha256=receipt_sha,
        post_swing_teacher_retry_authorization=str(authorization),
        post_swing_teacher_retry_authorization_sha256=authorization_sha,
        post_swing_teacher_root_linear_velocity_limit_mps=2.0,
        post_swing_teacher_root_angular_velocity_limit_radps=4.0,
        post_swing_require_ready_at_init=True,
        post_swing_fail_fast_first_reset=True,
        post_swing_first_reset_min_adopted_count=4,
        post_swing_first_reset_min_adopted_fraction=1.0,
        post_swing_first_reset_selection_tolerance=0.0,
        post_swing_first_reset_require_readback=True,
    )
    monkeypatch.setattr(
        robot,
        "write_root_state_to_sim",
        lambda root, env_ids=None: robot.calls.append(("root", root.clone(), env_ids.clone())),
    )
    with pytest.raises(RuntimeError, match="root readback differs"):
        cmd._resample_command(torch.arange(4))


def test_post_swing_ready_at_init_requires_exact_teacher_receipt(clips):
    with pytest.raises(ValueError, match="require an immutable"):
        _make_motion_command(
            [clips[0], clips[1]],
            post_swing_start_prob=0.25,
            post_swing_require_ready_at_init=True,
        )


def test_post_swing_teacher_receipt_requires_frozen_retry_authorization(
    clips, tmp_path
):
    receipt, receipt_sha, _, _ = _write_post_swing_teacher_receipt(
        tmp_path, [clips[0], clips[1]], count=4
    )
    with pytest.raises(ValueError, match="receipt and retry authorization"):
        _make_motion_command(
            [clips[0], clips[1]],
            post_swing_start_prob=1.0,
            post_swing_min_fill=4,
            post_swing_teacher_receipt=str(receipt),
            post_swing_teacher_receipt_sha256=receipt_sha,
        )


def test_rally_hold_heading_is_hold_only_and_monotone():
    yaw = torch.tensor([0.0, 0.6, 0.6])
    quat = torch.zeros(3, 4)
    quat[:, 0] = torch.cos(yaw / 2.0)
    quat[:, 3] = torch.sin(yaw / 2.0)
    motion = types.SimpleNamespace(in_hold=torch.tensor([True, True, False]))
    cmd = types.SimpleNamespace(
        base_quat_w=quat,
        num_envs=3,
        device="cpu",
        _motion=lambda: motion,
    )
    reward = hope_rewards_mod.hold_heading(_fake_env(racket_target=cmd), "racket_target", std=0.6)
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] == pytest.approx(float(torch.exp(torch.tensor(-1.0))))
    assert reward[2] == 0.0
    with pytest.raises(ValueError, match="finite and > 0"):
        hope_rewards_mod.hold_heading(_fake_env(racket_target=cmd), "racket_target", std=0.0)


def test_rally_foot_orientation_gate_only_zeros_hold_rows():
    motion = types.SimpleNamespace(
        in_hold=torch.tensor([True, False]),
        joint_pos=torch.zeros(2, 2),
    )
    asset = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=torch.tensor([[1.0, 2.0], [1.0, 2.0]])))
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda _name: motion),
        scene={"robot": asset},
    )
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[0, 1])
    always = hope_rewards_mod.foot_orientation_discipline(env, "motion", asset_cfg, hold_gate=False)
    gated = hope_rewards_mod.foot_orientation_discipline(env, "motion", asset_cfg, hold_gate=True)
    assert torch.equal(always, torch.tensor([3.0, 3.0]))
    assert torch.equal(gated, torch.tensor([0.0, 3.0]))


def _qdot_limit_env(limits=None):
    names = [f"joint_{index:02d}" for index in range(31)]
    joint_vel = torch.zeros(2, 31)
    joint_vel[0, :] = 8.5
    joint_vel[1, 0] = 15.0
    if limits is None:
        limits = torch.full((2, 31), 10.0)
    asset = types.SimpleNamespace(
        joint_names=list(names),
        data=types.SimpleNamespace(
            joint_names=list(names),
            joint_vel=joint_vel,
            joint_vel_limits=limits,
        ),
    )
    env = types.SimpleNamespace(scene={"robot": asset}, common_step_counter=1)
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=list(range(31)))
    return env, asset_cfg


def test_qdot_limit_hinge_uses_actual_runtime_limits_and_normalized_sum():
    # 2026-07-25 SUM 裁定:单关节违规不再被 ÷31 稀释,罚整份
    env, asset_cfg = _qdot_limit_env()
    result = hope_rewards_mod.joint_velocity_limit_hinge(
        env, asset_cfg, margin=0.85, expected_joint_count=31
    )
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx((1.5 - 0.85) ** 2)


def test_qdot_probe_and_hinge_share_one_observation_but_book_activation_separately():
    env, asset_cfg = _qdot_limit_env()
    probe = hope_rewards_mod.joint_velocity_limit_hinge_probe(env, asset_cfg)
    assert torch.equal(probe, torch.zeros(2))
    reward = hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)
    assert reward[1] > 0.0

    snapshot = hope_rewards_mod.consume_joint_velocity_limit_hinge_activation_counters(env)
    assert snapshot["observed_sample_count"].item() == 2
    assert snapshot["hinge_active_sample_count"].item() == 2
    assert snapshot["excess_sample_count"].item() == 1
    assert snapshot["normalized_excess_square_sum"].item() == pytest.approx(
        reward.sum().item()
    )
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_joint_velocity_limit_hinge_activation_counters(env).values()
    )


def test_sparse_virtual_reward_ledger_is_exact_per_action_and_resets_once():
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command._clip_names = {0: "forehand", 1: "backhand"}
    # 族行号表(0=正手族,1=反手族):真 __init__ 会懒建;fake 直接预置 legacy 2-clip 同值表。
    command._clip_family_rows_t = torch.tensor([0, 1], dtype=torch.long)
    names = (
        "strike_opportunity_count",
        "virtual_capture_count",
        "virtual_net_clear_count",
        "virtual_landing_valid_count",
        "virtual_legal_return_count",
    )
    command._sparse_reward_eligibility_counters = {
        name: torch.zeros((), dtype=torch.long) for name in names
    }
    for family in command._clip_names.values():
        for name in names:
            command._sparse_reward_eligibility_counters[f"{name}_{family}"] = torch.zeros(
                (), dtype=torch.long
            )
    command._motion_term = types.SimpleNamespace(
        _multiseg=True, clip_id=torch.tensor([0, 0, 1, 1])
    )
    command._event_timing_bound = True

    command._book_sparse_reward_eligibility(
        exact_strike=torch.tensor([True, True, True, True]),
        capture=torch.tensor([True, False, True, False]),
        net_clear=torch.tensor([True, False, False, False]),
        landing_valid=torch.tensor([True, False, True, False]),
        legal_return=torch.tensor([True, False, False, False]),
    )
    snapshot = command.consume_sparse_reward_eligibility_counters()
    assert snapshot["strike_opportunity_count"].item() == 4
    assert snapshot["virtual_capture_count"].item() == 2
    assert snapshot["virtual_net_clear_count"].item() == 1
    assert snapshot["virtual_landing_valid_count"].item() == 2
    assert snapshot["virtual_legal_return_count"].item() == 1
    assert snapshot["strike_opportunity_count_forehand"].item() == 2
    assert snapshot["strike_opportunity_count_backhand"].item() == 2
    assert snapshot["virtual_capture_count_forehand"].item() == 1
    assert snapshot["virtual_capture_count_backhand"].item() == 1
    assert snapshot["virtual_legal_return_count_forehand"].item() == 1
    assert snapshot["virtual_legal_return_count_backhand"].item() == 0
    assert all(
        value.item() == 0
        for value in command.consume_sparse_reward_eligibility_counters().values()
    )


def test_qdot_limit_hinge_reloads_runtime_limits_after_first_call():
    limits = torch.full((2, 31), 10.0)
    env, asset_cfg = _qdot_limit_env(limits)
    first = hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)
    assert first[1] > 0.0

    # A runtime limit update must affect the very next reward; a first-call cache would keep
    # charging against 10 rad/s and fail this assertion.
    limits.fill_(20.0)
    second = hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)
    assert torch.equal(second, torch.zeros_like(second))

    # The same live path must fail closed if a later update introduces an invalid limit.
    limits[:, 4] = 0.0
    with pytest.raises(RuntimeError, match="finite, positive"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)


def test_qdot_limit_hinge_rejects_order_zero_nonfinite_and_per_env_drift():
    env, asset_cfg = _qdot_limit_env()
    with pytest.raises(ValueError, match="exact 31-joint"):
        hope_rewards_mod.joint_velocity_limit_hinge(
            env, asset_cfg, expected_joint_count=31.0
        )

    asset_cfg.joint_ids[0], asset_cfg.joint_ids[1] = 1, 0
    with pytest.raises(RuntimeError, match="identity 31-joint"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)

    limits = torch.full((2, 31), 10.0)
    limits[:, 7] = 0.0
    env, asset_cfg = _qdot_limit_env(limits)
    with pytest.raises(RuntimeError, match="finite, positive"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)

    limits = torch.full((2, 31), 10.0)
    limits[:, 8] = float("nan")
    env, asset_cfg = _qdot_limit_env(limits)
    with pytest.raises(RuntimeError, match="finite, positive"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)

    limits = torch.full((2, 31), 10.0)
    limits[1, 8] = 11.0
    env, asset_cfg = _qdot_limit_env(limits)
    with pytest.raises(RuntimeError, match="identical articulation velocity limits"):
        hope_rewards_mod.joint_velocity_limit_hinge(env, asset_cfg)


def test_clamped_action_keeps_reset_aware_previous_processed_qdes():
    n = 2
    names = list(_A3_JOINTS)
    limits = torch.stack(
        (
            torch.full((n, len(names)), -0.5),
            torch.full((n, len(names)), 0.5),
        ),
        dim=-1,
    )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            default_joint_pos=torch.zeros(n, len(names)),
            soft_joint_pos_limits=limits,
        )
    )
    cfg = types.SimpleNamespace(
        asset_name="robot", scale=1.0, use_default_offset=True, clamp=True
    )
    env = types.SimpleNamespace(
        scene={"robot": asset}, num_envs=n, device="cpu"
    )
    action = hope_actions_mod.ClampedJointPositionAction(cfg, env)

    first = torch.zeros(n, len(names))
    first[:, 0] = torch.tensor([0.8, -0.8])
    action.process_actions(first)
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.5, -0.5])
    assert action.previous_processed_qdes_valid.tolist() == [False, False]

    second = torch.full((n, len(names)), 0.1)
    action.process_actions(second)
    assert action.previous_processed_qdes[:, 0].tolist() == pytest.approx([0.5, -0.5])
    assert action.previous_processed_qdes_valid.tolist() == [True, True]

    action.reset(env_ids=torch.tensor([0]))
    third = torch.full((n, len(names)), 0.2)
    action.process_actions(third)
    assert action.previous_processed_qdes[:, 0].tolist() == pytest.approx([0.1, 0.1])
    assert action.previous_processed_qdes_valid.tolist() == [False, True]


def test_post_strike_clock_counts_control_ticks_and_closes_on_wrap_or_reveal():
    cmd = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    cmd.num_envs = 4
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(step_dt=0.02, common_step_counter=10)
    cmd.pre_strike = torch.zeros(4, dtype=torch.bool)
    cmd._post_strike_elapsed_s = torch.zeros(4)
    cmd._post_strike_elapsed_valid = torch.zeros(4, dtype=torch.bool)
    cmd._post_strike_elapsed_last_step = -1
    motion = types.SimpleNamespace(
        retiming_active=True,
        time_steps_f=torch.full((4,), 60.0),
        # Deliberately time-varying and irrelevant: wall time must not be inferred from it.
        speed_scale=torch.tensor([0.1, 0.5, 2.0, 4.0]),
        just_resampled=torch.zeros(4, dtype=torch.bool),
        event_just_installed=torch.zeros(4, dtype=torch.bool),
    )
    cmd._motion_term = motion
    exact = torch.tensor([True, False, True, False])
    hope_commands_mod.RacketTargetCommand._advance_post_strike_elapsed(cmd, exact)
    age, same_attempt = cmd.post_strike_age_and_same_attempt()
    assert age.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert same_attempt.tolist() == [True, False, True, False]

    # A duplicate call in one manager step is idempotent; the next unique tick adds exactly 20 ms.
    hope_commands_mod.RacketTargetCommand._advance_post_strike_elapsed(
        cmd, torch.zeros(4, dtype=torch.bool)
    )
    assert age.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    cmd._env.common_step_counter = 11
    motion.just_resampled[0] = True
    motion.event_just_installed[2] = True
    hope_commands_mod.RacketTargetCommand._advance_post_strike_elapsed(
        cmd, torch.zeros(4, dtype=torch.bool)
    )
    assert age.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert same_attempt.tolist() == [False, False, False, False]


def test_post_strike_clock_advances_exactly_twenty_ms_per_unique_tick():
    cmd = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    cmd.num_envs = 1
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(step_dt=0.02, common_step_counter=1)
    cmd.pre_strike = torch.zeros(1, dtype=torch.bool)
    cmd._post_strike_elapsed_s = torch.zeros(1)
    cmd._post_strike_elapsed_valid = torch.zeros(1, dtype=torch.bool)
    cmd._post_strike_elapsed_last_step = -1
    cmd._motion_term = types.SimpleNamespace(
        just_resampled=torch.zeros(1, dtype=torch.bool),
        event_just_installed=torch.zeros(1, dtype=torch.bool),
        speed_scale=torch.tensor([4.0]),
    )
    cmd._advance_post_strike_elapsed(torch.ones(1, dtype=torch.bool))
    for step in range(2, 12):
        cmd._env.common_step_counter = step
        cmd._motion_term.speed_scale.fill_(0.1 if step % 2 else 4.0)
        cmd._advance_post_strike_elapsed(torch.zeros(1, dtype=torch.bool))
    age, valid = cmd.post_strike_age_and_same_attempt()
    assert age.item() == pytest.approx(0.20)
    assert valid.item() is True


def test_post_strike_clock_arms_on_positive_half_tick_exact_opportunity():
    cmd = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    cmd.num_envs = 1
    cmd.device = "cpu"
    cmd._env = types.SimpleNamespace(step_dt=0.02, common_step_counter=1)
    # exact_strike may fire at +8 ms, before the strict tts>0 pre-strike flag falls.
    cmd.pre_strike = torch.ones(1, dtype=torch.bool)
    cmd._post_strike_elapsed_s = torch.zeros(1)
    cmd._post_strike_elapsed_valid = torch.zeros(1, dtype=torch.bool)
    cmd._post_strike_elapsed_last_step = -1
    cmd._motion_term = types.SimpleNamespace(
        just_resampled=torch.zeros(1, dtype=torch.bool),
        event_just_installed=torch.zeros(1, dtype=torch.bool),
    )
    cmd._advance_post_strike_elapsed(torch.ones(1, dtype=torch.bool))
    age, valid = cmd.post_strike_age_and_same_attempt()
    assert age.item() == 0.0
    assert valid.item() is True

    cmd._env.common_step_counter = 2
    cmd.pre_strike.zero_()
    cmd._advance_post_strike_elapsed(torch.zeros(1, dtype=torch.bool))
    assert age.item() == pytest.approx(0.02)
    assert valid.item() is True


def _processed_qdes_slew_env(*, step_dt=0.02):
    n = 4
    names = list(_A3_JOINTS)
    selected = [
        index
        for index, name in enumerate(names)
        if name in hope_rewards_mod._PROCESSED_QDES_RECOVERY_JOINT_NAMES
    ]
    previous = torch.zeros(n, len(names))
    processed = torch.zeros_like(previous)
    # qdot limit 10 rad/s * 0.02 s = 0.2 rad allowance. env0 stays below margin;
    # env1 has u=1 on every selected joint; env2 is reset-invalid; env3 is off-window.
    processed[0, selected] = 0.1
    processed[1, selected] = 0.2
    processed[2, selected] = 2.0
    processed[3, selected] = 0.2
    # An arbitrarily large arm target must not enter the 15-joint mean.
    arm_index = names.index("right_elbow_joint")
    processed[:, arm_index] = 100.0
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            joint_vel_limits=torch.full((n, len(names)), 10.0),
        )
    )
    action = types.SimpleNamespace(
        processed_actions=processed,
        previous_processed_qdes=previous,
        previous_processed_qdes_valid=torch.tensor([True, True, False, True]),
        _asset=asset,
        _joint_names=names,
        _joint_ids=slice(None),
    )
    command = types.SimpleNamespace(
        post_strike_age_and_same_attempt=lambda: (
            torch.tensor([0.20, 1.55, 0.50, 0.50]),
            torch.tensor([True, True, True, False]),
        )
    )
    return types.SimpleNamespace(
        step_dt=step_dt,
        common_step_counter=17,
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
        command_manager=types.SimpleNamespace(get_term=lambda name: command),
    )


def test_processed_qdes_slew_formula_gate_invalid_first_step_and_update_ledger():
    env = _processed_qdes_slew_env()
    probe = hope_rewards_mod.processed_qdes_slew_hinge_probe(env)
    reward = hope_rewards_mod.processed_qdes_slew_hinge(env)
    expected_tail = 1.0 - math.exp(-1.0)
    assert torch.equal(probe, torch.zeros(4))
    # 2026-07-25 SUM 裁定:15 关节全违规 = 15 份 tail(旧 mean 语义下是 1 份)
    assert reward.tolist() == pytest.approx([0.0, 15.0 * expected_tail, 0.0, 0.0])

    counters = (
        hope_rewards_mod.consume_processed_qdes_slew_hinge_activation_counters(env)
    )
    assert counters["observed_sample_count"].item() == 4
    assert counters["previous_qdes_valid_sample_count"].item() == 3
    assert counters["previous_qdes_invalid_first_step_sample_count"].item() == 1
    assert counters["recovery_eligible_sample_count"].item() == 2
    assert counters["reward_enabled_eligible_sample_count"].item() == 2
    assert counters["tail_active_sample_count"].item() == 1
    assert counters["above_margin_joint_count"].item() == 15
    assert counters["gated_tail_value_sum"].item() == pytest.approx(15.0 * expected_tail)
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_processed_qdes_slew_hinge_activation_counters(
            env
        ).values()
    )


def test_processed_qdes_slew_rejects_control_dt_drift():
    with pytest.raises(RuntimeError, match=r"step_dt == 0\.02"):
        hope_rewards_mod.processed_qdes_slew_hinge(
            _processed_qdes_slew_env(step_dt=0.01)
        )


def _exact_behavior_command(termination_manager, num_envs=4):
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command.device = "cpu"
    command.num_envs = num_envs
    command._env = types.SimpleNamespace(termination_manager=termination_manager)
    command._exact_behavior_decision_counters = {}
    command._exact_attempt_active = torch.zeros(num_envs, dtype=torch.bool)
    command._exact_attempt_completed = torch.zeros(num_envs, dtype=torch.bool)
    command._exact_pending_completion = torch.zeros(num_envs, dtype=torch.bool)
    command._clip_names = {0: "forehand", 1: "backhand"}
    sparse_names = (
        "strike_opportunity_count",
        "virtual_capture_count",
        "virtual_net_clear_count",
        "virtual_landing_valid_count",
        "virtual_legal_return_count",
    )
    command._sparse_reward_eligibility_counters = {
        name: torch.zeros((), dtype=torch.long) for name in sparse_names
    }
    for family in command._clip_names.values():
        for name in sparse_names:
            command._sparse_reward_eligibility_counters[f"{name}_{family}"] = torch.zeros(
                (), dtype=torch.long
            )
    command._ensure_exact_behavior_decision_counters()
    return command


def test_exact_behavior_terminal_reset_separates_physical_pre_post_and_guards():
    reasons = {
        "base_fell_tilt": torch.tensor([True, False, False, False]),
        "base_too_low": torch.tensor([False, False, True, True]),
        "anchor_pos": torch.tensor([False, True, False, False]),
        "time_out": torch.zeros(4, dtype=torch.bool),
    }
    tm = types.SimpleNamespace(
        active_terms=list(reasons),
        get_term=lambda name: reasons[name],
        terminated=torch.ones(4, dtype=torch.bool),
        time_outs=torch.zeros(4, dtype=torch.bool),
    )
    command = _exact_behavior_command(tm)
    # Env 0 fell before its private task reveal.  The remaining falls were
    # publicly revealed and occurred after the nominal strike boundary.
    command._action_ball_task_valid = torch.tensor(
        [False, True, True, True], dtype=torch.bool
    )
    command._exact_attempt_active[:] = True
    command._close_exact_swing_attempts(torch.arange(4))
    command._book_exact_behavior_terminal_reset(
        torch.arange(4),
        pre_strike=torch.tensor([True, True, False, True]),
        recovering=torch.tensor([False, False, False, True]),
    )

    first = command.consume_exact_behavior_decision_counters()
    assert first["terminal_reset_count"].item() == 4
    assert first["swing_outcome_count"].item() == 4
    assert first["swing_completion_count"].item() == 0
    assert first["physical_fall_count"].item() == 3
    assert first["pre_strike_physical_fall_count"].item() == 1
    assert first["post_strike_physical_fall_count"].item() == 2
    assert first["non_physical_terminal_reset_count"].item() == 1
    assert first["termination_reason_base_fell_tilt_count"].item() == 1
    assert first["termination_reason_base_too_low_count"].item() == 2
    assert (
        first["termination_reason_base_fell_tilt_hidden_wait_count"].item()
        == 1
    )
    assert (
        first[
            "termination_reason_base_fell_tilt_revealed_pre_strike_count"
        ].item()
        == 0
    )
    assert first["termination_reason_base_fell_tilt_post_strike_count"].item() == 0
    assert first["termination_reason_base_too_low_hidden_wait_count"].item() == 0
    assert (
        first[
            "termination_reason_base_too_low_revealed_pre_strike_count"
        ].item()
        == 0
    )
    assert first["termination_reason_base_too_low_post_strike_count"].item() == 2
    for reason in ("base_fell_tilt", "base_too_low"):
        assert first[f"termination_reason_{reason}_count"].item() == sum(
            first[f"termination_reason_{reason}_{phase}_count"].item()
            for phase in ("hidden_wait", "revealed_pre_strike", "post_strike")
        )
    assert first["termination_reason_anchor_pos_count"].item() == 1
    # Consume is a hard PPO-update boundary: the next transaction cannot inherit any event.
    second = command.consume_exact_behavior_decision_counters()
    assert all(value.item() == 0 for value in second.values())


def test_task_wait_reveal_books_one_edge_per_environment_and_update_transaction():
    command = _exact_behavior_command(
        types.SimpleNamespace(active_terms=()), num_envs=4
    )
    command._action_ball_task_wait_schedule = object()
    command._action_ball_task_valid = torch.tensor(
        [False, False, True, False], dtype=torch.bool
    )
    command._action_ball_task_wait_elapsed_ticks = torch.tensor(
        [0, 1, 0, 2], dtype=torch.long
    )
    command._action_ball_task_wait_total_ticks = torch.tensor(
        [1, 2, 0, 4], dtype=torch.long
    )
    command._action_ball_task_wait_last_advance_step = 0
    command._action_ball_diagnostic_unauthorized = False
    command._env.common_step_counter = 1
    refreshed = []
    command._motion = lambda: types.SimpleNamespace(
        refresh_action_ball_revealed_body_reference=(
            lambda mask: refreshed.append(mask.detach().clone())
        )
    )

    command._advance_action_ball_task_wait()
    assert torch.equal(
        command._action_ball_task_valid,
        torch.tensor([True, True, True, False]),
    )
    assert command._exact_behavior_decision_counters[
        "task_reveal_reached_count"
    ].item() == 2
    assert torch.equal(refreshed[-1], torch.tensor([True, True, False, False]))

    # Same policy-step token is idempotent; the remaining task reveals once on
    # the next token and all three edges are delivered in the update snapshot.
    command._advance_action_ball_task_wait()
    assert command._exact_behavior_decision_counters[
        "task_reveal_reached_count"
    ].item() == 2
    command._env.common_step_counter = 2
    command._advance_action_ball_task_wait()
    snapshot = command.consume_exact_behavior_decision_counters()
    assert snapshot["task_reveal_reached_count"].item() == 3
    assert command.consume_exact_behavior_decision_counters()[
        "task_reveal_reached_count"
    ].item() == 0


def test_task_wait_arm_books_hidden_phase_exposure_denominator_once():
    command = _exact_behavior_command(
        types.SimpleNamespace(active_terms=()), num_envs=3
    )
    command._action_ball_task_wait_schedule = object()
    command._action_ball_task_wait_highwater = types.SimpleNamespace(
        record=lambda **_kwargs: types.SimpleNamespace(wait_ticks=2)
    )
    command._action_ball_task_wait_total_ticks = torch.zeros(3, dtype=torch.long)
    command._action_ball_task_wait_elapsed_ticks = torch.zeros(3, dtype=torch.long)
    command._action_ball_task_valid = torch.ones(3, dtype=torch.bool)
    command.time_to_strike = torch.zeros(3)
    command._env.step_dt = 0.02
    time_to_contact = torch.ones(3, dtype=torch.float64)
    pre_swing_wait = torch.ones(3, dtype=torch.float64)
    motion = types.SimpleNamespace(
        _action_ball_time_to_contact_s=time_to_contact,
        _action_ball_pre_swing_wait_s=pre_swing_wait,
        action_ball_time_to_contact_remaining_s=time_to_contact,
        bind_action_ball_public_task_valid=lambda _value: None,
    )
    command._motion = lambda: motion

    command._action_ball_arm_task_wait(
        torch.tensor([0, 2]),
        host_identity_rows=((0, 0, 0, 1), (2, 0, 0, 1)),
        true_reset=True,
    )

    assert command._exact_behavior_decision_counters[
        "task_wait_started_count"
    ].item() == 2
    assert torch.equal(
        command._action_ball_task_valid,
        torch.tensor([False, True, False]),
    )
    assert torch.equal(
        command._action_ball_task_wait_total_ticks,
        torch.tensor([2, 0, 2]),
    )


def test_each_fall_reason_partitions_hidden_revealed_pre_and_post_strike():
    reasons = {
        "base_fell_tilt": torch.ones(3, dtype=torch.bool),
        "base_too_low": torch.ones(3, dtype=torch.bool),
        "robot_hit_table": torch.ones(3, dtype=torch.bool),
    }
    command = _exact_behavior_command(
        types.SimpleNamespace(
            active_terms=list(reasons),
            get_term=lambda name: reasons[name],
            terminated=torch.ones(3, dtype=torch.bool),
            time_outs=torch.zeros(3, dtype=torch.bool),
        ),
        num_envs=3,
    )
    command._action_ball_task_valid = torch.tensor(
        [False, True, True], dtype=torch.bool
    )

    command._book_exact_behavior_terminal_reset(
        torch.arange(3),
        pre_strike=torch.tensor([True, True, False]),
        recovering=torch.zeros(3, dtype=torch.bool),
    )
    snapshot = command.consume_exact_behavior_decision_counters()

    for reason in ("base_fell_tilt", "base_too_low", "robot_hit_table"):
        assert snapshot[f"termination_reason_{reason}_count"].item() == 3
        assert snapshot[
            f"termination_reason_{reason}_hidden_wait_count"
        ].item() == 1
        assert snapshot[
            f"termination_reason_{reason}_revealed_pre_strike_count"
        ].item() == 1
        assert snapshot[
            f"termination_reason_{reason}_post_strike_count"
        ].item() == 1


def test_exact_swing_closeout_pairs_window_denominator_and_transfers_wrap_strike():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=2)
    # Two attempts started before this decision window.  Only the first reached its strike;
    # a defensive exact strike on the current wrap step is parked for the second env's NEW task.
    command._exact_attempt_active[:] = True
    command._exact_attempt_completed[:] = torch.tensor([True, False])
    command._exact_pending_completion[:] = torch.tensor([False, True])

    command._close_exact_swing_attempts(torch.arange(2))
    first = command.consume_exact_behavior_decision_counters()
    assert first["swing_start_count"].item() == 0
    assert first["strike_opportunity_count"].item() == 0
    assert first["swing_outcome_count"].item() == 2
    assert first["swing_completion_count"].item() == 1
    assert torch.equal(command._exact_attempt_completed, torch.tensor([False, True]))
    assert not bool(command._exact_pending_completion.any())

    # The parked wrap strike is paired only when its new attempt later closes; no prior-window
    # start/strike counter is needed to reconstruct this window's bounded completion ratio.
    command._close_exact_swing_attempts(torch.tensor([1]))
    second = command.consume_exact_behavior_decision_counters()
    assert second["swing_outcome_count"].item() == 1
    assert second["swing_completion_count"].item() == 1


def test_exact_initial_tts_buckets_pair_closeout_and_sparse_outcomes_per_update():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=4)
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        _multiseg=False,
        just_resampled=torch.zeros(4, dtype=torch.bool),
    )
    command._exact_attempt_active[:] = True
    with torch.inference_mode():
        command._assign_exact_attempt_initial_tts(
            torch.arange(4), torch.tensor([0.49, 0.50, 0.90, 0.91])
        )
        command._book_sparse_reward_eligibility(
            exact_strike=torch.tensor([True, True, True, False]),
            capture=torch.tensor([True, False, True, False]),
            net_clear=torch.tensor([True, False, True, False]),
            landing_valid=torch.tensor([True, False, True, False]),
            legal_return=torch.tensor([False, False, True, False]),
        )
        command._exact_attempt_completed[:] = torch.tensor([True, False, True, False])
        command._close_exact_swing_attempts(torch.arange(4))
    snapshot = command.consume_exact_behavior_decision_counters()

    for bucket in ("lt_0p5", "eq_0p5", "gt_0p5_le_0p9", "gt_0p9"):
        assert snapshot[f"planner_initial_tts_{bucket}_swing_outcome_count"].item() == 1
    assert snapshot["planner_initial_tts_lt_0p5_swing_completion_count"].item() == 1
    assert snapshot["planner_initial_tts_eq_0p5_swing_completion_count"].item() == 0
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_swing_completion_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_swing_completion_count"].item() == 0
    assert snapshot["planner_initial_tts_lt_0p5_exact_strike_timing_tick_count"].item() == 1
    assert snapshot["planner_initial_tts_eq_0p5_exact_strike_timing_tick_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_exact_strike_timing_tick_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_exact_strike_timing_tick_count"].item() == 0
    assert snapshot["planner_initial_tts_lt_0p5_virtual_capture_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_virtual_capture_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p5_le_0p9_virtual_legal_return_count"].item() == 1
    assert snapshot["planner_initial_tts_lt_0p5_virtual_legal_return_count"].item() == 0

    second = command.consume_exact_behavior_decision_counters()
    assert all(
        value.item() == 0
        for name, value in second.items()
        if name.startswith("planner_initial_tts_")
    )


def test_exact_initial_tts_bucket_defers_same_wrap_sparse_outcome_to_new_task():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=2)
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        _multiseg=False,
        just_resampled=torch.tensor([True, False]),
    )
    command._exact_attempt_active[:] = True
    command._assign_exact_attempt_initial_tts(
        torch.arange(2), torch.tensor([0.49, 0.50])
    )
    command._book_sparse_reward_eligibility(
        exact_strike=torch.tensor([True, True]),
        capture=torch.tensor([True, True]),
        net_clear=torch.tensor([True, True]),
        landing_valid=torch.tensor([True, True]),
        legal_return=torch.tensor([True, True]),
    )

    # Env 0's strike belongs to the task that starts on this wrap.  Close the old <0.5 task,
    # then assign the new >0.9 task and flush the parked outcomes into that new bucket.
    command._close_exact_swing_attempts(torch.tensor([0]))
    command._assign_exact_attempt_initial_tts(torch.tensor([0]), torch.tensor([1.10]))
    snapshot = command.consume_exact_behavior_decision_counters()
    assert snapshot["planner_initial_tts_lt_0p5_swing_outcome_count"].item() == 1
    assert snapshot["planner_initial_tts_lt_0p5_exact_strike_timing_tick_count"].item() == 0
    assert snapshot["planner_initial_tts_eq_0p5_exact_strike_timing_tick_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_exact_strike_timing_tick_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_virtual_capture_count"].item() == 1
    assert snapshot["planner_initial_tts_gt_0p9_virtual_legal_return_count"].item() == 1


def test_exact_terminal_reason_masks_reject_numeric_truthiness():
    reasons = {"base_fell_tilt": torch.tensor([0.0, 0.2])}
    tm = types.SimpleNamespace(
        active_terms=list(reasons),
        get_term=lambda name: reasons[name],
        terminated=torch.tensor([False, True]),
        time_outs=torch.zeros(2, dtype=torch.bool),
    )
    command = _exact_behavior_command(tm, num_envs=2)
    with pytest.raises(TypeError, match="boolean dtype"):
        command._book_exact_behavior_terminal_reset(
            torch.arange(2),
            pre_strike=torch.ones(2, dtype=torch.bool),
            recovering=torch.zeros(2, dtype=torch.bool),
        )
    snapshot = command.consume_exact_behavior_decision_counters()
    assert snapshot["terminal_reset_count"].item() == 0
    assert snapshot["physical_fall_count"].item() == 0


def test_exact_terminal_reason_masks_reject_broadcastable_matrix_shape():
    reasons = {"base_fell_tilt": torch.zeros(2, 1, dtype=torch.bool)}
    tm = types.SimpleNamespace(
        active_terms=list(reasons),
        get_term=lambda name: reasons[name],
        terminated=torch.tensor([False, True]),
        time_outs=torch.zeros(2, dtype=torch.bool),
    )
    command = _exact_behavior_command(tm, num_envs=2)
    with pytest.raises(ValueError, match="one-dimensional"):
        command._book_exact_behavior_terminal_reset(
            torch.arange(2),
            pre_strike=torch.ones(2, dtype=torch.bool),
            recovering=torch.zeros(2, dtype=torch.bool),
        )


def test_exact_ready_balance_aggregates_have_explicit_phase_and_sensor_denominators():
    tm = types.SimpleNamespace(active_terms=())
    command = _exact_behavior_command(tm)
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        in_hold=torch.tensor([True, False, True, True]),
    )
    command._event_timing_bound = True
    command.metrics = {
        "base_upright": torch.cos(torch.tensor([0.0, 0.5, 0.2, 0.4])),
        "foot_contact_frac": torch.tensor([1.0, 0.0, 0.5, 1.0]),
        "foot_slip_speed": torch.tensor([0.1, 9.0, 0.2, 0.3]),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor(
                [[0.0, 0.0, 1.0], [9.0, 9.0, 1.0], [0.3, 0.4, 1.0], [0.0, 1.0, 1.0]]
            ),
            root_lin_vel_w=torch.tensor(
                [[0.3, 0.4, 0.0], [9.0, 9.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
        )
    )
    command.base_target_pos_w = torch.zeros(4, 2)
    command._foot_idx_robot = [0, 1]
    command._foot_idx_contact = [0, 1]
    command._contact_sensor = object()

    command._book_exact_ready_behavior_samples()
    first = command.consume_exact_behavior_decision_counters()
    assert first["ready_phase_sample_count"].item() == 3
    assert first["ready_planner_task_entry_sample_count"].item() == 0
    assert first["ready_planner_legacy_hold_violation_count"].item() == 0
    assert first["ready_foot_sensor_unavailable_sample_count"].item() == 0
    for key in (
        "ready_tilt_eligible_sample_count",
        "ready_base_speed_eligible_sample_count",
        "ready_station_offset_eligible_sample_count",
        "ready_foot_contact_eligible_sample_count",
        "ready_foot_slip_eligible_sample_count",
    ):
        assert first[key].item() == 3
    assert first["ready_tilt_rad_sum"].item() == pytest.approx(0.6, abs=1e-5)
    assert first["ready_base_speed_xy_mps_sum"].item() == pytest.approx(1.5)
    assert first["ready_station_offset_m_sum"].item() == pytest.approx(1.5)
    assert first["ready_foot_contact_fraction_sum"].item() == pytest.approx(2.5)
    assert first["ready_foot_slip_speed_mps_sum"].item() == pytest.approx(0.6)

    # No hold eligibility means no fabricated zero-valued observation in the next update.
    command._motion_term.in_hold.zero_()
    command._book_exact_ready_behavior_samples()
    second = command.consume_exact_behavior_decision_counters()
    assert second["ready_tilt_eligible_sample_count"].item() == 0
    assert second["ready_foot_contact_eligible_sample_count"].item() == 0


def test_exact_ready_planner_first_post_install_sample_has_nonzero_denominators_once():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()))
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        planner_revision_enabled=True,
        in_hold=torch.zeros(4, dtype=torch.bool),
        _planner_active=torch.ones(4, dtype=torch.bool),
    )
    command._event_timing_bound = True
    command._planner_control_epoch = torch.tensor([1, 1, 2, 2])
    command._planner_task_id = torch.tensor([1, 2, 1, 3])
    command._planner_task_revision = torch.ones(4, dtype=torch.long)
    command.metrics = {
        "base_upright": torch.cos(torch.tensor([0.1, 0.2, 0.3, 0.4])),
        "foot_contact_frac": torch.tensor([1.0, 0.5, 1.0, 0.5]),
        "foot_slip_speed": torch.tensor([0.0, 0.1, 0.2, 0.3]),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.2, 1.0], [0.3, 0.4, 1.0]]
            ),
            root_lin_vel_w=torch.tensor(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.3, 0.4, 0.0]]
            ),
        )
    )
    command.base_target_pos_w = torch.zeros(4, 2)
    command._foot_idx_robot = [0, 1]
    command._foot_idx_contact = [0, 1]
    command._contact_sensor = object()

    command._book_exact_ready_behavior_samples()
    first = command.consume_exact_behavior_decision_counters()
    assert first["ready_phase_sample_count"].item() == 4
    assert first["ready_planner_task_entry_sample_count"].item() == 4
    assert first["ready_planner_legacy_hold_violation_count"].item() == 0
    for key in (
        "ready_tilt_eligible_sample_count",
        "ready_base_speed_eligible_sample_count",
        "ready_station_offset_eligible_sample_count",
        "ready_foot_contact_eligible_sample_count",
        "ready_foot_slip_eligible_sample_count",
    ):
        assert first[key].item() == 4
    assert first["ready_foot_sensor_unavailable_sample_count"].item() == 0

    # A same-ball revision changes only task_revision and must not create another ready sample.
    command._planner_task_revision += 17
    command._book_exact_ready_behavior_samples()
    second = command.consume_exact_behavior_decision_counters()
    assert second["ready_phase_sample_count"].item() == 0
    assert second["ready_tilt_eligible_sample_count"].item() == 0

    # A same-episode next ball changes task_id and produces one new task-entry sample.
    command._planner_task_id[2] += 1
    command._book_exact_ready_behavior_samples()
    third = command.consume_exact_behavior_decision_counters()
    assert third["ready_phase_sample_count"].item() == 1
    assert third["ready_planner_task_entry_sample_count"].item() == 1
    assert third["ready_foot_contact_eligible_sample_count"].item() == 1

    # A partial vector-env true reset reuses task_id=1 under a new epoch.  Only the reset env is
    # new; unchanged neighbors must keep their one-shot latches.
    command._planner_control_epoch[1] += 1
    command._planner_task_id[1] = 1
    command._book_exact_ready_behavior_samples()
    fourth = command.consume_exact_behavior_decision_counters()
    assert fourth["ready_phase_sample_count"].item() == 1
    assert fourth["ready_planner_task_entry_sample_count"].item() == 1
    assert fourth["ready_tilt_eligible_sample_count"].item() == 1

    # An installed identity is not sampled while inactive and is not prematurely latched.  It is
    # sampled exactly once when the task becomes active on a later metrics tick.
    command._planner_task_id[3] += 1
    command._motion_term._planner_active[3] = False
    command._book_exact_ready_behavior_samples()
    inactive = command.consume_exact_behavior_decision_counters()
    assert inactive["ready_phase_sample_count"].item() == 0
    command._motion_term._planner_active[3] = True
    command._book_exact_ready_behavior_samples()
    activated = command.consume_exact_behavior_decision_counters()
    assert activated["ready_phase_sample_count"].item() == 1
    assert activated["ready_planner_task_entry_sample_count"].item() == 1


def test_exact_ready_planner_legacy_hold_is_violation_not_eligibility():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=3)
    command.planner_revision_enabled = True
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        planner_revision_enabled=True,
        in_hold=torch.tensor([True, True, False]),
        _planner_active=torch.tensor([True, False, True]),
    )
    command._event_timing_bound = True
    command._planner_control_epoch = torch.ones(3, dtype=torch.long)
    command._planner_task_id = torch.ones(3, dtype=torch.long)
    command.metrics = {
        "base_upright": torch.ones(3),
        "foot_contact_frac": torch.ones(3),
        "foot_slip_speed": torch.zeros(3),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor(
                [[0.0, 0.0, 1.0], [9.0, 9.0, 1.0], [0.1, 0.2, 1.0]]
            ),
            root_lin_vel_w=torch.zeros(3, 3),
        )
    )
    command.base_target_pos_w = torch.zeros(3, 2)
    command._foot_idx_robot = [0, 1]
    command._foot_idx_contact = [0, 1]
    command._contact_sensor = object()

    command._book_exact_ready_behavior_samples()
    first = command.consume_exact_behavior_decision_counters()
    # Both unexpected hold envs are witnessed, but only the two active new task identities enter
    # the ready denominator.  In particular, inactive env 1's hold cannot leak into the sums.
    assert first["ready_planner_legacy_hold_violation_count"].item() == 2
    assert first["ready_phase_sample_count"].item() == 2
    assert first["ready_planner_task_entry_sample_count"].item() == 2
    assert first["ready_station_offset_eligible_sample_count"].item() == 2
    assert first["ready_station_offset_m_sum"].item() == pytest.approx(
        math.sqrt(0.1**2 + 0.2**2)
    )

    # The same identities stay one-shot.  A persistent illegal hold remains visible as a fresh
    # violation each metrics tick but never creates another ready observation.
    command._book_exact_ready_behavior_samples()
    second = command.consume_exact_behavior_decision_counters()
    assert second["ready_planner_legacy_hold_violation_count"].item() == 2
    assert second["ready_phase_sample_count"].item() == 0
    assert second["ready_tilt_eligible_sample_count"].item() == 0


def test_exact_ready_missing_foot_sensor_is_explicitly_unavailable_not_zero_observation():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()), num_envs=2)
    command._motion_term = types.SimpleNamespace(
        event_timing_enabled=False,
        in_hold=torch.tensor([True, True]),
    )
    command._event_timing_bound = True
    command.metrics = {
        "base_upright": torch.ones(2),
        # Historical zero-filled metric buffers may exist even when no sensor resolved.  They must
        # not become eligible observations.
        "foot_contact_frac": torch.zeros(2),
        "foot_slip_speed": torch.zeros(2),
    }
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]]),
            root_lin_vel_w=torch.zeros(2, 3),
        )
    )
    command.base_target_pos_w = torch.zeros(2, 2)
    command._foot_idx_robot = []
    command._foot_idx_contact = []
    command._contact_sensor = None

    command._book_exact_ready_behavior_samples()
    snapshot = command.consume_exact_behavior_decision_counters()
    assert snapshot["ready_phase_sample_count"].item() == 2
    assert snapshot["ready_tilt_eligible_sample_count"].item() == 2
    assert snapshot["ready_foot_contact_eligible_sample_count"].item() == 0
    assert snapshot["ready_foot_slip_eligible_sample_count"].item() == 0
    assert snapshot["ready_foot_contact_fraction_sum"].item() == 0.0
    assert snapshot["ready_foot_slip_speed_mps_sum"].item() == 0.0
    assert snapshot["ready_foot_sensor_unavailable_sample_count"].item() == 2


def test_exact_planner_revision_activation_counters_book_and_reset_per_update():
    command = _exact_behavior_command(types.SimpleNamespace(active_terms=()))
    command.planner_revision_enabled = True
    command._planner_revision_profile = types.SimpleNamespace(
        min_tts_s=0.02, early_deadline_tolerance_s=1.0e-6
    )
    command._planner_initial_tts_mixture = None

    command._book_planner_revision_decisions(
        attempted_tts=torch.tensor([0.10, 0.02000001, 0.02, 0.08]),
        accepted=torch.tensor([True, True, False, False]),
    )
    first = command.consume_exact_behavior_decision_counters()
    assert first["planner_revision_attempt_count"].item() == 4
    assert first["planner_revision_accepted_count"].item() == 2
    assert first["planner_revision_rejected_count"].item() == 2
    assert first["planner_revision_last_precontact_attempt_count"].item() == 2
    assert first["planner_revision_last_precontact_accepted_count"].item() == 1

    second = command.consume_exact_behavior_decision_counters()
    for name in (
        "planner_revision_attempt_count",
        "planner_revision_accepted_count",
        "planner_revision_rejected_count",
        "planner_revision_last_precontact_attempt_count",
        "planner_revision_last_precontact_accepted_count",
        "planner_revision_actor_visible_count",
        "planner_revision_last_precontact_actor_visible_count",
    ):
        assert second[name].item() == 0


def test_training_governor_keeps_final_actor_interval_for_point_zero_two_revision():
    motion = commands_mod.MotionCommand.__new__(commands_mod.MotionCommand)
    motion.device = "cpu"
    motion.planner_revision_enabled = True
    motion._planner_revision_profile = planner_revision_mod.PhaseGovernorProfile(
        policy_dt_s=0.02,
        min_tts_s=0.02,
        max_tts_s=2.0,
        max_phase_rate_per_s=4.0,
        max_phase_acceleration_per_s2=20.0,
        max_deadline_revision_delta_s=0.25,
        max_position_revision_delta_m=0.10,
        max_velocity_revision_delta_mps=0.50,
        max_normal_revision_delta_rad=0.20,
        early_deadline_tolerance_s=1.0e-6,
    )
    motion.time_steps_f = torch.zeros(1)
    motion.speed_scale = torch.ones(1)
    motion.metrics = {
        "planner_revision_accepted": torch.zeros(1),
        "planner_revision_rejected": torch.zeros(1),
    }
    motion._planner_active = torch.zeros(1, dtype=torch.bool)
    motion._planner_control_epoch = torch.zeros(1, dtype=torch.long)
    motion._planner_task_id = torch.zeros(1, dtype=torch.long)
    motion._planner_task_revision = torch.full((1,), -1, dtype=torch.long)
    motion._planner_start_step = torch.zeros(1)
    motion._planner_strike_step = torch.zeros(1)
    motion._planner_phase_rate = torch.zeros(1)
    motion._planner_slow_only_next = torch.zeros(1, dtype=torch.bool)
    motion._planner_desired_tts = torch.zeros(1)
    motion._planner_begin_tts = torch.zeros(1)
    motion._planner_truth_tts = torch.zeros(1)
    # 带符号孪生时钟(2026-07-25 C1):真源码与 truth tts 一起写,fake 必须同步喂
    motion._planner_truth_tts_signed = torch.zeros(1)
    motion._planner_begin_target_pos = torch.zeros(1, 3)
    motion._planner_begin_target_vel = torch.zeros(1, 3)
    motion._planner_begin_target_normal = torch.zeros(1, 3)

    ids = torch.tensor([0], dtype=torch.long)
    position = torch.tensor([[0.2, -0.1, 0.9]])
    velocity = torch.tensor([[-2.0, 0.0, -0.2]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    motion.begin_planner_task(
        ids,
        control_epoch=torch.tensor([7]),
        task_id=torch.tensor([10]),
        strike_step=torch.tensor([1.0]),
        initial_tts=torch.tensor([0.5]),
        target_position=position,
        target_velocity=velocity,
        target_normal=normal,
    )

    accepted_tts = []
    revision = 1
    for tick in range(1, 25):
        delta = motion._advance_planner_phase(torch.tensor([False]))
        motion.time_steps_f += delta
        if tick >= 20:
            revision += 1
            # Exercise the actual float32 countdown path; 0.50 - 24*0.02 is
            # 0.0199999101 on the training grid, not the rounded decimal 0.02.
            tts = motion._planner_truth_tts[ids].clone()
            accepted = motion.submit_planner_revision(
                ids,
                control_epoch=torch.tensor([7]),
                task_id=torch.tensor([10]),
                task_revision=torch.tensor([revision]),
                desired_tts=tts,
                target_position=position,
                target_velocity=velocity,
                target_normal=normal,
            )
            assert accepted.tolist() == [True]
            accepted_tts.append(tts.item())

    assert accepted_tts == pytest.approx([0.10, 0.08, 0.06, 0.04, 0.02], abs=1.0e-6)
    assert torch.equal(
        motion._planner_desired_tts[ids], torch.full((1,), 0.02)
    )
    assert motion.time_steps_f.item() < 1.0

    delta = motion._advance_planner_phase(torch.tensor([False]))
    motion.time_steps_f += delta
    assert motion.time_steps_f.item() == pytest.approx(1.0)
    post_contact = motion.submit_planner_revision(
        ids,
        control_epoch=torch.tensor([7]),
        task_id=torch.tensor([10]),
        task_revision=torch.tensor([revision + 1]),
        desired_tts=torch.tensor([0.02]),
        target_position=position,
        target_velocity=velocity,
        target_normal=normal,
    )
    assert post_contact.tolist() == [False]


def test_planner_revision_metrics_remain_full_env_after_eligible_set_shrinks():
    """Compact revision decisions must remain safe for CommandTerm.reset global indexing."""

    num_envs = 4096
    motion = commands_mod.MotionCommand.__new__(commands_mod.MotionCommand)
    motion.num_envs = num_envs
    motion.device = "cpu"
    motion.planner_revision_enabled = True
    motion._planner_revision_profile = planner_revision_mod.PhaseGovernorProfile(
        policy_dt_s=0.02,
        min_tts_s=0.02,
        max_tts_s=2.0,
        max_phase_rate_per_s=4.0,
        max_phase_acceleration_per_s2=20.0,
        max_deadline_revision_delta_s=0.25,
        max_position_revision_delta_m=0.10,
        max_velocity_revision_delta_mps=0.50,
        max_normal_revision_delta_rad=0.20,
        early_deadline_tolerance_s=1.0e-6,
    )
    motion.time_steps_f = torch.zeros(num_envs)
    motion.speed_scale = torch.ones(num_envs)
    motion.metrics = {
        "planner_revision_accepted": torch.ones(num_envs),
        "planner_revision_rejected": torch.ones(num_envs),
        "planner_phase_rate_per_s": torch.zeros(num_envs),
        "planner_truth_tts_s": torch.zeros(num_envs),
    }
    motion._planner_active = torch.ones(num_envs, dtype=torch.bool)
    motion._planner_control_epoch = torch.full((num_envs,), 7, dtype=torch.long)
    motion._planner_task_id = torch.full((num_envs,), 10, dtype=torch.long)
    motion._planner_task_revision = torch.ones(num_envs, dtype=torch.long)
    motion._planner_start_step = torch.zeros(num_envs)
    motion._planner_strike_step = torch.full((num_envs,), 100.0)
    motion._planner_phase_rate = torch.zeros(num_envs)
    motion._planner_slow_only_next = torch.zeros(num_envs, dtype=torch.bool)
    motion._planner_desired_tts = torch.full((num_envs,), 0.50)
    motion._planner_begin_tts = torch.full((num_envs,), 0.50)
    motion._planner_truth_tts = torch.full((num_envs,), 0.50)
    # 带符号孪生时钟(2026-07-25 C1):真源码与 truth tts 一起写,fake 必须同步喂
    motion._planner_truth_tts_signed = torch.full((num_envs,), 0.50)
    motion._planner_begin_target_pos = torch.zeros(num_envs, 3)
    motion._planner_begin_target_vel = torch.zeros(num_envs, 3)
    motion._planner_begin_target_normal = torch.zeros(num_envs, 3)
    motion._planner_begin_target_normal[:, 0] = 1.0

    # The phase step clears stale decisions for every environment before the currently eligible
    # subset submits its compact revision rows.
    motion._advance_planner_phase(torch.zeros(num_envs, dtype=torch.bool))
    ids = torch.tensor([1, 4001], dtype=torch.long)
    accepted = motion.submit_planner_revision(
        ids,
        control_epoch=torch.tensor([7, 7]),
        task_id=torch.tensor([10, 999]),  # one accept, one deliberate reject
        task_revision=torch.tensor([2, 2]),
        desired_tts=torch.tensor([0.48, 0.48]),
        target_position=torch.zeros(2, 3),
        target_velocity=torch.zeros(2, 3),
        target_normal=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert torch.equal(accepted, torch.tensor([True, False]))
    assert motion.metrics["planner_revision_accepted"].shape == (num_envs,)
    assert motion.metrics["planner_revision_rejected"].shape == (num_envs,)
    assert motion.metrics["planner_revision_accepted"].sum().item() == 1
    assert motion.metrics["planner_revision_rejected"].sum().item() == 1
    assert motion.metrics["planner_revision_accepted"][1].item() == 1
    assert motion.metrics["planner_revision_rejected"][4001].item() == 1

    # Reproduce the reset pattern from the A4 failure: the last three reset positions carry high
    # global env ids. Every registered metric must remain safely gatherable by those ids.
    reset_env_ids = torch.tensor(list(range(68)) + [4001, 4050, 4095], dtype=torch.long)
    for metric in motion.metrics.values():
        assert metric.shape[0] == num_envs
        assert metric[reset_env_ids].shape[0] == len(reset_env_ids)


@pytest.mark.parametrize("num_segments", [1, 5, 6, 93])
def test_balanced_round_robin_sampler_is_exact_for_arbitrary_clip_counts(
    num_segments,
):
    order = tuple(f"clip-{index}" for index in range(num_segments))
    sampler = commands_mod._BalancedRoundRobinClipSampler(
        num_segments, seed=20260727, clip_order=order, device="cpu"
    )
    cumulative = torch.zeros(num_segments, dtype=torch.long)
    batches = (1, max(1, num_segments - 1), num_segments + 3, 2 * num_segments + 1)
    for batch_size in batches:
        sampled = sampler.sample(batch_size)
        local = torch.bincount(sampled, minlength=num_segments)
        cumulative += local
        assert int(local.max() - local.min()) <= 1
        assert int(cumulative.max() - cumulative.min()) <= 1


def test_balanced_round_robin_is_deterministic_across_batching_and_seeded_shuffle():
    order = tuple(f"clip-{index}" for index in range(93))
    one_batch = commands_mod._BalancedRoundRobinClipSampler(
        93, seed=17, clip_order=order, device="cpu"
    )
    split_batches = commands_mod._BalancedRoundRobinClipSampler(
        93, seed=17, clip_order=order, device="cpu"
    )
    sizes = (5, 1, 97, 6, 181)
    expected = one_batch.sample(sum(sizes))
    actual = torch.cat([split_batches.sample(size) for size in sizes])
    assert torch.equal(actual, expected)
    assert not torch.equal(expected[:93], torch.arange(93))

    repeated = commands_mod._BalancedRoundRobinClipSampler(
        93, seed=17, clip_order=order, device="cpu"
    )
    other_seed = commands_mod._BalancedRoundRobinClipSampler(
        93, seed=18, clip_order=order, device="cpu"
    )
    assert torch.equal(repeated.sample(200), expected[:200])
    assert not torch.equal(other_seed.sample(93), expected[:93])


def test_balanced_round_robin_state_round_trip_and_identity_checks():
    order = tuple(f"clip-{index}" for index in range(6))
    source = commands_mod._BalancedRoundRobinClipSampler(
        6, seed=42, clip_order=order, device="cpu"
    )
    source.sample(17)
    state = source.state_dict()

    resumed = commands_mod._BalancedRoundRobinClipSampler(
        6, seed=42, clip_order=order, device="cpu"
    )
    resumed.load_state_dict(state)
    assert torch.equal(source.sample(250), resumed.sample(250))

    wrong_count = commands_mod._BalancedRoundRobinClipSampler(
        5, seed=42, clip_order=order[:5], device="cpu"
    )
    with pytest.raises(ValueError, match="num_segments"):
        wrong_count.load_state_dict(state)
    wrong_order = commands_mod._BalancedRoundRobinClipSampler(
        6, seed=42, clip_order=tuple(reversed(order)), device="cpu"
    )
    with pytest.raises(ValueError, match="clip_order"):
        wrong_order.load_state_dict(state)
    wrong_seed = commands_mod._BalancedRoundRobinClipSampler(
        6, seed=43, clip_order=order, device="cpu"
    )
    with pytest.raises(ValueError, match="seed"):
        wrong_seed.load_state_dict(state)


def test_motion_command_balanced_sampling_assigns_shuffled_env_ids_and_restores(clips):
    motion_files = [clips[index % len(clips)] for index in range(6)]
    cmd, _ = _make_motion_command(
        motion_files,
        num_envs=8,
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=123,
    )
    first_ids = torch.tensor([7, 2, 5, 0, 6])
    first_expected = cmd._balanced_clip_sampler.permutation[: len(first_ids)].clone()
    cmd._adaptive_sampling(first_ids)
    assert torch.equal(cmd.clip_id[first_ids], first_expected)

    state = cmd.balanced_clip_sampler_state_dict()
    assert state["cursor"] == len(first_ids)
    resumed, _ = _make_motion_command(
        motion_files,
        num_envs=8,
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=123,
    )
    resumed.load_balanced_clip_sampler_state_dict(state)
    second_ids = torch.tensor([4, 1, 3])
    cmd._adaptive_sampling(second_ids)
    resumed._adaptive_sampling(second_ids)
    assert torch.equal(cmd.clip_id[second_ids], resumed.clip_id[second_ids])
    assert (
        cmd.balanced_clip_sampler_state_dict()
        == resumed.balanced_clip_sampler_state_dict()
    )


def test_motion_command_balanced_mode_does_not_consume_global_rng(clips):
    cmd, _ = _make_motion_command(
        [clips[0], clips[1]],
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=7,
    )
    torch.manual_seed(991)
    expected_next = torch.rand(8)
    torch.manual_seed(991)
    cmd._adaptive_sampling(torch.arange(cmd.num_envs))
    assert torch.equal(torch.rand(8), expected_next)


def test_motion_command_default_keeps_legacy_torch_randint_path(clips):
    cmd, _ = _make_motion_command([clips[0], clips[1]])
    assert cmd.balanced_clip_sampler_state_dict() is None
    torch.manual_seed(117)
    expected = torch.randint(
        0, cmd.motion.num_segments, (cmd.num_envs,), device=cmd.device
    )
    torch.manual_seed(117)
    cmd._adaptive_sampling(torch.arange(cmd.num_envs))
    assert torch.equal(cmd.clip_id, expected)


def test_motion_command_exact_resume_schema_includes_disabled_sampler(clips):
    cmd, _ = _make_motion_command([clips[0], clips[1]])
    state = cmd.exact_resume_state_dict()
    assert state["state_kind"] == "whole_body_tracking.MotionCommand"
    assert state["schema_version"] == 2
    assert state["balanced_clip_sampler"] is None
    assert set(state["adaptive_sampling"]) == {
        "bin_failed_count",
        "current_bin_failed",
    }
    assert state["adaptive_sampling"]["bin_failed_count"].device.type == "cpu"
    assert state["post_swing_replay"] == {
        "root": None,
        "joint_pos": None,
        "joint_vel": None,
        "ptr": 0,
        "count": 0,
        "first_reset_checked": False,
    }
    cmd.load_exact_resume_state_dict(state)
    with pytest.raises(ValueError, match="strict=True"):
        cmd.load_exact_resume_state_dict(state, strict=False)


def test_motion_command_exact_resume_round_trip_and_identity_fail_loud(clips):
    motion_files = [clips[index % len(clips)] for index in range(6)]
    source, _ = _make_motion_command(
        motion_files,
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=55,
    )
    source._adaptive_sampling(torch.arange(5))
    state = source.exact_resume_state_dict()

    resumed, _ = _make_motion_command(
        motion_files,
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=55,
    )
    resumed.load_exact_resume_state_dict(state)
    source._adaptive_sampling(torch.arange(source.num_envs))
    resumed._adaptive_sampling(torch.arange(resumed.num_envs))
    assert torch.equal(source.clip_id, resumed.clip_id)

    wrong_order = copy.deepcopy(state)
    sampler_state = wrong_order["balanced_clip_sampler"]
    sampler_state["clip_order"] = tuple(reversed(sampler_state["clip_order"]))
    with pytest.raises(ValueError, match="clip_order"):
        resumed.load_exact_resume_state_dict(wrong_order)

    wrong_identity = copy.deepcopy(state)
    wrong_identity["identity"]["motion"]["clip_order"] = tuple(
        reversed(wrong_identity["identity"]["motion"]["clip_order"])
    )
    with pytest.raises(ValueError, match="identity"):
        resumed.load_exact_resume_state_dict(wrong_identity)


def test_motion_command_exact_resume_preserves_all_legacy_curriculum_state(clips):
    cfg = {
        "post_swing_start_prob": 0.25,
        "post_swing_buffer_size": 4,
        "post_swing_min_fill": 2,
    }
    source, source_robot = _make_motion_command([clips[2]], **cfg)
    source.bin_failed_count.copy_(
        torch.linspace(0.25, 1.25, source.bin_count)
    )
    source._current_bin_failed.copy_(
        torch.arange(source.bin_count, dtype=source._current_bin_failed.dtype)
    )
    size = source.cfg.post_swing_buffer_size
    joint_count = source.robot.data.joint_pos.shape[-1]
    source._post_swing_root = torch.arange(
        size * 13, dtype=source.robot.data.root_state_w.dtype
    ).reshape(size, 13)
    source._post_swing_joint_pos = torch.arange(
        size * joint_count, dtype=source.robot.data.joint_pos.dtype
    ).reshape(size, joint_count)
    source._post_swing_joint_vel = -source._post_swing_joint_pos.clone()
    source._post_swing_ptr = 3
    source._post_swing_count = 3
    source._post_swing_first_reset_checked = True
    state = source.exact_resume_state_dict()

    resumed, resumed_robot = _make_motion_command([clips[2]], **cfg)
    resumed.load_exact_resume_state_dict(state)
    assert torch.equal(resumed.bin_failed_count, source.bin_failed_count)
    assert torch.equal(resumed._current_bin_failed, source._current_bin_failed)
    assert torch.equal(resumed._post_swing_root, source._post_swing_root)
    assert torch.equal(resumed._post_swing_joint_pos, source._post_swing_joint_pos)
    assert torch.equal(resumed._post_swing_joint_vel, source._post_swing_joint_vel)
    assert resumed._post_swing_ptr == 3
    assert resumed._post_swing_count == 3
    assert resumed._post_swing_first_reset_checked is True
    assert resumed._post_swing_root.device == source._post_swing_root.device

    # The restored adaptive sampler makes the same next-rollout draw.
    ids = torch.arange(source.num_envs)
    torch.manual_seed(90210)
    source._adaptive_sampling(ids)
    torch.manual_seed(90210)
    resumed._adaptive_sampling(ids)
    assert torch.equal(resumed.time_steps, source.time_steps)

    # The restored replay population selects the same physical reset state.
    replay_ids = torch.tensor([0, 3, 7])
    torch.manual_seed(77)
    source._write_post_swing_states(replay_ids)
    torch.manual_seed(77)
    resumed._write_post_swing_states(replay_ids)
    assert torch.equal(
        source_robot.data.root_state_w[replay_ids],
        resumed_robot.data.root_state_w[replay_ids],
    )
    assert torch.equal(
        source_robot.data.joint_pos[replay_ids],
        resumed_robot.data.joint_pos[replay_ids],
    )


def test_motion_command_exact_resume_state_is_detached_cpu_snapshot(clips):
    cmd, _ = _make_motion_command(
        [clips[2]],
        post_swing_start_prob=0.25,
        post_swing_buffer_size=4,
        post_swing_min_fill=2,
    )
    cmd.bin_failed_count.fill_(2.0)
    cmd._post_swing_root = torch.ones(4, 13)
    cmd._post_swing_joint_pos = torch.ones(4, _N_JOINTS)
    cmd._post_swing_joint_vel = torch.ones(4, _N_JOINTS)
    state = cmd.exact_resume_state_dict()
    cmd.bin_failed_count.zero_()
    cmd._post_swing_root.zero_()
    assert torch.all(state["adaptive_sampling"]["bin_failed_count"] == 2.0)
    assert torch.all(state["post_swing_replay"]["root"] == 1.0)
    for section in ("adaptive_sampling", "post_swing_replay"):
        for value in state[section].values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"


def test_motion_command_exact_resume_rejects_config_shape_device_and_ring_drift(clips):
    cfg = {
        "post_swing_start_prob": 0.25,
        "post_swing_buffer_size": 4,
        "post_swing_min_fill": 2,
    }
    source, _ = _make_motion_command([clips[2]], **cfg)
    state = source.exact_resume_state_dict()

    different_cfg, _ = _make_motion_command(
        [clips[2]], **dict(cfg, adaptive_alpha=0.5)
    )
    with pytest.raises(ValueError, match="identity"):
        different_cfg.load_exact_resume_state_dict(state)

    bad_shape = copy.deepcopy(state)
    bad_shape["adaptive_sampling"]["bin_failed_count"] = torch.zeros(
        source.bin_count + 1
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        source.load_exact_resume_state_dict(bad_shape)

    bad_device = copy.deepcopy(state)
    bad_device["adaptive_sampling"]["bin_failed_count"] = torch.empty(
        source.bin_count, device="meta"
    )
    with pytest.raises(ValueError, match="serialized on the CPU"):
        source.load_exact_resume_state_dict(bad_device)

    partial_ring = copy.deepcopy(state)
    partial_ring["post_swing_replay"]["root"] = torch.zeros(4, 13)
    with pytest.raises(ValueError, match="partially serialized"):
        source.load_exact_resume_state_dict(partial_ring)

    bad_cursor = copy.deepcopy(state)
    bad_cursor["post_swing_replay"]["ptr"] = 4
    with pytest.raises(ValueError, match="outside the configured ring"):
        source.load_exact_resume_state_dict(bad_cursor)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda state: state.pop("balanced_clip_sampler"), "keys"),
        (lambda state: state.update(extra=True), "keys"),
        (lambda state: state.update(state_kind="other"), "state_kind"),
        (lambda state: state.update(schema_version=3), "schema_version"),
    ],
)
def test_motion_command_exact_resume_rejects_schema_drift(clips, mutation, match):
    cmd, _ = _make_motion_command([clips[0], clips[1]])
    state = cmd.exact_resume_state_dict()
    mutation(state)
    with pytest.raises(ValueError, match=match):
        cmd.load_exact_resume_state_dict(state)


# --------------------------------------------------------------------------------------------- #
# 起点扰动斜坡:MotionCommand 必须读 utils/training_contract.py 的那一条法则,
# 而不是自己在这里再实现一遍插值。缺席时逐字节等于旧路径。
# --------------------------------------------------------------------------------------------- #
_START_POSE_RAMP_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _start_pose_ramp_contract():
    return commands_mod.MotionCommand._training_contract_module()


def test_motion_command_binds_the_repository_start_pose_ramp_law():
    module = _start_pose_ramp_contract()
    expected = os.path.abspath(
        os.path.join(
            MDP_DIR, "..", "..", "..", "utils", "training_contract.py"
        )
    )
    assert os.path.realpath(module.__file__) == os.path.realpath(expected)
    # 默认缺席 = 旧路径
    assert commands_mod.MotionCommandCfg.start_pose_ramp is None
    # 再取一次必须是同一个已缓存对象(不允许每次复位重新执行仓库字节)
    assert _start_pose_ramp_contract() is module


def _ramp_stub(ramp, *, pose_seed=None, hold=(0, 0)):
    zero = {axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES}
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            pose_range=dict(zero if pose_seed is None else pose_seed),
            velocity_range=dict(zero),
            joint_position_range=(0.0, 0.0),
            hold_steps_range=hold,
        ),
        _start_pose_ramp=ramp,
        _start_pose_ramp_enabled=bool(ramp["enabled"]),
        _training_contract_module=(
            commands_mod.MotionCommand._training_contract_module
        ),
    )


def test_start_pose_ramp_effective_ranges_walk_seed_to_declared_endpoint():
    contract = _start_pose_ramp_contract()
    ramp = contract.validate_action_ball_start_pose_ramp(
        contract.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="four_cell"
    )
    stub = _ramp_stub(ramp)
    at_zero = commands_mod.MotionCommand._effective_reset_range_list(
        stub, "pose_range", 0.0
    )
    assert at_zero == [(0.0, 0.0)] * 6
    at_one = commands_mod.MotionCommand._effective_reset_range_list(
        stub, "pose_range", 1.0
    )
    assert [list(pair) for pair in at_one] == [
        ramp["pose_range"][axis] for axis in _START_POSE_RAMP_AXES
    ]
    half = commands_mod.MotionCommand._effective_reset_range_list(
        stub, "pose_range", 0.5
    )
    assert half[1] == pytest.approx((-1.2625 / 2.0, 1.2625 / 2.0))
    # 速度终点是零,所以任何进度上都不许冒出出生初速度
    for progress in (0.0, 0.37, 1.0):
        assert commands_mod.MotionCommand._effective_reset_range_list(
            stub, "velocity_range", progress
        ) == [(0.0, 0.0)] * 6
    # ActionBall 的等待归收据所有,hold 窗口在任何进度上都保持零
    assert commands_mod.MotionCommand._effective_hold_steps_range(stub, 1.0) == (0, 0)


def test_start_pose_ramp_absent_returns_the_static_config_unchanged():
    contract = _start_pose_ramp_contract()
    disabled = contract.validate_action_ball_start_pose_ramp(None, name="absent")
    seed = {axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES}
    seed["y"] = (-0.3, 0.3)
    stub = _ramp_stub(disabled, pose_seed=seed, hold=(4, 9))
    for progress in (0.0, 0.5, 1.0):
        effective = commands_mod.MotionCommand._effective_reset_range_list(
            stub, "pose_range", progress
        )
        assert effective[1] == (-0.3, 0.3)
        assert commands_mod.MotionCommand._effective_hold_steps_range(
            stub, progress
        ) == (4, 9)
        assert commands_mod.MotionCommand._effective_joint_position_range(
            stub, progress
        ) == (0.0, 0.0)
    assert commands_mod.MotionCommand.start_pose_ramp_progress(stub) == 0.0


def test_start_pose_ramp_progress_requires_an_integer_step_counter():
    contract = _start_pose_ramp_contract()
    ramp = contract.validate_action_ball_start_pose_ramp(
        contract.ACTION_BALL_START_POSE_RAMP_FOUR_CELL, name="four_cell"
    )
    stub = _ramp_stub(ramp)
    stub._env = types.SimpleNamespace(common_step_counter=48000)
    assert commands_mod.MotionCommand.start_pose_ramp_progress(stub) == 0.5
    stub._env = types.SimpleNamespace(common_step_counter=None)
    with pytest.raises(RuntimeError, match="common_step_counter"):
        commands_mod.MotionCommand.start_pose_ramp_progress(stub)
    stub._env = types.SimpleNamespace(common_step_counter=-1)
    with pytest.raises(RuntimeError, match="common_step_counter"):
        commands_mod.MotionCommand.start_pose_ramp_progress(stub)


def _canonical_ready_guard_stub(*, pose_range, ramp_enabled):
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            stand_start_prob=1.0,
            post_swing_start_prob=0.0,
            post_swing_teacher_receipt="",
            post_swing_require_ready_at_init=False,
            post_swing_fail_fast_first_reset=False,
            post_swing_first_reset_require_readback=False,
            wrap_teleport=False,
            clip_switch_prob=0.0,
            event_timing_mode="disabled",
            rsi_skip_settle_frames=0,
            joint_position_range=(0.0, 0.0),
            stand_start_yaw_range=(0.0, 0.0),
            pose_range=pose_range,
            velocity_range={axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES},
        ),
        _start_pose_ramp_enabled=ramp_enabled,
        _range_is_exact_zero_pair=(
            commands_mod.MotionCommand._range_is_exact_zero_pair
        ),
        _mapping_ranges_are_exact_zero=(
            commands_mod.MotionCommand._mapping_ranges_are_exact_zero
        ),
    )


def test_canonical_ready_guard_still_refuses_reset_noise_without_a_ramp():
    """Softening the guard must not remove it when no ramp is declared.

    ``_validate_canonical_ready_config`` is the runtime twin of train.py's
    hard gate.  Without a declared ramp its old all-zero clause has to stand
    exactly as before; with one, the seed bound is enforced separately by
    ``_configure_start_pose_ramp``.
    """

    noisy = {axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES}
    noisy["x"] = (-0.1, 0.1)
    with pytest.raises(ValueError, match=r"pose_range entries must be \(0, 0\)"):
        commands_mod.MotionCommand._validate_canonical_ready_config(
            _canonical_ready_guard_stub(pose_range=noisy, ramp_enabled=False)
        )
    # 声明了斜坡之后同一份静态种子放行(边界由 _configure_start_pose_ramp 管)
    commands_mod.MotionCommand._validate_canonical_ready_config(
        _canonical_ready_guard_stub(pose_range=noisy, ramp_enabled=True)
    )
    # 全零 + 无斜坡仍然通过(旧行为)
    commands_mod.MotionCommand._validate_canonical_ready_config(
        _canonical_ready_guard_stub(
            pose_range={axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES},
            ramp_enabled=False,
        )
    )
    # 斜坡不接管 stand_start_yaw_range:出生朝向只有 pose_range.yaw 一个来源
    stub = _canonical_ready_guard_stub(
        pose_range={axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES},
        ramp_enabled=True,
    )
    stub.cfg.stand_start_yaw_range = (-0.2, 0.2)
    with pytest.raises(ValueError, match="stand_start_yaw_range"):
        commands_mod.MotionCommand._validate_canonical_ready_config(stub)


def test_start_pose_ramp_configure_refuses_a_seed_outside_the_endpoint():
    contract = _start_pose_ramp_contract()
    spec = copy.deepcopy(dict(contract.ACTION_BALL_START_POSE_RAMP_FOUR_CELL))
    seed = {axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES}
    seed["y"] = (-2.0, 2.0)  # 超出声明的 +/-1.2625 终点
    stub = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            start_pose_ramp=spec,
            pose_range=seed,
            velocity_range={axis: (0.0, 0.0) for axis in _START_POSE_RAMP_AXES},
            joint_position_range=(0.0, 0.0),
        ),
        _training_contract_module=(
            commands_mod.MotionCommand._training_contract_module
        ),
    )
    with pytest.raises(ValueError, match=r"\[0, endpoint\]"):
        commands_mod.MotionCommand._configure_start_pose_ramp(stub)
    # 种子落在终点内则接受,并把规范化后的 payload 绑上
    seed["y"] = (-0.5, 0.5)
    commands_mod.MotionCommand._configure_start_pose_ramp(stub)
    assert stub._start_pose_ramp_enabled is True
    assert stub._start_pose_ramp["kind"] == (
        contract.ACTION_BALL_START_POSE_RAMP_KIND
    )
    assert len(stub._start_pose_ramp_sha256) == 64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
