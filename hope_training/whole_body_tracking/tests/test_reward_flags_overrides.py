"""reward-flags-0709 — train.py override-translation unit tests (NO Isaac imports).

Covers the eight default-OFF flag sets of docs/research/reward_staged_design_2026-07-08.md plus
the new task.rewards fail-loud whitelist, at the TRANSLATION layer (_apply_task_overrides):

* task.rewards unknown-key fail-loud (_REWARD_KEYS) + regression that every rewards/motion/racket/
  terminations key in the real cfg/task/*.yaml files is whitelisted (so the new check cannot brick
  an existing task).
* V1  rewards.free_wrist_vel_mimic       — wrist dropped from motion_body_lin_vel only.
* V2  rewards.motion_scale_in_window     — window_scale params wired onto the (non-None) motion terms.
* 1c  racket.strike_window_pos_s/wide_s  — split-window cfg fields set.
* C2a rewards.face_gate_by_pos+radius    — pos_gate_radius param on racket_velocity/normal;
                                           half-configured gate raises either way.
* B2  rewards.racket_guidance_weight     — weight set; positive weight raises.
* R-a task.actor_leg_ref_mask            — actor command obs func swapped; critic untouched.
* R-c motion.rsi_skip_settle_frames / rsi_hold_root_stand_z — MotionCommandCfg fields set.
* R-b terminations.envelope_as_penalty(+envelope_penalty_weight) — anchor_pos/ee_body_pos removed,
      tracking_envelope weight set (default -1.0), track_envelope_violation switched on;
      weight-without-flag and weight>=0 raise; absolute terminations untouched.
* default path: a DeployParity-like task node without any new key leaves every new attr untouched.

train.py is imported directly (its top-level imports are hydra/omegaconf only — no Isaac); the
env cfg is a plain-namespace fake, exactly the level _apply_task_overrides operates on.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py -q
"""

from __future__ import annotations

import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
CFG_TASK_DIR = os.path.abspath(os.path.join(HERE, "..", "cfg", "task"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import train as train_mod  # noqa: E402  (hydra/omegaconf only at import time)


# --------------------------------------------------------------------------------------------- #
# fakes: the minimal env-cfg surface _apply_task_overrides touches
# --------------------------------------------------------------------------------------------- #
class _Term:
    """RewTerm/ObsTerm/DoneTerm stand-in: weight + params dict + func slot."""

    def __init__(self, weight=1.0, params=None, func="orig_func", body_names=None):
        self.weight = weight
        self.params = dict(params) if params is not None else {}
        if body_names is not None:
            self.params["body_names"] = list(body_names)
        self.func = func


class _NS(types.SimpleNamespace):
    pass


_WRIST = "right_wrist_yaw_Link"
_UPPER = ["torso_Link", "right_elbow_Link", _WRIST]


def _make_env_cfg(anchor_pos_none=True):
    """DeployParity-shaped fake env cfg (motion_global_anchor_pos removed, like the real cfg)."""
    rewards = _NS(
        racket_position=_Term(weight=14.0, params={"std": 0.2}),
        racket_velocity=_Term(weight=10.0, params={"std": 1.0}),
        racket_normal=_Term(weight=5.0, params={"std": 0.30}),
        racket_guidance=_Term(weight=0.0, params={"command_name": "racket_target", "d_max": 0.5}),
        tracking_envelope=_Term(weight=0.0, params={"command_name": "motion", "threshold": 0.25}),
        motion_global_anchor_pos=None if anchor_pos_none else _Term(weight=0.5, params={"std": 0.3}),
        motion_global_anchor_ori=_Term(weight=0.5, params={"std": 0.4}),
        motion_body_pos=_Term(weight=1.0, params={"std": 0.3}, body_names=_UPPER),
        motion_body_ori=_Term(weight=1.0, params={"std": 0.4}, body_names=_UPPER),
        motion_body_lin_vel=_Term(weight=1.0, params={"std": 1.0}, body_names=_UPPER),
        motion_body_ang_vel=_Term(weight=1.0, params={"std": 3.14}, body_names=_UPPER),
        action_rate_l2=_Term(weight=-0.1),
        joint_torques=_Term(weight=-3e-5),
    )
    racket_target = _NS(
        strike_phase=0.46,
        strike_window_s=0.12,
        strike_window_pos_s=None,
        strike_window_wide_s=None,
        strike_success_pos_thresh=0.075,
        track_envelope_violation=False,
    )
    motion = _NS(
        wrap_teleport=False,
        stand_start_prob=0.25,
        hold_steps_range=(0, 100),
        stand_start_min_hold=25,
        post_swing_start_prob=0.0,
        post_swing_buffer_size=4096,
        post_swing_min_fill=256,
        post_swing_min_hold=25,
        clip_switch_prob=0.0,
        speed_scale_range=(1.0, 1.0),
        speed_scale_per_clip=None,
        rsi_skip_settle_frames=0,
        rsi_hold_root_stand_z=False,
    )
    observations = _NS(
        policy=_NS(command=_Term(func="generated_commands", params={"command_name": "motion"})),
        critic=_NS(command=_Term(func="generated_commands", params={"command_name": "motion"})),
    )
    terminations = _NS(
        time_out="TIME_OUT",
        anchor_pos="ANCHOR_POS_TERM",
        anchor_ori="ANCHOR_ORI_TERM",
        ee_body_pos="EE_BODY_POS_TERM",
        base_fell_tilt="BASE_FELL_TILT_TERM",
        base_too_low="BASE_TOO_LOW_TERM",
    )
    return _NS(
        rewards=rewards,
        commands=_NS(motion=motion, racket_target=racket_target),
        observations=observations,
        terminations=terminations,
        scene=_NS(env_spacing=2.5),
        episode_length_s=10.0,
        sim=_NS(dt=0.005),
        decimation=4,
        actions=_NS(joint_pos=_NS(clamp=False)),
    )


def _apply(task, env_cfg=None):
    env_cfg = env_cfg if env_cfg is not None else _make_env_cfg()
    applied = train_mod._apply_task_overrides(env_cfg, task, clip_name=None)
    return env_cfg, applied


# --------------------------------------------------------------------------------------------- #
# fail-loud whitelist
# --------------------------------------------------------------------------------------------- #
def test_empty_task_applies_nothing():
    env_cfg, applied = _apply({})
    assert applied == []
    assert env_cfg.rewards.racket_guidance.weight == 0.0
    assert env_cfg.commands.racket_target.strike_window_pos_s is None


def test_unknown_rewards_key_fails_loud():
    with pytest.raises(train_mod._OverrideError, match="face_gate_radiuss"):
        _apply({"rewards": {"face_gate_radiuss": 0.15}})  # typo'd key: must raise, never no-op


def test_all_real_task_yaml_keys_are_whitelisted():
    """Regression: the new task.rewards/_TERMINATION_KEYS whitelists must accept every key the
    real task YAMLs already set (otherwise the fail-loud check bricks existing tasks)."""
    yaml = pytest.importorskip("yaml")
    checked = 0
    for fn in sorted(os.listdir(CFG_TASK_DIR)):
        if not fn.endswith(".yaml"):
            continue
        with open(os.path.join(CFG_TASK_DIR, fn)) as fh:
            doc = yaml.safe_load(fh) or {}
        for node_key, whitelist in (
            ("rewards", train_mod._REWARD_KEYS),
            ("motion", train_mod._MOTION_KEYS),
            ("racket", train_mod._RACKET_KEYS),
            ("terminations", train_mod._TERMINATION_KEYS),
        ):
            node = doc.get(node_key)
            if node is None:
                continue
            unknown = sorted(str(k) for k in node.keys() if str(k) not in whitelist)
            assert not unknown, f"{fn}: {node_key} keys {unknown} missing from the whitelist"
            checked += 1
    assert checked >= 4  # the four tasks with rewards blocks at minimum


# --------------------------------------------------------------------------------------------- #
# V1 free_wrist_vel_mimic
# --------------------------------------------------------------------------------------------- #
def test_v1_free_wrist_vel_mimic_drops_wrist_from_lin_vel_only():
    env_cfg, applied = _apply({"rewards": {"free_wrist_vel_mimic": True}})
    assert _WRIST not in env_cfg.rewards.motion_body_lin_vel.params["body_names"]
    # the orientation/ang-vel mimic lists are free_wrist_ori_mimic's business — untouched here
    assert _WRIST in env_cfg.rewards.motion_body_ori.params["body_names"]
    assert _WRIST in env_cfg.rewards.motion_body_ang_vel.params["body_names"]
    assert _WRIST in env_cfg.rewards.motion_body_pos.params["body_names"]
    assert any("motion_body_lin_vel.body_names" in a for a in applied)


def test_v1_false_is_noop():
    env_cfg, applied = _apply({"rewards": {"free_wrist_vel_mimic": False}})
    assert _WRIST in env_cfg.rewards.motion_body_lin_vel.params["body_names"]
    assert applied == []


def test_v1_requires_explicit_body_list():
    env_cfg = _make_env_cfg()
    del env_cfg.rewards.motion_body_lin_vel.params["body_names"]
    with pytest.raises(train_mod._OverrideError):
        _apply({"rewards": {"free_wrist_vel_mimic": True}}, env_cfg)


# --------------------------------------------------------------------------------------------- #
# V2 motion_scale_in_window
# --------------------------------------------------------------------------------------------- #
def test_v2_wires_window_scale_onto_non_none_motion_terms():
    env_cfg, applied = _apply({"rewards": {"motion_scale_in_window": 0.25}})
    for name in ("motion_global_anchor_ori", "motion_body_pos", "motion_body_ori",
                 "motion_body_lin_vel", "motion_body_ang_vel"):
        term = getattr(env_cfg.rewards, name)
        assert term.params["window_scale"] == 0.25, name
        assert term.params["window_command_name"] == "racket_target", name
    assert env_cfg.rewards.motion_global_anchor_pos is None  # removed term skipped, no crash
    assert any("motion_scale_in_window=0.25" in a for a in applied)
    # weights themselves untouched (the scaling is per-env inside the reward funcs)
    assert env_cfg.rewards.motion_body_lin_vel.weight == 1.0


def test_v2_absent_leaves_params_untouched():
    env_cfg, _ = _apply({"rewards": {"motion_scale": 1.0}})
    assert "window_scale" not in env_cfg.rewards.motion_body_lin_vel.params


def test_v2_requires_racket_command():
    env_cfg = _make_env_cfg()
    env_cfg.commands = _NS(motion=env_cfg.commands.motion)  # no racket_target
    with pytest.raises(train_mod._OverrideError):
        _apply({"rewards": {"motion_scale_in_window": 0.25}}, env_cfg)


# --------------------------------------------------------------------------------------------- #
# 1c split strike windows
# --------------------------------------------------------------------------------------------- #
def test_1c_split_windows_set_cfg_fields():
    env_cfg, applied = _apply(
        {"racket": {"strike_window_pos_s": 0.02, "strike_window_wide_s": 0.10}})
    C = env_cfg.commands.racket_target
    assert C.strike_window_pos_s == 0.02
    assert C.strike_window_wide_s == 0.10
    assert C.strike_window_s == 0.12  # the base window stays the yaml value
    assert any("strike_window_pos_s=0.02" in a for a in applied)
    assert any("strike_window_wide_s=0.1" in a for a in applied)


# --------------------------------------------------------------------------------------------- #
# C2a proximity power-gate
# --------------------------------------------------------------------------------------------- #
def test_face_gate_sets_radius_param_on_vel_and_normal_only():
    env_cfg, applied = _apply({"rewards": {"face_gate_by_pos": True, "face_gate_radius": 0.15}})
    assert env_cfg.rewards.racket_velocity.params["pos_gate_radius"] == 0.15
    assert env_cfg.rewards.racket_normal.params["pos_gate_radius"] == 0.15
    assert "pos_gate_radius" not in env_cfg.rewards.racket_position.params
    assert any("face_gate_by_pos=true" in a for a in applied)


def test_face_gate_half_configured_raises_both_ways():
    with pytest.raises(train_mod._OverrideError, match="face_gate_radius"):
        _apply({"rewards": {"face_gate_by_pos": True}})
    with pytest.raises(train_mod._OverrideError, match="face_gate_by_pos"):
        _apply({"rewards": {"face_gate_radius": 0.15}})


# --------------------------------------------------------------------------------------------- #
# B2 constant guidance
# --------------------------------------------------------------------------------------------- #
def test_guidance_weight_set_and_sign_checked():
    env_cfg, applied = _apply({"rewards": {"racket_guidance_weight": -0.75}})
    assert env_cfg.rewards.racket_guidance.weight == -0.75
    assert any("racket_guidance.weight=-0.75" in a for a in applied)
    with pytest.raises(train_mod._OverrideError, match="must be <= 0"):
        _apply({"rewards": {"racket_guidance_weight": 0.5}})


# --------------------------------------------------------------------------------------------- #
# R-a actor leg-reference mask
# --------------------------------------------------------------------------------------------- #
def _stub_mdp_module(monkeypatch):
    """train.py lazily imports whole_body_tracking...mdp inside the R-a branch — stub the chain."""
    fake_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    fake_mdp.generated_commands_actor_leg_masked = "LEG_MASKED_FUNC"
    fake_tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    fake_tracking.mdp = fake_mdp
    fake_tasks = types.ModuleType("whole_body_tracking.tasks")
    fake_tasks.tracking = fake_tracking
    fake_root = types.ModuleType("whole_body_tracking")
    fake_root.tasks = fake_tasks
    for name, m in (
        ("whole_body_tracking", fake_root),
        ("whole_body_tracking.tasks", fake_tasks),
        ("whole_body_tracking.tasks.tracking", fake_tracking),
        ("whole_body_tracking.tasks.tracking.mdp", fake_mdp),
    ):
        monkeypatch.setitem(sys.modules, name, m)
    return fake_mdp


def test_ra_swaps_actor_command_func_only(monkeypatch):
    fake_mdp = _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply({"actor_leg_ref_mask": True})
    assert env_cfg.observations.policy.command.func is fake_mdp.generated_commands_actor_leg_masked
    assert env_cfg.observations.critic.command.func == "generated_commands"  # critic untouched
    assert any("generated_commands_actor_leg_masked" in a for a in applied)


def test_ra_false_is_noop(monkeypatch):
    _stub_mdp_module(monkeypatch)
    env_cfg, applied = _apply({"actor_leg_ref_mask": False})
    assert env_cfg.observations.policy.command.func == "generated_commands"
    assert applied == []


# --------------------------------------------------------------------------------------------- #
# R-c RSI birth fixes
# --------------------------------------------------------------------------------------------- #
def test_rc_motion_flags_set():
    env_cfg, applied = _apply(
        {"motion": {"rsi_skip_settle_frames": 6, "rsi_hold_root_stand_z": True}})
    M = env_cfg.commands.motion
    assert M.rsi_skip_settle_frames == 6 and isinstance(M.rsi_skip_settle_frames, int)
    assert M.rsi_hold_root_stand_z is True
    assert any("rsi_skip_settle_frames=6" in a for a in applied)
    assert any("rsi_hold_root_stand_z=True" in a for a in applied)


def test_rc_absent_keeps_defaults():
    env_cfg, _ = _apply({"motion": {"hold_steps_range": [0, 100]}})
    assert env_cfg.commands.motion.rsi_skip_settle_frames == 0
    assert env_cfg.commands.motion.rsi_hold_root_stand_z is False


# --------------------------------------------------------------------------------------------- #
# R-b envelope-as-penalty
# --------------------------------------------------------------------------------------------- #
def test_rb_softens_envelope_and_wires_penalty_and_accounting():
    env_cfg, applied = _apply(
        {"terminations": {"envelope_as_penalty": True, "envelope_penalty_weight": -1.0}})
    T = env_cfg.terminations
    assert T.anchor_pos is None and T.ee_body_pos is None  # envelope no longer terminates
    # absolute terminations + anchor_ori stay
    assert T.anchor_ori == "ANCHOR_ORI_TERM"
    assert T.base_fell_tilt == "BASE_FELL_TILT_TERM"
    assert T.base_too_low == "BASE_TOO_LOW_TERM"
    assert env_cfg.rewards.tracking_envelope.weight == -1.0
    assert env_cfg.commands.racket_target.track_envelope_violation is True
    assert any("envelope_as_penalty" in a for a in applied)
    assert any("tracking_envelope.weight=-1.0" in a for a in applied)


def test_rb_default_weight_is_minus_one():
    env_cfg, _ = _apply({"terminations": {"envelope_as_penalty": True}})
    assert env_cfg.rewards.tracking_envelope.weight == -1.0


def test_rb_guards():
    with pytest.raises(train_mod._OverrideError, match="envelope_as_penalty"):
        _apply({"terminations": {"envelope_penalty_weight": -1.0}})  # weight without the flag
    with pytest.raises(train_mod._OverrideError, match="must be < 0"):
        _apply({"terminations": {"envelope_as_penalty": True, "envelope_penalty_weight": 1.0}})
    with pytest.raises(train_mod._OverrideError, match="does not\n?.*consume|translation layer"):
        _apply({"terminations": {"unknown_termination_knob": 1}})


def test_rb_off_leaves_terminations_untouched():
    env_cfg, applied = _apply({"terminations": {"envelope_as_penalty": False}})
    assert env_cfg.terminations.anchor_pos == "ANCHOR_POS_TERM"
    assert env_cfg.terminations.ee_body_pos == "EE_BODY_POS_TERM"
    assert env_cfg.rewards.tracking_envelope.weight == 0.0
    assert applied == []


# --------------------------------------------------------------------------------------------- #
# default-path guard: a realistic flag-free task node touches none of the new machinery
# --------------------------------------------------------------------------------------------- #
def test_deployparity_like_task_without_new_flags_is_untouched():
    task = {
        "rewards": {
            "racket_position_weight": 14.0, "racket_position_std": 0.20,
            "racket_velocity_weight": 10.0, "racket_velocity_std": 1.0,
            "racket_normal_weight": 5.0, "racket_normal_std": 0.30,
            "free_wrist_ori_mimic": False,
            "action_rate_weight": -0.10,
        },
        "racket": {"strike_window_s": 0.12},
        "motion": {"hold_steps_range": [0, 100], "post_swing_start_prob": 0.0},
    }
    env_cfg, applied = _apply(task)
    C = env_cfg.commands.racket_target
    R = env_cfg.rewards
    assert C.strike_window_pos_s is None and C.strike_window_wide_s is None
    assert C.track_envelope_violation is False
    assert "pos_gate_radius" not in R.racket_velocity.params
    assert "window_scale" not in R.motion_body_lin_vel.params
    assert R.racket_guidance.weight == 0.0 and R.tracking_envelope.weight == 0.0
    assert env_cfg.commands.motion.rsi_skip_settle_frames == 0
    assert env_cfg.commands.motion.rsi_hold_root_stand_z is False
    assert env_cfg.observations.policy.command.func == "generated_commands"
    assert env_cfg.terminations.anchor_pos == "ANCHOR_POS_TERM"
    # sanity: the plain overrides did land
    assert R.racket_position.weight == 14.0 and R.racket_position.params["std"] == 0.20
    assert len(applied) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
