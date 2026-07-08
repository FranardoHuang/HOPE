"""Unit tests for the 2026-07-08 exam contract-alignment flags in mujoco_eval_onnx.py
(--qdes-clamp / --hold-ref stand; fixE retrial follow-up).

Loads scripts/mujoco_eval_onnx.py directly by file path — mujoco/onnxruntime are LAZY imports
inside MujocoRobot/OnnxPolicy, so the module (and the pure helpers under test) import with numpy
only. Covers:

* soft_joint_limits: 0.9-about-midpoint arithmetic against hand-computed values; custom factor;
  unlimited joints (MJCF hi <= lo) -> (-inf, +inf) so a clamp never touches them; asymmetric
  ranges keep the midpoint (NOT 0.9 * bound).
* the clamp semantics --qdes-clamp enables: in-soft-range q_des is untouched (healthy-arm
  no-perturbation, the fixC six-cell adjudication invariant), out-of-range q_des pins to the
  soft bound (the fixE clamp-rider case), unlimited joints pass any magnitude.
* stand_hold_refs: joint_pos -> default_q, joint_vel -> zeros, every other refs entry passes
  through as the SAME object, the input dict and the joint_vel array are NOT mutated (refs_table
  entries are shared across steps — mutation would poison later swings).
* default-path wiring: run_rollout defaults are qdes_clamp=False / hold_ref="clip" and the CLI
  defaults match (flags off == byte-identical legacy exam; every booked score stays comparable).

Run:  pytest hope_training/whole_body_tracking/tests/test_mujoco_eval_align_flags.py
  or: python3 hope_training/whole_body_tracking/tests/test_mujoco_eval_align_flags.py
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.abspath(os.path.join(HERE, "..", "scripts", "mujoco_eval_onnx.py"))


def _load_module():
    spec = importlib.util.spec_from_file_location("mj_eval_align_under_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


# ---------------------------------------------------------------------------------------------
# soft_joint_limits
# ---------------------------------------------------------------------------------------------
def test_soft_limits_midpoint_arithmetic():
    # symmetric range: soft = 0.9 * bounds
    lo, hi = M.soft_joint_limits(np.array([[-1.0, 1.0]]))
    assert np.allclose([lo[0], hi[0]], [-0.9, 0.9])
    # asymmetric range: shrink about the MIDPOINT, not 0.9 * each bound
    lo, hi = M.soft_joint_limits(np.array([[0.0, 2.0]]))   # mid 1.0, half 1.0 -> 0.9
    assert np.allclose([lo[0], hi[0]], [0.1, 1.9])
    lo, hi = M.soft_joint_limits(np.array([[-2.0, 0.5]]))  # mid -0.75, half 1.25 -> 1.125
    assert np.allclose([lo[0], hi[0]], [-1.875, 0.375])


def test_soft_limits_factor_and_default():
    assert M.SOFT_JOINT_POS_LIMIT_FACTOR == 0.9  # Isaac agibot_a3.py soft_joint_pos_limit_factor
    lo, hi = M.soft_joint_limits(np.array([[-1.0, 1.0]]), factor=0.5)
    assert np.allclose([lo[0], hi[0]], [-0.5, 0.5])


def test_soft_limits_unlimited_joint_is_infinite():
    # MJCF "unlimited" convention: range hi <= lo (typically 0 0) -> clamp must never touch it
    lo, hi = M.soft_joint_limits(np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]]))
    assert lo[0] == -np.inf and hi[0] == np.inf
    assert lo[1] == -np.inf and hi[1] == np.inf
    assert np.isfinite(lo[2]) and np.isfinite(hi[2])


# ---------------------------------------------------------------------------------------------
# the clamp --qdes-clamp applies (np.clip against the soft limits)
# ---------------------------------------------------------------------------------------------
def test_clamp_semantics():
    rng = np.array([[-1.0, 1.0], [-2.0, 2.0], [0.0, 0.0]])
    lo, hi = M.soft_joint_limits(rng)
    # fixC-shaped healthy q_des (inside every soft range): the clamp is a NO-OP
    q_in = np.array([0.5, -1.7, 3.0])
    assert np.array_equal(np.clip(q_in, lo, hi), q_in)
    # fixE-shaped clamp-rider q_des (way past the limits): pinned to the soft bound
    q_out = np.array([7.8, -19.0, 42.0])
    clipped = np.clip(q_out, lo, hi)
    assert np.allclose(clipped[:2], [0.9, -1.8])
    assert clipped[2] == 42.0  # unlimited joint passes any magnitude


# ---------------------------------------------------------------------------------------------
# stand_hold_refs
# ---------------------------------------------------------------------------------------------
def test_stand_hold_refs_replaces_joints_only_and_never_mutates():
    default_q = np.linspace(-0.3, 0.3, 31)
    joint_pos = np.ones(31) * 0.123
    joint_vel = np.ones(31) * 4.56
    body_pos_w = np.arange(42.0).reshape(14, 3)
    refs = {"joint_pos": joint_pos, "joint_vel": joint_vel, "body_pos_w": body_pos_w}
    out = M.stand_hold_refs(refs, default_q)
    # joint command re-pointed at READY STAND
    assert out["joint_pos"] is default_q
    assert np.array_equal(out["joint_vel"], np.zeros(31))
    # everything else passes through as the SAME object (body/anchor refs untouched)
    assert out["body_pos_w"] is body_pos_w
    # the shared refs_table entry is NOT mutated (a mutation would poison every later step)
    assert out is not refs
    assert refs["joint_pos"] is joint_pos and np.all(refs["joint_pos"] == 0.123)
    assert np.all(refs["joint_vel"] == 4.56)
    assert np.all(joint_vel == 4.56)  # zeros_like allocated fresh, input array untouched


# ---------------------------------------------------------------------------------------------
# default-path wiring (flags off == byte-identical legacy exam)
# ---------------------------------------------------------------------------------------------
def test_run_rollout_defaults_are_off():
    sig = inspect.signature(M.run_rollout)
    assert sig.parameters["qdes_clamp"].default is False
    assert sig.parameters["hold_ref"].default == "clip"


def test_cli_defaults_are_off():
    # main() builds its parser inline; scrape the two add_argument calls from the source instead
    # of running main (which would need mujoco + an ONNX). Defaults must be OFF/legacy.
    import re
    src = inspect.getsource(M)
    # store_true => default False; --hold-ref must default to the legacy "clip"
    m = re.search(r'add_argument\("--hold-ref",\s*choices=\["clip",\s*"stand"\],\s*default="clip"', src)
    assert m, "--hold-ref must default to the legacy 'clip' semantics"
    m = re.search(r'add_argument\("--qdes-clamp",\s*action="store_true"', src)
    assert m, "--qdes-clamp must be a store_true flag (default OFF)"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
