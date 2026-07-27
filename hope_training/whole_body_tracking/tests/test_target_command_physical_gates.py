"""Construction-time physical gates on the commanded racket target (CPU, isaaclab STUBBED).

TORCH-DEPENDENT: this file loads the REAL hope_commands.py, so it runs on a pod venv with torch, not
on the py3.8 host. The host-runnable half of the same subject (the shipped YAML/env-cfg values, the
runner's per-family ratios and the pinned-at-zero alarm) is test_commanded_contact_geometry.py.

What is pinned:

* G1 a commanded contact point out over the table must clear the table surface by one ball radius.
      The failure it prevents shipped for months: a forehand box bound below 0.76 m + 0.02 m, four
      runs at virtual_return_rate_forehand = exactly 0.0000, no error and no warning.
* G6 a per-clip target-velocity box must demand a return that travels toward the opponent (+x),
      with one explicit escape hatch.
* G4 a per-clip strike phase must lie in [0, 1]; _strike_frame_for_clip adds
      round(p * (seg_len - 1)) with no clamp, so a stray value reads a neighbouring clip's pose.

Run (pod):  python -m pytest hope_training/whole_body_tracking/tests/test_target_command_physical_gates.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch  # noqa: F401  (the real hope_commands import chain needs it)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_reward_flags_mdp import hope_commands_mod  # noqa: E402  (installs the isaaclab stub)

RT = hope_commands_mod.RacketTargetCommand
CFG = hope_commands_mod.RacketTargetCommandCfg

_GEOMETRY_PATH = os.path.join(
    os.path.dirname(HERE),
    "source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py",
)
_spec = importlib.util.spec_from_file_location("hope_geometry_for_gates", _GEOMETRY_PATH)
GEOMETRY = importlib.util.module_from_spec(_spec)
sys.modules["hope_geometry_for_gates"] = GEOMETRY
_spec.loader.exec_module(GEOMETRY)

# Legal boxes for the two clips of the shipped unified policy, both clearing the table.
LEGAL_BOXES = (
    ((0.58, 0.78), (-0.64, -0.24), (0.78, 0.98)),   # forehand
    ((0.56, 0.76), (-0.07, 0.33), (0.93, 1.13)),    # backhand
)
LEGAL_VELS = (
    ((1.05, 2.05), (0.96, 1.96), (0.31, 1.11)),     # forehand
    ((1.61, 2.61), (-1.21, -0.21), (0.00, 0.71)),   # backhand
)


def _rt(**cfg_kwargs):
    """A RacketTargetCommand carrying only what the construction-time gates read."""

    rt = RT.__new__(RT)
    rt.cfg = CFG(**cfg_kwargs)
    rt._vb_ball_r = GEOMETRY.BALL_RADIUS  # same constant __init__ caches from geometry.py
    rt._clip_names = {0: "forehand", 1: "backhand"}
    return rt


def _min_contact_z(rt) -> float:
    return float(rt.cfg.vb_table_surface_z) + rt._vb_ball_r


def _assert_boxes(rt, boxes):
    for clip_id, clip_rng in enumerate(boxes):
        rt._assert_contact_clears_table(
            clip_id,
            rt._commanded_target_x_hi(float(clip_rng[0][1])),
            float(clip_rng[2][0]),
            "racket_pos_range_per_clip",
        )


# --------------------------------------------------------------------------------------------- #
# G1 — commanded contact clears the table
# --------------------------------------------------------------------------------------------- #
def test_legal_box_over_the_table_is_accepted():
    rt = _rt(target_mode="uniform")
    _assert_boxes(rt, LEGAL_BOXES)
    assert _min_contact_z(rt) == pytest.approx(0.78)


def test_below_table_forehand_box_is_refused_by_name_and_value():
    """The exact shipped defect: forehand z floor 0.72 against a 0.76 m surface."""

    rt = _rt(target_mode="uniform")
    bad = (((0.58, 0.78), (-0.64, -0.24), (0.72, 0.92)), LEGAL_BOXES[1])
    with pytest.raises(ValueError) as excinfo:
        _assert_boxes(rt, bad)
    message = str(excinfo.value)
    assert "forehand" in message                       # names the offending clip
    assert "0.7200" in message                         # names the offending z
    assert "0.7800" in message                         # names the required minimum
    assert "racket_pos_range_per_clip" in message      # names where it came from


def test_box_that_never_reaches_the_table_is_out_of_scope():
    """A low target behind the near table edge is a reach exercise, not an impossible contact."""

    rt = _rt(target_mode="uniform")
    _assert_boxes(rt, (((0.25, 0.45), (-0.6, -0.2), (0.40, 0.70)),))


def test_hitter_pure_station_span_is_added_before_the_table_test():
    """hitter_pure boxes are STATION-relative, so the station's forward span rides on top."""

    behind = _rt(target_mode="hitter_pure", base_target_x_range=(0.0, 0.0))
    _assert_boxes(behind, (((0.45, 0.45), (-0.6, -0.2), (0.60, 0.90)),))

    over = _rt(target_mode="hitter_pure", base_target_x_range=(-0.10, 0.30))
    with pytest.raises(ValueError, match="past the near table edge"):
        _assert_boxes(over, (((0.45, 0.45), (-0.6, -0.2), (0.60, 0.90)),))


def test_reference_strike_point_below_the_table_is_refused():
    """The reference half: in reference_perturbed mode the clip's strike point IS the target centre."""

    rt = _rt(target_mode="reference_perturbed")
    # x is the reference point's own: base_target_*_range moves the BASE in this mode, not the target.
    rt._assert_contact_clears_table(0, 0.68, 0.82, "reference strike point")
    with pytest.raises(ValueError, match="reference strike point"):
        rt._assert_contact_clears_table(0, 0.68, 0.74, "reference strike point")


def test_gate_reads_the_table_constants_from_cfg_not_from_a_literal():
    """Move the virtual table and the legal floor moves with it."""

    rt = _rt(target_mode="uniform", vb_table_surface_z=0.90, vb_table_near_x=0.5)
    assert _min_contact_z(rt) == pytest.approx(0.92)
    with pytest.raises(ValueError, match="0.9200"):
        _assert_boxes(rt, LEGAL_BOXES)


# --------------------------------------------------------------------------------------------- #
# G6 — commanded target velocity points at the opponent
# --------------------------------------------------------------------------------------------- #
def test_shipped_velocity_boxes_point_at_the_opponent():
    _rt(racket_vel_range_per_clip=LEGAL_VELS)._assert_target_velocity_points_forward()


@pytest.mark.parametrize("x_lo", [0.0, -0.5])
def test_non_forward_velocity_box_is_refused(x_lo):
    bad = (((x_lo, 2.05), (0.96, 1.96), (0.31, 1.11)), LEGAL_VELS[1])
    rt = _rt(racket_vel_range_per_clip=bad)
    with pytest.raises(ValueError) as excinfo:
        rt._assert_target_velocity_points_forward()
    message = str(excinfo.value)
    assert "forehand" in message
    assert "allow_non_forward_target_velocity" in message


def test_escape_hatch_permits_a_deliberate_non_forward_box():
    bad = (((-0.5, 2.05), (0.96, 1.96), (0.31, 1.11)), LEGAL_VELS[1])
    rt = _rt(racket_vel_range_per_clip=bad, allow_non_forward_target_velocity=True)
    rt._assert_target_velocity_points_forward()


def test_escape_hatch_defaults_off():
    assert CFG.allow_non_forward_target_velocity is False


# --------------------------------------------------------------------------------------------- #
# G4 — strike phase sanity
# --------------------------------------------------------------------------------------------- #
def test_in_range_strike_phases_are_accepted():
    rt = _rt(strike_phase_per_clip=(0.0, 0.47, 1.0))
    assert rt._strike_phases_cfg(3) == (0.0, 0.47, 1.0)
    assert _rt(strike_phase_per_clip=())._strike_phases_cfg(2) == ()


@pytest.mark.parametrize("phases", [(0.47, 1.5), (-0.1, 0.333), (0.47, 33.0)])
def test_out_of_range_strike_phase_is_refused(phases):
    rt = _rt(strike_phase_per_clip=phases)
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        rt._strike_phases_cfg(len(phases))


def test_length_mismatch_still_takes_precedence():
    rt = _rt(strike_phase_per_clip=(0.47, 1.5))
    with pytest.raises(ValueError, match="loaded motion has 6 segment"):
        rt._strike_phases_cfg(6)


def test_out_of_range_phase_would_have_read_a_neighbouring_clip():
    """Provenance for the gate: the frame arithmetic it protects has no clamp."""

    import inspect

    src = inspect.getsource(RT._strike_frame_for_clip)
    assert "seg_start + round(phase * (seg_len - 1))" in src
