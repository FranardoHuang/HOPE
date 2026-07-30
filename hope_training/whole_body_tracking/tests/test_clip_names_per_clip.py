"""N-stroke per-clip YAML addressing + the per-clip incoming-ball regime (CPU, no torch).

人话:以前"每 clip 一行"的框表只认 forehand/backhand 两个字面名字,所以五个动作根本写不出来,
同族两个 clip 还会撞成一行;来球速度更是只有一个全局框,谁的最佳来球速度落在框外谁就永远低分。
这组测试钉住三件事:
  1. ``racket.clip_names`` 声明有序的每-clip 键,N 个动作都能寻址;
  2. 行数对不上当场报错(报出缺哪个 clip),不再悄悄用短表;
  3. ``vb_vel_range_per_clip`` / ``vb_spin_abs_max_per_clip`` 存在且同样 fail-closed。

Run: python -m pytest hope_training/whole_body_tracking/tests/test_clip_names_per_clip.py -q
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
TRAIN_PY = REPO / "hope_training" / "whole_body_tracking" / "scripts" / "train.py"


@pytest.fixture(scope="module")
def T():
    spec = importlib.util.spec_from_file_location("_train_for_test", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_train_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


FIVE = ["fh_loop", "bh_loop_c", "s0_highpress", "bh_block", "fh_block_syn"]


def _box(x, y, z):
    return {"x": list(x), "y": list(y), "z": list(z)}


def _five_boxes(names=None):
    names = names or FIVE
    return {n: _box((1.0 + i * 0.1, 2.0 + i * 0.1), (-1.0, 1.0), (0.0, 1.5))
            for i, n in enumerate(names)}


# --------------------------------------------------------------- clip_names --- #
def test_default_is_the_legacy_two_names(T):
    assert T._resolve_clip_names({}) == ("forehand", "backhand")
    assert T._resolve_clip_names({"clip_names": None}) == ("forehand", "backhand")


def test_five_ordered_names_are_accepted(T):
    assert T._resolve_clip_names({"clip_names": FIVE}) == tuple(FIVE)


def test_duplicate_names_fail_closed(T):
    with pytest.raises(T._OverrideError, match="duplicate name"):
        T._resolve_clip_names({"clip_names": ["fh_loop", "fh_loop", "bh_block"]})


def test_empty_name_list_fails_closed(T):
    with pytest.raises(T._OverrideError, match="empty list"):
        T._resolve_clip_names({"clip_names": []})


# ------------------------------------------------------- per-clip racket boxes --- #
def test_five_clip_velocity_boxes_round_trip(T):
    rk = {"clip_names": FIVE, "vel_range_per_clip": _five_boxes()}
    out = T._resolve_vel_range_per_clip(rk, T._resolve_clip_names(rk))
    assert len(out) == 5
    assert out[0][0] == (1.0, 2.0)          # fh_loop is clip 0
    assert out[4][0] == (1.4, 2.4)          # fh_block_syn is clip 4


def test_two_clips_of_the_same_family_no_longer_collide(T):
    """The whole point: fh_loop and fh_block_syn are both forehand and now get DIFFERENT rows."""
    rk = {"clip_names": FIVE, "vel_range_per_clip": _five_boxes()}
    out = T._resolve_vel_range_per_clip(rk, T._resolve_clip_names(rk))
    assert out[FIVE.index("fh_loop")] != out[FIVE.index("fh_block_syn")]


def test_unknown_clip_name_names_the_expected_set(T):
    rk = {"clip_names": FIVE, "vel_range_per_clip": {"sidespin_flick": _box((1, 2), (0, 1), (0, 1))}}
    with pytest.raises(T._OverrideError, match="unknown clip name"):
        T._resolve_vel_range_per_clip(rk, T._resolve_clip_names(rk))


def test_short_table_fails_closed_and_names_the_missing_clips(T):
    rk = {"clip_names": FIVE, "vel_range_per_clip": _five_boxes(FIVE[:3])}
    with pytest.raises(T._OverrideError) as exc:
        T._resolve_vel_range_per_clip(rk, T._resolve_clip_names(rk))
    msg = str(exc.value)
    assert "3 row(s)" in msg and "5 clip(s)" in msg
    assert "bh_block" in msg and "fh_block_syn" in msg


def test_legacy_two_name_yaml_is_unchanged(T):
    rk = {"vel_range_per_clip": {"forehand": _box((1.5, 3.5), (-1, 1), (0, 1.5)),
                                 "backhand": _box((1.2, 2.4), (-1, 1), (0, 1.2))}}
    out = T._resolve_vel_range_per_clip(rk, T._resolve_clip_names(rk))
    assert out == (((1.5, 3.5), (-1.0, 1.0), (0.0, 1.5)),
                   ((1.2, 2.4), (-1.0, 1.0), (0.0, 1.2)))


def test_legacy_partial_table_now_fails_instead_of_silently_shortening(T):
    """The old parser returned a 1-row table here, which then rode the family expansion."""
    rk = {"vel_range_per_clip": {"forehand": _box((1.5, 3.5), (-1, 1), (0, 1.5))}}
    with pytest.raises(T._OverrideError, match="backhand"):
        T._resolve_vel_range_per_clip(rk, T._resolve_clip_names(rk))


def test_position_boxes_use_the_same_addressing(T):
    rk = {"clip_names": FIVE, "pos_range_per_clip": _five_boxes()}
    out = T._resolve_pos_range_per_clip(rk, T._resolve_clip_names(rk))
    assert len(out) == 5


def test_missing_axis_is_named(T):
    rk = {"clip_names": FIVE,
          "vel_range_per_clip": {n: {"x": [1, 2], "y": [-1, 1]} for n in FIVE}}
    with pytest.raises(T._OverrideError, match="missing 'z'"):
        T._resolve_vel_range_per_clip(rk, T._resolve_clip_names(rk))


# ------------------------------------------------- per-clip INCOMING-ball regime --- #
def test_incoming_ball_box_is_per_clip(T):
    """A block gets fast balls and a loop slow ones, in the SAME run."""
    rk = {"clip_names": FIVE, "vb_vel_range_per_clip": {
        "fh_loop": _box((-3.0, -1.5), (-0.6, 0.6), (-1.0, 0.5)),
        "bh_loop_c": _box((-3.0, -1.5), (-0.6, 0.6), (-1.0, 0.5)),
        "s0_highpress": _box((-4.0, -2.0), (-0.6, 0.6), (-1.0, 0.5)),
        "bh_block": _box((-5.5, -3.5), (-0.6, 0.6), (-1.0, 0.5)),
        "fh_block_syn": _box((-5.5, -3.5), (-0.6, 0.6), (-1.0, 0.5)),
    }}
    out = T._resolve_vb_vel_range_per_clip(rk, T._resolve_clip_names(rk))
    assert len(out) == 5
    assert out[FIVE.index("bh_block")][0] == (-5.5, -3.5)      # the block sees FAST balls
    assert out[FIVE.index("fh_loop")][0] == (-3.0, -1.5)       # the loop sees SLOW ones


def test_incoming_ball_box_length_check_mirrors_the_racket_boxes(T):
    rk = {"clip_names": FIVE,
          "vb_vel_range_per_clip": {"fh_loop": _box((-3, -1.5), (-0.6, 0.6), (-1, 0.5))}}
    with pytest.raises(T._OverrideError) as exc:
        T._resolve_vb_vel_range_per_clip(rk, T._resolve_clip_names(rk))
    assert "1 row(s)" in str(exc.value) and "5 clip(s)" in str(exc.value)


def test_incoming_spin_ceiling_per_clip_accepts_list_or_map(T):
    names = T._resolve_clip_names({"clip_names": FIVE})
    as_list = T._resolve_vb_spin_abs_max_per_clip({"vb_spin_abs_max_per_clip": [10, 20, 30, 40, 50]},
                                                  names)
    assert as_list == (10.0, 20.0, 30.0, 40.0, 50.0)
    as_map = T._resolve_vb_spin_abs_max_per_clip(
        {"vb_spin_abs_max_per_clip": dict(zip(FIVE, [10, 20, 30, 40, 50]))}, names)
    assert as_map == as_list


def test_incoming_spin_ceiling_length_check(T):
    names = T._resolve_clip_names({"clip_names": FIVE})
    with pytest.raises(T._OverrideError, match="5 clip"):
        T._resolve_vb_spin_abs_max_per_clip({"vb_spin_abs_max_per_clip": [10, 20]}, names)


def test_new_keys_are_whitelisted(T):
    for key in ("clip_names", "vb_vel_range_per_clip", "vb_spin_abs_max_per_clip"):
        assert key in T._RACKET_KEYS, key


# ------------------------------------------------------------- launch-source gate --- #
class _FakeCommandOld:
    """A command class from a checkout that predates the physical-validity guards."""


class _FakeCommandNew:
    def _assert_contact_clears_table(self):
        pass

    def _assert_target_velocity_points_forward(self):
        pass

    def _assert_reference_strike_can_return_its_own_regime(self):
        pass


def _fake_module(T, command_cls, name):
    import types
    mod = types.ModuleType(name)
    mod.__file__ = f"/fake/checkout/{name}.py"
    if command_cls is not None:
        mod.RacketTargetCommand = command_cls
    sys.modules[name] = mod
    cfg_cls = type("RacketTargetCommandCfg", (), {"__module__": name})
    return cfg_cls()


def test_launch_refuses_a_checkout_without_the_guards(T):
    cfg = _fake_module(T, _FakeCommandOld, "_fake_stale_checkout")
    with pytest.raises(T._OverrideError) as exc:
        T._assert_physical_validity_guards_present(cfg)
    msg = str(exc.value)
    assert "3 of 3 physical-validity guard(s)" in msg
    assert "_assert_contact_clears_table" in msg
    assert "_assert_reference_strike_can_return_its_own_regime" in msg
    assert "/fake/checkout/_fake_stale_checkout.py" in msg


def test_launch_accepts_a_checkout_with_all_three_guards(T):
    cfg = _fake_module(T, _FakeCommandNew, "_fake_fresh_checkout")
    T._assert_physical_validity_guards_present(cfg)      # must not raise


def test_launch_refuses_a_module_without_the_command_at_all(T):
    cfg = _fake_module(T, None, "_fake_empty_checkout")
    with pytest.raises(T._OverrideError, match="no RacketTargetCommand"):
        T._assert_physical_validity_guards_present(cfg)


def test_the_real_checkout_passes_its_own_gate(T):
    """This repo must satisfy the gate it ships — otherwise the gate is decorative."""
    src = (REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
           / "whole_body_tracking" / "tasks" / "tracking" / "mdp" / "hope_commands.py").read_text()
    for name, _what in T._REQUIRED_PHYSICAL_GUARDS:
        assert f"def {name}(" in src, name


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"target_mode": "uniform", "target_noise_white": 0.001}, False),
        ({"target_mode": "reference_perturbed"}, True),
        ({"target_mode": "solved"}, True),
        ({"target_mode": "task_first"}, True),
        ({"target_mode": "action_ball"}, True),
        ({"target_mode": "uniform", "question_bank": "/tmp/bank.npz"}, True),
        ({"target_mode": "uniform", "clip_names_per_clip": ("fh", "bh")}, True),
        ({"target_mode": "uniform", "racket_pos_range_per_clip": ()}, True),
        ({"target_mode": "uniform", "racket_vel_range_per_clip": ()}, True),
    ],
)
def test_launch_source_gate_only_covers_physical_constructions(T, fields, expected):
    import types

    assert T._physical_validity_guards_required(types.SimpleNamespace(**fields)) is expected
