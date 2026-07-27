"""Commanded-contact physical rule + the zero-return alarm (dependency-light, runs on the py3.8 host).

Two things are pinned here:

* Every shipped task YAML commands a racket contact point a ball can actually occupy. The rule comes
  from the geometry constants, not from a retyped number: a box whose x reaches past the near table
  edge (``vb_table_near_x``) must have ``z_lo >= vb_table_surface_z + BALL_RADIUS``. A forehand box
  bound at z [0.594, 0.794] against a 0.76 m table surface is what pinned
  ``virtual_return_rate_forehand`` at exactly 0.0000 across four runs with nothing complaining.
* The runner turns that signature into an alarm: per-family outcome ratios stay ``None`` when a side
  was never eligible (never 0.0 — the distinction is the whole point), and a side whose cumulative
  legal returns stay at zero past the opportunity threshold gets a loud line, then an abort.

The GATE BEHAVIOUR itself (RacketTargetCommand refusing such a config at construction) needs torch +
the isaaclab stub and lives in test_target_command_physical_gates.py, which runs on a pod venv. What
runs here instead is a source-level check that the gates are still wired into both call sites.

Run:  python3 -m pytest hope_training/whole_body_tracking/tests/test_commanded_contact_geometry.py -q
"""

from __future__ import annotations

import ast
import importlib.util
import math
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source/whole_body_tracking/whole_body_tracking"
CMD_PATH = SRC / "tasks/tracking/mdp/hope_commands.py"
ENVCFG_PATH = SRC / "tasks/tracking/config/agibot_a3/hope_env_cfg.py"
RUNNER_PATH = SRC / "utils/my_on_policy_runner.py"
GEOMETRY_PATH = SRC / "tasks/table_tennis/geometry.py"
CFG_TASK_DIR = ROOT / "cfg/task"


def _load_pure_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses on py3.8 resolves field types through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GEOMETRY = _load_pure_module("hope_geometry_under_test", GEOMETRY_PATH)
CMD_SOURCE = CMD_PATH.read_text(encoding="utf-8")


def _cfg_default(field: str) -> float:
    """Read a RacketTargetCommandCfg float default straight out of the shipped source."""

    match = re.search(rf"^\s*{field}:\s*float\s*=\s*([0-9.eE+-]+)\s*$", CMD_SOURCE, re.MULTILINE)
    assert match is not None, f"{field} default not found in {CMD_PATH}"
    return float(match.group(1))


TABLE_NEAR_X = _cfg_default("vb_table_near_x")
TABLE_SURFACE_Z = _cfg_default("vb_table_surface_z")
MIN_CONTACT_Z = TABLE_SURFACE_Z + GEOMETRY.BALL_RADIUS


def _task_yamls():
    return sorted(p for p in CFG_TASK_DIR.glob("*.yaml"))


def _racket_node(path: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return doc.get("racket") or {}


def _world_x_hi(racket: dict, box: dict) -> float:
    """Farthest env-frame x the commanded target can reach from this box."""

    x_hi = float(box["x"][1])
    if racket.get("target_mode") == "hitter_pure":
        # hitter_pure boxes are STATION-relative, so the commanded station's forward span rides along.
        x_hi += float(max(racket.get("base_target_x_range", (0.0, 0.0))))
    return x_hi


def test_geometry_pins_the_minimum_legal_contact_height():
    assert GEOMETRY.TABLE_HEIGHT == pytest.approx(TABLE_SURFACE_Z)
    assert MIN_CONTACT_Z == pytest.approx(0.78)


@pytest.mark.parametrize("path", _task_yamls(), ids=lambda p: p.name)
def test_shipped_task_yaml_commands_a_reachable_contact_point(path):
    racket = _racket_node(path)
    boxes = racket.get("pos_range_per_clip")
    if not boxes:
        pytest.skip("no per-clip racket target box in this task")
    for clip, box in boxes.items():
        x_hi = _world_x_hi(racket, box)
        if x_hi <= TABLE_NEAR_X:
            continue
        assert float(box["z"][0]) >= MIN_CONTACT_Z - 1e-9, (
            f"{path.name}: {clip} target box reaches x={x_hi:.3f} m (past the near table edge "
            f"{TABLE_NEAR_X:.3f}) with z_lo={box['z'][0]} — below the table surface "
            f"{TABLE_SURFACE_Z} + ball radius {GEOMETRY.BALL_RADIUS} = {MIN_CONTACT_Z:.3f}"
        )


@pytest.mark.parametrize("path", _task_yamls(), ids=lambda p: p.name)
def test_shipped_task_yaml_commands_a_return_toward_the_opponent(path):
    racket = _racket_node(path)
    boxes = racket.get("vel_range_per_clip")
    if not boxes:
        pytest.skip("no per-clip racket velocity box in this task")
    for clip, box in boxes.items():
        assert float(box["x"][0]) > 0.0, (
            f"{path.name}: {clip} target velocity box has x_lo={box['x'][0]} <= 0, so the commanded "
            f"return can point away from the opponent (+x is toward the opponent)"
        )


def test_env_cfg_hitter_pure_default_boxes_clear_the_table():
    """The code default used by verify/export paths that never go through train.py."""

    tree = ast.parse(ENVCFG_PATH.read_text(encoding="utf-8"))
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "racket_pos_range_per_clip"
            for target in node.targets
        )
    ]
    assert len(literals) == 1, "expected exactly one racket_pos_range_per_clip default"
    boxes = ast.literal_eval(literals[0])
    assert boxes[0][2] == (0.78, 1.08)  # forehand: floor raised to the legal minimum, span preserved
    for clip_id, (x_range, _y_range, z_range) in enumerate(boxes):
        # base_target_x_range is (0.0, 0.0) in this cfg, so the station adds nothing.
        if float(x_range[1]) <= TABLE_NEAR_X:
            continue
        assert float(z_range[0]) >= MIN_CONTACT_Z - 1e-9, f"clip {clip_id} box floor {z_range[0]}"


def test_table_clearance_gate_is_wired_into_both_construction_paths():
    """A gate nobody calls is the bug it was written to prevent."""

    assert '"racket_pos_range_per_clip",' in CMD_SOURCE  # box half, in __init__
    assert '"reference strike point",' in CMD_SOURCE     # reference half, after the motion resolves
    assert CMD_SOURCE.count("self._assert_contact_clears_table(") == 2
    assert "self._assert_target_velocity_points_forward()" in CMD_SOURCE
    assert "allow_non_forward_target_velocity" in CMD_SOURCE


# --------------------------------------------------------------------------------------------- #
# runner: per-family outcome ratios + the pinned-at-zero alarm
# --------------------------------------------------------------------------------------------- #


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_runner_module(monkeypatch):
    """Load the real runner module with torch/rsl_rl/isaaclab stubbed (host has none of them)."""

    fake_torch = _module("torch", Tensor=type("Tensor", (), {}))
    for package in ("rsl_rl", "rsl_rl.runners", "isaaclab_rl", "whole_body_tracking",
                    "whole_body_tracking.utils", "whole_body_tracking.tasks",
                    "whole_body_tracking.tasks.tracking",
                    "whole_body_tracking.tasks.tracking.mdp"):
        stub = _module(package)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, package, stub)
    contract = _load_pure_module(
        "training_contract_contact_geometry_under_test", SRC / "utils/training_contract.py"
    )
    modules = {
        "torch": fake_torch,
        "rsl_rl.env": _module("rsl_rl.env", VecEnv=type("VecEnv", (), {})),
        "rsl_rl.runners.on_policy_runner": _module(
            "rsl_rl.runners.on_policy_runner", OnPolicyRunner=type("OnPolicyRunner", (), {})
        ),
        "isaaclab_rl.rsl_rl": _module(
            "isaaclab_rl.rsl_rl", export_policy_as_onnx=lambda *args, **kwargs: None
        ),
        "whole_body_tracking.utils.exporter": _module(
            "whole_body_tracking.utils.exporter",
            attach_onnx_metadata=lambda *args, **kwargs: None,
            export_motion_policy_as_onnx=lambda *args, **kwargs: False,
            is_empirical_normalizer=lambda value: False,
        ),
        "whole_body_tracking.utils.training_contract": contract,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return _load_pure_module("motion_runner_contact_geometry_under_test", RUNNER_PATH)


def test_zero_denominator_is_unavailable_not_zero(monkeypatch):
    runner_module = _load_runner_module(monkeypatch)
    assert runner_module._ratio_or_none({"n": 3, "d": 0}, "n", "d") is None
    assert runner_module._ratio_or_none({}, "n", "d") is None
    assert runner_module._ratio_or_none({"n": 0, "d": 4}, "n", "d") == pytest.approx(0.0)


def test_per_family_outcome_ratios_separate_never_eligible_from_always_failing(monkeypatch):
    runner_module = _load_runner_module(monkeypatch)
    derived = runner_module.exact_behavior_decision_values(
        {
            "strike_opportunity_count_forehand": 40,
            "virtual_capture_count_forehand": 10,
            "virtual_legal_return_count_forehand": 0,
            "strike_opportunity_count_backhand": 0,
        }
    )
    # forehand strikes and never returns -> an honest 0.0; backhand never struck -> unavailable.
    assert derived["virtual_legal_return_per_strike_forehand"] == pytest.approx(0.0)
    assert derived["virtual_capture_per_strike_forehand"] == pytest.approx(0.25)
    assert derived["virtual_legal_return_per_strike_backhand"] is None
    assert derived["virtual_capture_per_strike_backhand"] is None


def test_zero_return_alarm_levels_follow_the_cumulative_thresholds(monkeypatch):
    runner_module = _load_runner_module(monkeypatch)
    below = {"strike_opportunity_count_forehand": 499, "virtual_legal_return_count_forehand": 0}
    at = {"strike_opportunity_count_forehand": 500, "virtual_legal_return_count_forehand": 0}
    abort = {"strike_opportunity_count_forehand": 5000, "virtual_legal_return_count_forehand": 0}
    healthy = {"strike_opportunity_count_forehand": 5000, "virtual_legal_return_count_forehand": 1}
    assert runner_module.zero_return_alarm_levels(below) == {}
    assert runner_module.zero_return_alarm_levels(at) == {"forehand": "alarm"}
    assert runner_module.zero_return_alarm_levels(abort) == {"forehand": "abort"}
    assert runner_module.zero_return_alarm_levels(healthy) == {}


def _alarm_runner(runner_module):
    runner = runner_module.MotionOnPolicyRunner.__new__(runner_module.MotionOnPolicyRunner)
    runner.logged = []
    runner._log_scalar = lambda tag, value, step: runner.logged.append((tag, value, step))
    return runner


def test_alarm_fires_at_the_threshold_and_aborts_at_the_abort_threshold(monkeypatch, capsys):
    runner_module = _load_runner_module(monkeypatch)
    runner = _alarm_runner(runner_module)
    window = {
        "strike_opportunity_count_forehand": 100,
        "virtual_legal_return_count_forehand": 0,
        "strike_opportunity_count_backhand": 100,
        "virtual_legal_return_count_backhand": 60,
    }

    for update in range(4):  # 400 cumulative forehand opportunities: below the alarm threshold
        runner._check_zero_return_alarm("racket_target", window, update)
    assert "[HOPE ALARM]" not in capsys.readouterr().out
    assert ("Live/racket_target/zero_return_alarm_forehand", 0.0, 3) in runner.logged

    runner._check_zero_return_alarm("racket_target", window, 4)  # 500 -> alarm
    out = capsys.readouterr().out
    assert "[HOPE ALARM]" in out
    assert "virtual_legal_return_count_forehand" in out
    assert "500 cumulative strike opportunities" in out
    assert ("Live/racket_target/zero_return_alarm_forehand", 1.0, 4) in runner.logged
    # the healthy side never alarms
    assert ("Live/racket_target/zero_return_alarm_backhand", 0.0, 4) in runner.logged

    for update in range(5, 49):  # up to 4900 cumulative: still only the loud line
        runner._check_zero_return_alarm("racket_target", window, update)
    capsys.readouterr()
    with pytest.raises(RuntimeError, match="HOPE ALARM"):
        runner._check_zero_return_alarm("racket_target", window, 49)  # 5000 cumulative -> abort


def test_alarm_ignores_families_the_ledger_never_reports(monkeypatch, capsys):
    runner_module = _load_runner_module(monkeypatch)
    runner = _alarm_runner(runner_module)
    runner._check_zero_return_alarm("racket_target", {"swing_outcome_count": 10}, 1)
    assert runner.logged == []
    assert "[HOPE ALARM]" not in capsys.readouterr().out


def test_alarm_state_is_per_command_term(monkeypatch, capsys):
    runner_module = _load_runner_module(monkeypatch)
    runner = _alarm_runner(runner_module)
    window = {"strike_opportunity_count_forehand": 400, "virtual_legal_return_count_forehand": 0}
    runner._check_zero_return_alarm("racket_target", window, 1)
    runner._check_zero_return_alarm("other_target", window, 1)
    assert "[HOPE ALARM]" not in capsys.readouterr().out
    runner._check_zero_return_alarm("racket_target", window, 2)
    assert "[HOPE ALARM] racket_target" in capsys.readouterr().out


def test_math_import_is_still_available_for_the_ratio_guard(monkeypatch):
    runner_module = _load_runner_module(monkeypatch)
    assert runner_module._ratio_or_none({"n": math.inf, "d": 1}, "n", "d") is None
