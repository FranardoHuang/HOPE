"""Torch tests for the incoming-ball birth-consistency gate (Franco 2026-07-28).

人话:低速球 + 短到球时间会把球的出生点反推到网这边("球出生在半路"),这种题
必须被具名拒绝 ball_birth_not_beyond_net,而正常从对面飞来的球必须照常通过。
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import types

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
MDP = (
    REPO
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
PKG = "whole_body_tracking.tasks.tracking.mdp"
GEOMETRY_PATH = (
    MDP.parent.parent / "table_tennis" / "geometry.py"
)
HOPE_COMMANDS_SOURCE = (MDP / "hope_commands.py").read_text(encoding="utf-8")


def _load(name):
    dotted = f"{PKG}.{name}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, str(MDP / f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


for prefix in (
    "whole_body_tracking",
    "whole_body_tracking.tasks",
    "whole_body_tracking.tasks.tracking",
    PKG,
):
    sys.modules.setdefault(prefix, types.ModuleType(prefix))
sys.modules[PKG].__path__ = [str(MDP)]
_load("virtual_ball")
_load("strike_spec_torch")
_load("stroke_prototypes_torch")
_load("stroke_adapt_torch")
CQ = _load("continuous_questions")

_geometry_ns: dict = {}
exec(compile(GEOMETRY_PATH.read_text(encoding="utf-8"), str(GEOMETRY_PATH), "exec"), _geometry_ns)
NET_X_ENV = 0.5 + _geometry_ns["NET_X"]  # default vb_table_near_x + geometry net plane


def test_slow_ball_short_ttc_is_rejected_by_name():
    """Birth point back-solved to mid-court must trip the named gate."""

    contact_x = torch.tensor(0.45, dtype=torch.float64)
    v_in_x = torch.tensor(-0.60, dtype=torch.float64)  # hard 0.4x floor regime
    ttc = torch.tensor(0.45, dtype=torch.float64)
    bound = CQ.ball_birth_x_lower_bound_m(
        float(contact_x), float(v_in_x), float(ttc)
    )
    assert bound == float(contact_x) + 0.60 * 0.45
    assert bound < NET_X_ENV + CQ.BALL_BIRTH_NET_MARGIN_M
    assert CQ.ball_birth_not_beyond_net(
        float(contact_x), float(v_in_x), float(ttc), net_x_m=NET_X_ENV
    )
    assert CQ.BALL_BIRTH_REJECTION_REASON == "ball_birth_not_beyond_net"


def test_normal_incoming_ball_passes():
    """A venue-typical ball (3 m/s class, >=1.4 s flight) must not be rejected."""

    contact_x = torch.tensor(0.45, dtype=torch.float64)
    v_in_x = torch.tensor(-2.33, dtype=torch.float64)
    ttc = torch.tensor(1.50, dtype=torch.float64)
    assert not CQ.ball_birth_not_beyond_net(
        float(contact_x), float(v_in_x), float(ttc), net_x_m=NET_X_ENV
    )
    # sign convention: the bound uses |v_x|, so a mistakenly positive x
    # velocity cannot smuggle a mid-court birth past the gate either.
    assert CQ.ball_birth_x_lower_bound_m(0.45, 2.33, 1.5) == (
        CQ.ball_birth_x_lower_bound_m(0.45, -2.33, 1.5)
    )


def test_boundary_semantics_and_runtime_wiring_are_exact():
    """Strictly-below rejects; at-margin passes; runtime uses the same constant."""

    at_margin = NET_X_ENV + CQ.BALL_BIRTH_NET_MARGIN_M
    assert not CQ.ball_birth_not_beyond_net(
        at_margin, 0.0, 0.0, net_x_m=NET_X_ENV
    )
    assert CQ.ball_birth_not_beyond_net(
        at_margin - 1.0e-9, 0.0, 0.0, net_x_m=NET_X_ENV
    )

    # hope_commands must pin the same margin in the solver contract payload ...
    match = re.search(r"\"net_margin_m\": ([0-9.]+),", HOPE_COMMANDS_SOURCE)
    assert match is not None
    assert float(match.group(1)) == CQ.BALL_BIRTH_NET_MARGIN_M
    # ... expose the named reason in the ordered solver schema ...
    schema_start = HOPE_COMMANDS_SOURCE.index("ordered_rejection_reason_schema")
    schema_text = HOPE_COMMANDS_SOURCE[schema_start : schema_start + 700]
    assert '"ball_birth_not_beyond_net"' in schema_text
    assert schema_text.index('"cycle_exceeds_episode_horizon"') < schema_text.index(
        '"ball_birth_not_beyond_net"'
    )
    # ... and evaluate the gate inside the refill timing-rejection loop with the
    # geometry-sourced net plane, before any teacher-rate verdict.
    loop_start = HOPE_COMMANDS_SOURCE.index("timing_by_flat_index = {}")
    loop_text = HOPE_COMMANDS_SOURCE[loop_start : loop_start + 4000]
    flat = " ".join(loop_text.split())
    assert "birth_x_lower_bound_m = ball_birth_x_lower_bound_m(" in flat
    assert (
        "if birth_x_lower_bound_m < ( net_x + BALL_BIRTH_NET_MARGIN_M ):" in flat
    )
    assert flat.index('timing_reason = "ball_birth_not_beyond_net"') < flat.index(
        'timing_reason = "teacher_rate_out_of_bounds"'
    )
