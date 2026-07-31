"""Split Kp/Kd domain-randomization contract regressions (host-only, no Isaac imports)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import yaml


HERE = Path(__file__).resolve().parent
WBT = HERE.parent
SCRIPTS = WBT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# The checked-in macOS host Python intentionally has no Isaac/Hydra runtime.  train.py only needs
# these two names at import/decorator time for this pure translation test.
try:  # pragma: no cover - the Pod/Isaac environment takes the real import path
    import hydra as _hydra  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - covered by the repository host test lane
    hydra_stub = types.ModuleType("hydra")
    hydra_stub.main = lambda **_kwargs: (lambda fn: fn)
    sys.modules["hydra"] = hydra_stub

try:  # pragma: no cover - the Pod/Isaac environment takes the real import path
    import omegaconf as _omegaconf  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - covered by the repository host test lane
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.ListConfig = list
    omegaconf_stub.OmegaConf = SimpleNamespace()
    sys.modules["omegaconf"] = omegaconf_stub

import train as train_mod  # noqa: E402  (hydra/omegaconf only at import time)


class _Term:
    def __init__(self):
        self.mode = "startup"
        self.params = {
            "stiffness_distribution_params": (9.0, 9.0),
            "damping_distribution_params": (9.0, 9.0),
            "operation": "add",
            "distribution": "uniform",
        }


def _events():
    return SimpleNamespace(randomize_pd_gains=_Term())


def _apply(dr, *, stable_ready_plant=False):
    events = _events()
    applied = []
    train_mod._apply_pd_gain_dr_override(
        events,
        dr,
        applied,
        stable_ready_plant=stable_ready_plant,
    )
    return events, applied


def test_checked_in_action_ball_defaults_pin_vendor_split_ranges():
    base = yaml.safe_load((WBT / "cfg/base/randomization_base.yaml").read_text())
    action_ball = yaml.safe_load(
        (WBT / "cfg/task/HOPEPingPongActionBall.yaml").read_text()
    )
    for config in (base, action_ball):
        dr = config["domain_rand"]
        assert dr["kp_gain_range"] == [0.8, 1.2]
        assert dr["kd_gain_range"] == [0.7, 1.3]
        assert "pd_gain_range" not in dr


def test_code_default_pins_vendor_log_uniform_ranges():
    source = (
        WBT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    ).read_text()
    assert '"stiffness_distribution_params": (0.8, 1.2)' in source
    assert '"damping_distribution_params": (0.7, 1.3)' in source
    assert '"distribution": "log_uniform"' in source
    event_source = source.split("randomize_pd_gains = EventTerm(", 1)[1].split(
        "\n    )", 1
    )[0]
    assert 'mode="startup"' in event_source
    assert 'mode="reset"' not in event_source


def test_split_ranges_apply_to_separate_axes_and_force_log_uniform_scale():
    events, applied = _apply(
        {"kp_gain_range": [0.8, 1.2], "kd_gain_range": [0.7, 1.3]}
    )
    params = events.randomize_pd_gains.params
    assert params["stiffness_distribution_params"] == (0.8, 1.2)
    assert params["damping_distribution_params"] == (0.7, 1.3)
    assert params["operation"] == "scale"
    assert params["distribution"] == "log_uniform"
    assert any("source=split" in marker for marker in applied)


def test_legacy_one_range_spelling_remains_compatible():
    events, applied = _apply({"pd_gain_range": [0.85, 1.15]})
    params = events.randomize_pd_gains.params
    assert params["stiffness_distribution_params"] == (0.85, 1.15)
    assert params["damping_distribution_params"] == (0.85, 1.15)
    assert any("source=legacy" in marker for marker in applied)


@pytest.mark.parametrize(
    "dr",
    (
        {"kp_gain_range": [0.8, 1.2], "kd_gain_range": [0.7, 1.3]},
        {"pd_gain_range": [0.85, 1.15]},
    ),
)
def test_reset_level_event_is_rejected_for_split_and_legacy_value_spellings(dr):
    events = _events()
    events.randomize_pd_gains.mode = "reset"
    with pytest.raises(train_mod._OverrideError, match="mode='startup'.*not reset timing"):
        train_mod._apply_pd_gain_dr_override(
            events,
            dr,
            [],
            stable_ready_plant=False,
        )


@pytest.mark.parametrize(
    "dr, match",
    [
        (
            {
                "pd_gain_range": [0.85, 1.15],
                "kp_gain_range": [0.8, 1.2],
                "kd_gain_range": [0.7, 1.3],
            },
            "cannot coexist",
        ),
        ({"kp_gain_range": [0.8, 1.2]}, "missing kd_gain_range"),
        ({"kd_gain_range": [0.7, 1.3]}, "missing kp_gain_range"),
        (
            {"kp_gain_range": [0.8, 1.2], "kd_gain_range": None},
            "half-enabled",
        ),
    ],
)
def test_ambiguous_legacy_or_half_split_spelling_fails_loud(dr, match):
    with pytest.raises(train_mod._OverrideError, match=match):
        train_mod._resolve_pd_gain_ranges(dr)


@pytest.mark.parametrize(
    "value",
    ([0.0, 1.2], [1.2, 0.8], [0.8], [0.8, float("inf")], "0.8,1.2", [True, 1.2]),
)
def test_bad_gain_range_fails_before_env_mutation(value):
    with pytest.raises(train_mod._OverrideError, match="finite|exactly|0 < lo"):
        train_mod._resolve_pd_gain_ranges(
            {"kp_gain_range": value, "kd_gain_range": [0.7, 1.3]}
        )


@pytest.mark.parametrize(
    "dr",
    (
        {"kp_gain_range": None, "kd_gain_range": None},
        {"pd_gain_range": None},
    ),
)
def test_explicit_disable_spelling_turns_off_the_single_event(dr):
    events, applied = _apply(dr)
    assert events.randomize_pd_gains is None
    assert any("disabled" in marker for marker in applied)


def test_stable_ready_n1_disables_both_vendor_gain_axes_without_reinterpreting_them():
    events, applied = _apply(
        {"kp_gain_range": [0.8, 1.2], "kd_gain_range": [0.7, 1.3]},
        stable_ready_plant=True,
    )
    assert events.randomize_pd_gains is None
    assert applied == [
        "events.randomize_pd_gains=None(stable_ready_plant diagnostic override)"
    ]


def test_absent_pd_keys_preserve_the_code_default_event():
    events, applied = _apply({"link_mass_range": [0.85, 1.15]})
    assert isinstance(events.randomize_pd_gains, _Term)
    assert applied == []
