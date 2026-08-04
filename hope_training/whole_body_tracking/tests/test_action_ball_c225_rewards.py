"""CPU tensor regressions for the C-only causal reward surface."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "action_ball_c225_rewards.py"
)
SPEC = importlib.util.spec_from_file_location("c225_rewards_under_test", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)
M.action_ball_task_valid_mask = lambda cmd: cmd._action_ball_task_valid


class _Manager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "racket_target"
        return self.command


def _env(command):
    return SimpleNamespace(command_manager=_Manager(command))


def _strike_command():
    return SimpleNamespace(
        metrics={"exact_strike_hit_rate": torch.tensor([1.0, 1.0, 0.0])},
        racket_pos_w=torch.zeros((3, 3)),
        _action_ball_attempt_active=torch.tensor([True, True, True]),
        _action_ball_task_valid=torch.tensor([True, True, True]),
        _action_ball_ball_contact_target_w=torch.tensor(
            [[0.02, 0.0, 0.0], [0.50, 0.0, 0.0], [0.02, 0.0, 0.0]]
        ),
    )


def _landing_command():
    return SimpleNamespace(
        cfg=SimpleNamespace(vb_landing_sigma=1.0),
        _vb_target_xy_per_env=torch.zeros((4, 2)),
        vb_landing_xy=torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [3.0, 0.0],
                [-1.0, 0.0],
            ]
        ),
        vb_fired=torch.tensor([True, True, True, True]),
        _action_ball_task_valid=torch.tensor([True, True, True, True]),
        vb_landing_valid=torch.tensor([True, True, True, True]),
        vb_net_crossed=torch.tensor([True, True, True, True]),
        vb_net_clear=torch.tensor([True, True, True, True]),
        vb_on_opponent=torch.tensor([True, False, False, False]),
        _vb_net_x=-0.5,
    )


def test_strike_bridge_is_exact_tick_only_and_miss_keeps_a_cauchy_tail():
    command = _strike_command()
    reward = M.c225_strike_ball_paddle_center_proximity(
        _env(command), "racket_target", std=0.15
    )
    assert reward[0].item() == pytest.approx(
        1.0 / (1.0 + (0.02 / 0.15) ** 2)
    )
    assert 0.0 < reward[1].item() < reward[0].item()
    assert reward[2].item() == 0.0

    command._action_ball_attempt_active[0] = False
    assert M.c225_strike_ball_paddle_center_proximity(
        _env(command), "racket_target", std=0.15
    )[0].item() == 0.0

    command._action_ball_attempt_active[0] = True
    command._action_ball_task_valid[0] = False
    assert M.c225_strike_ball_paddle_center_proximity(
        _env(command), "racket_target", std=0.15
    )[0].item() == 0.0


def test_landing_hierarchy_is_contact_flight_gated_and_off_table_decays():
    command = _landing_command()
    reward = M.c225_landing_outcome_actual_contact(
        _env(command), "racket_target"
    )
    assert reward[0].item() == pytest.approx(1.0)
    assert reward[1].item() == pytest.approx(0.5)
    assert 0.0 < reward[2].item() < reward[1].item()
    assert reward[3].item() == 0.0

    for field in ("vb_fired", "vb_landing_valid", "vb_net_crossed", "vb_net_clear"):
        rejected = _landing_command()
        getattr(rejected, field)[0] = False
        assert M.c225_landing_outcome_actual_contact(
            _env(rejected), "racket_target"
        )[0].item() == 0.0

    waiting = _landing_command()
    waiting._action_ball_task_valid[0] = False
    assert M.c225_landing_outcome_actual_contact(
        _env(waiting), "racket_target"
    )[0].item() == 0.0


def test_task_valid_zero_masks_every_c211_outcome_income():
    strike = _strike_command()
    strike._action_ball_task_valid[:] = False
    assert torch.count_nonzero(
        M.c225_strike_ball_paddle_center_proximity(
            _env(strike), "racket_target", std=0.15
        )
    ).item() == 0

    landing = _landing_command()
    landing._action_ball_task_valid[:] = False
    assert torch.count_nonzero(
        M.c225_landing_outcome_actual_contact(_env(landing), "racket_target")
    ).item() == 0


def test_c225_reward_source_never_reads_desired_contact_or_inverse_targets():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    used_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not {
        "racket_target_pos_w",
        "racket_target_vel_w",
        "racket_target_normal_w",
        "target_normal_cmd",
    } & used_attributes
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "FACE_AREA_CENTER_XZ_FROM_SITE_M" not in source
    assert "_action_ball_mount_signs" not in source
    assert "official_racket_site" in source


def test_paddle_center_is_exact_live_official_site_without_face_offset():
    command = _strike_command()
    command.racket_pos_w = torch.tensor(
        [[0.123, -0.456, 0.789], [1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]
    )
    center, finite = M._paddle_center_w(command)
    assert torch.equal(center, command.racket_pos_w)
    assert finite.tolist() == [True, True, True]
    # The helper returns the authoritative site itself, not site+offset or a
    # teacher/measured tensor with a similar name.
    command.measured_racket_site_pos_w = command.racket_pos_w + 0.123
    center_again, _ = M._paddle_center_w(command)
    assert torch.equal(center_again, command.racket_pos_w)


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"std": 0.0}, "std"),
        ({"std": float("nan")}, "std"),
    ),
)
def test_invalid_strike_kernel_contract_fails_closed(kwargs, match):
    command = _strike_command()
    with pytest.raises(ValueError, match=match):
        M.c225_strike_ball_paddle_center_proximity(
            _env(command), "racket_target", **kwargs
        )


def test_nonzero_landing_delay_and_tier_inversion_fail_closed():
    command = _landing_command()
    with pytest.raises(ValueError, match="settle_delay"):
        M.c225_landing_outcome_actual_contact(
            _env(command), "racket_target", settle_delay_s=0.02
        )
    with pytest.raises(ValueError, match="off_table_frac"):
        M.c225_landing_outcome_actual_contact(
            _env(command), "racket_target", base_frac=0.4, off_table_frac=0.5
        )
