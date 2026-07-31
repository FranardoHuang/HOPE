"""Compatibility coverage for legacy HOPE task reward-pack selection.

The repository default treats an absent ``task.rewards.reward_pack`` as v2.  That is appropriate
for ActionBall, whose environment class declares the complete virtual-ball outcome stack, but the
older HOPE/Hitter/Rally reward classes deliberately do not.  These tests exercise the real Hydra
composition and the trainer's pack-expansion boundary so a legacy task cannot silently regress to
an impossible v2 composition.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "cfg"
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as train_mod  # noqa: E402  (Hydra/OmegaConf only at import time)


LEGACY_TASKS = (
    "HOPEPingPong",
    "HOPEPingPongDeployParity",
    "HOPEPingPongRealSensor",
    "HOPEPingPongHitter",
    "HOPEPingPongHitterPure",
    "HOPEPingPongHitterPureRally",
    "HOPEPingPongHitterPureRallyV3",
)


def _compose_task(task_name: str):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(CFG_DIR.resolve()),
    ):
        return hydra.compose(
            config_name="train",
            overrides=[f"task={task_name}"],
        ).task


@pytest.mark.parametrize("task_name", LEGACY_TASKS)
def test_legacy_task_compose_pins_byte_preserving_reward_pack_v1(task_name):
    task = _compose_task(task_name)

    assert task.rewards.reward_pack == "v1"

    # v1 must return the exact composed rewards node and must not inspect or mutate the reward
    # cfg.  A deliberately empty env sentinel makes any accidental direct-term access fail here.
    applied: list[str] = []
    original_rewards_node = task.rewards
    expanded = train_mod._expand_reward_pack(
        SimpleNamespace(),
        task,
        original_rewards_node,
        applied,
    )
    assert expanded is original_rewards_node
    assert applied == ["rewards.reward_pack=v1 (legacy baseline)"]


@pytest.mark.parametrize(
    "task_name",
    ("HOPEPingPongActionBall", "HOPEPingPongActionBallA3VendorV1"),
)
def test_action_ball_compose_stays_explicitly_on_reward_pack_v2(task_name):
    task = _compose_task(task_name)
    assert task.rewards.reward_pack == "v2"

