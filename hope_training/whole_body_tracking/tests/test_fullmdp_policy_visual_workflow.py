"""Focused tests for the FullMDP checkpoint visual workflow."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import types

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_TEMP_IMPORT_STUBS = []
if "hydra" not in sys.modules and importlib.util.find_spec("hydra") is None:
    hydra_stub = types.ModuleType("hydra")

    def _main(**_kwargs):
        return lambda function: function

    hydra_stub.main = _main
    sys.modules["hydra"] = hydra_stub
    _TEMP_IMPORT_STUBS.append("hydra")
if (
    "omegaconf" not in sys.modules
    and importlib.util.find_spec("omegaconf") is None
):
    omega_stub = types.ModuleType("omegaconf")

    class _ListConfig(list):
        pass

    class _OmegaConf:
        pass

    omega_stub.ListConfig = _ListConfig
    omega_stub.OmegaConf = _OmegaConf
    sys.modules["omegaconf"] = omega_stub
    _TEMP_IMPORT_STUBS.append("omegaconf")

import play as play_mod  # noqa: E402

for _stub_name in _TEMP_IMPORT_STUBS:
    sys.modules.pop(_stub_name, None)


def _cfg(**overrides):
    class _Cfg(dict):
        __getattr__ = dict.__getitem__

    values = {
        "task": {"action_ball_full_mdp_runtime": True},
        "action_ball_dynamic_ready_bootstrap": True,
        "action_ball_dynamic_ready_artifact_path": "/exact/ready.json",
        "action_ball_dynamic_ready_artifact_sha256": "a" * 64,
        "action_ball_dynamic_ready_nominal_receipt_path": "/exact/hold.json",
        "action_ball_dynamic_ready_nominal_receipt_sha256": "b" * 64,
    }
    values.update(overrides)
    return _Cfg(values)


def test_policy_visual_installs_training_dynamic_ready_binding():
    motion = SimpleNamespace(
        motion_file="/exact/take061.npz",
        action_ball_dynamic_ready=None,
    )
    env_cfg = SimpleNamespace(
        commands=SimpleNamespace(
            motion=motion,
            racket_target=SimpleNamespace(
                clip_names_per_clip=("take061_slow_block_phase4_v1",)
            ),
        )
    )
    calls = []

    def load_binding(**kwargs):
        calls.append(kwargs)
        return {"binding_sha256": "c" * 64}

    binding = play_mod._install_action_ball_dynamic_ready_playback(
        _cfg(), env_cfg, load_binding=load_binding
    )
    assert binding == {"binding_sha256": "c" * 64}
    assert motion.action_ball_dynamic_ready is binding
    assert calls == [
        {
            "artifact_path": "/exact/ready.json",
            "artifact_sha256": "a" * 64,
            "nominal_hold_receipt_path": "/exact/hold.json",
            "nominal_hold_receipt_sha256": "b" * 64,
            "action_order": ["take061_slow_block_phase4_v1"],
            "motion_paths": ["/exact/take061.npz"],
        }
    ]


def test_policy_visual_without_dynamic_ready_does_not_mutate_motion():
    motion = SimpleNamespace(motion_file="/exact/take061.npz")
    env_cfg = SimpleNamespace(commands=SimpleNamespace(motion=motion))
    cfg = _cfg(
        action_ball_dynamic_ready_bootstrap=False,
        action_ball_dynamic_ready_artifact_path=None,
        action_ball_dynamic_ready_artifact_sha256=None,
        action_ball_dynamic_ready_nominal_receipt_path=None,
        action_ball_dynamic_ready_nominal_receipt_sha256=None,
    )
    assert (
        play_mod._install_action_ball_dynamic_ready_playback(cfg, env_cfg)
        is None
    )
    assert not hasattr(motion, "action_ball_dynamic_ready")


def test_explicit_video_directory_is_fresh_and_no_clobber(tmp_path):
    output = tmp_path / "policy-video"
    assert play_mod._prepare_video_output_directory(
        str(output), str(tmp_path / "unused")
    ) == str(output)
    assert output.is_dir()
    with pytest.raises(ValueError, match="already exists"):
        play_mod._prepare_video_output_directory(
            str(output), str(tmp_path / "unused")
        )
