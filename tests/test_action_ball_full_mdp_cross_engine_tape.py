from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/action_ball_full_mdp_cross_engine_tape.py"
SPEC = importlib.util.spec_from_file_location("full_mdp_cross_engine_tape_test", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def _arrays(*, initial_delta=0.0, tick_delta=0.0):
    tape, config, _config_sha, _tape_sha = M.action_tape_numpy()
    result = {"actions": tape}
    for name, shape in M._required_shapes(config).items():
        if name == "actions":
            continue
        dtype = np.bool_ if name in M.DISCRETE_FIELDS else np.float32
        result[name] = np.zeros(shape, dtype=dtype)
    result["initial_root_pos"][0, 0] = initial_delta
    result["root_pos"][3, 0, 0] = tick_delta
    result["initial_root_quat"][:, 0] = 1.0
    result["root_quat"][:, :, 0] = 1.0
    return result


def test_tracked_cross_engine_tape_is_recomputable():
    config, digest = M.load_config()
    payload, tape_sha = M.generate_action_bytes(config)
    assert digest == M.CONFIG_SHA256
    assert len(payload) == 512 * 48 * 31 * 4
    assert tape_sha == "dff37981204156b442815d274aa59201c485749fa0364218a6d7afd13418961d"
    assert payload[:20].hex() == "e19591bc14c3d23b5d529e3cacae253c6edb4d3c"


def test_comparison_separates_initial_from_post_step_difference(tmp_path):
    names = [f"joint_{index}" for index in range(31)]
    isaac = tmp_path / "isaac"
    mujoco = tmp_path / "mujoco"
    M.write_probe_record(
        isaac,
        backend="isaac",
        arrays=_arrays(),
        joint_names=names,
        runtime_identity={"kind": "test"},
    )
    M.write_probe_record(
        mujoco,
        backend="mujoco",
        arrays=_arrays(initial_delta=0.25, tick_delta=0.5),
        joint_names=names,
        runtime_identity={"kind": "test"},
    )
    output = tmp_path / "comparison.json"
    result = M.compare_records(isaac, mujoco, output)
    assert result["first_exact_difference"] == {
        "phase": "initial",
        "tick": -1,
        "field": "initial_root_pos",
        "index": (0, 0),
    }
    assert result["numeric_difference_envelope"]["initial_root_pos"]["max_abs"] == 0.25
    assert result["numeric_difference_envelope"]["root_pos"]["max_abs"] == 0.5
    assert result["physics_parity_authority"] is False
    with pytest.raises(FileExistsError):
        M.compare_records(isaac, mujoco, output)


def test_probe_wiring_precedes_the_policy_runner():
    train = (
        ROOT / "hope_training/whole_body_tracking/scripts/train.py"
    ).read_text(encoding="utf-8")
    hook = "if action_ball_full_mdp_fixed_action_probe_output is not None:"
    assert hook in train
    assert train.index(hook) < train.index("runner_type = (")
    assert "_run_action_ball_full_mdp_isaac_fixed_action_probe" in train
    assert "FULLMDP_ISAAC_FIXED_ACTION_PROBE_STARTED" in train
    genesis = "_consume_action_ball_full_mdp_fresh_genesis_reset("
    assert train.count(genesis) == 2
    assert train.rindex(genesis) < train.index(hook)
    assert train.rindex(genesis) < train.index("runner_type = (")
    mujoco = (
        ROOT / "scripts/probe_mujoco_full_mdp_h48_tape.py"
    ).read_text(encoding="utf-8")
    assert "cross-engine-probe" in mujoco
    assert "runner._verify_full_a_runtime_postimport(runtime)" in mujoco
