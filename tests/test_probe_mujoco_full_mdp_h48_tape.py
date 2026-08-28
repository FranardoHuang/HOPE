from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_mujoco_full_mdp_h48_tape.py"
SPEC = importlib.util.spec_from_file_location("mujoco_h48_tape_probe", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_tracked_config_and_splitmix_tape_are_externally_recomputable():
    cfg, digest = M.load_config()
    payload, tape_digest = M.generate_action_bytes(cfg)
    state = cfg["action_tape"]["seed"]
    reference = bytearray()
    for _ in range(48 * 64 * 31):
        state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        value ^= value >> 31
        unit = (value >> 40) / float(1 << 24)
        reference.extend(struct.pack("<f", -0.02 + 0.04 * unit))
    assert digest == M.CONFIG_SHA256
    assert cfg["num_envs"] == 64 and cfg["num_ticks"] == 48
    assert len(payload) == 48 * 64 * 31 * 4
    assert tape_digest == "ea688a3a4a469255c9f3c39a9e2b1796b8b7603dc0fde3db0e5f906698a13425"
    assert payload == reference
    assert struct.unpack("<5f", payload[:20]) == pytest.approx((
        -0.017771663144230843, 0.0064319465309381485,
        0.019326383247971535, 0.010112445801496506,
        0.01256452314555645,
    ), rel=0.0, abs=0.0)
    values = np.frombuffer(payload, dtype="<f4")
    assert values.min() >= -0.02 and values.max() < 0.02


def test_config_loader_rejects_symlink_and_changed_bytes(tmp_path, monkeypatch):
    linked = tmp_path / "linked.json"
    linked.symlink_to(M.CONFIG)
    monkeypatch.setattr(M, "CONFIG", linked)
    with pytest.raises(M.ProbeError, match="canonical regular"):
        M.load_config()
    changed = tmp_path / "changed.json"
    cfg = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    cfg["scientific_scope"]["training_authorized"] = True
    changed.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(M, "CONFIG", changed.resolve())
    with pytest.raises(M.ProbeError, match="SHA differs"):
        M.load_config()


def _record(root: Path, *, delta: float = 0.0, discrete_flip: bool = False,
            wrong_done_shape: bool = False, fake_identity: bool = False):
    root.mkdir()
    cfg, config_sha = M.load_config()
    action_bytes, tape_sha = M.generate_action_bytes(cfg)
    arrays = {"actions": np.frombuffer(action_bytes, dtype="<f4").reshape(48, 64, 31).copy(),
        "initial_qpos": np.zeros((64, 3), dtype=np.float32),
        "initial_qvel": np.zeros((64, 3), dtype=np.float32),
        "initial_actor": np.zeros((64, M.ACTOR_WIDTH), dtype=np.float32),
        "initial_critic": np.zeros((64, M.CRITIC_WIDTH), dtype=np.float32),
        "reward_terms": np.zeros(
            (48, 64, M.REWARD_TERM_COUNT), dtype=np.float32
        ),
        "actor": np.zeros((48, 64, M.ACTOR_WIDTH), dtype=np.float32),
        "critic": np.zeros((48, 64, M.CRITIC_WIDTH), dtype=np.float32),
        "qpos": np.zeros((48, 64, 3), dtype=np.float32),
        "qvel": np.zeros((48, 64, 3), dtype=np.float32)}
    arrays["actor"][1, 1, 2] = delta
    for name in M.DISCRETE_FIELDS:
        arrays[name] = np.zeros((48, 64), dtype=np.int64)
    arrays["done"][0, 0] = int(discrete_flip)
    for event in M.STRATA_EVENTS.values():
        arrays["event__" + event] = np.zeros((48, 64), dtype=np.bool_)
    if wrong_done_shape:
        arrays["done"] = arrays["done"][:1]
    with (root / M.ARRAYS_NAME).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    payload = (root / M.ARRAYS_NAME).read_bytes()
    summary = {
        "schema_version": 2,
        "record_type": "mujoco_full_mdp_h48_fixed_tape_v2",
        "config_sha256": "c" * 64 if fake_identity else config_sha,
        "action_tape_sha256": "t" * 64 if fake_identity else tape_sha,
        "source": {"commit": root.name, "dirty": False},
        "observation": {
            "kind": M.OBSERVATION_KIND,
            "actor_width": M.ACTOR_WIDTH,
            "critic_width": M.CRITIC_WIDTH,
        },
        "natural_h48_strata": {"reveal": 999},
        "arrays_npz_sha256": hashlib.sha256(payload).hexdigest(),
    }
    M._write_json_x(root / M.SUMMARY_NAME, summary)


def test_compare_reports_measured_envelope_without_readiness_verdict(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _record(baseline)
    _record(candidate, delta=0.125, discrete_flip=True)
    output = tmp_path / "comparison.json"
    result = M._compare(baseline, candidate, output)
    assert result["same_exact_tape"] is True
    assert result["all_discrete_exact"] is False
    assert result["discrete_mismatch_cells"]["done"] == 1
    assert result["numeric_difference_envelope"]["actor"] == {
        "exact": False, "max_abs": 0.125,
        "mean_abs": pytest.approx(0.125 / (48 * 64 * M.ACTOR_WIDTH)),
    }
    assert result["missing_natural_strata"] == {
        name: "未测" for name in M.STRATA_EVENTS
    }
    assert "verdict" not in result and result["promotion_authority"] is False
    with pytest.raises(FileExistsError):
        M._compare(baseline, candidate, output)


def test_record_loader_detects_array_tamper(tmp_path):
    root = tmp_path / "record"
    _record(root)
    with np.load(root / M.ARRAYS_NAME, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["qpos"][0, 0, 0] = 1.0
    with (root / M.ARRAYS_NAME).open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(M.ProbeError, match="identity differs"):
        M._load_record(root)


def test_record_loader_binds_tracked_tape_shapes_and_raw_events(tmp_path):
    fake = tmp_path / "fake"
    _record(fake, fake_identity=True)
    with pytest.raises(M.ProbeError, match="identity differs"):
        M._load_record(fake)
    wrong = tmp_path / "wrong"
    _record(wrong, wrong_done_shape=True)
    with pytest.raises(M.ProbeError, match="shape differs: done"):
        M._load_record(wrong)


def test_cli_is_public_and_source_calls_real_full_a_boundary():
    shown = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True,
                           capture_output=True, check=False)
    assert shown.returncode == 0 and "probe" in shown.stdout and "compare" in shown.stdout
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("runner._bind_full_a_runtime") < source.index("import torch")
    assert "wait.FullMdpInitialWaitVecEnv" in source
    assert "full_a_mode=True" in source
    assert "_epoch_phase[" not in source and "_full_a_launch_rows" not in source
    assert M.REWARD_TERM_COUNT == 28
    assert M.REWARD_TERM_COUNT == len(M.reward_contract.MANAGER_NAMES)
    assert M.OBSERVATION_KIND == "action_ball_full_mdp_semantic_observation_v3"
    assert (M.ACTOR_WIDTH, M.CRITIC_WIDTH) == (215, 231)
