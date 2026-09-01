import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "render_mujoco_fullmdp_policy_trace.py"


def _load():
    spec = importlib.util.spec_from_file_location("render_policy_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_root(tmp_path):
    root = tmp_path / "trace"
    root.mkdir(parents=True)
    qpos = np.zeros((4, 7), dtype=np.float32)
    qpos[:, 3] = 1.0
    np.savez_compressed(root / "controller_trace.npz", qpos_world0=qpos)
    (root / "runtime.mjb").write_bytes(b"current-runtime")
    summary = {
        "kind": "action_ball_mujoco_full_mdp_controller_trace_v2",
        "schema_version": 2,
        "diagnostic_unauthorized": True,
        "checkpoint_authority": False,
        "checkpoint_path": "/tmp/model.pt",
        "checkpoint_sha256": "a" * 64,
        "policy_steps": 3,
        "trace_npz": "controller_trace.npz",
        "trace_npz_sha256": _sha(root / "controller_trace.npz"),
        "qpos_world0_shape": [4, 7],
        "runtime_mjb": {
            "relative_locator": "runtime.mjb",
            "sha256": _sha(root / "runtime.mjb"),
            "size_bytes": (root / "runtime.mjb").stat().st_size,
        },
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def test_loads_only_current_finite_exact_trace(tmp_path):
    module = _load()
    root = _trace_root(tmp_path)
    summary, qpos, mjb = module._load_trace(root)
    assert summary["schema_version"] == 2
    assert qpos.shape == (4, 7)
    assert mjb == root / "runtime.mjb"

    summary_path = root / "summary.json"
    payload = json.loads(summary_path.read_text())
    payload["kind"] = "action_ball_mujoco_full_mdp_controller_trace_v1"
    summary_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract differs"):
        module._load_trace(root)


def test_rejects_qpos_or_runtime_byte_drift(tmp_path):
    module = _load()
    root = _trace_root(tmp_path)
    with np.load(root / "controller_trace.npz", allow_pickle=False) as archive:
        qpos = archive["qpos_world0"].copy()
    qpos[0, 0] = np.nan
    np.savez_compressed(root / "controller_trace.npz", qpos_world0=qpos)
    payload = json.loads((root / "summary.json").read_text())
    payload["trace_npz_sha256"] = _sha(root / "controller_trace.npz")
    (root / "summary.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="qpos contract differs"):
        module._load_trace(root)

    root = _trace_root(tmp_path / "peer")
    (root / "runtime.mjb").write_bytes(b"drift")
    with pytest.raises(ValueError, match="bytes differ"):
        module._load_trace(root)
