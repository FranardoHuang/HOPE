"""Numerical ONNX smoke for the native exporter's baked observation normalizer.

This test needs the RunPod/Isaac environment (torch, onnx, onnxruntime, isaaclab_rl). The local
dependency-light contract test remains in ``test_export_obs_norm_contract.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from whole_body_tracking.utils.exporter import _OnnxMotionPolicyExporter


class _FixedNormalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([1.0, -2.0]))
        self.register_buffer("scale", torch.tensor([2.0, 4.0]))

    def forward(self, x):
        return (x - self.mean) / self.scale


def _minimal_exporter(normalizer):
    exporter = _OnnxMotionPolicyExporter.__new__(_OnnxMotionPolicyExporter)
    torch.nn.Module.__init__(exporter)
    actor = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        actor[0].weight.copy_(torch.tensor([[3.0, -5.0]]))
    exporter.actor = actor
    exporter.normalizer = normalizer
    exporter.verbose = False
    exporter.joint_pos = torch.zeros(1, 1)
    exporter.joint_vel = torch.zeros(1, 1)
    exporter.body_pos_w = torch.zeros(1, 1, 3)
    exporter.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    exporter.body_lin_vel_w = torch.zeros(1, 1, 3)
    exporter.body_ang_vel_w = torch.zeros(1, 1, 3)
    exporter.time_step_total = 1
    return exporter, actor


@pytest.mark.parametrize("baked", [False, True])
def test_native_onnx_graph_matches_exactly_one_normalization(tmp_path, baked):
    normalizer = _FixedNormalizer() if baked else torch.nn.Identity()
    exporter, actor = _minimal_exporter(normalizer)
    exporter.export(str(tmp_path), "policy.onnx")

    obs = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
    with torch.no_grad():
        expected = actor(normalizer(obs)).numpy()
        raw_actor = actor(obs).numpy()
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(
        ["actions"],
        {"obs": obs.numpy(), "time_step": np.zeros((1, 1), dtype=np.float32)},
    )[0]

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    if baked:
        assert not np.allclose(got, raw_actor), "normalizer was declared baked but absent from graph"
