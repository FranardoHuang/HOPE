from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_std_sidecar.py"


def _write_checkpoint(path: Path, normalizer_std: torch.Tensor) -> None:
    torch.save(
        {
            "model_state_dict": {"std": torch.ones(31)},
            "obs_norm_state_dict": {
                "_mean": torch.zeros_like(normalizer_std),
                "_std": normalizer_std,
                "count": torch.tensor(1234),
            },
        },
        path,
    )


def test_zero_normalizer_std_is_preserved_when_epsilon_protects_divisor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_0.pt"
    _write_checkpoint(checkpoint, torch.tensor([0.0, 0.5, 0.0, 1.0]))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(checkpoint)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = np.load(tmp_path / "exported" / "obs_norm.npz")
    np.testing.assert_array_equal(payload["std"], np.asarray([0.0, 0.5, 0.0, 1.0], dtype=np.float32))
    assert float(payload["eps"]) == np.float32(1e-2)
    assert "zeros=2" in result.stdout


def test_negative_normalizer_std_remains_fatal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_0.pt"
    _write_checkpoint(checkpoint, torch.tensor([0.0, -0.1, 1.0]))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(checkpoint)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "std>=0" in result.stderr
