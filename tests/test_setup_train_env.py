from __future__ import annotations

import os
from pathlib import Path
import subprocess
import hashlib


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "hope_training/whole_body_tracking/setup_train_env.sh"
CONSTRAINTS = (
    REPO_ROOT / "configs/action_ball_isaac51_external_venv_constraints_20260829.txt"
)


def _source(command: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            f"source {SETUP!s}; status=$?; "
            f"if [ $status -ne 0 ]; then exit $status; fi; {command}",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_setup_refuses_ambient_path_discovery(tmp_path: Path) -> None:
    result = _source(
        ":",
        env={"PATH": os.environ["PATH"], "WANDB_DIR": str(tmp_path / "wandb")},
    )
    assert result.returncode != 0
    assert "set HOPE_ISAAC_PYTHON explicitly" in result.stderr


def test_setup_uses_only_explicit_runtime_identity(tmp_path: Path) -> None:
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "source").mkdir(parents=True)
    venv_site = tmp_path / "site-packages"
    venv_site.mkdir()
    result = _source(
        'printf "%s\\n%s\\n%s\\n" "$HOPE_ISAAC_PYTHON" "$HOPE_ISAACLAB_ROOT" "$HOPE_WBT_PYTHONPATH"',
        env={
            "PATH": os.environ["PATH"],
            "WANDB_DIR": str(tmp_path / "wandb"),
            "HOPE_ISAAC_PYTHON": "/usr/bin/python3",
            "HOPE_ISAACLAB_ROOT": str(isaaclab),
            "HOPE_ISAAC_VENV_SITE": str(venv_site),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "/usr/bin/python3" in result.stdout
    assert str(isaaclab) in result.stdout
    assert str(venv_site) in result.stdout


def test_isaac51_external_constraints_are_exact_and_complete() -> None:
    payload = CONSTRAINTS.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "6b1c9a34cd3ac40a970c5bf53330069b420f5f993d298b34ec72ff55ccea81e9"
    )
    rows = {
        line.split("==", 1)[0].lower().replace("_", "-"): line
        for line in payload.decode().splitlines()
        if line and not line.startswith("#")
    }
    assert len(rows) == 83
    assert rows["torch"] == "torch==2.7.0+cu128"
    assert rows["rsl-rl-lib"] == "rsl-rl-lib==3.1.2"
    assert rows["tensordict"] == "tensordict==0.10.0"
    assert rows["torchrl"] == "torchrl==0.10.1"
    assert rows["numpy"] == "numpy==1.26.4"
    assert rows["packaging"] == "packaging==23.2"
