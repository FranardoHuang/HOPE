"""Dependency-light judge preflight tests for the binary zero-friction plant control."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "scripts" / "judge.sh"


def _run_judge(
    tmp_path: Path,
    friction,
    *,
    schema: int | None = 3,
    station_obs: bool = False,
    hard_actor_contract: str | None = None,
):
    run_dir = tmp_path / "agibot_a3_hope_virtualball" / "run"
    params = run_dir / "params"
    params.mkdir(parents=True)
    checkpoint = run_dir / "model_1.pt"
    checkpoint.touch()
    fh = tmp_path / "fh.npz"
    bh = tmp_path / "bh.npz"
    train_bank = tmp_path / "questions_train.npz"
    exam_bank = tmp_path / "questions_exam.npz"
    for path in (fh, bh, train_bank, exam_bank):
        path.touch()

    env = {
        "commands": {
            "motion": {
                "motion_file": [str(fh), str(bh)],
                "allow_legacy_link_origin_velocity": False,
            },
            "racket_target": {
                "strike_phase_per_clip": [0.4, 0.6],
                "question_bank": str(train_bank),
                "face_command": True,
                "face_command_pairing": "shared_plus_y",
            },
        },
        "face_command_obs": True,
        "station_obs": station_obs,
    }
    (params / "env.yaml").write_text(json.dumps(env), encoding="utf-8")
    if schema is not None:
        actor_contract = hard_actor_contract or (
            "deploy_parity_station181" if station_obs else "deploy_parity_face179"
        )
        (params / "training_contract.json").write_text(
            json.dumps(
                {
                    "schema_version": schema,
                    "joint_friction_coefficients": friction,
                    "actor_obs_contract": actor_contract,
                    "actor_obs_total_dim": {
                        "deploy_parity": 175,
                        "deploy_parity_face179": 179,
                        "deploy_parity_station181": 181,
                    }.get(actor_contract, -1),
                }
            ),
            encoding="utf-8",
        )

    return subprocess.run(
        [
            "bash",
            str(JUDGE),
            str(run_dir),
            str(checkpoint),
            "--dry-run",
            "--gpu",
            "0",
            "--task",
            "HOPEPingPongVirtualBall",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_judge_replays_only_the_all_zero_schema3_plant(tmp_path):
    zero = _run_judge(tmp_path / "zero", [0.0] * 31)
    assert zero.returncode == 0, zero.stdout
    assert "task.plant.zero_joint_friction=true" in zero.stdout
    assert "task.actor_obs_contract=deploy_parity_face179" in zero.stdout
    assert "task.actor_obs_contract=null" not in zero.stdout
    assert "31/31 exact zero" in zero.stdout

    nonzero = _run_judge(tmp_path / "nonzero", [0.1] * 31)
    assert nonzero.returncode == 0, nonzero.stdout
    assert "task.plant.zero_joint_friction=true" not in nonzero.stdout
    assert "31/31 non-zero; task default false" in nonzero.stdout


def test_judge_binds_station181_and_rejects_hard_contract_flag_disagreement(tmp_path):
    station = _run_judge(tmp_path / "station", [0.0] * 31, station_obs=True)
    assert station.returncode == 0, station.stdout
    assert "task.actor_obs_contract=deploy_parity_station181" in station.stdout

    mismatch = _run_judge(
        tmp_path / "mismatch",
        [0.0] * 31,
        hard_actor_contract="deploy_parity",
    )
    assert mismatch.returncode != 0
    assert "actor 与 env.yaml 观测开关不一致" in mismatch.stdout


@pytest.mark.parametrize(
    ("friction", "message"),
    [
        ([0.0] * 30, "31 维数组"),
        ([0.0] * 30 + [0.1], "混合 friction 向量"),
        ([0.1] * 30 + [float("nan")], "负数/NaN/Inf"),
    ],
)
def test_judge_rejects_malformed_or_mixed_schema3_friction(tmp_path, friction, message):
    result = _run_judge(tmp_path, friction)
    assert result.returncode != 0
    assert message in result.stdout


def test_judge_keeps_legacy_missing_or_schema2_contract_on_default_plant(tmp_path):
    missing = _run_judge(tmp_path / "missing", None, schema=None)
    assert missing.returncode == 0, missing.stdout
    assert "task.plant.zero_joint_friction=true" not in missing.stdout
    assert "task.actor_obs_contract=deploy_parity_face179" in missing.stdout
    assert "legacy/no schema-3 hard contract" in missing.stdout

    schema2 = _run_judge(tmp_path / "schema2", None, schema=2)
    assert schema2.returncode == 0, schema2.stdout
    assert "task.plant.zero_joint_friction=true" not in schema2.stdout
    assert "legacy schema-2 contract" in schema2.stdout
