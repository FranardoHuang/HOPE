"""Safety and claim-boundary tests for the fixed-centre launcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hope_training.whole_body_tracking.mujoco_native.scripts import (
    launch_mujoco_fixed_center_diagnostic as launch,
)


def _authority_args(tmp_path: Path) -> list[str]:
    arguments = []
    for flag in (
        "plant",
        "robot-tape",
        "question",
        "selected-rubber-manifest",
        "mjcf",
    ):
        path = tmp_path / f"{flag}.json"
        path.write_bytes((flag + "\n").encode("ascii"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path_flag = "plant-contract" if flag == "plant" else flag
        sha_flag = (
            "expected-plant-sha256" if flag == "plant" else f"expected-{flag}-sha256"
        )
        arguments.extend([f"--{path_flag}", str(path), f"--{sha_flag}", digest])
    return arguments


def test_plan_is_read_only_and_labels_4096_unmeasured(tmp_path: Path, capsys) -> None:
    output = tmp_path / "must_not_exist"
    code = launch.main(
        [
            *_authority_args(tmp_path),
            "--output-dir",
            str(output),
            "--num-envs",
            "4096",
        ]
    )
    assert code == 0
    assert not output.exists()
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "plan"
    assert len(result["recipe_source_sha256"]) == 64
    assert result["outputs"]["launch_preparation"].endswith("launch_preparation.json")
    assert result["workload"]["num_envs_4096_plan_shape_supported"] is True
    assert result["workload"]["matched_4096_runtime_measured"] is False
    assert result["claims"]["physical_ball_parked_during_wait"] is False
    assert result["diagnostic_unauthorized"] is True
    assert result["formal_authorized"] is False


def test_execute_refuses_4096_before_creating_output(tmp_path: Path, capsys) -> None:
    output = tmp_path / "must_not_exist"
    code = launch.main(
        [
            *_authority_args(tmp_path),
            "--output-dir",
            str(output),
            "--num-envs",
            "4096",
            "--execute",
            "--confirm-diagnostic-unauthorized",
        ]
    )
    assert code == 2
    assert not output.exists()
    assert "4096 is unmeasured" in capsys.readouterr().err


def test_execute_requires_explicit_diagnostic_acknowledgement(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "must_not_exist"
    code = launch.main(
        [
            *_authority_args(tmp_path),
            "--output-dir",
            str(output),
            "--execute",
        ]
    )
    assert code == 2
    assert not output.exists()
    assert "--confirm-diagnostic-unauthorized" in capsys.readouterr().err
