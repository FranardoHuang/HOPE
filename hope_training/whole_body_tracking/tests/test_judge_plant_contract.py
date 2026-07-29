"""Dependency-light judge preflight tests for the binary zero-friction plant control."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

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
    motion_exact: bool = True,
    pairing: str = "shared_plus_y",
    diagnostic_unauthorized: object = False,
    checkpoint_lineage: object | None = None,
    checkpoint_schema: object | None = None,
):
    run_dir = tmp_path / "agibot_a3_hope_virtualball" / "run"
    params = run_dir / "params"
    params.mkdir(parents=True)
    checkpoint = run_dir / "model_1.pt"
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
                "face_command_pairing": pairing,
                **(
                    {"target_mode": "action_ball"}
                    if diagnostic_unauthorized is not False
                    else {}
                ),
            },
        },
        "face_command_obs": True,
        "station_obs": station_obs,
    }
    (params / "env.yaml").write_text(json.dumps(env), encoding="utf-8")
    if schema is not None:
        actor_contract = hard_actor_contract or (
            "action_ball_n2"
            if diagnostic_unauthorized is not False
            else (
                "deploy_parity_station181"
                if station_obs
                else "deploy_parity_face179"
            )
        )
        hard_contract = {
            "schema_version": schema,
            "joint_friction_coefficients": friction,
            "actor_obs_contract": actor_contract,
            "actor_obs_total_dim": {
                "deploy_parity": 175,
                "deploy_parity_face179": 179,
                "deploy_parity_station181": 181,
                "action_ball_n2": 183,
            }.get(actor_contract, -1),
            "motion_kinematics_exact": motion_exact,
            "face_command_pairing": pairing,
            **(
                {"target_mode": "action_ball"}
                if diagnostic_unauthorized is not False
                else {}
            ),
        }
        if diagnostic_unauthorized is not False:
            hard_contract["action_ball_training"] = {
                "schema_version": 1,
                "authorization": {
                    "diagnostic_unauthorized": diagnostic_unauthorized,
                    "formal_evidence_prohibited": True,
                    "curriculum_promotion_prohibited": True,
                    "exact_export_prohibited": True,
                    "formal_judge_prohibited": True,
                },
                "runtime": {
                    "diagnostic_unauthorized": True,
                    "evaluator_authority": {
                        "diagnostic_unauthorized": True,
                        "formal_authority_available": False,
                        "formal_launch_requires_code_pinned_receipt": True,
                        "runtime_or_manifest_may_self_authorize": False,
                        "authority_binding": {"kind": "diagnostic"},
                        "authority_state_owner_sha256": "a" * 64,
                    },
                },
                "motion_admission": {
                    "diagnostic_unauthorized": True,
                    "training_authorized": False,
                },
            }
        (params / "training_contract.json").write_text(
            json.dumps(hard_contract),
            encoding="utf-8",
        )

    infos = {}
    if schema == 3:
        bound_schema = (
            schema if checkpoint_schema is None else checkpoint_schema
        )
        lineage = checkpoint_lineage
        if lineage is None:
            lineage = 0 if diagnostic_unauthorized is True else 1
        contract_bytes = (params / "training_contract.json").read_bytes()
        infos = {
            "training_contract_schema_version": bound_schema,
            "training_contract_sha256": hashlib.sha256(
                contract_bytes
            ).hexdigest(),
            "training_contract_lineage_exact": lineage,
        }
    checkpoint.write_text(json.dumps({"infos": infos}), encoding="utf-8")

    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(
        "import json\n"
        "def load(path, **kwargs):\n"
        "    with open(path, encoding='utf-8') as stream:\n"
        "        return json.load(stream)\n",
        encoding="utf-8",
    )
    process_env = dict(os.environ)
    process_env["JUDGE_CHECKPOINT_PYTHON"] = sys.executable
    process_env["PYTHONPATH"] = (
        str(fake_modules)
        + os.pathsep
        + process_env.get("PYTHONPATH", "")
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
        env=process_env,
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
    assert "--allow-inexact-contract" not in zero.stdout
    assert "/workspace/franco/env.sh" not in zero.stdout
    assert "source /workspace/hope_isaac_venv/bin/activate" in zero.stdout
    assert "setup_train_env.sh" in zero.stdout

    nonzero = _run_judge(tmp_path / "nonzero", [0.1] * 31)
    assert nonzero.returncode == 0, nonzero.stdout
    assert "task.plant.zero_joint_friction=true" not in nonzero.stdout
    assert "31/31 non-zero; task default false" in nonzero.stdout
    assert "--allow-inexact-contract" in nonzero.stdout


def test_judge_adds_diagnostic_escape_for_inexact_motion_pairing_or_plant(tmp_path):
    inexact_motion = _run_judge(
        tmp_path / "motion", [0.1] * 31, motion_exact=False
    )
    assert inexact_motion.returncode == 0, inexact_motion.stdout
    assert "--allow-inexact-contract" in inexact_motion.stdout
    assert "schema-3 diagnostic lineage" in inexact_motion.stdout

    legacy_pairing = _run_judge(
        tmp_path / "pairing", [0.1] * 31, pairing="legacy_signed_vs_A"
    )
    assert legacy_pairing.returncode == 0, legacy_pairing.stdout
    assert "--allow-inexact-contract" in legacy_pairing.stdout

    exact = _run_judge(tmp_path / "exact", [0.0] * 31)
    assert exact.returncode == 0, exact.stdout
    assert "--allow-inexact-contract" not in exact.stdout


def test_judge_action_ball_diagnostic_is_nonformal_and_requires_inexact_escape(
    tmp_path,
):
    diagnostic = _run_judge(
        tmp_path,
        [0.0] * 31,
        diagnostic_unauthorized=True,
    )
    assert diagnostic.returncode == 0, diagnostic.stdout
    assert "--allow-inexact-contract" in diagnostic.stdout
    assert "action-ball diagnostic_unauthorized" in diagnostic.stdout
    assert "non-formal/non-bookable" in diagnostic.stdout
    assert "formal candidate" not in diagnostic.stdout


def test_judge_lineage_zero_formal_sidecar_is_nonformal_and_inexact(
    tmp_path,
):
    result = _run_judge(
        tmp_path,
        [0.0] * 31,
        checkpoint_lineage=0,
    )
    assert result.returncode == 0, result.stdout
    assert "--allow-inexact-contract" in result.stdout
    assert "checkpoint lineage=0" in result.stdout
    assert "non-formal/non-bookable" in result.stdout
    assert "formal candidate" not in result.stdout


def test_judge_rejects_diagnostic_contract_with_lineage_one(tmp_path):
    result = _run_judge(
        tmp_path,
        [0.0] * 31,
        diagnostic_unauthorized=True,
        checkpoint_lineage=1,
    )
    assert result.returncode != 0
    assert "provenance laundering" in result.stdout


@pytest.mark.parametrize("invalid", [1.0, "true", 1])
def test_judge_rejects_non_boolean_action_ball_diagnostic_brand(
    tmp_path, invalid
):
    result = _run_judge(
        tmp_path,
        [0.0] * 31,
        diagnostic_unauthorized=invalid,
    )
    assert result.returncode != 0
    assert "必须是 exact bool" in result.stdout


@pytest.mark.parametrize("invalid", [1.0, "1", True, False])
def test_judge_rejects_non_integer_checkpoint_lineage(tmp_path, invalid):
    result = _run_judge(
        tmp_path,
        [0.0] * 31,
        checkpoint_lineage=invalid,
    )
    assert result.returncode != 0
    assert "必须是 plain integer 0/1" in result.stdout


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
    assert "--allow-inexact-contract" in missing.stdout

    schema2 = _run_judge(tmp_path / "schema2", None, schema=2)
    assert schema2.returncode == 0, schema2.stdout
    assert "task.plant.zero_joint_friction=true" not in schema2.stdout
    assert "legacy schema-2 contract" in schema2.stdout
    assert "--allow-inexact-contract" in schema2.stdout


def test_judge_preflights_both_mjeval_graph_and_runtime_dependencies_before_export():
    script = JUDGE.read_text(encoding="utf-8")
    dep_check = script.index("import onnx\nimport onnxruntime")
    kit_lock = script.index('exec 8>"$JUDGE_KIT_BOOT_LOCK"')
    export_call = script.index('eval "$EXPORT_CMD"')
    assert dep_check < kit_lock < export_call
    assert "mjeval venv 缺正式判卷依赖" in script
