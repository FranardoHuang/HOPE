from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase1_task_revision_0p5_exam.py"
QUEUE = ROOT / "configs" / "phase1_task_revision_supercombo_20260716.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("task_revision_0p5_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E = _load_module()


def test_build_plan_binds_registered_checkpoint_paper_and_inexact_lane():
    plan = E.build_plan(
        QUEUE,
        job_id="taskrev_p1_core_high_noise",
        milestone=1800,
        eval_gpu=2,
    )

    assert plan["pod"] == "pod1"
    assert plan["training_gpu"] == 1
    assert plan["eval_gpu"] == 2
    assert plan["milestone_offset_from_parent"] == 200
    assert plan["expected_claim_content_sha256"] == (
        "1dd91278489d694f21aa02f3fe4be75f744a10dfe538d3e48f393e59dbb1894f"
    )
    assert plan["paper"] == {
        "path": (
            "/workspace/codexschema/phase1_task_revision_supercombo_20260716/"
            "papers/timing_exam_0p5_k100.schedule.json"
        ),
        "file_sha256": E.PAPER_FILE_SHA256,
        "semantic_sha256": E.PAPER_SEMANTIC_SHA256,
    }
    assert plan["formal_evidence_eligible"] is False
    assert plan["evaluation_contract_exact"] is False
    assert plan["source_closure"]["evaluator"] == (
        "67300ba2faae0f3443496219f1c6cf3fcc16afa182b45e6f95d4fbb82c60c094"
    )


@pytest.mark.parametrize("milestone", [1, 1799, 1801, 999999])
def test_build_plan_rejects_unregistered_absolute_milestone(milestone):
    with pytest.raises(E.ExamError, match="not a registered absolute milestone"):
        E.build_plan(
            QUEUE,
            job_id="taskrev_p1_core_high_noise",
            milestone=milestone,
            eval_gpu=2,
        )


def test_build_plan_rejects_blocked_transport_cell_and_unknown_gpu():
    with pytest.raises(Exception, match="NO-LAUNCH|launch"):
        E.build_plan(
            QUEUE,
            job_id="taskrev_p1_core_low_noise",
            milestone=1800,
            eval_gpu=2,
        )
    with pytest.raises(E.ExamError, match="not on pod1"):
        E.build_plan(
            QUEUE,
            job_id="taskrev_p1_core_high_noise",
            milestone=1800,
            eval_gpu=99,
        )


def test_default_cli_is_dry_run_and_never_calls_execute(monkeypatch, capsys):
    def forbidden(_plan):
        raise AssertionError("dry-run attempted SSH execution")

    monkeypatch.setattr(E, "execute", forbidden)
    rc = E.main(
        [
            "--queue",
            str(QUEUE),
            "--job-id",
            "taskrev_p1_core_high_noise",
            "--milestone",
            "1800",
            "--eval-gpu",
            "2",
        ]
    )
    assert rc == 0
    value = json.loads(capsys.readouterr().out)
    assert value["mode"] == "inspect"
    assert value["dry_run"] is True


def test_execute_requires_exact_confirmation_before_ssh(monkeypatch, capsys):
    def forbidden(_plan):
        raise AssertionError("bad confirmation reached SSH execution")

    monkeypatch.setattr(E, "execute", forbidden)
    rc = E.main(
        [
            "--queue",
            str(QUEUE),
            "--job-id",
            "taskrev_p1_core_high_noise",
            "--milestone",
            "1800",
            "--eval-gpu",
            "2",
            "--execute",
            "--confirm",
            "WRONG",
        ]
    )
    assert rc == 2
    assert "confirmation token mismatch" in capsys.readouterr().err


def test_remote_consumer_is_single_attempt_locked_inexact_and_signal_free():
    source = E.REMOTE_PROGRAM
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert '"+allow_inexact_contract=true"' in source
    assert 'row.get("tts_ticks") != 25' in source
    assert 'len(paper.get("rows", [])) != 100' in source
    assert '"formal_evidence_eligible": False' in source
    assert '"evaluation_contract_exact": False' in source
    assert '"trainer_or_robot_signals": []' in source
    assert "os.kill(" not in source
    assert "os.killpg(" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "signal." not in source
    assert "timeout=" not in source
    assert "for attempt in range" not in source


def test_remote_command_is_one_embedded_python_invocation():
    plan = E.build_plan(
        QUEUE,
        job_id="taskrev_p1_core_high_noise",
        milestone=1800,
        eval_gpu=2,
    )
    command = E._remote_command(plan)
    assert command.startswith("/workspace/hope_isaac_venv/bin/python -B -c ")
    assert "ssh" not in command
    assert E.REMOTE_PROGRAM not in command
