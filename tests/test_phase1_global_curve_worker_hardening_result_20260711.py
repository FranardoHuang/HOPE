from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "configs" / "phase1_global_curve_worker_hardening_result_20260711.json"
CONFIG = ROOT / "configs" / "phase1_global_curve_worker_hardening_20260711.json"
TOOL = ROOT / "scripts" / "replace_phase1_global_curve_workers_20260711.py"
WORKER = (
    ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "scripts"
    / "phase1_checkpoint_curve_worker.py"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_global_hardening_result_binds_final_tools_and_exact_signal_scope():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "complete_six_global_curve_workers_hardened"
    assert result["config"]["sha256"] == _sha(CONFIG)
    assert result["tool"]["sha256"] == _sha(TOOL)
    assert result["hardened_worker"]["sha256"] == _sha(WORKER)
    signal = result["signal_audit"]
    assert len(signal["exact_worker_pgids_only"]) == 6
    assert len(set(signal["exact_worker_pgids_only"])) == 6
    assert signal["sigkill_used"] is False
    assert signal["trainer_pgids_signalled"] == []
    assert signal["judge_pgids_signalled"] == []
    assert signal["real_robot_commands"] is False


def test_each_queue_has_unique_new_pgid_and_content_addressed_correction():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    queues = [
        row
        for pod in result["pods"].values()
        for row in pod["queues"].values()
    ]
    assert len(queues) == 6
    assert len({row["legacy_pgid"] for row in queues}) == 6
    assert len({row["hardened_pgid"] for row in queues}) == 6
    assert not ({row["legacy_pgid"] for row in queues} & {row["hardened_pgid"] for row in queues})
    for row in queues:
        for key in (
            "legacy_manifest_sha256",
            "hardened_manifest_sha256",
            "correction_sha256",
            "launch_sha256",
        ):
            assert len(row[key]) == 64
    states = {
        name: digest
        for row in queues
        for name, digest in row["rejudged_states"].items()
    }
    assert len(states) == 5
    assert all(len(digest) == 64 for digest in states.values())
    assert result["legacy_completed_job_policy"]["hard_reuse_allowed"] is False
    assert result["legacy_completed_job_policy"]["hardened_rejudged_jobs"] == 5
    assert result["postcondition"]["all_rejudged_states_bind_manifest_job_and_job_contract_sha256"] is True
