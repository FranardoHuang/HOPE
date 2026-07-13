from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "configs" / "phase1_signed_face_boot_root_cause_results_20260714.json"
PREREG = ROOT / "configs" / "phase1_signed_face_boot_diagnostic_prereg_20260714.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readonly_ledger_binds_all_outer_evidence_receipts() -> None:
    result = _load(RESULT)
    assert result["status"] == "failure_boundary_narrowed_root_cause_not_proven_retry_forbidden"
    assert result["human_owner"] == "Franco"
    assert result["executor"] == "Codex"
    expected = {
        "diagnosis_report": (5955, "b54cb06a50bfa5f0994b1768beb995577b03a360eb4dfaefca13959c1c2d76af"),
        "inventory": (157048, "b18935129364cb342a4d3989caf56821bc0f5cb3dbae79c9a409e26d0e21cc1d"),
        "exact_archive": (8509440, "29dabc9e23fc7f4d4f1713a75bb9bc3be20009b19f90f51838a439d65e0283a6"),
        "system_snapshot": (60056, "02b78e2d4db982145e57d9bcbe82768799b2756b21636a031052c7a1b30d1e25"),
    }
    for key, receipt in expected.items():
        row = result["evidence_receipts"][key]
        assert (row["bytes"], row["sha256"]) == receipt
        assert row["local_restore_path"].startswith(
            "vendor_assets/phase1_signed_face_boot_root_cause_20260714/pod1_v6v8_"
        )
    assert result["evidence_receipts"]["exact_archive"]["entry_count"] == 47
    assert result["collection_contract"] == {
        "successful_read_only_ssh_connections": 3,
        "remote_file_written": False,
        "remote_signal_sent": False,
        "process_started_or_restarted": False,
        "hardware_command_sent": False,
        "audit_observed_utc": ["2026-07-13T18:18:10Z", "2026-07-13T18:22:48Z"],
    }


def test_local_postmortem_bytes_match_ledger_when_restored() -> None:
    receipts = _load(RESULT)["evidence_receipts"]
    restored = [ROOT / row["local_restore_path"] for row in receipts.values()]
    if not all(path.is_file() for path in restored):
        pytest.skip("ignored Pod1 postmortem evidence is not restored on this host")
    for row in receipts.values():
        path = ROOT / row["local_restore_path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]
    archive = ROOT / receipts["exact_archive"]["local_restore_path"]
    with tarfile.open(archive, mode="r:") as handle:
        assert len(handle.getmembers()) == receipts["exact_archive"]["entry_count"] == 47


def test_fact_inference_unknown_boundaries_are_explicit() -> None:
    result = _load(RESULT)
    assert result["facts"]["classification"] == "observed_or_byte_recomputed"
    assert result["inferences"]["classification"] == "reasoned_not_directly_observed"
    assert result["unknowns"]["classification"] == "not_observed_or_not_identified"
    assert len(result["unknowns"]["items"]) == 5
    for key, row in result["inferences"].items():
        if key == "classification":
            continue
        assert row["root_cause_proven"] is False
    assert result["decision"]["boot_root_cause_closed"] is False


def test_two_d_failures_share_exact_asset_and_same_boundary() -> None:
    facts = _load(RESULT)["facts"]
    table = facts["table_asset"]
    assert table["bytes_each"] == 683433
    assert table["byte_identical"] is True
    assert table["v6_sha256"] == table["v8_sha256"] == (
        "c6fc99a804d60ea6db90931bb8edb3d324ed63fec767ffe6ced424ccf4ad2996"
    )

    d = facts["d_comparison"]
    assert d["normalized_training_argv_equal"] is True
    assert d["v6_argv_token_count"] + 1 == d["v8_argv_token_count"]
    assert d["only_versioned_differences"] == ["run_name", "v8_training_launch_claim_sha256"]
    for version in ("v6", "v8"):
        assert d[version]["pid"] == d[version]["pgid"]
        assert d[version]["boot_timeout_s"] == 900
        assert d[version]["kit_log"]["last_semantic_operation"] == "load_table_usd"
        assert d[version]["kit_log"]["physx_context_seen"] is False
    assert d["v8"]["exit_code"] == 124
    assert d["v8"]["signal_scope"] == "exact_pgid"
    assert d["v8"]["automatic_retry_forbidden"] is True


def test_adjacent_c_controls_cross_table_to_physx_quickly() -> None:
    controls = _load(RESULT)["facts"]["adjacent_c_boundary_controls"]
    assert controls["v6"]["table_to_physx_ms"] == (
        controls["v6"]["physx_context_elapsed_ms"] - controls["v6"]["table_load_elapsed_ms"]
    ) == 2339
    assert controls["v8"]["table_to_physx_ms"] == (
        controls["v8"]["physx_context_elapsed_ms"] - controls["v8"]["table_load_elapsed_ms"]
    ) == 3031
    assert controls["v6"]["last_iteration"] == controls["v8"]["last_iteration"] == "24/25"
    assert controls["v6"]["d_started_while_c_still_active"] is True
    assert controls["v8"]["serial_nonconcurrent_d_reproduction"] is True
    assert controls["v8"]["d_start_after_c_clean_shutdown_s"] == 44


def test_capacity_and_shm_observations_do_not_overclaim_cause() -> None:
    facts = _load(RESULT)["facts"]
    capacity = facts["postmortem_capacity_snapshot"]
    assert capacity["temporal_scope"] == "postmortem_not_historical_failure_instant"
    assert capacity["trainer_worker_judge_kit_match_count"] == 0
    assert capacity["gpu_utilization_percent_each"] == [0, 0, 0]
    assert capacity["gpu_memory_mib_each"] == [0, 0, 0]
    assert capacity["host_memory_available_gib"] == 976
    assert capacity["workspace_free_gib"] == 362
    assert capacity["dev_shm_free_gib"] == 201
    assert facts["carbonite_shared_memory"]["causal_conclusion"] == "unknown_correlation_only"
    assert facts["kernel_evidence"]["dmesg_readable"] is False
    assert facts["kernel_evidence"]["absence_proves_no_historical_kernel_event"] is False


def test_diagnostic_prereg_is_minimal_factorial_design_only() -> None:
    prereg = _load(PREREG)
    assert prereg["status"] == "design_only_not_authorized_not_materialized"
    assert prereg["design"]["factorial"] == "2x2"
    factors = prereg["design"]["factors"]
    assert len(factors["launch_ordinal"]) == 2
    assert len(factors["process_namespace_and_cleanup"]) == 2
    cells = prereg["design"]["cells"]
    assert [cell["cell_id"] for cell in cells] == ["O1_H", "O4_H", "O1_I", "O4_I"]
    assert {(cell["launch_ordinal"], cell["ipc_namespace"]) for cell in cells} == {
        (1, "host"),
        (4, "host"),
        (1, "private_per_process"),
        (4, "private_per_process"),
    }
    assert all(cell["host_shared_memory_mutation_allowed"] is False for cell in cells)
    scope = prereg["design"]["probe_scope"]
    assert scope["scene_boot_only"] is True
    assert scope["learning_iteration_allowed"] is False
    assert scope["checkpoint_write_allowed"] is False
    assert scope["host_carbonite_unlink_forbidden"] is True
    assert scope["broad_signal_forbidden"] is True
    assert len(prereg["design"]["decision_rules"]) == 5
    assert set(prereg["authorization"].values()) == {False}


def test_boot_ledger_and_prereg_do_not_grant_downstream_authority() -> None:
    result = _load(RESULT)
    for key in (
        "automatic_retry_authorized",
        "recipe_identical_retry_authorized",
        "training_authorized",
        "l1_activation_materialized",
        "l2_authorized",
        "judge_authorized",
        "formal_score_authorized",
        "deployment_authorized",
        "hardware_authorized",
    ):
        assert result["decision"][key] is False
    assert result["decision"]["diagnostic_prereg_path"] == PREREG.relative_to(ROOT).as_posix()
