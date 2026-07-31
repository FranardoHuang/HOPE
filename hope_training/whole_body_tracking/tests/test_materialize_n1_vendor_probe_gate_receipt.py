from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_n1_vendor_probe_gate_receipt.py"
)
SPEC = importlib.util.spec_from_file_location("vendor_probe_gate_materializer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class _FakeBoolTensor:
    def __init__(self, value: bool):
        self._value = value

    def all(self):
        return self

    def any(self):
        return self

    def item(self):
        return self._value


class _FakeTensor:
    def __init__(self, values, shape):
        self.values = list(values)
        self.shape = tuple(shape)

    def numel(self):
        return len(self.values)

    def item(self):
        if len(self.values) != 1:
            raise ValueError("not scalar")
        return self.values[0]

    def is_floating_point(self):
        return True

    def __lt__(self, value):
        return _FakeBoolTensor(any(item < value for item in self.values))


class _FakeTorch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value):
        return _FakeBoolTensor(
            all(math.isfinite(float(item)) for item in value.values)
        )


def _behavior_records(stage: str) -> list[dict]:
    updates = M.EXPECTED_STAGES[stage]["max_iterations"]
    result = []
    for update in range(updates):
        counters = {
            "physical_fall_count": 0,
            "pre_strike_physical_fall_count": 0,
            "post_strike_physical_fall_count": 0,
            "terminal_reset_count": 0,
            "timeout_reset_count": 0,
            "non_physical_terminal_reset_count": 0,
            "ready_nonfinite_value_count": 0,
            "strike_opportunity_count": 1,
            "swing_start_count": 1,
            "swing_outcome_count": 1,
            "reference_guard_union_count": 0,
            "reference_guard_reference_only_count": 0,
            "reference_guard_reference_and_hard_count": 0,
            "reference_guard_hard_without_snapshot_count": 0,
            M._ENTRY_COUNT: 1,
            M._ENTRY_NONFINITE: 0,
            **{key: 0 for key in M._ENTRY_BUCKETS},
            "termination_reason_joint_actual_forbidden_count": 0,
            "termination_reason_joint_qdes_forbidden_count": 0,
            "termination_reason_robot_hit_table_count": 0,
        }
        counters[M._ENTRY_BUCKETS[0]] = 1
        result.append(
            {
                "event": "hope_exact_behavior_update",
                "schema_version": 1,
                "ppo_update": update,
                "counters": counters,
            }
        )
    return result


def _joint_records(stage: str, *, actual_hard: int = 0) -> list[dict]:
    updates = M.EXPECTED_STAGES[stage]["max_iterations"]
    return [
        {
            "event": "hope_joint_safety_diagnostic_compact_update",
            "schema_version": 1,
            "status": "diagnostic_compact_optimizer_committed_and_ledger_acknowledged",
            "ppo_update": update,
            "num_envs": M.EXPECTED_STAGES[stage]["num_envs"],
            "policy_step_count": M.ROLLOUT_STEPS_PER_UPDATE,
            "minimum_hard_gap_rad": 0.1,
            "counter_totals": {
                "actual_hard_edge_events": actual_hard if update == 0 else 0,
                "qdes_events": 0,
                "policy_crossing_events": 0,
                "substep_crossing_events": 0,
                "policy_steps": M.EXPECTED_STAGES[stage]["num_envs"] * 24,
                "complete_policy_steps": M.EXPECTED_STAGES[stage]["num_envs"] * 24,
            },
        }
        for update in range(updates)
    ]


def test_behavior_gate_uses_bounded_rates_and_requires_reachability() -> None:
    stage = "probe"
    result = M._validate_behavior(
        _behavior_records(stage),
        stage=stage,
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert result["reachability_and_failure_rates"]["pass"] is True
    assert result["strike_window_entry_conservation"]["matches"] is True

    no_strike = _behavior_records(stage)
    for row in no_strike:
        row["counters"]["strike_opportunity_count"] = 0
    with pytest.raises(M.ReceiptRefused, match="reachability/rate"):
        M._validate_behavior(
            no_strike,
            stage=stage,
            updates=M.EXPECTED_STAGES[stage]["max_iterations"],
            num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
        )

    no_entry = _behavior_records(stage)
    for row in no_entry:
        row["counters"][M._ENTRY_COUNT] = 0
        row["counters"][M._ENTRY_BUCKETS[0]] = 0
    with pytest.raises(M.ReceiptRefused, match="entry=0"):
        M._validate_behavior(
            no_entry,
            stage=stage,
            updates=M.EXPECTED_STAGES[stage]["max_iterations"],
            num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
        )


def test_old_failed_probe_actual_hard_cannot_mint_pass() -> None:
    stage = "probe"
    # The historical 89082 probe had 14,086 actual-hard events.  One event is
    # already enough to prove that a PASS receipt cannot be materialized.
    with pytest.raises(M.ReceiptRefused, match="zero gate"):
        M._validate_joint_safety(
            _joint_records(stage, actual_hard=14_086),
            [],
            updates=M.EXPECTED_STAGES[stage]["max_iterations"],
            num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
        )

    predicted_only = _joint_records(stage)
    predicted_only[0]["counter_totals"]["policy_crossing_events"] = 17
    predicted_only[0]["counter_totals"]["substep_crossing_events"] = 9
    accepted = M._validate_joint_safety(
        predicted_only,
        [],
        updates=M.EXPECTED_STAGES[stage]["max_iterations"],
        num_envs=M.EXPECTED_STAGES[stage]["num_envs"],
    )
    assert accepted["aggregate_counter_totals"]["policy_crossing_events"] == 17


def test_push_timer_is_longer_than_strict_upper_bound() -> None:
    duration = (
        M.EXPECTED_STAGES["push_evidence"]["max_iterations"]
        * M.ROLLOUT_STEPS_PER_UPDATE
        * M.POLICY_DT_S
    )
    assert duration == 15.36
    assert duration > M.PUSH_INTERVAL_RANGE_S[1]


def test_completion_marker_is_exact_once_and_binds_both_contracts() -> None:
    row = {
        "cleanup_complete": True,
        "completed_ppo_updates": 5,
        "event": "hope_training_complete",
        "num_envs": 4096,
        "schema_version": 1,
        "stage": "probe",
        "training_contract_sha256": "a" * 64,
        "training_launch_claim_sha256": "b" * 64,
        "vendor_runtime_training_contract_sha256": "c" * 64,
    }
    assert M._validate_completion(
        [row],
        stage="probe",
        num_envs=4096,
        updates=5,
        expected_claim_sha256="b" * 64,
        expected_hard_contract_sha256="a" * 64,
        expected_vendor_contract_sha256="c" * 64,
    ) == row
    with pytest.raises(M.ReceiptRefused, match="exactly one"):
        M._validate_completion(
            [],
            stage="probe",
            num_envs=4096,
            updates=5,
            expected_claim_sha256="b" * 64,
            expected_hard_contract_sha256="a" * 64,
            expected_vendor_contract_sha256="c" * 64,
        )
    tampered = dict(row)
    tampered["vendor_runtime_training_contract_sha256"] = "d" * 64
    with pytest.raises(M.ReceiptRefused, match="identity differs"):
        M._validate_completion(
            [tampered],
            stage="probe",
            num_envs=4096,
            updates=5,
            expected_claim_sha256="b" * 64,
            expected_hard_contract_sha256="a" * 64,
            expected_vendor_contract_sha256="c" * 64,
        )


def test_checkpoint_normalizer_requires_persisted_finite_state_and_count() -> None:
    state = {
        "_mean": _FakeTensor([0.0] * 194, (194,)),
        "_std": _FakeTensor([1.0] * 194, (194,)),
        "count": _FakeTensor([10.0], ()),
    }
    summary = M._normalizer_checkpoint_summary(
        state,
        role="actor",
        expected_features=194,
        torch_module=_FakeTorch,
    )
    assert summary["feature_count"] == 194
    assert summary["count"] == 10.0

    with pytest.raises(M.ReceiptRefused, match="lacks actor"):
        M._normalizer_checkpoint_summary(
            None,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )
    nonfinite = dict(state)
    nonfinite["_mean"] = _FakeTensor(
        [float("nan"), *([0.0] * 193)], (194,)
    )
    with pytest.raises(M.ReceiptRefused, match="non-finite"):
        M._normalizer_checkpoint_summary(
            nonfinite,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )
    zero_count = dict(state)
    zero_count["count"] = _FakeTensor([0.0], ())
    with pytest.raises(M.ReceiptRefused, match="finite and positive"):
        M._normalizer_checkpoint_summary(
            zero_count,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )
    negative_scale = dict(state)
    negative_scale["_std"] = _FakeTensor(
        [-1.0, *([1.0] * 193)], (194,)
    )
    with pytest.raises(M.ReceiptRefused, match="scale is negative"):
        M._normalizer_checkpoint_summary(
            negative_scale,
            role="actor",
            expected_features=194,
            torch_module=_FakeTorch,
        )


def test_checkpoint_normalizer_count_must_not_regress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = [10.0, 11.0, 9.0, 12.0, 13.0]
    for index in M.EXPECTED_CHECKPOINT_INDICES["probe"]:
        (tmp_path / f"model_{index}.pt").touch()

    def fake_summary(_path, *, expected_iteration, **_kwargs):
        normalizer = {"count": counts[expected_iteration]}
        return {
            "actor_normalizer": dict(normalizer),
            "critic_normalizer": dict(normalizer),
        }

    monkeypatch.setattr(M, "_checkpoint_summary", fake_summary)
    with pytest.raises(M.ReceiptRefused, match="count regressed"):
        M._inspect_checkpoints(
            tmp_path,
            stage="probe",
            expected_claim_sha256="a" * 64,
            expected_contract_sha256="b" * 64,
        )


def _push_records(stage: str) -> list[dict]:
    updates = M.EXPECTED_STAGES[stage]["max_iterations"]
    result = []
    for update in range(updates):
        active = stage == "push_evidence" and update == updates - 1
        result.append(
            {
                "event": "hope_push_velocity_diagnostic_update",
                "schema_version": 1,
                "ppo_update": update,
                "counters": {
                    "event_call_count": 1 if active else 0,
                    "env_application_count": 4096 if active else 0,
                    "delta_nonfinite_element_count": 0,
                    "axes": {
                        axis: {
                            "observed_delta_min": -0.05 if active else None,
                            "observed_delta_max": 0.05 if active else None,
                            "below_range_count": 0,
                            "above_range_count": 0,
                        }
                        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
                    },
                },
            }
        )
    return result


def test_runtime_push_counter_is_required_not_inferred_from_duration() -> None:
    records = _push_records("push_evidence")
    result = M._validate_push_velocity(
        records, stage="push_evidence", updates=32, num_envs=4096
    )
    assert result["aggregate"]["env_application_count"] == 4096

    missing = _push_records("push_evidence")
    missing[-1]["counters"]["event_call_count"] = 0
    missing[-1]["counters"]["env_application_count"] = 0
    for values in missing[-1]["counters"]["axes"].values():
        values["observed_delta_min"] = None
        values["observed_delta_max"] = None
    with pytest.raises(M.ReceiptRefused, match="population-equivalent"):
        M._validate_push_velocity(
            missing, stage="push_evidence", updates=32, num_envs=4096
        )


def test_scientific_argv_excludes_only_operational_axes() -> None:
    argv = [
        "/python",
        "/train.py",
        "task=HOPEPingPongActionBallA3VendorV1",
        M._V.STABLE_READY_PLANT_OVERRIDE,
        f"task.actor_obs_contract={M.ACTOR_OBS_CONTRACT}",
        "seed=0",
        "num_envs=4096",
        "max_iterations=5",
        "algo.runner.save_interval=1",
        "device=cuda:0",
        "run_name=probe",
    ]
    normalized, digest = M._scientific_argv(argv)
    assert "seed=0" in normalized
    assert not any(item.startswith("num_envs=") for item in normalized)
    assert len(digest) == 64


def test_materialized_receipt_is_self_reference_free_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_commit = "b" * 40
    evidence_commit = gate_commit
    identity = {"action_id": "bh_loop_c", "scientific_argv_canonical_sha256": "c" * 64}
    monkeypatch.setattr(
        M,
        "_verify_gate_source",
        lambda *_args, **_kwargs: {"path": M.PRODUCER_SOURCE, "sha256": "d" * 64},
    )
    monkeypatch.setattr(
        M,
        "_stage_evidence",
        lambda *_args, expected_stage, **_kwargs: (
            {
                "stage": expected_stage,
                "source_commit": evidence_commit,
                "control_step_action_delay": {
                    "training_contract_sha256": "e" * 64
                },
                "runtime_abi": {"runtime": {"version": "same"}},
            },
            identity,
        ),
    )
    monkeypatch.setattr(
        M.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    receipt = M.materialize(
        gate_checkout=tmp_path,
        gate_source_commit=gate_commit,
        evidence_source_commit=evidence_commit,
        probe_namespace=tmp_path,
        probe_run_dir=tmp_path,
        push_namespace=tmp_path,
        push_run_dir=tmp_path,
        receipt_repo_path="configs/n1_vendor_probe_gate_20260731/pass.json",
        long_spec_repo_path=(
            "configs/n1_vendor_launch_20260731/bh_loop_c.long.template.json"
        ),
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["producer"]["self_reference_free"] is True
    assert "artifact_commit" not in json.dumps(receipt)
    assert receipt["content_sha256"] == M._canonical_sha(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )

    output = tmp_path / "receipt.json"
    M._write_no_clobber(output, receipt)
    with pytest.raises(M.ReceiptRefused, match="no-clobber"):
        M._write_no_clobber(output, receipt)
    assert output.read_bytes() == M._canonical_bytes(receipt) + b"\n"


def test_content_tamper_changes_seal() -> None:
    unsigned = {"schema_version": 1, "kind": M.RECEIPT_KIND, "verdict": "PASS"}
    sealed = {**unsigned, "content_sha256": M._canonical_sha(unsigned)}
    tampered = copy.deepcopy(sealed)
    tampered["verdict"] = "FAIL"
    assert tampered["content_sha256"] != M._canonical_sha(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
