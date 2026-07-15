from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/phase1_frame0_wait_recovery_contract_v2_20260715.json"
VALIDATOR_PATH = ROOT / "scripts/validate_phase1_frame0_wait_recovery_contract_v2.py"

SPEC = importlib.util.spec_from_file_location(
    "validate_phase1_frame0_wait_recovery_contract_v2", VALIDATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _load() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _validate_mutation(tmp_path: Path, value: dict, name: str = "mutated.json") -> None:
    path = _write(tmp_path / name, value)
    VALIDATOR.validate_contract(path, VALIDATOR.sha256_file(path))


def _validate_raw(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "raw.json"
    path.write_text(raw, encoding="utf-8")
    VALIDATOR.validate_contract(path, VALIDATOR.sha256_file(path))


def test_checked_in_contract_passes_design_and_blocks_launch(capsys):
    digest = VALIDATOR.sha256_file(CONTRACT)
    assert digest == VALIDATOR.EXPECTED_CONTRACT_SHA256
    assert VALIDATOR.main([
        "--contract", str(CONTRACT),
        "--expected-contract-sha256", digest,
        "--mode", "design-check",
    ]) == 0
    output = capsys.readouterr().out
    assert '"status": "pass_design_only"' in output
    assert '"old_prereg_unchanged": true' in output
    assert '"adapter_implemented": false' in output
    assert '"future_action_hidden_before_reveal": true' in output
    assert '"velocity_reference": "root_joint_body_all_zero"' in output

    assert VALIDATOR.main([
        "--contract", str(CONTRACT),
        "--expected-contract-sha256", digest,
        "--mode", "launch-check",
    ]) == 1
    blocked = capsys.readouterr().err
    assert "LAUNCH BLOCKED" in blocked
    assert "selected_action_frame0_source_adapter" in blocked
    assert "ready_numeric_tolerances" in blocked
    assert "continuous_carry_state_runtime_receipt" in blocked
    assert "source_audit.adapter_implemented" in blocked


def test_v1_parent_bytes_are_unchanged_and_bound():
    value = _load()
    parent = ROOT / value["immutable_parent"]["repo_path"]
    assert len(parent.read_bytes()) == 17008
    assert VALIDATOR.sha256_file(parent) == (
        "ca7806df83b650546cf4406963bb231622a248c8e04e944991a371e44d810616"
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            '  "schema_version": 2,\n',
            '  "schema_version": 2,\n  "schema_version": 2,\n',
            "duplicate JSON key",
        ),
        (
            '    "pose_source": "selected_public_action_exact_frame0",\n',
            '    "pose_source": "selected_public_action_exact_frame0",\n'
            '    "pose_source": "duplicate",\n',
            "duplicate JSON key",
        ),
        (
            '      "threshold_binding": null\n',
            '      "threshold_binding": NaN\n',
            "non-finite JSON constant",
        ),
        (
            '      "threshold_binding": null\n',
            '      "threshold_binding": 1e9999\n',
            "non-finite JSON number",
        ),
    ],
)
def test_strict_json_rejects_duplicates_and_nonfinite(tmp_path: Path, old: str, new: str, message: str):
    raw = CONTRACT.read_text(encoding="utf-8").replace(old, new, 1)
    with pytest.raises(VALIDATOR.ContractError, match=message):
        _validate_raw(tmp_path, raw)


def test_contract_identity_and_type_are_exact(tmp_path: Path):
    value = _load()
    value["schema_version"] = True
    with pytest.raises(VALIDATOR.ContractError, match="schema_version"):
        _validate_mutation(tmp_path, value, "bool-schema.json")

    value = _load()
    value["contract_id"] = "renamed"
    with pytest.raises(VALIDATOR.ContractError, match="contract_id"):
        _validate_mutation(tmp_path, value, "renamed.json")

    value = _load()
    value["silent_escape"] = True
    with pytest.raises(VALIDATOR.ContractError, match="top-level keyset"):
        _validate_mutation(tmp_path, value, "extra.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["reference_contract"].__setitem__(
                "default_joint_pos_substitution_allowed", True
            ),
            "frame0 reference contract",
        ),
        (
            lambda value: value["reference_contract"]["velocity_fields_exact_zero"].remove(
                "root_angular_velocity"
            ),
            "frame0 reference contract",
        ),
        (
            lambda value: value["reference_contract"]["xy_anchor"].__setitem__(
                "live_per_tick_reanchor_allowed", True
            ),
            "frame0 reference contract",
        ),
        (
            lambda value: value["phase_contract"]["post_swing_pre_reveal_recovery"].__setitem__(
                "future_action_fields_visible", True
            ),
            "phase and reveal contract",
        ),
        (
            lambda value: value["nonleakage_contract"].__setitem__(
                "before_reveal_future_frame0_visible", True
            ),
            "nonleakage contract",
        ),
    ],
)
def test_frame0_zero_velocity_xy_anchor_and_nonleakage_are_frozen(
    tmp_path: Path, mutation, message: str
):
    value = _load()
    mutation(value)
    with pytest.raises(VALIDATOR.ContractError, match=message):
        _validate_mutation(tmp_path, value)


@pytest.mark.parametrize(
    "field",
    [
        "simulator_root_write_allowed",
        "simulator_joint_write_allowed",
        "teleport_allowed",
        "episode_reset_allowed",
        "observation_history_clear_allowed",
        "last_action_clear_or_replace_allowed",
        "action_delay_ring_clear_allowed",
        "target_delay_ring_clear_allowed",
        "noise_or_dropout_state_clear_allowed",
        "per_swing_bias_clear_allowed",
    ],
)
def test_continuous_episode_cannot_reset_or_clear_carried_state(tmp_path: Path, field: str):
    value = _load()
    value["continuous_episode_carry_contract"][field] = True
    with pytest.raises(VALIDATOR.ContractError, match="carry-state contract"):
        _validate_mutation(tmp_path, value, f"carry-{field}.json")


def test_ready_is_fail_closed_conjunction_and_not_a_weighted_score(tmp_path: Path):
    value = _load()
    ready = value["ready_contract"]
    assert ready["all_conjuncts_required"] is True
    assert ready["positive_reward_can_offset_failure"] is False
    assert ready["numeric_thresholds_bound"] is False
    assert all(row["threshold_binding"] is None for row in ready["conjuncts"])

    ready["positive_reward_can_offset_failure"] = True
    with pytest.raises(VALIDATOR.ContractError, match="cannot be offset"):
        _validate_mutation(tmp_path, value, "reward-offset.json")


def test_design_contract_cannot_silently_claim_adapter_or_launch(tmp_path: Path):
    value = _load()
    value["source_audit"]["adapter_implemented"] = True
    with pytest.raises(VALIDATOR.ContractError, match="source audit"):
        _validate_mutation(tmp_path, value, "fake-adapter.json")

    value = _load()
    value["launch_authorized"] = True
    with pytest.raises(VALIDATOR.ContractError, match="must not authorize"):
        _validate_mutation(tmp_path, value, "fake-launch.json")


def test_parent_and_audited_source_bindings_fail_closed(tmp_path: Path):
    value = _load()
    value["immutable_parent"]["sha256"] = "0" * 64
    with pytest.raises(VALIDATOR.ContractError, match="parent binding"):
        _validate_mutation(tmp_path, value, "bad-parent.json")

    value = _load()
    value["audited_source"]["git_blobs"]["motion_command"]["sha256"] = "0" * 64
    with pytest.raises(VALIDATOR.ContractError, match="blob bindings"):
        _validate_mutation(tmp_path, value, "bad-source.json")
