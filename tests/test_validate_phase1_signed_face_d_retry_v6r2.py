import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase1_signed_face_d_retry_prereg_v6r2_20260714.json"
VALIDATOR = ROOT / "scripts/validate_phase1_signed_face_d_retry_v6r2.py"
SPEC = importlib.util.spec_from_file_location(
    "phase1_signed_face_d_retry_v6r2", VALIDATOR
)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def manifest():
    return M.load_manifest(CONFIG)


def test_manifest_is_source_only_without_retry_or_runtime_design():
    data = manifest()
    assert data["manifest_id"].endswith("validator-correction-20260714-v6r2")
    assert data["source_installation"]["only_permitted_action"] == "static-validate"
    assert data["source_installation"]["runtime_actions_must_fail"] == [
        "validate",
        "plan",
        "launch",
        "finalize",
    ]
    for removed in (
        "retry_authority",
        "runtime",
        "mixed_finalizer",
        "foreign_v6",
        "original_terminal_cells",
        "original_l1_checkpoint_audit",
    ):
        assert removed not in data
    serialized = json.dumps(data, sort_keys=True)
    assert "new_run_name" not in serialized
    assert "training_launch_attempt_ordinal" not in serialized
    assert "only_command_change" not in serialized


def test_superseded_validator_correction_binds_exact_historical_evidence():
    correction = manifest()["superseded_validator_correction"]
    assert correction["superseded_manifest_id"] == M.V6R1_MANIFEST_ID
    assert correction["superseded_config"]["sha256"] == M.V6R1_CONFIG_SHA256
    assert correction["superseded_consumer"]["sha256"] == M.V6R1_CONSUMER_SHA256
    bug = correction["historical_bug_correction"]
    assert bug["immutable_checkpoint_audit"]["d_row_run_dirs"] == []
    assert bug["correct_required_state"] == "absent"
    assert bug["entry_kinds_that_must_fail"] == [
        "directory",
        "regular_file",
        "symlink",
        "special",
        "unstatable",
    ]
    assert bug["old_outer_evidence"]["pid_dead"] is True
    assert bug["old_outer_evidence"]["checkpoint_count"] == 0


def test_terminal_v8_barrier_is_exact_and_not_recipe_evidence():
    barrier = manifest()["foreign_v8_terminal_barrier"]
    assert barrier["serial_launcher_cell_ordinal_zero_based"] == 3
    assert barrier["preceding_cell_c_terminal_checkpoint"] == "model_24.pt"
    assert barrier["failure_scope"] == "pre_contract_boot_not_learning_or_recipe_result"
    assert barrier["face_reward_causality_claim_forbidden"] is True
    assert barrier["failure_evidence"]["sha256"] == M.V8_FAILURE_SHA256
    assert barrier["final_launch_state_evidence"]["sha256"] == M.V8_FINAL_STATE_SHA256
    assert barrier["automatic_retry_forbidden"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x["superseded_validator_correction"]["historical_bug_correction"].__setitem__(
            "correct_required_state", "directory"
        ),
        lambda x: x["superseded_validator_correction"]["superseded_config"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda x: x["foreign_v8_terminal_barrier"]["failure_evidence"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda x: x.__setitem__("launch_authorized", True),
        lambda x: x["next_step"].__setitem__(
            "v6r2_must_not_be_reused_as_launcher", False
        ),
    ],
)
def test_any_manifest_semantic_mutation_fails_closed(tmp_path, mutate):
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(M.ContractError, match="semantic content changed"):
        M.load_manifest(path)


def test_static_cli_binds_final_self_hashes():
    config_sha = M.sha256_file(CONFIG)
    validator_sha = M.sha256_file(VALIDATOR)
    assert (
        M.main(
            [
                "--config",
                str(CONFIG),
                "--expected-config-sha256",
                config_sha,
                "--expected-validator-sha256",
                validator_sha,
                "static-validate",
            ]
        )
        == 0
    )
    with pytest.raises(M.ContractError, match="v6r2 manifest SHA changed"):
        M.main(
            [
                "--config",
                str(CONFIG),
                "--expected-config-sha256",
                "0" * 64,
                "--expected-validator-sha256",
                validator_sha,
                "static-validate",
            ]
        )


@pytest.mark.parametrize("action", ["validate", "plan", "launch", "finalize"])
def test_every_runtime_action_is_explicitly_rejected(action):
    with pytest.raises(M.ContractError, match="source-only and NOT LAUNCHED"):
        M.main(
            [
                "--config",
                str(CONFIG),
                "--expected-config-sha256",
                M.sha256_file(CONFIG),
                "--expected-validator-sha256",
                M.sha256_file(VALIDATOR),
                action,
            ]
        )


def test_validator_has_no_dead_runtime_or_launch_surface():
    source = VALIDATOR.read_text(encoding="utf-8")
    for forbidden in (
        "build_retry_command",
        "verify_superseded_v6r1",
        "verify_old_d_failure",
        "process_entry_exists",
        "matching_entries",
        "parse_launch_state",
        "subprocess",
        "Popen",
        "os.kill",
        "killpg",
        "signal.",
        "pkill",
        "killall",
        "pgrep -f",
        "actor_leg_ref_mask",
        "run_name=",
    ):
        assert forbidden not in source
