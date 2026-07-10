"""CPU-only tests for scripts/audit_runpod_terminal_runs.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_runpod_terminal_runs.py"
_SPEC = importlib.util.spec_from_file_location("audit_runpod_terminal_runs", _SCRIPT)
audit = importlib.util.module_from_spec(_SPEC)
sys.modules["audit_runpod_terminal_runs"] = audit
_SPEC.loader.exec_module(audit)


RUN_NAMES = {
    "R5b_seed2": "2026-07-09_01-00-00_s1w4_R5b_seed2_v4rg",
    "ST1": "2026-07-09_01-01-00_s1w4_ST1_stagger_v4rg",
    "G1": "2026-07-09_01-02-00_s1w4_G1_swingsyn",
    "G2": "2026-07-09_01-03-00_s1w4_G2_v4rgsyn",
    "C1": "2026-07-09_01-04-00_s1w4_C1_combo",
    "C2": "2026-07-09_01-05-00_s1w4_C2_combo",
    "C3": "2026-07-09_01-06-00_s1w4_C3_combo",
    "C4": "2026-07-09_01-07-00_s1w4_C4_combo",
    "N1": "2026-07-09_01-08-00_s1w4_N1_tts_noise",
    "T3": "2026-07-09_01-09-00_s1w4_T3_v5topp_std",
    "R8b_seed2": "2026-07-09_01-10-00_s1w4_R8b_seed2_v4rg",
}


def _make_run(root: Path, name: str, *, terminal_report: bool = False) -> Path:
    run = root / name
    (run / "params").mkdir(parents=True)
    (run / "params" / "env.yaml").write_text("commands: {}\n")
    (run / "params" / "env.pkl").write_bytes(b"env-config-pickle")
    (run / "params" / "agent.pkl").write_bytes(b"agent-config-pickle")
    for number in audit.expected_checkpoint_numbers(13000, 16999, 100):
        (run / f"model_{number}.pt").write_bytes(f"checkpoint-{number}".encode())
    (run / "judge").mkdir()
    tag = 16999 if terminal_report else 16400
    (run / "judge" / f"judge_report_model_{tag}_20260710_000000.md").write_text(
        f"- checkpoint: `model_{tag}.pt`\n"
    )
    (run / "exported").mkdir()
    (run / "exported" / "policy.onnx").write_bytes(b"onnx")
    (run / "exported" / "learned_std.npy").write_bytes(b"std")
    (run / "exported" / "obs_norm.npz").write_bytes(b"norm")
    return run


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "agibot_a3_hope_virtualball"
    root.mkdir()
    for label, name in RUN_NAMES.items():
        _make_run(root, name, terminal_report=(label == "C4"))
    # The active continuation must not collide with the exact G1 arm mapping.
    _make_run(root, "2026-07-10_07-30-00_s1w4_G1b_swingsyn_facerescue")
    return root


def _audit_all(run_root: Path):
    return audit.audit_all(
        run_root,
        audit.default_specs(),
        checkpoint_start=13000,
        terminal=16999,
        checkpoint_interval=100,
        judge_script=_SCRIPT.with_name("judge.sh"),
        judge_steps=15000,
        judge_seed=0,
        judge_gpu=0,
    )


def test_default_map_is_unique_and_excludes_g1b(run_root: Path):
    candidates = audit.discover_run_dirs(run_root)
    mapped = audit.map_runs(audit.default_specs(), candidates)
    assert len(mapped) == 11
    assert mapped["G1"].name.endswith("s1w4_G1_swingsyn")
    assert "G1b" not in mapped["G1"].name


def test_complete_early_judged_runs_emit_exact_terminal_commands(run_root: Path):
    reports = {item.label: item for item in _audit_all(run_root)}
    g2 = reports["G2"]
    assert g2.status == "NEEDS_TERMINAL_JUDGE"
    assert g2.checkpoint_count == 41
    assert g2.checkpoint_sequence_exact
    assert g2.latest_report_checkpoint == 16400
    assert g2.terminal_checkpoint.endswith("model_16999.pt")
    assert "model_16999.pt" in g2.dry_run_command
    assert "--steps 15000" in g2.dry_run_command
    assert g2.dry_run_command.endswith("--dry-run")
    assert "model_16400.pt" not in g2.judge_command

    c4 = reports["C4"]
    assert c4.status == "TERMINAL_JUDGED"
    assert c4.terminal_report_exists


def test_missing_checkpoint_sequence_is_invalid(run_root: Path):
    missing = run_root / RUN_NAMES["N1"] / "model_15000.pt"
    missing.unlink()
    reports = {item.label: item for item in _audit_all(run_root)}
    n1 = reports["N1"]
    assert n1.status == "INVALID"
    assert not n1.checkpoint_sequence_exact
    assert any("missing=[15000]" in error for error in n1.errors)


def test_duplicate_match_fails_loud(run_root: Path):
    _make_run(run_root, "2026-07-09_02-00-00_s1w4_C3_second_copy")
    with pytest.raises(audit.MappingError, match="C3: expected exactly one match"):
        audit.map_runs(audit.default_specs(), audit.discover_run_dirs(run_root))


def test_unmatched_label_lists_terminal_candidates(run_root: Path):
    spec = [audit.ArmSpec("NOPE", r"(?:^|[_-])NOPE(?:[_-]|$)")]
    with pytest.raises(audit.MappingError) as excinfo:
        audit.map_runs(spec, audit.discover_run_dirs(run_root))
    message = str(excinfo.value)
    assert "found 0: <none>" in message
    assert RUN_NAMES["G2"] in message


def test_missing_env_pickle_warns_but_does_not_block_terminal_judge(run_root: Path):
    (run_root / RUN_NAMES["G2"] / "params" / "env.pkl").unlink()
    reports = {item.label: item for item in _audit_all(run_root)}
    g2 = reports["G2"]
    assert g2.status == "NEEDS_TERMINAL_JUDGE"
    assert not g2.env_pickle_exists
    assert any("formal Isaac scorecard" in warning for warning in g2.warnings)


def test_missing_agent_pickle_warns_but_does_not_block_terminal_judge(run_root: Path):
    (run_root / RUN_NAMES["G2"] / "params" / "agent.pkl").unlink()
    reports = {item.label: item for item in _audit_all(run_root)}
    g2 = reports["G2"]
    assert g2.status == "NEEDS_TERMINAL_JUDGE"
    assert not g2.agent_pickle_exists
    assert any("runner/normalizer" in warning for warning in g2.warnings)
