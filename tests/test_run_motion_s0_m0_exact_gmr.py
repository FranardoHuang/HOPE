from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_motion_s0_m0_exact_gmr.py"
SPEC = importlib.util.spec_from_file_location("run_motion_s0_m0_exact_gmr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GMR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GMR
SPEC.loader.exec_module(GMR)


def test_closed_ready_windows_map_to_exact_30_hz_samples():
    left_1_before = GMR.closed_window_sample_mapping([0.0, 0.75], 30, 105, "before")
    left_1_after = GMR.closed_window_sample_mapping([2.75, 3.25], 30, 105, "after")
    assert left_1_before["indices"] == list(range(23))
    assert left_1_after["indices"] == list(range(83, 98))

    # 0.833333 is deliberately below 25/30, so it must not silently round up.
    high_press = GMR.closed_window_sample_mapping([0.0, 0.833333], 30, 88, "S0")
    assert high_press["last_index"] == 24


def test_pip_freeze_normalization_is_sorted_lf_with_one_terminal_lf():
    assert GMR.normalized_pip_freeze_bytes(" z==2\r\n\nA==1\n") == b"A==1\nz==2\n"


def test_stance_gate_preserves_both_components_and_rejects_narrower_finish():
    tolerances = {
        "fore_aft_component_abs_error_max": 0.03,
        "lateral_component_abs_error_max": 0.03,
        "lateral_narrowing_max": 0.005,
        "minimum_initial_abs_lateral_separation": 0.05,
    }
    passed = GMR.evaluate_stance_vectors([0.04, -0.24], [0.05, -0.242], tolerances)
    assert passed["stance_passed"] is True
    assert passed["fore_aft_stagger_initial_m"] == 0.04
    assert passed["lateral_separation_initial_signed_m"] == -0.24

    # The vector component is still within the broad 3 cm ready-set tolerance,
    # but a 2 cm narrower stance is independently forbidden.
    narrowed = GMR.evaluate_stance_vectors([0.04, -0.24], [0.05, -0.22], tolerances)
    assert narrowed["checks"]["lateral_component_preserved"] is True
    assert narrowed["checks"]["not_narrowed"] is False
    assert narrowed["stance_passed"] is False

    fore_aft_lost = GMR.evaluate_stance_vectors([0.04, -0.24], [0.08, -0.245], tolerances)
    assert fore_aft_lost["checks"]["fore_aft_component_preserved"] is False
    assert fore_aft_lost["stance_passed"] is False


def test_explicit_joint_bijection_can_reorder_gmr_dofs():
    bijection = [
        {
            "gmr_dof_index": index,
            "gmr_joint": f"j{index}",
            "canonical_qpos_index": 7 + ((index + 1) % 31),
            "canonical_joint": f"j{index}",
        }
        for index in range(31)
    ]
    assert GMR.reorder_dof_row_to_canonical(list(range(31)), bijection) == [
        30.0,
        *map(float, range(30)),
    ]


def test_tracked_m0_stance_prereg_and_mutations_fail_closed():
    plan = json.loads(
        (ROOT / "configs" / "motion_exact_gmr_m0_prereg_20260713.json").read_text(
            encoding="utf-8"
        )
    )
    GMR._validate_m0_stance(plan, plan["inputs"])

    narrowed = copy.deepcopy(plan)
    narrowed["m0_stance_contract"]["preregistered_tolerances_m"][
        "lateral_narrowing_max"
    ] = 0.03
    with pytest.raises(GMR.ContractError, match="tolerances"):
        GMR._validate_m0_stance(narrowed, narrowed["inputs"])

    shifted = copy.deepcopy(plan)
    shifted["m0_stance_contract"]["ready_window_sample_mappings"][0]["ready_after"][
        "indices"
    ][0] = 82
    with pytest.raises(GMR.ContractError, match="sample mapping"):
        GMR._validate_m0_stance(shifted, shifted["inputs"])


def test_blocked_runtime_receipt_is_machine_readable_and_rejects_substitution():
    runtime = json.loads(
        (ROOT / "configs" / "motion_s0_m0_exact_gmr_runtime_20260713.json").read_text(
            encoding="utf-8"
        )
    )
    GMR._validate_blocked_runtime(runtime, ROOT)

    # The tracked/canonical A3 order is useful evidence but may not be copied
    # into the retarget XML field whose direct parser output was truncated.
    substituted = copy.deepcopy(runtime)
    substituted["ignored_gmr_source"]["retarget_joint_order"] = substituted[
        "a3_robot_contract"
    ]["joint_order"]
    with pytest.raises(GMR.ContractError, match="must remain empty"):
        GMR._validate_blocked_runtime(substituted)

    hidden_gap = copy.deepcopy(runtime)
    hidden_gap["required_unresolved_evidence"].pop()
    with pytest.raises(GMR.ContractError, match="unresolved evidence list"):
        GMR._validate_blocked_runtime(hidden_gap)


@pytest.mark.parametrize(
    "name",
    [
        "motion_exact_gmr_s0_prereg_20260713.json",
        "motion_exact_gmr_m0_prereg_20260713.json",
    ],
)
def test_tracked_batch_plans_fail_at_bound_runtime_gap(name):
    plan = ROOT / "configs" / name
    expected_sha = GMR.sha256_file(plan)
    with pytest.raises(GMR.ContractError, match="intentionally blocked.*retarget_body_order"):
        GMR.validate_plan(plan, expected_sha, ROOT)


def test_write_json_exclusive_is_no_clobber(tmp_path):
    target = tmp_path / "report.json"
    GMR.write_json_exclusive(target, {"status": "first"})
    with pytest.raises(FileExistsError):
        GMR.write_json_exclusive(target, {"status": "second"})
    assert '"first"' in target.read_text(encoding="utf-8")


def test_real_directory_rejects_symlink_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(GMR.ContractError, match="must not traverse a symlink"):
        GMR.verify_real_directory(alias, "fixture")


def _consume_fixture(tmp_path: Path) -> tuple[dict, dict]:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    output_root = tmp_path / "output"
    plan = {
        "batch_kind": "s0_static_high_press",
        "source_materialization": {"completion_manifest": {"sha256": "1" * 64}},
        "ignored_gmr_source": {"commit": "2" * 40},
        "_runtime_contract_binding": {
            "path": "configs/runtime.json",
            "bytes": 1,
            "sha256": "5" * 64,
        },
        "execution_contract": {
            "timeout_seconds_per_asset": 10,
            "warmup_threshold_strict_lt": 0.0001,
            "warmup_max_rounds": 200,
            "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "converter_argv_template": [
                "{python}",
                "{converter}",
                "--gvhmr_pred_file",
                "{input}",
                "--robot",
                "agibot_a3",
                "--save_path",
                "{output}",
            ],
        },
        "a3_robot_contract": {},
        "output_contract": {
            "output_root": str(output_root),
            "result_suffix": ".exact_franco_donor_betas.gmr.pkl",
            "completion_manifest_filename": "completion_manifest.json",
        },
        "s0_semantic_guard": {
            "observed_ball_contact": None,
            "strike_effectiveness": None,
        },
        "m0_stance_contract": None,
    }
    inspected = {
        "output_root": output_root,
        "canonical_mjcf": tmp_path / "a3.xml",
        "gmr": {"root": tmp_path, "converter": tmp_path / "converter.py"},
        "python": Path(sys.executable),
        "rows": [
            {
                "asset_id": "static_backhand_high_press",
                "frames": 3,
                "input": {"path": str(source), "bytes": 6, "sha256": "3" * 64},
                "input_path": str(source),
            }
        ],
    }
    return plan, inspected


def test_consume_publishes_completion_last(monkeypatch, tmp_path):
    plan, inspected = _consume_fixture(tmp_path)
    monkeypatch.setattr(GMR, "inspect_plan", lambda *_: inspected)
    monkeypatch.setattr(GMR, "verify_gmr_source", lambda *_: inspected["gmr"])
    monkeypatch.setattr(GMR, "verify_materialization", lambda *_: inspected["rows"])
    monkeypatch.setattr(GMR, "verify_tree_contract", lambda *_: inspected["canonical_mjcf"])
    monkeypatch.setattr(GMR, "load_gmr_payload", lambda *_: {"fps": 30.0})
    def fake_auditor(plan, python, auditor, output, log, audit, frames, env):
        audit.write_text("{}\n", encoding="utf-8")
        return {"warmup": {"rounds": 1, "max_dq": 0.0}}

    monkeypatch.setattr(GMR, "run_auditor", fake_auditor)

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--save_path") + 1])
        output.write_bytes(b"gmr")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(GMR.subprocess, "run", fake_run)
    published: list[Path] = []
    original = GMR.write_json_exclusive

    def recording_write(path, payload):
        published.append(path)
        original(path, payload)

    monkeypatch.setattr(GMR, "write_json_exclusive", recording_write)
    completion = GMR.consume(plan, tmp_path / "plan.json", "4" * 64, ROOT)
    assert completion.name == "completion_manifest.json"
    assert published[-1] == completion
    assert (completion.parent / "outputs" / "static_backhand_high_press.exact_franco_donor_betas.gmr.pkl").is_file()
    assert (completion.parent / "bindings" / "static_backhand_high_press.json").is_file()


def test_failed_consume_preserves_evidence_without_completion(monkeypatch, tmp_path):
    plan, inspected = _consume_fixture(tmp_path)
    monkeypatch.setattr(GMR, "inspect_plan", lambda *_: inspected)
    monkeypatch.setattr(
        GMR.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 9),
    )
    with pytest.raises(GMR.ContractError, match="converter failed"):
        GMR.consume(plan, tmp_path / "plan.json", "4" * 64, ROOT)
    root = inspected["output_root"]
    assert not (root / "completion_manifest.json").exists()
    failure = (root / "bindings" / "static_backhand_high_press.json").read_text(
        encoding="utf-8"
    )
    assert "failed_preserved_no_completion_manifest" in failure
