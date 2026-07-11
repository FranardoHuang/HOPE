from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SCRIPT = ROOT / "scripts" / "generate_phase1_scaleout_curve_manifests.py"
SPEC = importlib.util.spec_from_file_location("scaleout_curve_generator", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)
WORKER_SCRIPT = ROOT / "scripts" / "phase1_checkpoint_curve_worker.py"
WORKER_SPEC = importlib.util.spec_from_file_location("checkpoint_curve_worker", WORKER_SCRIPT)
worker = importlib.util.module_from_spec(WORKER_SPEC)
assert WORKER_SPEC.loader is not None
WORKER_SPEC.loader.exec_module(worker)


def _inputs():
    bindings = json.loads(generator.DEFAULT_BINDINGS.read_text(encoding="utf-8"))
    matrix = json.loads((REPO_ROOT / bindings["scaleout_matrix"]).read_text(encoding="utf-8"))
    return matrix, bindings


def test_scaleout_generation_partitions_18_new_arms_into_independent_queues():
    matrix, bindings = _inputs()
    manifests = generator.build_manifests(matrix, bindings)

    assert set(manifests) == {
        "phase1_checkpoint_curve_scaleout_causal_pod1_20260711.json",
        "phase1_checkpoint_curve_scaleout_fresh_pod1_20260711.json",
        "phase1_checkpoint_curve_scaleout_causal_pod2_20260711.json",
        "phase1_checkpoint_curve_scaleout_fresh_pod2_20260711.json",
    }
    for pod in ("pod1", "pod2"):
        causal = manifests[f"phase1_checkpoint_curve_scaleout_causal_{pod}_20260711.json"]
        fresh = manifests[f"phase1_checkpoint_curve_scaleout_fresh_{pod}_20260711.json"]
        assert len(causal["jobs"]) == 2 * 4
        assert len(fresh["jobs"]) == 7 * 9
        assert len({job["run_dir"] for job in causal["jobs"]}) == 2
        assert len({job["run_dir"] for job in fresh["jobs"]}) == 7


def test_jobs_are_milestone_major_and_freeze_a_non_decisive_clean_q10():
    matrix, bindings = _inputs()
    manifests = generator.build_manifests(matrix, bindings)

    for manifest in manifests.values():
        expected_arm_count = 2 if manifest["queue"] == "causal" else 7
        barrier_ids = [job["barrier_id"] for job in manifest["jobs"]]
        for offset in range(0, len(barrier_ids), expected_arm_count):
            assert len(set(barrier_ids[offset : offset + expected_arm_count])) == 1
        assert all(job["seed"] == 0 for job in manifest["jobs"])
        assert all(job["noise_scales"] == "0.0" for job in manifest["jobs"])
        for job in manifest["jobs"]:
            assert job["extra_args"][:2] == ["--schedule-k", "20"]
            if job["expected_evaluation_contract_exact"]:
                assert job["extra_args"] == ["--schedule-k", "20"]
            else:
                assert job["extra_args"] == [
                    "--schedule-k", "20", "--exam-extra", "--allow-inexact-contract"
                ]
        assert all(job["screen_only"] is True for job in manifest["jobs"])
        assert manifest["screen_policy"]["stop_or_promote_allowed"] is False


def test_causal_terminal_uses_the_runner_final_saved_iteration():
    matrix, bindings = _inputs()
    manifests = generator.build_manifests(matrix, bindings)
    causal_jobs = [
        job
        for name, manifest in manifests.items()
        if "_causal_" in name
        for job in manifest["jobs"]
    ]
    terminal_jobs = [job for job in causal_jobs if job["barrier_id"] == "causal_20998"]
    assert len(terminal_jobs) == 4
    assert all(job["id"].endswith("_20998_clean_q10") for job in terminal_jobs)
    assert all(job["checkpoint"].endswith("/model_20998.pt") for job in terminal_jobs)
    assert not any("20999" in job["id"] or "20999" in job["checkpoint"] for job in causal_jobs)


def test_existing_arm_cadence_manifests_use_the_same_real_terminal():
    for pod, family in (("pod1", "M3"), ("pod2", "M2")):
        path = REPO_ROOT / "configs" / f"phase1_checkpoint_curve_cadence_{pod}_20260711.json"
        manifest = worker.load_manifest(path)
        terminal = [job for job in manifest["jobs"] if "_20998_clean_q10" in job["id"]]
        assert {job["id"] for job in terminal} == {
            f"{family}_old_20998_clean_q10",
            f"{family}_S1_20998_clean_q10",
        }
        assert all(job["checkpoint"].endswith("/model_20998.pt") for job in terminal)
        assert not any("20999" in job["id"] or "20999" in job["checkpoint"] for job in manifest["jobs"])
        assert all(job["screen_only"] is True for job in manifest["jobs"])
        assert manifest["screen_policy"]["stop_or_promote_allowed"] is False


def test_existing_fresh_cadence_is_independent_of_causal_terminal():
    for pod, seed in (("pod1", 1), ("pod2", 2)):
        path = REPO_ROOT / "configs" / f"phase1_checkpoint_curve_cadence_fresh_{pod}_20260711.json"
        manifest = worker.load_manifest(path)
        ids = [job["id"] for job in manifest["jobs"]]
        first_iteration = 4000 if pod == "pod1" else 6000
        assert ids[0] == f"fresh_SZ_seed{seed}_{first_iteration}_clean_q10"
        assert ids[-1] == f"fresh_SZ_seed{seed}_16999_clean_q10"
        assert all("fresh_SZ" in job_id for job_id in ids)
        assert all(job["screen_only"] is True for job in manifest["jobs"])
        assert manifest["screen_policy"]["stop_or_promote_allowed"] is False


def test_fresh_contract_roles_preserve_target_diagnostic_and_inexact_boundaries():
    matrix, bindings = _inputs()
    manifests = generator.build_manifests(matrix, bindings)
    fresh_jobs = [
        job
        for name, manifest in manifests.items()
        if "_fresh_" in name
        for job in manifest["jobs"]
    ]

    by_cell = {}
    for job in fresh_jobs:
        by_cell.setdefault(job["cell"], job)
    assert by_cell["SZ"]["evaluation_role"] == "formal_target"
    assert by_cell["SZ"]["expected_evaluation_contract_exact"] is True
    assert by_cell["SZ"]["formal_target"] is True
    assert by_cell["SP"]["evaluation_role"] == "plant_diagnostic_non_target_inexact"
    assert by_cell["SP"]["expected_evaluation_contract_exact"] is False
    assert by_cell["SP"]["formal_target"] is False
    for cell in ("LZ", "LP"):
        assert by_cell[cell]["evaluation_role"] == "legacy_pairing_diagnostic_inexact"
        assert by_cell[cell]["expected_evaluation_contract_exact"] is False
        assert by_cell[cell]["formal_target"] is False


def test_checked_in_manifests_are_deterministic_products():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    for path in sorted((REPO_ROOT / "configs").glob("phase1_checkpoint_curve_scaleout_*_20260711.json")):
        assert worker.load_manifest(path)["jobs"]
