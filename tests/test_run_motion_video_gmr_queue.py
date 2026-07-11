from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_motion_video_gmr_queue.py"
SPEC = importlib.util.spec_from_file_location("run_motion_video_gmr_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def _row(asset_id: str, path: Path) -> dict:
    return {
        "asset_id": asset_id,
        "result_path": str(path),
        "result_sha256": QUEUE.sha256_file(path),
        "result_bytes": path.stat().st_size,
        "frames": 5,
        "structural_status": "pass",
    }


def test_manifest_selection_preserves_manifest_or_requested_order(tmp_path):
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    manifest = {"results": [_row("a", first), _row("b", second)]}
    assert [row["asset_id"] for row in QUEUE.select_results(manifest, None)] == ["a", "b"]
    assert [row["asset_id"] for row in QUEUE.select_results(manifest, ["b", "a"])] == ["b", "a"]
    with pytest.raises(QUEUE.QueueError, match="unique"):
        QUEUE.select_results(manifest, ["a", "a"])
    with pytest.raises(QUEUE.QueueError, match="absent"):
        QUEUE.select_results(manifest, ["missing"])


def test_verify_inputs_binds_full_bytes(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"exact-gvhmr")
    row = _row("clip", source)
    QUEUE.verify_inputs([row])
    source.write_bytes(b"changed")
    with pytest.raises(QUEUE.QueueError, match="verification failed"):
        QUEUE.verify_inputs([row])


def test_command_and_environment_are_cpu_only_and_keep_required_warmup(tmp_path):
    command = QUEUE.build_command(
        Path("/env/python"),
        Path("/gmr/scripts/gvhmr_to_robot.py"),
        Path("/input/result.pt"),
        Path("/output/result.pkl"),
    )
    assert command == [
        "/env/python",
        "/gmr/scripts/gvhmr_to_robot.py",
        "--gvhmr_pred_file",
        "/input/result.pt",
        "--robot",
        "agibot_a3",
        "--save_path",
        "/output/result.pkl",
    ]
    assert "--no-warmup" not in command
    assert "--velocity-limit" not in command
    env = QUEUE.build_environment(tmp_path, 7, 6)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONPATH"] == str(tmp_path)
    assert env["OMP_NUM_THREADS"] == "7"
    assert env["MKL_NUM_THREADS"] == "6"


def test_completed_binding_requires_exact_output_log_audit_and_contract(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "output.pkl"
    log = tmp_path / "run.log"
    audit_path = tmp_path / "audit.json"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    log.write_text("warm-up pass 3 max_dq=1e-5\n", encoding="utf-8")
    row = _row("clip", source)
    contract = {"gmr_commit": "abc", "formal_eligible": False}
    audit = {
        "status": "pass",
        "result_sha256": QUEUE.sha256_file(output),
        "run_log_sha256": QUEUE.sha256_file(log),
        "actual_frames": 5,
        "body_shape_contract": "diagnostic_video_betas",
        "formal_eligible": False,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    binding = {
        "status": "complete",
        "source_sha256": row["result_sha256"],
        "processing_contract": contract,
        "body_shape_contract": "diagnostic_video_betas",
        "formal_eligible": False,
        "output_sha256": QUEUE.sha256_file(output),
        "run_log_path": str(log),
        "run_log_sha256": QUEUE.sha256_file(log),
        "structural_audit_path": str(audit_path),
        "structural_audit_sha256": QUEUE.sha256_file(audit_path),
    }
    assert QUEUE.completed_binding_matches(binding, row, output, contract)
    output.write_bytes(b"mutated")
    assert not QUEUE.completed_binding_matches(binding, row, output, contract)


def test_load_manifest_rejects_unsafe_asset_id_and_formal_promotion(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "status": "complete",
        "formal_eligible": False,
        "results": [_row("../escape", source)],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QUEUE.QueueError, match="unsafe"):
        QUEUE.load_manifest(manifest_path)
    payload["results"][0]["asset_id"] = "safe"
    payload["formal_eligible"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QUEUE.QueueError, match="formal_eligible=false"):
        QUEUE.load_manifest(manifest_path)


def test_source_bundle_must_verify_and_advertise_clean_head(tmp_path):
    repo = tmp_path / "gmr"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True)
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=test",
            "-c", "user.email=test@example.com", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    bundle = tmp_path / "gmr.bundle"
    subprocess.run(
        ["git", "-C", str(repo), "bundle", "create", str(bundle), "HEAD"],
        check=True,
    )
    report = QUEUE.verify_source_bundle(repo, bundle, commit)
    assert report["verified_commit"] == commit
    assert report["bytes"] == bundle.stat().st_size
    assert report["sha256"] == QUEUE.sha256_file(bundle)
    with pytest.raises(QUEUE.QueueError, match="does not advertise"):
        QUEUE.verify_source_bundle(repo, bundle, "0" * 40)
