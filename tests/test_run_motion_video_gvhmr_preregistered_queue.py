from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_motion_video_gvhmr_preregistered_queue.py"
SPEC = importlib.util.spec_from_file_location("run_motion_video_gvhmr_preregistered_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def test_legacy_runner_refuses_new_intake_before_source_or_gpu_work(capsys, tmp_path):
    rc = Q.legacy.main(
        [
            "--manifest",
            str(ROOT / "configs" / "motion_video_intake_20260713.json"),
            "--source-root",
            str(tmp_path / "missing-raw"),
            "--gvhmr-root",
            str(tmp_path / "missing-gvhmr"),
            "--python",
            str(tmp_path / "missing-python"),
            "--state-dir",
            str(tmp_path / "state"),
            "--gpu",
            "0",
        ]
    )
    assert rc == 2
    assert "preregistration-only" in capsys.readouterr().err
    assert not (tmp_path / "state").exists()


def test_legacy_runner_cannot_bypass_guard_by_renaming_intake_id(capsys, tmp_path):
    manifest = json.loads(
        (ROOT / "configs" / "motion_video_intake_20260713.json").read_text(encoding="utf-8")
    )
    manifest["intake_id"] = "renamed-to-try-bypass"
    changed = tmp_path / "renamed.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    rc = Q.legacy.main(
        [
            "--manifest",
            str(changed),
            "--source-root",
            str(tmp_path / "missing-raw"),
            "--gvhmr-root",
            str(tmp_path / "missing-gvhmr"),
            "--python",
            str(tmp_path / "missing-python"),
            "--state-dir",
            str(tmp_path / "state"),
            "--gpu",
            "0",
        ]
    )
    assert rc == 2
    assert "schema_version>=2 intake is preregistration-only" in capsys.readouterr().err
    assert not (tmp_path / "state").exists()


def test_secure_parser_has_no_runtime_override_surface():
    with pytest.raises(SystemExit):
        Q.parse_args(
            [
                "--prereg",
                "prereg.json",
                "--execution-record",
                "record.json",
                "--nvidia-smi",
                "/tmp/fake",
            ]
        )
    with pytest.raises(SystemExit):
        Q.parse_args(
            [
                "--prereg",
                "prereg.json",
                "--execution-record",
                "record.json",
                "--poll-seconds",
                "0.01",
            ]
        )


def test_source_is_hashed_before_and_after_child(tmp_path):
    source = tmp_path / "asset.mp4"
    source.write_bytes(b"exact-source")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    descriptor, before = Q.open_bound_source(source, source.stat().st_size, expected)
    try:
        source.write_bytes(b"drift-source")
        with pytest.raises(Q.QueueError, match="changed during execution"):
            Q.verify_bound_source_after(descriptor, source, before, len(b"exact-source"), expected)
    finally:
        os.close(descriptor)


def test_source_path_replacement_is_detected(tmp_path):
    source = tmp_path / "asset.mp4"
    source.write_bytes(b"exact-source")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    descriptor, before = Q.open_bound_source(source, source.stat().st_size, expected)
    try:
        replacement = tmp_path / "replacement.mp4"
        replacement.write_bytes(b"exact-source")
        os.replace(replacement, source)
        with pytest.raises(Q.QueueError, match="replaced"):
            Q.verify_bound_source_after(descriptor, source, before, len(b"exact-source"), expected)
    finally:
        os.close(descriptor)


def test_child_source_is_private_snapshot_when_staging_bytes_are_changed_and_restored(tmp_path):
    exact = b"exact-source-bytes"
    source_root = tmp_path / "raw"
    source = source_root / "static" / "pai.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(exact)
    source_stat = source.stat()
    manifest = {
        "assets": [
            {
                "id": "static_backhand_high_press",
                "source_relpath": "static/pai.mp4",
                "bytes": len(exact),
                "sha256": hashlib.sha256(exact).hexdigest(),
            }
        ]
    }
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    snapshot_root, records, held = Q.materialize_bound_source_snapshots(
        manifest,
        ["static_backhand_high_press"],
        source_root=source_root,
        state_dir=state_dir,
    )
    snapshot = snapshot_root / "static" / "pai.mp4"
    descriptor, before, bound_path = held["static_backhand_high_press"]
    try:
        # Reproduce the reviewer's staging-path attack: same-length drift, then
        # restore the exact bytes and mtime.  The child pathname is a distinct
        # private inode and remains exact throughout.
        source.write_bytes(b"drift-source-bytes")
        source.write_bytes(exact)
        os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        assert bound_path == snapshot
        assert records["static_backhand_high_press"]["snapshot_path"] == str(snapshot)
        assert snapshot.read_bytes() == exact
        assert snapshot.stat().st_ino != source.stat().st_ino
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400
        Q.verify_bound_source_after(
            descriptor,
            snapshot,
            before,
            len(exact),
            hashlib.sha256(exact).hexdigest(),
        )
    finally:
        os.close(descriptor)
        os.chmod(snapshot_root, 0o700)
        os.chmod(snapshot.parent, 0o700)
        os.chmod(snapshot, 0o600)


def test_secure_main_passes_snapshot_root_not_staging_root_to_child(monkeypatch, tmp_path):
    exact = b"exact-source-bytes"
    source_root = tmp_path / "raw"
    public_source = source_root / "static" / "pai.mp4"
    public_source.parent.mkdir(parents=True)
    public_source.write_bytes(exact)
    public_source_stat = public_source.stat()
    gvhmr_root = tmp_path / "GVHMR"
    (gvhmr_root / "outputs" / "demo").mkdir(parents=True)
    state_root = tmp_path / "state"
    asset = {
        "id": "static_backhand_high_press",
        "source_relpath": "static/pai.mp4",
        "bytes": len(exact),
        "sha256": hashlib.sha256(exact).hexdigest(),
        "media": {"frames": 1},
    }
    manifest = {
        "intake_id": "motion-video-intake-20260713-v12-static-lateral",
        "processing_order": ["static_backhand_high_press"],
        "assets": [asset],
    }
    prereg = {
        "processing_order": ["static_backhand_high_press"],
        "execution_batch": {
            "batch_id": "static_high_press_s0_v1",
            "asset_ids": ["static_backhand_high_press"],
        },
        "intake": {"sha256": "1" * 64},
        "manual_event_review": {"sha256": "2" * 64},
        "franco_priority_context": {"sha256": "3" * 64},
        "gvhmr_runtime": {"commit": "4" * 40},
        "output_contract": {
            "outputs": [
                {
                    "asset_id": "static_backhand_high_press",
                    "path": str(
                        gvhmr_root / "outputs" / "demo" / "pai" / "hmr4d_results.pt"
                    ),
                }
            ]
        },
    }
    record = {
        "state_root": str(state_root),
        "source_root": str(source_root),
        "gvhmr_root": str(gvhmr_root),
        "python": "/usr/bin/python3",
        "gpu_physical_index": 0,
        "max_used_mib": 19000,
        "poll_seconds": 30.0,
        "wait_timeout_seconds": 0.0,
        "nvidia_smi": {"realpath": "/usr/bin/nvidia-smi", "sha256": "5" * 64},
        "authorization_scope": "offline_gvhmr_only",
        "live_dependency_binding": {
            "checkpoint_tree": {"sha256": "6" * 64},
            "python_environment": {"pip_freeze_sha256": "7" * 64},
        },
    }
    binding = {
        "static": {
            "prereg": prereg,
            "intake": manifest,
            "tool_paths": {
                "result_auditor": ROOT / "scripts" / "audit_gvhmr_result.py",
                "legacy_intake_guard": ROOT / "scripts" / "run_motion_video_gvhmr_queue.py",
            },
        },
        "record": record,
        "prereg_sha256": "8" * 64,
        "execution_record_sha256": "9" * 64,
    }
    monkeypatch.setattr(Q, "validate_execution_record_for_launch", lambda *_a, **_k: binding)
    observed: dict[str, Path] = {}

    def fake_run_asset(_asset, *, source_root, **_kwargs):
        observed["source_root"] = source_root
        child_source = source_root / "static" / "pai.mp4"
        public_source.write_bytes(b"drift-source-bytes")
        assert child_source.read_bytes() == exact
        public_source.write_bytes(exact)
        os.utime(
            public_source,
            ns=(public_source_stat.st_atime_ns, public_source_stat.st_mtime_ns),
        )
        return False

    monkeypatch.setattr(Q.legacy, "run_asset", fake_run_asset)
    rc = Q.main(["--prereg", "unused.json", "--execution-record", "unused-record.json"])
    assert rc == 1
    assert observed["source_root"] == state_root / "bound_sources"
    assert observed["source_root"] != source_root

    snapshot = state_root / "bound_sources" / "static" / "pai.mp4"
    os.chmod(state_root / "bound_sources", 0o700)
    os.chmod(snapshot.parent, 0o700)
    os.chmod(snapshot, 0o600)


def test_private_snapshot_rejects_symlinked_staging_asset(tmp_path):
    exact = b"exact-source"
    source_root = tmp_path / "raw"
    source_root.mkdir()
    real = source_root / "real.mp4"
    real.write_bytes(exact)
    linked_parent = source_root / "static"
    linked_parent.mkdir()
    (linked_parent / "pai.mp4").symlink_to(real)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    manifest = {
        "assets": [
            {
                "id": "static_backhand_high_press",
                "source_relpath": "static/pai.mp4",
                "bytes": len(exact),
                "sha256": hashlib.sha256(exact).hexdigest(),
            }
        ]
    }
    with pytest.raises(Q.PreregError, match="symlink"):
        Q.materialize_bound_source_snapshots(
            manifest,
            ["static_backhand_high_press"],
            source_root=source_root,
            state_dir=state_dir,
        )


def test_output_namespace_reservation_is_atomic_across_state_roots(tmp_path):
    output = tmp_path / "outputs" / "demo" / "stem" / "hmr4d_results.pt"
    output.parent.parent.mkdir(parents=True)
    locks, namespaces = Q.reserve_output_namespaces(
        [{"asset_id": "one", "path": str(output)}],
        prereg_sha256="a" * 64,
        execution_record_sha256="b" * 64,
    )
    try:
        assert namespaces == [str(output.parent)]
        claim = json.loads((output.parent / ".hope_gvhmr_claim.json").read_text())
        assert claim["asset_id"] == "one"
        with pytest.raises(Q.QueueError, match="already exists"):
            Q.reserve_output_namespaces(
                [{"asset_id": "one", "path": str(output)}],
                prereg_sha256="a" * 64,
                execution_record_sha256="c" * 64,
            )
    finally:
        for descriptor in locks:
            os.close(descriptor)


def test_output_namespace_rejects_symlinked_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    output = linked / "stem" / "hmr4d_results.pt"
    with pytest.raises(Q.PreregError, match="symlink"):
        Q.reserve_output_namespaces(
            [{"asset_id": "one", "path": str(output)}],
            prereg_sha256="a" * 64,
            execution_record_sha256="b" * 64,
        )
