from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_motion_video_gvhmr_prereg.py"
SPEC = importlib.util.spec_from_file_location("validate_motion_video_gvhmr_prereg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_static_preregistration_is_closed_and_exact():
    static_result = V.validate_static_prereg(
        ROOT / "configs" / "motion_video_gvhmr_prereg_20260713.json"
    )
    motion_result = V.validate_static_prereg(
        ROOT / "configs" / "motion_video_gvhmr_motion_prereg_20260713.json"
    )
    assert static_result["prereg"]["processing_order"] == [
        "v12_forehand_block",
        "v12_backhand_block",
        "static_backhand_high_press",
        "lateral_step_left_1",
        "lateral_step_left_2",
        "lateral_step_right_1",
        "lateral_step_right_2",
    ]
    assert static_result["prereg"]["execution_batch"]["asset_ids"] == [
        "static_backhand_high_press"
    ]
    assert motion_result["prereg"]["execution_batch"]["asset_ids"] == [
        "lateral_step_left_1",
        "lateral_step_left_2",
        "lateral_step_right_1",
        "lateral_step_right_2",
    ]
    assert static_result["prereg"]["downstream_consumer_closure"]["post_gvhmr_consumer"] == "none"
    assert motion_result["prereg"]["execution_batch"]["v12_authorized"] is False


def test_prereg_whitelist_rejects_even_semantically_unused_byte_change(tmp_path):
    prereg = _payload(ROOT / "configs" / "motion_video_gvhmr_prereg_20260713.json")
    prereg["unregistered_note"] = "byte change must require a new committed preregistration"
    path = tmp_path / "motion_video_gvhmr_prereg_20260713.json"
    path.write_text(json.dumps(prereg), encoding="utf-8")
    with pytest.raises(V.PreregError, match="exact committed S0/M0"):
        V.validate_static_prereg(path)


def test_committed_franco_priority_review_keeps_contact_truth_null():
    review = _payload(
        ROOT / "configs" / "motion_video_franco_backhand_loop_visual_review_20260713.json"
    )
    V.validate_franco_priority_review(review)
    assert [row["nominal_event_anchor_frame"] for row in review["assets"]] == [49, 50]
    assert all(row["contact_truth"] is None for row in review["assets"])


def test_franco_priority_review_rejects_contact_truth_claim():
    review = _payload(
        ROOT / "configs" / "motion_video_franco_backhand_loop_visual_review_20260713.json"
    )
    review["assets"][0]["contact_truth"] = "claimed_contact"
    with pytest.raises(V.PreregError, match="contact truth must remain null"):
        V.validate_franco_priority_review(review)


def test_exact_batches_reject_v12_and_narrow_terminal_stance(tmp_path):
    static_prereg = _payload(ROOT / "configs" / "motion_video_gvhmr_prereg_20260713.json")
    static_prereg["execution_batch"]["asset_ids"] = ["v12_forehand_block"]
    static_path = tmp_path / "motion_video_gvhmr_prereg_20260713.json"
    static_path.write_text(json.dumps(static_prereg), encoding="utf-8")
    with pytest.raises(V.PreregError, match="exact S0/M0"):
        V.validate_static_prereg(static_path)

    motion_prereg = _payload(
        ROOT / "configs" / "motion_video_gvhmr_motion_prereg_20260713.json"
    )
    motion_prereg["future_robot_stance_contract"][
        "narrower_feet_together_substitute_allowed"
    ] = True
    motion_path = tmp_path / "motion_video_gvhmr_motion_prereg_20260713.json"
    motion_path.write_text(json.dumps(motion_prereg), encoding="utf-8")
    with pytest.raises(V.PreregError, match="narrower feet-together"):
        V.validate_static_prereg(motion_path)


def test_each_live_audit_view_contains_only_its_exact_batch():
    static = V.validate_static_prereg(
        ROOT / "configs" / "motion_video_gvhmr_prereg_20260713.json"
    )
    motion = V.validate_static_prereg(
        ROOT / "configs" / "motion_video_gvhmr_motion_prereg_20260713.json"
    )
    assert V.execution_intake(static)["processing_order"] == ["static_backhand_high_press"]
    assert V.execution_intake(motion)["processing_order"] == [
        "lateral_step_left_1",
        "lateral_step_left_2",
        "lateral_step_right_1",
        "lateral_step_right_2",
    ]


def test_manual_review_rejects_reordered_assets():
    intake = V.load_manifest(ROOT / "configs" / "motion_video_intake_20260713.json")
    review = _payload(ROOT / "configs" / "motion_video_manual_event_review_20260713.json")
    review["assets"] = list(reversed(review["assets"]))
    with pytest.raises(V.PreregError, match="exactly follow"):
        V.validate_manual_review(review, intake, intake["processing_order"])


def test_manual_review_rejects_anchor_outside_window():
    intake = V.load_manifest(ROOT / "configs" / "motion_video_intake_20260713.json")
    review = _payload(ROOT / "configs" / "motion_video_manual_event_review_20260713.json")
    review["assets"][0]["nominal_event_anchor_s"] = 2.1
    with pytest.raises(V.PreregError, match="anchor must lie"):
        V.validate_manual_review(review, intake, intake["processing_order"])


def test_manual_review_rejects_contact_truth_claim():
    intake = V.load_manifest(ROOT / "configs" / "motion_video_intake_20260713.json")
    review = _payload(ROOT / "configs" / "motion_video_manual_event_review_20260713.json")
    review["review_method"]["anchor_semantics"] = "true ball contact"
    with pytest.raises(V.PreregError, match="deny ball-contact truth"):
        V.validate_manual_review(review, intake, intake["processing_order"])


def test_manual_review_rejects_narrow_terminal_stance_substitute():
    intake = V.load_manifest(ROOT / "configs" / "motion_video_intake_20260713.json")
    review = _payload(ROOT / "configs" / "motion_video_manual_event_review_20260713.json")
    review["foot_stance_contract"]["rule"] = "finish feet together"
    with pytest.raises(V.PreregError, match="initial ready-window foot separation"):
        V.validate_manual_review(review, intake, intake["processing_order"])


def test_manual_review_binds_exactly_four_lateral_stance_candidates():
    intake = V.load_manifest(ROOT / "configs" / "motion_video_intake_20260713.json")
    review = _payload(ROOT / "configs" / "motion_video_manual_event_review_20260713.json")
    review["foot_stance_contract"]["applies_to"] = review["foot_stance_contract"][
        "applies_to"
    ][:-1]
    with pytest.raises(V.PreregError, match="exactly the four lateral"):
        V.validate_manual_review(review, intake, intake["processing_order"])


def test_exclusive_record_writer_never_overwrites(tmp_path):
    target = tmp_path / "record.json"
    V.write_json_exclusive(target, {"status": "first"})
    original = target.read_bytes()
    with pytest.raises(V.PreregError, match="already exists"):
        V.write_json_exclusive(target, {"status": "second"})
    assert target.read_bytes() == original


def test_attest_materializes_exact_launch_without_authorizing_downstream(monkeypatch, tmp_path):
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}\n", encoding="utf-8")
    intake_path = tmp_path / "intake.json"
    intake_path.write_text("{}\n", encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text("{}\n", encoding="utf-8")
    franco_review_path = tmp_path / "franco-review.json"
    franco_review_path.write_text("{}\n", encoding="utf-8")
    queue_path = tmp_path / "queue.py"
    queue_path.write_text("pass\n", encoding="utf-8")
    auditor_path = tmp_path / "auditor.py"
    auditor_path.write_text("pass\n", encoding="utf-8")
    record_path = tmp_path / "record.json"
    source_root = tmp_path / "raw"
    gvhmr_root = tmp_path / "GVHMR"
    python = tmp_path / "python"
    state_root = tmp_path / "state"
    for path in (source_root, gvhmr_root):
        path.mkdir()
    python.write_text("", encoding="utf-8")
    prereg = {
        "pod_contract": {
            "allowed_pod_ids": ["pod1", "pod2"],
            "execution_record_path": str(record_path),
            "state_root": str(state_root),
            "pod_id_semantics": "caller assertion",
        },
        "gvhmr_runtime": {"gpu": {"poll_seconds": 30.0, "wait_timeout_seconds": 0.0}},
        "intake": {"sha256": "a" * 64},
        "manual_event_review": {"sha256": "b" * 64},
        "franco_priority_context": {"sha256": "c" * 64},
        "processing_order": ["one"],
        "execution_batch": {"asset_ids": ["one"]},
    }
    static = {
        "prereg": prereg,
        "prereg_path": prereg_path.resolve(),
        "prereg_sha256": "c" * 64,
        "intake_path": intake_path.resolve(),
        "review_path": review_path.resolve(),
        "franco_review_path": franco_review_path.resolve(),
        "tool_paths": {"queue": queue_path, "result_auditor": auditor_path},
    }
    live = {
        "gvhmr_head": "d" * 40,
        "gvhmr_entrypoint_sha256": "e" * 64,
        "checkpoint_tree": {"files": 1, "bytes": 1, "sha256": "f" * 64},
        "python_environment": {"version": "Python test", "pip_freeze_sha256": "0" * 64},
        "nvidia_smi": {"path": "/usr/bin/nvidia-smi", "sha256": "1" * 64},
        "inputs": [],
    }
    monkeypatch.setattr(V, "validate_static_prereg", lambda *_args, **_kwargs: static)
    monkeypatch.setattr(V, "verify_live_contract", lambda *_args, **_kwargs: live)
    record = V.attest(
        prereg_path,
        pod_id="pod1",
        source_root=source_root,
        gvhmr_root=gvhmr_root,
        python=python,
        record_path=record_path,
        gpu=2,
        max_used_mib=19000,
    )
    assert record["status"] == "ready_for_exact_offline_gvhmr"
    assert record["downstream_consumer"] == "none_beyond_structural_auditor"
    assert record["launch_argv"][-4:] == [
        "--prereg",
        str(prereg_path.resolve()),
        "--execution-record",
        str(record_path.resolve()),
    ]
    assert _payload(record_path) == record
    with pytest.raises(V.PreregError, match="already exists"):
        V.attest(
            prereg_path,
            pod_id="pod1",
            source_root=source_root,
            gvhmr_root=gvhmr_root,
            python=python,
            record_path=record_path,
            gpu=2,
            max_used_mib=19000,
        )


def _record_fixture(monkeypatch, tmp_path):
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}\n", encoding="utf-8")
    record_path = tmp_path / "record.json"
    intake_path = tmp_path / "intake.json"
    intake_path.write_text("{}\n", encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text("{}\n", encoding="utf-8")
    franco_review_path = tmp_path / "franco-review.json"
    franco_review_path.write_text("{}\n", encoding="utf-8")
    queue_path = tmp_path / "queue.py"
    queue_path.write_text("pass\n", encoding="utf-8")
    auditor_path = tmp_path / "auditor.py"
    auditor_path.write_text("pass\n", encoding="utf-8")
    source_root = tmp_path / "raw"
    gvhmr_root = tmp_path / "GVHMR"
    python = tmp_path / "motion-python"
    state_root = tmp_path / "state"
    source_root.mkdir()
    gvhmr_root.mkdir()
    python.write_bytes(b"python")
    prereg = {
        "pod_contract": {
            "allowed_pod_ids": ["pod1", "pod2"],
            "execution_record_path": str(record_path),
            "staged_source_root": str(source_root),
            "state_root": str(state_root),
            "pod_id_semantics": "caller assertion",
        },
        "gvhmr_runtime": {
            "root": str(gvhmr_root),
            "python": {"executable": str(python)},
            "gpu": {
                "allowed_physical_indices": [0, 1, 2],
                "max_used_mib_before_each_asset": 19000,
                "poll_seconds": 30.0,
                "wait_timeout_seconds": 0.0,
            },
        },
        "intake": {"sha256": "a" * 64},
        "manual_event_review": {"sha256": "b" * 64},
        "franco_priority_context": {"sha256": "c" * 64},
        "processing_order": ["one"],
        "execution_batch": {"asset_ids": ["one"]},
    }
    static = {
        "prereg": prereg,
        "prereg_path": prereg_path.resolve(),
        "prereg_sha256": "c" * 64,
        "intake_path": intake_path.resolve(),
        "review_path": review_path.resolve(),
        "franco_review_path": franco_review_path.resolve(),
        "tool_paths": {"queue": queue_path, "result_auditor": auditor_path},
    }
    live = {
        "gvhmr_head": "d" * 40,
        "gvhmr_entrypoint_sha256": "e" * 64,
        "checkpoint_tree": {"files": 1, "bytes": 1, "sha256": "f" * 64},
        "python_environment": {"version": "Python test", "pip_freeze_sha256": "0" * 64},
        "nvidia_smi": {
            "path": "/usr/bin/nvidia-smi",
            "realpath": "/usr/bin/nvidia-smi",
            "bytes": 1,
            "sha256": "1" * 64,
        },
        "inputs": [],
    }
    host_python = V.regular_file_fingerprint(Path(V.sys.executable).resolve(), "host Python")
    record = {
        "status": "ready_for_exact_offline_gvhmr",
        "authorization_scope": "offline_gvhmr_video_to_smplx_and_structural_audit_only",
        "prereg_sha256": static["prereg_sha256"],
        "manifest_path": str(intake_path.resolve()),
        "manifest_sha256": prereg["intake"]["sha256"],
        "manual_event_review_sha256": prereg["manual_event_review"]["sha256"],
        "franco_priority_review_path": str(franco_review_path.resolve()),
        "franco_priority_review_sha256": prereg["franco_priority_context"]["sha256"],
        "manual_event_review_path": str(review_path.resolve()),
        "validator_path": str(Path(V.__file__).resolve()),
        "validator_sha256": V.sha256_file(Path(V.__file__).resolve()),
        "host_python": host_python,
        "queue_sha256": V.sha256_file(queue_path),
        "result_auditor_sha256": V.sha256_file(auditor_path),
        "processing_order": ["one"],
        "launch_argv": [
            host_python["realpath"],
            str(queue_path),
            "--prereg",
            str(prereg_path.resolve()),
            "--execution-record",
            str(record_path),
        ],
        "host_identity": V.host_identity(),
        "source_root": str(source_root),
        "gvhmr_root": str(gvhmr_root),
        "python": str(python),
        "state_root": str(state_root),
        "gpu_physical_index": 2,
        "max_used_mib": 19000,
        "poll_seconds": 30.0,
        "wait_timeout_seconds": 0.0,
        "static_camera": True,
        "pod_id": "pod1",
        "pod_id_semantics": "caller assertion",
        "downstream_consumer": "none_beyond_structural_auditor",
        "nvidia_smi": live["nvidia_smi"],
        "live_dependency_binding": live,
    }
    monkeypatch.setattr(V, "validate_static_prereg", lambda *_args, **_kwargs: static)
    monkeypatch.setattr(V, "verify_live_contract", lambda *_args, **_kwargs: live)
    return prereg_path, record_path, record


def test_launch_validation_binds_validator_argv_and_nvidia_smi(monkeypatch, tmp_path):
    prereg_path, record_path, record = _record_fixture(monkeypatch, tmp_path)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    result = V.validate_execution_record_for_launch(prereg_path, record_path)
    assert result["record"]["nvidia_smi"] == result["live"]["nvidia_smi"]

    for field, bad_value, message in (
        ("validator_sha256", "0" * 64, "validator SHA"),
        ("launch_argv", ["/tmp/fake"], "launch_argv"),
        (
            "nvidia_smi",
            {"path": "/tmp/fake", "realpath": "/tmp/fake", "bytes": 1, "sha256": "2" * 64},
            "nvidia-smi",
        ),
    ):
        changed = copy.deepcopy(record)
        changed[field] = bad_value
        record_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(V.PreregError, match=message):
            V.validate_execution_record_for_launch(prereg_path, record_path)


def test_launch_validation_rejects_runtime_fingerprint_drift(monkeypatch, tmp_path):
    prereg_path, record_path, record = _record_fixture(monkeypatch, tmp_path)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    changed_live = copy.deepcopy(record["live_dependency_binding"])
    changed_live["checkpoint_tree"]["sha256"] = "9" * 64
    monkeypatch.setattr(V, "verify_live_contract", lambda *_args, **_kwargs: changed_live)
    with pytest.raises(V.PreregError, match="live dependency/input binding changed"):
        V.validate_execution_record_for_launch(prereg_path, record_path)


def test_static_prereg_rejects_duplicate_forbidden_stage(monkeypatch, tmp_path):
    prereg = _payload(ROOT / "configs" / "motion_video_gvhmr_prereg_20260713.json")
    prereg["authorization"]["forbidden_stages"].append(
        prereg["authorization"]["forbidden_stages"][0]
    )
    path = tmp_path / "motion_video_gvhmr_prereg_20260713.json"
    path.write_text(json.dumps(prereg), encoding="utf-8")
    intake = ROOT / "configs" / "motion_video_intake_20260713.json"
    review = ROOT / "configs" / "motion_video_manual_event_review_20260713.json"
    franco_review = (
        ROOT / "configs" / "motion_video_franco_backhand_loop_visual_review_20260713.json"
    )

    def exact_file(binding, _repo_root, label):
        if label == "intake":
            return intake
        if label == "manual_event_review":
            return review
        if label == "franco_priority_context":
            return franco_review
        raise AssertionError("tool closure should not be reached after forbidden-stage failure")

    monkeypatch.setattr(V, "require_exact_file", exact_file)
    monkeypatch.setattr(V, "validate_franco_priority_review", lambda *_args, **_kwargs: None)
    with pytest.raises(V.PreregError, match="forbidden stages"):
        V.validate_static_prereg(path)


def test_static_prereg_rejects_unsafe_entrypoint(monkeypatch, tmp_path):
    prereg = _payload(ROOT / "configs" / "motion_video_gvhmr_prereg_20260713.json")
    prereg["gvhmr_runtime"]["entrypoint_relpath"] = "../outside.py"
    path = tmp_path / "motion_video_gvhmr_prereg_20260713.json"
    path.write_text(json.dumps(prereg), encoding="utf-8")
    intake = ROOT / "configs" / "motion_video_intake_20260713.json"
    review = ROOT / "configs" / "motion_video_manual_event_review_20260713.json"
    franco_review = (
        ROOT / "configs" / "motion_video_franco_backhand_loop_visual_review_20260713.json"
    )
    tool_lookup = {
        "tool_closure.intake_auditor": ROOT / "scripts" / "audit_motion_video_intake.py",
        "tool_closure.queue": ROOT / "scripts" / "run_motion_video_gvhmr_preregistered_queue.py",
        "tool_closure.result_auditor": ROOT / "scripts" / "audit_gvhmr_result.py",
        "tool_closure.legacy_intake_guard": ROOT / "scripts" / "run_motion_video_gvhmr_queue.py",
        "tool_closure.historical_20260711_queue_source_archive": ROOT
        / "docs"
        / "experiments"
        / "archive"
        / "run_motion_video_gvhmr_queue_20260711.py.gz",
    }

    def exact_file(_binding, _repo_root, label):
        if label == "intake":
            return intake
        if label == "manual_event_review":
            return review
        if label == "franco_priority_context":
            return franco_review
        return tool_lookup[label]

    original_resolve = V.resolve_repo_file

    def resolve_file(repo_root, relative, label):
        if label == "tool_closure.execution_validator":
            return Path(V.__file__).resolve()
        return original_resolve(repo_root, relative, label)

    monkeypatch.setattr(V, "require_exact_file", exact_file)
    monkeypatch.setattr(V, "resolve_repo_file", resolve_file)
    monkeypatch.setattr(V, "validate_franco_priority_review", lambda *_args, **_kwargs: None)
    with pytest.raises(V.PreregError, match="safe tools/demo/demo.py"):
        V.validate_static_prereg(path)
