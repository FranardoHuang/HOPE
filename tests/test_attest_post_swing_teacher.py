"""Dependency-light red-team tests for the one-shot teacher attestor."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/attest_post_swing_teacher.py"
SPEC = importlib.util.spec_from_file_location("attest_post_swing_teacher_test", SCRIPT)
A = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = A
SPEC.loader.exec_module(A)
QUEUE_RUNTIME_SCRIPT = ROOT / "hope_training/whole_body_tracking/scripts/lean_queue_runtime.py"
QUEUE_RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "attest_post_swing_teacher_queue_runtime_test", QUEUE_RUNTIME_SCRIPT
)
QUEUE_RUNTIME = importlib.util.module_from_spec(QUEUE_RUNTIME_SPEC)
sys.modules[QUEUE_RUNTIME_SPEC.name] = QUEUE_RUNTIME
QUEUE_RUNTIME_SPEC.loader.exec_module(QUEUE_RUNTIME)
CONTROLLER_SCRIPT = ROOT / "scripts/run_preregistered_post_swing_capture.py"
CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "attest_post_swing_teacher_controller_test", CONTROLLER_SCRIPT
)
CONTROLLER = importlib.util.module_from_spec(CONTROLLER_SPEC)
sys.modules[CONTROLLER_SPEC.name] = CONTROLLER
CONTROLLER_SPEC.loader.exec_module(CONTROLLER)


class _Tensor:
    def is_floating_point(self):
        return True


class _Finite:
    def __init__(self, count=0):
        self.count = count

    def __invert__(self):
        return self

    def sum(self):
        return self

    def item(self):
        return self.count


class _EvilPickle:
    def __init__(self, command: str):
        self.command = command

    def __reduce__(self):
        return os.system, (self.command,)


def _document(value):
    return A._json_document(value)


def _fixture(tmp_path: Path, monkeypatch, *, lineage=1, legacy_forgery=False):
    motion = tmp_path / "motion.npz"
    np.savez(motion, marker=np.array([1], dtype=np.int64))
    hard = {
        "schema_version": 3,
        "articulation_joint_names": ["j0", "j1"],
        "joint_velocity_limits": [5.0, 6.0],
        "motion_clips": [
            {"index": 0, "sha256": A.teacher.sha256_file(motion)}
        ],
    }
    hard_raw = _document(hard)
    capture = tmp_path / "capture"
    capture.mkdir()
    reviewed_producer_path = (
        ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    )
    capture_source_checkout = tmp_path / "capture-source-906"
    producer_path = (
        capture_source_checkout
        / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    )
    producer_path.parent.mkdir(parents=True)
    producer_path.write_bytes(reviewed_producer_path.read_bytes())
    producer_sha = A._sha(producer_path.read_bytes())
    root = np.zeros((4, 13), dtype=np.float32)
    root[:, 3] = 1.0
    zeros = np.zeros((4, 2), dtype=np.float32)
    state_path = capture / A.teacher.CAPTURE_STATE_NAME
    np.savez(
        state_path,
        root_state_origin_relative=root,
        joint_pos=zeros,
        joint_vel=zeros,
    )
    motion_rows = [{"index": 0, "sha256": A.teacher.sha256_file(motion)}]
    claim = {
        "schema_version": 1,
        "artifact_kind": A.teacher.CAPTURE_CLAIM_KIND,
        "producer_source_sha256": producer_sha,
        "runtime_hard_contract_sha256": A._sha(hard_raw),
        "target_count": 4,
        "motion_clips": motion_rows,
        "joint_names": ["j0", "j1"],
        "exclusive_create": True,
    }
    claim_path = capture / A.teacher.CAPTURE_CLAIM_NAME
    claim_path.write_bytes(_document(claim))
    result = {
        "schema_version": 2,
        "artifact_kind": A.teacher.CAPTURE_RESULT_KIND,
        "capture_contract": dict(A.teacher.CAPTURE_CONTRACT),
        "evidence": {
            "producer_source_sha256": producer_sha,
            "runtime_hard_contract_sha256": A._sha(hard_raw),
            "exclusive_claim_sha256": A._sha(claim_path.read_bytes()),
            "exclusive_claim_relative_path": A.teacher.CAPTURE_CLAIM_NAME,
            "no_clobber": True,
        },
        "motion_clips": motion_rows,
        "states": {
            "relative_path": A.teacher.CAPTURE_STATE_NAME,
            "sha256": A.teacher.sha256_file(state_path),
            "count": 4,
            "root_shape": [4, 13],
            "joint_pos_shape": [4, 2],
            "joint_vel_shape": [4, 2],
            "joint_names": ["j0", "j1"],
        },
    }
    if legacy_forgery:
        result["schema_version"] = 1
        result["producer"] = {
            "callback_method": "MotionCommand._capture_post_swing_states",
            "writer_source_sha256": "7" * 64,
            "callback_source_sha256": producer_sha,
            "runtime_hard_contract_sha256": A._sha(hard_raw),
            "no_clobber": True,
        }
        result["callback_batches"] = 1
        del result["evidence"]
    result_path = capture / A.teacher.CAPTURE_RESULT_NAME
    result_path.write_bytes(_document(result))

    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    hard_path = run / "params/training_contract.json"
    hard_path.write_bytes(hard_raw)
    checkpoint_path = run / "model_200.pt"
    checkpoint_path.write_bytes(b"immutable-fake-checkpoint")

    checkpoint_source_commit = "a" * 40
    claim_content = {
        "schema_version": 1,
        "job_id": "fresh-c-v1v2-clean-control-seed3",
        "action": "control",
        "pod": "pod2",
        "gpu": 1,
        "source": {"checkout": "/fake/checkpoint-source", "commit": checkpoint_source_commit},
        "run_name": "fresh_c_v1v2_clean_control_seed3",
        "run_dir": "/fake/checkpoint-source/logs/fresh_c_v1v2_clean_control_seed3",
        "runtime_binding": True,
        "seed": 3,
        "budget": {
            "num_envs": 4096,
            "max_iterations": 1001,
            "save_interval": 100,
            "milestones": [200, 500, 1000],
        },
        "inputs": {
            "motion": {"action": "shared", "bindings": {}},
            "bank": {"action": "shared", "train_path": "/fake/train.npz", "train_arg": "task.bank"},
            "exam": {"action": "shared", "path": "/fake/exam.npz", "family": "schema3"},
        },
        "training_argv_without_claim": ["/fake/python", "scripts/train.py", "seed=3"],
    }
    claim_sha = A._sha(A._canonical_content(claim_content))
    claim = {
        "schema_version": 2,
        "content": claim_content,
        "content_sha256": claim_sha,
        "training_argv": [
            *claim_content["training_argv_without_claim"],
            f"++training_launch_claim_sha256={claim_sha}",
        ],
    }
    claim_path = tmp_path / "launch_claim.json"
    claim_path.write_bytes(_document(claim))

    checkpoint = {
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": A._sha(hard_raw),
            "training_contract_lineage_exact": lineage,
            "training_launch_claim_sha256": claim_sha,
        },
        "model_state_dict": {"weight": _Tensor()},
    }
    def restricted_load(*_args, **kwargs):
        assert kwargs.get("weights_only") is True
        return checkpoint

    fake_torch = types.SimpleNamespace(
        Tensor=_Tensor,
        load=restricted_load,
        isfinite=lambda value: _Finite(0),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        A,
        "_git_state",
        lambda checkout, expected_commit, label: {"commit": expected_commit, "clean": True},
    )
    monkeypatch.setattr(
        A,
        "_current_git_state",
        lambda checkout, label: {
            "commit": "9" * 40 if label == "capture producer source checkout" else "e" * 40,
            "clean": True,
        },
    )
    output_receipt = capture / "teacher_receipt.json"
    retry_authorization = {
        "schema_version": 1,
        "artifact_kind": A.RETRY_AUTHORIZATION_KIND,
        "authorization_id": "test-v3-attestor-attempt2",
        "v3_plan": {"plan_id": capture.name, "file_sha256": "f" * 64},
        "capture": {
            "output_directory": str(capture),
            "output_receipt": str(output_receipt),
            "capture_claim_sha256": A._sha(
                (capture / A.teacher.CAPTURE_CLAIM_NAME).read_bytes()
            ),
            "states_sha256": A._sha(state_path.read_bytes()),
            "result_sha256": A._sha(result_path.read_bytes()),
            "state_count": 4,
        },
        "teacher": {
            "checkpoint_sha256": A._sha(checkpoint_path.read_bytes()),
            "hard_contract_sha256": A._sha(hard_path.read_bytes()),
            "launch_claim_content_sha256": claim_sha,
        },
        "capture_source": {
            "commit": "9" * 40,
            "producer_source_sha256": producer_sha,
        },
        "attestor_source": {
            "commit": "e" * 40,
            "attestor_source_sha256": A._sha(SCRIPT.read_bytes()),
        },
        "decision": {
            "capture_retry_authorized": False,
            "attestor_attempt2_authorized": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
        },
    }
    retry_authorization_path = tmp_path / "retry_authorization.json"
    retry_authorization_path.write_bytes(_document(retry_authorization))
    args = types.SimpleNamespace(
        capture_result=result_path,
        checkpoint=checkpoint_path,
        hard_contract=hard_path,
        launch_claim=claim_path,
        capture_source_checkout=capture_source_checkout,
        motion=[motion],
        root_linear_limit_mps=2.0,
        root_angular_limit_radps=4.0,
        retry_authorization=retry_authorization_path,
        expected_retry_authorization_sha256=A._sha(retry_authorization_path.read_bytes()),
        output_receipt=output_receipt,
    )
    return args


def test_attestor_binds_real_checkpoint_contract_sources_and_exclusive_claim(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    result = A.attest(args)
    assert result["count"] == 4
    receipt_path = Path(result["receipt"])
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt_raw == A._json_document(receipt)
    assert receipt_raw.endswith(b"\n") and not receipt_raw.endswith(b"\n\n")
    assert result["sha256"] == A._sha(receipt_raw)
    assert receipt["schema_version"] == 2
    assert receipt["teacher"]["fresh_lineage"] is True
    assert receipt["attestation"]["checkpoint_source"]["commit"] == "a" * 40
    assert receipt["attestation"]["schema_version"] == 2
    assert receipt["attestation"]["capture_source"] == {
        "commit": "9" * 40,
        "clean": True,
        "producer_source_sha256": A._sha(
            (
                args.capture_source_checkout
                / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
            ).read_bytes()
        ),
    }
    assert receipt["attestation"]["attestor_source"] == {
        "commit": "e" * 40,
        "clean": True,
        "attestor_source_sha256": A._sha(SCRIPT.read_bytes()),
    }
    assert receipt["attestation"]["retry_authorization"] == {
        "authorization_id": "test-v3-attestor-attempt2",
        "file_sha256": args.expected_retry_authorization_sha256,
        "v3_plan_file_sha256": "f" * 64,
    }
    with pytest.raises(A.AttestationError, match="already exists"):
        A.attest(args)
    assert receipt_path.read_bytes() == receipt_raw


def test_split_lineage_receipt_passes_loader_and_controller_status_binding(
    tmp_path, monkeypatch
):
    args = _fixture(tmp_path, monkeypatch)
    result = A.attest(args)
    receipt_raw = Path(result["receipt"]).read_bytes()
    receipt = json.loads(receipt_raw)
    launch_claim = json.loads(args.launch_claim.read_bytes())
    producer_path = (
        args.capture_source_checkout
        / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    )
    plan = {
        "teacher_checkpoint": {
            "training_source_commit": "a" * 40,
            "sha256": A._sha(args.checkpoint.read_bytes()),
            "hard_contract": {"sha256": A._sha(args.hard_contract.read_bytes())},
            "launch_claim": {"content_sha256": launch_claim["content_sha256"]},
        },
        "ordered_motion_inputs": [
            {"path": str(args.motion[0]), "sha256": A._sha(args.motion[0].read_bytes())}
        ],
        "capture_source": {
            "commit": "9" * 40,
            "files": {
                "producer": {
                    "sha256": A._sha(producer_path.read_bytes()),
                }
            },
        },
    }
    monkeypatch.setattr(
        CONTROLLER,
        "_status_retry_authorization",
        lambda *_args: {
            "attestor_source": receipt["attestation"]["attestor_source"],
            "receipt_binding": receipt["attestation"]["retry_authorization"],
            "status_source_commit": "b" * 40,
        },
    )
    CONTROLLER._validate_teacher_receipt_binding(
        plan,
        args.capture_result.parent,
        receipt_raw,
        current_script=CONTROLLER_SCRIPT,
        plan_sha256="f" * 64,
    )
    assert (
        receipt["attestation"]["capture_source"]["commit"]
        != receipt["attestation"]["attestor_source"]["commit"]
    )


def test_real_schema2_claim_hashes_content_without_document_newline(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    raw = args.launch_claim.read_bytes()
    envelope = json.loads(raw)
    content = envelope["content"]
    accepted_content, digest, _, _ = A._claim(raw)
    assert accepted_content == content
    assert raw.endswith(b"\n")
    assert digest == A._sha(A._canonical_content(content))
    assert digest == QUEUE_RUNTIME.canonical_sha256(content)
    assert digest != A._sha(A._json_document(content))


def test_schema2_claim_rejects_digest_that_includes_document_newline(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    envelope = json.loads(args.launch_claim.read_bytes())
    envelope["content_sha256"] = A._sha(A._json_document(envelope["content"]))
    args.launch_claim.write_bytes(A._json_document(envelope))
    with pytest.raises(A.AttestationError, match="canonical digest mismatch"):
        A.attest(args)
    assert not args.output_receipt.exists()


@pytest.mark.parametrize("dirty_label", ["capture producer source checkout", "attestor source checkout"])
def test_attestor_rejects_dirty_producer_or_attestor_checkout(
    tmp_path, monkeypatch, dirty_label
):
    args = _fixture(tmp_path, monkeypatch)

    def dirty(checkout, label):
        if label == dirty_label:
            raise A.AttestationError(f"{label} is not clean")
        return {"commit": "9" * 40 if "capture producer" in label else "e" * 40, "clean": True}

    monkeypatch.setattr(A, "_current_git_state", dirty)
    with pytest.raises(A.AttestationError, match="is not clean"):
        A.attest(args)
    assert not args.output_receipt.exists()


@pytest.mark.parametrize("field", ["commit", "attestor_source_sha256"])
def test_attestor_rejects_retry_authorization_for_different_clean_attestor(
    tmp_path, monkeypatch, field
):
    args = _fixture(tmp_path, monkeypatch)
    authorization = json.loads(args.retry_authorization.read_bytes())
    authorization["attestor_source"][field] = (
        "d" * 40 if field == "commit" else "d" * 64
    )
    args.retry_authorization.write_bytes(A._json_document(authorization))
    args.expected_retry_authorization_sha256 = A._sha(
        args.retry_authorization.read_bytes()
    )
    with pytest.raises(A.AttestationError, match="rebound from the immutable v3 attempt"):
        A.attest(args)
    assert not args.output_receipt.exists()


def test_attestor_rejects_legacy_public_writer_callback_label_forgery(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch, legacy_forgery=True)
    with pytest.raises(A.AttestationError, match="keys differ"):
        A.attest(args)
    assert not args.output_receipt.exists()


def test_attestor_rejects_checkpoint_without_exact_fresh_lineage(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch, lineage=0)
    with pytest.raises(A.AttestationError, match="fresh/exact lineage"):
        A.attest(args)
    assert not args.output_receipt.exists()


def test_checkpoint_weights_only_rejects_malicious_pickle_without_execution(tmp_path, monkeypatch):
    marker = tmp_path / "pickle_executed"
    raw = pickle.dumps(_EvilPickle(f"touch {marker}"), protocol=4)

    def restricted_loader(stream, *, weights_only, **_kwargs):
        if weights_only is not True:  # exact regression model for the former unsafe call
            return pickle.loads(stream.read())
        raise RuntimeError("restricted weights-only unpickler rejected GLOBAL os.system")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(Tensor=_Tensor, load=restricted_loader),
    )
    with pytest.raises(A.AttestationError, match="cannot load actual checkpoint bytes"):
        A._checkpoint(raw, "a" * 64, "b" * 64)
    assert not marker.exists()
