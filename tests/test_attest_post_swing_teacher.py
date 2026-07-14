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


def _canonical(value):
    return A._canonical(value)


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
    hard_raw = _canonical(hard)
    capture = tmp_path / "capture"
    capture.mkdir()
    producer_path = (
        ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    )
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
    claim_path.write_bytes(_canonical(claim))
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
    result_path.write_bytes(_canonical(result))

    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    hard_path = run / "params/training_contract.json"
    hard_path.write_bytes(hard_raw)
    checkpoint_path = run / "model_200.pt"
    checkpoint_path.write_bytes(b"immutable-fake-checkpoint")

    checkpoint_source_commit = "a" * 40
    claim_content = {
        "schema_version": 1,
        "source": {"checkout": "/fake/checkpoint-source", "commit": checkpoint_source_commit},
    }
    claim_sha = A._sha(_canonical(claim_content))
    claim = {"schema_version": 2, "content": claim_content, "content_sha256": claim_sha}
    claim_path = tmp_path / "launch_claim.json"
    claim_path.write_bytes(_canonical(claim))

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
        A.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="b" * 40 + "\n"),
    )
    args = types.SimpleNamespace(
        capture_result=result_path,
        checkpoint=checkpoint_path,
        hard_contract=hard_path,
        launch_claim=claim_path,
        capture_source_checkout=ROOT,
        motion=[motion],
        root_linear_limit_mps=2.0,
        root_angular_limit_radps=4.0,
        output_receipt=capture / "teacher_receipt.json",
    )
    return args


def test_attestor_binds_real_checkpoint_contract_sources_and_exclusive_claim(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    result = A.attest(args)
    assert result["count"] == 4
    receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["teacher"]["fresh_lineage"] is True
    assert receipt["attestation"]["checkpoint_source"]["commit"] == "a" * 40
    assert receipt["attestation"]["capture_source"]["commit"] == "b" * 40
    with pytest.raises(A.AttestationError, match="already exists"):
        A.attest(args)


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
