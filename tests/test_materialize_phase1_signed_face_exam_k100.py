from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONSUMER = ROOT / "scripts/materialize_phase1_signed_face_exam_k100.py"
CONFIG = ROOT / "configs/phase1_signed_face_exam_k100_activation_prereg_20260714.json"
SCHEDULE_MODULE = ROOT / "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py"
SIGNED_FACE_SCORER = ROOT / "hope_training/whole_body_tracking/scripts/virtual_return_scorer.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load(CONSUMER, "signed_exam_paper_consumer_under_test")
S = _load(SCHEDULE_MODULE, "signed_exam_schedule_under_test")
V = _load(SIGNED_FACE_SCORER, "signed_exam_face_scorer_under_test")


class ArrayBox:
    """Small torch-like CPU wrapper for pure schedule/face-contract tests."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def item(self):
        return self.value.item()

    def __getitem__(self, key):
        return ArrayBox(self.value[key])

    def __setitem__(self, key, value):
        self.value[key] = value


def _manifest():
    return P.load_manifest(CONFIG, repo_root=ROOT)


def _ids(counts=(60, 60)):
    return tuple(
        tuple(P.sha256_bytes(f"signed-paper:{clip}:{row}".encode()) for row in range(count))
        for clip, count in enumerate(counts)
    )


def _write_schedule(path: Path, artifact) -> None:
    path.write_bytes(S.canonical_json_bytes(S.artifact_document(artifact)) + b"\n")


def _synthetic_bank(path: Path, counts=(60, 60)):
    path.write_bytes(b"private-bank-byte-fixture")
    qmax = max(counts)
    incoming = np.zeros((2, qmax, 3), dtype=np.float64)
    demanded_vel = np.zeros_like(incoming)
    demanded_normal = np.zeros_like(incoming)
    for clip, count in enumerate(counts):
        for row in range(count):
            incoming[clip, row] = [-1.0 - row / 1000.0, clip / 10.0, -0.2]
            demanded_vel[clip, row] = [2.0, row / 100.0, 0.3]
        demanded_normal[clip, :count, 0] = 1.0 if clip == 0 else -1.0
    return SimpleNamespace(
        counts=ArrayBox(np.asarray(counts, dtype=np.int64)),
        incoming_vel=ArrayBox(incoming),
        incoming_spin=ArrayBox(np.zeros_like(incoming)),
        demanded_vel=ArrayBox(demanded_vel),
        demanded_normal=ArrayBox(demanded_normal),
        metadata={},
        source_path=str(path),
    )


def test_frozen_manifest_is_paper_only_signed_and_full_denominator():
    value = _manifest()
    assert value["bank"]["sha256"] == P.EXPECTED_BANK["sha256"]
    assert value["bank"]["source_family_sha256"] == P.EXPECTED_BANK[
        "source_family_sha256"
    ]
    assert value["signed_face_contract"] == P.EXPECTED_FACE_CONTRACT
    assert value["paper"]["denominator"] == {
        "aggregate": 100,
        "forehand": 50,
        "backhand": 50,
    }
    assert value["paper"]["all_scheduled_attempts_in_denominator"] is True
    assert value["authorization"] == P.EXPECTED_AUTHORIZATION
    assert set(value["authorization"].values()) <= {True, False}


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["bank"].__setitem__("sha256", "0" * 64),
        lambda value: value["bank"].__setitem__("source_family_sha256", "0" * 64),
        lambda value: value["paper"].__setitem__("schedule_seed", 1),
        lambda value: value["paper"]["denominator"].__setitem__("forehand", 49),
        lambda value: value["signed_face_contract"].__setitem__(
            "signed_face_required", False
        ),
        lambda value: value["authorization"].__setitem__("judge_started", True),
        lambda value: value["source_bindings"].__setitem__(
            "signed_face_scorer", value["source_bindings"]["schedule_module"]
        ),
    ),
)
def test_manifest_mutations_fail_closed(tmp_path, mutation):
    value = json.loads(CONFIG.read_text())
    mutation(value)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value))
    with pytest.raises(P.PaperError, match="frozen paper contract"):
        P.load_manifest(path, repo_root=ROOT)


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(P.PaperError, match="duplicate JSON key"):
        P.load_json(path)


def test_schedule_is_deterministic_unique_balanced_and_new_bank_bound(tmp_path):
    ids = _ids()
    kwargs = dict(
        bank_sha256=P.EXPECTED_BANK["sha256"],
        clip_names=("forehand", "backhand"),
        question_ids=ids,
        per_clip_quota=50,
        schedule_seed=0,
        hold_range=(0, 100),
    )
    first = S.materialize_balanced_bank_exam_schedule(**kwargs)
    second = S.materialize_balanced_bank_exam_schedule(**kwargs)
    assert first == second
    assert len(first.items) == 100
    assert [sum(item.clip == clip for item in first.items) for clip in range(2)] == [50, 50]
    assert len({item.question_id for item in first.items}) == 100
    path = tmp_path / "new.schedule.json"
    _write_schedule(path, first)
    loaded, receipt = P.validate_schedule(
        schedule_module=S,
        schedule_path=path,
        bank_sha256=P.EXPECTED_BANK["sha256"],
        clip_names=("forehand", "backhand"),
        question_ids=ids,
        paper_contract=P.EXPECTED_PAPER,
    )
    assert loaded == first
    assert receipt["selected_per_side"] == {"forehand": 50, "backhand": 50}


def test_old_bank_schedule_is_rejected_even_if_shape_and_quota_match(tmp_path):
    ids = _ids()
    old = S.materialize_balanced_bank_exam_schedule(
        bank_sha256="d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096",
        clip_names=("forehand", "backhand"),
        question_ids=ids,
        per_clip_quota=50,
        schedule_seed=0,
        hold_range=(0, 100),
    )
    path = tmp_path / "old.schedule.json"
    _write_schedule(path, old)
    with pytest.raises(P.PaperError, match="exact-bank validation"):
        P.validate_schedule(
            schedule_module=S,
            schedule_path=path,
            bank_sha256=P.EXPECTED_BANK["sha256"],
            clip_names=("forehand", "backhand"),
            question_ids=ids,
            paper_contract=P.EXPECTED_PAPER,
        )


def test_repeated_question_and_side_shortage_are_rejected():
    ids = [list(side) for side in _ids()]
    ids[0][1] = ids[0][0]
    with pytest.raises(ValueError, match="duplicate atomic question ids"):
        S.materialize_balanced_bank_exam_schedule(
            bank_sha256=P.EXPECTED_BANK["sha256"],
            clip_names=("forehand", "backhand"),
            question_ids=ids,
            per_clip_quota=50,
        )
    with pytest.raises(ValueError, match="exceeds available"):
        S.materialize_balanced_bank_exam_schedule(
            bank_sha256=P.EXPECTED_BANK["sha256"],
            clip_names=("forehand", "backhand"),
            question_ids=_ids((50, 49)),
            per_clip_quota=50,
        )


def test_wrong_physical_face_and_unsigned_activation_are_rejected(tmp_path):
    bank = _synthetic_bank(tmp_path / "bank.bin")
    audit = P.validate_signed_targets(
        bank, P.EXPECTED_FACE_CONTRACT, signed_face_scorer=V
    )
    assert audit["forehand"]["physical_B_min_x"] == 1.0
    assert audit["backhand"]["physical_B_min_x"] == 1.0
    bank.demanded_normal[1, :60, 0] = 1.0
    with pytest.raises(P.PaperError, match="physical-B"):
        P.validate_signed_targets(
            bank, P.EXPECTED_FACE_CONTRACT, signed_face_scorer=V
        )

    schedule = {"path": "/tmp/paper", "file_sha256": "1" * 64}
    content = {
        "status": "paper_materialized_not_started",
        "manifest": {"sha256": "2" * 64},
        "consumer": {"sha256": "3" * 64},
        "bank": {"sha256": P.EXPECTED_BANK["sha256"]},
        "schedule": schedule,
        "signed_face_contract": {
            **P.EXPECTED_FACE_CONTRACT,
            "signed_face_required": False,
        },
        "scoring_denominator": P.EXPECTED_PAPER["denominator"],
        "authorization": P.EXPECTED_AUTHORIZATION,
    }
    document = P._activation_document(content)
    with pytest.raises(P.PaperError, match="exact signed-face"):
        P.validate_activation_document(
            document,
            manifest_sha256="2" * 64,
            consumer_sha256="3" * 64,
            bank_sha256=P.EXPECTED_BANK["sha256"],
            schedule_receipt=schedule,
        )


def test_activation_is_report_last_and_partial_output_cannot_be_reused(tmp_path, monkeypatch):
    manifest = copy.deepcopy(_manifest())
    bank_path = tmp_path / "bank.bin"
    bank = _synthetic_bank(bank_path)
    manifest["bank"] = {
        **manifest["bank"],
        "path": str(bank_path),
        "bytes": bank_path.stat().st_size,
        "sha256": P.sha256_file(bank_path),
        "question_counts": {"forehand": 60, "backhand": 60},
    }
    bank.metadata = {
        "schema_version": 3,
        "split": "exam",
        "physics_contract_sha256": manifest["bank"]["physics_contract_sha256"],
        "source_family_sha256": manifest["bank"]["source_family_sha256"],
    }
    output_root = tmp_path / "paper-output"
    manifest["output"] = {**manifest["output"], "root": str(output_root)}
    fake_loader = SimpleNamespace(load_question_bank=lambda *args, **kwargs: bank)
    original_load = P._load_module

    def load_bound(path, name):
        if path == manifest["_resolved_sources"]["bank_loader"]:
            return fake_loader
        if path == manifest["_resolved_sources"]["schedule_module"]:
            return S
        if path == manifest["_resolved_sources"]["signed_face_scorer"]:
            return V
        return original_load(path, name)

    monkeypatch.setattr(P, "_load_module", load_bound)
    original_write = P._write_exclusive

    def fail_activation(path, payload):
        if path.name.endswith("activation.json"):
            raise RuntimeError("injected activation write failure")
        original_write(path, payload)

    monkeypatch.setattr(P, "_write_exclusive", fail_activation)
    with pytest.raises(RuntimeError, match="injected"):
        P.consume(CONFIG, manifest, repo_root=ROOT)
    assert (output_root / manifest["output"]["schedule_basename"]).is_file()
    assert not (output_root / manifest["output"]["activation_basename"]).exists()

    monkeypatch.setattr(P, "_write_exclusive", original_write)
    with pytest.raises(P.PaperError, match="output root exists"):
        P.consume(CONFIG, manifest, repo_root=ROOT)
