"""Pure regressions for the finite, immutable Stage-1 BankExam paper."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "venue_ball_sampler.py"
SHARED_MODULE_PATH = ROOT / "scripts" / "bank_exam_schedule.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("venue_bank_schedule_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = _load_module()


def _load_shared_module():
    spec = importlib.util.spec_from_file_location(
        "shared_bank_schedule_under_test", SHARED_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S = _load_shared_module()


def _schedule(*, seed=7, k=None, counts=(4, 3), hold=(2, 5)):
    ids = [
        tuple(f"fh-atom-{row}" for row in range(counts[0])),
        tuple(f"bh-atom-{row}" for row in range(counts[1])),
    ]
    return V.materialize_bank_exam_schedule(
        q_counts=counts,
        clip_names=("forehand", "backhand"),
        question_ids=ids,
        hold_range=hold,
        schedule_seed=seed,
        schedule_k=k,
        bank_sha256="a" * 64,
    )


def test_full_schedule_is_immutable_unique_and_deterministic():
    items_a, sha_a = _schedule()
    items_b, sha_b = _schedule()
    assert isinstance(items_a, tuple)
    assert items_a == items_b
    assert sha_a == sha_b and len(sha_a) == 64
    assert len(items_a) == 7
    assert [item.schedule_index for item in items_a] == list(range(7))
    assert len({(item.clip, item.bank_row) for item in items_a}) == 7
    assert len({item.question_id for item in items_a}) == 7
    assert all(2 <= item.hold_steps <= 5 for item in items_a)
    with pytest.raises(FrozenInstanceError):
        items_a[0].bank_row = 99


def test_fixed_k_is_stratified_without_replacement():
    items, _ = _schedule(k=5, counts=(6, 4))
    assert len(items) == 5
    # Proportional largest-remainder allocation: 3 forehand + 2 backhand.
    assert [sum(item.clip == c for item in items) for c in range(2)] == [3, 2]
    assert len({(item.clip, item.bank_row) for item in items}) == 5
    with pytest.raises(ValueError):
        _schedule(k=8, counts=(4, 3))


def test_attempt_seed_provides_common_random_numbers_per_question():
    items, _ = _schedule()
    item = items[3]
    z_a = np.random.default_rng(item.attempt_seed).standard_normal((4, 31))
    z_b = np.random.default_rng(item.attempt_seed).standard_normal((4, 31))
    assert np.array_equal(z_a, z_b)
    # Noise scale multiplies a shared standard-Gaussian stream; it does not change the stream.
    assert np.array_equal(0.05 * z_a, 0.05 * z_b)
    assert items[0].attempt_seed != items[1].attempt_seed


def test_atomic_question_id_binds_bank_row_clip_spin_and_answer():
    kw = dict(
        bank_sha256="b" * 64,
        clip=0,
        bank_row=3,
        incoming_vel=np.array([-2.0, 0.1, -0.2]),
        incoming_spin=np.array([0.0, 0.0, 0.0]),
        demanded_vel=np.array([1.0, 2.0, 3.0]),
        demanded_normal=np.array([0.0, 1.0, 0.0]),
    )
    base = V.atomic_bank_question_id(**kw)
    assert base == V.atomic_bank_question_id(**kw)
    for key, replacement in (
        ("bank_row", 4),
        ("clip", 1),
        ("incoming_spin", np.array([0.0, 0.0, 0.1])),
        ("demanded_vel", np.array([1.0, 2.0, 3.1])),
        ("demanded_normal", np.array([0.1, 0.99, 0.0])),
        ("bank_sha256", "c" * 64),
    ):
        changed = dict(kw)
        changed[key] = replacement
        assert V.atomic_bank_question_id(**changed) != base


def test_shared_atomic_question_id_is_byte_compatible_with_production_sampler():
    kw = dict(
        bank_sha256="9" * 64,
        clip=1,
        bank_row=7,
        incoming_vel=np.array([-2.0, -0.0, -0.25], dtype=np.float32),
        incoming_spin=np.array([0.0, -0.0, 0.0], dtype=np.float32),
        demanded_vel=np.array([1.125, 2.25, -0.0], dtype=np.float32),
        demanded_normal=np.array([-0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert S.atomic_bank_question_id(**kw) == V.atomic_bank_question_id(**kw)


def test_sampler_cursor_is_finite_counts_samples_and_never_wraps():
    items, sha = _schedule(k=2, counts=(2, 2), hold=(0, 0))
    sampler = V.BankExamSampler.__new__(V.BankExamSampler)
    sampler.schedule = items
    sampler.schedule_sha256 = sha
    sampler._schedule_next = 0
    sampler.clip_names = ("forehand", "backhand")
    sampler.selected = [sum(item.clip == c for item in items) for c in range(2)]
    sampler.asked = [0, 0]
    sampler.wrapped = [0, 0]
    sampler.n_samples = sampler.n_solve_fail = sampler.n_sign_reject = sampler.iters_acc = 0
    sampler.q_contact = np.zeros((2, 3))
    sampler.q_income = np.zeros((2, 2, 3))
    sampler.q_spin = np.zeros((2, 2, 3))
    sampler.q_vel = np.ones((2, 2, 3))
    sampler.q_nrm = np.tile(np.array([0.0, 1.0, 0.0]), (2, 2, 1))
    sampler.q_landing = np.array([2.5, 0.0])

    draws = [sampler.sample(None), sampler.sample(None)]
    assert [draw.schedule_index for draw in draws] == [0, 1]
    assert sampler.exhausted and sampler.remaining == 0
    assert sampler.n_samples == 2 and sampler.asked == sampler.selected
    assert sampler.wrapped == [0, 0]
    with pytest.raises(RuntimeError, match="wrapping/repeating"):
        sampler.sample(None)


def test_denominator_report_uses_final_evaluation_exactness_not_bank_leg_only():
    sampler = V.BankExamSampler.__new__(V.BankExamSampler)
    sampler.contract_exact = True
    sampler.schedule = [object(), object()]
    sampler.schedule_seed = 0
    sampler.schedule_sha256 = "a" * 64
    sampler.schedule_hold_range = (0, 100)
    sampler.clip_names = ("forehand", "backhand")
    sampler.q_counts = [1, 1]
    sampler.selected = [1, 1]
    sampler.asked = [1, 1]
    sampler.wrapped = [0, 0]
    sampler.q_diff = np.zeros((2, 1), dtype=np.float64)
    sampler.bank_meta = {}
    sampler.q_landing = np.array([2.5, 0.0], dtype=np.float64)

    inherited = sampler.denominator_report()
    downgraded = sampler.denominator_report(evaluation_contract_exact=False)
    assert inherited[0] == "  evaluation_contract_exact=true"
    assert downgraded[0] == "  evaluation_contract_exact=false"


def _shared_inputs(counts=(6, 4)):
    names = ("forehand", "backhand")
    ids = tuple(
        tuple(
            hashlib.sha256(f"schema-v3-fixture:{clip}:{row}".encode()).hexdigest()
            for row in range(count)
        )
        for clip, count in enumerate(counts)
    )
    return names, ids


def _shared_schedule(*, counts=(6, 4), quota=3, seed=17, hold=(2, 5)):
    names, ids = _shared_inputs(counts)
    artifact = S.materialize_balanced_bank_exam_schedule(
        bank_sha256="c" * 64,
        clip_names=names,
        question_ids=ids,
        per_clip_quota=quota,
        schedule_seed=seed,
        hold_range=hold,
    )
    return artifact, names, ids


def _load_shared(path, names, ids, *, bank_sha="c" * 64):
    return S.load_schedule_artifact(
        path,
        expected_bank_sha256=bank_sha,
        expected_clip_names=names,
        expected_question_ids=ids,
    )


def test_shared_schedule_has_exact_per_clip_quota_and_hash_derived_order():
    artifact_a, names, ids = _shared_schedule()
    artifact_b, _, _ = _shared_schedule()
    assert artifact_a == artifact_b
    assert artifact_a.schema_version == 3
    assert artifact_a.bank_schema_version == 3
    assert artifact_a.no_wrap is True
    assert artifact_a.hold_semantics == "stand-policy-actions-then-raw-frame0-v1"
    assert artifact_a.schedule_k == 6
    assert [item.clip for item in artifact_a.items] == [0, 1, 0, 1, 0, 1]
    assert [sum(item.clip == clip for item in artifact_a.items) for clip in range(2)] == [3, 3]
    assert len({(item.clip, item.bank_row) for item in artifact_a.items}) == 6
    assert len(set(artifact_a.schedule_question_ids)) == 6
    assert all(item.question_id.startswith(f"{names[item.clip]}:") for item in artifact_a.items)
    assert all(2 <= item.hold_steps <= 5 for item in artifact_a.items)
    assert len({item.attempt_seed for item in artifact_a.items}) == 6
    assert artifact_a.schedule_sha256 == S.canonical_schedule_sha256(artifact_a)

    changed, _, _ = _shared_schedule(seed=18)
    assert changed.schedule_sha256 != artifact_a.schedule_sha256
    assert changed.items != artifact_a.items
    with pytest.raises(ValueError, match="exceeds available"):
        S.materialize_balanced_bank_exam_schedule(
            bank_sha256="c" * 64,
            clip_names=names,
            question_ids=ids,
            per_clip_quota=5,
        )


def test_shared_artifact_is_written_canonically_and_loads_strictly(tmp_path):
    artifact, names, ids = _shared_schedule()
    path = tmp_path / "paper.json"
    assert S.write_schedule_artifact(artifact, path) == str(path.resolve())
    assert path.read_bytes() == (
        S.canonical_json_bytes(S.artifact_document(artifact)) + b"\n"
    )
    assert _load_shared(path, names, ids) == artifact

    with pytest.raises(ValueError, match="bank SHA"):
        _load_shared(path, names, ids, bank_sha="d" * 64)
    with pytest.raises(ValueError, match="clip order"):
        S.load_schedule_artifact(
            path,
            expected_bank_sha256="c" * 64,
            expected_clip_names=names[::-1],
            expected_question_ids=ids[::-1],
        )
    wrong_ids = [list(values) for values in ids]
    wrong_ids[artifact.items[0].clip][artifact.items[0].bank_row] = "e" * 64
    with pytest.raises(ValueError, match="question id mismatch"):
        _load_shared(path, names, tuple(tuple(values) for values in wrong_ids))


def test_shared_artifact_rejects_tamper_even_before_bank_validation(tmp_path):
    artifact, names, ids = _shared_schedule()
    path = tmp_path / "tampered.json"
    document = S.artifact_document(artifact)
    document["items"][0]["hold_steps"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule SHA mismatch"):
        _load_shared(path, names, ids)


def test_shared_artifact_rejects_rehashed_noncanonical_question_order(tmp_path):
    artifact, names, ids = _shared_schedule()
    path = tmp_path / "reordered.json"
    document = S.artifact_document(artifact)
    # Swap two forehand rows while preserving round-robin clip positions and every row's own
    # question/seed/hold tuple.  Re-hashing defeats a checksum-only loader, but deterministic
    # rematerialization against the full bank must still reject the forged order.
    document["items"][0], document["items"][2] = (
        document["items"][2],
        document["items"][0],
    )
    document["items"][0]["schedule_index"] = 0
    document["items"][2]["schedule_index"] = 2
    document["schedule_sha256"] = S.canonical_schedule_sha256(document)
    path.write_bytes(S.canonical_json_bytes(document) + b"\n")
    with pytest.raises(ValueError, match="deterministic schema-v3 materialization"):
        _load_shared(path, names, ids)


def test_atomic_id_helpers_and_sampler_install_share_one_finite_paper(tmp_path):
    bank_path = tmp_path / "exam_bank.npz"
    bank_path.write_bytes(b"deterministic schema-v3 bank fixture")
    bank_sha = S.sha256_file(bank_path)
    counts = np.array([3, 2], dtype=np.int64)
    incoming = np.array(
        [
            [[-2.0, 0.1, -0.2], [-2.1, 0.0, -0.3], [-2.2, -0.1, -0.4]],
            [[-1.5, 0.2, -0.1], [-1.6, 0.1, -0.2], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    spin = np.zeros_like(incoming)
    demanded_vel = -incoming
    demanded_normal = np.zeros_like(incoming)
    demanded_normal[..., 1] = 1.0
    metadata = {"schema_version": 3, "split": "exam"}
    question_bank = SimpleNamespace(
        metadata=metadata,
        source_path=str(bank_path),
        counts=counts,
        incoming_vel=incoming,
        incoming_spin=spin,
        demanded_vel=demanded_vel,
        demanded_normal=demanded_normal,
    )
    question_ids = S.derive_question_bank_question_ids(question_bank)

    sampler = V.BankExamSampler.__new__(V.BankExamSampler)
    sampler.bank_meta = metadata
    sampler.bank_path = str(bank_path)
    sampler.bank_sha256 = bank_sha
    sampler.contract_exact = True
    sampler.clip_names = ("forehand", "backhand")
    sampler.q_counts = counts.tolist()
    sampler.q_income = incoming
    sampler.q_spin = spin
    sampler.q_vel = demanded_vel
    sampler.q_nrm = demanded_normal
    sampler.q_contact = np.zeros((2, 3), dtype=np.float64)
    sampler.q_landing = np.array([2.5, 0.0], dtype=np.float64)
    assert S.derive_sampler_question_ids(sampler) == question_ids
    sampler.contract_exact = False
    with pytest.raises(ValueError, match="historical diagnostic"):
        S.derive_sampler_question_ids(sampler)
    assert S.derive_sampler_question_ids(
        sampler, allow_inexact_contract=True
    ) == question_ids
    sampler.contract_exact = True

    artifact = S.materialize_balanced_bank_exam_schedule(
        bank_sha256=bank_sha,
        clip_names=sampler.clip_names,
        question_ids=question_ids,
        per_clip_quota=2,
        schedule_seed=29,
        hold_range=(0, 0),
    )
    assert S.install_schedule_on_sampler(sampler, artifact) is sampler
    assert sampler.schedule_sha256 == artifact.schedule_sha256
    assert sampler.selected == [2, 2]
    draws = [sampler.sample(None) for _ in range(4)]
    assert [draw.clip for draw in draws] == [0, 1, 0, 1]
    assert [draw.question_id for draw in draws] == list(artifact.schedule_question_ids)
    assert sampler.exhausted and sampler.remaining == 0
    with pytest.raises(RuntimeError, match="wrapping/repeating"):
        sampler.sample(None)

    sampler.contract_exact = False
    with pytest.raises(ValueError, match="historical diagnostic"):
        S.install_schedule_on_sampler(sampler, artifact)
    assert S.install_schedule_on_sampler(
        sampler, artifact, allow_inexact_contract=True
    ) is sampler
