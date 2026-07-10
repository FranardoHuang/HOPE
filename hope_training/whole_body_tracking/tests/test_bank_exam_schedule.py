"""Pure regressions for the finite, immutable Stage-1 BankExam paper."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "venue_ball_sampler.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("venue_bank_schedule_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = _load_module()


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
