import importlib.util
import inspect
from pathlib import Path
import sys

import pytest


SUPPORT_PATH = Path(__file__).with_name("test_action_ball_runtime.py")
SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "action_ball_runtime_highwater_test_support", SUPPORT_PATH
)
SUPPORT = importlib.util.module_from_spec(SUPPORT_SPEC)
assert SUPPORT_SPEC.loader is not None
sys.modules[SUPPORT_SPEC.name] = SUPPORT
SUPPORT_SPEC.loader.exec_module(SUPPORT)
R = SUPPORT.R


class RecordingPool(R.LazyActionTaskPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_calls = []

    def _validate_refill_batch(self, **kwargs):
        result = super()._validate_refill_batch(**kwargs)
        self.validation_calls.append(
            {
                "sample_draw_floor": kwargs.get("sample_draw_floor"),
                "staged_sample_draw_highwater": kwargs.get(
                    "staged_sample_draw_highwater"
                ),
                "staged_sample_draw_starts": tuple(
                    kwargs.get("staged_sample_draw_starts", ())
                ),
                "staged_sample_draw_ranges": tuple(
                    kwargs.get("staged_sample_draw_ranges", ())
                ),
                "last_sample_draw_end": result[2],
            }
        )
        return result


class ReverseTapeOrderSolver(SUPPORT.Solver):
    """Return per-birth rows in request order after sampling in reverse."""

    def solve_many(self, requests):
        self.batch_calls += 1
        batch_by_birth = {}
        for request in reversed(requests):
            self.requests.append(request)
            sample_index = self.highwater_by_uid.get(
                request.action_uid, (-1, 0)
            )[0] + 1
            receipt = self._make_task(
                request,
                sample_index,
                request.swing_generation_start,
            )
            self._record_task(receipt)
            self._record_assignment(request, (sample_index,))
            self.sequence += 1
            batch_by_birth[request.birth.canonical_sha256] = (
                R.ActionPoolRefillBatch(
                    action_uid=request.action_uid,
                    proposed_count=1,
                    proposal_sample_indices=(sample_index,),
                    receipts=(receipt,),
                )
            )
        return tuple(
            batch_by_birth[request.birth.canonical_sha256]
            for request in requests
        )


class OverlappingDrawRangeSolver(SUPPORT.Solver):
    def solve_many(self, requests):
        batches = list(super().solve_many(requests))
        first = batches[0].receipts[0]
        second = batches[1]
        forged = SUPPORT.replace(
            second.receipts[0],
            sample_draw_start=first.sample_draw_start + 1,
            sample_draw_end=first.sample_draw_end + 1,
        )
        batches[1] = R.ActionPoolRefillBatch(
            action_uid=second.action_uid,
            proposed_count=second.proposed_count,
            proposal_sample_indices=second.proposal_sample_indices,
            receipts=(forged,),
        )
        return tuple(batches)


class OffsetSecondDrawRangeSolver(SUPPORT.Solver):
    def __init__(self, second_start_offset):
        super().__init__()
        self.second_start_offset = second_start_offset

    def solve_many(self, requests):
        batches = list(super().solve_many(requests))
        first = batches[0].receipts[0]
        second = batches[1]
        sample_draw_start = (
            first.sample_draw_start + self.second_start_offset
        )
        forged = SUPPORT.replace(
            second.receipts[0],
            sample_draw_start=sample_draw_start,
            sample_draw_end=(
                sample_draw_start + R.SAMPLER_SAMPLE_DRAW_COUNT
            ),
        )
        batches[1] = R.ActionPoolRefillBatch(
            action_uid=second.action_uid,
            proposed_count=second.proposed_count,
            proposal_sample_indices=second.proposal_sample_indices,
            receipts=(forged,),
        )
        return tuple(batches)


class ReplayedIndexAndDrawRangeSolver(SUPPORT.Solver):
    def solve_many(self, requests):
        batches = list(super().solve_many(requests))
        first = batches[0].receipts[0]
        second = batches[1]
        forged = SUPPORT.replace(
            second.receipts[0],
            sample_index=first.sample_index,
            sample_draw_start=first.sample_draw_start + 1,
            sample_draw_end=first.sample_draw_end + 1,
        )
        batches[1] = R.ActionPoolRefillBatch(
            action_uid=second.action_uid,
            proposed_count=second.proposed_count,
            proposal_sample_indices=second.proposal_sample_indices,
            receipts=(forged,),
        )
        return tuple(batches)


def _pool_with_births(
    *,
    birth_count,
    diagnostic,
    refill_size,
    solver,
):
    broker, _provider = SUPPORT._broker(
        1, diagnostic_unauthorized=diagnostic
    )
    births = tuple(
        SUPPORT._reserve(broker, env_id=env_id)
        for env_id in range(birth_count)
    )
    broker.commit_many_true_reset(
        tuple(
            R.BirthCommitRequest(
                birth.env_id,
                birth.reset_generation,
                birth.canonical_sha256,
            )
            for birth in births
        )
    )
    broker.consume_many_true_reset(
        tuple(SUPPORT._claim(birth) for birth in births)
    )
    pool = RecordingPool(
        SUPPORT._bindings(1),
        SUPPORT._pins(),
        "no_move",
        refill_size=refill_size,
        diagnostic_unauthorized=diagnostic,
    )
    pool.bind_solver(solver)
    pool.bind_birth_authority(broker)
    return pool, births


def test_diagnostic_refills_chain_one_draw_highwater_per_uid():
    solver = SUPPORT.Solver()
    pool, births = _pool_with_births(
        birth_count=32,
        diagnostic=True,
        refill_size=1,
        solver=solver,
    )

    tasks = pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0)
            for birth in births
        )
    )

    calls = pool.validation_calls
    draw_ends = tuple(
        call["last_sample_draw_end"] for call in calls
    )
    assert len(tasks) == len(births)
    assert tuple(task.sample_index for task in tasks) == tuple(
        range(len(births))
    )
    assert len(calls) == len(births)
    assert tuple(
        call["staged_sample_draw_highwater"] for call in calls
    ) == (0, *draw_ends[:-1])
    assert all(call["sample_draw_floor"] == 0 for call in calls)
    assert all(
        not call["staged_sample_draw_ranges"] for call in calls
    )
    assert tuple(
        len(call["staged_sample_draw_starts"]) for call in calls
    ) == tuple(range(len(births)))
    assert all(
        previous < current
        for previous, current in zip(draw_ends, draw_ends[1:])
    )


def test_diagnostic_highwater_preserves_disjoint_reverse_tape_order():
    pool, births = _pool_with_births(
        birth_count=2,
        diagnostic=True,
        refill_size=1,
        solver=ReverseTapeOrderSolver(),
    )

    tasks = pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0)
            for birth in births
        )
    )

    assert tuple(task.sample_index for task in tasks) == (1, 0)
    assert tuple(
        call["staged_sample_draw_highwater"]
        for call in pool.validation_calls
    ) == (
        0,
        pool.validation_calls[0]["last_sample_draw_end"],
    )
    assert all(
        not call["staged_sample_draw_ranges"]
        for call in pool.validation_calls
    )


def test_diagnostic_highwater_rejects_cross_birth_overlap():
    pool, births = _pool_with_births(
        birth_count=2,
        diagnostic=True,
        refill_size=1,
        solver=OverlappingDrawRangeSolver(),
    )

    with pytest.raises(
        R.ActionBallContractError,
        match="sample draw range replayed/overlapped",
    ):
        pool.request_many(
            tuple(
                R.ActionTaskIssueRequest(birth, 0)
                for birth in births
            )
        )
    assert pool.materialized_action_uids == ()


@pytest.mark.parametrize(
    ("second_start_offset", "accepted"),
    (
        (-18, True),
        (-17, False),
        (17, False),
        (18, True),
    ),
    ids=(
        "left-touch",
        "left-one-draw-overlap",
        "right-one-draw-overlap",
        "right-touch",
    ),
)
def test_diagnostic_fixed_18_draw_overlap_boundaries(
    second_start_offset,
    accepted,
):
    assert R.SAMPLER_SAMPLE_DRAW_COUNT == 18
    pool, births = _pool_with_births(
        birth_count=2,
        diagnostic=True,
        refill_size=1,
        solver=OffsetSecondDrawRangeSolver(second_start_offset),
    )
    requests = tuple(
        R.ActionTaskIssueRequest(birth, 0) for birth in births
    )

    if accepted:
        tasks = pool.request_many(requests)
        assert tuple(task.sample_index for task in tasks) == (0, 1)
    else:
        with pytest.raises(
            R.ActionBallContractError,
            match="sample draw range replayed/overlapped",
        ):
            pool.request_many(requests)
        assert pool.materialized_action_uids == ()


def test_diagnostic_actual_index_error_precedes_draw_overlap():
    pool, births = _pool_with_births(
        birth_count=2,
        diagnostic=True,
        refill_size=1,
        solver=ReplayedIndexAndDrawRangeSolver(),
    )

    with pytest.raises(
        R.ActionBallContractError,
        match="sample index replayed/went backwards",
    ):
        pool.request_many(
            tuple(
                R.ActionTaskIssueRequest(birth, 0)
                for birth in births
            )
        )
    assert pool.materialized_action_uids == ()


def test_formal_refills_keep_cross_birth_range_validation():
    solver = SUPPORT.RoundInterleavedSolver()
    pool, births = _pool_with_births(
        birth_count=2,
        diagnostic=False,
        refill_size=2,
        solver=solver,
    )

    tasks = pool.request_many(
        tuple(
            R.ActionTaskIssueRequest(birth, 0)
            for birth in births
        )
    )

    assert tuple(task.sample_index for task in tasks) == (0, 1)
    assert len(pool.validation_calls) == 2
    assert pool.validation_calls[0]["sample_draw_floor"] == 0
    assert not pool.validation_calls[0][
        "staged_sample_draw_ranges"
    ]
    assert pool.validation_calls[1]["sample_draw_floor"] == 0
    assert len(
        pool.validation_calls[1]["staged_sample_draw_ranges"]
    ) == 2


def test_refill_error_priority_remains_index_before_draw_range():
    source = inspect.getsource(
        R.LazyActionTaskPool._validate_refill_batch
    )
    assert source.index(
        "task solver sample index replayed/went backwards"
    ) < source.index(
        "task solver sample draw range replayed/overlapped"
    )
    diagnostic_source = inspect.getsource(
        R.LazyActionTaskPool._request_many_diagnostic
    )
    assert "staged_sample_draw_ranges_by_uid" not in diagnostic_source
