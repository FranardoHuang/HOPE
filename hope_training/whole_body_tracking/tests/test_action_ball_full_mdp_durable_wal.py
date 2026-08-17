"""Pure forensic-reader counterexamples for the phase-1 ActionEpoch WAL."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

WAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/utils"
    / "action_ball_full_mdp_durable_wal.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "action_ball_full_mdp_durable_wal_direct_test", WAL_PATH
)
wal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wal)


SEGMENT = "1a:2b"
RANK = 7


def _telemetry(
    update=0, env_steps=4, operation=1, drain=1, start=0, end=1
):
    return {
        "ppo_update": update,
        "completed_environment_steps": env_steps,
        "epoch_operation_sequence": operation,
        "epoch_drain_sequence": drain,
        "epoch_commit_start": start,
        "epoch_commit_end": end,
    }


def _pair(telemetry, *, start=0, segment=SEGMENT, rank=RANK):
    _stdout, pending = wal.encode_pending(
        segment_id=segment, rank=rank, telemetry=telemetry
    )
    ack = wal.encode_epoch_ack(
        pending_line=pending,
        pending_byte_start=start,
        pending_byte_end=start + len(pending),
    )
    return pending, ack


def _scan(path, *, rank=RANK, segment=SEGMENT):
    return wal.read_forensic_committed_frontier(
        path, expected_rank=rank, expected_segment_id=segment
    )


def test_reader_distinguishes_complete_prefix_and_unterminated_eof_tail(
    tmp_path
):
    pending, ack = _pair(_telemetry())
    tail = b'{"kind":"action_ball_epoch_durable_pending_v2"'
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(pending + ack + tail)

    result = _scan(path)
    assert result == {
        "schema_version": 1,
        "kind": "action_ball_epoch_forensic_committed_frontier_v1",
        "status": "forensic_only_not_checkpoint_or_resume_authority",
        "diagnostic_unauthorized": True,
        "rank": RANK,
        "segment_id": SEGMENT,
        "complete_prefix_bytes": len(pending) + len(ack),
        "ignored_eof_tail_bytes": len(tail),
        "durable_epoch_ack_count": 1,
        "pending_without_epoch_ack": False,
        "committed_frontier": _telemetry(),
    }


def test_every_unterminated_next_pending_byte_cut_preserves_committed_frontier(
    tmp_path
):
    first_pending, first_ack = _pair(_telemetry())
    prefix = first_pending + first_ack
    next_pending, _next_ack = _pair(
        _telemetry(
            update=1, env_steps=8, operation=2, drain=2, start=1, end=2
        ),
        start=len(prefix),
    )
    path = tmp_path / "rank_0007.jsonl"
    for cut in range(len(next_pending)):
        path.write_bytes(prefix + next_pending[:cut])
        result = _scan(path)
        assert result["complete_prefix_bytes"] == len(prefix)
        assert result["ignored_eof_tail_bytes"] == cut
        assert result["pending_without_epoch_ack"] is False
        assert result["durable_epoch_ack_count"] == 1
        assert result["committed_frontier"] == _telemetry()


def test_reader_reports_complete_pending_only_without_commit_authority(tmp_path):
    pending, _ack = _pair(_telemetry())
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(pending)
    result = _scan(path)
    assert result["pending_without_epoch_ack"] is True
    assert result["durable_epoch_ack_count"] == 0
    assert result["committed_frontier"] is None
    assert "checkpoint" in result["status"]
    assert "resume" in result["status"]


def test_reader_rejects_malformed_complete_middle_line(tmp_path):
    pending, ack = _pair(_telemetry())
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(pending + b'{"bad":}\n' + ack)
    with pytest.raises(RuntimeError, match="malformed complete line"):
        _scan(path)


def test_reader_rejects_duplicate_key_in_complete_middle_line(tmp_path):
    pending, ack = _pair(_telemetry())
    duplicate = b'{"kind":"one","kind":"two"}\n'
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(pending + ack + duplicate)
    with pytest.raises(RuntimeError, match="malformed complete line"):
        _scan(path)


def test_reader_rejects_epoch_ack_without_pending(tmp_path):
    _pending, ack = _pair(_telemetry())
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(ack)
    with pytest.raises(RuntimeError, match="EPOCH_ACK without PENDING"):
        _scan(path)


@pytest.mark.parametrize("foreign", ("rank", "segment"))
def test_reader_rejects_foreign_rank_or_segment(tmp_path, foreign):
    segment = "3c:4d" if foreign == "segment" else SEGMENT
    rank = 8 if foreign == "rank" else RANK
    pending, ack = _pair(_telemetry(), segment=segment, rank=rank)
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(pending + ack)
    with pytest.raises(RuntimeError, match="foreign rank/segment"):
        _scan(path)


@pytest.mark.parametrize(
    "second",
    (
        _telemetry(update=2, env_steps=8, operation=2, drain=2, start=1, end=2),
        _telemetry(update=1, env_steps=8, operation=3, drain=2, start=1, end=2),
        _telemetry(update=1, env_steps=8, operation=2, drain=3, start=1, end=2),
        _telemetry(update=1, env_steps=8, operation=2, drain=2, start=0, end=2),
        _telemetry(update=1, env_steps=7, operation=2, drain=2, start=1, end=2),
        _telemetry(update=1, env_steps=9, operation=2, drain=2, start=1, end=2),
    ),
)
def test_reader_rejects_foreign_update_envsteps_operation_drain_or_commit_frontier(
    tmp_path, second
):
    first_pending, first_ack = _pair(_telemetry())
    start = len(first_pending) + len(first_ack)
    second_pending, second_ack = _pair(second, start=start)
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(first_pending + first_ack + second_pending + second_ack)
    with pytest.raises(RuntimeError, match="frontier is noncontiguous"):
        _scan(path)


@pytest.mark.parametrize("field", ("record_key", "status", "pending_byte_end"))
def test_reader_rejects_mutated_ack_key_status_or_pending_span(tmp_path, field):
    pending, ack = _pair(_telemetry())
    row = json.loads(ack)
    if field == "pending_byte_end":
        row[field] += 1
    else:
        row[field] = "foreign"
    bad_ack = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path = tmp_path / "rank_0007.jsonl"
    path.write_bytes(pending + bad_ack)
    with pytest.raises(RuntimeError, match="EPOCH_ACK"):
        _scan(path)
