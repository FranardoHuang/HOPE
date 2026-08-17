"""Dependency-light forensic chronology for the diagnostic ActionEpoch WAL.

This module deliberately has no checkpoint, restore, replay, Torch, or Isaac
dependency.  A committed frontier here means only that a complete PENDING row
was followed by its independently fsynced EPOCH_ACK row in the same segment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat


SCHEMA_VERSION = 2
WAL_KIND = "action_ball_epoch_optimizer_update_durable_wal_v2"
PENDING_KIND = "action_ball_epoch_durable_pending_v2"
EPOCH_ACK_KIND = "action_ball_epoch_durable_ack_v2"
PENDING_STATUS = "optimizer_succeeded_durable_pending_destructive_ack"
EPOCH_ACK_STATUS = "destructive_epoch_ack_durable"
FORENSIC_KIND = "action_ball_epoch_forensic_committed_frontier_v1"
FORENSIC_STATUS = "forensic_only_not_checkpoint_or_resume_authority"

_FRONTIER_FIELDS = (
    "ppo_update",
    "completed_environment_steps",
    "epoch_operation_sequence",
    "epoch_drain_sequence",
    "epoch_commit_start",
    "epoch_commit_end",
)
_PENDING_KEYS = {
    "schema_version", "kind", "status", "record_key",
    "diagnostic_unauthorized", "segment_id", "rank",
    "pending_ack_telemetry",
} | set(_FRONTIER_FIELDS)
_ACK_KEYS = {
    "schema_version", "kind", "status", "record_key",
    "diagnostic_unauthorized", "segment_id", "rank",
    "pending_record_key", "pending_byte_start",
    "pending_byte_end",
} | set(_FRONTIER_FIELDS)


def _exact_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"durable WAL {name} must be a nonnegative exact int")
    return value


def _identity(segment_id: object, rank: object) -> tuple[str, int]:
    if (
        type(segment_id) is not str
        or not segment_id
        or len(segment_id) > 128
        or any(character not in "0123456789abcdef:" for character in segment_id)
    ):
        raise RuntimeError("durable WAL segment_id is invalid")
    return segment_id, _exact_int(rank, name="rank")


def _row_key(kind: str, segment_id: str, rank: int, update: int) -> str:
    return f"{kind}:{segment_id}:rank={rank}:update={update}"


def _line(record: dict) -> bytes:
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    if encoded.count(b"\n") != 1:
        raise RuntimeError("durable WAL row is not one canonical JSONL line")
    return encoded


def encode_pending(
    *, segment_id: str, rank: int, telemetry: dict
) -> tuple[bytes, bytes]:
    """Encode the pre-destructive-ACK row and exact stdout telemetry bytes."""

    segment_id, rank = _identity(segment_id, rank)
    if type(telemetry) is not dict:
        raise RuntimeError("durable WAL pending telemetry is not an exact dict")
    frontier = {
        name: _exact_int(telemetry.get(name), name=name)
        for name in _FRONTIER_FIELDS
    }
    update = frontier["ppo_update"]
    if (
        frontier["completed_environment_steps"] <= 0
        or frontier["epoch_operation_sequence"] <= 0
        or frontier["epoch_drain_sequence"] <= 0
        or frontier["epoch_commit_end"] < frontier["epoch_commit_start"]
    ):
        raise RuntimeError("durable WAL pending frontier is invalid")
    ack_json = json.dumps(
        telemetry, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if b"\n" in ack_json:
        raise RuntimeError("durable WAL stdout telemetry contains a newline")
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": PENDING_KIND,
        "status": PENDING_STATUS,
        "record_key": _row_key("PENDING", segment_id, rank, update),
        "diagnostic_unauthorized": True,
        "segment_id": segment_id,
        "rank": rank,
        **frontier,
        "pending_ack_telemetry": telemetry,
    }
    return ack_json, _line(record)


def encode_epoch_ack(
    *, pending_line: bytes, pending_byte_start: int, pending_byte_end: int
) -> bytes:
    """Encode the post-owner-ACK row referencing the exact pending byte span."""

    if (
        type(pending_line) is not bytes
        or not pending_line.endswith(b"\n")
        or pending_line.count(b"\n") != 1
    ):
        raise RuntimeError("durable WAL pending line ABI differs")
    start = _exact_int(pending_byte_start, name="pending_byte_start")
    end = _exact_int(pending_byte_end, name="pending_byte_end")
    if end != start + len(pending_line):
        raise RuntimeError("durable WAL pending byte span differs")
    pending = _decode_line(pending_line[:-1], line_number=0)
    _validate_pending(pending)
    update = pending["ppo_update"]
    ack = {
        "schema_version": SCHEMA_VERSION,
        "kind": EPOCH_ACK_KIND,
        "status": EPOCH_ACK_STATUS,
        "record_key": _row_key(
            "EPOCH_ACK", pending["segment_id"], pending["rank"], update
        ),
        "diagnostic_unauthorized": True,
        "segment_id": pending["segment_id"],
        "rank": pending["rank"],
        **{name: pending[name] for name in _FRONTIER_FIELDS},
        "pending_record_key": pending["record_key"],
        "pending_byte_start": start,
        "pending_byte_end": end,
    }
    return _line(ack)


def _decode_line(raw: bytes, *, line_number: int) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except BaseException as exc:
        raise RuntimeError(
            f"durable WAL has a malformed complete line {line_number}"
        ) from exc
    if type(value) is not dict:
        raise RuntimeError(f"durable WAL line {line_number} is not an exact object")
    return value


def _validate_pending(row: dict) -> None:
    if (
        set(row) != _PENDING_KEYS
        or row.get("schema_version") != SCHEMA_VERSION
        or row.get("kind") != PENDING_KIND
        or row.get("status") != PENDING_STATUS
        or row.get("diagnostic_unauthorized") is not True
    ):
        raise RuntimeError("durable WAL PENDING schema/status differs")
    segment_id, rank = _identity(row["segment_id"], row["rank"])
    frontier = {name: _exact_int(row[name], name=name) for name in _FRONTIER_FIELDS}
    if (
        row["record_key"]
        != _row_key("PENDING", segment_id, rank, frontier["ppo_update"])
        or frontier["completed_environment_steps"] <= 0
        or frontier["epoch_operation_sequence"] <= 0
        or frontier["epoch_drain_sequence"] <= 0
        or frontier["epoch_commit_end"] < frontier["epoch_commit_start"]
        or type(row["pending_ack_telemetry"]) is not dict
        or any(
            row["pending_ack_telemetry"].get(name) != value
            for name, value in frontier.items()
        )
    ):
        raise RuntimeError("durable WAL PENDING identity/frontier differs")


def _validate_ack(row: dict) -> None:
    if (
        set(row) != _ACK_KEYS
        or row.get("schema_version") != SCHEMA_VERSION
        or row.get("kind") != EPOCH_ACK_KIND
        or row.get("status") != EPOCH_ACK_STATUS
        or row.get("diagnostic_unauthorized") is not True
    ):
        raise RuntimeError("durable WAL EPOCH_ACK schema/status differs")
    segment_id, rank = _identity(row["segment_id"], row["rank"])
    frontier = {name: _exact_int(row[name], name=name) for name in _FRONTIER_FIELDS}
    if (
        row["record_key"]
        != _row_key("EPOCH_ACK", segment_id, rank, frontier["ppo_update"])
        or type(row["pending_record_key"]) is not str
        or _exact_int(row["pending_byte_start"], name="pending_byte_start")
        > _exact_int(row["pending_byte_end"], name="pending_byte_end")
    ):
        raise RuntimeError("durable WAL EPOCH_ACK identity/span differs")


def read_forensic_committed_frontier(
    path: str | Path, *, expected_rank: int, expected_segment_id: str
) -> dict:
    """Strictly scan stable bytes; return no replay or checkpoint authority."""

    expected_segment_id, expected_rank = _identity(
        expected_segment_id, expected_rank
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(Path(path), flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("durable WAL reader requires a regular file")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("durable WAL shortened during forensic read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size) != (
            before.st_dev, before.st_ino, before.st_size
        ):
            raise RuntimeError("durable WAL changed during forensic read")
    finally:
        os.close(fd)
    data = b"".join(chunks)
    prefix_end = data.rfind(b"\n") + 1
    prefix, tail = data[:prefix_end], data[prefix_end:]
    pending = None
    committed = None
    environment_step_stride = None
    ack_count = 0
    offset = 0
    for line_number, raw in enumerate(prefix.splitlines(keepends=True), 1):
        if not raw.endswith(b"\n") or raw == b"\n":
            raise RuntimeError(f"durable WAL has a bad complete line {line_number}")
        row = _decode_line(raw[:-1], line_number=line_number)
        if row.get("kind") == PENDING_KIND:
            _validate_pending(row)
            if pending is not None:
                raise RuntimeError("durable WAL has a second PENDING before EPOCH_ACK")
            if row["rank"] != expected_rank or row["segment_id"] != expected_segment_id:
                raise RuntimeError("durable WAL PENDING has foreign rank/segment")
            if committed is None:
                environment_step_stride = row["completed_environment_steps"]
                exact = (
                    row["ppo_update"] == 0
                    and row["epoch_operation_sequence"] == 1
                    and row["epoch_drain_sequence"] == 1
                    and row["epoch_commit_start"] == 0
                )
            else:
                exact = (
                    row["ppo_update"] == committed["ppo_update"] + 1
                    and row["completed_environment_steps"]
                    == committed["completed_environment_steps"]
                    + environment_step_stride
                    and row["epoch_operation_sequence"]
                    == committed["epoch_operation_sequence"] + 1
                    and row["epoch_drain_sequence"]
                    == committed["epoch_drain_sequence"] + 1
                    and row["epoch_commit_start"] == committed["epoch_commit_end"]
                )
            if not exact:
                raise RuntimeError("durable WAL PENDING frontier is noncontiguous")
            pending = (row, offset, offset + len(raw))
        elif row.get("kind") == EPOCH_ACK_KIND:
            _validate_ack(row)
            if pending is None:
                raise RuntimeError("durable WAL has EPOCH_ACK without PENDING")
            source, start, end = pending
            if (
                row["rank"] != expected_rank
                or row["segment_id"] != expected_segment_id
                or row["pending_record_key"] != source["record_key"]
                or row["pending_byte_start"] != start
                or row["pending_byte_end"] != end
                or any(row[name] != source[name] for name in _FRONTIER_FIELDS)
            ):
                raise RuntimeError("durable WAL EPOCH_ACK reference/frontier is foreign")
            committed = {name: row[name] for name in _FRONTIER_FIELDS}
            pending = None
            ack_count += 1
        else:
            raise RuntimeError(f"durable WAL line {line_number} has an unknown kind")
        offset += len(raw)
    return {
        "schema_version": 1,
        "kind": FORENSIC_KIND,
        "status": FORENSIC_STATUS,
        "diagnostic_unauthorized": True,
        "rank": expected_rank,
        "segment_id": expected_segment_id,
        "complete_prefix_bytes": len(prefix),
        "ignored_eof_tail_bytes": len(tail),
        "durable_epoch_ack_count": ack_count,
        "pending_without_epoch_ack": pending is not None,
        "committed_frontier": committed,
    }
