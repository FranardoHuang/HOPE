"""Dependency-light cumulative-scoreboard schema regressions."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "scoreboard_eval.py"
SPEC = importlib.util.spec_from_file_location("scoreboard_eval_tested", MODULE)
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
assert SPEC.loader is not None
SPEC.loader.exec_module(S)


def test_new_or_empty_scoreboard_gets_exactly_one_current_header(tmp_path: Path):
    for name, precreate in (("new.csv", False), ("empty.csv", True)):
        board = tmp_path / name
        if precreate:
            board.touch()
        S.append_scoreboard_rows(board, [{"label": "candidate", "protocol": "implicit"}])
        with board.open(newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == S.CSV_COLUMNS
        assert len(rows) == 2
        assert len(rows[1]) == len(S.CSV_COLUMNS)


def test_matching_header_appends_without_a_second_header(tmp_path: Path):
    board = tmp_path / "scoreboard.csv"
    S.append_scoreboard_rows(board, [{"label": "first"}])
    S.append_scoreboard_rows(board, [{"label": "second"}])
    with board.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows.count(S.CSV_COLUMNS) == 1
    assert len(rows) == 3


def test_old_header_fails_closed_without_mutating_bytes(tmp_path: Path):
    board = tmp_path / "scoreboard.csv"
    before = b"timestamp,label,protocol\nold,row,implicit\n"
    board.write_bytes(before)
    with pytest.raises(S.ScoreboardSchemaError, match="fresh --out-root"):
        S.append_scoreboard_rows(board, [{"label": "must_not_append"}])
    assert board.read_bytes() == before
