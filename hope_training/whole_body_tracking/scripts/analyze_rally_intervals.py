#!/usr/bin/env python3
"""Measure conservative same-player rally intervals from detected racket strikes.

The venue strike detector does not carry an explicit rally id and may miss
contacts.  This tool therefore accepts only three *consecutive detected*
contacts in one take whose paddle identities are A -> B -> A and whose two
legs are both no longer than a fixed bound.  The result is a conservative
timing audit, not a fitted match-play distribution.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXCLUDED_PREFIXES = ("dianqiu",)
QUANTILES = (0.10, 0.50, 0.90)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"quantile probability outside [0, 1]: {probability}")
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_strikes(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("strikes input must be a JSON list")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"strike row {index} must be an object")
        take = raw.get("take")
        paddle = raw.get("paddle")
        contact_time = raw.get("t_c")
        if not isinstance(take, str) or not take:
            raise ValueError(f"strike row {index} has no take")
        if not isinstance(paddle, str) or not paddle:
            raise ValueError(f"strike row {index} has no paddle")
        if isinstance(contact_time, bool) or not isinstance(contact_time, (int, float)):
            raise ValueError(f"strike row {index} has no numeric t_c")
        contact_time = float(contact_time)
        if not math.isfinite(contact_time):
            raise ValueError(f"strike row {index} has non-finite t_c")
        rows.append({"take": take, "paddle": paddle, "t_c": contact_time})
    return rows


def extract_aba(
    rows: Iterable[dict[str, Any]],
    *,
    max_leg_s: float,
    excluded_prefixes: tuple[str, ...] = DEFAULT_EXCLUDED_PREFIXES,
) -> list[dict[str, Any]]:
    if not math.isfinite(max_leg_s) or max_leg_s <= 0.0:
        raise ValueError("max_leg_s must be finite and positive")
    by_take: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if any(row["take"].startswith(prefix) for prefix in excluded_prefixes):
            continue
        by_take[row["take"]].append(row)

    samples: list[dict[str, Any]] = []
    for take, take_rows in sorted(by_take.items()):
        ordered = sorted(take_rows, key=lambda row: row["t_c"])
        for first, opponent, next_same in zip(ordered, ordered[1:], ordered[2:]):
            if first["paddle"] != next_same["paddle"]:
                continue
            if first["paddle"] == opponent["paddle"]:
                continue
            first_leg = opponent["t_c"] - first["t_c"]
            second_leg = next_same["t_c"] - opponent["t_c"]
            if first_leg <= 0.0 or second_leg <= 0.0:
                raise ValueError(f"non-increasing contact times in take {take}")
            if first_leg > max_leg_s or second_leg > max_leg_s:
                continue
            samples.append(
                {
                    "take": take,
                    "paddle": first["paddle"],
                    "t_start_s": first["t_c"],
                    "self_to_opponent_s": first_leg,
                    "opponent_to_self_s": second_leg,
                    "same_player_interval_s": first_leg + second_leg,
                }
            )
    return samples


def summarize(
    source: Path,
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    max_leg_s: float,
    excluded_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("no A-B-A samples passed the conservative filter")

    def quantile_block(field: str) -> dict[str, float]:
        values = [sample[field] for sample in samples]
        return {
            f"q{int(probability * 100):02d}": linear_quantile(values, probability)
            for probability in QUANTILES
        }

    return {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_rows": len(rows),
        "filter": {
            "ordering": "within take by t_c",
            "pattern": "three consecutive detected contacts A->B->A",
            "max_each_leg_s": max_leg_s,
            "excluded_take_prefixes": list(excluded_prefixes),
        },
        "accepted_samples": len(samples),
        "accepted_by_take": dict(sorted(Counter(sample["take"] for sample in samples).items())),
        "same_player_interval_s": quantile_block("same_player_interval_s"),
        "self_to_opponent_s": quantile_block("self_to_opponent_s"),
        "opponent_to_self_s": quantile_block("opponent_to_self_s"),
        "samples": samples,
        "interpretation": (
            "conservative timing audit only; the detector has no rally id and may miss contacts, "
            "and the accepted sample is not a fitted match-play distribution"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strikes_json", type=Path)
    parser.add_argument("--max-leg-s", type=float, default=2.5)
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        dest="excluded_prefixes",
        help="take prefix to exclude; repeatable (default: dianqiu)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit individual accepted triples from JSON output",
    )
    args = parser.parse_args()

    source = args.strikes_json.resolve()
    excluded = tuple(args.excluded_prefixes or DEFAULT_EXCLUDED_PREFIXES)
    rows = load_strikes(source)
    samples = extract_aba(rows, max_leg_s=args.max_leg_s, excluded_prefixes=excluded)
    report = summarize(
        source,
        rows,
        samples,
        max_leg_s=args.max_leg_s,
        excluded_prefixes=excluded,
    )
    if args.summary_only:
        report.pop("samples")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
