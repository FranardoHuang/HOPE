#!/usr/bin/env python3
"""Replace family-default racket-face signs with measured-hit discovery signs.

This is a one-way authority upgrade for the reviewed 73-action ChingMu catalog.  Discovery reports
may compare the two physical robot faces offline; the production retarget solver then consumes the
resulting signed catalog and is not allowed to reselect a face silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_ACTIONS = 73
EXPECTED_DISCOVERY_SCHEMA = 3
EXPECTED_DISCOVERY_KIND = "chingmu_canonical_racket_full_phase_retarget_v3"
SIGN_SOURCE_PREFIX = "measured-hit-signed-face-discovery-v1"


class ResignError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ResignError(f"refusing to overwrite existing output: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def resign(*, source_catalog: Path, reports_dir: Path, output: Path) -> dict[str, Any]:
    source_sha = _sha256(source_catalog)
    catalog = json.loads(source_catalog.read_text())
    clips = catalog.get("clips")
    if (
        catalog.get("n_clips") != EXPECTED_ACTIONS
        or not isinstance(clips, list)
        or len(clips) != EXPECTED_ACTIONS
    ):
        raise ResignError("source catalog is not the reviewed 73-action set")
    uids = [row.get("uid") for row in clips if isinstance(row, dict)]
    if len(uids) != EXPECTED_ACTIONS or len(set(uids)) != EXPECTED_ACTIONS:
        raise ResignError("source catalog UIDs are invalid or duplicated")

    report_paths = sorted(reports_dir.glob("*.json"))
    if len(report_paths) != EXPECTED_ACTIONS:
        raise ResignError(
            f"expected {EXPECTED_ACTIONS} discovery reports, got {len(report_paths)}"
        )
    reports: dict[str, tuple[dict[str, Any], str]] = {}
    for path in report_paths:
        report = json.loads(path.read_text())
        uid = report.get("action_id")
        if (
            report.get("schema_version") != EXPECTED_DISCOVERY_SCHEMA
            or report.get("kind") != EXPECTED_DISCOVERY_KIND
            or not isinstance(uid, str)
            or path.stem != uid
            or uid in reports
        ):
            raise ResignError(f"invalid or duplicate discovery report: {path}")
        if report.get("sources", {}).get("catalog", {}).get("sha256") != source_sha:
            raise ResignError(f"discovery report catalog SHA mismatch: {uid}")
        gates = report.get("gates")
        required_hit_gates = (
            "hit_position_le_0p05_m",
            "hit_face_le_5_deg",
            "hit_long_axis_le_5_deg",
            "hit_so3_le_5_deg",
            "hit_velocity_direction_observable",
            "hit_velocity_direction_le_15_deg",
            "hit_velocity_relative_le_0p20",
        )
        if not isinstance(gates, dict) or not all(gates.get(key) is True for key in required_hit_gates):
            raise ResignError(f"discovery report lacks an admitted measured hit: {uid}")
        sign = report.get("teacher", {}).get("robot_mount_normal_sign")
        if sign not in (-1, -1.0, 1, 1.0):
            raise ResignError(f"discovery report has invalid robot face sign: {uid}")
        reports[uid] = (report, _sha256(path))
    if set(reports) != set(uids):
        raise ResignError("discovery report UID set differs from the catalog")

    sign_rows = []
    old_counts = {"+1": 0, "-1": 0}
    new_counts = {"+1": 0, "-1": 0}
    flips = []
    for row in clips:
        uid = row["uid"]
        old_sign = row.get("mount_normal_sign")
        if old_sign not in (-1, 1):
            raise ResignError(f"source catalog sign is invalid: {uid}")
        report, report_sha = reports[uid]
        sign = int(report["teacher"]["robot_mount_normal_sign"])
        row["mount_normal_sign"] = sign
        row["mount_normal_sign_source"] = (
            f"{SIGN_SOURCE_PREFIX};report_sha256={report_sha}"
        )
        old_counts[f"{int(old_sign):+d}"] += 1
        new_counts[f"{sign:+d}"] += 1
        if sign != int(old_sign):
            flips.append(uid)
        sign_rows.append(
            {
                "clip_id": row.get("clip_id"),
                "uid": uid,
                "robot_mount_normal_sign": sign,
                "discovery_report_sha256": report_sha,
            }
        )
    sign_map_sha = hashlib.sha256(_canonical_bytes(sign_rows)).hexdigest()
    catalog["sign_override_note"] = (
        "family-default/FK-era signs superseded by per-action measured signed-hit-face "
        "discovery; production retarget must consume this pinned map"
    )
    catalog["sign_authority"] = {
        "schema_version": 1,
        "kind": "chingmu73_measured_hit_robot_face_sign_map_v1",
        "source_catalog_sha256": source_sha,
        "discovery_report_schema_version": EXPECTED_DISCOVERY_SCHEMA,
        "discovery_report_kind": EXPECTED_DISCOVERY_KIND,
        "sign_map_sha256": sign_map_sha,
        "old_counts": old_counts,
        "new_counts": new_counts,
        "flipped_uids": flips,
    }
    payload = _canonical_bytes(catalog)
    _write_no_replace(output, payload)
    return {
        "source_catalog_sha256": source_sha,
        "output_catalog_sha256": hashlib.sha256(payload).hexdigest(),
        "sign_map_sha256": sign_map_sha,
        "actions": len(sign_rows),
        "old_counts": old_counts,
        "new_counts": new_counts,
        "flipped_uids": flips,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = resign(
        source_catalog=args.source_catalog.resolve(),
        reports_dir=args.reports_dir.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
