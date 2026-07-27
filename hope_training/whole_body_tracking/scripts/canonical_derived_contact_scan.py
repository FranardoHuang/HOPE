#!/usr/bin/env python3
"""Derive a contact-candidate frame directly from ONE bound schema-2 source npz.

Plain language
--------------
The five original canonical motions get their strike frame from a 2026-07-10
"v1" record.  A recording made after that date has no v1 record and can never
acquire one, so it needs an honest way to say where its strike is.  This tool
is that way: it reads the exact bytes the marker authority will bind, scans
them with one written-down rule, and writes an evidence file that is closed on
the input SHA-256 and on this script's own SHA-256.

What the evidence is NOT
-----------------------
* It is not observed ball contact.  There is no ball in these recordings.
* It is not a returnability certificate.  No returnability oracle runs here.
* It is not a behavior authorization, and it is not post-retime.  The frame
  index is in the SOURCE timing of the bound npz, before any compiler retime.
* The scanned body is the wrist link ORIGIN, an explicitly declared carrier
  proxy for the blade.  The blade site lives in the MJCF and is not present in
  the npz, so this tool cannot and does not claim blade-site geometry.

Rule ``max_carrier_speed_at_or_above_min_height``
------------------------------------------------
1. Take the body ``--body`` from ``body_names``.
2. Eligible frames are those with ``body_pos_w[:, body, 2] >= min_height_m``.
3. The anchor frame is the eligible frame with the largest
   ``||body_lin_vel_w[frame, body]||``; ties break to the lowest index.
4. The span is the maximal contiguous run of frames containing the anchor in
   which speed is ``>= span_speed_fraction * max_speed`` (eligibility on height
   is NOT re-applied inside the run; the run is a speed plateau, and the file
   records both facts).

Usage::

    python3 canonical_derived_contact_scan.py \
        --source vendor_assets/.../SHADOW_fh_loop_high_yaw152.npz \
        --body right_wrist_yaw_Link \
        --min-height-m 0.88 \
        --span-speed-fraction 0.6 \
        --out vendor_assets/.../fh_loop_high_yaw152_contact_scan_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

RULE_ID = "max_carrier_speed_at_or_above_min_height"
ARTIFACT_SCHEMA_VERSION = 1
PUBLICATION_CLASS = "evidence_only_not_artifact_authorization"
SEMANTIC_KIND = "derived_kinematic_carrier_speed_peak_frame"


class DerivedContactScanError(ValueError):
    """The scan inputs are not exactly what this tool is allowed to read."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_source(
    source: Path,
    *,
    body_name: str,
    min_height_m: float,
    span_speed_fraction: float,
) -> dict[str, Any]:
    """Run the rule and return the raw result block (no provenance yet)."""

    if not 0.0 < span_speed_fraction < 1.0:
        raise DerivedContactScanError("span_speed_fraction must lie in (0, 1)")
    with np.load(source, allow_pickle=False) as payload:
        names = [str(name) for name in payload["body_names"]]
        if body_name not in names:
            raise DerivedContactScanError(
                f"{source.name} has no body named {body_name!r}"
            )
        index = names.index(body_name)
        positions = np.asarray(payload["body_pos_w"], dtype=np.float64)
        velocities = np.asarray(payload["body_lin_vel_w"], dtype=np.float64)
        fps = int(np.asarray(payload["fps"]).reshape(-1)[0])
        position_point = str(np.asarray(payload["body_pos_point"]).item())
        velocity_point = str(np.asarray(payload["body_lin_vel_point"]).item())
    if positions.ndim != 3 or velocities.shape != positions.shape:
        raise DerivedContactScanError("body_pos_w / body_lin_vel_w shapes disagree")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise DerivedContactScanError("source kinematics contain NaN or infinity")

    frames = int(positions.shape[0])
    height = positions[:, index, 2]
    speed = np.linalg.norm(velocities[:, index, :], axis=1)
    eligible = np.flatnonzero(height >= min_height_m)
    if eligible.size == 0:
        raise DerivedContactScanError(
            f"no frame of {source.name} has {body_name} at or above "
            f"{min_height_m} m; the rule cannot select an anchor"
        )
    anchor = int(eligible[int(np.argmax(speed[eligible]))])
    peak = float(speed[anchor])
    floor = span_speed_fraction * peak
    start = anchor
    while start - 1 >= 0 and speed[start - 1] >= floor:
        start -= 1
    end = anchor
    while end + 1 < frames and speed[end + 1] >= floor:
        end += 1
    span_heights = height[start : end + 1]
    return {
        "source_frames": frames,
        "source_fps": fps,
        "body_name": body_name,
        "body_index": index,
        "body_position_point": position_point,
        "body_linear_velocity_point": velocity_point,
        "anchor_frame": anchor,
        "anchor_speed_mps": peak,
        "anchor_height_m": float(height[anchor]),
        "span_inclusive": [start, end],
        "span_speed_floor_mps": float(floor),
        "span_min_height_m": float(span_heights.min()),
        "span_max_height_m": float(span_heights.max()),
        "span_entirely_at_or_above_min_height": bool(
            float(span_heights.min()) >= min_height_m
        ),
        "eligible_frame_count": int(eligible.size),
    }


def build_artifact(
    *,
    repo_root: Path,
    source: Path,
    out: Path,
    body_name: str,
    min_height_m: float,
    span_speed_fraction: float,
    argv: list[str],
) -> dict[str, Any]:
    tool_path = Path(__file__).resolve()
    result = scan_source(
        source,
        body_name=body_name,
        min_height_m=min_height_m,
        span_speed_fraction=span_speed_fraction,
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": out.stem,
        "publication_class": PUBLICATION_CLASS,
        "authorization": {
            "training_authorized": False,
            "behavior_authorized": False,
            "hardware_authorized": False,
            "artifact_promotion_authorized": False,
        },
        "semantic_kind": SEMANTIC_KIND,
        "non_claims": {
            "observed_ball_contact": False,
            "returnability_certified": False,
            "post_retime_window": False,
            "blade_site_geometry": False,
            "v1_authority_lineage": False,
        },
        "notes": (
            "The scanned body is a wrist link origin used as an explicitly "
            "declared blade-carrier proxy.  The blade site is defined in the "
            "MJCF and is absent from this npz, so no blade-site claim is made. "
            "Frame indices are in the SOURCE timing of the bound npz."
        ),
        "derivation": {
            "tool": tool_path.relative_to(repo_root).as_posix(),
            "tool_sha256": sha256_file(tool_path),
            "cli_argv": argv,
            "rule_id": RULE_ID,
            "thresholds": {
                "min_height_m": float(min_height_m),
                "span_speed_fraction": float(span_speed_fraction),
                "min_height_m_open_question": (
                    "OPEN QUESTION 8.1 (owner's call): 0.88 m is carried over "
                    "from configs/stroke_prototypes_v1_20260727.json "
                    "contact_rule.min_site_z_w_m and is not owner-ratified."
                ),
            },
        },
        "input_source": {
            "path": source.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(source),
        },
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--source", required=True)
    parser.add_argument("--body", default="right_wrist_yaw_Link")
    parser.add_argument("--min-height-m", type=float, required=True)
    parser.add_argument("--span-speed-fraction", type=float, default=0.6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(raw_argv)

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[3]
    )
    source = (repo_root / args.source).resolve()
    out = (repo_root / args.out).resolve()
    if not source.is_file():
        raise DerivedContactScanError(f"source does not exist: {source}")
    artifact = build_artifact(
        repo_root=repo_root,
        source=source,
        out=out,
        body_name=args.body,
        min_height_m=args.min_height_m,
        span_speed_fraction=args.span_speed_fraction,
        argv=raw_argv,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["result"], indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
