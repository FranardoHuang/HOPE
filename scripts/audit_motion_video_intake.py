#!/usr/bin/env python3
"""Fail-loud audit for a content-addressed raw motion-video intake manifest.

The videos stay outside git.  The tracked manifest records their bytes, media
contract, semantic action slot, and candidate grouping so a local machine or
private processing pod can prove that it received the same recordings before
running GVHMR/GMR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "motion_video_intake_20260711.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
OUTPUT_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SIDES = {"forehand", "backhand"}
STROKES_V1 = {"block", "loop"}
STROKES_V2 = STROKES_V1 | {"high_press"}
ROLES_V2 = {"stroke", "lateral_locomotion_teacher"}
MOVEMENT_DIRECTIONS = {"left", "right"}
MEDIA_FIELDS = {
    "codec": "codec_name",
    "width": "width",
    "height": "height",
    "pixel_format": "pix_fmt",
    "color_space": "color_space",
    "color_transfer": "color_transfer",
    "color_primaries": "color_primaries",
    "r_frame_rate": "r_frame_rate",
    "avg_frame_rate": "avg_frame_rate",
    "frames": "nb_frames",
}


class IntakeError(ValueError):
    """A manifest or local-asset contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntakeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> Any:
    raise IntakeError(f"non-finite JSON constant: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise IntakeError(f"non-finite JSON number: {token}")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except (OSError, json.JSONDecodeError, IntakeError) as exc:
        raise IntakeError(f"cannot read manifest {path}: {exc}") from None
    validate_manifest(data)
    return data


def _required(mapping: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise IntakeError(f"{context} missing required keys: {missing}")


def validate_manifest(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise IntakeError("manifest root must be an object")
    _required(
        data,
        {"schema_version", "intake_id", "source_root_hint", "processing_order", "assets"},
        "manifest",
    )
    schema_version = data["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, 2}
    ):
        raise IntakeError(
            f"unsupported schema_version={schema_version!r}; expected 1 or 2"
        )
    assets = data["assets"]
    if not isinstance(assets, list) or not assets:
        raise IntakeError("manifest assets must be a non-empty list")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    candidate_ranks: dict[str, list[int]] = {}
    for index, asset in enumerate(assets):
        context = f"assets[{index}]"
        if not isinstance(asset, dict):
            raise IntakeError(f"{context} must be an object")
        required_asset_keys = {
            "id", "source_relpath", "sha256", "bytes", "collection", "side",
            "stroke", "action_slot", "candidate_group", "candidate_rank", "media",
        }
        if schema_version == 2:
            required_asset_keys |= {"role", "movement_direction"}
        _required(asset, required_asset_keys, context)
        asset_id = asset["id"]
        if not isinstance(asset_id, str) or not ASSET_ID_RE.fullmatch(asset_id):
            raise IntakeError(
                f"{context}.id must match {ASSET_ID_RE.pattern!r} for safe binding/log paths"
            )
        if asset_id in seen_ids:
            raise IntakeError(f"duplicate asset id {asset_id!r}")
        seen_ids.add(asset_id)

        rel = Path(str(asset["source_relpath"]))
        if rel.is_absolute() or ".." in rel.parts:
            raise IntakeError(f"{asset_id}: source_relpath must be a safe relative path")
        # Check the output stem before the extension.  Hidden/dot-only names such
        # as ``..mp4`` have no pathlib suffix, but their real safety failure is
        # that the stem would escape or alias GVHMR/GMR output bindings.
        if not OUTPUT_STEM_RE.fullmatch(rel.stem) or rel.stem in {".", ".."}:
            raise IntakeError(f"{asset_id}: source filename stem is unsafe for GVHMR output")
        if rel.suffix.lower() != ".mp4":
            raise IntakeError(f"{asset_id}: source video must use the .mp4 extension")
        rel_text = rel.as_posix()
        if rel_text in seen_paths:
            raise IntakeError(f"duplicate source_relpath {rel_text!r}")
        seen_paths.add(rel_text)
        expected_hash = asset["sha256"]
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise IntakeError(f"{asset_id}: sha256 must be 64 lowercase hexadecimal characters")
        if expected_hash in seen_hashes:
            raise IntakeError(f"duplicate video sha256 {expected_hash}")
        seen_hashes.add(expected_hash)
        if (
            isinstance(asset["bytes"], bool)
            or not isinstance(asset["bytes"], int)
            or asset["bytes"] <= 0
        ):
            raise IntakeError(f"{asset_id}: bytes must be a positive integer")
        if schema_version == 1:
            if asset["side"] not in SIDES or asset["stroke"] not in STROKES_V1:
                raise IntakeError(
                    f"{asset_id}: side/stroke must be one of "
                    f"{sorted(SIDES)}/{sorted(STROKES_V1)}"
                )
            expected_slot = f"{asset['side']}_{asset['stroke']}"
            if asset["action_slot"] != expected_slot:
                raise IntakeError(
                    f"{asset_id}: action_slot={asset['action_slot']!r}, "
                    f"expected {expected_slot!r}"
                )
        else:
            role = asset["role"]
            if role not in ROLES_V2:
                raise IntakeError(f"{asset_id}: unsupported role {role!r}")
            if role == "stroke":
                if asset["side"] not in SIDES or asset["stroke"] not in STROKES_V2:
                    raise IntakeError(
                        f"{asset_id}: stroke role requires side/stroke in "
                        f"{sorted(SIDES)}/{sorted(STROKES_V2)}"
                    )
                expected_slot = f"{asset['side']}_{asset['stroke']}"
                if asset["action_slot"] != expected_slot:
                    raise IntakeError(
                        f"{asset_id}: action_slot={asset['action_slot']!r}, "
                        f"expected {expected_slot!r}"
                    )
                if asset["movement_direction"] is not None:
                    raise IntakeError(
                        f"{asset_id}: stroke role must have movement_direction=null"
                    )
            else:
                if asset["side"] is not None or asset["stroke"] is not None:
                    raise IntakeError(
                        f"{asset_id}: locomotion teacher must have side/stroke=null"
                    )
                if asset["action_slot"] != "lateral_step_teacher":
                    raise IntakeError(
                        f"{asset_id}: locomotion teacher action_slot must be "
                        "'lateral_step_teacher'"
                    )
                if asset["movement_direction"] not in MOVEMENT_DIRECTIONS:
                    raise IntakeError(
                        f"{asset_id}: locomotion teacher movement_direction must be one of "
                        f"{sorted(MOVEMENT_DIRECTIONS)}"
                    )

        group = asset["candidate_group"]
        rank = asset["candidate_rank"]
        if (group is None) != (rank is None):
            raise IntakeError(f"{asset_id}: candidate_group and candidate_rank must be both set or null")
        if group is not None:
            if (
                not isinstance(group, str)
                or not group
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank <= 0
            ):
                raise IntakeError(f"{asset_id}: invalid candidate group/rank")
            candidate_ranks.setdefault(group, []).append(rank)

        media = asset["media"]
        if not isinstance(media, dict):
            raise IntakeError(f"{asset_id}: media must be an object")
        _required(media, set(MEDIA_FIELDS) | {"duration_s"}, f"{asset_id}.media")
        if (
            isinstance(media["frames"], bool)
            or not isinstance(media["frames"], int)
            or media["frames"] <= 1
        ):
            raise IntakeError(f"{asset_id}: media.frames must be an integer > 1")
        if (
            isinstance(media["duration_s"], bool)
            or not isinstance(media["duration_s"], (int, float))
            or not math.isfinite(float(media["duration_s"]))
            or float(media["duration_s"]) <= 0.0
        ):
            raise IntakeError(f"{asset_id}: media.duration_s must be positive and finite")

    for group, ranks in candidate_ranks.items():
        ordered = sorted(ranks)
        expected = list(range(1, len(ordered) + 1))
        if ordered != expected:
            raise IntakeError(f"candidate group {group!r} ranks must be contiguous {expected}, got {ordered}")

    processing_order = data["processing_order"]
    if (
        not isinstance(processing_order, list)
        or not all(isinstance(item, str) for item in processing_order)
        or len(set(processing_order)) != len(processing_order)
        or set(processing_order) != seen_ids
    ):
        raise IntakeError("processing_order must list every asset id exactly once")


def resolve_source_root(manifest: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    env_root = os.environ.get("HOPE_MOTION_VIDEO_ROOT")
    raw = env_root if env_root else str(manifest["source_root_hint"])
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def resolve_asset_path(source_root: Path, relpath: str) -> Path:
    root = source_root.resolve()
    path = (root / relpath).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise IntakeError(f"asset path escapes source root: {relpath!r}") from None
    return path


def probe_video(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    executable = shutil.which(ffprobe) if os.sep not in ffprobe else ffprobe
    if not executable:
        raise IntakeError(f"ffprobe executable not found: {ffprobe!r}")
    command = [
        executable,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "format=duration,size:stream=codec_name,width,height,pix_fmt,color_space,color_transfer,color_primaries,r_frame_rate,avg_frame_rate,nb_frames",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or f"exit {result.returncode}"
        raise IntakeError(f"ffprobe failed for {path}: {message}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise IntakeError(f"ffprobe returned invalid JSON for {path}: {exc}") from None
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise IntakeError(f"{path}: expected one selected video stream, got {len(streams)}")
    stream = dict(streams[0])
    stream.update(payload.get("format") or {})
    return stream


def compare_media(asset_id: str, expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for manifest_key, probe_key in MEDIA_FIELDS.items():
        wanted = expected[manifest_key]
        got = actual.get(probe_key)
        if manifest_key in {"width", "height", "frames"}:
            try:
                got = int(got)
            except (TypeError, ValueError):
                pass
        if got != wanted:
            failures.append(f"{asset_id}: media.{manifest_key} expected {wanted!r}, got {got!r}")
    try:
        duration = float(actual.get("duration"))
    except (TypeError, ValueError):
        duration = math.nan
    if not math.isclose(duration, float(expected["duration_s"]), rel_tol=0.0, abs_tol=1.0e-6):
        failures.append(
            f"{asset_id}: media.duration_s expected {expected['duration_s']!r}, got {duration!r}"
        )
    return failures


def audit_assets(
    manifest: dict[str, Any], source_root: Path, *, ffprobe: str = "ffprobe", skip_media: bool = False
) -> list[str]:
    failures: list[str] = []
    for asset in manifest["assets"]:
        asset_id = asset["id"]
        path = resolve_asset_path(source_root, asset["source_relpath"])
        if not path.is_file():
            failures.append(f"{asset_id}: missing source video {path}")
            continue
        size = path.stat().st_size
        if size != asset["bytes"]:
            failures.append(f"{asset_id}: byte size expected {asset['bytes']}, got {size}")
            continue
        digest = sha256_file(path)
        if digest != asset["sha256"]:
            failures.append(f"{asset_id}: SHA-256 expected {asset['sha256']}, got {digest}")
            continue
        if not skip_media:
            try:
                actual = probe_video(path, ffprobe=ffprobe)
            except IntakeError as exc:
                failures.append(str(exc))
                continue
            failures.extend(compare_media(asset_id, asset["media"], actual))
        if not any(item.startswith(f"{asset_id}:") for item in failures):
            print(
                f"[OK] {asset_id}: {asset['media']['frames']} frames, "
                f"{asset['media']['duration_s']:.6f} s, sha256={digest[:12]}..."
            )
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root containing manifest source_relpath entries; default: HOPE_MOTION_VIDEO_ROOT or source_root_hint",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Verify paths, bytes, and SHA-256 only; intended for minimal processing hosts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest.resolve())
        source_root = resolve_source_root(manifest, args.source_root)
        failures = audit_assets(
            manifest, source_root, ffprobe=args.ffprobe, skip_media=bool(args.skip_media)
        )
    except IntakeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        print(
            f"[audit_motion_video_intake] FAIL: {len(failures)} violation(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"[audit_motion_video_intake] PASS: {len(manifest['assets'])} videos under {source_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
