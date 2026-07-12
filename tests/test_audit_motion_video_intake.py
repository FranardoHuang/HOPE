from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_motion_video_intake.py"
SPEC = importlib.util.spec_from_file_location("audit_motion_video_intake", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INTAKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INTAKE)


def _asset(path: str = "Franco/example.mp4") -> dict:
    return {
        "id": "example",
        "source_relpath": path,
        "sha256": "a" * 64,
        "bytes": 3,
        "collection": "franco",
        "side": "forehand",
        "stroke": "block",
        "action_slot": "forehand_block",
        "candidate_group": None,
        "candidate_rank": None,
        "media": {
            "codec": "hevc",
            "width": 1920,
            "height": 1080,
            "pixel_format": "yuv420p",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30/1",
            "frames": 60,
            "duration_s": 2.0,
        },
    }


def _manifest(asset: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "intake_id": "test",
        "source_root_hint": "${HOME}/Downloads",
        "processing_order": [(asset or _asset())["id"]],
        "assets": [asset or _asset()],
    }


def test_manifest_rejects_path_escape():
    with pytest.raises(INTAKE.IntakeError, match="safe relative"):
        INTAKE.validate_manifest(_manifest(_asset("../secret.mp4")))


def test_manifest_rejects_action_slot_mismatch():
    asset = _asset()
    asset["action_slot"] = "backhand_block"
    with pytest.raises(INTAKE.IntakeError, match="action_slot"):
        INTAKE.validate_manifest(_manifest(asset))


def test_schema2_accepts_high_press_and_locomotion_teacher():
    high_press = _asset("static/pai.mp4")
    high_press.update(
        id="backhand_high_press",
        sha256="b" * 64,
        side="backhand",
        stroke="high_press",
        action_slot="backhand_high_press",
        role="stroke",
        movement_direction=None,
    )
    locomotion = _asset("motion/left_step.mp4")
    locomotion.update(
        id="lateral_step_left",
        sha256="c" * 64,
        side=None,
        stroke=None,
        action_slot="lateral_step_teacher",
        role="lateral_locomotion_teacher",
        movement_direction="left",
    )
    manifest = _manifest(high_press)
    manifest["schema_version"] = 2
    manifest["assets"].append(locomotion)
    manifest["processing_order"].append("lateral_step_left")
    INTAKE.validate_manifest(manifest)


def test_schema2_role_semantics_fail_closed():
    asset = _asset("motion/left_step.mp4")
    asset.update(
        role="lateral_locomotion_teacher",
        movement_direction="left",
        side=None,
        stroke=None,
        action_slot="forehand_block",
    )
    manifest = _manifest(asset)
    manifest["schema_version"] = 2
    with pytest.raises(INTAKE.IntakeError, match="action_slot"):
        INTAKE.validate_manifest(manifest)

    asset["action_slot"] = "lateral_step_teacher"
    asset["movement_direction"] = None
    with pytest.raises(INTAKE.IntakeError, match="movement_direction"):
        INTAKE.validate_manifest(manifest)


def test_manifest_load_rejects_duplicate_keys_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(INTAKE.IntakeError, match="duplicate JSON key"):
        INTAKE.load_manifest(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}')
    with pytest.raises(INTAKE.IntakeError, match="non-finite JSON"):
        INTAKE.load_manifest(nonfinite)

    overflow = tmp_path / "overflow.json"
    overflow.write_text('{"schema_version":1e999}')
    with pytest.raises(INTAKE.IntakeError, match="non-finite JSON number"):
        INTAKE.load_manifest(overflow)


def test_manifest_rejects_non_integer_schema_version():
    manifest = _manifest()
    manifest["schema_version"] = True
    with pytest.raises(INTAKE.IntakeError, match="unsupported schema_version"):
        INTAKE.validate_manifest(manifest)

    manifest["schema_version"] = []
    with pytest.raises(INTAKE.IntakeError, match="unsupported schema_version"):
        INTAKE.validate_manifest(manifest)


def test_manifest_rejects_unsafe_asset_id_and_output_stem():
    asset = _asset()
    asset["id"] = "../escape"
    manifest = _manifest(asset)
    manifest["processing_order"] = ["../escape"]
    with pytest.raises(INTAKE.IntakeError, match="safe binding"):
        INTAKE.validate_manifest(manifest)

    asset = _asset("Franco/..mp4")
    with pytest.raises(INTAKE.IntakeError, match="filename stem"):
        INTAKE.validate_manifest(_manifest(asset))

    asset = _asset("Franco/example.mov")
    with pytest.raises(INTAKE.IntakeError, match=".mp4 extension"):
        INTAKE.validate_manifest(_manifest(asset))


def test_candidate_ranks_must_be_contiguous():
    first = _asset("Franco/a.mp4")
    first.update(id="a", sha256="a" * 64, candidate_group="g", candidate_rank=1)
    second = _asset("Franco/b.mp4")
    second.update(id="b", sha256="b" * 64, candidate_group="g", candidate_rank=3)
    manifest = _manifest(first)
    manifest["assets"].append(second)
    manifest["processing_order"].append("b")
    with pytest.raises(INTAKE.IntakeError, match="contiguous"):
        INTAKE.validate_manifest(manifest)


def test_resolve_asset_path_stays_under_root(tmp_path):
    nested = INTAKE.resolve_asset_path(tmp_path, "Franco/example.mp4")
    assert nested == (tmp_path / "Franco" / "example.mp4").resolve()
    with pytest.raises(INTAKE.IntakeError, match="escapes"):
        INTAKE.resolve_asset_path(tmp_path, "../outside.mp4")


def test_media_comparison_is_strict_and_duration_tolerant():
    expected = _asset()["media"]
    actual = {
        "codec_name": "hevc",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "nb_frames": "60",
        "duration": "2.0000004",
    }
    assert INTAKE.compare_media("example", expected, actual) == []
    actual["nb_frames"] = "59"
    assert "media.frames" in INTAKE.compare_media("example", expected, actual)[0]


def test_audit_checks_hash_before_media(tmp_path, monkeypatch):
    source = tmp_path / "Franco" / "example.mp4"
    source.parent.mkdir()
    source.write_bytes(b"abc")
    asset = _asset()
    asset["sha256"] = INTAKE.sha256_file(source)
    manifest = _manifest(asset)
    INTAKE.validate_manifest(manifest)

    monkeypatch.setattr(INTAKE, "probe_video", lambda *_args, **_kwargs: {
        "codec_name": "hevc",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "nb_frames": "60",
        "duration": "2.0",
    })
    assert INTAKE.audit_assets(manifest, tmp_path) == []

    source.write_bytes(b"abd")
    failures = INTAKE.audit_assets(manifest, tmp_path)
    assert len(failures) == 1
    assert "SHA-256" in failures[0]
