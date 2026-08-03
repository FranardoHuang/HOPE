#!/usr/bin/env python3
"""Run the exact Pod Isaac hold and consume its A211 frame-0 receipt.

The wrapper creates one deterministic dynamic-ready-shaped probe input outside
the repository, invokes the existing live ``check_table_obstacle_scene.py``
nominal-hold path for exactly ``task_close_ticks``, and delegates publication
to the independent fail-closed consumer.  It never starts PPO.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CONSUMER_FILE = SCRIPT_DIR / "consume_action_ball_a211_frame0_nominal_hold.py"
PROBE_FILE = SCRIPT_DIR / "check_table_obstacle_scene.py"


def _load_consumer():
    spec = importlib.util.spec_from_file_location(
        "_a211_frame0_hold_consumer", CONSUMER_FILE
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot import A211 frame0 hold consumer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_C = _load_consumer()


class RunError(RuntimeError):
    """The exact wrapper preflight or live child failed."""


def _outside_repo(path: Path, root: Path, *, name: str) -> Path:
    absolute = path.expanduser().absolute()
    parent = absolute.parent.resolve(strict=True)
    absolute = parent / absolute.name
    try:
        absolute.relative_to(root)
    except ValueError:
        return absolute
    raise RunError("%s must stay outside the exact clean checkout" % name)


def _write_probe(path: Path, document: dict) -> str:
    payload = _C.canonical_bytes(document) + b"\n"
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    if path.read_bytes() != payload:
        raise RunError("probe input durable readback differs")
    return _C.sha256_file(path)


def _live_command(
    *,
    python: str,
    device: str,
    motion_path: Path,
    probe_path: Path,
    probe_sha256: str,
    live_path: Path,
    screenshot_dir: Path,
    duration_s: float,
) -> list[str]:
    return [
        python,
        str(PROBE_FILE),
        "--task", "HOPE-PingPong-ActionBall-AgibotA3-v0",
        "--num-envs", "1",
        "--device", device,
        "--table-obstacle", "on",
        "--motion-file", str(motion_path),
        "--nominal-hold", str(probe_path),
        "--nominal-hold-sha256", probe_sha256,
        "--nominal-hold-receipt-out", str(live_path),
        "--duration-s", format(duration_s, ".17g"),
        "--screenshot-dir", str(screenshot_dir),
    ]


def run(args: argparse.Namespace) -> dict:
    root = Path(args.repo_root).resolve(strict=True)
    source_commit = _C._commit(args.probe_source_commit, name="probe_source_commit")
    artifact_source_commit = _C._commit(
        args.artifact_source_commit, name="artifact_source_commit"
    )
    _C.verify_exact_clean_source(root, source_commit)
    ancestor = _C._git(
        root,
        ("merge-base", "--is-ancestor", artifact_source_commit, source_commit),
    )
    if ancestor.returncode != 0:
        raise RunError("artifact source commit is not a probe source ancestor")
    frame_path = _C._tracked_input(
        root, args.frame0_artifact_path, args.expected_frame0_artifact_sha256,
        source_commit, name="frame0 artifact",
    )
    template_path = _C._tracked_input(
        root, args.plant_template_path, args.expected_plant_template_sha256,
        source_commit, name="plant template",
    )
    motion_path = _C._tracked_input(
        root, args.motion_path, args.expected_motion_sha256,
        source_commit, name="motion",
    )
    frame, _ = _C._strict_json(frame_path, name="frame0 artifact", newline=True)
    # This tracked legacy authority is pretty-printed rather than canonical JSON.
    # Its exact bytes are already pinned by the caller-provided file SHA and the
    # clean source commit, so require strict finite JSON without rewriting it.
    template, _ = _C._strict_json(
        template_path, name="plant template", newline=None, canonical=False
    )
    committed_artifact = _C._git(
        root,
        (
            "show",
            artifact_source_commit + ":" + _C._relative(
                args.frame0_artifact_path, name="frame0 artifact path"
            ),
        ),
        binary=True,
    )
    if committed_artifact.returncode or committed_artifact.stdout != frame_path.read_bytes():
        raise RunError("artifact source commit does not contain the exact frame0 bytes")

    work_dir = _outside_repo(Path(args.work_dir), root, name="work_dir")
    if work_dir.exists() or work_dir.is_symlink():
        raise RunError("fresh no-clobber work_dir already exists")
    os.mkdir(work_dir, 0o755)
    probe_path = work_dir / "a211_frame0_nominal_hold.probe_input.v1.json"
    live_path = work_dir / "a211_frame0_nominal_hold.live_safety.v1.json"
    screenshot_dir = work_dir / "screenshots"
    probe = _C.derive_probe_input(
        frame0_artifact=frame,
        frame0_file_sha256=_C.sha256_file(frame_path),
        artifact_source_commit=artifact_source_commit,
        plant_template=template,
        plant_template_file_sha256=_C.sha256_file(template_path),
        motion_path=motion_path,
        motion_sha256=args.expected_motion_sha256,
        probe_source_commit=source_commit,
    )
    probe_sha = _write_probe(probe_path, probe)
    duration = frame["task_close_ticks"] * frame["policy_dt_s"]
    command = _live_command(
        python=args.python,
        device=args.device,
        motion_path=motion_path,
        probe_path=probe_path,
        probe_sha256=probe_sha,
        live_path=live_path,
        screenshot_dir=screenshot_dir,
        duration_s=duration,
    )
    completed = subprocess.run(command, cwd=str(root), check=False)
    if completed.returncode != 0:
        raise RunError(
            "live Isaac nominal hold failed with exit code %d; preserved %s"
            % (completed.returncode, work_dir)
        )
    _C.verify_exact_clean_source(root, source_commit)
    consumer_args = argparse.Namespace(
        repo_root=str(root),
        probe_source_commit=source_commit,
        artifact_source_commit=artifact_source_commit,
        frame0_artifact_path=args.frame0_artifact_path,
        expected_frame0_artifact_sha256=args.expected_frame0_artifact_sha256,
        plant_template_path=args.plant_template_path,
        expected_plant_template_sha256=args.expected_plant_template_sha256,
        motion_path=args.motion_path,
        expected_motion_sha256=args.expected_motion_sha256,
        probe_input=str(probe_path),
        live_receipt=str(live_path),
        output=args.output,
    )
    result = _C.consume(consumer_args)
    return {
        **result,
        "work_dir": str(work_dir),
        "live_evidence_preserved": str(live_path),
        "screenshots_preserved": str(screenshot_dir),
        "completed_live_command": command,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--probe-source-commit", required=True)
    parser.add_argument("--artifact-source-commit", required=True)
    parser.add_argument("--frame0-artifact-path", required=True)
    parser.add_argument("--expected-frame0-artifact-sha256", required=True)
    parser.add_argument("--plant-template-path", required=True)
    parser.add_argument("--expected-plant-template-sha256", required=True)
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--python", required=True, help="exact Pod Isaac Python executable")
    parser.add_argument("--work-dir", required=True, help="fresh path outside checkout")
    parser.add_argument("--output", required=True, help="fresh repository-relative final receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except (_C.ReceiptError, RunError, OSError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
