#!/usr/bin/env python3
"""Render one current FullMDP MuJoCo policy trace from its exact qpos tape.

The input is the fresh artifact root emitted by the current controller-trace
mode.  This is a visual developer diagnostic, not a training, success,
promotion, deployment, or hardware gate.  It deliberately supports only the
current v2 trace contract and adds no compatibility path for older traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


TRACE_KIND = "action_ball_mujoco_full_mdp_controller_trace_v2"
WIDTH = 960
HEIGHT = 720


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_no_clobber(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def _load_trace(root: Path) -> tuple[dict, np.ndarray, Path]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("trace root differs")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("kind") != TRACE_KIND
        or summary.get("schema_version") != 2
        or summary.get("diagnostic_unauthorized") is not True
        or summary.get("checkpoint_authority") is not False
    ):
        raise ValueError("controller trace contract differs")
    trace_path = root / summary.get("trace_npz", "")
    runtime = summary.get("runtime_mjb")
    if (
        not isinstance(runtime, dict)
        or runtime.get("relative_locator") != "runtime.mjb"
        or not isinstance(runtime.get("sha256"), str)
        or type(runtime.get("size_bytes")) is not int
    ):
        raise ValueError("controller trace runtime MJB contract differs")
    mjb_path = root / runtime["relative_locator"]
    if (
        not trace_path.is_file()
        or _sha256(trace_path) != summary.get("trace_npz_sha256")
        or not mjb_path.is_file()
        or mjb_path.stat().st_size != runtime["size_bytes"]
        or _sha256(mjb_path) != runtime["sha256"]
    ):
        raise ValueError("controller trace bytes differ")
    with np.load(trace_path, allow_pickle=False) as archive:
        if "qpos_world0" not in archive.files:
            raise ValueError("controller trace qpos is unavailable")
        qpos = np.asarray(archive["qpos_world0"], dtype=np.float64)
    expected = (int(summary.get("policy_steps", -1)) + 1,)
    if (
        qpos.ndim != 2
        or qpos.shape[:1] != expected
        or list(qpos.shape) != summary.get("qpos_world0_shape")
        or not np.isfinite(qpos).all()
    ):
        raise ValueError("controller trace qpos contract differs")
    return summary, qpos, mjb_path


def _write_video(path: Path, frames, *, fps: int) -> int:
    process = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(fps), "-i", "pipe:0", "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    count = 0
    assert process.stdin is not None
    try:
        for frame in frames:
            image = np.asarray(frame, dtype=np.uint8)
            if image.shape != (HEIGHT, WIDTH, 3):
                raise RuntimeError("rendered frame shape differs")
            process.stdin.write(image.tobytes())
            count += 1
    finally:
        process.stdin.close()
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    if args.out.exists() or args.fps <= 0:
        raise SystemExit("output must be fresh and fps positive")
    summary, qpos, mjb_path = _load_trace(args.trace_root)
    args.out.mkdir(parents=True, exist_ok=False)

    import mujoco
    from PIL import Image, ImageDraw

    model = mujoco.MjModel.from_binary_path(str(mjb_path))
    if int(model.nq) != qpos.shape[1]:
        raise RuntimeError("render model qpos width differs")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145.0
    camera.elevation = -17.0
    camera.distance = 4.0
    camera.lookat[:] = (1.35, -0.05, 0.9)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    def render(index: int) -> np.ndarray:
        data.qpos[:] = qpos[index]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 330, 48), fill=(0, 0, 0))
        draw.text((18, 18), f"policy step {index}/{len(qpos)-1}", fill=(255, 255, 255))
        return np.asarray(image)

    stills = {}
    for label, index in (
        ("start", 0),
        ("middle", (len(qpos) - 1) // 2),
        ("final", len(qpos) - 1),
    ):
        path = args.out / f"{label}.png"
        Image.fromarray(render(index)).save(path)
        stills[label] = {
            "policy_step": index, "path": path.name, "sha256": _sha256(path)
        }
    video_path = args.out / "policy_fixed_camera.mp4"
    frame_count = _write_video(
        video_path, (render(index) for index in range(len(qpos))), fps=args.fps
    )
    renderer.close()
    receipt = {
        "kind": "action_ball_mujoco_full_mdp_policy_visual_v1",
        "diagnostic_unauthorized": True,
        "trace_summary_sha256": _sha256(
            args.trace_root.expanduser().resolve(strict=True) / "summary.json"
        ),
        "checkpoint": {
            "path": summary["checkpoint_path"],
            "sha256": summary["checkpoint_sha256"],
        },
        "runtime_mjb": summary["runtime_mjb"],
        "fixed_camera": {
            "azimuth": camera.azimuth,
            "elevation": camera.elevation,
            "distance": camera.distance,
            "lookat": camera.lookat.tolist(),
        },
        "stills": stills,
        "video": {
            "path": video_path.name,
            "fps": args.fps,
            "frames": frame_count,
            "sha256": _sha256(video_path),
        },
        "non_claims": [
            "not training or policy success evidence by itself",
            "not promotion, export, deployment, or hardware evidence",
        ],
    }
    _write_json_no_clobber(args.out / "visual_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
