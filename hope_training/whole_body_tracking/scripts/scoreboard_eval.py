#!/usr/bin/env python3
"""Fixed-protocol sim2sim scoreboard for exported HOPE ping-pong policies (G06 acceptance).

Runs `mujoco_eval_onnx.py` under three FIXED, fully deterministic protocols and appends one
labeled row per protocol to a cumulative scoreboard CSV, so any two checkpoints are compared
on identical target sequences (same seed => identical per-swing targets, clips, and dither):

  implicit        --pd-mode implicit            Isaac-faithful cross-check (implicitfast + kd damping)
  explicit        --pd-mode explicit            AGI deploy-sim gate ("falls here = falls on the robot")
  deploy_faithful --pd-mode explicit --deploy-faithful
                                                C++ runner episode protocol (stand start, hold,
                                                rest between swings, no teleports, fall-only ends)

Usage (any machine with mujoco + onnxruntime for the child process):
    python scripts/scoreboard_eval.py --onnx <exported/policy.onnx> --label <run_label> \
        --motion-files <fh.npz> <bh.npz> [--steps 2400] [--protocols implicit explicit deploy_faithful]

Outputs under --out-root (default <wbt>/logs/scoreboard/):
    <label>/<protocol>/            raw mujoco_eval CSVs + captured stdout
    <label>/summary.json           all parsed metrics for this label
    scoreboard.csv                 one appended row per (label, protocol) — the comparison table

The eval child resolves strike phases from the ONNX `clip_strike_phases` metadata by default; pass
--strike-phase-per-clip only for legacy exports without the metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

WBT = pathlib.Path(__file__).resolve().parents[1]
EVAL = WBT / "scripts" / "mujoco_eval_onnx.py"

PROTOCOLS = {
    "implicit": ["--pd-mode", "implicit"],
    "explicit": ["--pd-mode", "explicit"],
    "deploy_faithful": ["--pd-mode", "explicit", "--deploy-faithful"],
    # Structure-only variant: the runner episode protocol (stand start, hold, rest, no teleports)
    # under the Isaac-faithful implicit actuator — separates "can it survive stand-entry multi-swing"
    # (P2.1) from "can it survive the explicit clipped PD" (actuator robustness / explicitpd_ft).
    "deploy_faithful_implicit": ["--pd-mode", "implicit", "--deploy-faithful"],
}

# stdout labels worth scraping (Block 1/2 + deploy-faithful Block 4); everything else comes from
# the per-strike CSV. Keys = scoreboard column names, values = the printed row label.
STDOUT_LABELS = {
    "n_episodes": "n_episodes",
    "fell_count": "fell(count)",
    "mean_episode_length": "mean_episode_length",
    "df_swing_starts": "swing_starts(total)",
    "df_swing_completions": "swing_completions(total)",
    "df_completion_rate": "completion_rate(uncond)",
    "df_falls": "falls",
    "df_mean_time_to_fall_s": "mean_time_to_fall(s)",
}

CSV_COLUMNS = [
    "timestamp", "label", "protocol", "onnx_sha12", "git_commit", "seed", "steps",
    "n_strikes", "composite_rate", "pos_rate", "vel_rate", "normal_rate",
    "pos_err_mean", "vel_err_mean", "normal_err_deg_mean",
    "fh_n", "fh_composite_rate", "bh_n", "bh_composite_rate",
    "n_episodes", "fell_count", "mean_episode_length",
    "df_swing_starts", "df_swing_completions", "df_completion_rate", "df_falls",
    "df_mean_time_to_fall_s",
]


def _sha12(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=WBT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _parse_strikes_csv(path: pathlib.Path) -> dict:
    rows = []
    if path.is_file():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)]
    out = {"n_strikes": len(rows)}
    if not rows:
        return out

    def rate(key):
        vals = [float(r[key]) for r in rows]
        return sum(vals) / len(vals)

    def mean(key):
        vals = [float(r[key]) for r in rows]
        return sum(vals) / len(vals)

    out.update(
        composite_rate=round(rate("composite_pass"), 4),
        pos_rate=round(rate("pos_pass"), 4),
        vel_rate=round(rate("vel_pass"), 4),
        normal_rate=round(rate("normal_pass"), 4),
        pos_err_mean=round(mean("pos_err"), 4),
        vel_err_mean=round(mean("vel_err"), 4),
        normal_err_deg_mean=round(mean("normal_err_deg"), 2),
    )
    for clip, prefix in (("forehand", "fh"), ("backhand", "bh")):
        sub = [r for r in rows if r.get("clip_name", "").lower().startswith(clip[:4])]
        out[f"{prefix}_n"] = len(sub)
        if sub:
            out[f"{prefix}_composite_rate"] = round(
                sum(float(r["composite_pass"]) for r in sub) / len(sub), 4)
    return out


def _parse_stdout(text: str) -> dict:
    out = {}
    for col, printed in STDOUT_LABELS.items():
        # summary rows look like: "<label>            <value>" (one right-justified col per mode)
        m = re.search(rf"^\s*{re.escape(printed)}\s+(\S+)\s*$", text, re.MULTILINE)
        if m:
            out[col] = m.group(1)
    return out


def run_protocol(args, protocol: str, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EVAL),
        "--onnx", str(args.onnx),
        "--motion-files", *[str(m) for m in args.motion_files],
        "--out-dir", str(out_dir),
        "--seed", str(args.seed),
        "--steps", str(args.steps),
        "--noise-scales", "0.0",
        *PROTOCOLS[protocol],
    ]
    if args.mjcf:
        cmd += ["--mjcf", str(args.mjcf)]
    if args.strike_phase_per_clip:
        cmd += ["--strike-phase-per-clip", *[str(p) for p in args.strike_phase_per_clip]]
    if args.ee_term_z is not None:
        cmd += ["--ee-term-z", str(args.ee_term_z)]
    for flag, vals in (("--pos-range-per-clip", args._pos_box12), ("--vel-range-per-clip", args._vel_box12)):
        if vals:
            cmd += [flag, *[str(v) for v in vals]]

    print(f"[scoreboard] {protocol}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "stdout.txt").write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"[scoreboard] {protocol} FAILED (rc={proc.returncode}) — see {out_dir}/stdout.txt",
              flush=True)
        return {"error": f"rc={proc.returncode}"}

    metrics = _parse_strikes_csv(out_dir / "mujoco_sim2sim_strikes.csv")
    metrics.update(_parse_stdout(proc.stdout))
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, type=pathlib.Path)
    p.add_argument("--label", required=True, help="short run label, e.g. p21_A_noteleport_it2000")
    p.add_argument("--motion-files", nargs="+", required=True, type=pathlib.Path,
                   help="clips in training order (clip0=forehand, clip1=backhand)")
    p.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    p.add_argument("--steps", type=int, default=2400, help="control steps per protocol (48 s sim)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mjcf", type=pathlib.Path, default=None)
    p.add_argument("--strike-phase-per-clip", nargs="+", type=float, default=None,
                   help="only for legacy ONNX without clip_strike_phases metadata")
    p.add_argument("--ee-term-z", type=float, default=None)
    p.add_argument("--task-yaml", type=pathlib.Path,
                   default=WBT / "cfg" / "task" / "HOPEPingPongDeployParity.yaml",
                   help="task YAML whose racket.pos_range_per_clip / vel_range_per_clip define the "
                        "IN-DISTRIBUTION eval target boxes (the eval script's built-ins are the "
                        "legacy fixed-plane distribution and floor blade-centered checkpoints).")
    p.add_argument("--no-task-boxes", action="store_true",
                   help="do not forward the task YAML target boxes (legacy fixed-plane eval)")
    p.add_argument("--out-root", type=pathlib.Path, default=WBT / "logs" / "scoreboard")
    args = p.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"[scoreboard] onnx not found: {args.onnx}")
    for m in args.motion_files:
        if not m.is_file():
            raise SystemExit(f"[scoreboard] motion file not found: {m}")

    # In-distribution target boxes from the trained task YAML (racket.pos/vel_range_per_clip).
    args._pos_box12, args._vel_box12 = None, None
    if not args.no_task_boxes and args.task_yaml.is_file():
        try:
            import yaml
            racket = (yaml.safe_load(args.task_yaml.read_text()) or {}).get("racket", {})

            def _flat12(block):
                if not block:
                    return None
                out = []
                for clip in ("forehand", "backhand"):
                    axes = block.get(clip)
                    if not axes:
                        return None
                    for ax in ("x", "y", "z"):
                        out += [float(axes[ax][0]), float(axes[ax][1])]
                return out

            args._pos_box12 = _flat12(racket.get("pos_range_per_clip"))
            args._vel_box12 = _flat12(racket.get("vel_range_per_clip"))
            print(f"[scoreboard] task boxes from {args.task_yaml.name}: "
                  f"pos={'ON' if args._pos_box12 else 'absent'} vel={'ON' if args._vel_box12 else 'absent'}")
        except ImportError:
            print("[scoreboard] WARNING: pyyaml unavailable — evaluating with the eval script's "
                  "LEGACY built-in target boxes (out-of-distribution for blade-centered checkpoints)")

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": args.label,
        "onnx_sha12": _sha12(args.onnx),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "steps": args.steps,
    }
    label_dir = args.out_root / args.label
    summary = {"meta": meta, "protocols": {}}

    for protocol in args.protocols:
        summary["protocols"][protocol] = run_protocol(args, protocol, label_dir / protocol)

    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    board = args.out_root / "scoreboard.csv"
    new_file = not board.is_file()
    with open(board, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for protocol, metrics in summary["protocols"].items():
            w.writerow({**meta, "protocol": protocol, **metrics})

    print(f"\n[scoreboard] label={args.label} onnx_sha12={meta['onnx_sha12']}")
    for protocol, m in summary["protocols"].items():
        if "error" in m:
            print(f"  {protocol:16s} ERROR {m['error']}")
            continue
        line = (f"  {protocol:16s} strikes={m.get('n_strikes', 0):4d} "
                f"composite={m.get('composite_rate', 'n/a')} pos={m.get('pos_rate', 'n/a')} "
                f"vel={m.get('vel_rate', 'n/a')} normal={m.get('normal_rate', 'n/a')}")
        if protocol == "deploy_faithful":
            line += (f" | df_completion={m.get('df_completion_rate', 'n/a')} "
                     f"df_falls={m.get('df_falls', 'n/a')}")
        print(line)
    print(f"[scoreboard] appended -> {board}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
