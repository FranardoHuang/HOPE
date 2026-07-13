#!/usr/bin/env python3
"""Fixed-protocol sim2sim scoreboard for exported HOPE ping-pong policies (G06 acceptance).

Runs `mujoco_eval_onnx.py` under three FIXED, fully deterministic protocols and appends one
labeled row per protocol to a cumulative scoreboard CSV, so any two checkpoints are compared
on identical target sequences (same seed => identical per-swing targets, clips, and dither):

  implicit        --pd-mode implicit            Isaac total-PD clip when schema-3 effort is bound;
                                                unbound legacy passive-kd is diagnostic only
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
    # explicit protocols carry --keep-passive: the AGI deploy sim applies explicit-Euler PD AND does
    # not zero the MJCF passive damping (S2R_KNOWHOW/SIM_FIDELITY_NOTE) — without it the "explicit"
    # gate is softer than the real validation sim.
    "explicit": ["--pd-mode", "explicit", "--keep-passive"],
    "deploy_faithful": ["--pd-mode", "explicit", "--keep-passive", "--deploy-faithful"],
    # Structure-only variant: the runner episode protocol (stand start, hold, rest, no teleports)
    # under the bound total-effort clip(P-D) actuator — separates "can it survive stand-entry
    # multi-swing" (P2.1) from "can it survive the deploy explicit-PD path" (actuator robustness /
    # explicitpd_ft).
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
    # protocol-provenance columns (appended so positional consumers of the older columns are
    # unaffected): which regime actually graded this row. Without them a clamped and an
    # unclamped row, or a stand-hold and a windup-hold row, are indistinguishable in the table.
    "evaluation_contract_exact", "ready_state_mode", "qdes_clamp",
    "hold_ref", "hold_ref_source", "strike_phase_source",
    "self_contact_step_frac", "implicit_effort_peak_ratio",
]


class ScoreboardSchemaError(RuntimeError):
    """Refuse an append that would make the cumulative CSV structurally ambiguous."""


def validate_scoreboard_header(board: pathlib.Path) -> bool:
    """Return whether a header must be created; reject every nonempty mismatched era.

    A CSV has one header.  Appending wider rows below an old header does not create a new "era";
    it corrupts the positional schema and hides the new provenance fields from ``DictReader``.
    Call this once before expensive protocols and again on the append file descriptor.
    """

    if not board.is_file() or board.stat().st_size == 0:
        return True
    with board.open(newline="") as stream:
        existing_header = next(csv.reader(stream), [])
    if existing_header != CSV_COLUMNS:
        raise ScoreboardSchemaError(
            f"{board} header has {len(existing_header)} columns, expected "
            f"{len(CSV_COLUMNS)}; use a fresh --out-root or explicitly migrate the old board"
        )
    return False


def append_scoreboard_rows(board: pathlib.Path, rows) -> None:
    """Append only beneath the exact current header, rechecking inside the write transaction."""

    board.parent.mkdir(parents=True, exist_ok=True)
    with board.open("a+", newline="") as stream:
        stream.seek(0, 2)
        new_file = stream.tell() == 0
        if not new_file:
            stream.seek(0)
            existing_header = next(csv.reader(stream), [])
            if existing_header != CSV_COLUMNS:
                raise ScoreboardSchemaError(
                    f"{board} header has {len(existing_header)} columns, expected "
                    f"{len(CSV_COLUMNS)}; use a fresh --out-root or explicitly migrate the old board"
                )
        stream.seek(0, 2)
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    if args.qdes_clamp:
        cmd += ["--qdes-clamp"]
    if args.hold_ref:
        cmd += ["--hold-ref", args.hold_ref]
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

    # Protocol provenance from the child's summary JSON: which regime actually graded this row.
    summary_path = out_dir / "mujoco_sim2sim_summary.json"
    if summary_path.is_file():
        try:
            child = json.loads(summary_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"[scoreboard] WARNING: unreadable {summary_path}: {exc}", flush=True)
        else:
            contract = child.get("execution_contract") or {}
            proto_sem = contract.get("protocol_semantics") or {}
            results = child.get("results") or []
            effort = child.get("implicit_effort_limit_diagnostics") or {}
            # Worst mode wins: a child run can carry several noise modes; surfacing only
            # results[0] would report "clean" while another mode self-collided.
            self_contact_fracs = [
                r["self_contact_step_frac"] for r in results
                if isinstance(r.get("self_contact_step_frac"), (int, float))
            ]
            metrics.update(
                evaluation_contract_exact=child.get("evaluation_contract_exact"),
                ready_state_mode=child.get("ready_state_mode"),
                qdes_clamp=contract.get("qdes_clamp"),
                hold_ref=proto_sem.get("hold_ref"),
                hold_ref_source=proto_sem.get("hold_ref_source"),
                strike_phase_source=proto_sem.get("strike_phase_source"),
                self_contact_step_frac=(max(self_contact_fracs) if self_contact_fracs else None),
                # Empty when the guard never armed (no bound schema-3 plant): 0.0 there would
                # read as "measured clean" when nothing was measured.
                implicit_effort_peak_ratio=(
                    effort.get("peak_abs_torque_over_limit")
                    if effort.get("limits_bound") else None
                ),
            )
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
    p.add_argument("--qdes-clamp", action="store_true",
                   help="forward --qdes-clamp to every protocol leg (training clamps by default "
                        "since 2026-07-06 and the C++ runner always has; default off keeps rows "
                        "byte-identical to booked legacy scoreboard scores)")
    p.add_argument("--hold-ref", choices=["auto", "clip", "stand"], default=None,
                   help="forward --hold-ref to every protocol leg (pre-2026-07-11 exports lack "
                        "motion_hold_reference metadata and silently fall back to the legacy "
                        "windup-crouch hold; default None = omit the flag, current behavior)")
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
    board = args.out_root / "scoreboard.csv"
    try:
        validate_scoreboard_header(board)
    except ScoreboardSchemaError as exc:
        raise SystemExit(f"[scoreboard] refusing schema-mismatched append: {exc}") from exc

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

    try:
        append_scoreboard_rows(
            board,
            (
                {**meta, "protocol": protocol, **metrics}
                for protocol, metrics in summary["protocols"].items()
            ),
        )
    except ScoreboardSchemaError as exc:
        raise SystemExit(f"[scoreboard] refusing schema-mismatched append: {exc}") from exc

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
