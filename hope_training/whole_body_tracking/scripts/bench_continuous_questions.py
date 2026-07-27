#!/usr/bin/env python3
"""Cost of the CONTINUOUS question path, per reset batch.

人话:训练每次 resample 要现场解一批球。这个脚本量它到底多贵——一次调用多少毫秒、每个 env 多少
微秒、解不出来的比例和原因分布。规矩:解不出来只能重画,绝不许拿别的目标顶上,所以拒绝率是要看的
数字,不是可以忽略的噪声。

Run (pod, CPU or CUDA):
    python scripts/bench_continuous_questions.py --n 4096 --iters 12 --repeat 5
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time
import types

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
MDP = (REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
       / "whole_body_tracking" / "tasks" / "tracking" / "mdp")
PKG = "whole_body_tracking.tasks.tracking.mdp"


def _load(name):
    dotted = f"{PKG}.{name}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, str(MDP / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4096, help="envs resampled in one batch")
    ap.add_argument("--iters", type=int, default=12, help="LM iterations")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol-m", type=float, default=0.02)
    args = ap.parse_args(argv)

    for p in ("whole_body_tracking", "whole_body_tracking.tasks",
              "whole_body_tracking.tasks.tracking", PKG):
        sys.modules.setdefault(p, types.ModuleType(p))
    sys.modules[PKG].__path__ = [str(MDP)]
    vb = _load("virtual_ball")
    _load("strike_spec_torch")
    _load("stroke_prototypes_torch")
    _load("stroke_adapt_torch")
    cq = _load("continuous_questions")

    prm = vb.load_venue_params(str(REPO / "configs" / "ball_physics_venue.yaml"))
    dev = torch.device(args.device)
    n = int(args.n)
    # generate() is ENV-LOCAL by construction (no origins argument): the caller adds the env
    # origin at install time, exactly like the bank seam does.
    clips = torch.zeros(n, dtype=torch.long, device=dev)
    cfg = cq.ContinuousQuestionCfg(n_iters=int(args.iters), tol_m=float(args.tol_m))

    # The trainer's own scorer grades against surface + ball radius (hope_commands.py:4493) and
    # the bank generator builds against the same plane; a bare 0.76 solves against a plane 20 mm
    # off the one training scores on.
    surf = 0.76 + float(prm.ball_radius)
    net_top = 0.76 + 0.1525 + float(prm.ball_radius)
    out = cq.generate(clips, prm, surface_z=surf, net_x=1.87, cfg=cfg,
                      net_top_z=net_top)                                   # warm up
    times = []
    for _ in range(int(args.repeat)):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = cq.generate(clips, prm, surface_z=surf, net_x=1.87, cfg=cfg, net_top_z=net_top)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    med = times[len(times) // 2]
    solved = float(out.ok.float().mean())
    print(f"device={args.device} n={n} iters={args.iters} tol={args.tol_m} "
          f"rounds<= {cfg.max_redraw_rounds}")
    print(f"  wall per call: median {med:.1f} ms  (min {times[0]:.1f}, max {times[-1]:.1f})")
    print(f"  per env:       {med * 1000.0 / n:.2f} us")
    print(f"  solved:        {solved * 100:.2f}%   exhausted rows: {out.exhausted}")
    print(f"  rounds used:   {out.rounds_used}")
    print(f"  reject reasons: {out.reason_counts or '{}'}")
    if out.ok.any():
        print(f"  resid p50/p95: {float(out.resid_m[out.ok].median()) * 1000:.1f} / "
              f"{float(out.resid_m[out.ok].quantile(0.95)) * 1000:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
