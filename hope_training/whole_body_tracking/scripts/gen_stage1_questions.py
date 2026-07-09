#!/usr/bin/env python3
"""Stage-1 question bank: fixed strike point, varying incoming speed, no spin, fixed landing.

Three-stage curriculum (NOW.md 2026-07-05, yikang): stage 1 teaches "return balls of
different speeds to a FIXED landing point from a FIXED strike point, with the CORRECT
inverse-solved face + racket velocity". This script builds that question bank offline:

  for each sampled incoming ball (speed in range, no spin, arriving at the fixed
  strike point) -> StrikeSpecPlanner.solve() (franco's LM inverse over the venue
  contact+Magnus chain) -> demanded face normal + racket velocity = the ANSWER.

Only answerable questions are kept (LM converged <=5 mm, speed budget respected,
AND the solved return CLEARS THE NET — spec.clears_net, the planner-side post-solve
annotation; net-rejects are counted per clip and recorded in the bank meta as
net_gate/net_rejects); the solvability rate is reported — if it is low, the fixed
strike point is badly chosen (move the point, do not train on unanswerable questions).

The fixed strike point defaults to the clip's HAND-ANNOTATED strike-frame blade
position (cfg/strike_annotations.yaml via analyze_strike_phase conventions), so the
motion prior and the question are automatically consistent.

ANCHOR MODE (--anchor {annotated,train-candidate}, default annotated = current behavior):
``train-candidate`` anchors the strike frame at the FIRST entry of the clip's registry
``train_phase_candidates`` field (the returnability-scan TRAINING-optimal phase, an
independent field a human copied from --phase-scan output; NOT video contact truth —
see the semantics boundary below) instead of the annotated ``phase``. A clip whose
registry entry has no such field refuses with SystemExit (run --phase-scan and copy
the suggestions in first). The anchor mode + chosen phase/frame are recorded in the
bank meta_json (top-level ``anchor`` + per-clip ``anchor_phase``/``anchor_frame``).

Self-test (--check, default on): every kept answer is replayed through the TRAINING-
side physics (tracking/mdp/virtual_ball.py, torch venue port): predict_paddle_contact
+ coarse_landing must land within tolerance of the target — a cross-implementation
closed loop (planner numpy solver vs trainer torch physics). It also ASSERTS
(SystemExit) that every kept answer clears net_top + ball radius in the torch flight,
so the numpy net gate and the torch check cross-validate.

Frames: the tracking env frame has the table near edge at x=vb_table_near_x and the
table CENTER line at y=0; the planner HOPE frame has the near-left corner at the
origin (x in [0, 2.74], y in [-1.525, 0], and z=0 at the TABLE SURFACE — the env
frame has the surface at z=+0.76). Positions convert by a pure translation
(env->hope: x -= near_x, y -= 0.7625, z -= 0.76); vectors are frame-parallel and
need no conversion.

GRIP + RALLY-YAW (0g calibration, registry 2026-07-06; --grip {registry,off}, default registry):
video clips never show the paddle — the registry `grip:` block stores the SOLVED grip rotation
(HUMAN blade = R_wrist @ Rz(alpha_z)Rx(beta_x) @ y_hat, franco's rally-prior inference) and each
clip carries `rally_yaw_deg`, the rally diagonal it was actually filmed on. This generator:
  * applies the grip INSIDE analyze_strike_phase.analyze(grip_rot=...) — the ONE grip
    application point. Bake detection is FAIL-CLOSED on the registry's face_normal_reliable
    field (exactly `false` = raw, string 'true-calibrated...' = baked and NEVER re-rotated,
    anything else refuses loudly) and runs regardless of --grip so a baked clip always reports
    grip_applied=True; a *_cal stem whose registry entry lacks the bake marker also refuses;
  * canonicalizes clip-derived strike VECTORS (blade velocity, face normal) to the registry's
    straight-line convention by rotating -rally_yaw_deg about the VERTICAL axis through the
    contact point (rally_canonicalize(), the ONE rotation path shared by bank mode and
    --phase-scan; the contact point lies on the axis and is invariant, question geometry is
    already straight-line by construction);
  * runs a SINGLE-APPLICATION self-check per clip: the strike-frame face is recomputed straight
    from the raw npz (one grip multiply + one yaw multiply) and must agree with the pipeline
    value to 1e-6 — a doubled (or dropped) rotation refuses to ship;
  * records grip_applied / rally_yaw_applied (+ per-clip grip mode, rally_yaw) as real JSON in
    the bank npz meta_json; stage1_question_bank.load_question_bank ENFORCES both flags at load
    time (ValueError unless grip_applied and rally_yaw_applied are true; allow_legacy=True is
    the explicit escape hatch) so training can never silently consume a raw or double-rotated
    bank.
MOUNT-OFFSET TRUTH (audit round 2): registry-mode FK rotates the mount offset by Rg, which
shifts the blade point by only a CONSTANT |Rg@r - r| ~ 1.6 cm — the mount offset
[0.2102, 0.0321, 0.0320] lies ~6 deg from Rg's rotation axis [0.992, 0.043, 0.119]. The bake's
~13 cm blade shift comes from RE-SOLVING THE WRIST JOINT CHAIN (csv_to_npz_mujoco --grip-rot),
which registry-mode FK does NOT reproduce: face normal + blade velocity are calibrated, but the
contact POSITION can sit up to ~11 cm from the robot-executable baked blade. Hence a raw clip
with a *_cal sibling in the registry is REFUSED (anchors must come from the _cal npz, registry
note 2026-07-06), and a raw clip without one gets a loud WARNING. The offset rotation itself is
kept (ASSUMPTION, franco to confirm) as the analyzer-side convention closest to the bake.

PHASE SCAN MODE (--phase-scan; merged from claude's 0d scratch scanner 2026-07-05 so there
is ONE stage-1 tool): instead of solving questions at the annotated frame only, walk EVERY frame
of the clip and ask "if contact happened here — face pinned to this frame's face, racket velocity
constrained to a cone around this frame's OWN clip velocity (--cone-deg/--cone-mag), contact at
this frame's blade point — what fraction of the sampled incoming balls has at least one legal
return?" (legal = lands on the opponent half past the depth guard AND clears the net; trainer-side
torch venue physics, same closed loop as --check).

SEMANTICS BOUNDARY (yikang 2026-07-05): the annotation registry records VIDEO TRUTH (when the
human touched the ball); this scan produces the TRAINING-OPTIMAL phase (which frame's kinematics
best suit robot returns). The two may legitimately differ (hopex clips are dry swings — only the
latter exists). The scan therefore NEVER writes the registry: it prints suggested
``train_phase_candidates`` for a HUMAN to copy into the yaml as an independent field.

TRAIN/EXAM SPLIT (--split {train,exam,all}, default all = current behavior): membership is a
deterministic hash of each question's incoming velocity (question_split, ~80/20), NOT a seed/order
partition — so a --split train bank and a --split exam bank are disjoint at ANY generation seeds.
Exams are scored only on the exam split (docs/stage_curriculum_v1.md).

STROKE GUARD (--stroke-guard {off,stats,enforce}, default enforce; 快筛层落地,
docs/research/stroke_interface_survey_2026-07-09.md §3.1)。人话:题要求的拍速 v* 越高、这条
clip 触球前的拍面行程 L 越短,触球前需要的最低加速度 a_min = (v*²−v0²)/(2L) 就越高;a_min
超过机器人实证做得到的拍点加速度包络 a_max(--stroke-budget-clips 实测 × --stroke-budget-scale,
v4rg 对 ×1.5 = synthesize_timing 时间律工具一族的同一预算口径,只是量在拍点笛卡尔空间而非
关节空间)= 该题在这条路径上**可证无时间律解**,不该发货。判据是松弛出来的必要条件 ⇒ 只能
sound-reject;**PASS ≠ 可行**(平衡/力矩耦合/接触可行性全被松弛掉,survey §3.1 诚实边界 1)。
  * v* = 该题需求拍速 |demanded_vel|(题库答案,不是 clip 标定速);v0 = 起步拍速(默认 0 =
    ready 静止起步,时间律合成同口径)。
  * L = L_deep:引拍最深帧 → 触球帧的拍面逐帧位移弧长和(行程账本 stroke_ledger_0709 口径,
    survey §3.1 诚实边界 3 钉死的那种;实现直接复用 extend_stroke.deep_frame_and_L,单一实现)。
    触球帧 = 本 bank 的 anchor 帧(annotated / train-candidate,与题目锚一致)。
  * 分母法则(v1 先默认 stats 出统计;2026-07-10 franco 拍板翻 enforce 默认——六对 v2 exam
    卷实测 0 拦,题库需求拍速 ≤2.9 m/s 远低于各 clip 行程上限 4.7-7.9 m/s,今天零代价,
    phase 2/3 出快题时自动启动保护):enforce 真拦(拒发货并计入分母报表,照 net-rejected
    先例);--stroke-guard stats 只统计不拦(摘要行报 "would reject N");off 完全关闭
    (不需要预算 clip)。
  * fail-loud:stats/enforce 下算不出 L(引拍窗退化/相位缺登记)、v* 非有限、预算 clip 缺
    body 数组,一律 SystemExit,绝不静默跳过——静默跳过会让分母悄悄变假。
  * --phase-scan 模式不适用(它不发货题目,没有逐题 v*)。

Run (Mac base env or pod venv; numpy+torch+yaml):
    python scripts/gen_stage1_questions.py \
        --clip forehand:/workspace/shared/motions/hope_forehand_hopex.npz \
        --clip backhand:/workspace/shared/motions/hope_backhand_hopex.npz \
        --n 512 --out cfg/stage1_questions.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WBT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WBT))

sys.path.insert(0, os.path.join(REPO, "hope_ws", "src", "hope_planner"))

TABLE_HALF_W = 0.7625   # env y=0 is the table centre line; hope_y = env_y - TABLE_HALF_W
TABLE_SURFACE_Z = 0.76  # env table surface height; hope_z = env_z - TABLE_SURFACE_Z


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_analyzer():
    return _load_mod("s1_asp", os.path.join(HERE, "analyze_strike_phase.py"))


def rally_canonicalize(vec_w, rally_yaw_deg):
    """THE single rally-yaw rotation path (registry convention, franco 2026-07-06).

    Anchors/questions are designed in the STRAIGHT-hit frame; a clip filmed on the rally
    diagonal ``rally_yaw_deg`` is canonicalized by rotating its world/env-frame strike VECTORS
    (blade velocity, face normal) by -rally_yaw_deg about the VERTICAL axis through the contact
    point. The contact point itself lies on the axis and is INVARIANT — positions are never
    passed through here. Shared by bank mode and --phase-scan; do not add a second rotation
    path. Accepts (3,) or (..., 3).
    """
    psi = np.deg2rad(-float(rally_yaw_deg))
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(vec_w, dtype=np.float64) @ R.T


def _grip_is_baked(clip: str, ann) -> bool:
    """FAIL-CLOSED bake detection (audit round 2). The only trusted marker is the registry's
    ``face_normal_reliable`` field: exactly ``false`` (bool) = raw video proxy, NOT baked; a
    string starting ``'true-calibrated'`` = csv_to_npz_mujoco --grip-rot regeneration (the grip
    already lives in the npz wrist joints). ANY other value — including bool ``true`` — is
    ambiguous about whether the npz carries the grip, and a wrong guess either double-rotates
    or drops the face, so it refuses loudly instead of guessing. No annotation entry at all
    -> not baked (there is no registry claim to consult)."""
    if ann is None:
        return False
    v = ann.get("face_normal_reliable")
    if v is False:
        return False
    if isinstance(v, str) and v.startswith("true-calibrated"):
        return True
    raise SystemExit(
        f"{clip}: face_normal_reliable={v!r} in strike_annotations.yaml is not a recognized "
        f"bake marker (must be exactly `false`, or a string starting 'true-calibrated') — "
        f"cannot decide whether the npz already carries the grip; fix the registry entry "
        f"before generating")


def resolve_grip_and_yaw(asp, name: str, path: str, annotations: dict, grip_block: dict,
                         grip_flag: str):
    """Per-clip grip rotation + rally yaw from the registry (annotation looked up here, same
    key resolution as the analyzer).

    Returns (grip_rot 3x3 | None, grip_desc, rally_yaw_deg, grip_mode) with grip_mode in
    {"registry", "baked", "off"}. grip_rot is None unless the registry rotation must be applied
    NOW (mode "registry"). Bake detection runs BEFORE the --grip off early-return: the bake
    lives in the npz wrist joints, so a baked clip is baked no matter what the flag says and
    the bank meta must record grip_applied=True for it.
    """
    keys = asp._annotation_keys(name, path)
    ann = next((annotations[k] for k in keys if k in annotations), None)
    stem = keys[0]
    ann_key = next((k for k in keys if k in annotations), stem)
    rally = float(ann.get("rally_yaw_deg") or 0.0) if ann else 0.0
    if ann is not None and ann.get("rally_yaw_deg") is None:
        print(f"[{name}] NOTE: registry entry has no rally_yaw_deg — assuming 0 (straight-line clip)")
    baked = _grip_is_baked(ann_key, ann)
    if stem.endswith("_cal") and not baked:
        raise SystemExit(
            f"{name}: clip stem {stem!r} names a grip-calibrated regeneration (*_cal) but its "
            f"registry entry does not carry the bake marker (face_normal_reliable: "
            f"true-calibrated) — filename and registry disagree about whether the grip is in "
            f"the npz; fix strike_annotations.yaml before generating")
    if baked:
        return None, "BAKED into the npz (csv_to_npz_mujoco --grip-rot; NOT re-applied)", rally, "baked"
    if grip_flag == "off":
        return None, "OFF (--grip off: raw wrist+Y proxy face)", rally, "off"
    if not grip_block:
        print(f"[{name}] NOTE: --grip registry but the yaml has no `grip:` block — "
              f"proceeding with the raw wrist+Y proxy face")
        return None, "OFF (no grip block in the registry)", rally, "off"
    if len(grip_block) == 1:
        session, g = next(iter(grip_block.items()))
        pinned = (ann or {}).get("grip_session")
        if pinned is not None and pinned != session:
            raise SystemExit(f"{name}: clip pins grip_session={pinned!r} but the registry's only "
                             f"grip session is {session!r} — registry and annotation disagree; "
                             f"fix strike_annotations.yaml before generating")
    else:
        session = (ann or {}).get("grip_session")
        if session not in grip_block:
            raise SystemExit(f"{name}: multiple grip sessions in the registry {sorted(grip_block)} — "
                             f"add `grip_session:` to the clip's annotation entry to pick one")
        g = grip_block[session]
    # Registry-mode grip on a RAW clip only fixes face + velocity; the contact POSITION moves
    # by the constant ~1.6 cm mount rotation, NOT the bake's ~13 cm wrist-chain re-solve. If a
    # *_cal regeneration exists, anchors MUST come from it (registry note 2026-07-06).
    cal_key = f"{ann_key}_cal"
    if cal_key in annotations:
        raise SystemExit(
            f"{name}: a grip-calibrated regeneration {cal_key!r} exists in the registry — "
            f"anchors/questions must come from the *_cal npz, not the original (the baked wrist "
            f"re-solve shifts blade POSITIONS up to ~13 cm; registry-mode FK reproduces only a "
            f"~1.6 cm constant mount shift). Point --clip at the *_cal motion file, or pass "
            f"--grip off to study the raw proxy face")
    print(f"[{name}] ** WARNING: registry-mode grip on a raw clip (no *_cal regeneration "
          f"registered) — face normal + blade velocity are calibrated, but the contact "
          f"POSITION only moves by the constant ~1.6 cm mount rotation and can sit up to "
          f"~11 cm from the robot-executable BAKED blade **")
    a, b = float(g["alpha_z_deg"]), float(g["beta_x_deg"])
    return (asp.grip_rotation(a, b),
            f"Rz({a:+.1f} deg)Rx({b:+.1f} deg) [registry:{session}]", rally, "registry")


def _print_grip_summary(name: str, grip_desc: str, grip_rot, rally_yaw_deg: float) -> None:
    print(f"[{name}] grip: {grip_desc} | rally_yaw: {rally_yaw_deg:+.1f} deg (clip strike "
          f"vectors rotated {-rally_yaw_deg:+.1f} deg about the vertical axis through the "
          f"contact point; the contact point itself is invariant)")
    if grip_rot is not None:
        print(f"[{name}] ** ASSUMPTION (franco to confirm): the blade POSITION mount offset "
              f"rotates with the grip (contact point + blade velocity use R_wrist@Rg). This "
              f"shifts the blade point by only a CONSTANT |Rg@r - r| ~ 1.6 cm (the mount "
              f"offset lies ~6 deg from Rg's rotation axis [0.992, 0.043, 0.119]) — it does "
              f"NOT reproduce the csv_to_npz_mujoco bake's ~13 cm shift, which comes from "
              f"re-solving the wrist joint chain **")


def _face_self_check(asp, path: str, frame: int, grip_rot, rally_yaw_deg: float,
                     pipeline_normal) -> None:
    """SINGLE-APPLICATION GUARD: recompute one strike frame's face straight from the raw npz
    (wrist quat -> R_wrist [@ Rg] -> +Y column, then exactly ONE -rally_yaw rotation) and
    assert 1e-6 agreement with the analyzer->rally_canonicalize pipeline value. A grip or
    rally_yaw applied twice (or dropped) anywhere in either path shows up here as tens of
    degrees and refuses to ship."""
    d = np.load(path)
    q = d["body_quat_w"].astype(np.float64)[frame, asp.RACKET_BODY]
    R = asp.quat_to_rot(q)
    if grip_rot is not None:
        R = R @ np.asarray(grip_rot, dtype=np.float64)
    n = rally_canonicalize(R[:, asp.NORMAL_AXIS], rally_yaw_deg)
    err = float(np.max(np.abs(n - np.asarray(pipeline_normal, dtype=np.float64))))
    if not (err < 1e-6):  # explicit raise — a bare assert vanishes under python -O
        raise SystemExit(
            f"single-application guard FAILED at frame {frame}: generator-path face {n} vs "
            f"pipeline face {np.asarray(pipeline_normal)} (max abs diff {err:.2e}) — "
            f"grip/rally_yaw applied twice (or not at all) somewhere; refusing to ship a "
            f"double-rotated bank")


def strike_state_from_clip(asp, name: str, path: str, annotations: dict,
                           grip_rot=None, rally_yaw_deg: float = 0.0,
                           anchor: str = "annotated"):
    """Fixed strike point (env frame) + clip reference normal/velocity from the anchor frame.

    ``anchor='annotated'`` (default) uses the registry ``phase`` (video contact truth);
    ``anchor='train-candidate'`` uses the FIRST entry of the registry's independent
    ``train_phase_candidates`` field (returnability-scan training-optimal frame) and refuses
    (SystemExit) if the clip has no such field. The grip is applied INSIDE asp.analyze() (the
    single grip application point — it rotates the face normal AND the blade position/velocity
    mount FK); clip_normal / clip_vel are then rally-canonicalized here via rally_canonicalize
    (the single rally application point for bank mode). clip_normal_raw is the uncalibrated,
    uncanonicalized wrist+Y proxy at the same frame, kept only for the old-vs-new difficulty
    print."""
    info = asp.analyze(name, path, use_blade=True, grip_rot=grip_rot)
    keys = asp._annotation_keys(name, path)
    ann = next((annotations[k] for k in keys if k in annotations), None)
    if ann is None:
        raise SystemExit(f"{name}: no entry in strike_annotations.yaml (keys tried: {keys}) — "
                         "annotate first, do not fall back to the speed peak")
    ann_key = next(k for k in keys if k in annotations)
    asp._apply_annotation(info, ann_key, ann)
    if anchor == "train-candidate":
        # TRAINING-optimal anchor: the registry's independent train_phase_candidates field
        # (human-copied --phase-scan output; NOT video contact truth). FIRST entry = the
        # scan's earliest returnability-optimal frame. Same phase->frame convention as
        # _annotation_frame (round(phase * (T-1))).
        cands = ann.get("train_phase_candidates") or []
        if not cands:
            raise SystemExit(
                f"{name}: --anchor train-candidate but registry entry {ann_key!r} has no "
                f"train_phase_candidates field — run --phase-scan and copy the suggested "
                f"candidates into strike_annotations.yaml (human step) before anchoring here")
        anchor_phase = float(cands[0])
        info["strike"] = max(0, min(int(round(anchor_phase * (info["T"] - 1))), info["T"] - 1))
    else:
        anchor_phase = (float(ann["phase"]) if ann.get("phase") is not None
                        else info["strike"] / max(info["T"] - 1, 1))
    s = info["strike"]
    # WORLD-frame face (frame-mix fix 2026-07-05, franco's catch: normal_root is PELVIS-frame
    # while positions/velocities are world — mid-swing pelvis rotation rotated every "face" by
    # tens of degrees), grip-calibrated in analyze(), rally-canonicalized here.
    clip_normal = rally_canonicalize(info["normal_w"][s], rally_yaw_deg)
    clip_vel = rally_canonicalize(info["clean_v_w"][s], rally_yaw_deg)
    _face_self_check(asp, path, int(s), grip_rot, rally_yaw_deg, clip_normal)
    raw = asp.analyze(name, path, use_blade=True)  # no grip, no yaw: the pre-0g proxy face
    return dict(
        contact_pos_env=info["racket_w"][s].copy(),      # tracking env frame (root at origin)
        clip_normal=clip_normal,
        clip_vel=clip_vel,
        clip_normal_raw=raw["normal_w"][s].copy(),
        strike_frame=int(s),
        n_frames=int(info["T"]),
        anchor=anchor,
        anchor_phase=float(anchor_phase),
        # full per-frame blade path (T,3) for the stroke guard's L_deep (positions are
        # rally-yaw invariant; same racket_w array the contact anchor comes from)
        blade_path=info["racket_w"].copy(),
    )


def sample_incoming(rng, n, speed_lo, speed_hi, vy_max, vz_lo, vz_hi):
    """Incoming ball velocity AT the strike point (env/world axes, toward the robot = -x)."""
    speed = rng.uniform(speed_lo, speed_hi, size=n)
    vy = rng.uniform(-vy_max, vy_max, size=n)
    vz = rng.uniform(vz_lo, vz_hi, size=n)
    vx_sq = np.maximum(speed**2 - vy**2 - vz**2, 0.25)  # keep a real toward-robot component
    vx = -np.sqrt(vx_sq)
    return np.stack([vx, vy, vz], axis=1)


EXAM_FRAC = 0.2  # deterministic hash-split target: ~80% train / ~20% exam


def question_split(v_in_row) -> str:
    """Deterministic train/exam membership of a question, a pure function of its incoming velocity.

    blake2b over the ROUNDED (1e-4 m/s) v_in bytes -> uniform in [0, 1); < EXAM_FRAC = exam.
    Membership depends ONLY on the question content, never on seed/order/solver outcome, so a
    train bank and an exam bank generated at ANY seeds can never share a question — the exam is
    guaranteed disjoint from training.
    """
    import hashlib

    key = np.round(np.asarray(v_in_row, dtype=np.float64), 4).tobytes()
    h = hashlib.blake2b(key, digest_size=8).digest()
    return "exam" if int.from_bytes(h, "big") / 2.0**64 < EXAM_FRAC else "train"


# ---------------------------------------------------------------- stroke guard ---- #
# 行程快筛守卫(survey 2026-07-09 §3.1 落地)— 模块 docstring 的 STROKE GUARD 节有人话版。
_ES_MOD = None


def _stroke_ledger():
    """extend_stroke.py = THE stroke-ledger implementation (deep_frame_and_L, the
    stroke_ledger_0709 L_deep 口径) — loaded lazily; numpy-only at import time."""
    global _ES_MOD
    if _ES_MOD is None:
        _ES_MOD = _load_mod("s1_es", os.path.join(HERE, "extend_stroke.py"))
    return _ES_MOD


def blade_acc_envelope(paths):
    """Per-clip peak blade-point Cartesian |acceleration| [m/s²] on the budget clips.

    Blade convention = synthesize_timing.blade_positions (wrist body + mount FK from the
    stored body arrays); differentiation = np.gradient twice at the clip's own fps (the
    csv_to_npz velocity convention). This is the Cartesian blade-point counterpart of the
    time-law family's joint-space |q̈| envelope — same budget PHILOSOPHY (实证"机器人做
    得到"的包络), matched to a_min's units because a_min is a Cartesian quantity
    (survey §3.1 honest boundary 2: until a dedicated tangential-acceleration calibration
    exists, the guard's reject is only as sound as this envelope convention — recorded in
    the bank meta so the caveat travels with the data). fail-loud on unreadable clips,
    missing body arrays, non-finite positions, or too-few frames.
    """
    if not paths:
        raise SystemExit("stroke-guard: empty budget clip list — nothing to calibrate a_max from")
    es = _stroke_ledger()
    per = {}
    for p in paths:
        try:
            d = dict(np.load(p))
        except Exception as exc:  # noqa: BLE001 — whatever np.load throws, name the file
            raise SystemExit(f"stroke-guard: cannot read budget clip {p!r}: {exc}") from None
        missing = [k for k in ("fps", "body_pos_w", "body_quat_w") if k not in d]
        if missing:
            raise SystemExit(
                f"stroke-guard: budget clip {p!r} lacks {missing} — the stored body arrays "
                f"(FK cache) are required to measure the blade-point acceleration envelope")
        blade = es.st.blade_positions(d)
        if blade.shape[0] < 5:
            raise SystemExit(f"stroke-guard: budget clip {p!r} has only {blade.shape[0]} "
                             f"frames — too short to differentiate twice")
        if not np.isfinite(blade).all():
            raise SystemExit(f"stroke-guard: budget clip {p!r} blade positions contain "
                             f"non-finite values — refusing to calibrate a_max from it")
        dt = 1.0 / float(np.asarray(d["fps"]).reshape(-1)[0])
        acc = np.gradient(np.gradient(blade, dt, axis=0), dt, axis=0)
        per[os.path.basename(p)] = float(np.linalg.norm(acc, axis=1).max())
    return per


def resolve_stroke_guard(mode, budget_clips, scale):
    """(a_max [m/s²], per-clip envelope dict) — or (None, {}) when the guard is off.

    a_max = max over the budget clips × scale, the SAME budget convention as
    synthesize_timing (--budget-clips v4rg pair, --budget-scale 1.5). stats/enforce
    WITHOUT budget clips refuses loudly: 统计模式也要有 a_max 才有数字,静默跳过会让
    "若开拦会拦掉 N 题" 的分母悄悄变假。
    """
    if mode == "off":
        return None, {}
    if not budget_clips:
        raise SystemExit(
            f"--stroke-guard {mode} needs --stroke-budget-clips (e.g. the v4rg pair "
            f"hope_forehand_v4rg_cal.npz hope_backhand_v4rg_cal.npz — the proven-executable "
            f"envelope, same budget convention as the time-law tools); pass "
            f"--stroke-guard off to generate without the guard")
    per = blade_acc_envelope(budget_clips)
    a_max = max(per.values()) * float(scale)
    if not (np.isfinite(a_max) and a_max > 0.0):
        raise SystemExit(f"stroke-guard: computed a_max {a_max!r} is not a positive finite "
                         f"number — the budget clips look degenerate")
    return a_max, per


def stroke_disposition(mode, over_budget: bool) -> str:
    """'ship' | 'flag' | 'reject' — 分母法则: stats 只计数不拦 (v1 default), enforce 才拦."""
    if not over_budget or mode == "off":
        return "ship"
    return "reject" if mode == "enforce" else "flag"


class StrokeGuard:
    """快筛层 (survey §3.1): a_min = (v*²−v0²)/(2·L_deep) 必要条件下界 vs 拍点包络 a_max.

    Sound-REJECT only: a_min > a_max ⇒ provably NO time law reaches v* on this clip path
    (the bound holds for ANY tangential-acceleration profile — 匀加速只是取等的那支).
    a_min ≤ a_max means nothing more than "not disproven": PASS ≠ feasible (balance,
    torque coupling and contact feasibility are all relaxed away).
    """

    def __init__(self, clip: str, L_deep: float, deep_frame: int, contact_frame: int,
                 a_max: float, v0: float = 0.0):
        if not (np.isfinite(L_deep) and L_deep > 1e-6):
            raise SystemExit(
                f"{clip}: stroke-guard cannot price this clip — L_deep={L_deep!r} m from "
                f"deep frame {deep_frame} to contact frame {contact_frame} (引拍行程退化/"
                f"不存在); fix the clip / phase registry, or pass --stroke-guard off")
        if not (np.isfinite(a_max) and a_max > 0.0):
            raise SystemExit(f"{clip}: stroke-guard a_max={a_max!r} is not positive finite")
        self.clip = clip
        self.L_deep = float(L_deep)
        self.deep_frame = int(deep_frame)
        self.contact_frame = int(contact_frame)
        self.a_max = float(a_max)
        self.v0 = float(v0)
        self.over_budget_count = 0

    @classmethod
    def from_blade_path(cls, clip: str, blade_path, contact_frame: int, a_max: float,
                        v0: float = 0.0):
        """Build from a clip's per-frame blade positions ((T,3), the analyzer racket_w
        array; positions are rally-yaw invariant and arc length is rotation-invariant,
        so no canonicalization is needed here). fail-loud on non-finite paths or a
        contact frame with no run-up."""
        blade = np.asarray(blade_path, dtype=np.float64)
        if blade.ndim != 2 or blade.shape[1] != 3:
            raise SystemExit(f"{clip}: stroke-guard expected a (T,3) blade path, "
                             f"got shape {blade.shape}")
        if not np.isfinite(blade).all():
            raise SystemExit(f"{clip}: stroke-guard: blade path contains non-finite values "
                             f"— cannot compute the stroke length L")
        c = int(contact_frame)
        if not (0 < c <= blade.shape[0] - 1):
            raise SystemExit(f"{clip}: stroke-guard: contact frame {c} leaves no pre-contact "
                             f"stroke to measure (T={blade.shape[0]})")
        d, L = _stroke_ledger().deep_frame_and_L(blade, c)
        return cls(clip, L, d, c, a_max, v0)

    @property
    def v_star_cap(self) -> float:
        """Largest demanded speed this clip's stroke can ship under a_max [m/s]."""
        return float(np.sqrt(2.0 * self.a_max * self.L_deep + self.v0 * self.v0))

    def a_min(self, v_star: float) -> float:
        """Necessary lower bound on pre-contact blade tangential acceleration [m/s²]."""
        v = float(v_star)
        if not np.isfinite(v):
            raise SystemExit(f"{self.clip}: stroke-guard: non-finite demanded speed "
                             f"v*={v_star!r} — the answer is corrupt, refusing to price it")
        return (v * v - self.v0 * self.v0) / (2.0 * self.L_deep)

    def check(self, v_star: float):
        """(a_min, over_budget) for one question; counts over-budget hits on the guard."""
        a = self.a_min(v_star)
        over = bool(a > self.a_max)
        if over:
            self.over_budget_count += 1
        return a, over


def phase_scan(asp, name: str, path: str, annotations: dict, grip_block: dict, args) -> None:
    """Per-frame returnability under the frame's own swing-velocity cone (torch batch/frame)."""
    import torch

    vb = _load_mod("s1_vb", os.path.join(
        WBT, "source", "whole_body_tracking", "whole_body_tracking",
        "tasks", "tracking", "mdp", "virtual_ball.py"))
    prm = vb.load_venue_params(os.path.join(REPO, "configs", "ball_physics_venue.yaml"))
    torch.set_default_dtype(torch.float64)

    keys = asp._annotation_keys(name, path)
    ann = next((annotations[k] for k in keys if k in annotations), None)
    grip_rot, grip_desc, rally_yaw, _grip_mode = resolve_grip_and_yaw(
        asp, name, path, annotations, grip_block, args.grip)
    _print_grip_summary(name, grip_desc, grip_rot, rally_yaw)
    info = asp.analyze(name, path, use_blade=True, grip_rot=grip_rot)
    label_frame = None
    if ann is not None:
        asp._apply_annotation(info, next(k for k in keys if k in annotations), ann)
        label_frame = int(info["strike"])
    T = int(info["T"])
    # Rally canonicalization of the whole clip's strike vectors, applied exactly ONCE here
    # (same rotation path as bank mode: rally_canonicalize). Per-frame contact POSITIONS are
    # invariant (the axis passes through each frame's own contact point).
    v_all = rally_canonicalize(info["clean_v_w"], rally_yaw)      # (T,3)
    n_all = rally_canonicalize(info["normal_w"], rally_yaw)       # (T,3)
    _face_self_check(asp, path, int(info["strike"]), grip_rot, rally_yaw,
                     n_all[int(info["strike"])])

    rng = np.random.default_rng(args.seed)
    v_in = sample_incoming(rng, args.scan_balls, *args.speed_range, args.vy_max, *args.vz_range)
    M, K = args.scan_balls, args.cone_vels

    net_x = args.near_x + 1.37
    far_x = args.near_x + 2.74
    half_w = TABLE_HALF_W
    net_top = TABLE_SURFACE_Z + 0.1525
    ball_r = float(prm.ball_radius)
    depth_x = net_x + 0.3

    t64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    scores = np.zeros(T)
    for t in range(T):
        v_clip = np.asarray(v_all[t], float)              # grip-FK'd + rally-canonicalized
        speed = float(np.linalg.norm(v_clip))
        if speed < args.scan_min_speed:
            continue
        v_dir = v_clip / speed
        ang = np.deg2rad(rng.uniform(0.0, args.cone_deg, size=K))
        aux = rng.standard_normal((K, 3))
        aux -= (aux @ v_dir)[:, None] * v_dir[None, :]
        aux /= np.linalg.norm(aux, axis=1, keepdims=True) + 1e-12
        dirs = np.cos(ang)[:, None] * v_dir[None, :] + np.sin(ang)[:, None] * aux
        v_r = dirs * (speed * rng.uniform(*args.cone_mag, size=K))[:, None]

        n_face = np.asarray(n_all[t], float)   # world frame (frame-mix fix), calibrated+canonical
        p_env = np.asarray(info["racket_w"][t], float)
        vv_in = t64(np.repeat(v_in, K, axis=0))               # (M*K,3)
        vv_r = t64(np.tile(v_r, (M, 1)))
        nn = t64(np.tile(n_face, (M * K, 1)))
        pp = t64(np.tile(p_env, (M * K, 1)))
        v_plus, w_plus = vb.predict_paddle_contact(vv_in, vv_r, nn, torch.zeros(M * K, 3), prm)
        land = vb.coarse_landing(pp, v_plus, w_plus, prm,
                                 surface_z=TABLE_SURFACE_Z + ball_r, net_x=net_x,
                                 h=0.01, n_steps=200)
        lx, ly = land["land_xy"][:, 0], land["land_xy"][:, 1]
        legal = (land["land_valid"] & land["net_valid"]
                 & (land["net_z"] > net_top + ball_r)
                 & (lx > depth_x) & (lx <= far_x) & (ly.abs() <= half_w))
        scores[t] = legal.view(M, K).any(dim=1).double().mean().item()

    top = np.argsort(-scores)[:3]
    band = [f"{t/(T-1):.2f}" for t in range(T) if scores[t] > 0.5]
    lbl = (f"registry frame {label_frame} (phase {label_frame/(T-1):.3f}) score "
           f"{scores[label_frame]:.0%}" if label_frame is not None else "no registry entry")
    print(f"[scan:{name}] T={T} | {lbl}")
    print(f"  top frames: " + ", ".join(
        f"f{t} (phase {t/(T-1):.3f}) {scores[t]:.0%}" for t in top))
    band_txt = (", ".join(band) if band else
                "NONE — no frame of this clip can return the sampled balls under its own cone")
    print(f"  >50% band: {band_txt}")
    cands = sorted({int(t) for t in top if scores[t] >= 0.5})
    if cands:
        print(f"  suggested (copy into strike_annotations.yaml by hand, independent field):\n"
              f"    train_phase_candidates: {[round(t/(T-1), 3) for t in cands]}"
              f"   # returnability-optimal, NOT video contact truth")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", required=True,
                    help="name:path.npz (repeatable; name = forehand/backhand)")
    ap.add_argument("--n", type=int, default=512, help="questions per clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed-range", type=float, nargs=2, default=[2.0, 5.0],
                    help="incoming ball speed at contact (m/s)")
    ap.add_argument("--vy-max", type=float, default=0.6)
    ap.add_argument("--vz-range", type=float, nargs=2, default=[-2.0, 0.3])
    ap.add_argument("--landing-env", type=float, nargs=2, default=[2.555, 0.0],
                    help="fixed landing target, env frame (default P2 half centre)")
    ap.add_argument("--near-x", type=float, default=0.5,
                    help="table near edge x in the env frame (cfg vb_table_near_x)")
    ap.add_argument("--speed-budget", type=float, default=4.0,
                    help="max |racket contact-point velocity| passed to the solver (m/s)")
    ap.add_argument("--out", default=os.path.join(WBT, "cfg", "stage1_questions.npz"))
    ap.add_argument("--no-check", action="store_true",
                    help="skip the torch virtual-ball closed-loop self-test")
    ap.add_argument("--grip", choices=("registry", "off"), default="registry",
                    help="apply the registry grip calibration (strike_annotations.yaml `grip:` "
                         "block: HUMAN blade = R_wrist @ Rz(alpha_z)Rx(beta_x) @ y_hat) to the "
                         "clip face normal AND blade position/velocity FK; clips already baked "
                         "by csv_to_npz_mujoco --grip-rot (face_normal_reliable: "
                         "true-calibrated) are detected and never re-rotated. "
                         "'off' = raw wrist+Y proxy (pre-0g behavior). rally_yaw "
                         "canonicalization is independent and always applied")
    ap.add_argument("--anchor", choices=("annotated", "train-candidate"), default="annotated",
                    help="strike-frame anchor: 'annotated' = the registry `phase` (video "
                         "contact truth; current behavior); 'train-candidate' = the FIRST "
                         "entry of the clip's independent `train_phase_candidates` registry "
                         "field (returnability-scan training-optimal frame; refuses with "
                         "SystemExit if the clip has no such field)")
    ap.add_argument("--split", choices=("train", "exam", "all"), default="all",
                    help="keep only this side of the deterministic ~80/20 hash split on the "
                         "incoming-velocity bytes (exam questions are disjoint from training at "
                         "ANY seed); 'all' = no split (current behavior)")
    # --- stroke guard (快筛层, survey 2026-07-09 §3.1 — see the module docstring) ---------
    ap.add_argument("--stroke-guard", choices=("off", "stats", "enforce"), default="enforce",
                    help="行程快筛守卫: a_min=(v*²−v0²)/(2·L_deep) > a_max ⇒ 该题在这条 clip "
                         "路径上可证无时间律解 (必要条件 sound-reject; PASS≠可行). "
                         "'enforce' (default, franco 2026-07-10 拍板; 六卷实测 0 拦) = 真拦 "
                         "(拒发货并计入分母报表, 照 net-rejected 先例); 'stats' = 只统计不拦, "
                         "摘要行报 '若开拦会拦掉 N 题'; 'off' = 完全关闭 (不需要预算 clip)")
    ap.add_argument("--stroke-budget-clips", nargs="+", default=None,
                    help="[stroke-guard] npz clips whose measured blade-point |acc| envelope "
                         "defines a_max (e.g. the v4rg pair — the proven-executable floor; "
                         "SAME budget convention as synthesize_timing --budget-clips, taken "
                         "at the blade point because a_min is Cartesian). REQUIRED unless "
                         "--stroke-guard off")
    ap.add_argument("--stroke-budget-scale", type=float, default=1.5,
                    help="[stroke-guard] a_max = envelope × this (default 1.5, the time-law "
                         "family's --budget-scale 口径)")
    ap.add_argument("--stroke-v0", type=float, default=0.0,
                    help="[stroke-guard] blade speed at stroke start [m/s] (default 0 = ready "
                         "静止起步, 时间律合成同口径)")
    # --- phase-scan mode (per-frame returnability under the clip's own swing-velocity cone) ---
    ap.add_argument("--phase-scan", action="store_true",
                    help="scan EVERY frame's returnability instead of building the bank; prints "
                         "suggested train_phase_candidates (never writes the registry)")
    ap.add_argument("--cone-deg", type=float, default=25.0,
                    help="[scan] max angle between candidate racket velocity and the frame's "
                         "clip velocity")
    ap.add_argument("--cone-mag", type=float, nargs=2, default=[0.6, 1.4],
                    help="[scan] candidate speed as a multiple of the frame's clip speed")
    ap.add_argument("--cone-vels", type=int, default=150, help="[scan] velocity candidates/frame")
    ap.add_argument("--scan-balls", type=int, default=24, help="[scan] incoming balls (fixed set)")
    ap.add_argument("--scan-min-speed", type=float, default=0.3,
                    help="[scan] frames with clip speed below this are unstrikeable (score 0)")
    args = ap.parse_args()

    from hope_planner.strike_spec_planner import StrikeSpecPlanner

    asp = _load_analyzer()
    annotations = asp._load_annotations(asp.DEFAULT_ANNOTATIONS)
    grip_block = asp._load_grip_block(asp.DEFAULT_ANNOTATIONS) if args.grip == "registry" else {}

    if args.phase_scan:
        for spec in args.clip:
            name, _, path = spec.partition(":")
            phase_scan(asp, name, path, annotations, grip_block, args)
        return 0

    # stroke guard a_max (v4rg 实证包络 × scale) — resolved ONCE, shared by every clip;
    # fail-loud here (before any solving) when stats/enforce lacks budget clips.
    stroke_a_max, stroke_env = resolve_stroke_guard(
        args.stroke_guard, args.stroke_budget_clips, args.stroke_budget_scale)
    if stroke_a_max is not None:
        env_txt = ", ".join(f"{k} {v:.2f}" for k, v in stroke_env.items())
        print(f"[stroke-guard] mode={args.stroke_guard}: blade-point |acc| envelope "
              f"{{{env_txt}}} m/s² × {args.stroke_budget_scale:g} -> a_max = "
              f"{stroke_a_max:.2f} m/s² (necessary-condition reject only; PASS != feasible)")

    planner = StrikeSpecPlanner()
    rng = np.random.default_rng(args.seed)

    env2hope = np.array([-args.near_x, -TABLE_HALF_W, -TABLE_SURFACE_Z])
    landing_hope = np.asarray(args.landing_env) + env2hope[:2]

    banks = {}
    clip_meta = {}
    grip_any = False
    for spec in args.clip:
        name, _, path = spec.partition(":")
        grip_rot, grip_desc, rally_yaw, grip_mode = resolve_grip_and_yaw(
            asp, name, path, annotations, grip_block, args.grip)
        _print_grip_summary(name, grip_desc, grip_rot, rally_yaw)
        st = strike_state_from_clip(asp, name, path, annotations, grip_rot, rally_yaw,
                                    anchor=args.anchor)
        guard = None
        if stroke_a_max is not None:
            # L_deep on THIS clip's blade path with THIS bank's anchor as the contact frame
            # (ledger 口径 via extend_stroke.deep_frame_and_L; fail-loud on degenerate strokes)
            guard = StrokeGuard.from_blade_path(name, st["blade_path"], st["strike_frame"],
                                                stroke_a_max, v0=args.stroke_v0)
            print(f"[{name}] stroke-guard: L_deep = {guard.L_deep:.4f} m (deep frame "
                  f"f{guard.deep_frame} -> contact f{guard.contact_frame}) -> shippable "
                  f"demanded speed cap |v*| <= {guard.v_star_cap:.2f} m/s at a_max "
                  f"{guard.a_max:.2f} m/s²")
        p_env = st["contact_pos_env"]
        p_hope = p_env + env2hope
        # Effective clip face for sign-alignment + NEW difficulty; raw wrist+Y proxy (no grip,
        # no yaw) kept only for the OLD difficulty column of the summary print.
        cn = st["clip_normal"] / np.linalg.norm(st["clip_normal"])
        cn_raw = st["clip_normal_raw"] / np.linalg.norm(st["clip_normal_raw"])

        v_in = sample_incoming(rng, args.n, *args.speed_range, args.vy_max, *args.vz_range)
        # Deterministic train/exam membership per question (hash of v_in — see question_split);
        # counted BEFORE solving so both counts and the answerable-fraction denominator are per split.
        splits = np.array([question_split(v) for v in v_in])
        n_train, n_exam = int((splits == "train").sum()), int((splits == "exam").sum())
        sel = np.arange(args.n) if args.split == "all" else np.where(splits == args.split)[0]
        kept, normals, vels, diffs, residuals, landings, in_cone = [], [], [], [], [], [], []
        diffs_old = []
        a_mins = []
        net_rejects = 0
        for i in sel:
            s = planner.solve(p_hope, v_in[i], None, landing_hope, args.speed_budget)
            if s is None or s.residual_m > planner.TOL_M:
                continue
            # NET GATE: the solver ANNOTATES (clears_net computed post-solve, same venue
            # flight model) and the bank DECIDES — a converged answer whose return would
            # clip the net (center at x=net_x below net_height + ball radius) is not a
            # trainable stage-1 answer. Counted separately from unsolvable.
            if not s.clears_net:
                net_rejects += 1
                continue
            # STROKE GUARD (survey §3.1): a_min = (v*²−v0²)/(2·L_deep) on the DEMANDED
            # speed |s.v_r| vs the blade-point acceleration envelope — a necessary
            # condition, so exceeding it PROVES no time law reaches v* on this path.
            # stats mode (v1 default) counts what enforce WOULD drop but ships the
            # question (分母法则: franco 看数字再拍板); enforce really drops it.
            a_min_q = None
            if guard is not None:
                a_min_q, over = guard.check(float(np.linalg.norm(s.v_r)))
                if stroke_disposition(args.stroke_guard, over) == "reject":
                    continue
            kept.append(i)
            # Store the normal SIGN-ALIGNED to the clip's (calibrated) +Y red face: contact
            # physics is sign-agnostic (orient_normal flips internally) but the face-tracking
            # reward is sign-sensitive — the raw solver sign is arbitrary and could demand a
            # 180-deg flip.
            nn = s.n if np.dot(s.n, cn) >= 0 else -s.n
            normals.append(nn)
            vels.append(s.v_r)
            residuals.append(s.residual_m)
            landings.append(s.landing_xy)
            diffs.append(np.degrees(np.arccos(np.clip(np.dot(nn, cn), -1, 1))))
            nn_old = s.n if np.dot(s.n, cn_raw) >= 0 else -s.n
            diffs_old.append(np.degrees(np.arccos(np.clip(np.dot(nn_old, cn_raw), -1, 1))))
            cv = st["clip_vel"]; cs = np.linalg.norm(cv)
            ang = np.degrees(np.arccos(np.clip(
                np.dot(s.v_r, cv) / (np.linalg.norm(s.v_r) * cs + 1e-12), -1, 1)))
            in_cone.append(bool(ang <= 25.0 and 0.6 * cs <= np.linalg.norm(s.v_r) <= 1.4 * cs))
            if guard is not None:
                a_mins.append(float(a_min_q))

        kept = np.array(kept, dtype=int)
        # The answerable-fraction denominator is the SELECTED split's question count, not args.n —
        # with --split the un-selected side is never solved, so args.n would understate the rate.
        rate = len(kept) / max(len(sel), 1)
        d = np.array(diffs)
        # denominator report, net-rejected 先例: enforce mode really removed the questions;
        # stats mode ships them and reports what WOULD have been removed (franco's number).
        stroke_txt = ""
        if guard is not None:
            stroke_txt = (f", stroke-rejected {guard.over_budget_count}"
                          if args.stroke_guard == "enforce" else
                          f", stroke-guard would reject {guard.over_budget_count} "
                          f"(stats mode, 不拦)")
        print(f"[{name}] anchor={args.anchor} phase {st['anchor_phase']:.3f} -> clip frame "
              f"{st['strike_frame']}/{st['n_frames']}, strike point env={np.round(p_env, 3)}, "
              f"split={args.split} (sampled {n_train} train / {n_exam} exam), "
              f"solvable {len(kept)}/{len(sel)} = {rate:.0%}, net-rejected {net_rejects}"
              f"{stroke_txt}")
        if guard is not None and len(a_mins):
            arr = np.asarray(a_mins)
            print(f"  stroke-guard[{args.stroke_guard}]: kept-question a_min "
                  f"med={np.median(arr):.2f} p90={np.percentile(arr, 90):.2f} "
                  f"max={arr.max():.2f} m/s² vs a_max {guard.a_max:.2f} "
                  f"(L_deep {guard.L_deep:.3f} m; necessary condition — PASS != feasible)")
        if rate < 0.5:
            print(f"  ** WARNING: solvability <50% — the fixed strike point is badly placed for "
                  f"this speed range; move the point / relax the budget instead of training on this **")
        if len(kept):
            d_old = np.array(diffs_old)
            face_desc = {"registry": "calibrated+canonical", "baked": "baked+canonical",
                         "off": "raw+canonical"}[grip_mode]
            print(f"  difficulty (demanded face vs clip face): "
                  f"OLD raw-proxy med={np.median(d_old):.1f} deg p90={np.percentile(d_old, 90):.1f}"
                  f"  ->  NEW {face_desc} med={np.median(d):.1f} deg "
                  f"p90={np.percentile(d, 90):.1f} max={d.max():.1f}  |  answers inside the "
                  f"clip swing-velocity cone (25deg/0.6-1.4x): {np.mean(in_cone):.0%}")
        banks[name] = dict(
            contact_pos_env=p_env, clip_normal=st["clip_normal"], clip_vel=st["clip_vel"],
            incoming_vel=v_in[kept], demanded_normal=np.array(normals),
            demanded_vel=np.array(vels), difficulty_deg=d,
            residual_m=np.array(residuals), landing_xy_hope=np.array(landings),
            in_cone=np.array(in_cone, dtype=bool),
        )
        if guard is not None:
            # per-kept-question necessary lower bound [m/s²] — in stats mode this includes
            # the over-budget (flagged) questions, so downstream can re-threshold offline
            banks[name]["a_min"] = np.asarray(a_mins, dtype=np.float64)
        clip_meta[name] = dict(grip_mode=grip_mode, grip_desc=grip_desc,
                               rally_yaw_deg=rally_yaw, anchor=st["anchor"],
                               anchor_phase=st["anchor_phase"],
                               anchor_frame=st["strike_frame"], n_frames=st["n_frames"],
                               net_rejects=net_rejects,
                               stroke_guard=(None if guard is None else dict(
                                   mode=args.stroke_guard,
                                   L_deep_m=round(guard.L_deep, 4),
                                   deep_frame=guard.deep_frame,
                                   contact_frame=guard.contact_frame,
                                   a_max_mps2=round(guard.a_max, 3),
                                   v0_mps=args.stroke_v0,
                                   v_star_cap_mps=round(guard.v_star_cap, 3),
                                   over_budget=guard.over_budget_count,
                                   enforced=args.stroke_guard == "enforce")))
        grip_any = grip_any or grip_mode in ("registry", "baked")

    # --- torch virtual-ball closed-loop self-test (trainer-side venue physics) ----------
    if not args.no_check:
        import torch
        vb = _load_mod("s1_vb", os.path.join(
            WBT, "source", "whole_body_tracking", "whole_body_tracking",
            "tasks", "tracking", "mdp", "virtual_ball.py"))
        prm = vb.load_venue_params(os.path.join(REPO, "configs", "ball_physics_venue.yaml"))
        torch.set_default_dtype(torch.float64)
        for name, b in banks.items():
            if not len(b["incoming_vel"]):
                continue
            N = len(b["incoming_vel"])
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
            p0 = t(np.tile(b["contact_pos_env"], (N, 1)))
            v_plus, w_plus = vb.predict_paddle_contact(
                t(b["incoming_vel"]), t(b["demanded_vel"]), t(b["demanded_normal"]),
                torch.zeros(N, 3), prm)
            land = vb.coarse_landing(
                p0, v_plus, w_plus, prm,
                surface_z=0.76 + prm.ball_radius,  # physical contact plane (surface + R)
                net_x=args.near_x + 1.37, h=0.01, n_steps=200)
            tgt = torch.tensor(args.landing_env)
            err = torch.linalg.norm(land["land_xy"] - tgt, dim=-1)
            ok = land["land_valid"]
            # Independent NET cross-check: the numpy planner gate (spec.clears_net, Euler
            # @1 kHz) already rejected net-clippers; the torch venue physics (RK4, same
            # clearance expression as --phase-scan `legal`) must AGREE that every KEPT
            # answer clears net_top + R — numpy gate and torch check cross-validate.
            net_top = TABLE_SURFACE_Z + 0.1525
            net_margin = land["net_z"] - (net_top + prm.ball_radius)
            net_ok = land["net_valid"] & (net_margin > 0.0)
            margin_txt = (f"min margin {float(net_margin[land['net_valid']].min()):+.3f} m"
                          if bool(land["net_valid"].any()) else "no net-plane crossing at all")
            print(f"[{name}] torch closed-loop: landed {int(ok.sum())}/{N}, "
                  f"|err| med={err[ok].median():.3f} m  p90={err[ok].quantile(0.9):.3f} m  "
                  f"(planner residual med={np.median(b['residual_m'])*1000:.1f} mm)  |  "
                  f"net cleared {int(net_ok.sum())}/{N} ({margin_txt})")
            if int((~net_ok).sum()):  # explicit raise — a bare assert vanishes under python -O
                raise SystemExit(
                    f"{name}: {int((~net_ok).sum())}/{N} KEPT answers fail the torch-side net "
                    f"clearance (net_top + ball radius) that the numpy planner gate said they "
                    f"pass — the two flight models disagree; refusing to write the bank")

    # grip_applied / rally_yaw_applied: SINGLE-APPLICATION flags — a bank with these set has
    # calibrated-face (registry-rotated or baked) and rally-canonicalized clip vectors already;
    # downstream must refuse to rotate again (see clips[...] for per-clip mode/yaw).
    meta = dict(seed=args.seed, speed_range=args.speed_range, vy_max=args.vy_max,
                vz_range=args.vz_range, landing_env=args.landing_env, near_x=args.near_x,
                speed_budget=args.speed_budget, split=args.split, exam_frac=EXAM_FRAC,
                grip=args.grip, grip_applied=bool(grip_any), rally_yaw_applied=True,
                anchor=args.anchor, net_gate=True,
                stroke_guard=dict(
                    mode=args.stroke_guard,
                    budget_clips=[os.path.abspath(p)
                                  for p in (args.stroke_budget_clips or [])],
                    budget_scale=args.stroke_budget_scale,
                    v0_mps=args.stroke_v0,
                    a_max_mps2=(None if stroke_a_max is None else round(stroke_a_max, 3)),
                    envelope_per_clip={k: round(v, 3) for k, v in stroke_env.items()},
                    criterion="necessary-only sound-reject: a_min=(v*^2-v0^2)/(2*L_deep) > "
                              "a_max (stroke_interface_survey_2026-07-09 §3.1); PASS != "
                              "feasible; a_max = empirical blade-point Cartesian |acc| "
                              "envelope x scale (survey honest boundary 2 applies)"),
                clips=clip_meta)
    flat = {f"{n}/{k}": v for n, b in banks.items() for k, v in b.items()}
    # REAL json (audit round 2) — load_question_bank parses this and refuses banks whose
    # grip_applied / rally_yaw_applied flags are not both true (allow_legacy escape hatch).
    flat["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **flat)
    print(f"[bank] wrote {args.out}  (clips: {list(banks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
