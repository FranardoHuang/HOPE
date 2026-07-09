"""Mode-B ("venue-balls") sampling + virtual-return scoring for mujoco_eval_onnx.py.

MODE A (--target-source boxes, the default) samples racket TARGETS from the per-clip training
boxes — it answers "does the policy hit what it was trained on" (in-distribution). MODE B
(--target-source venue-balls) answers the REALISM question "can the policy handle what the venue
actually throws": it samples INCOMING BALLS from the fitted venue distribution
(configs/incoming_ball_venue.yaml, pooled/matchlike spec — the EVALUATION principle documented in
that file), derives the racket state each ball DEMANDS via the StrikeSpec inverse planner
(hope_ws/src/hope_planner: contact model + drag/Magnus flight, both the shared venue fit), and
feeds that demand through the unchanged target pipeline. At the exact-strike frame the ACHIEVED
racket state is pushed through the same contact + flight chain to score the virtual return
landing ("回球成功率").

PER-STRIKE FLOW (VenueBallSampler.sample):
  1. incoming ball at-strike state ~ venue matchlike spec:
       contact position  ~ U(contact_pos_q10_q90)          (venue table frame)
       velocity          ~ U(vb_boxes_matchlike vel boxes)  (vx < 0 by construction)
       spin              ~ isotropic direction x U(0, vb_spin_abs_max=34 rad/s) magnitude
                           (uniform-magnitude, NOT uniform-in-ball: mean 17 rad/s ~= the measured
                           spin_mag_mean 18.2; per-axis-uniform would bias the corners)
  2. swing side from the ball: y < fh_y_split -> forehand (clip 0, targets on -y), else backhand.
  3. landing target ~ U(landing box) on the opponent half (defaults derived from the TRAINING
     virtual-table geometry, see FRAMES below).
  4. StrikeSpecPlanner.solve(p_ball, v_ball, omega_ball, landing_xy, racket_speed_budget)
     -> racket target = (pos = ball contact point, vel = spec.v_r, normal = spec.n sign-matched
     to the clip's reference face normal, since the +Y paddle-face convention differs per swing
     and +/-n span the same face plane). solve() returning None (LM no-converge / speed budget)
     rejects the whole draw and resamples; rejections are counted and reported.

FIXED-NORMAL MODE (fixed_normal=True; --venue-fixed-normal; structural-finding path A,
docs/motion_and_contract_v3.md §6): step 4 uses StrikeSpecPlanner.solve_fixed_normal with the
normal PINNED at the swing side's clip reference face normal — the planner adapts to the
clip-locked face the policy actually produces, solving velocity only. The demanded normal then
equals what the policy naturally swings, so the eval answers "what return rate can the CURRENT
policy + an adapted planner reach" (the zero-training deployment ceiling). Expect a higher
solve-rejection rate (2 landing DOF chased with the face DOF given up); rejections resample the
whole draw (ball + landing) exactly like free mode, i.e. the planner effectively picks
reachable landings — that selection IS the deployment semantics being measured.

RETURN SCORING (VenueBallSampler.score_return, at the exact-strike frame, mirrors the training
gate in hope_commands._vb_evaluate):
  contacted = pos_err < vb_capture_radius (0.095 m) AND the paddle moving INTO the ball along the
  oriented contact normal faster than vb_min_approach_speed (0.3 m/s). The ACHIEVED racket state
  (site linear velocity + face normal) and the SAMPLED incoming ball go through
  ball_contact.predict_paddle_contact -> integrate_to_table_plane (flight start = the ACHIEVED
  racket position, training parity). landed_ok = contacted AND landing valid AND on the opponent
  half within bounds AND net cleared (net-plane z from a short pre-net integration of the same
  flight model).

FRAMES (three, all axis-aligned — translations only):
  * venue table frame ... incoming_ball_venue.yaml: receiver at the -X end, net at x=0, z above
                          the TABLE SURFACE. Incoming balls have vx < 0 (toward the receiver) —
                          the same orientation as the eval world, so velocities/spins map 1:1.
  * eval world (env) .... robot pelvis at the origin facing +x. The virtual table follows the
                          TRAINING convention (hope_commands RacketTargetCommandCfg: the model
                          under eval was trained with this table): near edge x = +0.5, surface
                          z = +0.76, centered on y — net at x = 1.87, far end x = 3.24. So
                          env = venue + (net_x_env, 0, table_surface_z).
                          NOTE: the task brief suggested a landing box of x [1.5, 2.6]; with the
                          training table that straddles the NET (x < 1.87 is our own half), so the
                          default box is geometry-derived instead: x [net+0.3(dink guard),
                          far-0.2], y [-0.5, 0.5]. Override via --venue-landing-*-range.
  * planner frame ....... hope_planner works with the table plane at z = 0
                          (integrate_to_table_plane finds the first downward z=0 crossing; x/y are
                          origin-agnostic). planner = env - (0, 0, table_surface_z).

V1 CAVEATS (documented, accepted):
  * Independent box sampling IGNORES the venue correlations (corr(vx,vz)=-0.44 etc. — see the
    yaml); only the sign structure is enforced (vx < 0, guaranteed by the matchlike box and
    asserted). A truncated-MVN sampler over (vx,vy,vz,z_contact,|w|) is the v2 upgrade once the
    covariance lands in the yaml.
  * The venue contact positions are a HUMAN receiver's — notably z (0.98-1.26 m above the floor)
    sits mostly ABOVE the robot's trained strike boxes (0.72-1.13 m). That is the point of the
    realism eval, not a bug: expect pos_pass to drop vs mode A.
  * spec.v_r and the contact model want the racket CONTACT-POINT velocity; the harness measures
    (and training scored) the racket-center site velocity — the omega_racket x r term (~<=0.2 m/s
    at |r| = 2 cm) is neglected on BOTH the demand and the scoring side, consistently.
  * Venue-spec numbers below MIRROR configs/incoming_ball_venue.yaml (repo convention, same as
    hope_planner/constants.py vs ball_physics_venue.yaml): do NOT read the yaml at runtime; if the
    venue refits, update both places.

Numpy-only. hope_planner is imported lazily by path (repo layout is fixed:
<repo>/hope_ws/src/hope_planner) so mode A keeps running without it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------------------------
# Venue incoming-ball spec — MIRRORS configs/incoming_ball_venue.yaml (fit 2026-07-04, pooled
# matchlike = tantiao excluded, n_v=82). EVAL principle from that file: realism mode uses the
# POOLED boxes, not the rebalanced training envelope.
# ---------------------------------------------------------------------------------------------
VENUE_YAML_REL = "configs/incoming_ball_venue.yaml"
# at_strike.contact_pos_q10_q90 (venue table frame: x rel. net, z above the table surface).
VENUE_CONTACT_POS_Q10_Q90 = ((-1.50, -0.92), (-0.35, 0.18), (0.22, 0.50))
# vb_boxes_matchlike velocity boxes (m/s, vx < 0 = toward the receiver).
VENUE_VEL_BOX_MATCHLIKE = ((-2.85, -0.82), (-0.46, 0.31), (-2.02, 0.49))
VENUE_SPIN_ABS_MAX = 34.0     # rad/s, vb_boxes_matchlike.vb_spin_abs_max (isotropic magnitude cap)

# Training virtual-table placement (hope_commands.RacketTargetCommandCfg defaults — the table the
# model under eval trained its landing rewards against) + capture-gate constants.
DEFAULT_TABLE_NEAR_X = 0.5    # vb_table_near_x: near table edge, env frame (robot 0.5 m behind)
DEFAULT_TABLE_SURFACE_Z = 0.76  # vb_table_surface_z: table surface above the env origin/floor
CAPTURE_RADIUS = 0.095        # vb_capture_radius: racket 0.075 + ball 0.020
MIN_APPROACH_SPEED = 0.3      # vb_min_approach_speed: phantom-block guard
LANDING_MIN_DEPTH = 0.3       # vb_min_landing_depth: dink guard -> default landing-box near edge
LANDING_FAR_MARGIN = 0.2      # default landing-box margin from the end line (eval choice)

# Forehand/backhand split on the ball's y (env frame == venue frame y): midpoint between the
# training forehand box's y-high (-0.24) and the backhand box's y-low (-0.07).
DEFAULT_FH_Y_SPLIT = -0.155


def hope_planner_path(repo_root: str) -> str:
    return os.path.join(repo_root, "hope_ws", "src", "hope_planner")


@dataclass
class VenueStrike:
    """One sampled venue ball + the racket demand it implies (all env/world frame)."""

    clip: int                     # swing clip chosen from the ball's y side
    ball_pos_w: np.ndarray        # (3,) incoming-ball position at strike == racket target position
    ball_vel_w: np.ndarray        # (3,) incoming-ball velocity at strike
    ball_spin_w: np.ndarray       # (3,) incoming-ball spin (rad/s)
    intended_landing_xy: np.ndarray  # (2,) sampled landing target on the opponent half
    target_pos_w: np.ndarray      # (3,) racket target position (= ball_pos_w)
    target_vel_w: np.ndarray      # (3,) racket target velocity (StrikeSpec.v_r)
    target_normal_w: np.ndarray   # (3,) racket target face normal (spec.n, clip-sign-matched)
    spec_speed: float             # |v_r| demanded (m/s)
    spec_iters: int               # LM iterations used by the solve
    tries: int                    # draws needed until a solvable ball (1 = first try)


@dataclass
class ReturnScore:
    """Virtual return of one exact-strike frame (achieved racket state x sampled ball)."""

    contacted: bool               # capture gate: pos_err < 0.095 AND approach > 0.3 m/s
    approach_speed: float         # v_racket . oriented contact normal (m/s)
    landing_valid: bool           # post-contact flight crossed the table plane downward
    landing_xy: np.ndarray        # (2,) achieved landing (env frame; nan if invalid)
    flight_time: float            # strike -> landing (s; nan if invalid)
    on_opponent: bool             # landing on the opponent half within table bounds
    net_crossed: bool             # flight crossed the net plane before landing
    net_z: float                  # ball z above the TABLE SURFACE at the net plane (nan if not)
    net_clear: bool               # net_crossed AND net_z > net height + ball radius
    landed_ok: bool               # contacted AND landing_valid AND on_opponent AND net_clear
    land_err: float               # ||landing - intended|| (m; nan if invalid landing)


class VenueBallSampler:
    """Sample venue incoming balls, invert them to racket demands, score virtual returns."""

    def __init__(self, repo_root, ref_normal_per_clip, num_clips,
                 table_near_x=DEFAULT_TABLE_NEAR_X, table_surface_z=DEFAULT_TABLE_SURFACE_Z,
                 landing_x_range=None, landing_y_range=(-0.5, 0.5),
                 fh_y_split=DEFAULT_FH_Y_SPLIT, speed_budget=10.0, max_tries=100,
                 fixed_normal=False, contact_fixed_env=None, spin_abs_max=None, vel_box=None):
        # hope_planner by path (numpy-only package; node.py's ROS import is NOT in __init__).
        hp = hope_planner_path(repo_root)
        if not os.path.isdir(os.path.join(hp, "hope_planner")):
            raise SystemExit(f"[FATAL] --target-source venue-balls needs hope_planner at {hp} "
                             f"(repo layout is fixed; found nothing there)")
        if hp not in sys.path:
            sys.path.insert(0, hp)
        from hope_planner.ball_contact import orient_normal, predict_paddle_contact
        from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor
        from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
        from hope_planner.strike_spec_planner import StrikeSpecPlanner

        self._orient_normal = orient_normal
        self._predict_contact = predict_paddle_contact
        self.physics = BallPhysics()
        self.config = PlannerConfig()
        self.table = TableParams()
        self.planner = StrikeSpecPlanner(self.physics, self.config, self.table)
        self.predictor = BallTrajectoryPredictor(self.physics, self.config, self.table)

        self.ref_normal_per_clip = np.asarray(ref_normal_per_clip, float)
        self.num_clips = int(num_clips)
        self.table_near_x = float(table_near_x)
        self.table_surface_z = float(table_surface_z)
        self.net_x = self.table_near_x + self.table.net_x            # env frame
        self.far_x = self.table_near_x + self.table.length
        self.half_w = self.table.width / 2.0
        self.net_clear_z = self.table.net_height + self.physics.radius  # above table surface
        if landing_x_range is None:
            landing_x_range = (self.net_x + LANDING_MIN_DEPTH, self.far_x - LANDING_FAR_MARGIN)
        self.landing_x_range = (float(landing_x_range[0]), float(landing_x_range[1]))
        self.landing_y_range = (float(landing_y_range[0]), float(landing_y_range[1]))
        assert self.landing_x_range[0] > self.net_x, \
            (f"landing x range {self.landing_x_range} reaches OUR half (net at x={self.net_x:.2f} "
             f"env frame — the training virtual table sits at near_x={self.table_near_x})")
        self.fh_y_split = float(fh_y_split)
        self.speed_budget = float(speed_budget)
        self.max_tries = int(max_tries)
        self.fixed_normal = bool(fixed_normal)
        # STAGE-EXAM overrides (2026-07-06, franco's staged-question doctrine): None = the venue
        # matchlike defaults (mode-B realism). Stage-1 paper: fixed contact point (env frame,
        # e.g. the v5-bh _cal anchor blade point), zero spin, venue-like speed tier.
        self.contact_fixed_env = None if contact_fixed_env is None else np.asarray(contact_fixed_env, float)
        self.spin_abs_max = VENUE_SPIN_ABS_MAX if spin_abs_max is None else float(spin_abs_max)
        self.vel_box = VENUE_VEL_BOX_MATCHLIKE if vel_box is None else tuple(vel_box)
        self.reset_counters()

    # ------------------------------------------------------------------------------------------
    def reset_counters(self):
        self.n_samples = 0            # accepted venue strikes handed to the rollout
        self.n_solve_fail = 0         # StrikeSpecPlanner.solve() -> None rejections
        self.n_sign_reject = 0        # sign-structure rejections (vx >= 0; box makes this ~0)
        self.iters_acc = 0            # LM iterations over accepted solves

    def counters(self):
        return dict(
            samples=self.n_samples, solve_fail=self.n_solve_fail, sign_reject=self.n_sign_reject,
            mean_solve_iters=(self.iters_acc / self.n_samples) if self.n_samples else float("nan"),
        )

    # ------------------------------------------------------------------------------------------
    def _u3(self, rng, box):
        return np.array([float(rng.uniform(lo, hi)) for lo, hi in box])

    def sample(self, rng) -> VenueStrike:
        """Draw one venue ball + its StrikeSpec racket demand (resampling on solve failure)."""
        for attempt in range(1, self.max_tries + 1):
            # 1. incoming ball at-strike state (venue table frame).
            if self.contact_fixed_env is not None:
                # stage exam: contact pinned (env frame given) -> convert to venue frame
                p_venue = self.contact_fixed_env - np.array([self.net_x, 0.0, self.table_surface_z])
            else:
                p_venue = self._u3(rng, VENUE_CONTACT_POS_Q10_Q90)
            v_ball = self._u3(rng, self.vel_box)
            w_dir = rng.standard_normal(3)
            w_dir /= max(float(np.linalg.norm(w_dir)), 1e-9)
            w_ball = w_dir * float(rng.uniform(0.0, self.spin_abs_max)) if self.spin_abs_max > 0 \
                else np.zeros(3)
            # Sign structure (correlation caveat, v1): the ball must approach the robot. The
            # matchlike box already guarantees vx < 0; full correlations are NOT enforced.
            if v_ball[0] >= 0.0:
                self.n_sign_reject += 1
                continue
            # 2. swing side from the ball's y (fh targets live on -y).
            clip = 0 if p_venue[1] < self.fh_y_split else 1
            clip = min(clip, self.num_clips - 1)
            # 3. landing target on the opponent half (env frame xy == planner xy).
            landing_xy = np.array([float(rng.uniform(*self.landing_x_range)),
                                   float(rng.uniform(*self.landing_y_range))])
            # 4. invert: what racket state does this ball demand? (planner frame: z rel. table)
            p_env = p_venue + np.array([self.net_x, 0.0, self.table_surface_z])
            p_planner = p_venue + np.array([self.net_x, 0.0, 0.0])
            if self.fixed_normal:
                # Path A: normal pinned at the clip's reference face — the planner adapts to
                # the policy's clip-locked face and only the velocity steers the landing.
                spec = self.planner.solve_fixed_normal(
                    p_planner, v_ball, w_ball, landing_xy, self.speed_budget,
                    n_fixed=self.ref_normal_per_clip[clip])
            else:
                spec = self.planner.solve(p_planner, v_ball, w_ball, landing_xy, self.speed_budget)
            if spec is None:
                self.n_solve_fail += 1
                continue
            # +/-n span the same face plane; match the sign of the clip's reference face normal
            # (the +Y paddle-face convention the harness scores nrm_err against) so a correct
            # strike scores ~0 deg instead of ~180.
            n = spec.n.copy()
            if float(np.dot(n, self.ref_normal_per_clip[clip])) < 0.0:
                n = -n
            self.n_samples += 1
            self.iters_acc += int(spec.iterations)
            return VenueStrike(
                clip=clip, ball_pos_w=p_env, ball_vel_w=v_ball, ball_spin_w=w_ball,
                intended_landing_xy=landing_xy, target_pos_w=p_env.copy(),
                target_vel_w=spec.v_r.copy(), target_normal_w=n,
                spec_speed=float(np.linalg.norm(spec.v_r)), spec_iters=int(spec.iterations),
                tries=attempt,
            )
        hint = (
            "With fixed_normal ON this is the EXPECTED verdict when the pinned clip face cannot "
            "legally return the venue distribution at all (measured 2026-07-05 for the hopex "
            "clips: forehand face reaches x<=1.4 m at ANY |v_r|<=6 — never clears the net; "
            "backhand only a net-hugging cross-court sliver outside the legal landing box). "
            "The path-A ceiling is then ~0, not a bug."
            if self.fixed_normal else
            "The landing box or speed budget is likely unreachable from the venue contact box — "
            "check the frame mapping.")
        raise SystemExit(
            f"[FATAL] venue-balls: no solvable incoming ball in {self.max_tries} draws "
            f"(solve_fail={self.n_solve_fail}, sign_reject={self.n_sign_reject}; "
            f"landing box {self.landing_x_range}x{self.landing_y_range}, speed budget "
            f"{self.speed_budget} m/s, fixed_normal={self.fixed_normal}). {hint}")

    # ------------------------------------------------------------------------------------------
    def _net_plane_z(self, p0, v0, w0):
        """Ball z (above the table surface) at the net plane, from a short pre-net integration.

        Same Euler scheme + physics (predictor._flight_acceleration: drag + gravity + Magnus) as
        integrate_to_table_plane, stopped at the net plane; landing before the net -> not crossed.
        Inputs in the PLANNER frame (z rel. table surface). Returns (net_z, net_crossed).
        """
        dt = self.config.dt_integrate
        max_steps = int(self.config.max_predict_time / dt)
        p = np.asarray(p0, float).copy()
        v = np.asarray(v0, float).copy()
        for _ in range(max_steps):
            a = self.predictor._flight_acceleration(v, w0)
            p_new = p + v * dt + 0.5 * a * dt ** 2
            v_new = v + a * dt
            if p[2] > 0.0 and p_new[2] <= 0.0 and p_new[0] < self.net_x:
                return float("nan"), False              # landed (or dropped) before the net
            if p[0] < self.net_x <= p_new[0]:
                dx = p_new[0] - p[0]
                frac = (self.net_x - p[0]) / dx if abs(dx) > 1e-12 else 0.5
                return float(p[2] + frac * (p_new[2] - p[2])), True
            p, v = p_new, v_new
        return float("nan"), False

    def score_return(self, strike: VenueStrike, racket_pos_w, racket_vel_w, racket_normal_w,
                     pos_err) -> ReturnScore:
        """Virtual return of the ACHIEVED racket state vs the SAMPLED ball (training-gate parity)."""
        v_in, w_in = strike.ball_vel_w, strike.ball_spin_w
        v_r = np.asarray(racket_vel_w, float)
        n_face = np.asarray(racket_normal_w, float)
        # capture gate (hope_commands._vb_evaluate): close at the strike frame AND moving INTO the
        # ball along the oriented contact normal.
        n_or = self._orient_normal(n_face, v_in, v_r)
        approach = float(np.dot(v_r, n_or))
        contacted = (float(pos_err) < CAPTURE_RADIUS) and (approach > MIN_APPROACH_SPEED)

        # contact + flight with the ACHIEVED racket state; flight starts at the ACHIEVED racket
        # position (training parity: _vb_evaluate rolls out from racket_pos_w).
        v_plus, w_plus = self._predict_contact(v_in, v_r, n_face, w_in, self.physics, self.config)
        p0_planner = np.asarray(racket_pos_w, float) - np.array([0.0, 0.0, self.table_surface_z])
        land = self.predictor.integrate_to_table_plane(p0_planner, v_plus, w_plus)

        landing_valid = land is not None
        landing_xy = np.array([float("nan")] * 2)
        flight_time = float("nan")
        on_opp = False
        land_err = float("nan")
        if landing_valid:
            landing_xy, flight_time = np.asarray(land[0], float), float(land[1])
            on_opp = (self.net_x < landing_xy[0] <= self.far_x) and (abs(landing_xy[1]) <= self.half_w)
            land_err = float(np.linalg.norm(landing_xy - strike.intended_landing_xy))
        net_z, net_crossed = self._net_plane_z(p0_planner, v_plus, w_plus)
        net_clear = bool(net_crossed and net_z > self.net_clear_z)
        return ReturnScore(
            contacted=bool(contacted), approach_speed=approach,
            landing_valid=bool(landing_valid), landing_xy=landing_xy, flight_time=flight_time,
            on_opponent=bool(on_opp), net_crossed=bool(net_crossed), net_z=float(net_z),
            net_clear=net_clear,
            landed_ok=bool(contacted and landing_valid and on_opp and net_clear),
            land_err=land_err,
        )


class BankExamSampler(VenueBallSampler):
    """STAGE-1 OFFICIAL EXAM (--target-source bank): questions straight from an exam bank npz.

    Same-source law (yikang handoff 2026-07-06): the S1 exam asks the exam split of the SAME
    bank family the arm trained on. Loading goes through stage1_question_bank.load_question_bank
    so the meta guards (grip_applied / rally_yaw_applied) are ENFORCED, never bypassed. Per
    question: target pos = the clip's fixed contact point, target vel/normal = the solved
    answer (demanded_normal feeds the 179-D obs tail per question), incoming ball = the
    question's own sampled ball (npz incoming_vel; spinless stage 1). Contact/flight/net
    judging is inherited from VenueBallSampler verbatim (same constants as the bank's own
    entry gate and the training reward).
    """

    def __init__(self, repo_root, ref_normal_per_clip, num_clips, bank_path,
                 clip_names=("forehand", "backhand"), **kw):
        super().__init__(repo_root=repo_root, ref_normal_per_clip=ref_normal_per_clip,
                         num_clips=num_clips, **kw)
        import json
        if num_clips != len(clip_names):
            raise SystemExit(f"[FATAL] bank exam has {len(clip_names)} clips, eval replays "
                             f"{num_clips} — motion files must be the same pair the bank was "
                             f"generated from (0=forehand, 1=backhand)")
        # torch-side loader = THE validation gate; numpy views afterwards. Import the MODULE FILE
        # directly when $HOPE_STAGE1_QB is set (spec_from_file_location) — the package __init__
        # chain pulls isaaclab, which the mjeval venv rightly lacks; the module itself only needs
        # json/numpy/torch (mjeval venv: cpu torch wheel installed 2026-07-06).
        qb_py = os.environ.get("HOPE_STAGE1_QB", "")
        if qb_py:
            import importlib.util as _ilu
            _sp = _ilu.spec_from_file_location("stage1_question_bank", qb_py)
            _m = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_m)
            load_question_bank = _m.load_question_bank
        else:
            from whole_body_tracking.tasks.tracking.mdp.stage1_question_bank import (
                load_question_bank,
            )
        qb = load_question_bank(bank_path, device="cpu", clip_names=clip_names)
        data = np.load(bank_path)
        self.bank_path = bank_path
        self.clip_names = tuple(clip_names)
        self.q_contact = qb.contact_pos.numpy().astype(np.float64)      # (C,3)
        self.q_vel = qb.demanded_vel.numpy().astype(np.float64)         # (C,Qmax,3)
        self.q_nrm = qb.demanded_normal.numpy().astype(np.float64)      # (C,Qmax,3)
        self.q_diff = qb.difficulty_deg.numpy().astype(np.float64)      # (C,Qmax)
        self.q_counts = [int(n) for n in qb.counts.numpy()]             # (C,)
        # A-frame (+Y) guard (2026-07-09 单翻病防复发, mjeval 侧的孪生卫兵): every demanded
        # normal must be same-side with the clip's +Y reference face. Bank exam scoring is
        # A-vs-A by design (measured raw +Y vs bank rows verbatim; the striking-face sign table
        # is REJECTED in bank mode upstream) — a bank re-emitted in the flipped ("B") convention
        # would silently grade every backhand ~180° wrong. Training has the same guard
        # (RacketTargetCommand._check_question_bank_face_frame); this one catches hand-passed
        # --exam-bank files that never went through a training env. Shipped banks: min dot 0.86.
        if ref_normal_per_clip is None:
            print("[bank-exam] WARN: A-frame guard SKIPPED (no ref_normal_per_clip passed) — "
                  "a flipped-convention bank would go undetected", flush=True)
            ref_arr = None
        else:
            ref_arr = np.asarray(ref_normal_per_clip, np.float64).reshape(len(clip_names), 3)
        if ref_arr is not None:
            for c, name in enumerate(clip_names):
                q = self.q_counts[c]
                if q <= 0:
                    continue
                d = self.q_nrm[c, :q] @ ref_arr[c]
                if float(d.min()) <= 0.0:
                    raise SystemExit(
                        f"[FATAL] exam bank {bank_path}: clip {name!r} has {int((d <= 0).sum())}/{q} "
                        f"demanded normals OPPOSITE the +Y reference face (min dot {float(d.min()):.4f}) "
                        f"— either a striking-face-convention ('B') bank, or the EVAL-side reference "
                        f"was flipped (MOUNT_NORMAL_SIGN_PER_CLIP / --mount-normal-sign-per-clip): "
                        f"check which side first. Bank scoring is +Y/A-frame by design; NEVER "
                        f"re-emit banks in the flipped convention.")
        self.q_income = []
        for c, name in enumerate(clip_names):
            inc = np.asarray(data[f"{name}/incoming_vel"], np.float64).reshape(-1, 3)
            if len(inc) != self.q_counts[c]:
                raise SystemExit(f"[FATAL] bank {bank_path}: {name} incoming_vel rows "
                                 f"{len(inc)} != question count {self.q_counts[c]}")
            self.q_income.append(inc)
        meta = json.loads(bytes(np.asarray(data["meta_json"], dtype=np.uint8)).decode("utf-8"))
        self.bank_meta = meta
        land = meta.get("landing_env") or [2.555, 0.0]
        self.q_landing = np.asarray(land, np.float64).reshape(2)
        # lazily shuffled with the rollout rng (deterministic per --seed); exhaust-then-wrap.
        self._order = None
        self._next = [0] * len(clip_names)
        self.asked = [0] * len(clip_names)
        self.wrapped = [0] * len(clip_names)

    def sample(self, rng):
        if self._order is None:
            self._order = [rng.permutation(n) for n in self.q_counts]
        rem = [max(n - self.asked[c], 0) for c, n in enumerate(self.q_counts)]
        tot = sum(rem)
        if tot > 0:
            c = int(rng.choice(len(rem), p=np.asarray(rem, np.float64) / tot))
        else:
            c = int(rng.integers(len(self.q_counts)))
        i = int(self._order[c][self._next[c]])
        self._next[c] += 1
        if self._next[c] >= self.q_counts[c]:
            self._next[c] = 0
            self.wrapped[c] += 1
        self.asked[c] += 1
        v = self.q_vel[c, i]
        return VenueStrike(
            clip=c, ball_pos_w=self.q_contact[c].copy(),
            ball_vel_w=self.q_income[c][i].copy(), ball_spin_w=np.zeros(3),
            intended_landing_xy=self.q_landing.copy(),
            target_pos_w=self.q_contact[c].copy(), target_vel_w=v.copy(),
            target_normal_w=self.q_nrm[c, i].copy(),
            spec_speed=float(np.linalg.norm(v)), spec_iters=0, tries=1)

    def denominator_report(self):
        """判卷分母法则 lines: per side — questions in the exam split (kept), how many were
        asked/wrapped this run, and the share of answers inside the 25-deg face cone."""
        lines = []
        for c, name in enumerate(self.clip_names):
            n = self.q_counts[c]
            cone = float((self.q_diff[c, :n] <= 25.0).mean()) if n else float("nan")
            med = float(np.median(self.q_diff[c, :n])) if n else float("nan")
            lines.append(f"  {name}: exam questions kept={n} asked={self.asked[c]} "
                         f"wrapped={self.wrapped[c]}x | face-cone(<=25deg) share={cone:.1%} "
                         f"difficulty median={med:.1f}deg")
        g = self.bank_meta.get("grip", "?")
        lines.append(f"  meta: grip={g} grip_applied={self.bank_meta.get('grip_applied')} "
                     f"rally_yaw_applied={self.bank_meta.get('rally_yaw_applied')} "
                     f"landing_env={self.q_landing.tolist()}")
        return lines
