"""CONTINUOUS question generation — draw a ball, solve the target, batched.

人话(owner 裁定):离散题库是**考试**用的,训练必须连续采样。所以训练侧不再回放一张定长的题
表,而是从连续分布里画球(速度/旋转/落点/触球点都是区间,按 clip 各有自己的框),再用成熟的逆解
当场把"该把拍子摆成什么样"算出来。解不出来的球是一个**响亮且被计数**的事件——重新画一颗,不许
悄悄拿别的目标顶上。

Two paths, ONE physics:

* CONTINUOUS (training) — this module. Ball drawn from a continuous parameterisation; answer from
  ``solve_strike_specs`` / ``solve_strike_specs_fixed_dir`` on the trainer's own venue model.
* DISCRETE (exam, reproducibility) — the pre-solved npz bank, unchanged. It stays.

They are the same physics because they call the SAME solver on the SAME ``VirtualBallParams``.
``tests/test_continuous_vs_bank_parity.py`` pushes one BANK ROW's ball through the continuous
solver and asserts both answers land within the solver's own tolerance of each other and of the
aim, clear the net, and carry the SAME FACE SIGN — it deliberately does NOT assert the demanded
state agrees bitwise, because it does not: the 5-DoF solution manifold under a shared speed
regulariser is not pointwise unique (a face rotated a few degrees with a compensating velocity
lands in the same place). ``parity_report`` below is that assertion body; the boot gate in
``hope_commands`` imports the SAME function, so the property is checked on this machine at this
commit, not only in pytest.

FAILURE POLICY, structural not disciplinary. A drawn ball with no legal answer is never
substituted. ``generate`` returns an ``ok`` mask plus a per-reason histogram; the caller redraws
those rows (bounded rounds) and, if any row still has no answer after the budget, that is a loud
counted event — ``QuestionDrawResult.exhausted``. The unsolved rows come back **NaN-filled**
(``resid_m`` is kept for diagnosis) so that no caller, now or later, can install one by forgetting
to mask on ``ok``.

FRAME. Everything here is ENV-LOCAL — the same convention as ``coarse_landing`` / the trainer's own
scorer. ``generate`` takes no ``origins`` and returns no world coordinates ON PURPOSE: the caller
adds the env origin at install time, exactly like the bank seam does. (The old signature took
``origins``, shifted the contact point by it but not aim_x, and compared against env-local
surface/net scalars; both callers happened to pass zeros so it never showed. At origin=(3,0,0) it
converged confidently on a target 3 m short with the net behind the robot.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

if __package__:
    from .strike_spec_torch import solve_strike_specs
    from .stroke_adapt_torch import (
        REASONS,
        _DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY,
        base_yaw_of,
        solve_strike_specs_fixed_dir,
    )
    from .virtual_ball import (
        VirtualBallParams,
        coarse_landing,
        predict_paddle_contact,
    )
else:
    # MuJoCo-GPU imports this exact four-file pure-Torch closure as top-level
    # modules from the pinned mdp source directory.  No package init, dynamic
    # loader, sys.path mutation, or second solver implementation is involved.
    from strike_spec_torch import solve_strike_specs
    from stroke_adapt_torch import (
        REASONS,
        _DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY,
        base_yaw_of,
        solve_strike_specs_fixed_dir,
    )
    from virtual_ball import (
        VirtualBallParams,
        coarse_landing,
        predict_paddle_contact,
    )

_EPS = 1e-9

# Private capability for the ActionBall diagnostic producer.  The ordinary
# public solver keeps every synchronous input/prototype proof below.  The
# diagnostic producer has already constructed the batch from strictly
# validated sampler births and pinned runtime tables, so repeating those
# proofs once per redraw round only drains the CUDA stream.  Exact identity
# prevents typo-shaped truthy flags from selecting the fast path.
_DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY = object()

#: Reject-reason code indices into ``stroke_adapt_torch.REASONS`` used by the free-solve path.
_R_NO_LANDING = 0
_R_RESID = 1
_R_SPEED_OVER = 2
_R_SPEED_UNDER = 3
_R_NET = 5
_R_FACE = 6
_R_CONTACT_ENVELOPE = 7
_CONTINUOUS_REASONS = REASONS + ("contact_normal_speed_out_of_fit",)

# Device-resident producer-fault ABI.  These bits describe the input row, not
# whether the numerical inverse solve found an installable answer.  A valid but
# infeasible question therefore keeps ``producer_fault_bits == 0`` and its
# ordinary construction rejection reason.  The future construction-bound R05
# owner must include this exact int64 row in its sole packed reveal and CENSOR
# every nonzero row before any task/ball write.
PRODUCER_FAULT_NONFINITE_PROPOSAL = 1 << 0
PRODUCER_FAULT_REFERENCE_NORMAL = 1 << 1
PRODUCER_FAULT_BASE_QUATERNION = 1 << 2
PRODUCER_FAULT_ACTION_RANGE = 1 << 3
PRODUCER_FAULT_PROTOTYPE_DIRECTION = 1 << 4
PRODUCER_FAULT_PROTOTYPE_SPEED = 1 << 5
PRODUCER_FAULT_PROTOTYPE_FACE_SIGN = 1 << 6
PRODUCER_FAULT_MASK = (1 << 7) - 1

# --- incoming-ball birth-consistency gate (Franco 2026-07-28) ---------------
# 人话:把来球"往回倒 ttc 秒",最少也得从这条线以外出发。低速球配短到球时间会把
# 出生点反推到网这边("球出生在半路"),这种题物理上不是从对面打过来的球,必须
# 具名拒绝。线性下界:真实来球在飞行中只会被阻力减速,出生点只会比这个下界更远,
# 所以"连下界都过不了网"就足以判死,不需要整段反向积分。
BALL_BIRTH_NET_MARGIN_M = 0.05
BALL_BIRTH_REJECTION_REASON = "ball_birth_not_beyond_net"


def ball_birth_x_lower_bound_m(
    contact_x_w_m: float,
    incoming_velocity_x_w_mps: float,
    time_to_contact_s: float,
) -> float:
    """Linear lower bound on the incoming ball birth x (env ``W`` frame).

    The incoming ball travels toward the robot (negative x), so integrating
    backwards in time ADDS ``|v_x| * ttc`` to the contact x.  Drag can only
    have decelerated the incoming ball, so the true birth x is at or beyond
    this bound.
    """
    return (
        float(contact_x_w_m)
        + abs(float(incoming_velocity_x_w_mps)) * float(time_to_contact_s)
    )


def ball_birth_not_beyond_net(
    contact_x_w_m: float,
    incoming_velocity_x_w_mps: float,
    time_to_contact_s: float,
    *,
    net_x_m: float,
    margin_m: float = BALL_BIRTH_NET_MARGIN_M,
) -> bool:
    """True when even the optimistic birth-x bound cannot clear the net plane."""
    return ball_birth_x_lower_bound_m(
        contact_x_w_m, incoming_velocity_x_w_mps, time_to_contact_s
    ) < float(net_x_m) + float(margin_m)

# The venue contact fit is not an extrapolation license.  A candidate whose
# selected physical face is not approaching, or whose normal relative speed is
# outside the fitted data range, is not installable.
CONTACT_NORMAL_SPEED_MIN_MPS = 1.4
CONTACT_NORMAL_SPEED_MAX_MPS = 7.2


@dataclass
class ContinuousQuestionCfg:
    """The continuous parameterisation. Every field is a RANGE, never a list of cases."""

    # incoming ball, env-local m/s; per-clip rows override the shared box when given
    vel_range: tuple = ((-4.5, -2.0), (-0.6, 0.6), (-1.0, 0.5))
    vel_range_per_clip: tuple | None = None
    spin_abs_max: float = 50.0
    spin_abs_max_per_clip: tuple | None = None
    # contact point (== the ball's arrival point), env-origin-relative m; per-clip rows required
    # for a multi-clip run (each stroke reaches a different region).
    pos_range_per_clip: tuple | None = None
    pos_range: tuple = ((0.50, 0.62), (-0.45, 0.45), (0.80, 1.20))
    # aim on the opponent half, env-local m (net at 1.87, far edge 3.24, |y| <= 0.7625)
    aim_x_range: tuple = (2.20, 2.90)
    aim_y_range: tuple = (-0.55, 0.55)
    # solver
    tol_m: float = 0.02
    n_iters: int = 12
    speed_budget: float = 3.4          # gate_speed_max - margin (pp_policy.hpp:234)
    max_redraw_rounds: int = 3
    fixed_direction: bool = False      # True -> the stroke adapter (direction = stroke identity)


@dataclass
class ProposalLedger:
    """Flat, lossless accounting for every ball proposed across every redraw round.

    ``QuestionDrawResult`` keeps only the admitted answer for each requested row (and the legacy
    ``attempted_v_ball_in`` keeps only its last draw).  Curriculum control needs a different
    object: every proposal, including a rejected hard draw that was later replaced by an easier
    admitted draw.  All present tensors share the leading ``P == proposal_count`` dimension, so
    a caller can mask by ``clip_id`` and bucket any position/velocity/spin axis without reverse
    engineering a reason histogram.
    """

    request_index: torch.Tensor       # (P,) row in the requested clip_ids batch
    clip_id: torch.Tensor             # (P,) action / motion index
    round_index: torch.Tensor         # (P,) one-based redraw round
    p_contact: torch.Tensor           # (P,3)
    v_ball_in: torch.Tensor           # (P,3)
    w_ball_in: torch.Tensor           # (P,3)
    aim_xy: torch.Tensor              # (P,2)
    reason_code: torch.Tensor         # (P,) -1 admitted, else index into REASONS
    admitted: torch.Tensor            # (P,) bool
    resid_m: torch.Tensor             # (P,) scorer-replayed residual for fixed-dir
    # Exact external-proposal provenance.  Random generate() leaves these None because its base
    # and reference provenance already live in the caller's run contract.
    ref_normal: torch.Tensor | None = None   # (P,3) raw +Y face, before normalisation
    base_quat: torch.Tensor | None = None    # (P,4) exact supplied wxyz

    def __len__(self) -> int:
        return int(self.clip_id.shape[0])


@dataclass(frozen=True)
class DeviceProposalSolveResult:
    """Device-resident result of one exact fixed-action proposal solve.

    This is a numerical construction result, not a portable receipt and not a
    complete Device-R05 candidate bank.  In particular, it contains no Python
    reason histogram, JSON/hash identity, or host packet.  A construction-bound
    question authority must still join these rows to Motion timing and the
    Racket/physical after-images before Device-R05 can select a cell.

    Invalid producer rows are represented by ``producer_fault_bits`` with the
    exact seven-bit ABI above.  They are sanitized before numerical evaluation
    so a NaN, bad quaternion or out-of-range action cannot poison another row,
    then NaN-masked in every installable geometric tensor.  Their original
    inputs and a schema-valid rejection code remain device-resident in
    ``proposals`` so the one packed reveal boundary can record and block from
    the same batch.  The bits are facts, not an authorization verdict.
    """

    p_contact: torch.Tensor
    v_racket: torch.Tensor
    n_racket: torch.Tensor
    v_ball_in: torch.Tensor
    w_ball_in: torch.Tensor
    aim_xy: torch.Tensor
    ok: torch.Tensor
    resid_m: torch.Tensor
    attempted_v_ball_in: torch.Tensor
    producer_fault_bits: torch.Tensor
    proposals: ProposalLedger


@dataclass(frozen=True)
class ProposalHostPacket:
    """Immutable host copy of the exact per-row solver outputs consumed by Python.

    The action-ball producer needs rejection codes, admission decisions and the solved racket
    rows to build receipts.  Copying those tensors independently drains the CUDA stream once per
    field.  ``solve_proposals`` instead packs them into one solver-dtype tensor, transfers it once,
    validates the two encoded discrete columns, and freezes the result here.  No mutable device
    view is retained as receipt evidence.
    """

    reason_codes: tuple[int, ...]
    admitted: tuple[bool, ...]
    racket_velocity_rows: tuple[tuple[float, float, float], ...]
    racket_normal_rows: tuple[tuple[float, float, float], ...]
    residual_rows: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.reason_codes)


@dataclass
class QuestionDrawResult:
    """One batch of solved questions plus the accounting that makes failures visible.

    Rows with ``ok=False`` are NaN in every geometric field — an unsolved ball is not installable
    by construction, not merely by convention. ``resid_m`` survives for diagnosis.
    """

    p_contact: torch.Tensor            # (N,3) the BALL's own arrival point, ENV-LOCAL
    v_racket: torch.Tensor             # (N,3)
    n_racket: torch.Tensor             # (N,3) unit, explicit face
    v_ball_in: torch.Tensor            # (N,3)
    w_ball_in: torch.Tensor            # (N,3)
    aim_xy: torch.Tensor               # (N,2) ENV-LOCAL
    ok: torch.Tensor                   # (N,) bool
    resid_m: torch.Tensor              # (N,)
    #: (N,3) the LAST ATTEMPTED incoming ball for EVERY row, solved or not. Accounting only — it
    #: is never an installable target (no answer travels with it), and it exists because a
    #: per-regime accept ledger cannot see which regime disappeared if the failed draws are NaN.
    attempted_v_ball_in: torch.Tensor = None
    rounds_used: int = 0
    exhausted: int = 0                 # rows still unsolved after max_redraw_rounds
    reason_counts: dict = field(default_factory=dict)
    proposal_count: int = 0
    proposals: ProposalLedger | None = None
    # ``solve_proposals`` always populates this immutable, one-transfer packet.  ``generate`` can
    # contain several redraws per requested row, so its proposal/output cardinalities differ and
    # it deliberately leaves the exact-once packet absent.
    proposal_host_packet: ProposalHostPacket | None = None


def _build_proposal_host_packet(
    *,
    reason_codes: torch.Tensor,
    admitted: torch.Tensor,
    producer_fault_bits: torch.Tensor,
    racket_velocity: torch.Tensor,
    racket_normal: torch.Tensor,
    residual: torch.Tensor,
) -> ProposalHostPacket:
    """Copy a diagnostic result through one ordered device-to-host transfer.

    This is the legacy portable wrapper's boundary, not Device-R05's future
    construction-bound packed reveal.  It fails synchronously on any producer
    fault; the production owner will instead encode the same bits in its global
    packet and select the typed CENSOR terminal.
    """

    if not isinstance(reason_codes, torch.Tensor) or reason_codes.dtype != torch.long:
        raise TypeError("proposal host packet reason codes must be torch.long")
    if not isinstance(admitted, torch.Tensor) or admitted.dtype != torch.bool:
        raise TypeError("proposal host packet admitted mask must be torch.bool")
    if (
        not isinstance(producer_fault_bits, torch.Tensor)
        or producer_fault_bits.dtype != torch.long
    ):
        raise TypeError("proposal host packet producer fault bits must be torch.long")
    if not isinstance(racket_velocity, torch.Tensor) or not isinstance(
        racket_normal, torch.Tensor
    ):
        raise TypeError("proposal host packet racket rows must be tensors")
    if not isinstance(residual, torch.Tensor):
        raise TypeError("proposal host packet residual must be a tensor")
    row_count = int(reason_codes.shape[0])
    if (
        tuple(reason_codes.shape) != (row_count,)
        or tuple(admitted.shape) != (row_count,)
        or tuple(producer_fault_bits.shape) != (row_count,)
        or tuple(racket_velocity.shape) != (row_count, 3)
        or tuple(racket_normal.shape) != (row_count, 3)
        or tuple(residual.shape) != (row_count,)
    ):
        raise ValueError("proposal host packet tensors have inconsistent shapes")
    dtype, device = racket_velocity.dtype, racket_velocity.device
    if (
        not dtype.is_floating_point
        or racket_normal.dtype != dtype
        or residual.dtype != dtype
        or racket_normal.device != device
        or residual.device != device
        or reason_codes.device != device
        or admitted.device != device
        or producer_fault_bits.device != device
    ):
        raise ValueError(
            "proposal host packet tensors must share one floating dtype and device"
        )

    # Float32/float64 represent the seven-bit producer mask, tiny integer reason
    # schema and 0/1 admission bit exactly.  Keeping all ten columns in the
    # solver dtype preserves the exact historical numeric rows while consuming
    # the producer fault in the same single host transfer.
    packed_rows = torch.cat(
        (
            producer_fault_bits.to(dtype=dtype).unsqueeze(-1),
            reason_codes.to(dtype=dtype).unsqueeze(-1),
            admitted.to(dtype=dtype).unsqueeze(-1),
            racket_velocity,
            racket_normal,
            residual.unsqueeze(-1),
        ),
        dim=-1,
    ).detach().cpu().tolist()

    host_reasons = []
    host_admitted = []
    host_velocity = []
    host_normal = []
    host_residual = []
    for row in packed_rows:
        if not isinstance(row, list) or len(row) != 10:
            raise RuntimeError("proposal host packet transfer returned a malformed row")
        raw_fault_bits = float(row[0])
        fault_bits = int(raw_fault_bits)
        if (
            not math.isfinite(raw_fault_bits)
            or raw_fault_bits != float(fault_bits)
            or fault_bits < 0
            or fault_bits & ~PRODUCER_FAULT_MASK
        ):
            raise RuntimeError(
                "proposal host packet producer fault bits are invalid"
            )
        if fault_bits:
            raise RuntimeError(
                "diagnostic proposal producer fault bits are nonzero: "
                f"0x{fault_bits:02x}"
            )
        raw_reason = float(row[1])
        reason = int(raw_reason)
        if not math.isfinite(raw_reason) or raw_reason != float(reason):
            raise RuntimeError("proposal host packet reason code is not an exact integer")
        raw_admitted = float(row[2])
        if raw_admitted not in (0.0, 1.0):
            raise RuntimeError("proposal host packet admitted value is not exactly 0 or 1")
        is_admitted = raw_admitted == 1.0
        if is_admitted:
            if reason != -1:
                raise RuntimeError(
                    "proposal host packet admitted row must carry reason code -1"
                )
        elif not 0 <= reason < len(_CONTINUOUS_REASONS):
            raise RuntimeError(
                "proposal host packet rejected row has an invalid reason code"
            )
        host_reasons.append(reason)
        host_admitted.append(is_admitted)
        host_velocity.append(tuple(float(value) for value in row[3:6]))
        host_normal.append(tuple(float(value) for value in row[6:9]))
        host_residual.append(float(row[9]))

    return ProposalHostPacket(
        reason_codes=tuple(host_reasons),
        admitted=tuple(host_admitted),
        racket_velocity_rows=tuple(host_velocity),
        racket_normal_rows=tuple(host_normal),
        residual_rows=tuple(host_residual),
    )


def _rows(table, clip_ids, device, dtype):
    t = torch.as_tensor(table, dtype=dtype, device=device)
    return t[clip_ids]


def _uniform_box(box_rows, gen, device, dtype):
    """box_rows (N,3,2) -> (N,3) uniform draw. Continuous by construction: no case list."""
    u = torch.rand(box_rows.shape[0], 3, device=device, dtype=dtype, generator=gen)
    return box_rows[:, :, 0] + (box_rows[:, :, 1] - box_rows[:, :, 0]) * u


def _selected_direction_world(v_hat_b, selected_clip_ids, yaw, device, dtype):
    """Rotate only each row's selected action direction, without materialising ``(M,K,3)``.

    The old adapter seam built every action direction for every proposal and selected one
    afterwards.  That is tolerable for K=5 but becomes an avoidable O(M*K) allocation for K=93.
    """
    d_b = v_hat_b.to(device=device, dtype=dtype)[selected_clip_ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    d = torch.stack([
        c * d_b[:, 0] - s * d_b[:, 1],
        s * d_b[:, 0] + c * d_b[:, 1],
        d_b[:, 2],
    ], dim=-1)
    return d / (torch.linalg.norm(d, dim=-1, keepdim=True) + _EPS)


def _empty_proposal_ledger(device, dtype) -> ProposalLedger:
    """Return a correctly typed/device-resident empty ledger (N=0 or zero redraw budget)."""
    return ProposalLedger(
        request_index=torch.empty((0,), dtype=torch.long, device=device),
        clip_id=torch.empty((0,), dtype=torch.long, device=device),
        round_index=torch.empty((0,), dtype=torch.long, device=device),
        p_contact=torch.empty((0, 3), dtype=dtype, device=device),
        v_ball_in=torch.empty((0, 3), dtype=dtype, device=device),
        w_ball_in=torch.empty((0, 3), dtype=dtype, device=device),
        aim_xy=torch.empty((0, 2), dtype=dtype, device=device),
        reason_code=torch.empty((0,), dtype=torch.long, device=device),
        admitted=torch.empty((0,), dtype=torch.bool, device=device),
        resid_m=torch.empty((0,), dtype=dtype, device=device),
    )


def _fixed_direction_contract(
    *,
    N: int,
    device,
    dtype,
    ref_normal: torch.Tensor | None,
    net_top_z: float | None,
    surface_z: float,
    net_x: float,
    tol_m: float,
    h: float,
    n_steps: int,
    speed_budget: float,
    protos,
    base_quat,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalize inputs that are mandatory for scorer-equivalent fixed-dir solves."""
    if N <= 0:
        raise ValueError("fixed-direction question generation needs at least one requested row")
    if protos is None or base_quat is None:
        raise ValueError(
            "ContinuousQuestionCfg.fixed_direction=True needs the stroke prototypes and "
            "the base orientation: the direction it holds fixed IS the stroke's identity"
        )
    if ref_normal is None:
        raise ValueError(
            "ContinuousQuestionCfg.fixed_direction=True needs ref_normal: fixed-dir answers must "
            "be signed to the runtime raw +Y clip face, including backhands"
        )
    if net_top_z is None or not math.isfinite(float(net_top_z)):
        raise ValueError(
            "ContinuousQuestionCfg.fixed_direction=True needs a finite scorer net_top_z"
        )
    if not math.isfinite(float(surface_z)) or float(net_top_z) <= float(surface_z):
        raise ValueError(
            f"net_top_z must be above the scorer landing plane, got "
            f"net_top_z={net_top_z!r}, surface_z={surface_z!r}"
        )
    if not math.isfinite(float(net_x)):
        raise ValueError(f"net_x must be finite, got {net_x!r}")
    if not math.isfinite(float(tol_m)) or float(tol_m) <= 0.0:
        raise ValueError(f"tol_m must be finite and positive, got {tol_m!r}")
    if not math.isfinite(float(h)) or float(h) <= 0.0:
        raise ValueError(f"h must be a finite positive scorer step, got {h!r}")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) <= 0:
        raise ValueError(f"n_steps must be a positive integer scorer horizon, got {n_steps!r}")
    if (
        isinstance(speed_budget, bool)
        or not math.isfinite(float(speed_budget))
        or float(speed_budget) <= 0.0
    ):
        raise ValueError(
            f"speed_budget must be finite and positive, got {speed_budget!r}"
        )
    if (
        isinstance(getattr(protos, "face_sign", None), bool)
        or not hasattr(protos, "face_sign")
    ):
        raise ValueError(
            "fixed-direction prototypes must declare per-action face_sign"
        )
    face_sign = torch.as_tensor(
        protos.face_sign, dtype=dtype, device=device
    )
    if (
        face_sign.ndim != 1
        or not bool(torch.isfinite(face_sign).all())
        or not bool(((face_sign == 1.0) | (face_sign == -1.0)).all())
    ):
        raise ValueError(
            "fixed-direction prototype face_sign must be a finite 1-D +/-1 table"
        )
    if (
        not math.isfinite(float(getattr(protos, "speed_min").min()))
        or not math.isfinite(float(getattr(protos, "speed_max").max()))
    ):
        raise ValueError("fixed-direction prototype speed bounds must be finite")
    if (
        not math.isfinite(float(getattr(protos, "v_hat_b").min()))
        or not math.isfinite(float(getattr(protos, "v_hat_b").max()))
    ):
        raise ValueError("fixed-direction prototype directions must be finite")

    bq = torch.as_tensor(base_quat, dtype=dtype, device=device)
    if bq.shape != (N, 4) or not bool(torch.isfinite(bq).all()):
        raise ValueError(f"base_quat must be finite (N,4), got {tuple(bq.shape)}")
    bq_norm = torch.linalg.norm(bq, dim=-1, keepdim=True)
    good_bq = torch.isfinite(bq_norm.squeeze(-1)) & (bq_norm.squeeze(-1) > _EPS) \
        & ((bq_norm.squeeze(-1) - 1.0).abs() <= 1.0e-3)
    if not bool(good_bq.all()):
        bad = torch.nonzero(~good_bq, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"base_quat must contain unit wxyz orientations; bad rows={bad[:8]}")
    rn = ref_normal.to(device=device, dtype=dtype)
    if rn.shape != (N, 3):
        raise ValueError(f"ref_normal must be (N,3) matching clip_ids, got {tuple(rn.shape)}")
    rn_norm = torch.linalg.norm(rn, dim=-1, keepdim=True)
    good_ref = torch.isfinite(rn).all(dim=-1) & torch.isfinite(rn_norm.squeeze(-1)) \
        & (rn_norm.squeeze(-1) > _EPS)
    if not bool(good_ref.all()):
        bad = torch.nonzero(~good_ref, as_tuple=False).flatten().tolist()
        raise ValueError(f"ref_normal must contain finite non-zero raw +Y faces; bad rows={bad[:8]}")
    return rn / rn_norm, bq / bq_norm


def _fixed_direction_replay(
    *,
    out: dict,
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    speed_min: torch.Tensor,
    speed_max: torch.Tensor,
    face_sign: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_top_z: float,
    tol_m: float,
    h: float,
    n_steps: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Replay a fixed-dir candidate through the scorer physics and return final acceptance.

    The adapter historically rebuilt a net top from ``surface_z`` and then added the ball radius
    again.  Runtime already passes ``net_top_z`` in the ball-centre convention.  This replay is the
    authority: exactly one explicit threshold, and the same rollout step/horizon as grading.
    """
    raw_n = out["n"]
    raw_n_norm = torch.linalg.norm(raw_n, dim=-1, keepdim=True)
    n_unit = raw_n / (raw_n_norm + _EPS)
    # ±n is the same collision plane, but NOT the same commanded raw +Y face.  Align only to the
    # runtime clip face; do not impose +x, which would silently flip a legitimate backhand.
    flip = torch.sum(n_unit * ref_normal, dim=-1, keepdim=True) < 0.0
    n_signed = torch.where(flip, -n_unit, n_unit)
    face_dot = torch.sum(n_signed * ref_normal, dim=-1)
    face_ok = (
        torch.isfinite(n_signed).all(dim=-1)
        & torch.isfinite(face_dot)
        & torch.isfinite(raw_n_norm.squeeze(-1))
        & (raw_n_norm.squeeze(-1) > _EPS)
        & (face_dot > 0.0)
    )

    physical_n = n_signed * face_sign[:, None]
    normal_speed = -torch.sum(
        (v_ball_in - out["v_r"]) * physical_n,
        dim=-1,
    )
    contact_ok = (
        torch.isfinite(normal_speed)
        & (normal_speed >= CONTACT_NORMAL_SPEED_MIN_MPS)
        & (normal_speed <= CONTACT_NORMAL_SPEED_MAX_MPS)
    )

    v_plus, w_plus = predict_paddle_contact(
        v_ball_in, out["v_r"], physical_n, w_ball_in, prm)
    land = coarse_landing(
        p_contact, v_plus, w_plus, prm, surface_z=surface_z, net_x=net_x,
        h=float(h), n_steps=int(n_steps),
    )
    resid_m = torch.linalg.norm(land["land_xy"] - aim_xy, dim=-1)
    speed = torch.linalg.norm(out["v_r"], dim=-1)
    land_ok = land["land_valid"] & torch.isfinite(land["land_xy"]).all(dim=-1)
    resid_ok = torch.isfinite(resid_m) & (resid_m < float(tol_m))
    speed_lo_ok = torch.isfinite(speed) & (speed >= speed_min - 1e-9)
    speed_hi_ok = torch.isfinite(speed) & (speed <= speed_max + 1e-9)
    # net_top_z is already the BALL-CENTRE clearance plane used by the runtime scorer.  No radius
    # or margin is added here.
    net_ok = (
        land["net_valid"]
        & torch.isfinite(land["net_z"])
        & (land["net_z"] > float(net_top_z))
    )
    good = (
        land_ok
        & resid_ok
        & speed_lo_ok
        & speed_hi_ok
        & net_ok
        & face_ok
        & contact_ok
    )

    reasons = torch.full_like(good, _R_NO_LANDING, dtype=torch.long)
    reasons = torch.where(land_ok, torch.full_like(reasons, _R_RESID), reasons)
    reasons = torch.where(
        land_ok & resid_ok & ~speed_hi_ok, torch.full_like(reasons, _R_SPEED_OVER), reasons)
    reasons = torch.where(
        land_ok & resid_ok & speed_hi_ok & ~speed_lo_ok,
        torch.full_like(reasons, _R_SPEED_UNDER), reasons)
    reasons = torch.where(
        land_ok & resid_ok & speed_lo_ok & speed_hi_ok & ~net_ok,
        torch.full_like(reasons, _R_NET), reasons)
    reasons = torch.where(
        land_ok & resid_ok & speed_lo_ok & speed_hi_ok & net_ok & ~face_ok,
        torch.full_like(reasons, _R_FACE), reasons)
    reasons = torch.where(
        land_ok
        & resid_ok
        & speed_lo_ok
        & speed_hi_ok
        & net_ok
        & face_ok
        & ~contact_ok,
        torch.full_like(reasons, _R_CONTACT_ENVELOPE),
        reasons,
    )
    reasons = torch.where(good, torch.full_like(reasons, -1), reasons)

    replayed = dict(out)
    replayed.update({
        "n": n_signed,
        "physical_n": physical_n,
        "contact_normal_speed_mps": normal_speed,
        "contact_envelope_ok": contact_ok,
        "speed": speed,
        "landing_xy": land["land_xy"],
        "land_valid": land["land_valid"],
        "resid_m": resid_m,
        "net_z": land["net_z"],
        "net_valid": land["net_valid"],
        "clears_net": net_ok,
        "ok": good,
        "reason": reasons,
    })
    return replayed, good, reasons


def _solve_fixed_direction_batch(
    *,
    clip_ids: torch.Tensor,
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    protos,
    base_quat: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_top_z: float,
    cfg: ContinuousQuestionCfg,
    h: float,
    n_steps: int,
    _diagnostic_prevalidated: bool = False,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """One fixed-action solve followed by the one authoritative scorer replay.

    Both the random producer and :func:`solve_proposals` call this exact function.  Keeping the
    adapter and replay here prevents an externally supplied curriculum proposal from acquiring a
    subtly different net, face, tolerance, or rollout contract than an internally drawn proposal.
    Inputs have already passed their caller's validation; this function neither samples nor
    modifies them.
    """
    device, dtype = p_contact.device, p_contact.dtype
    yaw = base_yaw_of(base_quat)
    d_m = _selected_direction_world(protos.v_hat_b, clip_ids, yaw, device, dtype)
    speed_min_m = protos.speed_min.to(device, dtype)[clip_ids]
    speed_max_m = torch.minimum(
        protos.speed_max.to(device, dtype)[clip_ids],
        torch.full_like(
            speed_min_m,
            float(cfg.speed_budget),
        ),
    )
    budget_infeasible = speed_min_m > speed_max_m
    if not _diagnostic_prevalidated and bool(
        budget_infeasible.any()
    ):
        bad = torch.nonzero(
            budget_infeasible, as_tuple=False
        ).flatten().tolist()
        raise ValueError(
            "fixed-direction global speed_budget is below the selected "
            f"prototype minimum for rows {bad[:8]}"
        )
    # A finite, positive, ordered source band remains attributable even when
    # its minimum lies above this run's global speed budget.  That relation is
    # ordinary construction infeasibility, not a malformed producer.  Give the
    # fixed-try numerical leaf a finite degenerate budget interval, then retain
    # the original minimum in the authoritative replay and force the ordinary
    # speed-under-min reason below.  The portable/formal entry keeps its legacy
    # fail-loud behavior through the branch above.
    solver_speed_min_m = torch.where(
        budget_infeasible, speed_max_m, speed_min_m
    )
    face_sign_m = protos.face_sign.to(device, dtype)[clip_ids]
    out = solve_strike_specs_fixed_dir(
        p_contact, v_ball_in, w_ball_in, aim_xy, d_m,
        solver_speed_min_m, speed_max_m,
        prm, surface_z=surface_z, net_x=net_x,
        # Map the adapter's legacy reconstructed threshold onto the scorer's explicit ball-centre
        # plane.  Its ok/reason are ignored; the replay below is the sole acceptance authority.
        net_height=float(net_top_z) - float(surface_z),
        net_margin_m=0.0, ball_radius=0.0,
        n_iters=int(cfg.n_iters), tol_m=float(cfg.tol_m),
        h=float(h), n_steps=int(n_steps),
        _diagnostic_fixed_try_lm_authority=(
            _DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY
            if _diagnostic_prevalidated
            else None
        ),
    )
    replayed, good, reasons = _fixed_direction_replay(
        out=out, p_contact=p_contact, v_ball_in=v_ball_in, w_ball_in=w_ball_in,
        aim_xy=aim_xy, ref_normal=ref_normal, speed_min=speed_min_m,
        speed_max=speed_max_m, face_sign=face_sign_m, prm=prm,
        surface_z=surface_z, net_x=net_x,
        net_top_z=float(net_top_z), tol_m=float(cfg.tol_m), h=float(h),
        n_steps=int(n_steps),
    )
    good = good & ~budget_infeasible
    reasons = torch.where(
        budget_infeasible,
        torch.full_like(reasons, _R_SPEED_UNDER),
        reasons,
    )
    replayed["ok"] = good
    replayed["reason"] = reasons
    return replayed, good, reasons


def _validate_external_proposals(
    *,
    clip_ids,
    p_contact,
    v_ball_in,
    w_ball_in,
    aim_xy,
    ref_normal,
    protos,
    base_quat,
    prm,
    surface_z,
    net_x,
    net_top_z,
    cfg,
    h,
    n_steps,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fail closed before an exact external proposal reaches the fixed-action solver."""
    if not isinstance(cfg, ContinuousQuestionCfg):
        raise TypeError(f"cfg must be ContinuousQuestionCfg, got {type(cfg).__name__}")
    if not cfg.fixed_direction:
        raise ValueError(
            "solve_proposals is fixed-direction only; free-direction proposals must use generate")
    if not isinstance(clip_ids, torch.Tensor) or clip_ids.ndim != 1:
        raise ValueError("clip_ids must be a rank-1 torch.Tensor")
    if clip_ids.dtype != torch.long:
        raise ValueError(f"clip_ids must have dtype torch.long, got {clip_ids.dtype}")
    N = int(clip_ids.shape[0])
    if N <= 0:
        raise ValueError("solve_proposals needs at least one proposal row")

    tensors = {
        "p_contact": (p_contact, (N, 3)),
        "v_ball_in": (v_ball_in, (N, 3)),
        "w_ball_in": (w_ball_in, (N, 3)),
        "aim_xy": (aim_xy, (N, 2)),
        "ref_normal": (ref_normal, (N, 3)),
        "base_quat": (base_quat, (N, 4)),
    }
    if not isinstance(p_contact, torch.Tensor) or not p_contact.dtype.is_floating_point:
        raise ValueError("p_contact must be a floating-point torch.Tensor")
    if p_contact.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"proposal dtype must be float32 or float64, got {p_contact.dtype}")
    device, dtype = p_contact.device, p_contact.dtype
    if clip_ids.device != device:
        raise ValueError("clip_ids and proposal tensors must be on the same device")
    for name, (value, shape) in tensors.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            got = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(f"{name} must have shape {shape}, got {got}")
        if value.device != device or value.dtype != dtype:
            raise ValueError(
                f"{name} must share p_contact device/dtype ({device}, {dtype}), "
                f"got ({value.device}, {value.dtype})")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    if protos is None:
        raise ValueError("protos is required")
    proto_fields = {
        "v_hat_b": (getattr(protos, "v_hat_b", None), 2),
        "speed_min": (getattr(protos, "speed_min", None), 1),
        "speed_max": (getattr(protos, "speed_max", None), 1),
    }
    for name, (value, rank) in proto_fields.items():
        if not isinstance(value, torch.Tensor) or value.ndim != rank:
            raise ValueError(f"protos.{name} must be a rank-{rank} torch.Tensor")
        if not value.dtype.is_floating_point:
            raise ValueError(f"protos.{name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"protos.{name} must contain only finite values")
    K = int(protos.v_hat_b.shape[0])
    if tuple(protos.v_hat_b.shape) != (K, 3) \
            or tuple(protos.speed_min.shape) != (K,) \
            or tuple(protos.speed_max.shape) != (K,) or K <= 0:
        raise ValueError("prototype direction/speed tables must have shapes (K,3), (K,), (K,)")
    dir_norm = torch.linalg.norm(protos.v_hat_b, dim=-1)
    if not bool((dir_norm > _EPS).all()):
        raise ValueError("protos.v_hat_b contains a zero direction")
    if not bool(((dir_norm - 1.0).abs() <= 1.0e-3).all()):
        raise ValueError("protos.v_hat_b directions must be unit length")
    if protos.speed_min.device != protos.speed_max.device \
            or protos.speed_min.dtype != protos.speed_max.dtype:
        raise ValueError("protos.speed_min and speed_max must share device/dtype")
    if not bool((protos.speed_min > 0.0).all()) \
            or not bool((protos.speed_max >= protos.speed_min).all()):
        raise ValueError("prototype speed bounds must satisfy 0 < speed_min <= speed_max")
    if bool((clip_ids < 0).any()) or bool((clip_ids >= K).any()):
        lo, hi = int(clip_ids.min()), int(clip_ids.max())
        raise ValueError(f"clip_ids out of range for K={K}: observed [{lo}, {hi}]")

    if isinstance(cfg.n_iters, bool) or int(cfg.n_iters) != cfg.n_iters \
            or int(cfg.n_iters) <= 0:
        raise ValueError(f"cfg.n_iters must be a positive integer, got {cfg.n_iters!r}")
    for name in (
        "k_d", "k_m", "g", "ball_radius", "inertia_coeff", "paddle_a_t",
        "paddle_b_t", "paddle_mu", "paddle_e_g1", "paddle_e_g2",
    ):
        if not hasattr(prm, name) or not math.isfinite(float(getattr(prm, name))):
            raise ValueError(f"prm.{name} must be finite")

    return _fixed_direction_contract(
        N=N, device=device, dtype=dtype, ref_normal=ref_normal, net_top_z=net_top_z,
        surface_z=surface_z, net_x=net_x, tol_m=cfg.tol_m, h=h, n_steps=n_steps,
        speed_budget=cfg.speed_budget, protos=protos, base_quat=base_quat,
    )


def _diagnostic_prevalidated_external_proposals(
    *,
    authority,
    clip_ids,
    p_contact,
    v_ball_in,
    w_ball_in,
    aim_xy,
    ref_normal,
    protos,
    base_quat,
    prm,
    surface_z,
    net_x,
    net_top_z,
    cfg,
    h,
    n_steps,
) -> tuple:
    """Prepare a producer-owned batch without observing dynamic tensor facts.

    Shape, dtype and device are host metadata and stay fail-loud here.  Every
    dynamic finite/unit/range predicate becomes one bit in the returned
    device-resident ``int64[P]`` producer-fault row.  Bad rows are replaced by
    finite deterministic solver inputs, but their original evidence remains in
    the ledger.  This helper never uses an asynchronous assertion as a delayed
    authorization check.

    The private capability protects only the portable diagnostic bypass.  The
    public device entry point supplies it internally and exposes the fault row
    for the future construction-bound authority to pack and CENSOR.
    """

    if authority is not _DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY:
        raise PermissionError(
            "diagnostic prevalidated solve requires the exact private authority"
        )
    if not isinstance(cfg, ContinuousQuestionCfg) or not cfg.fixed_direction:
        raise ValueError(
            "diagnostic prevalidated solve is fixed-direction only"
        )
    if (
        isinstance(cfg.n_iters, bool)
        or int(cfg.n_iters) != cfg.n_iters
        or int(cfg.n_iters) <= 0
    ):
        raise ValueError(
            f"cfg.n_iters must be a positive integer, got {cfg.n_iters!r}"
        )
    scalar_checks = {
        "surface_z": surface_z,
        "net_x": net_x,
        "net_top_z": net_top_z,
        "cfg.tol_m": cfg.tol_m,
        "cfg.speed_budget": cfg.speed_budget,
        "h": h,
    }
    for name, value in scalar_checks.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if float(net_top_z) <= float(surface_z):
        raise ValueError("net_top_z must be above surface_z")
    if float(cfg.tol_m) <= 0.0 or float(cfg.speed_budget) <= 0.0:
        raise ValueError(
            "cfg.tol_m and cfg.speed_budget must be positive"
        )
    if float(h) <= 0.0:
        raise ValueError("h must be positive")
    if (
        isinstance(n_steps, bool)
        or int(n_steps) != n_steps
        or int(n_steps) <= 0
    ):
        raise ValueError(
            f"n_steps must be a positive integer, got {n_steps!r}"
        )
    for name in (
        "k_d", "k_m", "g", "ball_radius", "inertia_coeff", "paddle_a_t",
        "paddle_b_t", "paddle_mu", "paddle_e_g1", "paddle_e_g2",
    ):
        if not hasattr(prm, name) or not math.isfinite(
            float(getattr(prm, name))
        ):
            raise ValueError(f"prm.{name} must be finite")
    if not isinstance(clip_ids, torch.Tensor) or clip_ids.ndim != 1:
        raise ValueError("clip_ids must be a rank-1 torch.Tensor")
    if clip_ids.dtype != torch.long:
        raise ValueError(
            f"clip_ids must have dtype torch.long, got {clip_ids.dtype}"
        )
    n_rows = int(clip_ids.shape[0])
    if n_rows <= 0:
        raise ValueError(
            "diagnostic prevalidated solve needs at least one proposal row"
        )
    if (
        not isinstance(p_contact, torch.Tensor)
        or not p_contact.dtype.is_floating_point
        or p_contact.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError(
            "p_contact must be a float32/float64 torch.Tensor"
        )
    device, dtype = p_contact.device, p_contact.dtype
    tensors = {
        "p_contact": (p_contact, (n_rows, 3)),
        "v_ball_in": (v_ball_in, (n_rows, 3)),
        "w_ball_in": (w_ball_in, (n_rows, 3)),
        "aim_xy": (aim_xy, (n_rows, 2)),
        "ref_normal": (ref_normal, (n_rows, 3)),
        "base_quat": (base_quat, (n_rows, 4)),
    }
    if clip_ids.device != device:
        raise ValueError(
            "clip_ids and diagnostic proposal tensors must share a device"
        )
    for name, (value, shape) in tensors.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            got = (
                None
                if not isinstance(value, torch.Tensor)
                else tuple(value.shape)
            )
            raise ValueError(f"{name} must have shape {shape}, got {got}")
        if value.device != device or value.dtype != dtype:
            raise ValueError(
                f"{name} must share p_contact device/dtype "
                f"({device}, {dtype}), got ({value.device}, {value.dtype})"
            )
    if protos is None:
        raise ValueError("diagnostic prevalidated solve requires prototypes")
    proto_fields = {
        "v_hat_b": (getattr(protos, "v_hat_b", None), 2),
        "speed_min": (getattr(protos, "speed_min", None), 1),
        "speed_max": (getattr(protos, "speed_max", None), 1),
        "face_sign": (getattr(protos, "face_sign", None), 1),
    }
    for name, (value, rank) in proto_fields.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != rank
            or not value.dtype.is_floating_point
        ):
            raise ValueError(
                f"protos.{name} must be a rank-{rank} floating tensor"
            )
    action_count = int(protos.v_hat_b.shape[0])
    if (
        action_count <= 0
        or tuple(protos.v_hat_b.shape) != (action_count, 3)
        or tuple(protos.speed_min.shape) != (action_count,)
        or tuple(protos.speed_max.shape) != (action_count,)
        or tuple(protos.face_sign.shape) != (action_count,)
    ):
        raise ValueError(
            "prototype tables must have aligned (K,3)/(K,) shapes"
        )
    base = base_quat
    reference = ref_normal
    proto_direction = protos.v_hat_b.to(device=device, dtype=dtype)
    proto_speed_min = protos.speed_min.to(device=device, dtype=dtype)
    proto_speed_max = protos.speed_max.to(device=device, dtype=dtype)
    proto_face_sign = protos.face_sign.to(device=device, dtype=dtype)
    base_norm = torch.linalg.norm(base, dim=-1)
    reference_norm = torch.linalg.norm(reference, dim=-1)
    proto_direction_norm = torch.linalg.norm(proto_direction, dim=-1)

    proposal_valid = (
        torch.isfinite(p_contact).all(dim=-1)
        & torch.isfinite(v_ball_in).all(dim=-1)
        & torch.isfinite(w_ball_in).all(dim=-1)
        & torch.isfinite(aim_xy).all(dim=-1)
    )
    reference_valid = (
        torch.isfinite(reference).all(dim=-1)
        & torch.isfinite(reference_norm)
        & (reference_norm > _EPS)
    )
    base_valid = (
        torch.isfinite(base).all(dim=-1)
        & torch.isfinite(base_norm)
        & (base_norm > _EPS)
        & ((base_norm - 1.0).abs() <= 1.0e-3)
    )
    action_valid = (clip_ids >= 0) & (clip_ids < action_count)
    safe_clip_ids = clip_ids.clamp(min=0, max=action_count - 1)

    direction_valid = (
        torch.isfinite(proto_direction).all(dim=-1)
        & torch.isfinite(proto_direction_norm)
        & (proto_direction_norm > _EPS)
        & ((proto_direction_norm - 1.0).abs() <= 1.0e-3)
    )
    speed_valid = (
        torch.isfinite(proto_speed_min)
        & torch.isfinite(proto_speed_max)
        & (proto_speed_min > 0.0)
        & (proto_speed_max >= proto_speed_min)
    )
    face_valid = (
        torch.isfinite(proto_face_sign)
        & ((proto_face_sign == 1.0) | (proto_face_sign == -1.0))
    )
    selected_direction_valid = action_valid & direction_valid[safe_clip_ids]
    selected_speed_valid = action_valid & speed_valid[safe_clip_ids]
    selected_face_valid = action_valid & face_valid[safe_clip_ids]

    producer_fault_bits = torch.zeros_like(clip_ids, dtype=torch.long)
    producer_fault_bits = torch.where(
        proposal_valid,
        producer_fault_bits,
        producer_fault_bits | PRODUCER_FAULT_NONFINITE_PROPOSAL,
    )
    producer_fault_bits = torch.where(
        reference_valid,
        producer_fault_bits,
        producer_fault_bits | PRODUCER_FAULT_REFERENCE_NORMAL,
    )
    producer_fault_bits = torch.where(
        base_valid,
        producer_fault_bits,
        producer_fault_bits | PRODUCER_FAULT_BASE_QUATERNION,
    )
    producer_fault_bits = torch.where(
        action_valid,
        producer_fault_bits,
        producer_fault_bits | PRODUCER_FAULT_ACTION_RANGE,
    )
    producer_fault_bits = torch.where(
        selected_direction_valid | ~action_valid,
        producer_fault_bits,
        producer_fault_bits | PRODUCER_FAULT_PROTOTYPE_DIRECTION,
    )
    producer_fault_bits = torch.where(
        selected_speed_valid | ~action_valid,
        producer_fault_bits,
        producer_fault_bits | PRODUCER_FAULT_PROTOTYPE_SPEED,
    )
    producer_fault_bits = torch.where(
        selected_face_valid | ~action_valid,
        producer_fault_bits,
        producer_fault_bits | PRODUCER_FAULT_PROTOTYPE_FACE_SIGN,
    )

    fallback_reference = torch.zeros_like(reference)
    fallback_reference[:, 0] = 1.0
    safe_reference = torch.where(
        reference_valid.unsqueeze(-1), reference, fallback_reference
    )
    ref_unit = safe_reference / torch.linalg.norm(
        safe_reference, dim=-1, keepdim=True
    )
    fallback_base = torch.zeros_like(base)
    fallback_base[:, 0] = 1.0
    safe_base = torch.where(base_valid.unsqueeze(-1), base, fallback_base)
    base_unit = safe_base / torch.linalg.norm(
        safe_base, dim=-1, keepdim=True
    )

    fallback_direction = torch.zeros_like(proto_direction)
    fallback_direction[:, 0] = 1.0
    safe_direction = torch.where(
        direction_valid.unsqueeze(-1), proto_direction, fallback_direction
    )
    fallback_speed_min = torch.full_like(
        proto_speed_min, 0.5 * float(cfg.speed_budget)
    )
    fallback_speed_max = torch.full_like(
        proto_speed_max, float(cfg.speed_budget)
    )
    safe_protos = SimpleNamespace(
        v_hat_b=safe_direction,
        speed_min=torch.where(
            speed_valid, proto_speed_min, fallback_speed_min
        ),
        speed_max=torch.where(
            speed_valid, proto_speed_max, fallback_speed_max
        ),
        face_sign=torch.where(
            face_valid, proto_face_sign, torch.ones_like(proto_face_sign)
        ),
    )

    fault_free = producer_fault_bits == 0
    fallback_p_contact = torch.zeros_like(p_contact)
    fallback_p_contact[:, 0] = float(net_x) - 0.8
    fallback_p_contact[:, 2] = float(surface_z) + 0.2
    fallback_v_ball_in = torch.zeros_like(v_ball_in)
    fallback_v_ball_in[:, 0] = -3.0
    fallback_aim_xy = torch.zeros_like(aim_xy)
    fallback_aim_xy[:, 0] = float(net_x) + 0.6
    safe_p_contact = torch.where(
        fault_free.unsqueeze(-1), p_contact, fallback_p_contact
    )
    safe_v_ball_in = torch.where(
        fault_free.unsqueeze(-1), v_ball_in, fallback_v_ball_in
    )
    safe_w_ball_in = torch.where(
        fault_free.unsqueeze(-1), w_ball_in, torch.zeros_like(w_ball_in)
    )
    safe_aim_xy = torch.where(
        fault_free.unsqueeze(-1), aim_xy, fallback_aim_xy
    )
    return (
        safe_clip_ids,
        safe_p_contact,
        safe_v_ball_in,
        safe_w_ball_in,
        safe_aim_xy,
        ref_unit,
        base_unit,
        safe_protos,
        producer_fault_bits,
    )


@torch.no_grad()
def solve_proposals_device(
    clip_ids: torch.Tensor,
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    *,
    protos,
    base_quat: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_top_z: float,
    cfg: ContinuousQuestionCfg,
    h: float = 0.01,
    n_steps: int = 100,
) -> DeviceProposalSolveResult:
    """Solve exact external proposals without observing a tensor on the host.

    ``P == N`` and each input row is solved once.  Shape/dtype/device mistakes
    fail before the solver.  Dynamic tensor predicates are encoded in
    ``producer_fault_bits`` without a host observation or delayed assertion.
    The eventual construction-bound consumer must include that exact row in the
    sole packed reveal boundary and CENSOR every nonzero row before installing
    any task or ball.

    This function deliberately uses the bit-exact fixed-try LM implementation:
    unlike the portable historical loop it contains no data-dependent Python
    ``bool(tensor)`` break.  It never calls ``_build_proposal_host_packet`` and
    never materializes a Python reason/count/identity object.
    """
    (
        safe_clip_ids,
        safe_p_contact,
        safe_v_ball_in,
        safe_w_ball_in,
        safe_aim_xy,
        ref_unit,
        base_unit,
        safe_protos,
        producer_fault_bits,
    ) = _diagnostic_prevalidated_external_proposals(
        authority=_DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY,
        clip_ids=clip_ids,
        p_contact=p_contact,
        v_ball_in=v_ball_in,
        w_ball_in=w_ball_in,
        aim_xy=aim_xy,
        ref_normal=ref_normal,
        protos=protos,
        base_quat=base_quat,
        prm=prm,
        surface_z=surface_z,
        net_x=net_x,
        net_top_z=net_top_z,
        cfg=cfg,
        h=h,
        n_steps=n_steps,
    )
    out, good, reasons = _solve_fixed_direction_batch(
        clip_ids=safe_clip_ids,
        p_contact=safe_p_contact,
        v_ball_in=safe_v_ball_in,
        w_ball_in=safe_w_ball_in,
        aim_xy=safe_aim_xy,
        ref_normal=ref_unit,
        protos=safe_protos,
        base_quat=base_unit,
        prm=prm,
        surface_z=surface_z,
        net_x=net_x,
        net_top_z=net_top_z,
        cfg=cfg,
        h=h,
        n_steps=n_steps,
        _diagnostic_prevalidated=True,
    )

    row_count = int(clip_ids.shape[0])
    fault_free = producer_fault_bits == 0
    good = good & fault_free
    reasons = torch.where(
        fault_free, reasons, torch.full_like(reasons, _R_NO_LANDING)
    )
    installable = good.unsqueeze(-1)
    p_out = torch.where(
        installable, p_contact, torch.full_like(p_contact, float("nan"))
    )
    v_in_out = torch.where(
        installable, v_ball_in, torch.full_like(v_ball_in, float("nan"))
    )
    w_in_out = torch.where(
        installable, w_ball_in, torch.full_like(w_ball_in, float("nan"))
    )
    aim_out = torch.where(
        installable, aim_xy, torch.full_like(aim_xy, float("nan"))
    )
    v_r_out = torch.where(
        installable, out["v_r"], torch.full_like(out["v_r"], float("nan"))
    )
    n_r_out = torch.where(
        installable, out["n"], torch.full_like(out["n"], float("nan"))
    )
    residual = torch.where(
        fault_free,
        torch.nan_to_num(out["resid_m"], nan=float("inf")),
        torch.full_like(out["resid_m"], float("inf")),
    ).clone()
    ledger = ProposalLedger(
        request_index=torch.arange(
            row_count, dtype=torch.long, device=p_contact.device
        ),
        clip_id=clip_ids.clone(),
        round_index=torch.ones(
            row_count, dtype=torch.long, device=p_contact.device
        ),
        p_contact=p_contact.clone(),
        v_ball_in=v_ball_in.clone(),
        w_ball_in=w_ball_in.clone(),
        aim_xy=aim_xy.clone(),
        reason_code=reasons.clone(),
        admitted=good.clone(),
        resid_m=residual.clone(),
        ref_normal=ref_normal.clone(),
        base_quat=base_quat.clone(),
    )
    return DeviceProposalSolveResult(
        p_contact=p_out,
        v_racket=v_r_out,
        n_racket=n_r_out,
        v_ball_in=v_in_out,
        w_ball_in=w_in_out,
        aim_xy=aim_out,
        ok=good.clone(),
        resid_m=residual,
        attempted_v_ball_in=v_ball_in.clone(),
        producer_fault_bits=producer_fault_bits.clone(),
        proposals=ledger,
    )


@torch.no_grad()
def solve_proposals(
    clip_ids: torch.Tensor,
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    *,
    protos,
    base_quat: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_top_z: float,
    cfg: ContinuousQuestionCfg,
    h: float = 0.01,
    n_steps: int = 100,
    _diagnostic_prevalidated_authority=None,
) -> QuestionDrawResult:
    """Portable result boundary with a separate diagnostic device core.

    The historical validation branch stays first so public exception ordering
    and serial-LM numerical behavior stay unchanged.  The private diagnostic
    path delegates to :func:`solve_proposals_device`.  Only this portable
    boundary constructs Python reason counts and the immutable one-transfer
    host packet.
    """
    if _diagnostic_prevalidated_authority is None:
        ref_unit, base_unit = _validate_external_proposals(
            clip_ids=clip_ids,
            p_contact=p_contact,
            v_ball_in=v_ball_in,
            w_ball_in=w_ball_in,
            aim_xy=aim_xy,
            ref_normal=ref_normal,
            protos=protos,
            base_quat=base_quat,
            prm=prm,
            surface_z=surface_z,
            net_x=net_x,
            net_top_z=net_top_z,
            cfg=cfg,
            h=h,
            n_steps=n_steps,
        )
        # Preserve the formal/public numerical algorithm.  The device core's
        # fixed-try LM is diagnostic-only; this branch keeps the historical
        # serial LM/fallback and its fail-loud speed-budget contract.
        out, good, reasons = _solve_fixed_direction_batch(
            clip_ids=clip_ids,
            p_contact=p_contact,
            v_ball_in=v_ball_in,
            w_ball_in=w_ball_in,
            aim_xy=aim_xy,
            ref_normal=ref_unit,
            protos=protos,
            base_quat=base_unit,
            prm=prm,
            surface_z=surface_z,
            net_x=net_x,
            net_top_z=net_top_z,
            cfg=cfg,
            h=h,
            n_steps=n_steps,
            _diagnostic_prevalidated=False,
        )
        row_count = int(clip_ids.shape[0])
        nan = float("nan")
        installable = good.unsqueeze(-1)
        p_out = torch.where(
            installable, p_contact, torch.full_like(p_contact, nan)
        )
        v_in_out = torch.where(
            installable, v_ball_in, torch.full_like(v_ball_in, nan)
        )
        w_in_out = torch.where(
            installable, w_ball_in, torch.full_like(w_ball_in, nan)
        )
        aim_out = torch.where(
            installable, aim_xy, torch.full_like(aim_xy, nan)
        )
        v_r_out = torch.where(
            installable, out["v_r"], torch.full_like(out["v_r"], nan)
        )
        n_r_out = torch.where(
            installable, out["n"], torch.full_like(out["n"], nan)
        )
        residual = torch.nan_to_num(
            out["resid_m"], nan=float("inf")
        ).clone()
        ledger = ProposalLedger(
            request_index=torch.arange(
                row_count, dtype=torch.long, device=p_contact.device
            ),
            clip_id=clip_ids.clone(),
            round_index=torch.ones(
                row_count, dtype=torch.long, device=p_contact.device
            ),
            p_contact=p_contact.clone(),
            v_ball_in=v_ball_in.clone(),
            w_ball_in=w_ball_in.clone(),
            aim_xy=aim_xy.clone(),
            reason_code=reasons.clone(),
            admitted=good.clone(),
            resid_m=residual.clone(),
            ref_normal=ref_normal.clone(),
            base_quat=base_quat.clone(),
        )
        host_packet = _build_proposal_host_packet(
            reason_codes=reasons,
            admitted=good,
            producer_fault_bits=torch.zeros_like(
                reasons, dtype=torch.long
            ),
            racket_velocity=v_r_out,
            racket_normal=n_r_out,
            residual=residual,
        )
        counts: dict = {}
        for is_admitted, code in zip(
            host_packet.admitted, host_packet.reason_codes
        ):
            if is_admitted:
                continue
            name = (
                _CONTINUOUS_REASONS[code]
                if 0 <= code < len(_CONTINUOUS_REASONS)
                else "unsolved"
            )
            counts[name] = counts.get(name, 0) + 1
        return QuestionDrawResult(
            p_contact=p_out,
            v_racket=v_r_out,
            n_racket=n_r_out,
            v_ball_in=v_in_out,
            w_ball_in=w_in_out,
            aim_xy=aim_out,
            ok=good.clone(),
            resid_m=residual,
            attempted_v_ball_in=v_ball_in.clone(),
            rounds_used=1,
            exhausted=sum(not value for value in host_packet.admitted),
            reason_counts=counts,
            proposal_count=row_count,
            proposals=ledger,
            proposal_host_packet=host_packet,
        )
    else:
        _diagnostic_prevalidated_external_proposals(
            authority=_diagnostic_prevalidated_authority,
            clip_ids=clip_ids,
            p_contact=p_contact,
            v_ball_in=v_ball_in,
            w_ball_in=w_ball_in,
            aim_xy=aim_xy,
            ref_normal=ref_normal,
            protos=protos,
            base_quat=base_quat,
            prm=prm,
            surface_z=surface_z,
            net_x=net_x,
            net_top_z=net_top_z,
            cfg=cfg,
            h=h,
            n_steps=n_steps,
        )

    device_result = solve_proposals_device(
        clip_ids,
        p_contact,
        v_ball_in,
        w_ball_in,
        aim_xy,
        ref_normal,
        protos=protos,
        base_quat=base_quat,
        prm=prm,
        surface_z=surface_z,
        net_x=net_x,
        net_top_z=net_top_z,
        cfg=cfg,
        h=h,
        n_steps=n_steps,
    )
    host_packet = _build_proposal_host_packet(
        reason_codes=device_result.proposals.reason_code,
        admitted=device_result.ok,
        producer_fault_bits=device_result.producer_fault_bits,
        racket_velocity=device_result.v_racket,
        racket_normal=device_result.n_racket,
        residual=device_result.resid_m,
    )
    counts: dict = {}
    for is_admitted, code in zip(
        host_packet.admitted, host_packet.reason_codes
    ):
        if is_admitted:
            continue
        name = (
            _CONTINUOUS_REASONS[code]
            if 0 <= code < len(_CONTINUOUS_REASONS)
            else "unsolved"
        )
        counts[name] = counts.get(name, 0) + 1

    return QuestionDrawResult(
        p_contact=device_result.p_contact,
        v_racket=device_result.v_racket,
        n_racket=device_result.n_racket,
        v_ball_in=device_result.v_ball_in,
        w_ball_in=device_result.w_ball_in,
        aim_xy=device_result.aim_xy,
        ok=device_result.ok,
        resid_m=device_result.resid_m,
        attempted_v_ball_in=device_result.attempted_v_ball_in,
        rounds_used=1,
        exhausted=sum(not value for value in host_packet.admitted),
        reason_counts=counts,
        proposal_count=int(clip_ids.shape[0]),
        proposals=device_result.proposals,
        proposal_host_packet=host_packet,
    )


@torch.no_grad()
def _solve_proposals_diagnostic_host_only(
    clip_ids: torch.Tensor,
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    *,
    protos,
    base_quat: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_top_z: float,
    cfg: ContinuousQuestionCfg,
    h: float = 0.01,
    n_steps: int = 100,
    _diagnostic_prevalidated_authority=None,
) -> tuple[ProposalHostPacket, dict]:
    """Return only the immutable host values consumed by diagnostic refill.

    This private path deliberately shares the prevalidated input proof, fixed-direction solver,
    scorer replay and host-packet encoder with :func:`solve_proposals`.  It omits the six
    installable/attempt tensors, lossless :class:`ProposalLedger`, and :class:`QuestionDrawResult`
    that the diagnostic refill caller never reads.  The public/formal API remains unchanged.
    """

    (
        safe_clip_ids,
        safe_p_contact,
        safe_v_ball_in,
        safe_w_ball_in,
        safe_aim_xy,
        ref_unit,
        base_unit,
        safe_protos,
        producer_fault_bits,
    ) = _diagnostic_prevalidated_external_proposals(
        authority=_diagnostic_prevalidated_authority,
        clip_ids=clip_ids,
        p_contact=p_contact,
        v_ball_in=v_ball_in,
        w_ball_in=w_ball_in,
        aim_xy=aim_xy,
        ref_normal=ref_normal,
        protos=protos,
        base_quat=base_quat,
        prm=prm,
        surface_z=surface_z,
        net_x=net_x,
        net_top_z=net_top_z,
        cfg=cfg,
        h=h,
        n_steps=n_steps,
    )
    out, good, reasons = _solve_fixed_direction_batch(
        clip_ids=safe_clip_ids,
        p_contact=safe_p_contact,
        v_ball_in=safe_v_ball_in,
        w_ball_in=safe_w_ball_in,
        aim_xy=safe_aim_xy,
        ref_normal=ref_unit,
        protos=safe_protos,
        base_quat=base_unit,
        prm=prm,
        surface_z=surface_z,
        net_x=net_x,
        net_top_z=net_top_z,
        cfg=cfg,
        h=h,
        n_steps=n_steps,
        _diagnostic_prevalidated=True,
    )

    fault_free = producer_fault_bits == 0
    good = good & fault_free
    reasons = torch.where(
        fault_free, reasons, torch.full_like(reasons, _R_NO_LANDING)
    )

    row_count, device, dtype = (
        int(clip_ids.shape[0]),
        p_contact.device,
        p_contact.dtype,
    )
    nan = float("nan")
    # Match the public packet's rejected-row representation exactly, but materialize only the two
    # geometric tensors that actually cross the host boundary.
    v_r_out = torch.full(
        (row_count, 3), nan, device=device, dtype=dtype
    )
    n_r_out = torch.full(
        (row_count, 3), nan, device=device, dtype=dtype
    )
    v_r_out[good] = out["v_r"][good]
    n_r_out[good] = out["n"][good]
    residual = torch.where(
        fault_free,
        torch.nan_to_num(out["resid_m"], nan=float("inf")),
        torch.full_like(out["resid_m"], float("inf")),
    ).clone()
    host_packet = _build_proposal_host_packet(
        reason_codes=reasons,
        admitted=good,
        producer_fault_bits=producer_fault_bits,
        racket_velocity=v_r_out,
        racket_normal=n_r_out,
        residual=residual,
    )
    reason_counts: dict = {}
    for is_admitted, code in zip(
        host_packet.admitted, host_packet.reason_codes
    ):
        if is_admitted:
            continue
        name = (
            _CONTINUOUS_REASONS[code]
            if 0 <= code < len(_CONTINUOUS_REASONS)
            else "unsolved"
        )
        reason_counts[name] = reason_counts.get(name, 0) + 1
    return host_packet, reason_counts


@torch.no_grad()
def generate(
    clip_ids: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    cfg: ContinuousQuestionCfg,
    ref_normal: torch.Tensor | None = None,
    net_top_z: float | None = None,
    protos=None,
    base_quat=None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    h: float = 0.01,
    n_steps: int = 100,
) -> QuestionDrawResult:
    """Draw N balls continuously and solve each one's demanded racket state. ALL ENV-LOCAL.

    ``clip_ids`` (N,) selects the per-clip regime and supplies device. ``ref_normal`` (N,3) is the
    face the answer's sign is matched to — pass the RUNTIME raw +Y clip face; without it the
    solver's arbitrary "+x opponent-facing" seed convention wins and, on a backhand clip whose
    clip normal has x < 0, EVERY row comes back with the face opposite the clip face (measured:
    763/763 and 746/746 rows on the two shipped banks). That is the M3c 单翻病 by construction.
    ``net_top_z`` (env-local z of the net top, ball radius included) turns on the net-clearance
    rejection the bank generator has always applied; None = no net test (byte-identical to the
    historical free-solve behaviour). ``h``/``n_steps`` are the rollout parameters — pass the
    SCORER's own (``vb_rollout_h`` / ``vb_rollout_steps``) so the solved answer and the graded
    answer cannot drift apart by coincidence; the defaults only match today by luck.

    COST: one call is ``n_iters * (5 probes + up to 4 trial steps)`` batched rollouts of
    ``coarse_landing`` — kernel-launch bound, so the wall cost is per-CALL not per-env (measured:
    48 rows cost 63% of what 8192 rows cost). Measure with ``scripts/bench_continuous_questions.py``
    before raising ``n_iters``; the fallback when the budget is blown is to make the call RARER
    (buffer more rows per call), then to lower ``n_iters`` (accuracy degrades visibly through
    ``resid_m``), NEVER to substitute an unsolved target.
    """
    device = clip_ids.device
    N = int(clip_ids.shape[0])
    gen = generator

    pos_box = (_rows(cfg.pos_range_per_clip, clip_ids, device, dtype)
               if cfg.pos_range_per_clip is not None
               else torch.as_tensor(cfg.pos_range, dtype=dtype, device=device)[None].expand(N, 3, 2))
    vel_box = (_rows(cfg.vel_range_per_clip, clip_ids, device, dtype)
               if cfg.vel_range_per_clip is not None
               else torch.as_tensor(cfg.vel_range, dtype=dtype, device=device)[None].expand(N, 3, 2))
    spin_max = (_rows(cfg.spin_abs_max_per_clip, clip_ids, device, dtype)
                if cfg.spin_abs_max_per_clip is not None
                else torch.full((N,), float(cfg.spin_abs_max), device=device, dtype=dtype))
    if ref_normal is not None:
        ref_normal = ref_normal.to(device=device, dtype=dtype)
        if ref_normal.shape != (N, 3):
            raise ValueError(f"ref_normal must be (N,3) matching clip_ids, got {tuple(ref_normal.shape)}")
    fixed_base_quat = None
    if cfg.fixed_direction:
        ref_normal, fixed_base_quat = _fixed_direction_contract(
            N=N, device=device, dtype=dtype, ref_normal=ref_normal, net_top_z=net_top_z,
            surface_z=surface_z, net_x=net_x, tol_m=cfg.tol_m, h=h, n_steps=n_steps,
            speed_budget=cfg.speed_budget, protos=protos, base_quat=base_quat,
        )

    nan = float("nan")
    p = torch.full((N, 3), nan, device=device, dtype=dtype)
    v_in = torch.full((N, 3), nan, device=device, dtype=dtype)
    w_in = torch.full((N, 3), nan, device=device, dtype=dtype)
    aim = torch.full((N, 2), nan, device=device, dtype=dtype)
    v_r = torch.full((N, 3), nan, device=device, dtype=dtype)
    n_r = torch.full((N, 3), nan, device=device, dtype=dtype)
    resid = torch.full((N,), float("inf"), device=device, dtype=dtype)
    v_att = torch.zeros(N, 3, device=device, dtype=dtype)
    ok = torch.zeros(N, dtype=torch.bool, device=device)
    counts: dict = {}
    rounds_used = 0
    proposal_parts = {
        "request_index": [], "clip_id": [], "round_index": [], "p_contact": [],
        "v_ball_in": [], "w_ball_in": [], "aim_xy": [], "reason_code": [],
        "admitted": [], "resid_m": [],
    }

    todo = torch.arange(N, device=device)
    for _round in range(1, int(cfg.max_redraw_rounds) + 1):
        if todo.numel() == 0:
            break
        rounds_used = _round                      # assigned AFTER the break: no off-by-one
        m = todo.numel()
        p_m = _uniform_box(pos_box[todo], gen, device, dtype)
        v_m = _uniform_box(vel_box[todo], gen, device, dtype)
        s_m = spin_max[todo].unsqueeze(-1)
        w_m = (torch.rand(m, 3, device=device, dtype=dtype, generator=gen) * 2.0 - 1.0) * s_m
        ax = torch.rand(m, device=device, dtype=dtype, generator=gen)
        ay = torch.rand(m, device=device, dtype=dtype, generator=gen)
        aim_m = torch.stack([
            cfg.aim_x_range[0] + (cfg.aim_x_range[1] - cfg.aim_x_range[0]) * ax,
            cfg.aim_y_range[0] + (cfg.aim_y_range[1] - cfg.aim_y_range[0]) * ay,
        ], dim=-1)

        if cfg.fixed_direction:
            out, good, reasons = _solve_fixed_direction_batch(
                clip_ids=clip_ids[todo], p_contact=p_m, v_ball_in=v_m, w_ball_in=w_m,
                aim_xy=aim_m, ref_normal=ref_normal[todo], protos=protos,
                base_quat=fixed_base_quat[todo], prm=prm, surface_z=surface_z,
                net_x=net_x, net_top_z=float(net_top_z), cfg=cfg, h=h,
                n_steps=n_steps,
            )
        else:
            out = solve_strike_specs(
                p_m, v_m, w_m, aim_m, prm, surface_z=surface_z, net_x=net_x,
                ref_normal=(None if ref_normal is None else ref_normal[todo]),
                speed_budget=float(cfg.speed_budget), n_iters=int(cfg.n_iters),
                tol_m=float(cfg.tol_m), h=float(h), n_steps=int(n_steps),
            )
            good = out["ok"]
            # NET CLEARANCE. solve_strike_specs deliberately leaves the net out of ``ok`` (its
            # historical callers never tested it); the bank generator rejects and COUNTS it, so
            # the continuous path must too or it trains on returns that clip the net.
            if net_top_z is not None:
                net_ok = out["net_valid"] & (out["net_z"] > float(net_top_z))
                good = good & net_ok
            else:
                net_ok = torch.ones_like(good)
            # REAL reason codes. This used to hard-code code 1 for every failure, so a no-landing
            # and a speed-cap rejection both surfaced as "resid_gt_tol" and every histogram anyone
            # ever printed was fiction. Same expressions/order as solve_strike_specs_fixed_dir.
            valid = out["land_valid"]
            resid_ok = out["resid_m"] < float(cfg.tol_m)
            speed_ok = out["speed"] <= float(cfg.speed_budget) + 1e-9
            reasons = torch.full_like(good, _R_NO_LANDING, dtype=torch.long)
            reasons = torch.where(valid, torch.full_like(reasons, _R_RESID), reasons)
            reasons = torch.where(valid & resid_ok & ~speed_ok,
                                  torch.full_like(reasons, _R_SPEED_OVER), reasons)
            reasons = torch.where(valid & resid_ok & speed_ok & ~net_ok,
                                  torch.full_like(reasons, _R_NET), reasons)
            reasons = torch.where(good, torch.full_like(reasons, -1), reasons)

        proposal_parts["request_index"].append(todo.clone())
        proposal_parts["clip_id"].append(clip_ids[todo].clone())
        proposal_parts["round_index"].append(
            torch.full((m,), _round, dtype=torch.long, device=device))
        proposal_parts["p_contact"].append(p_m.clone())
        proposal_parts["v_ball_in"].append(v_m.clone())
        proposal_parts["w_ball_in"].append(w_m.clone())
        proposal_parts["aim_xy"].append(aim_m.clone())
        proposal_parts["reason_code"].append(reasons.clone())
        proposal_parts["admitted"].append(good.clone())
        proposal_parts["resid_m"].append(
            torch.nan_to_num(out["resid_m"], nan=float("inf")).clone())

        v_att[todo] = v_m
        sel = todo[good]
        p[sel], v_in[sel], w_in[sel], aim[sel] = p_m[good], v_m[good], w_m[good], aim_m[good]
        v_r[sel], n_r[sel] = out["v_r"][good], out["n"][good]
        resid[sel] = out["resid_m"][good]
        ok[sel] = True
        bad = ~good
        if bool(bad.any()):
            # The unsolved rows keep their (finite) residual for diagnosis and STAY NaN everywhere
            # else — see the class docstring: not installable, structurally.
            resid[todo[bad]] = torch.nan_to_num(out["resid_m"][bad], nan=float("inf"))
            for code in reasons[bad].tolist():
                name = (
                    _CONTINUOUS_REASONS[code]
                    if 0 <= code < len(_CONTINUOUS_REASONS)
                    else "unsolved"
                )
                counts[name] = counts.get(name, 0) + 1
        todo = todo[bad]

    if proposal_parts["clip_id"]:
        proposal_ledger = ProposalLedger(**{
            name: torch.cat(parts, dim=0) for name, parts in proposal_parts.items()
        })
    else:
        proposal_ledger = _empty_proposal_ledger(device, dtype)
    proposal_count = len(proposal_ledger)

    return QuestionDrawResult(
        p_contact=p, v_racket=v_r, n_racket=n_r, v_ball_in=v_in, w_ball_in=w_in, aim_xy=aim,
        ok=ok, resid_m=resid, attempted_v_ball_in=v_att, rounds_used=rounds_used,
        exhausted=int((~ok).sum()),
        reason_counts=counts,
        proposal_count=proposal_count,
        proposals=proposal_ledger,
    )


@torch.no_grad()
def parity_report(
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    bank_vel: torch.Tensor,
    bank_normal: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    net_top_z: float,
    speed_budget: float = 4.0,
    tol_m: float = 0.005,
    n_iters: int = 12,
    h: float = 0.01,
    n_steps: int = 100,
    land_tol_m: float = 0.02,
    aim_tol_m: float = 0.006,
    speed_tol_mps: float = 0.10,
    min_coverage: float = 0.90,
) -> dict:
    """Push a BANK ROW's ball through the continuous solver and compare PHYSICAL CONSEQUENCES.

    人话:"两条路同物理"能断言的形式是**后果**(落点/过网/拍面在哪一侧/速度模长),不是 demanded
    state 逐位相等——5 自由度解流形在共享速度正则下不是逐点唯一的,拍面转 4 度、速度补一点,落点
    一样。写一句逐位相等的断言等于写一句在某个容差下碰巧能过的假话。

    Returns ``{"failures": [str, ...], "stats": {...}}``. Callers assert ``failures == []``. Used by
    BOTH ``tests/test_continuous_vs_bank_parity.py`` and the boot anchor gate, so the two can never
    drift apart.
    """
    dev, dt = p_contact.device, p_contact.dtype
    out = solve_strike_specs(
        p_contact, v_ball_in, w_ball_in, aim_xy, prm, surface_z=surface_z, net_x=net_x,
        ref_normal=ref_normal, speed_budget=float(speed_budget), n_iters=int(n_iters),
        tol_m=float(tol_m), h=float(h), n_steps=int(n_steps),
    )
    ok = out["ok"]
    n = int(p_contact.shape[0])
    fails: list = []
    cov = float(ok.float().mean()) if n else 0.0
    if cov < float(min_coverage):
        fails.append(
            f"A. coverage: the torch solver answered {int(ok.sum())}/{n} = {cov:.4f} of the rows "
            f"the numpy planner answered (< {min_coverage})"
        )
    stats = {"n": n, "coverage": cov}
    if int(ok.sum()) == 0:
        return {"failures": fails, "stats": stats}

    m = ok
    n_c, n_b = out["n"][m], bank_normal[m]
    ref = ref_normal[m]
    d_c = torch.sum(n_c * ref, dim=-1)
    d_b = torch.sum(n_b * ref, dim=-1)
    stats["min_dot_continuous_vs_clipface"] = float(d_c.min())
    stats["min_dot_bank_vs_clipface"] = float(d_b.min())
    if float(d_c.min()) <= 0.0:
        fails.append(
            f"B. face sign: {int((d_c <= 0).sum())}/{int(m.sum())} continuous answers point "
            f"OPPOSITE the clip face (min dot {float(d_c.min()):.4f}) — ref_normal sign alignment "
            f"is not reaching solve_strike_specs; this IS the M3c 单翻病"
        )
    if float(d_b.min()) <= 0.0:
        fails.append(
            f"B. face sign: {int((d_b <= 0).sum())}/{int(m.sum())} BANK answers point opposite the "
            f"clip face (min dot {float(d_b.min()):.4f}) — the bank itself is in the flipped "
            f"convention"
        )

    def _land(v_r, n_face):
        v_plus, w_plus = predict_paddle_contact(v_ball_in[m], v_r, n_face, w_ball_in[m], prm)
        return coarse_landing(p_contact[m], v_plus, w_plus, prm, surface_z=surface_z,
                              net_x=net_x, h=float(h), n_steps=int(n_steps))

    l_c, l_b = _land(out["v_r"][m], n_c), _land(bank_vel[m], n_b)
    for tag, land in (("continuous", l_c), ("bank", l_b)):
        if not bool(land["land_valid"].all()):
            fails.append(f"C. {tag}: {int((~land['land_valid']).sum())} answers never land")
        # D. net clearance — the acceptance the bank generator has always applied.
        clears = land["net_valid"] & (land["net_z"] > float(net_top_z))
        if not bool(clears.all()):
            fails.append(f"D. {tag}: {int((~clears).sum())} answers do not clear the net")
    both = l_c["land_valid"] & l_b["land_valid"]
    if bool(both.any()):
        d_land = torch.linalg.norm(l_c["land_xy"][both] - l_b["land_xy"][both], dim=-1)
        stats["land_delta_p50_m"] = float(d_land.median())
        stats["land_delta_max_m"] = float(d_land.max())
        if float(d_land.max()) > float(land_tol_m):
            fails.append(
                f"C. the two paths land {float(d_land.max()):.4f} m apart (> {land_tol_m}) for the "
                f"SAME ball — they are not the same physics"
            )
        for tag, land in (("continuous", l_c), ("bank", l_b)):
            d_aim = torch.linalg.norm(land["land_xy"][both] - aim_xy[m][both], dim=-1)
            stats[f"aim_err_max_m_{tag}"] = float(d_aim.max())
            if float(d_aim.max()) > float(aim_tol_m):
                fails.append(
                    f"C. {tag}: worst landing is {float(d_aim.max()):.4f} m from the aim point "
                    f"(> {aim_tol_m})"
                )
    # F. speed MAGNITUDE only. |v_r| is a physical consequence; the direction is not unique on the
    # solution manifold, so a bitwise v_r assertion would be a threshold-tuned lie.
    ds = (torch.linalg.norm(out["v_r"][m], dim=-1) - torch.linalg.norm(bank_vel[m], dim=-1)).abs()
    stats["speed_delta_max_mps"] = float(ds.max())
    if float(ds.max()) > float(speed_tol_mps):
        fails.append(f"F. demanded speed magnitudes differ by {float(ds.max()):.4f} m/s "
                     f"(> {speed_tol_mps})")
    ang = torch.rad2deg(torch.arccos(torch.sum(n_c * n_b, dim=-1).clamp(-1.0, 1.0)))
    stats["face_angle_max_deg"] = float(ang.max())      # reported, deliberately NOT asserted
    del dev, dt
    return {"failures": fails, "stats": stats}


@torch.no_grad()
def face_sign_negative_control(
    p_contact: torch.Tensor,
    v_ball_in: torch.Tensor,
    w_ball_in: torch.Tensor,
    aim_xy: torch.Tensor,
    ref_normal: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    speed_budget: float = 4.0,
    tol_m: float = 0.005,
    n_iters: int = 12,
) -> tuple[int, int]:
    """WITHOUT ``ref_normal``: how many solved rows come back on the WRONG side of the clip face.

    人话:这是"符号对齐还在不在"的反向证据。谁把 ref_normal 删掉,这个数会从"全翻"变成"不翻",
    测试当场红——不用起 env、不用等 3000 迭代。返回 (翻转行数, 解出行数)。
    """
    out = solve_strike_specs(
        p_contact, v_ball_in, w_ball_in, aim_xy, prm, surface_z=surface_z, net_x=net_x,
        ref_normal=None, speed_budget=float(speed_budget), n_iters=int(n_iters), tol_m=float(tol_m),
    )
    ok = out["ok"]
    if int(ok.sum()) == 0:
        return 0, 0
    d = torch.sum(out["n"][ok] * ref_normal[ok], dim=-1)
    return int((d < 0).sum()), int(ok.sum())
