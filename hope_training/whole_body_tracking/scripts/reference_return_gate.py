"""Can this clip's bound strike frame return the balls it will actually be posed?

人话:一个 clip 绑定的"参考击球状态"(触球点 + 拍速 + 拍面),对着**这个 clip 自己的来球箱**
采一批球,用仓库自己的 NumPy 记分器算:有多少球能合法回到对方半台。一个都回不去,就说明这个
clip 的击球帧或者它的球箱是错的——而这种错误的现场签名,正是"某一侧回球率无声地钉在 0.0000"。
所以它必须在建环境时当场炸,不能等训练曲线。

The scorer is ``scripts/virtual_return_scorer.VirtualReturnScorer`` — the SAME contact + flight +
landing contract the in-training ``virtual_return_rate`` metric is specified by — so this gate and
the metric can never disagree about what "returned" means.  Nothing here imports Isaac or torch.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from virtual_return_scorer import (  # noqa: E402
    VirtualReturnScorer,
    VirtualReturnSpec,
    load_venue_params,
    physical_b_to_raw_a,
)

# ITTF table, in the env-local frame the trainer uses (z = 0 at the FLOOR).
TABLE_LENGTH_M = 2.74
TABLE_HALF_WIDTH_M = 0.7625
NET_HEIGHT_M = 0.1525


def score_reference_returns(
    *,
    p_contact_w,
    v_racket_w,
    n_racket_w,
    vel_box,
    spin_abs_max: float,
    surface_z: float = 0.76,
    near_x: float = 0.5,
    n_samples: int = 256,
    seed: int = 0,
    face_sign: float = 1.0,
    venue_yaml: str | None = None,
    capture_radius: float = 0.095,
    min_approach_speed: float = 0.3,
) -> float:
    """Legal-return fraction of one reference strike state over one incoming-ball box.

    Parameters
    ----------
    p_contact_w, v_racket_w, n_racket_w
        The clip's bound strike state: contact position, racket velocity, and the PHYSICAL striking
        face normal (opponent-facing), all env-local (z above the FLOOR).
    vel_box
        ``((x_lo,x_hi), (y_lo,y_hi), (z_lo,z_hi))`` incoming-ball velocity box, m/s.
    spin_abs_max
        Per-axis uniform |spin| ceiling, rad/s.
    surface_z, near_x
        Table surface height and near-edge x in the same frame (``vb_table_surface_z`` /
        ``vb_table_near_x``).

    Returns the fraction of sampled balls whose strike lands legally on the opponent half. The ball
    is defined to arrive AT the contact point (``pos_err = 0``), which is exactly the trainer's own
    construction — so a low fraction is a statement about the strike state, not about aiming.
    """
    params = load_venue_params(venue_yaml)
    spec = VirtualReturnSpec(
        table_surface_z=float(surface_z),
        net_x=float(near_x) + TABLE_LENGTH_M / 2.0,
        far_x=float(near_x) + TABLE_LENGTH_M,
        half_width=TABLE_HALF_WIDTH_M,
        net_height=NET_HEIGHT_M,
        capture_radius=float(capture_radius),
        min_approach_speed=float(min_approach_speed),
    )
    scorer = VirtualReturnScorer(
        params, spec, mount_normal_sign_per_clip=(float(face_sign),), signed_face_required=True
    )

    n_b = np.asarray(n_racket_w, dtype=float).reshape(3)
    n_b = n_b / (float(np.linalg.norm(n_b)) + 1e-12)
    raw_a = physical_b_to_raw_a(n_b, float(face_sign))

    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = [
        (float(a), float(b)) for a, b in vel_box
    ]
    rng = np.random.default_rng(int(seed))
    n = max(int(n_samples), 1)
    vel = np.stack([
        rng.uniform(x_lo, x_hi, n), rng.uniform(y_lo, y_hi, n), rng.uniform(z_lo, z_hi, n)
    ], axis=-1)
    s = float(spin_abs_max)
    spin = rng.uniform(-s, s, (n, 3)) if s > 0.0 else np.zeros((n, 3))

    legal = 0
    for i in range(n):
        out = scorer.score(
            ball_vel=vel[i], ball_spin=spin[i],
            racket_pos=np.asarray(p_contact_w, dtype=float).reshape(3),
            racket_vel=np.asarray(v_racket_w, dtype=float).reshape(3),
            racket_normal_raw_a=raw_a, target_normal_raw_a=raw_a,
            clip_id=0, pos_err=0.0,
        )
        legal += int(bool(out.landed_ok))
    return legal / float(n)
