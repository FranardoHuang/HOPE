"""Hand-written float64 3-vector cross products, bit-identical to np.cross.

np.cross performs exactly these IEEE-754 multiplies/subtracts per component
but routes them through ~10 us of python/ufunc dispatch per call; in the
planner's 1 kHz flight loop the Magnus term np.cross(omega, v) was ~60 % of
the per-step cost (benchmarks/benchmark_planner_latency.py --profile). The
helpers below do the SAME arithmetic in the SAME order, so on float64 the
results are equal bit for bit — asserted over random states including zeros,
signed zeros and subnormals by test/test_strike_spec_fast.py; if numpy ever
reordered its component arithmetic that test fails loudly and the safe move
is to revert callers to np.cross.
"""

from __future__ import annotations

import numpy as np


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a x b for 3-vectors. Bit-identical to np.cross(a, b) on float64."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def cross_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise a_i x b_i for (M, 3) arrays. Bit-identical to np.cross(A, B)."""
    return np.stack([
        A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1],
        A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2],
        A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0],
    ], axis=1)
