"""Fit-lineage NumPy reference used by the ball-physics parity gates.

The original reference lived outside the repository in the venue fitting
workspace.  That copy was lost, so this module reconstructs the same public API
from the fit-lineage sources that were already committed here.  It deliberately
does not import the Torch implementation under test.

Constants are read from ``configs/ball_physics_venue.yaml``.  Call
``reference_lineage()`` to record the exact source/config SHA-256 tuple used by
a parity result.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
from dataclasses import dataclass

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTACT_MODEL_PATH = os.path.join(_HERE, "contact_model.py")


def _load_sibling(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fit-lineage reference source: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cm = _load_sibling("_bpf_contact_model", _CONTACT_MODEL_PATH)


@functools.lru_cache(maxsize=1)
def _venue_path() -> str:
    explicit = os.environ.get("HOPE_BALL_PHYSICS_YAML")
    if explicit:
        path = os.path.realpath(explicit)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"$HOPE_BALL_PHYSICS_YAML={explicit} is not a file; refusing fallback"
            )
        return path

    d = _HERE
    while True:
        candidate = os.path.join(d, "configs", "ball_physics_venue.yaml")
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError(
                "configs/ball_physics_venue.yaml not found above reference_oracle.py; "
                "set $HOPE_BALL_PHYSICS_YAML"
            )
        d = parent


@functools.lru_cache(maxsize=1)
def _venue():
    import yaml

    with open(_venue_path(), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"venue physics YAML is not a mapping: {_venue_path()}")
    return data


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_lineage() -> dict[str, object]:
    """Return a stable content identity for every byte source this oracle uses."""

    files = {
        "reference_oracle.py": _sha256(__file__),
        "contact_model.py": _sha256(_CONTACT_MODEL_PATH),
        "ball_physics_venue.yaml": _sha256(_venue_path()),
    }
    canonical = "".join(f"{name}:{files[name]}\n" for name in sorted(files))
    return {
        "schema_version": 1,
        "files_sha256": files,
        "combined_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    }


def _unit_normals(value, *, label: str) -> np.ndarray:
    """Normalize one or more 3-vectors, rejecting zero/non-finite input loudly.

    Scaling by the largest component before taking the norm avoids overflow and
    underflow for finite but unusually scaled vectors.  A zero vector has no
    contact-plane semantics and must never be allowed to turn into NaNs.
    """

    arr = np.atleast_2d(np.asarray(value, dtype=float))
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"{label} must have shape (3,) or (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    scale = np.max(np.abs(arr), axis=1, keepdims=True)
    if np.any(scale == 0.0):
        raise ValueError(f"{label} contains a zero normal")
    scaled = arr / scale
    length = np.linalg.norm(scaled, axis=1, keepdims=True)
    if not np.all(np.isfinite(length)) or np.any(length == 0.0):
        raise ValueError(f"{label} cannot be normalized safely")
    unit = scaled / length
    if not np.all(np.isfinite(unit)):
        raise ValueError(f"{label} normalization produced a non-finite value")
    return unit


def _g_vec() -> np.ndarray:
    return np.array([0.0, 0.0, -float(_venue()["flight"]["g"])])


def _ball_radius() -> float:
    return float(_venue()["ball"]["radius"])


@dataclass
class SpinEquationParams:
    e_eff: float
    a_t: float
    b_t: float
    mu_safety: float


def predict_contact(v_minus, v_r, n, omega_minus, params):
    """Single-contact wrapper over the vectorized fit-lineage implementation."""

    normal = _unit_normals(n, label="contact normal")
    return _cm.predict_contact(
        v_minus,
        v_r,
        normal,
        omega_minus,
        e_eff=params.e_eff,
        a_t=params.a_t,
        b_t=params.b_t,
        mu=params.mu_safety,
    )


def rk4(p, v, omega, k_d, h, k_m=0.0):
    """One RK4 flight step for ``a = g-k_d|v|v+k_m(omega x v)``."""

    p = np.asarray(p, float)
    v = np.asarray(v, float)
    omega = np.asarray(omega, float)
    g = _g_vec()

    def acc(vv):
        return g - k_d * np.linalg.norm(vv) * vv + k_m * np.cross(omega, vv)

    a1 = acc(v)
    v2 = v + 0.5 * h * a1
    a2 = acc(v2)
    v3 = v + 0.5 * h * a2
    a3 = acc(v3)
    v4 = v + h * a3
    a4 = acc(v4)
    p_new = p + (h / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4)
    v_new = v + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return p_new, v_new


@dataclass
class Table:
    center_m: object
    normal: object
    surface_z_m: float


def table_bounce(v, omega, n, params=None):
    """Apply the fit-lineage contact equation to a static table surface."""

    if params is None:
        tab = _venue()["contact"]["table"]
        params = SpinEquationParams(
            e_eff=float(tab["e_eff"]),
            a_t=float(tab["a_t"]),
            b_t=float(tab["b_t"]),
            mu_safety=float(tab["mu_safety"]),
        )
    out = predict_contact(v, np.zeros(3), n, omega, params)
    return out["v_plus"][0], out["omega_plus"][0]


def simulate(p0, v0, omega0, k_d, table, t_total, bounce=True, h=5e-4, k_m=0.0):
    """Integrate flight and optional table bounces.

    Returns ``(times, positions, contacts)``.  Contact occurs when the ball
    center crosses ``table.surface_z_m + radius`` while moving downward.
    """

    p = np.asarray(p0, float).copy()
    v = np.asarray(v0, float).copy()
    w = np.asarray(omega0, float).copy()
    normals = _unit_normals(table.normal, label="table normal")
    if normals.shape[0] != 1:
        raise ValueError(f"table normal must contain exactly one vector, got {normals.shape[0]}")
    n_t = normals[0]
    contact_z = float(table.surface_z_m) + _ball_radius()

    times, positions, contacts = [0.0], [p.copy()], []
    t = 0.0
    while t < t_total - 1e-12:
        step = min(h, t_total - t)
        p1, v1 = rk4(p, v, w, k_d, step, k_m)
        if bounce and v[2] < 0.0 and p[2] > contact_z >= p1[2]:
            lo, hi = 0.0, step
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                pm, _ = rk4(p, v, w, k_d, mid, k_m)
                if pm[2] > contact_z:
                    lo = mid
                else:
                    hi = mid
            fraction = 0.5 * (lo + hi)
            pc, vc = rk4(p, v, w, k_d, fraction, k_m)
            v_out, w_out = table_bounce(vc, w, n_t)
            t += fraction
            contacts.append(
                {
                    "t": t,
                    "p": pc.copy(),
                    "v_in": vc.copy(),
                    "v_out": np.asarray(v_out, float).copy(),
                    "omega_out": np.asarray(w_out, float).copy(),
                }
            )
            p = pc
            v = np.asarray(v_out, float).copy()
            w = np.asarray(w_out, float).copy()
            times.append(t)
            positions.append(p.copy())
            continue
        p, v = p1, v1
        t += step
        times.append(t)
        positions.append(p.copy())
    return np.asarray(times), np.asarray(positions), contacts
