#!/usr/bin/env python3
"""Fail-closed ActionBall fitted-physics equivalence audit.

This audit answers a narrower question than a policy or teacher-motion gate:

* do the pinned ActionBall profile, the Isaac code-driven ball path, and the
  formal MuJoCo fitted-ball path consume the same physical constants;
* do the two independently implemented contact equations produce the same
  outgoing linear and angular state; and
* are table/net geometry and the two fitted-MuJoCo timesteps bound to the same
  values?

There is deliberately no claimed mapping from the venue contact fit to native
MuJoCo ``solref/solimp/friction``.  Both formal engines disable the ball's
native collision pair and apply the venue ``(e_eff, a_t, b_t, mu)`` impulse in
code.  Calling a guessed native material "equivalent" is an audit failure.

Formal mode requires:

* an exact profile-pins SHA;
* an exact fitted-MuJoCo PASS receipt SHA with compiled 1.0/0.5 ms scenes;
* Torch execution of the real Isaac ``virtual_ball`` and ``physical_ball``
  contact helpers; and
* an exact clean Git commit.

The host-only ``--diagnostic`` mode still checks parameters, source contracts,
NumPy formulae, mutation witnesses, and timestep convergence, but returns
``BLOCKED`` rather than a formal PASS when Torch or a fitted receipt is absent.
It never authorizes training, deployment, or hardware.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
RECEIPT_CLASS = "action_ball_cross_engine_physics_equivalence_v1"
FORMAL_DT_S = (0.001, 0.0005)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_PATHS = {
    "venue": "configs/ball_physics_venue.yaml",
    "contact_model": "hope_training/ball_physics_fit/contact_model.py",
    "fitted_gate": (
        "hope_training/whole_body_tracking/scripts/"
        "mujoco_teacher_motion_fitted_ball_gate.py"
    ),
    "profile_pins_validator": (
        "hope_training/whole_body_tracking/scripts/"
        "mujoco_teacher_motion_native_ball_diagnostic.py"
    ),
    "virtual_ball": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/virtual_ball.py"
    ),
    "physical_ball": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/physical_ball.py"
    ),
    "shadow_ball": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/shadow_ball.py"
    ),
    "hope_env_cfg": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    ),
    "geometry": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/geometry.py"
    ),
    "table_frame": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/table_frame.py"
    ),
    "mujoco_table_scene": "scripts/mujoco_table_scene.py",
}


class PhysicsEquivalenceError(RuntimeError):
    """The cross-engine physical identity or response did not close."""


class _UniqueSafeLoader:
    """Namespace populated lazily because PyYAML is optional at import time."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PhysicsEquivalenceError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _finite(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) not in (int, float):
        raise PhysicsEquivalenceError(f"{label} must be one finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (positive and result <= 0.0)
        or (nonnegative and result < 0.0)
    ):
        raise PhysicsEquivalenceError(f"{label} is outside its physical domain")
    return result


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicate(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise PhysicsEquivalenceError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            out[key] = value
        return out

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PhysicsEquivalenceError(
                    f"{label} contains forbidden JSON constant {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicsEquivalenceError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PhysicsEquivalenceError(f"{label} must be one JSON object")
    return value


def _strict_yaml(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise PhysicsEquivalenceError("PyYAML is required") from exc

    class Loader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        out = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in out:
                raise PhysicsEquivalenceError(
                    f"{label} contains duplicate YAML key {key!r}"
                )
            out[key] = loader.construct_object(value_node, deep=deep)
        return out

    Loader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        value = yaml.load(payload.decode("utf-8"), Loader=Loader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PhysicsEquivalenceError(f"{label} is not strict YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PhysicsEquivalenceError(f"{label} must be one YAML object")
    return value


def _plain_file(path: Path, *, root: Path, label: str) -> tuple[Path, bytes]:
    root = root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    lexical = candidate.absolute()
    current = Path(lexical.parts[0])
    for part in lexical.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise PhysicsEquivalenceError(
                f"cannot inspect {label} path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise PhysicsEquivalenceError(
                f"{label} contains a symlink component: {current}"
            )
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PhysicsEquivalenceError(
            f"{label} must resolve inside repository root"
        ) from exc
    if not resolved.is_file():
        raise PhysicsEquivalenceError(f"{label} is not a regular file")
    return resolved, resolved.read_bytes()


def _snapshot_sources(root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for role, relative in _SOURCE_PATHS.items():
        path, payload = _plain_file(Path(relative), root=root, label=role)
        rows[role] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    return rows


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PhysicsEquivalenceError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_formal_profile_pins(
    root: Path,
    profile: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run the exact validator shared with both MuJoCo teacher gates."""

    profile_validator = _load_module(
        "_action_ball_equivalence_profile_pins_validator",
        root / _SOURCE_PATHS["profile_pins_validator"],
    )
    try:
        return profile_validator.validate_profile_pins(
            profile,
            manifest=None,
            repo_root=root,
        )
    except profile_validator.GateError as exc:
        raise PhysicsEquivalenceError(
            f"formal profile-pins contract failed: {exc}"
        ) from exc


def _source_contracts(root: Path, snapshots: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind the behavioral seams that parameter equality relies on.

    Runtime numeric replay below is the primary proof.  These source checks
    prevent it from being misread as parity with native engine contacts:
    fitted MuJoCo and Isaac must both keep the ball collision pair disabled,
    use zero angular damping/torque, and delegate to the code contact law.
    """

    required = {
        "fitted_gate": (
            '"contype": "0"',
            '"conaffinity": "0"',
            "contact_model.predict_contact(",
            "venue.paddle_g1 * math.exp(",
            "e_eff=venue.table_e",
            "option.set(\"gravity\"",
        ),
        "hope_env_cfg": (
            "collision_enabled=False",
            "linear_damping=0.0",
            "angular_damping=0.0",
            "enable_gyroscopic_forces=False",
            "mass=ball_m",
            "radius=ball_r",
        ),
        "physical_ball": (
            "return _vb.predict_paddle_contact(",
            "def predict_table_contact(",
            "self._ball.write_root_velocity_to_sim(",
            "self._ball.set_external_force_and_torque(",
        ),
        "virtual_ball": (
            "def predict_paddle_contact(",
            "paddle_e_g1 * torch.exp(",
            "delta_omega = -(1.0 / (c * R))",
            "def flight_accel(",
        ),
        "shadow_ball": (
            "return mass * (-k_d * speed * vel_clipped",
            "Torque is zero",
        ),
    }
    rows = []
    for role, tokens in required.items():
        relative = snapshots[role]["path"]
        source = (root / relative).read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=relative)
        except SyntaxError as exc:
            raise PhysicsEquivalenceError(
                f"{role} source is not valid Python: {exc}"
            ) from exc
        missing = [token for token in tokens if token not in source]
        if missing:
            raise PhysicsEquivalenceError(
                f"{role} no longer implements the frozen physical seam: {missing}"
            )
        rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": snapshots[role]["sha256"],
                "required_semantic_tokens": list(tokens),
                "pass": True,
            }
        )
    return rows


def _venue_values(venue: Mapping[str, Any]) -> dict[str, float]:
    try:
        ball = venue["ball"]
        flight = venue["flight"]
        table = venue["contact"]["table"]
        paddle = venue["contact"]["paddle"]
    except (KeyError, TypeError) as exc:
        raise PhysicsEquivalenceError(
            f"venue YAML schema is incomplete: {exc}"
        ) from exc
    return {
        "ball_mass_kg": _finite(ball["mass"], "ball.mass", positive=True),
        "ball_radius_m": _finite(ball["radius"], "ball.radius", positive=True),
        "inertia_coeff": _finite(
            ball["inertia_coeff"], "ball.inertia_coeff", positive=True
        ),
        "gravity_mps2": _finite(flight["g"], "flight.g", positive=True),
        "flight_k_d_per_m": _finite(
            flight["k_d"], "flight.k_d", nonnegative=True
        ),
        "flight_k_m": _finite(
            flight["k_m"], "flight.k_m", nonnegative=True
        ),
        "spin_decay_per_s": 0.0,
        "table_e_eff": _finite(
            table["e_eff"], "contact.table.e_eff", nonnegative=True
        ),
        "table_a_t": _finite(
            table["a_t"], "contact.table.a_t", nonnegative=True
        ),
        "table_b_t": _finite(table["b_t"], "contact.table.b_t"),
        "table_mu": _finite(
            table["mu_safety"],
            "contact.table.mu_safety",
            nonnegative=True,
        ),
        "paddle_constant_e_fit": _finite(
            paddle["e_eff"], "contact.paddle.e_eff", nonnegative=True
        ),
        "paddle_e_g1": _finite(
            paddle["e_exp_g1"],
            "contact.paddle.e_exp_g1",
            nonnegative=True,
        ),
        "paddle_e_g2": _finite(
            paddle["e_exp_g2"], "contact.paddle.e_exp_g2"
        ),
        "paddle_a_t": _finite(
            paddle["a_t"], "contact.paddle.a_t", nonnegative=True
        ),
        "paddle_b_t": _finite(paddle["b_t"], "contact.paddle.b_t"),
        "paddle_mu": _finite(
            paddle["mu_safety"],
            "contact.paddle.mu_safety",
            nonnegative=True,
        ),
    }


def _profile_values(
    profile: Mapping[str, Any],
    *,
    profile_sha256: str,
    venue_sha256: str,
    validated_profile: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    payload = profile.get("physics_payload")
    if not isinstance(payload, Mapping):
        raise PhysicsEquivalenceError("profile pins omit physics_payload")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind")
        != "whole_body_tracking.action_ball.physics_and_scorer"
        or profile.get("physics_profile_sha256") != _canonical_sha256(payload)
    ):
        raise PhysicsEquivalenceError(
            "profile physics payload identity/seal is false"
        )
    venue_source = payload.get("venue_source")
    if (
        not isinstance(venue_source, Mapping)
        or set(venue_source) != {"path", "file_sha256"}
        or venue_source["path"] != _SOURCE_PATHS["venue"]
        or venue_source["file_sha256"] != venue_sha256
    ):
        raise PhysicsEquivalenceError(
            "profile does not bind the exact venue YAML bytes"
        )
    params = payload.get("virtual_ball_params")
    expected_param_keys = {
        "k_d",
        "k_m",
        "g",
        "ball_radius",
        "inertia_coeff",
        "paddle_a_t",
        "paddle_b_t",
        "paddle_mu",
        "paddle_e_g1",
        "paddle_e_g2",
    }
    if not isinstance(params, Mapping) or set(params) != expected_param_keys:
        raise PhysicsEquivalenceError(
            "profile virtual_ball_params key set is not exact"
        )
    geometry = payload.get("geometry_and_grading")
    expected_geometry_keys = {
        "table_surface_z_m",
        "ball_center_surface_z_m",
        "opponent_near_x_m",
        "net_x_m",
        "ball_center_net_top_z_m",
        "opponent_far_x_m",
        "table_half_width_m",
        "minimum_landing_depth_m",
        "capture_radius_m",
        "minimum_approach_speed_mps",
    }
    if not isinstance(geometry, Mapping) or set(geometry) != expected_geometry_keys:
        raise PhysicsEquivalenceError(
            "profile geometry_and_grading key set is not exact"
        )
    return payload, [
        {
            "role": "profile_pins",
            "sha256": profile_sha256,
            "physics_profile_sha256": profile["physics_profile_sha256"],
            "solver_profile_sha256": validated_profile[
                "solver_profile_sha256"
            ],
            "source_blob_map_sha256": validated_profile[
                "source_blob_map_sha256"
            ],
            "source_authority": validated_profile["source_authority"][
                "authority"
            ],
            "solver_source_count": len(
                validated_profile[
                    "solver_implementation_source_sha256"
                ]
            ),
            "contact_geometry_sha256": validated_profile[
                "contact_geometry_sha256"
            ],
            "venue_source_path": venue_source["path"],
            "venue_source_sha256": venue_source["file_sha256"],
        }
    ]


def _assert_close(
    actual: float, expected: float, label: str, *, tolerance: float = 1.0e-12
) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise PhysicsEquivalenceError(
            f"{label} mismatch: {actual!r} != {expected!r}"
        )


def _parameter_ledger(
    values: Mapping[str, float],
    profile_payload: Mapping[str, Any],
    geometry_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    prm = profile_payload["virtual_ball_params"]
    profile_pairs = {
        "ball_radius_m": prm["ball_radius"],
        "inertia_coeff": prm["inertia_coeff"],
        "gravity_mps2": prm["g"],
        "flight_k_d_per_m": prm["k_d"],
        "flight_k_m": prm["k_m"],
        "paddle_e_g1": prm["paddle_e_g1"],
        "paddle_e_g2": prm["paddle_e_g2"],
        "paddle_a_t": prm["paddle_a_t"],
        "paddle_b_t": prm["paddle_b_t"],
        "paddle_mu": prm["paddle_mu"],
    }
    rows = []
    for name, profile_value in profile_pairs.items():
        expected = values[name]
        _assert_close(profile_value, expected, f"profile {name}", tolerance=0.0)
        rows.append(
            {
                "parameter": name,
                "value": expected,
                "venue_source": _SOURCE_PATHS["venue"],
                "profile_binding": "direct_virtual_ball_params",
                "profile_value": float(profile_value),
                "mujoco_consumer": "fitted_gate.VenueParams",
                "isaac_consumer": "virtual_ball.VirtualBallParams",
                "exact_equal": True,
            }
        )
    for name in (
        "ball_mass_kg",
        "spin_decay_per_s",
        "table_e_eff",
        "table_a_t",
        "table_b_t",
        "table_mu",
        "paddle_constant_e_fit",
    ):
        rows.append(
            {
                "parameter": name,
                "value": values[name],
                "venue_source": _SOURCE_PATHS["venue"],
                "profile_binding": "transitive_exact_venue_file_sha256",
                "profile_value": values[name],
                "mujoco_consumer": "fitted_gate.load_venue_yaml",
                "isaac_consumer": (
                    "physical_ball.load_venue_table_params"
                    if name.startswith("table_")
                    else "hope_env_cfg/physical_ball"
                ),
                "exact_equal": True,
            }
        )
    for attr, name in (
        ("BALL_MASS", "ball_mass_kg"),
        ("BALL_RADIUS", "ball_radius_m"),
        ("BALL_INERTIA_COEFF", "inertia_coeff"),
    ):
        _assert_close(
            getattr(geometry_module, attr),
            values[name],
            f"table_tennis.geometry.{attr}",
            tolerance=1.0e-12,
        )
    geometry = profile_payload["geometry_and_grading"]
    near = _finite(geometry["opponent_near_x_m"], "opponent_near_x_m")
    surface = _finite(geometry["table_surface_z_m"], "table_surface_z_m")
    derived = {
        "table_length_m": float(geometry_module.TABLE_LENGTH),
        "table_width_m": float(geometry_module.TABLE_WIDTH),
        "table_height_m": float(geometry_module.TABLE_HEIGHT),
        "table_thickness_m": float(geometry_module.TABLE_THICKNESS),
        "table_surface_z_m": surface,
        "table_near_x_m": near,
        "table_far_x_m": near + float(geometry_module.TABLE_LENGTH),
        "net_x_m": near + float(geometry_module.NET_X),
        "net_height_m": float(geometry_module.NET_HEIGHT),
        "net_top_ball_center_z_m": (
            surface
            + float(geometry_module.NET_HEIGHT)
            + values["ball_radius_m"]
        ),
        "net_thickness_m": float(geometry_module.NET_THICKNESS),
        "net_overhang_m": float(geometry_module.NET_OVERHANG),
    }
    checks = {
        "opponent_far_x_m": derived["table_far_x_m"],
        "table_half_width_m": 0.5 * derived["table_width_m"],
        "net_x_m": derived["net_x_m"],
        "ball_center_net_top_z_m": derived["net_top_ball_center_z_m"],
        "ball_center_surface_z_m": surface + values["ball_radius_m"],
    }
    for key, expected in checks.items():
        _assert_close(geometry[key], expected, f"profile geometry {key}")
    return rows, derived


def _orient_normal(
    normal: Sequence[float],
    velocity: Sequence[float],
    surface_velocity: Sequence[float],
) -> np.ndarray:
    n = np.asarray(normal, np.float64)
    norm = float(np.linalg.norm(n))
    if not math.isfinite(norm) or norm <= 0.0:
        raise PhysicsEquivalenceError("contact normal is degenerate")
    n = n / norm
    if float(np.dot(np.asarray(velocity) - np.asarray(surface_velocity), n)) > 0.0:
        n = -n
    return n


def _contact_oracle(
    velocity: Sequence[float],
    surface_velocity: Sequence[float],
    normal: Sequence[float],
    spin: Sequence[float],
    *,
    radius: float,
    inertia_coeff: float,
    e_eff: float,
    a_t: float,
    b_t: float,
    mu: float,
    epsilon: float,
) -> dict[str, Any]:
    v = np.asarray(velocity, np.float64)
    vr = np.asarray(surface_velocity, np.float64)
    omega = np.asarray(spin, np.float64)
    n = _orient_normal(normal, v, vr)
    r = -float(radius) * n
    u = v + np.cross(omega, r) - vr
    u_n = float(np.dot(u, n))
    u_t_vec = u - u_n * n
    u_t = float(np.linalg.norm(u_t_vec))
    cos_theta = abs(u_n) / (math.hypot(u_t, u_n) + float(epsilon))
    raw = (float(a_t) + float(b_t) * cos_theta) * u_t
    cap = float(mu) * (1.0 + float(e_eff)) * abs(u_n)
    impulse_t = min(max(raw, 0.0), cap)
    direction = u_t_vec / (u_t + float(epsilon))
    dv_t = (
        -impulse_t * direction
        if u_t > float(epsilon)
        else np.zeros(3, np.float64)
    )
    dv_n = -(1.0 + float(e_eff)) * u_n * n
    dw = -(1.0 / (float(inertia_coeff) * float(radius))) * np.cross(
        n, dv_t
    )
    return {
        "velocity_plus_mps": v + dv_n + dv_t,
        "spin_plus_radps": omega + dw,
        "oriented_normal_w": n,
        "u_n_mps": u_n,
        "u_t_mps": u_t,
        "cos_theta": cos_theta,
        "tangential_gain": float(a_t) + float(b_t) * cos_theta,
        "tangential_raw_delta_mps": raw,
        "tangential_cap_delta_mps": cap,
        "tangential_delta_mps": impulse_t,
        "cap_binds": bool(raw > cap),
        "normal_delta_v_mps": float(np.linalg.norm(dv_n)),
        "tangential_delta_v_mps": float(np.linalg.norm(dv_t)),
    }


def _contact_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paddle = [
        {
            "case_id": "paddle_nominal_oblique",
            "v": [-4.2, 0.7, -0.3],
            "vr": [2.0, -0.4, 0.2],
            "n": [1.0, 0.0, 0.0],
            "w": [12.0, -24.0, 48.0],
        },
        {
            "case_id": "paddle_mu_cap",
            "v": [-2.0, 12.0, -4.0],
            "vr": [0.0, 0.0, 0.0],
            "n": [1.0, 0.0, 0.0],
            "w": [0.0, 5.0, -10.0],
        },
        {
            "case_id": "paddle_normal_flip",
            "v": [3.5, -1.0, 0.8],
            "vr": [0.2, 0.1, -0.1],
            "n": [1.0, 0.0, 0.0],
            "w": [25.0, 15.0, -20.0],
        },
    ]
    table = [
        {
            "case_id": "table_nominal",
            "v": [1.1, -0.6, -3.0],
            "vr": [0.0, 0.0, 0.0],
            "n": [0.0, 0.0, 1.0],
            "w": [20.0, -30.0, 8.0],
        },
        {
            "case_id": "table_mu_cap",
            "v": [20.0, -12.0, -1.0],
            "vr": [0.0, 0.0, 0.0],
            "n": [0.0, 0.0, 1.0],
            "w": [0.0, 0.0, 0.0],
        },
        {
            "case_id": "table_b_t_oblique",
            "v": [4.0, 2.0, -2.0],
            "vr": [0.0, 0.0, 0.0],
            "n": [0.0, 0.0, 1.0],
            "w": [-15.0, 10.0, 5.0],
        },
    ]
    return paddle, table


def _numpy_contact_audit(
    values: Mapping[str, float],
    *,
    contact_model: Any,
) -> dict[str, Any]:
    if (
        abs(float(contact_model.R_BALL) - values["ball_radius_m"]) > 1.0e-12
        or abs(float(contact_model.C_INERTIA) - values["inertia_coeff"])
        > 1.0e-12
    ):
        raise PhysicsEquivalenceError(
            "MuJoCo contact-model radius/inertia differ from venue"
        )
    paddle_cases, table_cases = _contact_cases()
    rows = []
    maxima = {
        "paddle_velocity_mps": 0.0,
        "paddle_spin_radps": 0.0,
        "table_velocity_mps": 0.0,
        "table_spin_radps": 0.0,
    }
    for kind, cases in (("paddle", paddle_cases), ("table", table_cases)):
        for case in cases:
            n = _orient_normal(case["n"], case["v"], case["vr"])
            relative_normal = abs(
                float(
                    np.dot(
                        np.asarray(case["v"]) - np.asarray(case["vr"]),
                        n,
                    )
                )
            )
            if kind == "paddle":
                e_eff = float(
                    np.clip(
                        values["paddle_e_g1"]
                        * math.exp(values["paddle_e_g2"] * relative_normal),
                        0.05,
                        0.95,
                    )
                )
                a_t, b_t, mu, epsilon = (
                    values["paddle_a_t"],
                    values["paddle_b_t"],
                    values["paddle_mu"],
                    1.0e-12,
                )
            else:
                e_eff = values["table_e_eff"]
                a_t, b_t, mu, epsilon = (
                    values["table_a_t"],
                    values["table_b_t"],
                    values["table_mu"],
                    1.0e-9,
                )
            oracle = _contact_oracle(
                case["v"],
                case["vr"],
                case["n"],
                case["w"],
                radius=values["ball_radius_m"],
                inertia_coeff=values["inertia_coeff"],
                e_eff=e_eff,
                a_t=a_t,
                b_t=b_t,
                mu=mu,
                epsilon=epsilon,
            )
            actual = contact_model.predict_contact(
                np.asarray(case["v"], np.float64)[None, :],
                np.asarray(case["vr"], np.float64)[None, :],
                np.asarray(case["n"], np.float64)[None, :],
                np.asarray(case["w"], np.float64)[None, :],
                e_eff,
                a_t,
                b_t,
                mu,
            )
            v_actual = np.asarray(actual["v_plus"][0], np.float64)
            w_actual = np.asarray(actual["omega_plus"][0], np.float64)
            v_error = float(
                np.max(np.abs(v_actual - oracle["velocity_plus_mps"]))
            )
            w_error = float(
                np.max(np.abs(w_actual - oracle["spin_plus_radps"]))
            )
            tolerance = 5.0e-11 if kind == "paddle" else 2.0e-6
            if max(v_error, w_error) > tolerance:
                raise PhysicsEquivalenceError(
                    f"{case['case_id']} contact formula mismatch "
                    f"{max(v_error, w_error):.3e} > {tolerance:.3e}"
                )
            maxima[f"{kind}_velocity_mps"] = max(
                maxima[f"{kind}_velocity_mps"], v_error
            )
            maxima[f"{kind}_spin_radps"] = max(
                maxima[f"{kind}_spin_radps"], w_error
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "pair": kind,
                    "e_eff": e_eff,
                    "a_t": a_t,
                    "b_t": b_t,
                    "mu": mu,
                    "u_n_mps": oracle["u_n_mps"],
                    "u_t_mps": oracle["u_t_mps"],
                    "cos_theta": oracle["cos_theta"],
                    "tangential_gain": oracle["tangential_gain"],
                    "tangential_raw_delta_mps": oracle[
                        "tangential_raw_delta_mps"
                    ],
                    "tangential_cap_delta_mps": oracle[
                        "tangential_cap_delta_mps"
                    ],
                    "tangential_delta_mps": oracle[
                        "tangential_delta_mps"
                    ],
                    "cap_binds": oracle["cap_binds"],
                    "normal_delta_v_mps": oracle["normal_delta_v_mps"],
                    "tangential_delta_v_mps": oracle[
                        "tangential_delta_v_mps"
                    ],
                    "numpy_mujoco_vs_isaac_oracle_velocity_linf_mps": v_error,
                    "numpy_mujoco_vs_isaac_oracle_spin_linf_radps": w_error,
                    "tolerance": tolerance,
                    "pass": True,
                }
            )
    if not any(
        row["cap_binds"] and row["pair"] == "paddle" for row in rows
    ) or not any(row["cap_binds"] and row["pair"] == "table" for row in rows):
        raise PhysicsEquivalenceError(
            "contact cases do not exercise both friction caps"
        )
    return {
        "formula": {
            "relative_velocity": "u=v_minus+omega_cross(-R*n)-v_surface",
            "normal": "dv_n=-(1+e_eff)*u_n*n",
            "tangent_gain": "g_t=a_t+b_t*cos(theta)",
            "tangent": (
                "s=min(max(g_t*norm(u_t),0),mu*(1+e_eff)*abs(u_n));"
                "dv_t=-s*unit(u_t)"
            ),
            "spin": "domega=-(1/(inertia_coeff*R))*n_cross(dv_t)",
            "paddle_restitution": (
                "clip(g1*exp(g2*abs(u_n)),0.05,0.95)"
            ),
            "table_restitution": "constant e_eff",
        },
        "cases": rows,
        "max_errors": maxima,
    }


def _flight_acceleration(
    velocity: np.ndarray,
    spin: np.ndarray,
    values: Mapping[str, float],
) -> np.ndarray:
    return (
        np.asarray((0.0, 0.0, -values["gravity_mps2"]), np.float64)
        - values["flight_k_d_per_m"]
        * float(np.linalg.norm(velocity))
        * velocity
        + values["flight_k_m"] * np.cross(spin, velocity)
    )


def _rk4_step(
    position: np.ndarray,
    velocity: np.ndarray,
    spin: np.ndarray,
    dt: float,
    values: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    def acceleration(v):
        return _flight_acceleration(v, spin, values)

    a1 = acceleration(velocity)
    a2 = acceleration(velocity + 0.5 * dt * a1)
    a3 = acceleration(velocity + 0.5 * dt * a2)
    a4 = acceleration(velocity + dt * a3)
    v_next = velocity + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    p_next = position + (dt / 6.0) * (
        velocity
        + 2 * (velocity + 0.5 * dt * a1)
        + 2 * (velocity + 0.5 * dt * a2)
        + (velocity + dt * a3)
    )
    return p_next, v_next


def _dual_dt_audit(values: Mapping[str, float]) -> dict[str, Any]:
    p0 = np.asarray((2.6, 0.1, 1.2), np.float64)
    v0 = np.asarray((-4.5, 0.7, -0.4), np.float64)
    spin = np.asarray((15.0, -20.0, 50.0), np.float64)
    horizon = 0.05
    rows = []
    for dt in FORMAL_DT_S:
        count = int(round(horizon / dt))
        if abs(count * dt - horizon) > 1.0e-15:
            raise PhysicsEquivalenceError("dual-dt horizon is not integral")
        p_mj, v_mj = p0.copy(), v0.copy()
        p_ix, v_ix = p0.copy(), v0.copy()
        for _ in range(count):
            p_mj, v_mj = _rk4_step(p_mj, v_mj, spin, dt, values)
            # physical_ball default n_sub=1: PhysX gravity plus the exact
            # current-state venue aero wrench gives this velocity update.
            v_ix = v_ix + dt * _flight_acceleration(v_ix, spin, values)
            p_ix = p_ix + dt * v_ix
        rows.append(
            {
                "dt_s": dt,
                "steps": count,
                "horizon_s": horizon,
                "mujoco_rk4_position_m": p_mj.tolist(),
                "isaac_euler_position_m": p_ix.tolist(),
                "mujoco_rk4_velocity_mps": v_mj.tolist(),
                "isaac_euler_velocity_mps": v_ix.tolist(),
                "position_delta_m": float(np.linalg.norm(p_mj - p_ix)),
                "velocity_delta_mps": float(np.linalg.norm(v_mj - v_ix)),
                "mujoco_spin_end_radps": spin.tolist(),
                "isaac_spin_end_radps": spin.tolist(),
                "spin_decay_per_s": values["spin_decay_per_s"],
            }
        )
    coarse, fine = rows
    for field in ("position_delta_m", "velocity_delta_mps"):
        if not (
            fine[field] < coarse[field]
            and fine[field] / coarse[field] <= 0.55
        ):
            raise PhysicsEquivalenceError(
                f"Isaac-Euler/MuJoCo-RK4 {field} does not converge under dt halving"
            )
    if any(
        row["mujoco_spin_end_radps"] != row["isaac_spin_end_radps"]
        or row["mujoco_spin_end_radps"] != spin.tolist()
        for row in rows
    ):
        raise PhysicsEquivalenceError("zero spin-decay contract is false")
    return {
        "interpretation": (
            "same continuous ODE and constant spin; different integrators are "
            "not claimed bit-identical, and the first-order Isaac-vs-RK4 "
            "difference must shrink under dt halving"
        ),
        "rows": rows,
        "fine_over_coarse_position_error": (
            fine["position_delta_m"] / coarse["position_delta_m"]
        ),
        "fine_over_coarse_velocity_error": (
            fine["velocity_delta_mps"] / coarse["velocity_delta_mps"]
        ),
        "pass": True,
    }


def _numeric_signature(
    values: Mapping[str, float],
    geometry: Mapping[str, float],
) -> np.ndarray:
    velocity = np.asarray((-4.2, 0.7, -0.3), np.float64)
    spin = np.asarray((12.0, -24.0, 48.0), np.float64)
    acceleration = _flight_acceleration(velocity, spin, values)
    spin_end = spin * math.exp(-values["spin_decay_per_s"] * 0.2)
    paddle_nominal = _contact_oracle(
        velocity,
        (2.0, -0.4, 0.2),
        (1.0, 0.0, 0.0),
        spin,
        radius=values["ball_radius_m"],
        inertia_coeff=values["inertia_coeff"],
        e_eff=float(
            np.clip(
                values["paddle_e_g1"]
                * math.exp(values["paddle_e_g2"] * 6.2),
                0.05,
                0.95,
            )
        ),
        a_t=values["paddle_a_t"],
        b_t=values["paddle_b_t"],
        mu=values["paddle_mu"],
        epsilon=1.0e-12,
    )
    paddle_cap = _contact_oracle(
        (-2.0, 12.0, -4.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 5.0, -10.0),
        radius=values["ball_radius_m"],
        inertia_coeff=values["inertia_coeff"],
        e_eff=float(
            np.clip(
                values["paddle_e_g1"]
                * math.exp(values["paddle_e_g2"] * 2.0),
                0.05,
                0.95,
            )
        ),
        a_t=values["paddle_a_t"],
        b_t=values["paddle_b_t"],
        mu=values["paddle_mu"],
        epsilon=1.0e-12,
    )
    table_nominal = _contact_oracle(
        (4.0, 2.0, -2.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (-15.0, 10.0, 5.0),
        radius=values["ball_radius_m"],
        inertia_coeff=values["inertia_coeff"],
        e_eff=values["table_e_eff"],
        a_t=values["table_a_t"],
        b_t=values["table_b_t"],
        mu=values["table_mu"],
        epsilon=1.0e-9,
    )
    table_cap = _contact_oracle(
        (20.0, -12.0, -1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        radius=values["ball_radius_m"],
        inertia_coeff=values["inertia_coeff"],
        e_eff=values["table_e_eff"],
        a_t=values["table_a_t"],
        b_t=values["table_b_t"],
        mu=values["table_mu"],
        epsilon=1.0e-9,
    )
    aero_force = values["ball_mass_kg"] * (
        acceleration + np.asarray((0.0, 0.0, values["gravity_mps2"]))
    )
    inertia = (
        values["inertia_coeff"]
        * values["ball_mass_kg"]
        * values["ball_radius_m"] ** 2
    )
    return np.concatenate(
        [
            acceleration,
            aero_force,
            spin_end,
            paddle_nominal["velocity_plus_mps"],
            paddle_nominal["spin_plus_radps"],
            paddle_cap["velocity_plus_mps"],
            table_nominal["velocity_plus_mps"],
            table_nominal["spin_plus_radps"],
            table_cap["velocity_plus_mps"],
            np.asarray(
                [
                    inertia,
                    geometry["table_length_m"],
                    geometry["table_width_m"],
                    geometry["table_surface_z_m"],
                    geometry["net_x_m"],
                    geometry["net_height_m"],
                    geometry["net_thickness_m"],
                ],
                np.float64,
            ),
        ]
    )


def _mutation_witnesses(
    values: Mapping[str, float],
    geometry: Mapping[str, float],
) -> list[dict[str, Any]]:
    base_values = dict(values)
    base_geometry = dict(geometry)
    baseline = _numeric_signature(base_values, base_geometry)
    mutations = {
        "ball_mass_kg": ("values", 1.01),
        "ball_radius_m": ("values", 1.01),
        "gravity_mps2": ("values", 1.001),
        "flight_k_d_per_m": ("values", 1.01),
        "flight_k_m": ("values", 1.01),
        "spin_decay_per_s": ("values_add", 0.1),
        "table_e_eff": ("values", 1.001),
        "table_a_t": ("values", 1.01),
        "table_b_t": ("values_add", 0.01),
        "table_mu": ("values", 0.99),
        "paddle_e_g1": ("values", 1.001),
        "paddle_e_g2": ("values", 1.01),
        "paddle_a_t": ("values", 1.01),
        "paddle_b_t": ("values_add", 0.01),
        "paddle_mu": ("values", 0.99),
        "table_length_m": ("geometry", 1.001),
        "table_width_m": ("geometry", 1.001),
        "table_surface_z_m": ("geometry_add", 0.001),
        "net_x_m": ("geometry_add", 0.001),
        "net_height_m": ("geometry", 1.001),
        "net_thickness_m": ("geometry", 1.01),
    }
    rows = []
    for name, (mode, amount) in mutations.items():
        changed_values = dict(base_values)
        changed_geometry = dict(base_geometry)
        if mode == "values":
            changed_values[name] *= amount
        elif mode == "values_add":
            changed_values[name] += amount
        elif mode == "geometry":
            changed_geometry[name] *= amount
        else:
            changed_geometry[name] += amount
        changed = _numeric_signature(changed_values, changed_geometry)
        delta = float(np.max(np.abs(changed - baseline)))
        if not math.isfinite(delta) or delta <= 1.0e-10:
            raise PhysicsEquivalenceError(
                f"mutation witness for {name} has no numeric impact"
            )
        rows.append(
            {
                "parameter": name,
                "mutation_mode": mode,
                "mutation_amount": amount,
                "signature_linf_delta": delta,
                "detected": True,
            }
        )
    return rows


def _torch_runtime_audit(
    root: Path,
    venue_path: Path,
    values: Mapping[str, float],
    *,
    contact_model: Any,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise PhysicsEquivalenceError(
            "formal audit requires Torch to execute the Isaac contact helpers"
        ) from exc
    physical = _load_module(
        "_action_ball_physical_ball_equivalence",
        root / _SOURCE_PATHS["physical_ball"],
    )
    virtual = physical._vb
    prm = virtual.load_venue_params(str(venue_path))
    table_prm = physical.load_venue_table_params(str(venue_path))
    paddle_cases, table_cases = _contact_cases()
    rows = []
    max_v, max_w = 0.0, 0.0
    for kind, cases in (("paddle", paddle_cases), ("table", table_cases)):
        for case in cases:
            tensor = lambda value: torch.tensor(
                [value], dtype=torch.float64, device="cpu"
            )
            if kind == "paddle":
                v_t, w_t = virtual.predict_paddle_contact(
                    tensor(case["v"]),
                    tensor(case["vr"]),
                    tensor(case["n"]),
                    tensor(case["w"]),
                    prm,
                )
                n = _orient_normal(case["n"], case["v"], case["vr"])
                u_n = abs(
                    float(
                        np.dot(
                            np.asarray(case["v"]) - np.asarray(case["vr"]),
                            n,
                        )
                    )
                )
                e_eff = float(
                    np.clip(
                        values["paddle_e_g1"]
                        * math.exp(values["paddle_e_g2"] * u_n),
                        0.05,
                        0.95,
                    )
                )
                a_t, b_t, mu = (
                    values["paddle_a_t"],
                    values["paddle_b_t"],
                    values["paddle_mu"],
                )
            else:
                v_t, w_t = physical.predict_table_contact(
                    tensor(case["v"]), tensor(case["w"]), table_prm
                )
                e_eff, a_t, b_t, mu = (
                    values["table_e_eff"],
                    values["table_a_t"],
                    values["table_b_t"],
                    values["table_mu"],
                )
            result = contact_model.predict_contact(
                np.asarray(case["v"], np.float64)[None, :],
                np.asarray(case["vr"], np.float64)[None, :],
                np.asarray(case["n"], np.float64)[None, :],
                np.asarray(case["w"], np.float64)[None, :],
                e_eff,
                a_t,
                b_t,
                mu,
            )
            v_error = float(
                np.max(
                    np.abs(
                        v_t.detach().cpu().numpy()[0]
                        - np.asarray(result["v_plus"][0])
                    )
                )
            )
            w_error = float(
                np.max(
                    np.abs(
                        w_t.detach().cpu().numpy()[0]
                        - np.asarray(result["omega_plus"][0])
                    )
                )
            )
            tolerance = 2.0e-9 if kind == "paddle" else 2.0e-6
            if max(v_error, w_error) > tolerance:
                raise PhysicsEquivalenceError(
                    f"actual Torch/MuJoCo {case['case_id']} mismatch"
                )
            max_v, max_w = max(max_v, v_error), max(max_w, w_error)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "pair": kind,
                    "velocity_linf_mps": v_error,
                    "spin_linf_radps": w_error,
                    "tolerance": tolerance,
                    "pass": True,
                }
            )
    velocities = torch.tensor(
        [[-4.5, 0.7, -0.4], [2.0, -1.0, 3.0]],
        dtype=torch.float64,
    )
    spins = torch.tensor(
        [[15.0, -20.0, 50.0], [0.0, 30.0, -10.0]],
        dtype=torch.float64,
    )
    actual_accel = virtual.flight_accel(velocities, spins, prm).numpy()
    expected_accel = np.stack(
        [
            _flight_acceleration(v, w, values)
            for v, w in zip(velocities.numpy(), spins.numpy())
        ]
    )
    accel_error = float(np.max(np.abs(actual_accel - expected_accel)))
    if accel_error > 2.0e-12:
        raise PhysicsEquivalenceError(
            "actual Torch/MuJoCo continuous flight acceleration differs"
        )
    return {
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "dtype": "float64",
        "contact_cases": rows,
        "max_velocity_linf_mps": max_v,
        "max_spin_linf_radps": max_w,
        "flight_acceleration_linf_mps2": accel_error,
        "pass": True,
    }


def _validate_fitted_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
    values: Mapping[str, float],
    venue_sha256: str,
    contact_model_sha256: str,
) -> dict[str, Any]:
    if (
        receipt.get("gate")
        != "mujoco_teacher_motion_fitted_ball_gate"
        or receipt.get("contact_authority")
        != "venue_fitted_swept_selected_face_v2"
        or receipt.get("formal_gate_executed") is not True
        or receipt.get("status") != "PASS"
        or receipt.get("verdict") != "PASS"
        or receipt.get("native_ball_contact_enabled") is not False
    ):
        raise PhysicsEquivalenceError(
            "fitted MuJoCo receipt is not a formal code-contact PASS"
        )
    declared_seal = _require_sha256(
        receipt.get("receipt_payload_sha256"),
        "fitted receipt payload seal",
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_payload_sha256")
    if _canonical_sha256(unsigned) != declared_seal:
        raise PhysicsEquivalenceError("fitted receipt payload seal is false")
    venue = receipt.get("venue")
    contact_model = receipt.get("contact_model")
    if (
        not isinstance(venue, Mapping)
        or venue.get("sha256") != venue_sha256
        or not str(venue.get("path", "")).endswith(_SOURCE_PATHS["venue"])
        or not isinstance(contact_model, Mapping)
        or contact_model.get("sha256") != contact_model_sha256
        or not str(contact_model.get("path", "")).endswith(
            _SOURCE_PATHS["contact_model"]
        )
    ):
        raise PhysicsEquivalenceError(
            "fitted receipt venue/contact-model bytes differ"
        )
    scenes = receipt.get("scene_contracts")
    if not isinstance(scenes, Mapping) or set(scenes) != {"0.0010", "0.0005"}:
        raise PhysicsEquivalenceError(
            "fitted receipt lacks exact 1.0/0.5 ms compiled scenes"
        )
    scene_rows = []
    obstacle_sha = None
    expected_inertia = (
        values["inertia_coeff"]
        * values["ball_mass_kg"]
        * values["ball_radius_m"] ** 2
    )
    for key, dt in (("0.0010", 0.001), ("0.0005", 0.0005)):
        scene = scenes[key]
        if not isinstance(scene, Mapping):
            raise PhysicsEquivalenceError(f"scene_contracts[{key}] is malformed")
        gravity = scene.get("gravity_mps2")
        if (
            scene.get("ball_native_contact_disabled") is not True
            or scene.get("timestep_s") != dt
            or gravity != [0.0, 0.0, -values["gravity_mps2"]]
        ):
            raise PhysicsEquivalenceError(
                f"fitted compiled scene {key} dynamics differ"
            )
        for field, expected in (
            ("ball_mass_kg", values["ball_mass_kg"]),
            ("ball_radius_m", values["ball_radius_m"]),
            ("ball_diagonal_inertia_kg_m2", expected_inertia),
        ):
            _assert_close(scene.get(field), expected, f"scene {key} {field}")
        digest = _require_sha256(
            scene.get("obstacle_geometry_sha256"),
            f"scene {key} obstacle geometry",
        )
        if obstacle_sha is None:
            obstacle_sha = digest
        elif digest != obstacle_sha:
            raise PhysicsEquivalenceError(
                "fitted scene obstacle geometry differs across timesteps"
            )
        scene_rows.append(
            {
                "dt_s": dt,
                "ball_native_contact_disabled": True,
                "ball_mass_kg": scene["ball_mass_kg"],
                "ball_radius_m": scene["ball_radius_m"],
                "ball_diagonal_inertia_kg_m2": scene[
                    "ball_diagonal_inertia_kg_m2"
                ],
                "gravity_mps2": gravity,
                "obstacle_geometry_sha256": digest,
                "pass": True,
            }
        )
    actions = receipt.get("actions")
    if not isinstance(actions, list) or not actions:
        raise PhysicsEquivalenceError("fitted receipt contains no action results")
    action_rows = []
    for index, action in enumerate(actions):
        if (
            not isinstance(action, Mapping)
            or action.get("verdict") != "PASS"
            or action.get("failure_reasons") != []
        ):
            raise PhysicsEquivalenceError(
                f"fitted action {index} is not a PASS"
            )
        binding = action.get("physical_task_binding")
        cases = binding.get("cases") if isinstance(binding, Mapping) else None
        if not isinstance(cases, list) or len(cases) != 6:
            raise PhysicsEquivalenceError(
                f"fitted action {index} lacks six physical controls"
            )
        for case in cases:
            dt_results = case.get("dt_results") if isinstance(case, Mapping) else None
            if not isinstance(dt_results, Mapping) or set(dt_results) != {
                "0.0010",
                "0.0005",
            }:
                raise PhysicsEquivalenceError(
                    f"fitted action {index} case lacks both dt results"
                )
        action_rows.append(
            {
                "action_id": action.get("action_id"),
                "case_count": len(cases),
                "all_cases_both_dt": True,
            }
        )
    return {
        "path_sha256": receipt_sha256,
        "payload_sha256": declared_seal,
        "scene_contracts": scene_rows,
        "actions": action_rows,
        "native_mujoco_pair_material_mapping_claimed": False,
        "ball_contact_authority": "code_driven_venue_impulse_only",
        "pass": True,
    }


def _git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    if _GIT_SHA.fullmatch(expected_commit) is None:
        raise PhysicsEquivalenceError("--code-commit must be one full Git SHA-1")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PhysicsEquivalenceError(f"cannot inspect Git checkout: {exc}") from exc
    if head != expected_commit or status.strip():
        raise PhysicsEquivalenceError(
            "formal audit requires the exact clean requested commit"
        )
    return {"code_commit": head, "clean": True}


def build_audit(
    *,
    repo_root: Path,
    profile_pins_path: Path,
    profile_pins_sha256: str,
    fitted_receipt_path: Path | None,
    fitted_receipt_sha256: str | None,
    formal: bool,
    code_commit: str | None,
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    snapshots = _snapshot_sources(root)
    venue_path, venue_raw = _plain_file(
        Path(_SOURCE_PATHS["venue"]), root=root, label="venue YAML"
    )
    venue_sha = _sha256_bytes(venue_raw)
    venue = _strict_yaml(venue_raw, "venue YAML")
    values = _venue_values(venue)
    profile_path, profile_raw = _plain_file(
        profile_pins_path, root=root, label="profile pins"
    )
    profile_sha = _sha256_bytes(profile_raw)
    if profile_sha != _require_sha256(
        profile_pins_sha256, "profile-pins SHA-256"
    ):
        raise PhysicsEquivalenceError("profile-pins bytes changed")
    profile = _strict_json(profile_raw, "profile pins")
    # Reuse the native/fitted teacher gate's one formal profile validator.
    # The previous audit only reopened physics_payload, so a legacy four-file
    # source map, WORKTREE pseudo-revision, missing source authority, or
    # drifted solver bytes could still produce a green source-contract row.
    validated_profile = _validate_formal_profile_pins(root, profile)
    profile_payload, profile_rows = _profile_values(
        profile,
        profile_sha256=profile_sha,
        venue_sha256=venue_sha,
        validated_profile=validated_profile,
    )
    geometry = _load_module(
        "_action_ball_equivalence_geometry",
        root / _SOURCE_PATHS["geometry"],
    )
    parameter_rows, geometry_values = _parameter_ledger(
        values, profile_payload, geometry
    )
    contact_model = _load_module(
        "_action_ball_equivalence_contact_model",
        root / _SOURCE_PATHS["contact_model"],
    )
    contact = _numpy_contact_audit(values, contact_model=contact_model)
    dual_dt = _dual_dt_audit(values)
    mutations = _mutation_witnesses(values, geometry_values)
    source_contracts = _source_contracts(root, snapshots)

    torch_result = None
    torch_error = None
    try:
        torch_result = _torch_runtime_audit(
            root,
            venue_path,
            values,
            contact_model=contact_model,
        )
    except PhysicsEquivalenceError as exc:
        torch_error = str(exc)
        if formal:
            raise

    fitted_result = None
    fitted_error = None
    if fitted_receipt_path is not None:
        if fitted_receipt_sha256 is None:
            raise PhysicsEquivalenceError(
                "fitted receipt path requires its preregistered SHA-256"
            )
        fitted_path, fitted_raw = _plain_file(
            fitted_receipt_path, root=root, label="fitted MuJoCo receipt"
        )
        actual_fitted_sha = _sha256_bytes(fitted_raw)
        if actual_fitted_sha != _require_sha256(
            fitted_receipt_sha256, "fitted receipt SHA-256"
        ):
            raise PhysicsEquivalenceError("fitted receipt bytes changed")
        fitted_result = _validate_fitted_receipt(
            _strict_json(fitted_raw, "fitted MuJoCo receipt"),
            receipt_sha256=actual_fitted_sha,
            values=values,
            venue_sha256=venue_sha,
            contact_model_sha256=snapshots["contact_model"]["sha256"],
        )
    elif formal:
        raise PhysicsEquivalenceError(
            "formal audit requires a fitted-MuJoCo PASS receipt"
        )
    else:
        fitted_error = "fitted-MuJoCo PASS receipt not supplied"

    git_identity = None
    if formal:
        if code_commit is None:
            raise PhysicsEquivalenceError(
                "formal audit requires --code-commit"
            )
        git_identity = _git_identity(root, code_commit)

    formal_pass = bool(
        formal
        and torch_result is not None
        and fitted_result is not None
        and git_identity is not None
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_class": RECEIPT_CLASS,
        "status": "PASS" if formal_pass else "DIAGNOSTIC_BLOCKED",
        "verdict": "PASS" if formal_pass else "BLOCKED",
        "formal": bool(formal),
        "identity": {
            "git": git_identity,
            "profile_pins": {
                "path": profile_path.relative_to(root).as_posix(),
                "sha256": profile_sha,
                "physics_profile_sha256": profile[
                    "physics_profile_sha256"
                ],
            },
            "venue": {
                "path": venue_path.relative_to(root).as_posix(),
                "sha256": venue_sha,
            },
            "sources": snapshots,
        },
        "parameter_ledger": parameter_rows,
        "profile_binding": profile_rows,
        "geometry": geometry_values,
        "source_contracts": source_contracts,
        "contact_equivalence": contact,
        "torch_isaac_runtime_equivalence": torch_result,
        "torch_blocker": torch_error,
        "dual_dt_flight": dual_dt,
        "fitted_mujoco_runtime_evidence": fitted_result,
        "fitted_receipt_blocker": fitted_error,
        "mutation_witnesses": mutations,
        "authority": {
            "isaac_ball_native_collision_enabled": False,
            "mujoco_ball_native_collision_enabled": False,
            "contact_authority": "code_driven_venue_impulse",
            "native_mujoco_solref_solimp_mapping_claimed": False,
            "spin_decay_per_s": 0.0,
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "native_mujoco_material_equivalence",
            "policy_return_success",
            "deployment_authorization",
            "hardware_authorization",
        ],
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    return receipt


def _exclusive_write(path: Path, document: Mapping[str, Any]) -> str:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    payload = _canonical_json_bytes(document) + b"\n"
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _sha256_bytes(payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile-pins", type=Path, required=True)
    parser.add_argument("--profile-pins-sha256", required=True)
    parser.add_argument("--fitted-receipt", type=Path)
    parser.add_argument("--fitted-receipt-sha256")
    parser.add_argument("--code-commit")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = build_audit(
            repo_root=args.repo_root,
            profile_pins_path=args.profile_pins,
            profile_pins_sha256=args.profile_pins_sha256,
            fitted_receipt_path=args.fitted_receipt,
            fitted_receipt_sha256=args.fitted_receipt_sha256,
            formal=not args.diagnostic,
            code_commit=args.code_commit,
        )
        output_sha = _exclusive_write(args.out, receipt)
    except (PhysicsEquivalenceError, OSError, ValueError) as exc:
        print(
            f"[cross-engine-physics-audit][FAIL] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        f"[cross-engine-physics-audit] {receipt['verdict']} "
        f"receipt={args.out} file_sha256={output_sha}"
    )
    return 0 if receipt["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
