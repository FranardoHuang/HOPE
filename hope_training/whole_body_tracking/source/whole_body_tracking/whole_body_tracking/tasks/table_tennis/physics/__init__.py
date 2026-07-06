"""Experimentally-calibrated, spin-aware ball physics (flight + contact + landing).

Pure ``torch`` + ``PyYAML`` (no Isaac imports) so it can be imported and unit-tested anywhere. The
fitted constants come from the shared ``configs/ball_physics_venue.yaml`` (see :mod:`.params`). The numpy
reference in ``Record/analysis/{flight_model,contact_model}`` is the regression oracle.
"""

from __future__ import annotations

from .flight import flight_accel, rk4_step
from .landing import LandingResult, predict_landing
from .params import (
    BallParams,
    BallPhysicsConfig,
    ContactParams,
    FlightParams,
    default_yaml_path,
    get_ball_physics,
    load_ball_physics,
)
from .spin_contact import orient_normal, predict_contact

__all__ = [
    "BallParams",
    "BallPhysicsConfig",
    "ContactParams",
    "FlightParams",
    "LandingResult",
    "default_yaml_path",
    "flight_accel",
    "get_ball_physics",
    "load_ball_physics",
    "orient_normal",
    "predict_contact",
    "predict_landing",
    "rk4_step",
]
