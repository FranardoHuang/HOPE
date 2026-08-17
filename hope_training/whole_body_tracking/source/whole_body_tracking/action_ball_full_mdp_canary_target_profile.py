#!/usr/bin/env python3
"""Code-owned target support for the N=2 no-save Full-MDP canary.

This is a diagnostic task configuration, not a safety or launch authority.
It converts the production ``ContinuousQuestionCfg`` default aim *box* into
three deterministic canary points and binds their tracking-frame provenance
to the canonical table geometry and HOPE-to-tracking translation.  The
current solved-training path grades one configured point, so this finite box
support must not be described as the formal training target distribution.

The resulting frame digest only proves which code-owned configuration values
were materialized.  It is not independent physical evidence, contact
evidence, or permission to train, checkpoint, promote, or control hardware.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple

import action_ball_device_profile_authority as _profile_authority


SCHEMA_VERSION = 1
KIND = "action_ball_full_mdp_n2_no_save_canary_target_profile_v1"
FRAME_ID = "tracking_env_table_xy_m"
CANARY_NUM_ENVS = 2
CANARY_SAVE_CHECKPOINTS = False

INTEGRATION_STATUS = "n2_no_save_diagnostic_config_constructible"
DIAGNOSTIC_UNAUTHORIZED = True
FORMAL_PROFILE = False
FORMAL_LAUNCH_AUTHORIZED = False
PRODUCTION_INTEGRATED = False
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

CELL_IDS: Tuple[str, ...] = (
    "canary_aim_box_center",
    "canary_aim_box_near_right_quartile",
    "canary_aim_box_deep_left_quartile",
)

_CQ_SOURCE = (
    "whole_body_tracking.tasks.tracking.mdp.continuous_questions."
    "ContinuousQuestionCfg.aim_x_range/aim_y_range"
)
_TRACKING_SOURCE = (
    "whole_body_tracking.tasks.tracking.mdp.hope_commands."
    "RacketTargetCommand.cfg.vb_table_near_x/vb_table_surface_z"
)
_GEOMETRY_SOURCE = (
    "whole_body_tracking.tasks.table_tennis.geometry."
    "TABLE_LENGTH/TABLE_WIDTH/NET_X"
)
_TRANSLATION_SOURCE = (
    "whole_body_tracking.tasks.table_tennis.table_frame.env_frame_offset"
)


class ActionBallFullMdpCanaryProfileError(RuntimeError):
    """The production defaults cannot define the diagnostic canary support."""


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ActionBallFullMdpCanaryProfileError(
            f"{label} must be a finite production code default"
        )
    return float(value)


def _range_pair(value: object, *, label: str) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise ActionBallFullMdpCanaryProfileError(
            f"{label} must be one exact two-value tuple"
        )
    lower = _finite_float(value[0], label=f"{label}[0]")
    upper = _finite_float(value[1], label=f"{label}[1]")
    if not lower < upper:
        raise ActionBallFullMdpCanaryProfileError(
            f"{label} must be a nondegenerate ordered box edge"
        )
    return lower, upper


def _production_defaults(*, racket_cfg: object | None = None) -> dict[str, object]:
    # Imports are lazy so the authority module remains dependency-light.  A
    # real factory already runs inside the Isaac task package and therefore
    # resolves these exact production classes, not a YAML or test fixture.
    from whole_body_tracking.tasks.table_tennis import geometry
    from whole_body_tracking.tasks.table_tennis import table_frame
    from whole_body_tracking.tasks.tracking.mdp.continuous_questions import (
        ContinuousQuestionCfg,
    )
    from whole_body_tracking.tasks.tracking.mdp.hope_commands import (
        RacketTargetCommandCfg,
    )

    if racket_cfg is None:
        # Dependency-light numeric tests may materialize the code default.  The
        # real factory always supplies the exact cfg retained by its already
        # constructed Racket owner; this fallback is not a runtime authority.
        racket_cfg = RacketTargetCommandCfg()
    if type(racket_cfg) is not RacketTargetCommandCfg:
        raise ActionBallFullMdpCanaryProfileError(
            "racket_cfg must be the exact constructed RacketTargetCommandCfg"
        )
    cq_cfg = ContinuousQuestionCfg()
    aim_x = _range_pair(cq_cfg.aim_x_range, label="ContinuousQuestionCfg.aim_x_range")
    aim_y = _range_pair(cq_cfg.aim_y_range, label="ContinuousQuestionCfg.aim_y_range")
    near_x = _finite_float(
        racket_cfg.vb_table_near_x,
        label="RacketTargetCommand.cfg.vb_table_near_x",
    )
    surface_z = _finite_float(
        racket_cfg.vb_table_surface_z,
        label="RacketTargetCommand.cfg.vb_table_surface_z",
    )
    table_length = _finite_float(geometry.TABLE_LENGTH, label="geometry.TABLE_LENGTH")
    table_width = _finite_float(geometry.TABLE_WIDTH, label="geometry.TABLE_WIDTH")
    hope_net_x = _finite_float(geometry.NET_X, label="geometry.NET_X")
    offset = table_frame.env_frame_offset(near_x, surface_z)
    expected_offset = (near_x, table_width / 2.0, surface_z)
    if (
        type(offset) is not tuple
        or len(offset) != 3
        or tuple(float(item) for item in offset) != expected_offset
    ):
        raise ActionBallFullMdpCanaryProfileError(
            "canonical HOPE-to-tracking translation differs"
        )
    half_width = table_width / 2.0
    net_x = near_x + hope_net_x
    far_x = near_x + table_length
    if not (
        near_x < net_x < aim_x[0] < aim_x[1] <= far_x
        and -half_width <= aim_y[0] < aim_y[1] <= half_width
    ):
        raise ActionBallFullMdpCanaryProfileError(
            "ContinuousQuestionCfg aim box is not inside the opponent table half"
        )
    return {
        "aim_x_range_m": aim_x,
        "aim_y_range_m": aim_y,
        "table_near_x_m": near_x,
        "table_far_x_m": far_x,
        "table_half_width_m": half_width,
        "table_surface_z_m": surface_z,
        "net_x_m": net_x,
        "hope_to_tracking_translation_m": expected_offset,
    }


def canary_target_profile_provenance_payload(
    *, racket_cfg: object | None = None
) -> Mapping[str, object]:
    """Return the canonical config-provenance payload for the canary frame.

    The payload is human/audit readable.  Its digest is an identity binding,
    not a second writer or an independent safety assertion.
    """

    values = _production_defaults(racket_cfg=racket_cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "authorization_semantics": (
            "diagnostic_config_provenance_only_not_independent_safety_evidence"
        ),
        "frame": {
            "frame_id": FRAME_ID,
            "origin": "tracking_environment_local_origin_on_floor",
            "axes": (
                "x_forward_toward_player_two",
                "y_left_from_player_one",
            ),
            "components": ("landing_x_m", "landing_y_m"),
            "hope_origin": "near_side_left_corner_of_table_surface",
            "hope_to_tracking_translation_m": values[
                "hope_to_tracking_translation_m"
            ],
        },
        "venue": {
            "table_near_x_m": values["table_near_x_m"],
            "table_far_x_m": values["table_far_x_m"],
            "table_half_width_m": values["table_half_width_m"],
            "table_surface_z_m": values["table_surface_z_m"],
            "net_x_m": values["net_x_m"],
        },
        "aim_box": {
            "x_range_m": values["aim_x_range_m"],
            "y_range_m": values["aim_y_range_m"],
        },
        "direct_code_sources": (
            _CQ_SOURCE,
            _TRACKING_SOURCE,
            _GEOMETRY_SOURCE,
            _TRANSLATION_SOURCE,
        ),
    }


def canary_target_profile_frame_binding_sha256(
    *, racket_cfg: object | None = None
) -> str:
    """Hash exact config provenance; do not interpret it as physical proof."""

    payload = canary_target_profile_provenance_payload(racket_cfg=racket_cfg)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_action_ball_full_mdp_canary_target_profile(
    *, racket_cfg: object | None = None
) -> _profile_authority.FrozenDeviceTargetProfileSpec:
    """Build the code-owned three-point N=2 no-save diagnostic support."""

    values = _production_defaults(racket_cfg=racket_cfg)
    x_lo, x_hi = values["aim_x_range_m"]
    y_lo, y_hi = values["aim_y_range_m"]
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    targets = (
        (x_lo + 0.5 * x_span, y_lo + 0.5 * y_span),
        (x_lo + 0.25 * x_span, y_lo + 0.25 * y_span),
        (x_lo + 0.75 * x_span, y_lo + 0.75 * y_span),
    )
    return _profile_authority.freeze_device_target_profile_spec(
        frame_id=FRAME_ID,
        frame_binding_sha256=canary_target_profile_frame_binding_sha256(
            racket_cfg=racket_cfg
        ),
        cell_ids=CELL_IDS,
        targets_xy_m=targets,
    )


__all__ = [
    "ActionBallFullMdpCanaryProfileError",
    "CANARY_NUM_ENVS",
    "CANARY_SAVE_CHECKPOINTS",
    "CELL_IDS",
    "DIAGNOSTIC_UNAUTHORIZED",
    "FORMAL_LAUNCH_AUTHORIZED",
    "FORMAL_PROFILE",
    "FRAME_ID",
    "INTEGRATION_STATUS",
    "KIND",
    "LAUNCH_AUTHORIZED",
    "PRODUCTION_INTEGRATED",
    "RUNTIME_INTEGRATED",
    "build_action_ball_full_mdp_canary_target_profile",
    "canary_target_profile_frame_binding_sha256",
    "canary_target_profile_provenance_payload",
]
