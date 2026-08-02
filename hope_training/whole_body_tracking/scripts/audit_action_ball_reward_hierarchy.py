#!/usr/bin/env python3
"""Host-only arithmetic audit for the A3 Vendor V2 reward hierarchy.

This is a configuration consequence, not a claim that PPO has learned the task.  It proves the
declared support/budget ordering and evaluates the old/new kernels on a frozen observed error row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


POLICY_DT_S = 0.02
BODY_MOTION_WEIGHT_SUM = 4.5  # four body kernels + global anchor orientation
# Exact cap from phi(d)=clamp(d,0,4.65m), weight=10 and policy_dt=.02.  Because reward is
# phi(previous)-phi(current), the undiscounted sum telescopes and closed loops cannot mint income.
PROGRESS_UPPER = 0.93


class HierarchyAuditError(ValueError):
    pass


def _positive_number(mapping, key: str, *, allow_zero: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise HierarchyAuditError(f"{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HierarchyAuditError(f"{key} must be numeric") from exc
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        relation = ">= 0" if allow_zero else "> 0"
        raise HierarchyAuditError(f"{key} must be finite and {relation}")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _single_default_parent(document: dict, *, expected: str, source: Path) -> None:
    defaults = document.get("defaults")
    if not isinstance(defaults, list):
        raise HierarchyAuditError(f"{source.name} must declare a defaults list")
    parents = [
        str(row).split("@", 1)[0]
        for row in defaults
        if isinstance(row, str) and row != "_self_"
    ]
    if parents != [expected]:
        raise HierarchyAuditError(
            f"{source.name} must inherit exactly {expected!r}, got {parents}"
        )


def _system_recipe_contract(task_path: Path, document: dict) -> dict:
    """Prove that the numeric leaf is attached to the complete ActionBall lineage."""

    _single_default_parent(
        document,
        expected="HOPEPingPongActionBallA3VendorV1",
        source=task_path,
    )
    vendor_v1_path = task_path.with_name("HOPEPingPongActionBallA3VendorV1.yaml")
    action_ball_path = task_path.with_name("HOPEPingPongActionBall.yaml")
    vendor_v1 = yaml.safe_load(vendor_v1_path.read_text())
    action_ball = yaml.safe_load(action_ball_path.read_text())
    _single_default_parent(
        vendor_v1,
        expected="HOPEPingPongActionBall",
        source=vendor_v1_path,
    )

    rewards = document.get("rewards") or {}
    racket = document.get("racket") or {}
    action_ball_rewards = action_ball.get("rewards") or {}
    action_ball_racket = action_ball.get("racket") or {}
    checks = {
        "full_body_mimic": rewards.get("full_body_mimic") is True,
        "measured_racket_teacher": (
            racket.get("motion_teacher_racket_source") == "measured_channel"
        ),
        "three_channel_monotonic_adaptive_fine": (
            racket.get("adaptive_sigma") is True
            and racket.get("adaptive_sigma_monotonic") is True
            and racket.get("adaptive_sigma_normal") is True
            and racket.get("adaptive_sigma_source") == "ball_exact_strike"
        ),
        "action_ball_target_mode": action_ball_racket.get("target_mode") == "action_ball",
        "ball_outcome_enabled": action_ball_racket.get("virtual_ball") is True,
        "table_obstacle_enabled": action_ball.get("table_obstacle") is True,
        "complete_reward_pack": action_ball_rewards.get("reward_pack") == "v2",
    }
    if not all(checks.values()):
        raise HierarchyAuditError(f"incomplete one-run ActionBall recipe: {checks}")
    return {
        "checks": checks,
        "vendor_parent_sha256": _sha256(vendor_v1_path),
        "action_ball_parent_sha256": _sha256(action_ball_path),
        "interpretation": (
            "the successor restores full-body mimic and inherits the ball target, outcome, "
            "table, and reward pack from the complete ActionBall lineage; Stage1 is not a "
            "runtime stage"
        ),
    }


def cauchy(error: float, scale: float) -> float:
    return 1.0 / (1.0 + (float(error) / float(scale)) ** 2)


def gaussian(error: float, scale: float) -> float:
    return math.exp(-((float(error) / float(scale)) ** 2))


def _inclusive_window_steps(half_window_s: float) -> int:
    ticks = int(round(float(half_window_s) / POLICY_DT_S))
    if not math.isclose(ticks * POLICY_DT_S, float(half_window_s), abs_tol=1.0e-12):
        raise HierarchyAuditError("strike windows must lie on the 50 Hz policy clock")
    return 2 * ticks + 1


def _default_action_catalog(task_path: Path) -> Path:
    return task_path.parents[4] / "assets/motions/chingmu73_20260728/CLIP_ORDER.json"


def _action_steps(action_catalog_path: Path) -> dict[str, int]:
    document = json.loads(action_catalog_path.read_text())
    clips = document.get("clips")
    if not isinstance(clips, list) or int(document.get("n_clips", -1)) != 73 or len(clips) != 73:
        raise HierarchyAuditError("reward hierarchy audit requires the exact 73-action catalog")
    result = {}
    for row in clips:
        uid = str(row.get("uid", ""))
        try:
            frames = int(row.get("T"))
        except (TypeError, ValueError) as exc:
            raise HierarchyAuditError(f"catalog action {uid!r} has invalid T") from exc
        if not uid or uid in result or frames <= 0:
            raise HierarchyAuditError(f"catalog action id/frame contract is invalid: {uid!r}/{frames}")
        result[uid] = frames
    return result


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    blend = position - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def build_audit(
    task_path: Path,
    *,
    observed_errors: tuple[float, float, float],
    action_catalog_path: Path | None = None,
) -> dict:
    document = yaml.safe_load(task_path.read_text())
    system_recipe = _system_recipe_contract(task_path, document)
    rewards = document.get("rewards") or {}
    racket = document.get("racket") or {}
    if rewards.get("racket_coarse_kernel") != "cauchy":
        raise HierarchyAuditError("V2 audit requires racket_coarse_kernel=cauchy")
    motion_scale = _positive_number(rewards, "motion_scale")
    paddle_weights = [
        _positive_number(rewards, f"motion_racket_{name}_weight")
        for name in ("position", "velocity", "normal", "long_axis")
    ]
    coarse_weights = [
        _positive_number(rewards, f"racket_{name}_coarse_weight")
        for name in ("position", "velocity", "normal")
    ]
    coarse_scales = [
        _positive_number(rewards, f"racket_{name}_coarse_std")
        for name in ("position", "velocity", "normal")
    ]
    adaptive_weights = [
        _positive_number(rewards, f"racket_{name}_weight")
        for name in ("position", "velocity", "normal")
    ]
    adaptive_start_scales = [
        _positive_number(rewards, f"racket_{name}_std")
        for name in ("position", "velocity", "normal")
    ]
    precision_weights = [
        _positive_number(rewards, f"racket_{name}_precision_weight")
        for name in ("position", "velocity", "normal")
    ]
    precision_scales = [
        _positive_number(rewards, f"racket_{name}_precision_std")
        for name in ("position", "velocity", "normal")
    ]
    position_half_window_s = _positive_number(racket, "strike_window_pos_s")
    wide_half_window_s = _positive_number(racket, "strike_window_wide_s")
    position_window_steps = _inclusive_window_steps(position_half_window_s)
    wide_window_steps = _inclusive_window_steps(wide_half_window_s)
    if position_window_steps >= wide_window_steps:
        raise HierarchyAuditError("position strike window must be tighter than the wide window")
    adaptive_bounds = {}
    for channel, acceptance_scale, start_scale in zip(
        ("pos", "vel", "normal"), precision_scales, adaptive_start_scales
    ):
        lower = _positive_number(racket, f"sigma_{channel}_min")
        upper = _positive_number(racket, f"sigma_{channel}_max")
        if not math.isclose(lower, acceptance_scale, rel_tol=0.0, abs_tol=1.0e-12):
            raise HierarchyAuditError(
                f"sigma_{channel}_min must equal the fixed precision acceptance width"
            )
        if not math.isclose(upper, start_scale, rel_tol=0.0, abs_tol=1.0e-12):
            raise HierarchyAuditError(
                f"racket {channel} adaptive std must start at sigma_{channel}_max"
            )
        if upper < lower:
            raise HierarchyAuditError(f"sigma_{channel}_max must be >= sigma_{channel}_min")
        adaptive_bounds[channel] = {"min": lower, "max": upper}
    adaptive_max_scales = [
        adaptive_bounds[channel]["max"] for channel in ("pos", "vel", "normal")
    ]
    channel_window_steps = [position_window_steps, wide_window_steps, wide_window_steps]
    motion_racket_window_scale = _positive_number(
        rewards, "motion_racket_scale_in_strike_window", allow_zero=True
    )
    if motion_racket_window_scale > 1.0:
        raise HierarchyAuditError("motion_racket_scale_in_strike_window must be <= 1")
    motion_racket_long_axis_window_scale = _positive_number(
        rewards,
        "motion_racket_long_axis_scale_in_strike_window",
        allow_zero=True,
    )
    if motion_racket_long_axis_window_scale > 1.0:
        raise HierarchyAuditError(
            "motion_racket_long_axis_scale_in_strike_window must be <= 1"
        )
    landing_weight = _positive_number(rewards, "virtual_landing_weight")
    landing_base_frac = _positive_number(rewards, "virtual_landing_base_frac")
    if landing_base_frac > 1.0:
        raise HierarchyAuditError("virtual_landing_base_frac must be <= 1")

    action_catalog_path = (
        _default_action_catalog(task_path)
        if action_catalog_path is None
        else action_catalog_path
    )
    action_steps = _action_steps(action_catalog_path)
    action_caps = {}
    for action, steps in action_steps.items():
        body_cap = steps * POLICY_DT_S * BODY_MOTION_WEIGHT_SUM * motion_scale
        teacher_paid_steps = steps - min(steps, wide_window_steps) * (
            1.0 - motion_racket_window_scale
        )
        long_axis_paid_steps = steps - min(steps, wide_window_steps) * (
            1.0 - motion_racket_long_axis_window_scale
        )
        paddle_cap = POLICY_DT_S * (
            teacher_paid_steps * sum(paddle_weights[:3])
            + long_axis_paid_steps * paddle_weights[3]
        )
        action_caps[action] = body_cap + paddle_cap
    broad_envelope_floor = sum(
        steps * POLICY_DT_S * 0.5 * weight
        for steps, weight in zip(channel_window_steps, coarse_weights)
    )
    target_at_fine_acceptance = sum(
        steps
        * POLICY_DT_S
        * (
            coarse_weight * cauchy(acceptance_scale, coarse_scale)
            + adaptive_weight * gaussian(acceptance_scale, final_adaptive_scale)
            + precision_weight * gaussian(acceptance_scale, acceptance_scale)
        )
        for (
            steps,
            coarse_weight,
            coarse_scale,
            adaptive_weight,
            final_adaptive_scale,
            precision_weight,
            acceptance_scale,
        ) in zip(
            channel_window_steps,
            coarse_weights,
            coarse_scales,
            adaptive_weights,
            [adaptive_bounds[channel]["min"] for channel in ("pos", "vel", "normal")],
            precision_weights,
            precision_scales,
        )
    )
    target_at_fine_acceptance_initial = sum(
        steps
        * POLICY_DT_S
        * (
            coarse_weight * cauchy(acceptance_scale, coarse_scale)
            + adaptive_weight * gaussian(acceptance_scale, adaptive_max_scale)
            + precision_weight * gaussian(acceptance_scale, acceptance_scale)
        )
        for (
            steps,
            coarse_weight,
            coarse_scale,
            adaptive_weight,
            adaptive_max_scale,
            precision_weight,
            acceptance_scale,
        ) in zip(
            channel_window_steps,
            coarse_weights,
            coarse_scales,
            adaptive_weights,
            adaptive_max_scales,
            precision_weights,
            precision_scales,
        )
    )
    target_kernel_max = sum(
        steps * POLICY_DT_S * (coarse_weight + adaptive_weight + precision_weight)
        for steps, coarse_weight, adaptive_weight, precision_weight in zip(
            channel_window_steps, coarse_weights, adaptive_weights, precision_weights
        )
    )
    landing_min = landing_weight * POLICY_DT_S * landing_base_frac
    landing_max = landing_weight * POLICY_DT_S

    action_rows = {}
    for action, action_cap in action_caps.items():
        target_upper = target_kernel_max + PROGRESS_UPPER
        action_rows[action] = {
            "frames": action_steps[action],
            "action_prior_undiscounted_cap": action_cap,
            "ball_target_broad_envelope_floor": broad_envelope_floor,
            "ball_target_income_at_fine_acceptance": target_at_fine_acceptance,
            "ball_target_initial_income_at_fine_acceptance": (
                target_at_fine_acceptance_initial
            ),
            "ball_target_kernel_plus_progress_upper": target_upper,
            "legal_landing_event_min": landing_min,
            "legal_landing_event_max": landing_max,
            "strict_order_at_fine_acceptance": (
                action_cap < target_at_fine_acceptance < landing_min
            ),
            "target_theoretical_upper_below_landing_min": target_upper < landing_min,
        }

    pos_error, vel_error, normal_error = observed_errors
    old = POLICY_DT_S * (
        position_window_steps
        * (4.0 * gaussian(pos_error, 0.075) + 1.0 * gaussian(pos_error, 0.30))
        + wide_window_steps
        * (
            0.5 * gaussian(vel_error, 0.50)
            + 0.5 * gaussian(normal_error, 0.262)
        )
    )
    new_components = [
        steps
        * POLICY_DT_S
        * weight
        * cauchy(error, scale)
        for error, weight, scale, steps in zip(
            observed_errors, coarse_weights, coarse_scales, channel_window_steps
        )
    ]
    adaptive_final_components = [
        steps * POLICY_DT_S * weight * gaussian(error, scale)
        for steps, weight, error, scale in zip(
            channel_window_steps,
            adaptive_weights,
            observed_errors,
            [adaptive_bounds[channel]["min"] for channel in ("pos", "vel", "normal")],
        )
    ]
    adaptive_initial_components = [
        steps * POLICY_DT_S * weight * gaussian(error, scale)
        for steps, weight, error, scale in zip(
            channel_window_steps, adaptive_weights, observed_errors, adaptive_max_scales
        )
    ]
    precision_components = [
        steps * POLICY_DT_S * weight * gaussian(error, scale)
        for steps, weight, error, scale in zip(
            channel_window_steps, precision_weights, observed_errors, precision_scales
        )
    ]
    new_final = (
        sum(new_components) + sum(adaptive_final_components) + sum(precision_components)
    )
    new_initial = (
        sum(new_components) + sum(adaptive_initial_components) + sum(precision_components)
    )
    all_pass = all(
        row["strict_order_at_fine_acceptance"]
        and row["target_theoretical_upper_below_landing_min"]
        for row in action_rows.values()
    )
    return {
        "schema_version": 1,
        "kind": "action_ball_reward_hierarchy_counterfactual_v1",
        "authorization": {
            "training": False,
            "promotion": False,
            "diagnostic_unauthorized": True,
        },
        "task_profile": str(task_path),
        "task_profile_sha256": _sha256(task_path),
        "system_recipe": system_recipe,
        "action_catalog": str(action_catalog_path),
        "action_catalog_sha256": _sha256(action_catalog_path),
        "constants": {
            "policy_dt_s": POLICY_DT_S,
            "position_half_window_s": position_half_window_s,
            "wide_half_window_s": wide_half_window_s,
            "channel_window_steps": {
                "position": position_window_steps,
                "velocity": wide_window_steps,
                "normal": wide_window_steps,
            },
            "adaptive_sigma_bounds": adaptive_bounds,
            "motion_racket_scale_in_strike_window": motion_racket_window_scale,
            "motion_racket_long_axis_scale_in_strike_window": (
                motion_racket_long_axis_window_scale
            ),
            "body_motion_weight_sum_before_scale": BODY_MOTION_WEIGHT_SUM,
            "adaptive_target_weight_sum": sum(adaptive_weights),
            "adaptive_target_start_scales": adaptive_start_scales,
            "precision_target_weight_sum": sum(precision_weights),
            "precision_target_scales": precision_scales,
            "landing_base_frac": landing_base_frac,
        },
        "actions": action_rows,
        "catalog_summary": {
            "action_count": len(action_rows),
            "action_prior_cap_p50": _percentile(list(action_caps.values()), 50.0),
            "action_prior_cap_p95": _percentile(list(action_caps.values()), 95.0),
            "action_prior_cap_max": max(action_caps.values()),
            "longest_action": max(action_steps, key=action_steps.get),
            "longest_action_frames": max(action_steps.values()),
            "target_income_at_fine_acceptance": target_at_fine_acceptance,
            "target_initial_income_at_fine_acceptance": (
                target_at_fine_acceptance_initial
            ),
            "broad_one_sigma_income": broad_envelope_floor,
            "target_kernel_max": target_kernel_max,
            "target_kernel_plus_progress_upper": target_kernel_max + PROGRESS_UPPER,
            "all_actions_strict_order_at_fine_acceptance": all(
                row["strict_order_at_fine_acceptance"] for row in action_rows.values()
            ),
        },
        "frozen_observed_exact_strike_counterfactual": {
            "errors": {
                "position_m": pos_error,
                "velocity_mps": vel_error,
                "normal_rad": normal_error,
            },
            "v1_window_kernel_income": old,
            "v2_window_kernel_income": new_final,
            "v2_window_kernel_income_at_final_sigma": new_final,
            "v2_window_kernel_income_at_initial_sigma": new_initial,
            "v2_broad_component_income": {
                "position": new_components[0],
                "velocity": new_components[1],
                "normal": new_components[2],
            },
            "v2_adaptive_fine_component_income_at_final_sigma": {
                "position": adaptive_final_components[0],
                "velocity": adaptive_final_components[1],
                "normal": adaptive_final_components[2],
            },
            "v2_adaptive_fine_component_income_at_initial_sigma": {
                "position": adaptive_initial_components[0],
                "velocity": adaptive_initial_components[1],
                "normal": adaptive_initial_components[2],
            },
            "v2_fixed_precision_component_income": {
                "position": precision_components[0],
                "velocity": precision_components[1],
                "normal": precision_components[2],
            },
            "interpretation": (
                "same-error reward counterfactual only; it proves the configured reward changes, "
                "not the learned-policy outcome"
            ),
        },
        "all_static_hierarchy_checks_pass": all_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_profile", type=Path)
    parser.add_argument("--action-catalog", type=Path)
    parser.add_argument("--position-error-m", type=float, default=0.6340)
    parser.add_argument("--velocity-error-mps", type=float, default=1.9595)
    parser.add_argument("--normal-error-deg", type=float, default=56.21)
    args = parser.parse_args()
    result = build_audit(
        args.task_profile.resolve(),
        action_catalog_path=(
            None if args.action_catalog is None else args.action_catalog.resolve()
        ),
        observed_errors=(
            args.position_error_m,
            args.velocity_error_mps,
            math.radians(args.normal_error_deg),
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["all_static_hierarchy_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
