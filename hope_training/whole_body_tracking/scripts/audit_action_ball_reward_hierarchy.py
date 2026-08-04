#!/usr/bin/env python3
"""Host-only arithmetic audit for the A3 Vendor V2 reward economy.

This is a configuration consequence, not a claim that PPO has learned the task.  In particular,
catalog frame counts are *not* wall-clock reward support: split-ready RESET_WAIT, the public
pre-swing bridge and teacher-rate scaling all add paid imitation steps.  The audit therefore keeps
the old catalog-only arithmetic as a labelled partial check and only evaluates per-swing timing
when an exact installed task receipt is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import yaml


POLICY_DT_S = 0.02
PPO_GAMMA_DEFAULT = 0.99
BODY_MOTION_WEIGHT_SUM = 4.5  # four body kernels + global anchor orientation
# Exact cap from phi(d)=clamp(d,0,4.65m), weight=10 and policy_dt=.02.  Because reward is
# phi(previous)-phi(current), the undiscounted sum telescopes and closed loops cannot mint income.
PROGRESS_UPPER = 0.93


class HierarchyAuditError(ValueError):
    pass


def _load_prelong_taxonomy(task_path: Path, *, profile: str) -> dict:
    """Load the exact fail-closed A211/C211 RewardManager composition."""

    source_path = (
        task_path.parents[2]
        / "source/whole_body_tracking/whole_body_tracking/utils"
        / "action_ball_prelong_semantics.py"
    )
    if not source_path.is_file():
        raise HierarchyAuditError("pre-long reward taxonomy source is missing")
    module_name = "action_ball_reward_hierarchy_prelong_semantics"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise HierarchyAuditError("cannot load the pre-long reward taxonomy")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        groups = module.prelong_group_term_weights(profile)
        safety = dict(module.PRELONG_EXCLUDED_SAFETY_TERM_WEIGHTS)
        probes = dict(module.PRELONG_EXCLUDED_PROBE_TERM_WEIGHTS)
        complete = module.expected_prelong_nonzero_reward_weights(profile)
    finally:
        sys.modules.pop(module_name, None)
    if not math.isclose(
        float(module.PRELONG_POLICY_DT_S), POLICY_DT_S, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise HierarchyAuditError("pre-long taxonomy policy dt drifted")
    return {
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "profile": profile,
        "manager_weight_to_per_step_income_scale": POLICY_DT_S,
        "scientific_groups": groups,
        "safety_terms_excluded_from_scientific_group_sums": safety,
        "manager_probes_with_zero_reward_callable": probes,
        "all_nonzero_manager_weights": complete,
        "interpretation": (
            "this is the exact pre-long non-zero RewardManager composition; missing, extra, "
            "or weight-drifted terms are rejected by the runtime pre-long ledger"
        ),
    }


def _discounted_constant_income(
    per_step_peak: float, steps: int, *, gamma: float
) -> float:
    if steps <= 0:
        return 0.0
    return float(per_step_peak) * (1.0 - gamma**int(steps)) / (1.0 - gamma)


def _termination_arbitrage_monitor(
    taxonomy: dict, *, legal_landing_min: float, legal_landing_max: float
) -> dict:
    """Expose, but do not fold into, the scientific layer-order gate.

    Landing is intentionally the largest positive task event.  That makes a
    legal return followed by a fall a separate safety-economy question rather
    than a reason to weaken the table objective.  The pre-long runtime gate
    must therefore report the joint landing/death stratum explicitly.
    """

    safety = taxonomy["safety_terms_excluded_from_scientific_group_sums"]
    death_weight = float(safety["death_penalty"])
    death_income = death_weight * POLICY_DT_S
    return {
        "death_penalty_weight": death_weight,
        "death_penalty_one_step_income": death_income,
        "legal_landing_plus_same_episode_death_floor": (
            legal_landing_min + death_income
        ),
        "legal_landing_plus_same_episode_death_max": (
            legal_landing_max + death_income
        ),
        "positive_success_then_fall_net_is_possible": (
            legal_landing_min + death_income > 0.0
        ),
        "required_runtime_stratum": (
            "legal_landing_count_with_post_contact_fall_or_termination; do not average it "
            "into all TASK_ACTIVE attempts"
        ),
    }


def _load_n1_timing_accounting(
    *,
    resolved: dict,
    taxonomy: dict,
    task_receipt_path: Path | None,
    gamma: float,
) -> dict:
    """Bind split-ready timing to wall clock, with a conservative one-tick edge allowance."""

    if task_receipt_path is None:
        return {
            "status": "BLOCKED_NO_INSTALLED_TASK_RECEIPT",
            "complete": False,
            "reason": (
                "catalog T does not encode hidden WAIT, pre_swing_wait_s, teacher_rate, "
                "scaled_t_cycle_s, or the installed contact tick"
            ),
        }
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise HierarchyAuditError("PPO gamma must be finite and in (0,1)")
    receipt = json.loads(task_receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise HierarchyAuditError("task receipt must contain a JSON mapping")
    task_wait = resolved.get("task_wait") or {}
    motion = resolved.get("motion") or {}
    if task_wait.get("enabled") is not True:
        raise HierarchyAuditError("timed N1 audit requires task_wait.enabled=true")
    if motion.get("action_ball_diagnostic_split_ready_teacher") is not True:
        raise HierarchyAuditError("timed N1 audit requires split-ready teacher semantics")
    wait_dt = _positive_number(task_wait, "policy_dt_s")
    contact_dt = _positive_number(receipt, "contact_time_step_s")
    if not math.isclose(wait_dt, POLICY_DT_S, abs_tol=1.0e-12) or not math.isclose(
        contact_dt, POLICY_DT_S, abs_tol=1.0e-12
    ):
        raise HierarchyAuditError("task wait/receipt policy dt drifted")
    min_wait = int(task_wait.get("min_wait_ticks", -1))
    max_wait = int(task_wait.get("max_wait_ticks", -1))
    if min_wait < 0 or max_wait < min_wait:
        raise HierarchyAuditError("invalid task WAIT tick envelope")
    contact_tick = int(receipt.get("time_to_contact_tick", -1))
    if contact_tick < 0:
        raise HierarchyAuditError("task receipt time_to_contact_tick must be non-negative")
    pre_wait_s = _positive_number(receipt, "pre_swing_wait_s", allow_zero=True)
    scaled_cycle_s = _positive_number(receipt, "scaled_t_cycle_s")
    teacher_rate = _positive_number(receipt, "teacher_rate")
    mimic_weights = taxonomy["scientific_groups"]["mimic"]
    if any(float(weight) <= 0.0 for weight in mimic_weights.values()):
        raise HierarchyAuditError("mimic taxonomy must contain only positive weights")
    mimic_raw_peak = sum(float(weight) for weight in mimic_weights.values())
    mimic_per_step_peak = mimic_raw_peak * POLICY_DT_S
    public_support_s = pre_wait_s + scaled_cycle_s
    # Command completion is evaluated on a discrete policy clock.  The upper
    # endpoint allows one extra evaluation tick, so a reported cap cannot be
    # made green by an off-by-one convention at the completion boundary.
    support_steps_min_lower = max(
        0, math.floor((min_wait * POLICY_DT_S + public_support_s) / POLICY_DT_S)
    )
    support_steps_min_upper = (
        math.ceil((min_wait * POLICY_DT_S + public_support_s) / POLICY_DT_S) + 1
    )
    support_steps_max_upper = (
        math.ceil((max_wait * POLICY_DT_S + public_support_s) / POLICY_DT_S) + 1
    )
    task_valid_support_steps_upper = (
        math.ceil(public_support_s / POLICY_DT_S) + 1
    )
    return {
        "status": "SELECTED_N1_RECEIPT_BOUND",
        "complete": True,
        "task_receipt_path": str(task_receipt_path),
        "task_receipt_sha256": _sha256(task_receipt_path),
        "teacher_rate": teacher_rate,
        "pre_swing_wait_s": pre_wait_s,
        "scaled_t_cycle_s": scaled_cycle_s,
        "public_teacher_support_s": public_support_s,
        "hidden_wait_ticks": {"min": min_wait, "max": max_wait},
        "contact_tick_after_reveal": contact_tick,
        "episode_contact_tick": {
            "earliest": min_wait + contact_tick,
            "latest": max_wait + contact_tick,
        },
        "mimic_raw_peak_per_step_before_dt": mimic_raw_peak,
        "mimic_peak_per_policy_step": mimic_per_step_peak,
        "mimic_reward_support_steps_conservative": {
            "global_lower": support_steps_min_lower,
            "min_wait_upper": support_steps_min_upper,
            "max_wait_upper": support_steps_max_upper,
            "task_valid_swing_upper": task_valid_support_steps_upper,
            "completion_edge_allowance_ticks": 1,
        },
        "mimic_undiscounted_cap_envelope": {
            "min_wait": support_steps_min_upper * mimic_per_step_peak,
            "max_wait": support_steps_max_upper * mimic_per_step_peak,
        },
        "mimic_episode_start_discounted_cap_envelope": {
            "min_wait": _discounted_constant_income(
                mimic_per_step_peak, support_steps_min_upper, gamma=gamma
            ),
            "max_wait": _discounted_constant_income(
                mimic_per_step_peak, support_steps_max_upper, gamma=gamma
            ),
        },
        "mimic_task_reveal_discounted_cap": _discounted_constant_income(
            mimic_per_step_peak,
            task_valid_support_steps_upper,
            gamma=gamma,
        ),
        "ready_mimic_undiscounted_cap_envelope": {
            "min_wait": min_wait * mimic_per_step_peak,
            "max_wait": max_wait * mimic_per_step_peak,
        },
        "eligibility_partition": {
            "ready_mimic": "task_invalid split-ready teacher; report in B_rollout/balance-ready",
            "task_valid_swing_mimic": (
                "from task reveal through single-stroke completion; use for B_motion^eligible"
            ),
            "runtime_ledger_split_required_before_long": True,
        },
        "gamma": gamma,
        "scope": (
            "selected installed N1 only; this does not extrapolate timing to the N73 catalog"
        ),
    }


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


def _single_parent_name(document: dict, *, source: Path) -> str:
    defaults = document.get("defaults")
    if not isinstance(defaults, list):
        raise HierarchyAuditError(f"{source.name} must declare a defaults list")
    parents = [
        str(row).split("@", 1)[0]
        for row in defaults
        if isinstance(row, str) and row != "_self_"
    ]
    if len(parents) != 1:
        raise HierarchyAuditError(
            f"{source.name} must inherit exactly one task profile, got {parents}"
        )
    return parents[0]


def _deep_merge(parent: object, child: object) -> object:
    if isinstance(parent, dict) and isinstance(child, dict):
        result = dict(parent)
        for key, value in child.items():
            if key == "defaults":
                continue
            result[key] = _deep_merge(result.get(key), value)
        return result
    return child


def _resolved_task_document(task_path: Path) -> tuple[dict, list[Path]]:
    """Resolve the single-parent local YAML chain used by the reward audit.

    This deliberately supports only repo-local task profiles.  It is not a
    replacement for Hydra composition; it is a fail-closed arithmetic audit of
    the reward/racket fields whose ownership stays in this one inheritance
    chain.
    """

    chain: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    current = task_path.resolve()
    while True:
        if current in seen:
            raise HierarchyAuditError("task profile inheritance contains a cycle")
        seen.add(current)
        document = yaml.safe_load(current.read_text())
        if not isinstance(document, dict):
            raise HierarchyAuditError(f"{current.name} must contain a YAML mapping")
        chain.append((current, document))
        if current.name == "HOPEPingPongActionBall.yaml":
            break
        parent = _single_parent_name(document, source=current)
        parent_path = current.with_name(f"{parent}.yaml")
        if not parent_path.is_file():
            raise HierarchyAuditError(
                f"{current.name} parent {parent!r} is not a local task profile"
            )
        current = parent_path.resolve()
    resolved: dict = {}
    for _path, document in reversed(chain):
        resolved = _deep_merge(resolved, document)
    return resolved, [path for path, _document in chain]


def _system_recipe_contract(
    task_path: Path,
    document: dict,
    resolved: dict,
    chain: list[Path],
) -> dict:
    """Prove that the numeric leaf is attached to the complete ActionBall lineage."""

    expected_chain = [
        "HOPEPingPongActionBallA3VendorV2.yaml",
        "HOPEPingPongActionBallA3VendorV1.yaml",
        "HOPEPingPongActionBall.yaml",
    ]
    if task_path.name == "HOPEPingPongActionBallA3VendorV2.yaml":
        _single_default_parent(
            document,
            expected="HOPEPingPongActionBallA3VendorV1",
            source=task_path,
        )
        expected_profile_chain = expected_chain
    elif task_path.name in {
        "HOPEPingPongActionBallA211VendorV2N1Learnability.yaml",
        "HOPEPingPongActionBallC211VendorV2N1Learnability.yaml",
    }:
        _single_default_parent(
            document,
            expected="HOPEPingPongActionBallA3VendorV2",
            source=task_path,
        )
        expected_profile_chain = [task_path.name, *expected_chain]
    elif task_path.name in {
        "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability.yaml",
        "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability.yaml",
    }:
        # 人话:两个发射器实际发的就是这两个 DR-L0 叶子。审计器以前拒收它们,
        # 于是 exp §5.3/§5.4 那份静态层级账根本不是对发射配方算的 —— 治理断链。
        #
        # The DR-L0 leaves are pure domain-randomization overlays: they compose the
        # retained-DR leaf via ``<parent>@_here_`` and only turn three DR switches off.
        # They therefore share the reward chain exactly, and refusing them left the
        # only static-hierarchy receipt describing a profile nobody launches.
        _single_default_parent(
            document,
            expected=task_path.name[: -len("DRL0Learnability.yaml")] + "Learnability",
            source=task_path,
        )
        expected_profile_chain = [
            task_path.name,
            task_path.name.replace("DRL0Learnability", "Learnability"),
            *expected_chain,
        ]
    else:
        raise HierarchyAuditError(
            "reward hierarchy audit accepts only VendorV2, its A211/C211 "
            "learnability leaves, or their DR-L0 overlays"
        )
    if [path.name for path in chain] != expected_profile_chain:
        raise HierarchyAuditError(
            "reward hierarchy task inheritance differs: "
            f"{[path.name for path in chain]!r}"
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

    rewards = resolved.get("rewards") or {}
    racket = resolved.get("racket") or {}
    action_ball_rewards = action_ball.get("rewards") or {}
    action_ball_racket = action_ball.get("racket") or {}
    is_c211 = (
        task_path.name
        in {
            "HOPEPingPongActionBallC211VendorV2N1Learnability.yaml",
            "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability.yaml",
        }
    )
    if is_c211:
        validity = racket.get("action_ball_target_validity_mask")
        # C211 consumes the sampler's incoming-ball question directly.  The
        # historical immutable_tape spelling was a target-information
        # ablation fixture and would also disable the live curriculum; it is
        # not the current no-inverse C211 source contract.
        c211_semantics = (
            racket.get("action_ball_target_source") == "direct_ball"
            and racket.get("action_ball_target_recipe") == "outcome_dense_only"
            and validity == [False, False, False]
        )
        fine_width_check = {
            "no_desired_contact_target": c211_semantics,
        }
        fine_width_mode = "not_applicable_c211"
    else:
        adaptive_flags = (
            racket.get("adaptive_sigma"),
            racket.get("adaptive_sigma_monotonic"),
            racket.get("adaptive_sigma_normal"),
        )
        if adaptive_flags == (True, True, True):
            fine_width_check = {
                "three_channel_monotonic_adaptive_fine": (
                    racket.get("adaptive_sigma_source") == "ball_exact_strike"
                )
            }
            fine_width_mode = "monotonic_adaptive"
        elif adaptive_flags == (False, False, False):
            fine_width_check = {"three_channel_static_fine": True}
            fine_width_mode = "static_rollout0"
        else:
            raise HierarchyAuditError(
                "three fine-width controller flags must be all true or all false"
            )
    checks = {
        "full_body_mimic": rewards.get("full_body_mimic") is True,
        "measured_racket_teacher": (
            racket.get("motion_teacher_racket_source") == "measured_channel"
        ),
        **fine_width_check,
        "action_ball_target_mode": action_ball_racket.get("target_mode") == "action_ball",
        "ball_outcome_enabled": action_ball_racket.get("virtual_ball") is True,
        "table_obstacle_enabled": action_ball.get("table_obstacle") is True,
        "complete_reward_pack": action_ball_rewards.get("reward_pack") == "v2",
    }
    if not all(checks.values()):
        raise HierarchyAuditError(f"incomplete one-run ActionBall recipe: {checks}")
    return {
        "checks": checks,
        "fine_width_mode": fine_width_mode,
        "profile_chain": [str(path) for path in chain],
        "profile_chain_sha256": [_sha256(path) for path in chain],
        "vendor_parent_sha256": _sha256(vendor_v1_path),
        "action_ball_parent_sha256": _sha256(action_ball_path),
        "interpretation": (
            "the successor restores full-body mimic and inherits the ball question, outcome, "
            "table, and reward pack from the complete ActionBall lineage; C211 replaces the "
            "desired-contact target with causal incoming-ball state; Stage1 is not a runtime stage"
        ),
    }


def _load_c211_reward_contract(task_path: Path) -> tuple[dict, Path, Path]:
    """Load the dependency-light C211 contract and bind both runtime source files."""

    package_root = task_path.parents[2]
    contract_path = (
        package_root
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
        / "action_ball_c211_trainability.py"
    )
    env_cfg_path = (
        package_root
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
        / "hope_env_cfg.py"
    )
    if not contract_path.is_file() or not env_cfg_path.is_file():
        raise HierarchyAuditError("C211 reward contract source files are missing")
    module_name = "action_ball_c211_reward_hierarchy_contract"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise HierarchyAuditError("cannot load the C211 reward contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        facts = module.c211_reward_contract_facts()
    finally:
        sys.modules.pop(module_name, None)
    if not isinstance(facts, dict):
        raise HierarchyAuditError("C211 reward contract facts must be a mapping")
    return facts, contract_path, env_cfg_path


def _build_c211_audit(
    task_path: Path,
    *,
    observed_errors: tuple[float, float, float],
    action_catalog_path: Path | None,
    task_receipt_path: Path | None,
    gamma: float,
    resolved: dict,
    system_recipe: dict,
) -> dict:
    """Audit C211's mimic / proximity / achieved-landing reward layers."""

    rewards = resolved.get("rewards") or {}
    racket = resolved.get("racket") or {}
    facts, contract_path, env_cfg_path = _load_c211_reward_contract(task_path)
    bridge = facts.get("strike_bridge") or {}
    economics = facts.get("economics") or {}
    landing = facts.get("landing") or {}
    strike_weight = _positive_number(bridge, "weight")
    strike_std = _positive_number(bridge, "std_m")
    policy_dt = _positive_number(economics, "policy_dt_s")
    if not math.isclose(policy_dt, POLICY_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise HierarchyAuditError("C211 reward contract policy dt drifted")
    landing_weight = _positive_number(rewards, "virtual_landing_weight")
    landing_base_frac = _positive_number(rewards, "virtual_landing_base_frac")
    if landing_base_frac >= 1.0:
        raise HierarchyAuditError("C211 legal landing base fraction must be in (0,1)")
    off_table_frac = 0.5
    if landing.get("opponent_side_off_table") != "0.5_times_same_gaussian":
        raise HierarchyAuditError("C211 off-table reward contract drifted")
    if bridge.get("kernel") != "cauchy_inverse_quadratic":
        raise HierarchyAuditError("C211 strike bridge must use the Cauchy kernel")
    if facts.get("desired_contact_position_velocity_face_consumed") is not False:
        raise HierarchyAuditError("C211 must not consume desired-contact targets")
    if facts.get("task_valid_required") is not True:
        raise HierarchyAuditError("C211 reward must require task_valid")

    forbidden_nonzero = (
        "base_position_weight",
        "racket_position_coarse_weight",
        "racket_velocity_coarse_weight",
        "racket_normal_coarse_weight",
        "racket_position_weight",
        "racket_velocity_weight",
        "racket_normal_weight",
        "racket_position_precision_weight",
        "racket_velocity_precision_weight",
        "racket_normal_precision_weight",
        "strike_capture_bonus_weight",
        "virtual_pass_net_weight",
        "virtual_landing_dense_weight",
    )
    for key in forbidden_nonzero:
        if _positive_number(rewards, key, allow_zero=True) != 0.0:
            raise HierarchyAuditError(f"C211 desired-target duplicate must stay zero: {key}")

    motion_scale = _positive_number(rewards, "motion_scale")
    paddle_weights = [
        _positive_number(rewards, f"motion_racket_{name}_weight")
        for name in ("position", "velocity", "normal", "long_axis")
    ]
    wide_half_window_s = _positive_number(racket, "strike_window_wide_s")
    wide_window_steps = _inclusive_window_steps(wide_half_window_s)
    motion_racket_window_scale = _positive_number(
        rewards, "motion_racket_scale_in_strike_window", allow_zero=True
    )
    motion_racket_long_axis_window_scale = _positive_number(
        rewards,
        "motion_racket_long_axis_scale_in_strike_window",
        allow_zero=True,
    )
    if motion_racket_window_scale > 1.0 or motion_racket_long_axis_window_scale > 1.0:
        raise HierarchyAuditError("C211 measured-paddle window scales must be <=1")

    taxonomy = _load_prelong_taxonomy(task_path, profile="C211")
    taxonomy_groups = taxonomy["scientific_groups"]
    if taxonomy_groups["strike"] != {
        "c225_strike_ball_paddle_center_proximity": strike_weight
    }:
        raise HierarchyAuditError(
            "C211 strike contract and exact pre-long taxonomy disagree"
        )
    if taxonomy_groups["outcome"] != {"virtual_landing": landing_weight}:
        raise HierarchyAuditError(
            "C211 landing contract and exact pre-long taxonomy disagree"
        )
    timing = _load_n1_timing_accounting(
        resolved=resolved,
        taxonomy=taxonomy,
        task_receipt_path=task_receipt_path,
        gamma=gamma,
    )
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

    strike_peak = strike_weight * POLICY_DT_S
    landing_min = landing_weight * POLICY_DT_S * landing_base_frac
    landing_max = landing_weight * POLICY_DT_S
    off_table_max = landing_weight * POLICY_DT_S * off_table_frac
    action_rows = {}
    for action, action_cap in action_caps.items():
        action_rows[action] = {
            "frames": action_steps[action],
            "native_catalog_frame_only_imitation_cap": action_cap,
            "strike_proximity_one_shot_peak": strike_peak,
            "legal_landing_event_min": landing_min,
            "legal_landing_event_max": landing_max,
            "opponent_side_off_table_event_max": off_table_max,
            "strict_order": action_cap < strike_peak < landing_min,
            "off_table_below_legal_landing_floor": off_table_max < landing_min,
        }
    distance = float(observed_errors[0])
    if not math.isfinite(distance) or distance < 0.0:
        raise HierarchyAuditError("C211 observed paddle-ball distance must be finite and >=0")
    ratio = distance / strike_std
    income = strike_peak / (1.0 + ratio * ratio)
    gradient = 0.0
    if distance > 0.0:
        gradient = -(
            2.0 * strike_peak * distance / (strike_std**2)
        ) / ((1.0 + ratio * ratio) ** 2)
    partial_catalog_pass = all(
        row["strict_order"] and row["off_table_below_legal_landing_floor"]
        for row in action_rows.values()
    )
    selected_n1 = None
    if timing["complete"]:
        earliest_contact = timing["episode_contact_tick"]["earliest"]
        latest_contact = timing["episode_contact_tick"]["latest"]
        discounted = {}
        for wait_name, contact_tick, mimic_key in (
            ("min_wait", earliest_contact, "min_wait"),
            ("max_wait", latest_contact, "max_wait"),
        ):
            mimic_cap = timing["mimic_episode_start_discounted_cap_envelope"][
                mimic_key
            ]
            strike_cap = strike_peak * gamma**contact_tick
            landing_floor = landing_min * gamma**contact_tick
            discounted[wait_name] = {
                "contact_tick": contact_tick,
                "mimic_cap": mimic_cap,
                "strike_proximity_peak": strike_cap,
                "legal_landing_floor": landing_floor,
                "strict_order": mimic_cap < strike_cap < landing_floor,
            }
        task_contact = timing["contact_tick_after_reveal"]
        task_reveal_discounted = {
            "contact_tick": task_contact,
            "mimic_cap": timing["mimic_task_reveal_discounted_cap"],
            "strike_proximity_peak": strike_peak * gamma**task_contact,
            "legal_landing_floor": landing_min * gamma**task_contact,
        }
        task_reveal_discounted["strict_order"] = (
            task_reveal_discounted["mimic_cap"]
            < task_reveal_discounted["strike_proximity_peak"]
            < task_reveal_discounted["legal_landing_floor"]
        )
        undiscounted = {
            "task_valid_swing_mimic_cap": (
                timing["mimic_reward_support_steps_conservative"][
                    "task_valid_swing_upper"
                ]
                * timing["mimic_peak_per_policy_step"]
            ),
            "strike_proximity_peak": strike_peak,
            "legal_landing_floor": landing_min,
        }
        undiscounted["strict_order"] = (
            undiscounted["task_valid_swing_mimic_cap"]
            < undiscounted["strike_proximity_peak"]
            < undiscounted["legal_landing_floor"]
        )
        selected_n1 = {
            "undiscounted_per_swing": undiscounted,
            "task_reveal_discounted_eligible": task_reveal_discounted,
            "rollout_start_discounted_diagnostic": discounted,
            "discounted_strict_order": task_reveal_discounted["strict_order"],
            "interpretation": (
                "the proximity term is a nominal-strike-tick Cauchy distance bridge, not a "
                "contact bonus; landing remains gated by actual selected-rubber achieved flight; "
                "hidden WAIT ready-mimic is reported in rollout accounting, not charged to the "
                "task-valid swing opportunity"
            ),
        }
    complete_hierarchy_pass = bool(
        selected_n1 is not None
        and selected_n1["discounted_strict_order"]
    )
    return {
        "schema_version": 1,
        "kind": "action_ball_c211_reward_hierarchy_v1",
        "authorization": {
            "training": False,
            "promotion": False,
            "diagnostic_unauthorized": True,
        },
        "task_profile": str(task_path),
        "task_profile_sha256": _sha256(task_path),
        "system_recipe": {
            **system_recipe,
            "c211_reward_contract_path": str(contract_path),
            "c211_reward_contract_sha256": _sha256(contract_path),
            "runtime_env_cfg_path": str(env_cfg_path),
            "runtime_env_cfg_sha256": _sha256(env_cfg_path),
        },
        "active_reward_taxonomy": taxonomy,
        "termination_arbitrage_monitor": (
            None
            if taxonomy is None
            else _termination_arbitrage_monitor(
                taxonomy,
                legal_landing_min=landing_min,
                legal_landing_max=landing_max,
            )
        ),
        "selected_n1_wall_clock": timing,
        "selected_n1_layer_accounting": selected_n1,
        "hierarchy_authority": {
            "complete_static_hierarchy_authorized": complete_hierarchy_pass,
            "n73_authorized": False,
            "status": (
                "PASS_SELECTED_N1_DISCOUNTED_LAYER_ORDER"
                if complete_hierarchy_pass
                else "FAIL_OR_BLOCKED_DISCOUNTED_LAYER_ORDER"
            ),
            "reason": (
                "a catalog-frame cap is insufficient; current authority requires an installed "
                "receipt and the task-reveal eligible discounted order. Hidden WAIT ready-mimic "
                "is a separate balance/readiness account"
            ),
        },
        "action_catalog": str(action_catalog_path),
        "action_catalog_sha256": _sha256(action_catalog_path),
        "constants": {
            "policy_dt_s": POLICY_DT_S,
            "strike_weight": strike_weight,
            "strike_std_m": strike_std,
            "strike_one_shot_peak": strike_peak,
            "landing_weight": landing_weight,
            "landing_base_frac": landing_base_frac,
            "legal_landing_min": landing_min,
            "legal_landing_max": landing_max,
            "off_table_frac": off_table_frac,
            "off_table_max": off_table_max,
            "wide_half_window_s": wide_half_window_s,
            "wide_window_steps": wide_window_steps,
            "motion_racket_scale_in_strike_window": motion_racket_window_scale,
            "motion_racket_long_axis_scale_in_strike_window": (
                motion_racket_long_axis_window_scale
            ),
        },
        "actions": action_rows,
        "catalog_summary": {
            "action_count": len(action_rows),
            "native_catalog_frame_only_imitation_cap_p50": _percentile(
                list(action_caps.values()), 50.0
            ),
            "native_catalog_frame_only_imitation_cap_p95": _percentile(
                list(action_caps.values()), 95.0
            ),
            "native_catalog_frame_only_imitation_cap_max": max(
                action_caps.values()
            ),
            "longest_action": max(action_steps, key=action_steps.get),
            "longest_action_frames": max(action_steps.values()),
            "strike_proximity_one_shot_peak": strike_peak,
            "legal_landing_event_min": landing_min,
            "legal_landing_event_max": landing_max,
            "opponent_side_off_table_event_max": off_table_max,
            "partial_catalog_frame_only_order": all(
                row["strict_order"] for row in action_rows.values()
            ),
            "scope_warning": (
                "catalog-frame-only imitation omits RESET_WAIT, pre-swing bridge, and "
                "teacher-rate wall-clock expansion"
            ),
        },
        "frozen_paddle_ball_distance_counterfactual": {
            "distance_m": distance,
            "income": income,
            "signed_derivative_wrt_distance": gradient,
            "nonzero_tail": income > 0.0 and math.isfinite(gradient),
            "velocity_and_face_errors_unused": True,
            "interpretation": (
                "configuration consequence only; this proves the fixed Cauchy tail, not "
                "policy learnability or a discounted reward hierarchy"
            ),
        },
        "partial_catalog_arithmetic_checks_pass": partial_catalog_pass,
        "all_static_hierarchy_checks_pass": complete_hierarchy_pass,
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
    task_receipt_path: Path | None = None,
    gamma: float = PPO_GAMMA_DEFAULT,
) -> dict:
    document = yaml.safe_load(task_path.read_text())
    resolved, chain = _resolved_task_document(task_path)
    system_recipe = _system_recipe_contract(
        task_path,
        document,
        resolved,
        chain,
    )
    if (
        task_path.name
        in {
            "HOPEPingPongActionBallC211VendorV2N1Learnability.yaml",
            "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability.yaml",
        }
    ):
        return _build_c211_audit(
            task_path,
            observed_errors=observed_errors,
            action_catalog_path=action_catalog_path,
            task_receipt_path=task_receipt_path,
            gamma=gamma,
            resolved=resolved,
            system_recipe=system_recipe,
        )
    rewards = resolved.get("rewards") or {}
    racket = resolved.get("racket") or {}
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
    adaptive_enabled = system_recipe["fine_width_mode"] == "monotonic_adaptive"
    adaptive_bounds = {}
    for channel, acceptance_scale, start_scale in zip(
        ("pos", "vel", "normal"), precision_scales, adaptive_start_scales
    ):
        if adaptive_enabled:
            lower = _positive_number(racket, f"sigma_{channel}_min")
            upper = _positive_number(racket, f"sigma_{channel}_max")
            if not math.isclose(
                lower, acceptance_scale, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise HierarchyAuditError(
                    f"sigma_{channel}_min must equal the fixed precision acceptance width"
                )
            if not math.isclose(
                upper, start_scale, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise HierarchyAuditError(
                    f"racket {channel} adaptive std must start at sigma_{channel}_max"
                )
            if upper < lower:
                raise HierarchyAuditError(
                    f"sigma_{channel}_max must be >= sigma_{channel}_min"
                )
        else:
            lower = start_scale
            upper = start_scale
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
    target_at_fine_acceptance_by_channel = [
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
    ]
    target_at_fine_acceptance = sum(target_at_fine_acceptance_by_channel)
    target_at_fine_acceptance_initial_by_channel = [
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
    ]
    target_at_fine_acceptance_initial = sum(
        target_at_fine_acceptance_initial_by_channel
    )
    target_kernel_max_by_channel = [
        steps * POLICY_DT_S * (coarse_weight + adaptive_weight + precision_weight)
        for steps, coarse_weight, adaptive_weight, precision_weight in zip(
            channel_window_steps, coarse_weights, adaptive_weights, precision_weights
        )
    ]
    target_kernel_max = sum(target_kernel_max_by_channel)
    landing_min = landing_weight * POLICY_DT_S * landing_base_frac
    landing_max = landing_weight * POLICY_DT_S

    action_rows = {}
    for action, action_cap in action_caps.items():
        target_upper = target_kernel_max + PROGRESS_UPPER
        action_rows[action] = {
            "frames": action_steps[action],
            "native_catalog_frame_only_imitation_cap": action_cap,
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
    partial_catalog_pass = all(
        row["strict_order_at_fine_acceptance"]
        and row["target_theoretical_upper_below_landing_min"]
        for row in action_rows.values()
    )
    is_a211 = (
        task_path.name
        in {
            "HOPEPingPongActionBallA211VendorV2N1Learnability.yaml",
            "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability.yaml",
        }
    )
    taxonomy = (
        _load_prelong_taxonomy(task_path, profile="A211") if is_a211 else None
    )
    timing = (
        _load_n1_timing_accounting(
            resolved=resolved,
            taxonomy=taxonomy,
            task_receipt_path=task_receipt_path,
            gamma=gamma,
        )
        if taxonomy is not None
        else {
            "status": "BLOCKED_NOT_AN_EXACT_PRELONG_PROFILE",
            "complete": False,
            "reason": "VendorV2 parent alone is not the exact A211 pre-long runtime composition",
        }
    )
    selected_n1 = None
    if timing["complete"]:
        half_windows = [
            position_window_steps // 2,
            wide_window_steps // 2,
            wide_window_steps // 2,
        ]

        def _discounted_window_income_at_contact(
            contact_tick: int, channel_incomes: list[float]
        ) -> float:
            total = 0.0
            for income, steps, half_window in zip(
                channel_incomes,
                channel_window_steps,
                half_windows,
            ):
                per_step = income / steps
                total += per_step * sum(
                    gamma ** (contact_tick + offset)
                    for offset in range(-half_window, half_window + 1)
                )
            return total

        discounted = {}
        for wait_name, contact_tick, mimic_key in (
            (
                "min_wait",
                timing["episode_contact_tick"]["earliest"],
                "min_wait",
            ),
            (
                "max_wait",
                timing["episode_contact_tick"]["latest"],
                "max_wait",
            ),
        ):
            mimic_cap = timing["mimic_episode_start_discounted_cap_envelope"][
                mimic_key
            ]
            target_income = _discounted_window_income_at_contact(
                contact_tick, target_at_fine_acceptance_by_channel
            )
            landing_floor = landing_min * gamma**contact_tick
            discounted[wait_name] = {
                "contact_tick": contact_tick,
                "mimic_cap": mimic_cap,
                "window_target_income_at_fine_acceptance": target_income,
                "legal_landing_floor": landing_floor,
                "racket_progress_undiscounted_telescoping_upper_not_added": (
                    PROGRESS_UPPER
                ),
                "strict_order_without_unrealized_progress": (
                    mimic_cap < target_income < landing_floor
                ),
            }
        task_contact = timing["contact_tick_after_reveal"]
        target_terms = taxonomy["scientific_groups"]["target"]
        # Fixed-center Take061 intentionally removes base_position: the robot is
        # born at the commanded station, so paying that term would be free
        # pre-strike income rather than footwork guidance.  Expansion must
        # restore it or replace it with a potential term.
        base_weight = float(target_terms.get("base_position", 0.0))
        progress_weight = float(target_terms["racket_progress"])
        base_position_upper = (
            base_weight
            * POLICY_DT_S
            * (1.0 - gamma**task_contact)
            / (1.0 - gamma)
        )
        progress_upper = progress_weight * POLICY_DT_S * 4.65
        window_fine = _discounted_window_income_at_contact(
            task_contact, target_at_fine_acceptance_by_channel
        )
        window_max = _discounted_window_income_at_contact(
            task_contact, target_kernel_max_by_channel
        )
        task_reveal_discounted = {
            "contact_tick": task_contact,
            "mimic_cap": timing["mimic_task_reveal_discounted_cap"],
            "target_guidance_lower_at_fine_acceptance": window_fine,
            "target_guidance_upper_components": {
                "window_kernel_max": window_max,
                "base_position_prestrike_max": base_position_upper,
                "racket_progress_theoretical_telescoping_upper_assumed_at_t0": (
                    progress_upper
                ),
            },
            "target_guidance_conservative_upper": (
                window_max + base_position_upper + progress_upper
            ),
            "legal_landing_floor": landing_min * gamma**task_contact,
        }
        task_reveal_discounted["motion_below_target_lower"] = (
            task_reveal_discounted["mimic_cap"]
            < task_reveal_discounted[
                "target_guidance_lower_at_fine_acceptance"
            ]
        )
        task_reveal_discounted["target_upper_below_landing"] = (
            task_reveal_discounted["target_guidance_conservative_upper"]
            < task_reveal_discounted["legal_landing_floor"]
        )
        task_reveal_discounted["strict_order_proved"] = (
            task_reveal_discounted["motion_below_target_lower"]
            and task_reveal_discounted["target_upper_below_landing"]
        )
        undiscounted = {
            "task_valid_swing_mimic_cap": (
                timing["mimic_reward_support_steps_conservative"][
                    "task_valid_swing_upper"
                ]
                * timing["mimic_peak_per_policy_step"]
            ),
            "target_guidance_lower_at_fine_acceptance": (
                target_at_fine_acceptance
            ),
            "target_guidance_conservative_upper": (
                target_kernel_max
                + base_weight * POLICY_DT_S * task_contact
                + progress_upper
            ),
            "legal_landing_floor": landing_min,
        }
        undiscounted["strict_order_proved"] = (
            undiscounted["task_valid_swing_mimic_cap"]
            < undiscounted["target_guidance_lower_at_fine_acceptance"]
            and undiscounted["target_guidance_conservative_upper"]
            < undiscounted["legal_landing_floor"]
        )
        selected_n1 = {
            "undiscounted_per_swing": undiscounted,
            "task_reveal_discounted_eligible": task_reveal_discounted,
            "rollout_start_window_only_diagnostic": discounted,
            "discounted_strict_order_proved": task_reveal_discounted[
                "strict_order_proved"
            ],
            "reported_outside_target_guidance": {
                "balance": "upright_exp and balance penalties",
                "actual_contact_bridges": (
                    "strike_capture_bonus, virtual_pass_net, and virtual_landing_dense"
                ),
            },
            "runtime_validation_requirement": (
                "the static upper uses the full theoretical racket_progress potential cap. "
                "The pre-long runtime ledger must still bind actual compatible target income "
                "on task-valid swings and monitor landing-versus-death incentives; this is a "
                "runtime validation requirement, not a static-order blocker"
            ),
        }
    complete_hierarchy_pass = bool(
        selected_n1 is not None
        and selected_n1["discounted_strict_order_proved"]
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
        "active_reward_taxonomy": taxonomy,
        "termination_arbitrage_monitor": (
            None
            if taxonomy is None
            else _termination_arbitrage_monitor(
                taxonomy,
                legal_landing_min=landing_min,
                legal_landing_max=landing_max,
            )
        ),
        "selected_n1_wall_clock": timing,
        "selected_n1_layer_accounting": selected_n1,
        "hierarchy_authority": {
            "complete_static_hierarchy_authorized": complete_hierarchy_pass,
            "n73_authorized": False,
            "status": (
                "PASS_SELECTED_N1_DISCOUNTED_LAYER_ORDER"
                if complete_hierarchy_pass
                else "FAIL_OR_BLOCKED_DISCOUNTED_LAYER_ORDER"
            ),
            "reason": (
                "catalog T is a partial kernel check only; complete authority requires exact "
                "wall-clock timing, task-invalid ready/task-valid swing mimic separation, and "
                "the discounted mimic < conservative target+progress envelope < landing order. "
                "Runtime still reports realized compatible target income separately"
            ),
        },
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
            "native_catalog_frame_only_imitation_cap_p50": _percentile(
                list(action_caps.values()), 50.0
            ),
            "native_catalog_frame_only_imitation_cap_p95": _percentile(
                list(action_caps.values()), 95.0
            ),
            "native_catalog_frame_only_imitation_cap_max": max(
                action_caps.values()
            ),
            "longest_action": max(action_steps, key=action_steps.get),
            "longest_action_frames": max(action_steps.values()),
            "target_income_at_fine_acceptance": target_at_fine_acceptance,
            "target_initial_income_at_fine_acceptance": (
                target_at_fine_acceptance_initial
            ),
            "broad_one_sigma_income": broad_envelope_floor,
            "target_kernel_max": target_kernel_max,
            "target_kernel_plus_progress_upper": target_kernel_max + PROGRESS_UPPER,
            "partial_catalog_frame_only_order_at_fine_acceptance": all(
                row["strict_order_at_fine_acceptance"] for row in action_rows.values()
            ),
            "scope_warning": (
                "catalog-frame-only imitation omits RESET_WAIT, pre-swing bridge, and "
                "teacher-rate wall-clock expansion"
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
        "partial_catalog_arithmetic_checks_pass": partial_catalog_pass,
        "all_static_hierarchy_checks_pass": complete_hierarchy_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_profile", type=Path)
    parser.add_argument("--action-catalog", type=Path)
    parser.add_argument("--task-receipt", type=Path)
    parser.add_argument("--gamma", type=float, default=PPO_GAMMA_DEFAULT)
    parser.add_argument("--position-error-m", type=float, default=0.6340)
    parser.add_argument("--velocity-error-mps", type=float, default=1.9595)
    parser.add_argument("--normal-error-deg", type=float, default=56.21)
    args = parser.parse_args()
    result = build_audit(
        args.task_profile.resolve(),
        action_catalog_path=(
            None if args.action_catalog is None else args.action_catalog.resolve()
        ),
        task_receipt_path=(
            None if args.task_receipt is None else args.task_receipt.resolve()
        ),
        gamma=args.gamma,
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
