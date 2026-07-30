#!/usr/bin/env python3
"""Diagnose the fixed-action teacher-rate distribution of one N=1 bundle.

This tool is deliberately read-only: it verifies an existing content-addressed
N=1 bundle and manifest, samples one deterministic proposal tape, runs the
same counter-rally precheck, batched continuous solver, exact face geometry,
and post-solver timing checks used by the training preflight, then writes one
JSON report to stdout.  Redirect stdout if a persistent report is desired.

The optional incoming-speed overrides are applied only to an in-memory copy of
the manifest.  They never rewrite or bless an artifact.  When the speed centre
moves, inherited maximum widths are clipped only as much as required for the
copied manifest to remain internally valid; the effective values are reported.

The script imports the materializer and solver graph from ``--repo-root``.
Consequently it may be copied to ``/tmp`` on a Pod and pointed at the exact
clean training checkout without importing Isaac, MuJoCo, or this script from
that checkout.  Dependencies are Python, NumPy, and Torch.

Example (run from the exact checkout on a Pod)::

    python /tmp/diagnose_n1_teacher_rate_tape.py \
      --repo-root "$PWD" \
      --bundle configs/n1_contact_dynamic_ready_20260730_r9/\
bh_block.bundle.v2.3267a3f6d303.json \
      --proposal-count 512 \
      --racket-face-speed-scale 0.7 \
      --device cuda

The report is evidence about proposal/solver geometry and timing only.  It is
not a training authorization, curriculum-promotion receipt, or policy result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = (
    SCRIPT_DIR.parents[2]
    if len(SCRIPT_DIR.parents) > 2
    else Path.cwd()
)
DEFAULT_PROPOSAL_COUNT = 512
DEFAULT_SEED = 0
DEFAULT_CONTACT_TIME_STEP_S = 0.02
DEFAULT_EPISODE_LENGTH_S = 10.0
DEFAULT_ATTEMPT_CLOSE_MARGIN_S = 0.02
PERCENTILES = (1, 5, 50, 95, 99)


class TeacherRateTapeError(ValueError):
    """Fail-closed bundle, identity, solver, or argument error."""


def _load_materializer(repo_root: Path) -> Any:
    path = (
        repo_root
        / "hope_training/whole_body_tracking/scripts/"
        "materialize_n1_contact_training_bundle.py"
    )
    if not path.is_file():
        raise TeacherRateTapeError(
            "materializer is missing from exact repo root: {}".format(path)
        )
    name = "_n1_teacher_rate_materializer_{}".format(
        __import__("hashlib").sha256(str(path).encode("utf-8")).hexdigest()[
            :16
        ]
    )
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise TeacherRateTapeError(
            "cannot load exact materializer {}".format(path)
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
    ):
        raise TeacherRateTapeError(
            "{} must be one plain finite number".format(label)
        )
    return float(value)


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise TeacherRateTapeError(
            "{} must be one positive integer".format(label)
        )
    return int(value)


def _resolve_user_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve(strict=True) if path.is_absolute() else (
        repo_root / path
    ).resolve(strict=True)


def _verify_reference(
    *,
    repo_root: Path,
    materializer: Any,
    reference: object,
    label: str,
) -> Tuple[Path, str]:
    if type(reference) is not dict:
        raise TeacherRateTapeError(
            "{} reference must be one mapping".format(label)
        )
    path_value = reference.get("path")
    sha_value = reference.get("sha256")
    if type(path_value) is not str or type(sha_value) is not str:
        raise TeacherRateTapeError(
            "{} reference must contain string path/sha256".format(label)
        )
    path, _ = materializer._resolve_repo_file(
        repo_root, path_value, label=label
    )
    expected = materializer._require_sha256(
        sha_value, label="{} SHA-256".format(label)
    )
    actual = materializer._sha256_file(path)
    if actual != expected:
        raise TeacherRateTapeError(
            "{} bytes changed: expected {}, got {}".format(
                label, expected, actual
            )
        )
    return path, actual


def _one_prototype_row(
    prototype: Mapping[str, Any],
    *,
    scope: str,
    action_id: str,
) -> Mapping[str, Any]:
    scopes = prototype.get("scopes")
    if type(scopes) is not dict or type(scopes.get(scope)) is not list:
        raise TeacherRateTapeError(
            "prototype lacks scope {!r}".format(scope)
        )
    rows = [
        row
        for row in scopes[scope]
        if type(row) is dict
        and row.get("enabled") is True
        and row.get("motion_id") == action_id
    ]
    if len(rows) != 1:
        raise TeacherRateTapeError(
            "prototype must contain exactly one enabled row for {!r}".format(
                action_id
            )
        )
    return rows[0]


def _apply_speed_overrides(
    manifest: Mapping[str, Any],
    *,
    center_mps: Optional[float],
    maximum_mps: Optional[float],
    lower_std_mps: Optional[float],
    upper_std_mps: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    copied = deepcopy(dict(manifest))
    actions = copied.get("actions")
    if type(actions) is not list or len(actions) != 1:
        raise TeacherRateTapeError(
            "teacher-rate tape accepts exact N=1 manifests only"
        )
    action = actions[0]
    if type(action) is not dict or type(action.get("ball_profile")) is not dict:
        raise TeacherRateTapeError("N=1 action lacks ball_profile")
    profile = action["ball_profile"]
    original = {
        key: profile.get(key)
        for key in (
            "incoming_speed_center_mps",
            "incoming_speed_min_mps",
            "incoming_speed_max_mps",
            "incoming_speed_std_lower_initial_mps",
            "incoming_speed_std_lower_max_mps",
            "incoming_speed_std_upper_initial_mps",
            "incoming_speed_std_upper_max_mps",
        )
    }
    if center_mps is not None:
        profile["incoming_speed_center_mps"] = center_mps
        # The manifest contract defines the lower hard support as exactly
        # 0.4 times the action-centred incoming speed.
        profile["incoming_speed_min_mps"] = 0.4 * center_mps
    if maximum_mps is not None:
        objective = copied.get("counter_rally_objective")
        if type(objective) is not dict:
            raise TeacherRateTapeError(
                "manifest lacks counter_rally_objective"
            )
        objective_maximum = _finite(
            objective["maximum_supported_ball_speed_mps"],
            label="objective maximum supported ball speed",
        )
        if maximum_mps > objective_maximum + 1.0e-12:
            raise TeacherRateTapeError(
                "incoming speed maximum exceeds objective support"
            )
        profile["incoming_speed_max_mps"] = maximum_mps
    if lower_std_mps is not None:
        profile["incoming_speed_std_lower_initial_mps"] = lower_std_mps
    if upper_std_mps is not None:
        profile["incoming_speed_std_upper_initial_mps"] = upper_std_mps

    center = _finite(
        profile["incoming_speed_center_mps"],
        label="effective incoming speed centre",
    )
    minimum = _finite(
        profile["incoming_speed_min_mps"],
        label="incoming speed minimum",
    )
    maximum = _finite(
        profile["incoming_speed_max_mps"],
        label="incoming speed maximum",
    )
    lower_initial = _finite(
        profile["incoming_speed_std_lower_initial_mps"],
        label="incoming speed lower initial std",
    )
    upper_initial = _finite(
        profile["incoming_speed_std_upper_initial_mps"],
        label="incoming speed upper initial std",
    )
    if not minimum <= center <= maximum:
        raise TeacherRateTapeError(
            "incoming speed centre lies outside manifest hard bounds"
        )
    lower_room = center - minimum
    upper_room = maximum - center
    if (
        lower_initial < 0.0
        or upper_initial < 0.0
        or lower_initial > lower_room + 1.0e-12
        or upper_initial > upper_room + 1.0e-12
    ):
        raise TeacherRateTapeError(
            "incoming speed initial std exceeds centre-to-bound room"
        )
    # A centre-only diagnostic must not fail merely because a legacy maximum
    # width was tied to the old centre.  Preserve it when possible and clip it
    # only to the new hard-bound room; never clip below the sampled width.
    lower_maximum = min(
        _finite(
            profile["incoming_speed_std_lower_max_mps"],
            label="incoming speed lower maximum std",
        ),
        lower_room,
    )
    upper_maximum = min(
        _finite(
            profile["incoming_speed_std_upper_max_mps"],
            label="incoming speed upper maximum std",
        ),
        upper_room,
    )
    if lower_initial > lower_maximum + 1.0e-12:
        raise TeacherRateTapeError(
            "incoming speed lower initial std exceeds effective maximum"
        )
    if upper_initial > upper_maximum + 1.0e-12:
        raise TeacherRateTapeError(
            "incoming speed upper initial std exceeds effective maximum"
        )
    profile["incoming_speed_std_lower_max_mps"] = lower_maximum
    profile["incoming_speed_std_upper_max_mps"] = upper_maximum
    effective = {
        key: profile[key] for key in original
    }
    return copied, {
        "requested": {
            "incoming_speed_center_mps": center_mps,
            "incoming_speed_max_mps": maximum_mps,
            "incoming_speed_std_lower_initial_mps": lower_std_mps,
            "incoming_speed_std_upper_initial_mps": upper_std_mps,
        },
        "original": original,
        "effective": effective,
        "claim": (
            "in_memory_diagnostic_override_only_not_an_artifact_or_"
            "training_authorization"
        ),
    }


def _apply_landing_aim_overrides(
    manifest: Mapping[str, Any],
    *,
    center_x_m: Optional[float],
    center_y_m: Optional[float],
    std_m: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    copied = deepcopy(dict(manifest))
    landing = copied.get("landing_aim")
    if type(landing) is not dict:
        raise TeacherRateTapeError("manifest lacks landing_aim")
    keys = (
        "center_w_xy_m",
        "min_w_xy_m",
        "max_w_xy_m",
        "std_lower_initial_m",
        "std_lower_max_m",
        "std_upper_initial_m",
        "std_upper_max_m",
    )
    original = {key: deepcopy(landing.get(key)) for key in keys}
    center = [
        _finite(value, label="landing aim centre")
        for value in landing["center_w_xy_m"]
    ]
    if center_x_m is not None:
        center[0] = _finite(
            center_x_m, label="landing aim X centre override"
        )
    if center_y_m is not None:
        center[1] = _finite(
            center_y_m, label="landing aim Y centre override"
        )
    lower_initial = [
        _finite(value, label="landing aim lower initial std")
        for value in landing["std_lower_initial_m"]
    ]
    upper_initial = [
        _finite(value, label="landing aim upper initial std")
        for value in landing["std_upper_initial_m"]
    ]
    if std_m is not None:
        std_m = _finite(std_m, label="landing aim std override")
        if std_m < 0.0:
            raise TeacherRateTapeError(
                "landing aim diagnostic std must be non-negative"
            )
        lower_initial = [std_m, std_m]
        upper_initial = [std_m, std_m]
    minimum = [
        _finite(value, label="landing aim hard minimum")
        for value in landing["min_w_xy_m"]
    ]
    maximum = [
        _finite(value, label="landing aim hard maximum")
        for value in landing["max_w_xy_m"]
    ]
    lower_maximum = [
        _finite(value, label="landing aim lower maximum std")
        for value in landing["std_lower_max_m"]
    ]
    upper_maximum = [
        _finite(value, label="landing aim upper maximum std")
        for value in landing["std_upper_max_m"]
    ]
    for index in range(2):
        # Expand, never shrink, the hard support enough to contain both the
        # requested centre and its initial sampling width.
        minimum[index] = min(
            minimum[index], center[index] - lower_initial[index]
        )
        maximum[index] = max(
            maximum[index], center[index] + upper_initial[index]
        )
        lower_room = center[index] - minimum[index]
        upper_room = maximum[index] - center[index]
        lower_maximum[index] = max(
            lower_initial[index],
            min(lower_maximum[index], lower_room),
        )
        upper_maximum[index] = max(
            upper_initial[index],
            min(upper_maximum[index], upper_room),
        )
    landing["center_w_xy_m"] = center
    landing["min_w_xy_m"] = minimum
    landing["max_w_xy_m"] = maximum
    landing["std_lower_initial_m"] = lower_initial
    landing["std_upper_initial_m"] = upper_initial
    landing["std_lower_max_m"] = lower_maximum
    landing["std_upper_max_m"] = upper_maximum
    effective = {key: deepcopy(landing[key]) for key in keys}
    return copied, {
        "requested": {
            "landing_aim_center_x_m": center_x_m,
            "landing_aim_center_y_m": center_y_m,
            "landing_aim_std_m": std_m,
        },
        "original": original,
        "effective": effective,
        "counter_rally_y_semantics": (
            "landing_y_is_determined_by_the_reverse_incoming_ray_and_the_"
            "sampled_landing_x;the_manifest_y_arm_remains_inactive"
        ),
        "claim": (
            "in_memory_diagnostic_override_only_not_an_artifact_or_"
            "training_authorization"
        ),
    }


def _float_tuple(value: Sequence[object]) -> Tuple[float, ...]:
    return tuple(float(component) for component in value)


def _norm(value: Sequence[object]) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in value))


def _quantiles(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {
            "count": 0,
            **{"p{}".format(percent): None for percent in PERCENTILES},
            "min": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    percentile_values = np.percentile(array, PERCENTILES)
    return {
        "count": int(array.size),
        **{
            "p{}".format(percent): float(value)
            for percent, value in zip(PERCENTILES, percentile_values)
        },
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _sample_row(sample: Any, index: int) -> Dict[str, object]:
    return {
        "proposal_index": index,
        "sample_id": sample.sample_id,
        "birth_id": sample.birth_id,
        "sampling_stratum": sample.sampling_stratum,
        "frontier_arm": sample.frontier_arm,
        "contact_w_m": list(_float_tuple(sample.contact_w_m)),
        "time_to_contact_s": float(sample.time_to_contact_s),
        "incoming_speed_mps": float(sample.incoming_speed_mps),
        "incoming_direction_b_yaw": list(
            _float_tuple(sample.incoming_direction_b_yaw)
        ),
        "incoming_velocity_w_mps": list(
            _float_tuple(sample.incoming_velocity_w_mps)
        ),
        "spin_w_radps": list(_float_tuple(sample.spin_w_radps)),
        "base_start_w_m": list(_float_tuple(sample.base_start_w_m)),
        "base_goal_w_m": list(_float_tuple(sample.base_goal_w_m)),
        "landing_aim_w_xy_m": list(
            _float_tuple(sample.landing_aim_w_xy_m)
        ),
        "status": "PENDING",
        "rejection_reason": None,
    }


def _run_tape(arguments: argparse.Namespace) -> Dict[str, object]:
    repo_root = Path(arguments.repo_root).resolve(strict=True)
    materializer = _load_materializer(repo_root)
    bundle_path = _resolve_user_path(repo_root, arguments.bundle)
    bundle = materializer._read_json(bundle_path, label="N=1 bundle")
    bundle_sha256 = materializer._sha256_file(bundle_path)
    if arguments.expected_bundle_sha256 is not None:
        expected_bundle = materializer._require_sha256(
            arguments.expected_bundle_sha256,
            label="expected bundle SHA-256",
        )
        if bundle_sha256 != expected_bundle:
            raise TeacherRateTapeError(
                "bundle bytes changed: expected {}, got {}".format(
                    expected_bundle, bundle_sha256
                )
            )
    if (
        bundle.get("schema_version") != 2
        or bundle.get("artifact_type") != "n1_contact_training_bundle_v2"
    ):
        raise TeacherRateTapeError(
            "bundle must be n1_contact_training_bundle_v2"
        )
    action_id = bundle.get("action_id")
    action_uid = bundle.get("action_uid")
    scope = bundle.get("scope")
    if (
        type(action_id) is not str
        or isinstance(action_uid, bool)
        or type(action_uid) is not int
        or type(scope) is not str
    ):
        raise TeacherRateTapeError(
            "bundle action identity/scope is malformed"
        )

    manifest_path, manifest_sha256 = _verify_reference(
        repo_root=repo_root,
        materializer=materializer,
        reference=bundle.get("manifest"),
        label="N=1 manifest",
    )
    if arguments.manifest is not None:
        explicit_manifest = _resolve_user_path(
            repo_root, arguments.manifest
        )
        if explicit_manifest != manifest_path:
            raise TeacherRateTapeError(
                "--manifest does not match the bundle's sealed manifest path"
            )
    profile_path, profile_sha256 = _verify_reference(
        repo_root=repo_root,
        materializer=materializer,
        reference=bundle.get("profile_pins"),
        label="profile pins",
    )
    prototype_path, prototype_sha256 = _verify_reference(
        repo_root=repo_root,
        materializer=materializer,
        reference=bundle.get("prototype"),
        label="prototype",
    )
    motion_path, motion_sha256 = _verify_reference(
        repo_root=repo_root,
        materializer=materializer,
        reference=bundle.get("motion"),
        label="motion",
    )

    modules = materializer._load_preflight_mdp_package(repo_root)
    manifest_module = modules["action_ball_manifest"]
    adapter_module = modules["action_ball_profile_adapter"]
    sampling_module = modules["action_ball_sampling"]
    counter_module = modules["counter_rally"]
    continuous = modules["continuous_questions"]
    contact_geometry = modules["racket_contact_geometry"]
    virtual_ball = modules["virtual_ball"]

    loaded = manifest_module.load_action_ball_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        verify_referenced_assets=True,
        repo_root=repo_root,
    )
    if (
        loaded.manifest.action_order != (action_id,)
        or loaded.manifest.actions[0].action_uid != action_uid
    ):
        raise TeacherRateTapeError(
            "bundle and strict manifest action identity disagree"
        )
    manifest_mapping = materializer._read_json(
        manifest_path, label="N=1 manifest"
    )
    action_mapping = manifest_mapping["actions"][0]
    if (
        action_mapping.get("motion_path")
        != bundle["motion"].get("path")
        or action_mapping.get("motion_sha256") != motion_sha256
    ):
        raise TeacherRateTapeError(
            "manifest and bundle motion identity disagree"
        )
    profile_document = materializer._read_json(
        profile_path, label="profile pins"
    )
    objective_sha256 = loaded.manifest.counter_rally_objective.sha256
    profile_pin = materializer._verify_profile_pins(
        repo_root=repo_root,
        path=profile_path,
        expected_sha256=profile_sha256,
        geometry=contact_geometry,
        objective_sha256=objective_sha256,
    )
    if (
        profile_pin["solver_profile_sha256"]
        != bundle["profile_pins"].get("solver_profile_sha256")
        or profile_pin["physics_profile_sha256"]
        != bundle["profile_pins"].get("physics_profile_sha256")
    ):
        raise TeacherRateTapeError(
            "bundle and verified profile-pins identities disagree"
        )

    prototype = materializer._read_json(
        prototype_path, label="prototype"
    )
    prototype_row = _one_prototype_row(
        prototype, scope=scope, action_id=action_id
    )
    nominal_face_speed_mps = _finite(
        prototype_row["racket_face_center_speed_nominal_mps"],
        label="prototype nominal racket face-centre speed",
    )
    original_face_speed_min_mps = _finite(
        prototype_row["racket_face_center_speed_min_mps"],
        label="prototype minimum racket face-centre speed",
    )
    original_face_speed_max_mps = _finite(
        prototype_row["racket_face_center_speed_max_mps"],
        label="prototype maximum racket face-centre speed",
    )
    if arguments.racket_face_speed_scale is None:
        effective_face_speed_min_mps = original_face_speed_min_mps
        effective_face_speed_max_mps = original_face_speed_max_mps
        requested_face_speed_scale = None
    else:
        requested_face_speed_scale = _finite(
            arguments.racket_face_speed_scale,
            label="racket face-centre speed scale",
        )
        if requested_face_speed_scale <= 0.0:
            raise TeacherRateTapeError(
                "racket face-centre speed scale must be positive"
            )
        fixed_face_speed_mps = (
            requested_face_speed_scale * nominal_face_speed_mps
        )
        effective_face_speed_min_mps = fixed_face_speed_mps
        effective_face_speed_max_mps = fixed_face_speed_mps
    face_speed_override = {
        "requested_scale": requested_face_speed_scale,
        "nominal_face_center_speed_mps": nominal_face_speed_mps,
        "original_speed_min_mps": original_face_speed_min_mps,
        "original_speed_max_mps": original_face_speed_max_mps,
        "effective_speed_min_mps": effective_face_speed_min_mps,
        "effective_speed_max_mps": effective_face_speed_max_mps,
        "velocity_direction_changed": False,
        "incoming_ball_changed_by_this_override": False,
        "claim": (
            "in_memory_solver_prototype_override_only_task_and_exact_face_"
            "geometry_are_recomputed"
        ),
    }
    strike_frame = (
        None if scope == materializer.SCOPE else int(
            prototype_row["contact_frame"]
        )
    )
    state = materializer._motion_state(
        motion_path=motion_path,
        action=action_mapping,
        geometry=contact_geometry,
        scope=scope,
        strike_frame=strike_frame,
    )
    manifest_override, speed_override = _apply_speed_overrides(
        manifest_mapping,
        center_mps=arguments.incoming_speed_center_mps,
        maximum_mps=arguments.incoming_speed_max_mps,
        lower_std_mps=arguments.incoming_speed_std_lower_mps,
        upper_std_mps=arguments.incoming_speed_std_upper_mps,
    )
    manifest_override, landing_aim_override = (
        _apply_landing_aim_overrides(
            manifest_override,
            center_x_m=arguments.landing_aim_center_x_m,
            center_y_m=arguments.landing_aim_center_y_m,
            std_m=arguments.landing_aim_std_m,
        )
    )
    validated = manifest_module.ActionBallManifest.from_mapping(
        manifest_override
    )
    adapted = adapter_module.adapt_action_ball_manifest(
        validated,
        ready_root_z_by_slot=(
            float(np.asarray(state["ready_root_w_m"])[2]),
        ),
    )
    profile = adapted.profiles[0]
    if int(profile.action_uid) != action_uid:
        raise TeacherRateTapeError(
            "adapted profile changed frozen action identity"
        )

    proposal_count = _positive_int(
        arguments.proposal_count, label="proposal count"
    )
    seed = int(arguments.seed)
    contact_time_step_s = _finite(
        arguments.contact_time_step_s,
        label="contact time step",
    )
    episode_length_s = _finite(
        arguments.episode_length_s,
        label="episode length",
    )
    close_margin_s = _finite(
        arguments.attempt_close_margin_s,
        label="attempt close margin",
    )
    if (
        contact_time_step_s <= 0.0
        or episode_length_s <= 0.0
        or close_margin_s <= 0.0
    ):
        raise TeacherRateTapeError(
            "contact time step, episode length, and close margin must be "
            "positive"
        )

    base_yaw = float(state["ready_yaw_rad"])
    levels = sampling_module.DomainLevels()
    sampler = sampling_module.ActionBallSampler(
        adapted.profiles,
        seed=seed,
        sampling_mixture=sampling_module.SamplingMixture(),
        contact_time_step_s=contact_time_step_s,
        diagnostic_unauthorized=True,
    )
    births = tuple(
        sampler.reserve_birth(
            action_uid=action_uid,
            domain_epoch=0,
            levels=levels,
            base_yaw_rad=base_yaw,
        )
        for _ in range(proposal_count)
    )
    samples = tuple(
        sampler.sample(
            birth=birth,
            action_uid=action_uid,
            domain_epoch=0,
            levels=levels,
            base_yaw_rad=base_yaw,
        )
        for birth in births
    )
    rows = [_sample_row(sample, index) for index, sample in enumerate(samples)]

    if validated.counter_rally_objective is None:
        raise TeacherRateTapeError(
            "N=1 manifest lacks counter-rally objective"
        )
    objective = counter_module.CounterRallyObjectiveProfile.from_mapping(
        manifest_override["counter_rally_objective"]
    )
    eligible_samples: List[Any] = []
    eligible_indices: List[int] = []
    reasons = Counter()
    for index, sample in enumerate(samples):
        precheck = counter_module.precheck_counter_rally_fixed_solver_proposal(
            frozen_action_uid=action_uid,
            solver_action_uid=action_uid,
            expected_objective_profile_sha256=objective.sha256,
            base_goal_env_xy_m=sample.base_goal_w_m[:2],
            base_yaw_env_rad=base_yaw,
            contact_offset_b_yaw_m=(
                sample.contact_offset_from_base_goal_b_yaw_m
            ),
            incoming_direction_b_yaw=sample.incoming_direction_b_yaw[:2],
            incoming_ball_speed_at_contact_mps=float(
                sample.incoming_speed_mps
            ),
            landing_depth_env_x_m=float(sample.landing_aim_w_xy_m[0]),
            profile=objective,
        )
        if not precheck.eligible_for_solver:
            reason = str(precheck.rejection_reason)
            rows[index]["status"] = "REJECTED"
            rows[index]["rejection_reason"] = reason
            reasons[reason] += 1
            continue
        eligible_samples.append(sample)
        eligible_indices.append(index)

    try:
        import torch
    except ImportError as error:
        raise TeacherRateTapeError(
            "Torch is required for the batched fixed-action solver"
        ) from error
    if arguments.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = arguments.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise TeacherRateTapeError(
            "CUDA device requested but torch.cuda.is_available() is false"
        )
    device = torch.device(device_name)

    pins_cfg = profile_document["cfg"]
    planes = profile_document["planes"]
    venue_source = profile_document["physics_payload"]["venue_source"]
    venue_path, _ = materializer._resolve_repo_file(
        repo_root,
        venue_source["path"],
        label="venue physics source",
    )
    venue_params = virtual_ball.load_venue_params(str(venue_path))
    solver_cfg = continuous.ContinuousQuestionCfg(
        fixed_direction=True,
        n_iters=int(pins_cfg["cq_n_iters"]),
        tol_m=float(pins_cfg["cq_tol_m"]),
        speed_budget=float(pins_cfg["cq_speed_budget"]),
    )

    if eligible_samples:
        dtype = torch.float32
        count = len(eligible_samples)
        prototype_tensors = types.SimpleNamespace(
            v_hat_b=torch.tensor(
                [prototype_row["racket_face_center_velocity_hat_b"]],
                dtype=dtype,
                device=device,
            ),
            speed_min=torch.tensor(
                [effective_face_speed_min_mps],
                dtype=dtype,
                device=device,
            ),
            speed_max=torch.tensor(
                [effective_face_speed_max_mps],
                dtype=dtype,
                device=device,
            ),
            face_sign=torch.tensor(
                [int(action_mapping["mount_normal_sign"])],
                dtype=dtype,
                device=device,
            ),
        )
        clip_ids = torch.zeros(count, dtype=torch.long, device=device)
        contact = torch.tensor(
            [sample.contact_w_m for sample in eligible_samples],
            dtype=dtype,
            device=device,
        )
        incoming = torch.tensor(
            [
                sample.incoming_velocity_w_mps
                for sample in eligible_samples
            ],
            dtype=dtype,
            device=device,
        )
        spin = torch.tensor(
            [sample.spin_w_radps for sample in eligible_samples],
            dtype=dtype,
            device=device,
        )
        aim = torch.tensor(
            [sample.landing_aim_w_xy_m for sample in eligible_samples],
            dtype=dtype,
            device=device,
        )
        reference_quat = np.asarray(
            state["reference_racket_quat_wxyz"], dtype=np.float64
        )
        reference_normal = (
            materializer._quat_to_rotation(reference_quat)
            @ np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        )
        ref_normal = torch.tensor(
            np.repeat(reference_normal[None, :], count, axis=0),
            dtype=dtype,
            device=device,
        )
        base_quat = torch.tensor(
            [materializer._yaw_quaternion(base_yaw)] * count,
            dtype=dtype,
            device=device,
        )
        solved = continuous.solve_proposals(
            clip_ids,
            contact,
            incoming,
            spin,
            aim,
            ref_normal,
            protos=prototype_tensors,
            base_quat=base_quat,
            prm=venue_params,
            surface_z=float(planes["surface_z"]),
            net_x=float(planes["net_x"]),
            net_top_z=float(planes["net_top_z"]),
            cfg=solver_cfg,
            h=float(pins_cfg["vb_rollout_h"]),
            n_steps=int(pins_cfg["vb_rollout_steps"]),
        )
        ordinary_ok = solved.ok.detach().cpu().tolist()
        reason_codes = (
            solved.proposals.reason_code.detach().cpu().tolist()
        )
        velocity_rows = solved.v_racket.detach().cpu().tolist()
        normal_rows = solved.n_racket.detach().cpu().tolist()
    else:
        ordinary_ok = []
        reason_codes = []
        velocity_rows = []
        normal_rows = []

    reason_schema = tuple(continuous._CONTINUOUS_REASONS)
    reference_quat_tuple = _float_tuple(
        state["reference_racket_quat_wxyz"]
    )
    reference_omega = _float_tuple(
        state["reference_racket_angular_velocity_w_radps"]
    )
    teacher_rate_geometry_solved: List[float] = []
    teacher_rate_admitted: List[float] = []
    face_speed_solver_ok: List[float] = []
    face_speed_admitted: List[float] = []
    site_speed_geometry_solved: List[float] = []
    site_speed_admitted: List[float] = []
    ttc_all = [float(sample.time_to_contact_s) for sample in samples]
    ttc_admitted: List[float] = []
    prewait_timing_evaluated: List[float] = []
    prewait_admitted: List[float] = []
    horizon_required_evaluated: List[float] = []
    horizon_required_admitted: List[float] = []
    horizon_slack_evaluated: List[float] = []
    horizon_slack_admitted: List[float] = []

    for solved_row, (sample, original_index) in enumerate(
        zip(eligible_samples, eligible_indices)
    ):
        row = rows[original_index]
        if not bool(ordinary_ok[solved_row]):
            code = int(reason_codes[solved_row])
            reason = (
                reason_schema[code]
                if 0 <= code < len(reason_schema)
                else "ordinary_solver_unknown"
            )
            row["status"] = "REJECTED"
            row["rejection_reason"] = reason
            reasons[reason] += 1
            continue
        target_face_velocity = _float_tuple(
            velocity_rows[solved_row]
        )
        target_face_speed = _norm(target_face_velocity)
        face_speed_solver_ok.append(target_face_speed)
        row["target_face_center_velocity_w_mps"] = list(
            target_face_velocity
        )
        row["target_face_center_speed_mps"] = target_face_speed
        row["solved_raw_a_normal_w"] = list(
            _float_tuple(normal_rows[solved_row])
        )

        birth_x = continuous.ball_birth_x_lower_bound_m(
            float(sample.contact_w_m[0]),
            float(sample.incoming_velocity_w_mps[0]),
            float(sample.time_to_contact_s),
        )
        row["ball_birth_x_lower_bound_m"] = float(birth_x)
        if birth_x < (
            float(planes["net_x"])
            + float(continuous.BALL_BIRTH_NET_MARGIN_M)
        ):
            reason = "ball_birth_not_beyond_net"
            row["status"] = "REJECTED"
            row["rejection_reason"] = reason
            reasons[reason] += 1
            continue

        geometry_kwargs = {
            "ball_contact_w_m": _float_tuple(sample.contact_w_m),
            "racket_face_center_velocity_w_mps": target_face_velocity,
            "solved_raw_a_normal_w": _float_tuple(
                normal_rows[solved_row]
            ),
            "mount_normal_sign": int(
                action_mapping["mount_normal_sign"]
            ),
            "reference_racket_quat_wxyz": reference_quat_tuple,
            "reference_racket_angular_velocity_w_radps": reference_omega,
            "reference_racket_site_speed_mps": float(
                profile.reference_racket_site_speed_mps
            ),
        }
        try:
            unrestricted = contact_geometry.solve_exact_face_contact(
                **geometry_kwargs,
                teacher_rate_min=1.0e-9,
                teacher_rate_max=1.0e9,
            )
        except contact_geometry.ExactFaceContactGeometryError as error:
            reason = str(error.reason)
            row["status"] = "REJECTED"
            row["rejection_reason"] = reason
            reasons[reason] += 1
            continue
        raw_rate = float(unrestricted.teacher_rate)
        raw_site_speed = _norm(
            unrestricted.racket_site_velocity_w_mps
        )
        teacher_rate_geometry_solved.append(raw_rate)
        site_speed_geometry_solved.append(raw_site_speed)
        row["teacher_rate_unrestricted"] = raw_rate
        row["target_racket_site_speed_mps"] = raw_site_speed
        row["target_racket_site_velocity_w_mps"] = list(
            _float_tuple(unrestricted.racket_site_velocity_w_mps)
        )
        try:
            geometry_solution = (
                contact_geometry.solve_exact_face_contact(
                    **geometry_kwargs,
                    teacher_rate_min=float(profile.teacher_rate_min),
                    teacher_rate_max=float(profile.teacher_rate_max),
                )
            )
        except contact_geometry.ExactFaceContactGeometryError as error:
            reason = str(error.reason)
            if reason == "teacher_rate_out_of_bounds":
                reason = (
                    "teacher_rate_below_min"
                    if raw_rate < float(profile.teacher_rate_min)
                    else "teacher_rate_above_max"
                )
            row["status"] = "REJECTED"
            row["rejection_reason"] = reason
            reasons[reason] += 1
            continue

        teacher_rate = float(geometry_solution.teacher_rate)
        scaled_t_hit = (
            float(profile.reference_t_hit_s) / teacher_rate
        )
        scaled_t_cycle = (
            float(profile.reference_t_cycle_s) / teacher_rate
        )
        prewait = float(sample.time_to_contact_s) - scaled_t_hit
        horizon_required = prewait + scaled_t_cycle + close_margin_s
        horizon_slack = episode_length_s - horizon_required
        prewait_timing_evaluated.append(prewait)
        horizon_required_evaluated.append(horizon_required)
        horizon_slack_evaluated.append(horizon_slack)
        row.update(
            {
                "teacher_rate": teacher_rate,
                "scaled_t_hit_s": scaled_t_hit,
                "scaled_t_cycle_s": scaled_t_cycle,
                "pre_swing_wait_s": prewait,
                "episode_horizon_required_s": horizon_required,
                "episode_horizon_slack_s": horizon_slack,
            }
        )
        if prewait < float(profile.reaction_margin_s):
            reason = "pre_swing_wait_below_reaction_margin"
        elif prewait > 1.0:
            reason = "pre_swing_wait_above_one_second"
        elif horizon_required > episode_length_s + 1.0e-12:
            reason = "cycle_exceeds_episode_horizon"
        else:
            reason = None
        if reason is not None:
            row["status"] = "REJECTED"
            row["rejection_reason"] = reason
            reasons[reason] += 1
            continue

        row["status"] = "ADMITTED"
        teacher_rate_admitted.append(teacher_rate)
        face_speed_admitted.append(target_face_speed)
        site_speed_admitted.append(raw_site_speed)
        ttc_admitted.append(float(sample.time_to_contact_s))
        prewait_admitted.append(prewait)
        horizon_required_admitted.append(horizon_required)
        horizon_slack_admitted.append(horizon_slack)

    admitted_count = sum(row["status"] == "ADMITTED" for row in rows)
    rejected_count = sum(row["status"] == "REJECTED" for row in rows)
    if admitted_count + rejected_count != proposal_count:
        raise AssertionError(
            "diagnostic tape does not conserve proposals"
        )
    tape_payload = [
        {
            "sample_id": sample.sample_id,
            "identity": sample.to_identity_receipt(),
        }
        for sample in samples
    ]
    return {
        "schema_version": 1,
        "kind": "n1_fixed_action_teacher_rate_proposal_tape_diagnostic_v1",
        "status": "PASS",
        "claims": {
            "read_only": True,
            "fixed_action_identity": True,
            "selector_executed": False,
            "training_authorization": False,
            "curriculum_promotion_evidence": False,
            "policy_evidence": False,
        },
        "inputs": {
            "repo_root": str(repo_root),
            "bundle_path": str(bundle_path),
            "bundle_sha256": bundle_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "profile_pins_path": str(profile_path),
            "profile_pins_sha256": profile_sha256,
            "prototype_path": str(prototype_path),
            "prototype_sha256": prototype_sha256,
            "motion_path": str(motion_path),
            "motion_sha256": motion_sha256,
            "action_id": action_id,
            "action_uid": action_uid,
            "scope": scope,
        },
        "execution": {
            "python_version": sys.version.split()[0],
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "device": str(device),
            "dtype": "float32",
            "seed": seed,
            "proposal_count": proposal_count,
            "contact_time_step_s": contact_time_step_s,
            "proposal_tape_sha256": (
                materializer._canonical_sha256(tape_payload)
            ),
            "implementation_source_sha256": {
                name: materializer._sha256_file(
                    repo_root
                    / materializer.MDP_RELATIVE_DIR
                    / "{}.py".format(name)
                )
                for name in sorted(modules)
            },
        },
        "speed_override": speed_override,
        "landing_aim_override": landing_aim_override,
        "racket_face_speed_override": face_speed_override,
        "effective_contract": {
            "teacher_rate_min": float(profile.teacher_rate_min),
            "teacher_rate_max": float(profile.teacher_rate_max),
            "reference_t_hit_s": float(profile.reference_t_hit_s),
            "reference_t_cycle_s": float(profile.reference_t_cycle_s),
            "reference_racket_site_speed_mps": float(
                profile.reference_racket_site_speed_mps
            ),
            "reaction_margin_s": float(profile.reaction_margin_s),
            "episode_length_s": episode_length_s,
            "attempt_close_margin_s": close_margin_s,
            "prototype_face_speed_min_mps": effective_face_speed_min_mps,
            "prototype_face_speed_max_mps": effective_face_speed_max_mps,
        },
        "counts": {
            "proposed": proposal_count,
            "precheck_eligible": len(eligible_samples),
            "geometry_solved_unrestricted": len(
                teacher_rate_geometry_solved
            ),
            "admitted": admitted_count,
            "rejected": rejected_count,
            "admit_rate": admitted_count / proposal_count,
            "rejection_reasons": dict(sorted(reasons.items())),
        },
        "distribution": {
            "teacher_rate_unrestricted_geometry_solved": _quantiles(
                teacher_rate_geometry_solved
            ),
            "teacher_rate_admitted": _quantiles(
                teacher_rate_admitted
            ),
            "target_face_center_speed_solver_ok_mps": _quantiles(
                face_speed_solver_ok
            ),
            "target_face_center_speed_admitted_mps": _quantiles(
                face_speed_admitted
            ),
            "target_racket_site_speed_geometry_solved_mps": _quantiles(
                site_speed_geometry_solved
            ),
            "target_racket_site_speed_admitted_mps": _quantiles(
                site_speed_admitted
            ),
            "time_to_contact_all_s": _quantiles(ttc_all),
            "time_to_contact_admitted_s": _quantiles(ttc_admitted),
            "pre_swing_wait_timing_evaluated_s": _quantiles(
                prewait_timing_evaluated
            ),
            "pre_swing_wait_admitted_s": _quantiles(
                prewait_admitted
            ),
            "episode_horizon_required_timing_evaluated_s": _quantiles(
                horizon_required_evaluated
            ),
            "episode_horizon_required_admitted_s": _quantiles(
                horizon_required_admitted
            ),
            "episode_horizon_slack_timing_evaluated_s": _quantiles(
                horizon_slack_evaluated
            ),
            "episode_horizon_slack_admitted_s": _quantiles(
                horizon_slack_admitted
            ),
        },
        "proposal_tape": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT_DEFAULT),
        help="exact clean training checkout whose solver/source pins are used",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="N=1 bundle path, absolute or relative to --repo-root",
    )
    parser.add_argument(
        "--expected-bundle-sha256",
        help="optional independent expected SHA-256 for the bundle bytes",
    )
    parser.add_argument(
        "--manifest",
        help=(
            "optional explicit manifest path; must equal the bundle-sealed "
            "path"
        ),
    )
    parser.add_argument(
        "--proposal-count",
        type=int,
        default=DEFAULT_PROPOSAL_COUNT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--contact-time-step-s",
        type=float,
        default=DEFAULT_CONTACT_TIME_STEP_S,
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=DEFAULT_EPISODE_LENGTH_S,
    )
    parser.add_argument(
        "--attempt-close-margin-s",
        type=float,
        default=DEFAULT_ATTEMPT_CLOSE_MARGIN_S,
    )
    parser.add_argument(
        "--landing-aim-center-x-m",
        type=float,
        help="diagnostic-only landing-aim world-X centre override",
    )
    parser.add_argument(
        "--landing-aim-center-y-m",
        type=float,
        help="diagnostic-only landing-aim world-Y centre override",
    )
    parser.add_argument(
        "--landing-aim-std-m",
        type=float,
        help=(
            "diagnostic-only symmetric landing-aim initial std for both "
            "axes and sides, for example 0.01"
        ),
    )
    parser.add_argument(
        "--racket-face-speed-scale",
        type=float,
        help=(
            "diagnostic-only fixed face-centre speed multiplier: both "
            "solver bounds become scale times prototype nominal while the "
            "teacher velocity direction and incoming ball tape stay fixed"
        ),
    )
    parser.add_argument(
        "--incoming-speed-center-mps",
        type=float,
        help="diagnostic-only incoming ball speed centre override",
    )
    parser.add_argument(
        "--incoming-speed-max-mps",
        type=float,
        help=(
            "diagnostic-only incoming ball hard maximum override; it may "
            "not exceed the counter-rally objective maximum"
        ),
    )
    parser.add_argument(
        "--incoming-speed-std-mps",
        type=float,
        help=(
            "diagnostic-only symmetric initial std override; mutually "
            "exclusive with side-specific std flags"
        ),
    )
    parser.add_argument(
        "--incoming-speed-std-lower-mps",
        type=float,
        help="diagnostic-only lower-side initial std override",
    )
    parser.add_argument(
        "--incoming-speed-std-upper-mps",
        type=float,
        help="diagnostic-only upper-side initial std override",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the batched solver: auto, cpu, cuda, cuda:N",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.incoming_speed_std_mps is not None and (
        arguments.incoming_speed_std_lower_mps is not None
        or arguments.incoming_speed_std_upper_mps is not None
    ):
        parser.error(
            "--incoming-speed-std-mps is mutually exclusive with "
            "side-specific std flags"
        )
    if arguments.incoming_speed_std_mps is not None:
        arguments.incoming_speed_std_lower_mps = (
            arguments.incoming_speed_std_mps
        )
        arguments.incoming_speed_std_upper_mps = (
            arguments.incoming_speed_std_mps
        )
    try:
        result = _run_tape(arguments)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "kind": (
                "n1_fixed_action_teacher_rate_proposal_tape_diagnostic_v1"
            ),
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(
            json.dumps(
                failure,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
