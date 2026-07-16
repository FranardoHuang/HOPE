"""Strict, opt-in timing-paper adapter for :mod:`isaac_bank_exam`.

The legacy evaluator remains native-clock unless an exact timing paper is
supplied.  This adapter validates that paper against the already validated
BankExam schedule, then activates the existing R14 float clock *after* the
native external-item installer has established clip/frame-0 state.  It does
not call ``play.py`` and therefore cannot inherit that export path's native-
clock reset.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from isaac_bank_exam_adapter import IsaacBankExamError


ARTIFACT_TYPE = "phase1-timing-exam-paper"
CONTRACT_ID = "phase1-timing-exam-0p5-k100-v1"
SIDE_ORDER = ("forehand", "backhand")
INITIAL_STATE_ID = "nominal-frame0-zero-velocity-v1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
DIAGNOSTIC_REASONS = (
    "timing paper uniform-phase time laws are not TOPP/dynamics-certified",
    "Isaac analytic timing lane has no self-hit or illegal table/net-contact instrumentation",
    "fixed-question policy exam bypasses the production planner; infeasibility is unmeasured",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IsaacBankExamError(f"timing paper is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IsaacBankExamError(f"duplicate timing-paper JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise IsaacBankExamError(f"non-finite timing-paper JSON constant: {value}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except IsaacBankExamError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IsaacBankExamError(f"cannot read strict timing paper {path}: {exc}") from exc


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise IsaacBankExamError(f"{label} must be one lowercase SHA-256")
    return value


def _keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise IsaacBankExamError(f"{label} schema changed: {actual}")
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IsaacBankExamError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IsaacBankExamError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise IsaacBankExamError(f"{label} must be finite{' and positive' if positive else ''}")
    return result


def _semantic_sha(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("paper_semantic_sha256", None)
    return canonical_sha256(payload)


def validate_timing_paper_document(value: Any) -> dict[str, Any]:
    top_keys = {
        "schema_version",
        "artifact_type",
        "contract_id",
        "spec_file_sha256",
        "spec_content_sha256",
        "source_schedule",
        "paper",
        "scoring",
        "execution",
        "rows",
        "paper_semantic_sha256",
    }
    document = dict(_keys(value, top_keys, "timing paper"))
    if (
        document["schema_version"] != 1
        or document["artifact_type"] != ARTIFACT_TYPE
        or document["contract_id"] != CONTRACT_ID
    ):
        raise IsaacBankExamError("timing paper schema/type/id changed")
    for field in ("spec_file_sha256", "spec_content_sha256", "paper_semantic_sha256"):
        _sha(document[field], field)
    if _semantic_sha(document) != document["paper_semantic_sha256"]:
        raise IsaacBankExamError("timing paper semantic SHA mismatch")

    source = _keys(
        document["source_schedule"],
        {
            "file_sha256",
            "semantic_sha256",
            "question_id_order_sha256",
            "bank_sha256",
            "bank_source_family_sha256",
        },
        "timing paper source_schedule",
    )
    for field in source:
        _sha(source[field], f"source_schedule.{field}")

    paper = _keys(
        document["paper"],
        {"policy_rate_hz", "baseline", "time_laws", "tts_sweep_plan"},
        "timing paper policy",
    )
    if paper["policy_rate_hz"] != 50:
        raise IsaacBankExamError("timing paper requires a 50 Hz policy rate")
    baseline = _keys(
        paper["baseline"],
        {
            "human_name",
            "tts_seconds",
            "tts_ticks",
            "initial_state_id",
            "expected_feasible",
            "feasibility_status",
            "scheduled_attempts",
            "per_side",
            "source_hold_steps_are_replaced",
            "source_attempt_seed_and_question_order_are_preserved",
        },
        "timing baseline",
    )
    if (
        baseline["tts_seconds"] != 0.5
        or baseline["tts_ticks"] != 25
        or baseline["initial_state_id"] != INITIAL_STATE_ID
        or baseline["expected_feasible"] is not None
        or baseline["feasibility_status"] != "hypothesis_not_certified"
        or baseline["scheduled_attempts"] != 100
        or baseline["per_side"] != {"forehand": 50, "backhand": 50}
        or baseline["source_hold_steps_are_replaced"] is not True
        or baseline["source_attempt_seed_and_question_order_are_preserved"] is not True
    ):
        raise IsaacBankExamError("timing baseline differs from the frozen 0.5 s K100 contract")

    laws = paper["time_laws"]
    if not isinstance(laws, list) or len(laws) != 2:
        raise IsaacBankExamError("timing paper requires exactly two side-specific time laws")
    law_keys = {
        "time_law_id",
        "side",
        "native_contact_ticks",
        "speed_scale",
        "contact_tts_ticks",
        "contact_tts_seconds",
        "topp_or_dynamics_certified",
    }
    by_side = {}
    for index, raw in enumerate(laws):
        law = _keys(raw, law_keys, f"time law {index}")
        side = law["side"]
        if side not in SIDE_ORDER or side in by_side:
            raise IsaacBankExamError("time laws must bind forehand/backhand exactly once")
        native = _plain_int(law["native_contact_ticks"], f"{side} native ticks", minimum=1)
        tts_ticks = _plain_int(law["contact_tts_ticks"], f"{side} TTS ticks", minimum=1)
        speed = _finite(law["speed_scale"], f"{side} speed", positive=True)
        if (
            tts_ticks != 25
            or _finite(law["contact_tts_seconds"], f"{side} TTS seconds") != 0.5
            or not math.isclose(native / speed, tts_ticks, rel_tol=0.0, abs_tol=1e-9)
            or law["topp_or_dynamics_certified"] is not False
        ):
            raise IsaacBankExamError(f"{side} time law differs from the diagnostic 0.5 s contract")
        if not isinstance(law["time_law_id"], str) or not law["time_law_id"].startswith("v4rg-"):
            raise IsaacBankExamError(f"{side} time_law_id is invalid")
        by_side[side] = law
    if set(by_side) != set(SIDE_ORDER):
        raise IsaacBankExamError("time laws do not cover both sides")

    scoring = _keys(
        document["scoring"],
        {
            "denominator",
            "per_side_pass_count",
            "per_side_total",
            "one_sided_wilson_confidence",
            "one_sided_wilson_lower_bound_at_31_of_50",
            "composite",
            "safety_zero_tolerance_fields",
            "formal_gate_requires_evaluation_contract_exact",
        },
        "timing scoring",
    )
    denominator = scoring["denominator"]
    if (
        denominator.get("policy") != "all_scheduled_attempts"
        or denominator.get("aggregate") != 100
        or denominator.get("forehand") != 50
        or denominator.get("backhand") != 50
        or denominator.get("missing_invalid_reset_or_infeasible_attempt_counts_as_failure") is not True
        or denominator.get("censoring_allowed") is not False
        or scoring["per_side_pass_count"] != 31
        or scoring["per_side_total"] != 50
        or scoring["formal_gate_requires_evaluation_contract_exact"] is not True
    ):
        raise IsaacBankExamError("timing all-attempt denominator or 31/50 gate changed")
    composite = scoring["composite"]
    if (
        composite.get("returned_required") is not True
        or composite.get("position_error_m_strict_lt") != 0.075
        or composite.get("velocity_error_mps_strict_lt") != 0.5
        or composite.get("signed_normal_error_deg_strict_lt") != 15.0
        or composite.get("unsigned_or_oriented_plane_fallback_allowed") is not False
    ):
        raise IsaacBankExamError("timing composite thresholds changed")
    if scoring["safety_zero_tolerance_fields"] != [
        "physical_fall",
        "self_hit",
        "illegal_table_or_net_contact",
        "reset_or_teleport",
        "deadline_shifted",
    ]:
        raise IsaacBankExamError("timing safety fields changed")

    execution = _keys(
        document["execution"],
        {
            "materializer_launches_no_evaluator",
            "output_must_not_exist",
            "atomic_no_replace",
            "isaac_diagnostic_evaluator_authorized",
            "isaac_diagnostic_requires_allow_inexact_contract",
            "trainer_authorized",
            "judge_authorized",
            "planner_or_runner_authorized",
            "stop_or_promote_authorized",
            "deployment_authorized",
            "real_robot_authorized",
        },
        "timing execution",
    )
    if (
        execution["materializer_launches_no_evaluator"] is not True
        or execution["output_must_not_exist"] is not True
        or execution["atomic_no_replace"] is not True
        or execution["isaac_diagnostic_evaluator_authorized"] is not True
        or execution["isaac_diagnostic_requires_allow_inexact_contract"] is not True
        or any(
            execution[field] is not False
            for field in (
                "trainer_authorized",
                "judge_authorized",
                "planner_or_runner_authorized",
                "stop_or_promote_authorized",
                "deployment_authorized",
                "real_robot_authorized",
            )
        )
    ):
        raise IsaacBankExamError("timing paper does not authorize only the inexact Isaac diagnostic")

    rows = document["rows"]
    row_keys = {
        "schedule_index",
        "question_id",
        "side",
        "initial_state_id",
        "tts_seconds",
        "tts_ticks",
        "time_law_id",
        "expected_feasible",
        "feasibility_status",
        "bank_row",
        "attempt_seed",
        "repeat",
        "source_hold_steps",
        "source_hold_steps_replaced",
    }
    if not isinstance(rows, list) or len(rows) != 100:
        raise IsaacBankExamError("timing paper must contain exactly 100 rows")
    ids = set()
    counts = Counter()
    for index, raw in enumerate(rows):
        row = _keys(raw, row_keys, f"timing row {index}")
        if row["schedule_index"] != index:
            raise IsaacBankExamError("timing rows must preserve contiguous source order")
        side = row["side"]
        if side not in SIDE_ORDER:
            raise IsaacBankExamError(f"timing row {index} has invalid side")
        question_id = row["question_id"]
        if (
            not isinstance(question_id, str)
            or not question_id.startswith(f"{side}:")
            or not SHA_RE.fullmatch(question_id.split(":", 1)[1])
            or question_id in ids
        ):
            raise IsaacBankExamError(f"timing row {index} question id is invalid/duplicate")
        ids.add(question_id)
        counts[side] += 1
        law = by_side[side]
        if (
            row["initial_state_id"] != INITIAL_STATE_ID
            or row["tts_seconds"] != 0.5
            or row["tts_ticks"] != 25
            or row["time_law_id"] != law["time_law_id"]
            or row["expected_feasible"] is not None
            or row["feasibility_status"] != "hypothesis_not_certified"
            or row["source_hold_steps_replaced"] is not True
            or row["repeat"] != 0
        ):
            raise IsaacBankExamError(f"timing row {index} does not consume the frozen baseline")
        _plain_int(row["bank_row"], f"timing row {index} bank_row")
        _plain_int(row["attempt_seed"], f"timing row {index} attempt_seed")
        _plain_int(row["source_hold_steps"], f"timing row {index} source_hold_steps")
    if dict(counts) != {"forehand": 50, "backhand": 50}:
        raise IsaacBankExamError("timing rows must remain 50/50 by side")
    return document


def load_timing_paper(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_semantic_sha256: str,
) -> dict[str, Any]:
    _sha(expected_file_sha256, "expected timing paper file SHA")
    _sha(expected_semantic_sha256, "expected timing paper semantic SHA")
    if not path.is_file() or sha256_file(path) != expected_file_sha256:
        raise IsaacBankExamError("timing paper file SHA mismatch or file missing")
    paper = validate_timing_paper_document(_load_json(path))
    if paper["paper_semantic_sha256"] != expected_semantic_sha256:
        raise IsaacBankExamError("timing paper semantic SHA differs from requested paper")
    paper["_validated_binding"] = {
        "path": str(path.resolve()),
        "file_sha256": expected_file_sha256,
        "semantic_sha256": expected_semantic_sha256,
    }
    return paper


def validate_paper_schedule_binding(
    paper: Mapping[str, Any],
    *,
    schedule_artifact: Any,
    schedule_path: Path,
) -> None:
    binding = paper.get("_validated_binding")
    if not isinstance(binding, Mapping):
        raise IsaacBankExamError("timing paper was not loaded through the strict file validator")
    source = paper["source_schedule"]
    if not schedule_path.is_file() or sha256_file(schedule_path) != source["file_sha256"]:
        raise IsaacBankExamError("timing paper source schedule file SHA mismatch")
    if (
        str(schedule_artifact.schedule_sha256) != source["semantic_sha256"]
        or str(schedule_artifact.bank_sha256) != source["bank_sha256"]
    ):
        raise IsaacBankExamError("timing paper source schedule semantic/bank binding mismatch")
    items = tuple(schedule_artifact.items)
    if len(items) != len(paper["rows"]):
        raise IsaacBankExamError("timing paper/source schedule length mismatch")
    order = [str(item.question_id) for item in items]
    if canonical_sha256(order) != source["question_id_order_sha256"]:
        raise IsaacBankExamError("timing paper source question-order SHA mismatch")
    for row, item in zip(paper["rows"], items):
        side = SIDE_ORDER[int(item.clip)]
        expected = {
            "schedule_index": int(item.schedule_index),
            "question_id": str(item.question_id),
            "side": side,
            "bank_row": int(item.bank_row),
            "attempt_seed": int(item.attempt_seed),
            "repeat": int(item.repeat),
            "source_hold_steps": int(item.hold_steps),
        }
        mismatch = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
        if mismatch:
            raise IsaacBankExamError(
                f"timing row {item.schedule_index} differs from source schedule: {mismatch}"
            )


def validate_runtime_time_laws(
    paper: Mapping[str, Any],
    *,
    segment_lengths: Sequence[int],
    strike_phases: Sequence[float],
) -> tuple[float, ...]:
    if len(segment_lengths) != 2 or len(strike_phases) != 2:
        raise IsaacBankExamError("0.5 s timing paper requires exactly two runtime motion clips")
    laws = {law["side"]: law for law in paper["paper"]["time_laws"]}
    speeds = []
    for clip, side in enumerate(SIDE_ORDER):
        length = _plain_int(int(segment_lengths[clip]), f"{side} segment length", minimum=2)
        phase = _finite(float(strike_phases[clip]), f"{side} strike phase")
        native = int(round(phase * (length - 1)))
        law = laws[side]
        if native != law["native_contact_ticks"]:
            raise IsaacBankExamError(
                f"{side} runtime native contact tick {native} != paper {law['native_contact_ticks']}"
            )
        speed = float(law["speed_scale"])
        if not math.isclose(native / speed, law["contact_tts_ticks"], rel_tol=0.0, abs_tol=1e-9):
            raise IsaacBankExamError(f"{side} runtime time law does not reach paper deadline")
        speeds.append(speed)
    return tuple(speeds)


def activate_runtime_retiming(
    motion_cmd: Any,
    *,
    env_ids: Any,
    clip_ids: Any,
    paper: Mapping[str, Any],
    segment_lengths: Sequence[int],
    strike_phases: Sequence[float],
    torch_module: Any,
) -> dict[str, Any]:
    """Activate existing R14 state after the native external installer set frame 0."""

    speeds = validate_runtime_time_laws(
        paper, segment_lengths=segment_lengths, strike_phases=strike_phases
    )
    if bool(getattr(motion_cmd, "retiming_active", False)) or getattr(
        motion_cmd, "_speed_per_clip", None
    ) is not None:
        raise IsaacBankExamError("timing rider requires a native-clock command before activation")
    if tuple(float(v) for v in motion_cmd.cfg.speed_scale_range) != (1.0, 1.0):
        raise IsaacBankExamError("timing rider refuses a preexisting random speed range")
    for name in (
        "clip_id",
        "time_steps",
        "time_steps_f",
        "speed_scale",
        "hold_counter",
        "just_resampled",
    ):
        if not hasattr(motion_cmd, name):
            raise IsaacBankExamError(f"timing rider runtime lacks motion_cmd.{name}")
    ids = torch_module.as_tensor(env_ids, device=motion_cmd.device, dtype=torch_module.long).reshape(-1)
    clips = torch_module.as_tensor(clip_ids, device=motion_cmd.device, dtype=torch_module.long).reshape(-1)
    if len(ids) != len(paper["rows"]) or len(clips) != len(ids):
        raise IsaacBankExamError("timing rider env/clip vectors do not match paper K")
    if bool((motion_cmd.hold_counter[ids] != 0).any()):
        raise IsaacBankExamError("timing rider requires zero effective hold after frame-0 install")
    expected_clips = torch_module.as_tensor(
        [0 if row["side"] == "forehand" else 1 for row in paper["rows"]],
        device=motion_cmd.device,
        dtype=torch_module.long,
    )
    if bool((clips != expected_clips).any()) or bool((motion_cmd.clip_id[ids] != clips).any()):
        raise IsaacBankExamError("timing rider clip vector differs from paper/native install")
    starts = motion_cmd.motion.seg_start[clips]
    if bool((motion_cmd.time_steps[ids] != starts).any()) or bool(
        (motion_cmd.time_steps_f[ids] != starts.float()).any()
    ):
        raise IsaacBankExamError("timing rider must start every clip at exact frame 0")
    speed_table = torch_module.as_tensor(speeds, device=motion_cmd.device, dtype=motion_cmd.time_steps_f.dtype)
    motion_cmd._speed_per_clip = speed_table
    motion_cmd.cfg.speed_scale_per_clip = speeds
    motion_cmd.retiming_active = True
    motion_cmd.speed_scale[ids] = speed_table[clips]
    return {
        "schema": "hope.isaac-timing-paper-runtime.v1",
        "paper_file_sha256": paper["_validated_binding"]["file_sha256"],
        "paper_semantic_sha256": paper["paper_semantic_sha256"],
        "speed_scale_per_clip": list(speeds),
        "effective_hold_steps": 0,
        "initial_state_id": INITIAL_STATE_ID,
        "native_external_installer_ran_first": True,
        "r14_float_clock_active": True,
        "play_or_export_path_used": False,
    }


def install_zero_velocity_frame0_reference(
    motion_cmd: Any,
    *,
    env_ids: Any,
    clip_ids: Any,
    paper: Mapping[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    """Make the first actor observation consume frame 0 with zero reference velocity.

    ``MotionCommand`` normally exposes the finite-difference velocity stored in the
    motion file at frame 0.  The accepted v4rg clips have non-zero frame-0 velocity,
    so merely putting the integer/float clocks at ``seg_start`` is not the frozen
    ``nominal-frame0-zero-velocity-v1`` state.  The timing rider owns an evaluator-
    local ``MotionLoader`` instance, so it may zero *only* the two segment-start
    velocity rows in memory.  Positions, later velocity rows, the source NPZ, and
    the saved training configuration remain untouched.

    This must run after the native installer and R14 activation but before
    ``install_external_exam_questions`` refreshes the first actor observation.
    """

    if not isinstance(paper.get("_validated_binding"), Mapping):
        raise IsaacBankExamError("frame-0 install requires a strictly loaded timing paper")
    ids = torch_module.as_tensor(
        env_ids, device=motion_cmd.device, dtype=torch_module.long
    ).reshape(-1)
    clips = torch_module.as_tensor(
        clip_ids, device=motion_cmd.device, dtype=torch_module.long
    ).reshape(-1)
    if len(ids) != len(paper["rows"]) or len(clips) != len(ids):
        raise IsaacBankExamError("frame-0 install env/clip vectors do not match paper K")
    if bool((motion_cmd.hold_counter[ids] != 0).any()) or bool(
        motion_cmd.in_hold[ids].any()
    ):
        raise IsaacBankExamError(
            "zero-velocity frame 0 is a released reference, not the legacy stand hold"
        )
    starts = motion_cmd.motion.seg_start[clips]
    if bool((motion_cmd.time_steps[ids] != starts).any()) or bool(
        (motion_cmd.time_steps_f[ids] != starts.float()).any()
    ):
        raise IsaacBankExamError("frame-0 reference install requires both clocks at seg_start")

    velocity_fields = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
    before_max: dict[str, float] = {}
    unique_starts = torch_module.unique(starts)
    for name in velocity_fields:
        value = getattr(motion_cmd.motion, name, None)
        if value is None or not hasattr(value, "shape") or value.shape[0] <= int(unique_starts.max()):
            raise IsaacBankExamError(f"frame-0 reference lacks motion.{name}")
        selected = value[unique_starts]
        if not bool(torch_module.isfinite(selected).all()):
            raise IsaacBankExamError(f"frame-0 motion.{name} contains NaN/Inf")
        before_max[name] = float(selected.abs().max()) if selected.numel() else 0.0
        value[unique_starts] = 0.0

    reference_velocity_fields = {
        "joint_vel": motion_cmd.joint_vel[ids],
        "body_lin_vel_w": motion_cmd.body_lin_vel_w[ids],
        "body_ang_vel_w": motion_cmd.body_ang_vel_w[ids],
    }
    after_max = {}
    for name, value in reference_velocity_fields.items():
        if not bool(torch_module.isfinite(value).all()):
            raise IsaacBankExamError(f"live frame-0 reference {name} contains NaN/Inf")
        maximum = float(value.abs().max()) if value.numel() else 0.0
        after_max[name] = maximum
        if maximum != 0.0:
            raise IsaacBankExamError(
                f"live frame-0 reference {name} is not exactly zero: max={maximum}"
            )
    expected_joint_pos = motion_cmd.motion.joint_pos[starts]
    if not bool(torch_module.equal(motion_cmd.joint_pos[ids], expected_joint_pos)):
        raise IsaacBankExamError("zeroing frame-0 velocity changed the frame-0 reference pose")
    return {
        "schema": "hope.isaac-timing-frame0-reference.v1",
        "initial_state_id": INITIAL_STATE_ID,
        "segment_start_rows": [int(value) for value in unique_starts.detach().cpu().tolist()],
        "source_velocity_max_abs_before_evaluator_override": before_max,
        "live_reference_velocity_max_abs_after_override": after_max,
        "reference_pose_is_exact_motion_frame0": True,
        "source_npz_or_saved_config_mutated": False,
        "evaluator_local_motion_loader_velocity_rows_overridden": True,
    }


def verify_runtime_retiming_preserved(
    motion_cmd: Any,
    *,
    paper: Mapping[str, Any],
    expected_profile: Mapping[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    """Fail closed if the evaluator lost or replaced the paper's R14 time law."""

    expected = tuple(float(value) for value in expected_profile["speed_scale_per_clip"])
    current = getattr(motion_cmd, "_speed_per_clip", None)
    configured = getattr(motion_cmd.cfg, "speed_scale_per_clip", None)
    if (
        not bool(getattr(motion_cmd, "retiming_active", False))
        or current is None
        or configured is None
        or tuple(float(value) for value in configured) != expected
    ):
        raise IsaacBankExamError("timing evaluator did not preserve the paper R14 time law")
    current_cpu = current.detach().to("cpu")
    expected_tensor = torch_module.as_tensor(expected, dtype=current_cpu.dtype)
    if not bool(torch_module.equal(current_cpu, expected_tensor)):
        raise IsaacBankExamError("runtime per-clip speed table changed during timing evaluation")
    if expected_profile.get("paper_semantic_sha256") != paper["paper_semantic_sha256"]:
        raise IsaacBankExamError("runtime retiming profile no longer binds the timing paper")
    return {
        "preserved_through_finalization": True,
        "speed_scale_per_clip": list(expected),
        "native_clock_fallback_observed": False,
        "play_or_export_path_used": False,
    }


def validate_zero_velocity_ready_state(
    paper: Mapping[str, Any],
    *,
    root_states: Any,
    joint_velocities: Any,
) -> dict[str, Any]:
    if any(row["initial_state_id"] != INITIAL_STATE_ID for row in paper["rows"]):
        raise IsaacBankExamError("timing paper initial-state ids are not uniform frame0")
    root = np.asarray(root_states, dtype=np.float64)
    qd = np.asarray(joint_velocities, dtype=np.float64)
    if root.ndim != 2 or root.shape[0] != len(paper["rows"]) or root.shape[1] != 13:
        raise IsaacBankExamError("timing ready root state must be Kx13")
    if qd.ndim != 2 or qd.shape[0] != len(paper["rows"]):
        raise IsaacBankExamError("timing ready joint velocity must have K rows")
    if not np.isfinite(root).all() or not np.isfinite(qd).all():
        raise IsaacBankExamError("timing ready state contains NaN/Inf")
    root_speed = float(np.max(np.abs(root[:, 7:13])))
    joint_speed = float(np.max(np.abs(qd)))
    if root_speed != 0.0 or joint_speed != 0.0:
        raise IsaacBankExamError(
            f"timing frame0 must be zero velocity, root_max={root_speed}, joint_max={joint_speed}"
        )
    return {
        "initial_state_id": INITIAL_STATE_ID,
        "root_velocity_max_abs": root_speed,
        "joint_velocity_max_abs": joint_speed,
        "exact_zero_velocity": True,
    }


def initialize_timing_record(record: dict[str, Any], row: Mapping[str, Any]) -> None:
    for field in (
        "schedule_index",
        "question_id",
        "side",
        "bank_row",
        "attempt_seed",
        "repeat",
    ):
        if record.get(field) != row[field]:
            raise IsaacBankExamError(f"timing record {row['schedule_index']} differs on {field}")
    record.update(
        {
            "timing_exam_enabled": True,
            "all_attempt_denominator_member": True,
            "eligible": True,
            "planner_infeasible": None,
            "infeasible": None,
            "planner_infeasible_source": "unmeasured_fixed_question_exam_bypasses_planner",
            "deadline_miss": False,
            "deadline_shifted": False,
            "deadline_step": int(row["tts_ticks"]),
            "exact_strike_step": None,
            "initial_state_id": row["initial_state_id"],
            "tts_seconds": float(row["tts_seconds"]),
            "tts_ticks": int(row["tts_ticks"]),
            "time_law_id": row["time_law_id"],
            "expected_feasible": row["expected_feasible"],
            "feasibility_status": row["feasibility_status"],
            "effective_hold_steps": 0,
        }
    )


def observe_timing_deadlines(records: Sequence[dict[str, Any]], *, step: int) -> None:
    for row in records:
        if row.get("timing_exam_enabled") and step >= int(row["deadline_step"]):
            if not bool(row.get("reached_exact", False)):
                row["deadline_miss"] = True


def finalize_timing_records(
    records: Sequence[dict[str, Any]],
    *,
    paper: Mapping[str, Any],
    evaluation_contract_exact: bool,
) -> dict[str, Any]:
    if len(records) != len(paper["rows"]):
        raise IsaacBankExamError("timing ledger does not cover every paper row")
    composite = paper["scoring"]["composite"]
    counts = {side: Counter() for side in SIDE_ORDER}
    for record, paper_row in zip(records, paper["rows"]):
        if record.get("schedule_index") != paper_row["schedule_index"]:
            raise IsaacBankExamError("timing ledger order differs from paper")
        if not bool(record.get("finalized", False)) or bool(record.get("censored", False)):
            raise IsaacBankExamError("timing ledger has an unfinalized/censored attempt")
        exact_step = record.get("exact_strike_step")
        if exact_step is not None:
            exact_step = _plain_int(
                exact_step,
                f"timing record {record['schedule_index']} exact_strike_step",
            )
        record["deadline_shifted"] = bool(
            record.get("reached_exact", False)
            and exact_step != int(record["deadline_step"])
        )
        record["deadline_miss"] = bool(record["deadline_miss"] or not record["reached_exact"])
        record["contact"] = bool(record.get("hit", False))
        success = (
            not record["deadline_miss"]
            and not record["deadline_shifted"]
            and bool(record.get("returned", False))
            and record.get("pos_error_m") is not None
            and record.get("vel_error_mps") is not None
            and record.get("normal_error_deg") is not None
            and float(record["pos_error_m"]) < composite["position_error_m_strict_lt"]
            and float(record["vel_error_mps"]) < composite["velocity_error_mps_strict_lt"]
            and float(record["normal_error_deg"])
            < composite["signed_normal_error_deg_strict_lt"]
        )
        record["composite"] = bool(success)
        record["safety"] = {
            "physical_fall": bool(record.get("physical_fall", False)),
            "self_hit": None,
            "illegal_table_or_net_contact": None,
            "reset_or_teleport": bool(record.get("guard_reset", False)),
            "deadline_shifted": bool(record["deadline_shifted"]),
            "complete": False,
        }
        side = record["side"]
        side_count = counts[side]
        side_count["scheduled"] += 1
        side_count["eligible"] += int(record["eligible"])
        side_count["planner_infeasible"] += int(record["planner_infeasible"] is True)
        side_count["planner_feasibility_unknown"] += int(
            record["planner_infeasible"] is None
        )
        side_count["infeasible"] += int(record["infeasible"] is True)
        side_count["deadline_miss"] += int(record["deadline_miss"])
        side_count["deadline_shifted"] += int(record["deadline_shifted"])
        side_count["contact"] += int(record["contact"])
        side_count["returned"] += int(bool(record["returned"]))
        side_count["composite"] += int(record["composite"])
        side_count["physical_fall"] += int(bool(record["physical_fall"]))
    per_side = {}
    threshold = int(paper["scoring"]["per_side_pass_count"])
    for side in SIDE_ORDER:
        row = dict(counts[side])
        if row.get("scheduled") != 50:
            raise IsaacBankExamError("timing summary lost the 50/side denominator")
        row["pass_count_required"] = threshold
        row["diagnostic_composite_pass"] = row.get("composite", 0) >= threshold
        per_side[side] = row
    performance = all(per_side[side]["diagnostic_composite_pass"] for side in SIDE_ORDER)
    physical_safe = all(per_side[side].get("physical_fall", 0) == 0 for side in SIDE_ORDER)
    return {
        "all_scheduled_attempts_in_denominator": True,
        "aggregate_denominator": 100,
        "per_side": per_side,
        "safety_observation_complete": False,
        "diagnostic_performance_pass": performance and physical_safe,
        "formal_gate_pass": False,
        "formal_gate_blockers": list(DIAGNOSTIC_REASONS),
        "evaluation_contract_exact_input": bool(evaluation_contract_exact),
    }
