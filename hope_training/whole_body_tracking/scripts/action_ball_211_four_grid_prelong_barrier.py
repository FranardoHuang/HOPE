#!/usr/bin/env python3
"""Fail-closed all-four scale4096 barrier for the A211/C211 Isaac grid."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
FOUR_GRID_FILE = SCRIPT_DIR / "action_ball_211_four_grid_contract.py"
A_LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_a211_four_arm_diagnostic.py"
C_LAUNCHER_FILE = SCRIPT_DIR / "launch_action_ball_c211_diagnostic.py"

SCHEMA_VERSION = 3
KIND = "action_ball_211_four_grid_scale4096_aggregate_receipt_v3"
AUTHORIZATION = "long4096_launch_barrier_only"
SCALE_BUDGET = [4096, 5, 1]
PHYSICAL_FALL_REASONS = ("base_fell_tilt", "base_too_low")
PHYSICAL_FALL_PHASES = (
    "hidden_wait",
    "revealed_pre_strike",
    "post_strike",
)
PRODUCER_SAFETY_KEYS = (
    "observed_ppo_updates",
    "actual_hard_edge_event_count",
    "actual_hard_terminal_count",
    "joint_qdes_forbidden_terminal_count",
    "joint_actual_forbidden_terminal_count",
    "strict_hard_termination_count",
    "table_contact_count",
    "nonfinite_count",
    "base_fell_tilt_terminal_count",
    "base_too_low_terminal_count",
    "physical_fall_by_reason_phase",
    "table_contact_by_phase",
    "task_wait_started_by_update",
    "task_wait_started_count",
    "task_reveal_reached_by_update",
    "task_reveal_reached_count",
)
STRICT_ZERO_SAFETY_KEYS = (
    "actual_hard_edge_event_count",
    "actual_hard_terminal_count",
    "joint_qdes_forbidden_terminal_count",
    "joint_actual_forbidden_terminal_count",
    "strict_hard_termination_count",
    "nonfinite_count",
)
TERMINAL_ACCEPTANCE_POLICY = {
    "implementation_safety": {
        "required_zero_counters": list(STRICT_ZERO_SAFETY_KEYS),
    },
    "behavioral_terminations": {
        "reasons": [*PHYSICAL_FALL_REASONS, "robot_hit_table"],
        "phases": list(PHYSICAL_FALL_PHASES),
        "finite_scale_cutoff": None,
        "acceptance": (
            "complete nonnegative total/phase attribution and exact conservation; "
            "the five-update constructibility gate does not require a fresh policy "
            "to have zero fall/table terminations"
        ),
    },
    "survival_denominators": {
        "updates": 5,
        "each_update_requires_nonzero": [
            "task_wait_started_count",
            "task_reveal_reached_count",
        ],
        "aggregate_must_equal_sum_of_updates": True,
    },
}
# 2026-08-05 层级对齐(exp §5.6 第 7 条):death -300.0 -> -10.0(post-dt -6.0 -> -0.2)。
# 本表是 four-grid manifest matched_contract.soft_weights 的第三份手抄;文件尾部
# (_F 载入之后)加了一道等值断言,今后任何一份漂了都会在 import 期就炸,不必等 launch。
EXPECTED_SAFETY_REWARD_ECONOMY = {
    "death_penalty": -10.0,
    "qdes_limit": -5.0,
    "qdes_projection": -5.0,
    "joint_limit": -5.0,
}
# 运行期 RewardManager 的项名与 selector 的 soft_weights 键名不同,但值必须逐一相等。
# 原本是把同一组价格手抄成第二份常量;改为按固定键名映射推导,两份副本漂移的可能被消除。
# 注:qdes_projection_penalty 取的是 params.objective_weight(-5.0),不是 RewardManager
# 固定的 -1.0 曝光权重 —— 与 launcher 的 _runtime_effective_soft_weights /
# _runtime_soft_weights 的口径一致,故与 prelong 语义表里的 -1.0 并非同一个量。
EXPECTED_RUNTIME_SAFETY_WEIGHTS = {
    runtime_name: EXPECTED_SAFETY_REWARD_ECONOMY[economy_name]
    for economy_name, runtime_name in (
        ("death_penalty", "death_penalty"),
        ("joint_limit", "joint_limit"),
        ("qdes_limit", "qdes_limit_barrier"),
        ("qdes_projection", "qdes_projection_penalty"),
    )
}
AUTHORIZED_LAYOUT = {
    # 2026-08-05 第二轴改版(第二次,exp §5.6.2d):cell_id 随轴改名,GPU 布局本身未变——
    # A 对同卡 gpu0、C 对同卡 gpu1、gpu2 留给 MuJoCo,不占用。
    "gpu0": [
        "A0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off",
        "A1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on",
    ],
    "gpu1": [
        "C0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off",
        "C1-base-safety-standard-init-sigma1p0-proprio-obs-noise-on",
    ],
    "gpu2": "reserved_for_mujoco",
    "legacy_vendor_v2_same_gpu": "prohibited_after_transition_drain",
    "pending_registry_scope": "A211_C211_physical_registry_only",
    "cross_checkout_legacy_local_pending_visibility": "not_provided",
}
TRANSITION_INVARIANT = {
    "required_before_first_grid_launch": [
        "drain_all_legacy_pending_reservations_and_live_trainers_on_all_three_gpus",
        "disable_legacy_N1_A225_C225_launchers_for_the_grid_window",
        "permit_A211_C211_writers_from_one_fresh_exact_checkout_only",
    ],
    "machine_verified_by_this_receipt": False,
    "operator_preflight_evidence_required": True,
    "cross_checkout_or_legacy_writer_atomicity_claimed": False,
}
SHARED_BINDING_KEYS = (
    "source_commit_sha",
    "four_grid_manifest_content_sha256",
    "motion_sha256",
    "dynamic_ready_artifact_file_sha256",
    "dynamic_ready_artifact_content_sha256",
    "dynamic_ready_nominal_receipt_file_sha256",
    "dynamic_ready_nominal_receipt_content_sha256",
    "teacher_frame0_artifact_file_sha256",
    "teacher_frame0_artifact_content_sha256",
    "split_ready_reset_wait_claim_sha256",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class BarrierRefused(RuntimeError):
    pass


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise BarrierRefused("cannot load barrier dependency %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_F = _load("_action_ball_211_barrier_grid", FOUR_GRID_FILE)

# 2026-08-05(exp §5.6):EXPECTED_SAFETY_REWARD_ECONOMY 必须逐字等于 four-grid manifest 的
# matched_contract.soft_weights。它写在 _F 载入之前(常量区),没法直接推导,所以在这里补一道
# import 期等值断言 —— 把"手抄副本静默漂移"变成"import 就炸"。
if (
    EXPECTED_SAFETY_REWARD_ECONOMY
    != _F.manifest()["matched_contract"]["soft_weights"]
):  # pragma: no cover - import-time ratchet
    raise BarrierRefused(
        "four-grid barrier safety reward economy differs from the sealed manifest"
    )

# 同理:AUTHORIZED_LAYOUT 的两张卡上的 cell_id 是手抄副本(常量区写在 _F 之前)。
# 2026-08-05 cell_id 改名后,这道 import 期断言保证布局表与权威同步,gpu2 仍空给 MuJoCo。
if (
    AUTHORIZED_LAYOUT["gpu0"] != list(_F.FAMILY_CELL_IDS["A211"])
    or AUTHORIZED_LAYOUT["gpu1"] != list(_F.FAMILY_CELL_IDS["C211"])
    or AUTHORIZED_LAYOUT["gpu2"] != "reserved_for_mujoco"
):  # pragma: no cover - import-time ratchet
    raise BarrierRefused(
        "four-grid barrier authorized GPU layout differs from the sealed manifest"
    )


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BarrierRefused("barrier value is not canonical JSON") from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Any, keys: Sequence[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        raise BarrierRefused("%s keys differ" % name)
    return dict(value)


def _sha(value: Any, *, name: str, nonzero: bool = False) -> str:
    if type(value) is not str or SHA_RE.fullmatch(value) is None:
        raise BarrierRefused("%s must be lowercase SHA-256" % name)
    if nonzero and value == "0" * 64:
        raise BarrierRefused("%s must not be the zero sentinel" % name)
    return value


def _counter(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise BarrierRefused("%s must be a nonnegative integer" % name)
    return value


def _phase_counts(value: Any, *, name: str) -> dict[str, int]:
    row = _exact(value, PHYSICAL_FALL_PHASES, name=name)
    return {
        phase: _counter(row[phase], name="%s %s" % (name, phase))
        for phase in PHYSICAL_FALL_PHASES
    }


def _validate_terminal_safety(value: Any) -> dict[str, Any]:
    """Consume the exact 16-key A/C terminal producer schema.

    Numerical/implementation failures are strict-zero.  Falls, too-low, and
    robot/table contact are truthful behavioral termination telemetry: the
    five-update constructibility gate requires complete phase attribution and
    conservation, not the circular claim that a fresh policy already produces
    zero such events.
    """

    row = _exact(value, PRODUCER_SAFETY_KEYS, name="terminal safety counters")
    observed_updates = _counter(
        row["observed_ppo_updates"], name="observed PPO updates"
    )
    if observed_updates != SCALE_BUDGET[1]:
        raise BarrierRefused("terminal safety counters do not cover five updates")
    for key in STRICT_ZERO_SAFETY_KEYS:
        if _counter(row[key], name="terminal %s" % key) != 0:
            raise BarrierRefused(
                "terminal implementation safety counter %s is nonzero" % key
            )

    raw_reason_phase = _exact(
        row["physical_fall_by_reason_phase"],
        PHYSICAL_FALL_REASONS,
        name="physical-fall reason-by-phase counters",
    )
    reason_phase: dict[str, dict[str, int]] = {}
    for reason in PHYSICAL_FALL_REASONS:
        total = _counter(
            row["%s_terminal_count" % reason],
            name="terminal %s total" % reason,
        )
        phases = _phase_counts(
            raw_reason_phase[reason], name="terminal %s phases" % reason
        )
        if sum(phases.values()) != total:
            raise BarrierRefused(
                "terminal %s total/phase counters do not conserve" % reason
            )
        reason_phase[reason] = phases

    table_total = _counter(
        row["table_contact_count"], name="terminal robot_hit_table total"
    )
    table_phases = _phase_counts(
        row["table_contact_by_phase"], name="terminal robot_hit_table phases"
    )
    if sum(table_phases.values()) != table_total:
        raise BarrierRefused(
            "terminal robot_hit_table total/phase counters do not conserve"
        )

    update_lists: dict[str, list[int]] = {}
    for prefix in ("task_wait_started", "task_reveal_reached"):
        raw_updates = row["%s_by_update" % prefix]
        if type(raw_updates) is not list or len(raw_updates) != SCALE_BUDGET[1]:
            raise BarrierRefused(
                "terminal %s denominator must cover five updates" % prefix
            )
        updates = [
            _counter(value, name="terminal %s update %d" % (prefix, index))
            for index, value in enumerate(raw_updates)
        ]
        if any(value == 0 for value in updates):
            raise BarrierRefused(
                "terminal %s denominator must be nonzero in every update" % prefix
            )
        total = _counter(
            row["%s_count" % prefix], name="terminal %s total" % prefix
        )
        if sum(updates) != total:
            raise BarrierRefused(
                "terminal %s per-update/aggregate counters do not conserve" % prefix
            )
        update_lists[prefix] = updates

    return {
        **copy.deepcopy(row),
        "physical_fall_by_reason_phase": reason_phase,
        "table_contact_by_phase": table_phases,
        "task_wait_started_by_update": update_lists["task_wait_started"],
        "task_reveal_reached_by_update": update_lists["task_reveal_reached"],
    }


def _validate_prelong_behavioral_binding(
    value: Any, *, safety: Mapping[str, Any]
) -> None:
    """Require the pre-long gate to bind the same non-circular safety ledger."""

    if type(value) is not dict or type(value.get("gate")) is not dict:
        raise BarrierRefused("pre-long gate binding is incomplete")
    gate = value["gate"]
    if (
        gate.get("status") != "PASS"
        or gate.get("ppo_updates") != SCALE_BUDGET[1]
        or gate.get("authorization") != "pre_long_terminal_telemetry_only"
    ):
        raise BarrierRefused("pre-long gate did not PASS the five-update policy")
    gate_safety = gate.get("safety")
    if type(gate_safety) is not dict:
        raise BarrierRefused("pre-long gate safety ledger is missing")
    expected_strict = {key: safety[key] for key in STRICT_ZERO_SAFETY_KEYS}
    if (
        gate_safety.get("strict_zero_counters") != expected_strict
        or gate_safety.get("task_wait_started_by_update")
        != safety["task_wait_started_by_update"]
        or gate_safety.get("task_wait_started_count")
        != safety["task_wait_started_count"]
        or gate_safety.get("task_reveal_reached_by_update")
        != safety["task_reveal_reached_by_update"]
        or gate_safety.get("task_reveal_reached_count")
        != safety["task_reveal_reached_count"]
        or gate_safety.get("table_contact_count") != safety["table_contact_count"]
        or gate_safety.get("table_contact_by_phase")
        != safety["table_contact_by_phase"]
        or gate_safety.get("unknown_attribution_count") != 0
    ):
        raise BarrierRefused("pre-long gate and terminal safety counters differ")

    survival = gate.get("survival_denominators")
    behavioral = (
        survival.get("behavioral_terminations")
        if type(survival) is dict
        else None
    )
    if type(behavioral) is not dict or set(behavioral) != {
        *PHYSICAL_FALL_REASONS,
        "robot_hit_table",
    }:
        raise BarrierRefused("pre-long behavioral termination ledger is incomplete")
    for reason in (*PHYSICAL_FALL_REASONS, "robot_hit_table"):
        observed = behavioral[reason]
        total = (
            safety["table_contact_count"]
            if reason == "robot_hit_table"
            else safety["%s_terminal_count" % reason]
        )
        phases = (
            safety["table_contact_by_phase"]
            if reason == "robot_hit_table"
            else safety["physical_fall_by_reason_phase"][reason]
        )
        if (
            type(observed) is not dict
            or observed.get("total_count") != total
            or observed.get("by_phase") != phases
            or observed.get("acceptance_threshold") is not None
        ):
            raise BarrierRefused(
                "pre-long %s attribution/cutoff differs" % reason
            )


def _stable_json(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    try:
        before = path.lstat()
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise BarrierRefused("%s cannot be read" % name) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or path.resolve(strict=True) != path
        or before.st_size <= 0
        or before.st_size > (16 << 20)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise BarrierRefused("%s is not one stable bounded file" % name)
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BarrierRefused("%s is not strict JSON" % name) from exc
    if type(value) is not dict or raw != canonical_bytes(value) + b"\n":
        raise BarrierRefused("%s is not canonical newline JSON" % name)
    return value, hashlib.sha256(raw).hexdigest()


def _pin(value: Any, *, name: str) -> tuple[dict[str, str], Path]:
    row = _exact(value, ("path", "sha256"), name=name)
    path_text = row["path"]
    if (
        type(path_text) is not str
        or not path_text
        or "\x00" in path_text
        or "\n" in path_text
        or not os.path.isabs(path_text)
        or os.path.normpath(path_text) != path_text
    ):
        raise BarrierRefused("%s path must be normalized absolute" % name)
    digest = _sha(row["sha256"], name="%s SHA" % name)
    return {"path": path_text, "sha256": digest}, Path(path_text)


def _shared_binding(lineage: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, str]:
    artifact = lineage.get("dynamic_ready_artifact")
    receipt = lineage.get("dynamic_ready_nominal_receipt")
    teacher = lineage.get("teacher_frame0_artifact")
    reset_wait = lineage.get("split_ready_reset_wait_authority")
    manifest = payload.get("bundle", {}).get("isaac_four_grid_manifest")
    if any(
        type(value) is not dict
        for value in (artifact, receipt, teacher, reset_wait, manifest)
    ):
        raise BarrierRefused("lineage shared-input closure is incomplete")
    dynamic_identity = reset_wait.get("dynamic_ready")
    nominal_identity = reset_wait.get("nominal_hold_receipt")
    if type(dynamic_identity) is not dict or type(nominal_identity) is not dict:
        raise BarrierRefused("split-ready reset/wait authority is incomplete")
    source = payload.get("spec", {}).get("source")
    row = {
        "source_commit_sha": source.get("commit_sha") if type(source) is dict else None,
        "four_grid_manifest_content_sha256": manifest.get("content_sha256"),
        "motion_sha256": lineage.get("motion", {}).get("sha256"),
        "dynamic_ready_artifact_file_sha256": artifact.get("sha256"),
        "dynamic_ready_artifact_content_sha256": dynamic_identity.get(
            "content_sha256"
        ),
        "dynamic_ready_nominal_receipt_file_sha256": receipt.get("sha256"),
        "dynamic_ready_nominal_receipt_content_sha256": nominal_identity.get(
            "content_sha256"
        ),
        "teacher_frame0_artifact_file_sha256": teacher.get("sha256"),
        "teacher_frame0_artifact_content_sha256": lineage.get(
            "teacher_frame0_artifact_content_sha256"
        ),
        "split_ready_reset_wait_claim_sha256": reset_wait.get("claim_sha256"),
    }
    return _validate_shared_binding(row)


def _validate_shared_binding(value: Any) -> dict[str, str]:
    row = _exact(value, SHARED_BINDING_KEYS, name="shared binding")
    if type(row["source_commit_sha"]) is not str or COMMIT_RE.fullmatch(
        row["source_commit_sha"]
    ) is None:
        raise BarrierRefused("shared source commit is invalid")
    for key in SHARED_BINDING_KEYS[1:]:
        _sha(row[key], name="shared %s" % key, nonzero=True)
    if (
        row["four_grid_manifest_content_sha256"] != _F.CONTENT_SHA256
        or row["motion_sha256"] != _F.CANONICAL_MOTION_SHA256
    ):
        raise BarrierRefused("shared sampler-manifest/motion authority differs")
    return {key: row[key] for key in SHARED_BINDING_KEYS}


def _family_modules():
    return {
        "A211": _load("_action_ball_211_barrier_a_runtime", A_LAUNCHER_FILE),
        "C211": _load("_action_ball_211_barrier_c_runtime", C_LAUNCHER_FILE),
    }


def _claim_for_result(module: Any, result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    namespace = module._B._absolute_path(
        result["namespace"], name="barrier scale namespace", must_exist=True
    )
    raw = (namespace / "launch_claim.json").read_bytes()
    outer = module._B._strict_json_bytes(raw, name="barrier launch claim")
    if raw != module._B._canonical_bytes(outer) + b"\n":
        raise BarrierRefused("barrier launch claim is not canonical")
    outer = module._exact_dict(
        outer,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="barrier launch claim",
    )
    payload = outer["canonical_payload"]
    if (
        outer["schema_version"] != module.SCHEMA_VERSION
        or outer["kind"] != module.CLAIM_KIND
        or outer["launch_claim_sha256"] != result["launch_claim_sha256"]
        or module.canonical_sha256(payload) != result["launch_claim_sha256"]
    ):
        raise BarrierRefused("barrier launch claim digest differs")
    return outer, payload


def _reaudit_runtime_safety_reward_economy(
    module: Any,
    *,
    family: str,
    selector: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, float]:
    artifact = materialization.get("runtime_effective_reward_artifact")
    try:
        pin, document = module._canonical_external_json(
            artifact, name="%s aggregate runtime reward artifact" % family
        )
        validated = module._OLD._validate_reward_materialization(pin)
        terms = document["terms"]
        if family == "A211":
            module._require_effective_learnability_terms(terms)
            observed = module._runtime_effective_soft_weights(terms, arm=selector)
        else:
            module._require_c211_outcome_terms(terms)
            observed = module._runtime_soft_weights(terms, recipe=selector)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise BarrierRefused(
            "%s runtime reward artifact cannot be re-audited" % family
        ) from exc
    except Exception as exc:
        accepted = tuple(
            error
            for error in (
                getattr(module, "LaunchRefused", None),
                getattr(getattr(module, "_OLD", None), "LaunchRefused", None),
            )
            if isinstance(error, type)
        )
        if accepted and isinstance(exc, accepted):
            raise BarrierRefused(
                "%s runtime reward artifact cannot be re-audited" % family
            ) from exc
        raise
    if (
        selector.get("soft_weights") != EXPECTED_SAFETY_REWARD_ECONOMY
        or observed != EXPECTED_RUNTIME_SAFETY_WEIGHTS
        or materialization.get("runtime_soft_weights")
        != EXPECTED_RUNTIME_SAFETY_WEIGHTS
        or validated.get("artifact") != pin
        or validated.get("effective_reward_recipe_sha256")
        != materialization.get("runtime_effective_reward_sha256")
        or validated.get("term_count")
        != materialization.get("runtime_effective_reward_term_count")
    ):
        raise BarrierRefused("%s runtime safety reward economy differs" % family)
    return dict(EXPECTED_SAFETY_REWARD_ECONOMY)


def _audit_cell(
    cell_id: str,
    result_pin: Mapping[str, Any],
    *,
    checkout: Path,
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    family = "A211" if cell_id in _F.FAMILY_CELL_IDS["A211"] else "C211"
    module = modules[family]
    normalized_pin, result = module._validated_stage_result(
        result_pin,
        expected_stage="scale4096",
        name="%s aggregate scale result" % cell_id,
    )
    outer, payload = _claim_for_result(module, result)
    spec, lineage, selector = module._revalidate_claim_payload(
        payload, claimed=True
    )
    observed_cell = selector["arm_id"] if family == "A211" else selector["recipe_id"]
    expected_gpu_index = 0 if family == "A211" else 1
    gpu = spec.get("gpu")
    output_contract = payload.get("output_contract")
    if (
        observed_cell != cell_id
        or selector.get("soft_weights") != EXPECTED_SAFETY_REWARD_ECONOMY
        or spec["stage"] != "scale4096"
        or spec["source"]["checkout"] != str(checkout)
        or spec["namespace"] != result["namespace"]
        or result["predecessor_result"] is not None
        or type(gpu) is not dict
        or gpu.get("index") != expected_gpu_index
        or gpu.get("require_empty") is not False
        or spec.get(module.COLOCATION_SPEC_KEY) is not True
        or type(output_contract) is not dict
        or output_contract.get("rate_evidence_eligible") is not False
        or output_contract.get("speed_benchmark_eligible") is not False
        or output_contract.get("colocated_stage") != "scale4096"
    ):
        raise BarrierRefused("%s scale claim selector/source differs" % cell_id)
    inputs = payload["materialization_inputs"]
    materialization = (
        inputs["arm_materialization"]
        if family == "A211"
        else inputs["reward_materialization"]
    )
    safety_reward_economy = _reaudit_runtime_safety_reward_economy(
        module,
        family=family,
        selector=selector,
        materialization=materialization,
    )
    if family == "A211":
        module._validate_predecessor_result(
            normalized_pin,
            checkout=checkout,
            expected_stage="scale4096",
            materialization=materialization,
            policy_materialization=inputs["policy_recipe_materialization"],
            oracle32=inputs["oracle32_receipt"],
        )
    else:
        module._validate_scale_predecessor(
            normalized_pin,
            checkout=checkout,
            materialization=materialization,
            policy=inputs["policy_recipe_materialization"],
            oracle32=inputs["oracle32_receipt"],
        )
    if (
        tuple(module.PHYSICAL_FALL_REASONS) != PHYSICAL_FALL_REASONS
        or tuple(module.PHYSICAL_FALL_PHASES) != PHYSICAL_FALL_PHASES
        or tuple(module._P.STRICT_ZERO_SAFETY_COUNTERS)
        != STRICT_ZERO_SAFETY_KEYS
    ):
        raise BarrierRefused(
            "%s launcher/pre-long safety schema differs from aggregate policy"
            % family
        )
    terminal = result["terminal_acceptance"]
    checkpoint = terminal["checkpoint"]
    safety = _validate_terminal_safety(terminal["safety_counters"])
    gate = terminal["prelong_gate"]
    _validate_prelong_behavioral_binding(gate, safety=safety)
    if (
        checkpoint.get("filename_iteration") != 5
        or checkpoint.get("embedded_iteration") != 5
        or checkpoint.get("all_tensors_finite") is not True
        or type(checkpoint.get("tensor_groups")) is not dict
        or not checkpoint["tensor_groups"]
        or type(gate) is not dict
        or gate.get("gate", {}).get("status") != "PASS"
    ):
        raise BarrierRefused("%s scale terminal evidence is incomplete" % cell_id)
    row = {
        "cell_id": cell_id,
        "task_family": family,
        "scale_result": normalized_pin,
        "launch_claim_sha256": outer["launch_claim_sha256"],
        "launch_result_content_sha256": result["content_sha256"],
        "gpu": {
            "index": gpu["index"],
            "uuid": gpu["uuid"],
            "colocation_opt_in": True,
            "rate_evidence_eligible": False,
        },
        "lineage_sha256": lineage["lineage_sha256"],
        "terminal_acceptance_content_sha256": terminal["content_sha256"],
        "model_5": {
            "sha256": checkpoint["sha256"],
            "filename_iteration": 5,
            "embedded_iteration": 5,
            "all_tensors_finite": True,
        },
        "safety_counters": safety,
        "safety_reward_economy": safety_reward_economy,
        "prelong_gate": {
            "status": "PASS",
            "content_sha256": gate["content_sha256"],
        },
        "shared_binding": _shared_binding(lineage, payload),
    }
    # 这里不再自校验:document_from_audits 会对每一格逐行跑 _validate_audit_row,
    # 原来一行审计要过两遍同一个校验器(4 格 = 8 次),现役唯一校验点在
    # document_from_audits,四格各一次,产出的规范化文档与原来完全一致。
    return row


def _validate_audit_row(value: Any, *, expected_cell: str) -> dict[str, Any]:
    row = _exact(
        value,
        (
            "cell_id",
            "task_family",
            "scale_result",
            "launch_claim_sha256",
            "launch_result_content_sha256",
            "gpu",
            "lineage_sha256",
            "terminal_acceptance_content_sha256",
            "model_5",
            "safety_counters",
            "safety_reward_economy",
            "prelong_gate",
            "shared_binding",
        ),
        name="aggregate cell",
    )
    expected_family = "A211" if expected_cell in _F.FAMILY_CELL_IDS["A211"] else "C211"
    if row["cell_id"] != expected_cell or row["task_family"] != expected_family:
        raise BarrierRefused("aggregate cell order/family differs")
    pin, _path = _pin(row["scale_result"], name="aggregate scale result")
    gpu = _exact(
        row["gpu"],
        ("index", "uuid", "colocation_opt_in", "rate_evidence_eligible"),
        name="aggregate GPU binding",
    )
    expected_gpu_index = 0 if expected_family == "A211" else 1
    if (
        type(gpu["index"]) is not int
        or isinstance(gpu["index"], bool)
        or gpu["index"] != expected_gpu_index
        or type(gpu["uuid"]) is not str
        or not gpu["uuid"].startswith("GPU-")
        or gpu["colocation_opt_in"] is not True
        or gpu["rate_evidence_eligible"] is not False
    ):
        raise BarrierRefused("aggregate GPU layout/rate exclusion differs")
    for key in (
        "launch_claim_sha256",
        "launch_result_content_sha256",
        "lineage_sha256",
        "terminal_acceptance_content_sha256",
    ):
        _sha(row[key], name="aggregate %s" % key, nonzero=True)
    model = _exact(
        row["model_5"],
        ("sha256", "filename_iteration", "embedded_iteration", "all_tensors_finite"),
        name="aggregate model_5",
    )
    if (
        model["filename_iteration"] != 5
        or model["embedded_iteration"] != 5
        or model["all_tensors_finite"] is not True
    ):
        raise BarrierRefused("aggregate model_5 identity differs")
    _sha(model["sha256"], name="aggregate model_5 SHA", nonzero=True)
    safety = _validate_terminal_safety(row["safety_counters"])
    if row["safety_reward_economy"] != EXPECTED_SAFETY_REWARD_ECONOMY:
        raise BarrierRefused("aggregate safety reward economy differs")
    gate = _exact(
        row["prelong_gate"], ("status", "content_sha256"), name="aggregate gate"
    )
    if gate["status"] != "PASS":
        raise BarrierRefused("aggregate gate did not PASS")
    _sha(gate["content_sha256"], name="aggregate gate SHA", nonzero=True)
    return {
        **copy.deepcopy(row),
        "scale_result": pin,
        "safety_counters": safety,
        "shared_binding": _validate_shared_binding(row["shared_binding"]),
    }


def document_from_audits(audits: Mapping[str, Any]) -> dict[str, Any]:
    if type(audits) is not dict or set(audits) != set(_F.CELL_IDS):
        raise BarrierRefused("aggregate must contain all four cells exactly once")
    rows = [
        _validate_audit_row(audits[cell_id], expected_cell=cell_id)
        for cell_id in _F.CELL_IDS
    ]
    shared = rows[0]["shared_binding"]
    if any(row["shared_binding"] != shared for row in rows[1:]):
        raise BarrierRefused("aggregate shared concrete source/ready binding differs")
    if len({row["scale_result"]["sha256"] for row in rows}) != 4:
        raise BarrierRefused("aggregate scale result is duplicated")
    if len({row["launch_claim_sha256"] for row in rows}) != 4:
        raise BarrierRefused("aggregate launch claim is duplicated")
    if (
        len({row["gpu"]["uuid"] for row in rows[:2]}) != 1
        or len({row["gpu"]["uuid"] for row in rows[2:]}) != 1
        or rows[0]["gpu"]["uuid"] == rows[2]["gpu"]["uuid"]
    ):
        raise BarrierRefused("aggregate family GPU UUID layout differs")
    cells = []
    for row in rows:
        cell = dict(row)
        cell.pop("shared_binding")
        cells.append(cell)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "PASS",
        "diagnostic_unauthorized": True,
        "authorization": AUTHORIZATION,
        "four_grid_manifest_content_sha256": _F.CONTENT_SHA256,
        "scale_budget": list(SCALE_BUDGET),
        "terminal_acceptance_policy": copy.deepcopy(TERMINAL_ACCEPTANCE_POLICY),
        "safety_reward_economy": dict(EXPECTED_SAFETY_REWARD_ECONOMY),
        "authorized_layout": dict(AUTHORIZED_LAYOUT),
        "transition_invariant": copy.deepcopy(TRANSITION_INVARIANT),
        "shared_binding": shared,
        "cells": cells,
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def build_receipt_document(
    cell_result_pins: Mapping[str, Any],
    *,
    checkout: Path,
    modules: Mapping[str, Any] | None = None,
    audit_cell: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if type(cell_result_pins) is not dict or set(cell_result_pins) != set(_F.CELL_IDS):
        raise BarrierRefused("builder requires one result pin for every grid cell")
    family_modules = _family_modules() if modules is None else modules
    auditor = _audit_cell if audit_cell is None else audit_cell
    audits = {
        cell_id: auditor(
            cell_id,
            cell_result_pins[cell_id],
            checkout=checkout,
            modules=family_modules,
        )
        for cell_id in _F.CELL_IDS
    }
    return document_from_audits(audits)


def validate_receipt(
    value: Any,
    *,
    checkout: Path,
    modules: Mapping[str, Any] | None = None,
    audit_cell: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pin, path = _pin(value, name="four-grid aggregate receipt")
    document, observed_sha = _stable_json(path, name="four-grid aggregate receipt")
    if observed_sha != pin["sha256"]:
        raise BarrierRefused("four-grid aggregate receipt file SHA differs")
    row = _exact(
        document,
        (
            "schema_version",
            "kind",
            "status",
            "diagnostic_unauthorized",
            "authorization",
            "four_grid_manifest_content_sha256",
            "scale_budget",
            "terminal_acceptance_policy",
            "safety_reward_economy",
            "authorized_layout",
            "transition_invariant",
            "shared_binding",
            "cells",
            "content_sha256",
        ),
        name="four-grid aggregate receipt",
    )
    if type(row["cells"]) is not list or len(row["cells"]) != 4:
        raise BarrierRefused("four-grid aggregate receipt cell count differs")
    pins = {}
    for expected_cell, cell in zip(_F.CELL_IDS, row["cells"]):
        if type(cell) is not dict or cell.get("cell_id") != expected_cell:
            raise BarrierRefused("four-grid aggregate receipt cell order differs")
        pins[expected_cell] = cell.get("scale_result")
    recomputed = build_receipt_document(
        pins,
        checkout=checkout,
        modules=modules,
        audit_cell=audit_cell,
    )
    if row != recomputed:
        raise BarrierRefused("four-grid aggregate receipt differs from live re-audit")
    return {"artifact": pin, **recomputed}


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise BarrierRefused("aggregate output must be fresh") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    for cell_id in _F.CELL_IDS:
        safe = cell_id.split("-", 1)[0].lower()
        parser.add_argument("--%s-result-path" % safe, required=True)
        parser.add_argument("--%s-result-sha256" % safe, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkout = Path(args.checkout)
    output = Path(args.output)
    pins = {}
    for cell_id in _F.CELL_IDS:
        safe = cell_id.split("-", 1)[0].lower()
        pins[cell_id] = {
            "path": getattr(args, safe + "_result_path"),
            "sha256": getattr(args, safe + "_result_sha256"),
        }
    document = build_receipt_document(pins, checkout=checkout)
    _write_exclusive(output, document)
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256((canonical_bytes(document) + b"\n")).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
