"""CPU-only tests for the all-four A211/C211 pre-long barrier."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/action_ball_211_four_grid_prelong_barrier.py"
)
A_LAUNCHER = SCRIPT.parent / "launch_action_ball_a211_four_arm_diagnostic.py"
C_LAUNCHER = SCRIPT.parent / "launch_action_ball_c211_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("four_grid_prelong_barrier", SCRIPT)
barrier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = barrier
SPEC.loader.exec_module(barrier)


def _producer_safety(
    *,
    tilt_phase=None,
    too_low_phase=None,
    table_phase=None,
):
    fall = {
        reason: {phase: 0 for phase in barrier.PHYSICAL_FALL_PHASES}
        for reason in barrier.PHYSICAL_FALL_REASONS
    }
    table = {phase: 0 for phase in barrier.PHYSICAL_FALL_PHASES}
    if tilt_phase is not None:
        fall["base_fell_tilt"][tilt_phase] = 1
    if too_low_phase is not None:
        fall["base_too_low"][too_low_phase] = 1
    if table_phase is not None:
        table[table_phase] = 1
    # Deliberately spell out the actual launcher producer row.  This fixture
    # must not derive its key set from the barrier under test.
    return {
        "observed_ppo_updates": 5,
        "actual_hard_edge_event_count": 0,
        "actual_hard_terminal_count": 0,
        "joint_qdes_forbidden_terminal_count": 0,
        "joint_actual_forbidden_terminal_count": 0,
        "strict_hard_termination_count": 0,
        "table_contact_count": sum(table.values()),
        "nonfinite_count": 0,
        "base_fell_tilt_terminal_count": sum(fall["base_fell_tilt"].values()),
        "base_too_low_terminal_count": sum(fall["base_too_low"].values()),
        "physical_fall_by_reason_phase": fall,
        "table_contact_by_phase": table,
        "task_wait_started_by_update": [12] * 5,
        "task_wait_started_count": 60,
        "task_reveal_reached_by_update": [10] * 5,
        "task_reveal_reached_count": 50,
    }


def _producer_safety_literal_keys(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_audit_scale4096_terminal"
    )
    matches = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "safety"
            and isinstance(node.value, ast.Dict)
        ):
            matches.append(node.value)
    assert len(matches) == 1
    keys = matches[0].keys
    assert all(isinstance(key, ast.Constant) and type(key.value) is str for key in keys)
    return tuple(key.value for key in keys)


def _producer_prelong_binding(safety):
    behavioral = {}
    for reason in (*barrier.PHYSICAL_FALL_REASONS, "robot_hit_table"):
        total = (
            safety["table_contact_count"]
            if reason == "robot_hit_table"
            else safety[f"{reason}_terminal_count"]
        )
        phases = (
            safety["table_contact_by_phase"]
            if reason == "robot_hit_table"
            else safety["physical_fall_by_reason_phase"][reason]
        )
        behavioral[reason] = {
            "total_count": total,
            "by_phase": copy.deepcopy(phases),
            "phase_exposure_denominators": {
                "hidden_wait": safety["task_wait_started_count"],
                "revealed_pre_strike": safety["task_reveal_reached_count"],
                "post_strike": 40,
            },
            "phase_rates": {phase: 0.0 for phase in barrier.PHYSICAL_FALL_PHASES},
            "acceptance_threshold": None,
        }
    return {
        "gate": {
            "status": "PASS",
            "ppo_updates": 5,
            "authorization": "pre_long_terminal_telemetry_only",
            "safety": {
                "strict_zero_counters": {
                    key: safety[key] for key in barrier.STRICT_ZERO_SAFETY_KEYS
                },
                "task_wait_started_by_update": safety[
                    "task_wait_started_by_update"
                ],
                "task_wait_started_count": safety["task_wait_started_count"],
                "task_reveal_reached_by_update": safety[
                    "task_reveal_reached_by_update"
                ],
                "task_reveal_reached_count": safety[
                    "task_reveal_reached_count"
                ],
                "table_contact_count": safety["table_contact_count"],
                "table_contact_by_phase": safety["table_contact_by_phase"],
                "unknown_attribution_count": 0,
            },
            "survival_denominators": {
                "behavioral_terminations": behavioral,
            },
        }
    }


def _shared():
    return {
        "source_commit_sha": "a" * 40,
        "four_grid_manifest_content_sha256": barrier._F.CONTENT_SHA256,
        "motion_sha256": barrier._F.CANONICAL_MOTION_SHA256,
        "dynamic_ready_artifact_file_sha256": "2" * 64,
        "dynamic_ready_artifact_content_sha256": "3" * 64,
        "dynamic_ready_nominal_receipt_file_sha256": "4" * 64,
        "dynamic_ready_nominal_receipt_content_sha256": "5" * 64,
        "teacher_frame0_artifact_file_sha256": "6" * 64,
        "teacher_frame0_artifact_content_sha256": "7" * 64,
        "split_ready_reset_wait_family_invariant_sha256": (
            barrier.family_invariant_reset_wait_sha256(_reset_wait("A"))
        ),
    }


# 现役 split_ready_reset_wait_authority 的真实形状(从 pod 上 A0/C0 的 launch_claim.json
# 抄下来的字段集合与族标签取值)。两族只在 family / timing_mode / receipt.path 上不同,
# 以及由它们派生的两颗 claim_sha256。
def _reset_wait(family: str):
    assert family in ("A", "C")
    timing_mode = "a_online_solver" if family == "A" else "c_direct_ball"
    recipe = "current_lm" if family == "A" else "outcome_dense_only"
    timing = {
        "action_manifest": {
            "path": "configs/core/take_061_unit04_bh.full.manifest.v3.json",
            "sha256": "a" * 64,
        },
        "claim_sha256": ("b" if family == "A" else "c") * 64,
        "family": family,
        "literal_center_question_sha256": "d" * 64,
        "physics_profile_sha256": "e" * 64,
        "receipt": {
            "content_sha256": "f" * 64,
            "path": "configs/tape/%s.target.task_receipt.v5.json" % recipe,
            "sha256": "0" * 63 + "1",
        },
        "sample_receipt_sha256": "0" * 63 + "2",
        "sampling_profile_sha256": "0" * 63 + "3",
        "solver_profile_sha256": "0" * 63 + "4",
        "timing_mode": timing_mode,
    }
    return {
        "bridge_learning_signal": "dense_mimic_after_task_reveal",
        "claim_sha256": ("8" if family == "A" else "9") * 64,
        "control_decimation": 4,
        "diagnostic_unauthorized": True,
        "dynamic_ready": {
            "content_sha256": "3" * 64,
            "sha256": "2" * 64,
        },
        "hidden_wait_required_physics_steps": 100,
        "hidden_wait_required_policy_steps": 25,
        "initial_center_timing_authority": timing,
        "kind": "action_ball_a211_split_ready_reset_wait_gate_v1",
        "nominal_hold_receipt": {
            "content_sha256": "5" * 64,
            "sha256": "4" * 64,
        },
        "observed_physics_steps": 240,
        "observed_policy_steps": 60,
        "passive_hold_after_reveal_required": False,
        "physical_reset_source": "dynamic_ready.physical_ready",
        "policy_dt_s": 0.02,
        "schema_version": 1,
        "teacher_physical_birth_separated": True,
        "teacher_source": "measured_motion.frame0",
        "time_to_teacher_start_at_reveal_s": 0.6923799138976297,
    }


def _lineage_for_family(family: str):
    shared = {
        "motion_sha256": barrier._F.CANONICAL_MOTION_SHA256,
        "dynamic_ready_artifact_file_sha256": "2" * 64,
        "dynamic_ready_nominal_receipt_file_sha256": "4" * 64,
        "teacher_frame0_artifact_file_sha256": "6" * 64,
        "teacher_frame0_artifact_content_sha256": "7" * 64,
    }
    return {
        "motion": {"path": "motion.npz", "sha256": shared["motion_sha256"]},
        "dynamic_ready_artifact": {
            "path": "ready.json",
            "sha256": shared["dynamic_ready_artifact_file_sha256"],
        },
        "dynamic_ready_nominal_receipt": {
            "path": "hold.json",
            "sha256": shared["dynamic_ready_nominal_receipt_file_sha256"],
        },
        "teacher_frame0_artifact": {
            "path": "teacher.json",
            "sha256": shared["teacher_frame0_artifact_file_sha256"],
        },
        "teacher_frame0_artifact_content_sha256": shared[
            "teacher_frame0_artifact_content_sha256"
        ],
        "split_ready_reset_wait_authority": _reset_wait(family),
    }


def _payload_for_shared(expected):
    return {
        "spec": {"source": {"commit_sha": expected["source_commit_sha"]}},
        "bundle": {
            "isaac_four_grid_manifest": {
                "content_sha256": expected["four_grid_manifest_content_sha256"]
            }
        },
    }


def _audits(tmp_path: Path):
    output = {}
    for index, cell_id in enumerate(barrier._F.CELL_IDS):
        digit = "%x" % (index + 7)
        output[cell_id] = {
            "cell_id": cell_id,
            "task_family": "A211" if cell_id.startswith("A") else "C211",
            "scale_result": {
                "path": str(tmp_path / ("scale-%d.json" % index)),
                "sha256": digit * 64,
            },
            "launch_claim_sha256": ("%x" % (index + 11)) * 64,
            "launch_result_content_sha256": ("%x" % (index + 3)) * 64,
            "gpu": {
                "index": 0 if cell_id.startswith("A") else 1,
                "uuid": (
                    "GPU-A0000000"
                    if cell_id.startswith("A")
                    else "GPU-C1111111"
                ),
                "colocation_opt_in": True,
                "rate_evidence_eligible": False,
            },
            "lineage_sha256": ("%x" % (index + 4)) * 64,
            "terminal_acceptance_content_sha256": ("%x" % (index + 5))
            * 64,
            # 跑满 5 个 update 之后落盘的末位是 model_4.pt / iter=4(RSL-RL 的
            # 迭代变量在循环体内取 0..N-1)。这里和上面的 safety 一样刻意写死字面量,
            # 不从被测模块反推;这个 4 由 test_action_ball_4096x5_terminal_index.py
            # 直接读 RSL-RL 活源码钉住。
            "terminal_model": {
                "sha256": ("%x" % (index + 6)) * 64,
                "filename_iteration": 4,
                "embedded_iteration": 4,
                "all_tensors_finite": True,
            },
            "safety_counters": _producer_safety(),
            "safety_reward_economy": dict(
                barrier.EXPECTED_SAFETY_REWARD_ECONOMY
            ),
            "prelong_gate": {
                "status": "PASS",
                "content_sha256": ("%x" % (index + 12)) * 64,
            },
            "shared_binding": _shared(),
        }
    return output


def test_barrier_safety_schema_is_the_exact_current_a_and_c_producer_shape():
    expected = barrier.PRODUCER_SAFETY_KEYS
    assert _producer_safety_literal_keys(A_LAUNCHER) == expected
    assert _producer_safety_literal_keys(C_LAUNCHER) == expected
    assert tuple(_producer_safety()) == expected
    assert "hard_termination_count" not in expected
    assert "strict_hard_termination_count" in expected


def test_nonzero_behavioral_terminations_are_attributed_not_circularly_rejected(
    tmp_path: Path,
):
    audits = _audits(tmp_path)
    first = barrier._F.CELL_IDS[0]
    safety = _producer_safety(
        tilt_phase="hidden_wait",
        too_low_phase="revealed_pre_strike",
        table_phase="post_strike",
    )
    audits[first]["safety_counters"] = safety
    document = barrier.document_from_audits(audits)
    observed = document["cells"][0]["safety_counters"]
    assert observed["base_fell_tilt_terminal_count"] == 1
    assert observed["base_too_low_terminal_count"] == 1
    assert observed["table_contact_count"] == 1
    assert observed["strict_hard_termination_count"] == 0
    assert document["terminal_acceptance_policy"]["behavioral_terminations"][
        "finite_scale_cutoff"
    ] is None
    barrier._validate_prelong_behavioral_binding(
        _producer_prelong_binding(safety), safety=safety
    )


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda row: row.__setitem__("hard_termination_count", 0),
            "keys differ",
        ),
        (
            lambda row: row.__setitem__("strict_hard_termination_count", 1),
            "implementation safety counter",
        ),
        (
            lambda row: row["table_contact_by_phase"].__setitem__(
                "post_strike", 1
            ),
            "do not conserve",
        ),
        (
            lambda row: row["task_reveal_reached_by_update"].__setitem__(2, 0),
            "nonzero in every update",
        ),
        (
            lambda row: row.__setitem__("task_wait_started_count", 59),
            "do not conserve",
        ),
    ),
)
def test_terminal_safety_rejects_schema_strict_or_conservation_drift(
    mutation, match
):
    safety = _producer_safety()
    mutation(safety)
    with pytest.raises(barrier.BarrierRefused, match=match):
        barrier._validate_terminal_safety(safety)


def test_prelong_binding_must_preserve_nonzero_behavioral_counts_and_none_cutoff():
    safety = _producer_safety(table_phase="revealed_pre_strike")
    binding = _producer_prelong_binding(safety)
    barrier._validate_prelong_behavioral_binding(binding, safety=safety)

    binding["gate"]["survival_denominators"]["behavioral_terminations"][
        "robot_hit_table"
    ]["acceptance_threshold"] = 0
    with pytest.raises(barrier.BarrierRefused, match="attribution/cutoff"):
        barrier._validate_prelong_behavioral_binding(binding, safety=safety)


def test_shared_binding_uses_split_ready_and_runtime_sampler_manifest_only():
    expected = _shared()
    lineage = _lineage_for_family("A")
    payload = _payload_for_shared(expected)
    observed = barrier._shared_binding(lineage, payload)
    assert set(observed) == set(expected)
    assert (
        observed["split_ready_reset_wait_family_invariant_sha256"]
        == barrier.family_invariant_reset_wait_sha256(
            lineage["split_ready_reset_wait_authority"]
        )
    )
    for key in expected:
        if key == "split_ready_reset_wait_family_invariant_sha256":
            continue
        assert observed[key] == expected[key]
    assert "immutable_tape" not in repr(expected)
    assert "exact_zero_handoff" not in repr(expected)
    # 这项现在是本模块现算的抹族 digest,不再是 lineage 里那颗 claim_sha256。
    assert (
        observed["split_ready_reset_wait_family_invariant_sha256"]
        != lineage["split_ready_reset_wait_authority"]["claim_sha256"]
    )


# --------------------------------------------------------------------------- #
# 抹族 digest:该拦的仍拦、误拦的不再拦
# --------------------------------------------------------------------------- #
def test_family_labels_alone_no_longer_split_the_shared_reset_wait_binding():
    """A 格与 C 格只差族标签时,同一份物理复位/WAIT 权威必须比得相等。

    这是修复前**永远过不去**的那一条:两族的 claim_sha256 因为嵌着 family/timing_mode/
    receipt 文件名而在任何正确网格上都不相等。
    """

    a = _reset_wait("A")
    c = _reset_wait("C")
    assert a["claim_sha256"] != c["claim_sha256"]
    assert a != c
    assert barrier.family_invariant_reset_wait_sha256(
        a
    ) == barrier.family_invariant_reset_wait_sha256(c)

    a_binding = barrier._shared_binding(
        _lineage_for_family("A"), _payload_for_shared(_shared())
    )
    c_binding = barrier._shared_binding(
        _lineage_for_family("C"), _payload_for_shared(_shared())
    )
    assert a_binding == c_binding


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda row: row.update(hidden_wait_required_policy_steps=26),
            id="hidden_wait_policy_steps",
        ),
        pytest.param(
            lambda row: row.update(hidden_wait_required_physics_steps=104),
            id="hidden_wait_physics_steps",
        ),
        pytest.param(lambda row: row.update(control_decimation=5), id="decimation"),
        pytest.param(lambda row: row.update(policy_dt_s=0.05), id="policy_dt"),
        pytest.param(
            lambda row: row.update(time_to_teacher_start_at_reveal_s=0.5),
            id="reveal_countdown",
        ),
        pytest.param(
            lambda row: row.update(passive_hold_after_reveal_required=True),
            id="passive_hold",
        ),
        pytest.param(
            lambda row: row.update(physical_reset_source="frame0"),
            id="physical_reset_source",
        ),
        pytest.param(
            lambda row: row["dynamic_ready"].update(content_sha256="9" * 64),
            id="dynamic_ready_content",
        ),
        pytest.param(
            lambda row: row["nominal_hold_receipt"].update(sha256="9" * 64),
            id="nominal_hold_file",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"]["action_manifest"].update(
                sha256="9" * 64
            ),
            id="action_manifest_sha",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"]["action_manifest"].update(
                path="configs/stale_core/manifest.json"
            ),
            id="action_manifest_path",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"].update(
                solver_profile_sha256="9" * 64
            ),
            id="solver_profile",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"].update(
                physics_profile_sha256="9" * 64
            ),
            id="physics_profile",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"].update(
                sampling_profile_sha256="9" * 64
            ),
            id="sampling_profile",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"].update(
                sample_receipt_sha256="9" * 64
            ),
            id="sample_receipt",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"].update(
                literal_center_question_sha256="9" * 64
            ),
            id="literal_center_question",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"]["receipt"].update(
                sha256="9" * 64
            ),
            id="task_receipt_file",
        ),
        pytest.param(
            lambda row: row["initial_center_timing_authority"]["receipt"].update(
                content_sha256="9" * 64
            ),
            id="task_receipt_content",
        ),
    ],
)
def test_reset_wait_digest_still_separates_every_non_family_difference(mutate):
    """粗一档就过不了:族标签之外的任何一处不同,抹族 digest 仍然必须分开。"""

    baseline = _reset_wait("A")
    mutated = _reset_wait("C")
    mutate(mutated)
    assert barrier.family_invariant_reset_wait_sha256(
        baseline
    ) != barrier.family_invariant_reset_wait_sha256(mutated)


def test_reset_wait_digest_refuses_unknown_or_missing_fields():
    extra = _reset_wait("A")
    extra["a_brand_new_switch"] = True
    with pytest.raises(barrier.BarrierRefused):
        barrier.family_invariant_reset_wait_sha256(extra)

    missing = _reset_wait("A")
    missing.pop("control_decimation")
    with pytest.raises(barrier.BarrierRefused):
        barrier.family_invariant_reset_wait_sha256(missing)

    timing_extra = _reset_wait("A")
    timing_extra["initial_center_timing_authority"]["new_pin_sha256"] = "9" * 64
    with pytest.raises(barrier.BarrierRefused):
        barrier.family_invariant_reset_wait_sha256(timing_extra)

    receipt_extra = _reset_wait("A")
    receipt_extra["initial_center_timing_authority"]["receipt"]["extra"] = 1
    with pytest.raises(barrier.BarrierRefused):
        barrier.family_invariant_reset_wait_sha256(receipt_extra)


def test_shared_binding_scope_self_reports_what_was_excluded_and_why():
    scope = barrier.SHARED_BINDING_SCOPE
    assert scope["compared_byte_equal_across_all_four_cells"] == list(
        barrier.SHARED_BINDING_KEYS
    )
    projection = scope["reset_wait_projection"]
    assert projection["digest_key"] in barrier.SHARED_BINDING_KEYS
    assert projection["family_scoped_fields_excluded"] == list(
        barrier.RESET_WAIT_FAMILY_SCOPED_FIELDS
    )
    assert projection["exclusion_reason"].strip()
    # 自陈的"仍然要求逐字节相同"清单必须真的还在 digest 里。
    for entry in projection["still_required_byte_equal_inside_the_digest"]:
        assert entry not in barrier.RESET_WAIT_FAMILY_SCOPED_FIELDS
    assert "split_ready_reset_wait_claim_sha256" not in barrier.SHARED_BINDING_KEYS


def _write(path: Path, value) -> str:
    raw = barrier.canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _runtime_reward_reaudit_fixture(family: str):
    pin = {"path": "/tmp/effective-reward.json", "sha256": "a" * 64}
    runtime_sha = "b" * 64
    term_count = 19
    observed = dict(barrier.EXPECTED_RUNTIME_SAFETY_WEIGHTS)

    class Refused(RuntimeError):
        pass

    old = SimpleNamespace(
        LaunchRefused=Refused,
        _validate_reward_materialization=lambda value: {
            "artifact": dict(value),
            "effective_reward_recipe_sha256": runtime_sha,
            "term_count": term_count,
        },
    )
    module = SimpleNamespace(
        LaunchRefused=Refused,
        _OLD=old,
        _canonical_external_json=lambda value, *, name: (
            dict(value),
            {"terms": [{"name": "fixture"}]},
        ),
        _require_effective_learnability_terms=lambda terms: None,
        _require_c211_outcome_terms=lambda terms: None,
        _runtime_effective_soft_weights=lambda terms, *, arm: dict(observed),
        _runtime_soft_weights=lambda terms, *, recipe: dict(observed),
    )
    selector = {"soft_weights": dict(barrier.EXPECTED_SAFETY_REWARD_ECONOMY)}
    materialization = {
        "runtime_effective_reward_artifact": pin,
        "runtime_effective_reward_sha256": runtime_sha,
        "runtime_effective_reward_term_count": term_count,
        "runtime_soft_weights": dict(observed),
    }
    return module, selector, materialization, observed


def test_all_four_pass_builds_one_exact_long_only_receipt(tmp_path: Path):
    document = barrier.document_from_audits(_audits(tmp_path))
    assert document["status"] == "PASS"
    assert document["authorization"] == "long4096_launch_barrier_only"
    assert document["scale_budget"] == [4096, 5, 1]
    assert document["schema_version"] == 5
    assert document["kind"].endswith("_v5")
    assert document["terminal_acceptance_policy"] == (
        barrier.TERMINAL_ACCEPTANCE_POLICY
    )
    # 2026-08-05 层级对齐(exp §5.6 第 7 条):death -300.0 -> -10.0。
    assert document["safety_reward_economy"] == {
        "death_penalty": -10.0,
        "qdes_limit": -5.0,
        "qdes_projection": -5.0,
        "joint_limit": -5.0,
    }
    assert [row["cell_id"] for row in document["cells"]] == list(
        barrier._F.CELL_IDS
    )
    assert document["authorized_layout"]["gpu0"] == list(
        barrier._F.FAMILY_CELL_IDS["A211"]
    )
    assert document["authorized_layout"]["gpu1"] == list(
        barrier._F.FAMILY_CELL_IDS["C211"]
    )
    assert document["transition_invariant"] == barrier.TRANSITION_INVARIANT
    assert document["transition_invariant"][
        "cross_checkout_or_legacy_writer_atomicity_claimed"
    ] is False
    unsigned = dict(document)
    seal = unsigned.pop("content_sha256")
    assert seal == barrier.canonical_sha256(unsigned)


@pytest.mark.parametrize("family", ("A211", "C211"))
def test_runtime_reward_reaudit_reopens_and_matches_all_four_safety_prices(family):
    module, selector, materialization, _observed = (
        _runtime_reward_reaudit_fixture(family)
    )
    assert barrier._reaudit_runtime_safety_reward_economy(
        module,
        family=family,
        selector=selector,
        materialization=materialization,
    ) == barrier.EXPECTED_SAFETY_REWARD_ECONOMY


@pytest.mark.parametrize("family", ("A211", "C211"))
@pytest.mark.parametrize("weight_name", tuple(barrier.EXPECTED_RUNTIME_SAFETY_WEIGHTS))
def test_runtime_reward_reaudit_rejects_each_resealed_safety_price_drift(
    family, weight_name
):
    module, selector, materialization, observed = (
        _runtime_reward_reaudit_fixture(family)
    )
    observed[weight_name] += 1.0
    with pytest.raises(barrier.BarrierRefused, match="economy differs"):
        barrier._reaudit_runtime_safety_reward_economy(
            module,
            family=family,
            selector=selector,
            materialization=materialization,
        )


@pytest.mark.parametrize(
    "field,bad",
    (
        ("runtime_effective_reward_sha256", "c" * 64),
        ("runtime_effective_reward_term_count", 20),
    ),
)
def test_runtime_reward_reaudit_rejects_artifact_semantic_sha_or_count_drift(
    field, bad
):
    module, selector, materialization, _observed = (
        _runtime_reward_reaudit_fixture("A211")
    )
    materialization[field] = bad
    with pytest.raises(barrier.BarrierRefused, match="economy differs"):
        barrier._reaudit_runtime_safety_reward_economy(
            module,
            family="A211",
            selector=selector,
            materialization=materialization,
        )


def test_missing_cell_is_refused(tmp_path: Path):
    audits = _audits(tmp_path)
    audits.pop(barrier._F.CELL_IDS[-1])
    with pytest.raises(barrier.BarrierRefused, match="all four"):
        barrier.document_from_audits(audits)


def test_duplicate_result_or_claim_is_refused(tmp_path: Path):
    audits = _audits(tmp_path)
    first, second = barrier._F.CELL_IDS[:2]
    audits[second]["scale_result"] = copy.deepcopy(audits[first]["scale_result"])
    with pytest.raises(barrier.BarrierRefused, match="result is duplicated"):
        barrier.document_from_audits(audits)

    audits = _audits(tmp_path)
    audits[second]["launch_claim_sha256"] = audits[first][
        "launch_claim_sha256"
    ]
    with pytest.raises(barrier.BarrierRefused, match="claim is duplicated"):
        barrier.document_from_audits(audits)


def test_shared_source_or_ready_sha_mismatch_is_refused(tmp_path: Path):
    audits = _audits(tmp_path)
    audits[barrier._F.CELL_IDS[-1]]["shared_binding"][
        "dynamic_ready_artifact_file_sha256"
    ] = "f" * 64
    with pytest.raises(barrier.BarrierRefused, match="shared concrete"):
        barrier.document_from_audits(audits)

    audits = _audits(tmp_path)
    audits[barrier._F.CELL_IDS[-1]]["shared_binding"][
        "split_ready_reset_wait_family_invariant_sha256"
    ] = "e" * 64
    with pytest.raises(barrier.BarrierRefused, match="shared concrete"):
        barrier.document_from_audits(audits)

    audits = _audits(tmp_path)
    audits[barrier._F.CELL_IDS[0]]["shared_binding"][
        "four_grid_manifest_content_sha256"
    ] = "f" * 64
    with pytest.raises(barrier.BarrierRefused, match="sampler-manifest/motion"):
        barrier.document_from_audits(audits)


@pytest.mark.parametrize(
    "field,mutate",
    (
        (
            "terminal_model",
            lambda row: row["terminal_model"].__setitem__(
                "all_tensors_finite", False
            ),
        ),
        # 差一格的旧手抄:声称末位是 model_5/iter=5 的收据必须被拒。
        (
            "terminal_model_off_by_one",
            lambda row: row["terminal_model"].update(
                {"filename_iteration": 5, "embedded_iteration": 5}
            ),
        ),
        (
            "terminal_model_filename_only",
            lambda row: row["terminal_model"].__setitem__(
                "filename_iteration", 5
            ),
        ),
        ("safety", lambda row: row["safety_counters"].__setitem__("nonfinite_count", 1)),
        (
            "safety_price",
            lambda row: row["safety_reward_economy"].__setitem__(
                "death_penalty", -30.0
            ),
        ),
        ("gate", lambda row: row["prelong_gate"].__setitem__("status", "FAIL")),
    ),
)
def test_nonpass_terminal_evidence_is_refused(tmp_path: Path, field, mutate):
    audits = _audits(tmp_path)
    mutate(audits[barrier._F.CELL_IDS[0]])
    with pytest.raises(barrier.BarrierRefused):
        barrier.document_from_audits(audits)


def test_receipt_validator_reaudits_all_four_and_rebuilds_exact_document(
    tmp_path: Path,
):
    audits = _audits(tmp_path)
    document = barrier.document_from_audits(audits)
    path = tmp_path / "aggregate.json"
    pin = {"path": str(path), "sha256": _write(path, document)}
    calls = []

    def audit(cell_id, result_pin, *, checkout, modules):
        calls.append((cell_id, result_pin, checkout, modules))
        return copy.deepcopy(audits[cell_id])

    validated = barrier.validate_receipt(
        pin,
        checkout=tmp_path,
        modules={},
        audit_cell=audit,
    )
    assert [row[0] for row in calls] == list(barrier._F.CELL_IDS)
    assert validated["content_sha256"] == document["content_sha256"]


def test_receipt_validator_rejects_resealed_self_report_drift(tmp_path: Path):
    audits = _audits(tmp_path)
    document = barrier.document_from_audits(audits)
    document["cells"][0]["prelong_gate"]["status"] = "FAIL"
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    document["content_sha256"] = barrier.canonical_sha256(unsigned)
    path = tmp_path / "aggregate.json"
    pin = {"path": str(path), "sha256": _write(path, document)}

    def audit(cell_id, result_pin, *, checkout, modules):
        return copy.deepcopy(audits[cell_id])

    with pytest.raises(barrier.BarrierRefused, match="live re-audit"):
        barrier.validate_receipt(
            pin,
            checkout=tmp_path,
            modules={},
            audit_cell=audit,
        )


# =========================================================================== #
# 真的经过 _audit_cell 的一组
#
# 人话:上面所有用例都从 ``_audits()`` 直接造"已经审计完的行",也就是从
# ``_audit_cell`` 的**出口**那一侧开始造数据,于是四格聚合真正的入口函数
# ``_audit_cell`` 一行都没被跑过。上一轮那句手抄的 ``filename_iteration != 5``
# 正是这样活下来的:它写在 ``_audit_cell`` 里,而没有任何一条测试经过 ``_audit_cell``。
#
# 下面这一组一律走 ``barrier._audit_cell`` / ``build_receipt_document`` 的真实路径:
# 真的落一份 scale4096 结果文件、真的落一份 ``launch_claim.json``、真的让
# ``_audit_cell`` 自己去读、去比、去重算摘要。
#
# 两族发射器用替身 —— ``modules=`` 本来就是现役注入点(``build_receipt_document`` /
# ``validate_receipt`` 都把它透传给 ``_audit_cell``)。替身只做**发射器该做的解析**
# (pin↔文件绑定、canonical JSON、把 payload 拆成 spec/lineage/selector),
# **不做任何 ``_audit_cell`` 自己的判断**;所以下面每一条拒收都必须是 ``_audit_cell``
# 出的,不是替身出的。替身的属性面由
# ``test_the_stand_in_launchers_ask_the_real_launchers_for_exactly_these_names``
# 与真发射器逐项对齐,防止替身漂成一份好用但虚构的接口。
# =========================================================================== #

# 两个发射器的真实常量,刻意手写字面量(不从被测模块反推)。
# 与真发射器的等值由下面的 surface 测试当场核对。
CLAIM_SCHEMA_VERSION = 2
CLAIM_KINDS = {
    "A211": "action_ball_a211_four_arm_diagnostic_claim_v2",
    "C211": "action_ball_c211_diagnostic_claim_v2",
}
RESULT_KINDS = {
    "A211": "action_ball_a211_four_arm_diagnostic_launch_result_v1",
    "C211": "action_ball_c211_diagnostic_launch_result_v1",
}
COLOCATION_SPEC_KEY = "allow_vendor_v2_colocation"
# 跑满 5 个 update 之后落盘的末位是 model_4.pt / iter=4。同 _audits(),这里刻意写死
# 字面量;这个 4 由 test_action_ball_4096x5_terminal_index.py 直接读 RSL-RL 活源码钉住。
TERMINAL_ITERATION_LITERAL = 4
RUNTIME_REWARD_SHA = "b" * 64
RUNTIME_REWARD_TERM_COUNT = 19

# ``_audit_cell`` 对两族分别取用的名字。真发射器必须**各有各的、且没有对方的**——
# 一旦某一边把两个都实现了,族分派就不再是可判定的。
AUDITOR_A_ONLY_MODULE_ATTRS = frozenset(
    {
        "_validate_predecessor_result",
        "_require_effective_learnability_terms",
        "_runtime_effective_soft_weights",
    }
)
AUDITOR_C_ONLY_MODULE_ATTRS = frozenset(
    {
        "_validate_scale_predecessor",
        "_require_c211_outcome_terms",
        "_runtime_soft_weights",
    }
)


class _LauncherRefused(RuntimeError):
    """替身发射器的 LaunchRefused。

    现役的两个 ``LaunchRefused`` 与 ``BarrierRefused`` 之间没有任何继承关系,
    替身照此办理 —— 这样"这条拒收是谁出的"在测试里是可判定的。
    """


def _fake_exact_dict(value, keys, *, name):
    if type(value) is not dict or set(value) != set(keys):
        raise _LauncherRefused("%s keys differ" % name)
    return dict(value)


def _fake_absolute_path(value, *, name, must_exist=False):
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\n" in value
        or not os.path.isabs(value)
        or os.path.normpath(value) != value
    ):
        raise _LauncherRefused("%s must be a normalized absolute path" % name)
    path = Path(value)
    if must_exist and not path.exists():
        raise _LauncherRefused("%s does not exist" % name)
    return path


def _fake_base_helpers():
    return SimpleNamespace(
        _absolute_path=_fake_absolute_path,
        _strict_json_bytes=lambda raw, *, name: json.loads(raw.decode("utf-8")),
        _canonical_bytes=barrier.canonical_bytes,
    )


def _fake_family_module(family, *, calls):
    """一份只会解析、不会判断的替身发射器。

    刻意**只**装本族那三个名字(A 装 ``_validate_predecessor_result`` 一族,
    C 装 ``_validate_scale_predecessor`` 一族)。分派错了就是 AttributeError,
    不会被"两边都有"糊过去。
    """

    observed = dict(barrier.EXPECTED_RUNTIME_SAFETY_WEIGHTS)

    def _validated_stage_result(value, *, expected_stage, name):
        pin = _fake_exact_dict(value, ("path", "sha256"), name=name)
        raw = Path(pin["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise _LauncherRefused("%s file digest differs" % name)
        row = json.loads(raw.decode("utf-8"))
        if row.get("stage") != expected_stage:
            raise _LauncherRefused("%s stage differs" % name)
        return pin, row

    def _revalidate_claim_payload(payload, *, claimed=True):
        if claimed is not True:
            raise _LauncherRefused("the barrier must revalidate a claimed payload")
        return (
            copy.deepcopy(payload["spec"]),
            copy.deepcopy(payload["lineage"]),
            copy.deepcopy(payload["selector"]),
        )

    def _record(name):
        def _call(value, **kwargs):
            calls.append((family, name, value, kwargs))
            return {"recorded": name}

        return _call

    old = SimpleNamespace(
        LaunchRefused=_LauncherRefused,
        _validate_reward_materialization=lambda pin: {
            "artifact": dict(pin),
            "effective_reward_recipe_sha256": RUNTIME_REWARD_SHA,
            "term_count": RUNTIME_REWARD_TERM_COUNT,
        },
    )
    module = SimpleNamespace(
        LaunchRefused=_LauncherRefused,
        SCHEMA_VERSION=CLAIM_SCHEMA_VERSION,
        CLAIM_KIND=CLAIM_KINDS[family],
        RESULT_KIND=RESULT_KINDS[family],
        COLOCATION_SPEC_KEY=COLOCATION_SPEC_KEY,
        # 刻意手写发射器的产出形状,不从 barrier 反推。
        PHYSICAL_FALL_REASONS=("base_fell_tilt", "base_too_low"),
        PHYSICAL_FALL_PHASES=(
            "hidden_wait",
            "revealed_pre_strike",
            "post_strike",
        ),
        # 真发射器加载的 pre-long gate 就是这一份文件;这里给真模块,
        # 好让"barrier 的手抄 STRICT_ZERO 与活 gate 漂开"这件事在测试里现形。
        _P=barrier._P,
        _B=_fake_base_helpers(),
        _OLD=old,
        canonical_sha256=barrier.canonical_sha256,
        _exact_dict=_fake_exact_dict,
        _validated_stage_result=_validated_stage_result,
        _revalidate_claim_payload=_revalidate_claim_payload,
        _canonical_external_json=lambda value, *, name: (
            dict(value),
            {"terms": [{"name": "fixture"}]},
        ),
    )
    if family == "A211":
        module._validate_predecessor_result = _record("_validate_predecessor_result")
        module._require_effective_learnability_terms = lambda terms: None
        module._runtime_effective_soft_weights = lambda terms, *, arm: dict(observed)
    else:
        module._validate_scale_predecessor = _record("_validate_scale_predecessor")
        module._require_c211_outcome_terms = lambda terms: None
        module._runtime_soft_weights = lambda terms, *, recipe: dict(observed)
    return module


def _runtime_materialization(index):
    return {
        "runtime_effective_reward_artifact": {
            "path": "/tmp/effective-reward-%d.json" % index,
            "sha256": ("%x" % (index + 9)) * 64,
        },
        "runtime_effective_reward_sha256": RUNTIME_REWARD_SHA,
        "runtime_effective_reward_term_count": RUNTIME_REWARD_TERM_COUNT,
        "runtime_soft_weights": dict(barrier.EXPECTED_RUNTIME_SAFETY_WEIGHTS),
        "content_sha256": ("%x" % (index + 10)) * 64,
    }


def _build_scale_cell(
    root: Path,
    cell_id: str,
    *,
    index: int,
    checkout: Path,
    edit=None,
    post_seal=None,
    reseal_claim=False,
    claim_edit=None,
    claim_bytes_edit=None,
    sibling_materialization=False,
):
    """落一份真的 scale4096 结果 + launch_claim.json,返回给 _audit_cell 的 pin。"""

    family = (
        "A211" if cell_id in barrier._F.FAMILY_CELL_IDS["A211"] else "C211"
    )
    letter = "A" if family == "A211" else "C"
    cell_root = root / "cells" / cell_id.split("-", 1)[0].lower()
    namespace = cell_root / "ns"
    namespace.mkdir(parents=True, exist_ok=True)
    selector_key = "arm_id" if family == "A211" else "recipe_id"
    materialization_key = (
        "arm_materialization" if family == "A211" else "reward_materialization"
    )
    inputs = {
        materialization_key: _runtime_materialization(index),
        "policy_recipe_materialization": {
            "content_sha256": ("%x" % (index + 2)) * 64
        },
        "oracle32_receipt": {"content_sha256": ("%x" % (index + 3)) * 64},
    }
    if sibling_materialization:
        sibling_key = (
            "reward_materialization"
            if family == "A211"
            else "arm_materialization"
        )
        drifted = _runtime_materialization(index)
        drifted["runtime_soft_weights"] = {
            name: value - 1.0
            for name, value in drifted["runtime_soft_weights"].items()
        }
        inputs[sibling_key] = drifted
    payload = {
        "spec": {
            selector_key: cell_id,
            "stage": "scale4096",
            "namespace": str(namespace),
            "source": {"checkout": str(checkout), "commit_sha": "a" * 40},
            "gpu": {
                "index": 0 if family == "A211" else 1,
                "uuid": "GPU-" + ("a" if family == "A211" else "c") * 8,
                "require_empty": False,
                "lock_path": str(cell_root / "gpu.lock"),
            },
            COLOCATION_SPEC_KEY: True,
        },
        "output_contract": {
            "rate_evidence_eligible": False,
            "speed_benchmark_eligible": False,
            "colocated_stage": "scale4096",
        },
        "materialization_inputs": inputs,
        "bundle": {
            "isaac_four_grid_manifest": {
                "content_sha256": barrier._F.CONTENT_SHA256
            }
        },
        "lineage": dict(
            _lineage_for_family(letter),
            lineage_sha256=("%x" % (index + 4)) * 64,
        ),
        "selector": {
            selector_key: cell_id,
            "soft_weights": dict(barrier.EXPECTED_SAFETY_REWARD_ECONOMY),
        },
    }
    safety = _producer_safety()
    terminal = {
        "content_sha256": ("%x" % (index + 5)) * 64,
        "checkpoint": {
            "sha256": ("%x" % (index + 6)) * 64,
            "filename_iteration": TERMINAL_ITERATION_LITERAL,
            "embedded_iteration": TERMINAL_ITERATION_LITERAL,
            "all_tensors_finite": True,
            "tensor_groups": {"actor": 8, "critic": 8, "std": 1},
        },
        "safety_counters": safety,
        "prelong_gate": dict(
            _producer_prelong_binding(safety),
            content_sha256=("%x" % (index + 12)) * 64,
        ),
    }
    result = {
        "schema_version": 1,
        "kind": RESULT_KINDS[family],
        "diagnostic_unauthorized": True,
        "accepted": True,
        "launch_claim_sha256": None,
        "stage": "scale4096",
        "namespace": str(namespace),
        "completion": {
            "completion_exit_code": "0",
            "terminal_kind": "clean_completion",
            "terminal_exit_code": "0",
        },
        "gpu_admission": {
            "index": payload["spec"]["gpu"]["index"],
            "uuid": payload["spec"]["gpu"]["uuid"],
        },
        "output_contract": copy.deepcopy(payload["output_contract"]),
        materialization_key: copy.deepcopy(inputs[materialization_key]),
        "policy_recipe_materialization": copy.deepcopy(
            inputs["policy_recipe_materialization"]
        ),
        "oracle32_receipt": copy.deepcopy(inputs["oracle32_receipt"]),
        "predecessor_result": None,
        "terminal_acceptance": terminal,
        "content_sha256": None,
    }
    if edit is not None:
        edit(payload, result)
    claim_sha = barrier.canonical_sha256(payload)
    if result["launch_claim_sha256"] is None:
        result["launch_claim_sha256"] = claim_sha
    if post_seal is not None:
        # 结果已经按旧 claim 摘要封好了,现在动 payload:
        # 如果 reseal_claim 为假,claim 文件里的摘要就与 payload 对不上;
        # 为真则 claim 文件自洽,但与结果里记的那颗对不上。
        post_seal(payload)
    unsigned = dict(result)
    unsigned.pop("content_sha256")
    result["content_sha256"] = barrier.canonical_sha256(unsigned)
    result_path = cell_root / "scale4096_result.json"
    pin = {"path": str(result_path), "sha256": _write(result_path, result)}
    outer = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "kind": CLAIM_KINDS[family],
        "launch_claim_sha256": (
            barrier.canonical_sha256(payload) if reseal_claim else claim_sha
        ),
        "canonical_payload": payload,
    }
    if claim_edit is not None:
        claim_edit(outer)
    raw = barrier.canonical_bytes(outer) + b"\n"
    if claim_bytes_edit is not None:
        raw = claim_bytes_edit(raw)
    (namespace / "launch_claim.json").write_bytes(raw)
    return pin


def _grid(root: Path, *, target=None, module_edit=None, **cell_edits):
    """四格全套真实输入;``cell_edits`` 只作用在 ``target`` 那一格上。"""

    root.mkdir(parents=True, exist_ok=True)
    checkout = root / "checkout"
    checkout.mkdir(exist_ok=True)
    calls = []
    modules = {
        family: _fake_family_module(family, calls=calls)
        for family in ("A211", "C211")
    }
    if module_edit is not None:
        module_edit(modules)
    pins = {}
    for index, cell_id in enumerate(barrier._F.CELL_IDS):
        pins[cell_id] = _build_scale_cell(
            root,
            cell_id,
            index=index,
            checkout=checkout,
            **(cell_edits if cell_id == target else {}),
        )
    return SimpleNamespace(
        pins=pins, modules=modules, checkout=checkout, calls=calls
    )


@pytest.mark.parametrize("family", ("A211", "C211"))
def test_audit_cell_accepts_a_real_scale_cell_and_emits_the_row_the_aggregator_wants(
    tmp_path: Path, family
):
    """正例:一格真的走完 _audit_cell,产出的行必须被下游唯一校验点接受。

    这一步从来没有被证明过 —— 以前的用例是把"审计后的行"手写出来喂给
    ``document_from_audits``,所以 ``_audit_cell`` 的出口形状与
    ``_validate_audit_row`` 的入口形状是否对得上,全靠人眼。
    """

    cell_id = barrier._F.FAMILY_CELL_IDS[family][0]
    case = _grid(tmp_path)
    row = barrier._audit_cell(
        cell_id, case.pins[cell_id], checkout=case.checkout, modules=case.modules
    )
    assert row["cell_id"] == cell_id
    assert row["task_family"] == family
    assert row["terminal_model"]["filename_iteration"] == TERMINAL_ITERATION_LITERAL
    assert row["terminal_model"]["embedded_iteration"] == TERMINAL_ITERATION_LITERAL
    assert row["terminal_model"]["all_tensors_finite"] is True
    assert row["gpu"]["index"] == (0 if family == "A211" else 1)
    assert row["gpu"]["colocation_opt_in"] is True
    assert row["gpu"]["rate_evidence_eligible"] is False
    assert row["prelong_gate"]["status"] == "PASS"
    assert row["safety_reward_economy"] == barrier.EXPECTED_SAFETY_REWARD_ECONOMY
    validated = barrier._validate_audit_row(row, expected_cell=cell_id)
    assert validated["cell_id"] == cell_id


def test_the_hand_built_audits_fixture_still_matches_what_the_real_auditor_emits(
    tmp_path: Path,
):
    """``_audits()`` 是 ``_audit_cell`` 出口的手抄副本 —— 这里把它钉回真值。

    副本一旦漂了,上面那一堆"从 _audits() 造行"的用例就会开始给一个不存在的
    形状发合格证。
    """

    case = _grid(tmp_path)
    fixture = _audits(tmp_path)
    for cell_id in barrier._F.CELL_IDS:
        row = barrier._audit_cell(
            cell_id,
            case.pins[cell_id],
            checkout=case.checkout,
            modules=case.modules,
        )
        hand = fixture[cell_id]
        assert set(row) == set(hand)
        for key in ("gpu", "terminal_model", "safety_counters", "prelong_gate"):
            assert set(row[key]) == set(hand[key]), key
        assert set(row["shared_binding"]) == set(hand["shared_binding"])
        assert set(row["scale_result"]) == set(hand["scale_result"])
        assert row["task_family"] == hand["task_family"]
        assert row["gpu"]["index"] == hand["gpu"]["index"]
        assert row["terminal_model"]["filename_iteration"] == (
            hand["terminal_model"]["filename_iteration"]
        )


def test_build_receipt_document_through_the_real_auditor_signs_all_four_cells(
    tmp_path: Path,
):
    case = _grid(tmp_path)
    document = barrier.build_receipt_document(
        case.pins, checkout=case.checkout, modules=case.modules
    )
    assert document["status"] == "PASS"
    assert document["authorization"] == barrier.AUTHORIZATION
    assert [row["cell_id"] for row in document["cells"]] == list(
        barrier._F.CELL_IDS
    )
    for row in document["cells"]:
        assert row["terminal_model"]["filename_iteration"] == (
            TERMINAL_ITERATION_LITERAL
        )
        assert row["terminal_model"]["embedded_iteration"] == (
            TERMINAL_ITERATION_LITERAL
        )
    unsigned = dict(document)
    seal = unsigned.pop("content_sha256")
    assert seal == barrier.canonical_sha256(unsigned)
    # A 格与 C 格只差族标签时,四格必须被判成同一份物理复位/WAIT 权威。
    assert barrier.SHA_RE.fullmatch(
        document["shared_binding"]["split_ready_reset_wait_family_invariant_sha256"]
    )


def test_validate_receipt_reaudits_the_four_cells_through_the_real_auditor(
    tmp_path: Path,
):
    case = _grid(tmp_path)
    document = barrier.build_receipt_document(
        case.pins, checkout=case.checkout, modules=case.modules
    )
    path = tmp_path / "aggregate.json"
    pin = {"path": str(path), "sha256": _write(path, document)}
    validated = barrier.validate_receipt(
        pin, checkout=case.checkout, modules=case.modules
    )
    assert validated["content_sha256"] == document["content_sha256"]

    # 把终局迭代号改回差一格的旧手抄,并重新封印整份文档。
    # 粗一档的检查("摘要自洽就放行")会放它过去;活体复审必须比出来。
    tampered = copy.deepcopy(document)
    tampered["cells"][0]["terminal_model"]["filename_iteration"] = 5
    tampered["cells"][0]["terminal_model"]["embedded_iteration"] = 5
    unsigned = dict(tampered)
    unsigned.pop("content_sha256")
    tampered["content_sha256"] = barrier.canonical_sha256(unsigned)
    other = tmp_path / "aggregate-tampered.json"
    bad_pin = {"path": str(other), "sha256": _write(other, tampered)}
    with pytest.raises(barrier.BarrierRefused, match="live re-audit"):
        barrier.validate_receipt(
            bad_pin, checkout=case.checkout, modules=case.modules
        )


def test_audit_cell_sends_each_family_to_its_own_predecessor_validator(
    tmp_path: Path,
):
    case = _grid(tmp_path)
    for cell_id in barrier._F.CELL_IDS:
        barrier._audit_cell(
            cell_id,
            case.pins[cell_id],
            checkout=case.checkout,
            modules=case.modules,
        )
    assert [(family, name) for family, name, _pin, _kw in case.calls] == [
        ("A211", "_validate_predecessor_result"),
        ("A211", "_validate_predecessor_result"),
        ("C211", "_validate_scale_predecessor"),
        ("C211", "_validate_scale_predecessor"),
    ]
    a_kwargs = case.calls[0][3]
    assert set(a_kwargs) == {
        "checkout",
        "expected_stage",
        "materialization",
        "policy_materialization",
        "oracle32",
    }
    assert a_kwargs["expected_stage"] == "scale4096"
    assert a_kwargs["checkout"] == case.checkout
    c_kwargs = case.calls[2][3]
    assert set(c_kwargs) == {
        "checkout",
        "materialization",
        "policy",
        "oracle32",
    }
    assert c_kwargs["checkout"] == case.checkout


def test_audit_cell_does_not_swallow_the_family_launcher_refusal(tmp_path: Path):
    """发射器自己的拒收必须原样冒出去,不许被 _audit_cell 吞成"这格没问题"。"""

    def _boom(modules):
        def _raise(value, **kwargs):
            raise _LauncherRefused("predecessor chain is broken")

        modules["A211"]._validate_predecessor_result = _raise
        modules["C211"]._validate_scale_predecessor = _raise

    case = _grid(tmp_path, module_edit=_boom)
    for cell_id in barrier._F.CELL_IDS:
        with pytest.raises(_LauncherRefused, match="predecessor chain"):
            barrier._audit_cell(
                cell_id,
                case.pins[cell_id],
                checkout=case.checkout,
                modules=case.modules,
            )


@pytest.mark.parametrize("family", ("A211", "C211"))
def test_audit_cell_reaudits_its_own_family_materialization_key(
    tmp_path: Path, family
):
    """A 读 arm_materialization、C 读 reward_materialization,读错键要能被看见。

    兄弟键这里**存在**且价格已经漂了 —— 读对键必须通过,读错键必须拒收。
    """

    cell_id = barrier._F.FAMILY_CELL_IDS[family][0]
    good = _grid(tmp_path / "good", target=cell_id, sibling_materialization=True)
    row = barrier._audit_cell(
        cell_id, good.pins[cell_id], checkout=good.checkout, modules=good.modules
    )
    assert row["safety_reward_economy"] == barrier.EXPECTED_SAFETY_REWARD_ECONOMY

    own_key = "arm_materialization" if family == "A211" else "reward_materialization"

    def _drift_own(payload, result):
        weights = payload["materialization_inputs"][own_key]["runtime_soft_weights"]
        payload["materialization_inputs"][own_key]["runtime_soft_weights"] = {
            name: value - 1.0 for name, value in weights.items()
        }

    bad = _grid(
        tmp_path / "bad",
        target=cell_id,
        sibling_materialization=True,
        edit=_drift_own,
    )
    with pytest.raises(
        barrier.BarrierRefused, match="runtime safety reward economy differs"
    ):
        barrier._audit_cell(
            cell_id, bad.pins[cell_id], checkout=bad.checkout, modules=bad.modules
        )


def _auditor_module_attributes():
    """AST 抓出 ``_audit_cell`` 一族真正向发射器要的每个名字。"""

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    wanted = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in (
            "_audit_cell",
            "_claim_for_result",
            "_reaudit_runtime_safety_reward_economy",
        ):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "module"
            ):
                wanted.add(inner.attr)
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "getattr"
                and len(inner.args) >= 2
                and isinstance(inner.args[0], ast.Name)
                and inner.args[0].id == "module"
                and isinstance(inner.args[1], ast.Constant)
                and type(inner.args[1].value) is str
            ):
                wanted.add(inner.args[1].value)
    return wanted


def test_the_stand_in_launchers_ask_the_real_launchers_for_exactly_these_names():
    """替身不许是虚构。

    ``_audit_cell`` 用到的每个 ``module.<名字>``,真发射器与替身都必须真的有。
    这道门补的是同一个病的最深一层:``modules=`` 在测试里永远被注入,于是
    ``_family_modules()``(现役唯一加载真发射器的地方)全仓零覆盖 —— 没人验过
    真模块到底给不给得出 ``_audit_cell`` 要的这一面。
    """

    wanted = _auditor_module_attributes()
    assert {
        "_validated_stage_result",
        "_revalidate_claim_payload",
        "COLOCATION_SPEC_KEY",
        "PHYSICAL_FALL_REASONS",
        "PHYSICAL_FALL_PHASES",
        "SCHEMA_VERSION",
        "CLAIM_KIND",
        "canonical_sha256",
        "_exact_dict",
        "_B",
        "_P",
        "_OLD",
    } <= wanted
    assert AUDITOR_A_ONLY_MODULE_ATTRS <= wanted
    assert AUDITOR_C_ONLY_MODULE_ATTRS <= wanted

    real = barrier._family_modules()
    calls = []
    fakes = {
        family: _fake_family_module(family, calls=calls)
        for family in ("A211", "C211")
    }
    for family in ("A211", "C211"):
        other = (
            AUDITOR_C_ONLY_MODULE_ATTRS
            if family == "A211"
            else AUDITOR_A_ONLY_MODULE_ATTRS
        )
        for attr in sorted(wanted - other):
            assert hasattr(real[family], attr), (family, attr)
            assert hasattr(fakes[family], attr), (family, attr)
        # 对方族的名字必须**不在**这一边,否则族分派不再可判定。
        for attr in sorted(other):
            assert not hasattr(real[family], attr), (family, attr)
            assert not hasattr(fakes[family], attr), (family, attr)
        assert real[family].SCHEMA_VERSION == CLAIM_SCHEMA_VERSION
        assert real[family].CLAIM_KIND == CLAIM_KINDS[family]
        assert real[family].RESULT_KIND == RESULT_KINDS[family]
        assert real[family].COLOCATION_SPEC_KEY == COLOCATION_SPEC_KEY
        assert tuple(real[family].PHYSICAL_FALL_REASONS) == (
            fakes[family].PHYSICAL_FALL_REASONS
        )
        assert tuple(real[family].PHYSICAL_FALL_PHASES) == (
            fakes[family].PHYSICAL_FALL_PHASES
        )


def test_the_barrier_strict_zero_tuple_is_the_live_pre_long_gate_tuple():
    """``_audit_cell`` 拿发射器加载的那份 pre-long gate 与自己的手抄常量逐项比。

    两份一旦漂开,四格聚合会**每一格**都拒收("launcher/pre-long safety schema
    differs")。所以这里直接比活值,而不是比两份手抄。
    """

    assert tuple(barrier._P.STRICT_ZERO_SAFETY_COUNTERS) == (
        barrier.STRICT_ZERO_SAFETY_KEYS
    )


def _drop_output_contract(payload, result):
    payload.pop("output_contract")


def _selector_points_at_the_sibling_cell(payload, result):
    """把 selector 换成**同族另一格** —— 同一张卡、同一套设置,只是另一个 DR 档位。"""

    selector = payload["selector"]
    key = "arm_id" if "arm_id" in selector else "recipe_id"
    siblings = barrier._F.FAMILY_CELL_IDS[
        "A211" if key == "arm_id" else "C211"
    ]
    selector[key] = (
        siblings[1] if selector[key] == siblings[0] else siblings[0]
    )


def _selector_keeps_the_tag_but_swaps_the_axis(payload, result):
    """保留 ``A0``/``C1`` 那个前缀标签,把后面的轴描述换成兄弟格的。

    产出的是一个**不存在**的 cell_id。只比前缀标签(``cell_id.split("-")[0]``)
    的那一档检查会放它过去。
    """

    selector = payload["selector"]
    key = "arm_id" if "arm_id" in selector else "recipe_id"
    siblings = barrier._F.FAMILY_CELL_IDS[
        "A211" if key == "arm_id" else "C211"
    ]
    current = selector[key]
    other = siblings[1] if current == siblings[0] else siblings[0]
    selector[key] = current.split("-", 1)[0] + "-" + other.split("-", 1)[1]


def _swap_gpu_to_the_other_family(payload, result):
    payload["spec"]["gpu"]["index"] = 1 - payload["spec"]["gpu"]["index"]


# 每一条都写清"粗一个档次的检查会放行什么"。
AUDIT_CELL_MUTATIONS = (
    # ---------------- launch claim 文件本身 ----------------
    pytest.param(
        {"claim_bytes_edit": lambda raw: raw + b" "},
        "launch claim is not canonical",
        id="claim_bytes_not_canonical",  # 粗档:json.loads 之后相等就放行
    ),
    pytest.param(
        {"claim_edit": lambda outer: outer.update(kind="x" + outer["kind"])},
        "launch claim digest differs",
        id="claim_kind_differs",  # 粗档:kind 里含族名就放行
    ),
    pytest.param(
        {"claim_edit": lambda outer: outer.update(schema_version=1)},
        "launch claim digest differs",
        id="claim_schema_version_differs",  # 粗档:只看 kind 不看版本
    ),
    pytest.param(
        {"post_seal": lambda payload: payload.update(a_new_switch=True)},
        "launch claim digest differs",
        # 粗档:只比 claim 文件里的那颗和结果里的那颗(两颗仍相等),不重算 payload
        id="claim_sha_no_longer_covers_the_payload",
    ),
    pytest.param(
        {
            "post_seal": lambda payload: payload.update(a_new_switch=True),
            "reseal_claim": True,
        },
        "launch claim digest differs",
        # 粗档:claim 文件"自洽"就放行 —— 但它已经不是这份结果的那颗 claim 了
        id="claim_reseals_itself_away_from_the_result",
    ),
    # ---------------- selector / 来源 / GPU / 产出契约 ----------------
    pytest.param(
        {"edit": _selector_points_at_the_sibling_cell},
        "selector/source differs",
        id="selector_is_the_sibling_cell",  # 粗档:同族就放行
    ),
    pytest.param(
        {"edit": _selector_keeps_the_tag_but_swaps_the_axis},
        "selector/source differs",
        id="selector_keeps_the_tag_but_swaps_the_axis",  # 粗档:只比 A0/C1 那个标签
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["selector"][
                "soft_weights"
            ].__setitem__("death_penalty", -300.0)
        },
        "selector/source differs",
        id="selector_carries_the_retired_death_price",  # 粗档:键集相同就放行
    ),
    pytest.param(
        {"edit": lambda payload, result: payload["spec"].update(stage="long4096")},
        "selector/source differs",
        id="claim_stage_disagrees_with_the_result",  # 粗档:只看结果里的 stage
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["spec"]["source"].update(
                checkout=payload["spec"]["source"]["checkout"] + "/"
            )
        },
        "selector/source differs",
        id="checkout_only_normalizes_to_the_same_tree",  # 粗档:normpath 之后相等
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["spec"].update(
                namespace=payload["spec"]["source"]["checkout"]
            )
        },
        "selector/source differs",
        id="claim_namespace_disagrees_with_the_result",  # 粗档:是个存在的绝对目录
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result.update(
                predecessor_result={"path": "/tmp/pred.json", "sha256": "d" * 64}
            )
        },
        "selector/source differs",
        id="scale_result_carries_a_predecessor",  # 粗档:predecessor 形状合法就放行
    ),
    pytest.param(
        {"edit": _swap_gpu_to_the_other_family},
        "selector/source differs",
        id="gpu_index_is_the_other_family_card",  # 粗档:index in (0, 1) 就放行
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["spec"]["gpu"].update(
                require_empty=0
            )
        },
        "selector/source differs",
        id="require_empty_is_falsy_but_not_false",  # 粗档:not gpu["require_empty"]
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["spec"].update(
                {COLOCATION_SPEC_KEY: 1}
            )
        },
        "selector/source differs",
        id="colocation_opt_in_is_truthy_but_not_true",  # 粗档:if spec.get(KEY)
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["output_contract"].update(
                rate_evidence_eligible=0
            )
        },
        "selector/source differs",
        id="rate_evidence_flag_is_falsy_but_not_false",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["output_contract"].update(
                speed_benchmark_eligible=0
            )
        },
        "selector/source differs",
        id="speed_benchmark_flag_is_falsy_but_not_false",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["output_contract"].update(
                colocated_stage="long4096"
            )
        },
        "selector/source differs",
        id="colocated_stage_is_the_long_run",  # 粗档:键存在就放行
    ),
    pytest.param(
        {"edit": _drop_output_contract},
        "selector/source differs",
        id="output_contract_is_missing",
    ),
    # ---------------- 发射器/pre-long 的安全词表 ----------------
    pytest.param(
        {
            "module_edit": lambda modules: [
                setattr(
                    module,
                    "PHYSICAL_FALL_PHASES",
                    tuple(reversed(module.PHYSICAL_FALL_PHASES)),
                )
                for module in modules.values()
            ]
        },
        "safety schema differs",
        id="launcher_phase_order_flipped",  # 粗档:比集合不比顺序
    ),
    pytest.param(
        {
            "module_edit": lambda modules: [
                setattr(
                    module,
                    "PHYSICAL_FALL_REASONS",
                    tuple(reversed(module.PHYSICAL_FALL_REASONS)),
                )
                for module in modules.values()
            ]
        },
        "safety schema differs",
        id="launcher_reason_order_flipped",  # 粗档:比集合不比顺序
    ),
    pytest.param(
        {
            "module_edit": lambda modules: [
                setattr(
                    module,
                    "_P",
                    SimpleNamespace(
                        STRICT_ZERO_SAFETY_COUNTERS=(
                            barrier.STRICT_ZERO_SAFETY_KEYS[1:]
                        )
                    ),
                )
                for module in modules.values()
            ]
        },
        "safety schema differs",
        id="prelong_gate_dropped_one_strict_zero_counter",  # 粗档:子集就放行
    ),
    # ---------------- 终局安全台账 ----------------
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "safety_counters"
            ].update(nonfinite_count=1)
        },
        "implementation safety counter",
        id="nonfinite_counter_is_nonzero",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "safety_counters"
            ]["table_contact_by_phase"].update(post_strike=1)
        },
        "do not conserve",
        id="table_phase_counts_stop_conserving",  # 粗档:总数没变就放行
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "safety_counters"
            ]["task_reveal_reached_by_update"].__setitem__(2, 0)
        },
        "nonzero in every update",
        id="one_update_has_an_empty_reveal_denominator",  # 粗档:只看聚合总数
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "safety_counters"
            ].update(observed_ppo_updates=4)
        },
        "do not cover five updates",
        id="safety_ledger_covers_four_updates",
    ),
    # ---------------- pre-long 门与安全台账的绑定 ----------------
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "prelong_gate"
            ]["gate"].update(status="FAIL")
        },
        "did not PASS",
        id="prelong_gate_says_fail",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "prelong_gate"
            ]["gate"].update(ppo_updates=4)
        },
        "did not PASS",
        id="prelong_gate_counted_four_updates",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "prelong_gate"
            ]["gate"]["safety"].update(
                task_wait_started_by_update=[11, 13, 12, 12, 12]
            )
        },
        "counters differ",
        id="gate_reshuffles_the_wait_denominator_keeping_the_total",  # 粗档:只比总数
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "prelong_gate"
            ]["gate"]["safety"].update(unknown_attribution_count=1)
        },
        "counters differ",
        id="gate_has_unattributed_terminations",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "prelong_gate"
            ]["gate"]["survival_denominators"]["behavioral_terminations"][
                "robot_hit_table"
            ].update(
                acceptance_threshold=0
            )
        },
        "attribution/cutoff",
        id="gate_reintroduces_a_falsy_acceptance_cutoff",  # 粗档:if threshold:
    ),
    # ---------------- 终局 checkpoint ----------------
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "checkpoint"
            ].update(filename_iteration=5, embedded_iteration=5)
        },
        "terminal evidence is incomplete",
        # 粗档:两个迭代号"互相自洽"就放行 —— 这正是上一轮活下来的那个差一格手抄
        id="terminal_checkpoint_is_the_old_off_by_one",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "checkpoint"
            ].update(filename_iteration=5)
        },
        "terminal evidence is incomplete",
        id="terminal_filename_iteration_alone_is_off_by_one",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "checkpoint"
            ].update(embedded_iteration=5)
        },
        "terminal evidence is incomplete",
        id="terminal_embedded_iteration_alone_is_off_by_one",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "checkpoint"
            ].update(filename_iteration="4")
        },
        "terminal evidence is incomplete",
        id="terminal_iteration_is_a_string",  # 粗档:int() 之后相等就放行
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "checkpoint"
            ].update(all_tensors_finite="yes")
        },
        "terminal evidence is incomplete",
        id="all_tensors_finite_is_truthy_but_not_true",  # 粗档:truthiness
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "checkpoint"
            ].update(tensor_groups={})
        },
        "terminal evidence is incomplete",
        id="tensor_groups_is_an_empty_dict",  # 粗档:type is dict 就放行
    ),
    pytest.param(
        {
            "edit": lambda payload, result: result["terminal_acceptance"][
                "checkpoint"
            ].update(tensor_groups=[])
        },
        "terminal evidence is incomplete",
        id="tensor_groups_is_a_list",
    ),
    # ---------------- 四格共享绑定(_audit_cell 现算,不是抄来的) ----------------
    pytest.param(
        {
            "edit": lambda payload, result: payload["bundle"][
                "isaac_four_grid_manifest"
            ].update(content_sha256="f" * 64)
        },
        "sampler-manifest/motion",
        id="claim_binds_a_foreign_four_grid_manifest",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["spec"]["source"].update(
                commit_sha="a" * 39
            )
        },
        "shared source commit is invalid",
        id="source_commit_is_not_forty_hex",
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["lineage"].update(
                teacher_frame0_artifact_content_sha256="0" * 64
            )
        },
        "zero sentinel",
        id="teacher_frame0_content_sha_is_the_zero_sentinel",  # 粗档:形状像 SHA
    ),
    pytest.param(
        {
            "edit": lambda payload, result: payload["lineage"][
                "split_ready_reset_wait_authority"
            ].update(a_brand_new_switch=True)
        },
        "keys differ",
        id="reset_wait_authority_grew_an_unclassified_field",
    ),
)


@pytest.mark.parametrize("family", ("A211", "C211"))
@pytest.mark.parametrize("mutation,match", AUDIT_CELL_MUTATIONS)
def test_audit_cell_refuses_each_face_it_is_supposed_to_refuse(
    tmp_path: Path, family, mutation, match
):
    """每一条拒收面:同一份夹具先当正例过一遍,再变异一处必须被拒。

    变异一律构造成"粗一个档次的检查会放行"(见每条 param 上的注释),
    这样这些用例能证明的是**这一档**的严格度,不是"随便动一下就红"。
    """

    cell_id = barrier._F.FAMILY_CELL_IDS[family][0]
    clean = _grid(tmp_path / "clean")
    accepted = barrier._audit_cell(
        cell_id,
        clean.pins[cell_id],
        checkout=clean.checkout,
        modules=clean.modules,
    )
    assert accepted["cell_id"] == cell_id

    case = _grid(tmp_path / "mutated", target=cell_id, **mutation)
    with pytest.raises(barrier.BarrierRefused, match=match):
        barrier._audit_cell(
            cell_id,
            case.pins[cell_id],
            checkout=case.checkout,
            modules=case.modules,
        )


def test_build_receipt_document_requires_one_pin_per_cell(tmp_path: Path):
    """``build_receipt_document`` 自己那条拒收也从来没被跑过(auditor 一直被注入)。"""

    case = _grid(tmp_path)
    partial = dict(case.pins)
    partial.pop(barrier._F.CELL_IDS[-1])
    with pytest.raises(barrier.BarrierRefused, match="every grid cell"):
        barrier.build_receipt_document(
            partial, checkout=case.checkout, modules=case.modules
        )
