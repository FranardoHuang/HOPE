"""CPU-only tests for the all-four A211/C211 pre-long barrier."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
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
        "split_ready_reset_wait_claim_sha256": "8" * 64,
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
            "model_5": {
                "sha256": ("%x" % (index + 6)) * 64,
                "filename_iteration": 5,
                "embedded_iteration": 5,
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
    lineage = {
        "motion": {"path": "motion.npz", "sha256": expected["motion_sha256"]},
        "dynamic_ready_artifact": {
            "path": "ready.json",
            "sha256": expected["dynamic_ready_artifact_file_sha256"],
        },
        "dynamic_ready_nominal_receipt": {
            "path": "hold.json",
            "sha256": expected[
                "dynamic_ready_nominal_receipt_file_sha256"
            ],
        },
        "teacher_frame0_artifact": {
            "path": "teacher.json",
            "sha256": expected["teacher_frame0_artifact_file_sha256"],
        },
        "teacher_frame0_artifact_content_sha256": expected[
            "teacher_frame0_artifact_content_sha256"
        ],
        "split_ready_reset_wait_authority": {
            "dynamic_ready": {
                "content_sha256": expected[
                    "dynamic_ready_artifact_content_sha256"
                ]
            },
            "nominal_hold_receipt": {
                "content_sha256": expected[
                    "dynamic_ready_nominal_receipt_content_sha256"
                ]
            },
            "claim_sha256": expected["split_ready_reset_wait_claim_sha256"],
        },
    }
    payload = {
        "spec": {"source": {"commit_sha": expected["source_commit_sha"]}},
        "bundle": {
            "isaac_four_grid_manifest": {
                "content_sha256": expected[
                    "four_grid_manifest_content_sha256"
                ]
            }
        },
    }
    assert barrier._shared_binding(lineage, payload) == expected
    assert "immutable_tape" not in repr(expected)
    assert "exact_zero_handoff" not in repr(expected)


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
    assert document["schema_version"] == 3
    assert document["kind"].endswith("_v3")
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
        "split_ready_reset_wait_claim_sha256"
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
        ("model_5", lambda row: row["model_5"].__setitem__("all_tensors_finite", False)),
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
