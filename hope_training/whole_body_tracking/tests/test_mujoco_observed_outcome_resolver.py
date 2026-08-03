"""Dependency-light gates for native observed net/landing evidence."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import observed_outcome_resolver as resolver  # noqa: E402
from mujoco_native import physical_ball_scene  # noqa: E402


CONTROL_DECIMATION = 4
PLANT_BINDING_SHA256 = "1" * 64
POLICY_STEP_DT_S = 0.02
RESOLVER_SOURCE_SHA256 = hashlib.sha256(
    Path(resolver.__file__).read_bytes()
).hexdigest()


def _compiled_obstacles(rows):
    geometry = {}
    ids = {}
    source_rows = (
        rows["table_top"],
        rows["robot_keepout"],
        rows["net"],
        *rows["net_posts"],
    )
    for geom_id, row in enumerate(source_rows, start=10):
        name = row["name"]
        ids[name] = geom_id
        geometry[name] = {
            "name": row["name"],
            "geom_id": geom_id,
            "body_id": 0,
            "primitive": "axis_aligned_box_full_extents_m",
            "center_mjcf_world_m": list(row["center_mjcf_world_m"]),
            "full_extents_m": list(row["full_extents_m"]),
        }
    return ids, geometry


def _authorities():
    table_scene = physical_ball_scene._load_table_scene_module()
    rows = table_scene.action_ball_policy_obstacle_geometry()
    geometry = table_scene.action_ball_policy_geometry_contract(rows)
    obstacle_ids, compiled_obstacles = _compiled_obstacles(rows)
    scene = {
        "schema_version": 1,
        "kind": "a3_mujoco_physical_ball_scene_binding_v1",
        "binding_sha256": "a" * 64,
        "assembled_xml_sha256": "b" * 64,
        "canonical_mjcf_sha256": "c" * 64,
        "table_geometry_contract_sha256": geometry["sha256"],
        "ball_contract_source": {"sha256": "d" * 64},
        "ball": {"radius_m": 0.02},
        "with_ball": True,
        "strict_pair_filter": True,
        "compiled_runtime": {
            "mujoco_version": "unit-test-backend",
            "model_timestep_s": 0.005,
            "ball_radius_m": 0.02,
            "obstacle_geom_ids": obstacle_ids,
            "obstacle_geometry": compiled_obstacles,
        },
    }
    binding = resolver.build_resolver_binding(
        scene_binding=scene,
        obstacle_rows=rows,
        plant_binding_sha256=PLANT_BINDING_SHA256,
        policy_step_dt_s=POLICY_STEP_DT_S,
        control_decimation=CONTROL_DECIMATION,
    )
    question = resolver.bind_question(
        resolver_binding=binding,
        question_source_sha256="e" * 64,
        landing_aim_xy_w_m=(2.3, -0.665),
        action_lineage_sha256="f" * 64,
    )
    return scene, rows, binding, question


def _replay_authority(scene, rows, question):
    return {
        "expected_scene_binding": scene,
        "expected_obstacle_rows": rows,
        "expected_plant_binding_sha256": PLANT_BINDING_SHA256,
        "expected_policy_step_dt_s": POLICY_STEP_DT_S,
        "expected_control_decimation": CONTROL_DECIMATION,
        "expected_resolver_source_sha256": RESOLVER_SOURCE_SHA256,
        "expected_question_source_sha256": question[
            "question_source_sha256"
        ],
        "expected_landing_aim_xy_w_m": question["landing_aim_xy_w_m"],
        "expected_action_lineage_sha256": question[
            "action_lineage_sha256"
        ],
    }


def _outgoing():
    return {
        "policy_tick": 1,
        "physics_substep": 0,
        "time_s": 0.100,
        "position_w_m": [0.7, -0.665, 1.1],
    }


def _sample(tick, substep, time_s, position, labels=()):
    return {
        "policy_tick": tick,
        "physics_substep": substep,
        "time_s": time_s,
        "ball_center_w_m": list(position),
        "active_contact_labels": list(labels),
    }


def _legal_samples():
    return [
        _sample(1, 1, 0.105, (1.80, -0.665, 1.10)),
        _sample(1, 2, 0.110, (2.00, -0.665, 1.00)),
        _sample(1, 3, 0.115, (2.30, -0.665, 0.78), ("table",)),
    ]


def _replay(samples):
    scene, rows, binding, question = _authorities()
    snapshot = resolver.replay_trace(
        resolver_binding=binding,
        question_binding=question,
        **_replay_authority(scene, rows, question),
        outgoing_flight=_outgoing(),
        samples=samples,
    )
    return binding, question, snapshot


def test_clean_native_crossing_and_first_table_contact_are_legal_and_replay_exact():
    scene, rows, _same_binding, _same_question = _authorities()
    binding, question, first = _replay(_legal_samples())
    second = resolver.replay_trace(
        resolver_binding=binding,
        question_binding=question,
        **_replay_authority(scene, rows, question),
        outgoing_flight=_outgoing(),
        samples=copy.deepcopy(_legal_samples()),
    )
    assert first == second
    assert first["status"] == resolver.STATUS_FIRST_TABLE_LANDING
    assert first["outcome_resolved"] is True
    assert first["net_crossing"]["cleared"] is True
    assert first["first_table_landing"][
        "strict_ball_radius_eroded_footprint"
    ] is True
    assert first["observed_net_clear"] is True
    assert first["observed_legal_landing"] is True
    assert resolver.validate_snapshot(
        first,
        question_binding=question,
        resolver_binding=binding,
        expected_question_binding_sha256=question["content_sha256"],
        expected_resolver_binding_sha256=binding["content_sha256"],
    ) == first


def test_net_collision_near_half_landing_and_floor_are_observed_failures():
    _binding, _question, net = _replay(
        [_sample(1, 1, 0.105, (1.84, -0.665, 0.90), ("net",))]
    )
    assert net["status"] == resolver.STATUS_NET_COLLISION
    assert net["observed_net_clear"] is False
    assert net["observed_legal_landing"] is False

    _binding, _question, near = _replay(
        [_sample(1, 1, 0.105, (1.50, -0.665, 0.78), ("table",))]
    )
    assert near["status"] == resolver.STATUS_FIRST_TABLE_LANDING
    assert near["observed_net_clear"] is False
    assert near["observed_legal_landing"] is False

    floor_samples = _legal_samples()[:2] + [
        _sample(1, 3, 0.115, (2.30, -0.665, 0.02), ("floor",))
    ]
    _binding, _question, floor = _replay(floor_samples)
    assert floor["status"] == resolver.STATUS_FLOOR_CONTACT
    assert floor["observed_net_clear"] is True
    assert floor["observed_legal_landing"] is False


def test_same_substep_crossing_and_contact_is_explicitly_ambiguous():
    _binding, _question, snapshot = _replay(
        [
            _sample(1, 1, 0.105, (1.80, -0.665, 1.10)),
            _sample(
                1,
                2,
                0.110,
                (2.20, -0.665, 0.78),
                ("table",),
            ),
        ]
    )
    assert snapshot["status"] == resolver.STATUS_SAME_SUBSTEP_AMBIGUOUS
    assert snapshot["outcome_resolved"] is False
    assert snapshot["observed_net_clear"] is None
    assert snapshot["observed_legal_landing"] is None
    assert "same_physics_substep" in snapshot["fail_closed_reason"]


def test_uncertified_crossing_segment_never_becomes_net_clear():
    _binding, _question, snapshot = _replay(
        [
            _sample(1, 1, 0.105, (1.80, -0.665, 0.80)),
            _sample(1, 2, 0.110, (2.00, -0.665, 1.20)),
            _sample(1, 3, 0.115, (2.30, -0.665, 0.78), ("table",)),
        ]
    )
    assert snapshot["net_crossing"]["ball_center_w_m"][2] > snapshot[
        "net_crossing"
    ]["required_center_z_w_m"]
    assert snapshot["net_crossing"][
        "strict_segment_endpoint_envelope_certified"
    ] is False
    assert snapshot["observed_net_clear"] is False
    assert snapshot["observed_legal_landing"] is False


@pytest.mark.parametrize(
    "landing",
    [
        (3.22, -0.665, 0.78),  # exact eroded far edge
        (2.30, -0.7425, 0.78),  # exact eroded side edge
        (2.30, -0.665, 0.70),  # underside/side contact, not top landing
    ],
)
def test_table_edge_side_and_underside_contacts_are_not_legal(landing):
    samples = _legal_samples()[:2] + [
        _sample(1, 3, 0.115, landing, ("table",))
    ]
    _binding, _question, snapshot = _replay(samples)
    assert snapshot["status"] == resolver.STATUS_FIRST_TABLE_LANDING
    assert snapshot["observed_net_clear"] is True
    assert snapshot["first_table_landing"][
        "native_top_landing_candidate"
    ] is False
    assert snapshot["observed_legal_landing"] is False


def test_outgoing_overlap_and_nonmonotonic_replay_fail_closed():
    _scene, _rows, binding, question = _authorities()
    state = resolver.ObservedOutcomeResolver(
        resolver_binding=binding, question_binding=question
    )
    state.arm(_outgoing(), active_contact_labels=("table",))
    snapshot = state.snapshot()
    assert snapshot["status"] == resolver.STATUS_OUTGOING_OVERLAP_AMBIGUOUS
    assert snapshot["outcome_resolved"] is False

    with pytest.raises(
        resolver.ObservedOutcomeResolverError, match="substep-continuous"
    ):
        resolver.replay_trace(
            resolver_binding=binding,
            question_binding=question,
            **_replay_authority(_scene, _rows, question),
            outgoing_flight=_outgoing(),
            samples=[_sample(1, 0, 0.100, (0.8, -0.665, 1.1))],
        )


@pytest.mark.parametrize(
    "samples",
    [
        [_sample(1, 2, 0.105, (1.0, -0.665, 1.1))],
        [_sample(1, 1, 0.106, (1.0, -0.665, 1.1))],
    ],
)
def test_replay_rejects_omitted_substep_and_wrong_exact_time_delta(samples):
    scene, rows, binding, question = _authorities()
    with pytest.raises(
        resolver.ObservedOutcomeResolverError,
        match="substep-continuous|time delta differs",
    ):
        resolver.replay_trace(
            resolver_binding=binding,
            question_binding=question,
            **_replay_authority(scene, rows, question),
            outgoing_flight=_outgoing(),
            samples=samples,
        )


def test_snapshot_transcript_cannot_omit_a_contact_and_reseal_summary():
    binding, question, snapshot = _replay(_legal_samples())
    omitted = copy.deepcopy(snapshot)
    omitted["transcript_samples"].pop()
    omitted["sample_count"] = len(omitted["transcript_samples"])
    omitted["last_sample"] = omitted["transcript_samples"][-1]
    omitted["last_sample_stamp"] = omitted["last_sample"]["stamp"]
    omitted["trace_sha256"] = resolver._trace_sha256(
        omitted["transcript_samples"]
    )
    omitted.pop("content_sha256")
    omitted = resolver._seal(omitted)
    with pytest.raises(
        resolver.ObservedOutcomeResolverError,
        match="cannot be rebuilt",
    ):
        resolver.validate_snapshot(
            omitted,
            question_binding=question,
            resolver_binding=binding,
            expected_question_binding_sha256=question["content_sha256"],
            expected_resolver_binding_sha256=binding["content_sha256"],
        )


def test_rollover_and_post_terminal_samples_remain_in_complete_trace():
    samples = _legal_samples() + [
        _sample(2, 0, 0.120, (2.31, -0.665, 0.80))
    ]
    _binding, _question, snapshot = _replay(samples)
    assert snapshot["status"] == resolver.STATUS_FIRST_TABLE_LANDING
    assert snapshot["sample_count"] == 5
    assert snapshot["last_sample"]["stamp"] == {
        "policy_tick": 2,
        "physics_substep": 0,
    }
    assert snapshot["trace_sha256"] == resolver._trace_sha256(
        snapshot["transcript_samples"]
    )


@pytest.mark.parametrize(
    "labels",
    [
        ("net", "table"),
        ("net", "floor"),
        ("table", "floor"),
        ("net", "table", "floor"),
    ],
)
def test_every_multiple_terminal_label_set_is_ambiguous(labels):
    _binding, _question, snapshot = _replay(
        [_sample(1, 1, 0.105, (1.0, -0.665, 0.8), labels)]
    )
    assert snapshot["status"] == resolver.STATUS_SAME_SUBSTEP_AMBIGUOUS
    assert snapshot["outcome_resolved"] is False
    assert snapshot["observed_net_clear"] is None
    assert snapshot["observed_legal_landing"] is None
    assert "multiple_terminal_contacts" in snapshot["fail_closed_reason"]


def test_geometry_drift_is_rejected_and_desired_aim_does_not_grade_observed_truth():
    scene, rows, binding, question = _authorities()
    bad_scene = copy.deepcopy(scene)
    bad_scene["table_geometry_contract_sha256"] = "0" * 64
    with pytest.raises(
        resolver.ObservedOutcomeResolverError, match="table geometry differs"
    ):
        resolver.build_resolver_binding(
            scene_binding=bad_scene,
            obstacle_rows=rows,
            plant_binding_sha256=PLANT_BINDING_SHA256,
            policy_step_dt_s=POLICY_STEP_DT_S,
            control_decimation=CONTROL_DECIMATION,
        )

    alternate_question = resolver.bind_question(
        resolver_binding=binding,
        question_source_sha256="0" * 64,
        landing_aim_xy_w_m=(1.0, 0.90),
        action_lineage_sha256="f" * 64,
    )
    alternate = resolver.replay_trace(
        resolver_binding=binding,
        question_binding=alternate_question,
        **_replay_authority(scene, rows, alternate_question),
        outgoing_flight=_outgoing(),
        samples=_legal_samples(),
    )
    assert alternate["status"] == resolver.STATUS_FIRST_TABLE_LANDING
    assert alternate["observed_net_clear"] is True
    assert alternate["observed_legal_landing"] is True

    snapshot = resolver.replay_trace(
        resolver_binding=binding,
        question_binding=question,
        **_replay_authority(scene, rows, question),
        outgoing_flight=_outgoing(),
        samples=_legal_samples(),
    )
    snapshot["observed_legal_landing"] = False
    with pytest.raises(
        resolver.ObservedOutcomeResolverError, match="content seal differs"
    ):
        resolver.validate_snapshot(
            snapshot,
            question_binding=question,
            resolver_binding=binding,
            expected_question_binding_sha256=question["content_sha256"],
            expected_resolver_binding_sha256=binding["content_sha256"],
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(resolver_source_sha256="0" * 64),
            "current source authority",
        ),
        (
            lambda value: value["geometry"].update(
                legal_ball_center_x_bounds_w_m=[1.0, 3.0]
            ),
            "not derived",
        ),
        (
            lambda value: value.update(semantics="invented_semantics"),
            "backend/semantics",
        ),
    ],
)
def test_resealed_source_geometry_and_semantic_drift_fail_closed(mutate, message):
    scene, rows, binding, _question = _authorities()
    drifted = copy.deepcopy(binding)
    drifted.pop("content_sha256")
    mutate(drifted)
    drifted = resolver._seal(drifted)
    with pytest.raises(resolver.ObservedOutcomeResolverError, match=message):
        resolver.validate_resolver_binding(
            drifted,
            expected_scene_binding=scene,
            expected_obstacle_rows=rows,
            expected_plant_binding_sha256=PLANT_BINDING_SHA256,
            expected_policy_step_dt_s=POLICY_STEP_DT_S,
            expected_control_decimation=CONTROL_DECIMATION,
            expected_resolver_source_sha256=RESOLVER_SOURCE_SHA256,
        )


def test_coordinated_geometry_reseal_and_question_rebind_need_external_parent():
    scene, rows, binding, question = _authorities()
    drifted = copy.deepcopy(binding)
    drifted.pop("content_sha256")
    drifted["geometry"]["table_x_bounds_w_m"] = [1.1, 3.3]
    drifted["geometry"]["legal_ball_center_x_bounds_w_m"] = [1.12, 3.28]
    drifted["geometry"]["robot_near_table_x_w_m"] = 1.1
    drifted = resolver._seal(drifted)
    assert resolver.validate_resolver_binding_seal(drifted) == drifted
    with pytest.raises(
        resolver.ObservedOutcomeResolverError,
        match="external scene/table authority",
    ):
        resolver.validate_resolver_binding(
            drifted,
            expected_scene_binding=scene,
            expected_obstacle_rows=rows,
            expected_plant_binding_sha256=PLANT_BINDING_SHA256,
            expected_policy_step_dt_s=POLICY_STEP_DT_S,
            expected_control_decimation=CONTROL_DECIMATION,
            expected_resolver_source_sha256=RESOLVER_SOURCE_SHA256,
        )

    rebound = resolver.bind_question(
        resolver_binding=binding,
        question_source_sha256=question["question_source_sha256"],
        landing_aim_xy_w_m=(1.0, 0.9),
        action_lineage_sha256=question["action_lineage_sha256"],
    )
    with pytest.raises(
        resolver.ObservedOutcomeResolverError,
        match="cannot be independently rebuilt",
    ):
        resolver.validate_question_binding(
            rebound,
            resolver_binding=binding,
            expected_question_source_sha256=question[
                "question_source_sha256"
            ],
            expected_landing_aim_xy_w_m=question["landing_aim_xy_w_m"],
            expected_action_lineage_sha256=question[
                "action_lineage_sha256"
            ],
        )


@pytest.mark.parametrize("drift", ["post", "ball"])
def test_compiled_five_geom_and_ball_radius_drift_from_source_is_rejected(drift):
    scene, rows, _binding, _question = _authorities()
    drifted = copy.deepcopy(scene)
    if drift == "post":
        post = physical_ball_scene.NET_GEOM_NAMES[1]
        drifted["compiled_runtime"]["obstacle_geometry"][post][
            "center_mjcf_world_m"
        ][1] += 0.01
    else:
        drifted["compiled_runtime"]["ball_radius_m"] = 0.021
    with pytest.raises(
        resolver.ObservedOutcomeResolverError,
        match="differs from table source authority|ball radius differs",
    ):
        resolver.build_resolver_binding(
            scene_binding=drifted,
            obstacle_rows=rows,
            plant_binding_sha256=PLANT_BINDING_SHA256,
            policy_step_dt_s=POLICY_STEP_DT_S,
            control_decimation=CONTROL_DECIMATION,
        )


def test_summary_has_exact_status_and_numerator_sum_closure():
    _binding, question, legal = _replay(_legal_samples())
    _binding, _question, net = _replay(
        [_sample(1, 1, 0.105, (1.84, -0.665, 0.90), ("net",))]
    )
    _binding, _question, tracking = _replay(
        [_sample(1, 1, 0.105, (1.00, -0.665, 1.10))]
    )
    _binding, _question, ambiguous = _replay(
        [
            _sample(1, 1, 0.105, (1.80, -0.665, 1.10)),
            _sample(1, 2, 0.110, (2.20, -0.665, 0.78), ("table",)),
        ]
    )
    rows = [legal, net, tracking, ambiguous]
    summary = resolver.summarize_snapshots(
        rows,
        question_binding_by_sha256={question["content_sha256"]: question},
        resolver_binding_by_sha256={
            _binding["content_sha256"]: _binding
        },
    )
    assert summary["rows"] == 4
    assert summary["armed"] == 4
    assert summary["resolved"] == 2
    assert summary["unresolved"] == 2
    assert summary["observed_net_clear"] == 1
    assert summary["observed_legal_landing"] == 1
    assert sum(summary["status_counts"].values()) == summary["rows"]
    assert all(summary["sum_closure"].values())
