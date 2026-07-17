import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from hope_planner.node_runtime_contract import (
    FormalBaseBarrierRejection,
    FormalBaseLease,
    FormalBasePosePlausibilityGuard,
    FormalBaseSourceState,
    FormalSourceFrameContract,
    FormalWireCounters,
    FormalWireExhaustion,
    MAX_EXACT_FLOAT64_INTEGER,
    SolveCadence,
    SourceStampGuard,
    SwingSideSelector,
    base_pose_is_fresh,
    base_yaw_relative_y,
    corrected_base_pose,
    latency_compensated_time_to_strike,
    ros_source_to_monotonic,
    ros_stamp_fields_to_seconds,
    validate_formal_source_clock_mode,
)
from hope_planner.planner import HOPEPlanner
from hope_planner.task_lifecycle import FormalTaskLifecycle


def test_marker_offset_is_unchanged_for_world_aligned_pose():
    base, quat = corrected_base_pose(
        (1.0, 2.0, 0.2),
        (1.0, 0.0, 0.0, 0.0),
        (0.3, -0.1, 0.4),
        policy_z_offset=0.76,
    )
    assert np.allclose(base, (1.3, 1.9, 1.36), atol=1.0e-12)
    assert np.array_equal(quat, np.array([1.0, 0.0, 0.0, 0.0]))


def test_marker_offset_rotates_with_ninety_degree_marker_yaw():
    half = math.sqrt(0.5)
    base, quat = corrected_base_pose(
        (1.0, 2.0, 0.5),
        (half, 0.0, 0.0, half),
        (0.4, 0.0, 0.2),
    )
    assert np.allclose(base, (1.0, 2.4, 0.7), atol=1.0e-12)
    assert np.allclose(quat, (half, 0.0, 0.0, half), atol=1.0e-12)


def test_missing_or_nonfinite_base_orientation_fails_closed():
    with pytest.raises(ValueError, match="orientation quaternion is missing"):
        corrected_base_pose((1, 2, 3), (0, 0, 0, 0), (0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        corrected_base_pose((1, 2, 3), (math.nan, 0, 0, 0), (0, 0, 0))


def test_formal_base_freshness_uses_mapped_source_age_and_fails_closed():
    assert not base_pose_is_fresh(None, 10.0, 0.2)
    assert base_pose_is_fresh(10.0, 10.2, 0.2)
    assert not base_pose_is_fresh(10.0, 10.200001, 0.2)
    assert not base_pose_is_fresh(10.1, 10.0, 0.2)
    assert not base_pose_is_fresh(math.nan, 10.0, 0.2)
    assert not base_pose_is_fresh(10.0, math.inf, 0.2)
    with pytest.raises(ValueError, match="base_pose_max_age_s"):
        base_pose_is_fresh(10.0, 10.1, math.nan)
    with pytest.raises(ValueError, match="base_pose_max_age_s"):
        base_pose_is_fresh(10.0, 10.1, -0.1)


def test_formal_base_lease_revokes_once_and_fresh_recovery_is_new_epoch():
    lease = FormalBaseLease()
    lease.accept(10.0, now_monotonic_s=10.0, max_age_s=0.2)
    assert lease.fresh(10.2, 0.2)
    assert not lease.expire(10.2, 0.2)
    assert lease.expire(10.200001, 0.2)
    assert not lease.expire(10.3, 0.2)
    assert not lease.fresh(10.3, 0.2)

    lease.accept(10.31, now_monotonic_s=10.31, max_age_s=0.2)
    assert lease.fresh(10.31, 0.2)
    assert lease.invalidate()
    assert not lease.invalidate()
    assert not lease.fresh(10.32, 0.2)


def test_FormalBaseSourceAgeExpiryAdvancesEpochBeforeRecovery():
    state = FormalBaseSourceState()
    counters = FormalWireCounters()
    state.accept_source(10.0, now_monotonic_s=10.0, max_age_s=0.2)

    # Expiry is evaluated before this callback's candidate is admitted.  Its
    # transition owns the next epoch and installs a monotonic source barrier.
    assert state.expire_before_admission(10.21, 0.2)
    assert counters.advance_epoch() == 2
    assert state.revoke_barrier_monotonic_s == 10.21

    # A packet received now can still carry an older source sample.  It must
    # not recover the just-revoked epoch even though its receive age is zero.
    with pytest.raises(ValueError, match="revoke barrier"):
        state.accept_source(10.20, now_monotonic_s=10.21, max_age_s=0.2)
    state.accept_source(10.22, now_monotonic_s=10.22, max_age_s=0.2)
    assert state.lease.active


def test_inactive_bad_base_still_installs_barrier_and_delayed_source_cannot_recover():
    state = FormalBaseSourceState()
    counters = FormalWireCounters()

    # A malformed callback is control-relevant even if no prior lease is
    # active. The ROS wrapper force-revokes once, advances the epoch and leaves
    # this monotonic source barrier behind.
    assert state.revoke(100.10, force=True)
    assert counters.advance_epoch() == 2
    assert state.revoke_barrier_monotonic_s == 100.10
    with pytest.raises(FormalBaseBarrierRejection, match="revoke barrier"):
        state.accept_source(100.05, now_monotonic_s=100.11, max_age_s=0.2)
    assert state.revoke_barrier_monotonic_s == 100.10

    # 100 Hz callbacks with 15 ms fixed source latency must not chase the
    # receive clock forever. The first source is still 5 ms pre-barrier; the
    # next source is 5 ms post-barrier and must recover under the same T0.
    constant_latency = FormalBaseSourceState()
    assert constant_latency.revoke(200.0, force=True)
    with pytest.raises(FormalBaseBarrierRejection):
        constant_latency.accept_source(
            199.995, now_monotonic_s=200.010, max_age_s=0.2
        )
    assert constant_latency.revoke_barrier_monotonic_s == 200.0
    constant_latency.accept_source(
        200.005, now_monotonic_s=200.020, max_age_s=0.2
    )
    assert constant_latency.lease.active
    assert constant_latency.revoke_barrier_monotonic_s == 200.0

    # If callback-entry expiry already installed the barrier, its later error
    # path must not force a second transition/epoch.
    state.accept_source(100.20, now_monotonic_s=100.20, max_age_s=0.2)
    assert state.expire_before_admission(100.41, 0.2)
    assert counters.advance_epoch() == 3
    assert not state.revoke(100.41, force=False)
    assert counters.control_epoch == 3


def test_formal_base_pose_plausibility_bounds_jumps_and_preserves_good_baseline():
    guard = FormalBasePosePlausibilityGuard()
    assert guard.min_source == (-3.0, -3.0, 0.4)
    assert guard.max_source == (3.0, 3.0, 1.5)
    assert guard.linear_slack_m == 0.05
    assert guard.max_linear_speed_mps == 8.0
    assert guard.angular_slack_rad == 0.15
    assert guard.max_angular_speed_radps == 12.0
    p0 = (-0.5, -0.75, 0.9)
    q0 = (1.0, 0.0, 0.0, 0.0)
    guard.commit(p0, q0, 10.0)

    # Inclusive hard-contract boundaries are accepted; a finite absurd pose
    # and a below-floor/above-arena pelvis fail closed.
    FormalBasePosePlausibilityGuard().validate((-3.0, 3.0, 0.4), q0, 10.5)
    with pytest.raises(ValueError, match="source-frame workspace"):
        guard.validate((1000.0, 0.0, 0.9), q0, 10.01)
    with pytest.raises(ValueError, match="source-frame workspace"):
        guard.validate((0.0, 0.0, 1.500001), q0, 10.01)

    # At 10 ms, 5 cm slack + 8 m/s permits 13 cm, not a 20 cm teleport.
    guard.validate((-0.38, -0.75, 0.9), q0, 10.01)
    with pytest.raises(ValueError, match="displacement"):
        guard.validate((-0.3, -0.75, 0.9), q0, 10.01)

    # Quaternion sign is equivalent; shortest-angle rejects a true pi jump.
    guard.validate(p0, (-1.0, 0.0, 0.0, 0.0), 10.01)
    with pytest.raises(ValueError, match="angular jump"):
        guard.validate(p0, (0.0, 0.0, 0.0, 1.0), 10.01)
    with pytest.raises(ValueError, match="did not advance"):
        guard.validate(p0, q0, 10.0)

    # Rejected candidates never poison the last accepted baseline; a later
    # physically reachable sample can still commit.
    assert guard.last_source_monotonic_s == 10.0
    guard.commit((-0.3, -0.75, 0.9), q0, 10.1)
    assert guard.last_source_monotonic_s == 10.1


def test_ReadyNeverClaimsReceiveFreshWhenSourceLeaseExpired():
    state = FormalBaseSourceState()
    # The callback itself is newly received at 20.20, but the mapped sensor
    # source is already 210 ms old under the 200 ms formal lease.
    with pytest.raises(ValueError, match="source lease is stale"):
        state.accept_source(19.99, now_monotonic_s=20.20, max_age_s=0.2)
    assert not state.ready_for_source(19.99, 20.20, 0.2)

    state.accept_source(20.01, now_monotonic_s=20.20, max_age_s=0.2)
    assert state.ready_for_source(20.01, 20.20, 0.2)
    assert not state.ready_for_source(20.01, 20.211, 0.2)


def test_formal_source_frames_are_nonempty_and_exact():
    frames = FormalSourceFrameContract(
        ball_frame_id="world", base_frame_id="world"
    )
    frames.validate_ball("world")
    frames.validate_base("world")
    with pytest.raises(ValueError, match="ball source frame_id mismatch"):
        frames.validate_ball("map")
    with pytest.raises(ValueError, match="base source frame_id mismatch"):
        frames.validate_base("/world")
    with pytest.raises(ValueError, match="non-empty"):
        FormalSourceFrameContract(ball_frame_id="", base_frame_id="world")
    with pytest.raises(ValueError, match="common frame"):
        FormalSourceFrameContract(ball_frame_id="world", base_frame_id="odom")
    with pytest.raises(ValueError, match="cannot be disabled"):
        FormalSourceFrameContract(
            ball_frame_id="world",
            base_frame_id="odom",
            common_frame_required=False,
        )


def test_formal_source_clock_mode_requires_an_exact_boolean_use_sim_time():
    assert validate_formal_source_clock_mode(False, "system") == "system"
    assert validate_formal_source_clock_mode(True, "sim") == "sim"
    with pytest.raises(ValueError, match="exact boolean"):
        validate_formal_source_clock_mode(None, "system")
    with pytest.raises(ValueError, match="exact boolean"):
        validate_formal_source_clock_mode(0, "system")
    with pytest.raises(ValueError, match="exactly match"):
        validate_formal_source_clock_mode(False, "sim")


def test_formal_wire_reserves_max_for_one_terminal_invalid_then_never_recovers():
    counters = FormalWireCounters(
        control_epoch=MAX_EXACT_FLOAT64_INTEGER - 1,
        racket_sequence=MAX_EXACT_FLOAT64_INTEGER - 1,
        base_sequence=MAX_EXACT_FLOAT64_INTEGER - 2,
    )
    with pytest.raises(FormalWireExhaustion):
        counters.advance_epoch()
    assert counters.next_base() == MAX_EXACT_FLOAT64_INTEGER - 1
    assert counters.reserve_terminal_invalid() == (
        MAX_EXACT_FLOAT64_INTEGER,
        MAX_EXACT_FLOAT64_INTEGER,
        MAX_EXACT_FLOAT64_INTEGER,
    )
    with pytest.raises(RuntimeError, match="permanently exhausted"):
        counters.next_racket()
    with pytest.raises(RuntimeError, match="permanently exhausted"):
        counters.reserve_terminal_invalid()


def test_ros_node_racket_rows_reference_latest_attempted_base_sequence():
    source = (
        Path(__file__).resolve().parents[1] / "hope_planner" / "node.py"
    ).read_text(encoding="utf-8")

    invalid_publisher = source.split(
        "def _publish_flat_racket_invalid", 1
    )[1].split("def _publish_flat_base", 1)[0]
    assert "base_sequence_ref=self._wire_counters.base_sequence" in invalid_publisher

    valid_publisher = source.split(
        "if self.flat_cmd_pub is not None:\n            formal_wire = {}", 1
    )[1].split("# The flat wire is the formal Gate-3 control path.", 1)[0]
    assert '"base_sequence_ref": self._wire_counters.base_sequence' in valid_publisher

    base_publisher = source.split("def _publish_flat_base", 1)[1].split(
        "def _formal_base_is_fresh", 1
    )[0]
    assert base_publisher.index('self._next_sequence("_base_sequence")') < (
        base_publisher.index("self.flat_base_pub.publish(base)")
    )

    terminal = source.split("def _publish_terminal_wire_exhaustion", 1)[1].split(
        "def _source_stamp_s", 1
    )[0]
    assert "base_sequence_ref=base_sequence" in terminal


def test_source_stamp_guard_rejects_replay_stale_future_and_regression():
    guard = SourceStampGuard(max_age_s=0.2, future_tolerance_s=0.02)
    guard.validate(9.9, 10.0)
    assert guard.last_accepted_stamp_s is None  # two-phase: body is not committed yet
    guard.commit(9.9)
    assert guard.last_accepted_stamp_s == 9.9
    with pytest.raises(ValueError, match="duplicated or regressed"):
        guard.accept(9.9, 10.0)
    with pytest.raises(ValueError, match="duplicated or regressed"):
        guard.accept(9.8, 10.0)
    with pytest.raises(ValueError, match="stale"):
        guard.accept(9.79, 10.0)
    with pytest.raises(ValueError, match="future"):
        guard.accept(10.021, 10.0)
    with pytest.raises(ValueError, match="non-negative"):
        guard.accept(-0.1, 0.0)
    with pytest.raises(ValueError, match="finite"):
        guard.accept(math.nan, 10.0)


def test_ros_source_mapping_preserves_upstream_age_and_runner_adds_dds_age():
    mapped = ros_source_to_monotonic(99.81, 100.0, 500.0)
    assert mapped == pytest.approx(499.81)
    # A 20 ms DDS/solve delay makes the end-to-end age 210 ms; publication
    # cannot reset it to 20 ms.
    assert 500.02 - mapped == pytest.approx(0.21)
    assert 1.0 - (500.02 - mapped) == pytest.approx(0.79)  # TTS decays end-to-end


def test_formal_tts_removes_transport_and_solve_age_without_granting_future_skew():
    assert latency_compensated_time_to_strike(0.50, 100.00, 100.08) == pytest.approx(
        0.42
    )
    # A source stamp inside the separate future-tolerance never creates extra
    # preparation time.
    assert latency_compensated_time_to_strike(0.50, 100.01, 100.00) == 0.50
    # Missed deadlines remain negative so the runner can reject them.
    assert latency_compensated_time_to_strike(0.05, 100.00, 100.08) == pytest.approx(
        -0.03
    )


@pytest.mark.parametrize(
    "sample_tts,source_stamp,now_ros",
    [
        (math.nan, 1.0, 1.0),
        (0.5, math.inf, 1.0),
        (0.5, 1.0, math.nan),
        (0.5, -0.1, 1.0),
        (0.5, 1.0, -0.1),
    ],
)
def test_formal_tts_rejects_nonfinite_or_negative_clock_inputs(
    sample_tts, source_stamp, now_ros
):
    with pytest.raises(ValueError):
        latency_compensated_time_to_strike(sample_tts, source_stamp, now_ros)


def test_ros_stamp_fields_reject_invalid_nanoseconds_and_negative_seconds():
    assert ros_stamp_fields_to_seconds(7, 123_000_000) == pytest.approx(7.123)
    with pytest.raises(ValueError, match="out of range"):
        ros_stamp_fields_to_seconds(7, 1_000_000_000)
    with pytest.raises(ValueError, match="out of range"):
        ros_stamp_fields_to_seconds(-1, 0)


def test_solve_cadence_preserves_explicit_every_sample_override_and_decimates():
    legacy = SolveCadence(0.0)
    assert [legacy.admit(t) for t in (1.0, 1.001, 1.001)] == [True, True, True]

    cadence = SolveCadence(0.033)
    assert cadence.admit(10.0)
    assert not cadence.admit(10.010)
    assert not cadence.admit(10.032999)
    assert cadence.admit(10.033)
    assert not cadence.admit(10.040)
    assert cadence.admit(9.0)  # clock regression starts a new deterministic epoch
    assert not cadence.admit(9.010)
    assert cadence.admit(9.033)
    with pytest.raises(ValueError):
        cadence.admit(math.nan)


def test_50hz_cadence_ingests_300hz_burst_and_only_revises_current_samples():
    class PlannerProbe:
        def __init__(self):
            self.ingested = []
            self.solved = []

        def push_measurement(self, t, p_ball):
            self.ingested.append((t, p_ball.copy()))

        def update(self, t, p_ball):
            self.ingested.append((t, p_ball.copy()))
            self.solved.append((t, p_ball.copy()))

    cadence = SolveCadence(0.02)
    planner = PlannerProbe()
    lifecycle = FormalTaskLifecycle()
    lifecycle.observe_epoch(7)
    lifecycle.explicit_rearm(7, no_ball_or_new_serve_confirmed=True)
    revisions = []

    for index in range(300):
        timestamp = index / 300.0
        sample = np.array([float(index), -float(index), 0.5])
        if cadence.admit(timestamp):
            planner.update(timestamp, sample)
            revisions.append(
                lifecycle.publish(
                    7, inbound_track_ready=True, solver_valid=True
                )
            )
        else:
            planner.push_measurement(timestamp, sample)

    assert len(planner.ingested) == 300
    assert len(planner.solved) == 50
    assert [row[0] for row in planner.solved] == pytest.approx(
        [index / 300.0 for index in range(0, 300, 6)]
    )
    # Every solve consumes the sample from that callback, never a cached earlier
    # measurement, while the final non-solve sample is still ingested.
    assert [int(row[1][0]) for row in planner.solved] == list(range(0, 300, 6))
    assert int(planner.ingested[-1][1][0]) == 299
    assert all(revision is not None for revision in revisions)
    assert {revision.task_id for revision in revisions} == {1}
    assert [revision.task_revision for revision in revisions] == list(range(1, 51))


def test_cadence_rejected_measurement_keeps_cached_command_and_stage2_tuple():
    planner = HOPEPlanner()
    cached_command = object()
    cached_strike = object()
    planner._latest_command = cached_command
    planner._latest_strike = cached_strike
    planner._latest_t = 4.0
    lifecycle = FormalTaskLifecycle()
    lifecycle.observe_epoch(7)
    lifecycle.explicit_rearm(7, no_ball_or_new_serve_confirmed=True)
    lifecycle.publish(7, inbound_track_ready=True, solver_valid=True)
    revision_before_skip = lifecycle.active_revision

    planner.push_measurement(4.01, np.array([1.0, 0.0, 0.5]))

    assert planner.racket_command is cached_command
    assert planner.strike_target is cached_strike
    assert planner._latest_t == 4.0
    assert planner.estimator.t_buffer == [4.01]
    assert lifecycle.active_task_id == 1
    assert lifecycle.active_revision == revision_before_skip


def test_replan_latest_changes_only_stage3_output():
    planner = HOPEPlanner()
    cached_strike = object()
    replanned_command = object()
    planner._latest_strike = cached_strike
    planner._latest_t = 7.0
    planner.target_planner.plan = lambda strike: replanned_command

    assert planner.replan_latest() is replanned_command
    assert planner.strike_target is cached_strike
    assert planner._latest_t == 7.0


def test_swing_side_uses_current_base_yaw_relative_intercept_and_hysteresis():
    selector = SwingSideSelector(split_y=0.0, hysteresis_y=0.04)
    base = (0.0, 1.0, 0.0)
    quat = (1.0, 0.0, 0.0, 0.0)
    assert selector.select((0.5, 0.95, 0.8), base, quat) == 1.0
    # Noise inside the +/-4 cm Schmitt band cannot flip the selected clip.
    assert selector.select((0.5, 1.02, 0.8), base, quat) == 1.0
    assert selector.select((0.5, 1.05, 0.8), base, quat) == -1.0
    assert selector.select((0.5, 0.98, 0.8), base, quat) == -1.0
    assert selector.select((0.5, 0.95, 0.8), base, quat) == 1.0


def test_ten_degree_yaw_counterexample_matches_cpp_tgt_b_and_selects_forehand():
    half_yaw = math.radians(10.0) / 2.0
    quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    base = (1.0, 2.0, 0.9)
    intercept = (1.67, 2.02, 0.9)
    relative_y = base_yaw_relative_y(intercept, base, quat)
    assert np.isclose(relative_y, -0.096648124, atol=1.0e-9)
    assert SwingSideSelector(split_y=0.0, hysteresis_y=0.04).select(
        intercept, base, quat
    ) == 1.0


def test_side_selection_rejects_missing_or_nonfinite_base_orientation():
    selector = SwingSideSelector()
    with pytest.raises(ValueError, match="orientation quaternion is missing"):
        selector.select((0.67, 0.02, 0.8), (0.0, 0.0, 0.9), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="must be finite"):
        selector.select((0.67, 0.02, 0.8), (0.0, 0.0, 0.9), (math.nan, 0, 0, 0))


@pytest.mark.parametrize("period", [-0.1, math.nan, math.inf])
def test_invalid_solve_period_fails_closed(period):
    with pytest.raises(ValueError):
        SolveCadence(period)


def test_gate3_sim_profile_binds_long_horizon_and_nonstarving_cadence():
    path = Path(__file__).resolve().parents[1] / "config" / "hope_planner.sim.yaml"
    params = yaml.safe_load(path.read_text(encoding="utf-8"))["hope_planner"][
        "ros__parameters"
    ]
    assert params["max_predict_time"] == 2.6
    assert params["solve_period_s"] == 0.033
    assert params["base_pose_max_age_s"] == 0.2
    assert params["ball_source_stamp_max_age_s"] == 0.2
    assert params["formal_source_clock_mode"] == "system"
    assert params["use_sim_time"] is False
    assert params["formal_ball_source_frame_id"] == "world"
    assert params["formal_base_source_frame_id"] == "odom"
    assert params["formal_common_frame_required"] is True
    assert params["use_shadow_solver"] is False
    with pytest.raises(ValueError, match="common frame"):
        FormalSourceFrameContract(
            ball_frame_id=params["formal_ball_source_frame_id"],
            base_frame_id=params["formal_base_source_frame_id"],
            common_frame_required=params["formal_common_frame_required"],
        )
    assert params["racket_flat_schema"] == 3
    assert params["marker_to_base_xyz"] == [0.0, 0.0, 0.0]

    base_path = Path(__file__).resolve().parents[1] / "config" / "hope_planner.yaml"
    base_params = yaml.safe_load(base_path.read_text(encoding="utf-8"))["hope_planner"][
        "ros__parameters"
    ]
    assert base_params["use_sim_time"] is False
    assert base_params["formal_ball_source_frame_id"] == "world"
    assert base_params["formal_base_source_frame_id"] == "world"
    assert base_params["formal_common_frame_required"] is True
    assert base_params["use_shadow_solver"] is False
    FormalSourceFrameContract(
        ball_frame_id=base_params["formal_ball_source_frame_id"],
        base_frame_id=base_params["formal_base_source_frame_id"],
        common_frame_required=base_params["formal_common_frame_required"],
    )

    vendor_path = Path(__file__).resolve().parents[4] / (
        "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/"
        "a3_pingpong_iceoryx_cfg.yaml"
    )
    vendor_cfg = yaml.safe_load(vendor_path.read_text(encoding="utf-8"))
    vendor_publishers = vendor_cfg["MujocoSimModule"]["publisher_options"]
    pelvis = next(
        publisher
        for publisher in vendor_publishers
        if publisher["topic"] == "/sim/a3/pelvis_pose"
    )
    assert pelvis["options"]["frame_id"] == params["formal_base_source_frame_id"]


def test_arena_and_task_revision_profiles_bind_50hz_solve_cadence():
    config_dir = Path(__file__).resolve().parents[1] / "config"
    arena = yaml.safe_load(
        (config_dir / "hope_planner.yaml").read_text(encoding="utf-8")
    )["hope_planner"]["ros__parameters"]
    revision = yaml.safe_load(
        (config_dir / "hope_planner.task_revision.yaml").read_text(
            encoding="utf-8"
        )
    )["hope_planner"]["ros__parameters"]

    assert arena["solve_period_s"] == 0.02
    assert revision["solve_period_s"] == 0.02


def test_task_revision_overlay_is_explicit_schema4_and_freezes_ball_boundaries():
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "hope_planner.task_revision.yaml"
    )
    params = yaml.safe_load(path.read_text(encoding="utf-8"))["hope_planner"][
        "ros__parameters"
    ]
    assert params == {
        "racket_flat_schema": 4,
        "solve_period_s": 0.02,
        "formal_task_no_ball_rearm_s": 0.10,
        "formal_task_inbound_vx_threshold_mps": -0.30,
        "formal_task_outbound_vx_threshold_mps": 0.30,
        "formal_task_plane_close_margin_m": 0.02,
        "formal_task_deadline_close_grace_s": 0.08,
    }


def test_ros_node_wires_one_corrected_base_to_flat_adaptive_x_and_side():
    source = (
        Path(__file__).resolve().parents[1] / "hope_planner" / "node.py"
    ).read_text(encoding="utf-8")
    callback = source.split("def _robot_pose_cb", 1)[1].split(
        "self.create_subscription(PoseStamped", 1
    )[0]
    assert callback.index("self._expire_formal_base_if_needed(received_now)") < (
        callback.index("self._formal_source_frames.validate_base")
    )
    assert callback.index("self._formal_source_frames.validate_base") < callback.index(
        "base_w, quat_wxyz = corrected_base_pose("
    )
    assert "base_w, quat_wxyz = corrected_base_pose(" in callback
    assert callback.index("self._formal_base_state.validate_candidate(") < (
        callback.index("self._formal_base_pose_guard.validate(")
    )
    assert callback.index("bx, by, bz =") < callback.index("self._robot_x = bx")
    assert callback.index("self._robot_position_w = base_w.copy()") < callback.index(
        "self._publish_flat_base("
    )
    assert callback.index("self._robot_quaternion_wxyz = quat_wxyz.copy()") < callback.index(
        "self._publish_flat_base("
    )
    assert '"invalid robot marker pose",' in callback
    assert callback.index("self._formal_base_state.accept_source(") < (
        callback.index("self._emit_base_ready_from_new_sample(")
    )
    assert callback.index("self._formal_base_pose_guard.commit(") < (
        callback.index("self._publish_flat_base(")
    )
    admission_prefix = callback.split("self._formal_base_state.accept_source(", 1)[0]
    assert admission_prefix.rfind(
        "expired_during_admission = self._expire_formal_base_if_needed("
    ) > admission_prefix.rfind("base_w, quat_wxyz = corrected_base_pose(")
    frame_reject = callback.split("self._formal_source_frames.validate_base", 1)[1].split(
        "source_stamp_s = self._source_stamp_s", 1
    )[0]
    assert "base source frame_id rejected" in frame_reject
    assert "force_epoch=not expired_this_callback" in frame_reject
    # Every bad-sample exit owns a fail-closed barrier even while the prior
    # lease is already inactive; callback-entry/admission expiry suppresses a
    # duplicate epoch advance in the same callback.
    assert callback.count("not expired_this_callback") == 4
    assert callback.count("and not barrier_only_rejection") == 2
    assert callback.count("FormalBaseBarrierRejection") == 2
    assert "expired_during_admission = self._expire_formal_base_if_needed(" in callback
    assert "expired_this_callback or expired_during_admission" in callback

    pose_callback = source.split("def _poses_cb", 1)[1]
    assert pose_callback.index("self._formal_source_frames.validate_ball") < (
        pose_callback.index("p_ball = np.array")
    )
    ball_frame_reject = pose_callback.split("ball source frame_id rejected", 1)[0]
    assert "self._publish_flat_racket_invalid()" in ball_frame_reject
    assert "_revoke_formal_base" not in ball_frame_reject
    assert "self._solve_cadence.admit(t)" in pose_callback
    assert "self.planner.push_measurement(t, p_ball)" in pose_callback
    assert "self._side_selector.select(" in pose_callback
    assert "self._robot_position_w," in pose_callback
    assert "self._robot_quaternion_wxyz," in pose_callback
    assert pose_callback.index(
        "self._expire_formal_base_if_needed(time.monotonic())"
    ) < pose_callback.index("if not solve_now:")
    assert "swing_sign=swing_sign" in pose_callback
    clear_method = source.split("def _clear_formal_base_geometry", 1)[1].split(
        "def _publish_formal_revocation", 1
    )[0]
    assert "self._robot_quaternion_wxyz = None" in clear_method
    assert "self._side_selector.sign = 0.0" in clear_method
    revoke_method = source.split("def _publish_formal_revocation", 1)[1].split(
        "def _revoke_formal_base", 1
    )[0]
    assert "self._publish_flat_racket_invalid()" in revoke_method
    assert "self._publish_flat_base(valid=False)" in revoke_method
    assert revoke_method.count("try:") >= 2


def test_ready_heartbeat_is_new_sample_only_throttled_and_timer_only_revokes():
    source = (
        Path(__file__).resolve().parents[1] / "hope_planner" / "node.py"
    ).read_text(encoding="utf-8")
    emitter = source.split("def _emit_base_ready_from_new_sample", 1)[1].split(
        "def _poses_cb", 1
    )[0]
    assert "self._formal_base_state.ready_for_source(" in emitter
    assert "self._base_ready_cadence.admit(now_monotonic_s)" in emitter
    timer = source.split("def _publish_diagnostics", 1)[1]
    assert "self._expire_formal_base_if_needed(time.monotonic())" in timer
    assert "_emit_base_ready_from_new_sample" not in timer


def test_node_binds_configurable_prediction_horizon_into_planner_config():
    source = (
        Path(__file__).resolve().parents[1] / "hope_planner" / "node.py"
    ).read_text(encoding="utf-8")
    assert 'self.declare_parameter("max_predict_time", 2.0)' in source
    assert 'self.declare_parameter("solve_period_s", 0.02)' in source
    assert 'max_predict_time=float(self.get_parameter("max_predict_time").value)' in source
    assert 'self.declare_parameter("base_pose_max_age_s", 0.2)' in source
    assert 'if not self.has_parameter("use_sim_time"):' in source
    assert "validate_formal_source_clock_mode(" in source
    assert 'self.declare_parameter("formal_common_frame_required", True)' in source
