from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "planner_revision.py"
)


def _load_module():
    name = "planner_revision_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load_module()


def _profile(**overrides):
    values = {
        "policy_dt_s": 0.02,
        "min_tts_s": 0.10,
        "max_tts_s": 2.00,
        "max_phase_rate_per_s": 4.0,
        "max_phase_acceleration_per_s2": 20.0,
        "max_deadline_revision_delta_s": 0.25,
        "max_position_revision_delta_m": 0.10,
        "max_velocity_revision_delta_mps": 0.50,
        "max_normal_revision_delta_rad": 0.20,
    }
    values.update(overrides)
    return P.PhaseGovernorProfile(**values)


def _truth(control_epoch=7, task_id=10, digest="a" * 64):
    return P.LatentTaskTruth(
        control_epoch=control_epoch,
        task_id=task_id,
        truth_sha256=digest,
    )


def _revision(
    *,
    control_epoch=7,
    task_id=10,
    task_revision=1,
    command_sequence=100,
    source_monotonic_s=5.0,
    digest="a" * 64,
    position=(0.2, -0.1, 0.9),
    velocity=(-2.0, 0.0, -0.2),
    normal=(1.0, 0.0, 0.0),
    tts=0.8,
):
    return P.PlannerTaskRevision(
        control_epoch=control_epoch,
        task_id=task_id,
        task_revision=task_revision,
        command_sequence=command_sequence,
        source_monotonic_s=source_monotonic_s,
        truth_sha256=digest,
        target_position_m=position,
        target_velocity_mps=velocity,
        target_normal=normal,
        desired_tts_s=tts,
    )


def _begin(profile=None):
    profile = profile or _profile()
    decision = P.begin_task(
        profile,
        P.PhaseGovernorLedger(),
        _truth(),
        _revision(),
        local_monotonic_s=20.0,
    )
    assert decision.accepted, decision.reason
    return profile, decision.ledger


def test_profile_is_complete_canonical_and_content_addressed():
    profile = _profile()
    document = profile.document()
    assert document["contract_version"] == "phase_governor_v1"
    assert document["schema_version"] == 1
    assert json.loads(P.canonical_profile_bytes(document)) == document
    assert profile.hard_contract()["profile_sha256"] == profile.profile_sha256
    assert len(profile.profile_sha256) == 64
    assert replace(profile, max_phase_rate_per_s=4.1).profile_sha256 != profile.profile_sha256
    assert P.PhaseGovernorProfile.from_mapping(document) == profile
    with pytest.raises(ValueError, match="missing"):
        P.PhaseGovernorProfile.from_mapping(
            {key: value for key, value in document.items() if key != "policy_dt_s"}
        )
    with pytest.raises(ValueError, match="unknown"):
        P.PhaseGovernorProfile.from_mapping({**document, "unbound_default": 1})


def _tts_mixture_document():
    return {
        "contract_version": "initial_tts_mixture_v1",
        "components": [
            {"name": "late_stress", "range_s": [0.25, 0.49], "weight": 0.15},
            {"name": "baseline_0p5", "range_s": [0.5, 0.5], "weight": 0.20},
            {"name": "fast_deploy", "range_s": [0.5, 0.9], "weight": 0.30},
            {"name": "broad_arrival", "range_s": [0.9, 1.7], "weight": 0.35},
        ],
    }


def test_initial_tts_mixture_has_explicit_0p5_mass_and_broad_support():
    mixture = P.InitialTtsMixture.from_mapping(_tts_mixture_document())
    assert mixture.support_range_s == (0.25, 1.7)
    mixture.validate_support(lo_s=0.25, hi_s=1.7)
    assert mixture.document() == _tts_mixture_document()
    baseline = next(row for row in mixture.components if row.name == "baseline_0p5")
    assert (baseline.lo_s, baseline.hi_s, baseline.weight) == (0.5, 0.5, 0.2)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(extra=True), "unknown"),
        (lambda d: d.pop("components"), "missing"),
        (lambda d: d["components"][0].update(weight=0.0), "weight"),
        (lambda d: d["components"][0].update(range_s=[0.6, 0.5]), "hi_s"),
        (lambda d: d["components"][1].update(name="late_stress"), "unique"),
        (lambda d: d["components"][0].update(weight=0.14), "sum to 1"),
    ],
)
def test_initial_tts_mixture_fails_closed(mutate, match):
    document = _tts_mixture_document()
    mutate(document)
    with pytest.raises(ValueError, match=match):
        P.InitialTtsMixture.from_mapping(document)


def test_initial_tts_mixture_support_must_bind_runtime_envelope():
    mixture = P.InitialTtsMixture.from_mapping(_tts_mixture_document())
    with pytest.raises(ValueError, match="exactly equal"):
        mixture.validate_support(lo_s=0.5, hi_s=1.7)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"policy_dt_s": 0.0}, "policy_dt_s"),
        ({"min_tts_s": 1.0, "max_tts_s": 1.0}, "greater"),
        ({"max_phase_rate_per_s": float("nan")}, "finite"),
        ({"max_normal_revision_delta_rad": math.pi + 0.01}, "<= pi"),
        ({"normal_unit_tolerance": 1.0}, "must be < 1"),
        ({"schema_version": 2}, "unsupported"),
    ],
)
def test_profile_fails_closed(overrides, match):
    with pytest.raises(ValueError, match=match):
        _profile(**overrides)


def test_strict_atomic_mapping_rejects_missing_unknown_nonfinite_and_bad_normal():
    payload = {
        "schema_version": 1,
        "control_epoch": 7,
        "task_id": 10,
        "task_revision": 1,
        "command_sequence": 100,
        "source_monotonic_s": 5.0,
        "truth_sha256": "a" * 64,
        "target_position_m": [0.2, -0.1, 0.9],
        "target_velocity_mps": [-2.0, 0.0, -0.2],
        "target_normal": [1.0, 0.0, 0.0],
        "desired_tts_s": 0.8,
    }
    parsed = P.PlannerTaskRevision.from_mapping(payload)
    assert parsed.target_position_m == (0.2, -0.1, 0.9)

    for broken, match in [
        ({key: value for key, value in payload.items() if key != "desired_tts_s"}, "missing"),
        ({**payload, "partial_position_x": 1.0}, "unknown"),
        ({**payload, "target_velocity_mps": [0.0, float("inf"), 0.0]}, "finite"),
        ({**payload, "task_id": 0}, "task_id"),
        ({**payload, "task_revision": 0}, "task_revision"),
        ({**payload, "command_sequence": 0}, "command_sequence"),
    ]:
        with pytest.raises(ValueError, match=match):
            P.PlannerTaskRevision.from_mapping(broken)

    profile, ledger = _begin()
    bad_normal = replace(
        _revision(task_revision=2, command_sequence=101, source_monotonic_s=5.1),
        target_normal=(0.5, 0.0, 0.0),
    )
    decision = P.revise_task(profile, ledger, bad_normal)
    assert not decision.accepted
    assert decision.ledger is ledger


def test_training_wiring_is_default_off_and_uses_one_top_level_block():
    commands = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    ).read_text()
    racket = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    ).read_text()
    train = (ROOT / "scripts/train.py").read_text()
    assert "planner_revision_enabled: bool = False" in commands
    assert "planner_revision_enabled: bool = False" in racket
    assert '_get(task, "planner_revision")' in train
    assert "commands.motion + commands.racket_target (task.planner_revision)" in train
    assert "enabled task.planner_revision is incomplete" in train
    for key in (
        "profile",
        "initial_tts_range_s",
        "initial_tts_mixture",
        "position_std_m",
        "velocity_std_mps",
        "normal_std_rad",
        "tts_std_s",
    ):
        assert f'"{key}"' in train
    # Deployment rejects a degenerate preparation-time range.  Training must not create a
    # checkpoint whose exact runtime contract cannot be parsed by the runner/exporter.
    assert "initial_tts[0] < initial_tts[1]" in train
    assert "< initial_tts[1]" in commands
    assert "< initial_tts[1]" in racket
    assert "planner_revision_initial_tts_mixture: dict | None = None" in commands
    assert "planner_revision_initial_tts_mixture: dict | None = None" in racket
    assert "InitialTtsMixture.from_mapping" in train
    assert "InitialTtsMixture.from_mapping" in commands
    assert "InitialTtsMixture.from_mapping" in racket
    assert "def _sample_planner_initial_tts" in racket
    assert "torch.multinomial" in racket
    assert "torch.bincount" in racket
    assert "initial_tts = self._sample_planner_initial_tts" in racket
    for counter in (
        "planner_initial_tts_sample_count",
        "planner_initial_tts_sub_0p5_count",
        "planner_initial_tts_exact_0p5_count",
        "planner_initial_tts_above_0p5_count",
        "planner_initial_tts_component_{index}_count",
    ):
        assert counter in racket


def test_training_revision_updates_actor_tuple_not_truth_or_question():
    source = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    ).read_text()
    body = source.split("def _revise_same_ball_actor_tuple", 1)[1].split(
        "def _compute_strike_timing", 1
    )[0]
    # The revision path reads truth, but only writes the separate visible ledger.
    assert "self.racket_target_pos_w[" not in body.split(
        "self._planner_visible_pos[accepted_ids]", 1
    )[1]
    assert "self._planner_visible_pos[accepted_ids]" in body
    assert "self._planner_visible_vel[accepted_ids]" in body
    assert "self._planner_visible_normal[accepted_ids]" in body
    assert "self._planner_visible_tts[accepted_ids]" in body
    assert "submit_planner_revision" in body


def test_coupled_transport_replaces_the_uncoupled_delay_ring_no_launch_guard():
    """2026-07-25:修订+延迟不再 NO-LAUNCH,改为耦合传输(提交侧在途环)。

    钉死的新合同:'live' tts 模式仍 fail-loud(元组晚到、时钟即时是矛盾传输语义);
    耦合模式 actor 端不叠观测延迟环(否则总延迟 2d);BEGIN 即时且作废旧球在途修订。
    """
    source = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    ).read_text()
    # 旧 NO-LAUNCH 守卫必须已拆除
    assert "are not launchable" not in source
    # 校验函数存在且对 'live' 模式 fail-loud
    guard = source.split("def _coupled_transport_mode", 1)[1].split(
        "def _target_delay_tts_mode", 1
    )[0]
    assert "planner_revision_enabled" in guard and "target_delay_steps" in guard
    assert "source_timestamp_compensated" in guard and "uncompensated" in guard
    assert "raise ValueError" in guard
    # __init__ 走这条校验;耦合模式下 actor 观测环步数强制 0
    assert "self._coupled_transport = _coupled_transport_mode(cfg)" in source
    assert (
        "self._actor_ring_steps = 0 if self._coupled_transport else self._delay_steps"
        in source
    )
    # 提交侧在途环存在;BEGIN 不过环且作废旧球在途修订
    assert "def _exchange_pending_planner_revision" in source
    begin = source.split("def _begin_same_ball_planner_task", 1)[1].split(
        "def _revise_same_ball_actor_tuple", 1
    )[0]
    assert "self._pend_valid[:, ids] = False" in begin
    assert "_exchange_pending_planner_revision" not in begin


def test_training_revision_envelopes_are_immutable_task_begin_not_visible_chain():
    commands = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    ).read_text()
    submit = commands.split("def submit_planner_revision", 1)[1].split(
        "def _advance_planner_phase", 1
    )[0]
    for token in (
        "self._planner_begin_tts",
        "self._planner_begin_target_pos",
        "self._planner_begin_target_vel",
        "self._planner_begin_target_normal",
        "deadline_delta_from_begin",
        "deadline_delta_from_visible",
    ):
        assert token in submit
    accepted = submit.split("if len(accepted_ids) > 0:", 1)[1]
    assert "_planner_begin_target" not in accepted
    assert "_planner_begin_tts" not in accepted

    racket = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    ).read_text()
    proposal = racket.split("def _revise_same_ball_actor_tuple", 1)[1].split(
        "def _compute_strike_timing", 1
    )[0]
    assert "proposal - self._planner_visible_pos" not in proposal
    assert "proposal - self._planner_visible_vel" not in proposal
    assert "begin_pos" in proposal
    assert "begin_vel" in proposal
    assert "deadline_jitter = deadline_jitter.clamp" in proposal


def test_training_phase_clock_is_monotonic_and_velocity_uses_actual_delta():
    source = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    ).read_text()
    advance = source.split("def _advance_planner_phase", 1)[1].split(
        "def _install_event_motion", 1
    )[0]
    assert "frame_delta.clamp(min=0.0)" in advance
    assert "torch.minimum(frame_delta, remaining_frames)" in advance
    assert "earliest = self._planner_minimum_finish_time" in advance
    assert "remaining_deadline <= earliest + dt" in advance
    assert "self.speed_scale = torch.where" in source
    assert "jv = jv * self.speed_scale[:, None]" in source
    assert "v = v * self.speed_scale[:, None, None]" in source
    begin = source.split("def begin_planner_task", 1)[1].split(
        "def _planner_minimum_finish_time", 1
    )[0]
    assert "minimum_tts = self._planner_minimum_finish_time" in begin
    assert "tts + profile.early_deadline_tolerance_s >= minimum_tts" in begin


def test_runtime_metadata_and_training_noise_are_separate_contracts():
    train = (ROOT / "scripts/train.py").read_text()
    commands = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
    ).read_text()
    hard = commands.split("def planner_revision_hard_contract", 1)[1].split(
        "def planner_revision_training_hard_contract", 1
    )[0]
    for key in (
        '"enabled"',
        '"revision_schema_version"',
        '"governor"',
        '"initial_tts_range_s"',
    ):
        assert key in hard
    assert "initial_tts_mixture" not in hard
    training_hard = commands.split(
        "def planner_revision_training_hard_contract", 1
    )[1].split("def begin_planner_task", 1)[0]
    assert 'return {"initial_tts_mixture": mixture.document()}' in training_hard
    assert '"planner_task_revision_training"' in train
    assert '"initial_tts_mixture"' in train
    assert 'planner_training_contract[\n                        "initial_tts_mixture"' in train
    assert 'attr(\n                        motion, "planner_revision_initial_tts_mixture"' not in train
    run = train.split("def _run(cfg):", 1)[1]
    assert run.index("validate_schema3_contract_structure(hard_contract)") < run.index(
        'contract_path = os.path.join(log_dir, "params", "training_contract.json")'
    )
    assert "explicit_weighted_mixture_over_initial_tts_range_s" in train
    assert '"truth_fields_immutable"' in train


def test_revision_changes_visible_tuple_but_never_latent_truth_or_phase():
    profile, ledger = _begin()
    assert ledger.active is not None
    truth_before = ledger.active.truth
    phase_before = ledger.active.phase
    revision = _revision(
        task_revision=2,
        command_sequence=101,
        source_monotonic_s=5.1,
        position=(0.24, -0.1, 0.9),
        velocity=(-1.8, 0.0, -0.2),
        normal=(math.cos(0.1), math.sin(0.1), 0.0),
        tts=0.75,
    )
    decision = P.revise_task(profile, ledger, revision)
    assert decision.accepted, decision.reason
    assert decision.ledger.active is not None
    assert decision.ledger.active.truth is truth_before
    assert decision.ledger.active.phase == phase_before
    assert decision.ledger.active.visible_revision == revision


@pytest.mark.parametrize(
    "revision, match",
    [
        (_revision(task_revision=1, command_sequence=101, source_monotonic_s=5.1), "task_revision"),
        (_revision(task_revision=2, command_sequence=100, source_monotonic_s=5.1), "command_sequence"),
        (_revision(task_revision=2, command_sequence=101, source_monotonic_s=5.0), "source_monotonic"),
        (_revision(task_id=11, task_revision=2, command_sequence=101, source_monotonic_s=5.1), "replace"),
        (
            _revision(control_epoch=8, task_revision=2, command_sequence=101, source_monotonic_s=5.1),
            "replace",
        ),
        (
            _revision(
                task_revision=2,
                command_sequence=101,
                source_monotonic_s=5.1,
                digest="b" * 64,
            ),
            "truth",
        ),
    ],
)
def test_revision_identity_sequence_and_source_are_strict(revision, match):
    profile, ledger = _begin()
    decision = P.revise_task(profile, ledger, revision)
    assert not decision.accepted
    assert match in decision.reason
    assert decision.ledger is ledger


def test_target_tuple_envelope_rejection_is_all_or_nothing():
    profile, ledger = _begin()
    assert ledger.active is not None
    old_visible = ledger.active.visible_revision
    candidates = [
        _revision(
            task_revision=2,
            command_sequence=101,
            source_monotonic_s=5.1,
            position=(0.31, -0.1, 0.9),
        ),
        _revision(
            task_revision=2,
            command_sequence=101,
            source_monotonic_s=5.1,
            velocity=(-1.0, 0.0, -0.2),
        ),
        _revision(
            task_revision=2,
            command_sequence=101,
            source_monotonic_s=5.1,
            normal=(math.cos(0.21), math.sin(0.21), 0.0),
        ),
    ]
    for candidate in candidates:
        decision = P.revise_task(profile, ledger, candidate)
        assert not decision.accepted
        assert decision.ledger is ledger
        assert decision.ledger.active.visible_revision == old_visible


def test_task_baseline_envelope_accepts_latest_value_jumps_but_rejects_ratchet():
    profile, ledger = _begin()

    # Revision 2 moves to one edge of the immutable task-wide training ball.
    edge_a = _revision(
        task_revision=2,
        command_sequence=101,
        source_monotonic_s=5.1,
        position=(0.285, -0.1, 0.9),
        velocity=(-1.55, 0.0, -0.2),
        normal=(math.cos(0.19), math.sin(0.19), 0.0),
    )
    first = P.revise_task(profile, ledger, edge_a)
    assert first.accepted, first.reason

    # A latest-value subscriber may skip revision 3 and next see revision 4 at
    # the opposite edge.  The visible-to-visible jump exceeds every per-step
    # gate, but the complete tuple remains inside the begin-task baseline ball.
    edge_b = _revision(
        task_revision=4,
        command_sequence=103,
        source_monotonic_s=5.3,
        position=(0.115, -0.1, 0.9),
        velocity=(-2.45, 0.0, -0.2),
        normal=(math.cos(-0.19), math.sin(-0.19), 0.0),
    )
    second = P.revise_task(profile, first.ledger, edge_b)
    assert second.accepted, second.reason
    assert second.ledger.active is not None
    assert second.ledger.active.baseline_revision == ledger.active.baseline_revision
    assert second.ledger.active.visible_revision == edge_b

    # Conversely, a sequence of individually small visible jumps cannot ratchet
    # the actor outside that immutable task-wide ball.
    outside = _revision(
        task_revision=5,
        command_sequence=104,
        source_monotonic_s=5.4,
        position=(0.03, -0.1, 0.9),
    )
    rejected = P.revise_task(profile, second.ledger, outside)
    assert not rejected.accepted
    assert "position revision" in rejected.reason
    assert rejected.ledger is second.ledger


def test_deadline_envelope_uses_begin_baseline_but_slow_only_uses_visible_deadline():
    profile, ledger = _begin()
    for _ in range(10):
        ledger = P.advance_phase(profile, ledger)
    assert ledger.active is not None
    baseline_deadline = ledger.active.baseline_deadline_local_s

    later = _revision(
        task_revision=2,
        command_sequence=101,
        source_monotonic_s=5.1,
        tts=baseline_deadline + 0.20 - ledger.active.local_monotonic_s,
    )
    accepted_later = P.revise_task(profile, ledger, later)
    assert accepted_later.accepted, accepted_later.reason
    assert accepted_later.ledger.active is not None
    assert accepted_later.ledger.active.slow_only_next_step

    # Skip revision 3 and cross to the opposite edge.  The absolute deadline is
    # still inside the begin-task ball even though it moved 0.40 s from the last
    # visible deadline; because this change is earlier, it must not set slow-only.
    earlier = _revision(
        task_revision=4,
        command_sequence=103,
        source_monotonic_s=5.3,
        tts=baseline_deadline - 0.20 - ledger.active.local_monotonic_s,
    )
    accepted_earlier = P.revise_task(profile, accepted_later.ledger, earlier)
    assert accepted_earlier.accepted, accepted_earlier.reason
    assert accepted_earlier.ledger.active is not None
    assert not accepted_earlier.ledger.active.slow_only_next_step

    outside = _revision(
        task_revision=5,
        command_sequence=104,
        source_monotonic_s=5.4,
        tts=baseline_deadline - 0.26 - ledger.active.local_monotonic_s,
    )
    rejected = P.revise_task(profile, accepted_earlier.ledger, outside)
    assert not rejected.accepted
    assert "deadline revision" in rejected.reason


def test_phase_is_monotonic_and_rate_and_acceleration_are_bounded():
    profile, ledger = _begin()
    phases = []
    rates = []
    for _ in range(150):
        ledger = P.advance_phase(profile, ledger)
        assert ledger.active is not None
        phases.append(ledger.active.phase)
        rates.append(ledger.active.phase_rate_per_s)
    assert phases == sorted(phases)
    assert phases[-1] == pytest.approx(1.0)
    assert all(0.0 <= rate <= profile.max_phase_rate_per_s for rate in rates)
    # Same length by construction; plain zip keeps the test runnable on Python 3.8.
    for previous, current in zip([0.0, *rates[:-1]], rates):
        assert abs(current - previous) <= (
            profile.max_phase_acceleration_per_s2 * profile.policy_dt_s + 1e-12
        )


@pytest.mark.parametrize(
    "tts_s, checkpoints",
    [
        (
            0.4,
            {
                1: (0.004, 0.4),
                10: (0.3895454545454533, 3.3909090909090445),
                13: (0.5970261994949448, 3.791445707070643),
                19: (0.92, 4.0),
            },
        ),
        (
            0.5,
            {
                1: (0.004, 0.4),
                20: (0.809682744022909, 2.378965699713386),
                21: (0.8612620580171767, 2.778965699713386),
                24: (0.9244206860057322, 3.9789656997133855),
            },
        ),
    ],
)
def test_urgent_deadline_golden_trace_matches_cpp_contract(tts_s, checkpoints):
    profile, ledger = _begin()
    assert ledger.active is not None
    initial = replace(ledger.active.visible_revision, desired_tts_s=tts_s)
    decision = P.begin_task(
        profile,
        P.PhaseGovernorLedger(),
        ledger.active.truth,
        initial,
        local_monotonic_s=20.0,
    )
    assert decision.accepted, decision.reason
    trace = {}
    current = decision.ledger
    for tick in range(1, max(checkpoints) + 1):
        current = P.advance_phase(profile, current)
        assert current.active is not None
        if tick in checkpoints:
            trace[tick] = (current.active.phase, current.active.phase_rate_per_s)
    for tick, expected in checkpoints.items():
        assert trace[tick] == pytest.approx(expected, abs=1e-12, rel=0.0)


def test_later_deadline_can_only_hold_or_slow_the_next_step():
    profile, ledger = _begin()
    for _ in range(10):
        ledger = P.advance_phase(profile, ledger)
    assert ledger.active is not None
    old_rate = ledger.active.phase_rate_per_s
    old_deadline = ledger.active.deadline_local_s
    remaining_old = old_deadline - ledger.active.local_monotonic_s
    delayed = _revision(
        task_revision=2,
        command_sequence=101,
        source_monotonic_s=5.1,
        tts=remaining_old + 0.10,
    )
    decision = P.revise_task(profile, ledger, delayed)
    assert decision.accepted, decision.reason
    advanced = P.advance_phase(profile, decision.ledger)
    assert advanced.active is not None
    assert advanced.active.phase_rate_per_s <= old_rate
    assert advanced.active.phase >= ledger.active.phase


def test_final_policy_interval_accepts_revision_then_post_contact_rejects():
    profile = _profile(min_tts_s=0.02, early_deadline_tolerance_s=1.0e-6)
    decision = P.begin_task(
        profile,
        P.PhaseGovernorLedger(),
        _truth(),
        _revision(tts=0.5),
        local_monotonic_s=20.0,
    )
    assert decision.accepted, decision.reason
    ledger = decision.ledger
    revision_number = 1
    accepted_tts = []
    float32_truth_tts = struct.unpack("f", struct.pack("f", 0.5))[0]
    float32_dt = struct.unpack("f", struct.pack("f", profile.policy_dt_s))[0]
    for tick in range(1, 25):
        ledger = P.advance_phase(profile, ledger)
        float32_truth_tts = struct.unpack(
            "f", struct.pack("f", float32_truth_tts - float32_dt)
        )[0]
        if tick >= 20:
            revision_number += 1
            tts = float32_truth_tts
            revised = P.revise_task(
                profile,
                ledger,
                _revision(
                    task_revision=revision_number,
                    command_sequence=99 + revision_number,
                    source_monotonic_s=5.0 + 0.1 * revision_number,
                    tts=tts,
                ),
            )
            assert revised.accepted, revised.reason
            accepted_tts.append(tts)
            ledger = revised.ledger

    assert accepted_tts == pytest.approx([0.10, 0.08, 0.06, 0.04, 0.02], abs=1.0e-6)
    assert ledger.active is not None
    assert ledger.active.phase < ledger.active.truth.strike_phase
    assert ledger.active.visible_revision.desired_tts_s == 0.02

    ledger = P.advance_phase(profile, ledger)
    assert ledger.active is not None
    assert ledger.active.phase == ledger.active.truth.strike_phase
    rejected = P.revise_task(
        profile,
        ledger,
        _revision(
            task_revision=revision_number + 1,
            command_sequence=100 + revision_number,
            source_monotonic_s=5.1 + 0.1 * revision_number,
            tts=0.02,
        ),
    )
    assert not rejected.accepted
    assert "post-contact" in rejected.reason
    assert rejected.ledger is ledger


def test_reachable_earlier_deadline_is_accepted_inside_all_envelopes():
    profile, ledger = _begin()
    for _ in range(10):
        ledger = P.advance_phase(profile, ledger)
    assert ledger.active is not None
    remaining_old = ledger.active.deadline_local_s - ledger.active.local_monotonic_s
    revision = _revision(
        task_revision=2,
        command_sequence=101,
        source_monotonic_s=5.1,
        tts=remaining_old - 0.05,
    )
    decision = P.revise_task(profile, ledger, revision)
    assert decision.accepted, decision.reason
    assert decision.ledger.active is not None
    assert decision.ledger.active.deadline_local_s < ledger.active.deadline_local_s


def test_unreachable_early_deadline_rejects_and_holds_last_revision():
    profile, ledger = _begin(_profile(max_deadline_revision_delta_s=1.0))
    for _ in range(5):
        ledger = P.advance_phase(profile, ledger)
    assert ledger.active is not None
    old_active = ledger.active
    impossible = _revision(
        task_revision=2,
        command_sequence=101,
        source_monotonic_s=5.1,
        tts=profile.min_tts_s,
    )
    decision = P.revise_task(profile, ledger, impossible)
    assert not decision.accepted
    assert "reachable phase envelope" in decision.reason
    assert decision.ledger is ledger
    advanced = P.advance_phase(profile, decision.ledger)
    assert advanced.active is not None
    assert advanced.active.visible_revision == old_active.visible_revision
    assert advanced.active.deadline_local_s == old_active.deadline_local_s
    assert advanced.active.phase >= old_active.phase


def test_deadline_delta_envelope_rejects_even_when_absolute_tts_is_valid():
    profile, ledger = _begin()
    assert ledger.active is not None
    revision = _revision(
        task_revision=2,
        command_sequence=101,
        source_monotonic_s=5.1,
        tts=ledger.active.visible_revision.desired_tts_s + 0.30,
    )
    decision = P.revise_task(profile, ledger, revision)
    assert not decision.accepted
    assert "deadline revision" in decision.reason
    assert decision.ledger is ledger


def test_task_id_must_increase_after_explicit_completion_and_rearm():
    profile, ledger = _begin()
    completed = P.complete_task(ledger)
    stale = P.begin_task(
        profile,
        completed,
        _truth(task_id=10),
        _revision(task_id=10, task_revision=5, command_sequence=101, source_monotonic_s=5.1),
        local_monotonic_s=21.0,
    )
    assert not stale.accepted
    assert stale.ledger is completed

    next_truth = _truth(task_id=11, digest="b" * 64)
    next_revision = _revision(
        task_id=11,
        task_revision=1,
        command_sequence=101,
        source_monotonic_s=5.1,
        digest="b" * 64,
    )
    accepted = P.begin_task(
        profile,
        completed,
        next_truth,
        next_revision,
        local_monotonic_s=21.0,
    )
    assert accepted.accepted, accepted.reason
    assert accepted.ledger.last_control_epoch == 7
    assert accepted.ledger.last_task_id == 11
    assert accepted.ledger.active.truth == next_truth


def test_initial_infeasible_deadline_and_non_increasing_global_sequence_do_not_start_task():
    profile = _profile(max_phase_rate_per_s=1.0, max_phase_acceleration_per_s2=1.0)
    empty = P.PhaseGovernorLedger()
    infeasible = P.begin_task(
        profile,
        empty,
        _truth(),
        _revision(tts=0.5),
        local_monotonic_s=20.0,
    )
    assert not infeasible.accepted
    assert infeasible.ledger is empty
    assert "reachable phase envelope" in infeasible.reason

    prior = P.PhaseGovernorLedger(
        last_control_epoch=7,
        last_task_id=9,
        last_command_sequence=100,
        last_source_monotonic_s=5.0,
    )
    duplicate_sequence = P.begin_task(
        _profile(),
        prior,
        _truth(task_id=10),
        _revision(command_sequence=100, source_monotonic_s=5.1),
        local_monotonic_s=20.0,
    )
    assert not duplicate_sequence.accepted
    assert duplicate_sequence.ledger is prior


def test_new_control_epoch_rearms_identity_and_ordering_domains_but_stale_epoch_fails():
    profile, ledger = _begin()
    completed = P.complete_task(ledger)
    restarted_truth = _truth(control_epoch=8, task_id=1, digest="b" * 64)
    restarted_revision = _revision(
        control_epoch=8,
        task_id=1,
        task_revision=1,
        command_sequence=1,
        source_monotonic_s=0.0,
        digest="b" * 64,
    )
    restarted = P.begin_task(
        profile,
        completed,
        restarted_truth,
        restarted_revision,
        local_monotonic_s=0.0,
    )
    assert restarted.accepted, restarted.reason
    assert restarted.ledger.last_control_epoch == 8
    assert restarted.ledger.last_task_id == 1
    assert restarted.ledger.last_command_sequence == 1

    rearmed = P.complete_task(restarted.ledger)
    stale = P.begin_task(
        profile,
        rearmed,
        _truth(control_epoch=7, task_id=100, digest="c" * 64),
        _revision(
            control_epoch=7,
            task_id=100,
            task_revision=1,
            command_sequence=999,
            source_monotonic_s=999.0,
            digest="c" * 64,
        ),
        local_monotonic_s=1.0,
    )
    assert not stale.accepted
    assert "control_epoch" in stale.reason
    assert stale.ledger is rearmed
