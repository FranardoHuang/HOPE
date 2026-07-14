"""Dependency-light invariants for post-strike T1 training timing."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/event_timing.py"
)
SPEC = importlib.util.spec_from_file_location("event_timing_under_test", MODULE_PATH)
ET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ET
SPEC.loader.exec_module(ET)


def _schedule_dict():
    return {
        "schema_version": 1,
        "schedule_id": "unit-post-strike-v1",
        "policy_rate_hz": 50,
        "sequences": [
            {
                "sequence_id": "sequence-feasible-then-unavailable",
                "rows": [
                    {
                        "question_id": "0000000000000001",
                        "clip_id": 0,
                        "bank_row": 0,
                        "reveal_ticks_after_prior_strike": 2,
                        "next_strike_ticks_after_prior_strike": 6,
                    },
                    {
                        "question_id": "0000000000000002",
                        "clip_id": 1,
                        "bank_row": 0,
                        "reveal_ticks_after_prior_strike": 1,
                        "next_strike_ticks_after_prior_strike": 4,
                        "available": False,
                        "unavailable_reason": "materializer_marked_unavailable",
                    },
                ],
            },
            {
                "sequence_id": "sequence-native-infeasible",
                "rows": [
                    {
                        "question_id": "0000000000000003",
                        "clip_id": 0,
                        "bank_row": 1,
                        "reveal_ticks_after_prior_strike": 1,
                        "next_strike_ticks_after_prior_strike": 3,
                    }
                ],
            },
        ],
    }


def _write_schedule(tmp_path: Path, value=None):
    value = _schedule_dict() if value is None else value
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = tmp_path / "event_schedule.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_schedule_is_bound_to_exact_bytes_and_rejects_ambiguous_json(tmp_path):
    path, digest = _write_schedule(tmp_path)
    schedule = ET.load_event_schedule(path, digest)
    assert schedule.source_sha256 == digest
    assert schedule.source_bytes == path.stat().st_size
    assert schedule.hard_contract()["sequence_lengths"] == [2, 1]

    # Semantically identical whitespace is still different immutable paper bytes.
    path.write_bytes(path.read_bytes() + b" \n")
    with pytest.raises(ValueError, match="byte SHA mismatch"):
        ET.load_event_schedule(path, digest)

    duplicate = b'{"schema_version":1,"schema_version":1,"schedule_id":"x","policy_rate_hz":50,"sequences":[]}'
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        ET.load_event_schedule(path, hashlib.sha256(duplicate).hexdigest())


def test_scheduler_never_reveals_before_explicit_exact_strike(tmp_path):
    path, digest = _write_schedule(tmp_path)
    schedule = ET.load_event_schedule(path, digest)
    scheduler = ET.EventTimingScheduler(
        schedule, num_envs=2, device="cpu", sequence_indices=[0, 1]
    )
    for _ in range(20):
        step = scheduler.advance([3, 2])
        assert len(step.install_env_ids) == 0
        assert len(step.unavailable_env_ids) == 0
        assert len(step.infeasible_env_ids) == 0
        assert len(step.deadline_env_ids) == 0
    assert not bool(scheduler.armed.any())

    accepted = scheduler.record_exact_strike(torch.tensor([0, 1], dtype=torch.long))
    assert accepted.tolist() == [0, 1]
    assert scheduler.origin_tick.tolist() == [20, 20]
    assert scheduler.reveal_tick.tolist() == [22, 21]
    assert scheduler.deadline_tick.tolist() == [26, 23]


def test_miss_consumes_and_unavailable_or_infeasible_never_shift_deadline(tmp_path):
    path, digest = _write_schedule(tmp_path)
    scheduler = ET.EventTimingScheduler(
        ET.load_event_schedule(path, digest),
        num_envs=2,
        device="cpu",
        sequence_indices=[0, 1],
    )
    scheduler.record_exact_strike([0, 1])

    # tick 1: sequence 1 reveals, but native strike needs 3 ticks inside a 2-tick notice.
    step = scheduler.advance([3, 2])
    assert step.infeasible_env_ids.tolist() == [1]
    assert len(step.install_env_ids) == 0
    assert scheduler.deadline_tick.tolist() == [6, 3]

    # tick 2: sequence 0 reveals atomically; hold=deadline-reveal-native=6-2-3=1.
    step = scheduler.advance([3, 2])
    assert step.install_env_ids.tolist() == [0]
    assert step.install_schedule_rows.tolist() == [0]
    assert step.install_clip_ids.tolist() == [0]
    assert step.install_bank_rows.tolist() == [0]
    assert step.install_hold_steps.tolist() == [1]
    assert scheduler.deadline_tick.tolist() == [6, 3]

    # tick 3: infeasible opportunity is due at the original tick, not at a replacement time.
    step = scheduler.advance([3, 2])
    assert step.deadline_env_ids.tolist() == [1]
    consumed = scheduler.finalize_deadlines()
    assert consumed.tolist() == [1]
    assert scheduler.opportunities_consumed.tolist() == [0, 1]
    assert scheduler.exhausted.tolist() == [False, True]

    for _ in range(3):
        step = scheduler.advance([3, 2])
    assert step.deadline_env_ids.tolist() == [0]
    # No success flag exists in finalize: a miss unconditionally consumes opportunity 0.
    consumed = scheduler.finalize_deadlines()
    assert consumed.tolist() == [0]
    assert scheduler.opportunities_consumed.tolist() == [1, 1]
    # Row 1 is anchored to the previous scheduled deadline (6), never to hit/install time.
    assert scheduler.origin_tick[0].item() == 6
    assert scheduler.reveal_tick[0].item() == 7
    assert scheduler.deadline_tick[0].item() == 10

    step = scheduler.advance([3, 2])
    assert step.unavailable_env_ids.tolist() == [0]
    assert len(step.install_env_ids) == 0
    assert scheduler.deadline_tick[0].item() == 10
    for _ in range(3):
        step = scheduler.advance([3, 2])
    assert step.deadline_env_ids.tolist() == [0]
    scheduler.finalize_deadlines()
    assert scheduler.opportunities_consumed.tolist() == [2, 1]
    assert scheduler.exhausted.tolist() == [True, True]


def test_due_deadline_cannot_be_silently_extended(tmp_path):
    path, digest = _write_schedule(tmp_path)
    scheduler = ET.EventTimingScheduler(
        ET.load_event_schedule(path, digest), num_envs=1, device="cpu", sequence_indices=[1]
    )
    scheduler.record_exact_strike([0])
    scheduler.advance([3, 2])
    scheduler.advance([3, 2])
    scheduler.advance([3, 2])
    assert scheduler.deadline_due.tolist() == [True]
    with pytest.raises(RuntimeError, match="must be finalized"):
        scheduler.advance([3, 2])


def test_emitted_hold_reaches_native_strike_on_immutable_deadline(tmp_path):
    path, digest = _write_schedule(tmp_path)
    scheduler = ET.EventTimingScheduler(
        ET.load_event_schedule(path, digest), num_envs=1, device="cpu", sequence_indices=[0]
    )
    scheduler.record_exact_strike([0])
    scheduler.advance([3, 2])
    step = scheduler.advance([3, 2])
    hold = step.install_hold_steps.item()
    native_frame = 0
    # MotionCommand installs after the reveal step's old-clock advance.  On each later step it
    # consumes hold first, then advances one native frame when unheld.
    for _tick in range(3, 7):
        held = hold > 0
        hold = max(hold - 1, 0)
        if not held:
            native_frame += 1
    assert native_frame == 3
    assert hold == 0


def _method_calls(path: Path, class_name: str, method_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    calls = set()
                    for child in ast.walk(item):
                        if not isinstance(child, ast.Call):
                            continue
                        target = child.func
                        if isinstance(target, ast.Attribute):
                            calls.add(target.attr)
                        elif isinstance(target, ast.Name):
                            calls.add(target.id)
                    return calls
    raise AssertionError(f"missing {class_name}.{method_name}")


def test_event_install_paths_cannot_reset_or_teleport_carry_state():
    mdp = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    motion_calls = _method_calls(mdp / "commands.py", "MotionCommand", "_install_event_motion")
    racket_calls = _method_calls(
        mdp / "hope_commands.py", "RacketTargetCommand", "_install_event_training_questions"
    )
    forbidden_motion = {
        "_resample_command",
        "_adaptive_sampling",
        "write_root_state_to_sim",
        "write_joint_state_to_sim",
        "reset",
    }
    assert not (motion_calls & forbidden_motion)
    assert "_reset_actor_target_state" not in racket_calls
    assert "_resample_command" not in racket_calls


def test_training_hard_contract_binds_every_existing_timing_knob():
    train = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    for key in (
        "motion_event_timing",
        "motion_clip_switch_prob",
        "motion_post_swing_start_prob",
        "motion_post_swing_buffer_size",
        "motion_post_swing_min_fill",
        "motion_post_swing_min_hold",
        "motion_post_swing_replay",
        "motion_rsi_skip_settle_frames",
        "motion_stagger_initial_clock",
        "motion_stagger_hold_max_steps",
        "racket_strike_phase",
        "racket_strike_window_s",
        "racket_midswing_resample_prob",
        "racket_midswing_resample_tts_floor",
        "racket_target_delay_steps",
        "racket_target_post_strike_dropout_s",
    ):
        assert f'"{key}"' in train
