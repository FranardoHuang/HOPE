from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = _module(SCRIPTS / "isaac_timing_exam_adapter.py", "isaac_timing_exam_adapter_tested")
M = _module(
    REPO / "scripts/materialize_phase1_timing_exam_0p5.py",
    "timing_exam_materializer_for_adapter_test",
)
SPEC_CONFIG = REPO / "configs/phase1_timing_exam_0p5_k100_20260716.json"


def _fixture(tmp_path: Path):
    items = []
    for index in range(100):
        clip = index % 2
        side = A.SIDE_ORDER[clip]
        items.append(
            SimpleNamespace(
                schedule_index=index,
                clip=clip,
                bank_row=index // 2,
                question_id=f"{side}:{A.sha256_bytes(f'q:{index}'.encode())}",
                hold_steps=(index * 7) % 101,
                attempt_seed=5000 + index,
                repeat=0,
            )
        )
    schedule_payload = {
        "artifact_type": "bank-exam-schedule",
        "bank_schema_version": 3,
        "bank_sha256": "b" * 64,
        "clip_order": list(A.SIDE_ORDER),
        "question_counts": [50, 50],
        "per_clip_quota": 50,
        "schedule_seed": 0,
        "hold_range": [0, 100],
        "hold_semantics": "stand-policy-actions-then-raw-frame0-v1",
        "no_wrap": True,
        "items": [vars(item) for item in items],
        "schema_version": 3,
    }
    schedule_payload["schedule_sha256"] = A.canonical_sha256(schedule_payload)
    schedule_path = tmp_path / "source.schedule.json"
    schedule_path.write_bytes(A.canonical_json_bytes(schedule_payload) + b"\n")
    schedule = SimpleNamespace(
        items=tuple(items),
        schedule_sha256=schedule_payload["schedule_sha256"],
        bank_sha256=schedule_payload["bank_sha256"],
    )
    spec = json.loads(SPEC_CONFIG.read_text(encoding="utf-8"))
    spec["source_schedule"] = {
        "path": "/workspace/test/source.schedule.json",
        "bytes": schedule_path.stat().st_size,
        "file_sha256": A.sha256_file(schedule_path),
        "semantic_sha256": schedule.schedule_sha256,
        "question_id_order_sha256": A.canonical_sha256(
            [item.question_id for item in items]
        ),
        "bank_sha256": schedule.bank_sha256,
        "bank_source_family_sha256": "c" * 64,
        "scheduled_attempts": 100,
        "per_side": {"forehand": 50, "backhand": 50},
    }
    paper = M.build_paper(
        spec=spec,
        spec_file_sha256="d" * 64,
        source_schedule={"items": [vars(item) for item in items]},
    )
    paper_path = tmp_path / "timing.paper.json"
    paper_path.write_bytes(A.canonical_json_bytes(paper) + b"\n")
    loaded = A.load_timing_paper(
        paper_path,
        expected_file_sha256=A.sha256_file(paper_path),
        expected_semantic_sha256=paper["paper_semantic_sha256"],
    )
    return loaded, paper_path, schedule_path, schedule


def test_exact_paper_consumes_every_question_and_binds_source_schedule(tmp_path: Path):
    paper, _, schedule_path, schedule = _fixture(tmp_path)
    A.validate_paper_schedule_binding(
        paper,
        schedule_artifact=schedule,
        schedule_path=schedule_path,
    )
    assert len(paper["rows"]) == 100
    assert all(row["tts_ticks"] == 25 for row in paper["rows"])
    assert all(row["initial_state_id"] == A.INITIAL_STATE_ID for row in paper["rows"])
    assert {row["time_law_id"] for row in paper["rows"]} == {
        "v4rg-uniform-phase-forehand-0p5-v1",
        "v4rg-uniform-phase-backhand-0p5-v1",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["rows"][0].__setitem__("tts_ticks", 26), "semantic SHA"),
        (
            lambda value: value["paper"]["time_laws"][0].__setitem__(
                "topp_or_dynamics_certified", True
            ),
            "semantic SHA",
        ),
        (lambda value: value["scoring"].__setitem__("per_side_pass_count", 30), "semantic SHA"),
    ],
)
def test_timing_paper_mutations_fail_closed(tmp_path: Path, mutation, message):
    paper, _, _, _ = _fixture(tmp_path)
    paper.pop("_validated_binding")
    mutation(paper)
    with pytest.raises(A.IsaacBankExamError, match=message):
        A.validate_timing_paper_document(paper)


def test_schedule_reorder_or_file_change_fails_before_runtime_activation(tmp_path: Path):
    paper, _, schedule_path, schedule = _fixture(tmp_path)
    schedule.items = tuple(reversed(schedule.items))
    with pytest.raises(A.IsaacBankExamError, match="question-order|timing row"):
        A.validate_paper_schedule_binding(
            paper,
            schedule_artifact=schedule,
            schedule_path=schedule_path,
        )
    schedule_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(A.IsaacBankExamError, match="file SHA mismatch"):
        A.validate_paper_schedule_binding(
            paper,
            schedule_artifact=schedule,
            schedule_path=schedule_path,
        )


def test_runtime_native_contact_ticks_must_match_paper(tmp_path: Path):
    paper, _, _, _ = _fixture(tmp_path)
    assert A.validate_runtime_time_laws(
        paper,
        segment_lengths=(141, 136),
        strike_phases=(0.47, 1.0 / 3.0),
    ) == (2.64, 1.8)
    with pytest.raises(A.IsaacBankExamError, match="runtime native contact tick"):
        A.validate_runtime_time_laws(
            paper,
            segment_lengths=(139, 136),
            strike_phases=(0.47, 1.0 / 3.0),
        )


class FakeTensor:
    def __init__(self, value, *, dtype=None):
        self.value = np.asarray(value, dtype=dtype)
        self.dtype = self.value.dtype

    def reshape(self, *shape):
        return FakeTensor(self.value.reshape(*shape))

    def __len__(self):
        return len(self.value)

    @property
    def shape(self):
        return self.value.shape

    def __getitem__(self, key):
        if isinstance(key, FakeTensor):
            key = key.value
        return FakeTensor(self.value[key])

    def __setitem__(self, key, value):
        if isinstance(key, FakeTensor):
            key = key.value
        if isinstance(value, FakeTensor):
            value = value.value
        self.value[key] = value

    def __ne__(self, other):
        if isinstance(other, FakeTensor):
            other = other.value
        return FakeTensor(self.value != other)

    def __eq__(self, other):
        if isinstance(other, FakeTensor):
            other = other.value
        return FakeTensor(self.value == other)

    def any(self):
        return bool(self.value.any())

    def all(self):
        return bool(self.value.all())

    def float(self):
        return FakeTensor(self.value.astype(np.float32))

    def abs(self):
        return FakeTensor(np.abs(self.value))

    def max(self):
        return self.value.max()

    def numel(self):
        return self.value.size

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device=None, dtype=None):
        return FakeTensor(self.value, dtype=dtype or self.value.dtype)

    def tolist(self):
        return self.value.tolist()


class FakeTorch:
    long = np.int64

    @staticmethod
    def as_tensor(value, *, device=None, dtype=None):
        if isinstance(value, FakeTensor):
            value = value.value
        return FakeTensor(value, dtype=dtype)

    @staticmethod
    def unique(value):
        return FakeTensor(np.unique(value.value))

    @staticmethod
    def isfinite(value):
        return FakeTensor(np.isfinite(value.value))

    @staticmethod
    def equal(left, right):
        return bool(np.array_equal(left.value, right.value))


class FakeMotionCommand(SimpleNamespace):
    @property
    def joint_pos(self):
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self):
        value = self.motion.joint_vel[self.time_steps]
        return FakeTensor(value.value * self.speed_scale.value[:, None])

    @property
    def body_lin_vel_w(self):
        value = self.motion.body_lin_vel_w[self.time_steps]
        return FakeTensor(value.value * self.speed_scale.value[:, None, None])

    @property
    def body_ang_vel_w(self):
        value = self.motion.body_ang_vel_w[self.time_steps]
        return FakeTensor(value.value * self.speed_scale.value[:, None, None])


def test_native_installer_then_r14_rider_consumes_per_row_time_law(tmp_path: Path):
    paper, _, _, _ = _fixture(tmp_path)
    clips = np.asarray([index % 2 for index in range(100)], dtype=np.int64)
    starts_table = FakeTensor([0, 141], dtype=np.int64)
    starts = starts_table[FakeTensor(clips)]
    segment_joint_pos = np.arange(276 * 3, dtype=np.float32).reshape(276, 3)
    segment_joint_vel = np.ones((276, 3), dtype=np.float32)
    segment_body_lin_vel = np.ones((276, 2, 3), dtype=np.float32)
    segment_body_ang_vel = np.ones((276, 2, 3), dtype=np.float32)
    motion = FakeMotionCommand(
        device="cpu",
        cfg=SimpleNamespace(speed_scale_range=(1.0, 1.0), speed_scale_per_clip=None),
        planner_revision_enabled=False,
        retiming_active=False,
        _speed_per_clip=None,
        motion=SimpleNamespace(
            seg_start=starts_table,
            joint_pos=FakeTensor(segment_joint_pos),
            joint_vel=FakeTensor(segment_joint_vel),
            body_lin_vel_w=FakeTensor(segment_body_lin_vel),
            body_ang_vel_w=FakeTensor(segment_body_ang_vel),
        ),
        clip_id=FakeTensor(clips.copy()),
        time_steps=FakeTensor(starts.value.copy()),
        time_steps_f=FakeTensor(starts.value.astype(np.float32)),
        speed_scale=FakeTensor(np.ones(100, dtype=np.float32)),
        hold_counter=FakeTensor(np.zeros(100, dtype=np.int64)),
        just_resampled=FakeTensor(np.zeros(100, dtype=bool)),
        in_hold=FakeTensor(np.zeros(100, dtype=bool)),
    )
    ready = A.install_zero_velocity_frame0_reference(
        motion,
        env_ids=np.arange(100),
        clip_ids=clips,
        paper=paper,
        torch_module=FakeTorch,
    )
    assert ready["reference_pose_is_exact_motion_frame0"] is True
    assert ready["live_reference_velocity_max_abs_after_override"] == {
        "joint_vel": 0.0,
        "body_lin_vel_w": 0.0,
        "body_ang_vel_w": 0.0,
    }
    profile = A.activate_runtime_retiming(
        motion,
        env_ids=np.arange(100),
        clip_ids=clips,
        paper=paper,
        segment_lengths=(141, 136),
        strike_phases=(0.47, 1.0 / 3.0),
        torch_module=FakeTorch,
    )
    assert motion.retiming_active is True
    np.testing.assert_allclose(
        motion.speed_scale.value,
        np.asarray([2.64 if clip == 0 else 1.8 for clip in clips]),
    )
    assert profile["native_external_installer_ran_first"] is True
    assert profile["native_zero_velocity_frame0_verified_before_activation"] is True
    assert profile["play_or_export_path_used"] is False
    final = A.verify_runtime_retiming_preserved(
        motion,
        paper=paper,
        expected_profile=profile,
        torch_module=FakeTorch,
    )
    assert final["preserved_through_finalization"] is True


def test_retiming_cannot_activate_before_zero_velocity_frame0_install(tmp_path: Path):
    paper, _, _, _ = _fixture(tmp_path)
    clips = np.asarray([index % 2 for index in range(100)], dtype=np.int64)
    starts_table = FakeTensor([0, 141], dtype=np.int64)
    starts = starts_table[FakeTensor(clips)]
    motion = FakeMotionCommand(
        device="cpu",
        cfg=SimpleNamespace(speed_scale_range=(1.0, 1.0), speed_scale_per_clip=None),
        planner_revision_enabled=False,
        retiming_active=False,
        _speed_per_clip=None,
        motion=SimpleNamespace(
            seg_start=starts_table,
            joint_pos=FakeTensor(np.zeros((276, 3), dtype=np.float32)),
            joint_vel=FakeTensor(np.ones((276, 3), dtype=np.float32)),
            body_lin_vel_w=FakeTensor(np.ones((276, 2, 3), dtype=np.float32)),
            body_ang_vel_w=FakeTensor(np.ones((276, 2, 3), dtype=np.float32)),
        ),
        clip_id=FakeTensor(clips.copy()),
        time_steps=FakeTensor(starts.value.copy()),
        time_steps_f=FakeTensor(starts.value.astype(np.float32)),
        speed_scale=FakeTensor(np.ones(100, dtype=np.float32)),
        hold_counter=FakeTensor(np.zeros(100, dtype=np.int64)),
        just_resampled=FakeTensor(np.zeros(100, dtype=bool)),
        in_hold=FakeTensor(np.zeros(100, dtype=bool)),
    )
    with pytest.raises(A.IsaacBankExamError, match="zero-velocity frame-0 command"):
        A.activate_runtime_retiming(
            motion,
            env_ids=np.arange(100),
            clip_ids=clips,
            paper=paper,
            segment_lengths=(141, 136),
            strike_phases=(0.47, 1.0 / 3.0),
            torch_module=FakeTorch,
        )
    assert motion.retiming_active is False
    assert motion._speed_per_clip is None


def test_saved_planner_clock_is_disabled_before_gym_make_and_unknown_owner_rejected():
    motion = SimpleNamespace(
        speed_scale_range=(1.0, 1.0),
        speed_scale_per_clip=None,
        event_timing_mode="disabled",
        planner_revision_enabled=True,
    )
    env_cfg = SimpleNamespace(commands=SimpleNamespace(motion=motion))
    profile = A.apply_timing_native_clock_eval_profile(env_cfg)
    assert profile["saved_planner_revision_enabled"] is True
    assert profile["runtime_planner_revision_enabled"] is False
    assert motion.planner_revision_enabled is False

    motion.event_timing_mode = "post_strike_t1"
    with pytest.raises(A.IsaacBankExamError, match="event-timing clock owner"):
        A.apply_timing_native_clock_eval_profile(env_cfg)


def test_frame0_requires_exact_zero_velocity(tmp_path: Path):
    paper, _, _, _ = _fixture(tmp_path)
    root = np.zeros((100, 13), dtype=np.float64)
    root[:, 3] = 1.0
    qd = np.zeros((100, 31), dtype=np.float64)
    profile = A.validate_zero_velocity_ready_state(
        paper,
        root_states=root,
        joint_velocities=qd,
    )
    assert profile["exact_zero_velocity"] is True
    qd[3, 5] = 1e-12
    with pytest.raises(A.IsaacBankExamError, match="must be zero velocity"):
        A.validate_zero_velocity_ready_state(
            paper,
            root_states=root,
            joint_velocities=qd,
        )


def _records(paper: dict, *, successes: int = 31):
    used = Counter({side: 0 for side in A.SIDE_ORDER})
    records = []
    for row in paper["rows"]:
        success = used[row["side"]] < successes
        used[row["side"]] += int(success)
        record = {
            "schedule_index": row["schedule_index"],
            "question_id": row["question_id"],
            "side": row["side"],
            "bank_row": row["bank_row"],
            "attempt_seed": row["attempt_seed"],
            "repeat": row["repeat"],
            "finalized": True,
            "censored": False,
            "reached_exact": success,
            "hit": success,
            "returned": success,
            "pos_error_m": 0.01 if success else None,
            "vel_error_mps": 0.1 if success else None,
            "normal_error_deg": 2.0 if success else None,
            "physical_fall": False,
            "guard_reset": False,
        }
        A.initialize_timing_record(record, row)
        if success:
            record["exact_strike_step"] = 25
        records.append(record)
    return records


def test_per_attempt_timing_ledger_and_all_attempt_summary(tmp_path: Path):
    paper, _, _, _ = _fixture(tmp_path)
    records = _records(paper)
    A.observe_timing_deadlines(records, step=25)
    summary = A.finalize_timing_records(
        records,
        paper=paper,
        evaluation_contract_exact=False,
    )
    assert summary["aggregate_denominator"] == 100
    assert summary["per_side"]["forehand"]["scheduled"] == 50
    assert summary["per_side"]["backhand"]["scheduled"] == 50
    assert summary["per_side"]["forehand"]["composite"] == 31
    assert summary["per_side"]["backhand"]["composite"] == 31
    assert summary["diagnostic_performance_pass"] is True
    assert summary["safety_observation_complete"] is False
    assert summary["formal_gate_pass"] is False
    assert all("eligible" in row and "deadline_miss" in row for row in records)
    assert all("infeasible" in row and "deadline_shifted" in row for row in records)
    assert all("contact" in row and "composite" in row and "safety" in row for row in records)


def test_missed_deadline_and_missing_safety_never_become_formal_pass(tmp_path: Path):
    paper, _, _, _ = _fixture(tmp_path)
    records = _records(paper)
    records[0]["reached_exact"] = False
    records[0]["hit"] = False
    records[0]["returned"] = False
    records[0]["pos_error_m"] = None
    records[0]["vel_error_mps"] = None
    records[0]["normal_error_deg"] = None
    records[0]["exact_strike_step"] = None
    A.observe_timing_deadlines(records, step=25)
    summary = A.finalize_timing_records(
        records,
        paper=paper,
        evaluation_contract_exact=True,
    )
    assert records[0]["deadline_miss"] is True
    assert records[0]["composite"] is False
    assert records[0]["safety"]["self_hit"] is None
    assert summary["formal_gate_pass"] is False
    assert set(summary["formal_gate_blockers"]) == set(A.DIAGNOSTIC_REASONS)


def test_early_or_late_exact_strike_is_a_deadline_shift_failure(tmp_path: Path):
    paper, _, _, _ = _fixture(tmp_path)
    records = _records(paper)
    records[0]["exact_strike_step"] = 24
    records[1]["exact_strike_step"] = 26
    summary = A.finalize_timing_records(
        records,
        paper=paper,
        evaluation_contract_exact=False,
    )
    assert records[0]["deadline_shifted"] is True
    assert records[1]["deadline_shifted"] is True
    assert records[0]["composite"] is False
    assert records[1]["composite"] is False
    assert summary["per_side"]["forehand"]["deadline_shifted"] == 1
    assert summary["per_side"]["backhand"]["deadline_shifted"] == 1


def test_evaluator_keeps_timing_mode_default_off_and_never_calls_play():
    source = (SCRIPTS / "isaac_bank_exam.py").read_text(encoding="utf-8")
    assert 'timing_paper_raw = str(_cfg(cfg, "timing_paper", "")).strip()' in source
    assert "if timing_paper is not None:" in source
    assert "motion_cmd.install_external_exam_timing(env_ids, clip_ids, holds)" in source
    assert "activate_runtime_retiming(" in source
    assert "apply_timing_native_clock_eval_profile(env_cfg)" in source
    assert source.index("apply_timing_native_clock_eval_profile(env_cfg)") < source.index(
        "env = gym.make"
    )
    assert source.index("motion_cmd.install_external_exam_timing") < source.index(
        'timing_ready_profile["motion_reference"] = install_zero_velocity_frame0_reference'
    ) < source.index("timing_runtime_profile = activate_runtime_retiming")
    assert "play.py" not in source
    assert source.index("policy = runner.get_inference_policy") < source.index(
        "inexact_reasons.extend(TIMING_DIAGNOSTIC_REASONS)"
    )
