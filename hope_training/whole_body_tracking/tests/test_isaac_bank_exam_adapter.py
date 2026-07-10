"""CPU-only contract tests for the evaluator-owned Isaac BankExam adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "isaac_bank_exam_adapter.py"
SPEC = importlib.util.spec_from_file_location("isaac_bank_exam_adapter_tested", MODULE)
A = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = A
assert SPEC.loader is not None
SPEC.loader.exec_module(A)

CKPT_MODULE = (
    ROOT / "source" / "whole_body_tracking" / "whole_body_tracking"
    / "utils" / "ckpt_compat.py"
)
CKPT_SPEC = importlib.util.spec_from_file_location("ckpt_compat_tested", CKPT_MODULE)
CKPT = importlib.util.module_from_spec(CKPT_SPEC)
assert CKPT_SPEC.loader is not None
CKPT_SPEC.loader.exec_module(CKPT)


class FakeTensor:
    def __init__(self, value="obs"):
        self.value = value
        self.shape = (2, 3)
        self.device = None

    def to(self, device):
        self.device = device
        return self


def _env_cfg():
    event = SimpleNamespace(mode="startup")
    nominal_event = SimpleNamespace(
        mode="startup",
        params={"pos_distribution_params": (-0.01, 0.01), "operation": "add"},
    )
    return SimpleNamespace(
        scene=SimpleNamespace(num_envs=4096),
        events=SimpleNamespace(
            physics_material=event,
            add_joint_default_pos=nominal_event,
            push_robot=None,
            note="keep",
        ),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
        commands=SimpleNamespace(
            motion=SimpleNamespace(
                stand_start_prob=0.25,
                post_swing_start_prob=0.2,
                stand_start_min_hold=25,
                stand_start_yaw_range=(-0.2, 0.2),
                hold_steps_range=(0, 100),
                wrap_teleport=False,
                clip_switch_prob=0.002,
                stagger_initial_clock=True,
                resampling_time_range=(5.0, 10.0),
                speed_scale_range=(1.0, 1.0),
                speed_scale_per_clip=None,
            ),
            racket_target=SimpleNamespace(
                midswing_resample_prob=0.02,
                achieved_target_mix_prob=0.3,
                target_delay_steps=2,
                target_jitter_pos_per_s=0.1,
                target_jitter_vel_per_s=0.1,
                target_noise_white=0.01,
                target_noise_ar1_sigma=0.02,
                target_dropout_prob=0.1,
                target_post_strike_dropout_s=0.2,
                target_bias_per_swing=0.01,
                shadow_ball=True,
                shadow_table=True,
                physical_ball=True,
                resampling_time_range=(5.0, 10.0),
                question_bank="train_split.npz",
            ),
        ),
    )


def _schedule():
    return tuple(
        SimpleNamespace(
            schedule_index=index,
            clip=index % 2,
            bank_row=index,
            question_id=f"{'forehand' if index % 2 == 0 else 'backhand'}:{index:064x}",
            hold_steps=index + 2,
            attempt_seed=100 + index,
            repeat=0,
        )
        for index in range(4)
    )


def bad_anchor_pos_z_only_hold_aware():
    pass


def bad_anchor_ori_hold_aware():
    pass


def bad_motion_body_pos_z_only_hold_aware():
    pass


HOLD_FUNCTIONS = {
    "anchor_pos": bad_anchor_pos_z_only_hold_aware,
    "anchor_ori": bad_anchor_ori_hold_aware,
    "ee_body_pos": bad_motion_body_pos_z_only_hold_aware,
}


def _terminations(*, aware: bool):
    names = {
        "anchor_pos": ("bad_anchor_pos_z_only", "bad_anchor_pos_z_only_hold_aware"),
        "anchor_ori": ("bad_anchor_ori", "bad_anchor_ori_hold_aware"),
        "ee_body_pos": (
            "bad_motion_body_pos_z_only",
            "bad_motion_body_pos_z_only_hold_aware",
        ),
    }
    return SimpleNamespace(**{
        term: SimpleNamespace(
            func=(HOLD_FUNCTIONS[term] if aware else legacy),
            params={"threshold": 0.25, **({"ignore_hold": True} if aware else {})},
        )
        for term, (legacy, current) in names.items()
    })


def _records(schedule):
    rows = []
    for item in schedule:
        rows.append({
            "schedule_index": item.schedule_index,
            "env_id": item.schedule_index,
            "clip": item.clip,
            "side": "forehand" if item.clip == 0 else "backhand",
            "bank_row": item.bank_row,
            "question_id": item.question_id,
            "repeat": item.repeat,
            "hold_steps": item.hold_steps,
            "attempt_seed": item.attempt_seed,
            "ready_state_sha256": "a" * 64,
            "start_step": 0,
            "end_step": 10,
            "finalize_reason": "clip_complete",
            "finalized": True,
            "censored": False,
            "physical_fall": False,
            "guard_reset": False,
            "reached_exact": True,
            "hit": True,
            "returned": item.schedule_index == 0,
            "pos_error_m": 0.01,
            "vel_error_mps": 0.2,
            "normal_error_deg": float("nan"),
            "landing_x": 2.5,
            "landing_y": 0.0,
            "net_clear": True,
        })
    return rows


def test_observation_adapter_handles_tuple_and_policy_mapping_and_rejects_unknown():
    for raw in (FakeTensor(), (FakeTensor(), {"ignored": True}), {"policy": FakeTensor()}):
        got = A.policy_observation_tensor(raw, device="cuda:0")
        assert isinstance(got, FakeTensor) and got.device == "cuda:0"
    with pytest.raises(A.IsaacBankExamError, match="no 'policy'"):
        A.policy_observation_tensor({"critic": FakeTensor()})


def test_nominal_profile_disables_randomness_but_never_replaces_train_bank():
    cfg = _env_cfg()
    profile = A.apply_nominal_eval_profile(cfg, num_envs=20)
    assert cfg.scene.num_envs == 20
    assert cfg.events.physics_material is None
    assert cfg.events.add_joint_default_pos is not None
    assert cfg.events.add_joint_default_pos.params["pos_distribution_params"] is None
    assert cfg.events.note == "keep"
    assert cfg.observations.policy.enable_corruption is False
    assert cfg.commands.motion.stand_start_prob == 1.0
    assert cfg.commands.motion.hold_steps_range == (0, 0)
    assert cfg.commands.motion.resampling_time_range == (1.0e9, 1.0e9)
    assert cfg.commands.racket_target.midswing_resample_prob == 0.0
    assert cfg.commands.racket_target.target_delay_steps == 0
    assert cfg.commands.racket_target.resampling_time_range == (1.0e9, 1.0e9)
    assert cfg.commands.racket_target.question_bank == "train_split.npz"
    assert profile["exam_bank_injected_into_training_cfg"] is False
    assert len(profile["sha256"]) == 64


def test_nominal_profile_rejects_unversioned_retiming():
    cfg = _env_cfg()
    cfg.commands.motion.speed_scale_range = (0.8, 1.0)
    with pytest.raises(A.IsaacBankExamError, match="native one-frame"):
        A.apply_nominal_eval_profile(cfg, num_envs=20)


def test_hold_aware_guard_profile_is_exact_by_default_and_legacy_override_is_labeled():
    legacy = SimpleNamespace(terminations=_terminations(aware=False))
    with pytest.raises(A.IsaacBankExamError, match="diagnostic-only"):
        A.apply_hold_aware_tracking_guard_profile(
            legacy,
            hold_aware_functions=HOLD_FUNCTIONS,
            allow_legacy_override=False,
        )
    profile = A.apply_hold_aware_tracking_guard_profile(
        legacy,
        hold_aware_functions=HOLD_FUNCTIONS,
        allow_legacy_override=True,
    )
    assert profile["overridden_terms"] == ["anchor_ori", "anchor_pos", "ee_body_pos"]
    assert legacy.terminations.anchor_pos.params["ignore_hold"] is True
    assert legacy.terminations.anchor_pos.func is bad_anchor_pos_z_only_hold_aware

    current = SimpleNamespace(terminations=_terminations(aware=True))
    exact = A.apply_hold_aware_tracking_guard_profile(
        current,
        hold_aware_functions=HOLD_FUNCTIONS,
        allow_legacy_override=False,
    )
    assert exact["overridden_terms"] == []


def test_runtime_train_bank_loader_opens_legacy_only_for_explicit_diagnostic(tmp_path: Path):
    bank = tmp_path / "legacy_train.npz"
    meta = np.frombuffer(json.dumps({"split": "train"}).encode("utf-8"), dtype=np.uint8)
    np.savez(bank, meta_json=meta, incoming=np.zeros((1, 3)))
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(
                question_bank=str(bank), question_bank_allow_legacy=False
            )
        )
    )
    with pytest.raises(A.IsaacBankExamError, match="diagnostic-only"):
        A.configure_runtime_train_bank_loader(cfg, allow_legacy_diagnostic=False)
    result = A.configure_runtime_train_bank_loader(cfg, allow_legacy_diagnostic=True)
    assert result["legacy_override"] is True
    assert result["split"] == "train"
    assert cfg.commands.racket_target.question_bank_allow_legacy is True


def test_runtime_motion_loader_opens_link_origin_signature_only_for_diagnostic(
    tmp_path: Path,
):
    motion_path = tmp_path / "legacy_link_velocity.npz"
    time = np.arange(5, dtype=np.float32) / 50.0
    pos = np.zeros((5, 1, 3), dtype=np.float32)
    pos[:, 0, 0] = 2.0 * time
    lin = np.gradient(pos, 1.0 / 50.0, axis=0)
    ang = np.zeros_like(pos)
    ang[:, 0, 2] = 1.0
    np.savez(
        motion_path,
        fps=np.array([50]),
        body_pos_w=pos,
        body_lin_vel_w=lin,
        body_ang_vel_w=ang,
    )
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            motion=SimpleNamespace(
                motion_file=str(motion_path),
                allow_legacy_link_origin_velocity=False,
            )
        )
    )
    with pytest.raises(A.IsaacBankExamError, match="formal BankExam refuses"):
        A.configure_runtime_motion_loader(cfg, allow_legacy_diagnostic=False)
    result = A.configure_runtime_motion_loader(cfg, allow_legacy_diagnostic=True)
    assert result["legacy_override"] is True
    assert cfg.commands.motion.allow_legacy_link_origin_velocity is True


def test_historical_config_hydration_uses_only_declared_defaults_and_is_inexact():
    @dataclass
    class Config:
        required: int
        later_default: bool = True

    cfg = Config(required=3)
    cfg.__dict__.pop("later_default")
    with pytest.raises(A.IsaacBankExamError, match="missing field cfg.later_default"):
        A.hydrate_missing_dataclass_defaults(
            {"cfg": cfg}, allow_legacy_diagnostic=False
        )
    result = A.hydrate_missing_dataclass_defaults(
        {"cfg": cfg}, allow_legacy_diagnostic=True
    )
    assert cfg.later_default is True
    assert result["hydrated_fields"] == {"cfg": ["later_default"]}


def test_action_noise_is_question_seeded_and_early_failure_cannot_shift_later_rows():
    schedule = _schedule()
    first = A.per_attempt_action_noise(schedule, n_steps=5, action_dim=3)
    second = A.per_attempt_action_noise(schedule, n_steps=5, action_dim=3)
    np.testing.assert_array_equal(first, second)
    # Consuming only one step for attempt 0 has no effect on attempt 1's independent stream.
    np.testing.assert_array_equal(
        first[1], np.random.default_rng(schedule[1].attempt_seed).standard_normal((5, 3))
    )


def test_ready_state_hash_is_origin_independent_after_caller_rebases_and_binds_velocity():
    root = np.arange(13, dtype=float)
    q = np.arange(31, dtype=float)
    qd = np.zeros(31)
    base = A.ready_state_sha256(root, q, qd)
    assert base == A.ready_state_sha256(root.copy(), q.copy(), qd.copy())
    changed = qd.copy(); changed[3] = 1e-6
    assert base != A.ready_state_sha256(root, q, changed)


def test_all_attempt_summary_keeps_physical_and_guard_failures_in_one_denominator():
    schedule = _schedule()
    rows = _records(schedule)
    rows[1].update(
        finalize_reason="physical_fall", physical_fall=True,
        reached_exact=False, hit=False, returned=False,
    )
    rows[2].update(
        finalize_reason="guard_reset", guard_reset=True,
        reached_exact=False, hit=False, returned=False,
    )
    A.validate_complete_ledger(rows, schedule)
    summary = A.summarize_attempts(rows, ("forehand", "backhand"))
    assert summary["n_attempts"] == 4
    assert summary["no_fall_rate"] == 0.75  # guard reset is not a physical fall
    assert summary["hit_rate"] == 0.5
    assert summary["return_rate"] == 0.25
    assert summary["guard_reset_rate"] == 0.25


def test_incomplete_or_censored_cell_fails_closed():
    schedule = _schedule()
    rows = _records(schedule)
    with pytest.raises(A.IsaacBankExamError, match="expected schedule"):
        A.validate_complete_ledger(rows[:-1], schedule)
    rows[2]["censored"] = True
    with pytest.raises(A.IsaacBankExamError, match="whole cell invalid"):
        A.validate_complete_ledger(rows, schedule)


def test_scorecard_json_is_strict_and_csv_has_same_attempt_count(tmp_path: Path):
    schedule = _schedule()
    rows = _records(schedule)
    document = A.write_scorecard(
        output_json=tmp_path / "score.json",
        output_csv=tmp_path / "score.csv",
        metadata={"evaluation_contract_exact": False},
        records=rows,
        schedule=schedule,
        clip_names=("forehand", "backhand"),
    )
    loaded = json.loads((tmp_path / "score.json").read_text())
    assert loaded == document
    assert loaded["attempts"][0]["normal_error_deg"] is None
    assert len((tmp_path / "score.csv").read_text().splitlines()) == len(schedule) + 1


def test_tolerant_checkpoint_loader_rejects_actor_key_missing_from_checkpoint(monkeypatch):
    class Value:
        shape = (2, 2)

    checkpoint = {"model_state_dict": {"critic.weight": Value()}}
    fake_torch = SimpleNamespace(load=lambda *args, **kwargs: checkpoint)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Policy:
        def state_dict(self):
            return {"actor.weight": Value(), "critic.weight": Value()}

        def load_state_dict(self, *args, **kwargs):
            raise AssertionError("an incomplete actor must never be loaded")

    runner = SimpleNamespace(
        empirical_normalization=False,
        alg=SimpleNamespace(policy=Policy()),
        load=lambda path: (_ for _ in ()).throw(RuntimeError("strict missing actor")),
    )
    with pytest.raises(RuntimeError, match="missing_from_checkpoint=.*actor.weight"):
        CKPT.load_actor_tolerant(runner, "diagnostic.pt")


def test_tolerant_checkpoint_loader_accepts_zero_std_only_with_positive_epsilon(
    monkeypatch,
):
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.shape = self.values.shape

        def numel(self):
            return self.values.size

        def item(self):
            return self.values.item()

        def __ge__(self, other):
            return self.values >= other

    mean = Tensor([[0.0, 1.0, 2.0]])
    std = Tensor([[1.0, 0.0, 3.0]])
    count = Tensor(100)
    checkpoint = {
        "model_state_dict": {},
        "obs_norm_state_dict": {"_mean": mean, "_std": std, "count": count},
    }
    fake_torch = SimpleNamespace(
        all=np.all,
        empty=lambda size: Tensor(np.empty(size)),
        isfinite=lambda tensor: np.isfinite(tensor.values),
        load=lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Normalizer:
        eps = 1.0e-2

        def state_dict(self):
            return {"_mean": Tensor([[0.0, 0.0, 0.0]])}

        def load_state_dict(self, state):
            self.loaded = state

    normalizer = Normalizer()
    runner = SimpleNamespace(
        empirical_normalization=True,
        obs_normalizer=normalizer,
        alg=SimpleNamespace(policy=SimpleNamespace()),
        load=lambda path: None,
    )
    assert CKPT.load_actor_tolerant(runner, "diagnostic.pt") is True
    assert normalizer.loaded is checkpoint["obs_norm_state_dict"]

    normalizer.eps = 0.0
    with pytest.raises(RuntimeError, match="eps=0.0"):
        CKPT.load_actor_tolerant(runner, "diagnostic.pt")
