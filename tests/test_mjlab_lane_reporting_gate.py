"""Regression tests for how an mjlab-lane run may be reported (a3_train_ppo.py).

PLAIN LANGUAGE.  On 2026-08-06 an audit found that this lane's headline number
was reported in a way that made a decent result look like a bad one:

  * ``reach`` and ``touch`` in the receipt are *weighted reward terms* with
    ceilings 2.0 and 4.0, not probabilities.  ``touch = 0.21`` was quoted as if
    it were a contact rate; it is a kernel mean of 5.4%.
  * the only binary contact counter ("did the racket touch the ball in this
    episode, yes or no") was wired into ``--eval`` and nowhere else, so no
    training curve could show it.  Measured properly the same policies go
    0.12% (zero policy) -> 49.2% / 97.8%.
  * one run is not a result on this engine: four identical runs gave
    touch = 0.21 / 0.46 / 0.59 / 0.61, and the published 0.21 was the worst.

These tests pin the arithmetic and the refusal logic.  The rest of the
acceptance is the mutation battery on the pod (feed ``--report`` a single run,
a run with the contact probe off, a checkpoint eval posing as the baseline, and
check that each one is refused by name).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _lane_dir() -> Path:
  """Where the lane sources live: in the repo, or on the pod at /workspace."""
  here = (Path(__file__).resolve().parents[1]
          / "hope_training" / "whole_body_tracking" / "mjlab_lane")
  if (here / "a3_train_ppo.py").is_file():
    return here
  return Path("/workspace/mjlab_lane")


@pytest.fixture(scope="module")
def mod():
  """Import a3_train_ppo, skipping where the mujoco stack is not installed."""
  pytest.importorskip("mujoco", reason="mjlab lane needs the mujoco stack")
  pytest.importorskip("mujoco_warp", reason="mjlab lane needs mujoco_warp")
  lane = _lane_dir()
  if not (lane / "a3_train_ppo.py").is_file():
    pytest.skip(f"mjlab lane sources not found at {lane}")
  sys.path.insert(0, str(lane))
  try:
    import a3_train_ppo  # noqa: PLC0415
  finally:
    sys.path.remove(str(lane))
  return a3_train_ppo


# --------------------------------------------------------------------------
# (a) The weighted reward terms must say, in the receipt, that they are
#     weighted -- and what they are weighted by.
# --------------------------------------------------------------------------


def test_the_two_shaping_terms_are_named_weighted(mod):
  cfg = mod.TaskCfg()
  env_terms = ("alive", "pose", "upright", "height", "reach_term_weighted",
               "touch_term_weighted", "action_rate", "joint_vel", "torque",
               "termination")
  ceilings = mod.reward_term_ceilings(cfg)
  assert set(env_terms) <= set(ceilings)
  # The old, misreadable names must be gone from the receipt vocabulary.
  assert "reach" not in ceilings and "touch" not in ceilings


def test_ceilings_come_from_the_weights_not_from_a_constant(mod):
  cfg = mod.TaskCfg(w_touch=7.0, w_reach=3.0)
  ceilings = mod.reward_term_ceilings(cfg)
  assert ceilings["touch_term_weighted"] == 7.0
  assert ceilings["reach_term_weighted"] == 3.0
  # Penalty terms cannot be positive, so their ceiling is 0.
  assert ceilings["torque"] == 0.0
  assert ceilings["termination"] == 0.0


def test_the_reported_number_is_divisible_back_into_a_kernel_mean(mod):
  """0.21 with w_touch=4 is a kernel mean of 5.4%, and the receipt says so."""
  cfg = mod.TaskCfg()
  rep = mod.reward_term_report({"touch_term_weighted": 0.21,
                                "reach_term_weighted": 0.98}, cfg)
  assert rep["reward_terms_max_possible"]["touch_term_weighted"] == 4.0
  assert rep["reward_kernel_mean"]["touch_kernel_mean"] == pytest.approx(0.0525)
  assert rep["reward_kernel_mean"]["reach_kernel_mean"] == pytest.approx(0.49)
  assert rep["reward_terms_are_weighted_not_probabilities"] is True
  assert "NOT probabilities" in rep["reward_terms_note"]


def test_the_note_points_at_the_binary_metric_by_name(mod):
  rep = mod.reward_term_report({"touch_term_weighted": 0.21}, mod.TaskCfg())
  assert mod.BINARY_CONTACT_KEY in rep["reward_terms_note"]


# --------------------------------------------------------------------------
# (b) The binary contact rate: measured, or explicitly not -- never a fake 0.
# --------------------------------------------------------------------------


def test_binary_rate_is_touched_over_finished_episodes(mod):
  got = mod.binary_contact_fields(probe_on=True, touched_episodes=49.0,
                                  episodes_finished=100.0)
  assert got["fraction_of_episodes_with_a_racket_touch"] == pytest.approx(0.49)
  assert got["measured"] is True


def test_no_finished_episode_is_null_not_zero(mod):
  """The old code divided by max(episodes, 1), so an empty window read 0.0."""
  got = mod.binary_contact_fields(probe_on=True, touched_episodes=0.0,
                                  episodes_finished=0.0)
  assert got["fraction_of_episodes_with_a_racket_touch"] is None
  assert got["measured"] is False
  assert got["reason"] == "NO_EPISODES_FINISHED"


def test_probe_off_is_null_not_zero(mod):
  got = mod.binary_contact_fields(probe_on=False, touched_episodes=0.0,
                                  episodes_finished=1000.0)
  assert got["fraction_of_episodes_with_a_racket_touch"] is None
  assert got["measured"] is False
  assert got["reason"] == "CONTACT_PROBE_OFF"
  assert got["probe"] == "OFF"


def test_a_measured_zero_is_still_a_measurement(mod):
  """A policy that really never touched the ball reports 0.0, and measured."""
  got = mod.binary_contact_fields(probe_on=True, touched_episodes=0.0,
                                  episodes_finished=800.0)
  assert got["fraction_of_episodes_with_a_racket_touch"] == 0.0
  assert got["measured"] is True


def test_zero_policy_baseline_arithmetic_matches_the_measured_receipt(mod):
  """0.12% is 1 episode in ~819 -- the number in the 2026-08-06 eval receipt."""
  got = mod.binary_contact_fields(probe_on=True, touched_episodes=1.0,
                                  episodes_finished=819.2)
  assert got["fraction_of_episodes_with_a_racket_touch"] == pytest.approx(
    0.001220703125, rel=1e-3)


# --------------------------------------------------------------------------
# Curve extraction and the run-to-run band.
# --------------------------------------------------------------------------


def _rec(rate):
  return {"contact": {"fraction_of_episodes_with_a_racket_touch": rate,
                      "probe": "ON" if rate is not None else "OFF"}}


def test_curve_keeps_holes_as_none(mod):
  curve = mod._binary_contact_curve([_rec(0.1), _rec(None), _rec(0.3)])
  assert curve == [0.1, None, 0.3]


def test_band_ignores_unmeasured_runs_instead_of_averaging_them_in(mod):
  band = mod._band([0.49, None, 0.98])
  assert band["n"] == 2
  assert band["lo"] == 0.49 and band["hi"] == 0.98
  assert band["spread"] == pytest.approx(0.49)


def test_band_of_nothing_is_none_not_zero(mod):
  band = mod._band([None, None])
  assert band == {"n": 0, "lo": None, "mean": None, "hi": None, "spread": None}


def test_contact_band_over_runs_reports_the_spread(mod):
  curves = [[_rec(0.10), _rec(0.49)], [_rec(0.12), _rec(0.98)]]
  band = mod._contact_band(curves, 2)
  assert band["measured"] is True
  assert band["band_lo"][-1] == pytest.approx(0.49)
  assert band["band_hi"][-1] == pytest.approx(0.98)
  assert band["per_run_last"] == [pytest.approx(0.49), pytest.approx(0.98)]
  assert band["final_decile_spread_x"] == pytest.approx(2.0, rel=1e-6)


def test_contact_band_of_runs_without_the_probe_refuses_to_invent_zeros(mod):
  curves = [[_rec(None), _rec(None)], [_rec(None), _rec(None)]]
  band = mod._contact_band(curves, 2)
  assert band["measured"] is False
  assert "touch reward term" in band["note"]


def test_contact_band_holes_do_not_pull_the_mean_to_zero(mod):
  curves = [[_rec(0.50)], [_rec(None)]]
  band = mod._contact_band(curves, 1)
  assert band["band_mean"][0] == pytest.approx(0.50)
  assert band["band_lo"][0] == pytest.approx(0.50)


def test_learning_summary_reports_the_binary_curve(mod):
  recs = [_rec(0.01), _rec(0.2), _rec(0.4), _rec(0.5)]
  got = mod._binary_contact_summary(recs, 1)
  assert got["measured"] is True
  assert got["iterations_measured"] == 4
  assert got["fraction_of_episodes_with_a_racket_touch_first"] == pytest.approx(0.01)
  assert got["fraction_of_episodes_with_a_racket_touch_last"] == pytest.approx(0.5)
  assert got["spearman_vs_iteration"] == pytest.approx(1.0)


def test_learning_summary_says_not_measured_when_the_probe_was_off(mod):
  got = mod._binary_contact_summary([_rec(None), _rec(None)], 1)
  assert got["measured"] is False
  assert got["reason"] == "CONTACT_PROBE_OFF"
  assert "fraction_of_episodes_with_a_racket_touch_last" not in got


# --------------------------------------------------------------------------
# (c)/(d) --report refuses everything the evidence does not support.
# --------------------------------------------------------------------------


def _good_run(name="TRAIN_s0", last=0.492):
  return (name, {
    "status": "completed",
    "seed": 0,
    "capacity": {"verdict": "PASS_NO_OVERFLOW"},
    "learning": {
      "iterations": 300,
      "binary_contact_rate": {
        "measured": True,
        "fraction_of_episodes_with_a_racket_touch_first": 0.002,
        "fraction_of_episodes_with_a_racket_touch_last": last,
      },
      # Isaac's ActionBall run ends the episode on robot-vs-table contact; this
      # lane has the table and no such guard, so a reportable run has to have
      # measured the channel and found nothing.  Added 2026-08-06 with the gate
      # that reads it -- a receipt shape and the rule that judges it move
      # together or the rule is judging a shape nobody writes.
      "robot_table_contact": {"measured": True,
                              "peak_fraction_of_episodes": 0.0},
      "reward_terms_last": {"touch_term_weighted": 0.21,
                            "reach_term_weighted": 0.98},
      "reward_terms_max_possible": {"touch_term_weighted": 4.0,
                                    "reach_term_weighted": 2.0},
    },
  })


def _good_baseline():
  return ("EVAL_zero", {
    "status": "completed",
    "mode": "zero",
    "policy_steps": 750,
    "nworld": 4096,
    "capacity": {"verdict": "PASS_NO_OVERFLOW"},
    "stats": {"contact": {
      "fraction_of_episodes_with_a_racket_touch": 0.001220703125,
      "measured": True}},
  })


def _codes(refusals):
  return [c for c, _ in refusals]


def test_two_good_runs_plus_a_zero_baseline_are_reportable(mod):
  assert mod.report_refusals([_good_run("s0", 0.492), _good_run("s1", 0.978)],
                             _good_baseline()) == []


def test_one_run_is_refused_by_name(mod):
  codes = _codes(mod.report_refusals([_good_run()], _good_baseline()))
  assert "SINGLE_SEED_NOT_EVIDENCE" in codes


def test_missing_zero_policy_baseline_is_refused(mod):
  codes = _codes(mod.report_refusals(
    [_good_run("s0"), _good_run("s1")], None))
  assert "NO_ZERO_POLICY_BASELINE" in codes


def test_a_checkpoint_eval_cannot_pose_as_the_baseline(mod):
  name, b = _good_baseline()
  b["mode"] = "ckpt"
  codes = _codes(mod.report_refusals([_good_run("s0"), _good_run("s1")],
                                     (name, b)))
  assert "BASELINE_IS_NOT_A_ZERO_POLICY_RUN" in codes


def test_a_baseline_without_a_binary_rate_is_refused(mod):
  name, b = _good_baseline()
  b["stats"]["contact"] = {"fraction_of_episodes_with_a_racket_touch": None}
  codes = _codes(mod.report_refusals([_good_run("s0"), _good_run("s1")],
                                     (name, b)))
  assert "BASELINE_HAS_NO_BINARY_CONTACT_RATE" in codes


def test_a_run_without_the_binary_rate_is_refused(mod):
  name, r = _good_run("s1")
  r["learning"]["binary_contact_rate"] = {"measured": False,
                                          "reason": "CONTACT_PROBE_OFF"}
  codes = _codes(mod.report_refusals([_good_run("s0"), (name, r)],
                                     _good_baseline()))
  assert "NO_BINARY_CONTACT_RATE" in codes


def test_a_legacy_receipt_with_no_status_is_refused(mod):
  """Pre-2026-08-06 receipts have no status and no contact block at all."""
  name, r = _good_run("TRAIN_s0_2026_08_05")
  del r["status"]
  del r["learning"]["binary_contact_rate"]
  codes = _codes(mod.report_refusals([_good_run("s1"), (name, r)],
                                     _good_baseline()))
  assert "RUN_DID_NOT_COMPLETE" in codes
  assert "NO_BINARY_CONTACT_RATE" in codes


def test_a_run_that_failed_the_capacity_gate_cannot_be_reported(mod):
  name, r = _good_run("s1")
  r["capacity"] = {"verdict": "OVERFLOW", "overflow_flags": ["BROADPHASE"]}
  codes = _codes(mod.report_refusals([_good_run("s0"), (name, r)],
                                     _good_baseline()))
  assert "RUN_HAS_NO_CAPACITY_PASS" in codes


def test_a_zero_sample_capacity_verdict_cannot_be_reported(mod):
  """NO_SAMPLES is not a PASS -- the reporting gate honours the capacity gate."""
  name, r = _good_run("s1")
  r["capacity"] = {"verdict": "NO_SAMPLES"}
  codes = _codes(mod.report_refusals([_good_run("s0"), (name, r)],
                                     _good_baseline()))
  assert "RUN_HAS_NO_CAPACITY_PASS" in codes


def test_a_run_whose_robot_touched_the_table_cannot_be_reported(mod):
  """Isaac terminates on this contact; here it only contaminates the curve.

  Deliberately a *tiny* rate: 1 episode in 4096.  A gate that only fired on an
  obviously broken run would let the interesting case -- a policy that has
  quietly learned to rest a hand on the table for balance -- straight through.
  """
  name, r = _good_run("s1")
  r["learning"]["robot_table_contact"] = {
    "measured": True, "peak_fraction_of_episodes": 1.0 / 4096.0}
  codes = _codes(mod.report_refusals([_good_run("s0"), (name, r)],
                                     _good_baseline()))
  assert "ROBOT_LEANED_ON_THE_TABLE" in codes


def test_a_run_that_never_measured_the_table_channel_is_refused(mod):
  """Unmeasured must not read as zero -- the same rule as the ball channel."""
  name, r = _good_run("s1")
  del r["learning"]["robot_table_contact"]
  codes = _codes(mod.report_refusals([_good_run("s0"), (name, r)],
                                     _good_baseline()))
  assert "ROBOT_TABLE_CONTACT_NOT_MEASURED" in codes
  assert "ROBOT_LEANED_ON_THE_TABLE" not in codes


def test_refusal_messages_name_the_run_and_stay_readable(mod):
  refusals = mod.report_refusals([_good_run("TRAIN_s0")], None)
  assert refusals, "a single run with no baseline must be refused"
  for code, why in refusals:
    assert code.isupper()
    assert len(why) > 40  # a code with no sentence is a counter nobody reads


# --------------------------------------------------------------------------
# The end-to-end report, on receipts in the shape the trainer writes.
# --------------------------------------------------------------------------


def test_report_writes_the_sentence_and_exits_zero(mod, tmp_path):
  runs = []
  for name, rate in (("TRAIN_s0", 0.492), ("TRAIN_s1", 0.978)):
    p = tmp_path / (name + ".json")
    p.write_text(json.dumps(_good_run(name, rate)[1]))
    runs.append(str(p))
  bp = tmp_path / "EVAL_zero.json"
  bp.write_text(json.dumps(_good_baseline()[1]))
  out = tmp_path / "REPORT.json"
  assert mod.report(runs, str(bp), str(out)) == 0
  got = json.loads(out.read_text())
  assert got["status"] == "reported"
  assert got["headline_metric"] == mod.BINARY_CONTACT_KEY
  assert "0.12%" in got["sentence"]
  assert "49.2%" in got["sentence"] and "97.8%" in got["sentence"]
  # 0.492 / 0.00122 = 403x; 0.978 / 0.00122 = 801x.
  assert got["headline_measurement"] == "on-policy training curve, final decile"
  assert got["headline_gain_vs_zero_policy_x"][0] == pytest.approx(403.0, rel=0.02)
  assert got["headline_gain_vs_zero_policy_x"][1] == pytest.approx(801.0, rel=0.02)
  assert got["headline_run_to_run_spread_x"] == pytest.approx(1.988, rel=0.01)
  ctx = got["weighted_reward_terms_for_context_only"]
  assert ctx["max_possible"]["touch_term_weighted"] == 4.0
  assert "NOT probabilities" in ctx["warning"]


def test_report_refuses_a_single_run_and_leaves_a_receipt(mod, tmp_path):
  p = tmp_path / "TRAIN_s0.json"
  p.write_text(json.dumps(_good_run()[1]))
  bp = tmp_path / "EVAL_zero.json"
  bp.write_text(json.dumps(_good_baseline()[1]))
  out = tmp_path / "REPORT.json"
  assert mod.report([str(p)], str(bp), str(out)) == 2
  got = json.loads(out.read_text())
  assert got["status"] == "refused"
  assert got["exit_code"] == 2
  assert "SINGLE_SEED_NOT_EVIDENCE" in [r["code"] for r in got["refusals"]]


def test_analyze_refuses_a_single_run_band(mod, tmp_path):
  """A one-run 'band' reports zero spread, which is a false claim."""
  p = tmp_path / "RUN_s0.jsonl"
  p.write_text(json.dumps({"mean_episode_return": 1.0}) + "\n")
  assert mod.analyze([str(p)], str(tmp_path / "BAND.json")) == 2


# --------------------------------------------------------------------------
# The env class must not drift back to the old term names.
# --------------------------------------------------------------------------


def test_env_reward_term_names_match_the_receipt_vocabulary(mod):
  src = (_lane_dir() / "a3_train_ppo.py").read_text()
  assert '"reach_term_weighted": cfg.w_reach * reach' in src
  assert '"touch_term_weighted": cfg.w_touch * touch' in src
  assert '"reach": cfg.w_reach' not in src
  assert '"touch": cfg.w_touch' not in src


def test_training_path_wires_the_contact_probe(mod):
  """T11(b): count_contacts used to be passed only by evaluate()."""
  src = (_lane_dir() / "a3_train_ppo.py").read_text()
  assert src.count("count_contacts=not args.no_contact_probe") == 2


def test_contact_probe_failure_is_fail_closed(mod):
  src = (_lane_dir() / "a3_train_ppo.py").read_text()
  assert "CONTACT_PROBE_UNAVAILABLE" in src
  assert "contact probe disabled" not in src


# --------------------------------------------------------------------------
# A flat curve is not a rising one.
# --------------------------------------------------------------------------


def test_a_flat_curve_has_no_trend_not_a_perfect_one(mod):
  """argsort(argsort(y)) ranked a constant series 0,1,2,... and scored +1.0.

  The binary contact rate is flat at 0.0 through the early part of every run,
  so that bug printed "rising monotonically" for a policy that had never once
  touched the ball.
  """
  import math as _math

  assert _math.isnan(mod._spearman([0.0, 0.0, 0.0, 0.0]))
  assert mod._spearman([0.0, 0.0, 0.1, 0.2]) > 0.8
  assert mod._spearman([0.4, 0.3, 0.2, 0.1]) == pytest.approx(-1.0)


def test_binary_contact_summary_does_not_claim_a_trend_when_flat(mod):
  import math as _math

  got = mod._binary_contact_summary([_rec(0.0), _rec(0.0), _rec(0.0)], 1)
  assert got["measured"] is True
  assert got["fraction_of_episodes_with_a_racket_touch_max"] == 0.0
  assert _math.isnan(got["spearman_vs_iteration"])


# --------------------------------------------------------------------------
# The deterministic-eval slot (where 49.2% / 97.8% actually came from).
# --------------------------------------------------------------------------


def _good_eval(name="EVAL_ckpt_s0", rate=0.492):
  return (name, {
    "status": "completed",
    "mode": "ckpt",
    "policy_steps": 750,
    "capacity": {"verdict": "PASS_NO_OVERFLOW"},
    "stats": {"contact": {"fraction_of_episodes_with_a_racket_touch": rate,
                          "measured": True}},
  })


def test_paired_ckpt_evals_are_accepted(mod):
  assert mod.report_refusals(
    [_good_run("s0", 0.30), _good_run("s1", 0.60)], _good_baseline(),
    [_good_eval("e0", 0.492), _good_eval("e1", 0.978)]) == []


def test_one_eval_for_two_runs_is_refused(mod):
  codes = _codes(mod.report_refusals(
    [_good_run("s0"), _good_run("s1")], _good_baseline(),
    [_good_eval("e0", 0.978)]))
  assert "EVAL_COUNT_DOES_NOT_MATCH_RUNS" in codes


def test_a_zero_policy_eval_cannot_fill_the_trained_slot(mod):
  name, e = _good_eval()
  e["mode"] = "zero"
  codes = _codes(mod.report_refusals(
    [_good_run("s0"), _good_run("s1")], _good_baseline(),
    [(name, e), _good_eval("e1", 0.978)]))
  assert "EVAL_IS_NOT_A_CHECKPOINT_RUN" in codes


def test_an_eval_without_a_contact_rate_is_refused(mod):
  name, e = _good_eval()
  e["stats"]["contact"] = {"fraction_of_episodes_with_a_racket_touch": None,
                           "measured": False, "reason": "NO_EPISODES_FINISHED"}
  codes = _codes(mod.report_refusals(
    [_good_run("s0"), _good_run("s1")], _good_baseline(),
    [(name, e), _good_eval("e1", 0.978)]))
  assert "EVAL_HAS_NO_BINARY_CONTACT_RATE" in codes


def test_an_eval_that_fired_the_capacity_gate_is_refused(mod):
  name, e = _good_eval()
  e["status"] = "gate_fired"
  e["capacity"] = {"verdict": "OVERFLOW"}
  codes = _codes(mod.report_refusals(
    [_good_run("s0"), _good_run("s1")], _good_baseline(),
    [(name, e), _good_eval("e1", 0.978)]))
  assert "EVAL_DID_NOT_COMPLETE_OR_PASS" in codes


def test_report_leads_with_the_eval_numbers_when_given(mod, tmp_path):
  runs, evals = [], []
  for name, tr, ev in (("TRAIN_s0", 0.30, 0.492), ("TRAIN_s1", 0.60, 0.978)):
    p = tmp_path / (name + ".json")
    p.write_text(json.dumps(_good_run(name, tr)[1]))
    runs.append(str(p))
    q = tmp_path / (name.replace("TRAIN", "EVAL") + ".json")
    q.write_text(json.dumps(_good_eval(name, ev)[1]))
    evals.append(str(q))
  bp = tmp_path / "EVAL_zero.json"
  bp.write_text(json.dumps(_good_baseline()[1]))
  out = tmp_path / "REPORT.json"
  assert mod.report(runs, str(bp), str(out), evals) == 0
  got = json.loads(out.read_text())
  assert "deterministic eval" in got["sentence"]
  assert "49.2%" in got["sentence"] and "97.8%" in got["sentence"]
  # The on-policy training-curve numbers are still in the receipt, labelled.
  assert got["trained_on_policy_training_curve"]["per_run_last_decile"] == [0.30, 0.60]
  assert got["trained_deterministic_eval"]["per_run"] == [0.492, 0.978]


def test_report_says_which_measurement_it_used_when_no_eval_given(mod, tmp_path):
  runs = []
  for name, tr in (("TRAIN_s0", 0.30), ("TRAIN_s1", 0.60)):
    p = tmp_path / (name + ".json")
    p.write_text(json.dumps(_good_run(name, tr)[1]))
    runs.append(str(p))
  bp = tmp_path / "EVAL_zero.json"
  bp.write_text(json.dumps(_good_baseline()[1]))
  out = tmp_path / "REPORT.json"
  assert mod.report(runs, str(bp), str(out)) == 0
  got = json.loads(out.read_text())
  assert "on-policy training curve" in got["sentence"]
  assert "trained_deterministic_eval" not in got


def test_a_baseline_that_did_not_pass_the_capacity_gate_is_refused(mod):
  """The do-nothing baseline is held to the same standard as the runs."""
  name, b = _good_baseline()
  b["capacity"] = {"verdict": "NO_SAMPLES"}
  codes = _codes(mod.report_refusals([_good_run("s0"), _good_run("s1")],
                                     (name, b)))
  assert "BASELINE_DID_NOT_COMPLETE_OR_PASS" in codes


def test_a_legacy_baseline_receipt_without_status_is_refused(mod):
  """The 2026-08-05 eval receipts predate `status` -- they cannot anchor."""
  name, b = _good_baseline()
  del b["status"]
  codes = _codes(mod.report_refusals([_good_run("s0"), _good_run("s1")],
                                     (name, b)))
  assert "BASELINE_DID_NOT_COMPLETE_OR_PASS" in codes


def test_the_headline_numbers_belong_to_the_headline_measurement(mod, tmp_path):
  """The gain/spread next to the sentence must be the same measurement as it.

  With an eval slot the sentence quotes the deterministic eval, so an
  unqualified `gain_vs_zero_policy_x` computed off the training curve would be
  the exact "which number is this?" trap this file exists to close.
  """
  runs, evals = [], []
  for name, tr, ev in (("TRAIN_s0", 0.30, 0.80), ("TRAIN_s1", 0.60, 0.56)):
    p = tmp_path / (name + ".json")
    p.write_text(json.dumps(_good_run(name, tr)[1]))
    runs.append(str(p))
    q = tmp_path / (name.replace("TRAIN", "EVAL") + ".json")
    q.write_text(json.dumps(_good_eval(name, ev)[1]))
    evals.append(str(q))
  bp = tmp_path / "EVAL_zero.json"
  bp.write_text(json.dumps(_good_baseline()[1]))
  out = tmp_path / "REPORT.json"
  assert mod.report(runs, str(bp), str(out), evals) == 0
  got = json.loads(out.read_text())
  assert got["headline_measurement"] == "deterministic eval"
  assert got["headline_per_run"] == [0.80, 0.56]
  # eval spread 0.80/0.56 = 1.43, training-curve spread 0.60/0.30 = 2.0
  assert got["headline_run_to_run_spread_x"] == pytest.approx(1.4286, rel=1e-3)
  assert got["run_to_run_spread_x_on_policy_training_curve"] == pytest.approx(2.0)
  assert "gain_vs_zero_policy_x" not in got  # unqualified key must not exist
