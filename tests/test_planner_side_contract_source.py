from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP = (
    ROOT
    / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong"
)


def test_formal_face_schemas_require_exact_explicit_side():
    source = (CPP / "pp_planner_input.hpp").read_text(encoding="utf-8")
    assert "if (face179 && a[2] != -1.0 && a[2] != 1.0)" in source
    assert "m.has_explicit_side = face179;" in source


def test_formal_base_plausibility_hard_contract_is_source_bound():
    source = (CPP / "pp_planner_input.hpp").read_text(encoding="utf-8")
    assert "Vec3 min_source = Vec3(-3.0, -3.0, 0.4);" in source
    assert "Vec3 max_source = Vec3(3.0, 3.0, 1.5);" in source
    assert "double linear_slack_m = 0.05;" in source
    assert "double max_linear_speed_mps = 8.0;" in source
    assert "double angular_slack_rad = 0.15;" in source
    assert "double max_angular_speed_radps = 12.0;" in source
    assert "PpFormalBaseTransitionPlausible(" in source


def test_179_consumes_atomic_side_while_legacy_keeps_y_inference():
    source = (CPP / "pp_policy.hpp").read_text(encoding="utf-8")
    engage = source.split("void PlannerEngageStep_", 1)[1].split(
        "void StreamTargetStep_", 1
    )[0]
    assert engage.count("resolve_planner_swing_sign(") >= 2
    assert "referenced_tgt_b[1]" in engage
    assert "tgt_b[1]" in engage
    assert "snap.cmd.has_explicit_side" in engage
    assert "snap.cmd.swing_sign" in engage
    assert "cfg_.planner_side_split_y" in engage
    assert "cfg_.planner_side_hysteresis_y" in engage


def test_side_resolver_fails_closed_for_missing_formal_side():
    source = (CPP / "pp_reference_clock.hpp").read_text(encoding="utf-8")
    function = source.split("inline bool resolve_planner_swing_sign", 1)[1]
    assert "if (!require_explicit_side)" in function
    assert "resolved_sign = swing_sign_from_target_y(target_y);" in function
    assert "!has_explicit_side" in function
    assert "target_y < split_y - hysteresis_y" in function
    assert "target_y > split_y + hysteresis_y" in function
    assert "return false;" in function


def test_formal179_invalid_is_rejected_at_sampled_tick_and_exact_base_is_required():
    source = (CPP / "pp_policy.hpp").read_text(encoding="utf-8")
    engage = source.split("void PlannerEngageStep_", 1)[1].split(
        "void StreamTargetStep_", 1
    )[0]
    freshness = engage.index("EvaluatePpPlannerFreshness(")
    epoch = engage.index(
        "snap.cmd.control_epoch != tick.referenced_base.control_epoch"
    )
    base_sequence = engage.index(
        "snap.cmd.base_sequence_ref != tick.referenced_base.base_sequence"
    )
    side = engage.index("resolve_planner_swing_sign(")
    commit = engage.index("planner_tts0_ = tts0;")
    assert "onnx_.obs_dim() == kObsDim179" in engage[freshness:freshness + 300]
    assert freshness < epoch < base_sequence < side < commit


def test_formal179_capture_and_commit_share_one_transaction_mutex():
    source = (CPP / "pp_policy.hpp").read_text(encoding="utf-8")
    input_source = (CPP / "pp_planner_input.hpp").read_text(encoding="utf-8")
    capture = source.split("CapturePlannerControlSnapshot_", 1)[1].split(
        "void PlannerEngageStep_", 1
    )[0]
    assert "racket_tx == base_tx" in capture
    assert "std::lock_guard<std::mutex> transaction_lk(*racket_tx);" in capture
    helper = input_source.split("bool PpWithPlannerInputsIfUnchanged", 1)[1]
    assert "std::lock_guard<std::mutex> transaction_lk(*racket_tx);" in helper
    assert "racket.GenerationCurrent" in helper
    assert "base.GenerationCurrent" not in helper
    assert "base.ExactFormal(" in helper
    assert "base.Latest(" in helper
    engage = source.split("void PlannerEngageStep_", 1)[1].split(
        "void StreamTargetStep_", 1
    )[0]
    linearize = engage.index("PpWithPlannerInputsIfUnchanged(")
    recheck = engage.index("const auto current_racket =", linearize)
    base_ref_recheck = engage.index(
        "current_racket.cmd.base_sequence_ref != exact_base.base_sequence",
        recheck,
    )
    latest_recheck = engage.index(
        "current_latest_base.control_epoch", base_ref_recheck
    )
    revoke_recheck = engage.index(
        "current_latest_base.revocation_generation", latest_recheck
    )
    frozen = engage.index("commit_frozen(committed_tts);", recheck)
    assert linearize < recheck < base_ref_recheck < latest_recheck < revoke_recheck < frozen
    assert 'set_planner_status_("input_pair_not_atomic")' in engage


def test_formal179_uses_latest_base_for_closed_loop_and_history_only_for_provenance():
    source = (CPP / "pp_policy.hpp").read_text(encoding="utf-8")
    capture = source.split("CapturePlannerControlSnapshot_", 1)[1].split(
        "void PlannerEngageStep_", 1
    )[0]
    latest = capture.index("out.base_fresh = base_in_->Latest(")
    exact = capture.index("out.referenced_base_fresh = base_in_->ExactFormal(")
    assert latest < exact
    assert capture.count("base_in_->PosePlausible(out.base)") == 2

    engage = source.split("void PlannerEngageStep_", 1)[1].split(
        "void StreamTargetStep_", 1
    )[0]
    assert "tick.base.pos[2] < cfg_.base_low_z" in engage
    assert "base_pos = tick.base.pos;" in source
    assert "tick.referenced_base.pos" in engage
    assert "yaw_quat(\n          tick.referenced_base.quat)" in engage
    assert "referenced_tgt_b[1]" in engage
    assert "tgt_b[1]" in engage
    assert "base_in_->PosePlausible(current_latest_base)" in engage

    # The tick-start plausible latest gate also protects post-swing level-0
    # recovery hold before any actor observation can consume external base.
    required = source.split("const bool required_base_fresh", 1)[1].split(
        "// LIVE PLANNER static stand at level 0", 1
    )[0]
    assert "planner_tick.base_fresh" in required
    assert "std::isfinite(planner_tick.base.pos[2])" in required
    assert "planner_tick.base.pos[2] >= cfg_.base_low_z" in required
    assert "!planner_have_hold_" in required
    assert "planner_base_lease_latched_" in required
    assert "PpFormalBaseLeaseUsable(" in required
    assert "planner_latched_base_epoch_" in required
    assert "planner_latched_base_revocation_generation_" in required
    assert "cmd.kp = Eigen::VectorXd::Zero" in required
    assert "level_.store(0);" in required
    assert "rearm_yaw_align();" in required

    compute = source.split("bool ComputeCommand", 1)[1].split(
        "PpOnnxPolicy& onnx()", 1
    )[0]
    capture_idx = compute.index("CapturePlannerControlSnapshot_(state)")
    engage_idx = compute.index("PlannerEngageStep_", capture_idx)
    zero_gain_idx = compute.index("if (force_zero_gain)", engage_idx)
    recovery_gate_idx = compute.index(
        "const bool formal179_recovery_lease_usable", zero_gain_idx
    )
    recovery_lease_idx = compute.index(
        "PpFormalBaseLeaseUsable(", recovery_gate_idx
    )
    low_base_idx = compute.index(
        "planner_tick.base.pos[2] >= cfg_.base_low_z", recovery_gate_idx
    )
    recovery_zero_idx = compute.index(
        "cmd.kp = Eigen::VectorXd::Zero", low_base_idx
    )
    actor_idx = compute.index("onnx_.mean_action(", zero_gain_idx)
    assert (
        capture_idx
        < engage_idx
        < zero_gain_idx
        < recovery_gate_idx
        < recovery_lease_idx
        < low_base_idx
        < recovery_zero_idx
        < actor_idx
    )


def test_formal179_waits_on_the_explicitly_selected_clip_before_commit():
    source = (CPP / "pp_policy.hpp").read_text(encoding="utf-8")
    engage = source.split("void PlannerEngageStep_", 1)[1].split(
        "void StreamTargetStep_", 1
    )[0]
    side = engage.index("resolve_planner_swing_sign(")
    clip = engage.index("const int eng_clip = clip_id_from_swing_sign(sign);")
    face = engage.index("Validate the formal face tuple before returning `waiting_tts`")
    timing = engage.index("EvaluateExactWindupTts(")
    commit = engage.index("planner_tts0_ = tts0;")
    assert side < clip < face < timing < commit
    assert "(onnx_.obs_dim() == kObsDim110 || onnx_.obs_dim() == kObsDim179)" in engage
    assert "tts, cfg_.engage_min_tts_s, max_tts0" in engage
    assert 'set_planner_status_("waiting_tts"); return;' in engage


def test_waiting_path_rechecks_freshness_and_face_every_tick():
    source = (CPP / "pp_policy.hpp").read_text(encoding="utf-8")
    engage = source.split("void PlannerEngageStep_", 1)[1].split(
        "void StreamTargetStep_", 1
    )[0]
    freshness = engage.index("EvaluatePpPlannerFreshness(")
    side = engage.index("resolve_planner_swing_sign(")
    target_gate = engage.index("if (cfg_.target_gate_enable)")
    face = engage.index("if (onnx_.obs_dim() == kObsDim179) {", target_gate)
    waiting = engage.index("PpPlannerTtsDecision::kWaiting")
    assert freshness < side < target_gate < face < waiting

    timing_source = (CPP / "pp_reference_clock.hpp").read_text(encoding="utf-8")
    helper = timing_source.split("inline PpPlannerTtsDecision EvaluateExactWindupTts", 1)[1]
    assert "time_to_strike > clip_max_windup_s" in helper
    assert "PpPlannerTtsDecision::kWaiting" in helper
