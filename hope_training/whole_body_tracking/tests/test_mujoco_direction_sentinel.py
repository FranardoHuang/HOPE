"""CPU-only fail-closed tests for MuJoCo direction evidence and score suppression."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "scripts" / "mujoco_eval_onnx.py"
SENTINEL_PATH = ROOT / "scripts" / "mujoco_checkpoint_direction_sentinel.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("mujoco_direction_eval_under_test", EVALUATOR_PATH)
S = _load("mujoco_direction_sentinel_under_test", SENTINEL_PATH)


def _binding(**overrides):
    values = dict(
        geom_body_ids=np.asarray([2, 3, 0]),
        site_body_ids=np.asarray([2]),
        robot_body_mask=np.asarray([False, False, True, True]),
        robot_geom_mask=np.asarray([True, True, False]),
        feet_geom_ids=(1,),
        racket_site_id=0,
        racket_geom_id=0,
        table_geom_ids=(2,),
        table_geom_names=E.TABLE_CONTACT_GEOM_NAMES,
        force_threshold_n=1.0,
    )
    values.update(overrides)
    result = E.validate_robot_table_contact_binding(**values)
    result["sha256"] = E.canonical_contract_sha256(result)
    return result


def test_table_binding_requires_world_table_and_same_body_racket_site_geom():
    valid = _binding()
    assert valid["table_geom_names"] == ["motion_table_top"]
    assert valid["nonfoot_robot_geom_ids"] == [0]
    assert valid["physx_sensor_semantics_exact"] is False

    with pytest.raises(ValueError, match="non-empty"):
        _binding(table_geom_ids=(), table_geom_names=())
    with pytest.raises(ValueError, match="world body"):
        _binding(geom_body_ids=np.asarray([2, 3, 2]))
    with pytest.raises(ValueError, match="same body"):
        _binding(site_body_ids=np.asarray([3]))
    with pytest.raises(ValueError, match="foot"):
        _binding(feet_geom_ids=(0, 1))


def test_table_contact_scan_counts_nonfoot_force_but_ignores_foot_only_contact():
    contacts = [
        SimpleNamespace(geom1=2, geom2=0, dist=-0.002),
        SimpleNamespace(geom1=2, geom2=1, dist=-0.004),
    ]
    forces = {0: np.asarray([1.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
              1: np.asarray([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])}

    def contact_force(_model, _data, index, out):
        out[:] = forces[index]

    names = {0: "right_racket_collision", 1: "left_foot", 2: "motion_table_top"}
    robot = SimpleNamespace(
        table_contact_contract=_binding(),
        data=SimpleNamespace(ncon=2, contact=contacts),
        table_geom_mask=np.asarray([False, False, True]),
        nonfoot_robot_geom_mask=np.asarray([True, False, False]),
        racket_collision_geom=0,
        model=object(),
        mj=SimpleNamespace(
            mj_contactForce=contact_force,
            mj_id2name=lambda _model, _kind, geom: names[geom],
            mjtObj=SimpleNamespace(mjOBJ_GEOM=1),
        ),
    )
    result = E.MujocoRobot.table_contact_scan(robot)
    assert result == {
        "contact_count": 1,
        "racket_contact_count": 1,
        "max_force_n": pytest.approx(1.2),
        "max_penetration_m": pytest.approx(0.002),
        "worst_pair": "right_racket_collision~motion_table_top",
    }

    robot.nonfoot_robot_geom_mask[:] = False
    assert E.MujocoRobot.table_contact_scan(robot)["contact_count"] == 0


def _velocity_robot(*, allow_proxy):
    steps = []
    forwards = []
    data = SimpleNamespace(
        qpos=np.zeros(1, dtype=np.float64),
        qvel=np.asarray([1.25], dtype=np.float64),
        ctrl=np.zeros(1, dtype=np.float64),
    )
    robot = SimpleNamespace(
        data=data,
        model=object(),
        mj=SimpleNamespace(
            mj_step=lambda _model, _data: steps.append("step"),
            mj_forward=lambda _model, _data: forwards.append("forward"),
        ),
        qadr=np.asarray([0]),
        vadr=np.asarray([0]),
        act_id=np.asarray([0]),
        explicit_mask=np.asarray([True]),
        ctrl_lo=np.asarray([-10.0]),
        ctrl_hi=np.asarray([10.0]),
        _effort_guard=None,
        joint_velocity_limits=np.asarray([1.0]),
        allow_velocity_limit_proxy=allow_proxy,
        velocity_limit_hit_count=0,
        velocity_limit_pre_hit_count=0,
        velocity_limit_peak_ratio=0.0,
        fail_on_self_contact=False,
        self_contact_scan=lambda: (0, 0.0, ""),
        table_contact_contract=E.unavailable_table_contact_contract("test"),
    )
    return robot, steps, forwards


def test_pre_substep_qvel_violation_refuses_mj_step_on_formal_path():
    robot, steps, _forwards = _velocity_robot(allow_proxy=False)
    with pytest.raises(SystemExit, match="refusing mj_step"):
        E.MujocoRobot.apply_pd_and_step(
            robot, np.zeros(1), np.zeros(1), np.zeros(1), 1
        )
    assert steps == []
    assert robot.last_control_velocity_limit["first_violation_phase"] == "pre_step"


def test_pre_substep_qvel_proxy_is_explicitly_nonexact_and_continues():
    robot, steps, forwards = _velocity_robot(allow_proxy=True)
    E.MujocoRobot.apply_pd_and_step(
        robot, np.zeros(1), np.zeros(1), np.zeros(1), 1
    )
    assert steps == ["step"]
    assert forwards == ["forward"]
    assert robot.data.qvel[0] == pytest.approx(1.0)
    receipt = robot.last_control_velocity_limit
    assert receipt["proxy_applied"] is True
    assert receipt["proxy_is_post_integration_nonexact"] is True
    assert receipt["first_violation_phase"] == "pre_step"


def test_score_authority_blocks_current_mujoco_contact_twin_and_velocity_proxy():
    authority = E.build_success_score_authority_contract(
        formal_execution_contract_ok=True,
        evaluation_contract_exact=True,
        velocity_limit_proxy_allowed=True,
        implicit_effort_proxy_nonexact=False,
        table_contact_contract=_binding(),
        table_hit_terminates_episode=True,
    )
    assert authority["success_scores_authorized"] is False
    assert "post_integration_velocity_proxy" in authority["blockers"]
    assert "robot_table_contact_semantics_not_physx_exact" in authority["blockers"]
    assert "mujoco_physx_table_sensor_parity_uncertified" in authority["blockers"]


def test_authority_cannot_be_opened_by_forging_physx_exact_table_field():
    table = _binding()
    table["physx_sensor_semantics_exact"] = True
    table["sha256"] = E.canonical_contract_sha256({
        key: value for key, value in table.items() if key != "sha256"
    })
    authority = E.build_success_score_authority_contract(
        formal_execution_contract_ok=True,
        evaluation_contract_exact=True,
        velocity_limit_proxy_allowed=False,
        implicit_effort_proxy_nonexact=False,
        table_contact_contract=table,
        table_hit_terminates_episode=True,
    )
    assert authority["success_scores_authorized"] is False
    assert authority[
        "mujoco_physx_table_sensor_parity_certificate_sha256"
    ] is None
    assert authority["blockers"] == [
        "mujoco_physx_table_sensor_parity_uncertified"
    ]


def test_authority_semantics_reject_self_consistent_sha_with_contradictory_booleans():
    table = _binding()
    authority = E.build_success_score_authority_contract(
        formal_execution_contract_ok=False,
        evaluation_contract_exact=False,
        velocity_limit_proxy_allowed=True,
        implicit_effort_proxy_nonexact=False,
        table_contact_contract=table,
        table_hit_terminates_episode=True,
    )
    forged = dict(authority)
    forged["blockers"] = []
    forged["plant_parity_valid_at_launch"] = True
    forged["success_scores_authorized"] = True
    forged["sha256"] = E.canonical_contract_sha256({
        key: value for key, value in forged.items() if key != "sha256"
    })
    with pytest.raises(SystemExit, match="contradict"):
        E.validated_success_score_authority_contract(
            forged, table_contact_contract=table
        )


def test_forged_content_addressed_table_contract_is_rejected():
    forged = _binding()
    forged["available"] = False
    with pytest.raises(SystemExit, match="internally inconsistent"):
        E.validated_content_addressed_contract(
            forged,
            kind="hope_mujoco_robot_table_contact_binding",
            label="table",
        )


def test_unauthorized_success_suppression_keeps_continuous_direction_and_safety():
    source = {
        "strike_composite_success_exact": 0.9,
        "returned": True,
        "venue": {"all": {"return_success_rate": 0.8, "contact_rate": 0.7}},
        "direction_diagnostics": {
            "actor_max_abs": 12.5,
            "signed_face_dot": -0.4,
            "table_hit_control_steps": 2,
        },
    }
    out = E.suppress_unauthorized_success_scores(source)
    assert out["strike_composite_success_exact"] is None
    assert out["returned"] is None
    assert out["venue"]["all"]["return_success_rate"] is None
    assert out["venue"]["all"]["contact_rate"] == pytest.approx(0.7)
    assert out["direction_diagnostics"] == source["direction_diagnostics"]


def test_same_tick_unsafe_keeps_errors_but_blocks_every_thresholded_pass():
    strike = E.StrikeAcc()
    strike.add(0.0, 0.0, 0.0, 2.0, 2.0, unsafe=True)
    assert strike.n == 1
    assert strike.pos_err == 0.0 and strike.vel_err == 0.0 and strike.nrm_err == 0.0
    assert strike.pos_pass == strike.vel_pass == strike.nrm_pass == strike.comp == 0

    ret = SimpleNamespace(
        signed_face_exact=True,
        physical_b_opponent_facing=True,
        signed_face_ok=True,
        contacted=True,
        landing_valid=True,
        on_opponent=True,
        net_clear=True,
        landed_ok=True,
        land_err=0.03,
    )
    venue = E.VenueAcc()
    venue.add(ret, 2.0, unsafe=True)
    assert venue.contacted == 1
    assert venue.land_errs == [pytest.approx(0.03)]
    assert venue.signed_face_ok == 0
    assert venue.physical_b_opponent_facing == 0
    assert venue.landing_valid == venue.on_opp == venue.net_clear == venue.landed_ok == 0


def _summary(*, numeric_score=False):
    actor = [0.0] * 31
    actor[0:2] = [0.1, -0.2]
    table = _binding()
    authority_body = E.build_success_score_authority_contract(
        formal_execution_contract_ok=False,
        evaluation_contract_exact=False,
        velocity_limit_proxy_allowed=True,
        implicit_effort_proxy_nonexact=False,
        table_contact_contract=table,
        table_hit_terminates_episode=True,
    )
    execution = {
        "schema_version": 1,
        "kind": "hope_mujoco_bank_execution_contract",
        "velocity_limit_proxy_allowed": True,
        "implicit_effort_proxy_nonexact": False,
        "robot_hit_table_terminates_episode": True,
        "protocol_semantics": {
            "formal_bank_execution_metadata_validated": False,
        },
        "robot_table_contact": table,
        "success_score_authority": authority_body,
    }
    execution["sha256"] = E.canonical_contract_sha256(execution)
    direction_record = {
        "step": 1,
        "clip": "backhand",
        "attempt_id": 0,
        "signed_racket_velocity_along_target_mps": 1.2,
        "racket_velocity_world_x_mps": 1.0,
        "signed_face_dot": 0.4,
        "signed_face_error_deg": 66.0,
        "actual_racket_velocity_w_mps": [1.0, 0.2, 0.1],
        "target_racket_velocity_w_mps": [1.2, 0.1, 0.2],
        "actual_signed_face_normal_w": [0.0, 1.0, 0.0],
        "target_face_normal_w": [0.0, 1.0, 0.0],
        "incoming_ball_velocity_w_mps": [-2.0, 0.0, 0.0],
        "contact_closing_speed_along_target_face_normal_mps": 0.2,
        "contact_direction_formula": (
            "dot(actual_racket_velocity - incoming_ball_velocity, "
            "target_face_normal)"
        ),
        "actual_racket_velocity_source": (
            "final_physics_substep_after_mj_step_before_post_step_qvel_proxy"
        ),
        "table_hit_same_step": False,
        "absolute_fall_same_step": False,
    }
    return {
        "schema_version": 4,
        "onnx_sha256": "f" * 64,
        "evaluation_contract_exact": False,
        "success_score_authority": authority_body,
        "success_score_authority_sha256": authority_body["sha256"],
        "success_scores_authorized": False,
        "robot_table_contact_contract": table,
        "robot_table_contact_contract_sha256": table["sha256"],
        "execution_contract": execution,
        "execution_contract_sha256": execution["sha256"],
        "results": [{
            "strike_composite_success_exact": 0.5 if numeric_score else None,
            "returned": None,
            "success_score_authority": authority_body,
            "success_scores_authorized": False,
            "direction_diagnostics": {
                "finite": True,
                "actor_first_action": actor,
                "actor_first_action_abs_max": 0.2,
                "actor_max_abs": 2.0,
                "actor_max_abs_per_joint": [2.0] * 31,
                "qdes_first_raw": [0.1] * 31,
                "qdes_first_applied": [0.1] * 31,
                "qdes_first_clamp_count": 0,
                "qdes_clamp_fraction": 0.1,
                "qvel_pre_step_peak_ratio": 0.8,
                "qvel_post_step_raw_peak_ratio": 0.9,
                "qvel_post_proxy_peak_ratio": 0.9,
                "qvel_proxy_control_steps": 0,
                "signed_direction_strikes": [direction_record],
                "table_hit_control_steps": 0,
                "physical_fall_events": 0,
            },
        }],
    }


def test_sentinel_gates_stop_on_plant_parity_and_reject_leaked_numeric_score():
    decision = S.build_stop_gates(
        _summary(),
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["stop_required"] is True
    assert decision["gates"]["content_addressed_execution_contract"]["pass"] is True
    assert decision["gates"]["direction_receipt_complete"]["pass"] is True
    assert decision["gates"]["finite_direction_evidence"]["pass"] is True
    assert decision["gates"]["plant_parity_valid"]["pass"] is False
    assert decision["gates"]["unauthorized_scores_suppressed"]["pass"] is True

    leaked = S.build_stop_gates(
        _summary(numeric_score=True),
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert leaked["gates"]["unauthorized_scores_suppressed"]["pass"] is False
    assert leaked["gates"]["unauthorized_scores_suppressed"]["observed"] == [
        "results[0].strike_composite_success_exact"
    ]


def test_sentinel_rejects_nested_contract_body_swap_and_null_direction_value():
    swapped = _summary()
    nested = dict(swapped["execution_contract"]["success_score_authority"])
    nested["suppression_semantics"] = "changed body with copied nested sha"
    swapped["execution_contract"]["success_score_authority"] = nested
    execution_body = {
        key: value
        for key, value in swapped["execution_contract"].items()
        if key != "sha256"
    }
    swapped["execution_contract"]["sha256"] = E.canonical_contract_sha256(
        execution_body
    )
    swapped["execution_contract_sha256"] = swapped["execution_contract"]["sha256"]
    decision = S.build_stop_gates(
        swapped,
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["gates"]["content_addressed_execution_contract"]["pass"] is False

    nonfinite = _summary()
    nonfinite["results"][0]["direction_diagnostics"][
        "signed_direction_strikes"
    ][0]["signed_face_dot"] = None
    decision = S.build_stop_gates(
        nonfinite,
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["gates"]["direction_receipt_complete"]["pass"] is False
    assert decision["gates"]["finite_direction_evidence"]["pass"] is False


def test_sentinel_requires_bank_paper_and_refuses_output_reuse(tmp_path):
    with pytest.raises(S.SentinelError, match="target-source"):
        S.validate_evaluator_arguments([])
    arguments = S.validate_evaluator_arguments([
        "--target-source", "bank",
        "--exam-bank", "/exam.npz",
        "--exam-schedule-json", "/schedule.json",
        "--mjcf", "/robot.xml",
        "--motion-files", "/a.npz",
    ])
    assert arguments[1] == "bank"

    output = tmp_path / "sentinel"
    assert S.reserve_output_directory(output) == output.resolve()
    with pytest.raises(S.SentinelError, match="already exists"):
        S.reserve_output_directory(output)


def test_sentinel_reserves_casefold_unique_nonreceipt_milestone_names(tmp_path):
    with pytest.raises(S.SentinelError, match="alias"):
        S.validate_milestone_names(["A", "a"])
    with pytest.raises(S.SentinelError, match="reserved"):
        S.validate_milestone_names(["direction_sentinel.JSON"])

    output = S.reserve_output_directory(tmp_path / "sentinel")
    children = S.reserve_milestone_directories(output, ["m0", "m1"])
    assert children == {"m0": output / "m0", "m1": output / "m1"}
    assert all(path.is_dir() for path in children.values())


def test_sentinel_rejects_missing_motion_files_value():
    with pytest.raises(S.SentinelError, match="motion-files.*requires a value"):
        S.validate_evaluator_arguments([
            "--target-source", "bank",
            "--exam-bank", "/exam.npz",
            "--exam-schedule-json", "/schedule.json",
            "--mjcf", "/robot.xml",
            "--motion-files",
        ])


@pytest.mark.parametrize(
    "extra",
    [
        ["--target-source", "boxes"],
        ["--target-s", "boxes"],
        ["--out-d=/tmp/existing"],
        ["--onn=/tmp/other.onnx"],
        ["--expected-onnx=/tmp/not-a-sha"],
        ["--motion-f", "/tmp/other.npz"],
    ],
)
def test_sentinel_rejects_duplicate_and_abbreviated_passthrough_flags(extra):
    base = [
        "--target-source", "bank",
        "--exam-bank", "/exam.npz",
        "--exam-schedule-json", "/schedule.json",
        "--mjcf", "/robot.xml",
        "--motion-files", "/a.npz",
    ]
    with pytest.raises(S.SentinelError):
        S.validate_evaluator_arguments([*base, *extra])


def test_csv_audit_requires_bound_files_and_blank_unauthorized_scores(tmp_path):
    summary_path = tmp_path / "mujoco_sim2sim_summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    paths = {
        "per_step_csv": tmp_path / "mujoco_sim2sim_log.csv",
        "per_strike_csv": tmp_path / "mujoco_sim2sim_strikes.csv",
        "per_attempt_csv": tmp_path / "mujoco_sim2sim_attempts.csv",
    }
    paths["per_step_csv"].write_text("step,actor_action_abs_max\n0,1.0\n", encoding="utf-8")
    paths["per_strike_csv"].write_text(
        "pos_pass,vel_pass,normal_pass,composite_pass,signed_face_ok,"
        "landed_ok,net_clear,cf_signed_face_ok,cf_landed_ok,cf_net_clear\n"
        ",,,,,,,,,\n",
        encoding="utf-8",
    )
    paths["per_attempt_csv"].write_text(
        "hit,returned,exact_composite\n1,,\n", encoding="utf-8"
    )
    summary = {
        "success_scores_authorized": False,
        "artifacts": {
            key: {"path": str(path), "sha256": S.sha256_file(path)}
            for key, path in paths.items()
        },
    }
    assert S.audit_evaluator_csv_artifacts(
        summary, summary_path=summary_path
    ) == ([], [])

    paths["per_strike_csv"].write_text(
        "pos_pass,vel_pass,normal_pass,composite_pass\n1,,,\n",
        encoding="utf-8",
    )
    summary["artifacts"]["per_strike_csv"]["sha256"] = S.sha256_file(
        paths["per_strike_csv"]
    )
    integrity, leaks = S.audit_evaluator_csv_artifacts(
        summary, summary_path=summary_path
    )
    assert integrity == []
    assert leaks == ["per_strike_csv:2:pos_pass"]


def test_sentinel_rejects_wrong_action_width_nonfinite_actor_and_top_level_score():
    wrong_width = _summary()
    wrong_width["results"][0]["direction_diagnostics"]["actor_first_action"] = [0.0]
    decision = S.build_stop_gates(
        wrong_width,
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["gates"]["direction_receipt_complete"]["pass"] is False

    nonfinite = _summary()
    nonfinite["results"][0]["direction_diagnostics"]["actor_max_abs_per_joint"][0] = None
    decision = S.build_stop_gates(
        nonfinite,
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["gates"]["direction_receipt_complete"]["pass"] is False

    leaked = _summary()
    leaked["return_success_rate"] = 0.9
    decision = S.build_stop_gates(
        leaked,
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["gates"]["unauthorized_scores_suppressed"]["pass"] is False
    assert decision["gates"]["unauthorized_scores_suppressed"]["observed"] == [
        "return_success_rate"
    ]


def test_sentinel_malformed_result_is_fail_closed_not_exception():
    malformed = _summary()
    malformed["results"] = [None]
    decision = S.build_stop_gates(
        malformed,
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["stop_required"] is True
    assert decision["gates"]["direction_receipt_complete"]["pass"] is False
    assert decision["gates"]["content_addressed_execution_contract"]["pass"] is False

    malformed = _summary()
    malformed["results"][0]["direction_diagnostics"][
        "table_hit_control_steps"
    ] = None
    decision = S.build_stop_gates(
        malformed,
        evaluator_exit_code=0,
        max_qdes_clamp_fraction=0.25,
        qvel_ratio_tolerance=1e-9,
    )
    assert decision["gates"]["direction_receipt_complete"]["pass"] is False
    assert decision["gates"]["no_robot_table_hit"]["pass"] is True


def test_milestone_parser_requires_absolute_unique_style_path(tmp_path):
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"fake")
    assert S.parse_milestone(f"model_100={onnx}") == (
        "model_100", onnx.resolve()
    )
    with pytest.raises(argparse.ArgumentTypeError):
        S.parse_milestone("bad label=/tmp/policy.onnx")


def test_bound_artifact_reader_uses_exact_bytes_and_rejects_wrong_sha(tmp_path):
    artifact = tmp_path / "policy.onnx"
    artifact.write_bytes(b"exact-once")
    payload, digest = E.read_sha256_bound_bytes(artifact, label="ONNX")
    assert payload == b"exact-once"
    assert digest == S.sha256_file(artifact)
    with pytest.raises(SystemExit, match="wrapper-bound"):
        E.read_sha256_bound_bytes(
            artifact, expected_sha256="0" * 64, label="ONNX"
        )


def test_sentinel_main_binds_pre_spawn_onnx_and_evaluator_identity(
    tmp_path, monkeypatch
):
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"fake-onnx")
    output = tmp_path / "sentinel"

    def fake_run(command, **_kwargs):
        expected_onnx = command[command.index("--expected-onnx-sha256") + 1]
        expected_evaluator = command[
            command.index("--expected-evaluator-sha256") + 1
        ]
        milestone_dir = Path(command[command.index("--out-dir") + 1])
        assert milestone_dir.is_dir()
        paths = {
            "per_step_csv": milestone_dir / "mujoco_sim2sim_log.csv",
            "per_strike_csv": milestone_dir / "mujoco_sim2sim_strikes.csv",
            "per_attempt_csv": milestone_dir / "mujoco_sim2sim_attempts.csv",
        }
        paths["per_step_csv"].write_text(
            "step,actor_action_abs_max\n0,1.0\n", encoding="utf-8"
        )
        paths["per_strike_csv"].write_text(
            "pos_pass,vel_pass,normal_pass,composite_pass\n,,,\n",
            encoding="utf-8",
        )
        paths["per_attempt_csv"].write_text(
            "hit,returned,exact_composite\n1,,\n", encoding="utf-8"
        )
        summary = _summary()
        summary["onnx_sha256"] = expected_onnx
        summary["execution_contract"]["evaluator_source_sha256"] = (
            expected_evaluator
        )
        execution_body = {
            key: value
            for key, value in summary["execution_contract"].items()
            if key != "sha256"
        }
        summary["execution_contract"]["sha256"] = E.canonical_contract_sha256(
            execution_body
        )
        summary["execution_contract_sha256"] = summary["execution_contract"][
            "sha256"
        ]
        summary["input_artifacts"] = {
            "onnx": {"path": str(onnx), "sha256": expected_onnx},
            "evaluator_source": {
                "path": str(EVALUATOR_PATH),
                "sha256": expected_evaluator,
            },
        }
        summary["artifacts"] = {
            key: {"path": str(path), "sha256": S.sha256_file(path)}
            for key, path in paths.items()
        }
        (milestone_dir / "mujoco_sim2sim_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(S.subprocess, "run", fake_run)
    result = S.main([
        "--milestone", f"n5={onnx}",
        "--output-dir", str(output),
        "--evaluator", str(EVALUATOR_PATH),
        "--",
        "--target-source", "bank",
        "--exam-bank", "/exam.npz",
        "--exam-schedule-json", "/schedule.json",
        "--mjcf", "/robot.xml",
        "--motion-files", "/a.npz",
    ])
    assert result == 3
    report = json.loads((output / "direction_sentinel.json").read_text())
    milestone = report["milestones"][0]
    assert milestone["gates"]["onnx_identity"]["pass"] is True
    assert milestone["gates"]["evaluator_identity"]["pass"] is True
    assert milestone["onnx_sha256"] == S.sha256_file(onnx)
