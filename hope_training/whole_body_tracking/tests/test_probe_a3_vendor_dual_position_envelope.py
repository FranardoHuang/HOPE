"""Host-only adversarial tests for the A3 dual-envelope stress producer."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_a3_vendor_dual_position_envelope.py"
SPEC = importlib.util.spec_from_file_location("a3_dual_envelope_probe_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "waist_roll_joint",
    "right_hip_pitch_joint",
    "waist_pitch_joint",
)
H_MECH = (
    (-1.0, 1.0),
    (-0.5, 0.7),
    (-1.1, 1.1),
    (-0.8, 0.6),
)
H_CTRL = (
    (-1.0, 1.0),
    (-0.476, 0.676),
    (-1.1, 1.1),
    (-0.772, 0.572),
)


def _tape():
    return PROBE.build_stress_tape(JOINT_NAMES, H_MECH, H_CTRL)


def _diagnostic_phase(*, attempt: int, capture: int, penetration: int):
    rows = []
    for joint in PROBE.STRESSED_JOINTS:
        rows.append(
            {
                "joint": joint,
                "max_abs_delta_qdot_rad_s": 3.0,
                "sides": {
                    side: {
                        "near_ctrl_edge_readback": 2,
                        "ctrl_penetration_readback": penetration,
                        "ballistic_attempt_proxy": attempt,
                        "capture_proxy": capture,
                        "ballistic_attempt_side_flip_proxy": 0,
                        "minimum_signed_ctrl_gap_rad": -0.001,
                        "minimum_signed_mechanical_gap_rad": 0.005,
                        "max_ctrl_penetration_dwell_readbacks": 1,
                        "nonfinite_readback_observed": False,
                    }
                    for side in PROBE.SIDES
                },
            }
        )
    return {
        "enabled": True,
        "physx_control_position_limits": {
            "enabled": True,
            "semantics": "kinematic H_ctrl proxy; not a PhysX constraint impulse getter",
            "joint_order": list(PROBE.STRESSED_JOINTS),
            "side_order": list(PROBE.SIDES),
            "ballistic_horizon_s": 0.02,
            "by_joint": rows,
        },
    }


def _diagnostic():
    return {
        "pre_step": _diagnostic_phase(attempt=2, capture=0, penetration=0),
        "post_step": _diagnostic_phase(attempt=1, capture=1, penetration=2),
    }


def _vendor_binding_fixture():
    task_actions = SimpleNamespace(
        control_step_action_delay_min=0,
        control_step_action_delay_max=2,
        pre_apply_guard_brake_mode="max_inward_until_nonoutward_v1",
        pre_apply_guard_margin_fraction=0.06,
        physx_control_position_limit_inset_fraction=0.02,
    )
    task = SimpleNamespace(
        name="HOPEPingPongActionBallA3VendorV1",
        gym_task="HOPE-PingPong-ActionBall-AgibotA3-v0",
        actions=task_actions,
    )
    runtime_actions = SimpleNamespace(**vars(task_actions))
    env_cfg = SimpleNamespace(
        actions=SimpleNamespace(joint_pos=runtime_actions),
    )
    return task, env_cfg


def _observations(tape):
    rows = []
    for row in tape:
        direction = row["direction"]
        reserve = row["cage_reserve_rad"]
        trajectory = []
        for tick_index in range(1, PROBE.POLICY_HORIZON_PHYSICS_TICKS + 1):
            if row["condition"] == "on":
                # A solver-sized tick-one H_ctrl penetration is accepted only
                # because it consumes less than the full reserve and remains
                # strictly inside H_mech.
                q = row["h_ctrl_edge_rad"] + direction * (
                    0.05 * reserve if tick_index == 1 else -0.05 * reserve
                )
                qdot = direction * 0.01 if tick_index == 1 else 0.0
            elif tick_index == 1:
                # Strictly inside H_mech but outside H_ctrl on first tick.
                q = row["h_ctrl_edge_rad"] + direction * 0.30 * reserve
                qdot = 0.0
            elif tick_index == 2:
                # Same-tape OFF is the positive control: without H_ctrl it
                # crosses H_mech within the policy horizon.
                q = row["h_mech_edge_rad"] + direction * 0.02 * reserve
                qdot = 0.0
            else:
                # Remaining ticks are irrelevant after the differential is proven.
                q = row["q0_rad"]
                qdot = 0.0
            trajectory.append(
                {
                    "tick_index": tick_index,
                    "elapsed_s": tick_index * PROBE.EXACT_PHYSICS_DT_S,
                    "q_rad": q,
                    "qdot_rad_s": qdot,
                    "qdes_rad": PROBE._float32_round(row["q0_rad"]),
                }
            )
        rows.append(
            {
                "env_id": row["env_id"],
                "joint": row["joint"],
                "side": row["side"],
                "condition": row["condition"],
                "q0_live_rad": PROBE._float32_round(row["q0_rad"]),
                "qdot0_live_rad_s": PROBE._float32_round(row["qdot0_rad_s"]),
                "trajectory": trajectory,
            }
        )
    return rows


def test_exact_formula_builds_eight_same_tape_on_off_cases():
    tape = _tape()
    assert len(tape) == 8
    assert [
        (row["joint"], row["side"], row["condition"])
        for row in tape
    ] == [
        (joint, side, condition)
        for joint in PROBE.STRESSED_JOINTS
        for side in PROBE.SIDES
        for condition in PROBE.CONDITIONS
    ]

    for row in tape:
        reserve = row["cage_reserve_rad"]
        assert abs(row["qdot0_rad_s"]) == pytest.approx(0.70 * reserve / 0.005)
        assert row["qdes_rad"] == row["q0_rad"]
        assert row["kinematic_mechanical_gap_rad"] == pytest.approx(0.40 * reserve)
        assert row["q0_rad"] + row["qdot0_rad_s"] * 0.005 == pytest.approx(
            row["kinematic_q_5ms_rad"]
        )

    for index in range(0, 8, 2):
        on = dict(tape[index])
        off = dict(tape[index + 1])
        assert on.pop("env_id") == index
        assert off.pop("env_id") == index + 1
        assert on.pop("condition") == "on"
        assert off.pop("condition") == "off"
        assert on == off


def test_runtime_schema_requires_four_exact_pair_aggregates():
    tape = _tape()
    runtime = PROBE.validate_runtime_result(
        tape,
        _observations(tape),
        _diagnostic(),
        physics_dt_s=0.005,
        live_limits_restored_exact=True,
    )
    assert runtime["all_rows_finite"] is True
    assert runtime["on_mechanical_touch_or_penetration_count"] == 0
    assert runtime["off_mechanical_touch_or_penetration_count"] == 4
    assert runtime["all_on_off_input_tapes_exact"] is True
    assert runtime["policy_horizon_physics_ticks"] == 4
    assert runtime["policy_horizon_s"] == pytest.approx(0.02)
    assert runtime["existing_diagnostic_verdict_role"] == "telemetry_only"
    assert len(runtime["observations"]) == 8
    aggregates = runtime["aggregate_by_joint_side"]
    assert len(aggregates) == 4
    for aggregate, tape_row in zip(aggregates, tape[::2], strict=True):
        reserve = tape_row["cage_reserve_rad"]
        assert aggregate["joint"] == tape_row["joint"]
        assert aggregate["side"] == tape_row["side"]
        assert aggregate["strict_5ms_kinematic_attempt_count"] == 2
        assert aggregate["trajectory_tick_count"] == 8
        assert aggregate["on_strict_hmech_tick_count"] == 4
        assert aggregate["on_ctrl_penetration_tick_count"] == 1
        assert aggregate["off_first_tick_ctrl_band_entry_count"] == 1
        assert aggregate["off_mech_touch_or_penetration_tick_count"] == 1
        assert aggregate["qdes_equal_q0_exact_tick_count"] == 8
        assert aggregate["same_tape_q0_qdot_qdes_exact"] is True
        assert aggregate["max_on_ctrl_penetration_rad"] == pytest.approx(
            0.05 * reserve
        )
        assert aggregate["min_on_mech_gap_rad"] == pytest.approx(0.95 * reserve)
        assert aggregate["max_off_mech_penetration_rad"] == pytest.approx(
            0.02 * reserve
        )
        assert aggregate["existing_20ms_ballistic_attempt_proxy_count"] == 2
        assert aggregate["post_20ms_ballistic_attempt_proxy_count"] == 1
        assert aggregate["existing_20ms_capture_proxy_count"] == 1
        assert aggregate["post_ctrl_penetration_readback_count"] == 2


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda tape, obs, diag: obs[0].update(
                trajectory=[
                    {
                        **obs[0]["trajectory"][0],
                        "q_rad": tape[0]["h_mech_edge_rad"],
                    },
                    *obs[0]["trajectory"][1:],
                ]
            ),
            "full cage reserve",
        ),
        (
            lambda tape, obs, diag: obs[1]["trajectory"][0].update(
                q_rad=tape[1]["q0_rad"]
            ),
            "OFF tick one is not",
        ),
        (
            lambda tape, obs, diag: obs[2]["trajectory"][2].update(
                qdes_rad=tape[2]["qdes_rad"] + 1e-9
            ),
            "q_des differs",
        ),
        (
            lambda tape, obs, diag: obs[3].update(
                q0_live_rad=0.0,
            ),
            "q0 differs",
        ),
        (
            lambda tape, obs, diag: obs[7].update(qdot0_live_rad_s=0.0),
            "initial qdot differs",
        ),
        (
            lambda tape, obs, diag: obs[4]["trajectory"].pop(),
            "exactly four",
        ),
        (
            lambda tape, obs, diag: obs[5].update(
                trajectory=[
                    obs[5]["trajectory"][0],
                    *[
                        {
                            **sample,
                            "q_rad": tape[5]["q0_rad"],
                        }
                        for sample in obs[5]["trajectory"][1:]
                    ],
                ]
            ),
            "did not touch/cross H_mech",
        ),
        (
            lambda tape, obs, diag: obs[6]["trajectory"][1].update(
                qdot_rad_s=float("nan")
            ),
            "must be one finite number",
        ),
        (
            lambda tape, obs, diag: diag.update(pre_step=[]),
            "recordable Mapping",
        ),
    ),
)
def test_tampered_outcome_or_proxy_semantics_fail_closed(mutation, match):
    tape = _tape()
    observations = _observations(tape)
    diagnostic = _diagnostic()
    mutation(tape, observations, diagnostic)
    with pytest.raises(PROBE.DualEnvelopeProbeError, match=match):
        PROBE.validate_runtime_result(
            tape,
            observations,
            diagnostic,
            physics_dt_s=0.005,
            live_limits_restored_exact=True,
        )


def test_legacy_first_tick_capture_proxy_zero_is_preserved_but_not_a_verdict():
    tape = _tape()
    diagnostic = _diagnostic()
    for row in diagnostic["post_step"]["physx_control_position_limits"]["by_joint"]:
        for side in PROBE.SIDES:
            row["sides"][side]["capture_proxy"] = 0
    runtime = PROBE.validate_runtime_result(
        tape,
        _observations(tape),
        diagnostic,
        physics_dt_s=0.005,
        live_limits_restored_exact=True,
    )
    assert runtime["existing_diagnostic"] == diagnostic
    assert all(
        row["existing_20ms_capture_proxy_count"] == 0
        for row in runtime["aggregate_by_joint_side"]
    )


def test_legacy_diagnostic_values_and_shape_are_telemetry_only():
    tape = _tape()
    diagnostic = {
        "pre_step": {
            "physx_control_position_limits": {
                "semantics": "arbitrary-old-proxy",
                "ballistic_horizon_s": 123.0,
                "by_joint": "not parsed",
            }
        },
        "post_step": {"legacy": {"counter": -99}},
    }
    runtime = PROBE.validate_runtime_result(
        tape,
        _observations(tape),
        diagnostic,
        physics_dt_s=0.005,
        live_limits_restored_exact=True,
    )
    assert runtime["existing_diagnostic"] == diagnostic
    assert all(
        row["existing_20ms_ballistic_attempt_proxy_count"] is None
        and row["post_20ms_ballistic_attempt_proxy_count"] is None
        and row["existing_20ms_capture_proxy_count"] is None
        and row["post_ctrl_penetration_readback_count"] is None
        for row in runtime["aggregate_by_joint_side"]
    )


def test_on_off_initial_qdot_must_remain_exact_same_tape():
    tape = _tape()
    tape[1]["qdot0_rad_s"] += 1.0e-9
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="exact same state tape"):
        PROBE.validate_runtime_result(
            tape,
            _observations(_tape()),
            _diagnostic(),
            physics_dt_s=0.005,
            live_limits_restored_exact=True,
        )


def test_restore_failure_can_never_mint_pass():
    tape = _tape()
    runtime = PROBE.validate_runtime_result(
        tape,
        _observations(tape),
        _diagnostic(),
        physics_dt_s=0.005,
        live_limits_restored_exact=True,
    )
    common = {
        "source_commit": "a" * 40,
        "source_script_sha256": "b" * 64,
        "task": "Task",
        "motion_files": [],
        "tape": tape,
        "runtime": runtime,
        "live_limit_identity": {},
        "error": None,
    }
    passing = PROBE.build_receipt(
        **common,
        restore={"attempted": True, "exact_readback": True, "error": None},
    )
    assert passing["status"] == "PASS"
    assert passing["schema_version"] == 3
    assert passing["kind"] == PROBE.KIND
    assert passing["contract"]["physics_ticks"] == 4
    unhashed = dict(passing)
    content_sha256 = unhashed.pop("content_sha256")
    assert content_sha256 == PROBE._sha256_bytes(
        PROBE._canonical_json_bytes(unhashed)
    )

    receipt = PROBE.build_receipt(
        **common,
        restore={
            "attempted": True,
            "exact_readback": False,
            "error": "tampered restore",
        },
    )
    assert receipt["status"] == "FAIL"
    assert receipt["training_authorized"] is False
    assert receipt["restore"]["exact_readback"] is False

    failure_evidence = {"observations": [{"env_id": 0}], "diagnostic": {}}
    receipt = PROBE.build_receipt(
        **{**common, "runtime": None, "error": "validation failed"},
        restore={"attempted": True, "exact_readback": True, "error": None},
        failure_evidence=failure_evidence,
    )
    assert receipt["status"] == "FAIL"
    assert receipt["failure_evidence"] == failure_evidence


def test_task_is_code_owned_and_cannot_be_overridden():
    with pytest.raises(SystemExit):
        PROBE._parse_args(
            [
                "--task",
                "Some-Other-Task-v0",
                "--motion-file",
                "/tmp/motion.npz",
                "--source-root",
                "/tmp/source",
                "--expected-source-commit",
                "a" * 40,
                "--output",
                "/tmp/out.json",
            ]
        )


def test_vendor_profile_and_translated_action_contract_are_fail_closed():
    task, env_cfg = _vendor_binding_fixture()
    PROBE._validate_vendor_profile_binding(task, env_cfg)

    mutations = (
        (task, "name", "HOPEPingPongActionBall"),
        (task, "gym_task", "Other-v0"),
        (task.actions, "control_step_action_delay_max", 0),
        (task.actions, "pre_apply_guard_brake_mode", "velocity_horizon_v1"),
        (task.actions, "pre_apply_guard_margin_fraction", 0.05),
        (task.actions, "physx_control_position_limit_inset_fraction", 0.0),
        (env_cfg.actions.joint_pos, "control_step_action_delay_max", 0),
        (env_cfg.actions.joint_pos, "pre_apply_guard_margin_fraction", 0.05),
        (
            env_cfg.actions.joint_pos,
            "physx_control_position_limit_inset_fraction",
            0.0,
        ),
    )
    for node, field, bad_value in mutations:
        original = getattr(node, field)
        setattr(node, field, bad_value)
        with pytest.raises(PROBE.DualEnvelopeProbeError):
            PROBE._validate_vendor_profile_binding(task, env_cfg)
        setattr(node, field, original)


def test_probe_binding_constants_are_pinned_to_vendor_leaf_source():
    yaml = pytest.importorskip("yaml")
    source = ROOT / "cfg/task/HOPEPingPongActionBallA3VendorV1.yaml"
    profile = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert profile["name"] == PROBE.VENDOR_TASK_PROFILE
    assert profile["defaults"][0] == "HOPEPingPongActionBall@_here_"
    actions = profile["actions"]
    assert (
        actions["control_step_action_delay_min"],
        actions["control_step_action_delay_max"],
    ) == PROBE.VENDOR_CONTROL_STEP_ACTION_DELAY
    assert actions["pre_apply_guard_brake_mode"] == PROBE.VENDOR_GUARD_BRAKE_MODE
    assert actions["pre_apply_guard_margin_fraction"] == pytest.approx(
        PROBE.VENDOR_GUARD_MARGIN_FRACTION
    )
    assert actions["physx_control_position_limit_inset_fraction"] == pytest.approx(
        PROBE.CONTROL_INSET_FRACTION
    )


@pytest.mark.parametrize(
    ("action_id", "motion_name"),
    (
        ("bh_loop_c", "bh_loop_c_upper_stable_v2.npz"),
        ("bh_block", "bh_block_upper_stable_v2.npz"),
    ),
)
def test_n1_motion_resolves_only_through_code_owned_vendor_registry(
    action_id,
    motion_name,
):
    motion = ROOT.parents[1] / "assets/motions/fivebind_20260727" / motion_name
    binding = PROBE._resolve_vendor_action_binding(ROOT.parents[1], [motion])
    assert binding["action_id"] == action_id
    assert binding["motion_path"] == str(motion.resolve(strict=True))
    assert len(binding["manifest_sha256"]) == 64
    assert Path(binding["manifest_path"]).is_file()


def test_live_path_cannot_fork_to_raw_gym_defaults():
    live_source = inspect.getsource(PROBE._run_live)
    materialize_source = inspect.getsource(PROBE._materialize_vendor_env_cfg)
    assert "parse_env_cfg" not in live_source
    assert "args.task" not in live_source
    assert "_materialize_vendor_env_cfg(args)" in live_source
    assert 'gym.make(vendor_binding["gym_task"]' in live_source
    assert "parse_env_cfg(\n        str(task.gym_task)" in materialize_source
    assert "train._apply_task_overrides(" in materialize_source
    assert "_validate_vendor_profile_binding(task, env_cfg)" in materialize_source


def test_initial_same_tape_evidence_comes_from_direct_physx_readback():
    live_source = inspect.getsource(PROBE._run_live)
    write = live_source.index("base.scene.write_data_to_sim()")
    q_readback = live_source.index(
        "q0_live_readback = root_view.get_dof_positions().detach().clone()"
    )
    qdot_readback = live_source.index(
        "qdot0_live_readback = root_view.get_dof_velocities().detach().clone()"
    )
    first_step = live_source.index("base.sim.step(render=False)")
    assert write < q_readback < qdot_readback < first_step
    assert "base.scene.update(0.0)" not in live_source
    assert '"q0_live_rad": float(\n                    q0_live_readback[' in live_source
    assert '"qdot0_live_rad_s": float(\n                    qdot0_live_readback[' in live_source


def test_live_stage_markers_are_unique_and_in_code_owned_order():
    live_source = inspect.getsource(PROBE._run_live)
    offsets = []
    for marker in PROBE.STAGE_MARKERS:
        call = f'_emit_stage_marker("{marker}")'
        assert live_source.count(call) == 1
        offsets.append(live_source.index(call))
    assert offsets == sorted(offsets)
    assert len(PROBE.STAGE_MARKERS) == len(set(PROBE.STAGE_MARKERS))


def test_kit_close_cannot_preempt_receipt_publication():
    live_source = inspect.getsource(PROBE._run_live)
    main_source = inspect.getsource(PROBE.main)
    assert "simulation_app.close()" not in live_source
    assert "_write_json_exclusive(output, payload)" in main_source
    assert "publication_complete = True" in main_source
    assert "simulation_app.close()" in main_source
    assert main_source.index("_write_json_exclusive(output, payload)") < main_source.index(
        "simulation_app.close()"
    )
    assert "os._exit(exit_code if publication_complete else 1)" in main_source


def test_motion_and_contact_debug_visualization_are_explicitly_disabled():
    motion = SimpleNamespace(debug_vis=True)
    contact = SimpleNamespace(debug_vis=True)
    env_cfg = SimpleNamespace(
        commands=SimpleNamespace(motion=motion),
        scene=SimpleNamespace(contact_forces=contact),
    )
    assert PROBE._disable_debug_visualization(env_cfg) == [
        "commands.motion.debug_vis",
        "scene.contact_forces.debug_vis",
    ]
    assert motion.debug_vis is False
    assert contact.debug_vis is False

    del contact.debug_vis
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="surface is absent"):
        PROBE._disable_debug_visualization(env_cfg)


def test_json_publication_is_canonical_and_no_clobber(tmp_path: Path):
    output = tmp_path / "receipt.json"
    payload = {"schema_version": 1, "status": "FAIL", "why": "test"}
    PROBE._write_json_exclusive(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert output.read_bytes() == PROBE._canonical_json_bytes(payload)
    with pytest.raises(FileExistsError):
        PROBE._write_json_exclusive(output, payload)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def test_exact_clean_head_rejects_source_tamper(tmp_path: Path):
    root = tmp_path / "source"
    script = root / "probe.py"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "probe@example.invalid")
    _git(root, "config", "user.name", "Probe Test")
    script.write_text("print('exact')\n", encoding="utf-8")
    _git(root, "add", "probe.py")
    _git(root, "commit", "-m", "exact")
    commit = _git(root, "rev-parse", "HEAD")

    assert PROBE._verify_clean_exact_checkout(root, commit, script_path=script) == commit
    script.write_text("print('tampered')\n", encoding="utf-8")
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="exactly clean"):
        PROBE._verify_clean_exact_checkout(root, commit, script_path=script)


def test_output_must_be_outside_source_and_isaaclab_and_absent(tmp_path: Path):
    source = tmp_path / "source"
    isaaclab = tmp_path / "IsaacLab"
    external = tmp_path / "receipts"
    source.mkdir()
    isaaclab.mkdir()
    external.mkdir()
    outside = external / "stress.json"
    assert PROBE._validate_output_path(outside, (source, isaaclab)) == outside
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="outside protected root"):
        PROBE._validate_output_path(source / "stress.json", (source, isaaclab))
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="outside protected root"):
        PROBE._validate_output_path(
            isaaclab / "stress.json", (source, isaaclab)
        )
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="already exists"):
        PROBE._validate_output_path(outside, (source, isaaclab))
