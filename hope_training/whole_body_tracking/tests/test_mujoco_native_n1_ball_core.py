"""Fail-closed contracts for the diagnostic MuJoCo N1 physical-ball core."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import n1_ball_core as n1  # noqa: E402
from mujoco_native import physical_ball_scene as scene  # noqa: E402
from mujoco_native import single_env  # noqa: E402
from mujoco_native import vec_env  # noqa: E402


CONTRACT = (
    REPO_ROOT
    / "configs/a3_vendor_runtime_authority_20260802_r8"
    / "bh_loop_c.shared_ready.training_contract.json"
)
BENCHMARK = (
    WBT_ROOT / "scripts/benchmark_mujoco_physical_ball_tax.py"
)


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("_n1_ball_tax_reference", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _question_payload(scene_sha: str) -> dict:
    return n1.build_question_payload(
        question_id="n1_center_000",
        scene_binding_sha256=scene_sha,
        birth_position_w_m=(2.2, 0.0, 1.2),
        birth_linear_velocity_w_mps=(-2.0, 0.0, -0.2),
        landing_aim_xy_w_m=(2.5, 0.0),
        nominal_time_to_contact_s=0.6,
    )


def _phase_reference_row(**overrides):
    row = {
        "motion_phase_context": "non_hold_swing_or_follow_through",
        "in_hold": False,
        "reference_terminations_enabled": True,
        "reference_anchor_pos_z_w_m": 0.9,
        "reference_anchor_projected_gravity_b_z": -1.0,
        "reference_ee_body_pos_z_w_m": [0.1, 0.2, 0.3, 0.4],
    }
    row.update(overrides)
    return row


def test_ball_contract_is_exact_and_strict(tmp_path):
    contract = scene.load_ball_contract(CONTRACT)
    assert contract.radius_m == pytest.approx(0.02)
    assert contract.mass_kg == pytest.approx(0.0034)
    assert contract.inertia_coeff == pytest.approx(2.0 / 3.0)
    assert contract.physics_step_dt_s == pytest.approx(0.005)
    assert contract.source_sha256 == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"physics_step_dt_s":0.005,"physics_step_dt_s":0.01}')
    with pytest.raises(scene.PhysicalBallSceneError, match="duplicate JSON key"):
        scene.load_ball_contract(duplicate)


def test_default_shared_assembler_preserves_ball_tax_xml_bytes():
    benchmark = _benchmark_module()
    table_scene = scene._load_table_scene_module()
    contract = scene.load_ball_contract(CONTRACT)
    canonical = scene.DEFAULT_MJCF.read_bytes()
    shared_xml, shared_receipt = scene.assemble_scene_xml(
        canonical,
        table_scene=table_scene,
        ball_contract=contract,
        with_ball=True,
    )
    reference_xml, reference_receipt = benchmark.assemble_scene_xml(
        canonical,
        table_scene=table_scene,
        ball_contract=contract.as_mapping(),
        with_ball=True,
    )
    assert shared_xml == reference_xml
    assert shared_receipt["assembled_xml_sha256"] == reference_receipt[
        "assembled_xml_sha256"
    ]


def test_scene_portable_hash_ignores_checkout_display_paths():
    table_scene = scene._load_table_scene_module()
    contract = scene.load_ball_contract(CONTRACT)
    _xml, receipt = scene.assemble_scene_xml(
        scene.DEFAULT_MJCF.read_bytes(),
        table_scene=table_scene,
        ball_contract=contract,
        with_ball=True,
        strict_pair_filter=True,
        include_floor_pair=True,
    )
    moved = copy.deepcopy(receipt)
    moved["ball_contract_source"]["path"] = "/workspace/other/contract.json"
    moved["table_scene_source"]["path"] = "/workspace/other/table.py"
    assert hashlib.sha256(scene._portable_binding_bytes(receipt)).hexdigest() == (
        hashlib.sha256(scene._portable_binding_bytes(moved)).hexdigest()
    )


def test_n1_scene_uses_explicit_pairs_and_excludes_robot_keepout():
    table_scene = scene._load_table_scene_module()
    contract = scene.load_ball_contract(CONTRACT)
    xml, receipt = scene.assemble_scene_xml(
        scene.DEFAULT_MJCF.read_bytes(),
        table_scene=table_scene,
        ball_contract=contract,
        with_ball=True,
        strict_pair_filter=True,
        include_floor_pair=True,
    )
    root = ET.fromstring(xml)
    ball = root.find(f".//geom[@name='{scene.BALL_GEOM_NAME}']")
    assert ball is not None
    assert ball.attrib["contype"] == "0"
    assert ball.attrib["conaffinity"] == "0"
    pair_targets = {
        pair.attrib["geom2"]
        for pair in root.findall(f"./contact/pair[@geom1='{scene.BALL_GEOM_NAME}']")
    }
    assert pair_targets == {
        scene.RACKET_GEOM_NAME,
        scene.TABLE_GEOM_NAME,
        *scene.NET_GEOM_NAMES,
        scene.FLOOR_GEOM_NAME,
    }
    assert table_scene.ACTION_BALL_ROBOT_KEEPOUT_NAME not in pair_targets
    assert receipt["robot_only_keepout_is_ball_surface"] is False
    assert receipt["ball"]["aerodynamics"] == "not_implemented"


def test_question_roundtrip_binds_scene_and_has_no_desired_contact(tmp_path):
    scene_sha = "a" * 64
    payload = _question_payload(scene_sha)
    assert payload["semantics"]["desired_at_contact"] == (
        "not_present_in_this_landing_only_core"
    )
    assert payload["diagnostic_unauthorized"] is True
    assert not any(payload["authorization"].values())
    path = tmp_path / "question.json"
    written_sha = n1.write_question(path, payload)
    question = n1.load_question(
        path,
        expected_file_sha256=written_sha,
        scene_binding_sha256=scene_sha,
    )
    assert question.source_sha256 == written_sha
    assert question.question_id == "n1_center_000"
    np.testing.assert_array_equal(question.birth_spin_w_radps, np.zeros(3))
    assert question.spin_valid is False
    with pytest.raises(single_env.ContractError, match="refusing to overwrite"):
        n1.write_question(path, payload)


def test_question_rejects_wrong_scene_nonfinite_and_invalid_spin(tmp_path):
    with pytest.raises(n1.N1BallCoreError, match="spin_valid=true is forbidden"):
        n1.build_question_payload(
            question_id="bad_spin",
            scene_binding_sha256="a" * 64,
            birth_position_w_m=(2.0, 0.0, 1.0),
            birth_linear_velocity_w_mps=(-2.0, 0.0, 0.0),
            birth_spin_w_radps=(0.0, 0.0, 0.0),
            landing_aim_xy_w_m=(2.5, 0.0),
            nominal_time_to_contact_s=0.5,
            spin_valid=True,
        )
    payload = _question_payload("a" * 64)
    path = tmp_path / "wrong_scene.json"
    n1.write_question(path, payload)
    with pytest.raises(n1.N1BallCoreError, match="different physical-ball scene"):
        n1.load_question(
            path,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scene_binding_sha256="b" * 64,
        )

    bad = tmp_path / "nan.json"
    bad.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(n1.N1BallCoreError, match="non-finite JSON"):
        n1.load_question(
            bad,
            expected_file_sha256=hashlib.sha256(bad.read_bytes()).hexdigest(),
            scene_binding_sha256="a" * 64,
        )


def test_question_content_seal_catches_tamper(tmp_path):
    payload = _question_payload("a" * 64)
    payload["task"]["landing_aim_xy_w_m"][0] += 0.01
    path = tmp_path / "tampered.json"
    path.write_bytes(n1._canonical_json_bytes(payload))
    with pytest.raises(n1.N1BallCoreError, match="content_sha256 mismatch"):
        n1.load_question(
            path,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scene_binding_sha256="a" * 64,
        )


def test_question_requires_external_file_sha(tmp_path):
    payload = _question_payload("a" * 64)
    path = tmp_path / "question.json"
    n1.write_question(path, payload)
    with pytest.raises(n1.N1BallCoreError, match="external authority"):
        n1.load_question(
            path,
            expected_file_sha256="0" * 64,
            scene_binding_sha256="a" * 64,
        )


def test_phase_reference_tape_roundtrip_binds_exact_sample_contract(tmp_path):
    contract = vec_env.phase_fidelity_sample_contract()
    payload = n1.build_phase_fidelity_reference_tape_payload(
        sample_contract=contract,
        plant_binding_sha256="a" * 64,
        scene_binding_sha256="b" * 64,
        robot_tape_sha256="c" * 64,
        authority_source_sha256="d" * 64,
        rows=(
            _phase_reference_row(),
            _phase_reference_row(
                motion_phase_context="recovery_hold",
                in_hold=True,
            ),
        ),
    )
    path = tmp_path / "phase_reference.json"
    file_sha = n1.write_phase_fidelity_reference_tape(path, payload)
    tape = n1.load_phase_fidelity_reference_tape(
        path,
        expected_file_sha256=file_sha,
        sample_contract=contract,
    )
    assert tape.sample_contract_sha256 == contract["content_sha256"]
    assert tape.ee_body_order == tuple(vec_env.PHASE_EE_BODY_NAMES)
    assert tape.robot_tape_sha256 == "c" * 64
    assert tape.rows[1].in_hold is True
    assert tape.authority_source_sha256 == "d" * 64

    with pytest.raises(n1.N1BallCoreError, match="file SHA differs"):
        n1.load_phase_fidelity_reference_tape(
            path,
            expected_file_sha256="0" * 64,
            sample_contract=contract,
        )


def test_phase_reference_tape_rejects_nonfrozen_gate_and_hold_disagreement():
    contract = vec_env.phase_fidelity_sample_contract()
    with pytest.raises(n1.N1BallCoreError, match="episode-frozen"):
        n1.build_phase_fidelity_reference_tape_payload(
            sample_contract=contract,
            plant_binding_sha256="a" * 64,
            scene_binding_sha256="b" * 64,
            robot_tape_sha256="c" * 64,
            authority_source_sha256="d" * 64,
            rows=(
                _phase_reference_row(reference_terminations_enabled=True),
                _phase_reference_row(reference_terminations_enabled=False),
            ),
        )
    with pytest.raises(n1.N1BallCoreError, match="hold/context disagree"):
        n1.build_phase_fidelity_reference_tape_payload(
            sample_contract=contract,
            plant_binding_sha256="a" * 64,
            scene_binding_sha256="b" * 64,
            robot_tape_sha256="c" * 64,
            authority_source_sha256="d" * 64,
            rows=(
                _phase_reference_row(
                    motion_phase_context="recovery_hold",
                    in_hold=False,
                ),
            ),
        )


def test_production_core_phase_sample_uses_live_native_body_state():
    core = object.__new__(n1.MujocoN1BallCore)
    core.phase_fidelity_reference_tape = SimpleNamespace(
        ee_body_order=tuple(vec_env.PHASE_EE_BODY_NAMES),
        rows=(
            n1.PhaseFidelityReferenceRow(
                motion_phase_context="non_hold_swing_or_follow_through",
                in_hold=False,
                reference_terminations_enabled=True,
                reference_anchor_pos_z_w_m=0.8,
                reference_anchor_projected_gravity_b_z=-0.1,
                reference_ee_body_pos_z_w_m=(0.3, 0.5, 0.7, 0.9),
            ),
        ),
    )
    core.policy_tick = 0
    core.plant = SimpleNamespace(_pelvis_body_id=1)
    core._phase_ee_body_ids = (2, 3, 4, 5)
    xmat = np.zeros((6, 9), dtype=np.float64)
    xmat[1] = np.eye(3, dtype=np.float64).reshape(-1)
    xpos = np.zeros((6, 3), dtype=np.float64)
    xpos[1, 2] = 0.5
    xpos[2:, 2] = [0.1, 0.2, 0.3, 0.4]
    core.data = SimpleNamespace(xmat=xmat, xpos=xpos)
    sample = core._phase_fidelity_sample()
    assert sample["anchor_pos_z_error_m"] == pytest.approx(0.3)
    assert sample["anchor_projected_gravity_z_error_abs"] == pytest.approx(0.9)
    assert sample["ee_body_pos_z_error_m"] == pytest.approx(
        [0.2, 0.3, 0.4, 0.5]
    )
    assert vec_env.exact_phase_fidelity_reasons(sample) == (
        "anchor_pos",
        "anchor_ori",
        "ee_body_pos",
    )


def test_contact_latch_collapses_persistent_points_and_invalidates_recontact():
    core = object.__new__(n1.MujocoN1BallCore)
    core.scene = SimpleNamespace(ball_geom_id=1, ball_qpos_adr=0, ball_dof_adr=0)
    core._racket_geom_id = 2
    core._table_geom_id = 3
    core._net_geom_ids = {4, 5, 6}
    core._floor_geom_id = 7
    core.policy_tick = 0
    core._active_contact_labels = set()
    core._events = []
    core._ambiguous_contact_substeps = 0
    core._racket_contact_edges = 0
    core._first_racket_contact_stamp = None
    core._outgoing_state = None
    core._contact_invalid_reasons = set()
    contract = n1.n1_reward_event_kernel.native_physical_event_facts_contract()
    core.native_physical_event_contract_sha256 = contract["content_sha256"]
    core._native_physical_event_source_binding = (
        n1.n1_reward_event_kernel.SourceBinding(
            source_id="mujoco_native/n1_ball_core.py",
            source_sha256="f" * 64,
            event_contract_sha256=contract["content_sha256"],
        )
    )
    core.data = SimpleNamespace(
        ncon=2,
        contact=[SimpleNamespace(geom1=1, geom2=2), SimpleNamespace(geom1=2, geom2=1)],
        time=0.005,
        qpos=np.asarray([0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        qvel=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    core._observe_substep(None, None, 0)
    core.data.time = 0.010
    core._observe_substep(None, None, 1)
    assert core._racket_contact_edges == 1
    assert [event["event"] for event in core._events] == ["racket"]

    core.data.ncon = 0
    core.data.contact = []
    core.data.time = 0.015
    core._observe_substep(None, None, 2)
    assert core._outgoing_state is not None
    assert core._first_racket_contact_stamp == {
        "policy_tick": 0,
        "physics_substep": 0,
    }
    facts = n1.n1_reward_event_kernel.validate_native_physical_event_facts(
        core.native_physical_event_facts(),
        expected_source=core.native_physical_event_source_binding,
    )
    assert facts["outgoing_flight"]["physics_substep"] == 2
    assert facts["selected_rubber_authority_available"] is False

    core.data.ncon = 1
    core.data.contact = [SimpleNamespace(geom1=1, geom2=2)]
    core.data.time = 0.020
    core._observe_substep(None, None, 3)
    assert core._racket_contact_edges == 2
    assert "racket_recontact" in core._contact_invalid_reasons


def test_real_mujoco_drop_hits_table_and_emits_one_or_more_edges(tmp_path):
    pytest.importorskip("mujoco")
    binding = single_env.load_plant_binding(CONTRACT)
    core = n1.MujocoN1BallCore(binding)
    robot_payload = single_env.build_probe_tape(binding, delay_steps=0)
    robot_path = tmp_path / "robot_tape.json"
    single_env.write_fixed_tape(robot_path, robot_payload)
    robot_tape = single_env.load_fixed_tape(robot_path, binding)

    table_geom_id = core.scene.obstacle_geom_ids[scene.TABLE_GEOM_NAME]
    probe = core.data
    core.mujoco.mj_forward(core.model, probe)
    center = np.asarray(probe.geom_xpos[table_geom_id], dtype=np.float64)
    top_z = float(center[2] + core.model.geom_size[table_geom_id, 2])
    payload = n1.build_question_payload(
        question_id="drop_on_table",
        scene_binding_sha256=core.scene_binding_sha256,
        birth_position_w_m=(center[0] + 0.4, center[1], top_z + 0.25),
        birth_linear_velocity_w_mps=(0.0, 0.0, -1.0),
        landing_aim_xy_w_m=(center[0] + 0.4, center[1]),
        nominal_time_to_contact_s=0.2,
    )
    question_path = tmp_path / "question.json"
    n1.write_question(question_path, payload)
    question = n1.load_question(
        question_path,
        expected_file_sha256=hashlib.sha256(question_path.read_bytes()).hexdigest(),
        scene_binding_sha256=core.scene_binding_sha256,
    )
    arrays, receipt = core.run_tape(robot_tape=robot_tape, question=question)
    assert arrays["q"].shape == (100, 31)
    assert arrays["ball_position_w_m"].shape == (100, 3)
    assert arrays["landing_aim_xy_w_m"].shape == (100, 2)
    assert receipt["status"] == "DIAGNOSTIC_MANUAL_NATIVE_BALL_PROBE_COMPLETE"
    assert receipt["counters"]["table_contact_edges"] >= 1
    assert receipt["counters"]["unexpected_contact_edges"] == 0
    assert receipt["observation_contract"]["format"] == (
        "purpose_grouped_not_final_flat_ABI"
    )
    assert receipt["known_limits"]["ppo"] == "not_implemented"
    assert receipt["diagnostic_unauthorized"] is True
    assert not any(receipt["authorization"].values())

    arrays_repeat, receipt_repeat = core.run_tape(
        robot_tape=robot_tape, question=question
    )
    for key in arrays:
        np.testing.assert_array_equal(arrays_repeat[key], arrays[key])
    assert receipt_repeat["events"] == receipt["events"]
    assert receipt_repeat["lineage"]["trace_content_sha256"] == receipt["lineage"][
        "trace_content_sha256"
    ]

    fresh = n1.MujocoN1BallCore(binding)
    arrays_fresh, receipt_fresh = fresh.run_tape(
        robot_tape=robot_tape, question=question
    )
    for key in arrays:
        np.testing.assert_array_equal(arrays_fresh[key], arrays[key])
    assert receipt_fresh["events"] == receipt["events"]


def test_real_production_core_emits_installed_phase_reference_sample(tmp_path):
    pytest.importorskip("mujoco")
    binding = single_env.load_plant_binding(CONTRACT)
    scene_probe = n1.MujocoN1BallCore(binding)
    robot_payload = single_env.build_probe_tape(binding, delay_steps=0)
    robot_path = tmp_path / "robot_tape.json"
    single_env.write_fixed_tape(robot_path, robot_payload)
    robot_tape = single_env.load_fixed_tape(robot_path, binding)
    sample_contract = vec_env.phase_fidelity_sample_contract()
    phase_payload = n1.build_phase_fidelity_reference_tape_payload(
        sample_contract=sample_contract,
        plant_binding_sha256=binding.binding_sha256,
        scene_binding_sha256=scene_probe.scene_binding_sha256,
        robot_tape_sha256=robot_tape.source_sha256,
        authority_source_sha256="e" * 64,
        rows=tuple(
            _phase_reference_row() for _ in range(robot_tape.actions.shape[0])
        ),
    )
    phase_path = tmp_path / "phase_reference.json"
    phase_file_sha = n1.write_phase_fidelity_reference_tape(
        phase_path, phase_payload
    )
    phase_tape = n1.load_phase_fidelity_reference_tape(
        phase_path,
        expected_file_sha256=phase_file_sha,
        sample_contract=sample_contract,
    )
    core = n1.MujocoN1BallCore(
        binding, phase_fidelity_reference_tape=phase_tape
    )
    payload = n1.build_question_payload(
        question_id="phase_reference_probe",
        scene_binding_sha256=core.scene_binding_sha256,
        birth_position_w_m=(2.3, 0.0, 1.5),
        birth_linear_velocity_w_mps=(-1.0, 0.0, 0.0),
        landing_aim_xy_w_m=(2.3, 0.0),
        nominal_time_to_contact_s=0.5,
    )
    question_path = tmp_path / "question.json"
    n1.write_question(question_path, payload)
    question = n1.load_question(
        question_path,
        expected_file_sha256=hashlib.sha256(
            question_path.read_bytes()
        ).hexdigest(),
        scene_binding_sha256=core.scene_binding_sha256,
    )
    core.reset(robot_tape=robot_tape, question=question)
    result = core.step(np.zeros(31, dtype=np.float64))
    assert core.phase_fidelity_sample_contract_sha256 == sample_contract[
        "content_sha256"
    ]
    assert set(result["phase_fidelity_sample"]) == set(
        sample_contract["sample_keys"]
    )
    vec_env.exact_phase_fidelity_reasons(result["phase_fidelity_sample"])
    facts = n1.n1_reward_event_kernel.validate_native_physical_event_facts(
        result["native_physical_event_facts"],
        expected_source=core.native_physical_event_source_binding,
    )
    assert facts["selected_rubber_authority_available"] is False


def test_precompiled_scene_injection_fails_closed_on_wrong_mjcf_sha():
    class FakeScene:
        canonical_xml_sha256 = "0" * 64
        collidable = True

    pytest.importorskip("mujoco")
    binding = single_env.load_plant_binding(CONTRACT)
    with pytest.raises(single_env.ContractError, match="does not bind"):
        single_env.MujocoSingleEnv(binding, precompiled_scene=FakeScene())
