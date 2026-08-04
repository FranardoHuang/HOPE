"""Fail-closed contracts for the exact-frame0 hold/tape adapter."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import exact_frame0_action_specific_hold as exact_hold  # noqa: E402
from mujoco_native import single_env as core  # noqa: E402


PRODUCER_TEST_PATH = Path(__file__).with_name(
    "test_check_table_obstacle_scene_producer.py"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _producer_test_module():
    name = "_exact_frame0_adapter_producer_fixture"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, PRODUCER_TEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FixtureBinding:
    def __init__(self, document, runtime_path: Path):
        runtime = document["runtime_plant"]
        self.source_path = str(runtime_path.resolve())
        self.source_sha256 = _sha(runtime_path.read_bytes())
        self.binding_sha256 = _sha(b"exact-frame0-fixture-binding")
        self.joint_names = tuple(runtime["joint_names"])
        self.default_joint_pos = np.asarray(
            runtime["default_joint_pos_rad"], np.float64
        )
        self.action_scale = np.asarray(runtime["action_scale_rad"], np.float64)
        self.stiffness = np.asarray(runtime["joint_stiffness"], np.float64)
        self.damping = np.asarray(runtime["joint_damping"], np.float64)
        self.effort_limits = np.asarray(
            runtime["joint_effort_limits"], np.float64
        )
        self.executed_qdes_limits = np.stack(
            (
                np.asarray(runtime["executed_qdes_lower_rad"], np.float64),
                np.asarray(runtime["executed_qdes_upper_rad"], np.float64),
            ),
            axis=1,
        )
        delay = runtime["control_step_action_delay"]
        self.delay_min_steps = int(delay["min_steps"])
        self.delay_max_steps = int(delay["max_steps"])
        self.policy_step_dt_s = float(runtime["policy_step_dt_s"])

    def decode_action(self, action):
        action = np.asarray(action, np.float64)
        raw = self.default_joint_pos + self.action_scale * action
        applied = np.clip(
            raw,
            self.executed_qdes_limits[:, 0],
            self.executed_qdes_limits[:, 1],
        )
        return raw, applied, int(np.count_nonzero(raw != applied))


def _replace_string(value, old: str, new: str):
    if isinstance(value, dict):
        return {key: _replace_string(row, old, new) for key, row in value.items()}
    if isinstance(value, list):
        return [_replace_string(row, old, new) for row in value]
    return new if value == old else value


def _write_motion(path: Path, document) -> str:
    teacher = document["teacher_reference"]
    q = np.asarray(teacher["joint_pos_rad"], np.float64)
    root_pos = np.asarray(teacher["root_pos_w_m"], np.float64)
    root_quat = np.asarray(teacher["root_quat_wxyz"], np.float64)
    order_sha = _sha(core.JOINT_ORDER_CONTRACT.read_bytes())
    with path.open("wb") as stream:
        np.savez(
            stream,
            joint_pos=q[None, :],
            joint_vel=np.zeros((1, core.ACTION_DIM), np.float64),
            body_pos_w=root_pos[None, None, :],
            body_quat_w=root_quat[None, None, :],
            body_lin_vel_w=np.zeros((1, 1, 3), np.float64),
            body_ang_vel_w=np.zeros((1, 1, 3), np.float64),
            body_names=np.asarray(["pelvis_link"]),
            body_pos_point=np.asarray("link_origin"),
            body_lin_vel_point=np.asarray("link_origin"),
            measured_racket_uid=np.asarray("exact_frame0_fixture"),
            measured_racket_joint_order_contract_id=np.asarray(
                core.JOINT_ORDER_CONTRACT_ID
            ),
            measured_racket_joint_order_contract_sha256=np.asarray(order_sha),
        )
    return _sha(path.read_bytes())


def _fixture(tmp_path: Path, monkeypatch):
    producer_test = _producer_test_module()
    artifact_path, document, _runtime = (
        producer_test._whole_body_threshold_frame0_nominal_hold_fixture(tmp_path)
    )
    motion_path = tmp_path / "exact_frame0_teacher.npz"
    motion_sha = _write_motion(motion_path, document)
    old_motion_sha = document["sources"]["stable_motion"]["sha256"]
    document = _replace_string(document, old_motion_sha, motion_sha)
    document["sources"]["stable_motion"]["path"] = str(motion_path.resolve())
    producer_test._rewrite_dynamic_ready(artifact_path, document)

    runtime_path = Path(
        document["sources"]["runtime_training_contract"]["path"]
    ).resolve()
    binding = _FixtureBinding(document, runtime_path)
    mjcf_path = Path(document["sources"]["mujoco_model"]["path"]).resolve()
    artifact_sha = _sha(artifact_path.read_bytes())
    mjcf_sha = _sha(mjcf_path.read_bytes())

    # Both the real repository validator and pytest's temporary directory are
    # below the filesystem root.  Candidate paths remain relative and replay
    # through the same resolver without weakening production containment.
    monkeypatch.setattr(core, "REPO_ROOT", Path("/"))
    return {
        "producer_test": producer_test,
        "artifact_path": artifact_path,
        "artifact": document,
        "artifact_sha": artifact_sha,
        "motion_path": motion_path,
        "motion_sha": motion_sha,
        "mjcf_path": mjcf_path,
        "mjcf_sha": mjcf_sha,
        "binding": binding,
        "action_id": document["action_id"],
    }


def _build(rows):
    return exact_hold.build_candidate(
        binding=rows["binding"],
        artifact_path=rows["artifact_path"],
        expected_artifact_sha256=rows["artifact_sha"],
        teacher_motion=rows["motion_path"],
        expected_teacher_motion_sha256=rows["motion_sha"],
        expected_action_id=rows["action_id"],
        mjcf_path=rows["mjcf_path"],
        expected_mjcf_sha256=rows["mjcf_sha"],
    )


def _rewrite(rows, document) -> None:
    rows["producer_test"]._rewrite_dynamic_ready(
        rows["artifact_path"], document
    )
    rows["artifact"] = document
    rows["artifact_sha"] = _sha(rows["artifact_path"].read_bytes())


def test_exact_frame0_artifact_roundtrips_candidate_and_tape(
    tmp_path, monkeypatch
):
    rows = _fixture(tmp_path, monkeypatch)
    candidate = _build(rows)
    assert core._canonical_json_bytes(candidate) == core._canonical_json_bytes(
        _build(rows)
    )
    assert candidate["kind"] == exact_hold.KIND
    assert candidate["schema_version"] == exact_hold.SCHEMA_VERSION
    assert "shared_lower_root_seed" not in candidate["sources"]
    assert candidate["semantics"] == {
        "teacher_reference_unchanged": True,
        "physical_reset": exact_hold.PHYSICAL_RESET_SEMANTICS,
        "controller_birth_target": exact_hold.CONTROLLER_BIRTH_SEMANTICS,
        "history_fill": exact_hold.HISTORY_FILL_SEMANTICS,
        "teacher_and_physical_reset_may_differ": False,
        "threshold_first_fallback_used": False,
        "certified_transition_s": 0.0,
    }
    np.testing.assert_array_equal(
        candidate["physical_ready"]["joint_pos"],
        rows["artifact"]["teacher_reference"]["joint_pos_rad"],
    )
    np.testing.assert_array_equal(
        candidate["physical_ready"]["root_pos"],
        rows["artifact"]["teacher_reference"]["root_pos_w_m"],
    )
    np.testing.assert_array_equal(
        candidate["physical_ready"]["root_quat_wxyz"],
        rows["artifact"]["teacher_reference"]["root_quat_wxyz"],
    )
    assert candidate["physical_ready"]["joint_vel"] == [0.0] * 31
    assert candidate["physical_ready"]["root_lin_vel_w"] == [0.0] * 3
    assert candidate["physical_ready"]["root_ang_vel_w"] == [0.0] * 3

    candidate_path = tmp_path / "candidate.json"
    candidate_raw = core._canonical_json_bytes(candidate)
    core._write_new_bytes(candidate_path, candidate_raw)
    candidate_sha = _sha(candidate_raw)
    tape_payload = core.build_probe_tape(
        rows["binding"],
        delay_steps=0,
        teacher_motion=rows["motion_path"],
        teacher_frame_index=0,
        hold_candidate=candidate_path,
        expected_hold_candidate_sha256=candidate_sha,
    )
    tape_path = tmp_path / "tape.json"
    core.write_fixed_tape(tape_path, tape_payload)
    tape = core.load_fixed_tape(tape_path, rows["binding"])
    assert tape.reset_state.mode == "action_specific_hold"
    assert tape.reset_state.hold_candidate_kind == exact_hold.KIND
    assert tape.reset_state.hold_candidate_schema_version == exact_hold.SCHEMA_VERSION
    np.testing.assert_array_equal(
        tape.reset_state.joint_pos,
        rows["artifact"]["physical_ready"]["joint_pos_rad"],
    )
    np.testing.assert_array_equal(
        tape.history_fill_action,
        rows["artifact"]["hold_candidate"]["normalized_actor_action"],
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "fallback",
        "nonzero_transition",
        "nonzero_velocity",
        "unequal_joint",
        "unequal_root",
        "unequal_quaternion",
    ),
)
def test_adapter_rejects_fallback_transition_velocity_or_endpoint(
    tmp_path, monkeypatch, mutation
):
    rows = _fixture(tmp_path, monkeypatch)
    document = copy.deepcopy(rows["artifact"])
    if mutation == "fallback":
        document["physical_birth_composition"][
            "exact_measured_frame0_selected"
        ] = False
        document["physical_birth_composition"]["optimizer_report"][
            "exact_measured_frame0_selected"
        ] = False
        document["physical_birth_static_evidence"]["optimizer_report"][
            "exact_measured_frame0_selected"
        ] = False
    elif mutation == "nonzero_transition":
        for handoff in (
            document["frame0_handoff"],
            document["physical_birth_composition"]["frame0_handoff"],
            document["physical_birth_static_evidence"]["frame0_handoff"],
        ):
            handoff["certified_transition_s"] = 0.01
    elif mutation == "nonzero_velocity":
        document["physical_ready"]["joint_vel_radps"][0] = 0.01
    elif mutation == "unequal_joint":
        document["physical_ready"]["joint_pos_rad"][0] += 0.01
    elif mutation == "unequal_root":
        document["physical_ready"]["root_pos_w_m"][0] += 0.01
    else:
        document["physical_ready"]["root_quat_wxyz"][1] += 0.01
    _rewrite(rows, document)
    with pytest.raises(core.ContractError, match="threshold-first|exact-frame0"):
        _build(rows)


def test_adapter_rejects_wrong_action_motion_and_file_sha(tmp_path, monkeypatch):
    rows = _fixture(tmp_path, monkeypatch)
    kwargs = {
        "binding": rows["binding"],
        "artifact_path": rows["artifact_path"],
        "expected_artifact_sha256": rows["artifact_sha"],
        "teacher_motion": rows["motion_path"],
        "expected_teacher_motion_sha256": rows["motion_sha"],
        "expected_action_id": rows["action_id"],
        "mjcf_path": rows["mjcf_path"],
        "expected_mjcf_sha256": rows["mjcf_sha"],
    }
    for field, bad_value, pattern in (
        ("expected_action_id", "wrong_action", "action id"),
        ("expected_teacher_motion_sha256", "0" * 64, "teacher motion SHA"),
        ("expected_artifact_sha256", "0" * 64, "artifact SHA"),
        ("expected_mjcf_sha256", "0" * 64, "MJCF SHA"),
    ):
        bad = dict(kwargs)
        bad[field] = bad_value
        with pytest.raises(core.ContractError, match=pattern):
            exact_hold.build_candidate(**bad)

    wrong_binding = copy.copy(rows["binding"])
    wrong_binding.source_sha256 = "0" * 64
    bad = dict(kwargs)
    bad["binding"] = wrong_binding
    with pytest.raises(core.ContractError, match="plant contract"):
        exact_hold.build_candidate(**bad)


def test_adapter_rejects_unsealed_static_evidence(tmp_path, monkeypatch):
    rows = _fixture(tmp_path, monkeypatch)
    document = copy.deepcopy(rows["artifact"])
    document["physical_birth_static_evidence"][
        "fresh_direct_robust_gate_passed"
    ] = False
    # Preserve the old content seal deliberately, while updating only the
    # externally supplied file SHA.  The schema loader must reject the edit.
    rows["artifact_path"].write_bytes(
        rows["producer_test"].P._canonical_json_bytes(document)
        if hasattr(rows["producer_test"], "P")
        else json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    )
    rows["artifact_sha"] = _sha(rows["artifact_path"].read_bytes())
    with pytest.raises(core.ContractError, match="content SHA"):
        _build(rows)


def test_tape_consumer_reopens_threshold_artifact(tmp_path, monkeypatch):
    rows = _fixture(tmp_path, monkeypatch)
    candidate = _build(rows)
    candidate_path = tmp_path / "candidate.json"
    candidate_raw = core._canonical_json_bytes(candidate)
    core._write_new_bytes(candidate_path, candidate_raw)
    tape_payload = core.build_probe_tape(
        rows["binding"],
        delay_steps=0,
        teacher_motion=rows["motion_path"],
        hold_candidate=candidate_path,
        expected_hold_candidate_sha256=_sha(candidate_raw),
    )
    tape_path = tmp_path / "tape.json"
    core.write_fixed_tape(tape_path, tape_payload)

    document = copy.deepcopy(rows["artifact"])
    document["physical_ready"]["joint_vel_radps"][0] = 0.01
    _rewrite(rows, document)
    with pytest.raises(core.ContractError, match="source.*SHA mismatch"):
        core.load_fixed_tape(tape_path, rows["binding"])


@pytest.mark.parametrize("mutation", ("semantics", "leg_partition"))
def test_candidate_rebuild_rejects_coordinated_reseal(
    tmp_path, monkeypatch, mutation
):
    rows = _fixture(tmp_path, monkeypatch)
    candidate = _build(rows)
    if mutation == "semantics":
        candidate["semantics"]["threshold_first_fallback_used"] = True
    else:
        replacement = list(range(12))
        candidate["physical_ready"]["leg_joint_indices"] = replacement
        candidate["physical_ready"]["leg_joint_names"] = [
            rows["binding"].joint_names[index] for index in replacement
        ]
    unsigned = dict(candidate)
    unsigned.pop("content_sha256")
    candidate["content_sha256"] = _sha(core._canonical_json_bytes(unsigned))
    raw = core._canonical_json_bytes(candidate)
    path = tmp_path / f"tampered_{mutation}.json"
    path.write_bytes(raw)
    with pytest.raises(core.ContractError, match="deterministic source projection"):
        core.build_probe_tape(
            rows["binding"],
            delay_steps=0,
            teacher_motion=rows["motion_path"],
            hold_candidate=path,
            expected_hold_candidate_sha256=_sha(raw),
        )


@pytest.mark.parametrize(
    "mutation",
    ("history_fill", "reset_joint", "candidate_content_identity"),
)
def test_exact_tape_rejects_reset_history_or_identity_tamper(
    tmp_path, monkeypatch, mutation
):
    rows = _fixture(tmp_path, monkeypatch)
    candidate = _build(rows)
    candidate_path = tmp_path / "candidate_for_tamper.json"
    candidate_raw = core._canonical_json_bytes(candidate)
    core._write_new_bytes(candidate_path, candidate_raw)
    tape = core.build_probe_tape(
        rows["binding"],
        delay_steps=0,
        teacher_motion=rows["motion_path"],
        hold_candidate=candidate_path,
        expected_hold_candidate_sha256=_sha(candidate_raw),
    )
    if mutation == "history_fill":
        tape["history_fill_action"][0] += 1.0e-5
        pattern = "history fill"
    elif mutation == "reset_joint":
        tape["reset_state"]["joint_pos"][0] += 1.0e-5
        pattern = "reset state"
    else:
        tape["reset_state"]["hold_candidate_content_sha256"] = "0" * 64
        pattern = "reset lineage"
    path = tmp_path / f"tampered_tape_{mutation}.json"
    path.write_bytes(core._canonical_json_bytes(tape))
    with pytest.raises(core.ContractError, match=pattern):
        core.load_fixed_tape(path, rows["binding"])


def test_threshold_validator_cache_rejects_source_drift(tmp_path, monkeypatch):
    module = exact_hold._load_threshold_validator()
    assert module is not None
    copied = tmp_path / "threshold_validator.py"
    copied.write_bytes(exact_hold.THRESHOLD_VALIDATOR_PATH.read_bytes())
    monkeypatch.setattr(exact_hold, "THRESHOLD_VALIDATOR_PATH", copied)
    assert exact_hold._load_threshold_validator() is module
    copied.write_bytes(copied.read_bytes() + b"\n# drift\n")
    with pytest.raises(core.ContractError, match="source drifted"):
        exact_hold._load_threshold_validator()


def test_runtime_reset_reopens_exact_candidate_and_artifact(tmp_path, monkeypatch):
    rows = _fixture(tmp_path, monkeypatch)
    candidate = _build(rows)
    candidate_path = tmp_path / "runtime_candidate.json"
    candidate_raw = core._canonical_json_bytes(candidate)
    core._write_new_bytes(candidate_path, candidate_raw)
    tape_payload = core.build_probe_tape(
        rows["binding"],
        delay_steps=0,
        teacher_motion=rows["motion_path"],
        hold_candidate=candidate_path,
        expected_hold_candidate_sha256=_sha(candidate_raw),
    )
    tape_path = tmp_path / "runtime_tape.json"
    core.write_fixed_tape(tape_path, tape_payload)
    tape = core.load_fixed_tape(tape_path, rows["binding"])
    runner = object.__new__(core.MujocoSingleEnv)
    runner.binding = rows["binding"]
    runner.mjcf_path = rows["mjcf_path"]
    runner.scene = SimpleNamespace(canonical_xml_sha256=rows["mjcf_sha"])
    runner._revalidate_action_specific_reset_authority(
        tape.reset_state, tape.history_fill_action
    )
    assert "_revalidate_action_specific_reset_authority" in inspect.getsource(
        core.MujocoSingleEnv.reset
    )

    document = copy.deepcopy(rows["artifact"])
    document["physical_ready"]["joint_vel_radps"][0] = 0.01
    _rewrite(rows, document)
    with pytest.raises(core.ContractError, match="source.*SHA mismatch"):
        runner._revalidate_action_specific_reset_authority(
            tape.reset_state, tape.history_fill_action
        )


def test_pair_materializer_is_no_clobber(tmp_path, monkeypatch):
    rows = _fixture(tmp_path, monkeypatch)
    candidate_path = tmp_path / "pair_candidate.json"
    tape_path = tmp_path / "pair_tape.json"
    result = exact_hold.materialize_candidate_and_tape(
        binding=rows["binding"],
        artifact_path=rows["artifact_path"],
        expected_artifact_sha256=rows["artifact_sha"],
        teacher_motion=rows["motion_path"],
        expected_teacher_motion_sha256=rows["motion_sha"],
        expected_action_id=rows["action_id"],
        mjcf_path=rows["mjcf_path"],
        expected_mjcf_sha256=rows["mjcf_sha"],
        candidate_output=candidate_path,
        tape_output=tape_path,
    )
    assert result["status"] == "EXACT_FRAME0_CANDIDATE_AND_TAPE_WRITTEN"
    candidate_before = candidate_path.read_bytes()
    tape_before = tape_path.read_bytes()
    with pytest.raises(core.ContractError, match="refuses existing outputs"):
        exact_hold.materialize_candidate_and_tape(
            binding=rows["binding"],
            artifact_path=rows["artifact_path"],
            expected_artifact_sha256=rows["artifact_sha"],
            teacher_motion=rows["motion_path"],
            expected_teacher_motion_sha256=rows["motion_sha"],
            expected_action_id=rows["action_id"],
            mjcf_path=rows["mjcf_path"],
            expected_mjcf_sha256=rows["mjcf_sha"],
            candidate_output=candidate_path,
            tape_output=tape_path,
        )
    assert candidate_path.read_bytes() == candidate_before
    assert tape_path.read_bytes() == tape_before
