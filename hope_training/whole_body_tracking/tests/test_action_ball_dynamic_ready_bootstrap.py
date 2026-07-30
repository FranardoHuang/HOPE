"""Dependency-light tests for the N=1 A3 dynamic-ready training bootstrap."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "dynamic_ready_training_contract_under_test", MODULE_PATH
)
TC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TC)


def _canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _sealed(document):
    result = dict(document)
    result["content_sha256"] = _sha(_canonical(result))
    return result


def _write_json(path, document):
    path.write_bytes(_canonical(document))
    return _sha(path.read_bytes())


def _materialize_inputs(tmp_path):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"pinned-motion-bytes")
    motion_sha = _sha(motion.read_bytes())
    joint_names = [f"joint_{index}" for index in range(31)]
    physical_q = [0.01 * index for index in range(31)]
    default_q = [0.0] * 31
    scale = [0.25] * 31
    hold_qdes = [0.10] * 31
    normalized = [0.40] * 31
    artifact = _sealed(
        {
            "schema_version": 1,
            "kind": TC.ACTION_BALL_DYNAMIC_READY_ARTIFACT_KIND,
            "action_id": "bh_block",
            "robot": {
                "family": "AgiBot A3",
                "joint_names": joint_names,
            },
            "authorization": {
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
                "isaac_nominal_hold_validated": False,
            },
            "sources": {
                "stable_motion": {
                    "path": str(motion),
                    "sha256": motion_sha,
                    "frame_index": 0,
                }
            },
            "physical_ready": {
                "root_pos_w_m": [0.0, 0.0, 1.0684],
                "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "joint_pos_rad": physical_q,
                "joint_vel_radps": [0.0] * 31,
            },
            "runtime_plant": {
                "default_joint_pos_rad": default_q,
                "action_scale_rad": scale,
            },
            "hold_candidate": {
                "hold_qdes_joint_pos_rad": hold_qdes,
                "normalized_actor_action": normalized,
            },
        }
    )
    artifact_path = tmp_path / "dynamic_ready.json"
    artifact_sha = _write_json(artifact_path, artifact)
    receipt = _sealed(
        {
            "schema_version": 1,
            "kind": TC.ACTION_BALL_DYNAMIC_READY_NOMINAL_HOLD_KIND,
            "verdict": "PASS",
            "action_id": "bh_block",
            "artifact": {
                "path": "/different/pod/path/dynamic_ready.json",
                "sha256": artifact_sha,
                "content_sha256": artifact["content_sha256"],
            },
            "motion_sha256": motion_sha,
            "plant_contract_match": True,
            "terminal_reasons": [],
            "generic_terminated": False,
            "generic_truncated": False,
        }
    )
    receipt_path = tmp_path / "nominal_hold.json"
    receipt_sha = _write_json(receipt_path, receipt)
    return {
        "motion": motion,
        "motion_sha": motion_sha,
        "artifact": artifact,
        "artifact_path": artifact_path,
        "artifact_sha": artifact_sha,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha": receipt_sha,
        "joint_names": joint_names,
        "physical_q": physical_q,
        "hold_qdes": hold_qdes,
        "normalized": normalized,
    }


def _load(rows):
    return TC.load_action_ball_dynamic_ready_runtime_binding(
        artifact_path=str(rows["artifact_path"]),
        artifact_sha256=rows["artifact_sha"],
        nominal_hold_receipt_path=str(rows["receipt_path"]),
        nominal_hold_receipt_sha256=rows["receipt_sha"],
        action_order=["bh_block"],
        motion_paths=[str(rows["motion"])],
    )


def _schema2_bootstrap(rows, binding):
    hard_lower = [-2.0] * 31
    hard_upper = [2.0] * 31
    return {
        "schema_version": 2,
        "kind": TC.ACTION_BALL_POLICY_BOOTSTRAP_KIND,
        "action_count": 1,
        "action_order": ["bh_block"],
        "joint_names": rows["joint_names"],
        "ready_source": {
            "semantics": (
                "action_ball_dynamic_ready.rows[action_slot].physical_ready"
            ),
            "motion_sha256_per_action": [rows["motion_sha"]],
            "physical_ready": binding["rows"][0]["physical_ready"],
            "identity": binding,
        },
        "decoder": {
            "semantics": "q_des=default_joint_pos+action_scale*action",
            "use_default_offset": True,
            "default_joint_pos": [0.0] * 31,
            "action_scale": [0.25] * 31,
            "normalized_bias": rows["normalized"],
            "target_joint_pos": rows["hold_qdes"],
            "startup_offset_delta_source": (
                "events.add_joint_default_pos.uniform_add"
            ),
            "startup_offset_delta_lower": [0.0] * 31,
            "startup_offset_delta_upper": [0.0] * 31,
        },
        "initialization": {
            "fresh_only": True,
            "resume_overwrite_prohibited": True,
            "output_layer_weight": "zeros",
            "output_layer_bias": "decoder.normalized_bias",
            "init_noise_std": 0.02,
            "sigma_envelope": 4.0,
        },
        "hard_inner_guard": {
            "limit_source": "articulation.data.joint_pos_limits",
            "margin_rad": 0.0,
            "margin_fraction": 0.02,
            "hard_lower": hard_lower,
            "hard_upper": hard_upper,
            "hard_inner_lower": [-1.92] * 31,
            "hard_inner_upper": [1.92] * 31,
        },
    }


def test_dynamic_ready_inputs_build_one_reproducible_runtime_binding(tmp_path):
    rows = _materialize_inputs(tmp_path)
    binding = _load(rows)

    assert binding["action_order"] == ["bh_block"]
    assert binding["motion_sha256_per_action"] == [rows["motion_sha"]]
    assert binding["rows"][0]["hold_qdes_joint_pos_rad"] == rows["hold_qdes"]
    assert binding["rows"][0]["normalized_actor_action"] == rows["normalized"]
    assert (
        TC.action_ball_dynamic_ready_binding_sha256(binding)
        == binding["binding_sha256"]
    )


def test_schema2_policy_bootstrap_binds_physical_ready_and_hold_qdes(tmp_path):
    rows = _materialize_inputs(tmp_path)
    binding = _load(rows)
    contract = _schema2_bootstrap(rows, binding)

    validated = TC.validate_action_ball_policy_bootstrap(
        contract, expected_action_count=1
    )

    assert validated["schema_version"] == 2
    assert (
        validated["ready_source"]["identity"]["binding_sha256"]
        == binding["binding_sha256"]
    )
    assert validated["decoder"]["target_joint_pos"] == rows["hold_qdes"]


def test_dynamic_ready_rejects_a_nominal_hold_with_generic_termination(tmp_path):
    rows = _materialize_inputs(tmp_path)
    unsigned = dict(rows["receipt"])
    unsigned.pop("content_sha256")
    unsigned["generic_terminated"] = True
    bad_receipt = _sealed(unsigned)
    rows["receipt_sha"] = _write_json(rows["receipt_path"], bad_receipt)

    with pytest.raises(ValueError, match="does not certify"):
        _load(rows)


def test_schema2_rejects_bias_that_does_not_decode_to_hold_target(tmp_path):
    rows = _materialize_inputs(tmp_path)
    binding = _load(rows)
    contract = _schema2_bootstrap(rows, binding)
    contract["decoder"]["normalized_bias"][0] = 0.0

    with pytest.raises(ValueError, match="disagrees with its runtime binding"):
        TC.validate_action_ball_policy_bootstrap(contract)


def test_schema1_shared_ready_bootstrap_remains_accepted():
    ready_q = [0.10] * 31
    joint_names = [f"joint_{index}" for index in range(31)]
    hard_lower = [-2.0] * 31
    hard_upper = [2.0] * 31
    contract = {
        "schema_version": 1,
        "kind": TC.ACTION_BALL_POLICY_BOOTSTRAP_KIND,
        "action_count": 1,
        "action_order": ["bh_block"],
        "joint_names": joint_names,
        "ready_source": {
            "semantics": "motion.joint_pos[motion.seg_start[action_slot]]",
            "canonical_ready_sha256": "",
            "canonical_ready_fk_sha256": "",
            "motion_sha256_per_action": ["a" * 64],
            "shared_ready_joint_pos": ready_q,
            "shared_ready_joint_pos_sha256": (
                TC.action_ball_shared_ready_sha256(
                    action_order=["bh_block"],
                    joint_names=joint_names,
                    shared_ready_joint_pos=ready_q,
                )
            ),
        },
        "decoder": {
            "semantics": "q_des=default_joint_pos+action_scale*action",
            "use_default_offset": True,
            "default_joint_pos": [0.0] * 31,
            "action_scale": [0.25] * 31,
            "normalized_bias": [0.40] * 31,
            "startup_offset_delta_source": (
                "events.add_joint_default_pos.uniform_add"
            ),
            "startup_offset_delta_lower": [0.0] * 31,
            "startup_offset_delta_upper": [0.0] * 31,
        },
        "initialization": {
            "fresh_only": True,
            "resume_overwrite_prohibited": True,
            "output_layer_weight": "zeros",
            "output_layer_bias": "decoder.normalized_bias",
            "init_noise_std": 0.02,
            "sigma_envelope": 4.0,
        },
        "hard_inner_guard": {
            "limit_source": "articulation.data.joint_pos_limits",
            "margin_rad": 0.0,
            "margin_fraction": 0.02,
            "hard_lower": hard_lower,
            "hard_upper": hard_upper,
            "hard_inner_lower": [-1.92] * 31,
            "hard_inner_upper": [1.92] * 31,
        },
    }

    assert TC.validate_action_ball_policy_bootstrap(contract)["schema_version"] == 1
