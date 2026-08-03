"""Host tests for the exact A211 frame0 live-hold wrapper/consumer."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = _load(
    "a211_frame0_hold_consumer_test",
    SCRIPT_DIR / "consume_action_ball_a211_frame0_nominal_hold.py",
)
R = _load(
    "a211_frame0_hold_wrapper_test",
    SCRIPT_DIR / "run_action_ball_a211_frame0_nominal_hold.py",
)

ARTIFACT_PATH = ROOT / (
    "configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/"
    "take_061_unit04_bh.frame0_exact.v1.json"
)
TEMPLATE_PATH = ROOT / (
    "configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/"
    "take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json"
)
MOTION_PATH = ROOT / (
    "assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz"
)
ARTIFACT_SOURCE_COMMIT = "5ed998f1e1526fa84dfc2198b064f9f8e6ab6068"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive(tmp_path: Path):
    frame = _json(ARTIFACT_PATH)
    template = _json(TEMPLATE_PATH)
    probe = C.derive_probe_input(
        frame0_artifact=frame,
        frame0_file_sha256=C.sha256_file(ARTIFACT_PATH),
        artifact_source_commit=ARTIFACT_SOURCE_COMMIT,
        plant_template=template,
        plant_template_file_sha256=C.sha256_file(TEMPLATE_PATH),
        motion_path=MOTION_PATH,
        motion_sha256=C.sha256_file(MOTION_PATH),
        probe_source_commit="a" * 40,
    )
    probe_path = tmp_path / "probe.json"
    probe_path.write_bytes(C.canonical_bytes(probe) + b"\n")
    return frame, template, probe, probe_path


def test_tracked_pretty_plant_template_is_strict_but_need_not_be_canonical():
    template, raw = C._strict_json(
        TEMPLATE_PATH,
        name="plant template",
        newline=None,
        canonical=False,
    )
    assert template["runtime_plant"]["control_decimation"] > 0
    assert raw == TEMPLATE_PATH.read_bytes()


def test_live_command_uses_artifact_owned_motion_without_cli_override(tmp_path: Path):
    command = R._live_command(
        python="python",
        device="cuda:0",
        probe_path=tmp_path / "probe.json",
        probe_sha256="a" * 64,
        live_path=tmp_path / "live.json",
        screenshot_dir=tmp_path / "screenshots",
        duration_s=4.0,
    )
    assert "--motion-file" not in command
    assert command[command.index("--nominal-hold") + 1].endswith("probe.json")


def _live(tmp_path: Path, frame: dict, template: dict, probe: dict, probe_path: Path):
    screenshots = []
    for index, label in enumerate((
        "raw_env_reset", "physical_ready_after_reset_write",
        "after_step_1", "after_step_10", "final",
    )):
        path = tmp_path / ("%02d.png" % index)
        path.write_bytes(("png-%s" % label).encode())
        screenshots.append({
            "label": label,
            "policy_step": (0, 0, 1, 10, 200)[index],
            "path": str(path),
            "sha256": C.sha256_file(path),
        })
    names = template["robot"]["joint_names"]
    zero = [0.0] * 31
    joint = {
        "schema_version": 1,
        "complete": True,
        "joint_order": names,
        "current_actual_hard_edge_joint_count": 0,
        "current_actual_hard_edge_joint_names": [],
        "substep_actual_hard_edge_joint_count": 0,
        "substep_actual_hard_edge_joint_names": [],
        "final_minimum_hard_gap_rad": 0.025,
        "preterminal_joint_pos_rad": zero,
        "preterminal_joint_vel_radps": zero,
        "final_joint_pos_rad": zero,
        "final_joint_vel_radps": zero,
        "hard_lower_rad": [-1.0] * 31,
        "hard_upper_rad": [1.0] * 31,
    }
    duration = frame["task_close_ticks"] * frame["policy_dt_s"]
    unsigned = {
        "schema_version": 1,
        "kind": C.GENERIC_RECEIPT_KIND,
        "verdict": "PASS",
        "action_id": frame["action_id"],
        "artifact": {
            "path": str(probe_path),
            "sha256": C.sha256_file(probe_path),
            "content_sha256": probe["content_sha256"],
        },
        "motion_sha256": frame["motion_sha256"],
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": False,
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "plant_contract_match": True,
        "control_step_action_delay_runtime": {
            "schema_version": 1,
            "kind": "whole_body_tracking.policy_control_step_action_delay_receipt",
            "num_envs": 1,
            "initialized_env_count": 1,
            "contract": template["runtime_plant"]["control_step_action_delay"],
            "lag_histogram": {"0": 1},
        },
        "active_terminations": list(C.REQUIRED_TERMINATIONS),
        "requested_duration_s": duration,
        "completed_duration_s": duration,
        "completed_policy_steps": frame["task_close_ticks"],
        "completed_physics_steps": frame["task_close_ticks"]
        * template["runtime_plant"]["control_decimation"],
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
        "minimum_root_z_m": 0.86,
        "maximum_root_tilt_rad": 0.12,
        "both_feet_contact_fraction": 1.0,
        "joint_safety_telemetry": joint,
        "screenshots": screenshots,
    }
    live = {**unsigned, "content_sha256": C.canonical_sha256(unsigned)}
    path = tmp_path / "live.json"
    path.write_bytes(C.canonical_bytes(live))
    return live, path


def test_probe_input_is_exact_frame0_zero_velocity_hold_derivation(tmp_path: Path):
    frame, template, probe, _path = _derive(tmp_path)
    assert probe["physical_ready"] == frame["frame0"]
    assert probe["hold_candidate"]["hold_qdes_joint_pos_rad"] == frame["frame0"]["joint_pos_rad"]
    assert probe["required_next_gate"]["exact_policy_steps"] == 200
    for base, scale, action, target in zip(
        template["runtime_plant"]["default_joint_pos_rad"],
        template["runtime_plant"]["action_scale_rad"],
        probe["hold_candidate"]["normalized_actor_action"],
        frame["frame0"]["joint_pos_rad"],
    ):
        assert base + scale * action == pytest.approx(target, abs=2.0e-7)
    assert probe["physical_ready"]["joint_vel_radps"] == [0.0] * 31


def test_exact_live_receipt_accepts_actual_safety_evidence(tmp_path: Path):
    frame, template, probe, probe_path = _derive(tmp_path)
    live, live_path = _live(tmp_path, frame, template, probe, probe_path)
    file_sha, content_sha = C.validate_live_receipt(
        live,
        raw_path=live_path,
        probe_path=probe_path,
        probe_file_sha256=C.sha256_file(probe_path),
        probe_content_sha256=probe["content_sha256"],
        frame0_artifact=frame,
        joint_names=template["robot"]["joint_names"],
        control_decimation=template["runtime_plant"]["control_decimation"],
        control_step_action_delay=template["runtime_plant"]["control_step_action_delay"],
    )
    assert file_sha == C.sha256_file(live_path)
    assert content_sha == live["content_sha256"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("completed_policy_steps", 199),
        lambda value: value.__setitem__("generic_terminated", True),
        lambda value: value["joint_safety_telemetry"].__setitem__(
            "substep_actual_hard_edge_joint_count", 1
        ),
        lambda value: value["joint_safety_telemetry"].__setitem__(
            "final_minimum_hard_gap_rad", 0.0
        ),
    ),
)
def test_live_receipt_tamper_never_becomes_pass(tmp_path: Path, mutation):
    frame, template, probe, probe_path = _derive(tmp_path)
    live, live_path = _live(tmp_path, frame, template, probe, probe_path)
    unsigned = copy.deepcopy(live)
    unsigned.pop("content_sha256")
    mutation(unsigned)
    bad = {**unsigned, "content_sha256": C.canonical_sha256(unsigned)}
    live_path.write_bytes(C.canonical_bytes(bad))
    with pytest.raises(C.ReceiptError, match="live receipt"):
        C.validate_live_receipt(
            bad,
            raw_path=live_path,
            probe_path=probe_path,
            probe_file_sha256=C.sha256_file(probe_path),
            probe_content_sha256=probe["content_sha256"],
            frame0_artifact=frame,
            joint_names=template["robot"]["joint_names"],
            control_decimation=template["runtime_plant"]["control_decimation"],
            control_step_action_delay=template["runtime_plant"]["control_step_action_delay"],
        )


def test_final_publication_is_canonical_and_no_clobber(tmp_path: Path):
    value = {"schema_version": 1, "kind": "fixture", "finite": 1.0}
    payload = C.canonical_bytes(value) + b"\n"
    pin = C._write_new(tmp_path, "receipt.json", payload)
    assert (tmp_path / "receipt.json").read_bytes() == payload
    assert pin["sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(C.ReceiptError, match="no-clobber"):
        C._write_new(tmp_path, "receipt.json", b"different\n")
    assert (tmp_path / "receipt.json").read_bytes() == payload


def test_wrapper_command_reuses_nominal_hold_for_exact_four_seconds(tmp_path: Path):
    command = R._live_command(
        python="/workspace/hope_isaac_venv/bin/python",
        device="cuda:2",
        probe_path=tmp_path / "probe.json",
        probe_sha256="a" * 64,
        live_path=tmp_path / "live.json",
        screenshot_dir=tmp_path / "screenshots",
        duration_s=4.0,
    )
    assert command[1] == str(R.PROBE_FILE)
    assert command[command.index("--num-envs") + 1] == "1"
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--duration-s") + 1] == "4"
    assert "--nominal-hold" in command
    assert "--motion-file" not in command
    assert "--screenshot-dir" in command
