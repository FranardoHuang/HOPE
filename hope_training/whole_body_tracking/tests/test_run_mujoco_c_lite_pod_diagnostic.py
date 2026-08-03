"""Structural and no-mutation tests for the native C-lite Pod runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/run_mujoco_c_lite_pod_diagnostic.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("_mujoco_c_lite_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, module, *, execute: bool = False, confirm: bool = False):
    paths = {}
    argv = []
    for flag, expected_flag, name in (
        ("plant-contract", "expected-plant-sha256", "plant.json"),
        ("robot-tape", "expected-robot-tape-sha256", "robot.json"),
        ("question", "expected-question-sha256", "question.json"),
        (
            "selected-rubber-manifest",
            "expected-selected-rubber-manifest-sha256",
            "manifest.json",
        ),
        ("mjcf", "expected-mjcf-sha256", "robot.xml"),
    ):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        paths[flag] = path
        argv.extend((f"--{flag}", str(path), f"--{expected_flag}", digest))
    argv.extend(("--output-dir", str(tmp_path / "new_run")))
    if execute:
        argv.append("--execute")
    if confirm:
        argv.append("--confirm-diagnostic-unauthorized")
    return module._parser().parse_args(argv), paths


def test_default_is_read_only_plan_and_binds_every_external_sha(tmp_path, capsys):
    module = _module()
    args, _paths = _args(tmp_path, module)
    assert (
        module.main(
            [
                "--plant-contract",
                str(args.plant_contract),
                "--expected-plant-sha256",
                args.expected_plant_sha256,
                "--robot-tape",
                str(args.robot_tape),
                "--expected-robot-tape-sha256",
                args.expected_robot_tape_sha256,
                "--question",
                str(args.question),
                "--expected-question-sha256",
                args.expected_question_sha256,
                "--selected-rubber-manifest",
                str(args.selected_rubber_manifest),
                "--expected-selected-rubber-manifest-sha256",
                args.expected_selected_rubber_manifest_sha256,
                "--mjcf",
                str(args.mjcf),
                "--expected-mjcf-sha256",
                args.expected_mjcf_sha256,
                "--output-dir",
                str(args.output_dir),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["kind"] == module.PLAN_KIND
    assert plan["mode"] == "plan"
    assert plan["workload"]["episode_steps"] == 2
    assert plan["workload"]["torch_device"] == "cpu"
    assert plan["diagnostic_unauthorized"] is True
    assert plan["formal_authorized"] is False
    assert set(plan["authorities"]) == {
        "plant_contract",
        "robot_tape",
        "question",
        "selected_rubber_manifest",
        "mjcf",
    }
    assert not args.output_dir.exists()


def test_execute_requires_confirmation_before_output_mutation(tmp_path, capsys):
    module = _module()
    args, _paths = _args(tmp_path, module, execute=True)
    argv = [
        "--plant-contract",
        str(args.plant_contract),
        "--expected-plant-sha256",
        args.expected_plant_sha256,
        "--robot-tape",
        str(args.robot_tape),
        "--expected-robot-tape-sha256",
        args.expected_robot_tape_sha256,
        "--question",
        str(args.question),
        "--expected-question-sha256",
        args.expected_question_sha256,
        "--selected-rubber-manifest",
        str(args.selected_rubber_manifest),
        "--expected-selected-rubber-manifest-sha256",
        args.expected_selected_rubber_manifest_sha256,
        "--mjcf",
        str(args.mjcf),
        "--expected-mjcf-sha256",
        args.expected_mjcf_sha256,
        "--output-dir",
        str(args.output_dir),
        "--execute",
    ]
    assert module.main(argv) == 2
    failure = json.loads(capsys.readouterr().err)
    assert "confirm-diagnostic-unauthorized" in failure["error"]
    assert not args.output_dir.exists()


def test_source_wires_real_authority_factory_and_exact_checkpoint_pair():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "MujocoN1DiagnosticVecEnv.from_authorities(" in source
    assert "enable_c_lite_reward=True" in source
    assert "diagnostic_episode_length=EPISODE_STEPS" in source
    assert "MujocoDiagnosticPPOTrainer(" in source
    assert "ResetBoundaryCheckpoint()" in source
    assert "checkpoint_api.save(checkpoint_path, source)" in source
    assert "ResetBoundaryCheckpoint().load(" in source
    assert '"_cold-child"' in source
    assert "subprocess.run(" in source
    assert "if cold_next != reference_next:" in source
    assert 'cold_result.get("transition_transcript") != reference_transcript' in source
    assert '"optimizer_state_sha256": _state_digest(' in source
    assert 'formal_authorized": False' in source
