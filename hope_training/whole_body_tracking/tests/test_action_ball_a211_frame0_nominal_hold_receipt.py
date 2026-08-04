"""Host tests for the exact A211 frame0 live-hold wrapper/consumer."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
TIMING_PATH = ROOT / (
    "configs/action_ball_n1_measured_20260803/fresh_592835dc_take061/"
    "rematerialized_1d5d9d44/tape/base_question.task_receipt.v5.5e09858672ac.json"
)
ARTIFACT_SOURCE_COMMIT = "5ed998f1e1526fa84dfc2198b064f9f8e6ab6068"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive(tmp_path: Path):
    frame = _json(ARTIFACT_PATH)
    timing = _json(TIMING_PATH)
    frame.pop("content_sha256")
    frame.pop("task_close_ticks")
    frame["schema_version"] = 2
    frame["kind"] = C._L.FRAME0_EXACT_ARTIFACT_KIND
    frame["source_kind"] = C._L.FRAME0_EXACT_SOURCE_KIND
    frame["timing_receipt"] = {
        "path": str(TIMING_PATH.relative_to(ROOT)),
        "sha256": C.sha256_file(TIMING_PATH),
    }
    frame["birth_horizon"] = {
        "schema_version": 1,
        "kind": "action_ball_frame0_dynamic_birth_horizon_v1",
        "derivation": "post_reset_coverage_plus_max_reset_wait_plus_ceil_pre_swing_wait",
        "timing_receipt_canonical_sha256": timing["canonical_sha256"],
        "policy_dt_s": 0.02,
        "post_reset_coverage_policy_ticks": 1,
        "max_reset_wait_policy_ticks": 25,
        "pre_swing_wait_s": timing["pre_swing_wait_s"],
        "pre_swing_wait_policy_ticks_ceil": 36,
        "required_policy_ticks": 62,
    }
    frame["content_sha256"] = C.canonical_sha256(frame)
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
        raw_nominal_path=tmp_path / "live.json",
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
            "policy_step": (0, 0, 1, 10, 62)[index],
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
    ticks = frame["birth_horizon"]["required_policy_ticks"]
    duration = ticks * frame["policy_dt_s"]
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
        "completed_policy_steps": ticks,
        "completed_physics_steps": ticks
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
    assert probe["required_next_gate"]["exact_policy_steps"] == 62
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


def test_wrapper_command_reuses_nominal_hold_for_exact_birth_horizon(tmp_path: Path):
    command = R._live_command(
        python="/workspace/hope_isaac_venv/bin/python",
        device="cuda:2",
        probe_path=tmp_path / "probe.json",
        probe_sha256="a" * 64,
        raw_nominal_path=tmp_path / "live.json",
        screenshot_dir=tmp_path / "screenshots",
        duration_s=1.24,
    )
    assert command[1] == str(R.PROBE_FILE)
    assert command[command.index("--num-envs") + 1] == "1"
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--duration-s") + 1] == "1.24"
    assert "--nominal-hold" in command
    assert "--motion-file" not in command
    assert "--screenshot-dir" in command


def test_raw_nominal_output_is_required_and_explicit():
    action = next(
        item
        for item in R._parser()._actions
        if item.dest == "raw_nominal_output"
    )
    assert action.required is True
    assert action.option_strings == ["--raw-nominal-output"]


def test_repo_outputs_are_relative_trackable_safe_and_no_clobber(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "repo"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)

    def trackable(_root, command, **_kwargs):
        assert command[:4] == ("check-ignore", "-q", "--no-index", "--")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(R._C, "_git", trackable)
    output, relative = R._fresh_repo_output(
        root, "receipts/raw.json", name="raw nominal output"
    )
    assert output == receipts / "raw.json"
    assert relative == "receipts/raw.json"

    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(R.RunError, match="no-clobber"):
        R._fresh_repo_output(
            root, "receipts/raw.json", name="raw nominal output"
        )
    with pytest.raises(R._C.ReceiptError, match="normalized relative path"):
        R._fresh_repo_output(
            root, "../raw.json", name="raw nominal output"
        )
    with pytest.raises(R._C.ReceiptError, match="normalized relative path"):
        R._fresh_repo_output(
            root, str(receipts / "absolute.json"), name="raw nominal output"
        )
    for alias in (
        "receipts//alias.json",
        "receipts/./alias.json",
        "receipts/alias.json/",
    ):
        with pytest.raises(R.RunError, match="explicit normalized relative path"):
            R._fresh_repo_output(root, alias, name="raw nominal output")

    (root / ".git").mkdir()
    with pytest.raises(R.RunError, match="Git metadata"):
        R._fresh_repo_output(
            root, ".git/raw.json", name="raw nominal output"
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(R.RunError, match="plain repository directory"):
        R._fresh_repo_output(
            root, "linked/raw.json", name="raw nominal output"
        )

    def ignored(_root, _command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(R._C, "_git", ignored)
    with pytest.raises(R.RunError, match="must not be Git-ignored"):
        R._fresh_repo_output(
            root, "receipts/ignored.json", name="raw nominal output"
        )


def _wrapper_harness(
    tmp_path: Path,
    monkeypatch,
    *,
    child_returncode: int = 0,
    publish_raw: bool = True,
    consumer_failure: bool = False,
    post_audit_failure: bool = False,
    mutate_raw_during_consume: bool = False,
    replace_final_during_post_audit: bool = False,
):
    root = tmp_path / "repo"
    inputs = root / "inputs"
    receipts = root / "receipts"
    inputs.mkdir(parents=True)
    receipts.mkdir()
    frame_path = inputs / "frame.json"
    template_path = inputs / "template.json"
    motion_path = inputs / "motion.npz"
    frame_path.write_bytes(b"frame-bytes")
    template_path.write_bytes(b"template-bytes")
    motion_path.write_bytes(b"motion-bytes")
    raw_path = receipts / "raw.json"
    final_path = receipts / "frame0.json"
    work_dir = tmp_path / "external-work"
    source_commit = "a" * 40
    artifact_source_commit = "b" * 40
    frame = {
        "policy_dt_s": 0.02,
        "birth_horizon": {"required_policy_ticks": 62},
    }
    template = {"runtime_plant": {}, "robot": {"joint_names": []}}
    state = {
        "commands": [],
        "consumer_args": [],
        "initial_verify": [],
        "candidate_verifications": [],
    }

    def fake_git(_root, command, *, binary=False):
        if command[0] == "check-ignore":
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if command[:2] == ("merge-base", "--is-ancestor"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[0] == "show":
            return SimpleNamespace(
                returncode=0,
                stdout=frame_path.read_bytes() if binary else "",
                stderr=b"" if binary else "",
            )
        raise AssertionError("unexpected Git command: %r" % (command,))

    def fake_clean(candidate_root, candidate_commit):
        state["initial_verify"].append((candidate_root, candidate_commit))

    def fake_tracked(_root, _relative, _sha, _commit, *, name):
        return {
            "frame0 artifact": frame_path,
            "plant template": template_path,
            "motion": motion_path,
        }[name]

    def fake_json(path, **_kwargs):
        return (frame if path == frame_path else template), path.read_bytes()

    def fake_child(command, **_kwargs):
        state["commands"].append(command)
        if publish_raw:
            Path(command[command.index("--nominal-hold-receipt-out") + 1]).write_bytes(
                b'{"kind":"raw-nominal"}'
            )
        screenshot_dir = Path(command[command.index("--screenshot-dir") + 1])
        screenshot_dir.mkdir()
        (screenshot_dir / "final.png").write_bytes(b"png")
        return SimpleNamespace(returncode=child_returncode)

    def fake_candidate_verify(candidate_root, candidate_commit, candidates):
        state["candidate_verifications"].append(
            (candidate_root, candidate_commit, tuple(candidates))
        )
        if post_audit_failure and len(candidates) == 2:
            if replace_final_during_post_audit:
                final_path.unlink()
                final_path.write_bytes(b"foreign replacement")
            raise R.RunError("post-consumer candidate audit failed")

    def fake_consume(namespace):
        state["consumer_args"].append(namespace)
        R._C.verify_exact_clean_source(
            Path(namespace.repo_root), namespace.probe_source_commit
        )
        if consumer_failure:
            raise C.ReceiptError("consumer rejected raw nominal evidence")
        payload = b"consumer\n"
        (Path(namespace.repo_root) / namespace.output).write_bytes(payload)
        if mutate_raw_during_consume:
            Path(namespace.live_receipt).write_bytes(b'{"kind":"changed-raw"}')
        return {
            "status": "PASS_RECEIPT_MATERIALIZED_COMMIT_REQUIRED",
            "receipt": {
                "path": namespace.output,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }

    monkeypatch.setattr(R._C, "_git", fake_git)
    monkeypatch.setattr(R._C, "verify_exact_clean_source", fake_clean)
    monkeypatch.setattr(R._C, "_tracked_input", fake_tracked)
    monkeypatch.setattr(R._C, "_strict_json", fake_json)
    monkeypatch.setattr(
        R._C, "derive_probe_input", lambda **_kwargs: {"kind": "probe"}
    )
    monkeypatch.setattr(R._C, "consume", fake_consume)
    monkeypatch.setattr(R, "_verify_exact_source_with_candidates", fake_candidate_verify)
    monkeypatch.setattr(R.subprocess, "run", fake_child)
    args = SimpleNamespace(
        repo_root=str(root),
        probe_source_commit=source_commit,
        artifact_source_commit=artifact_source_commit,
        frame0_artifact_path="inputs/frame.json",
        expected_frame0_artifact_sha256="d" * 64,
        plant_template_path="inputs/template.json",
        expected_plant_template_sha256="e" * 64,
        motion_path="inputs/motion.npz",
        expected_motion_sha256="f" * 64,
        device="cuda:0",
        python="python3",
        work_dir=str(work_dir),
        raw_nominal_output="receipts/raw.json",
        output="receipts/frame0.json",
    )
    return args, state, raw_path, final_path, work_dir


def test_wrapper_binds_producer_raw_path_directly_to_consumer_and_final_receipt(
    tmp_path: Path, monkeypatch
):
    args, state, raw_path, final_path, work_dir = _wrapper_harness(
        tmp_path, monkeypatch
    )
    result = R.run(args)
    command = state["commands"][0]
    consumer_args = state["consumer_args"][0]
    assert command[command.index("--nominal-hold-receipt-out") + 1] == str(raw_path)
    assert consumer_args.live_receipt == str(raw_path)
    assert consumer_args.output == "receipts/frame0.json"
    assert raw_path.is_file()
    assert final_path.is_file()
    assert result["raw_nominal_receipt"]["path"] == "receipts/raw.json"
    assert result["live_evidence_preserved"] == str(raw_path)
    assert {path.name for path in work_dir.iterdir()} == {
        "a211_frame0_nominal_hold.probe_input.v1.json",
        "screenshots",
    }
    verifications = state["candidate_verifications"]
    assert tuple(item[1] for item in verifications[0][2]) == (
        "receipts/raw.json",
    )
    assert tuple(item[1] for item in verifications[-1][2]) == (
        "receipts/raw.json",
        "receipts/frame0.json",
    )


def test_live_failure_preserves_raw_but_never_calls_consumer_or_writes_final(
    tmp_path: Path, monkeypatch
):
    args, state, raw_path, final_path, _work_dir = _wrapper_harness(
        tmp_path, monkeypatch, child_returncode=2
    )
    with pytest.raises(R.RunError, match="exit code 2"):
        R.run(args)
    assert raw_path.is_file()
    assert not final_path.exists()
    assert state["consumer_args"] == []
    assert tuple(
        item[1] for item in state["candidate_verifications"][0][2]
    ) == ("receipts/raw.json",)


def test_missing_or_rejected_raw_never_fabricates_consumer_receipt(
    tmp_path: Path, monkeypatch
):
    args, state, raw_path, final_path, _work_dir = _wrapper_harness(
        tmp_path, monkeypatch, publish_raw=False
    )
    with pytest.raises(R._C.ReceiptError, match="cannot inspect raw nominal receipt"):
        R.run(args)
    assert not raw_path.exists()
    assert not final_path.exists()
    assert state["consumer_args"] == []
    assert state["candidate_verifications"][0][2] == ()


def test_consumer_failure_leaves_raw_candidate_without_final_receipt(
    tmp_path: Path, monkeypatch
):
    args, state, raw_path, final_path, _work_dir = _wrapper_harness(
        tmp_path, monkeypatch, consumer_failure=True
    )
    with pytest.raises(C.ReceiptError, match="consumer rejected"):
        R.run(args)
    assert raw_path.is_file()
    assert not final_path.exists()
    assert len(state["consumer_args"]) == 1


def test_post_consumer_audit_failure_removes_fresh_final_receipt(
    tmp_path: Path, monkeypatch
):
    args, _state, raw_path, final_path, _work_dir = _wrapper_harness(
        tmp_path, monkeypatch, post_audit_failure=True
    )
    with pytest.raises(R.RunError, match="post-consumer candidate audit failed"):
        R.run(args)
    assert raw_path.is_file()
    assert not final_path.exists()


def test_raw_identity_change_during_consume_removes_final_receipt(
    tmp_path: Path, monkeypatch
):
    args, _state, raw_path, final_path, _work_dir = _wrapper_harness(
        tmp_path, monkeypatch, mutate_raw_during_consume=True
    )
    with pytest.raises(R.RunError, match="raw nominal receipt changed"):
        R.run(args)
    assert raw_path.read_bytes() == b'{"kind":"changed-raw"}'
    assert not final_path.exists()


def test_post_audit_refuses_to_delete_replaced_final_path(
    tmp_path: Path, monkeypatch
):
    args, _state, raw_path, final_path, _work_dir = _wrapper_harness(
        tmp_path,
        monkeypatch,
        post_audit_failure=True,
        replace_final_during_post_audit=True,
    )
    with pytest.raises(R.RunError, match="identity changed; cleanup refused"):
        R.run(args)
    assert raw_path.is_file()
    assert final_path.read_bytes() == b"foreign replacement"


def test_raw_and_consumer_outputs_cannot_alias(
    tmp_path: Path, monkeypatch
):
    args, state, _raw_path, final_path, work_dir = _wrapper_harness(
        tmp_path, monkeypatch
    )
    args.output = args.raw_nominal_output
    with pytest.raises(R.RunError, match="outputs must be distinct"):
        R.run(args)
    assert state["commands"] == []
    assert not final_path.exists()
    assert not work_dir.exists()


def test_failed_child_without_raw_audits_an_empty_candidate_set(
    tmp_path: Path, monkeypatch
):
    args, state, raw_path, final_path, _work_dir = _wrapper_harness(
        tmp_path, monkeypatch, child_returncode=1, publish_raw=False
    )
    with pytest.raises(R.RunError, match="exit code 1"):
        R.run(args)
    assert not raw_path.exists()
    assert not final_path.exists()
    assert state["candidate_verifications"][0][2] == ()


def test_generated_candidate_audit_allows_only_exact_untracked_paths_and_sources(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "repo"
    scripts = root / "scripts"
    receipts = root / "receipts"
    scripts.mkdir(parents=True)
    receipts.mkdir()
    probe_file = scripts / "probe.py"
    consumer_file = scripts / "consumer.py"
    wrapper_file = scripts / "wrapper.py"
    for path, payload in (
        (probe_file, b"probe"),
        (consumer_file, b"consumer"),
        (wrapper_file, b"wrapper"),
    ):
        path.write_bytes(payload)
    raw_path = receipts / "raw.json"
    raw_path.write_bytes(b"raw")
    source_commit = "1" * 40
    state = {
        "records": ["?? receipts/raw.json"],
        "bad_source": False,
    }

    def fake_git(_root, command, *, binary=False):
        if command == ("rev-parse", "HEAD"):
            return SimpleNamespace(
                returncode=0, stdout=source_commit + "\n", stderr=""
            )
        if command == (
            "status", "--porcelain=v1", "-z", "--untracked-files=all",
        ):
            stdout = "\0".join(state["records"])
            if stdout:
                stdout += "\0"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if command[0] == "show" and binary:
            relative = command[1].split(":", 1)[1]
            payload = (root / relative).read_bytes()
            if state["bad_source"] and relative == "scripts/probe.py":
                payload = b"different-commit-bytes"
            return SimpleNamespace(returncode=0, stdout=payload, stderr=b"")
        raise AssertionError("unexpected Git command: %r" % (command,))

    monkeypatch.setattr(R, "PROBE_FILE", probe_file)
    monkeypatch.setattr(R, "CONSUMER_FILE", consumer_file)
    monkeypatch.setattr(R, "__file__", str(wrapper_file))
    monkeypatch.setattr(R._C, "_git", fake_git)
    candidate = ((raw_path, "receipts/raw.json"),)
    R._verify_exact_source_with_candidates(root, source_commit, candidate)

    state["records"] = ["?? receipts/raw.json", "?? surprise.txt"]
    with pytest.raises(R.RunError, match="changed outside"):
        R._verify_exact_source_with_candidates(root, source_commit, candidate)

    state["records"] = [" M scripts/probe.py", "?? receipts/raw.json"]
    with pytest.raises(R.RunError, match="changed outside"):
        R._verify_exact_source_with_candidates(root, source_commit, candidate)

    state["records"] = ["?? receipts/raw.json"]
    state["bad_source"] = True
    with pytest.raises(R.RunError, match="differs from exact source commit"):
        R._verify_exact_source_with_candidates(root, source_commit, candidate)

    state["bad_source"] = False
    raw_path.unlink()
    raw_path.symlink_to(wrapper_file)
    with pytest.raises(R._C.ReceiptError, match="non-symlink"):
        R._verify_exact_source_with_candidates(root, source_commit, candidate)
