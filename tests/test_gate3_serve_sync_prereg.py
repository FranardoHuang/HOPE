from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/gate3_serve_sync_prereg_20260712.json"
VALIDATOR_PATH = ROOT / "scripts/validate_gate3_serve_sync_prereg.py"
SPEC = importlib.util.spec_from_file_location("validate_gate3_serve_sync", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _load() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _current_source_fixture() -> dict:
    """Rebind only source bytes in memory; the preregistration stays frozen."""
    doc = copy.deepcopy(_load())
    for dependency in doc["planner_policy_source_dependencies"]:
        dependency["sha256"] = VALIDATOR.sha256_file(ROOT / dependency["path"])
    plan_gate = doc["plan_gate_dependency"]
    plan_gate["source_sha256"] = VALIDATOR.sha256_file(ROOT / plan_gate["source_path"])
    plan_gate["legacy_audit_sha256"] = VALIDATOR.sha256_file(
        ROOT / plan_gate["legacy_audit_path"]
    )
    for key in ("fake_ball_source", "vendor_sim_config"):
        record = doc["frame_contract"][key]
        record["sha256"] = VALIDATOR.sha256_file(ROOT / record["path"])
    return doc


def test_tracked_preregistration_fails_closed_after_reviewed_source_changes(capsys):
    digest = VALIDATOR.sha256_file(PREREG)
    assert VALIDATOR.main(
        [
            "--repo-root",
            str(ROOT),
            "--prereg",
            str(PREREG),
            "--expected-prereg-sha256",
            digest,
            "--mode",
            "design-check",
        ]
    ) == 2
    assert "actual source SHA changed" in capsys.readouterr().err


def test_design_passes_and_launch_reports_every_missing_binding(capsys, monkeypatch):
    doc = _current_source_fixture()
    monkeypatch.setattr(
        VALIDATOR,
        "EXPECTED_SOURCE_DEPENDENCIES",
        copy.deepcopy(doc["planner_policy_source_dependencies"]),
    )
    monkeypatch.setattr(
        VALIDATOR, "EXPECTED_PLAN_GATE_DEPENDENCY", copy.deepcopy(doc["plan_gate_dependency"])
    )
    monkeypatch.setattr(
        VALIDATOR, "EXPECTED_FRAME_CONTRACT", copy.deepcopy(doc["frame_contract"])
    )
    original_read_json = VALIDATOR.read_json

    def read_json(path):
        if Path(path).resolve() == PREREG.resolve():
            return copy.deepcopy(doc)
        return original_read_json(path)

    monkeypatch.setattr(VALIDATOR, "read_json", read_json)
    digest = VALIDATOR.sha256_file(PREREG)
    assert (
        VALIDATOR.main(
            [
                "--repo-root",
                str(ROOT),
                "--prereg",
                str(PREREG),
                "--expected-prereg-sha256",
                digest,
                "--mode",
                "design-check",
            ]
        )
        == 0
    )
    design = json.loads(capsys.readouterr().out)
    assert design == {
        "active_status_runtime_present": False,
        "frame_contract_ready": False,
        "launch_authorized": False,
        "machine_ack_runtime_present": False,
        "one_shot_serve_ready": False,
        "publisher_arm_runtime_present": False,
        "runtime_blocker_count": len(VALIDATOR.REQUIRED_BINDINGS),
        "source_dependencies_verified": 18,
        "status": "pass_design_only",
        "vendor_backend_status_runtime_present": False,
    }

    assert (
        VALIDATOR.main(
            [
                "--repo-root",
                str(ROOT),
                "--prereg",
                str(PREREG),
                "--expected-prereg-sha256",
                digest,
                "--mode",
                "launch-check",
            ]
        )
        == 1
    )
    blocked = capsys.readouterr().err
    assert "LAUNCH BLOCKED" in blocked
    for binding in VALIDATOR.REQUIRED_BINDINGS:
        assert f"MISSING {binding}\n" in blocked
    assert "ball=world base=odom" in blocked
    assert "READY_NO_BALL plus WAITING_BALL_READY" in blocked
    assert "BACKEND_READY_NO_BALL" in blocked
    assert "publisher cursor" in blocked
    assert "one_shot=true max_serves=1" in blocked


def test_reviewed_source_subset_is_exact_and_full_runtime_closure_stays_blocked():
    doc = _load()
    dependencies = doc["planner_policy_source_dependencies"]
    assert dependencies == VALIDATOR.EXPECTED_SOURCE_DEPENDENCIES
    assert len(dependencies) == 18
    assert [item["path"] for item in dependencies] == [
        "hope_ws/src/hope_planner/hope_planner/node.py",
        "hope_ws/src/hope_planner/hope_planner/node_runtime_contract.py",
        "hope_ws/src/hope_planner/hope_planner/flat_command_wire.py",
        "hope_ws/src/hope_planner/config/hope_planner.yaml",
        "hope_ws/src/hope_planner/config/hope_planner.sim.yaml",
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_planner_input.hpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_policy.hpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_frame_math.hpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
            "a3_pingpong/pp_reference_clock.hpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/"
            "a3_pingpong_main.cpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/robot_io/"
            "a3_aimrt_backend.hpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/robot_io/"
            "a3_aimrt_backend.cpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/config/"
            "a3_aimrt_config.pingpong_ros2body.yaml"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/config/"
            "a3_runtime_config.pingpong.yaml"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/robot_io/"
            "robot_io_backend.hpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_deploy/"
            "a3_policy_driver.hpp"
        ),
        (
            "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/"
            "a3_policy_driver.cpp"
        ),
        "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/CMakeLists.txt",
    ]
    drifted = []
    for dependency in dependencies:
        result = subprocess.run(
            [
                "git",
                "show",
                f"6d6b778aff4970f90c0e6df0e0ea63ce30fbe380:{dependency['path']}",
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert VALIDATOR.hashlib.sha256(result.stdout).hexdigest() == dependency["sha256"]
        if VALIDATOR.sha256_file(ROOT / dependency["path"]) != dependency["sha256"]:
            drifted.append(dependency["path"])
    assert "hope_ws/src/hope_planner/hope_planner/node.py" in drifted
    assert _load()["runtime_bindings"]["planner_runtime_dependency_closure_sha256"] is None
    assert _load()["runtime_bindings"]["runner_transitive_shared_library_closure_sha256"] is None


def test_log_fragments_are_diagnostic_only_and_never_authorize():
    doc = _load()
    diagnostic = doc["diagnostic_only_logs"]
    assert diagnostic["authorization_value"] is False
    assert doc["planner_logging_environment"]["may_contribute_to_authorization_guard"] is False
    assert doc["planner_logging_environment"]["required_exact"] == {
        "PYTHONUNBUFFERED": "1",
        "RCUTILS_LOGGING_USE_STDOUT": "1",
        "RCUTILS_LOGGING_BUFFERED_STREAM": "0",
    }
    authorization = doc["publisher_arm_contract"]["authorization_guard"]
    transitions = "\n".join(
        f"{item['state']} {item['next']} {item['guard']}"
        for item in doc["state_machine"]["success_path"]
    )
    for fragment in (diagnostic["runner_fragment"], diagnostic["planner_fragment"]):
        assert fragment not in authorization
        assert fragment not in transitions
    for forbidden in ("stdout", "stderr", "log fragment", "log marker"):
        assert forbidden not in authorization.lower()
        assert forbidden not in transitions.lower()
    assert "READY_NO_BALL" in authorization
    assert "WAITING_BALL_READY" in authorization
    assert "atomic no-replace ledger" in authorization


def test_state_graph_is_one_way_single_arm_and_terminal_absorbing():
    machine = _load()["state_machine"]
    success = machine["success_path"]
    expected_order = [
        "PLAN_ONLY",
        "RUNTIME_BINDINGS_VERIFIED",
        "WAIT_MACHINE_ACK",
        "ACK_ACCEPTED",
        "OWNED_DISARMED",
        "ARM_COMMITTED",
        "ONE_SHOT_ACTIVE",
        "TERMINAL_DISARMED_SUCCESS",
    ]
    assert [success[0]["state"]] + [edge["next"] for edge in success] == expected_order
    rank = {state: index for index, state in enumerate(expected_order)}
    assert all(rank[edge["state"]] < rank[edge["next"]] for edge in success)

    arm_edges = [
        edge
        for edge in success
        if edge["state"] == "OWNED_DISARMED" and edge["next"] == "ARM_COMMITTED"
    ]
    assert len(arm_edges) == 1
    assert "single-use arm token" in arm_edges[0]["guard"]
    assert "atomically consumes" in success[5]["guard"]
    assert "before every later sample" in success[6]["guard"]
    assert "before every later 300 Hz trajectory sample" in _load()[
        "publisher_arm_contract"
    ]["per_sample_guard"]
    assert "before the next publish" in _load()["publisher_arm_contract"][
        "per_sample_guard"
    ]

    failure_edges = machine["failure_edges"]
    assert {edge["state"] for edge in failure_edges} == set(VALIDATOR.NONTERMINAL_STATES)
    assert all(edge["next"] == "TERMINAL_DISARMED_FAILED" for edge in failure_edges)
    terminals = set(machine["terminal_states"])
    all_edges = success + failure_edges
    assert not any(edge["state"] in terminals for edge in all_edges)
    assert "absorbing" in machine["terminal_semantics"]
    assert "no outgoing transition" in machine["terminal_semantics"]


def test_tracked_fake_ball_is_explicitly_blocked_until_exact_one_shot_exists():
    doc = _load()
    contract = doc["one_shot_serve_contract"]
    assert contract["required_parameters"] == {
        "one_shot": True,
        "max_serves": 1,
        "rate_hz": 300.0,
        "trajectory_count": 1,
        "auto_reset": False,
    }
    assert contract["current_source_supports_contract"] is False
    assert contract["launch_effect"].startswith("BLOCKED_NO_PUBLISHER")
    for forbidden in ("reset", "ACK reuse", "second serve", "post-terminal publish"):
        assert forbidden in contract["publish_count_rule"]
    source = (
        ROOT / doc["frame_contract"]["fake_ball_source"]["path"]
    ).read_text(encoding="utf-8")
    assert "self._reset()" in source
    assert "self._serve_i += 1" in source
    for binding in (
        "fake_ball_publisher_one_shot_config_sha256",
        "fake_ball_publisher_trajectory_sha256",
        "fake_ball_publisher_arm_token_schema_sha256",
        "fake_ball_publisher_terminal_evidence_sha256",
    ):
        assert doc["runtime_bindings"][binding] is None
    prohibited = set(doc["prohibited"])
    assert "publish_without_consumed_single_use_arm_token" in prohibited
    assert "reuse_or_reset_one_shot_publisher" in prohibited
    assert "rearm_or_retry_after_terminal_state" in prohibited


def test_machine_ack_binds_causality_identity_content_and_hot_restart():
    ack = _load()["machine_ack_contract"]
    assert ack["transport"].endswith("stdout and stderr are excluded")
    assert ack["clock_domain"] == "same_host_CLOCK_MONOTONIC"
    assert ack["planner_state"] == "READY_NO_BALL"
    assert ack["runner_state"] == "WAITING_BALL_READY"
    assert ack["vendor_state"] == "BACKEND_READY_NO_BALL"
    for required in (
        "run_nonce",
        "session_nonce",
        "source_epoch",
        "base_sequence_anchor",
        "base_sequence_current",
        "base_source_age_ms",
        "base_lease_valid",
        "actor_base_ready",
        "pid",
        "pgid",
        "proc_start_ticks",
        "executable_sha256",
        "argv_sha256",
        "config_closure_sha256",
        "environment_sha256",
        "policy_model_sha256",
        "runtime_closure_sha256",
    ):
        assert required in ack["planner_required_fields"]
        assert required in ack["runner_required_fields"]
    for required in (
        "run_nonce",
        "session_nonce",
        "backend_session_nonce",
        "pid",
        "pgid",
        "proc_start_ticks",
        "vendor_mjcf_sha256",
        "vendor_plant_sha256",
        "environment_sha256",
        "runtime_closure_sha256",
    ):
        assert required in ack["vendor_required_fields"]
    assert "base_revocation_generation_anchor" in ack["runner_required_fields"]
    assert "base_revocation_generation_current" in ack["runner_required_fields"]
    assert "clock_sample_sequence" in ack["planner_required_fields"]
    assert "acknowledged_clock_sample_sequence" in ack["runner_required_fields"]
    assert "runner/vendor acknowledge the exact planner clock" in ack["joint_acceptance"]
    assert "not a frozen current sequence" not in ack["base_refresh_rule"]
    assert "immutable readiness evidence" in ack["base_refresh_rule"]
    assert "may remain equal" in ack["base_refresh_rule"]
    assert "planner, runner, or vendor backend" in ack["restart_rule"]
    assert "before the next publish" in ack["restart_rule"]


def test_post_arm_statuses_are_bounded_forward_and_observable_per_sample():
    doc = _load()
    active = doc["active_status_contract"]
    assert active["status_freshness_max_age_ms"] == 40
    assert active["post_arm_transition_deadline_ms"] == 60
    assert "first_publish_monotonic_ns" in active["post_arm_deadline_origin"]
    assert "can never slide or reset" in active["post_arm_deadline_origin"]
    assert active["publisher_active_state"] == "ONE_SHOT_ACTIVE"
    assert "PID, PGID, proc start ticks" in active["publisher_identity_rule"]
    assert "environment and runtime/transitive closure" in active[
        "publisher_identity_rule"
    ]
    assert "next_sample_index" in active["publisher_cursor_rule"]
    assert "terminal success requires next=N and last=N-1" in active[
        "publisher_cursor_rule"
    ]
    assert active["planner"]["prearm_state"] == "READY_NO_BALL"
    assert active["planner"]["postarm_states"] == ["BALL_OBSERVED", "COMMANDING"]
    assert active["runner"]["prearm_state"] == "WAITING_BALL_READY"
    assert active["runner"]["postarm_states"] == ["TRACKING", "ACTOR_ACTIVE"]
    assert active["vendor"]["prearm_state"] == "BACKEND_READY_NO_BALL"
    assert active["vendor"]["postarm_states"] == ["BACKEND_ONE_SHOT_ACTIVE"]
    assert "after the deadline" in active["transition_rule"]
    assert "base_sequence_anchor" in active["base_sequence_rule"]
    assert "base_sequence_current" in active["base_sequence_rule"]
    assert "base_lease_valid=true" in active["base_sequence_rule"]
    assert "base_revocation_generation_current" in active["runner_revocation_rule"]
    assert "before the next publish" in active["runner_revocation_rule"]
    assert "actor_base_ready must be true" in active["actor_base_ready_rule"]
    assert "runner latest z at or above base_low" in active["actor_base_ready_rule"]
    assert "recovery hold" in active["actor_base_ready_rule"]
    assert "prearm WAITING true" in active["runner_actor_runtime_ready_rule"]
    assert "ACTOR_ACTIVE true requires exactly one accepted active clip" in active[
        "runner_actor_runtime_ready_rule"
    ]
    ledger = doc["publisher_arm_contract"]["atomic_ledger_required_fields"]
    assert "vendor_pid_pgid_start_ticks" in ledger
    assert "publisher_pid_pgid_start_ticks" in ledger
    assert "publisher_pidfd_cgroup_identity_sha256" in ledger
    assert "publisher_environment_sha256" in ledger
    assert "publisher_runtime_closure_sha256" in ledger
    assert "publisher_machine_status_schema_sha256" in ledger
    assert "vendor_backend_session_nonce" in ledger
    assert "arm_commit_record_identity_sha256" in ledger
    assert "first_publish_record_identity_sha256" in ledger
    assert "arm_committed_monotonic_ns" not in ledger
    assert "base_sequence_anchor" in ledger
    assert "base_revocation_generation_anchor" in ledger
    per_sample = doc["publisher_arm_contract"]["per_sample_guard"]
    for required in (
        "vendor owned pidfd/cgroup liveness",
        "base_sequence_current",
        "base_source_age_ms",
        "base_lease_valid=true",
        "actor_base_ready=true",
        "runner_actor_runtime_ready=true",
        "base_revocation_generation_current",
        "exact supervisor-observed next/last sample indices",
    ):
        assert required in per_sample
    for binding in (
        "vendor_machine_status_schema_sha256",
        "vendor_pidfd_cgroup_identity_sha256",
        "vendor_backend_readiness_session_sha256",
        "fake_ball_publisher_environment_sha256",
        "fake_ball_publisher_machine_status_schema_sha256",
        "fake_ball_publisher_runtime_closure_sha256",
        "fake_ball_publisher_pidfd_cgroup_identity_sha256",
        "fake_ball_publisher_arm_commit_record_schema_sha256",
        "fake_ball_publisher_first_publish_record_schema_sha256",
        "vendor_environment_sha256",
    ):
        assert doc["runtime_bindings"][binding] is None
    arm = doc["publisher_arm_contract"]
    assert "arm_committed_monotonic_ns" in arm["arm_commit_record"]
    assert "ACK ledger pre-binds this record identity" in arm["arm_commit_record"]
    assert "after fsync it re-reads/verifies" in arm["first_publish_record"]
    assert "revalidates every live first-sample guard immediately before" in arm[
        "first_publish_record"
    ]


def test_world_odom_mismatch_and_unbound_frame_items_are_explicit_blockers():
    doc = _load()
    frame = doc["frame_contract"]
    assert frame["formal_common_frame_required"] is True
    assert frame["current_ball_frame"] == "world"
    assert frame["current_base_frame"] == "odom"
    assert frame["current_frames_match"] is False
    assert frame["frame_transform_bound"] is False
    assert frame["fake_ball_frame_parameter_bound"] is False
    assert frame["launch_effect"].startswith("BLOCKED_NO_PUBLISHER")
    assert VALIDATOR.sha256_file(ROOT / frame["fake_ball_source"]["path"]) == frame[
        "fake_ball_source"
    ]["sha256"]
    assert VALIDATOR.sha256_file(ROOT / frame["vendor_sim_config"]["path"]) == frame[
        "vendor_sim_config"
    ]["sha256"]
    assert _load()["runtime_bindings"]["fake_ball_publisher_frame_id_binding_sha256"] is None
    assert _load()["runtime_bindings"]["frame_transform_source_target_sha256"] is None
    assert 'self.declare_parameter("frame_id", "world")' in (
        ROOT / frame["fake_ball_source"]["path"]
    ).read_text(encoding="utf-8")
    assert (ROOT / frame["vendor_sim_config"]["path"]).read_text(
        encoding="utf-8"
    ).count("frame_id: odom") >= 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update({"launch_authorized": True}),
        lambda d: d.update({"schema_version": True}),
        lambda d: d["diagnostic_only_logs"].update({"authorization_value": True}),
        lambda d: d["diagnostic_only_logs"].update({"runner_fragment": "MOTION"}),
        lambda d: d["planner_logging_environment"]["required_exact"].update(
            {"RCUTILS_LOGGING_BUFFERED_STREAM": "1"}
        ),
        lambda d: d["machine_ack_contract"].update({"transport": "parse stdout markers"}),
        lambda d: d["machine_ack_contract"]["planner_required_fields"].remove(
            "session_nonce"
        ),
        lambda d: d["machine_ack_contract"].update({"restart_rule": "reuse old ready"}),
        lambda d: d["active_status_contract"].update(
            {"post_arm_transition_deadline_ms": 6000}
        ),
        lambda d: d["active_status_contract"].pop("runner_revocation_rule"),
        lambda d: d["active_status_contract"].pop("actor_base_ready_rule"),
        lambda d: d["active_status_contract"].pop("publisher_cursor_rule"),
        lambda d: d["active_status_contract"].update(
            {"post_arm_deadline_origin": "sliding now plus 60 ms"}
        ),
        lambda d: d["active_status_contract"].pop("runner_actor_runtime_ready_rule"),
        lambda d: d["publisher_arm_contract"].update(
            {"authorization_guard": "runner MOTION log marker"}
        ),
        lambda d: d["publisher_arm_contract"].pop("per_sample_guard"),
        lambda d: d["publisher_arm_contract"].pop("arm_commit_record"),
        lambda d: d["publisher_arm_contract"].pop("first_publish_record"),
        lambda d: d["state_machine"]["success_path"][2].update(
            {"guard": "stdout marker order is enough"}
        ),
        lambda d: d["one_shot_serve_contract"]["required_parameters"].update(
            {"max_serves": 2}
        ),
        lambda d: d["frame_contract"].update({"current_frames_match": True}),
        lambda d: d["frame_contract"].update({"frame_transform_bound": True}),
        lambda d: d["frame_contract"]["vendor_sim_config"].update(
            {"sha256": "0" * 64}
        ),
        lambda d: d["planner_policy_source_dependencies"].pop(),
        lambda d: d["planner_policy_source_dependencies"][0].update(
            {"sha256": "0" * 64}
        ),
        lambda d: d["runtime_bindings"].update(
            {"exact_supervisor_source_sha256": "0" * 64}
        ),
        lambda d: d["prohibited"].remove("pkill"),
    ],
)
def test_safety_relaxations_fail_closed(tmp_path: Path, mutation):
    doc = _load()
    mutation(doc)
    path = _write(tmp_path, doc)
    with pytest.raises(VALIDATOR.ContractError):
        VALIDATOR.validate_prereg(path, VALIDATOR.sha256_file(path))


def test_duplicate_keys_and_nonfinite_json_fail_closed(tmp_path: Path):
    text = PREREG.read_text(encoding="utf-8")
    duplicate = text.replace(
        '  "schema_version": 4,\n',
        '  "schema_version": 4,\n  "schema_version": 4,\n',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(VALIDATOR.ContractError, match="duplicate JSON key"):
        VALIDATOR.validate_prereg(
            duplicate_path, VALIDATOR.sha256_file(duplicate_path)
        )

    nonfinite = text.replace('"schema_version": 4', '"schema_version": NaN', 1)
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(VALIDATOR.ContractError, match="non-finite JSON constant"):
        VALIDATOR.validate_prereg(
            nonfinite_path, VALIDATOR.sha256_file(nonfinite_path)
        )


def test_actual_source_byte_change_fails_closed(tmp_path: Path, monkeypatch):
    dependencies = _current_source_fixture()["planner_policy_source_dependencies"]
    monkeypatch.setattr(
        VALIDATOR, "EXPECTED_SOURCE_DEPENDENCIES", copy.deepcopy(dependencies)
    )
    for dependency in dependencies:
        source = ROOT / dependency["path"]
        destination = tmp_path / dependency["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    VALIDATOR.validate_source_dependencies(tmp_path, dependencies)
    changed = tmp_path / dependencies[0]["path"]
    changed.write_bytes(changed.read_bytes() + b"\n# changed\n")
    with pytest.raises(VALIDATOR.ContractError, match="actual source SHA changed"):
        VALIDATOR.validate_source_dependencies(tmp_path, dependencies)


def test_containment_and_canonical_cli_path_fail_closed(tmp_path: Path, capsys):
    with pytest.raises(VALIDATOR.ContractError, match="contained relative path"):
        VALIDATOR._resolve_bound_repo_file(ROOT, "../outside", "test dependency")
    with pytest.raises(VALIDATOR.ContractError, match="contained relative path"):
        VALIDATOR._resolve_bound_repo_file(ROOT, str(PREREG), "test dependency")

    digest = VALIDATOR.sha256_file(PREREG)
    copied = tmp_path / PREREG.name
    copied.write_bytes(PREREG.read_bytes())
    assert (
        VALIDATOR.main(
            [
                "--repo-root",
                str(ROOT),
                "--prereg",
                str(copied),
                "--expected-prereg-sha256",
                digest,
                "--mode",
                "design-check",
            ]
        )
        == 2
    )
    assert "canonical file" in capsys.readouterr().err
    assert (
        VALIDATOR.main(
            [
                "--repo-root",
                str(tmp_path),
                "--prereg",
                str(copied),
                "--expected-prereg-sha256",
                digest,
                "--mode",
                "design-check",
            ]
        )
        == 2
    )
    assert "Git worktree top level" in capsys.readouterr().err


def test_validator_has_no_runtime_launch_wait_or_signal_path():
    source = VALIDATOR_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "Popen",
        "shell=True",
        "killpg",
        "os.kill",
        "signal.",
        "ros2 run",
        "check_output",
        "check_call",
    ):
        assert forbidden not in source
    assert "subprocess.run(" in source
    assert '["git", "--no-optional-locks", "-C", str(repo_root), *args]' in source
    assert 'env["GIT_OPTIONAL_LOCKS"] = "0"' in source
    assert "if args not in allowed:" in source
