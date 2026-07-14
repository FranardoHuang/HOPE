from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_motion_s0_m0_exact_gmr_v2.py"
SPEC = importlib.util.spec_from_file_location("run_motion_s0_m0_exact_gmr_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GMR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GMR
SPEC.loader.exec_module(GMR)

S0 = ROOT / "configs" / "motion_exact_gmr_s0_prereg_20260714_v2.json"
M0 = ROOT / "configs" / "motion_exact_gmr_m0_prereg_20260714_v2.json"
RUNTIME = ROOT / "configs" / "motion_s0_m0_exact_gmr_runtime_20260714_v2.json"
SNAPSHOT = ROOT / "configs" / "motion_s0_m0_exact_gmr_pip_freeze_56b0f8af_v2.txt"


@pytest.mark.parametrize("plan_path", [S0, M0])
def test_real_v2_plans_pass_static_and_never_reuse_v1_root(plan_path):
    plan = GMR.validate_plan(plan_path, GMR.base.sha256_file(plan_path), ROOT)
    assert plan["attempt_version"] == 2
    assert plan["output_contract"]["output_root"].endswith("/exact_gmr_v2")
    assert "/exact_gmr_v1" not in plan["output_contract"]["output_root"]


def test_tracked_freeze_snapshot_is_complete_canonical_and_content_addressed():
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    python = runtime["execution_contract"]["python_environment"]
    snapshot = python["pip_freeze_snapshot"]
    payload = SNAPSHOT.read_bytes()
    assert len(payload) == 4702
    assert len(payload.splitlines()) == 234
    assert hashlib.sha256(payload).hexdigest() == (
        "56b0f8af9677b279bbb4925b6f49113f484dcb9ded1ed8d9bc56af71f304c694"
    )
    assert GMR.base.normalized_pip_freeze_bytes(payload.decode()) == payload
    assert snapshot["path"].endswith("pip_freeze_56b0f8af_v2.txt")
    assert snapshot["sha256"] == python["pip_freeze_sha256"]
    GMR._validate_snapshot_contract(python, ROOT)


def test_snapshot_reordering_truncation_and_hash_only_substitute_fail_closed(
    monkeypatch, tmp_path
):
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    python = runtime["execution_contract"]["python_environment"]
    lines = SNAPSHOT.read_text(encoding="utf-8").splitlines()
    alternate_path = copy.deepcopy(python)
    alternate_path["pip_freeze_snapshot"]["path"] = "configs/alternate.txt"
    with pytest.raises(GMR.base.ContractError, match="exact tracked path"):
        GMR._validate_snapshot_contract(alternate_path, ROOT)

    cases = {
        "reordered": ("\n".join(reversed(lines)) + "\n", "canonical"),
        "truncated": ("\n".join(lines[:-1]) + "\n", "snapshot and Python"),
        "hash-only": (python["pip_freeze_sha256"] + "\n", "snapshot and Python"),
    }
    for name, (payload, error) in cases.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(payload, encoding="utf-8")
        mutated = copy.deepcopy(python)
        mutated["pip_freeze_snapshot"].update(
            {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "line_count": len(path.read_bytes().splitlines()),
            }
        )
        monkeypatch.setattr(GMR.base, "verify_regular_file", lambda *_args, **_kwargs: path)
        with pytest.raises(GMR.base.ContractError, match=error):
            GMR._validate_snapshot_contract(mutated, ROOT)

    # A tracked relative mutation reaches the byte-level canonicality check.
    tracked = ROOT / "configs" / "motion_s0_m0_exact_gmr_pip_freeze_56b0f8af_v2.txt"
    reordered = copy.deepcopy(python)
    reordered["pip_freeze_snapshot"] = {
        "path": str(tracked.relative_to(ROOT)),
        "bytes": tracked.stat().st_size,
        "sha256": hashlib.sha256(tracked.read_bytes()).hexdigest(),
        "line_count": 233,
        "normalization": GMR.SNAPSHOT_NORMALIZATION,
    }
    with pytest.raises(GMR.base.ContractError, match="line_count"):
        GMR._validate_snapshot_contract(reordered, ROOT)


def test_v2_python_verifier_sanitizes_extensions_before_v1_fingerprint(monkeypatch):
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    plan = {"execution_contract": runtime["execution_contract"]}

    class ReachedBase(RuntimeError):
        pass

    def inspect_sanitized(received, _gmr_root):
        python = received["execution_contract"]["python_environment"]
        assert "pip_freeze_snapshot" not in python
        assert "direct_imports" not in python
        raise ReachedBase

    monkeypatch.setattr(GMR.base, "verify_python", inspect_sanitized)
    with pytest.raises(ReachedBase):
        GMR.verify_python_v2(plan, Path("/tmp/gmr"), ROOT)


def test_direct_import_contract_rejects_missing_module_and_escaped_metadata():
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    python = runtime["execution_contract"]["python_environment"]
    GMR._validate_direct_import_contract(python)

    missing = copy.deepcopy(python)
    missing["direct_imports"].pop("scipy")
    with pytest.raises(GMR.base.ContractError, match="exact five modules"):
        GMR._validate_direct_import_contract(missing)

    escaped = copy.deepcopy(python)
    escaped["direct_imports"]["numpy"]["metadata"]["path"] = "/tmp/METADATA"
    with pytest.raises(GMR.base.ContractError, match="escaped dist-info root"):
        GMR._validate_direct_import_contract(escaped)

    empty_version = copy.deepcopy(python)
    empty_version["direct_imports"]["torch"]["version"] = ""
    with pytest.raises(GMR.base.ContractError, match="version must be non-empty"):
        GMR._validate_direct_import_contract(empty_version)


def test_v2_tool_contract_must_bind_immutable_v1_base_consumer():
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    missing = copy.deepcopy(runtime)
    missing["tool_contract"].pop("base_consumer_v1")
    with pytest.raises(GMR.base.ContractError, match="tool contract field closure"):
        GMR._verify_v2_tool_contract(missing, ROOT)


def test_record_entry_must_bind_the_direct_module_origin(tmp_path):
    site = tmp_path / "site-packages"
    origin = site / "demo" / "__init__.py"
    dist = site / "demo-1.0.dist-info"
    origin.parent.mkdir(parents=True)
    dist.mkdir(parents=True)
    origin.write_bytes(b"value = 1\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(origin.read_bytes()).digest()).decode().rstrip("=")
    record = dist / "RECORD"
    record.write_text(
        f"demo/__init__.py,sha256={digest},{origin.stat().st_size}\n"
        "demo-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    assert GMR._record_origin_entry_matches(record, origin, dist)
    record.write_text("demo/__init__.py,sha256=wrong,10\n", encoding="utf-8")
    assert not GMR._record_origin_entry_matches(record, origin, dist)


def test_attempt_marker_and_v2_output_root_are_fail_closed(tmp_path):
    plan = json.loads(S0.read_text(encoding="utf-8"))
    plan.pop("attempt_version")
    path = tmp_path / "missing-attempt.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(GMR.base.ContractError, match="attempt 2"):
        GMR.validate_plan(path, GMR.base.sha256_file(path), ROOT)

    plan = json.loads(S0.read_text(encoding="utf-8"))
    plan["output_contract"]["output_root"] = (
        "/workspace/codexschema/motion_video_intake_20260713_s0/exact_gmr_v1"
    )
    path = tmp_path / "v1-root.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(GMR.base.ContractError, match="new no-clobber output root"):
        GMR.validate_plan(path, GMR.base.sha256_file(path), ROOT)


def test_v2_plan_duplicate_json_keys_fail_closed(tmp_path):
    text = S0.read_text(encoding="utf-8")
    duplicate = text.replace(
        '  "schema_version": 1,',
        '  "schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    path = tmp_path / "duplicate-plan.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(GMR.base.ContractError, match="duplicate JSON key.*schema_version"):
        GMR.validate_plan(path, GMR.base.sha256_file(path), ROOT)


def _consume_fixture(tmp_path: Path) -> tuple[dict, dict]:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    output_root = tmp_path / "output-v2"
    plan = {
        "attempt_version": 2,
        "batch_kind": "s0_static_high_press",
        "source_materialization": {"completion_manifest": {"sha256": "1" * 64}},
        "ignored_gmr_source": {"commit": "2" * 40},
        "_runtime_contract_binding": {
            "path": "configs/runtime-v2.json",
            "bytes": 1,
            "sha256": "5" * 64,
        },
        "execution_contract": {
            "timeout_seconds_per_asset": 10,
            "warmup_threshold_strict_lt": 0.0001,
            "warmup_max_rounds": 200,
            "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "converter_argv_template": [
                "{python}",
                "{converter}",
                "--gvhmr_pred_file",
                "{input}",
                "--robot",
                "agibot_a3",
                "--save_path",
                "{output}",
            ],
        },
        "a3_robot_contract": {},
        "batch_serialization_contract": {
            "mode": "advisory_flock_exclusive_across_s0_m0_consume",
            "lock_path": str(tmp_path / "shared-v2.consume.lock"),
            "lock_payload_utf8": GMR.SERIALIZATION_LOCK_PAYLOAD.decode().rstrip("\n"),
            "inspect_writes_lock": False,
            "consume_batches": ["s0_static_high_press", "m0_lateral_teachers"],
            "batch_order_dependency": False,
        },
        "output_contract": {
            "output_root": str(output_root),
            "result_suffix": ".exact_franco_donor_betas.gmr.pkl",
            "completion_manifest_filename": "completion_manifest.json",
        },
        "s0_semantic_guard": {"observed_ball_contact": None, "strike_effectiveness": None},
        "m0_stance_contract": None,
    }
    inspected = {
        "output_root": output_root,
        "canonical_mjcf": tmp_path / "a3.xml",
        "gmr": {"root": tmp_path, "converter": tmp_path / "converter.py"},
        "python": Path(sys.executable),
        "rows": [
            {
                "asset_id": "static_backhand_high_press",
                "frames": 3,
                "input": {"path": str(source), "bytes": 6, "sha256": "3" * 64},
                "input_path": str(source),
            }
        ],
    }
    return plan, inspected


def _install_successful_converter(monkeypatch, tmp_path, inspected):
    monkeypatch.setattr(GMR, "inspect_plan", lambda *_: inspected)
    monkeypatch.setattr(GMR.base, "verify_gmr_source", lambda *_: inspected["gmr"])
    monkeypatch.setattr(GMR.base, "verify_materialization", lambda *_: inspected["rows"])
    monkeypatch.setattr(
        GMR.base, "verify_tree_contract", lambda *_: inspected["canonical_mjcf"]
    )
    monkeypatch.setattr(GMR.base, "verify_a3_orders_and_sites", lambda *_: None)
    monkeypatch.setattr(GMR.base, "load_gmr_payload", lambda *_: {"fps": 30.0})

    def fake_auditor(plan, python, auditor, output, log, audit, frames, env):
        audit.write_text("{}\n", encoding="utf-8")
        return {"warmup": {"rounds": 1, "max_dq": 0.0}}

    monkeypatch.setattr(GMR.base, "run_auditor", fake_auditor)

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--save_path") + 1])
        output.write_bytes(b"gmr")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(GMR.subprocess, "run", fake_run)


def _install_test_serialization_path(monkeypatch, plan):
    path = plan["batch_serialization_contract"]["lock_path"]
    monkeypatch.setattr(GMR, "SERIALIZATION_LOCK_PATH", path)


def test_serialization_contract_rejects_order_dependency_and_path_drift(tmp_path):
    plan, _ = _consume_fixture(tmp_path)
    contract = plan["batch_serialization_contract"]
    original_path = GMR.SERIALIZATION_LOCK_PATH
    try:
        GMR.SERIALIZATION_LOCK_PATH = contract["lock_path"]
        GMR._validate_serialization_contract(contract)
        ordered = copy.deepcopy(contract)
        ordered["batch_order_dependency"] = True
        with pytest.raises(GMR.base.ContractError, match="serialization contract changed"):
            GMR._validate_serialization_contract(ordered)
        escaped = copy.deepcopy(contract)
        escaped["lock_path"] = str(tmp_path / "other.lock")
        with pytest.raises(GMR.base.ContractError, match="serialization contract changed"):
            GMR._validate_serialization_contract(escaped)
    finally:
        GMR.SERIALIZATION_LOCK_PATH = original_path


def test_consume_holds_shared_flock_and_initializes_exact_marker(monkeypatch, tmp_path):
    plan, _ = _consume_fixture(tmp_path)
    _install_test_serialization_path(monkeypatch, plan)
    lock_path = Path(plan["batch_serialization_contract"]["lock_path"])
    sentinel = tmp_path / "done"

    def assert_locked(*_args):
        fd = lock_path.open("r+")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fd.close()
        return sentinel

    monkeypatch.setattr(GMR, "_consume_locked", assert_locked)
    assert GMR.consume(plan, tmp_path / "plan.json", "4" * 64, ROOT) == sentinel
    assert lock_path.read_bytes() == GMR.SERIALIZATION_LOCK_PAYLOAD


def test_serialization_lock_symlink_and_payload_drift_fail_closed(monkeypatch, tmp_path):
    plan, _ = _consume_fixture(tmp_path)
    _install_test_serialization_path(monkeypatch, plan)
    lock_path = Path(plan["batch_serialization_contract"]["lock_path"])
    target = tmp_path / "target"
    target.write_bytes(GMR.SERIALIZATION_LOCK_PAYLOAD)
    lock_path.symlink_to(target)
    with pytest.raises(GMR.base.ContractError, match="cannot open v2 serialization lock"):
        with GMR._exclusive_batch_lock(plan):
            pass
    lock_path.unlink()
    lock_path.write_bytes(b"tampered\n")
    with pytest.raises(GMR.base.ContractError, match="payload drifted"):
        with GMR._exclusive_batch_lock(plan):
            pass


def test_v2_consume_publishes_completion_last_after_post_runtime_check(monkeypatch, tmp_path):
    plan, inspected = _consume_fixture(tmp_path)
    _install_test_serialization_path(monkeypatch, plan)
    _install_successful_converter(monkeypatch, tmp_path, inspected)
    checks: list[str] = []

    def verify(*_args):
        checks.append("python-v2")
        return inspected["python"]

    monkeypatch.setattr(GMR, "verify_python_v2", verify)
    published: list[Path] = []
    original = GMR.base.write_json_exclusive

    def recording_write(path, payload):
        published.append(path)
        original(path, payload)

    monkeypatch.setattr(GMR.base, "write_json_exclusive", recording_write)
    completion = GMR.consume(plan, tmp_path / "plan-v2.json", "4" * 64, ROOT)
    assert checks == ["python-v2"]
    assert published[-1] == completion
    payload = json.loads(completion.read_text(encoding="utf-8"))
    assert payload["attempt_version"] == 2


def test_post_runtime_drift_preserves_outputs_without_false_completion(monkeypatch, tmp_path):
    plan, inspected = _consume_fixture(tmp_path)
    _install_test_serialization_path(monkeypatch, plan)
    _install_successful_converter(monkeypatch, tmp_path, inspected)
    monkeypatch.setattr(
        GMR,
        "verify_python_v2",
        lambda *_: (_ for _ in ()).throw(GMR.base.ContractError("post-runtime drift")),
    )
    with pytest.raises(GMR.base.ContractError, match="post-runtime drift"):
        GMR.consume(plan, tmp_path / "plan-v2.json", "4" * 64, ROOT)
    assert (inspected["output_root"] / "outputs").is_dir()
    assert not (inspected["output_root"] / "completion_manifest.json").exists()
