from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_phase1_signed_face_c3d3_k100.py"
MANIFEST_PATH = ROOT / "configs" / "phase1_signed_face_c3d3_k100_execution_20260714.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("c3d3_k100_under_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load_module()


def _manifest():
    return R.load_manifest(MANIFEST_PATH, repo_root=ROOT)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _attestor_request(cell_id: str, tmp_path: Path) -> tuple[Path, dict]:
    manifest = _manifest()
    cell = manifest["cells"][cell_id]
    checkpoint_python = "/runtime/isaac/bin/python"
    evaluator_python = "/runtime/mjeval/bin/python"
    request = {
        "schema_version": 1,
        "request_id": f"c3d3-k100-{cell_id.lower()}-exact-v1",
        "status": R.A.REQUEST_STATUS,
        "human_owner": "Franco",
        "executor": "Codex",
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "source_checkout": {
            "path": str(ROOT),
            "commit": "a" * 40,
            "tree": "b" * 40,
            "clean_required": True,
        },
        "checkpoint": {
            "path": cell["checkpoint"]["path"],
            "bytes": 123,
            "sha256": cell["checkpoint"]["sha256"],
            "filename_iteration": 24,
            "embedded_iteration": 24,
            "training_contract_schema_version": 3,
            "training_contract_sha256": cell["adjacent_hard_contract"]["sha256"],
            "training_contract_lineage_exact": 1,
            "producer_claim": {
                "path": str(tmp_path / f"{cell_id}.producer_claim.json"),
                "file_sha256": "c" * 64,
                "canonical_sha256": cell["producer_claim_canonical_sha256"],
            },
        },
        "adjacent_hard_contract": {
            "path": cell["adjacent_hard_contract"]["path"],
            "bytes": 456,
            "sha256": cell["adjacent_hard_contract"]["sha256"],
            "plant_contract_sha256": "d" * 64,
        },
        "runtime": {
            "checkpoint_python": {
                "path": checkpoint_python,
                "resolved_path": "/usr/bin/python3.10",
                "resolved_sha256": "e" * 64,
                "fingerprint": {
                    "implementation": "CPython",
                    "version": "3.10 fixture",
                    "executable": checkpoint_python,
                    "packages": {"torch": "2.0-fixture"},
                },
            },
            "evaluator_python": {
                "path": evaluator_python,
                "resolved_path": "/usr/bin/python3.10",
                "resolved_sha256": "e" * 64,
                "fingerprint": {
                    "implementation": "CPython",
                    "version": "3.10 fixture",
                    "executable": evaluator_python,
                    "packages": {
                        "mujoco": "fixture",
                        "numpy": "fixture",
                        "onnx": "fixture",
                        "onnxruntime": "fixture",
                    },
                },
            },
        },
        "mjcf": {
            "path": "/runtime/a3.xml",
            "bytes": 789,
            "sha256": "f" * 64,
        },
        "output": {
            "root": str(Path(manifest["_attestor_manifest"]["output"]["root"]) / cell["checkpoint"]["sha256"])
        },
        "authorization": copy.deepcopy(R.A.ATTESTATION_AUTHORIZATION),
    }
    path = tmp_path / f"{cell_id}.attestor_request.json"
    _write_json(path, request)
    return path, request


def _file_spec(path: Path, *, content_sha: str | None = None) -> dict:
    value = {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 1,
        "file_sha256": R.sha256_file(path) if path.exists() else "1" * 64,
    }
    if content_sha is not None:
        value["content_sha256"] = content_sha
    return value


def _execution_request(tmp_path: Path) -> tuple[Path, dict]:
    manifest = _manifest()
    attestor_requests = {}
    attestations = {}
    env_yamls = {}
    for cell_id in R.CELL_ORDER:
        path, value = _attestor_request(cell_id, tmp_path)
        attestor_requests[cell_id] = _file_spec(path, content_sha=R.canonical_sha256(value))
        root = Path(value["output"]["root"])
        attestations[cell_id] = {
            "claim": {
                "path": str(root / manifest["_attestor_manifest"]["output"]["claim_basename"]),
                "bytes": 10,
                "file_sha256": "2" * 64,
                "content_sha256": "3" * 64,
            },
            "evidence": {
                "path": str(root / manifest["_attestor_manifest"]["output"]["evidence_basename"]),
                "bytes": 11,
                "file_sha256": "4" * 64,
                "content_sha256": "5" * 64,
            },
        }
        env_yamls[cell_id] = {
            "path": str(Path(manifest["cells"][cell_id]["checkpoint"]["path"]).parent / "params" / "env.yaml"),
            "bytes": 12,
            "file_sha256": "6" * 64,
        }
    request = {
        "schema_version": 1,
        "request_id": "c3d3-signed-k100-paired-execution-v1",
        "status": R.REQUEST_STATUS,
        "human_owner": "Franco",
        "executor": "Codex",
        "simulation_only": True,
        "real_robot_commands_forbidden": True,
        "manifest": _file_spec(MANIFEST_PATH),
        "attestor_requests": attestor_requests,
        "attestations": attestations,
        "env_yamls": env_yamls,
        "runtime": {
            "gpu_by_cell": {"C3": 1, "D3": 2},
            "isaac_activation": {
                "path": "/runtime/isaac/bin/activate",
                "bytes": 13,
                "file_sha256": "7" * 64,
            },
            "mjeval_activation": {
                "path": "/runtime/mjeval/bin/activate",
                "bytes": 14,
                "file_sha256": "8" * 64,
            },
            "kit_boot_lock": "/workspace/.kit_boot.lock",
        },
        "output": {"root": manifest["execution"]["output_root"]},
        "authorization": copy.deepcopy(R.REQUEST_AUTHORIZATION),
    }
    path = tmp_path / "pair_execution_request.json"
    _write_json(path, request)
    return path, request


def _load_execution_request(path: Path):
    manifest = _manifest()
    return R.load_request(path, MANIFEST_PATH, manifest, read_attestor_requests=True)


def test_frozen_manifest_binds_terminal_pair_attestor_and_same_k100():
    manifest = _manifest()
    assert R.sha256_file(RUNNER_PATH) == manifest["source_bindings"]["runner"]["sha256"]
    assert manifest["paired_l1_receipt"]["sha256"] == R.PAIR_SHA
    assert {
        cell: manifest["cells"][cell]["checkpoint"]["sha256"] for cell in R.CELL_ORDER
    } == R.CHECKPOINT_SHA_BY_CELL
    assert manifest["paper"]["schedule"]["file_sha256"] == (
        "f2777dcd02080ba68b839c76ea9d3f14c938457c9bc01b5692fe86ae59157ec7"
    )
    assert manifest["paper"]["activation"]["file_sha256"] == (
        "e0125b0e937655672e68ac79578c075e4cf8e99fc1cad5655bcb7e3e4a977bb4"
    )
    assert manifest["authorization"] == R.MANIFEST_AUTHORIZATION


def test_static_and_source_plan_are_read_only(tmp_path: Path):
    before = set(tmp_path.iterdir())
    for mode, status in (
        ("static-validate", "source_reviewed_runtime_request_and_attestations_required"),
        ("source-plan", "source_plan_only_runtime_request_absent_no_launch"),
    ):
        completed = subprocess.run(
            [sys.executable, str(RUNNER_PATH), mode], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["status"] == status
        assert result["writes_or_launches_performed"] is False
        assert result["l2_training_authorized"] is False
        assert result["real_robot_authorized"] is False
    assert set(tmp_path.iterdir()) == before


def test_exact_request_plan_binds_both_attestor_requests(tmp_path: Path):
    path, _ = _execution_request(tmp_path)
    request = _load_execution_request(path)
    assert set(request["_attestor_requests"]) == set(R.CELL_ORDER)
    assert request["runtime"]["gpu_by_cell"] == {"C3": 1, "D3": 2}
    assert request["_attestor_requests"]["C3"]["checkpoint"]["sha256"] == R.CHECKPOINT_SHA_BY_CELL["C3"]
    assert request["_attestor_requests"]["D3"]["checkpoint"]["sha256"] == R.CHECKPOINT_SHA_BY_CELL["D3"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["authorization"].__setitem__("l2_training_authorized", True),
        lambda value: value["authorization"].__setitem__("second_seed_authorized", True),
        lambda value: value["authorization"].__setitem__("signals_to_existing_processes_allowed", True),
        lambda value: value["runtime"]["gpu_by_cell"].__setitem__("D3", 1),
        lambda value: value["output"].__setitem__("root", "/tmp/reuse"),
        lambda value: value["env_yamls"]["C3"].__setitem__("path", "/tmp/env.yaml"),
        lambda value: value["attestations"]["D3"]["claim"].__setitem__("path", "/tmp/claim.json"),
        lambda value: value["attestor_requests"].pop("D3"),
    ),
)
def test_pair_request_rejects_relaxation_missing_binding_or_namespace_escape(tmp_path: Path, mutation):
    path, request = _execution_request(tmp_path)
    mutation(request)
    _write_json(path, request)
    with pytest.raises(R.ContractError):
        _load_execution_request(path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["checkpoint"].__setitem__("sha256", "0" * 64),
        lambda value: value["checkpoint"].__setitem__("embedded_iteration", 25),
        lambda value: value["checkpoint"].__setitem__("training_contract_lineage_exact", True),
        lambda value: value["checkpoint"]["producer_claim"].__setitem__("canonical_sha256", "0" * 64),
        lambda value: value["adjacent_hard_contract"].__setitem__("sha256", "0" * 64),
        lambda value: value["source_checkout"].__setitem__("path", "/tmp/foreign-eval-worktree"),
        lambda value: value["authorization"].__setitem__("judge_started", True),
    ),
)
def test_attestor_request_must_be_exact_terminal_binding(tmp_path: Path, mutation):
    path, request = _execution_request(tmp_path)
    c3_path = Path(request["attestor_requests"]["C3"]["path"])
    c3 = json.loads(c3_path.read_text(encoding="utf-8"))
    mutation(c3)
    _write_json(c3_path, c3)
    request["attestor_requests"]["C3"] = _file_spec(c3_path, content_sha=R.canonical_sha256(c3))
    _write_json(path, request)
    with pytest.raises(R.ContractError):
        _load_execution_request(path)


def test_runtime_face_contract_rejects_integer_sign_alias():
    exact = copy.deepcopy(R.RUNTIME_FACE_CONTRACT)
    R.validate_runtime_face_contract(exact)
    wrong = copy.deepcopy(exact)
    wrong["mount_normal_sign_per_clip"] = [1, -1]
    with pytest.raises(R.ContractError):
        R.validate_runtime_face_contract(wrong)


def test_judge_command_has_same_schedule_and_no_inexact_or_repaper_escape():
    command = R._judge_command(
        judge=Path("/eval/scripts/judge.sh"), staged_run=Path("/isolated/run"),
        checkpoint="/train/model_24.pt", gpu=1,
        schedule="/paper/signed.schedule.json", bank="/paper/exam.npz",
    )
    assert command[:4] == ["bash", "/eval/scripts/judge.sh", "/isolated/run", "/train/model_24.pt"]
    assert "--exam-bank" in command
    assert command[command.index("--noise-scales") + 1] == "0.0"
    assert "--exam-schedule-json /paper/signed.schedule.json" in command
    assert not any("allow-inexact" in part for part in command)
    assert "--schedule-k" not in command


def test_no_clobber_writer_preserves_first_artifact(tmp_path: Path):
    path = tmp_path / "claim.json"
    R._write_exclusive(path, {"version": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        R._write_exclusive(path, {"version": 2})
    assert path.read_bytes() == before


def test_plan_and_execute_require_runtime_request(capsys):
    with pytest.raises(R.ContractError, match="requires --request"):
        R.main(["plan"])
    with pytest.raises(R.ContractError, match="requires --request"):
        R.main(["execute"])
