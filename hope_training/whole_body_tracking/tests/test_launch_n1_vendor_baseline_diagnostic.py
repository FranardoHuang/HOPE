from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_n1_vendor_baseline_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("n1_vendor_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
VENDOR_CONTRACT_SHA = "9" * 64


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _spec(tmp_path: Path, *, seed: int, stage: str) -> dict:
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    isaac_python = tmp_path / "python.sh"
    isaac_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    isaac_python.chmod(0o755)
    namespace_parent = tmp_path / "runs"
    namespace_parent.mkdir(exist_ok=True)
    namespace = namespace_parent / f"vendor-seed{seed}-{stage}"
    budget = {
        "smoke": (1, 2, 1),
        "probe": (4096, 5, 1),
        "long": (4096, 20_001, 100),
    }[stage]
    return {
        "schema_version": 1,
        "kind": L.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(isaac_python),
        },
        "action_id": "bh_loop_c",
        "scope": "upper",
        "bundle": {"path": "bundle.json", "sha256": "b" * 64},
        "policy_contract_sha256": "c" * 64,
        "reward_profile": L.REWARD_PROFILE,
        "expected_effective_reward_recipe_sha256": "d" * 64,
        L.VENDOR_CONTRACT_FIELD: VENDOR_CONTRACT_SHA,
        "seed": seed,
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "gpu": {
            "index": seed,
            "uuid": f"GPU-vendor-seed-{seed}",
            "owner": "Franco",
            "lock_path": f"/tmp/hope_lean_queue_gpu{seed}.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


def _bundle() -> dict:
    return {
        "motion": {"path": "motions/bh_loop_c_upper.npz"},
        "manifest": {"path": "manifest.json", "sha256": "e" * 64},
        "dynamic_ready": {
            "artifact": {"path": "ready.json", "sha256": "f" * 64},
            "nominal_hold_receipt": {
                "path": "hold.json",
                "sha256": "0" * 64,
            },
        },
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("stage", ["smoke", "probe"])
def test_exact_seed_and_stage_namespaces_are_accepted(
    tmp_path: Path, seed: int, stage: str
) -> None:
    normalized = L._validate_spec_document(_spec(tmp_path, seed=seed, stage=stage))
    assert normalized["seed"] == seed
    assert normalized["stage"] == stage
    assert Path(normalized["namespace"]).name == f"vendor-seed{seed}-{stage}"


def test_other_seed_and_non_exact_stage_are_refused(tmp_path: Path) -> None:
    bad_seed = _spec(tmp_path, seed=0, stage="smoke")
    bad_seed["seed"] = 3
    with pytest.raises(L.LaunchRefused, match="seed must be exactly"):
        L._validate_spec_document(bad_seed)

    bad_budget = _spec(tmp_path, seed=0, stage="probe")
    bad_budget["max_iterations"] = 6
    with pytest.raises(L.LaunchRefused, match="exactly 4096 envs / 5 updates"):
        L._validate_spec_document(bad_budget)

    bad_stage = _spec(tmp_path, seed=0, stage="smoke")
    bad_stage["stage"] = "canary"
    with pytest.raises(L.LaunchRefused, match="stage must be"):
        L._validate_spec_document(bad_stage)

    long_stage = _spec(tmp_path, seed=0, stage="long")
    with pytest.raises(L.LaunchRefused, match="vendor_probe_gate_receipt"):
        L._validate_spec_document(long_stage)

    missing_contract = _spec(tmp_path, seed=0, stage="smoke")
    del missing_contract[L.VENDOR_CONTRACT_FIELD]
    with pytest.raises(L.LaunchRefused, match="requires"):
        L._validate_spec_document(missing_contract)

    block_action = _spec(tmp_path, seed=0, stage="smoke")
    block_action["action_id"] = "bh_block"
    with pytest.raises(L.LaunchRefused, match="exactly bh_loop_c"):
        L._validate_spec_document(block_action)


def test_argv_selects_vendor_profile_without_mutating_task_owned_dr(
    tmp_path: Path,
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="smoke"))
    argv = L._build_training_argv(spec, _bundle())

    assert f"task={L.TASK_PROFILE_ID}" in argv
    assert "task=HOPEPingPongActionBall" not in argv
    assert "task.racket.action_ball_diagnostic_unauthorized=true" in argv
    assert "algo.policy.init_noise_std=0.02" in argv
    forbidden = (
        "stable_ready_plant",
        "push.enable",
        "push_robot",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
        "task.rewards.motion_scale=",
        "task.rewards.racket_position_weight=",
        "task.rewards.racket_velocity_weight=",
        "task.rewards.racket_normal_weight=",
    )
    assert not any(fragment in item for item in argv for fragment in forbidden)


def test_legacy_dynamic_ready_without_vendor_contract_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            {"sources": {"stable_motion": {"sha256": "a" * 64}}},
        ),
    )
    with pytest.raises(L.LaunchRefused, match="legacy bundle refused"):
        L._validate_vendor_runtime_binding(
            tmp_path, "a" * 40, _bundle(), VENDOR_CONTRACT_SHA
        )


def test_dynamic_ready_vendor_contract_must_match_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = {
        "sources": {
            "runtime_training_contract": {"sha256": VENDOR_CONTRACT_SHA}
        }
    }
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            artifact,
        ),
    )
    binding = L._validate_vendor_runtime_binding(
        tmp_path, "a" * 40, _bundle(), VENDOR_CONTRACT_SHA
    )
    assert binding["runtime_training_contract_sha256"] == VENDOR_CONTRACT_SHA

    with pytest.raises(L.LaunchRefused, match="differs from spec"):
        L._validate_vendor_runtime_binding(
            tmp_path, "a" * 40, _bundle(), "8" * 64
        )


def test_vendor_bundle_gate_rejects_v1_and_accepts_v2_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    motion_sha = "e" * 64
    candidate = {
        "schema_version": 2,
        "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
        "action_id": "bh_loop_c",
        "robot": {"family": "AgiBot A3"},
        "sources": {
            "stable_motion": {
                "frame_index": 0,
                "sha256": motion_sha,
            }
        },
        "runtime_plant": {},
        "required_next_gate": {
            "kind": L._B.NOMINAL_HOLD_RECEIPT_KIND,
            "zero_terminal_required": ["joint_actual_forbidden"],
        },
    }
    candidate["content_sha256"] = L._B._canonical_ascii_sha256(candidate)
    artifact_pin = {"path": "ready.v2.json", "sha256": "f" * 64}
    receipt = {
        "schema_version": 1,
        "kind": L._B.NOMINAL_HOLD_RECEIPT_KIND,
        "verdict": "PASS",
        "action_id": "bh_loop_c",
        "motion_sha256": motion_sha,
        "plant_contract_match": True,
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
        "artifact": {
            "sha256": artifact_pin["sha256"],
            "content_sha256": candidate["content_sha256"],
        },
        "active_terminations": ["joint_actual_forbidden"],
    }
    receipt["content_sha256"] = L._B.canonical_sha256(receipt)
    values = iter(
        (
            (artifact_pin, candidate),
            ({"path": "hold.json", "sha256": "0" * 64}, receipt),
        )
    )
    monkeypatch.setattr(
        L._B, "_load_tracked_json", lambda *args, **kwargs: next(values)
    )
    result = L._validate_vendor_dynamic_ready(
        tmp_path,
        "a" * 40,
        {
            "artifact": artifact_pin,
            "nominal_hold_receipt": {
                "path": "hold.json",
                "sha256": "0" * 64,
            },
        },
        action_id="bh_loop_c",
        motion_sha256=motion_sha,
    )
    assert result["artifact"] == artifact_pin

    legacy = dict(candidate)
    legacy["schema_version"] = 1
    legacy["kind"] = "agibot_a3_action_dynamic_ready_candidate_v1"
    legacy["content_sha256"] = L._B._canonical_ascii_sha256(
        {key: value for key, value in legacy.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (artifact_pin, legacy),
    )
    with pytest.raises(L.LaunchRefused, match="schema-v2"):
        L._validate_vendor_dynamic_ready(
            tmp_path,
            "a" * 40,
            {
                "artifact": artifact_pin,
                "nominal_hold_receipt": {
                    "path": "hold.json",
                    "sha256": "0" * 64,
                },
            },
            action_id="bh_loop_c",
            motion_sha256=motion_sha,
        )


def test_repo_real_old_bh_loop_dynamic_ready_is_rejected() -> None:
    checkout = Path(__file__).resolve().parents[3]
    relative = "configs/a3_dynamic_ready_20260730/bh_loop_c.dynamic_ready.v1.json"
    artifact = checkout / relative
    pin = {
        "path": relative,
        "sha256": L._B.sha256_file(artifact),
    }
    bundle = _bundle()
    bundle["dynamic_ready"]["artifact"] = pin
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(L.LaunchRefused, match="differs from spec"):
        L._validate_vendor_runtime_binding(
            checkout, commit, bundle, VENDOR_CONTRACT_SHA
        )


def test_tracked_identity_resolves_materialized_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    manifest_path = checkout / L.VENDOR_IDENTITY_MANIFEST_SOURCE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert L._B.sha256_file(manifest_path) == L.VENDOR_IDENTITY_MANIFEST_SHA256
    for source_pin in manifest["sources"].values():
        assert (
            L._B.sha256_file(checkout / source_pin["path"])
            == source_pin["sha256"]
        )
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            manifest,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (args[2], checkout / args[2]["path"]),
    )

    validated = L._validate_vendor_identity_manifest(checkout, "a" * 40)
    assert validated == {
        "manifest": {
            "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
            "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
        },
        "runtime_training_contract_sha256": manifest[
            "runtime_materialization"
        ]["training_contract_sha256"],
    }


def test_tracked_identity_still_refuses_awaiting_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    manifest_path = checkout / L.VENDOR_IDENTITY_MANIFEST_SOURCE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "awaiting_runtime_materialization"
    manifest["runtime_materialization"]["training_contract_sha256"] = None
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            manifest,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (args[2], checkout / args[2]["path"]),
    )

    with pytest.raises(L.LaunchRefused, match="awaiting exact runtime"):
        L._validate_vendor_identity_manifest(checkout, "a" * 40)


def test_vendor_wrapper_replaces_legacy_robot_source_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(L._B, "sha256_file", lambda path: "7" * 64)
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda root, commit, pin, **kwargs: (pin, root / pin["path"]),
    )

    sources = L._validate_runtime_sources(checkout, "a" * 40)
    assert sources["vendor A3 robot source"]["path"] == L.ROBOT_SOURCE
    assert "historical N1 stable-ready robot source" not in sources


def test_actual_authority_receipt_pin_matches_materialized_file() -> None:
    checkout = Path(__file__).resolve().parents[3]
    authority_module = L._load_vendor_authority_module(checkout)
    receipt_path = checkout / authority_module.RECEIPT_REPO_PATH
    assert L.VENDOR_AUTHORITY_RECEIPT_SHA256 is not None
    assert L.VENDOR_AUTHORITY_RECEIPT_SHA256 == L._B.sha256_file(receipt_path)


def test_actual_authority_loader_and_full_candidate_validator_are_both_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []

    class AuthorityError(RuntimeError):
        pass

    def load(receipt_path, **kwargs):
        calls.append(("load", kwargs))
        return {
            "receipt_path": "authority.json",
            "receipt_sha256": "a" * 64,
            "runtime_training_contract": {
                "path": "contract.json",
                "sha256": VENDOR_CONTRACT_SHA,
                "schema_version": 3,
            },
            "runtime_plant_identity": {"authority": True},
            "verified_vendor_runtime": {
                "action_id": "bh_loop_c",
                "motion_sha256": "d" * 64,
            },
        }

    def validate(candidate, authority):
        calls.append(("validate", candidate))
        assert authority["runtime_plant_identity"] == {"authority": True}
        return {"full_candidate_plant": True}

    module = SimpleNamespace(
        RECEIPT_REPO_PATH="configs/authority.json",
        VendorRuntimeAuthorityError=AuthorityError,
        load_and_validate_vendor_runtime_authority=load,
        validate_candidate_runtime_plant_against_vendor_authority=validate,
    )
    monkeypatch.setattr(L, "VENDOR_AUTHORITY_RECEIPT_SHA256", "a" * 64)
    monkeypatch.setattr(L, "_load_vendor_authority_module", lambda root: module)
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.v2.json", "sha256": "b" * 64},
            {
                "schema_version": 2,
                "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
                "action_id": "bh_loop_c",
                "sources": {
                    "stable_motion": {"sha256": "d" * 64}
                },
            },
        ),
    )

    result = L._validate_actual_vendor_authority(
        tmp_path,
        "c" * 40,
        _bundle(),
        VENDOR_CONTRACT_SHA,
    )
    assert [name for name, _value in calls] == ["load", "validate"]
    load_kwargs = calls[0][1]
    assert load_kwargs["expected_receipt_sha256"] == "a" * 64
    assert (
        load_kwargs["expected_runtime_training_contract_sha256"]
        == VENDOR_CONTRACT_SHA
    )
    assert load_kwargs["launch_commit"] == "c" * 40
    assert result["runtime_plant_identity"] == {"full_candidate_plant": True}

    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.v2.json", "sha256": "b" * 64},
            {
                "schema_version": 2,
                "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
                "action_id": "bh_block",
                "sources": {
                    "stable_motion": {"sha256": "e" * 64}
                },
            },
        ),
    )
    with pytest.raises(L.LaunchRefused, match="bh_loop_c authority"):
        L._validate_actual_vendor_authority(
            tmp_path,
            "c" * 40,
            _bundle(),
            VENDOR_CONTRACT_SHA,
        )


def test_host_plan_binds_vendor_profile_and_single_gpu_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _spec(tmp_path, seed=2, stage="probe")
    spec_path = tmp_path / "run.json"
    spec_path.write_bytes(_canonical(document))

    monkeypatch.setattr(
        L._B,
        "_verify_clean_source",
        lambda checkout, commit: {
            "checkout": str(checkout),
            "commit_sha": commit,
        },
    )
    monkeypatch.setattr(
        L._B,
        "_validate_runtime_sources",
        lambda checkout, commit: {
            "N1 vendor diagnostic launcher": {
                "path": L.LAUNCHER_SOURCE,
                "sha256": "1" * 64,
            },
            "N1 diagnostic safety base": {
                "path": L.BASE_LAUNCHER_SOURCE,
                "sha256": "2" * 64,
            },
            f"immutable task profile {L.TASK_PROFILE_ID}": {
                "path": L.TASK_PROFILE_SOURCE,
                "sha256": "3" * 64,
            },
            "vendor A3 robot source": {
                "path": L.ROBOT_SOURCE,
                "sha256": "4" * 64,
            },
            "vendor runtime training-contract identity manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "vendor runtime authority validator": {
                "path": L.VENDOR_AUTHORITY_MODULE_SOURCE,
                "sha256": "5" * 64,
            },
        },
    )
    monkeypatch.setattr(
        L._B,
        "_validate_runtime_asset_environment",
        lambda: {"fixture": True},
    )
    monkeypatch.setattr(
        L._B,
        "_validate_bundle",
        lambda *args, **kwargs: copy.deepcopy(_bundle()),
    )
    monkeypatch.setattr(
        L._B, "_check_rsl_namespace_available", lambda *args: None
    )
    monkeypatch.setattr(
        L,
        "_validate_vendor_identity_manifest",
        lambda checkout, commit: {
            "manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "runtime_training_contract_sha256": VENDOR_CONTRACT_SHA,
        },
    )
    monkeypatch.setattr(
        L,
        "_validate_actual_vendor_authority",
        lambda checkout, commit, bundle, contract_sha: {
            "receipt_path": "authority.json",
            "receipt_sha256": "6" * 64,
            "runtime_training_contract": {
                "path": "training_contract.json",
                "sha256": contract_sha,
                "schema_version": 3,
            },
            "runtime_plant_identity": {"fixture": True},
        },
    )
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            {"sources": {"stable_motion": {"sha256": "a" * 64}}},
        ),
    )
    with pytest.raises(L.LaunchRefused, match="legacy bundle refused"):
        L.build_plan(spec_path)

    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            {
                "sources": {
                    "runtime_training_contract": {
                        "sha256": VENDOR_CONTRACT_SHA
                    }
                }
            },
        ),
    )

    monkeypatch.setattr(
        L,
        "_validate_vendor_identity_manifest",
        lambda checkout, commit: {
            "manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "runtime_training_contract_sha256": "8" * 64,
        },
    )
    with pytest.raises(L.LaunchRefused, match="tracked authority"):
        L.build_plan(spec_path)

    monkeypatch.setattr(
        L,
        "_validate_vendor_identity_manifest",
        lambda checkout, commit: {
            "manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "runtime_training_contract_sha256": VENDOR_CONTRACT_SHA,
        },
    )

    plan = L.build_plan(spec_path)
    payload = plan["canonical_payload"]
    assert plan["kind"] == L.CLAIM_KIND
    assert payload["kind"] == L.CLAIM_KIND
    assert payload["spec"]["gpu"]["index"] == 2
    assert payload["spec"]["gpu"]["require_empty"] is True
    assert payload["spec"]["stage"] == "probe"
    assert (
        payload["spec"][L.VENDOR_CONTRACT_FIELD]
        == VENDOR_CONTRACT_SHA
    )
    assert f"task={L.TASK_PROFILE_ID}" in payload["training_argv"]
    assert payload["formal_evidence_prohibited"] is True
    assert payload["curriculum_promotion_prohibited"] is True
    assert payload["vendor_runtime_authority"]["receipt_sha256"] == "6" * 64
    assert plan["launch_claim_sha256"] == L.canonical_sha256(payload)
