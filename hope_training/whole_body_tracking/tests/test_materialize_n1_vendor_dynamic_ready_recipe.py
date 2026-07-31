"""Host-only tests for the narrow vendor dynamic-ready recipe wrapper."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_n1_vendor_dynamic_ready_recipe.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vendor_dynamic_ready_recipe_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


NEW_POLICY_SHA = "a" * 64
BINDING_SHA = "b" * 64


def _spec(tmp_path: Path) -> dict:
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    namespace = tmp_path / "a3vendor-dynamic-recipe-unit"
    return {
        "schema_version": L.SCHEMA_VERSION,
        "kind": L.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "c" * 40,
            "isaac_python": str(Path(sys.executable).resolve()),
        },
        "action_id": L.ACTION_ID,
        "bundle": dict(L.BUNDLE_PIN),
        "vendor_runtime_training_contract_sha256": (
            L.VENDOR_RUNTIME_CONTRACT_SHA256
        ),
        "stage": L.STAGE,
        "seed": L.SEED,
        "num_envs": L.NUM_ENVS,
        "max_iterations": L.MAX_ITERATIONS,
        "gpu": {
            "index": 0,
            "uuid": "GPU-unit-test",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu0.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


def _bundle() -> dict:
    return {
        "bundle": dict(L.BUNDLE_PIN),
        "action_id": L.ACTION_ID,
        "scope": L.SCOPE,
        "motion": {
            "path": "assets/motions/bh_loop_c.npz",
            "sha256": "d" * 64,
        },
        "manifest": {
            "path": "configs/bh_loop_c.manifest.json",
            "sha256": "e" * 64,
        },
        "dynamic_ready": {
            "artifact": {
                "path": "configs/bh_loop_c.dynamic_ready.json",
                "sha256": "f" * 64,
            },
            "nominal_hold_receipt": {
                "path": "configs/bh_loop_c.nominal_hold.json",
                "sha256": "1" * 64,
            },
        },
    }


def _vendor_inputs() -> dict:
    return {
        "bundle": _bundle(),
        "required_identity": {
            "runtime_training_contract_sha256": (
                L.VENDOR_RUNTIME_CONTRACT_SHA256
            )
        },
        "actual_authority": {
            "runtime_training_contract": {
                "sha256": L.VENDOR_RUNTIME_CONTRACT_SHA256
            }
        },
        "runtime_binding": {
            "runtime_training_contract_sha256": (
                L.VENDOR_RUNTIME_CONTRACT_SHA256
            )
        },
        "dynamic_ready_binding": {
            "schema_version": 2,
            "kind": "action_ball_dynamic_ready_runtime_binding_v2",
            "action_order": [L.ACTION_ID],
            "motion_sha256_per_action": ["d" * 64],
            "binding_sha256": BINDING_SHA,
        },
    }


def _write_canonical(path: Path, document: dict) -> None:
    path.write_bytes(L._S._canonical_bytes(document) + b"\n")


def _recipe_document(policy_sha: str = NEW_POLICY_SHA) -> dict:
    bootstrap = {
        "schema_version": 2,
        "action_count": 1,
        "action_order": [L.ACTION_ID],
        "ready_source": {"identity": {"binding_sha256": BINDING_SHA}},
    }
    return {
        "schema_version": 1,
        "kind": "action_ball_shared_ready_policy_recipe_materialization_v1",
        "action_count": 1,
        "action_order": [L.ACTION_ID],
        "policy_contract_sha256": policy_sha,
        "action_ball_ppo_runner_recipe": {
            "schema_version": 1,
            "sha256": policy_sha,
            "recipe": {"policy_initialization": bootstrap},
        },
        "policy_bootstrap": bootstrap,
    }


def test_spec_is_exact_recipe_only_and_rejects_old_27bf_policy(
    tmp_path: Path,
) -> None:
    assert L.OLD_SHARED_READY_POLICY_SHA256 == (
        "27bf405e5677fe2e7bab6fcc15c166901734048dd334b8b0abc3a8ffef3ce416"
    )
    validated = L._validate_spec_document(_spec(tmp_path))
    assert validated["action_id"] == L.ACTION_ID
    assert validated["seed"] == 0
    assert validated["num_envs"] == 1
    assert validated["max_iterations"] == 0

    old_policy = _spec(tmp_path)
    old_policy["vendor_runtime_training_contract_sha256"] = (
        L.OLD_SHARED_READY_POLICY_SHA256
    )
    with pytest.raises(L.LaunchRefused, match="old 27bf shared-ready"):
        L._validate_spec_document(old_policy)

    operator_policy = _spec(tmp_path)
    operator_policy["policy_contract_sha256"] = L.OLD_SHARED_READY_POLICY_SHA256
    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_spec_document(operator_policy)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("action_id", "bh_block", "bh_loop_c"),
        ("seed", 1, "seed"),
        ("num_envs", 2, "one env"),
        ("max_iterations", 1, "zero PPO"),
        ("stage", "smoke", "recipe-only"),
    ),
)
def test_non_recipe_budget_or_identity_is_refused(
    tmp_path: Path, field: str, value, message: str
) -> None:
    document = _spec(tmp_path)
    document[field] = value
    with pytest.raises(L.LaunchRefused, match=message):
        L._validate_spec_document(document)


def test_bundle_pin_tamper_and_spent_namespace_are_refused(tmp_path: Path) -> None:
    tampered = _spec(tmp_path)
    tampered["bundle"]["sha256"] = "0" * 64
    with pytest.raises(L.LaunchRefused, match="exact code pin"):
        L._validate_spec_document(tampered)

    spent = _spec(tmp_path)
    Path(spent["namespace"]).mkdir()
    with pytest.raises(L.LaunchRefused, match="permanently spent"):
        L._validate_spec_document(spent)


def test_training_argv_reuses_vendor_dynamic_ready_recipe_infrastructure(
    tmp_path: Path,
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path))
    argv = L._build_training_argv(spec, _vendor_inputs())
    assert f"task={L.TASK_PROFILE_ID}" in argv
    assert "action_ball_dynamic_ready_bootstrap=true" in argv
    assert "num_envs=1" in argv
    assert "max_iterations=0" in argv
    assert "seed=0" in argv
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + L.RECIPE_SENTINEL_POLICY_SHA256
    ) in argv
    assert (
        "action_ball_policy_recipe_output_path="
        + str(Path(spec["namespace"]) / L.RECIPE_FILENAME)
    ) in argv
    assert not any(L.OLD_SHARED_READY_POLICY_SHA256 in item for item in argv)
    assert not any("action_ball_shared_ready_bootstrap" in item for item in argv)
    assert argv.count(L._V.STABLE_READY_PLANT_OVERRIDE) == 1


def test_vendor_inputs_require_identity_authority_bundle_and_schema2_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path))
    bundle = _bundle()
    calls = []
    monkeypatch.setattr(L._V._B, "_validate_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        L._V,
        "_validate_vendor_identity_manifest",
        lambda *a, **k: {
            "manifest": {"path": "identity", "sha256": "2" * 64},
            "runtime_training_contract_sha256": (
                L.VENDOR_RUNTIME_CONTRACT_SHA256
            ),
        },
    )

    def authority(*args, **kwargs):
        calls.append("authority")
        return {
            "runtime_training_contract": {
                "sha256": L.VENDOR_RUNTIME_CONTRACT_SHA256
            }
        }

    monkeypatch.setattr(L._V, "_validate_actual_vendor_authority", authority)
    monkeypatch.setattr(
        L._V,
        "_validate_vendor_runtime_binding",
        lambda *a, **k: {
            "runtime_training_contract_sha256": (
                L.VENDOR_RUNTIME_CONTRACT_SHA256
            )
        },
    )
    loader = SimpleNamespace(
        ACTION_BALL_DYNAMIC_READY_RUNTIME_BINDING_KIND_V2=(
            "action_ball_dynamic_ready_runtime_binding_v2"
        ),
        load_action_ball_dynamic_ready_runtime_binding=lambda **kwargs: {
            "schema_version": 2,
            "kind": "action_ball_dynamic_ready_runtime_binding_v2",
            "action_order": [L.ACTION_ID],
            "motion_sha256_per_action": [bundle["motion"]["sha256"]],
            "binding_sha256": BINDING_SHA,
        },
    )
    monkeypatch.setattr(L, "_load_training_contract_module", lambda root: loader)

    result = L._validate_vendor_inputs(
        Path(spec["source"]["checkout"]),
        spec["source"]["commit_sha"],
        spec,
    )
    assert calls == ["authority"]
    assert result["bundle"] == bundle
    assert result["dynamic_ready_binding"]["binding_sha256"] == BINDING_SHA

    monkeypatch.setattr(
        L._V,
        "_validate_actual_vendor_authority",
        lambda *a, **k: (_ for _ in ()).throw(
            L.LaunchRefused("authority receipt SHA differs")
        ),
    )
    with pytest.raises(L.LaunchRefused, match="authority receipt SHA differs"):
        L._validate_vendor_inputs(
            Path(spec["source"]["checkout"]),
            spec["source"]["commit_sha"],
            spec,
        )


def test_dirty_checkout_fails_before_any_recipe_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "recipe.spec.json"
    _write_canonical(spec_path, _spec(tmp_path))
    monkeypatch.setattr(
        L._S,
        "_verify_clean_source",
        lambda *a, **k: (_ for _ in ()).throw(
            L.LaunchRefused("source checkout is dirty")
        ),
    )
    with pytest.raises(L.LaunchRefused, match="source checkout is dirty"):
        L.build_plan(spec_path)


def test_plan_is_zero_ppo_and_explicitly_non_authorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _spec(tmp_path)
    spec_path = tmp_path / "recipe.spec.json"
    _write_canonical(spec_path, document)
    clean = {
        "checkout": document["source"]["checkout"],
        "commit_sha": document["source"]["commit_sha"],
        "clean": True,
    }
    monkeypatch.setattr(L._S, "_verify_clean_source", lambda *a, **k: clean)
    monkeypatch.setattr(L, "_validate_runtime_sources", lambda *a, **k: {})
    monkeypatch.setattr(
        L._S, "_validate_runtime_asset_environment", lambda: {"fixture": True}
    )
    monkeypatch.setattr(L, "_validate_vendor_inputs", lambda *a, **k: _vendor_inputs())

    plan = L.build_plan(spec_path)
    payload = plan["canonical_payload"]
    assert payload["ppo_updates_authorized"] == 0
    assert payload["launch_prohibited"] is True
    assert payload["export_prohibited"] is True
    assert payload["judge_prohibited"] is True
    assert payload["hardware_authority_prohibited"] is True
    assert payload["output_contract"]["ppo_update_count"] == 0
    assert payload["output_contract"]["checkpoints"] == []


def test_materialized_recipe_yields_new_smoke_policy_sha_and_rejects_old_one(
    tmp_path: Path,
) -> None:
    recipe_path = tmp_path / "recipe.json"
    _write_canonical(recipe_path, _recipe_document())
    result = L._validate_materialized_recipe(
        recipe_path, expected_binding_sha256=BINDING_SHA
    )
    assert result["policy_training_contract_sha256"] == NEW_POLICY_SHA
    assert result["launch_authorized"] is False
    assert result["export_authorized"] is False
    assert result["judge_authorized"] is False
    assert result["hardware_authorized"] is False

    old_path = tmp_path / "old.json"
    _write_canonical(
        old_path, _recipe_document(L.OLD_SHARED_READY_POLICY_SHA256)
    )
    with pytest.raises(L.LaunchRefused, match="exact vendor dynamic-ready"):
        L._validate_materialized_recipe(
            old_path, expected_binding_sha256=BINDING_SHA
        )


def test_launch_reuses_identity_gpu_lock_and_no_clobber_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe_path = tmp_path / "recipe.json"
    _write_canonical(recipe_path, _recipe_document())
    plan = {
        "canonical_payload": {
            "output_contract": {
                "recipe": str(recipe_path),
                "dynamic_ready_binding_sha256": BINDING_SHA,
            }
        }
    }
    observed = {}

    def safety_launch(value, *, confirm_claim):
        observed["plan"] = value
        observed["claim"] = confirm_claim
        return {"accepted": True, "kind": L.RESULT_KIND}

    monkeypatch.setattr(L._I, "launch", safety_launch)
    result = L.launch(plan, confirm_claim="f" * 64)
    assert observed == {"plan": plan, "claim": "f" * 64}
    assert result["policy_training_contract_sha256"] == NEW_POLICY_SHA
    assert result["launch_authorized"] is False


def test_template_has_no_operator_policy_or_budget_axis(tmp_path: Path) -> None:
    args = SimpleNamespace(
        checkout=str(tmp_path),
        commit_sha="c" * 40,
        isaac_python=str(Path(sys.executable).resolve()),
        gpu_index=0,
        gpu_uuid="GPU-unit-test",
        owner="Franco",
        namespace=str(tmp_path / "a3vendor-dynamic-recipe-template"),
    )
    document = L._template_document(args)
    assert document["action_id"] == L.ACTION_ID
    assert document["seed"] == 0
    assert document["num_envs"] == 1
    assert document["max_iterations"] == 0
    assert "policy_contract_sha256" not in document
    assert document["bundle"] == L.BUNDLE_PIN
