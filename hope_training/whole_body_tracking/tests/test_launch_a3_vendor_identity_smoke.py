from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_a3_vendor_identity_smoke.py"
)

# The production registry is deliberately in the next r6 artifact epoch: every
# identity pin is planned but unmaterialized, so importing the production
# bootstrap launcher must fail closed.  These unit tests exercise the launcher's
# already-materialized r5 protocol with an in-memory registry snapshot; they do
# not weaken that production import gate or write placeholder pins into source.
REGISTRY_SCRIPT = SCRIPT.with_name("a3_vendor_action_registry.py")
_ORIGINAL_SPEC_FROM_FILE = importlib.util.spec_from_file_location
_REGISTRY_SPEC = _ORIGINAL_SPEC_FROM_FILE(
    "_a3_vendor_registry_for_identity_smoke_test", REGISTRY_SCRIPT
)
assert _REGISTRY_SPEC is not None and _REGISTRY_SPEC.loader is not None
_REAL_REGISTRY = importlib.util.module_from_spec(_REGISTRY_SPEC)
sys.modules[_REGISTRY_SPEC.name] = _REAL_REGISTRY
_REGISTRY_SPEC.loader.exec_module(_REAL_REGISTRY)

_R5_IDENTITY_PINS = {
    "bh_loop_c": {
        "identity_prototype": (
            "configs/a3_vendor_identity_bootstrap_20260801_r5/"
            "bh_loop_c.vendor_identity.prototype.v2.json",
            "365eb07b410586dc159c1f8d3548255d08f2f82dd0c6d1de3a5273d8a02b9a8d",
        ),
        "identity_repin_receipt": (
            "configs/a3_vendor_identity_bootstrap_20260801_r5/"
            "bh_loop_c.identity_bootstrap_repin.v1.json",
            "5ae2117a0dd64da6fc9adf59c436326a6ef6d724b9af2c13de7ad0ac83a74ca2",
        ),
        "identity_manifest": (
            "configs/a3_vendor_identity_bootstrap_20260801_r5/"
            "bh_loop_c.vendor_identity.manifest.v3.json",
            "141062e2d6811039374528243d0cb90d469304e45db9fa7464cbc7a5675b7907",
        ),
        "required_identity_manifest": (
            "configs/a3_vendor_runtime_contract_20260801_r5/"
            "required_identity.bh_loop_c.v1.json",
            "d388d1b73733759aee0550f29388bf4ac7e3adb223edf2422df4357d3bae54be",
        ),
        "runtime_contract": (
            "configs/a3_vendor_runtime_authority_20260801_r5/"
            "bh_loop_c.shared_ready.training_contract.json",
            "af892a223fed6459749d675f3481efde78fb7347f166b344a56576e2379c8b55",
        ),
        "runtime_authority_receipt": (
            "configs/a3_vendor_runtime_authority_20260801_r5/"
            "bh_loop_c.vendor_runtime_authority.v1.json",
            "2ef81141746bddfc90ee3333afcd9abbf30139b85ff645edc05d9709ad91932e",
        ),
    },
    "bh_block": {
        "identity_prototype": (
            "configs/a3_vendor_identity_bootstrap_20260801_r5/"
            "bh_block.vendor_identity.prototype.v2.json",
            "c5a0ab79dca616008cd40cf3f796f77aeee2270fecb92a6befecf9f6e7874eb0",
        ),
        "identity_repin_receipt": (
            "configs/a3_vendor_identity_bootstrap_20260801_r5/"
            "bh_block.identity_bootstrap_repin.v1.json",
            "93f591e4fd8b4c16ccf546396abdbce9e67cf1857c6d9a5ab5c230399fa1bb21",
        ),
        "identity_manifest": (
            "configs/a3_vendor_identity_bootstrap_20260801_r5/"
            "bh_block.vendor_identity.manifest.v3.json",
            "4f09af08530eef10443e330570ba5a83ddeef4dd2f0b9df13bdbf27463db1326",
        ),
        "required_identity_manifest": (
            "configs/a3_vendor_runtime_contract_20260801_r5/"
            "required_identity.bh_block.v1.json",
            "0f7256371ddfcc0ff22203c343c594588569aa88fa2f4ceeec154ef9c634c4ed",
        ),
        "runtime_contract": (
            "configs/a3_vendor_runtime_authority_20260801_r5/"
            "bh_block.shared_ready.training_contract.json",
            "1edcd4065f6cfd476a0e8b2d36c9c454354095ff7877fa08f57e9397dd4d9fcd",
        ),
        "runtime_authority_receipt": (
            "configs/a3_vendor_runtime_authority_20260801_r5/"
            "bh_block.vendor_runtime_authority.v1.json",
            "a2b9886398a821722015b420409b782fab2059b2598d9a1ecf25ab32bbdfba01",
        ),
    },
}


def _r5_materialized_config(action_id: str):
    config = _REAL_REGISTRY.ACTION_CONFIGS[action_id]
    pins = {
        name: _REAL_REGISTRY.ArtifactPin(path, digest)
        for name, (path, digest) in _R5_IDENTITY_PINS[action_id].items()
    }
    return replace(
        config,
        identity_source_commit="cf3e07c2a674f796047c1919d4f949db0f57a1d5",
        identity_repin_producer=_REAL_REGISTRY.ArtifactPin(
            "hope_training/whole_body_tracking/scripts/"
            "materialize_a3_vendor_identity_manifest.py",
            "b90bac5f30d801b02e4c074a95ae207493214d91938d91890590a7c1aeeb801a",
        ),
        **pins,
    )


_R5_ACTION_CONFIGS = MappingProxyType(
    {
        action_id: _r5_materialized_config(action_id)
        for action_id in ("bh_loop_c", "bh_block")
    }
)


def _r5_action_source_identity(config):
    return {
        "schema_version": 1,
        "action_id": config.action_id,
        "scope": config.scope,
        "stable_motion": dict(_REAL_REGISTRY.stable_pin(config.stable_motion)),
        "stable_source_manifest": dict(
            _REAL_REGISTRY.stable_pin(config.stable_source_manifest)
        ),
        "stable_source_prototype": dict(
            _REAL_REGISTRY.stable_pin(config.stable_source_prototype)
        ),
        "planned_paths": {
            name: getattr(config, name).path
            for name in (
                "identity_prototype",
                "identity_repin_receipt",
                "identity_manifest",
                "required_identity_manifest",
                "runtime_contract",
                "runtime_authority_receipt",
            )
        },
    }


def _r5_action_source_identity_sha256(config) -> str:
    payload = json.dumps(
        _r5_action_source_identity(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class _MaterializedR5RegistryLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        for name in dir(_REAL_REGISTRY):
            if not name.startswith("__"):
                setattr(module, name, getattr(_REAL_REGISTRY, name))
        module.ACTION_CONFIGS = _R5_ACTION_CONFIGS
        module.ALLOWED_ACTION_IDS = frozenset(_R5_ACTION_CONFIGS)
        module.DEFAULT_ACTION_ID = "bh_loop_c"

        def get_action_config(action_id):
            if type(action_id) is not str or action_id not in _R5_ACTION_CONFIGS:
                raise module.VendorActionRegistryError(
                    "vendor action_id must be one of: "
                    + ", ".join(sorted(_R5_ACTION_CONFIGS))
                )
            return _R5_ACTION_CONFIGS[action_id]

        module.get_action_config = get_action_config
        module.action_source_identity = _r5_action_source_identity
        module.action_source_identity_sha256 = _r5_action_source_identity_sha256
        module.action_source_registry_pin = lambda config: {
            "path": module.REGISTRY_REPO_PATH,
            "action_id": config.action_id,
            "source_identity_sha256": _r5_action_source_identity_sha256(config),
        }


def _fixture_aware_spec_from_file_location(name, location, *args, **kwargs):
    if name == "_hope_vendor_identity_action_registry":
        return importlib.machinery.ModuleSpec(
            name, _MaterializedR5RegistryLoader(), origin=str(location)
        )
    return _ORIGINAL_SPEC_FROM_FILE(name, location, *args, **kwargs)


importlib.util.spec_from_file_location = _fixture_aware_spec_from_file_location
SPEC = importlib.util.spec_from_file_location("a3_vendor_identity_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
try:
    SPEC.loader.exec_module(L)
finally:
    importlib.util.spec_from_file_location = _ORIGINAL_SPEC_FROM_FILE
    sys.modules.pop("_hope_vendor_identity_action_registry", None)

REWARD_SHA = L.EXPECTED_REWARD_RECIPE_SHA256
POLICY_SHA = "b" * 64


def test_production_import_fail_closes_during_r7_source_epoch() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "awaiting code-pinned identity source commit materialization" in output
    assert "template" not in result.stdout


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


def _spec(tmp_path: Path, *, stage: str = "recipe") -> dict:
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    isaac_python = tmp_path / "isaac-python"
    isaac_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    isaac_python.chmod(0o755)
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    namespace = runs / f"a3vendor-identity-{stage}-gpu0-r1"
    return {
        "schema_version": L.SCHEMA_VERSION,
        "kind": L.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "c" * 40,
            "isaac_python": str(isaac_python),
        },
        "action_id": L.ACTION_ID,
        "motion": dict(L.MOTION_PIN),
        "manifest": dict(L.MANIFEST_PIN),
        "expected_effective_reward_recipe_sha256": REWARD_SHA,
        "policy_contract_sha256": None if stage == "recipe" else POLICY_SHA,
        "seed": 0,
        "stage": stage,
        "num_envs": 1,
        "max_iterations": 1 if stage == "recipe" else 2,
        "save_interval": 1,
        "gpu": {
            "index": 0,
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "owner": "Franco",
            "lock_path": "/tmp/hope_lean_queue_gpu0.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


@pytest.mark.parametrize(
    ("stage", "iterations", "policy_sha"),
    (("recipe", 1, None), ("smoke", 2, POLICY_SHA)),
)
def test_only_exact_two_stage_protocol_is_accepted(
    tmp_path: Path, stage: str, iterations: int, policy_sha: str | None
) -> None:
    normalized = L._validate_spec_document(_spec(tmp_path, stage=stage))
    assert normalized["stage"] == stage
    assert normalized["action_id"] == L.ACTION_ID
    assert normalized["num_envs"] == 1
    assert normalized["max_iterations"] == iterations
    assert normalized["save_interval"] == 1
    assert normalized["policy_contract_sha256"] == policy_sha


def test_stage_budget_seed_policy_and_long_fail_closed(tmp_path: Path) -> None:
    bad_budget = _spec(tmp_path, stage="smoke")
    bad_budget["max_iterations"] = 3
    with pytest.raises(L.LaunchRefused, match="exactly 1 env / 2 PPO"):
        L._validate_spec_document(bad_budget)

    bad_seed = _spec(tmp_path, stage="smoke")
    bad_seed["seed"] = 1
    with pytest.raises(L.LaunchRefused, match="seed must be exactly 0"):
        L._validate_spec_document(bad_seed)

    missing_policy = _spec(tmp_path, stage="smoke")
    missing_policy["policy_contract_sha256"] = None
    with pytest.raises(L.LaunchRefused, match="policy_contract_sha256"):
        L._validate_spec_document(missing_policy)

    sentinel = _spec(tmp_path, stage="smoke")
    sentinel["policy_contract_sha256"] = L.RECIPE_SENTINEL_POLICY_SHA256
    with pytest.raises(L.LaunchRefused, match="sentinel"):
        L._validate_spec_document(sentinel)

    recipe_with_policy = _spec(tmp_path, stage="recipe")
    recipe_with_policy["policy_contract_sha256"] = POLICY_SHA
    with pytest.raises(L.LaunchRefused, match="must be null"):
        L._validate_spec_document(recipe_with_policy)

    long = _spec(tmp_path, stage="smoke")
    long["stage"] = "long"
    with pytest.raises(L.LaunchRefused, match="recipe.*smoke"):
        L._validate_spec_document(long)


def test_fixed_task_inputs_and_no_override_surface(tmp_path: Path) -> None:
    wrong_reward = _spec(tmp_path)
    wrong_reward["expected_effective_reward_recipe_sha256"] = "a" * 64
    with pytest.raises(L.LaunchRefused, match="fixed.*reward receipt"):
        L._validate_spec_document(wrong_reward)

    changed_motion = _spec(tmp_path)
    changed_motion["motion"] = {
        "path": "other.npz",
        "sha256": L.MOTION_PIN["sha256"],
    }
    with pytest.raises(L.LaunchRefused, match="fixed tracked.*motion"):
        L._validate_spec_document(changed_motion)

    changed_manifest = _spec(tmp_path)
    changed_manifest["manifest"] = {
        "path": L.MANIFEST_PIN["path"],
        "sha256": "d" * 64,
    }
    with pytest.raises(L.LaunchRefused, match="fixed tracked.*manifest"):
        L._validate_spec_document(changed_manifest)

    task_override = _spec(tmp_path)
    task_override["task"] = "HOPEPingPongActionBall"
    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_spec_document(task_override)

    dynamic_override = _spec(tmp_path)
    dynamic_override["dynamic_ready"] = {"path": "old.json"}
    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_spec_document(dynamic_override)


def test_action_registry_rejects_unknown_and_cross_action_pins(
    tmp_path: Path,
) -> None:
    unknown = _spec(tmp_path)
    unknown["action_id"] = "bh_unknown"
    with pytest.raises(L.LaunchRefused, match="must be one of"):
        L._validate_spec_document(unknown)

    block = _spec(tmp_path)
    block["action_id"] = "bh_block"
    block["motion"] = dict(
        L._R.stable_pin(L._R.ACTION_CONFIGS["bh_block"].stable_motion)
    )
    with pytest.raises(
        L.LaunchRefused,
        match="fixed tracked code-pinned bh_block.*manifest",
    ):
        L._validate_spec_document(block)

    cross_action = _spec(tmp_path)
    cross_action["motion"] = dict(
        L._R.stable_pin(L._R.ACTION_CONFIGS["bh_block"].stable_motion)
    )
    with pytest.raises(L.LaunchRefused, match="code-pinned bh_loop_c.*motion"):
        L._validate_spec_document(cross_action)


def test_fixed_reward_sha_is_justified_by_vendor_leaf_overrides() -> None:
    checkout = Path(__file__).resolve().parents[3]
    task = yaml.safe_load((checkout / L.TASK_SOURCE).read_text(encoding="utf-8"))
    assert set(task) == {
        "defaults",
        "name",
        "rewards",
        "actions",
        "push",
        "force_push",
    }
    assert task["rewards"] == {
        "action_acc_weight": 0.0,
        "racket_position_coarse_weight": 1.0,
        "racket_position_coarse_std": 0.30,
    }
    assert task["force_push"] == {"enable": False}
    assert L.EXPECTED_REWARD_RECIPE_SHA256 == (
        "845d75b4f409725e9dfc7070b1070a6dd6385486c79a6c5a1aec60c41c42ff02"
    )


def test_namespace_and_no_clobber_are_identity_specific(tmp_path: Path) -> None:
    wrong_name = _spec(tmp_path)
    namespace = Path(wrong_name["namespace"]).with_name("generic-smoke")
    wrong_name["namespace"] = str(namespace)
    wrong_name["log_path"] = str(namespace / "run.log")
    with pytest.raises(L.LaunchRefused, match="a3vendor-identity"):
        L._validate_spec_document(wrong_name)

    spent = _spec(tmp_path)
    Path(spent["namespace"]).mkdir()
    with pytest.raises(L.LaunchRefused, match="permanently spent"):
        L._validate_spec_document(spent)

    bad_gpu = _spec(tmp_path)
    bad_gpu["gpu"]["uuid"] = "0"
    with pytest.raises(L.LaunchRefused, match="explicit GPU UUID"):
        L._validate_spec_document(bad_gpu)


def _scientific_inputs() -> dict:
    return {
        "motion": dict(L.MOTION_PIN),
        "manifest": dict(L.MANIFEST_PIN),
    }


@pytest.mark.parametrize("stage", ["recipe", "smoke"])
def test_training_argv_is_vendor_shared_ready_and_never_dynamic(
    tmp_path: Path, stage: str
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, stage=stage))
    argv = L._build_training_argv(spec, _scientific_inputs())
    assert f"task={L.TASK_PROFILE_ID}" in argv
    assert "action_ball_shared_ready_bootstrap=true" in argv
    assert "task.racket.action_ball_diagnostic_unauthorized=true" in argv
    assert "algo.policy.init_noise_std=0.02" in argv
    assert argv.count(L.POLICY_NOISE_STD_OVERRIDE) == 1
    assert [
        item
        for item in argv
        if item.startswith(
            (
                "algo.policy.noise_std_type=",
                "+algo.policy.noise_std_type=",
            )
        )
    ] == [L.POLICY_NOISE_STD_OVERRIDE]
    assert f"seed={L.SEED}" in argv
    assert "num_envs=1" in argv
    assert (
        "task.racket.action_ball_manifest_path="
        + str(Path(spec["source"]["checkout"]) / L.MANIFEST_PIN["path"])
    ) in argv
    assert not any(L.SOURCE_MANIFEST_PIN["path"] in item for item in argv)
    assert not any("task=HOPEPingPongActionBall" == item for item in argv)
    forbidden = (
        "action_ball_dynamic_ready",
        "checkpoint_path=",
        "stable_ready_plant",
        "task.push",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
    )
    assert not any(fragment in item for item in argv for fragment in forbidden)
    recipe_args = [
        item for item in argv if item.startswith("action_ball_policy_recipe_output_path=")
    ]
    if stage == "recipe":
        assert recipe_args == [
            "action_ball_policy_recipe_output_path="
            + str(Path(spec["namespace"]) / L.RECIPE_FILENAME)
        ]
        assert (
            "task.racket.action_ball_policy_contract_sha256="
            + L.RECIPE_SENTINEL_POLICY_SHA256
        ) in argv
    else:
        assert recipe_args == []
        assert (
            "task.racket.action_ball_policy_contract_sha256=" + POLICY_SHA
        ) in argv


@pytest.mark.parametrize(
    "overrides",
    (
        [],
        ["algo.policy.noise_std_type=scalar"],
        ["+algo.policy.noise_std_type=log"],
        [
            "algo.policy.noise_std_type=log",
            "algo.policy.noise_std_type=log",
        ],
        [
            "algo.policy.noise_std_type=log",
            "+algo.policy.noise_std_type=scalar",
        ],
    ),
)
def test_policy_noise_std_type_guard_rejects_missing_duplicate_or_conflict(
    overrides: list[str],
) -> None:
    with pytest.raises(L.LaunchRefused, match="exact-once.*log override"):
        L._require_unique_log_std_override(overrides)


def test_policy_noise_std_type_guard_accepts_only_canonical_log_override() -> None:
    L._require_unique_log_std_override([L.POLICY_NOISE_STD_OVERRIDE])


def test_output_contract_distinguishes_recipe_and_two_update_smoke(
    tmp_path: Path,
) -> None:
    recipe = L._validate_spec_document(_spec(tmp_path, stage="recipe"))
    recipe_output = L._rsl_output_contract(recipe)
    assert recipe_output["ppo_update_count"] == 0
    assert recipe_output["namespace_recipe"].endswith(L.RECIPE_FILENAME)
    assert recipe_output["training_contract"] is None
    assert recipe_output["checkpoints"] == []
    assert recipe_output["required_runtime_log_events"] == []

    smoke = L._validate_spec_document(_spec(tmp_path, stage="smoke"))
    smoke_output = L._rsl_output_contract(smoke)
    assert smoke_output["ppo_update_count"] == 2
    assert smoke_output["training_contract"] == "params/training_contract.json"
    assert smoke_output["checkpoints"] == ["model_0.pt", "model_1.pt"]
    assert smoke_output["required_runtime_log_events"] == [
        "HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON=",
        "HOPE_RSL_RL_RUNTIME_ABI_JSON=",
        "HOPE_POLICY_STD_UPDATE_JSON=",
    ]
    assert L.EXPERIMENT_NAME in smoke_output["rsl_experiment_root"]


def test_scientific_inputs_require_exact_n1_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "schema_version": 3,
        "action_order": [L.ACTION_ID],
        "mobility_mode": "no_move",
        "actions": [
            {
                "action_id": L.ACTION_ID,
                "motion_path": L.MOTION_PIN["path"],
                "motion_sha256": L.MOTION_PIN["sha256"],
            }
        ],
    }
    monkeypatch.setattr(
        L._S,
        "_verify_tracked_file",
        lambda *args, **kwargs: (dict(L.MOTION_PIN), tmp_path / "motion.npz"),
    )
    monkeypatch.setattr(
        L._S,
        "_load_tracked_json",
        lambda *args, **kwargs: (dict(L.MANIFEST_PIN), manifest),
    )
    result = L._validate_scientific_inputs(
        tmp_path, "c" * 40, L.MOTION_PIN, L.MANIFEST_PIN
    )
    assert result == _scientific_inputs()

    manifest["action_order"] = ["bh_block"]
    with pytest.raises(L.LaunchRefused, match="exact N=1"):
        L._validate_scientific_inputs(
            tmp_path, "c" * 40, L.MOTION_PIN, L.MANIFEST_PIN
        )


def test_real_tracked_n1_manifest_closes_launcher_pin() -> None:
    checkout = Path(__file__).resolve().parents[3]
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "ls-files",
            "--error-unmatch",
            L.MANIFEST_PIN["path"],
        ],
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("bootstrap manifest awaits integration commit")
    commit_sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = L._validate_scientific_inputs(
        checkout,
        commit_sha,
        L.MOTION_PIN,
        L.MANIFEST_PIN,
    )

    assert result == _scientific_inputs()


def _bootstrap_documents(checkout: Path) -> dict[str, dict]:
    pins = {
        "profile": L.PROFILE_PIN,
        "prototype": L.PROTOTYPE_PIN,
        "manifest": L.MANIFEST_PIN,
        "receipt": L.RECEIPT_PIN,
        "source_prototype": L.SOURCE_PROTOTYPE_PIN,
        "source_manifest": L.SOURCE_MANIFEST_PIN,
    }
    return {
        name: json.loads((checkout / pin["path"]).read_text(encoding="utf-8"))
        for name, pin in pins.items()
    }


def _validate_real_bootstrap_documents(documents: dict[str, dict]) -> None:
    config = L._R.get_action_config(L.ACTION_ID)
    L._validate_bootstrap_repin_documents(
        profile=documents["profile"],
        prototype=documents["prototype"],
        manifest=documents["manifest"],
        receipt=documents["receipt"],
        source_prototype=documents["source_prototype"],
        source_manifest=documents["source_manifest"],
        source_map=documents["profile"]["solver_implementation_source_sha256"],
        action_registry_pin=dict(L._R.action_source_registry_pin(config)),
    )


def test_real_bootstrap_repin_documents_close_all_four_artifacts() -> None:
    checkout = Path(__file__).resolve().parents[3]
    documents = _bootstrap_documents(checkout)
    _validate_real_bootstrap_documents(documents)


@pytest.mark.parametrize("artifact", ("profile", "prototype", "manifest", "receipt"))
def test_each_bootstrap_artifact_tamper_is_rejected(artifact: str) -> None:
    checkout = Path(__file__).resolve().parents[3]
    documents = deepcopy(_bootstrap_documents(checkout))
    if artifact == "profile":
        documents[artifact]["solver_payload"]["solve"]["n_iters"] += 1
    elif artifact == "prototype":
        documents[artifact]["scopes"]["upper"][0]["contact_frame"] += 1
    elif artifact == "manifest":
        documents[artifact]["actions"][0]["action_uid"] += 1
    else:
        documents[artifact]["authorization"]["training"] = True
    with pytest.raises(L.LaunchRefused):
        _validate_real_bootstrap_documents(documents)


def test_new_identity_receipt_requires_exact_tracked_action_registry_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = L._action_identity_pins

    def with_new_producer(action_id):
        pins = original(action_id)
        pins["producer"] = {
            "path": L.IDENTITY_REPIN_PRODUCER_SOURCE,
            "sha256": "9" * 64,
        }
        return pins

    monkeypatch.setattr(L, "_action_identity_pins", with_new_producer)
    with pytest.raises(L.LaunchRefused, match="requires.*action registry pin"):
        L._expected_repin_receipt(L.ACTION_ID)
    registry_pin = {
        "path": L.ACTION_REGISTRY_SOURCE,
        "action_id": L.ACTION_ID,
        "source_identity_sha256": "8" * 64,
    }
    expected = L._expected_repin_receipt(
        L.ACTION_ID, action_registry_pin=registry_pin
    )
    assert expected["inputs"]["action_registry"] == registry_pin
    assert expected["inputs"]["producer"]["sha256"] == "9" * 64


def _materialized_action_config(action_id: str, producer_sha: str):
    config = L._R.ACTION_CONFIGS[action_id]
    return replace(
        config,
        identity_source_commit="d" * 40,
        identity_repin_producer=L._R.ArtifactPin(
            L.IDENTITY_REPIN_PRODUCER_SOURCE, producer_sha
        ),
        identity_prototype=L._R.ArtifactPin(
            config.identity_prototype.path, "1" * 64
        ),
        identity_repin_receipt=L._R.ArtifactPin(
            config.identity_repin_receipt.path, "2" * 64
        ),
        identity_manifest=L._R.ArtifactPin(
            config.identity_manifest.path, "3" * 64
        ),
    )


def test_per_action_producer_pins_coexist_and_cross_action_receipt_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _materialized_action_config(
        "bh_loop_c", L.LEGACY_REGISTRY_FREE_PRODUCER_SHA256
    )
    block = _materialized_action_config("bh_block", "9" * 64)
    configs = {"bh_loop_c": loop, "bh_block": block}
    monkeypatch.setattr(L._R, "get_action_config", lambda action_id: configs[action_id])

    loop_receipt = L._expected_repin_receipt("bh_loop_c")
    block_registry = dict(L._R.action_source_registry_pin(block))
    block_receipt = L._expected_repin_receipt(
        "bh_block", action_registry_pin=block_registry
    )
    assert "action_registry" not in loop_receipt["inputs"]
    assert block_receipt["inputs"]["action_registry"]["action_id"] == "bh_block"
    assert loop_receipt["inputs"]["producer"]["sha256"] != (
        block_receipt["inputs"]["producer"]["sha256"]
    )

    documents = deepcopy(
        _bootstrap_documents(Path(__file__).resolve().parents[3])
    )
    documents["receipt"] = block_receipt
    with pytest.raises(L.LaunchRefused, match="receipt differs"):
        L._validate_bootstrap_repin_documents(
            profile=documents["profile"],
            prototype=documents["prototype"],
            manifest=documents["manifest"],
            receipt=documents["receipt"],
            source_prototype=documents["source_prototype"],
            source_manifest=documents["source_manifest"],
            source_map=documents["profile"]["solver_implementation_source_sha256"],
            action_id="bh_loop_c",
        )


def test_registry_source_identity_excludes_later_producer_repin() -> None:
    config = L._R.ACTION_CONFIGS["bh_block"]
    before = L._R.action_source_identity_sha256(config)
    after = L._R.action_source_identity_sha256(
        replace(
            config,
            identity_source_commit="d" * 40,
            identity_repin_producer=L._R.ArtifactPin(
                config.identity_repin_producer.path, "9" * 64
            ),
        )
    )
    assert before == after


def test_old_stable_manifest_is_rejected_by_vendor_identity_spec(
    tmp_path: Path,
) -> None:
    document = _spec(tmp_path)
    document["manifest"] = dict(L.SOURCE_MANIFEST_PIN)
    with pytest.raises(L.LaunchRefused, match="vendor-identity N=1 manifest"):
        L._validate_spec_document(document)


def test_real_tracked_bootstrap_repin_closes_current_commit(tmp_path: Path) -> None:
    source_checkout = Path(__file__).resolve().parents[3]
    checkout = tmp_path / "clean-checkout"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--shared",
            "--no-checkout",
            str(source_checkout),
            str(checkout),
        ],
        check=True,
    )
    commit_sha = subprocess.run(
        ["git", "-C", str(source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "-q", commit_sha],
        check=True,
    )
    required = (
        L.PROFILE_PIN,
        L.PROTOTYPE_PIN,
        L.MANIFEST_PIN,
        L.RECEIPT_PIN,
    )
    tracked = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", *[pin["path"] for pin in required]],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if set(tracked) != {pin["path"] for pin in required}:
        pytest.skip("bootstrap outputs await integration commit")
    result = L._validate_bootstrap_repin_artifacts(checkout, commit_sha)
    assert result["profile_pins"] == dict(L.PROFILE_PIN)
    assert result["prototype"] == dict(L.PROTOTYPE_PIN)
    assert result["manifest"] == dict(L.MANIFEST_PIN)
    assert result["receipt"] == dict(L.RECEIPT_PIN)


def test_dirty_source_refuses_before_any_runtime_or_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"], check=True
    )
    tracked = checkout / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_text("dirty\n", encoding="utf-8")

    document = _spec(tmp_path, stage="recipe")
    document["source"]["checkout"] = str(checkout)
    document["source"]["commit_sha"] = commit
    spec_path = tmp_path / "dirty-spec.json"
    spec_path.write_bytes(_canonical(document))
    runtime_called = False

    def unexpected(*args, **kwargs):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime validation must not run after dirty source")

    monkeypatch.setattr(L, "_validate_runtime_sources", unexpected)
    with pytest.raises(L.LaunchRefused, match="clean|dirty|status"):
        L.build_plan(spec_path)
    assert runtime_called is False


def test_build_plan_claim_seals_bootstrap_repin_before_runtime_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _spec(tmp_path, stage="recipe")
    spec_path = tmp_path / "identity-spec.json"
    spec_path.write_bytes(_canonical(document))
    events: list[str] = []
    bootstrap = {
        "source_commit": L.BOOTSTRAP_SOURCE_COMMIT,
        "profile_pins": dict(L.PROFILE_PIN),
        "prototype": dict(L.PROTOTYPE_PIN),
        "manifest": dict(L.MANIFEST_PIN),
        "receipt": dict(L.RECEIPT_PIN),
    }
    scientific = _scientific_inputs()

    monkeypatch.setattr(L._S, "_verify_clean_source", lambda *args: {"clean": True})
    monkeypatch.setattr(L, "_validate_runtime_sources", lambda *args: {"runtime": {}})

    def validate_bootstrap(*args):
        events.append("bootstrap")
        return bootstrap

    def validate_runtime_assets():
        events.append("runtime_assets")
        return {"pinned": True}

    monkeypatch.setattr(L, "_validate_bootstrap_repin_artifacts", validate_bootstrap)
    monkeypatch.setattr(
        L._S, "_validate_runtime_asset_environment", validate_runtime_assets
    )
    monkeypatch.setattr(L, "_validate_scientific_inputs", lambda *args: scientific)
    monkeypatch.setattr(L, "_check_rsl_namespace_available", lambda *args: None)

    plan = L.build_plan(spec_path)

    assert events == ["bootstrap", "runtime_assets"]
    assert plan["canonical_payload"]["bootstrap_repin_artifacts"] == bootstrap
    assert plan["canonical_payload"]["scientific_inputs"]["manifest"] == dict(
        L.MANIFEST_PIN
    )


def test_launch_refuses_occupied_gpu_before_namespace_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, stage="recipe"))
    plan = {
        "launch_claim_sha256": "e" * 64,
        "canonical_payload": {
            "spec": spec,
            "runtime_assets": {"pinned": True},
        },
    }
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(L._S, "_verify_clean_source", lambda *args: {})
    monkeypatch.setattr(L._S, "_validate_runtime_asset_claim", lambda value: value)
    monkeypatch.setattr(L._S, "_open_gpu_lock", lambda path: os.open(lock_path, os.O_RDWR))
    monkeypatch.setattr(
        L._S,
        "_verify_gpu_empty",
        lambda *args: (_ for _ in ()).throw(L.LaunchRefused("GPU occupied")),
    )
    with pytest.raises(L.LaunchRefused, match="GPU occupied"):
        L.launch(plan, confirm_claim="e" * 64)
    assert not Path(spec["namespace"]).exists()


def test_internal_exec_preserves_code_pinned_venv_entry_path(
    tmp_path: Path,
) -> None:
    system_python = tmp_path / "system-python"
    system_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    system_python.chmod(0o755)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(system_python)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    namespace = tmp_path / "namespace"
    spec = {"source": {"isaac_python": str(venv_python)}}

    command = L._build_internal_exec_command(
        spec, checkout, namespace, "a" * 64, 17
    )

    assert command[0] == str(venv_python)
    assert command[0] != str(venv_python.resolve())
    assert command[1] == str(checkout / L.LAUNCHER_SOURCE)
    assert command[-1] == "17"


def test_identity_exec_uses_shared_claim_owned_runtime_asset_environment() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "**_S._runtime_asset_exec_environment(runtime_assets)" in source
    assert 'os.environ.get("LD_LIBRARY_PATH")' not in source
    assert 'runtime_assets["private_glu"]["directory"]' not in source


def test_internal_second_gpu_check_closes_plan_launch_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_spec = _spec(tmp_path, stage="smoke")
    namespace = Path(raw_spec["namespace"])
    namespace.mkdir()
    spec = L._validate_spec_document(raw_spec, namespace_claimed=True)
    lock_path = tmp_path / "race-gpu.lock"
    lock_path.write_text("", encoding="utf-8")
    spec["gpu"]["lock_path"] = str(lock_path)
    scientific = _scientific_inputs()
    argv = L._build_training_argv(spec, scientific)
    output = L._rsl_output_contract(spec)
    bootstrap = {
        "profile_pins": dict(L.PROFILE_PIN),
        "prototype": dict(L.PROTOTYPE_PIN),
        "manifest": dict(L.MANIFEST_PIN),
        "receipt": dict(L.RECEIPT_PIN),
    }
    payload = {
        "kind": L.CLAIM_KIND,
        "spec": spec,
        "source": {"clean": True},
        "runtime_sources": {"launcher": {"path": L.LAUNCHER_SOURCE}},
        "bootstrap_repin_artifacts": bootstrap,
        "runtime_assets": {"pinned": True},
        "scientific_inputs": scientific,
        "training_argv": argv,
        "output_contract": output,
    }
    claim_sha = L.canonical_sha256(payload)
    plan = {
        "schema_version": L.SCHEMA_VERSION,
        "kind": L.CLAIM_KIND,
        "launch_claim_sha256": claim_sha,
        "canonical_payload": payload,
    }
    claim_path = namespace / "launch_claim.json"
    claim_path.write_bytes(_canonical(plan))
    monkeypatch.setattr(
        L,
        "_validate_spec_document",
        lambda document, namespace_claimed=False: spec,
    )
    monkeypatch.setattr(L._S, "_verify_clean_source", lambda *args: {"clean": True})
    monkeypatch.setattr(
        L, "_validate_runtime_sources", lambda *args: payload["runtime_sources"]
    )
    monkeypatch.setattr(
        L, "_validate_bootstrap_repin_artifacts", lambda *args: bootstrap
    )
    monkeypatch.setattr(L._S, "_validate_runtime_asset_claim", lambda value: value)
    monkeypatch.setattr(
        L, "_validate_scientific_inputs", lambda *args: scientific
    )
    monkeypatch.setattr(L, "_build_training_argv", lambda *args: argv)
    monkeypatch.setattr(L, "_rsl_output_contract", lambda *args: output)
    monkeypatch.setattr(
        L._S,
        "_verify_gpu_empty",
        lambda *args: (_ for _ in ()).throw(
            L.LaunchRefused("GPU occupied after namespace claim")
        ),
    )
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(L.LaunchRefused, match="after namespace claim"):
            L._internal_exec(claim_path, claim_sha, descriptor)
    finally:
        os.close(descriptor)
    assert not (namespace / "pre_exec_gpu_admission.json").exists()


def test_concrete_pod_template_has_no_hydra_override_channel(tmp_path: Path) -> None:
    args = argparse.Namespace(
        stage="smoke",
        action_id=L.ACTION_ID,
        checkout="/workspace/franco/a3vendor_commit",
        commit_sha="f" * 40,
        isaac_python="/workspace/hope_isaac_venv/bin/python",
        policy_contract_sha256=POLICY_SHA,
        gpu_index=2,
        gpu_uuid="GPU-473a79f3-8736-6c7f-c3db-290c6be385b8",
        owner="Franco",
        namespace="/workspace/franco/a3vendor-identity-smoke-gpu2-r1",
    )
    document = L._template_document(args)
    assert set(document) == set(L._SPEC_KEYS)
    assert document["stage"] == "smoke"
    assert document["action_id"] == L.ACTION_ID
    assert document["num_envs"] == 1
    assert document["max_iterations"] == 2
    assert document["seed"] == 0
    assert document["gpu"]["lock_path"] == "/tmp/hope_lean_queue_gpu2.lock"
    assert "overrides" not in document


def test_default_and_explicit_loop_templates_build_identical_training_argv(
    tmp_path: Path,
) -> None:
    raw = _spec(tmp_path, stage="smoke")
    common = {
        "stage": "smoke",
        "checkout": raw["source"]["checkout"],
        "commit_sha": raw["source"]["commit_sha"],
        "isaac_python": raw["source"]["isaac_python"],
        "policy_contract_sha256": POLICY_SHA,
        "gpu_index": 0,
        "gpu_uuid": raw["gpu"]["uuid"],
        "owner": raw["gpu"]["owner"],
        "namespace": raw["namespace"],
    }
    implicit = L._template_document(argparse.Namespace(**common))
    explicit = L._template_document(
        argparse.Namespace(**common, action_id=L.ACTION_ID)
    )
    assert implicit == explicit
    implicit_spec = L._validate_spec_document(implicit)
    explicit_spec = L._validate_spec_document(explicit)
    assert L._build_training_argv(
        implicit_spec, _scientific_inputs()
    ) == L._build_training_argv(explicit_spec, _scientific_inputs())


def test_runtime_source_set_pins_identity_delay_and_std_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    observed: list[dict[str, str]] = []

    def verify(checkout_arg, commit, pin, *, name, **kwargs):
        observed.append(dict(pin))
        return pin, checkout_arg / pin["path"]

    monkeypatch.setattr(L._S, "_verify_tracked_file", verify)
    result = L._validate_runtime_sources(checkout, "f" * 40)
    observed_paths = {pin["path"] for pin in observed}
    assert L.LAUNCHER_SOURCE in observed_paths
    assert L.ACTION_REGISTRY_SOURCE in observed_paths
    assert L.TRAIN_SOURCE in observed_paths
    assert L.TASK_SOURCE in observed_paths
    assert L.ROBOT_SOURCE in observed_paths
    assert L.TRAINING_CONTRACT_SOURCE in observed_paths
    assert L.ACTION_SOURCE in observed_paths
    assert L.RUNNER_SOURCE in observed_paths
    assert L.KIT_LAUNCHER_SOURCE in observed_paths
    assert L.IDENTITY_REPIN_PRODUCER_SOURCE in observed_paths
    assert L.PROFILE_PINNER_SOURCE in observed_paths
    assert dict(L.PRODUCER_PIN) in observed
    assert dict(L.PINNER_PIN) in observed
    assert len(result) == len(observed)
