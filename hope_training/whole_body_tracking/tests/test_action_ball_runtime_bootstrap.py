import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


WBT = Path(__file__).resolve().parents[1]
MDP = (
    WBT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


I = _load(
    "action_ball_runtime_bootstrap_test_inbox",
    MDP / "action_ball_evaluation_inbox.py",
)
PACKAGE = "action_ball_runtime_bootstrap_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(MDP)]
sys.modules[PACKAGE] = package
sys.modules[PACKAGE + ".action_ball_evaluation_inbox"] = I
IDENTITY = _load(
    PACKAGE + ".action_ball_frozen_eval_identity",
    MDP / "action_ball_frozen_eval_identity.py",
)
B = _load(
    PACKAGE + ".action_ball_runtime_bootstrap",
    MDP / "action_ball_runtime_bootstrap.py",
)
TRAIN_PATH = WBT / "scripts" / "train.py"


def _canonical_write(path: Path, document):
    path.write_text(
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def _fixture(tmp_path, monkeypatch):
    params = tmp_path / "params"
    params.mkdir()
    contract = params / "training_contract.json"
    env_pickle = params / "env.pkl"
    agent_pickle = params / "agent.pkl"
    identity_path = params / "action_ball_frozen_eval_runtime.json"
    runtime_code = tmp_path / "runtime.py"
    runtime_inventory = tmp_path / "runtime_inventory.json"
    launch_claim = tmp_path / "launch_claim.json"
    contract.write_bytes(b'{"schema_version":3}\n')
    env_pickle.write_bytes(b"exact-env-pickle")
    agent_pickle.write_bytes(b"exact-agent-pickle")
    runtime_code.write_bytes(b"# exact runtime code\n")
    inventory_content = {"python": {"version": "test"}}
    inventory_document = {
        "schema_version": 1,
        "kind": "action_ball_runtime_inventory_v1",
        "content": inventory_content,
        "content_sha256": I.canonical_sha256(inventory_content),
    }
    _canonical_write(runtime_inventory, inventory_document)
    inventory_receipt = I.artifact_receipt(runtime_inventory)
    claim_payload = {
        "source_checkout": str(tmp_path),
        "source_commit_sha": "a" * 40,
        "namespace": str(tmp_path),
        "runtime_code_sha256": {
            "runtime.py": I.artifact_receipt(runtime_code)["sha256"],
        },
        "isaac_python_runtime": {
            "runtime_inventory": {
                "path": str(runtime_inventory),
                "file_sha256": inventory_receipt["sha256"],
                "content_sha256": inventory_document["content_sha256"],
                "kind": inventory_document["kind"],
            }
        },
        "manifest": {"sha256": "1" * 64},
        "prototype": {"sha256": "2" * 64},
        "training_recipe": {"purpose": "test"},
        "training_recipe_sha256": "3" * 64,
        "stage": "test_stage",
        "stage_budget": {"num_envs": 2},
        "ordered_action_ids": [1, 2],
        "solver_profile_sha256": "4" * 64,
        "physics_profile_sha256": "5" * 64,
        "policy_contract_sha256": "6" * 64,
        "proposal_sampler_contract_sha256": "7" * 64,
    }
    claim_payload["argv_without_launch_claim"] = [
        "train.py",
        "++training_launch_claim_path={}".format(launch_claim),
    ]
    claim_sha256 = I.canonical_sha256(claim_payload)
    claim_document = {
        "schema_version": 3,
        "kind": "action_ball_no_clobber_launch_claim_v3",
        "launch_claim_sha256": claim_sha256,
        "canonical_payload": claim_payload,
        "argv": [
            *claim_payload["argv_without_launch_claim"],
            "++training_launch_claim_sha256={}".format(claim_sha256),
        ],
        "confirmation_claim_sha256": claim_sha256,
    }
    _canonical_write(launch_claim, claim_document)
    source = {
        "repo_root": str(tmp_path),
        "object_format": "sha1",
        "head_commit_oid": "a" * 40,
        "detached": True,
        "clean": True,
    }
    identity_content = {
        "runtime_identity_contract_sha256": (
            I.RUNTIME_IDENTITY_CONTRACT_SHA256
        ),
        "resolved_recipe_contract_sha256": (
            I.RESOLVED_EVALUATION_RECIPE_CONTRACT_SHA256
        ),
        "task_id": B.TASK_ID,
        "training_launch_claim_sha256": claim_sha256,
        "training_contract": I.artifact_receipt(contract),
        "environment_config_pickle": I.artifact_receipt(env_pickle),
        "agent_config_pickle": I.artifact_receipt(agent_pickle),
        "interpreter": {
            "path": "/usr/bin/python3",
            "sha256": "8" * 64,
            "size_bytes": 123,
            "implementation": "CPython",
            "version": "3.10.0",
            "cache_tag": "cpython-310",
        },
        "packages": {"torch": "test"},
        "source": source,
    }
    identity_document = {
        "schema_version": 1,
        "kind": "action_ball_frozen_eval_runtime_identity",
        "content": identity_content,
        "content_sha256": I.canonical_sha256(identity_content),
    }
    _canonical_write(identity_path, identity_document)

    def validate_identity(document, **kwargs):
        assert document == identity_document
        assert kwargs["repo_root"] == tmp_path
        assert kwargs["training_launch_claim_sha256"] == claim_sha256
        return dict(identity_content)

    monkeypatch.setattr(
        B.runtime_identity,
        "validate_runtime_identity_document",
        validate_identity,
    )
    return {
        "contract": contract,
        "env": env_pickle,
        "agent": agent_pickle,
        "identity": identity_path,
        "identity_document": identity_document,
        "launch_claim": launch_claim,
        "claim_sha256": claim_sha256,
        "runtime_inventory": runtime_inventory,
    }


def test_runtime_bootstrap_mints_reopens_and_cross_binds_exact_bytes(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    B.durably_sync_runtime_inputs(
        paths["contract"],
        paths["env"],
        paths["agent"],
        paths["identity"],
    )
    document = B.build_runtime_bootstrap_receipt_document(
        repo_root=tmp_path,
        task_id=B.TASK_ID,
        training_launch_claim_sha256=paths["claim_sha256"],
        launch_claim_path=paths["launch_claim"],
        training_contract_path=paths["contract"],
        environment_config_pickle_path=paths["env"],
        agent_config_pickle_path=paths["agent"],
        runtime_identity_path=paths["identity"],
    )
    output = tmp_path / "params" / B.RECEIPT_FILENAME
    published = B.publish_runtime_bootstrap_receipt(
        output_path=output,
        document=document,
    )
    assert published["content_sha256"] == document["content_sha256"]
    assert published["artifact_receipt"] == I.artifact_receipt(output)
    content = B.validate_runtime_bootstrap_receipt_document(
        published["document"],
        expected_repo_root=tmp_path,
        expected_task_id=B.TASK_ID,
        expected_training_launch_claim_sha256=paths["claim_sha256"],
        expected_launch_claim_path=paths["launch_claim"],
        expected_training_contract_path=paths["contract"],
        expected_environment_config_pickle_path=paths["env"],
        expected_agent_config_pickle_path=paths["agent"],
        expected_runtime_identity_path=paths["identity"],
        expected_runtime_inventory_path=paths["runtime_inventory"],
        expected_source_commit_oid="a" * 40,
    )
    assert content["environment_config_pickle"] == I.artifact_receipt(
        paths["env"]
    )
    with pytest.raises(
        B.RuntimeBootstrapReceiptError,
        match="already spent",
    ):
        B.publish_runtime_bootstrap_receipt(
            output_path=output,
            document=document,
        )


def test_runtime_bootstrap_rejects_bound_input_drift(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    document = B.build_runtime_bootstrap_receipt_document(
        repo_root=tmp_path,
        task_id=B.TASK_ID,
        training_launch_claim_sha256=paths["claim_sha256"],
        launch_claim_path=paths["launch_claim"],
        training_contract_path=paths["contract"],
        environment_config_pickle_path=paths["env"],
        agent_config_pickle_path=paths["agent"],
        runtime_identity_path=paths["identity"],
    )
    paths["env"].write_bytes(b"mutated-env")
    with pytest.raises(
        B.RuntimeBootstrapReceiptError,
        match="bytes drifted",
    ):
        B.validate_runtime_bootstrap_receipt_document(
            document,
            expected_repo_root=tmp_path,
            expected_task_id=B.TASK_ID,
            expected_training_launch_claim_sha256=paths["claim_sha256"],
        )


def test_runtime_bootstrap_rejects_unknown_envelope_fields(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    document = B.build_runtime_bootstrap_receipt_document(
        repo_root=tmp_path,
        task_id=B.TASK_ID,
        training_launch_claim_sha256=paths["claim_sha256"],
        launch_claim_path=paths["launch_claim"],
        training_contract_path=paths["contract"],
        environment_config_pickle_path=paths["env"],
        agent_config_pickle_path=paths["agent"],
        runtime_identity_path=paths["identity"],
    )
    document["unexpected"] = True
    with pytest.raises(
        B.RuntimeBootstrapReceiptError,
        match="envelope is not exact",
    ):
        B.validate_runtime_bootstrap_receipt_document(
            document,
            expected_repo_root=tmp_path,
            expected_task_id=B.TASK_ID,
            expected_training_launch_claim_sha256=paths["claim_sha256"],
        )


def test_launch_claim_validator_rejects_non_namespace_path_override(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    document = json.loads(
        paths["launch_claim"].read_text(encoding="ascii")
    )
    payload = document["canonical_payload"]
    payload["argv_without_launch_claim"] = [
        "train.py",
        "++training_launch_claim_path={}".format(
            tmp_path / "wrong" / "launch_claim.json"
        ),
    ]
    claim_sha256 = I.canonical_sha256(payload)
    document["launch_claim_sha256"] = claim_sha256
    document["confirmation_claim_sha256"] = claim_sha256
    document["argv"] = [
        *payload["argv_without_launch_claim"],
        "++training_launch_claim_sha256={}".format(claim_sha256),
    ]
    with pytest.raises(
        B.RuntimeBootstrapReceiptError,
        match="exact path once",
    ):
        B.validate_action_ball_launch_claim_document(
            document,
            expected_sha256=claim_sha256,
        )


def test_train_binds_post_dump_receipt_before_resume_or_learning():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    dump = source.index("dump_pickle(env_pickle_path, env_cfg)")
    publish = source.index(
        "runtime_bootstrap.publish_runtime_bootstrap_receipt("
    )
    bind = source.index(
        "bind_runtime_bootstrap(\n",
        publish,
    )
    resume = source.index(
        "_load_requested_checkpoint()\n",
        bind,
    )
    learn = source.index("runner.learn(\n", resume)
    assert dump < publish < bind < resume < learn
    assert "launch_claim_path=training_launch_claim_path" in source


def test_location_free_lineage_is_stable_across_run_directories(
    tmp_path, monkeypatch
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _fixture(first_root, monkeypatch)
    first_document = B.build_runtime_bootstrap_receipt_document(
        repo_root=first_root,
        task_id=B.TASK_ID,
        training_launch_claim_sha256=first["claim_sha256"],
        launch_claim_path=first["launch_claim"],
        training_contract_path=first["contract"],
        environment_config_pickle_path=first["env"],
        agent_config_pickle_path=first["agent"],
        runtime_identity_path=first["identity"],
    )
    second = _fixture(second_root, monkeypatch)
    second_document = B.build_runtime_bootstrap_receipt_document(
        repo_root=second_root,
        task_id=B.TASK_ID,
        training_launch_claim_sha256=second["claim_sha256"],
        launch_claim_path=second["launch_claim"],
        training_contract_path=second["contract"],
        environment_config_pickle_path=second["env"],
        agent_config_pickle_path=second["agent"],
        runtime_identity_path=second["identity"],
    )
    assert first_document["content_sha256"] != (
        second_document["content_sha256"]
    )
    assert B.runtime_bootstrap_lineage_payload_sha256(
        first_document["content"]
    ) == B.runtime_bootstrap_lineage_payload_sha256(
        second_document["content"]
    )
