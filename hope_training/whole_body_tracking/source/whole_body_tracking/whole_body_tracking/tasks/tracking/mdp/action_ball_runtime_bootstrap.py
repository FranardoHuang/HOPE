"""Immutable post-dump runtime lineage for formal ActionBall checkpoints.

The launch claim exists before ``env.pkl`` and ``agent.pkl`` are written, so it
cannot bind those bytes.  This receipt is minted only after all runtime inputs
and the independently rebuildable runtime identity are durable.  Every formal
checkpoint binds both its canonical content SHA and the receipt file bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Dict, Mapping, Optional

from . import action_ball_evaluation_inbox as inbox_protocol
from . import action_ball_frozen_eval_identity as runtime_identity


SCHEMA_VERSION = 1
RECEIPT_KIND = "action_ball_runtime_bootstrap_receipt_v1"
RECEIPT_FILENAME = "action_ball_runtime_bootstrap_receipt.json"
TASK_ID = runtime_identity.TASK_ID

RUNTIME_BOOTSTRAP_CONTRACT = {
    "schema_version": SCHEMA_VERSION,
    "kind": RECEIPT_KIND,
    "purpose": (
        "bind the exact post-dump ActionBall training contract, environment "
        "pickle, agent pickle, and independently reconstructed runtime "
        "identity to one launch claim before the first checkpoint is saved"
    ),
    "publication": (
        "canonical JSON, O_EXCL, fsync file, descriptor rehash, fsync parent"
    ),
    "checkpoint_binding": (
        "checkpoint infos and exact-resume state bind both content SHA and "
        "the immutable receipt artifact path/SHA/size"
    ),
}
RUNTIME_BOOTSTRAP_CONTRACT_SHA256 = inbox_protocol.canonical_sha256(
    RUNTIME_BOOTSTRAP_CONTRACT
)

_CONTENT_KEYS = {
    "runtime_bootstrap_contract_sha256",
    "task_id",
    "training_launch_claim_sha256",
    "launch_claim",
    "training_contract",
    "environment_config_pickle",
    "agent_config_pickle",
    "runtime_identity",
    "runtime_inventory",
    "runtime_identity_content_sha256",
    "runtime_identity_contract_sha256",
    "source",
    "lineage_payload",
    "lineage_payload_sha256",
}
_SOURCE_KEYS = {
    "repo_root",
    "object_format",
    "head_commit_oid",
    "detached",
    "clean",
}
_SHA_CHARS = frozenset("0123456789abcdef")


class RuntimeBootstrapReceiptError(RuntimeError):
    """The post-dump runtime receipt is absent, mutable, or inconsistent."""


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise RuntimeBootstrapReceiptError(
            "{} must be a lowercase SHA-256 digest".format(label)
        )
    return value


def _normalized_absolute(value: object, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise RuntimeBootstrapReceiptError(
            "{} must be an absolute path".format(label)
        )
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or os.path.normpath(value) != value.rstrip(os.sep)
    ):
        raise RuntimeBootstrapReceiptError(
            "{} must be an absolute normalized path".format(label)
        )
    return path


def _artifact(
    path: object,
    *,
    label: str,
) -> Dict[str, object]:
    try:
        receipt = inbox_protocol.artifact_receipt(path)
        # artifact_receipt hashes a stable descriptor/path pair; the byte read
        # additionally requires a single-link regular file.
        inbox_protocol.read_artifact_receipt_bytes(
            receipt,
            label=label,
        )
    except Exception as exc:
        raise RuntimeBootstrapReceiptError(
            "cannot bind {}".format(label)
        ) from exc
    return receipt


def _artifact_semantic(receipt: Mapping[str, object]) -> Dict[str, object]:
    return {
        "sha256": receipt["sha256"],
        "size_bytes": receipt["size_bytes"],
    }


def _validate_launch_claim_document(
    document: object,
    *,
    expected_sha256: str,
) -> Dict[str, object]:
    if type(document) is not dict or set(document) != {
        "schema_version",
        "kind",
        "launch_claim_sha256",
        "canonical_payload",
        "argv",
        "confirmation_claim_sha256",
    }:
        raise RuntimeBootstrapReceiptError(
            "ActionBall launch claim envelope is not exact"
        )
    payload = document["canonical_payload"]
    argv = document["argv"]
    argv_without_claim = (
        payload.get("argv_without_launch_claim")
        if type(payload) is dict
        else None
    )
    if (
        document["schema_version"] != 3
        or document["kind"] != "action_ball_no_clobber_launch_claim_v3"
        or type(payload) is not dict
        or document["launch_claim_sha256"] != expected_sha256
        or document["confirmation_claim_sha256"] != expected_sha256
        or inbox_protocol.canonical_sha256(payload) != expected_sha256
        or type(argv) is not list
        or type(argv_without_claim) is not list
        or argv
        != [
            *argv_without_claim,
            "++training_launch_claim_sha256={}".format(
                expected_sha256
            ),
        ]
    ):
        raise RuntimeBootstrapReceiptError(
            "ActionBall launch claim canonical binding is invalid"
        )
    namespace = _normalized_absolute(
        payload.get("namespace"),
        label="ActionBall launch namespace",
    )
    expected_claim_path = namespace / "launch_claim.json"
    expected_path_override = (
        "++training_launch_claim_path={}".format(expected_claim_path)
    )
    if (
        argv_without_claim.count(expected_path_override) != 1
        or any(
            type(value) is not str
            or value.startswith("++training_launch_claim_sha256=")
            for value in argv_without_claim
        )
        or sum(
            type(value) is str
            and value.startswith("++training_launch_claim_path=")
            for value in argv_without_claim
        )
        != 1
    ):
        raise RuntimeBootstrapReceiptError(
            "ActionBall launch claim does not bind its exact path once"
        )
    return payload


def validate_action_ball_launch_claim_document(
    document: object,
    *,
    expected_sha256: str,
) -> Dict[str, object]:
    """Public exact validator shared by trainer, sidecar, and stage verifier."""

    return _validate_launch_claim_document(
        document,
        expected_sha256=_sha256(
            expected_sha256,
            label="expected ActionBall launch claim",
        ),
    )


def _runtime_identity_semantic(
    identity_content: Mapping[str, object],
) -> Dict[str, object]:
    interpreter = identity_content.get("interpreter")
    source = identity_content.get("source")
    if (
        type(interpreter) is not dict
        or type(source) is not dict
        or set(source) != _SOURCE_KEYS
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime identity lacks semantic interpreter/source identity"
        )
    interpreter_semantic = {
        key: value
        for key, value in interpreter.items()
        if key != "path"
    }
    source_semantic = {
        key: value
        for key, value in source.items()
        if key != "repo_root"
    }
    return {
        "runtime_identity_contract_sha256": identity_content[
            "runtime_identity_contract_sha256"
        ],
        "resolved_recipe_contract_sha256": identity_content[
            "resolved_recipe_contract_sha256"
        ],
        "task_id": identity_content["task_id"],
        "training_contract": _artifact_semantic(
            identity_content["training_contract"]
        ),
        "environment_config_pickle": _artifact_semantic(
            identity_content["environment_config_pickle"]
        ),
        "agent_config_pickle": _artifact_semantic(
            identity_content["agent_config_pickle"]
        ),
        "interpreter": interpreter_semantic,
        "packages": identity_content["packages"],
        "source": source_semantic,
    }


def _build_location_free_lineage_payload(
    *,
    claim_payload: Mapping[str, object],
    training_contract: Mapping[str, object],
    environment_config_pickle: Mapping[str, object],
    agent_config_pickle: Mapping[str, object],
    runtime_identity_content: Mapping[str, object],
    runtime_inventory: Mapping[str, object],
    repo_root: Path,
) -> Dict[str, object]:
    runtime_code = claim_payload.get("runtime_code_sha256")
    inventory_identity = (
        claim_payload.get("isaac_python_runtime", {}).get(
            "runtime_inventory"
        )
        if type(claim_payload.get("isaac_python_runtime")) is dict
        else None
    )
    if (
        type(runtime_code) is not dict
        or not runtime_code
        or any(
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            for relative in runtime_code
        )
        or type(inventory_identity) is not dict
    ):
        raise RuntimeBootstrapReceiptError(
            "launch claim lacks runtime code/inventory identity"
        )
    code_semantics = {}
    for relative, expected_sha256 in sorted(runtime_code.items()):
        receipt = _artifact(
            repo_root / relative,
            label="runtime code {}".format(relative),
        )
        if receipt["sha256"] != expected_sha256:
            raise RuntimeBootstrapReceiptError(
                "runtime code differs from launch claim: {}".format(relative)
            )
        code_semantics[relative] = _artifact_semantic(receipt)
    if (
        runtime_inventory["sha256"]
        != inventory_identity.get("file_sha256")
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime inventory bytes differ from launch claim"
        )
    manifest = claim_payload.get("manifest")
    prototype = claim_payload.get("prototype")
    training_recipe = claim_payload.get("training_recipe")
    stage_budget = claim_payload.get("stage_budget")
    if not all(
        type(value) is dict
        for value in (
            manifest,
            prototype,
            training_recipe,
            stage_budget,
        )
    ):
        raise RuntimeBootstrapReceiptError(
            "launch claim lacks formal ActionBall semantic bindings"
        )
    return {
        "schema_version": 1,
        "kind": "action_ball_runtime_bootstrap_location_free_lineage",
        "task_id": TASK_ID,
        "source": {
            key: value
            for key, value in claim_payload.items()
            if key in ("source_commit_sha",)
        },
        "stage": claim_payload.get("stage"),
        "stage_budget": stage_budget,
        "ordered_action_ids": claim_payload.get("ordered_action_ids"),
        "manifest_sha256": manifest.get("sha256"),
        "prototype_sha256": prototype.get("sha256"),
        "training_recipe_sha256": claim_payload.get(
            "training_recipe_sha256"
        ),
        "solver_profile_sha256": claim_payload.get(
            "solver_profile_sha256"
        ),
        "physics_profile_sha256": claim_payload.get(
            "physics_profile_sha256"
        ),
        "policy_contract_sha256": claim_payload.get(
            "policy_contract_sha256"
        ),
        "proposal_sampler_contract_sha256": claim_payload.get(
            "proposal_sampler_contract_sha256"
        ),
        "training_contract": _artifact_semantic(training_contract),
        "environment_config_pickle": _artifact_semantic(
            environment_config_pickle
        ),
        "agent_config_pickle": _artifact_semantic(
            agent_config_pickle
        ),
        "runtime_identity": _runtime_identity_semantic(
            runtime_identity_content
        ),
        "runtime_inventory": {
            **_artifact_semantic(runtime_inventory),
            "content_sha256": inventory_identity.get("content_sha256"),
            "kind": inventory_identity.get("kind"),
        },
        "runtime_code": code_semantics,
    }


def runtime_bootstrap_lineage_payload(
    content: Mapping[str, object],
) -> Dict[str, object]:
    if type(content) is not dict or set(content) != _CONTENT_KEYS:
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap content is not exact"
        )
    payload = content["lineage_payload"]
    if (
        type(payload) is not dict
        or inbox_protocol.canonical_sha256(payload)
        != content["lineage_payload_sha256"]
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap lineage payload SHA is invalid"
        )
    return json.loads(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def runtime_bootstrap_lineage_payload_sha256(
    content: Mapping[str, object],
) -> str:
    payload = runtime_bootstrap_lineage_payload(content)
    return inbox_protocol.canonical_sha256(payload)


def _strict_document_bytes(document: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap receipt is not canonical JSON"
        ) from exc


def durably_sync_runtime_inputs(*paths: object) -> None:
    """Fsync exact single-link runtime files and their parent directories."""

    if not paths:
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap has no inputs to sync"
        )
    parents = set()
    for raw_path in paths:
        path = Path(os.path.abspath(os.fspath(raw_path)))
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or path.is_symlink()
        ):
            raise RuntimeBootstrapReceiptError(
                "runtime bootstrap input is not a single-link regular file: "
                "{}".format(path)
            )
        descriptor = os.open(
            str(path),
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_size,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
            ):
                raise RuntimeBootstrapReceiptError(
                    "runtime bootstrap input changed while opening: {}".format(
                        path
                    )
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
        ):
            raise RuntimeBootstrapReceiptError(
                "runtime bootstrap input changed while syncing: {}".format(
                    path
                )
            )
        parents.add(path.parent)
    for parent in sorted(parents, key=str):
        descriptor = os.open(
            str(parent),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def build_runtime_bootstrap_receipt_document(
    *,
    repo_root: object,
    task_id: str,
    training_launch_claim_sha256: str,
    launch_claim_path: object,
    training_contract_path: object,
    environment_config_pickle_path: object,
    agent_config_pickle_path: object,
    runtime_identity_path: object,
) -> Dict[str, object]:
    """Build one receipt after verifying all four immutable runtime inputs."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    if task_id != TASK_ID:
        raise RuntimeBootstrapReceiptError(
            "formal runtime bootstrap requires task {!r}".format(TASK_ID)
        )
    claim_sha256 = _sha256(
        training_launch_claim_sha256,
        label="training launch claim",
    )
    launch_claim = _artifact(
        launch_claim_path,
        label="ActionBall launch claim",
    )
    try:
        claim_document = inbox_protocol.strict_read_json(
            launch_claim["path"],
            label="ActionBall launch claim",
        )
        claim_payload = _validate_launch_claim_document(
            claim_document,
            expected_sha256=claim_sha256,
        )
    except Exception as exc:
        raise RuntimeBootstrapReceiptError(
            "cannot validate the exact ActionBall launch claim"
        ) from exc
    if claim_payload.get("source_checkout") != str(root):
        raise RuntimeBootstrapReceiptError(
            "launch claim source checkout differs from runtime root"
        )
    expected_claim_path = (
        _normalized_absolute(
            claim_payload.get("namespace"),
            label="ActionBall launch namespace",
        )
        / "launch_claim.json"
    )
    observed_claim_path = _normalized_absolute(
        launch_claim["path"],
        label="ActionBall launch claim artifact",
    ).resolve(strict=True)
    if (
        expected_claim_path.resolve(strict=True)
        != observed_claim_path
    ):
        raise RuntimeBootstrapReceiptError(
            "launch claim artifact differs from its argv-bound path"
        )
    training_contract = _artifact(
        training_contract_path,
        label="training_contract.json",
    )
    env_pickle = _artifact(
        environment_config_pickle_path,
        label="env.pkl",
    )
    agent_pickle = _artifact(
        agent_config_pickle_path,
        label="agent.pkl",
    )
    identity_receipt = _artifact(
        runtime_identity_path,
        label="ActionBall runtime identity",
    )
    inventory_identity = claim_payload.get("isaac_python_runtime")
    inventory_identity = (
        inventory_identity.get("runtime_inventory")
        if type(inventory_identity) is dict
        else None
    )
    if type(inventory_identity) is not dict:
        raise RuntimeBootstrapReceiptError(
            "launch claim lacks a runtime inventory receipt"
        )
    runtime_inventory = _artifact(
        inventory_identity.get("path"),
        label="ActionBall runtime inventory",
    )
    try:
        identity_document = inbox_protocol.strict_read_json(
            runtime_identity_path,
            label="ActionBall runtime identity",
        )
        identity_content = runtime_identity.validate_runtime_identity_document(
            identity_document,
            repo_root=root,
            task_id=task_id,
            training_launch_claim_sha256=claim_sha256,
            training_contract_path=training_contract["path"],
            environment_config_pickle_path=env_pickle["path"],
            agent_config_pickle_path=agent_pickle["path"],
        )
    except Exception as exc:
        raise RuntimeBootstrapReceiptError(
            "runtime identity is not the live post-dump runtime"
        ) from exc
    if (
        identity_content.get("training_contract") != training_contract
        or identity_content.get("environment_config_pickle") != env_pickle
        or identity_content.get("agent_config_pickle") != agent_pickle
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime identity artifact bindings differ from post-dump bytes"
        )
    source = identity_content.get("source")
    if (
        type(source) is not dict
        or set(source) != _SOURCE_KEYS
        or source.get("repo_root") != str(root)
        or source.get("detached") is not True
        or source.get("clean") is not True
    ):
        raise RuntimeBootstrapReceiptError(
            "formal runtime bootstrap requires the exact clean detached source"
        )
    if source.get("head_commit_oid") != claim_payload.get(
        "source_commit_sha"
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime identity source commit differs from launch claim"
        )
    lineage_payload = _build_location_free_lineage_payload(
        claim_payload=claim_payload,
        training_contract=training_contract,
        environment_config_pickle=env_pickle,
        agent_config_pickle=agent_pickle,
        runtime_identity_content=identity_content,
        runtime_inventory=runtime_inventory,
        repo_root=root,
    )
    content = {
        "runtime_bootstrap_contract_sha256": (
            RUNTIME_BOOTSTRAP_CONTRACT_SHA256
        ),
        "task_id": task_id,
        "training_launch_claim_sha256": claim_sha256,
        "launch_claim": launch_claim,
        "training_contract": training_contract,
        "environment_config_pickle": env_pickle,
        "agent_config_pickle": agent_pickle,
        "runtime_identity": identity_receipt,
        "runtime_inventory": runtime_inventory,
        "runtime_identity_content_sha256": identity_document[
            "content_sha256"
        ],
        "runtime_identity_contract_sha256": identity_content[
            "runtime_identity_contract_sha256"
        ],
        "source": dict(source),
        "lineage_payload": lineage_payload,
        "lineage_payload_sha256": inbox_protocol.canonical_sha256(
            lineage_payload
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "content": content,
        "content_sha256": inbox_protocol.canonical_sha256(content),
    }


def validate_runtime_bootstrap_receipt_document(
    document: object,
    *,
    expected_repo_root: object,
    expected_task_id: str,
    expected_training_launch_claim_sha256: str,
    expected_launch_claim_path: Optional[object] = None,
    expected_training_contract_path: Optional[object] = None,
    expected_environment_config_pickle_path: Optional[object] = None,
    expected_agent_config_pickle_path: Optional[object] = None,
    expected_runtime_identity_path: Optional[object] = None,
    expected_runtime_inventory_path: Optional[object] = None,
    expected_source_commit_oid: Optional[str] = None,
) -> Dict[str, object]:
    """Reopen every bound artifact and prove the receipt against live state."""

    if type(document) is not dict or set(document) != {
        "schema_version",
        "kind",
        "content",
        "content_sha256",
    }:
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap receipt envelope is not exact"
        )
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["kind"] != RECEIPT_KIND
        or type(document["content"]) is not dict
        or set(document["content"]) != _CONTENT_KEYS
        or document["content_sha256"]
        != inbox_protocol.canonical_sha256(document["content"])
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap receipt envelope is invalid"
        )
    content = document["content"]
    if (
        content["runtime_bootstrap_contract_sha256"]
        != RUNTIME_BOOTSTRAP_CONTRACT_SHA256
        or content["task_id"] != expected_task_id
        or content["training_launch_claim_sha256"]
        != _sha256(
            expected_training_launch_claim_sha256,
            label="expected training launch claim",
        )
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap contract/task/claim binding differs"
        )
    root = Path(expected_repo_root).expanduser().resolve(strict=True)
    expected_paths = {
        "launch_claim": expected_launch_claim_path,
        "training_contract": expected_training_contract_path,
        "environment_config_pickle": (
            expected_environment_config_pickle_path
        ),
        "agent_config_pickle": expected_agent_config_pickle_path,
        "runtime_identity": expected_runtime_identity_path,
        "runtime_inventory": expected_runtime_inventory_path,
    }
    for name, expected_path in expected_paths.items():
        receipt = content[name]
        try:
            inbox_protocol.verify_artifact_receipt(
                receipt,
                label="runtime bootstrap {}".format(name),
            )
            inbox_protocol.read_artifact_receipt_bytes(
                receipt,
                label="runtime bootstrap {}".format(name),
            )
        except Exception as exc:
            raise RuntimeBootstrapReceiptError(
                "runtime bootstrap {} bytes drifted".format(name)
            ) from exc
        if expected_path is not None:
            expected = Path(expected_path).expanduser().resolve(strict=True)
            observed = _normalized_absolute(
                receipt["path"],
                label="runtime bootstrap {} path".format(name),
            ).resolve(strict=True)
            if observed != expected:
                raise RuntimeBootstrapReceiptError(
                    "runtime bootstrap {} path differs".format(name)
                )
    identity_path = content["runtime_identity"]["path"]
    try:
        claim_document = inbox_protocol.strict_read_json(
            content["launch_claim"]["path"],
            label="runtime bootstrap launch claim",
        )
        claim_payload = _validate_launch_claim_document(
            claim_document,
            expected_sha256=expected_training_launch_claim_sha256,
        )
        identity_document = inbox_protocol.strict_read_json(
            identity_path,
            label="runtime bootstrap identity",
        )
        identity_content = runtime_identity.validate_runtime_identity_document(
            identity_document,
            repo_root=root,
            task_id=expected_task_id,
            training_launch_claim_sha256=(
                expected_training_launch_claim_sha256
            ),
            training_contract_path=content["training_contract"]["path"],
            environment_config_pickle_path=content[
                "environment_config_pickle"
            ]["path"],
            agent_config_pickle_path=content[
                "agent_config_pickle"
            ]["path"],
        )
    except Exception as exc:
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap identity no longer matches live state"
        ) from exc
    source = content["source"]
    if (
        type(source) is not dict
        or set(source) != _SOURCE_KEYS
        or source != identity_content.get("source")
        or source.get("repo_root") != str(root)
        or source.get("detached") is not True
        or source.get("clean") is not True
        or identity_document.get("content_sha256")
        != content["runtime_identity_content_sha256"]
        or identity_content.get("runtime_identity_contract_sha256")
        != content["runtime_identity_contract_sha256"]
        or identity_content.get("training_contract")
        != content["training_contract"]
        or identity_content.get("environment_config_pickle")
        != content["environment_config_pickle"]
        or identity_content.get("agent_config_pickle")
        != content["agent_config_pickle"]
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap identity/source/artifact cross-binding failed"
        )
    if (
        expected_source_commit_oid is not None
        and source.get("head_commit_oid") != expected_source_commit_oid
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap source commit differs from launch claim"
        )
    if (
        claim_payload.get("source_checkout") != str(root)
        or source.get("head_commit_oid")
        != claim_payload.get("source_commit_sha")
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap claim/source checkout binding differs"
        )
    recomputed_lineage = _build_location_free_lineage_payload(
        claim_payload=claim_payload,
        training_contract=content["training_contract"],
        environment_config_pickle=content[
            "environment_config_pickle"
        ],
        agent_config_pickle=content["agent_config_pickle"],
        runtime_identity_content=identity_content,
        runtime_inventory=content["runtime_inventory"],
        repo_root=root,
    )
    if (
        recomputed_lineage
        != runtime_bootstrap_lineage_payload(content)
        or inbox_protocol.canonical_sha256(recomputed_lineage)
        != content["lineage_payload_sha256"]
    ):
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap location-free lineage payload drifted"
        )
    return dict(content)


def publish_runtime_bootstrap_receipt(
    *,
    output_path: object,
    document: Mapping[str, object],
) -> Dict[str, object]:
    """Publish one no-clobber receipt and rehash bytes through its write FD."""

    path = Path(os.path.abspath(os.fspath(output_path)))
    if path.name != RECEIPT_FILENAME:
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap receipt filename is not canonical"
        )
    parent = path.parent
    info = parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or parent.is_symlink():
        raise RuntimeBootstrapReceiptError(
            "runtime bootstrap parent is not a real directory"
        )
    encoded = _strict_document_bytes(document)
    expected_digest = hashlib.sha256(encoded).hexdigest()
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        try:
            descriptor = os.open(str(path), flags, 0o600)
        except FileExistsError as exc:
            raise RuntimeBootstrapReceiptError(
                "runtime bootstrap receipt namespace is already spent: "
                "{}".format(path)
            ) from exc
        view = memoryview(encoded)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise RuntimeBootstrapReceiptError(
                    "runtime bootstrap receipt write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or total != len(encoded)
            or digest.hexdigest() != expected_digest
        ):
            raise RuntimeBootstrapReceiptError(
                "runtime bootstrap receipt descriptor rehash failed"
            )
        os.close(descriptor)
        descriptor = -1
        directory_fd = os.open(
            str(parent),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # An O_EXCL-created partial receipt intentionally remains a loud spent
        # namespace after failure; silently deleting it would permit a second
        # writer.
    try:
        reopened = inbox_protocol.strict_read_json(
            path,
            label="runtime bootstrap receipt",
        )
    except Exception as exc:
        raise RuntimeBootstrapReceiptError(
            "published runtime bootstrap receipt is unreadable"
        ) from exc
    if reopened != document:
        raise RuntimeBootstrapReceiptError(
            "published runtime bootstrap receipt differs from memory"
        )
    receipt = _artifact(
        path,
        label="runtime bootstrap receipt file",
    )
    return {
        "document": reopened,
        "content_sha256": reopened["content_sha256"],
        "artifact_receipt": receipt,
    }


__all__ = [
    "RECEIPT_FILENAME",
    "RECEIPT_KIND",
    "RUNTIME_BOOTSTRAP_CONTRACT",
    "RUNTIME_BOOTSTRAP_CONTRACT_SHA256",
    "RuntimeBootstrapReceiptError",
    "build_runtime_bootstrap_receipt_document",
    "durably_sync_runtime_inputs",
    "publish_runtime_bootstrap_receipt",
    "runtime_bootstrap_lineage_payload",
    "runtime_bootstrap_lineage_payload_sha256",
    "validate_action_ball_launch_claim_document",
    "validate_runtime_bootstrap_receipt_document",
]
