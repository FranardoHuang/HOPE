#!/usr/bin/env python3
"""Materialize and activate one exact signed-face K100 BankExam paper.

This consumer deliberately has no trainer, judge, simulator, subprocess, SSH, or
signal surface.  ``static-validate`` checks the frozen source contract without
requiring the private bank.  ``consume`` requires the exact private rebound bank,
rebuilds every atomic question ID from its bytes, materializes the existing
schema-v3 balanced schedule, publishes it without replacement, validates the
published bytes again, and writes the activation document last.

The activation is a paper receipt only.  It cannot authorize L2, a judge, a
second seed, deployment, or hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np


MANIFEST_ID = "phase1-signed-face-exam-k100-paper-activation-20260714-v1"
ARTIFACT_KIND = "phase1_signed_face_exam_k100_paper_activation"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_BANK = {
    "path": (
        "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/"
        "schema3_exam_bank_rebind_v1/"
        "s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz"
    ),
    "bytes": 63_643,
    "sha256": "60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca",
    "schema_version": 3,
    "split": "exam",
    "clip_order": ["forehand", "backhand"],
    "question_counts": {"forehand": 183, "backhand": 188},
    "physics_contract_sha256": (
        "09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95"
    ),
    "source_family_sha256": (
        "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db"
    ),
    "rebind_result": {
        "path": "configs/phase1_signed_face_exam_bank_rebind_results_20260714.json",
        "sha256": "5deb0aa1e7f7b45bcf15804368c4ef0ac2566779fda5a07a448e19453f29c549",
        "runtime_report_sha256": (
            "dd4332edb47f1fb1f4d51ca00ceed612dbcadf9e395eb536c9b73bef9de69ad0"
        ),
        "runtime_report_content_sha256": (
            "7bdf4d6c2fccaf0b377e6bb76188c0b1d9abd1cd67cd322cbca2b1539a8a19d4"
        ),
    },
}
EXPECTED_FACE_CONTRACT = {
    "schema_version": 1,
    "achieved_and_target_frame": "mount_plusY_A",
    "external_frame": "physical_striking_face_B",
    "physical_B_to_raw_A": "raw_A=mount_normal_sign_per_clip[clip]*physical_B",
    "clip_order": ["forehand", "backhand"],
    "mount_normal_sign_per_clip": [1.0, -1.0],
    "signed_face_required": True,
    "unsigned_or_oriented_plane_fallback_allowed": False,
    "identity_checked_before_orient_normal": True,
    "physical_B_min_x_strict": 1.0e-6,
    "unit_normal_atol": 2.0e-4,
    "identity_gate": (
        "dot(achieved_raw_A,target_raw_A)>0_and_achieved_physical_B.x>1e-6_and_"
        "target_physical_B.x>1e-6_before_orient_normal"
    ),
}
EXPECTED_PAPER = {
    "artifact_type": "bank-exam-schedule",
    "bank_schema_version": 3,
    "per_clip_quota": 50,
    "schedule_k": 100,
    "schedule_seed": 0,
    "hold_range": [0, 100],
    "hold_semantics": "stand-policy-actions-then-raw-frame0-v1",
    "no_wrap": True,
    "repeat": 0,
    "selection": "deterministic_hash_order_without_replacement_round_robin_v3",
    "question_id_contract": "bank-exam-question-v1_includes_exact_rebound_bank_sha",
    "question_ids_rebuilt_from_exact_bank": True,
    "old_schedule_input_allowed": False,
    "all_scheduled_attempts_in_denominator": True,
    "missing_invalid_or_reset_attempts_count_as_failures": True,
    "denominator": {"aggregate": 100, "forehand": 50, "backhand": 50},
    "legacy_schedule_receipts_forbidden": [
        {
            "file_sha256": (
                "66e89986a2b726d529179fcb4c745625ebed0380d59664caceefc55e86071cb3"
            ),
            "semantic_sha256": (
                "7dc6af822fb4130b8c324843f179d77f882d1326306bb19802b00f94447dff3e"
            ),
            "question_id_order_sha256": (
                "b87e81a34ff2d31766e17345f0a8c9d77665b78874093e26bdae257e8ed21f91"
            ),
        }
    ],
}
EXPECTED_OUTPUT = {
    "root": (
        "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/"
        "signed_face_exam_k100_v1"
    ),
    "schedule_basename": "signed_face_exam_k100.schedule.json",
    "activation_basename": "signed_face_exam_k100.activation.json",
    "root_must_not_exist": True,
    "schedule_write_no_replace": True,
    "activation_written_last": True,
}
EXPECTED_AUTHORIZATION = {
    "paper_materialization_only": True,
    "auto_start": False,
    "trainer_started": False,
    "judge_started": False,
    "l2_training_authorized": False,
    "second_seed_authorized": False,
    "checkpoint_stop_or_promote_authorized": False,
    "formal_score_authorized": False,
    "gate3_authorized": False,
    "deployment_authorized": False,
    "real_robot_authorized": False,
}
EXPECTED_SOURCE_BINDINGS = {
    "schedule_module": {
        "path": "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py",
        "sha256": "32721f018f6a35a42aa12ff0a7e48c0d9bc513d238988d953241ee625744b23b",
    },
    "bank_loader": {
        "path": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/stage1_question_bank.py"
        ),
        "sha256": "47db31320c8a37ad95d082a179dcdfad45ca1b740696d54e22069fcdaab8b8b0",
    },
    "signed_face_scorer": {
        "path": "hope_training/whole_body_tracking/scripts/virtual_return_scorer.py",
        "sha256": "9d01da15a6f24166d4d185ede26a2bd29c9c61d02d15942beadb35b335e0f5ec",
    },
    "rebind_result": {
        "path": "configs/phase1_signed_face_exam_bank_rebind_results_20260714.json",
        "sha256": "5deb0aa1e7f7b45bcf15804368c4ef0ac2566779fda5a07a448e19453f29c549",
    },
}


class PaperError(RuntimeError):
    """One immutable-paper invariant failed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PaperError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise PaperError(f"non-finite JSON constant: {value}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperError(f"cannot read strict JSON {path}: {exc}") from exc


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise PaperError(f"{label} must be one lowercase SHA-256")
    return value


def _exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise PaperError(f"{label} differs from the frozen paper contract")


def _source_path(repo_root: Path, spec: Mapping[str, Any], label: str) -> Path:
    if set(spec) != {"path", "sha256"}:
        raise PaperError(f"{label} source binding must contain only path/sha256")
    relative = Path(str(spec["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PaperError(f"{label} source path must be repository-relative")
    require_sha(spec["sha256"], f"{label} source")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise PaperError(f"{label} source escapes repository root") from exc
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise PaperError(f"{label} source bytes changed or are missing")
    return path


def load_manifest(path: Path, *, repo_root: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise PaperError("paper manifest root must be a JSON object")
    expected_keys = {
        "schema_version",
        "manifest_id",
        "purpose",
        "status",
        "recorded_local_date",
        "human_owner",
        "executor",
        "simulation_only",
        "real_robot_commands_forbidden",
        "bank",
        "source_bindings",
        "signed_face_contract",
        "paper",
        "output",
        "authorization",
    }
    if set(value) != expected_keys:
        raise PaperError("paper manifest top-level schema changed")
    if value.get("schema_version") != 1 or value.get("manifest_id") != MANIFEST_ID:
        raise PaperError("unexpected paper manifest schema/id")
    if (
        value.get("purpose")
        != "Rebuild one deterministic immutable 50-per-side K100 paper from the exact signed-face rebound exam bank and publish a paper-only activation after strict no-replace validation."
        or value.get("recorded_local_date") != "2026-07-14"
        or value.get("human_owner") != "Franco"
        or value.get("executor") != "Codex"
    ):
        raise PaperError("paper purpose/ownership/date changed")
    if value.get("status") != "source_reviewed_runtime_consume_not_run":
        raise PaperError("paper manifest must not claim runtime materialization")
    if (
        value.get("simulation_only") is not True
        or value.get("real_robot_commands_forbidden") is not True
    ):
        raise PaperError("paper materialization must stay simulation-only and forbid hardware")
    _exact(value.get("bank"), EXPECTED_BANK, "bank")
    _exact(value.get("signed_face_contract"), EXPECTED_FACE_CONTRACT, "signed-face contract")
    _exact(value.get("paper"), EXPECTED_PAPER, "paper")
    _exact(value.get("output"), EXPECTED_OUTPUT, "output")
    _exact(value.get("authorization"), EXPECTED_AUTHORIZATION, "authorization")

    sources = value.get("source_bindings")
    expected_names = {
        "consumer",
        "schedule_module",
        "bank_loader",
        "signed_face_scorer",
        "rebind_result",
    }
    if not isinstance(sources, dict) or set(sources) != expected_names:
        raise PaperError("source_bindings closure changed")
    if sources.get("consumer", {}).get("path") != (
        "scripts/materialize_phase1_signed_face_exam_k100.py"
    ):
        raise PaperError("consumer source path changed")
    for name, expected in EXPECTED_SOURCE_BINDINGS.items():
        _exact(sources.get(name), expected, f"{name} source binding")
    resolved = {
        name: _source_path(repo_root, spec, name) for name, spec in sources.items()
    }
    if resolved["consumer"] != Path(__file__).resolve():
        raise PaperError("manifest consumer path does not resolve to this script")
    if resolved["rebind_result"] != (repo_root / EXPECTED_BANK["rebind_result"]["path"]).resolve():
        raise PaperError("rebind result source path changed")
    receipt = load_json(resolved["rebind_result"])
    output = receipt.get("output") if isinstance(receipt, dict) else None
    decision = receipt.get("decision") if isinstance(receipt, dict) else None
    if (
        not isinstance(output, dict)
        or not isinstance(decision, dict)
        or output.get("bank", {}).get("sha256") != EXPECTED_BANK["sha256"]
        or output.get("bank", {}).get("bytes") != EXPECTED_BANK["bytes"]
        or output.get("physics_contract_sha256") != EXPECTED_BANK["physics_contract_sha256"]
        or output.get("source_family_sha256") != EXPECTED_BANK["source_family_sha256"]
        or output.get("report", {}).get("sha256")
        != EXPECTED_BANK["rebind_result"]["runtime_report_sha256"]
        or output.get("report", {}).get("content_sha256")
        != EXPECTED_BANK["rebind_result"]["runtime_report_content_sha256"]
        or decision.get("bank_adopted_as_exact_rebound_exam_input") is not True
        or decision.get("paper_activation_materialized") is not False
    ):
        raise PaperError("rebind result does not authorize exactly the frozen next data step")
    value["_resolved_sources"] = resolved
    return value


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PaperError(f"cannot import bound source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_signed_targets(
    bank: Any,
    face_contract: Mapping[str, Any],
    *,
    signed_face_scorer: Any,
) -> dict[str, Any]:
    """Prove every bank target remains raw-A and maps to opponent-facing physical-B."""
    try:
        signs = signed_face_scorer.validate_mount_normal_sign_per_clip(
            face_contract["mount_normal_sign_per_clip"]
        )
    except (TypeError, ValueError) as exc:
        raise PaperError(f"invalid signed-face sign table: {exc}") from exc
    if (
        float(signed_face_scorer._UNIT_NORMAL_ATOL) != face_contract["unit_normal_atol"]
        or float(signed_face_scorer._PHYSICAL_B_MIN_X)
        != face_contract["physical_B_min_x_strict"]
    ):
        raise PaperError("signed-face scorer thresholds differ from paper contract")
    names = tuple(face_contract["clip_order"])
    counts = np.asarray(bank.counts.detach().cpu().numpy(), dtype=np.int64)
    normals = np.asarray(bank.demanded_normal.detach().cpu().numpy(), dtype=np.float64)
    if counts.shape != (len(names),) or normals.ndim != 3 or normals.shape[0] != len(names):
        raise PaperError("bank side/count/normal tensor shape changed")
    result: dict[str, Any] = {}
    for clip, (name, sign, count) in enumerate(zip(names, signs, counts.tolist())):
        if count <= 0 or normals.shape[1] < count or normals.shape[2] != 3:
            raise PaperError(f"{name} has no complete normal rows")
        raw_a = normals[clip, :count]
        norm = np.linalg.norm(raw_a, axis=1)
        try:
            physical_b = np.stack(
                [
                    signed_face_scorer.raw_a_to_physical_b(row, sign)
                    for row in raw_a
                ]
            )
        except (TypeError, ValueError) as exc:
            raise PaperError(f"{name} raw-A target fails physical-B contract: {exc}") from exc
        min_x = float(np.min(physical_b[:, 0]))
        result[name] = {
            "rows": count,
            "mount_normal_sign": sign,
            "raw_A_unit_max_abs_error": float(np.max(np.abs(norm - 1.0))),
            "physical_B_min_x": min_x,
        }
    return result


def validate_schedule(
    *,
    schedule_module: Any,
    schedule_path: Path,
    bank_sha256: str,
    clip_names: tuple[str, ...],
    question_ids: tuple[tuple[str, ...], ...],
    paper_contract: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    try:
        artifact = schedule_module.load_schedule_artifact(
            schedule_path,
            expected_bank_sha256=bank_sha256,
            expected_clip_names=clip_names,
            expected_question_ids=question_ids,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PaperError(f"published schedule failed exact-bank validation: {exc}") from exc
    items = artifact.items
    selected = {
        name: sum(item.clip == clip for item in items)
        for clip, name in enumerate(clip_names)
    }
    ids = [item.question_id for item in items]
    old = paper_contract["legacy_schedule_receipts_forbidden"]
    receipt = {
        "path": str(schedule_path.resolve()),
        "bytes": schedule_path.stat().st_size,
        "file_sha256": sha256_file(schedule_path),
        "semantic_sha256": artifact.schedule_sha256,
        "question_id_order_sha256": canonical_sha256(ids),
        "question_id_order": ids,
        "schedule_k": len(items),
        "selected_per_side": selected,
    }
    expected_sides = {
        name: paper_contract["denominator"][name] for name in clip_names
    }
    if (
        artifact.bank_sha256 != bank_sha256
        or artifact.per_clip_quota != paper_contract["per_clip_quota"]
        or len(items) != paper_contract["schedule_k"]
        or artifact.schedule_seed != paper_contract["schedule_seed"]
        or list(artifact.hold_range) != paper_contract["hold_range"]
        or artifact.hold_semantics != paper_contract["hold_semantics"]
        or artifact.no_wrap is not True
        or any(item.repeat != paper_contract["repeat"] for item in items)
        or selected != expected_sides
        or len(ids) != len(set(ids))
    ):
        raise PaperError("materialized schedule violates K100/full-denominator contract")
    for legacy in old:
        if any(
            receipt[key] == legacy[key]
            for key in ("file_sha256", "semantic_sha256", "question_id_order_sha256")
        ):
            raise PaperError("materialized schedule collides with a forbidden legacy paper receipt")
    return artifact, receipt


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _activation_document(content: Mapping[str, Any]) -> dict[str, Any]:
    material = dict(content)
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "content_sha256": canonical_sha256(material),
        "content": material,
    }


def validate_activation_document(
    document: Mapping[str, Any],
    *,
    manifest_sha256: str,
    consumer_sha256: str,
    bank_sha256: str,
    schedule_receipt: Mapping[str, Any],
) -> None:
    if set(document) != {"schema_version", "artifact_kind", "content_sha256", "content"}:
        raise PaperError("activation top-level schema changed")
    content = document["content"]
    if (
        document["schema_version"] != 1
        or document["artifact_kind"] != ARTIFACT_KIND
        or not isinstance(content, dict)
        or document["content_sha256"] != canonical_sha256(content)
        or content.get("status") != "paper_materialized_not_started"
        or content.get("manifest", {}).get("sha256") != manifest_sha256
        or content.get("consumer", {}).get("sha256") != consumer_sha256
        or content.get("bank", {}).get("sha256") != bank_sha256
        or content.get("schedule") != schedule_receipt
        or content.get("signed_face_contract") != EXPECTED_FACE_CONTRACT
        or content.get("scoring_denominator") != EXPECTED_PAPER["denominator"]
        or content.get("authorization") != EXPECTED_AUTHORIZATION
    ):
        raise PaperError("activation is not the exact signed-face K100 paper receipt")


def consume(manifest_path: Path, manifest: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    bank_spec = manifest["bank"]
    bank_path = Path(bank_spec["path"])
    if (
        not bank_path.is_file()
        or bank_path.is_symlink()
        or bank_path.stat().st_size != bank_spec["bytes"]
        or sha256_file(bank_path) != bank_spec["sha256"]
    ):
        raise PaperError("exact rebound exam-bank bytes are missing or changed")

    sources = manifest["_resolved_sources"]
    loader = _load_module(sources["bank_loader"], "signed_exam_bank_loader_bound")
    schedule_module = _load_module(
        sources["schedule_module"], "signed_exam_schedule_module_bound"
    )
    signed_face_scorer = _load_module(
        sources["signed_face_scorer"], "signed_exam_face_scorer_bound"
    )
    bank = loader.load_question_bank(
        str(bank_path),
        device="cpu",
        clip_names=tuple(bank_spec["clip_order"]),
        allow_legacy=False,
        expected_split="exam",
    )
    meta = bank.metadata
    counts = {
        name: int(bank.counts[index].item())
        for index, name in enumerate(bank_spec["clip_order"])
    }
    if (
        meta.get("schema_version") != bank_spec["schema_version"]
        or meta.get("split") != bank_spec["split"]
        or meta.get("physics_contract_sha256") != bank_spec["physics_contract_sha256"]
        or meta.get("source_family_sha256") != bank_spec["source_family_sha256"]
        or counts != bank_spec["question_counts"]
    ):
        raise PaperError("loaded bank metadata/counts differ from exact rebound receipt")
    signed_targets = validate_signed_targets(
        bank,
        manifest["signed_face_contract"],
        signed_face_scorer=signed_face_scorer,
    )
    question_ids = schedule_module.derive_question_bank_question_ids(
        bank, bank_sha256=bank_spec["sha256"]
    )
    if any(len(set(ids)) != len(ids) for ids in question_ids):
        raise PaperError("bank contains duplicate atomic question IDs")

    try:
        artifact = schedule_module.materialize_balanced_bank_exam_schedule(
            bank_sha256=bank_spec["sha256"],
            clip_names=tuple(bank_spec["clip_order"]),
            question_ids=question_ids,
            per_clip_quota=manifest["paper"]["per_clip_quota"],
            schedule_seed=manifest["paper"]["schedule_seed"],
            hold_range=tuple(manifest["paper"]["hold_range"]),
        )
    except (TypeError, ValueError) as exc:
        raise PaperError(f"cannot materialize exact K100 schedule: {exc}") from exc
    root = Path(manifest["output"]["root"])
    if root.exists():
        raise PaperError(f"no-clobber: output root exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    schedule_path = root / manifest["output"]["schedule_basename"]
    activation_path = root / manifest["output"]["activation_basename"]
    schedule_bytes = schedule_module.canonical_json_bytes(
        schedule_module.artifact_document(artifact)
    ) + b"\n"
    _write_exclusive(schedule_path, schedule_bytes)
    _, schedule_receipt = validate_schedule(
        schedule_module=schedule_module,
        schedule_path=schedule_path,
        bank_sha256=bank_spec["sha256"],
        clip_names=tuple(bank_spec["clip_order"]),
        question_ids=question_ids,
        paper_contract=manifest["paper"],
    )

    manifest_sha = sha256_file(manifest_path)
    consumer_sha = sha256_file(Path(__file__).resolve())
    content = {
        "manifest_id": MANIFEST_ID,
        "status": "paper_materialized_not_started",
        "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
        "consumer": {"path": str(Path(__file__).resolve()), "sha256": consumer_sha},
        "bank": {
            "path": str(bank_path.resolve()),
            "bytes": bank_path.stat().st_size,
            "sha256": bank_spec["sha256"],
            "schema_version": 3,
            "split": "exam",
            "question_counts": counts,
            "physics_contract_sha256": bank_spec["physics_contract_sha256"],
            "source_family_sha256": bank_spec["source_family_sha256"],
        },
        "source_bindings": manifest["source_bindings"],
        "signed_face_contract": manifest["signed_face_contract"],
        "signed_target_audit": signed_targets,
        "schedule": schedule_receipt,
        "scoring_denominator": manifest["paper"]["denominator"],
        "all_scheduled_attempts_in_denominator": True,
        "authorization": manifest["authorization"],
    }
    document = _activation_document(content)
    validate_activation_document(
        document,
        manifest_sha256=manifest_sha,
        consumer_sha256=consumer_sha,
        bank_sha256=bank_spec["sha256"],
        schedule_receipt=schedule_receipt,
    )
    _write_exclusive(activation_path, canonical_json_bytes(document) + b"\n")
    reread = load_json(activation_path)
    validate_activation_document(
        reread,
        manifest_sha256=manifest_sha,
        consumer_sha256=consumer_sha,
        bank_sha256=bank_spec["sha256"],
        schedule_receipt=schedule_receipt,
    )
    return {
        "status": "paper_materialized_not_started",
        "schedule": schedule_receipt,
        "activation": {
            "path": str(activation_path),
            "bytes": activation_path.stat().st_size,
            "sha256": sha256_file(activation_path),
            "content_sha256": reread["content_sha256"],
            "written_last": True,
        },
        "authorization": EXPECTED_AUTHORIZATION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-consumer-sha256", required=True)
    parser.add_argument("command", choices=("static-validate", "consume"))
    args = parser.parse_args(argv)

    config = args.config.resolve()
    repo_root = args.repo_root.resolve()
    require_sha(args.expected_config_sha256, "expected config")
    require_sha(args.expected_consumer_sha256, "expected consumer")
    if not config.is_file() or sha256_file(config) != args.expected_config_sha256:
        raise PaperError("config bytes differ from caller-bound SHA")
    if sha256_file(Path(__file__).resolve()) != args.expected_consumer_sha256:
        raise PaperError("consumer bytes differ from caller-bound SHA")
    manifest = load_manifest(config, repo_root=repo_root)
    if manifest["source_bindings"]["consumer"]["sha256"] != args.expected_consumer_sha256:
        raise PaperError("manifest/caller consumer SHA disagree")
    if args.command == "static-validate":
        print(
            json.dumps(
                {
                    "status": "source_reviewed_runtime_consume_not_run",
                    "manifest_id": MANIFEST_ID,
                    "bank_sha256": EXPECTED_BANK["sha256"],
                    "schedule_k": 100,
                    "attempts_per_side": 50,
                    "output_created": False,
                    "trainer_started": False,
                    "judge_started": False,
                    "real_robot_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    result = consume(config, manifest, repo_root=repo_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PaperError as exc:
        print(f"[signed-exam-paper][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2)
