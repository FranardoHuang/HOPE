#!/usr/bin/env python3
"""Create one balanced schema-v3 BankExam paper for both simulator legs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SOURCE_MODULE = (
    REPO_ROOT / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/stage1_question_bank.py"
)
sys.path.insert(0, str(SCRIPT_DIR))

from bank_exam_schedule import (  # noqa: E402
    artifact_document,
    derive_question_bank_question_ids,
    materialize_balanced_bank_exam_schedule,
    sha256_file,
    write_schedule_artifact,
)


def _load_question_bank_module():
    spec = importlib.util.spec_from_file_location("schedule_stage1_question_bank", SOURCE_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage-1 bank module: {SOURCE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam-bank", required=True)
    parser.add_argument("--per-clip-quota", required=True, type=int)
    parser.add_argument("--schedule-seed", type=int, default=0)
    parser.add_argument("--hold-range", nargs=2, type=int, default=(0, 100))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank_path = os.path.abspath(os.path.expanduser(args.exam_bank))
    if not os.path.isfile(bank_path):
        raise SystemExit(f"[FATAL] exam bank does not exist: {bank_path}")
    bank_sha_before = sha256_file(bank_path)
    qb = _load_question_bank_module().load_question_bank(
        bank_path,
        device="cpu",
        clip_names=("forehand", "backhand"),
        allow_legacy=False,
        expected_split="exam",
    )
    bank_sha = sha256_file(bank_path)
    if bank_sha != bank_sha_before:
        raise SystemExit(
            "[FATAL] exam bank changed while its validated arrays were being loaded: "
            f"before={bank_sha_before}, after={bank_sha}"
        )
    question_ids = derive_question_bank_question_ids(qb, bank_sha256=bank_sha)
    artifact = materialize_balanced_bank_exam_schedule(
        bank_sha256=bank_sha,
        clip_names=("forehand", "backhand"),
        question_ids=question_ids,
        per_clip_quota=args.per_clip_quota,
        schedule_seed=args.schedule_seed,
        hold_range=tuple(args.hold_range),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_schedule_artifact(artifact, output)
    document = artifact_document(artifact)
    print(
        f"[bank-exam-schedule] {output} K={len(artifact.items)} "
        f"quota={artifact.per_clip_quota}/clip sha256={artifact.schedule_sha256}"
    )
    print(
        "[bank-exam-schedule] order="
        + ",".join(item.question_id for item in artifact.items)
    )
    if document["bank_sha256"] != bank_sha:  # defensive, should be impossible after construction
        raise RuntimeError("written schedule no longer binds the loaded exam bank")


if __name__ == "__main__":
    main()
