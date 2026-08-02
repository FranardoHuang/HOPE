#!/usr/bin/env python3
"""Build the diagnostic single-question N1 tape from offline task receipts.

Every target receipt must describe the same physical question.  The builder
stores all five target recipes in one canonical JSON container, so an ablation
cannot silently change the incoming ball, landing aim, base or clock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


MDP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
TAPE_MODULE_PATH = MDP_ROOT / "action_ball_fixed_question_tape.py"
TAPE_SPEC = importlib.util.spec_from_file_location(
    "_build_action_ball_immutable_n1_tape_module", TAPE_MODULE_PATH
)
if TAPE_SPEC is None or TAPE_SPEC.loader is None:
    raise RuntimeError("cannot load action_ball_fixed_question_tape.py")
tape_module = importlib.util.module_from_spec(TAPE_SPEC)
sys.modules[TAPE_SPEC.name] = tape_module
TAPE_SPEC.loader.exec_module(tape_module)
action_ball_runtime = tape_module._runtime


def _recipe_mapping(values, *, label):
    result = {}
    for raw in values:
        recipe, separator, value = raw.partition("=")
        if not separator or recipe not in tape_module.TARGET_RECIPES or not value:
            raise ValueError(
                f"{label} must be RECIPE=VALUE for one of "
                f"{tape_module.TARGET_RECIPES}, got {raw!r}"
            )
        if recipe in result:
            raise ValueError(f"{label} repeats recipe {recipe!r}")
        result[recipe] = value
    missing = sorted(set(tape_module.TARGET_RECIPES) - set(result))
    if missing:
        raise ValueError(f"{label} is missing recipes {missing}")
    return result


def _load_receipt(path):
    source = Path(path).expanduser().resolve(strict=True)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"task receipt is not JSON: {source}") from error
    return action_ball_runtime.ActionBallTaskReceipt.from_dict(document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-receipt", required=True)
    parser.add_argument(
        "--target-receipt",
        action="append",
        default=[],
        metavar="RECIPE=PATH",
        help="repeat exactly once for each of the five target recipes",
    )
    parser.add_argument(
        "--target-producer-sha256",
        action="append",
        default=[],
        metavar="RECIPE=SHA256",
        help="repeat exactly once for each target producer",
    )
    parser.add_argument("--expected-action-uid", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    receipt_paths = _recipe_mapping(
        args.target_receipt, label="--target-receipt"
    )
    producer_shas = _recipe_mapping(
        args.target_producer_sha256,
        label="--target-producer-sha256",
    )
    question = _load_receipt(args.question_receipt)
    if question.action_uid != args.expected_action_uid:
        raise ValueError(
            "question receipt action UID differs from --expected-action-uid: "
            f"{question.action_uid} != {args.expected_action_uid}"
        )
    targets = {
        recipe: _load_receipt(receipt_paths[recipe])
        for recipe in tape_module.TARGET_RECIPES
    }
    tape = tape_module.ImmutableN1QuestionTape.from_receipts(
        question_receipt=question,
        target_receipts=targets,
        target_producer_sha256=producer_shas,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            tape.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    output.write_bytes(raw)
    report = {
        "path": str(output),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_sha256": tape.canonical_sha256,
        "base_question_sha256": tape.question_sha256,
        "action_uid": question.action_uid,
        "row_count": 1,
        "question_shape": [1, tape_module.QUESTION_WIDTH],
        "install_shape_per_recipe": [1, tape_module.INSTALL_WIDTH],
        "observation_shape_per_recipe": [
            1,
            tape_module.OBSERVATION_WIDTH,
        ],
        "target_lineage": {
            recipe: tape.target_lineage(recipe)
            for recipe in tape_module.TARGET_RECIPES
        },
        "online_lm_calls_at_reset": 0,
        "physical_rng_draws_at_reset": 0,
        "diagnostic_unauthorized": True,
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
