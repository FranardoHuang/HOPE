"""Host-only tests for the strict task-first manifest contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_module("task_first_curriculum", MODULE_DIR / "task_first_curriculum.py")
M = _load_module("task_first_manifest_under_test", MODULE_DIR / "task_first_manifest.py")


def _gate():
    return {
        "min_attempts": 100,
        "enter_success_lower_bound": 0.90,
        "exit_success_lower_bound": 0.70,
        "enter_unsafe_upper_bound": 0.05,
        "exit_unsafe_upper_bound": 0.10,
        "enter_dwell_updates": 2,
        "exit_dwell_updates": 3,
        "max_stall_updates": 100,
        "stall_policy": "fail",
        "confidence_z": 1.959963984540054,
    }


def _action(index):
    return {
        "action_id": f"action_{index:03d}",
        "action_uid": index + 1,
        "motion_path": f"motions/action_{index:03d}.npz",
        "motion_sha256": hashlib.sha256(f"motion-{index}".encode()).hexdigest(),
        "strike_phase": 0.50,
        "family_sign": 1 if index % 2 == 0 else -1,
        "mount_normal_sign": -1,
        "position_half_extent_m": [0.25, 0.30, 0.15],
        "speed_delta_mps": 1.5,
        "face_cone_deg": 20.0,
        "base_center_shift_xy_m": [-0.05, 0.0],
        "base_half_extent_xy_m": [0.20, 0.15],
    }


def _document(action_count=5, *, training_authorized=True):
    actions = [_action(index) for index in range(action_count)]
    return {
        "schema_version": 1,
        "manifest_id": f"task_first_n{action_count}_v1",
        "training_authorized": training_authorized,
        "action_order": [action["action_id"] for action in actions],
        "actions": actions,
        "gate": _gate(),
        "holdout": {
            "seed": 20260727,
            "samples_per_action": 1000,
            "split_id": "heldout_v1",
        },
        "notes": "Host-only schema fixture.",
    }


def _write(tmp_path, document, *, name="manifest.json", **json_kwargs):
    path = tmp_path / name
    text = json.dumps(document, allow_nan=True, **json_kwargs)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_schema_supports_arbitrary_n_and_preserves_exact_order(tmp_path, action_count):
    path = _write(tmp_path, _document(action_count))
    receipt = M.load_task_first_manifest(path)
    manifest = receipt.manifest

    assert len(manifest.actions) == action_count
    assert manifest.action_order == tuple(
        f"action_{index:03d}" for index in range(action_count)
    )
    assert tuple(action.action_id for action in manifest.actions) == manifest.action_order
    assert receipt.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt.canonical_sha256 == M.canonical_manifest_sha256(manifest)
    with pytest.raises(FrozenInstanceError):
        manifest.manifest_id = "mutated"
    with pytest.raises(FrozenInstanceError):
        manifest.actions[0].speed_delta_mps = 100.0


def test_file_digest_is_startup_binding_and_canonical_digest_is_deterministic(tmp_path):
    document = _document()
    compact = _write(
        tmp_path,
        document,
        name="compact.json",
        sort_keys=True,
        separators=(",", ":"),
    )
    pretty = _write(
        tmp_path,
        dict(reversed(tuple(document.items()))),
        name="pretty.json",
        indent=2,
    )

    compact_receipt = M.load_task_first_manifest(compact)
    pretty_receipt = M.load_task_first_manifest(pretty)
    assert compact_receipt.file_sha256 != pretty_receipt.file_sha256
    assert compact_receipt.canonical_sha256 == pretty_receipt.canonical_sha256
    assert compact_receipt.manifest.to_mapping() == pretty_receipt.manifest.to_mapping()

    bound = M.load_task_first_manifest(
        compact, expected_sha256=compact_receipt.file_sha256
    )
    assert bound.file_sha256 == compact_receipt.file_sha256
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        M.load_task_first_manifest(compact, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="64 lowercase"):
        M.load_task_first_manifest(compact, expected_sha256="A" * 64)


def test_unauthorized_manifest_is_inspectable_metadata_but_not_launchable(tmp_path):
    path = _write(tmp_path, _document(training_authorized=False))
    receipt = M.load_task_first_manifest(path)
    assert receipt.manifest.training_authorized is False
    with pytest.raises(ValueError, match="metadata-only"):
        M.load_task_first_manifest(path, require_training_authorized=True)
    with pytest.raises(ValueError, match="must be a bool"):
        M.load_task_first_manifest(path, require_training_authorized=1)


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("top", "extra"),
        ("action", "extra"),
        ("gate", "extra"),
        ("holdout", "extra"),
    ],
)
def test_unknown_keys_are_rejected_at_every_schema_level(tmp_path, scope, field):
    document = _document()
    target = {
        "top": document,
        "action": document["actions"][0],
        "gate": document["gate"],
        "holdout": document["holdout"],
    }[scope]
    target[field] = 1
    with pytest.raises(ValueError, match="invalid keys"):
        M.load_task_first_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("top", "notes"),
        ("action", "face_cone_deg"),
        ("gate", "confidence_z"),
        ("holdout", "split_id"),
    ],
)
def test_missing_keys_are_rejected_at_every_schema_level(tmp_path, scope, field):
    document = _document()
    target = {
        "top": document,
        "action": document["actions"][0],
        "gate": document["gate"],
        "holdout": document["holdout"],
    }[scope]
    del target[field]
    with pytest.raises(ValueError, match="invalid keys"):
        M.load_task_first_manifest(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (("schema_version",), True, "plain integer"),
        (("training_authorized",), 1, "must be a bool"),
        (("actions", 0, "action_uid"), True, "plain integer"),
        (("actions", 0, "strike_phase"), True, "plain finite"),
        (("actions", 0, "position_half_extent_m", 1), True, "plain finite"),
        (("gate", "min_attempts"), True, "plain integer"),
        (("holdout", "seed"), True, "plain integer"),
        (("notes",), False, "must be a string"),
    ],
)
def test_bool_is_never_accepted_as_string_integer_or_number(
    tmp_path, location, value, message
):
    document = _document()
    target = document
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises((TypeError, ValueError), match=message):
        M.load_task_first_manifest(_write(tmp_path, document))


@pytest.mark.parametrize("constant", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_numbers_are_rejected_during_decode(tmp_path, constant):
    document = _document()
    document["actions"][0]["speed_delta_mps"] = constant
    with pytest.raises(ValueError, match="JSON constant"):
        M.load_task_first_manifest(_write(tmp_path, document))


def test_duplicate_json_keys_are_rejected(tmp_path):
    document = _document()
    encoded = json.dumps(document)
    duplicate = encoded[:-1] + ',"notes":"duplicate"}'
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        M.load_task_first_manifest(path)


@pytest.mark.parametrize(
    "motion_path",
    [
        "/abs/motion.npz",
        "../motion.npz",
        "motions/../../motion.npz",
        r"C:\motions\motion.npz",
        r"motions\..\motion.npz",
        ".",
    ],
)
def test_motion_path_must_be_unambiguous_relative_and_cannot_escape(
    tmp_path, motion_path
):
    document = _document()
    document["actions"][0]["motion_path"] = motion_path
    with pytest.raises(ValueError, match="motion_path"):
        M.load_task_first_manifest(_write(tmp_path, document))


def test_action_order_uid_and_sha_identity_fail_closed(tmp_path):
    swapped = _document()
    swapped["actions"][0], swapped["actions"][1] = (
        swapped["actions"][1],
        swapped["actions"][0],
    )
    with pytest.raises(ValueError, match="same IDs and order"):
        M.load_task_first_manifest(_write(tmp_path, swapped))

    duplicate_uid = _document()
    duplicate_uid["actions"][1]["action_uid"] = duplicate_uid["actions"][0]["action_uid"]
    with pytest.raises(ValueError, match="duplicate action_uid"):
        M.load_task_first_manifest(_write(tmp_path, duplicate_uid))

    for invalid_uid in (0, M.MAX_ACTION_UID + 1):
        invalid = _document()
        invalid["actions"][0]["action_uid"] = invalid_uid
        with pytest.raises(ValueError, match="action_uid"):
            M.load_task_first_manifest(
                _write(tmp_path, invalid, name=f"uid-{invalid_uid}.json")
            )

    invalid_sha = _document()
    invalid_sha["actions"][0]["motion_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="64 lowercase"):
        M.load_task_first_manifest(_write(tmp_path, invalid_sha, name="sha.json"))


def test_action_envelope_and_holdout_bounds_are_strict(tmp_path):
    cases = [
        (("actions", 0, "strike_phase"), 1.01, "strike_phase"),
        (("actions", 0, "family_sign"), 0, "family_sign"),
        (("actions", 0, "mount_normal_sign"), 0, "mount_normal_sign"),
        (("actions", 0, "position_half_extent_m", 0), -0.01, "position_half"),
        (("actions", 0, "speed_delta_mps"), -0.01, "speed_delta"),
        (("actions", 0, "face_cone_deg"), 90.01, "face_cone"),
        (("actions", 0, "base_half_extent_xy_m", 0), -0.01, "base_half"),
        (("holdout", "seed"), -1, "holdout.seed"),
        (("holdout", "samples_per_action"), 0, "samples_per_action"),
    ]
    for index, (location, value, message) in enumerate(cases):
        document = _document()
        target = document
        for key in location[:-1]:
            target = target[key]
        target[location[-1]] = value
        with pytest.raises(ValueError, match=message):
            M.load_task_first_manifest(
                _write(tmp_path, document, name=f"bounds-{index}.json")
            )


def test_empty_duplicate_and_mismatched_action_banks_are_rejected(tmp_path):
    empty = _document(1)
    empty["action_order"] = []
    empty["actions"] = []
    with pytest.raises(ValueError, match="at least one"):
        M.load_task_first_manifest(_write(tmp_path, empty, name="empty.json"))

    duplicate = _document()
    duplicate["action_order"][1] = duplicate["action_order"][0]
    duplicate["actions"][1]["action_id"] = duplicate["actions"][0]["action_id"]
    with pytest.raises(ValueError, match="duplicate action IDs"):
        M.load_task_first_manifest(_write(tmp_path, duplicate, name="duplicate.json"))

    missing = _document()
    missing["actions"].pop()
    with pytest.raises(ValueError, match="same IDs and order"):
        M.load_task_first_manifest(_write(tmp_path, missing, name="missing.json"))
