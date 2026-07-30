#!/usr/bin/env python3
"""Launch one fail-closed, single-GPU N=1 ActionBall reward-screen run.

This is deliberately a *diagnostic* launcher.  It does not consume or mint a
formal action-set identity, evaluator authority, promotion receipt, exact
resume claim, export authority, or judge authority.  Its four public
stages are:

* ``smoke``: exactly one environment and two PPO updates; and
* ``probe``: exactly 4096 environments and five PPO updates; and
* ``canary``: a bounded first reward-screen budget; and
* ``milestone1000``: exactly 4096 environments and 1001 PPO updates, saving
  every 100 updates so the diagnostic naturally emits ``model_1000``; and
* ``long``: exactly 4096 environments and 20001 PPO updates, saving every
  100 updates so the finite run ends after emitting ``model_20000``.

The launcher has no arbitrary Hydra override input.  It accepts one canonical
JSON spec, verifies an exact clean Git commit and a tracked N=1 bundle, binds
one of three reviewed reward profiles, claims a fresh absolute namespace,
holds the shared physical-GPU flock for the complete trainer lifetime, checks
the selected GPU is empty twice, and delegates Kit boot supervision to
``launch_kit_training_locked.sh``.

Typical use::

    python3 scripts/launch_n1_reward_screen_diagnostic.py plan --spec /abs/run.json
    python3 scripts/launch_n1_reward_screen_diagnostic.py launch \
      --spec /abs/run.json --confirm-claim <sha256 printed by plan>

The spec itself is operational state and therefore lives outside the clean
checkout.  Every scientific input it names must be an exact tracked blob in
the selected commit.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
SPEC_KIND = "n1_reward_screen_diagnostic_spec_v1"
CLAIM_KIND = "n1_reward_screen_diagnostic_claim_v1"
BUNDLE_KIND_V1 = "n1_contact_training_bundle_v1"
BUNDLE_KIND_V2 = "n1_contact_training_bundle_v2"
BUNDLE_KIND = BUNDLE_KIND_V2
ALLOWED_BUNDLE_IDENTITIES = frozenset(
    ((1, BUNDLE_KIND_V1), (2, BUNDLE_KIND_V2))
)
CONTACT_KIND = "n1_contact_alignment_receipt_v1"
DYNAMIC_READY_KIND = "agibot_a3_action_dynamic_ready_candidate_v1"
NOMINAL_HOLD_RECEIPT_KIND = "isaac_action_ball_nominal_hold_v1"
EXPERIMENT_NAME = "agibot_a3_hope_action_ball_n1_reward_screen_diagnostic"
ALLOWED_ACTIONS = frozenset(("bh_loop_c", "bh_block"))
ALLOWED_STAGES = frozenset(
    ("smoke", "probe", "canary", "milestone1000", "long")
)
ALLOWED_SCOPES = frozenset(("upper", "full"))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_n1_reward_screen_diagnostic.py"
)
TRAIN_SOURCE = "hope_training/whole_body_tracking/scripts/train.py"
TASK_SOURCE = (
    "hope_training/whole_body_tracking/cfg/task/"
    "HOPEPingPongActionBall.yaml"
)
KIT_LAUNCHER_SOURCE = (
    "hope_training/whole_body_tracking/scripts/"
    "launch_kit_training_locked.sh"
)
WBT_RELATIVE = Path("hope_training/whole_body_tracking")
MDP_RELATIVE = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
CONTACT_GEOMETRY_SOURCE = f"{MDP_RELATIVE}/racket_contact_geometry.py"
SOLVER_IMPLEMENTATION_SOURCES = (
    "hope_commands.py",
    "continuous_questions.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
    "counter_rally.py",
    "counter_rally_torch.py",
)

# The clean Git checkout intentionally excludes the proprietary A3 URDF and
# generated USD closure.  Pin the reviewed Pod runtime copy by every file that
# the root USD references; a model.usd-only check is not a complete asset
# identity.  Tests may replace these constants with fixture bytes.
A3_RUNTIME_USD_BUNDLE_SHA256: Mapping[str, str] = {
    ".asset_hash": "3816a1a4bbca423e575650b6d6065f5141a7c840b02dd30c72d4278a225ed499",
    "config.yaml": "3e35ad4c3ef7c21a10ce413be3ce28777bb83afee4b63fc245b30bd59a9818c2",
    "configuration/model_base.usd": (
        "8e521141bfee4274b8a2369d382cdd8aac9bb1cfcae5bfa480666a1935a7fb42"
    ),
    "configuration/model_physics.usd": (
        "5b5fc00b96566be295a0cd4eb6b0cd276e360d9cca189057cef452ad0bfc7981"
    ),
    "configuration/model_sensor.usd": (
        "c76c5bdd9e9b5434d72b45c9001858a9c80363656272011ed50d1419149ca60a"
    ),
    "model.usd": "1b3fecd7685cd98ca80de226fbf89985b77b8a8cfc6a36f18fcc22e65080693c",
}
PRIVATE_GLU_LIBRARY = "libGLU.so.1.3.1"
PRIVATE_GLU_SONAME = "libGLU.so.1"
PRIVATE_GLU_SHA256 = (
    "af791d1ee2acf25417f612290e634248fd716cf5da0374ba21160fb264eaeab4"
)

# The reviewed screen changes either task tracking or imitation, never both.
# Outcome, regularization, soft/hard-limit, table and fall terms stay
# byte-identical.
REWARD_PROFILES: Mapping[str, Mapping[str, float]] = {
    "current_low": {
        "racket_position_weight": 4.0,
        "racket_velocity_weight": 0.5,
        "racket_normal_weight": 0.5,
        "motion_scale": 1.0,
    },
    "mimic_x2": {
        "racket_position_weight": 4.0,
        "racket_velocity_weight": 0.5,
        "racket_normal_weight": 0.5,
        "motion_scale": 2.0,
    },
    "task_strong_x4": {
        "racket_position_weight": 16.0,
        "racket_velocity_weight": 2.0,
        "racket_normal_weight": 2.0,
        "motion_scale": 1.0,
    },
}

_SPEC_KEYS = (
    "schema_version",
    "kind",
    "source",
    "action_id",
    "scope",
    "bundle",
    "policy_contract_sha256",
    "reward_profile",
    "expected_effective_reward_recipe_sha256",
    "seed",
    "stage",
    "num_envs",
    "max_iterations",
    "save_interval",
    "gpu",
    "namespace",
    "log_path",
)
_SOURCE_KEYS = ("checkout", "commit_sha", "isaac_python")
_PIN_KEYS = ("path", "sha256")
_GPU_KEYS = (
    "index",
    "uuid",
    "owner",
    "lock_path",
    "require_empty",
)
_BUNDLE_KEYS = (
    "schema_version",
    "artifact_type",
    "action_id",
    "action_uid",
    "scope",
    "source_manifest",
    "motion",
    "profile_pins",
    "prototype",
    "manifest",
    "contact_alignment",
    "geometry",
    "claims",
)
_BUNDLE_V2_KEYS = (*_BUNDLE_KEYS, "dynamic_ready")
_DYNAMIC_READY_KEYS = ("artifact", "nominal_hold_receipt")
_BUNDLE_CLAIM_KEYS = (
    "selector_executed",
    "action_identity_frozen_before_ball_sampling",
    "contact_alignment_claim",
    "landing_claim",
    "post_bounce_claim",
    "baseline_crossing_claim",
    "deployment_claim",
)
_FULL_DIAGNOSTIC_CLAIM_KEYS = (
    *_BUNDLE_CLAIM_KEYS,
    "diagnostic_only",
    "training_authorized",
)
_CONTACT_KEYS = (
    "schema_version",
    "artifact_type",
    "status",
    "action_id",
    "action_uid",
    "scope",
    "source_manifest",
    "motion",
    "profile_pins",
    "geometry",
    "timing",
    "frames",
    "alignment",
    "claims",
)
_PROFILE_PIN_KEYS = (
    "path",
    "sha256",
    "solver_profile_sha256",
    "physics_profile_sha256",
    "geometry_payload_sha256",
)
_TIMING_KEYS = (
    "fps_hz",
    "frame_count",
    "contact_frame",
    "manifest_t_hit_s",
    "motion_t_hit_s",
    "manifest_t_cycle_s",
    "motion_t_cycle_s",
    "t_hit_abs_error_s",
    "t_cycle_abs_error_s",
)
_FRAME_KEYS = (
    "task_contact_frame",
    "teacher_reference_frame",
    "world_z_origin",
)
_UPPER_ALIGNMENT_KEYS = (
    "threshold_m",
    "ready_root_z_w_m",
    "legacy_absolute_contact_z_w_m",
    "corrected_contact_offset_z_b_yaw_m",
    "task_contact_offset_center_b_yaw_m",
    "teacher_racket_site_b_yaw_m",
    "teacher_selected_face_center_b_yaw_m",
    "task_to_teacher_site_distance_m",
    "task_to_teacher_face_center_distance_m",
    "center_gate_point",
    "center_gate_distance_m",
    "center_within_threshold",
)
_FULL_ALIGNMENT_KEYS = (
    "threshold_m",
    "ready_root_z_w_m",
    "retargeted_contact_center_z_w_m",
    "contact_center_authority",
    "upper_contact_center_preserved",
    "task_contact_offset_center_b_yaw_m",
    "teacher_racket_site_b_yaw_m",
    "teacher_selected_face_center_b_yaw_m",
    "task_to_teacher_site_distance_m",
    "task_to_teacher_face_center_distance_m",
    "center_gate_point",
    "center_gate_distance_m",
    "center_within_threshold",
)
_UPPER_RETARGETED_ALIGNMENT_KEYS = _FULL_ALIGNMENT_KEYS


class LaunchRefused(RuntimeError):
    """A fail-closed, user-actionable launch refusal."""


def _reject_duplicate_pairs(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaunchRefused(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LaunchRefused(
                    f"{name} contains forbidden JSON constant {token}"
                )
            ),
        )
    except LaunchRefused:
        raise
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise LaunchRefused(f"{name} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise LaunchRefused(f"{name} root must be an object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LaunchRefused("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_ascii_sha256(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LaunchRefused("value is not canonical ASCII JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_dict(
    value: Any, keys: Sequence[str], *, name: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise LaunchRefused(f"{name} must be an object")
    expected = frozenset(keys)
    actual = frozenset(value)
    if actual != expected:
        raise LaunchRefused(
            f"{name} keys differ: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise LaunchRefused(f"{name} must be 64 lowercase hexadecimal chars")
    return value


def _plain_int(
    value: Any, *, name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum:
        raise LaunchRefused(f"{name} must be a plain integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise LaunchRefused(f"{name} must be <= {maximum}")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LaunchRefused(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LaunchRefused(f"{name} must be a finite number")
    return result


def _absolute_path(
    value: Any, *, name: str, must_exist: bool = False
) -> Path:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\n" in value
        or not os.path.isabs(value)
        or os.path.normpath(value) != value
    ):
        raise LaunchRefused(
            f"{name} must be a normalized absolute path"
        )
    path = Path(value)
    if must_exist and not path.exists():
        raise LaunchRefused(f"{name} does not exist: {path}")
    return path


def _relative_path(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
    ):
        raise LaunchRefused(f"{name} must be a non-empty POSIX relative path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise LaunchRefused(f"{name} is not lexically normalized")
    return value


def _stable_regular_file(path: Path, *, name: str) -> os.stat_result:
    try:
        before = path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be inspected: {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise LaunchRefused(f"{name} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise LaunchRefused(f"{name} must not resolve through a symlink")
    return before


def _run_git(
    checkout: Path, args: Sequence[str], *, text: bool = True
) -> subprocess.CompletedProcess[Any]:
    git = shutil.which("git", path="/usr/bin:/bin:/usr/local/bin")
    if git is None:
        raise LaunchRefused("git is unavailable")
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "LANG": "C", "LC_ALL": "C"}
    return subprocess.run(
        [git, "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=text,
        env=env,
    )


def _verify_clean_source(
    checkout: Path, expected_commit: str
) -> dict[str, Any]:
    if not checkout.is_absolute() or checkout.resolve(strict=True) != checkout:
        raise LaunchRefused(
            "source.checkout must be an existing real absolute directory"
        )
    head = _run_git(checkout, ("rev-parse", "--verify", "HEAD"))
    if head.returncode != 0:
        raise LaunchRefused(f"cannot read source HEAD: {head.stderr.strip()}")
    actual = head.stdout.strip()
    if actual != expected_commit:
        raise LaunchRefused(
            f"source HEAD differs: expected={expected_commit}, actual={actual}"
        )
    status_result = _run_git(
        checkout, ("status", "--porcelain=v1", "--untracked-files=all")
    )
    if status_result.returncode != 0:
        raise LaunchRefused(
            f"cannot inspect source cleanliness: {status_result.stderr.strip()}"
        )
    if status_result.stdout:
        first = status_result.stdout.splitlines()[0]
        raise LaunchRefused(f"source checkout is dirty: {first}")
    return {"checkout": str(checkout), "commit_sha": actual, "clean": True}


def _verify_tracked_file(
    checkout: Path,
    commit_sha: str,
    pin: Any,
    *,
    name: str,
    extra_keys: Sequence[str] = (),
) -> tuple[dict[str, Any], Path]:
    row = _exact_dict(pin, (*_PIN_KEYS, *extra_keys), name=name)
    relative = _relative_path(row["path"], name=f"{name}.path")
    expected_sha = _sha256(row["sha256"], name=f"{name}.sha256")
    path = checkout / relative
    _stable_regular_file(path, name=name)
    stage = _run_git(checkout, ("ls-files", "--stage", "--", relative))
    if stage.returncode != 0 or not stage.stdout.strip():
        raise LaunchRefused(f"{name} is not tracked at {relative}")
    rows = stage.stdout.splitlines()
    if len(rows) != 1:
        raise LaunchRefused(f"{name} has ambiguous Git index identity")
    parts = rows[0].split(None, 3)
    if len(parts) != 4 or parts[0] not in ("100644", "100755"):
        raise LaunchRefused(f"{name} must be a normal tracked Git blob")
    blob = _run_git(
        checkout,
        ("show", f"{commit_sha}:{relative}"),
        text=False,
    )
    if blob.returncode != 0:
        detail = blob.stderr.decode("utf-8", "replace").strip()
        raise LaunchRefused(f"{name} is absent from exact commit: {detail}")
    committed_sha = hashlib.sha256(blob.stdout).hexdigest()
    observed_sha = sha256_file(path)
    if committed_sha != expected_sha or observed_sha != expected_sha:
        raise LaunchRefused(
            f"{name} SHA differs: pin={expected_sha}, "
            f"commit={committed_sha}, worktree={observed_sha}"
        )
    return dict(row), path


def _load_tracked_json(
    checkout: Path,
    commit_sha: str,
    pin: Any,
    *,
    name: str,
    extra_keys: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized, path = _verify_tracked_file(
        checkout,
        commit_sha,
        pin,
        name=name,
        extra_keys=extra_keys,
    )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LaunchRefused(f"{name} cannot be read: {exc}") from exc
    return normalized, _strict_json_bytes(raw, name=name)


def _same_pin(left: Any, right: Any, *, name: str) -> None:
    if type(left) is not dict or type(right) is not dict:
        raise LaunchRefused(f"{name} pins must be objects")
    for field in _PIN_KEYS:
        if left.get(field) != right.get(field):
            raise LaunchRefused(f"{name} {field} differs")


def _verify_content_seal(
    document: Mapping[str, Any],
    *,
    name: str,
    ensure_ascii: bool,
) -> str:
    content_sha = _sha256(
        document.get("content_sha256"),
        name=f"{name}.content_sha256",
    )
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    actual = (
        _canonical_ascii_sha256(unsigned)
        if ensure_ascii
        else canonical_sha256(unsigned)
    )
    if actual != content_sha:
        raise LaunchRefused(
            f"{name}.content_sha256 does not seal canonical content"
        )
    return content_sha


def _validate_dynamic_ready(
    checkout: Path,
    commit_sha: str,
    value: Any,
    *,
    action_id: str,
    motion_sha256: str,
) -> dict[str, dict[str, Any]]:
    row = _exact_dict(
        value, _DYNAMIC_READY_KEYS, name="N1 bundle.dynamic_ready"
    )
    artifact_pin, candidate = _load_tracked_json(
        checkout,
        commit_sha,
        row["artifact"],
        name="N1 dynamic-ready artifact",
    )
    candidate_content_sha = _verify_content_seal(
        candidate,
        name="N1 dynamic-ready artifact",
        ensure_ascii=True,
    )
    robot = candidate.get("robot")
    sources = candidate.get("sources")
    stable_motion = (
        sources.get("stable_motion") if type(sources) is dict else None
    )
    if (
        candidate.get("schema_version") != 1
        or candidate.get("kind") != DYNAMIC_READY_KIND
        or candidate.get("action_id") != action_id
        or type(robot) is not dict
        or robot.get("family") != "AgiBot A3"
        or type(stable_motion) is not dict
        or stable_motion.get("frame_index") != 0
        or stable_motion.get("sha256") != motion_sha256
    ):
        raise LaunchRefused(
            "N1 dynamic-ready artifact is not the exact A3 action/motion"
        )

    receipt_pin, receipt = _load_tracked_json(
        checkout,
        commit_sha,
        row["nominal_hold_receipt"],
        name="N1 nominal-hold receipt",
    )
    _verify_content_seal(
        receipt,
        name="N1 nominal-hold receipt",
        ensure_ascii=False,
    )
    receipt_artifact = receipt.get("artifact")
    required_gate = candidate.get("required_next_gate")
    required_terminations = (
        required_gate.get("zero_terminal_required")
        if type(required_gate) is dict
        else None
    )
    active_terminations = receipt.get("active_terminations")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != NOMINAL_HOLD_RECEIPT_KIND
        or receipt.get("verdict") != "PASS"
        or receipt.get("action_id") != action_id
        or receipt.get("motion_sha256") != motion_sha256
        or receipt.get("plant_contract_match") is not True
        or receipt.get("terminal_reasons") != []
        or receipt.get("generic_terminated") is not False
        or receipt.get("generic_truncated") is not False
        or type(receipt_artifact) is not dict
        or receipt_artifact.get("sha256") != artifact_pin["sha256"]
        or receipt_artifact.get("content_sha256")
        != candidate_content_sha
        or type(required_gate) is not dict
        or required_gate.get("kind") != NOMINAL_HOLD_RECEIPT_KIND
        or type(required_terminations) is not list
        or type(active_terminations) is not list
        or not all(
            type(reason) is str and reason in active_terminations
            for reason in required_terminations
        )
    ):
        raise LaunchRefused(
            "N1 nominal-hold receipt does not prove the exact dynamic-ready "
            "action/motion/plant with zero terminal"
        )
    return {
        "artifact": artifact_pin,
        "nominal_hold_receipt": receipt_pin,
    }


def _vec3(value: Any, *, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) for item in value)
    ):
        raise LaunchRefused(f"{name} must be a three-number JSON array")
    return tuple(_finite(item, name=f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _validate_contact_receipt(
    document: dict[str, Any],
    *,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    row = _exact_dict(document, _CONTACT_KEYS, name="contact receipt")
    if row["schema_version"] != 1:
        raise LaunchRefused("contact receipt schema_version must be 1")
    if row["artifact_type"] != CONTACT_KIND:
        raise LaunchRefused(
            f"contact receipt artifact_type must be {CONTACT_KIND!r}"
        )
    if row["status"] != "PASS":
        raise LaunchRefused("contact receipt status must be PASS")
    for field in ("action_id", "action_uid", "scope"):
        if row[field] != bundle[field]:
            raise LaunchRefused(f"contact receipt {field} differs from bundle")
    for field in ("source_manifest", "motion"):
        _same_pin(row[field], bundle[field], name=f"contact.{field}")
    if row["geometry"] != bundle["geometry"]:
        raise LaunchRefused("contact geometry pin differs from the bundle")
    if row["profile_pins"] != bundle["profile_pins"]:
        raise LaunchRefused(
            "contact profile_pins differ from the bundle"
        )

    action = manifest["actions"][0]
    timing = _exact_dict(row["timing"], _TIMING_KEYS, name="contact.timing")
    fps = _finite(timing["fps_hz"], name="contact.timing.fps_hz")
    if fps <= 0.0:
        raise LaunchRefused("contact timing fps_hz must be positive")
    frame_count = _plain_int(
        timing["frame_count"], name="contact.timing.frame_count", minimum=2
    )
    contact_frame = _plain_int(
        timing["contact_frame"],
        name="contact.timing.contact_frame",
        minimum=0,
        maximum=frame_count - 1,
    )
    manifest_t_hit = _finite(
        timing["manifest_t_hit_s"], name="contact.timing.manifest_t_hit_s"
    )
    manifest_t_cycle = _finite(
        timing["manifest_t_cycle_s"], name="contact.timing.manifest_t_cycle_s"
    )
    motion_t_hit = _finite(
        timing["motion_t_hit_s"], name="contact.timing.motion_t_hit_s"
    )
    motion_t_cycle = _finite(
        timing["motion_t_cycle_s"], name="contact.timing.motion_t_cycle_s"
    )
    if abs(manifest_t_hit - float(action["reference_t_hit_s"])) > 1.0e-9:
        raise LaunchRefused("contact manifest_t_hit_s differs from N1 manifest")
    if abs(manifest_t_cycle - float(action["reference_t_cycle_s"])) > 1.0e-9:
        raise LaunchRefused("contact manifest_t_cycle_s differs from N1 manifest")
    if abs(motion_t_hit - contact_frame / fps) > 1.0e-9:
        raise LaunchRefused("contact motion_t_hit_s differs from contact_frame/fps")
    if abs(motion_t_cycle - (frame_count - 1) / fps) > 1.0e-9:
        raise LaunchRefused(
            "contact motion_t_cycle_s differs from (frame_count-1)/fps"
        )
    hit_error = _finite(
        timing["t_hit_abs_error_s"], name="contact.timing.t_hit_abs_error_s"
    )
    cycle_error = _finite(
        timing["t_cycle_abs_error_s"],
        name="contact.timing.t_cycle_abs_error_s",
    )
    if abs(hit_error - abs(manifest_t_hit - motion_t_hit)) > 1.0e-9:
        raise LaunchRefused("contact t_hit_abs_error_s is not recomputable")
    if abs(cycle_error - abs(manifest_t_cycle - motion_t_cycle)) > 1.0e-9:
        raise LaunchRefused("contact t_cycle_abs_error_s is not recomputable")
    if hit_error > 1.0 / fps + 1.0e-12 or cycle_error > 1.0 / fps + 1.0e-12:
        raise LaunchRefused("contact timing differs by more than one source frame")

    frames = _exact_dict(row["frames"], _FRAME_KEYS, name="contact.frames")
    expected_frames = {
        "task_contact_frame": "B_yaw_relative_to_actual_spawn_goal",
        "teacher_reference_frame": "B_yaw_at_frame0",
        "world_z_origin": "floor",
    }
    if frames != expected_frames:
        raise LaunchRefused(
            "contact frame contract is not base-relative task / frame0 teacher / floor-z"
        )

    scope = bundle["scope"]
    raw_alignment = row["alignment"]
    if scope == "full":
        alignment_keys = _FULL_ALIGNMENT_KEYS
        alignment_mode = "full_retargeted"
    elif (
        isinstance(raw_alignment, dict)
        and set(raw_alignment) == set(_UPPER_RETARGETED_ALIGNMENT_KEYS)
    ):
        alignment_keys = _UPPER_RETARGETED_ALIGNMENT_KEYS
        alignment_mode = "stable_upper_retargeted"
    else:
        alignment_keys = _UPPER_ALIGNMENT_KEYS
        alignment_mode = "legacy_upper_corrected_z"
    alignment = _exact_dict(
        raw_alignment, alignment_keys, name="contact.alignment"
    )
    threshold = _finite(
        alignment["threshold_m"], name="contact.alignment.threshold_m"
    )
    if abs(threshold - 0.03) > 1.0e-12:
        raise LaunchRefused("contact center threshold must be exactly 0.03 m")
    ready_z = _finite(
        alignment["ready_root_z_w_m"],
        name="contact.alignment.ready_root_z_w_m",
    )
    task = _vec3(
        alignment["task_contact_offset_center_b_yaw_m"],
        name="contact.alignment.task_contact_offset_center_b_yaw_m",
    )
    teacher_site = _vec3(
        alignment["teacher_racket_site_b_yaw_m"],
        name="contact.alignment.teacher_racket_site_b_yaw_m",
    )
    teacher_face = _vec3(
        alignment["teacher_selected_face_center_b_yaw_m"],
        name="contact.alignment.teacher_selected_face_center_b_yaw_m",
    )
    manifest_center = _vec3(
        action["ball_profile"]["contact_offset_center_b_yaw_m"],
        name="manifest contact center",
    )
    if alignment_mode == "legacy_upper_corrected_z":
        legacy_z = _finite(
            alignment["legacy_absolute_contact_z_w_m"],
            name="contact.alignment.legacy_absolute_contact_z_w_m",
        )
        corrected_z = _finite(
            alignment["corrected_contact_offset_z_b_yaw_m"],
            name="contact.alignment.corrected_contact_offset_z_b_yaw_m",
        )
        if abs((legacy_z - ready_z) - corrected_z) > 1.0e-9:
            raise LaunchRefused(
                "contact corrected z is not legacy world z minus ready root z"
            )
        if abs(task[2] - corrected_z) > 1.0e-9:
            raise LaunchRefused(
                "contact task z is not the corrected base-relative z"
            )
    else:
        retargeted_z = _finite(
            alignment["retargeted_contact_center_z_w_m"],
            name="contact.alignment.retargeted_contact_center_z_w_m",
        )
        expected_authority = (
            "full_motion_selected_rubber_face_center_at_explicit_strike_frame"
            if alignment_mode == "full_retargeted"
            else (
                "a3_stable_upper_selected_rubber_face_center_at_pinned_"
                "strike_frame"
            )
        )
        if alignment["contact_center_authority"] != expected_authority:
            raise LaunchRefused(
                "retargeted contact center authority does not match its "
                "full/stable-upper scope"
            )
        if alignment["upper_contact_center_preserved"] is not False:
            raise LaunchRefused(
                "retargeted contact center must not preserve the old upper center"
            )
        if abs((retargeted_z - ready_z) - task[2]) > 1.0e-9:
            raise LaunchRefused(
                "retargeted contact z is not ready root z plus task z"
            )
    if _distance(task, manifest_center) > 1.0e-9:
        raise LaunchRefused("contact task center differs from N1 manifest")
    site_distance = _finite(
        alignment["task_to_teacher_site_distance_m"],
        name="contact.alignment.task_to_teacher_site_distance_m",
    )
    face_distance = _finite(
        alignment["task_to_teacher_face_center_distance_m"],
        name="contact.alignment.task_to_teacher_face_center_distance_m",
    )
    if abs(site_distance - _distance(task, teacher_site)) > 1.0e-9:
        raise LaunchRefused("contact site distance is not recomputable")
    if abs(face_distance - _distance(task, teacher_face)) > 1.0e-9:
        raise LaunchRefused("contact face-center distance is not recomputable")
    if alignment["center_gate_point"] != "selected_rubber_face_center":
        raise LaunchRefused(
            "contact center gate must use selected_rubber_face_center"
        )
    gate_distance = _finite(
        alignment["center_gate_distance_m"],
        name="contact.alignment.center_gate_distance_m",
    )
    if abs(gate_distance - face_distance) > 1.0e-9:
        raise LaunchRefused("contact center gate distance differs from face distance")
    if gate_distance > threshold or alignment["center_within_threshold"] is not True:
        raise LaunchRefused("contact center is outside the 0.03 m gate")

    claim_keys = (
        _FULL_DIAGNOSTIC_CLAIM_KEYS
        if bundle["scope"] == "full"
        else _BUNDLE_CLAIM_KEYS
    )
    claims = _exact_dict(row["claims"], claim_keys, name="contact.claims")
    _validate_claims(
        claims, name="contact.claims", scope=bundle["scope"]
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "t_hit_error_s": hit_error,
        "t_cycle_error_s": cycle_error,
        "center_gate_distance_m": gate_distance,
        "center_threshold_m": threshold,
    }


def _validate_claims(
    claims: dict[str, Any], *, name: str, scope: str
) -> None:
    expected = {
        "selector_executed": False,
        "action_identity_frozen_before_ball_sampling": True,
        "contact_alignment_claim": True,
        "landing_claim": False,
        "post_bounce_claim": False,
        "baseline_crossing_claim": False,
        "deployment_claim": False,
    }
    if scope == "full":
        expected.update(
            {
                "diagnostic_only": True,
                "training_authorized": False,
            }
        )
    if claims != expected:
        raise LaunchRefused(
            f"{name} must claim frozen-action contact alignment only"
        )


def _validate_bundle(
    checkout: Path,
    commit_sha: str,
    bundle_pin: dict[str, Any],
    *,
    expected_action: str,
    expected_scope: str,
    require_dynamic_ready: bool = False,
) -> dict[str, Any]:
    normalized_bundle_pin, bundle = _load_tracked_json(
        checkout, commit_sha, bundle_pin, name="N1 bundle"
    )
    identity = (
        bundle.get("schema_version"),
        bundle.get("artifact_type"),
    )
    if identity not in ALLOWED_BUNDLE_IDENTITIES:
        raise LaunchRefused(
            "N1 bundle must be one supported schema/artifact pair: "
            f"{sorted(ALLOWED_BUNDLE_IDENTITIES)!r}"
        )
    is_v2 = identity == (2, BUNDLE_KIND_V2)
    bundle = _exact_dict(
        bundle,
        _BUNDLE_V2_KEYS if is_v2 else _BUNDLE_KEYS,
        name="N1 bundle",
    )
    if require_dynamic_ready and not is_v2:
        raise LaunchRefused(
            "dynamic-ready launch requires schema 2 / "
            f"{BUNDLE_KIND_V2!r}; schema-v1 remains read-compatible only"
        )
    if bundle["action_id"] != expected_action:
        raise LaunchRefused("N1 bundle action_id differs from spec")
    _plain_int(bundle["action_uid"], name="N1 bundle.action_uid", minimum=1)
    if bundle["scope"] != expected_scope:
        raise LaunchRefused("N1 bundle scope differs from spec")
    claim_keys = (
        _FULL_DIAGNOSTIC_CLAIM_KEYS
        if expected_scope == "full"
        else _BUNDLE_CLAIM_KEYS
    )
    claims = _exact_dict(
        bundle["claims"], claim_keys, name="N1 bundle.claims"
    )
    _validate_claims(
        claims, name="N1 bundle.claims", scope=expected_scope
    )

    source_manifest_pin, _source_manifest = _load_tracked_json(
        checkout,
        commit_sha,
        bundle["source_manifest"],
        name="N1 bundle source manifest",
    )
    motion_pin, _motion_path = _verify_tracked_file(
        checkout, commit_sha, bundle["motion"], name="N1 bundle motion"
    )
    dynamic_ready = (
        _validate_dynamic_ready(
            checkout,
            commit_sha,
            bundle["dynamic_ready"],
            action_id=expected_action,
            motion_sha256=motion_pin["sha256"],
        )
        if is_v2
        else None
    )
    geometry_pin, _geometry_path = _verify_tracked_file(
        checkout,
        commit_sha,
        bundle["geometry"],
        name="N1 bundle geometry",
        extra_keys=("payload_sha256", "kind"),
    )
    if geometry_pin["path"] != CONTACT_GEOMETRY_SOURCE:
        raise LaunchRefused(
            "N1 bundle geometry must pin the runtime racket_contact_geometry.py"
        )
    profile_pin, profile_document = _load_tracked_json(
        checkout,
        commit_sha,
        bundle["profile_pins"],
        name="N1 bundle profile pins",
        extra_keys=(
            "solver_profile_sha256",
            "physics_profile_sha256",
            "geometry_payload_sha256",
        ),
    )
    for field in ("solver_profile_sha256", "physics_profile_sha256"):
        if profile_document.get(field) != profile_pin[field]:
            raise LaunchRefused(
                f"N1 profile document {field} differs from bundle pin"
            )
    for payload_name, digest_field in (
        ("solver_payload", "solver_profile_sha256"),
        ("physics_payload", "physics_profile_sha256"),
    ):
        payload_value = profile_document.get(payload_name)
        if payload_value is None:
            raise LaunchRefused(
                f"N1 profile document lacks {payload_name}"
            )
        if canonical_sha256(payload_value) != profile_pin[digest_field]:
            raise LaunchRefused(
                f"N1 profile {payload_name} canonical SHA differs from "
                f"{digest_field}"
            )
    contact_geometry = _exact_dict(
        profile_document.get("contact_geometry"),
        ("payload", "sha256"),
        name="N1 profile contact_geometry",
    )
    contact_geometry_sha = _sha256(
        contact_geometry["sha256"],
        name="N1 profile contact_geometry.sha256",
    )
    if canonical_sha256(contact_geometry["payload"]) != contact_geometry_sha:
        raise LaunchRefused(
            "N1 profile contact_geometry SHA does not seal its payload"
        )
    if (
        contact_geometry_sha != profile_pin["geometry_payload_sha256"]
        or geometry_pin["payload_sha256"] != contact_geometry_sha
    ):
        raise LaunchRefused(
            "N1 contact-geometry payload SHA differs across profile and source pin"
        )
    contact_kind = (
        contact_geometry["payload"].get("kind")
        if type(contact_geometry["payload"]) is dict
        else None
    )
    if geometry_pin["kind"] != contact_kind:
        raise LaunchRefused(
            "N1 contact-geometry kind differs across payload and source pin"
        )
    solver_payload = profile_document["solver_payload"]
    if solver_payload.get("contact_geometry") != contact_geometry:
        raise LaunchRefused(
            "N1 solver payload does not bind the exact contact geometry"
        )
    source_hashes = profile_document.get(
        "solver_implementation_source_sha256"
    )
    if (
        type(source_hashes) is not dict
        or set(source_hashes) != set(SOLVER_IMPLEMENTATION_SOURCES)
        or solver_payload.get("implementation_source_sha256") != source_hashes
    ):
        raise LaunchRefused(
            "N1 profile must bind the exact seven solver implementation sources"
        )
    for filename in SOLVER_IMPLEMENTATION_SOURCES:
        source_sha = _sha256(
            source_hashes[filename],
            name=f"N1 profile source {filename}",
        )
        _verify_tracked_file(
            checkout,
            commit_sha,
            {
                "path": f"{MDP_RELATIVE}/{filename}",
                "sha256": source_sha,
            },
            name=f"N1 solver implementation {filename}",
        )
    if (
        source_hashes["racket_contact_geometry.py"]
        != geometry_pin["sha256"]
    ):
        raise LaunchRefused(
            "N1 geometry source SHA differs from solver implementation pin"
        )
    prototype_pin, prototype = _load_tracked_json(
        checkout,
        commit_sha,
        bundle["prototype"],
        name="N1 bundle prototype",
        extra_keys=("schema_version", "scope"),
    )
    if (
        prototype_pin["schema_version"] != 2
        or prototype_pin["scope"] != expected_scope
        or prototype.get("schema_version") != 2
    ):
        raise LaunchRefused(
            "N1 prototype must be schema 2 with the exact spec scope"
        )
    prototype_scopes = prototype.get("scopes")
    if (
        type(prototype_scopes) is not dict
        or set(prototype_scopes) != {expected_scope}
    ):
        raise LaunchRefused(
            "N1 prototype document must contain exactly the spec scope"
        )
    full_solver_preflight = None
    if expected_scope == "full":
        provenance = prototype.get("provenance")
        if type(provenance) is not dict:
            raise LaunchRefused(
                "full N1 prototype is missing exact solver admission provenance"
            )
        preflight = provenance.get("full_solver_admission_preflight")
        if type(preflight) is not dict:
            raise LaunchRefused(
                "full N1 prototype is missing full_solver_admission_preflight"
            )
        if (
            preflight.get("schema_version") != 1
            or preflight.get("kind")
            != "full_fixed_action_exact_solver_admission_preflight_v1"
        ):
            raise LaunchRefused(
                "full N1 solver admission preflight has an unknown schema/kind"
            )
        proposal_count = _plain_int(
            preflight.get("proposal_count"),
            name="full solver preflight proposal_count",
            minimum=1,
        )
        admitted_count = _plain_int(
            preflight.get("admitted_count"),
            name="full solver preflight admitted_count",
            minimum=1,
            maximum=proposal_count,
        )
        rejected_count = _plain_int(
            preflight.get("rejected_count"),
            name="full solver preflight rejected_count",
            minimum=0,
            maximum=proposal_count,
        )
        if admitted_count + rejected_count != proposal_count:
            raise LaunchRefused(
                "full N1 solver admission counts do not conserve proposals"
            )
        admit_rate = preflight.get("admit_rate")
        if (
            isinstance(admit_rate, bool)
            or not isinstance(admit_rate, (int, float))
            or not math.isfinite(float(admit_rate))
            or not math.isclose(
                float(admit_rate),
                admitted_count / proposal_count,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise LaunchRefused(
                "full N1 solver admission rate differs from exact counts"
            )
        diagnostic_gate = preflight.get("diagnostic_gate")
        if type(diagnostic_gate) is not dict:
            raise LaunchRefused(
                "full N1 solver admission diagnostic gate is missing"
            )
        if (
            diagnostic_gate.get("status") != "PASS"
            or diagnostic_gate.get("runtime_per_birth_redraw_replay") is not False
            or _plain_int(
                diagnostic_gate.get("environment_count"),
                name="full solver preflight diagnostic environment_count",
                minimum=1,
            )
            != 4096
            or _plain_int(
                diagnostic_gate.get("zero_admission_canary_group_count"),
                name="full solver preflight zero-admission group count",
                minimum=0,
            )
            != 0
        ):
            raise LaunchRefused(
                "full N1 solver admission diagnostic gate is not PASS"
            )
        full_solver_preflight = {
            "schema_version": 1,
            "kind": preflight["kind"],
            "proposal_count": proposal_count,
            "admitted_count": admitted_count,
            "rejected_count": rejected_count,
            "admit_rate": float(admit_rate),
            "diagnostic_status": "PASS",
        }
    manifest_pin, manifest = _load_tracked_json(
        checkout,
        commit_sha,
        bundle["manifest"],
        name="N1 bundle manifest",
        extra_keys=("schema_version", "action_order"),
    )
    if (
        manifest_pin["schema_version"] != 3
        or manifest_pin["action_order"] != [expected_action]
        or manifest.get("schema_version") != 3
        or manifest.get("action_order") != [expected_action]
        or manifest.get("mobility_mode") != "no_move"
    ):
        raise LaunchRefused(
            "N1 manifest must be schema 3, exact N=1, no_move"
        )
    actions = manifest.get("actions")
    if not isinstance(actions, list) or len(actions) != 1:
        raise LaunchRefused("N1 manifest must contain exactly one action row")
    action = actions[0]
    if type(action) is not dict:
        raise LaunchRefused("N1 manifest action row must be an object")
    if (
        action.get("action_id") != expected_action
        or action.get("action_uid") != bundle["action_uid"]
    ):
        raise LaunchRefused("N1 manifest action identity differs from bundle")
    _same_pin(
        {
            "path": action.get("motion_path"),
            "sha256": action.get("motion_sha256"),
        },
        motion_pin,
        name="manifest motion",
    )
    prototype_ref = manifest.get("prototype")
    if type(prototype_ref) is not dict:
        raise LaunchRefused("N1 manifest prototype pin is missing")
    _same_pin(prototype_ref, prototype_pin, name="manifest prototype")
    if prototype_ref.get("scope") != expected_scope:
        raise LaunchRefused("N1 manifest prototype scope differs from spec")
    holdout = manifest.get("holdout")
    if (
        type(holdout) is not dict
        or holdout.get("samples_per_action") != 768
    ):
        raise LaunchRefused(
            "N1 manifest holdout.samples_per_action must be exactly 768"
        )
    objective = manifest.get("counter_rally_objective")
    if type(objective) is not dict or objective.get("mode") != "counter_rally_v1":
        raise LaunchRefused("N1 manifest must carry counter_rally_v1 objective")
    if (
        manifest.get("solver_profile_sha256")
        != profile_pin["solver_profile_sha256"]
        or manifest.get("physics_profile_sha256")
        != profile_pin["physics_profile_sha256"]
    ):
        raise LaunchRefused(
            "N1 manifest solver/physics pins differ from bundle profile_pins"
        )
    for field in (
        "solver_profile_sha256",
        "physics_profile_sha256",
        "geometry_payload_sha256",
    ):
        _sha256(profile_pin[field], name=f"N1 profile_pins.{field}")

    contact_pin, contact = _load_tracked_json(
        checkout,
        commit_sha,
        bundle["contact_alignment"],
        name="N1 bundle contact alignment",
        extra_keys=("schema_version", "status"),
    )
    if contact_pin["schema_version"] != 1 or contact_pin["status"] != "PASS":
        raise LaunchRefused("N1 contact pin must be schema 1 PASS")
    contact_summary = _validate_contact_receipt(
        contact, bundle=bundle, manifest=manifest
    )
    return {
        "bundle": normalized_bundle_pin,
        "action_id": expected_action,
        "action_uid": bundle["action_uid"],
        "scope": expected_scope,
        "source_manifest": source_manifest_pin,
        "motion": motion_pin,
        "profile_pins": profile_pin,
        "geometry": geometry_pin,
        "prototype": prototype_pin,
        "manifest": manifest_pin,
        "contact_alignment": contact_pin,
        "contact_summary": contact_summary,
        **(
            {}
            if dynamic_ready is None
            else {"dynamic_ready": dynamic_ready}
        ),
        **(
            {}
            if full_solver_preflight is None
            else {
                "full_solver_admission_preflight": full_solver_preflight
            }
        ),
    }


def _validate_gpu(value: Any) -> dict[str, Any]:
    row = _exact_dict(value, _GPU_KEYS, name="spec.gpu")
    index = _plain_int(row["index"], name="spec.gpu.index", maximum=31)
    uuid = row["uuid"]
    if (
        type(uuid) is not str
        or not uuid.startswith("GPU-")
        or len(uuid) < 8
        or "," in uuid
        or "\n" in uuid
    ):
        raise LaunchRefused("spec.gpu.uuid must be an explicit GPU UUID")
    owner = row["owner"]
    if (
        type(owner) is not str
        or owner != owner.strip()
        or not owner
        or owner.lower()
        in {"codex", "claude", "fable", "agent", "unassigned"}
    ):
        raise LaunchRefused("spec.gpu.owner must be an explicit human name")
    lock_path = _absolute_path(
        row["lock_path"], name="spec.gpu.lock_path"
    )
    expected_lock = Path(f"/tmp/hope_lean_queue_gpu{index}.lock")
    if lock_path != expected_lock:
        raise LaunchRefused(
            f"spec.gpu.lock_path must be {expected_lock}"
        )
    if row["require_empty"] is not True:
        raise LaunchRefused("spec.gpu.require_empty must be true")
    return {
        "index": index,
        "uuid": uuid,
        "owner": owner,
        "lock_path": str(lock_path),
        "require_empty": True,
    }


def _validate_budget(
    stage: Any, num_envs: Any, max_iterations: Any, save_interval: Any
) -> dict[str, Any]:
    if type(stage) is not str or stage not in ALLOWED_STAGES:
        raise LaunchRefused(
            "stage must be smoke, exact probe, canary, exact milestone1000, "
            "or exact long"
        )
    envs = _plain_int(num_envs, name="num_envs", minimum=1)
    iterations = _plain_int(
        max_iterations, name="max_iterations", minimum=1
    )
    save = _plain_int(save_interval, name="save_interval", minimum=1)
    if stage == "smoke":
        if (envs, iterations, save) != (1, 2, 1):
            raise LaunchRefused(
                "smoke is exactly 1 env / 2 updates / save interval 1"
            )
    elif stage == "probe":
        if (envs, iterations, save) != (4096, 5, 1):
            raise LaunchRefused(
                "probe is exactly 4096 envs / 5 updates / save interval 1"
            )
    elif stage == "canary":
        if not (16 <= envs <= 1024):
            raise LaunchRefused("canary num_envs must be in [16,1024]")
        if not (10 <= iterations <= 2000):
            raise LaunchRefused(
                "canary max_iterations must be in [10,2000]"
            )
        if save > iterations or save > 200:
            raise LaunchRefused(
                "canary save_interval must be <= iterations and <= 200"
            )
    elif stage == "milestone1000":
        if (envs, iterations, save) != (4096, 1001, 100):
            raise LaunchRefused(
                "milestone1000 is exactly 4096 envs / 1001 updates / "
                "save interval 100"
            )
    elif (envs, iterations, save) != (4096, 20_001, 100):
        raise LaunchRefused(
            "long is exactly 4096 envs / 20001 updates / save interval 100"
        )
    return {
        "stage": stage,
        "num_envs": envs,
        "max_iterations": iterations,
        "save_interval": save,
    }


def _build_training_argv(
    spec: dict[str, Any], bundle: dict[str, Any]
) -> list[str]:
    checkout = Path(spec["source"]["checkout"])
    wbt = checkout / WBT_RELATIVE
    motion = checkout / bundle["motion"]["path"]
    manifest = checkout / bundle["manifest"]["path"]
    dynamic_ready = bundle.get("dynamic_ready")
    if type(dynamic_ready) is not dict:
        raise LaunchRefused(
            "dynamic-ready training argv requires one validated bundle-v2 pin"
        )
    artifact = checkout / dynamic_ready["artifact"]["path"]
    nominal_receipt = (
        checkout / dynamic_ready["nominal_hold_receipt"]["path"]
    )
    weights = REWARD_PROFILES[spec["reward_profile"]]
    json_list = lambda values: json.dumps(  # noqa: E731
        values, separators=(",", ":"), ensure_ascii=False
    )
    return [
        spec["source"]["isaac_python"],
        str(wbt / "scripts/train.py"),
        "task=HOPEPingPongActionBall",
        "algo=ppo",
        "algo.policy.init_noise_std=0.02",
        "action_ball_dynamic_ready_bootstrap=true",
        f"action_ball_dynamic_ready_artifact_path={artifact}",
        (
            "action_ball_dynamic_ready_artifact_sha256="
            f"{dynamic_ready['artifact']['sha256']}"
        ),
        (
            "action_ball_dynamic_ready_nominal_receipt_path="
            f"{nominal_receipt}"
        ),
        (
            "action_ball_dynamic_ready_nominal_receipt_sha256="
            f"{dynamic_ready['nominal_hold_receipt']['sha256']}"
        ),
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        f"seed={spec['seed']}",
        f"num_envs={spec['num_envs']}",
        f"max_iterations={spec['max_iterations']}",
        f"algo.runner.save_interval={spec['save_interval']}",
        f"run_name={Path(spec['namespace']).name}",
        f"task.experiment_name={EXPERIMENT_NAME}",
        (
            "expected_effective_reward_recipe_sha256="
            f"{spec['expected_effective_reward_recipe_sha256']}"
        ),
        "task.actor_obs_contract="
        "action_ball_table_pose_twist_heading_task_teacher_start_v2",
        (
            "task.rewards.full_body_mimic="
            f"{'true' if spec['scope'] == 'full' else 'false'}"
        ),
        f"+task.rewards.motion_scale={weights['motion_scale']}",
        (
            "task.rewards.racket_position_weight="
            f"{weights['racket_position_weight']}"
        ),
        (
            "task.rewards.racket_velocity_weight="
            f"{weights['racket_velocity_weight']}"
        ),
        (
            "task.rewards.racket_normal_weight="
            f"{weights['racket_normal_weight']}"
        ),
        f"motion_file={json_list([str(motion)])}",
        f"task.racket.clip_names={json_list([spec['action_id']])}",
        "task.racket.target_mode=action_ball",
        f"task.racket.action_ball_manifest_path={manifest}",
        (
            "task.racket.action_ball_manifest_sha256="
            f"{bundle['manifest']['sha256']}"
        ),
        (
            "task.racket.action_ball_policy_contract_sha256="
            f"{spec['policy_contract_sha256']}"
        ),
        "task.racket.action_ball_diagnostic_unauthorized=true",
        # The first policy keeps the historical material/default-offset axes
        # but removes the shared waist-equilibrium CoM/mass/PD perturbations.
        # Those axes return only after a robust-hold certificate covers them.
        "+task.domain_rand.stable_ready_plant=true",
        "+task.racket.reference_guard_mode=metrics_only",
        f"task.racket.action_ball_seed={spec['seed']}",
        "task.racket.question_bank=",
        "task.racket.question_bank_allow_legacy=false",
        "task.racket.cq_anchor_bank=",
        "task.racket.exam_bank=",
    ]


def _validate_spec_document(
    document: dict[str, Any], *, namespace_claimed: bool = False
) -> dict[str, Any]:
    row = _exact_dict(document, _SPEC_KEYS, name="launch spec")
    if row["schema_version"] != SCHEMA_VERSION or row["kind"] != SPEC_KIND:
        raise LaunchRefused(
            f"launch spec must be schema {SCHEMA_VERSION} / {SPEC_KIND!r}"
        )
    source = _exact_dict(row["source"], _SOURCE_KEYS, name="spec.source")
    checkout = _absolute_path(
        source["checkout"], name="spec.source.checkout", must_exist=True
    )
    commit = source["commit_sha"]
    if type(commit) is not str or COMMIT_RE.fullmatch(commit) is None:
        raise LaunchRefused("spec.source.commit_sha must be 40 lowercase hex")
    isaac_python = _absolute_path(
        source["isaac_python"],
        name="spec.source.isaac_python",
        must_exist=True,
    )
    try:
        python_info = isaac_python.stat()
    except OSError as exc:
        raise LaunchRefused(f"cannot stat Isaac Python: {exc}") from exc
    if not stat.S_ISREG(python_info.st_mode) or not os.access(
        isaac_python, os.X_OK
    ):
        raise LaunchRefused("spec.source.isaac_python must be executable")
    action = row["action_id"]
    if type(action) is not str or action not in ALLOWED_ACTIONS:
        raise LaunchRefused("action_id must be bh_loop_c or bh_block")
    scope = row["scope"]
    if type(scope) is not str or scope not in ALLOWED_SCOPES:
        raise LaunchRefused("scope must be upper or full")
    bundle_pin = _exact_dict(row["bundle"], _PIN_KEYS, name="spec.bundle")
    policy_sha = _sha256(
        row["policy_contract_sha256"], name="policy_contract_sha256"
    )
    profile = row["reward_profile"]
    if type(profile) is not str or profile not in REWARD_PROFILES:
        raise LaunchRefused(
            f"reward_profile must be one of {sorted(REWARD_PROFILES)!r}"
        )
    reward_sha = _sha256(
        row["expected_effective_reward_recipe_sha256"],
        name="expected_effective_reward_recipe_sha256",
    )
    seed = _plain_int(row["seed"], name="seed", maximum=(1 << 31) - 1)
    budget = _validate_budget(
        row["stage"],
        row["num_envs"],
        row["max_iterations"],
        row["save_interval"],
    )
    gpu = _validate_gpu(row["gpu"])
    namespace = _absolute_path(row["namespace"], name="namespace")
    if (
        namespace.name in ("", ".", "..")
        or SAFE_COMPONENT_RE.fullmatch(namespace.name) is None
    ):
        raise LaunchRefused("namespace basename is not a safe run component")
    log_path = _absolute_path(row["log_path"], name="log_path")
    if log_path != namespace / "run.log":
        raise LaunchRefused("log_path must be exactly <namespace>/run.log")
    if os.path.lexists(namespace):
        if not namespace_claimed:
            raise LaunchRefused(
                f"run namespace already exists and is permanently spent: {namespace}"
            )
        try:
            info = namespace.lstat()
        except OSError as exc:
            raise LaunchRefused(
                f"claimed namespace cannot be inspected: {exc}"
            ) from exc
        if (
            not stat.S_ISDIR(info.st_mode)
            or namespace.resolve(strict=True) != namespace
        ):
            raise LaunchRefused(
                "claimed namespace must remain a real directory"
            )
    elif namespace_claimed:
        raise LaunchRefused("claimed namespace vanished before trainer exec")
    parent = namespace.parent
    if not parent.exists() or parent.resolve(strict=True) != parent:
        raise LaunchRefused(
            "namespace parent must be an existing real absolute directory"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": commit,
            "isaac_python": str(isaac_python),
        },
        "action_id": action,
        "scope": scope,
        "bundle": dict(bundle_pin),
        "policy_contract_sha256": policy_sha,
        "reward_profile": profile,
        "expected_effective_reward_recipe_sha256": reward_sha,
        "seed": seed,
        **budget,
        "gpu": gpu,
        "namespace": str(namespace),
        "log_path": str(log_path),
    }


def _validate_runtime_sources(
    checkout: Path, commit_sha: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, label in (
        (LAUNCHER_SOURCE, "N1 diagnostic launcher"),
        (TRAIN_SOURCE, "training entrypoint"),
        (TASK_SOURCE, "ActionBall task config"),
        (KIT_LAUNCHER_SOURCE, "Kit locked launcher"),
    ):
        pin = {"path": relative, "sha256": sha256_file(checkout / relative)}
        normalized, _path = _verify_tracked_file(
            checkout, commit_sha, pin, name=label
        )
        result[label] = normalized
    actual_launcher = Path(__file__).resolve()
    expected_launcher = checkout / LAUNCHER_SOURCE
    if actual_launcher != expected_launcher:
        raise LaunchRefused(
            "running launcher is not the exact selected checkout path"
        )
    return result


def _validate_runtime_asset_paths(
    usd_path: Path, glu_directory: Path
) -> dict[str, Any]:
    """Pin the ignored A3 USD closure and private GLU before Kit starts."""

    if (
        not usd_path.is_absolute()
        or usd_path.name != "model.usd"
        or usd_path.resolve(strict=True) != usd_path
    ):
        raise LaunchRefused(
            "HOPE_AGIBOT_A3_USD_PATH must be one real absolute model.usd"
        )
    bundle_root = usd_path.parent
    if bundle_root.resolve(strict=True) != bundle_root or not bundle_root.is_dir():
        raise LaunchRefused("A3 preconverted USD root must be a real directory")
    files: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in A3_RUNTIME_USD_BUNDLE_SHA256.items():
        path = bundle_root / relative
        info = _stable_regular_file(path, name=f"A3 runtime USD {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise LaunchRefused(
                f"A3 runtime USD {relative} SHA differs: "
                f"expected={expected_sha}, actual={actual_sha}"
            )
        files[relative] = {
            "sha256": actual_sha,
            "size_bytes": int(info.st_size),
        }
    if (
        not glu_directory.is_absolute()
        or glu_directory.resolve(strict=True) != glu_directory
        or not glu_directory.is_dir()
    ):
        raise LaunchRefused("private GLU root must be one real absolute directory")
    library = glu_directory / PRIVATE_GLU_LIBRARY
    info = _stable_regular_file(library, name="private GLU library")
    actual_glu_sha = sha256_file(library)
    if actual_glu_sha != PRIVATE_GLU_SHA256:
        raise LaunchRefused(
            "private GLU library SHA differs: "
            f"expected={PRIVATE_GLU_SHA256}, actual={actual_glu_sha}"
        )
    soname = glu_directory / PRIVATE_GLU_SONAME
    try:
        soname_info = soname.lstat()
        soname_target = os.readlink(soname)
    except OSError as exc:
        raise LaunchRefused(f"private GLU soname cannot be inspected: {exc}") from exc
    if (
        not stat.S_ISLNK(soname_info.st_mode)
        or soname_target != PRIVATE_GLU_LIBRARY
        or soname.resolve(strict=True) != library
    ):
        raise LaunchRefused(
            "private GLU soname must point directly to the pinned library"
        )
    return {
        "schema_version": 1,
        "kind": "n1_a3_runtime_asset_pins_v1",
        "urdf_importer_no_ui": "1",
        "a3_preconverted_usd": {
            "path": str(usd_path),
            "bundle_root": str(bundle_root),
            "files": files,
        },
        "private_glu": {
            "directory": str(glu_directory),
            "library": str(library),
            "sha256": actual_glu_sha,
            "size_bytes": int(info.st_size),
            "soname": str(soname),
            "soname_target": soname_target,
        },
    }


def _validate_runtime_asset_environment() -> dict[str, Any]:
    """Resolve the reviewed external assets from the plan-time environment."""

    if os.environ.get("HOPE_URDF_IMPORTER_NO_UI") != "1":
        raise LaunchRefused("HOPE_URDF_IMPORTER_NO_UI must equal 1")
    usd_path = _absolute_path(
        os.environ.get("HOPE_AGIBOT_A3_USD_PATH"),
        name="HOPE_AGIBOT_A3_USD_PATH",
        must_exist=True,
    )
    library_path = os.environ.get("LD_LIBRARY_PATH")
    if (
        type(library_path) is not str
        or not library_path
        or "\x00" in library_path
    ):
        raise LaunchRefused(
            "LD_LIBRARY_PATH must begin with the reviewed private GLU root"
        )
    first = library_path.split(os.pathsep, 1)[0]
    glu_directory = _absolute_path(
        first, name="private GLU LD_LIBRARY_PATH entry", must_exist=True
    )
    return _validate_runtime_asset_paths(usd_path, glu_directory)


def _validate_runtime_asset_claim(value: Any) -> dict[str, Any]:
    """Re-hash the exact paths sealed into a launch claim."""

    if type(value) is not dict:
        raise LaunchRefused("runtime asset pins are missing from launch claim")
    usd = value.get("a3_preconverted_usd")
    glu = value.get("private_glu")
    if type(usd) is not dict or type(glu) is not dict:
        raise LaunchRefused("runtime asset claim is malformed")
    observed = _validate_runtime_asset_paths(
        _absolute_path(
            usd.get("path"),
            name="claimed A3 preconverted USD",
            must_exist=True,
        ),
        _absolute_path(
            glu.get("directory"),
            name="claimed private GLU root",
            must_exist=True,
        ),
    )
    if observed != value:
        raise LaunchRefused("runtime asset pins drifted after plan")
    return observed


def _check_rsl_namespace_available(
    checkout: Path, namespace_name: str
) -> None:
    root = (
        checkout
        / WBT_RELATIVE
        / "logs/rsl_rl"
        / EXPERIMENT_NAME
    )
    if not os.path.lexists(root):
        return
    if root.resolve(strict=True) != root or not root.is_dir():
        raise LaunchRefused("RSL experiment root is not a real directory")
    suffix = f"_{namespace_name}-DIAGNOSTIC_UNAUTHORIZED"
    spent = sorted(
        child.name for child in root.iterdir() if child.name.endswith(suffix)
    )
    if spent:
        raise LaunchRefused(
            f"trainer run_name is already spent: {spent[0]}"
        )


def build_plan(spec_path: Path) -> dict[str, Any]:
    spec_path = _absolute_path(
        str(spec_path), name="--spec", must_exist=True
    )
    _stable_regular_file(spec_path, name="launch spec")
    raw = spec_path.read_bytes()
    document = _strict_json_bytes(raw, name="launch spec")
    if raw != _canonical_bytes(document) + b"\n":
        raise LaunchRefused("launch spec must be canonical JSON plus newline")
    spec = _validate_spec_document(document)
    checkout = Path(spec["source"]["checkout"])
    commit = spec["source"]["commit_sha"]
    source = _verify_clean_source(checkout, commit)
    runtime_sources = _validate_runtime_sources(checkout, commit)
    runtime_assets = _validate_runtime_asset_environment()
    bundle = _validate_bundle(
        checkout,
        commit,
        spec["bundle"],
        expected_action=spec["action_id"],
        expected_scope=spec["scope"],
        require_dynamic_ready=True,
    )
    _check_rsl_namespace_available(checkout, Path(spec["namespace"]).name)
    argv = _build_training_argv(spec, bundle)
    claim_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "diagnostic_unauthorized": True,
        "formal_evidence_prohibited": True,
        "curriculum_promotion_prohibited": True,
        "long_stage_prohibited": spec["stage"] != "long",
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
        "source": source,
        "runtime_sources": runtime_sources,
        "runtime_assets": runtime_assets,
        "bundle": bundle,
        "reward_weights": dict(REWARD_PROFILES[spec["reward_profile"]]),
        "training_argv": argv,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLAIM_KIND,
        "launch_claim_sha256": canonical_sha256(claim_payload),
        "canonical_payload": claim_payload,
    }


def _open_gpu_lock(lock_path: Path) -> int:
    try:
        before = lock_path.lstat()
    except OSError as exc:
        raise LaunchRefused(
            f"GPU lifetime lock must already exist: {lock_path}: {exc}"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise LaunchRefused("GPU lifetime lock must be a regular file")
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise LaunchRefused(f"cannot open GPU lifetime lock: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        after = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (after.st_dev, after.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise LaunchRefused("GPU lock pathname identity changed")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LaunchRefused("GPU lifetime lock is already owned") from exc
        os.set_inheritable(descriptor, True)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _trusted_nvidia_smi() -> tuple[str, str]:
    requested = shutil.which(
        "nvidia-smi", path="/usr/bin:/bin:/usr/local/bin"
    )
    if requested is None:
        raise LaunchRefused("nvidia-smi is unavailable")
    path = Path(requested).resolve(strict=True)
    _stable_regular_file(path, name="nvidia-smi")
    if not os.access(path, os.X_OK):
        raise LaunchRefused("nvidia-smi is not executable")
    return str(path), sha256_file(path)


def _verify_gpu_empty(index: int, uuid: str) -> dict[str, Any]:
    nvidia_smi, binary_sha = _trusted_nvidia_smi()
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "LANG": "C", "LC_ALL": "C"}
    identity = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if identity.returncode != 0:
        raise LaunchRefused(
            f"nvidia-smi identity query failed: {identity.stderr.strip()}"
        )
    observed: dict[int, str] = {}
    for line in identity.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            raise LaunchRefused(f"unparseable GPU identity row: {line!r}")
        observed[int(parts[0])] = parts[1]
    if observed.get(index) != uuid:
        raise LaunchRefused(
            f"GPU {index} UUID differs: expected={uuid}, "
            f"actual={observed.get(index)!r}"
        )
    occupancy = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if occupancy.returncode != 0:
        raise LaunchRefused(
            f"nvidia-smi compute query failed: {occupancy.stderr.strip()}"
        )
    for line in occupancy.stdout.splitlines():
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",", 2)]
        if (
            len(parts) != 3
            or not parts[0].startswith("GPU-")
            or not parts[1].isdigit()
        ):
            raise LaunchRefused(f"unparseable GPU compute row: {line!r}")
        if parts[0] == uuid:
            raise LaunchRefused(
                f"GPU {index} is occupied by pid={parts[1]} "
                f"process={parts[2]!r}"
            )
    return {
        "index": index,
        "uuid": uuid,
        "compute_process_count": 0,
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_sha256": binary_sha,
    }


def _write_exclusive_json(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise LaunchRefused(f"no-clobber write failed for {path}: {exc}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(value))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _claim_namespace(plan: dict[str, Any]) -> Path:
    namespace = Path(plan["canonical_payload"]["spec"]["namespace"])
    try:
        os.mkdir(namespace, 0o700)
    except FileExistsError as exc:
        raise LaunchRefused(
            f"run namespace already exists and is permanently spent: {namespace}"
        ) from exc
    except OSError as exc:
        raise LaunchRefused(f"cannot claim namespace {namespace}: {exc}") from exc
    _write_exclusive_json(namespace / "launch_claim.json", plan)
    return namespace


def _internal_exec(claim_path: Path, expected_sha: str, lock_fd: int) -> int:
    """Revalidate an immutable claim, recheck the GPU, then exec the trainer."""

    claim_path = _absolute_path(
        str(claim_path), name="internal claim", must_exist=True
    )
    _stable_regular_file(claim_path, name="internal launch claim")
    raw = claim_path.read_bytes()
    plan = _strict_json_bytes(raw, name="internal launch claim")
    if raw != _canonical_bytes(plan) + b"\n":
        raise LaunchRefused("internal launch claim is not canonical")
    outer = _exact_dict(
        plan,
        (
            "schema_version",
            "kind",
            "launch_claim_sha256",
            "canonical_payload",
        ),
        name="internal launch claim",
    )
    if (
        outer["schema_version"] != SCHEMA_VERSION
        or outer["kind"] != CLAIM_KIND
        or outer["launch_claim_sha256"] != expected_sha
        or canonical_sha256(outer["canonical_payload"]) != expected_sha
    ):
        raise LaunchRefused("internal launch claim digest differs")
    payload = outer["canonical_payload"]
    if type(payload) is not dict or payload.get("kind") != CLAIM_KIND:
        raise LaunchRefused("internal launch payload kind differs")
    spec = _validate_spec_document(
        payload["spec"], namespace_claimed=True
    )
    checkout = Path(spec["source"]["checkout"])
    _verify_clean_source(checkout, spec["source"]["commit_sha"])
    runtime = _validate_runtime_sources(checkout, spec["source"]["commit_sha"])
    if runtime != payload["runtime_sources"]:
        raise LaunchRefused("runtime source identity drifted after namespace claim")
    runtime_assets = _validate_runtime_asset_claim(payload.get("runtime_assets"))
    bundle = _validate_bundle(
        checkout,
        spec["source"]["commit_sha"],
        spec["bundle"],
        expected_action=spec["action_id"],
        expected_scope=spec["scope"],
        require_dynamic_ready=True,
    )
    if bundle != payload["bundle"]:
        raise LaunchRefused("N1 bundle identity drifted after namespace claim")
    argv = _build_training_argv(spec, bundle)
    if argv != payload["training_argv"]:
        raise LaunchRefused("training argv differs from immutable claim")
    lock_path = Path(spec["gpu"]["lock_path"])
    try:
        lock_info = os.fstat(lock_fd)
        path_info = lock_path.lstat()
    except OSError as exc:
        raise LaunchRefused(f"inherited GPU lock cannot be verified: {exc}") from exc
    if (
        not stat.S_ISREG(lock_info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or (lock_info.st_dev, lock_info.st_ino)
        != (path_info.st_dev, path_info.st_ino)
    ):
        raise LaunchRefused("inherited GPU lock differs from shared lock path")
    try:
        # Re-locking the inherited open-file description is a no-op when the
        # parent already owns it and safely acquires it if an intermediate
        # exec unexpectedly dropped the lock state.  A conflicting owner
        # therefore remains fail-closed.
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise LaunchRefused(
            "inherited GPU lock is not owned by this launch"
        ) from exc
    second_gpu_check = _verify_gpu_empty(
        spec["gpu"]["index"], spec["gpu"]["uuid"]
    )
    _write_exclusive_json(
        Path(spec["namespace"]) / "pre_exec_gpu_admission.json",
        {
            "schema_version": 1,
            "kind": "n1_reward_screen_pre_exec_gpu_admission_v1",
            "launch_claim_sha256": expected_sha,
            "gpu": second_gpu_check,
        },
    )
    wbt = checkout / WBT_RELATIVE
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(wbt / "source/whole_body_tracking"),
        "CUDA_VISIBLE_DEVICES": str(spec["gpu"]["index"]),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "offline",
        "HOPE_URDF_IMPORTER_NO_UI": runtime_assets[
            "urdf_importer_no_ui"
        ],
        "HOPE_AGIBOT_A3_USD_PATH": runtime_assets[
            "a3_preconverted_usd"
        ]["path"],
        "LD_LIBRARY_PATH": runtime_assets["private_glu"]["directory"],
    }
    os.chdir(wbt)
    os.execve(argv[0], argv, environment)
    raise AssertionError("os.execve returned")


def launch(plan: dict[str, Any], *, confirm_claim: str) -> dict[str, Any]:
    expected = _sha256(confirm_claim, name="--confirm-claim")
    if expected != plan["launch_claim_sha256"]:
        raise LaunchRefused(
            "--confirm-claim differs from the freshly recomputed plan"
        )
    spec = plan["canonical_payload"]["spec"]
    checkout = Path(spec["source"]["checkout"])
    # Repeat clean-source validation immediately before taking mutable state.
    _verify_clean_source(checkout, spec["source"]["commit_sha"])
    _validate_runtime_asset_claim(
        plan["canonical_payload"].get("runtime_assets")
    )
    lock_fd = _open_gpu_lock(Path(spec["gpu"]["lock_path"]))
    namespace: Path | None = None
    try:
        first_gpu_check = _verify_gpu_empty(
            spec["gpu"]["index"], spec["gpu"]["uuid"]
        )
        namespace = _claim_namespace(plan)
        _write_exclusive_json(
            namespace / "pre_launch_gpu_admission.json",
            {
                "schema_version": 1,
                "kind": "n1_reward_screen_pre_launch_gpu_admission_v1",
                "launch_claim_sha256": expected,
                "gpu": first_gpu_check,
            },
        )
        launcher = checkout / KIT_LAUNCHER_SOURCE
        state_path = Path(spec["log_path"] + ".launch")
        internal_command = [
            str(Path(sys.executable).resolve()),
            str(checkout / LAUNCHER_SOURCE),
            "_exec",
            "--claim",
            str(namespace / "launch_claim.json"),
            "--claim-sha256",
            expected,
            "--gpu-lock-fd",
            str(lock_fd),
        ]
        environment = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C",
            "LC_ALL": "C",
            "KIT_BOOT_MARKER": "Learning iteration",
            "KIT_BOOT_TIMEOUT_S": "1800",
            "KIT_BOOT_STALE_TIMEOUT_S": "900",
            "KIT_BOOT_POLL_S": "5",
            "KIT_BOOT_STATE_FILE": str(state_path),
        }
        result = subprocess.run(
            [str(launcher), spec["log_path"], *internal_command],
            cwd=checkout / WBT_RELATIVE,
            env=environment,
            pass_fds=(lock_fd,),
            check=False,
        )
        if result.returncode != 0:
            raise LaunchRefused(
                f"locked Kit launcher returned {result.returncode}; "
                f"namespace remains spent at {namespace}"
            )
        return {
            "schema_version": 1,
            "kind": "n1_reward_screen_diagnostic_launch_result_v1",
            "launch_claim_sha256": expected,
            "namespace": str(namespace),
            "log_path": spec["log_path"],
            "state_path": str(state_path),
            "gpu": spec["gpu"],
            "diagnostic_unauthorized": True,
            "accepted": True,
        }
    finally:
        # The trainer inherited this exact open-file description through the
        # locked Kit launcher.  Closing our copy leaves its lifetime copy held.
        os.close(lock_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "launch"):
        child = sub.add_parser(command)
        child.add_argument("--spec", required=True)
        if command == "launch":
            child.add_argument("--confirm-claim", required=True)
    internal = sub.add_parser("_exec", help=argparse.SUPPRESS)
    internal.add_argument("--claim", required=True)
    internal.add_argument("--claim-sha256", required=True)
    internal.add_argument("--gpu-lock-fd", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "_exec":
            return _internal_exec(
                Path(args.claim), args.claim_sha256, args.gpu_lock_fd
            )
        plan = build_plan(Path(args.spec))
        if args.command == "plan":
            print(
                json.dumps(
                    plan,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        result = launch(plan, confirm_claim=args.confirm_claim)
        print(
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0
    except LaunchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
