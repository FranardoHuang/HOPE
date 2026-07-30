#!/usr/bin/env python3
"""Materialize one content-addressed N=1 fast-ball diagnostic bundle.

This is a deliberately narrow derivative materializer.  It starts from one
already sealed N=1 contact bundle, changes only the incoming-speed and landing
aim support in its strict manifest, and republishes that manifest plus a new
bundle pin.  Motion, action identity, prototype, fixed-action solver, physics,
geometry, contact alignment, dynamic-ready, and teacher-rate bounds remain
byte-for-byte/pin-for-pin unchanged.

The defaults encode the 2026-07-30 ``bh_block`` 1.1x incoming-speed comparison:
4.661464290649453 m/s incoming centre, a 0.4x hard floor, a 7.0 m/s hard
ceiling, the source 0.15 m/s initial one-sided widths, and the unchanged
landing aim [2.555, 0.0] m.  This comparison is diagnostic only: its fixed
4096-proposal tape admitted 2763/4096 (67.4560546875%), below the formal 95%
admission threshold.  Solver rejection conditions the accepted population, so
the output must not be described as an unbiased A/B comparison.
All numeric parameters are locked to that exact tape; changing one requires a
new tape and a new producer.

Outputs are canonical UTF-8 JSON followed by exactly one newline.  Their
filenames include the first twelve hexadecimal digits of the complete raw-file
SHA-256.  Publication is no-clobber and is followed by an exact byte/hash/JSON
round trip.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = (
    SCRIPT_DIR.parents[2]
    if len(SCRIPT_DIR.parents) > 2
    else Path.cwd()
)
SOURCE_BUNDLE_RELATIVE_PATH = Path(
    "configs/n1_contact_dynamic_ready_20260730_r9/"
    "bh_block.bundle.v2.3267a3f6d303.json"
)
SOURCE_BUNDLE_SHA256 = (
    "3267a3f6d303e415180d4d1df49a84a7b14ec2899026e4c67aa69f88e7cbe2eb"
)
OUTPUT_DIR_RELATIVE_PATH = Path("configs/n1_contact_fastball_20260730")

ACTION_ID = "bh_block"
ACTION_UID = 3707627670665312
SOURCE_SPEED_CENTER_MPS = 4.2376948096813205
DEFAULT_SPEED_MULTIPLIER = 1.1
DEFAULT_SPEED_CENTER_MPS = 4.661464290649453
DEFAULT_SPEED_FLOOR_RATIO = 0.4
DEFAULT_SPEED_MAX_MPS = 7.0
DEFAULT_SPEED_INITIAL_WIDTH_MPS = 0.15
DEFAULT_LANDING_AIM_X_M = 2.555
DEFAULT_LANDING_AIM_Y_M = 0.0
DEFAULT_LANDING_INITIAL_WIDTH_M = 0.01
EXPECTED_TEACHER_RATE_MIN = 0.6
EXPECTED_TEACHER_RATE_MAX = 1.0

TAPE_PROPOSALS = 4096
TAPE_GEOMETRY_SOLVED_UNRESTRICTED = 2769
TAPE_ADMITTED = 2763
TAPE_SHA256 = (
    "0335220da643ad3c6c21d72177af9ccb617b10415eb9cb48035d143d29c82581"
)
TAPE_REJECTION_REASONS = {
    "resid_gt_tol": 1327,
    "teacher_rate_below_min": 6,
}
TAPE_TEACHER_RATE_MEAN = 0.7205540609680153
TAPE_TEACHER_RATE_P01 = 0.6250536329770616
TAPE_TEACHER_RATE_P50 = 0.7159498953132029
TAPE_TEACHER_RATE_P99 = 0.8361400507908392
TAPE_SITE_SPEED_MEAN_MPS = 1.149070673485166

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_ID = (
    "action_ball_n1_bh_block_upper_contact_counter_rally_"
    "fastball_1p1x_teacher_rate_bootstrap_diagnostic_v1"
)
HOLDOUT_SPLIT_ID = (
    "heldout_ball_bh_block_counter_rally_"
    "fastball_1p1x_diagnostic_n1_v1"
)


class FastBallMaterializationError(ValueError):
    """The source bundle or requested derivative is not exact and admissible."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
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


def _strict_object(
    pairs: Sequence[Tuple[str, object]]
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FastBallMaterializationError(
                "duplicate JSON key {!r}".format(key)
            )
        result[key] = value
    return result


def _load_json(
    path: Path, *, expected_sha256: Optional[str] = None
) -> Any:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FastBallMaterializationError(
            "{} SHA-256 mismatch: expected {}, observed {}".format(
                path, expected_sha256, digest
            )
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FastBallMaterializationError(
                    "non-finite JSON token {!r} in {}".format(token, path)
                )
            ),
        )
    except UnicodeDecodeError as error:
        raise FastBallMaterializationError(
            "{} is not UTF-8 JSON".format(path)
        ) from error
    return value


def _require_object(value: Any, *, label: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise FastBallMaterializationError(
            "{} must be one JSON object".format(label)
        )
    return value


def _require_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FastBallMaterializationError(
            "{} must be a finite number".format(label)
        )
    result = float(value)
    if not math.isfinite(result):
        raise FastBallMaterializationError(
            "{} must be finite".format(label)
        )
    return result


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise FastBallMaterializationError(
            "{} must be lowercase SHA-256".format(label)
        )
    return value


def _repo_relative(path: Path, repo_root: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        return PurePosixPath(resolved.relative_to(repo_root)).as_posix()
    except ValueError as error:
        raise FastBallMaterializationError(
            "{} must be inside repo root".format(label)
        ) from error


def _resolve_pin_path(
    repo_root: Path,
    pin: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    if type(pin) is not dict:
        raise FastBallMaterializationError(
            "{} pin must be an object".format(label)
        )
    path_value = pin.get("path")
    if type(path_value) is not str or not path_value:
        raise FastBallMaterializationError(
            "{} pin path must be non-empty".format(label)
        )
    expected_sha = _require_sha256(
        pin.get("sha256"), label="{}.sha256".format(label)
    )
    path = (repo_root / path_value).resolve()
    _repo_relative(path, repo_root, label=label)
    if not path.is_file():
        raise FastBallMaterializationError(
            "{} pinned file does not exist: {}".format(label, path)
        )
    observed_sha = _sha256_bytes(path.read_bytes())
    if observed_sha != expected_sha:
        raise FastBallMaterializationError(
            "{} pinned SHA mismatch: expected {}, observed {}".format(
                label, expected_sha, observed_sha
            )
        )
    return path


def _require_pair(value: Any, *, label: str) -> List[float]:
    if type(value) is not list or len(value) != 2:
        raise FastBallMaterializationError(
            "{} must contain exactly two numbers".format(label)
        )
    return [
        _require_finite(item, label="{}[{}]".format(label, index))
        for index, item in enumerate(value)
    ]


def _extend_bounds_for_initial_width(
    *,
    center: Sequence[float],
    lower_bound: Sequence[float],
    upper_bound: Sequence[float],
    initial_lower: Sequence[float],
    initial_upper: Sequence[float],
) -> Tuple[List[float], List[float]]:
    """Keep old hard bounds unless the requested initial support needs room."""

    new_min = []
    new_max = []
    for index in range(2):
        new_min.append(
            min(float(lower_bound[index]), center[index] - initial_lower[index])
        )
        new_max.append(
            max(float(upper_bound[index]), center[index] + initial_upper[index])
        )
    return new_min, new_max


def _clip_widths_to_room(
    *,
    old_widths: Sequence[float],
    center: Sequence[float],
    bounds: Sequence[float],
    lower_side: bool,
) -> List[float]:
    result = []
    for index in range(2):
        room = (
            center[index] - bounds[index]
            if lower_side
            else bounds[index] - center[index]
        )
        if room < 0.0:
            raise FastBallMaterializationError(
                "landing aim centre lies outside its hard bounds"
            )
        result.append(min(float(old_widths[index]), float(room)))
    return result


def _diagnostic_note(
    source_note: str,
    *,
    speed_center_mps: float,
    speed_floor_ratio: float,
    speed_max_mps: float,
    speed_initial_width_mps: float,
    landing_center: Sequence[float],
    landing_initial_width_m: float,
) -> str:
    rate = TAPE_ADMITTED / TAPE_PROPOSALS
    return (
        source_note.rstrip()
        + " Fast-ball teacher-rate bootstrap comparison (diagnostic only, "
        "not a formal manifest/admission claim): incoming speed centre "
        "{:.15g} m/s ({:.6g}x the source {:.15g} m/s centre), hard "
        "floor {:.6g}x centre, hard ceiling {:.6g} m/s, and independent "
        "initial lower/upper width {:.6g} m/s. Landing aim centre is "
        "[{:.15g}, {:.15g}] m with independent initial lower/upper width "
        "{:.6g} m. The exact fixed 4096-proposal diagnostic tape "
        "(SHA-256 {}) solved {} unrestricted geometry rows and admitted "
        "{}/{} ({:.6%}); unrestricted teacher-rate mean/p01/p50/p99 were "
        "{:.6g}/{:.6g}/{:.6g}/{:.6g}; rejections were {}. This is below "
        "formal 95% admission threshold and therefore supports only a "
        "diagnostic comparison. The action identity, motion, prototype, "
        "fixed-action solver, physics, geometry, dynamic-ready, contact "
        "alignment, teacher-rate interval [0.6, 1.0], landing aim, and "
        "all initial widths are unchanged. Mean solved site speed was "
        "{:.15g} m/s. Solver rejection conditions the accepted population; "
        "this is not an unbiased A/B comparison."
    ).format(
        speed_center_mps,
        speed_center_mps / SOURCE_SPEED_CENTER_MPS,
        SOURCE_SPEED_CENTER_MPS,
        speed_floor_ratio,
        speed_max_mps,
        speed_initial_width_mps,
        landing_center[0],
        landing_center[1],
        landing_initial_width_m,
        TAPE_SHA256,
        TAPE_GEOMETRY_SOLVED_UNRESTRICTED,
        TAPE_ADMITTED,
        TAPE_PROPOSALS,
        rate,
        TAPE_TEACHER_RATE_MEAN,
        TAPE_TEACHER_RATE_P01,
        TAPE_TEACHER_RATE_P50,
        TAPE_TEACHER_RATE_P99,
        json.dumps(
            TAPE_REJECTION_REASONS,
            sort_keys=True,
            separators=(",", ":"),
        ),
        TAPE_SITE_SPEED_MEAN_MPS,
    )


def _build_manifest(
    source_manifest: Mapping[str, Any],
    *,
    speed_center_mps: float,
    speed_floor_ratio: float,
    speed_max_mps: float,
    speed_initial_width_mps: float,
    landing_aim_x_m: float,
    landing_aim_y_m: float,
    landing_initial_width_m: float,
) -> Dict[str, Any]:
    manifest = deepcopy(source_manifest)
    if manifest.get("schema_version") != 3:
        raise FastBallMaterializationError(
            "source manifest must use schema_version=3"
        )
    if manifest.get("action_order") != [ACTION_ID]:
        raise FastBallMaterializationError(
            "source manifest must contain exact action_order ['bh_block']"
        )
    actions = manifest.get("actions")
    if type(actions) is not list or len(actions) != 1:
        raise FastBallMaterializationError(
            "source manifest must contain exactly one action row"
        )
    action = _require_object(actions[0], label="source action")
    if (
        action.get("action_id") != ACTION_ID
        or action.get("action_uid") != ACTION_UID
    ):
        raise FastBallMaterializationError(
            "source action identity differs from sealed bh_block identity"
        )
    teacher_rate_min = _require_finite(
        action.get("teacher_rate_min"), label="teacher_rate_min"
    )
    teacher_rate_max = _require_finite(
        action.get("teacher_rate_max"), label="teacher_rate_max"
    )
    if (
        teacher_rate_min != EXPECTED_TEACHER_RATE_MIN
        or teacher_rate_max != EXPECTED_TEACHER_RATE_MAX
    ):
        raise FastBallMaterializationError(
            "source teacher-rate bounds must remain exactly [0.6, 1.0]"
        )
    profile = _require_object(
        action.get("ball_profile"), label="source ball_profile"
    )
    source_center = _require_finite(
        profile.get("incoming_speed_center_mps"),
        label="source incoming speed centre",
    )
    if source_center != SOURCE_SPEED_CENTER_MPS:
        raise FastBallMaterializationError(
            "source incoming speed centre differs from sealed baseline"
        )
    if not 0.0 < speed_floor_ratio < 1.0:
        raise FastBallMaterializationError(
            "incoming speed floor ratio must lie strictly inside (0, 1)"
        )
    speed_min_mps = speed_floor_ratio * speed_center_mps
    if not (
        0.0
        < speed_min_mps
        <= speed_center_mps
        <= speed_max_mps
    ):
        raise FastBallMaterializationError(
            "incoming speed floor/centre/ceiling are not ordered"
        )
    if not (
        0.0 < speed_initial_width_mps
        <= speed_center_mps - speed_min_mps
        and speed_initial_width_mps
        <= speed_max_mps - speed_center_mps
    ):
        raise FastBallMaterializationError(
            "initial incoming-speed widths do not fit the hard support"
        )
    old_lower_max = _require_finite(
        profile.get("incoming_speed_std_lower_max_mps"),
        label="source incoming lower max width",
    )
    old_upper_max = _require_finite(
        profile.get("incoming_speed_std_upper_max_mps"),
        label="source incoming upper max width",
    )
    profile["incoming_speed_center_mps"] = speed_center_mps
    profile["incoming_speed_min_mps"] = speed_min_mps
    profile["incoming_speed_max_mps"] = speed_max_mps
    profile["incoming_speed_std_lower_initial_mps"] = (
        speed_initial_width_mps
    )
    profile["incoming_speed_std_upper_initial_mps"] = (
        speed_initial_width_mps
    )
    profile["incoming_speed_std_lower_max_mps"] = min(
        old_lower_max, speed_center_mps - speed_min_mps
    )
    profile["incoming_speed_std_upper_max_mps"] = min(
        old_upper_max, speed_max_mps - speed_center_mps
    )

    landing = _require_object(
        manifest.get("landing_aim"), label="source landing_aim"
    )
    old_min = _require_pair(
        landing.get("min_w_xy_m"), label="landing_aim.min_w_xy_m"
    )
    old_max = _require_pair(
        landing.get("max_w_xy_m"), label="landing_aim.max_w_xy_m"
    )
    old_lower_widths = _require_pair(
        landing.get("std_lower_max_m"),
        label="landing_aim.std_lower_max_m",
    )
    old_upper_widths = _require_pair(
        landing.get("std_upper_max_m"),
        label="landing_aim.std_upper_max_m",
    )
    landing_center = [landing_aim_x_m, landing_aim_y_m]
    initial_lower = [
        landing_initial_width_m,
        landing_initial_width_m,
    ]
    initial_upper = [
        landing_initial_width_m,
        landing_initial_width_m,
    ]
    hard_min, hard_max = _extend_bounds_for_initial_width(
        center=landing_center,
        lower_bound=old_min,
        upper_bound=old_max,
        initial_lower=initial_lower,
        initial_upper=initial_upper,
    )
    landing["center_w_xy_m"] = landing_center
    landing["min_w_xy_m"] = hard_min
    landing["max_w_xy_m"] = hard_max
    landing["std_lower_initial_m"] = initial_lower
    landing["std_upper_initial_m"] = initial_upper
    landing["std_lower_max_m"] = _clip_widths_to_room(
        old_widths=old_lower_widths,
        center=landing_center,
        bounds=hard_min,
        lower_side=True,
    )
    landing["std_upper_max_m"] = _clip_widths_to_room(
        old_widths=old_upper_widths,
        center=landing_center,
        bounds=hard_max,
        lower_side=False,
    )

    holdout = _require_object(
        manifest.get("holdout"), label="source holdout"
    )
    if holdout.get("samples_per_action") != 768:
        raise FastBallMaterializationError(
            "source holdout must retain exactly 768 samples per action"
        )
    holdout["split_id"] = HOLDOUT_SPLIT_ID
    source_note = manifest.get("notes")
    if type(source_note) is not str or not source_note:
        raise FastBallMaterializationError(
            "source manifest notes must be non-empty"
        )
    manifest["manifest_id"] = MANIFEST_ID
    manifest["notes"] = _diagnostic_note(
        source_note,
        speed_center_mps=speed_center_mps,
        speed_floor_ratio=speed_floor_ratio,
        speed_max_mps=speed_max_mps,
        speed_initial_width_mps=speed_initial_width_mps,
        landing_center=landing_center,
        landing_initial_width_m=landing_initial_width_m,
    )
    return manifest


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_written(
    path: Path,
    *,
    expected_bytes: bytes,
    expected_sha256: str,
) -> Any:
    observed = path.read_bytes()
    if observed != expected_bytes:
        raise FastBallMaterializationError(
            "written bytes differ from prepared payload: {}".format(path)
        )
    if _sha256_bytes(observed) != expected_sha256:
        raise FastBallMaterializationError(
            "written SHA-256 differs from prepared digest: {}".format(path)
        )
    reparsed = _load_json(path, expected_sha256=expected_sha256)
    if _canonical_json_bytes(reparsed) != expected_bytes:
        raise FastBallMaterializationError(
            "written JSON is not canonical+newline: {}".format(path)
        )
    return reparsed


def materialize_fast_ball_bootstrap(
    *,
    repo_root: Path,
    source_bundle: Path,
    expected_source_bundle_sha256: str,
    output_dir: Path,
    speed_center_mps: float = DEFAULT_SPEED_CENTER_MPS,
    speed_floor_ratio: float = DEFAULT_SPEED_FLOOR_RATIO,
    speed_max_mps: float = DEFAULT_SPEED_MAX_MPS,
    speed_initial_width_mps: float = DEFAULT_SPEED_INITIAL_WIDTH_MPS,
    landing_aim_x_m: float = DEFAULT_LANDING_AIM_X_M,
    landing_aim_y_m: float = DEFAULT_LANDING_AIM_Y_M,
    landing_initial_width_m: float = DEFAULT_LANDING_INITIAL_WIDTH_M,
) -> Dict[str, Any]:
    """Create and strictly reread one fast-ball manifest/bundle pair."""

    root = Path(repo_root).resolve(strict=True)
    source_bundle_path = Path(source_bundle).resolve(strict=True)
    _repo_relative(source_bundle_path, root, label="source bundle")
    _require_sha256(
        expected_source_bundle_sha256,
        label="expected source bundle SHA-256",
    )
    source_bundle_document = _require_object(
        _load_json(
            source_bundle_path,
            expected_sha256=expected_source_bundle_sha256,
        ),
        label="source bundle",
    )
    if (
        source_bundle_document.get("schema_version") != 2
        or source_bundle_document.get("artifact_type")
        != "n1_contact_training_bundle_v2"
        or source_bundle_document.get("action_id") != ACTION_ID
        or source_bundle_document.get("action_uid") != ACTION_UID
        or source_bundle_document.get("scope") != "upper"
    ):
        raise FastBallMaterializationError(
            "source bundle is not the sealed upper bh_block v2 identity"
        )
    for key in (
        "source_manifest",
        "profile_pins",
        "motion",
        "prototype",
        "manifest",
        "contact_alignment",
        "dynamic_ready",
        "geometry",
    ):
        pin = source_bundle_document.get(key)
        if key == "dynamic_ready":
            dynamic_ready = _require_object(
                pin, label="source bundle dynamic_ready"
            )
            _resolve_pin_path(
                root,
                dynamic_ready.get("artifact"),
                label="dynamic-ready artifact",
            )
            _resolve_pin_path(
                root,
                dynamic_ready.get("nominal_hold_receipt"),
                label="nominal-hold receipt",
            )
        else:
            _resolve_pin_path(root, pin, label="source bundle {}".format(key))

    manifest_pin = _require_object(
        source_bundle_document.get("manifest"),
        label="source bundle manifest pin",
    )
    source_manifest_path = _resolve_pin_path(
        root, manifest_pin, label="source manifest"
    )
    source_manifest = _require_object(
        _load_json(
            source_manifest_path,
            expected_sha256=manifest_pin["sha256"],
        ),
        label="source manifest",
    )

    numeric_arguments = {
        "speed_center_mps": speed_center_mps,
        "speed_floor_ratio": speed_floor_ratio,
        "speed_max_mps": speed_max_mps,
        "speed_initial_width_mps": speed_initial_width_mps,
        "landing_aim_x_m": landing_aim_x_m,
        "landing_aim_y_m": landing_aim_y_m,
        "landing_initial_width_m": landing_initial_width_m,
    }
    finite_arguments = {
        name: _require_finite(value, label=name)
        for name, value in numeric_arguments.items()
    }
    tape_bound_arguments = {
        "speed_center_mps": DEFAULT_SPEED_CENTER_MPS,
        "speed_floor_ratio": DEFAULT_SPEED_FLOOR_RATIO,
        "speed_max_mps": DEFAULT_SPEED_MAX_MPS,
        "speed_initial_width_mps": DEFAULT_SPEED_INITIAL_WIDTH_MPS,
        "landing_aim_x_m": DEFAULT_LANDING_AIM_X_M,
        "landing_aim_y_m": DEFAULT_LANDING_AIM_Y_M,
        "landing_initial_width_m": DEFAULT_LANDING_INITIAL_WIDTH_M,
    }
    changed = {
        name: {
            "expected": tape_bound_arguments[name],
            "observed": finite_arguments[name],
        }
        for name in tape_bound_arguments
        if finite_arguments[name] != tape_bound_arguments[name]
    }
    if changed:
        raise FastBallMaterializationError(
            "this producer is locked to exact diagnostic tape {} and "
            "cannot accept numeric overrides without a new tape: {}".format(
                TAPE_SHA256, changed
            )
        )
    if finite_arguments["speed_center_mps"] <= 0.0:
        raise FastBallMaterializationError(
            "speed_center_mps must be positive"
        )
    if finite_arguments["speed_initial_width_mps"] <= 0.0:
        raise FastBallMaterializationError(
            "speed_initial_width_mps must be positive"
        )
    if finite_arguments["landing_initial_width_m"] <= 0.0:
        raise FastBallMaterializationError(
            "landing_initial_width_m must be positive"
        )
    manifest = _build_manifest(source_manifest, **finite_arguments)

    destination = Path(output_dir).resolve()
    output_relative_dir = _repo_relative(
        destination, root, label="output directory"
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_bytes)
    manifest_name = "bh_block.manifest.v3.{}.json".format(
        manifest_sha[:12]
    )
    manifest_relative = (
        PurePosixPath(output_relative_dir) / manifest_name
    ).as_posix()

    bundle = deepcopy(source_bundle_document)
    bundle["manifest"] = {
        "path": manifest_relative,
        "sha256": manifest_sha,
        "schema_version": 3,
        "action_order": [ACTION_ID],
    }
    bundle_bytes = _canonical_json_bytes(bundle)
    bundle_sha = _sha256_bytes(bundle_bytes)
    bundle_name = "bh_block.bundle.v2.{}.json".format(bundle_sha[:12])
    bundle_relative = (
        PurePosixPath(output_relative_dir) / bundle_name
    ).as_posix()

    destination.mkdir(parents=True, exist_ok=True)
    outputs = (
        (destination / manifest_name, manifest_bytes),
        (destination / bundle_name, bundle_bytes),
    )
    collisions = [path for path, _payload in outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "no-clobber output already exists: "
            + ", ".join(str(path) for path in collisions)
        )
    for path, payload in outputs:
        _exclusive_write(path, payload)

    written_manifest = _require_object(
        _verify_written(
            destination / manifest_name,
            expected_bytes=manifest_bytes,
            expected_sha256=manifest_sha,
        ),
        label="written manifest",
    )
    written_bundle = _require_object(
        _verify_written(
            destination / bundle_name,
            expected_bytes=bundle_bytes,
            expected_sha256=bundle_sha,
        ),
        label="written bundle",
    )
    if written_manifest != manifest:
        raise FastBallMaterializationError(
            "written manifest semantic roundtrip differs"
        )
    if written_bundle != bundle:
        raise FastBallMaterializationError(
            "written bundle semantic roundtrip differs"
        )
    if written_bundle["manifest"] != {
        "path": manifest_relative,
        "sha256": manifest_sha,
        "schema_version": 3,
        "action_order": [ACTION_ID],
    }:
        raise FastBallMaterializationError(
            "written bundle does not pin the new strict manifest"
        )
    source_without_manifest = deepcopy(source_bundle_document)
    source_without_manifest.pop("manifest")
    bundle_without_manifest = deepcopy(written_bundle)
    bundle_without_manifest.pop("manifest")
    if bundle_without_manifest != source_without_manifest:
        raise FastBallMaterializationError(
            "new bundle changed an identity other than its manifest pin"
        )
    return {
        "status": "PASS",
        "publication_class": "diagnostic_comparison_only",
        "formal_admission_claim": False,
        "source_bundle_path": _repo_relative(
            source_bundle_path, root, label="source bundle"
        ),
        "source_bundle_sha256": expected_source_bundle_sha256,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha,
        "bundle_path": bundle_relative,
        "bundle_sha256": bundle_sha,
        "incoming_speed_center_mps": (
            finite_arguments["speed_center_mps"]
        ),
        "incoming_speed_multiplier": (
            finite_arguments["speed_center_mps"]
            / SOURCE_SPEED_CENTER_MPS
        ),
        "landing_aim_center_w_xy_m": [
            finite_arguments["landing_aim_x_m"],
            finite_arguments["landing_aim_y_m"],
        ],
        "exact_tape": {
            "sha256": TAPE_SHA256,
            "proposal_count": TAPE_PROPOSALS,
            "geometry_solved_unrestricted_count": (
                TAPE_GEOMETRY_SOLVED_UNRESTRICTED
            ),
            "admitted_count": TAPE_ADMITTED,
            "admit_rate": TAPE_ADMITTED / TAPE_PROPOSALS,
            "rejection_reasons": dict(TAPE_REJECTION_REASONS),
            "teacher_rate_unrestricted": {
                "mean": TAPE_TEACHER_RATE_MEAN,
                "p01": TAPE_TEACHER_RATE_P01,
                "p50": TAPE_TEACHER_RATE_P50,
                "p99": TAPE_TEACHER_RATE_P99,
            },
            "formal_minimum_admit_rate": 0.95,
            "formal_threshold_status": "FAIL",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT_DEFAULT),
        help="exact training repository root",
    )
    parser.add_argument(
        "--source-bundle",
        default=str(SOURCE_BUNDLE_RELATIVE_PATH),
    )
    parser.add_argument(
        "--expected-source-bundle-sha256",
        default=SOURCE_BUNDLE_SHA256,
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR_RELATIVE_PATH),
    )
    parser.add_argument(
        "--incoming-speed-center-mps",
        type=float,
        default=DEFAULT_SPEED_CENTER_MPS,
    )
    parser.add_argument(
        "--incoming-speed-floor-ratio",
        type=float,
        default=DEFAULT_SPEED_FLOOR_RATIO,
    )
    parser.add_argument(
        "--incoming-speed-max-mps",
        type=float,
        default=DEFAULT_SPEED_MAX_MPS,
    )
    parser.add_argument(
        "--incoming-speed-initial-width-mps",
        type=float,
        default=DEFAULT_SPEED_INITIAL_WIDTH_MPS,
    )
    parser.add_argument(
        "--landing-aim-x-m",
        type=float,
        default=DEFAULT_LANDING_AIM_X_M,
    )
    parser.add_argument(
        "--landing-aim-y-m",
        type=float,
        default=DEFAULT_LANDING_AIM_Y_M,
    )
    parser.add_argument(
        "--landing-initial-width-m",
        type=float,
        default=DEFAULT_LANDING_INITIAL_WIDTH_M,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_parser().parse_args(argv)
    repo_root = Path(arguments.repo_root).resolve(strict=True)

    def under_root(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else repo_root / candidate

    result = materialize_fast_ball_bootstrap(
        repo_root=repo_root,
        source_bundle=under_root(arguments.source_bundle),
        expected_source_bundle_sha256=(
            arguments.expected_source_bundle_sha256
        ),
        output_dir=under_root(arguments.output_dir),
        speed_center_mps=arguments.incoming_speed_center_mps,
        speed_floor_ratio=arguments.incoming_speed_floor_ratio,
        speed_max_mps=arguments.incoming_speed_max_mps,
        speed_initial_width_mps=(
            arguments.incoming_speed_initial_width_mps
        ),
        landing_aim_x_m=arguments.landing_aim_x_m,
        landing_aim_y_m=arguments.landing_aim_y_m,
        landing_initial_width_m=arguments.landing_initial_width_m,
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
