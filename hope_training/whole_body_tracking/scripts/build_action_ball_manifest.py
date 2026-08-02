#!/usr/bin/env python3
"""Build + verify action-conditioned ball-first manifests (schema v3) from a ChingMu unit batch.

人话:把逐 unit 的 ChingMu 素材表(击球帧、站位、来球速度、球位)翻译成 action-ball 严格
manifest——每个动作一份"以实测最佳球为中心"的来球分布,再用正式 loader/adapter/sampler
在 CPU 上验证。只产 JSON,不发射训练,不授权任何 admission。

Producer order bound by the output: ``action -> incoming-ball sample -> frozen solve -> attempt``.

Subcommands
    build   read the batch manifest + clips, emit one manifest JSON (+ .sha256 sidecar
            + .buildreport.json sidecar), then re-load it through the strict loader.
    verify  load an emitted manifest, adapt to sampler profiles, and smoke-sample
            birth+sample rounds per action with validity checks and a reject histogram.

Measured centres (per action)
    contact_offset  = R_z(-yaw_before) @ (ball_pos_hit - station) in B_yaw;
                      z = absolute env ball contact z - canonical-ready root z
    incoming_speed  = |v_in_fit_hope_ms|
    incoming_dir    = R_z(-yaw_before) @ normalize(v_in) (B_yaw; inbound cone from -X axis)
    base_spawn      = station in env W frame (hope -> env via table_frame translation)
    racket authority defaults to ``legacy_fk`` for reproducibility:
        speed       = physical right_racket site (wrist FK + RACKET_SITE_OFFSET_WRIST_M),
                      +/-2-frame central difference at hit_frame_50 (suggest_face_sign 口径)
        mount sign  = sign(n . v) at the hit frame (suggest_face_sign 口径)
    ``--racket-authority measured_channel`` instead requires the complete schema-v4
    measured-racket contract in every NPZ, takes the admitted robot mount-face sign from that
    contract, and computes speed from its physical blade-centre trajectory.  It never falls back
    to wrist FK.  Either sign is cross-checked against the family expectation FH:+1 / BH:-1.

Deliberate deviations from the verbal spec (fail-loud, reported in the build report):
    * time_to_contact centre: the requested 1.0 s centre is infeasible for most units because
      the loader requires min >= t_hit / teacher_rate_min + reaction_margin.  Default policy
      quantizes the midpoint (or --ttc-center-s) to an interior policy tick and derives every
      enabled curriculum width as an integer number of policy ticks.
    * teacher_rate_min is raised above the CLI default per action when the certified
      time-to-contact window would otherwise be narrower than --min-ttc-window-s.
    * solver/physics profile SHA-256 default to test-fixture placeholders; a launch must
      re-pin them from the runtime contract (the runtime cross-check will refuse otherwise).
    * the manifest holdout is a formal promotion split and therefore contains at least
      768 samples per action.  Smaller canary/diagnostic windows must remain separate
      evaluator artifacts; --holdout-samples cannot relabel them as formal heldout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPTS_DIR.parents[2]
MDP_DIR_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
DEFAULT_EXCLUDE = ("Take_085_unit00_FH",)
FAMILY_MAP = {"FH": "forehand", "BH": "backhand"}
FAMILY_EXPECTED_SIGN = {"FH": 1, "BH": -1}
FPS = 50.0
DEFAULT_POLICY_DT_S = 1.0 / FPS
RACKET_AUTHORITY_LEGACY_FK = "legacy_fk"
RACKET_AUTHORITY_MEASURED_CHANNEL = "measured_channel"
MEASURED_RACKET_SCHEMA_VERSION = 4
ROBOT_BUTT_TO_BLADE_AXIS_LOCAL = (
    1.0 / math.sqrt(2.0),
    0.0,
    1.0 / math.sqrt(2.0),
)
ROBOT_RIGID_VISUAL_MESH_SHA256 = (
    "442ff2ecb82d3da481f1500d8a788192ba7d8bc2969f4d8c9d98266ea116b4dd"
)
MEASURED_RACKET_ARRAY_KEYS = (
    "measured_racket_site_pos_w",
    "measured_racket_normal_w",
    "measured_racket_long_axis_w",
)
MEASURED_RACKET_META_KEYS = (
    "measured_racket_schema_version",
    "measured_racket_position_semantics",
    "measured_racket_normal_semantics",
    "measured_racket_long_axis_semantics",
    "measured_racket_robot_mount_normal_sign",
    "measured_racket_robot_butt_to_blade_axis_local",
    "measured_racket_robot_rigid_visual_mesh_sha256",
    "measured_racket_source_sha256",
    "measured_racket_retarget_admitted",
    "measured_racket_retarget_receipt_sha256",
    "measured_racket_joint_order_contract_id",
    "measured_racket_joint_order_contract_sha256",
)
FRESH_N5_ACTION_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
FRESH_N5_FORBIDDEN_ACTION_IDS = frozenset(
    {"fh_loop", "fh_block_syn"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mdp_modules(repo_root: Path):
    """Import the real manifest/adapter/sampler modules as plain top-level modules."""
    mdp_dir = repo_root / MDP_DIR_REL
    if not mdp_dir.is_dir():
        raise SystemExit(f"mdp dir not found: {mdp_dir}")
    if str(mdp_dir) not in sys.path:
        sys.path.insert(0, str(mdp_dir))
    manifest_mod = _load_module("action_ball_manifest", mdp_dir / "action_ball_manifest.py")
    sampling_mod = _load_module("action_ball_sampling", mdp_dir / "action_ball_sampling.py")
    curriculum_mod = _load_module("action_ball_curriculum", mdp_dir / "action_ball_curriculum.py")
    adapter_mod = _load_module(
        "action_ball_profile_adapter", mdp_dir / "action_ball_profile_adapter.py"
    )
    return manifest_mod, sampling_mod, curriculum_mod, adapter_mod


def _rot_z(yaw_rad: float, x: float, y: float):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (c * x - s * y, s * x + c * y)


def _normalize3(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n <= 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _tangent_frame(center):
    """Right-handed orthonormal (u, v) with cross(u, v) == center (unit)."""
    up = (0.0, 0.0, 1.0)
    if abs(_dot(center, up)) > 0.99:
        up = (1.0, 0.0, 0.0)
    u = _normalize3(_cross(up, center))
    v = _normalize3(_cross(center, u))
    return u, v


def _vec2(text: str, name: str):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"{name} must be 'x,y'")
    return parts


def _vec3(text: str, name: str):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"{name} must be 'x,y,z'")
    return parts


def _runtime_style_racket_site_speed(npz_path: Path, strike_frame: int, window: int) -> float:
    """Replicate the runtime's clean reference strike speed as closely as possible.

    hope_commands._ensure_reference_strike_state computes, in float32 motion buffers:
        blade[t] = wrist_pos[t] + R(wrist_quat[t]) @ mount_offset
        clean_lin = (blade[clamp(s+W)] - blade[clamp(s-W)]) / (2*W*dt)
    Note the denominator stays 2*W*dt even when the index clamps at a segment edge,
    unlike suggest_face_sign's span-corrected difference.  The runtime cross-checks
    the manifest pin at abs_tol=1e-6, so we keep the math in float32.
    """
    import numpy as np

    data = np.load(str(npz_path))
    names = [str(n) for n in data["body_names"]] if "body_names" in data.files else None
    widx = names.index("right_wrist_yaw_Link") if names else 31
    pos = np.asarray(data["body_pos_w"], dtype=np.float32)[:, widx]
    quat = np.asarray(data["body_quat_w"], dtype=np.float32)[:, widx]
    total = pos.shape[0]
    offset = np.array([0.21021, 0.032078, 0.032036], dtype=np.float32)

    def blade(frame: int) -> "np.ndarray":
        frame = min(max(frame, 0), total - 1)
        w, x, y, z = (quat[frame, i] for i in range(4))
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        return pos[frame] + rot @ offset

    dt = np.float32(1.0 / FPS)
    diff = blade(strike_frame + window) - blade(strike_frame - window)
    vel = diff / (np.float32(2.0) * np.float32(window) * dt)
    return float(np.linalg.norm(vel.astype(np.float32)))


def _npz_scalar_text(data, key: str, npz_path: Path) -> str:
    """Read one string-like NPZ scalar without enabling pickle."""

    import numpy as np

    raw = np.asarray(data[key]).reshape(-1)
    if raw.size != 1:
        raise SystemExit(f"{npz_path}: {key} must be a scalar")
    value = raw[0]
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(f"{npz_path}: {key} is not valid UTF-8") from exc
    return str(value)


def _require_lower_sha256(value: str, *, key: str, npz_path: Path) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SystemExit(f"{npz_path}: {key} is not a lowercase SHA-256")


def _measured_channel_racket_authority(
    npz_path: Path,
    strike_frame: int,
    window: int,
) -> dict:
    """Load one fail-closed schema-v4 measured-racket authority row.

    The signed physical normal is already the selected hitting face.  The robot-side +Y/-Y face
    needed by the ActionBall manifest is therefore the admitted
    ``measured_racket_robot_mount_normal_sign`` metadata, not a second FK ``sign(n dot v)`` guess.
    Speed follows the runtime clean-reference convention: float32 physical blade-centre positions,
    segment-clamped indices, and the unchanged ``2*W*dt`` denominator at an edge.
    """

    import numpy as np

    if int(window) < 1:
        raise SystemExit(
            "--clean-vel-window must be at least 1 in measured_channel mode"
        )
    required = (*MEASURED_RACKET_ARRAY_KEYS, *MEASURED_RACKET_META_KEYS)
    with np.load(str(npz_path), allow_pickle=False) as data:
        files = set(data.files)
        missing = [key for key in required if key not in files]
        if missing:
            raise SystemExit(
                f"{npz_path}: --racket-authority measured_channel requires the complete "
                f"schema-v4 measured-racket contract; missing {missing}"
            )
        if "body_pos_w" not in files or "fps" not in files:
            raise SystemExit(
                f"{npz_path}: measured_channel requires body_pos_w and fps to bind "
                "the measured teacher to the robot motion clock"
            )
        body_pos_shape = np.asarray(data["body_pos_w"]).shape
        fps_raw = np.asarray(data["fps"]).reshape(-1)
        if (
            len(body_pos_shape) < 1
            or body_pos_shape[0] < 2
            or fps_raw.size != 1
            or not np.issubdtype(fps_raw.dtype, np.number)
            or not np.isfinite(fps_raw[0])
            or float(fps_raw[0]) != FPS
        ):
            raise SystemExit(
                f"{npz_path}: measured_channel requires a >=2-frame {FPS:g} Hz "
                "robot motion clock"
            )
        robot_frame_count = int(body_pos_shape[0])

        raw_schema = np.asarray(data["measured_racket_schema_version"]).reshape(-1)
        if (
            raw_schema.size != 1
            or not np.issubdtype(raw_schema.dtype, np.number)
            or not np.isfinite(raw_schema[0])
            or float(raw_schema[0]) != float(MEASURED_RACKET_SCHEMA_VERSION)
        ):
            raise SystemExit(
                f"{npz_path}: measured_racket_schema_version must be exactly "
                f"{MEASURED_RACKET_SCHEMA_VERSION}"
            )

        semantics = {
            "measured_racket_position_semantics": "physical_blade_center",
            "measured_racket_normal_semantics": "signed_physical_hitting_face",
            "measured_racket_long_axis_semantics": "measured_paddle_butt_to_blade",
        }
        for key, expected in semantics.items():
            actual = _npz_scalar_text(data, key, npz_path)
            if actual != expected:
                raise SystemExit(
                    f"{npz_path}: {key} must be {expected!r}, got {actual!r}"
                )

        sign_raw = np.asarray(
            data["measured_racket_robot_mount_normal_sign"]
        ).reshape(-1)
        if (
            sign_raw.size != 1
            or not np.issubdtype(sign_raw.dtype, np.number)
            or not np.isfinite(sign_raw[0])
            or float(sign_raw[0]) not in (-1.0, 1.0)
        ):
            raise SystemExit(
                f"{npz_path}: measured_racket_robot_mount_normal_sign must be scalar +1/-1"
            )
        mount_sign = int(float(sign_raw[0]))
        axis_local = np.asarray(
            data["measured_racket_robot_butt_to_blade_axis_local"],
            dtype=np.float64,
        ).reshape(-1)
        if axis_local.shape != (3,) or not np.array_equal(
            axis_local, np.asarray(ROBOT_BUTT_TO_BLADE_AXIS_LOCAL)
        ):
            raise SystemExit(
                f"{npz_path}: measured racket robot butt-to-blade axis changed"
            )
        rigid_mesh_sha256 = _npz_scalar_text(
            data, "measured_racket_robot_rigid_visual_mesh_sha256", npz_path
        )
        if rigid_mesh_sha256 != ROBOT_RIGID_VISUAL_MESH_SHA256:
            raise SystemExit(
                f"{npz_path}: measured racket rigid-racket visual mesh SHA changed"
            )

        source_sha256 = _npz_scalar_text(
            data, "measured_racket_source_sha256", npz_path
        )
        receipt_sha256 = _npz_scalar_text(
            data, "measured_racket_retarget_receipt_sha256", npz_path
        )
        joint_order_sha256 = _npz_scalar_text(
            data, "measured_racket_joint_order_contract_sha256", npz_path
        )
        for key, value in (
            ("measured_racket_source_sha256", source_sha256),
            ("measured_racket_retarget_receipt_sha256", receipt_sha256),
            ("measured_racket_joint_order_contract_sha256", joint_order_sha256),
        ):
            _require_lower_sha256(value, key=key, npz_path=npz_path)

        joint_order_id = _npz_scalar_text(
            data, "measured_racket_joint_order_contract_id", npz_path
        )
        if joint_order_id != "a3-gmr-dof-pos-to-runtime-articulation-v1":
            raise SystemExit(
                f"{npz_path}: measured racket joint-order contract id changed"
            )
        admitted = np.asarray(data["measured_racket_retarget_admitted"]).reshape(-1)
        if (
            admitted.size != 1
            or not np.issubdtype(admitted.dtype, np.number)
            or not np.isfinite(admitted[0])
            or float(admitted[0]) != 1.0
        ):
            raise SystemExit(
                f"{npz_path}: measured racket teacher requires an admitted canonical-site retarget"
            )

        position = np.asarray(
            data["measured_racket_site_pos_w"], dtype=np.float32
        )
        normal = np.asarray(data["measured_racket_normal_w"], dtype=np.float64)
        long_axis = np.asarray(
            data["measured_racket_long_axis_w"], dtype=np.float64
        )

    if position.ndim != 2 or position.shape[1:] != (3,) or position.shape[0] < 2:
        raise SystemExit(
            f"{npz_path}: measured_racket_site_pos_w must have shape [T,3], T>=2, "
            f"got {position.shape}"
        )
    expected = position.shape
    if expected[0] != robot_frame_count:
        raise SystemExit(
            f"{npz_path}: measured racket has {expected[0]} frames but robot motion "
            f"has {robot_frame_count}"
        )
    if normal.shape != expected or long_axis.shape != expected:
        raise SystemExit(
            f"{npz_path}: measured racket position/normal/long-axis must all be "
            f"{expected}, got {position.shape}/{normal.shape}/{long_axis.shape}"
        )
    if (
        not np.isfinite(position).all()
        or not np.isfinite(normal).all()
        or not np.isfinite(long_axis).all()
    ):
        raise SystemExit(f"{npz_path}: measured racket channel contains non-finite values")
    normal_norm = np.linalg.norm(normal, axis=-1)
    long_axis_norm = np.linalg.norm(long_axis, axis=-1)
    if float(np.max(np.abs(normal_norm - 1.0))) > 1.0e-3:
        raise SystemExit(f"{npz_path}: measured racket normals are not unit length")
    if float(np.max(np.abs(long_axis_norm - 1.0))) > 1.0e-3:
        raise SystemExit(f"{npz_path}: measured racket long axes are not unit length")
    if float(np.max(np.abs(np.sum(normal * long_axis, axis=-1)))) > 1.0e-3:
        raise SystemExit(
            f"{npz_path}: measured racket face/long axes are not orthogonal"
        )

    frame_count = int(position.shape[0])
    strike_frame = int(strike_frame)
    if strike_frame < 0 or strike_frame >= frame_count:
        raise SystemExit(
            f"{npz_path}: strike frame {strike_frame} outside [0,{frame_count})"
        )
    lo = max(0, strike_frame - int(window))
    hi = min(frame_count - 1, strike_frame + int(window))
    if hi == lo:
        raise SystemExit(
            f"{npz_path}: clip too short for measured-racket finite difference"
        )
    dt = np.float32(1.0 / FPS)
    velocity = (position[hi] - position[lo]) / (
        np.float32(2.0) * np.float32(window) * dt
    )
    runtime_speed = float(np.linalg.norm(velocity.astype(np.float32)))
    if not math.isfinite(runtime_speed) or runtime_speed <= 0.0:
        raise SystemExit(
            f"{npz_path}: measured racket site speed at hit frame is not positive"
        )
    velocity_f64 = (
        position[hi].astype(np.float64) - position[lo].astype(np.float64)
    ) / (2.0 * float(window) / FPS)
    tool_speed = float(np.linalg.norm(velocity_f64))
    signed_face_velocity_cos = float(
        np.dot(normal[strike_frame], velocity_f64 / tool_speed)
    )
    # Reconstruct the robot's unsigned local +Y face diagnostic so the legacy report invariant
    # remains meaningful: sign(mount_sign_cos_clean) should name the selected robot face.  The
    # measured normal itself is already signed, hence raw +Y = signed_normal * mount_sign.
    robot_raw_face_velocity_cos = mount_sign * signed_face_velocity_cos
    return {
        "suggested_sign": mount_sign,
        "speed_clean": tool_speed,
        "runtime_speed": runtime_speed,
        "cos_clean": robot_raw_face_velocity_cos,
        "signed_face_velocity_cos": signed_face_velocity_cos,
        "ambiguous": False,
        "raw_agrees": None,
        "schema_version": MEASURED_RACKET_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "retarget_receipt_sha256": receipt_sha256,
        "joint_order_contract_sha256": joint_order_sha256,
        "robot_butt_to_blade_axis_local": list(ROBOT_BUTT_TO_BLADE_AXIS_LOCAL),
        "robot_rigid_visual_mesh_sha256": rigid_mesh_sha256,
        "frame_count": frame_count,
    }


def _load_measured_bank_receipt(
    *,
    receipt_path: Path,
    expected_sha256: str,
    batch_path: Path,
    batch_sha256: str,
    batch_root: Path,
    units: list,
) -> dict:
    """Bind selected source metadata rows to versioned measured NPZ bytes."""

    _require_lower_sha256(
        expected_sha256,
        key="--expected-measured-bank-receipt-sha256",
        npz_path=receipt_path,
    )
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise SystemExit(
            f"measured bank receipt must be one real file: {receipt_path}"
        )
    actual_receipt_sha256 = _sha256_file(receipt_path)
    if actual_receipt_sha256 != expected_sha256:
        raise SystemExit(
            "measured bank receipt bytes drifted: "
            f"expected {expected_sha256}, got {actual_receipt_sha256}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read measured bank receipt {receipt_path}: {exc}") from exc
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind")
        != "chingmu73_measured_racket_schema_v4_repo_import"
    ):
        raise SystemExit(
            "measured bank receipt must be the schema-v4 repository import receipt"
        )
    source_manifest = receipt.get("source_manifest")
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("file") != batch_path.name
        or source_manifest.get("sha256") != batch_sha256
    ):
        raise SystemExit(
            "measured bank receipt does not bind the exact --batch-manifest bytes"
        )
    authority_manifest = receipt.get("authorities", {}).get("source_manifest", {})
    if authority_manifest.get("sha256") != batch_sha256:
        raise SystemExit(
            "measured bank receipt source-manifest authority SHA disagrees with the batch"
        )
    publication = receipt.get("publication")
    if publication != {
        "all_npz_sha256_verified_before_publish": True,
        "historical_bank_overwritten": False,
        "versioned_sibling": True,
    }:
        raise SystemExit(
            "measured bank receipt lacks the no-overwrite, preverified publication contract"
        )

    receipt_root = receipt_path.parent.resolve()
    if batch_root.resolve() != receipt_root:
        raise SystemExit(
            f"measured bank receipt root {receipt_root} differs from --batch-root "
            f"{batch_root.resolve()}"
        )
    actions = receipt.get("actions")
    if not isinstance(actions, list) or len(actions) != len(units):
        raise SystemExit(
            f"measured bank receipt must contain exactly {len(units)} selected actions"
        )
    denominators = receipt.get("denominators")
    if not isinstance(denominators, dict):
        raise SystemExit("measured bank receipt lacks denominators")
    for key in (
        "catalog_actions",
        "materialized_npz",
        "schema_v4_npz",
        "solver_admitted",
        "solver_all_gates_true",
        "fk_audit_admitted",
        "fk_audit_all_gates_true",
        "fk_audit_finite",
    ):
        if type(denominators.get(key)) is not int or denominators[key] != len(units):
            raise SystemExit(
                f"measured bank receipt denominator {key} must equal {len(units)}"
            )

    rows_by_uid = {}
    seen_files = set()
    total_frames = 0
    for clip_id, (unit, row) in enumerate(zip(units, actions)):
        if not isinstance(row, dict):
            raise SystemExit(f"measured bank receipt action {clip_id} is not an object")
        uid = unit["uid"]
        filename = row.get("file")
        if (
            type(row.get("clip_id")) is not int
            or row.get("clip_id") != clip_id
            or row.get("uid") != uid
            or type(filename) is not str
            or not filename
            or Path(filename).name != filename
        ):
            raise SystemExit(
                f"measured bank receipt action order/path disagrees at {uid}"
            )
        if uid in rows_by_uid:
            raise SystemExit(f"duplicate measured bank receipt UID {uid}")
        if filename in seen_files:
            raise SystemExit(f"duplicate measured bank receipt file {filename}")
        if (
            type(row.get("frames")) is not int
            or row.get("frames") != unit.get("T")
            or type(row.get("hit_frame_50")) is not int
            or row.get("hit_frame_50") != unit.get("hit_frame_50")
            or type(row.get("robot_mount_normal_sign")) is not int
            or row.get("robot_mount_normal_sign") not in (-1, 1)
        ):
            raise SystemExit(
                f"measured bank receipt frame/hit/sign metadata disagrees at {uid}"
            )
        digest = row.get("sha256")
        if type(digest) is not str:
            raise SystemExit(f"measured bank receipt SHA is missing at {uid}")
        _require_lower_sha256(
            digest,
            key=f"measured bank receipt action {uid} sha256",
            npz_path=receipt_path,
        )
        candidate = receipt_root / filename
        if candidate.is_symlink() or candidate.resolve().parent != receipt_root:
            raise SystemExit(
                f"measured bank receipt action {uid} is not one real root-local file"
            )
        rows_by_uid[uid] = row
        seen_files.add(filename)
        total_frames += int(row["frames"])
    if denominators.get("total_materialized_frames") != total_frames:
        raise SystemExit(
            "measured bank receipt total_materialized_frames disagrees with its actions"
        )
    return {
        "path": receipt_path,
        "sha256": actual_receipt_sha256,
        "root": receipt_root,
        "rows_by_uid": rows_by_uid,
    }


def _canonical_ready_root_z(npz_path: Path) -> float:
    """Return the motion's canonical-ready pelvis/root Z (frame 0, body 0)."""
    import numpy as np

    with np.load(str(npz_path), allow_pickle=False) as data:
        missing = {"body_names", "body_pos_w"} - set(data.files)
        if missing:
            raise SystemExit(
                f"{npz_path}: motion is missing {sorted(missing)}"
            )
        names_raw = np.asarray(data["body_names"])
        body_pos_w = np.asarray(data["body_pos_w"])
    if names_raw.ndim != 1:
        raise SystemExit(
            f"{npz_path}: body_names must be one-dimensional"
        )
    body_names = []
    for value in names_raw.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        body_names.append(str(value))
    if (
        not body_names
        or len(body_names) != len(set(body_names))
        or any(not name for name in body_names)
    ):
        raise SystemExit(
            f"{npz_path}: body_names must be unique non-empty strings"
        )
    try:
        root_index = body_names.index("pelvis_link")
    except ValueError as exc:
        raise SystemExit(
            f"{npz_path}: body_names is missing pelvis_link"
        ) from exc
    if (
        body_pos_w.ndim != 3
        or body_pos_w.shape[0] < 1
        or body_pos_w.shape[1] != len(body_names)
        or body_pos_w.shape[2] != 3
    ):
        raise SystemExit(
            f"{npz_path}: body_pos_w must have shape [T, B, 3], "
            f"got {body_pos_w.shape}"
        )
    root_z = float(np.float32(body_pos_w[0, root_index, 2]))
    if not math.isfinite(root_z):
        raise SystemExit(
            f"{npz_path}: canonical-ready pelvis/root Z must be finite"
        )
    return root_z


def _ttc_lattice(
    *,
    continuous_min_s: float,
    continuous_max_s: float,
    requested_center_s: float | None,
    requested_initial_width_s: float,
    policy_dt_s: float,
    label: str,
) -> dict:
    """Quantize a proven TTC interval inward onto the policy-step lattice."""

    values = (
        continuous_min_s,
        continuous_max_s,
        requested_initial_width_s,
        policy_dt_s,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise SystemExit(f"{label}: TTC lattice inputs must be finite")
    step = float(policy_dt_s)
    if step <= 0.0:
        raise SystemExit(f"{label}: policy_dt_s must be positive")
    if requested_initial_width_s <= 0.0:
        raise SystemExit(
            f"{label}: requested TTC initial width must be positive"
        )
    if not 0.0 <= continuous_min_s < continuous_max_s:
        raise SystemExit(
            f"{label}: invalid continuous TTC interval "
            f"[{continuous_min_s}, {continuous_max_s}]"
        )
    epsilon = 1.0e-12
    lower_tick = int(math.ceil(continuous_min_s / step - epsilon))
    upper_tick = int(math.floor(continuous_max_s / step + epsilon))
    if upper_tick - lower_tick < 2:
        raise SystemExit(
            f"{label}: TTC interval contains fewer than three policy ticks; "
            "cannot keep both curriculum sides enabled"
        )
    if requested_center_s is None:
        center_raw = 0.5 * (continuous_min_s + continuous_max_s)
    else:
        if not math.isfinite(float(requested_center_s)):
            raise SystemExit(f"{label}: requested TTC center must be finite")
        center_raw = float(requested_center_s)
    center_tick = int(math.floor(center_raw / step + 0.5))
    center_tick = min(
        max(center_tick, lower_tick + 1),
        upper_tick - 1,
    )
    initial_ticks_requested = max(
        1,
        int(math.floor(float(requested_initial_width_s) / step + 0.5)),
    )
    lower_max_ticks = center_tick - lower_tick
    upper_max_ticks = upper_tick - center_tick
    lower_initial_ticks = min(initial_ticks_requested, lower_max_ticks)
    upper_initial_ticks = min(initial_ticks_requested, upper_max_ticks)
    if lower_initial_ticks < 1 or upper_initial_ticks < 1:
        raise SystemExit(
            f"{label}: both TTC curriculum sides require at least one tick"
        )

    def seconds(ticks: int) -> float:
        return float(ticks * step)

    return {
        "policy_dt_s": step,
        "lower_tick": lower_tick,
        "center_tick": center_tick,
        "upper_tick": upper_tick,
        # Decimal control periods are not exact binary floats.  When the proven lower bound is
        # mathematically on a policy tick, e.g. ``0.96 / 0.6 + 0.1 == 1.7``, Python may represent
        # it as 1.7000000000000002 while ``85 * 0.02`` is 1.7.  The manifest loader is correctly
        # strict about feasibility, so publish the conservative side of that one-ULP ambiguity
        # rather than weakening its gate or jumping a whole control tick.
        "min_s": max(seconds(lower_tick), float(continuous_min_s)),
        "center_s": seconds(center_tick),
        "max_s": min(seconds(upper_tick), float(continuous_max_s)),
        "lower_initial_s": seconds(lower_initial_ticks),
        "lower_max_s": seconds(lower_max_ticks),
        "upper_initial_s": seconds(upper_initial_ticks),
        "upper_max_s": seconds(upper_max_ticks),
    }


def _validate_fresh_n5_build_request(
    *,
    units: list,
    args,
    exact_geometry_sha256: str,
) -> None:
    action_ids = tuple(str(unit.get("uid", "")).lower() for unit in units)
    if action_ids != FRESH_N5_ACTION_ORDER:
        raise SystemExit(
            "fresh N5 action order must be exactly "
            f"{list(FRESH_N5_ACTION_ORDER)}, got {list(action_ids)}"
        )
    if set(action_ids) & FRESH_N5_FORBIDDEN_ACTION_IDS:
        raise SystemExit("fresh N5 contains forbidden fh_loop or fh_block_syn")
    expected_families = ("BH", "FH", "BH", "BH", "FH")
    families = tuple(unit.get("family") for unit in units)
    if families != expected_families:
        raise SystemExit(
            "fresh N5 family order must be BH,FH,BH,BH,FH; "
            f"got {list(families)}"
        )
    if args.skip_npz_hash:
        raise SystemExit("fresh N5 forbids --skip-npz-hash")
    if args.prototype_scope != "upper":
        raise SystemExit("fresh N5 launch scope must be exactly upper")
    if not args.motion_path_prefix:
        raise SystemExit(
            "fresh N5 requires a repo-relative --motion-path-prefix"
        )
    motion_prefix = Path(args.motion_path_prefix)
    if motion_prefix.is_absolute() or ".." in motion_prefix.parts:
        raise SystemExit(
            "fresh N5 --motion-path-prefix must stay inside the repo root"
        )
    prototype_path = Path(args.prototype_path)
    if prototype_path.is_absolute() or ".." in prototype_path.parts:
        raise SystemExit(
            "fresh N5 --prototype-path must stay inside the repo root"
        )
    expected_geometry = args.expected_geometry_source_sha256
    if expected_geometry != exact_geometry_sha256:
        raise SystemExit(
            "fresh N5 expected geometry SHA must equal the current "
            f"exact_face_contact_v2 payload: expected={expected_geometry!r}, "
            f"current={exact_geometry_sha256}"
        )
    placeholders = {
        hashlib.sha256(b"solver").hexdigest(),
        hashlib.sha256(b"physics").hexdigest(),
    }
    if (
        args.solver_profile_sha256 in placeholders
        or args.physics_profile_sha256 in placeholders
    ):
        raise SystemExit(
            "fresh N5 forbids placeholder solver/physics profile pins"
        )


def _build_action(unit, args, face_row, report_rows):
    uid_raw = unit["uid"]
    action_id = uid_raw.lower()
    family_raw = unit["family"]
    family = FAMILY_MAP[family_raw]
    motion_sha256 = unit["npz_sha256"]

    # Native runtime truth (hope_commands._initialize_action_ball_runtime):
    #   t_hit  = round(strike_phase*(T-1)) * step_dt  == hit_frame_50 / 50
    #   t_cycle = (T-1) * step_dt                     == (T-1) / 50   (NOT T/50)
    # and the strike frame must lie strictly inside the segment.
    hit_frame = int(unit["hit_frame_50"])
    total = int(unit["T"])
    if not 0 < hit_frame < total - 1:
        raise SystemExit(f"{uid_raw}: hit frame {hit_frame} not strictly inside segment T={total}")
    if round(unit["strike_phase"] * (total - 1)) != hit_frame:
        raise SystemExit(
            f"{uid_raw}: rounded strike_phase*(T-1) disagrees with hit_frame_50; "
            "the runtime would target a different strike frame"
        )
    t_hit = hit_frame / FPS
    t_cycle = (total - 1) / FPS
    if not t_cycle > t_hit:
        raise SystemExit(f"{uid_raw}: (T-1)/50 must exceed hit_frame_50/50")

    # --- certified teacher-rate window and time-to-contact domain -----------------
    margin = args.reaction_margin_s
    rate_max = args.teacher_rate_max
    denom = t_hit + 1.0 - margin - args.min_ttc_window_s
    if denom <= t_hit:
        raise SystemExit(f"{uid_raw}: min TTC window leaves no feasible teacher_rate_min")
    rate_min_needed = t_hit / denom * (1.0 + 1e-9)
    rate_min = max(args.teacher_rate_min, rate_min_needed)
    rate_min_bumped = rate_min > args.teacher_rate_min
    if not rate_min <= 1.0 <= rate_max:
        raise SystemExit(f"{uid_raw}: teacher rate range [{rate_min}, {rate_max}] excludes 1.0")

    ttc_min = t_hit / rate_min + margin
    ttc_max = t_hit / rate_max + 1.0
    if not ttc_min < ttc_max:
        raise SystemExit(f"{uid_raw}: empty time-to-contact window [{ttc_min}, {ttc_max}]")
    ttc_lattice = _ttc_lattice(
        continuous_min_s=ttc_min,
        continuous_max_s=ttc_max,
        requested_center_s=args.ttc_center_s,
        requested_initial_width_s=args.ttc_std_initial_s,
        policy_dt_s=args.policy_dt_s,
        label=uid_raw,
    )
    ttc_min = ttc_lattice["min_s"]
    ttc_center = ttc_lattice["center_s"]
    ttc_max = ttc_lattice["max_s"]
    ttc_lower_initial = ttc_lattice["lower_initial_s"]
    ttc_lower_max = ttc_lattice["lower_max_s"]
    ttc_upper_initial = ttc_lattice["upper_initial_s"]
    ttc_upper_max = ttc_lattice["upper_max_s"]

    # --- measured ball centre, rotated into B_yaw ---------------------------------
    yaw_rad = math.radians(unit["yaw_before_deg"])
    station = unit["station_xy_hope_m"]
    ball = unit["ball_pos_hit_hope_m"]
    dx, dy = ball[0] - station[0], ball[1] - station[1]
    off_x, off_y = _rot_z(-yaw_rad, dx, dy)
    # ``ball[2]`` is measured above the table surface, hence the first sum is
    # an absolute environment-local contact Z.  The sampler later reconstructs
    # world contact position as ``base_goal + contact_offset`` and the adapter
    # installs the selected motion's canonical-ready root Z into base_goal.
    # Subtract it here exactly once; retaining the absolute Z as an offset would
    # add the ready root height a second time at runtime.
    absolute_contact_z = float(ball[2]) + float(args.surface_z)
    ready_root_z = _canonical_ready_root_z(args._npz_path_for_unit)
    off_z = absolute_contact_z - ready_root_z
    contact_center = (off_x, off_y, off_z)
    contact_lower_max = tuple(args.contact_std_max)
    contact_upper_max = tuple(args.contact_std_max)
    contact_lower_initial = tuple(args.contact_std_initial)
    contact_upper_initial = tuple(args.contact_std_initial)
    contact_min = tuple(c - w for c, w in zip(contact_center, contact_lower_max))
    contact_max = tuple(c + w for c, w in zip(contact_center, contact_upper_max))

    v_in = unit["v_in_fit_hope_ms"]
    speed_center = math.sqrt(v_in[0] ** 2 + v_in[1] ** 2 + v_in[2] ** 2)
    d_hope = _normalize3(v_in)
    dir_x, dir_y = _rot_z(-yaw_rad, d_hope[0], d_hope[1])
    incoming_center = _normalize3((dir_x, dir_y, d_hope[2]))
    incoming_u, incoming_v = _tangent_frame(incoming_center)

    if args.inbound_axis_mode == "env_neg_x_in_b_yaw":
        # The certified inbound support means "balls approach from the table side"
        # (env -X).  For side-on ready stances (fivebind aim-rotated clips, frame-0
        # pelvis yaw up to ~114 deg) env -X expressed in B_yaw is far from B_yaw -X,
        # so the axis must be rotated per action; the fixed -X axis of the ChingMu
        # batches is the yaw~0 special case of the same rule.
        axis_x, axis_y = _rot_z(-yaw_rad, -1.0, 0.0)
        inbound_axis = _normalize3((axis_x, axis_y, 0.0))
    else:
        inbound_axis = (-1.0, 0.0, 0.0)
    center_to_axis_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, _dot(incoming_center, inbound_axis))))
    )
    radius_deg = math.hypot(args.dir_std_max_deg, args.dir_std_max_deg)
    min_cosine = args.inbound_min_cosine
    limit_deg = math.degrees(math.acos(min_cosine))
    relaxed = False
    if center_to_axis_deg + radius_deg > limit_deg - args.inbound_safety_deg:
        needed = center_to_axis_deg + radius_deg + args.inbound_safety_deg
        if needed >= 89.5:
            raise SystemExit(
                f"{uid_raw}: incoming direction {center_to_axis_deg:.1f} deg off the inbound "
                "axis; cannot certify an inbound cone (min_cosine would need to be < 0)"
            )
        min_cosine = math.cos(math.radians(needed))
        relaxed = True

    speed_min = 0.4 * speed_center  # loader requires exactly 0.4x
    speed_lower_max = min(args.speed_lower_max_frac * speed_center, speed_center - speed_min)
    speed_upper_max = args.speed_upper_max
    speed_max = speed_center + speed_upper_max
    speed_lower_initial = min(args.speed_std_initial, speed_lower_max)
    speed_upper_initial = min(args.speed_std_initial, speed_upper_max)

    spin_center = args.spin_mag_center
    spin_min = 0.0
    spin_max = args.spin_mag_max
    spin_lower_max = min(args.spin_mag_lower_std_max, spin_center - spin_min)
    spin_lower_initial = min(args.spin_mag_lower_std_initial, spin_lower_max)
    spin_upper_max = min(args.spin_mag_upper_std_max, spin_max - spin_center)
    spin_upper_initial = min(args.spin_mag_upper_std_initial, spin_upper_max)
    # canonical no-spin direction frame (test-fixture convention)
    spin_dir_center = (0.0, 1.0, 0.0)
    spin_dir_u = (0.0, 0.0, 1.0)
    spin_dir_v = (1.0, 0.0, 0.0)

    spawn_x = station[0] + args.near_x
    spawn_y = station[1] + TABLE_WIDTH / 2.0
    spawn_center = (spawn_x, spawn_y)
    spawn_span = tuple(args.base_spawn_span)
    spawn_std_initial = tuple(args.base_spawn_std_initial)
    spawn_std_max = tuple(args.base_spawn_std_max)
    spawn_min = tuple(c - w for c, w in zip(spawn_center, spawn_span))
    spawn_max = tuple(c + w for c, w in zip(spawn_center, spawn_span))

    dir_sides = {}
    for prefix, initial, maximum in (
        ("incoming_direction", args.dir_std_initial_deg, args.dir_std_max_deg),
        ("spin_direction", args.spin_dir_std_initial_deg, args.spin_dir_std_max_deg),
    ):
        for side in ("u_neg", "u_pos", "v_neg", "v_pos"):
            dir_sides[f"{prefix}_tangent_{side}_initial_deg"] = initial
            dir_sides[f"{prefix}_tangent_{side}_max_deg"] = maximum

    ball_profile = {
        "contact_offset_center_b_yaw_m": list(contact_center),
        "contact_offset_std_lower_initial_m": list(contact_lower_initial),
        "contact_offset_std_lower_max_m": list(contact_lower_max),
        "contact_offset_std_upper_initial_m": list(contact_upper_initial),
        "contact_offset_std_upper_max_m": list(contact_upper_max),
        "contact_offset_min_b_yaw_m": list(contact_min),
        "contact_offset_max_b_yaw_m": list(contact_max),
        "time_to_contact_center_s": ttc_center,
        "time_to_contact_std_lower_initial_s": ttc_lower_initial,
        "time_to_contact_std_lower_max_s": ttc_lower_max,
        "time_to_contact_std_upper_initial_s": ttc_upper_initial,
        "time_to_contact_std_upper_max_s": ttc_upper_max,
        "time_to_contact_min_s": ttc_min,
        "time_to_contact_max_s": ttc_max,
        "incoming_direction_center_b_yaw": list(incoming_center),
        "incoming_direction_tangent_u_b_yaw": list(incoming_u),
        "incoming_direction_tangent_v_b_yaw": list(incoming_v),
        "incoming_inbound_axis_b_yaw": list(inbound_axis),
        "incoming_inbound_min_cosine": min_cosine,
        "incoming_speed_center_mps": speed_center,
        "incoming_speed_std_lower_initial_mps": speed_lower_initial,
        "incoming_speed_std_lower_max_mps": speed_lower_max,
        "incoming_speed_std_upper_initial_mps": speed_upper_initial,
        "incoming_speed_std_upper_max_mps": speed_upper_max,
        "incoming_speed_min_mps": speed_min,
        "incoming_speed_max_mps": speed_max,
        "spin_direction_center_b_yaw": list(spin_dir_center),
        "spin_direction_tangent_u_b_yaw": list(spin_dir_u),
        "spin_direction_tangent_v_b_yaw": list(spin_dir_v),
        "spin_magnitude_center_radps": spin_center,
        "spin_magnitude_std_lower_initial_radps": spin_lower_initial,
        "spin_magnitude_std_lower_max_radps": spin_lower_max,
        "spin_magnitude_std_upper_initial_radps": spin_upper_initial,
        "spin_magnitude_std_upper_max_radps": spin_upper_max,
        "spin_magnitude_min_radps": spin_min,
        "spin_magnitude_max_radps": spin_max,
        "base_spawn_center_w_xy_m": list(spawn_center),
        "base_spawn_std_lower_initial_m": list(spawn_std_initial),
        "base_spawn_std_lower_max_m": list(spawn_std_max),
        "base_spawn_std_upper_initial_m": list(spawn_std_initial),
        "base_spawn_std_upper_max_m": list(spawn_std_max),
        "base_spawn_min_w_xy_m": list(spawn_min),
        "base_spawn_max_w_xy_m": list(spawn_max),
        "base_travel_center_b_yaw_xy_m": [0.0, 0.0],
        "base_travel_std_lower_initial_m": [0.0, 0.0],
        "base_travel_std_lower_max_m": [0.0, 0.0],
        "base_travel_std_upper_initial_m": [0.0, 0.0],
        "base_travel_std_upper_max_m": [0.0, 0.0],
        "base_travel_min_b_yaw_xy_m": [0.0, 0.0],
        "base_travel_max_b_yaw_xy_m": [0.0, 0.0],
    }
    ball_profile.update(dir_sides)

    mount_sign = int(face_row["suggested_sign"])
    if args.racket_authority == RACKET_AUTHORITY_MEASURED_CHANNEL:
        racket_speed = float(face_row["runtime_speed"])
    else:
        racket_speed = _runtime_style_racket_site_speed(
            args._npz_path_for_unit, hit_frame, args.clean_vel_window
        )
    tool_speed = float(face_row["speed_clean"])
    if racket_speed <= 0.0:
        raise SystemExit(f"{uid_raw}: racket site speed at hit frame is not positive")

    report_row = {
        "uid": uid_raw,
        "action_id": action_id,
        "family": family,
        "t_hit_s": t_hit,
        "t_cycle_s": t_cycle,
        "teacher_rate_min": rate_min,
        "teacher_rate_min_bumped": rate_min_bumped,
        "ttc_window_s": [ttc_min, ttc_max],
        "ttc_center_s": ttc_center,
        "ttc_lattice": ttc_lattice,
        "contact_offset_center_b_yaw_m": list(contact_center),
        "incoming_speed_center_mps": speed_center,
        "incoming_dir_angle_to_inbound_axis_deg": center_to_axis_deg,
        "inbound_min_cosine": min_cosine,
        "inbound_min_cosine_relaxed": relaxed,
        "inbound_axis_mode": args.inbound_axis_mode,
        "incoming_inbound_axis_b_yaw": list(inbound_axis),
        "base_spawn_center_w_xy_m": list(spawn_center),
        "racket_site_speed_mps": racket_speed,
        "racket_site_speed_tool_f64_mps": tool_speed,
        "racket_site_speed_tool_delta_mps": abs(racket_speed - tool_speed),
        "mount_normal_sign": mount_sign,
        "mount_sign_matches_family": mount_sign == FAMILY_EXPECTED_SIGN[family_raw],
        "mount_sign_ambiguous": bool(face_row["ambiguous"]),
        "mount_sign_cos_clean": float(face_row["cos_clean"]),
        "mount_sign_raw_agrees": face_row.get("raw_agrees"),
    }
    if args.racket_authority == RACKET_AUTHORITY_MEASURED_CHANNEL:
        report_row.update(
            {
                "racket_authority": RACKET_AUTHORITY_MEASURED_CHANNEL,
                "measured_racket_schema_version": face_row["schema_version"],
                "measured_racket_source_sha256": face_row["source_sha256"],
                "measured_racket_retarget_receipt_sha256": face_row[
                    "retarget_receipt_sha256"
                ],
                "measured_racket_joint_order_contract_sha256": face_row[
                    "joint_order_contract_sha256"
                ],
                "measured_signed_face_velocity_cos": face_row[
                    "signed_face_velocity_cos"
                ],
            }
        )
    report_rows.append(report_row)

    if args.motion_path_prefix:
        motion_path = args.motion_path_prefix.rstrip("/") + "/" + unit["npz"].split("/")[-1]
    else:
        motion_path = unit["npz"]
    return {
        "action_id": action_id,
        "family": family,
        "motion_path": motion_path,
        "motion_sha256": motion_sha256,
        "strike_phase": unit["strike_phase"],
        "reference_t_hit_s": t_hit,
        "reference_t_cycle_s": t_cycle,
        "reference_racket_site_speed_mps": racket_speed,
        "reaction_margin_s": margin,
        "teacher_rate_min": rate_min,
        "teacher_rate_max": rate_max,
        "mount_normal_sign": mount_sign,
        "ball_profile": ball_profile,
    }


def cmd_build(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    batch_path = Path(args.batch_manifest).resolve()
    batch_root = Path(args.batch_root).resolve() if args.batch_root else batch_path.parent
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch_sha = _sha256_file(batch_path)

    exclude = set(args.exclude)
    units = [u for u in batch["units"] if u["uid"] not in exclude]
    if len(units) != args.expect_units:
        raise SystemExit(
            f"expected {args.expect_units} units after exclusions, got {len(units)} "
            "(pass --expect-units to override)"
        )

    measured_bank = None
    if args.racket_authority == RACKET_AUTHORITY_MEASURED_CHANNEL:
        if args.skip_npz_hash:
            raise SystemExit(
                "measured_channel forbids --skip-npz-hash; measured bytes are bound "
                "by --measured-bank-receipt"
            )
        if (
            args.measured_bank_receipt is None
            or args.expected_measured_bank_receipt_sha256 is None
        ):
            raise SystemExit(
                "measured_channel requires --measured-bank-receipt and "
                "--expected-measured-bank-receipt-sha256"
            )
        measured_bank = _load_measured_bank_receipt(
            receipt_path=Path(args.measured_bank_receipt).resolve(),
            expected_sha256=args.expected_measured_bank_receipt_sha256,
            batch_path=batch_path,
            batch_sha256=batch_sha,
            batch_root=batch_root,
            units=units,
        )
    elif (
        args.measured_bank_receipt is not None
        or args.expected_measured_bank_receipt_sha256 is not None
    ):
        raise SystemExit(
            "--measured-bank-receipt is only valid with "
            "--racket-authority measured_channel"
        )

    if args.contact_std_initial[0] > args.contact_std_initial[1]:
        raise SystemExit("contact std initial x must be <= y")
    if args.contact_std_max[0] > args.contact_std_max[1]:
        raise SystemExit("contact std max x must be <= y")
    if args.contact_std_max[0] > 0.10:
        raise SystemExit("contact std max x must be <= 0.10 m (schema hard cap)")

    manifest_mod, _, _, _ = _mdp_modules(repo_root)
    if args.fresh_n5_upper:
        _validate_fresh_n5_build_request(
            units=units,
            args=args,
            exact_geometry_sha256=(
                manifest_mod._exact_face_geometry_source_sha256()
            ),
        )
    holdout_floor = max(
        manifest_mod.FORMAL_HOLDOUT_SAMPLES_PER_ACTION_MIN,
        args.min_proposals,
        args.min_safe_closed,
    )
    if args.holdout_samples < holdout_floor:
        raise SystemExit(
            f"holdout samples {args.holdout_samples} below formal per-action "
            f"minimum {holdout_floor}; smaller canary/diagnostic windows must "
            "remain separate evaluator artifacts and cannot populate manifest "
            "holdout"
        )

    # Preserve the legacy FK authority byte-for-byte by default.  The explicit measured mode never
    # imports or calls its FK sign tool, so a missing/partial measured contract cannot fall back.
    compute_face_sign = None
    if args.racket_authority == RACKET_AUTHORITY_LEGACY_FK:
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        from suggest_face_sign import compute_face_sign  # noqa: E402

    report_rows = []
    actions = []
    for source_unit in units:
        unit = source_unit
        receipt_row = None
        if measured_bank is not None:
            receipt_row = measured_bank["rows_by_uid"][source_unit["uid"]]
            unit = dict(source_unit)
            unit["npz"] = receipt_row["file"]
            unit["npz_sha256"] = receipt_row["sha256"]
            npz_path = (measured_bank["root"] / receipt_row["file"]).resolve()
        else:
            npz_path = (batch_root / unit["npz"]).resolve()
        if not npz_path.is_file():
            raise SystemExit(f"missing clip: {npz_path}")
        if not args.skip_npz_hash:
            actual = _sha256_file(npz_path)
            if actual != unit["npz_sha256"]:
                raise SystemExit(
                    f"{unit['uid']}: clip bytes drifted from batch manifest "
                    f"(expected {unit['npz_sha256']}, got {actual})"
                )
        if args.racket_authority == RACKET_AUTHORITY_MEASURED_CHANNEL:
            face_row = _measured_channel_racket_authority(
                npz_path,
                int(unit["hit_frame_50"]),
                args.clean_vel_window,
            )
            if int(unit["T"]) != face_row["frame_count"]:
                raise SystemExit(
                    f"{unit['uid']}: batch manifest T={unit['T']} differs from "
                    f"measured motion T={face_row['frame_count']}"
                )
            if face_row["suggested_sign"] != receipt_row[
                "robot_mount_normal_sign"
            ]:
                raise SystemExit(
                    f"{unit['uid']}: measured NPZ mount sign disagrees with bank receipt"
                )
        else:
            face_row = compute_face_sign(
                str(npz_path), int(unit["hit_frame_50"])
            )
        args._npz_path_for_unit = npz_path
        action = _build_action(unit, args, face_row, report_rows)
        action["action_uid"] = manifest_mod.derive_action_ball_action_uid(
            action["action_id"], action["family"], action["motion_sha256"]
        )
        actions.append(action)

    ordered_keys = (
        "action_id",
        "action_uid",
        "motion_path",
        "motion_sha256",
        "strike_phase",
        "reference_t_hit_s",
        "reference_t_cycle_s",
        "reference_racket_site_speed_mps",
        "reaction_margin_s",
        "teacher_rate_min",
        "teacher_rate_max",
        "family",
        "mount_normal_sign",
        "ball_profile",
    )
    actions = [{key: action[key] for key in ordered_keys} for action in actions]

    prototype_path = repo_root / args.prototype_path
    if not prototype_path.is_file():
        raise SystemExit(f"prototype file not found: {prototype_path}")

    if args.landing_center_env is None:
        landing_center = [args.near_x + 0.75 * TABLE_LENGTH, 0.0]
    else:
        landing_center = list(args.landing_center_env)
    landing_span = list(args.landing_span)
    landing_std_initial = list(args.landing_std_initial)
    landing_std_max = list(args.landing_std_max)

    document = {
        "schema_version": 3,
        "manifest_id": args.manifest_id,
        "mobility_mode": "no_move",
        "action_order": [action["action_id"] for action in actions],
        "prototype": {
            "path": args.prototype_path,
            "sha256": _sha256_file(prototype_path),
            "scope": args.prototype_scope,
        },
        "solver_profile_sha256": args.solver_profile_sha256,
        "physics_profile_sha256": args.physics_profile_sha256,
        "landing_aim": {
            "center_w_xy_m": landing_center,
            "std_lower_initial_m": landing_std_initial,
            "std_lower_max_m": landing_std_max,
            "std_upper_initial_m": landing_std_initial,
            "std_upper_max_m": landing_std_max,
            "min_w_xy_m": [c - w for c, w in zip(landing_center, landing_span)],
            "max_w_xy_m": [c + w for c, w in zip(landing_center, landing_span)],
        },
        "actions": actions,
        "curriculum": {
            "min_proposals": args.min_proposals,
            "min_safe_closed": args.min_safe_closed,
            "target_failure_rate": args.target_failure_rate,
            "failure_band_half_width": args.failure_band_half_width,
            "min_solver_admit_rate": 0.95,
            "min_install_rate": 0.95,
            "min_start_rate": 0.95,
            "min_close_rate": 0.95,
            "max_other_unsafe_rate": 0.02,
            "confidence_z": 1.96,
            "max_center_failures": 8,
        },
        "holdout": {
            "seed": args.holdout_seed,
            "samples_per_action": args.holdout_samples,
            "split_id": args.holdout_split_id,
        },
        "notes": (
            "Built by build_action_ball_manifest.py from "
            f"{batch_path.name} (sha256 {batch_sha}); motion_path and prototype path are "
            "relative to the training repo root. "
            + (
                "solver/physics profile pins were supplied by the caller (host-side "
                "computed from repository contract functions; runtime boot must confirm). "
                if (
                    args.solver_profile_sha256 != hashlib.sha256(b"solver").hexdigest()
                    or args.physics_profile_sha256 != hashlib.sha256(b"physics").hexdigest()
                )
                else "solver_profile_sha256/physics_profile_sha256 are PLACEHOLDER pins and "
                "must be re-pinned from the runtime contract before any launch. "
            )
            + "Metadata only; grants no motion admission."
        ),
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    out_path.write_text(encoded, encoding="utf-8")

    loaded = manifest_mod.load_action_ball_manifest(
        out_path,
        verify_referenced_assets=args.fresh_n5_upper,
        repo_root=repo_root if args.fresh_n5_upper else None,
    )
    file_sha = loaded.file_sha256
    (out_path.parent / (out_path.name + ".sha256")).write_text(
        f"{file_sha}  {out_path.name}\n", encoding="utf-8"
    )

    bumped = [r["uid"] for r in report_rows if r["teacher_rate_min_bumped"]]
    relaxed = [r["uid"] for r in report_rows if r["inbound_min_cosine_relaxed"]]
    sign_mismatch = [r["uid"] for r in report_rows if not r["mount_sign_matches_family"]]
    ambiguous = [r["uid"] for r in report_rows if r["mount_sign_ambiguous"]]
    speeds = sorted(r["racket_site_speed_mps"] for r in report_rows)
    policy_dt = float(args.policy_dt_s)
    episode_need = max(
        r["ttc_window_s"][1]
        + (r["t_cycle_s"] - r["t_hit_s"]) / r["teacher_rate_min"]
        + policy_dt
        for r in report_rows
    )
    report = {
        "builder": "build_action_ball_manifest.py",
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": batch_sha,
        "excluded_uids": sorted(exclude),
        "n_actions": len(actions),
        "out": str(out_path),
        "file_sha256": file_sha,
        "canonical_sha256": loaded.canonical_sha256,
        "racket_site_speed_mps": {
            "min": speeds[0],
            "median": speeds[len(speeds) // 2],
            "max": speeds[-1],
        },
        "min_episode_length_s_needed": episode_need,
        "teacher_rate_min_bumped_uids": bumped,
        "inbound_min_cosine_relaxed_uids": relaxed,
        "mount_sign_family_mismatch_uids": sign_mismatch,
        "mount_sign_ambiguous_uids": ambiguous,
        "per_action": report_rows,
    }
    if args.racket_authority == RACKET_AUTHORITY_MEASURED_CHANNEL:
        report["racket_authority"] = RACKET_AUTHORITY_MEASURED_CHANNEL
        report["measured_bank_receipt"] = str(measured_bank["path"])
        report["measured_bank_receipt_sha256"] = measured_bank["sha256"]
    report_path = out_path.parent / (out_path.name.replace(".json", "") + ".buildreport.json")
    report_path.write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"WROTE {out_path}")
    print(f"file_sha256      {file_sha}")
    print(f"canonical_sha256 {loaded.canonical_sha256}")
    print(f"actions          {len(actions)}")
    print(
        "racket site speed m/s min/median/max "
        f"{speeds[0]:.3f}/{speeds[len(speeds) // 2]:.3f}/{speeds[-1]:.3f}"
    )
    if bumped:
        print(f"WARN teacher_rate_min bumped above CLI default for: {bumped}")
    if relaxed:
        print(f"WARN inbound min_cosine relaxed for: {relaxed}")
    if sign_mismatch:
        print(f"WARN mount sign disagrees with family expectation for: {sign_mismatch}")
    if ambiguous:
        print(f"WARN mount sign ambiguous (|cos|<0.10 or slow) for: {ambiguous}")
    print(f"report           {report_path}")
    return 0


def _check_sample(sample, profile, min_cosine_by_uid):
    """Return a list of violated-reason strings for one sample (empty == legal)."""
    reasons = []

    def _in(lo, x, hi, name):
        if not (lo - 1e-9 <= x <= hi + 1e-9):
            reasons.append(name)

    for axis in range(3):
        _in(
            profile.contact_offset_min_b_yaw_m[axis],
            sample.contact_offset_from_base_goal_b_yaw_m[axis],
            profile.contact_offset_max_b_yaw_m[axis],
            f"contact_offset[{axis}]_out_of_bounds",
        )
    _in(
        profile.time_to_contact_min_s,
        sample.time_to_contact_s,
        profile.time_to_contact_max_s,
        "time_to_contact_out_of_bounds",
    )
    _in(
        profile.incoming_speed_min_mps,
        sample.incoming_speed_mps,
        profile.incoming_speed_max_mps,
        "incoming_speed_out_of_bounds",
    )
    _in(
        profile.spin_magnitude_min_radps,
        sample.spin_magnitude_radps,
        profile.spin_magnitude_max_radps,
        "spin_magnitude_out_of_bounds",
    )
    for axis in range(2):
        _in(
            profile.base_spawn_min_w_m[axis],
            sample.base_start_w_m[axis],
            profile.base_spawn_max_w_m[axis],
            f"base_spawn[{axis}]_out_of_bounds",
        )
        _in(
            profile.landing_aim_min_w_xy_m[axis],
            sample.landing_aim_w_xy_m[axis],
            profile.landing_aim_max_w_xy_m[axis],
            f"landing_aim[{axis}]_out_of_bounds",
        )
    direction = sample.incoming_direction_b_yaw
    norm = math.sqrt(sum(c * c for c in direction))
    if abs(norm - 1.0) > 1e-6:
        reasons.append("incoming_direction_not_unit")
    inbound = _dot(direction, profile.incoming_inbound_axis_b_yaw)
    if inbound < min_cosine_by_uid - 1e-9:
        reasons.append("incoming_direction_outside_inbound_cone")
    spin_dir = sample.spin_direction_b_yaw
    if abs(math.sqrt(sum(c * c for c in spin_dir)) - 1.0) > 1e-6:
        reasons.append("spin_direction_not_unit")
    if profile.mobility_mode == "no_move" and sample.base_goal_w_m != sample.base_start_w_m:
        reasons.append("no_move_base_goal_drifted")
    recomposed = tuple(
        sample.incoming_speed_mps * component for component in sample.incoming_direction_w
    )
    if any(abs(a - b) > 1e-9 for a, b in zip(recomposed, sample.incoming_velocity_w_mps)):
        reasons.append("incoming_velocity_inconsistent")
    try:
        sample.verify_sample_id()
    except ValueError:
        reasons.append("sample_id_mismatch")
    return reasons


def cmd_verify(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest_mod, sampling_mod, _, adapter_mod = _mdp_modules(repo_root)

    if args.assets_root:
        loaded = manifest_mod.load_action_ball_manifest(
            Path(args.manifest).resolve(),
            verify_referenced_assets=True,
            repo_root=Path(args.assets_root).resolve(),
        )
        print(f"REFERENCED ASSETS VERIFIED under {args.assets_root}")
    else:
        loaded = manifest_mod.load_action_ball_manifest(Path(args.manifest).resolve())
    manifest = loaded.manifest
    print(f"LOADED {args.manifest}")
    print(f"file_sha256      {loaded.file_sha256}")
    print(f"canonical_sha256 {loaded.canonical_sha256}")
    print(
        f"actions {len(manifest.actions)}  mobility {manifest.mobility_mode}  "
        f"target_failure_rate {manifest.curriculum.target_failure_rate}"
    )

    bundle = adapter_mod.adapt_action_ball_manifest(manifest)
    sampler = sampling_mod.ActionBallSampler(list(bundle.profiles), seed=args.seed)

    if args.action_ids:
        chosen = list(args.action_ids)
    else:
        by_family = {}
        for action in manifest.actions:
            by_family.setdefault(action.family, action.action_id)
        chosen = list(by_family.values())
        slowest = max(manifest.actions, key=lambda a: a.reference_t_hit_s)
        if slowest.action_id not in chosen:
            chosen.append(slowest.action_id)
        for action in manifest.actions:
            if len(chosen) >= args.n_actions:
                break
            if action.action_id not in chosen:
                chosen.append(action.action_id)
        chosen = chosen[: args.n_actions]

    uid_by_id = {a.action_id: a.action_uid for a in manifest.actions}
    profile_by_uid = {p.action_uid: p for p in bundle.profiles}
    min_cos_by_uid = {
        a.action_uid: a.ball_profile.incoming_inbound_min_cosine for a in manifest.actions
    }

    all_ok = True
    for action_id in chosen:
        uid = uid_by_id[action_id]
        profile = profile_by_uid[uid]
        for label, level in (("level0", 0.0), ("level1", 1.0)):
            n = args.samples if label == "level1" else args.samples_level0
            if n <= 0:
                continue
            levels = sampling_mod.DomainLevels(
                **{
                    field: level
                    for field in sampling_mod.DomainLevels.__dataclass_fields__
                }
            )
            histogram = {}
            legal = 0
            for _ in range(n):
                birth = sampler.reserve_birth(
                    action_uid=uid,
                    domain_epoch=args.domain_epoch,
                    levels=levels,
                    base_yaw_rad=args.base_yaw_rad,
                )
                sample = sampler.sample(
                    birth=birth,
                    action_uid=uid,
                    domain_epoch=args.domain_epoch,
                    levels=levels,
                    base_yaw_rad=args.base_yaw_rad,
                )
                reasons = _check_sample(sample, profile, min_cos_by_uid[uid])
                if reasons:
                    for reason in reasons:
                        histogram[reason] = histogram.get(reason, 0) + 1
                else:
                    legal += 1
            rate = legal / n
            print(
                f"SMOKE {action_id} {label} birth+sample n={n} legal={legal} "
                f"({rate:.1%}) rejects={json.dumps(histogram, sort_keys=True)}"
            )
            if legal != n:
                all_ok = False
    print("SMOKE RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build one manifest JSON from the ChingMu batch")
    b.add_argument("--batch-manifest", required=True)
    b.add_argument("--batch-root", default=None, help="root for unit npz paths (default: manifest dir)")
    b.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    b.add_argument("--out", required=True)
    b.add_argument("--manifest-id", required=True)
    b.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    b.add_argument("--expect-units", type=int, default=73)
    b.add_argument("--skip-npz-hash", action="store_true")
    b.add_argument(
        "--racket-authority",
        choices=(
            RACKET_AUTHORITY_LEGACY_FK,
            RACKET_AUTHORITY_MEASURED_CHANNEL,
        ),
        default=RACKET_AUTHORITY_LEGACY_FK,
        help=(
            "source of mount face and strike-site speed: legacy_fk preserves the "
            "historical wrist-FK/sign(n.v) build; measured_channel requires the "
            "complete admitted schema-v4 measured-racket contract and never falls back"
        ),
    )
    b.add_argument(
        "--measured-bank-receipt",
        default=None,
        help=(
            "required with measured_channel: schema-v4 bank import receipt whose "
            "per-action file/SHA rows replace the legacy SOURCE_MANIFEST NPZ bindings"
        ),
    )
    b.add_argument(
        "--expected-measured-bank-receipt-sha256",
        default=None,
        help=(
            "required with measured_channel: expected SHA-256 of "
            "--measured-bank-receipt"
        ),
    )
    b.add_argument("--clean-vel-window", type=int, default=2,
                   help="frames W for the runtime-style clean strike velocity "
                        "(must equal racket.clean_strike_vel_window at launch)")
    b.add_argument("--motion-path-prefix", default=None,
                   help="rewrite motion_path to '<prefix>/<basename>' (e.g. a repo-root-"
                        "relative clips dir so launch referenced-asset verification passes)")
    b.add_argument("--reaction-margin-s", type=float, default=0.10)
    b.add_argument("--teacher-rate-min", type=float, default=0.6)
    b.add_argument("--teacher-rate-max", type=float, default=1.0)
    b.add_argument("--min-ttc-window-s", type=float, default=0.1)
    b.add_argument("--ttc-center-s", type=float, default=None,
                   help="requested TTC centre, quantized to an interior policy tick "
                        "(default: quantized window midpoint)")
    b.add_argument("--ttc-std-initial-s", type=float, default=0.05)
    b.add_argument(
        "--policy-dt-s",
        type=float,
        default=DEFAULT_POLICY_DT_S,
        help=(
            "exact policy-step duration used to quantize TTC center and all "
            "curriculum widths (default: 0.02 s)"
        ),
    )
    b.add_argument("--contact-std-initial", type=lambda t: _vec3(t, "--contact-std-initial"),
                   default=[0.03, 0.05, 0.05])
    b.add_argument("--contact-std-max", type=lambda t: _vec3(t, "--contact-std-max"),
                   default=[0.08, 0.20, 0.15])
    b.add_argument("--speed-std-initial", type=float, default=0.15)
    b.add_argument("--speed-lower-max-frac", type=float, default=0.6)
    b.add_argument("--speed-upper-max", type=float, default=1.0)
    b.add_argument("--dir-std-initial-deg", type=float, default=3.0)
    b.add_argument("--dir-std-max-deg", type=float, default=15.0)
    b.add_argument("--spin-dir-std-initial-deg", type=float, default=3.0)
    b.add_argument("--spin-dir-std-max-deg", type=float, default=15.0)
    b.add_argument("--spin-mag-center", type=float, default=0.0)
    b.add_argument("--spin-mag-max", type=float, default=60.0)
    b.add_argument("--spin-mag-lower-std-initial", type=float, default=5.0)
    b.add_argument("--spin-mag-lower-std-max", type=float, default=40.0)
    b.add_argument("--spin-mag-upper-std-initial", type=float, default=5.0)
    b.add_argument("--spin-mag-upper-std-max", type=float, default=40.0)
    b.add_argument("--inbound-axis-mode", choices=["fixed_neg_x", "env_neg_x_in_b_yaw"],
                   default="fixed_neg_x",
                   help="certified inbound-support axis: fixed B_yaw -X (ChingMu batches, "
                        "yaw~0 stations) or env -X rotated into B_yaw per action "
                        "(aim-rotated side-on ready stances, e.g. fivebind)")
    b.add_argument("--inbound-min-cosine", type=float, default=0.20)
    b.add_argument("--inbound-safety-deg", type=float, default=0.5)
    b.add_argument("--near-x", type=float, default=0.5)
    b.add_argument("--surface-z", type=float, default=0.76)
    b.add_argument("--landing-center-env", type=lambda t: _vec2(t, "--landing-center-env"),
                   default=None, help="default: opponent half centre (near_x + 2.055, 0)")
    b.add_argument("--landing-span", type=lambda t: _vec2(t, "--landing-span"),
                   default=[0.45, 0.60])
    b.add_argument("--landing-std-initial", type=lambda t: _vec2(t, "--landing-std-initial"),
                   default=[0.01, 0.01])
    b.add_argument("--landing-std-max", type=lambda t: _vec2(t, "--landing-std-max"),
                   default=[0.25, 0.45])
    b.add_argument("--base-spawn-span", type=lambda t: _vec2(t, "--base-spawn-span"),
                   default=[0.30, 0.40])
    b.add_argument("--base-spawn-std-initial", type=lambda t: _vec2(t, "--base-spawn-std-initial"),
                   default=[0.01, 0.01])
    b.add_argument("--base-spawn-std-max", type=lambda t: _vec2(t, "--base-spawn-std-max"),
                   default=[0.15, 0.25])
    b.add_argument("--prototype-path", default="configs/stroke_prototypes_v1_20260727.json")
    b.add_argument("--prototype-scope", default="full")
    b.add_argument(
        "--fresh-n5-upper",
        action="store_true",
        help=(
            "fail-closed fresh launch profile: exact five-action upper/no_move "
            "order, exact-face geometry pin, real asset hashes, and no legacy "
            "fh_loop/fh_block_syn"
        ),
    )
    b.add_argument(
        "--expected-geometry-source-sha256",
        default=None,
        help=(
            "required with --fresh-n5-upper; must equal the current "
            "exact_face_contact_v2 geometry payload SHA-256"
        ),
    )
    b.add_argument("--solver-profile-sha256",
                   default=hashlib.sha256(b"solver").hexdigest(),
                   help="PLACEHOLDER default; re-pin from the runtime solver contract")
    b.add_argument("--physics-profile-sha256",
                   default=hashlib.sha256(b"physics").hexdigest(),
                   help="PLACEHOLDER default; re-pin from the runtime physics contract")
    b.add_argument("--min-proposals", type=int, default=256)
    b.add_argument("--min-safe-closed", type=int, default=256)
    b.add_argument("--target-failure-rate", type=float, default=0.10)
    b.add_argument("--failure-band-half-width", type=float, default=0.025)
    b.add_argument("--holdout-seed", type=int, default=20260728)
    b.add_argument(
        "--holdout-samples",
        type=int,
        default=768,
        help=(
            "formal heldout samples per action (minimum/default: 768); "
            "smaller canary/diagnostic windows are separate evaluator artifacts"
        ),
    )
    b.add_argument("--holdout-split-id", default="heldout_ball_chingmu73_v1")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="strict-load a manifest and smoke birth+sample")
    v.add_argument("--manifest", required=True)
    v.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    v.add_argument("--assets-root", default=None,
                   help="also verify prototype + motion bytes under this trusted root")
    v.add_argument("--seed", type=int, default=20260728)
    v.add_argument("--n-actions", type=int, default=3)
    v.add_argument("--action-ids", nargs="*", default=None)
    v.add_argument("--samples", type=int, default=100, help="birth+sample rounds at level 1.0")
    v.add_argument("--samples-level0", type=int, default=20,
                   help="extra birth+sample rounds at level 0.0")
    v.add_argument("--domain-epoch", type=int, default=0)
    v.add_argument("--base-yaw-rad", type=float, default=0.0)
    v.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
