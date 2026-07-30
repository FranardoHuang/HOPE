"""Immutable training/export execution-contract helpers.

This module deliberately has no Isaac, Torch, Hydra, or ONNX imports.  It is shared by the
training entry point and both export paths, and its duck-typed runtime extractor is covered by
dependency-light tests.  Schema 3 is the first schema that binds the policy's execution values
(joint/action order, decoder, nominal PD envelope, q-des limits, timing, body/reference order and
the exact actor layout) rather than only task-level configuration.
"""

from __future__ import annotations

import ast
import base64
import binascii
import functools
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from collections.abc import Mapping, MutableMapping


TRAINING_CONTRACT_SCHEMA_VERSION = 3
ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH = 1
ACTOR_LEG_REF_MASK_PROVENANCE_KEY = "actor_leg_ref_mask_provenance_epoch"
ACTOR_LEG_REF_MASK_PROVENANCE_BINDING_KEY = "actor_leg_ref_mask_provenance_sha256"
CHECKPOINT_CONTRACT_SCHEMA_KEY = "training_contract_schema_version"
CHECKPOINT_CONTRACT_SHA_KEY = "training_contract_sha256"
CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY = "training_contract_lineage_exact"
CHECKPOINT_LAUNCH_CLAIM_SHA_KEY = "training_launch_claim_sha256"
ACTION_BALL_TRAINING_KEY = "action_ball_training"
FINITE_PRECLAMP_QDES_PROJECTION_KEY = (
    "finite_preclamp_qdes_projection_enabled"
)
FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY = (
    "finite_projection_soft_envelope_inset_fraction"
)
ACTION_BALL_ACTION_SET_IDENTITY_KEY = "action_set_identity"
ACTION_BALL_DIAGNOSTIC_METADATA_KEY = "action_ball_diagnostic_unauthorized"
FORMAL_EVIDENCE_BOOKABLE_METADATA_KEY = "formal_evidence_bookable"
ACTION_BALL_POLICY_BOOTSTRAP_KIND = (
    "action_ball_shared_ready_actor_bootstrap_v1"
)
ACTION_BALL_DYNAMIC_READY_RUNTIME_BINDING_KIND = (
    "action_ball_dynamic_ready_runtime_binding_v1"
)
ACTION_BALL_DYNAMIC_READY_ARTIFACT_KIND = (
    "agibot_a3_action_dynamic_ready_candidate_v1"
)
ACTION_BALL_DYNAMIC_READY_NOMINAL_HOLD_KIND = (
    "isaac_action_ball_nominal_hold_v1"
)
ACTION_BALL_ACTION_SET_SOURCE_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_action_set_contract.py"
)
ACTION_BALL_ACTION_SET_CONTRACT_KIND = (
    "whole_body_tracking.action_ball.action_set_contract"
)
ACTION_BALL_LAUNCH_CLAIM_KIND = "action_ball_no_clobber_launch_claim_v3"
ACTION_BALL_LAUNCH_CLAIM_SCHEMA_VERSION = 3
_ACTION_BALL_ACTOR_OBS_LAYOUTS = (
    ("action_ball_table_pose_n", 190),
    ("action_ball_n", 181),
)
ACTION_BALL_ACTION_SET_METADATA_KEYS = (
    "action_ball_profile_id",
    "action_ball_expected_n",
    "action_ball_scope",
    "action_ball_mobility_mode",
    "action_ball_action_order",
    "action_ball_ordered_action_uids",
    "action_ball_order_uid_digest_sha256",
    "action_ball_manifest_sha256",
    "action_ball_action_set_contract_sha256",
    "action_ball_action_set_contract_source_sha256",
)
_ACTION_BALL_STALE_DONOR_IDENTITY_KEYS = frozenset(
    {
        *ACTION_BALL_ACTION_SET_METADATA_KEYS,
        # Pre-contract development spellings must never survive a donor copy
        # and be mistaken for the checkpoint-bound names above.
        "action_set_profile_id",
        "action_set_contract_sha256",
        "action_set_contract_source_sha256",
    }
)
_ACTION_BALL_ACTION_SET_CODE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
        "actor_obs_contract",
        "actor_obs_width",
        "namespace_identity",
        "contract_sha256",
    }
)
_ACTION_BALL_ACTION_SET_TRAINING_KEYS = frozenset(
    {
        *_ACTION_BALL_ACTION_SET_CODE_KEYS,
        "contract_source_path",
        "contract_source_sha256",
    }
)
_ACTION_BALL_AUTHORIZATION_KEYS = frozenset(
    {
        "diagnostic_unauthorized",
        "formal_evidence_prohibited",
        "curriculum_promotion_prohibited",
        "exact_export_prohibited",
        "formal_judge_prohibited",
    }
)
_ACTION_BALL_POLICY_BOOTSTRAP_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "action_count",
        "action_order",
        "joint_names",
        "ready_source",
        "decoder",
        "initialization",
        "hard_inner_guard",
    }
)
_ACTION_BALL_POLICY_BOOTSTRAP_READY_KEYS = frozenset(
    {
        "semantics",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "motion_sha256_per_action",
        "shared_ready_joint_pos",
        "shared_ready_joint_pos_sha256",
    }
)
_ACTION_BALL_POLICY_BOOTSTRAP_READY_V2_KEYS = frozenset(
    {
        "semantics",
        "motion_sha256_per_action",
        "physical_ready",
        "identity",
    }
)
_ACTION_BALL_POLICY_BOOTSTRAP_DECODER_KEYS = frozenset(
    {
        "semantics",
        "use_default_offset",
        "default_joint_pos",
        "action_scale",
        "normalized_bias",
        "startup_offset_delta_source",
        "startup_offset_delta_lower",
        "startup_offset_delta_upper",
    }
)
_ACTION_BALL_POLICY_BOOTSTRAP_DECODER_V2_KEYS = frozenset(
    {
        *_ACTION_BALL_POLICY_BOOTSTRAP_DECODER_KEYS,
        "target_joint_pos",
    }
)
_ACTION_BALL_POLICY_BOOTSTRAP_INITIALIZATION_KEYS = frozenset(
    {
        "fresh_only",
        "resume_overwrite_prohibited",
        "output_layer_weight",
        "output_layer_bias",
        "init_noise_std",
        "sigma_envelope",
    }
)
_ACTION_BALL_POLICY_BOOTSTRAP_GUARD_KEYS = frozenset(
    {
        "limit_source",
        "margin_rad",
        "margin_fraction",
        "hard_lower",
        "hard_upper",
        "hard_inner_lower",
        "hard_inner_upper",
    }
)
_ACTION_BALL_DYNAMIC_READY_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "binding_sha256",
        "action_order",
        "motion_sha256_per_action",
        "rows",
    }
)
_ACTION_BALL_DYNAMIC_READY_ROW_KEYS = frozenset(
    {
        "action_id",
        "physical_ready",
        "hold_qdes_joint_pos_rad",
        "normalized_actor_action",
        "artifact",
        "nominal_hold_receipt",
    }
)
_ACTION_BALL_DYNAMIC_READY_PHYSICAL_KEYS = frozenset(
    {
        "root_pos_w_m",
        "root_quat_wxyz",
        "joint_pos_rad",
        "joint_vel_radps",
    }
)
_ACTION_BALL_DYNAMIC_READY_PIN_KEYS = frozenset(
    {"path", "sha256", "content_sha256"}
)
SCHEMA3_TASK_KEYS = (
    "racket_control_point",
    "racket_control_point_offset_wrist_m",
)

PLANNER_TASK_REVISION_KEY = "planner_task_revision"
PLANNER_TASK_REVISION_TRAINING_KEY = "planner_task_revision_training"
_PLANNER_TASK_REVISION_KEYS = frozenset(
    {"enabled", "revision_schema_version", "governor", "initial_tts_range_s"}
)
_PLANNER_GOVERNOR_KEYS = frozenset(
    {"contract_version", "schema_version", "profile_sha256", "profile"}
)
_PLANNER_GOVERNOR_PROFILE_KEYS = frozenset(
    {
        "policy_dt_s",
        "min_tts_s",
        "max_tts_s",
        "max_phase_rate_per_s",
        "max_phase_acceleration_per_s2",
        "max_deadline_revision_delta_s",
        "max_position_revision_delta_m",
        "max_velocity_revision_delta_mps",
        "max_normal_revision_delta_rad",
        "normal_unit_tolerance",
        "early_deadline_tolerance_s",
        "contract_version",
        "schema_version",
    }
)
_PLANNER_TASK_REVISION_TRAINING_KEYS = frozenset(
    {
        "initial_tts_sampling_semantics",
        "initial_tts_mixture",
        "initial_feasibility_gate",
        "dynamics_certified_action_tau_min_bound",
        "timing_exam_semantics",
        "position_std_m",
        "velocity_std_mps",
        "normal_std_rad",
        "tts_std_s",
        "truth_fields_immutable",
        "actor_revision_fields",
    }
)
_PLANNER_INITIAL_TTS_MIXTURE_KEYS = frozenset(
    {"contract_version", "components"}
)
_PLANNER_INITIAL_TTS_COMPONENT_KEYS = frozenset(
    {"name", "range_s", "weight"}
)

# Isaac Lab 2.1 passes ``ImplicitActuatorCfg.friction`` to PhysX as a dimensionless,
# load-dependent joint-friction coefficient.  It is *not* MuJoCo ``frictionloss`` (a constant
# Coulomb torque in N m).  Keep these strings in the immutable contract so a consumer cannot
# silently copy the same-looking numbers between physics backends and call that exact parity.
JOINT_FRICTION_BACKEND = "physx"
JOINT_FRICTION_SEMANTICS = "load_dependent_spatial_force_coefficient"
JOINT_FRICTION_UNITS = "dimensionless"
MOTION_BODY_LIN_VEL_POINTS = ("center_of_mass", "link_origin")

# Ground/terrain plant identity (task.plant ground/terrain keys, 2026-07-22).  人话:地面材质
# 摩擦、机器人 body 材质随机化范围、地形平/不平——这些都是"物理世界长什么样"的语义,变了就
# 是另一套 plant,平地 checkpoint 不能静默续训到粗糙地/滑地上。历史字节默认(下表)必须用
# 【整块缺席】拼写,让所有历史 checkpoint 在 _contract_diff 下逐字节兼容;任何显式偏离默认的
# 值都会让合同长出 ground_plant 键 -> 对旧谱系 resume 直接 fail-loud。
#
# 2026-07-29 抬脚地形修复:rough 的物理形态从"generator 全局地形"(会把 env origins 和克隆
# 桌子拆散,从未产出可用 checkpoint)换成"per-env 零均值凹凸垫,只铺机器人一侧,桌子足迹平在
# z=0"。合同字符串随之换名——两种形态是不同的 plant,不许靠同名静默互认 resume。
GROUND_PLANT_KEY = "ground_plant"
GROUND_PLANT_TERRAIN_PLANE = "plane"
GROUND_PLANT_TERRAIN_ROUGH = "robot_side_zero_mean_patch"
GROUND_PLANT_DEFAULT = {
    "ground_static_friction": 1.0,
    "ground_dynamic_friction": 1.0,
    "robot_material_static_friction_range": [0.3, 1.6],
    "robot_material_dynamic_friction_range": [0.3, 1.2],
    "terrain_type": GROUND_PLANT_TERRAIN_PLANE,
    "terrain_rough_height_range_m": None,
}
_GROUND_PLANT_KEYS = frozenset(
    {
        "schema_version",
        "ground_static_friction",
        "ground_dynamic_friction",
        "robot_material_static_friction_range",
        "robot_material_dynamic_friction_range",
        "terrain_type",
        "terrain_rough_height_range_m",
    }
)
# 2026-07-29 opt-in:逐桶 dynamic=min(static, dynamic)(isaaclab make_consistent)。false 的
# 唯一拼写是【键缺席】(= 历史独立采样,任何已存 6 键块逐字节兼容);true 时块里长出这个键。
GROUND_PLANT_MAKE_CONSISTENT_KEY = "robot_material_make_consistent"

RUNTIME_EXECUTION_KEYS = (
    "articulation_joint_names",
    "action_joint_ids",
    "joint_names",
    "default_joint_pos",
    "action_scale",
    "joint_stiffness",
    "joint_damping",
    "joint_effort_limits",
    "joint_actuator_types",
    "joint_armature",
    "joint_friction_coefficients",
    "joint_velocity_limits",
    "joint_friction_backend",
    "joint_friction_semantics",
    "joint_friction_units",
    "qdes_joint_pos_limits",
    "action_use_default_offset",
    "qdes_clamp",
    "physics_step_dt_s",
    "policy_step_dt_s",
    "control_decimation",
    "actor_obs_contract",
    "actor_obs_mode",
    "actor_obs_total_dim",
    "actor_obs_term_names",
    "actor_obs_term_dims",
    "observation_history_lengths",
    "articulation_body_names",
    "body_names",
    "body_indices",
    "anchor_body_name",
    "anchor_body_index",
    "motion_segment_lengths",
    "motion_clip_fps",
    "motion_kinematics_contracts",
    "motion_kinematics_exact",
)


def _require_exact_mapping_keys(value: object, expected: frozenset[str], *, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")
    return value


def _canonical_action_ball_json(value: object) -> str:
    """Canonical JSON shared by the launch claim, contract, and ONNX metadata."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("action-ball action-set identity is not canonical JSON data") from exc


def _action_ball_canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_action_ball_json(value).encode("utf-8")).hexdigest()


def _action_ball_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _action_ball_repo_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"{name} must be a normalized repo-relative POSIX path")
    return value


def _action_ball_order_uid_digest(
    ordered_action_ids: list[str], ordered_action_uids: list[int]
) -> str:
    rows = [
        {"index": index, "action_id": action_id, "action_uid": action_uid}
        for index, (action_id, action_uid) in enumerate(
            zip(ordered_action_ids, ordered_action_uids)
        )
    ]
    return _action_ball_canonical_sha256(
        {"schema_version": 1, "ordered_actions": rows}
    )


def _parse_action_ball_actor_obs_contract(
    value: object,
) -> tuple[int, int] | None:
    """Return ``(N, width)`` for either supported ActionBall actor layout.

    The legacy layout remains readable for existing checkpoints and claims.
    New table-pose layouts add nine base-pose scalars, so their width is
    ``190 + N`` instead of ``181 + N``.  Both spellings bind the exact action
    count; leading-zero or out-of-range suffixes are deliberately rejected.
    """

    if type(value) is not str:
        return None
    for prefix, base_width in _ACTION_BALL_ACTOR_OBS_LAYOUTS:
        if not value.startswith(prefix):
            continue
        suffix = value[len(prefix) :]
        if (
            not suffix.isdigit()
            or suffix.startswith("0")
        ):
            return None
        action_count = int(suffix)
        if not 1 <= action_count <= 1024:
            return None
        return action_count, base_width + action_count
    return None


def _action_ball_literal_assignment(source: bytes, variable: str) -> object:
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("action-set contract source is not valid UTF-8 Python") from exc
    values = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == variable
            for target in targets
        ):
            values.append(node.value)
    if len(values) != 1:
        raise ValueError(
            f"action-set contract source requires one {variable} assignment"
        )
    try:
        return ast.literal_eval(values[0])
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{variable} must be a Python literal") from exc


def validate_action_ball_action_set_identity_block(value: object) -> dict:
    """Validate the code-owned action-set identity embedded in a training contract.

    ``contract_sha256`` binds exactly the row returned by
    ``action_ball_action_set_contract.py``.  The source-path/SHA pair is kept
    outside that row digest because it binds the registry implementation
    itself.  Both are mandatory for a formal training contract.
    """

    row = _require_exact_mapping_keys(
        value,
        _ACTION_BALL_ACTION_SET_TRAINING_KEYS,
        name="schema-3 action_ball_training.action_set_identity",
    )
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise ValueError("action-set identity schema_version must be integer 1")
    if row["kind"] != ACTION_BALL_ACTION_SET_CONTRACT_KIND:
        raise ValueError("action-set identity kind is unsupported")
    profile_id = row["profile_id"]
    experiment_name = row["experiment_name"]
    if (
        type(profile_id) is not str
        or not profile_id
        or profile_id.strip() != profile_id
        or type(experiment_name) is not str
        or not experiment_name
        or experiment_name.strip() != experiment_name
    ):
        raise ValueError("action-set profile_id/experiment_name must be non-empty strings")
    expected_n = row["expected_n"]
    if (
        type(expected_n) is not int
        or isinstance(expected_n, bool)
        or not 1 <= expected_n <= 1024
    ):
        raise ValueError("action-set expected_n must be a plain integer in [1,1024]")
    action_ids = row["ordered_action_ids"]
    action_uids = row["ordered_action_uids"]
    if (
        type(action_ids) is not list
        or len(action_ids) != expected_n
        or any(type(item) is not str or not item for item in action_ids)
        or len(set(action_ids)) != expected_n
    ):
        raise ValueError("action-set ordered_action_ids must be exact unique N")
    if (
        type(action_uids) is not list
        or len(action_uids) != expected_n
        or any(
            type(item) is not int or isinstance(item, bool) or item < 0
            for item in action_uids
        )
        or len(set(action_uids)) != expected_n
    ):
        raise ValueError("action-set ordered_action_uids must be exact unique N")
    if row["scope"] not in ("upper", "full"):
        raise ValueError("action-set scope must be upper/full")
    if row["mobility_mode"] not in ("no_move", "move"):
        raise ValueError("action-set mobility_mode must be no_move/move")
    digest = _action_ball_sha256(
        row["order_uid_digest_sha256"],
        name="action-set order_uid_digest_sha256",
    )
    if digest != _action_ball_order_uid_digest(action_ids, action_uids):
        raise ValueError("action-set order_uid_digest_sha256 does not bind ID/UID order")
    _action_ball_repo_relative_path(row["manifest_path"], name="action-set manifest_path")
    _action_ball_sha256(row["manifest_sha256"], name="action-set manifest_sha256")
    actor_layout = _parse_action_ball_actor_obs_contract(
        row["actor_obs_contract"]
    )
    if actor_layout is None or actor_layout[0] != expected_n:
        raise ValueError(
            "action-set actor_obs_contract must be a supported ActionBall "
            f"layout with exact N={expected_n}"
        )
    expected_actor_width = actor_layout[1]
    if (
        type(row["actor_obs_width"]) is not int
        or isinstance(row["actor_obs_width"], bool)
        or row["actor_obs_width"] != expected_actor_width
    ):
        raise ValueError(
            "action-set actor_obs_width does not match its ActionBall layout"
        )
    if row["namespace_identity"] != f"n{expected_n}-{digest[:12]}":
        raise ValueError("action-set namespace_identity does not bind N/order")
    contract_sha = _action_ball_sha256(
        row["contract_sha256"], name="action-set contract_sha256"
    )
    code_row = {
        key: row[key]
        for key in _ACTION_BALL_ACTION_SET_CODE_KEYS
        if key != "contract_sha256"
    }
    if _action_ball_canonical_sha256(code_row) != contract_sha:
        raise ValueError("action-set contract_sha256 does not bind the code-owned row")
    if row["contract_source_path"] != ACTION_BALL_ACTION_SET_SOURCE_PATH:
        raise ValueError("action-set contract_source_path is not the code-owned registry")
    _action_ball_sha256(
        row["contract_source_sha256"],
        name="action-set contract_source_sha256",
    )
    # Return a JSON-only copy so no mutable Mapping subclass can change after validation.
    return json.loads(_canonical_action_ball_json(dict(row)))


def validate_action_ball_action_set_runtime_identity(
    identity: object,
    *,
    actor_obs_contract: object,
    actor_obs_width: object,
    manifest_path: object,
    manifest_sha256: object,
    scope: object,
    mobility_mode: object,
    ordered_action_ids: object,
    ordered_action_uids: object,
    experiment_name: object | None = None,
) -> dict:
    """Cross-check one claim-bound identity against an instantiated runtime view."""

    row = validate_action_ball_action_set_identity_block(identity)
    live_ids = list(ordered_action_ids) if isinstance(ordered_action_ids, (list, tuple)) else None
    live_uids = list(ordered_action_uids) if isinstance(ordered_action_uids, (list, tuple)) else None
    comparisons = {
        "actor_obs_contract": (row["actor_obs_contract"], actor_obs_contract),
        "actor_obs_width": (row["actor_obs_width"], actor_obs_width),
        "manifest_path": (row["manifest_path"], manifest_path),
        "manifest_sha256": (row["manifest_sha256"], manifest_sha256),
        "scope": (row["scope"], scope),
        "mobility_mode": (row["mobility_mode"], mobility_mode),
        "ordered_action_ids": (row["ordered_action_ids"], live_ids),
        "ordered_action_uids": (row["ordered_action_uids"], live_uids),
    }
    if experiment_name is not None:
        comparisons["experiment_name"] = (row["experiment_name"], experiment_name)
    mismatch = {
        key: {"claim": expected, "runtime": actual}
        for key, (expected, actual) in comparisons.items()
        if expected != actual
    }
    if mismatch:
        raise ValueError(
            "action-set launch claim disagrees with live runtime: "
            + _canonical_action_ball_json(mismatch)
        )
    return row


def load_action_ball_action_set_identity_from_launch_claim(
    path: str | Path,
    *,
    expected_claim_sha256: str,
    actual_argv: object,
) -> dict:
    """Load one launcher claim and return its code/source-bound action identity.

    This consumer deliberately does not reconstruct identity from Hydra or the
    manifest.  It checks the exact claim digest, the code-owned registry-row
    digest, the registry source SHA in the launcher's runtime source map, and
    the source bytes below the claim's exact checkout.
    """

    expected_sha = _action_ball_sha256(
        expected_claim_sha256, name="training launch-claim SHA-256"
    )
    claim_path = Path(path)
    try:
        raw = claim_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read action-ball launch claim: {exc}") from exc

    def no_duplicate_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number {token!r}")
        return value

    def reject_constant(token):
        raise ValueError(f"non-finite JSON constant {token!r}")

    try:
        claim = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "action-ball launch claim must be strict UTF-8 JSON without duplicate keys"
        ) from exc
    claim = _require_exact_mapping_keys(
        claim,
        frozenset(
            {
                "schema_version",
                "kind",
                "launch_claim_sha256",
                "canonical_payload",
                "argv",
                "confirmation_claim_sha256",
            }
        ),
        name="action-ball launch claim",
    )
    if (
        type(claim["schema_version"]) is not int
        or claim["schema_version"] != ACTION_BALL_LAUNCH_CLAIM_SCHEMA_VERSION
        or claim["kind"] != ACTION_BALL_LAUNCH_CLAIM_KIND
    ):
        raise ValueError("action-ball launch claim schema/kind is unsupported")
    if (
        claim["launch_claim_sha256"] != expected_sha
        or claim["confirmation_claim_sha256"] != expected_sha
        or _action_ball_canonical_sha256(claim["canonical_payload"]) != expected_sha
    ):
        raise ValueError("action-ball launch claim digest/confirmation mismatch")
    payload = claim["canonical_payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("action-ball launch claim payload must be an object")
    argv = claim["argv"]
    if (
        type(argv) is not list
        or len(argv) != 10
        or any(type(item) is not str for item in argv)
    ):
        raise ValueError("action-ball launch claim argv is not an exact no-site envelope")
    if (
        type(actual_argv) not in (list, tuple)
        or any(type(item) is not str for item in actual_argv)
        or list(actual_argv) != argv
    ):
        raise ValueError(
            "action-ball launch claim argv differs from the actual kernel argv"
        )
    if argv[1:5] != ["-I", "-B", "-S", "-c"] or not argv[5]:
        raise ValueError("action-ball launch claim no-site flags are not exact")
    try:
        contract_raw = base64.b64decode(
            argv[9].encode("ascii"), validate=True
        )
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise ValueError(
            "action-ball launch claim no-site contract is not canonical base64"
        ) from exc
    if (
        hashlib.sha256(contract_raw).hexdigest() != argv[8]
        or base64.b64encode(contract_raw).decode("ascii") != argv[9]
    ):
        raise ValueError(
            "action-ball launch claim no-site contract digest/base64 differs"
        )
    try:
        no_site_contract = json.loads(
            contract_raw.decode("utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "action-ball launch claim no-site contract must be strict JSON"
        ) from exc
    no_site_contract = _require_exact_mapping_keys(
        no_site_contract,
        frozenset(
            {
                "schema_version",
                "kind",
                "bootstrap",
                "entrypoint",
                "import_roots",
                "entrypoint_argv",
            }
        ),
        name="action-ball no-site argv contract",
    )
    entrypoint_argv = no_site_contract["entrypoint_argv"]
    if (
        no_site_contract["schema_version"] != 1
        or no_site_contract["kind"]
        != "action_ball_python_nosite_argv_contract_v1"
        or type(entrypoint_argv) is not list
        or not entrypoint_argv
        or any(type(item) is not str for item in entrypoint_argv)
        or entrypoint_argv[-1]
        != f"++training_launch_claim_sha256={expected_sha}"
        or _canonical_action_ball_json(no_site_contract).encode("utf-8")
        != contract_raw
    ):
        raise ValueError("action-ball launch claim argv is not exactly self-bound")
    base_argv = payload.get("argv_without_launch_claim")
    if (
        type(base_argv) is not list
        or len(base_argv) != 10
        or any(type(item) is not str for item in base_argv)
        or argv[:8] != base_argv[:8]
    ):
        raise ValueError(
            "action-ball launch claim argv is not derived from its "
            "claim-bound base no-site envelope"
        )
    try:
        base_raw = base64.b64decode(
            base_argv[9].encode("ascii"), validate=True
        )
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise ValueError(
            "action-ball base no-site contract is not canonical base64"
        ) from exc
    if (
        hashlib.sha256(base_raw).hexdigest() != base_argv[8]
        or base64.b64encode(base_raw).decode("ascii") != base_argv[9]
    ):
        raise ValueError("action-ball base no-site contract digest/base64 differs")
    try:
        base_contract = json.loads(
            base_raw.decode("utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "action-ball base no-site contract must be strict JSON"
        ) from exc
    base_contract = _require_exact_mapping_keys(
        base_contract,
        frozenset(
            {
                "schema_version",
                "kind",
                "bootstrap",
                "entrypoint",
                "import_roots",
                "entrypoint_argv",
            }
        ),
        name="action-ball base no-site argv contract",
    )
    if (
        _canonical_action_ball_json(base_contract).encode("utf-8") != base_raw
        or base_contract["schema_version"] != 1
        or base_contract["kind"]
        != "action_ball_python_nosite_argv_contract_v1"
        or {
            key: no_site_contract[key]
            for key in no_site_contract
            if key != "entrypoint_argv"
        }
        != {
            key: base_contract[key]
            for key in base_contract
            if key != "entrypoint_argv"
        }
        or entrypoint_argv
        != [
            *base_contract["entrypoint_argv"],
            f"++training_launch_claim_sha256={expected_sha}",
        ]
    ):
        raise ValueError(
            "action-ball claim no-site contract is not the unique "
            "claim-token extension of its bound base contract"
        )
    isolated_entrypoint = payload.get("isolated_training_entrypoint")
    if (
        not isinstance(isolated_entrypoint, Mapping)
        or isolated_entrypoint.get("nosite_argv_contract") != base_contract
        or isolated_entrypoint.get("nosite_argv_contract_sha256")
        != base_argv[8]
    ):
        raise ValueError(
            "action-ball base no-site contract differs from the isolated "
            "training entrypoint identity"
        )
    code_identity = _require_exact_mapping_keys(
        payload.get("action_set_contract"),
        _ACTION_BALL_ACTION_SET_CODE_KEYS,
        name="launch claim action_set_contract",
    )
    runtime_sources = payload.get("runtime_code_sha256")
    if not isinstance(runtime_sources, Mapping):
        raise ValueError("launch claim runtime_code_sha256 must be an object")
    source_sha = _action_ball_sha256(
        runtime_sources.get(ACTION_BALL_ACTION_SET_SOURCE_PATH),
        name="launch claim action-set contract source SHA-256",
    )
    training_identity = {
        **dict(code_identity),
        "contract_source_path": ACTION_BALL_ACTION_SET_SOURCE_PATH,
        "contract_source_sha256": source_sha,
    }
    training_identity = validate_action_ball_action_set_identity_block(training_identity)
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("launch claim manifest must be an object")
    payload_checks = {
        "launch_profile": (training_identity["profile_id"], payload.get("launch_profile")),
        "ordered_action_ids": (
            training_identity["ordered_action_ids"],
            payload.get("ordered_action_ids"),
        ),
        "manifest.path": (training_identity["manifest_path"], manifest.get("path")),
        "manifest.sha256": (
            training_identity["manifest_sha256"],
            manifest.get("sha256"),
        ),
    }
    payload_mismatch = {
        key: {"action_set": expected, "payload": actual}
        for key, (expected, actual) in payload_checks.items()
        if expected != actual
    }
    if payload_mismatch:
        raise ValueError(
            "launch claim duplicates disagree with action-set identity: "
            + _canonical_action_ball_json(payload_mismatch)
        )
    checkout_raw = payload.get("source_checkout")
    if type(checkout_raw) is not str or not checkout_raw:
        raise ValueError("launch claim source_checkout must be a non-empty path")
    checkout = Path(checkout_raw)
    if not checkout.is_absolute():
        raise ValueError("launch claim source_checkout must be absolute")
    source_path = checkout.joinpath(
        *PurePosixPath(ACTION_BALL_ACTION_SET_SOURCE_PATH).parts
    )
    try:
        checkout_resolved = checkout.resolve(strict=True)
        source_resolved = source_path.resolve(strict=True)
        source_resolved.relative_to(checkout_resolved)
        if source_path.is_symlink() or not source_resolved.is_file():
            raise ValueError("action-set contract source must be a regular non-symlink file")
        source_bytes = source_resolved.read_bytes()
        actual_source_sha = hashlib.sha256(source_bytes).hexdigest()
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot verify action-set contract source bytes: {exc}") from exc
    if actual_source_sha != source_sha:
        raise ValueError(
            "action-set contract source bytes differ from launch claim: "
            f"claim={source_sha}, actual={actual_source_sha}"
        )
    registry = _action_ball_literal_assignment(
        source_bytes, "ACTION_SET_CONTRACTS"
    )
    if not isinstance(registry, dict):
        raise ValueError("ACTION_SET_CONTRACTS must be a literal dict")
    literal_row = registry.get(training_identity["profile_id"])
    literal_keys = (
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
    )
    expected_literal_row = {
        key: training_identity[key] for key in literal_keys
    }
    if literal_row != expected_literal_row:
        raise ValueError(
            "launch claim action-set identity is not the exact literal "
            "registered in its code-owned source"
        )
    policies = _action_ball_literal_assignment(
        source_bytes, "ACTION_SET_PROFILE_POLICIES"
    )
    if not isinstance(policies, dict):
        raise ValueError("ACTION_SET_PROFILE_POLICIES must be a literal dict")
    policy = policies.get(training_identity["profile_id"])
    policy_keys = {
        "expected_n",
        "scope",
        "mobility_mode",
        "required_action_ids",
        "retired_action_ids",
    }
    if not isinstance(policy, dict) or set(policy) != policy_keys:
        raise ValueError(
            "launch claim profile is missing its exact code-owned policy"
        )
    required = policy["required_action_ids"]
    retired = policy["retired_action_ids"]
    if (
        policy["expected_n"] != training_identity["expected_n"]
        or policy["scope"] != training_identity["scope"]
        or policy["mobility_mode"] != training_identity["mobility_mode"]
        or type(required) is not list
        or required != training_identity["ordered_action_ids"]
        or type(retired) is not list
        or any(type(item) is not str or not item for item in retired)
        or len(retired) != len(set(retired))
        or set(required).intersection(retired)
    ):
        raise ValueError(
            "launch claim action-set identity violates its code-owned "
            "profile policy"
        )
    return training_identity


def action_ball_action_set_identity(contract: Mapping | None) -> dict | None:
    """Return the validated formal action-set identity, if the contract has one."""

    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("training contract root must be an object")
    action_ball = contract.get(ACTION_BALL_TRAINING_KEY)
    if action_ball is None:
        return None
    if not isinstance(action_ball, Mapping):
        raise ValueError("schema-3 action_ball_training must be an object")
    identity = action_ball.get(ACTION_BALL_ACTION_SET_IDENTITY_KEY)
    if identity is None:
        return None
    return validate_action_ball_action_set_identity_block(identity)


def bind_action_ball_action_set_metadata(
    metadata: MutableMapping[str, str],
    contract: Mapping | None,
    *,
    lineage_exact: bool,
) -> bool:
    """Replace all donor action-set labels from the exact checkpoint contract.

    A diagnostic or non-exact lineage receives no formal action-set metadata.
    Stale donor keys are always removed first, including for non-ActionBall
    checkpoints, so a carrier graph cannot launder another policy's identity.
    """

    if type(lineage_exact) is not bool:
        raise ValueError("action-ball export lineage_exact must be an exact boolean")
    for key in _ACTION_BALL_STALE_DONOR_IDENTITY_KEYS:
        metadata.pop(key, None)
    if contract is None:
        return False
    diagnostic = validate_action_ball_training_authorization(contract)
    identity = action_ball_action_set_identity(contract)
    if ACTION_BALL_TRAINING_KEY not in contract:
        return False
    if diagnostic or not lineage_exact:
        return False
    if identity is None:
        raise ValueError(
            "formal exact ActionBall export is missing action_set_identity"
        )
    metadata.update(
        {
            "action_ball_profile_id": identity["profile_id"],
            "action_ball_expected_n": str(identity["expected_n"]),
            "action_ball_scope": identity["scope"],
            "action_ball_mobility_mode": identity["mobility_mode"],
            "action_ball_action_order": _canonical_action_ball_json(
                identity["ordered_action_ids"]
            ),
            "action_ball_ordered_action_uids": _canonical_action_ball_json(
                identity["ordered_action_uids"]
            ),
            "action_ball_order_uid_digest_sha256": identity[
                "order_uid_digest_sha256"
            ],
            "action_ball_manifest_sha256": identity["manifest_sha256"],
            "action_ball_action_set_contract_sha256": identity[
                "contract_sha256"
            ],
            "action_ball_action_set_contract_source_sha256": identity[
                "contract_source_sha256"
            ],
        }
    )
    return True


def _planner_finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _canonical_planner_json(value: Mapping, *, trailing_newline: bool) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("planner task revision contract is not canonical JSON data") from exc
    return encoded + ("\n" if trailing_newline else "")


def _validate_planner_initial_tts_mixture(
    value: object, *, support_lo: float, support_hi: float
) -> None:
    """Validate the complete training-only mixture without leaking it into ONNX metadata."""

    mixture = _require_exact_mapping_keys(
        value,
        _PLANNER_INITIAL_TTS_MIXTURE_KEYS,
        name="planner_task_revision_training.initial_tts_mixture",
    )
    if mixture["contract_version"] != "initial_tts_mixture_v1":
        raise ValueError("planner initial-TTS mixture contract version is unsupported")
    components = mixture["components"]
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("planner initial-TTS mixture components must be a non-empty list")
    names: set[str] = set()
    total_weight = 0.0
    component_lows: list[float] = []
    component_highs: list[float] = []
    has_exact_0p5 = False
    has_sub_0p5_stress = False
    for index, raw_component in enumerate(components):
        component = _require_exact_mapping_keys(
            raw_component,
            _PLANNER_INITIAL_TTS_COMPONENT_KEYS,
            name=f"planner initial-TTS mixture component {index}",
        )
        name = component["name"]
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError(
                f"planner initial-TTS mixture component {index} name is invalid"
            )
        if name in names:
            raise ValueError("planner initial-TTS mixture component names must be unique")
        names.add(name)
        interval = component["range_s"]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(
                f"planner initial-TTS mixture component {index} range_s must contain two values"
            )
        lo = _planner_finite_number(
            interval[0], name=f"planner initial-TTS component {index} lo_s", minimum=0.0
        )
        hi = _planner_finite_number(
            interval[1], name=f"planner initial-TTS component {index} hi_s", minimum=lo
        )
        weight = _planner_finite_number(
            component["weight"],
            name=f"planner initial-TTS component {index} weight",
            strictly_positive=True,
        )
        total_weight += weight
        component_lows.append(lo)
        component_highs.append(hi)
        has_exact_0p5 |= lo == 0.5 and hi == 0.5
        has_sub_0p5_stress |= lo < 0.5
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("planner initial-TTS mixture weights must sum to 1")
    if min(component_lows) != support_lo or max(component_highs) != support_hi:
        raise ValueError(
            "planner initial-TTS mixture support must equal initial_tts_range_s"
        )
    if not has_exact_0p5:
        raise ValueError("planner initial-TTS mixture requires an exact 0.5 s point mass")
    if not has_sub_0p5_stress:
        raise ValueError("planner initial-TTS mixture requires a sub-0.5 s stress stratum")


def planner_task_revision_metadata(contract: Mapping) -> str | None:
    """Return the sole C++ metadata JSON for a checkpoint-bound revision profile.

    Absence of both revision keys is the byte-compatible legacy/OFF spelling and returns ``None``.
    Any partial spelling fails closed.  The returned JSON is reconstructed only from the immutable
    schema-3 sidecar; callers must never fill it from the current export environment or an ONNX
    donor.  ``clip_seg_lengths`` and ``clip_strike_phases`` remain separate established metadata
    keys, but are validated here because the C++ phase governor maps normalized phase to the strike
    frame derived from that exact pair.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("training contract root must be an object")
    revision_present = PLANNER_TASK_REVISION_KEY in contract
    training_present = PLANNER_TASK_REVISION_TRAINING_KEY in contract
    if not revision_present and not training_present:
        return None
    if revision_present != training_present:
        raise ValueError(
            "schema-3 planner task revision is half-configured: planner_task_revision and "
            "planner_task_revision_training must either both be present or both be absent"
        )
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 3:
        raise ValueError("planner task revision metadata requires schema-3 training contract")

    revision = _require_exact_mapping_keys(
        contract[PLANNER_TASK_REVISION_KEY],
        _PLANNER_TASK_REVISION_KEYS,
        name="planner_task_revision",
    )
    if revision["enabled"] is not True:
        raise ValueError("planner_task_revision.enabled must be true when the block is present")
    if (
        type(revision["revision_schema_version"]) is not int
        or revision["revision_schema_version"] != 1
    ):
        raise ValueError("planner_task_revision.revision_schema_version must be integer 1")

    governor = _require_exact_mapping_keys(
        revision["governor"],
        _PLANNER_GOVERNOR_KEYS,
        name="planner_task_revision.governor",
    )
    profile = _require_exact_mapping_keys(
        governor["profile"],
        _PLANNER_GOVERNOR_PROFILE_KEYS,
        name="planner_task_revision.governor.profile",
    )
    if governor["contract_version"] != "phase_governor_v1" or profile[
        "contract_version"
    ] != "phase_governor_v1":
        raise ValueError("planner task revision requires phase_governor_v1")
    if (
        type(governor["schema_version"]) is not int
        or governor["schema_version"] != 1
        or type(profile["schema_version"]) is not int
        or profile["schema_version"] != 1
    ):
        raise ValueError("planner task revision governor schema_version must be integer 1")

    profile_sha = governor["profile_sha256"]
    if (
        type(profile_sha) is not str
        or len(profile_sha) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha)
    ):
        raise ValueError("planner task revision profile_sha256 must be 64 lowercase hex characters")
    canonical_profile = _canonical_planner_json(profile, trailing_newline=True)
    calculated_profile_sha = hashlib.sha256(canonical_profile.encode("utf-8")).hexdigest()
    if profile_sha != calculated_profile_sha:
        raise ValueError("planner task revision profile_sha256 does not bind the canonical profile")

    policy_dt = _planner_finite_number(
        profile["policy_dt_s"], name="planner profile policy_dt_s", strictly_positive=True
    )
    min_tts = _planner_finite_number(
        profile["min_tts_s"], name="planner profile min_tts_s", strictly_positive=True
    )
    max_tts = _planner_finite_number(
        profile["max_tts_s"], name="planner profile max_tts_s", strictly_positive=True
    )
    if max_tts <= min_tts:
        raise ValueError("planner profile max_tts_s must be greater than min_tts_s")
    _planner_finite_number(
        profile["max_phase_rate_per_s"],
        name="planner profile max_phase_rate_per_s",
        strictly_positive=True,
    )
    _planner_finite_number(
        profile["max_phase_acceleration_per_s2"],
        name="planner profile max_phase_acceleration_per_s2",
        strictly_positive=True,
    )
    for key in (
        "max_deadline_revision_delta_s",
        "max_position_revision_delta_m",
        "max_velocity_revision_delta_mps",
        "early_deadline_tolerance_s",
    ):
        _planner_finite_number(
            profile[key], name=f"planner profile {key}", minimum=0.0
        )
    max_normal_delta = _planner_finite_number(
        profile["max_normal_revision_delta_rad"],
        name="planner profile max_normal_revision_delta_rad",
        minimum=0.0,
    )
    if max_normal_delta > math.pi:
        raise ValueError("planner profile max_normal_revision_delta_rad must be <= pi")
    normal_tolerance = _planner_finite_number(
        profile["normal_unit_tolerance"],
        name="planner profile normal_unit_tolerance",
        minimum=0.0,
    )
    if normal_tolerance >= 1.0:
        raise ValueError("planner profile normal_unit_tolerance must be < 1")

    raw_initial_tts = revision["initial_tts_range_s"]
    if not isinstance(raw_initial_tts, (list, tuple)) or len(raw_initial_tts) != 2:
        raise ValueError("planner_task_revision.initial_tts_range_s must contain two values")
    initial_lo = _planner_finite_number(
        raw_initial_tts[0], name="planner initial_tts_range_s[0]"
    )
    initial_hi = _planner_finite_number(
        raw_initial_tts[1], name="planner initial_tts_range_s[1]"
    )
    if initial_lo < min_tts or initial_hi > max_tts or initial_hi <= initial_lo:
        raise ValueError(
            "planner_task_revision.initial_tts_range_s must be strictly ordered inside the "
            "governor TTS envelope"
        )
    contract_policy_dt = _planner_finite_number(
        contract.get("policy_step_dt_s"),
        name="schema-3 policy_step_dt_s",
        strictly_positive=True,
    )
    if policy_dt != contract_policy_dt:
        raise ValueError(
            "planner profile policy_dt_s disagrees with schema-3 policy_step_dt_s"
        )

    segment_lengths = contract.get("motion_segment_lengths")
    strike_phases = contract.get("strike_phase_per_clip")
    if (
        not isinstance(segment_lengths, (list, tuple))
        or not segment_lengths
        or not isinstance(strike_phases, (list, tuple))
        or len(strike_phases) != len(segment_lengths)
    ):
        raise ValueError(
            "planner task revision requires one checkpoint-bound strike phase per motion segment"
        )
    # Plain zip is safe here: the check above already rejected unequal lengths, and
    # zip(strict=True) would not import on the Python 3.8 host interpreter.
    for index, (raw_length, raw_phase) in enumerate(
        zip(segment_lengths, strike_phases)
    ):
        if isinstance(raw_length, bool) or type(raw_length) is not int or raw_length <= 0:
            raise ValueError(f"planner clip {index} segment length must be a positive integer")
        phase = _planner_finite_number(
            raw_phase, name=f"planner clip {index} strike phase"
        )
        if phase < 0.0 or phase > 1.0:
            raise ValueError(f"planner clip {index} strike phase must be in [0, 1]")

    training = _require_exact_mapping_keys(
        contract[PLANNER_TASK_REVISION_TRAINING_KEY],
        _PLANNER_TASK_REVISION_TRAINING_KEYS,
        name="planner_task_revision_training",
    )
    if training["initial_tts_sampling_semantics"] != (
        "explicit_weighted_mixture_over_initial_tts_range_s"
    ):
        raise ValueError("planner revision initial TTS sampling semantics are unsupported")
    _validate_planner_initial_tts_mixture(
        training["initial_tts_mixture"],
        support_lo=initial_lo,
        support_hi=initial_hi,
    )
    if training["initial_feasibility_gate"] != (
        "normalized_phase_rate_and_acceleration_envelope_only"
    ):
        raise ValueError("planner revision feasibility-gate semantics are unsupported")
    if type(training["dynamics_certified_action_tau_min_bound"]) is not bool:
        raise ValueError("planner revision dynamics certification flag must be boolean")
    if training["timing_exam_semantics"] != {
        "0.5_s": "required_baseline_gate",
        "below_0.5_s": "stress_diagnostic_not_support_floor",
    }:
        raise ValueError("planner revision timing-exam semantics are unsupported")
    for key in ("position_std_m", "velocity_std_mps", "normal_std_rad", "tts_std_s"):
        _planner_finite_number(
            training[key], name=f"planner_task_revision_training.{key}", minimum=0.0
        )
    if training["truth_fields_immutable"] != [
        "question_bank_row",
        "physical_ball",
        "reward_target",
        "critic_target",
    ]:
        raise ValueError("planner revision immutable truth field list is invalid")
    if training["actor_revision_fields"] != [
        "target_position",
        "target_velocity",
        "signed_target_normal",
        "time_to_strike",
    ]:
        raise ValueError("planner revision actor field list is invalid")

    # nlohmann::json uses a sorted object map and compact dump for the C++ parser.  This exact
    # spelling keeps exporter outputs deterministic and makes the embedded profile SHA replayable.
    return _canonical_planner_json(revision, trailing_newline=False)


def bind_planner_task_revision_metadata(
    metadata: MutableMapping[str, str], contract: Mapping | None
) -> str | None:
    """Replace any donor/runtime claim with the checkpoint-side revision contract.

    This is intentionally destructive before validation: callers write artifacts only after all
    checks pass, so a failed contract cannot leave a stale donor claim available for later code to
    copy.  ``None`` or an OFF/legacy contract always removes the metadata key.
    """

    metadata.pop(PLANNER_TASK_REVISION_KEY, None)
    if contract is None:
        return None
    encoded = planner_task_revision_metadata(contract)
    if encoded is not None:
        metadata[PLANNER_TASK_REVISION_KEY] = encoded
    return encoded


def validate_training_launch_claim_sha256(value: str) -> str:
    """Validate the immutable launcher claim embedded in checkpoint ``infos``.

    Launch claims are operational provenance rather than part of the scientific hard contract.
    Their spelling stays deliberately strict so whitespace or case normalization cannot make two
    distinct atomic claims look identical after the process starts.
    """

    if type(value) is not str or len(value) != 64 or any(
        ch not in "0123456789abcdef" for ch in value
    ):
        raise ValueError("training_launch_claim_sha256 must be 64 lowercase hex characters")
    return value


def actor_leg_ref_mask_provenance_required(contract: Mapping) -> bool:
    """Return whether the actor has the 62-D command term whose leg semantics may be masked."""

    names = contract.get("actor_obs_term_names")
    dims = contract.get("actor_obs_term_dims")
    if not isinstance(names, (list, tuple)) or not isinstance(dims, (list, tuple)):
        return False
    return any(
        str(name) == "command" and type(dim) is int and dim == 62
        for name, dim in zip(names, dims)
    )


def require_actor_leg_ref_mask_provenance(contract: Mapping) -> None:
    """Reject command-bearing contracts from before explicit mask/unmasked provenance existed."""

    if not actor_leg_ref_mask_provenance_required(contract):
        if contract.get("actor_leg_ref_mask") is True:
            raise ValueError("actor_leg_ref_mask is invalid without a 62-D command term")
        return
    epoch = contract.get(ACTOR_LEG_REF_MASK_PROVENANCE_KEY)
    if isinstance(epoch, bool) or type(epoch) is not int or epoch != ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH:
        raise ValueError(
            "command-bearing training contract lacks actor_leg_ref_mask_provenance_epoch=1; "
            "pre-epoch masked and unmasked checkpoints are indistinguishable"
        )


def actor_leg_ref_mask_provenance_payload(
    *, training_contract_sha256: str, source_checkpoint_sha256: str, masked: bool
) -> str:
    """Build the stable payload binding mask semantics to one contract/checkpoint pair."""

    contract_sha = str(training_contract_sha256).strip()
    checkpoint_sha = str(source_checkpoint_sha256).strip()
    for name, value in (
        ("training_contract_sha256", contract_sha),
        ("source_checkpoint_sha256", checkpoint_sha),
    ):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{name} must be 64 lowercase hex characters")
    if type(masked) is not bool:
        raise ValueError("actor_leg_ref_mask provenance masked value must be boolean")
    return (
        f"actor_leg_ref_mask_provenance_epoch={ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH}\n"
        f"actor_leg_ref_mask={1 if masked else 0}\n"
        f"training_contract_sha256={contract_sha}\n"
        f"source_checkpoint_sha256={checkpoint_sha}\n"
    )


def actor_leg_ref_mask_provenance_sha256(
    *, training_contract_sha256: str, source_checkpoint_sha256: str, masked: bool
) -> str:
    payload = actor_leg_ref_mask_provenance_payload(
        training_contract_sha256=training_contract_sha256,
        source_checkpoint_sha256=source_checkpoint_sha256,
        masked=masked,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def bind_actor_leg_ref_mask_metadata(
    metadata: MutableMapping[str, str], contract: Mapping | None
) -> None:
    """Replace donor mask provenance with the checkpoint contract's authoritative value.

    The key is intentionally only-when-true, but absence means unmasked only beside epoch 1.
    Clearing first is essential for standalone exports, where a donor may otherwise contaminate
    a checkpoint or launder ambiguous pre-epoch provenance.
    """

    metadata.pop("actor_leg_ref_mask", None)
    metadata.pop(ACTOR_LEG_REF_MASK_PROVENANCE_KEY, None)
    metadata.pop(ACTOR_LEG_REF_MASK_PROVENANCE_BINDING_KEY, None)
    if contract is None or not actor_leg_ref_mask_provenance_required(contract):
        return
    require_actor_leg_ref_mask_provenance(contract)
    masked = contract.get("actor_leg_ref_mask") is True
    metadata[ACTOR_LEG_REF_MASK_PROVENANCE_KEY] = str(ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH)
    metadata[ACTOR_LEG_REF_MASK_PROVENANCE_BINDING_KEY] = (
        actor_leg_ref_mask_provenance_sha256(
            training_contract_sha256=metadata.get("training_contract_sha256", ""),
            source_checkpoint_sha256=metadata.get("source_checkpoint_sha256", ""),
            masked=masked,
        )
    )
    if masked:
        metadata["actor_leg_ref_mask"] = "1"
        # Old consumers ignore unknown mask metadata. Keep the established exactness bit false so
        # they also reject this unsupported observation semantics instead of publishing it.
        metadata["training_contract_exact"] = "0"


def _canonical_actor_leg_ref_mask_callables():
    """Return the two exact command functions allowed to mint epoch-1 provenance."""

    from isaaclab.envs.mdp import generated_commands
    from whole_body_tracking.tasks.tracking.mdp.hope_observations import (
        generated_commands_actor_leg_masked,
    )

    return (
        (generated_commands, False),
        (generated_commands_actor_leg_masked, True),
    )


def _classify_actor_leg_ref_mask_callable(func) -> bool | None:
    for canonical, masked in _canonical_actor_leg_ref_mask_callables():
        if func is canonical:
            return masked
    return None


def resolve_motion_body_lin_vel_points(kinematics_contracts) -> tuple[str, ...]:
    """Resolve the runtime linear-velocity point independently for every motion clip.

    Declared COM/link-origin points are authoritative even for an inexact diagnostic contract.
    The one historical null spelling ``legacy_unbound_assumed_com`` is known to have been loaded
    through Isaac's COM-velocity channel, so it resolves to COM while its separate ``exact=False``
    provenance remains unchanged.  No other null or unknown spelling is safe to guess.
    """

    if not isinstance(kinematics_contracts, (list, tuple)) or not kinematics_contracts:
        raise ValueError("motion_kinematics_contracts must be a non-empty array")
    resolved = []
    for index, item in enumerate(kinematics_contracts):
        if not isinstance(item, Mapping):
            raise ValueError(f"motion kinematics clip {index} must be an object")
        point = item.get("body_lin_vel_point")
        if point in MOTION_BODY_LIN_VEL_POINTS:
            resolved.append(str(point))
            continue
        status = item.get("status")
        if point is None and status == "legacy_unbound_assumed_com":
            resolved.append("center_of_mass")
            continue
        if point is None:
            raise ValueError(
                f"motion kinematics clip {index} has unresolved null body_lin_vel_point "
                f"for status {status!r}"
            )
        raise ValueError(
            f"motion kinematics clip {index} has unknown body_lin_vel_point {point!r}"
        )
    return tuple(resolved)


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _flat_floats(value, *, name: str, expected: int | None = None) -> list[float]:
    raw = _tolist(value)
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise RuntimeError(f"{name} must be a vector or a single nominal row, got {len(raw)} rows")
        raw = raw[0]
    out = [float(item) for item in raw]
    if expected is not None and len(out) != expected:
        raise RuntimeError(f"{name} has {len(out)} values, expected {expected}")
    if any(not math.isfinite(item) for item in out):
        raise RuntimeError(f"{name} contains NaN/Inf")
    return out


def _nominal_row(value, *, name: str, expected: int) -> list[float]:
    raw = _tolist(value)
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return _flat_floats(raw, name=name, expected=expected)


def _joint_ids(value, count: int) -> list[int]:
    if isinstance(value, slice):
        if value != slice(None):
            return list(range(count))[value]
        return list(range(count))
    return [int(item) for item in _tolist(value)]


def _joint_actuator_types(robot, count: int) -> list[str]:
    """Resolve the instantiated actuator integration model for every articulation joint."""
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Mapping) or not actuators:
        raise RuntimeError("robot.actuators is unavailable; actuator integration cannot be proven")
    result: list[str | None] = [None] * count
    for group_name, actuator in actuators.items():
        if not hasattr(actuator, "joint_indices") or not hasattr(actuator, "is_implicit_model"):
            raise RuntimeError(
                f"actuator group {group_name!r} lacks joint_indices/is_implicit_model"
            )
        kind = "implicit" if bool(actuator.is_implicit_model) else "explicit"
        for joint_id in _joint_ids(actuator.joint_indices, count):
            if not (0 <= joint_id < count):
                raise RuntimeError(
                    f"actuator group {group_name!r} has out-of-range joint id {joint_id}"
                )
            if result[joint_id] is not None:
                raise RuntimeError(
                    f"joint {joint_id} belongs to multiple actuator groups"
                )
            result[joint_id] = kind
    missing = [index for index, value in enumerate(result) if value is None]
    if missing:
        raise RuntimeError(f"actuator integration is unresolved for joints {missing}")
    return [str(value) for value in result]


def _policy_layout(env) -> tuple[list[str], list[int], int]:
    manager = env.observation_manager
    names = [str(name) for name in manager.active_terms["policy"]]
    raw_dims = manager.group_obs_term_dim["policy"]
    dims = []
    for dim in raw_dims:
        if isinstance(dim, (tuple, list)):
            if len(dim) != 1:
                raise RuntimeError(f"policy observation term has non-flat dimension {dim!r}")
            dim = dim[0]
        dims.append(int(dim))
    total = manager.group_obs_dim["policy"]
    if isinstance(total, (tuple, list)):
        if len(total) != 1:
            raise RuntimeError(f"policy observation group has non-flat dimension {total!r}")
        total = total[0]
    total = int(total)
    if len(names) != len(dims) or sum(dims) != total:
        raise RuntimeError(
            f"invalid policy layout: names={len(names)} dims={dims} total={total}"
        )
    return names, dims, total


def _observation_history_lengths(env, names: list[str]) -> list[int]:
    group_cfg = env.observation_manager.cfg.policy
    if group_cfg.history_length is not None:
        return [int(group_cfg.history_length)] * len(names)
    cfg_by_name = group_cfg.to_dict()
    out = []
    for name in names:
        raw = cfg_by_name[name]["history_length"]
        value = 0 if raw is None else int(raw)
        out.append(1 if value == 0 else value)
    return out


def runtime_execution_facts(
    env, actor_contract, *, allow_legacy_actor_leg_ref_mask_ambiguity: bool = False
) -> dict:
    """Extract schema-3 execution facts from the instantiated, startup-initialized environment."""
    robot = env.scene["robot"]
    data = robot.data
    articulation_names = list(
        getattr(data, "joint_names", getattr(robot, "joint_names", ()))
    )
    if not articulation_names or len(set(articulation_names)) != len(articulation_names):
        raise RuntimeError("robot articulation joint names are empty or non-unique")
    n = len(articulation_names)

    action = env.action_manager.get_term("joint_pos")
    ids = _joint_ids(getattr(action, "_joint_ids", slice(None)), n)
    identity = list(range(n))
    if ids != identity:
        raise RuntimeError(
            "schema-3 ONNX requires identity action/articulation order because actions and baked "
            f"reference joints share one joint_names contract; got action_joint_ids={ids}"
        )
    joint_names = [articulation_names[index] for index in ids]
    action_cfg = getattr(action, "cfg", getattr(env.cfg.actions, "joint_pos", None))
    use_default_offset = bool(getattr(action_cfg, "use_default_offset", False))
    if not use_default_offset:
        raise RuntimeError(
            "schema-3 deploy decoder requires JointPositionAction use_default_offset=True"
        )

    if not hasattr(data, "default_joint_pos_nominal"):
        raise RuntimeError(
            "robot.data.default_joint_pos_nominal is missing; the startup nominal-pose capture "
            "must run before writing a schema-3 contract"
        )
    default_q = _flat_floats(
        data.default_joint_pos_nominal, name="default_joint_pos_nominal", expected=n
    )
    action_scale = _nominal_row(action._scale, name="action_scale", expected=n)
    kp = _nominal_row(data.default_joint_stiffness, name="joint_stiffness", expected=n)
    kd = _nominal_row(data.default_joint_damping, name="joint_damping", expected=n)
    effort = _nominal_row(data.joint_effort_limits, name="joint_effort_limits", expected=n)
    actuator_types = _joint_actuator_types(robot, n)
    armature = _nominal_row(
        data.default_joint_armature, name="joint_armature", expected=n
    )
    friction = _nominal_row(
        data.default_joint_friction_coeff,
        name="joint_friction_coefficients",
        expected=n,
    )
    velocity_limits = _nominal_row(
        data.joint_vel_limits, name="joint_velocity_limits", expected=n
    )
    if any(value <= 0.0 for value in action_scale):
        raise RuntimeError("action_scale must be finite and positive")
    if any(value <= 0.0 for value in kp) or any(value <= 0.0 for value in kd):
        raise RuntimeError("nominal joint stiffness/damping must be finite and positive")
    if any(value <= 0.0 for value in effort):
        raise RuntimeError("joint_effort_limits must be finite and positive")
    if any(value < 0.0 for value in armature):
        raise RuntimeError("joint_armature must be finite and non-negative")
    if any(value < 0.0 for value in friction):
        raise RuntimeError("joint_friction_coefficients must be finite and non-negative")
    if any(value <= 0.0 for value in velocity_limits):
        raise RuntimeError("joint_velocity_limits must be finite and positive")

    limits_raw = _tolist(data.soft_joint_pos_limits)
    if (
        limits_raw
        and isinstance(limits_raw[0], (list, tuple))
        and limits_raw[0]
        and isinstance(limits_raw[0][0], (list, tuple))
    ):
        # Isaac stores [num_envs, num_joints, 2]; select the nominal first environment.
        limits_raw = limits_raw[0]
    if len(limits_raw) != n:
        raise RuntimeError(f"soft_joint_pos_limits has {len(limits_raw)} joints, expected {n}")
    limits = []
    for index, pair in enumerate(limits_raw):
        if len(pair) != 2:
            raise RuntimeError(f"soft_joint_pos_limits[{index}] is not [lo, hi]")
        lo, hi = float(pair[0]), float(pair[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise RuntimeError(f"invalid soft q-des limits for {joint_names[index]}: {(lo, hi)}")
        limits.append([lo, hi])

    physics_dt = float(env.physics_dt)
    policy_dt = float(env.step_dt)
    decimation = int(env.cfg.decimation)
    if not (math.isfinite(physics_dt) and physics_dt > 0.0):
        raise RuntimeError(f"invalid physics dt {physics_dt!r}")
    if not (math.isfinite(policy_dt) and policy_dt > 0.0) or decimation <= 0:
        raise RuntimeError(f"invalid policy dt/decimation {policy_dt!r}/{decimation!r}")
    if not math.isclose(policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"policy dt {policy_dt:.17g} != physics dt {physics_dt:.17g} * {decimation}"
        )

    obs_names, obs_dims, obs_total = _policy_layout(env)
    history = _observation_history_lengths(env, obs_names)
    if actor_contract is not None:
        expected_layout = [(term.name, int(term.dim)) for term in actor_contract.terms]
        if list(zip(obs_names, obs_dims)) != expected_layout or obs_total != int(
            actor_contract.total_dim
        ):
            raise RuntimeError("actor contract object does not match the instantiated policy layout")

    motion = env.command_manager.get_term("motion")
    body_ids = [int(item) for item in _tolist(motion.body_indexes)]
    robot_body_names = list(getattr(robot, "body_names", ()))
    if (
        not robot_body_names
        or any(not str(name) for name in robot_body_names)
        or len(set(robot_body_names)) != len(robot_body_names)
    ):
        raise RuntimeError("robot body_names are unavailable or non-unique")
    if any(index < 0 or index >= len(robot_body_names) for index in body_ids):
        raise RuntimeError(f"resolved motion body indices are out of range: {body_ids}")
    body_names = [robot_body_names[index] for index in body_ids]
    configured_bodies = [str(name) for name in motion.cfg.body_names]
    if body_names != configured_bodies:
        raise RuntimeError(
            f"resolved motion body order {body_names} != configured order {configured_bodies}"
        )
    anchor_name = str(motion.cfg.anchor_body_name)
    if anchor_name not in body_names:
        raise RuntimeError(f"motion anchor {anchor_name!r} is absent from resolved body order")
    anchor_index = body_names.index(anchor_name)
    segment_lengths = [int(value) for value in _tolist(motion.motion.seg_len)]
    if not segment_lengths or any(value <= 0 for value in segment_lengths):
        raise RuntimeError(f"invalid motion segment lengths {segment_lengths}")
    clip_fps = [float(value) for value in motion.motion.per_clip_fps]
    if len(clip_fps) != len(segment_lengths):
        raise RuntimeError(
            "motion fps count does not match segments: "
            f"{len(clip_fps)} vs {len(segment_lengths)}"
        )
    policy_hz = 1.0 / policy_dt
    if any(
        not math.isfinite(value)
        or value <= 0.0
        or not math.isclose(value, policy_hz, rel_tol=0.0, abs_tol=1e-9)
        for value in clip_fps
    ):
        raise RuntimeError(
            f"motion clip fps {clip_fps} must all equal policy rate {policy_hz:.12g} Hz"
        )
    kinematics_contracts = [dict(item) for item in motion.motion.kinematics_contracts]
    if len(kinematics_contracts) != len(segment_lengths):
        raise RuntimeError(
            "motion kinematics-contract count does not match segments: "
            f"{len(kinematics_contracts)} vs {len(segment_lengths)}"
        )
    try:
        resolve_motion_body_lin_vel_points(kinematics_contracts)
    except ValueError as exc:
        raise RuntimeError(f"unresolved motion velocity-point contract: {exc}") from exc
    kinematics_exact = bool(motion.motion.kinematics_contract_exact)
    if kinematics_exact != all(bool(item.get("exact", False)) for item in kinematics_contracts):
        raise RuntimeError("motion kinematics exact flag disagrees with per-clip contracts")

    qdes_clamp = bool(
        getattr(action, "_clamp_enabled", getattr(env.cfg.actions.joint_pos, "clamp", False))
    )
    # OFF is encoded by total absence so every legacy/default schema-3 contract stays
    # byte-identical.  ON must be proved twice: by the composed action config and by the
    # instantiated action term's exact runtime property.  Falling back from a true config to the
    # config value would let a stale action implementation falsely claim constrained-action
    # projection support.
    projection_cfg = getattr(
        action_cfg,
        "project_finite_preclamp_qdes_without_termination",
        False,
    )
    if type(projection_cfg) is not bool:
        raise RuntimeError(
            "project_finite_preclamp_qdes_without_termination must be an exact boolean"
        )
    projection_missing = object()
    projection_runtime = getattr(
        action,
        FINITE_PRECLAMP_QDES_PROJECTION_KEY,
        projection_missing,
    )
    if projection_runtime is projection_missing:
        if projection_cfg:
            raise RuntimeError(
                "finite q_des projection is enabled in config but the instantiated "
                "action term exposes no runtime projection property"
            )
        projection_runtime = False
    if type(projection_runtime) is not bool:
        raise RuntimeError(
            f"{FINITE_PRECLAMP_QDES_PROJECTION_KEY} must be an exact boolean"
        )
    if projection_runtime is not projection_cfg:
        raise RuntimeError(
            "finite q_des projection config/runtime facts disagree"
        )
    projection_inset = None
    if projection_runtime:
        projection_inset_cfg = getattr(
            action_cfg,
            FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY,
            None,
        )
        if (
            isinstance(projection_inset_cfg, bool)
            or not isinstance(projection_inset_cfg, (int, float))
            or not math.isfinite(float(projection_inset_cfg))
            or not 0.0 <= float(projection_inset_cfg) < 0.5
        ):
            raise RuntimeError(
                "finite_projection_soft_envelope_inset_fraction must be "
                "finite and lie in [0, 0.5)"
            )
        projection_inset_runtime = getattr(
            action,
            FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY,
            projection_missing,
        )
        if projection_inset_runtime is projection_missing:
            raise RuntimeError(
                "finite q_des projection is enabled in config but the "
                "instantiated action term exposes no runtime soft-envelope inset"
            )
        if (
            isinstance(projection_inset_runtime, bool)
            or not isinstance(projection_inset_runtime, (int, float))
            or not math.isfinite(float(projection_inset_runtime))
            or float(projection_inset_runtime) != float(projection_inset_cfg)
        ):
            raise RuntimeError(
                "finite q_des projection soft-envelope inset config/runtime "
                "facts disagree"
            )
        projection_inset = float(projection_inset_runtime)
    # R-a masking leaves the 62-D layout unchanged, so both the true-only mask bit and an always
    # present provenance epoch are needed: only epoch=1 + absent mask proves unmasked. Detection
    # unwraps only structurally empty partials and accepts only the two canonical callables.  Bound
    # args/kwargs can change command selection or semantics and may not be discarded as provenance.
    _cmd_term = getattr(
        getattr(getattr(env.cfg, "observations", None), "policy", None), "command", None
    )
    _cmd_func = getattr(_cmd_term, "func", None)
    # Only the exact built-in partial type is transparent.  ``functools.partial`` is subclassable,
    # and a subclass may override ``__call__`` while keeping canonical ``.func`` plus empty
    # ``args``/``keywords``.  Treating such an object as a plain partial would mint provenance for
    # behavior that is not the canonical callable.  Apply the exact-type rule at every unwrap layer.
    while type(_cmd_func) is functools.partial:
        if _cmd_func.args or _cmd_func.keywords:
            _cmd_func = None
            break
        _cmd_func = _cmd_func.func
    facts = {
        "articulation_joint_names": articulation_names,
        "action_joint_ids": ids,
        "joint_names": joint_names,
        "default_joint_pos": default_q,
        "action_scale": action_scale,
        "joint_stiffness": kp,
        "joint_damping": kd,
        "joint_effort_limits": effort,
        "joint_actuator_types": actuator_types,
        "joint_armature": armature,
        "joint_friction_coefficients": friction,
        "joint_velocity_limits": velocity_limits,
        "joint_friction_backend": JOINT_FRICTION_BACKEND,
        "joint_friction_semantics": JOINT_FRICTION_SEMANTICS,
        "joint_friction_units": JOINT_FRICTION_UNITS,
        "qdes_joint_pos_limits": limits,
        "action_use_default_offset": use_default_offset,
        "qdes_clamp": qdes_clamp,
        **(
            {
                FINITE_PRECLAMP_QDES_PROJECTION_KEY: True,
                FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY: (
                    projection_inset
                ),
            }
            if projection_runtime
            else {}
        ),
        "physics_step_dt_s": physics_dt,
        "policy_step_dt_s": policy_dt,
        "control_decimation": decimation,
        "actor_obs_contract": getattr(actor_contract, "name", None),
        "actor_obs_mode": getattr(actor_contract, "obs_mode", None),
        "actor_obs_total_dim": obs_total,
        "actor_obs_term_names": obs_names,
        "actor_obs_term_dims": obs_dims,
        "observation_history_lengths": history,
        "articulation_body_names": robot_body_names,
        "body_names": body_names,
        "body_indices": body_ids,
        "anchor_body_name": anchor_name,
        "anchor_body_index": anchor_index,
        "motion_segment_lengths": segment_lengths,
        "motion_clip_fps": clip_fps,
        "motion_kinematics_contracts": kinematics_contracts,
        "motion_kinematics_exact": kinematics_exact,
    }
    if any(name == "command" and dim == 62 for name, dim in zip(obs_names, obs_dims)):
        actor_leg_ref_mask = _classify_actor_leg_ref_mask_callable(_cmd_func)
        if actor_leg_ref_mask is None and not allow_legacy_actor_leg_ref_mask_ambiguity:
            raise RuntimeError(
                "62-D actor command func is not one of the two canonical epoch-1 callables; "
                "wrappers, partial subclasses, partials with bound args/kwargs, and copied "
                "provenance attributes are not authoritative"
            )
        if actor_leg_ref_mask is not None:
            facts[ACTOR_LEG_REF_MASK_PROVENANCE_KEY] = ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH
        if actor_leg_ref_mask is True:
            facts["actor_leg_ref_mask"] = True
    return facts


_A3_LOWER_BODY_RUNTIME_JOINT_ORDER = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
_A3_LOWER_BODY_LEGS = tuple(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER[-12:])


def _wave_finite(value: object, *, name: str, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"schema-3 {name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"schema-3 {name} must be a finite number")
    if positive and value <= 0.0:
        raise ValueError(f"schema-3 {name} must be finite and > 0")
    if nonnegative and value < 0.0:
        raise ValueError(f"schema-3 {name} must be finite and >= 0")
    return value


def _validate_lower_body_wave_contracts(
    contract: Mapping,
    joint_names: list[str],
    articulation_joint_names: list[str],
) -> None:
    pose = contract.get("lower_body_pose_imitation_reward")
    bundle = contract.get("lower_body_stability_bundle_reward")
    if (pose is None) != (bundle is None):
        raise ValueError("schema-3 explicit Wave-B cells require both B1 and B2 blocks")
    if pose is not None:
        pose = _require_exact_mapping_keys(
            pose,
            frozenset(
                {
                    "schema_version", "enabled", "probe_enabled", "activation_ledger",
                    "weight", "std_rad", "support_pre_s", "support_post_s",
                    "racket_command_name", "motion_command_name", "joint_count",
                    "joint_names", "joint_order", "reference_joint_order", "formula",
                    "gate", "success_conditioned",
                }
            ),
            name="schema-3 lower_body_pose_imitation_reward",
        )
        if type(pose["schema_version"]) is not int or pose["schema_version"] != 1:
            raise ValueError("schema-3 lower_body_pose_imitation_reward schema_version must be 1")
        if (
            not isinstance(pose["enabled"], bool)
            or pose["probe_enabled"] is not True
            or pose["success_conditioned"] is not False
        ):
            raise ValueError("schema-3 lower_body_pose_imitation_reward flags are invalid")
        weight = _wave_finite(
            pose["weight"], name="lower_body_pose_imitation_reward.weight", nonnegative=True
        )
        _wave_finite(pose["std_rad"], name="lower_body_pose_imitation_reward.std_rad", positive=True)
        _wave_finite(
            pose["support_pre_s"],
            name="lower_body_pose_imitation_reward.support_pre_s",
            nonnegative=True,
        )
        _wave_finite(
            pose["support_post_s"],
            name="lower_body_pose_imitation_reward.support_post_s",
            nonnegative=True,
        )
        if pose["enabled"] != (weight > 0.0):
            raise ValueError("schema-3 lower_body_pose_imitation_reward enabled disagrees with weight")
        expected = {
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "activation_ledger": "weight_independent_control_step_counters",
            "joint_count": 12,
            "joint_names": list(_A3_LOWER_BODY_LEGS),
            "joint_order": "canonical_deploy_order_selected_by_name",
            "reference_joint_order": "motion_command_runtime_articulation_identity",
            "formula": "exp(-mean(square(q_leg-qref_leg))/square(std_rad))",
            "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
        }
        for key, value in expected.items():
            if pose[key] != value:
                raise ValueError(
                    f"schema-3 lower_body_pose_imitation_reward {key} must be {value!r}"
                )

    if bundle is not None:
        bundle = _require_exact_mapping_keys(
            bundle,
            frozenset(
                {
                    "schema_version", "enabled", "probe_enabled", "activation_ledger",
                    "weight", "min_stance_width_m", "stance_scale_m",
                    "leg_velocity_margin_radps", "leg_velocity_scale_radps",
                    "support_pre_s", "support_post_s", "racket_command_name",
                    "motion_command_name", "leg_joint_count", "leg_joint_names",
                    "foot_body_names", "joint_order", "stance_width_frame", "components",
                    "formula", "gate", "success_conditioned", "uses_motion_reference",
                    "duplicates_slip_or_upright",
                }
            ),
            name="schema-3 lower_body_stability_bundle_reward",
        )
        if type(bundle["schema_version"]) is not int or bundle["schema_version"] != 1:
            raise ValueError("schema-3 lower_body_stability_bundle_reward schema_version must be 1")
        if (
            not isinstance(bundle["enabled"], bool)
            or bundle["probe_enabled"] is not True
            or bundle["success_conditioned"] is not False
            or bundle["uses_motion_reference"] is not False
            or bundle["duplicates_slip_or_upright"] is not False
        ):
            raise ValueError("schema-3 lower_body_stability_bundle_reward flags are invalid")
        weight = _wave_finite(bundle["weight"], name="lower_body_stability_bundle_reward.weight")
        if weight > 0.0 or bundle["enabled"] != (weight < 0.0):
            raise ValueError("schema-3 lower_body_stability_bundle_reward weight/enabled is invalid")
        for name in (
            "min_stance_width_m", "stance_scale_m", "leg_velocity_scale_radps",
        ):
            _wave_finite(bundle[name], name=f"lower_body_stability_bundle_reward.{name}", positive=True)
        for name in ("leg_velocity_margin_radps", "support_pre_s", "support_post_s"):
            _wave_finite(
                bundle[name], name=f"lower_body_stability_bundle_reward.{name}", nonnegative=True
            )
        expected = {
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "activation_ledger": "weight_independent_control_step_counters",
            "leg_joint_count": 12,
            "leg_joint_names": list(_A3_LOWER_BODY_LEGS),
            "foot_body_names": ["left_ankle_roll_Link", "right_ankle_roll_Link"],
            "joint_order": "canonical_deploy_order_selected_by_name",
            "stance_width_frame": "base_yaw_lateral_signed_left_minus_right",
            "components": [
                "stance_width_lower_hinge",
                "twelve_leg_realized_qdot_tail",
            ],
            "formula": "mean(bounded_stance_tail,bounded_leg_qdot_tail)",
            "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
        }
        for key, value in expected.items():
            if bundle[key] != value:
                raise ValueError(
                    f"schema-3 lower_body_stability_bundle_reward {key} must be {value!r}"
                )
        articulation_bodies = contract.get("articulation_body_names")
        if (
            not isinstance(articulation_bodies, (list, tuple))
            or any(name not in articulation_bodies for name in bundle["foot_body_names"])
        ):
            raise ValueError(
                "schema-3 lower_body_stability_bundle_reward foot bodies are absent from articulation"
            )

    # The live articulation enumerates the same 31 joints breadth-first; require
    # the exact A3 name set and runtime==articulation identity, not one fixed order.
    if (pose is not None or bundle is not None) and (
        not isinstance(joint_names, (list, tuple))
        or len(joint_names) != len(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER)
        or len(set(joint_names)) != len(joint_names)
        or set(joint_names) != set(_A3_LOWER_BODY_RUNTIME_JOINT_ORDER)
        or list(articulation_joint_names or []) != list(joint_names)
    ):
        raise ValueError(
            "schema-3 lower-body rewards require exact A3 runtime/articulation joint identity"
        )
    if pose is not None and bundle is not None and pose["enabled"] and bundle["enabled"]:
        raise ValueError("schema-3 Wave-B B1 and B2 rewards are mutually exclusive")


# --------------------------------------------------------------------------------------------- #
# Wave-P random base push (PACE/BeyondMimic-style shove; default OFF = HITTER no-push recipe).
# --------------------------------------------------------------------------------------------- #
PUSH_ROBOT_EVENT_KEY = "push_robot_event"
PUSH_ROBOT_EVENT_FUNC = "push_by_setting_velocity"
PUSH_ROBOT_EVENT_MODE = "interval"
PUSH_ROBOT_ANG_AXES = ("none", "yaw", "rpy")
_PUSH_ROBOT_EVENT_KEYS = frozenset(
    {
        "schema_version", "enabled", "func", "mode", "interval_range_s",
        "vel_xy_mps", "ang_vel_radps", "ang_axes", "velocity_range",
    }
)
# 合并互斥推(Franco 2026-07-25:速度踢 + 持续力推合并成【一个】interval 事件,每次触发按
# force_prob 抽签二选一,防两种随机推同帧叠加)。v1 flag 语义:legacy 块不带
# combined_exclusive 键、拼写逐字节不变;合并块必须带全下面的键面(含力分支的同冲量记账
# robot_mass_kg / delta_v_equiv_mps),缺一个/多一个都 fail-loud。
PUSH_COMBINED_EVENT_FUNC = "push_combined_exclusive"
_PUSH_COMBINED_ASSEMBLY_KEYS = frozenset(
    {
        "schema_version", "enabled", "combined_exclusive", "func", "mode",
        "interval_range_s", "force_prob",
        "vel_xy_mps", "ang_vel_radps", "ang_axes", "velocity_range",
        "force_n", "duration_s", "duration_steps", "control_dt_s",
        "body_name", "application_point",
    }
)
_PUSH_COMBINED_EVENT_KEYS = _PUSH_COMBINED_ASSEMBLY_KEYS | {
    "robot_mass_kg", "delta_v_equiv_mps",
}


def _ground_plant_range(value, *, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a [lo, hi] pair")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a [lo, hi] pair") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must be a [lo, hi] pair")
    lo = _wave_finite(items[0], name=f"{name}[0]", nonnegative=True)
    hi = _wave_finite(items[1], name=f"{name}[1]", nonnegative=True)
    if lo > hi:
        raise ValueError(f"{name} must satisfy 0 <= lo <= hi")
    return [lo, hi]


def ground_plant_block(
    *,
    ground_static_friction,
    ground_dynamic_friction,
    robot_material_static_friction_range,
    robot_material_dynamic_friction_range,
    terrain_type,
    terrain_rough_height_range_m,
    robot_material_make_consistent=False,
) -> dict | None:
    """Canonical ground/terrain plant identity block.

    人话:把"地面多滑、机器人身体材质随机到什么范围、地是平的还是随机凹凸"拼成一个规范块。
    与历史字节默认(平地、地面 1.0/1.0、机器人材质 (0.3,1.6)/(0.3,1.2))完全相等时返回
    ``None`` = 合同不写这个键,所有历史 checkpoint 逐字节兼容;任何偏离都返回完整块,resume
    对账(_contract_diff)会把它当成另一套 plant 拒绝静默续训。This is the single
    validation/assembly source shared by the train.py ``task.plant`` override path and the
    schema-3 contract validator.
    """

    static = _wave_finite(
        ground_static_friction, name="ground_plant.ground_static_friction", nonnegative=True
    )
    dynamic = _wave_finite(
        ground_dynamic_friction, name="ground_plant.ground_dynamic_friction", nonnegative=True
    )
    if dynamic > static:
        raise ValueError(
            "ground_plant ground dynamic friction must not exceed static friction "
            f"(got static={static}, dynamic={dynamic})"
        )
    static_range = _ground_plant_range(
        robot_material_static_friction_range,
        name="ground_plant.robot_material_static_friction_range",
    )
    dynamic_range = _ground_plant_range(
        robot_material_dynamic_friction_range,
        name="ground_plant.robot_material_dynamic_friction_range",
    )
    if terrain_type not in (GROUND_PLANT_TERRAIN_PLANE, GROUND_PLANT_TERRAIN_ROUGH):
        raise ValueError(
            "ground_plant terrain_type must be "
            f"{GROUND_PLANT_TERRAIN_PLANE!r} or {GROUND_PLANT_TERRAIN_ROUGH!r}, "
            f"got {terrain_type!r}"
        )
    if terrain_type == GROUND_PLANT_TERRAIN_PLANE:
        if terrain_rough_height_range_m is not None:
            raise ValueError(
                "ground_plant plane terrain must not carry terrain_rough_height_range_m"
            )
        height = None
    else:
        height = _ground_plant_range(
            terrain_rough_height_range_m, name="ground_plant.terrain_rough_height_range_m"
        )
        if height[1] <= 0.0:
            raise ValueError(
                "ground_plant rough terrain requires height hi > 0 (flat ground is spelled "
                "terrain_type=plane with a null range)"
            )
        if height[1] > 0.5:
            raise ValueError(
                "ground_plant rough terrain height hi > 0.5 m is not a plausible arena floor"
            )
        band = height[1] - height[0]
        if band < 0.01 - 1e-12:
            raise ValueError(
                "ground_plant rough band (hi - lo) must be >= 0.01 m: heights are "
                "re-centred to ±(hi-lo)/2 about z=0 and quantized at 5 mm"
            )
        if band > 0.15 + 1e-12:
            raise ValueError(
                "ground_plant rough band (hi - lo) must be <= 0.15 m: beyond that the "
                "height-field slope wall correction breaks the flat table boundary"
            )
        ratio = (band / 2.0) / 0.005
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                "ground_plant rough band (hi - lo) must be a multiple of 0.01 m "
                "(5 mm height quantization)"
            )
    if not isinstance(robot_material_make_consistent, bool):
        raise ValueError(
            "ground_plant robot_material_make_consistent must be a bool"
        )
    block = {
        "schema_version": 1,
        "ground_static_friction": static,
        "ground_dynamic_friction": dynamic,
        "robot_material_static_friction_range": static_range,
        "robot_material_dynamic_friction_range": dynamic_range,
        "terrain_type": terrain_type,
        "terrain_rough_height_range_m": height,
    }
    # false 的唯一拼写是键缺席(历史 6 键块逐字节兼容);true 才让块长出这个键。
    if robot_material_make_consistent:
        block[GROUND_PLANT_MAKE_CONSISTENT_KEY] = True
    if {key: value for key, value in block.items() if key != "schema_version"} == (
        GROUND_PLANT_DEFAULT
    ):
        return None
    return block


def _validate_ground_plant_contract(contract: Mapping) -> None:
    """Optional ground/terrain plant block: absent = historical default plant (byte-identical).

    A present block must be internally consistent with its own canonical re-assembly, and a
    block that spells the historical default is refused — the default has exactly one spelling
    (total absence) so resume diffs stay byte-exact across the whole lineage.
    """

    block = contract.get(GROUND_PLANT_KEY)
    if block is None:
        if GROUND_PLANT_KEY in contract:
            raise ValueError(
                "schema-3 ground_plant must be omitted when default, not null"
            )
        return
    if not isinstance(block, Mapping):
        raise ValueError("schema-3 ground_plant must be a mapping")
    # Optional 2026-07-29 key: only the literal True may appear; False is spelled by omission
    # (keeps any pre-existing 6-key block byte-exact under this validator).
    make_consistent = False
    if GROUND_PLANT_MAKE_CONSISTENT_KEY in block:
        if block[GROUND_PLANT_MAKE_CONSISTENT_KEY] is not True:
            raise ValueError(
                "schema-3 ground_plant robot_material_make_consistent equal to the "
                "default (false) must be spelled by omitting the key"
            )
        make_consistent = True
    base = {
        key: value
        for key, value in dict(block).items()
        if key != GROUND_PLANT_MAKE_CONSISTENT_KEY
    }
    base = _require_exact_mapping_keys(
        base, _GROUND_PLANT_KEYS, name="schema-3 ground_plant"
    )
    if type(base["schema_version"]) is not int or base["schema_version"] != 1:
        raise ValueError("schema-3 ground_plant schema_version must be integer 1")
    try:
        expected = ground_plant_block(
            ground_static_friction=base["ground_static_friction"],
            ground_dynamic_friction=base["ground_dynamic_friction"],
            robot_material_static_friction_range=base[
                "robot_material_static_friction_range"
            ],
            robot_material_dynamic_friction_range=base[
                "robot_material_dynamic_friction_range"
            ],
            robot_material_make_consistent=make_consistent,
            terrain_type=base["terrain_type"],
            terrain_rough_height_range_m=base["terrain_rough_height_range_m"],
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 ground_plant is invalid: {exc}") from exc
    if expected is None:
        raise ValueError(
            "schema-3 ground_plant equal to the historical default must be spelled by "
            "omitting the block"
        )
    if dict(block) != expected:
        raise ValueError(
            "schema-3 ground_plant does not equal its canonical re-assembly"
        )


def push_robot_event_block(
    *, enable, interval_range_s, vel_xy_mps, ang_vel_radps, ang_axes,
    combined_exclusive=False, force_prob=None, force_n=None, duration_s=None,
    control_dt_s=None,
):
    """Translate the push flag group into the canonical ``push_robot_event`` contract block.

    人话:把"要不要推、隔几秒推一次、水平推多快、角速度踢多快、踢哪些转轴"翻译成 push_robot
    事件的对称速度区间表 + 合同块。``enable=False`` 返回 ``None``(= 不推,合同不写这个块,
    所有历史/在跑配置逐位不变),但此时任何非零幅度都是配置错误(fail-closed:关着的开关
    不许挂着上膛的参数)。This is the single validation/assembly source shared by the env cfg
    flag path, the train.py ``task.push`` override, and the schema-3 contract validator.

    合并互斥模式(Franco 2026-07-25;默认 ``combined_exclusive=False`` = legacy 块逐字节不变):
    ``combined_exclusive=True`` 时速度踢与持续力推合并成【一个】interval 事件,每次触发按
    ``force_prob`` 抽签二选一(严格 0<p<1;0/1 等价单类型推,请直接用 legacy 单事件),力分支
    参数(``force_n``/``duration_s``/``control_dt_s``)复用 :func:`force_push_event_block` 的同一套
    校验(含"duration 必须整数个控制步"与同冲量 Δv_equiv = F·Δt/m 的记账键面)。combined 关着
    时四个合并参数必须全 ``None``(关着的开关不许挂上膛参数),开着时缺一个都 fail-loud。
    """

    if not isinstance(enable, bool):
        raise ValueError("push_robot_event enable must be an explicit boolean")
    if not isinstance(combined_exclusive, bool):
        raise ValueError(
            "push_robot_event combined_exclusive must be an explicit boolean"
        )
    combined_fields = (
        ("force_prob", force_prob),
        ("force_n", force_n),
        ("duration_s", duration_s),
        ("control_dt_s", control_dt_s),
    )
    if not combined_exclusive:
        loaded = sorted(name for name, value in combined_fields if value is not None)
        if loaded:
            raise ValueError(
                f"push_robot_event combined_exclusive=false may not carry merged-push "
                f"fields {loaded} — delete them or set combined_exclusive=true"
            )
    if not enable:
        if combined_exclusive:
            raise ValueError(
                "push_robot_event combined_exclusive=true requires enable=true — "
                "a disabled merged push must spell combined_exclusive=false "
                "(关着的合并开关不许上膛)"
            )
        for name, value in (
            ("vel_xy_mps", vel_xy_mps),
            ("ang_vel_radps", ang_vel_radps),
        ):
            if value is not None and (isinstance(value, bool) or float(value) != 0.0):
                raise ValueError(
                    f"push_robot_event disabled but {name}={value!r} is nonzero — "
                    "delete the dormant amplitude or set enable=true"
                )
        if ang_axes not in (None, "none"):
            raise ValueError(
                f"push_robot_event disabled but ang_axes={ang_axes!r} selects push axes — "
                "delete it or set enable=true"
            )
        return None
    try:
        interval_items = list(interval_range_s)
    except TypeError as exc:
        raise ValueError(
            "push_robot_event interval_range_s must be a [lo, hi] pair of seconds"
        ) from exc
    if len(interval_items) != 2:
        raise ValueError(
            "push_robot_event interval_range_s must be a [lo, hi] pair of seconds"
        )
    interval_lo = _wave_finite(
        interval_items[0], name="push_robot_event.interval_range_s[0]", positive=True
    )
    interval_hi = _wave_finite(
        interval_items[1], name="push_robot_event.interval_range_s[1]", positive=True
    )
    if interval_lo > interval_hi:
        raise ValueError(
            "push_robot_event interval_range_s must satisfy 0 < lo <= hi"
        )
    vel = _wave_finite(vel_xy_mps, name="push_robot_event.vel_xy_mps", nonnegative=True)
    ang = _wave_finite(
        ang_vel_radps, name="push_robot_event.ang_vel_radps", nonnegative=True
    )
    if ang_axes not in PUSH_ROBOT_ANG_AXES:
        raise ValueError(
            f"push_robot_event ang_axes must be one of {PUSH_ROBOT_ANG_AXES}, got {ang_axes!r}"
        )
    if ang_axes == "none" and ang != 0.0:
        raise ValueError(
            "push_robot_event ang_axes='none' requires ang_vel_radps=0 "
            "(pick ang_axes='yaw'|'rpy' to push angular velocity)"
        )
    if ang_axes != "none" and ang == 0.0:
        raise ValueError(
            f"push_robot_event ang_axes={ang_axes!r} requires ang_vel_radps > 0 "
            "(a zero angular push must spell ang_axes='none')"
        )
    if vel == 0.0 and ang == 0.0:
        raise ValueError(
            "push_robot_event enabled but every amplitude is zero — "
            "an enabled push must push something (or set enable=false)"
        )
    velocity_range = {"x": [-vel, vel], "y": [-vel, vel]}
    if ang_axes == "yaw":
        velocity_range["yaw"] = [-ang, ang]
    elif ang_axes == "rpy":
        velocity_range["roll"] = [-ang, ang]
        velocity_range["pitch"] = [-ang, ang]
        velocity_range["yaw"] = [-ang, ang]
    if not combined_exclusive:
        return {
            "schema_version": 1,
            "enabled": True,
            "func": PUSH_ROBOT_EVENT_FUNC,
            "mode": PUSH_ROBOT_EVENT_MODE,
            "interval_range_s": [interval_lo, interval_hi],
            "vel_xy_mps": vel,
            "ang_vel_radps": ang,
            "ang_axes": ang_axes,
            "velocity_range": velocity_range,
        }
    # --- 合并互斥模式:力分支复用 force_push_event_block 的同一套校验/装配(单一来源)。 ---
    missing = sorted(name for name, value in combined_fields if value is None)
    if missing:
        raise ValueError(
            f"push_robot_event combined_exclusive=true requires the complete merged "
            f"recipe; missing {missing}"
        )
    force_block = force_push_event_block(
        enable=True,
        interval_range_s=[interval_lo, interval_hi],
        force_n=force_n,
        duration_s=duration_s,
        control_dt_s=control_dt_s,
    )
    prob = _wave_finite(force_prob, name="push_robot_event.force_prob")
    if not (0.0 < prob < 1.0):
        raise ValueError(
            f"push_robot_event force_prob must lie strictly inside (0, 1), got {prob!r} "
            "— 0/1 就是单类型推,请直接用 legacy 单事件而不是合并模式"
        )
    return {
        "schema_version": 1,
        "enabled": True,
        "combined_exclusive": True,
        "func": PUSH_COMBINED_EVENT_FUNC,
        "mode": PUSH_ROBOT_EVENT_MODE,
        "interval_range_s": [interval_lo, interval_hi],
        "force_prob": prob,
        "vel_xy_mps": vel,
        "ang_vel_radps": ang,
        "ang_axes": ang_axes,
        "velocity_range": velocity_range,
        "force_n": force_block["force_n"],
        "duration_s": force_block["duration_s"],
        "duration_steps": force_block["duration_steps"],
        "control_dt_s": force_block["control_dt_s"],
        "body_name": force_block["body_name"],
        "application_point": force_block["application_point"],
    }


def _validate_push_robot_event_contract(contract: Mapping) -> None:
    """Wave-P random base-push block (task.push; PACE/BeyondMimic push, HITTER default = none).

    Absent block = push disabled (every historical/no-push run, byte-identical contract).  A
    present block is always an ENABLED push and must be internally consistent: its stored
    velocity box must equal the canonical re-assembly from its own amplitudes/axes, so a
    hand-edited or drifted sidecar cannot smuggle a different push recipe past a resume.

    v1 flag 语义(合并互斥推):legacy 块不带 ``combined_exclusive`` 键、按下面的原逻辑逐字节
    校验;带 ``combined_exclusive`` 键的块按合并拼写走 :func:`_validate_push_combined_event_block`
    (键面必须一个不多一个不少,含力分支同冲量记账)。两种拼写之外的键面一律 fail-loud。
    """

    block = contract.get(PUSH_ROBOT_EVENT_KEY)
    if block is None:
        if PUSH_ROBOT_EVENT_KEY in contract:
            raise ValueError(
                "schema-3 push_robot_event must be omitted when disabled, not null"
            )
        return
    if isinstance(block, Mapping) and "combined_exclusive" in block:
        _validate_push_combined_event_block(block)
        return
    block = _require_exact_mapping_keys(
        block, _PUSH_ROBOT_EVENT_KEYS, name="schema-3 push_robot_event"
    )
    if type(block["schema_version"]) is not int or block["schema_version"] != 1:
        raise ValueError("schema-3 push_robot_event schema_version must be integer 1")
    if block["enabled"] is not True:
        raise ValueError(
            "schema-3 push_robot_event enabled must be true "
            "(a disabled push is spelled by omitting the block)"
        )
    if block["func"] != PUSH_ROBOT_EVENT_FUNC:
        raise ValueError(
            f"schema-3 push_robot_event func must be {PUSH_ROBOT_EVENT_FUNC!r}"
        )
    if block["mode"] != PUSH_ROBOT_EVENT_MODE:
        raise ValueError(
            f"schema-3 push_robot_event mode must be {PUSH_ROBOT_EVENT_MODE!r}"
        )
    try:
        expected = push_robot_event_block(
            enable=True,
            interval_range_s=block["interval_range_s"],
            vel_xy_mps=block["vel_xy_mps"],
            ang_vel_radps=block["ang_vel_radps"],
            ang_axes=block["ang_axes"],
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 push_robot_event is invalid: {exc}") from exc
    stored_range = block["velocity_range"]
    if not isinstance(stored_range, Mapping):
        raise ValueError("schema-3 push_robot_event velocity_range must be an object")
    normalized_range = {}
    for axis, rng in stored_range.items():
        if (
            not isinstance(rng, (list, tuple))
            or len(rng) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in rng)
        ):
            raise ValueError(
                f"schema-3 push_robot_event velocity_range.{axis} must be a [lo, hi] pair"
            )
        normalized_range[str(axis)] = [float(rng[0]), float(rng[1])]
    stored_interval = [float(v) for v in block["interval_range_s"]]
    if (
        normalized_range != expected["velocity_range"]
        or stored_interval != expected["interval_range_s"]
    ):
        raise ValueError(
            "schema-3 push_robot_event is internally inconsistent: the stored "
            "velocity_range/interval does not equal the canonical assembly from "
            "vel_xy_mps/ang_vel_radps/ang_axes"
        )


def _validate_push_combined_event_block(block) -> None:
    """merged-exclusive push 块(``combined_exclusive`` 键在场 = 按合并拼写校验)。

    人话:合并互斥推的合同块必须与自己的 canonical 重装配逐位一致——速度箱、触发区间、
    duration_steps、force_prob 有一个漂了就拒收;力分支的同冲量记账(robot_mass_kg /
    delta_v_equiv_mps = F·Δt/m 重算)照 force_push_event 同款硬核对,手改 sidecar 走不过
    resume。combined_exclusive=false 的合并块是非法拼写(合并关着 = legacy 块,不带这个键)。
    """

    block = _require_exact_mapping_keys(
        block, _PUSH_COMBINED_EVENT_KEYS,
        name="schema-3 push_robot_event (combined_exclusive spelling)",
    )
    if type(block["schema_version"]) is not int or block["schema_version"] != 1:
        raise ValueError("schema-3 push_robot_event schema_version must be integer 1")
    if block["enabled"] is not True:
        raise ValueError(
            "schema-3 push_robot_event enabled must be true "
            "(a disabled push is spelled by omitting the block)"
        )
    if block["combined_exclusive"] is not True:
        raise ValueError(
            "schema-3 push_robot_event combined_exclusive must be true — a merged "
            "block carries the key only when the merged sampler is ON (combined off "
            "is spelled by the legacy block WITHOUT the key)"
        )
    if block["func"] != PUSH_COMBINED_EVENT_FUNC:
        raise ValueError(
            f"schema-3 combined push_robot_event func must be {PUSH_COMBINED_EVENT_FUNC!r}"
        )
    if block["mode"] != PUSH_ROBOT_EVENT_MODE:
        raise ValueError(
            f"schema-3 combined push_robot_event mode must be {PUSH_ROBOT_EVENT_MODE!r}"
        )
    if block["body_name"] != FORCE_PUSH_BODY_NAME:
        raise ValueError(
            f"schema-3 combined push_robot_event body_name must be {FORCE_PUSH_BODY_NAME!r}"
        )
    if block["application_point"] != FORCE_PUSH_APPLICATION_POINT:
        raise ValueError(
            "schema-3 combined push_robot_event application_point must be "
            f"{FORCE_PUSH_APPLICATION_POINT!r} — the wrench lands on the pelvis LINK "
            "ORIGIN, and labelling it as the COM is exactly the Yikang V9 mistake"
        )
    try:
        expected = push_robot_event_block(
            enable=True,
            interval_range_s=block["interval_range_s"],
            vel_xy_mps=block["vel_xy_mps"],
            ang_vel_radps=block["ang_vel_radps"],
            ang_axes=block["ang_axes"],
            combined_exclusive=True,
            force_prob=block["force_prob"],
            force_n=block["force_n"],
            duration_s=block["duration_s"],
            control_dt_s=block["control_dt_s"],
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 push_robot_event is invalid: {exc}") from exc
    stored_range = block["velocity_range"]
    if not isinstance(stored_range, Mapping):
        raise ValueError("schema-3 push_robot_event velocity_range must be an object")
    normalized_range = {}
    for axis, rng in stored_range.items():
        if (
            not isinstance(rng, (list, tuple))
            or len(rng) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in rng)
        ):
            raise ValueError(
                f"schema-3 push_robot_event velocity_range.{axis} must be a [lo, hi] pair"
            )
        normalized_range[str(axis)] = [float(rng[0]), float(rng[1])]
    stored_interval = [float(v) for v in block["interval_range_s"]]
    if (
        normalized_range != expected["velocity_range"]
        or stored_interval != expected["interval_range_s"]
        or type(block["duration_steps"]) is not int
        or block["duration_steps"] != expected["duration_steps"]
    ):
        raise ValueError(
            "schema-3 push_robot_event is internally inconsistent: the stored "
            "velocity_range/interval/duration_steps does not equal the canonical "
            "assembly from its own amplitudes/axes/duration_s/control_dt_s"
        )
    try:
        expected_full = bind_force_push_runtime_mass(
            expected, robot_mass_kg=block["robot_mass_kg"]
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 push_robot_event is invalid: {exc}") from exc
    if (
        isinstance(block["delta_v_equiv_mps"], bool)
        or not isinstance(block["delta_v_equiv_mps"], (int, float))
        or float(block["delta_v_equiv_mps"]) != expected_full["delta_v_equiv_mps"]
    ):
        raise ValueError(
            "schema-3 push_robot_event is internally inconsistent: delta_v_equiv_mps "
            "must equal force_n * duration_s / robot_mass_kg recomputed (力分支单次"
            "冲量与速度推档位对表)"
        )


# --------------------------------------------------------------------------------------------- #
# F-axis interval FORCE push (matched-impulse companion of the Wave-P velocity push; default OFF).
# --------------------------------------------------------------------------------------------- #
FORCE_PUSH_EVENT_KEY = "force_push_event"
FORCE_PUSH_EVENT_FUNC = "push_by_applying_wrench"
FORCE_PUSH_SWEEP_FUNC = "sweep_expired_force_pushes"
FORCE_PUSH_EVENT_MODE = "interval"
FORCE_PUSH_BODY_NAME = "pelvis_link"
# 施力点语义显式化(Yikang V9 教训:把 link 原点标成 COM):PhysX 在 positions=None 时把外力
# 施加在 link 原点,合同必须诚实写 "pelvis_link_origin",不许标 COM。
FORCE_PUSH_APPLICATION_POINT = "pelvis_link_origin"
_FORCE_PUSH_ASSEMBLY_KEYS = frozenset(
    {
        "schema_version", "enabled", "func", "mode", "interval_range_s",
        "force_n", "duration_s", "duration_steps", "control_dt_s",
        "body_name", "application_point",
    }
)
_FORCE_PUSH_EVENT_KEYS = _FORCE_PUSH_ASSEMBLY_KEYS | {
    "robot_mass_kg", "delta_v_equiv_mps",
}


def force_push_event_block(
    *, enable, interval_range_s, force_n, duration_s, control_dt_s
):
    """Translate the force-push flag group into the canonical assembly block (no runtime mass yet).

    人话:把"要不要力推、隔几秒推一次、推多少牛、持续几秒"翻译成 force_push 事件的合同装配块。
    ``enable=False`` 返回 ``None``(= 不推,合同不写这个块,历史/在跑配置逐字节不变),但此时
    非零 force_n 是配置错误(fail-closed:关着的开关不许挂上膛的力)。``duration_s`` 必须是
    控制步长的整数倍(恒力持续整数个控制步,0.30 s = 15 步 @ 50 Hz),否则 fail-loud。This is
    the single validation/assembly source shared by the env cfg flag path
    (hope_env_cfg.apply_force_push_event), the train.py ``task.force_push`` override, and the
    schema-3 contract validator. 运行时质量与 Δv_equiv 由 :func:`bind_force_push_runtime_mass`
    在拿到真实 articulation 质量后追加。
    """

    if not isinstance(enable, bool):
        raise ValueError("force_push_event enable must be an explicit boolean")
    if not enable:
        if force_n is not None and (isinstance(force_n, bool) or float(force_n) != 0.0):
            raise ValueError(
                f"force_push_event disabled but force_n={force_n!r} is nonzero — "
                "delete the dormant force or set enable=true"
            )
        return None
    try:
        interval_items = list(interval_range_s)
    except TypeError as exc:
        raise ValueError(
            "force_push_event interval_range_s must be a [lo, hi] pair of seconds"
        ) from exc
    if len(interval_items) != 2:
        raise ValueError(
            "force_push_event interval_range_s must be a [lo, hi] pair of seconds"
        )
    interval_lo = _wave_finite(
        interval_items[0], name="force_push_event.interval_range_s[0]", positive=True
    )
    interval_hi = _wave_finite(
        interval_items[1], name="force_push_event.interval_range_s[1]", positive=True
    )
    if interval_lo > interval_hi:
        raise ValueError(
            "force_push_event interval_range_s must satisfy 0 < lo <= hi"
        )
    force = _wave_finite(force_n, name="force_push_event.force_n", positive=True)
    duration = _wave_finite(
        duration_s, name="force_push_event.duration_s", positive=True
    )
    control_dt = _wave_finite(
        control_dt_s, name="force_push_event.control_dt_s", positive=True
    )
    duration_steps = int(round(duration / control_dt))
    if duration_steps < 1 or abs(duration_steps * control_dt - duration) > 1e-9:
        raise ValueError(
            "force_push_event duration_s must be a positive whole number of control "
            f"steps (duration_s={duration!r} is not an integer multiple of "
            f"control_dt_s={control_dt!r})"
        )
    return {
        "schema_version": 1,
        "enabled": True,
        "func": FORCE_PUSH_EVENT_FUNC,
        "mode": FORCE_PUSH_EVENT_MODE,
        "interval_range_s": [interval_lo, interval_hi],
        "force_n": force,
        "duration_s": duration,
        "duration_steps": duration_steps,
        "control_dt_s": control_dt,
        "body_name": FORCE_PUSH_BODY_NAME,
        "application_point": FORCE_PUSH_APPLICATION_POINT,
    }


def bind_force_push_runtime_mass(block, *, robot_mass_kg):
    """Append the runtime articulation mass + matched-impulse Δv to an assembly block.

    人话:合同必须记运行时真实读到的机器人总质量与换算出的 Δv_equiv = force_n × duration_s /
    m_robot,供与速度推档位(p02/p035/p05/p08)对表。装配块形状不对、质量非正,一律 raise。
    合并互斥推的装配块(_PUSH_COMBINED_ASSEMBLY_KEYS)同样可绑——记的是【力分支单次】的
    Δv_equiv(速度分支的 Δv 直接就是 vel_xy_mps,无需换算)。
    """

    if not isinstance(block, Mapping) or (
        set(block) != _FORCE_PUSH_ASSEMBLY_KEYS
        and set(block) != _PUSH_COMBINED_ASSEMBLY_KEYS
    ):
        raise ValueError(
            "force_push_event runtime binding requires the exact canonical assembly block "
            "(legacy force_push or combined_exclusive push)"
        )
    if block.get("enabled") is not True:
        raise ValueError("force_push_event runtime binding requires an enabled block")
    mass = _wave_finite(
        robot_mass_kg, name="force_push_event.robot_mass_kg", positive=True
    )
    delta_v = float(block["force_n"]) * float(block["duration_s"]) / mass
    return {**block, "robot_mass_kg": mass, "delta_v_equiv_mps": delta_v}


def _validate_force_push_event_contract(contract: Mapping) -> None:
    """F-axis interval force-push block (task.force_push; matched-impulse vs the velocity push).

    Absent block = force push disabled (every historical/no-push run, byte-identical contract).
    A present block is always an ENABLED push and must be internally consistent: interval /
    duration_steps must equal the canonical re-assembly from its own fields, and the recorded
    Δv_equiv must equal force_n x duration_s / robot_mass_kg recomputed — a hand-edited sidecar
    cannot smuggle a different impulse past a resume. ``application_point`` 必须是
    ``pelvis_link_origin``(Yikang V9 反例:link 原点被标成 COM)。
    """

    block = contract.get(FORCE_PUSH_EVENT_KEY)
    if block is None:
        if FORCE_PUSH_EVENT_KEY in contract:
            raise ValueError(
                "schema-3 force_push_event must be omitted when disabled, not null"
            )
        return
    block = _require_exact_mapping_keys(
        block, _FORCE_PUSH_EVENT_KEYS, name="schema-3 force_push_event"
    )
    if type(block["schema_version"]) is not int or block["schema_version"] != 1:
        raise ValueError("schema-3 force_push_event schema_version must be integer 1")
    if block["enabled"] is not True:
        raise ValueError(
            "schema-3 force_push_event enabled must be true "
            "(a disabled force push is spelled by omitting the block)"
        )
    if block["func"] != FORCE_PUSH_EVENT_FUNC:
        raise ValueError(
            f"schema-3 force_push_event func must be {FORCE_PUSH_EVENT_FUNC!r}"
        )
    if block["mode"] != FORCE_PUSH_EVENT_MODE:
        raise ValueError(
            f"schema-3 force_push_event mode must be {FORCE_PUSH_EVENT_MODE!r}"
        )
    if block["body_name"] != FORCE_PUSH_BODY_NAME:
        raise ValueError(
            f"schema-3 force_push_event body_name must be {FORCE_PUSH_BODY_NAME!r}"
        )
    if block["application_point"] != FORCE_PUSH_APPLICATION_POINT:
        raise ValueError(
            "schema-3 force_push_event application_point must be "
            f"{FORCE_PUSH_APPLICATION_POINT!r} — the wrench lands on the pelvis LINK "
            "ORIGIN, and labelling it as the COM is exactly the Yikang V9 mistake"
        )
    try:
        expected = force_push_event_block(
            enable=True,
            interval_range_s=block["interval_range_s"],
            force_n=block["force_n"],
            duration_s=block["duration_s"],
            control_dt_s=block["control_dt_s"],
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 force_push_event is invalid: {exc}") from exc
    stored_interval = [float(v) for v in block["interval_range_s"]]
    if (
        stored_interval != expected["interval_range_s"]
        or type(block["duration_steps"]) is not int
        or block["duration_steps"] != expected["duration_steps"]
    ):
        raise ValueError(
            "schema-3 force_push_event is internally inconsistent: the stored "
            "interval/duration_steps does not equal the canonical assembly from "
            "duration_s/control_dt_s"
        )
    try:
        expected_full = bind_force_push_runtime_mass(
            expected, robot_mass_kg=block["robot_mass_kg"]
        )
    except ValueError as exc:
        raise ValueError(f"schema-3 force_push_event is invalid: {exc}") from exc
    if (
        isinstance(block["delta_v_equiv_mps"], bool)
        or not isinstance(block["delta_v_equiv_mps"], (int, float))
        or float(block["delta_v_equiv_mps"]) != expected_full["delta_v_equiv_mps"]
    ):
        raise ValueError(
            "schema-3 force_push_event is internally inconsistent: delta_v_equiv_mps "
            "must equal force_n * duration_s / robot_mass_kg recomputed"
        )


def _validate_post_swing_settle_debt_contract(contract: Mapping) -> None:
    """S1 post-swing settle debt block (Jiayi V13 post-swing debts idea, clean main-side redo).

    Absent block = mechanism untouched (legacy/default runs).  A present block must bind the
    exact five-debt formula, the shared same-attempt recovery-window gate, and every margin/scale
    the run trained with; it is mutually exclusive with an enabled Wave-B lower-body mechanism.
    """

    settle = contract.get("post_swing_settle_debt_reward")
    if settle is None:
        return
    settle = _require_exact_mapping_keys(
        settle,
        frozenset(
            {
                "schema_version", "enabled", "probe_enabled", "activation_ledger",
                "weight", "base_lin_margin_mps", "base_lin_scale_mps",
                "base_ang_margin_radps", "base_ang_scale_radps",
                "tilt_margin_rad", "tilt_scale_rad",
                "nominal_root_z_m", "root_height_deadband_m", "root_height_scale_m",
                "foot_slip_margin_mps", "foot_slip_scale_mps",
                "recovery_start_s", "recovery_end_s",
                "racket_command_name", "motion_command_name", "foot_body_names",
                "components", "formula", "gate", "age_source",
                "success_conditioned", "uses_motion_reference",
            }
        ),
        name="schema-3 post_swing_settle_debt_reward",
    )
    if type(settle["schema_version"]) is not int or settle["schema_version"] != 1:
        raise ValueError("schema-3 post_swing_settle_debt_reward schema_version must be 1")
    if (
        not isinstance(settle["enabled"], bool)
        or settle["probe_enabled"] is not True
        or settle["success_conditioned"] is not False
        or settle["uses_motion_reference"] is not False
    ):
        raise ValueError("schema-3 post_swing_settle_debt_reward flags are invalid")
    weight = _wave_finite(settle["weight"], name="post_swing_settle_debt_reward.weight")
    if weight > 0.0 or settle["enabled"] != (weight < 0.0):
        raise ValueError("schema-3 post_swing_settle_debt_reward weight/enabled is invalid")
    for name in (
        "base_lin_scale_mps", "base_ang_scale_radps", "tilt_scale_rad",
        "nominal_root_z_m", "root_height_scale_m", "foot_slip_scale_mps",
    ):
        _wave_finite(settle[name], name=f"post_swing_settle_debt_reward.{name}", positive=True)
    for name in (
        "base_lin_margin_mps", "base_ang_margin_radps", "tilt_margin_rad",
        "root_height_deadband_m", "foot_slip_margin_mps", "recovery_start_s",
    ):
        _wave_finite(
            settle[name], name=f"post_swing_settle_debt_reward.{name}", nonnegative=True
        )
    end = _wave_finite(
        settle["recovery_end_s"],
        name="post_swing_settle_debt_reward.recovery_end_s",
        positive=True,
    )
    if float(settle["recovery_start_s"]) >= end:
        raise ValueError(
            "schema-3 post_swing_settle_debt_reward recovery window must satisfy 0 <= start < end"
        )
    expected = {
        "racket_command_name": "racket_target",
        "motion_command_name": "motion",
        "activation_ledger": "weight_independent_control_step_counters",
        "foot_body_names": ["left_ankle_roll_Link", "right_ankle_roll_Link"],
        "components": [
            "base_quiet_lin",
            "base_quiet_ang",
            "tilt_debt",
            "root_height_debt",
            "settle_foot_slip",
        ],
        "formula": "mean(5x(1-exp(-square(relu(x-margin)/scale))))",
        "gate": "same_attempt_post_strike_age_s_inclusive",
        "age_source": "per_env_exact_strike_control_tick_latch",
    }
    for key, value in expected.items():
        if settle[key] != value:
            raise ValueError(
                f"schema-3 post_swing_settle_debt_reward {key} must be {value!r}"
            )
    articulation_bodies = contract.get("articulation_body_names")
    if (
        not isinstance(articulation_bodies, (list, tuple))
        or any(name not in articulation_bodies for name in settle["foot_body_names"])
    ):
        raise ValueError(
            "schema-3 post_swing_settle_debt_reward foot bodies are absent from articulation"
        )
    if settle["enabled"]:
        for other_key in (
            "lower_body_pose_imitation_reward",
            "lower_body_stability_bundle_reward",
        ):
            other = contract.get(other_key)
            if isinstance(other, Mapping) and other.get("enabled") is True:
                raise ValueError(
                    "schema-3 S1 post_swing_settle_debt and Wave-B lower-body rewards are "
                    "mutually exclusive"
                )


def _validate_qdes_limit_barrier_contract(contract: Mapping) -> None:
    """Wave-Q all-joint q_des position-limit barrier block (Jiayi V14 idea, top-k removed).

    Absent block = mechanism untouched (legacy/default runs).  A present block must bind the
    exact dense all-31-joint formula, the margin fraction, and the deploy-parity position-limit
    source; no top-k and no joint subset are representable.
    """

    barrier = contract.get("qdes_limit_barrier_reward")
    if barrier is None:
        if contract.get("actual_joint_limit_barrier_reward") is not None:
            raise ValueError(
                "actual_joint_limit_barrier_reward requires a schema-2 "
                "qdes_limit_barrier_reward"
            )
        return
    if not isinstance(barrier, Mapping):
        raise ValueError("schema-3 qdes_limit_barrier_reward must be an object")
    schema = barrier.get("schema_version")
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError(
            "schema-3 qdes_limit_barrier_reward schema_version must be 1 or 2"
        )
    legacy_keys = frozenset(
        {
            "schema_version", "enabled", "probe_enabled", "activation_ledger",
            "weight", "margin_frac", "action_name", "joint_count",
            "joint_order", "position_limit_source", "formula", "gate",
        }
    )
    v2_keys = frozenset(
        {
            "schema_version", "enabled", "probe_enabled", "term_name",
            "probe_term_name", "term_callable", "probe_callable",
            "activation_ledger", "weight", "margin_frac", "penalty_floor",
            "shape_rate", "stance_eps", "margin_floor", "action_name",
            "joint_count", "joint_order", "position_source",
            "position_limit_source", "default_stance_source", "formula",
            "aggregation", "per_joint_cap", "gate",
        }
    )
    if schema == 2 and set(barrier) == set(legacy_keys):
        raise ValueError(
            "schema-3 qdes_limit_barrier_reward schema_version 2 cannot use "
            "the legacy schema-1 payload"
        )
    barrier = _require_exact_mapping_keys(
        barrier,
        legacy_keys if schema == 1 else v2_keys,
        name="schema-3 qdes_limit_barrier_reward",
    )
    if not isinstance(barrier["enabled"], bool) or barrier["probe_enabled"] is not True:
        raise ValueError("schema-3 qdes_limit_barrier_reward flags are invalid")
    weight = _wave_finite(barrier["weight"], name="qdes_limit_barrier_reward.weight")
    if weight > 0.0 or barrier["enabled"] != (weight < 0.0):
        raise ValueError("schema-3 qdes_limit_barrier_reward weight/enabled is invalid")
    margin_frac = _wave_finite(
        barrier["margin_frac"], name="qdes_limit_barrier_reward.margin_frac"
    )
    if not 0.0 < margin_frac < 0.5:
        raise ValueError(
            "schema-3 qdes_limit_barrier_reward margin_frac must be in (0, 0.5)"
        )
    joint_names = contract.get("joint_names")
    articulation_names = contract.get("articulation_joint_names")
    if (
        type(barrier["joint_count"]) is not int
        or barrier["joint_count"] != 31
        or not isinstance(joint_names, (list, tuple))
        or len(joint_names) != 31
        or not isinstance(articulation_names, (list, tuple))
        or [str(value) for value in articulation_names]
        != [str(value) for value in joint_names]
    ):
        raise ValueError(
            "schema-3 qdes_limit_barrier_reward requires identity 31-joint order"
        )
    if schema == 1:
        expected = {
            "action_name": "joint_pos",
            "activation_ledger": "weight_independent_control_step_counters",
            "joint_order": "runtime_articulation_identity",
            "position_limit_source": "articulation.data.soft_joint_pos_limits",
            "formula": (
                # 2026-07-25 站姿豁免:与 train.py _QDES_LIMIT_BARRIER_FORMULA 逐字节一致;
                # 旧公式的 sidecar 在此 fail loud —— 数学变了就不许静默续训。
                "sum(1-exp(-square(relu(m_eff-min(qdes-lo,hi-qdes)/(hi-lo))/m_eff)));"
                "m_eff=min(margin_frac,min(default_q-lo,hi-default_q)/(hi-lo)-0.005)"
            ),
            "gate": "dense_every_control_step",
        }
    else:
        penalty_floor = _wave_finite(
            barrier["penalty_floor"],
            name="qdes_limit_barrier_reward.penalty_floor",
        )
        if not 0.0 < penalty_floor < 1.0:
            raise ValueError(
                "schema-3 qdes_limit_barrier_reward penalty_floor must be in (0, 1)"
            )
        expected = {
            "term_name": "qdes_limit_barrier",
            "probe_term_name": "qdes_limit_barrier_probe",
            "term_callable": (
                "whole_body_tracking.tasks.tracking.mdp."
                "qdes_limit_barrier_v2"
            ),
            "probe_callable": (
                "whole_body_tracking.tasks.tracking.mdp."
                "qdes_limit_barrier_v2_probe"
            ),
            "action_name": "joint_pos",
            "activation_ledger": "weight_independent_control_step_counters",
            "shape_rate": 4.0,
            "stance_eps": 0.005,
            "margin_floor": 0.005,
            "joint_order": "runtime_articulation_identity",
            "position_source": "joint_pos.processed_actions",
            "position_limit_source": "articulation.data.soft_joint_pos_limits",
            "default_stance_source": "articulation.data.default_joint_pos",
            "formula": (
                "sum(where(u>0,penalty_floor+(1-penalty_floor)*"
                "(1-exp(-shape_rate*clamp(u,0,1)))/(1-exp(-shape_rate)),0));"
                "u=relu(m_eff-min(q-lo,hi-q)/(hi-lo))/m_eff;"
                "m_eff=min(margin_frac,min(default_q-lo,hi-default_q)/(hi-lo)-stance_eps);"
                "require_all(m_eff>margin_floor)"
            ),
            "aggregation": "sum_all_31_joints",
            "per_joint_cap": 1.0,
            "gate": "dense_every_control_step",
        }
    for key, value in expected.items():
        if barrier[key] != value:
            raise ValueError(
                f"schema-3 qdes_limit_barrier_reward {key} must be exactly {value!r}"
            )
    actual = contract.get("actual_joint_limit_barrier_reward")
    if schema == 1:
        if actual is not None:
            raise ValueError(
                "schema-1 qdes_limit_barrier_reward cannot bind an actual-q "
                "schema-2 block"
            )
        return
    if actual is None:
        raise ValueError(
            "schema-2 qdes_limit_barrier_reward requires the independent "
            "actual_joint_limit_barrier_reward block"
        )
    actual_keys = frozenset(
        {
            "schema_version", "enabled", "probe_enabled", "term_name",
            "probe_term_name", "term_callable", "probe_callable",
            "activation_ledger", "weight", "margin_frac", "penalty_floor",
            "shape_rate", "stance_eps", "margin_floor", "asset_name",
            "joint_count", "joint_order", "position_source",
            "position_limit_source", "default_stance_source", "formula",
            "aggregation", "per_joint_cap", "gate",
        }
    )
    actual = _require_exact_mapping_keys(
        actual,
        actual_keys,
        name="schema-3 actual_joint_limit_barrier_reward",
    )
    if actual["schema_version"] != 2 or type(actual["schema_version"]) is not int:
        raise ValueError(
            "schema-3 actual_joint_limit_barrier_reward schema_version must be 2"
        )
    if not isinstance(actual["enabled"], bool) or actual["probe_enabled"] is not True:
        raise ValueError(
            "schema-3 actual_joint_limit_barrier_reward flags are invalid"
        )
    actual_weight = _wave_finite(
        actual["weight"], name="actual_joint_limit_barrier_reward.weight"
    )
    if actual_weight > 0.0 or actual["enabled"] != (actual_weight < 0.0):
        raise ValueError(
            "schema-3 actual_joint_limit_barrier_reward weight/enabled is invalid"
        )
    actual_margin = _wave_finite(
        actual["margin_frac"],
        name="actual_joint_limit_barrier_reward.margin_frac",
    )
    actual_floor = _wave_finite(
        actual["penalty_floor"],
        name="actual_joint_limit_barrier_reward.penalty_floor",
    )
    if not 0.0 < actual_margin < 0.5 or not 0.0 < actual_floor < 1.0:
        raise ValueError(
            "schema-3 actual_joint_limit_barrier_reward parameters are invalid"
        )
    actual_expected = {
        "term_name": "joint_limit",
        "probe_term_name": "actual_joint_limit_barrier_probe",
        "term_callable": (
            "whole_body_tracking.tasks.tracking.mdp."
            "actual_joint_limit_barrier_v2"
        ),
        "probe_callable": (
            "whole_body_tracking.tasks.tracking.mdp."
            "actual_joint_limit_barrier_v2_probe"
        ),
        "activation_ledger": "weight_independent_control_step_counters",
        "shape_rate": 4.0,
        "stance_eps": 0.005,
        "margin_floor": 0.005,
        "asset_name": "robot",
        "joint_count": 31,
        "joint_order": "runtime_articulation_identity",
        "position_source": "articulation.data.joint_pos",
        "position_limit_source": "articulation.data.soft_joint_pos_limits",
        "default_stance_source": "articulation.data.default_joint_pos",
        "formula": expected["formula"],
        "aggregation": "sum_all_31_joints",
        "per_joint_cap": 1.0,
        "gate": "dense_every_control_step",
    }
    for key, value in actual_expected.items():
        if actual[key] != value:
            raise ValueError(
                "schema-3 actual_joint_limit_barrier_reward "
                f"{key} must be exactly {value!r}"
            )
    for key in ("weight", "margin_frac", "penalty_floor"):
        if actual[key] != barrier[key]:
            raise ValueError(
                "schema-3 qdes/actual soft-limit barrier v2 "
                f"{key} must match exactly"
            )


def _optional_exact_bool(mapping: Mapping, key: str, *, name: str) -> bool:
    """Read an optional boolean without accepting JSON truthy lookalikes."""

    if key not in mapping:
        return False
    value = mapping[key]
    if type(value) is not bool:
        raise ValueError(f"{name}.{key} must be an exact boolean")
    return value


def _action_ball_bootstrap_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _action_ball_bootstrap_float_vector(
    value: object, *, name: str, expected: int
) -> list[float]:
    if (
        not isinstance(value, (list, tuple))
        or isinstance(value, (str, bytes))
        or len(value) != expected
    ):
        raise ValueError(f"{name} must contain exactly {expected} numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{name}[{index}] must be a finite number")
        result.append(float(item))
    return result


def action_ball_shared_ready_sha256(
    *,
    action_order: list[str] | tuple[str, ...],
    joint_names: list[str] | tuple[str, ...],
    shared_ready_joint_pos: list[float] | tuple[float, ...],
) -> str:
    """Hash the exact action order, joint order and shared ready vector.

    This digest is deliberately independent of filesystem locations. The
    enclosing training contract separately binds every motion byte and the
    canonical-ready registry digests.
    """

    document = {
        "schema_version": 1,
        "semantics": "motion.joint_pos[motion.seg_start[action_slot]]",
        "action_order": list(action_order),
        "joint_names": list(joint_names),
        "shared_ready_joint_pos": list(shared_ready_joint_pos),
    }
    try:
        encoded = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("shared-ready digest input is not finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def action_ball_dynamic_ready_binding_sha256(value: Mapping) -> str:
    """Hash one runtime binding without its self-authenticating digest."""

    if not isinstance(value, Mapping):
        raise ValueError("action-ball dynamic-ready binding must be an object")
    unsigned = dict(value)
    unsigned.pop("binding_sha256", None)
    return _action_ball_canonical_sha256(unsigned)


def _validate_action_ball_dynamic_ready_physical_ready(
    value: object, *, name: str
) -> dict:
    physical = _require_exact_mapping_keys(
        value,
        _ACTION_BALL_DYNAMIC_READY_PHYSICAL_KEYS,
        name=name,
    )
    root_pos = _action_ball_bootstrap_float_vector(
        physical["root_pos_w_m"], name=f"{name}.root_pos_w_m", expected=3
    )
    root_quat = _action_ball_bootstrap_float_vector(
        physical["root_quat_wxyz"],
        name=f"{name}.root_quat_wxyz",
        expected=4,
    )
    quat_norm = math.sqrt(sum(component * component for component in root_quat))
    if not math.isclose(quat_norm, 1.0, rel_tol=0.0, abs_tol=2.0e-5):
        raise ValueError(f"{name}.root_quat_wxyz must be a unit quaternion")
    joint_pos = _action_ball_bootstrap_float_vector(
        physical["joint_pos_rad"],
        name=f"{name}.joint_pos_rad",
        expected=31,
    )
    joint_vel = _action_ball_bootstrap_float_vector(
        physical["joint_vel_radps"],
        name=f"{name}.joint_vel_radps",
        expected=31,
    )
    if any(value != 0.0 for value in joint_vel):
        raise ValueError(f"{name}.joint_vel_radps must be exact zeros")
    return {
        "root_pos_w_m": root_pos,
        "root_quat_wxyz": root_quat,
        "joint_pos_rad": joint_pos,
        "joint_vel_radps": joint_vel,
    }


def _validate_action_ball_dynamic_ready_pin(
    value: object, *, name: str
) -> dict:
    pin = _require_exact_mapping_keys(
        value, _ACTION_BALL_DYNAMIC_READY_PIN_KEYS, name=name
    )
    path = pin["path"]
    if (
        type(path) is not str
        or not path
        or not Path(path).is_absolute()
    ):
        raise ValueError(f"{name}.path must be one non-empty absolute path")
    return {
        "path": path,
        "sha256": _action_ball_bootstrap_sha256(
            pin["sha256"], name=f"{name}.sha256"
        ),
        "content_sha256": _action_ball_bootstrap_sha256(
            pin["content_sha256"], name=f"{name}.content_sha256"
        ),
    }


def validate_action_ball_dynamic_ready_runtime_binding(
    value: object, *, expected_action_count: int | None = None
) -> dict:
    """Validate the path-pinned ActionBall dynamic-ready runtime binding."""

    binding = _require_exact_mapping_keys(
        value,
        _ACTION_BALL_DYNAMIC_READY_BINDING_KEYS,
        name="action-ball dynamic-ready runtime binding",
    )
    if (
        type(binding["schema_version"]) is not int
        or binding["schema_version"] != 1
        or binding["kind"] != ACTION_BALL_DYNAMIC_READY_RUNTIME_BINDING_KIND
    ):
        raise ValueError(
            "action-ball dynamic-ready runtime binding must use schema 1"
        )
    action_order = binding["action_order"]
    if (
        not isinstance(action_order, (list, tuple))
        or len(action_order) != 1
        or any(type(item) is not str or not item for item in action_order)
    ):
        raise ValueError(
            "action-ball dynamic-ready runtime binding currently supports exact N=1"
        )
    if expected_action_count is not None and expected_action_count != 1:
        raise ValueError(
            "action-ball dynamic-ready runtime binding disagrees with actor count"
        )
    motion_digests = binding["motion_sha256_per_action"]
    if not isinstance(motion_digests, (list, tuple)) or len(motion_digests) != 1:
        raise ValueError(
            "action-ball dynamic-ready runtime binding requires one motion SHA"
        )
    motion_sha = _action_ball_bootstrap_sha256(
        motion_digests[0],
        name="action-ball dynamic-ready runtime binding motion SHA",
    )
    rows = binding["rows"]
    if not isinstance(rows, (list, tuple)) or len(rows) != 1:
        raise ValueError(
            "action-ball dynamic-ready runtime binding requires one ordered row"
        )
    row = _require_exact_mapping_keys(
        rows[0],
        _ACTION_BALL_DYNAMIC_READY_ROW_KEYS,
        name="action-ball dynamic-ready runtime binding row",
    )
    if row["action_id"] != action_order[0]:
        raise ValueError(
            "action-ball dynamic-ready row order disagrees with action_order"
        )
    physical = _validate_action_ball_dynamic_ready_physical_ready(
        row["physical_ready"],
        name="action-ball dynamic-ready runtime binding physical_ready",
    )
    hold_qdes = _action_ball_bootstrap_float_vector(
        row["hold_qdes_joint_pos_rad"],
        name="action-ball dynamic-ready runtime binding hold_qdes_joint_pos_rad",
        expected=31,
    )
    normalized = _action_ball_bootstrap_float_vector(
        row["normalized_actor_action"],
        name="action-ball dynamic-ready runtime binding normalized_actor_action",
        expected=31,
    )
    artifact_pin = _validate_action_ball_dynamic_ready_pin(
        row["artifact"], name="action-ball dynamic-ready artifact pin"
    )
    receipt_pin = _validate_action_ball_dynamic_ready_pin(
        row["nominal_hold_receipt"],
        name="action-ball dynamic-ready nominal-hold receipt pin",
    )
    actual_binding_sha = action_ball_dynamic_ready_binding_sha256(binding)
    expected_binding_sha = _action_ball_bootstrap_sha256(
        binding["binding_sha256"],
        name="action-ball dynamic-ready runtime binding SHA",
    )
    if actual_binding_sha != expected_binding_sha:
        raise ValueError(
            "action-ball dynamic-ready runtime binding SHA is not reproducible"
        )
    return {
        "schema_version": 1,
        "kind": ACTION_BALL_DYNAMIC_READY_RUNTIME_BINDING_KIND,
        "binding_sha256": expected_binding_sha,
        "action_order": list(action_order),
        "motion_sha256_per_action": [motion_sha],
        "rows": [
            {
                "action_id": str(row["action_id"]),
                "physical_ready": physical,
                "hold_qdes_joint_pos_rad": hold_qdes,
                "normalized_actor_action": normalized,
                "artifact": artifact_pin,
                "nominal_hold_receipt": receipt_pin,
            }
        ],
    }


def _strict_action_ball_json_bytes(payload: bytes, *, name: str) -> dict:
    """Decode one finite JSON object while rejecting duplicate object keys."""

    def _pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = item
        return result

    def _constant(token):
        raise ValueError(f"{name} contains non-finite JSON token {token!r}")

    try:
        decoded = payload.decode("utf-8", "strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain one JSON object")
    return dict(value)


def _pinned_action_ball_json_file(
    path_value: object, expected_sha256: object, *, name: str
) -> tuple[Path, str, dict]:
    if type(path_value) is not str or not path_value:
        raise ValueError(f"{name} path must be a non-empty string")
    requested = Path(path_value).expanduser()
    if not requested.is_absolute():
        raise ValueError(f"{name} path must be absolute")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} path cannot be resolved") from exc
    if requested != resolved or requested.is_symlink() or not resolved.is_file():
        raise ValueError(
            f"{name} must be one canonical regular file without symlink components"
        )
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"{name} cannot be read") from exc
    actual_sha = hashlib.sha256(payload).hexdigest()
    expected_sha = _action_ball_bootstrap_sha256(
        expected_sha256, name=f"expected {name} SHA"
    )
    if actual_sha != expected_sha:
        raise ValueError(f"{name} file SHA does not match its pin")
    return resolved, actual_sha, _strict_action_ball_json_bytes(
        payload, name=name
    )


def _sealed_action_ball_json_content_sha256(
    document: Mapping, *, name: str
) -> str:
    seal = _action_ball_bootstrap_sha256(
        document.get("content_sha256"), name=f"{name} content SHA"
    )
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    try:
        encoded = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{name} content is not finite canonical JSON") from exc
    if hashlib.sha256(encoded).hexdigest() != seal:
        raise ValueError(f"{name} content SHA is not reproducible")
    return seal


def load_action_ball_dynamic_ready_runtime_binding(
    *,
    artifact_path: str,
    artifact_sha256: str,
    nominal_hold_receipt_path: str,
    nominal_hold_receipt_sha256: str,
    action_order: list[str] | tuple[str, ...],
    motion_paths: list[str] | tuple[str, ...],
) -> dict:
    """Load, cross-pin and seal the exact N=1 A3 dynamic-ready binding."""

    if (
        not isinstance(action_order, (list, tuple))
        or len(action_order) != 1
        or type(action_order[0]) is not str
        or not action_order[0]
        or not isinstance(motion_paths, (list, tuple))
        or len(motion_paths) != 1
    ):
        raise ValueError("dynamic-ready bootstrap currently requires exact N=1")
    artifact_file, artifact_file_sha, artifact = (
        _pinned_action_ball_json_file(
            artifact_path,
            artifact_sha256,
            name="action-ball dynamic-ready artifact",
        )
    )
    receipt_file, receipt_file_sha, receipt = _pinned_action_ball_json_file(
        nominal_hold_receipt_path,
        nominal_hold_receipt_sha256,
        name="action-ball dynamic-ready nominal-hold receipt",
    )
    artifact_content_sha = _sealed_action_ball_json_content_sha256(
        artifact, name="action-ball dynamic-ready artifact"
    )
    receipt_content_sha = _sealed_action_ball_json_content_sha256(
        receipt, name="action-ball dynamic-ready nominal-hold receipt"
    )
    action_id = action_order[0]
    if (
        artifact.get("schema_version") != 1
        or artifact.get("kind") != ACTION_BALL_DYNAMIC_READY_ARTIFACT_KIND
        or artifact.get("action_id") != action_id
    ):
        raise ValueError(
            "dynamic-ready artifact schema, kind, or action id is invalid"
        )
    robot = artifact.get("robot")
    authorization = artifact.get("authorization")
    if (
        not isinstance(robot, Mapping)
        or robot.get("family") != "AgiBot A3"
        or not isinstance(robot.get("joint_names"), list)
        or len(robot["joint_names"]) != 31
        or len(set(robot["joint_names"])) != 31
        or any(type(name) is not str or not name for name in robot["joint_names"])
        or not isinstance(authorization, Mapping)
        or set(authorization)
        != {
            "training_authorized",
            "deployment_authorized",
            "hardware_authorized",
            "isaac_nominal_hold_validated",
        }
        or any(value is not False for value in authorization.values())
    ):
        raise ValueError(
            "dynamic-ready artifact must be an unauthorized exact AgiBot A3 candidate"
        )
    try:
        physical = _validate_action_ball_dynamic_ready_physical_ready(
            artifact["physical_ready"],
            name="action-ball dynamic-ready artifact physical_ready",
        )
        runtime_plant = artifact["runtime_plant"]
        hold_candidate = artifact["hold_candidate"]
        stable_motion = artifact["sources"]["stable_motion"]
    except (KeyError, TypeError) as exc:
        raise ValueError("dynamic-ready artifact core fields are missing") from exc
    default_q = _action_ball_bootstrap_float_vector(
        runtime_plant.get("default_joint_pos_rad"),
        name="dynamic-ready artifact default_joint_pos_rad",
        expected=31,
    )
    action_scale = _action_ball_bootstrap_float_vector(
        runtime_plant.get("action_scale_rad"),
        name="dynamic-ready artifact action_scale_rad",
        expected=31,
    )
    if any(scale <= 0.0 for scale in action_scale):
        raise ValueError("dynamic-ready artifact action_scale_rad must be positive")
    hold_qdes = _action_ball_bootstrap_float_vector(
        hold_candidate.get("hold_qdes_joint_pos_rad"),
        name="dynamic-ready artifact hold_qdes_joint_pos_rad",
        expected=31,
    )
    normalized = _action_ball_bootstrap_float_vector(
        hold_candidate.get("normalized_actor_action"),
        name="dynamic-ready artifact normalized_actor_action",
        expected=31,
    )
    for index, (default, scale, action, target) in enumerate(
        zip(default_q, action_scale, normalized, hold_qdes)
    ):
        if not math.isclose(
            default + scale * action,
            target,
            rel_tol=0.0,
            abs_tol=2.0e-7,
        ):
            raise ValueError(
                "dynamic-ready normalized actor action does not decode to "
                f"hold q_des at joint {index}"
            )
    motion_path = Path(str(motion_paths[0])).expanduser().resolve(strict=True)
    if not motion_path.is_file():
        raise ValueError("dynamic-ready motion path must be a regular file")
    motion_sha = hashlib.sha256(motion_path.read_bytes()).hexdigest()
    if (
        not isinstance(stable_motion, Mapping)
        or stable_motion.get("sha256") != motion_sha
        or stable_motion.get("frame_index") != 0
    ):
        raise ValueError(
            "dynamic-ready artifact must bind frame0 of the loaded motion bytes"
        )
    receipt_artifact = receipt.get("artifact")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != ACTION_BALL_DYNAMIC_READY_NOMINAL_HOLD_KIND
        or receipt.get("verdict") != "PASS"
        or receipt.get("action_id") != action_id
        or receipt.get("motion_sha256") != motion_sha
        or receipt.get("plant_contract_match") is not True
        or receipt.get("terminal_reasons") != []
        or receipt.get("generic_terminated") is not False
        or receipt.get("generic_truncated") is not False
        or not isinstance(receipt_artifact, Mapping)
        or receipt_artifact.get("sha256") != artifact_file_sha
        or receipt_artifact.get("content_sha256") != artifact_content_sha
    ):
        raise ValueError(
            "nominal-hold receipt does not certify this exact dynamic-ready artifact"
        )
    binding = {
        "schema_version": 1,
        "kind": ACTION_BALL_DYNAMIC_READY_RUNTIME_BINDING_KIND,
        "action_order": [action_id],
        "motion_sha256_per_action": [motion_sha],
        "rows": [
            {
                "action_id": action_id,
                "physical_ready": physical,
                "hold_qdes_joint_pos_rad": hold_qdes,
                "normalized_actor_action": normalized,
                "artifact": {
                    "path": str(artifact_file),
                    "sha256": artifact_file_sha,
                    "content_sha256": artifact_content_sha,
                },
                "nominal_hold_receipt": {
                    "path": str(receipt_file),
                    "sha256": receipt_file_sha,
                    "content_sha256": receipt_content_sha,
                },
            }
        ],
    }
    binding["binding_sha256"] = action_ball_dynamic_ready_binding_sha256(
        binding
    )
    return validate_action_ball_dynamic_ready_runtime_binding(
        binding, expected_action_count=1
    )


def validate_action_ball_policy_bootstrap(
    value: object, *, expected_action_count: int | None = None
) -> dict:
    """Validate the fresh-only ActionBall actor initialization contract.

    The bootstrap does not change the deployed action decoder. It initializes
    a fresh actor's last linear layer to emit the normalized residual from the
    robot default pose to either the historical shared ready (schema 1) or the
    nominal-hold-certified dynamic q_des (schema 2), with a small Gaussian
    exploration scale. A resumed policy must never be overwritten.
    """

    block = _require_exact_mapping_keys(
        value,
        _ACTION_BALL_POLICY_BOOTSTRAP_KEYS,
        name="action-ball policy bootstrap",
    )
    schema_version = block["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version not in (1, 2)
        or block["kind"] != ACTION_BALL_POLICY_BOOTSTRAP_KIND
    ):
        raise ValueError(
            "action-ball policy bootstrap must use schema 1 or 2"
        )
    action_count = block["action_count"]
    allowed_action_counts = (1, 5) if schema_version == 1 else (1,)
    if type(action_count) is not int or action_count not in allowed_action_counts:
        raise ValueError(
            "shared-ready schema 1 supports exact N=1/N=5, while "
            "dynamic-ready schema 2 currently supports exact N=1"
        )
    if (
        expected_action_count is not None
        and action_count != expected_action_count
    ):
        raise ValueError(
            "action-ball policy bootstrap action_count disagrees with actor contract"
        )
    action_order = block["action_order"]
    if (
        not isinstance(action_order, (list, tuple))
        or len(action_order) != action_count
        or any(type(item) is not str or not item for item in action_order)
        or len(set(action_order)) != action_count
    ):
        raise ValueError(
            "action-ball policy bootstrap requires one unique action id per slot"
        )
    joint_names = block["joint_names"]
    if (
        not isinstance(joint_names, (list, tuple))
        or len(joint_names) != 31
        or any(type(item) is not str or not item for item in joint_names)
        or len(set(joint_names)) != 31
    ):
        raise ValueError(
            "action-ball policy bootstrap is bound to 31 unique A3 joints"
        )

    ready_keys = (
        _ACTION_BALL_POLICY_BOOTSTRAP_READY_KEYS
        if schema_version == 1
        else _ACTION_BALL_POLICY_BOOTSTRAP_READY_V2_KEYS
    )
    ready = _require_exact_mapping_keys(
        block["ready_source"],
        ready_keys,
        name="action-ball policy bootstrap ready_source",
    )
    if schema_version == 1:
        if (
            ready["semantics"]
            != "motion.joint_pos[motion.seg_start[action_slot]]"
        ):
            raise ValueError(
                "action-ball bootstrap ready source semantics changed"
            )
        for key in ("canonical_ready_sha256", "canonical_ready_fk_sha256"):
            raw = ready[key]
            if raw != "":
                _action_ball_bootstrap_sha256(
                    raw,
                    name=f"action-ball policy bootstrap ready_source.{key}",
                )
    elif (
        ready["semantics"]
        != "action_ball_dynamic_ready.rows[action_slot].physical_ready"
    ):
        raise ValueError(
            "dynamic-ready bootstrap source semantics changed"
        )
    motion_digests = ready["motion_sha256_per_action"]
    if (
        not isinstance(motion_digests, (list, tuple))
        or len(motion_digests) != action_count
    ):
        raise ValueError(
            "action-ball policy bootstrap requires one motion SHA per action"
        )
    for index, digest in enumerate(motion_digests):
        _action_ball_bootstrap_sha256(
            digest,
            name=(
                "action-ball policy bootstrap "
                f"ready_source.motion_sha256_per_action[{index}]"
            ),
        )
    dynamic_binding = None
    if schema_version == 1:
        ready_q = _action_ball_bootstrap_float_vector(
            ready["shared_ready_joint_pos"],
            name="action-ball policy bootstrap shared_ready_joint_pos",
            expected=31,
        )
        expected_ready_sha = action_ball_shared_ready_sha256(
            action_order=list(action_order),
            joint_names=list(joint_names),
            shared_ready_joint_pos=ready_q,
        )
        if (
            _action_ball_bootstrap_sha256(
                ready["shared_ready_joint_pos_sha256"],
                name=(
                    "action-ball policy bootstrap "
                    "shared_ready_joint_pos_sha256"
                ),
            )
            != expected_ready_sha
        ):
            raise ValueError(
                "action-ball policy bootstrap shared-ready SHA is not reproducible"
            )
        target_q = ready_q
    else:
        physical_ready = _validate_action_ball_dynamic_ready_physical_ready(
            ready["physical_ready"],
            name="action-ball policy bootstrap physical_ready",
        )
        ready_q = physical_ready["joint_pos_rad"]
        dynamic_binding = validate_action_ball_dynamic_ready_runtime_binding(
            ready["identity"], expected_action_count=action_count
        )
        if (
            dynamic_binding["action_order"] != list(action_order)
            or dynamic_binding["motion_sha256_per_action"]
            != list(motion_digests)
            or dynamic_binding["rows"][0]["physical_ready"]
            != physical_ready
        ):
            raise ValueError(
                "dynamic-ready policy bootstrap identity disagrees with its "
                "ready source"
            )
        target_q = list(
            dynamic_binding["rows"][0]["hold_qdes_joint_pos_rad"]
        )

    decoder_keys = (
        _ACTION_BALL_POLICY_BOOTSTRAP_DECODER_KEYS
        if schema_version == 1
        else _ACTION_BALL_POLICY_BOOTSTRAP_DECODER_V2_KEYS
    )
    decoder = _require_exact_mapping_keys(
        block["decoder"],
        decoder_keys,
        name="action-ball policy bootstrap decoder",
    )
    if (
        decoder["semantics"] != "q_des=default_joint_pos+action_scale*action"
        or decoder["use_default_offset"] is not True
    ):
        raise ValueError(
            "action-ball policy bootstrap may not change the default-offset decoder"
        )
    default_q = _action_ball_bootstrap_float_vector(
        decoder["default_joint_pos"],
        name="action-ball policy bootstrap default_joint_pos",
        expected=31,
    )
    scale = _action_ball_bootstrap_float_vector(
        decoder["action_scale"],
        name="action-ball policy bootstrap action_scale",
        expected=31,
    )
    bias = _action_ball_bootstrap_float_vector(
        decoder["normalized_bias"],
        name="action-ball policy bootstrap normalized_bias",
        expected=31,
    )
    if schema_version == 2:
        decoder_target = _action_ball_bootstrap_float_vector(
            decoder["target_joint_pos"],
            name="action-ball policy bootstrap target_joint_pos",
            expected=31,
        )
        if decoder_target != target_q:
            raise ValueError(
                "dynamic-ready decoder target disagrees with its runtime binding"
            )
        if (
            dynamic_binding is None
            or list(dynamic_binding["rows"][0]["normalized_actor_action"])
            != bias
        ):
            raise ValueError(
                "dynamic-ready decoder bias disagrees with its runtime binding"
            )
    if (
        decoder["startup_offset_delta_source"]
        != "events.add_joint_default_pos.uniform_add"
    ):
        raise ValueError(
            "action-ball policy bootstrap startup offset source changed"
        )
    startup_delta_lower = _action_ball_bootstrap_float_vector(
        decoder["startup_offset_delta_lower"],
        name="action-ball policy bootstrap startup_offset_delta_lower",
        expected=31,
    )
    startup_delta_upper = _action_ball_bootstrap_float_vector(
        decoder["startup_offset_delta_upper"],
        name="action-ball policy bootstrap startup_offset_delta_upper",
        expected=31,
    )
    if any(
        lower > upper
        for lower, upper in zip(startup_delta_lower, startup_delta_upper)
    ):
        raise ValueError(
            "action-ball policy bootstrap startup offset envelope is invalid"
        )
    if any(item <= 0.0 for item in scale):
        raise ValueError("action-ball policy bootstrap action_scale must be positive")
    for index, (default, gain, normalized, target) in enumerate(
        zip(default_q, scale, bias, target_q)
    ):
        reconstructed = default + gain * normalized
        if not math.isclose(
            reconstructed, target, rel_tol=0.0, abs_tol=2.0e-7
        ):
            raise ValueError(
                "action-ball policy bootstrap normalized bias does not decode "
                f"to its q_des target at joint {index}"
            )

    initialization = _require_exact_mapping_keys(
        block["initialization"],
        _ACTION_BALL_POLICY_BOOTSTRAP_INITIALIZATION_KEYS,
        name="action-ball policy bootstrap initialization",
    )
    if (
        initialization["fresh_only"] is not True
        or initialization["resume_overwrite_prohibited"] is not True
        or initialization["output_layer_weight"] != "zeros"
        or initialization["output_layer_bias"] != "decoder.normalized_bias"
    ):
        raise ValueError(
            "action-ball actor bootstrap must be a fresh-only zero-weight/bias initialization"
        )
    noise_std = initialization["init_noise_std"]
    sigma = initialization["sigma_envelope"]
    if (
        type(noise_std) not in (int, float)
        or not math.isfinite(float(noise_std))
        or float(noise_std) != 0.02
        or type(sigma) not in (int, float)
        or float(sigma) != 4.0
    ):
        raise ValueError(
            "action-ball policy bootstrap requires init_noise_std=0.02 "
            "and a 4-sigma envelope"
        )

    guard = _require_exact_mapping_keys(
        block["hard_inner_guard"],
        _ACTION_BALL_POLICY_BOOTSTRAP_GUARD_KEYS,
        name="action-ball policy bootstrap hard_inner_guard",
    )
    margin_rad = guard["margin_rad"]
    margin_fraction = guard["margin_fraction"]
    if (
        guard["limit_source"] != "articulation.data.joint_pos_limits"
        or type(margin_rad) not in (int, float)
        or float(margin_rad) != 0.0
        or type(margin_fraction) not in (int, float)
        or float(margin_fraction) != 0.02
    ):
        raise ValueError(
            "action-ball bootstrap hard-inner guard must match the existing "
            "two-percent physical-limit termination"
        )
    hard_lower = _action_ball_bootstrap_float_vector(
        guard["hard_lower"],
        name="action-ball policy bootstrap hard_lower",
        expected=31,
    )
    hard_upper = _action_ball_bootstrap_float_vector(
        guard["hard_upper"],
        name="action-ball policy bootstrap hard_upper",
        expected=31,
    )
    inner_lower = _action_ball_bootstrap_float_vector(
        guard["hard_inner_lower"],
        name="action-ball policy bootstrap hard_inner_lower",
        expected=31,
    )
    inner_upper = _action_ball_bootstrap_float_vector(
        guard["hard_inner_upper"],
        name="action-ball policy bootstrap hard_inner_upper",
        expected=31,
    )
    for index, (
        lo,
        hi,
        inner_lo,
        inner_hi,
        target,
        gain,
        offset_lo,
        offset_hi,
    ) in enumerate(
        zip(
            hard_lower,
            hard_upper,
            inner_lower,
            inner_upper,
            target_q,
            scale,
            startup_delta_lower,
            startup_delta_upper,
        )
    ):
        if not lo < hi:
            raise ValueError(
                f"action-ball policy bootstrap hard limits are invalid at joint {index}"
            )
        expected_inner_lo = lo + 0.02 * (hi - lo)
        expected_inner_hi = hi - 0.02 * (hi - lo)
        if (
            not math.isclose(
                inner_lo, expected_inner_lo, rel_tol=0.0, abs_tol=2.0e-7
            )
            or not math.isclose(
                inner_hi, expected_inner_hi, rel_tol=0.0, abs_tol=2.0e-7
            )
        ):
            raise ValueError(
                "action-ball policy bootstrap hard-inner envelope is not "
                f"reproducible at joint {index}"
            )
        radius = 4.0 * 0.02 * gain
        if not (
            inner_lo < target + offset_lo - radius
            and target + offset_hi + radius < inner_hi
        ):
            raise ValueError(
                "action-ball policy bootstrap 4-sigma plus startup-offset "
                f"q_des envelope reaches the hard forbidden band at joint {index}"
            )
    return dict(block)


def validate_action_ball_training_authorization(contract: Mapping) -> bool:
    """Validate and return the action-ball diagnostic authorization brand.

    The block is optional so every pre-action-ball schema-3 sidecar remains byte compatible.
    Once present, however, its five downstream rights and the runtime, evaluator and motion
    views must all agree. A truthy string or float must never turn an unauthorized diagnostic
    into a formal artifact.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("training contract root must be an object")
    target_mode = contract.get("target_mode")
    actor_contract = contract.get("actor_obs_contract")
    actor_prefixed = (
        type(actor_contract) is str
        and any(
            actor_contract.startswith(prefix)
            for prefix, _base_width in _ACTION_BALL_ACTOR_OBS_LAYOUTS
        )
    )
    block_present = ACTION_BALL_TRAINING_KEY in contract
    projection_present = FINITE_PRECLAMP_QDES_PROJECTION_KEY in contract
    projection_inset_present = (
        FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY in contract
    )
    if (
        projection_present
        and contract[FINITE_PRECLAMP_QDES_PROJECTION_KEY] is not True
    ):
        raise ValueError(
            "schema-3 finite_preclamp_qdes_projection_enabled must be the "
            "exact boolean true when present"
        )
    if projection_inset_present:
        inset = contract[FINITE_PROJECTION_SOFT_ENVELOPE_INSET_FRACTION_KEY]
        if (
            isinstance(inset, bool)
            or not isinstance(inset, (int, float))
            or not math.isfinite(float(inset))
            or not 0.0 <= float(inset) < 0.5
        ):
            raise ValueError(
                "schema-3 finite projection soft-envelope inset must be "
                "finite and lie in [0, 0.5)"
            )
    action_ball_intent = (
        target_mode == "action_ball" or actor_prefixed or block_present
    )
    if not action_ball_intent:
        if projection_present or projection_inset_present:
            raise ValueError(
                "schema-3 finite q_des projection is ActionBall-only"
            )
        return False
    if target_mode != "action_ball":
        raise ValueError(
            "schema-3 action-ball authorization requires target_mode='action_ball'"
        )
    actor_layout = _parse_action_ball_actor_obs_contract(actor_contract)
    if actor_layout is None:
        raise ValueError(
            "schema-3 action-ball authorization requires "
            "actor_obs_contract=action_ball_n<N> or "
            "action_ball_table_pose_n<N> for N in [1,1024]"
        )
    actor_count, expected_actor_width = actor_layout
    if contract.get("actor_obs_total_dim") != expected_actor_width:
        raise ValueError(
            "schema-3 action-ball actor_obs_total_dim does not match its "
            "exact actor_obs_contract layout"
        )
    if not block_present:
        raise ValueError(
            "schema-3 action-ball contract is missing the mandatory "
            "action_ball_training authorization block"
        )
    if not projection_present:
        raise ValueError(
            "schema-3 action-ball contract is missing the immutable finite "
            "pre-clamp q_des projection runtime fact"
        )
    if not projection_inset_present:
        raise ValueError(
            "schema-3 action-ball contract is missing the immutable finite "
            "q_des projection soft-envelope inset fact"
        )
    if (
        type(contract.get("schema_version")) is not int
        or contract["schema_version"] != TRAINING_CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "action_ball_training authorization requires a plain-integer "
            "schema-3 training contract"
        )
    action_ball = contract[ACTION_BALL_TRAINING_KEY]
    if not isinstance(action_ball, Mapping):
        raise ValueError("schema-3 action_ball_training must be an object")
    action_ball_schema = action_ball.get("schema_version")
    if type(action_ball_schema) is not int or action_ball_schema != 1:
        raise ValueError(
            "schema-3 action_ball_training.schema_version must be integer 1"
        )
    policy_bootstrap = action_ball.get("policy_bootstrap")
    if policy_bootstrap is not None:
        validate_action_ball_policy_bootstrap(
            policy_bootstrap, expected_action_count=actor_count
        )
    authorization = _require_exact_mapping_keys(
        action_ball.get("authorization"),
        _ACTION_BALL_AUTHORIZATION_KEYS,
        name="schema-3 action_ball_training.authorization",
    )
    for key, value in authorization.items():
        if type(value) is not bool:
            raise ValueError(
                "schema-3 action_ball_training.authorization."
                f"{key} must be an exact boolean"
            )
    diagnostic = authorization["diagnostic_unauthorized"]
    expected_authorization = {
        key: diagnostic for key in _ACTION_BALL_AUTHORIZATION_KEYS
    }
    if dict(authorization) != expected_authorization:
        raise ValueError(
            "schema-3 action-ball authorization contains contradictory "
            "diagnostic/formal rights"
        )

    runtime = action_ball.get("runtime")
    motion_admission = action_ball.get("motion_admission")
    if not isinstance(runtime, Mapping) or not isinstance(motion_admission, Mapping):
        raise ValueError(
            "schema-3 action-ball authorization requires runtime and "
            "motion_admission objects"
        )
    runtime_diagnostic = _optional_exact_bool(
        runtime,
        "diagnostic_unauthorized",
        name="schema-3 action_ball_training.runtime",
    )
    motion_diagnostic = _optional_exact_bool(
        motion_admission,
        "diagnostic_unauthorized",
        name="schema-3 action_ball_training.motion_admission",
    )
    if runtime_diagnostic != diagnostic or motion_diagnostic != diagnostic:
        raise ValueError(
            "schema-3 action-ball diagnostic authorization disagrees across "
            "training/runtime/motion-admission contracts"
        )

    evaluator = runtime.get("evaluator_authority")
    if not isinstance(evaluator, Mapping):
        raise ValueError(
            "schema-3 action-ball authorization requires an evaluator_authority object"
        )
    evaluator_diagnostic = _optional_exact_bool(
        evaluator,
        "diagnostic_unauthorized",
        name="schema-3 action_ball_training.runtime.evaluator_authority",
    )
    if "formal_authority_available" not in evaluator:
        raise ValueError(
            "schema-3 action-ball evaluator_authority requires "
            "formal_authority_available"
        )
    evaluator_formal = _optional_exact_bool(
        evaluator,
        "formal_authority_available",
        name="schema-3 action_ball_training.runtime.evaluator_authority",
    )
    if evaluator_diagnostic != diagnostic or evaluator_formal == diagnostic:
        raise ValueError(
            "schema-3 action-ball evaluator authority disagrees with the "
            "diagnostic/formal authorization"
        )

    if diagnostic:
        evaluator = _require_exact_mapping_keys(
            evaluator,
            frozenset(
                {
                    "diagnostic_unauthorized",
                    "formal_authority_available",
                    "formal_launch_requires_code_pinned_receipt",
                    "runtime_or_manifest_may_self_authorize",
                    "authority_binding",
                    "authority_state_owner_sha256",
                }
            ),
            name=(
                "schema-3 diagnostic "
                "action_ball_training.runtime.evaluator_authority"
            ),
        )
        for key, expected in (
            ("formal_launch_requires_code_pinned_receipt", True),
            ("runtime_or_manifest_may_self_authorize", False),
        ):
            actual = _optional_exact_bool(
                evaluator,
                key,
                name="schema-3 action_ball_training.runtime.evaluator_authority",
            )
            if actual is not expected:
                raise ValueError(
                    "schema-3 diagnostic action-ball evaluator authority must "
                    "remain code-pinned and may not self-authorize"
                )
        if "training_authorized" not in motion_admission:
            raise ValueError(
                "schema-3 diagnostic action-ball motion admission requires "
                "training_authorized=false"
            )
        training_authorized = _optional_exact_bool(
            motion_admission,
            "training_authorized",
            name="schema-3 action_ball_training.motion_admission",
        )
        if training_authorized:
            raise ValueError(
                "schema-3 diagnostic action-ball motion admission must set "
                "training_authorized=false"
            )
    elif motion_admission.get("authorization_purpose") != "training":
        raise ValueError(
            "schema-3 formal action-ball motion admission must be "
            "training-authorized"
        )
    identity_raw = action_ball.get(ACTION_BALL_ACTION_SET_IDENTITY_KEY)
    if diagnostic:
        # A diagnostic may carry the inspected identity for debugging, but it
        # can never export it as formal metadata (the binding helper strips all
        # such keys for non-exact lineages).
        if identity_raw is not None:
            validate_action_ball_action_set_identity_block(identity_raw)
        return True
    if identity_raw is None:
        raise ValueError(
            "schema-3 formal action-ball contract is missing action_set_identity"
        )
    identity = validate_action_ball_action_set_identity_block(identity_raw)
    preflight = action_ball.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError(
            "schema-3 formal action-ball contract requires a preflight object"
        )
    manifest = preflight.get("manifest")
    prototype = preflight.get("prototype")
    if not isinstance(manifest, Mapping) or not isinstance(prototype, Mapping):
        raise ValueError(
            "schema-3 formal action-ball preflight requires manifest/prototype objects"
        )
    validate_action_ball_action_set_runtime_identity(
        identity,
        actor_obs_contract=contract.get("actor_obs_contract"),
        actor_obs_width=contract.get("actor_obs_total_dim"),
        manifest_path=manifest.get("path"),
        manifest_sha256=manifest.get("file_sha256"),
        scope=prototype.get("scope"),
        mobility_mode=preflight.get("mobility_mode"),
        ordered_action_ids=preflight.get("action_order"),
        ordered_action_uids=preflight.get("action_uids"),
    )
    segment_lengths = contract.get("motion_segment_lengths")
    if (
        not isinstance(segment_lengths, (list, tuple))
        or len(segment_lengths) != identity["expected_n"]
    ):
        raise ValueError(
            "schema-3 formal action-ball motion segments do not have exact action-set N"
        )
    bindings = preflight.get("action_bindings")
    if not isinstance(bindings, (list, tuple)) or len(bindings) != identity["expected_n"]:
        raise ValueError(
            "schema-3 formal action-ball preflight action_bindings do not have exact N"
        )
    binding_ids = [
        item.get("action_id") if isinstance(item, Mapping) else None
        for item in bindings
    ]
    binding_uids = [
        item.get("action_uid") if isinstance(item, Mapping) else None
        for item in bindings
    ]
    if (
        binding_ids != identity["ordered_action_ids"]
        or binding_uids != identity["ordered_action_uids"]
    ):
        raise ValueError(
            "schema-3 formal action-ball preflight bindings disagree with action_set_identity"
        )
    return diagnostic


def bind_action_ball_diagnostic_metadata(
    metadata: MutableMapping[str, str],
    contract: Mapping | None,
    *,
    lineage_exact: bool,
) -> bool:
    """Clear donor brands and stamp a diagnostic action-ball export as non-bookable."""

    if type(lineage_exact) is not bool:
        raise ValueError("action-ball export lineage_exact must be an exact boolean")
    metadata.pop(ACTION_BALL_DIAGNOSTIC_METADATA_KEY, None)
    metadata.pop(FORMAL_EVIDENCE_BOOKABLE_METADATA_KEY, None)
    if contract is None:
        return False
    diagnostic = validate_action_ball_training_authorization(contract)
    if not diagnostic:
        return False
    if lineage_exact:
        raise ValueError(
            "diagnostic_unauthorized action-ball contract cannot claim "
            "training_contract_lineage_exact=1"
        )
    metadata["training_contract_exact"] = "0"
    metadata[ACTION_BALL_DIAGNOSTIC_METADATA_KEY] = "1"
    metadata[FORMAL_EVIDENCE_BOOKABLE_METADATA_KEY] = "0"
    return True


def validate_schema3_contract_structure(contract: Mapping) -> None:
    """Validate a schema-3 sidecar without promoting it to a formal-exact lineage.

    Schema 3 binds the instantiated execution contract even for deliberately diagnostic runs
    (for example, a causal continuation on an untagged legacy motion).  Those sidecars still need
    complete, internally consistent runtime facts and an adjacent checkpoint hash binding; the
    narrower :func:`validate_schema3_contract` adds the formal schema-2 motion requirement.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("training contract root must be an object")
    schema = contract.get("schema_version", 0)
    if type(schema) is not int:
        raise ValueError("training-contract schema_version must be a plain integer")
    if schema != TRAINING_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported formal training-contract schema {schema}; expected "
            f"{TRAINING_CONTRACT_SCHEMA_VERSION}"
        )
    missing = [key for key in (*RUNTIME_EXECUTION_KEYS, *SCHEMA3_TASK_KEYS) if key not in contract]
    if missing:
        raise ValueError("schema-3 training contract missing execution facts: " + ", ".join(missing))
    # Optional only as a complete pair.  The OFF/legacy spelling is total absence; an enabled
    # block must bind the exact C++ governor JSON plus the clip layout that defines strike frames.
    planner_task_revision_metadata(contract)
    validate_action_ball_training_authorization(contract)
    actor_names_raw = contract["actor_obs_term_names"]
    actor_dims_raw = contract["actor_obs_term_dims"]
    actor_history_raw = contract["observation_history_lengths"]
    if (
        not isinstance(actor_names_raw, (list, tuple))
        or not isinstance(actor_dims_raw, (list, tuple))
        or not isinstance(actor_history_raw, (list, tuple))
        or not actor_names_raw
        or len(actor_names_raw) != len(actor_dims_raw)
        or len(actor_names_raw) != len(actor_history_raw)
    ):
        raise ValueError(
            "schema-3 actor observation names/dims/history must be non-empty equal-length arrays"
        )
    if any(type(name) is not str or not name.strip() for name in actor_names_raw):
        raise ValueError("schema-3 actor observation names must be non-empty strings")
    if len(set(actor_names_raw)) != len(actor_names_raw):
        raise ValueError("schema-3 actor observation names must be unique")
    if any(type(dim) is not int or dim <= 0 for dim in actor_dims_raw):
        raise ValueError("schema-3 actor observation dims must be positive integers")
    if any(type(value) is not int or value <= 0 for value in actor_history_raw):
        raise ValueError("schema-3 actor observation history lengths must be positive integers")
    actor_total = contract["actor_obs_total_dim"]
    if type(actor_total) is not int or actor_total != sum(actor_dims_raw):
        raise ValueError("schema-3 actor_obs_total_dim must equal the sum of term dims")
    for key in ("face_command_enabled", "motion_allow_legacy_link_origin_velocity"):
        if key in contract and not isinstance(contract[key], bool):
            raise ValueError(f"schema-3 {key} must be boolean when present")
    # Per-clip swing-family table (spdmix v2 硬绑定一).  Optional as total absence: legacy/default
    # contracts carry no key and stay byte-identical.  When present it must be a complete,
    # legal-valued table — one "forehand"/"backhand" string per loaded motion segment with both
    # families represented — because every swing_sign/obs/target-side decision keys off it.
    # 人话:合同里记了家族表就整表核对,记错的表比没记还危险(判分/观测全按它走)。
    if "motion_clip_family_per_clip" in contract:
        families = contract["motion_clip_family_per_clip"]
        if (
            not isinstance(families, (list, tuple))
            or not families
            or any(type(value) is not str for value in families)
            or any(value not in ("forehand", "backhand") for value in families)
        ):
            raise ValueError(
                "schema-3 motion_clip_family_per_clip must be a non-empty array of "
                "'forehand'/'backhand' strings"
            )
        # Both-families is a rule about the UNIFIED policy: with two or more clips every
        # swing_sign/obs/target-side decision keys off the split, so a one-sided table trains one
        # lane and leaves the other dead. A SINGLE-clip run has no split — one constant for every
        # env — so the rule protects nothing there while denying that run any way to state which
        # hand it is. 人话:按动作数量判,不是一刀切。单动作臂本来就没有两条通道可分,
        # 却因为这条规则连"我是反手"都说不出口,只能落进默认被当成正手。
        if len(families) >= 2 and (
            "forehand" not in families or "backhand" not in families
        ):
            raise ValueError(
                "schema-3 motion_clip_family_per_clip must name at least one forehand and one "
                "backhand clip"
            )
        segment_lengths = contract["motion_segment_lengths"]
        if not isinstance(segment_lengths, (list, tuple)) or len(families) != len(
            segment_lengths
        ):
            raise ValueError(
                "schema-3 motion_clip_family_per_clip must declare exactly one family per loaded "
                "motion segment"
            )
    if ACTOR_LEG_REF_MASK_PROVENANCE_KEY in contract:
        epoch = contract[ACTOR_LEG_REF_MASK_PROVENANCE_KEY]
        if (
            isinstance(epoch, bool)
            or type(epoch) is not int
            or epoch != ACTOR_LEG_REF_MASK_PROVENANCE_EPOCH
        ):
            raise ValueError("schema-3 actor_leg_ref_mask_provenance_epoch must be integer 1")
    if "actor_leg_ref_mask" in contract and contract["actor_leg_ref_mask"] is not True:
        raise ValueError("schema-3 actor_leg_ref_mask must be true when present")
    require_actor_leg_ref_mask_provenance(contract)
    if "face_command_pairing" in contract and contract["face_command_pairing"] not in (
        "shared_plus_y",
        "legacy_signed_vs_A",
    ):
        raise ValueError("schema-3 face_command_pairing is invalid")
    if contract.get("actor_obs_contract") == "deploy_parity_face179":
        if contract.get("face_command_enabled") is not True:
            raise ValueError("formal face179 schema-3 contract requires face_command_enabled=true")
        if contract.get("face_command_pairing") != "shared_plus_y":
            raise ValueError("formal face179 schema-3 contract requires shared_plus_y")
        # 拍面符号表(spdmix 硬绑定三拆除)。两种判法,按合同里有没有家族表二选一:
        # * 家族表缺席 = legacy 2-clip 臂:仍要求逐字 [+1,-1],报错文本一字不改。
        # * 家族表在场(六 clip 变速烤入等):按族核对——每 clip 一个符号,正手族全 +1、
        #   反手族全 -1(正手打红面/+Y,反手打黑面/−Y;变速变体不改变拍面)。
        #   家族表本身已在上面整表校验过(取值/两族齐全/长度==clip 数),这里只对符号。
        families = contract.get("motion_clip_family_per_clip")
        try:
            raw_face_signs = contract["mount_normal_sign_per_clip"]
            if any(isinstance(value, bool) for value in raw_face_signs):
                raise ValueError
            face_signs = tuple(float(value) for value in raw_face_signs)
        except (KeyError, TypeError, ValueError) as exc:
            if families is None:
                raise ValueError(
                    "formal face179 schema-3 contract requires mount_normal_sign_per_clip=[+1,-1]"
                ) from exc
            raise ValueError(
                "formal face179 schema-3 contract requires one numeric "
                "mount_normal_sign_per_clip entry per clip"
            ) from exc
        if families is None:
            if face_signs != (1.0, -1.0):
                raise ValueError(
                    "formal face179 schema-3 contract requires mount_normal_sign_per_clip=[+1,-1]"
                )
        else:
            if len(face_signs) != len(families):
                raise ValueError(
                    "formal face179 schema-3 contract requires one mount_normal_sign_per_clip "
                    f"entry per clip: got {len(face_signs)} signs for {len(families)} clips"
                )
            expected_face_signs = tuple(
                1.0 if family == "forehand" else -1.0 for family in families
            )
            if face_signs != expected_face_signs:
                raise ValueError(
                    "formal face179 schema-3 contract requires face sign +1 for every forehand "
                    "clip and -1 for every backhand clip: got mount_normal_sign_per_clip="
                    f"{list(face_signs)} for motion_clip_family_per_clip={list(families)}"
                )

    joint_names = contract["joint_names"]
    if not isinstance(joint_names, (list, tuple)) or not joint_names:
        raise ValueError("schema-3 joint_names must be a non-empty array")
    n = len(joint_names)
    if len(set(str(value) for value in joint_names)) != n:
        raise ValueError("schema-3 joint_names must be unique")
    raw_actuator_types = contract["joint_actuator_types"]
    actuator_types = (
        list(raw_actuator_types)
        if isinstance(raw_actuator_types, (list, tuple))
        else []
    )
    if len(actuator_types) != n or any(
        value not in ("implicit", "explicit") for value in actuator_types
    ):
        raise ValueError(
            "schema-3 joint_actuator_types must contain one implicit|explicit value per joint"
        )

    def finite_vector(key: str, *, positive: bool) -> None:
        value = contract[key]
        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise ValueError(f"schema-3 {key} must contain one value per joint")
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"schema-3 {key} must be numeric") from exc
        if any(
            not math.isfinite(item) or (item <= 0.0 if positive else item < 0.0)
            for item in numbers
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"schema-3 {key} must be finite and {qualifier}")

    finite_vector("joint_stiffness", positive=True)
    finite_vector("joint_damping", positive=True)
    finite_vector("joint_effort_limits", positive=True)
    finite_vector("joint_armature", positive=False)
    finite_vector("joint_friction_coefficients", positive=False)
    finite_vector("joint_velocity_limits", positive=True)

    qdot_hinge = contract.get("joint_velocity_limit_hinge_reward")
    if qdot_hinge is not None:
        if not isinstance(qdot_hinge, Mapping):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward must be an object or null"
            )
        if type(qdot_hinge.get("schema_version")) is not int or qdot_hinge.get(
            "schema_version"
        ) != 1:
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward schema_version must be integer 1"
            )
        enabled = qdot_hinge.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward enabled must be boolean"
            )
        raw_weight = qdot_hinge.get("weight")
        raw_margin = qdot_hinge.get("margin")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward weight must be finite and <= 0"
            )
        if isinstance(raw_margin, bool) or not isinstance(raw_margin, (int, float)):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward margin must be finite and in (0, 1)"
            )
        weight = float(raw_weight)
        margin = float(raw_margin)
        if not math.isfinite(weight) or weight > 0.0:
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward weight must be finite and <= 0"
            )
        if not math.isfinite(margin) or not 0.0 < margin < 1.0:
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward margin must be finite and in (0, 1)"
            )
        if enabled != (weight < 0.0):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward enabled disagrees with weight"
            )
        articulation_joint_names = contract["articulation_joint_names"]
        if (
            n != 31
            or type(qdot_hinge.get("joint_count")) is not int
            or qdot_hinge.get("joint_count") != 31
            or not isinstance(articulation_joint_names, (list, tuple))
            or [str(value) for value in articulation_joint_names]
            != [str(value) for value in joint_names]
        ):
            raise ValueError(
                "schema-3 joint_velocity_limit_hinge_reward requires identity 31-joint order"
            )
        expected_fixed = {
            "asset_name": "robot",
            "joint_order": "runtime_articulation_identity",
            "velocity_limit_source": "runtime_execution_facts.joint_velocity_limits",
            # 2026-07-25 SUM 裁定:与 train.py 逐字节一致;旧 mean 串 sidecar 在此 fail loud。
            "formula": "sum(relu(abs(qd)/joint_velocity_limits-margin)^2)",
        }
        for key, expected in expected_fixed.items():
            if qdot_hinge.get(key) != expected:
                raise ValueError(
                    "schema-3 joint_velocity_limit_hinge_reward "
                    f"{key} must be exactly {expected!r}"
                )

    qdes_slew = contract.get("processed_qdes_slew_hinge_reward")
    if qdes_slew is not None:
        if not isinstance(qdes_slew, Mapping):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward must be an object"
            )
        if type(qdes_slew.get("schema_version")) is not int or qdes_slew.get(
            "schema_version"
        ) != 1:
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward schema_version must be integer 1"
            )
        enabled = qdes_slew.get("enabled")
        raw_weight = qdes_slew.get("weight")
        raw_margin = qdes_slew.get("margin")
        raw_start = qdes_slew.get("recovery_start_s")
        raw_end = qdes_slew.get("recovery_end_s")
        if not isinstance(enabled, bool) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (raw_weight, raw_margin, raw_start, raw_end)
        ):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward has invalid enabled/weight/margin/window"
            )
        weight, margin, start, end = map(
            float, (raw_weight, raw_margin, raw_start, raw_end)
        )
        if (
            not all(math.isfinite(value) for value in (weight, margin, start, end))
            or weight > 0.0
            or not 0.0 < margin < 1.0
            or start < 0.0
            or start >= end
            or enabled != (weight < 0.0)
        ):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward has invalid enabled/weight/margin/window"
            )
        selected_names = qdes_slew.get("joint_names")
        expected_names = {
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        }
        if (
            type(qdes_slew.get("joint_count")) is not int
            or qdes_slew.get("joint_count") != 15
            or not isinstance(selected_names, (list, tuple))
            or len(selected_names) != 15
            or set(selected_names) != expected_names
            or [name for name in joint_names if name in expected_names]
            != list(selected_names)
        ):
            raise ValueError(
                "schema-3 processed_qdes_slew_hinge_reward requires the runtime-ordered 15 A3 waist/leg joints"
            )
        expected_fixed = {
            "action_name": "joint_pos",
            "command_name": "racket_target",
            "joint_order": "runtime_articulation_subsequence",
            "control_dt_s": 0.02,
            "control_dt_source": "env_cfg.sim.dt_times_decimation",
            "velocity_limit_source": "runtime_execution_facts.joint_velocity_limits",
            "age_source": "per_env_exact_strike_control_tick_latch",
            "formula": (
                # 2026-07-25 SUM 裁定:与 train.py 逐字节一致;旧 mean 串 sidecar 在此 fail loud。
                "sum(1-exp(-square(relu(abs(delta_processed_qdes)/(joint_velocity_limits*0.02)-margin)/(1-margin))))"
            ),
            "gate": "same_attempt_post_strike_age_s_inclusive",
        }
        for key, expected in expected_fixed.items():
            if qdes_slew.get(key) != expected:
                raise ValueError(
                    "schema-3 processed_qdes_slew_hinge_reward "
                    f"{key} must be exactly {expected!r}"
                )
    _validate_lower_body_wave_contracts(
        contract,
        [str(value) for value in joint_names],
        [str(value) for value in contract["articulation_joint_names"]],
    )
    _validate_post_swing_settle_debt_contract(contract)
    _validate_qdes_limit_barrier_contract(contract)
    _validate_push_robot_event_contract(contract)
    _validate_force_push_event_contract(contract)
    _validate_ground_plant_contract(contract)
    if contract["joint_friction_backend"] != JOINT_FRICTION_BACKEND:
        raise ValueError("schema-3 joint_friction_backend must be physx")
    if contract["joint_friction_semantics"] != JOINT_FRICTION_SEMANTICS:
        raise ValueError(
            "schema-3 joint_friction_semantics does not describe Isaac/PhysX joint friction"
        )
    if contract["joint_friction_units"] != JOINT_FRICTION_UNITS:
        raise ValueError("schema-3 joint_friction_units must be dimensionless")

    articulation_body_names_raw = contract["articulation_body_names"]
    if not isinstance(articulation_body_names_raw, (list, tuple)):
        raise ValueError("schema-3 articulation_body_names must be non-empty and unique")
    articulation_body_names = [str(value) for value in articulation_body_names_raw]
    if (
        not articulation_body_names
        or any(not value for value in articulation_body_names)
        or len(set(articulation_body_names)) != len(articulation_body_names)
    ):
        raise ValueError("schema-3 articulation_body_names must be non-empty and unique")
    selected_body_names_raw = contract["body_names"]
    selected_body_indices = contract["body_indices"]
    if (
        not isinstance(selected_body_names_raw, (list, tuple))
        or not isinstance(selected_body_indices, (list, tuple))
        or len(selected_body_names_raw) != len(selected_body_indices)
        or not selected_body_names_raw
    ):
        raise ValueError("schema-3 selected body names/indices are malformed")
    selected_body_names = [str(value) for value in selected_body_names_raw]
    if any(not value for value in selected_body_names) or len(set(selected_body_names)) != len(
        selected_body_names
    ):
        raise ValueError("schema-3 selected body names must be non-empty and unique")
    try:
        parsed_body_indices = []
        for raw_index in selected_body_indices:
            index = int(raw_index)
            if isinstance(raw_index, bool) or float(raw_index) != float(index):
                raise ValueError
            parsed_body_indices.append(index)
        if any(
            index < 0 or index >= len(articulation_body_names)
            for index in parsed_body_indices
        ):
            raise IndexError
        resolved_selected = [
            articulation_body_names[index] for index in parsed_body_indices
        ]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("schema-3 body_indices are outside articulation_body_names") from exc
    if resolved_selected != selected_body_names:
        raise ValueError("schema-3 selected body names do not match articulation body indices")

    segment_lengths = contract["motion_segment_lengths"]
    clip_fps = contract["motion_clip_fps"]
    kinematics = contract["motion_kinematics_contracts"]
    if not isinstance(segment_lengths, (list, tuple)) or not segment_lengths:
        raise ValueError("schema-3 motion_segment_lengths must be positive")
    try:
        parsed_segment_lengths = [int(value) for value in segment_lengths]
    except (TypeError, ValueError) as exc:
        raise ValueError("schema-3 motion_segment_lengths must be positive") from exc
    if any(value <= 0 for value in parsed_segment_lengths):
        raise ValueError("schema-3 motion_segment_lengths must be positive")
    if (
        not isinstance(clip_fps, (list, tuple))
        or len(clip_fps) != len(segment_lengths)
        or not isinstance(kinematics, (list, tuple))
        or len(kinematics) != len(segment_lengths)
    ):
        raise ValueError("schema-3 motion fps/kinematics counts must match segments")
    try:
        policy_hz = 1.0 / float(contract["policy_step_dt_s"])
        parsed_fps = [float(value) for value in clip_fps]
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("schema-3 motion_clip_fps/policy_step_dt_s are invalid") from exc
    if any(
        not math.isfinite(value)
        or value <= 0.0
        or not math.isclose(value, policy_hz, rel_tol=0.0, abs_tol=1e-9)
        for value in parsed_fps
    ):
        raise ValueError("schema-3 every motion clip fps must equal the policy rate")
    expected_body_order = articulation_body_names
    clip_exact_flags = []
    for index, item in enumerate(kinematics):
        if not isinstance(item, Mapping):
            raise ValueError(f"schema-3 motion kinematics clip {index} must be an object")
        missing_item = [
            key
            for key in (
                "schema_version",
                "body_pos_point",
                "body_lin_vel_point",
                "body_names",
                "exact",
            )
            if key not in item
        ]
        if missing_item:
            raise ValueError(
                f"schema-3 motion kinematics clip {index} is missing "
                + ", ".join(missing_item)
            )
        if not isinstance(item["exact"], bool):
            raise ValueError(f"schema-3 motion kinematics clip {index} exact must be boolean")
        clip_exact = item["exact"] is True
        clip_exact_flags.append(clip_exact)
        try:
            item_schema = (
                None if item.get("schema_version") is None else int(item["schema_version"])
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has invalid schema_version"
            )
        if item_schema not in (None, 1, 2):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has unsupported schema_version "
                f"{item_schema!r}"
            )
        pos_point = item.get("body_pos_point")
        vel_point = item.get("body_lin_vel_point")
        if pos_point not in (None, "link_origin") or vel_point not in (
            None,
            "link_origin",
            "center_of_mass",
        ):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} has invalid point semantics"
            )
        raw_body_names = item.get("body_names")
        if raw_body_names is not None:
            if not isinstance(raw_body_names, (list, tuple)):
                raise ValueError(
                    f"schema-3 motion kinematics clip {index} body_names must be an array or null"
                )
            item_body_names = [str(value) for value in raw_body_names]
            if (
                not item_body_names
                or any(not value for value in item_body_names)
                or len(set(item_body_names)) != len(item_body_names)
                or item_body_names != expected_body_order
            ):
                raise ValueError(
                    f"schema-3 motion kinematics clip {index} body_names do not match the "
                    "runtime articulation"
                )
        status = item.get("status")
        if (not clip_exact) and (not isinstance(status, str) or not status.strip()):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} status must be non-empty"
            )
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} status must be non-empty when present"
            )
        if clip_exact and (
            item_schema != 2
            or pos_point != "link_origin"
            or vel_point != "center_of_mass"
            or raw_body_names is None
        ):
            raise ValueError(
                f"schema-3 motion kinematics clip {index} claims exact without an exact "
                "schema-2 body order"
            )
        if not clip_exact and item_schema == 2:
            raise ValueError(
                f"schema-3 motion kinematics clip {index} is schema-2 but marked inexact"
            )
    resolve_motion_body_lin_vel_points(kinematics)
    motion_exact = contract["motion_kinematics_exact"]
    if not isinstance(motion_exact, bool) or motion_exact != all(clip_exact_flags):
        raise ValueError(
            "schema-3 motion_kinematics_exact disagrees with the per-clip contracts"
        )


def validate_schema3_contract(contract: Mapping) -> None:
    """Validate the formal-exact subset of the schema-3 execution contract."""

    validate_schema3_contract_structure(contract)
    if validate_action_ball_training_authorization(contract):
        raise ValueError(
            "schema-3 formal validation rejects diagnostic_unauthorized "
            "action-ball contracts"
        )
    if contract["motion_kinematics_exact"] is not True:
        raise ValueError("schema-3 formal lineage requires motion_kinematics_exact=true")


def checkpoint_claims_contract(checkpoint: Mapping) -> bool:
    """Return whether checkpoint infos claim any adjacent training-contract binding."""

    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    return isinstance(infos, Mapping) and any(
        key in infos
        for key in (
            CHECKPOINT_CONTRACT_SCHEMA_KEY,
            CHECKPOINT_CONTRACT_SHA_KEY,
            CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY,
        )
    )


def checkpoint_contract_binding(checkpoint: Mapping) -> tuple[int | None, str | None]:
    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    if not isinstance(infos, Mapping):
        return None, None
    schema_raw = infos.get(CHECKPOINT_CONTRACT_SCHEMA_KEY)
    digest_raw = infos.get(CHECKPOINT_CONTRACT_SHA_KEY)
    schema = schema_raw if type(schema_raw) is int else None
    digest = None if digest_raw is None else str(digest_raw).strip().lower()
    return schema, digest


def checkpoint_contract_lineage_exact(checkpoint: Mapping) -> bool:
    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    value = (
        infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY)
        if isinstance(infos, Mapping)
        else None
    )
    if value is None:
        return False
    if type(value) is int and value in (0, 1):
        return value == 1
    raise ValueError(
        f"{CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY} must be a plain integer 0/1"
    )


def require_checkpoint_contract_binding(
    checkpoint: Mapping, *, schema: int, sha256: str, require_lineage_exact: bool = True
) -> None:
    if type(schema) is not int:
        raise ValueError("expected training-contract schema must be a plain integer")
    if type(require_lineage_exact) is not bool:
        raise ValueError("require_lineage_exact must be an exact boolean")
    bound_schema, bound_sha = checkpoint_contract_binding(checkpoint)
    expected_sha = str(sha256).strip().lower()
    if bound_schema != schema or bound_sha != expected_sha:
        raise ValueError(
            "checkpoint is not bound to the adjacent training contract: "
            f"checkpoint schema/sha={bound_schema}/{bound_sha}, file={schema}/{expected_sha}"
        )
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise ValueError("training-contract SHA256 is malformed")
    infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
    if (
        schema == TRAINING_CONTRACT_SCHEMA_VERSION
        and (
            not isinstance(infos, Mapping)
            or CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY not in infos
        )
    ):
        raise ValueError(
            "schema-3 checkpoint contract binding must explicitly declare "
            f"{CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY}=0 or 1"
        )
    lineage_is_exact = checkpoint_contract_lineage_exact(checkpoint)
    if require_lineage_exact and not lineage_is_exact:
        lineage_exact = (
            infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY)
            if isinstance(infos, Mapping)
            else None
        )
        raise ValueError(
            "checkpoint contract binding is not exact-lineage eligible "
            f"({CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY}={lineage_exact!r})"
        )
