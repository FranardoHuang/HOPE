"""Exact native-MuJoCo port of the Isaac ActionBall robot/table guard.

The Isaac rule is a pose keep-out, not a resolved-contact test: 63 pinned robot
proxy OBBs for 62 collision meshes and one live racket-blade OBB are compared with the
five inflated table-assembly AABBs at every physics substep.  Broadening each
OBB to a world AABB is only the PREFILTER; the verdict is the exact 15-axis
separating-axis test.  This module consumes the same immutable component
artifact and fails closed if any source identity drifts.

人话:这个文件是 Isaac 撞桌判据的 MuJoCo 复刻。它一度停在旧版"胀成正方盒子就算撞",
比 Isaac 现役判据更容易误报;现在两边都用精确 SAT,跨引擎结论才可比。
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import isaac_live_constants


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_TERMINATION_CONFIG = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
EXPECTED_ISAAC_TERMINATION_CONFIG_SEMANTIC_AST_SHA256 = (
    "a6607ef2df55d25ab37d41ea40f1c8ab51b31ad41b278450c2848f5a2fdff4d3"
)
ISAAC_TERMINATION_CALLABLES = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/terminations.py"
)
EXPECTED_ISAAC_TERMINATION_CALLABLES_SEMANTIC_AST_SHA256 = (
    "25683f03897c252577ae8384e4bca6d4d62ab7ac1c60c2b02f329a8a57a37de9"
)
#: Hoisted so the re-pin helpers in the tests read the live selector list
#: instead of keeping a fourth hand-copy of it.  A selector added here is
#: covered everywhere at once.
ISAAC_TERMINATION_CONFIG_SELECTORS = (
    ("assignment", "TABLE_HIT_FORCE_THRESHOLD_N"),
    ("assignment", "TABLE_HIT_MARGIN_M"),
    # The proxy pointer belongs in this slice: swapping which artifact the
    # guard reads swaps the robot the guard believes in.
    ("assignment", "TABLE_COLLISION_PROXY_ARTIFACT_PATH"),
    ("assignment", "TABLE_COLLISION_PROXY_ARTIFACT_SHA256"),
    ("function", "table_hit_done_term"),
    ("class_header", "HOPEDeployParityTerminationsCfg"),
    ("class_assignments", "HOPEDeployParityTerminationsCfg|robot_hit_table"),
    ("class_header", "HOPEActionBallTerminationsCfg"),
)
ISAAC_TERMINATION_CALLABLE_SELECTORS = (
    ("assignment", "_A3_COLLISION_PROXY_SOURCE_URDF_RELATIVE"),
    ("assignment", "_A3_COLLISION_PROXY_SOURCE_URDF_SHA256"),
    ("assignment", "_A3_COLLISION_PROXY_ISAACLAB_ASSET_HASH"),
    ("assignment", "_A3_COLLISION_PROXY_ASSET_HASH_EXCLUDED_CONFIG_KEYS"),
    ("assignment", "_A3_COLLISION_PROXY_PLANT_IDENTITY_KIND"),
    ("assignment", "_A3_COLLISION_PROXY_PLANT_ASSET_ROOT_NAME"),
    ("assignment", "_A3_COLLISION_PROXY_SOURCE_COMPONENT_COUNT"),
    ("assignment", "_A3_COLLISION_PROXY_COMPONENT_COUNT"),
    ("assignment", "_A3_COLLISION_PROXY_MUJOCO_MJCF_RELATIVE"),
    ("assignment", "_A3_COLLISION_PROXY_MUJOCO_MJCF_SHA256"),
    ("assignment", "_A3_COLLISION_PROXY_MUJOCO_TARGET_COLLIDERS"),
    ("assignment", "_A3_COLLISION_PROXY_LEFT_GRIPPER_SOURCE_LINKS"),
    ("assignment", "_A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256"),
    ("assignment", "_A3_COLLISION_PROXY_RUNTIME_USD_TOTAL_FILE_BYTES"),
    ("assignment", "_A3_COLLISION_PROXY_RUNTIME_USD_FILES"),
    ("assignment", "_TABLE_GUARD_OBSTACLE_ROLES"),
    ("function", "_rederive_isaaclab_asset_hash"),
    ("function", "_verify_live_bundle_is_a_cache_of_this_plant"),
    ("function", "geometric_table_contact_hit_mask"),
    ("class", "TableGuardAttribution"),
    ("function", "_obb_aabb_sat_overlap"),
    ("function", "_geometric_table_contact_attribution_unchecked"),
    ("function", "geometric_table_contact_attribution"),
    ("function", "_geometric_table_contact_hit_mask_unchecked"),
    ("function", "sample_robot_table_contact_current"),
    ("function", "robot_hit_table"),
)
ISAAC_ACTION_LATCH = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_actions.py"
)
EXPECTED_ISAAC_ACTION_LATCH_SEMANTIC_AST_SHA256 = (
    "e4a0560fdb594adfad4496c5038210c7d19502da25b058852cd3d00b87e79431"
)
CANONICAL_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
EXPECTED_CANONICAL_MJCF_SHA256 = (
    "70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a"
)
MUJOCO_IDENTITY_MANIFEST = REPO_ROOT / "configs/a3_mujoco_identity_v2_20260803.json"
EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256 = (
    "b8fc5deaaff8d213c2d077a0e7892b30d7f5a6c77c3d06dc029e3a2616d54d91"
)
CANONICAL_MUJOCO_IDENTITY_PY = (
    REPO_ROOT / "hope_training/whole_body_tracking/scripts/canonical_mujoco_identity.py"
)
EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256 = (
    "e43609988a371a76e5daab7545c608338ba159100c52cb50dc61b12a872fe2e1"
)
EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256 = (
    "472219ae346d9217b7d1af860d462a18d6ed8507c5cbb9c0f1ddcd6f964dfd7a"
)
COLLISION_PROXY_ARTIFACT = (
    REPO_ROOT
    / "configs/a3_table_collision_proxy_a3p0807_20260808/"
    "a3_table_collision_components.v2.json"
)
EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256 = (
    "7f26e55b2f24b02f751c7b078f94426ec2524c810b0ff6cb5cfea58e884e07cc"
)
EXPECTED_COLLISION_PROXY_SOURCE_COMPONENT_COUNT = 62
EXPECTED_COLLISION_PROXY_COMPONENT_COUNT = 63
EXPECTED_COLLISION_PROXY_MUJOCO_TARGET_COLLIDERS = (
    "right_hand_finger_collision",
    "right_hand_palm_collision",
    "right_hand_thumb_collision",
    "right_racket_collision",
    "right_racket_handle_collision",
    "right_wrist_yaw_collision",
)
##
# Plant identity for the collision proxy.
#
# Until 2026-08-08 this lane checked the artifact's file digest, schema, body
# order, self-sealed content digest, component count, finiteness and coverage
# -- and never opened the ``source_urdf`` or ``runtime_usd_bundle`` blocks the
# artifact writes about itself, although the Isaac lane did.  That asymmetry
# was not a style difference: it meant a proxy of a different robot, correctly
# sealed, would have been accepted here and refused there.
#
# The two lanes now ask the same question, in the form each can answer.  This
# lane has no Pod USD bundle, so it re-derives IsaacLab's ``.asset_hash`` from
# the converter configuration the artifact carries plus the tracked plant URDF
# on disk.  Same derivation, same verdict, no bundle required.
##
PLANT_SOURCE_URDF = (
    REPO_ROOT / "agi/URDF/A3P-P1-32dof-0807-OP3-pingpang/urdf/model.urdf"
)
EXPECTED_PLANT_SOURCE_URDF_RELATIVE = (
    "agi/URDF/A3P-P1-32dof-0807-OP3-pingpang/urdf/model.urdf"
)
EXPECTED_PLANT_SOURCE_URDF_SHA256 = (
    "15c83f5f3beea71350583143aef4d622d5219df65a0bed9a660a0edb7d388d09"
)
EXPECTED_RUNTIME_USD_BUNDLE_TREE_SHA256 = (
    "365ba37edd5e5e1d4fac22f2cbb3ec871ead7bb49aeadb50161ef523a9ae6747"
)
EXPECTED_RUNTIME_USD_TOTAL_FILE_BYTES = 60519988
#: The same six rows the Isaac gate compares, in the same canonical order.
EXPECTED_RUNTIME_USD_FILES = [
    {
        "path": ".asset_hash",
        "sha256": "a78a2f8fb207cbf479cc1b308cf9d3c58e1a55eb7da9dbc2caf34be697e9c993",
        "size": 32,
    },
    {
        "path": "config.yaml",
        "sha256": "f349c3f4d80a915f5ca3ce53d49785dfd7e6eeca2645dcd7b402d4d8a2288eb9",
        "size": 1685,
    },
    {
        "path": "configuration/model_base.usd",
        "sha256": "108a4b45b96a8db8396d3a8feb995481c5db87efcde80066e6347ed494e658fc",
        "size": 60504873,
    },
    {
        "path": "configuration/model_physics.usd",
        "sha256": "390cf66cc052ea697e88e9ef0131bf7e2eee96e70c35c0861e1ce33d363747f5",
        "size": 11078,
    },
    {
        "path": "configuration/model_sensor.usd",
        "sha256": "4e16201f146db3240b8a0082ae14e3aca41255a75812c5331bf8f4e39701355c",
        "size": 687,
    },
    {
        "path": "model.usd",
        "sha256": "13e5ecfe02238fbf1d20c13ed7177e18ed93d84bca8e0a592b6605f7fb85f351",
        "size": 1633,
    },
]
EXPECTED_ISAACLAB_ASSET_HASH = "676efde5febed3c0fde0f2ad59650cdf"
EXPECTED_PLANT_IDENTITY_KIND = "a3_collision_proxy_plant_identity_v2"
EXPECTED_PLANT_ASSET_ROOT_NAME = "agibot_a3p_p1_0807_v1"
ASSET_HASH_EXCLUDED_CONFIG_KEYS = ("asset_path", "usd_dir", "usd_file_name")
# The 20 OmniPicker3 left-gripper collision links the 0807 plant introduces.
EXPECTED_LEFT_GRIPPER_SOURCE_LINKS = (
    "left_base_link",
    "left_link1",
    "left_link10",
    "left_link11",
    "left_link11-1",
    "left_link13",
    "left_link14",
    "left_link14-1",
    "left_link15",
    "left_link17",
    "left_link18",
    "left_link2",
    "left_link3",
    "left_link4",
    "left_link4-1",
    "left_link6",
    "left_link7",
    "left_link7-1",
    "left_link8",
    "left_link9",
)
EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256 = (
    "f6aab7524a3b6583ae7ced8da8b2b5d9d1bbe0ea0c72b3b688fefaf6ff66cc6a"
)

TABLE_GUARD_MARGIN_M = 0.02
COMPONENT_WORLD_AABB_GUARD_M = 1.0e-6
RACKET_BODY_NAME = "right_wrist_yaw_Link"
RACKET_BLADE_CENTER_OFFSET_WRIST_M = np.asarray(
    (0.206194, 0.025474, 0.028020), dtype=np.float64
)
RACKET_BLADE_LOCAL_HALF_AXES_M = np.diag(
    np.asarray((0.082, 0.008, 0.082), dtype=np.float64)
)
TABLE_ASSEMBLY_ROLES = ("top", "keepout", "net", "post_left", "post_right")
TABLE_CONTACT_BODY_NAMES = (
    "pelvis_link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "waist_yaw_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "waist_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "head_yaw_Link",
    "left_shoulder_pitch_Link",
    "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "head_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
)


class TableTerminationContractError(RuntimeError):
    """The exact Isaac robot/table termination cannot be reproduced."""


# --------------------------------------------------------------------------
# Live-value parity for the constants above
#
# 人话:上面那几行(桌面外扩 2 cm、拍面盒子的中心与半轴、五段桌台的名字、
# 碰撞代理文件与它的 SHA)真源都在 Isaac 侧,这里存的是**副本**。副本此前只被
# 语义 AST 指纹罩着 —— 而指纹只说"源文件那几个节点的字节没动过"。源文件一动,
# 把指纹重钉成新值是一行的事,副本跟没跟上没人查:5ed998f1 就是这么让桌面终局
# 的复刻停在原地两天的。
#
# 下面这组检查改成**直接把 Isaac 现在那个数读出来跟副本比**。它读的是
# ``table_hit_done_term()`` 真正塞进 DoneTerm 的 ``params``,不是同名模块常量 ——
# 所以"把常量改了"和"把这个 term 改成用另一个常量"两种漂移都拦得住。
# --------------------------------------------------------------------------

#: The Isaac factory whose ``params={...}`` is what the live ``robot_hit_table``
#: DoneTerm is actually constructed with.
ISAAC_TABLE_TERM_FACTORY = "table_hit_done_term"


def _mirrored_blade_half_extents_m() -> tuple:
    """The three half extents this port encodes as a diagonal half-axis matrix."""

    axes = np.asarray(RACKET_BLADE_LOCAL_HALF_AXES_M, dtype=np.float64)
    if axes.shape != (3, 3) or not np.array_equal(axes, np.diag(np.diagonal(axes))):
        raise TableTerminationContractError(
            "racket blade half-axis matrix is no longer the Isaac diagonal box"
        )
    return tuple(float(value) for value in np.diagonal(axes))


def mirrored_isaac_constant_entries() -> tuple:
    """``(key, source, selector, mirrored_value)`` for every hand copy here.

    Built on call, not at import, because the source paths are module globals
    that the drift tests repoint at mutated copies.
    """

    return (
        (
            "table_guard_margin_m",
            ISAAC_TERMINATION_CONFIG,
            ("function_return_param", ISAAC_TABLE_TERM_FACTORY, "margin"),
            TABLE_GUARD_MARGIN_M,
        ),
        (
            "racket_body_name",
            ISAAC_TERMINATION_CONFIG,
            ("function_return_param", ISAAC_TABLE_TERM_FACTORY, "racket_body_name"),
            RACKET_BODY_NAME,
        ),
        (
            "racket_blade_center_offset_wrist_m",
            ISAAC_TERMINATION_CONFIG,
            (
                "function_return_param",
                ISAAC_TABLE_TERM_FACTORY,
                "racket_blade_center_offset_wrist_m",
            ),
            tuple(
                float(value)
                for value in np.asarray(
                    RACKET_BLADE_CENTER_OFFSET_WRIST_M, dtype=np.float64
                )
            ),
        ),
        (
            "racket_blade_half_extents_m",
            ISAAC_TERMINATION_CONFIG,
            (
                "function_return_param",
                ISAAC_TABLE_TERM_FACTORY,
                "racket_blade_half_extents_m",
            ),
            _mirrored_blade_half_extents_m(),
        ),
        (
            "collision_proxy_artifact_repo_relative_path",
            ISAAC_TERMINATION_CONFIG,
            (
                "function_return_param",
                ISAAC_TABLE_TERM_FACTORY,
                "collision_proxy_artifact_path",
            ),
            COLLISION_PROXY_ARTIFACT.relative_to(REPO_ROOT).as_posix(),
        ),
        (
            "collision_proxy_artifact_sha256",
            ISAAC_TERMINATION_CONFIG,
            (
                "function_return_param",
                ISAAC_TABLE_TERM_FACTORY,
                "collision_proxy_artifact_sha256",
            ),
            EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256,
        ),
        (
            "table_assembly_roles",
            ISAAC_TERMINATION_CALLABLES,
            ("assignment", "_TABLE_GUARD_OBSTACLE_ROLES"),
            TABLE_ASSEMBLY_ROLES,
        ),
    )


def live_isaac_constant_blockers() -> tuple:
    """Every hand-copied Isaac constant here that no longer equals the live one."""

    try:
        entries = mirrored_isaac_constant_entries()
    except TableTerminationContractError as exc:
        return (f"table_guard_mirror_self_inconsistent:{exc}",)
    return isaac_live_constants.parity_blockers("table_guard", entries)


def _strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _sha256_file(path: Path, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TableTerminationContractError(f"cannot read {label} source") from exc


def _portable_ast_dump(node: ast.AST) -> str:
    """Serialize source semantics without Python-version-only empty fields."""

    def normalize(value: Any) -> Any:
        # Python 3.8 alone materializes this semantics-free wrapper around
        # subscript slices; Python 3.9+ parses the child directly.
        if isinstance(value, ast.Index):
            return normalize(value.value)
        if isinstance(value, ast.ExtSlice):
            return [
                "Tuple",
                [
                    ["elts", normalize(value.dims)],
                    ["ctx", normalize(ast.Load())],
                ],
            ]
        if isinstance(value, ast.AST):
            fields = []
            for field, child in ast.iter_fields(value):
                # Python 3.12 added empty ``type_params`` to defs/classes.  It
                # does not change the semantics of source accepted by 3.10.
                if field == "type_params" and child == []:
                    continue
                fields.append([field, normalize(child)])
            return [type(value).__name__, fields]
        if isinstance(value, list):
            return [normalize(child) for child in value]
        # ``ast.Constant`` may contain JSON-incompatible Python literals.
        # Encode them explicitly so the semantic digest is stable on every
        # supported interpreter instead of depending on ``repr`` details.
        if value is Ellipsis:
            return ["__constant__", "ellipsis"]
        if isinstance(value, bytes):
            return ["__constant_bytes_hex__", value.hex()]
        if isinstance(value, complex):
            return ["__constant_complex__", value.real, value.imag]
        return value

    return json.dumps(
        normalize(node),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _semantic_ast_sha256(
    path: Path, selectors: tuple[tuple[str, str], ...], label: str
) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise TableTerminationContractError(f"cannot parse {label} source") from exc
    nodes = tuple(ast.walk(tree))

    def assignment_names(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            return ()
        return tuple(
            target.id for target in targets if isinstance(target, ast.Name)
        )

    def class_header(node: ast.ClassDef) -> dict[str, Any]:
        return {
            "decorators": [
                _portable_ast_dump(item)
                for item in node.decorator_list
            ],
            "bases": [
                _portable_ast_dump(item) for item in node.bases
            ],
            "keywords": [
                _portable_ast_dump(item) for item in node.keywords
            ],
        }

    selected = []
    for kind, name in selectors:
        if kind == "class":
            matches = [
                node
                for node in nodes
                if isinstance(node, ast.ClassDef) and node.name == name
            ]
        elif kind == "function":
            matches = [
                node
                for node in nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ]
        elif kind == "assignment":
            matches = [node for node in nodes if name in assignment_names(node)]
        elif kind == "class_header":
            classes = [
                node
                for node in nodes
                if isinstance(node, ast.ClassDef) and node.name == name
            ]
            matches = [] if len(classes) != 1 else [class_header(classes[0])]
        elif kind == "class_assignments":
            try:
                class_name, raw_names = name.split("|", 1)
            except ValueError as exc:
                raise TableTerminationContractError(
                    f"malformed {label} class-assignment selector"
                ) from exc
            required_names = tuple(raw_names.split(","))
            if not required_names or any(not item for item in required_names):
                raise TableTerminationContractError(
                    f"malformed {label} class-assignment selector"
                )
            classes = [
                node
                for node in nodes
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            matches = []
            if len(classes) == 1:
                assignments = [
                    node
                    for node in classes[0].body
                    if set(assignment_names(node)) & set(required_names)
                ]
                observed_names = tuple(
                    item
                    for node in assignments
                    for item in assignment_names(node)
                    if item in required_names
                )
                if (
                    len(observed_names) == len(required_names)
                    and set(observed_names) == set(required_names)
                ):
                    matches = [
                        {
                            "class_header": class_header(classes[0]),
                            "assignments_in_source_order": [
                                {
                                    "names": assignment_names(node),
                                    "ast": _portable_ast_dump(node),
                                }
                                for node in assignments
                            ],
                        }
                    ]
        else:
            raise TableTerminationContractError(
                f"unsupported {label} semantic selector {kind}:{name}"
            )
        if len(matches) != 1:
            raise TableTerminationContractError(
                f"{label} semantic selector {kind}:{name} is not unique"
            )
        selected.append(
            {
                "kind": kind,
                "name": name,
                "ast": (
                    _portable_ast_dump(matches[0])
                    if isinstance(matches[0], ast.AST)
                    else matches[0]
                ),
            }
        )
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_isaac_source_authority() -> dict[str, str]:
    """Reopen exact table term/predicate/latch AST slices, ignoring unrelated WIP."""

    config_sha = _semantic_ast_sha256(
        ISAAC_TERMINATION_CONFIG,
        ISAAC_TERMINATION_CONFIG_SELECTORS,
        "Isaac robot/table termination config",
    )
    callable_sha = _semantic_ast_sha256(
        ISAAC_TERMINATION_CALLABLES,
        ISAAC_TERMINATION_CALLABLE_SELECTORS,
        "Isaac robot/table termination callables",
    )
    action_latch_sha = _semantic_ast_sha256(
        ISAAC_ACTION_LATCH,
        (
            ("class", "_PhysicsSubstepTableContactLatch"),
            ("function", "_sample_table_contact_current"),
            ("function", "apply_actions"),
            ("function", "finalize_table_contact_substep_readback"),
        ),
        "Isaac robot/table action latch",
    )
    if config_sha != EXPECTED_ISAAC_TERMINATION_CONFIG_SEMANTIC_AST_SHA256:
        raise TableTerminationContractError(
            "Isaac robot/table termination config semantic AST SHA-256 drifted"
        )
    if (
        callable_sha
        != EXPECTED_ISAAC_TERMINATION_CALLABLES_SEMANTIC_AST_SHA256
    ):
        raise TableTerminationContractError(
            "Isaac robot/table termination callables semantic AST SHA-256 drifted"
        )
    if action_latch_sha != EXPECTED_ISAAC_ACTION_LATCH_SEMANTIC_AST_SHA256:
        raise TableTerminationContractError(
            "Isaac robot/table action-latch semantic AST SHA-256 drifted"
        )
    # 人话:上面三道门只说"源文件的字节跟我钉的一样"。源文件一动,把这三行重钉
    # 成新值是一行的事,而这个文件顶部那几个手抄常量跟没跟上,过去没有任何机制
    # 在看。这道门把手抄件跟 Isaac 现役 ``robot_hit_table`` 真正用的参数逐个比值,
    # 所以"光重钉指纹"从今往后不再放行。
    constant_blockers = live_isaac_constant_blockers()
    if constant_blockers:
        raise TableTerminationContractError(
            "Isaac robot/table guard constants were hand-copied and no longer "
            "equal the live Isaac source (re-pinning the AST SHA does not port "
            "them): " + "; ".join(constant_blockers)
        )
    return {
        "config_semantic_ast_sha256": config_sha,
        "callables_semantic_ast_sha256": callable_sha,
        "action_latch_semantic_ast_sha256": action_latch_sha,
        "live_constant_parity": (
            "margin_racket_body_blade_box_proxy_path_sha_assembly_roles"
        ),
        "live_constant_parity_constants_compared": str(
            len(mirrored_isaac_constant_entries())
        ),
    }


def _owner_frame_contract(
    mujoco: Any, model: Any, *, body_name_prefix: str = ""
) -> dict[str, Any]:
    """Serialize the exact local frames that give the 62 OBB rows meaning."""

    if (
        type(body_name_prefix) is not str
        or body_name_prefix != body_name_prefix.strip()
        or body_name_prefix.startswith("/")
        or "//" in body_name_prefix
        or (body_name_prefix and not body_name_prefix.endswith("/"))
    ):
        raise TableTerminationContractError(
            "MuJoCo table-guard body namespace prefix is malformed"
        )
    body_ids: list[int] = []
    for name in TABLE_CONTACT_BODY_NAMES:
        live_name = body_name_prefix + name
        body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, live_name)
        )
        if body_id <= 0:
            raise TableTerminationContractError(
                f"MuJoCo model is missing exact A3 table-guard body {live_name!r}"
            )
        body_ids.append(body_id)
    if len(set(body_ids)) != len(TABLE_CONTACT_BODY_NAMES):
        raise TableTerminationContractError(
            "MuJoCo table-guard body mapping is not a 32-body bijection"
        )
    selected_names = dict(zip(body_ids, TABLE_CONTACT_BODY_NAMES))
    rows: list[dict[str, Any]] = []
    for name, body_id in zip(TABLE_CONTACT_BODY_NAMES, body_ids):
        parent_id = int(model.body_parentid[body_id])
        if parent_id == 0:
            parent_name = "__world__"
        else:
            parent_name = selected_names.get(parent_id)
            if parent_name is None:
                raise TableTerminationContractError(
                    f"table-guard body {name!r} has an unregistered parent"
                )
        position = np.asarray(model.body_pos[body_id], dtype=np.float64)
        quaternion = np.asarray(model.body_quat[body_id], dtype=np.float64)
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or not np.isclose(
                float(np.linalg.norm(quaternion)), 1.0, rtol=0.0, atol=1.0e-12
            )
        ):
            raise TableTerminationContractError(
                f"table-guard body {name!r} has a malformed owner-local frame"
            )
        rows.append(
            {
                "name": name,
                "parent": parent_name,
                "body_pos_m": position.tolist(),
                "body_quat_wxyz": quaternion.tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "a3_table_collision_owner_local_frame_contract_v1",
        "body_rows": rows,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    payload["body_ids"] = body_ids
    return payload


def _assert_owner_frame_contract_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    if (
        expected.get("kind") != "a3_table_collision_owner_local_frame_contract_v1"
        or observed.get("kind") != expected.get("kind")
        or expected.get("body_rows") != observed.get("body_rows")
        or expected.get("content_sha256") != observed.get("content_sha256")
    ):
        raise TableTerminationContractError(
            "live MuJoCo owner-local body frames differ from the pre-registered plant"
        )


def consume_verified_owner_frame_contract(
    mujoco: Any, verified_identity: Any
) -> dict[str, Any]:
    """Derive owner-frame identity from one already-verified base model.

    The canonical verifier owns compilation and its before/after unchanged
    checks.  Consumers use this narrow adapter so they do not compile the same
    72 MB plant a second time or reimplement the 32-body projection.
    """

    consume = getattr(verified_identity, "consume_verified_model", None)
    if not callable(consume):
        raise TableTerminationContractError(
            "verified MuJoCo identity cannot expose its checked model"
        )
    contract = consume(lambda model: _owner_frame_contract(mujoco, model))
    if (
        not isinstance(contract, Mapping)
        or contract.get("kind")
        != "a3_table_collision_owner_local_frame_contract_v1"
        or not isinstance(contract.get("body_rows"), list)
        or len(contract["body_rows"]) != len(TABLE_CONTACT_BODY_NAMES)
        or not isinstance(contract.get("content_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", contract["content_sha256"]) is None
    ):
        raise TableTerminationContractError(
            "verified MuJoCo owner-local frame contract is malformed"
        )
    return dict(contract)


def _load_canonical_identity_module() -> Any:
    verifier_sha = _sha256_file(
        CANONICAL_MUJOCO_IDENTITY_PY, "canonical MuJoCo identity verifier"
    )
    if verifier_sha != EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256:
        raise TableTerminationContractError(
            "canonical MuJoCo identity verifier SHA-256 drifted"
        )
    module_name = "_table_termination_canonical_mujoco_identity"
    spec = importlib.util.spec_from_file_location(
        module_name, CANONICAL_MUJOCO_IDENTITY_PY
    )
    if spec is None or spec.loader is None:
        raise TableTerminationContractError(
            "cannot import canonical MuJoCo identity verifier"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - imported verifier is fail-closed
        raise TableTerminationContractError(
            "cannot load canonical MuJoCo identity verifier"
        ) from exc
    return module


_REGISTERED_PLANT_KEYS = {
    "root_mjcf_sha256",
    "identity_manifest_path",
    "identity_manifest_sha256",
    "portable_identity_sha256",
}


def _registered_plant(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize one atomic plant-generation join for the live guard."""

    row = (
        {
            "root_mjcf_sha256": EXPECTED_CANONICAL_MJCF_SHA256,
            "identity_manifest_path": MUJOCO_IDENTITY_MANIFEST,
            "identity_manifest_sha256": EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256,
            "portable_identity_sha256": EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256,
        }
        if value is None
        else dict(value)
    )
    if set(row) != _REGISTERED_PLANT_KEYS:
        raise TableTerminationContractError(
            "registered MuJoCo plant identity surface differs"
        )
    for key in _REGISTERED_PLANT_KEYS - {"identity_manifest_path"}:
        if (
            type(row[key]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None
        ):
            raise TableTerminationContractError(
                "registered MuJoCo plant identity digest differs"
            )
    row["identity_manifest_path"] = Path(row["identity_manifest_path"])
    return row


@lru_cache(maxsize=4)
def _verified_registered_owner_frames(
    mujoco: Any,
    selected_root: str,
    identity_manifest_path: str,
    identity_manifest_sha256: str,
    portable_identity_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    """Compile the registered base once per process/toolchain and retain no model."""

    identity_module = _load_canonical_identity_module()
    verified = identity_module.verify_exact_mujoco_identity(
        mjcf_path=selected_root,
        expected_manifest_path=identity_manifest_path,
        trusted_expected_manifest_sha256=identity_manifest_sha256,
    )
    if verified.portable_identity_sha256 != portable_identity_sha256:
        raise TableTerminationContractError(
            "portable MuJoCo plant identity SHA-256 drifted"
        )
    frames = consume_verified_owner_frame_contract(mujoco, verified)
    return (
        frames,
        verified.portable_identity_sha256,
        verified.verification_receipt_sha256,
    )


def bind_pre_registered_owner_frames(
    mujoco: Any,
    model: Any,
    mjcf_path: Path | str,
    *,
    body_name_prefix: str = "",
    registered_plant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the live augmented scene to the registered base plant and OBB frames."""

    selected_root = Path(mjcf_path).expanduser().resolve()
    registered = _registered_plant(registered_plant)
    # A one-shot run snapshots the plant into a fresh namespace.  Its absolute
    # path is not plant identity; exact root bytes, portable source closure and
    # live owner-local frames below are the three independent authorities.
    if (
        _sha256_file(selected_root, "canonical root MJCF")
        != registered["root_mjcf_sha256"]
    ):
        raise TableTerminationContractError("selected root MJCF SHA-256 drifted")
    if (
        _sha256_file(
            registered["identity_manifest_path"], "MuJoCo identity manifest"
        )
        != registered["identity_manifest_sha256"]
    ):
        raise TableTerminationContractError("MuJoCo identity manifest SHA-256 drifted")
    try:
        (
            expected_frames,
            portable_identity_sha256,
            verification_receipt_sha256,
        ) = _verified_registered_owner_frames(
            mujoco,
            str(selected_root),
            str(registered["identity_manifest_path"]),
            registered["identity_manifest_sha256"],
            registered["portable_identity_sha256"],
        )
    except TableTerminationContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - verifier exposes its own error type
        raise TableTerminationContractError(
            "pre-registered portable MuJoCo plant identity did not verify"
        ) from exc
    observed_frames = _owner_frame_contract(
        mujoco, model, body_name_prefix=body_name_prefix
    )
    _assert_owner_frame_contract_equal(expected_frames, observed_frames)
    return {
        "root_mjcf_path": str(selected_root),
        "root_mjcf_sha256": registered["root_mjcf_sha256"],
        "identity_manifest_path": str(registered["identity_manifest_path"]),
        "identity_manifest_sha256": registered["identity_manifest_sha256"],
        "identity_verifier_path": str(CANONICAL_MUJOCO_IDENTITY_PY),
        "identity_verifier_sha256": EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256,
        "portable_identity_sha256": portable_identity_sha256,
        "verification_receipt_sha256": verification_receipt_sha256,
        "owner_local_frame_sha256": expected_frames["content_sha256"],
    }


@dataclass(frozen=True)
class CollisionComponents:
    component_ids: tuple[str, ...]
    owner_indices: np.ndarray
    local_centers_m: np.ndarray
    local_half_axes_m: np.ndarray
    artifact_sha256: str
    content_sha256: str


def _rederive_isaaclab_asset_hash(config: Mapping[str, Any], urdf_path: Path) -> str:
    """Redo IsaacLab's ``.asset_hash`` offline; no Isaac, no Kit, no bundle.

    Byte-compatible with
    ``isaaclab/sim/converters/asset_converter_base.py::_config_to_hash``: MD5
    over ``json.dumps`` of the converter configuration with the three path keys
    removed, then over the source asset file in 64 KiB chunks.
    """

    payload = dict(config)
    for key in ASSET_HASH_EXCLUDED_CONFIG_KEYS:
        payload.pop(key, None)
    digest = hashlib.md5()
    digest.update(json.dumps(payload).encode())
    with open(urdf_path, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_collision_proxy_plant_identity(document: Mapping[str, Any]) -> None:
    """Read the two blocks this lane used to skip, and finish the derivation.

    ``source_urdf`` and ``runtime_usd_bundle`` are the artifact's own claims
    about which robot it measured; the plant URDF on disk is re-hashed rather
    than taken on the artifact's word, and IsaacLab's asset hash is recomputed
    from the carried converter configuration plus those exact bytes.  A proxy
    of any other robot fails the last step no matter how consistent the rest
    of the document is.
    """

    import yaml

    source_urdf = document.get("source_urdf")
    if (
        not isinstance(source_urdf, dict)
        or source_urdf.get("path") != EXPECTED_PLANT_SOURCE_URDF_RELATIVE
        or source_urdf.get("sha256") != EXPECTED_PLANT_SOURCE_URDF_SHA256
    ):
        raise TableTerminationContractError(
            "collision proxy does not name the reviewed A3 plant URDF"
        )
    try:
        observed = hashlib.sha256(PLANT_SOURCE_URDF.read_bytes()).hexdigest()
    except OSError as exc:
        raise TableTerminationContractError(
            "collision proxy plant URDF cannot be read"
        ) from exc
    if observed != EXPECTED_PLANT_SOURCE_URDF_SHA256:
        raise TableTerminationContractError(
            "collision proxy plant URDF on disk differs from its own pin"
        )

    runtime_usd = document.get("runtime_usd_bundle")
    if (
        not isinstance(runtime_usd, dict)
        or runtime_usd.get("bundle_tree_sha256")
        != EXPECTED_RUNTIME_USD_BUNDLE_TREE_SHA256
        or runtime_usd.get("file_count") != 6
        or runtime_usd.get("total_file_bytes")
        != EXPECTED_RUNTIME_USD_TOTAL_FILE_BYTES
        or runtime_usd.get("symlinks_forbidden") is not True
    ):
        raise TableTerminationContractError(
            "collision proxy does not bind the reviewed six-file runtime USD tree"
        )
    bundle_files = runtime_usd.get("files")
    if bundle_files != EXPECTED_RUNTIME_USD_FILES:
        raise TableTerminationContractError(
            "collision proxy runtime USD file map differs from the six-file pin"
        )
    config_sha_in_bundle = {
        str(row["path"]): str(row["sha256"]) for row in bundle_files
    }["config.yaml"]

    identity = document.get("plant_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("kind") != EXPECTED_PLANT_IDENTITY_KIND
        or identity.get("plant_asset_root_name") != EXPECTED_PLANT_ASSET_ROOT_NAME
        or identity.get("isaaclab_asset_hash") != EXPECTED_ISAACLAB_ASSET_HASH
        or identity.get("isaaclab_asset_hash_excluded_config_keys")
        != list(ASSET_HASH_EXCLUDED_CONFIG_KEYS)
    ):
        raise TableTerminationContractError(
            "collision proxy carries no derivation proof for the reviewed plant"
        )
    config_text = identity.get("converter_config_yaml")
    if (
        not isinstance(config_text, str)
        or hashlib.sha256(config_text.encode("ascii")).hexdigest()
        != identity.get("converter_config_sha256")
        or identity.get("converter_config_sha256") != config_sha_in_bundle
    ):
        raise TableTerminationContractError(
            "collision proxy converter configuration is not the pinned config.yaml"
        )
    try:
        config = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise TableTerminationContractError(
            "collision proxy converter configuration is not YAML"
        ) from exc
    if not isinstance(config, dict) or f"/{EXPECTED_PLANT_ASSET_ROOT_NAME}/" not in str(
        config.get("asset_path")
    ):
        raise TableTerminationContractError(
            "collision proxy converter configuration names a different asset package"
        )
    rederived = _rederive_isaaclab_asset_hash(config, PLANT_SOURCE_URDF)
    if rederived != EXPECTED_ISAACLAB_ASSET_HASH:
        raise TableTerminationContractError(
            "collision proxy is not derived from the reviewed plant: IsaacLab "
            f"asset hash recomputes to {rederived}, pinned "
            f"{EXPECTED_ISAACLAB_ASSET_HASH}"
        )
    if document.get("left_gripper_source_links") != list(
        EXPECTED_LEFT_GRIPPER_SOURCE_LINKS
    ):
        raise TableTerminationContractError(
            "collision proxy does not declare the 20 left OmniPicker3 gripper links"
        )


@lru_cache(maxsize=4)
def _load_collision_components_cached(
    artifact_path: str, expected_file_sha256: str
) -> CollisionComponents:
    path = Path(artifact_path)
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise TableTerminationContractError(
            "robot/table collision proxy is not strict ASCII JSON"
        ) from exc
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_file_sha256:
        raise TableTerminationContractError(
            "robot/table collision proxy artifact SHA-256 drifted"
        )
    if not isinstance(document, dict):
        raise TableTerminationContractError("collision proxy root must be an object")
    if (
        document.get("schema_version") != 2
        or document.get("artifact_type")
        != "a3_table_collision_component_multi_obb_v2"
        or tuple(document.get("body_order", ())) != TABLE_CONTACT_BODY_NAMES
    ):
        raise TableTerminationContractError(
            "collision proxy schema or exact 32-body order drifted"
        )
    content_sha = document.get("content_sha256")
    if not isinstance(content_sha, str) or len(content_sha) != 64:
        raise TableTerminationContractError("collision proxy content SHA is malformed")
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != content_sha:
        raise TableTerminationContractError("collision proxy content SHA mismatch")
    _verify_collision_proxy_plant_identity(document)
    mujoco_binding = document.get("mujoco_actual_collision_binding")
    target_colliders = (
        mujoco_binding.get("target_colliders")
        if isinstance(mujoco_binding, dict)
        else None
    )
    binding_sha = (
        mujoco_binding.get("content_sha256")
        if isinstance(mujoco_binding, dict)
        else None
    )
    try:
        expected_mjcf_relative = CANONICAL_MJCF.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise TableTerminationContractError(
            "canonical MuJoCo model escaped the repository"
        ) from exc
    if (
        not isinstance(mujoco_binding, dict)
        or mujoco_binding.get("collision_semantics")
        != "mesh_convex_hull_plus_analytic_primitives"
        or mujoco_binding.get("mjcf_path") != expected_mjcf_relative
        or mujoco_binding.get("mjcf_sha256")
        != EXPECTED_CANONICAL_MJCF_SHA256
        or not isinstance(target_colliders, list)
        or any(not isinstance(row, dict) for row in target_colliders)
        or tuple(row.get("name") for row in target_colliders)
        != EXPECTED_COLLISION_PROXY_MUJOCO_TARGET_COLLIDERS
        or not isinstance(binding_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", binding_sha) is None
    ):
        raise TableTerminationContractError(
            "collision proxy does not bind the exact live MuJoCo wrist "
            "collision inventory"
        )
    unsigned_mujoco_binding = dict(mujoco_binding)
    unsigned_mujoco_binding.pop("content_sha256", None)
    if hashlib.sha256(
        json.dumps(
            unsigned_mujoco_binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest() != binding_sha:
        raise TableTerminationContractError(
            "MuJoCo collision binding content SHA mismatch"
        )
    components = document.get("components")
    if (
        not isinstance(components, list)
        or len(components) != EXPECTED_COLLISION_PROXY_COMPONENT_COUNT
        or document.get("component_count")
        != EXPECTED_COLLISION_PROXY_COMPONENT_COUNT
        or document.get("source_component_count")
        != EXPECTED_COLLISION_PROXY_SOURCE_COMPONENT_COUNT
    ):
        raise TableTerminationContractError(
            "collision proxy must contain "
            f"{EXPECTED_COLLISION_PROXY_COMPONENT_COUNT} components"
        )
    body_index = {name: index for index, name in enumerate(TABLE_CONTACT_BODY_NAMES)}
    component_ids = []
    owner_indices = []
    centers = []
    half_axes = []
    owner_coverage = set()
    for row in components:
        if not isinstance(row, dict):
            raise TableTerminationContractError("collision proxy component is malformed")
        component_id = row.get("component_id")
        owner = row.get("owner_body_name")
        try:
            center = np.asarray(row.get("local_center_owner_m"), dtype=np.float64)
            axes = np.asarray(row.get("local_half_axes_owner_m"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TableTerminationContractError(
                "collision proxy component geometry is non-numeric"
            ) from exc
        if (
            not isinstance(component_id, str)
            or not component_id
            or owner not in body_index
            or center.shape != (3,)
            or axes.shape != (3, 3)
            or not np.isfinite(center).all()
            or not np.isfinite(axes).all()
            or np.any(np.linalg.norm(axes, axis=1) <= 0.0)
        ):
            raise TableTerminationContractError(
                "collision proxy component metadata or geometry is malformed"
            )
        component_ids.append(component_id)
        owner_indices.append(body_index[str(owner)])
        centers.append(center)
        half_axes.append(axes)
        owner_coverage.add(str(owner))
    if (
        component_ids != sorted(component_ids)
        or len(set(component_ids)) != EXPECTED_COLLISION_PROXY_COMPONENT_COUNT
        or owner_coverage != set(TABLE_CONTACT_BODY_NAMES)
    ):
        raise TableTerminationContractError(
            "collision proxy components are not canonical or body-complete"
        )
    missing_gripper = sorted(
        set(EXPECTED_LEFT_GRIPPER_SOURCE_LINKS)
        - {str(row.get("source_link_name")) for row in components}
    )
    if missing_gripper:
        raise TableTerminationContractError(
            "collision proxy omits left OmniPicker3 gripper collision links: "
            f"{missing_gripper}"
        )
    arrays = (
        np.asarray(owner_indices, dtype=np.int64),
        np.asarray(centers, dtype=np.float64),
        np.asarray(half_axes, dtype=np.float64),
    )
    for value in arrays:
        value.setflags(write=False)
    return CollisionComponents(
        component_ids=tuple(component_ids),
        owner_indices=arrays[0],
        local_centers_m=arrays[1],
        local_half_axes_m=arrays[2],
        artifact_sha256=actual_sha,
        content_sha256=content_sha,
    )


def load_collision_components() -> CollisionComponents:
    return _load_collision_components_cached(
        str(COLLISION_PROXY_ARTIFACT), EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256
    )


def _validated_table_aabbs(geometry_contract: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(geometry_contract, Mapping):
        raise TableTerminationContractError("table geometry contract must be a mapping")
    payload = geometry_contract.get("payload")
    supplied_sha = geometry_contract.get("sha256")
    if not isinstance(payload, dict):
        raise TableTerminationContractError("table geometry payload is missing")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    actual_sha = hashlib.sha256(encoded).hexdigest()
    if (
        supplied_sha != actual_sha
        or actual_sha != EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256
    ):
        raise TableTerminationContractError("exact ActionBall table geometry SHA drifted")
    obstacles = payload.get("obstacles")
    if (
        not isinstance(obstacles, list)
        or len(obstacles) != 5
        or tuple(row.get("role") for row in obstacles) != TABLE_ASSEMBLY_ROLES
    ):
        raise TableTerminationContractError("table geometry is not the exact five-part assembly")
    centers = np.asarray(
        [row.get("center_mjcf_world_m") for row in obstacles], dtype=np.float64
    )
    extents = np.asarray(
        [row.get("full_extents_m") for row in obstacles], dtype=np.float64
    )
    if (
        centers.shape != (5, 3)
        or extents.shape != (5, 3)
        or not np.isfinite(centers).all()
        or not np.isfinite(extents).all()
        or np.any(extents <= 0.0)
    ):
        raise TableTerminationContractError("table geometry centers/extents are malformed")
    half = 0.5 * extents + TABLE_GUARD_MARGIN_M
    lo = centers - half
    hi = centers + half
    lo.setflags(write=False)
    hi.setflags(write=False)
    return lo, hi


def _obb_aabb_sat_overlap(
    obb_center: np.ndarray,
    obb_half_axes: np.ndarray,
    aabb_lo: np.ndarray,
    aabb_hi: np.ndarray,
    broad_phase: np.ndarray,
) -> np.ndarray:
    """NumPy transcription of Isaac ``terminations._obb_aabb_sat_overlap``.

    人话:盒子是斜的,所以"把斜盒子胀成正盒子再比"会把空角落也算成撞上。
    这里做的是 15 轴精确判定,和 Isaac 现役终止判据逐条对齐。

    ``obb_center`` is ``[N,3]`` and ``obb_half_axes`` is ``[N,3,3]``; each of the
    last-but-one rows is one rotated half-axis vector, not merely an extent.
    Results are ``[N,O]``.  SAT is evaluated only for the positive pairs of the
    conservative world-AABB ``broad_phase``; every other pair is exactly false,
    so no world-AABB approximation reaches the verdict.  Degenerate
    cross-product axes impose no constraint, as required by SAT, and the
    comparison is inclusive so a touching face counts as overlap — both exactly
    as the Isaac kernel does.
    """

    shape = (obb_center.shape[0], aabb_lo.shape[0])
    result = np.zeros(shape, dtype=bool)
    candidate = np.argwhere(broad_phase)
    if candidate.size == 0:
        return result
    obb_index, box_index = candidate[:, 0], candidate[:, 1]
    pair_center = obb_center[obb_index]
    pair_half_axes = obb_half_axes[obb_index]
    box_center = 0.5 * (aabb_lo + aabb_hi)
    box_half = 0.5 * (aabb_hi - aabb_lo)
    delta = box_center[box_index] - pair_center

    axis_norm = np.linalg.norm(pair_half_axes, axis=-1)
    safe_norm = np.maximum(axis_norm, np.finfo(np.float64).tiny)
    obb_unit_axes = pair_half_axes / safe_norm[..., None]
    overlap = np.ones((candidate.shape[0],), dtype=bool)

    def apply_axis(axis: np.ndarray) -> None:
        # Projection radii may use an unnormalised axis; every term then carries
        # the same scale and a degenerate cross axis reduces to ``0 <= 0``.
        separation = np.abs(np.sum(delta * axis, axis=-1))
        obb_radius = np.sum(
            np.abs(np.sum(pair_half_axes * axis[:, None, :], axis=-1)), axis=-1
        )
        box_radius = np.sum(box_half[box_index] * np.abs(axis), axis=-1)
        np.logical_and(
            overlap, separation <= obb_radius + box_radius, out=overlap
        )

    world_axes = np.eye(3, dtype=np.float64)
    for world_axis in range(3):
        apply_axis(np.broadcast_to(world_axes[world_axis], pair_center.shape))
    for obb_axis in range(3):
        axis = obb_unit_axes[:, obb_axis, :]
        apply_axis(axis)
        for world_axis in range(3):
            apply_axis(
                np.cross(
                    axis, np.broadcast_to(world_axes[world_axis], axis.shape)
                )
            )
    result[obb_index, box_index] = overlap
    return result


def geometric_robot_table_hit(
    body_pos_w: Any,
    body_rotation_w: Any,
    components: CollisionComponents,
    aabb_lo: Any,
    aabb_hi: Any,
    *,
    racket_body_index: int,
) -> bool:
    """NumPy equivalent of Isaac's exact OBB-vs-AABB terminal kernel.

    Mirrors ``terminations._geometric_table_contact_hit_mask_unchecked``: the
    conservative world AABB is a PREFILTER, and the verdict is the exact 15-axis
    separating-axis test on the component / racket-blade OBBs.  Isaac stopped
    terminating on the broad phase in 5ed998f1 (2026-08-04); this port kept
    returning the retired broad-phase verdict until it was corrected, which made
    it fire on rotated boxes whose empty corners merely straddled a table AABB.
    """

    positions = np.asarray(body_pos_w, dtype=np.float64)
    rotations = np.asarray(body_rotation_w, dtype=np.float64)
    lo = np.asarray(aabb_lo, dtype=np.float64)
    hi = np.asarray(aabb_hi, dtype=np.float64)
    if (
        positions.shape != (32, 3)
        or rotations.shape != (32, 3, 3)
        or components.owner_indices.shape
        != (EXPECTED_COLLISION_PROXY_COMPONENT_COUNT,)
        or components.local_centers_m.shape
        != (EXPECTED_COLLISION_PROXY_COMPONENT_COUNT, 3)
        or components.local_half_axes_m.shape
        != (EXPECTED_COLLISION_PROXY_COMPONENT_COUNT, 3, 3)
        or lo.shape != (5, 3)
        or hi.shape != (5, 3)
        or type(racket_body_index) is not int
        or not 0 <= racket_body_index < 32
    ):
        raise TableTerminationContractError("robot/table guard runtime shapes drifted")
    if (
        not np.isfinite(positions).all()
        or not np.isfinite(rotations).all()
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(hi < lo)
    ):
        return True

    owner_pos = positions[components.owner_indices]
    owner_rotation = rotations[components.owner_indices]
    component_center = owner_pos + np.einsum(
        "cij,cj->ci", owner_rotation, components.local_centers_m
    )
    rotated_axes = np.einsum(
        "cij,ckj->cki", owner_rotation, components.local_half_axes_m
    )
    component_half = np.zeros_like(component_center)
    for local_axis in range(3):
        component_half += np.abs(rotated_axes[:, local_axis, :])
    component_half += COMPONENT_WORLD_AABB_GUARD_M
    component_lo = component_center - component_half
    component_hi = component_center + component_half
    component_broad = np.ones(
        (EXPECTED_COLLISION_PROXY_COMPONENT_COUNT, 5), dtype=bool
    )
    for axis in range(3):
        component_broad &= (
            (component_hi[:, axis, None] >= lo[None, :, axis])
            & (component_lo[:, axis, None] <= hi[None, :, axis])
        )
    component_exact = _obb_aabb_sat_overlap(
        component_center, rotated_axes, lo, hi, component_broad
    )
    if bool(np.any(component_exact)):
        return True

    racket_rotation = rotations[racket_body_index]
    blade_center = positions[racket_body_index] + (
        racket_rotation @ RACKET_BLADE_CENTER_OFFSET_WRIST_M
    )
    blade_axes = np.einsum(
        "ij,kj->ki", racket_rotation, RACKET_BLADE_LOCAL_HALF_AXES_M
    )
    blade_half = np.sum(np.abs(blade_axes), axis=0)
    blade_lo = blade_center - blade_half
    blade_hi = blade_center + blade_half
    blade_broad = np.all(
        (blade_hi[None, :] >= lo) & (blade_lo[None, :] <= hi), axis=1
    )
    blade_exact = _obb_aabb_sat_overlap(
        blade_center[None, :], blade_axes[None, :, :], lo, hi, blade_broad[None, :]
    )[0]
    return bool(np.any(blade_exact))


class ExactRobotTableGuard:
    """Run-static authority binding plus one exact per-substep pose sampler."""

    def __init__(
        self,
        mujoco: Any,
        model: Any,
        geometry_contract: Mapping[str, Any],
        *,
        mjcf_path: Path | str,
        body_name_prefix: str = "",
        registered_plant: Mapping[str, Any] | None = None,
    ):
        self.source_receipt = verify_isaac_source_authority()
        self.identity_receipt = bind_pre_registered_owner_frames(
            mujoco,
            model,
            mjcf_path,
            body_name_prefix=body_name_prefix,
            registered_plant=registered_plant,
        )
        self.components = load_collision_components()
        self.aabb_lo, self.aabb_hi = _validated_table_aabbs(geometry_contract)
        body_ids = _owner_frame_contract(
            mujoco, model, body_name_prefix=body_name_prefix
        )["body_ids"]
        self.body_ids = np.asarray(body_ids, dtype=np.int64)
        self.body_ids.setflags(write=False)
        self.racket_body_index = TABLE_CONTACT_BODY_NAMES.index(RACKET_BODY_NAME)

    def sample(self, data: Any) -> bool:
        positions = np.asarray(data.xpos[self.body_ids], dtype=np.float64)
        rotations = np.asarray(data.xmat[self.body_ids], dtype=np.float64).reshape(
            32, 3, 3
        )
        return geometric_robot_table_hit(
            positions,
            rotations,
            self.components,
            self.aabb_lo,
            self.aabb_hi,
            racket_body_index=self.racket_body_index,
        )


__all__ = [
    "CANONICAL_MJCF",
    "CANONICAL_MUJOCO_IDENTITY_PY",
    "COLLISION_PROXY_ARTIFACT",
    "CollisionComponents",
    "EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256",
    "EXPECTED_CANONICAL_MJCF_SHA256",
    "EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256",
    "EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256",
    "EXPECTED_ISAAC_ACTION_LATCH_SEMANTIC_AST_SHA256",
    "EXPECTED_ISAAC_TERMINATION_CALLABLES_SEMANTIC_AST_SHA256",
    "EXPECTED_ISAAC_TERMINATION_CONFIG_SEMANTIC_AST_SHA256",
    "EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256",
    "EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256",
    "ExactRobotTableGuard",
    "ISAAC_ACTION_LATCH",
    "ISAAC_TABLE_TERM_FACTORY",
    "ISAAC_TERMINATION_CALLABLE_SELECTORS",
    "ISAAC_TERMINATION_CALLABLES",
    "ISAAC_TERMINATION_CONFIG",
    "MUJOCO_IDENTITY_MANIFEST",
    "RACKET_BLADE_CENTER_OFFSET_WRIST_M",
    "RACKET_BLADE_LOCAL_HALF_AXES_M",
    "RACKET_BODY_NAME",
    "TABLE_ASSEMBLY_ROLES",
    "TABLE_CONTACT_BODY_NAMES",
    "TABLE_GUARD_MARGIN_M",
    "TableTerminationContractError",
    "bind_pre_registered_owner_frames",
    "consume_verified_owner_frame_contract",
    "geometric_robot_table_hit",
    "live_isaac_constant_blockers",
    "load_collision_components",
    "mirrored_isaac_constant_entries",
    "verify_isaac_source_authority",
]
