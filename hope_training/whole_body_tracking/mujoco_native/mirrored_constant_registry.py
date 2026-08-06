"""Every module-level constant in this lane must say how it is guarded.

人话:这个文件是**通用护栏**,不是又补一处算一处。

这条 MuJoCo 复刻车道已经被同一个形状咬了四次:

1. ``5ed998f1`` —— Isaac 把桌面终局从广相 AABB 改成精确 SAT,**同一个提交**把复刻侧的
   AST 指纹扩到覆盖新函数并重新盖章,语义没移植。两天没人发现。
2. ``5c4ced66`` —— 改了 Isaac 叶子却没重钉镜像 SHA;审计发现语义没漂,但**测试根本没
   能力说这句话**,因为它比的是测试文件里第三份手抄字面量。
3. ``ee_body_pos`` —— 子类把参考包络收窄成只有双脚,复刻抄的是父类那份四个身体的名单;
   指纹的选择器没点这条项的名,**一个 bit 都没动过**。
4. ``TerminationsCfg`` 的类体顺序 —— 终止原因的先后是一份跨两个文件、三个类的手抄件,
   而 ``base_config`` 选择器只点了 ``time_out`` 一个名字。

共同点永远是同一句:**指纹只证明字节没动,不证明抄对了**;而且没人有一张"这个模块里
到底有多少份手抄件"的清单,所以每次都是事后一处一处捞。

这里换个方向:**枚举**。测试把每个复刻模块的模块级常量全数出来,要求每一个都在下面
被显式分类。新加一个常量而不分类 —— 测试当场红。分类里有一档是
``LIVE_VALUE_COMPARED``,测试会真的把常量的值跟"活值比对入口实际拿去比的那个值"对上,
所以"我说它被活值罩着"这句话本身也是被机器检的,不是自称。

**这张表不能默认放行**:没有兜底通配,没有"其余的都算本地常量"。豁免必须一条一条写
下来,并且写明是哪一档理由;真欠的债只能进 :data:`OPEN_MIRROR_DEBT`,而那一档强制要
求填"在哪、怎么修、为什么这轮没修"。

护栏诚实的边界(别过度宣称):它保证不了"某人把一份新的手抄 Isaac 常量硬说成
``NOT_MIRRORED``"。它保证的是**这件事必须有人动手写一行、署上一个理由档位**,而不是
像前四次那样悄无声息地混进来。
"""

from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
from typing import Any, Sequence


MODULE_DIR = Path(__file__).resolve().parent
PACKAGE = "mujoco_native"


class MirroredConstantRegistryError(RuntimeError):
    """The registry no longer describes the modules it claims to describe."""


# ---------------------------------------------------------------------------
# The reason vocabulary.  Closed on purpose: a free-text reason is a reason
# nobody can machine-check.
# ---------------------------------------------------------------------------

#: The value is compared, as a VALUE, against what the live Isaac source ships
#: right now.  ``detail`` is ``"<provider>:<key>"``; the meta test resolves the
#: provider and asserts the module constant equals the value that provider hands
#: to the live comparison.  This is the only class of entry that re-pinning a
#: digest cannot satisfy.
LIVE_VALUE_COMPARED = "live_value_compared"

#: There is no hand copy at all: the constant is *read* out of the live source at
#: import time.  The meta test asserts the assignment is not a literal, so
#: quietly replacing the live read with the number it happened to return fails.
LIVE_VALUE_DERIVED = "live_value_derived"

#: Computed from other constants in the lane (``math.cos(...)``, ``sum(...)``,
#: a slice of another tuple).  Same literal check: it must stay computed.
DERIVED_IN_MODULE = "derived_in_module"

#: A path to a live source file.  The meta test asserts the file exists, so an
#: upstream rename fails on a host test instead of on a pod boot.
LIVE_SOURCE_PATH = "live_source_path"

#: A pinned SHA-256 of a file this module also names.  ``detail`` is the name of
#: the ``Path`` constant; the meta test re-reads that file and recomputes the
#: digest.  "Re-pin" and "port" are the same act for these -- there is no
#: pinned-but-not-ported middle state -- so the only real risk is editing the
#: file and forgetting, which this catches on the host.
PINNED_FILE_DIGEST = "pinned_file_digest"

#: A pinned digest whose subject is NOT a file this module names (a derived
#: payload, or a file only a launcher resolves).  ``detail`` must name who
#: recomputes it.  Weaker than :data:`PINNED_FILE_DIGEST` and deliberately a
#: separate word, so the difference is visible in the table.
PINNED_EXTERNAL_DIGEST = "pinned_external_digest"

#: The constant is not itself compared, but everything it contributes to a
#: live-compared object is.  ``detail`` is ``"<provider>:<key>"`` naming that
#: object; the meta test asserts the constant is contained in it, in order.  A
#: string must appear in it; a sequence must be a subsequence of it.  Weaker
#: than :data:`LIVE_VALUE_COMPARED` and deliberately a separate word.
FLOWS_INTO_LIVE_COMPARISON = "flows_into_live_comparison"

#: This lane's own vocabulary -- receipt ``kind`` strings, schema versions,
#: blocker lists, status enums, its own diagnostic knobs.  There is no Isaac
#: twin to drift away from.
NOT_MIRRORED = "not_mirrored"

#: A real hand copy of an Isaac value that is NOT live-compared yet.  Requires an
#: :data:`OPEN_MIRROR_DEBT` entry saying where the live twin is, how to close it,
#: and why this round did not.  This is the only honest place to park debt: it is
#: in code, the test enforces the note, and it cannot rot into silence.
MIRRORED_TODO = "mirrored_isaac_value_not_yet_live_compared"

REASONS = (
    LIVE_VALUE_COMPARED,
    LIVE_VALUE_DERIVED,
    DERIVED_IN_MODULE,
    LIVE_SOURCE_PATH,
    PINNED_FILE_DIGEST,
    PINNED_EXTERNAL_DIGEST,
    FLOWS_INTO_LIVE_COMPARISON,
    NOT_MIRRORED,
    MIRRORED_TODO,
)

#: Reasons whose ``detail`` field is mandatory.
REASONS_REQUIRING_DETAIL = (
    LIVE_VALUE_COMPARED,
    PINNED_FILE_DIGEST,
    PINNED_EXTERNAL_DIGEST,
    FLOWS_INTO_LIVE_COMPARISON,
)


# ---------------------------------------------------------------------------
# Live-value providers: "what does the live comparison actually compare?"
# ---------------------------------------------------------------------------


def _entry_values(entries: Sequence[Sequence[Any]]) -> dict:
    """``(key, ..., mirrored)`` rows -> ``{key: mirrored}``."""

    values = {}
    for entry in entries:
        key = entry[0]
        if key in values:
            raise MirroredConstantRegistryError(f"duplicate live entry key {key!r}")
        values[key] = entry[-1]
    return values


def _phase_termination_values() -> dict:
    module = importlib.import_module(f"{PACKAGE}.vec_env")
    return _entry_values(module.mirrored_isaac_termination_entries())


def _table_guard_values() -> dict:
    module = importlib.import_module(f"{PACKAGE}.table_termination")
    return _entry_values(module.mirrored_isaac_constant_entries())


def _native_source_digest_values() -> dict:
    module = importlib.import_module(f"{PACKAGE}.n1_reward_event_kernel")
    return _entry_values(module.mirrored_source_digest_entries())


def _action_ball_211_abi_values() -> dict:
    """The widths the live-parity check reads off the profiles it compares.

    Both profiles must agree, because ``live_source_parity_blockers`` compares
    ``profile.actor.width`` per profile: if the two disagreed, one of them would
    no longer be the constant this module publishes.
    """

    module = importlib.import_module(f"{PACKAGE}.action_ball_211_abi")
    values = {}
    for lane in ("actor", "critic"):
        widths = {
            getattr(profile, lane).width for profile in module.PROFILES.values()
        }
        if len(widths) != 1:
            raise MirroredConstantRegistryError(
                f"the two 211 profiles disagree on the {lane} width: {widths!r}"
            )
        values[f"{lane}_width"] = widths.pop()
    for label, profile in module.PROFILES.items():
        key = label.lower()
        values[f"{key}_actor_row_names"] = tuple(
            name for name, _width in profile.actor.layout
        )
        values[f"{key}_wait_mask_tail"] = tuple(
            module._wait_mask_tail_names(profile.actor.layout) or ()
        )
    return values


def _reference_envelope_values() -> dict:
    module = importlib.import_module(f"{PACKAGE}.vec_env")
    envelope = importlib.import_module(f"{PACKAGE}.isaac_reference_envelope")
    live = envelope.live_reference_envelope(
        module.PHASE_TERMINATIONS_MIRRORED_ISAAC_CLASS
    )
    return {
        "mirrored_class": live["class_name"],
        "ee_body_names": live["body_names"],
        "ee_body_pos_z_threshold_m": live["threshold_m"],
    }


def _reason_order_values() -> dict:
    module = importlib.import_module(f"{PACKAGE}.vec_env")
    envelope = importlib.import_module(f"{PACKAGE}.isaac_reference_envelope")
    live_order = envelope.live_termination_reason_order(
        module.PHASE_TERMINATIONS_MIRRORED_ISAAC_CLASS,
        config_path=module.TERMINATION_SOURCE_CONFIG,
        base_config_path=module.TERMINATION_SOURCE_BASE_CONFIG,
    )
    timeouts = envelope.live_timeout_term_names(
        module.PHASE_TERMINATIONS_MIRRORED_ISAAC_CLASS,
        config_path=module.TERMINATION_SOURCE_CONFIG,
        base_config_path=module.TERMINATION_SOURCE_BASE_CONFIG,
    )
    hard = tuple(term for term in live_order if term not in timeouts)
    return {"active_order": live_order, "hard_order": hard}


LIVE_VALUE_PROVIDERS = {
    "phase_termination": _phase_termination_values,
    "table_guard": _table_guard_values,
    "native_source_digest": _native_source_digest_values,
    "action_ball_211_abi": _action_ball_211_abi_values,
    "reference_envelope": _reference_envelope_values,
    "reason_order": _reason_order_values,
}


# ---------------------------------------------------------------------------
# Open debt.  MIRRORED_TODO entries must appear here.
# ---------------------------------------------------------------------------

#: ``"<module>.<CONSTANT>" -> (live twin, how to close it, why not this round)``.
#: 人话:这是这轮**明确没修**的那几处,连"在哪、怎么改、为什么先不改"一起写死在代码
#: 里。测试强制每条 ``MIRRORED_TODO`` 都在这儿有一行,所以它不会烂成沉默。
OPEN_MIRROR_DEBT = {
    "action_ball_c211_env.C211_UPRIGHT_STD": (
        "hope_env_cfg.py :: HOPERewardsCfg.upright_exp params={'std': math.sqrt(0.2)}",
        "isaac_live_constants 的白名单求值器目前折不出 math.sqrt(0.2) 这种 Call,"
        "要先给它加一小撮纯函数(math.sqrt/cos/sin/pi)才能读活值;加完后按 "
        "('class_term_param', <RewardsCfg>, 'upright_exp', 'std') 注册。",
        "放宽求值器白名单会扩大'猜'的面,这轮的定调是收紧不放宽,不在同一批里做。",
    ),
    "action_ball_c211_env.C211_ACTION_RATE_CLAMP": (
        "hope_env_cfg.py :: HOPERewardsCfg.action_rate_clamped "
        "params={'value_clamp': 9.0}",
        "值是纯字面量,求值器现在就读得出;缺的是给 c211_env 建一张和 "
        "table_termination 同款的 mirrored_isaac_constant_entries 表并接到它的收据上。",
        "c211_env 的 Isaac 侧奖励镜像不止这一条(见下面两条),要做就一次做完整张表,"
        "这轮的预算给了终止顺序那条链。",
    ),
    "action_ball_c211_env.TRACKED_BODY_NAMES": (
        "Isaac 的 motion 跟踪 body 名单(命令侧,不在 terminations 里)",
        "先定位真源到底是 cfg 里的哪一个符号(命令项参数还是机器人叶子的列表),"
        "再按 ('class_term_param', ...) 或 ('assignment', ...) 注册。",
        "真源尚未定位到唯一符号;没定位清楚就注册等于给门喂一个猜的答案。",
    ),
    "action_ball_c211_env.C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES": (
        "hope_env_cfg.py :: HOPERewardsCfg 里这 14 条 RewTerm 的名字与顺序",
        "复用 isaac_reference_envelope 那套 live_declared_terms / 顺序推导,"
        "把它指到 RewardsCfg 的继承链上 —— 和终止项这条链是同一个机制。",
        "奖励项的类链比终止项长且有 weight=0.0 的'默认跳过'语义,"
        "要先想清楚'声明了但权重为零'算不算实现,这轮不带着未定义的语义上门。",
    ),
    "action_ball_c211_env.C211_RACKET_LONG_AXIS_LOCAL": (
        "Isaac 侧球拍长轴的局部方向(racket_contact_geometry / hope_rewards)",
        "定位到符号后按 ('assignment', ...) 注册;值是 math.sqrt(0.5) 组成的元组,"
        "和 C211_UPRIGHT_STD 卡在同一个求值器限制上。",
        "同上:求值器白名单这轮不动。",
    ),
}


# ---------------------------------------------------------------------------
# The table itself.  One line per module-level constant, no wildcards.
# ---------------------------------------------------------------------------

_L = LIVE_VALUE_COMPARED
_D = LIVE_VALUE_DERIVED
_M = DERIVED_IN_MODULE
_P = LIVE_SOURCE_PATH
_F = PINNED_FILE_DIGEST
_X = PINNED_EXTERNAL_DIGEST
_N = NOT_MIRRORED
_T = MIRRORED_TODO
_I = FLOWS_INTO_LIVE_COMPARISON

_ABI_LINEAGE = "launch_mujoco_action_ball_211_diagnostic._source_lineage 逐文件重算"

CLASSIFICATION: dict = {
    "action_ball_211_abi.py": {
        "ACTOR_WIDTH": (_L, "action_ball_211_abi:actor_width"),
        "CRITIC_WIDTH": (_L, "action_ball_211_abi:critic_width"),
        "A211_SOURCE_SHA256": (_X, _ABI_LINEAGE),
        "C211_SOURCE_SHA256": (_X, _ABI_LINEAGE),
        "A211_TASK_LEAF_SHA256": (_X, _ABI_LINEAGE),
        "C211_TASK_LEAF_SHA256": (_X, _ABI_LINEAGE),
        "WAIT_MASK_CONTRACT_IDENTITY": (_N, ""),
        "PLANT_OBSERVATION_AUTHORITY_KIND": (_N, ""),
        "MEASURED_MIMIC_AUTHORITY_KIND": (_N, ""),
        "TASK_QUESTION_AUTHORITY_KIND": (_N, ""),
        "_A_TASK_NAMES": (_I, "action_ball_211_abi:a211_wait_mask_tail"),
        "_C_TASK_NAMES": (_I, "action_ball_211_abi:c211_wait_mask_tail"),
        "A211_PROFILE": (_M, ""),
        "C211_PROFILE": (_M, ""),
        "PROFILES": (_M, ""),
        "WAIT_MASK_TAIL_ANCHOR_FIELD": (
            _I,
            "action_ball_211_abi:a211_actor_row_names",
        ),
        "MIRRORED_IDENTITY_SYMBOLS": (_N, ""),
        "_ABSENT": (_N, ""),
    },
    "action_ball_a211_env.py": {
        "A211_ENV_KIND": (_N, ""),
        "A211_TASK_PROVIDER_KIND": (_M, ""),
        "A211_TARGET_RECIPE": (_N, ""),
        "A211_REWARD_SCOPE": (_N, ""),
        "A211_REWARD_CONTRACT_IDENTITY": (_N, ""),
        "A211_TASK_REWARD_CONTRACT_IDENTITY": (_N, ""),
        "A211_POLICY_DT_S": (_N, ""),
        "A211_POSITION_HALF_WINDOW_S": (_N, ""),
        "A211_WIDE_HALF_WINDOW_S": (_N, ""),
        "A211_CAPTURE_POST_DT_WEIGHT": (_N, ""),
        "A211_PASS_NET_POST_DT_WEIGHT": (_N, ""),
        "A211_LANDING_DENSE_POST_DT_WEIGHT": (_N, ""),
        "A211_LEGAL_LANDING_POST_DT_WEIGHT": (_N, ""),
        "A211_NET_MARGIN_M": (_N, ""),
        "A211_NET_SIGMA_M": (_N, ""),
        "A211_LANDING_SIGMA_M": (_N, ""),
        "A211_LANDING_LEGAL_BASE_FRAC": (_N, ""),
        "A211_BASE_POSITION_WEIGHT": (_N, ""),
        "A211_BASE_POSITION_STD_M": (_N, ""),
        "A211_RACKET_PROGRESS_WEIGHT": (_N, ""),
        "A211_RACKET_PROGRESS_POTENTIAL_CAP_M": (_N, ""),
        "COUNTER_RALLY_PY": (_P, ""),
        "COUNTER_RALLY_TORCH_PY": (_P, ""),
        "A211_TARGET_CHANNELS": (_M, ""),
        "A211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS": (_N, ""),
        "A211_FORMAL_BLOCKERS": (_N, ""),
    },
    "action_ball_c211_env.py": {
        "C211_ENV_KIND": (_N, ""),
        "C211_PLANT_PROVIDER_KIND": (_M, ""),
        "C211_MIMIC_PROVIDER_KIND": (_M, ""),
        "C211_TASK_PROVIDER_KIND": (_M, ""),
        "C211_TARGET_RECIPE": (_N, ""),
        "C211_REWARD_SCOPE": (_N, ""),
        "C211_REWARD_CONTRACT_IDENTITY": (_N, ""),
        "C211_TASK_REWARD_CONTRACT_IDENTITY": (_N, ""),
        "C211_POLICY_DT_S": (_N, ""),
        "C211_STRIKE_STD_M": (_N, ""),
        "C211_STRIKE_POST_DT_WEIGHT": (_N, ""),
        "C211_LANDING_SIGMA_M": (_N, ""),
        "C211_LANDING_POST_DT_WEIGHT": (_N, ""),
        "C211_LANDING_LEGAL_BASE_FRAC": (_N, ""),
        "C211_LANDING_OFF_TABLE_FRAC": (_N, ""),
        "C211_ROLLOUT_H_S": (_N, ""),
        "C211_ROLLOUT_STEPS": (_N, ""),
        "C211_UPRIGHT_STD": (_T, ""),
        "C211_ACTION_RATE_CLAMP": (_T, ""),
        "C211_RACKET_LONG_AXIS_LOCAL": (_T, ""),
        "VIRTUAL_BALL_PY": (_P, ""),
        "VENUE_PHYSICS_YAML": (_P, ""),
        "C211_TRAINABILITY_PY": (_P, ""),
        "C225_REWARD_PY": (_P, ""),
        "HOPE_REWARDS_PY": (_P, ""),
        "HOPE_ENV_CFG_PY": (_P, ""),
        "TRAIN_PY": (_P, ""),
        "VENDOR_V2_TASK_YAML": (_P, ""),
        "C211_TASK_YAML": (_P, ""),
        "TRACKED_BODY_NAMES": (_T, ""),
        "ANCHOR_BODY_NAME": (_N, ""),
        "ROOT_BODY_NAME": (_N, ""),
        "RIGHT_WRIST_BODY_NAME": (_N, ""),
        "MIMIC_BODY_NAMES": (_M, ""),
        "C211_IMPLEMENTED_ISAAC_PRIOR_TERM_NAMES": (_T, ""),
        "C211_UNAVAILABLE_ISAAC_REWARD_TERMS": (_N, ""),
        "C211_CROSS_ENGINE_REWARD_SEMANTIC_GAPS": (_N, ""),
        "FORMAL_BLOCKERS": (_N, ""),
        "SAFE_READY_AUTHORITY_STATUS": (_N, ""),
    },
    "action_specific_hold.py": {
        "KIND": (_N, ""),
        "SCHEMA_VERSION": (_N, ""),
        "SCRIPTS_DIR": (_M, ""),
    },
    "checkpoint.py": {
        "CHECKPOINT_KIND": (_N, ""),
        "CHECKPOINT_SCHEMA_VERSION": (_N, ""),
    },
    "exact_frame0_action_specific_hold.py": {
        "KIND": (_M, ""),
        "SCHEMA_VERSION": (_M, ""),
        "THRESHOLD_VALIDATOR_PATH": (_P, ""),
        "PHYSICAL_RESET_SEMANTICS": (_N, ""),
        "CONTROLLER_BIRTH_SEMANTICS": (_N, ""),
        "HISTORY_FILL_SEMANTICS": (_N, ""),
        "NON_CLAIMS": (_N, ""),
        "_THRESHOLD_MODULE_NAME": (_N, ""),
    },
    "fixed_center_recipe.py": {
        "RECIPE_KIND": (_N, ""),
        "RECIPE_SOURCE_SHA256": (_M, ""),
        "READINESS_KIND": (_M, ""),
        "TASK_SLICE": (_N, ""),
        "FORMAL_BLOCKERS": (_N, ""),
        "TASK_WAIT_SOURCE": (_P, ""),
    },
    "isaac_live_constants.py": {
        "SELECTOR_KINDS": (_N, ""),
    },
    # 这张表自己也要被这道门数一遍 —— 否则"给护栏加一个常量"就成了唯一的免检通道。
    "mirrored_constant_registry.py": {
        "MODULE_DIR": (_M, ""),
        "PACKAGE": (_N, ""),
        "LIVE_VALUE_COMPARED": (_N, ""),
        "LIVE_VALUE_DERIVED": (_N, ""),
        "DERIVED_IN_MODULE": (_N, ""),
        "LIVE_SOURCE_PATH": (_N, ""),
        "PINNED_FILE_DIGEST": (_N, ""),
        "PINNED_EXTERNAL_DIGEST": (_N, ""),
        "FLOWS_INTO_LIVE_COMPARISON": (_N, ""),
        "NOT_MIRRORED": (_N, ""),
        "MIRRORED_TODO": (_N, ""),
        "REASONS": (_M, ""),
        "REASONS_REQUIRING_DETAIL": (_M, ""),
        "LIVE_VALUE_PROVIDERS": (_M, ""),
        "OPEN_MIRROR_DEBT": (_N, ""),
        "CLASSIFICATION": (_M, ""),
        "MODULES_WITHOUT_CONSTANTS": (_N, ""),
        "_ABI_LINEAGE": (_N, ""),
        "_PARTITION_KEYS": (_N, ""),
        "_L": (_M, ""),
        "_D": (_M, ""),
        "_M": (_M, ""),
        "_P": (_M, ""),
        "_F": (_M, ""),
        "_X": (_M, ""),
        "_N": (_M, ""),
        "_T": (_M, ""),
        "_I": (_M, ""),
    },
    "isaac_reference_envelope.py": {
        "REPO_ROOT": (_M, ""),
        "ISAAC_TERMINATION_CONFIG": (_P, ""),
        "ISAAC_BODY_NAME_LISTS": (_P, ""),
        "ISAAC_BASE_TERMINATION_CONFIG": (_P, ""),
        "BODY_NAME_MODULE": (_N, ""),
        "ACTION_BALL_TERMINATIONS_CLASS": (_N, ""),
        "DEPLOY_PARITY_TERMINATIONS_CLASS": (_N, ""),
        "BASE_TERMINATIONS_CLASS": (_N, ""),
        "EXTERNAL_TERMINATION_BASES": (_M, ""),
        "REFERENCE_ENVELOPE_TERM": (_N, ""),
        "DECLARED_TERMS": (_N, ""),
        "BODY_NAME_VOCABULARY_SYMBOLS": (_N, ""),
    },
    "n1_ball_core.py": {
        "QUESTION_KIND": (_N, ""),
        "RECEIPT_KIND": (_N, ""),
        "TRACE_KIND": (_N, ""),
        "PHASE_FIDELITY_REFERENCE_TAPE_KIND": (_N, ""),
        "FIXED_QUESTION_TAPE_PY": (_P, ""),
    },
    "n1_reward_event_kernel.py": {
        "N1_REWARD_EVENT_KERNEL_KIND": (_N, ""),
        "NATIVE_PHYSICAL_EVENT_FACTS_KIND": (_N, ""),
        "NATIVE_PHYSICAL_EVENT_FACTS_CONTRACT_KIND": (_N, ""),
        "NATIVE_CONTACT_INVALID_REASONS": (_N, ""),
        "EXPECTED_OBSERVED_OUTCOME_RESOLVER_SOURCE_SHA256": (
            _L,
            "native_source_digest:observed_outcome_resolver_source_sha256",
        ),
        "EXPECTED_N1_BALL_CORE_SOURCE_SHA256": (
            _L,
            "native_source_digest:n1_ball_core_source_sha256",
        ),
        "EXPECTED_PHYSICAL_BALL_SCENE_SOURCE_SHA256": (
            _L,
            "native_source_digest:physical_ball_scene_source_sha256",
        ),
        "EXPECTED_TABLE_SCENE_SOURCE_SHA256": (
            _L,
            "native_source_digest:table_scene_source_sha256",
        ),
        "MODULE_DIR": (_M, ""),
        "REPO_ROOT": (_M, ""),
    },
    "n1_scalar_reward.py": {
        "_ELIGIBILITY_BOOL_FIELDS": (_N, ""),
    },
    "observed_outcome_resolver.py": {
        "REPO_ROOT": (_M, ""),
        "RESOLVER_BINDING_KIND": (_N, ""),
        "QUESTION_BINDING_KIND": (_N, ""),
        "SNAPSHOT_KIND": (_N, ""),
        "SUMMARY_KIND": (_N, ""),
        "STATUS_UNARMED": (_N, ""),
        "STATUS_TRACKING": (_N, ""),
        "STATUS_NET_COLLISION": (_N, ""),
        "STATUS_FIRST_TABLE_LANDING": (_N, ""),
        "STATUS_FLOOR_CONTACT": (_N, ""),
        "STATUS_SAME_SUBSTEP_AMBIGUOUS": (_N, ""),
        "STATUS_OUTGOING_OVERLAP_AMBIGUOUS": (_N, ""),
        "STATUSES": (_M, ""),
        "RESOLVED_STATUSES": (_M, ""),
        "AMBIGUOUS_STATUSES": (_M, ""),
        "CONTACT_LABELS": (_N, ""),
        "TIME_DELTA_ABS_TOLERANCE_S": (_N, ""),
        "SEMANTICS": (_N, ""),
    },
    "physical_ball_scene.py": {
        "REPO_ROOT": (_M, ""),
        "DEFAULT_MJCF": (_P, ""),
        "TABLE_SCENE_PY": (_P, ""),
        "BALL_BODY_NAME": (_N, ""),
        "BALL_JOINT_NAME": (_N, ""),
        "BALL_GEOM_NAME": (_N, ""),
        "RACKET_GEOM_NAME": (_N, ""),
        "TABLE_GEOM_NAME": (_N, ""),
        "ROBOT_KEEPOUT_GEOM_NAME": (_N, ""),
        "NET_GEOM_NAMES": (_N, ""),
        "TABLE_ASSEMBLY_GEOM_NAMES": (_M, ""),
        "FLOOR_GEOM_NAME": (_N, ""),
    },
    "selected_rubber_classifier.py": {
        "REPO_ROOT": (_M, ""),
        "DEFAULT_MJCF": (_P, ""),
        "OFFICIAL_URDF": (_P, ""),
        "OFFICIAL_URDF_MESH_DIR": (_M, ""),
        "IDENTITY_MANIFEST": (_P, ""),
        "GEOMETRY_SOURCE_PY": (_P, ""),
        "CLASSIFIER_BINDING_KIND": (_N, ""),
        "ACTION_LINEAGE_KIND": (_N, ""),
        "CLASSIFICATION_KIND": (_N, ""),
        "RACKET_SITE_NAME": (_N, ""),
        "GENERIC_BLADE_GEOM_NAME": (_N, ""),
        "RAW_A_AXIS_LOCAL": (_N, ""),
        "STATUS_SELECTED": (_N, ""),
        "STATUS_OPPOSITE": (_N, ""),
        "STATUS_EDGE_RIM_AMBIGUOUS": (_N, ""),
        "STATUS_BETWEEN_PLANES_AMBIGUOUS": (_N, ""),
        "CLASSIFICATION_STATUSES": (_M, ""),
        "AMBIGUITY_EDGE_RIM": (_N, ""),
        "AMBIGUITY_BETWEEN_PLANES": (_N, ""),
    },
    "single_env.py": {
        "ACTION_DIM": (_N, ""),
        "JOINT_BOUNDS_TOLERANCE_RAD": (_N, ""),
        "FIXED_TAPE_TICKS": (_N, ""),
        "TAPE_KIND": (_N, ""),
        "RECEIPT_KIND": (_N, ""),
        "TRACE_KIND": (_N, ""),
        "ACTION_SPECIFIC_HOLD_KIND": (_N, ""),
        "ACTION_SPECIFIC_HOLD_SCHEMA_VERSION": (_N, ""),
        "EXACT_FRAME0_ACTION_SPECIFIC_HOLD_KIND": (_N, ""),
        "EXACT_FRAME0_ACTION_SPECIFIC_HOLD_SCHEMA_VERSION": (_N, ""),
        "REPO_ROOT": (_M, ""),
        "DEFAULT_MJCF": (_P, ""),
        "TABLE_SCENE_PY": (_P, ""),
        "JOINT_ORDER_CONTRACT": (_P, ""),
        "JOINT_ORDER_CONTRACT_ID": (_N, ""),
    },
    "table_termination.py": {
        "REPO_ROOT": (_M, ""),
        "ISAAC_TERMINATION_CONFIG": (_P, ""),
        "EXPECTED_ISAAC_TERMINATION_CONFIG_SEMANTIC_AST_SHA256": (
            _X,
            "verify_isaac_source_authority 用选择器重算语义 AST 摘要;"
            "它保护的那几个值另有 live_isaac_constant_blockers 逐个比值",
        ),
        "ISAAC_TERMINATION_CALLABLES": (_P, ""),
        "EXPECTED_ISAAC_TERMINATION_CALLABLES_SEMANTIC_AST_SHA256": (
            _X,
            "verify_isaac_source_authority;SAT 判据本体是函数体,只能靠语义摘要",
        ),
        "ISAAC_TERMINATION_CALLABLE_SELECTORS": (_N, ""),
        "ISAAC_ACTION_LATCH": (_P, ""),
        "EXPECTED_ISAAC_ACTION_LATCH_SEMANTIC_AST_SHA256": (
            _X,
            "verify_isaac_source_authority",
        ),
        "CANONICAL_MJCF": (_P, ""),
        "EXPECTED_CANONICAL_MJCF_SHA256": (_F, "CANONICAL_MJCF"),
        "MUJOCO_IDENTITY_MANIFEST": (_P, ""),
        "EXPECTED_MUJOCO_IDENTITY_MANIFEST_SHA256": (_F, "MUJOCO_IDENTITY_MANIFEST"),
        "CANONICAL_MUJOCO_IDENTITY_PY": (_P, ""),
        "EXPECTED_CANONICAL_MUJOCO_IDENTITY_PY_SHA256": (
            _F,
            "CANONICAL_MUJOCO_IDENTITY_PY",
        ),
        "EXPECTED_PORTABLE_MUJOCO_IDENTITY_SHA256": (
            _X,
            "_verified_registered_owner_frames 让 canonical_mujoco_identity 重算"
            "可移植身份摘要,主体是派生载荷不是文件",
        ),
        "COLLISION_PROXY_ARTIFACT": (_P, ""),
        "EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256": (_F, "COLLISION_PROXY_ARTIFACT"),
        "EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256": (
            _X,
            "_validated_table_aabbs 对运行期桌面几何契约重算,主体是派生载荷",
        ),
        "TABLE_GUARD_MARGIN_M": (_L, "table_guard:table_guard_margin_m"),
        "COMPONENT_WORLD_AABB_GUARD_M": (
            _N,
            "",
        ),
        "RACKET_BODY_NAME": (_L, "table_guard:racket_body_name"),
        "RACKET_BLADE_CENTER_OFFSET_WRIST_M": (
            _L,
            "table_guard:racket_blade_center_offset_wrist_m",
        ),
        # 人话:副本存的是 3x3 对角矩阵,拿去比的是它的三条半轴;中间那个 helper
        # 顺带断言"它还是个对角盒",所以这条 via 不是绕过检查,是把检查串起来。
        "RACKET_BLADE_LOCAL_HALF_AXES_M": (
            _L,
            "table_guard:racket_blade_half_extents_m via _mirrored_blade_half_extents_m",
        ),
        "TABLE_ASSEMBLY_ROLES": (_L, "table_guard:table_assembly_roles"),
        "TABLE_CONTACT_BODY_NAMES": (_N, ""),
        "ISAAC_TABLE_TERM_FACTORY": (_N, ""),
    },
    "trainer.py": {
        "DIAGNOSTIC_TRAINER_RECEIPT_KIND": (_N, ""),
        "DIAGNOSTIC_UPDATE_RECEIPT_KIND": (_N, ""),
        "NORMALIZER_BINDING_KIND": (_N, ""),
        "TERMINAL_ROW_TELEMETRY_CONTRACT_KIND": (_N, ""),
        "TERMINAL_ROW_TELEMETRY_RECEIPT_KIND": (_N, ""),
        "NORMALIZER_UPDATE_RULE": (_N, ""),
        "NORMALIZER_WAIT_OUTPUT_RULE": (_N, ""),
        "TIMEOUT_BOOTSTRAP_RULE": (_N, ""),
        "_IDENTITY_FIELDS": (_N, ""),
        "ACTOR_INIT_MODE_ZERO_WEIGHT_READY_BIAS": (_N, ""),
        "ACTOR_INIT_MODE_DEFAULT": (_N, ""),
        "ACTOR_INIT_MODES": (_M, ""),
        "FOUR_SIGMA_GATE_SKIPPED_REASON": (_N, ""),
    },
    "vec_env.py": {
        "OBSERVATION_LAYOUT": (_N, ""),
        "OBSERVATION_WIDTH": (_M, ""),
        "REWARD_BLOCKERS": (_N, ""),
        "C_LITE_REWARD_KIND": (_N, ""),
        "C_LITE_STRIKE_WINDOW_HALF_WIDTH_S": (_N, ""),
        "C_LITE_PORTABLE_PARENT_KIND": (_N, ""),
        "C_LITE_FORMAL_BLOCKERS": (_M, ""),
        "DIAGNOSTIC_TRAINER_RECEIPT_KIND": (_N, ""),
        "FORMAL_TERMINATION_BLOCKERS": (_N, ""),
        "EXACT_BASE_TERMINATION_REASON_ORDER": (
            _L,
            "reason_order:base_and_joint_bucket",
        ),
        "EXACT_PHASE_FIDELITY_REASON_ORDER": (
            _L,
            "reason_order:phase_fidelity_bucket",
        ),
        "EXACT_TABLE_GUARD_REASON_ORDER": (_L, "reason_order:table_guard_bucket"),
        "EXACT_HARD_TERMINATION_REASON_ORDER": (_L, "reason_order:hard_order"),
        "EXACT_ACTIVE_TERMINATION_REASON_ORDER": (_L, "reason_order:active_order"),
        "BASE_FELL_TILT_LIMIT_ANGLE_RAD": (
            _L,
            "phase_termination:base_fell_tilt_limit_angle_rad",
        ),
        "BASE_FELL_TILT_MIN_UP_WORLD_Z": (_M, ""),
        "BASE_TOO_LOW_MINIMUM_HEIGHT_M": (
            _L,
            "phase_termination:base_too_low_minimum_height_m",
        ),
        "TERMINATION_SOURCE_CONFIG": (_M, ""),
        "TERMINATION_SOURCE_CALLABLES": (_M, ""),
        "TERMINATION_SOURCE_ACTION_LATCH": (_M, ""),
        "TERMINATION_SOURCE_PHASE_WRAPPERS": (_P, ""),
        "TERMINATION_SOURCE_PHASE_GATE": (_P, ""),
        "TERMINATION_SOURCE_BASE_CONFIG": (_M, ""),
        "TERMINATION_SOURCE_A3_BODY_NAMES": (_P, ""),
        "EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256": (
            _X,
            "_phase_fidelity_sample_contract_cached 用选择器重算;它保护的值另有 "
            "live_isaac_termination_constant_blockers 逐个比值",
        ),
        "EXPECTED_PHASE_BASE_CONFIG_SEMANTIC_AST_SHA256": (
            _X,
            "_phase_fidelity_sample_contract_cached;根类的类体顺序另有 "
            "live_termination_reason_order_blockers 逐位比",
        ),
        "EXPECTED_PHASE_RAW_CALLABLES_SEMANTIC_AST_SHA256": (
            _X,
            "_phase_fidelity_sample_contract_cached;判据本体是函数体",
        ),
        "EXPECTED_PHASE_WRAPPERS_SEMANTIC_AST_SHA256": (
            _X,
            "_phase_fidelity_sample_contract_cached;判据本体是函数体",
        ),
        "EXPECTED_PHASE_GATE_SEMANTIC_AST_SHA256": (
            _X,
            "_phase_fidelity_sample_contract_cached;判据本体是函数体",
        ),
        "EXPECTED_PHASE_BODY_NAMES_SEMANTIC_AST_SHA256": (
            _X,
            "_phase_fidelity_sample_contract_cached;名单值本身由 "
            "live_body_name_vocabulary 读活值",
        ),
        "JOINT_ACTUAL_FORBIDDEN_BOUNDS_TOLERANCE_RAD": (_M, ""),
        "JOINT_QDES_FORBIDDEN_MARGIN_RAD": (
            _L,
            "phase_termination:joint_qdes_forbidden_margin_rad",
        ),
        "JOINT_QDES_FORBIDDEN_MARGIN_FRACTION": (
            _L,
            "phase_termination:joint_qdes_forbidden_margin_fraction",
        ),
        "JOINT_QDES_FINITE_PROJECTION_ENABLED": (_N, ""),
        "PHASE_ANCHOR_POS_Z_THRESHOLD_M": (
            _L,
            "phase_termination:phase_anchor_pos_z_threshold_m",
        ),
        "PHASE_ANCHOR_ORI_PROJECTED_GRAVITY_Z_THRESHOLD": (
            _L,
            "phase_termination:phase_anchor_ori_projected_gravity_z_threshold",
        ),
        "PHASE_TERMINATIONS_MIRRORED_ISAAC_CLASS": (
            _L,
            "reference_envelope:mirrored_class",
        ),
        "_ACTION_BALL_REFERENCE_ENVELOPE": (_D, ""),
        "PHASE_EE_BODY_POS_Z_THRESHOLD_M": (
            _L,
            "reference_envelope:ee_body_pos_z_threshold_m",
        ),
        "PHASE_EE_BODY_NAMES": (_L, "reference_envelope:ee_body_names"),
        "PHASE_CONTEXTS": (_N, ""),
        "CONTACT_EVENT_LABELS": (_N, ""),
        "PLANT_COUNTER_KEYS": (_N, ""),
        "PLANT_MAX_KEYS": (_N, ""),
        "_PHASE_LIVE_COMPANIONS": (_M, ""),
        "_DEPLOY_PARITY_CFG": (_M, ""),
        "_ACTION_BALL_CFG": (_M, ""),
    },
}

#: Modules with no module-level constants at all still have to be named, so a
#: new file cannot appear unnoticed.
MODULES_WITHOUT_CONSTANTS = ("__init__.py",)


# ---------------------------------------------------------------------------
# Reading the modules back
# ---------------------------------------------------------------------------


def lane_modules() -> tuple:
    """Every ``.py`` directly under ``mujoco_native`` (not the sub-packages)."""

    return tuple(sorted(path.name for path in MODULE_DIR.glob("*.py")))


def module_level_constants(module_name: str) -> dict:
    """``NAME -> assigned AST node`` for module-level upper-case assignments."""

    source = MODULE_DIR / module_name
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    found: dict = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets: Sequence[ast.AST] = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
        else:
            continue
        value = getattr(statement, "value", None)
        if value is None:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id.isupper()
                and not target.id.startswith("__")
            ):
                found[target.id] = value
    return found


def _is_literal(node: ast.AST) -> bool:
    """True when the assignment is a plain hand-written literal."""

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_literal(key) and _is_literal(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp):
        return _is_literal(node.operand)
    return False


def _module_attribute(module_name: str, dotted: str) -> Any:
    module = importlib.import_module(f"{PACKAGE}.{Path(module_name).stem}")
    value: Any = module
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def _is_ordered_subset(needle: Sequence, haystack: Sequence) -> bool:
    iterator = iter(haystack)
    return all(item in iterator for item in needle)


def _same(left: Any, right: Any) -> bool:
    """Value equality that survives numpy arrays and tuple/list mixing."""

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency here
        np = None
    if np is not None and (
        isinstance(left, np.ndarray) or isinstance(right, np.ndarray)
    ):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def registry_blockers() -> tuple:
    """Every way this registry no longer describes the lane it claims to.

    人话:这就是那道通用门。返回空 = "每个模块级常量都被显式分类了,而且每条自称
    '被活值罩着'的都真的对上了活值比对入口拿去比的那个数"。
    """

    blockers: list = []

    declared_modules = set(CLASSIFICATION) | set(MODULES_WITHOUT_CONSTANTS)
    live_modules = set(lane_modules())
    for name in sorted(live_modules - declared_modules):
        blockers.append(
            f"mirror_module_unclassified:{name}:新模块必须在 CLASSIFICATION 或 "
            "MODULES_WITHOUT_CONSTANTS 里显式登记"
        )
    for name in sorted(declared_modules - live_modules):
        blockers.append(f"mirror_module_absent:{name}:登记了但文件不在了")

    for name in sorted(set(MODULES_WITHOUT_CONSTANTS) & live_modules):
        present = module_level_constants(name)
        if present:
            blockers.append(
                f"mirror_module_grew_constants:{name}:{sorted(present)!r}"
            )

    provider_cache: dict = {}
    used_providers: set = set()

    for module_name in sorted(set(CLASSIFICATION) & live_modules):
        table = CLASSIFICATION[module_name]
        present = module_level_constants(module_name)
        stem = Path(module_name).stem
        for constant in sorted(set(present) - set(table)):
            blockers.append(
                f"mirrored_constant_unclassified:{stem}.{constant}:"
                "新增的模块级常量必须显式说明它是怎么被罩住的"
            )
        for constant in sorted(set(table) - set(present)):
            blockers.append(
                f"mirrored_constant_absent:{stem}.{constant}:"
                "登记了但常量已经不在模块里了"
            )

        for constant in sorted(set(table) & set(present)):
            entry = table[constant]
            if not isinstance(entry, tuple) or len(entry) != 2:
                blockers.append(
                    f"mirrored_constant_entry_malformed:{module_name}.{constant}"
                )
                continue
            reason, detail = entry
            qualified = f"{Path(module_name).stem}.{constant}"
            if reason not in REASONS:
                blockers.append(
                    f"mirrored_constant_reason_unknown:{qualified}:{reason!r}"
                )
                continue
            if reason in REASONS_REQUIRING_DETAIL and not str(detail).strip():
                blockers.append(
                    f"mirrored_constant_detail_missing:{qualified}:{reason} "
                    "这一档必须写清楚是谁在比"
                )
                continue

            node = present[constant]

            if reason in (LIVE_VALUE_DERIVED, DERIVED_IN_MODULE):
                if _is_literal(node):
                    blockers.append(
                        f"mirrored_constant_is_a_literal_after_all:{qualified}:"
                        f"{reason} 这一档要求它是算出来/读出来的,不是写死的字面量"
                    )
                continue

            if reason == LIVE_SOURCE_PATH:
                try:
                    value = _module_attribute(module_name, constant)
                    exists = Path(value).is_file()
                except Exception as exc:  # noqa: BLE001 - must fail closed
                    blockers.append(
                        f"mirrored_source_path_unreadable:{qualified}:{exc}"
                    )
                    continue
                if not exists:
                    blockers.append(
                        f"mirrored_source_path_absent:{qualified}:{value}"
                    )
                continue

            if reason == PINNED_FILE_DIGEST:
                try:
                    pinned = _module_attribute(module_name, constant)
                    path = Path(_module_attribute(module_name, str(detail)))
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:  # noqa: BLE001 - must fail closed
                    blockers.append(
                        f"mirrored_file_digest_unreadable:{qualified}:{exc}"
                    )
                    continue
                if actual != pinned:
                    blockers.append(
                        f"mirrored_file_digest_differs:{qualified}:"
                        f"file={path.name} live={actual} pin={pinned}"
                    )
                continue

            if reason == PINNED_EXTERNAL_DIGEST:
                try:
                    pinned = _module_attribute(module_name, constant)
                except Exception as exc:  # noqa: BLE001 - must fail closed
                    blockers.append(
                        f"mirrored_external_digest_unreadable:{qualified}:{exc}"
                    )
                    continue
                if not (
                    isinstance(pinned, str)
                    and len(pinned) == 64
                    and all(item in "0123456789abcdef" for item in pinned)
                ):
                    blockers.append(
                        f"mirrored_external_digest_malformed:{qualified}:{pinned!r}"
                    )
                continue

            if reason == MIRRORED_TODO:
                if qualified not in OPEN_MIRROR_DEBT:
                    blockers.append(
                        f"mirrored_todo_without_a_written_plan:{qualified}:"
                        "MIRRORED_TODO 必须在 OPEN_MIRROR_DEBT 里写明"
                        "真源在哪 / 怎么修 / 为什么这轮没修"
                    )
                continue

            if reason == NOT_MIRRORED:
                continue

            # LIVE_VALUE_COMPARED / FLOWS_INTO_LIVE_COMPARISON
            spec, _, accessor = str(detail).partition(" via ")
            provider_name, _, key = spec.partition(":")
            used_providers.add(provider_name)
            if provider_name not in LIVE_VALUE_PROVIDERS:
                blockers.append(
                    f"mirrored_live_provider_unknown:{qualified}:{provider_name!r}"
                )
                continue
            if provider_name not in provider_cache:
                try:
                    provider_cache[provider_name] = dict(
                        LIVE_VALUE_PROVIDERS[provider_name]()
                    )
                except Exception as exc:  # noqa: BLE001 - must fail closed
                    provider_cache[provider_name] = exc
            provided = provider_cache[provider_name]
            if isinstance(provided, Exception):
                blockers.append(
                    f"mirrored_live_provider_failed:{qualified}:{provided}"
                )
                continue
            if key in provided:
                compared = provided[key]
            elif key in _PARTITION_KEYS:
                compared = _partition_value(key)
            else:
                blockers.append(
                    f"mirrored_live_key_absent:{qualified}:{key!r} 不在 "
                    f"{provider_name} 实际拿去比的那批值里"
                )
                continue
            try:
                mirrored = _module_attribute(module_name, constant)
                if accessor.strip():
                    mirrored = _module_attribute(module_name, accessor.strip())()
            except Exception as exc:  # noqa: BLE001 - must fail closed
                blockers.append(f"mirrored_constant_unreadable:{qualified}:{exc}")
                continue

            if reason == FLOWS_INTO_LIVE_COMPARISON:
                container = tuple(compared)
                if isinstance(mirrored, str):
                    contained = mirrored in container
                else:
                    contained = _is_ordered_subset(tuple(mirrored), container)
                if not contained:
                    blockers.append(
                        f"mirrored_constant_not_in_the_compared_object:{qualified}:"
                        f"constant={mirrored!r} compared_object={container!r} —— "
                        "它自称流进的那个活值比对里根本没有它"
                    )
                continue

            if not _same(mirrored, compared):
                blockers.append(
                    f"mirrored_constant_not_the_compared_value:{qualified}:"
                    f"constant={mirrored!r} compared={compared!r} —— 这个常量并没有"
                    "流进它自称的那道活值比对"
                )

    for name in sorted(set(LIVE_VALUE_PROVIDERS) - used_providers):
        blockers.append(
            f"mirrored_live_provider_unused:{name}:没有任何常量引用它,"
            "要么是接线断了,要么这个 provider 该删"
        )

    for qualified in sorted(OPEN_MIRROR_DEBT):
        module_stem, _, constant = qualified.rpartition(".")
        table = CLASSIFICATION.get(f"{module_stem}.py", {})
        if table.get(constant, (None, None))[0] != MIRRORED_TODO:
            blockers.append(
                f"mirror_debt_is_stale:{qualified}:"
                "OPEN_MIRROR_DEBT 里还留着,但常量已经不是 MIRRORED_TODO 了"
            )
        note = OPEN_MIRROR_DEBT[qualified]
        if not (isinstance(note, tuple) and len(note) == 3) or any(
            not str(item).strip() for item in note
        ):
            blockers.append(
                f"mirror_debt_incomplete:{qualified}:"
                "必须是(真源在哪, 怎么修, 为什么这轮没修)三段,且都不能为空"
            )

    return tuple(blockers)


#: Buckets the reason-order provider does not publish directly, because they are
#: a *partition* of the live hard order rather than a value the live source
#: names.  Resolved against the live order so the replica's three lists cannot
#: quietly stop covering it.
_PARTITION_KEYS = (
    "phase_fidelity_bucket",
    "base_and_joint_bucket",
    "table_guard_bucket",
)


def _partition_value(key: str) -> Any:
    """The live slice of the hard order that a replica bucket must equal."""

    module = importlib.import_module(f"{PACKAGE}.vec_env")
    hard = _reason_order_values()["hard_order"]
    buckets = {
        "phase_fidelity_bucket": module.EXACT_PHASE_FIDELITY_REASON_ORDER,
        "base_and_joint_bucket": module.EXACT_BASE_TERMINATION_REASON_ORDER,
        "table_guard_bucket": module.EXACT_TABLE_GUARD_REASON_ORDER,
    }
    claimed = set(buckets[key])
    # The live projection: exactly the live hard terms this bucket claims, in the
    # live order.  Equality with the replica constant therefore proves both "no
    # term it claims is gone" and "it is still in Isaac's order".
    return tuple(term for term in hard if term in claimed)


def registry_receipt() -> dict:
    """Self-reported telemetry: how much of the lane is guarded, and how."""

    counts: dict = {reason: 0 for reason in REASONS}
    for table in CLASSIFICATION.values():
        for reason, _detail in table.values():
            if reason in counts:
                counts[reason] += 1
    return {
        "schema_version": 1,
        "kind": "a3_mujoco_mirrored_constant_registry_v1",
        "modules_classified": len(CLASSIFICATION),
        "constants_classified": sum(len(table) for table in CLASSIFICATION.values()),
        "constants_by_reason": counts,
        "live_value_providers": sorted(LIVE_VALUE_PROVIDERS),
        "open_mirror_debt": sorted(OPEN_MIRROR_DEBT),
    }


__all__ = [
    "CLASSIFICATION",
    "FLOWS_INTO_LIVE_COMPARISON",
    "LIVE_VALUE_COMPARED",
    "LIVE_VALUE_DERIVED",
    "LIVE_VALUE_PROVIDERS",
    "MIRRORED_TODO",
    "MODULES_WITHOUT_CONSTANTS",
    "MirroredConstantRegistryError",
    "NOT_MIRRORED",
    "OPEN_MIRROR_DEBT",
    "PINNED_EXTERNAL_DIGEST",
    "PINNED_FILE_DIGEST",
    "REASONS",
    "lane_modules",
    "module_level_constants",
    "registry_blockers",
    "registry_receipt",
]
