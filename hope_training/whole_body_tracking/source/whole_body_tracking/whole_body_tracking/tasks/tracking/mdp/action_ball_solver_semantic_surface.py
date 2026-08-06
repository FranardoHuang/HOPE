"""Per-symbol semantic surface of the action-ball fixed-action solver.

人话:这份文件回答一个问题 —— "**改了哪些代码,才算改了题**"。

以前那枚 solver profile pin 对五份源文件做**整文件 SHA**。整文件 SHA 分不清
"改了求解器的数学" 和 "改了 checkpoint 怎么存盘",于是一次纯注释提交、一次
序列化作用域重构,都会让训练在 boot 处硬崩,而题目一个字没变。反过来它也没有
更严:``strike_spec_torch.py`` 里的定向逆解种子函数**根本不在那五份里**,今天
改掉它就能改掉答案而 pin 纹丝不动。

这份文件把那枚 pin 从"整文件字节"改成"**逐符号语义**":

* ``COVERED`` 显式列出进入指纹的符号(函数/类/常量),按文件分组。指纹只对这些
  符号取,取的是**剥掉 docstring、跨 Python 版本归一化后的 AST**,所以注释、
  空行、换行位置、docstring 都不动它,而任何表达式/常量/字段顺序的改动都动它。
* ``EXCLUDED`` 显式列出**有意排除**的符号,每一个都带一个理由码。排除是**列举
  式**的,不是默认放行。
* 三道 fail-closed 的门(见 ``surface_blockers``)保证"覆盖面不小于它声称保护的
  语义面":
  1. 五份纯求解器源文件(``FULLY_ENUMERATED_SOURCES``)里**每一个**符号都必须
     出现在 COVERED 或 EXCLUDED 里 —— 新加一个函数而不分类,直接拒绝启动。
  2. 任何被 COVERED 符号引用、且能解析到被钉文件里的名字,也必须已分类 ——
     入口开始调一个新助手函数而不分类,直接拒绝启动。
  3. 排除理由里凡是**声称"这条路根本走不到它"**的那几种
     (``UNREACHABLE_CLAIM_REASONS``),必须真的走不到 —— 一旦某个 COVERED 符号
     引用了它,这条理由就是假话,直接拒绝启动。

排除清单**不进指纹**。这正是本次收窄的目的:新增一个"存盘/记账/遥测"符号
必须被显式分类(否则门开火),但分类完之后 pin 不动,训练不再被无关提交打断。
把一个已覆盖符号挪进排除清单则一定会动 pin —— 因为它的摘要从 ``covered`` 里
消失了。

**指纹管不到的那一半:声明与实际之间。** pin 封的是"payload 里声明的数字";
真正喂给求解器的数字要经过一条**传递线**,而那条线住在
``RacketTargetCommand._initialize_action_ball_runtime``(1700 多行接线,理由码
``runtime_wiring``)。把整条接线塞进覆盖面会让每次接线改动都作废题库,那就退回
了本模块要解决的原问题;所以传递线换了个形状:

* ``hope_commands.action_ball_declared_solver_knobs`` 是那五个旋钮的**唯一出处**,
  payload 和自检读的是同一份(不许有第三份手抄的期望值);
* ``hope_commands.action_ball_solver_cfg_from_declaration`` 是"旋钮 -> 求解器 cfg"
  的**唯一一处映射**,窄到可以整个进覆盖面;
* ``hope_commands.action_ball_assert_solver_runtime_matches_declaration`` 在
  **三个入口点里**把封好的 payload 和活着的 ``solver_cfg`` / ``prm`` / 三个平面 /
  rollout 参数逐字段比,不等就 fail-closed。入口点在覆盖面里,所以**删掉这次调用
  本身也会动 pin**。

反例(这三条改动以前全部逃逸,是本模块自己引入的净放宽):``tol_m`` 乘 0.5、
``speed_budget`` 乘 2、``fixed_direction=True -> False``。

前两条是**静默改答案**:题照出、答案变了,没有任何门会响。第三条不静默 ——
``solve_proposals`` 的前置 ``_validate_external_proposals`` 会以
"solve_proposals is fixed-direction only" 当场拒绝,整条 action-ball 路在第一次
补池时硬崩(``generate`` 那条会分叉到 free-direction 的路**不是** action-ball
在走的路,这一点必须说准)。但它同样是个真缺陷:pin 的 payload 里
``"fixed_direction": True`` 这句**声明**变成了假话,而 pin 一动不动。

门 3 管的是另外半边:``other_product_line``("action-ball 从不调用它")这类理由
以前只是**写在这里的一句话**,没人验过。现在它是可执行的断言。

指纹算法刻意不使用 ``ast.dump``:``ast.dump`` 的字段集合在 3.8→3.12 之间变过
(``Index`` 包装、``type_params``),会让同一份源码在不同解释器上给出不同摘要。
``_normalized_ast_text`` 只走一份显式的字段白/黑名单并拆掉 ``Index``/``ExtSlice``。
"""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

SEMANTIC_SURFACE_SCHEMA_VERSION = 1
SEMANTIC_SURFACE_KIND = "whole_body_tracking.action_ball.solver_semantic_surface"
SYMBOL_DIGEST_ALGORITHM = "docstring_stripped_version_normalized_ast_sha256_v1"

#: The three call-graph entry points that define "turning a drawn question into
#: accept/reject plus the answer".  They are documented here because the covered
#: set below was adjudicated as their forward closure; the closure gate in
#: ``surface_blockers`` keeps that claim honest as the code moves.
SEMANTIC_ENTRY_POINTS = (
    "hope_commands.py:RacketTargetCommand._action_ball_refill_pool_many",
    "hope_commands.py:RacketTargetCommand._action_ball_frozen_eval_solve",
    "hope_commands.py:RacketTargetCommand._action_ball_replay_emitted_tasks",
)

#: Sources whose *every* symbol must be classified.  These five files exist only
#: to solve, so enumerating them completely is cheap and leaves no default-allow
#: hole.  ``hope_commands.py`` is deliberately not in this set: it is a 26k-line
#: module that owns dozens of unrelated command terms, and its solver surface is
#: guarded by the reference-closure gate instead.
FULLY_ENUMERATED_SOURCES = (
    "continuous_questions.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "strike_spec_torch.py",
    "virtual_ball.py",
)

#: Every source this surface reads.  ``strike_spec_torch.py`` is new here: it was
#: in no pin at all (not solver, not runtime, not the offline pinner) while
#: ``stroke_adapt_torch`` imports its ``_seed``/``_face_from_angles`` and every
#: fixed-direction inverse solve runs through them.
PINNED_SOURCES = ("hope_commands.py",) + FULLY_ENUMERATED_SOURCES

EXCLUSION_REASONS: Dict[str, str] = {
    "checkpoint_state_serialization": (
        "Encodes/decodes/reconciles the exact-resume state packet. It does not "
        "decide what the question is, only how that state is written, read and "
        "audited. It keeps its own fail-closed mechanism "
        "(_ACTION_BALL_SOLVER_STATE_SCHEMA_VERSION plus an integrity_sha256 that "
        "covers every field); this exclusion is only valid while that mechanism "
        "stays fail-closed."
    ),
    "birth_audit_ledger": (
        "Per-birth issue/transcript bookkeeping and its range predicates. Every "
        "reader only reconciles or admits; none of it feeds an RNG or any "
        "question field. It decides whether a run may continue, not what is asked."
    ),
    "runtime_wiring": (
        "Chooses which question source is bound (online solver / immutable tape / "
        "banded question bank) and constructs the adapters. That choice is already "
        "pinned by the target-source branch plus the tape/bank SHA in cfg. This "
        "exclusion does NOT say the wiring cannot move an answer -- it can, by "
        "handing the solver a number other than the one the payload declares. "
        "That is why every declared number is cross-checked against the live "
        "object inside the covered entry points by "
        "action_ball_assert_solver_runtime_matches_declaration; this exclusion is "
        "only valid while that comparison stays fail-closed and stays called."
    ),
    "telemetry_and_counters": (
        "Notes, ledger payloads and rejection histograms. Reporting only; no "
        "influence on accept/reject or on the answer."
    ),
    "grading_and_observation": (
        "Scores or observes what the racket actually achieved. The solver profile "
        "owns the demanded contact state; grading belongs to the reward/grading "
        "contract. The two must agree, which is why this exclusion is named out "
        "loud rather than left implicit."
    ),
    "question_production_sampling": (
        "Produces the birth content (base pose, domain levels) rather than solving "
        "it. Already pinned by sampler_contract_sha256, levels_sha256 and "
        "domain_authority_sha256."
    ),
    "other_product_line": (
        "Belongs to the free-direction target_mode='solved' product line. "
        "action-ball never calls it; the solver profile kind is literally "
        "whole_body_tracking.continuous_questions.solve_proposals."
    ),
    "venue_parameter_loading": (
        "Parses the venue YAML. The ten parsed numbers are written into the "
        "physics profile payload one by one, so a parser change already moves the "
        "physics profile SHA."
    ),
    "stroke_selector": (
        "Picks which stroke prototype to use. The action-ball action is frozen per "
        "episode, so the selector is unreachable on this path."
    ),
    "swept_contact_grading": (
        "In-rally swept contact adjudication. Grading, not question production."
    ),
    "convenience_accessor": (
        "A scalar/length convenience wrapper whose logic the solver inlines at its "
        "own call site; the inlined comparison is covered, this wrapper is not on "
        "the path."
    ),
    "self_check_only": (
        "Boot-time self test / negative control. It reports, it does not decide."
    ),
    "module_export_list": (
        "__all__ re-export bookkeeping. Renaming an export cannot change an answer; "
        "the exported callables are themselves classified."
    ),
    "overapproximated_name_collision": (
        "The closure gate resolves references by bare name on purpose, so an "
        "unrelated symbol that happens to share a name with an attribute used on "
        "the solver path gets pulled into the domain. This entry says out loud "
        "that the collision was looked at and the symbol is not solver logic."
    ),
}

#: Reason codes whose *text* asserts the symbol is never reached on the solve
#: path.  That claim is machine-checkable, so gate 3 in ``surface_blockers``
#: checks it instead of trusting the wording: if a covered symbol references one
#: of these, the stated reason is false and the surface refuses to build.
#:
#: 人话:排除理由分两种。一种是"它根本不在这条路上"(可验证),一种是"它在路上
#: 但改不了答案"(要人来判)。前者不许只靠命名 —— ``fixed_direction`` 那次事故
#: 正是"一次编辑同时让声明和排除理由都变成假话",而假话的那半边恰好是可验证的
#: 那一种。
UNREACHABLE_CLAIM_REASONS = frozenset(
    {
        "other_product_line",
        "stroke_selector",
        "convenience_accessor",
    }
)

# --------------------------------------------------------------------------- #
# Covered symbols: the semantic surface itself.                                #
# --------------------------------------------------------------------------- #
COVERED: Dict[str, Tuple[str, ...]] = {
    "hope_commands.py": (
        # The three entry points carry real semantic constants in their own
        # bodies, not just wiring: the solver_field_contract tuple order, the
        # bare `<= 1.0` pre-swing wait bound, the 1.0e-12 cycle-vs-horizon
        # epsilon and the `net_x + BALL_BIRTH_NET_MARGIN_M` comparison sense.
        # None of those four numbers is declared anywhere in the payload.
        "RacketTargetCommand._action_ball_refill_pool_many",
        "RacketTargetCommand._action_ball_frozen_eval_solve",
        "RacketTargetCommand._action_ball_replay_emitted_tasks",
        # C211's no-inverse-solve branch decides whether its questions are
        # admitted at all.
        "RacketTargetCommand._action_ball_assert_emitted_task_reference_and_timing",
        # The question identity function: add or drop a field here and every
        # question is renamed.
        "_action_ball_exact_question_payload",
        "_action_ball_semantic_levels",
        "_action_ball_canonical_sha256",
        # The declaration half of the contract. Pinning it alongside the
        # execution half is what stops "declaration says 1.4, code says 0.05"
        # from being invisible.
        "action_ball_solver_profile_contract",
        "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION",
        # The declaration/actual bridge.  A payload number and the number the
        # solver is handed are two different things, and the wiring between
        # them lives in ``_initialize_action_ball_runtime`` -- 1700 lines this
        # surface deliberately does NOT cover, because covering it would
        # invalidate the question bank on every wiring edit.  These six symbols
        # are the narrow, coverable replacement: one function owns the knob
        # values, one owns the single mapping into ``ContinuousQuestionCfg``,
        # one compares the sealed declaration against the live objects field by
        # field, and three constants carry the values those three read.  All
        # three solve entry points call the comparison, so removing the call
        # moves this pin too.
        "action_ball_declared_solver_knobs",
        "action_ball_solver_cfg_from_declaration",
        "action_ball_assert_solver_runtime_matches_declaration",
        "_ACTION_BALL_SOLVER_FIXED_DIRECTION",
        # The named list of venue-physics numbers the cross-check walks: shrink
        # it and the check silently stops comparing a parameter the solver
        # integrates with.
        "_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES",
        # The one departure from the declaration a diagnostic run is allowed to
        # make. It is a constant so that "the diagnostic exemption" cannot be
        # widened into a free expression without moving the pin.
        "_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS",
        # Which of the two solver paths a question takes.
        "RacketTargetCommand._counter_rally_enabled",
        # The executable knobs the payload declares numerically. Pinning the
        # declaration alongside the number is what stops a default edit from
        # sliding past an offline pin document that was minted before it.
        "RacketTargetCommandCfg.cq_n_iters",
        "RacketTargetCommandCfg.cq_tol_m",
        "RacketTargetCommandCfg.cq_speed_budget",
        "RacketTargetCommandCfg.cq_max_redraw_rounds",
        "RacketTargetCommandCfg.cq_overdraw",
        "RacketTargetCommandCfg.vb_rollout_h",
        "RacketTargetCommandCfg.vb_rollout_steps",
        "RacketTargetCommandCfg.mount_normal_sign",
    ),
    "continuous_questions.py": (
        "_EPS",
        "_DIAGNOSTIC_PREVALIDATED_SOLVE_AUTHORITY",
        "_R_NO_LANDING",
        "_R_RESID",
        "_R_SPEED_OVER",
        "_R_SPEED_UNDER",
        "_R_NET",
        "_R_FACE",
        "_R_CONTACT_ENVELOPE",
        "_CONTINUOUS_REASONS",
        "BALL_BIRTH_NET_MARGIN_M",
        "BALL_BIRTH_REJECTION_REASON",
        "ball_birth_x_lower_bound_m",
        "CONTACT_NORMAL_SPEED_MIN_MPS",
        "CONTACT_NORMAL_SPEED_MAX_MPS",
        "ContinuousQuestionCfg",
        "ContinuousQuestionCfg.vel_range",
        "ContinuousQuestionCfg.vel_range_per_clip",
        "ContinuousQuestionCfg.spin_abs_max",
        "ContinuousQuestionCfg.spin_abs_max_per_clip",
        "ContinuousQuestionCfg.pos_range_per_clip",
        "ContinuousQuestionCfg.pos_range",
        "ContinuousQuestionCfg.aim_x_range",
        "ContinuousQuestionCfg.aim_y_range",
        "ContinuousQuestionCfg.tol_m",
        "ContinuousQuestionCfg.n_iters",
        "ContinuousQuestionCfg.speed_budget",
        "ContinuousQuestionCfg.max_redraw_rounds",
        "ContinuousQuestionCfg.fixed_direction",
        "ProposalLedger",
        "ProposalLedger.request_index",
        "ProposalLedger.clip_id",
        "ProposalLedger.round_index",
        "ProposalLedger.p_contact",
        "ProposalLedger.v_ball_in",
        "ProposalLedger.w_ball_in",
        "ProposalLedger.aim_xy",
        "ProposalLedger.reason_code",
        "ProposalLedger.admitted",
        "ProposalLedger.resid_m",
        "ProposalLedger.ref_normal",
        "ProposalLedger.base_quat",
        "ProposalHostPacket",
        "ProposalHostPacket.reason_codes",
        "ProposalHostPacket.admitted",
        "ProposalHostPacket.racket_velocity_rows",
        "ProposalHostPacket.racket_normal_rows",
        "ProposalHostPacket.residual_rows",
        "QuestionDrawResult",
        "QuestionDrawResult.p_contact",
        "QuestionDrawResult.v_racket",
        "QuestionDrawResult.n_racket",
        "QuestionDrawResult.v_ball_in",
        "QuestionDrawResult.w_ball_in",
        "QuestionDrawResult.aim_xy",
        "QuestionDrawResult.ok",
        "QuestionDrawResult.resid_m",
        "QuestionDrawResult.attempted_v_ball_in",
        "QuestionDrawResult.rounds_used",
        "QuestionDrawResult.exhausted",
        "QuestionDrawResult.reason_counts",
        "QuestionDrawResult.proposal_count",
        "QuestionDrawResult.proposals",
        "QuestionDrawResult.proposal_host_packet",
        "_build_proposal_host_packet",
        "_rows",
        "_selected_direction_world",
        "_fixed_direction_contract",
        "_fixed_direction_replay",
        "_solve_fixed_direction_batch",
        "_validate_external_proposals",
        "_diagnostic_prevalidated_external_proposals",
        "solve_proposals",
        "_solve_proposals_diagnostic_host_only",
    ),
    "racket_contact_geometry.py": (
        "Vec3",
        "Quat",
        "Matrix3",
        "EXACT_FACE_CONTACT_SCHEMA_VERSION",
        "EXACT_FACE_CONTACT_KIND",
        "RACKET_SITE_OFFSET_WRIST_M",
        "RACKET_BUTT_TO_BLADE_AXIS_LOCAL",
        "RACKET_RIGID_VISUAL_MESH_SHA256",
        "LEGACY_ISAAC_SITE_OFFSET_WRIST_M",
        "FACE_AREA_CENTER_XZ_FROM_SITE_M",
        "RED_OUTER_Y_FROM_SITE_M",
        "BLACK_OUTER_Y_FROM_SITE_M",
        "BALL_RADIUS_M",
        "RED_SELECTED_FACE_MESH_SHA256",
        "BLACK_SELECTED_FACE_MESH_SHA256",
        "SELECTED_FACE_CENTER_TO_BOUNDARY_MIN_M",
        "FORMAL_FACE_EDGE_GUARD_M",
        "SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M",
        "SELECTED_FACE_SWEEP_CLEARANCE_TOLERANCE_M",
        "SELECTED_FACE_SWEEP_BISECTION_STEPS",
        "SELECTED_FACE_SWEEP_BALL_BACKPROP_MAX_DT_S",
        "POLAR_ROTATION_SINGULAR_TOLERANCE",
        "OFFICIAL_RED_BALL_CENTER_FROM_SITE_M",
        "RED_FACE_SIGN",
        "BLACK_FACE_SIGN",
        "TEACHER_RATE_BOUNDARY_ABS_TOL",
        "QUATERNION_UNIT_PRESERVE_ABS_TOL",
        "GEOMETRY_SOURCE_PAYLOAD",
        "GEOMETRY_SOURCE_BYTES",
        "GEOMETRY_SOURCE_SHA256",
        "ExactFaceContactGeometryError",
        "ExactFaceContactGeometryError.__init__",
        "canonical_teacher_rate_from_site_speed",
        "_finite",
        "_vec3",
        "_quat",
        "_validate_face_sign",
        "_add",
        "_sub",
        "_scale",
        "_dot",
        "_cross",
        "_norm",
        "_unit",
        "face_normal_local",
        "face_center_from_site_local",
        "ball_center_from_site_local",
        "quat_multiply_wxyz",
        "canonical_quat_wxyz",
        "quat_rotate_wxyz",
        "minimal_rotation_quat_wxyz",
        "command_orientation_preserve_reference_twist",
        "site_target_from_ball_center",
        "ExactFaceContactSolution",
        "ExactFaceContactSolution.geometry_source_sha256",
        "ExactFaceContactSolution.mount_normal_sign",
        "ExactFaceContactSolution.racket_command_quat_wxyz",
        "ExactFaceContactSolution.racket_site_target_w_m",
        "ExactFaceContactSolution.racket_face_center_velocity_w_mps",
        "ExactFaceContactSolution.racket_site_velocity_w_mps",
        "ExactFaceContactSolution.racket_command_angular_velocity_w_radps",
        "ExactFaceContactSolution.teacher_rate",
        "solve_exact_face_contact",
        "torch_exact_contact_state",
        # solve_exact_face_contact calls this to turn the face-centre velocity
        # into the installed site velocity; it is the executable half of the
        # payload's velocity_points.mapping string.
        "site_velocity_from_face_center",
    ),
    "stroke_adapt_torch.py": (
        "_EPS",
        "REASONS",
        "BALL_RADIUS_M",
        "_DIAGNOSTIC_FIXED_TRY_LM_AUTHORITY",
        "base_yaw_of",
        "_forward_landing_fixed_dir",
        "solve_strike_specs_fixed_dir",
        "radians",
    ),
    "strike_spec_torch.py": (
        "_EPS",
        "_face_from_angles",
        "_seed",
    ),
    "virtual_ball.py": (
        "_EPS",
        "VirtualBallParams",
        "VirtualBallParams.k_d",
        "VirtualBallParams.k_m",
        "VirtualBallParams.g",
        "VirtualBallParams.ball_radius",
        "VirtualBallParams.inertia_coeff",
        "VirtualBallParams.paddle_a_t",
        "VirtualBallParams.paddle_b_t",
        "VirtualBallParams.paddle_mu",
        "VirtualBallParams.paddle_e_g1",
        "VirtualBallParams.paddle_e_g2",
        "VirtualBallParams.source_path",
        "_normalize",
        "orient_normal",
        "_paddle_contact_kinematics",
        "predict_paddle_contact",
        "flight_accel",
        "rk4_step",
        "_coarse_landing_eager",
        "_FAST_ENV",
        "_fused_kernel",
        "_parity_cache",
        "_build_fused_kernel",
        "_f32",
        "_FUSED_BLOCK",
        "_fused_rollout",
        "_parity_probe",
        "_fast_path_admitted",
        "coarse_landing",
    ),
}

# --------------------------------------------------------------------------- #
# Deliberate exclusions.  Explicit, by name, with a reason code each.          #
# Adding a symbol here does NOT move the pin; removing one from COVERED does.  #
# --------------------------------------------------------------------------- #
EXCLUDED: Dict[str, Dict[str, str]] = {
    "hope_commands.py": {
        # --- checkpoint serialization: the main battleground of eccb30cd /
        # 308db7f0, and the reason this narrowing exists at all.
        "RacketTargetCommand._action_ball_solver_mutable_state_dict": "checkpoint_state_serialization",
        "RacketTargetCommand._action_ball_decode_solver_mutable_state": "checkpoint_state_serialization",
        "RacketTargetCommand._action_ball_load_solver_mutable_state": "checkpoint_state_serialization",
        "RacketTargetCommand._action_ball_exact_resume_state_dict": "checkpoint_state_serialization",
        "RacketTargetCommand._action_ball_validate_exact_resume_state_dict": "checkpoint_state_serialization",
        "RacketTargetCommand._action_ball_load_exact_resume_state_dict": "checkpoint_state_serialization",
        "_ActionBallPoolSolverAdapter.state_dict": "checkpoint_state_serialization",
        "_ActionBallPoolSolverAdapter.load_state_dict": "checkpoint_state_serialization",
        "_ActionBallBirthProviderAdapter.state_dict": "checkpoint_state_serialization",
        "_ActionBallBirthProviderAdapter.load_state_dict": "checkpoint_state_serialization",
        "_ActionBallDomainAuthorityAdapter.state_dict": "checkpoint_state_serialization",
        "_ActionBallDomainAuthorityAdapter.load_state_dict": "checkpoint_state_serialization",
        "_ActionBallDrainResetRuntimeSource.state_dict": "checkpoint_state_serialization",
        "_ActionBallDrainResetRuntimeSource.load_state_dict": "checkpoint_state_serialization",
        "_ACTION_BALL_SOLVER_STATE_SCHEMA_VERSION": "checkpoint_state_serialization",
        "_ACTION_BALL_TASK_TRANSCRIPT_SCOPE_EXACT": "checkpoint_state_serialization",
        "_ACTION_BALL_TASK_TRANSCRIPT_SCOPE_DIAGNOSTIC": "checkpoint_state_serialization",
        "_ACTION_BALL_TASK_TRANSCRIPT_SCOPES": "checkpoint_state_serialization",
        "RacketTargetCommand._action_ball_task_transcript_scope": "checkpoint_state_serialization",
        # --- per-birth audit ledger
        "RacketTargetCommand._action_ball_provide_birth": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_assert_issued_birth": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_task_transcript_for_birth": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_emitted_task_count_for": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_birth_catalogs_are_live_only": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_expected_admitted_task_counts_live_only": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_online_solver_owns_admitted_task_counts": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_assert_emitted_sample": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_assert_emitted_tasks": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_assert_proposal_assignments": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_assert_proposal_assignments_against": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_decode_provider_births": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_decode_provider_history": "birth_audit_ledger",
        "RacketTargetCommand._action_ball_decode_task_transcripts": "birth_audit_ledger",
        # --- wiring
        "RacketTargetCommand._initialize_action_ball_runtime": "runtime_wiring",
        "_ActionBallPoolSolverAdapter.__init__": "runtime_wiring",
        "RacketTargetCommand._action_ball_refill_pool": "runtime_wiring",
        # --- telemetry
        "RacketTargetCommand._action_ball_note": "telemetry_and_counters",
        "RacketTargetCommand._action_ball_ledger_payload": "telemetry_and_counters",
        "RacketTargetCommand._action_ball_host_proposal_rows": "telemetry_and_counters",
        "RacketTargetCommand._action_ball_diagnostic_host_packet": "telemetry_and_counters",
        "_action_ball_diagnostic_host_packet": "telemetry_and_counters",
        "_action_ball_host_bool_packet": "telemetry_and_counters",
        "_action_ball_pack_diagnostic_install_rows": "telemetry_and_counters",
        "_action_ball_validate_tensor_predicate": "telemetry_and_counters",
        # --- grading / observation
        "RacketTargetCommand._action_ball_exact_achieved_contact_state": "grading_and_observation",
        "RacketTargetCommand._action_ball_cache_reference_host_rows": "grading_and_observation",
        # --- question production (sampling side), receipts
        "_action_ball_frozen_eval_receipt": "question_production_sampling",
        "_action_ball_assert_frozen_eval_receipt": "question_production_sampling",
        "RacketTargetCommand._action_ball_parse_sampler_birth": "question_production_sampling",
        # --- misc plumbing reached by bare-name over-approximation
        "_action_ball_sha256_file": "telemetry_and_counters",
        "_action_ball_strict_json_bytes": "telemetry_and_counters",
        "RacketTargetCommand.cfg": "overapproximated_name_collision",
        "RacketTargetCommand.command": "overapproximated_name_collision",
        "RacketTargetCommand.__init__": "overapproximated_name_collision",
        "_ActionBallBirthProviderAdapter.__init__": "overapproximated_name_collision",
        "_ActionBallDomainAuthorityAdapter.__init__": "overapproximated_name_collision",
        "_ActionBallDrainResetRuntimeSource.__init__": "overapproximated_name_collision",
    },
    "continuous_questions.py": {
        "generate": "other_product_line",
        "_uniform_box": "other_product_line",
        "parity_report": "self_check_only",
        "face_sign_negative_control": "self_check_only",
        "_empty_proposal_ledger": "other_product_line",
        "ball_birth_not_beyond_net": "convenience_accessor",
        "ProposalLedger.__len__": "convenience_accessor",
        "ProposalHostPacket.__len__": "convenience_accessor",
    },
    "racket_contact_geometry.py": {
        "torch_swept_selected_face_contact": "swept_contact_grading",
        "face_center_velocity_from_site": "swept_contact_grading",
        "legacy_colocation_error_m": "grading_and_observation",
        "polar_interpolate_quat_wxyz": "swept_contact_grading",
        "polar_interpolate_rotation_matrix": "swept_contact_grading",
        "quat_to_rotation_matrix_wxyz": "swept_contact_grading",
        "rotation_matrix_to_quat_wxyz": "swept_contact_grading",
        "_matrix3": "swept_contact_grading",
        "__all__": "module_export_list",
    },
    "stroke_adapt_torch.py": {
        "select_stroke_batch": "stroke_selector",
        "closing_speed_demand": "stroke_selector",
        "dir_deviation_deg": "stroke_selector",
        "direction_world": "stroke_selector",
        "to_b_yaw": "stroke_selector",
        "PREDICATES": "stroke_selector",
        "TABLE_SURFACE_Z_W_FLOOR": "stroke_selector",
    },
    "strike_spec_torch.py": {
        "_forward_landing": "other_product_line",
        "solve_strike_specs": "other_product_line",
    },
    "virtual_ball.py": {
        "load_venue_params": "venue_parameter_loading",
        "default_venue_yaml_path": "venue_parameter_loading",
        "classify_action_ball_contact": "grading_and_observation",
        "paddle_contact_state": "grading_and_observation",
        "signed_face_hemisphere": "grading_and_observation",
        "action_ball_sweep_identity_valid": "grading_and_observation",
        "finite_action_ball_rollout_inputs": "grading_and_observation",
        "_assert_tensor_validation": "grading_and_observation",
        "PADDLE_NORMAL_SPEED_MIN_MPS": "grading_and_observation",
        "PADDLE_NORMAL_SPEED_MAX_MPS": "grading_and_observation",
        "PADDLE_NORMAL_SPEED_PARITY_ABS_TOL_MPS": "grading_and_observation",
        "ACTION_BALL_CONTACT_REJECTION_COUNTERS": "grading_and_observation",
    },
}


class SolverSemanticSurfaceError(ValueError):
    """Raised when the declared surface cannot be built from the live sources."""


# --------------------------------------------------------------------------- #
# Symbol extraction and digesting                                              #
# --------------------------------------------------------------------------- #
_SKIPPED_AST_FIELDS = frozenset({"ctx", "type_comment", "type_params"})
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _strip_docstring(body: Sequence[ast.stmt]) -> list:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return list(body[1:])
    return list(body)


def _normalized_ast_text(node: Any) -> str:
    """Version-stable textual normal form of one AST node.

    ``ast.dump`` is deliberately avoided: its field set moved between 3.8 and
    3.12 (``Index`` wrappers disappeared in 3.9, ``type_params`` arrived in
    3.12), so the same source would digest differently depending on which
    interpreter re-pinned it.  Here the field list is walked explicitly, three
    known-unstable/redundant fields are dropped, and the 3.8 ``Index``/
    ``ExtSlice`` wrappers are unwrapped.
    """

    if isinstance(node, ast.AST):
        # 3.8 wraps every subscript in Index(); 3.9+ does not.
        index_type = getattr(ast, "Index", None)
        if index_type is not None and isinstance(node, index_type):
            return _normalized_ast_text(node.value)
        ext_slice_type = getattr(ast, "ExtSlice", None)
        if ext_slice_type is not None and isinstance(node, ext_slice_type):
            return "Tuple(elts=[%s])" % ",".join(
                _normalized_ast_text(item) for item in node.dims
            )
        parts = []
        for field in node._fields:
            if field in _SKIPPED_AST_FIELDS:
                continue
            value = getattr(node, field, None)
            if field == "body" and isinstance(
                node, (ast.Module, ast.ClassDef) + _FUNCTION_NODES
            ):
                value = _strip_docstring(value or [])
            parts.append("%s=%s" % (field, _normalized_ast_text(value)))
        return "%s(%s)" % (type(node).__name__, ",".join(parts))
    if isinstance(node, list):
        return "[%s]" % ",".join(_normalized_ast_text(item) for item in node)
    if isinstance(node, bytes):
        return "b" + repr(node.decode("latin-1"))
    if isinstance(node, str):
        return repr(node)
    if node is Ellipsis:
        return "Ellipsis"
    return repr(node)


def _symbol_digest(node: Any) -> str:
    return hashlib.sha256(
        _normalized_ast_text(node).encode("utf-8")
    ).hexdigest()


def _class_shell(node: ast.ClassDef) -> ast.ClassDef:
    """A class node with its methods and annotated fields removed.

    Methods and fields are digested as their own symbols, so a class-level
    digest that still contained them would make every method edit look like a
    change to the class itself.
    """

    shell = ast.ClassDef(
        name=node.name,
        bases=node.bases,
        keywords=node.keywords,
        body=[
            statement
            for statement in _strip_docstring(node.body)
            if not isinstance(statement, _FUNCTION_NODES + (ast.Assign, ast.AnnAssign))
        ],
        decorator_list=node.decorator_list,
    )
    return shell


def symbol_digests(source: str, *, filename: str) -> "Dict[str, str]":
    """Digest every module-level and class-member symbol of one source file."""

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:  # pragma: no cover - defensive
        raise SolverSemanticSurfaceError(
            f"cannot parse pinned solver source {filename}: {error}"
        ) from error
    digests: Dict[str, str] = {}

    def record(name: str, node: Any) -> None:
        if name in digests:
            raise SolverSemanticSurfaceError(
                f"{filename} defines {name} more than once; the semantic surface "
                "cannot tell the two definitions apart"
            )
        digests[name] = _symbol_digest(node)

    for node in tree.body:
        if isinstance(node, _FUNCTION_NODES):
            record(node.name, node)
        elif isinstance(node, ast.ClassDef):
            record(node.name, _class_shell(node))
            for sub in node.body:
                if isinstance(sub, _FUNCTION_NODES):
                    record("%s.%s" % (node.name, sub.name), sub)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            record("%s.%s" % (node.name, target.id), sub)
                elif isinstance(sub, ast.AnnAssign) and isinstance(
                    sub.target, ast.Name
                ):
                    record("%s.%s" % (node.name, sub.target.id), sub)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    record(target.id, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            record(node.target.id, node)
    return digests


def _symbol_nodes(source: str, *, filename: str) -> "Dict[str, Any]":
    tree = ast.parse(source, filename=filename)
    nodes: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, _FUNCTION_NODES):
            nodes[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, _FUNCTION_NODES):
                    nodes["%s.%s" % (node.name, sub.name)] = sub
    return nodes


def _referenced_names(node: Any) -> "set":
    """Every bare name and attribute name mentioned inside one symbol.

    Deliberately over-approximate: ``self.foo()`` contributes ``foo`` without
    proving which class owns it.  Over-approximation is the safe direction for a
    coverage gate -- it can demand that an unrelated same-named symbol be
    classified, never the reverse.
    """

    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


# --------------------------------------------------------------------------- #
# The three fail-closed coverage gates                                         #
# --------------------------------------------------------------------------- #
def _declared_names(filename: str) -> "set":
    return set(COVERED.get(filename, ())) | set(EXCLUDED.get(filename, {}))


def _excluded_symbols_referenced_by_covered(
    digests: "Dict[str, Dict[str, str]]", nodes: "Dict[str, Dict[str, Any]]"
) -> "Dict[Tuple[str, str], Tuple[str, ...]]":
    """Which excluded symbols the covered closure actually touches, and from where.

    Same bare-name over-approximation as the closure gate.  The result is used two
    ways: gate 3 turns "this symbol is never called here" into an executable
    claim, and the declaration publishes the rest so that the remaining
    exclusions -- the ones that *are* reached but are argued not to move an
    answer -- are visible to a reader instead of buried in this file.
    """

    reached: Dict[Tuple[str, str], set] = {}
    for filename, covered in sorted(COVERED.items()):
        for name in covered:
            node = nodes.get(filename, {}).get(name)
            if node is None:
                continue
            for referenced in sorted(_referenced_names(node)):
                for other in PINNED_SOURCES:
                    excluded = EXCLUDED.get(other, {})
                    candidates = [referenced] if referenced in excluded else []
                    candidates.extend(
                        candidate
                        for candidate in digests.get(other, {})
                        if "." in candidate
                        and candidate.rpartition(".")[2] == referenced
                        and candidate in excluded
                    )
                    for candidate in candidates:
                        reached.setdefault((other, candidate), set()).add(
                            "%s:%s" % (filename, name)
                        )
    return {key: tuple(sorted(value)) for key, value in sorted(reached.items())}


def surface_blockers(read_source: Callable[[str], str]) -> Tuple[str, ...]:
    """List every way the declared surface fails to cover what it claims.

    ``read_source`` maps a pinned file name to its text.  Empty result means:

    * every covered symbol still exists in its file;
    * every reason code used by ``EXCLUDED`` is defined;
    * every symbol of the five fully enumerated solver sources is classified;
    * every name a covered symbol references that resolves into a pinned source
      is classified too;
    * every exclusion whose reason *claims* the symbol is off the solve path is
      in fact unreachable from the covered closure.

    Anything else is a blocker, and the caller must refuse to boot.  A blocker is
    never "just log it": a surface that silently covers less than it claims is
    exactly the hole that let a stale pin look authoritative.
    """

    blockers = []
    digests: Dict[str, Dict[str, str]] = {}
    nodes: Dict[str, Dict[str, Any]] = {}
    for filename in PINNED_SOURCES:
        try:
            source = read_source(filename)
        except Exception as error:  # noqa: BLE001 - any read failure fails closed
            blockers.append(f"pinned_source_unreadable:{filename}:{error}")
            continue
        try:
            digests[filename] = symbol_digests(source, filename=filename)
            nodes[filename] = _symbol_nodes(source, filename=filename)
        except SolverSemanticSurfaceError as error:
            blockers.append(f"pinned_source_unparsable:{filename}:{error}")
    if blockers:
        return tuple(blockers)

    for filename, reasons in sorted(EXCLUDED.items()):
        if filename not in PINNED_SOURCES:
            blockers.append(f"excluded_file_is_not_pinned:{filename}")
            continue
        for name, reason in sorted(reasons.items()):
            if reason not in EXCLUSION_REASONS:
                blockers.append(
                    f"exclusion_reason_undefined:{filename}:{name}:{reason}"
                )

    for filename, covered in sorted(COVERED.items()):
        if filename not in PINNED_SOURCES:
            blockers.append(f"covered_file_is_not_pinned:{filename}")
            continue
        overlap = sorted(set(covered) & set(EXCLUDED.get(filename, {})))
        for name in overlap:
            blockers.append(f"symbol_both_covered_and_excluded:{filename}:{name}")
        for name in covered:
            if name not in digests.get(filename, {}):
                blockers.append(f"covered_symbol_absent:{filename}:{name}")

    # Gate 1 -- full enumeration of the five pure solver sources.
    for filename in FULLY_ENUMERATED_SOURCES:
        declared = _declared_names(filename)
        for name in sorted(digests.get(filename, {})):
            if name not in declared:
                blockers.append(f"symbol_unclassified:{filename}:{name}")

    # Gate 2 -- reference closure out of every covered symbol.
    declared_by_file = {name: _declared_names(name) for name in PINNED_SOURCES}
    for filename, covered in sorted(COVERED.items()):
        for name in covered:
            node = nodes.get(filename, {}).get(name)
            if node is None:
                continue  # constants and dataclass fields have no call graph
            for referenced in sorted(_referenced_names(node)):
                for other in PINNED_SOURCES:
                    if referenced in declared_by_file[other]:
                        continue
                    other_digests = digests.get(other, {})
                    if referenced in other_digests:
                        blockers.append(
                            "referenced_symbol_unclassified:"
                            f"{other}:{referenced}:from:{filename}:{name}"
                        )
                    else:
                        qualified = [
                            candidate
                            for candidate in other_digests
                            if candidate.rpartition(".")[2] == referenced
                            and "." in candidate
                        ]
                        for candidate in qualified:
                            if candidate not in declared_by_file[other]:
                                blockers.append(
                                    "referenced_symbol_unclassified:"
                                    f"{other}:{candidate}:from:{filename}:{name}"
                                )

    # Gate 3 -- an exclusion reason that claims "action-ball never calls it" has
    # to be true.  Those reasons were written as prose and never checked; the
    # ``fixed_direction`` accident is what showed that a sentence in this file
    # is not evidence.  Reachability is, so it is computed.
    reached = _excluded_symbols_referenced_by_covered(digests, nodes)
    for (filename, name), callers in sorted(reached.items()):
        reason = EXCLUDED[filename][name]
        if reason in UNREACHABLE_CLAIM_REASONS:
            blockers.append(
                "exclusion_claims_unreachable_but_is_reached:"
                f"{filename}:{name}:{reason}:from:{','.join(callers)}"
            )
    return tuple(sorted(set(blockers)))


# --------------------------------------------------------------------------- #
# The sealed payload and the self-describing declaration                       #
# --------------------------------------------------------------------------- #
def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def semantic_surface_contract(read_source: Callable[[str], str]) -> dict:
    """Build the sealed per-symbol surface payload, or refuse.

    The payload carries the covered symbol digests and the coverage policy.  It
    deliberately does **not** carry the excluded name list: adding a new
    serialization/ledger/telemetry symbol must be a conscious classification
    (the gates enforce that) but must not invalidate an in-flight run's pin.
    Moving a symbol the other way -- out of ``COVERED`` -- always moves the pin,
    because its digest disappears from the payload.
    """

    blockers = surface_blockers(read_source)
    if blockers:
        raise SolverSemanticSurfaceError(
            "action-ball solver semantic surface does not cover what it claims:\n  "
            + "\n  ".join(blockers)
        )
    covered_digests: Dict[str, Dict[str, str]] = {}
    for filename, covered in COVERED.items():
        digests = symbol_digests(read_source(filename), filename=filename)
        covered_digests[filename] = {name: digests[name] for name in covered}
    payload = {
        "schema_version": SEMANTIC_SURFACE_SCHEMA_VERSION,
        "kind": SEMANTIC_SURFACE_KIND,
        "symbol_digest_algorithm": SYMBOL_DIGEST_ALGORITHM,
        "entry_points": list(SEMANTIC_ENTRY_POINTS),
        "coverage_policy": {
            "fully_enumerated_sources": list(FULLY_ENUMERATED_SOURCES),
            "pinned_sources": list(PINNED_SOURCES),
            "unclassified_symbol_is_fail_closed": True,
            "reference_closure": (
                "every_name_referenced_by_a_covered_symbol_that_resolves_into_a_"
                "pinned_source_must_be_declared_covered_or_excluded"
            ),
            "exclusions_are_declared_by_name_and_reason_outside_this_digest": True,
            "unreachability_claiming_exclusion_reasons": sorted(
                UNREACHABLE_CLAIM_REASONS
            ),
            "declared_numbers_are_cross_checked_against_the_live_solver": (
                "hope_commands.action_ball_assert_solver_runtime_matches_"
                "declaration, called from every entry point above"
            ),
        },
        "covered": {
            filename: dict(sorted(symbols.items()))
            for filename, symbols in sorted(covered_digests.items())
        },
    }
    return {"payload": payload, "sha256": _canonical_sha256(payload)}


def semantic_surface_declaration(read_source: Callable[[str], str]) -> dict:
    """The receipt half: what was covered, what was excluded, and why.

    This is what makes the pin self-describing.  It is emitted into the offline
    profile-pins document and into the boot telemetry so that "which symbols does
    this pin actually protect" is answerable without reading this module.
    """

    contract = semantic_surface_contract(read_source)
    used_reasons = sorted(
        {reason for reasons in EXCLUDED.values() for reason in reasons.values()}
    )
    nodes = {
        filename: _symbol_nodes(read_source(filename), filename=filename)
        for filename in PINNED_SOURCES
    }
    digests = {
        filename: symbol_digests(read_source(filename), filename=filename)
        for filename in PINNED_SOURCES
    }
    reached = _excluded_symbols_referenced_by_covered(digests, nodes)
    return {
        "schema_version": SEMANTIC_SURFACE_SCHEMA_VERSION,
        "kind": SEMANTIC_SURFACE_KIND + ".declaration",
        "sha256": contract["sha256"],
        "covered_symbol_count": sum(
            len(symbols) for symbols in contract["payload"]["covered"].values()
        ),
        "covered": {
            filename: sorted(symbols)
            for filename, symbols in contract["payload"]["covered"].items()
        },
        "excluded": {
            filename: dict(sorted(reasons.items()))
            for filename, reasons in sorted(EXCLUDED.items())
        },
        "exclusion_reasons": {
            reason: EXCLUSION_REASONS[reason] for reason in used_reasons
        },
        # The audit half nobody could compute from the lists above: which of the
        # excluded symbols the covered closure actually touches.  Everything here
        # is an exclusion whose reason has to mean "reached, but cannot move an
        # answer" -- the stronger claim.  Everything NOT here is unreachable from
        # the closure, and for the reason codes in
        # ``unreachability_claiming_exclusion_reasons`` that is enforced.
        "excluded_but_reached_from_covered": {
            "%s:%s" % (filename, name): {
                "reason": EXCLUDED[filename][name],
                "referenced_from": list(callers),
            }
            for (filename, name), callers in reached.items()
        },
        "unreachability_claiming_exclusion_reasons": sorted(
            UNREACHABLE_CLAIM_REASONS
        ),
        "entry_points": list(SEMANTIC_ENTRY_POINTS),
    }


def source_reader_for_directory(module_dir) -> Callable[[str], str]:
    """Read pinned sources from a directory (the live runtime case)."""

    from pathlib import Path

    root = Path(module_dir)

    def read(filename: str) -> str:
        return (root / filename).read_text(encoding="utf-8")

    return read
