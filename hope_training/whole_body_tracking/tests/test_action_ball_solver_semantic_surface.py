"""Mutation tests for the per-symbol action-ball solver semantic surface.

人话:这份测试回答三个问题,一个都不能少。

1. **等强**:凡是"会改题、会改答案"的改动,新 pin 必须仍然拒绝。
   下面 ``EQUISTRENGTH_MUTANTS`` 每一条都是第一步调研列出的"必须仍能抓到"的
   一类,包括上一轮实测过的 A3/A4/A5/A6 以及 min→-1000。每一条都先断言
   "这段文本确实在源码里出现过",再断言指纹变了 —— 上一轮 A7 就是 sed 没匹配上
   而被当成"存活",这里不许再发生。

2. **收窄有效**:纯注释、纯 docstring、以及真实的两笔 checkpoint 序列化重构
   (eccb30cd / 308db7f0)和一笔纯注释提交(3e64bea9)必须**放行** —— 这正是
   本轮的目的。

3. **不许自我豁免**:往语义面里加一个新符号却不更新覆盖清单,必须被抓到。
   这对应"选择器覆盖面小于它声称保护的语义面"那个已经咬过一次的洞。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


WHOLE_BODY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WHOLE_BODY_ROOT.parents[1]
MDP_DIR = (
    WHOLE_BODY_ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
)
SURFACE_SOURCE = MDP_DIR / "action_ball_solver_semantic_surface.py"


def _load_surface_module():
    """Host-load the stdlib-only surface module, exactly like the pinner does."""

    spec = importlib.util.spec_from_file_location(
        "_test_action_ball_solver_semantic_surface", SURFACE_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SURFACE = _load_surface_module()


def _live_reader():
    return SURFACE.source_reader_for_directory(MDP_DIR)


def _live_surface_sha256() -> str:
    return SURFACE.semantic_surface_contract(_live_reader())["sha256"]


def _covered_digests(reader):
    """The covered symbol digests alone, with the coverage gates NOT run.

    Used to show what a digest can and cannot see: an edit that rebinds which
    method runs leaves every one of these byte-identical, which is exactly why
    a gate outside the digest has to exist.
    """

    return {
        filename: {
            name: SURFACE.symbol_digests(
                reader(filename), filename=filename
            )[name]
            for name in covered
        }
        for filename, covered in SURFACE.COVERED.items()
    }


def _mutated_reader(replacements):
    """A reader that serves the live sources with textual replacements applied.

    Every replacement must actually match, otherwise the "mutant survived"
    verdict would be measuring nothing.  That failure mode has already bitten
    once (a sed pattern that never matched was recorded as a surviving mutant),
    so a missing needle is a hard test failure, not a skip.
    """

    live = _live_reader()
    cache = {}
    for filename, old, new in replacements:
        text = cache.get(filename, live(filename))
        if old not in text:
            raise AssertionError(
                f"mutation target not present in {filename}: {old!r}"
            )
        cache[filename] = text.replace(old, new)

    def read(name: str) -> str:
        return cache.get(name, live(name))

    return read


# --------------------------------------------------------------------------- #
# 1. Equistrength: every change that moves a question or an answer must move    #
#    the pin.                                                                   #
# --------------------------------------------------------------------------- #
EQUISTRENGTH_MUTANTS = (
    (
        "contact_normal_speed_min_widened",
        [
            (
                "continuous_questions.py",
                "CONTACT_NORMAL_SPEED_MIN_MPS = 1.4",
                "CONTACT_NORMAL_SPEED_MIN_MPS = -1000.0",
            )
        ],
    ),
    (
        "contact_normal_speed_max_widened",
        [
            (
                "continuous_questions.py",
                "CONTACT_NORMAL_SPEED_MAX_MPS = 7.2",
                "CONTACT_NORMAL_SPEED_MAX_MPS = 40.0",
            )
        ],
    ),
    (
        "ball_birth_net_margin_sign_flipped",
        [
            (
                "continuous_questions.py",
                "BALL_BIRTH_NET_MARGIN_M = 0.05",
                "BALL_BIRTH_NET_MARGIN_M = -0.05",
            )
        ],
    ),
    (
        "rejection_code_priority_reordered",
        [
            (
                "continuous_questions.py",
                "_R_NO_LANDING = 0\n_R_RESID = 1",
                "_R_NO_LANDING = 1\n_R_RESID = 0",
            )
        ],
    ),
    (
        "lm_rejection_codes_reordered",
        [
            (
                "continuous_questions.py",
                "_R_LM_SOLVE_INFO = 8\n_R_LM_SOLVE_NONFINITE = 9",
                "_R_LM_SOLVE_INFO = 9\n_R_LM_SOLVE_NONFINITE = 8",
            )
        ],
    ),
    (
        "lm_rejection_reason_missing",
        [
            (
                "continuous_questions.py",
                '    "lm_solve_info_nonzero",\n'
                '    "lm_solve_nonfinite",',
                '    "lm_solve_info_nonzero",',
            )
        ],
    ),
    (
        "lm_rejection_reason_names_reordered",
        [
            (
                "continuous_questions.py",
                '    "lm_solve_info_nonzero",\n'
                '    "lm_solve_nonfinite",',
                '    "lm_solve_nonfinite",\n'
                '    "lm_solve_info_nonzero",',
            )
        ],
    ),
    (
        "legacy_runtime_reason_prefix_drops_lm_rejections",
        [
            (
                "hope_commands.py",
                '        "lm_solve_info_nonzero",\n'
                '        "lm_solve_nonfinite",\n'
                '        "teacher_site_rate_geometry_unsolved",',
                '        "teacher_site_rate_geometry_unsolved",',
            )
        ],
    ),
    (
        "fixed_direction_admission_conjunct_dropped",
        [
            (
                "continuous_questions.py",
                "        & net_ok\n        & face_ok\n        & contact_ok",
                "        & net_ok\n        & face_ok",
            )
        ],
    ),
    (
        "ball_birth_lower_bound_direction_flipped",
        [
            (
                "continuous_questions.py",
                "        float(contact_x_w_m)\n"
                "        + abs(float(incoming_velocity_x_w_mps))"
                " * float(time_to_contact_s)",
                "        float(contact_x_w_m)\n"
                "        - abs(float(incoming_velocity_x_w_mps))"
                " * float(time_to_contact_s)",
            )
        ],
    ),
    (
        "strike_spec_seed_mirror_law_e0",
        [
            (
                "strike_spec_torch.py",
                "e = torch.full_like(v_o_n, 0.5)",
                "e = torch.full_like(v_o_n, 0.93)",
            )
        ],
    ),
    (
        "strike_spec_seed_fixed_point_iterations",
        [
            (
                "strike_spec_torch.py",
                "    for _ in range(3):",
                "    for _ in range(1):",
            )
        ],
    ),
    (
        "strike_spec_face_basis_handedness",
        [
            (
                "strike_spec_torch.py",
                "    b2 = torch.cross(n, b1, dim=-1)",
                "    b2 = torch.cross(b1, n, dim=-1)",
            )
        ],
    ),
    (
        "virtual_ball_gravity",
        [
            (
                "virtual_ball.py",
                "    a[..., 2] -= prm.g",
                "    a[..., 2] -= prm.g * 1.05",
            )
        ],
    ),
    (
        "virtual_ball_rk4_weights",
        [
            (
                "virtual_ball.py",
                "    v_new = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)",
                "    v_new = v + (h / 6.0) * (a1 + 2 * a2 + 3 * a3 + a4)",
            )
        ],
    ),
    (
        "teacher_rate_boundary_tolerance",
        [
            (
                "racket_contact_geometry.py",
                "TEACHER_RATE_BOUNDARY_ABS_TOL = 5.0e-7",
                "TEACHER_RATE_BOUNDARY_ABS_TOL = 0.5",
            )
        ],
    ),
    (
        "exact_face_contact_quadratic_root_branch",
        [
            (
                "racket_contact_geometry.py",
                "        q = -0.5 * (b + math.copysign(sqrt_disc, b))",
                "        q = -0.5 * (b - math.copysign(sqrt_disc, b))",
            )
        ],
    ),
    (
        "fixed_direction_lm_damping",
        [
            (
                "stroke_adapt_torch.py",
                "lam = torch.full((N,), 1e-3, device=dev, dtype=dt)",
                "lam = torch.full((N,), 1e-1, device=dev, dtype=dt)",
            )
        ],
    ),
    (
        "solver_field_contract_column_order",
        [
            (
                "hope_commands.py",
                '                    ("incoming_velocity_w_mps", 3),\n'
                '                    ("spin_w_radps", 3),',
                '                    ("spin_w_radps", 3),\n'
                '                    ("incoming_velocity_w_mps", 3),',
            )
        ],
    ),
    (
        "pre_swing_wait_upper_bound_literal",
        [
            (
                "hope_commands.py",
                "                        <= pre_swing_wait_s\n"
                "                        <= 1.0",
                "                        <= pre_swing_wait_s\n"
                "                        <= 2.0",
            )
        ],
    ),
    (
        "cycle_versus_horizon_epsilon",
        [
            (
                "hope_commands.py",
                "> self._action_ball_episode_length_s + 1.0e-12",
                "> self._action_ball_episode_length_s + 1.0",
            )
        ],
    ),
    (
        "question_identity_field_dropped",
        [
            (
                "hope_commands.py",
                '        "time_to_contact_s": float(sample.time_to_contact_s),\n'
                '        "incoming_velocity_w_mps": list('
                "sample.incoming_velocity_w_mps),",
                '        "incoming_velocity_w_mps": list('
                "sample.incoming_velocity_w_mps),",
            )
        ],
    ),
    # --- The declaration/actual bridge.  Before this batch each of the next
    # nine edits changed the numbers the solver is handed (or disarmed the gate
    # that checks them) while leaving the pin at c196cf79: the mapping lived
    # inside ``_initialize_action_ball_runtime``, which the surface excludes.
    (
        "solver_cfg_tolerance_halved_at_the_mapping",
        [
            (
                "hope_commands.py",
                '        tol_m=knobs["tol_m"],',
                '        tol_m=knobs["tol_m"] * 0.5,',
            )
        ],
    ),
    (
        "solver_cfg_speed_budget_doubled_at_the_mapping",
        [
            (
                "hope_commands.py",
                '        speed_budget=knobs["global_speed_budget_mps"],',
                '        speed_budget=knobs["global_speed_budget_mps"] * 2.0,',
            )
        ],
    ),
    (
        "declared_knob_n_iters_bumped",
        [
            (
                "hope_commands.py",
                '        "n_iters": int(cfg.cq_n_iters),',
                '        "n_iters": int(cfg.cq_n_iters) + 5,',
            )
        ],
    ),
    (
        "fixed_direction_flag_flipped",
        [
            (
                "hope_commands.py",
                "_ACTION_BALL_SOLVER_FIXED_DIRECTION = True",
                "_ACTION_BALL_SOLVER_FIXED_DIRECTION = False",
            )
        ],
    ),
    (
        "declaration_cross_check_comparison_disarmed",
        [
            (
                "hope_commands.py",
                "        if type(declared) is not type(actual) or declared != actual",
                "        if False",
            )
        ],
    ),
    (
        "declaration_cross_check_stops_comparing_the_redraw_budget",
        [
            (
                "hope_commands.py",
                "            overdraw=effective_cq_overdraw,\n"
                "            maximum_rounds=maximum_rounds,\n"
                "            diagnostic_unauthorized=diagnostic_unauthorized,\n"
                '            call_site="_action_ball_refill_pool_many",',
                "            overdraw=None,\n"
                "            maximum_rounds=None,\n"
                "            diagnostic_unauthorized=diagnostic_unauthorized,\n"
                '            call_site="_action_ball_refill_pool_many",',
            )
        ],
    ),
    (
        "declaration_cross_check_call_deleted_from_the_frozen_evaluator",
        [
            (
                "hope_commands.py",
                "        action_ball_assert_solver_runtime_matches_declaration(\n"
                "            solver_declaration=self._action_ball_solver_contract"
                '["payload"],\n'
                "            physics_declaration=self._action_ball_physics_contract"
                '["payload"],\n'
                "            answer_input_declaration=(\n"
                "                self._action_ball_answer_input_contract"
                '["payload"]\n'
                "            ),\n"
                "            expected_solver_profile_sha256=(\n"
                "                self._action_ball_manifest."
                "solver_profile_sha256\n"
                "            ),\n"
                "            expected_prototype_file_sha256=(\n"
                "                self._action_ball_manifest.prototype.sha256\n"
                "            ),\n"
                "            prototypes=self._action_ball_prototypes,\n"
                "            reference_normal_rows=self._ref_racket_normal_raw_"
                "w_per_clip,\n"
                "            solver_cfg=self._action_ball_solver_cfg,\n"
                "            prm=self._action_ball_prm,\n"
                "            planes=(surface_z, net_x, net_top_z),\n"
                "            rollout_h=rollout_h,\n"
                "            rollout_steps=rollout_steps,\n"
                "            overdraw=None,\n"
                "            maximum_rounds=None,\n"
                "            diagnostic_unauthorized=bool(\n"
                '                getattr(self, "_action_ball_diagnostic_unauthorized"'
                ", False)\n"
                "            ),\n"
                '            call_site="_action_ball_frozen_eval_solve",\n'
                "        )\n",
                "",
            )
        ],
    ),
    (
        "cross_checked_virtual_ball_parameter_dropped",
        [
            (
                "hope_commands.py",
                '    "paddle_mu",\n    "paddle_e_g1",',
                '    "paddle_e_g1",',
            )
        ],
    ),
    (
        "diagnostic_redraw_round_exemption_widened",
        [
            (
                "hope_commands.py",
                "_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS = 64",
                "_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS = 6400",
            )
        ],
    ),
    (
        "c211_emitted_task_reference_predicate",
        [
            (
                "hope_commands.py",
                "                or receipt.geometry_source_sha256\n"
                "                != contact_geometry.GEOMETRY_SOURCE_SHA256",
                "                or receipt.geometry_source_sha256\n"
                "                == contact_geometry.GEOMETRY_SOURCE_SHA256",
            )
        ],
    ),
    (
        "solver_profile_schema_version",
        [
            (
                "hope_commands.py",
                "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION = 3",
                "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION = 4",
            )
        ],
    ),
    # --- The second batch of escapes: the two solver ARGUMENTS, and the gate's
    # own identity.  Each of these edits used to leave the pin at 5fb9e472.
    (
        "answer_input_digest_stops_covering_a_prototype_column",
        [
            (
                "hope_commands.py",
                '    "speed_min",\n    "speed_max",\n    "v_star_cap",',
                '    "speed_min",\n    "v_star_cap",',
            )
        ],
    ),
    (
        "answer_input_digest_stops_covering_the_reference_normal_rows",
        [
            (
                "hope_commands.py",
                "    columns.append(reference_normal_rows.reshape(-1)"
                ".to(torch.float64))",
                "    columns.append(reference_normal_rows.reshape(-1)"
                ".to(torch.float64) * 0.0)",
            )
        ],
    ),
    (
        "answer_input_contract_stops_anchoring_face_sign_to_the_manifest",
        [
            (
                "hope_commands.py",
                "    if live_signs != expected_signs:",
                "    if False:",
            )
        ],
    ),
    (
        "declaration_cross_check_stops_recomputing_the_sealed_payload_sha",
        [
            (
                "hope_commands.py",
                '        ("solver.payload.canonical_sha256",\n'
                "         str(expected_solver_profile_sha256),\n"
                "         _action_ball_canonical_sha256(solver_declaration)),",
                '        ("solver.payload.canonical_sha256",\n'
                "         str(expected_solver_profile_sha256),\n"
                "         str(expected_solver_profile_sha256)),",
            )
        ],
    ),
    (
        "declaration_cross_check_stops_comparing_the_live_answer_inputs",
        [
            (
                "hope_commands.py",
                '        ("answer_inputs.live_digest_sha256",\n'
                '         str(answer_input_declaration["live_digest_sha256"]),\n'
                '         str(live_answer_inputs["sha256"])),',
                "",
            )
        ],
    ),
    (
        "adapter_identity_attestation_disarmed",
        [
            (
                "hope_commands.py",
                "    if drift:\n        raise RuntimeError(\n"
                '            "action-ball pool solver adapter does not hold the '
                'covered entry "',
                "    if False:\n        raise RuntimeError(\n"
                '            "action-ball pool solver adapter does not hold the '
                'covered entry "',
            )
        ],
    ),
    (
        "adapter_identity_attestation_call_deleted_from_the_refill_entry_point",
        [
            (
                "hope_commands.py",
                "        action_ball_assert_solver_adapter_binds_these_entry_"
                "points(\n"
                "            adapter=self._action_ball_pool_solver,\n"
                "            expected={\n"
                '                "solve": self._action_ball_refill_pool,\n'
                '                "solve_many": self._action_ball_refill_pool_'
                "many,\n"
                '                "assert_emitted_sample": (\n'
                "                    self._action_ball_assert_emitted_sample\n"
                "                ),\n"
                '                "assert_emitted_tasks": self._action_ball_'
                "assert_emitted_tasks,\n"
                '                "emitted_task_count_for": (\n'
                "                    self._action_ball_emitted_task_count_for\n"
                "                ),\n"
                '                "task_transcript_for_birth": (\n'
                "                    self._action_ball_task_transcript_for_"
                "birth\n"
                "                ),\n"
                '                "assert_proposal_assignments": (\n'
                "                    self._action_ball_assert_proposal_"
                "assignments\n"
                "                ),\n"
                "            },\n"
                '            call_site="_action_ball_refill_pool_many",\n'
                "            required=True,\n"
                "        )\n",
                "",
            )
        ],
    ),
)


@pytest.mark.parametrize(
    "label,replacements",
    EQUISTRENGTH_MUTANTS,
    ids=[label for label, _ in EQUISTRENGTH_MUTANTS],
)
def test_semantic_surface_still_rejects_every_question_changing_edit(
    label, replacements
):
    baseline = _live_surface_sha256()
    mutated = SURFACE.semantic_surface_contract(_mutated_reader(replacements))
    assert mutated["sha256"] != baseline, (
        f"mutant {label} survived: the narrowed solver pin no longer notices a "
        "change that moves the questions or the answers"
    )


# --------------------------------------------------------------------------- #
# 2. Narrowing works: the edits this change exists to stop breaking runs.      #
# --------------------------------------------------------------------------- #
NARROWING_CONTROLS = (
    (
        "comment_only_in_every_pinned_source",
        [
            (name, "from __future__ import annotations",
             "from __future__ import annotations\n# mutated: a comment")
            for name in SURFACE.PINNED_SOURCES
        ],
    ),
    (
        "docstring_only_in_a_covered_function",
        [
            (
                "continuous_questions.py",
                '    """Solve exact external proposals without observing a tensor on the host.',
                '    """MUTATED DOCSTRING that says something entirely different'
                " about this function.",
            )
        ],
    ),
    (
        "blank_lines_and_reflow_are_invisible",
        [
            (
                "virtual_ball.py",
                "    speed = torch.linalg.norm(v, dim=-1, keepdim=True)",
                "\n\n    speed = torch.linalg.norm(\n        v, dim=-1, keepdim=True\n    )",
            )
        ],
    ),
)


@pytest.mark.parametrize(
    "label,replacements",
    NARROWING_CONTROLS,
    ids=[label for label, _ in NARROWING_CONTROLS],
)
def test_semantic_surface_ignores_non_semantic_edits(label, replacements):
    baseline = _live_surface_sha256()
    mutated = SURFACE.semantic_surface_contract(_mutated_reader(replacements))
    assert mutated["sha256"] == baseline, (
        f"control {label} moved the pin: the narrowing did not actually narrow"
    )


def test_new_unreferenced_symbol_in_hope_commands_does_not_move_the_pin():
    """A new bookkeeping helper nobody covered calls must not invalidate a run.

    This is the shape of the two serialization-scope commits: they added
    ``_action_ball_*`` helpers that only the checkpoint/ledger code calls.
    ``hope_commands.py`` is not fully enumerated, so an unreferenced new symbol
    is allowed through -- and, because it is unreferenced, no covered symbol
    changed either.
    """

    baseline = _live_surface_sha256()
    reader = _mutated_reader(
        [
            (
                "hope_commands.py",
                "def _action_ball_canonical_sha256(",
                "def _action_ball_brand_new_ledger_helper(rows):\n"
                "    return len(rows)\n\n\n"
                "def _action_ball_canonical_sha256(",
            )
        ]
    )
    assert SURFACE.surface_blockers(reader) == ()
    assert SURFACE.semantic_surface_contract(reader)["sha256"] == baseline


# The real branch history this narrowing exists for.  Under the old whole-file
# SHA each of these minted a different solver profile and a running curriculum
# refused to boot against its own manifest; none of them touched a question.
REAL_NARROWING_COMMITS = (
    ("423f5409", "parent: the revision the live manifest was pinned against"),
    ("eccb30cd", "checkpoint serialization scope"),
    ("308db7f0", "checkpoint serialization scope"),
    ("3e64bea9", "comment-only"),
)


def _historical_reader(revision):
    sources = {}
    for name in SURFACE.PINNED_SOURCES:
        relative = (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/" + name
        )
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{revision}:{relative}"],
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        sources[name] = result.stdout.decode("utf-8")

    def read(name: str) -> str:
        return sources[name]

    return read


#: Covered symbols added after the four historical narrowing revisions.  They
#: do not exist at those revisions, so the historical comparison is evaluated
#: with them removed -- explicitly, by name, from a list this file owns, so
#: that "the historical test quietly stopped covering something" cannot happen
#: without editing it.  This is only a historical-source evolution map: every
#: listed name must still be covered by today's live semantic surface.
#:
#: ``_ACTION_BALL_DIAGNOSTIC_MAX_EXTERNAL_PROPOSAL_ROUNDS`` is deliberately NOT
#: here: this batch only added it to the coverage list, the constant itself has
#: existed all along, so it can and should still be compared across revisions.
COVERED_SYMBOLS_ADDED_AFTER_HISTORICAL_REVISIONS = {
    "continuous_questions.py": (
        "_R_LM_SOLVE_INFO",
        "_R_LM_SOLVE_NONFINITE",
        "DeviceProposalSolveResult",
        "DeviceProposalSolveResult.p_contact",
        "DeviceProposalSolveResult.v_racket",
        "DeviceProposalSolveResult.n_racket",
        "DeviceProposalSolveResult.v_ball_in",
        "DeviceProposalSolveResult.w_ball_in",
        "DeviceProposalSolveResult.aim_xy",
        "DeviceProposalSolveResult.ok",
        "DeviceProposalSolveResult.resid_m",
        "DeviceProposalSolveResult.attempted_v_ball_in",
        "DeviceProposalSolveResult.producer_fault_bits",
        "DeviceProposalSolveResult.proposals",
        "PRODUCER_FAULT_NONFINITE_PROPOSAL",
        "PRODUCER_FAULT_REFERENCE_NORMAL",
        "PRODUCER_FAULT_BASE_QUATERNION",
        "PRODUCER_FAULT_ACTION_RANGE",
        "PRODUCER_FAULT_PROTOTYPE_DIRECTION",
        "PRODUCER_FAULT_PROTOTYPE_SPEED",
        "PRODUCER_FAULT_PROTOTYPE_FACE_SIGN",
        "PRODUCER_FAULT_MASK",
        "solve_proposals_device",
    ),
    "hope_commands.py": (
        "action_ball_declared_solver_knobs",
        "action_ball_solver_cfg_from_declaration",
        "action_ball_assert_solver_runtime_matches_declaration",
        "_ACTION_BALL_SOLVER_FIXED_DIRECTION",
        "_ACTION_BALL_VIRTUAL_BALL_PARAM_NAMES",
        # The second batch: the two solver ARGUMENTS no payload declared
        # (``protos`` / ``ref_normal``) and the adapter-identity attestation.
        "action_ball_live_answer_input_digest",
        "action_ball_answer_input_contract",
        "_ACTION_BALL_ANSWER_INPUT_SCHEMA_VERSION",
        "_ACTION_BALL_ANSWER_INPUT_PROTOTYPE_COLUMNS",
        "action_ball_assert_solver_adapter_binds_these_entry_points",
        "_ActionBallPoolSolverAdapter.action_ball_bound_entry_points",
    )
}


def test_real_commits_that_invalidated_the_whole_file_pin_are_now_transparent(
    monkeypatch,
):
    """423f5409 -> eccb30cd -> 308db7f0 -> 3e64bea9 must all be the same pin.

    This is the ground truth for "was the narrowing real".  It is evaluated with
    today's surface module against those revisions' sources, so it also proves
    the covered symbol list is not tied to one snapshot of the tree.

    The surface has since grown a declaration/actual bridge whose symbols did
    not exist at those revisions, so they are removed for this comparison --
    explicitly, by name, from a list this file owns.  Restricting silently (to
    "whatever happens to exist at every revision") would let a future deletion
    shrink the historical claim without anyone noticing.
    """

    restricted = {}
    for filename, covered in SURFACE.COVERED.items():
        added = COVERED_SYMBOLS_ADDED_AFTER_HISTORICAL_REVISIONS.get(
            filename, ()
        )
        for name in added:
            assert name in covered, (filename, name)
        restricted[filename] = tuple(
            name for name in covered if name not in added
        )
    monkeypatch.setattr(SURFACE, "COVERED", restricted)

    digests = {}
    for revision, description in REAL_NARROWING_COMMITS:
        reader = _historical_reader(revision)
        if reader is None:
            pytest.skip(f"revision {revision} is not reachable in this checkout")
        blockers = SURFACE.surface_blockers(reader)
        assert blockers == (), (revision, description, blockers)
        digests[revision] = SURFACE.semantic_surface_contract(reader)["sha256"]

    distinct = set(digests.values())
    assert len(distinct) == 1, (
        "the narrowing failed: these commits changed no question but still "
        f"move the solver pin: {digests}"
    )


# --------------------------------------------------------------------------- #
# 3. No self-exemption: coverage may never silently shrink below the surface.  #
# --------------------------------------------------------------------------- #
def test_live_checkout_has_no_coverage_blockers():
    assert SURFACE.surface_blockers(_live_reader()) == ()


def test_device_proposal_boundary_is_covered_not_excluded():
    required = {
        "DeviceProposalSolveResult",
        "DeviceProposalSolveResult.p_contact",
        "DeviceProposalSolveResult.v_racket",
        "DeviceProposalSolveResult.n_racket",
        "DeviceProposalSolveResult.v_ball_in",
        "DeviceProposalSolveResult.w_ball_in",
        "DeviceProposalSolveResult.aim_xy",
        "DeviceProposalSolveResult.ok",
        "DeviceProposalSolveResult.resid_m",
        "DeviceProposalSolveResult.attempted_v_ball_in",
        "DeviceProposalSolveResult.producer_fault_bits",
        "DeviceProposalSolveResult.proposals",
        "PRODUCER_FAULT_NONFINITE_PROPOSAL",
        "PRODUCER_FAULT_REFERENCE_NORMAL",
        "PRODUCER_FAULT_BASE_QUATERNION",
        "PRODUCER_FAULT_ACTION_RANGE",
        "PRODUCER_FAULT_PROTOTYPE_DIRECTION",
        "PRODUCER_FAULT_PROTOTYPE_SPEED",
        "PRODUCER_FAULT_PROTOTYPE_FACE_SIGN",
        "PRODUCER_FAULT_MASK",
        "solve_proposals_device",
    }
    covered = set(SURFACE.COVERED["continuous_questions.py"])
    excluded = set(SURFACE.EXCLUDED["continuous_questions.py"])
    assert required <= covered
    assert required.isdisjoint(excluded)


def test_lm_rejection_reason_abi_is_covered_not_excluded():
    continuous_required = {
        "_R_LM_SOLVE_INFO",
        "_R_LM_SOLVE_NONFINITE",
        "_CONTINUOUS_REASONS",
        "_fixed_direction_replay",
    }
    runtime_required = {
        "action_ball_solver_profile_contract",
        "RacketTargetCommand._action_ball_refill_pool_many",
    }
    continuous_covered = set(SURFACE.COVERED["continuous_questions.py"])
    continuous_excluded = set(SURFACE.EXCLUDED["continuous_questions.py"])
    runtime_covered = set(SURFACE.COVERED["hope_commands.py"])
    runtime_excluded = set(SURFACE.EXCLUDED["hope_commands.py"])
    assert continuous_required <= continuous_covered
    assert continuous_required.isdisjoint(continuous_excluded)
    assert runtime_required <= runtime_covered
    assert runtime_required.isdisjoint(runtime_excluded)


def test_fresh_racket_protocol_and_drain_name_collisions_are_explicit():
    """Fresh runtime plumbing must not be mislabelled as solver mathematics."""

    excluded = SURFACE.EXCLUDED["hope_commands.py"]
    assert excluded[
        "RacketTargetCommand._initialize_action_ball_full_mdp_racket_protocol_state"
    ] == "fresh_full_mdp_runtime_protocol"
    for symbol in (
        "_ActionBallContinuousRacketPreparedPpoDrainPack.pack",
        "_ActionBallContinuousRacketPreparedPpoDrainPack.authority",
    ):
        assert excluded[symbol] == "overapproximated_name_collision"

    declaration = SURFACE.semantic_surface_declaration(_live_reader())
    reached = declaration["excluded_but_reached_from_covered"]
    assert (
        "hope_commands.py:"
        "RacketTargetCommand._initialize_action_ball_full_mdp_racket_protocol_state"
        not in reached
    )


def test_new_symbol_in_a_fully_enumerated_source_is_refused():
    """Adding a function to a pure solver source without classifying it."""

    reader = _mutated_reader(
        [
            (
                "continuous_questions.py",
                "def solve_proposals(",
                "def _smuggled_solver_helper(x):\n"
                "    return x * 2.0\n\n\n"
                "def solve_proposals(",
            )
        ]
    )
    blockers = SURFACE.surface_blockers(reader)
    assert any(
        blocker
        == "symbol_unclassified:continuous_questions.py:_smuggled_solver_helper"
        for blocker in blockers
    ), blockers
    with pytest.raises(SURFACE.SolverSemanticSurfaceError):
        SURFACE.semantic_surface_contract(reader)


def test_new_helper_called_from_a_covered_symbol_is_refused():
    """A covered entry point that starts calling an unclassified helper."""

    reader = _mutated_reader(
        [
            (
                "hope_commands.py",
                "def _action_ball_canonical_sha256(",
                "def _action_ball_smuggled_question_tweak(rows):\n"
                "    return rows\n\n\n"
                "def _action_ball_canonical_sha256(",
            ),
            (
                "hope_commands.py",
                "                solver_field_contract = (",
                "                _action_ball_smuggled_question_tweak(None)\n"
                "                solver_field_contract = (",
            ),
        ]
    )
    blockers = SURFACE.surface_blockers(reader)
    assert any(
        blocker.startswith(
            "referenced_symbol_unclassified:hope_commands.py:"
            "_action_ball_smuggled_question_tweak"
        )
        for blocker in blockers
    ), blockers
    with pytest.raises(SURFACE.SolverSemanticSurfaceError):
        SURFACE.semantic_surface_contract(reader)


def test_quietly_dropping_a_symbol_from_coverage_moves_the_pin(monkeypatch):
    """Coverage can shrink, but never silently: the pin moves when it does."""

    baseline = _live_surface_sha256()
    shrunk = dict(SURFACE.COVERED)
    shrunk["virtual_ball.py"] = tuple(
        name for name in shrunk["virtual_ball.py"] if name != "flight_accel"
    )
    monkeypatch.setattr(SURFACE, "COVERED", shrunk)
    monkeypatch.setattr(
        SURFACE,
        "EXCLUDED",
        {
            **SURFACE.EXCLUDED,
            "virtual_ball.py": {
                **SURFACE.EXCLUDED["virtual_ball.py"],
                "flight_accel": "grading_and_observation",
            },
        },
    )
    assert SURFACE.surface_blockers(_live_reader()) == ()
    assert _live_surface_sha256() != baseline


def test_a_symbol_cannot_be_both_covered_and_excluded(monkeypatch):
    monkeypatch.setattr(
        SURFACE,
        "EXCLUDED",
        {
            **SURFACE.EXCLUDED,
            "virtual_ball.py": {
                **SURFACE.EXCLUDED["virtual_ball.py"],
                "flight_accel": "grading_and_observation",
            },
        },
    )
    blockers = SURFACE.surface_blockers(_live_reader())
    assert (
        "symbol_both_covered_and_excluded:virtual_ball.py:flight_accel"
        in blockers
    )


def test_an_unreachability_claiming_exclusion_that_is_reached_is_refused(
    monkeypatch,
):
    """Gate 3: "action-ball never calls it" has to be true, not just written down.

    This is the half of the ``fixed_direction`` accident that a symbol digest
    cannot see.  ``other_product_line`` says the free-direction line is dead
    code on this path; the moment a covered symbol reaches it, that sentence is
    false and the surface must refuse rather than keep quoting itself.
    """

    reader = _mutated_reader(
        [
                (
                    "continuous_questions.py",
                    "\n    out, good, reasons = _solve_fixed_direction_batch(\n"
                    "        clip_ids=safe_clip_ids,\n"
                    "        p_contact=safe_p_contact,\n",
                    "\n    _uniform_box(None, None, None, None)\n"
                    "    out, good, reasons = _solve_fixed_direction_batch(\n"
                    "        clip_ids=safe_clip_ids,\n"
                    "        p_contact=safe_p_contact,\n",
                )
        ]
    )
    blockers = SURFACE.surface_blockers(reader)
    assert any(
        blocker.startswith(
            "exclusion_claims_unreachable_but_is_reached:"
            "continuous_questions.py:_uniform_box:other_product_line"
        )
        for blocker in blockers
    ), blockers
    with pytest.raises(SURFACE.SolverSemanticSurfaceError):
        SURFACE.semantic_surface_contract(reader)


def test_gate_three_only_polices_reasons_that_claim_unreachability():
    """A reached exclusion whose reason does not claim unreachability is fine.

    ``_action_ball_note`` (telemetry) really is called from a covered entry
    point.  Its reason is "reporting only", not "never called", so it must not
    be a blocker -- otherwise gate 3 would just be a second coverage gate with a
    different name.
    """

    declaration = SURFACE.semantic_surface_declaration(_live_reader())
    reached = declaration["excluded_but_reached_from_covered"]
    assert (
        "hope_commands.py:RacketTargetCommand._action_ball_note" in reached
    ), sorted(reached)
    assert SURFACE.surface_blockers(_live_reader()) == ()
    for key, entry in reached.items():
        assert entry["reason"] not in SURFACE.UNREACHABLE_CLAIM_REASONS, key


def test_declaration_publishes_which_exclusions_the_closure_actually_reaches():
    """The receipt has to say what it did NOT check, or it is not a receipt.

    Everything in ``excluded_but_reached_from_covered`` is an exclusion resting
    on the stronger claim "reached, but cannot move an answer".  Publishing the
    set is what makes that claim auditable without re-deriving the call graph.
    """

    declaration = SURFACE.semantic_surface_declaration(_live_reader())
    reached = declaration["excluded_but_reached_from_covered"]
    assert reached, "the closure reaches at least the telemetry note helper"
    for key, entry in reached.items():
        filename, _, symbol = key.partition(":")
        assert filename in SURFACE.PINNED_SOURCES
        assert symbol in SURFACE.EXCLUDED[filename]
        assert entry["reason"] == SURFACE.EXCLUDED[filename][symbol]
        assert entry["referenced_from"], key
    # Everything classified but unreached is absent, and that is the claim the
    # unreachability reason codes rest on.
    for filename, reasons in SURFACE.EXCLUDED.items():
        for symbol, reason in reasons.items():
            if reason in SURFACE.UNREACHABLE_CLAIM_REASONS:
                assert "%s:%s" % (filename, symbol) not in reached


# --------------------------------------------------------------------------- #
# 4. Gate 4: the producer of a solver ARGUMENT may not be unclassified.        #
# --------------------------------------------------------------------------- #
def test_the_reference_normal_producer_must_stay_classified(monkeypatch):
    """``_ensure_reference_strike_state`` was in neither list, and that was a hole.

    It builds ``self._ref_racket_normal_raw_w_per_clip``, which the covered
    entry points hand straight to ``solve_proposals`` as ``ref_normal``.
    Rotating those rows changes every answer.  Gates 1--3 could never reach it:
    the covered symbols mention the ATTRIBUTE name, never the method that
    writes it.  Un-classify it here and the surface must refuse.
    """

    shrunk = dict(SURFACE.EXCLUDED["hope_commands.py"])
    removed = shrunk.pop(
        "RacketTargetCommand._ensure_reference_strike_state"
    )
    assert removed == "reference_strike_state_production"
    monkeypatch.setattr(
        SURFACE,
        "EXCLUDED",
        {**SURFACE.EXCLUDED, "hope_commands.py": shrunk},
    )
    blockers = SURFACE.surface_blockers(_live_reader())
    assert (
        "attribute_producer_unclassified:hope_commands.py:"
        "RacketTargetCommand._ensure_reference_strike_state:writes:"
        "_ref_racket_normal_raw_w_per_clip"
    ) in blockers, blockers
    with pytest.raises(SURFACE.SolverSemanticSurfaceError):
        SURFACE.semantic_surface_contract(_live_reader())


def test_a_brand_new_unclassified_writer_of_a_solver_argument_is_refused():
    """The general shape, not just the one instance that was already there."""

    reader = _mutated_reader(
        [
            (
                "hope_commands.py",
                "    def _ensure_reference_strike_state(self):",
                "    def _smuggled_reference_normal_tweak(self):\n"
                "        self._ref_racket_normal_raw_w_per_clip = None\n\n"
                "    def _ensure_reference_strike_state(self):",
            )
        ]
    )
    blockers = SURFACE.surface_blockers(reader)
    assert any(
        blocker.startswith(
            "attribute_producer_unclassified:hope_commands.py:"
            "RacketTargetCommand._smuggled_reference_normal_tweak"
        )
        for blocker in blockers
    ), blockers
    with pytest.raises(SURFACE.SolverSemanticSurfaceError):
        SURFACE.semantic_surface_contract(reader)


def test_the_exclusion_that_can_move_an_answer_says_so_in_its_own_words():
    """An exclusion reason that hides the truth is worse than no exclusion.

    ``reference_strike_state_production`` is the one entry on the list whose
    symbol CAN change an answer.  Its reason text has to say that out loud, name
    what holds it honest instead, and name the residual as an open hole -- the
    same shape ``runtime_wiring`` already uses.  This test reads the emitted
    strings, not the intent.
    """

    reason = SURFACE.EXCLUSION_REASONS["reference_strike_state_production"]
    assert "ref_normal" in reason
    assert "open hole" in reason
    assert "R10" in reason
    declaration = SURFACE.semantic_surface_declaration(_live_reader())
    entry = declaration["attribute_producers"][
        "hope_commands.py:RacketTargetCommand._ensure_reference_strike_state"
    ]
    assert entry["classification"] == "reference_strike_state_production"
    assert "_ref_racket_normal_raw_w_per_clip" in entry["writes"]


# --------------------------------------------------------------------------- #
# 5. Gate 5: a digest proves a body did not change, not that it is what runs.  #
# --------------------------------------------------------------------------- #
def test_rebinding_a_solve_entry_point_to_an_uncovered_method_is_refused():
    """Escape 1, exactly as measured: the digest does not move, the gate fires.

    The wiring binds ``solve_many=`` to a method only the excluded region names.
    No covered symbol changed, so ``semantic_surface`` is byte-identical -- this
    test asserts that too, because the point is that the digest CANNOT see it.
    """

    reader = _mutated_reader(
        [
            (
                "hope_commands.py",
                "                solve_many=self._action_ball_refill_pool_many,",
                "                solve_many=self._action_ball_smuggled_many,",
            )
        ]
    )
    blockers = SURFACE.surface_blockers(reader)
    # The line number is part of the blocker so a reader can find the site; it
    # is deliberately not asserted, because pinning it here would make every
    # unrelated edit above that line fail this test for the wrong reason.
    assert any(
        blocker.startswith("pool_solver_binding_undeclared:hope_commands.py:")
        and blocker.endswith(
            ":solve_many:self._action_ball_smuggled_many"
        )
        for blocker in blockers
    ), blockers
    with pytest.raises(SURFACE.SolverSemanticSurfaceError):
        SURFACE.semantic_surface_contract(reader)
    # And every covered digest is byte-identical -- that is the whole point.
    # The pin proves these bodies did not change; the gate is what proves they
    # are still what runs.
    assert _covered_digests(reader) == _covered_digests(_live_reader())


def test_dropping_a_policed_adapter_slot_entirely_is_refused():
    """Omitting the keyword is not a way around the allow-list."""

    reader = _mutated_reader(
        [
            (
                "hope_commands.py",
                "                solve=self._action_ball_refill_pool,\n",
                "",
            )
        ]
    )
    blockers = SURFACE.surface_blockers(reader)
    assert any(
        blocker.startswith("pool_solver_binding_absent:hope_commands.py")
        and blocker.endswith(":solve")
        for blocker in blockers
    ), blockers


def test_every_declared_pool_solver_binding_is_actually_used():
    """An allow-list entry nobody binds is a hole waiting to be filled.

    The list is not in the digest, so a stale entry costs nothing to leave
    behind -- and a stale entry is exactly a pre-approved rebinding target.
    """

    live = set(
        SURFACE.semantic_surface_declaration(_live_reader())[
            "pool_solver_bindings"
        ].values()
    )
    declared = {
        text
        for texts in SURFACE.POOL_SOLVER_BINDINGS.values()
        for text in texts
    }
    assert declared == live, declared.symmetric_difference(live)


def test_every_exclusion_reason_is_defined_and_used():
    used = {
        reason
        for reasons in SURFACE.EXCLUDED.values()
        for reason in reasons.values()
    }
    assert used <= set(SURFACE.EXCLUSION_REASONS)
    unused = set(SURFACE.EXCLUSION_REASONS) - used
    assert unused == set(), (
        "an exclusion reason nobody uses is documentation rot: " + repr(unused)
    )


def test_declaration_names_what_it_covers_and_what_it_refuses_to():
    declaration = SURFACE.semantic_surface_declaration(_live_reader())
    assert declaration["covered_symbol_count"] > 0
    assert set(declaration["covered"]) <= set(SURFACE.PINNED_SOURCES)
    assert set(declaration["excluded"]) <= set(SURFACE.PINNED_SOURCES)
    # Every excluded symbol carries a reason, and every reason carries prose.
    for filename, reasons in declaration["excluded"].items():
        for symbol, reason in reasons.items():
            assert declaration["exclusion_reasons"][reason].strip(), (
                f"{filename}:{symbol} is excluded without an explanation"
            )
    # strike_spec_torch was in no pin at all before this change; the whole point
    # of the surface is that it now is.
    assert "strike_spec_torch.py" in declaration["covered"]
    assert "_seed" in declaration["covered"]["strike_spec_torch.py"]


def test_surface_fails_closed_when_a_pinned_source_cannot_be_read():
    def broken(name: str) -> str:
        if name == "virtual_ball.py":
            raise OSError("no such file")
        return _live_reader()(name)

    blockers = SURFACE.surface_blockers(broken)
    assert any(
        blocker.startswith("pinned_source_unreadable:virtual_ball.py")
        for blocker in blockers
    ), blockers
    with pytest.raises(SURFACE.SolverSemanticSurfaceError):
        SURFACE.semantic_surface_contract(broken)


def test_symbol_digest_is_stable_across_repeated_reads(tmp_path):
    copied = tmp_path / "mdp"
    copied.mkdir()
    for name in SURFACE.PINNED_SOURCES:
        shutil.copy2(MDP_DIR / name, copied / name)
    shutil.copy2(SURFACE_SOURCE, copied / SURFACE_SOURCE.name)
    reader = SURFACE.source_reader_for_directory(copied)
    assert (
        SURFACE.semantic_surface_contract(reader)["sha256"]
        == _live_surface_sha256()
    )
