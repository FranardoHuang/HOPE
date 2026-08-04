import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MDP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
TAPE_PATH = MDP_ROOT / "action_ball_fixed_question_tape.py"
BUILDER_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_action_ball_immutable_n1_tape.py"
)
SPEC = importlib.util.spec_from_file_location(
    "action_ball_fixed_question_tape_test_target", TAPE_PATH
)
T = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = T
SPEC.loader.exec_module(T)
R = T._runtime


def _digest(label):
    return hashlib.sha256(str(label).encode("utf-8")).hexdigest()


def _yaw_quat(yaw):
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _binding():
    return R.ActionBinding(
        action_uid=10_073,
        action_slot=0,
        motion_path="vendor_assets/motions/fixed_n1.npz",
        motion_sha256=_digest("motion"),
        profile_sha256=_digest("profile"),
    )


def _pins():
    return R.RuntimePins(
        manifest_sha256=_digest("manifest"),
        sampler_sha256=_digest("sampler"),
        domain_authority_sha256=_digest("domain"),
        physics_sha256=_digest("physics"),
        solver_sha256=_digest("solver"),
    )


def _registry_sha256(binding, pins):
    return R._registry_sha256((binding,), pins, "no_move")


def _birth(env_id=0, *, birth_index=0, generation=1):
    binding = _binding()
    pins = _pins()
    levels = R.ActionDomainLevels()
    claim = R.ActionDomainClaim(
        authority_contract_sha256=pins.domain_authority_sha256,
        arm_catalog_sha256=R.ARM_CATALOG_SHA256,
        action_uid=binding.action_uid,
        domain_epoch=0,
        domain_levels=levels,
        levels_sha256=levels.canonical_sha256,
        profile_sha256=binding.profile_sha256,
        mobility_mode="no_move",
    )
    draw_start = birth_index * R.SAMPLER_BIRTH_DRAW_COUNT
    yaw = 0.0
    spawn = (-0.1, 0.0, 0.0)
    birth_identity = {
        "schema_version": R.SAMPLER_SCHEMA_VERSION,
        "kind": "base_birth",
        "sampler_contract_sha256": pins.sampler_sha256,
        "arm_catalog_sha256": R.ARM_CATALOG_SHA256,
        "action_uid": binding.action_uid,
        "domain_epoch": 0,
        "levels_sha256": levels.canonical_sha256,
        "profile_sha256": binding.profile_sha256,
        "birth_index": birth_index,
        "draw_start": draw_start,
        "draw_end": draw_start + R.SAMPLER_BIRTH_DRAW_COUNT,
        "mobility_mode": "no_move",
        "base_yaw_rad": yaw,
        "base_start_w_m": spawn,
    }
    return R.ActionBirthReceipt(
        registry_sha256=_registry_sha256(binding, pins),
        env_id=env_id,
        reset_generation=generation,
        action_uid=binding.action_uid,
        action_slot=binding.action_slot,
        domain_epoch=0,
        domain_claim_sha256=claim.canonical_sha256,
        domain_authority_sha256=pins.domain_authority_sha256,
        domain_levels=levels,
        arm_catalog_sha256=R.ARM_CATALOG_SHA256,
        levels_sha256=levels.canonical_sha256,
        sampler_birth_sha256=R._sha256_json(birth_identity),
        sampler_birth_index=birth_index,
        sampler_draw_start=draw_start,
        sampler_draw_end=draw_start + R.SAMPLER_BIRTH_DRAW_COUNT,
        mobility_mode="no_move",
        base_yaw_rad=yaw,
        base_quat_wxyz=_yaw_quat(yaw),
        base_spawn_w_m=spawn,
        manifest_sha256=pins.manifest_sha256,
        sampler_sha256=pins.sampler_sha256,
        profile_sha256=binding.profile_sha256,
        motion_sha256=binding.motion_sha256,
        physics_sha256=pins.physics_sha256,
        solver_sha256=pins.solver_sha256,
    )


def _source_task(birth):
    contact = (0.55, 0.0, 0.90)
    incoming = (-4.0, 0.1, -0.2)
    incoming_speed = math.sqrt(sum(value * value for value in incoming))
    incoming_direction = tuple(value / incoming_speed for value in incoming)
    spin = (0.0, 12.0, 1.0)
    spin_magnitude = math.sqrt(sum(value * value for value in spin))
    spin_direction = tuple(value / spin_magnitude for value in spin)
    racket_velocity = (3.0, 0.2, 0.4)
    geometry = R._contact_geometry.solve_exact_face_contact(
        ball_contact_w_m=contact,
        racket_face_center_velocity_w_mps=racket_velocity,
        solved_raw_a_normal_w=(1.0, 0.0, 0.0),
        mount_normal_sign=1,
        reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
        reference_racket_site_speed_mps=3.0,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
    )
    timing = R.derive_action_teacher_site_timing(
        racket_site_velocity_w_mps=geometry.racket_site_velocity_w_mps,
        time_to_contact_s=1.2,
        reference_t_hit_s=0.42,
        reference_t_cycle_s=1.2,
        reference_racket_site_speed_mps=3.0,
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
    )
    draw_start = 1_000
    question = {
        "schema_version": R.SAMPLER_SCHEMA_VERSION,
        "kind": "swing_sample",
        "sampler_contract_sha256": birth.sampler_sha256,
        "arm_catalog_sha256": birth.arm_catalog_sha256,
        "sample_index": 0,
        "action_uid": birth.action_uid,
        "domain_epoch": birth.domain_epoch,
        "domain_levels": birth.domain_levels.to_dict(),
        "birth_id": birth.sampler_birth_sha256,
        "profile_sha256": birth.profile_sha256,
        "levels_sha256": birth.levels_sha256,
        "draw_start": draw_start,
        "draw_end": draw_start + R.SAMPLER_SAMPLE_DRAW_COUNT,
        "mobility_mode": birth.mobility_mode,
        "base_yaw_rad": birth.base_yaw_rad,
        "base_start_w_m": birth.base_spawn_w_m,
        "base_spawn_latent_w_m": birth.base_spawn_w_m,
        "base_travel_latent_b_yaw_m": (0.0, 0.0, 0.0),
        "base_goal_w_m": birth.base_spawn_w_m,
        "contact_offset_from_base_goal_b_yaw_m": (
            contact[0] - birth.base_spawn_w_m[0],
            contact[1] - birth.base_spawn_w_m[1],
            contact[2] - birth.base_spawn_w_m[2],
        ),
        "contact_w_m": contact,
        "time_to_contact_s": 1.2,
        "incoming_speed_mps": incoming_speed,
        "incoming_direction_b_yaw": incoming_direction,
        "incoming_direction_w": incoming_direction,
        "incoming_velocity_w_mps": incoming,
        "spin_magnitude_radps": spin_magnitude,
        "spin_direction_b_yaw": spin_direction,
        "spin_direction_w": spin_direction,
        "spin_w_radps": spin,
        "landing_aim_w_xy_m": (2.5, -0.1),
    }
    return R.ActionBallTaskReceipt.from_birth(
        birth,
        sample_sha256=R._sha256_json(question),
        sample_index=0,
        sample_draw_start=draw_start,
        sample_draw_end=draw_start + R.SAMPLER_SAMPLE_DRAW_COUNT,
        swing_generation=0,
        base_goal_w_m=birth.base_spawn_w_m,
        base_spawn_latent_w_m=birth.base_spawn_w_m,
        base_travel_latent_b_yaw_m=(0.0, 0.0, 0.0),
        contact_offset_from_base_goal_b_yaw_m=(0.65, 0.0, 0.90),
        ball_contact_w_m=contact,
        racket_site_target_w_m=geometry.racket_site_target_w_m,
        time_to_contact_s=1.2,
        incoming_speed_mps=incoming_speed,
        incoming_direction_b_yaw=incoming_direction,
        incoming_velocity_w_mps=incoming,
        spin_magnitude_radps=spin_magnitude,
        spin_direction_b_yaw=spin_direction,
        incoming_spin_w_radps=spin,
        landing_aim_w_xy_m=(2.5, -0.1),
        mount_normal_sign=1,
        racket_normal_w=(1.0, 0.0, 0.0),
        reference_racket_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_racket_angular_velocity_w_radps=(0.0, 0.0, 0.0),
        racket_command_quat_wxyz=geometry.racket_command_quat_wxyz,
        racket_face_center_velocity_w_mps=(
            geometry.racket_face_center_velocity_w_mps
        ),
        racket_site_velocity_w_mps=geometry.racket_site_velocity_w_mps,
        racket_command_angular_velocity_w_radps=(
            geometry.racket_command_angular_velocity_w_radps
        ),
        geometry_source_sha256=geometry.geometry_source_sha256,
        reference_t_hit_s=0.42,
        reference_t_cycle_s=1.2,
        reference_racket_site_speed_mps=3.0,
        required_racket_site_speed_mps=timing.required_racket_site_speed_mps,
        reaction_margin_s=0.05,
        teacher_rate_min=0.8,
        teacher_rate_max=1.2,
        teacher_rate=timing.teacher_rate,
        scaled_t_hit_s=timing.scaled_t_hit_s,
        scaled_t_cycle_s=timing.scaled_t_cycle_s,
        pre_swing_wait_s=timing.pre_swing_wait_s,
        solver_residual_m=0.004,
    )


def _tape():
    receipt = _source_task(_birth())
    return T.ImmutableN1QuestionTape.from_receipts(
        question_receipt=receipt,
        target_receipts={recipe: receipt for recipe in T.TARGET_RECIPES},
        target_producer_sha256={
            recipe: _digest(f"producer-{recipe}")
            for recipe in T.TARGET_RECIPES
        },
    )


def _request(birth, refill_index=1):
    return R.ActionPoolRefillRequest(
        action_uid=birth.action_uid,
        action_slot=birth.action_slot,
        refill_index=refill_index,
        minimum_receipts=1,
        swing_generation_start=0,
        mobility_mode="no_move",
        registry_sha256=birth.registry_sha256,
        binding=_binding(),
        pins=_pins(),
        birth=birth,
    )


class _ExpandOnlyTensor:
    """Dependency-light tensor double that forbids value/env iteration."""

    def __init__(self, shape, calls, storage):
        self.shape = tuple(shape)
        self.calls = calls
        self.storage = storage

    def __iter__(self):
        raise AssertionError("device expansion must not iterate tensor values")

    def unsqueeze(self, dim):
        assert dim == 0
        self.calls["unsqueeze"] += 1
        return _ExpandOnlyTensor((1, *self.shape), self.calls, self.storage)

    def expand(self, *shape):
        self.calls["expand"] += 1
        return _ExpandOnlyTensor(shape, self.calls, self.storage)


def _expand_only_tensor(width, *, row_zero_batched=False):
    calls = {"unsqueeze": 0, "expand": 0}
    storage = object()
    shape = (1, width) if row_zero_batched else (width,)
    return _ExpandOnlyTensor(shape, calls, storage), calls, storage


def test_all_five_recipes_share_one_question_and_fixed_width():
    tape = _tape()
    assert tape.question_sha256 == tape.to_dict()["question_sha256"]
    assert tape.to_dict()["row_count"] == 1
    assert tape.to_dict()["question_shape"] == [1, 15]
    assert tape.to_dict()["install_shape_per_recipe"] == [1, 31]
    assert tape.to_dict()["observation_shape_per_recipe"] == [1, 24]
    assert T.ImmutableN1QuestionTape.from_dict(tape.to_dict()) == tape

    rows = {recipe: tape.observation_row(recipe) for recipe in T.TARGET_RECIPES}
    assert {len(row) for row in rows.values()} == {24}
    assert {row[: T.QUESTION_WIDTH] for row in rows.values()} == {
        tape.question_values
    }
    assert rows["outcome_dense_only"][T.QUESTION_WIDTH :] == (0.0,) * 9

    lineages = [tape.target_lineage(recipe) for recipe in T.TARGET_RECIPES]
    assert {row["base_question_sha256"] for row in lineages} == {
        tape.question_sha256
    }
    assert len({row["target_producer_sha256"] for row in lineages}) == 5
    assert len({row["target_column_sha256"] for row in lineages}) == 3

    view = tape.reset_batch_view("analytic_full", batch_size=4096)
    assert view.batch_size == 4096
    assert view.row_index == 0
    assert len(view.install_row) == 31
    assert len(view.observation_row) == 24
    assert view.lineage["base_question_sha256"] == tape.question_sha256


@pytest.mark.parametrize("recipe", T.TARGET_RECIPES)
def test_timing_rows_have_one_fixed_runtime_width(recipe):
    tape = _tape()
    row = tape.timing_row(recipe)
    view = tape.reset_batch_view(recipe, batch_size=4096)

    assert T.TIMING_WIDTH == sum(width for _name, width in T.TIMING_LAYOUT)
    assert len(row) == T.TIMING_WIDTH == 15
    assert view.timing_row == row


@pytest.mark.parametrize("batch_size", (1, 4096))
@pytest.mark.parametrize("row_zero_batched", (False, True))
@pytest.mark.parametrize(
    ("expand_method", "width"),
    (
        ("expand_install_rows", T.INSTALL_WIDTH),
        ("expand_observation_rows", T.OBSERVATION_WIDTH),
        ("expand_timing_rows", T.TIMING_WIDTH),
    ),
)
def test_solver_reset_batch_view_uses_constant_device_expand_work(
    batch_size, row_zero_batched, expand_method, width
):
    solver = T.FixedQuestionTapeSolver(
        tape=_tape(),
        target_recipe="analytic_full",
        solver_contract_sha256=_pins().solver_sha256,
    )
    before = solver.state_dict()
    row, row_calls, row_storage = _expand_only_tensor(
        width, row_zero_batched=row_zero_batched
    )
    view = solver.reset_batch_view(batch_size=batch_size)
    rows = getattr(view, expand_method)(row)

    assert rows.shape == (batch_size, width)
    assert rows.storage is row_storage
    assert row_calls == {
        "unsqueeze": int(not row_zero_batched),
        "expand": 1,
    }
    assert view.lineage["solver_contract_sha256"] == _pins().solver_sha256
    assert view.lineage["state_owner_sha256"] == solver.state_owner_sha256
    with pytest.raises(TypeError):
        view.lineage["state_owner_sha256"] = "0" * 64
    assert solver.state_dict() == before


def test_solver_reset_batch_view_is_hash_and_receipt_free_after_init(monkeypatch):
    solver = T.FixedQuestionTapeSolver(
        tape=_tape(),
        target_recipe="teacher_pos_face_no_velocity",
        solver_contract_sha256=_pins().solver_sha256,
    )
    template = solver.reset_batch_view(batch_size=1)
    before = solver.state_dict()
    forbidden_calls = []
    sha256_json = T._sha256_json

    def forbidden(*args, **kwargs):
        forbidden_calls.append((args, kwargs))
        raise AssertionError("reset batch view rebuilt per-env identity")

    monkeypatch.setattr(T, "_sha256_json", forbidden)
    monkeypatch.setattr(solver, "_materialize", forbidden)
    monkeypatch.setattr(solver, "materialize_many", forbidden)
    monkeypatch.setattr(solver, "solve_many", forbidden)
    view = solver.reset_batch_view(batch_size=4096)
    assert view.batch_size == 4096
    assert view.install_row is template.install_row
    assert view.observation_row is template.observation_row
    assert view.timing_row is template.timing_row
    assert view.lineage is template.lineage
    assert forbidden_calls == []
    monkeypatch.setattr(T, "_sha256_json", sha256_json)
    assert solver.state_dict() == before


@pytest.mark.parametrize("recipe", T.TARGET_RECIPES)
def test_solver_reset_batch_view_install_row_matches_legacy_receipt(recipe):
    solver = T.FixedQuestionTapeSolver(
        tape=_tape(),
        target_recipe=recipe,
        solver_contract_sha256=_pins().solver_sha256,
    )
    view = solver.reset_batch_view(batch_size=1)
    batch = solver(_request(_birth(env_id=3, birth_index=3)))
    receipt = batch.receipts[0]
    receipt_install_row = T._flatten_layout(
        {
            name: getattr(receipt, name)
            for name, _width in T.INSTALL_LAYOUT
        },
        T.INSTALL_LAYOUT,
    )

    assert view.install_row == receipt_install_row
    assert view.lineage["target_recipe"] == recipe
    assert tuple(view.lineage["target_validity_mask"]) == (
        T.TARGET_VALIDITY_BY_RECIPE[recipe]
    )


@pytest.mark.parametrize("batch_size", (0, -1, True, 1.0))
def test_solver_reset_batch_view_rejects_non_positive_plain_sizes(batch_size):
    solver = T.FixedQuestionTapeSolver(
        tape=_tape(),
        target_recipe="outcome_dense_only",
        solver_contract_sha256=_pins().solver_sha256,
    )
    with pytest.raises(ValueError, match="positive plain integer"):
        solver.reset_batch_view(batch_size=batch_size)


def test_file_bytes_are_pinned_and_canonical(tmp_path):
    tape = _tape()
    path = tmp_path / "n1-tape.json"
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
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert T.load_immutable_n1_tape(
        path, expected_file_sha256=digest
    ) == tape
    with pytest.raises(ValueError, match="file SHA"):
        T.load_immutable_n1_tape(
            path, expected_file_sha256="0" * 64
        )


def test_builder_emits_one_canonical_five_recipe_container(tmp_path):
    builder_spec = importlib.util.spec_from_file_location(
        "action_ball_fixed_question_builder_test_target", BUILDER_PATH
    )
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec.loader is not None
    sys.modules[builder_spec.name] = builder
    builder_spec.loader.exec_module(builder)

    receipt = _source_task(_birth())
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            receipt.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "tape.json"
    arguments = [
        "--question-receipt",
        str(receipt_path),
        "--expected-action-uid",
        str(receipt.action_uid),
        "--output",
        str(output),
    ]
    for recipe in T.TARGET_RECIPES:
        arguments.extend(("--target-receipt", f"{recipe}={receipt_path}"))
        arguments.extend(
            (
                "--target-producer-sha256",
                f"{recipe}={_digest('producer-' + recipe)}",
            )
        )
    assert builder.main(arguments) == 0
    raw = output.read_bytes()
    built = T.load_immutable_n1_tape(
        output,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert built.question_sha256 == _tape().question_sha256
    assert set(built.targets) == set(T.TARGET_RECIPES)


def test_legacy_4096_receipts_keep_zero_lm_and_rng_draws():
    tape = _tape()
    solver = T.FixedQuestionTapeSolver(
        tape=tape,
        target_recipe="analytic_full",
        solver_contract_sha256=_pins().solver_sha256,
    )
    births = tuple(
        _birth(env_id=index, birth_index=index) for index in range(4096)
    )
    batches, issues = solver.materialize_many(
        tuple(_request(birth) for birth in births)
    )
    assert len(batches) == len(issues) == 4096
    assert sum(batch.proposed_count for batch in batches) == 4096
    assert sum(len(batch.receipts) for batch in batches) == 4096
    assert len({issue.task_receipt.sample_sha256 for issue in issues}) == 4096
    assert {
        issue.lineage["base_question_sha256"] for issue in issues
    } == {tape.question_sha256}
    assert {issue.observation_row for issue in issues} == {
        tape.observation_row("analytic_full")
    }
    assert solver.online_lm_calls == 0
    assert solver.physical_rng_draws == 0
    assert solver.state_dict()["online_lm_calls"] == 0
    assert solver.state_dict()["physical_rng_draws"] == 0
    assert solver.sample_highwater_for(_binding().action_uid)[0] == 4095

    # The module imports neither the continuous producer nor its LM entrypoint.
    tree = ast.parse(TAPE_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("continuous_questions" in name for name in imports)


def test_reset_batch_view_does_not_perturb_legacy_identity_or_replay_state():
    tape = _tape()
    fast = T.FixedQuestionTapeSolver(
        tape=tape,
        target_recipe="analytic_no_velocity",
        solver_contract_sha256=_pins().solver_sha256,
    )
    control = T.FixedQuestionTapeSolver(
        tape=tape,
        target_recipe="analytic_no_velocity",
        solver_contract_sha256=_pins().solver_sha256,
    )
    requests = tuple(
        _request(_birth(env_id=index, birth_index=index), refill_index=index + 1)
        for index in range(8)
    )

    initial_state = fast.state_dict()
    view = fast.reset_batch_view(batch_size=len(requests))
    assert fast.state_dict() == initial_state
    fast_batches, fast_issues = fast.materialize_many(requests)
    control_batches, control_issues = control.materialize_many(requests)

    assert fast_batches == control_batches
    assert fast_issues == control_issues
    assert fast.state_dict() == control.state_dict()
    assert R._sha256_json(fast.state_dict()) == R._sha256_json(
        control.state_dict()
    )
    assert fast.online_lm_calls == control.online_lm_calls == 0
    assert fast.physical_rng_draws == control.physical_rng_draws == 0
    assert fast.sample_highwater_for(_binding().action_uid) == (
        control.sample_highwater_for(_binding().action_uid)
    )
    assert {
        issue.lineage["logical_sample_index"] for issue in fast_issues
    } == set(range(8))
    assert {
        issue.task_receipt.swing_generation for issue in fast_issues
    } == {0}
    assert {
        issue.observation_row for issue in fast_issues
    } == {view.observation_row}
    for request in requests:
        birth_sha = request.birth.canonical_sha256
        assert fast.task_transcript_for_birth(birth_sha) == (
            control.task_transcript_for_birth(birth_sha)
        )


def test_fixed_base_mismatch_fails_before_issue_and_state_is_atomic():
    tape = _tape()
    solver = T.FixedQuestionTapeSolver(
        tape=tape,
        target_recipe="outcome_dense_only",
        solver_contract_sha256=_pins().solver_sha256,
    )
    wrong = _birth(env_id=1, birth_index=1)
    object.__setattr__(wrong, "base_spawn_w_m", (0.0, 0.0, 0.0))
    before = solver.state_dict()
    with pytest.raises(ValueError, match="fixed-domain identity"):
        solver.materialize_many((_request(wrong),))
    assert solver.state_dict() == before


def test_solver_state_roundtrip_preserves_authority_and_lineage():
    tape = _tape()
    solver = T.FixedQuestionTapeSolver(
        tape=tape,
        target_recipe="teacher_pos_face_no_velocity",
        solver_contract_sha256=_pins().solver_sha256,
    )
    batch = solver(_request(_birth(env_id=7, birth_index=7)))
    receipt = batch.receipts[0]
    saved = solver.state_dict()
    restored = T.FixedQuestionTapeSolver(
        tape=tape,
        target_recipe="teacher_pos_face_no_velocity",
        solver_contract_sha256=_pins().solver_sha256,
    )
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    restored.assert_emitted_sample(receipt)
    restored.assert_emitted_tasks((receipt,))
    assert restored.lineage_for_task(receipt)["base_question_sha256"] == (
        tape.question_sha256
    )
