import ast
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MDP_ROOT = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = _load(
    "action_ball_banded_question_bank_test_target",
    MDP_ROOT / "action_ball_banded_question_bank.py",
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
FIXTURE = importlib.import_module("test_action_ball_fixed_question_tape")
FIXTURE_RUNTIME = FIXTURE.R
R = B._runtime


def _fixture_call(function, *args, **kwargs):
    prior = FIXTURE.R
    try:
        FIXTURE.R = R
        return function(*args, **kwargs)
    finally:
        FIXTURE.R = prior


def _binding():
    return _fixture_call(FIXTURE._binding)


def _pins():
    return _fixture_call(FIXTURE._pins)


def _birth(*args, **kwargs):
    return _fixture_call(FIXTURE._birth, *args, **kwargs)


def _birth_at_levels(levels, *, epoch=1):
    birth = _birth()
    identity = {
        "schema_version": R.SAMPLER_SCHEMA_VERSION,
        "kind": "base_birth",
        "sampler_contract_sha256": birth.sampler_sha256,
        "arm_catalog_sha256": birth.arm_catalog_sha256,
        "action_uid": birth.action_uid,
        "domain_epoch": epoch,
        "levels_sha256": levels.canonical_sha256,
        "profile_sha256": birth.profile_sha256,
        "birth_index": birth.sampler_birth_index,
        "draw_start": birth.sampler_draw_start,
        "draw_end": birth.sampler_draw_end,
        "mobility_mode": birth.mobility_mode,
        "base_yaw_rad": birth.base_yaw_rad,
        "base_start_w_m": birth.base_spawn_w_m,
    }
    claim = R.ActionDomainClaim(
        authority_contract_sha256=birth.domain_authority_sha256,
        arm_catalog_sha256=birth.arm_catalog_sha256,
        action_uid=birth.action_uid,
        domain_epoch=epoch,
        domain_levels=levels,
        levels_sha256=levels.canonical_sha256,
        profile_sha256=birth.profile_sha256,
        mobility_mode=birth.mobility_mode,
    )
    return R.ActionBirthReceipt(
        **{
            **birth.__dict__,
            "domain_epoch": epoch,
            "domain_levels": levels,
            "levels_sha256": levels.canonical_sha256,
            "sampler_birth_sha256": R._sha256_json(identity),
            "domain_claim_sha256": claim.canonical_sha256,
        }
    )


def _birth_at_base(*, env_id, birth_index, yaw, spawn):
    birth = _birth(env_id=env_id, birth_index=birth_index)
    identity = {
        "schema_version": R.SAMPLER_SCHEMA_VERSION,
        "kind": "base_birth",
        "sampler_contract_sha256": birth.sampler_sha256,
        "arm_catalog_sha256": birth.arm_catalog_sha256,
        "action_uid": birth.action_uid,
        "domain_epoch": birth.domain_epoch,
        "levels_sha256": birth.levels_sha256,
        "profile_sha256": birth.profile_sha256,
        "birth_index": birth_index,
        "draw_start": birth.sampler_draw_start,
        "draw_end": birth.sampler_draw_end,
        "mobility_mode": birth.mobility_mode,
        "base_yaw_rad": yaw,
        "base_start_w_m": spawn,
    }
    return R.ActionBirthReceipt(
        **{
            **birth.__dict__,
            "base_yaw_rad": yaw,
            "base_quat_wxyz": (
                math.cos(0.5 * yaw),
                0.0,
                0.0,
                math.sin(0.5 * yaw),
            ),
            "base_spawn_w_m": spawn,
            "sampler_birth_sha256": R._sha256_json(identity),
        }
    )


def _receipt(*, env_id=0, birth_index=0, generation=1):
    birth = _birth(
            env_id=env_id,
            birth_index=birth_index,
            generation=generation,
        )
    return _fixture_call(FIXTURE._source_task, birth)


def _receipt_rebased_to_birth(template, birth):
    kwargs = {
        name: getattr(template, name) for name in B._MATERIALIZED_FIELDS
    }
    contact = template.ball_contact_w_m
    kwargs.update(
        {
            "base_goal_w_m": birth.base_spawn_w_m,
            "base_spawn_latent_w_m": birth.base_spawn_w_m,
            "base_travel_latent_b_yaw_m": (0.0, 0.0, 0.0),
            "contact_offset_from_base_goal_b_yaw_m": tuple(
                contact[index] - birth.base_spawn_w_m[index]
                for index in range(3)
            ),
        }
    )
    sample_identity = template._sampler_identity_payload()
    sample_identity.update(
        {
            "birth_id": birth.sampler_birth_sha256,
            "base_yaw_rad": birth.base_yaw_rad,
            "base_start_w_m": birth.base_spawn_w_m,
            "base_spawn_latent_w_m": kwargs["base_spawn_latent_w_m"],
            "base_travel_latent_b_yaw_m": kwargs[
                "base_travel_latent_b_yaw_m"
            ],
            "base_goal_w_m": kwargs["base_goal_w_m"],
            "contact_offset_from_base_goal_b_yaw_m": kwargs[
                "contact_offset_from_base_goal_b_yaw_m"
            ],
        }
    )
    return R.ActionBallTaskReceipt.from_birth(
        birth,
        sample_sha256=R._sha256_json(sample_identity),
        sample_index=template.sample_index,
        sample_draw_start=template.sample_draw_start,
        sample_draw_end=template.sample_draw_end,
        swing_generation=template.swing_generation,
        **kwargs,
    )


def _bank_from_blocks(blocks):
    blocks = tuple(blocks)
    rows = tuple(row for block in blocks for row in block.rows)
    coverage_payload = {
        "schema_version": 1,
        "kind": "action_ball_reachable_domain_level_blocks",
        "arm_catalog_sha256": R.ARM_CATALOG_SHA256,
        "arm_keys": list(R.ARM_KEYS),
        "expected_action_uids": sorted(
            {int(block.key["action_uid"]) for block in blocks}
        ),
        "reachable_arm_keys_by_action": [
            {
                "action_uid": uid,
                "reachable_arm_keys": list(R.ARM_KEYS),
            }
            for uid in sorted({int(block.key["action_uid"]) for block in blocks})
        ],
        "reachable_blocks": [
            {
                "action_uid": int(block.key["action_uid"]),
                "levels_sha256": block.rows[0].domain_levels.canonical_sha256,
                "domain_levels": block.rows[0].domain_levels.to_dict(),
            }
            for block in blocks
        ],
    }
    coverage_sha = B._sha256_json(coverage_payload)
    return B.BandedQuestionBank(
        split_seed=20260804,
        blocks=blocks,
        coverage={
            **coverage_payload,
            "source_file_sha256": "c" * 64,
            "source_canonical_sha256": coverage_sha,
        },
        question_lineage=B.question_lineage_for_blocks(blocks),
        producer_lineage={
            "schema_version": 1,
            "kind": "action_ball_banded_question_bank.offline_solved_receipts",
            "row_order": "canonical_receipt_sha256",
            "producer_source_sha256": "a" * 64,
            "bank_module_source_sha256": "9" * 64,
            "inputs": [
                {
                    "source_id": f"test-fixture-{index}",
                    "solver_mode": "current_lm_only",
                    "block_key_sha256": block.key_sha256,
                    "file_sha256": "b" * 64,
                    "offline_producer_source_sha256": "d" * 64,
                    "offline_input_root_sha256": "e" * 64,
                    "proposed_count": len(block.rows),
                    "admitted_count": len(block.rows),
                    "rejections": [],
                    "receipt_canonical_sha256": [
                        row.canonical_sha256 for row in block.rows
                    ],
                }
                for index, block in enumerate(blocks)
            ],
        },
    )


def _bank(rows=None):
    if rows is None:
        rows = (_receipt(),)
    return _bank_from_blocks((B.BandedQuestionBlock.from_receipts(rows),))


def _request(birth, *, refill_index=1, count=1, swing_start=0):
    return R.ActionPoolRefillRequest(
        action_uid=birth.action_uid,
        action_slot=birth.action_slot,
        refill_index=refill_index,
        minimum_receipts=count,
        swing_generation_start=swing_start,
        mobility_mode=birth.mobility_mode,
        registry_sha256=birth.registry_sha256,
        binding=_binding(),
        pins=_pins(),
        birth=birth,
    )


def test_center_bank_is_canonical_file_pinned_and_exactly_keyed(tmp_path):
    bank = _bank()
    assert bank.canonical_payload["online_solver_calls_per_reset"] == 0
    assert bank.canonical_payload["online_solver_calls_per_step"] == 0
    assert bank.canonical_payload["runtime_operation"] == (
        "preindexed_cached_row_lookup_only"
    )
    assert bank.canonical_payload["missing_block_policy"] == "fail_closed"
    assert B.BandedQuestionBank.from_dict(bank.to_dict()) == bank
    block = bank.blocks[0]
    assert block.key["levels_sha256"] == R.ActionDomainLevels().canonical_sha256
    assert block.key["action_uid"] == _binding().action_uid
    assert block.key["profile_sha256"] == _binding().profile_sha256
    assert block.key["solver_sha256"] == _pins().solver_sha256

    raw = (
        json.dumps(
            bank.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    path = tmp_path / "center-bank.json"
    path.write_bytes(raw)
    assert B.load_banded_question_bank(
        path, expected_file_sha256=hashlib.sha256(raw).hexdigest()
    ) == bank
    with pytest.raises(ValueError, match="file SHA"):
        B.load_banded_question_bank(path, expected_file_sha256="0" * 64)


def test_bank_seals_32_arm_coverage_question_variants_and_offline_denominators():
    bank = _bank()
    payload = bank.canonical_payload
    assert payload["coverage"]["arm_keys"] == list(R.ARM_KEYS)
    assert len(payload["coverage"]["arm_keys"]) == 32
    assert payload["offline_solve_ledger"] == {
        "proposed_count": 1,
        "admitted_count": 1,
        "rejected_count": 0,
        "by_reason": [],
        "by_block": [
            {
                "block_key_sha256": bank.blocks[0].key_sha256,
                "proposed_count": 1,
                "admitted_count": 1,
                "rejected_count": 0,
                "by_reason": [],
            }
        ],
    }
    variants = payload["question_lineage"]["variants"]
    assert variants[0]["observation_variant"] == "A111"
    assert variants[0]["target_recipe"] == "current_lm"
    assert variants[0]["target_validity_mask"] == [True, True, True]
    assert variants[0]["supported_by_bank"] is True
    assert variants[1]["target_recipe"] == "outcome_dense_only"
    assert variants[1]["observation_variant"] == "C000"
    assert variants[1]["target_validity_mask"] == [False, False, False]
    assert variants[1]["supported_by_bank"] is False
    assert variants[1]["required_source"] == "immutable_tape"
    assert variants[0]["base_question_root_sha256"] == variants[1][
        "base_question_root_sha256"
    ]


def test_bank_rejects_missing_or_excess_declared_reachable_level_vector():
    source = _receipt()
    block = B.BandedQuestionBlock.from_receipts((source,))
    bank = _bank_from_blocks((block,))
    raw = bank.to_dict()
    raw["coverage"]["reachable_blocks"][0]["domain_levels"][
        "contact_y_upper"
    ] = 0.25
    raw["coverage"]["reachable_blocks"][0]["levels_sha256"] = (
        R.ActionDomainLevels(contact_y_upper=0.25).canonical_sha256
    )
    with pytest.raises(ValueError, match="exactly cover"):
        B.BandedQuestionBank.from_dict(raw)


def test_nested_lineage_is_immutable_and_base_identity_is_sealed():
    bank = _bank()
    before = bank.canonical_sha256
    with pytest.raises(TypeError):
        bank.producer_lineage["inputs"][0]["source_id"] = "changed"
    with pytest.raises(AttributeError):
        bank.coverage["reachable_blocks"].append({})
    assert bank.canonical_sha256 == before
    assert B._base_identity_sha256(
        base_yaw_rad=0.0,
        base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        base_spawn_w_m=(0.0, 0.0, 0.0),
    ) != B._base_identity_sha256(
        base_yaw_rad=-0.0,
        base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        base_spawn_w_m=(0.0, 0.0, 0.0),
    )


def test_solver_rejects_wrong_current_lm_identity_before_any_reset():
    source = _receipt()
    with pytest.raises(ValueError, match="configured current_lm solver"):
        B.BandedQuestionBankSolver(
            bank=_bank((source,)), solver_contract_sha256="0" * 64
        )


def test_pending_release_preflight_is_pure_and_checks_next_block():
    center = _receipt()
    levels = R.ActionDomainLevels(contact_y_upper=0.25)
    expanded = _fixture_call(FIXTURE._source_task, _birth_at_levels(levels))
    bank = _bank_from_blocks(
        (
            B.BandedQuestionBlock.from_receipts((center,)),
            B.BandedQuestionBlock.from_receipts((expanded,)),
        )
    )
    target = SimpleNamespace(
        phase="marginal",
        arm_frontier_indices=(0,) * 32,
        arm_probe_indices=tuple(
            1 if arm == "contact_y_upper" else 0 for arm in R.ARM_KEYS
        ),
        selected_arm_key="contact_y_upper",
        joint_probe_index=0,
        joint_rho_index=0,
    )
    pending = SimpleNamespace(
        key=SimpleNamespace(
            action_uid=center.action_uid,
            profile_sha256=center.profile_sha256,
            mobility=center.mobility_mode,
        ),
        release_id_sha256="f" * 64,
        target=target,
    )
    before = bank.canonical_sha256
    receipt = bank.preflight_pending_releases((pending,))
    assert receipt["release_count"] == 1
    assert receipt["releases"][0]["levels_sha256"] == levels.canonical_sha256
    assert receipt["online_solver_calls"] == 0
    assert bank.canonical_sha256 == before

    center_only = _bank((center,))
    with pytest.raises(ValueError, match="no unique exact next block"):
        center_only.preflight_pending_releases((pending,))
    assert center_only.canonical_sha256 == _bank((center,)).canonical_sha256


def test_global_reset_preflights_banded_release_before_first_mutation():
    source = (MDP_ROOT / "hope_commands.py").read_text(encoding="utf-8")
    block = source.split('if phase == "commit_global_reset":', 1)[1].split(
        'if phase == "after_global_reset":', 1
    )[0]

    pending = block.index("pending_releases = tuple(")
    preflight = block.index("bank.preflight_pending_releases(pending_releases)")
    drain = block.index("self._action_ball_force_drain_for_release()")
    barrier = block.index("issue_global_pre_reset_barrier()")
    commit = block.index("commit_release(")
    assert pending < preflight < drain < barrier < commit


def test_draft_center_materialization_has_zero_question_solve_and_pure_selection():
    source = _receipt()
    bank = _bank((source,))
    solver = B.BandedQuestionBankSolver(
        bank=bank,
        solver_contract_sha256=source.solver_sha256,
    )
    # The low-level solver has no authorization flag. Runtime/trainer admission
    # separately confines it to diagnostic construction until tape closure.
    birth = _birth(env_id=7, birth_index=9, generation=3)
    request = _request(birth, count=3, swing_start=4)
    batch = solver(request)

    assert len(batch.receipts) == 3
    assert batch.proposal_sample_indices == (0, 1, 2)
    assert [row.swing_generation for row in batch.receipts] == [4, 5, 6]
    assert solver.online_lm_calls == 0
    assert solver.physical_rng_draws == 0
    assert solver.sample_highwater_for(source.action_uid) == (
        2,
        batch.receipts[-1].sample_draw_end,
    )
    solver.assert_emitted_tasks(batch.receipts)
    solver.assert_emitted_sample(batch.receipts[0])

    block = bank.blocks[0]
    for receipt in batch.receipts:
        assert block.row_index_for(
            birth_sha256=receipt.birth_sha256,
            swing_generation=receipt.swing_generation,
            split_seed=bank.split_seed,
            base_yaw_rad=receipt.base_yaw_rad,
            base_quat_wxyz=receipt.base_quat_wxyz,
            base_spawn_w_m=receipt.base_spawn_w_m,
        ) == 0


def test_missing_curriculum_block_fails_closed_without_advancing_state():
    source = _receipt()
    solver = B.BandedQuestionBankSolver(
        bank=_bank((source,)),
        solver_contract_sha256=source.solver_sha256,
    )
    before = solver.state_dict()
    changed_levels = R.ActionDomainLevels(contact_y_upper=0.25)
    changed = _birth_at_levels(changed_levels)
    with pytest.raises(ValueError, match="no exact block"):
        solver(_request(changed))
    assert solver.state_dict() == before


def test_solver_state_is_per_block_not_an_emitted_receipt_history():
    source = _receipt()
    bank = _bank((source,))
    solver = B.BandedQuestionBankSolver(
        bank=bank,
        solver_contract_sha256=source.solver_sha256,
    )
    requests = tuple(
        _request(
            _birth(env_id=index, birth_index=index, generation=1),
            refill_index=index + 1,
            swing_start=index,
        )
        for index in range(64)
    )
    batches = solver.solve_many(requests)
    assert len(batches) == 64
    state = solver.state_dict()
    assert len(state["blocks"]) == 1
    assert state["blocks"][0]["cursor"] == 64
    assert state["blocks"][0]["emitted_count"] == 64
    serialized = json.dumps(state, sort_keys=True)
    assert "emitted_tasks" not in serialized
    assert "provider_history" not in serialized
    assert "birth_sha256" not in serialized

    restored = B.BandedQuestionBankSolver(
        bank=bank,
        solver_contract_sha256=source.solver_sha256,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.sample_highwater_for(source.action_uid) == (
        63,
        batches[-1].receipts[-1].sample_draw_end,
    )


def test_fixed_bank_cold_replay_preserves_rng_reason_and_counter_state():
    source = _receipt()
    bank = _bank((source,))
    first = B.BandedQuestionBankSolver(
        bank=bank, solver_contract_sha256=source.solver_sha256
    )
    first(_request(_birth(env_id=1, birth_index=1), count=2, swing_start=0))
    checkpoint = first.state_dict()
    cold = B.BandedQuestionBankSolver(
        bank=bank, solver_contract_sha256=source.solver_sha256
    )
    cold.load_state_dict(checkpoint)
    next_birth = _birth(env_id=2, birth_index=2)
    request = _request(next_birth, refill_index=2, count=3, swing_start=2)
    first_batch = first(request)
    cold_batch = cold(request)
    assert first_batch == cold_batch
    assert first.state_dict() == cold.state_dict()
    assert first.online_lm_calls == cold.online_lm_calls == 0
    assert first.physical_rng_draws == cold.physical_rng_draws == 0
    assert first.emitted_task_count_for(source.action_uid) == 5
    assert first_batch.proposed_count == len(first_batch.receipts) == 3

    missing = _birth_at_levels(R.ActionDomainLevels(contact_y_upper=0.25))
    before = first.state_dict()
    with pytest.raises(ValueError) as first_error:
        first(_request(missing, refill_index=3))
    with pytest.raises(ValueError) as cold_error:
        cold(_request(missing, refill_index=3))
    assert str(first_error.value) == str(cold_error.value)
    assert "no exact block" in str(first_error.value)
    assert first.state_dict() == cold.state_dict() == before


def test_row_selection_filters_exact_live_base_before_hashing():
    birth_a = _birth_at_base(
        env_id=1, birth_index=1, yaw=0.0, spawn=(-0.1, 0.0, 0.0)
    )
    birth_b = _birth_at_base(
        env_id=2, birth_index=2, yaw=0.0, spawn=(-0.2, 0.1, 0.0)
    )
    row_a = _fixture_call(FIXTURE._source_task, birth_a)
    row_b = _receipt_rebased_to_birth(row_a, birth_b)
    bank = _bank((row_a, row_b))
    block = bank.blocks[0]
    for birth, expected in ((birth_a, row_a), (birth_b, row_b)):
        for swing_generation in range(12):
            index = block.row_index_for(
                birth_sha256=birth.canonical_sha256,
                swing_generation=swing_generation,
                split_seed=bank.split_seed,
                sample_index=swing_generation,
                base_yaw_rad=birth.base_yaw_rad,
                base_quat_wxyz=birth.base_quat_wxyz,
                base_spawn_w_m=birth.base_spawn_w_m,
            )
            selected = block.rows[index]
            assert selected.base_yaw_rad == expected.base_yaw_rad
            assert selected.base_quat_wxyz == expected.base_quat_wxyz
            assert selected.base_spawn_w_m == expected.base_spawn_w_m

    solver = B.BandedQuestionBankSolver(
        bank=bank, solver_contract_sha256=row_a.solver_sha256
    )
    batches = solver.solve_many(
        (
            _request(birth_a, refill_index=1),
            _request(birth_b, refill_index=2),
        )
    )
    assert batches[0].receipts[0].base_spawn_w_m == birth_a.base_spawn_w_m
    assert batches[1].receipts[0].base_spawn_w_m == birth_b.base_spawn_w_m

    incompatible = _birth_at_base(
        env_id=3, birth_index=3, yaw=-0.3, spawn=(-0.4, -0.2, 0.0)
    )
    before = solver.state_dict()
    with pytest.raises(ValueError, match="exact live base identity"):
        solver(_request(incompatible, refill_index=3))
    assert solver.state_dict() == before


def test_pool_protocol_accepts_bounded_solver_owned_block_state():
    source = _receipt()
    bank = _bank((source,))
    solver = B.BandedQuestionBankSolver(
        bank=bank,
        solver_contract_sha256=source.solver_sha256,
    )
    pool = R.LazyActionTaskPool(
        (_binding(),),
        _pins(),
        "no_move",
        refill_size=4,
        diagnostic_unauthorized=False,
    )
    pool.bind_solver(solver)
    assert pool.diagnostic_fast_path is False
    assert pool._solver_delegates_birth_task_transcripts() is True
    pool._assert_all_task_transcripts_pure()

    state = pool.state_dict()
    restored_solver = B.BandedQuestionBankSolver(
        bank=bank,
        solver_contract_sha256=source.solver_sha256,
    )
    restored_pool = R.LazyActionTaskPool(
        (_binding(),),
        _pins(),
        "no_move",
        refill_size=4,
        diagnostic_unauthorized=False,
    )
    restored_pool.bind_solver(restored_solver)
    restored_pool.load_state_dict(state)
    assert restored_pool.state_dict() == state


def test_runtime_source_enum_and_three_way_solver_branch_are_present():
    text = (MDP_ROOT / "hope_commands.py").read_text(encoding="utf-8")
    assert '"banded_question_bank",' in text
    assert "load_banded_question_bank" in text
    assert "BandedQuestionBankSolver" in text
    assert "if banded_question_bank is not None:" in text
    assert "elif immutable_tape is None:" in text
    assert "must contain one exact center block" in text
    assert "DomainLevels().sha256" not in text
    assert "ActionDomainLevels().canonical_sha256" in text
    sampling = _load(
        "action_ball_sampling_center_sha_test_target",
        MDP_ROOT / "action_ball_sampling.py",
    )
    sampling_levels = sampling.DomainLevels()
    runtime_levels = R.ActionDomainLevels(**sampling_levels.as_dict())
    assert B._sha256_json(sampling_levels.as_dict()) == (
        runtime_levels.canonical_sha256
    )
    train = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert 'target_source == "banded_question_bank"' in train
    assert "load_banded_question_bank(" in train
    assert train.count('"action_ball_banded_question_bank_path"') >= 2
    assert train.count('"action_ball_banded_question_bank_sha256"') >= 2


def test_trainable_a211_uses_online_cache_and_c211_uses_direct_ball():
    train_path = ROOT / "scripts" / "train.py"
    tree = ast.parse(train_path.read_text(encoding="utf-8"), filename=str(train_path))
    finalizer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finalize_action_ball_training_cfg"
    )
    a211_branch = next(
        node
        for node in ast.walk(finalizer)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "a211_trainable"
    )
    assert len(a211_branch.orelse) == 1
    c211_branch = a211_branch.orelse[0]
    assert isinstance(c211_branch, ast.If)
    assert isinstance(c211_branch.test, ast.Name)
    assert c211_branch.test.id == "c211_trainable"

    assert any(
        isinstance(comparison, ast.Compare)
        and isinstance(comparison.left, ast.Name)
        and comparison.left.id == "target_source"
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.NotEq)
        and isinstance(comparison.comparators[0], ast.Constant)
        and comparison.comparators[0].value == "online_solver"
        for comparison in ast.walk(a211_branch)
    )
    assert any(
        isinstance(comparison, ast.Compare)
        and isinstance(comparison.left, ast.Name)
        and comparison.left.id == "target_recipe"
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.NotEq)
        and len(comparison.comparators) == 1
        and isinstance(comparison.comparators[0], ast.Constant)
        and comparison.comparators[0].value == "current_lm"
        for comparison in ast.walk(a211_branch)
    )
    assert any(
        isinstance(comparison, ast.Compare)
        and isinstance(comparison.left, ast.Name)
        and comparison.left.id == "target_source"
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.NotEq)
        and len(comparison.comparators) == 1
        and isinstance(comparison.comparators[0], ast.Constant)
        and comparison.comparators[0].value == "direct_ball"
        for comparison in ast.walk(c211_branch)
    )
    finalizer_source = train_path.read_text(encoding="utf-8")
    assert "banded_question_bank is construction-only" in finalizer_source
    assert "formal or expanding-curriculum training is" in finalizer_source
    assert "direct_ball is outcome_dense_only/000" in finalizer_source

    command_source = (MDP_ROOT / "hope_commands.py").read_text(encoding="utf-8")
    guard = command_source.index(
        "action_ball_target_source='banded_question_bank' is"
    )
    load = command_source.index("banded_question_bank = load_banded_question_bank(")
    assert guard < load
    assert '"diagnostic_unauthorized_required": True' in command_source
    assert '"zero_counter_scope": "target_question_solver_only"' in command_source
    assert '"birth_sampler_rng": "separate_tracked_mutable_state"' in command_source
