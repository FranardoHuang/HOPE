#!/usr/bin/env python3
"""Mutation tests for the C10 post-contact placement family contract.

The active family schema intentionally does not freeze actor/critic widths and
does not claim constructed-runtime or launch readiness.  The small ``_inputs``
fixture at the end is compatibility-only input for the separate legacy graph
extractor tests; the superseded builder must reject it.
"""

from dataclasses import fields as dataclass_fields, replace
import hashlib
import hmac
import json
from pathlib import Path
import struct
import sys

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
sys.path.insert(0, str(SOURCE_ROOT))

import action_ball_ac_family_contract as contract  # noqa: E402
import action_ball_landing_placement as c04  # noqa: E402
import action_ball_continuous_successor as c03  # noqa: E402
import action_ball_landing_outcome_mailbox as c05  # noqa: E402


_TARGET_PROFILE_SHA256 = "6" * 64
_TARGET_SELECTION_AUTHORITY_SHA256 = "a" * 64


def _runtime_passthrough(snapshot):
    return snapshot


def _other_runtime_passthrough(snapshot):
    return snapshot


def _placement_from_original_mailbox(mailbox_entry):
    return mailbox_entry


def _not_the_canonical_c04_scorer(profile, task_identity, facts):
    return c04.score_landing_placement(profile, task_identity, facts)


def _runtime_reads_family(snapshot):
    return snapshot.family


def _runtime_reads_live_job_id(snapshot):
    return snapshot.live_job_id


def _runtime_concatenates_family_key(snapshot):
    return snapshot["fam" + "ily"]


_STATIC_COMMON_SCALE = 1.0


def _runtime_reads_static_common_scale(snapshot):
    if _STATIC_COMMON_SCALE > 0.0:
        return snapshot
    return None


def _placement_reads_current_target(mailbox_entry, current_target):
    return mailbox_entry, current_target


def _placement_reads_command_metric(mailbox_entry, command_metric):
    return mailbox_entry, command_metric


def _placement_concatenates_current_target(mailbox_entry):
    return mailbox_entry["current" + "_target"]


def _placement_reads_target_alias(mailbox_entry):
    return mailbox_entry.target


def _placement_uses_command_name(mailbox_entry, **params):
    return mailbox_entry, params


def _callable(role, *, function=_runtime_passthrough, bound_params=None):
    return contract.ResolvedCallableInput(
        function=function,
        bound_params=(
            {"component": role} if bound_params is None else bound_params
        ),
        input_schema_id=contract.C10_COMMON_CALLABLE_INPUT_SCHEMAS[role],
    )


def _placement_profile():
    return c04.LandingPlacementProfile(
        frame_id="fixture_env_frame",
        frame_binding_sha256="9" * 64,
        contact_source_semantics=c04.SELECTED_RUBBER_CONTACT_AUTHORITY,
        table_surface_z_m=0.76,
        ball_radius_m=0.02,
        ball_center_landing_plane_z_m=0.78,
        net_x_m=1.87,
        net_mesh_top_z_m=0.9125,
        ball_center_net_clear_z_m=0.9325,
        opponent_table_x_min_m=1.87,
        opponent_table_x_max_m=3.24,
        table_y_min_m=-0.4,
        table_y_max_m=0.4,
        alpha_broad=0.4,
        sigma_broad_m=0.5,
        sigma_narrow_m=0.1,
        on_table_gate=1.0,
        off_table_gate=0.5,
    )


def _task_identities():
    profile = _placement_profile()

    def identity(target, receipt_digit, instance_digit):
        return c04.LandingPlacementTaskIdentity(
            frame_id=profile.frame_id,
            frame_binding_sha256=profile.frame_binding_sha256,
            profile_sha256=profile.canonical_sha256,
            task_receipt_sha256=receipt_digit * 64,
            semantic_binding_sha256="c" * 64,
            instance_binding_sha256=instance_digit * 64,
            target_x_m=target[0],
            target_y_m=target[1],
        )

    return (
        identity((2.50, 0.00), "b", "d"),
        identity((2.60, 0.20), "e", "f"),
        identity((2.70, -0.20), "7", "8"),
    )


def _observation():
    actor = (b"actor-precontact-0", b"actor-precontact-1")
    critic = (b"critic-precontact-0", b"critic-precontact-1")
    first_task, second_task, _ = _task_identities()
    return contract.C10UnfrozenObservationContract(
        actor_width=None,
        critic_width=None,
        width_status=contract.C10_OBSERVATION_WIDTH_STATUS,
        actor_layout_candidate_bytes=b"candidate-actor-layout-order-frame-source",
        critic_layout_candidate_bytes=b"candidate-critic-layout-order-frame-source",
        observation_provider=_callable("observation_provider"),
        desired_at_contact_provider=_callable("desired_at_contact_provider"),
        normalizer=_callable("normalizer"),
        normalizer_semantics=contract.C10_PRECONTACT_NORMALIZER_SEMANTICS,
        normalizer_state_bytes=b"same-fresh-normalizer-state",
        desired_at_contact_fact_tape=(
            contract.C10DesiredAtContactFact(
                point_index=0,
                task_sha256=first_task.task_receipt_sha256,
                position_world_m=(0.1, 0.2, 0.3),
                velocity_world_mps=(1.0, 2.0, 3.0),
                face_normal_world=(0.0, 1.0, 0.0),
                valid=True,
            ),
            contract.C10DesiredAtContactFact(
                point_index=1,
                task_sha256=second_task.task_receipt_sha256,
                position_world_m=(0.4, 0.5, 0.6),
                velocity_world_mps=(4.0, 5.0, 6.0),
                face_normal_world=(0.0, 1.0, 0.0),
                valid=True,
            ),
        ),
        desired_at_contact_validity_semantics=(
            "common_task_fact_no_treatment_mask_v1"
        ),
        actor_normalized_precontact_tape=actor,
        actor_policy_input_precontact_tape=actor,
        critic_normalized_precontact_tape=critic,
        critic_policy_input_precontact_tape=critic,
        precontact_family_mask_applied=False,
        precontact_treatment_slots=(),
    )


def _contact_rewards():
    return tuple(
        contract.C10ContactRewardTerm(
            name=name,
            reward=contract.ResolvedCallableInput(
                function=_runtime_passthrough,
                bound_params={"term": name},
                input_schema_id=contract.C10_CONTACT_REWARD_INPUT_SCHEMA,
            ),
            weight=1.0 + 0.1 * index,
            one_shot=True,
            same_transition=True,
            full_shot_keyed=True,
            payment_semantics=contract.C10_CONTACT_PAYMENT_SEMANTICS,
        )
        for index, name in enumerate(contract.C10_CONTACT_REWARD_TERM_NAMES)
    )


def _placement_source():
    profile = _placement_profile()
    return contract.C10PlacementSource(
        mailbox_reader=_callable(
            "placement_mailbox_reader",
            function=_placement_from_original_mailbox,
            bound_params={"mailbox": "landing_outcome_full_shot_key"},
        ),
        canonical_scorer=contract.ResolvedCallableInput(
            function=c04.score_landing_placement,
            bound_params={"profile_sha256": profile.canonical_sha256},
            input_schema_id=contract.C10_COMMON_CALLABLE_INPUT_SCHEMAS[
                "placement_canonical_scorer"
            ],
        ),
        canonical_profile=profile,
        source_semantics=contract.C10_PLACEMENT_SOURCE_SEMANTICS,
        eligibility_semantics=contract.C10_PLACEMENT_ELIGIBILITY,
        mailbox_kind="action_ball_landing_outcome_mailbox_full_shot_key_v1",
        selected_rubber_contact_required=True,
        after_the_fact_required=True,
    )


def _common():
    return contract.C10CommonRuntime(
        backend_binding_bytes=b"same-constructed-backend-binding-candidate",
        common_recipe_bytes=b"same-full-mdp-common-recipe",
        post_dt_budget_receipt_bytes=b"same-post-dt-budget-receipt-candidate",
        continuous_target_profile_sha256=_TARGET_PROFILE_SHA256,
        continuous_target_selection_authority_sha256=(
            _TARGET_SELECTION_AUTHORITY_SHA256
        ),
        continuous_target_runtime_dtype=c03.RUNTIME_TARGET_DTYPE,
        flight_horizon_steps=12,
        flight_slot_capacity=2,
        mailbox_slot_capacity=3,
        observation=_observation(),
        question_provider=_callable("question_provider"),
        selected_rubber_contact=_callable("selected_rubber_contact"),
        contact_rewards=_contact_rewards(),
        on_table_scorer=_callable("on_table_scorer"),
        on_table_success_weight=2.0,
        on_table_success_semantics=contract.C10_ON_TABLE_SUCCESS_SEMANTICS,
        success_denominator=_callable("success_denominator"),
        curriculum=_callable("curriculum"),
        common_on_table_outcome_consumer=_callable(
            "common_on_table_outcome_consumer",
            function=_placement_from_original_mailbox,
            bound_params={
                "consumer": "common_on_table_outcome",
                "mailbox": "c05_paid_original_shot",
            },
        ),
        post_contact_placement_raw_consumer=_callable(
            "post_contact_placement_raw_consumer",
            function=_placement_from_original_mailbox,
            bound_params={
                "consumer": "post_contact_placement_raw",
                "mailbox": "c05_paid_original_shot",
            },
        ),
        placement_source=_placement_source(),
        post_contact_placement_consumer_scheduled=True,
        post_contact_placement_manager_weight=1.0,
        plant_step=_callable("plant_step"),
        ppo_step=_callable("ppo_step"),
        recovery_contract_bytes=b"same-nonzero-recovery-and-ready-contract",
    )


def _shot_key(*, swing_generation, task_sha256, sample_digit):
    return c05.LandingOutcomeShotKey(
        env_id=7,
        reset_generation=2,
        swing_generation=swing_generation,
        action_uid=73,
        action_slot=0,
        birth_sha256="1" * 64,
        sample_sha256=sample_digit * 64,
        task_sha256=task_sha256,
        run_id="fresh-c10-fixture-run",
        carry_chain_id="fresh-c10-fixture-chain",
        shot_index=swing_generation + 1,
        source_sha256="8" * 64,
        config_sha256="9" * 64,
        receipt_content_sha256=str((int(sample_digit) + 1) % 10) * 64,
    )


def _shot_key_mapping(key):
    return key.full_key_dict()


def _float32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _target_selection(key, task_identity):
    runtime_x = _float32(task_identity.target_x_m)
    runtime_y = _float32(task_identity.target_y_m)
    return c03.TargetSelectionReceipt(
        profile_sha256=_TARGET_PROFILE_SHA256,
        selection_authority_sha256=_TARGET_SELECTION_AUTHORITY_SHA256,
        runtime_dtype=c03.RUNTIME_TARGET_DTYPE,
        target_generation=key.swing_generation,
        task_ref_sha256=(
            c03.ContinuousActionTaskReceiptRef.from_runtime_mapping(
                key.runtime_dict()
            ).canonical_sha256
        ),
        requested_target_x_m=task_identity.target_x_m,
        requested_target_y_m=task_identity.target_y_m,
        runtime_target_x_m=runtime_x,
        runtime_target_y_m=runtime_y,
        semantic_sha256=c03.target_semantic_sha256(
            _TARGET_PROFILE_SHA256,
            runtime_x,
            runtime_y,
        ),
    )


def _fixed_tape():
    profile = _placement_profile()
    first_task, second_task, third_task = _task_identities()
    first = _shot_key(
        swing_generation=3,
        task_sha256=first_task.task_receipt_sha256,
        sample_digit="2",
    )
    second = _shot_key(
        swing_generation=4,
        task_sha256=second_task.task_receipt_sha256,
        sample_digit="4",
    )
    third = _shot_key(
        swing_generation=5,
        task_sha256=third_task.task_receipt_sha256,
        sample_digit="5",
    )
    blocker = _shot_key(
        swing_generation=2,
        task_sha256=first_task.task_receipt_sha256,
        sample_digit="6",
    )
    first_target = _target_selection(first, first_task)
    second_target = _target_selection(second, second_task)
    third_target = _target_selection(third, third_task)
    first_facts = c04.LandingPlacementFacts(
        frame_id=profile.frame_id,
        profile_sha256=profile.canonical_sha256,
        task_identity_sha256=first_task.canonical_sha256,
        contact_valid=True,
        first_plane_crossing_valid=True,
        first_plane_crossing_nonfinite=False,
        first_plane_crossing_contract_fault=False,
        first_plane_crossing_x_m=first_task.target_x_m,
        first_plane_crossing_y_m=first_task.target_y_m,
        ball_center_net_crossed=True,
        ball_center_net_clear=True,
    )
    second_facts = c04.LandingPlacementFacts(
        frame_id=profile.frame_id,
        profile_sha256=profile.canonical_sha256,
        task_identity_sha256=second_task.canonical_sha256,
        contact_valid=False,
        first_plane_crossing_valid=False,
        first_plane_crossing_nonfinite=False,
        first_plane_crossing_contract_fault=False,
        first_plane_crossing_x_m=None,
        first_plane_crossing_y_m=None,
        ball_center_net_crossed=False,
        ball_center_net_clear=False,
    )
    first_score = c04.score_landing_placement(profile, first_task, first_facts)
    second_score = c04.score_landing_placement(profile, second_task, second_facts)
    mailbox = c05.LandingOutcomeMailbox(capacity=3)
    mailbox.open(
        task_key=first,
        profile=profile,
        task_identity=first_task,
        source_step=20,
        source_ball_center_xyz_m=(1.8, first_task.target_y_m, 1.1),
        flight_horizon_step=32,
        contact_valid=True,
    )
    first_previous = (1.8, first_task.target_y_m, 1.1)
    for step in range(21, 31):
        mailbox.observe_flight(
            task_key=first,
            profile=profile,
            task_identity=first_task,
            source_step=20,
            step=step,
            previous_ball_center_xyz_m=first_previous,
            current_ball_center_xyz_m=first_previous,
            ball_center_net_crossed=False,
            ball_center_net_clear=False,
            post_physics_descending_crossing_xy_m=None,
        )
    mailbox.observe_flight(
        task_key=first,
        profile=profile,
        task_identity=first_task,
        source_step=20,
        step=31,
        previous_ball_center_xyz_m=first_previous,
        current_ball_center_xyz_m=(
            2.6,
            first_task.target_y_m,
            1.05,
        ),
        ball_center_net_crossed=True,
        ball_center_net_clear=True,
        post_physics_descending_crossing_xy_m=(
            first_task.target_x_m,
            first_task.target_y_m,
        ),
    )
    mailbox.pay(
        task_key=first,
        profile=profile,
        task_identity=first_task,
        source_step=20,
        payment_step=31,
    )
    mailbox.open(
        task_key=second,
        profile=profile,
        task_identity=second_task,
        source_step=70,
        source_ball_center_xyz_m=(2.2, 0.0, 0.9),
        flight_horizon_step=82,
        contact_valid=False,
    )
    second_previous = (2.2, 0.0, 0.9)
    for step in range(71, 82):
        mailbox.observe_flight(
            task_key=second,
            profile=profile,
            task_identity=second_task,
            source_step=70,
            step=step,
            previous_ball_center_xyz_m=second_previous,
            current_ball_center_xyz_m=second_previous,
            ball_center_net_crossed=False,
            ball_center_net_clear=False,
            post_physics_descending_crossing_xy_m=None,
        )
    mailbox.observe_flight(
        task_key=second,
        profile=profile,
        task_identity=second_task,
        source_step=70,
        step=82,
        previous_ball_center_xyz_m=second_previous,
        current_ball_center_xyz_m=(2.3, 0.0, 0.85),
        ball_center_net_crossed=False,
        ball_center_net_clear=False,
        post_physics_descending_crossing_xy_m=None,
    )
    mailbox.pay(
        task_key=second,
        profile=profile,
        task_identity=second_task,
        source_step=70,
        payment_step=83,
    )
    mailbox_checkpoint = mailbox.to_checkpoint()
    midsequence_checkpoint_bytes = b"external-midsequence-slot-mailbox-state"
    midsequence_states = tuple(
        contract.C10MidsequenceStateSnapshot(
            state=state,
            full_shot_key=first,
            physics_stamp=contract.C10PhysicsStamp(
                control_step=control_step,
                physics_substep=physics_substep,
                event_phase=event_phase,
            ),
            flight_slot=(0 if state in ("INBOUND", "OPEN") else None),
            mailbox_slot=(None if state in ("INBOUND", "OPEN") else 0),
            physical_retired=state not in ("INBOUND", "OPEN"),
            ball_generation=first.swing_generation,
            flight_horizon_steps=12,
            original_target_x_m=first_task.target_x_m,
            original_target_y_m=first_task.target_y_m,
            profile_sha256=profile.canonical_sha256,
            common_outcome_consumed=common_consumed,
            placement_treatment_consumed=treatment_consumed,
        )
        for (
            state,
            control_step,
            physics_substep,
            event_phase,
            common_consumed,
            treatment_consumed,
        ) in (
            ("INBOUND", 19, 0, contract.C10_EVENT_PHASE_INBOUND, False, False),
            ("OPEN", 20, 0, contract.C10_EVENT_PHASE_CONTACT, False, False),
            (
                "SETTLED_UNPAID",
                31,
                0,
                contract.C10_EVENT_PHASE_LANDING,
                False,
                False,
            ),
            (
                "PARTIALLY_PAID",
                31,
                1,
                contract.C10_EVENT_PHASE_LANDING,
                True,
                False,
            ),
            ("PAID", 31, 2, contract.C10_EVENT_PHASE_LANDING, True, True),
        )
    )
    midsequence_root = contract.c10_midsequence_checkpoint_candidate_root(
        midsequence_states,
        midsequence_checkpoint_bytes,
    )
    return contract.C10FixedTape(
        tape_bytes=b"same-three-shot-fixed-tape-candidate",
        paired_replay_id=first.run_id,
        common_call_input_trace=(b"common-trace-0", b"common-trace-1"),
        no_contact_strike_shot_key=second,
        no_contact_desired_fact_index=1,
        no_contact_strike_fact_tick=70,
        no_contact_strike_payment_tick=70,
        no_contact_reward_fire_counts=(1,) * 10,
        no_contact_reward_payments=tuple(0.25 + index for index in range(10)),
        no_contact_selected_milestone_payment=0.0,
        no_contact_outcome_payment=0.0,
        on_table_success_values=(True, False),
        on_table_reward_values=(2.0, 0.0),
        common_on_table_consumer_fire_counts=(1, 1),
        placement_treatment_consumer_fire_counts=(1, 1),
        placement_points=(
            contract.C10PlacementTapePoint(
                shot_key=first,
                mailbox_shot_key=first,
                desired_fact_index=0,
                selected_rubber_contact=True,
                strike_fact_tick=20,
                selected_contact_stamp=contract.C10PhysicsStamp(
                    20, 0, contract.C10_EVENT_PHASE_CONTACT
                ),
                net_crossing_stamp=contract.C10PhysicsStamp(
                    20, 0, contract.C10_EVENT_PHASE_NET
                ),
                landing_stamp=contract.C10PhysicsStamp(
                    20, 0, contract.C10_EVENT_PHASE_LANDING
                ),
                physical_slot=0,
                mailbox_slot=0,
                physical_retired_at_settlement=True,
                ball_generation=first.swing_generation,
                settlement_tick=31,
                payment_tick=31,
                next_target_reveal_tick=25,
                next_precontact_observation_tick=25,
                next_strike_fact_tick=70,
                target_selection=first_target,
                task_identity=first_task,
                facts=first_facts,
                score=first_score,
                next_shot_key=second,
                next_target_selection=second_target,
                next_task_identity=second_task,
            ),
            contract.C10PlacementTapePoint(
                shot_key=second,
                mailbox_shot_key=second,
                desired_fact_index=1,
                selected_rubber_contact=False,
                strike_fact_tick=70,
                selected_contact_stamp=None,
                net_crossing_stamp=None,
                landing_stamp=None,
                physical_slot=1,
                mailbox_slot=1,
                physical_retired_at_settlement=True,
                ball_generation=second.swing_generation,
                settlement_tick=82,
                payment_tick=83,
                next_target_reveal_tick=90,
                next_precontact_observation_tick=90,
                next_strike_fact_tick=120,
                target_selection=second_target,
                task_identity=second_task,
                facts=second_facts,
                score=second_score,
                next_shot_key=third,
                next_target_selection=third_target,
                next_task_identity=third_task,
            ),
        ),
        landing_outcome_mailbox_checkpoint=mailbox_checkpoint,
        landing_outcome_mailbox_checkpoint_sha256=(
            mailbox_checkpoint["canonical_sha256"]
        ),
        flight_slot_capacity_witnesses=(
            contract.C10FlightSlotCapacityWitness(
                open_shot_key=first,
                next_shot_key=second,
                occupied_owners=(
                    contract.C10FlightSlotOwner(
                        shot_key=first,
                        slot=0,
                        state="OPEN",
                    ),
                ),
                assigned_next_slot=1,
                decision="ADMITTED",
            ),
            contract.C10FlightSlotCapacityWitness(
                open_shot_key=first,
                next_shot_key=second,
                occupied_owners=(
                    contract.C10FlightSlotOwner(
                        shot_key=first,
                        slot=0,
                        state="OPEN",
                    ),
                    contract.C10FlightSlotOwner(
                        shot_key=blocker,
                        slot=1,
                        state="INBOUND",
                    ),
                ),
                assigned_next_slot=None,
                decision="REJECTED_PRE_REVEAL_CAPACITY",
            ),
        ),
        midsequence_states=midsequence_states,
        midsequence_checkpoint_bytes=midsequence_checkpoint_bytes,
        midsequence_external_root_sha256=midsequence_root,
        restore_invokes_env_reset=False,
    )


def _checkpoint(**changes):
    value = contract.C10CheckpointLineage(
        initial_state_bytes=b"same-fresh-policy-state",
        optimizer_state_bytes=b"same-fresh-optimizer-state",
        normalizer_state_bytes=b"same-fresh-normalizer-state",
        rng_state_bytes=b"same-fresh-rng-state",
        resume_requested=False,
        parent_checkpoint_sha256=None,
        parent_abi_sha256=None,
    )
    return replace(value, **changes)


def _c10_inputs(family, **changes):
    value = contract.C10ResolvedRuntimeInputs(
        family=family,
        backend="isaac",
        identity={
            "live_job_id": "fresh-c10-%s-live-job" % family.lower(),
            "run_name": "fresh_c10_%s" % family.lower(),
            "namespace": "fresh_c10_%s_namespace" % family.lower(),
        },
        common=_common(),
        fixed_tape=_fixed_tape(),
        checkpoint=_checkpoint(),
        post_contact_placement_treatment_gain=1.0 if family == "A" else 0.0,
    )
    return replace(value, **changes)


def _projection(family, **changes):
    return contract.build_c10_family_projection(_c10_inputs(family, **changes))


def _replace_tuple(values, index, item):
    output = list(values)
    output[index] = item
    return tuple(output)


def test_c10_supersedes_245_353_and_leaves_widths_explicitly_unfrozen():
    assert contract.C10_SCHEMA_VERSION == 1
    assert contract.C10_CONTRACT_STATUS == "ACTIVE_SCHEMA_WIDTH_UNFROZEN_NO_LAUNCH"
    assert contract.C10_ACTOR_WIDTH is None
    assert contract.C10_CRITIC_WIDTH is None
    assert contract.C10_FAMILY_CONTRACT_MAPPING["actor_width"] is None
    assert contract.C10_FAMILY_CONTRACT_MAPPING["critic_width"] is None
    assert contract.C10_TYPED_FAMILY_SCHEMA_READY is True
    assert contract.C10_CONSTRUCTED_RUNTIME_READY is False
    assert contract.C10_FIXED_TAPE_VALUE_READY is False
    assert contract.C10_CALLABLE_NONINTERFERENCE_READY is False
    assert contract.C10_LAUNCH_GATE_READY is False
    assert contract.LAUNCH_GATE_READY is False
    assert contract.TYPED_SCHEMA_READY is False
    assert contract.RESOLVED_OBJECT_BUILDER_READY is False
    assert contract.FIXED_TAPE_SCHEMA_READY is False
    assert contract.COMPATIBILITY_245_353_LAYOUT_EXPORTS_ONLY is True


def test_old_portable_sha_is_immutable_superseded_evidence_only():
    assert contract.SUPERSEDED_ACTOR_WIDTH == 245
    assert contract.SUPERSEDED_CRITIC_WIDTH == 353
    assert contract.SUPERSEDED_PORTABLE_ABI_SHA256 == (
        "506078dae8eb6db1c02e6c48b25fcc3ed40c2ce3bec4711fb634a2df00e17382"
    )
    assert contract.C10_FAMILY_CONTRACT_MAPPING[
        "superseded_checkpoint_compatible"
    ] is False
    assert contract.C10_FAMILY_CONTRACT_SHA256 != (
        contract.SUPERSEDED_PORTABLE_ABI_SHA256
    )
    assert contract.SUPERSEDED_C07_ACTOR_WIDTH == 245
    assert contract.SUPERSEDED_C07_CRITIC_WIDTH == 353
    assert (
        contract.SUPERSEDED_C07_PORTABLE_ABI_SHA256
        == contract.SUPERSEDED_PORTABLE_ABI_SHA256
    )
    for tombstone in (
        contract.build_c07_family_projection,
        contract.validate_c07_family_pair,
    ):
        with pytest.raises(contract.ACFamilyContractError, match="SUPERSEDED"):
            tombstone(None)


def test_c10_parent_authority_is_sealed_and_identity_free():
    a = _projection("A").to_mapping()
    c = _projection("C").to_mapping()
    assert a["contract_authority_kind"] == contract.C10_CONTRACT_AUTHORITY_KIND
    assert a["contract_authority_sha256"] == contract.C10_CONTRACT_AUTHORITY_SHA256
    assert c["contract_authority_sha256"] == a["contract_authority_sha256"]
    assert contract.canonical_sha256(contract.C10_CONTRACT_AUTHORITY_PAYLOAD) == (
        contract.C10_CONTRACT_AUTHORITY_SHA256
    )
    authority_text = repr(contract.C10_CONTRACT_AUTHORITY_PAYLOAD)
    assert "run_name" not in authority_text
    assert "namespace" not in authority_text
    assert "output_dir" not in authority_text


def test_fixed_tape_c05_identity_is_shared_replay_source_not_live_job_identity():
    a = _projection("A").to_mapping()
    c = _projection("C").to_mapping()
    assert a["identity"] != c["identity"]
    a_tape = a["common_runtime"]["fixed_tape"]
    c_tape = c["common_runtime"]["fixed_tape"]
    assert a_tape["paired_replay_source_sha256"] == c_tape[
        "paired_replay_source_sha256"
    ]
    assert contract.C10_CONTRACT_AUTHORITY_PAYLOAD[
        "fixed_tape_c05_identity_scope"
    ] == "shared_paired_replay_id_distinct_from_required_live_job_id_v2"

    inputs = _c10_inputs("A")
    source_run_id = inputs.fixed_tape.placement_points[0].shot_key.run_id
    with pytest.raises(contract.ACFamilyContractError, match="required live_job_id"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                identity={
                    "live_job_id": source_run_id,
                    "run_name": "independent-live-run",
                    "namespace": "independent-live-namespace",
                },
            )
        )
    with pytest.raises(contract.ACFamilyContractError, match="typed live_job_id"):
        contract.build_c10_family_projection(replace(inputs, identity={}))


def test_c10_parent_requires_nonempty_post_dt_budget_receipt():
    inputs = _c10_inputs("A")
    common = replace(inputs.common, post_dt_budget_receipt_bytes=b"")
    with pytest.raises(contract.ACFamilyContractError, match="post_dt_budget"):
        contract.build_c10_family_projection(replace(inputs, common=common))


def test_valid_c10_pair_has_exact_common_core_and_only_postcontact_treatment_delta():
    a = _projection("A")
    c = _projection("C")
    pair = contract.validate_c10_family_pair(a, c)
    a_map = a.to_mapping()
    c_map = c.to_mapping()
    assert a_map["common_runtime"] == c_map["common_runtime"]
    assert a_map["checkpoint_lineage"] == c_map["checkpoint_lineage"]
    assert a_map["abi_state"] == c_map["abi_state"]
    assert a_map["treatment"]["post_contact_placement_treatment_gain"] == 1.0
    assert c_map["treatment"]["post_contact_placement_treatment_gain"] == 0.0
    assert a_map["common_runtime"]["contract"][
        "post_contact_placement_consumer_scheduled"
    ] is True
    assert a_map["common_runtime"]["contract"][
        "post_contact_placement_manager_weight"
    ] > 0.0
    assert a_map["treatment_witness"]["guidance_payment_nonzero_count"] == 1
    assert c_map["treatment_witness"]["guidance_payment_all_zero"] is True
    assert pair.allowed_delta_paths == contract.C10_ALLOWED_DELTA_PATHS
    assert pair.common_runtime_sha256
    assert pair.evidence_level == contract.C10_EVIDENCE_LEVEL
    assert pair.launch_gate_ready is False
    assert a_map["evidence_level"] == (
        "typed_schema_candidate_no_constructed_capability_v1"
    )


def test_profile_and_manager_numbers_remain_unfrozen_launch_holds():
    authority = contract.C10_CONTRACT_AUTHORITY_PAYLOAD
    assert authority["placement_profile_parameters_status"].startswith("UNFROZEN")
    assert authority["reward_manager_weights_status"].startswith("UNFROZEN")
    assert contract.C10_FIXED_TAPE_VALUE_READY is False
    assert contract.C10_PHYSICS_STAMP_RUNTIME_READY is False
    assert contract.C10_SLOT_CAPACITY_RUNTIME_READY is False
    assert contract.C10_MIDSEQUENCE_CHECKPOINT_RUNTIME_READY is False
    assert contract.C10_LAUNCH_GATE_READY is False
    assert authority["capacity_profile_fields"] == [
        "flight_horizon_steps",
        "flight_slot_capacity",
        "mailbox_slot_capacity",
    ]
    assert authority["flight_slot_ownership_states"] == ["INBOUND", "OPEN"]
    assert authority["mailbox_slot_ownership_states"] == [
        "SETTLED_UNPAID",
        "PARTIALLY_PAID",
        "PAID",
    ]
    assert "retire_physical" in authority["slot_transfer_semantics"]
    assert authority["landing_placement_torch_authority_source_sha256"] == (
        "a1e8c41089ff7373b2befdf6b6e7719ba315e7987c445633af179cee3d237c4d"
    )
    assert authority["landing_placement_torch_authority_test_sha256"] == (
        "ef9b93c0283d439447092b08eba9a8700bc3ba1ae8389c754d3fb7c37398c86e"
    )
    assert authority["landing_placement_authority_source_sha256"] == (
        "3e2e056336a8c021c20bd255c474487cb2346a3dcdfcca8a1b1a608dd90636e2"
    )


def test_physics_stamp_orders_same_control_substep_by_fixed_event_phase():
    mapping = _projection("A").to_mapping()
    point = mapping["common_runtime"]["fixed_tape"]["placement_points"][0]
    stamps = (
        point["selected_contact_stamp"],
        point["net_crossing_stamp"],
        point["landing_stamp"],
    )
    assert {(item["control_step"], item["physics_substep"]) for item in stamps} == {
        (20, 0)
    }
    assert tuple(item["event_phase"] for item in stamps) == (
        contract.C10_EVENT_PHASE_CONTACT,
        contract.C10_EVENT_PHASE_NET,
        contract.C10_EVENT_PHASE_LANDING,
    )

    inputs = _c10_inputs("A")
    first = inputs.fixed_tape.placement_points[0]
    changed = replace(
        first,
        net_crossing_stamp=replace(first.net_crossing_stamp, control_step=19),
    )
    with pytest.raises(contract.ACFamilyContractError, match="CONTACT < NET < LANDING"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                fixed_tape=replace(
                    inputs.fixed_tape,
                    placement_points=_replace_tuple(
                        inputs.fixed_tape.placement_points, 0, changed
                    ),
                ),
            )
        )


def test_k_slot_witness_retains_open_old_ball_and_rejects_full_pool_before_reveal():
    mapping = _projection("A").to_mapping()
    common = mapping["common_runtime"]["contract"]
    assert common["flight_slot_capacity"] == 2
    assert common["mailbox_slot_capacity"] == 3
    witnesses = mapping["common_runtime"]["fixed_tape"][
        "flight_slot_capacity_witnesses"
    ]
    assert tuple(item["decision"] for item in witnesses) == (
        "ADMITTED",
        "REJECTED_PRE_REVEAL_CAPACITY",
    )
    inputs = _c10_inputs("A")
    rejected = replace(
        inputs.fixed_tape.flight_slot_capacity_witnesses[1],
        occupied_owners=(
            inputs.fixed_tape.flight_slot_capacity_witnesses[1].occupied_owners[0],
        ),
    )
    with pytest.raises(contract.ACFamilyContractError, match="capacity exhaustion"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                fixed_tape=replace(
                    inputs.fixed_tape,
                    flight_slot_capacity_witnesses=(
                        inputs.fixed_tape.flight_slot_capacity_witnesses[0],
                        rejected,
                    ),
                ),
            )
        )

    admitted = inputs.fixed_tape.flight_slot_capacity_witnesses[0]
    missing_old_owner = replace(admitted, occupied_owners=())
    with pytest.raises(contract.ACFamilyContractError, match="exact OPEN owner"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                fixed_tape=replace(
                    inputs.fixed_tape,
                    flight_slot_capacity_witnesses=(
                        missing_old_owner,
                        inputs.fixed_tape.flight_slot_capacity_witnesses[1],
                    ),
                ),
            )
        )


def test_flight_and_mailbox_capacities_and_ownership_states_cannot_be_conflated():
    inputs = _c10_inputs("A")
    with pytest.raises(contract.ACFamilyContractError, match="separate mailbox"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                common=replace(inputs.common, mailbox_slot_capacity=2),
            )
        )
    settled = inputs.fixed_tape.midsequence_states[2]
    illegal = replace(
        settled,
        flight_slot=0,
        physical_retired=False,
    )
    states = _replace_tuple(inputs.fixed_tape.midsequence_states, 2, illegal)
    checkpoint_bytes = inputs.fixed_tape.midsequence_checkpoint_bytes
    root = contract.c10_midsequence_checkpoint_candidate_root(
        states,
        checkpoint_bytes,
    )
    with pytest.raises(contract.ACFamilyContractError, match="live only in mailbox"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                fixed_tape=replace(
                    inputs.fixed_tape,
                    midsequence_states=states,
                    midsequence_external_root_sha256=root,
                ),
            )
        )


@pytest.mark.parametrize(
    "change,match",
    (
        ({"midsequence_external_root_sha256": "0" * 64}, "external root pin"),
        ({"restore_invokes_env_reset": True}, "without env.reset"),
        ({"midsequence_states": ()}, "exact ordered state set"),
    ),
)
def test_midsequence_checkpoint_capability_is_external_pinned_and_reset_free(
    change, match
):
    inputs = _c10_inputs("A")
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(
            replace(inputs, fixed_tape=replace(inputs.fixed_tape, **change))
        )


@pytest.mark.parametrize(
    "common_change,point_change,match",
    (
        ({"flight_horizon_steps": 13}, {}, "common profile horizon"),
        ({"flight_slot_capacity": 1}, {}, "flight slot capacity"),
        ({}, {"ball_generation": 99}, "ball_generation"),
    ),
)
def test_flight_horizon_slot_and_ball_generation_are_bound(
    common_change, point_change, match
):
    inputs = _c10_inputs("A")
    points = inputs.fixed_tape.placement_points
    if point_change:
        points = _replace_tuple(points, 0, replace(points[0], **point_change))
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(
            replace(
                inputs,
                common=replace(inputs.common, **common_change),
                fixed_tape=replace(inputs.fixed_tape, placement_points=points),
            )
        )


@pytest.mark.parametrize(
    "change,match",
    (
        ({"post_contact_placement_consumer_scheduled": False}, "both A and C"),
        ({"post_contact_placement_manager_weight": 0.0}, "common and positive"),
    ),
)
def test_placement_manager_term_is_scheduled_and_positive_for_both_families(
    change, match
):
    inputs = _c10_inputs("C")
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(
            replace(inputs, common=replace(inputs.common, **change))
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "common_on_table_consumer_fire_counts",
        "placement_treatment_consumer_fire_counts",
    ),
)
def test_c06_common_outcome_and_placement_treatment_consumers_are_distinct_once(
    field_name,
):
    mapping = _projection("C").to_mapping()
    tape = mapping["common_runtime"]["fixed_tape"]
    assert tape["placement_dual_consumer_names"] == list(
        contract.C10_PLACEMENT_CONSUMER_NAMES
    )
    assert len(set(tape["common_on_table_consumer_ledger_keys"])) == 2
    assert len(set(tape["placement_treatment_consumer_ledger_keys"])) == 2
    assert tape["common_on_table_consumer_ledger_keys"] != tape[
        "placement_treatment_consumer_ledger_keys"
    ]
    inputs = _c10_inputs("C")
    with pytest.raises(contract.ACFamilyContractError, match="exactly one consumption"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                fixed_tape=replace(
                    inputs.fixed_tape,
                    **{field_name: (1, 0)},
                ),
            )
        )


def test_all_ten_contact_rewards_are_ordered_positive_one_shot_and_common():
    mapping = _projection("A").to_mapping()
    terms = mapping["common_runtime"]["contract"]["contact_rewards"]
    assert tuple(term["name"] for term in terms) == (
        contract.C10_CONTACT_REWARD_TERM_NAMES
    )
    assert len(terms) == 10
    assert all(term["weight"] > 0.0 for term in terms)
    assert mapping["common_runtime"]["fixed_tape"][
        "ordered_strike_reward_names"
    ] == list(contract.C10_CONTACT_REWARD_TERM_NAMES)
    assert contract.C10_CONTACT_REWARD_TERM_NAMES == (
        "racket_position",
        "racket_velocity",
        "racket_normal",
        "racket_position_coarse",
        "racket_velocity_coarse",
        "racket_normal_coarse",
        "racket_position_precision",
        "racket_velocity_precision",
        "racket_normal_precision",
        "paddle_center_proximity",
    )
    assert mapping["common_runtime"]["contract"][
        "contact_reward_order_authority_sha256"
    ] == contract.C10_STRIKE_FACT_CONSUMER_ORDER_SHA256


def test_c_precontact_zero_mask_is_rejected_before_pair_validation():
    inputs = _c10_inputs("C")
    observation = inputs.common.observation
    zeroed = bytes(len(observation.actor_policy_input_precontact_tape[0]))
    bad_observation = replace(
        observation,
        actor_policy_input_precontact_tape=_replace_tuple(
            observation.actor_policy_input_precontact_tape,
            0,
            zeroed,
        ),
    )
    with pytest.raises(contract.ACFamilyContractError, match="zero-mask is forbidden"):
        contract.build_c10_family_projection(
            replace(inputs, common=replace(inputs.common, observation=bad_observation))
        )


def test_explicit_precontact_family_mask_or_treatment_slot_is_rejected():
    inputs = _c10_inputs("C")
    for observation in (
        replace(inputs.common.observation, precontact_family_mask_applied=True),
        replace(
            inputs.common.observation,
            precontact_treatment_slots=("desired_at_contact_valid",),
        ),
        replace(
            inputs.common.observation,
            desired_at_contact_validity_semantics="placement_treatment_valid_v1",
        ),
    ):
        with pytest.raises(
            contract.ACFamilyContractError,
            match="pre-contact|pre/contact|validity",
        ):
            contract.build_c10_family_projection(
                replace(inputs, common=replace(inputs.common, observation=observation))
            )


def test_any_declared_actor_or_critic_width_is_rejected_until_constructed_runtime():
    inputs = _c10_inputs("A")
    for field_name in ("actor_width", "critic_width"):
        observation = replace(inputs.common.observation, **{field_name: 245})
        with pytest.raises(contract.ACFamilyContractError, match="explicit None"):
            contract.build_c10_family_projection(
                replace(inputs, common=replace(inputs.common, observation=observation))
            )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("opaque", "C10DesiredAtContactFact"),
        ("all_invalid", "at least one valid"),
        ("bad_shape", "exactly 3"),
        ("bad_face", "unit world-frame"),
        ("bad_index", "contiguous ordered"),
        ("misaligned", "one-for-one"),
    ),
)
def test_desired_at_contact_is_typed_3_plus_3_plus_3_and_tape_aligned(
    mutation, match
):
    inputs = _c10_inputs("A")
    observation = inputs.common.observation
    facts = observation.desired_at_contact_fact_tape
    if mutation == "opaque":
        observation = replace(
            observation,
            desired_at_contact_fact_tape=(b"opaque-nine-ish-bytes", facts[1]),
        )
    elif mutation == "all_invalid":
        observation = replace(
            observation,
            desired_at_contact_fact_tape=tuple(
                replace(
                    fact,
                    position_world_m=(0.0, 0.0, 0.0),
                    velocity_world_mps=(0.0, 0.0, 0.0),
                    face_normal_world=(0.0, 0.0, 0.0),
                    valid=False,
                )
                for fact in facts
            ),
        )
    elif mutation == "bad_shape":
        observation = replace(
            observation,
            desired_at_contact_fact_tape=(
                replace(facts[0], position_world_m=(0.0, 1.0)),
                facts[1],
            ),
        )
    elif mutation == "bad_face":
        observation = replace(
            observation,
            desired_at_contact_fact_tape=(
                replace(facts[0], face_normal_world=(0.0, 2.0, 0.0)),
                facts[1],
            ),
        )
    elif mutation == "bad_index":
        observation = replace(
            observation,
            desired_at_contact_fact_tape=(
                replace(facts[0], point_index=2),
                facts[1],
            ),
        )
    else:
        observation = replace(
            observation,
            desired_at_contact_fact_tape=(facts[0],),
        )
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(
            replace(inputs, common=replace(inputs.common, observation=observation))
        )


def test_hidden_wait_invalid_desired_fact_uses_positive_zero_and_is_allowed():
    inputs = _c10_inputs("A")
    observation = inputs.common.observation
    hidden = contract.C10DesiredAtContactFact(
        point_index=2,
        task_sha256="0" * 64,
        position_world_m=(0.0, 0.0, 0.0),
        velocity_world_mps=(0.0, 0.0, 0.0),
        face_normal_world=(0.0, 0.0, 0.0),
        valid=False,
    )
    observation = replace(
        observation,
        desired_at_contact_fact_tape=observation.desired_at_contact_fact_tape
        + (hidden,),
        actor_normalized_precontact_tape=observation.actor_normalized_precontact_tape
        + (b"hidden-wait",),
        actor_policy_input_precontact_tape=observation.actor_policy_input_precontact_tape
        + (b"hidden-wait",),
        critic_normalized_precontact_tape=observation.critic_normalized_precontact_tape
        + (b"hidden-wait",),
        critic_policy_input_precontact_tape=observation.critic_policy_input_precontact_tape
        + (b"hidden-wait",),
    )
    projection = contract.build_c10_family_projection(
        replace(inputs, common=replace(inputs.common, observation=observation))
    )
    assert projection.to_mapping()["common_runtime"]["contract"]["observation"][
        "desired_at_contact_point_count"
    ] == 3


@pytest.mark.parametrize(
    "mutation",
    (
        "layout",
        "values",
        "desired_values",
        "provider",
        "desired_provider",
        "normalizer",
    ),
)
def test_a_c_observation_provider_normalizer_or_values_may_not_differ(mutation):
    c_inputs = _c10_inputs("C")
    observation = c_inputs.common.observation
    if mutation == "layout":
        observation = replace(
            observation,
            actor_layout_candidate_bytes=b"different-actor-order",
        )
    elif mutation == "values":
        changed = (b"different-common-value",) + observation.actor_normalized_precontact_tape[1:]
        observation = replace(
            observation,
            actor_normalized_precontact_tape=changed,
            actor_policy_input_precontact_tape=changed,
        )
    elif mutation == "desired_values":
        first = observation.desired_at_contact_fact_tape[0]
        observation = replace(
            observation,
            desired_at_contact_fact_tape=(
                replace(first, position_world_m=(9.0, 9.0, 9.0)),
                observation.desired_at_contact_fact_tape[1],
            ),
        )
    elif mutation == "provider":
        observation = replace(
            observation,
            observation_provider=_callable(
                "observation_provider",
                function=_other_runtime_passthrough,
            ),
        )
    elif mutation == "desired_provider":
        observation = replace(
            observation,
            desired_at_contact_provider=_callable(
                "desired_at_contact_provider",
                function=_other_runtime_passthrough,
            ),
        )
    else:
        observation = replace(
            observation,
            normalizer=_callable("normalizer", function=_other_runtime_passthrough),
        )
    c = contract.build_c10_family_projection(
        replace(c_inputs, common=replace(c_inputs.common, observation=observation))
    )
    with pytest.raises(contract.ACFamilyContractError, match="common_runtime"):
        contract.validate_c10_family_pair(_projection("A"), c)


def test_contact_payment_difference_is_not_a_family_treatment():
    c_inputs = _c10_inputs("C")
    payments = _replace_tuple(c_inputs.fixed_tape.no_contact_reward_payments, 0, 99.0)
    c = contract.build_c10_family_projection(
        replace(c_inputs, fixed_tape=replace(c_inputs.fixed_tape, no_contact_reward_payments=payments))
    )
    with pytest.raises(contract.ACFamilyContractError, match="common_runtime"):
        contract.validate_c10_family_pair(_projection("A"), c)


def test_ten_strike_payments_have_an_explicit_legal_no_contact_witness():
    inputs = _c10_inputs("A")
    selected_point = inputs.fixed_tape.placement_points[0]
    tape = replace(
        inputs.fixed_tape,
        no_contact_strike_shot_key=selected_point.shot_key,
        no_contact_desired_fact_index=0,
        no_contact_strike_fact_tick=selected_point.strike_fact_tick,
        no_contact_strike_payment_tick=selected_point.strike_fact_tick,
    )
    with pytest.raises(contract.ACFamilyContractError, match="legal no-contact"):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))


@pytest.mark.parametrize(
    "tape_change,match",
    (
        ({"no_contact_strike_payment_tick": 71}, "same post-physics transition"),
        (
            {"no_contact_strike_fact_tick": 0, "no_contact_strike_payment_tick": 0},
            "exact-fact transition tick",
        ),
        ({"no_contact_reward_fire_counts": (1,) * 9 + (2,)}, "exactly once"),
        ({"no_contact_reward_payments": (0.0,) + tuple(1.0 for _ in range(9))}, "all ten positive"),
        ({"no_contact_selected_milestone_payment": 1.0}, "milestone"),
        ({"no_contact_outcome_payment": 1.0}, "outcome"),
    ),
)
def test_ten_contact_payments_require_keyed_same_tick_once_only_witness(
    tape_change, match
):
    inputs = _c10_inputs("A")
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(
            replace(inputs, fixed_tape=replace(inputs.fixed_tape, **tape_change))
        )


@pytest.mark.parametrize("mutation", ("weight", "provider", "one_shot", "order"))
def test_contact_reward_definition_difference_is_not_allowed(mutation):
    c_inputs = _c10_inputs("C")
    terms = c_inputs.common.contact_rewards
    if mutation == "weight":
        terms = _replace_tuple(terms, 0, replace(terms[0], weight=9.0))
    elif mutation == "provider":
        terms = _replace_tuple(
            terms,
            0,
            replace(
                terms[0],
                reward=replace(terms[0].reward, function=_other_runtime_passthrough),
            ),
        )
    elif mutation == "one_shot":
        terms = _replace_tuple(terms, 0, replace(terms[0], one_shot=False))
    else:
        terms = (terms[1], terms[0]) + terms[2:]
    changed = replace(c_inputs, common=replace(c_inputs.common, contact_rewards=terms))
    if mutation in ("one_shot", "order"):
        with pytest.raises(contract.ACFamilyContractError, match="one-shot|must be"):
            contract.build_c10_family_projection(changed)
    else:
        c = contract.build_c10_family_projection(changed)
        with pytest.raises(contract.ACFamilyContractError, match="common_runtime"):
            contract.validate_c10_family_pair(_projection("A"), c)


def test_family_cannot_enter_any_common_callable_source_or_bound_params():
    inputs = _c10_inputs("A")
    for provider in (
        _callable("question_provider", function=_runtime_reads_family),
        _callable("question_provider", function=_runtime_reads_live_job_id),
        _callable("question_provider", function=_runtime_concatenates_family_key),
        _callable("question_provider", bound_params={"mode": "A"}),
        _callable("question_provider", bound_params={"family": "shared"}),
    ):
        with pytest.raises(
            contract.ACFamilyContractError,
            match="family|identity|live_job_id",
        ):
            contract.build_c10_family_projection(
                replace(inputs, common=replace(inputs.common, question_provider=provider))
            )


def test_referenced_primitive_globals_are_value_hashed_in_common_callable_binding():
    global _STATIC_COMMON_SCALE

    original = _STATIC_COMMON_SCALE
    try:
        a_inputs = _c10_inputs("A")
        a_provider = _callable(
            "question_provider",
            function=_runtime_reads_static_common_scale,
        )
        a = contract.build_c10_family_projection(
            replace(a_inputs, common=replace(a_inputs.common, question_provider=a_provider))
        )
        _STATIC_COMMON_SCALE = 2.0
        c_inputs = _c10_inputs("C")
        c_provider = _callable(
            "question_provider",
            function=_runtime_reads_static_common_scale,
        )
        c = contract.build_c10_family_projection(
            replace(c_inputs, common=replace(c_inputs.common, question_provider=c_provider))
        )
        with pytest.raises(
            contract.ACFamilyContractError,
            match="common_runtime|typed-input revalidation",
        ):
            contract.validate_c10_family_pair(a, c)
    finally:
        _STATIC_COMMON_SCALE = original


@pytest.mark.parametrize(
    "function,match",
    (
        (_placement_reads_current_target, "current target|mailbox_entry"),
        (_placement_reads_command_metric, "command metric|mailbox_entry"),
    ),
)
def test_placement_treatment_cannot_read_current_target_or_command_metric(function, match):
    inputs = _c10_inputs("A")
    source = replace(
        inputs.common.placement_source,
        mailbox_reader=_callable(
            "placement_mailbox_reader",
            function=function,
            bound_params={"mailbox": "landing_outcome_full_shot_key"},
        ),
    )
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(
            replace(inputs, common=replace(inputs.common, placement_source=source))
        )


@pytest.mark.parametrize(
    "function",
    (_placement_concatenates_current_target, _placement_reads_target_alias),
)
def test_placement_cannot_hide_current_target_behind_concat_or_target_alias(function):
    inputs = _c10_inputs("A")
    source = replace(
        inputs.common.placement_source,
        mailbox_reader=_callable(
            "placement_mailbox_reader",
            function=function,
            bound_params={"mailbox": "landing_outcome_full_shot_key"},
        ),
    )
    with pytest.raises(contract.ACFamilyContractError, match="target|dynamic string"):
        contract.build_c10_family_projection(
            replace(inputs, common=replace(inputs.common, placement_source=source))
        )


@pytest.mark.parametrize(
    "bound_params",
    (
        {"source": "current_target_xy"},
        {"command_name": "racket_target"},
        {"metric": "landing"},
    ),
)
def test_placement_bound_params_cannot_alias_current_question_state(bound_params):
    inputs = _c10_inputs("A")
    source = replace(
        inputs.common.placement_source,
        mailbox_reader=_callable(
            "placement_mailbox_reader",
            function=_placement_uses_command_name,
            bound_params=bound_params,
        ),
    )
    with pytest.raises(
        contract.ACFamilyContractError,
        match="original-shot|current target/command metric",
    ):
        contract.build_c10_family_projection(
            replace(inputs, common=replace(inputs.common, placement_source=source))
        )


def test_placement_mailbox_must_match_original_full_shot_key():
    inputs = _c10_inputs("A")
    point = inputs.fixed_tape.placement_points[0]
    changed = replace(
        point,
        mailbox_shot_key=replace(point.shot_key, sample_sha256="9" * 64),
    )
    tape = replace(
        inputs.fixed_tape,
        placement_points=_replace_tuple(inputs.fixed_tape.placement_points, 0, changed),
    )
    with pytest.raises(contract.ACFamilyContractError, match="original C05 full shot key"):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))


def test_one_full_shot_key_cannot_receive_duplicate_placement_payment():
    inputs = _c10_inputs("A")
    first = inputs.fixed_tape.placement_points[0]
    tape = replace(
        inputs.fixed_tape,
        on_table_success_values=(True, True),
        on_table_reward_values=(2.0, 2.0),
        placement_points=(first, first),
    )
    with pytest.raises(contract.ACFamilyContractError, match="exactly one|fixed placement cohort"):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))


def test_placement_must_settle_and_pay_after_original_strike_fact():
    inputs = _c10_inputs("A")
    point = replace(inputs.fixed_tape.placement_points[0], settlement_tick=20)
    tape = replace(
        inputs.fixed_tape,
        placement_points=_replace_tuple(inputs.fixed_tape.placement_points, 0, point),
    )
    with pytest.raises(contract.ACFamilyContractError, match="original strike fact"):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))


def test_old_shot_can_settle_and_pay_after_next_target_reveal_under_original_key():
    point = _c10_inputs("A").fixed_tape.placement_points[0]
    assert point.next_target_reveal_tick < point.settlement_tick
    mapping = _projection("A").to_mapping()
    resolved = mapping["common_runtime"]["fixed_tape"]["placement_points"][0]
    assert resolved["next_target_reveal_tick"] < resolved["settlement_tick"]
    assert resolved["mailbox_payment_idempotency_sha256"]


@pytest.mark.parametrize("which", ("current", "next"))
def test_float32_target_generation_must_match_its_c05_swing_generation(which):
    inputs = _c10_inputs("A")
    point = inputs.fixed_tape.placement_points[0]
    if which == "current":
        point = replace(
            point,
            target_selection=replace(
                point.target_selection,
                target_generation=point.shot_key.swing_generation + 9,
            ),
        )
    else:
        point = replace(
            point,
            next_target_selection=replace(
                point.next_target_selection,
                target_generation=point.next_shot_key.swing_generation + 9,
            ),
        )
    with pytest.raises(contract.ACFamilyContractError, match="generation"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                fixed_tape=replace(
                    inputs.fixed_tape,
                    placement_points=_replace_tuple(
                        inputs.fixed_tape.placement_points, 0, point
                    ),
                ),
            )
        )


def test_successor_target_must_be_visible_in_precontact_policy_time():
    inputs = _c10_inputs("A")
    point = replace(
        inputs.fixed_tape.placement_points[0],
        next_target_reveal_tick=70,
        next_precontact_observation_tick=70,
    )
    with pytest.raises(contract.ACFamilyContractError, match="strictly before"):
        contract.build_c10_family_projection(
            replace(
                inputs,
                fixed_tape=replace(
                    inputs.fixed_tape,
                    placement_points=_replace_tuple(
                        inputs.fixed_tape.placement_points, 0, point
                    ),
                ),
            )
        )


@pytest.mark.parametrize("mutation", ("same_task", "bad_swing"))
def test_continuous_successor_is_new_and_generation_ordered(mutation):
    inputs = _c10_inputs("A")
    point = inputs.fixed_tape.placement_points[0]
    if mutation == "same_task":
        next_key = replace(
            point.next_shot_key,
            task_sha256=point.task_identity.task_receipt_sha256,
        )
        point = replace(
            point,
            next_task_identity=point.task_identity,
            next_shot_key=next_key,
            next_target_selection=_target_selection(next_key, point.task_identity),
        )
        match = "newly sampled"
    else:
        next_key = replace(
            point.next_shot_key,
            swing_generation=point.shot_key.swing_generation + 2,
        )
        point = replace(
            point,
            next_shot_key=next_key,
            next_target_selection=_target_selection(
                next_key,
                point.next_task_identity,
            ),
        )
        match = "swing_generation"
    tape = replace(
        inputs.fixed_tape,
        placement_points=_replace_tuple(inputs.fixed_tape.placement_points, 0, point),
    )
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))


def test_adjacent_targets_must_differ_after_canonical_float32_cast():
    inputs = _c10_inputs("A")
    point = inputs.fixed_tape.placement_points[0]
    near_duplicate = replace(
        point.next_task_identity,
        target_x_m=point.task_identity.target_x_m + 1.0e-10,
        target_y_m=point.task_identity.target_y_m,
    )
    next_key = point.next_shot_key
    point = replace(
        point,
        next_task_identity=near_duplicate,
        next_shot_key=next_key,
        next_target_selection=_target_selection(next_key, near_duplicate),
    )
    tape = replace(
        inputs.fixed_tape,
        placement_points=_replace_tuple(inputs.fixed_tape.placement_points, 0, point),
    )
    with pytest.raises(contract.ACFamilyContractError, match="newly sampled"):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))


def test_c05_mailbox_checkpoint_requires_an_external_authority_sha():
    inputs = _c10_inputs("A")
    tape = replace(
        inputs.fixed_tape,
        landing_outcome_mailbox_checkpoint_sha256="0" * 64,
    )
    with pytest.raises(contract.ACFamilyContractError, match="C05 mailbox checkpoint"):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))


def test_placement_score_is_canonical_c04_reexecution_not_caller_reported():
    inputs = _c10_inputs("A")
    original = inputs.fixed_tape.placement_points[0]
    profile = inputs.common.placement_source.canonical_profile
    different_facts = replace(
        original.facts,
        first_plane_crossing_x_m=original.task_identity.target_x_m + 0.2,
    )
    different_score = c04.score_landing_placement(
        profile,
        original.task_identity,
        different_facts,
    )
    point = replace(original, score=different_score)
    tape = replace(
        inputs.fixed_tape,
        placement_points=_replace_tuple(inputs.fixed_tape.placement_points, 0, point),
    )
    with pytest.raises(contract.ACFamilyContractError, match="C04 scorer reexecution"):
        contract.build_c10_family_projection(replace(inputs, fixed_tape=tape))
    source = _projection("A").to_mapping()["common_runtime"]["contract"][
        "placement_source"
    ]
    assert source["score_authority"] == (
        "canonical_c04_profile_task_facts_score_reexecution_v1"
    )


def test_c04_profile_freezes_cauchy_gaussian_and_table_gates():
    source = _projection("A").to_mapping()["common_runtime"]["contract"][
        "placement_source"
    ]
    profile = source["canonical_profile"]
    assert profile["cauchy_definition"] == c04.CAUCHY_DEFINITION
    assert profile["gaussian_definition"] == c04.GAUSSIAN_DEFINITION
    assert profile["on_table_gate"] == 1.0
    assert profile["off_table_gate"] == 0.5


def test_selected_rubber_contact_is_required_for_nonzero_placement_payment():
    inputs = _c10_inputs("A")
    first = inputs.fixed_tape.placement_points[0]
    points = _replace_tuple(
        inputs.fixed_tape.placement_points,
        0,
        replace(first, selected_rubber_contact=False),
    )
    with pytest.raises(
        contract.ACFamilyContractError,
        match="contact authority",
    ):
        contract.build_c10_family_projection(
            replace(inputs, fixed_tape=replace(inputs.fixed_tape, placement_points=points))
        )


@pytest.mark.parametrize(
    "source_change,match",
    (
        (
            {
                "canonical_scorer": contract.ResolvedCallableInput(
                    function=_not_the_canonical_c04_scorer,
                    bound_params={
                        "profile_sha256": _placement_profile().canonical_sha256
                    },
                    input_schema_id=contract.C10_COMMON_CALLABLE_INPUT_SCHEMAS[
                        "placement_canonical_scorer"
                    ],
                )
            },
            "canonical C04",
        ),
        (
            {
                "canonical_scorer": replace(
                    _placement_source().canonical_scorer,
                    bound_params={"profile_sha256": "0" * 64},
                )
            },
            "exact profile SHA",
        ),
        ({"selected_rubber_contact_required": False}, "selected-rubber"),
        ({"after_the_fact_required": False}, "after-the-fact"),
    ),
)
def test_placement_source_semantics_are_closed(source_change, match):
    inputs = _c10_inputs("A")
    source = replace(inputs.common.placement_source, **source_change)
    with pytest.raises(contract.ACFamilyContractError, match=match):
        contract.build_c10_family_projection(
            replace(inputs, common=replace(inputs.common, placement_source=source))
        )


@pytest.mark.parametrize("family,gain", (("A", 0.0), ("A", 2.0), ("C", 1.0)))
def test_only_family_axis_has_exact_a_one_c_zero_treatment_gain(family, gain):
    with pytest.raises(contract.ACFamilyContractError, match="A=1 and C=0"):
        contract.build_c10_family_projection(
            _c10_inputs(
                family,
                post_contact_placement_treatment_gain=gain,
            )
        )


@pytest.mark.parametrize(
    "field_name",
    ("on_table_scorer", "success_denominator", "curriculum"),
)
def test_on_table_scorer_denominator_and_curriculum_are_exact_common(field_name):
    c_inputs = _c10_inputs("C")
    original = getattr(c_inputs.common, field_name)
    common = replace(
        c_inputs.common,
        **{
            field_name: replace(
                original,
                function=_other_runtime_passthrough,
            )
        },
    )
    c = contract.build_c10_family_projection(replace(c_inputs, common=common))
    with pytest.raises(contract.ACFamilyContractError, match="common_runtime"):
        contract.validate_c10_family_pair(_projection("A"), c)


def test_common_on_table_payment_cannot_be_hidden_in_family_treatment():
    c_inputs = _c10_inputs("C")
    values = _replace_tuple(c_inputs.fixed_tape.on_table_reward_values, 0, 0.0)
    with pytest.raises(contract.ACFamilyContractError, match="common scorer/weight"):
        contract.build_c10_family_projection(
            replace(
                c_inputs,
                fixed_tape=replace(
                    c_inputs.fixed_tape,
                    on_table_reward_values=values,
                ),
            )
        )


def test_observation_and_checkpoint_normalizer_state_must_be_the_same_bytes():
    inputs = _c10_inputs("A")
    checkpoint = replace(
        inputs.checkpoint,
        normalizer_state_bytes=b"different-fresh-normalizer-state",
    )
    with pytest.raises(contract.ACFamilyContractError, match="normalizer state"):
        contract.build_c10_family_projection(
            replace(inputs, checkpoint=checkpoint)
        )


def test_superseded_checkpoint_and_any_resume_are_rejected():
    with pytest.raises(contract.ACFamilyContractError, match="245/353 checkpoint"):
        contract.build_c10_family_projection(
            _c10_inputs(
                "A",
                checkpoint=_checkpoint(
                    resume_requested=True,
                    parent_checkpoint_sha256="7" * 64,
                    parent_abi_sha256=contract.SUPERSEDED_PORTABLE_ABI_SHA256,
                ),
            )
        )
    with pytest.raises(contract.ACFamilyContractError, match="only fresh no-parent"):
        contract.build_c10_family_projection(
            _c10_inputs("A", checkpoint=_checkpoint(resume_requested=True))
        )


def test_identity_only_namespaces_may_differ_but_backend_or_recipe_may_not():
    contract.validate_c10_family_pair(_projection("A"), _projection("C"))
    c_inputs = _c10_inputs("C")
    for changed in (
        replace(c_inputs, backend="mujoco"),
        replace(
            c_inputs,
            common=replace(c_inputs.common, common_recipe_bytes=b"different-recipe"),
        ),
    ):
        with pytest.raises(contract.ACFamilyContractError, match="backend|common_runtime"):
            contract.validate_c10_family_pair(
                _projection("A"),
                contract.build_c10_family_projection(changed),
            )


def test_direct_or_tampered_projection_cannot_enter_pair_gate():
    forged = contract.C10ResolvedProjection(
        _payload_json=b"{}",
        _auth_tag=b"fake",
        _token=object(),
        _source_inputs=_c10_inputs("A"),
    )
    with pytest.raises(contract.ACFamilyContractError, match="must come from"):
        contract.validate_c10_family_pair(forged, _projection("C"))
    a = _projection("A")
    tampered = replace(a, _payload_json=a._payload_json + b" ")
    with pytest.raises(contract.ACFamilyContractError, match="authentication"):
        contract.validate_c10_family_pair(tampered, _projection("C"))

    c = _projection("C")
    forged_payload = c.to_mapping()
    forged_payload["treatment"]["post_contact_placement_treatment_gain"] = 1.0
    forged_payload_json = json.dumps(
        forged_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    readable_key_forgery = replace(
        c,
        _payload_json=forged_payload_json,
        _auth_tag=hmac.new(
            contract._C10_BUILD_AUTH_KEY,  # noqa: SLF001 - hostile mutation test
            forged_payload_json,
            hashlib.sha256,
        ).digest(),
        _token=contract._C10_BUILD_TOKEN,  # noqa: SLF001 - hostile mutation test
    )
    with pytest.raises(contract.ACFamilyContractError, match="typed-input revalidation"):
        contract.validate_c10_family_pair(_projection("A"), readable_key_forgery)


def test_artifact_is_audit_only_and_launch_false():
    artifact = contract.c10_projection_artifact(_projection("A"))
    assert artifact["projection"]["launch_gate_ready"] is False
    assert artifact["projection"]["abi_state"]["actor_width"] is None
    with pytest.raises(contract.ACFamilyContractError, match="must come from"):
        contract.validate_c10_family_pair(artifact["projection"], _projection("C"))


# ---------------------------------------------------------------------------
# Compatibility-only fixture for test_action_ball_ac_runtime_projection.py.
# No old projection can be built from it after C10.
# ---------------------------------------------------------------------------

_OLD_COMMON_INPUT_SCHEMAS = {
    "question_provider": "action_ball_common_question_v1",
    "common_snapshot": "action_ball_frozen_common_snapshot_input_v1",
    "common_observation_pack": "action_ball_frozen_common_snapshot_v1",
    "normalizer": "action_ball_masked_observation_pair_v1",
    "guide_projection": "action_ball_normalized_observation_and_guide_switch_v1",
    "common_reward": "action_ball_common_reward_facts_v1",
    "guide_reward": "action_ball_desired_contact_facts_and_weight_v1",
    "exact_contact": "action_ball_full_key_strike_fact_v1",
    "landing_outcome": "action_ball_full_key_landing_fact_v1",
    "recovery_termination": "action_ball_common_lifecycle_snapshot_v1",
    "plant_step": "action_ball_common_plant_step_v1",
    "ppo_step": "action_ball_common_rollout_batch_v1",
}


def _old_common():
    return contract.CommonRuntimeCallables(
        **{
            field.name: contract.ResolvedCallableInput(
                function=_runtime_passthrough,
                bound_params={"component": field.name},
                input_schema_id=_OLD_COMMON_INPUT_SCHEMAS[field.name],
            )
            for field in dataclass_fields(contract.CommonRuntimeCallables)
        }
    )


def _inputs(family):
    """Superseded typed input used only to exercise runtime fail-closed seams."""

    empty_group = contract.ResolvedObservationGroup(terms=())
    return contract.ResolvedRuntimeInputs(
        family=family,
        backend="isaac",
        backend_binding_bytes=b"superseded-binding",
        initial_normalizer_state=contract.FreshNormalizerState(
            actor_sample_count=0,
            critic_sample_count=0,
            actor_mean_f32=b"x",
            actor_m2_f32=b"x",
            critic_mean_f32=b"x",
            critic_m2_f32=b"x",
            epsilon=1.0e-8,
            clip=10.0,
        ),
        fresh_checkpoint_state=contract.FreshCheckpointState(
            policy_state_bytes=b"x",
            optimizer_state_bytes=b"x",
            rng_state_bytes=b"x",
            rollout_state_bytes=b"x",
            parent_checkpoint_sha256=None,
            resume_requested=False,
        ),
        fixed_inter_shot_cadence_ticks=1,
        policy_dt_seconds=0.02,
        identity={"run_name": "superseded_%s" % family.lower()},
        actor=empty_group,
        critic=empty_group,
        common=_old_common(),
        fixed_tape=contract.FixedTapeBytes(
            tape_bytes=b"x",
            actor_raw_pre_treatment=(),
            critic_raw_pre_treatment=(),
            actor_normalized_pre_treatment=(),
            critic_normalized_pre_treatment=(),
            actor_final=(),
            critic_final=(),
            common_reward=(),
            guide_reward=(),
            termination=(),
            lifecycle_phase=(),
            call_input_trace=(),
            outcome_atomicity=contract.OutcomeAtomicityWitness(
                previous_reveal_tick=0,
                open_observation_tick=1,
                settlement_tick=2,
                payment_tick=2,
                close_tick=2,
                closed_observation_tick=3,
                next_reveal_tick=4,
                next_open_tick=4,
                observed_mailbox_states=("OPEN", "CLOSED", "OPEN"),
                ordered_events=("OPEN", "CLOSE", "OPEN"),
            ),
        ),
        guide_enabled=family == "A",
        guide_reward_weight=1.0 if family == "A" else 0.0,
    )


def test_superseded_builder_rejects_compatibility_fixture():
    with pytest.raises(contract.ACFamilyContractError, match="SUPERSEDED"):
        contract.build_resolved_runtime_projection(_inputs("A"))
