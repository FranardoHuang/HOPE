from __future__ import annotations

import base64
import copy
from dataclasses import MISSING, fields, replace
import hashlib
import inspect
import math
from pathlib import Path
import struct
import sys

import pytest


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import action_ball_physical_flight_contract as C  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_full_mdp_reveal_boundary_v4_pins_are_exact():
    assert C.FULL_MDP_REVEAL_BOUNDARY_SOURCE_SHA256 == (
        "a5762b2e4838a3bdc58c2a30822467d27e4fb1006a37fcc3faf3948f7c2c24fe"
    )
    assert C.FULL_MDP_REVEAL_BOUNDARY_PACKET_SCHEMA_VERSION == 4
    assert C.FULL_MDP_REVEAL_BOUNDARY_ROW_INTEGRITY_SCHEMA_SHA256 == (
        "cfc212a4ef2fd2078df99114c28f55df93b0605e0a126049b24b07fc636b16aa"
    )
    assert C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256 == (
        "4e715720b741991905d7c6cf8aa5ddf6c5a1e617773b6132aa33368468736cdd"
    )


def _seal(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["canonical_sha256"] = C.canonical_sha256(payload)
    return result


def _reseal(value: dict[str, object]) -> dict[str, object]:
    payload = {key: item for key, item in value.items() if key != "canonical_sha256"}
    return _seal(payload)


def _pin(
    source_kind: str,
    *,
    schema_version: int = 1,
    source_schema_sha256: str | None = None,
    **payload: object,
) -> C.CanonicalJsonContentPin:
    source = _seal(
        {
            "schema_version": schema_version,
            "kind": source_kind,
            **payload,
        }
    )
    return C.CanonicalJsonContentPin.from_sealed_mapping(
        source,
        expected_source_kind=source_kind,
        source_schema_sha256=(
            source_schema_sha256 or _sha("schema:" + source_kind)
        ),
    )


def _capacity(
    *, cadence: int = 5, horizon: int = 10
) -> C.FrozenFlightCapacityReceipt:
    capacity = horizon // cadence + 1
    fixed_tape_sha = _sha(f"fixed-tape:{cadence}:{horizon}")
    clock_root = _sha("control-step-clock-root")
    authority = _pin(
        "fixture_flight_capacity_numeric_authority_v1",
        clock_kind="constructed_control_step_v1",
        control_step_clock_root_sha256=clock_root,
        cadence_control_steps=cadence,
        max_flight_horizon_control_steps=horizon,
        flight_capacity=capacity,
        inclusive_interval_semantics=C.INCLUSIVE_INTERVAL_SEMANTICS,
        same_tick_ordering=C.SAME_TICK_ORDERING,
        fixed_tape_sha256=fixed_tape_sha,
        source_sha256=_sha("capacity-source"),
        config_sha256=_sha("capacity-config"),
        contract_sha256=_sha("capacity-contract"),
    )
    return C.FrozenFlightCapacityReceipt(
        integration_status=C.INTEGRATION_STATUS,
        numeric_authority=authority,
        fixed_tape_sha256=fixed_tape_sha,
        clock_kind="constructed_control_step_v1",
        control_step_clock_root_sha256=clock_root,
        cadence_control_steps=cadence,
        max_flight_horizon_control_steps=horizon,
        inclusive_interval_semantics=C.INCLUSIVE_INTERVAL_SEMANTICS,
        same_tick_ordering=C.SAME_TICK_ORDERING,
        capacity_formula=C.CAPACITY_FORMULA,
        required_inclusive_flight_capacity=capacity,
        configured_flight_capacity=capacity,
    )


def _task(*, env_id: int = 0, swing: int = 3) -> C.PhysicalFlightTaskRef:
    return C.PhysicalFlightTaskRef(
        env_id=env_id,
        reset_generation=7,
        swing_generation=swing,
        action_uid=101 + env_id,
        action_slot=2,
        birth_sha256=_sha(f"birth:{env_id}"),
        sample_sha256=_sha(f"sample:{env_id}:{swing}"),
        task_sha256=_sha(f"task:{env_id}:{swing}"),
    )


def _outcome(*, env_id: int = 0, swing: int = 3) -> C.PhysicalFlightOutcomeKey:
    task = _task(env_id=env_id, swing=swing)
    return C.PhysicalFlightOutcomeKey(
        env_id=task.env_id,
        reset_generation=task.reset_generation,
        swing_generation=task.swing_generation,
        action_uid=task.action_uid,
        action_slot=task.action_slot,
        birth_sha256=task.birth_sha256,
        sample_sha256=task.sample_sha256,
        task_sha256=task.task_sha256,
        run_id="fixture-run",
        carry_chain_id=f"fixture-chain-{env_id}",
        shot_index=swing + 1,
        source_sha256=_sha("outcome-source"),
        config_sha256=_sha("outcome-config"),
        receipt_content_sha256=_sha(f"outcome-receipt:{env_id}:{swing}"),
    )


def _state() -> C.CanonicalPhysicalBallStateF32:
    return C.CanonicalPhysicalBallStateF32(
        position_env_m=(-0.0, 1.0, 0.75),
        quaternion_wxyz=(1.0, 0.0, -0.0, 0.0),
        linear_velocity_world_mps=(3.5, -0.25, 0.125),
        angular_velocity_world_radps=(0.0, 12.0, -7.0),
    )


def _install(
    *,
    capacity: C.FrozenFlightCapacityReceipt | None = None,
    env_id: int = 0,
    slot: int = 1,
    swing: int = 3,
    reveal: int = 7,
) -> C.PhysicalBallInstallPayload:
    capacity = capacity or _capacity()
    task = _task(env_id=env_id, swing=swing)
    outcome = _outcome(env_id=env_id, swing=swing)
    state = _state()
    deadline = reveal + min(2, capacity.max_flight_horizon_control_steps)
    horizon = reveal + capacity.max_flight_horizon_control_steps
    if deadline <= reveal:
        raise ValueError("fixture install requires H > 0")
    frame = _pin(
        "fixture_env_origin_transform_receipt_v1",
        frame_id="fixture_env_local_world_aligned",
        env_id=env_id,
        env_origin_world_f32_be_hex=["00000000", "00000000", "00000000"],
        transform_semantics=C.POSITION_FRAME,
    )
    state_binding = C.installed_ball_state_binding_sha256(
        state_f32_sha256=state.state_bytes_sha256,
        frame_id="fixture_env_local_world_aligned",
        frame_binding_sha256=frame.source_canonical_sha256,
        env_id=env_id,
        reveal_control_step=reveal,
        selected_contact_deadline_control_step=deadline,
        first_crossing_horizon_control_step=horizon,
        task_ref_sha256=task.canonical_sha256,
        outcome_key_sha256=outcome.canonical_sha256,
    )
    return C.PhysicalBallInstallPayload(
        integration_status=C.INTEGRATION_STATUS,
        capacity_receipt=capacity,
        capacity_receipt_sha256=capacity.canonical_sha256,
        env_id=env_id,
        flight_slot=slot,
        ball_generation=swing,
        task_ref=task,
        task_ref_sha256=task.canonical_sha256,
        outcome_key=outcome,
        outcome_key_sha256=outcome.canonical_sha256,
        ball_construction_receipt_sha256=_sha(
            f"construction:{env_id}:{swing}"
        ),
        inbound_ball_sha256=_sha(f"inbound:{env_id}:{swing}"),
        frame_id="fixture_env_local_world_aligned",
        frame_binding_authority=frame,
        frame_binding_sha256=frame.source_canonical_sha256,
        position_frame=C.POSITION_FRAME,
        quaternion_order=C.QUATERNION_ORDER,
        linear_velocity_frame=C.LINEAR_VELOCITY_FRAME,
        angular_velocity_frame=C.ANGULAR_VELOCITY_FRAME,
        state_epoch=C.INSTALL_STATE_EPOCH,
        reveal_control_step=reveal,
        selected_contact_deadline_control_step=deadline,
        first_crossing_horizon_control_step=horizon,
        state_f32=state,
        state_f32_sha256=state.state_bytes_sha256,
        installed_ball_state_sha256=state_binding,
    )


def _parked(
    capacity: C.FrozenFlightCapacityReceipt,
    *,
    env_id: int,
    slot: int,
    version: int,
) -> C.PhysicalFlightSlotSnapshot:
    return C.PhysicalFlightSlotSnapshot(
        capacity_receipt_sha256=capacity.canonical_sha256,
        capacity_value=capacity.configured_flight_capacity,
        env_id=env_id,
        slot_index=slot,
        scene_body_name=f"physical_ball_env{env_id}_slot{slot}",
        lifecycle=C.SLOT_PARKED,
        ball_generation=None,
        inbound_ball_sha256=None,
        outcome_key=None,
        outcome_key_sha256=None,
        install_payload_sha256=None,
        installed_ball_state_sha256=None,
        current_state_f32=None,
        current_state_f32_sha256=None,
        reveal_control_step=None,
        last_control_step=0,
        last_physics_substep=0,
        last_sim_step=0,
        mutation_version=version,
        physically_parked=True,
        published_to_runtime=False,
    )


def _live(
    install: C.PhysicalBallInstallPayload, *, version: int
) -> C.PhysicalFlightSlotSnapshot:
    return C.PhysicalFlightSlotSnapshot(
        capacity_receipt_sha256=install.capacity_receipt_sha256,
        capacity_value=install.capacity_receipt.configured_flight_capacity,
        env_id=install.env_id,
        slot_index=install.flight_slot,
        scene_body_name=(
            f"physical_ball_env{install.env_id}_slot{install.flight_slot}"
        ),
        lifecycle=C.SLOT_IN_FLIGHT,
        ball_generation=install.ball_generation,
        inbound_ball_sha256=install.inbound_ball_sha256,
        outcome_key=install.outcome_key,
        outcome_key_sha256=install.outcome_key_sha256,
        install_payload_sha256=install.canonical_sha256,
        installed_ball_state_sha256=install.installed_ball_state_sha256,
        current_state_f32=install.state_f32,
        current_state_f32_sha256=install.state_f32_sha256,
        reveal_control_step=install.reveal_control_step,
        last_control_step=install.reveal_control_step,
        last_physics_substep=0,
        last_sim_step=0,
        mutation_version=version,
        physically_parked=False,
        published_to_runtime=True,
    )


def _preview_mapping(
    rows: tuple[tuple[C.PhysicalBallInstallPayload, C.PhysicalFlightSlotSnapshot], ...]
) -> dict[str, object]:
    def r05_slot(
        slot: C.PhysicalFlightSlotSnapshot,
    ) -> dict[str, object]:
        if slot.lifecycle == C.SLOT_PARKED:
            values = {
                "slot_index": slot.slot_index,
                "lifecycle_state": "empty",
                "physical_retired": True,
                "owner_key_sha256": None,
                "ball_generation": None,
                "inbound_ball_sha256": None,
                "dynamic_state_sha256": None,
            }
        else:
            values = {
                "slot_index": slot.slot_index,
                "lifecycle_state": "inbound",
                "physical_retired": False,
                "owner_key_sha256": slot.outcome_key_sha256,
                "ball_generation": slot.ball_generation,
                "inbound_ball_sha256": slot.inbound_ball_sha256,
                "dynamic_state_sha256": slot.current_state_f32_sha256,
            }
        return _seal(
            {
                "schema_version": 1,
                "kind": C.R05_BALL_SLOT_SNAPSHOT_KIND,
                **values,
            }
        )

    preview_rows = []
    for install, pre_slot in rows:
        capacity = install.capacity_receipt.configured_flight_capacity
        before_slots = tuple(
            _parked(
                install.capacity_receipt,
                env_id=install.env_id,
                slot=slot_index,
                version=pre_slot.mutation_version,
            )
            for slot_index in range(capacity)
        )
        after_slots = list(before_slots)
        after_slots[install.flight_slot] = _live(
            install,
            version=pre_slot.mutation_version + 1,
        )
        r05_before = tuple(r05_slot(slot) for slot in before_slots)
        r05_after = tuple(r05_slot(slot) for slot in after_slots)
        preserved = tuple(
            slot["owner_key_sha256"]
            for index, slot in enumerate(r05_before)
            if index != install.flight_slot
            and slot["owner_key_sha256"] is not None
            and slot["physical_retired"] is False
        )
        plan = _seal(
            {
                "schema_version": 2,
                "kind": C.R05_BALL_SLOT_PLAN_KIND,
                "capacity": capacity,
                "snapshot_sha256": C.canonical_sha256(list(r05_before)),
                "selected_slot_index": install.flight_slot,
                "previous_slot_index": None,
                "reused_previous_slot": False,
                "preserved_live_owner_key_sha256": list(preserved),
                "new_ball_generation": install.ball_generation,
                "new_inbound_ball_sha256": install.inbound_ball_sha256,
                "new_ball_dynamic_state_sha256": install.state_f32_sha256,
                "physical_ball_install_payload_sha256": (
                    install.canonical_sha256
                ),
                "reused_retired_owner_key_sha256": None,
            }
        )
        preview_rows.append(
            _seal(
                {
                    "schema_version": 2,
                    "kind": C.R05_REVEAL_FINAL_INSTALL_ROW_KIND,
                    "integration_status": C.INTEGRATION_STATUS,
                    "phase": "REVEAL_FINAL_PREVIEWED",
                    "public_visible": False,
                    "policy_opportunity_created": False,
                "reveal_facts": {
                    "env_id": install.env_id,
                    "reveal_step": install.reveal_control_step,
                    "deadline_step": install.selected_contact_deadline_control_step,
                },
                    "ball_slot_plan": plan,
                "prepared_reveal": {
                    "selected_task_ref": install.task_ref.to_mapping(),
                    "outcome_key": install.outcome_key.to_mapping(),
                },
                    "pre_install_ball_slots": list(r05_before),
                    "post_install_ball_slots": list(r05_after),
                "physical_ball_install_payload_sha256": install.canonical_sha256,
                "selected_task_ref_sha256": install.task_ref_sha256,
                "outcome_key_sha256": install.outcome_key_sha256,
                }
            )
        )
    return _seal(
        {
            "schema_version": 2,
            "kind": C.REVEAL_FINAL_PREVIEW_KIND,
            "integration_status": C.INTEGRATION_STATUS,
            "phase": "REVEAL_FINAL_PREVIEWED",
            "public_visible": False,
            "policy_opportunity_created": False,
            "owner_checkpoint_before_sha256": _sha("r05-owner-before"),
            "prepared_batch": {"fixture": "private-prepared-batch"},
            "sampler_checkpoint_before_commit_sha256": _sha(
                "sampler-before"
            ),
            "sampler_checkpoint_after_commit_sha256": _sha(
                "sampler-after"
            ),
            "untouched_rows_before_sha256": _sha("untouched-before"),
            "untouched_rows_after_sha256": _sha("untouched-before"),
            "sampler_checkpoint_before_commit": {"fixture": "before"},
            "sampler_checkpoint_after_commit": {"fixture": "after"},
            "reveal_final_rows": preview_rows,
            "all_owner_install_root_sha256": _sha("all-owner-install-root"),
        }
    )


def _prepare(
    *, capacity: C.FrozenFlightCapacityReceipt | None = None
) -> C.PhysicalInstallPrepareReceipt:
    capacity = capacity or _capacity()
    install = _install(capacity=capacity)
    pre_slots = tuple(
        _parked(capacity, env_id=0, slot=slot, version=11)
        for slot in range(capacity.configured_flight_capacity)
    )
    pre = pre_slots[install.flight_slot]
    row = C.PreparedPhysicalInstallRow(
        env_id=0,
        slot_index=install.flight_slot,
        pre_slot_snapshot=pre,
        pre_slot_snapshot_sha256=pre.canonical_sha256,
        install_payload=install,
        install_payload_sha256=install.canonical_sha256,
    )
    preview_mapping = _preview_mapping(((install, pre),))
    preview = C.CanonicalJsonContentPin.from_sealed_mapping(
        preview_mapping,
        expected_source_kind=C.REVEAL_FINAL_PREVIEW_KIND,
        source_schema_sha256=_sha("r05-preview-schema"),
    )
    return C.PhysicalInstallPrepareReceipt(
        integration_status=C.INTEGRATION_STATUS,
        capacity_receipt_sha256=capacity.canonical_sha256,
        reveal_final_preview=preview,
        num_envs=1,
        reset_generations=(7,),
        physical_owner_checkpoint_before_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=1,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=11,
                next_prepare_nonce=29,
                reset_generations=(7,),
                slots=pre_slots,
                poisoned=False,
            )
        ),
        mutation_version_before=11,
        prepare_nonce=29,
        selected_env_ids=(0,),
        pre_slot_snapshots=pre_slots,
        rows=(row,),
        pre_slots_root_sha256=C.physical_slot_root(pre_slots),
        live_state_mutated=False,
        runtime_publication_created=False,
    )


def _prearm_marker_mapping(
    prepare: C.PhysicalInstallPrepareReceipt,
) -> dict[str, object]:
    return _seal(
        {
            "schema_version": 1,
            "kind": C.REVEAL_PREPARE_BOUNDARY_MARKER_KIND,
            "selected_env_ids": list(prepare.selected_env_ids),
            "reveal_final_preview_sha256": (
                prepare.reveal_final_preview.source_canonical_sha256
            ),
            "boundary_packet_version": 1,
            "boundary_packet_root_sha256": _sha("packed-boundary"),
            "boundary_transfer_count": 1,
            "selected_pass_count": len(prepare.selected_env_ids),
            "selected_fault_count": 0,
            "ordered_child_token_roots": [
                [
                    kind,
                    (
                        prepare.canonical_sha256
                        if kind == "physical_ball"
                        else _sha(f"{kind}-prepared")
                    ),
                ]
                for kind in C.R05_PREARM_CHILD_OWNER_KINDS
            ],
        }
    )


def _prearm_marker_pin(
    prepare: C.PhysicalInstallPrepareReceipt,
) -> C.CanonicalJsonContentPin:
    mapping = _prearm_marker_mapping(prepare)
    return C.CanonicalJsonContentPin.from_sealed_mapping(
        mapping,
        expected_source_kind=C.REVEAL_PREPARE_BOUNDARY_MARKER_KIND,
        source_schema_sha256=(
            C.REVEAL_PREPARE_BOUNDARY_MARKER_SCHEMA_SHA256
        ),
    )


def _global_boundary_receipt_mapping(
    prepare: C.PhysicalInstallPrepareReceipt,
    *,
    decision: str,
    d05_construction_admissible: bool = True,
    d05_selected_primary_fault: tuple[int, ...] | None = None,
    child_fault_on_censor: bool = True,
) -> dict[str, object]:
    selected = prepare.selected_env_ids
    censored = decision == C.FULL_MDP_REVEAL_DECISION_CENSOR
    d05_faults = d05_selected_primary_fault or (0,) * len(selected)
    rows = []
    for kind in C.FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER:
        failed = censored and child_fault_on_censor and kind == "racket"
        rows.append(
            {
                "kind": "action_ball_full_mdp_reveal_boundary_owner_row_v1",
                "owner_kind": kind,
                "owner_mutation_version": (
                    prepare.mutation_version_before
                    if kind == "physical_ball"
                    else 0
                ),
                "owner_token_root_sha256": (
                    prepare.canonical_sha256
                    if kind == "physical_ball"
                    else _sha(f"{kind}-boundary-token")
                ),
                "fault_schema_sha256": _sha(f"{kind}-fault-schema"),
                "allowed_fault_mask": 1,
                "selected_pass": [not failed for _ in selected],
                "selected_fault_bits": [1 if failed else 0 for _ in selected],
            }
        )
    num_envs = prepare.num_envs
    packet_nbytes = 256 + 55 * num_envs
    child_failed = censored and child_fault_on_censor
    return _seal(
        {
            "schema_version": 1,
            "kind": C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND,
            "packet_schema_version": (
                C.FULL_MDP_REVEAL_BOUNDARY_PACKET_SCHEMA_VERSION
            ),
            "boundary_sequence": 1,
            "reveal_final_preview_schema_version": (
                prepare.reveal_final_preview.source_schema_version
            ),
            "reveal_final_preview_sha256": (
                prepare.reveal_final_preview.source_canonical_sha256
            ),
            "num_envs": num_envs,
            "selected_env_ids": list(selected),
            "ordered_owner_kinds": list(
                C.FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER
            ),
            "ordered_owner_rows": rows,
            "packet_nbytes": packet_nbytes,
            "packet_sha256": _sha(f"boundary-packet:{decision}"),
            "device_type": "cpu",
            "device_index": None,
            "boundary_transfer_count": 1,
            "transfer_attempt_count_total": 1,
            "transfer_success_count_total": 1,
            "transfer_bytes_total": packet_nbytes,
            "transfer_elapsed_ns_total": 1,
            "selected_pass_count": 0 if child_failed else len(selected),
            "selected_fault_count": len(selected) if child_failed else 0,
            "decision": decision,
            "d05_construction_admissible": d05_construction_admissible,
            "d05_owner_fault_present": any(d05_faults),
            "d05_selected_primary_fault": list(d05_faults),
        }
    )


def _global_boundary_receipt_pin(
    prepare: C.PhysicalInstallPrepareReceipt,
    *,
    decision: str,
) -> C.CanonicalJsonContentPin:
    return C.CanonicalJsonContentPin.from_sealed_mapping(
        _global_boundary_receipt_mapping(prepare, decision=decision),
        expected_source_kind=C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND,
        source_schema_sha256=(
            C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ),
    )


def _r05_terminal_evidence(
    prepare: C.PhysicalInstallPrepareReceipt,
    boundary: C.CanonicalJsonContentPin,
    *,
    decision: str,
) -> tuple[
    C.R05TerminalClaimProjection,
    C.CanonicalJsonContentPin,
    C.CanonicalJsonContentPin,
]:
    mapping = boundary.decoded_mapping
    authority_schema = C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
    authority_source = _sha("full-mdp-reveal-boundary-source")
    authority_sha = C.canonical_sha256(
        {
            "schema_version": 1,
            "kind": C.R05_TERMINAL_BOUNDARY_AUTHORITY_KIND,
            "authority_domain": C.FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN,
            "authority_schema_sha256": authority_schema,
            "authority_source_sha256": authority_source,
        }
    )
    participants = tuple(
        _seal(
            {
                "schema_version": 1,
                "kind": C.R05_TERMINAL_BOUNDARY_PARTICIPANT_ROOT_KIND,
                "participant_domain": (
                    C.FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
                ),
                "participant_kind": owner_kind,
                "participant_root_sha256": row["owner_token_root_sha256"],
            }
        )
        for owner_kind, row in zip(
            C.FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER,
            mapping["ordered_owner_rows"],
        )
    )
    censor_evidence = ()
    if decision == C.FULL_MDP_REVEAL_DECISION_CENSOR:
        participant_index = C.FULL_MDP_REVEAL_BOUNDARY_OWNER_ORDER.index(
            "racket"
        )
        participant = mapping["ordered_owner_rows"][participant_index]
        primary = prepare.selected_env_ids[0]
        censor_evidence = tuple(
            _seal(
                {
                    "schema_version": 1,
                    "kind": C.R05_TERMINAL_BOUNDARY_CENSOR_EVIDENCE_KIND,
                    "env_id": env_id,
                    "primary_failure_env_id": primary,
                    "participant_domain": (
                        C.FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN
                    ),
                    "participant_kind": "racket",
                    "participant_root_sha256": participant[
                        "owner_token_root_sha256"
                    ],
                    "failure_receipt_sha256": C.canonical_sha256(
                        participant
                    ),
                    "reason": "owner_preterminal_receipt_censored",
                    "censor_fact_sha256": _sha(f"censor-fact:{env_id}"),
                    "producer_schema_sha256": authority_schema,
                    "producer_source_sha256": authority_source,
                }
            )
            for env_id in prepare.selected_env_ids
        )
    projection_mapping = _seal(
        {
            "schema_version": 1,
            "kind": C.R05_TERMINAL_BOUNDARY_PROJECTION_KIND,
            "authority_domain": C.FULL_MDP_REVEAL_BOUNDARY_AUTHORITY_DOMAIN,
            "authority_schema_sha256": authority_schema,
            "authority_source_sha256": authority_source,
            "decision_mapping_schema_version": 1,
            "source_decision": decision,
            "decision": decision,
            "reveal_final_preview_schema_version": (
                prepare.reveal_final_preview.source_schema_version
            ),
            "reveal_final_preview_sha256": (
                prepare.reveal_final_preview.source_canonical_sha256
            ),
            "selected_env_ids": list(prepare.selected_env_ids),
            "boundary_receipt_kind": boundary.source_kind,
            "boundary_receipt_sha256": boundary.source_canonical_sha256,
            "boundary_packet_schema_version": mapping[
                "packet_schema_version"
            ],
            "boundary_packet_sha256": mapping["packet_sha256"],
            "ordered_participant_roots": list(participants),
            "ordered_censor_evidence": list(censor_evidence),
        }
    )
    projection_pin = C.CanonicalJsonContentPin.from_sealed_mapping(
        projection_mapping,
        expected_source_kind=C.R05_TERMINAL_BOUNDARY_PROJECTION_KIND,
        source_schema_sha256=(
            C.R05_TERMINAL_BOUNDARY_PROJECTION_SCHEMA_SHA256
        ),
    )
    terminal_kind = (
        C.COMMITTED_REVEAL_BATCH_KIND
        if decision == C.FULL_MDP_REVEAL_DECISION_ACCEPT
        else C.CENSORED_REVEAL_BATCH_KIND
    )
    marker = _seal(
        {
            "schema_version": 1,
            "kind": "action_ball_continuous_reveal_terminal_boundary_marker_v1",
            "terminal_boundary_authority_sha256": authority_sha,
            "terminal_boundary_projection": projection_mapping,
        }
    )
    terminal_mapping = _seal(
        {
            "schema_version": 2,
            "kind": terminal_kind,
            (
                "global_prearm_marker"
                if decision == C.FULL_MDP_REVEAL_DECISION_ACCEPT
                else "terminal_boundary_marker"
            ): marker,
        }
    )
    terminal_raw = C.canonical_json_bytes(terminal_mapping)
    content_mapping = _seal(
        {
            "schema_version": 1,
            "kind": C.R05_PREPARED_TERMINAL_CONTENT_PIN_KIND,
            "terminal_schema_version": 2,
            "terminal_kind": terminal_kind,
            "terminal_canonical_sha256": terminal_mapping[
                "canonical_sha256"
            ],
            "content_bytes_base64": base64.b64encode(terminal_raw).decode(
                "ascii"
            ),
            "content_byte_length": len(terminal_raw),
            "content_bytes_sha256": hashlib.sha256(terminal_raw).hexdigest(),
        }
    )
    content_pin = C.CanonicalJsonContentPin.from_sealed_mapping(
        content_mapping,
        expected_source_kind=C.R05_PREPARED_TERMINAL_CONTENT_PIN_KIND,
        source_schema_sha256=(
            C.R05_PREPARED_TERMINAL_CONTENT_PIN_SCHEMA_SHA256
        ),
    )
    claim = C.R05TerminalClaimProjection(
        decision=decision,
        selected_env_ids=prepare.selected_env_ids,
        reveal_final_preview_schema_version=(
            prepare.reveal_final_preview.source_schema_version
        ),
        reveal_final_preview_sha256=(
            prepare.reveal_final_preview.source_canonical_sha256
        ),
        global_boundary_receipt_kind=boundary.source_kind,
        global_boundary_receipt_sha256=boundary.source_canonical_sha256,
        global_boundary_packet_schema_version=mapping["packet_schema_version"],
        global_boundary_packet_sha256=mapping["packet_sha256"],
        terminal_boundary_authority_sha256=authority_sha,
        terminal_boundary_projection_sha256=(
            projection_pin.source_canonical_sha256
        ),
        terminal_content_pin_sha256=content_pin.source_canonical_sha256,
        terminal_kind=terminal_kind,
        terminal_sha256=terminal_mapping["canonical_sha256"],
    )
    return claim, projection_pin, content_pin


def test_capacity_has_no_defaults_and_derives_only_from_pinned_inclusive_c_h():
    assert all(field.default is MISSING for field in fields(C.FrozenFlightCapacityReceipt))
    signature = inspect.signature(C.FrozenFlightCapacityReceipt)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )

    cases = ((5, 0, 1), (5, 4, 1), (5, 5, 2), (5, 9, 2), (5, 10, 3))
    for cadence, horizon, expected in cases:
        receipt = _capacity(cadence=cadence, horizon=horizon)
        assert receipt.configured_flight_capacity == expected
        assert receipt.required_inclusive_flight_capacity == expected
        restored = C.FrozenFlightCapacityReceipt.from_mapping(
            receipt.to_mapping(),
            expected_canonical_sha256=receipt.canonical_sha256,
        )
        assert restored == receipt


@pytest.mark.parametrize("bad_cadence", [0, True, 1.0])
def test_capacity_rejects_nonpositive_or_nonexact_cadence(bad_cadence):
    capacity = _capacity()
    expected = "allowed range" if bad_cadence == 0 else "exact int"
    with pytest.raises(C.PhysicalFlightContractError, match=expected):
        replace(capacity, cadence_control_steps=bad_cadence)


def test_capacity_rejects_forged_k_order_clock_and_external_pin_replay():
    receipt = _capacity(cadence=5, horizon=9)
    with pytest.raises(C.PhysicalFlightContractError, match="C/H derivation"):
        replace(
            receipt,
            required_inclusive_flight_capacity=3,
            configured_flight_capacity=3,
        )
    with pytest.raises(C.PhysicalFlightContractError, match="semantics"):
        replace(receipt, same_tick_ordering="retire_before_reveal")
    with pytest.raises(C.PhysicalFlightContractError, match="authority content"):
        replace(receipt, control_step_clock_root_sha256=_sha("other-clock"))

    other = _capacity(cadence=5, horizon=10)
    with pytest.raises(C.ExternalContentPinError, match="external content pin"):
        C.FrozenFlightCapacityReceipt.from_mapping(
            other.to_mapping(),
            expected_canonical_sha256=receipt.canonical_sha256,
        )


def test_state_is_exact_ordered_positive_zero_big_endian_binary32():
    state = _state()
    mapping = state.to_mapping()
    encoded = mapping["state_f32_be_hex"]
    assert len(encoded) == 104
    assert encoded[0:8] == "00000000"
    assert encoded[8:16] == "3f800000"
    assert encoded[24:32] == "3f800000"
    assert encoded[40:48] == "00000000"
    assert mapping["state_bytes_sha256"] == state.state_bytes_sha256
    assert tuple(mapping["components"]) == C.STATE_COMPONENTS

    underflow = replace(state, position_env_m=(-1.0e-50, 1.0, 0.75))
    assert underflow.to_mapping()["state_f32_be_hex"][0:8] == "00000000"
    assert underflow == state
    restored = C.CanonicalPhysicalBallStateF32.from_mapping(
        mapping, expected_canonical_sha256=state.canonical_sha256
    )
    assert restored == state


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, -math.inf, 1.0e100])
def test_state_rejects_bool_nonfinite_and_binary32_overflow(bad):
    with pytest.raises(C.PhysicalFlightContractError):
        replace(_state(), position_env_m=(bad, 1.0, 0.75))


def test_state_rejects_negative_zero_bytes_width_order_and_zero_quaternion():
    state = _state()
    mapping = state.to_mapping()
    mapping["state_f32_be_hex"] = (
        "80000000" + mapping["state_f32_be_hex"][8:]
    )
    mapping = _reseal(mapping)
    with pytest.raises(C.PhysicalFlightContractError, match="negative zero"):
        C.CanonicalPhysicalBallStateF32.from_mapping(
            mapping, expected_canonical_sha256=mapping["canonical_sha256"]
        )

    reordered = _state().to_mapping()
    reordered["components"] = list(reversed(C.STATE_COMPONENTS))
    reordered = _reseal(reordered)
    with pytest.raises(C.PhysicalFlightContractError, match="metadata differs"):
        C.CanonicalPhysicalBallStateF32.from_mapping(
            reordered,
            expected_canonical_sha256=reordered["canonical_sha256"],
        )
    with pytest.raises(C.PhysicalFlightContractError, match="all zero"):
        replace(_state(), quaternion_wxyz=(0.0, -0.0, 0.0, 0.0))


def test_task_and_outcome_are_full_exact_and_externally_pinned():
    task = _task()
    outcome = _outcome()
    assert outcome.task_ref == task
    assert C.PhysicalFlightTaskRef.from_mapping(
        task.to_mapping(), expected_canonical_sha256=task.canonical_sha256
    ) == task
    assert C.PhysicalFlightOutcomeKey.from_mapping(
        outcome.to_mapping(), expected_canonical_sha256=outcome.canonical_sha256
    ) == outcome

    suffix_changed = replace(outcome, receipt_content_sha256=_sha("changed"))
    with pytest.raises(C.ExternalContentPinError):
        C.PhysicalFlightOutcomeKey.from_mapping(
            suffix_changed.to_mapping(),
            expected_canonical_sha256=outcome.canonical_sha256,
        )
    with pytest.raises(C.PhysicalFlightContractError, match="keys differ"):
        mapping = task.to_mapping()
        mapping["unknown"] = 1
        C.PhysicalFlightTaskRef.from_mapping(
            mapping, expected_canonical_sha256=task.canonical_sha256
        )


def test_install_roundtrip_binds_full_state_frame_timing_task_outcome_and_capacity():
    install = _install()
    restored = C.PhysicalBallInstallPayload.from_mapping(
        install.to_mapping(),
        expected_canonical_sha256=install.canonical_sha256,
    )
    assert restored == install
    assert install.first_crossing_horizon_control_step == (
        install.reveal_control_step
        + install.capacity_receipt.max_flight_horizon_control_steps
    )
    assert install.frame_binding_sha256 == (
        install.frame_binding_authority.source_canonical_sha256
    )

    with pytest.raises(C.PhysicalFlightContractError, match="horizon"):
        replace(
            install,
            first_crossing_horizon_control_step=(
                install.first_crossing_horizon_control_step + 1
            ),
        )
    changed_outcome = _outcome(swing=4)
    with pytest.raises(C.PhysicalFlightContractError, match="task/outcome"):
        replace(
            install,
            outcome_key=changed_outcome,
            outcome_key_sha256=changed_outcome.canonical_sha256,
        )
    with pytest.raises(C.PhysicalFlightContractError, match="frame"):
        replace(install, quaternion_order="xyzw")
    with pytest.raises(C.PhysicalFlightContractError, match="outside frozen capacity"):
        replace(
            install,
            flight_slot=install.capacity_receipt.configured_flight_capacity,
        )


def test_digest_only_install_and_checkpoint_are_explicit_tombstones():
    legacy_install = {
        "schema_version": 1,
        "kind": "action_ball_physical_ball_install_digest_v1",
        "installed_ball_state_sha256": _sha("state"),
        "canonical_sha256": _sha("legacy-install"),
    }
    with pytest.raises(C.DigestOnlyPayloadTombstonedError):
        C.PhysicalBallInstallPayload.from_mapping(
            legacy_install,
            expected_canonical_sha256=legacy_install["canonical_sha256"],
        )
    legacy_checkpoint = {
        "schema_version": 1,
        "kind": "action_ball_physical_flight_checkpoint_digest_v1",
        "owner_state_bytes_sha256": _sha("state"),
        "canonical_sha256": _sha("legacy-checkpoint"),
    }
    with pytest.raises(C.DigestOnlyPayloadTombstonedError):
        C.PhysicalFlightCheckpointReceipt.from_mapping(
            legacy_checkpoint,
            expected_canonical_sha256=legacy_checkpoint["canonical_sha256"],
        )


def test_prepare_is_full_preview_bound_private_nonmutating_and_full_k():
    prepare = _prepare()
    assert prepare.selected_env_ids == (0,)
    assert len(prepare.pre_slot_snapshots) == (
        prepare.rows[0].install_payload.capacity_receipt.configured_flight_capacity
    )
    assert not prepare.live_state_mutated
    assert not prepare.runtime_publication_created
    restored = C.PhysicalInstallPrepareReceipt.from_mapping(
        prepare.to_mapping(),
        expected_canonical_sha256=prepare.canonical_sha256,
    )
    assert restored == prepare

    with pytest.raises(C.PhysicalFlightContractError, match="complete selected-env"):
        replace(prepare, pre_slot_snapshots=prepare.pre_slot_snapshots[:-1])
    with pytest.raises(C.PhysicalFlightContractError, match="cannot mutate"):
        replace(prepare, live_state_mutated=True)
    with pytest.raises(C.PhysicalFlightContractError, match="sorted"):
        replace(prepare, selected_env_ids=(0, 0))


def test_prepare_rejects_digest_only_r05_state_alias_and_cross_row_payload():
    prepare = _prepare()
    preview = dict(prepare.reveal_final_preview.decoded_mapping)
    preview_rows = list(preview["reveal_final_rows"])
    row = dict(preview_rows[0])
    row["physical_ball_install_payload_sha256"] = row["ball_slot_plan"][
        "new_ball_dynamic_state_sha256"
    ]
    preview_rows[0] = _reseal(row)
    preview["reveal_final_rows"] = preview_rows
    preview = _reseal(preview)
    digest_only_pin = C.CanonicalJsonContentPin.from_sealed_mapping(
        preview,
        expected_source_kind=C.REVEAL_FINAL_PREVIEW_KIND,
        source_schema_sha256=_sha("r05-preview-schema"),
    )
    with pytest.raises(C.PhysicalFlightContractError, match="complete physical"):
        replace(prepare, reveal_final_preview=digest_only_pin)

    other_install = _install(swing=4)
    prior = prepare.rows[0]
    swapped = C.PreparedPhysicalInstallRow(
        env_id=prior.env_id,
        slot_index=prior.slot_index,
        pre_slot_snapshot=prior.pre_slot_snapshot,
        pre_slot_snapshot_sha256=prior.pre_slot_snapshot_sha256,
        install_payload=other_install,
        install_payload_sha256=other_install.canonical_sha256,
    )
    with pytest.raises(C.PhysicalFlightContractError, match="preview"):
        replace(prepare, rows=(swapped,))


def test_commit_and_censor_consume_exact_global_boundary_while_abort_is_zero_mutation():
    prepare = _prepare()
    assert not hasattr(prepare, "boundary_packet_root_sha256")
    assert not hasattr(prepare, "boundary_transfer_count")
    install = prepare.rows[0].install_payload
    live = _live(install, version=12)
    row = C.CommittedPhysicalInstallRow(
        env_id=install.env_id,
        slot_index=install.flight_slot,
        install_payload_sha256=install.canonical_sha256,
        committed_slot_snapshot=live,
        committed_slot_snapshot_sha256=live.canonical_sha256,
    )
    after_slots = list(prepare.pre_slot_snapshots)
    after_slots[install.env_id * install.capacity_receipt.configured_flight_capacity + install.flight_slot] = live
    accepted_boundary = _global_boundary_receipt_pin(
        prepare,
        decision=C.FULL_MDP_REVEAL_DECISION_ACCEPT,
    )
    physical_fault_schema = _sha("physical_ball-fault-schema")
    (
        accepted_claim,
        accepted_terminal_projection,
        accepted_terminal_content,
    ) = _r05_terminal_evidence(
        prepare,
        accepted_boundary,
        decision=C.FULL_MDP_REVEAL_DECISION_ACCEPT,
    )
    accepted_terminal_sha = accepted_claim.terminal_sha256
    commit = C.PhysicalInstallCommitReceipt(
        integration_status=C.INTEGRATION_STATUS,
        prepare_receipt=prepare,
        prepare_receipt_sha256=prepare.canonical_sha256,
        global_reveal_boundary_receipt=accepted_boundary,
        global_reveal_boundary_receipt_sha256=(
            accepted_boundary.source_canonical_sha256
        ),
        physical_boundary_fault_schema_sha256=physical_fault_schema,
        r05_terminal_claim=accepted_claim,
        r05_terminal_claim_sha256=accepted_claim.canonical_sha256,
        r05_terminal_boundary_projection=accepted_terminal_projection,
        r05_terminal_content_pin=accepted_terminal_content,
        r05_terminal_kind=C.COMMITTED_REVEAL_BATCH_KIND,
        r05_terminal_sha256=accepted_terminal_sha,
        physical_owner_checkpoint_before_sha256=(
            prepare.physical_owner_checkpoint_before_sha256
        ),
        physical_owner_checkpoint_after_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=prepare.capacity_receipt_sha256,
                num_envs=prepare.num_envs,
                flight_capacity=(
                    install.capacity_receipt.configured_flight_capacity
                ),
                mutation_version=12,
                next_prepare_nonce=prepare.prepare_nonce + 1,
                reset_generations=prepare.reset_generations,
                slots=tuple(after_slots),
                poisoned=False,
            )
        ),
        mutation_version_before=11,
        mutation_version_after=12,
        rows=(row,),
        committed_slots_root_sha256=C.physical_slot_root((live,)),
        live_state_mutated=True,
        runtime_publication_created=True,
    )
    assert C.PhysicalInstallCommitReceipt.from_mapping(
        commit.to_mapping(), expected_canonical_sha256=commit.canonical_sha256
    ) == commit
    assert not hasattr(commit, "committed_reveal_batch")
    with pytest.raises(C.PhysicalFlightContractError, match="differs"):
        replace(
            commit,
            r05_terminal_claim_sha256=_sha("swapped-terminal-claim"),
            r05_terminal_sha256=_sha("swapped-terminal"),
        )

    bad_boundary_mapping = copy.deepcopy(
        _global_boundary_receipt_mapping(
            prepare,
            decision=C.FULL_MDP_REVEAL_DECISION_ACCEPT,
        )
    )
    bad_boundary_mapping["ordered_owner_rows"][2][
        "owner_token_root_sha256"
    ] = _sha(
        "different-physical-prepare"
    )
    bad_boundary_mapping = _reseal(bad_boundary_mapping)
    bad_boundary = C.CanonicalJsonContentPin.from_sealed_mapping(
        bad_boundary_mapping,
        expected_source_kind=C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND,
        source_schema_sha256=(
            C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ),
    )
    with pytest.raises(C.PhysicalFlightContractError, match="differs"):
        replace(
            commit,
            global_reveal_boundary_receipt=bad_boundary,
            global_reveal_boundary_receipt_sha256=(
                bad_boundary.source_canonical_sha256
            ),
        )

    censored_boundary = _global_boundary_receipt_pin(
        prepare,
        decision=C.FULL_MDP_REVEAL_DECISION_CENSOR,
    )
    (
        censored_claim,
        censored_terminal_projection,
        censored_terminal_content,
    ) = _r05_terminal_evidence(
        prepare,
        censored_boundary,
        decision=C.FULL_MDP_REVEAL_DECISION_CENSOR,
    )
    censored_terminal_sha = censored_claim.terminal_sha256
    censor = C.PhysicalInstallCensorReceipt(
        integration_status=C.INTEGRATION_STATUS,
        prepare_receipt=prepare,
        prepare_receipt_sha256=prepare.canonical_sha256,
        global_reveal_boundary_receipt=censored_boundary,
        global_reveal_boundary_receipt_sha256=(
            censored_boundary.source_canonical_sha256
        ),
        physical_boundary_fault_schema_sha256=physical_fault_schema,
        r05_terminal_claim=censored_claim,
        r05_terminal_claim_sha256=censored_claim.canonical_sha256,
        r05_terminal_boundary_projection=censored_terminal_projection,
        r05_terminal_content_pin=censored_terminal_content,
        r05_terminal_kind=C.CENSORED_REVEAL_BATCH_KIND,
        r05_terminal_sha256=censored_terminal_sha,
        physical_owner_checkpoint_before_sha256=(
            prepare.physical_owner_checkpoint_before_sha256
        ),
        physical_owner_checkpoint_after_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=prepare.capacity_receipt_sha256,
                num_envs=prepare.num_envs,
                flight_capacity=(
                    install.capacity_receipt.configured_flight_capacity
                ),
                mutation_version=12,
                next_prepare_nonce=prepare.prepare_nonce + 1,
                reset_generations=prepare.reset_generations,
                slots=prepare.pre_slot_snapshots,
                poisoned=False,
            )
        ),
        mutation_version_before=11,
        mutation_version_after=12,
        slots_root_before_sha256=prepare.pre_slots_root_sha256,
        slots_root_after_sha256=prepare.pre_slots_root_sha256,
        scene_state_mutated=False,
        slot_state_mutated=False,
        owner_chronology_mutated=True,
        runtime_publication_created=False,
        policy_opportunity_created=False,
    )
    assert C.PhysicalInstallCensorReceipt.from_mapping(
        censor.to_mapping(), expected_canonical_sha256=censor.canonical_sha256
    ) == censor
    with pytest.raises(C.PhysicalFlightContractError, match="zero install"):
        replace(censor, policy_opportunity_created=True)

    d05_censored_boundary = C.CanonicalJsonContentPin.from_sealed_mapping(
        _global_boundary_receipt_mapping(
            prepare,
            decision=C.FULL_MDP_REVEAL_DECISION_CENSOR,
            d05_construction_admissible=False,
            d05_selected_primary_fault=(17,),
            child_fault_on_censor=False,
        ),
        expected_source_kind=C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND,
        source_schema_sha256=(
            C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ),
    )
    d05_mapping, physical_row = (
        C._verified_full_mdp_reveal_boundary_receipt(
            d05_censored_boundary,
            expected_decision=C.FULL_MDP_REVEAL_DECISION_CENSOR,
        )
    )
    assert d05_mapping["d05_owner_fault_present"] is True
    assert tuple(physical_row["selected_pass"]) == (True,)

    inadmissible_accept_mapping = _global_boundary_receipt_mapping(
        prepare,
        decision=C.FULL_MDP_REVEAL_DECISION_ACCEPT,
        d05_construction_admissible=False,
    )
    inadmissible_accept = C.CanonicalJsonContentPin.from_sealed_mapping(
        inadmissible_accept_mapping,
        expected_source_kind=C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND,
        source_schema_sha256=(
            C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ),
    )
    with pytest.raises(
        C.PhysicalFlightContractError,
        match="conservation/decision",
    ):
        replace(
            commit,
            global_reveal_boundary_receipt=inadmissible_accept,
            global_reveal_boundary_receipt_sha256=(
                inadmissible_accept.source_canonical_sha256
            ),
        )

    inconsistent_d05_mapping = copy.deepcopy(
        _global_boundary_receipt_mapping(
            prepare,
            decision=C.FULL_MDP_REVEAL_DECISION_CENSOR,
            d05_construction_admissible=False,
            d05_selected_primary_fault=(17,),
            child_fault_on_censor=False,
        )
    )
    inconsistent_d05_mapping["d05_owner_fault_present"] = False
    inconsistent_d05 = C.CanonicalJsonContentPin.from_sealed_mapping(
        _reseal(inconsistent_d05_mapping),
        expected_source_kind=C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_KIND,
        source_schema_sha256=(
            C.FULL_MDP_REVEAL_BOUNDARY_RECEIPT_SCHEMA_SHA256
        ),
    )
    with pytest.raises(
        C.PhysicalFlightContractError,
        match="conservation/decision",
    ):
        replace(
            censor,
            global_reveal_boundary_receipt=inconsistent_d05,
            global_reveal_boundary_receipt_sha256=(
                inconsistent_d05.source_canonical_sha256
            ),
        )

    abort = C.PhysicalInstallAbortReceipt(
        integration_status=C.INTEGRATION_STATUS,
        prepare_receipt=prepare,
        prepare_receipt_sha256=prepare.canonical_sha256,
        physical_owner_checkpoint_before_sha256=(
            prepare.physical_owner_checkpoint_before_sha256
        ),
        physical_owner_checkpoint_after_sha256=(
            prepare.physical_owner_checkpoint_before_sha256
        ),
        mutation_version_before=11,
        mutation_version_after=11,
        live_state_mutated=False,
        runtime_publication_created=False,
    )
    assert C.PhysicalInstallAbortReceipt.from_mapping(
        abort.to_mapping(), expected_canonical_sha256=abort.canonical_sha256
    ) == abort
    with pytest.raises(C.PhysicalFlightContractError, match="preserve"):
        replace(abort, mutation_version_after=12)


def test_prepare_rejects_resealed_unknown_r05_preimage_lifecycle():
    prepare = _prepare()
    mapping = copy.deepcopy(prepare.reveal_final_preview.decoded_mapping)
    row = mapping["reveal_final_rows"][0]
    pre_slot = row["pre_install_ball_slots"][0]
    pre_slot["lifecycle_state"] = "totally_wrong_r06_lifecycle"
    row["pre_install_ball_slots"][0] = _reseal(pre_slot)
    mapping["reveal_final_rows"][0] = _reseal(row)
    mapping = _reseal(mapping)
    pin = C.CanonicalJsonContentPin.from_sealed_mapping(
        mapping,
        expected_source_kind=C.REVEAL_FINAL_PREVIEW_KIND,
        source_schema_sha256=_sha("r05-preview-schema"),
    )
    with pytest.raises(C.PhysicalFlightContractError, match="physical install payload"):
        replace(prepare, reveal_final_preview=pin)


def test_prepare_accepts_mixed_per_slot_versions_but_rejects_future_slot_version():
    prepare = _prepare()
    mixed = list(prepare.pre_slot_snapshots)
    nonselected = next(
        index
        for index, slot in enumerate(mixed)
        if slot.slot_index != prepare.rows[0].slot_index
    )
    mixed[nonselected] = replace(mixed[nonselected], mutation_version=3)
    accepted = replace(
        prepare,
        pre_slot_snapshots=tuple(mixed),
        pre_slots_root_sha256=C.physical_slot_root(tuple(mixed)),
        physical_owner_checkpoint_before_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=prepare.capacity_receipt_sha256,
                num_envs=prepare.num_envs,
                flight_capacity=(
                    prepare.rows[0]
                    .install_payload.capacity_receipt.configured_flight_capacity
                ),
                mutation_version=prepare.mutation_version_before,
                next_prepare_nonce=prepare.prepare_nonce,
                reset_generations=prepare.reset_generations,
                slots=tuple(mixed),
                poisoned=False,
            )
        ),
    )
    assert accepted.pre_slot_snapshots[nonselected].mutation_version == 3

    future = list(mixed)
    future[nonselected] = replace(
        future[nonselected], mutation_version=prepare.mutation_version_before + 1
    )
    with pytest.raises(C.PhysicalFlightContractError, match="capacity/version"):
        replace(
            prepare,
            pre_slot_snapshots=tuple(future),
            pre_slots_root_sha256=C.physical_slot_root(tuple(future)),
        )


def test_retire_preserves_full_identity_and_never_releases_mailbox():
    install = _install()
    capacity = install.capacity_receipt
    pre = _live(install, version=12)
    post = replace(
        pre,
        lifecycle=C.SLOT_RETIRED,
        mutation_version=13,
        physically_parked=True,
        published_to_runtime=False,
    )
    settlement = _pin(
        C.PHYSICAL_SETTLEMENT_AUTHORITY_KIND,
        schema_version=2,
        source_schema_sha256=C.PHYSICAL_SETTLEMENT_AUTHORITY_SCHEMA_SHA256,
        mailbox_lifecycle="SETTLED_UNPAID",
        r06_owner_mutation_version=3,
        r06_after_root_sha256=_sha("r06-after"),
        physical_retire_rows=[
            {
                "env_id": install.env_id,
                "slot_index": install.flight_slot,
                "outcome_key_sha256": install.outcome_key_sha256,
                "ball_generation": install.ball_generation,
            }
        ],
    )
    row = C.PhysicalRetireRow(
        env_id=install.env_id,
        slot_index=install.flight_slot,
        outcome_key=install.outcome_key,
        outcome_key_sha256=install.outcome_key_sha256,
        settlement_authority=settlement,
        pre_slot_snapshot=pre,
        post_slot_snapshot=post,
    )
    pre_owner = list(
        _parked(capacity, env_id=0, slot=slot, version=12)
        for slot in range(capacity.configured_flight_capacity)
    )
    pre_owner[install.flight_slot] = pre
    post_owner = list(pre_owner)
    post_owner[install.flight_slot] = post
    receipt = C.PhysicalRetireReceipt(
        integration_status=C.INTEGRATION_STATUS,
        physical_owner_checkpoint_before_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=1,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=12,
                next_prepare_nonce=29,
                reset_generations=(7,),
                slots=tuple(pre_owner),
                poisoned=False,
            )
        ),
        physical_owner_checkpoint_after_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=1,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=13,
                next_prepare_nonce=29,
                reset_generations=(7,),
                slots=tuple(post_owner),
                poisoned=False,
            )
        ),
        mutation_version_before=12,
        mutation_version_after=13,
        num_envs=1,
        flight_capacity=capacity.configured_flight_capacity,
        reset_generations=(7,),
        next_prepare_nonce=29,
        pre_owner_slot_snapshots=tuple(pre_owner),
        post_owner_slot_snapshots=tuple(post_owner),
        rows=(row,),
        pre_slots_root_sha256=C.physical_slot_root((pre,)),
        post_slots_root_sha256=C.physical_slot_root((post,)),
        physical_flight_released=True,
        mailbox_lifecycle_mutated=False,
        scene_bodies_parked=True,
    )
    assert receipt.rows[0].post_slot_snapshot.outcome_key == install.outcome_key
    assert C.PhysicalRetireReceipt.from_mapping(
        receipt.to_mapping(), expected_canonical_sha256=receipt.canonical_sha256
    ) == receipt
    with pytest.raises(C.PhysicalFlightContractError, match="ownership boundary"):
        replace(receipt, mailbox_lifecycle_mutated=True)

    # Owner-wide operation ordering and per-slot mutation ordering are distinct.
    # A slot untouched by intervening owner operations may legitimately lag.
    lagging_receipt = replace(
        receipt,
        mutation_version_before=20,
        mutation_version_after=21,
        physical_owner_checkpoint_before_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=1,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=20,
                next_prepare_nonce=29,
                reset_generations=(7,),
                slots=tuple(pre_owner),
                poisoned=False,
            )
        ),
        physical_owner_checkpoint_after_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=1,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=21,
                next_prepare_nonce=29,
                reset_generations=(7,),
                slots=tuple(post_owner),
                poisoned=False,
            )
        ),
    )
    assert lagging_receipt.rows[0].post_slot_snapshot.mutation_version == 13


def test_selected_true_reset_parks_all_k_and_preserves_unselected_root():
    capacity = _capacity()
    install = _install(capacity=capacity)
    before = list(
        _parked(capacity, env_id=0, slot=slot, version=20)
        for slot in range(capacity.configured_flight_capacity)
    )
    after = tuple(
        _parked(capacity, env_id=0, slot=slot, version=21)
        for slot in range(capacity.configured_flight_capacity)
    )
    row = C.PhysicalTrueResetRow(
        env_id=0,
        prior_reset_generation=7,
        next_reset_generation=8,
        pre_slot_snapshots=tuple(before),
        post_slot_snapshots=after,
    )
    unselected = C.physical_slot_root(())
    receipt = C.PhysicalTrueResetReceipt(
        integration_status=C.INTEGRATION_STATUS,
        zero_open_all_owner_closure=_pin(
            C.PHYSICAL_ZERO_OPEN_RESET_CLOSURE_KIND,
            schema_version=2,
            selected_env_ids=[0],
            open_flight_count=0,
            open_mailbox_count=0,
        ),
        selected_env_ids=(0,),
        rows=(row,),
        physical_owner_checkpoint_before_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=1,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=20,
                next_prepare_nonce=29,
                reset_generations=(7,),
                slots=tuple(before),
                poisoned=False,
            )
        ),
        physical_owner_checkpoint_after_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=1,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=21,
                next_prepare_nonce=29,
                reset_generations=(8,),
                slots=after,
                poisoned=False,
            )
        ),
        mutation_version_before=20,
        mutation_version_after=21,
        num_envs=1,
        flight_capacity=capacity.configured_flight_capacity,
        reset_generations_before=(7,),
        reset_generations_after=(8,),
        next_prepare_nonce=29,
        pre_owner_slot_snapshots=tuple(before),
        post_owner_slot_snapshots=after,
        selected_slots_root_before_sha256=C.physical_slot_root(tuple(before)),
        selected_slots_root_after_sha256=C.physical_slot_root(after),
        unselected_slots_root_before_sha256=unselected,
        unselected_slots_root_after_sha256=unselected,
        env_reset_invoked=False,
        mailbox_lifecycle_mutated=False,
    )
    assert all(slot.lifecycle == C.SLOT_PARKED for slot in receipt.rows[0].post_slot_snapshots)
    assert C.PhysicalTrueResetReceipt.from_mapping(
        receipt.to_mapping(), expected_canonical_sha256=receipt.canonical_sha256
    ) == receipt
    with pytest.raises(C.PhysicalFlightContractError, match="parity"):
        replace(receipt, unselected_slots_root_after_sha256=_sha("changed"))
    bad_after = list(after)
    bad_after[0] = replace(
        bad_after[0], mutation_version=before[0].mutation_version + 2
    )
    with pytest.raises(C.PhysicalFlightContractError, match="projection"):
        replace(row, post_slot_snapshots=tuple(bad_after))
    live_before = list(before)
    live_before[install.flight_slot] = _live(install, version=20)
    live_row = replace(row, pre_slot_snapshots=tuple(live_before))
    with pytest.raises(C.PhysicalFlightContractError, match="ownership/parity"):
        replace(
            receipt,
            rows=(live_row,),
            pre_owner_slot_snapshots=tuple(live_before),
            selected_slots_root_before_sha256=C.physical_slot_root(
                tuple(live_before)
            ),
            physical_owner_checkpoint_before_sha256=(
                C.physical_owner_checkpoint_root(
                    capacity_receipt_sha256=capacity.canonical_sha256,
                    num_envs=1,
                    flight_capacity=capacity.configured_flight_capacity,
                    mutation_version=20,
                    next_prepare_nonce=29,
                    reset_generations=(7,),
                    slots=tuple(live_before),
                    poisoned=False,
                )
            ),
        )

    # Full-owner validation also covers untouched environments: their bytes
    # remain exact, but they may not carry a future slot version or an outcome
    # from a reset generation different from that environment's owner header.
    unselected_slots = [
        _parked(capacity, env_id=1, slot=slot, version=19)
        for slot in range(capacity.configured_flight_capacity)
    ]
    env1_key = replace(install.outcome_key, env_id=1, reset_generation=9)
    unselected_slots[install.flight_slot] = replace(
        _live(install, version=19),
        env_id=1,
        scene_body_name=(
            f"physical_ball_env1_slot{install.flight_slot}"
        ),
        lifecycle=C.SLOT_RETIRED,
        outcome_key=env1_key,
        outcome_key_sha256=env1_key.canonical_sha256,
        physically_parked=True,
        published_to_runtime=False,
    )
    unselected_slots_tuple = tuple(unselected_slots)
    pre_owner_n2 = tuple(before) + unselected_slots_tuple
    post_owner_n2 = after + unselected_slots_tuple
    valid_n2 = replace(
        receipt,
        num_envs=2,
        reset_generations_before=(7, 9),
        reset_generations_after=(8, 9),
        pre_owner_slot_snapshots=pre_owner_n2,
        post_owner_slot_snapshots=post_owner_n2,
        unselected_slots_root_before_sha256=C.physical_slot_root(
            unselected_slots_tuple
        ),
        unselected_slots_root_after_sha256=C.physical_slot_root(
            unselected_slots_tuple
        ),
        physical_owner_checkpoint_before_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=2,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=20,
                next_prepare_nonce=29,
                reset_generations=(7, 9),
                slots=pre_owner_n2,
                poisoned=False,
            )
        ),
        physical_owner_checkpoint_after_sha256=(
            C.physical_owner_checkpoint_root(
                capacity_receipt_sha256=capacity.canonical_sha256,
                num_envs=2,
                flight_capacity=capacity.configured_flight_capacity,
                mutation_version=21,
                next_prepare_nonce=29,
                reset_generations=(8, 9),
                slots=post_owner_n2,
                poisoned=False,
            )
        ),
    )
    assert valid_n2.num_envs == 2

    future_unselected = list(unselected_slots_tuple)
    future_unselected[0] = replace(future_unselected[0], mutation_version=22)
    future_unselected_tuple = tuple(future_unselected)
    future_pre_owner = tuple(before) + future_unselected_tuple
    future_post_owner = after + future_unselected_tuple
    with pytest.raises(C.PhysicalFlightContractError, match="ownership/parity"):
        replace(
            valid_n2,
            pre_owner_slot_snapshots=future_pre_owner,
            post_owner_slot_snapshots=future_post_owner,
            unselected_slots_root_before_sha256=C.physical_slot_root(
                future_unselected_tuple
            ),
            unselected_slots_root_after_sha256=C.physical_slot_root(
                future_unselected_tuple
            ),
            physical_owner_checkpoint_before_sha256=(
                C.physical_owner_checkpoint_root(
                    capacity_receipt_sha256=capacity.canonical_sha256,
                    num_envs=2,
                    flight_capacity=capacity.configured_flight_capacity,
                    mutation_version=20,
                    next_prepare_nonce=29,
                    reset_generations=(7, 9),
                    slots=future_pre_owner,
                    poisoned=False,
                )
            ),
            physical_owner_checkpoint_after_sha256=(
                C.physical_owner_checkpoint_root(
                    capacity_receipt_sha256=capacity.canonical_sha256,
                    num_envs=2,
                    flight_capacity=capacity.configured_flight_capacity,
                    mutation_version=21,
                    next_prepare_nonce=29,
                    reset_generations=(8, 9),
                    slots=future_post_owner,
                    poisoned=False,
                )
            ),
        )

    wrong_generation_slots = list(unselected_slots_tuple)
    wrong_key = replace(env1_key, reset_generation=10)
    wrong_generation_slots[install.flight_slot] = replace(
        wrong_generation_slots[install.flight_slot],
        outcome_key=wrong_key,
        outcome_key_sha256=wrong_key.canonical_sha256,
    )
    wrong_generation_tuple = tuple(wrong_generation_slots)
    wrong_pre_owner = tuple(before) + wrong_generation_tuple
    wrong_post_owner = after + wrong_generation_tuple
    with pytest.raises(C.PhysicalFlightContractError, match="ownership/parity"):
        replace(
            valid_n2,
            pre_owner_slot_snapshots=wrong_pre_owner,
            post_owner_slot_snapshots=wrong_post_owner,
            unselected_slots_root_before_sha256=C.physical_slot_root(
                wrong_generation_tuple
            ),
            unselected_slots_root_after_sha256=C.physical_slot_root(
                wrong_generation_tuple
            ),
            physical_owner_checkpoint_before_sha256=(
                C.physical_owner_checkpoint_root(
                    capacity_receipt_sha256=capacity.canonical_sha256,
                    num_envs=2,
                    flight_capacity=capacity.configured_flight_capacity,
                    mutation_version=20,
                    next_prepare_nonce=29,
                    reset_generations=(7, 9),
                    slots=wrong_pre_owner,
                    poisoned=False,
                )
            ),
            physical_owner_checkpoint_after_sha256=(
                C.physical_owner_checkpoint_root(
                    capacity_receipt_sha256=capacity.canonical_sha256,
                    num_envs=2,
                    flight_capacity=capacity.configured_flight_capacity,
                    mutation_version=21,
                    next_prepare_nonce=29,
                    reset_generations=(8, 9),
                    slots=wrong_post_owner,
                    poisoned=False,
                )
            ),
        )


def test_checkpoint_carries_full_grid_full_owner_bytes_nonce_and_external_pin():
    capacity = _capacity(cadence=5, horizon=5)
    slots = tuple(
        _parked(
            capacity,
            env_id=env_id,
            slot=slot,
            version=29 if env_id == 0 else 31,
        )
        for env_id in range(2)
        for slot in range(capacity.configured_flight_capacity)
    )
    park_row = struct.pack(
        "<13f",
        0.0,
        0.0,
        -20.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    scene_raw = park_row * (2 * capacity.configured_flight_capacity)
    owner_state = {
        **C.PHYSICAL_OWNER_STATE_SCHEMA,
        "capacity_receipt_sha256": capacity.canonical_sha256,
        "num_envs": 2,
        "flight_capacity": capacity.configured_flight_capacity,
        "owner_mutation_version": 31,
        "next_prepare_nonce": 42,
        "reset_generation": [7, 8],
        "slot_snapshots": [slot.to_mapping() for slot in slots],
        "scene_state_shape": [2, capacity.configured_flight_capacity, 13],
        "scene_state_f32_base64": base64.b64encode(scene_raw).decode("ascii"),
        "scene_state_byte_length": len(scene_raw),
        "scene_state_bytes_sha256": hashlib.sha256(scene_raw).hexdigest(),
        "flight_lifecycle_code": [
            [0] * capacity.configured_flight_capacity for _ in range(2)
        ],
        "observation_ordinal": [
            [-1] * capacity.configured_flight_capacity for _ in range(2)
        ],
        "previous_ball_center_m": [
            [[0.0, 0.0, 0.0] for _ in range(capacity.configured_flight_capacity)]
            for _ in range(2)
        ],
        "device_fault": [
            [False] * capacity.configured_flight_capacity for _ in range(2)
        ],
        "pending_r06_settlement_ack": None,
        "poisoned": False,
    }
    owner_bytes = C.canonical_json_bytes(owner_state)
    receipt = C.PhysicalFlightCheckpointReceipt(
        integration_status=C.INTEGRATION_STATUS,
        capacity_receipt=capacity,
        capacity_receipt_sha256=capacity.canonical_sha256,
        checkpoint_boundary_authority=_pin(
            "fixture_checkpoint_boundary_receipt_v1",
            complete_env_step=True,
            rollout_storage_empty=True,
        ),
        num_envs=2,
        flight_capacity=capacity.configured_flight_capacity,
        mutation_version=31,
        next_prepare_nonce=42,
        pending_prepare_receipt_sha256=None,
        slot_snapshots=slots,
        slot_root_sha256=C.physical_slot_root(slots),
        owner_state_schema_sha256=C.PHYSICAL_OWNER_STATE_SCHEMA_SHA256,
        owner_state_bytes_base64=base64.b64encode(owner_bytes).decode("ascii"),
        owner_state_byte_length=len(owner_bytes),
        owner_state_bytes_sha256=hashlib.sha256(owner_bytes).hexdigest(),
        complete_env_step=True,
        env_reset_invoked=False,
    )
    restored = C.PhysicalFlightCheckpointReceipt.from_mapping(
        receipt.to_mapping(),
        expected_canonical_sha256=receipt.canonical_sha256,
    )
    assert restored == receipt
    with pytest.raises(C.PhysicalFlightContractError, match="full slot grid"):
        replace(receipt, slot_snapshots=slots[:-1])
    with pytest.raises(C.PhysicalFlightContractError, match="cannot retain"):
        replace(receipt, pending_prepare_receipt_sha256=_sha("pending"))
    with pytest.raises(C.PhysicalFlightContractError, match="byte pin"):
        replace(receipt, owner_state_byte_length=len(owner_bytes) + 1)
    pending_owner = copy.deepcopy(owner_state)
    pending_owner["pending_r06_settlement_ack"] = {
        "snapshot_root_sha256": _sha("transient-r06"),
        "owner_mutation_version": 3,
        "mailbox_slot": [
            [-1] * capacity.configured_flight_capacity for _ in range(2)
        ],
    }
    pending_bytes = C.canonical_json_bytes(pending_owner)
    with pytest.raises(C.PhysicalFlightContractError, match="transient R06"):
        replace(
            receipt,
            owner_state_bytes_base64=base64.b64encode(pending_bytes).decode(
                "ascii"
            ),
            owner_state_byte_length=len(pending_bytes),
            owner_state_bytes_sha256=hashlib.sha256(pending_bytes).hexdigest(),
        )
    future = list(slots)
    future[0] = replace(future[0], mutation_version=32)
    with pytest.raises(C.PhysicalFlightContractError, match="full slot grid"):
        replace(
            receipt,
            slot_snapshots=tuple(future),
            slot_root_sha256=C.physical_slot_root(tuple(future)),
        )


def test_module_is_portable_preintegration_and_has_no_science_defaults():
    assert C.RUNTIME_INTEGRATED is False
    assert C.POD_VALIDATED is False
    assert C.LAUNCH_AUTHORIZED is False
    source = Path(C.__file__).read_text(encoding="utf-8")
    forbidden = ("import torch", "import numpy", "import omni", "import isaaclab")
    assert not any(item in source for item in forbidden)
    assert "action_ball_continuous_runtime_transaction" not in source
    assert "whole_body_tracking.tasks" not in source
