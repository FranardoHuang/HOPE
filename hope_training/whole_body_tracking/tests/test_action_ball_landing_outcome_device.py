from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, fields, replace
import gc
import hashlib
import hmac
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Mapping
import weakref

import pytest
import torch


try:  # Local torch 2.0 exposes only the one-argument overload.
    torch._assert_async(torch.tensor(True), "probe")
except TypeError:  # pragma: no cover - the exact Isaac runtime accepts a message
    _torch_assert_async = torch._assert_async

    def _assert_async_compat(condition, message=""):
        try:
            return _torch_assert_async(condition)
        except RuntimeError as exc:
            raise RuntimeError(message) from exc

    torch._assert_async = _assert_async_compat


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
_MDP_ROOT = (
    _SOURCE_ROOT
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
for _path in (_SOURCE_ROOT, _MDP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import action_ball_landing_placement as cpu  # noqa: E402


_PATH = _MDP_ROOT / "action_ball_landing_outcome_device.py"
# Load under a canonical package identity without executing the task package's
# Isaac registration side effects.  The production module may therefore use
# its package-relative Torch scorer import while this focused unit test remains
# runnable in the lean Torch environment.
_PACKAGE_PATHS = (
    ("whole_body_tracking", _SOURCE_ROOT / "whole_body_tracking"),
    ("whole_body_tracking.tasks", _SOURCE_ROOT / "whole_body_tracking" / "tasks"),
    (
        "whole_body_tracking.tasks.tracking",
        _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking",
    ),
    (
        "whole_body_tracking.tasks.tracking.config",
        _SOURCE_ROOT
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config",
    ),
    (
        "whole_body_tracking.tasks.tracking.config.agibot_a3",
        _SOURCE_ROOT
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3",
    ),
    ("whole_body_tracking.tasks.tracking.mdp", _MDP_ROOT),
)
for _package_name, _package_path in _PACKAGE_PATHS:
    _package = sys.modules.get(_package_name)
    if _package is None:
        _package = ModuleType(_package_name)
        _package.__path__ = [str(_package_path)]
        sys.modules[_package_name] = _package
    elif not hasattr(_package, "__path__"):
        _package.__path__ = [str(_package_path)]

_CANONICAL_MODULE_NAME = (
    "whole_body_tracking.tasks.tracking.mdp.action_ball_landing_outcome_device"
)
D = sys.modules.get(_CANONICAL_MODULE_NAME)
if D is None:
    _SPEC = importlib.util.spec_from_file_location(_CANONICAL_MODULE_NAME, _PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    D = importlib.util.module_from_spec(_SPEC)
    sys.modules[_CANONICAL_MODULE_NAME] = D
    _SPEC.loader.exec_module(D)
setattr(
    sys.modules["whole_body_tracking.tasks.tracking.mdp"],
    "action_ball_landing_outcome_device",
    D,
)

_HELPER_MODULES: dict[str, object] = {}
_C10_PROJECTIONS: dict[str, tuple[object, Mapping[str, object], str]] = {}
_C10_AUTHORITIES: dict[str, object] = {}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _test_helper(filename: str) -> object:
    """Load an existing focused-test builder without copying its large fixture."""

    cached = _HELPER_MODULES.get(filename)
    if cached is not None:
        return cached
    path = Path(__file__).with_name(filename)
    module_name = f"_landing_outcome_device_helper_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _HELPER_MODULES[filename] = module
    return module


def _c10_projection(family: str) -> tuple[object, Mapping[str, object], str]:
    cached = _C10_PROJECTIONS.get(family)
    if cached is not None:
        return cached
    helper = _test_helper("test_action_ball_ac_family_contract.py")
    projection = helper._projection(family)
    artifact = helper.contract.c10_projection_artifact(projection)
    payload = artifact["projection"]
    identity_sha256 = helper.contract.canonical_sha256(payload["identity"])
    result = (projection, payload, identity_sha256)
    _C10_PROJECTIONS[family] = result
    return result


def _payment_authority(family: str) -> object:
    cached = _C10_AUTHORITIES.get(family)
    if cached is not None:
        return cached
    projection, _, identity_sha256 = _c10_projection(family)
    authority = D.build_c10_family_payment_authority(
        projection,
        expected_projection_sha256=projection.canonical_sha256,
        expected_identity_sha256=identity_sha256,
        expected_c10_contract_sha256=(
            _test_helper("test_action_ball_ac_family_contract.py")
            .contract.C10_CONTRACT_AUTHORITY_SHA256
        ),
    )
    _C10_AUTHORITIES[family] = authority
    return authority


def _profile() -> cpu.LandingPlacementProfile:
    # Test-only values.  Nothing in this fixture is an adopted R06 profile,
    # kernel scale, geometry, or recommendation.
    return cpu.LandingPlacementProfile(
        frame_id="fixture_env_frame",
        frame_binding_sha256="9" * 64,
        contact_source_semantics=cpu.SELECTED_RUBBER_CONTACT_AUTHORITY,
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


def _registry() -> D.LandingOutcomeTextRegistry:
    return D.LandingOutcomeTextRegistry(
        run_ids=("run-r05-focused",),
        carry_chain_ids=("carry-0",),
    )


def _capacity_authority(
    *,
    flight_slots: int,
    mailbox_slots: int,
    flight_horizon_ticks: int | None = None,
    mailbox_horizon_ticks: int | None = None,
) -> D.LandingOutcomeCapacityAuthority:
    """Mint explicit fixture C/H/K bytes without adopting science values."""

    cadence = 6
    flight_horizon = (
        (flight_slots - 1) * cadence + cadence - 1
        if flight_horizon_ticks is None
        else flight_horizon_ticks
    )
    mailbox_horizon = (
        (mailbox_slots - 1) * cadence + cadence - 1
        if mailbox_horizon_ticks is None
        else mailbox_horizon_ticks
    )
    assert flight_horizon // cadence + 1 == flight_slots
    assert mailbox_horizon // cadence + 1 == mailbox_slots
    policy_clock = {
        "fixture_only": True,
        "control_step_dt_ratio": {"numerator": 1, "denominator": 50},
    }
    resolved_graph_sha = _sha("fixture-resolved-graph")
    four_shot_sha = _sha("fixture-four-shot-tape")
    clock_payload = {
        "kind": D.CAPACITY_CLOCK_BINDING_KIND,
        "resolved_graph_receipt_sha256": resolved_graph_sha,
        "four_shot_tape_receipt_sha256": four_shot_sha,
        "policy_clock": policy_clock,
    }
    clock_sha = D._canonical_sha256(clock_payload)
    flight_witness_sha = _sha("fixture-flight-horizon-witness")
    mailbox_witness_sha = _sha("fixture-mailbox-horizon-witness")
    tail_closure_tick = max(flight_horizon, mailbox_horizon)
    order_payload = {
        "kind": D.CAPACITY_INCLUSIVE_EVENT_ORDER_KIND,
        "control_step_clock_sha256": clock_sha,
        "four_shot_tape_receipt_sha256": four_shot_sha,
        "interval_semantics": "closed_reveal_through_release_ticks",
        "same_tick_order": "new_reveal_admission_before_prior_owner_release",
        "capacity_formula": "floor(horizon_ticks/cadence_ticks)+1",
        "cadence_ticks": cadence,
        "flight_horizon_ticks": flight_horizon,
        "flight_horizon_witness_sha256": flight_witness_sha,
        "mailbox_horizon_ticks": mailbox_horizon,
        "mailbox_horizon_witness_sha256": mailbox_witness_sha,
        "tail_closure_tick": tail_closure_tick,
    }
    payload = {
        "schema_version": D.SCHEMA_VERSION,
        "kind": D.CAPACITY_AUTHORITY_KIND,
        "materialization_sha256": _sha("fixture-materialization"),
        "numeric_authority_sha256": _sha("fixture-numeric-authority"),
        "resolved_graph_receipt_sha256": resolved_graph_sha,
        "four_shot_tape_receipt_sha256": four_shot_sha,
        "policy_clock": policy_clock,
        "control_step_clock_sha256": clock_sha,
        "inclusive_event_order_witness_sha256": D._canonical_sha256(
            order_payload
        ),
        "cadence_ticks": cadence,
        "flight_horizon_ticks": flight_horizon,
        "flight_horizon_witness_sha256": flight_witness_sha,
        "required_flight_slot_capacity": flight_slots,
        "flight_slot_capacity": flight_slots,
        "mailbox_horizon_ticks": mailbox_horizon,
        "mailbox_horizon_witness_sha256": mailbox_witness_sha,
        "required_mailbox_capacity": mailbox_slots,
        "mailbox_capacity": mailbox_slots,
        "tail_closure_tick": tail_closure_tick,
    }
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    authority = D.LandingOutcomeCapacityAuthority(
        _payload_json=payload_json,
        _auth_tag=hmac.new(
            D._CAPACITY_AUTH_KEY,
            payload_json,
            hashlib.sha256,
        ).digest(),
        _token=D._CAPACITY_AUTH_TOKEN,
    )
    D._owned_capacity_authority(authority)
    return authority


def _binding(
    *,
    profile: cpu.LandingPlacementProfile,
    registry: D.LandingOutcomeTextRegistry,
    payment_authority: object,
    capacity_authority: D.LandingOutcomeCapacityAuthority,
) -> D.LandingOutcomeRuntimeBinding:
    # Positive values exercise explicit binding only.  They are neither
    # adopted Manager weights nor an adopted post-dt budget.
    return D.LandingOutcomeRuntimeBinding(
        common_on_table_manager_weight=(
            payment_authority.common_on_table_manager_weight
        ),
        placement_manager_shell_weight=(
            payment_authority.placement_manager_shell_weight
        ),
        post_dt_budget_sha256=payment_authority.post_dt_budget_sha256,
        text_registry_sha256=registry.canonical_sha256,
        capacity_authority_sha256=capacity_authority.canonical_sha256,
        numeric_materialization_sha256=(
            capacity_authority.materialization_sha256
        ),
        numeric_authority_sha256=capacity_authority.numeric_authority_sha256,
        resolved_graph_receipt_sha256=(
            capacity_authority.resolved_graph_receipt_sha256
        ),
        four_shot_tape_receipt_sha256=(
            capacity_authority.four_shot_tape_receipt_sha256
        ),
        control_step_clock_sha256=(
            capacity_authority.control_step_clock_sha256
        ),
        inclusive_event_order_witness_sha256=(
            capacity_authority.inclusive_event_order_witness_sha256
        ),
        flight_horizon_witness_sha256=(
            capacity_authority.flight_horizon_witness_sha256
        ),
        mailbox_horizon_witness_sha256=(
            capacity_authority.mailbox_horizon_witness_sha256
        ),
        cadence_ticks=capacity_authority.cadence_ticks,
        flight_horizon_ticks=capacity_authority.flight_horizon_ticks,
        mailbox_horizon_ticks=capacity_authority.mailbox_horizon_ticks,
        flight_slot_capacity=capacity_authority.flight_slot_capacity,
        mailbox_capacity=capacity_authority.mailbox_capacity,
        tail_closure_tick=capacity_authority.tail_closure_tick,
        r05_source_sha256=(
            D.R05_RUNTIME_TRANSACTION_SOURCE_SHA256
            or D.R05_RUNTIME_TRANSACTION_OBSERVED_SOURCE_SHA256
        ),
        r05_contract_sha256=D.R05_RUNTIME_TRANSACTION_CONTRACT_SHA256,
        c05_source_sha256=D.C05_LANDING_OUTCOME_SOURCE_SHA256,
        landing_profile_sha256=payment_authority.landing_profile_sha256,
        c10_contract_sha256=payment_authority.contract_sha256,
        c10_projection_sha256=payment_authority.projection_sha256,
        c10_identity_sha256=payment_authority.identity_sha256,
    )


def _coordinator(
    *,
    rows: int = 1,
    flight_slots: int = 1,
    mailbox_slots: int = 1,
    family: str = "A",
    flight_horizon_ticks: int | None = None,
    mailbox_horizon_ticks: int | None = None,
    bind_physical_park: bool = True,
    device: torch.device | str = "cpu",
) -> D.ActionBallLandingOutcomeDeviceCoordinator:
    profile = _profile()
    registry = _registry()
    payment_authority = _payment_authority(family)
    capacity_authority = _capacity_authority(
        flight_slots=flight_slots,
        mailbox_slots=mailbox_slots,
        flight_horizon_ticks=flight_horizon_ticks,
        mailbox_horizon_ticks=mailbox_horizon_ticks,
    )
    owner = D.ActionBallLandingOutcomeDeviceCoordinator(
        num_envs=rows,
        flight_slot_capacity=flight_slots,
        mailbox_capacity=mailbox_slots,
        device=device,
        dtype=torch.float32,
        profile=profile,
        runtime_binding=_binding(
            profile=profile,
            registry=registry,
            payment_authority=payment_authority,
            capacity_authority=capacity_authority,
        ),
        payment_authority=payment_authority,
        capacity_authority=capacity_authority,
        text_registry=registry,
    )
    # These are test-side host authorities, never runtime-device state.
    owner._test_text_registry = registry
    owner._test_payment_authority = payment_authority
    if bind_physical_park:
        _bind_test_physical_park_owner(owner)
    return owner


_TEST_PHYSICAL_MODULE = (
    "_action_ball_landing_outcome_device_physical_park_fixture"
)


def _test_physical_park_owner_type() -> type:
    """Build a narrow retained-token fixture for R06-focused tests.

    The real physical-first/R06-last scene transaction is covered by
    ``test_action_ball_physical_flight_device.py``.  This fixture deliberately
    models only that owner's prepared/committed token authority so the R06
    ledger tests cannot fall back to the tombstoned caller-selected retire API.
    """

    cached = sys.modules.get(_TEST_PHYSICAL_MODULE)
    if cached is not None:
        return cached.ActionBallPhysicalFlightDeviceOwner

    module = ModuleType(_TEST_PHYSICAL_MODULE)
    module.__file__ = str(_MDP_ROOT / "action_ball_physical_flight_device.py")

    class PhysicalParkCleanupMaskCapability:
        def __init__(
            self,
            *,
            device_mask: torch.Tensor,
            owner_identity: object,
            prepared_token: object,
        ) -> None:
            self._device_mask = device_mask
            self._owner_identity = owner_identity
            self._prepared_token = prepared_token

    class ActionBallPhysicalFlightDeviceOwner:
        def __init__(
            self,
            r06_owner: D.ActionBallLandingOutcomeDeviceCoordinator,
        ) -> None:
            self._owner_identity = object()
            self._r06_owner = r06_owner
            self._prepared_token: object | None = None
            self._claim: object | None = None
            self._committed_token: object | None = None

        def prepare_r06_physical_park(
            self,
            r06_prepared_retire: object,
        ) -> object:
            if self._prepared_token is not None:
                raise AssertionError("test physical park already has a prepare")
            prepared_token = object()
            physical_capability = PhysicalParkCleanupMaskCapability(
                device_mask=torch.zeros_like(
                    r06_prepared_retire._cleanup_mask,
                    dtype=torch.bool,
                ),
                owner_identity=self._owner_identity,
                prepared_token=prepared_token,
            )
            claim = D.LandingOutcomePhysicalParkPreparedTokenClaim(
                r06_prepared_retire=r06_prepared_retire,
                r06_cleanup_capability=(
                    r06_prepared_retire._cleanup_capability
                ),
                physical_cleanup_capability=physical_capability,
                _physical_prepared_token=prepared_token,
            )
            self._prepared_token = prepared_token
            self._claim = claim
            return prepared_token

        def commit_r06_physical_park(self, prepared_token: object) -> None:
            if (
                prepared_token is not self._prepared_token
                or self._claim is None
            ):
                raise AssertionError("test physical park token is stale")
            self._committed_token = prepared_token

        def require_owned_r06_physical_park_prepared_token(
            self,
            prepared_token: object,
        ) -> object:
            if (
                prepared_token is not self._prepared_token
                or self._claim is None
            ):
                raise AssertionError("test physical prepare is stale")
            return self._claim

        def require_committed_r06_physical_park_prepared_token(
            self,
            prepared_token: object,
        ) -> object:
            if (
                prepared_token is not self._prepared_token
                or prepared_token is not self._committed_token
                or self._claim is None
            ):
                raise AssertionError("test physical park is not committed")
            return self._claim

        def retire_completed(self) -> None:
            self._prepared_token = None
            self._claim = None
            self._committed_token = None

    for cls in (
        PhysicalParkCleanupMaskCapability,
        ActionBallPhysicalFlightDeviceOwner,
    ):
        cls.__module__ = _TEST_PHYSICAL_MODULE
        cls.__qualname__ = cls.__name__
        setattr(module, cls.__name__, cls)
    sys.modules[_TEST_PHYSICAL_MODULE] = module
    return ActionBallPhysicalFlightDeviceOwner


def _bind_test_physical_park_owner(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
) -> object:
    physical_type = _test_physical_park_owner_type()
    physical = physical_type(owner)
    authority = D.mint_landing_outcome_physical_park_token_authority(
        physical
    )
    owner.bind_physical_park_token_authority(physical, authority)
    owner._test_physical_park_owner = physical
    return physical


def _commit_physical_retire(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
    postphysics_result: D.PostPhysicsMutationResult,
) -> D.PhysicalRetireMutationResult:
    """Exercise the production all-grid physical-first/R06-last API."""

    physical = owner._test_physical_park_owner
    prepared = owner.prepare_physical_retire(
        postphysics_result,
        postphysics_result.settled_mask,
    )
    physical_prepared = physical.prepare_r06_physical_park(prepared)
    armed = owner.arm_physical_retire(prepared, physical_prepared)
    physical.commit_r06_physical_park(physical_prepared)
    result = owner.commit_prevalidated_physical_retire(armed)
    physical.retire_completed()
    return result


def _digest_rows(rows: int, offset: int) -> torch.Tensor:
    base = torch.arange(32, dtype=torch.int64).unsqueeze(0)
    env = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    return ((base + env * 17 + offset) % 255 + 1).to(dtype=torch.uint8)


def _hex_token(value: str, rows: int) -> torch.Tensor:
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8).unsqueeze(0).expand(rows, 32).clone()


def _key(
    rows: int = 1,
    *,
    swing: int = 0,
    uid_offset: int = 0,
    digest_offset: int = 0,
) -> D.DeviceLandingOutcomeKey:
    env = torch.arange(rows, dtype=torch.int64)
    digest_names = (
        "birth_sha256",
        "sample_sha256",
        "task_sha256",
        "run_id",
        "carry_chain_id",
        "source_sha256",
        "config_sha256",
        "receipt_content_sha256",
    )
    digests = {
        name: _digest_rows(rows, digest_offset + 11 * (index + 1))
        for index, name in enumerate(digest_names)
    }
    return D.DeviceLandingOutcomeKey(
        env_id=env,
        reset_generation=torch.ones(rows, dtype=torch.int64),
        swing_generation=torch.full((rows,), swing, dtype=torch.int64),
        action_uid=1000 + uid_offset + env,
        action_slot=(uid_offset + env) % 73,
        birth_sha256=digests["birth_sha256"],
        sample_sha256=digests["sample_sha256"],
        task_sha256=digests["task_sha256"],
        run_id=digests["run_id"],
        carry_chain_id=digests["carry_chain_id"],
        shot_index=torch.full((rows,), swing + 1, dtype=torch.int64),
        source_sha256=digests["source_sha256"],
        config_sha256=digests["config_sha256"],
        receipt_content_sha256=digests["receipt_content_sha256"],
    )


def _install(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
    *,
    ordinal: int = 0,
    reveal: int = 1,
    contact_deadline: int = 3,
    crossing_horizon: int = 6,
    target_xy: tuple[float, float] = (2.2, 0.0),
) -> tuple[torch.Tensor, D.DeviceLandingOutcomeKey, D.DeviceMutationResult]:
    """Seed one legacy-shaped R06 row for leaf physics/reward unit tests.

    This is intentionally not an ingress or authorization fixture.  The live
    fixed-N Physical -> R06 launch/full-key contract is exercised by the
    ActionEpoch direct and D05 row-wise suites.
    """

    del crossing_horizon  # The capacity witness owns this value.
    device = owner.device
    accepted = torch.zeros(owner.num_envs, dtype=torch.bool, device=device)
    rejected = torch.zeros_like(accepted)
    fault_bits = torch.zeros(owner.num_envs, dtype=torch.int64, device=device)
    empty_flights = torch.nonzero(
        owner._flight_state[0].eq(D.FLIGHT_EMPTY), as_tuple=False
    ).flatten()
    empty_mailboxes = torch.nonzero(
        ~owner._mailbox_reserved[0], as_tuple=False
    ).flatten()
    if empty_flights.numel() == 0 or empty_mailboxes.numel() == 0:
        rejected[0] = True
        fault_bits[0] = (
            D.FAULT_FLIGHT_COLLISION
            if empty_flights.numel() == 0
            else D.FAULT_MAILBOX_COLLISION
        )
        owner._ingress_fault_bits[0].bitwise_or_(fault_bits[0])
        owner._record_fault_events(fault_bits)
        owner._increment_mutation()
        return (
            torch.zeros((1, D.TOKEN_BYTES), dtype=torch.uint8, device=device),
            _key(swing=ordinal, uid_offset=100 * ordinal),
            owner._mutation_result(accepted, rejected, fault_bits),
        )

    flight_slot = int(empty_flights[0].item())
    mailbox_slot = int(empty_mailboxes[0].item())
    key = _key(
        swing=ordinal,
        uid_offset=100 * ordinal,
        digest_offset=37 * ordinal,
    )
    full_key = _digest_rows(1, 151 + 37 * ordinal).to(device=device)
    full_key_receipt = _digest_rows(1, 157 + 37 * ordinal).to(device=device)
    committed_reveal = _digest_rows(1, 163 + 37 * ordinal).to(device=device)
    install_receipt = _digest_rows(1, 169 + 37 * ordinal).to(device=device)
    task_token = _digest_rows(1, 175 + 37 * ordinal).to(device=device)

    for name in D._INT_KEY_FIELDS:
        owner._flight_key_ints[name][0, flight_slot].copy_(
            getattr(key, name)[0].to(device=device)
        )
    for name in D._DIGEST_KEY_FIELDS:
        owner._flight_key_digests[name][0, flight_slot].copy_(
            getattr(key, name)[0].to(device=device)
        )
    owner._flight_full_key_sha256[0, flight_slot].copy_(full_key[0])
    owner._flight_full_key_receipt_sha256[0, flight_slot].copy_(
        full_key_receipt[0]
    )
    owner._flight_committed_reveal_sha256[0, flight_slot].copy_(
        committed_reveal[0]
    )
    owner._flight_install_receipt_sha256[0, flight_slot].copy_(
        install_receipt[0]
    )
    owner._flight_task_identity_token[0, flight_slot].copy_(task_token[0])
    owner._flight_ball_generation[0, flight_slot] = ordinal
    owner._flight_mailbox_slot[0, flight_slot] = mailbox_slot
    owner._flight_target_xy_m[0, flight_slot].copy_(
        torch.tensor(target_xy, dtype=owner.dtype, device=device)
    )
    owner._flight_reveal_control_step[0, flight_slot] = reveal
    owner._flight_contact_deadline_control_step[0, flight_slot] = contact_deadline
    owner._flight_crossing_horizon_control_step[0, flight_slot] = (
        reveal + int(owner.capacity_authority.flight_horizon_ticks)
    )
    owner._flight_state[0, flight_slot] = D.FLIGHT_INBOUND
    owner._flight_physical_retired[0, flight_slot] = False
    owner._flight_fault_bits[0, flight_slot] = 0

    owner._mailbox_reserved[0, mailbox_slot] = True
    owner._mailbox_action_epoch[0, mailbox_slot] = False
    owner._mailbox_reservation_token[0, mailbox_slot].copy_(full_key[0])
    owner._mailbox_reservation_generation[0, mailbox_slot] = ordinal
    owner._mailbox_reserved_flight_slot[0, mailbox_slot] = flight_slot

    owner._replay_valid[0] = True
    owner._replay_reset_generation[0] = key.reset_generation[0].to(device=device)
    owner._replay_swing_generation[0] = key.swing_generation[0].to(device=device)
    owner._replay_action_epoch[0] = False
    owner._replay_full_key_sha256[0].copy_(full_key[0])
    owner._reset_generation_highwater[0] = torch.maximum(
        owner._reset_generation_highwater[0],
        key.reset_generation[0].to(device=device),
    )
    accepted[0] = True
    owner._installed_total.add_(1)
    owner._increment_mutation()
    return full_key, key, owner._mutation_result(accepted, rejected, fault_bits)


def _stamp(
    shape: tuple[int, int],
    mask: torch.Tensor,
    *,
    step: int,
    substep: int,
    phase: int,
) -> D.PhysicsStampBatch:
    device = mask.device
    return D.PhysicsStampBatch(
        control_step=torch.where(
            mask,
            torch.full(shape, step, dtype=torch.int64, device=device),
            torch.full(shape, -1, dtype=torch.int64, device=device),
        ),
        physics_substep=torch.where(
            mask,
            torch.full(shape, substep, dtype=torch.int32, device=device),
            torch.full(shape, -1, dtype=torch.int32, device=device),
        ),
        event_phase=torch.where(
            mask,
            torch.full(shape, phase, dtype=torch.int8, device=device),
            torch.full(shape, -1, dtype=torch.int8, device=device),
        ),
    )


def _post_one(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
    *,
    full_key: torch.Tensor,
    generation: int,
    flight_slot: int,
    ordinal: int,
    step: int,
    previous: tuple[float, float, float],
    current: tuple[float, float, float],
    contact: bool = False,
    contact_center: tuple[float, float, float] = (2.2, 0.0, 0.9),
    outgoing_anchor: tuple[float, float, float] = (2.2, 0.0, 0.9),
    net: bool = False,
    net_clear: bool = False,
    report_delivered: bool = True,
    crossing_event: bool = False,
    crossing_xy: tuple[float, float] = (2.2, 0.0),
    nonfinite: bool = False,
    producer_fault: bool = False,
    overflow: bool = False,
    event_substep: int = 2,
    contact_step: int | None = None,
    net_step: int | None = None,
    crossing_step: int | None = None,
    close_physical_retire: bool = True,
    consume_contact_authority: bool = True,
    physical_publication_identity: object | None = None,
) -> object:
    assert owner.num_envs == 1
    shape = (1, owner.flight_slot_capacity)
    device = owner.device
    observe = torch.zeros(shape, dtype=torch.bool, device=device)
    observe[0, flight_slot] = True
    hashes = torch.zeros(shape + (32,), dtype=torch.uint8, device=device)
    hashes[0, flight_slot] = full_key[0]
    generations = torch.full(shape, -1, dtype=torch.int64, device=device)
    generations[0, flight_slot] = generation
    ordinals = torch.full(shape, -1, dtype=torch.int64, device=device)
    ordinals[0, flight_slot] = ordinal
    previous_tensor = torch.zeros(
        shape + (3,), dtype=torch.float32, device=device
    )
    current_tensor = torch.zeros_like(previous_tensor)
    previous_tensor[0, flight_slot] = torch.tensor(previous, device=device)
    current_tensor[0, flight_slot] = torch.tensor(current, device=device)
    contact_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    contact_mask[0, flight_slot] = contact
    net_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    net_mask[0, flight_slot] = net
    clear_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    clear_mask[0, flight_slot] = net_clear
    report_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    report_mask[0, flight_slot] = report_delivered
    crossing_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    crossing_mask[0, flight_slot] = crossing_event
    contact_center_tensor = torch.zeros(
        shape + (3,), dtype=torch.float32, device=device
    )
    outgoing_anchor_tensor = torch.zeros_like(contact_center_tensor)
    contact_center_tensor[0, flight_slot] = torch.tensor(
        contact_center, device=device
    )
    outgoing_anchor_tensor[0, flight_slot] = torch.tensor(
        outgoing_anchor, device=device
    )
    crossing_xy_tensor = torch.zeros(
        shape + (2,), dtype=torch.float32, device=device
    )
    crossing_xy_tensor[0, flight_slot] = torch.tensor(
        crossing_xy, device=device
    )

    def one(value: bool) -> torch.Tensor:
        result = torch.zeros(shape, dtype=torch.bool, device=device)
        result[0, flight_slot] = value
        return result

    publication_identity = (
        object()
        if physical_publication_identity is None
        else physical_publication_identity
    )
    result = owner.publish_post_physics(
        D.PostPhysicsFlightBatch(
            observe_mask=observe,
            full_key_sha256=hashes,
            ball_generation=generations,
            observation_ordinal=ordinals,
            previous_ball_center_m=previous_tensor,
            current_ball_center_m=current_tensor,
            observation_stamp=_stamp(
                shape,
                observe,
                step=step,
                substep=event_substep,
                phase=D.PHASE_LANDING,
            ),
            selected_contact_event=contact_mask,
            selected_contact_ball_center_m=contact_center_tensor,
            selected_contact_outgoing_segment_anchor_m=outgoing_anchor_tensor,
            selected_contact_stamp=_stamp(
                shape,
                contact_mask,
                step=step if contact_step is None else contact_step,
                substep=event_substep,
                phase=D.PHASE_CONTACT,
            ),
            net_crossing_event=net_mask,
            net_clear_at_crossing=clear_mask,
            net_crossing_stamp=_stamp(
                shape,
                net_mask,
                step=step if net_step is None else net_step,
                substep=event_substep,
                phase=D.PHASE_NET,
            ),
            crossing_report_delivered=report_mask,
            first_descending_crossing_event=crossing_mask,
            first_descending_crossing_xy_m=crossing_xy_tensor,
            first_descending_crossing_stamp=_stamp(
                shape,
                crossing_mask,
                step=step if crossing_step is None else crossing_step,
                substep=event_substep,
                phase=D.PHASE_LANDING,
            ),
            nonfinite_observation=one(nonfinite),
            producer_contract_fault=one(producer_fault),
            engine_overflow=one(overflow),
            physical_publication_identity=publication_identity,
        )
    )
    if consume_contact_authority:
        owner.consume_owned_post_physics_contact_authority(
            result.contact_authority,
            expected_publication_identity=publication_identity,
        )
    if close_physical_retire:
        _commit_physical_retire(owner, result)
    return result


def _settle_on_table(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
    *,
    full_key: torch.Tensor,
    generation: int = 0,
    flight_slot: int = 0,
    crossing_event: bool = True,
) -> None:
    _post_one(
        owner,
        full_key=full_key,
        generation=generation,
        flight_slot=flight_slot,
        ordinal=0,
        step=1,
        previous=(2.1, 0.0, 1.1),
        current=(2.2, 0.0, 0.7),
        contact=True,
        contact_center=(2.2, 0.0, 0.9),
        outgoing_anchor=(2.2, 0.0, 0.9),
        net=True,
        net_clear=True,
        crossing_event=crossing_event,
        crossing_xy=(2.2, 0.0),
        event_substep=3,
    )


def _settle_batch_rows(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
    *,
    full_keys: torch.Tensor,
    ball_generations: torch.Tensor,
    observe_envs: tuple[int, ...],
    step: int,
    retire: bool = True,
) -> D.PostPhysicsMutationResult:
    shape = (owner.num_envs, owner.flight_slot_capacity)
    observe = torch.zeros(shape, dtype=torch.bool, device=owner.device)
    observe[list(observe_envs), 0] = True
    key_grid = torch.zeros(shape + (32,), dtype=torch.uint8, device=owner.device)
    if full_keys.ndim == 2:
        key_grid[:, 0].copy_(full_keys)
    else:
        key_grid.copy_(full_keys)
    generations = torch.full(
        shape, -1, dtype=torch.int64, device=owner.device
    )
    if ball_generations.ndim == 1:
        generations[:, 0].copy_(ball_generations)
    else:
        generations.copy_(ball_generations)
    previous = torch.zeros(shape + (3,), dtype=torch.float32, device=owner.device)
    current = torch.zeros_like(previous)
    contact_center = torch.zeros_like(previous)
    crossing_xy = torch.zeros(shape + (2,), dtype=torch.float32, device=owner.device)
    previous[list(observe_envs), 0] = torch.tensor((2.1, 0.0, 1.1), device=owner.device)
    current[list(observe_envs), 0] = torch.tensor((2.2, 0.0, 0.7), device=owner.device)
    contact_center[list(observe_envs), 0] = torch.tensor((2.2, 0.0, 0.9), device=owner.device)
    crossing_xy[list(observe_envs), 0] = torch.tensor((2.2, 0.0), device=owner.device)
    publication = object()
    result = owner.publish_post_physics(
        D.PostPhysicsFlightBatch(
            observe_mask=observe,
            full_key_sha256=key_grid,
            ball_generation=torch.where(observe, generations, -1),
            observation_ordinal=torch.where(
                observe,
                torch.zeros(shape, dtype=torch.int64, device=owner.device),
                torch.full(shape, -1, dtype=torch.int64, device=owner.device),
            ),
            previous_ball_center_m=previous,
            current_ball_center_m=current,
            observation_stamp=_stamp(shape, observe, step=step, substep=3, phase=D.PHASE_LANDING),
            selected_contact_event=observe,
            selected_contact_ball_center_m=contact_center,
            selected_contact_outgoing_segment_anchor_m=contact_center,
            selected_contact_stamp=_stamp(shape, observe, step=step, substep=3, phase=D.PHASE_CONTACT),
            net_crossing_event=observe,
            net_clear_at_crossing=observe,
            net_crossing_stamp=_stamp(shape, observe, step=step, substep=3, phase=D.PHASE_NET),
            crossing_report_delivered=observe,
            first_descending_crossing_event=observe,
            first_descending_crossing_xy_m=crossing_xy,
            first_descending_crossing_stamp=_stamp(shape, observe, step=step, substep=3, phase=D.PHASE_LANDING),
            nonfinite_observation=torch.zeros(shape, dtype=torch.bool, device=owner.device),
            producer_contract_fault=torch.zeros(shape, dtype=torch.bool, device=owner.device),
            engine_overflow=torch.zeros(shape, dtype=torch.bool, device=owner.device),
            physical_publication_identity=publication,
        )
    )
    owner.consume_owned_post_physics_contact_authority(
        result.contact_authority,
        expected_publication_identity=publication,
    )
    if retire:
        _commit_physical_retire(owner, result)
    return result


def _current_action_epoch_outcome(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
) -> SimpleNamespace:
    """Expose a settled leaf row through the production direct-R06 ABI.

    ``_install`` deliberately builds a legacy-shaped leaf row so physics can be
    unit-tested without minting ActionEpoch launch authority.  At this one
    classifier boundary we mark the already-retained row as ActionEpoch-owned;
    exact key/launch authority remains covered by the direct integration suite.
    """

    candidate = (
        owner._mailbox_history_valid
        & owner._mailbox_physical_retired
        & owner._mailbox_state.eq(D.MAILBOX_SETTLED_UNPAID)
    )
    owner._mailbox_action_epoch.copy_(candidate)
    rows = owner.project_current_action_epoch_outcome_rows()
    assert rows.valid.shape == (1,)
    assert rows.valid[0].item() is True
    bits = int(rows.valid_bits[0].item())
    values = rows.fact_values[0]

    def fact(offset: int) -> bool:
        return bool(values[offset].item())

    return SimpleNamespace(
        rows=rows,
        present=bool(bits & D.R06_ACTION_EPOCH_PRESENT),
        source_valid=bool(bits & D.R06_ACTION_EPOCH_SOURCE_VALID),
        policy_eligible=bool(bits & D.R06_ACTION_EPOCH_POLICY_ELIGIBLE),
        settlement_cause=int(rows.outcome_code[0].item()),
        owner_fault_bits=int(rows.owner_fault_bits[0].item()),
        common_on_table_outcome=fact(D.R06_ACTION_EPOCH_COMMON_ON_TABLE_F32),
        canonical_total=float(
            values[D.R06_ACTION_EPOCH_CANONICAL_TOTAL_F32].item()
        ),
        on_opponent_table=fact(D.R06_ACTION_EPOCH_ON_TABLE_F32),
        contact_valid=fact(D.R06_ACTION_EPOCH_CONTACT_VALID_F32),
        first_plane_crossing_valid=fact(
            D.R06_ACTION_EPOCH_CROSSING_VALID_F32
        ),
        ball_center_net_crossed=fact(D.R06_ACTION_EPOCH_NET_CROSSED_F32),
        ball_center_net_clear=fact(D.R06_ACTION_EPOCH_NET_CLEAR_F32),
    )


def _assert_safety_cleanup_hidden_from_policy(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
    outcome: SimpleNamespace,
    *,
    expected_causes: tuple[int, ...],
) -> None:
    """A producer fault remains diagnostic but cannot enter policy facts."""

    assert outcome.present is True
    assert outcome.source_valid is False
    assert outcome.policy_eligible is False
    assert outcome.settlement_cause in expected_causes
    assert outcome.owner_fault_bits != 0
    assert torch.count_nonzero(outcome.rows.fact_values).item() == 0
    assert owner._device_sticky_poison.item() is True


def _assert_deep_equal(left: object, right: object, path: str = "$") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left, right), path
        return
    if hasattr(left, "__dataclass_fields__"):
        assert type(right) is type(left), path
        for field in fields(left):
            _assert_deep_equal(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{path}.{field.name}",
            )
        return
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping), path
        assert set(left) == set(right), path
        for name in left:
            _assert_deep_equal(left[name], right[name], f"{path}.{name}")
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_deep_equal(left_item, right_item, f"{path}[{index}]")
        return
    assert left == right, path


def _assert_same_checkpoint_state(
    before: Mapping[str, object], after: Mapping[str, object]
) -> None:
    for name in (
        "schema_version",
        "num_envs",
        "flight_slot_capacity",
        "mailbox_capacity",
        "dtype",
        "profile_payload",
        "runtime_binding",
        "payment_authority",
        "mutation_version",
        "tensor_manifest",
        "tensor_bytes_sha256",
        "tensors",
    ):
        _assert_deep_equal(before[name], after[name], f"$.{name}")


def _business_state_snapshot(
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
) -> dict[str, torch.Tensor]:
    """Exclude only incident/fault telemetry from the runtime-owned tensors."""

    return {
        name: tensor.detach().clone()
        for name, tensor in owner._checkpoint_tensors().items()
        if "fault" not in name
        and name not in ("mutation_version", "terminal_resolution_total")
    }


def _assert_business_state_unchanged(
    before: Mapping[str, torch.Tensor],
    owner: D.ActionBallLandingOutcomeDeviceCoordinator,
) -> None:
    after = _business_state_snapshot(owner)
    assert set(after) == set(before)
    for name, expected in before.items():
        assert torch.equal(after[name], expected), name


def _reseal_checkpoint(checkpoint: dict[str, object]) -> str:
    tensors = checkpoint["tensors"]
    assert isinstance(tensors, Mapping)
    checkpoint["tensor_manifest"] = D._tensor_manifest(tensors)
    checkpoint["tensor_bytes_sha256"] = D._tensor_bytes_sha256(tensors)
    checkpoint["checkpoint_content_sha256"] = D._checkpoint_content_sha256(
        checkpoint
    )
    return checkpoint["checkpoint_content_sha256"]


def test_contract_has_two_explicit_capacities_and_c05_fourteen_field_order():
    signature = inspect.signature(D.ActionBallLandingOutcomeDeviceCoordinator)
    for name in (
        "flight_slot_capacity",
        "mailbox_capacity",
        "profile",
        "runtime_binding",
        "payment_authority",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty
    for field in fields(D.LandingOutcomeRuntimeBinding):
        assert field.default is MISSING
        assert field.default_factory is MISSING
    assert tuple(field.name for field in fields(D.DeviceLandingOutcomeKey)) == (
        "env_id",
        "reset_generation",
        "swing_generation",
        "action_uid",
        "action_slot",
        "birth_sha256",
        "sample_sha256",
        "task_sha256",
        "run_id",
        "carry_chain_id",
        "shot_index",
        "source_sha256",
        "config_sha256",
        "receipt_content_sha256",
    )
    assert D.RUNTIME_INTEGRATED is False
    assert D.CUDA_PROFILED is False
    assert D.FORMAL_EXACT_RESUME_INTEGRATED is False
    assert D.LAUNCH_AUTHORIZED is False
    assert "FlightLifecycleSnapshotBatch" in D.__all__
    assert "PostPhysicsMutationResult" in D.__all__


def test_physical_owner_gets_exact_lifecycle_and_typed_settlement_rows():
    owner = _coordinator()
    full_key, _, installed = _install(owner)
    assert installed.accepted.item() is True

    before = owner.current_flight_lifecycle_snapshot()
    assert isinstance(before, D.FlightLifecycleSnapshotBatch)
    assert before.state[0, 0].item() == D.FLIGHT_INBOUND
    assert torch.equal(before.full_key_sha256[0, 0], full_key[0])
    assert before.ball_generation[0, 0].item() == 0
    assert before.mailbox_slot[0, 0].item() == 0

    settled = _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.1, 0.0, 1.1),
        current=(2.2, 0.0, 0.7),
        contact=True,
        contact_center=(2.2, 0.0, 0.9),
        outgoing_anchor=(2.2, 0.0, 0.9),
        net=True,
        net_clear=True,
        crossing_event=True,
        crossing_xy=(2.2, 0.0),
        event_substep=3,
        close_physical_retire=False,
    )
    assert isinstance(settled, D.PostPhysicsMutationResult)
    assert settled.settled_mask[0, 0].item() is True
    assert (
        settled.settlement_cause[0, 0].item()
        == D.SETTLEMENT_CAUSE_FIRST_CROSSING
    )
    assert settled.flight_slot[0, 0].item() == 0
    assert torch.equal(settled.full_key_sha256[0, 0], full_key[0])
    assert settled.ball_generation[0, 0].item() == 0

    after = owner.current_flight_lifecycle_snapshot()
    assert after.state[0, 0].item() == D.FLIGHT_SETTLED_RETAINED
    assert torch.equal(after.full_key_sha256[0, 0], full_key[0])
    assert after.ball_generation[0, 0].item() == 0
    assert after.mailbox_slot[0, 0].item() == 0
    assert after.mailbox_state[0, 0].item() == D.MAILBOX_SETTLED_UNPAID
    assert torch.equal(after.mailbox_full_key_sha256[0, 0], full_key[0])
    assert after.mailbox_ball_generation[0, 0].item() == 0
    assert after.mailbox_reserved_flight_slot[0, 0].item() == 0
    assert after.mailbox_history_valid[0, 0].item() is True
    assert after.mailbox_physical_retired[0, 0].item() is False
    for name in D._KEY_FIELDS:
        assert torch.equal(
            getattr(after.mailbox_task_key, name)[0, 0],
            getattr(after.task_key, name)[0, 0],
        )
    assert torch.equal(after.mutation_version, settled.mutation_version)

    retired = _commit_physical_retire(owner, settled)
    assert retired.accepted.item() is True


def test_postphysics_result_is_an_exclusive_pending_capability_even_when_empty():
    owner = _coordinator(bind_physical_park=False)
    full_key, _, installed = _install(owner)
    assert installed.accepted.item() is True
    result = _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.0, 0.0, 1.2),
        current=(2.1, 0.0, 1.1),
        report_delivered=False,
        close_physical_retire=False,
    )
    assert result.settled_mask.any().item() is False
    snapshot = owner.current_flight_lifecycle_snapshot()
    assert snapshot.state[0, 0].item() in (D.FLIGHT_INBOUND, D.FLIGHT_OPEN)
    with pytest.raises(D.LandingOutcomeDeviceError, match="unclosed post-physics"):
        owner.project_current_action_epoch_outcome_rows()
    with pytest.raises(D.LandingOutcomeDeviceError, match="unclosed post-physics"):
        owner.drain_ppo_boundary(update_index=0)
    with pytest.raises(D.LandingOutcomeDeviceError, match="unclosed post-physics"):
        owner.state_dict(object())
    with pytest.raises(D.LandingOutcomeDeviceError, match="unclosed post-physics"):
        _post_one(
            owner,
            full_key=full_key,
            generation=0,
            flight_slot=0,
            ordinal=1,
            step=2,
            previous=(2.1, 0.0, 1.1),
            current=(2.2, 0.0, 1.0),
            report_delivered=False,
            close_physical_retire=False,
        )
    with pytest.raises(D.LandingOutcomeDeviceError, match="not bound"):
        owner.prepare_physical_retire(result, result.settled_mask)
    assert owner._latest_post_physics_settlement.result is result


def test_postphysics_contact_authority_is_causal_exact_and_one_shot():
    owner = _coordinator()
    full_key, key, installed = _install(owner)
    assert installed.accepted.item() is True
    publication = object()
    result = _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.0, 0.0, 1.2),
        current=(2.1, 0.0, 1.1),
        contact=True,
        report_delivered=False,
        close_physical_retire=False,
        consume_contact_authority=False,
        physical_publication_identity=publication,
    )
    assert result.physical_publication_identity is publication
    assert result.new_valid_contact_mask.item() is True
    assert result.contact_authority.publication_identity is publication
    authority_mask = result.contact_authority.new_valid_contact_mask
    assert authority_mask.item() is True
    authority_mask.zero_()
    assert result.contact_authority.new_valid_contact_mask.item() is True
    # The raw caller bit and cumulative live fact are evidence only.  Mutating
    # either clone cannot forge or change the retained causal authority.
    result.new_valid_contact_mask.zero_()
    assert owner.flight_state.item() == D.FLIGHT_SETTLED_RETAINED
    swapped = object()
    with pytest.raises(D.LandingOutcomeDeviceError, match="publication-swapped"):
        owner.consume_owned_post_physics_contact_authority(
            result.contact_authority,
            expected_publication_identity=swapped,
        )
    with pytest.raises(D.LandingOutcomeDeviceError, match="unconsumed"):
        owner.prepare_physical_retire(result, result.settled_mask)

    view = owner.consume_owned_post_physics_contact_authority(
        result.contact_authority,
        expected_publication_identity=publication,
    )
    assert view.publication_identity is publication
    assert view.new_valid_contact_mask.item() is True
    assert view.task_key.action_uid.item() == key.action_uid.item()
    assert torch.equal(view.full_key_sha256[0, 0], full_key[0])
    assert view.ball_generation.item() == 0
    assert view.flight_slot.item() == 0
    assert view.observation_ordinal.item() == 0
    assert view.selected_contact_stamp.control_step.item() == 1
    assert torch.equal(view.mutation_version, result.mutation_version)
    view.new_valid_contact_mask.zero_()
    with pytest.raises(D.LandingOutcomeDeviceError, match="replayed"):
        owner.consume_owned_post_physics_contact_authority(
            result.contact_authority,
            expected_publication_identity=publication,
        )

    _commit_physical_retire(owner, result)


def test_postphysics_contact_authority_rejects_raw_or_cumulative_contact_forgery():
    mismatch_owner = _coordinator()
    mismatch_key, _, installed = _install(mismatch_owner)
    assert installed.accepted.item() is True
    mismatch_publication = object()
    mismatch = _post_one(
        mismatch_owner,
        full_key=mismatch_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.0, 0.0, 1.2),
        current=(2.1, 0.0, 1.1),
        contact=True,
        contact_center=(2.2, 0.0, 0.9),
        outgoing_anchor=(2.3, 0.0, 0.9),
        report_delivered=False,
        close_physical_retire=False,
        consume_contact_authority=False,
        physical_publication_identity=mismatch_publication,
    )
    assert mismatch.new_valid_contact_mask.item() is False
    mismatch_view = (
        mismatch_owner.consume_owned_post_physics_contact_authority(
            mismatch.contact_authority,
            expected_publication_identity=mismatch_publication,
        )
    )
    assert mismatch_view.new_valid_contact_mask.item() is False
    _commit_physical_retire(mismatch_owner, mismatch)

    # Cumulative state can be true while the causal new-contact mask is false.
    # Exercise that exact distinction directly on a second open-flight cadence
    # without using another rollout or a caller-owned raw event bit.
    owner = _coordinator()
    full_key, _, installed = _install(owner)
    assert installed.accepted.item() is True
    owner._flight_state.fill_(D.FLIGHT_OPEN)
    owner._flight_contact_valid.fill_(True)
    owner._flight_contact_stamp_control.fill_(0)
    owner._flight_contact_stamp_substep.fill_(0)
    owner._flight_outgoing_anchor_m[0, 0] = torch.tensor((2.0, 0.0, 1.2))
    owner._flight_observation_ordinal.fill_(0)
    owner._flight_last_ball_center_m[0, 0] = torch.tensor((2.0, 0.0, 1.2))
    owner._flight_last_observation_control.fill_(1)
    owner._flight_last_observation_substep.fill_(0)
    publication = object()
    result = _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=1,
        step=2,
        previous=(2.0, 0.0, 1.2),
        current=(2.1, 0.0, 1.1),
        contact=False,
        report_delivered=False,
        close_physical_retire=False,
        consume_contact_authority=False,
        physical_publication_identity=publication,
    )
    result.new_valid_contact_mask.fill_(True)
    view = owner.consume_owned_post_physics_contact_authority(
        result.contact_authority,
        expected_publication_identity=publication,
    )
    assert view.new_valid_contact_mask.item() is False
    assert owner._flight_contact_valid.item() is True
    _commit_physical_retire(owner, result)


def test_postphysics_contact_authority_is_owner_local_and_all_false_is_legal():
    owner = _coordinator()
    foreign_owner = _coordinator()
    full_key, _, installed = _install(owner)
    assert installed.accepted.item() is True
    publication = object()
    result = _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.0, 0.0, 1.2),
        current=(2.1, 0.0, 1.1),
        contact=False,
        report_delivered=False,
        close_physical_retire=False,
        consume_contact_authority=False,
        physical_publication_identity=publication,
    )
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale, foreign"):
        foreign_owner.consume_owned_post_physics_contact_authority(
            result.contact_authority,
            expected_publication_identity=publication,
        )
    view = owner.consume_owned_post_physics_contact_authority(
        result.contact_authority,
        expected_publication_identity=publication,
    )
    assert view.new_valid_contact_mask.any().item() is False
    assert owner._poisoned is False
    _commit_physical_retire(owner, result)


def test_cuda_postphysics_contact_authority_is_device_only_and_clone_sealed(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("focused CUDA contact-authority gate requires CUDA")
    cpu_owner = _coordinator()
    full_key, key, installed = _install(cpu_owner)
    assert installed.accepted.item() is True
    owner = _coordinator(device=torch.device("cuda", torch.cuda.current_device()))
    device_key = D.DeviceLandingOutcomeKey(
        **{
            name: getattr(key, name).to(owner.device)
            for name in D._KEY_FIELDS
        }
    )
    device_full_key = full_key.to(owner.device)
    shape = owner._flight_shape
    owner._flight_state.fill_(D.FLIGHT_INBOUND)
    owner._flight_physical_retired.zero_()
    for name in D._INT_KEY_FIELDS:
        owner._flight_key_ints[name].copy_(getattr(device_key, name).reshape(shape))
    for name in D._DIGEST_KEY_FIELDS:
        owner._flight_key_digests[name].copy_(
            getattr(device_key, name).reshape(shape + (D.TOKEN_BYTES,))
        )
    owner._flight_full_key_sha256.copy_(
        device_full_key.reshape(shape + (D.TOKEN_BYTES,))
    )
    owner._flight_ball_generation.zero_()
    owner._flight_mailbox_slot.zero_()
    owner._flight_reveal_control_step.fill_(1)
    owner._flight_contact_deadline_control_step.fill_(3)
    owner._flight_crossing_horizon_control_step.fill_(6)
    owner._flight_last_ball_center_m[0, 0] = torch.tensor(
        (2.0, 0.0, 1.2), dtype=torch.float32, device=owner.device
    )
    publication = object()
    original_item = torch.Tensor.item
    original_bool = torch.Tensor.__bool__

    def forbidden_item(value):
        if value.is_cuda:
            raise AssertionError("contact authority synchronized CUDA via item")
        return original_item(value)

    def forbidden_bool(value):
        if value.is_cuda:
            raise AssertionError("contact authority synchronized CUDA via bool")
        return original_bool(value)

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden_bool)
    result = _post_one(
        owner,
        full_key=device_full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.0, 0.0, 1.2),
        current=(2.1, 0.0, 1.1),
        contact=True,
        report_delivered=False,
        close_physical_retire=False,
        consume_contact_authority=False,
        physical_publication_identity=publication,
    )
    view = owner.consume_owned_post_physics_contact_authority(
        result.contact_authority,
        expected_publication_identity=publication,
    )
    view.new_valid_contact_mask.zero_()
    assert type(result.contact_authority) is D.LandingOutcomePostPhysicsContactAuthority
    _commit_physical_retire(owner, result)


def test_direct_r06_same_substep_contact_net_landing_is_legal():
    owner = _coordinator()
    full_key, _, installed = _install(owner)
    assert installed.accepted.item() is True
    _settle_on_table(owner, full_key=full_key)

    outcome = _current_action_epoch_outcome(owner)
    assert outcome.present is True
    assert outcome.source_valid is True
    assert outcome.policy_eligible is True
    assert outcome.settlement_cause == D.SETTLEMENT_CAUSE_FIRST_CROSSING
    assert outcome.contact_valid is True
    assert outcome.first_plane_crossing_valid is True
    assert outcome.ball_center_net_crossed is True
    assert outcome.ball_center_net_clear is True
    assert outcome.on_opponent_table is True
    assert outcome.common_on_table_outcome is True
    assert outcome.canonical_total > 0.0


@pytest.mark.parametrize("mode", ("segment_only", "event_only", "absent", "mismatch"))
def test_direct_r06_c05_crossing_report_matrix(mode: str):
    owner = _coordinator()
    full_key, _, _ = _install(owner)
    _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.0, 0.0, 1.1),
        current=(2.1, 0.0, 1.0),
        contact=True,
        contact_center=(2.1, 0.0, 1.02),
        outgoing_anchor=(2.1, 0.0, 1.02),
    )
    segment = mode in ("segment_only", "mismatch")
    event = mode in ("event_only", "mismatch")
    _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=1,
        step=2,
        previous=(2.1, 0.0, 1.0),
        current=(2.2, 0.0, 0.7) if segment else (2.2, 0.0, 1.01),
        net=True,
        net_clear=True,
        report_delivered=mode != "absent",
        crossing_event=event,
        crossing_xy=(2.4, 0.0) if mode == "mismatch" else (2.2, 0.0),
    )
    outcome = _current_action_epoch_outcome(owner)
    if mode in ("segment_only", "event_only"):
        assert outcome.source_valid is True
        assert outcome.settlement_cause == D.SETTLEMENT_CAUSE_FIRST_CROSSING
        assert outcome.first_plane_crossing_valid is True
    else:
        _assert_safety_cleanup_hidden_from_policy(
            owner,
            outcome,
            expected_causes=(D.SETTLEMENT_CAUSE_PRODUCER_CONTRACT_FAULT,),
        )


@pytest.mark.parametrize("crossing_mode", ("event_only", "segment_and_event"))
def test_direct_r06_distinguishes_policy_nonfinite_from_infrastructure_fault(
    crossing_mode: str,
):
    crossing = _coordinator()
    crossing_hash, _, _ = _install(crossing)
    _post_one(
        crossing,
        full_key=crossing_hash,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.1, 0.0, 1.1),
        current=(
            (2.2, 0.0, 1.0)
            if crossing_mode == "event_only"
            else (2.2, 0.0, 0.7)
        ),
        contact=True,
        net=True,
        net_clear=True,
        report_delivered=True,
        crossing_event=True,
        crossing_xy=(float("nan"), 0.0),
    )
    policy_outcome = _current_action_epoch_outcome(crossing)
    assert policy_outcome.settlement_cause == D.SETTLEMENT_CAUSE_NONFINITE
    assert policy_outcome.source_valid is True
    assert policy_outcome.policy_eligible is True
    assert policy_outcome.first_plane_crossing_valid is False
    assert policy_outcome.canonical_total == 0.0

    unrelated = _coordinator()
    unrelated_hash, _, _ = _install(unrelated)
    _post_one(
        unrelated,
        full_key=unrelated_hash,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.1, 0.0, 1.1),
        current=(2.1, 0.0, 1.0),
        contact=True,
    )
    _post_one(
        unrelated,
        full_key=unrelated_hash,
        generation=0,
        flight_slot=0,
        ordinal=1,
        step=2,
        previous=(2.1, 0.0, 1.0),
        current=(2.2, 0.0, 0.9),
        nonfinite=True,
    )
    _assert_safety_cleanup_hidden_from_policy(
        unrelated,
        _current_action_epoch_outcome(unrelated),
        expected_causes=(D.SETTLEMENT_CAUSE_NONFINITE,),
    )


def test_direct_r06_contact_deadline_ignores_incoming_events():
    owner = _coordinator()
    full_key, _, _ = _install(owner, contact_deadline=3)
    _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.2, 0.0, 0.9),
        current=(2.2, 0.0, 0.7),
        net=True,
        net_clear=True,
        crossing_event=True,
    )
    for ordinal, step, previous, current in (
        (1, 2, (2.2, 0.0, 0.7), (2.2, 0.0, 0.8)),
        (2, 3, (2.2, 0.0, 0.8), (2.2, 0.0, 0.81)),
    ):
        _post_one(
            owner,
            full_key=full_key,
            generation=0,
            flight_slot=0,
            ordinal=ordinal,
            step=step,
            previous=previous,
            current=current,
        )
    outcome = _current_action_epoch_outcome(owner)
    assert outcome.source_valid is True
    assert outcome.settlement_cause == D.SETTLEMENT_CAUSE_CONTACT_DEADLINE
    assert outcome.contact_valid is False
    assert outcome.ball_center_net_crossed is False
    assert outcome.first_plane_crossing_valid is False
    assert outcome.common_on_table_outcome is False
    assert outcome.canonical_total == 0.0


@pytest.mark.parametrize(
    "fault",
    ("ordinal_gap", "endpoint_gap", "future_event", "late_contact", "anchor_mismatch"),
)
def test_direct_r06_observation_and_stamp_faults_fail_closed(fault: str):
    owner = _coordinator(flight_horizon_ticks=3)
    full_key, _, _ = _install(owner, contact_deadline=2, crossing_horizon=4)
    if fault == "late_contact":
        _post_one(
            owner,
            full_key=full_key,
            generation=0,
            flight_slot=0,
            ordinal=0,
            step=1,
            previous=(2.0, 0.0, 1.1),
            current=(2.1, 0.0, 1.0),
        )
        _post_one(
            owner,
            full_key=full_key,
            generation=0,
            flight_slot=0,
            ordinal=1,
            step=3,
            previous=(2.1, 0.0, 1.0),
            current=(2.2, 0.0, 0.9),
            contact=True,
            contact_step=3,
        )
    else:
        _post_one(
            owner,
            full_key=full_key,
            generation=0,
            flight_slot=0,
            ordinal=0,
            step=1,
            previous=(2.0, 0.0, 1.1),
            current=(2.1, 0.0, 1.0),
            contact=True,
            contact_center=(2.1, 0.0, 1.02),
            outgoing_anchor=(
                (2.5, 0.0, 1.02)
                if fault == "anchor_mismatch"
                else (2.1, 0.0, 1.02)
            ),
        )
        if fault != "anchor_mismatch":
            _post_one(
                owner,
                full_key=full_key,
                generation=0,
                flight_slot=0,
                ordinal=2 if fault == "ordinal_gap" else 1,
                step=2,
                previous=(2.0, 0.0, 1.0)
                if fault == "endpoint_gap"
                else (2.1, 0.0, 1.0),
                current=(2.2, 0.0, 0.9),
                crossing_event=fault == "future_event",
                crossing_step=3 if fault == "future_event" else None,
            )
    _assert_safety_cleanup_hidden_from_policy(
        owner,
        _current_action_epoch_outcome(owner),
        expected_causes=(
            D.SETTLEMENT_CAUSE_PRODUCER_CONTRACT_FAULT,
            D.SETTLEMENT_CAUSE_PROTOCOL_FAULT,
        ),
    )


def test_direct_r06_plane_endpoint_and_exact_crossing_horizon():
    crossing = _coordinator(flight_horizon_ticks=2)
    crossing_hash, _, _ = _install(
        crossing, contact_deadline=2, crossing_horizon=3
    )
    _post_one(
        crossing,
        full_key=crossing_hash,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.2, 0.0, 0.9),
        current=(2.2, 0.0, 0.78),
        contact=True,
        contact_center=(2.2, 0.0, 0.78),
        outgoing_anchor=(2.2, 0.0, 0.78),
    )
    _post_one(
        crossing,
        full_key=crossing_hash,
        generation=0,
        flight_slot=0,
        ordinal=1,
        step=2,
        previous=(2.2, 0.0, 0.78),
        current=(2.2, 0.0, 0.7),
        net=True,
        net_clear=True,
    )
    assert (
        _current_action_epoch_outcome(crossing).settlement_cause
        == D.SETTLEMENT_CAUSE_FIRST_CROSSING
    )

    horizon = _coordinator(flight_horizon_ticks=1)
    horizon_hash, _, _ = _install(
        horizon, contact_deadline=2, crossing_horizon=2
    )
    _post_one(
        horizon,
        full_key=horizon_hash,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.2, 0.0, 1.1),
        current=(2.2, 0.0, 1.0),
        contact=True,
        contact_center=(2.2, 0.0, 1.02),
        outgoing_anchor=(2.2, 0.0, 1.02),
    )
    _post_one(
        horizon,
        full_key=horizon_hash,
        generation=0,
        flight_slot=0,
        ordinal=1,
        step=2,
        previous=(2.2, 0.0, 1.0),
        current=(2.2, 0.0, 0.99),
    )
    horizon_outcome = _current_action_epoch_outcome(horizon)
    assert horizon_outcome.settlement_cause == D.SETTLEMENT_CAUSE_CROSSING_HORIZON
    assert horizon_outcome.canonical_total == 0.0


def test_direct_r06_retained_q0_does_not_block_q1_physical_slot_reuse():
    owner = _coordinator(flight_slots=1, mailbox_slots=2)
    q0_hash, _, _ = _install(owner)
    _settle_on_table(owner, full_key=q0_hash)
    assert owner.flight_state.item() == D.FLIGHT_EMPTY
    assert owner._mailbox_physical_retired[0, 0].item() is True

    q1_hash, _, installed = _install(
        owner,
        ordinal=1,
        reveal=5,
        contact_deadline=7,
        crossing_horizon=10,
    )
    assert installed.accepted.item() is True
    assert owner.flight_state.item() == D.FLIGHT_INBOUND
    assert q1_hash.ne(q0_hash).any()
    q0_outcome = _current_action_epoch_outcome(owner)
    assert q0_outcome.source_valid is True
    assert q0_outcome.common_on_table_outcome is True


def test_direct_r06_geometry_hit_without_net_clear_is_not_legal_return():
    owner = _coordinator()
    full_key, _, _ = _install(owner)
    _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.1, 0.0, 1.1),
        current=(2.2, 0.0, 0.7),
        contact=True,
        net=True,
        net_clear=False,
        crossing_event=True,
        crossing_xy=(2.2, 0.0),
    )
    outcome = _current_action_epoch_outcome(owner)
    assert outcome.source_valid is True
    assert outcome.on_opponent_table is True
    assert outcome.ball_center_net_crossed is True
    assert outcome.ball_center_net_clear is False
    assert outcome.common_on_table_outcome is False
    assert outcome.canonical_total == 0.0


def test_c10_projection_authenticates_family_gain_and_rejects_caller_reversal():
    authorities = {}
    for family, expected_gain in (("A", 1.0), ("C", 0.0)):
        projection, _, identity_sha256 = _c10_projection(family)
        kwargs = {
            "projection": projection,
            "expected_projection_sha256": projection.canonical_sha256,
            "expected_identity_sha256": identity_sha256,
            "expected_c10_contract_sha256": (
                _test_helper("test_action_ball_ac_family_contract.py")
                .contract.C10_CONTRACT_AUTHORITY_SHA256
            ),
        }
        authority = D.build_c10_family_payment_authority(**kwargs)
        authorities[family] = authority
        assert authority.family == family
        assert authority.placement_treatment_gain == expected_gain
        assert authority.projection_sha256 == projection.canonical_sha256
        assert authority.identity_sha256 == identity_sha256
        with pytest.raises(AttributeError):
            authority.placement_treatment_gain = 1.0 - expected_gain
        for forbidden in ("family", "gain", "reverse", "placement_treatment_gain"):
            with pytest.raises(TypeError):
                D.build_c10_family_payment_authority(
                    **kwargs, **{forbidden: expected_gain}
                )
        for pin_name in (
            "expected_projection_sha256",
            "expected_identity_sha256",
            "expected_c10_contract_sha256",
        ):
            with pytest.raises(D.LandingOutcomeDeviceError, match="pin"):
                D.build_c10_family_payment_authority(
                    **{**kwargs, pin_name: "0" * 64}
                )

    assert authorities["A"].canonical_sha256 != authorities["C"].canonical_sha256


def test_deep_empty_checkpoint_roundtrip_preserves_every_owned_tensor():
    owner = _coordinator(rows=2, flight_slots=2, mailbox_slots=3)
    receipt = owner.drain_ppo_boundary(update_index=0)
    checkpoint = owner.state_dict(receipt)
    assert len(checkpoint["tensor_bytes_sha256"]) == 64
    assert len(checkpoint["checkpoint_content_sha256"]) == 64
    restored = _coordinator(rows=2, flight_slots=2, mailbox_slots=3)
    restored.load_state_dict(
        checkpoint,
        expected_checkpoint_content_sha256=checkpoint[
            "checkpoint_content_sha256"
        ],
    )
    restored_receipt = restored.drain_ppo_boundary(update_index=1)
    restored_checkpoint = restored.state_dict(restored_receipt)
    _assert_same_checkpoint_state(checkpoint, restored_checkpoint)


def test_fault_telemetry_counts_one_incident_not_each_sticky_copy():
    owner = _coordinator()
    full_key, _, _ = _install(owner)
    _post_one(
        owner,
        full_key=full_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.1, 0.0, 1.1),
        current=(2.2, 0.0, 0.9),
        contact=True,
        producer_fault=True,
    )
    bit = D.FAULT_PRODUCER_CONTRACT
    sticky_copies = sum(
        int((torch.bitwise_and(tensor, bit) != 0).sum().item())
        for tensor in (
            owner._post_fault_bits,
            owner._flight_fault_bits,
            owner._mailbox_fault_bits,
        )
    )
    assert sticky_copies >= 2
    fault_index = tuple(name for name, _ in D.FAULTS).index("producer_contract")
    first = owner.drain_ppo_boundary(update_index=0)
    second = owner.drain_ppo_boundary(update_index=1)
    assert first.fault_counts[fault_index] == 1
    assert second.fault_counts[fault_index] == 1


def test_boundary_receipt_rejects_faults_and_staleness():
    stale_owner = _coordinator()
    full_key, _, _ = _install(stale_owner)
    receipt = stale_owner.drain_ppo_boundary(update_index=0)
    assert receipt.checkpoint_safe is True
    _settle_on_table(stale_owner, full_key=full_key)
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale|latest"):
        stale_owner.state_dict(receipt)

    faulted = _coordinator()
    fault_key, _, _ = _install(faulted)
    _post_one(
        faulted,
        full_key=fault_key,
        generation=0,
        flight_slot=0,
        ordinal=0,
        step=1,
        previous=(2.1, 0.0, 1.1),
        current=(2.2, 0.0, 0.9),
        contact=True,
        producer_fault=True,
    )
    fault_receipt = faulted.drain_ppo_boundary(update_index=0)
    assert fault_receipt.checkpoint_safe is False
    with pytest.raises(D.LandingOutcomeDeviceError, match="invariant|safe"):
        faulted.state_dict(fault_receipt)


@pytest.mark.parametrize(
    "corruption",
    ("negative_cadence", "two_mailboxes_one_flight", "orphan_reservation"),
)
def test_resealed_checkpoint_with_new_external_root_still_fails_owner_invariants(
    corruption: str,
):
    owner = _coordinator(mailbox_slots=2)
    _install(owner)
    receipt = owner.drain_ppo_boundary(update_index=0)
    checkpoint = owner.state_dict(receipt)
    forged = deepcopy(checkpoint)
    tensors = forged["tensors"]
    if corruption == "negative_cadence":
        tensors["flight_contact_deadline_control_step"][0, 0] = -1
    elif corruption == "two_mailboxes_one_flight":
        tensors["mailbox_reserved"][0, 1] = True
        tensors["mailbox_reservation_token"][0, 1].copy_(
            tensors["mailbox_reservation_token"][0, 0]
        )
        tensors["mailbox_reservation_generation"][0, 1] = tensors[
            "mailbox_reservation_generation"
        ][0, 0]
        tensors["mailbox_reserved_flight_slot"][0, 1] = tensors[
            "mailbox_reserved_flight_slot"
        ][0, 0]
    else:
        tensors["mailbox_reserved"][0, 1] = True
        tensors["mailbox_reservation_token"][0, 1].fill_(173)
        tensors["mailbox_reservation_generation"][0, 1] = 99
        tensors["mailbox_reserved_flight_slot"][0, 1] = 0
    forged_root = _reseal_checkpoint(forged)
    restored = _coordinator(mailbox_slots=2)
    with pytest.raises(D.LandingOutcomeDeviceError, match="invariant"):
        restored.load_state_dict(
            forged,
            expected_checkpoint_content_sha256=forged_root,
        )


@pytest.mark.parametrize("corruption", ("all_negative", "lifecycle_partial_order"))
def test_resealed_empty_checkpoint_rejects_counter_values_that_preserve_old_differences(
    corruption: str,
):
    owner = _coordinator()
    receipt = owner.drain_ppo_boundary(update_index=0)
    checkpoint = owner.state_dict(receipt)
    forged = deepcopy(checkpoint)
    tensors = forged["tensors"]
    if corruption == "all_negative":
        totals = (-1, -1, -1, -1, -1, -1)
    else:
        # Every old difference still equals the empty active counts, but
        # retired=2 > settled=1 violates the lifecycle partial order.
        totals = (2, 1, 2, 1, 1, 1)
    installed, settled, retired, payment0, payment1, closed = totals
    tensors["installed_total"].fill_(installed)
    tensors["settled_total"].fill_(settled)
    tensors["retired_total"].fill_(retired)
    tensors["payment_totals"][0] = payment0
    tensors["payment_totals"][1] = payment1
    tensors["closed_total"].fill_(closed)
    forged["receipt"] = replace(
        forged["receipt"],
        installed_total=installed,
        settled_total=settled,
        retired_total=retired,
        common_payment_total=payment0,
        placement_payment_total=payment1,
        closed_total=closed,
    )
    forged_root = _reseal_checkpoint(forged)
    restored = _coordinator()
    with pytest.raises(D.LandingOutcomeDeviceError, match="invariant"):
        restored.load_state_dict(
            forged,
            expected_checkpoint_content_sha256=forged_root,
        )


def test_resealed_checkpoint_rejects_two_active_mailboxes_mapped_to_one_retained_flight():
    owner = _coordinator(mailbox_slots=2)
    full_key, _, _ = _install(owner)
    _settle_on_table(owner, full_key=full_key)
    receipt = owner.drain_ppo_boundary(update_index=0)
    checkpoint = owner.state_dict(receipt)
    forged = deepcopy(checkpoint)
    tensors = forged["tensors"]
    for name, tensor in tensors.items():
        if (
            name.startswith("mailbox_")
            and tensor.ndim >= 2
            and tuple(tensor.shape[:2]) == (1, 2)
        ):
            tensor[0, 1].copy_(tensor[0, 0])
    # Keep every legacy difference and the lifecycle partial order valid:
    # installed-retired=1 live flight; settled-closed=2 live mailboxes.
    tensors["installed_total"].fill_(2)
    tensors["settled_total"].fill_(2)
    tensors["retired_total"].fill_(1)
    forged["receipt"] = replace(
        forged["receipt"],
        mailbox_state_counts=(0, 2, 0, 0),
        installed_total=2,
        settled_total=2,
        retired_total=1,
    )
    forged_root = _reseal_checkpoint(forged)
    restored = _coordinator(mailbox_slots=2)
    with pytest.raises(D.LandingOutcomeDeviceError, match="invariant"):
        restored.load_state_dict(
            forged,
            expected_checkpoint_content_sha256=forged_root,
        )


def test_process_local_hmac_is_only_a_consistency_seal_and_remains_hold():
    hold = (
        "module-private tokens and source-derived HMAC keys are same-process "
        "consistency seals, not security capabilities"
    )
    assert hold in D.HOLD_REASONS
    assert D.RUNTIME_INTEGRATED is False
    assert D.LAUNCH_AUTHORIZED is False

    authority = _payment_authority("A")
    forged_authority = replace(authority, _auth_tag=b"\x00" * 32)
    with pytest.raises(D.LandingOutcomeDeviceError, match="authentication"):
        _ = forged_authority.family

def test_device_closure_materializes_only_at_declared_cold_and_boundary_sites():
    d2h = '.to(device="cpu").tolist()'
    forbidden = (".cpu(", ".item(", ".tolist(", "torch.cuda.synchronize")
    module_functions = {
        name: value
        for name, value in vars(D).items()
        if inspect.isfunction(value) and value.__module__ == D.__name__
    }
    for name, function in module_functions.items():
        source = inspect.getsource(function)
        expected_d2h = 1 if name == "_tensor_bytes_sha256" else 0
        assert source.count(d2h) == expected_d2h, name
        without_allowed = source.replace(d2h, "")
        assert not any(token in without_allowed for token in forbidden), name

    methods = inspect.getmembers(
        D.ActionBallLandingOutcomeDeviceCoordinator,
        predicate=inspect.isfunction,
    )
    for name, method in methods:
        source = inspect.getsource(method)
        expected_d2h = 1 if name == "drain_ppo_boundary" else 0
        assert source.count(d2h) == expected_d2h, name
        without_allowed = source.replace(d2h, "")
        assert not any(token in without_allowed for token in forbidden), name

    assert "bool(" not in inspect.getsource(
        D._device_values_from_install_receipt
    )




def test_physical_retire_commit_only_gates_committed_park_then_copies_image():
    source = inspect.getsource(
        D.ActionBallLandingOutcomeDeviceCoordinator
        .commit_prevalidated_physical_retire
    )
    assert source.count("require_committed_prepared_token(") == 1
    assert ".copy_(" in source
    for forbidden in (
        "_prearm_physical_cleanup_union",
        "prepared_physical_retire_result",
        "armed_physical_retire_result",
        "_tensor(",
        "torch.",
        ".clone(",
        ".detach(",
        "capacity_authority",
    ):
        assert forbidden not in source, forbidden


def test_scale_target_is_structural_only_and_makes_no_performance_claim():
    owner = _coordinator(rows=D.SCALE_TARGET_NUM_ENVS, flight_slots=2, mailbox_slots=3)
    assert owner.num_envs == 4096
    assert owner.flight_slot_capacity == 2
    assert owner.mailbox_capacity == 3
    assert D.CUDA_PROFILED is False
