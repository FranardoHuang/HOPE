"""Dependency-light Reward callables for the fresh ActionBall full MDP.

This module freezes the RewardManager-facing callable and ordering contract.  It
does not install a manager configuration and it does not claim that the current
runtime owner implements the required top-level publication/close API.

One pre-reward publication opens a cycle.  Exactly fourteen real consumers must
then view their causal owner fact, compute the raw value, and record payment --
including a raw zero -- before the top owner may close the cycle.  A normal
miss, fall, or low reward is therefore data, not an infrastructure failure.
Skipping a callable, calling one twice, using a caller boolean in place of the
Physical selected-contact ledger, or throwing midway leaves the cycle open and
cannot authorize close/reset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import NoReturn
import weakref

import torch


SCHEMA_VERSION = 1
RUNTIME_INTEGRATED = False
MANAGER_CONFIG_INTEGRATED = False
CUDA_PROFILED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True

R03_CONSUMERS = (
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
PHYSICAL_SELECTED_CONTACT_CONSUMER = "physical_selected_contact"
R06_CONSUMERS = (
    "common_on_table_outcome",
    "post_contact_placement_guidance",
)
R07_CONSUMER = "common_recovery_reward_v1"
ORDERED_CONSUMERS = (
    *(f"r03:{name}" for name in R03_CONSUMERS),
    f"physical:{PHYSICAL_SELECTED_CONTACT_CONSUMER}",
    *(f"r06:{name}" for name in R06_CONSUMERS),
    f"r07:{R07_CONSUMER}",
)
EXPECTED_PAYMENT_COUNT = 14
if len(ORDERED_CONSUMERS) != EXPECTED_PAYMENT_COUNT:
    raise RuntimeError("fresh full-MDP Reward ABI is not fourteen consumers")

TOP_PUBLISH_METHOD = "publish_full_mdp_pre_reward"
TOP_REQUIRE_PUBLISH_METHOD = "require_owned_full_mdp_pre_reward"
TOP_CLOSE_METHOD = "close_full_mdp_reward_cycle"


class FreshFullMdpRewardError(RuntimeError):
    """Base error for the fresh fourteen-consumer Reward contract."""


class FreshFullMdpRewardConstructionHold(FreshFullMdpRewardError):
    """The production graph is missing an exact owner/consumer seam."""


class FreshFullMdpRewardCycleError(FreshFullMdpRewardError):
    """One Reward cycle was skipped, duplicated, replayed, or left open."""


@dataclass(frozen=True)
class _RewardOwners:
    r03: object
    physical: object
    r06: object
    r07: object
    num_envs: int
    device: torch.device


@dataclass(frozen=True)
class _CyclePayload:
    graph_identity: object
    sequence: int
    control_step: int
    top_publication: object
    physical_reward_cycle: object
    r06_reward_cycle: object


class FreshFullMdpRewardCycle:
    """Opaque identity for one pre-reward -> fourteen payments -> close cycle."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("fresh full-MDP Reward cycles are graph-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("fresh full-MDP Reward cycles are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("fresh full-MDP Reward cycles cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("fresh full-MDP Reward cycles cannot be copied")


@dataclass(frozen=True)
class _PaymentRecord:
    graph_identity: object
    cycle: FreshFullMdpRewardCycle
    consumer: str
    ordinal: int
    owner_payment_result: object


_REGISTRY_LOCK = threading.RLock()
_CYCLE_REGISTRY: weakref.WeakKeyDictionary[
    FreshFullMdpRewardCycle, _CyclePayload
] = weakref.WeakKeyDictionary()


def _mint_cycle(payload: _CyclePayload) -> FreshFullMdpRewardCycle:
    value = object.__new__(FreshFullMdpRewardCycle)
    with _REGISTRY_LOCK:
        _CYCLE_REGISTRY[value] = payload
    return value


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FreshFullMdpRewardCycleError(
            f"{label} must be an exact nonnegative int"
        )
    return value


def _exact_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FreshFullMdpRewardConstructionHold(
            f"{label} must be an exact positive int"
        )
    return value


def _method(owner: object, name: str, *, label: str):
    value = getattr(owner, name, None)
    if not callable(value):
        raise FreshFullMdpRewardConstructionHold(
            f"{label} lacks required exact method {name}"
        )
    return value


def _validate_owners(
    binding: object,
) -> _RewardOwners:
    required = ("r03", "physical", "r06", "r07", "num_envs", "device")
    if any(not hasattr(binding, name) for name in required):
        raise FreshFullMdpRewardConstructionHold(
            "Reward owner binding lacks the exact four-owner batch identity"
        )
    num_envs = _exact_positive_int(binding.num_envs, label="num_envs")
    device = torch.device(binding.device)
    owners = _RewardOwners(
        r03=binding.r03,
        physical=binding.physical,
        r06=binding.r06,
        r07=binding.r07,
        num_envs=num_envs,
        device=device,
    )
    if tuple(getattr(owners.r03, "consumers", ())) != R03_CONSUMERS:
        raise FreshFullMdpRewardConstructionHold(
            "R03 owner does not expose the exact ordered-ten ABI"
        )
    if tuple(getattr(owners.r06, "consumers", ())) != R06_CONSUMERS:
        raise FreshFullMdpRewardConstructionHold(
            "R06 owner does not expose the exact two-consumer ABI"
        )
    for method_name in ("view", "record_payment"):
        _method(owners.r03, method_name, label="R03 owner")
    # There is deliberately no fallback to caller ``contact``/``valid``
    # booleans or cumulative counters.  These two owner-issued methods are the
    # only selected-contact source admitted by construction.
    _method(
        owners.physical,
        "selected_contact_reward_view",
        label="Physical owner",
    )
    _method(
        owners.physical,
        "record_selected_contact_reward_payment",
        label="Physical owner",
    )
    for method_name in ("view", "record_payment"):
        _method(owners.r06, method_name, label="R06 owner")
    for method_name in ("reward_view", "record_reward_payment"):
        _method(owners.r07, method_name, label="R07 owner")
    for label, owner in (
        ("R03", owners.r03),
        ("Physical", owners.physical),
        ("R06", owners.r06),
        ("R07", owners.r07),
    ):
        if not hasattr(owner, "device"):
            raise FreshFullMdpRewardConstructionHold(
                f"{label} does not publish its exact device"
            )
        if getattr(owner, "num_envs", None) != num_envs:
            raise FreshFullMdpRewardConstructionHold(
                f"{label} num_envs differs from the top binding"
            )
        if torch.device(owner.device) != device:
            raise FreshFullMdpRewardConstructionHold(
                f"{label} device differs from the top binding"
            )
    return owners


class FreshFullMdpRewardGraph:
    """Four-owner fourteen-consumer cycle coordinator.

    This retained template graph is explicitly diagnostic and permanently
    unauthorized.  The production runtime uses the direct ActionEpoch graph.
    """

    def __init__(
        self,
        *,
        runtime_owner: object,
        runtime_lease: object,
        owners: _RewardOwners,
        diagnostic_unauthorized: bool,
    ) -> None:
        diagnostic = bool(diagnostic_unauthorized)
        if not diagnostic:
            raise FreshFullMdpRewardConstructionHold(
                "template Reward graph is diagnostic-only"
            )
        self._runtime_owner = runtime_owner
        self._runtime_lease = runtime_lease
        self._owners = owners
        self._diagnostic_unauthorized = diagnostic
        self._identity = object()
        self._sequence = 0
        self._active_cycle: FreshFullMdpRewardCycle | None = None
        # These are private order records containing the actual leaf-minted
        # verdict.  The graph deliberately mints no payment receipt of its own.
        self._payments: dict[str, _PaymentRecord] = {}
        self._construction_failed = False

    @property
    def launch_authorized(self) -> bool:
        return False

    @property
    def diagnostic_unauthorized(self) -> bool:
        return self._diagnostic_unauthorized

    @property
    def num_envs(self) -> int:
        return self._owners.num_envs

    @property
    def device(self) -> torch.device:
        return self._owners.device

    @property
    def active_cycle(self) -> FreshFullMdpRewardCycle | None:
        return self._active_cycle

    @property
    def construction_failed(self) -> bool:
        """Whether a post-bind construction failure made this graph unusable."""

        return self._construction_failed

    def _fail_construction(self) -> None:
        """Cold-discard marker; no rollback or rebinding is ever authorized."""

        self._construction_failed = True

    def _active_payload(self) -> _CyclePayload:
        if self._construction_failed:
            raise FreshFullMdpRewardConstructionHold(
                "fresh Reward graph failed after construction bind; cold discard required"
            )
        cycle = self._active_cycle
        if cycle is None:
            raise FreshFullMdpRewardCycleError(
                "Reward callable ran before the pre-reward DoneTerm publisher"
            )
        with _REGISTRY_LOCK:
            payload = _CYCLE_REGISTRY.get(cycle)
        if payload is None or payload.graph_identity is not self._identity:
            raise FreshFullMdpRewardCycleError(
                "active Reward cycle is foreign or no longer owned"
            )
        return payload

    def _require_unpaid(self, consumer: str) -> _CyclePayload:
        payload = self._active_payload()
        if consumer not in ORDERED_CONSUMERS:
            raise FreshFullMdpRewardCycleError(
                f"unknown fresh Reward consumer {consumer!r}"
            )
        if consumer in self._payments:
            raise FreshFullMdpRewardCycleError(
                f"fresh Reward consumer {consumer!r} ran twice"
            )
        ordinal = len(self._payments)
        expected = (
            ORDERED_CONSUMERS[ordinal]
            if ordinal < EXPECTED_PAYMENT_COUNT
            else None
        )
        if consumer != expected:
            raise FreshFullMdpRewardCycleError(
                "fresh Reward consumer order differs before leaf mutation: "
                f"expected {expected!r}, got {consumer!r}"
            )
        return payload

    def _record_payment(
        self, consumer: str, *, owner_payment_result: object = None
    ) -> None:
        self._require_unpaid(consumer)
        if owner_payment_result is None:
            raise FreshFullMdpRewardCycleError(
                "fresh Reward owner did not mint a payment verdict"
            )
        cycle = self._active_cycle
        assert cycle is not None
        self._payments[consumer] = _PaymentRecord(
            graph_identity=self._identity,
            cycle=cycle,
            consumer=consumer,
            ordinal=len(self._payments),
            owner_payment_result=owner_payment_result,
        )

    def begin_pre_reward(self, *, control_step: int) -> torch.Tensor:
        """Ask the bound top owner to publish R03+R07 exactly once."""

        step = _exact_nonnegative_int(control_step, label="control_step")
        if self._active_cycle is not None:
            raise FreshFullMdpRewardCycleError(
                "previous Reward cycle is still open"
            )
        publish = _method(
            self._runtime_owner,
            TOP_PUBLISH_METHOD,
            label="top runtime owner",
        )
        require = _method(
            self._runtime_owner,
            TOP_REQUIRE_PUBLISH_METHOD,
            label="top runtime owner",
        )
        publication = publish(
            runtime_lease=self._runtime_lease,
            control_step=step,
        )
        owned = require(
            publication,
            runtime_lease=self._runtime_lease,
            control_step=step,
        )
        terminated = getattr(owned, "terminated", None)
        time_out = getattr(owned, "time_out", None)
        r03_publication = getattr(owned, "r03_publication", None)
        r07_publication = getattr(owned, "r07_publication", None)
        physical_reward_cycle = getattr(owned, "physical_reward_cycle", None)
        r06_reward_cycle = getattr(owned, "r06_reward_cycle", None)
        if r03_publication is None or r07_publication is None:
            raise FreshFullMdpRewardCycleError(
                "top pre-reward publication did not bind both R03 and R07"
            )
        for name, value in (("terminated", terminated), ("time_out", time_out)):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (self.num_envs,)
                or value.dtype != torch.bool
                or value.device != self.device
            ):
                raise FreshFullMdpRewardCycleError(
                    f"top pre-reward {name} must be device bool [num_envs]"
                )
        # Fresh DoneTerm is installed with ``time_out=False`` by the manager
        # contract.  Do not launch a reduction or D2H here to re-check a tensor
        # just authenticated by its causal top owner: that would add hot-path
        # synchronization while proving only same-writer self-consistency.
        self._sequence += 1
        cycle = _mint_cycle(
            _CyclePayload(
                graph_identity=self._identity,
                sequence=self._sequence,
                control_step=step,
                top_publication=publication,
                physical_reward_cycle=physical_reward_cycle,
                r06_reward_cycle=r06_reward_cycle,
            )
        )
        self._active_cycle = cycle
        self._payments = {}
        return terminated

    def close_after_reward(self) -> object:
        """Close through the top owner only after all fourteen receipts exist."""

        payload = self._active_payload()
        missing = tuple(
            consumer for consumer in ORDERED_CONSUMERS if consumer not in self._payments
        )
        if missing:
            raise FreshFullMdpRewardCycleError(
                "fresh after-reward close is missing consumers: "
                + ", ".join(missing)
            )
        ordered_records = tuple(self._payments[name] for name in ORDERED_CONSUMERS)
        ordered_owner_results = []
        for name, record in zip(ORDERED_CONSUMERS, ordered_records):
            if (
                type(record) is not _PaymentRecord
                or record.graph_identity is not self._identity
                or record.cycle is not self._active_cycle
                or record.consumer != name
                or record.ordinal != len(ordered_owner_results)
                or record.owner_payment_result is None
            ):
                raise FreshFullMdpRewardCycleError(
                    "fresh Reward owner verdict/order record is absent, foreign, or reordered"
                )
            ordered_owner_results.append(record.owner_payment_result)
        ordered_owner_verdicts = tuple(ordered_owner_results)
        close = _method(
            self._runtime_owner,
            TOP_CLOSE_METHOD,
            label="top runtime owner",
        )
        # If close raises, the cycle deliberately remains open.  No wrapper
        # exception may convert partial payment into close/reset authority.
        result = close(
            runtime_lease=self._runtime_lease,
            pre_reward_publication=payload.top_publication,
            ordered_owner_payment_results=ordered_owner_verdicts,
            ordered_consumers=ORDERED_CONSUMERS,
        )
        self._active_cycle = None
        self._payments = {}
        return result


def construct_diagnostic_fresh_full_mdp_reward_graph(
    *,
    runtime_owner: object,
    runtime_lease: object,
    owner_binding: object,
) -> FreshFullMdpRewardGraph:
    """Build an explicitly unauthorized dependency-light test graph."""

    return FreshFullMdpRewardGraph(
        runtime_owner=runtime_owner,
        runtime_lease=runtime_lease,
        owners=_validate_owners(owner_binding),
        diagnostic_unauthorized=True,
    )


def _graph(env: object, graph_attr: str) -> FreshFullMdpRewardGraph:
    value = getattr(env, graph_attr, None)
    if type(value) is not FreshFullMdpRewardGraph:
        raise FreshFullMdpRewardConstructionHold(
            f"env.{graph_attr} is not the exact fresh Reward graph"
        )
    return value


def _positive_scale(std: float) -> float:
    if isinstance(std, bool) or type(std) not in (int, float):
        raise ValueError("fresh full-MDP Reward std must be a host number")
    scale = float(std)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("fresh full-MDP Reward std must be finite and positive")
    return scale


def _r03_tracking(
    env: object,
    *,
    consumer: str,
    component: str,
    std: float,
    graph_attr: str,
) -> torch.Tensor:
    graph = _graph(env, graph_attr)
    receipt_name = f"r03:{consumer}"
    graph._require_unpaid(receipt_name)
    scale = _positive_scale(std)
    view = graph._owners.r03.view(consumer)
    eligible = view.eligible & view.validity
    if component == "position":
        error = torch.linalg.vector_norm(
            view.achieved_position - view.target_position, dim=-1
        )
    elif component == "velocity":
        error = torch.linalg.vector_norm(
            view.achieved_velocity - view.target_velocity, dim=-1
        )
    elif component == "normal":
        cosine = torch.sum(
            view.achieved_face_normal * view.target_face_normal, dim=-1
        ).clamp(-1.0, 1.0)
        error = torch.acos(cosine)
    else:
        raise ValueError(f"unknown R03 component {component!r}")
    if consumer.endswith("_coarse"):
        raw = torch.reciprocal(1.0 + torch.square(error / scale))
    else:
        raw = torch.exp(-torch.square(error / scale))
    payment = torch.where(eligible, raw, torch.zeros_like(raw))
    result = graph._owners.r03.record_payment(consumer, payment)
    graph._record_payment(receipt_name, owner_payment_result=result)
    return payment


def r03_racket_position(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_position", component="position", std=std, graph_attr=graph_attr
    )


def r03_racket_velocity(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_velocity", component="velocity", std=std, graph_attr=graph_attr
    )


def r03_racket_normal(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_normal", component="normal", std=std, graph_attr=graph_attr
    )


def r03_racket_position_coarse(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_position_coarse", component="position", std=std, graph_attr=graph_attr
    )


def r03_racket_velocity_coarse(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_velocity_coarse", component="velocity", std=std, graph_attr=graph_attr
    )


def r03_racket_normal_coarse(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_normal_coarse", component="normal", std=std, graph_attr=graph_attr
    )


def r03_racket_position_precision(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_position_precision", component="position", std=std, graph_attr=graph_attr
    )


def r03_racket_velocity_precision(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_velocity_precision", component="velocity", std=std, graph_attr=graph_attr
    )


def r03_racket_normal_precision(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r03_tracking(
        env, consumer="racket_normal_precision", component="normal", std=std, graph_attr=graph_attr
    )


def r03_paddle_center_proximity(
    env: object, std: float, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    graph = _graph(env, graph_attr)
    receipt_name = "r03:paddle_center_proximity"
    graph._require_unpaid(receipt_name)
    scale = _positive_scale(std)
    owner = graph._owners.r03
    view = owner.view("paddle_center_proximity")
    distance = torch.linalg.vector_norm(
        view.achieved_position - view.ball_position, dim=-1
    )
    raw = torch.reciprocal(1.0 + torch.square(distance / scale))
    payment = torch.where(
        view.eligible & view.validity, raw, torch.zeros_like(raw)
    )
    result = owner.record_payment("paddle_center_proximity", payment)
    graph._record_payment(receipt_name, owner_payment_result=result)
    return payment


def physical_selected_contact(
    env: object, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    """Pay unit raw reward only through the Physical exact contact ledger."""

    graph = _graph(env, graph_attr)
    receipt_name = f"physical:{PHYSICAL_SELECTED_CONTACT_CONSUMER}"
    graph._require_unpaid(receipt_name)
    owner = graph._owners.physical
    payload = graph._active_payload()
    view = owner.selected_contact_reward_view()
    eligible = getattr(view, "eligible", None)
    if (
        not isinstance(eligible, torch.Tensor)
        or eligible.shape != (graph.num_envs,)
        or eligible.dtype != torch.bool
        or eligible.device != graph.device
    ):
        raise FreshFullMdpRewardCycleError(
            "Physical selected-contact view is not exact device bool [num_envs]"
        )
    raw = eligible.to(dtype=torch.float32)
    result = owner.record_selected_contact_reward_payment(view, raw_reward=raw)
    if getattr(result, "rejected", None) is None:
        raise FreshFullMdpRewardCycleError(
            "Physical selected-contact payment lacks its owner verdict"
        )
    graph._record_payment(receipt_name, owner_payment_result=result)
    return raw


def _reward_epoch(graph: FreshFullMdpRewardGraph) -> torch.Tensor:
    payload = graph._active_payload()
    return torch.full(
        (graph.num_envs,),
        payload.control_step,
        dtype=torch.int64,
        device=graph.device,
    )


def _r06_payment(
    env: object, *, consumer: str, graph_attr: str
) -> torch.Tensor:
    graph = _graph(env, graph_attr)
    receipt_name = f"r06:{consumer}"
    graph._require_unpaid(receipt_name)
    owner = graph._owners.r06
    payload = graph._active_payload()
    epoch = _reward_epoch(graph)
    if payload.r06_reward_cycle is None:
        view = owner.view(consumer, epoch)
    else:
        view = owner.view(
            consumer,
            reward_cycle_token=payload.r06_reward_cycle,
        )
    if consumer == "common_on_table_outcome":
        raw = view.common_on_table_outcome.to(dtype=view.canonical_total.dtype)
    elif consumer == "post_contact_placement_guidance":
        raw = view.canonical_total * view.placement_treatment_gain
    else:
        raise ValueError(f"unknown R06 consumer {consumer!r}")
    raw = torch.where(view.policy_eligible, raw, torch.zeros_like(raw))
    payment_kwargs = dict(
        mask=view.eligible,
        full_key_sha256=view.full_key_sha256,
        ball_generation=view.ball_generation,
        raw_reward=raw,
    )
    if payload.r06_reward_cycle is None:
        payment_kwargs["reward_epoch"] = epoch
    else:
        payment_kwargs["reward_cycle_token"] = payload.r06_reward_cycle
    result = owner.record_payment(consumer, **payment_kwargs)
    graph._record_payment(receipt_name, owner_payment_result=result)
    # Manager output is per environment while the durable mailbox is [N,K].
    return raw.sum(dim=1)


def r06_common_on_table_outcome(
    env: object, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r06_payment(
        env, consumer="common_on_table_outcome", graph_attr=graph_attr
    )


def r06_post_contact_placement_guidance(
    env: object, graph_attr: str = "action_ball_full_mdp_reward_graph"
) -> torch.Tensor:
    return _r06_payment(
        env, consumer="post_contact_placement_guidance", graph_attr=graph_attr
    )


def r07_continuous_recovery(
    env: object,
    *,
    manager_weight: float,
    graph_attr: str = "action_ball_full_mdp_reward_graph",
) -> torch.Tensor:
    """Pay R07 every step; its owner already supplies the weighted value."""

    if (
        isinstance(manager_weight, bool)
        or type(manager_weight) not in (int, float)
        or float(manager_weight) != 1.0
    ):
        raise FreshFullMdpRewardConstructionHold(
            "R07 RewardManager weight must equal exactly 1"
        )
    graph = _graph(env, graph_attr)
    receipt_name = f"r07:{R07_CONSUMER}"
    graph._require_unpaid(receipt_name)
    owner = graph._owners.r07
    view = owner.reward_view(R07_CONSUMER)
    payment = view.weighted_reward
    if (
        not isinstance(payment, torch.Tensor)
        or payment.shape != (graph.num_envs,)
        or payment.device != graph.device
    ):
        raise FreshFullMdpRewardCycleError(
            "R07 weighted reward must be device-local [num_envs]"
        )
    result = owner.record_reward_payment(R07_CONSUMER, payment)
    graph._record_payment(receipt_name, owner_payment_result=result)
    return payment


__all__ = [
    "SCHEMA_VERSION",
    "RUNTIME_INTEGRATED",
    "MANAGER_CONFIG_INTEGRATED",
    "CUDA_PROFILED",
    "LAUNCH_AUTHORIZED",
    "DIAGNOSTIC_UNAUTHORIZED",
    "R03_CONSUMERS",
    "R06_CONSUMERS",
    "R07_CONSUMER",
    "ORDERED_CONSUMERS",
    "EXPECTED_PAYMENT_COUNT",
    "FreshFullMdpRewardError",
    "FreshFullMdpRewardConstructionHold",
    "FreshFullMdpRewardCycleError",
    "FreshFullMdpRewardCycle",
    "FreshFullMdpRewardGraph",
    "construct_diagnostic_fresh_full_mdp_reward_graph",
    "r03_racket_position",
    "r03_racket_velocity",
    "r03_racket_normal",
    "r03_racket_position_coarse",
    "r03_racket_velocity_coarse",
    "r03_racket_normal_coarse",
    "r03_racket_position_precision",
    "r03_racket_velocity_precision",
    "r03_racket_normal_precision",
    "r03_paddle_center_proximity",
    "physical_selected_contact",
    "r06_common_on_table_outcome",
    "r06_post_contact_placement_guidance",
    "r07_continuous_recovery",
]
