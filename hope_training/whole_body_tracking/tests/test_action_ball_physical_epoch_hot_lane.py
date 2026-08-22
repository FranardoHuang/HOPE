"""Focused real-D05 -> Physical ActionEpoch hot-lane counterexamples.

The formal checkpoint/R10 lane remains untouched.  No contact fact is
fabricated; the tests cover retain, owner-derived staggered due launch, exact
scene mutation, R06 install callpoint, persistent K re-arm, and source guards.
"""

from __future__ import annotations

import inspect
import importlib.util
from dataclasses import fields
from pathlib import Path
import sys
import types

import pytest
import torch

try:  # Local torch 2.0 has only the one-argument overload.
    torch._assert_async(torch.tensor(True), "probe")
except TypeError:  # pragma: no cover - exact Isaac runtime accepts the message
    _torch_assert_async = torch._assert_async

    def _assert_async_compat(condition, message=""):
        try:
            return _torch_assert_async(condition)
        except RuntimeError as exc:
            raise RuntimeError(message) from exc

    torch._assert_async = _assert_async_compat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import action_ball_full_mdp_reset_genesis as genesis  # noqa: E402


MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
_PACKAGE_PATHS = (
    ("whole_body_tracking", SOURCE / "whole_body_tracking"),
    ("whole_body_tracking.tasks", SOURCE / "whole_body_tracking" / "tasks"),
    (
        "whole_body_tracking.tasks.tracking",
        SOURCE / "whole_body_tracking" / "tasks" / "tracking",
    ),
    ("whole_body_tracking.tasks.tracking.mdp", MDP),
)
for package_name, package_path in _PACKAGE_PATHS:
    package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
    paths = list(getattr(package, "__path__", ()))
    if str(package_path) not in paths:
        package.__path__ = [*paths, str(package_path)]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = _load(name, path)
    parent_name, attribute = name.rsplit(".", 1)
    setattr(sys.modules[parent_name], attribute, module)
    return module


selected_reset = _canonical_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_selected_reset",
    MDP / "action_ball_full_mdp_selected_reset.py",
)
sys.modules["action_ball_full_mdp_selected_reset"] = selected_reset
epoch = _canonical_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch",
    MDP / "action_ball_full_mdp_epoch.py",
)
sys.modules["action_ball_full_mdp_epoch"] = epoch
physical = _canonical_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_physical_flight_device",
    MDP / "action_ball_physical_flight_device.py",
)
from test_action_ball_physical_flight_contract import _capacity  # noqa: E402
from test_action_ball_physical_flight_device import (  # noqa: E402
    _focused_postphysics_stamp,
    _install_focused_postphysics_stamp_module,
)
import test_action_ball_continuous_runtime_transaction_device as d05t  # noqa: E402


def _physical_owner(device: torch.device, *, num_envs: int = 2):
    capacity = _capacity(cadence=5, horizon=5)  # inclusive capacity K=2
    issue = genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=num_envs, device=device
    )
    scene = physical.TensorPhysicalFlightScenePort(
        num_envs=num_envs, flight_capacity=2, device=device
    )
    owner = physical.ActionBallPhysicalFlightDeviceOwner(
        num_envs=num_envs,
        capacity_receipt=capacity,
        expected_capacity_receipt_sha256=capacity.canonical_sha256,
        reset_genesis_authority=issue.authority,
        reset_genesis_receipt=issue.receipt,
        scene_body_names=tuple(
            f"action_ball_flight_ball_{slot:03d}" for slot in range(2)
        ),
        scene_port=scene,
    )
    owner._action_epoch_owner = types.SimpleNamespace(
        shot_slot_capacity=1,
        project_keyed_postphysics_activity_mask=(
            lambda *, owner: torch.zeros(
                (num_envs, 1), dtype=torch.bool, device=device
            )
        ),
        poison_owner_write=lambda *_args, **_kwargs: None,
    )
    return owner, scene




class _R06InstallProbe:
    """Consumes only Physical's no-argument exact active projection."""

    def __init__(self, owner):
        self.owner = owner
        self.due_rows = []
        self.views = []
        self.flight_state = torch.zeros(
            (owner.num_envs, owner.flight_capacity),
            dtype=torch.int8,
            device=owner.device,
        )

    def install_action_ball_full_mdp_epoch_launch_from_physical(self):
        view = self.owner.action_epoch_r06_launch_projection()
        view = self.owner.require_owned_action_epoch_r06_launch_projection(view)
        self.due_rows.append(view.due.clone())
        self.views.append(view)


def _shot_key(*, num_envs, device, offset=0, width=1):
    shape = (num_envs, width) if width > 0 else (num_envs,)
    row = torch.arange(num_envs, dtype=torch.int64, device=device)
    if width > 0:
        row = row[:, None].expand(shape)
    values = {
        name: (row + offset + index * 100).contiguous()
        for index, name in enumerate(
            physical._row_identity.ActionEpochShotKey.__dataclass_fields__
        )
    }
    values["action_slot"] = torch.zeros(shape, dtype=torch.int64, device=device)
    return physical._row_identity.ActionEpochShotKey(**values)


def _pending(owner, *, key, pending, ordinal, flight_slot=None):
    n = owner.num_envs
    device = owner.device
    if flight_slot is None:
        flight_slot = torch.zeros(n, dtype=torch.int64, device=device)
    return physical._ActionEpochPendingLaunchState(
        pending=pending,
        flight_slot=flight_slot,
        shot_key=key,
        publication_ordinal=ordinal,
        physical_state_f32=torch.arange(
            n * physical.STATE_WIDTH, dtype=torch.float32, device=device
        ).reshape(n, physical.STATE_WIDTH),
        target_xy_m=torch.zeros((n, 2), dtype=torch.float32, device=device),
        launch_control_step=torch.full(
            (n,), 7, dtype=torch.int64, device=device
        ),
        contact_deadline_control_step=torch.full(
            (n,), 8, dtype=torch.int64, device=device
        ),
        crossing_horizon_control_step=torch.full(
            (n,), 9, dtype=torch.int64, device=device
        ),
    )


class _EpochLaunchStub:
    def __init__(self, owner, key):
        self.owner = owner
        self.launch_views = []
        self.shot_slot_capacity = 1
        n = owner.num_envs
        self.record = types.SimpleNamespace(
            current_task_slot=torch.zeros(
                n, dtype=torch.int64, device=owner.device
            ),
            identity=types.SimpleNamespace(shot_key=key),
            publication_ordinal=torch.arange(
                n, dtype=torch.int64, device=owner.device
            )[:, None],
            phase=torch.full(
                (n, 1), epoch.PHASE_REVEAL_COMMITTED,
                dtype=torch.int64, device=owner.device,
            ),
            selected_mask=torch.ones(
                (n, 1), dtype=torch.bool, device=owner.device
            ),
        )

    def current(self):
        return self.record

    def project_keyed_postphysics_activity_mask(self, *, owner):
        assert owner is self.owner
        return torch.ones(
            (self.owner.num_envs, self.shot_slot_capacity),
            dtype=torch.bool,
            device=self.owner.device,
        )

    def refresh_physical_launch_rows(self):
        view = self.owner.action_epoch_r06_launch_projection()
        assert self.owner._r06_owner.views[-1] is view
        self.launch_views.append(view)
        return self.record

    def poison_owner_write(self, *_args, **_kwargs):
        return None


class _MotionLaunchStub:
    def __init__(self, owner, key):
        self.owner = owner
        self.control_tick = torch.full(
            (owner.num_envs,), 7, dtype=torch.int64, device=owner.device
        )
        self.set_key(key)

    def set_key(self, key):
        self.action_uid = key.action_uid
        self.task_identity = key.task_identity
        self.reset_generation = key.reset_generation
        self.swing_generation = key.ball_generation

    def action_ball_continuous_motion_observation_projection(self):
        return self

    def require_owned_action_ball_continuous_motion_observation(self, token):
        assert token is self
        return types.SimpleNamespace(
            motion_owner=self,
            control_tick=self.control_tick,
            action_uid=self.action_uid,
            task_identity=self.task_identity,
            reset_generation=self.reset_generation,
            swing_generation=self.swing_generation,
        )


def _bind_launch_stubs(owner, key):
    epoch_owner = _EpochLaunchStub(owner, key)
    motion = _MotionLaunchStub(owner, physical._row_identity.ActionEpochShotKey(
        **{
            field.name: getattr(key, field.name)[:, 0]
            for field in fields(physical._row_identity.ActionEpochShotKey)
        }
    ))
    r06 = _R06InstallProbe(owner)
    owner._action_epoch_owner = epoch_owner
    owner._action_epoch_motion_owner = motion
    owner._r06_owner = r06
    return epoch_owner, motion, r06


def _accepted_view(owner, *, token, accepted):
    n = owner.num_envs
    shape = (n, 1)
    full_key = _shot_key(num_envs=n, device=owner.device, width=1)
    mask = accepted[:, None]
    key = physical._row_identity.ActionEpochShotKey(
        **{
            field.name: torch.where(
                mask,
                getattr(full_key, field.name),
                torch.full_like(getattr(full_key, field.name), -1),
            ).contiguous()
            for field in fields(physical._row_identity.ActionEpochShotKey)
        }
    )
    identity = epoch.EpochIdentityPayload(
        shot_key=key,
        **{
            name: torch.where(
                mask,
                torch.arange(n, dtype=torch.int64)[:, None] + 10,
                torch.full(shape, -1, dtype=torch.int64),
            ).contiguous()
            for name in (
                "scheduled_ordinal",
                "target_generation",
                "selected_cell",
                "candidate_identity",
            )
        },
    )
    clock_value = lambda value: torch.where(  # noqa: E731
        mask,
        torch.full(shape, value, dtype=torch.int64),
        torch.full(shape, -1, dtype=torch.int64),
    ).contiguous()
    clocks = epoch.EpochClockPayload(
        reveal_tick=clock_value(4),
        contact_tick=clock_value(5),
        launch_tick=clock_value(6),
        deadline_tick=clock_value(7),
        next_reveal_tick=clock_value(8),
    )
    task_f32 = torch.zeros((n, 1, epoch.TASK_F32_WIDTH), dtype=torch.float32)
    task_f32[:, 0, -physical.STATE_WIDTH:] = torch.arange(
        n * physical.STATE_WIDTH, dtype=torch.float32
    ).reshape(n, physical.STATE_WIDTH)
    task_f32 = torch.where(mask[:, :, None], task_f32, torch.zeros_like(task_f32))
    return d05t.r05.DeviceR05AcceptedRowsView(
        transaction=token,
        publication_ordinal=torch.where(
            mask,
            torch.arange(n, dtype=torch.int64)[:, None] + 100,
            torch.full(shape, -1, dtype=torch.int64),
        ).contiguous(),
        target_xy_m=torch.where(
            mask[:, :, None],
            torch.arange(n * 2, dtype=torch.float32).reshape(n, 1, 2),
            torch.zeros((n, 1, 2), dtype=torch.float32),
        ).contiguous(),
        identity=identity,
        clocks=clocks,
        task=epoch.EpochTaskPayload(
            task_f32=task_f32.contiguous(), task_valid=mask.contiguous()
        ),
        rng_counter=torch.zeros(shape, dtype=torch.int64),
    )












def test_physical_epoch_hot_sources_have_no_host_or_current_epoch_rejoin():
    launch = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.launch_action_epoch
    )
    arm = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.
        _arm_current_action_epoch_physics_fact_source
    )
    publish = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.
        publish_action_epoch_post_physics
    )
    for source in (launch, arm):
        for forbidden in (".item(", ".cpu(", ".tolist(", ".numpy("):
            assert forbidden not in source
    refresh = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.
        refresh_action_epoch_host_activity
    )
    assert refresh.count(".item()") == 1
    for forbidden in (".cpu(", ".tolist(", ".numpy(", ".nonzero("):
        assert forbidden not in refresh
    assert "epoch_owner.current()" not in publish
    assert "self._action_epoch_flight_shot_key.clone()" not in publish
    assert "shot_key=self._action_epoch_flight_shot_key" in publish
    # Physical's one-shot packet is consumed synchronously by Epoch/R06.  A
    # second same-writer full-grid snapshot is neither an independent fact nor
    # a safety boundary.
    assert "observe_mask=observe.detach().clone()" not in publish
    assert "publication_ordinal" in publish
    assert "refresh_physical_postphysics_rows" in publish
    assert "publish_physical_launch_rows" not in launch
    assert "refresh_physical_launch_rows()" in launch


def test_r06_live_row_forces_dense_activity_when_physical_is_empty():
    owner, _scene = _physical_owner(torch.device("cpu"))
    r06_state = torch.zeros((2, 2), dtype=torch.int8)
    r06_state[1, 0] = physical.R06_FLIGHT_INBOUND
    owner._r06_owner = types.SimpleNamespace(flight_state=r06_state)
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    assert owner._action_epoch_host_activity_has_work is True


def test_host_activity_verdict_keeps_transport_and_keyed_facts_distinct():
    idle, _scene = _physical_owner(torch.device("cpu"))
    assert idle._fixed_flight_slot_grid.is_contiguous()
    idle._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    idle.refresh_action_epoch_host_activity(next_control_step=1)
    idle_verdict = idle.action_epoch_host_activity_verdict(control_step=1)
    assert type(idle_verdict) is physical.ActionEpochHostActivityVerdict
    assert idle_verdict.transport_work is False
    assert idle_verdict.keyed_epoch_work is False

    retired, _scene = _physical_owner(torch.device("cpu"))
    retired._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    retired._action_epoch_owner = types.SimpleNamespace(
        shot_slot_capacity=1,
        project_keyed_postphysics_activity_mask=(
            lambda *, owner: torch.ones((2, 1), dtype=torch.bool)
        ),
    )
    retired.refresh_action_epoch_host_activity(next_control_step=1)
    retired_verdict = retired.action_epoch_host_activity_verdict(control_step=1)
    assert retired_verdict.transport_work is False
    assert retired_verdict.keyed_epoch_work is True


def test_keyed_activity_projection_failure_invalidates_cache_fail_dense():
    owner, _scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    assert owner.action_epoch_host_activity_verdict(
        control_step=1
    ).transport_work is False
    owner._last_postphysics_exact_stamp = (1, 0, 1, 1, 1)

    def fail_projection(*, owner):
        raise RuntimeError("keyed activity failure")

    owner._action_epoch_owner.project_keyed_postphysics_activity_mask = (
        fail_projection
    )
    with pytest.raises(RuntimeError, match="keyed activity failure"):
        owner.refresh_action_epoch_host_activity(next_control_step=2)
    assert owner._action_epoch_host_activity_control_step is None
    assert owner._action_epoch_host_activity_has_work is True
    assert owner._action_epoch_host_activity_has_keyed_work is True
    with pytest.raises(
        physical.PhysicalEpochIntegrationHold, match="absent, stale, or replayed"
    ):
        owner.action_epoch_host_activity_verdict(control_step=2)


def test_activity_refresh_rejects_a_second_control_boundary_reduction():
    owner, _scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    with pytest.raises(
        physical.PhysicalEpochIntegrationHold,
        match="stale, skipped, or replayed",
    ):
        owner.refresh_action_epoch_host_activity(next_control_step=1)


def test_idle_pair_rejects_missing_or_duplicate_pre_and_post(monkeypatch):
    _install_focused_postphysics_stamp_module(monkeypatch)
    missing_pre, _scene = _physical_owner(torch.device("cpu"))
    with pytest.raises(
        physical.PhysicalEpochIntegrationHold,
        match="no exact pre-physics pair",
    ):
        missing_pre.publish_action_epoch_post_physics(
            _focused_postphysics_stamp()
        )
    assert missing_pre._poisoned

    owner, _scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    owner.launch_action_epoch()
    with pytest.raises(
        physical.PhysicalEpochIntegrationHold,
        match="unconsumed post pair",
    ):
        owner.launch_action_epoch()
    owner.publish_action_epoch_post_physics(_focused_postphysics_stamp())
    with pytest.raises(
        physical.PhysicalEpochIntegrationHold,
        match="no exact pre-physics pair",
    ):
        owner.publish_action_epoch_post_physics(
            _focused_postphysics_stamp(control=2, sim_step=2)
        )
    assert owner._poisoned


def test_false_then_d05_accept_refreshes_next_control_dense(monkeypatch):
    _install_focused_postphysics_stamp_module(monkeypatch)
    owner, _scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    owner.launch_action_epoch()
    owner.publish_action_epoch_post_physics(_focused_postphysics_stamp())

    token = object()
    view = _accepted_view(
        owner, token=token, accepted=torch.tensor((True, False))
    )

    class D05:
        def require_owned_action_epoch_accepted(self, actual, *, owner_kind):
            assert actual is token and owner_kind == "physical_ball"
            return view

    owner._action_epoch_device_r05_owner = D05()
    owner.retain_action_epoch_launch(token)
    owner.refresh_action_epoch_host_activity(next_control_step=2)
    assert owner._action_epoch_host_activity_has_work is True


def test_activity_cache_transitions_idle_accept_retire_idle(monkeypatch):
    _install_focused_postphysics_stamp_module(monkeypatch)
    owner, _scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    assert owner._action_epoch_host_activity_has_work is False
    owner.launch_action_epoch()
    owner.publish_action_epoch_post_physics(_focused_postphysics_stamp())

    token = object()
    accepted = torch.tensor((True, False))
    view = _accepted_view(owner, token=token, accepted=accepted)

    class D05:
        def require_owned_action_epoch_accepted(self, actual, *, owner_kind):
            assert actual is token and owner_kind == "physical_ball"
            return view

    owner._action_epoch_device_r05_owner = D05()
    owner.retain_action_epoch_launch(token)
    owner.refresh_action_epoch_host_activity(next_control_step=2)
    assert owner._action_epoch_host_activity_has_work is True

    # Model the same state changes that the already-covered dense retire tail
    # publishes, then cross the next exact control boundary.
    owner._action_epoch_pending_launch.pending.zero_()
    owner._published.zero_()
    owner._parked.fill_(True)
    owner._lifecycle.zero_()
    owner._action_epoch_active_flight_slot.fill_(-1)
    for field in fields(physical._row_identity.ActionEpochShotKey):
        getattr(owner._action_epoch_flight_shot_key, field.name).fill_(-1)
    owner._action_epoch_flight_publication_ordinal.fill_(-1)
    owner._selected_contact_pending.zero_()
    owner._last_postphysics_exact_stamp = (2, 0, 1, 2, 1)
    owner.refresh_action_epoch_host_activity(next_control_step=3)
    assert owner._action_epoch_host_activity_has_work is False


def test_retire_midcontrol_stays_dense_until_next_boundary_refresh():
    owner, _scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    owner._published[0, 0] = True
    owner._parked[0, 0] = False
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    owner._published.zero_()
    owner._parked.fill_(True)
    owner._last_postphysics_exact_stamp = (1, 0, 2, 1, 1)
    assert owner._action_epoch_cached_idle_for_next_substep() is False
    owner._last_postphysics_exact_stamp = (1, 1, 2, 2, 1)
    owner.refresh_action_epoch_host_activity(next_control_step=2)
    assert owner._action_epoch_host_activity_has_work is False


def test_reset_restore_and_checkpoint_boundaries_invalidate_or_reject_idle_cache():
    invalidation = (
        "self._action_epoch_host_activity_control_step = None",
        "self._action_epoch_host_activity_has_work = True",
        "self._action_epoch_host_activity_has_keyed_work = True",
    )
    for method in (
        physical.ActionBallPhysicalFlightDeviceOwner.
        commit_prevalidated_selected_true_reset,
        physical.ActionBallPhysicalFlightDeviceOwner.true_reset_many,
    ):
        source = inspect.getsource(method)
        assert all(line in source for line in invalidation)
    for method in (
        physical.ActionBallPhysicalFlightDeviceOwner._require_checkpoint_idle,
        physical.ActionBallPhysicalFlightDeviceOwner._require_selected_reset_idle,
    ):
        assert "self._action_epoch_substep_pair is not None" in inspect.getsource(
            method
        )


def test_launch_source_orders_r06_then_epoch_pull_before_projection_clear():
    source = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.launch_action_epoch
    )
    ordered = (
        "r06_install()",
        "epoch_owner.refresh_physical_launch_rows()",
        "self._action_epoch_active_r06_launch = None",
    )
    positions = tuple(source.index(fragment) for fragment in ordered)
    assert positions == tuple(sorted(positions))


def test_postphysics_source_orders_one_capture_through_physical_and_r06_retire():
    source = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.
        publish_action_epoch_post_physics
    )
    ordered = (
        "facts = self.capture_post_physics_facts(stamp)",
        "ActionEpochR06PostPhysicsProjection(",
        "refresh_epoch()",
        "publish_direct()",
        "retire_direct()",
        'kind="retire"',
        "r06_publish()",
        "self._active_postphysics_capture = None",
    )
    assert source.count(ordered[0]) == 1
    positions = tuple(
        source.rindex(fragment)
        if fragment == "self._active_postphysics_capture = None"
        else source.index(fragment)
        for fragment in ordered
    )
    assert positions == tuple(sorted(positions))
    assert "retired[env_ids" not in source
    assert "clear_slot_mask = retired & active_slot_mask" in source
    assert "identity_grid.copy_(" in source


@pytest.mark.parametrize("pending_values", (None, (False, False)))
def test_zero_flight_fixed_tapes_use_paired_idle_without_dense_mutation(
    monkeypatch, pending_values
):
    _install_focused_postphysics_stamp_module(monkeypatch)
    owner, scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    if pending_values is not None:
        owner._action_epoch_pending_launch = _pending(
            owner,
            key=_shot_key(num_envs=2, device=owner.device, width=0),
            pending=torch.tensor(pending_values, dtype=torch.bool),
            ordinal=torch.tensor((4, 5), dtype=torch.int64),
        )
    monkeypatch.setattr(
        physical.ActionBallPhysicalFlightDeviceOwner,
        "capture_post_physics_facts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("idle pair reached dense capture")
        ),
    )
    before = owner._clone_action_epoch_direct_state()
    mutation_before = owner._mutation_version
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    assert owner._action_epoch_host_activity_has_work is False
    for substep in range(4):
        owner.launch_action_epoch()
        owner.publish_action_epoch_post_physics(
            _focused_postphysics_stamp(
                substep=substep,
                decimation=4,
                sim_step=substep + 1,
            )
        )
    assert owner._action_epoch_substep_pair is None
    assert owner._last_postphysics_exact_stamp == (1, 3, 4, 4, 1)
    assert owner._mutation_version == mutation_before
    assert scene.apply_count == 0
    assert not owner._action_epoch_direct_state_mismatch(before)


@pytest.mark.parametrize("pending_values", ((True, False), (True, True)))
def test_mixed_and_full_fixed_tapes_keep_dense_capture(
    monkeypatch, pending_values
):
    _install_focused_postphysics_stamp_module(monkeypatch)
    owner, _scene = _physical_owner(torch.device("cpu"))
    current_grid = _shot_key(num_envs=2, device=owner.device, width=1)
    _epoch_owner, _motion, _r06 = _bind_launch_stubs(owner, current_grid)
    key = physical._row_identity.ActionEpochShotKey(
        **{
            field.name: getattr(current_grid, field.name)[:, 0]
            for field in fields(physical._row_identity.ActionEpochShotKey)
        }
    )
    owner._action_epoch_pending_launch = _pending(
        owner,
        key=key,
        pending=torch.tensor(pending_values, dtype=torch.bool),
        ordinal=torch.tensor((4, 5), dtype=torch.int64),
    )
    owner.refresh_action_epoch_host_activity(next_control_step=1)
    assert owner._action_epoch_host_activity_has_work is True
    owner.launch_action_epoch()
    seen = []

    class DensePathReached(RuntimeError):
        pass

    def capture(_actual, stamp):
        seen.append(stamp.exact_tuple())
        raise DensePathReached

    monkeypatch.setattr(
        physical.ActionBallPhysicalFlightDeviceOwner,
        "capture_post_physics_facts",
        capture,
    )
    with pytest.raises(DensePathReached):
        owner.publish_action_epoch_post_physics(_focused_postphysics_stamp())
    assert seen == [(1, 0, 1, 1, 1)]


@pytest.mark.parametrize("num_envs", (2, 64))
def test_full_n_d05_accept_view_stages_only_accepted_rows(num_envs):
    owner, _scene = _physical_owner(torch.device("cpu"), num_envs=num_envs)
    token = object()
    accepted = torch.arange(num_envs).remainder(2).eq(0)
    view = _accepted_view(owner, token=token, accepted=accepted)

    class D05:
        def require_owned_action_epoch_accepted(self, actual, *, owner_kind):
            assert actual is token and owner_kind == "physical_ball"
            return view

    owner._action_epoch_device_r05_owner = D05()
    owner.retain_action_epoch_launch(token)

    pending = owner._action_epoch_pending_launch
    assert torch.equal(pending.pending, accepted)
    assert torch.equal(
        pending.publication_ordinal[accepted],
        view.publication_ordinal[:, 0][accepted],
    )
    for field in fields(physical._row_identity.ActionEpochShotKey):
        assert torch.equal(
            getattr(pending.shot_key, field.name)[accepted],
            getattr(view.identity.shot_key, field.name)[:, 0][accepted],
        )


def test_full_n_d05_allzero_accept_is_bytewise_noop():
    owner, _scene = _physical_owner(torch.device("cpu"))
    initial = _pending(
        owner,
        key=_shot_key(num_envs=2, device=owner.device, width=0),
        pending=torch.tensor((False, True)),
        ordinal=torch.tensor((-1, 9), dtype=torch.int64),
    )
    initial.physical_state_f32[1, 0] = torch.nan
    initial.physical_state_f32[1, 1] = -0.0
    owner._action_epoch_pending_launch = initial
    before = owner._clone_action_epoch_direct_state()
    token = object()
    view = _accepted_view(
        owner, token=token, accepted=torch.zeros(2, dtype=torch.bool)
    )

    class D05:
        def require_owned_action_epoch_accepted(self, actual, *, owner_kind):
            assert actual is token and owner_kind == "physical_ball"
            return view

    owner._action_epoch_device_r05_owner = D05()
    owner.retain_action_epoch_launch(token)
    assert not owner._action_epoch_direct_state_mismatch(before)


def test_direct_state_is_in_selected_reset_and_checkpoint_hold_paths():
    finalize = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.finalize_selected_true_reset
    )
    stale = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner._selected_reset_stale_mismatch
    )
    commit = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.
        commit_prevalidated_selected_true_reset
    )
    checkpoint_gate = inspect.getsource(
        physical.ActionBallPhysicalFlightDeviceOwner.
        _require_action_epoch_checkpoint_clear
    )
    assert "_selected_reset_action_epoch_direct_state" in finalize
    assert "_action_epoch_direct_state_mismatch" in stale
    assert "action_epoch_direct_state_after" in commit
    assert "pending.pending.any()" in checkpoint_gate
    assert "_action_epoch_flight_publication_ordinal.ge(0).any()" in checkpoint_gate


def test_checkpoint_gate_holds_pending_or_live_direct_state():
    owner, _scene = _physical_owner(torch.device("cpu"), num_envs=2)
    owner._action_epoch_pending_launch = _pending(
        owner,
        key=_shot_key(num_envs=2, device=owner.device, width=0),
        pending=torch.tensor([True, False], dtype=torch.bool),
        ordinal=torch.tensor([10, -1], dtype=torch.int64),
    )
    with pytest.raises(physical.PhysicalEpochIntegrationHold, match="checkpoint HOLD"):
        owner._require_action_epoch_checkpoint_clear()

    owner._action_epoch_pending_launch = None
    owner._action_epoch_active_flight_slot[1] = 0
    with pytest.raises(physical.PhysicalEpochIntegrationHold, match="checkpoint HOLD"):
        owner._require_action_epoch_checkpoint_clear()

    owner._action_epoch_active_flight_slot[1] = -1
    owner._action_epoch_flight_publication_ordinal[1, 0] = 11
    with pytest.raises(physical.PhysicalEpochIntegrationHold, match="checkpoint HOLD"):
        owner._require_action_epoch_checkpoint_clear()


@pytest.mark.parametrize("num_envs", (2, 64))
def test_direct_state_partial_reset_preserves_peer_bytes_and_slot_mapping(
    num_envs,
):
    owner, _scene = _physical_owner(torch.device("cpu"), num_envs=num_envs)
    pending_key = _shot_key(
        num_envs=num_envs, device=owner.device, offset=10, width=0
    )
    pending = _pending(
        owner,
        key=pending_key,
        pending=torch.ones(num_envs, dtype=torch.bool, device=owner.device),
        ordinal=torch.arange(num_envs, dtype=torch.int64, device=owner.device),
        flight_slot=torch.arange(
            num_envs, dtype=torch.int64, device=owner.device
        ).remainder(2),
    )
    pending.physical_state_f32[1, 0] = torch.nan
    pending.physical_state_f32[1, 1] = -0.0
    owner._action_epoch_pending_launch = pending
    owner._action_epoch_active_flight_slot.copy_(pending.flight_slot)
    for index, field in enumerate(fields(physical._row_identity.ActionEpochShotKey)):
        grid = getattr(owner._action_epoch_flight_shot_key, field.name)
        grid.copy_(
            torch.arange(num_envs, dtype=torch.int64, device=owner.device)[:, None]
            * 10
            + torch.arange(2, dtype=torch.int64, device=owner.device)[None, :]
            + index * 1000
        )
    owner._action_epoch_flight_shot_key.action_slot.zero_()
    owner._action_epoch_flight_publication_ordinal.copy_(
        torch.arange(num_envs * 2, dtype=torch.int64).reshape(num_envs, 2)
    )
    before = owner._clone_action_epoch_direct_state()
    selected = torch.zeros(num_envs, dtype=torch.bool)
    selected[0] = True

    after = owner._selected_reset_action_epoch_direct_state(selected)

    assert not after.pending_launch.pending[0]
    assert after.active_flight_slot[0].item() == -1
    assert torch.all(after.flight_publication_ordinal[0].eq(-1))
    peer = ~selected
    for field in fields(physical._row_identity.ActionEpochShotKey):
        assert physical._device_bitwise_equal(
            getattr(after.flight_shot_key, field.name)[peer],
            getattr(before.flight_shot_key, field.name)[peer],
        )
        assert physical._device_bitwise_equal(
            getattr(after.pending_launch.shot_key, field.name)[peer],
            getattr(before.pending_launch.shot_key, field.name)[peer],
        )
    for field in fields(physical._ActionEpochPendingLaunchState):
        if field.name != "shot_key":
            assert physical._device_bitwise_equal(
                getattr(after.pending_launch, field.name)[peer],
                getattr(before.pending_launch, field.name)[peer],
            )


@pytest.mark.parametrize(
    "field_name",
    tuple(physical._row_identity.ActionEpochShotKey.__dataclass_fields__),
)
def test_launch_rejects_each_mutated_row_key_field(field_name):
    owner, scene = _physical_owner(torch.device("cpu"))
    current_grid = _shot_key(num_envs=2, device=owner.device, width=1)
    _epoch_owner, _motion, _r06 = _bind_launch_stubs(owner, current_grid)
    pending_key = physical._row_identity.ActionEpochShotKey(
        **{
            field.name: getattr(current_grid, field.name)[:, 0].clone()
            for field in fields(physical._row_identity.ActionEpochShotKey)
        }
    )
    getattr(pending_key, field_name).add_(1)
    owner._action_epoch_pending_launch = _pending(
        owner,
        key=pending_key,
        pending=torch.ones(2, dtype=torch.bool),
        ordinal=torch.tensor((3, 4), dtype=torch.int64),
    )

    before = scene.read_state_env().clone()
    owner.launch_action_epoch()
    assert torch.equal(scene.read_state_env(), before)
    assert torch.all(
        torch.bitwise_and(
            owner._action_epoch_runtime_fault_bits,
            physical._ACTION_EPOCH_RUNTIME_FAULT_DUE_IDENTITY_LOST,
        ).ne(0)
    )
    assert torch.all(owner._action_epoch_pending_launch.pending)


def test_named_runtime_fault_fails_at_existing_activity_boundary():
    owner, _scene = _physical_owner(torch.device("cpu"))
    owner._r06_owner = types.SimpleNamespace(
        flight_state=torch.zeros((2, 2), dtype=torch.int8)
    )
    owner._action_epoch_runtime_fault_bits[0] = (
        physical._ACTION_EPOCH_RUNTIME_FAULT_DUE_IDENTITY_LOST
    )

    with pytest.raises(
        physical.PhysicalEpochIntegrationHold,
        match="due_identity_lost",
    ):
        owner.refresh_action_epoch_host_activity(next_control_step=1)
    assert owner._poisoned
    assert owner._poison_reason == (
        "Physical ActionEpoch runtime fault: due_identity_lost"
    )


def test_invalid_d05_accept_is_not_staged_for_physx():
    owner, _scene = _physical_owner(torch.device("cpu"))
    token = object()
    view = _accepted_view(
        owner, token=token, accepted=torch.tensor((True, False))
    )
    # An accepted row whose deadline precedes launch is a cross-owner fault,
    # not a row that may be forwarded and asserted after the scene write.
    view.clocks.deadline_tick[0, 0] = view.clocks.launch_tick[0, 0] - 1

    class D05:
        def require_owned_action_epoch_accepted(self, actual, *, owner_kind):
            assert actual is token and owner_kind == "physical_ball"
            return view

    owner._action_epoch_device_r05_owner = D05()
    owner.retain_action_epoch_launch(token)
    assert not bool(owner._action_epoch_pending_launch.pending.any())
    assert owner._action_epoch_runtime_fault_bits[0].item() == (
        physical._ACTION_EPOCH_RUNTIME_FAULT_ACCEPT_NOT_LAUNCHABLE
    )


@pytest.mark.parametrize("num_envs", (2, 64))
def test_launch_keeps_k_slot_permutation_and_peer_publication_independent(
    num_envs,
):
    owner, _scene = _physical_owner(torch.device("cpu"), num_envs=num_envs)
    current_grid = _shot_key(num_envs=num_envs, device=owner.device, width=1)
    epoch_owner, _motion, r06 = _bind_launch_stubs(owner, current_grid)
    current_key = physical._row_identity.ActionEpochShotKey(
        **{
            field.name: getattr(current_grid, field.name)[:, 0]
            for field in fields(physical._row_identity.ActionEpochShotKey)
        }
    )
    slots = torch.arange(num_envs, dtype=torch.int64).remainder(2)
    ordinal = torch.arange(num_envs, dtype=torch.int64) + 20
    owner._action_epoch_pending_launch = _pending(
        owner,
        key=current_key,
        pending=torch.ones(num_envs, dtype=torch.bool),
        ordinal=ordinal,
        flight_slot=slots,
    )
    # The journal may advance independently; it is never the Physical join.
    epoch_owner.record.publication_ordinal.add_(1000)

    owner.launch_action_epoch()

    rows = torch.arange(num_envs, dtype=torch.int64)
    assert r06.due_rows[-1].all()
    assert epoch_owner.launch_views[-1] is r06.views[-1]
    assert torch.equal(
        owner._action_epoch_flight_shot_key.action_uid[rows, slots],
        current_key.action_uid,
    )
    assert torch.equal(
        owner._action_epoch_flight_publication_ordinal[rows, slots], ordinal
    )


def test_launch_projection_is_fixed_n_with_neutral_unselected_rows():
    owner, _scene = _physical_owner(torch.device("cpu"))
    current_grid = _shot_key(num_envs=2, device=owner.device, width=1)
    _epoch_owner, _motion, r06 = _bind_launch_stubs(owner, current_grid)
    key = physical._row_identity.ActionEpochShotKey(
        **{
            field.name: getattr(current_grid, field.name)[:, 0]
            for field in fields(physical._row_identity.ActionEpochShotKey)
        }
    )
    owner._action_epoch_pending_launch = _pending(
        owner,
        key=key,
        pending=torch.tensor((True, False)),
        ordinal=torch.tensor((8, 9), dtype=torch.int64),
    )
    owner.launch_action_epoch()
    view = r06.views[-1]
    assert view.selected_mask.tolist() == [True, False]
    assert view.due.tolist() == [True, False]
    assert view.late_launch.tolist() == [False, False]
    assert view.flight_slot.tolist() == [0, -1]
    assert view.publication_ordinal.tolist() == [8, -1]
    assert torch.all(view.target_xy_m[1].eq(0))
    for field in fields(physical._row_identity.ActionEpochShotKey):
        assert getattr(view.shot_key, field.name)[1].item() == -1
