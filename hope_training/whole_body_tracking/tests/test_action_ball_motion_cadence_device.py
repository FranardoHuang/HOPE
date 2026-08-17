"""Production Motion cadence authority tests on the exact MotionCommand."""

from __future__ import annotations

import inspect
from pathlib import Path
import copy
import sys
from types import MethodType

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "source" / "whole_body_tracking"
_MDP = _SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (_SOURCE, _MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import action_ball_continuous_runtime_transaction_device as device_r05  # noqa: E402
import test_action_ball_continuous_motion_bridge as bridge  # noqa: E402


_MDP_PACKAGE_NAME = "whole_body_tracking.tasks.tracking.mdp"
_MDP_PACKAGE = sys.modules[_MDP_PACKAGE_NAME]
_package_paths = list(getattr(_MDP_PACKAGE, "__path__", ()))
if str(_MDP) not in _package_paths:
    _MDP_PACKAGE.__path__ = [*_package_paths, str(_MDP)]
_CANONICAL_EPOCH_BEFORE_TIMING_IMPORT = (
    device_r05._require_action_epoch_module()
)
import action_ball_manifest as _manifest  # noqa: E402

for _name, _module in (
    ("action_ball_manifest", _manifest),
    ("commands", sys.modules[_MDP_PACKAGE_NAME + ".commands"]),
    ("hope_commands", sys.modules[_MDP_PACKAGE_NAME + ".hope_commands"]),
):
    sys.modules.setdefault(_MDP_PACKAGE_NAME + "." + _name, _module)
    setattr(_MDP_PACKAGE, _name, _module)
import action_ball_full_mdp_diagnostic_action_timing as timing  # noqa: E402,F401

_CANONICAL_EPOCH_AFTER_TIMING_IMPORT = (
    device_r05._require_action_epoch_module()
)
assert (
    _CANONICAL_EPOCH_AFTER_TIMING_IMPORT
    is _CANONICAL_EPOCH_BEFORE_TIMING_IMPORT
)
import action_ball_motion_cadence_device as cadence  # noqa: E402


C = bridge.C


def test_timing_import_preserves_canonical_epoch_module() -> None:
    assert (
        _CANONICAL_EPOCH_AFTER_TIMING_IMPORT
        is _CANONICAL_EPOCH_BEFORE_TIMING_IMPORT
    )


def _published_row_projection_motion(*, num_envs: int):
    command, _ = bridge._configure_unbound_command(num_envs=num_envs)
    uids = command._action_ball_continuous_code_owned_action_uids()
    command.cfg.clip_family_per_clip = tuple(
        "forehand" if slot % 2 == 0 else "backhand"
        for slot in range(len(uids))
    )
    command.bind_action_ball_continuous_parent_authorities(
        continuous_contract_authority_sha256=(
            bridge.CONTINUOUS_CONTRACT_AUTHORITY_SHA256
        ),
        recovery_contract_authority_sha256=(
            bridge.RECOVERY_CONTRACT_AUTHORITY_SHA256
        ),
        **bridge._schedule_projection(),
    )
    owner = cadence.construct_production_motion_cadence_authority(
        motion_owner=command
    )
    return command, owner


def test_action_family_catalog_is_clone_only_and_ignores_filename_and_sign():
    command, owner = _published_row_projection_motion(num_envs=2)
    first = owner.project_action_stroke_family_catalog()
    command.cfg.clip_family_per_clip = tuple("forehand" for _ in first.action_uids)
    command._motion_files = tuple("looks_like_fh.npz" for _ in first.action_uids)
    second = owner.project_action_stroke_family_catalog()
    assert first is not second and first == second
    assert first.action_uids == command._action_ball_continuous_code_owned_action_uids()
    assert first.family_codes[1] == 2
    source = inspect.getsource(type(owner).__init__)
    assert "motion_file" not in source and "mount_normal_sign" not in source


def _seal_row_projection_publication(command, *, common_step: int = 0) -> None:
    command._env.common_step_counter = common_step
    command._action_ball_continuous_published_common_step = common_step
    command._seal_action_ball_continuous_current_projection(common_step)


def _assert_parent_identity_unbound(parent) -> None:
    assert parent._bound_motion is None
    assert parent._bound_action_slot is None
    assert parent._bound_action_uid is None
    assert parent._bound_action_uids is None


def test_code_owned_diagnostic_profile_binds_exact_n2_motion_parent() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    assert profile["schema_version"] == 1
    assert profile["kind"] == (
        "whole_body_tracking.action_ball_continuous_motion_projection_v1"
    )
    assert "target" not in profile
    assert "ball" not in profile
    assert "reward" not in profile
    assert profile == parent.require_owned_motion_profile(receipt)
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    parent.bind_exact_parent_schedule(command, receipt)
    schedule = command._action_ball_continuous_schedule_projection
    assert dict(schedule) == {
        "frozen_at_step": 0,
        "sequence_origin_step": 0,
        "first_reveal_step": 2,
        "cadence_steps": 293,
        "deadline_offset_steps": 2,
        "upcoming_action_slot": 0,
        "upcoming_action_uid": command._action_ball_action_uids[0],
    }
    assert command._action_ball_continuous_parent_authority_binding is not None
    identity = parent.project_bound_action_identity(
        receipt, motion_owner=command
    )
    repeated = parent.project_bound_action_identity(
        receipt, motion_owner=command
    )
    assert type(identity) is cadence.DiagnosticMotionParentActionIdentity
    assert type(identity).__dataclass_params__.frozen is True
    assert identity is not repeated
    assert identity == repeated
    assert identity.authority is parent
    assert identity.motion_owner is command
    assert identity.action_slot == 0
    assert identity.action_uid == identity.action_uids[identity.action_slot]
    assert identity.action_uids == (
        command._action_ball_continuous_code_owned_action_uids()
    )
    assert len(identity.action_uids) == command.motion.num_segments
    assert len(set(identity.action_uids)) == len(identity.action_uids)
    assert all(type(uid) is int and uid > 0 for uid in identity.action_uids)
    parent.bind_exact_parent_schedule(command, receipt)


def test_parent_action_identity_requires_bound_receipt_and_motion_identity() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="not bound",
    ):
        parent.project_bound_action_identity(receipt, motion_owner=command)

    parent.bind_exact_parent_schedule(command, receipt)
    foreign_parent, foreign_receipt, _ = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="receipt is foreign",
    ):
        parent.project_bound_action_identity(
            foreign_receipt, motion_owner=command
        )
    foreign_command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="owner is foreign",
    ):
        parent.project_bound_action_identity(
            receipt, motion_owner=foreign_command
        )
    assert foreign_parent is not parent


def test_parent_bind_rejects_instance_noop_binder_without_retaining() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )

    def noop_binder(self, **_kwargs):
        assert self is command

    command.bind_action_ball_continuous_parent_authorities = MethodType(
        noop_binder, command
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="owner method differs: bind_action_ball_continuous_parent_authorities",
    ):
        parent.bind_exact_parent_schedule(command, receipt)
    _assert_parent_identity_unbound(parent)
    assert command._action_ball_continuous_schedule_projection is None
    assert command._action_ball_continuous_parent_authority_binding is None


def test_parent_bind_rejects_class_level_binder_substitute(
    monkeypatch,
) -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )

    def substitute(self, **_kwargs):
        assert self is command

    monkeypatch.setattr(
        type(command),
        "bind_action_ball_continuous_parent_authorities",
        substitute,
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="owner method differs: bind_action_ball_continuous_parent_authorities",
    ):
        parent.bind_exact_parent_schedule(command, receipt)
    _assert_parent_identity_unbound(parent)
    assert command._action_ball_continuous_schedule_projection is None
    assert command._action_ball_continuous_parent_authority_binding is None


def test_parent_bind_exact_binder_raise_retains_nothing() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )

    def fail_plain_int(self, _value, *, name, minimum=0):
        del name, minimum
        assert self is command
        raise RuntimeError("injected exact binder validation failure")

    command._action_ball_plain_int = MethodType(fail_plain_int, command)
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="exact Motion parent schedule bind failed",
    ):
        parent.bind_exact_parent_schedule(command, receipt)
    _assert_parent_identity_unbound(parent)
    assert command._action_ball_continuous_schedule_projection is None
    assert command._action_ball_continuous_parent_authority_binding is None


def test_parent_bind_partial_motion_write_retains_nothing(monkeypatch) -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    command_type = type(command)
    original_setattr = command_type.__setattr__

    def fail_second_motion_write(self, name, value):
        if (
            self is command
            and name
            == "_action_ball_continuous_parent_authority_binding"
            and value is not None
        ):
            raise RuntimeError("injected second Motion bind write failure")
        return original_setattr(self, name, value)

    monkeypatch.setattr(command_type, "__setattr__", fail_second_motion_write)
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="exact Motion parent schedule bind failed",
    ):
        parent.bind_exact_parent_schedule(command, receipt)
    _assert_parent_identity_unbound(parent)
    assert command._action_ball_continuous_schedule_projection is not None
    assert command._action_ball_continuous_parent_authority_binding is None


def test_parent_bind_rejects_wrong_exact_binder_after_image(
    monkeypatch,
) -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    command_type = type(command)
    original_setattr = command_type.__setattr__

    def corrupt_binding_after_image(self, name, value):
        if (
            self is command
            and name
            == "_action_ball_continuous_parent_authority_binding"
            and value is not None
        ):
            value = (*value[:3], "0" * 64)
        return original_setattr(self, name, value)

    monkeypatch.setattr(
        command_type, "__setattr__", corrupt_binding_after_image
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="exact Motion parent schedule bind after-image differs",
    ):
        parent.bind_exact_parent_schedule(command, receipt)
    _assert_parent_identity_unbound(parent)
    assert command._action_ball_continuous_schedule_projection is not None
    assert command._action_ball_continuous_parent_authority_binding is not None


def test_parent_projection_ignores_motion_private_schedule_binding_replacement() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    parent.bind_exact_parent_schedule(command, receipt)
    expected = parent.project_bound_action_identity(
        receipt, motion_owner=command
    )

    command._action_ball_continuous_schedule_projection = {
        "caller_selected": True
    }
    command._action_ball_continuous_parent_authority_binding = object()
    projected = parent.project_bound_action_identity(
        receipt, motion_owner=command
    )
    assert projected is not expected
    assert projected == expected
    assert projected.action_uids is expected.action_uids


def test_parent_rebind_rejects_action_table_drift_and_projection_stays_retained() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    parent.bind_exact_parent_schedule(command, receipt)
    expected = parent.project_bound_action_identity(
        receipt, motion_owner=command
    )
    command._action_ball_action_uids = tuple(
        uid + 1 for uid in expected.action_uids
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="diagnostic Motion parent action identity drifted",
    ):
        parent.bind_exact_parent_schedule(command, receipt)
    assert parent.project_bound_action_identity(
        receipt, motion_owner=command
    ) == expected


def test_parent_projection_rejects_retained_table_and_slot_uid_drift() -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=profile
    )
    parent.bind_exact_parent_schedule(command, receipt)
    original_uids = parent._bound_action_uids
    parent._bound_action_uids = (
        original_uids[0],
        original_uids[0],
        *original_uids[2:],
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="action identity table differs",
    ):
        parent.project_bound_action_identity(receipt, motion_owner=command)

    parent._bound_action_uids = original_uids
    parent._bound_action_uid += 1
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="slot/UID identity drifted",
    ):
        parent.project_bound_action_identity(receipt, motion_owner=command)


def test_diagnostic_profile_is_opaque_no_save_and_foreign_safe() -> None:
    owner, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    foreign, foreign_receipt, _ = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError, match="foreign"
    ):
        owner.require_owned_motion_profile(foreign_receipt)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(owner)
    with pytest.raises(TypeError, match="cannot be saved"):
        owner.__reduce__()
    profile["continuous_contract_authority_sha256"] = "0" * 64
    assert owner.require_owned_motion_profile(receipt)[
        "continuous_contract_authority_sha256"
    ] != "0" * 64
    command, _ = bridge._configure_unbound_command(
        num_envs=2, profile=foreign.require_owned_motion_profile(foreign_receipt)
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError, match="foreign"
    ):
        owner.bind_exact_parent_schedule(command, foreign_receipt)


@pytest.mark.parametrize("num_envs", (1, 2, 64))
def test_diagnostic_parent_accepts_every_positive_n_and_production_stays_hold(
    num_envs: int,
) -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=num_envs, profile=profile
    )
    parent.bind_exact_parent_schedule(command, receipt)
    identity = parent.project_bound_action_identity(
        receipt, motion_owner=command
    )
    assert identity.motion_owner is command
    assert tuple(command.clip_id.shape) == (num_envs,)
    with pytest.raises(
        cadence.MotionCadenceProductionSourceHold,
        match="lacks owner-issued C01 four-shot and C02",
    ):
        cadence.construct_production_motion_parent_schedule_authority()


@pytest.mark.parametrize("invalid", (True, False, 0, -1, 1.0, "2"))
def test_diagnostic_parent_rejects_nonpositive_or_nonexact_n(
    invalid: object,
) -> None:
    parent, receipt, profile = (
        cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
    )
    command, _ = bridge._configure_unbound_command(
        num_envs=1, profile=profile
    )
    command.num_envs = invalid
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="positive exact int",
    ):
        parent.bind_exact_parent_schedule(command, receipt)


def test_no_arg_current_row_projection_preserves_partial_due_and_close() -> None:
    command, owner = _published_row_projection_motion(num_envs=2)
    command._action_ball_continuous_reveal_due.copy_(
        torch.tensor([False, True], dtype=torch.bool)
    )
    command._action_ball_continuous_closed_mask.copy_(
        torch.tensor([True, False], dtype=torch.bool)
    )
    command._action_ball_continuous_close_reason.copy_(
        torch.tensor(
            [C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_UNPLAYED, 0],
            dtype=torch.int64,
        )
    )
    _seal_row_projection_publication(command)

    # The publication point, not the first consumer call, freezes the row.
    command._action_ball_continuous_reveal_due.logical_not_()
    command._action_ball_continuous_closed_mask.logical_not_()
    command._action_ball_continuous_close_reason.fill_(
        C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_PLAYED_SUFFIX
    )

    projected = owner.project_current_action_epoch_rows()

    assert type(projected) is C.ActionBallContinuousMotionProjection
    assert projected.reveal_due.tolist() == [False, True]
    assert projected.closed_mask.tolist() == [True, False]
    assert projected.close_reason.tolist() == [
        C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_UNPLAYED,
        C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE,
    ]
    assert not hasattr(projected, "selected_env_index")
    assert not hasattr(projected, "selected_mask")
    assert not hasattr(projected, "verdict")
    assert not hasattr(projected, "action_uid")
    assert tuple(
        inspect.signature(
            cadence.ActionBallMotionCadenceAuthority.project_current_action_epoch_rows
        ).parameters
    ) == ("self",)


def test_current_row_projection_isolates_mutation_and_rejects_next_tick_replay() -> None:
    command, owner = _published_row_projection_motion(num_envs=2)
    _seal_row_projection_publication(command)
    projected = owner.project_current_action_epoch_rows()
    projected.closed_mask[0] = True
    repeated = owner.project_current_action_epoch_rows()
    assert repeated.closed_mask.tolist() == [False, False]
    assert repeated.closed_mask.data_ptr() != projected.closed_mask.data_ptr()

    command, owner = _published_row_projection_motion(num_envs=2)
    _seal_row_projection_publication(command)
    owner.project_current_action_epoch_rows()
    command._env.common_step_counter = 1
    with pytest.raises(RuntimeError, match="stale or Command order is swapped"):
        owner.project_current_action_epoch_rows()


def test_current_row_projection_seals_inside_inference_mode() -> None:
    command, owner = _published_row_projection_motion(num_envs=2)
    command._action_ball_continuous_reveal_due.copy_(
        torch.tensor([True, False], dtype=torch.bool)
    )

    with torch.inference_mode():
        _seal_row_projection_publication(command)
        first = owner.project_current_action_epoch_rows()
        first.reveal_due.logical_not_()
        repeated = owner.project_current_action_epoch_rows()

    # Inference tensors deliberately have no version counter.  Publication
    # remains valid because correctness comes from clone isolation and the
    # current manager tick, not a same-writer ``tensor._version`` receipt.
    with pytest.raises(RuntimeError, match="do not track version counter"):
        _ = repeated.reveal_due._version
    assert repeated.reveal_due.tolist() == [True, False]
    assert first.reveal_due.tolist() == [False, True]
    assert repeated.reveal_due.data_ptr() != first.reveal_due.data_ptr()


def test_current_row_projection_rejects_instance_method_substitution() -> None:
    command, owner = _published_row_projection_motion(num_envs=2)
    _seal_row_projection_publication(command)

    def substituted(self):
        assert self is command
        return object()

    command.action_ball_continuous_current_projection = MethodType(
        substituted, command
    )
    with pytest.raises(
        cadence.MotionCadenceAuthorityConflictError,
        match="current row projection method differs",
    ):
        owner.project_current_action_epoch_rows()


def test_current_row_projection_path_has_no_hot_host_sync() -> None:
    sources = (
        inspect.getsource(
            cadence.ActionBallMotionCadenceAuthority.project_current_action_epoch_rows
        ),
        inspect.getsource(
            C.MotionCommand.action_ball_continuous_current_projection
        ),
        inspect.getsource(
            C.MotionCommand._seal_action_ball_continuous_current_projection
        ),
    )
    for source in sources:
        for forbidden in (
            ".item(",
            ".cpu(",
            ".numpy(",
            ".tolist(",
            "torch.equal(",
            "torch._assert_async(",
        ):
            assert forbidden not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_current_row_projection_remains_full_n_and_device_resident_on_cuda() -> None:
    command, owner = _published_row_projection_motion(num_envs=2)
    device = torch.device("cuda", torch.cuda.current_device())
    for name, value in tuple(vars(command).items()):
        if torch.is_tensor(value):
            setattr(command, name, value.to(device))
    for name, value in tuple(vars(command.motion).items()):
        if torch.is_tensor(value):
            setattr(command.motion, name, value.to(device))
    for name, value in tuple(command.metrics.items()):
        if torch.is_tensor(value):
            command.metrics[name] = value.to(device)
    command._env.scene.env_origins = command._env.scene.env_origins.to(device)
    command.device = device
    command._action_ball_continuous_reveal_due.copy_(
        torch.tensor([True, False], dtype=torch.bool, device=device)
    )
    _seal_row_projection_publication(command)

    projected = owner.project_current_action_epoch_rows()

    assert type(projected) is C.ActionBallContinuousMotionProjection
    assert projected.reveal_due.tolist() == [True, False]
    for name in (
        "episode_tick",
        "reveal_due",
        "closed_mask",
        "close_reason",
        "deadline_due",
        "scheduled_ordinal",
        "reveal_tick",
        "deadline_tick",
        "next_reveal_tick",
        "ready_at_reveal",
        "motion_active",
        "ready_reference_active",
        "suffix_complete",
        "reset_generation",
        "swing_generation",
    ):
        assert getattr(projected, name).device == device


@pytest.mark.parametrize("num_envs", (1, 2, 64))
def test_current_row_projection_has_one_exact_type_for_all_n(
    num_envs: int,
) -> None:
    command, owner = _published_row_projection_motion(num_envs=num_envs)
    due = torch.arange(num_envs, dtype=torch.int64).remainder(2).bool()
    command._action_ball_continuous_reveal_due.copy_(due)
    _seal_row_projection_publication(command)

    projected = owner.project_current_action_epoch_rows()

    assert type(projected) is C.ActionBallContinuousMotionProjection
    assert projected.reveal_due.dtype is torch.bool
    assert projected.closed_mask.dtype is torch.bool
    assert projected.close_reason.dtype is torch.int64
    assert tuple(projected.reveal_due.shape) == (num_envs,)
    assert tuple(projected.closed_mask.shape) == (num_envs,)
    assert tuple(projected.close_reason.shape) == (num_envs,)
    assert torch.equal(projected.reveal_due, due)
