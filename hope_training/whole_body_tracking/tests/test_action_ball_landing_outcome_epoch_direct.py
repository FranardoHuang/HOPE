"""Focused ActionEpoch publication tests for the real R06 mailbox."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
SOURCE = ROOT / "source/whole_body_tracking"
MDP = SOURCE / "whole_body_tracking/tasks/tracking/mdp"
for path in (TESTS, SOURCE, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_action_ball_landing_outcome_device as T  # noqa: E402

# The legacy R06 fixture imports the global-drain module only to build an
# unrelated D05 boundary token.  Current shared bytes use Python 3.9 type-alias
# expressions at import time, while the exact local CPU interpreter is 3.8.
# A postponed-annotation test load preserves the runtime class semantics.
_DRAIN = MDP / "action_ball_full_mdp_ppo_drain.py"
_drain_source = _DRAIN.read_text(encoding="utf-8")
_drain_source = _drain_source.replace(
    "CheckpointFieldIdentity = (\n"
    "    tuple[str, str, int] | tuple[str, str, int, int]\n"
    ")\n"
    "CheckpointSchemaIdentity = tuple[\n"
    "    tuple[str, tuple[CheckpointFieldIdentity, ...]], ...\n"
    "]",
    "CheckpointFieldIdentity = object\nCheckpointSchemaIdentity = object",
)
_drain_name = (
    "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_ppo_drain"
)
_drain_module = sys.modules.get(_drain_name)
if _drain_module is None:
    _drain_module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(_drain_name, loader=None)
    )
    _drain_module.__file__ = str(_DRAIN)
    sys.modules[_drain_name] = _drain_module
    exec(compile(_drain_source, str(_DRAIN), "exec"), _drain_module.__dict__)
setattr(
    sys.modules["whole_body_tracking.tasks.tracking.mdp"],
    "action_ball_full_mdp_ppo_drain",
    _drain_module,
)

_device_helper_path = TESTS / "test_action_ball_continuous_runtime_transaction_device.py"
_device_helper_source = _device_helper_path.read_text(encoding="utf-8")
_device_helper_source = _device_helper_source.replace(
    "_DRAIN_SPEC = importlib.util.spec_from_file_location(_DRAIN_NAME, _GLOBAL_DRAIN_PATH)\n"
    "assert _DRAIN_SPEC is not None and _DRAIN_SPEC.loader is not None\n"
    "global_drain = importlib.util.module_from_spec(_DRAIN_SPEC)\n"
    "sys.modules[_DRAIN_NAME] = global_drain\n"
    "setattr(sys.modules[\"whole_body_tracking.tasks.tracking.mdp\"],\n"
    "        \"action_ball_full_mdp_ppo_drain\", global_drain)\n"
    "_DRAIN_SPEC.loader.exec_module(global_drain)",
    "global_drain = sys.modules[_DRAIN_NAME]",
)
_device_helper_name = "_landing_outcome_device_helper_test_action_ball_continuous_runtime_transaction_device"
_device_helper = importlib.util.module_from_spec(
    importlib.util.spec_from_loader(_device_helper_name, loader=None)
)
_device_helper.__file__ = str(_device_helper_path)
sys.modules[_device_helper_name] = _device_helper
exec(
    compile(_device_helper_source, str(_device_helper_path), "exec"),
    _device_helper.__dict__,
)
T._HELPER_MODULES[_device_helper_path.name] = _device_helper
sys.modules.setdefault(
    "test_action_ball_continuous_runtime_transaction_device", _device_helper
)


def _load_epoch():
    name = "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        name, MDP / "action_ball_full_mdp_epoch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load_epoch()
D = T.D

if sys.version_info < (3, 10):
    # The checked-in runtime uses dataclass slots (Python >=3.10).  The local
    # CPU torch environment is 3.8; load the same code with only that storage
    # optimization removed so the real Physical -> R06 graph remains runnable.
    _genesis_path = SOURCE / "action_ball_full_mdp_reset_genesis.py"
    _genesis_source = _genesis_path.read_text(encoding="utf-8").replace(
        ", slots=True", ""
    ).replace("@dataclass(slots=True)", "@dataclass")
    _genesis = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(
            "action_ball_full_mdp_reset_genesis", loader=None
        )
    )
    _genesis.__file__ = str(_genesis_path)
    sys.modules["action_ball_full_mdp_reset_genesis"] = _genesis
    exec(
        compile(_genesis_source, str(_genesis_path), "exec"),
        _genesis.__dict__,
    )


def test_previous_paid_row_projection_is_exact_clone_only_full_n():
    owner = T._coordinator(rows=2)
    owner._previous_paid_valid.copy_(torch.tensor((True, False)))
    owner._previous_paid_action_epoch.copy_(torch.tensor((True, False)))
    for name, values in (
        ("reset_generation", (3, -1)),
        ("ball_generation", (7, -1)),
        ("action_uid", (11, -1)),
        ("action_slot", (0, -1)),
        ("shot_index", (5, -1)),
        ("task_identity", (101, -1)),
        ("outcome_identity", (201, -1)),
        ("ball_identity", (301, -1)),
    ):
        getattr(owner, "_previous_paid_" + name).copy_(
            torch.tensor(values, dtype=torch.int64)
        )
    owner._previous_paid_publication_ordinal.copy_(torch.tensor((19, -1)))
    owner._previous_paid_settlement_control_step.copy_(torch.tensor((23, -1)))
    owner._previous_paid_payment_step.copy_(torch.tensor((29, -1)))
    owner._previous_paid_payment_step_highwater.copy_(torch.tensor((29, -1)))

    projected = owner.project_previous_paid_action_epoch_rows()
    assert type(projected) is D.PreviousPaidActionEpochRows
    assert torch.equal(projected.valid, torch.tensor((True, False)))
    assert torch.equal(projected.shot_key.action_uid, torch.tensor((11, -1)))
    assert torch.equal(projected.publication_ordinal, torch.tensor((19, -1)))
    assert torch.equal(projected.settlement_step, torch.tensor((23, -1)))
    assert torch.equal(projected.payment_step, torch.tensor((29, -1)))

    projected.valid.zero_()
    projected.shot_key.action_uid.fill_(999)
    projected.payment_step.fill_(999)
    assert torch.equal(owner._previous_paid_valid, torch.tensor((True, False)))
    assert torch.equal(owner._previous_paid_action_uid, torch.tensor((11, -1)))
    assert torch.equal(owner._previous_paid_payment_step, torch.tensor((29, -1)))


class _PhysicalEpochSource:
    def action_epoch_r06_launch_projection(self):
        raise AssertionError("R06 payment tests must not pull Physical launch")

    def require_owned_action_epoch_r06_postphysics_projection(self):
        raise AssertionError("R06 payment tests must not pull Physical postphysics")


def _bound_epoch(owner):
    epoch = E.ActionEpochOwner(
        num_envs=owner.num_envs,
        device=owner.device,
        shot_slot_capacity=1,
        initial_reset_generation=1,
    )
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(
            (owner.num_envs,), dtype=torch.bool, device=owner.device
        ),
        reset_generation=torch.ones(
            (owner.num_envs,), dtype=torch.int64, device=owner.device
        ),
    )
    owner.bind_action_ball_full_mdp_epoch_owner(epoch)
    physical = _PhysicalEpochSource()
    epoch.bind_fact_owner("physical_ball", physical)
    epoch.bind_async_owner("physical_ball", physical)
    return epoch, physical


def _row_key(owner, *, task_delta: int = 0):
    return E.ActionEpochShotKey(
        reset_generation=torch.tensor((1, 1), dtype=torch.int64, device=owner.device),
        ball_generation=torch.tensor((3, 4), dtype=torch.int64, device=owner.device),
        action_uid=torch.tensor((7, 7), dtype=torch.int64, device=owner.device),
        action_slot=torch.zeros(2, dtype=torch.int64, device=owner.device),
        shot_index=torch.tensor((11, 12), dtype=torch.int64, device=owner.device),
        task_identity=torch.tensor(
            (101 + task_delta, 102 + task_delta),
            dtype=torch.int64,
            device=owner.device,
        ),
        outcome_identity=torch.tensor((201, 202), dtype=torch.int64, device=owner.device),
        ball_identity=torch.tensor((301, 302), dtype=torch.int64, device=owner.device),
    )


def _seed_rowwise_mailbox(owner, key, *, publication=(17, 18), settlement=(23, 24)):
    target = torch.zeros(owner._mailbox_shape, dtype=torch.bool, device=owner.device)
    target[:, 0] = True
    owner._mailbox_action_epoch[:, 0] = True
    owner._mailbox_history_valid[:, 0] = True
    owner._mailbox_physical_retired[:, 0] = True
    owner._mailbox_reserved[:, 0] = True
    owner._mailbox_state[:, 0] = D.MAILBOX_SETTLED_UNPAID
    for field in E.fields(E.ActionEpochShotKey):
        getattr(owner, "_mailbox_" + field.name)[:, 0] = getattr(key, field.name)
    owner._mailbox_publication_ordinal[:, 0] = torch.tensor(
        publication, dtype=torch.int64, device=owner.device
    )
    owner._mailbox_settlement_control_step[:, 0] = torch.tensor(
        settlement, dtype=torch.int64, device=owner.device
    )
    owner._mailbox_observation_ordinal[:, 0] = 0
    return target


def _install_payment_projector(epoch, rows):
    epoch._current_payment_rows = rows.clone()


def test_rowwise_outcome_projection_uses_retained_key_without_current_epoch():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    key = _row_key(owner)
    _seed_rowwise_mailbox(owner, key)
    owner._mailbox_settlement_cause[:, 0] = D.SETTLEMENT_CAUSE_CONTACT_DEADLINE
    projected = owner.project_current_action_epoch_outcome_rows()
    assert type(projected) is D.ActionEpochR06OutcomeRows
    assert torch.equal(projected.valid, torch.ones(2, dtype=torch.bool))
    assert torch.equal(projected.shot_key.shot_index, key.shot_index)
    assert torch.equal(projected.publication_ordinal, torch.tensor((17, 18)))
    assert torch.equal(projected.settlement_step, torch.tensor((23, 24)))
    assert tuple(projected.fact_values.shape) == (2, D.R06_ACTION_EPOCH_FACT_F32_WIDTH)
    projected.shot_key.task_identity.fill_(999)
    projected.publication_ordinal.fill_(999)
    assert torch.equal(owner._mailbox_task_identity[:, 0], key.task_identity)
    assert torch.equal(owner._mailbox_publication_ordinal[:, 0], torch.tensor((17, 18)))


def test_rowwise_payment_exact_key_close_duplicate_and_unconsumed_hold():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    key = _row_key(owner)
    _seed_rowwise_mailbox(owner, key)
    payment = E.ActionEpochRewardPaymentRows(
        valid=torch.ones(2, dtype=torch.bool, device=owner.device),
        shot_key=key.clone(),
        payment_step=torch.tensor((29, 30), dtype=torch.int64, device=owner.device),
    )
    _install_payment_projector(epoch, payment)
    epoch._publication.current.publication_ordinal.fill_(999)

    owner.close_action_ball_full_mdp_epoch_reward_rows()
    assert torch.equal(owner._previous_paid_payment_step, payment.payment_step)
    assert not owner._mailbox_history_valid.any()
    version = owner._mutation_version.clone()
    owner.close_action_ball_full_mdp_epoch_reward_rows()
    assert torch.equal(owner._mutation_version, version)

    newer = _row_key(owner, task_delta=1000)
    _seed_rowwise_mailbox(
        owner, newer, publication=(19, 20), settlement=(31, 32)
    )
    epoch._current_payment_rows = E.ActionEpochRewardPaymentRows(
        valid=torch.ones(2, dtype=torch.bool, device=owner.device),
        shot_key=newer,
        payment_step=torch.tensor((33, 34), dtype=torch.int64, device=owner.device),
    )
    with pytest.raises(RuntimeError, match="overwrite unconsumed debt"):
        owner.close_action_ball_full_mdp_epoch_reward_rows()


def test_closed_row_projection_releases_only_exact_previous_paid_debt():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    key = _row_key(owner)
    _seed_rowwise_mailbox(owner, key)
    payment = E.ActionEpochRewardPaymentRows(
        valid=torch.ones(2, dtype=torch.bool, device=owner.device),
        shot_key=key.clone(),
        payment_step=torch.tensor((29, 30), dtype=torch.int64, device=owner.device),
    )
    _install_payment_projector(epoch, payment)
    owner.close_action_ball_full_mdp_epoch_reward_rows()
    peer_before = {
        name: tensor[1].detach().clone()
        for name, tensor in owner._checkpoint_tensors().items()
        if name.startswith("previous_paid_") and tensor.ndim >= 1
    }
    epoch._current_closed_rows = E.ActionEpochClosedRows(
        valid=torch.tensor((True, False), dtype=torch.bool, device=owner.device),
        shot_key=key.clone(),
    )
    owner.consume_closed_action_epoch_rows()

    assert not bool(owner._previous_paid_valid[0])
    assert int(owner._previous_paid_action_uid[0]) == -1
    assert int(owner._previous_paid_payment_step[0]) == -1
    assert int(owner._previous_paid_payment_step_highwater[0]) == 29
    assert bool(owner._previous_paid_valid[1])
    for name, expected in peer_before.items():
        assert torch.equal(owner._checkpoint_tensors()[name][1], expected)


def test_partial_payment_closes_only_valid_row_and_retains_peer_mailbox():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    key = _row_key(owner)
    _seed_rowwise_mailbox(owner, key)
    payment = E.ActionEpochRewardPaymentRows(
        valid=torch.tensor((True, False), dtype=torch.bool, device=owner.device),
        shot_key=key.clone(),
        payment_step=torch.tensor((29, 30), dtype=torch.int64, device=owner.device),
    )
    _install_payment_projector(epoch, payment)
    owner.close_action_ball_full_mdp_epoch_reward_rows()
    assert torch.equal(owner._previous_paid_valid, torch.tensor((True, False)))
    assert not bool(owner._mailbox_history_valid[0, 0])
    assert bool(owner._mailbox_history_valid[1, 0])
    assert bool(owner._mailbox_action_epoch[1, 0])


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("wrong_task_same_ordinal", "mismatched retained identity"),
        ("wrong_ball_same_ordinal", "mismatched retained identity"),
        ("wrong_shot_same_ordinal", "mismatched retained identity"),
        ("payment_before_settlement", "preceded settlement"),
        ("payment_step_regression", "regressed"),
    ),
)
def test_rowwise_payment_failure_boundaries(mode, message):
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    mailbox_key = _row_key(owner)
    target = _seed_rowwise_mailbox(owner, mailbox_key)
    payment_key = mailbox_key.clone()
    if mode.startswith("wrong_"):
        field = {
            "wrong_task_same_ordinal": "task_identity",
            "wrong_ball_same_ordinal": "ball_identity",
            "wrong_shot_same_ordinal": "shot_index",
        }[mode]
        getattr(payment_key, field).add_(1000)
    if mode == "payment_step_regression":
        owner._previous_paid_payment_step_highwater.fill_(40)
    payment_step = (
        torch.tensor((22, 23), dtype=torch.int64, device=owner.device)
        if mode == "payment_before_settlement"
        else torch.tensor((29, 30), dtype=torch.int64, device=owner.device)
    )
    payment = E.ActionEpochRewardPaymentRows(
        valid=torch.ones(2, dtype=torch.bool, device=owner.device),
        shot_key=payment_key,
        payment_step=payment_step,
    )
    _install_payment_projector(epoch, payment)
    with pytest.raises(RuntimeError, match=message):
        owner.close_action_ball_full_mdp_epoch_reward_rows()
    assert owner._device_sticky_poison.all()
    assert owner._mailbox_history_valid[target].all()
    assert not owner._previous_paid_valid.any()


def test_fixed_n_physical_launch_installs_due_and_neutral_rows_without_compaction():
    import test_action_ball_physical_epoch_hot_lane as hot

    device = torch.device("cpu")
    physical, _scene = hot._physical_owner(device, num_envs=2)
    owner = T._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=2,
        bind_physical_park=False,
        device=device,
    )
    epoch, _ = _bound_epoch(owner)
    physical._action_epoch_owner = epoch
    physical.bind_r06_owner(owner)

    selected = torch.tensor((True, False), dtype=torch.bool, device=device)
    due = selected.clone()
    key = E.ActionEpochShotKey(
        reset_generation=torch.tensor((1, -1), dtype=torch.int64, device=device),
        ball_generation=torch.tensor((3, -1), dtype=torch.int64, device=device),
        action_uid=torch.tensor((7, -1), dtype=torch.int64, device=device),
        action_slot=torch.tensor((0, -1), dtype=torch.int64, device=device),
        shot_index=torch.tensor((11, -1), dtype=torch.int64, device=device),
        task_identity=torch.tensor((101, -1), dtype=torch.int64, device=device),
        outcome_identity=torch.tensor((201, -1), dtype=torch.int64, device=device),
        ball_identity=torch.tensor((301, -1), dtype=torch.int64, device=device),
    )
    target = torch.zeros((2, 2), dtype=torch.float32, device=device)
    target[0, 0] = (
        owner.profile.opponent_table_x_min_m
        + owner.profile.opponent_table_x_max_m
    ) / 2.0
    target[0, 1] = (
        owner.profile.table_y_min_m + owner.profile.table_y_max_m
    ) / 2.0
    physical._action_epoch_active_r06_launch = hot.physical.ActionEpochR06LaunchProjection(
        selected_mask=selected,
        due=due,
        late_launch=torch.zeros(2, dtype=torch.bool, device=device),
        flight_slot=torch.tensor((0, -1), dtype=torch.int64, device=device),
        shot_key=key,
        publication_ordinal=torch.tensor((17, -1), dtype=torch.int64, device=device),
        target_xy_m=target,
        launch_control_step=torch.tensor((3, -1), dtype=torch.int64, device=device),
        contact_deadline_control_step=torch.tensor((4, -1), dtype=torch.int64, device=device),
        crossing_horizon_control_step=torch.tensor((5, -1), dtype=torch.int64, device=device),
        physical_owner=physical,
        epoch_owner=epoch,
        owner_identity=physical._owner_identity,
        _token=hot.physical._ACTION_EPOCH_R06_LAUNCH_TOKEN,
    )
    try:
        owner.install_action_ball_full_mdp_epoch_launch_from_physical()
    finally:
        physical._action_epoch_active_r06_launch = None

    assert owner._flight_state[0, 0] == D.FLIGHT_INBOUND
    assert not owner._flight_action_epoch[1].any()
    assert owner._mailbox_reserved[0].sum() == 1
    assert not owner._mailbox_reserved[1].any()

    before = owner._mutation_version.clone()
    neutral_i64 = torch.full((2,), -1, dtype=torch.int64, device=device)
    neutral_key = E.ActionEpochShotKey(
        **{
            field.name: neutral_i64.clone()
            for field in E.fields(E.ActionEpochShotKey)
        }
    )
    physical._action_epoch_active_r06_launch = hot.physical.ActionEpochR06LaunchProjection(
        selected_mask=torch.zeros(2, dtype=torch.bool, device=device),
        due=torch.zeros(2, dtype=torch.bool, device=device),
        late_launch=torch.zeros(2, dtype=torch.bool, device=device),
        flight_slot=neutral_i64.clone(),
        shot_key=neutral_key,
        publication_ordinal=neutral_i64.clone(),
        target_xy_m=torch.zeros((2, 2), dtype=torch.float32, device=device),
        launch_control_step=neutral_i64.clone(),
        contact_deadline_control_step=neutral_i64.clone(),
        crossing_horizon_control_step=neutral_i64.clone(),
        physical_owner=physical,
        epoch_owner=epoch,
        owner_identity=physical._owner_identity,
        _token=hot.physical._ACTION_EPOCH_R06_LAUNCH_TOKEN,
    )
    try:
        owner.install_action_ball_full_mdp_epoch_launch_from_physical()
    finally:
        physical._action_epoch_active_r06_launch = None
    assert torch.equal(owner._mutation_version, before)
