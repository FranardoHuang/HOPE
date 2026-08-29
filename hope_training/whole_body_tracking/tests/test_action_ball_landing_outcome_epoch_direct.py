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


def test_r06_bind_rejects_named_fault_registry_drift(monkeypatch):
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch = E.ActionEpochOwner(
        num_envs=owner.num_envs,
        device=owner.device,
        shot_slot_capacity=1,
        initial_reset_generation=1,
    )
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=owner.device),
        reset_generation=torch.ones(2, dtype=torch.int64, device=owner.device),
    )
    names = list(E.ACTION_EPOCH_ROW_FAULT_NAMES)
    target = E.ROW_FAULT_R06_PAYMENT_BEFORE_SETTLEMENT
    names = tuple(
        (bit, "renamed_payment_fault" if bit == target else name)
        for bit, name in names
    )
    monkeypatch.setattr(E, "ACTION_EPOCH_ROW_FAULT_NAMES", names)

    with pytest.raises(D.LandingOutcomeDeviceError, match="row-fault ABI differs"):
        owner.bind_action_ball_full_mdp_epoch_owner(epoch)


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


def _row_fault_words(epoch):
    return epoch._undrained_row_fault_bits.detach().cpu().tolist()


def _assert_row_fault_words(epoch, row0_bits):
    assert _row_fault_words(epoch) == [row0_bits, 0]


def _snapshot_mailbox_row(owner, row):
    return {
        name: tensor[row].detach().clone()
        for name, tensor in owner._checkpoint_tensors().items()
        if name.startswith("mailbox_")
        and tensor.ndim >= 2
        and tuple(tensor.shape[:2]) == owner._mailbox_shape
    }


def _assert_mailbox_row_unchanged(owner, row, before):
    current = owner._checkpoint_tensors()
    for name, expected in before.items():
        assert torch.equal(current[name][row], expected), name


def _duplicate_mailbox_row(owner, row=0):
    for name, tensor in vars(owner).items():
        if (
            name.startswith("_mailbox_")
            and name != "_mailbox_slot_ids"
            and type(tensor) is torch.Tensor
            and tensor.ndim >= 2
            and tuple(tensor.shape[:2]) == owner._mailbox_shape
        ):
            tensor[row, 1].copy_(tensor[row, 0])


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


def test_outcome_duplicate_is_named_neutral_and_does_not_hide_healthy_peer():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    key = _row_key(owner)
    _seed_rowwise_mailbox(owner, key)
    owner._mailbox_settlement_cause[:, 0] = D.SETTLEMENT_CAUSE_CONTACT_DEADLINE
    _duplicate_mailbox_row(owner)
    before = {
        row: _snapshot_mailbox_row(owner, row) for row in range(owner.num_envs)
    }

    projected = owner.project_current_action_epoch_outcome_rows()

    _assert_row_fault_words(
        epoch, E.ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE
    )
    assert projected.valid.tolist() == [False, True]
    assert projected.publication_ordinal.tolist() == [-1, 18]
    assert not projected.fact_values[0].any()
    for row, before_row in before.items():
        _assert_mailbox_row_unchanged(owner, row, before_row)


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
    mailbox_before = {
        row: _snapshot_mailbox_row(owner, row) for row in range(owner.num_envs)
    }
    owner.close_action_ball_full_mdp_epoch_reward_rows()
    assert _row_fault_words(epoch) == [
        E.ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE,
        E.ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE,
    ]
    for row, before_row in mailbox_before.items():
        _assert_mailbox_row_unchanged(owner, row, before_row)


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


@pytest.mark.parametrize(
    ("mode", "expected_bit"),
    (
        ("invalid_projection", E.ROW_FAULT_R06_CLOSED_PROJECTION_CONTRACT),
        ("debt_mismatch", E.ROW_FAULT_R06_CLOSED_DEBT_MISMATCH),
    ),
)
def test_closed_row_fault_retains_bad_debt_and_releases_healthy_peer(
    mode, expected_bit
):
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
    assert _row_fault_words(epoch) == [0, 0]
    row0_before = {
        name: tensor[0].detach().clone()
        for name, tensor in owner._checkpoint_tensors().items()
        if name.startswith("previous_paid_") and tensor.ndim >= 1
    }
    closed_key = key.clone()
    if mode == "invalid_projection":
        closed_key.action_uid[0] = 0
    else:
        closed_key.task_identity[0].add_(1000)
    epoch._current_closed_rows = E.ActionEpochClosedRows(
        valid=torch.ones(2, dtype=torch.bool, device=owner.device),
        shot_key=closed_key,
    )

    owner.consume_closed_action_epoch_rows()

    _assert_row_fault_words(epoch, expected_bit)
    assert bool(owner._previous_paid_valid[0])
    assert not bool(owner._previous_paid_valid[1])
    current = owner._checkpoint_tensors()
    for name, expected in row0_before.items():
        assert torch.equal(current[name][0], expected), name


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
    ("mode", "expected_bit"),
    (
        ("invalid_projection", E.ROW_FAULT_R06_PAYMENT_PROJECTION_CONTRACT),
        ("duplicate_mailbox", E.ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE),
        ("wrong_task_same_ordinal", E.ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED),
        ("wrong_ball_same_ordinal", E.ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED),
        ("wrong_shot_same_ordinal", E.ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED),
        ("payment_before_settlement", E.ROW_FAULT_R06_PAYMENT_BEFORE_SETTLEMENT),
        ("payment_step_regression", E.ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION),
        (
            "unconsumed_debt_overwrite",
            E.ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE,
        ),
    ),
)
def test_rowwise_payment_fault_masks_only_bad_row_and_preserves_causal_bit(
    mode, expected_bit
):
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    mailbox_key = _row_key(owner)
    _seed_rowwise_mailbox(owner, mailbox_key)
    payment_key = mailbox_key.clone()
    if mode.startswith("wrong_"):
        field = {
            "wrong_task_same_ordinal": "task_identity",
            "wrong_ball_same_ordinal": "ball_identity",
            "wrong_shot_same_ordinal": "shot_index",
        }[mode]
        getattr(payment_key, field)[0].add_(1000)
    if mode == "duplicate_mailbox":
        _duplicate_mailbox_row(owner)
    if mode == "payment_step_regression":
        owner._previous_paid_payment_step_highwater[0] = 40
    if mode == "unconsumed_debt_overwrite":
        owner._previous_paid_valid[0] = True
    payment_step = torch.tensor(
        (22 if mode == "payment_before_settlement" else 29, 30),
        dtype=torch.int64,
        device=owner.device,
    )
    if mode == "invalid_projection":
        payment_step[0] = -1
    payment = E.ActionEpochRewardPaymentRows(
        valid=torch.ones(2, dtype=torch.bool, device=owner.device),
        shot_key=payment_key,
        payment_step=payment_step,
    )
    _install_payment_projector(epoch, payment)
    row0_before = _snapshot_mailbox_row(owner, 0)

    owner.close_action_ball_full_mdp_epoch_reward_rows()

    _assert_row_fault_words(epoch, expected_bit)
    _assert_mailbox_row_unchanged(owner, 0, row0_before)
    assert bool(owner._mailbox_history_valid[0, 0])
    assert not bool(owner._mailbox_history_valid[1, 0])
    assert bool(owner._previous_paid_valid[1])
    assert int(owner._previous_paid_payment_step[1]) == 30
    if mode != "unconsumed_debt_overwrite":
        assert not bool(owner._previous_paid_valid[0])
    else:
        assert bool(owner._previous_paid_valid[0])


def test_payment_compound_fault_packs_both_causes_without_mutating_bad_row():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    key = _row_key(owner)
    _seed_rowwise_mailbox(owner, key)
    owner._previous_paid_valid[0] = True
    owner._previous_paid_payment_step_highwater[0] = 40
    payment = E.ActionEpochRewardPaymentRows(
        valid=torch.ones(2, dtype=torch.bool, device=owner.device),
        shot_key=key.clone(),
        payment_step=torch.tensor((29, 30), dtype=torch.int64, device=owner.device),
    )
    _install_payment_projector(epoch, payment)
    row0_before = _snapshot_mailbox_row(owner, 0)

    owner.close_action_ball_full_mdp_epoch_reward_rows()

    expected = (
        E.ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION
        | E.ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE
    )
    _assert_row_fault_words(epoch, expected)
    _assert_mailbox_row_unchanged(owner, 0, row0_before)
    assert bool(owner._previous_paid_valid[0])
    assert not bool(owner._mailbox_history_valid[1, 0])
    assert bool(owner._previous_paid_valid[1])

    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.row_fault_bits.tolist() == [expected, 0]


def test_r06_runtime_fault_helper_rejects_unknown_or_compound_reason_bits():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    baseline = epoch._undrained_row_fault_bits.clone()
    rows = torch.tensor((True, False), dtype=torch.bool, device=owner.device)
    for invalid_bit in (
        1 << 40,
        E.ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION
        | E.ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE,
    ):
        with pytest.raises(D.LandingOutcomeDeviceError, match="ABI differs"):
            owner._latch_action_epoch_row_fault(
                rows, epoch_reason_bit=invalid_bit
            )
    assert torch.equal(epoch._undrained_row_fault_bits, baseline)


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

    # The direct install writes only R06's typed ActionEpoch plane.  The
    # observation projection must therefore select it without consulting the
    # still-default legacy DeviceLandingOutcomeKey buffers.
    assert int(owner._flight_key_ints["action_uid"][0, 0]) == 0
    observation_projection = owner.action_ball_full_mdp_observation_projection()
    observation = owner.require_owned_action_epoch_current_flight_observation(
        observation_projection,
        current_shot_key=key,
        current_publication_ordinal=torch.tensor(
            (17, -1), dtype=torch.int64, device=device
        ),
    )
    assert type(observation) is D.ActionEpochR06CurrentFlightObservationView
    assert observation.r06_owner is owner
    assert observation.publication_identity is observation_projection
    assert observation.flight_slot.tolist() == [0, -1]
    assert not observation.contact_valid.any()
    assert not observation.net_crossed.any()
    assert not observation.net_clear.any()

    observation.flight_slot.fill_(1)
    observation.contact_valid.fill_(True)
    observation.net_crossed.fill_(True)
    observation.net_clear.fill_(True)
    assert int(owner._flight_state[0, 0]) == D.FLIGHT_INBOUND
    assert not owner._flight_contact_valid.any()
    assert not owner._flight_net_crossed.any()
    assert not owner._flight_net_clear.any()

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


@pytest.mark.parametrize(
    ("mode", "expected_bit"),
    (
        ("selection", E.ROW_FAULT_R06_LAUNCH_SELECTION_CONTRACT),
        ("identity", E.ROW_FAULT_R06_LAUNCH_IDENTITY_CONTRACT),
    ),
)
def test_launch_fault_masks_bad_row_and_installs_healthy_peer(mode, expected_bit):
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

    selected = torch.ones(2, dtype=torch.bool, device=device)
    due = torch.ones(2, dtype=torch.bool, device=device)
    key = _row_key(owner)
    flight_slot = torch.zeros(2, dtype=torch.int64, device=device)
    publication = torch.tensor((17, 18), dtype=torch.int64, device=device)
    target = torch.empty((2, 2), dtype=torch.float32, device=device)
    target[:, 0] = (
        owner.profile.opponent_table_x_min_m
        + owner.profile.opponent_table_x_max_m
    ) / 2.0
    target[:, 1] = (
        owner.profile.table_y_min_m + owner.profile.table_y_max_m
    ) / 2.0
    launch = torch.tensor((3, 3), dtype=torch.int64, device=device)
    deadline = torch.tensor((4, 4), dtype=torch.int64, device=device)
    horizon = torch.tensor((5, 5), dtype=torch.int64, device=device)
    if mode == "selection":
        selected[0] = False
        flight_slot[0] = -1
        publication[0] = -1
        target[0].zero_()
        launch[0] = deadline[0] = horizon[0] = -1
        for field in E.fields(E.ActionEpochShotKey):
            getattr(key, field.name)[0] = -1
    else:
        key.action_uid[0] = 0
    physical._action_epoch_active_r06_launch = hot.physical.ActionEpochR06LaunchProjection(
        selected_mask=selected,
        due=due,
        late_launch=torch.zeros(2, dtype=torch.bool, device=device),
        flight_slot=flight_slot,
        shot_key=key,
        publication_ordinal=publication,
        target_xy_m=target,
        launch_control_step=launch,
        contact_deadline_control_step=deadline,
        crossing_horizon_control_step=horizon,
        physical_owner=physical,
        epoch_owner=epoch,
        owner_identity=physical._owner_identity,
        _token=hot.physical._ACTION_EPOCH_R06_LAUNCH_TOKEN,
    )
    try:
        owner.install_action_ball_full_mdp_epoch_launch_from_physical()
    finally:
        physical._action_epoch_active_r06_launch = None

    _assert_row_fault_words(epoch, expected_bit)
    assert not owner._flight_action_epoch[0].any()
    assert not owner._mailbox_reserved[0].any()
    assert bool(owner._flight_action_epoch[1, 0])
    assert bool(owner._mailbox_reserved[1, 0])


_OBSERVATION_KEY_VALUES = {
    "reset_generation": 1,
    "ball_generation": 3,
    "action_uid": 7,
    "action_slot": 0,
    "shot_index": 11,
    "task_identity": 101,
    "outcome_identity": 201,
    "ball_identity": 301,
}


def _observation_key(owner, **overrides):
    values = {**_OBSERVATION_KEY_VALUES, **overrides}
    return E.ActionEpochShotKey(
        **{
            name: torch.tensor(
                (value, -1), dtype=torch.int64, device=owner.device
            )
            for name, value in values.items()
        }
    )


def _seed_typed_observation_flight(
    owner,
    *,
    row=0,
    slot,
    key,
    publication=17,
    state=D.FLIGHT_INBOUND,
    action_epoch=True,
    contact=False,
    crossed=False,
    clear=False,
):
    for field in E.fields(E.ActionEpochShotKey):
        getattr(owner, "_flight_" + field.name)[row, slot] = getattr(
            key, field.name
        )[row]
    owner._flight_publication_ordinal[row, slot] = publication
    owner._flight_state[row, slot] = state
    owner._flight_action_epoch[row, slot] = action_epoch
    owner._flight_contact_valid[row, slot] = contact
    owner._flight_net_crossed[row, slot] = crossed
    owner._flight_net_clear[row, slot] = clear


def _project_current_flight(owner, key, *, publication=17):
    return owner.require_owned_action_epoch_current_flight_observation(
        owner.action_ball_full_mdp_observation_projection(),
        current_shot_key=key,
        current_publication_ordinal=torch.tensor(
            (publication, -1), dtype=torch.int64, device=owner.device
        ),
    )


def test_current_flight_projection_selects_full_key_collision_in_k2():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    current = _observation_key(owner)
    prior = _observation_key(
        owner,
        task_identity=901,
        outcome_identity=902,
        ball_identity=903,
    )
    _seed_typed_observation_flight(
        owner,
        slot=0,
        key=prior,
        state=D.FLIGHT_SETTLED_RETAINED,
        contact=True,
    )
    _seed_typed_observation_flight(
        owner,
        slot=1,
        key=current,
        state=D.FLIGHT_OPEN,
        contact=True,
        crossed=True,
        clear=True,
    )

    projected = _project_current_flight(owner, current)
    assert projected.flight_slot.tolist() == [1, -1]
    assert projected.contact_valid.tolist() == [True, False]
    assert projected.net_crossed.tolist() == [True, False]
    assert projected.net_clear.tolist() == [True, False]
    assert tuple(field.name for field in D.fields(type(projected))) == (
        "r06_owner",
        "publication_identity",
        "flight_slot",
        "contact_valid",
        "net_crossed",
        "net_clear",
    )
    for value, dtype in (
        (projected.flight_slot, torch.int64),
        (projected.contact_valid, torch.bool),
        (projected.net_crossed, torch.bool),
        (projected.net_clear, torch.bool),
    ):
        assert tuple(value.shape) == (2,)
        assert value.dtype == dtype
        assert value.device == owner.device
        assert value.is_contiguous()


@pytest.mark.parametrize(
    "identity_field", ("task_identity", "outcome_identity", "ball_identity")
)
def test_current_flight_projection_rejects_each_omitted_identity_collision(
    identity_field,
):
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    current = _observation_key(owner)
    stale = _observation_key(
        owner, **{identity_field: _OBSERVATION_KEY_VALUES[identity_field] + 1}
    )
    _seed_typed_observation_flight(
        owner,
        slot=0,
        key=stale,
        state=D.FLIGHT_OPEN,
        contact=True,
    )
    _seed_typed_observation_flight(
        owner,
        slot=1,
        key=current,
        state=D.FLIGHT_OPEN,
        contact=True,
        crossed=True,
        clear=True,
    )

    projected = _project_current_flight(owner, current)
    assert projected.flight_slot.tolist() == [1, -1]
    assert projected.contact_valid.tolist() == [True, False]
    assert projected.net_crossed.tolist() == [True, False]
    assert projected.net_clear.tolist() == [True, False]


def test_current_flight_projection_rejects_publication_mismatch():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    key = _observation_key(owner)
    _seed_typed_observation_flight(owner, slot=0, key=key, publication=17)

    projected = _project_current_flight(owner, key, publication=18)
    assert projected.flight_slot.tolist() == [-1, -1]
    assert not projected.contact_valid.any()
    assert not projected.net_crossed.any()
    assert not projected.net_clear.any()


@pytest.mark.parametrize(
    ("key_overrides", "publication"),
    (({"action_uid": 0}, 17), ({}, -1)),
)
def test_current_flight_projection_rejects_invalid_key_or_publication(
    key_overrides, publication
):
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    key = _observation_key(owner, **key_overrides)
    _seed_typed_observation_flight(
        owner,
        slot=0,
        key=key,
        publication=publication,
        state=D.FLIGHT_OPEN,
        contact=True,
        crossed=True,
        clear=True,
    )

    projected = _project_current_flight(
        owner, key, publication=publication
    )
    assert projected.flight_slot.tolist() == [-1, -1]
    assert not projected.contact_valid.any()
    assert not projected.net_crossed.any()
    assert not projected.net_clear.any()


def test_current_flight_duplicate_is_named_neutral_and_keeps_healthy_peer():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    key = _row_key(owner)
    _seed_typed_observation_flight(owner, row=0, slot=0, key=key, publication=17)
    _seed_typed_observation_flight(owner, row=0, slot=1, key=key, publication=17)
    _seed_typed_observation_flight(
        owner,
        row=1,
        slot=0,
        key=key,
        publication=18,
        contact=True,
        crossed=True,
        clear=True,
    )

    projected = owner.require_owned_action_epoch_current_flight_observation(
        owner.action_ball_full_mdp_observation_projection(),
        current_shot_key=key,
        current_publication_ordinal=torch.tensor(
            (17, 18), dtype=torch.int64, device=owner.device
        ),
    )

    _assert_row_fault_words(epoch, E.ROW_FAULT_R06_CURRENT_FLIGHT_DUPLICATE)
    assert projected.flight_slot.tolist() == [-1, 0]
    assert projected.contact_valid.tolist() == [False, True]
    assert projected.net_crossed.tolist() == [False, True]
    assert projected.net_clear.tolist() == [False, True]


@pytest.mark.parametrize(
    ("action_epoch", "state"),
    (
        (False, D.FLIGHT_OPEN),
        (True, D.FLIGHT_EMPTY),
        (True, D.FLIGHT_SETTLED_RETAINED),
    ),
)
def test_current_flight_projection_neutralizes_stale_or_nonlive_typed_rows(
    action_epoch, state
):
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    key = _observation_key(owner)
    _seed_typed_observation_flight(
        owner,
        slot=0,
        key=key,
        state=state,
        action_epoch=action_epoch,
        contact=True,
        crossed=True,
        clear=True,
    )

    projected = _project_current_flight(owner, key)
    assert projected.flight_slot.tolist() == [-1, -1]
    assert not projected.contact_valid.any()
    assert not projected.net_crossed.any()
    assert not projected.net_clear.any()


def test_current_flight_projection_rejects_foreign_handle_and_publication_abi():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    key = _observation_key(owner)
    with pytest.raises(D.LandingOutcomeDeviceError, match="forged or foreign"):
        owner.require_owned_action_epoch_current_flight_observation(
            object(),
            current_shot_key=key,
            current_publication_ordinal=torch.tensor((17, -1)),
        )
    with pytest.raises(D.LandingOutcomeDeviceError, match="ABI differs"):
        owner.require_owned_action_epoch_current_flight_observation(
            owner.action_ball_full_mdp_observation_projection(),
            current_shot_key=key,
            current_publication_ordinal=torch.tensor((17, -1), dtype=torch.int32),
        )


def _launch_selected_reset_clock_tape_into_real_r06():
    """Launch local Motion 7/8/9 into global Physical/R06 99/100/101."""

    import test_action_ball_physical_epoch_hot_lane as hot

    device = torch.device("cpu")
    physical_owner, _scene = hot._physical_owner(device, num_envs=2)
    r06_owner = T._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=2,
        bind_physical_park=False,
        device=device,
    )
    epoch_owner, _ = _bound_epoch(r06_owner)
    physical_owner._action_epoch_owner = epoch_owner
    physical_owner.bind_r06_owner(r06_owner)
    epoch_owner._physical_launch_projection = (
        physical_owner.action_epoch_r06_launch_projection
    )

    key_grid = hot._shot_key(num_envs=2, device=device, width=1)
    row_key = E.ActionEpochShotKey(
        **{
            field.name: getattr(key_grid, field.name)[:, 0]
            for field in E.fields(E.ActionEpochShotKey)
        }
    )
    publication = torch.tensor((17, 18), dtype=torch.int64, device=device)
    record = epoch_owner._publication.current
    assert record is not None
    record.current_task_slot.zero_()
    record.phase[:, 0].fill_(E.PHASE_REVEAL_COMMITTED)
    record.publication_ordinal[:, 0].copy_(publication)
    for field in E.fields(E.ActionEpochShotKey):
        getattr(record.identity.shot_key, field.name)[:, 0].copy_(
            getattr(row_key, field.name)
        )

    motion_owner = hot._MotionLaunchStub(physical_owner, row_key)
    # Row 0 was selected for reset and reached local launch tick 7.  Row 1 was
    # not selected; its continuing episode remains at global-aligned tick 99.
    motion_owner.control_tick.copy_(
        torch.tensor((7, 99), dtype=torch.int64, device=device)
    )
    physical_owner._action_epoch_motion_owner = motion_owner
    pending = hot._pending(
        physical_owner,
        key=row_key,
        pending=torch.ones(2, dtype=torch.bool, device=device),
        ordinal=publication,
        flight_slot=torch.zeros(2, dtype=torch.int64, device=device),
    )
    # The unselected peer has its own later launch tape and must not be pulled
    # into the selected row's translated 99/100/101 R06 chronology.
    pending.launch_motion_tick[1] = 100
    pending.contact_deadline_motion_tick[1] = 101
    pending.crossing_horizon_motion_tick[1] = 102
    pending.target_xy_m[:, 0].fill_(
        (
            r06_owner.profile.opponent_table_x_min_m
            + r06_owner.profile.opponent_table_x_max_m
        )
        / 2.0
    )
    pending.target_xy_m[:, 1].fill_(
        (r06_owner.profile.table_y_min_m + r06_owner.profile.table_y_max_m)
        / 2.0
    )
    physical_owner._action_epoch_pending_launch = pending

    # The last completed global boundary is 99; the next Physical control is
    # 100.  Physical must translate retained per-row Motion clocks exactly once.
    physical_owner._last_postphysics_exact_stamp = (99, 0, 1, 99, 1)
    physical_owner.launch_action_epoch()

    assert physical_owner._action_epoch_pending_launch.pending.tolist() == [False, True]
    assert r06_owner._flight_state[:, 0].tolist() == [D.FLIGHT_INBOUND, D.FLIGHT_EMPTY]
    assert r06_owner._flight_reveal_control_step[:, 0].tolist() == [99, -1]
    assert r06_owner._flight_contact_deadline_control_step[:, 0].tolist() == [100, -1]
    assert r06_owner._flight_crossing_horizon_control_step[:, 0].tolist() == [101, -1]
    assert not r06_owner._mailbox_reserved[1].any()
    assert epoch_owner._undrained_row_fault_bits.tolist() == [0, 0]
    return physical_owner, r06_owner, epoch_owner, row_key, publication


def _real_physical_r06_packet(
    *,
    physical_owner,
    epoch_owner,
    row_key,
    publication,
    control_step,
    observation_ordinal,
    physics_substep=0,
    contact=False,
    crossing=False,
    previous_center=None,
):
    import test_action_ball_physical_epoch_hot_lane as hot

    device = physical_owner.device
    shape = (physical_owner.num_envs, physical_owner.flight_capacity)
    observe = torch.zeros(shape, dtype=torch.bool, device=device)
    observe[0, 0] = True
    key = E.ActionEpochShotKey(
        **{
            field.name: torch.full(
                shape, -1, dtype=torch.int64, device=device
            )
            for field in E.fields(E.ActionEpochShotKey)
        }
    )
    for field in E.fields(E.ActionEpochShotKey):
        getattr(key, field.name)[0, 0] = getattr(row_key, field.name)[0]
    publication_grid = torch.full(
        shape, -1, dtype=torch.int64, device=device
    )
    publication_grid[0, 0] = publication[0]
    ordinal_grid = torch.full(shape, -1, dtype=torch.int64, device=device)
    ordinal_grid[0, 0] = observation_ordinal

    target_xy = torch.tensor(
        (
            (
                physical_owner._r06_owner.profile.opponent_table_x_min_m
                + physical_owner._r06_owner.profile.opponent_table_x_max_m
            )
            / 2.0,
            (
                physical_owner._r06_owner.profile.table_y_min_m
                + physical_owner._r06_owner.profile.table_y_max_m
            )
            / 2.0,
        ),
        dtype=torch.float32,
        device=device,
    )
    plane = physical_owner._r06_owner.profile.ball_center_landing_plane_z_m
    high_center = torch.tensor(
        (target_xy[0], target_xy[1], plane + 0.2),
        dtype=torch.float32,
        device=device,
    )
    low_center = torch.tensor(
        (target_xy[0], target_xy[1], plane - 0.2),
        dtype=torch.float32,
        device=device,
    )
    previous = torch.zeros(shape + (3,), dtype=torch.float32, device=device)
    current = torch.zeros_like(previous)
    previous[0, 0] = high_center if previous_center is None else previous_center
    current[0, 0] = low_center if crossing else high_center

    def stamp(*, active, phase):
        control = torch.full(shape, -1, dtype=torch.int64, device=device)
        substep = torch.full(shape, -1, dtype=torch.int32, device=device)
        event_phase = torch.full(shape, -1, dtype=torch.int8, device=device)
        if active:
            control[0, 0] = control_step
            substep[0, 0] = physics_substep
            event_phase[0, 0] = phase
        return hot.physical.PhysicsStampGrid(
            control_step=control,
            physics_substep=substep,
            event_phase=event_phase,
        )

    event = torch.zeros(shape, dtype=torch.bool, device=device)
    event[0, 0] = contact
    crossing_event = torch.zeros(shape, dtype=torch.bool, device=device)
    crossing_event[0, 0] = crossing
    report = torch.zeros(shape, dtype=torch.bool, device=device)
    # After first contact, Physical must continue to report the crossing
    # channel even when this observation contains no crossing event.
    report[0, 0] = contact or observation_ordinal > 0
    contact_center = torch.zeros_like(previous)
    contact_center[0, 0] = high_center
    crossing_xy = torch.zeros(shape + (2,), dtype=torch.float32, device=device)
    crossing_xy[0, 0] = target_xy
    source_step = torch.full(shape, -1, dtype=torch.int64, device=device)
    source_step[0, 0] = control_step
    return hot.physical.ActionEpochR06PostPhysicsProjection(
        observe_mask=observe,
        flight_slot=physical_owner._fixed_flight_slot_grid.detach().clone(),
        shot_key=key,
        publication_ordinal=publication_grid,
        observation_ordinal=ordinal_grid,
        previous_ball_center_m=previous,
        current_ball_center_m=current,
        observation_stamp=stamp(active=True, phase=D.PHASE_LANDING),
        selected_contact_event=event,
        selected_contact_ball_center_m=contact_center,
        selected_contact_outgoing_segment_anchor_m=contact_center.clone(),
        selected_contact_stamp=stamp(active=contact, phase=D.PHASE_CONTACT),
        net_crossing_event=torch.zeros(shape, dtype=torch.bool, device=device),
        net_clear_at_crossing=torch.zeros(shape, dtype=torch.bool, device=device),
        net_crossing_stamp=stamp(active=False, phase=D.PHASE_NET),
        crossing_report_delivered=report,
        first_descending_crossing_event=crossing_event,
        first_descending_crossing_xy_m=crossing_xy,
        first_descending_crossing_stamp=stamp(
            active=crossing, phase=D.PHASE_LANDING
        ),
        nonfinite_observation=torch.zeros(shape, dtype=torch.bool, device=device),
        producer_contract_fault=torch.zeros(shape, dtype=torch.bool, device=device),
        engine_overflow=torch.zeros(shape, dtype=torch.bool, device=device),
        owner_fault_bits=torch.zeros(shape, dtype=torch.int64, device=device),
        fact_valid_bits=torch.zeros(shape, dtype=torch.int64, device=device),
        fact_source_step=source_step,
        fact_f32=torch.zeros(
            shape + (hot.physical.PHYSICAL_EPOCH_FACT_F32_WIDTH,),
            dtype=torch.float32,
            device=device,
        ),
        physical_owner=physical_owner,
        epoch_owner=epoch_owner,
        _owner_identity=physical_owner._owner_identity,
        _token=hot.physical._ACTION_EPOCH_R06_POSTPHYSICS_TOKEN,
    )


@pytest.mark.parametrize(
    ("mode", "expected_cause"),
    (
        ("no_contact_deadline", D.SETTLEMENT_CAUSE_CONTACT_DEADLINE),
        ("contact_no_cross_horizon", D.SETTLEMENT_CAUSE_CROSSING_HORIZON),
        ("early_crossing", D.SETTLEMENT_CAUSE_FIRST_CROSSING),
    ),
)
def test_selected_reset_local_clock_maps_once_across_real_physical_r06_owners(
    mode, expected_cause
):
    (
        physical_owner,
        r06_owner,
        epoch_owner,
        row_key,
        publication,
    ) = _launch_selected_reset_clock_tape_into_real_r06()

    contact = mode != "no_contact_deadline"
    first = _real_physical_r06_packet(
        physical_owner=physical_owner,
        epoch_owner=epoch_owner,
        row_key=row_key,
        publication=publication,
        control_step=100,
        observation_ordinal=0,
        contact=contact,
        crossing=mode == "early_crossing",
    )
    physical_owner._action_epoch_active_r06_postphysics = first
    try:
        result = r06_owner.publish_action_ball_full_mdp_epoch_post_physics()
        retire = r06_owner.retire_action_ball_full_mdp_epoch_post_physics()
    finally:
        physical_owner._action_epoch_active_r06_postphysics = None
    r06_owner.publish_action_ball_full_mdp_epoch_facts()

    if mode == "contact_no_cross_horizon":
        assert result.accepted[0, 0]
        assert not result.settled_mask.any()
        assert result.new_valid_contact_mask[0, 0]
        assert not retire.retired_mask.any()
        second = _real_physical_r06_packet(
            physical_owner=physical_owner,
            epoch_owner=epoch_owner,
            row_key=row_key,
            publication=publication,
            control_step=101,
            observation_ordinal=1,
            contact=False,
            crossing=False,
            previous_center=first.current_ball_center_m[0, 0],
        )
        physical_owner._action_epoch_active_r06_postphysics = second
        try:
            result = r06_owner.publish_action_ball_full_mdp_epoch_post_physics()
            retire = r06_owner.retire_action_ball_full_mdp_epoch_post_physics()
        finally:
            physical_owner._action_epoch_active_r06_postphysics = None
        r06_owner.publish_action_ball_full_mdp_epoch_facts()

    assert result.accepted.tolist() == [[True, False], [False, False]]
    assert not result.rejected.any()
    assert result.fault_bits.tolist() == [[0, 0], [0, 0]]
    assert result.settled_mask.tolist() == [[True, False], [False, False]]
    assert int(result.settlement_cause[0, 0]) == expected_cause
    assert int(result.settlement_cause[0, 0]) != D.SETTLEMENT_CAUSE_PROTOCOL_FAULT
    assert retire.retired_mask.tolist() == [[True, False], [False, False]]

    # The asynchronous peer was neither compacted into row 0's slot nor
    # observed/settled under row 0's global chronology.
    assert physical_owner._action_epoch_pending_launch.pending.tolist() == [False, True]
    assert r06_owner._flight_state[1, 0] == D.FLIGHT_EMPTY
    assert not r06_owner._mailbox_reserved[1].any()
    assert epoch_owner._undrained_row_fault_bits.tolist() == [0, 0]


def _publish_contact_without_settlement():
    physical_owner, r06_owner, epoch_owner, row_key, publication = (
        _launch_selected_reset_clock_tape_into_real_r06()
    )
    packet = _real_physical_r06_packet(
        physical_owner=physical_owner,
        epoch_owner=epoch_owner,
        row_key=row_key,
        publication=publication,
        control_step=100,
        observation_ordinal=0,
        contact=True,
        crossing=False,
    )
    physical_owner._action_epoch_active_r06_postphysics = packet
    try:
        result = r06_owner.publish_action_ball_full_mdp_epoch_post_physics()
        retire = r06_owner.retire_action_ball_full_mdp_epoch_post_physics()
    finally:
        physical_owner._action_epoch_active_r06_postphysics = None
    assert not result.settled_mask.any()
    assert not retire.retired_mask.any()
    return physical_owner, r06_owner, epoch_owner, row_key, publication, packet


def test_control_window_samples_early_crossing_before_one_mailbox_finalize():
    physical, r06_owner, epoch_owner, row_key, publication = (
        _launch_selected_reset_clock_tape_into_real_r06()
    )
    contact = _real_physical_r06_packet(
        physical_owner=physical,
        epoch_owner=epoch_owner,
        row_key=row_key,
        publication=publication,
        control_step=100,
        physics_substep=0,
        observation_ordinal=0,
        contact=True,
        crossing=False,
    )
    physical._action_epoch_active_r06_postphysics = contact
    try:
        first = r06_owner.sample_action_ball_full_mdp_epoch_post_physics()
    finally:
        physical._action_epoch_active_r06_postphysics = None
    assert first.accepted[0, 0]
    assert not first.settled_mask.any()

    crossing = _real_physical_r06_packet(
        physical_owner=physical,
        epoch_owner=epoch_owner,
        row_key=row_key,
        publication=publication,
        control_step=100,
        physics_substep=1,
        observation_ordinal=1,
        contact=False,
        crossing=True,
        previous_center=contact.current_ball_center_m[0, 0],
    )
    physical._action_epoch_active_r06_postphysics = crossing
    try:
        second = r06_owner.sample_action_ball_full_mdp_epoch_post_physics()
    finally:
        physical._action_epoch_active_r06_postphysics = None
    assert second.settled_mask.tolist() == [[True, False], [False, False]]
    assert not r06_owner._mailbox_history_valid.any()
    assert r06_owner._flight_state[0, 0] == D.FLIGHT_OPEN
    with pytest.raises(D.LandingOutcomeDeviceError, match="lifetime differs"):
        r06_owner.finalize_action_ball_full_mdp_epoch_post_physics_control(
            physics_substeps_per_control=4
        )
    with pytest.raises(D.LandingOutcomeDeviceError, match="control window"):
        r06_owner.project_current_action_epoch_outcome_rows()

    for physics_substep in (2, 3):
        inactive = _real_physical_r06_packet(
            physical_owner=physical,
            epoch_owner=epoch_owner,
            row_key=row_key,
            publication=publication,
            control_step=100,
            physics_substep=physics_substep,
            observation_ordinal=-1,
        )
        inactive.observe_mask.zero_()
        physical._action_epoch_active_r06_postphysics = inactive
        try:
            sampled = r06_owner.sample_action_ball_full_mdp_epoch_post_physics()
        finally:
            physical._action_epoch_active_r06_postphysics = None
        assert not sampled.settled_mask.any()

    finalized = r06_owner.finalize_action_ball_full_mdp_epoch_post_physics_control(
        physics_substeps_per_control=4
    )
    assert finalized.settled_mask.tolist() == [[True, False], [False, False]]
    assert r06_owner._mailbox_history_valid[0].any()
    retired = r06_owner.retire_action_ball_full_mdp_epoch_post_physics()
    assert retired.retired_mask.tolist() == [[True, False], [False, False]]

    before = len(epoch_owner._publication.pending_log)
    for substep in range(4):
        r06_owner.publish_action_ball_full_mdp_epoch_control_substep_facts(
            substep_index=substep
        )
    entries = epoch_owner._publication.pending_log[before:]
    assert [entry.transition for entry in entries] == ["R06_OUTCOME_ROWS"] * 4
    masks = [dict(zip(entry.delta.names, entry.delta.values))["event_mask"] for entry in entries]
    assert [bool(mask.any()) for mask in masks] == [False, True, False, False]
    assert r06_owner._action_epoch_control_replay is None


def test_current_settlement_zero_event_does_not_pull_retained_projector():
    _physical, r06_owner, epoch_owner, _key, _publication, _packet = (
        _publish_contact_without_settlement()
    )
    pending = r06_owner._pending_action_epoch_current_settlement_delta
    assert type(pending) is D.ActionEpochR06CurrentSettlementDelta
    assert not pending.rows.valid.any()
    mailbox_before = r06_owner._mailbox_history_valid.detach().clone()

    def forbidden_retained_scan():
        raise AssertionError("hot current-settlement publish rescanned mailbox history")

    epoch_owner._r06_outcome_projection = forbidden_retained_scan
    journal_size = len(epoch_owner._publication.pending_log)
    r06_owner.publish_action_ball_full_mdp_epoch_facts()

    assert r06_owner._pending_action_epoch_current_settlement_delta is None
    assert torch.equal(r06_owner._mailbox_history_valid, mailbox_before)
    assert len(epoch_owner._publication.pending_log) == journal_size + 1
    entry = epoch_owner._publication.pending_log[-1]
    assert entry.transition == "R06_OUTCOME_ROWS"
    delta = dict(zip(entry.delta.names, entry.delta.values))
    assert not delta["event_mask"].any()
    assert not delta["owner_fault_bits"].any()
    assert not delta["predicate_bits"].any()


def test_current_settlement_delta_blocks_overwrite_and_replay():
    physical, r06_owner, epoch_owner, row_key, publication, first = (
        _publish_contact_without_settlement()
    )
    pending = r06_owner._pending_action_epoch_current_settlement_delta
    second = _real_physical_r06_packet(
        physical_owner=physical,
        epoch_owner=epoch_owner,
        row_key=row_key,
        publication=publication,
        control_step=101,
        observation_ordinal=1,
        previous_center=first.current_ball_center_m[0, 0],
    )
    physical._action_epoch_active_r06_postphysics = second
    try:
        with pytest.raises(D.LandingOutcomeDeviceError, match="unconsumed ActionEpoch"):
            r06_owner.publish_action_ball_full_mdp_epoch_post_physics()
    finally:
        physical._action_epoch_active_r06_postphysics = None

    epoch_owner.refresh_r06_outcome_rows(pending)
    assert r06_owner._pending_action_epoch_current_settlement_delta is None
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale, foreign, replayed"):
        epoch_owner.refresh_r06_outcome_rows(pending)


def test_current_settlement_duplicate_newest_is_named_and_neutral():
    owner = T._coordinator(rows=2, flight_slots=2, mailbox_slots=2)
    epoch, _physical = _bound_epoch(owner)
    settled = torch.tensor(
        ((True, True), (False, False)), dtype=torch.bool, device=owner.device
    )
    prepared = D._ActionEpochOutcomeCandidateGrid(
        candidate=settled,
        shot_key_values=torch.zeros(
            (2, 2, len(E.fields(E.ActionEpochShotKey))),
            dtype=torch.int64,
            device=owner.device,
        ),
        publication_ordinal=torch.tensor(
            ((7, 7), (-1, -1)), dtype=torch.int64, device=owner.device
        ),
        settlement_step=torch.zeros((2, 2), dtype=torch.int64, device=owner.device),
        policy_eligible=torch.zeros((2, 2), dtype=torch.bool, device=owner.device),
        fact_values=torch.zeros(
            (2, 2, D.R06_ACTION_EPOCH_FACT_F32_WIDTH),
            dtype=torch.float32,
            device=owner.device,
        ),
        outcome_code=torch.zeros((2, 2), dtype=torch.int64, device=owner.device),
        owner_fault_bits=torch.zeros((2, 2), dtype=torch.int64, device=owner.device),
    )
    owner._prepared_action_epoch_current_settlement_delta = prepared
    owner._action_epoch_post_physics_settled_mask = settled

    delta = owner._mint_action_epoch_current_settlement_delta(settled)

    assert not delta.rows.valid.any()
    assert delta.rows.publication_ordinal.tolist() == [-1, -1]
    _assert_row_fault_words(
        epoch, E.ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE
    )
    assert owner.require_owned_action_epoch_current_settlement_delta(
        delta, expected_epoch_owner=epoch
    ) is delta
