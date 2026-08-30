"""Focused construction tests for the opt-in frozen-teacher diagnostic."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
import types

import numpy as np
import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_full_mdp_teacher_replay as replay
import mujoco_gpu_ac_full_mdp_initial_wait_env as wait_env
import mujoco_gpu_ac_full_mdp_wait_rsl3 as runner
import mujoco_gpu_ac_table_keepout as keepout


def test_teacher_replay_wait_bridge_and_affine_decoder_are_exact():
    hold = torch.tensor([[1.0, 2.0]])
    frame0 = torch.tensor([[5.0, 10.0]])
    previous = hold.clone()
    commands = []
    for valid, frozen in ((False, 4), (True, 3), (True, 2), (True, 1), (True, 0)):
        previous = replay.frozen_teacher_qdes(
            torch=torch,
            task_valid=torch.tensor([valid]),
            hold_qdes=hold,
            previous_qdes=previous,
            teacher_qdes=frame0,
            frozen_steps=torch.tensor([frozen]),
            bridge=wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge,
        )
        commands.append(previous.clone())
    assert torch.equal(commands[0], hold)
    assert torch.equal(commands[-1], frame0)
    offset = torch.tensor([0.5, -1.0])
    scale = torch.tensor([0.5, 2.0])
    action = replay.decode_teacher_qdes_to_action(
        torch=torch, qdes=frame0, action_offset=offset, action_scale=scale
    )
    torch.testing.assert_close(offset + scale * action[0], frame0[0])


def test_teacher_replay_frozen_counter_and_hash_are_content_exact():
    frozen = replay.remaining_teacher_frozen_steps(
        torch=torch,
        common_step=12,
        reveal_tick=torch.tensor([10]),
        pre_swing_wait_s=torch.tensor([0.10]),
        step_dt=0.02,
    )
    assert frozen.tolist() == [3]
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    expected = hashlib.sha256(
        np.asarray([[1.0, 2.0]], dtype="<f4").tobytes()
    ).hexdigest()
    assert replay.tensor_f32_sha256(value) == expected


@pytest.mark.parametrize(
    ("dtype", "encoded_seconds", "expected"),
    (
        (torch.float32, 0.10, [3, 3, 4]),
        (torch.float64, 0.10, [3, 3, 4]),
        (torch.float32, 0.06, [1, 1, 2]),
        (torch.float64, 0.06, [1, 1, 2]),
    ),
)
def test_teacher_replay_frozen_counter_only_snaps_dtype_roundoff_at_tick_boundary(
    dtype, encoded_seconds, expected
):
    encoded_boundary = torch.tensor([encoded_seconds], dtype=dtype)
    below_boundary = torch.nextafter(
        encoded_boundary, torch.full_like(encoded_boundary, -torch.inf)
    )
    above_boundary = torch.nextafter(
        encoded_boundary, torch.full_like(encoded_boundary, torch.inf)
    )
    frozen = replay.remaining_teacher_frozen_steps(
        torch=torch,
        common_step=12,
        reveal_tick=torch.tensor([10, 10, 10]),
        pre_swing_wait_s=torch.cat(
            (below_boundary, encoded_boundary, above_boundary)
        ),
        step_dt=0.02,
    )
    # The encoded schedule and its lower neighbour both end at the intended
    # integer tick.  One representable float above it is a real positive margin
    # and must keep the following tick instead of being swallowed by boundary
    # normalization.
    assert frozen.tolist() == expected


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_teacher_replay_frozen_counter_strictly_ceils_non_boundary_wait(dtype):
    frozen = replay.remaining_teacher_frozen_steps(
        torch=torch,
        common_step=12,
        reveal_tick=torch.tensor([10]),
        pre_swing_wait_s=torch.tensor([0.061], dtype=dtype),
        step_dt=0.02,
    )
    assert frozen.tolist() == [2]


def test_teacher_replay_decimal_timebase_encodes_three_fiftieths_as_point_zero_six():
    numerator, denominator = replay._decimal_policy_timebase_ratio(0.02)
    assert (numerator, denominator) == (1, 50)
    literal_boundary = torch.tensor([0.06], dtype=torch.float64)
    upper_neighbour = torch.nextafter(
        literal_boundary, torch.full_like(literal_boundary, torch.inf)
    )
    canonical_boundary = (
        torch.tensor([3.0], dtype=torch.float64)
        * float(numerator)
        / float(denominator)
    )
    assert torch.equal(canonical_boundary, literal_boundary)
    assert not torch.equal(canonical_boundary, upper_neighbour)


def test_pre_step_snapshot_drives_request_while_post_step_is_independent():
    env = types.SimpleNamespace(
        _epoch_task_valid=torch.tensor([True]),
        _full_a_teacher_frame=torch.tensor([4]),
        _full_a_motion_phase_code=torch.tensor([1]),
        _full_a_teacher_joint_pos=torch.tensor([[2.0, 4.0]]),
        _epoch_clock_ticks=torch.tensor([[10, 11, 12, 13, 14]]),
        _full_a_pre_swing_wait_s=torch.tensor([0.0]),
    )
    pre = replay.capture_teacher_replay_pre_step(env)
    env._epoch_task_valid.fill_(False)
    env._full_a_teacher_frame.fill_(9)
    env._full_a_motion_phase_code.fill_(4)
    env._full_a_teacher_joint_pos.fill_(99.0)
    requested = replay.frozen_teacher_qdes(
        torch=torch,
        task_valid=pre.task_valid,
        hold_qdes=torch.zeros((1, 2)),
        previous_qdes=torch.zeros((1, 2)),
        teacher_qdes=pre.teacher_qdes,
        frozen_steps=torch.zeros(1, dtype=torch.long),
        bridge=wait_env.portable_question.step_diagnostic_split_ready_qdes_bridge,
    )
    assert pre.task_valid.tolist() == [True]
    assert pre.teacher_frame.tolist() == [4]
    assert pre.motion_phase.tolist() == [1]
    assert requested.tolist() == [[2.0, 4.0]]
    assert env._epoch_task_valid.tolist() == [False]
    assert env._full_a_teacher_frame.tolist() == [9]


def test_contact_patch_boundary_does_not_alias_final_forward_to_substep():
    substep = replay.contact_capture_boundary(
        transition_start_step=17,
        capture_boundary="physics_substep_poststate",
        physics_substep_index=19,
        decimation=20,
    )
    final = replay.contact_capture_boundary(
        transition_start_step=17,
        capture_boundary="post_forward_final",
        physics_substep_index=None,
        decimation=20,
    )
    assert substep["completed_physics_substeps"] == 19
    assert final["completed_physics_substeps"] == 20
    assert substep["physics_substep_index"] == 19
    assert final["physics_substep_index"] is None
    with pytest.raises(ValueError, match="physics substep"):
        replay.contact_capture_boundary(
            transition_start_step=17,
            capture_boundary="physics_substep_poststate",
            physics_substep_index=0,
            decimation=20,
        )
    with pytest.raises(ValueError, match="final-forward"):
        replay.contact_capture_boundary(
            transition_start_step=17,
            capture_boundary="post_forward_final",
            physics_substep_index=20,
            decimation=20,
        )


def test_different_shot_generation_is_not_a_valid_patch_key():
    question = torch.tensor([1.0, 2.0])
    question_sha = replay.tensor_f32_sha256(question)
    with pytest.raises(RuntimeError, match="different shot"):
        replay.validate_contact_patch_shot(
            contact_patch={
                "present": True,
                "reset_generation": 4,
                "question_f32_sha256": question_sha,
            },
            question_reset_generation=3,
            question_f32_sha256=question_sha,
        )


def test_contact_patch_is_absent_until_explicitly_enabled():
    bare = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="not enabled"):
        wait_env.FullMdpInitialWaitVecEnv.diagnostic_first_generic_contact_patch(
            bare
        )
    source = inspect.getsource(
        wait_env.FullMdpInitialWaitVecEnv._full_a_latch_ball_contacts
    )
    tree = ast.parse(textwrap.dedent(source))
    getattr_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "_diagnostic_contact_patch_consumer"
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value is None
    ]
    assert len(getattr_calls) == 1
    guarded_calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "patch_consumer"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.IsNot)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
        ):
            continue
        guarded_calls.extend(
            child
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "patch_consumer"
        )
    assert len(guarded_calls) == 1


def test_contact_patch_bridge_distinguishes_torch_proxy_warp_and_device_duck():
    calls = []

    class FakeWarpArray:
        pass

    class FakeWarp:
        array = FakeWarpArray

        @staticmethod
        def to_torch(value):
            calls.append(value)
            return torch.tensor([3.0])

    class FakeTorchArray:
        def __init__(self, tensor):
            self.tensor = tensor
            self.device = tensor.device
            self.indexes = []

        def __getitem__(self, index):
            self.indexes.append(index)
            return self.tensor[index]

    direct = torch.tensor([1.0])
    proxy_view = torch.tensor([2.0])
    proxy = FakeTorchArray(proxy_view)
    warp_value = FakeWarpArray()
    bridge = wait_env.FullMdpInitialWaitVecEnv._diagnostic_contact_field_torch_view

    assert bridge(
        torch=torch,
        warp=FakeWarp,
        torch_array_type=FakeTorchArray,
        value=direct,
    ) is direct
    bridged_proxy = bridge(
        torch=torch,
        warp=FakeWarp,
        torch_array_type=FakeTorchArray,
        value=proxy,
    )
    assert torch.equal(bridged_proxy, proxy_view)
    assert bridged_proxy.data_ptr() == proxy_view.data_ptr()
    assert proxy.indexes == [Ellipsis]
    bridged_warp = bridge(
        torch=torch,
        warp=FakeWarp,
        torch_array_type=FakeTorchArray,
        value=warp_value,
    )
    assert torch.equal(bridged_warp, torch.tensor([3.0]))
    assert calls == [warp_value]

    device_only = types.SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(TypeError, match="no supported Torch view"):
        bridge(
            torch=torch,
            warp=FakeWarp,
            torch_array_type=FakeTorchArray,
            value=device_only,
        )


def _table_attribution_rig():
    return types.SimpleNamespace(
        _torch=torch,
        decimation=20,
        _diagnostic_contact_patch_transition_start_step=41,
        _diagnostic_first_table_terminal_source=None,
        _diagnostic_first_resolved_substep=None,
        _diagnostic_table_tick_first_positive=None,
        _diagnostic_table_tick_keepout=False,
        _diagnostic_table_tick_final_resolved=False,
        _diagnostic_table_tick_resolved_any_substep=False,
        _diagnostic_table_attribution_consumer=True,
    )


def _keepout_witness(*, component_id="torso_box"):
    return {
        "schema": "action_ball_keepout_first_witness_v1",
        "selection": "first_production_order_overlap",
        "component_index": 7,
        "component_id": component_id,
        "component_kind": "body_proxy",
        "owner_body_index": 9,
        "owner_body_name": "torso_Link",
        "table_index": 0,
        "table_role": "top",
        "sat_signed_margin_m": -0.003,
        "root_position_env_m": [0.0, 0.0, 1.0],
        "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "owner_position_env_m": [0.2, 0.0, 1.1],
        "owner_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "plant_identity": {
            "root_mjcf_sha256": "a" * 64,
            "identity_manifest_sha256": "b" * 64,
            "portable_identity_sha256": "c" * 64,
            "verification_receipt_sha256": "d" * 64,
            "owner_local_frame_sha256": "e" * 64,
        },
    }


def test_keepout_witness_winner_is_component_major_and_unknown_fails_closed():
    exact = torch.zeros((63, 5), dtype=torch.bool)
    exact[12, 0] = True
    exact[3, 4] = True
    assert keepout._host_test_first_overlap_index(exact) == (3, 4)
    with pytest.raises(RuntimeError, match="no SAT witness"):
        keepout._host_test_first_overlap_index(torch.zeros_like(exact))


def test_table_attribution_freezes_first_positive_before_reset_once_only():
    env = _table_attribution_rig()
    capture = wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution
    finalize = wait_env.FullMdpInitialWaitVecEnv.diagnostic_table_attribution_tick
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=1,
        capture_boundary="physics_substep_poststate",
    )
    capture(
        env,
        keepout=torch.tensor([True]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=2,
        capture_boundary="physics_substep_poststate",
        keepout_witness=_keepout_witness(),
    )
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([True]),
        resolved_is_final=False,
        substep_index=3,
        capture_boundary="physics_substep_poststate",
    )
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([False]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
    )
    result = finalize(
        env,
        torch.tensor([
            wait_env.FULLMDP_TERMINATION_BITS["robot_hit_table"]
        ]),
    )
    assert result["keepout_source"] is True
    assert result["backend_resolved_table_contact"] is False
    assert result["resolved_any_substep"] is True
    assert result["first_resolved_substep"]["physics_substep_index"] == 3
    assert result["first_table_terminal_source"] == {
        "transition_start_step": 41,
        "capture_boundary": "physics_substep_poststate",
        "physics_substep_index": 2,
        "completed_physics_substeps": 2,
        "keepout_source": True,
        "backend_resolved_table_contact": False,
        "keepout_witness": _keepout_witness(),
    }

    wait_env.FullMdpInitialWaitVecEnv._begin_diagnostic_table_attribution_tick(env)
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([True]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
        keepout_witness=_keepout_witness() if keepout else None,
    )
    second = finalize(
        env,
        torch.tensor([
            wait_env.FULLMDP_TERMINATION_BITS["robot_hit_table"]
        ]),
    )
    assert second["first_table_terminal_source"] == result[
        "first_table_terminal_source"
    ]


@pytest.mark.parametrize(
    ("keepout", "resolved", "terminal"),
    ((False, False, True), (True, False, False), (False, True, False)),
)
def test_table_attribution_unknown_or_unmatched_source_fails_closed(
    keepout, resolved, terminal
):
    env = _table_attribution_rig()
    wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution(
        env,
        keepout=torch.tensor([keepout]),
        resolved=torch.tensor([resolved]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
    )
    bits = (
        wait_env.FULLMDP_TERMINATION_BITS["robot_hit_table"] if terminal else 0
    )
    with pytest.raises(RuntimeError, match="source is unknown"):
        wait_env.FullMdpInitialWaitVecEnv.diagnostic_table_attribution_tick(
            env, torch.tensor([bits])
        )


def test_keepout_witness_is_frozen_once_and_unknown_fails_closed():
    env = _table_attribution_rig()
    capture = wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution
    capture(
        env,
        keepout=torch.tensor([True]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=4,
        capture_boundary="physics_substep_poststate",
        keepout_witness=_keepout_witness(component_id="first"),
    )
    capture(
        env,
        keepout=torch.tensor([True]),
        resolved=torch.tensor([False]),
        resolved_is_final=False,
        substep_index=5,
        capture_boundary="physics_substep_poststate",
        keepout_witness=_keepout_witness(component_id="second"),
    )
    assert env._diagnostic_table_tick_first_positive["keepout_witness"][
        "component_id"
    ] == "first"

    unknown = _keepout_witness()
    unknown["table_role"] = "mystery"
    fresh = _table_attribution_rig()
    with pytest.raises(RuntimeError, match="witness is unknown"):
        capture(
            fresh,
            keepout=torch.tensor([True]),
            resolved=torch.tensor([False]),
            resolved_is_final=False,
            substep_index=1,
            capture_boundary="physics_substep_poststate",
            keepout_witness=unknown,
        )


def test_substep_resolved_contact_can_separate_before_canonical_final_forward():
    env = _table_attribution_rig()
    capture = wait_env.FullMdpInitialWaitVecEnv._capture_diagnostic_table_attribution
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([True]),
        resolved_is_final=False,
        substep_index=7,
        capture_boundary="physics_substep_poststate",
    )
    capture(
        env,
        keepout=torch.tensor([False]),
        resolved=torch.tensor([False]),
        resolved_is_final=True,
        substep_index=None,
        capture_boundary="post_forward_final",
    )
    result = wait_env.FullMdpInitialWaitVecEnv.diagnostic_table_attribution_tick(
        env, torch.tensor([0])
    )
    assert result["resolved_any_substep"] is True
    assert result["backend_resolved_table_contact"] is False
    assert result["keepout_source"] is False
    assert result["first_table_terminal_source"] is None
    assert result["first_resolved_substep"] == {
        "transition_start_step": 41,
        "capture_boundary": "physics_substep_poststate",
        "physics_substep_index": 7,
        "completed_physics_substeps": 7,
        "backend_resolved_table_contact": True,
    }


def test_teacher_replay_cli_is_n1_zero_ppo_and_reuses_live_owner():
    signature = inspect.signature(runner.main)
    assert signature.parameters["diagnostic_teacher_replay"].default is False
    assert signature.parameters["diagnostic_teacher_replay_steps"].default == 180
    source = inspect.getsource(runner._run_teacher_replay)
    assert "env.step(action)" in source
    assert "ppo_update_calls\": 0" in source
    assert "enable_diagnostic_first_generic_contact_patch" in source
    assert "_epoch_task_f32" in source
    assert "_full_a_launch_state_f32" in source
