"""Focused CPU contracts for the A211/C211 hidden-WAIT teacher tuple."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
MDP = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
COMMANDS_PATH = MDP / "commands.py"
HOPE_COMMANDS_PATH = MDP / "hope_commands.py"
OBSERVATIONS_PATH = MDP / "hope_observations.py"
REWARDS_PATH = MDP / "hope_rewards.py"
TASK_CFG = ROOT / "cfg" / "task"


def _quat_mul(lhs, rhs):
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_inv(quat):
    result = quat.clone()
    result[..., 1:] *= -1.0
    return result / torch.sum(torch.square(quat), dim=-1, keepdim=True)


def _quat_apply(quat, vector):
    xyz = quat[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def _yaw_quat(quat):
    yaw = torch.atan2(
        2.0 * (quat[..., 0] * quat[..., 3] + quat[..., 1] * quat[..., 2]),
        1.0 - 2.0 * (torch.square(quat[..., 2]) + torch.square(quat[..., 3])),
    )
    result = torch.zeros_like(quat)
    result[..., 0] = torch.cos(0.5 * yaw)
    result[..., 3] = torch.sin(0.5 * yaw)
    return result


def _load_motion_teacher_view():
    source = COMMANDS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(COMMANDS_PATH))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MotionCommand"
    )
    wanted = {
        "_advance_action_ball_task_timing",
        "refresh_action_ball_revealed_body_reference",
        "_action_ball_safe_ready_wait_mask",
        "_capture_action_ball_safe_ready_reference",
        "action_ball_time_to_contact_remaining_s",
        "action_ball_pre_swing_wait_remaining_s",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    methods = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert {node.name for node in methods} == wanted
    view = ast.ClassDef(
        name="MotionTeacherView",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[view], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "quat_apply": _quat_apply,
        "quat_inv": _quat_inv,
        "quat_mul": _quat_mul,
        "yaw_quat": _yaw_quat,
    }
    exec(compile(module, str(COMMANDS_PATH), "exec"), namespace)
    return namespace["MotionTeacherView"]


def _load_racket_wait_view():
    source = HOPE_COMMANDS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(HOPE_COMMANDS_PATH))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
    )
    method = next(
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_advance_action_ball_task_wait"
    )
    view = ast.ClassDef(
        name="RacketWaitView",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            view,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "_TASK_REVEAL_REACHED_COUNTER": "task_reveal_reached_count",
        "_action_ball_validate_tensor_predicate": (
            lambda predicate, message, **_kwargs: (
                None
                if bool(torch.all(predicate))
                else (_ for _ in ()).throw(RuntimeError(message))
            )
        ),
    }
    exec(compile(module, str(HOPE_COMMANDS_PATH), "exec"), namespace)
    return namespace["RacketWaitView"]


def _load_action_ball_211_mask_helpers():
    source = OBSERVATIONS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OBSERVATIONS_PATH))
    wanted = {
        "_action_ball_211_task_valid",
        "_action_ball_211_mask_task_value",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(OBSERVATIONS_PATH), "exec"), namespace)
    return namespace


def _load_stage1_motion_and_command():
    source = OBSERVATIONS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OBSERVATIONS_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_stage1_motion_and_command"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            helper,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"_cmd": lambda env, _name: env.command}
    exec(compile(module, str(OBSERVATIONS_PATH), "exec"), namespace)
    return namespace["_stage1_motion_and_command"]


def _motion_teacher_fixture():
    view = _load_motion_teacher_view()()
    dtype = torch.float64
    measured_joint = torch.arange(31, dtype=dtype).reshape(1, 31) + 100.0
    safe_joint = torch.arange(31, dtype=dtype).reshape(1, 31) - 10.0
    measured_body_pos = torch.tensor(
        [[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]], dtype=dtype
    )
    measured_body_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=dtype
    )
    view.num_envs = 2
    view.device = torch.device("cpu")
    view.canonical_ready_mode = True
    view.action_ball_diagnostic_split_ready_teacher = True
    view._action_ball_public_task_valid = torch.tensor([False, True])
    view._action_ball_safe_ready_reference_pending = torch.zeros(2, dtype=torch.bool)
    view._action_ball_safe_ready_pending_count = 0
    view._action_ball_dynamic_ready_physical_joint_pos_rad = safe_joint
    view.clip_id = torch.zeros(2, dtype=torch.long)
    view.time_steps = torch.zeros(2, dtype=torch.long)
    view.in_hold = torch.zeros(2, dtype=torch.bool)
    view.retiming_active = False
    view.speed_scale = torch.ones(2, dtype=dtype)
    view.body_indexes = torch.arange(2, dtype=torch.long)
    view.motion_anchor_body_index = 0
    view.robot_anchor_body_index = 0
    view.cfg = SimpleNamespace(body_names=("pelvis", "right_wrist"))
    view._pose_reference_steps = lambda: torch.zeros(2, dtype=torch.long)
    view._env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.zeros((2, 3), dtype=dtype))
    )
    view.motion = SimpleNamespace(
        joint_pos=measured_joint,
        joint_vel=torch.full((1, 31), 4.0, dtype=dtype),
        body_pos_w=measured_body_pos,
        body_quat_w=measured_body_quat,
        body_lin_vel_w=torch.full((1, 2, 3), 5.0, dtype=dtype),
        body_ang_vel_w=torch.full((1, 2, 3), 6.0, dtype=dtype),
        seg_start=torch.zeros(1, dtype=torch.long),
        seg_len=torch.ones(1, dtype=torch.long),
        num_segments=1,
    )
    view.robot = SimpleNamespace(
        body_names=("pelvis", "right_wrist"),
        data=SimpleNamespace(
            body_pos_w=torch.tensor(
                [
                    [[20.0, 30.0, 40.0], [0.0, 0.0, 0.0]],
                    [[50.0, 60.0, 70.0], [0.0, 0.0, 0.0]],
                ],
                dtype=dtype,
            ),
            body_quat_w=torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                ],
                dtype=dtype,
            ),
        ),
    )
    view._action_ball_safe_ready_body_pos_w = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
        ],
        dtype=dtype,
    )
    view._action_ball_safe_ready_body_quat_w = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=dtype,
    )
    view.body_pos_relative_w = view._action_ball_safe_ready_body_pos_w.clone()
    view.body_quat_relative_w = view._action_ball_safe_ready_body_quat_w.clone()
    view._multiseg = False
    return view, safe_joint, measured_joint, measured_body_pos, measured_body_quat


@pytest.mark.parametrize(
    "task_yaml",
    (
        "HOPEPingPongActionBallA211VendorV2N1Learnability.yaml",
        "HOPEPingPongActionBallC211VendorV2N1Learnability.yaml",
    ),
)
def test_hidden_wait_joint_body_paddle_tuple_reveals_atomically_at_frame0(
    task_yaml,
):
    source = (TASK_CFG / task_yaml).read_text(encoding="utf-8")
    assert source.count("action_ball_diagnostic_split_ready_teacher: true") == 1
    (
        view,
        safe_joint,
        measured_joint,
        measured_body_pos,
        measured_body_quat,
    ) = _motion_teacher_fixture()
    reward_namespace = _load_reward_helpers()
    paddle_cmd, measured_paddle, measured_long = _paddle_fixture(
        reward_namespace, motion=view
    )

    torch.testing.assert_close(view.joint_pos[0], safe_joint[0])
    torch.testing.assert_close(view.joint_pos[1], measured_joint[0])
    torch.testing.assert_close(
        view.body_pos_w[0], view._action_ball_safe_ready_body_pos_w[0]
    )
    torch.testing.assert_close(view.body_pos_w[1], measured_body_pos[0])
    torch.testing.assert_close(
        view.body_quat_w[0], view._action_ball_safe_ready_body_quat_w[0]
    )
    torch.testing.assert_close(view.body_quat_w[1], measured_body_quat[0])
    assert torch.count_nonzero(view.joint_vel[0]) == 0
    assert torch.count_nonzero(view.body_lin_vel_w[0]) == 0
    assert torch.count_nonzero(view.body_ang_vel_w[0]) == 0
    assert torch.equal(view.joint_vel[1], torch.full((31,), 4.0))
    assert torch.equal(view.body_lin_vel_w[1], torch.full((2, 3), 5.0))
    assert torch.equal(view.body_ang_vel_w[1], torch.full((2, 3), 6.0))
    hidden_paddle = reward_namespace["_stage1_aligned_clip_site_target"](paddle_cmd)
    hidden_long = reward_namespace["_stage1_aligned_clip_long_axis_target"](paddle_cmd)
    torch.testing.assert_close(
        hidden_paddle[0][0], torch.tensor([4.2, 5.0, 6.0], dtype=torch.float64)
    )
    assert not torch.equal(hidden_paddle[0][0], measured_paddle[0][0])
    assert not torch.equal(hidden_long[0], measured_long[0])

    # Reproduce the command-manager ordering: Motion has already cached the
    # hidden safe-ready aligned bodies, then Racket flips the one public bit.
    stale_wait_body = view.body_pos_relative_w[0].clone()
    untouched_active_body = view.body_pos_relative_w[1].clone()
    view._action_ball_public_task_valid[0] = True
    view.refresh_action_ball_revealed_body_reference(torch.tensor([True, False]))

    torch.testing.assert_close(view.joint_pos, measured_joint.expand(2, -1))
    torch.testing.assert_close(view.body_pos_w, measured_body_pos.expand(2, -1, -1))
    torch.testing.assert_close(view.body_quat_w, measured_body_quat.expand(2, -1, -1))
    torch.testing.assert_close(
        view.body_pos_relative_w[0],
        torch.tensor(
            [[20.0, 30.0, 12.0], [23.0, 33.0, 15.0]],
            dtype=torch.float64,
        ),
    )
    assert not torch.equal(view.body_pos_relative_w[0], stale_wait_body)
    torch.testing.assert_close(view.body_pos_relative_w[1], untouched_active_body)
    torch.testing.assert_close(view.body_quat_relative_w[0], measured_body_quat[0])
    assert torch.equal(view.joint_vel[0], torch.full((31,), 4.0))
    assert torch.equal(view.body_lin_vel_w[0], torch.full((2, 3), 5.0))
    assert torch.equal(view.body_ang_vel_w[0], torch.full((2, 3), 6.0))

    revealed_paddle = reward_namespace["_stage1_aligned_clip_site_target"](paddle_cmd)
    revealed_long = reward_namespace["_stage1_aligned_clip_long_axis_target"](
        paddle_cmd
    )
    for actual, expected in zip(revealed_paddle, measured_paddle):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(revealed_long, measured_long)


def test_motion_then_racket_reveals_async_rows_exactly_at_private_wait_tick():
    """Regression for the production CommandManager Motion -> Racket order.

    RESET_WAIT is intentionally private, so this does not make the WAIT phase
    Markov.  It only proves that the already-defined public ABI changes in one
    transaction: task tuple, base target, both clocks, and the whole teacher
    tuple all remain hidden/safe-ready through W-1 and become task/frame-0 on
    W, with the receipt clocks restored exactly (rather than one tick early or
    late).
    """

    (
        motion,
        safe_joint,
        measured_joint,
        _measured_body_pos,
        _measured_body_quat,
    ) = _motion_teacher_fixture()
    dtype = torch.float64
    dt = 0.02
    wait_ticks = torch.tensor([2, 4], dtype=torch.long)
    wait_s = wait_ticks.to(dtype=dtype) * dt
    receipt_ttc = torch.tensor([1.20, 1.60], dtype=dtype)
    receipt_teacher_start = torch.tensor([0.70, 0.90], dtype=dtype)

    # This is the state immediately after _action_ball_arm_task_wait(): the
    # immutable receipt clocks have each been extended by the private wait.
    motion._action_ball_task_timing_active = torch.ones(2, dtype=torch.bool)
    motion._action_ball_time_to_contact_s = receipt_ttc + wait_s
    motion._action_ball_pre_swing_wait_s = receipt_teacher_start + wait_s
    motion._action_ball_task_age_s = torch.zeros(2, dtype=dtype)
    motion._action_ball_scaled_t_cycle_s = torch.full((2,), 10.0, dtype=dtype)
    motion._action_ball_teacher_rate = torch.ones(2, dtype=dtype)
    motion._action_ball_birth_broker = SimpleNamespace(diagnostic_fast_path=True)
    motion._action_ball_diagnostic_pending_row_count = 0
    motion._action_ball_reset_generation = torch.ones(2, dtype=torch.long)
    motion._resolve_pending_action_ball_tasks = lambda: None
    motion.time_steps_f = torch.zeros(2, dtype=dtype)
    motion.time_steps = torch.zeros(2, dtype=torch.long)
    motion.motion.seg_len = torch.ones(1, dtype=torch.long)
    motion.hold_counter = torch.zeros(2, dtype=torch.long)
    motion.metrics = {}
    motion._env.step_dt = dt

    task_valid = torch.zeros(2, dtype=torch.bool)
    motion._action_ball_public_task_valid = task_valid
    racket = _load_racket_wait_view()()
    racket._env = SimpleNamespace(common_step_counter=0)
    racket._action_ball_task_wait_schedule = object()
    racket._action_ball_task_wait_last_advance_step = -1
    racket._action_ball_task_wait_elapsed_ticks = torch.zeros(
        2, dtype=torch.long
    )
    racket._action_ball_task_wait_total_ticks = wait_ticks.clone()
    racket._action_ball_task_valid = task_valid
    racket._action_ball_diagnostic_unauthorized = False
    reveal_counters = {
        "task_reveal_reached_count": torch.zeros((), dtype=torch.long)
    }
    racket._ensure_exact_behavior_decision_counters = lambda: reveal_counters
    racket._motion = lambda: motion

    observation = _load_action_ball_211_mask_helpers()
    mask_value = observation["_action_ball_211_mask_task_value"]
    public_command = SimpleNamespace(
        _action_ball_task_valid=task_valid,
        robot=SimpleNamespace(
            data=SimpleNamespace(root_pos_w=torch.zeros((2, 3), dtype=dtype))
        ),
    )
    task_source = torch.arange(18, dtype=dtype).reshape(2, 9) + 1.0
    base_source = torch.tensor([[0.3, -0.4], [0.5, -0.6]], dtype=dtype)
    reward_namespace = _load_reward_helpers()
    paddle_cmd, measured_paddle, measured_long = _paddle_fixture(
        reward_namespace, motion=motion
    )

    reveal_tick = [-1, -1]
    for tick in range(1, int(wait_ticks.max().item()) + 1):
        # Production ordering: Motion consumes this policy interval first.
        motion._advance_action_ball_task_timing()

        # On the reveal tick itself, the pre-Racket view must still be the WAIT
        # view; Racket owns the single false -> true transaction below.
        pre_racket_valid = task_valid.clone()
        pre_racket_task = mask_value(public_command, task_source)
        pre_racket_joint = motion.joint_pos.clone()
        for env_id, private_w in enumerate(wait_ticks.tolist()):
            if tick <= private_w:
                assert not pre_racket_valid[env_id]
                assert torch.count_nonzero(pre_racket_task[env_id]) == 0
                torch.testing.assert_close(
                    pre_racket_joint[env_id], safe_joint[0]
                )

        racket._env.common_step_counter = tick
        racket._advance_action_ball_task_wait()

        public_task = mask_value(public_command, task_source)
        public_base = mask_value(public_command, base_source)
        public_ttc = mask_value(
            public_command,
            motion.action_ball_time_to_contact_remaining_s.unsqueeze(-1),
        )
        public_teacher_start = mask_value(
            public_command,
            motion.action_ball_pre_swing_wait_remaining_s.unsqueeze(-1),
        )
        public_paddle = reward_namespace[
            "_stage1_aligned_clip_site_target"
        ](paddle_cmd)
        public_long = reward_namespace[
            "_stage1_aligned_clip_long_axis_target"
        ](paddle_cmd)

        for env_id, private_w in enumerate(wait_ticks.tolist()):
            should_be_public = tick >= private_w
            assert bool(task_valid[env_id]) is should_be_public
            if should_be_public and reveal_tick[env_id] < 0:
                reveal_tick[env_id] = tick

                # Every public task field appears on this same Racket tick.
                torch.testing.assert_close(public_task[env_id], task_source[env_id])
                torch.testing.assert_close(public_base[env_id], base_source[env_id])
                torch.testing.assert_close(
                    public_ttc[env_id, 0], receipt_ttc[env_id]
                )
                torch.testing.assert_close(
                    public_teacher_start[env_id, 0],
                    receipt_teacher_start[env_id],
                )

                # The teacher switches directly from physical safe-ready to
                # the measured frame-0 joint/body/paddle tuple.  There is no
                # connector frame and no stale Motion cache for this tick.
                assert motion.time_steps[env_id].item() == 0
                torch.testing.assert_close(
                    motion.joint_pos[env_id], measured_joint[0]
                )
                assert not torch.equal(
                    motion.body_pos_relative_w[env_id],
                    motion._action_ball_safe_ready_body_pos_w[env_id],
                )
                for actual, expected in zip(public_paddle, measured_paddle):
                    torch.testing.assert_close(actual[env_id], expected[env_id])
                torch.testing.assert_close(
                    public_long[env_id], measured_long[env_id]
                )
            elif not should_be_public:
                assert torch.count_nonzero(public_task[env_id]) == 0
                assert torch.count_nonzero(public_base[env_id]) == 0
                assert torch.count_nonzero(public_ttc[env_id]) == 0
                assert torch.count_nonzero(public_teacher_start[env_id]) == 0
                torch.testing.assert_close(
                    motion.joint_pos[env_id], safe_joint[0]
                )
                torch.testing.assert_close(
                    motion.body_pos_relative_w[env_id],
                    motion._action_ball_safe_ready_body_pos_w[env_id],
                )

    assert reveal_tick == wait_ticks.tolist()
    assert reveal_counters["task_reveal_reached_count"].item() == 2


def test_reset_capture_replaces_previous_episode_body_and_paddle_cache_before_first_obs():
    view, _, _, _, _ = _motion_teacher_fixture()
    namespace = _load_reward_helpers()
    paddle_cmd, _, _ = _paddle_fixture(namespace, motion=view)
    poison_pos = torch.full_like(view.body_pos_relative_w[0], -999.0)
    poison_quat = torch.full_like(view.body_quat_relative_w[0], -777.0)
    view.body_pos_relative_w[0] = poison_pos
    view.body_quat_relative_w[0] = poison_quat
    view._action_ball_safe_ready_body_pos_w[0] = poison_pos
    view._action_ball_safe_ready_body_quat_w[0] = poison_quat
    untouched_pos = view.body_pos_relative_w[1].clone()
    untouched_quat = view.body_quat_relative_w[1].clone()
    view._action_ball_safe_ready_reference_pending[0] = True
    view._action_ball_safe_ready_pending_count = 1

    # CommandTerm.reset() resamples but does not run Motion._update_command.
    # The first Stage-1 observation binding must therefore perform the capture
    # itself, regardless of which concrete observation term comes first.
    observation_binding = _load_stage1_motion_and_command()
    observation_binding(
        SimpleNamespace(
            command=SimpleNamespace(_motion=lambda: view, robot=view.robot)
        ),
        "racket_target",
    )

    expected_pos = view.robot.data.body_pos_w[0, view.body_indexes]
    expected_quat = view.robot.data.body_quat_w[0, view.body_indexes]
    torch.testing.assert_close(view._action_ball_safe_ready_body_pos_w[0], expected_pos)
    torch.testing.assert_close(
        view._action_ball_safe_ready_body_quat_w[0], expected_quat
    )
    torch.testing.assert_close(view.body_pos_relative_w[0], expected_pos)
    torch.testing.assert_close(view.body_quat_relative_w[0], expected_quat)
    torch.testing.assert_close(view.body_pos_relative_w[1], untouched_pos)
    torch.testing.assert_close(view.body_quat_relative_w[1], untouched_quat)
    assert not view._action_ball_safe_ready_reference_pending[0]
    assert view._action_ball_safe_ready_pending_count == 0

    # Prove the paddle/reward path is independently safe when it is the first
    # consumer after another reset, rather than relying on the observation
    # binding above to have captured the tuple first.
    view.body_pos_relative_w[0] = poison_pos
    view.body_quat_relative_w[0] = poison_quat
    view._action_ball_safe_ready_body_pos_w[0] = poison_pos
    view._action_ball_safe_ready_body_quat_w[0] = poison_quat
    view._action_ball_safe_ready_reference_pending[0] = True
    view._action_ball_safe_ready_pending_count = 1
    safe_paddle = namespace["_stage1_aligned_clip_site_target"](paddle_cmd)
    torch.testing.assert_close(view.body_pos_relative_w[0], expected_pos)
    torch.testing.assert_close(view.body_quat_relative_w[0], expected_quat)
    torch.testing.assert_close(
        safe_paddle[0][0],
        torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64),
    )
    assert not torch.any(safe_paddle[0][0] == -999.0)


def _load_reward_helpers():
    source = REWARDS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REWARDS_PATH))
    wanted = {
        "action_ball_task_valid_mask",
        "_stage1_split_ready_wait_mask",
        "_stage1_quat_normalize",
        "_stage1_quat_mul",
        "_stage1_quat_apply",
        "_stage1_split_ready_safe_racket_tuple",
        "_stage1_select_split_ready_site_target",
        "_cauchy_tracking_kernel",
        "_window_pos",
        "_target_component_valid",
        "_target_position_now",
        "_pos_kernel_raw",
        "racket_position_tracking_exp",
        "base_position_tracking_exp",
        "pre_strike_foot_slip",
        "strike_capture_bonus",
        "virtual_landing_dense_actual_contact",
        "motion_racket_position_tracking_cauchy",
        "motion_racket_velocity_tracking_cauchy",
        "motion_racket_normal_tracking_cauchy",
        "motion_racket_long_axis_tracking_cauchy",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "math": math,
        "torch": torch,
        "_dbg_log": lambda *_args, **_kwargs: None,
        "racket_contact_geometry": SimpleNamespace(
            RACKET_BUTT_TO_BLADE_AXIS_LOCAL=(
                1.0 / math.sqrt(2.0),
                0.0,
                1.0 / math.sqrt(2.0),
            )
        ),
    }
    exec(compile(module, str(REWARDS_PATH), "exec"), namespace)
    return namespace


def _paddle_fixture(namespace, *, motion=None):
    dtype = torch.float64
    if motion is None:
        task_valid = torch.tensor([False, True])
        motion = SimpleNamespace(
            action_ball_diagnostic_split_ready_teacher=True,
            _action_ball_public_task_valid=task_valid,
            robot=SimpleNamespace(body_names=("pelvis", "right_wrist")),
            cfg=SimpleNamespace(body_names=("pelvis", "right_wrist")),
            body_pos_relative_w=torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                    [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
                ],
                dtype=dtype,
            ),
            body_quat_relative_w=torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                ],
                dtype=dtype,
            ),
            motion=SimpleNamespace(num_segments=1),
            clip_id=torch.zeros(2, dtype=torch.long),
            _multiseg=False,
        )
    else:
        task_valid = motion._action_ball_public_task_valid
    cmd = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        pre_strike=torch.zeros(2, dtype=torch.bool),
        strike_window=torch.zeros(2, dtype=torch.bool),
        _action_ball_task_valid=task_valid,
        _motion=lambda: motion,
        _racket_mode="wrist_offset",
        _racket_body_index=-1,
        _wrist_body_index=1,
        _mount_offset=torch.tensor([[0.2, 0.0, 0.0]] * 2, dtype=dtype),
        _mount_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=dtype),
        _mount_signs_cfg=lambda _count: (-1.0,),
        cfg=SimpleNamespace(
            mount_normal_axis=1,
            mount_normal_sign=1.0,
            mount_normal_sign_per_clip=(),
            motion_teacher_racket_source="measured_channel",
        ),
    )
    measured = (
        torch.tensor([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]], dtype=dtype),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype),
        torch.tensor([[7.0, 7.0, 7.0], [6.0, 6.0, 6.0]], dtype=dtype),
    )
    measured_long = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    select = namespace["_stage1_select_split_ready_site_target"]
    namespace["_stage1_aligned_clip_site_target"] = lambda _cmd: select(_cmd, measured)
    namespace["_stage1_aligned_clip_long_axis_target"] = lambda _cmd: torch.where(
        namespace["_stage1_split_ready_wait_mask"](_cmd)[:, None],
        namespace["_stage1_split_ready_safe_racket_tuple"](_cmd)[3],
        measured_long,
    )
    namespace["_cmd"] = lambda _env, _name: cmd
    namespace["_window_wide"] = lambda _cmd: torch.zeros(2, dtype=torch.bool)
    return cmd, measured, measured_long


def test_hidden_wait_paddle_tuple_has_no_measured_frame0_leak_and_rewards_match():
    namespace = _load_reward_helpers()
    cmd, measured, measured_long = _paddle_fixture(namespace)
    selected = namespace["_stage1_aligned_clip_site_target"](cmd)
    selected_long = namespace["_stage1_aligned_clip_long_axis_target"](cmd)
    expected_safe_pos = torch.tensor([1.2, 2.0, 3.0], dtype=torch.float64)
    expected_safe_face = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
    expected_safe_long = torch.tensor(
        [1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(selected[0][0], expected_safe_pos)
    torch.testing.assert_close(selected[1][0], expected_safe_face)
    assert torch.count_nonzero(selected[2][0]) == 0
    torch.testing.assert_close(selected_long[0], expected_safe_long)
    for actual, expected in zip(selected, measured):
        torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(selected_long[1], measured_long[1])

    cmd.racket_pos_w = selected[0].clone()
    cmd.racket_normal_w = selected[1].clone()
    cmd.racket_lin_vel_w = selected[2].clone()
    cmd.racket_long_axis_w = selected_long.clone()
    for name in (
        "motion_racket_position_tracking_cauchy",
        "motion_racket_velocity_tracking_cauchy",
        "motion_racket_normal_tracking_cauchy",
        "motion_racket_long_axis_tracking_cauchy",
    ):
        actual = namespace[name](object(), "racket_target", std=1.0)
        torch.testing.assert_close(actual, torch.ones(2, dtype=torch.float64))

    # One public bit reveals every paddle component together.  The WAIT row
    # switches straight to the measured producer; there is no connector row.
    cmd._action_ball_task_valid.fill_(True)
    revealed = namespace["_stage1_aligned_clip_site_target"](cmd)
    revealed_long = namespace["_stage1_aligned_clip_long_axis_target"](cmd)
    for actual, expected in zip(revealed, measured):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(revealed_long, measured_long)


def test_invalid_wait_masks_task_contact_and_outcome_reward_but_keeps_balance():
    namespace = _load_reward_helpers()
    dtype = torch.float64
    cmd = SimpleNamespace(
        _action_ball_task_valid=torch.tensor([False, True]),
        pre_strike=torch.ones(2, dtype=torch.bool),
        strike_window=torch.ones(2, dtype=torch.bool),
        racket_pos_w=torch.zeros((2, 3), dtype=dtype),
        racket_target_pos_w=torch.zeros((2, 3), dtype=dtype),
        racket_target_vel_w=torch.zeros((2, 3), dtype=dtype),
        time_to_strike=torch.zeros(2, dtype=dtype),
        base_pos_w=torch.zeros((2, 3), dtype=dtype),
        base_target_pos_w=torch.zeros((2, 2), dtype=dtype),
        foot_slip_in_contact=torch.tensor([2.0, 3.0], dtype=dtype),
        vb_fired=torch.ones(2, dtype=torch.bool),
        vb_landing_valid=torch.ones(2, dtype=torch.bool),
        vb_landing_xy=torch.zeros((2, 2), dtype=dtype),
        _vb_target_xy_per_env=torch.zeros((2, 2), dtype=dtype),
        cfg=SimpleNamespace(vb_landing_sigma=1.0),
    )
    namespace["_cmd"] = lambda _env, _name: cmd

    expected_eligible = torch.tensor([0.0, 1.0], dtype=dtype)
    torch.testing.assert_close(
        namespace["racket_position_tracking_exp"](object(), "racket_target", std=1.0),
        expected_eligible,
    )
    torch.testing.assert_close(
        namespace["base_position_tracking_exp"](object(), "racket_target", std=1.0),
        expected_eligible,
    )
    torch.testing.assert_close(
        namespace["strike_capture_bonus"](object(), "racket_target"),
        expected_eligible.float(),
    )
    torch.testing.assert_close(
        namespace["virtual_landing_dense_actual_contact"](object(), "racket_target"),
        expected_eligible,
    )
    # Balance/safety shaping is intentionally task-independent during WAIT.
    torch.testing.assert_close(
        namespace["pre_strike_foot_slip"](object(), "racket_target"),
        cmd.foot_slip_in_contact,
    )


def _load_racket_ledger_view():
    source = HOPE_COMMANDS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(HOPE_COMMANDS_PATH))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
    )
    wanted = {
        "_action_ball_close_attempts",
        "_book_sparse_reward_eligibility",
        "_vb_book_strike_step",
    }
    methods = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert {node.name for node in methods} == wanted
    view = ast.ClassDef(
        name="RacketLedgerView",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            view,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "_batched_host_scalar_values": lambda tensors: tuple(
            float(value.item()) for value in tensors
        ),
        "_action_ball_validate_tensor_predicate": lambda predicate, *_args, **_kwargs: (
            None
            if bool(torch.all(predicate))
            else (_ for _ in ()).throw(AssertionError("predicate failed"))
        ),
        "_EXACT_STRIKE_TIMING_COUNTER": "exact_strike_timing_tick_count",
        "_ACTION_BALL_LEDGER_NAMES": (
            "P",
            "A",
            "I",
            "S",
            "C",
            "H",
            "L",
            "F",
            "U_table",
            "U_fall",
            "U_collision",
            "U_joint_qdes",
            "U_joint_actual",
            "X",
        ),
        "_ACTION_BALL_CONTACT_REJECTION_COUNTERS": (
            "virtual_contact_face_reject_count",
            "virtual_contact_geometry_reject_count",
            "virtual_contact_nonfinite_reject_count",
            "virtual_contact_u_n_below_fit_reject_count",
            "virtual_contact_u_n_above_fit_reject_count",
        ),
    }
    exec(compile(module, str(HOPE_COMMANDS_PATH), "exec"), namespace)
    return namespace["RacketLedgerView"], namespace


def _racket_ledger_fixture():
    ledger_type, namespace = _load_racket_ledger_view()
    view = ledger_type()
    view.device = torch.device("cpu")
    # Complete the production constructor contract for the extracted view.
    # Legacy hidden-WAIT evidence must never silently opt into fixed-view
    # semantics merely because this dependency-light fixture omitted a field.
    view._action_ball_fixed_view_enabled = False
    view._action_ball_task_wait_schedule = object()
    view._action_ball_task_valid = torch.tensor([False, True])
    view._action_ball_attempt_active = torch.ones(2, dtype=torch.bool)
    view._action_ball_attempt_action = torch.zeros(2, dtype=torch.long)
    view._action_ball_attempt_legal = torch.zeros(2, dtype=torch.bool)
    view._action_ball_attempt_hit = torch.zeros(2, dtype=torch.bool)
    view._action_ball_bindings = (object(),)
    view._action_ball_diagnostic_unauthorized = False
    view._action_ball_ledger = torch.zeros(
        (len(namespace["_ACTION_BALL_LEDGER_NAMES"]), 1), dtype=torch.long
    )
    view._action_ball_live_ledger = lambda: view._action_ball_ledger
    view._action_ball_task_by_env = ["hidden", "active"]
    view._action_ball_task_ref_by_env = ["hidden", "active"]
    view._counter_rally_task_identity_by_env = ["hidden", "active"]
    view._counter_rally_reward_terms = torch.ones((2, 5), dtype=torch.float64)
    view._counter_rally_accepted = torch.ones(2, dtype=torch.bool)
    view._counter_rally_legal_first_landing = torch.ones(2, dtype=torch.bool)
    view._counter_rally_primary_reason_code = torch.zeros(2, dtype=torch.long)
    view._action_ball_invalidate_virtual_contact_history = lambda _ids: None
    view._clip_names = {}
    view._book_exact_timing_bucket_sparse_events = lambda masks: setattr(
        view, "_last_sparse_masks", masks
    )
    view._motion = lambda: SimpleNamespace()
    view._metric_bucket_accounting_enabled = lambda _motion: False
    view._action_ball_diagnostic_device_telemetry_enabled = lambda: False
    view._vb_exact_acc = 0.0
    view._vb_hit_acc = 0.0
    view._vb_net_acc = 0.0
    view._vb_land_valid_acc = 0.0
    view._vb_inb_acc = 0.0
    view._rally_returned = torch.zeros(2, dtype=torch.bool)
    view._rally_pending_return = torch.zeros(2, dtype=torch.bool)
    view._action_ball_enabled = True
    return view, namespace


def test_invalid_wait_contributes_no_closed_or_sparse_denominator():
    view, namespace = _racket_ledger_fixture()
    ids = torch.arange(2, dtype=torch.long)

    # Both rows carry an installed attempt, but only the revealed row is an
    # eligible closed swing.  Its miss must be represented honestly as 0/1.
    view._action_ball_close_attempts(ids, true_reset=False, active_host_env_ids=(1,))
    names = namespace["_ACTION_BALL_LEDGER_NAMES"]
    assert view._action_ball_ledger[names.index("C"), 0].item() == 1
    assert view._action_ball_ledger[names.index("H"), 0].item() == 0
    assert view._action_ball_ledger[names.index("L"), 0].item() == 0
    assert view._action_ball_ledger[names.index("F"), 0].item() == 1

    # Sparse opportunity/contact/outcome counters use the same task-valid
    # ownership.  Two identical raw events become exactly one eligible event.
    both = torch.ones(2, dtype=torch.bool)
    view._book_sparse_reward_eligibility(
        exact_strike=both,
        capture=both,
        net_clear=both,
        landing_valid=both,
        legal_return=both,
    )
    counters = view._sparse_reward_eligibility_counters
    for name in (
        namespace["_EXACT_STRIKE_TIMING_COUNTER"],
        "strike_opportunity_count",
        "virtual_capture_count",
        "virtual_net_clear_count",
        "virtual_landing_valid_count",
        "virtual_legal_return_count",
    ):
        assert counters[name].item() == 1


@pytest.mark.parametrize("use_counter_rally_acceptance", (False, True))
def test_invalid_wait_cannot_latch_hit_or_legal_attempt_outcome(
    use_counter_rally_acceptance,
):
    view, _ = _racket_ledger_fixture()
    both = torch.ones(2, dtype=torch.bool)
    accepted = both if use_counter_rally_acceptance else None

    view._vb_book_strike_step(
        decay=1.0,
        exact_strike=both,
        gate=both,
        net_clear=both,
        land_valid=both,
        legal=both,
        counter_rally_accepted=accepted,
    )

    expected = torch.tensor([False, True])
    assert torch.equal(view._action_ball_attempt_hit, expected)
    assert torch.equal(view._action_ball_attempt_legal, expected)


def test_a211_and_c211_enable_the_same_split_ready_teacher():
    for name in (
        "HOPEPingPongActionBallA211VendorV2N1Learnability.yaml",
        "HOPEPingPongActionBallC211VendorV2N1Learnability.yaml",
    ):
        source = (TASK_CFG / name).read_text(encoding="utf-8")
        assert source.count("action_ball_diagnostic_split_ready_teacher: true") == 1
        assert "action_ball_diagnostic_split_ready_teacher: false" not in source
