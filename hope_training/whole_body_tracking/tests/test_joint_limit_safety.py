"""Fail-closed joint-position safety primitives.

These tests are host-only.  ``test_reward_flags_mdp`` supplies the lightweight Isaac Lab stubs
already used by the rest of the MDP unit suite.

The Pod 1-env Isaac smoke must additionally attest the real manager order, not merely repeat these
fakes: one policy step at ``step_dt=0.02`` has exactly four ``apply_actions`` records at fresh
articulation timestamps ``t + [0, .005, .010, .015]`` and one DoneTerm post-step readback at
``t + .020``; q/qdot must correspond to those timestamps, ``apply_call_count==decimation==4``,
``post_readback_count==1``, and the snapshot must be complete before the next policy step begins.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import tracemalloc
import types
import xml.etree.ElementTree as ET

import pytest
import torch

from test_reward_flags_mdp import hope_actions_mod, hope_rewards_mod, terminations_mod
from test_training_launch_claim import _load_contract_module, _load_runner_module


_REPO_ROOT = Path(__file__).resolve().parents[3]
_A3_URDF = (
    _REPO_ROOT
    / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"
)
_A3_CFG_SOURCE = (
    _REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
_VENDOR_HCTRL_SELECTED_NAMES = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
_VENDOR_HCTRL_TEST_INDICES = (5, 8, 19, 30)


def _vendor_hctrl_test_joint_names() -> list[str]:
    names = [f"j{index}" for index in range(31)]
    for index, name in zip(
        _VENDOR_HCTRL_TEST_INDICES, _VENDOR_HCTRL_SELECTED_NAMES
    ):
        names[index] = name
    return names


def _limits(num_envs: int, joint_count: int, lo: float = -1.0, hi: float = 1.0):
    lower = torch.full((num_envs, joint_count), lo)
    upper = torch.full((num_envs, joint_count), hi)
    return torch.stack((lower, upper), dim=-1)


class _FakeRootPhysxView:
    """Exact CPU limit view matching the Isaac Lab 2.1 get/set contract."""

    def __init__(self, limits: torch.Tensor):
        self._limits = limits.detach().cpu().clone()
        self.set_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def get_dof_limits(self) -> torch.Tensor:
        return self._limits.clone()

    def set_dof_limits(
        self, limits: torch.Tensor, *, indices: torch.Tensor
    ) -> None:
        assert limits.device.type == "cpu"
        assert indices.device.type == "cpu"
        self._limits[indices] = limits[indices]
        self.set_calls.append((limits.clone(), indices.clone()))


def _a3_soft_joint_limit_factor() -> float:
    tree = ast.parse(_A3_CFG_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "AGIBOT_A3_CFG"
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, ast.Call)
        for keyword in node.value.keywords:
            if keyword.arg == "soft_joint_pos_limit_factor":
                return float(ast.literal_eval(keyword.value))
    raise AssertionError("AGIBOT_A3_CFG soft_joint_pos_limit_factor is missing")


def _a3_hard_joint_limits() -> tuple[list[str], torch.Tensor]:
    root = ET.parse(_A3_URDF).getroot()
    rows = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") not in {"revolute", "prismatic"}:
            continue
        limit = joint.find("limit")
        assert limit is not None
        rows.append(
            (
                joint.attrib["name"],
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
        )
    names = [row[0] for row in rows]
    limits = torch.tensor(
        [[row[1], row[2]] for row in rows], dtype=torch.float64
    )
    return names, limits


def test_vendor_dual_position_envelopes_are_nested_on_all_real_a3_joints():
    """Prove Q⊂6%-guard⊆H_ctrl and the four selected H_ctrl insets."""

    names, hard = _a3_hard_joint_limits()
    assert len(names) == len(set(names)) == 31
    assert torch.all(torch.isfinite(hard))
    hard_lower, hard_upper = hard.unbind(dim=-1)
    hard_travel = hard_upper - hard_lower
    assert torch.all(hard_travel > 0.0)

    # Isaac Lab forms the configured 0.9 soft envelope about each hard-limit midpoint.  ActionBall
    # then reserves another 5% of that soft span for finite q_des projection.  Reproduce the exact
    # two source-owned factors instead of substituting an already-materialized receipt.
    soft_factor = _a3_soft_joint_limit_factor()
    assert soft_factor == pytest.approx(0.9)
    hard_mid = 0.5 * (hard_lower + hard_upper)
    soft_half = 0.5 * soft_factor * hard_travel
    soft_lower = hard_mid - soft_half
    soft_upper = hard_mid + soft_half
    projection_inset = 0.05 * (soft_upper - soft_lower)
    projected_lower = soft_lower + projection_inset
    projected_upper = soft_upper - projection_inset

    risk_lower_5 = hard_lower + 0.05 * hard_travel
    risk_upper_5 = hard_upper - 0.05 * hard_travel
    risk_lower_6 = hard_lower + 0.06 * hard_travel
    risk_upper_6 = hard_upper - 0.06 * hard_travel
    control_lower_2 = hard_lower.clone()
    control_upper_2 = hard_upper.clone()
    selected = torch.tensor(
        [name in _VENDOR_HCTRL_SELECTED_NAMES for name in names]
    )
    assert selected.sum().item() == 4
    control_lower_2[selected] += 0.02 * hard_travel[selected]
    control_upper_2[selected] -= 0.02 * hard_travel[selected]
    target_lower_5 = torch.maximum(projected_lower, risk_lower_5)
    target_upper_5 = torch.minimum(projected_upper, risk_upper_5)
    target_lower_6 = torch.maximum(projected_lower, risk_lower_6)
    target_upper_6 = torch.minimum(projected_upper, risk_upper_6)

    assert torch.equal(target_lower_5, target_lower_6)
    assert torch.equal(target_upper_5, target_upper_6)
    assert torch.all(hard_lower[selected] < control_lower_2[selected])
    assert torch.all(control_lower_2 < risk_lower_6)
    assert torch.all(risk_lower_6 <= projected_lower)
    assert torch.all(projected_lower < projected_upper)
    assert torch.all(projected_upper <= risk_upper_6)
    assert torch.all(risk_upper_6 < control_upper_2)
    assert torch.all(control_upper_2[selected] < hard_upper[selected])
    assert torch.equal(control_lower_2[~selected], hard_lower[~selected])
    assert torch.equal(control_upper_2[~selected], hard_upper[~selected])
    assert torch.allclose(
        risk_lower_6 - risk_lower_5,
        0.01 * hard_travel,
        rtol=0.0,
        atol=1e-15,
    )
    assert torch.allclose(
        risk_upper_5 - risk_upper_6,
        0.01 * hard_travel,
        rtol=0.0,
        atol=1e-15,
    )

    waist_roll = names.index("waist_roll_joint")
    assert target_lower_6[waist_roll].item() == pytest.approx(
        -0.2827433388230814, abs=1e-15
    )
    assert target_upper_6[waist_roll].item() == pytest.approx(
        0.2827433388230814, abs=1e-15
    )
    assert (risk_lower_6 - risk_lower_5)[waist_roll].item() == pytest.approx(
        0.006981317007977318, abs=1e-15
    )
    for name in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
        ankle_roll = names.index(name)
        assert control_lower_2[ankle_roll].item() == pytest.approx(
            -0.33510321638291124, abs=1e-15
        )
        assert control_upper_2[ankle_roll].item() == pytest.approx(
            0.33510321638291124, abs=1e-15
        )


def _action_and_env(
    *,
    num_envs: int = 2,
    joint_count: int = 2,
    scale: float = 1.0,
    offset: torch.Tensor | None = None,
    clamp: bool = True,
    guard: bool = False,
    guard_policy_dt_s: float | None = 0.1,
    guard_margin_rad: float | None = 0.0,
    guard_margin_fraction: float | None = 0.0,
    guard_brake_mode: str = "velocity_horizon_v1",
    project_finite_qdes: bool = False,
    projection_soft_inset_fraction: float = 0.05,
    runtime_step_dt: float = 0.1,
    decimation: int = 4,
    runtime_physics_dt: float | None = None,
    expected_decimation: int | None = 4,
    terminal_archive_capacity: int | None = 16,
    control_step_action_delay_min: int = 0,
    control_step_action_delay_max: int = 0,
    physx_control_position_limit_inset_fraction: float = 0.0,
    action_ball_diagnostic_unauthorized: bool = False,
    diagnostic_compact_evidence: bool = False,
    target_mode: str = "action_ball",
    joint_names: list[str] | None = None,
):
    names = (
        [f"j{index}" for index in range(joint_count)]
        if joint_names is None
        else list(joint_names)
    )
    assert len(names) == joint_count
    offset = (
        torch.zeros(num_envs, joint_count)
        if offset is None
        else offset.clone()
    )
    limits = _limits(num_envs, joint_count)
    hard_limits = _limits(num_envs, joint_count, lo=-1.2, hi=1.2)
    root_physx_view = _FakeRootPhysxView(hard_limits)
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            default_joint_pos=offset,
            default_joint_pos_limits=hard_limits.clone(),
            soft_joint_pos_limits=limits,
            # Keep a visible hard-vs-soft distinction: q_des clamps at +/-1.0, while terminal
            # crossing guards use the physical +/-1.2 envelope.
            joint_pos_limits=hard_limits,
            joint_pos=torch.zeros(num_envs, joint_count),
            joint_vel=torch.zeros(num_envs, joint_count),
            _sim_timestamp=0.0,
            _joint_pos=types.SimpleNamespace(timestamp=0.0),
            _joint_vel=types.SimpleNamespace(timestamp=0.0),
        ),
        root_physx_view=root_physx_view,
        _ALL_INDICES=torch.arange(num_envs, dtype=torch.long),
    )
    cfg = types.SimpleNamespace(
        asset_name="robot",
        scale=scale,
        use_default_offset=True,
        clamp=clamp,
        pre_apply_limit_guard=guard,
        pre_apply_guard_policy_dt_s=guard_policy_dt_s,
        pre_apply_guard_margin_rad=guard_margin_rad,
        pre_apply_guard_margin_fraction=guard_margin_fraction,
        pre_apply_guard_brake_mode=guard_brake_mode,
        pre_apply_guard_expected_decimation=expected_decimation,
        pre_apply_guard_terminal_archive_capacity=terminal_archive_capacity,
        project_finite_preclamp_qdes_without_termination=project_finite_qdes,
        finite_projection_soft_envelope_inset_fraction=(
            projection_soft_inset_fraction
        ),
        control_step_action_delay_min=control_step_action_delay_min,
        control_step_action_delay_max=control_step_action_delay_max,
        physx_control_position_limit_inset_fraction=(
            physx_control_position_limit_inset_fraction
        ),
        pre_apply_guard_diagnostic_compact_evidence=(
            diagnostic_compact_evidence
        ),
    )
    env = types.SimpleNamespace(
        scene={"robot": asset},
        num_envs=num_envs,
        device="cpu",
        step_dt=runtime_step_dt,
        physics_dt=(
            runtime_step_dt / decimation
            if runtime_physics_dt is None
            else runtime_physics_dt
        ),
        cfg=types.SimpleNamespace(
            decimation=decimation,
            commands=types.SimpleNamespace(
                racket_target=types.SimpleNamespace(
                    target_mode=target_mode,
                    action_ball_diagnostic_unauthorized=(
                        action_ball_diagnostic_unauthorized
                    ),
                )
            ),
        ),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        common_step_counter=0,
    )
    action = hope_actions_mod.ClampedJointPositionAction(cfg, env)
    env.action_manager = types.SimpleNamespace(
        get_term=lambda name: action if name == "joint_pos" else None
    )
    return action, env, asset


def test_stage1_compact_joint_safety_evidence_is_explicit_and_diagnostic_only():
    with pytest.raises(ValueError, match="requires.*diagnostic_unauthorized=true"):
        _action_and_env(
            target_mode="reference_perturbed",
            diagnostic_compact_evidence=True,
            action_ball_diagnostic_unauthorized=False,
        )

    action, _, _ = _action_and_env(
        target_mode="reference_perturbed",
        diagnostic_compact_evidence=True,
        action_ball_diagnostic_unauthorized=True,
    )
    assert action._joint_safety_diagnostic_compact_evidence is True
    assert action._actual_joint_forbidden_diagnostic_enabled is True


def test_vendor_physx_control_envelope_writes_and_reads_back_without_mutating_hard_or_soft():
    names = _vendor_hctrl_test_joint_names()
    action, _, asset = _action_and_env(
        num_envs=2,
        joint_count=31,
        guard=True,
        guard_margin_fraction=0.06,
        project_finite_qdes=True,
        physx_control_position_limit_inset_fraction=0.02,
        joint_names=names,
    )
    mechanical = _limits(2, 31, lo=-1.2, hi=1.2)
    span = mechanical[..., 1] - mechanical[..., 0]
    expected = mechanical.clone()
    selected_ids = list(_VENDOR_HCTRL_TEST_INDICES)
    expected[:, selected_ids, 0] += 0.02 * span[:, selected_ids]
    expected[:, selected_ids, 1] -= 0.02 * span[:, selected_ids]
    assert action.physx_control_position_limit_inset_fraction == 0.02
    assert action.physx_control_joint_names == tuple(names)
    assert (
        isinstance(action.physx_control_position_limit_readback_sha256, str)
        and len(action.physx_control_position_limit_readback_sha256) == 64
    )
    assert (
        isinstance(action.physx_control_setter_no_mutation_sha256, str)
        and len(action.physx_control_setter_no_mutation_sha256) == 64
        and action.physx_control_setter_no_mutation_sha256
        != action.physx_control_position_limit_readback_sha256
    )
    assert torch.equal(action.physx_control_joint_pos_limits, expected)
    assert torch.equal(asset.root_physx_view.get_dof_limits(), expected)
    assert torch.equal(
        asset.data.joint_pos_limits, _limits(2, 31, lo=-1.2, hi=1.2)
    )
    assert torch.equal(
        asset.data.soft_joint_pos_limits, _limits(2, 31, lo=-1.0, hi=1.0)
    )
    assert len(asset.root_physx_view.set_calls) == 1
    assert torch.equal(
        expected[:, selected_ids],
        asset.root_physx_view.get_dof_limits()[:, selected_ids],
    )
    unchanged_ids = [
        index for index in range(31) if index not in _VENDOR_HCTRL_TEST_INDICES
    ]
    assert torch.equal(
        asset.root_physx_view.get_dof_limits()[:, unchanged_ids],
        mechanical[:, unchanged_ids],
    )
    action.verify_physx_control_position_limit_readback()

    contract = action.physx_control_position_limits_contract()
    assert set(contract) == {
        "enabled",
        "selected_joint_names",
        "selected_joint_indices",
        "inset_fraction_per_side_hard_span",
        "unselected_joint_count",
        "joint_order",
        "mechanical_joint_pos_limits",
        "control_joint_pos_limits",
        "readback_sha256",
        "mechanical_edge_ledger_uses_h_mech",
        "soft_qdes_envelope_unchanged",
    }
    assert contract["enabled"] is True
    assert contract["selected_joint_names"] == _VENDOR_HCTRL_SELECTED_NAMES
    assert contract["selected_joint_indices"] == _VENDOR_HCTRL_TEST_INDICES
    assert contract["inset_fraction_per_side_hard_span"] == 0.02
    assert contract["unselected_joint_count"] == 27
    assert contract["joint_order"] == tuple(names)
    assert torch.equal(contract["mechanical_joint_pos_limits"], mechanical)
    assert torch.equal(contract["control_joint_pos_limits"], expected)
    assert contract["readback_sha256"] == (
        action.physx_control_position_limit_readback_sha256
    )
    assert contract["mechanical_edge_ledger_uses_h_mech"] is True
    assert contract["soft_qdes_envelope_unchanged"] is True
    contract["control_joint_pos_limits"].zero_()
    assert torch.equal(action.physx_control_joint_pos_limits, expected)

    asset.root_physx_view._limits[0, 0, 0] += 0.01
    with pytest.raises(RuntimeError, match="no longer equal H_ctrl"):
        action.verify_physx_control_position_limit_readback()


def test_vendor_physx_control_startup_verify_allows_calibrated_default_q(capsys):
    names = _vendor_hctrl_test_joint_names()
    action, _, asset = _action_and_env(
        num_envs=2,
        joint_count=31,
        guard=True,
        guard_margin_fraction=0.06,
        project_finite_qdes=True,
        physx_control_position_limit_inset_fraction=0.02,
        joint_names=names,
    )
    # Startup calibration is allowed to alter default q after the setter-time no-mutation proof.
    asset.data.default_joint_pos.add_(0.001)
    action.reset()
    output = capsys.readouterr().out
    assert "PHYSX CONTROL POSITION LIMITS STARTUP VERIFY" in output
    assert action._physx_control_runtime_verify_pending is False


def test_vendor_physx_control_envelope_rejects_missing_or_reordered_selected_joint():
    missing = _vendor_hctrl_test_joint_names()
    missing[19] = "j19"
    with pytest.raises(RuntimeError, match="every code-owned selected joint"):
        _action_and_env(
            num_envs=1,
            joint_count=31,
            guard=True,
            guard_margin_fraction=0.06,
            project_finite_qdes=True,
            physx_control_position_limit_inset_fraction=0.02,
            joint_names=missing,
        )

    reordered = _vendor_hctrl_test_joint_names()
    reordered[19], reordered[30] = reordered[30], reordered[19]
    with pytest.raises(RuntimeError, match="code-owned selected joint order"):
        _action_and_env(
            num_envs=1,
            joint_count=31,
            guard=True,
            guard_margin_fraction=0.06,
            project_finite_qdes=True,
            physx_control_position_limit_inset_fraction=0.02,
            joint_names=reordered,
        )


def test_vendor_physx_control_compact_diagnostic_covers_selected_joints_and_sides():
    names = _vendor_hctrl_test_joint_names()
    action, _, asset = _action_and_env(
        num_envs=1,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        guard_margin_fraction=0.06,
        project_finite_qdes=True,
        physx_control_position_limit_inset_fraction=0.02,
        action_ball_diagnostic_unauthorized=True,
        runtime_step_dt=0.02,
        joint_names=names,
    )
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None

    # All selected axes approach alternating H_ctrl sides and ballistic-cross in one horizon.
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    asset.data.joint_pos[0, 5] = -1.15
    asset.data.joint_vel[0, 5] = -1.0
    asset.data.joint_pos[0, 8] = 1.15
    asset.data.joint_vel[0, 8] = 1.0
    asset.data.joint_pos[0, 19] = -1.15
    asset.data.joint_vel[0, 19] = -1.0
    asset.data.joint_pos[0, 30] = 1.15
    asset.data.joint_vel[0, 30] = 1.0
    action.process_actions(torch.zeros(1, 31))
    _finish_guarded_policy_step(action, asset)
    first = action.consume_actual_joint_forbidden_diagnostic()[
        "physx_control_position_limits"
    ]
    assert first["enabled"] is True
    assert first["joint_order"] == list(_VENDOR_HCTRL_SELECTED_NAMES)
    assert first["semantics"].endswith("not a PhysX constraint impulse getter")
    assert first["readback_env_samples"] == 5
    assert first["total_ballistic_attempt_proxy_count"] == 20
    assert first["ballistic_attempt_proxy_rate"] == pytest.approx(0.5)
    by_joint = {row["joint"]: row["sides"] for row in first["by_joint"]}
    assert by_joint["waist_roll_joint"]["lower"]["near_ctrl_edge_readback"] == 5
    assert by_joint["waist_roll_joint"]["lower"]["ballistic_attempt_proxy"] == 5
    assert by_joint["waist_pitch_joint"]["upper"]["near_ctrl_edge_readback"] == 5
    assert by_joint["waist_pitch_joint"]["upper"]["ballistic_attempt_proxy"] == 5
    assert by_joint["left_ankle_roll_joint"]["lower"][
        "ballistic_attempt_proxy"
    ] == 5
    assert by_joint["right_ankle_roll_joint"]["upper"][
        "ballistic_attempt_proxy"
    ] == 5
    assert by_joint["waist_roll_joint"]["lower"]["ctrl_penetration_readback"] == 0
    assert by_joint["waist_pitch_joint"]["upper"]["ctrl_penetration_readback"] == 0
    assert by_joint["waist_roll_joint"]["lower"][
        "minimum_signed_mechanical_gap_rad"
    ] == pytest.approx(0.05, abs=1.0e-6)

    # Returning inside with non-outward velocity is reported only as a capture proxy.
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    action.process_actions(torch.zeros(1, 31))
    _finish_guarded_policy_step(action, asset)
    second = action.consume_actual_joint_forbidden_diagnostic()[
        "physx_control_position_limits"
    ]
    by_joint = {row["joint"]: row["sides"] for row in second["by_joint"]}
    assert by_joint["waist_roll_joint"]["lower"]["capture_proxy"] == 1
    assert by_joint["waist_pitch_joint"]["upper"]["capture_proxy"] == 1
    assert by_joint["left_ankle_roll_joint"]["lower"]["capture_proxy"] == 1
    assert by_joint["right_ankle_roll_joint"]["upper"]["capture_proxy"] == 1
    rows = {row["joint"]: row for row in second["by_joint"]}
    assert rows["waist_roll_joint"]["max_abs_delta_qdot_rad_s"] == pytest.approx(1.0)
    assert rows["waist_pitch_joint"]["max_abs_delta_qdot_rad_s"] == pytest.approx(1.0)


def test_vendor_physx_control_diagnostic_tracks_dwell_side_flip_and_qdot_jump():
    names = _vendor_hctrl_test_joint_names()
    action, _, asset = _action_and_env(
        num_envs=1,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        guard_margin_fraction=0.06,
        project_finite_qdes=True,
        physx_control_position_limit_inset_fraction=0.02,
        action_ball_diagnostic_unauthorized=True,
        runtime_step_dt=0.02,
        joint_names=names,
    )
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    asset.data.joint_pos[0, 5] = -1.16
    asset.data.joint_vel[0, 5] = -1.0
    action._record_physx_control_position_limit_diagnostic(
        joint_pos=asset.data.joint_pos, joint_vel=asset.data.joint_vel
    )
    action._record_physx_control_position_limit_diagnostic(
        joint_pos=asset.data.joint_pos, joint_vel=asset.data.joint_vel
    )
    asset.data.joint_pos[0, 5] = 1.16
    asset.data.joint_vel[0, 5] = 1.0
    action._record_physx_control_position_limit_diagnostic(
        joint_pos=asset.data.joint_pos, joint_vel=asset.data.joint_vel
    )
    payload = action.consume_actual_joint_forbidden_diagnostic()[
        "physx_control_position_limits"
    ]
    rows = {row["joint"]: row for row in payload["by_joint"]}
    roll = rows["waist_roll_joint"]
    assert roll["sides"]["lower"]["max_ctrl_penetration_dwell_readbacks"] == 2
    assert roll["sides"]["upper"]["ballistic_attempt_side_flip_proxy"] == 1
    assert roll["max_abs_delta_qdot_rad_s"] == pytest.approx(2.0)
    assert roll["sides"]["lower"][
        "minimum_signed_mechanical_gap_rad"
    ] == pytest.approx(0.04, abs=1.0e-6)
    assert roll["sides"]["upper"][
        "minimum_signed_mechanical_gap_rad"
    ] == pytest.approx(0.04, abs=1.0e-6)


def test_ordinary_action_does_not_touch_physx_position_limits():
    action, _, asset = _action_and_env()
    assert action.physx_control_position_limit_inset_fraction == 0.0
    assert action.physx_control_joint_pos_limits is None
    assert asset.root_physx_view.set_calls == []
    assert action.physx_control_position_limits_contract() == {"enabled": False}
    action.verify_physx_control_position_limit_readback()


@pytest.mark.parametrize("value", [True, 0, 0.01, 0.03, float("nan")])
def test_physx_control_envelope_rejects_non_code_owned_values(value):
    with pytest.raises(ValueError, match="exact code-owned float"):
        _action_and_env(
            physx_control_position_limit_inset_fraction=value,
        )


def test_physx_control_envelope_requires_exact_vendor_guard_and_projection():
    with pytest.raises(ValueError, match="finite q_des projection"):
        _action_and_env(
            guard=True,
            guard_margin_fraction=0.06,
            physx_control_position_limit_inset_fraction=0.02,
        )
    with pytest.raises(ValueError, match="exact six-percent"):
        _action_and_env(
            guard=True,
            guard_margin_fraction=0.05,
            project_finite_qdes=True,
            physx_control_position_limit_inset_fraction=0.02,
        )


def _set_sim_timestamp(asset, timestamp_s: float) -> None:
    asset.data._sim_timestamp = timestamp_s
    asset.data._joint_pos.timestamp = timestamp_s
    asset.data._joint_vel.timestamp = timestamp_s


def _finish_guarded_policy_step(action, asset):
    """Model the pinned Isaac order: 4 apply writes, then one post-step DoneTerm readback."""

    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    start = float(asset.data._sim_timestamp)
    physics_dt = float(action._pre_apply_guard_physics_dt_s)
    decimation = int(action._pre_apply_guard_decimation)
    for index in range(decimation):
        _set_sim_timestamp(asset, start + index * physics_dt)
        action.apply_actions()
    _set_sim_timestamp(asset, start + decimation * physics_dt)
    action.finalize_joint_safety_post_step_readback()


def _two_step_cross_reset_action_ball_ledger(
    *,
    policy_horizon_only_crossing: bool = False,
    reset_after_safe_step: bool = False,
    diagnostic_compact_evidence: bool = False,
):
    action, env, asset = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
        action_ball_diagnostic_unauthorized=(
            diagnostic_compact_evidence
        ),
    )
    receipts = ["a" * 64, "b" * 64]
    action_uids = torch.tensor([101, 202], dtype=torch.long)
    birth_generations = torch.tensor([7, 9], dtype=torch.long)
    swing_generations = torch.tensor([3, 4], dtype=torch.long)
    command = types.SimpleNamespace(
        action_ball_enabled=True,
        action_ball_episode_generation=birth_generations,
        action_ball_reset_generation=birth_generations,
        action_ball_swing_generation=swing_generations,
        action_ball_ordered_action_uids=(101, 202, 303),
        action_ball_action_uid_for_envs=lambda ids: action_uids[ids],
        action_ball_birth_receipt_sha256=lambda env_id: receipts[env_id],
        action_ball_hard_contract=lambda: {
            "action_uids": [101, 202, 303],
        },
    )
    env.command_manager = types.SimpleNamespace(
        active_terms=["racket_target"],
        get_term=lambda name: command if name == "racket_target" else None,
    )
    termination_masks = {
        name: torch.zeros(2, dtype=torch.bool)
        for name in (
            "base_fell_tilt",
            "base_too_low",
            "joint_actual_forbidden",
            "joint_qdes_forbidden",
            "robot_hit_table",
        )
    }
    env.termination_manager = types.SimpleNamespace(
        active_terms=list(termination_masks),
        terminated=torch.zeros(2, dtype=torch.bool),
        time_outs=torch.zeros(2, dtype=torch.bool),
        get_term=lambda name: termination_masks[name],
    )

    env.episode_length_buf[:] = torch.tensor([41, 17])
    if policy_horizon_only_crossing:
        # Physical +/-1.2 limits and zero inset: one 5 ms physics tick stays
        # inside (1.055), while the full 20 ms policy guard reaches 1.22.
        # The producer uses the latter receding-control horizon.
        asset.data.joint_pos[0, 0] = 1.0
        asset.data.joint_vel[0, 0] = 11.0
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    if reset_after_safe_step:
        asset.data.joint_pos.zero_()
        asset.data.joint_vel.zero_()
        env.episode_length_buf += 1
        action.process_actions(torch.zeros(2, 2))
        _finish_guarded_policy_step(action, asset)
    env.reset_terminated = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([False, False])
    env.episode_length_buf[0] = 43 if reset_after_safe_step else 42
    action.reset(env_ids=torch.tensor([0]))

    # The next policy step belongs to a new immutable birth for env 0.  Env 1 stays on the exact
    # previous identity, which exercises both sides of the runner's generation transition check.
    receipts[0] = "c" * 64
    action_uids[0] = 303
    birth_generations[0] = 8
    swing_generations[0] = 0
    if policy_horizon_only_crossing:
        asset.data.joint_pos.zero_()
        asset.data.joint_vel.zero_()
    env.episode_length_buf[:] = torch.tensor(
        [0, 19 if reset_after_safe_step else 18]
    )
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    return action, env


def test_forbidden_zone_exact_edges_are_closed_and_nearby_interior_is_allowed():
    # Limits [-1, 1], per-side inset = 0.1 rad + 10% * 2 rad = 0.3 rad.
    # The allowed interval is OPEN (-0.7, 0.7).
    positions = torch.tensor([[-0.7000, -0.6999, 0.6999, 0.7000]])
    limits = _limits(1, 4)
    per_joint = terminations_mod.joint_position_forbidden_zone_per_joint(
        positions,
        limits,
        margin_rad=0.1,
        margin_fraction=0.1,
    )
    assert per_joint.tolist() == [[True, False, False, True]]
    assert terminations_mod.joint_position_forbidden_zone_mask(
        positions,
        limits,
        margin_rad=0.1,
        margin_fraction=0.1,
    ).tolist() == [True]


def test_forbidden_zone_nonfinite_and_invalid_envelopes_fail_safe():
    positions = torch.tensor(
        [
            [0.0, float("nan"), 0.0, 0.0],
            [float("inf"), 0.0, 0.0, 0.0],
        ]
    )
    limits = _limits(2, 4)
    limits[0, 2, 0] = float("nan")
    limits[0, 3] = torch.tensor([0.5, 0.5])  # zero travel
    limits[1, 2] = torch.tensor([1.0, -1.0])  # reversed
    # 0.6 rad on a 2-rad envelope consumes the remaining interval when added per side.
    limits[1, 3] = torch.tensor([-0.5, 0.5])
    per_joint = terminations_mod.joint_position_forbidden_zone_per_joint(
        positions,
        limits,
        margin_rad=0.6,
        margin_fraction=0.0,
    )
    assert per_joint.tolist() == [
        [False, True, True, True],
        [True, False, True, True],
    ]


@pytest.mark.parametrize(
    ("margin_rad", "margin_fraction"),
    [
        (-0.01, 0.0),
        (float("nan"), 0.0),
        (0.0, -0.01),
        (0.0, 0.5),
        (0.0, float("inf")),
        (True, 0.0),
        (0.0, False),
        ("0.0", 0.0),
    ],
)
def test_forbidden_zone_rejects_implicit_or_invalid_margins(
    margin_rad, margin_fraction
):
    with pytest.raises(ValueError, match="margin"):
        terminations_mod.joint_position_forbidden_zone_mask(
            torch.zeros(1, 1),
            _limits(1, 1),
            margin_rad=margin_rad,
            margin_fraction=margin_fraction,
        )


def test_forbidden_zone_rejects_broadcast_and_dtype_shape_shortcuts():
    position = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="exactly match"):
        terminations_mod.joint_position_forbidden_zone_mask(
            position,
            _limits(1, 2),  # broadcasting across environments is forbidden
            margin_rad=0.0,
            margin_fraction=0.0,
        )
    with pytest.raises(RuntimeError, match="floating"):
        terminations_mod.joint_position_forbidden_zone_mask(
            position.to(torch.int64),
            _limits(2, 2).to(torch.int64),
            margin_rad=0.0,
            margin_fraction=0.0,
        )
    with pytest.raises(RuntimeError, match="identical device and dtype"):
        terminations_mod.joint_position_forbidden_zone_mask(
            position,
            _limits(2, 2).to(torch.float64),
            margin_rad=0.0,
            margin_fraction=0.0,
        )


def test_pre_clamp_qdes_is_affine_target_and_clamp_cannot_hide_violation():
    offset = torch.tensor([[0.10, -0.20], [0.10, -0.20]])
    action, env, asset = _action_and_env(scale=2.0, offset=offset, clamp=True)
    normalized_actor_action = torch.tensor([[0.70, 0.00], [0.00, 0.00]])
    action.process_actions(normalized_actor_action)

    # env0 asks for 0.10 + 2*0.70 = 1.50 rad.  The PD target is clamped to 1.00, but the
    # safety source preserves 1.50; it is neither normalized raw action (0.70) nor clamped (1.00).
    assert action.pre_clamp_qdes[:, 0].tolist() == pytest.approx([1.50, 0.10])
    assert action.processed_actions[:, 0].tolist() == pytest.approx([1.00, 0.10])
    assert action.pre_clamp_qdes_valid.tolist() == [True, True]

    # Realized q is still harmless.  Only the pre-clamp termination exposes the unsafe command.
    asset.data.joint_pos.zero_()
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]


def test_pre_clamp_validity_is_reset_per_environment_and_nonfinite_fails_safe():
    action, env, _ = _action_and_env()
    action.process_actions(torch.tensor([[0.25, 0.0], [0.25, 0.0]]))
    action.reset(env_ids=torch.tensor([0]))
    assert action.pre_clamp_qdes_valid.tolist() == [False, True]
    assert action.pre_clamp_qdes[0].tolist() == [0.0, 0.0]
    # Stale/reset-invalid env0 never terminates.
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]

    action.process_actions(
        torch.tensor([[float("nan"), 0.0], [float("inf"), 0.0]])
    )
    # Evidence remains non-finite for the DoneTerm, while the target that apply_actions/PhysX sees
    # falls back to the preceding finite deploy-space target.
    assert torch.isnan(action.pre_clamp_qdes[0, 0])
    assert torch.isinf(action.pre_clamp_qdes[1, 0])
    assert torch.all(torch.isfinite(action.processed_actions))
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.0, 0.25])
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, True]


def test_nonfinite_first_qdes_after_reset_uses_finite_default_before_physics():
    offset = torch.tensor([[0.30, -0.20], [0.10, -0.40]])
    action, _, _ = _action_and_env(offset=offset)
    action.process_actions(torch.zeros(2, 2))
    action.reset(env_ids=torch.tensor([0]))
    action.process_actions(
        torch.tensor([[float("nan"), 0.0], [float("nan"), 0.0]])
    )
    # env0 has no valid history after reset -> current default 0.30.  env1 keeps its preceding
    # processed target 0.10.  Neither non-finite request reaches apply_actions.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.30, 0.10])
    assert torch.all(torch.isfinite(action.processed_actions))


def test_nonfinite_qdes_with_no_finite_fallback_fails_before_apply():
    action, _, asset = _action_and_env(num_envs=1, joint_count=1)
    asset.data.default_joint_pos[:] = float("nan")
    with pytest.raises(RuntimeError, match="no finite q_des fallback"):
        action.process_actions(torch.tensor([[float("nan")]]))


def test_pre_apply_guard_triggers_on_outward_but_not_inward_ballistic_velocity():
    action, env, asset = _action_and_env(guard=True)
    # Both are near the upper soft edge, but the terminal envelope is the physical +/-1.2 hard
    # limit.  Over one explicit 0.1-s policy horizon env0 moves outward to 1.25 and must brake;
    # env1 moves inward to 0.65 and remains eligible.
    asset.data.joint_pos[:, 0] = torch.tensor([0.95, 0.95])
    asset.data.joint_vel[:, 0] = torch.tensor([3.0, -3.0])
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    assert action.pre_apply_crossing_violation_latch.tolist() == [True, False]
    assert action.pre_apply_joint_safety_latch.tolist() == [True, False]
    assert action.pre_apply_crossing_violation_joint_latch.tolist() == [
        [True, False],
        [False, False],
    ]
    assert action.pre_apply_crossing_violation_joint_count.tolist() == [
        [1, 0],
        [0, 0],
    ]
    # Derived brake target q-qdot*dt = 0.65.  The inward env keeps its nominal zero target.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.65, 0.0])
    assert torch.all(torch.isfinite(action.processed_actions))
    assert torch.all(action.processed_actions >= -1.0)
    assert torch.all(action.processed_actions <= 1.0)

    # The DoneTerm consumes the pre-physics sticky latch even though proposed qdes itself was safe.
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]


def test_pre_apply_guard_latches_raw_qdes_and_never_sends_boundary_request():
    action, env, asset = _action_and_env(guard=True)
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    action.process_actions(torch.tensor([[1.20, 0.0], [0.25, 0.0]]))

    assert action.pre_clamp_qdes[:, 0].tolist() == pytest.approx([1.20, 0.25])
    assert action.pre_apply_qdes_violation_latch.tolist() == [True, False]
    assert action.pre_apply_qdes_violation_joint_count.tolist() == [
        [1, 0],
        [0, 0],
    ]
    # The violating joint gets the derived stationary brake target q=0, not the soft edge.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.0, 0.25])
    assert torch.all(torch.isfinite(action.processed_actions))
    _finish_guarded_policy_step(action, asset)
    # Legacy/default tasks retain their historical finite-q_des termination behavior.
    assert action.finite_preclamp_qdes_projection_enabled is False
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]


def test_action_ball_finite_qdes_projection_is_dense_shaping_not_reset():
    action, env, asset = _action_and_env(
        num_envs=2,
        joint_count=31,
        guard=True,
        guard_margin_fraction=0.02,
        project_finite_qdes=True,
    )
    proposals = torch.zeros(2, 31)
    proposals[:, 0] = torch.tensor([-1.20, 2.00])
    action.process_actions(proposals)

    # Both finite requests cross the physical hard-inner edge, but the drive sees the nearest
    # ActionBall execution-envelope projection rather than the state-derived brake target.
    # A finite saturation is not copied into the formal hard-safety ledger; otherwise that
    # ledger would demand a terminal archive and fence PPO despite the deliberate no-reset path.
    assert action.pre_apply_qdes_violation_latch.tolist() == [False, False]
    assert action.pre_apply_qdes_violation_joint_count.sum().item() == 0
    assert action.processed_actions[:, 0].tolist() == pytest.approx([-0.9, 0.9])
    assert action.nominal_projected_qdes[:, 0].tolist() == pytest.approx(
        [-0.9, 0.9]
    )
    assert action.nominal_projection_span[:, 0].tolist() == pytest.approx(
        [1.8, 1.8]
    )
    assert torch.all(torch.isfinite(action.processed_actions))
    _finish_guarded_policy_step(action, asset)
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.02,
    ).tolist() == [False, False]

    values = hope_rewards_mod.qdes_projection_penalty(
        env, action_name="joint_pos", shape_rate=4.0
    )
    assert values[0].item() > 0.0
    assert values[1].item() > values[0].item()
    expected = 1.0 - torch.exp(
        -4.0 * torch.tensor([0.3 / 1.8, 1.1 / 1.8])
    )
    assert values.tolist() == pytest.approx(expected.tolist())

    # Explicit zero uses a unit RewardManager weight plus the objective dose
    # inside the callable.  It must return exact zero while preserving the
    # same unweighted exposure ledger for causal comparison.
    zero_values = hope_rewards_mod.qdes_projection_penalty(
        env,
        action_name="joint_pos",
        shape_rate=4.0,
        objective_weight=0.0,
    )
    assert zero_values.tolist() == [0.0, 0.0]
    ablation_magnitude = hope_rewards_mod.qdes_projection_penalty(
        env,
        action_name="joint_pos",
        shape_rate=4.0,
        objective_weight=-2.5,
    )
    assert ablation_magnitude.tolist() == pytest.approx(
        (2.5 * expected).tolist()
    )

    counters = getattr(
        env, hope_rewards_mod._QDES_PROJECTION_ACTIVATION_ATTR
    )
    assert counters["observed_sample_count"].item() == 2
    assert counters["projection_sample_count"].item() == 2
    assert counters["nonfinite_sample_count"].item() == 0
    assert counters["projection_joint_count"][0].item() == 2
    assert counters["projection_joint_count"][1:].sum().item() == 0
    assert counters["lower_projection_joint_count"][0].item() == 1
    assert counters["upper_projection_joint_count"][0].item() == 1
    assert counters["normalized_projection_distance_sum"][0].item() == pytest.approx(
        0.3 / 1.8 + 1.1 / 1.8
    )
    assert counters["normalized_projection_distance_max"][0].item() == pytest.approx(
        1.1 / 1.8
    )
    assert counters["max_normalized_projection_distance"].item() == pytest.approx(
        1.1 / 1.8
    )
    assert counters["penalty_value_sum"].item() == pytest.approx(expected.sum().item())

    # The existing once-per-update q_des ledger consumer also exports scalar per-joint side/count,
    # mean and max telemetry, so the runner need not learn a second logging protocol.
    hope_rewards_mod._soft_limit_barrier_v2_counter_state(
        env,
        hope_rewards_mod._QDES_LIMIT_BARRIER_V2_ACTIVATION_ATTR,
        values,
    )
    hope_rewards_mod._soft_limit_barrier_v2_counter_state(
        env,
        hope_rewards_mod._ACTUAL_LIMIT_BARRIER_V2_ACTIVATION_ATTR,
        values,
    )
    snapshot = hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(
        env
    )
    assert snapshot["projection_joint_00_trigger_count"].item() == 2
    assert snapshot[
        "projection_joint_00_saturation_env_step_count"
    ].item() == 2
    assert snapshot[
        "projection_joint_00_saturation_env_step_ratio"
    ].item() == pytest.approx(1.0)
    assert snapshot[
        "projection_joint_00_lower_saturation_env_step_count"
    ].item() == 1
    assert snapshot[
        "projection_joint_00_upper_saturation_env_step_count"
    ].item() == 1
    assert snapshot[
        "projection_joint_00_mean_normalized_excess"
    ].item() == pytest.approx((0.3 / 1.8 + 1.1 / 1.8) / 2.0)
    assert snapshot[
        "projection_joint_00_max_normalized_excess"
    ].item() == pytest.approx(1.1 / 1.8)
    assert snapshot[
        "projection_mean_normalized_projection_distance"
    ].item() == pytest.approx((0.3 / 1.8 + 1.1 / 1.8) / 2.0)
    assert snapshot["projection_saturation_sample_step_ratio"].item() == pytest.approx(
        1.0
    )


def test_action_ball_projection_inset_is_identity_inside_and_legacy_is_unchanged():
    action, _, _ = _action_and_env(
        num_envs=3,
        joint_count=2,
        guard=True,
        project_finite_qdes=True,
    )
    proposals = torch.tensor(
        [
            [0.85, -0.85],  # strictly inside the inset execution envelope
            [0.95, -0.95],  # inside soft limits, outside the execution envelope
            [0.90, -0.90],  # exact execution-envelope edges remain executable
        ]
    )
    action.process_actions(proposals)

    assert torch.equal(action.raw_actions, proposals)
    assert torch.equal(action.pre_clamp_qdes, proposals)
    assert torch.equal(action.processed_actions[0], proposals[0])
    assert action.processed_actions[1].tolist() == pytest.approx([0.9, -0.9])
    assert action.processed_actions[2].tolist() == pytest.approx([0.9, -0.9])
    assert torch.equal(action.nominal_projected_qdes, action.processed_actions)
    assert torch.allclose(
        action.nominal_projection_span,
        torch.full_like(action.nominal_projection_span, 1.8),
    )

    legacy, _, _ = _action_and_env(
        num_envs=1,
        joint_count=2,
        guard=True,
        project_finite_qdes=False,
        target_mode="hitter",
    )
    legacy_proposal = torch.tensor([[0.95, -0.95]])
    legacy.process_actions(legacy_proposal)
    assert torch.equal(legacy.raw_actions, legacy_proposal)
    assert torch.equal(legacy.pre_clamp_qdes, legacy_proposal)
    assert torch.equal(legacy.processed_actions, legacy_proposal)
    assert torch.equal(
        legacy.nominal_projection_span,
        torch.full_like(legacy.nominal_projection_span, 2.0),
    )


def test_action_ball_projection_only_qdes_nonfinite_is_terminal_and_actual_owns_limit():
    action, env, asset = _action_and_env(
        num_envs=3,
        joint_count=2,
        guard=True,
        guard_margin_fraction=0.02,
        project_finite_qdes=True,
    )
    asset.data.joint_pos[:, 0] = torch.tensor([0.0, 0.95, 1.16])
    asset.data.joint_vel[:, 0] = torch.tensor([0.0, 3.0, 0.0])
    proposals = torch.zeros(3, 2)
    proposals[0, 0] = float("nan")
    action.process_actions(proposals)
    _finish_guarded_policy_step(action, asset)

    # env0 is a non-finite actor output; env1 predicts a ballistic hard-inner crossing; env2 is
    # already inside the two-percent physical inner band.  Both physical rows still receive a
    # finite brake target, but the q_des term owns only the non-finite request.  The recoverable
    # env2 inner-band state is trained by the actual-q barrier rather than reset; only a raw
    # mechanical hard edge belongs to the actual Done term.
    assert action.physical_hard_safety_latch.tolist() == [False, True, True]
    assert torch.all(torch.isfinite(action.processed_actions))
    assert action.processed_actions[:, 0].tolist() == pytest.approx(
        [0.0, 0.65, 0.9]
    )
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.02,
    ).tolist() == [True, False, False]
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "joint_pos_limits",
        0.0,
        0.02,
    ).tolist() == [False, False, False]


def test_action_ball_actual_diagnostic_separates_inner_event_from_raw_hard_terminal():
    action, env, asset = _action_and_env(
        num_envs=2,
        joint_count=2,
        guard=True,
        guard_margin_fraction=0.02,
        project_finite_qdes=True,
        action_ball_diagnostic_unauthorized=True,
    )
    env.episode_length_buf[:] = torch.tensor([17, 18])
    asset.data.joint_pos[:, 0] = torch.tensor([-1.16, -1.21])
    asset.data.joint_vel.zero_()
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "joint_pos_limits",
        0.0,
        0.02,
    ).tolist() == [False, True]

    payload = action.consume_actual_joint_forbidden_diagnostic()
    assert payload["total_safety_event_count"] == 2
    assert payload["total_hard_terminal_count"] == 1
    later = payload["age_buckets"]["episode_age_gt_1"]
    assert later["safety_event_count"] == 2
    assert later["hard_terminal_count"] == 1
    assert later["mean_safety_event_episode_age"] == pytest.approx(17.5)
    assert later["by_joint"][0]["joint"] == "j0"
    assert later["by_joint"][0]["counts"]["current_lower"] == 2
    assert (
        later["by_joint"][0]["counts"]["substep_actual_hard_edge"] == 1
    )


def test_action_ball_nonfinite_request_has_finite_projection_reward_before_reset():
    action, env, asset = _action_and_env(
        num_envs=1,
        joint_count=31,
        guard=True,
        guard_margin_fraction=0.02,
        project_finite_qdes=True,
    )
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    proposal = torch.zeros(1, 31)
    proposal[0, 0] = float("nan")
    action.process_actions(proposal)

    assert torch.all(torch.isfinite(action.processed_actions))
    assert torch.all(torch.isfinite(action.nominal_projected_qdes))
    value = hope_rewards_mod.qdes_projection_penalty(
        env, action_name="joint_pos", shape_rate=4.0
    )
    assert torch.isfinite(value).all()
    assert value.item() == pytest.approx(1.0 - math.exp(-4.0))
    counters = getattr(
        env, hope_rewards_mod._QDES_PROJECTION_ACTIVATION_ATTR
    )
    assert counters["nonfinite_sample_count"].item() == 1
    assert counters["projection_joint_count"][0].item() == 1
    assert counters["projection_joint_count"][1:].sum().item() == 0


def test_pre_apply_guard_soft_only_qdes_intrusion_clamps_but_does_not_terminate():
    action, env, asset = _action_and_env(guard=True)
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    # 1.10 is outside deploy soft +1.0 but still strictly inside physical hard +1.2.
    action.process_actions(torch.tensor([[1.10, 0.0], [0.25, 0.0]]))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [False, False]
    assert action.processed_actions[:, 0].tolist() == pytest.approx([1.0, 0.25])
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]


def test_pre_apply_guard_nonfinite_state_latches_and_uses_finite_brake_target():
    offset = torch.tensor([[0.30, -0.20], [0.10, -0.40]])
    action, _, asset = _action_and_env(offset=offset, guard=True)
    asset.data.joint_pos[0, 0] = float("nan")
    asset.data.joint_vel[1, 0] = float("inf")
    action.process_actions(torch.zeros(2, 2))
    assert action.pre_apply_crossing_violation_latch.tolist() == [True, True]
    assert torch.all(torch.isfinite(action.processed_actions))
    # q NaN falls back to default 0.30; qdot Inf becomes zero.  Both remain inside soft limits.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.30, 0.0])


def test_pre_apply_guard_uses_closed_two_percent_physical_hard_inset():
    action, _, asset = _action_and_env(
        guard=True, guard_margin_fraction=0.02
    )
    hard = asset.data.joint_pos_limits[0, 0]
    inner_upper = hard[1] - 0.02 * (hard[1] - hard[0])
    asset.data.joint_pos[:, 0] = torch.stack(
        (inner_upper - 1.0e-4, inner_upper)
    )
    asset.data.joint_vel.zero_()
    action.process_actions(torch.zeros(2, 2))
    # The remaining interval is open: an exact 2%-inset edge is forbidden.
    assert action.pre_apply_crossing_violation_latch.tolist() == [False, True]


def test_pre_apply_guard_sticky_state_counts_and_partial_reset():
    action, _, asset = _action_and_env(guard=True)
    asset.data.joint_pos[:, 0] = torch.tensor([0.95, 0.0])
    asset.data.joint_vel[:, 0] = torch.tensor([3.0, 0.0])
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [True, False]

    # A subsequent safe proposal/state cannot erase forensic evidence before episode reset.
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [True, False]
    assert action.pre_apply_crossing_violation_joint_count[:, 0].tolist() == [1, 0]

    # Make env1 violate too, then prove partial reset clears only env0's latch/counters.
    action.process_actions(torch.tensor([[0.0, 0.0], [1.2, 0.0]]))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [True, True]
    action.reset(env_ids=torch.tensor([0]))
    assert action.pre_apply_joint_safety_latch.tolist() == [False, True]
    assert action.pre_apply_qdes_violation_joint_count[0].tolist() == [0, 0]
    assert action.pre_apply_qdes_violation_joint_count[1, 0].item() == 1
    assert action.pre_apply_crossing_violation_joint_count[0].tolist() == [0, 0]


def test_pre_apply_guard_is_opt_in_and_legacy_finite_target_is_unchanged():
    action, _, asset = _action_and_env(guard=False)
    asset.data.joint_pos[:, 0] = 0.95
    asset.data.joint_vel[:, 0] = 1.0
    proposal = torch.tensor([[0.90, 0.0], [1.20, 0.0]])
    action.process_actions(proposal)
    # Guard-off preserves the old deploy clamp exactly: safe 0.90 stays 0.90; 1.20 clamps to 1.0.
    assert torch.equal(
        action.processed_actions,
        torch.tensor([[0.90, 0.0], [1.00, 0.0]]),
    )
    assert action.pre_apply_joint_safety_latch.tolist() == [False, False]
    assert action.pre_apply_qdes_violation_joint_count.sum().item() == 0
    assert action.pre_apply_crossing_violation_joint_count.sum().item() == 0


def test_legacy_apply_actions_dispatch_is_unchanged_and_ledger_stays_disabled(monkeypatch):
    dispatched = []

    def fake_apply(term):
        dispatched.append(term.processed_actions.detach().clone())

    monkeypatch.setattr(
        hope_actions_mod.JointPositionAction,
        "apply_actions",
        fake_apply,
        raising=False,
    )
    action, _, _ = _action_and_env(guard=False)
    action.process_actions(torch.tensor([[0.25, 0.0], [1.2, 0.0]]))
    expected = action.processed_actions.detach().clone()
    action.apply_actions()
    assert len(dispatched) == 1
    assert torch.equal(dispatched[0], expected)
    assert action.joint_safety_ledger_snapshot() == {
        "schema_version": 1,
        "enabled": False,
    }


def test_legacy_reset_delegates_original_ids_without_safety_archive(monkeypatch):
    action, _, _ = _action_and_env(guard=False)
    # Sentinels make any accidental containment indexing visible.  These buffers are inert in the
    # legacy mode and reset must not add three device kernels to that established hot path.
    action._max_inward_direction_latch.fill_(1)
    action._max_inward_release_hold.fill_(True)
    action._max_inward_release_qdes.fill_(0.5)
    calls = []

    def fake_reset(term, env_ids=None):
        calls.append(env_ids)

    monkeypatch.setattr(
        hope_actions_mod.JointPositionAction,
        "reset",
        fake_reset,
        raising=False,
    )
    ids = torch.tensor([1])
    action.reset(env_ids=ids)
    assert len(calls) == 1
    assert calls[0] is ids
    assert action._max_inward_direction_latch.eq(1).all().item()
    assert action._max_inward_release_hold.all().item()
    assert action._max_inward_release_qdes.eq(0.5).all().item()
    assert action.joint_safety_ledger_snapshot() == {
        "schema_version": 1,
        "enabled": False,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clamp": False, "guard_policy_dt_s": 0.1},
        {"guard_policy_dt_s": None},
        {"guard_policy_dt_s": 0.2},
        {"guard_margin_rad": None},
        {"guard_margin_rad": -0.01},
        {"guard_margin_fraction": None},
        {"guard_margin_fraction": 0.5},
        {"expected_decimation": None},
        {"decimation": 2, "expected_decimation": 4},
        {"terminal_archive_capacity": None},
        {"terminal_archive_capacity": 0},
    ],
)
def test_pre_apply_guard_requires_explicit_runtime_matched_contract(kwargs):
    with pytest.raises(ValueError, match="pre_apply"):
        _action_and_env(guard=True, **kwargs)


def test_pre_apply_guard_rejects_soft_envelope_outside_physical_hard_limits():
    action, _, asset = _action_and_env(guard=True)
    asset.data.joint_pos_limits[:] = _limits(2, 2, lo=-0.5, hi=0.5)
    with pytest.raises(RuntimeError, match="soft deploy envelope"):
        action.process_actions(torch.zeros(2, 2))


def test_physics_substep_ledger_records_exact_decimation_and_fresh_timestamps():
    action, _, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    physics_dt = 0.025
    for index, q_value in enumerate((0.00, 0.10, 0.20, 0.30)):
        _set_sim_timestamp(asset, index * physics_dt)
        asset.data.joint_pos[:, 0] = q_value
        asset.data.joint_vel[:, 0] = 0.5
        action.apply_actions()
    _set_sim_timestamp(asset, 4 * physics_dt)
    asset.data.joint_pos[:, 0] = 0.40
    action.finalize_joint_safety_post_step_readback()

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["expected_apply_calls"] == 4
    assert snapshot["apply_call_count"] == 4
    assert snapshot["post_readback_count"] == 1
    assert snapshot["complete"] is True
    assert snapshot["record_kind"] == ("apply", "apply", "apply", "apply", "post")
    assert snapshot["call_index"] == (0, 1, 2, 3, 4)
    assert snapshot["timestamp_s"] == pytest.approx(
        (0.0, 0.025, 0.05, 0.075, 0.1)
    )
    assert snapshot["q"][:, 0, 0].tolist() == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4]
    )
    # hard upper is +1.2, so the final readback gap is 0.8.
    assert snapshot["hard_upper_gap"][-1, 0, 0].item() == pytest.approx(0.8)
    assert snapshot["env_valid"].all().item()

    # Snapshot tensors are detached clones, not a mutation channel into the live ledger.
    snapshot["q"].fill_(99.0)
    assert action.joint_safety_ledger_snapshot()["q"][0, 0, 0].item() == 0.0


def test_physics_substep_guard_keeps_policy_horizon_and_preserves_safe_target():
    action, _, asset = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        guard_margin_fraction=0.1,
        runtime_step_dt=0.02,
        runtime_physics_dt=0.005,
    )
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.full((2, 2), 0.25))
    safe_target_before = action.processed_actions[1].detach().clone()

    # apply0 starts safe.  This models the important failure mode: the implicit drive/plant
    # accelerates outward after the policy-step check rather than arriving with a large qdot.
    _set_sim_timestamp(asset, 0.0)
    asset.data.joint_pos[0, 0] = 0.20
    asset.data.joint_vel[0, 0] = 0.0
    asset.data.joint_pos[1] = torch.tensor([0.20, -0.30])
    asset.data.joint_vel[1] = torch.tensor([0.10, -0.10])
    action.apply_actions()
    assert not action.physics_substep_hard_crossing_joint_latch[0, 0].item()
    assert torch.equal(action.processed_actions[1], safe_target_before)

    _set_sim_timestamp(asset, 0.005)
    # The 10%-inset hard upper edge is 0.96.  Env 0 remains safe over one 5-ms
    # physics tick (0.90 + 4*0.005 = 0.92), but crosses over the validated
    # 20-ms policy/reaction horizon (0.98) after accelerating between substeps,
    # and must receive q-v*0.02 = 0.82.
    asset.data.joint_pos[0, 0] = 0.90
    asset.data.joint_vel[0, 0] = 4.0
    asset.data.joint_pos[1] = torch.tensor([0.20, -0.30])
    asset.data.joint_vel[1] = torch.tensor([0.10, -0.10])
    action.apply_actions()

    assert action.physics_substep_hard_crossing_joint_latch[0, 0].item()
    assert action.processed_actions[0, 0].item() == pytest.approx(0.82)
    assert torch.equal(action.processed_actions[1], safe_target_before)
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["apply_call_count"] == 2
    assert snapshot["timestamp_s"] == pytest.approx((0.0, 0.005))


def test_lag_two_nominal_queue_cannot_delay_or_absorb_immediate_safety_brake(
    monkeypatch,
):
    """A plant-state brake is a post-queue drive override, never another delayed action."""

    action, _, asset = _action_and_env(
        num_envs=2,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        guard_margin_fraction=0.05,
        project_finite_qdes=True,
        runtime_step_dt=0.02,
        runtime_physics_dt=0.005,
        control_step_action_delay_min=2,
        control_step_action_delay_max=2,
    )
    action.reset()
    # Both lag-2 rows are initially filled with normalized zero (default q_des).  Only env 0 has
    # an outward state whose 20-ms ballistic projection crosses the 5%-inset hard envelope.
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    asset.data.joint_pos[0, 0] = 0.95
    asset.data.joint_vel[0, 0] = 10.0
    actor_action = torch.zeros(2, 31)
    actor_action[:, 0] = -0.60

    action.process_actions(actor_action)

    # The actor command was enqueued at age zero while lag two emitted the older zero hold.
    assert action.control_step_action_delay_lag_steps.tolist() == [2, 2]
    assert torch.equal(action._policy_action_delay._history[:, 0, :], actor_action)
    assert torch.count_nonzero(action._policy_action_delay._history[:, 1, :]).item() == 0
    assert torch.equal(action.raw_actions, actor_action)
    # env 0 nevertheless gets the immediate q-v*T brake target 0.75; the safe env keeps the
    # actually-due zero/default target.  Thus the brake is neither the current nor delayed actor row.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.75, 0.0])
    assert action.pre_apply_crossing_violation_latch.tolist() == [True, False]

    history_before_apply = action._policy_action_delay._history.clone()
    dispatched = []

    def capture_apply(term):
        dispatched.append(term.processed_actions.detach().clone())

    monkeypatch.setattr(
        hope_actions_mod.JointPositionAction,
        "apply_actions",
        capture_apply,
        raising=False,
    )
    _set_sim_timestamp(asset, 0.0)
    action.apply_actions()

    # The first physics write sees the brake immediately even though nominal lag remains two, and
    # the substep guard never feeds that brake back into the policy-delay history.
    assert len(dispatched) == 1
    assert dispatched[0][:, 0].tolist() == pytest.approx([0.75, 0.0])
    assert action.control_step_action_delay_lag_steps.tolist() == [2, 2]
    assert torch.equal(action._policy_action_delay._history, history_before_apply)
    assert not action.physics_substep_actual_hard_edge_latch.any().item()


@pytest.mark.parametrize(
    ("position", "velocity", "expected_direction", "expected_target"),
    [
        (-0.95, -3.0, 1, 1.0),
        (0.95, 3.0, -1, -1.0),
    ],
)
def test_max_inward_containment_is_lower_upper_symmetric(
    position, velocity, expected_direction, expected_target
):
    action, _, asset = _action_and_env(
        guard=True,
        guard_margin_fraction=0.05,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    asset.data.joint_pos[:, 0] = position
    asset.data.joint_vel[:, 0] = velocity
    action.process_actions(torch.zeros(2, 2))

    assert action.max_inward_joint_safety_containment_enabled is True
    assert action.max_inward_direction_latch[:, 0].tolist() == [
        expected_direction,
        expected_direction,
    ]
    assert action.processed_actions[:, 0].tolist() == pytest.approx(
        [expected_target, expected_target]
    )
    assert not action.max_inward_release_hold.any().item()


def test_max_inward_containment_latches_across_process_actions_calls():
    action, _, asset = _action_and_env(
        guard=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    asset.data.joint_pos[:, 0] = 0.95
    asset.data.joint_vel[:, 0] = 3.0
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    # The next state no longer predicts a crossing, but velocity has not reversed.  A fresh actor
    # proposal therefore cannot erase the episode-local side latch.
    asset.data.joint_pos[:, 0] = 0.50
    asset.data.joint_vel[:, 0] = 0.10
    proposal = torch.full((2, 2), 0.75)
    action.process_actions(proposal)
    assert action.max_inward_direction_latch[:, 0].tolist() == [-1, -1]
    assert action.processed_actions[:, 0].tolist() == pytest.approx([-1.0, -1.0])
    assert torch.equal(action.raw_actions, proposal)


def test_max_inward_release_requires_clearance_then_q_hold_until_next_policy():
    action, _, asset = _action_and_env(
        guard=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    asset.data.joint_pos[:, 0] = 1.05
    asset.data.joint_vel[:, 0] = 2.0
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    # qdot has reversed, but q=1.05 is still outside target_upper=1.0, so the first write remains
    # maximum inward.  The second readback is inside the executable envelope and captures q_hold.
    asset.data.joint_pos[:, 0] = 1.05
    asset.data.joint_vel[:, 0] = -0.2
    proposal = torch.full((2, 2), 0.25)
    action.process_actions(proposal)
    start = float(asset.data._sim_timestamp)
    _set_sim_timestamp(asset, start)
    action.apply_actions()
    assert action.processed_actions[:, 0].tolist() == pytest.approx([-1.0, -1.0])
    assert not action.max_inward_release_hold[:, 0].any().item()

    asset.data.joint_pos[:, 0] = 0.99
    asset.data.joint_vel[:, 0] = -0.2
    _set_sim_timestamp(asset, start + 0.025)
    action.apply_actions()
    assert action.max_inward_release_hold[:, 0].all().item()
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.99, 0.99])

    # A checkpoint taken after the q_hold write owns that cross-policy state.  Strict restore must
    # reproduce it, then clear it exactly at the resumed next-policy boundary.
    hold_state = action.action_delay_exact_resume_state_dict()
    resumed, _, resumed_asset = _action_and_env(
        guard=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    assert resumed.control_step_action_delay_enabled is False
    assert resumed.action_runtime_state_required is True
    resumed.load_action_delay_exact_resume_state_dict(hold_state, strict=True)
    assert resumed.max_inward_release_hold[:, 0].all().item()
    resumed_asset.data.joint_pos[:, 0] = 0.99
    resumed_asset.data.joint_vel[:, 0] = -0.2
    resumed.process_actions(proposal)
    assert not resumed.max_inward_release_hold[:, 0].any().item()
    assert resumed.max_inward_direction_latch[:, 0].tolist() == [0, 0]
    assert resumed.processed_actions[:, 0].tolist() == pytest.approx([0.25, 0.25])

    _set_sim_timestamp(asset, start + 0.05)
    action.apply_actions()
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.99, 0.99])
    _set_sim_timestamp(asset, start + 0.075)
    action.apply_actions()
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.99, 0.99])
    _set_sim_timestamp(asset, start + 0.1)
    action.finalize_joint_safety_post_step_readback()

    # The release latch clears only here, at the following policy boundary.  The newly due actor
    # target is then restored without ever entering raw-action or delay history as a brake row.
    action.process_actions(proposal)
    assert action.max_inward_direction_latch[:, 0].tolist() == [0, 0]
    assert not action.max_inward_release_hold[:, 0].any().item()
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.25, 0.25])
    assert torch.equal(action.raw_actions, proposal)


@pytest.mark.parametrize(
    (
        "first_position",
        "first_velocity",
        "opposite_position",
        "opposite_velocity",
        "first_direction",
        "opposite_direction",
        "opposite_target",
    ),
    [
        (0.95, 3.0, 0.99, -30.0, -1, 1, 1.0),
        (-0.95, -3.0, -0.99, 30.0, 1, -1, -1.0),
    ],
)
def test_max_inward_containment_relatches_on_opposite_ballistic_risk(
    first_position,
    first_velocity,
    opposite_position,
    opposite_velocity,
    first_direction,
    opposite_direction,
    opposite_target,
):
    action, _, asset = _action_and_env(
        guard=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    asset.data.joint_pos[:, 0] = first_position
    asset.data.joint_vel[:, 0] = first_velocity
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    assert action.max_inward_direction_latch[:, 0].tolist() == [
        first_direction,
        first_direction,
    ]

    # Inertia has now made the full-horizon prediction cross the opposite side.  Retaining the
    # old endpoint would accelerate into that edge, so the physical-risk side is relatched.
    asset.data.joint_pos[:, 0] = opposite_position
    asset.data.joint_vel[:, 0] = opposite_velocity
    action.process_actions(torch.zeros(2, 2))
    assert action.max_inward_direction_latch[:, 0].tolist() == [
        opposite_direction,
        opposite_direction,
    ]
    assert action.processed_actions[:, 0].tolist() == pytest.approx(
        [opposite_target, opposite_target]
    )


@pytest.mark.parametrize(
    (
        "first_position",
        "first_velocity",
        "dual_position",
        "dual_velocity",
        "latched_direction",
        "legacy_brake_target",
    ),
    [
        (0.95, 3.0, 1.15, -30.0, -1, 1.0),
        (-0.95, -3.0, -1.15, 30.0, 1, -1.0),
    ],
)
def test_max_inward_dual_side_risk_uses_q_minus_vt_not_stale_endpoint(
    first_position,
    first_velocity,
    dual_position,
    dual_velocity,
    latched_direction,
    legacy_brake_target,
):
    action, _, asset = _action_and_env(
        guard=True,
        guard_margin_fraction=0.05,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    asset.data.joint_pos[:, 0] = first_position
    asset.data.joint_vel[:, 0] = first_velocity
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    # q occupies one hard inset while q+v*T crosses the opposite inset.  Both side masks are true,
    # so retaining either endpoint would be arbitrary; the bounded legacy q-vT target opposes the
    # measured velocity for this write while terminal/crossing evidence remains latched.
    asset.data.joint_pos[:, 0] = dual_position
    asset.data.joint_vel[:, 0] = dual_velocity
    action.process_actions(torch.zeros(2, 2))
    assert action.max_inward_direction_latch[:, 0].tolist() == [
        latched_direction,
        latched_direction,
    ]
    assert action.processed_actions[:, 0].tolist() == pytest.approx(
        [legacy_brake_target, legacy_brake_target]
    )


def test_max_inward_release_hold_is_cancelled_if_outward_risk_returns():
    action, _, asset = _action_and_env(
        guard=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    asset.data.joint_pos[:, 0] = 0.95
    asset.data.joint_vel[:, 0] = 3.0
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    asset.data.joint_pos[:, 0] = 0.99
    asset.data.joint_vel[:, 0] = -0.2
    action.process_actions(torch.zeros(2, 2))
    start = float(asset.data._sim_timestamp)
    _set_sim_timestamp(asset, start)
    action.apply_actions()
    assert action.max_inward_release_hold[:, 0].all().item()
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.99, 0.99])

    # A later fresh readback in the same policy step is outward/dangerous again.  The stale hold
    # must be revoked immediately and the originally latched inward endpoint restored.
    asset.data.joint_vel[:, 0] = 3.0
    _set_sim_timestamp(asset, start + 0.025)
    action.apply_actions()
    assert not action.max_inward_release_hold[:, 0].any().item()
    assert action.max_inward_direction_latch[:, 0].tolist() == [-1, -1]
    assert action.processed_actions[:, 0].tolist() == pytest.approx([-1.0, -1.0])


def test_max_inward_lag_two_and_exact_resume_preserve_queue_and_direction():
    kwargs = dict(
        num_envs=2,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
        runtime_physics_dt=0.005,
        guard_margin_fraction=0.05,
        project_finite_qdes=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
        control_step_action_delay_min=2,
        control_step_action_delay_max=2,
    )
    action, _, asset = _action_and_env(**kwargs)
    action.reset()
    asset.data.joint_pos[:, 0] = 0.95
    asset.data.joint_vel[:, 0] = 10.0
    actor_action = torch.zeros(2, 31)
    actor_action[:, 0] = 0.60
    action.process_actions(actor_action)

    history = action._policy_action_delay._history.clone()
    assert action.processed_actions[:, 0].tolist() == pytest.approx([-0.9, -0.9])
    assert action.nominal_projected_qdes[:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert action.max_inward_direction_latch[:, 0].tolist() == [-1, -1]
    assert torch.equal(action.raw_actions, actor_action)
    assert torch.equal(action._policy_action_delay._history, history)
    _finish_guarded_policy_step(action, asset)

    state = action.action_delay_exact_resume_state_dict()
    assert state["brake_mode"] == "max_inward_until_nonoutward_v1"
    resumed, _, resumed_asset = _action_and_env(**kwargs)
    resumed.reset()
    resumed.load_action_delay_exact_resume_state_dict(state, strict=True)
    assert torch.equal(resumed._policy_action_delay._history, history)
    assert resumed.max_inward_direction_latch[:, 0].tolist() == [-1, -1]

    resumed_asset.data.joint_pos[:, 0] = 0.50
    resumed_asset.data.joint_vel[:, 0] = 0.10
    next_actor = torch.zeros(2, 31)
    next_actor[:, 0] = -0.75
    resumed.process_actions(next_actor)
    assert resumed.processed_actions[:, 0].tolist() == pytest.approx([-0.9, -0.9])
    assert torch.equal(resumed.raw_actions, next_actor)


def test_max_inward_containment_does_not_relax_raw_hard_terminal():
    action, env, asset = _action_and_env(
        guard=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
        project_finite_qdes=True,
    )
    asset.data.joint_pos[:, 0] = torch.tensor([1.21, 0.0])
    asset.data.joint_vel.zero_()
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]
    assert torch.all(torch.isfinite(action.processed_actions))


def test_max_inward_partial_reset_clears_only_selected_episode_latch():
    action, _, asset = _action_and_env(
        guard=True,
        guard_brake_mode="max_inward_until_nonoutward_v1",
    )
    asset.data.joint_pos[:, 0] = torch.tensor([-0.95, 0.95])
    asset.data.joint_vel[:, 0] = torch.tensor([-3.0, 3.0])
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    action._max_inward_release_hold[:, 0] = True
    action._max_inward_release_qdes[:, 0] = torch.tensor([-0.5, 0.5])
    action.reset(env_ids=torch.tensor([0]))
    assert action.max_inward_direction_latch[:, 0].tolist() == [0, -1]
    assert not action.max_inward_release_hold[0].any().item()
    assert action.max_inward_release_hold[1, 0].item()
    assert action._max_inward_release_qdes[:, 0].tolist() == pytest.approx([0.0, 0.5])


def test_max_inward_mode_is_exact_and_legacy_resume_schema_is_unchanged():
    with pytest.raises(ValueError, match="pre_apply_guard_brake_mode"):
        _action_and_env(guard=True, guard_brake_mode="max_inward")
    with pytest.raises(ValueError, match="requires pre_apply_limit_guard"):
        _action_and_env(
            guard=False,
            guard_brake_mode="max_inward_until_nonoutward_v1",
        )

    legacy, _, _ = _action_and_env(
        joint_count=31,
        guard=True,
        control_step_action_delay_min=2,
        control_step_action_delay_max=2,
    )
    legacy.reset()
    state = legacy.action_delay_exact_resume_state_dict()
    assert state["kind"] == "whole_body_tracking.policy_control_step_action_delay"
    assert "brake_mode" not in state


@pytest.mark.parametrize("project_finite_qdes", [False, True])
def test_physics_substep_ledger_catches_crossing_and_bounced_actual_hard_edge(
    project_finite_qdes,
):
    action, env, asset = _action_and_env(
        guard=True, project_finite_qdes=project_finite_qdes
    )
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))

    # apply0: q is inside hard +1.2 but q+qdot*0.1 crosses it.  Every fresh
    # substep readback retains the validated policy/reaction horizon.
    _set_sim_timestamp(asset, 0.0)
    asset.data.joint_pos[0, 0] = 1.15
    asset.data.joint_vel[0, 0] = 3.0
    action.apply_actions()
    assert action.physics_substep_hard_crossing_latch.tolist() == [True, False]
    assert action.processed_actions[0, 0].item() == pytest.approx(0.85)

    # apply1 observes an actual hard-edge breach.  Later readbacks bounce safely inside.
    _set_sim_timestamp(asset, 0.025)
    asset.data.joint_pos[0, 0] = 1.21
    asset.data.joint_vel[0, 0] = 0.0
    action.apply_actions()
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [True, False]
    for index in (2, 3):
        _set_sim_timestamp(asset, index * 0.025)
        asset.data.joint_pos.zero_()
        asset.data.joint_vel.zero_()
        action.apply_actions()
    _set_sim_timestamp(asset, 0.1)
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()

    # Either DoneTerm may run first; both use the same idempotent final post-step readback.
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    actual = terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "joint_pos_limits",
        0.0,
        0.0,
    )
    qdes = terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.0,
    )
    assert actual.tolist() == [True, False]  # sticky substep edge survives the safe bounce
    assert qdes.tolist() == (
        [False, False] if project_finite_qdes else [True, False]
    )
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["complete"] is True
    assert snapshot["hard_crossing"][0, 0, 0].item()
    assert snapshot["actual_hard_edge"][1, 0, 0].item()
    assert snapshot["substep_crossing_joint_count"][0, 0].item() >= 1
    assert snapshot["substep_actual_joint_count"][0, 0].item() == 1


def test_physics_substep_ledger_rejects_stale_or_missing_apply_sequence():
    action, env, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    _set_sim_timestamp(asset, 0.0)
    action.apply_actions()
    # The second apply must observe timestamp + physics_dt, never the previous stale state.
    with pytest.raises(RuntimeError, match="stale or skipped"):
        action.apply_actions()
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["apply_call_count"] == 1
    assert snapshot["record_count"] == 1
    assert action.physics_substep_hard_crossing_joint_count.sum().item() == 0

    action2, env2, asset2 = _action_and_env(guard=True)
    action2.process_actions(torch.zeros(2, 2))
    _set_sim_timestamp(asset2, 0.0)
    with pytest.raises(RuntimeError, match="exactly 4 prior apply_actions"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env2,
            "joint_pos",
            "joint_pos_limits",
            0.0,
            0.0,
        )


def test_physics_substep_lazy_buffer_timestamp_failure_is_transactional():
    action, _, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    _set_sim_timestamp(asset, 0.0)
    asset.data._joint_vel.timestamp = -0.025
    with pytest.raises(RuntimeError, match="lazy-buffer timestamps"):
        action.apply_actions()
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["apply_call_count"] == 0
    assert snapshot["record_count"] == 0
    assert action.physics_substep_hard_crossing_joint_count.sum().item() == 0
    assert action.physics_substep_actual_hard_edge_joint_count.sum().item() == 0


def test_update_accumulator_allows_multiple_nonterminal_steps_before_consume():
    action, _, asset = _action_and_env(guard=True)
    for _ in range(2):
        action.process_actions(torch.zeros(2, 2))
        _finish_guarded_policy_step(action, asset)

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["policy_step_sequence"] == 1
    assert snapshot["complete"] is True
    assert snapshot["terminal_archive_used"] == 0
    assert snapshot["policy_step_summary_used"] == 2
    assert [
        item["policy_step_sequence"]
        for item in snapshot["identity_bound_policy_steps"]
    ] == [0, 1]
    assert snapshot["identity_bound_policy_steps"][0]["complete"].all().item()
    assert snapshot["current_action_episode_identity"][
        "action_episode_sequence"
    ].tolist() == [0, 0]
    accumulated = snapshot["since_last_consume"]
    assert accumulated["policy_step_count"].tolist() == [2, 2]
    assert accumulated["complete_policy_step_count"].tolist() == [2, 2]
    assert accumulated["incomplete_policy_step_count"].tolist() == [0, 0]
    assert accumulated["apply_readback_count"].tolist() == [8, 8]
    assert accumulated["post_readback_count"].tolist() == [2, 2]
    assert accumulated["timestamp_invariant_pass_count"].tolist() == [2, 2]

    token, first = action.prepare_joint_safety_ledger_consume()
    assert first["since_last_consume"]["has_data"] is True
    action.acknowledge_joint_safety_ledger(token)
    second_token, second = action.prepare_joint_safety_ledger_consume()
    assert second["since_last_consume"]["has_data"] is False
    assert second["since_last_consume"]["consume_sequence"] == 1
    assert second["since_last_consume"]["policy_step_count"].sum().item() == 0
    assert second["terminal_archive_used"] == 0
    assert second["policy_step_summary_used"] == 0
    action.acknowledge_joint_safety_ledger(second_token)


def test_terminal_reset_before_consume_preserves_cpu_archive_and_birth_identity():
    action, env, asset = _action_and_env(guard=True)
    receipts = ("a" * 64, "b" * 64)
    action_uids = torch.tensor([101, 202], dtype=torch.long)
    command = types.SimpleNamespace(
        action_ball_enabled=True,
        action_ball_episode_generation=torch.tensor([7, 9], dtype=torch.long),
        action_ball_swing_generation=torch.tensor([3, 4], dtype=torch.long),
        action_ball_action_uid_for_envs=lambda ids: action_uids[ids],
        action_ball_birth_receipt_sha256=lambda env_id: receipts[env_id],
    )
    env.command_manager = types.SimpleNamespace(
        get_term=lambda name: command if name == "racket_target" else None
    )
    env.episode_length_buf[:] = torch.tensor([41, 17])
    action.process_actions(torch.zeros(2, 2))
    asset.data.joint_pos[0, 0] = 1.21
    _finish_guarded_policy_step(action, asset)

    env.termination_manager = types.SimpleNamespace(
        terminated=torch.tensor([False, False]),
        time_outs=torch.tensor([False, False]),
    )
    env.reset_terminated = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([False, False])
    env.episode_length_buf[0] = 42
    action.reset(env_ids=torch.tensor([0]))
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [False, False]

    snapshot = action.joint_safety_ledger_snapshot()
    assert not snapshot["env_valid"][:, 0].any().item()
    assert snapshot["terminal_archive_used"] == 1
    assert snapshot["policy_step_summary_used"] == 1
    step_identity = snapshot["identity_bound_policy_steps"][0][
        "action_identity"
    ]
    assert step_identity["action_uid"].tolist() == [101, 202]
    assert step_identity["birth_generation"].tolist() == [7, 9]
    assert snapshot["terminal_archive_payload_bytes"] > 0
    archived = snapshot["terminal_archives"][0]
    assert archived["env_id"] == 0
    assert archived["reasons"] == ("unsafe", "reset")
    assert archived["reset_hook_observed"] is True
    assert archived["terminated"] is True
    assert archived["timed_out"] is False
    assert archived["action_episode_sequence"] == 0
    assert archived["episode_length"] == 42
    assert archived["episode_length_at_policy_start"] == 41
    assert archived["episode_length_at_reset_hook"] == 42
    assert archived["action_uid"] == 101
    assert archived["birth_generation"] == 7
    assert archived["swing_generation"] == 3
    assert archived["birth_receipt_sha256"] == receipts[0]
    assert archived["included_in_accumulator"] is True
    transcript = archived["transcript"]
    assert transcript["complete"] is True
    assert transcript["q"].device.type == "cpu"
    assert transcript["actual_hard_edge"][:, 0].any().item()
    assert transcript["joint_pos_timestamp_s"] == pytest.approx(
        (0.0, 0.025, 0.05, 0.075, 0.1)
    )

    # Export mutation cannot corrupt the retained single-consumer evidence.
    transcript["q"].fill_(99.0)
    assert (
        action.joint_safety_ledger_snapshot()["terminal_archives"][0][
            "transcript"
        ]["q"][0, 0].item()
        != 99.0
    )
    token, consumed = action.prepare_joint_safety_ledger_consume()
    assert consumed["terminal_archive_used"] == 1
    action.acknowledge_joint_safety_ledger(token)
    assert action.joint_safety_ledger_snapshot()["terminal_archive_used"] == 0


def test_partial_reset_archives_incomplete_row_and_other_row_can_complete():
    action, _, asset = _action_and_env(guard=True)
    action.process_actions(torch.zeros(2, 2))
    for index in range(2):
        _set_sim_timestamp(asset, index * 0.025)
        action.apply_actions()
    action.reset(env_ids=torch.tensor([0]))
    for index in range(2, 4):
        _set_sim_timestamp(asset, index * 0.025)
        action.apply_actions()
    _set_sim_timestamp(asset, 0.1)
    action.finalize_joint_safety_post_step_readback()

    snapshot = action.joint_safety_ledger_snapshot()
    accumulated = snapshot["since_last_consume"]
    assert accumulated["policy_step_count"].tolist() == [1, 1]
    assert accumulated["complete_policy_step_count"].tolist() == [0, 1]
    assert accumulated["incomplete_policy_step_count"].tolist() == [1, 0]
    assert snapshot["terminal_archive_used"] == 1
    assert snapshot["policy_step_summary_used"] == 1
    assert snapshot["identity_bound_policy_steps"][0]["complete"].tolist() == [
        False,
        True,
    ]
    archived = snapshot["terminal_archives"][0]
    assert archived["env_id"] == 0
    assert archived["reasons"] == ("reset",)
    assert archived["transcript"]["complete"] is False
    assert archived["transcript"]["apply_call_count"] == 2
    assert archived["transcript"]["post_readback_count"] == 0
    assert not snapshot["env_valid"][:, 0].any().item()
    assert snapshot["env_valid"][:, 1].all().item()


def test_terminal_archive_overflow_is_sticky_and_never_overwrites():
    action, _, asset = _action_and_env(
        guard=True, terminal_archive_capacity=1
    )
    action.process_actions(torch.zeros(2, 2))
    asset.data.joint_pos[:, 0] = torch.tensor([1.21, -1.21])
    _finish_guarded_policy_step(action, asset)
    with pytest.raises(RuntimeError, match="archive overflow"):
        action.reset(env_ids=torch.tensor([0, 1]))

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["terminal_archive_capacity"] == 1
    assert snapshot["terminal_archive_used"] == 0
    assert snapshot["terminal_archive_overflow_latch"] is True
    assert snapshot["terminal_archive_overflow_count"] == 1
    with pytest.raises(RuntimeError, match="one-shot consume is disabled"):
        action.consume_joint_safety_ledger()
    after = action.joint_safety_ledger_snapshot()
    assert after["terminal_archive_used"] == 0
    assert after["terminal_archive_overflow_latch"] is True
    assert after["terminal_archive_overflow_count"] == 1
    with pytest.raises(RuntimeError, match="overflow is sticky"):
        action.process_actions(torch.zeros(2, 2))


def test_diagnostic_reset_batches_keep_compact_device_evidence_at_4096():
    num_envs = 4096
    action, _, asset = _action_and_env(
        num_envs=num_envs,
        joint_count=1,
        guard=True,
        terminal_archive_capacity=num_envs,
        action_ball_diagnostic_unauthorized=True,
    )
    all_envs = torch.arange(num_envs)

    action.process_actions(torch.zeros(num_envs, 1))
    _finish_guarded_policy_step(action, asset)
    action.reset(env_ids=all_envs)
    first = action.joint_safety_ledger_snapshot()
    assert first["terminal_archive_used"] == 0
    assert first["terminal_archive_overflow_latch"] is False
    assert first["terminal_archives"] == ()

    for env_ids in (
        torch.arange(0, num_envs, 2),
        torch.arange(1, num_envs, 2),
        torch.arange(0, num_envs, 3),
    ):
        action.process_actions(torch.zeros(num_envs, 1))
        _finish_guarded_policy_step(action, asset)
        action.reset(env_ids=env_ids)

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["terminal_archive_used"] == 0
    assert snapshot["terminal_archive_overflow_latch"] is False
    assert snapshot["terminal_archive_overflow_count"] == 0
    assert snapshot["terminal_archives"] == ()


def test_diagnostic_full_batch_skips_dense_summary_and_indexed_reduction(
    monkeypatch,
):
    action, _, asset = _action_and_env(
        num_envs=8,
        joint_count=3,
        guard=True,
        action_ball_diagnostic_unauthorized=True,
    )
    ledger = action._joint_safety_ledger
    assert ledger is not None
    monkeypatch.setattr(
        ledger,
        "aggregate_rows",
        lambda _ids: (_ for _ in ()).throw(
            AssertionError(
                "diagnostic full-batch accumulation used indexed gathers"
            )
        ),
    )

    action.process_actions(torch.zeros(8, 3))
    # These buffers belong only to formal per-policy-step summaries.  A compact
    # diagnostic must not spend rollout bandwidth clearing or refilling them.
    action._joint_safety_current_step_qdes_joint_count.fill_(251)
    action._joint_safety_current_step_policy_crossing_joint_count.fill_(252)
    action._joint_safety_current_step_substep_crossing_joint_count.fill_(253)
    action._joint_safety_current_step_actual_hard_edge_joint_count.fill_(254)
    action._joint_safety_current_step_minimum_hard_gap.fill_(-123.0)
    _finish_guarded_policy_step(action, asset)
    action.reset(env_ids=torch.tensor([0, 3, 7]))

    assert torch.all(
        action._joint_safety_current_step_qdes_joint_count.eq(251)
    )
    assert torch.all(
        action._joint_safety_current_step_policy_crossing_joint_count.eq(252)
    )
    assert torch.all(
        action._joint_safety_current_step_substep_crossing_joint_count.eq(253)
    )
    assert torch.all(
        action._joint_safety_current_step_actual_hard_edge_joint_count.eq(254)
    )
    assert torch.all(
        action._joint_safety_current_step_minimum_hard_gap.eq(-123.0)
    )
    token, snapshot = action.prepare_joint_safety_ledger_consume()
    since = snapshot["since_last_consume"]
    assert since["policy_step_count"].tolist() == [1] * 8
    assert since["complete_policy_step_count"].tolist() == [1] * 8
    assert since["incomplete_policy_step_count"].tolist() == [0] * 8
    assert since["apply_readback_count"].tolist() == [4] * 8
    assert since["post_readback_count"].tolist() == [1] * 8
    assert snapshot["identity_bound_policy_steps"] == ()
    assert snapshot["terminal_archives"] == ()
    action.acknowledge_joint_safety_ledger(token)


def test_diagnostic_fixed_tape_matches_formal_counters_without_receipt_reads():
    def make_action(*, diagnostic: bool):
        action, env, asset = _action_and_env(
            num_envs=2,
            joint_count=2,
            guard=True,
            action_ball_diagnostic_unauthorized=diagnostic,
        )
        receipts = ("a" * 64, "b" * 64)
        receipt_reads = []
        command = types.SimpleNamespace(
            action_ball_enabled=True,
            action_ball_episode_generation=torch.tensor(
                [7, 9], dtype=torch.long
            ),
            action_ball_swing_generation=torch.tensor(
                [3, 4], dtype=torch.long
            ),
            action_ball_action_uid_for_envs=lambda ids: torch.tensor(
                [101, 202], dtype=torch.long
            )[ids],
            action_ball_birth_receipt_sha256=lambda env_id: (
                receipt_reads.append(env_id) or receipts[env_id]
            ),
        )
        env.command_manager = types.SimpleNamespace(
            get_term=lambda name: (
                command if name == "racket_target" else None
            )
        )
        return action, asset, receipts, receipt_reads

    def play_fixed_tape(action, asset):
        steps = (
            (
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                torch.zeros(2, 2),
            ),
            (
                torch.tensor([[2.0, 0.0], [0.0, -2.0]]),
                torch.tensor([[1.21, 0.0], [0.0, -1.21]]),
                torch.zeros(2, 2),
            ),
            (
                torch.zeros(2, 2),
                torch.tensor([[0.2, 0.0], [0.0, -0.2]]),
                torch.tensor([[12.0, 0.0], [0.0, -12.0]]),
            ),
        )
        for actions, joint_pos, joint_vel in steps:
            asset.data.joint_pos.zero_()
            asset.data.joint_vel.zero_()
            action.process_actions(actions)
            asset.data.joint_pos.copy_(joint_pos)
            asset.data.joint_vel.copy_(joint_vel)
            _finish_guarded_policy_step(action, asset)
        return action.prepare_joint_safety_ledger_consume()

    formal, formal_asset, receipts, formal_receipt_reads = make_action(
        diagnostic=False
    )
    compact, compact_asset, _, compact_receipt_reads = make_action(
        diagnostic=True
    )
    formal_token, formal_snapshot = play_fixed_tape(formal, formal_asset)
    compact_token, compact_snapshot = play_fixed_tape(
        compact, compact_asset
    )

    formal_since = formal_snapshot["since_last_consume"]
    compact_since = compact_snapshot["since_last_consume"]
    for name in (
        "policy_step_count",
        "complete_policy_step_count",
        "incomplete_policy_step_count",
        "apply_readback_count",
        "post_readback_count",
        "timestamp_invariant_pass_count",
        "hard_crossing_latch",
        "actual_hard_edge_latch",
        "qdes_joint_count",
        "policy_crossing_joint_count",
        "substep_hard_crossing_joint_count",
        "actual_hard_edge_joint_count",
        "minimum_hard_lower_gap",
        "minimum_hard_upper_gap",
    ):
        assert torch.equal(compact_since[name], formal_since[name]), name

    # The old formal schema remains byte-identifiable: receipts are resolved
    # once per birth and retained on every immutable policy-step identity.
    assert formal_receipt_reads == [0, 1]
    assert all(
        summary["action_identity"]["birth_receipt_sha256"] == receipts
        for summary in formal_snapshot["identity_bound_policy_steps"]
    )
    # The non-promotable compact path must never pull Python SHA strings into
    # the rollout hot loop; its device counters are the only training evidence.
    assert compact_receipt_reads == []
    assert compact_snapshot["identity_bound_policy_steps"] == ()
    assert compact_snapshot["terminal_archives"] == ()

    formal.acknowledge_joint_safety_ledger(formal_token)
    compact.acknowledge_joint_safety_ledger(compact_token)


def test_one_shot_consume_fails_closed_even_when_guard_is_disabled():
    action, _, _ = _action_and_env(guard=False)
    assert action.joint_safety_ledger_snapshot()["enabled"] is False

    with pytest.raises(RuntimeError, match="one-shot consume is disabled"):
        action.consume_joint_safety_ledger()


@pytest.mark.parametrize(
    "bad_ids",
    [
        torch.tensor([0.9]),
        torch.tensor([True]),
        [0.9],
        [True],
    ],
)
def test_joint_safety_reset_rejects_noninteger_environment_ids(bad_ids):
    action, _, _ = _action_and_env(guard=True)
    with pytest.raises(TypeError, match="integer"):
        action.reset(env_ids=bad_ids)


def test_runner_consumes_once_persists_every_step_and_keeps_cross_reset_identity(
    monkeypatch, tmp_path, capsys
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    prepare_calls = []
    acknowledge_calls = []
    original_prepare = action.prepare_joint_safety_ledger_consume
    original_acknowledge = action.acknowledge_joint_safety_ledger

    def counted_prepare():
        prepare_calls.append(1)
        return original_prepare()

    def counted_acknowledge(token):
        acknowledge_calls.append(token)
        return original_acknowledge(token)

    action.prepare_joint_safety_ledger_consume = counted_prepare
    action.acknowledge_joint_safety_ledger = counted_acknowledge
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0

    first = runner._consume_joint_safety_update(
        12, expected_action_term=action
    )
    second = runner._consume_joint_safety_update(
        12, expected_action_term=action
    )

    assert first is second
    assert prepare_calls == [1]
    assert len(acknowledge_calls) == 1
    assert first["consume_sequence"] == 0
    assert first["policy_step_count"] == 2
    assert len(first["per_policy_step_sparse_counters"]) == 2
    assert first["terminal_archive_count"] == 1
    assert first["terminal_reason_counts"] == {"reset": 1}
    artifact = tmp_path / first["artifact"]["path"]
    assert artifact.is_file()
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert payload["status"] == "prepared_before_optimizer"
    archives = payload["terminal"]["entries"]
    assert len(archives) == 1
    assert archives[0]["storage"] == "full_forensic"
    archived = archives[0]["archive"]
    assert archived["action_uid"] == 101
    assert archived["birth_generation"] == 7
    assert archived["birth_receipt_sha256"] == "a" * 64
    identity = payload["identity"]
    assert identity["initial"]["action_uid"].tolist() == [101, 202]
    assert identity["changes"]["action_uid"]["index"].tolist() == [[1, 0]]
    assert identity["changes"]["action_uid"]["value"].tolist() == [303]
    assert (tmp_path / first["optimizer_commit"]["path"]).is_file()
    # The exact prepared generation is acknowledged only after its optimizer marker is durable.
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is False
    )
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    ]
    assert len(lines) == 1
    assert json.loads(lines[0].split("=", 1)[1]) == first


def test_runner_validates_crossing_over_the_full_policy_guard_horizon(
    monkeypatch,
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger(
        policy_horizon_only_crossing=True
    )
    _, snapshot = action.prepare_joint_safety_ledger_consume()
    transcript = snapshot["terminal_archives"][0]["transcript"]
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.num_steps_per_env = 2
    contract = runner._joint_safety_runtime_contract(action)

    q = transcript["q"][0, 0]
    qdot = transcript["qdot"][0, 0]
    travel = contract["hard_upper"][0] - contract["hard_lower"][0]
    inner_upper = contract["hard_upper"][0] - (
        contract["margin_rad"] + contract["margin_fraction"] * travel
    )
    assert q + qdot * contract["physics_dt_s"] < inner_upper
    assert (
        q
        + qdot
        * contract["physics_dt_s"]
        * contract["expected_apply_calls"]
        >= inner_upper
    )
    assert transcript["hard_crossing"][0, 0].item()

    validated = runner._validate_joint_safety_update_snapshot(
        snapshot, step=0, contract=contract
    )
    assert validated["archive_count"] == 1


def test_runner_distinguishes_episode_sticky_counts_from_current_transcript(
    monkeypatch,
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger(
        policy_horizon_only_crossing=True,
        reset_after_safe_step=True,
    )
    _, snapshot = action.prepare_joint_safety_ledger_consume()
    transcript = snapshot["terminal_archives"][0]["transcript"]
    assert not transcript["hard_crossing"].any().item()
    assert transcript["substep_crossing_joint_count"][0].item() > 0

    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.num_steps_per_env = 3
    contract = runner._joint_safety_runtime_contract(action)
    validated = runner._validate_joint_safety_update_snapshot(
        snapshot, step=0, contract=contract
    )
    assert validated["archive_count"] == 1


def test_protected_task_detection_covers_action_ball_and_upper_safe(monkeypatch):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    action_ball_cfg = types.SimpleNamespace(
        commands=types.SimpleNamespace(
            racket_target=types.SimpleNamespace(
                target_mode="action_ball",
                action_ball_diagnostic_unauthorized=False,
            )
        )
    )
    runner.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(cfg=action_ball_cfg)
    )
    assert runner._effective_reward_activation_task_kind() == "action_ball"
    action_ball_cfg.commands.racket_target.action_ball_diagnostic_unauthorized = (
        True
    )
    assert runner._effective_reward_activation_task_kind() is None
    callback_calls = []
    runner.env.unwrapped.command_manager = types.SimpleNamespace(
        active_terms=("racket_target",),
        get_term=lambda _name: types.SimpleNamespace(
            on_rollout_end=lambda step: callback_calls.append(step)
        ),
    )
    runner._notify_command_terms_rollout_end(7)
    assert callback_calls == []

    upper_cfg_type = type("HOPEPingPongUpperSafeAgibotA3EnvCfg", (), {})
    upper_cfg = upper_cfg_type()
    upper_cfg.commands = types.SimpleNamespace(
        racket_target=types.SimpleNamespace(target_mode="virtual_ball")
    )
    runner.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(cfg=upper_cfg)
    )
    assert runner._effective_reward_activation_task_kind() == "upper_safe"


def test_learn_consumes_joint_safety_before_command_rollout_callback(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    callback_observations = []
    call_order = []
    command = env.command_manager.get_term("racket_target")

    def on_rollout_end(step):
        call_order.append("curriculum_callback")
        callback_observations.append(
            (
                step,
                action.joint_safety_ledger_snapshot()["since_last_consume"][
                    "has_data"
                ],
                list((tmp_path / "joint_safety_ledgers").glob("*.pt")),
            )
        )

    command.on_rollout_end = on_rollout_end

    effective_reward_module = types.ModuleType(
        "whole_body_tracking.utils.effective_reward_recipe"
    )

    class FakeRewardLedger:
        def __init__(self, *_args, **_kwargs):
            pass

        def begin_environment_step(self):
            return "fake-step"

        def observe_after_environment_step(self, *_args):
            pass

        def prepare_update(self, step, **_kwargs):
            activation = {
                "recipe_sha256": "a" * 64,
                "step_dt_s": 0.02,
                "num_envs": 2,
                "environment_step_count": 2,
                "ppo_update": step,
            }
            return {
                "ppo_update": step,
                "activation": activation,
                "per_action": {},
                "safety": {},
                "action_ball_conservation": {
                    "event": "hope_reward_episode_segmented_closure_update",
                    "schema_version": 1,
                    "status": "PASS",
                    "evidence_source": "live_isaac_reward_manager",
                    "capture_mode": "reward_manager_reset_pre_clear_hook",
                    "task_kind": "action_ball",
                    "ppo_update": step,
                    "recipe_sha256": activation["recipe_sha256"],
                    "step_dt_s": activation["step_dt_s"],
                    "num_envs": activation["num_envs"],
                    "segment_key_fields": ["env_id", "reset_generation"],
                    "all_reward_manager_term_names": ["death_penalty"],
                    "completed_episode_count": 0,
                    "completed_episode_segments": [],
                    "reset_batches": [],
                    "open_episode_count": 2,
                    "open_episode_segments": [
                        {"env_id": 0},
                        {"env_id": 1},
                    ],
                    "dashboard_normalization": {
                        "status": "NOT_OBSERVED_NO_RESET",
                        "reset_batch_count": 0,
                    },
                    "e2_eligible": False,
                    "checks": {
                        "status": "PASS",
                        "environment_step_count": activation[
                            "environment_step_count"
                        ],
                        "all_step_reward_buf_equals_all_term_sums": "PASS",
                        "all_episode_sums_equal_captured_term_sums": "PASS",
                        "all_reset_episode_sums_cleared": "PASS",
                        "exact_environment_step_coverage": "PASS",
                    },
                },
                "status": "frozen_validated_before_optimizer",
            }

        def acknowledge_update(self, _prepared):
            pass

        def close(self):
            pass

    effective_reward_module.EffectiveRewardActivationLedger = FakeRewardLedger
    effective_reward_module.ActionBoundRewardEvidenceLedger = FakeRewardLedger
    effective_reward_module.canonical_effective_reward_activation_json = (
        lambda record: json.dumps(record, sort_keys=True)
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        effective_reward_module,
    )

    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def one_update(_self, **_kwargs):
        _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", one_update, raising=False)
    wrapper = types.SimpleNamespace(unwrapped=env, step=lambda *_a, **_k: None)
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = wrapper
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0
    runner.current_learning_iteration = 12
    runner.alg = types.SimpleNamespace(
        update=lambda: call_order.append("optimizer")
    )
    runner._effective_reward_activation_task_kind = lambda: "action_ball"
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False
    # This focused test owns only the joint-safety/Reward/curriculum ordering.
    # Frozen evaluation has an independent owner-contract suite and is disabled
    # here so the deliberately tiny fake CommandManager need not impersonate it.
    runner._service_action_ball_frozen_evaluation = lambda _step: False
    original_prepare = runner._prepare_joint_safety_update
    original_commit_marker = runner._persist_joint_safety_optimizer_commit
    original_directory_fsync = runner._joint_safety_fsync_directory
    original_acknowledge = action.acknowledge_joint_safety_ledger

    def ordered_prepare(*args, **kwargs):
        call_order.append("prepare_validate_persist")
        return original_prepare(*args, **kwargs)

    def ordered_commit_marker(prepared):
        call_order.append("durable_optimizer_marker")
        return original_commit_marker(prepared)

    def ordered_directory_fsync(directory):
        call_order.append("directory_fsync")
        return original_directory_fsync(directory)

    def ordered_acknowledge(token):
        call_order.append("ledger_ack")
        return original_acknowledge(token)

    runner._prepare_joint_safety_update = ordered_prepare
    runner._persist_joint_safety_optimizer_commit = ordered_commit_marker
    runner._joint_safety_fsync_directory = ordered_directory_fsync
    action.acknowledge_joint_safety_ledger = ordered_acknowledge

    runner.learn(num_learning_iterations=1)

    assert len(callback_observations) == 1
    step, has_data_at_callback, artifacts_at_callback = callback_observations[0]
    assert step == 12
    assert has_data_at_callback is False
    assert len(artifacts_at_callback) == 1
    assert call_order == [
        "prepare_validate_persist",
        "directory_fsync",
        # Action-bound Reward evidence has its own pre-optimizer durable
        # receipt and post-optimizer commit marker.
        "directory_fsync",
        "optimizer",
        "directory_fsync",
        "durable_optimizer_marker",
        "directory_fsync",
        "ledger_ack",
        "curriculum_callback",
    ]


def test_inflight_frozen_eval_fences_next_optimizer_and_sidecar_failure(
    monkeypatch, tmp_path
):
    """A published request cannot be judged after another PPO update."""

    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    calls = []

    class FailingSidecarTerm:
        def __init__(self):
            self.stage = None
            self.inflight_polls = 0

        def action_ball_frozen_evaluation_boundary(
            self, *, phase, runner_bindings=None, **_kwargs
        ):
            if phase == "poll" and self.stage is None:
                return {
                    "request_seq": 0,
                    "request_due": True,
                    "needs_global_reset": False,
                    "needs_ack_checkpoint": False,
                    "stage": None,
                    "requires_runner_binding": False,
                }
            if phase == "publish_request":
                assert runner_bindings == {
                    "policy_generation": 5,
                    "policy_state": "frozen",
                }
                self.stage = "published"
                calls.append("request_published")
                return {
                    "request_seq": 0,
                    "published": True,
                }
            if phase == "poll" and self.stage == "published":
                self.inflight_polls += 1
                if self.inflight_polls == 1:
                    return {
                        "request_seq": 0,
                        "request_due": False,
                        "needs_global_reset": False,
                        "needs_ack_checkpoint": False,
                        "stage": "published",
                        "requires_runner_binding": False,
                    }
                raise RuntimeError("sidecar liveness failed")
            raise AssertionError(
                f"unexpected frozen-eval phase {phase!r}"
            )

    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def two_updates(_self, *, num_learning_iterations, **_kwargs):
        for _ in range(num_learning_iterations):
            _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", two_updates, raising=False)
    monkeypatch.setattr(
        runner_mod.time,
        "sleep",
        lambda _seconds: calls.append("fenced_poll_wait"),
    )

    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(),
        step=lambda *_args, **_kwargs: None,
        reset=lambda: calls.append("env_reset"),
    )
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 1
    runner.rank = 0
    runner.current_learning_iteration = 5
    runner.alg = types.SimpleNamespace(
        update=lambda: calls.append("optimizer")
    )
    runner._effective_reward_activation_task_kind = lambda: None
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False
    term = FailingSidecarTerm()
    runner._action_ball_frozen_eval_term = lambda: term
    runner._notify_command_terms_rollout_end = (
        lambda step: calls.append(f"rollout_end:{step}")
    )
    runner._action_ball_control_checkpoint = (
        lambda *, purpose, **_kwargs: tmp_path / f"{purpose}.pt"
    )
    runner._frozen_eval_runner_bindings = (
        lambda *, policy_generation: {
            "policy_generation": policy_generation,
            "policy_state": "frozen",
        }
    )
    runner._action_ball_frozen_eval_poll_interval_s = 0.001

    with pytest.raises(RuntimeError, match="sidecar liveness failed"):
        runner.learn(num_learning_iterations=2)

    assert calls.count("optimizer") == 1
    assert calls == [
        "optimizer",
        "rollout_end:5",
        "request_published",
        "fenced_poll_wait",
    ]


def test_runner_rejects_sticky_overflow_before_consuming(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env, _ = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
    )
    action._joint_safety_policy_step_summary_overflow_latch = True
    action._joint_safety_policy_step_summary_overflow_count = 1
    calls = []
    original_consume = action.consume_joint_safety_ledger

    def counted_consume():
        calls.append(1)
        return original_consume()

    action.consume_joint_safety_ledger = counted_consume
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 1
    runner.rank = 0

    with pytest.raises(RuntimeError, match="overflow is already latched"):
        runner._consume_joint_safety_update(
            0, expected_action_term=action
        )
    assert calls == []
    assert not (tmp_path / "joint_safety_ledgers").exists()


def test_runner_rejects_terminal_archive_identity_drift(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    token, snapshot = action.prepare_joint_safety_ledger_consume()
    snapshot["terminal_archives"][0]["action_uid"] = 999
    action.prepare_joint_safety_ledger_consume = lambda: (token, snapshot)
    acknowledge_calls = []
    action.acknowledge_joint_safety_ledger = lambda value: acknowledge_calls.append(
        value
    )
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0

    with pytest.raises(
        RuntimeError, match="action_uid does not match its policy-step identity"
    ):
        runner._consume_joint_safety_update(
            3, expected_action_term=action
        )
    assert acknowledge_calls == []
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is True
    )
    assert not list((tmp_path / "joint_safety_ledgers").glob("*.pt"))


def test_two_phase_consume_freezes_every_mutator_and_requires_exact_token():
    action, _, asset = _action_and_env(guard=True)
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    token, snapshot = action.prepare_joint_safety_ledger_consume()
    assert snapshot["since_last_consume"]["has_data"] is True
    with pytest.raises(RuntimeError, match="prepared but not acknowledged"):
        action.prepare_joint_safety_ledger_consume()
    with pytest.raises(RuntimeError, match="refusing process_actions"):
        action.process_actions(torch.zeros(2, 2))
    with pytest.raises(RuntimeError, match="refusing reset"):
        action.reset(env_ids=torch.tensor([0]))
    with pytest.raises(RuntimeError, match="refusing apply_actions"):
        action.apply_actions()
    with pytest.raises(RuntimeError, match="refusing finalize"):
        action.finalize_joint_safety_post_step_readback()
    with pytest.raises(RuntimeError, match="token does not match"):
        action.acknowledge_joint_safety_ledger((*token[:-1], "0" * 64))
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True

    action.acknowledge_joint_safety_ledger(token)
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is False
    with pytest.raises(RuntimeError, match="no prepared consume"):
        action.acknowledge_joint_safety_ledger(token)


def test_two_phase_ack_rejects_private_evidence_mutation_without_clearing():
    action, _, asset = _action_and_env(guard=True)
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    token, _ = action.prepare_joint_safety_ledger_consume()
    action._joint_safety_accumulator_policy_steps[0] += 1

    with pytest.raises(RuntimeError, match="evidence changed"):
        action.acknowledge_joint_safety_ledger(token)
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


def test_joint_safety_fingerprint_tracks_normal_and_inference_tensors():
    def fingerprint(value):
        hasher = hashlib.sha256()
        hope_actions_mod.ClampedJointPositionAction._joint_safety_fingerprint_value(
            hasher, value
        )
        return hasher.hexdigest()

    normal = torch.zeros(3)
    normal_before = fingerprint(normal)
    normal.add_(1.0)
    assert fingerprint(normal) != normal_before

    with torch.inference_mode():
        inference = torch.arange(3.0)
        replacement = inference.clone()
    inference_before = fingerprint(inference)
    assert fingerprint(inference) == inference_before
    assert fingerprint(replacement) != inference_before
    with torch.inference_mode():
        inference.resize_(4)
    assert fingerprint(inference) != inference_before


def test_prepare_returns_borrowed_view_without_recursive_export_clone(
    monkeypatch,
):
    action, _, asset = _action_and_env(
        num_envs=256,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
    )
    for _ in range(4):
        action.process_actions(torch.zeros(256, 31))
        _finish_guarded_policy_step(action, asset)

    fingerprint_calls = []
    original_fingerprint = action._joint_safety_evidence_fingerprint

    def counted_fingerprint():
        fingerprint_calls.append(1)
        return original_fingerprint()

    def forbidden_export_clone(_value):
        raise AssertionError(
            "prepare must not recursively clone the monitoring export"
        )

    monkeypatch.setattr(
        action, "_joint_safety_evidence_fingerprint", counted_fingerprint
    )
    monkeypatch.setattr(
        action, "_joint_safety_export_clone", forbidden_export_clone
    )
    tracemalloc.start()
    start = time.perf_counter()
    token, snapshot = action.prepare_joint_safety_ledger_consume()
    elapsed = time.perf_counter() - start
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert fingerprint_calls == [1]
    assert elapsed < 0.25
    assert peak_bytes < 512 * 1024
    assert "q" not in snapshot
    assert "qdot" not in snapshot
    assert snapshot["identity_bound_policy_steps"][0] is (
        action._joint_safety_policy_step_summaries[0]
    )
    assert (
        snapshot["since_last_consume"]["policy_step_count"].data_ptr()
        == action._joint_safety_accumulator_policy_steps.data_ptr()
    )
    assert (
        snapshot["since_last_consume"][
            "actual_hard_edge_joint_count"
        ].data_ptr()
        == action._joint_safety_accumulator_actual_hard_edge_joint_count.data_ptr()
    )

    action.acknowledge_joint_safety_ledger(token)
    assert fingerprint_calls == [1, 1]


def test_4096_by_24_prepare_has_bounded_python_peak_and_no_dense_clone(
    monkeypatch,
):
    action, _, asset = _action_and_env(
        num_envs=4096,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
        terminal_archive_capacity=64,
    )
    zero_action = torch.zeros(4096, 31)
    for _ in range(24):
        action.process_actions(zero_action)
        _finish_guarded_policy_step(action, asset)

    monkeypatch.setattr(
        action,
        "_joint_safety_export_clone",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("prepare attempted a recursive dense export clone")
        ),
    )
    tracemalloc.start()
    start = time.perf_counter()
    token, snapshot = action.prepare_joint_safety_ledger_consume()
    elapsed = time.perf_counter() - start
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert elapsed < 1.0
    assert peak_bytes < 1024 * 1024
    assert len(snapshot["identity_bound_policy_steps"]) == 24
    assert snapshot["terminal_archive_used"] == 0
    assert (
        snapshot["identity_bound_policy_steps"][-1][
            "minimum_hard_gap"
        ].data_ptr()
        == action._joint_safety_policy_step_summaries[-1][
            "minimum_hard_gap"
        ].data_ptr()
    )
    action.acknowledge_joint_safety_ledger(token)


def test_runner_rejects_self_consistent_incomplete_readback_window(
    monkeypatch
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    _, snapshot = action.prepare_joint_safety_ledger_consume()
    summary = snapshot["identity_bound_policy_steps"][0]
    summary["complete"][0] = False
    summary["apply_readback_count"][0] = 3
    summary["timestamp_invariant_pass"][0] = False
    since = snapshot["since_last_consume"]
    since["complete_policy_step_count"][0] -= 1
    since["incomplete_policy_step_count"][0] += 1
    since["apply_readback_count"][0] -= 1
    since["timestamp_invariant_pass_count"][0] -= 1

    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.num_steps_per_env = 2
    contract = runner._joint_safety_runtime_contract(action)
    with pytest.raises(RuntimeError, match="incomplete"):
        runner._validate_joint_safety_update_snapshot(
            snapshot, step=0, contract=contract
        )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


@pytest.mark.parametrize(
    "corruption,match",
    [
        (
            lambda transcript: transcript["q"].__setitem__(
                (0, 0), float("nan")
            ),
            "actual-hard-edge mask|hard gaps",
        ),
        (
            lambda transcript: transcript.__setitem__("record_count", 999),
            "complete bound",
        ),
        (
            lambda transcript: transcript.__setitem__(
                "record_kind", ("forged",) * 5
            ),
            "kind/index",
        ),
        (
            lambda transcript: transcript[
                "substep_actual_joint_count"
            ].__setitem__(0, 99),
            "substep masks/counters",
        ),
        (
            lambda transcript: transcript.__setitem__(
                "joint_vel_timestamp_s",
                (0.0, 0.005, 0.010, 0.015, 99.0),
            ),
            "lazy-buffer timestamps",
        ),
    ],
)
def test_runner_deeply_rejects_terminal_transcript_corruption(
    monkeypatch, corruption, match
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    _, snapshot = action.prepare_joint_safety_ledger_consume()
    corruption(snapshot["terminal_archives"][0]["transcript"])
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.num_steps_per_env = 2
    contract = runner._joint_safety_runtime_contract(action)

    with pytest.raises(RuntimeError, match=match):
        runner._validate_joint_safety_update_snapshot(
            snapshot, step=0, contract=contract
        )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


def test_runner_persistence_failure_keeps_prepared_action_evidence(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0
    runner._persist_joint_safety_update = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected disk full")
        )
    )

    with pytest.raises(OSError, match="disk full"):
        runner._prepare_joint_safety_update(
            0, expected_action_term=action
        )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True
    assert runner._joint_safety_pending_prepared["status"] == "snapshot_frozen"
    with pytest.raises(RuntimeError, match="prepared but not acknowledged"):
        action.process_actions(torch.zeros(2, 2))


def test_actual_hard_edge_is_durable_and_blocks_optimizer(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env, asset = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
    )
    action.process_actions(torch.zeros(2, 2))
    asset.data.joint_pos[0, 0] = 1.21
    _finish_guarded_policy_step(action, asset)
    env.reset_terminated = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([False, False])
    action.reset(env_ids=torch.tensor([0]))

    effective_reward_module = types.ModuleType(
        "whole_body_tracking.utils.effective_reward_recipe"
    )

    class FakeRewardLedger:
        def __init__(self, *_args, **_kwargs):
            pass

        def observe_after_environment_step(self):
            pass

        def finish_update(self, step):
            return {"ppo_update": step}

    effective_reward_module.EffectiveRewardActivationLedger = FakeRewardLedger
    effective_reward_module.ActionBoundRewardEvidenceLedger = FakeRewardLedger
    effective_reward_module.canonical_effective_reward_activation_json = (
        lambda record: json.dumps(record, sort_keys=True)
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        effective_reward_module,
    )
    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def one_update(_self, **_kwargs):
        _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", one_update, raising=False)
    optimizer_calls = []
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(
        unwrapped=env, step=lambda *_a, **_k: None
    )
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 1
    runner.rank = 0
    runner.current_learning_iteration = 0
    runner.alg = types.SimpleNamespace(
        update=lambda: optimizer_calls.append("optimizer")
    )
    runner._effective_reward_activation_task_kind = lambda: "upper_safe"
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False

    with pytest.raises(RuntimeError, match="refusing PPO update"):
        runner.learn(num_learning_iterations=1)
    assert optimizer_calls == []
    artifacts = list(
        (tmp_path / "joint_safety_ledgers").glob("*.prepared.pt")
    )
    assert len(artifacts) == 1
    payload = torch.load(
        artifacts[0], map_location="cpu", weights_only=False
    )
    assert payload["status"] == "fatal_actual_hard_edge"
    assert payload["fatal_flags"]["actual_hard_edge_event_count"] > 0
    assert not list(
        (tmp_path / "joint_safety_ledgers").glob(
            "*.optimizer_commit.json"
        )
    )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


def test_diagnostic_finite_hard_edge_is_terminal_training_sample(
    monkeypatch, tmp_path, capsys
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env, asset = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
        action_ball_diagnostic_unauthorized=True,
    )
    action.process_actions(torch.zeros(2, 2))
    asset.data.joint_pos[0, 0] = 1.21
    _finish_guarded_policy_step(action, asset)
    env.reset_terminated = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([False, False])
    action.reset(env_ids=torch.tensor([0]))

    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def one_update(_self, **_kwargs):
        _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", one_update, raising=False)
    optimizer_calls = []
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(
        unwrapped=env, step=lambda *_a, **_k: None
    )
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 1
    runner.rank = 0
    runner.current_learning_iteration = 0
    runner.alg = types.SimpleNamespace(
        update=lambda: optimizer_calls.append("optimizer")
    )
    runner._effective_reward_activation_task_kind = lambda: None
    runner._service_action_ball_frozen_evaluation = lambda *_args: None
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False

    runner.learn(num_learning_iterations=1)
    assert optimizer_calls == ["optimizer"]
    output_lines = capsys.readouterr().out.splitlines()
    joint_lines = [
        line
        for line in output_lines
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    ]
    assert len(joint_lines) == 1
    payload = json.loads(joint_lines[0].split("=", 1)[1])
    assert payload["status"] == (
        "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
    )
    assert payload["formal_authority"] is False
    assert payload["counter_totals"]["actual_hard_edge_events"] > 0
    assert payload["terminal_archive_count"] == 0
    assert not (tmp_path / "joint_safety_ledgers").exists()
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is False


def test_4096_by_24_safe_compact_artifact_stays_within_budget(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env, asset = _action_and_env(
        num_envs=4096,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
        terminal_archive_capacity=64,
    )
    for _ in range(24):
        action.process_actions(torch.zeros(4096, 31))
        _finish_guarded_policy_step(action, asset)
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 24
    runner.rank = 0
    original_prepare = action.prepare_joint_safety_ledger_consume
    prepare_metrics = {}

    def measured_prepare():
        tracemalloc.start()
        start = time.perf_counter()
        token, snapshot = original_prepare()
        elapsed = time.perf_counter() - start
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        prepare_metrics.update(
            elapsed_s=elapsed,
            python_peak_bytes=peak_bytes,
        )
        assert "q" not in snapshot
        assert "qdot" not in snapshot
        assert "hard_lower_gap" not in snapshot
        assert (
            snapshot["identity_bound_policy_steps"][0][
                "minimum_hard_gap"
            ].data_ptr()
            == action._joint_safety_policy_step_summaries[0][
                "minimum_hard_gap"
            ].data_ptr()
        )
        assert (
            snapshot["since_last_consume"][
                "actual_hard_edge_joint_count"
            ].data_ptr()
            == action._joint_safety_accumulator_actual_hard_edge_joint_count.data_ptr()
        )
        return token, snapshot

    action.prepare_joint_safety_ledger_consume = measured_prepare

    record = runner._consume_joint_safety_update(
        0, expected_action_term=action
    )
    assert prepare_metrics["elapsed_s"] < 1.0
    assert prepare_metrics["python_peak_bytes"] < 1024 * 1024
    assert (
        record["artifact"]["size_bytes"]
        <= runner_mod._JOINT_SAFETY_NORMAL_ARTIFACT_MAX_BYTES
    )
    payload = torch.load(
        tmp_path / record["artifact"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    assert (
        payload["budgets"]["core_payload_bytes"]
        <= runner_mod._JOINT_SAFETY_CORE_PAYLOAD_MAX_BYTES
    )
    assert payload["terminal"]["archive_count"] == 0
    assert all(
        step["sparse_counters"][field]["nonzero_cells"] == 0
        for step in payload["policy_steps"]
        for field in (
            "qdes_joint_count",
            "policy_crossing_joint_count",
            "substep_hard_crossing_joint_count",
            "actual_hard_edge_joint_count",
        )
    )


def test_physics_substep_ledger_partial_reset_atomically_clears_rows_and_counters():
    action, _, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    for index in range(4):
        _set_sim_timestamp(asset, index * 0.025)
        asset.data.joint_pos[:, 0] = torch.tensor([1.21, -1.21])
        asset.data.joint_vel.zero_()
        action.apply_actions()
    _set_sim_timestamp(asset, 0.1)
    action.finalize_joint_safety_post_step_readback()
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [True, True]

    action.reset(env_ids=torch.tensor([0]))
    snapshot = action.joint_safety_ledger_snapshot()
    assert not snapshot["env_valid"][:, 0].any().item()
    assert snapshot["env_valid"][:, 1].all().item()
    assert torch.isnan(snapshot["q"][:, 0]).all().item()
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [False, True]
    assert action.physics_substep_actual_hard_edge_joint_count[0].sum().item() == 0
    assert action.physics_substep_actual_hard_edge_joint_count[1, 0].item() == 5


def test_physics_substep_contract_pins_decimation_times_physics_dt():
    with pytest.raises(ValueError, match=r"physics_dt \* decimation"):
        _action_and_env(
            guard=True,
            decimation=4,
            runtime_physics_dt=0.01,  # 4 * .01 != policy/runtime step_dt .1
        )


def test_actual_joint_position_inner_margin_is_nonterminal_but_raw_edge_is_terminal():
    _, env, asset = _action_and_env()
    # On [-1, 1], the 10%-of-travel inner edge is +/-0.8.  Crossing that recoverable
    # diagnostic/Reward band is not a Done; reaching the raw mechanical edge is.
    asset.data.joint_pos[:] = torch.tensor([[0.7999, 0.0], [-0.8, 0.0]])
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    result = terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.1,
    )
    assert result.tolist() == [False, False]

    asset.data.joint_pos[1, 0] = -1.0
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.1,
    ).tolist() == [False, True]
    asset.data.joint_pos[0, 0] = float("nan")
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.1,
    ).tolist() == [True, True]


def test_actual_joint_position_limit_source_is_explicit_not_guessed():
    _, env, asset = _action_and_env()
    asset.data.joint_pos[:] = torch.tensor([[1.2, 0.0], [0.0, 0.0]])
    asset.data.joint_pos_limits = _limits(2, 2, lo=-2.0, hi=2.0)
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]


def test_done_terms_reject_reordered_or_partial_joint_contracts():
    action, env, asset = _action_and_env()
    action.process_actions(torch.zeros(2, 2))

    action._joint_names = list(reversed(action._joint_names))
    with pytest.raises(RuntimeError, match="joint-name order"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )
    action._joint_names = list(asset.data.joint_names)
    action._joint_ids = [1, 0]
    with pytest.raises(RuntimeError, match="identity order"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )

    bad_cfg = types.SimpleNamespace(name="robot", joint_ids=[1, 0])
    with pytest.raises(RuntimeError, match="identity order"):
        terminations_mod.actual_joint_position_forbidden_zone(
            env,
            bad_cfg,
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )
    non_integer_cfg = types.SimpleNamespace(name="robot", joint_ids=[0.0, 1.0])
    with pytest.raises(RuntimeError, match="integer joint_ids"):
        terminations_mod.actual_joint_position_forbidden_zone(
            env,
            non_integer_cfg,
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )


def test_done_terms_require_explicit_existing_limit_source_and_exact_runtime_shape():
    action, env, asset = _action_and_env()
    action.process_actions(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="limit_source"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "guessed_limits",
            0.0,
            0.0,
        )

    asset.data.soft_joint_pos_limits = _limits(1, 2)
    with pytest.raises(RuntimeError, match="num_envs, num_joints, 2"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )
