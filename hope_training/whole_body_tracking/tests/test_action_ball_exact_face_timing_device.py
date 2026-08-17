from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load(
    "action_ball_exact_face_timing_device_under_test",
    MDP / "action_ball_exact_face_timing_device.py",
)
G = _load(
    "racket_contact_geometry_scalar_oracle",
    MDP / "racket_contact_geometry.py",
)


def _fixture(device: torch.device, *, dtype: torch.dtype = torch.float64):
    # A non-axis-aligned fixed tape.  Each solved normal was derived by rotating
    # raw +Y with the reference quaternion, then applying another modest world
    # rotation.  This distinguishes twist preservation from rebuilding a frame
    # from the normal alone.
    ball = torch.tensor(
        [[0.52, -0.19, 0.94], [0.57, 0.24, 1.03], [0.49, 0.04, 0.87]],
        dtype=dtype,
        device=device,
    )
    face_velocity = torch.tensor(
        [[2.7, 0.4, 0.15], [2.1, -0.35, 0.5], [3.2, 0.2, -0.1]],
        dtype=dtype,
        device=device,
    )
    solved_normal = torch.tensor(
        [[-0.1951800146, 0.9759000729, 0.0975900073],
         [0.2860387768, 0.9534625892, -0.0953462589],
         [0.0990147543, 0.9901475430, 0.0990147543]],
        dtype=dtype,
        device=device,
    )
    reference_quat = torch.tensor(
        [[0.9798070079, 0.0499897520, 0.1899610575, -0.0399918016],
         [0.9649012814, -0.1099888730, 0.0399959538, 0.2349762504],
         [0.9847265389, 0.1499583515, -0.0799777875, 0.0399888937]],
        dtype=dtype,
        device=device,
    )
    reference_omega = torch.tensor(
        [[0.7, -0.3, 7.0], [-0.4, 1.2, -5.5], [0.2, 0.5, 8.2]],
        dtype=dtype,
        device=device,
    )
    return {
        "ball_contact_w_m": ball,
        "racket_face_center_velocity_w_mps": face_velocity,
        "solved_raw_a_normal_w": solved_normal,
        "mount_normal_sign": torch.tensor([1.0, -1.0, 1.0], dtype=dtype, device=device),
        "reference_racket_quat_wxyz": reference_quat,
        "reference_racket_angular_velocity_w_radps": reference_omega,
        "reference_racket_site_speed_mps": torch.tensor([2.5, 2.0, 3.0], dtype=dtype, device=device),
        "teacher_rate_min": torch.tensor([0.1, 0.1, 0.1], dtype=dtype, device=device),
        "teacher_rate_max": torch.tensor([3.0, 3.0, 3.0], dtype=dtype, device=device),
        "time_to_contact_s": torch.tensor([1.25, 1.30, 1.20], dtype=dtype, device=device),
        "reference_t_hit_s": torch.tensor([0.45, 0.40, 0.50], dtype=dtype, device=device),
        "reference_t_cycle_s": torch.tensor([0.85, 0.80, 0.90], dtype=dtype, device=device),
        "reaction_margin_s": torch.tensor([0.10, 0.10, 0.10], dtype=dtype, device=device),
        "attempt_close_margin_s": torch.tensor([0.20, 0.20, 0.20], dtype=dtype, device=device),
        "episode_length_s": torch.tensor([3.0, 3.0, 3.0], dtype=dtype, device=device),
    }


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_valid_fixed_tape_matches_scalar_exact_face_and_timing_fieldwise(device):
    values = _fixture(device)
    result = D.solve_exact_face_timing_device(**values)
    assert torch.equal(result.construction_reason, torch.full((3,), -1, dtype=torch.int64, device=device))
    assert torch.equal(result.producer_fault_bits, torch.zeros(3, dtype=torch.int64, device=device))
    assert torch.all(result.admitted)

    for row in range(3):
        def vector(name):
            return tuple(float(value) for value in values[name][row].detach().cpu())

        scalar = G.solve_exact_face_contact(
            ball_contact_w_m=vector("ball_contact_w_m"),
            racket_face_center_velocity_w_mps=vector("racket_face_center_velocity_w_mps"),
            solved_raw_a_normal_w=vector("solved_raw_a_normal_w"),
            mount_normal_sign=float(values["mount_normal_sign"][row].detach().cpu()),
            reference_racket_quat_wxyz=vector("reference_racket_quat_wxyz"),
            reference_racket_angular_velocity_w_radps=vector("reference_racket_angular_velocity_w_radps"),
            reference_racket_site_speed_mps=float(values["reference_racket_site_speed_mps"][row].detach().cpu()),
            teacher_rate_min=float(values["teacher_rate_min"][row].detach().cpu()),
            teacher_rate_max=float(values["teacher_rate_max"][row].detach().cpu()),
        )
        expected_vectors = {
            "racket_command_quat_wxyz": scalar.racket_command_quat_wxyz,
            "racket_site_target_w_m": scalar.racket_site_target_w_m,
            "racket_face_center_velocity_w_mps": scalar.racket_face_center_velocity_w_mps,
            "racket_site_velocity_w_mps": scalar.racket_site_velocity_w_mps,
            "racket_command_angular_velocity_w_radps": scalar.racket_command_angular_velocity_w_radps,
        }
        for name, expected in expected_vectors.items():
            actual = getattr(result, name)[row].detach().cpu()
            assert torch.allclose(actual, torch.tensor(expected, dtype=torch.float64), rtol=0.0, atol=4.0e-11), name
        expected_rate = scalar.teacher_rate
        expected_speed = math.sqrt(sum(value * value for value in scalar.racket_site_velocity_w_mps))
        expected_hit = float(values["reference_t_hit_s"][row].detach().cpu()) / expected_rate
        expected_cycle = float(values["reference_t_cycle_s"][row].detach().cpu()) / expected_rate
        expected_wait = float(values["time_to_contact_s"][row].detach().cpu()) - expected_hit
        for name, expected in (
            ("required_racket_site_speed_mps", expected_speed),
            ("teacher_rate", expected_rate),
            ("scaled_t_hit_s", expected_hit),
            ("scaled_t_cycle_s", expected_cycle),
            ("pre_swing_wait_s", expected_wait),
        ):
            assert float(getattr(result, name)[row].detach().cpu()) == pytest.approx(expected, rel=0.0, abs=4.0e-11), name


def _assert_row_masked(result, row: int):
    for name in (
        "racket_command_quat_wxyz",
        "racket_site_target_w_m",
        "racket_face_center_velocity_w_mps",
        "racket_site_velocity_w_mps",
        "racket_command_angular_velocity_w_radps",
        "required_racket_site_speed_mps",
        "teacher_rate",
        "scaled_t_hit_s",
        "scaled_t_cycle_s",
        "pre_swing_wait_s",
    ):
        assert torch.isnan(getattr(result, name)[row]).all(), name


def test_nonfinite_and_invalid_producer_rows_set_bits_reason_and_nan_mask():
    values = _fixture(torch.device("cpu"))
    values["ball_contact_w_m"][0, 2] = torch.nan
    values["mount_normal_sign"][1] = 0.0
    values["reference_racket_quat_wxyz"][2].zero_()
    result = D.solve_exact_face_timing_device(**values)
    assert int(result.producer_fault_bits[0]) & D.PRODUCER_FAULT_NONFINITE_GEOMETRY_INPUT
    assert int(result.producer_fault_bits[1]) & D.PRODUCER_FAULT_INVALID_FACE_SIGN
    assert int(result.producer_fault_bits[2]) & D.PRODUCER_FAULT_INVALID_REFERENCE_QUATERNION
    assert torch.equal(
        result.construction_reason,
        torch.full((3,), D.CONSTRUCTION_REASON_TEACHER_SITE_RATE_GEOMETRY_UNSOLVED, dtype=torch.int64),
    )
    for row in range(3):
        _assert_row_masked(result, row)


def test_float32_inputs_return_float32_installable_payload():
    result = D.solve_exact_face_timing_device(
        **_fixture(torch.device("cpu"), dtype=torch.float32)
    )
    assert result.admitted.all()
    for name in (
        "racket_command_quat_wxyz",
        "racket_site_target_w_m",
        "racket_face_center_velocity_w_mps",
        "racket_site_velocity_w_mps",
        "racket_command_angular_velocity_w_radps",
        "required_racket_site_speed_mps",
        "teacher_rate",
        "scaled_t_hit_s",
        "scaled_t_cycle_s",
        "pre_swing_wait_s",
    ):
        assert getattr(result, name).dtype == torch.float32


def test_canonical_rate_ulp_preserves_closed_prewait_boundary():
    values = _fixture(torch.device("cpu"))
    scalar = G.solve_exact_face_contact(
        ball_contact_w_m=tuple(float(value) for value in values["ball_contact_w_m"][0]),
        racket_face_center_velocity_w_mps=tuple(
            float(value) for value in values["racket_face_center_velocity_w_mps"][0]
        ),
        solved_raw_a_normal_w=tuple(
            float(value) for value in values["solved_raw_a_normal_w"][0]
        ),
        mount_normal_sign=float(values["mount_normal_sign"][0]),
        reference_racket_quat_wxyz=tuple(
            float(value) for value in values["reference_racket_quat_wxyz"][0]
        ),
        reference_racket_angular_velocity_w_radps=tuple(
            float(value)
            for value in values["reference_racket_angular_velocity_w_radps"][0]
        ),
        reference_racket_site_speed_mps=float(
            values["reference_racket_site_speed_mps"][0]
        ),
        teacher_rate_min=float(values["teacher_rate_min"][0]),
        teacher_rate_max=float(values["teacher_rate_max"][0]),
    )
    scalar_hit = float(values["reference_t_hit_s"][0]) / scalar.teacher_rate
    values["time_to_contact_s"][0] = scalar_hit + 1.0
    result = D.solve_exact_face_timing_device(**values)
    assert result.producer_fault_bits[0] == 0
    assert result.construction_reason[0] == D.CONSTRUCTION_REASON_ADMITTED
    assert result.teacher_rate[0] == scalar.teacher_rate
    assert result.pre_swing_wait_s[0] == 1.0


def test_scalar_quaternion_unit_preserve_seam_is_exactly_mirrored():
    values = _fixture(torch.device("cpu"))
    nextafter_one = float(
        torch.nextafter(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(float("inf"), dtype=torch.float64),
        )
    )
    values["reference_racket_quat_wxyz"][0] = torch.tensor(
        [nextafter_one, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    values["solved_raw_a_normal_w"][0] = torch.tensor(
        [0.0, 1.0, 0.0], dtype=torch.float64
    )
    result = D.solve_exact_face_timing_device(**values)
    def host_vector(name):
        return tuple(float(value) for value in values[name][0])

    scalar = G.solve_exact_face_contact(
        ball_contact_w_m=host_vector("ball_contact_w_m"),
        racket_face_center_velocity_w_mps=host_vector(
            "racket_face_center_velocity_w_mps"
        ),
        solved_raw_a_normal_w=(0.0, 1.0, 0.0),
        mount_normal_sign=1.0,
        reference_racket_quat_wxyz=(
            nextafter_one, 0.0, 0.0, 0.0
        ),
        reference_racket_angular_velocity_w_radps=host_vector(
            "reference_racket_angular_velocity_w_radps"
        ),
        reference_racket_site_speed_mps=float(
            values["reference_racket_site_speed_mps"][0]
        ),
        teacher_rate_min=float(values["teacher_rate_min"][0]),
        teacher_rate_max=float(values["teacher_rate_max"][0]),
    )
    assert tuple(result.racket_command_quat_wxyz[0]) == (
        scalar.racket_command_quat_wxyz
    )


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("degenerate_root", D.CONSTRUCTION_REASON_TEACHER_SITE_RATE_GEOMETRY_UNSOLVED),
        ("antipodal_root", D.CONSTRUCTION_REASON_TEACHER_SITE_RATE_GEOMETRY_UNSOLVED),
        ("rate", D.CONSTRUCTION_REASON_TEACHER_RATE_OUT_OF_BOUNDS),
        ("prewait", D.CONSTRUCTION_REASON_PRE_SWING_WAIT_OUT_OF_BOUNDS),
        ("cycle", D.CONSTRUCTION_REASON_CYCLE_EXCEEDS_EPISODE_HORIZON),
    ],
)
def test_adversarial_root_rate_prewait_and_cycle_are_blocked(mutation, expected_reason):
    values = _fixture(torch.device("cpu"))
    if mutation == "degenerate_root":
        values["racket_face_center_velocity_w_mps"][0].zero_()
        values["reference_racket_angular_velocity_w_radps"][0].zero_()
    elif mutation == "antipodal_root":
        reference = values["reference_racket_quat_wxyz"][0:1]
        raw_a = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        # Local helper is only used to construct a real antipodal row, not as
        # the assertion oracle.
        values["solved_raw_a_normal_w"][0] = -D._quat_rotate(reference, raw_a)[0]
    elif mutation == "rate":
        values["teacher_rate_max"][0] = 1.0
    elif mutation == "prewait":
        values["time_to_contact_s"][0] = 0.05
    else:
        values["episode_length_s"][0] = 0.9
    result = D.solve_exact_face_timing_device(**values)
    assert result.producer_fault_bits[0] == 0
    assert result.construction_reason[0] == expected_reason
    assert not result.admitted[0]
    _assert_row_masked(result, 0)


def test_source_has_no_dynamic_host_observation_or_delayed_assertion_authority():
    source = (MDP / "action_ball_exact_face_timing_device.py").read_text()
    for forbidden in (
        ".item(",
        ".cpu(",
        ".tolist(",
        ".numpy(",
        "bool(",
        "_assert_async",
    ):
        assert forbidden not in source
    assert "No production consumer is bound yet" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_hot_path_trips_on_any_synchronizing_host_observation():
    values = _fixture(torch.device("cuda"))
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        result = D.solve_exact_face_timing_device(**values)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    # This assertion stays on-device.  The test runner may synchronize after
    # the debug mode is restored; the construction function itself may not.
    assert result.construction_reason.device.type == "cuda"
