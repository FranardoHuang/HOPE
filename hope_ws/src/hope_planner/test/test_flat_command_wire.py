import math
from pathlib import Path

import pytest

from hope_planner.flat_command_wire import (
    BASE_FLAT_SCHEMA_V1,
    BASE_FLAT_SCHEMA_V2_EPOCH,
    BASE_FLAT_V1_SIZE,
    BASE_FLAT_V2_SIZE,
    MAX_EXACT_FLOAT64_INTEGER,
    RACKET_FLAT_SCHEMA_V1,
    RACKET_FLAT_SCHEMA_V2_FACE179,
    RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
    RACKET_FLAT_V1_SIZE,
    RACKET_FLAT_V2_SIZE,
    RACKET_FLAT_V3_SIZE,
    pack_base_pose_flat,
    pack_invalid_base_pose_flat,
    pack_invalid_racket_command_flat,
    pack_racket_command_flat,
    pack_racket_command_flat_fail_closed,
)


BASE = dict(
    valid=True,
    swing_sign=-1.0,
    position_w=(0.1, 0.2, 0.3),
    velocity_w=(1.1, 1.2, 1.3),
    time_to_strike=0.4,
    strike_time=5.0,
    frame_code=0,
)


def test_schema1_preserves_legacy_twelve_value_layout():
    row = pack_racket_command_flat(schema=RACKET_FLAT_SCHEMA_V1, **BASE)
    assert len(row) == RACKET_FLAT_V1_SIZE
    assert row == [1.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 0.4, 5.0, 0.0]


def test_schema2_appends_atomic_world_normal_and_zero_rho():
    row = pack_racket_command_flat(
        schema=RACKET_FLAT_SCHEMA_V2_FACE179,
        normal_cmd_w=(0.6, 0.8, 0.0),
        rho=0.0,
        **BASE,
    )
    assert len(row) == RACKET_FLAT_V2_SIZE
    assert row[:12] == [2.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 0.4, 5.0, 0.0]
    assert row[12:] == [0.6, 0.8, 0.0, 0.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normal_cmd_w": None},
        {"normal_cmd_w": (1.0, 0.0, 0.01)},
        {"normal_cmd_w": (math.nan, 0.0, 1.0)},
        {"normal_cmd_w": (-1.0, 0.0, 0.0)},
        {"normal_cmd_w": (0.0, 0.0, 1.0)},
        {"normal_cmd_w": (1.0, 0.0, 0.0), "rho": 0.1},
        {"normal_cmd_w": (1.0, 0.0, 0.0), "swing_sign": 0.0},
    ],
)
def test_schema2_rejects_malformed_or_non_opponent_facing_command(kwargs):
    with pytest.raises(ValueError):
        pack_racket_command_flat(
            schema=RACKET_FLAT_SCHEMA_V2_FACE179,
            **{**BASE, **kwargs},
        )


def test_unknown_schema_and_frame_fail_closed():
    with pytest.raises(ValueError):
        pack_racket_command_flat(schema=4, **BASE)
    with pytest.raises(ValueError):
        pack_racket_command_flat(schema=1, **{**BASE, "frame_code": 2})
    with pytest.raises(ValueError):
        pack_racket_command_flat(
            schema=RACKET_FLAT_SCHEMA_V2_FACE179,
            normal_cmd_w=(1.0, 0.0, 0.0),
            **{**BASE, "frame_code": 1},
        )


def test_schema2_publisher_turns_bad_payload_into_exact_invalid_revocation():
    good, error = pack_racket_command_flat_fail_closed(
        schema=RACKET_FLAT_SCHEMA_V2_FACE179,
        normal_cmd_w=(1.0, 0.0, 0.0),
        **BASE,
    )
    assert error is None and good[1] == 1.0

    revoked, error = pack_racket_command_flat_fail_closed(
        schema=RACKET_FLAT_SCHEMA_V2_FACE179,
        normal_cmd_w=(-1.0, 0.0, 0.0),
        **BASE,
    )
    assert error is not None
    assert revoked == pack_invalid_racket_command_flat(
        schema=RACKET_FLAT_SCHEMA_V2_FACE179
    )
    assert len(revoked) == RACKET_FLAT_V2_SIZE
    assert revoked[1] == 0.0
    assert revoked[11] == 0.0
    assert revoked[12:] == [1.0, 0.0, 0.0, 0.0]


def test_schema3_binds_epoch_base_sequence_and_same_host_monotonic_stamp():
    row = pack_racket_command_flat(
        schema=RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
        normal_cmd_w=(1.0, 0.0, 0.0),
        control_epoch=7,
        command_sequence=11,
        base_sequence_ref=5,
        source_monotonic_s=123.5,
        **BASE,
    )
    assert len(row) == RACKET_FLAT_V3_SIZE
    assert row[:16] == [
        3.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 0.4, 5.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ]
    assert row[16:] == [7.0, 11.0, 5.0, 123.5]
    assert pack_invalid_racket_command_flat(
        schema=RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
        control_epoch=8,
        command_sequence=12,
        base_sequence_ref=6,
        source_monotonic_s=124.0,
    )[16:] == [8.0, 12.0, 6.0, 124.0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_epoch", -1),
        ("control_epoch", 1.5),
        ("control_epoch", math.nan),
        ("command_sequence", MAX_EXACT_FLOAT64_INTEGER + 1),
        ("base_sequence_ref", -1),
        ("base_sequence_ref", 1.5),
        ("base_sequence_ref", MAX_EXACT_FLOAT64_INTEGER + 1),
        ("source_monotonic_s", -0.1),
        ("source_monotonic_s", math.inf),
    ],
)
def test_schema3_rejects_unrepresentable_causality_metadata(field, value):
    kwargs = dict(
        schema=RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
        normal_cmd_w=(1.0, 0.0, 0.0),
        control_epoch=7,
        command_sequence=11,
        base_sequence_ref=5,
        source_monotonic_s=123.5,
        **BASE,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        pack_racket_command_flat(**kwargs)


def test_schema3_fail_closed_revocation_keeps_exact_base_reference_and_size():
    revoked, error = pack_racket_command_flat_fail_closed(
        schema=RACKET_FLAT_SCHEMA_V3_FACE179_EPOCH,
        normal_cmd_w=(-1.0, 0.0, 0.0),
        control_epoch=MAX_EXACT_FLOAT64_INTEGER,
        command_sequence=MAX_EXACT_FLOAT64_INTEGER,
        base_sequence_ref=MAX_EXACT_FLOAT64_INTEGER,
        source_monotonic_s=123.5,
        **BASE,
    )
    assert error is not None
    assert len(revoked) == RACKET_FLAT_V3_SIZE == 20
    assert revoked[1] == 0.0
    assert revoked[16:] == [
        float(MAX_EXACT_FLOAT64_INTEGER),
        float(MAX_EXACT_FLOAT64_INTEGER),
        float(MAX_EXACT_FLOAT64_INTEGER),
        123.5,
    ]


def test_base_schema2_carries_same_epoch_and_has_canonical_invalid_row():
    legacy = pack_base_pose_flat(
        schema=BASE_FLAT_SCHEMA_V1,
        valid=True,
        position_w=(1.0, 2.0, 0.9),
        quaternion_wxyz=(2.0, 0.0, 0.0, 0.0),
    )
    assert len(legacy) == BASE_FLAT_V1_SIZE
    assert legacy == [1.0, 1.0, 1.0, 2.0, 0.9, 1.0, 0.0, 0.0, 0.0]

    formal = pack_base_pose_flat(
        schema=BASE_FLAT_SCHEMA_V2_EPOCH,
        valid=True,
        position_w=(1.0, 2.0, 0.9),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        control_epoch=7,
        base_sequence=5,
        source_monotonic_s=123.0,
    )
    assert len(formal) == BASE_FLAT_V2_SIZE
    assert formal[9:] == [7.0, 5.0, 123.0]
    revoked = pack_invalid_base_pose_flat(
        schema=BASE_FLAT_SCHEMA_V2_EPOCH,
        control_epoch=8,
        base_sequence=6,
        source_monotonic_s=124.0,
    )
    assert revoked == [
        2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        8.0, 6.0, 124.0,
    ]


def test_formal_flat_publication_precedes_best_effort_custom_mirror():
    node_source = (
        Path(__file__).resolve().parents[1] / "hope_planner" / "node.py"
    ).read_text(encoding="utf-8")
    section = node_source.split(
        "# The flat wire is the formal Gate-3 control path.", 1
    )[1].split("# `valid_commands` is the control-visible count", 1)[0]
    assert section.index("self.flat_cmd_pub.publish(fm)") < section.index(
        "self.cmd_pub.publish(out)"
    )
    assert "except Exception as exc" in section
