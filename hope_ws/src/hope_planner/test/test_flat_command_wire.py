import math

import pytest

from hope_planner.flat_command_wire import (
    RACKET_FLAT_SCHEMA_V1,
    RACKET_FLAT_SCHEMA_V2_FACE179,
    RACKET_FLAT_V1_SIZE,
    RACKET_FLAT_V2_SIZE,
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
    ],
)
def test_schema2_rejects_malformed_or_non_opponent_facing_command(kwargs):
    with pytest.raises(ValueError):
        pack_racket_command_flat(
            schema=RACKET_FLAT_SCHEMA_V2_FACE179,
            **BASE,
            **kwargs,
        )


def test_unknown_schema_and_frame_fail_closed():
    with pytest.raises(ValueError):
        pack_racket_command_flat(schema=3, **BASE)
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
