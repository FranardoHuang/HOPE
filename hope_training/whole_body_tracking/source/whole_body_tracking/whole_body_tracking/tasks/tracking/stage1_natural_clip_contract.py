"""Code-owned identities for the first ball-free natural-clip Stage-1 lanes.

This module is deliberately dependency-free.  Training, launch materialization and receipts all
consume the same rows, so a clip path, hash or strike frame cannot drift between YAML, a launcher
and the progress ledger without failing closed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage1NaturalClipLane:
    lane_id: str
    action_id: str
    side: str
    motion_path: str
    motion_sha256: str
    frame_count: int
    strike_frame: int
    cycle_seconds: float

    @property
    def strike_phase(self) -> float:
        return self.strike_frame / (self.frame_count - 1)


STAGE1_NATURAL_CLIP_LANES = (
    Stage1NaturalClipLane(
        lane_id="bh_quality_take061_unit15",
        action_id="take_061_unit15_bh",
        side="BH",
        motion_path=(
            "assets/motions/chingmu73_20260728/"
            "hope_Take_061_unit15_BH.npz"
        ),
        motion_sha256=(
            "476db8cabb9d00c300f88f7b2e2e7846d4802a126fac98a1da499a4762fdeebf"
        ),
        frame_count=96,
        strike_frame=49,
        cycle_seconds=1.90,
    ),
    Stage1NaturalClipLane(
        lane_id="fh_stable_take058_unit04",
        action_id="take_058_unit04_fh",
        side="FH",
        motion_path=(
            "assets/motions/chingmu73_20260728/"
            "hope_Take_058_unit04_FH.npz"
        ),
        motion_sha256=(
            "6b60255f0530e50b9d37f863d3f9c1b68bd25d35c8dd6bfc54a52bd045a89a7d"
        ),
        frame_count=91,
        strike_frame=49,
        cycle_seconds=1.80,
    ),
    Stage1NaturalClipLane(
        lane_id="bh_diverse_take060_unit09",
        action_id="take_060_unit09_bh",
        side="BH",
        motion_path=(
            "assets/motions/chingmu73_20260728/"
            "hope_Take_060_unit09_BH.npz"
        ),
        motion_sha256=(
            "6d6ff7621267a2bbcb20aeeedba719a10a6b6e6a49eeee9586f78031762073f1"
        ),
        frame_count=66,
        strike_frame=36,
        cycle_seconds=1.30,
    ),
)

STAGE1_NATURAL_CLIP_LANES_BY_ID = {
    lane.lane_id: lane for lane in STAGE1_NATURAL_CLIP_LANES
}
STAGE1_NATURAL_CLIP_LANES_BY_SHA256 = {
    lane.motion_sha256: lane for lane in STAGE1_NATURAL_CLIP_LANES
}

if len(STAGE1_NATURAL_CLIP_LANES_BY_ID) != len(STAGE1_NATURAL_CLIP_LANES):
    raise RuntimeError("Stage-1 natural-clip lane_id values must be unique")
if len(STAGE1_NATURAL_CLIP_LANES_BY_SHA256) != len(
    STAGE1_NATURAL_CLIP_LANES
):
    raise RuntimeError("Stage-1 natural-clip motion SHA-256 values must be unique")
