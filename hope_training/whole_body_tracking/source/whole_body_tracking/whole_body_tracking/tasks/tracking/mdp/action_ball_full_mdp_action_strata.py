"""Pure constants and immutable catalog data for action-only telemetry."""

from dataclasses import dataclass


STROKE_FAMILY_UNKNOWN = 0
STROKE_FAMILY_FOREHAND = 1
STROKE_FAMILY_BACKHAND = 2
STROKE_FAMILY_NAMES = ("unknown", "forehand", "backhand")


@dataclass(frozen=True)
class ActionStrokeFamilyCatalog:
    action_uids: tuple[int, ...]
    family_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.action_uids) is not tuple
            or not self.action_uids
            or type(self.family_codes) is not tuple
            or len(self.action_uids) != len(self.family_codes)
            or any(type(uid) is not int or uid <= 0 for uid in self.action_uids)
            or len(set(self.action_uids)) != len(self.action_uids)
            or any(type(code) is not int or code not in (1, 2) for code in self.family_codes)
        ):
            raise ValueError("action stroke-family catalog differs")

    def clone(self) -> "ActionStrokeFamilyCatalog":
        return ActionStrokeFamilyCatalog(self.action_uids, self.family_codes)
