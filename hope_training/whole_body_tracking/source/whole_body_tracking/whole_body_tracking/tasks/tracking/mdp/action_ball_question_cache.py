"""Exact, active-birth-bounded cache for ActionBall curriculum questions.

This is deliberately not a question source and not a distribution cache.  The
curriculum/sampler still emits every question and owns its RNG transcript.  We
only reuse the deterministic numeric solver answer when the *complete semantic
question* is byte-identical.  A changed level, band/stratum, base pose, physics
pin, motion pin, or any continuous ball/aim value therefore misses naturally.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence


_SCHEMA_VERSION = 2


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def exact_question_sha256(payload: Mapping[str, object]) -> str:
    """Hash one complete, finite semantic question."""

    if not isinstance(payload, Mapping):
        raise TypeError("question payload must be a mapping")
    # The caller constructs a fresh plain-JSON payload and hashing is
    # synchronous.  A deep copy here would add roughly one extra full tree
    # traversal per environment (4096 times at reset) without strengthening
    # the digest; ``json.dumps`` never mutates its input.
    frozen = dict(payload)
    return _canonical_sha256(
        {
            "schema_version": _SCHEMA_VERSION,
            "kind": "action_ball_exact_curriculum_question",
            "payload": frozen,
        }
    )


@dataclass(frozen=True)
class CachedQuestionAnswer:
    """One solver row, encoded with exact IEEE-754 payload bits."""

    reason_code: int
    admitted: bool
    float64_bits_hex: str

    def __post_init__(self) -> None:
        if type(self.reason_code) is not int or type(self.admitted) is not bool:
            raise TypeError("cached reason/admission values have invalid types")
        if type(self.float64_bits_hex) is not str:
            raise TypeError("cached solver float payload must be a string")

    @classmethod
    def from_values(
        cls,
        *,
        reason_code: int,
        admitted: bool,
        racket_velocity: Sequence[float],
        racket_normal: Sequence[float],
        residual: float,
    ) -> "CachedQuestionAnswer":
        if type(reason_code) is not int or type(admitted) is not bool:
            raise TypeError("cached reason/admission values have invalid types")
        velocity = tuple(float(value) for value in racket_velocity)
        normal = tuple(float(value) for value in racket_normal)
        if len(velocity) != 3 or len(normal) != 3:
            raise ValueError("cached racket rows must each have width three")
        values = (*velocity, *normal, float(residual))
        return cls(
            reason_code=reason_code,
            admitted=admitted,
            float64_bits_hex=struct.pack("!7d", *values).hex(),
        )

    def values(self) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        try:
            raw = bytes.fromhex(self.float64_bits_hex)
        except ValueError as error:
            raise ValueError("cached solver float payload is not hexadecimal") from error
        if len(raw) != 56:
            raise ValueError("cached solver float payload has the wrong width")
        values = struct.unpack("!7d", raw)
        velocity = tuple(float(value) for value in values[:3])
        normal = tuple(float(value) for value in values[3:6])
        residual = float(values[6])
        if self.admitted:
            if self.reason_code != -1 or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError("admitted cached answer is not finite/reason=-1")
        elif self.reason_code < 0:
            raise ValueError("rejected cached answer has no rejection code")
        return velocity, normal, residual

    def to_dict(self) -> dict[str, object]:
        # Decode once on serialization so corrupt in-memory rows fail before a
        # checkpoint can bless them.
        self.values()
        return {
            "reason_code": self.reason_code,
            "admitted": self.admitted,
            "float64_bits_hex": self.float64_bits_hex,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CachedQuestionAnswer":
        if not isinstance(value, Mapping) or set(value) != {
            "reason_code",
            "admitted",
            "float64_bits_hex",
        }:
            raise ValueError("cached answer has invalid keys")
        answer = cls(
            reason_code=value["reason_code"],  # type: ignore[arg-type]
            admitted=value["admitted"],  # type: ignore[arg-type]
            float64_bits_hex=value["float64_bits_hex"],  # type: ignore[arg-type]
        )
        answer.values()
        return answer


class ExactCurriculumQuestionCache:
    """Exact answers retained for active births plus one hot row per action.

    A formal refill may contain several distinct questions for the same action.
    Every answer referenced by an active episode birth must therefore coexist:
    replacing the previous row would make the pool's immediate pure replay
    fail.  Retirement drops rows no longer referenced by any live birth while
    retaining the most recently consumed exact question per action.  Capacity
    is consequently bounded by active-birth semantic rows plus the action
    catalog, rather than by training lifetime or by one unsafe row per action.
    """

    def __init__(self, action_uids: Sequence[int]) -> None:
        uids = tuple(action_uids)
        if not uids or any(type(uid) is not int or uid <= 0 for uid in uids):
            raise ValueError("cache action_uids must be positive plain integers")
        if len(set(uids)) != len(uids):
            raise ValueError("cache action_uids must be unique")
        self._action_uids = tuple(sorted(uids))
        self._rows: dict[int, dict[str, CachedQuestionAnswer]] = {}
        self._last_question_by_uid: dict[int, str] = {}
        self._birth_keys: dict[str, set[tuple[int, str]]] = {}
        self._consumer_hit_count = 0
        self._novel_producer_count = 0

    @property
    def consumer_hit_count(self) -> int:
        return self._consumer_hit_count

    @property
    def novel_producer_count(self) -> int:
        return self._novel_producer_count

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self._rows.values())

    @property
    def active_birth_count(self) -> int:
        return len(self._birth_keys)

    @property
    def active_birth_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(self._birth_keys))

    def clone(self) -> "ExactCurriculumQuestionCache":
        clone = ExactCurriculumQuestionCache(self._action_uids)
        clone.load_state_dict(self.state_dict())
        return clone

    @staticmethod
    def _question_digest(value: object) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("question_sha256 must be lowercase SHA-256")
        return value

    @staticmethod
    def _birth_digest(value: object) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("birth_sha256 must be lowercase SHA-256")
        return value

    def _bind_birth(
        self,
        *,
        action_uid: int,
        question_sha256: str,
        birth_sha256: str | None,
    ) -> None:
        if birth_sha256 is None:
            return
        birth = self._birth_digest(birth_sha256)
        self._birth_keys.setdefault(birth, set()).add(
            (action_uid, question_sha256)
        )

    def _set_hot(self, *, action_uid: int, question_sha256: str) -> None:
        previous = self._last_question_by_uid.get(action_uid)
        self._last_question_by_uid[action_uid] = question_sha256
        if previous is None or previous == question_sha256:
            return
        previous_identity = (action_uid, previous)
        if any(
            previous_identity in identities
            for identities in self._birth_keys.values()
        ):
            return
        action_rows = self._rows.get(action_uid)
        if action_rows is not None:
            action_rows.pop(previous, None)
            if not action_rows:
                del self._rows[action_uid]

    def peek(self, *, action_uid: int, question_sha256: str) -> CachedQuestionAnswer | None:
        if action_uid not in self._action_uids:
            raise ValueError(f"unknown cache action_uid {action_uid!r}")
        digest = self._question_digest(question_sha256)
        return self._rows.get(action_uid, {}).get(digest)

    def note_hit(
        self,
        *,
        action_uid: int,
        question_sha256: str,
        birth_sha256: str | None = None,
    ) -> CachedQuestionAnswer:
        answer = self.peek(action_uid=action_uid, question_sha256=question_sha256)
        if answer is None:
            raise RuntimeError("cannot count a cache hit for a missing exact question")
        self._bind_birth(
            action_uid=action_uid,
            question_sha256=question_sha256,
            birth_sha256=birth_sha256,
        )
        self._set_hot(
            action_uid=action_uid,
            question_sha256=question_sha256,
        )
        self._consumer_hit_count += 1
        return answer

    def note_in_batch_reuse(
        self,
        *,
        action_uid: int,
        question_sha256: str,
        answer: CachedQuestionAnswer,
        birth_sha256: str | None = None,
    ) -> None:
        """Count a duplicate of a novel row solved earlier in the same batch.

        The caller still holds the exact staged answer and supplies it here for
        validation.  The installed row remains retained for the owning birth,
        so subsequent pool replay never needs to call the inverse solver.
        """

        if action_uid not in self._action_uids:
            raise ValueError(f"unknown cache action_uid {action_uid!r}")
        self._question_digest(question_sha256)
        if not isinstance(answer, CachedQuestionAnswer):
            raise TypeError("cache answer must be CachedQuestionAnswer")
        answer.values()
        retained = self.peek(
            action_uid=action_uid,
            question_sha256=question_sha256,
        )
        if retained != answer:
            raise RuntimeError("in-batch reuse answer differs from retained exact row")
        self._bind_birth(
            action_uid=action_uid,
            question_sha256=question_sha256,
            birth_sha256=birth_sha256,
        )
        self._set_hot(
            action_uid=action_uid,
            question_sha256=question_sha256,
        )
        self._consumer_hit_count += 1

    def install_novel(
        self,
        *,
        action_uid: int,
        question_sha256: str,
        answer: CachedQuestionAnswer,
        birth_sha256: str | None = None,
    ) -> None:
        if self.peek(action_uid=action_uid, question_sha256=question_sha256) is not None:
            raise RuntimeError("exact question is already cached; count it as a hit")
        if not isinstance(answer, CachedQuestionAnswer):
            raise TypeError("cache answer must be CachedQuestionAnswer")
        answer.values()
        self._rows.setdefault(action_uid, {})[question_sha256] = answer
        self._bind_birth(
            action_uid=action_uid,
            question_sha256=question_sha256,
            birth_sha256=birth_sha256,
        )
        self._set_hot(
            action_uid=action_uid,
            question_sha256=question_sha256,
        )
        self._novel_producer_count += 1

    def retire_births(self, birth_sha256s: Sequence[str]) -> None:
        """Drop rows whose final active-birth owner has retired.

        The per-action most-recent row stays hot across episode resets, which
        is the intended static-question fast path.  Every other row is kept
        exactly while at least one active birth can still ask the pool to
        replay a receipt that references it.
        """

        births = tuple(self._birth_digest(value) for value in birth_sha256s)
        if len(set(births)) != len(births):
            raise ValueError("retired birth SHA values must be unique")
        for birth in births:
            # A birth may retire before it ever requested a task; that is a
            # legitimate no-row retirement, not a cache authority failure.
            self._birth_keys.pop(birth, None)
        referenced = {
            identity
            for identities in self._birth_keys.values()
            for identity in identities
        }
        hot = set(self._last_question_by_uid.items())
        retained = referenced | hot
        for uid in tuple(self._rows):
            self._rows[uid] = {
                digest: answer
                for digest, answer in self._rows[uid].items()
                if (uid, digest) in retained
            }
            if not self._rows[uid]:
                del self._rows[uid]

    def state_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "action_ball_exact_curriculum_question_cache",
            "action_uids": list(self._action_uids),
            "capacity_policy": "active_birth_rows_plus_one_hot_row_per_action",
            "consumer_hit_count": self._consumer_hit_count,
            "novel_producer_count": self._novel_producer_count,
            "last_questions": [
                {
                    "action_uid": uid,
                    "question_sha256": self._last_question_by_uid[uid],
                }
                for uid in sorted(self._last_question_by_uid)
            ],
            "birth_refs": [
                {
                    "birth_sha256": birth,
                    "question_keys": [
                        {
                            "action_uid": uid,
                            "question_sha256": digest,
                        }
                        for uid, digest in sorted(self._birth_keys[birth])
                    ],
                }
                for birth in sorted(self._birth_keys)
            ],
            "rows": [
                {
                    "action_uid": uid,
                    "question_sha256": digest,
                    "answer": answer.to_dict(),
                }
                for uid in sorted(self._rows)
                for digest, answer in sorted(self._rows[uid].items())
            ],
        }
        return {**payload, "integrity_sha256": _canonical_sha256(payload)}

    def load_state_dict(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("question cache state must be a mapping")
        expected = {
            "schema_version",
            "kind",
            "action_uids",
            "capacity_policy",
            "consumer_hit_count",
            "novel_producer_count",
            "last_questions",
            "birth_refs",
            "rows",
            "integrity_sha256",
        }
        if set(state) != expected:
            raise ValueError("question cache state has invalid keys")
        payload = {key: copy.deepcopy(state[key]) for key in expected if key != "integrity_sha256"}
        if state["integrity_sha256"] != _canonical_sha256(payload):
            raise ValueError("question cache state integrity mismatch")
        if (
            state["schema_version"] != _SCHEMA_VERSION
            or state["kind"] != "action_ball_exact_curriculum_question_cache"
            or state["action_uids"] != list(self._action_uids)
            or state["capacity_policy"]
            != "active_birth_rows_plus_one_hot_row_per_action"
        ):
            raise ValueError("question cache immutable identity mismatch")
        hits = state["consumer_hit_count"]
        novel = state["novel_producer_count"]
        if type(hits) is not int or hits < 0 or type(novel) is not int or novel < 0:
            raise ValueError("question cache counters are invalid")
        raw_rows = state["rows"]
        if not isinstance(raw_rows, list):
            raise ValueError("question cache rows must be a list")
        rows: dict[int, dict[str, CachedQuestionAnswer]] = {}
        previous_identity: tuple[int, str] | None = None
        for raw in raw_rows:
            if not isinstance(raw, Mapping) or set(raw) != {
                "action_uid",
                "question_sha256",
                "answer",
            }:
                raise ValueError("question cache row has invalid keys")
            uid = raw["action_uid"]
            digest = raw["question_sha256"]
            if type(uid) is not int or uid not in self._action_uids:
                raise ValueError("question cache row has unknown action UID")
            # Reuse peek validation without mutating this object.
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("question cache row digest is invalid")
            identity = (uid, digest)
            if previous_identity is not None and identity <= previous_identity:
                raise ValueError("question cache rows must be unique and ordered")
            previous_identity = identity
            rows.setdefault(uid, {})[digest] = CachedQuestionAnswer.from_dict(
                raw["answer"]
            )

        raw_last = state["last_questions"]
        if not isinstance(raw_last, list):
            raise ValueError("question cache last_questions must be a list")
        last_questions: dict[int, str] = {}
        previous_uid = 0
        for raw in raw_last:
            if not isinstance(raw, Mapping) or set(raw) != {
                "action_uid",
                "question_sha256",
            }:
                raise ValueError("question cache hot row has invalid keys")
            uid = raw["action_uid"]
            digest = self._question_digest(raw["question_sha256"])
            if (
                type(uid) is not int
                or uid not in self._action_uids
                or uid <= previous_uid
                or digest not in rows.get(uid, {})
            ):
                raise ValueError("question cache hot rows are invalid")
            previous_uid = uid
            last_questions[uid] = digest

        raw_birth_refs = state["birth_refs"]
        if not isinstance(raw_birth_refs, list):
            raise ValueError("question cache birth_refs must be a list")
        birth_keys: dict[str, set[tuple[int, str]]] = {}
        previous_birth = ""
        for raw in raw_birth_refs:
            if not isinstance(raw, Mapping) or set(raw) != {
                "birth_sha256",
                "question_keys",
            }:
                raise ValueError("question cache birth ref has invalid keys")
            birth = self._birth_digest(raw["birth_sha256"])
            if birth <= previous_birth:
                raise ValueError("question cache birth refs must be unique and ordered")
            previous_birth = birth
            raw_keys = raw["question_keys"]
            if not isinstance(raw_keys, list) or not raw_keys:
                raise ValueError("question cache birth ref must own at least one row")
            identities: set[tuple[int, str]] = set()
            previous_key: tuple[int, str] | None = None
            for raw_key in raw_keys:
                if not isinstance(raw_key, Mapping) or set(raw_key) != {
                    "action_uid",
                    "question_sha256",
                }:
                    raise ValueError("question cache birth key has invalid keys")
                uid = raw_key["action_uid"]
                digest = self._question_digest(raw_key["question_sha256"])
                identity = (uid, digest)
                if (
                    type(uid) is not int
                    or uid not in self._action_uids
                    or digest not in rows.get(uid, {})
                    or (previous_key is not None and identity <= previous_key)
                ):
                    raise ValueError("question cache birth keys are invalid")
                previous_key = identity
                identities.add(identity)
            birth_keys[birth] = identities
        retained = {
            identity
            for identities in birth_keys.values()
            for identity in identities
        } | set(last_questions.items())
        serialized = {
            (uid, digest)
            for uid, action_rows in rows.items()
            for digest in action_rows
        }
        if serialized != retained:
            raise ValueError("question cache contains unowned non-hot rows")
        # Atomic commit after every row has validated.
        self._rows = rows
        self._last_question_by_uid = last_questions
        self._birth_keys = birth_keys
        self._consumer_hit_count = hits
        self._novel_producer_count = novel
