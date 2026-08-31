#!/usr/bin/env python3
"""Deterministic, finite-support targets for continuous ActionBall shots.

This module is deliberately independent of Isaac, MuJoCo, NumPy, and Torch.
It seals a finite target profile and owns one deterministic random stream per
environment.  A shot target is revealed only after all construction-time
feasibility predicates have been materialized.  Selection is then one direct
draw from the feasible set with the previous *numerical* target removed; there
is no rejection loop and an infeasible proposal is never a policy opportunity.

The semantic target identity excludes ``cell_id``.  It binds the coordinate
frame, frame authority, ordered components, runtime ``float32`` dtype, exact
quantization contract, and canonical big-endian binary32 bytes.  Consequently
a renamed cell cannot make the same runtime numerical target count as a fresh
question, and neither float64-only distinctions nor ``-0.0`` can evade the
adjacent-target exclusion.

This is a pre-integration primitive only.  It does not read the current fixed
center target, mutate a runtime, or claim that continuous play is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import struct
from typing import Callable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
PROFILE_KIND = "action_ball_continuous_target_profile_v2"
TARGET_SEMANTIC_KIND = "action_ball_continuous_target_semantic_v2"
SELECTION_KIND = "action_ball_continuous_target_selection_v2"
CHECKPOINT_KIND = "action_ball_continuous_target_sampler_checkpoint_v2"
STREAM_KIND = "action_ball_continuous_target_sampler_stream_v2"
LEGACY_PROFILE_KIND = "action_ball_continuous_target_profile_v1"
LEGACY_CHECKPOINT_KIND = (
    "action_ball_continuous_target_sampler_checkpoint_v1"
)
RUNTIME_DTYPE = "float32"
QUANTIZATION_CONTRACT = (
    "ieee754_binary32_round_ties_to_even_big_endian_bytes_v1"
)

_U64_MODULUS = 1 << 64
_U64_MASK = _U64_MODULUS - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


class ContinuousTargetSamplerError(ValueError):
    """Base class for fail-closed target-profile and sampler failures."""


class NoFeasibleDifferentTargetError(ContinuousTargetSamplerError):
    """No construction-feasible target differs from the previous target."""


class TargetFeasibilityError(ContinuousTargetSamplerError):
    """A feasibility producer did not return one exact, complete bool mask."""


def canonical_sha256(value: object) -> str:
    """Return repository-style canonical JSON SHA-256 bytes."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ContinuousTargetSamplerError(
            "value is not finite canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def target_semantic_sha256(
    *,
    frame_id: str,
    frame_binding_sha256: str,
    runtime_dtype: str,
    quantization_contract: str,
    components: Sequence[str],
    target: Sequence[Real],
) -> str:
    """Hash only numerical target semantics, never a replaceable cell ID."""

    clean_frame = _nonempty_text(frame_id, label="frame_id")
    clean_binding = _sha256(
        frame_binding_sha256, label="frame_binding_sha256"
    )
    clean_dtype = _runtime_dtype(runtime_dtype)
    clean_quantization = _quantization_contract(quantization_contract)
    clean_components = _components(components)
    clean_target = _runtime_target_tuple(
        target,
        expected_width=len(clean_components),
        label="target",
    )
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": TARGET_SEMANTIC_KIND,
            "frame_id": clean_frame,
            "frame_binding_sha256": clean_binding,
            "runtime_dtype": clean_dtype,
            "quantization_contract": clean_quantization,
            "components": list(clean_components),
            "target_f32_be_hex": [
                _float32_be_hex(item) for item in clean_target
            ],
        }
    )


@dataclass(frozen=True)
class TargetCell:
    """One named point in a finite target support."""

    cell_id: str
    target: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cell_id", _nonempty_text(self.cell_id, label="cell_id")
        )
        object.__setattr__(
            self,
            "target",
            _target_tuple(self.target, expected_width=None, label="target"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"cell_id": self.cell_id, "target": list(self.target)}

    @classmethod
    def from_mapping(cls, value: object) -> "TargetCell":
        payload = _exact_mapping(
            value, ("cell_id", "target"), label="target cell"
        )
        return cls(cell_id=payload["cell_id"], target=payload["target"])


@dataclass(frozen=True)
class ContinuousTargetProfile:
    """Content-addressed finite target support in one bound frame."""

    frame_id: str
    frame_binding_sha256: str
    runtime_dtype: str
    quantization_contract: str
    components: Tuple[str, ...]
    cells: Tuple[TargetCell, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", _nonempty_text(self.frame_id, label="frame_id")
        )
        object.__setattr__(
            self,
            "frame_binding_sha256",
            _sha256(
                self.frame_binding_sha256, label="frame_binding_sha256"
            ),
        )
        object.__setattr__(
            self,
            "runtime_dtype",
            _runtime_dtype(self.runtime_dtype),
        )
        object.__setattr__(
            self,
            "quantization_contract",
            _quantization_contract(self.quantization_contract),
        )
        clean_components = _components(self.components)
        object.__setattr__(self, "components", clean_components)

        if not isinstance(self.cells, (tuple, list)):
            raise ContinuousTargetSamplerError("cells must be a finite sequence")
        raw_cells = tuple(self.cells)
        clean_cells = tuple(
            TargetCell(
                cell_id=cell.cell_id,
                target=_runtime_target_tuple(
                    cell.target,
                    expected_width=len(clean_components),
                    label="cells[%d].target" % index,
                ),
            )
            if isinstance(cell, TargetCell)
            else cell
            for index, cell in enumerate(raw_cells)
        )
        if len(clean_cells) < 2:
            raise ContinuousTargetSamplerError(
                "target support must contain at least 2 selectable cells"
            )
        if any(not isinstance(cell, TargetCell) for cell in clean_cells):
            raise ContinuousTargetSamplerError(
                "every target support entry must be a TargetCell"
            )
        object.__setattr__(self, "cells", clean_cells)

        cell_ids: set[str] = set()
        for index, cell in enumerate(clean_cells):
            if cell.cell_id in cell_ids:
                raise ContinuousTargetSamplerError(
                    "target support cell_id values must be unique"
                )
            cell_ids.add(cell.cell_id)
            if len(cell.target) != len(clean_components):
                raise ContinuousTargetSamplerError(
                    "cells[%d].target width differs from components" % index
                )
            # Cell IDs describe fixed-width draw slots, not distinct physical
            # coordinates.  A zero-width action domain legitimately repeats
            # one target while retaining deterministic slot identity.

    def semantic_sha256(self, cell: TargetCell) -> str:
        if not isinstance(cell, TargetCell):
            raise ContinuousTargetSamplerError("cell must be a TargetCell")
        return target_semantic_sha256(
            frame_id=self.frame_id,
            frame_binding_sha256=self.frame_binding_sha256,
            runtime_dtype=self.runtime_dtype,
            quantization_contract=self.quantization_contract,
            components=self.components,
            target=cell.target,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PROFILE_KIND,
            "frame_id": self.frame_id,
            "frame_binding_sha256": self.frame_binding_sha256,
            "runtime_dtype": self.runtime_dtype,
            "quantization_contract": self.quantization_contract,
            "components": list(self.components),
            "cells": [cell.to_mapping() for cell in self.cells],
        }

    @property
    def profile_sha256(self) -> str:
        return canonical_sha256(self._payload())

    @property
    def canonical_sha256(self) -> str:
        """Compatibility spelling used by adjacent content-addressed types."""

        return self.profile_sha256

    def to_mapping(self) -> dict[str, object]:
        payload = self._payload()
        payload["profile_sha256"] = self.profile_sha256
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousTargetProfile":
        _reject_legacy_v1_envelope(
            value,
            label="target profile",
            legacy_kind=LEGACY_PROFILE_KIND,
        )
        expected = (
            "schema_version",
            "kind",
            "frame_id",
            "frame_binding_sha256",
            "runtime_dtype",
            "quantization_contract",
            "components",
            "cells",
            "profile_sha256",
        )
        payload = _exact_mapping(value, expected, label="target profile")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
        ):
            raise ContinuousTargetSamplerError(
                "target profile schema_version differs"
            )
        if type(payload["kind"]) is not str or payload["kind"] != PROFILE_KIND:
            raise ContinuousTargetSamplerError("target profile kind differs")
        declared_sha = _sha256(
            payload["profile_sha256"], label="profile_sha256"
        )
        raw_cells = payload["cells"]
        if not isinstance(raw_cells, (list, tuple)):
            raise ContinuousTargetSamplerError(
                "target profile cells must be a finite sequence"
            )
        result = cls(
            frame_id=payload["frame_id"],
            frame_binding_sha256=payload["frame_binding_sha256"],
            runtime_dtype=payload["runtime_dtype"],
            quantization_contract=payload["quantization_contract"],
            components=payload["components"],
            cells=tuple(TargetCell.from_mapping(cell) for cell in raw_cells),
        )
        if result.profile_sha256 != declared_sha:
            raise ContinuousTargetSamplerError("target profile SHA differs")
        return result


@dataclass(frozen=True)
class TargetSelection:
    """One revealed target; generation is also its policy opportunity index."""

    profile_sha256: str
    env_id: int
    target_generation: int
    cell_id: str
    frame_id: str
    frame_binding_sha256: str
    runtime_dtype: str
    quantization_contract: str
    components: Tuple[str, ...]
    target: Tuple[float, ...]
    semantic_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_sha256",
            _sha256(self.profile_sha256, label="profile_sha256"),
        )
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, label="env_id", minimum=0)
        )
        object.__setattr__(
            self,
            "target_generation",
            _plain_int(
                self.target_generation,
                label="target_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self, "cell_id", _nonempty_text(self.cell_id, label="cell_id")
        )
        object.__setattr__(
            self, "frame_id", _nonempty_text(self.frame_id, label="frame_id")
        )
        object.__setattr__(
            self,
            "frame_binding_sha256",
            _sha256(
                self.frame_binding_sha256, label="frame_binding_sha256"
            ),
        )
        object.__setattr__(
            self,
            "runtime_dtype",
            _runtime_dtype(self.runtime_dtype),
        )
        object.__setattr__(
            self,
            "quantization_contract",
            _quantization_contract(self.quantization_contract),
        )
        clean_components = _components(self.components)
        object.__setattr__(self, "components", clean_components)
        clean_target = _runtime_target_tuple(
            self.target,
            expected_width=len(clean_components),
            label="target",
        )
        object.__setattr__(self, "target", clean_target)
        declared = _sha256(self.semantic_sha256, label="semantic_sha256")
        expected = target_semantic_sha256(
            frame_id=self.frame_id,
            frame_binding_sha256=self.frame_binding_sha256,
            runtime_dtype=self.runtime_dtype,
            quantization_contract=self.quantization_contract,
            components=self.components,
            target=self.target,
        )
        if declared != expected:
            raise ContinuousTargetSamplerError(
                "selection semantic_sha256 differs from numerical target"
            )
        object.__setattr__(self, "semantic_sha256", declared)

    @property
    def policy_opportunity_generation(self) -> int:
        """Only a revealed feasible target creates a policy opportunity."""

        return self.target_generation

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": SELECTION_KIND,
            "profile_sha256": self.profile_sha256,
            "env_id": self.env_id,
            "target_generation": self.target_generation,
            "cell_id": self.cell_id,
            "frame_id": self.frame_id,
            "frame_binding_sha256": self.frame_binding_sha256,
            "runtime_dtype": self.runtime_dtype,
            "quantization_contract": self.quantization_contract,
            "components": list(self.components),
            "target": list(self.target),
            "semantic_sha256": self.semantic_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self._payload())

    def to_mapping(self) -> dict[str, object]:
        payload = self._payload()
        payload["canonical_sha256"] = self.canonical_sha256
        return payload


@dataclass(frozen=True)
class _EnvState:
    rng_state: int
    draw_count: int
    target_generation: int
    previous_cell_id: str
    previous_semantic_sha256: str


class ContinuousTargetSampler:
    """Per-environment deterministic sampler with exact checkpoint replay."""

    def __init__(self, profile: ContinuousTargetProfile, *, seed: int) -> None:
        if not isinstance(profile, ContinuousTargetProfile):
            raise ContinuousTargetSamplerError(
                "profile must be a ContinuousTargetProfile"
            )
        self.profile = profile
        self.seed = _plain_int(
            seed, label="seed", minimum=0, maximum=_U64_MASK
        )
        self._states: dict[int, _EnvState] = {}
        self._sample_in_progress = False

    def sample_next(
        self,
        env_id: int,
        *,
        feasibility_callback: Optional[Callable[[TargetCell], bool]] = None,
        feasible_mask: Optional[Sequence[bool]] = None,
    ) -> TargetSelection:
        """Reveal one fresh target after pre-materializing feasibility.

        ``feasibility_callback`` and ``feasible_mask`` are two equivalent
        construction-time producers and are mutually exclusive.  Every cell
        is classified before RNG/state/generation changes.  The draw is made
        directly from feasible cells whose semantic target differs from the
        last reveal, so this method contains no rejection loop.
        """

        if self._sample_in_progress:
            raise TargetFeasibilityError(
                "sample_next is not reentrant during target construction"
            )
        self._sample_in_progress = True
        try:
            clean_env_id = _plain_int(env_id, label="env_id", minimum=0)
            mask = self._materialize_feasibility(
                feasibility_callback=feasibility_callback,
                feasible_mask=feasible_mask,
            )
            prior = self._states.get(clean_env_id)
            previous_semantic = (
                None if prior is None else prior.previous_semantic_sha256
            )

            candidates: list[tuple[TargetCell, str]] = []
            for cell, feasible in zip(self.profile.cells, mask):
                if not feasible:
                    continue
                semantic_sha = self.profile.semantic_sha256(cell)
                if semantic_sha == previous_semantic:
                    continue
                candidates.append((cell, semantic_sha))

            if not candidates:
                raise NoFeasibleDifferentTargetError(
                    "no construction-feasible target differs from the "
                    "previous numerical target"
                )

            rng_state = (
                self._initial_stream_state(clean_env_id)
                if prior is None
                else prior.rng_state
            )
            next_rng_state, draw = _splitmix64(rng_state)
            # One multiply-high maps one U64 directly into finite support;
            # selection never loops or consumes a hidden retry draw.
            candidate_index = (draw * len(candidates)) >> 64
            cell, semantic_sha = candidates[candidate_index]
            generation = 1 if prior is None else prior.target_generation + 1
            draw_count = 1 if prior is None else prior.draw_count + 1

            selection = TargetSelection(
                profile_sha256=self.profile.profile_sha256,
                env_id=clean_env_id,
                target_generation=generation,
                cell_id=cell.cell_id,
                frame_id=self.profile.frame_id,
                frame_binding_sha256=self.profile.frame_binding_sha256,
                runtime_dtype=self.profile.runtime_dtype,
                quantization_contract=self.profile.quantization_contract,
                components=self.profile.components,
                target=cell.target,
                semantic_sha256=semantic_sha,
            )
            self._states[clean_env_id] = _EnvState(
                rng_state=next_rng_state,
                draw_count=draw_count,
                target_generation=generation,
                previous_cell_id=cell.cell_id,
                previous_semantic_sha256=semantic_sha,
            )
            return selection
        finally:
            self._sample_in_progress = False

    def _materialize_feasibility(
        self,
        *,
        feasibility_callback: Optional[Callable[[TargetCell], bool]],
        feasible_mask: Optional[Sequence[bool]],
    ) -> Tuple[bool, ...]:
        if feasibility_callback is not None and feasible_mask is not None:
            raise TargetFeasibilityError(
                "provide feasibility_callback or feasible_mask, not both"
            )
        if feasibility_callback is not None:
            if not callable(feasibility_callback):
                raise TargetFeasibilityError(
                    "feasibility_callback must be callable"
                )
            values: list[bool] = []
            for cell in self.profile.cells:
                try:
                    result = feasibility_callback(cell)
                except Exception as exc:
                    raise TargetFeasibilityError(
                        "feasibility_callback failed before target reveal"
                    ) from exc
                if type(result) is not bool:
                    raise TargetFeasibilityError(
                        "feasibility_callback must return an exact bool"
                    )
                values.append(result)
            return tuple(values)
        if feasible_mask is None:
            return (True,) * len(self.profile.cells)
        if not isinstance(feasible_mask, (tuple, list)):
            raise TargetFeasibilityError(
                "feasible_mask must be a finite bool sequence"
            )
        if len(feasible_mask) != len(self.profile.cells):
            raise TargetFeasibilityError(
                "feasible_mask width must equal finite target support"
            )
        if any(type(item) is not bool for item in feasible_mask):
            raise TargetFeasibilityError(
                "feasible_mask entries must be exact bools"
            )
        return tuple(feasible_mask)

    def _initial_stream_state(self, env_id: int) -> int:
        digest = canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": STREAM_KIND,
                "profile_sha256": self.profile.profile_sha256,
                "seed": self.seed,
                "env_id": env_id,
            }
        )
        return int(digest[:16], 16)

    def checkpoint_env_state(self, env_id: int) -> Optional[dict[str, object]]:
        """Return one exact checkpoint row without serializing every world.

        Batched reveal construction advances many independent environment
        streams on one private sampler clone.  Rebuilding the complete
        checkpoint after every row makes that otherwise linear operation
        quadratic in the number of worlds.  This method exposes the same
        immutable row that :meth:`checkpoint` emits, while retaining the
        complete checkpoint as the sole portable envelope and root.
        """

        clean_env_id = _plain_int(env_id, label="env_id", minimum=0)
        state = self._states.get(clean_env_id)
        if state is None:
            return None
        return {
            "env_id": clean_env_id,
            "rng_state": state.rng_state,
            "draw_count": state.draw_count,
            "target_generation": state.target_generation,
            "previous_cell_id": state.previous_cell_id,
            "previous_semantic_sha256": state.previous_semantic_sha256,
        }

    def checkpoint(self) -> dict[str, object]:
        """Return a content-addressed, non-mutating exact-resume checkpoint."""

        environments = [
            self.checkpoint_env_state(env_id)
            for env_id in sorted(self._states)
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "profile_sha256": self.profile.profile_sha256,
            "runtime_dtype": self.profile.runtime_dtype,
            "quantization_contract": self.profile.quantization_contract,
            "seed": self.seed,
            "environments": environments,
        }
        result = dict(payload)
        result["checkpoint_sha256"] = canonical_sha256(payload)
        return result

    @classmethod
    def from_checkpoint(
        cls,
        profile: ContinuousTargetProfile,
        value: object,
    ) -> "ContinuousTargetSampler":
        _reject_legacy_v1_envelope(
            value,
            label="sampler checkpoint",
            legacy_kind=LEGACY_CHECKPOINT_KIND,
        )
        expected = (
            "schema_version",
            "kind",
            "profile_sha256",
            "runtime_dtype",
            "quantization_contract",
            "seed",
            "environments",
            "checkpoint_sha256",
        )
        payload = _exact_mapping(value, expected, label="sampler checkpoint")
        declared_checkpoint_sha = _sha256(
            payload["checkpoint_sha256"], label="checkpoint_sha256"
        )
        unsigned = {
            key: payload[key] for key in expected if key != "checkpoint_sha256"
        }
        if canonical_sha256(unsigned) != declared_checkpoint_sha:
            raise ContinuousTargetSamplerError("sampler checkpoint SHA differs")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != SCHEMA_VERSION
        ):
            raise ContinuousTargetSamplerError(
                "sampler checkpoint schema_version differs"
            )
        if (
            type(payload["kind"]) is not str
            or payload["kind"] != CHECKPOINT_KIND
        ):
            raise ContinuousTargetSamplerError("sampler checkpoint kind differs")
        declared_profile_sha = _sha256(
            payload["profile_sha256"], label="profile_sha256"
        )
        if declared_profile_sha != profile.profile_sha256:
            raise ContinuousTargetSamplerError(
                "sampler checkpoint profile SHA differs"
            )
        if _runtime_dtype(payload["runtime_dtype"]) != profile.runtime_dtype:
            raise ContinuousTargetSamplerError(
                "sampler checkpoint runtime dtype differs"
            )
        if (
            _quantization_contract(payload["quantization_contract"])
            != profile.quantization_contract
        ):
            raise ContinuousTargetSamplerError(
                "sampler checkpoint quantization contract differs"
            )
        result = cls(profile, seed=payload["seed"])
        raw_environments = payload["environments"]
        if not isinstance(raw_environments, (tuple, list)):
            raise ContinuousTargetSamplerError(
                "sampler checkpoint environments must be a finite sequence"
            )
        previous_env_id = -1
        by_cell_id = {cell.cell_id: cell for cell in profile.cells}
        for row_value in raw_environments:
            row = _exact_mapping(
                row_value,
                (
                    "env_id",
                    "rng_state",
                    "draw_count",
                    "target_generation",
                    "previous_cell_id",
                    "previous_semantic_sha256",
                ),
                label="sampler checkpoint environment",
            )
            env_id = _plain_int(row["env_id"], label="env_id", minimum=0)
            if env_id <= previous_env_id:
                raise ContinuousTargetSamplerError(
                    "sampler checkpoint environments must be strictly sorted"
                )
            previous_env_id = env_id
            rng_state = _plain_int(
                row["rng_state"],
                label="rng_state",
                minimum=0,
                maximum=_U64_MASK,
            )
            draw_count = _plain_int(
                row["draw_count"], label="draw_count", minimum=1
            )
            generation = _plain_int(
                row["target_generation"],
                label="target_generation",
                minimum=1,
            )
            if draw_count != generation:
                raise ContinuousTargetSamplerError(
                    "draw_count must equal revealed target_generation"
                )
            expected_rng_state = (
                result._initial_stream_state(env_id)
                + draw_count * _SPLITMIX_GAMMA
            ) & _U64_MASK
            if rng_state != expected_rng_state:
                raise ContinuousTargetSamplerError(
                    "rng_state differs from seed/env/profile/draw_count provenance"
                )
            previous_cell_id = _nonempty_text(
                row["previous_cell_id"], label="previous_cell_id"
            )
            if previous_cell_id not in by_cell_id:
                raise ContinuousTargetSamplerError(
                    "previous_cell_id is outside the bound target profile"
                )
            previous_semantic = _sha256(
                row["previous_semantic_sha256"],
                label="previous_semantic_sha256",
            )
            expected_semantic = profile.semantic_sha256(
                by_cell_id[previous_cell_id]
            )
            if previous_semantic != expected_semantic:
                raise ContinuousTargetSamplerError(
                    "previous target semantic SHA differs from its cell"
                )
            result._states[env_id] = _EnvState(
                rng_state=rng_state,
                draw_count=draw_count,
                target_generation=generation,
                previous_cell_id=previous_cell_id,
                previous_semantic_sha256=previous_semantic,
            )
        return result


def _splitmix64(state: int) -> tuple[int, int]:
    next_state = (state + _SPLITMIX_GAMMA) & _U64_MASK
    value = next_state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    value ^= value >> 31
    return next_state, value & _U64_MASK


def _reject_legacy_v1_envelope(
    value: object,
    *,
    label: str,
    legacy_kind: str,
) -> None:
    if not isinstance(value, Mapping):
        return
    if (
        value.get("schema_version") == LEGACY_SCHEMA_VERSION
        or value.get("kind") == legacy_kind
    ):
        raise ContinuousTargetSamplerError(
            "%s legacy v1 is tombstoned: it lacks runtime float32 "
            "quantization identity" % label
        )


def _exact_mapping(
    value: object, expected_keys: Sequence[str], *, label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContinuousTargetSamplerError("%s must be a mapping" % label)
    expected = frozenset(expected_keys)
    actual = frozenset(value)
    if actual != expected:
        raise ContinuousTargetSamplerError(
            "%s keys differ: missing=%r unknown=%r"
            % (label, sorted(expected - actual), sorted(actual - expected))
        )
    return dict(value)


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise ContinuousTargetSamplerError("%s must be an exact int" % label)
    if value < minimum:
        raise ContinuousTargetSamplerError(
            "%s must be >= %d" % (label, minimum)
        )
    if maximum is not None and value > maximum:
        raise ContinuousTargetSamplerError(
            "%s must be <= %d" % (label, maximum)
        )
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContinuousTargetSamplerError(
            "%s must be a non-empty string" % label
        )
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuousTargetSamplerError(
            "%s must be a lowercase SHA-256" % label
        )
    return value


def _runtime_dtype(value: object) -> str:
    if value != RUNTIME_DTYPE or type(value) is not str:
        raise ContinuousTargetSamplerError(
            "runtime_dtype must equal %r" % RUNTIME_DTYPE
        )
    return value


def _quantization_contract(value: object) -> str:
    if value != QUANTIZATION_CONTRACT or type(value) is not str:
        raise ContinuousTargetSamplerError(
            "quantization_contract must equal %r" % QUANTIZATION_CONTRACT
        )
    return value


def _components(value: object) -> Tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ContinuousTargetSamplerError(
            "components must be a non-empty finite sequence"
        )
    result = tuple(
        _nonempty_text(component, label="component") for component in value
    )
    if len(set(result)) != len(result):
        raise ContinuousTargetSamplerError("components must be unique")
    return result


def _target_tuple(
    value: object,
    *,
    expected_width: Optional[int],
    label: str,
) -> Tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ContinuousTargetSamplerError(
            "%s must be a non-empty finite sequence" % label
        )
    if expected_width is not None and len(value) != expected_width:
        raise ContinuousTargetSamplerError(
            "%s width differs from components" % label
        )
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ContinuousTargetSamplerError(
                "%s[%d] must be a finite number" % (label, index)
            )
        clean = float(item)
        if not math.isfinite(clean):
            raise ContinuousTargetSamplerError(
                "%s[%d] must be a finite number" % (label, index)
            )
        result.append(0.0 if clean == 0.0 else clean)
    return tuple(result)


def _runtime_target_tuple(
    value: object,
    *,
    expected_width: Optional[int],
    label: str,
) -> Tuple[float, ...]:
    source = _target_tuple(
        value,
        expected_width=expected_width,
        label=label,
    )
    result = []
    for index, item in enumerate(source):
        try:
            encoded = struct.pack(">f", item)
        except (OverflowError, struct.error) as exc:
            raise ContinuousTargetSamplerError(
                "%s[%d] is not finite in runtime float32" % (label, index)
            ) from exc
        quantized = struct.unpack(">f", encoded)[0]
        if not math.isfinite(quantized):
            raise ContinuousTargetSamplerError(
                "%s[%d] is not finite in runtime float32" % (label, index)
            )
        # Both semantic equality and bytes use positive zero.  A negative-zero
        # spelling is never a fresh runtime target.
        result.append(0.0 if quantized == 0.0 else quantized)
    return tuple(result)


def _float32_be_hex(value: float) -> str:
    clean = 0.0 if value == 0.0 else value
    try:
        return struct.pack(">f", clean).hex()
    except (OverflowError, struct.error) as exc:
        raise ContinuousTargetSamplerError(
            "target is not encodable as runtime float32"
        ) from exc
