"""Precomputed curriculum-banded ActionBall question source.

Unlike :mod:`action_ball_fixed_question_tape`, this draft source models a cache
whose blocks can follow curriculum levels.  Every cache block is addressed by
the exact live domain-level SHA plus the action/profile/runtime solver identity.
Resets only select and materialize an already solved row; this module has no
online solver callback.

The low-level artifact/solver remains construction-only.  Trainer/runtime
admission requires ``diagnostic_unauthorized=true`` until the cached-question
sample tape is reconciled with the separately advancing base-birth sampler and
finite artifacts prove exact future birth/base coverage.  In particular, the
zero RNG counter below is scoped to this question solver; it is not a claim that
episode births consume no RNG.

The first producer may publish only the center block.  A later curriculum
claim for which no exact block exists fails closed instead of falling back to
the online solver or reusing the center questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

try:
    from . import action_ball_fixed_question_tape as _fixed
except ImportError:
    _fixed_path = Path(__file__).with_name("action_ball_fixed_question_tape.py")
    _fixed_spec = importlib.util.spec_from_file_location(
        "_banded_question_fixed_tape", _fixed_path
    )
    if _fixed_spec is None or _fixed_spec.loader is None:
        raise
    _fixed = importlib.util.module_from_spec(_fixed_spec)
    sys.modules[_fixed_spec.name] = _fixed
    _fixed_spec.loader.exec_module(_fixed)


_runtime = _fixed._runtime
SCHEMA_VERSION = 2
STATE_SCHEMA_VERSION = 1
KIND = "action_ball_banded_question_bank"
SELECTION = (
    "sha256_exact_block_birth_swing_split_seed_required_stratum_"
    "sealed_base_identity_mod_bucket"
)
BASE_IDENTITY_KIND = "action_ball_base_identity.exact_canonical_json.v1"
CURRENT_LM_RECIPE = "current_lm"
CURRENT_LM_VALIDITY_MASK = (True, True, True)
UNSUPPORTED_C_RECIPE = "outcome_dense_only"
UNSUPPORTED_C_VALIDITY_MASK = (False, False, False)

_BLOCK_KEY_FIELDS = (
    "levels_sha256",
    "action_uid",
    "action_slot",
    "profile_sha256",
    "motion_sha256",
    "manifest_sha256",
    "sampler_sha256",
    "domain_authority_sha256",
    "arm_catalog_sha256",
    "physics_sha256",
    "solver_sha256",
    "mobility_mode",
)


def _sha256_json(value: object) -> str:
    return _fixed._sha256_json(value)


def _sha256(value: object, *, name: str) -> str:
    return _fixed._sha256(value, name=name)


def _plain_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a plain integer >= {minimum}")
    return value


def _exact_mapping(value: object, keys: Sequence[str], *, name: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly {tuple(keys)!r}")
    return value


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_plain(item) for item in value]
    return value


def _block_key_for_receipt(receipt: object) -> dict:
    return {
        "levels_sha256": receipt.levels_sha256,
        "action_uid": receipt.action_uid,
        "action_slot": receipt.action_slot,
        "profile_sha256": receipt.profile_sha256,
        "motion_sha256": receipt.motion_sha256,
        "manifest_sha256": receipt.manifest_sha256,
        "sampler_sha256": receipt.sampler_sha256,
        "domain_authority_sha256": receipt.domain_authority_sha256,
        "arm_catalog_sha256": receipt.arm_catalog_sha256,
        "physics_sha256": receipt.physics_sha256,
        "solver_sha256": receipt.solver_sha256,
        "mobility_mode": receipt.mobility_mode,
    }


def _block_key_for_birth(birth: object) -> dict:
    return _block_key_for_receipt(birth)


def _validate_block_key(value: object) -> dict:
    row = _exact_mapping(value, _BLOCK_KEY_FIELDS, name="band block key")
    result = {
        "levels_sha256": _sha256(row["levels_sha256"], name="levels_sha256"),
        "action_uid": _plain_int(row["action_uid"], name="action_uid", minimum=1),
        "action_slot": _plain_int(row["action_slot"], name="action_slot"),
        "profile_sha256": _sha256(row["profile_sha256"], name="profile_sha256"),
        "motion_sha256": _sha256(row["motion_sha256"], name="motion_sha256"),
        "manifest_sha256": _sha256(row["manifest_sha256"], name="manifest_sha256"),
        "sampler_sha256": _sha256(row["sampler_sha256"], name="sampler_sha256"),
        "domain_authority_sha256": _sha256(
            row["domain_authority_sha256"], name="domain_authority_sha256"
        ),
        "arm_catalog_sha256": _sha256(
            row["arm_catalog_sha256"], name="arm_catalog_sha256"
        ),
        "physics_sha256": _sha256(row["physics_sha256"], name="physics_sha256"),
        "solver_sha256": _sha256(row["solver_sha256"], name="solver_sha256"),
        "mobility_mode": str(row["mobility_mode"]),
    }
    if result["mobility_mode"] not in ("no_move", "move"):
        raise ValueError("band block mobility_mode is invalid")
    return result


def _base_identity_sha256(
    *,
    base_yaw_rad: float,
    base_quat_wxyz: Sequence[float],
    base_spawn_w_m: Sequence[float],
) -> str:
    """Seal the exact serialized base tuple behind one canonical identity.

    The cache is produced by replaying the same sampler tape that will create
    live births.  It is therefore intentionally exact, not tolerance based.
    Callers compare this sealed identity rather than performing three ad-hoc
    Python float/tuple equalities at the selection seam.
    """

    quat = tuple(base_quat_wxyz)
    spawn = tuple(base_spawn_w_m)
    if len(quat) != 4 or len(spawn) != 3:
        raise ValueError("banded question base identity has invalid shape")
    values = (base_yaw_rad, *quat, *spawn)
    if any(type(value) not in (int, float) for value in values):
        raise ValueError("banded question base identity must be numeric")
    return _sha256_json(
        {
            "kind": BASE_IDENTITY_KIND,
            "base_yaw_rad": float(base_yaw_rad),
            "base_quat_wxyz": [float(value) for value in quat],
            "base_spawn_w_m": [float(value) for value in spawn],
        }
    )


def _base_identity_for_receipt(receipt: object) -> str:
    return _base_identity_sha256(
        base_yaw_rad=receipt.base_yaw_rad,
        base_quat_wxyz=receipt.base_quat_wxyz,
        base_spawn_w_m=receipt.base_spawn_w_m,
    )


def _levels_from_pending_release(pending_release: object) -> object:
    """Derive the exact next published 32-arm vector without mutating curriculum."""

    target = getattr(pending_release, "target", None)
    phase = getattr(target, "phase", None)
    frontier_indices = getattr(target, "arm_frontier_indices", None)
    probe_indices = getattr(target, "arm_probe_indices", None)
    selected = getattr(target, "selected_arm_key", None)
    joint_probe_index = getattr(target, "joint_probe_index", None)
    joint_rho_index = getattr(target, "joint_rho_index", None)
    if (
        phase not in ("center", "marginal", "joint", "steady")
        or not isinstance(frontier_indices, (tuple, list))
        or len(frontier_indices) != len(_runtime.ARM_KEYS)
        or not isinstance(probe_indices, (tuple, list))
        or len(probe_indices) != len(_runtime.ARM_KEYS)
    ):
        raise ValueError("pending release target cannot define an exact 32-arm vector")
    levels_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    if any(type(value) is not int or value not in range(5) for value in frontier_indices):
        raise ValueError("pending release frontier indices are invalid")
    values = [0.0] * len(_runtime.ARM_KEYS)
    if phase == "marginal":
        if selected not in _runtime.ARM_KEYS:
            raise ValueError("pending marginal release lacks one selected arm")
        index = _runtime.ARM_KEYS.index(selected)
        probe = probe_indices[index]
        if type(probe) is not int or probe not in range(5):
            raise ValueError("pending marginal release probe index is invalid")
        values[index] = levels_grid[probe]
    elif phase in ("joint", "steady"):
        rho_index = joint_probe_index if phase == "joint" else joint_rho_index
        if type(rho_index) is not int or rho_index not in range(5):
            raise ValueError("pending joint release rho index is invalid")
        rho = levels_grid[rho_index]
        values = [levels_grid[index] * rho for index in frontier_indices]
    return _runtime.ActionDomainLevels(
        **dict(zip(_runtime.ARM_KEYS, values))
    )


@dataclass(frozen=True)
class BandedQuestionBlock:
    """One exact curriculum/action/runtime cache block."""

    key: Mapping[str, object]
    rows: Tuple[object, ...]
    _key_sha256: str = field(init=False, repr=False, compare=False)
    _content_sha256: str = field(init=False, repr=False, compare=False)
    _row_indices: Mapping[Tuple[str, str], Tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        key = _validate_block_key(self.key)
        if not isinstance(self.rows, (tuple, list)) or not self.rows:
            raise ValueError("banded question block rows must be non-empty")
        rows = tuple(self.rows)
        if any(
            not isinstance(row, _runtime.ActionBallTaskReceipt) for row in rows
        ):
            raise TypeError("banded question rows must be ActionBallTaskReceipt objects")
        for row in rows:
            if _block_key_for_receipt(row) != key:
                raise ValueError("banded question row differs from its exact block key")
        row_shas = [row.canonical_sha256 for row in rows]
        if len(row_shas) != len(set(row_shas)):
            raise ValueError("banded question block repeats a solved row")
        mixture_payloads = [
            None
            if row.sampling_mixture is None
            else row.sampling_mixture.as_dict()
            for row in rows
        ]
        if any(payload != mixture_payloads[0] for payload in mixture_payloads):
            raise ValueError(
                "banded question block rows must share one sampling mixture"
            )
        if rows[0].sampling_mixture is not None:
            required = set(rows[0].sampling_mixture.schedule)
            available = {row.sampling_stratum for row in rows}
            if not required.issubset(available):
                raise ValueError(
                    "banded question block omits a required sampling stratum"
                )
        object.__setattr__(self, "key", MappingProxyType(key))
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "_key_sha256", _sha256_json(key))
        content = {
            "key": key,
            "row_count": len(rows),
            "rows": [row.to_dict() for row in rows],
        }
        object.__setattr__(self, "_content_sha256", _sha256_json(content))
        indices = {}
        for index, row in enumerate(rows):
            identity = (row.sampling_stratum, _base_identity_for_receipt(row))
            indices.setdefault(identity, []).append(index)
        object.__setattr__(
            self,
            "_row_indices",
            MappingProxyType(
                {identity: tuple(values) for identity, values in indices.items()}
            ),
        )

    @property
    def key_sha256(self) -> str:
        return self._key_sha256

    @property
    def content_payload(self) -> dict:
        return {
            "key": dict(self.key),
            "row_count": len(self.rows),
            "rows": [row.to_dict() for row in self.rows],
        }

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    def to_dict(self) -> dict:
        return {
            **self.content_payload,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BandedQuestionBlock":
        row = _exact_mapping(
            value,
            ("key", "row_count", "rows", "content_sha256"),
            name="banded question block",
        )
        if not isinstance(row["rows"], (tuple, list)):
            raise ValueError("banded question block rows must be a list")
        rows = tuple(
            _runtime.ActionBallTaskReceipt.from_dict(item) for item in row["rows"]
        )
        block = cls(key=row["key"], rows=rows)
        if row["row_count"] != len(rows):
            raise ValueError("banded question block row_count mismatch")
        if _sha256(row["content_sha256"], name="block content_sha256") != (
            block.content_sha256
        ):
            raise ValueError("banded question block content SHA mismatch")
        return block

    @classmethod
    def from_receipts(cls, rows: Sequence[object]) -> "BandedQuestionBlock":
        rows = tuple(rows)
        if not rows:
            raise ValueError("cannot build an empty banded question block")
        return cls(key=_block_key_for_receipt(rows[0]), rows=rows)

    def row_index_for(
        self,
        *,
        birth_sha256: str,
        swing_generation: int,
        split_seed: int,
        sample_index: int | None = None,
        base_yaw_rad: float,
        base_quat_wxyz: Sequence[float],
        base_spawn_w_m: Sequence[float],
    ) -> int:
        base_quat_wxyz = tuple(base_quat_wxyz)
        base_spawn_w_m = tuple(base_spawn_w_m)
        if len(base_quat_wxyz) != 4 or len(base_spawn_w_m) != 3:
            raise ValueError("banded question live base identity has invalid shape")
        mixture = self.rows[0].sampling_mixture
        if mixture is None:
            required_stratum = "domain"
        else:
            if sample_index is None:
                raise ValueError(
                    "mixture-backed band selection requires sample_index"
                )
            required_stratum = mixture.stratum_for(
                _plain_int(sample_index, name="sample_index")
            )
        base_identity = _base_identity_sha256(
            base_yaw_rad=base_yaw_rad,
            base_quat_wxyz=base_quat_wxyz,
            base_spawn_w_m=base_spawn_w_m,
        )
        eligible = self._row_indices.get((required_stratum, base_identity), ())
        if not eligible:
            raise ValueError(
                "banded question block has no row for the required sampling "
                "stratum and exact live base identity"
            )
        digest = _sha256_json(
            {
                "schema_version": SCHEMA_VERSION,
                "selection": SELECTION,
                "block_content_sha256": self.content_sha256,
                "birth_sha256": _sha256(birth_sha256, name="birth_sha256"),
                "swing_generation": _plain_int(
                    swing_generation, name="swing_generation"
                ),
                "split_seed": _plain_int(split_seed, name="split_seed"),
                "required_stratum": required_stratum,
                "base_identity_sha256": base_identity,
            }
        )
        return eligible[int(digest, 16) % len(eligible)]


def base_question_root_for_blocks(blocks: Sequence[BandedQuestionBlock]) -> str:
    blocks = tuple(blocks)
    if not blocks or any(not isinstance(block, BandedQuestionBlock) for block in blocks):
        raise ValueError("base-question root requires banded question blocks")
    return _sha256_json(
        {
            "kind": "action_ball_banded_base_questions.v1",
            "blocks": [
                {
                    "key_sha256": block.key_sha256,
                    "row_canonical_sha256": sorted(
                        row.canonical_sha256 for row in block.rows
                    ),
                }
                for block in sorted(blocks, key=lambda item: item.key_sha256)
            ],
        }
    )


def question_lineage_for_blocks(blocks: Sequence[BandedQuestionBlock]) -> dict:
    root = base_question_root_for_blocks(blocks)
    return {
        "schema_version": 1,
        "base_question_root_sha256": root,
        "variants": [
            {
                "consumer": "A",
                "observation_variant": "A111",
                "target_recipe": CURRENT_LM_RECIPE,
                "target_validity_mask": list(CURRENT_LM_VALIDITY_MASK),
                "supported_by_bank": True,
                "required_source": "banded_question_bank",
                "base_question_root_sha256": root,
            },
            {
                "consumer": "C",
                "observation_variant": "C000",
                "target_recipe": UNSUPPORTED_C_RECIPE,
                "target_validity_mask": list(UNSUPPORTED_C_VALIDITY_MASK),
                "supported_by_bank": False,
                "required_source": "immutable_tape",
                "base_question_root_sha256": root,
            },
        ],
    }


@dataclass(frozen=True)
class BandedQuestionBank:
    split_seed: int
    blocks: Tuple[BandedQuestionBlock, ...]
    coverage: Mapping[str, object]
    question_lineage: Mapping[str, object]
    producer_lineage: Mapping[str, object]
    _block_by_key_sha256: Mapping[str, BandedQuestionBlock] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        split_seed = _plain_int(self.split_seed, name="split_seed")
        if not isinstance(self.blocks, (tuple, list)) or not self.blocks:
            raise ValueError("banded question bank must contain at least one block")
        blocks = tuple(self.blocks)
        if any(not isinstance(block, BandedQuestionBlock) for block in blocks):
            raise TypeError("banded question bank blocks have invalid type")
        keys = [block.key_sha256 for block in blocks]
        if len(keys) != len(set(keys)):
            raise ValueError("banded question bank repeats an exact block key")
        projected_blocks = [
            {
                "action_uid": int(block.key["action_uid"]),
                "levels_sha256": str(block.key["levels_sha256"]),
            }
            for block in blocks
        ]
        projected_pairs = {
            (item["action_uid"], item["levels_sha256"])
            for item in projected_blocks
        }
        if len(projected_pairs) != len(projected_blocks):
            raise ValueError(
                "banded question bank has multiple exact blocks for one "
                "action/domain-level vector"
            )
        identity_fields = tuple(
            field_name
            for field_name in _BLOCK_KEY_FIELDS
            if field_name != "levels_sha256"
        )
        for action_uid in {item["action_uid"] for item in projected_blocks}:
            families = {
                tuple(block.key[field_name] for field_name in identity_fields)
                for block in blocks
                if block.key["action_uid"] == action_uid
            }
            if len(families) != 1:
                raise ValueError(
                    "banded question levels for one action must share one exact "
                    "profile/motion/manifest/sampler/domain/physics/solver identity"
                )
        coverage = _exact_mapping(
            self.coverage,
            (
                "schema_version",
                "kind",
                "arm_catalog_sha256",
                "arm_keys",
                "expected_action_uids",
                "reachable_arm_keys_by_action",
                "source_file_sha256",
                "source_canonical_sha256",
                "reachable_blocks",
            ),
            name="banded question bank coverage",
        )
        if (
            coverage["schema_version"] != 1
            or coverage["kind"]
            != "action_ball_reachable_domain_level_blocks"
            or _sha256(
                coverage["arm_catalog_sha256"], name="coverage arm_catalog_sha256"
            )
            != _runtime.ARM_CATALOG_SHA256
            or tuple(coverage["arm_keys"]) != tuple(_runtime.ARM_KEYS)
        ):
            raise ValueError("banded question coverage arm catalog/schema mismatch")
        source_file_sha = _sha256(
            coverage["source_file_sha256"], name="coverage source_file_sha256"
        )
        source_canonical_sha = _sha256(
            coverage["source_canonical_sha256"],
            name="coverage source_canonical_sha256",
        )
        expected_action_uids = tuple(coverage["expected_action_uids"])
        if (
            not expected_action_uids
            or any(type(uid) is not int or uid < 1 for uid in expected_action_uids)
            or len(expected_action_uids) != len(set(expected_action_uids))
        ):
            raise ValueError("coverage expected_action_uids must be unique positive ints")
        raw_arm_rows = coverage["reachable_arm_keys_by_action"]
        if not isinstance(raw_arm_rows, (tuple, list)):
            raise ValueError("coverage reachable_arm_keys_by_action must be a list")
        arm_rows = []
        for raw in raw_arm_rows:
            item = _exact_mapping(
                raw,
                ("action_uid", "reachable_arm_keys"),
                name="coverage reachable arm row",
            )
            uid = _plain_int(item["action_uid"], name="coverage arm action_uid", minimum=1)
            arm_keys = tuple(item["reachable_arm_keys"])
            if (
                not arm_keys
                or len(arm_keys) != len(set(arm_keys))
                or any(arm not in _runtime.ARM_KEYS for arm in arm_keys)
            ):
                raise ValueError("coverage reachable arm keys are invalid")
            arm_rows.append({"action_uid": uid, "reachable_arm_keys": list(arm_keys)})
        if (
            {row["action_uid"] for row in arm_rows} != set(expected_action_uids)
            or len(arm_rows) != len(expected_action_uids)
            or {item["action_uid"] for item in projected_blocks}
            != set(expected_action_uids)
        ):
            raise ValueError("coverage action set does not match reachable blocks")
        reachable_arms = {
            row["action_uid"]: set(row["reachable_arm_keys"]) for row in arm_rows
        }
        raw_reachable = coverage["reachable_blocks"]
        if not isinstance(raw_reachable, (tuple, list)) or not raw_reachable:
            raise ValueError("banded question coverage must declare reachable blocks")
        reachable = []
        for index, raw in enumerate(raw_reachable):
            item = _exact_mapping(
                raw,
                ("action_uid", "levels_sha256", "domain_levels"),
                name=f"coverage reachable_blocks[{index}]",
            )
            action_uid = _plain_int(
                item["action_uid"], name="coverage action_uid", minimum=1
            )
            levels = _runtime.ActionDomainLevels.from_dict(item["domain_levels"])
            levels_sha = _sha256(
                item["levels_sha256"], name="coverage levels_sha256"
            )
            if levels_sha != levels.canonical_sha256:
                raise ValueError("coverage level vector SHA differs from its 32-arm payload")
            if any(
                getattr(levels, arm) != 0.0
                and arm not in reachable_arms.get(action_uid, set())
                for arm in _runtime.ARM_KEYS
            ):
                raise ValueError(
                    "coverage level vector activates an arm declared unreachable"
                )
            reachable.append(
                {
                    "action_uid": action_uid,
                    "levels_sha256": levels_sha,
                    "domain_levels": levels.to_dict(),
                }
            )
        reachable_pairs = {
            (item["action_uid"], item["levels_sha256"]) for item in reachable
        }
        if len(reachable_pairs) != len(reachable) or reachable_pairs != projected_pairs:
            raise ValueError(
                "banded question blocks do not exactly cover the declared reachable "
                "action/domain-level vectors"
            )
        normalized_coverage = {
            "schema_version": 1,
            "kind": "action_ball_reachable_domain_level_blocks",
            "arm_catalog_sha256": _runtime.ARM_CATALOG_SHA256,
            "arm_keys": list(_runtime.ARM_KEYS),
            "expected_action_uids": sorted(expected_action_uids),
            "reachable_arm_keys_by_action": sorted(
                arm_rows, key=lambda item: item["action_uid"]
            ),
            "source_file_sha256": source_file_sha,
            "source_canonical_sha256": source_canonical_sha,
            "reachable_blocks": sorted(
                reachable,
                key=lambda item: (item["action_uid"], item["levels_sha256"]),
            ),
        }
        question_root = base_question_root_for_blocks(blocks)
        question_lineage = _exact_mapping(
            self.question_lineage,
            ("schema_version", "base_question_root_sha256", "variants"),
            name="banded question bank question_lineage",
        )
        if (
            question_lineage["schema_version"] != 1
            or _sha256(
                question_lineage["base_question_root_sha256"],
                name="base_question_root_sha256",
            )
            != question_root
        ):
            raise ValueError("banded question base-question root mismatch")
        expected_variants = (
            {
                "consumer": "A",
                "observation_variant": "A111",
                "target_recipe": CURRENT_LM_RECIPE,
                "target_validity_mask": list(CURRENT_LM_VALIDITY_MASK),
                "supported_by_bank": True,
                "required_source": "banded_question_bank",
                "base_question_root_sha256": question_root,
            },
            {
                "consumer": "C",
                "observation_variant": "C000",
                "target_recipe": UNSUPPORTED_C_RECIPE,
                "target_validity_mask": list(UNSUPPORTED_C_VALIDITY_MASK),
                "supported_by_bank": False,
                "required_source": "immutable_tape",
                "base_question_root_sha256": question_root,
            },
        )
        if tuple(question_lineage["variants"]) != expected_variants:
            raise ValueError(
                "banded question variants must state A/current_lm support and "
                "C/outcome_dense_only immutable-tape exclusion exactly"
            )
        lineage = self.producer_lineage
        lineage = _exact_mapping(
            lineage,
            (
                "schema_version",
                "kind",
                "row_order",
                "producer_source_sha256",
                "bank_module_source_sha256",
                "inputs",
            ),
            name="banded question bank producer_lineage",
        )
        if (
            lineage["schema_version"] != 1
            or lineage["kind"]
            != "action_ball_banded_question_bank.offline_solved_receipts"
        ):
            raise ValueError("banded question bank producer lineage schema/kind mismatch")
        if lineage["row_order"] != "canonical_receipt_sha256":
            raise ValueError("banded question bank producer row order is invalid")
        _sha256(
            lineage["producer_source_sha256"],
            name="producer_source_sha256",
        )
        _sha256(
            lineage["bank_module_source_sha256"],
            name="bank_module_source_sha256",
        )
        if not isinstance(lineage["inputs"], (tuple, list)):
            raise ValueError("banded question bank producer inputs must be a list")
        inputs = []
        for raw in lineage["inputs"]:
            item = _exact_mapping(
                raw,
                (
                    "source_id",
                    "solver_mode",
                    "block_key_sha256",
                    "file_sha256",
                    "offline_producer_source_sha256",
                    "offline_input_root_sha256",
                    "proposed_count",
                    "admitted_count",
                    "rejections",
                    "receipt_canonical_sha256",
                ),
                name="banded question bank producer input",
            )
            if type(item["source_id"]) is not str or not item["source_id"]:
                raise ValueError("banded question producer source_id must be non-empty")
            if item["solver_mode"] != "current_lm_only":
                raise ValueError("banded question offline solver_mode must be current_lm_only")
            receipt_shas = item["receipt_canonical_sha256"]
            if not isinstance(receipt_shas, (tuple, list)) or not receipt_shas:
                raise ValueError("banded question producer input receipts must be non-empty")
            proposed = _plain_int(item["proposed_count"], name="proposed_count")
            admitted = _plain_int(item["admitted_count"], name="admitted_count")
            if admitted != len(receipt_shas) or admitted > proposed:
                raise ValueError(
                    "offline solve admitted denominator must equal published receipts "
                    "and not exceed proposals"
                )
            raw_rejections = item["rejections"]
            if not isinstance(raw_rejections, (tuple, list)):
                raise ValueError("offline solve rejections must be a list")
            rejections = []
            for rejection_index, rejection in enumerate(raw_rejections):
                rejection_row = _exact_mapping(
                    rejection,
                    ("reason", "count"),
                    name=f"offline rejection[{rejection_index}]",
                )
                if (
                    type(rejection_row["reason"]) is not str
                    or not rejection_row["reason"]
                ):
                    raise ValueError("offline rejection reason must be non-empty")
                count = _plain_int(
                    rejection_row["count"], name="offline rejection count", minimum=1
                )
                rejections.append({"reason": rejection_row["reason"], "count": count})
            if (
                len({item["reason"] for item in rejections}) != len(rejections)
                or sum(item["count"] for item in rejections) != proposed - admitted
            ):
                raise ValueError(
                    "offline rejection reasons must be unique and conserve P=A+R"
                )
            inputs.append(
                {
                    "source_id": item["source_id"],
                    "solver_mode": "current_lm_only",
                    "block_key_sha256": _sha256(
                        item["block_key_sha256"], name="producer block_key_sha256"
                    ),
                    "file_sha256": _sha256(
                        item["file_sha256"], name="producer input file_sha256"
                    ),
                    "offline_producer_source_sha256": _sha256(
                        item["offline_producer_source_sha256"],
                        name="offline_producer_source_sha256",
                    ),
                    "offline_input_root_sha256": _sha256(
                        item["offline_input_root_sha256"],
                        name="offline_input_root_sha256",
                    ),
                    "proposed_count": proposed,
                    "admitted_count": admitted,
                    "rejections": sorted(
                        rejections, key=lambda rejection: rejection["reason"]
                    ),
                    "receipt_canonical_sha256": [
                        _sha256(value, name="producer receipt canonical_sha256")
                        for value in receipt_shas
                    ],
                }
            )
        block_by_key = {block.key_sha256: block for block in blocks}
        if (
            len({item["block_key_sha256"] for item in inputs}) != len(inputs)
            or {item["block_key_sha256"] for item in inputs} != set(block_by_key)
        ):
            raise ValueError(
                "offline solve inputs must map one-to-one onto every exact bank block"
            )
        for item in inputs:
            expected_receipts = {
                row.canonical_sha256
                for row in block_by_key[item["block_key_sha256"]].rows
            }
            if (
                set(item["receipt_canonical_sha256"]) != expected_receipts
                or len(item["receipt_canonical_sha256"]) != len(expected_receipts)
            ):
                raise ValueError(
                    "offline solve input admitted receipt set differs from its block"
                )
        object.__setattr__(self, "split_seed", split_seed)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(
            self,
            "_block_by_key_sha256",
            MappingProxyType({block.key_sha256: block for block in blocks}),
        )
        object.__setattr__(self, "coverage", _deep_freeze(normalized_coverage))
        object.__setattr__(
            self,
            "question_lineage",
            _deep_freeze(
                {
                    "schema_version": 1,
                    "base_question_root_sha256": question_root,
                    "variants": [dict(item) for item in expected_variants],
                }
            ),
        )
        object.__setattr__(
            self,
            "producer_lineage",
            _deep_freeze(
                {
                    "schema_version": 1,
                    "kind": lineage["kind"],
                    "row_order": lineage["row_order"],
                    "producer_source_sha256": lineage["producer_source_sha256"],
                    "bank_module_source_sha256": lineage[
                        "bank_module_source_sha256"
                    ],
                    "inputs": inputs,
                }
            ),
        )

    @property
    def canonical_payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "selection": SELECTION,
            "split_seed": self.split_seed,
            "runtime_operation": "preindexed_cached_row_lookup_only",
            "online_solver_calls_per_reset": 0,
            "online_solver_calls_per_step": 0,
            "missing_block_policy": "fail_closed",
            "target_recipe": CURRENT_LM_RECIPE,
            "target_validity_mask": list(CURRENT_LM_VALIDITY_MASK),
            "coverage": _deep_plain(self.coverage),
            "question_lineage": _deep_plain(self.question_lineage),
            "producer_lineage": _deep_plain(self.producer_lineage),
            "offline_solve_ledger": {
                "proposed_count": sum(
                    item["proposed_count"] for item in self.producer_lineage["inputs"]
                ),
                "admitted_count": sum(
                    item["admitted_count"] for item in self.producer_lineage["inputs"]
                ),
                "rejected_count": sum(
                    rejection["count"]
                    for item in self.producer_lineage["inputs"]
                    for rejection in item["rejections"]
                ),
                "by_reason": sorted(
                    (
                        {
                            "reason": reason,
                            "count": sum(
                                rejection["count"]
                                for item in self.producer_lineage["inputs"]
                                for rejection in item["rejections"]
                                if rejection["reason"] == reason
                            ),
                        }
                        for reason in {
                            rejection["reason"]
                            for item in self.producer_lineage["inputs"]
                            for rejection in item["rejections"]
                        }
                    ),
                    key=lambda item: item["reason"],
                ),
                "by_block": [
                    {
                        "block_key_sha256": item["block_key_sha256"],
                        "proposed_count": item["proposed_count"],
                        "admitted_count": item["admitted_count"],
                        "rejected_count": sum(
                            rejection["count"] for rejection in item["rejections"]
                        ),
                        "by_reason": [
                            dict(rejection) for rejection in item["rejections"]
                        ],
                    }
                    for item in sorted(
                        self.producer_lineage["inputs"],
                        key=lambda value: value["block_key_sha256"],
                    )
                ],
            },
            "blocks": [
                block.to_dict()
                for block in sorted(self.blocks, key=lambda item: item.key_sha256)
            ],
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha256_json(self.canonical_payload)

    def to_dict(self) -> dict:
        return {**self.canonical_payload, "canonical_sha256": self.canonical_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "BandedQuestionBank":
        keys = (
            "schema_version",
            "kind",
            "selection",
            "split_seed",
            "runtime_operation",
            "online_solver_calls_per_reset",
            "online_solver_calls_per_step",
            "missing_block_policy",
            "target_recipe",
            "target_validity_mask",
            "coverage",
            "question_lineage",
            "producer_lineage",
            "offline_solve_ledger",
            "blocks",
            "canonical_sha256",
        )
        row = _exact_mapping(value, keys, name="banded question bank")
        fixed = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "selection": SELECTION,
            "runtime_operation": "preindexed_cached_row_lookup_only",
            "online_solver_calls_per_reset": 0,
            "online_solver_calls_per_step": 0,
            "missing_block_policy": "fail_closed",
            "target_recipe": CURRENT_LM_RECIPE,
            "target_validity_mask": list(CURRENT_LM_VALIDITY_MASK),
        }
        if any(row[name] != expected for name, expected in fixed.items()):
            raise ValueError("banded question bank fixed contract mismatch")
        if not isinstance(row["blocks"], (tuple, list)):
            raise ValueError("banded question bank blocks must be a list")
        bank = cls(
            split_seed=row["split_seed"],
            blocks=tuple(BandedQuestionBlock.from_dict(item) for item in row["blocks"]),
            coverage=row["coverage"],
            question_lineage=row["question_lineage"],
            producer_lineage=row["producer_lineage"],
        )
        if row["offline_solve_ledger"] != bank.canonical_payload[
            "offline_solve_ledger"
        ]:
            raise ValueError("banded question offline solve ledger mismatch")
        if _sha256(row["canonical_sha256"], name="bank canonical_sha256") != (
            bank.canonical_sha256
        ):
            raise ValueError("banded question bank canonical SHA mismatch")
        return bank

    def block_for_birth(self, birth: object) -> BandedQuestionBlock:
        key_sha = _sha256_json(_block_key_for_birth(birth))
        try:
            return self._block_by_key_sha256[key_sha]
        except KeyError as exc:
            raise ValueError(
                "banded question bank has no exact block for the live "
                f"levels/action/profile/solver identity {key_sha}"
            ) from exc

    def preflight_pending_releases(
        self, pending_releases: Sequence[object]
    ) -> dict:
        """Fail before drain/commit unless every next published block exists.

        This method is pure: it neither advances bank cursors nor touches the
        curriculum object.  The release coordinator must call it while the old
        domain is still installed and before burning live rollout work.
        """

        pending = tuple(item for item in pending_releases if item is not None)
        if not pending:
            raise ValueError("banded question release preflight has no pending release")
        checked = []
        for item in pending:
            key = getattr(item, "key", None)
            action_uid = getattr(key, "action_uid", None)
            profile_sha256 = getattr(key, "profile_sha256", None)
            mobility = getattr(key, "mobility", None)
            if (
                type(action_uid) is not int
                or action_uid < 1
                or type(profile_sha256) is not str
                or mobility not in ("no_move", "move")
            ):
                raise ValueError("pending release lacks exact action/profile/mobility identity")
            levels = _levels_from_pending_release(item)
            matches = [
                block
                for block in self.blocks
                if block.key["action_uid"] == action_uid
                and block.key["profile_sha256"] == profile_sha256
                and block.key["mobility_mode"] == mobility
                and block.key["levels_sha256"] == levels.canonical_sha256
            ]
            if len(matches) != 1:
                raise ValueError(
                    "banded question release preflight has no unique exact next "
                    "block for action/profile/mobility/domain levels"
                )
            checked.append(
                {
                    "release_id_sha256": _sha256(
                        getattr(item, "release_id_sha256", None),
                        name="pending release_id_sha256",
                    ),
                    "action_uid": action_uid,
                    "levels_sha256": levels.canonical_sha256,
                    "block_key_sha256": matches[0].key_sha256,
                    "block_content_sha256": matches[0].content_sha256,
                }
            )
        payload = {
            "schema_version": 1,
            "kind": "action_ball_banded_question_release_preflight",
            "bank_canonical_sha256": self.canonical_sha256,
            "release_count": len(checked),
            "releases": checked,
            "online_solver_calls": 0,
        }
        return {**payload, "preflight_sha256": _sha256_json(payload)}


def load_banded_question_bank(
    path: str | Path, *, expected_file_sha256: str
) -> BandedQuestionBank:
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _sha256(
        expected_file_sha256, name="action_ball_banded_question_bank_sha256"
    ):
        raise ValueError("banded question bank file SHA differs from configured authority")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("banded question bank is not UTF-8 JSON") from error
    return BandedQuestionBank.from_dict(document)


def _template_sample_identity(
    *, template: object, birth: object, sample_index: int, draw_start: int
) -> dict:
    payload = template._sampler_identity_payload()
    payload.update(
        {
            "sample_index": sample_index,
            "action_uid": birth.action_uid,
            "domain_epoch": birth.domain_epoch,
            "domain_levels": birth.domain_levels.to_dict(),
            "birth_id": birth.sampler_birth_sha256,
            "profile_sha256": birth.profile_sha256,
            "levels_sha256": birth.levels_sha256,
            "draw_start": draw_start,
            "draw_end": draw_start + _runtime.SAMPLER_SAMPLE_DRAW_COUNT,
            "mobility_mode": birth.mobility_mode,
            "base_yaw_rad": birth.base_yaw_rad,
            "base_start_w_m": list(birth.base_spawn_w_m),
        }
    )
    if template.sampling_mixture is not None:
        payload.update(
            {
                "birth_index": birth.sampler_birth_index,
                "birth_sampling_stratum": birth.sampling_stratum,
                "birth_sampling_levels": birth.sampling_levels.to_dict(),
                "birth_frontier_arm": birth.frontier_arm,
            }
        )
    return payload


_MATERIALIZED_FIELDS = (
    "base_goal_w_m",
    "base_spawn_latent_w_m",
    "base_travel_latent_b_yaw_m",
    "contact_offset_from_base_goal_b_yaw_m",
    "ball_contact_w_m",
    "racket_site_target_w_m",
    "time_to_contact_s",
    "incoming_speed_mps",
    "incoming_direction_b_yaw",
    "incoming_velocity_w_mps",
    "spin_magnitude_radps",
    "spin_direction_b_yaw",
    "incoming_spin_w_radps",
    "landing_aim_w_xy_m",
    "mount_normal_sign",
    "racket_normal_w",
    "reference_racket_quat_wxyz",
    "reference_racket_angular_velocity_w_radps",
    "racket_command_quat_wxyz",
    "racket_face_center_velocity_w_mps",
    "racket_site_velocity_w_mps",
    "racket_command_angular_velocity_w_radps",
    "geometry_source_sha256",
    "reference_t_hit_s",
    "reference_t_cycle_s",
    "reference_racket_site_speed_mps",
    "required_racket_site_speed_mps",
    "reaction_margin_s",
    "teacher_rate_min",
    "teacher_rate_max",
    "teacher_rate",
    "scaled_t_hit_s",
    "scaled_t_cycle_s",
    "pre_swing_wait_s",
    "solver_residual_m",
    "contact_time_step_s",
    "time_to_contact_tick",
    "birth_index",
    "birth_sampling_stratum",
    "birth_sampling_levels",
    "birth_frontier_arm",
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    "counter_rally_task",
)


def _empty_block_state(block: BandedQuestionBlock) -> dict:
    return {
        "block_key_sha256": block.key_sha256,
        "block_content_sha256": block.content_sha256,
        "cursor": 0,
        "sample_highwater_index": -1,
        "sample_highwater_draw_end": 0,
        "emitted_count": 0,
        "rolling_digest": _sha256_json(
            {"kind": "banded_question_block_empty_transcript", "block": block.content_sha256}
        ),
    }


class BandedQuestionBankSolver:
    """Pool solver backed only by exact precomputed cache rows."""

    # LazyActionTaskPool owns compact per-birth roots.  Duplicating them here
    # would reintroduce an unbounded birth history; block evidence below is the
    # solver's sole mutable authority.
    pool_owns_birth_task_transcripts = True

    def __init__(
        self, *, bank: BandedQuestionBank, solver_contract_sha256: str
    ) -> None:
        if not isinstance(bank, BandedQuestionBank):
            raise TypeError("bank must be BandedQuestionBank")
        self.bank = bank
        self.solver_contract_sha256 = _sha256(
            solver_contract_sha256, name="solver_contract_sha256"
        )
        block_solver_shas = {str(block.key["solver_sha256"]) for block in bank.blocks}
        if block_solver_shas != {self.solver_contract_sha256}:
            raise ValueError(
                "banded question bank solver identity differs from the configured "
                "current_lm solver contract"
            )
        self.state_owner_sha256 = _sha256_json(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "kind": "banded_question_bank_state_owner",
                "bank_canonical_sha256": bank.canonical_sha256,
                "solver_contract_sha256": self.solver_contract_sha256,
            }
        )
        self._block_state = {
            block.key_sha256: _empty_block_state(block) for block in bank.blocks
        }

    @property
    def online_lm_calls(self) -> int:
        return 0

    @property
    def physical_rng_draws(self) -> int:
        return 0

    def _action_highwater(self, action_uid: int, states=None) -> Tuple[int, int]:
        states = self._block_state if states is None else states
        rows = [
            states[block.key_sha256]
            for block in self.bank.blocks
            if block.key["action_uid"] == action_uid
        ]
        if not rows:
            return (-1, 0)
        return (
            max(row["sample_highwater_index"] for row in rows),
            max(row["sample_highwater_draw_end"] for row in rows),
        )

    @staticmethod
    def _assert_birth_matches_template(birth: object, template: object) -> None:
        expected = (
            template.action_uid,
            template.action_slot,
            template.profile_sha256,
            template.motion_sha256,
            template.manifest_sha256,
            template.sampler_sha256,
            template.domain_authority_sha256,
            template.arm_catalog_sha256,
            template.physics_sha256,
            template.solver_sha256,
            template.mobility_mode,
            template.levels_sha256,
            template.domain_levels,
        )
        actual = (
            birth.action_uid,
            birth.action_slot,
            birth.profile_sha256,
            birth.motion_sha256,
            birth.manifest_sha256,
            birth.sampler_sha256,
            birth.domain_authority_sha256,
            birth.arm_catalog_sha256,
            birth.physics_sha256,
            birth.solver_sha256,
            birth.mobility_mode,
            birth.levels_sha256,
            birth.domain_levels,
        )
        if actual != expected or _base_identity_for_receipt(birth) != (
            _base_identity_for_receipt(template)
        ):
            raise ValueError(
                "live birth differs from selected precomputed row identity/base"
            )

    def _materialize(
        self,
        *,
        block: BandedQuestionBlock,
        birth: object,
        swing_generation: int,
        sample_index: int,
        draw_start: int,
    ) -> Tuple[object, int]:
        row_index = block.row_index_for(
            birth_sha256=birth.canonical_sha256,
            swing_generation=swing_generation,
            split_seed=self.bank.split_seed,
            sample_index=sample_index,
            base_yaw_rad=birth.base_yaw_rad,
            base_quat_wxyz=birth.base_quat_wxyz,
            base_spawn_w_m=birth.base_spawn_w_m,
        )
        template = block.rows[row_index]
        self._assert_birth_matches_template(birth, template)
        identity = _template_sample_identity(
            template=template,
            birth=birth,
            sample_index=sample_index,
            draw_start=draw_start,
        )
        kwargs = {name: getattr(template, name) for name in _MATERIALIZED_FIELDS}
        if template.sampling_mixture is not None:
            kwargs.update(
                {
                    "birth_index": birth.sampler_birth_index,
                    "birth_sampling_stratum": birth.sampling_stratum,
                    "birth_sampling_levels": birth.sampling_levels,
                    "birth_frontier_arm": birth.frontier_arm,
                }
            )
        receipt = _runtime.ActionBallTaskReceipt.from_birth(
            birth,
            sample_sha256=_sha256_json(identity),
            sample_index=sample_index,
            sample_draw_start=draw_start,
            sample_draw_end=draw_start + _runtime.SAMPLER_SAMPLE_DRAW_COUNT,
            swing_generation=swing_generation,
            **kwargs,
        )
        return receipt, row_index

    def materialize_many(self, requests: Sequence[object]):
        requests = tuple(requests)
        if not requests:
            raise ValueError("banded question bank request batch must be non-empty")
        staged = {key: dict(value) for key, value in self._block_state.items()}
        batches = []
        for request in requests:
            block = self.bank.block_for_birth(request.birth)
            state = staged[block.key_sha256]
            receipts = []
            indices = []
            for offset in range(request.minimum_receipts):
                last_index, last_draw_end = self._action_highwater(
                    request.action_uid, staged
                )
                sample_index = last_index + 1
                draw_start = max(last_draw_end, int(request.birth.sampler_draw_end))
                swing_generation = request.swing_generation_start + offset
                receipt, row_index = self._materialize(
                    block=block,
                    birth=request.birth,
                    swing_generation=swing_generation,
                    sample_index=sample_index,
                    draw_start=draw_start,
                )
                state["cursor"] += 1
                state["sample_highwater_index"] = sample_index
                state["sample_highwater_draw_end"] = receipt.sample_draw_end
                state["emitted_count"] += 1
                state["rolling_digest"] = _sha256_json(
                    {
                        "prior": state["rolling_digest"],
                        "block_content_sha256": block.content_sha256,
                        "birth_sha256": request.birth.canonical_sha256,
                        "refill_index": request.refill_index,
                        "sample_index": sample_index,
                        "row_index": row_index,
                        "task_receipt_sha256": receipt.canonical_sha256,
                    }
                )
                receipts.append(receipt)
                indices.append(sample_index)
            batches.append(
                _runtime.ActionPoolRefillBatch(
                    action_uid=request.action_uid,
                    proposed_count=len(indices),
                    proposal_sample_indices=tuple(indices),
                    receipts=tuple(receipts),
                )
            )
        self._block_state = staged
        return tuple(batches)

    def __call__(self, request: object):
        return self.materialize_many((request,))[0]

    def solve_many(self, requests: Sequence[object]):
        return self.materialize_many(requests)

    def _assert_receipt_matches_bank(self, receipt: object) -> None:
        if not isinstance(receipt, _runtime.ActionBallTaskReceipt):
            raise ValueError("banded question authority requires a task receipt")
        key_sha = _sha256_json(_block_key_for_receipt(receipt))
        blocks = {block.key_sha256: block for block in self.bank.blocks}
        if key_sha not in blocks:
            raise ValueError("task receipt has no exact banded question block")
        block = blocks[key_sha]
        row_index = block.row_index_for(
            birth_sha256=receipt.birth_sha256,
            swing_generation=receipt.swing_generation,
            split_seed=self.bank.split_seed,
            sample_index=receipt.sample_index,
            base_yaw_rad=receipt.base_yaw_rad,
            base_quat_wxyz=receipt.base_quat_wxyz,
            base_spawn_w_m=receipt.base_spawn_w_m,
        )
        template = block.rows[row_index]
        expected_sample_sha256 = _sha256_json(
            _template_sample_identity(
                template=template,
                birth=receipt,
                sample_index=receipt.sample_index,
                draw_start=receipt.sample_draw_start,
            )
        )
        if receipt.sample_sha256 != expected_sample_sha256:
            raise ValueError(
                "task receipt sample identity differs from selected cached row"
            )
        for name in _MATERIALIZED_FIELDS:
            if name in (
                "birth_index",
                "birth_sampling_stratum",
                "birth_sampling_levels",
                "birth_frontier_arm",
            ):
                continue
            if getattr(receipt, name) != getattr(template, name):
                raise ValueError(
                    f"task receipt field {name} differs from selected cached row"
                )

    def assert_emitted_sample(self, receipt: object) -> None:
        self._assert_receipt_matches_bank(receipt)

    def assert_emitted_tasks(self, receipts: Sequence[object]) -> None:
        for receipt in receipts:
            self._assert_receipt_matches_bank(receipt)

    def emitted_task_count_for(self, action_uid: int) -> int:
        return sum(
            state["emitted_count"]
            for block, state in (
                (block, self._block_state[block.key_sha256])
                for block in self.bank.blocks
            )
            if block.key["action_uid"] == action_uid
        )

    def task_transcript_for_birth(self, _birth_sha256: str):
        raise RuntimeError(
            "banded question bank delegates per-birth transcript roots to the pool"
        )

    def assert_proposal_assignments(self, assignments: Sequence[object]) -> None:
        for assignment in assignments:
            block = self.bank.block_for_birth(assignment.birth)
            highwater = self.sample_highwater_for(assignment.birth.action_uid)[0]
            if any(index > highwater for index in assignment.proposal_sample_indices):
                raise ValueError("banded question assignment exceeds solver highwater")
            if block.key_sha256 not in self._block_state:
                raise ValueError("banded question assignment block is unavailable")

    def sample_highwater_for(self, action_uid: int):
        return self._action_highwater(action_uid)

    def state_dict(self) -> dict:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "bank_canonical_sha256": self.bank.canonical_sha256,
            "solver_contract_sha256": self.solver_contract_sha256,
            "state_owner_sha256": self.state_owner_sha256,
            "physical_rng_draws": 0,
            "online_lm_calls": 0,
            "blocks": [
                dict(self._block_state[key]) for key in sorted(self._block_state)
            ],
        }
        return {**payload, "integrity_sha256": _sha256_json(payload)}

    def load_state_dict(self, state: object) -> None:
        keys = (
            "schema_version",
            "bank_canonical_sha256",
            "solver_contract_sha256",
            "state_owner_sha256",
            "physical_rng_draws",
            "online_lm_calls",
            "blocks",
            "integrity_sha256",
        )
        row = _exact_mapping(state, keys, name="banded question solver state")
        payload = {name: row[name] for name in keys if name != "integrity_sha256"}
        if _sha256(row["integrity_sha256"], name="state integrity_sha256") != (
            _sha256_json(payload)
        ):
            raise ValueError("banded question solver state integrity mismatch")
        fixed = {
            "schema_version": STATE_SCHEMA_VERSION,
            "bank_canonical_sha256": self.bank.canonical_sha256,
            "solver_contract_sha256": self.solver_contract_sha256,
            "state_owner_sha256": self.state_owner_sha256,
            "physical_rng_draws": 0,
            "online_lm_calls": 0,
        }
        if any(row[name] != value for name, value in fixed.items()):
            raise ValueError("banded question solver state identity mismatch")
        if not isinstance(row["blocks"], (tuple, list)):
            raise ValueError("banded question solver blocks state must be a list")
        block_keys = (
            "block_key_sha256",
            "block_content_sha256",
            "cursor",
            "sample_highwater_index",
            "sample_highwater_draw_end",
            "emitted_count",
            "rolling_digest",
        )
        restored = {}
        available = {block.key_sha256: block for block in self.bank.blocks}
        for raw in row["blocks"]:
            item = _exact_mapping(raw, block_keys, name="banded block solver state")
            key_sha = _sha256(item["block_key_sha256"], name="block_key_sha256")
            if key_sha in restored or key_sha not in available:
                raise ValueError("banded question solver state has an unknown/duplicate block")
            block = available[key_sha]
            if item["block_content_sha256"] != block.content_sha256:
                raise ValueError("banded question block content identity mismatch")
            cursor = _plain_int(item["cursor"], name="block cursor")
            emitted = _plain_int(item["emitted_count"], name="block emitted_count")
            index = item["sample_highwater_index"]
            draw_end = item["sample_highwater_draw_end"]
            if type(index) is not int or type(draw_end) is not int:
                raise ValueError("banded question block highwater must use integers")
            if cursor != emitted or (emitted == 0) != (index == -1 and draw_end == 0):
                raise ValueError("banded question block cursor/highwater mismatch")
            if emitted and (index < 0 or draw_end < 1):
                raise ValueError("banded question block highwater is invalid")
            restored[key_sha] = {
                "block_key_sha256": key_sha,
                "block_content_sha256": block.content_sha256,
                "cursor": cursor,
                "sample_highwater_index": index,
                "sample_highwater_draw_end": draw_end,
                "emitted_count": emitted,
                "rolling_digest": _sha256(item["rolling_digest"], name="rolling_digest"),
            }
        if set(restored) != set(available):
            raise ValueError("banded question solver state omits a bank block")
        for action_uid in {
            int(block.key["action_uid"]) for block in self.bank.blocks
        }:
            action_rows = [
                restored[block.key_sha256]
                for block in self.bank.blocks
                if block.key["action_uid"] == action_uid
            ]
            emitted_count = sum(row["emitted_count"] for row in action_rows)
            highwater = max(
                row["sample_highwater_index"] for row in action_rows
            )
            if highwater != emitted_count - 1:
                raise ValueError(
                    "banded question action highwater does not conserve "
                    "the contiguous admitted sample tape"
                )
        self._block_state = restored


__all__ = [
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "KIND",
    "SELECTION",
    "BASE_IDENTITY_KIND",
    "CURRENT_LM_RECIPE",
    "CURRENT_LM_VALIDITY_MASK",
    "BandedQuestionBlock",
    "BandedQuestionBank",
    "BandedQuestionBankSolver",
    "base_question_root_for_blocks",
    "question_lineage_for_blocks",
    "load_banded_question_bank",
]
