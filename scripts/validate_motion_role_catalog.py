#!/usr/bin/env python3
"""Fail-loud validator for the motion-role semantic catalog.

人话:configs/motion_role_catalog.json 是动作素材语义的唯一真源——哪些是原地击球、
哪四条是共享横移脚步模块、哪一对 npz 是现役 formal 训练动作。本脚本把 catalog 与三份
内容寻址来源清单逐条对账:intake 条目必须与两份 intake 清单精确互覆盖(asset_id 与
SHA-256 逐字一致,清单文件本身重算 SHA 也要对上),v4rg 一对必须与 phase1 fresh 资产
清单一致;角色/方向/触发范围/训练授权规则全部 fail-closed,任何一条不符即整体拒绝。
它不授权任何训练、仿真或真机动作,只锁语义。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "configs" / "motion_role_catalog.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

V4RG_MANIFEST = "configs/phase1_fresh_v3_asset_manifest_20260711.json"
INTAKE_MANIFESTS = (
    "configs/motion_video_intake_20260711.json",
    "configs/motion_video_intake_20260713.json",
)
EXPECTED_SOURCE_MANIFESTS = (V4RG_MANIFEST,) + INTAKE_MANIFESTS

STATIONARY_ROLE = "stationary_strike"
FOOTWORK_ROLE = "shared_lateral_footwork_module"
ALLOWED_ROLES = {STATIONARY_ROLE, FOOTWORK_ROLE}
ALLOWED_KINDS = {"runtime_npz", "raw_video"}
MOVEMENT_DIRECTIONS = {"left", "right"}
FOOTWORK_PREFIX = "motion/"
FOOTWORK_SCOPE = "strike_intent_triggered_only"
STATIONARY_SCOPE = "own_action_slot"
STATIONARY_LOWER_BODY = "legs_free_weight_shift_allowed"
FOOTWORK_LOWER_BODY = "prepare_strike_recover_lower_body_and_root_reference"
FOOTWORK_GATE_STATUS = "rejected_stance_0_of_4"

COMMON_ENTRY_KEYS = {
    "asset_id", "asset_kind", "source_manifest", "sha256", "motion_role",
    "stationary_strike", "lower_body_semantics", "movement_direction",
    "shared_across_action_slots", "activation_scope", "training_authorized",
    "vendor_mujoco_dynamic_pass", "grandfathered_formal_runtime_pair",
}


class CatalogError(ValueError):
    """A catalog or source-manifest contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> Any:
    raise CatalogError(f"non-finite JSON constant: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise CatalogError(f"non-finite JSON number: {token}")
    return value


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except (OSError, json.JSONDecodeError, CatalogError) as exc:
        raise CatalogError(f"cannot read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: root must be a JSON object")
    return data


def _required(mapping: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise CatalogError(f"{context} missing required keys: {missing}")


def _require_bool(entry: dict[str, Any], key: str, context: str) -> bool:
    value = entry[key]
    if not isinstance(value, bool):
        raise CatalogError(f"{context}.{key} must be a JSON boolean, got {value!r}")
    return value


def _resolve_manifest_path(repo_root: Path, relpath: str) -> Path:
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise CatalogError(f"source manifest path must be a safe relative path: {relpath!r}")
    return repo_root / rel


def _load_source_manifests(
    catalog: dict[str, Any], repo_root: Path
) -> dict[str, dict[str, Any]]:
    recorded = catalog["source_manifests"]
    if not isinstance(recorded, dict):
        raise CatalogError("source_manifests must be an object")
    if set(recorded) != set(EXPECTED_SOURCE_MANIFESTS):
        raise CatalogError(
            f"source_manifests must record exactly {sorted(EXPECTED_SOURCE_MANIFESTS)}, "
            f"got {sorted(recorded)}"
        )
    loaded: dict[str, dict[str, Any]] = {}
    for relpath, binding in recorded.items():
        context = f"source_manifests[{relpath!r}]"
        if not isinstance(binding, dict):
            raise CatalogError(f"{context} must be an object")
        _required(binding, {"sha256"}, context)
        expected = binding["sha256"]
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise CatalogError(f"{context}.sha256 must be 64 lowercase hex characters")
        path = _resolve_manifest_path(repo_root, relpath)
        if not path.is_file():
            raise CatalogError(f"{context}: missing source manifest file {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise CatalogError(
                f"{context}: recomputed SHA-256 {actual} does not match recorded {expected}"
            )
        loaded[relpath] = load_strict_json(path)
    return loaded


def _intake_assets(manifest: dict[str, Any], relpath: str) -> dict[str, dict[str, Any]]:
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise CatalogError(f"{relpath}: assets must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            raise CatalogError(f"{relpath}: every asset must be an object")
        _required(asset, {"id", "sha256", "source_relpath"}, f"{relpath} asset")
        asset_id = asset["id"]
        if asset_id in by_id:
            raise CatalogError(f"{relpath}: duplicate intake asset id {asset_id!r}")
        by_id[asset_id] = asset
    return by_id


def _validate_v4rg_entry(
    entry: dict[str, Any], v4rg_manifest: dict[str, Any], context: str
) -> None:
    _required(entry, {"manifest_motion_key", "source_basename"}, context)
    key = entry["manifest_motion_key"]
    motions = v4rg_manifest.get("motions")
    if not isinstance(motions, dict) or key not in motions:
        raise CatalogError(f"{context}: manifest_motion_key {key!r} not in v4rg motions")
    motion = motions[key]
    if entry["source_basename"] != motion["basename"]:
        raise CatalogError(
            f"{context}: source_basename {entry['source_basename']!r} != manifest "
            f"basename {motion['basename']!r}"
        )
    if entry["asset_id"] != Path(motion["basename"]).stem:
        raise CatalogError(
            f"{context}: asset_id must equal the npz basename stem "
            f"{Path(motion['basename']).stem!r}"
        )
    if entry["sha256"] != motion["sha256"]:
        raise CatalogError(
            f"{context}: sha256 {entry['sha256']} != v4rg manifest {motion['sha256']}"
        )
    if entry["motion_role"] != STATIONARY_ROLE:
        raise CatalogError(f"{context}: the v4rg pair must be {STATIONARY_ROLE!r}")


def _validate_stationary(entry: dict[str, Any], context: str) -> None:
    if _require_bool(entry, "stationary_strike", context) is not True:
        raise CatalogError(f"{context}: {STATIONARY_ROLE} requires stationary_strike=true")
    if entry["movement_direction"] is not None:
        raise CatalogError(f"{context}: {STATIONARY_ROLE} requires movement_direction=null")
    if _require_bool(entry, "shared_across_action_slots", context) is not False:
        raise CatalogError(
            f"{context}: {STATIONARY_ROLE} requires shared_across_action_slots=false"
        )
    if entry["activation_scope"] != STATIONARY_SCOPE:
        raise CatalogError(
            f"{context}: {STATIONARY_ROLE} requires activation_scope={STATIONARY_SCOPE!r}"
        )
    if entry["lower_body_semantics"] != STATIONARY_LOWER_BODY:
        raise CatalogError(
            f"{context}: {STATIONARY_ROLE} requires "
            f"lower_body_semantics={STATIONARY_LOWER_BODY!r}"
        )


def _validate_footwork(entry: dict[str, Any], context: str) -> None:
    if _require_bool(entry, "stationary_strike", context) is not False:
        raise CatalogError(f"{context}: {FOOTWORK_ROLE} requires stationary_strike=false")
    if _require_bool(entry, "shared_across_action_slots", context) is not True:
        raise CatalogError(
            f"{context}: {FOOTWORK_ROLE} requires shared_across_action_slots=true; a "
            "footwork module bound to a single action slot is a forbidden standalone "
            "locomotion action"
        )
    if entry["activation_scope"] != FOOTWORK_SCOPE:
        raise CatalogError(
            f"{context}: {FOOTWORK_ROLE} requires activation_scope={FOOTWORK_SCOPE!r}; "
            "there is no standalone locomotion or stop-teacher activation"
        )
    if entry["lower_body_semantics"] != FOOTWORK_LOWER_BODY:
        raise CatalogError(
            f"{context}: {FOOTWORK_ROLE} requires "
            f"lower_body_semantics={FOOTWORK_LOWER_BODY!r}"
        )
    direction = entry["movement_direction"]
    if direction not in MOVEMENT_DIRECTIONS:
        raise CatalogError(
            f"{context}: {FOOTWORK_ROLE} requires movement_direction in "
            f"{sorted(MOVEMENT_DIRECTIONS)}"
        )
    _required(entry, {"input_gate_status"}, context)
    if entry["input_gate_status"] != FOOTWORK_GATE_STATUS:
        raise CatalogError(
            f"{context}: the M0 footwork clips must keep "
            f"input_gate_status={FOOTWORK_GATE_STATUS!r}; renaming the role does not "
            "overwrite the safety fact"
        )
    filename = Path(str(entry["source_relpath"])).name
    if not filename.startswith(f"{direction}_dang"):
        raise CatalogError(
            f"{context}: movement_direction={direction!r} contradicts source filename "
            f"{filename!r}"
        )


def validate_catalog(catalog: dict[str, Any], repo_root: Path) -> dict[str, int]:
    _required(
        catalog,
        {
            "schema_version", "catalog_id", "purpose", "human_note",
            "semantic_authority", "formal_runtime_pair", "frozen_semantics",
            "source_manifests", "entries",
        },
        "catalog",
    )
    schema_version = catalog["schema_version"]
    if isinstance(schema_version, bool) or schema_version != 1:
        raise CatalogError(f"unsupported schema_version={schema_version!r}; expected 1")

    authority = catalog["semantic_authority"]
    if not isinstance(authority, dict):
        raise CatalogError("semantic_authority must be an object")
    _required(authority, {"statement", "supersedes"}, "semantic_authority")
    supersedes = authority["supersedes"]
    if not isinstance(supersedes, list) or not supersedes:
        raise CatalogError("semantic_authority.supersedes must be a non-empty list")
    superseded_manifests = set()
    for index, item in enumerate(supersedes):
        context = f"semantic_authority.supersedes[{index}]"
        if not isinstance(item, dict):
            raise CatalogError(f"{context} must be an object")
        _required(item, {"manifest", "fields", "note"}, context)
        superseded_manifests.add(item["manifest"])
    if INTAKE_MANIFESTS[1] not in superseded_manifests:
        raise CatalogError(
            "semantic_authority.supersedes must declare the historical role/action_slot "
            f"fields of {INTAKE_MANIFESTS[1]}"
        )

    pair = catalog["formal_runtime_pair"]
    if not isinstance(pair, dict):
        raise CatalogError("formal_runtime_pair must be an object")
    _required(pair, {"asset_ids", "statement"}, "formal_runtime_pair")
    pair_ids = pair["asset_ids"]
    if (
        not isinstance(pair_ids, list)
        or len(pair_ids) != 2
        or len(set(pair_ids)) != 2
        or not all(isinstance(item, str) for item in pair_ids)
    ):
        raise CatalogError("formal_runtime_pair.asset_ids must list exactly two distinct ids")

    manifests = _load_source_manifests(catalog, repo_root)
    intake_by_manifest = {
        relpath: _intake_assets(manifests[relpath], relpath)
        for relpath in INTAKE_MANIFESTS
    }
    expected_entry_count = 2 + sum(len(assets) for assets in intake_by_manifest.values())

    entries = catalog["entries"]
    if not isinstance(entries, list) or not entries:
        raise CatalogError("catalog entries must be a non-empty list")
    if len(entries) != expected_entry_count:
        raise CatalogError(
            f"catalog must have exactly {expected_entry_count} entries "
            f"(v4rg pair + every intake asset), got {len(entries)}"
        )

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_intake: dict[str, set[str]] = {relpath: set() for relpath in INTAKE_MANIFESTS}
    grandfathered_ids: list[str] = []
    v4rg_keys: set[str] = set()
    role_counts = {STATIONARY_ROLE: 0, FOOTWORK_ROLE: 0}

    for index, entry in enumerate(entries):
        context = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise CatalogError(f"{context} must be an object")
        _required(entry, COMMON_ENTRY_KEYS, context)
        asset_id = entry["asset_id"]
        if not isinstance(asset_id, str) or not ASSET_ID_RE.fullmatch(asset_id):
            raise CatalogError(f"{context}.asset_id must match {ASSET_ID_RE.pattern!r}")
        context = f"entries[{index}] ({asset_id})"
        if asset_id in seen_ids:
            raise CatalogError(f"duplicate catalog asset_id {asset_id!r}")
        seen_ids.add(asset_id)

        sha = entry["sha256"]
        if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
            raise CatalogError(f"{context}: sha256 must be 64 lowercase hex characters")
        if sha in seen_hashes:
            raise CatalogError(f"{context}: duplicate asset sha256 {sha}")
        seen_hashes.add(sha)

        kind = entry["asset_kind"]
        if kind not in ALLOWED_KINDS:
            raise CatalogError(f"{context}: unsupported asset_kind {kind!r}")
        role = entry["motion_role"]
        if role not in ALLOWED_ROLES:
            raise CatalogError(
                f"{context}: unsupported motion_role {role!r}; standalone locomotion "
                "actions and standalone stop teachers do not exist in this library"
            )
        role_counts[role] += 1

        source_manifest = entry["source_manifest"]
        if source_manifest not in EXPECTED_SOURCE_MANIFESTS:
            raise CatalogError(
                f"{context}: source_manifest {source_manifest!r} is not a recorded "
                "source manifest"
            )

        vendor_pass = _require_bool(entry, "vendor_mujoco_dynamic_pass", context)
        authorized = _require_bool(entry, "training_authorized", context)
        grandfathered = _require_bool(entry, "grandfathered_formal_runtime_pair", context)
        if authorized and not vendor_pass and not grandfathered:
            raise CatalogError(
                f"{context}: training_authorized=true without a vendor MuJoCo dynamic "
                "pass is only allowed for the grandfathered formal runtime pair"
            )
        if grandfathered:
            grandfathered_ids.append(asset_id)

        if kind == "runtime_npz":
            if source_manifest != V4RG_MANIFEST:
                raise CatalogError(
                    f"{context}: runtime_npz entries must come from {V4RG_MANIFEST}"
                )
            if not grandfathered:
                raise CatalogError(
                    f"{context}: the v4rg runtime pair must be explicitly marked "
                    "grandfathered_formal_runtime_pair=true"
                )
            _validate_v4rg_entry(entry, manifests[V4RG_MANIFEST], context)
            key = entry["manifest_motion_key"]
            if key in v4rg_keys:
                raise CatalogError(f"{context}: duplicate manifest_motion_key {key!r}")
            v4rg_keys.add(key)
            _validate_stationary(entry, context)
            continue

        # raw_video entries
        if source_manifest == V4RG_MANIFEST:
            raise CatalogError(f"{context}: raw_video entries must come from an intake")
        if grandfathered:
            raise CatalogError(
                f"{context}: only the v4rg runtime pair may be grandfathered"
            )
        if authorized:
            raise CatalogError(
                f"{context}: raw video assets are never training_authorized"
            )
        _required(entry, {"source_relpath"}, context)
        intake_assets = intake_by_manifest[source_manifest]
        if asset_id not in intake_assets:
            raise CatalogError(
                f"{context}: asset_id not present in {source_manifest}"
            )
        intake_asset = intake_assets[asset_id]
        if sha != intake_asset["sha256"]:
            raise CatalogError(
                f"{context}: sha256 {sha} != intake value {intake_asset['sha256']}"
            )
        if entry["source_relpath"] != intake_asset["source_relpath"]:
            raise CatalogError(
                f"{context}: source_relpath {entry['source_relpath']!r} != intake value "
                f"{intake_asset['source_relpath']!r}"
            )
        seen_intake[source_manifest].add(asset_id)

        is_footwork_path = str(entry["source_relpath"]).startswith(FOOTWORK_PREFIX)
        if is_footwork_path and role != FOOTWORK_ROLE:
            raise CatalogError(
                f"{context}: every motion/ clip is the shared lateral footwork module, "
                f"got role {role!r}"
            )
        if not is_footwork_path and role != STATIONARY_ROLE:
            raise CatalogError(
                f"{context}: every non-motion/ strike video is a stationary strike, "
                f"got role {role!r}"
            )
        if role == FOOTWORK_ROLE:
            _validate_footwork(entry, context)
        else:
            _validate_stationary(entry, context)

    if v4rg_keys != {"forehand", "backhand"}:
        raise CatalogError(
            f"catalog must bind exactly the v4rg forehand+backhand pair, got {sorted(v4rg_keys)}"
        )
    for relpath, assets in intake_by_manifest.items():
        missing = sorted(set(assets) - seen_intake[relpath])
        if missing:
            raise CatalogError(
                f"exact-cover violation: {relpath} assets missing from catalog: {missing}"
            )
    if sorted(grandfathered_ids) != sorted(pair_ids):
        raise CatalogError(
            "grandfathered_formal_runtime_pair entries must be exactly "
            f"formal_runtime_pair.asset_ids {sorted(pair_ids)}, got {sorted(grandfathered_ids)}"
        )
    if role_counts[FOOTWORK_ROLE] != 4:
        raise CatalogError(
            f"expected exactly 4 shared-footwork clips, got {role_counts[FOOTWORK_ROLE]}"
        )
    return {
        "entries": len(entries),
        STATIONARY_ROLE: role_counts[STATIONARY_ROLE],
        FOOTWORK_ROLE: role_counts[FOOTWORK_ROLE],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Root that catalog source_manifest paths are relative to",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        catalog = load_strict_json(args.catalog.resolve())
        counts = validate_catalog(catalog, args.repo_root.resolve())
    except CatalogError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(
        "[validate_motion_role_catalog] PASS: "
        f"{counts['entries']} entries, "
        f"{counts[STATIONARY_ROLE]} stationary strikes, "
        f"{counts[FOOTWORK_ROLE]} shared footwork clips"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
