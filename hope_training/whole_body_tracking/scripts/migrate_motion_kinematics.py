#!/usr/bin/env python3
"""Migrate a legacy motion NPZ to the schema-2 COM-velocity/body-order contract.

Formal use is fail-closed: an untagged input requires an explicit
``--source-point`` assertion.  For V5 / MuJoCo-converter artifacts produced
before 2026-07-10 use ``--source-point link_origin``; for original Isaac
``csv_to_npz.py`` artifacts use ``--source-point center_of_mass``.  Never infer
this from a filename.

Example (legacy V5):

    python scripts/migrate_motion_kinematics.py \
      --input hope_forehand_v5.npz --output hope_forehand_v5_comv.npz \
      --source-point link_origin --mjcf /path/a3_pingpong.xml \
      --body-order /path/body_order.txt

If the source was written under an older articulation body order, bind that
order with ``--body-order`` and pass the current runtime order separately:

    python scripts/migrate_motion_kinematics.py \
      --input legacy.npz --output reordered_comv.npz \
      --source-point link_origin --mjcf /path/a3_pingpong.xml \
      --body-order /path/source_body_order.txt \
      --target-body-order /path/current_runtime_body_order.txt

The output preserves every source field, converts link velocity with
``v_com = v_link + omega x R(q) r_link_to_com``, and records the source SHA and
asserted source semantics.  It never overwrites the input.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from motion_kinematics_contract import (
    BODY_LIN_VEL_POINT,
    BODY_POS_POINT,
    KINEMATICS_SCHEMA_VERSION,
    MIGRATION_SOURCE_POINT_KEY,
    MIGRATION_SOURCE_SHA256_KEY,
    MIGRATION_TOOL_KEY,
    link_fd_signature,
    link_velocity_to_com,
    metadata_arrays,
    normalize_body_names,
    read_metadata,
)


BODY_ARRAY_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_body_order(path: str) -> tuple[str, ...]:
    names = tuple(line.strip() for line in Path(path).read_text().splitlines() if line.strip())
    if not names or len(set(names)) != len(names):
        raise SystemExit("[migrate-motion] body-order must contain unique non-empty names")
    return names


def _body_com_offsets(mjcf: str, body_names: tuple[str, ...]) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf)
    offsets = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise SystemExit(f"[migrate-motion] MJCF has no body {name!r}")
        offsets.append(model.body_ipos[bid].copy())
    return np.asarray(offsets, dtype=np.float64)


def migrate_arrays(
    data: dict,
    *,
    source_point: str,
    com_pos_b: np.ndarray | None,
    body_names: tuple[str, ...] | list[str],
    target_body_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict, dict]:
    """Pure array migration used by the CLI and tests."""

    if source_point not in ("link_origin", "center_of_mass"):
        raise ValueError(f"unsupported source point {source_point!r}")
    out = {key: np.array(value, copy=True) for key, value in data.items()}
    old_source = np.asarray(out["body_lin_vel_w"], dtype=np.float64)
    if old_source.ndim != 3:
        raise ValueError(f"body_lin_vel_w must be (T,B,3), got {old_source.shape}")
    source_names = normalize_body_names(body_names, expected_count=old_source.shape[1])
    target_names = (
        source_names
        if target_body_names is None
        else normalize_body_names(target_body_names, expected_count=old_source.shape[1])
    )
    if set(target_names) != set(source_names):
        missing = sorted(set(source_names) - set(target_names))
        extra = sorted(set(target_names) - set(source_names))
        raise ValueError(
            "target body order must be a permutation of the source order; "
            f"missing={missing} extra={extra}"
        )
    permutation = np.asarray([source_names.index(name) for name in target_names], dtype=np.int64)
    reordered = tuple(permutation.tolist()) != tuple(range(len(source_names)))
    for key in BODY_ARRAY_KEYS:
        if key not in out:
            raise ValueError(f"motion is missing required body array {key!r}")
        value = np.asarray(out[key])
        if value.ndim < 2 or value.shape[1] != len(source_names):
            raise ValueError(
                f"{key} must have body axis 1 of length {len(source_names)}, got {value.shape}"
            )
        out[key] = np.take(value, permutation, axis=1)

    old = np.asarray(out["body_lin_vel_w"], dtype=np.float64)
    if old.ndim != 3:
        raise ValueError(f"body_lin_vel_w must be (T,B,3), got {old.shape}")
    if source_point == "link_origin":
        if com_pos_b is None:
            raise ValueError("link_origin migration requires body COM offsets")
        converted = link_velocity_to_com(
            old,
            out["body_ang_vel_w"],
            out["body_quat_w"],
            np.asarray(com_pos_b, dtype=np.float64),
        )
        out["body_lin_vel_w"] = converted.astype(np.asarray(data["body_lin_vel_w"]).dtype)
    else:
        converted = old
    out.update(metadata_arrays(body_names=target_names))
    delta = converted - old
    report = {
        "source_point": source_point,
        "body_order_reordered": reordered,
        "body_permutation": permutation.tolist(),
        "max_velocity_delta_mps": float(np.max(np.linalg.norm(delta, axis=-1))),
        "rms_velocity_delta_mps": float(np.sqrt(np.mean(delta * delta))),
    }
    return out, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-point", choices=("link_origin", "center_of_mass"), default=None,
                    help="required assertion for untagged legacy inputs")
    ap.add_argument("--mjcf", help="required for link_origin -> COM conversion")
    ap.add_argument(
        "--body-order",
        help="source NPZ body-column names; required for legacy files without schema-2 body_names",
    )
    ap.add_argument(
        "--target-body-order",
        help="optional current runtime body order; reorders every body array by name before migration",
    )
    args = ap.parse_args(argv)

    src, dst = Path(args.input).resolve(), Path(args.output).resolve()
    if src == dst:
        raise SystemExit("[migrate-motion] REFUSING to overwrite the source NPZ")
    if not src.is_file():
        raise SystemExit(f"[migrate-motion] input not found: {src}")
    if dst.exists():
        raise SystemExit(f"[migrate-motion] output already exists: {dst}")

    with np.load(src) as loaded:
        meta = read_metadata(loaded)
        data = {key: np.array(loaded[key], copy=True) for key in loaded.files}
        signature = link_fd_signature(loaded)
    source_point = args.source_point
    if (
        source_point is not None
        and meta.body_lin_vel_point in ("link_origin", "center_of_mass")
        and source_point != meta.body_lin_vel_point
    ):
        raise SystemExit(
            "[migrate-motion] --source-point contradicts the source's declared "
            f"body_lin_vel_point={meta.body_lin_vel_point!r}"
        )
    if source_point is None:
        if meta.exact_motion_command_v2:
            source_point = BODY_LIN_VEL_POINT
        elif meta.body_lin_vel_point in ("link_origin", "center_of_mass"):
            source_point = meta.body_lin_vel_point
        else:
            raise SystemExit(
                "[migrate-motion] untagged legacy input: pass --source-point link_origin for old "
                "MuJoCo/retime outputs, or --source-point center_of_mass for old Isaac outputs. "
                f"Content audit={signature}"
            )

    if args.body_order:
        body_names = _read_body_order(args.body_order)
        if meta.body_names is not None and tuple(meta.body_names) != body_names:
            raise SystemExit(
                "[migrate-motion] --body-order disagrees with the source schema body_names"
            )
    elif meta.body_names is not None:
        body_names = tuple(meta.body_names)
    else:
        raise SystemExit(
            "[migrate-motion] legacy input has no bound body order; pass --body-order"
        )

    target_body_names = (
        _read_body_order(args.target_body_order) if args.target_body_order else body_names
    )
    if set(target_body_names) != set(body_names):
        missing = sorted(set(body_names) - set(target_body_names))
        extra = sorted(set(target_body_names) - set(body_names))
        raise SystemExit(
            "[migrate-motion] target body order must be a permutation of the source order; "
            f"missing={missing} extra={extra}"
        )

    com_pos = None
    if source_point == "link_origin":
        if not args.mjcf:
            raise SystemExit(
                "[migrate-motion] link_origin conversion requires --mjcf"
            )
        com_pos = _body_com_offsets(args.mjcf, target_body_names)

    out, report = migrate_arrays(
        data,
        source_point=source_point,
        com_pos_b=com_pos,
        body_names=body_names,
        target_body_names=target_body_names,
    )
    out[MIGRATION_SOURCE_SHA256_KEY] = np.array(_sha256(src))
    out[MIGRATION_SOURCE_POINT_KEY] = np.array(source_point)
    out[MIGRATION_TOOL_KEY] = np.array("migrate_motion_kinematics.py/v2")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    print(
        f"[migrate-motion] {src} -> {dst}; schema={KINEMATICS_SCHEMA_VERSION} "
        f"pos={BODY_POS_POINT} lin_vel={BODY_LIN_VEL_POINT}; source={source_point}; "
        f"body_reordered={report['body_order_reordered']}; "
        f"max_delta={report['max_velocity_delta_mps']:.6f} m/s; "
        f"rms_delta={report['rms_velocity_delta_mps']:.6f} m/s; source_sha256={_sha256(src)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
