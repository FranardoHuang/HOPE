#!/usr/bin/env python3
"""fivebind BINDINGS -> chingmu_manifest_v1-style batch table for the N5 action-ball build.

人话:把 fivebind 四件 upper 剪辑(BINDINGS.json 的锚帧/拍速/拍面/接触点)翻译成
build_action_ball_manifest.py 吃的批次表。fivebind 剪辑世界系 == 训练 env W 系
(verify_five.py: TAB_NEAR_X=0.5, TAB_HALF_W=0.7625, TAB_Z=0.76),所以"hope 系"在
这里定义为 env 系平移 (-0.5, -0.7625)(xy)与 -0.76(z, 仅接触点),使 builder 的
station->spawn 与 ball-station 变换正好还原 env 几何:

    station_xy_hope_m   = root0_xy(npz frame-0 pelvis) - (0.5, 0.7625)
    ball_pos_hit_hope_m = strike_point_env - (0.5, 0.7625, 0.76)
    yaw_before_deg      = npz frame-0 root yaw(与 BINDINGS pelvis_yaw_after_deg 交叉核对)
    v_in_fit_hope_ms    = 逆解标称来球(env 向量,平移不变):venue 来球分布中被锚帧
                          拍面状态经场馆接触法合法回台的子集的质心("inverse-solved
                          from anchor blade state";由 --v-in-json 提供,含 provenance)
    v_out_fit_hope_ms   = 该标称来球在锚帧拍面状态下经场馆接触法的出球(记录用;
                          builder 不消费 v_out)

station 语义备注:BINDINGS 无 station 字段;station 取 npz frame-0 pelvis xy(即
canonical-ready 共享 ready 站位,也是 runtime 出生锚点所用的 root),不取 0——取 0 会把
contact_offset 错锚到剪辑世界原点。

Only produces JSON. Does not launch training, grants no admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TAB_NEAR_X = 0.5
TAB_HALF_W = 0.7625
TAB_Z = 0.76
FPS = 50
CLIPS = ("fh_loop", "bh_loop_c", "bh_block", "s0_highpress")
FAMILY_TO_BATCH = {"forehand": "FH", "backhand": "BH"}
YAW_XCHECK_TOL_DEG = 0.01


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True, help="fivebind BINDINGS.json")
    parser.add_argument("--clips-dir", required=True,
                        help="dir holding the four *_upper_fivebind.npz (bytes must match BINDINGS)")
    parser.add_argument("--v-in-json", required=True,
                        help="anchor inverse-solve JSON (per-clip v_in_centroid/v_out_nominal + provenance)")
    parser.add_argument("--npz-prefix", default="motions/fivebind_n5_20260728",
                        help="repo-relative prefix recorded as the unit npz path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bindings_path = Path(args.bindings)
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    vin_path = Path(args.v_in_json)
    vin_doc = json.loads(vin_path.read_text(encoding="utf-8"))
    clips_dir = Path(args.clips_dir)

    units = []
    for name in CLIPS:
        row = bindings["clips"][name]
        if not row["usable"]:
            raise SystemExit(f"{name}: BINDINGS marks the clip unusable")
        verification = row["verification"]
        npz_path = clips_dir / Path(row["clip_path"]).name
        actual_sha = _sha256_file(npz_path)
        if actual_sha != row["clip_sha256"]:
            raise SystemExit(
                f"{name}: clip bytes drifted from BINDINGS "
                f"(expected {row['clip_sha256']}, got {actual_sha})"
            )
        data = np.load(str(npz_path), allow_pickle=False)
        quat0 = np.asarray(data["body_quat_w"], dtype=np.float32)[0, 0].astype(np.float64)
        root0 = np.asarray(data["body_pos_w"], dtype=np.float32)[0, 0].astype(np.float64)
        w, x, y, z = quat0
        yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        yaw_bindings = float(row["rotation"]["pelvis_yaw_after_deg"])
        if abs(yaw_deg - yaw_bindings) > YAW_XCHECK_TOL_DEG:
            raise SystemExit(
                f"{name}: frame-0 root yaw {yaw_deg:.6f} deg disagrees with "
                f"BINDINGS pelvis_yaw_after_deg {yaw_bindings:.6f}"
            )
        total = int(np.asarray(data["joint_pos"]).shape[0])
        if total != int(row["T"]):
            raise SystemExit(f"{name}: npz T={total} != BINDINGS T={row['T']}")

        solve = vin_doc["clips"][name]
        strike_env = [float(v) for v in verification["strike_point_env_frame_m"]]
        family = FAMILY_TO_BATCH[row["family"]]
        units.append({
            "uid": name,
            "family": family,
            "npz": f"{args.npz_prefix.rstrip('/')}/{npz_path.name}",
            "npz_sha256": actual_sha,
            "T": total,
            "fps": FPS,
            "hit_frame_50": int(row["strike_frame"]),
            "strike_phase": float(row["strike_phase"]),
            "yaw_before_deg": yaw_deg,
            "station_xy_hope_m": [float(root0[0]) - TAB_NEAR_X, float(root0[1]) - TAB_HALF_W],
            "ball_pos_hit_hope_m": [
                strike_env[0] - TAB_NEAR_X,
                strike_env[1] - TAB_HALF_W,
                strike_env[2] - TAB_Z,
            ],
            "v_in_fit_hope_ms": [float(v) for v in solve["v_in_centroid"]],
            "v_out_fit_hope_ms": [float(v) for v in solve["v_out_nominal"]],
            "w_out_nominal_radps": [float(v) for v in solve["w_out_nominal"]],
            "world_z0": "floor",
            "station_provenance": (
                "npz frame-0 pelvis xy (shared canonical-ready station; BINDINGS has no "
                "station field) minus the env->hope table translation (0.5, 0.7625)"
            ),
            "v_in_provenance": (
                "inverse-solved from anchor blade state: centroid of the venue incoming-ball "
                "distribution subset returned legally by the BINDINGS anchor blade state "
                "(strike_point/strike_velocity/signed_face_normal) under the venue contact "
                "law; see v_in_json provenance"
            ),
            "anchor_exact_bindings": float(verification["anchor_exact_return_rate"]),
            "anchor_exact_inverse_solve_check": float(solve["anchor_exact_check"]),
            "blade_speed_ctrl_point_mps": float(verification["blade_speed_mps"]),
            "mount_normal_sign_bindings": float(row["mount_normal_sign"]),
        })

    document = {
        "schema": "fivebind_n5_batch_table_v1 (chingmu_manifest_v1-compatible units)",
        "source_bindings": str(bindings_path),
        "source_bindings_sha256": _sha256_file(bindings_path),
        "v_in_json": str(vin_path),
        "v_in_json_sha256": _sha256_file(vin_path),
        "frame_note": (
            "fivebind clip world frame == training env W frame (table near edge x=0.5, "
            "centred y=0, surface z=0.76). hope frame here := env frame translated by "
            "(-0.5, -0.7625) in xy; contact z additionally -0.76 (above table surface). "
            "Vectors (v_in/v_out) are translation-invariant and identical in both frames."
        ),
        "units": units,
    }
    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {out_path}")
    for unit in units:
        print(
            f"  {unit['uid']:14s} {unit['family']} T={unit['T']} hit={unit['hit_frame_50']} "
            f"yaw0={unit['yaw_before_deg']:9.4f} deg |v_in|="
            f"{math.sqrt(sum(v * v for v in unit['v_in_fit_hope_ms'])):.4f} m/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
