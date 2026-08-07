#!/usr/bin/env python3
"""把 measured clip 的某一帧「坐到地面上」——只动十二个腿关节,别的一个数不动。

人话:重定向出来的 clip 每一帧两只脚都悬在空中约 `1 cm`。原因不在动捕、也不在这条 clip,
在重定向那一步的脚部目标(见 §5.6.7「十」):GMR 的 `smplx_to_a3.json` 把
`left/right_ankle_roll_Link` 的 **body 原点**直接拉到人体**踝关节中心**,
横向偏移 `±0.02 m`、**竖直偏移 0**,而 `ground_height` 也写着 `0.0`。
参与者的踝关节中心离地板约 `97 mm`,A3 的踝原点在鞋底放平时只有 `67.46 mm` ——
**两者差的那三厘米没有任何一项去补**,于是整条库的脚都吊在空中,脚面一倾斜再吃掉一部分,
剩下的就是实测的 `3.96 .. 19.59 mm`。

所以「接地」的正确方向不是把整个人往下压(那会把骨盆和拍子一起拽下去,等于毁掉动作),
而是**把脚放回地面**:骨盆(root)和 `19` 个非腿关节逐位不动,只重解 `12` 个腿关节,
让两块**碰撞**鞋底平贴到地板、各自的水平位置与朝向(yaw)保持原值。
这正是仓库已有的 `canonical_grounded_ready.solve_g1_donor_root`(G1),
出生姿态那一侧每天都在跑它;这条链路只是从来没跑过。

由此得到一条按构造成立的性质:**root 和非腿关节一个 bit 都没变,所以从骨盆往上
(腰、双臂、拍子)的世界坐标逐位不变**,本工具会把它当成一条判据实测出来,而不是当成假设。

本工具是**未授权诊断**:只出报告,不写任何 artifact、不改任何门限、不授权训练或上机。

用法(pod,需要 mujoco + scipy;不需要 GPU):
    CUDA_VISIBLE_DEVICES= /workspace/hope_isaac_venv/bin/python \\
        hope_training/whole_body_tracking/scripts/ground_measured_clip_to_floor.py \\
        --motion assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz \\
        --frames all --json out.json
    ... --library assets/motions/chingmu73_measured_v4_20260803 --frames 0 --json sweep.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REPO_ROOT = SCRIPTS_DIR.parents[2]
DEFAULT_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong"
    / "a3_pingpong.xml"
)

# 拍子在 MJCF 里的 site 名。末端轨迹的「没被破坏」就是拿它逐帧比出来的。
RACKET_SITE_NAME = "right_racket"


class ClipGroundingError(RuntimeError):
    """本工具的任何 fail-closed 拒绝。"""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_quaternion_fixed_point(quat: np.ndarray) -> np.ndarray:
    """把四元数归一化到**归一化的不动点**,再交给 ``ReadyState``。

    为什么要这一步:``canonical_grounded_ready.ReadyState`` 每次构造都做一次
    ``q / ||q||``,而这个操作在浮点上**不是幂等的** —— 对 `Take_061_unit04_BH` 的
    `57` 帧实测,`15` 帧在第二次归一化时最低位就变了。G1 解完之后要用
    ``np.array_equal`` 逐位比对 root 有没有被动过,于是那些帧会被自己的护栏
    以 `G1 changed the donor root` 拒掉 —— 拒的是浮点尾数,不是解算器。

    这里先迭代到 ``q / ||q|| == q``(逐位)再送进去,双方就都在同一个不动点上。
    迭代不收敛就 fail-closed:宁可不出数,也不出一份"root 到底动没动"说不清的报告。
    """

    base = np.asarray(quat, np.float64).copy()
    norm = float(np.linalg.norm(base))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ClipGroundingError("clip root quaternion is not usable")
    # 有些四元数的归一化落进一个**两周期**而不是不动点(实测 `Take_061_unit04_BH`
    # 的 frame 26 就是),这时把输入按相对 `2^-49` 量级挪一点点再迭代 —— 挪动量
    # 比四元数本身的表示精度还小一个数量级,代表的旋转是同一个,但足以跳出周期。
    for attempt in range(64):
        out = base * (1.0 + attempt * 2.0**-49)
        for _ in range(8):
            nxt = out / float(np.linalg.norm(out))
            if np.array_equal(nxt, out):
                return out
            out = nxt
    raise ClipGroundingError(
        "clip root quaternion has no floating-point normalisation fixed point; "
        "refusing to report a root-preservation claim that cannot be checked bitwise"
    )


def _parse_frames(spec: str, frames: int) -> tuple[int, ...]:
    text = str(spec).strip().lower()
    if text == "all":
        return tuple(range(frames))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            low_text, _, high_text = part.partition("-")
            low, high = int(low_text), int(high_text)
            if low > high:
                raise ClipGroundingError(f"frame range {part!r} runs backwards")
            out.extend(range(low, high + 1))
        else:
            out.append(int(part))
    if not out:
        raise ClipGroundingError("no frames selected")
    for index in out:
        if index < 0 or index >= frames:
            raise ClipGroundingError(
                f"frame {index} is outside the {frames}-frame clip"
            )
    return tuple(dict.fromkeys(out))


class _Measurer:
    """在**同一个** exact plant 上量鞋底/站宽/末端,免得两边用不同的尺。"""

    def __init__(self, backend: Any, mujoco: Any) -> None:
        self._backend = backend
        self._mujoco = mujoco
        model = backend.model
        self._model = model
        self._data = mujoco.MjData(model)
        self._qpos_adr = np.asarray(backend._binding.joint_qpos_adrs, np.int64)
        self._foot_bodies = tuple(backend._foot_bodies)
        self._foot_geoms = tuple(backend._foot_collision_geoms)
        self._racket_site = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RACKET_SITE_NAME)
        )
        if self._racket_site < 0:
            raise ClipGroundingError(
                f"exact model has no {RACKET_SITE_NAME!r} site; refusing to claim "
                "the racket trajectory survived without being able to measure it"
            )
        self.body_names = [
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index))
            for index in range(int(model.nbody))
        ]

    def _install(self, state: Any) -> None:
        data = self._data
        data.qpos[:] = np.asarray(self._model.qpos0, np.float64)
        data.qpos[0:3] = state.root_pos_w
        data.qpos[3:7] = state.root_quat_wxyz
        data.qpos[self._qpos_adr] = state.joint_pos
        data.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, self._data)

    def _sole_vertices_world(self, geom: int) -> np.ndarray:
        model = self._model
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = np.asarray(model.mesh_vert[start : start + count], np.float64)
        rotation = np.asarray(self._data.geom_xmat[geom], np.float64).reshape(3, 3)
        return local @ rotation.T + np.asarray(
            self._data.geom_xpos[geom], np.float64
        )

    def measure(self, state: Any) -> dict[str, Any]:
        self._install(state)
        data = self._data
        soles = [self._sole_vertices_world(geom) for geom in self._foot_geoms]
        ankles = [
            np.asarray(data.xpos[body], np.float64) for body in self._foot_bodies
        ]
        tilts = []
        for body in self._foot_bodies:
            rotation = np.asarray(data.xmat[body], np.float64).reshape(3, 3)
            tilts.append(
                float(np.degrees(np.arccos(float(np.clip(rotation[2, 2], -1.0, 1.0)))))
            )
        self._mujoco.mj_comPos(self._model, data)
        com = np.asarray(data.subtree_com[0], np.float64)
        return {
            "sole_lowest_z_m": [float(rows[:, 2].min()) for rows in soles],
            "ankle_origin_z_m": [float(row[2]) for row in ankles],
            "foot_tilt_deg": tilts,
            "stance_width_m": float(np.linalg.norm(ankles[0][:2] - ankles[1][:2])),
            "com_w_m": com.tolist(),
            "com_two_sole_footprint_margin_m": _footprint_margin(com, soles),
            "racket_site_pos_w_m": np.asarray(
                data.site_xpos[self._racket_site], np.float64
            ).tolist(),
            "body_pos_w_m": np.asarray(data.xpos, np.float64).copy(),
            "mujoco_floor_contacts": int(data.ncon),
        }


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(points, 9), axis=0)
    if unique.shape[0] < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def _chain(rows: np.ndarray) -> list[np.ndarray]:
        stack: list[np.ndarray] = []
        for point in rows:
            while len(stack) >= 2:
                cross = np.cross(stack[-1] - stack[-2], point - stack[-2])
                if cross > 0.0:
                    break
                stack.pop()
            stack.append(point)
        return stack

    lower = _chain(ordered)[:-1]
    upper = _chain(ordered[::-1])[:-1]
    return np.asarray(lower + upper)


def _hull_margin(point: np.ndarray, hull: np.ndarray) -> float:
    if hull.shape[0] < 3:
        return float("-inf")
    best = float("inf")
    count = hull.shape[0]
    for index in range(count):
        a = hull[index]
        edge = hull[(index + 1) % count] - a
        length = float(np.linalg.norm(edge))
        if length < 1.0e-12:
            continue
        outward = np.array([edge[1], -edge[0]], dtype=np.float64) / length
        best = min(best, -float(np.dot(point - a, outward)))
    return best


def _footprint_margin(com: np.ndarray, soles: Sequence[np.ndarray]) -> float | None:
    """质心对**两只完整碰撞鞋底**地面投影的有符号裕度。

    这不是准入门,是给那道门配的对照读数:门用的是 MuJoCo 实际吐出来的接触点,
    而 MuJoCo 的 plane-mesh 碰撞每只脚只返回三四个代表点(实测每只脚 `3` 个,
    凸包 `55.1 cm²`,而这只鞋底真正的投影是 `284.8 cm²`)。两个数一起报,
    「支撑面外」到底是机器人的事还是取样的事,读收据的人自己能看出来。
    """

    hull = _convex_hull_2d(np.vstack(list(soles))[:, :2])
    value = _hull_margin(np.asarray(com, np.float64)[:2], hull)
    return None if not np.isfinite(value) else float(value)


GATE_NAMES = (
    "sole_floor",
    "double_support",
    "joint_limits",
    "collision",
    "foot_pose",
    "leg_to_foot_jacobian",
    "support_margin",
)


def _gate_table(receipt: Any) -> dict[str, Any]:
    """直接读收据自己的 ``gates`` 表,不另建一份手抄的判定。"""

    gates = receipt["gates"]
    table = {name: str(gates[name]) for name in GATE_NAMES}
    table["exact_model_identity"] = str(gates["exact_model_identity"])
    table["static_ground_dynamics"] = str(gates["static_ground_dynamics"])
    return table


def _audit_state(
    state: Any,
    *,
    grounded: Any,
    backend: Any,
    identity: Any,
    config: Any,
    label: str,
) -> dict[str, Any]:
    """跑仓库自己的九门,目标脚位取该状态自己的脚位(所以 foot_pose 恒过)。

    这样「接地前」和「接地后」问的是同一批门,唯一的区别就是这一帧本身。
    """

    result = grounded._audit_and_build_result(
        label,
        state,
        backend.foot_poses(state),
        source={"mode": "as_is_audit_no_solve"},
        backend=backend,
        expected_model_identity=identity,
        config=config,
        _return_draft=True,
    )
    return dict(result.receipt_payload)


def ground_clip(
    *,
    motion: Path,
    frames_spec: str,
    grounded: Any,
    backend: Any,
    identity: Any,
    measurer: _Measurer,
    config: Any,
) -> dict[str, Any]:
    payload = np.load(motion)
    for key in ("joint_pos", "body_pos_w", "body_quat_w"):
        if key not in payload.files:
            raise ClipGroundingError(f"motion npz is missing {key!r}")
    joint_pos = np.asarray(payload["joint_pos"], np.float64)
    root_pos = np.asarray(payload["body_pos_w"], np.float64)[:, 0]
    root_quat = np.asarray(payload["body_quat_w"], np.float64)[:, 0]
    total = int(joint_pos.shape[0])
    if joint_pos.shape[1] != len(backend.joint_names):
        raise ClipGroundingError("clip joint width does not match the runtime plant")
    selected = _parse_frames(frames_spec, total)

    leg = set(str(name) for name in grounded.LEG_JOINT_NAMES)
    names = [str(name) for name in backend.joint_names]
    leg_index = [i for i, name in enumerate(names) if name in leg]
    nonleg_index = [i for i, name in enumerate(names) if name not in leg]

    rows: list[dict[str, Any]] = []
    for index in selected:
        donor = grounded.ReadyState(
            joint_pos[index],
            root_pos[index],
            _unit_quaternion_fixed_point(root_quat[index]),
        )
        before_geometry = measurer.measure(donor)
        before_receipt = _audit_state(
            donor,
            grounded=grounded,
            backend=backend,
            identity=identity,
            config=config,
            label="ungrounded-measured-frame",
        )
        before = _describe(
            donor,
            donor=donor,
            receipt=before_receipt,
            before_geometry=before_geometry,
            geometry=before_geometry,
            measurer=measurer,
            names=names,
            leg_index=leg_index,
            nonleg_index=nonleg_index,
        )
        # 「接地前」是拿自己跟自己比,那几项位移恒为 0,留着会读成一句主张。删掉。
        for key in (
            "racket_site_shift_mm",
            "body_shift_mm",
            "leg_joint_delta_deg",
            "leg_joint_max_abs_delta_deg",
            "nonleg_joint_max_abs_delta_rad",
            "root_bitwise_preserved",
        ):
            before.pop(key, None)
        row: dict[str, Any] = {"frame": index, "before": before}
        for stage, solver in (
            ("G1", grounded.solve_g1_donor_root),
            ("G1S", grounded.solve_g1_support_edge_projection),
        ):
            try:
                solved = solver(
                    donor,
                    backend=backend,
                    expected_model_identity=identity,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                row[stage] = {
                    "status": "SOLVER_REFUSED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                continue
            described = _describe(
                solved.state,
                donor=donor,
                receipt=dict(solved.receipt),
                before_geometry=before_geometry,
                geometry=measurer.measure(solved.state),
                measurer=measurer,
                names=names,
                leg_index=leg_index,
                nonleg_index=nonleg_index,
            )
            described["status"] = "SOLVED"
            if stage == "G1S":
                described["common_foot_target_shift_w_mm"] = [
                    value * 1000.0
                    for value in solved.receipt["source"][
                        "final_common_target_shift_w_m"
                    ]
                ]
                described["projection_iterations"] = len(
                    solved.receipt["source"]["projection_attempts"]
                )
            row[stage] = described
            # G1 already clears every gate -> the projection variant is not needed;
            # record that fact instead of paying for a change nobody asked for.
            if stage == "G1" and all(
                described["gates"][name] == "PASS" for name in GATE_NAMES
            ):
                row["G1S"] = {
                    "status": "NOT_REQUIRED",
                    "reason": "G1 already passes every static geometry gate",
                }
                break
        rows.append(row)

    return {
        "motion": str(motion),
        "motion_sha256": _file_sha256(motion),
        "frames_total": total,
        "frames_audited": list(selected),
        "rows": rows,
    }


def _describe(
    state: Any,
    *,
    donor: Any,
    receipt: Any,
    before_geometry: dict[str, Any],
    geometry: dict[str, Any],
    measurer: _Measurer,
    names: Sequence[str],
    leg_index: Sequence[int],
    nonleg_index: Sequence[int],
) -> dict[str, Any]:
    """一帧的完整读数:九门 + 几何 + 逐关节改动 + 末端有没有被动过。"""

    delta = np.asarray(state.joint_pos, np.float64) - np.asarray(
        donor.joint_pos, np.float64
    )
    body_delta = np.linalg.norm(
        geometry["body_pos_w_m"] - before_geometry["body_pos_w_m"], axis=1
    )
    racket_delta = float(
        np.linalg.norm(
            np.asarray(geometry["racket_site_pos_w_m"])
            - np.asarray(before_geometry["racket_site_pos_w_m"])
        )
    )
    footprint = geometry["com_two_sole_footprint_margin_m"]
    return {
        "verdict": receipt["verdict"],
        "gates": _gate_table(receipt),
        "sole_lowest_z_mm": [v * 1000.0 for v in geometry["sole_lowest_z_m"]],
        "ankle_origin_z_mm": [v * 1000.0 for v in geometry["ankle_origin_z_m"]],
        "foot_tilt_deg": geometry["foot_tilt_deg"],
        "stance_width_m": geometry["stance_width_m"],
        "gate_support_margin_mm": _margin_mm(receipt),
        "two_sole_footprint_margin_mm": (
            None if footprint is None else footprint * 1000.0
        ),
        "mujoco_floor_contacts": geometry["mujoco_floor_contacts"],
        "root_bitwise_preserved": bool(
            np.array_equal(state.root_pos_w, donor.root_pos_w)
            and np.array_equal(state.root_quat_wxyz, donor.root_quat_wxyz)
        ),
        "nonleg_joint_max_abs_delta_rad": float(np.abs(delta[nonleg_index]).max()),
        "leg_joint_delta_deg": {
            names[i]: float(np.degrees(delta[i])) for i in leg_index
        },
        "leg_joint_max_abs_delta_deg": float(
            np.degrees(np.abs(delta[leg_index]).max())
        ),
        "racket_site_shift_mm": racket_delta * 1000.0,
        "body_shift_mm": {
            "max_over_all_bodies": float(body_delta.max() * 1000.0),
            "max_over_pelvis_and_upper_bodies": float(
                max(
                    (
                        body_delta[i]
                        for i, name in enumerate(measurer.body_names)
                        if not _is_leg_body(name)
                    ),
                    default=0.0,
                )
                * 1000.0
            ),
            "worst_body": measurer.body_names[int(np.argmax(body_delta))],
        },
    }


def _is_leg_body(name: str) -> bool:
    return any(
        token in name
        for token in ("hip_", "knee_", "ankle_")
    )


def _margin_mm(receipt: Any) -> float | None:
    value = receipt["static_geometry"]["support"]["margin_m"]
    return None if value is None else float(value) * 1000.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=None)
    parser.add_argument("--library", type=Path, default=None)
    parser.add_argument("--frames", default="0")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    if (args.motion is None) == (args.library is None):
        raise ClipGroundingError("give exactly one of --motion / --library")

    import mujoco  # noqa: PLC0415

    import canonical_grounded_ready as grounded  # noqa: PLC0415
    import materialize_a3_dynamic_ready_contract as dynamic  # noqa: PLC0415

    mjcf = Path(args.mjcf).expanduser().resolve()
    identity = dynamic._derive_exact_model_identity(
        mjcf_path=mjcf, mjcf_sha256=_file_sha256(mjcf)
    )
    backend = grounded.MujocoGroundedReadyBackend.load(identity)
    measurer = _Measurer(backend, mujoco)
    config = grounded.GroundedReadyConfig()

    if args.motion is not None:
        clips = [Path(args.motion).expanduser().resolve()]
    else:
        clips = sorted(Path(args.library).expanduser().resolve().glob("hope_*.npz"))
        if not clips:
            raise ClipGroundingError("library contains no hope_*.npz clips")

    reports = []
    for clip in clips:
        reports.append(
            ground_clip(
                motion=clip,
                frames_spec=args.frames,
                grounded=grounded,
                backend=backend,
                identity=identity,
                measurer=measurer,
                config=config,
            )
        )
        print(f"[{len(reports)}/{len(clips)}] {clip.name}", file=sys.stderr, flush=True)

    report = {
        "kind": "measured_clip_floor_grounding_report_v1",
        "schema_version": 1,
        "diagnostic_unauthorized": True,
        "training_authorized": False,
        "hardware_authorized": False,
        "semantics": (
            "per frame: keep the donor root and all 19 non-leg joints bitwise fixed "
            "and re-solve the 12 leg joints so both COLLISION soles sit flat on the "
            "floor at the configured contact preload.  G1 "
            "(canonical_grounded_ready.solve_g1_donor_root) also keeps each foot's "
            "floor-plane position and yaw; G1S "
            "(canonical_grounded_ready.solve_g1_support_edge_projection) is only "
            "reported when G1 leaves a static geometry gate red, and adds one common "
            "floor-plane translation of both foot targets derived from the limiting "
            "support edge"
        ),
        "mjcf": str(mjcf),
        "mjcf_sha256": identity.mjcf_sha256,
        "target_contact_preload_m": config.target_contact_preload_m,
        "clips": reports,
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.json is not None:
        Path(args.json).write_text(text + "\n")
    else:
        print(text)

    rows = [row for clip in reports for row in clip["rows"]]
    print("\n人话摘要:", file=sys.stderr)
    print(f"  帧数:{len(rows)}", file=sys.stderr)
    for stage in ("G1", "G1S"):
        solved = [row for row in rows if row.get(stage, {}).get("status") == "SOLVED"]
        skipped = [
            row for row in rows if row.get(stage, {}).get("status") == "NOT_REQUIRED"
        ]
        print(
            f"  -- {stage}:解出 {len(solved)},"
            f"「不需要」{len(skipped)},"
            f"拒绝 {len(rows) - len(solved) - len(skipped)}",
            file=sys.stderr,
        )
        if not solved:
            continue
        for gate in GATE_NAMES:
            before = sum(1 for row in solved if row["before"]["gates"][gate] == "PASS")
            after = sum(1 for row in solved if row[stage]["gates"][gate] == "PASS")
            print(
                f"     {gate:22s} 接地前 {before}/{len(solved)}"
                f" -> 接地后 {after}/{len(solved)}",
                file=sys.stderr,
            )
        racket = np.array([row[stage]["racket_site_shift_mm"] for row in solved])
        upper = np.array(
            [
                row[stage]["body_shift_mm"]["max_over_pelvis_and_upper_bodies"]
                for row in solved
            ]
        )
        leg = np.array([row[stage]["leg_joint_max_abs_delta_deg"] for row in solved])
        print(
            f"     拍子位移 最大 {racket.max():.3e} mm;"
            f"骨盆及以上所有 body 位移 最大 {upper.max():.3e} mm;"
            f"腿关节改动 最大 {leg.max():.3f} 度",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
