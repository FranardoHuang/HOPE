# G03 Data Processing And Physics Calibration

Status: Partial

**2026-08-02 measured-racket 更正与实修（状态不变）：**最终 motion teacher 必须来自
实测 racket，不能把 `q_retarget -> right_racket FK` 自洽地当作真值。之前“ChingMu
raw racket 未恢复/sidecar 只有 ball”的判断错误：本机 ChingMu 原件有41组人体/拍子/桌
BVH 和26组球 BVH，Pod 对74个源 unit 具备完整 unit NPZ/JSON/retarget PKL；73库明确排除
`Take_085_unit00_FH`。unit channel 含 `paddle_blade_hope_m`/`paddle_butt_hope_m`/
`paddle_normal_hope`，hit JSON 含 signed `face_normal_hope`。

独立复核证明旧 schema-v3 把 site-local `+X` 误当球拍长轴；URDF/MJCF rigid-mesh
ground truth 是 `(local +X + local +Z)/sqrt(2)`。旧 v3 因此 revoked。本地 schema-v4 sibling
`assets/motions/chingmu73_measured_v4_20260803` 已完成 solver/materializer/独立 FK audit `73/73`，
50 Hz 共 `5107` 帧；hit 最坏 position/face/long-axis/SO(3)=
`.879 mm/.174 deg/.126 deg/.197 deg`。bank receipt SHA=`e6f0283f…b727a82`。

Gate 仍为 `Partial`：上面的 `73/73` 只表示 measured-racket **运动学/FK 对齐**通过，不能推出
机械准入。新增 fail-closed 工具
`hope_training/whole_body_tracking/scripts/audit_measured_racket_mechanical_admission.py`
对 v4 的 `73/73` 动作、`5107` 帧逐样本检查 URDF joint position、stored/finite-difference
velocity，并计算 finite-difference acceleration。真实结果是
`actions_mechanically_admitted=0/73`：仅 `16/73`（BH `9/59`、FH `7/14`）通过当前可判定的
position+velocity 门，另外 `57/73` 有观测到的 hard failure。即使那 16 条也仍缺权威
acceleration limits、实际 torque-speed curves 和逐帧 inverse-dynamics torque，故全部保持
`diagnostic_unauthorized=true`，不得提升为 training-ready teacher、部署或真机资产。

复现命令（有 hard failure 时退出码为 `2`）：

```bash
python hope_training/whole_body_tracking/scripts/audit_measured_racket_mechanical_admission.py \
  --bank assets/motions/chingmu73_measured_v4_20260803 \
  --output-json /tmp/chingmu73_mechanical.json
```

下一步应先取得 authority acceleration limits 与 actuator torque-speed curves，生成逐帧
inverse-dynamics torque，再用同一全库审计重算并补 reference-tracking rollout；不能用 URDF 的独立
effort/velocity 上限拼成矩形可行域。source-capsule/compiler 还需无损传递 schema-v4 measured
channels，prototype 仍缺 `velocity_contract`，且还需 content-bound marker→official-site 原始生成收据。
真实 mass/CoM/inertia 依然单列为 racket physics calibration。
恢复路径见
[setup_local_sync](../operations/setup_local_sync.md)，几何合同见
[racket_contact_geometry](../interfaces/racket_contact_geometry.md)。

## Goal

Turn raw real-world ball data into calibrated physical parameters and testable planner inputs.

This gate supports real-to-sim and planner validation.

## Inputs

- Raw bag data from G02.
- Processed CSV trajectories.
- HITTER-compatible ball flight and bounce assumptions.

## Outputs

- Clean segmented trajectories.
- Calibrated drag and restitution parameters.
- Planner tests and regression data.
- Known limitations for spin, bounce, and outlier handling.

## Related Directories

- `calib_bags/`
- `calib_csv/`
- `hope_ws/src/hope_planner`
- `HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md`

## Operation Docs

- [../operations/build_and_test.md](../operations/build_and_test.md)
- [../operations/run_planner.md](../operations/run_planner.md)

## Acceptance Criteria

- Raw bag to CSV conversion is reproducible.
- Trajectory segmentation is reproducible.
- Calibration procedure records fitted constants and units.
- Planner tests cover state estimation, trajectory prediction, racket target planning, and calibration utilities.

## Current State

Done:

- `hope_planner` tests pass from the repo root with `PYTHONPATH=hope_ws/src/hope_planner`: 26 tests on 2026-06-26.
- Existing tests cover ball state estimation, trajectory prediction, calibration, quaternion utilities, racket target planning, and CSV splitting.
- Processed calibration CSVs and chunk manifests exist.
- Stage 3 racket target planning now uses the same quadratic-drag-plus-gravity free-flight model as Stage 2 for outgoing velocity shooting and net-clearance interpolation, instead of solving outgoing returns ballistically while inbound prediction used drag.
- Pure Python planner tests now cover opponent-facing racket normals, degenerate/sideways normal cases, drag-aware outgoing landing, bounce-then-cross hit-plane prediction, and quaternion local-`+x` alignment.

Done (2026-07-03, venue ball-physics fit v1):

- **Accepted physical constants are now recorded in `configs/ball_physics_venue.yaml`**
  (flight k_d/k_m, table/paddle contact blocks; every value carries provenance, CI, and
  the (speed, SR, v_n) validity envelope). Methodology + F1–F8 falsification verdicts:
  `docs/ball_physics_fit_report.md`; pipeline of record: `hope_training/ball_physics_fit/`.
- Spin is measured (ball quaternion channel, scale validated aerodynamically, ≤15 rev/s
  coverage) and modeled in flight (Magnus) and at contacts (tangential-impulse spin
  equation). The paddle block got its FIRST real-racket fit (150 strikes); paddle
  restitution is velocity-dependent (F4 KILL — consume the yaml `e_exp_*` keys).
- Outlier rejection and QA gates are implemented and documented in the pipeline
  (Stage 0: sampling / units-frame / gravity magnitude+tilt gates; robust losses and
  quality gates throughout; `test_oracle_present.py` fails loudly when data is absent).

Not done:

- Double-bounce behavior is not modeled (landing predictor is first-bounce only).
- Magnus saturation above SR ≈ 0.7 is unvalidated — the venue data never reaches it
  (F2 inconclusive by coverage); needs a dedicated high-spin capture.
- The venue rig's 9 mm noise leaves the table tangential block at the v0 values and
  paddle a_t identified only through the velocity channel (0.52, CI [0.46, 0.61]);
  Stage-4 absolute landing bars not met against observed/terminal-window ground
  truth (through-paddle median ≈ 0.25 m on full coverage n=82, vs the 0.10 m
  target — error budget dominated by paddle model form + racket-state accuracy;
  flight model is fine: 67 mm at contact from measured out-state, 5 mm at ~100 ms
  before landing. See `predict_check.py` H0/H1/H2 decomposition).

## Risks

- HITTER assumes negligible spin; this can fail against skilled opponents.
- Bounce parameters fitted on one ball coating may not transfer.
- A model that looks good on curated samples can fail on live noisy mocap.

## Next Steps

1. Record the current fitted parameters and source dataset.
2. Add a small planner regression dataset if needed.
3. Define explicit tests for short balls, deep balls, spin, and double-bounce cases.
