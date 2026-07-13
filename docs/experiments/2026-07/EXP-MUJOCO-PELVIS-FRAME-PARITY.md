# EXP-MUJOCO-PELVIS-FRAME-PARITY — MuJoCo 倒下是否来自 pelvis 坐标错位？

- 状态：completed
- 工作类型：forensic + source correction
- 阶段/轴：Isaac→MuJoCo parity / reset point 与 actor gyro frame
- 人类负责人：yikang
- 执行者：Codex
- 工作分支：`codex/mujoco-com-reset-frame`
- 基线：`origin/main@88c86293d232`
- 最高证据等级：E2（真实 A3 MJCF CPU smoke；无 policy behavior）

## 问题和边界

要区分三个命题：

1. URDF/MJCF 是否存在会让机器人静态站立就倒的四元数、重力轴、关节轴或关节顺序硬错；
2. Python MuJoCo evaluator 是否把 pelvis 的速度点或表达轴喂错；
3. 已发现的 source mismatch 是否足以解释当前跨引擎击球差距。

本实验不改 checkpoint、动作、Reward、PD、接触参数、正式题表或 vendor backend，也不运行 Pod、
PPO 或真机。ready-state 因果四格已有独立预注册，本实验不重复运行或认领。

## 只读取证

- URDF/MJCF 的 31 个活动关节逐名一致；关节轴无符号差，限位最大差约 `4.13e-6 rad`。
- evaluator 的 `wxyz`、`R=world<-body`、Z-up、`R^T*[0,0,-1]` projected gravity 和按名字
  建立的 qpos/qvel/actuator permutation 未发现硬错。
- 当前 model-4000 K100 记录为 `physical_falls=0`、`guard_resets=100`；它的问题是正手位置误差
  Isaac `0.024775 m` 对 MuJoCo `0.131535 m`，差 `0.106759 m`，不是“一进 MuJoCo 就物理倒”。
- 正式初态仍不相同：Isaac pelvis x 为 `0`，vendor named stand 为 `-0.0416378 m`；这是已登记、
  未运行的 causal hypothesis，不能从 strike error 中直接相减。

## 找到并修正的 source mismatch

1. Exact motion 的 pelvis position 是 link origin，但 linear velocity 是该 pelvis rigid body 的
   COM 世界速度。旧 evaluator 把 COM 速度直接写入 link-origin freejoint translation。修正为：

   ```text
   v_origin^W = v_com^W - omega^W x (R_WB * body_ipos[pelvis])
   ```

   A3 pelvis offset 约 `0.1273 m`。checkpoint-bound schema-3 native/standalone export 用
   `motion_body_lin_vel_points` 逐 clip 绑定 COM/link-origin：COM 转换，显式 link-origin 直写并
   保持 exact-ineligible。旧 exact schema-2 包按历史合同窄兼容为 all-COM；旧 inexact/missing
   aggregate metadata 无法证明属于哪个点，teacher-reference reset 直接 FATAL，不再猜。
2. `base_ang_vel` 原用 `mjOBJ_BODY/local`，实际表达在 compiled inertia-principal axes；策略和
   projected gravity 要 pelvis link/IMU axes。改用 `mjOBJ_XBODY/local`。两轴约差 `0.3315 deg`，
   是每个 actor step 的真实错位，但量级不足以单独证明 10.68 cm strike gap 的因果。
3. evaluator 现在 fail-loud 要求 `pelvis_link` 自身恰好一个 freejoint，且 qpos/qvel 地址都从 0
   开始；球等其他 body 的 freejoint 不受禁止。

另有一个未并入本修复的 vendor ROS blocker：非零 `SimReset` 发布语义是 odom/world 的
link-origin twist，subscriber 却把 world angular velocity 原样写入 body-local qvel。当前正式和
脚本 keyframe 都写全零 twist，不触发该 bug；它需另立 G04/G07 接口 ticket。

## 可复现运行

```bash
/Users/yyk956614/anaconda3/envs/backend/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_motion_kinematics_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_align_flags.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_reference_reset_com_frame.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  tests/test_view_a3_stand.py

/Users/yyk956614/anaconda3/envs/backend/bin/python -m pytest -q tests

/Users/yyk956614/anaconda3/envs/backend/bin/python \
  scripts/view_a3_stand.py --check --duration 10 --print-every 1
```

结果：上述相关合同 `96 passed, 0 skipped`；formal CPU group `115 passed, 0 skipped`；完整合同
union `183 passed, 0 skipped`；支持的根目录 suite `554 passed`；10 秒 plain-MuJoCo PD stand
`diagnostic_pass=true`、状态 finite、pelvis-z 最大漂移 `1.816 mm`、最大倾角 `0.311 deg`、
双脚接触率 `100%`。仓库无范围 `pytest -q` 仍会遇到两个历史同名模块收集冲突和 backend 环境缺
`hydra` 的两个收集错误；本 ticket 的范围套件均已通过。

两轮独立 review 均未发现 P0/P1/P2：复核了 MuJoCo `BODY=xipos/ximat`、
`XBODY=xpos/xmat`、freejoint linear/world-link-origin 与 angular/body-local 语义，以及 mixed
exact claim、point-count、其他 free body 和旧 standalone donor 负控。

2026-07-13 10:37 UTC 对用户给出的两个 Pod 做了只读复核：第一个 endpoint 两次在 SSH key
exchange 阶段被对端 reset；第二个 endpoint 的 3 张 RTX 5090 均为 `0 MiB / 0%`，没有 train、
Isaac 或 MuJoCo evaluator 进程。其 `/workspace/franco/nohope` 为 `16a94b1`，本地缓存的
`origin/main` 为 `7b85546`，均早于本分支基线 `88c8629`；未执行 `fetch`，也未修改远端状态。
因此没有一台现存 Pod 可以作为修复后 rollout 证据。

## 决定

- **adopt** Python evaluator 的 COM→origin、gyro link-frame 和 freejoint fail-loud 修复。
- **inconclusive** 对“frame mismatch 是当前跨引擎 strike gap 主因”的判断：静态站立反证了 gross
  frame failure，两个 source bug 已闭合，但正式 ready-state 四格、plant/contact/effort 与修后 K100
  尚未运行。
- formal `stand-keyframe` K100 的 reset qvel 为零，因此 COM reset 修复不改变它；gyro 修复会改变
  每个 actor observation。合并后必须用同一 immutable K100 重跑，不能复用旧分数。

权威接口与 Gate：[policy observation](../../interfaces/policy_observation_action.md)、
[frames](../../interfaces/frames_and_coordinates.md)、[G04](../../gates/G04_sim_modeling_mujoco_isaac.md)、
[G06](../../gates/G06_isaac_to_mujoco.md)、[G07](../../gates/G07_mujoco_to_real.md)。
