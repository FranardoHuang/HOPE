# MuJoCo native fixed-tape / diagnostic VecEnv

这份工序同时保留底层 single-env fixed-tape 证据和当前 no-reward diagnostic
`VecEnv` 证据。single-env 部分只验证一件事：真实 vendor A3 MJCF 加五实体桌/网场景，
能否按 schema-3 的 31-D
`action -> episode-fixed delay -> affine qdes -> 5% soft-envelope interior 与 hard-inner 交集投影
-> total-PD` 合同确定性走完100个 control tick。

当前 native stack 已有 physical-ball core、76-D purpose-group observation、deterministic
batched reset、finite diagnostic rollout、physics-substep contact-event ledger、exact tape timeout 以及
`base_fell_tilt/base_too_low/joint_actual_forbidden` 的 exact termination subset。第三条在每个
control step 后按 runtime joint order 比较实际 `q` 与 MuJoCo `model.jnt_range`，边界容差按
Isaac raw hard-edge 语义固定为 `0`，并 sticky 保留同一 control tick 内任一 physics substep 的触边；非有限状态、非有限/倒置区间、`q <= lower` 或
`q >= upper` 均触发。三条使用 sticky latch，reason order 固定为 tilt、height、joint actual。
Isaac 配置和 termination callable 两份源码都以整文件 SHA-256 固定，任一漂移均 fail closed。
它仍没有被授权的 Reward、PPO、
checkpoint save/resume 或 export；正常 `VecEnv.step()` 在碰 physics 前就拒绝。所有输出都带
`diagnostic_unauthorized=true`；成功不能授权 canonical training、promotion、deployment 或真机命令。
当前 PASS 工序把两种状态分开：所选动作 NPZ/frame 0 只是 immutable teacher
reference；physical reset 必须来自再审计的 sealed hold candidate，history fill 与 probe
action 中心使用同一 LP hold qdes。不传 `--hold-candidate` 的 teacher-q0 reset 是第4节
保留的历史负对照，会重现安全 FAIL，不是当前主命令。
工具还会重验 NPZ 内的 joint-order contract ID/SHA 与
`configs/a3_joint_order_bijection_v1.json` 的当前字节一致；不允许只靠 31 列宽度猜测列语义。
每次读 tape 时还会重读 teacher NPZ、重算 SHA 和指定 frame，并验证 delay history fill
确实解码到 sealed LP hold qdes；tape 中的自声明 lineage 不能单独写入 receipt。

## 1. 生成不可变 fixed tape

从空 `/tmp` 开始时，先在仓库根目录生成 sealed physical-reset/hold candidate：

```bash
python -m hope_training.whole_body_tracking.mujoco_native.action_specific_hold \
  --contract configs/a3_vendor_runtime_authority_20260802_r8/bh_loop_c.shared_ready.training_contract.json \
  --teacher-motion assets/motions/chingmu_n1_take061u04_mechanical_candidate_v5_20260803/hope_Take_061_unit04_BH.measured_v5.npz \
  --teacher-frame 0 \
  --seed-dynamic-ready configs/a3_vendor_dynamic_ready_20260802_r8/bh_loop_c.dynamic_ready.v1.json \
  --expected-seed-sha256 3d604feb33145471b5dcc21279f26bc12e0351f0d158d7dc20dc8ed54517c306 \
  --output /tmp/take061_v5_action_hold_candidate.json
```

只有输出文件 SHA 为
`1930cc71df19960aa6c0470bdc0304f364e2a30532dbc415131f22e3079e1f32` 才继续生成 tape：

```bash
python -m hope_training.whole_body_tracking.mujoco_native.single_env make-tape \
  --contract configs/a3_vendor_runtime_authority_20260802_r8/bh_loop_c.shared_ready.training_contract.json \
  --teacher-motion assets/motions/chingmu_n1_take061u04_mechanical_candidate_v5_20260803/hope_Take_061_unit04_BH.measured_v5.npz \
  --teacher-frame 0 \
  --hold-candidate /tmp/take061_v5_action_hold_candidate.json \
  --expected-hold-candidate-sha256 1930cc71df19960aa6c0470bdc0304f364e2a30532dbc415131f22e3079e1f32 \
  --delay 0 \
  --tape /tmp/a3_mujoco_delay0_tape.json
```

该 v5 资产 SHA 必须是
`5899706b32cd60fa7b1e08094cd39d6aae59f4ed9b95c493026fb3b3dbb98101`。它的 NPZ 内部明确写有
`measured_racket_mechanical_admission=0` 和 `diagnostic_unauthorized=1`；即使本 runner 完成，也不得称为
canonical motion safety 或 N1 放行。

工具拒绝覆盖已有文件。标准输出记录 tape SHA、plant-binding SHA、teacher motion SHA、`100x31`
形状和 delay。
分别生成 delay `0/1/2` 时要使用三个不同输出路径。

## 2. 在带 MuJoCo Python 包的隔离环境运行

```bash
python -m hope_training.whole_body_tracking.mujoco_native.single_env run \
  --contract configs/a3_vendor_runtime_authority_20260802_r8/bh_loop_c.shared_ready.training_contract.json \
  --tape /tmp/a3_mujoco_delay0_tape.json \
  --trace /tmp/a3_mujoco_delay0_trace.npz \
  --receipt /tmp/a3_mujoco_delay0_receipt.json
```

receipt 必须同时满足：

- `status=DIAGNOSTIC_FIXED_TAPE_COMPLETE`；
- `counters.policy_ticks=100`、`counters.physics_substeps=400`；
- `reasons.fixed_tape_complete=1`；
- contract、binding、tape、vendor MJCF、augmented scene、table geometry 和 trace SHA 均非空；
- teacher lineage 精确绑定 `Take_061_unit04_BH`、motion SHA 和 frame 0；physical-reset
  lineage 另绑 hold-candidate file/content SHA 及其重算 semantics；
- delay histogram 只有本 episode 的一个固定 lag；
- safety 区保留 joint-velocity、table/self-contact、首次触碰 tick/pair、pelvis 高度/朝上分量和
  penetration 计数；
- 四个 authorization 位全部为 false。

两次相同输入的 `trace_content_sha256` 必须一致。NPZ 文件 SHA 另行记录；科学 parity 使用不依赖
zip 容器元数据的 content SHA。

`DIAGNOSTIC_FIXED_TAPE_COMPLETE` 只表示100 tick 执行链走完。joint velocity 在每个 `mj_step` 后
采样，包含最后一个 physics substep；self-contact 必须保留 maximum penetration 与 worst pair。
只要出现 table/self-contact 或
joint-velocity 越界，`safety.diagnostic_no_contact_gate_passed` 就是 false；不能把 runner complete
误报成 plant safety pass。

## 3. 测试

无 MuJoCo 的 host 仍可验证 JSON、31-D order、delay、decoder、SHA 和 no-clobber：

```bash
pytest -q hope_training/whole_body_tracking/tests/test_mujoco_native_single_env.py
```

在安装 `mujoco` 的环境，同一命令会额外编译 vendor A3 + 五实体桌网并执行完整100-tick smoke；
该用例在无依赖 host 明确显示 `skipped`，不能把 skip 报成真实 runner 已通过。

当前 diagnostic VecEnv 的聚焦回归为：

```bash
pytest -q \
  hope_training/whole_body_tracking/tests/test_mujoco_native_single_env.py \
  hope_training/whole_body_tracking/tests/test_mujoco_native_n1_ball_core.py \
  hope_training/whole_body_tracking/tests/test_mujoco_native_vec_env.py
```

`test_mujoco_native_vec_env.py` 要同时确认：76-D 列布局不漂移；普通 `step()` 在
physics 前 fail closed；substep event 顺序、去重与累积计数严格；tape timeout sticky；
base 两个阈值在边界不触发、只有严格小于才触发；同时触发时 reason order 固定；
joint actual 在 raw hard edge 内侧安全、边界触发且 sticky；配置/callable 源 SHA 漂移拒绝；任一已装
hard termination 后不能继续 step，只能显式 reset。

## 4. 2026-08-03 v5 exact diagnostic 结果

exact motion/audit/receipt SHA 分别为 `5899706b…b98101`、`c968ea8b…c2ff6`、
`756218ed…05ddde`。当时实际消费的 root MJCF SHA 为 `70c4fd65…36c0a`。delay 0/1/2 都走完
100 policy ticks / 400 physics substeps，但安全门全部 **FAIL**：

| delay | velocity events | self pairs / substeps | table pairs / substeps | max self / table penetration | first contacts |
|---:|---:|---:|---:|---:|---|
| 0 | 18 | 411 / 271 | 159 / 136 | 9.12 / 12.81 mm | tick 9: left hand–left hip; right wrist–table |
| 1 | 20 | 421 / 279 | 149 / 134 | 26.90 / 5.22 mm | tick 9: same pairs |
| 2 | 26 | 435 / 275 | 154 / 139 | 18.52 / 5.13 mm | tick 9: same pairs |

delay 0 在 reset 时是无接触状态；pelvis 从 `z=0.89184 m, up_z=0.86947` 在 0.20 s 内移到
`z=0.80308 m, up_z=0.8830`，随后两类接触同时出现。右腕–桌最短距离从 reset 的 `203.81 mm`
收缩到 tick 8 的 `32.18 mm`，tick 9 穿入 `11.63 mm`；左手–左髋从 `98.34 mm` 收缩到
tick 9 穿入 `2.15 mm`。将 probe 改成完全恒定的 teacher-q0 hold 仍在 tick 8/9 碰自身/球桌，
将 root 向下试探移动 10.6–30 mm 也不能消除失败。因此当前证据指向开放环姿态在该 plant/PD
下快速失稳，不是 delay 分支、小幅 probe 或单一 root-ground 高度偏差；不得通过关闭接触门
或放宽限制把该诊断改写成 PASS。运行产物位于 `/tmp/a3_mj_take061_v5.xeL7rC`。

## 5. 根因修复：teacher reference 与 physical birth state 分离

进一步 exact-model 检查确认，v5 frame 0 本身是动态挥拍帧，不是静态可站立的 controller birth
state：root 约有 `29 deg` 倾斜，右/左脚最近 sole-floor gap 约为 `10.605/15.959 mm`，reset
没有 floor contact。单纯下移 root 无法同时得到双脚合法接触；把双脚重做平也不足以解决问题，因为
保持完整 v5 root/下肢时，当前 runtime qdes envelope 内不存在可行的静态重力 hold。最明显的三个
无约束需求是：`waist_roll qdes=-0.36737 < -0.282743`、
`waist_pitch qdes=-0.95138 < -0.402473`、
`right_ankle_roll qdes=0.45048 > 0.282743`。

因此修复不是改 teacher，也不是关碰撞，而是分开两种状态：

- teacher reference 仍严格指向未修改的 v5 motion/frame 0；
- physical reset 使用已审计 shared-ready 的 root + 12 个腿关节，再覆盖 v5 的全部非腿关节；
- 在**当前 exact MJCF** 上重跑 joint/collision/sole/double-support/support/static-ground gates；
- 用 double-support LP 在 runtime qdes/effort 交集内求 controller birth 的 hold qdes；
- delay history fill 使用同一个 LP hold action，而不是 physical q，也不是 teacher q0。

旧 `bh_loop_c.dynamic_ready.v1.json` 只提供 shared root/leg 数值种子；它的旧 MJCF 声明和旧 hold
不被继承。生成器必须重验 seed SHA，并在当前 MJCF 上重新审计：

```bash
python -m hope_training.whole_body_tracking.mujoco_native.action_specific_hold \
  --contract configs/a3_vendor_runtime_authority_20260802_r8/bh_loop_c.shared_ready.training_contract.json \
  --teacher-motion assets/motions/chingmu_n1_take061u04_mechanical_candidate_v5_20260803/hope_Take_061_unit04_BH.measured_v5.npz \
  --teacher-frame 0 \
  --seed-dynamic-ready configs/a3_vendor_dynamic_ready_20260802_r8/bh_loop_c.dynamic_ready.v1.json \
  --expected-seed-sha256 3d604feb33145471b5dcc21279f26bc12e0351f0d158d7dc20dc8ed54517c306 \
  --output /tmp/take061_v5_action_hold_candidate.json
```

然后把 candidate 的**文件 SHA**显式传给 tape builder：

```bash
python -m hope_training.whole_body_tracking.mujoco_native.single_env make-tape \
  --contract configs/a3_vendor_runtime_authority_20260802_r8/bh_loop_c.shared_ready.training_contract.json \
  --teacher-motion assets/motions/chingmu_n1_take061u04_mechanical_candidate_v5_20260803/hope_Take_061_unit04_BH.measured_v5.npz \
  --teacher-frame 0 \
  --hold-candidate /tmp/take061_v5_action_hold_candidate.json \
  --expected-hold-candidate-sha256 1930cc71df19960aa6c0470bdc0304f364e2a30532dbc415131f22e3079e1f32 \
  --delay 0 \
  --tape /tmp/take061_v5_action_hold_d0.tape.json
```

consumer 会重验 candidate file/content seal、training contract、teacher、joint order、seed 和 root
MJCF 的路径与 SHA，重算 physical/hold semantics，并确认 history fill 精确解码到 sealed hold qdes。
candidate 和 receipt 都保留 `diagnostic_unauthorized=true`，不能据此宣称 measured-racket mechanical
admission、Isaac parity、canonical training 或真机安全。

2026-08-03 在 root MJCF `70c4fd65…36c0a` 上生成的 candidate file/content SHA 是
`1930cc71…e1f32` / `cce7483e…c417`；static gates 全 PASS，support margin `24.7715 mm`，LP
的最大 normalized available hold torque 为 `0.9959587`。投影到执行 envelope 内的 `+-0.02`
deterministic probe 结果如下：

| delay | qdes clamp | velocity / self / table events | effort clip events | min pelvis z / up_z | no-contact gate |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 / 0 / 0 | 1108 | 1.02171 m / 0.96664 | PASS |
| 1 | 0 | 0 / 0 / 0 | 1098 | 1.02316 m / 0.96465 | PASS |
| 2 | 0 | 0 / 0 / 0 | 1084 | 1.02345 m / 0.96301 | PASS |

完整 diagnostic 产物在 `/tmp/take061_v5_action_hold_final.AMNCcp`。三条都完成 100 policy ticks /
400 physics substeps，且 table/self penetration 都为 0。effort saturation 仍然很多，必须作为后续
controller/trajectory 诊断项保留，不能把本结果写成 policy learnability 或训练放行。

带 `mujoco==3.3.7` 和 `scipy==1.13.1` 的隔离环境回归结果是 `22 passed`；无这两个依赖的 host
结果是 `17 passed, 5 skipped`。

## 6. 2026-08-03 diagnostic VecEnv exact-subset 证据

`deec4a52c758b1f173436d4522e3e13e7ccb7bfd` 在 native physical-ball core 上增加了
sequential CPU diagnostic VecEnv、strict physics-substep contact ledger 和 exact tape-timeout latch。
`41411c3b6a6ef3ad03c2cba41370e84709066d8d` 继续绑定以下 exact base subset：

- `base_fell_tilt`: `pelvis_up_world_z < cos(0.7)`；
- `base_too_low`: `pelvis_link_origin_height_w_m < 0.5`；
- 两者都是 control step 后取样的严格小于判定，阈值本身不触发；
- latch 为 sticky；同时触发时子集内 reason order 是 `base_fell_tilt -> base_too_low`。

该子集的真源是
`HOPEDeployParityTerminationsCfg`，源文件 SHA-256 固定为
`490ad557eb966dc8399a7eddd2bf78e2ee6a6b6c8dae02c58e835baee0391c58`。每个进程首次生成
termination blocker receipt 时重读并校验该 SHA，漂移就 fail closed；通过后缓存不可变
template，并给每个 caller 深拷贝。4096 次 cache-hit receipt 调用合计 `.446 ms`，
receipt SHA-256=`353382b4…3789`。

exact clean Pod 证据：

- checkout: `/workspace/franco/actionball_mujoco_41411c3b_20260803`；
- 上述三个聚焦测试集：`48 passed in 15.71 s`。

2026-08-03 后续分支只改了同一 Isaac 配置文件中与 termination 无关的 N73
说明文字；`base_fell_tilt/base_too_low` 两条定义未变。MuJoCo 绑定已重新锁定当前整文件
SHA-256=`a012013c…3357f42`，host 三组聚焦回归为 `40 passed, 8 skipped`；skip 仍是缺少
MuJoCo/SciPy 的真引擎用例，不是通过。

这仍是 `Partial / diagnostic_unauthorized`。已实装的 base subset 不是完整 termination
union；剩余必须闭合的是 Isaac-equivalent robot/table collision termination、joint actual/qdes
hard edge 及 reason order、phase fidelity/recovery termination、terminated-batch compact reset 与
terminal observation、teacher/official-racket-site p/v/signed-face/long-axis、完整 contact-to-flight/net/landing
reward、PPO、save/resume 和 export。

2026-08-03 joint-actual successor：native single-env 现在每个 physics substep 对 31 个
runtime-order `model.jnt_range` 做 finite/increasing 校验并粘住 raw hard-edge 事件；
post-control-step 再按 Isaac `actual_joint_position_forbidden_zone` 的 exact
`q<=lower || q>=upper` 语义复核。配置 `margin_rad=0`，所以 tolerance 也是 `0.0`，不能用
epsilon 提前杀掉仍在硬边界内的状态。理由顺序为
`base_fell_tilt -> base_too_low -> joint_actual_forbidden`，首次理由 sticky；完整 termination、
reward 和 PPO 仍 fail-closed。
