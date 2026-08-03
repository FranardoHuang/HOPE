# MuJoCo native fixed-tape / diagnostic VecEnv

这份工序同时保留底层 single-env fixed-tape 证据和当前 no-reward diagnostic
`VecEnv` 证据。single-env 部分只验证一件事：真实 vendor A3 MJCF 加五实体桌/网场景，
能否按 schema-3 的 31-D
`action -> episode-fixed delay -> affine qdes -> 5% soft-envelope interior 与 hard-inner 交集投影
-> total-PD` 合同确定性走完100个 control tick。

当前 native stack 已有 physical-ball core、76-D purpose-group observation、deterministic
batched reset、finite diagnostic rollout、physics-substep contact-event ledger、exact tape timeout 以及
`base_fell_tilt/base_too_low/joint_actual_forbidden/joint_qdes_forbidden/robot_hit_table` 的
exact termination subset。joint actual 在每个
control step 后按 runtime joint order 比较实际 `q` 与 MuJoCo `model.jnt_range`，边界容差按
Isaac raw hard-edge 语义固定为 `0`，并 sticky 保留同一 control tick 内任一 physics substep 的触边；非有限状态、非有限/倒置区间、`q <= lower` 或
`q >= upper` 均触发。三条使用 sticky latch，reason order 固定为 tilt、height、joint actual。
Isaac termination authority 按实际消费的 class inheritance、direct term assignment/source
order、function 与 assignment selected-AST semantic SHA-256 固定；相关阈值、term/order、
callable 或 latch 漂移 fail closed，无关 prototype/reward 行不影响。
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
  hope_training/whole_body_tracking/tests/test_mujoco_native_table_termination.py \
  hope_training/whole_body_tracking/tests/test_mujoco_native_vec_env.py \
  hope_training/whole_body_tracking/tests/test_mujoco_n1_reward_event_kernel.py \
  hope_training/whole_body_tracking/tests/test_plant_contract_v1.py \
  hope_training/whole_body_tracking/tests/test_judge_plant_contract.py
```

`test_mujoco_native_vec_env.py` 要同时确认：76-D 列布局不漂移；普通 `step()` 在
physics 前 fail closed；substep event 顺序、去重与累积计数严格；tape timeout sticky；
base 两个阈值在边界不触发、只有严格小于才触发；同时触发时 reason order 固定；
joint actual 在 raw hard edge 内侧安全、边界触发且 sticky；配置/callable 源 SHA 漂移拒绝。还要确认 joint qdes 的 finite-projection
语义：有限越界 proposal 不 reset，NaN/Inf pre-clamp affine qdes 才触发。加入 table 后 hard
subset 的顺序固定为 tilt→height→robot table→joint qdes→joint actual。robot/table guard
另须验证 canonical root MJCF、base/live 32 个 owner body 的 parent/local-frame identity、
五实体桌体、43-component artifact/live racket OBB、`0.02 m` margin、inclusive overlap、
每 substep 采样和 control-step sticky；PlantBinding/ledger/VecEnv 均只接受 decimation=4。
加入 phase-fidelity 最小切片后，完整 hard reason order 为 anchor pos→anchor ori→end-effector
body pos→tilt→height→robot table→joint qdes→joint actual。前三项严格使用 `>0.25 m`、`>0.8`、
`any(>0.25 m)`，阈值边界安全；`recovery_hold/in_hold=true` 与 episode-frozen
`reference_terminations_enabled=false` 都屏蔽 reference verdict。测试还必须验证 selected-AST
semantic source drift、ABI key/finite/body-width、all-or-none core advertisement、漏样本 fail closed，
以及 phase reason 命中后只 compact reset 对应 env。

当前 host native+plant 扩展口径为 `115 passed, 18 skipped`。Pod1 已在 detached clean
`454416b978cec7def19a067d613e4280d3024203` 使用 `/workspace/hope_isaac_venv/bin/python`
复核 native=`107 passed`、plant=`26 passed`、runner guards=`25 passed`，合计
`158 passed, 0 skipped, 0 failed`。该结果只关闭 Python 3.10/MuJoCo/Torch 这一层 runtime
回归，不解除下文 Reward/PPO/normal-step blocker。
selected-source digest 使用 Python 3.10+ 一致的 portable AST 序列化：忽略新版本才增加的空
`type_params`，显式编码 `Ellipsis/bytes/complex`，其余语义仍由 exact selected node
SHA fail closed。当前 diagnostic VecEnv
以 `episode_dones=exact_hard_terminations OR time_outs` 逐 env compact reset：只清命中行的
core、episode length、hard latch 与 ledger，survivor 行跨步连续。`DiagnosticBatchStep.observations`
是 reset 后 next observation；`terminal_observations` 只在 `terminal_observation_mask` 命中行表示
reset 前终态，`per_env_ledgers` 是 reset 前且 caller-owned 的 deep copy。测试必须修改 nested
`all_reasons`/contact pair 并证明不能污染 survivor live ledger，还要验证 table guard 的首次 substep
进入 terminal snapshot、只 reset 命中 env、新 episode latch 清零。

rollout v4 receipt 的 `question_source_sha256_by_env` 按 env 保留异构 question lineage，不能只写
第 0 行。完整 semantic 对象也在 receipt 内；未返回的 terminal trace 用绑定 shape/dtype/SHA/
validity-mask source 的 digest-only descriptor 表示。按 receipt 声明的 ordered inputs，receipt 加
返回 trace 必须能独立重算 `trace_and_event_sha256`。v4 semantic 另绑定 phase sample contract
SHA、runtime availability 与逐 env reference-tape SHA lineage；termination transcript 逐 env 保留
canonical phase sample，不能只保留最终 reason。v4 还绑定 native physical-event contract
SHA 并把逐 env validated facts transcript 放入同一 digest。production core 只有显式安装下述 external
MotionCommand reference tape 才广告 ABI；默认无 tape 时 receipt 必须保持
`exact_phase_fidelity_runtime_sample_available=false`，formal blocker 为
`native_core_phase_fidelity_reference_tape_not_installed`。合法安装后只能关闭 termination union，
完整 Reward/PPO/save/resume/export 仍关闭，不能单独授权正常 PPO。

### 3.1 phase-fidelity external reference tape 安装

禁止从 `time_to_contact_s`、ball event 或 live action slot 猜 swing/follow-through/recovery。允许的唯一
native 输入是 `a3_mujoco_phase_fidelity_reference_tape_v1`，由外部 Isaac MotionCommand 导出并以
调用方提供的 expected file SHA 校验。payload 必须绑定：

- 当前 `plant_binding_sha256`、`scene_binding_sha256`、`robot_tape_sha256`；
- exact `a3_mujoco_phase_fidelity_sample_contract_v1` content SHA；
- `sample_timing=post_control_step`、anchor=`pelvis_link`，以及固定 feet/hands body order；
- external authority source SHA；
- 与 robot tape action row 数完全相同的 reference rows。每行包含 reference anchor z、reference
  projected-gravity body-z、四个 reference body z、`in_hold`/phase context 与
  `reference_terminations_enabled`；最后一项整 episode 不得变化。

Python VecEnv factory 通过成对参数安装：

```python
MujocoN1DiagnosticVecEnv.from_authorities(
    ...,
    phase_fidelity_reference_tape_path=phase_tape_path,
    expected_phase_fidelity_reference_tape_sha256=phase_tape_file_sha256,
)
```

single-core CLI 的 `run` 子命令对应使用：

```text
--phase-fidelity-reference-tape PHASE_TAPE.json \
--expected-phase-fidelity-reference-tape-sha256 <exact-file-sha256>
```

path/SHA 必须同时提供；不提供时 core 不广告 phase sample ABI。安装后，core 每个 control step 用
live `pelvis_link` link-origin/rotation 与四个 body `xpos` 计算误差。VecEnv 会再次核对所有 core 的
contract SHA 与逐 tick sample；任一步失败都会将 batch 标为必须 full reset。该路径没有仓内正式
reference tape，也未在 Pod 真 MuJoCo/torch runtime 验证，所以当前写 `未测`，不得据此开放 PPO。

### 3.2 native physical reward-event facts

production core 现在广告 `a3_mujoco_n1_physical_event_facts_contract_v1`，并在每个
`step` 返回累计的 racket contact edge、首个 contact stamp、simultaneous/recontact
invalid reasons，以及带 policy tick/substep 的首个 contact-free outgoing position、线速度和自旋。
VecEnv 的 all-or-none 规则要求全部 core 同时广告同一 contract；广告后漏样本、source/
contract SHA 不同、非有限 vector 或事件乱序会使整批失效并要求 full reset。validated
facts 出现在 `DiagnosticBatchStep.per_env_native_physical_event_facts`，并在 compact reset 前
写入 rollout v4 的 `native_physical_event_transcript` 及总 digest。

这不是 reward 开关。contract 固定 `selected_rubber_authority_available=false` 和
`reward_authorized=false`；当前 `right_racket_collision` 命中不能代签 selected-rubber face，也没有
desired-contact/window、outgoing predictor、observed legal net/landing、swing closure 或 per-term reward
magnitude/weights authority。因此 normal `VecEnv.step()`、PPO、save/cold-load 仍在 physics 前拒绝。
exact resume 还需序列化 MuJoCo `MjData`、delay queue、core contact/outgoing state、VecEnv ledger/buffers 和
Python/NumPy/Torch RNG；不得把仅 policy/optimizer 的 cold load 称为 resume parity。

exact `7135d5ce` 的 Pod1 clean checkout 随后执行同四组完整回归=
`72 passed in 17.44 s`；该结果早于 compact-reset/lineage successor，只验证当时 table-guard
路径，当前 successor 尚未 exact Pod 重验，正常 PPO 授权仍保持关闭。

## 4. 2026-08-03 v5 exact diagnostic 结果

current qdes successor exact `0d1d641e` 已物化到 Pod1
`/workspace/franco/actionball_mujoco_0d1d641e_20260803`。Pod runtime 执行完整
native suite=`55 passed in 13.59 s`，host 上因缺 MuJoCo/SciPy 跳过的 optional tests
在 Pod 全部实际执行。该结果不开放 normal PPO `step()`；它只是当前
diagnostic scene/VecEnv/termination subset 的 exact-Pod 回归。

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

2026-08-03 后续分支在同一 Isaac 配置文件加入与 termination 无关的 observation prototype；
`base_fell_tilt/base_too_low` 两条定义未变。table/base/phase 现统一按实际消费的 AST 语义节点锁定，
避免无关 class/reward 行触发假漂移，同时相关 threshold/term order/callable/latch/gate/hold/body-order
变更仍拒绝。当前五组扩展回归为 `89 passed, 18 skipped`；skip 仍是缺少
torch/MuJoCo/SciPy 的集成或真引擎用例，不是通过。

这仍是 `Partial / diagnostic_unauthorized`。robot/table、joint actual/qdes、compact reset/terminal
observation 与 phase-fidelity strict predicate/sample ABI 已实装；production core 也能消费显式 external
reference tape 并计算 runtime sample，但仓库尚无安装到当前 run 的正式 tape，所以默认 formal union
仍未闭合。其后仍需 teacher/official-racket-site
p/v/signed-face/long-axis、完整 contact-to-flight/net/landing
reward、PPO、save/resume 和 export。

2026-08-03 joint-actual successor：native single-env 现在每个 physics substep 对 31 个
runtime-order `model.jnt_range` 做 finite/increasing 校验并粘住 raw hard-edge 事件；
post-control-step 再按 Isaac `actual_joint_position_forbidden_zone` 的 exact
`q<=lower || q>=upper` 语义复核。配置 `margin_rad=0`，所以 tolerance 也是 `0.0`，不能用
epsilon 提前杀掉仍在硬边界内的状态。理由顺序为
`base_fell_tilt -> base_too_low -> joint_actual_forbidden`，首次理由 sticky；完整 termination、
reward 和 PPO 仍 fail-closed。
