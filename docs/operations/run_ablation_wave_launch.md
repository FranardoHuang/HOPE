# 标准工序：发一波消融

**每步 = 做什么 → 通过条件。** 不过就停，不许跳步补位。
排序/算力/seed 纪律看 [runbook](../runbook.md#统一队列排序与算力纪律)，本页只管发射动作。
本页把散在 `run_phase1_push_robustness_wave.md`、`run_phase1_balance_temporal_matrix.md`、
`run_on_runpod.md` 和 2026-07-26 实验记录里的同一套纪律收成一处。

## 发射前

1. **认领**：从 [NOW 唯一队列](../NOW.md#统一工作队列唯一优先级账本)取最靠前且依赖已满足的项。
   不许建第二个队列。
2. **冻 commit**：clean detached exact commit，写进每臂的 `source_commit.txt`。
3. **渲染命令到文件**：队列渲染器把完整 argv 写进 `arms/<job>/argv.txt`。
   **⚠ argv 模板必须取自引号完好的命令文件，不得取自进程表**——从 `ps` 抄会丢 shell 引号，
   Hydra mapping 被打碎（2026-07-26 首发即死，同波内又犯一次）。
4. **启动锁存在**：`ssh <pod> 'test -x /workspace/bin/kit_boot_lock.sh && echo LOCK_OK'`。
   **不用** `launch_kit_training_locked.sh`（其 `180 s` stale 门是 Wave A v8/v9 死因）。
5. **run 目录 no-clobber**：所有 run_dir 与 probe dir **都不得已存在**；创建用一次原子 `mkdir`。
   基础设施失败（stale/SIGABRT/身份竞态，**非** NaN/OOM 科学失败）允许在 fresh namespace 下
   逐字 retry 一次（`_r2` 后缀），同 phase 第二次失败转根因线；原 namespace 永久只读。
6. **GPU 真的有位**：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`，
   目标卡 **compute PID < 4**（4 条/卡是上限）。
   - 同时看利用率和日志活性，**不只看显存或 PID**；重复行按唯一纯数字 PID 计数。
   - 卡上现有进程**先 ps 认领**：谁的、活的还是死的（僵尸评估占槽让发射器死等 3 小时）。
   - **绝不抢占在跑的格**。快照会过期，每次发射前重查。
   - ⚠ 旧的"每卡零 compute 进程"口径已被推翻——它导致每卡第一格上卡后其余三格被拒。

### 未变配方的诊断续跑快线

下面六项逐字不变时，fresh diagnostic/canary 续跑不重复做 repin、动作 bundle 物化或 host
大联合回归：

1. exact source commit；
2. launcher 路径与 bytes SHA；
3. ordered action bundle 路径与 bytes SHA；
4. effective Reward recipe SHA；
5. PPO/policy contract SHA；
6. diagnostic/formal 与 fixed/dynamic curriculum 模式。

续跑只做四件事：回读 GPU owner/UUID、创建 fresh no-clobber namespace、渲染并保存 canonical
plan/argv、用 `/workspace/bin/kit_boot_lock.sh` 串行启动。任一身份字段改变，只重做受影响的
物化和专项检查，不把无关 host 套件重新跑一遍。

这条快线只购买“能否优化并产出 policy”的诊断证据；它不能把
`training_authorized=false`、关闭 formal evidence fence 或冻结课程的运行写成正式 Gate/curriculum
晋级证据。NaN/Inf、真实 hard limit、table hit、fall 和动作/Reward/PPO 身份漂移仍是硬停止条件，
不得借快线关闭。

小时巡检不能只看一次 NVML 利用率或 `run.log` 修改时间。对每个绑定 namespace，必须从所有新增的
完整 RSL-RL iteration block 记录：

- update、collection、learning 墙钟；
- [`collection_vector_step_wall_s`](../DEFINITIONS.md#collection-vector-step-wall-s)：
  `collection_wall_s / num_steps_per_env`；
- [`amortized_e2e_vector_step_wall_s`](../DEFINITIONS.md#amortized-e2e-vector-step-wall-s)：
  `iteration_wall_s / num_steps_per_env`；
- [`collection_environment_step_us`](../DEFINITIONS.md#collection-environment-step-us)：
  `collection_wall_s × 10^6 / (num_envs × num_steps_per_env)`；
- [`collection_environment_steps_per_s`](../DEFINITIONS.md#collection-environment-steps-per-s)：
  `(num_envs × num_steps_per_env) / collection_wall_s`；
- terminal、qdes/actual hard-limit、reference、table/fall 和 strike opportunity。

`num_steps_per_env` 必须从该 run 的 contract/agent receipt 读取，不能在巡检脚本里静默假设 24。
reason mask 可重叠，禁止相加冒充 terminal。已绑定 namespace 没有匹配活进程时必须报
`MISSING/EXITED`，不能因为 `ps` 没列出就从报告中消失。当前任务只允许显式 Pod1 target；
no-argument 巡检不得自动读取过期 wave 文件或连接 Pod2。

同卡第二条 operator-direct 诊断只用于 breadth，不是 canonical/formal 发射。它仍须保存 exact
plan/argv/identity、fresh no-clobber namespace，并用 Kit boot lock 串行启动；一旦在 scene
construction 或 PhysX start 停止前进，保留日志后按 exact PGID 关闭，不循环重试。`4ff48b21`
的 task-strong direct 就在 PhysX start 活锁，故未算作活跃 Reward 臂。

### ActionBall A3 upper q/qd 修复与 hot-path 快线

`eaf55fba` 已把 recoverable 2%-inner occupancy 从 Done 中拆出，但 Pod1 4096-env 前五轮仍有
`2.5k--4.2k/update` 的 raw-hard terminal，且 q_des projection/nonfinite 为零、episode 未到
`t_hit`。当前禁止继续靠放宽 actual hard edge、加 env 或改 Reward 掩盖该反例。旧
`canonical_ready_v1` 的零脚接触只描述 donor 本身，不能外推现役 upper。qvel-only successor
虽修正了恒定腿 qpos 与 stale qvel 的 schema 矛盾，但后续双动作 4096×5 已证明它不是 reset
根因：loop/block episode 约 `23/12` 步、strike 恒零，actual raw-hard 仍爆炸。真正合同缺陷是
该几何双脚接触的深蹲前倾 pose 不是 exact A3 runtime implicit-PD 的闭环 stable hold。
`candidate_id=G1` 只是 A3 候选代号，不是机器人型号。

发射顺序：

1. qvel-only 是已完成但被行为 probe 否决的 schema 清理，不再作为 long 输入。upper successor
   必须保留 head/arm q/qd、三腰相对 frame-0 的轨迹增量与 qd、frame
   count/timing/strike，改用
   `configs/a3_upper_stable_stand_v2.json`：
   - 12 个腿 qpos 使用 `AGIBOT_A3_CFG.init_state` runtime stand，腿 qd exact zero；
   - 三腰 q 轨迹按各自 frame-0 做常量平移，使 ready 等于 runtime default 零位，但保留
     动作增量和 qd；
   - root X/Y 和 source yaw 不变，root upright、`z=1.0684 m`；
   - exact A3 重算 body FK/velocity；racket world pose 允许随正确 root 改变，故旧 ball/task
     binding 必须全部重物化；
   - 输出目录 no-clobber、receipt last、三类 authorization false。
   Pod deterministic hold 与训练 smoke 必须证明该 plant birth 至少稳定跨过 `t_hit+margin`。
2. 历史 qvel-only receipt 仍须诚实保留以下已验证事实，但不得再外推成稳定性证书：
   - exact A3 上双脚接触、joint limit 与 unsupported/self-collision 检查 PASS；
     static-contact LP 必须原样记入 receipt；`feasible=null/missing scipy` 不得写成 PASS；
   - 所有 joint qpos、root、frame count、strike frame 不变；
   - 每帧 `right_racket` site position/orientation/linear/angular velocity 不变；
   - 首末 joint/root/body velocity exact zero；
   - 输出目录 no-clobber、report last、三类 authorization false。
3. full scope 不允许此 upper replacement；必须完整重编
   `grounded ready → selected core/window → grounded ready`，再重跑 aim/phase/physical-strike
   binding。旧 fivebind 的 SHA、帧号、旋转和证书不能跨 bytes 继承。
4. 新 upper bundle 先自然跑 `1 env × 2 update`；通过后 fresh 4096-env 只跑五轮定位。必须记录
   mean episode、strike opportunity、raw-hard/table/fall/nonfinite、q_des projection、
   collection seconds、environment-steps/s 和 finite checkpoint。episode 未越过 `t_hit` 或
   strike 仍为零时不得发 long。
5. stable-upper 是 birth/reference/actor 合同修复，无需 old/new 学习 A/B；它的验收是上面
   不变量与行为门。若修复后仍有
   mass raw-hard，才允许一条 birth-only、substep-only、policy-step 末已恢复的 diagnostic
   canary；task phase/current-post edge/nonfinite 继续 Done。
6. upper contact receipt 有两个互斥 schema：历史资产使用 corrected-Z alignment；
   stable-upper 使用整块 contact box retargeted alignment。后者的 authority 必须是
   `a3_stable_upper_selected_rubber_face_center_at_pinned_strike_frame`，必须声明不保留旧
   upper center，并闭合 `retargeted_world_z = ready_root_z + task_z`。不要把 full-motion
   authority 或 legacy 两个 Z 字段混入 stable-upper receipt。
7. N1 producer 的 `SUPPORTED_ACTIONS` 必须指向 stable-upper v2 exact motion bytes，并使用
   与其 `_runtime_site_velocity` 相同的 finite-difference strike speed；不得拿 MuJoCo site
   trace 的近似速度或 v1 bundle 跨 bytes 复用。

stable-upper v2 仍不能用 `qdes=physical_q` 保持到击球窗。动作专属 hold 候选必须在 Pod 的
`hope_isaac_venv` 里生成（系统 Python 没有 exact MuJoCo/HiGHS 环境）：

```bash
/workspace/hope_isaac_venv/bin/python \
  hope_training/whole_body_tracking/scripts/materialize_a3_dynamic_ready_contract.py \
  --action-id <bh_loop_c|bh_block> \
  --motion <absolute-stable-v2.npz> \
  --expected-motion-sha256 <sha256> \
  --stable-receipt <absolute-stable-v2.receipt.json> \
  --expected-stable-receipt-sha256 <sha256> \
  --runtime-contract <absolute-training_contract.json> \
  --expected-runtime-contract-sha256 <sha256> \
  --mjcf <absolute-a3_pingpong.xml> \
  --expected-mjcf-sha256 <sha256> \
  --output <fresh-no-clobber-candidate.json>
```

该工具保留历史 ground-LP feasibility 默认，只为 hold 显式选择按正负可执行力矩归一化的
minimax 目标；它还必须按 exact mapping 在 A3 runtime joint order 与 MuJoCo post-root
actuator order 间 scatter/gather。产物三类 authorization 均为 false。随后用 Isaac
nominal-hold 模式先截取未经 artifact 覆盖的 `raw_env_reset`，再截取
`physical_ready_after_reset_write`、step 1、step 10 和 final/pre-terminal，并持续送同一
hold qdes 到至少 `t_hit+margin`；截图、actual-hard/table/fall 与 root/foot 遥测一起判定
“原生 reset 就怪”“候选 ready 写坏”还是“出生正常但 plant 随后漂移”。通过前不得把候选接进
trainer。
每个动作必须使用 fresh no-clobber 输出，且 Pod 的物理 GPU 先用 NVML/owner lock 只读核对：

```bash
env -u CUDA_VISIBLE_DEVICES \
  HOPE_URDF_IMPORTER_NO_UI=1 \
  HOPE_AGIBOT_A3_USD_PATH=/workspace/franco/runtime_assets/a3_preconverted_usd_1b3fecd7/model.usd \
  LD_LIBRARY_PATH=/workspace/franco/runtime_assets/libglu_af791d1e${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
/workspace/hope_isaac_venv/bin/python \
  hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py \
  --task HOPE-PingPong-ActionBall-AgibotA3-v0 \
  --num-envs 1 \
  --device cuda:<FREE_PHYSICAL_GPU> \
  --nominal-hold <absolute-dynamic-ready-candidate.json> \
  --nominal-hold-sha256 <candidate-file-sha256> \
  --nominal-hold-receipt-out <fresh-receipt.json> \
  --duration-s <at-least-t_hit-plus-margin> \
  --screenshot-dir <fresh-frame-directory>
```

这是 simulator-only diagnostic：它关闭动作参考偏离终止与随机化，但保留
actual-hard、qdes nonfinite、table 和 fall；任何首个 terminal 立即停止，并把上一安全帧记为
`preterminal`。带截图的 Vulkan/RTX probe 不设置 `CUDA_VISIBLE_DEVICES`，直接把 Pod 物理卡号
写进 `--device cuda:N`；否则 Isaac 4.5 的 GPU/渲染枚举可能不一致。Pod 的 headless importer
还必须显式设置 `HOPE_URDF_IMPORTER_NO_UI=1`。fresh checkout 必须复用按内容钉住的 A3
preconverted USD；否则绝对 URDF 路径变化会触发同一资产的重复转换。当前 Pod 副本四层 SHA
依次为 `1b3fecd7… / 8e521141… / 5b5fc00b… / c76c5bdd…`，private GLU 的
`libGLU.so.1.3.1` 为 `af791d1e…`。日志必须越过 scene creation，不能把
`Simulation App Shutting Down` 的零退出误写成截图证据。正式 table receipt 仍遵守“单卡可见、
logical cuda:0”合同。该 receipt 与截图不授权训练、部署或真机。

nominal-hold 不读取 32 份 `force_matrix_w` 做 table receipt 的 USD/filtered-contact 枚举；该
枚举由独立 formal table smoke 负责。hold 仍保留并逐步读取 `robot_hit_table`、fall、
qdes-nonfinite 与 actual-hard term，因此这不是删除桌碰安全真值，只是把无关的收据物化从 reset
姿势诊断中移走。

以下性能改动若 Pod focused parity 通过，可直接进入 replacement，不另开学习 A/B：

- immutable frozen receipt 的 external SHA cache，不得改变 dataclass/pickle/wire/exact-resume；
- 同一 manager policy step 的 `_compute_strike_timing` 正常路径只算一次，direct/reset 仍重算；
- global + per-action error reduction 保持，`N<=8192` 的布尔 count 改为可精确表示的
  float32 后合成一次
  [device-to-host transfer（D2H，设备到主机传输）](../DEFINITIONS.md#device-to-host-transfer)，逐 step Python EMA、adaptive sigma 和
  reference perturb 逐值不变；
- `fired_valid` 空集保持旧 metric 的 device mask；`exact_strike` no-strike 早退暂不动。

这些补丁仍须在 Pod 跑 focused pytest、fixed-tape parity 和 profiler；“数学等价”免的是学习
A/B，不是回归。只有 4096-env healthy baseline 达到至少 `15k environment-steps/s` 且出现
strike 数据后，才启动 Reward/reference/curriculum 剂量比较。

#### Reset receipt granularity decision

Move the full human-readable receipt/transcript ceremony out of per-reset hot paths after the
segmented profiler and exact parity checks pass. CC's `(33-4)s / 4100 ≈ 7.1 ms` estimate is an
upper bound per env-reset, not yet a causal measurement; it also contains other ActionBall work.
This is a deterministic implementation optimization and does not need a learning A/B.

The replacement must keep a compact, checkpoint-bounded event journal containing at least
action/env/reset/swing identity, domain epoch/levels, birth↔sample assignment, proposal
admit/reject and reason, the exact GPU float32 task/solver output, lifecycle/outcome, curriculum
generation, and every table/fall/actual-hard/nonfinite truth. `seed + config` alone cannot
reconstruct which env reset when, the policy-caused outcome, or the exact solver task. Launch-time
motion/manifest/solver/physics/Reward SHA and admission gates remain unchanged; full JSON,
historical replay, hash-chain sealing and detailed reports move to checkpoint/hourly materialization.

Acceptance is Pod-only: fixed proposal tape task parity, counter/reason parity, old-receipt
canonical-byte reconstruction from the compact journal, uninterrupted-vs-exact-resume equivalence,
and segmented reset/throughput timing. Do not implement a hash-only journal or use a live mutable
GPU view as immutable evidence.

N1 diagnostic launcher 的 budget 名称固定为：

- `smoke`：`1 env × 2 update × save1`；
- `probe`：`4096 env × 5 update × save1`；
- `canary`：`16--1024 env` 的有界 Reward screen；
- [`milestone1000`](../DEFINITIONS.md#n1-milestone1000)：exact
  `4096 env × 1001 update × save100`，自然产出 `model_1000.pt`；
- [`long`](../DEFINITIONS.md#n1-diagnostic-long)：exact
  `4096 env × 20001 update × save100`，自然产出 `model_20000.pt` 的 finite reviewed 长跑预算。

`probe` 只能使用 exact 三元组，不能借 `canary` 或 `long` 填任意值。它仅验收同一 setting 在真实
并行规模下的构造、吞吐、reset 分账和 finite checkpoint，不产生 Reward 胜负或 curriculum
promotion 结论。`milestone1000` 用来买足够长的首轮学习证据，但仍是 diagnostic；它不因运行
更久就自动升级成 formal、curriculum promotion、第二 seed 或真机证据。`long` 同样是有限预算，
不再使用一个不可达的超大 iteration 哨兵来冒充“一路跑”。

### ActionBall finite q_des / reference-reset 切换

本段是 `curr-launch-fix` 功能分支候选；`origin/main/docs/NOW.md` 仍是运行态权威。旧 run 使用
“finite q_des 越包络即 reset”合同，不能 exact resume 成新合同。发射前必须在 effective Reward、
policy/runtime hard contract 和 canonical argv 中同时回读：

1. [`finite q_des execution projection`](../DEFINITIONS.md#finite-qdes-execution-projection)：
   包络内 `executed_qdes == raw_qdes`，有限越界时执行最近合法值，raw action/log-prob 不改；
2. [`qdes_projection_penalty`](../DEFINITIONS.md#qdes-projection-penalty)读取投影**前**的归一化超出量，
   首发 weight=`-5`；`-20` 只允许命名清楚的消融臂；
3. [`reference_guard_mode=metrics_only`](../DEFINITIONS.md#reference-metrics-only)：anchor/body/ee
   只记 counter，不 reset、不额外给 Reward；
4. nonfinite raw q_des、实际关节 current/substep raw mechanical hard edge、table hit 和 fall
   仍各自 hard reset，不能被 reference mode 或 projection 开关屏蔽；仅 predicted ballistic
   crossing 必须生成有限 brake target 而不 reset。实际 q 进入 hard edge 内侧 `2%` 只记
   joint/side/dwell 并由 actual-q barrier 持续收费，不再 reset；q_des clamp 只约束 drive
   target，真实 q 仍可能因隐式 PD、重力、接触和惯性进入该软带。

若 q_des projection/nonfinite 为零、fall 很低，而 `joint_actual_forbidden` 仍让 mean episode
length 长期短于 `t_hit`，先做单变量的滚动刹车 A/B：每个 fresh physics-substep 仍重新读 q/qdot，
但 prediction/brake horizon 保持一个完整 policy/control step（当前 `20 ms`），不能静默缩成
单个 physics tick（当前 `5 ms`）。不得同时改 Reward、Done 或 safety-band margin；只有
1-env smoke 后的 fresh 4096 run 让 episode 越过 `t_hit` 且出现持续 strike，才可替换长跑。

`5e94f21b` 已按上述判据给出反例：updates 1–16 的 actual-joint reset 为
`4,791.6/update`、mean episode `20.19` steps，最近十窗没有相对 `5dbb` 的吞吐提升。下一单变量
改为 [`finite_projection_soft_envelope_inset_fraction=0.05`](../DEFINITIONS.md#finite-projection-soft-inset)：
只在 ActionBall finite projection 模式把 soft q_des 包络上下侧各内缩 `5%`；raw action、
log-prob、Done、actual hard `2%` 安全带和 Reward 权重均不改。该比例必须同时从 action runtime
与 schema-3 training contract 回读；缺字段、config/runtime 不等或旧 checkpoint resume 均拒绝。
必须先用 Pod 1-env smoke，再做 4096 同 seed；source tests 不能代替 Isaac 结果。

`478f485b` 也已给出反例：q_des termination/projection penalty 均为零，但 4096-env updates
0--6 仍约 `4.7k actual-joint reset/update`，mean episode `19--24<t_hit` 且 strike 为零。
不要再增加 q_des inset 或直接放宽 actual band。下一 short diagnostic 必须在每个 PPO update
检查一条 `HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON`，至少含：

- exact articulation `joint_order`；
- episode age `<=1` / `>1` 的 terminal 分母、mean/max age；
- 每个非零 joint 的 current lower/upper/nonfinite-or-invalid、substep actual-hard、
  pre-apply nonfinite q_des 和 predicted-crossing overlap。

这些计数只用于定位，不能晋级 checkpoint。计数器必须在 device 上累加；rollout hot path
不得 host sync，update boundary 才允许一次小批量 D2H。schema v1 旧跑的 terminal 总数与同一
update 的旧 `joint_actual_forbidden` raw count 对账；切换软/硬语义后的 schema v2 必须分别报
`total_safety_event_count` 与 `total_hard_terminal_count`，只有后者与新 termination reason
对账。若不同，先停在诊断，不改训练 setting。

`8d2a1bcd` 已完成旧语义定位：三轮 4096-env 的 event 分母为
`3,187/4,457/5,087`，主因是 left ankle pitch lower，且绝大多数没有 raw-hard overlap。
后继 source 必须先 1-env×2-update smoke，再 fresh 4096 跑至少 3 个完整 update；晋级要求：

- mean episode 的 trailing window 越过 `t_hit≈31`，且 strike opportunity 不再恒零；
- `total_hard_terminal_count == joint_actual_forbidden reason`；
- 2%-inner safety event 可以存在，但 actual-q barrier contribution 必须非零并分账；
- raw hard/table/fall/nonfinite 不得相对 `8d2a1bcd` 增加；
- 报告每 update wall time 和 environment-steps/s，不用更多 env 掩盖 reset。

N=1 launcher 的 canonical Hydra argv 必须逐字包含
`+task.racket.reference_guard_mode=metrics_only`。该键不在 task YAML 中，少写 `+` 会在 compose
阶段失败；不能把 source-level argv 测试当作真实 Hydra 通过。full scope 还必须从 prototype
回读 `full_solver_admission_preflight.diagnostic_gate.status=PASS`，缺该 provenance 的旧 bundle
一律拒绝。

每轮同时记录 per-joint projection trigger、正负侧、投影前 mean/max excess 和执行值恰好贴边的
saturation fraction。触发率下降只能说明候选趋势；冻结 policy、把 projection penalty 置零后复测
仍低，才说明 policy 自己学会了限位。CaT 连续违规调制和 PPO policy-mean bound loss 会改变训练
目标/runner，本轮不临时叠加。

性能判断使用健康对照：旧 `6.4 s collection/update` 来自 mean episode length=`1`、
`98,304 reset/update` 的失败 probe；同代码修正 stand hold 后代表值约 `4.49 s`。旧 v2 也曾把
早期 `1,690–2,301 ee_body_pos reset/update` 学到后期 `1–7/update`。当前旧语义 ActionBall
loop/block 约为 `27/48 s per update`，分别由 ee/qdes 主导；finite-qdes 切换对 block 预计省
`14–17 s/update`，但只有 fresh run 的上面四个 timing 字段能验收该预计。

reference 动作无需因本切换重做：upper/full loop/block 四件老师的 hard/soft/2%-inner crossing
均为 `0`，block 全片 normalized hard/soft margin `0.115081/0.072312` 不小于 loop。该证据只排除
“block 老师贴限”根因，不替代 fresh rollout、table/fall 或实际关节安全检查。

## 冒烟

7. **一格 2-iter smoke**（旧代口径：`4096 env × 24 step × 2 update` full-scene probe，
   或 `512 env × 25 iter` 机制检查）。
   - **让它自然退出**；不许手工 TERM/KILL 探针来伪造"通过"。
   - **通过条件**：`grep -nE 'WARN|Error|Traceback'` 的 WARN 行**全部**进摘要、
     Error/Traceback **零条**；且 `grep -Fc 'q_des CLAMP ACTIVE' > 0`
     （限位剪切 2026-07-06 起默认开，缺这行 = 有人显式关了，只允许出现在老配方复现对照臂上）；
     且 `mean_episode_length` 不恒为 1；若恒为 1，必须按 reason ledger 区分 qdes/reference、
     actual/nonfinite/table/fall 与出生错位，不能再一律写成出生位问题。
   - 一格通过即可发全矩阵（精简治理，不设多层仪式）；高风险波才逐臂解锁。

## 发射

8. **严格串行**：同 pod 内 boot 串行（由 `kit_boot_lock.sh` 持锁保证），
   **看到该臂首个 `Learning iteration` 之后**才轮到同 pod 下一臂。
   两个 pod 可以并行各 boot 一格。相邻两次 launch 错峰 **≥60 s**（同秒启动撞 CUDA 枚举，
   报 "no suitable CUDA GPU"）。
9. **日志目录先 mkdir**（目录不存在 ⇒ 发射壳当场死，连报错都看不到，两犯）。
10. **run_name 当场进实验 run table**；责任/优先级变了才动 NOW。
11. **发射后回读 config**：从 run 的 wandb `debug.log` grep `motion_file` 核对，
    确认 `strike_phase` / `mount_normal_sign` / reward pack 与绑定收据一致。
    **⚠ `motion_file` 路径写错会静默回退到 WandB registry**——不回读就不知道训的是哪条片。

## 监控（只读不写）

12. **首迭代判定**：必须看到 `Learning iteration` **且**绑定的 PID/PGID/starttime/cmdline
    仍是同一活进程。**只有 resume 行不算首迭代**——它会让监视器误报。
13. **boot/stale 双门**：首个 `Learning iteration` 的总 boot timeout 是 `1800 s`，但日志
    连续没有任何推进时，launcher 的 `KIT_BOOT_STALE_TIMEOUT_S=900` 会更早 fail-closed。
    因此“进程仍在高 CPU 构造”不等于可静默等满 1800 秒；若大型 env 构造需更久，必须先量化
    startup phase 并显式修改、记录该合同，不能把 launcher 的 900 秒自然停止写成 PPO 失败。
    出现首个 `Learning iteration` 后仍以 900 秒无推进为 stale。
    task teacher-rate 的边界只有一个真源：
    `canonical_teacher_rate_from_site_speed`。producer 与 Motion consumer 都必须调用它；其
    `5e-7` 绝对容差只吸收 GPU float32 接缝，不会 clip、重定时或放宽 action support。consumer
    不得另写严格区间复验，否则合法边界题会在长跑中随机 Traceback。
14. **里程碑算术**：fresh 要写出 `model_1000.pt` 必须传 `max_iterations=1001`（0 起数）；
    热启动把相对偏移加到父迭代号。**终版存档名是 `model_13599` 不是 `13600`**——
    等 13600 会永远等。
15. **摘要抓异常不抓预期**：WARN 必进摘要。
16. **后台任务卫生**：一个目的一个监视器，目的消失立刻停；**超时参数不生效，必须显式停**；
    每次汇报清点"几个活着、各干什么"。

### N=1 动作专属 dynamic-ready fresh 发射（2026-07-30）

新一代 N=1 upper bundle 使用 schema 2，并把
[动作专属动态准备合同](../DEFINITIONS.md#action-specific-dynamic-ready)的 candidate 与独立
Isaac nominal-hold PASS receipt 一起钉住。launcher 会把
`action_ball_dynamic_ready_bootstrap=true`（物理出生仍是 motion frame 0，但控制目标与 actor
初始输出使用该动作 hold qdes）及两件文件的 path/SHA 逐字传给 trainer；旧 bundle v1 只允许
审计读取，不能进入该发射路径。shared-ready 与 dynamic-ready 不得同时打开，resume 不得覆盖
fresh actor bias。正式写 smoke spec 前，先在 Pod 以 `1 env`、diagnostic 和同一 dynamic-ready
双 pin 运行 `action_ball_policy_recipe_output_path=<fresh absolute path>`；该 recipe-only 构造
只写 exact PPO/policy contract，不做 PPO update。spec 的 `policy_contract_sha256` 必须取自这份
新 schema-2 recipe，旧 shared-ready SHA 不得复用。

fresh successor 还必须把 actor 合同切换为
  [`action_ball_table_pose_twist_heading_task_teacher_start_v2`](../DEFINITIONS.md#action-ball-teacher-start-contract)，
即 exact fixed-194：相对 racket position residual、demanded velocity 和 raw-A face normal 全部
统一到 yaw-heading frame；另加桌面中心 frame 下的 base XYZ、完整连续 6D orientation 与
root-COM 三轴线速度，并直接提供同一 Motion phase governor 的
`time_to_teacher_start_s=max(pre_swing_wait_s-task_age_s,0)`。policy recipe 只绑定
PPO/decoder/dynamic-ready，不绑定 observation 名称或宽度，所以现有 tracked schema-2 recipe
可复用；旧 182/191-D checkpoint、旧混合-frame 194-D checkpoint，以及当前兼容合同下的
194-D checkpoint 均不能因同宽复用。v2 hard contract 必须由 fresh smoke 的实际 term order/width
和 checkpoint 证明。该 sim
输入由 rigid-body truth 构造；当前 C++ builder 不支持 fixed-194 v2，且真实 marker→base 旋转外参、
gyro 外参和线速度估计器尚未闭合，因此这一步只授权 Pod 训练、不授权真机。

当前 launcher 还逐字加入
`+task.domain_rand.stable_ready_plant=true`。它保留旧 robot-material DR 与 policy recipe
钉住的 joint-default `±0.01 rad`，关闭 torso CoM、link mass 和 PD-gain DR。理由不是方便过门：
旧 loop/block `4096×5` 在第一次 PPO 前分别有 `860/864` 个 env 撞 `waist_roll` raw hard，
hard-env Jaccard `0.982`，而 qdes 与 teacher 都有余量，证明 full DR 的共享 plant 已压过当前
ready 稳定域。fresh `4096×5` 必须验证该 profile 能跨 `t_hit` 且不再 hard 爆炸；1000 update
后再按具名 DR 轴逐项恢复。

验证顺序是 Pod focused tests → `1 env × 2 update` → `4096 env × 5 update` → fresh
`milestone1000`。前两门只判断构造、finite checkpoint、reset 后 q/qdes/last-action 一致，以及
episode 是否能够活到动作 `t_hit`；五轮没有 strike 不能判策略不可学习。进入千轮诊断后在
`200/500/1000` 观察 fatal、finite、teacher imitation、击球机会与真实安全；按历史经验，击球
学习结论至少等到约 1000 updates 和足够 eligible denominator。`milestone1000` 到点后才决定
是否进入 reviewed `long` 或开 Reward/reference/curriculum canary。

旧 diagnostic runner 跳过 formal Reward 时也跳过 joint-safety consumer，却仍每 policy step
生产摘要；4096 槽在约 `170 × 24` policy steps 后必然溢出。fresh successor 必须启用
[diagnostic joint-safety drain](../DEFINITIONS.md#diagnostic-joint-safety-drain)，每个 PPO update
沿已有 prepare→optimizer→commit/ack 事务排空。已经 sticky overflow 的旧 checkpoint 不得清闩
续跑，必须 fresh no-clobber 重发。

## 停止

17. **禁止 `pkill` / `killall` / `pgrep -f` 后批量发信号**——会命中 ssh 远端 shell 或相似 run。
18. 从**经核对的 launch sidecar** 读 exact PID/PGID（不得用命令行模式搜索结果代替所有权 sidecar），
    先保全 checkpoint/contract/log 并核对迭代号/finite/schema/合同 SHA，
    再 `kill -TERM -- "-$PGID"`，仅在 TERM 未退出且证据已落盘时 `kill -KILL`。
19. **通过条件**：exact PGID 消失、认领槽位无本臂 compute PID、`kit_boot_lock` 无 holder、
    其他臂完好、什么都没删。完整流程见
    [RunPod 精确停止](run_on_runpod.md#已登记-phase-1-实验臂的算力释放)。

## 判死之前

- **硬止损**（立即停）：合同/哈希错、NaN/Inf、不可恢复 crash、开关未生效、train/exam 泄漏、
  结果删失或分母错误。先保留日志，不自动换配方重试。
- **证据止损**：至少两个相邻里程碑 **且** 至少一个 `50/侧` 判卷点都显示候选在**较差侧**被对照支配，
  且无其他预注册主指标补偿，才停整对。
- **固定预算 canary 到点自然结束**，它只决定"要不要买第二 seed / 延长"，
  **不能据此宣称 family 永久失败**。
- 同一 `(family, seed)` 的 old/S1 是**不可拆的一对**；除硬失败外不准只停差的一边。
- **跟踪三合格不用来判死臂**（拍面 25° 误差照样 79% 上台的教训）。

## 想把这些变成闸门

第 3、8、9 步现在只是文字。确切检查与落点见
[应当变成闸门的规则](rules_that_should_be_gates.md) P1 #8、#9。

## 带题库的臂:发射清单(2026-07-27 实测,每条都是护栏当场炸出来的)

> 起因见 [EXP-V2-REWARD-FREEZE §5](../experiments/2026-07/EXP-V2-REWARD-FREEZE-20260726.md)。
> 人话:凡是吃虚拟球落点奖励的臂,球拍速度指令**必须是解出来的答案**,不能是盒子里随机抽的。
> 随机抽的话,"听话地照速度走"和"把球打回去"在多数抽样下是反的,回球率注定接近零。

### 0. 先确认题库存在

**这一步最容易被跳过,而它是 2026-07-27 那次"跑不起来"的唯一原因**——两台 pod 上一份题库都没有。

```
PYTHONPATH=<repo>/hope_ws/src/hope_planner \
python scripts/gen_stage1_questions.py \
  --clip <family>:<compiled_clip.npz> \
  --grip off \
  --split train --n 8192 \
  --stroke-guard stats \
  --stroke-budget-clips <fh_v4rg_cal.npz> <bh_v4rg_cal.npz> \
  --out <bank.npz>
```

- `--grip off` 是**硬要求**。默认值 `registry` 会对未注册的 canonical clip 悄悄套一个 40.26° 人手握拍角;
  marker 是唯一权威。
- n=8192 单核约 50 分钟。生成完先看三个数:可解率(应 ≥85%)、torch 闭环落点(应全中、中位 ~3 mm)、
  过网(应全过)。任何一个不对,是**击球点选错了**,不是题库参数问题——挪点,别拿不可解的题去训练。
- 还要看一个泛化量:`answers inside the clip swing-velocity cone`。这个百分比越低,策略要偏离老师越多。
  实测窄带(来球 2–5 m/s)26%,宽带(1.5–7 m/s)21%。

### 1. 发射器必须写全的开关

| 开关 | 值 | 不写会怎样 |
|---|---|---|
| `task.racket.question_bank` | 题库路径 | 护栏拒绝开机(anti-correlated 构型) |
| `task.racket.face_command` | `true` | 护栏拒绝:解出来的拍面没人打分 |
| `task.racket.mount_normal_sign_per_clip` | 每 clip 一个 ±1 | 护栏拒绝:符号错会悄悄判错哪一面胶皮 |
| `task.racket.achieved_target_mix_prob` | `0.0` | 护栏拒绝:HER 回放在题库覆写之前,混进来的目标没解过 |
| `task.racket.vel_range_per_clip` | `null` | 护栏拒绝:题库模式下这是死旋钮 |
| `task.racket.ref_vel_scale` | `1.0` | 同上 |

**hydra 一律用 `++` 前缀**。`+` 在 key 已存在时会炸,`++` 两种情况都对——不要靠猜 key 在不在 yaml 里。

### 2. 拍面观测通道(`face_command_obs`)现在开不了单臂

开它要走 179D 契约 `deploy_parity_face179`,而 schema-3 结构校验**硬性要求正反手双 clip**
(单反手臂给不出 `[+1,-1]`,家族表也要求两族齐全)。单 clip 臂只能 `face_command_obs=false`——
拍面**仍然被主项 `racket_normal` 打分**(权重 0.5),只是策略看不见指令值、得自己从来球推。
双 clip 臂才能开这条通道,那本身就是一条干净的消融。

### 3. 同节点串行起

同一台 pod 上三个 Isaac 进程间隔 2 秒同时起,其中一个在 URDF 导入阶段
`malloc(): invalid size (unsorted)` 堆崩。**等前一个进 Learning iteration 再起下一个。**
