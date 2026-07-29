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
4. nonfinite raw q_des、实际关节 hard-limit、ballistic/substep crossing、table hit 和 fall 仍各自
   hard reset，不能被 reference mode 或 projection 开关屏蔽。

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
13. **stale 门**：首个 `Learning iteration` 之前容忍 `1800 s` 无推进；之后收紧到 `900 s`。
14. **里程碑算术**：fresh 要写出 `model_1000.pt` 必须传 `max_iterations=1001`（0 起数）；
    热启动把相对偏移加到父迭代号。**终版存档名是 `model_13599` 不是 `13600`**——
    等 13600 会永远等。
15. **摘要抓异常不抓预期**：WARN 必进摘要。
16. **后台任务卫生**：一个目的一个监视器，目的消失立刻停；**超时参数不生效，必须显式停**；
    每次汇报清点"几个活着、各干什么"。

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
