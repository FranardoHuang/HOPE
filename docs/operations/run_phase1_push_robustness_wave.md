# Phase 1 push 鲁棒性 12 臂波：从零到发射

本页只操作 12 臂 push 波 `{W,V} × {p02, p035, p05, yaw, ang, fast}` 的仿真训练队列。实验真源是
[EXP-P1-PUSH-ROBUSTNESS-20260721](../experiments/2026-07/EXP-P1-PUSH-ROBUSTNESS-20260721.md)；
共享代号人话见[术语表](../DEFINITIONS.md)。本页不授权真机、部署、第二 seed 或对照重买。

入口与不变量（人话：这轮用哪些文件、哪些数字永远不变）：

- config：`configs/phase1_push_robustness_20260721.yaml`（12 臂唯一机器队列＋冻结
  `launch_order`；已落盘、待合入 main，合入前本页只读不执行）
- renderer：`scripts/run_phase1_push_robustness_queue.py`（lean runner：只打印计划表/核对单/
  逐字 SSH 命令，自己绝不 SSH、绝不发信号、绝不写远端；本波不写死 pod/gpu，渲染时用
  `--pod`/`--gpu` 注入认领到的空槽）
- 本轮**没有** manifest-SHA 多层审批链：与矩阵同款，统一计划表（`--plan`）＋依赖核对单
  （`--checklist`）＋逐字渲染人工核对；parent checkpoint SHA-256 直接钉死在 renderer 里
  （W `2caab3dd...fcce`、V `ad901910...2716`，与矩阵同两份 `model_6700`）
- 远端 checkout：`/workspace/codexschema/nohope_push_20260721`（clean detached exact commit）
- namespace：`/workspace/codexschema/phase1_push_robustness_20260721`
  （probe 在 `probes/<job_id>`，science 在 `runs/<job_id>`，no-clobber）
- 每臂：`4096 environments`、`seed=3`、从各自 parent `model_6700` 续 `10001` updates、
  save/100；probe 每臂 `2` updates 自然退出到 `model_6701.pt`
- 配方：矩阵 `C+S0` 逐字（`action_rate_weight=-0.1`、slew hinge `0`、三稳定机制 weight `0`、
  probe 全开）＋该臂八个 `task.push.*` 键显式值（`enable`＋`interval_range_s`＋六个
  五键 `enable/interval_range_s/vel_xy_mps/ang_vel_radps/ang_axes`，x/y 对称 ±v 由 contract 单源展开、z 永不推）；对照是矩阵在跑的 `w_c_s0`/`v_c_s0`，
  **不发对照**
- 发射器：远端用 `/workspace/bin/kit_boot_lock.sh` + `setsid nohup`；**不用**
  `launch_kit_training_locked.sh`（其 `180 s` stale 门是 Wave A v8/v9 死因）
- watchdog：boot 停滞 `1800 s`、首个 iteration 后停滞 `900 s` 才判死；唯一允许的重试是逐字
  重发一次、run_name 加 `_r2` 后缀；同 phase 二次失败转根因线
- 谱系：`checkpoint_allow_contract_mismatch=true`，全部后代只作诊断、formal-exact-ineligible

不要手改 rendered SSH argv、run root、GPU 或权重；要变化就改 config/tests/实验记录后重新复核。

## 0. 核对 main 权威与 NOW 认领

人话：确认本地就是最新 `origin/main`、push 键实现＋队列文件已进 main、统一队列 NOW 里有这轮
的责任人/执行者/分支/queue id 认领行。任一失败只许跑本地 validate/plan，不许执行任何 SSH。

```bash
PUSH_ROOT="$(git rev-parse --show-toplevel)"
cd "$PUSH_ROOT"

git fetch origin main                     # 人话：取最新 main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"   # 人话：本地必须等于 origin/main
test -z "$(git status --porcelain)"       # 人话：工作树必须干净
git cat-file -e origin/main:configs/phase1_push_robustness_20260721.yaml
git cat-file -e origin/main:scripts/run_phase1_push_robustness_queue.py
git show origin/main:docs/NOW.md | grep -n "phase1_push_robustness_20260721"
# 人话：NOW 的同一条认领行必须同时含 责任人 franco、执行者 与本 queue id；缺任一即失败
```

功能分支里的 `NOW` 不是授权。全局顺序与算力归属只看最新 `origin/main:docs/NOW.md`；本队列
只定义局部依赖 `本地验证 → 冻结 exact commit → 核对单全过 → 空槽认领 → 逐臂 probe →
逐臂 science`。

## 1. 本地验证、计划表与核对单

人话：先跑本轮相关的本地单测（push 键接线/EventTerm 参数化的新单测＋12 臂队列渲染器单测），
再让 renderer 打印 12 臂计划表和发射前核对单；默认调用不生成任何远端命令。

```bash
# 人话：跑 push 事件接线/合同块、reward 覆盖键、schema-3 合同与 12 臂队列渲染器的
# focused 单测；pass 数以合入 main 时冻结值为准，测试不过不许继续
python -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/tests/test_push_robot_events.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  tests/test_run_phase1_push_robustness_queue.py

# 人话：打印 12 臂计划表（每臂 人话 + run_name + push 键值 + launch_order 序号）；这是唯一队列表
python3 scripts/run_phase1_push_robustness_queue.py --plan

# 人话：打印发射前依赖核对单；下面第 2–5 步就是逐条执行它
python3 scripts/run_phase1_push_robustness_queue.py --checklist
```

## 2. 冻结 exact commit（渲染解锁的唯一钥匙）

人话：config 里 `source.commit` 出厂是占位符 `PENDING_EXACT_COMMIT`，renderer 在占位符下拒绝
渲染任何 SSH 命令。键面已统一为五键幅度面（renderer 单测逐字交叉断言 train.py `_PUSH_KEYS`），把 push
覆盖键实现（trainer 白名单＋EventTerm 接线＋单测）合入 main 后，把该字段改成那次合并的 40 位
exact commit 并重新过一遍单测；不得填"当时的 HEAD"以外的任何近似值。**同时 diff 本波 commit 与矩阵 queue commit（`efc50f3f...27d1`）之间的训练路径**：
无行为性改动则对照有效并把 diff 结论记进实验记录；有行为性改动则对照失效，必须停下重议。

## 3. Pod clone 同步 exact commit

人话：两个 Pod 各建一个干净的、钉在 exact commit 上的独立 checkout；不能复用别的实验的旧
目录，也不能带本地改动。渲染出的每条远端命令还会自检一遍这三条，不满足就当场退出。

```bash
# 在每个 Pod 上（Pod1 与 Pod2 各做一次）：
git -C /workspace/codexschema/nohope_push_20260721 fetch origin
git -C /workspace/codexschema/nohope_push_20260721 checkout --detach <EXACT_COMMIT>
git -C /workspace/codexschema/nohope_push_20260721 status --porcelain   # 人话：必须输出为空
```

## 4. 资产、parent SHA 与 push 键核对

人话：训练要用的忽略资产、两个 parent checkpoint、push 新键和 boot lock 都先核对，错一样都
不发射（对应 `--checklist` 相应条目）。

```bash
# 人话：push 键是并行新实现——grep 远端 train.py 的 task.push 白名单（_PUSH_KEYS），
# 确认它与冻结 commit 的 queue config 里 12 臂 override 用的键面逐字一致
# （统一冻结面：enable / interval_range_s / vel_xy_mps / ang_vel_radps / ang_axes）。
# 主控合并时必须统一；grep 不一致＝键面没统一，全部臂不发
ssh <pod> 'grep -n "task.push\|_PUSH_KEYS" \
  /workspace/codexschema/nohope_push_20260721/hope_training/whole_body_tracking/scripts/train.py'

# 人话：五条资产路径逐一存在（USD、正反手动作、题库、A3 资产树；路径与矩阵 config 相同）
# 人话：parent checkpoint 指纹必须等于 renderer 钉死的期望值——
#   W model_6700 期望 2caab3dde3a0ac6c051ff8ac65385a641cac152aa3f84b640126b5ed7b96fcce
#   V model_6700 期望 ad9019100f199f23669829b0fbc4f8c2ad45c8073f930348f177da9487332716
ssh <pod> 'sha256sum <W_model_6700.pt> <V_model_6700.pt>'

# 人话：boot 串行锁必须存在且可执行；本轮不用 180 s stale 门的 locked launcher
ssh <pod> 'test -x /workspace/bin/kit_boot_lock.sh && echo LOCK_OK'

# 人话：namespace 必须全新（no-clobber）；12 个 run_dir 与 12 个 probe dir 都不得已存在
ssh <pod> 'ls /workspace/codexschema/phase1_push_robustness_20260721 2>&1'
```

## 5. 空槽认领：单一队列，不抢在跑矩阵

人话：本波不建专属 GPU 池、不给 pod/卡分角色。矩阵格停格（三类停止规则）或跑满 `10001`
updates 自然完结时释放槽位；出现空槽就把 config `launch_order` **最前面的未发臂**填进去
（父本与档位交错，前 6 臂覆盖全部 6 档）。当前已知的第一个空槽是矩阵 `v_h_s2` 封格后的
Pod2 GPU2 槽位。

```bash
# 人话：确认候选槽位所在卡当前 compute PID < 4（矩阵修正后的并发预检；4 条/卡是上限）
ssh <pod> 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv'

# 人话：确认要复用槽位的矩阵格确实已终结——它的 run_dir 里 exact PGID 已消失、
# 最后 checkpoint 已收档；不许对任何在跑矩阵格发信号来"腾槽"
```

铁律：**绝不抢占在跑矩阵格**；一次只认领一个空槽、发一臂；臂的实际落位（pod/gpu）以
claim/binding 实录为准并记回实验记录运行表。

## 6. 渲染并逐字执行 probe 命令

人话：某臂认领到空槽后，先渲染该臂的 2-update 完整场景探针命令，人工核对后逐字执行。probe
只证明"这臂配方能在真 4096-env scene 里跑起来并把账记对"，不是科学成绩。

```bash
# 人话：渲染单臂 probe 命令，把认领到的空槽用 --pod/--gpu 注入（示例＝launch_order 首臂
# w_p02 填进矩阵 v_h_s2 封格腾出的 Pod2 GPU2）；输出是逐字 SSH 命令文本
python3 scripts/run_phase1_push_robustness_queue.py \
  --render-stage probe --render-job w_p02 --pod pod2 --gpu 2
# 人话：执行前核对注释行的 job_id / 注入槽位 / run_name / 八个 push 键值与计划表一致
```

每条远端命令自带 fail-closed 前置检查（clean detached exact commit、资产存在、同卡 compute
PID < 4、run_dir no-clobber），随后经 `kit_boot_lock.sh` 串行 boot 并后台运行，立即返回
`launcher_pid`。

错峰纪律（人话：Kit 冷启动很脆，必须一个一个来）：

- 同 Pod 内 boot 串行（`kit_boot_lock` 持锁保证）；矩阵曾在 11 个并发 Isaac 运行时的冷启动
  窗口连死两次（`v_h_s2`），空槽刚出现时同 Pod 其余卡仍在满载训练，更要老实排队；
- 看到该臂首个 `Learning iteration` 后，才轮到同 Pod 下一臂；
- 相邻两次 launch 错峰 `≥60 s`（同秒启动会撞 CUDA 枚举）；
- probe 让它自然退出；不许手工 TERM/KILL 探针来伪造"通过"。

## 7. probe 收口：逐臂核对后才解锁该臂 science

人话：逐臂解锁，不攒收据集中开闸。某臂 probe 自然退出后按下面的单子核对，全过才允许渲染并
执行**该臂**的 science 命令；任何一条不过就停发该臂并排查。

```bash
PROBE_DIR=/workspace/codexschema/phase1_push_robustness_20260721/probes/<job_id>

# 人话：真的进入过学习循环（应出现 2 个 update 的 Learning iteration 行）
ssh <pod> "grep -n 'Learning iteration' $PROBE_DIR/run.log | tail -n 4"

# 人话：终档存在——probe 的自然退出证据
ssh <pod> "ls $PROBE_DIR/**/model_6701.pt"

# 人话：摘要抓异常不抓预期——WARN 行必须全部进摘要，Error/Traceback 一条都不能有
ssh <pod> "grep -nE 'WARN|Error|Traceback' $PROBE_DIR/run.log | tail -n 60"

# 人话：确认 q_des CLAMP ACTIVE 行存在——限位剪切默认开；缺这行＝有人显式关了限位剪切
ssh <pod> "grep -Fc 'q_des CLAMP ACTIVE' $PROBE_DIR/run.log"

# 人话：push 配置回显必须在 applied 行里出现——间隔与全部幅度区间都要与计划表一致；
# 对照矩阵配方的三机制 probe ledger tag 也要照常落盘
ssh <pod> "grep -nE 'push|settle_debt|lower_body|processed_qdes' $PROBE_DIR/run.log | tail -n 30"
```

push 特有注意：probe 只有约 `0.96 s` 仿真时间——`5–15 s` 间隔臂在 probe 内**预期零次
push 触发**，`1–3 s` 的 `fast` 臂也只会有零星触发。所以 probe 只核对配置回显与 ledger tag
存在，**不得**把"没推过"当失败，也不得把触发计数当机制证据；触发量级留到 science 的
`6900/7200/7700` 观察点核对（16 s 局长、`5–15 s` 间隔下每 env 每局期望 1–3 次）。

```bash
# 人话：该臂 probe 全过后，渲染并逐字执行该臂的科学长训命令；--pod/--gpu 注入与该臂
# probe 相同的认领槽位
python3 scripts/run_phase1_push_robustness_queue.py \
  --render-stage science --render-job <job_id> --pod <pod1|pod2> --gpu <0|1|2>
```

## 8. 长训监控与摘要纪律

人话：监控只读不写；摘要必须抓异常而不是只抓预期信号，WARN 必进摘要。

```bash
# 人话：看六卡占用与 compute PID（矩阵+push 混跑时每卡仍应 ≤4 条 trainer）
ssh <pod> 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'

# 人话：看某臂训练进度（最新 iteration 行）
ssh <pod> 'grep -n "Learning iteration" <run_dir>/run.log | tail -n 3'

# 人话：异常摘要（WARN|Error|Traceback）+ q_des CLAMP ACTIVE 存在性，与 probe 同一纪律
ssh <pod> 'grep -nE "WARN|Error|Traceback" <run_dir>/run.log | tail -n 50'
ssh <pod> 'grep -Fc "q_des CLAMP ACTIVE" <run_dir>/run.log'

# 人话：列 checkpoint，确认 save/100 与最新里程碑
ssh <pod> 'ls <run_dir>/logs/rsl_rl/*/model_*.pt | tail -n 5'
```

观察点纪律：`6900/7200/7700` 只判崩溃/合同/接线（含 push 触发计数量级）；`8700/10700` 看
中段；`12700/16700` 才有完整单 seed 结论；push 臂早期 fall/completion 比对照难看是**预期**
（它们在被推的环境里训练），不构成停臂理由；不因稀疏 reward 早期为零而停。stale 判定以
`1800 s`（boot）/`900 s`（首迭代后）为准；基础设施失败按 `_r2` 规则最多逐字重发一次，同
phase 二次失败转根因线，原 namespace 永久只读。停臂只允许实验记录冻结的三类停止规则。

## 9. Judge 步骤（终点同卷，主判据，仍是诊断）

人话：`model_16700` 终档后，对每臂与其 matched 矩阵对照（`w_c_s0`/`v_c_s0` 同 milestone
终档）跑同一张 K100 考卷。**考卷内没有 push**——这是唯一把"环境更难"与"策略更脆"分开的
比较面；因 W/V 谱系是 diagnostic，结果只用于筛选，不授权部署。

```bash
# 人话：标准判卷入口；自动解析 env.yaml 的动作对/题库并出双侧考卷
bash hope_training/whole_body_tracking/scripts/judge.sh <run_dir> <run_dir>/.../model_16700.pt
```

- 用 signed-face composite 的同一张 immutable K100 卷（100 题、正反手各 50），不得混卷；
- judge 子进程与训练共用 Kit boot 锁；先确认该 Pod 无正在 boot 的臂再判卷；
- 假设 1–3（最优幅度/角速度收益/高频 vs 低频）全部在这张卷上按实验记录冻结的门裁决；
- 胜者剂量正名须到 exact-lineage 链（当前唯一 exact P0 候选：qdot treatment `model_1000`
  一族，见 NOW）重跑，不给本轮后代补 seed。

## 10. 收口清单

人话：一臂结束（自然终档或停臂）都按同一张单子收口，不留悬空进程和无主证据。

1. 重验该臂 `run_dir` 下 `source_commit.txt`、`launcher.out` 与真实 PID/PGID/starttime/argv；
   确需停臂时按 [RunPod 纪律](run_on_runpod.md#已登记-phase-1-实验臂的算力释放)对 exact PGID
   先 TERM 再 KILL；禁止 `pkill -f`、`killall`；
2. 确认 exact PGID 消失、认领槽位无本臂 compute PID、`kit_boot_lock` 无 holder；
3. 停臂必须在实验记录写明原因（NaN/OOM、lineage 错误、同 parent 被支配 ≥4000 iter 三类
   之一）与最后 checkpoint；
4. 收档：run.log、launcher.out、source_commit.txt、terminal checkpoint SHA-256、probe 核对
   记录、该臂实际落位（pod/gpu）与认领时刻；
5. 释放的槽位回到单一队列：还有未发臂就发计划表最前的下一臂，没有就留给下一轮预注册；
6. 文档同步：实验记录运行表与结果、`PROGRESS.md` 一条带日期摘要、长训启动/成绩/owner 变化
   时更新最新 main 的 `NOW` 统一队列；重要结论进 main 才动 `TIMELINE`；
7. `_r2` retry 只发一次且逐字同配方；同 phase 二次失败转根因线。

本页没有任何真机命令。G07 安全门闭合前不得把训练 argv 改成部署或真实机器人控制。
