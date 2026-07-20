# Phase 1 时序×稳定 24 格矩阵：从零到发射

本页只操作 24 格 `{W,V} × {N,C,H} × {S0,S1,S2,S3}` 矩阵的仿真训练队列。实验真源是
[EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720](../experiments/2026-07/EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md)；
共享代号人话见[术语表](../DEFINITIONS.md)。本页不授权真机、部署、第二 seed 或 M0 横移老师。

入口与不变量（人话：这轮用哪些文件、哪些数字永远不变）：

- config：`configs/phase1_balance_temporal_matrix_20260720.yaml`（24 格唯一机器队列）
- renderer：`scripts/run_phase1_balance_temporal_matrix_queue.py`（lean runner：只打印计划表/
  核对单/逐字 SSH 命令，自己绝不 SSH、绝不发信号、绝不写远端）
- 本轮**没有** manifest-SHA 多层审批链：按发射工序教训改用统一计划表（`--plan`）＋依赖核对单
  （`--checklist`）＋逐字渲染人工核对；parent checkpoint SHA-256 直接钉死在 renderer 里
- 远端 checkout：`/workspace/codexschema/nohope_btm_20260720`（clean detached exact commit）
- namespace：`/workspace/codexschema/phase1_balance_temporal_matrix_20260720`
  （probe 在 `probes/<job_id>`，science 在 `runs/<job_id>`，no-clobber）
- 每格：`4096 environments`、`seed=3`、从各自 parent `model_6700` 续 `10001` updates、save/100；
  probe 每格 `2` updates 自然退出到 `model_6701.pt`
- 发射器：远端用 `/workspace/bin/kit_boot_lock.sh` + `setsid nohup`；**不用**
  `launch_kit_training_locked.sh`（其 `180 s` stale 门是 Wave A v8/v9 死因）
- watchdog：boot 停滞 `1800 s`、首个 iteration 后停滞 `900 s` 才判死；唯一允许的重试是逐字
  重发一次、run_name 加 `_r2` 后缀
- 谱系：`checkpoint_allow_contract_mismatch=true`，全部后代只作诊断、formal-exact-ineligible

不要手改 rendered SSH argv、run root、GPU 或权重；要变化就改 config/tests/实验记录后重新复核。

## 0. 核对 main 权威与 NOW 认领

人话：确认本地就是最新 `origin/main`、队列文件已进 main、统一队列 NOW 里有这轮的
责任人/执行者/分支/queue id 认领行。任一失败只许跑本地 validate/plan，不许执行任何 SSH。

```bash
BTM_ROOT="$(git rev-parse --show-toplevel)"
cd "$BTM_ROOT"

git fetch origin main                     # 人话：取最新 main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"   # 人话：本地必须等于 origin/main
test -z "$(git status --porcelain)"       # 人话：工作树必须干净
git cat-file -e origin/main:configs/phase1_balance_temporal_matrix_20260720.yaml
git cat-file -e origin/main:scripts/run_phase1_balance_temporal_matrix_queue.py
git show origin/main:docs/NOW.md | grep -n "phase1_balance_temporal_matrix_20260720"
# 人话：NOW 的同一条认领行必须同时含 责任人 franco、执行者 与本 queue id；缺任一即失败
```

功能分支里的 `NOW` 不是授权。全局顺序与算力归属只看最新 `origin/main:docs/NOW.md`；本队列只
定义局部依赖 `本地验证 → 冻结 exact commit → 核对单全过 → 逐格 probe → 逐格 science`。

## 1. 本地验证、计划表与核对单

人话：先跑本轮相关的本地单测，再让 renderer 打印 24 格计划表和发射前核对单；默认调用不生成
任何远端命令。

```bash
# 人话：跑 S1/S2/S3 reward、schema-3 合同与 24 格队列渲染器的 focused 单测；
# pass 数以合入 main 时冻结值为准，测试不过不许继续
python -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/tests/test_lower_body_stability_wave.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  tests/test_run_phase1_balance_temporal_matrix_queue.py

# 人话：打印 24 格计划表（每格 人话 + run_name + pod/gpu + 五个权重）；这是唯一队列表
python3 scripts/run_phase1_balance_temporal_matrix_queue.py --plan

# 人话：打印发射前依赖核对单；下面第 2–4 步就是逐条执行它
python3 scripts/run_phase1_balance_temporal_matrix_queue.py --checklist
```

## 2. 冻结 exact commit（渲染解锁的唯一钥匙）

人话：config 里 `source.commit` 出厂是占位符 `PENDING_EXACT_COMMIT`，renderer 在占位符下拒绝
渲染任何 SSH 命令。主控把全部并行实现（含 S1 的 trainer 键）合入 main 后，把该字段改成那次
合并的 40 位 exact commit 并重新过一遍单测；不得填“当时的 HEAD”以外的任何近似值。

## 3. Pod clone 同步 exact commit

人话：两个 Pod 各建一个干净的、钉在 exact commit 上的独立 checkout；不能复用别的实验的旧
目录，也不能带本地改动。渲染出的每条远端命令还会自检一遍这三条，不满足就当场退出。

```bash
# 在每个 Pod 上（Pod1 与 Pod2 各做一次）：
git -C /workspace/codexschema/nohope_btm_20260720 fetch origin
git -C /workspace/codexschema/nohope_btm_20260720 checkout --detach <EXACT_COMMIT>
git -C /workspace/codexschema/nohope_btm_20260720 status --porcelain   # 人话：必须输出为空
```

## 4. 资产、parent SHA 与 trainer 键核对

人话：训练要用的忽略资产、两个 parent checkpoint、S1 新键和 boot lock 都先核对，错一样都不
发射（对应 `--checklist` 第 2–7 条）。

```bash
# 人话：S1 是并行新实现——grep 远端 train.py 确认 post_swing_settle_debt_weight、
# post_swing_settle_debt_probe_weight 及全部 post_swing_settle_debt_* 参数键已进
# _REWARD_KEYS 白名单；缺一个则所有 S 档 boot 即 fail-loud，全部不发
ssh <pod> 'grep -n "post_swing_settle_debt" \
  /workspace/codexschema/nohope_btm_20260720/hope_training/whole_body_tracking/scripts/train.py'

# 人话：五条资产路径逐一存在（USD、正反手动作、题库、A3 资产树）
# 人话：parent checkpoint 指纹必须等于 renderer 钉死的期望值——
#   W model_6700 期望 2caab3dde3a0ac6c051ff8ac65385a641cac152aa3f84b640126b5ed7b96fcce
#   V model_6700 期望 ad9019100f199f23669829b0fbc4f8c2ad45c8073f930348f177da9487332716
ssh <pod> 'sha256sum <W_model_6700.pt> <V_model_6700.pt>'

# 人话：boot 串行锁必须存在且可执行；本轮不用 180 s stale 门的 locked launcher
ssh <pod> 'test -x /workspace/bin/kit_boot_lock.sh && echo LOCK_OK'

# 人话：namespace 必须全新（no-clobber）；24 个 run_dir 与 24 个 probe dir 都不得已存在
ssh <pod> 'ls /workspace/codexschema/phase1_balance_temporal_matrix_20260720 2>&1'

# 人话：每张目标卡零 compute 进程后才允许发射该卡
ssh <pod> 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv'
```

## 5. 渲染并逐字执行 probe 命令

人话：先渲染 24 格的 2-update 完整场景探针命令，人工核对后逐字执行。probe 只证明“这格配方
能在真 4096-env scene 里跑起来并把账记对”，不是科学成绩。

```bash
# 人话：渲染单格（推荐首格先单发）或全部 probe 命令；输出是逐字 SSH 命令文本
python3 scripts/run_phase1_balance_temporal_matrix_queue.py --render-stage probe --job w_n_s0
python3 scripts/run_phase1_balance_temporal_matrix_queue.py --render-stage probe --job all
# 人话：执行前逐格核对注释行的 job_id / pod / gpu / run_name 与计划表一致
```

每条远端命令自带 fail-closed 前置检查（clean detached exact commit、资产存在、GPU 零 compute、
run_dir no-clobber），随后经 `kit_boot_lock.sh` 串行 boot 并后台运行，立即返回 `launcher_pid`。

错峰纪律（人话：Kit 冷启动很脆，必须一个一个来）：

- 同 Pod 内 boot 串行（`kit_boot_lock` 持锁保证），两个 Pod 可以并行各 boot 一格；
- 看到该格首个 `Learning iteration` 后，才轮到同 Pod 下一格；
- 相邻两次 launch 错峰 `≥60 s`（同秒启动会撞 CUDA 枚举）；
- probe 让它自然退出；不许手工 TERM/KILL 探针来伪造“通过”。

## 6. probe 收口：逐格核对后才解锁该格 science

人话：本轮是逐格解锁，不攒 24 份收据集中开闸。某格 probe 自然退出后按下面的单子核对，全过
才允许渲染并执行**该格**的 science 命令；任何一条不过就停发该格并排查。

```bash
PROBE_DIR=/workspace/codexschema/phase1_balance_temporal_matrix_20260720/probes/<job_id>

# 人话：真的进入过学习循环（应出现 2 个 update 的 Learning iteration 行）
ssh <pod> "grep -n 'Learning iteration' $PROBE_DIR/run.log | tail -n 4"

# 人话：终档存在——probe 的自然退出证据
ssh <pod> "ls $PROBE_DIR/**/model_6701.pt"

# 人话：摘要抓异常不抓预期——WARN 行必须全部进摘要，Error/Traceback 一条都不能有
ssh <pod> "grep -nE 'WARN|Error|Traceback' $PROBE_DIR/run.log | tail -n 60"

# 人话：确认 q_des CLAMP ACTIVE 行存在——限位剪切默认开；缺这行＝有人显式关了限位剪切
ssh <pod> "grep -Fc 'q_des CLAMP ACTIVE' $PROBE_DIR/run.log"

# 人话：三机制 probe ledger 真的落盘（S 档 enabled 计数须与该格 weight 一一对应；
# 完整 update 的 observed 期望 4096×24=98304，恢复窗资格分母允许单 update 为零但两步合计非零）
ssh <pod> "grep -nE 'settle_debt|lower_body|processed_qdes' $PROBE_DIR/run.log | tail -n 20"
```

```bash
# 人话：该格 probe 全过后，渲染并逐字执行该格的科学长训命令
python3 scripts/run_phase1_balance_temporal_matrix_queue.py --render-stage science --job <job_id>
```

science 铺满后每卡恰好 4 条 trainer（同一 `(parent,T)` 的四个 S 档共卡）。发射后把每格实际
落位与 `launcher_pid`、`source_commit.txt` 记回实验记录运行表。

## 7. 长训监控与摘要纪律

人话：监控只读不写；摘要必须抓异常而不是只抓预期信号，WARN 必进摘要。

```bash
# 人话：看六卡占用与 compute PID（满载时每卡应稳定 ~4 条 trainer）
ssh <pod> 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'

# 人话：看某格训练进度（最新 iteration 行）
ssh <pod> 'grep -n "Learning iteration" <run_dir>/run.log | tail -n 3'

# 人话：异常摘要（WARN|Error|Traceback）+ q_des CLAMP ACTIVE 存在性，与 probe 同一纪律
ssh <pod> 'grep -nE "WARN|Error|Traceback" <run_dir>/run.log | tail -n 50'
ssh <pod> 'grep -Fc "q_des CLAMP ACTIVE" <run_dir>/run.log'

# 人话：列 checkpoint，确认 save/100 与最新里程碑
ssh <pod> 'ls <run_dir>/logs/rsl_rl/*/model_*.pt | tail -n 5'
```

观察点纪律：`6900/7200/7700` 只判崩溃/合同/接线；`8700/10700` 看中段；`12700/16700` 才有完整
单 seed 结论；不因稀疏 reward 早期为零而停。stale 判定以 `1800 s`（boot）/`900 s`（首迭代后）
为准；基础设施失败按 `_r2` 规则最多逐字重发一次，同 phase 二次失败转根因线，原 namespace
永久只读。

## 8. Judge 步骤（终点同卷，仍是诊断）

人话：`model_16700` 终档后，对领先格与其 matched 对照跑同一张 K100 考卷；因 W/V 谱系是
diagnostic，结果只用于筛选，不授权部署。

```bash
# 人话：标准判卷入口；自动解析 env.yaml 的动作对/题库并出双侧考卷
bash hope_training/whole_body_tracking/scripts/judge.sh <run_dir> <run_dir>/.../model_16700.pt
```

- 用 signed-face composite 的同一张 immutable K100 卷（100 题、正反手各 50），不得混卷；
- judge 子进程与训练共用 Kit boot 锁；先确认该 Pod 无正在 boot 的格再判卷；
- 胜者机制正名须到 exact-lineage 链（当前唯一 exact P0 候选：qdot treatment `model_1000`
  一族，见 NOW）重跑，不给本轮后代补 seed。

## 9. 收口清单

人话：一格结束（自然终档或停格）都按同一张单子收口，不留悬空进程和无主证据。

1. 重验该格 `run_dir` 下 `source_commit.txt`、`launcher.out` 与真实 PID/PGID/starttime/argv；
   确需停格时按 [RunPod 纪律](run_on_runpod.md#已登记-phase-1-实验臂的算力释放)对 exact PGID
   先 TERM 再 KILL；禁止 `pkill -f`、`killall`；
2. 确认 exact PGID 消失、assigned GPU 无 compute PID、`kit_boot_lock` 无 holder；
3. 停格必须在实验记录写明原因（NaN/OOM、lineage 错误、同 parent 同 T 档被支配 ≥4000 iter
   三类之一）与最后 checkpoint；
4. 收档：run.log、launcher.out、source_commit.txt、terminal checkpoint SHA-256、probe 核对记录；
5. 文档同步：实验记录运行表与结果、`PROGRESS.md` 一条带日期摘要、长训启动/成绩/owner 变化
   时更新最新 main 的 `NOW` 统一队列；重要结论进 main 才动 `TIMELINE`；
6. `_r2` retry 只发一次且逐字同配方；同 phase 二次失败转根因线。

本页没有任何真机命令。G07 安全门闭合前不得把训练 argv 改成部署或真实机器人控制。
