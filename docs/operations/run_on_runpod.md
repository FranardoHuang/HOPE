# Run Training On The Shared RunPod

Status: Active (pod provisioned 2026-07-01/02; verified with a full smoke + 10-iteration train on 2026-07-03)

The team currently has two RunPod endpoints. The primary shared training pod has 3× RTX 5090
(32 GB, sm_120), 128 cores, 1 TB RAM and a 500 GB persistent volume at `/workspace`. The live
source of truth for pod-side rules is
`/workspace/README_MULTIUSER.md` **on the pod** — read it after logging in; this file records how
to get on, what is verified, and the repo-side conventions.

## Access

```bash
# primary
ssh root@162.43.172.171 -p 18333 -i ~/.ssh/id_ed25519_runpod

# second endpoint (current 2026-07-10)
ssh root@162.43.172.181 -p 13146 -i ~/.ssh/id_ed25519_runpod
```

- Endpoint current as of 2026-07-04 (port history: 17424 → 15320 → 18333 across restarts; the key
  stays the same). Every restart confirmed `/workspace` survives intact; only running processes and
  `/root` are lost. RunPod may assign a new IP/port on any re-provision — update this file when it
  changes.
- **After EVERY restart, run `bash /workspace/restore_root.sh` once** — it re-wires `/root/.bashrc`
  (claude/codex/git-lfs on PATH for bare logins) and restores the GitHub key if a persistent one
  exists under `/workspace/.ssh_github/`. Then re-run the `--quick` smoke.
- Note: a full Stop→Start rebuilds the container but does NOT move hosts (the 500 GB volume is
  host-local and pins the pod to its machine) — verified 2026-07-04 via unchanged `/proc/uptime`.
- Each teammate (and each teammate's coding agent) uses the same root login; separation is by
  directory, not by account. Work ONLY under your own `/workspace/<name>/`.
- The pod holds one GitHub deploy key (dongc1's) shared for pull/push; per-clone `user.name`/
  `user.email` are already set so commits are attributed correctly.

### Source-only validation lane

Codex may use either endpoint for isolated Linux/ROS/C++/Python/Isaac contract
validation, but must not inspect, start, stop or schedule the training jobs it
does not own. Put the validation checkout under a separate
`/workspace/<name>/...` directory, force `CUDA_VISIBLE_DEVICES=''` for CPU
tests, and do not reuse a live run directory. This boundary still permits
compiler and Isaac Lab source inspection; it does not permit “checking how
training is going”.

2026-07-10 source-only acceptance, with no GPU or training-process access:

- primary pod GCC 13 Release: 188 passed/4 skipped portable, 202 passed/4
  skipped with ROS 2 Jazzy; runner/runtime probe built;
- second pod: whole-body source suite 435 passed/4 optional-asset skips;
  planner 107 passed;
- the Release build exposed and closed the global `-ffast-math` NaN/Inf
  safety failure. Reproducible commands are in
  [build_and_test.md](build_and_test.md).

### Phase-1 same-paper evaluator lane (2026-07-11)

The user explicitly authorized the current Codex task to inspect and control
the Phase-1 evaluator jobs on both endpoints. The first audit found cc's old
prototype `eval_deterministic.py` processes spinning on a tuple observation
error and holding Kit resources, plus stale watchdog/export shells. Their exact
process groups were terminated with `SIGTERM`; no unrelated training process
was killed. Both pods then showed all six GPUs at `P8`, `0 MiB`, with no Kit
cache-lock holder. This cleanup is operational evidence only, not a model
score.

Run the current evaluator from an isolated `/workspace/<name>/nohope` clone and
store generated banks/schedules/ledgers under that user's persistent directory.
Do not modify a historical run directory: it is an input artifact. Before each
Isaac launch, repeat the process/GPU/cache-lock audit, obey the pod-wide
one-Kit-boot rule, and verify the result by strict JSON/CSV artifacts rather
than exit code alone. Use the same schedule JSON for the MuJoCo companion cell;
the commands and diagnostic/exact boundary are in
[run_training.md](run_training.md#shared-schema-v3-bankexam-isaac--mujoco).

The historical M3f/M2/G1 canary must pass the explicit inexact flag in both
simulators and remain non-bookable. A clean schedule/question-order match is a
ruler acceptance test, not permission to rename those checkpoints exact.
The CPU BankExam now locates its standalone question-bank loader from the
checkout; it needs no Isaac task-package import and no manually exported
`HOPE_STAGE1_QB`. A temporary diagnostic install of `toml==0.10.2` was made in
Pod1's mjeval venv on 2026-07-11, but it is not a reproducible dependency and
is not required on a fresh Pod.

## Layout And Per-User Convention

```
/workspace/                       # 500 GB persistent volume (survives pod restart)
├── hope_isaac_venv/              # SHARED venv: Isaac Sim 4.5 + Isaac Lab 2.1 + torch 2.7.0 cu128 — read-only by convention
├── IsaacLab/                     # SHARED Isaac Lab 2.1.0 source — do not edit
├── shared/motions/, shared/assets/   # shared motion clips + generated agibot_a3 assets
├── <name>/nohope                 # your own clone/branches; GPU lane comes from NOW/experiment queue, never a permanent owner map
├── <name>/env.sh                 # source in EVERY shell; any CUDA pin is the current lane, not ownership
├── setup_user.sh                 # provision a new user: bash setup_user.sh <name> <gpu> [git_name] [git_email]
├── smoke_test/                   # environment smoke suite + logs
└── README_MULTIUSER.md           # pod-side rules (authoritative)
```

Every shell session:

```bash
source /workspace/<name>/env.sh   # activates venv, puts YOUR clone first on PYTHONPATH, may pin the currently assigned GPU
cd $HOPE_WBT                      # = /workspace/<name>/nohope/hope_training/whole_body_tracking
```

The shared venv deliberately does NOT contain `whole_body_tracking`: without sourcing your
`env.sh`, imports fail loudly instead of silently running someone else's branch.

Training (same commands as `run_training.md`, e.g.):

```bash
python scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=<your clip.npz> motion_file_2=<your clip2.npz> \
  logger=tensorboard run_name=<name>_<experiment>
```

For the optional R15 v5 arm only, first copy `/workspace/shared/motions/hope_forehand_v5.npz` and `hope_backhand_v5.npz` into your clone's ignored `hope_training/motions/preprocessed/`, then use the explicit override command in `run_training.md`. Do not switch `cfg/train.yaml`, `cfg/play.yaml`, or task YAML defaults to v5.

## Smoke Test

After a pod restart, an env change, or provisioning a user:

```bash
bash /workspace/smoke_test/run_smoke.sh <name>            # full: real env build + step
bash /workspace/smoke_test/run_smoke.sh <name> --quick    # fast: no env build
bash /workspace/smoke_test/run_smoke.sh <name> --train    # + 10-iteration real training
bash /workspace/smoke_test/run_smoke.sh --concurrent      # all 3 GPUs at once (after pod changes)
```

Logs land in `/workspace/smoke_test/logs/`. The suite checks torch/numpy pins, real CUDA kernels on
each 5090, venv isolation, dependency drift vs `freeze_baseline.txt`, Kit boot, task registration,
that imports come from YOUR clone, and a real env build + step.

## Read-Only Terminal-Checkpoint Inventory

For historical Phase-1 runs, inventory terminal checkpoints and sidecars
without launching, resuming, judging or deleting anything:

```bash
cd "$HOPE_WBT"
python scripts/audit_runpod_terminal_runs.py \
  --run-root logs/rsl_rl/agibot_a3_hope_virtualball \
  --judge-steps 15000 --judge-gpu 0
```

The built-in map requires exactly one directory for each of the eleven known
arms and fails on missing or duplicate matches.  Use `--arm LABEL=REGEX` only
after auditing a renamed tree.  The script's `DRY-RUN` and `JUDGE` lines are
printed instructions, not executed commands.  Current work does not authorize
restoring old training; any later artifact recovery must first pass the
schema-v3 canary and follow `NOW.md`.

## 已登记 Phase-1 实验臂的算力释放

只有人类负责人已经明确作出资源决定、且该臂属于当前任务获授权管理的范围时，才执行本节。它是
进程所有权与证据保全流程，不是 q10 统计停止规则；`screen_only=true` 仍表示 q10 不能晋级，既有
q50 合同中的 `whole_arm_stop_allowed=false` 也不会被本节改写。

1. 从该臂的 `run.log.launch` 或经审计的 launch sidecar 读取 exact PID/PGID；同时核对 run name、
   命令、训练 checkout 与预期 arm 一致。不得用命令行模式搜索结果代替所有权 sidecar。
2. 保存完整日志和最新 checkpoint 路径/SHA。验证文件名迭代号等于 checkpoint 内嵌迭代号、全部
   tensor finite、schema/lineage 正确，并确认相邻 `params/training_contract.json` SHA 与 checkpoint
   内嵌值一致。任何一项失败都先保全和诊断，不发信号。
3. 用进程表按**数值 PGID**列出该组，确认没有不属于本臂的 live child；检查共享 Kit lock 及其
   holder，不删除任何 live holder 的锁。仍有 judge/worker/Kit 所有权不清时停止操作。
4. 先向 exact group 发送 `TERM`，再按数值 PGID 复核。只有证据已落盘、组内成员仍属于本臂、
   没有 live child/Kit-lock holder 且 TERM 确实不退出时，才允许向同一 exact group 发送 `KILL`。
5. 最后验证该 PGID 已消失、checkpoint/log 没有被删除，且所有其他接受臂、worker、judge 和 GPU
   状态仍符合清单。记录信号、最后 checkpoint、SHA 和继续运行的臂。

Linux 命令骨架如下；`PGID` 必须来自已核对 sidecar，不能手填猜测：

```bash
PGID=<exact_numeric_pgid_from_launch_sidecar>
ps -eo pid=,ppid=,pgid=,sid=,stat=,cmd= | awk -v g="$PGID" '$3 == g'
fuser /workspace/.cache/ov/_cache.lock || true
kill -TERM -- "-$PGID"
# 复核上述证据和组成员；只有全部条件仍成立且 TERM 不退出时：
kill -KILL -- "-$PGID"
```

严禁 `pkill`、`killall`、`pgrep -f` 后批量发信号或任何 broad pattern kill。真实机器人进程不在本节
授权范围内。2026-07-13 分两波停止全部 16 条 fresh 广度臂的实际记录见
[Phase-1 拍面×plant 广度实验](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。
第二波额外产出并入库两份 no-clobber pre-stop audit，绑定进程组成员、launch/log SHA、
最新 checkpoint 迭代/SHA/finite 与 checkpoint↔相邻合同。TERM 未退出时，只在 Kit lock
无 holder、进程组只含 trainer/git helper 后向同一 exact PGID 发 KILL。两 Pod 最终
无 trainer、GPU 0 MiB/0%，train/eval checkout 仍 clean exact。与已停 trainer 对应的四个等待型
fresh curve worker 在 childless/lock-free 复核后也只按各自 PGID TERM 退出。

<a id="2026-07-20-action-slew-wave-a-启动前状态与发射纪律"></a>
## 2026-07-20 action-slew Wave A 启动前状态与发射纪律

这里的 [Wave A](../DEFINITIONS.md#balance-stability-waves) 只指 W/V action-slew 六格；下半身 Wave B
不共享本节的发射授权。M0 四份 moving exact-GMR 已因 `stance_passed=0/4` 且 formal/schema2/training/hardware
均未授权而输入门拒绝，禁止把它们填到空闲 GPU；immediate Wave B 只保留静态 v4rg 或 non-demo 设计。

2026-07-20 的只读快照中，两 Pod 的六张 RTX 5090 都没有 NVML compute application，显存为 `0 MiB`、
利用率为 `0%`；这里的“空”只指 GPU trainer/Kit compute，不替代对普通 shell、ROS 或其他无 GPU 进程的
所有权审计。这个快照会过期，每次发射前必须重新查 `nvidia-smi`、exact process group 和 Kit lock。

同日首次预检说明“GPU 空”还不等于 ready；随后按 no-clobber staging 和 exact-PGID 规则闭合了输入与残留：

| 发射输入/主机状态 | 首次发现 | 2026-07-20 闭合态 | 仍须由 launch gate 重验 |
| --- | --- | --- | --- |
| clean detached source C1 `54c9a626…3906` | 两台 queue checkout 均缺 | 两台 exact checkout 已恢复 | clean/detached HEAD 与 required-file SHA |
| W/V parent 完整 run | Pod1 完整；Pod2 两份均缺 | Pod2 W/V 已完整 staged 到 queue exact 路径 | 两个 checkpoint、contract 与四类 finite state；remote manifest gate 会在每台都验 W/V |
| preconverted A3 USD | Pod1 6-file bundle 完整；Pod2 整包缺失 | Pod2 完整 6-file bundle 已 staged | `model.usd` SHA 加 bundle file-count/bytes/tree SHA；不能只验单文件 |
| 正/反手动作与 schema-3 train bank | 两台 SHA 一致 | 未改动 | manifest 仍逐项绑定 |
| A3 runtime asset 候选树 | 两台均为 46 files / `15,378,264 B`，tree SHA `a9512461…a90` | source exact path 已恢复 | 在 queue exact path 重算，不能借用候选路径结论 |
| 普通进程审计 | Pod1 有 PGID `2010084`、`2010190`、`2010154`，其中 planner 独立在 `2010190` | 三个 exact PGID 已按身份对账停止并复核 absent | 每次发射仍重查 PID/PGID/starttime/argv，不能从历史 absent 外推 |

最初这次闭合只恢复了发射输入，没有启动 probe；随后唯一启动的 Pod1 W-C probe2 自然退出，但被错误的
逐-update recovery 非零门假拒绝，未产生 receipt。不得把 staging 或该假拒绝签成科学结果，也不得先启动
再补 SHA。probe3 的 W-C/W-N/V-C 后来均自然退出，但 verifier 未绑定 24-step 完整 rollout，故其中 W-C
旧 receipt 和另两格 runtime 都不能解锁 train。probe4 W-C 又因命令错写
`algo.num_steps_per_env=24` 而在远端 Hydra compose 门 fail closed；该门位于建 run-dir 与 Kit launcher 之前，
所以 v3 root/log/checkpoint/process/GPU compute 全部不存在。

probe5（第五次、只检查合同而不作科学比较的两-update 探针）改用真实的
[`algo.runner.num_steps_per_env=24`](../DEFINITIONS.md#ppo-num-steps-per-env)，表示每个环境每次 PPO update
收集 24 step，并在 v4 根中发布了 W-C、V-C、W-N、V-N、V-H 五份 exact receipt。W-H 在 trainer binding
之前 fail closed：旧 supervisor 只读一次 `Popen` child 身份，恰好采到 fork→exec 过渡态 argv，误判为
trainer identity mismatch。该格没有 first iteration、trainer binding、RSL checkpoint、receipt 或 GPU
training compute；失败后 exact leader PGID/child PID 均 absent。因此它是启动取证竞态，不是 H 机制负例。
probe5 的 v4 manifest 文件 SHA-256 是
`6bfa73587968f8f0af71b5617e8c324f75114b304bbe1452d0b0e4617d1f51bc`，content SHA-256 是
`093a4cc7a0ce91aad74948ed39b581e5f4a0693ba114f3144062b6cc4386a462`；五份 receipt 与
`/workspace/codexschema/phase1_balance_action_slew_v4_20260720` 都冻结为历史，不能补写，也不能与新一轮混用。

probe6（第六次同类合同探针）只发射了 W-C。`2026-07-20T03:01:10Z`，生成的 supervisor Python 在
locked launcher child bind 前以 `IndentationError: unexpected indent` 退出。根因是 failure transaction
helper 对已经 shlex-quoted 的 multiline program 每个换行后加了两个空格，改变了 transaction body bytes。
该格没有 leader/child evidence、binding、terminal marker、checkpoint 或 receipt；PID=PGID `2712318` 经
stable double-scan 确认 absent，Pod1 GPU0 empty，Kit/cache locks free，审计没有发 signal。其余五格没有
发射，所以这是启动器生成缺陷，不是机制负例。v5 根
`/workspace/codexschema/phase1_balance_action_slew_v5_20260720` 与 manifest
文件/content SHA-256
`718bbee0a556cc3640ee636e20f8eb2adb293cee8d0bb1820afceebf5ce1a267` /
`3cbb019e9d315abdc687d1635c9deffc13eae9577ee2d820b0e3c30ba9b7cfd8` 冻结为不可变历史，禁止补写或重发。

probe7（第七次同类合同探针）中，W-C、V-C、V-N 都自然退出并通过 exact verifier receipt。W-N 在
`2026-07-20T03:22:50Z` 到达 RSL 后冻结于 `Starting the simulation`，没有 learning iteration、trainer
binding、terminal checkpoint 或 receipt。locked `180 s` watchdog 只向 exact 身份发送 TERM/KILL 并以
rc `125` 结束；随后 exact groups、Pod1 GPU1、Kit/cache locks 全部闭合。V-N 在 W-N 失败被确认前 `6 s`
已经发射，后来仍自然退出并验证通过；W-H/V-H 从未发射。这只能说明一次 transient infrastructure failure，
不能解释成 `action_rate=0`、W/V parent 或任何平滑机制的负例。v6 根
`/workspace/codexschema/phase1_balance_action_slew_v6_20260720`、三份 receipt、以及 config/runner 和 manifest
文件/content SHA-256
`912bd8d212791d99ce6a6851a8f05c12d182cdfa9d5566e02381f1b4703b8f3c` /
`3fbaf23f97fdb40e05a448f9f769267b21c7cca3bd767aa082c0d5b965ecd7d7` /
`4552fe23abd551d8959a9de05cc5f9d761d0da25eed88138d61fa45cc6558e9e` /
`6e3518d97d48fad550e7971a5178b1f11c15895696f03d30d5a62d1e27741640` 冻结为历史，禁止重试、补写或混用。

probe8 的 v7 只发射了 W-N。它在 `2026-07-20T03:44:19.857Z` 于 Pod1 物理 GPU1 到达
`Starting the simulation`（`sim.reset()` 边界），随后因 `malloc(): invalid size (unsorted)` 触发
`SIGABRT`，并于 `03:44:32.112Z` 结束；trainer terminal status 记录 exit `-6`，外层 transaction 返回 rc `134`。它没有 first
iteration、trainer binding、terminal checkpoint 或 receipt，其余 W-C/W-H/V-C/V-N/V-H 五格从未发射。
失败后 exact leader/child 均 absent、Pod1 全部 GPU 无 compute context、Kit/cache lock 无 holder；可访问的
系统日志与 telemetry 快照未见 Xid/OOM。v7 根 `/workspace/codexschema/phase1_balance_action_slew_v7_20260720`、失败证据与
config/runner 和 manifest 文件/content SHA-256
`0c84613f05439237f6e36d37e0c9210984465d928b9c0cba50999bd8995145f9` /
`2bc9d59e21413a812a742529c6f3291f5710c384e0f4af7ee7098f33b25ba17d` /
`887c0b9e097e50300d83eef27e587d112f70132958e6e8d9b68af74437fa7231` /
`13f92d5eda71e90abd6a14a1498c2afd98d3cb825cb26e2f5b74958b5a795f84` 冻结为不可变历史：禁止补写、同
namespace 重发或与后续收据混用。closure 闭合不是重试授权，该失败也不是 N 机制负例。
旧 manifest `d7e95130…a2e47`、`2d3e7955…3bae17`、`283fd002…1eefe` 及 v1/v2/v3 输出根也只作不可变历史。

probe9（第九次同类非科学合同探针）已在绑定 manifest 下严格串行完成 6/6，不再是待发射工作。fresh no-clobber
根为 `/workspace/codexschema/phase1_balance_action_slew_v8_20260720`，run name 固定为
`phase1_balance_slew_probe9_{job}_seed3_20260720`。queue config/runner bytes SHA-256 分别是
`c7ec75a9917b8bdcf7976186633b021c69fb82e591898dad6b8d5c93cfdb37d5` /
`24b5f7831ad49c2b88266fed65c37e6e4bcdddaacab28ffa917fce66ef918db1`；其
[`launch manifest`](../DEFINITIONS.md#balance-launch-manifest)
[`phase1_balance_action_slew_launch_manifest_20260720.json`](../../configs/phase1_balance_action_slew_launch_manifest_20260720.json)
文件 SHA-256=`688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353`，content SHA-256=
`97c36e471fb8fc6b93fe212f20846de6697db518192e7a45c6618e5924947e28`。清单存在当时只授权命令渲染；每格
真实发射前都由 remote preflight 对 exact 路径重算全部哈希、tree/count/bytes、source clean HEAD 与 GPU。

probe9 supervisor 在 `Popen` 后、首次 `/proc` 读取前先不可覆盖地发布 `trainer_child_evidence.json`，
再只跟踪其中同一个 child PID，最多等 `5 s`、每 `10 ms` 重读一次。每次 identity
sample 都在读 cmdline 和 `getpgid` 前后双读 `/proc/<pid>/stat`，解析 PID/PGRP/starttime；两次 tuple 必须
一致，且 PGRP 必须与 `getpgid` 一致。第一次可读的 starttime 成为不可变锚；任一次可读身份偏离 supervisor
exact PGID、starttime 改变，或超时前始终没有出现 exact trainer argv 都 fail closed。缺失中的 `/proc` 可
重试，但不会放宽到“相似命令”。identity 失败还会用 no-clobber 方式持久化 child PID、expected PGID、
first starttime、最后一次 sanitised identity 与失败原因到 `trainer_identity_failure.json`。

probe9 按 W-N→W-C→W-H→V-C→V-N→V-H 全局严格串行完成；六格均 natural exit，且每格都在
下一格前通过 verifier、不可覆盖 receipt 和 exact process/GPU/lock closure。六份 receipt 文件 SHA-256 为：
W-N `ee8c53780d6bb5f0ce5b0c31032cd8c336bfb63f922ff54b5c7d86fe5788c5ff`，
W-C `b948a4d8c701263a7805768ca7821b4257b07f008500e2aac7a5a237415018d5`，
W-H `a80502c9d9968601e844cc475ec287181d51821c88dc5a1c6728a621943a8111`，
V-C `c3db6c38446028dafe6a95455a65d0a2c12e9ab488be72fcf57546012f10edc1`，
V-N `06919a60dea539dda46357db39867f73cf9d4b019ab5313b72076d786e974bc7`，
V-H `b7a24015cc939cd19054be0710b5eead2dfb9954a8d4a9e8669b8980bdeb1ec2`。六格共用 verifier program SHA-256
`d736a205b10f1a68375df4fc51af0df547d1c0cf8096e8fa1d1867dabf590ebc`；本地 fresh v8 receipt set 重验通过，set
SHA-256=`cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`。

probe9 后渲染的 v8 long-train 制品绑定
`origin/main=d5c08bb91728edfa75801a630531527aeb2ae06c`，路径为
`/tmp/phase1_balance_slew_train_commands_d5c08bb9.json`，文件 SHA-256=
`be346f94cf6bf738da36804bf59f6a60bc5249f3c6bf5474abf617358db4b42a`。只执行了 W-N（Pod1 GPU0）；
`sim.reset()` 日志静止 `180 s` 后 locked launcher 对 exact 绑定组完成有界收敛并返回 `125`，
没有 first `Learning iteration`。外层返回 `121` 是因为旧审计错把 probe-only child evidence 当成
train 必需；人工 stable 双扫已确认 exact leader/PGID、GPU compute 与 Kit/cache lock holder 全空，
没有再发 signal。其余 W-C/W-H/V-C/V-N/V-H 五格未发射。完整冻结证据见
[`phase1_balance_action_slew_train_v8_attempt1_result_20260720.json`](../../configs/phase1_balance_action_slew_train_v8_attempt1_result_20260720.json)。
该失败发生在首轮学习前，没有任何 Reward 结果；v8 根与制品均 immutable，禁止重试 W-N 或继续五格。

fresh v9 失败审计显式接受 stage：probe 需要 leader 加 mandatory trainer child；train 是 leader-only，
`trainer_child_evidence.json` 与 `trainer_identity_failure.json` 必须 absent。审计不发 signal；它只对
exact PGID（probe 再加 child PID）连续两扫，仅 stable empty 时通过，且 train 在第二扫后再验
child evidence 仍 absent。三类 evidence 都按 canonical JSON、exact producer schema 与严格类型校验；probe
可选 identity-failure evidence 的初扫与最终扫还必须是同一 exact snapshot，出现、替换或删除都拒绝。
launcher 与 state/marker/binding/fatal postcheck 仍属同一 transaction；任一格非零都停
整批。人工还必须闭合 exact identity、assigned GPU 和 Kit/cache locks；禁止 automatic retry、broad
`pkill`/`killall` 或模式匹配清场。

两 Pod 首次预检时都存在 `/workspace/.cache/ov/_cache.lock`，且 `fuser` 没有 holder；紧邻闭合操作重验后，
两个 exact orphan lock 已删除。历史删除不保证下一次 boot 时仍无锁；如果文件重新出现，必须先证明没有
live holder 和未知 Kit，再只删除这个 exact 文件：

```bash
if [ -e /workspace/.cache/ov/_cache.lock ]; then
  if fuser -s /workspace/.cache/ov/_cache.lock; then
    echo "refuse: Kit cache lock has a live holder" >&2
    exit 1
  fi
  rm -- /workspace/.cache/ov/_cache.lock
fi
```

这不授权删除任何其他 cache、run directory、checkpoint 或日志。若进程/lock 所有权不清，停止而不是用
`pkill`、`killall` 或模式匹配清场。

以下是 probe9 [`W/V × C/N/H`](../DEFINITIONS.md#balance-action-slew-matrix) 已完成的历史渲染流程；
probe9/v8 已冻结，不得重新执行这些 probe 命令。当时远端训练 checkout 为 clean detached
`54c9a62656f0e60e5bb41cbcfa0e5a972b793906`：

```bash
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$BALANCE_REPO_ROOT"

python3 scripts/run_phase1_balance_action_slew_queue.py --stage probe
```

`--stage probe` 表示六格各 `4096 env × 24 step/env × 2 update` 的非科学完整场景探针；默认输出会显示 manifest gate
blocked。先独立生成并审过 [`launch manifest`](../DEFINITIONS.md#balance-launch-manifest)，把 exact 路径和
文件 SHA 放入任务专用 `BALANCE_LAUNCH_MANIFEST`、`BALANCE_LAUNCH_MANIFEST_SHA256`。随后才可用
[`--authorize-launch`](../DEFINITIONS.md#balance-command-render-latch)渲染命令；runner 自身会重复检查同一
`origin/main` authority，脚本仍不会自行 SSH、写远端或启动进程：

```bash
set -euo pipefail
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
BALANCE_LAUNCH_MANIFEST="$BALANCE_REPO_ROOT/configs/phase1_balance_action_slew_launch_manifest_20260720.json"
BALANCE_LAUNCH_MANIFEST_SHA256="688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353"
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
git cat-file -e origin/main:configs/phase1_balance_action_slew_launch_manifest_20260720.json
python3 - <<'PY'
import subprocess

text = subprocess.check_output(
    ["git", "show", "origin/main:docs/NOW.md"], text=True
)
title = "- **[11｜P1] 稳定机制 Wave A/B。**"
start = text.find(title)
if start < 0:
    raise SystemExit("Wave A/B claim missing from origin/main NOW")
end = text.find("\n- **[", start + len(title))
entry = text[start:] if end < 0 else text[start:end]
required = (
    "责任人 franco；执行者 Codex；执行分支",
    "Franco_codex/balance-ablation-round-20260720",
    "phase1_balance_action_slew_20260720",
)
if any(value not in entry for value in required):
    raise SystemExit("Wave A/B owner/executor/branch not bound in one NOW entry")
PY

python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage probe --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  > /tmp/phase1_balance_slew_probe9_commands.json
```

实际 probe9 执行时，`origin/main` 四道门均通过，且没有把 JSON 整体 pipe 给 shell。它使用 swap mapping：
W-N/W-C/W-H 在 Pod1 物理 GPU0/1/2，V-C/V-N/V-H 在 Pod2 物理 GPU0/1/2，并按
W-N→W-C→W-H→V-C→V-N→V-H 全局串行。每条均保留 `.launch` sidecar，并核对 PID、PGID、leader
starttime、argv、source HEAD=`54c9a62656f0e60e5bb41cbcfa0e5a972b793906`、物理 GPU、first iteration 和 fatal scan。

probe 必须自然退出到 absolute milestone `[6701]` 的 `model_6701.pt`（exclusive iteration upper
bound=`6702`）。退出后逐条执行原 JSON 的 `jobs[].probe_verifier_command`；它会在确认 policy/value/full
optimizer/two normalizers finite、C/N/H exact hard-contract/applied markers、lineage=`0`，以及 6700/6701
processed-q_des、completion/fall/legal-return、ready-tilt、qdot tag 的分母与守恒账；processed-q_des
recovery-eligible 允许单个 update 为零但两步合计必须非零；processed-q_des/qdot observed 必须逐 update
精确等于 `4096×24=98304`，其余预注册分母逐 update 非零，并在确认
无 fatal、leader/PGID/GPU 均释放后，向 `jobs[].probe_receipt_remote_path` 不可覆盖地写收据。不能套用
fresh-probe 相对 milestone `[1]`。将六份收据逐字节复制成本地
[`probe receipt set`](../DEFINITIONS.md#balance-probe-receipt-set)，exact 布局为
`$BALANCE_PROBE_RECEIPTS_DIR/{w_c,w_n,w_h,v_c,v_n,v_h}/probe_receipt.json`；probe9 的历史 exact 目录固定为
`/tmp/phase1_balance_action_slew_probe_receipts_20260720_v8`。
该本地目录只对 trusted operator 开放写权；内容寻址和多重 SHA 绑定用于防止下载损坏、
旧收据混入和配置漂移，不是抵御恶意本地 root 的数字签名；详见
[`receipt trusted-operator boundary`](../DEFINITIONS.md#balance-receipt-trust-boundary)。操作者必须从远端
`jobs[].probe_receipt_remote_path` 逐字节复制 verifier 生成的 exact bytes，不得手写或“修复”收据。

没有 `--probe-approved` 人工捷径。probe9 manifest 下六份 fresh v8 收据已全部通过本地重验，因而已解锁
`+200/+500/+1000` 科学 continuation 命令渲染；probe5 的五份 v4 收据、probe6 的 v5 失败记录、probe7
的三份 v6 收据与 probe8 的 v7 失败证据都不能补齐 v8，也不能和任一 probe9 receipt 组合。脚本仍只生成
命令，不执行：

```bash
set -euo pipefail
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
BALANCE_LAUNCH_MANIFEST="$BALANCE_REPO_ROOT/configs/phase1_balance_action_slew_launch_manifest_20260720.json"
BALANCE_LAUNCH_MANIFEST_SHA256="688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353"
BALANCE_PROBE_RECEIPTS_DIR="/tmp/phase1_balance_action_slew_probe_receipts_20260720_v8"
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
git cat-file -e origin/main:configs/phase1_balance_action_slew_launch_manifest_20260720.json
python3 - <<'PY'
import subprocess

text = subprocess.check_output(
    ["git", "show", "origin/main:docs/NOW.md"], text=True
)
title = "- **[11｜P1] 稳定机制 Wave A/B。**"
start = text.find(title)
if start < 0:
    raise SystemExit("Wave A/B claim missing from origin/main NOW")
end = text.find("\n- **[", start + len(title))
entry = text[start:] if end < 0 else text[start:end]
required = (
    "责任人 franco；执行者 Codex；执行分支",
    "Franco_codex/balance-ablation-round-20260720",
    "phase1_balance_action_slew_20260720",
)
if any(value not in entry for value in required):
    raise SystemExit("Wave A/B owner/executor/branch not bound in one NOW entry")
PY

python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage train --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  --probe-receipts-dir "$BALANCE_PROBE_RECEIPTS_DIR" \
  > /tmp/phase1_balance_slew_train_commands.json
```

上述是 probe9/v8 的冻结历史流程。最后 v8 long-train 制品是前述 `be346f94…b42a`，只发
W-N 就在 pre-first-iteration 失败；所有旧 JSON、v8 root 与剩余五格都禁止继续。

fresh v9/probe10 的 current manifest 文件/content SHA-256 是
`664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a` /
`36ceb3c77dc056f4565378a92b03da58865378d86c5849085ba066631cea456c`，config/runner SHA-256 是
`3bf5085ea8396513d162b9cce249dfb761b39b2827ec722959343c953683e59e` /
`0fff4515cbe7e62798e8c39f701851c46e68287c7321e7618161fa9dde4789ce`，root 为
`/workspace/codexschema/phase1_balance_action_slew_v9_20260720`。当前只是
**source ready / preregistered / not launched**。先在当时最新 `origin/main` 做 NO-LAUNCH plan，再渲染：

```bash
set -euo pipefail
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$BALANCE_REPO_ROOT"
python3 scripts/run_phase1_balance_action_slew_queue.py --stage probe \
  > /tmp/phase1_balance_slew_probe10_plan.json

git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
BALANCE_LAUNCH_MANIFEST="$BALANCE_REPO_ROOT/configs/phase1_balance_action_slew_launch_manifest_20260720.json"
BALANCE_LAUNCH_MANIFEST_SHA256="664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a"
test "$(sha256sum "$BALANCE_LAUNCH_MANIFEST" | awk '{print $1}')" = "$BALANCE_LAUNCH_MANIFEST_SHA256"
python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage probe --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  > /tmp/phase1_balance_slew_probe10_commands.json
```

不能把 JSON 整体 pipe 给 shell；只能逐条执行审过的 `ssh_argv`。probe10 必须严格
W-N→W-C→W-H→V-C→V-N→V-H 全局串行，每格 receipt 和 exact closure 先于下一格。六份 fresh
v9 receipts 收齐到 `/tmp/phase1_balance_action_slew_probe_receipts_20260720_v9/{job}/probe_receipt.json`
后才能渲染 train；probe9/v8 receipt 不可复用：

```bash
BALANCE_PROBE_RECEIPTS_DIR="/tmp/phase1_balance_action_slew_probe_receipts_20260720_v9"
python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage train --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  --probe-receipts-dir "$BALANCE_PROBE_RECEIPTS_DIR" \
  > /tmp/phase1_balance_slew_train_v9_commands.json
```

probe/train 保持 same swap：W-N/W-C/W-H 是 Pod1 GPU0/1/2，V-C/V-N/V-H 是 Pod2 GPU0/1/2。long-train
在每个 Pod 内必须串行 Kit boot；前一格真实进入 `Learning iteration` 且 boot lock 释放后，才发同
Pod 下一格。任一失败停止新发射并闭合 exact identity/GPU/lock；无 automatic retry，禁止 broad
signal。详细科学量尺见 [实验记录](../experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md)。

## Hard Rules (summary — full list in the pod README)

1. `/root/` is the ephemeral container disk (wiped on restart). Everything goes under `/workspace`.
2. Never `pip install/uninstall` in the shared venv, and never `pip install -e` your
   `whole_body_tracking` into it. New common deps: announce, install, refresh
   `/workspace/smoke_test/freeze_baseline.txt`.
3. Do not edit `/workspace/IsaacLab`; override behavior in your own code.
4. GPU 没有永久的人名所有权。交互式/临时作业默认一人一张 GPU；使用前必须按 `NOW`/实验 queue
   核对当前分配、查 `nvidia-smi`、通知其他人，
   并显式设置 `CUDA_VISIBLE_DEVICES=<n>`。这条不覆盖已在 [NOW 唯一队列](../NOW.md#统一工作队列唯一优先级账本)
   登记的广度消融波：广度波在短测确认显存/利用率后可同卡并发 3–4 条同类
   4096-env 任务；关键路径/长跑仍独占。新任务必须先跨所有可用 GPU 各放一条，再开始第二、第三轮，
   Pod1 才有第四轮；已经运行且合同绑定的实验不迁移。被他人占用或没有通过前置门的卡直接跳过，
   不得用重复 seed 或已失败配方补位。日期化实测和完整约束只看
   [跑批作战手册](../runbook.md#rtx-5090-实测算力手册)，不在本文建第二份优先级队列。
5. WandB login is global and ephemeral (`/root/.netrc`); pass `WANDB_API_KEY=...` per run or use
   `logger=tensorboard`.

## Launch Rules (hard-won 2026-07-03 — read before starting ANY job over ssh)

1. **One Kit boot at a time, pod-wide.** (2026-07-20 修复：`kit_boot_lock.sh` 旧版把持锁的
   fd 9 泄漏给长跑训练子进程——训练跑多久锁就被占多久，同 pod 后续所有发射永久阻塞在 flock 上；
   探针类短任务退出快所以从未暴露。现版本以 `9>&-` 启动子进程，锁只在 boot 阶段持有。两 pod 均已
   打补丁，原件备份为 `kit_boot_lock.sh.bak_20260720`。同日第二个教训：广度波 4 条/卡时，发射预检
   要求"该卡零 compute 进程"会自相矛盾——检查应为"同卡已有进程 < 4"。）
   原始规则： Parallel Isaac boots deadlock each other (worst with cold
   caches right after a restart: five jobs sat 20+ min at 0 progress). Wrap every training/play
   launch in the global boot lock:
   ```bash
   /workspace/bin/kit_boot_lock.sh /path/to/run.log python scripts/train.py ...
   ```
   It serializes only the boot (releases when "Learning iteration" appears), then jobs run in
   parallel normally.
2. **Detach properly or your job dies with the ssh session (“陪葬”).** `nohup ... &` alone is NOT
   enough under flaky connections; use `setsid nohup <cmd> </dev/null > log 2>&1 &` (the boot-lock
   wrapper does this for you). Judge success by artifacts (checkpoints/ONNX on disk), never by the
   session surviving.
3. **Never use broad process-pattern signals for managed experiments.** `pkill -f` can match the
   ssh command itself or an unrelated arm with a similar run name. Read the owned PID/PGID from a
   verified launch sidecar, inspect that exact numeric group, and follow the evidence-preserving
   TERM→conditional-KILL procedure above. `pkill`/`killall` are not accepted substitutes.
4. **A SIGKILL'd Kit leaves an orphaned cache lock that hangs every later boot.** Symptom: a lone
   job freezes at the AutoNode-registration boot phase forever. Check `fuser
   /workspace/.cache/ov/_cache.lock` — no holder = orphaned; `rm` it and relaunch. (A lock held by
   a LIVE process is healthy — do not delete that one.) Prefer SIGTERM first when killing Kits.
5. **Hosts with a broken render stack hang the boot in the URDF importer's UI build.** Symptom
   (2026-07-03 host): every headless boot freezes forever right after the AutoNode phase at ~110%
   CPU, with NO stale cache lock; faulthandler shows the main thread inside
   `isaacsim.asset.importer.urdf .../ui_utils.py string_filed_builder` — the extension builds its
   import window even headless, and the first `omni.ui.StringField` never returns when the host's
   iray/RTX stack is broken (bare CUDA fine). Fix: `export HOPE_URDF_IMPORTER_NO_UI=1` (env.sh) —
   the shared venv's extscache extension is patched to skip `build_ui()` under this flag (unset =
   stock). Headless no-camera training/eval then works; anything needing RTX rendering/cameras is
   still dead on such a host (`_wait_for_viewport` hangs, `rtx.neuraylib` fails to load).
6. **Verify every launch.** After starting a job, confirm within ~60 s that its log exists and the
   process is alive; a launcher that prints nothing probably did nothing (two silent queue failures
   cost us 30 idle GPU-minutes).

## Known Quirks

- **No-PID 100% GPU / 575 W host-stuck state**: observed 2026-07-09 on the second RunPod
  endpoint `74.2.96.48` (ports `16389`, `10473`, then a newly opened `16442`). Symptom: all three
  visible RTX 5090s report
  `P1`, `GPU-Util` 99-100%, about `574-575 W`, and about `1538 MiB` used, while `nvidia-smi`,
  `nvidia-smi pmon`, `--query-compute-apps`, and container `/dev/nvidia*` fd scans show **no
  running processes**. Killing the only visible Isaac/Python jobs cleared compute-app entries and
  all device fds, but the 100%/575 W state persisted after waiting. `nvidia-smi --gpu-reset -i
  0,1,2` returned `Resetting GPU ... is not supported` for all three cards. A pod/container restart
  changed SSH port and container hostname but kept the same GPU UUIDs, so Stop/Start can leave the
  same physical GPUs pinned in a bad host-driver state. The `16442` pod reproduced the same GPU
  UUIDs and host boot id with a fresh container hostname, confirming that "new pod" can still mean
  the same bad host/GPU allocation when RunPod keeps the volume/host locality. Treat this as a
  RunPod host/GPU-driver issue, not as user training load. Before launching any job on a
  new/restarted pod, run:
  `nvidia-smi --query-gpu=index,uuid,pstate,utilization.gpu,memory.used,power.draw --format=csv`.
  If the no-PID full-load state is present, migrate to a pod on a fresh host or ask RunPod support
  for a host-side GPU/driver reset; do not burn time debugging project code.
  Follow-up: a separate newly opened endpoint `74.2.96.37:14746` had a different host boot id,
  different GPU UUIDs, and driver `590.48.01` (vs `580.126.09` on `74.2.96.48`), but showed the
  same no-PID 99-100% / 575 W state. This broadens the issue from one bad allocation to a likely
  RunPod provider/host-isolation problem for those RTX 5090 nodes.
- **git-lfs**: the git-lfs filters are NOT configured globally (global gitconfig lives on the
  ephemeral disk). Each clone needs a one-time `git lfs install --local && git lfs pull` (persisted
  in the clone's own `.git/config` on `/workspace`; franco's clone done 2026-07-03). Symptom when
  missing: vendor CSV/PNG files appear as 3-line pointer stubs and `git pull` prints "Encountered N
  files that should have been pointers, but weren't".
- **Phantom `modified:` on vendor CSV/PNGs**: several `agi/code_deployment/.../a3_runtime/*.csv`
  and two analysis PNGs were committed as raw blobs while `.gitattributes` marks them LFS, so
  LFS-enabled machines show them as permanently modified. Harmless (pull/push work); repo-side
  normalization is on the fixes list.
- **Isaac Kit can crash with exit code 0 AND eat the last stdout flush on exit**: judge job success
  from artifacts (checkpoints on disk), not `$?` and not only log sentinels. The train smoke was
  updated on 2026-07-03 to accept `model_9.pt` on disk as the success sentinel after a run whose
  final "Learning iteration 9/10" line never reached the log.
- **Pod restart**: `/workspace` survives; `/root` (wandb login, apt packages, bash history, ssh
  keys, installed CLIs) does not. Restart-proofed on 2026-07-03: `/workspace/bin` (git-lfs, claude,
  kit_boot_lock.sh) + `CLAUDE_CONFIG_DIR=/workspace/.claude` are wired into every `env.sh`; the
  GitHub deploy key however still lives in `/root/.ssh` and must be re-installed after each restart
  (persistent-key decision pending). After restart: re-`source env.sh`, re-run `--quick` smoke, and
  expect the FIRST Kit boot to be slow (cold caches — all the more reason for the boot lock).

## Adding A New User (or Agent Workspace)

```bash
bash /workspace/setup_user.sh <name> <gpu_id> <git_user_name> <git_email>
source /workspace/<name>/env.sh
cd $HOPE_ROOT && git lfs install --local && git lfs pull
git switch -c <your-branch>
bash /workspace/smoke_test/run_smoke.sh <name>
```

`setup_user.sh` is idempotent: it clones the repo, writes `env.sh` +
`setup_train_env.local.sh`, and copies the git-ignored `assets/agibot_a3` from
`/workspace/shared/assets` (fresh clones cannot build the env without it).

## Wave-B 专用入口

下肢稳定的 W/V×B0/B1/B2 六格不得手拼 SSH。使用
[`run_phase1_lower_body_stability_wave.md`](run_phase1_lower_body_stability_wave.md) 的 manifest、自然退出 probe、
六 receipt 解锁与 exact PGID 流程；本页的通用 Pod 纪律仍然适用。

## Update Rule

Any change to the pod endpoint, venv contents, folder convention, or smoke procedure must update
this file AND `/workspace/README_MULTIUSER.md` on the pod.
