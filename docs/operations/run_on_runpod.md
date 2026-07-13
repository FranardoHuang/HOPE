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
├── <name>/nohope                 # your own clone, your own branches (yikang=GPU0, franco=GPU1, jiayi=GPU2)
├── <name>/env.sh                 # source in EVERY shell before working
├── setup_user.sh                 # provision a new user: bash setup_user.sh <name> <gpu> [git_name] [git_email]
├── smoke_test/                   # environment smoke suite + logs
└── README_MULTIUSER.md           # pod-side rules (authoritative)
```

Every shell session:

```bash
source /workspace/<name>/env.sh   # activates venv, puts YOUR clone first on PYTHONPATH, pins YOUR GPU
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

## Hard Rules (summary — full list in the pod README)

1. `/root/` is the ephemeral container disk (wiped on restart). Everything goes under `/workspace`.
2. Never `pip install/uninstall` in the shared venv, and never `pip install -e` your
   `whole_body_tracking` into it. New common deps: announce, install, refresh
   `/workspace/smoke_test/freeze_baseline.txt`.
3. Do not edit `/workspace/IsaacLab`; override behavior in your own code.
4. 交互式/临时作业默认一人一张 GPU；借用前必须查 `nvidia-smi`、通知其他人，
   并显式设置 `CUDA_VISIBLE_DEVICES=<n>`。这条不覆盖已在 [NOW 唯一队列](../NOW.md#统一工作队列唯一优先级账本)
   登记的广度消融波：广度波在短测确认显存/利用率后可同卡并发 3–4 条同类
   4096-env 任务；关键路径/长跑仍独占。新任务必须先跨所有可用 GPU 各放一条，再开始第二、第三轮，
   Pod1 才有第四轮；已经运行且合同绑定的实验不迁移。被他人占用或没有通过前置门的卡直接跳过，
   不得用重复 seed 或已失败配方补位。日期化实测和完整约束只看
   [跑批作战手册](../runbook.md#rtx-5090-实测算力手册)，不在本文建第二份优先级队列。
5. WandB login is global and ephemeral (`/root/.netrc`); pass `WANDB_API_KEY=...` per run or use
   `logger=tensorboard`.

## Launch Rules (hard-won 2026-07-03 — read before starting ANY job over ssh)

1. **One Kit boot at a time, pod-wide.** Parallel Isaac boots deadlock each other (worst with cold
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

## Update Rule

Any change to the pod endpoint, venv contents, folder convention, or smoke procedure must update
this file AND `/workspace/README_MULTIUSER.md` on the pod.
