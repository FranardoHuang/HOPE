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

## Hard Rules (summary — full list in the pod README)

1. `/root/` is the ephemeral container disk (wiped on restart). Everything goes under `/workspace`.
2. Never `pip install/uninstall` in the shared venv, and never `pip install -e` your
   `whole_body_tracking` into it. New common deps: announce, install, refresh
   `/workspace/smoke_test/freeze_baseline.txt`.
3. Do not edit `/workspace/IsaacLab`; override behavior in your own code.
4. One GPU per person by default; borrow only after `nvidia-smi` + a heads-up, via
   `CUDA_VISIBLE_DEVICES=<n>`.
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
3. **pkill self-match kills your own session.** `ssh pod 'pkill -f myscript'` matches the ssh
   command line itself → session dies mid-command with exit 255 and later commands silently never
   run. Always bracket the first char: `pkill -f "[m]yscript"`.
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
