# Run Training On The Shared RunPod

Status: Active (pod provisioned 2026-07-01/02; verified with a full smoke + 10-iteration train on 2026-07-03)

The team shares one RunPod for Isaac training: 3× RTX 5090 (32 GB, sm_120), 128 cores, 1 TB RAM,
500 GB persistent volume at `/workspace`. The live source of truth for pod-side rules is
`/workspace/README_MULTIUSER.md` **on the pod** — read it after logging in; this file records how
to get on, what is verified, and the repo-side conventions.

## Access

```bash
ssh root@162.43.172.171 -p 15320 -i ~/.ssh/id_ed25519_runpod
```

- Endpoint current as of 2026-07-03 evening (port changed 17424 → 15320 after a pod restart the
  same day — the restart confirmed `/workspace` survives intact incl. checkpoints/venvs; only
  running processes and `/root` were lost). RunPod may assign a new IP/port on any re-provision —
  update this file when it changes.
- Each teammate (and each teammate's coding agent) uses the same root login; separation is by
  directory, not by account. Work ONLY under your own `/workspace/<name>/`.
- The pod holds one GitHub deploy key (dongc1's) shared for pull/push; per-clone `user.name`/
  `user.email` are already set so commits are attributed correctly.

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
4. **Verify every launch.** After starting a job, confirm within ~60 s that its log exists and the
   process is alive; a launcher that prints nothing probably did nothing (two silent queue failures
   cost us 30 idle GPU-minutes).

## Known Quirks

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
