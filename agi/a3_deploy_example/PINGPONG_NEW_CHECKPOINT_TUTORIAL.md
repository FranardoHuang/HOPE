# Ping-Pong: New Checkpoint → ONNX → AGI Deploy → MuJoCo → Hardware (full tutorial)

End-to-end, copy-paste workflow for taking a **freshly trained HOPE/HITTER ping-pong
checkpoint (`model_*.pt`)** all the way to the **AGI A3 robot**, via ONNX export,
contract/parity checks, AGI MuJoCo, Rockchip cross-build, and a staged hardware
bring-up.

Written for someone who has **never run the pipeline**. Every command block starts with a
**`Where:`** line giving **which distrobox/container** and **which folder to `cd` into**
before you run it (and whether the env must be sourced). The explicit `distrobox enter …` /
`cd …` lines inside each block make it copy-paste-safe from a fresh shell — re-running them
is harmless. Nothing here modifies AGI's original files or
AGI's MuJoCo simulation — we only align our ONNX runner (`a3_deploy_onnx_ref_pingpong`)
to AGI's robot-IO backend.

> Our policy is **`obs 180 → act 31`** (BeyondMimic-style WBC), **NOT** AGI's original
> `obs 1570 → act 29` HITTER-tokenizer policy. They are **not** interchangeable: AGI's
> native `a3_deploy_onnx_ref` runner cannot load our ONNX, which is exactly why we ship
> a separate `a3_deploy_onnx_ref_pingpong` binary with our own C++ front-end.

> ### 🔴 CURRENT DEPLOY (2026-07-02) — read before going to the robot
> **Shipped policy: `model_p4_deployparity.onnx` — 175-D obs / 31-act**, all-implicit-PD
> training (matches the real actuation). The worked example throughout Stages 1–4 uses the
> older 180-D `model_15200` — that's fine, the **same** binary auto-detects 175 vs 180 from
> the ONNX input; only the obs width differs. The runner config already points at p4:
> [`config/a3_runtime_config.pingpong.yaml`](src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml).
>
> **Deploy FOREHAND ONLY.** p4 forehand passed the AGI-MuJoCo gate (10 clean cycles, 0 guard
> trips); **backhand is NOT deploy-ready** on any model. On the robot press `1`/`f`; **never `b`.**
>
> **⚠️ The staged `dist/a3_deploy_rockchip/` (what ships to the robot) is STALE** — a
> 2026-07-01 binary carrying `model_15200` and **none** of the 07-02 sim2real fall fixes.
> The x86 dist is current; the rockchip one is not. **You MUST rebuild rockchip (Stage 5)
> with p4 before shipping** (the builder auto-stages the model — no manual copy). Sim
> recheck first with `scripts/pp_freebase_watch.sh`, then Stage 5 → Stage 6, low gain,
> `--single-swing`, forehand only.

---

## 0. The three environments (read this first)

| distrobox / host | What it is | Which stages run here |
|---|---|---|
| **`grasping`** | GPU box, Isaac Lab + Isaac Sim (omnidrones image). Training env lives here; also the conda env `hope-motion-py310` (plain Python with `mujoco`+`onnxruntime`). | **1** export `.pt`→ONNX · **2** training-side MuJoCo / in-Isaac eval |
| **`hope`** | ROS 2 **Jazzy** + AimRT + g++13 build box (`ros2-jazzy:2026-06-17`). | **3** install ONNX + build + parity · **4** AGI MuJoCo deploy-path test · **5** Rockchip cross-compile |
| **robot MDU** (Rockchip/aarch64) | The robot's onboard compute unit, reached through the **HDU** jump host. | **6** hardware bring-up |

Enter a box with `distrobox enter <name>`. Two **gotchas that bite every time**:

- The **`hope`** box has a broken `.bashrc` (points at a non-existent ROS). **Every new
  shell in `hope` must manually run** `source /opt/ros/jazzy/setup.bash`.
- The **`grasping`** training env must be **sourced** (`source setup_train_env.sh`), once
  per terminal — it defines the `hope_isaac_py` launcher and the W&B/registry vars.

```bash
# Host shell — list the boxes any time:
distrobox list
# Expect: grasping (Up), hope (Up), ros2-jazzy (base image, unused directly)
```

---

## Stage 1 — Train checkpoint `.pt` → ONNX export

**Run in distrobox: `grasping`.** (This is `scripts/play.py`, an Isaac Lab Hydra entry —
it needs Isaac, so it runs on the GPU box, never in `hope`.)

### 1.1 Where checkpoints live & how to pick one

Trained checkpoints are written under the training logs, one dir per run:

```
hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/<RUN_DIR>/model_<N>.pt
```

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking

# List runs newest-first, then the checkpoints inside the run you want:
ls -dt logs/rsl_rl/agibot_a3_hope/*/
ls -t  logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch/model_*.pt | head
```

**How to choose:** pick the checkpoint your evaluation said was best — not blindly the
highest step number. The current shipped champion is the **unified forehand+backhand**
policy **`model_15200`** from run **`2026-06-28_23-01-24_phase050_perclippos_scratch`**.
(Rank candidates with `scripts/eval_deterministic.py`, not by W&B rollout reward — see
the project memory on "run comparison".)

### 1.2 Set up the training env (once per terminal)

**Where:** box **`grasping`** · `cd ~/workspace/HOPE/hope_training/whole_body_tracking` (continues the shell from 1.1).

```bash
distrobox enter grasping                                 # skip if already inside grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh        # prints "[hope] training env ready"
# sanity: should print 1.3.2 and a non-empty path
hope_isaac_py -c "import hydra; print(hydra.__version__)"
echo "$HOPE_WBT_PYTHONPATH"
```

### 1.3 Export the ONNX (bake BOTH clips for a unified policy)

The **canonical, verified** command is just `checkpoint=...`. The `HOPEPingPong` task
config already provides **both** clips (`registry_name` = forehand, `registry_name_2` =
backhand), so the export automatically bakes the full forehand+backhand reference and
prints `[play.py] UNIFIED 2-clip export: clip0=...hope_forehand  clip1=...hope_backhand`:

**Where:** box **`grasping`** · `cd ~/workspace/HOPE/hope_training/whole_body_tracking` · env sourced (1.2).

```bash
distrobox enter grasping                                 # skip if already inside grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking    # RUN path below is relative to here
RUN=logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch

hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo headless=true num_envs=2 \
  checkpoint=$RUN/model_15200.pt
# -> $RUN/exported/policy.onnx   (+ learned_std.npy)
# (pulls the two clips from the W&B motions registry; run `wandb login` once if needed)
```

If you want a **fully offline** export (no W&B) and have the two motion `.npz` files
locally, pass them explicitly as a list — `motion_file` *is* a `play.yaml` key:

**Where:** same shell — box **`grasping`**, in `~/workspace/HOPE/hope_training/whole_body_tracking`, env sourced.

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo headless=true num_envs=2 \
  checkpoint=$RUN/model_15200.pt \
  'motion_file=[logs/rsl_rl/eval_motion/fh.npz, logs/rsl_rl/eval_motion/bh.npz]'
```

> ⚠️ **CRITICAL — both clips, or the backhand dies.** A unified policy exported with
> only one clip bakes a **forehand-only** reference (`time_step_total`=95 instead of
> ~200). At deploy the backhand `time_step`s clamp to the last forehand frame → the
> reference freezes → **the backhand never swings**. The task config's `registry_name_2`
> (or a `motion_file` list) is what prevents this — so don't override the task to a
> single clip.
>
> Hydra note: `registry_name_2` lives in the **task** config, not `play.yaml`. Don't
> pass `registry_name_2=...` as a bare CLI override (Hydra rejects keys absent from
> `play.yaml`); rely on the task default, or use `+registry_name_2=...` to add it.

**Output filename produced:** `policy.onnx` next to the checkpoint, under
`$RUN/exported/policy.onnx`. In AGI deploy we rename/copy it to the canonical
**`model_15200.onnx`** (Stage 3).

### 1.4 Verify the ONNX size and input/output dims

**Where:** box **`grasping`** · `cd ~/workspace/HOPE/hope_training/whole_body_tracking` (same shell; `$RUN` still set from 1.3).

```bash
# size: BOTH clips ~1.24 MB; forehand-only ~1.14 MB. If it's ~1.14 MB you forgot clip 2.
ls -la $RUN/exported/policy.onnx

# preferred: the contract inspector prints the train-vs-deploy dims + a per-joint
# kp/kd/default/action_scale table vs the AGI deploy hpp. Pass --onnx explicitly
# (its default points at a different run). Needs only onnxruntime+numpy.
python scripts/inspect_a3_deploy_contract.py --onnx $RUN/exported/policy.onnx

# or a quick raw dim+metadata dump (any python with onnxruntime):
python3 - "$RUN/exported/policy.onnx" <<'PY'
import sys, onnxruntime as ort
s = ort.InferenceSession(sys.argv[1])
print("in :", [(i.name, i.shape) for i in s.get_inputs()])
print("out:", [(o.name, o.shape) for o in s.get_outputs()])
md = s.get_modelmeta().custom_metadata_map
for k in ["joint_names","default_joint_pos","action_scale","joint_stiffness","joint_damping","body_names"]:
    print(k, "->", str(md.get(k,"<MISSING>"))[:90])
PY
```

**PASS criteria:**
- input `obs` shape `[1, 180]` **plus** `time_step` `[1, 1]`;
- output `actions` shape `[1, 31]` (the policy also emits reference body/joint side
  outputs — that's expected);
- this is **180 → 31**, *not* AGI's **1570 → 29**. If you see 1570/29 you exported the
  wrong policy.
- metadata present: 31 `joint_names`, 31-vector `action_scale` / `default_joint_pos` /
  `joint_stiffness` (kp) / `joint_damping` (kd), 14 `body_names`.

---

## Stage 2 — Training-side simulation eval (before AGI deploy)

**Run in distrobox: `grasping`.** Two checks; the MuJoCo one is the important verdict.

### 2.1 (optional) In-Isaac rollout — confirms export→load→rollout pipeline

**Where:** box **`grasping`** · `cd ~/workspace/HOPE/hope_training/whole_body_tracking` · env sourced (re-exports the onnx too).

```bash
distrobox enter grasping                                 # skip if already inside grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
  checkpoint="$RUN/model_15200.pt" \
  headless=true video=true video_length=300
# headless+video terminates cleanly and writes videos/play/play.mp4
# (both clips bake from the task cfg; headless=true video=false would loop forever)
```

### 2.2 MuJoCo cross-simulator eval (the real sim-to-sim verdict)

This loads the **same ONNX** in a *different physics engine* (MuJoCo, no Isaac),
rebuilds the exact **180-D** obs, and prints the strike-composite metrics per swing.
Use a **plain Python env** with `mujoco`+`onnxruntime` (the conda env), **not**
`hope_isaac_py`:

```bash
distrobox enter grasping
conda activate hope-motion-py310
cd ~/workspace/HOPE/hope_training/whole_body_tracking
RUN=logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch

python scripts/mujoco_eval_onnx.py \
  --onnx $RUN/exported/policy.onnx \
  --std  $RUN/exported/learned_std.npy \
  --mjcf ~/workspace/HOPE/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
  --noise-scales 0.0 0.05 --steps 4000 --pd-mode implicit --seed 0 --ee-term-z 100
```

**Why these flags:** `--pd-mode implicit` matches Isaac's `ImplicitActuator` (the
faithful test; `explicit` inflates velocity error from actuator discretization).
`--noise-scales 0.0 0.05` runs deterministic **and** lightly-dithered. `--ee-term-z 100`
disables the *training* tracking-guard so swings run to completion (it is a reset device,
not a fall).

**PASS criteria (validated for `model_15200`):**
- **Zero falls** at both `0.0` and `0.05` (full episodes). Deterministic is the deploy
  path — no dither needed.
- **Forehand** composite **~0.90** (pos_err ~0.06 m).
- **Backhand** composite **~0.50** (position-limited: racket ~0.085 m off target;
  velocity & normal pass).

Outputs: `mujoco_sim2sim_log.csv` (per-step, includes `racket_speed`) and
`mujoco_sim2sim_strikes.csv` (per-strike) in the run dir (or `--out-dir`).

---

## Stage 3 — Install the ONNX into AGI deploy + build + contract/parity checks

**Run in distrobox: `hope`.** (ROS 2 Jazzy + AimRT + g++.) Remember to
`source /opt/ros/jazzy/setup.bash` in every new `hope` shell.

### 3.1 Copy the ONNX to the durable asset

The **durable source of truth** is `assets/a3_runtime/models/model_15200.onnx`. Copy your
freshly exported `policy.onnx` onto that name. **Don't** stage into `dist/` here — the
build in 3.3 does `rm -rf dist/`, so dist-staging is the *last* step (3.3).

```bash
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example

cp ~/workspace/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch/exported/policy.onnx \
   assets/a3_runtime/models/model_15200.onnx
```

### 3.2 Point the runner config at it

**Where:** box **`hope`** — edit the file under `~/workspace/HOPE/agi/a3_deploy_example/`. The
runtime config field is **`onnx.model_path`** in the **`src/` copy (source of truth)**
[`src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml`](src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml):

```yaml
onnx:
  model_path: ../models/model_15200.onnx
```

**How the runner resolves this — get it wrong and the model load silently fails.**
`a3_pingpong_main.cpp` → `Resolve()`:
- the runner is launched **from `dist/a3_deploy_<arch>/`** with
  `--runtime-cfg config/a3_runtime_config.pingpong.yaml`, so the config's directory
  (`cfgdir`) is `dist/.../config`;
- a **relative** `model_path` is joined onto `cfgdir`: `../models/model_15200.onnx` →
  `dist/.../config/../models/model_15200.onnx` = **`dist/.../models/model_15200.onnx`** ✓;
- an **absolute** path is used as-is; there is **no cwd / repo-root fallback**.

> ⚠️ **Do NOT** set this to a repo-root path like
> `agi/a3_deploy_example/assets/a3_runtime/models/model_15200.onnx`. `Resolve()` joins it
> onto `cfgdir` → `dist/.../config/agi/a3_deploy_example/assets/...`, which does **not**
> exist → the model load **fails**. Keep it config-dir-relative (`../models/...`); that is
> exactly why Stage 3.3 stages the model into `dist/.../models/`.

- Keep the filename `model_15200.onnx` → **no edit needed**. Rename it → change this one key.

### 3.3 Build the x86 package with the ping-pong runtime cfg

**Where:** box **`hope`** · `cd ~/workspace/HOPE/agi/a3_deploy_example`.

The recommended path is now to pass the ping-pong runtime cfg directly to the build script.
That makes the packager carry the `model_15200.onnx`, preserve a
`config/a3_runtime_config.pingpong.yaml` alias, and generate `run_a3_pingpong.sh` in the
fresh dist package:

```bash
distrobox enter hope                                     # skip if already inside hope
cd ~/workspace/HOPE/agi/a3_deploy_example
source /opt/ros/jazzy/setup.bash
bash scripts/build_a3_deploy_pkg.sh \
  --arch x86_64 \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml \
  --jobs $(nproc)
```

**Confirm the runner will find the model**:

```bash
ls -la dist/a3_deploy_x86_64/config/a3_runtime_config.pingpong.yaml
ls -la dist/a3_deploy_x86_64/models/model_15200.onnx
ls -la dist/a3_deploy_x86_64/run_a3_pingpong.sh
```

The packager still writes the normalized runtime cfg as `config/a3_runtime_config.yaml`, but now
also keeps the ping-pong basename alias so all later commands can continue using
`config/a3_runtime_config.pingpong.yaml`.

### 3.4 Dump ONNX metadata (confirm the contract fields)

```bash
distrobox enter hope -- bash -lc '
python3 - <<PY
import onnxruntime as ort
s=ort.InferenceSession("/home/dongc1/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64/models/model_15200.onnx")
md=s.get_modelmeta().custom_metadata_map
for k in ["joint_names","default_joint_pos","action_scale","joint_stiffness","joint_damping","body_names"]:
    print(k,"->",str(md.get(k,"<MISSING>"))[:120])
print("in :", [(i.name,i.shape) for i in s.get_inputs()])
print("out:", [(o.name,o.shape) for o in s.get_outputs()])
PY
'
```

**How the runner uses this metadata** (read at load by `pp_onnx_policy.hpp`):
- decode: **`q_des = default_joint_pos + action * action_scale`** (Isaac joint order),
  `use_default_offset`, no clip;
- **`kp = joint_stiffness`, `kd = joint_damping`** are published as-is, because the real
  A3 backend runs **implicit PD** — the gains MUST match training or the robot behaves
  differently than in sim;
- joint outputs are scattered **by joint name** onto the 31 backend slots
  (`MakeA3Layout31`: waist3, neck2, Larm7, Rarm7, Lleg6, Rleg6);
- **neck is passive**: slots [3,4] are overwritten to `q=0, kp=40, kd=2`; the model's
  neck outputs are discarded.

### 3.5 Run the contract / parity / smoke checks

**(a) C++ ↔ Python ONNX parity + end-to-end smoke** (the wrapper builds both harnesses,
generates the Python reference, diffs, and runs the first-tick dump):

**Where:** box **`hope`** · `cd ~/workspace/HOPE/agi/a3_deploy_example` (`source /opt/ros/jazzy/setup.bash` first).

```bash
distrobox enter hope                                     # skip if already inside hope
cd ~/workspace/HOPE/agi/a3_deploy_example
bash scripts/pingpong_parity/run_parity.sh
# default MODEL=assets/a3_runtime/models/model_15200.onnx ; needs a python with onnxruntime
# (set PYBIN=/path/to/python if `python3` lacks onnxruntime)
```

**(b) The three contract gates** (joint-map bijection, full CommandFn, 180-D obs builder):

```bash
distrobox enter hope -- bash -lc '
cd /home/dongc1/workspace/HOPE/agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref
ORT=/home/dongc1/workspace/HOPE/agi/a3_deploy_example/thirdparty/onnxruntime/onnxruntime-linux-x64-1.19.2
MODEL=/home/dongc1/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64/models/model_15200.onnx
OUT=/tmp/pp_gates; mkdir -p $OUT
# (a) jointmap bijection + default_q scatter
g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 -I $ORT/include \
  include/a3_pingpong/test/pp_jointmap_test.cpp -L $ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib -o $OUT/jm && $OUT/jm $MODEL | tail -2
# (b) full CommandFn (neck passive / gains / |action| bounded / swing sweep)
g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 -I $ORT/include \
  include/a3_pingpong/test/pp_policy_test.cpp -L $ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib -o $OUT/pol && $OUT/pol $MODEL | tail -3
# (c) 180-D obs builder vs Python golden (eigen-only)
g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 \
  include/a3_pingpong/test/pp_parity_test.cpp -o $OUT/par && $OUT/par include/a3_pingpong/test/golden.txt | tail -3
'
```

**PASS criteria (all must hold):**
- ONNX input/output dims **180 → 31** (Stage 1.4 / 3.4).
- `JOINTMAP PASS` — all **31 backend slots filled exactly once** (bijection).
- `POLICY CALLBACK PASS` — `max|action|` bounded (~14), `fails=0`, neck passive.
- `PARITY PASS` — obs error ~1e-16; **C++ vs Python ONNX action `max|Δ| ≈ 1e-6`**
  (this pass: 9.5e-7).
- e2e smoke (in `run_parity.sh`): `q_des` **finite & bounded** (max ≈ 0.88 rad),
  `kp ∈ [20,250]`, `kd ∈ [2,8]`.
- metadata == training contract, **0 mismatches**.

---

## Stage 4 — AGI MuJoCo deploy-path test (do NOT modify AGI's sim)

**Run in distrobox: `hope`.** This closes the **same I/O loop the real robot uses**
(obs → jointmap → decode → command → sync → iceoryx) against AGI's MuJoCo. It proves
**deploy-path closure**, **not** robot stability — for clean swings use the **hoist**
(fixed-base) variant. **Two terminals.**

### Terminal 1 — start AGI MuJoCo (a3_pingpong sim) + iceoryx

```bash
distrobox enter hope
cd ~/workspace/HOPE/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install/bin
source /opt/ros/jazzy/setup.bash
for p in ros2_plugin_proto aimrt_msgs joint_msgs mujoco_sim_msgs; do source ../share/$p/local_setup.bash; done
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd):$(pwd)/../lib"

# clean any stale shared memory / processes from a previous run:
pkill -9 -x aimrt_main 2>/dev/null; pkill -9 -x iox-roudi 2>/dev/null; rm -f /dev/shm/iox*

# start the iceoryx broker, then the sim with the HOIST ping-pong config:
setsid ./iox-roudi >/tmp/iox-roudi.log 2>&1 </dev/null & sleep 1
MUJOCO_GL=egl ./aimrt_main --cfg_file_path=./cfg/a3_pingpong_hoist_cfg.yaml
# drop the `MUJOCO_GL=egl` prefix if you have a real display and want the viewer.
```

### Terminal 2 — start our ping-pong runner (staged)

```bash
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}"

./a3_deploy_onnx_ref_pingpong \
  --runtime-cfg config/a3_runtime_config.pingpong.yaml \
  --start passive --level 1 --legs-passive
```

**Runtime keys** (type in the runner's terminal):

| key | meaning |
|---|---|
| `p` | PASSIVE (limp, zero gain) |
| `h` | SHADOW (compute the policy, **do not publish** command) |
| `s` | **PD_STAND — holds nominal pose, does NOT swing** (do not judge swing here) |
| `m` | **MOTION — compute AND publish** (the only mode that swings) |
| `0` / `1` | swing level: **0 = hold-windup (no swing)**, **1 = swing** |
| `f` / `b` | scripted swing direction: **f = forehand**, **b = backhand** (default forehand) |
| `[` / `]` | gain_scale − / + 0.1 |
| `,` / `.` | swing_speed − / + 0.1 (slow the action if actuators can't keep up) |
| `q` | quit |

**Recommended order:** `p` → `h` → `s` → `m` → `1` (and `f`/`b` to pick direction).

> ⚠️ **`s` never swings.** PD_STAND just holds the nominal pose. A "tiny swing" in `s`
> is expected — it is **not** a swing at all. Only judge swing amplitude once the status
> line shows **`mode=MOTION level=1`**, and check that **`ts` is advancing** (the strike
> clock) and **`|act|` is oscillating** with the swing phase. See "Stage 9 — After
> validation" for the small-swing decision tree.

Useful flags: `--start passive|pd_stand|shadow|motion`, `--level 0|1`,
`--backhand` (scripted backhand instead of forehand),
`--gain-scale F` (try 0.4–0.8 if it buzzes; **0.4 under-drives the swing** — step up
with `]`), `--legs-passive` (hoist: legs hold, no balancing — **upper-body/waist swing
only**), `--swing-speed F`, `--no-publish`/`--dry-run` (= SHADOW),
`--warmup-sec S` (non-interactive safe start: hold PD_STAND S s then auto-switch to
`--start`), `--obs-csv PATH`, `--trace-csv PATH`, `--use-imu-yaw` (only with a real
world-yaw localizer; default OFF is hardware-safe).

A **consolidated CONFIG banner** prints at startup — eyeball `start_mode / level /
swing_dir / loc_mode / legs_passive / gain_scale / swing_speed / publish / model /
trace_csv / obs_csv` there before you touch anything.

**What to look for (acceptance):**
- all **6 state topics ready** (waist / leg / arm / neck / pelvis_imu / torso_imu);
- `sync_complete` / `sync_aligned` **stable**;
- policy **rate ≈ 50 Hz**, `infer_ms` < 20 ms, `halts ≈ 0`;
- **first-tick dump** sane (obs blocks, IMU, q_des/kp/kd all printed once);
- `projected_gravity ≈ [0, 0, -1]` when upright; `base_quat ≈ [1,0,0,0]`;
- `motion_anchor_pos ≈ 0` in `perfect_tracking` mode;
- `|action|` **bounded** and oscillating with swing phase (not monotonically diverging);
- `q_des/kp/kd` sane; neck does **not** buzz (passive); wrists steady;
- no immediate divergence.

**Cleanup between runs** (after both terminals quit):

**Where:** box **`hope`** (either Terminal 1 or 2 — directory doesn't matter; these are global pkills).

```bash
pkill -9 -x aimrt_main; pkill -9 -x iox-roudi; rm -f /dev/shm/iox*
```

> This test proves the I/O **contract** runs on a robot-shaped interface. It does **not**
> prove the policy is absolutely stable: AGI's free-base sim uses *explicit* Euler PD and
> can diverge ~0.1 s after the connect gap — a **sim-fidelity** artifact, not a transfer
> bug. Judge swing stability from the **hoist** variant and from the implicit-PD MuJoCo
> eval (Stage 2.2). **Never patch AGI's MJCF/PD.**

---

## Is the swing hard-coded?

**Short answer: the joint trajectory is _not_ hard-coded — it is a _learned_ swing.**
But the _target_ it swings at _is_ a scripted test value right now (no ball planner yet).
Be precise about which is which:

- **The 31-DOF joint trajectory is generated by the learned ONNX policy, every tick.**
  There is no recorded joint-angle sequence being replayed onto the motors. Each 20 ms
  the runner builds a 180-D observation, runs the policy, and decodes
  `q_des = default_q + action × action_scale`. Change the policy `.onnx` and the swing
  changes — nothing in C++ dictates the arm angles.
- **The forehand/backhand _posture_ is learned**, captured in the policy weights plus the
  reference clip the ONNX carries internally (baked at export, exposed as obs-independent
  side-outputs `command[0:62]` = ref joint_pos/vel). The policy tracks that reference and
  fills in balance/stabilisation. We did not author any pose in C++.
- **The current strike _target_ IS scripted.** `PpPolicy::ScriptedTarget` emits a fixed
  front-right target (`racket_pos_w ≈ (0.40, ∓0.22, 0.82)`, `racket_vel_w ≈ (1,0,0)`) and a
  periodic strike clock. This is a **bring-up test target**, not a real aim.
- **The forehand/backhand toggle only swaps the scripted target + clip.** Pressing `f`/`b`
  (or `--backhand`) flips the target's **y-sign** and the `swing_type` flag, and selects the
  matching baked clip (clip 0 forehand / clip 1 backhand). It does **not** load a new pose
  library and does **not** read the robot's surroundings — it just tells the (one) learned
  policy which side to swing.
- **There is no live ball planner yet.** No ball tracker, no trajectory estimate, no
  automatic forehand-vs-backhand decision. The runner cannot hit a real incoming ball today.
- **To hit real balls, replace `ScriptedTarget` with planner output** — hit position,
  velocity, racket normal and hit-time derived from a ball-trajectory estimator. The
  observation builder, ONNX policy, decode and joint map all stay exactly as they are; only
  the target source changes. That is the one remaining gap between "learned swing on a
  scripted target" and "ping-pong".

### Dataflow (one 50 Hz tick)

```
  RobotState (real)            Scripted (C++, no planner)         ONNX side-outputs (baked clip)
  q, dq  (31, SDK)             racket_target_pos_w / vel_w        ref joint_pos / joint_vel (31+31)
  pelvis IMU quat, gyro        swing_type (f/b)            -----> ref tracked-body poses (anchor)
  torso  IMU quat              time_to_strike (strike clock)            |
        |                              |  time_step_for(clip,tts)        |
        |                              v                                 |
        |                       time_step (clip frame) ----------------> PpOnnxPolicy.refs(time_step)
        |                              |                                 |
        +---------------+--------------+----------------+----------------+
                        v
            build_obs_180(...)  ->  180-D observation
              command(62) | anchor_pos_b(3) anchor_ori_b(6) | base_ang_vel(3)
              joint_pos_rel(31) joint_vel(31) last_action(31) | proj_grav(3)
              base_target_pos_b(2) racket_target_pos_b(3) racket_target_vel_w(3)
              time_to_strike(1) swing_type(1)
                        |
                        v
            PpOnnxPolicy.mean_action(obs, time_step)  ->  31-D action  (LEARNED)
                        |
                        v
            target_q = default_q + action × action_scale   (Isaac order)
                        |
                        v
            scatter Isaac -> 31 SDK slots (pp_joint_map)
                        |
            POST-ONNX OVERRIDES (the only hard overrides):
              • neck slots [3,4] -> q=0, kp=40, kd=2     (always passive)
              • legs slots [19..30] -> nominal           (only if --legs-passive)
              • q_des clamped to A3 joint limits          (safety; counts logged)
              • kp,kd × gain_scale                        (operator scalar)
                        |
                        v
            RobotCommand {q_des, kp, kd, dq_des=0, tau_ff=0}  ->  AGI backend (implicit PD)
```

### Source-of-truth table

| Component | Current source | Hard-coded / learned / sensor / future |
|---|---|---|
| 31-DOF joint actions | ONNX policy, recomputed every tick | **learned** |
| ref joint_pos / joint_vel (`command[0:62]`) | ONNX side-outputs (baked clip, indexed by `time_step`) | **learned** (baked reference) |
| ref tracked-body anchor pose | ONNX side-outputs (baked clip) | **learned** (baked reference) |
| racket target **position** | `ScriptedTarget` constant `racket_pos_w` | **hard-coded** (→ future planner) |
| racket target **velocity** | `ScriptedTarget` constant `racket_vel_w` | **hard-coded** (→ future planner) |
| racket **normal** / strike orientation | not an explicit input; implied by `swing_type` + learned posture | **learned** (no explicit normal yet → future planner) |
| `swing_type` (forehand/backhand flag) | `f`/`b` key or `--backhand` (flips y-sign + clip) | **hard-coded toggle** (→ future planner picks side) |
| `time_step` (strike clock / clip frame) | `time_to_strike` → `ClipLayout::time_step_for` | **hard-coded** schedule (→ future planner sets hit-time) |
| forehand/backhand **choice** | operator key / startup flag | **hard-coded** (→ future planner from ball) |
| neck command | ONNX output **discarded**; held q=0, kp=40, kd=2 | **hard-coded** passive override |
| leg command | ONNX output passes through, **unless** `--legs-passive` → nominal | **learned** (or held override on hoist) |
| waist_roll (and all) `q_des` clamp | `clamp_q_to_limits` to A3 URDF range | **hard-coded** safety net |
| kp / kd | ONNX metadata (`joint_stiffness`/`joint_damping`) × `gain_scale` | **learned** gains (operator-scaled) |
| base / torso orientation | real pelvis + torso IMU | **runtime sensor** |
| base / torso world **position** | reference (perfect_tracking) — no localizer | **hard-coded** stand-in (→ future localizer) |
| **incoming ball** | — | **missing** (→ future tracker + planner) |

**Bottom line:** the robot is doing a **learned** whole-body swing (policy → 31-DOF action,
decoded to `q_des` every tick), tracking a **learned** reference posture, aimed at a
**scripted** test target. It is not replaying a recorded pose, and it is not yet hitting a
real ball — the missing piece is a ball tracker + planner feeding `ScriptedTarget`'s slot.

> The runtime now states this explicitly: the **RUN CONFIG** banner prints
> `target_src = SCRIPTED … NO live ball planner`, `action_src = ONNX policy (LEARNED …)`,
> and `post_onnx = neck HELD | legs … | q_des CLAMPED`; the per-second `[obs]` block prints
> `racket_target_pos_b`, `racket_target_vel_w`, `swing=±1(FOREHAND/BACKHAND)`, `tts`, and
> `[SCRIPTED target -- no live planner]`.

---

## Stage 5 — Cross-compile for Rockchip / MDU

**Run on the HOST shell (NOT inside `distrobox enter hope`).** *Cross-compilation* =
build on the x86 dev box but produce an **aarch64 (ARM) binary** that runs on the robot's
Rockchip MDU. The build **needs Docker**: `build_a3_deploy_pkg.sh --arch rockchip`
re-invokes itself inside the `a3-rockchip-builder` Docker container, which carries the
aarch64 toolchain + sysroot + ROS (so `aarch64-linux-gnu-g++` not being on your PATH is
expected — the container has it). One command.

> ⚠️ **Docker-vs-distrobox trap (bites every time).** The **`hope`** distrobox has ROS
> but **no `docker`** → the build fails instantly with `docker: command not found`. Docker
> lives on the **host**. So this stage is the OPPOSITE of the x86 build: run it from a
> **host terminal** (`exit` the distrobox first). The host has **no ROS**, and that's fine
> — you do **not** `source /opt/ros/jazzy/setup.bash` here; the Docker image carries ROS.
> (`docker info` should print a server version; if not, you're still inside `hope`.)
>
> ⚠️ **Always re-run this build after editing the runner** (`pp_*` headers, `*main.cpp`)
> — `dist/a3_deploy_rockchip/` is NOT auto-rebuilt and will silently ship a stale binary.
> **Gate before shipping:** `strings dist/a3_deploy_rockchip/a3_deploy_onnx_ref_pingpong |
> grep -E "swing complete|clip layout from ONNX"` must print — these markers are UNIQUE to
> the 2026-07-02 binary (the single-swing + ONNX-clip-metadata fixes). Do **not** gate on
> `RUN CONFIG`/`f=forehand`: those are present in the stale Jul-01 binary too and give false
> confidence. Also confirm the binary mtime is from *this* build (see
> [[hope-a3-deploy-bringup-findings]] stale-binary detection).

```bash
# HOST terminal (not distrobox). Do NOT source ROS here.
cd ~/workspace/HOPE/agi/a3_deploy_example

find . -name '._*' -type f -delete                          # remove macOS junk (usually 0; harmless)
bash scripts/build_a3_deploy_pkg.sh \
  --arch rockchip \
  --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml \
  --jobs $(nproc)

# confirm the package + that the binary is really aarch64 AND has the latest runner:
ls -la dist/a3_deploy_rockchip/a3_deploy_onnx_ref_pingpong    # mtime must be from THIS build
file   dist/a3_deploy_rockchip/a3_deploy_onnx_ref_pingpong   # -> ELF 64-bit ... ARM aarch64
# Gate on markers UNIQUE to the 2026-07-02 binary. NOTE: "RUN CONFIG"/"f=forehand" are
# ALSO in the old Jul-01 stale binary, so they do NOT discriminate — use these instead:
strings dist/a3_deploy_rockchip/a3_deploy_onnx_ref_pingpong | grep -E "swing complete|clip layout from ONNX" \
  || echo "!! STALE BUILD (pre-07-02) — wipe build/a3_pkg_rockchip and rebuild before shipping"

ls -la dist/a3_deploy_rockchip/config/a3_runtime_config.pingpong.yaml
grep  model_path dist/a3_deploy_rockchip/config/a3_runtime_config.pingpong.yaml  # -> models/model_p4_deployparity.onnx
ls -la dist/a3_deploy_rockchip/models/model_p4_deployparity.onnx
ls -la dist/a3_deploy_rockchip/run_a3_pingpong.sh
```

### ⚠️ If `config/` or `models/` came out EMPTY after the build (host-staging failure — 2026-07-02)

The rockchip build compiles the binaries **inside Docker**, but the asset-staging step
(`stage_runtime_config_and_assets` — copies the ONNX into `models/`, writes the runtime cfg,
emits `run_a3_pingpong.sh`) runs **on the HOST afterwards** and needs `python3` + **PyYAML**
on the host. The host usually has neither, so staging **fails silently**: you get fresh
aarch64 binaries but `models/` is empty and `config/` has no `a3_runtime_config*.yaml`. The
MDU then dies with `terminate ... YAML::BadFile: config/a3_runtime_config.pingpong.yaml`.

**Fix — copy the arch-independent assets from the x86 dist.** The ONNX, the yaml configs and
the run scripts are **identical across arch** (only the ELF binaries differ), so build the x86
package first (Stage 3.3), then:
```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
SRC=dist/a3_deploy_x86_64; DST=dist/a3_deploy_rockchip
# runtime + aimrt configs (yaml/xml):
cp "$SRC"/config/a3_runtime_config.pingpong.yaml "$SRC"/config/a3_runtime_config.yaml "$DST"/config/
cp "$SRC"/config/a3_aimrt_config.yaml "$SRC"/config/a3_aimrt_config.iceoryx.yaml \
   "$SRC"/config/a3_aimrt_config.ros2.yaml "$SRC"/config/a3_aimrt_config.pingpong_iceoryx.yaml "$DST"/config/
# model (same onnx both arches):
mkdir -p "$DST"/models
cp "$SRC"/models/model_p4_deployparity.onnx "$DST"/models/
# run scripts (bash):
cp "$SRC"/run_a3_pingpong.sh "$SRC"/run_a3.sh "$SRC"/run_a3_probe.sh "$SRC"/setup_ros2_msgs.bash "$DST"/
# verify complete:
grep model_path "$DST"/config/a3_runtime_config.pingpong.yaml   # -> models/model_p4_deployparity.onnx (already correct — no edit)
ls -la "$DST"/models/model_p4_deployparity.onnx "$DST"/config/a3_runtime_config.pingpong.yaml
file  "$DST"/a3_deploy_onnx_ref_pingpong                        # -> aarch64
strings "$DST"/a3_deploy_onnx_ref_pingpong | grep -E "swing complete|clip layout from ONNX"
```
(Alternative: `pip install pyyaml` on the host and rebuild — but the copy is faster and
proven. The `model_path` copied from x86 is already `models/model_p4_deployparity.onnx`, so
no manual edit is needed.)

### Ship it to the MDU through the HDU jump host

`<HDU_WIFI_IP>` = the field Wi-Fi IP of the **HDU jump host** (you must be able to
`ping` it). `<MDU_IP>` = the robot's internal compute-unit address (default
`10.42.10.12`, confirm on-site).

**Where:** box **`hope`** (or the host shell — `rsync`/`ssh` don't need ROS) · `cd ~/workspace/HOPE/agi/a3_deploy_example` (the `dist/...` source path is relative to here).

```bash
cd ~/workspace/HOPE/agi/a3_deploy_example
ssh -J agi@<HDU_WIFI_IP> agi@<MDU_IP> 'mkdir -p /agibot/a3_deploy'
rsync -azP -e "ssh -J agi@<HDU_WIFI_IP>" \
  dist/a3_deploy_rockchip/ \
  agi@<MDU_IP>:/agibot/a3_deploy/
```

> The **trailing `/`** on the source path matters — without it, rsync nests an extra
> `a3_deploy_rockchip/` level under the target.

---

## Stage 6 — Real robot / hardware bring-up (safety-first, staged)

**Runs on the robot MDU** (the aarch64 binary you just shipped), driven over SSH through
the HDU jump host.

### 6.0 Safety preconditions — non-negotiable

- [ ] Robot is **hoisted / on the safety rope**; feet off the ground or under low-power support.
- [ ] **Physical e-stop in hand**; press it once and release to confirm the loop works.
- [ ] ≥ **1.5 m** clear of people/obstacles around the arm radius.
- [ ] You know the software stops: `p` (PASSIVE) · `q` (quit) · `Ctrl-C`.
- [ ] **Never** pass `--auto-start`. **Never** auto-launch directly into MOTION.
- [ ] Run **level 0 before level 1**, and start MOTION at **low gain** (`--gain-scale 0.4`).

### 6.1 MDU Terminal A — stop services, start EtherCAT

```bash
ssh -J agi@<HDU_WIFI_IP> agi@<MDU_IP>
sudo systemctl stop agibot_pm
source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0
bash scripts/hal_ethercat/start_hal_ethercat.sh
# keep this terminal OPEN — it is sending/receiving joint + IMU state.
```

### 6.2 MDU Terminal B — confirm the binary + sync, then stage up

```bash
ssh -J agi@<HDU_WIFI_IP> agi@<MDU_IP>
cd /agibot/a3_deploy
file ./a3_deploy_onnx_ref_pingpong          # MUST say aarch64 (else you shipped the x86 pkg)
source /agibot/software/v0/entry/env/env.sh # if it later says "required robot env not found"
```

Stage strictly in this order. `taskset -c 4-7` pins to the RK3588 performance cores to
cut inference-latency jitter. Keep `--gain-scale 0.4` until you've seen it stable.

```bash
RT=config/a3_runtime_config.pingpong.yaml

# 6.2a  DRY-RUN / receive-only (PASSIVE, limp): verify the 6 state topics + sync.
#   --single-swing = play the clip ONCE then auto-hold stand (no end->windup snap); press
#   `1` for each forehand. Recommended for robot bring-up. (Same session drives 6.2a–e;
#   you do NOT relaunch between key presses.)
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --legs-passive --gain-scale 0.4 --single-swing \
  --trace-csv /tmp/pp_trace.csv --obs-csv /tmp/pp_obs.csv
#   WATCH: 6 topics ready, sync_complete / sync_aligned stable.
#   (If it says "required robot env not found", source /agibot/software/v0/entry/env/env.sh first.)

# 6.2b  PROBE latency in SHADOW: press `h` (compute, NO publish). Read the FIRST-TICK DUMP.
#   WATCH: rate ≈ 50 Hz, infer_ms < 20 ms, proj_grav ≈ [0,0,-1], |action| bounded, robot still.

# 6.2c  PD_STAND: press `s`. Wait ~3 s and confirm a stable, quiet stand.

# 6.2d  WATCHDOG / E-STOP test: hit the PHYSICAL e-stop -> confirm safe-halt (< 200 ms).
#   Also confirm `p` and Ctrl-C both stop it. Re-arm only when you're ready.

# 6.2e  LOW-GAIN MOTION: press `m` (still --gain-scale 0.4). Then `0` (hold-windup) FIRST.
#   Only after level 0 is visibly stable: press `1` for ONE forehand (--single-swing auto-
#   holds stand after each swing; press `1` again to repeat). FOREHAND ONLY — never press
#   `b` (backhand is not deploy-ready). Watch neck/wrist.
```

Mapping to runner modes: PASSIVE=`p`/`--start passive`, SHADOW=`h`/`--dry-run`,
PD_STAND=`s`, MOTION=`m`, level=`0`/`1`, gain=`--gain-scale`/`[`,`]`.

> Note: AGI's **native rknn** harness (`run_a3.sh`, `run_a3_probe.sh`, `--dry-run`,
> `taskset`) in `HARDWARE_BRINGUP_CHECKLIST.md` is a separate first-pass that proves the
> EtherCAT/network/sync/PD_STAND/e-stop chain **with AGI's own policy** (it cannot load
> our 180/31 ONNX). It's worth running once to validate the hardware harness; our policy
> then runs through the `a3_deploy_onnx_ref_pingpong` binary as above.

### 6.2f Leg modes — upper-body vs whole-body vs pure policy (hard-won 2026-07-02)

The 6.2 flow uses `--legs-passive` = **upper-body only** (legs frozen at nominal; only
waist+arms swing) — stable, the safest first pass, but **not** whole-body. Beyond it there are
three modes; the right one depends on **how the feet bear weight**. `RT=config/a3_runtime_config.pingpong.yaml`.

**A. Upper-body only — `--legs-passive`.** Legs held stiff at nominal. Good for the first swing
on a hoist. (What 6.2 shows.)

**B. Whole-body, weight-bearing — `--auto-leg-hold`.** Legs+waist HELD at level 0 (stable ready
stand), RELEASED to the policy at level 1 (self-balancing swing); a squat/tilt guard reverts to
level 0 if a leg sinks or the body tilts. Do **not** just delete `--legs-passive` (that lets the
policy drive the legs even while standing → collapse). Feet must bear weight, so the held stand
needs real gains:
```bash
# feet on the ground, rope as a catch:
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --auto-leg-hold --official-stand --leg-gain-scale 1.0 \
  --gain-scale 0.4 --single-swing
```
- `--official-stand` gives the HELD legs+waist AGI's ground-stand gains so they don't sag
  (without it the held waist gets only policy_kp×0.4 ≈ 20 and the torso topples).
- Released-leg stiffness = **`--leg-gain-scale`** (policy kp × scale): start 1.0, step
  1.0→1.5→2.0 if the knees sink; add `--ankle-gain-scale 1.5` if it pitches forward.
- **Do NOT use `--leg-stand-gains`** here — it forces AGI's kp≈2000 onto the *released, moving*
  legs and **oscillates/buzzes into a protection trip** on a partially-hoisted base (observed
  2026-07-02). `--leg-gain-scale` is the safe middle knob (no kp2000 cliff).

**C. Pure policy ("和 policy 一样") — no holds, no scaling.** The policy drives all 31 joints with
the trained implicit-PD gains at gain 1.0 (neck stays passive by contract). This is the real
self-balancing policy and it **requires the robot fully on the ground bearing its own weight**
(rope slack, catch only) — the condition it was trained on; on a hoist it flails (OOD).
```bash
# robot fully on the ground, feet bearing full weight, e-stop in hand:
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --official-stand --single-swing
```
- KEY: with NO `--auto-leg-hold`/`--legs-passive`, `--official-stand` **only sets the `s`
  PD_STAND hold gains — it does NOT touch the MOTION policy gains** (`leg_official`/`waist_held_off`
  in `a3_pingpong_main.cpp` are both false without those flags). So `s` stands the robot up firmly
  and `m` hands off to the pure policy (knee kp≈250 trained, etc.) unchanged.
- Without `--official-stand`, `s` uses the gentle hoist PD (kp≈60) which **cannot bear weight on
  the ground → "no force", collapses.** PD_STAND is a hold mode, not the policy.
- `--gain-scale` omitted = 1.0 = exactly the policy. Lower it (e.g. 0.8) only to soften all gains
  uniformly (no freezing) for a safety margin — but 1.0 is the true policy.

> **Partial hoist is the WORST regime for gain tuning.** With the rope taking part of the weight
> (feet lightly loaded), kp2000 (`--official-stand` stand / `--leg-stand-gains`) buzzes, while
> gentle gains sag — you can't win. Commit to ONE clean state: **fully on the ground** (weight on
> feet, rope slack) for whole-body / pure policy, OR **fully hoisted** for an upper-body-only
> (`--legs-passive`) check.

### 6.3 Hardware verification gate

- [ ] dry-run: 6 topics ready; `sync_complete`/`sync_aligned` stable.
- [ ] probe: `infer_ms` < 20 ms; skew / sample-age within thresholds; watchdog not mis-firing.
- [ ] PD_STAND: stable quiet stand after `s`.
- [ ] e-stop: physical e-stop halts arm + gait **< 200 ms** (competition requirement;
      measure with 120 fps video or logs).
- [ ] MOTION: level 0 stable, then a recognizable forehand at level 1 — no fall.
- [ ] joint name/order matches the target machine; command vector length is 31.

### 6.4 Logs to capture for AGI staff

| log | how |
|---|---|
| **first-tick dump** | the one-shot block the runner prints on tick 0 (save stdout) |
| **trace CSV** | `--trace-csv /tmp/pp_trace.csv` (per-joint des·q·qd·kp·kd) |
| **obs CSV** | `--obs-csv /tmp/pp_obs.csv`, then `python scripts/analyze_obs_log.py /tmp/pp_obs.csv` |
| **sync/alignment logs** | runner stdout (rate / sync_complete / sync_aligned / halts) + `/tmp/iox-roudi.log` |

---

## After validation — next debugging steps (bring-up)

Once the contract is validated, this is how to debug the *behavior* on sim/hardware.

### A. Verify you are really in MOTION (not still holding)

The startup **CONFIG banner** and the 1 Hz status line tell you everything. A real,
driven swing shows **all** of:

- status `mode=MOTION` and `level=1` (in `s`/PD_STAND or `level=0` the robot does **not** swing);
- `ts=` **advancing** each second (the strike clock sweeping the baked clip, e.g. `…→28→78→0→…`);
- `|act|` (and `maxact`) **oscillating** with the swing phase, not flat ~0;
- `sdir=FOREHAND`/`BACKHAND` is what you intend.

> ⚠️ **SHADOW (`h`) FREEZES the swing clock.** The driver advances `tick_idx` only on
> *published* ticks, and SHADOW does not publish — so in SHADOW you see `ticks=0`,
> `rate=0`, **`ts=0` frozen**, and the policy runs against the **static windup frame**
> (`tts` constant). That's expected: SHADOW is for checking obs/IMU/IO health and the
> first-tick dump safely, **not** for previewing the swing. The clock (and a real swing)
> only run in **MOTION**. So capture the swing in MOTION, and read any `ts`-frozen /
> waist_roll numbers from SHADOW as the windup pose only, not the swing.

### B. Small-swing decision tree

Read the 1 Hz status + the per-joint `[diag]` `RIGHT ARM` block (`cmdR`=commanded range,
`measR`=measured range, `trk`=measR/cmdR):

| symptom in status/diag | meaning | fix |
|---|---|---|
| `mode=PD_STAND` / `level=0` | not swinging by design | press `m` then `1` |
| `mode=SHADOW`, `ts=0`/`ticks=0`/`rate=0` frozen | **SHADOW freezes the clock by design** (no publish ⇒ no tick) | press `m` (MOTION) to run the clock |
| `mode=MOTION` but `ts` stuck, `|act|≈0` | swing clock not advancing / ONNX not driving | check `level=1`, check obs (Stage 3/7), re-export |
| `cmdR` large but `measR` small, low `trk%` | **actuator/gain under-shoot** | raise `gain_scale` (`]` → 1.0); slow `swing-speed` (`,`); on AGI MuJoCo this is expected (next item) |
| swing fine in implicit MuJoCo, small in AGI MuJoCo | explicit-PD sim artifact | judge from Stage 2.2 (`--pd-mode implicit`) or hardware, not AGI's sim |
| small only at `--gain-scale 0.4` | over-conservative start | step gain to 1.0 with `]` |

### C. Explicit-PD MuJoCo under-shoot (why AGI MuJoCo looks weak)

AGI's `aimrt_mujoco_sim` drives `<motor>` actuators with **explicit PD, no gravity
comp** (`ctrl = kp·(q_des−q) + kd·(dq_des−dq)`). The soft training gains (waist/arm
kp 20–50) then sag/under-shoot: the waist droops ~0.45 rad and the arm under-shoots.
The **real MDU backend is implicit-PD** (matches Isaac training) and does **not** show
this. So: a forward lean or small swing **in AGI MuJoCo is a sim-fidelity artifact**,
not a transfer bug — confirm against the implicit-PD MuJoCo eval (Stage 2.2) or hardware.
**Never patch AGI's MJCF/PD.**

### D. Is `legs_passive` enabled?

Check the CONFIG banner `legs_passive = true/false`. With `--legs-passive` (the hoisted
default) the **legs are held at nominal** and only the **upper body + waist** swing — so
do not expect leg motion or balancing; it validates the arm/waist swing only.

### E. Inspect the `waist_roll` clamp/mismatch

The A3 `waist_roll` limit is **±0.349 rad** (`pp_joint_limits.hpp` slot 1). The policy can
command more roll than that; the runner **clamps `q_des` before publish** (never removed).
To see how much:

- **Live:** the status `clamp=N` field (joints clamped this tick) + a one-shot
  `WARN clamp-rate: waist_roll clamped on X% of ticks (max viol … rad)` when it's chronic.
- **Offline (source attribution):** `python scripts/analyze_obs_log.py /tmp/pp_obs.csv` prints a
  **waist_roll audit**: the baked **reference** `obs_5` range vs the policy **q_des**
  (`act_5·0.230`) range, the % over ±0.349, and whether the over-limit command comes from
  the **reference clip** (`obs_5` itself exceeds ±0.349 → retarget/soften the clip) or the
  **policy action** (reference in-range → action/ONNX; clamp keeps it safe, consider lower
  waist_roll gain). **Do not remove the clamp.** Acceptable resolutions: (1) accept + document
  the clamp; (2) soften the reference/action for waist_roll; (3) rely on the high-clamp WARN.

### F. Forehand vs backhand (scripted)

`m`+`1` validates **forehand** by default. For backhand, either start with `--backhand`
or press **`b`** at runtime (press **`f`** to go back). This flips the scripted target's
y-sign and selects the baked backhand clip; the status shows `sdir=BACKHAND`. (No live
planner yet — this is the scripted test path.)

### G. Capture a clean evidence package for AGI

Save to a timestamped dir: `trace.csv`, `obs.csv`, `first_tick.txt`, `status.log`.

**MuJoCo (box `hope`, two terminals).** Terminal 1 = the sim (Stage 4 Terminal 1, unchanged).
Terminal 2 = the runner with capture:

```bash
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}"
RUN=~/workspace/HOPE/agi/a3_deploy_example/logs/agi_capture/$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"

# start in SHADOW (computes, no publish) for the obs/IO health check, capture all:
./a3_deploy_onnx_ref_pingpong --runtime-cfg config/a3_runtime_config.pingpong.yaml \
  --start shadow --level 1 --legs-passive \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
#   keys: h (SHADOW: obs/IO health only — ts FROZEN, no swing) -> m (MOTION: clock runs)
#         -> 1 (swing) -> dwell ~10s -> b (backhand) dwell ~10s -> f -> q
#   NOTE: the swing is captured in the MOTION portion; SHADOW only logs the windup frame.
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
python scripts/analyze_obs_log.py "$RUN/obs.csv" "$RUN/trace.csv"   # writes *.report.txt
```

**MDU / hardware hoisted (staged, save logs).** Robot hoisted, e-stop in hand:

```bash
ssh -J agi@<HDU_WIFI_IP> agi@<MDU_IP>
cd /agibot/a3_deploy; source /agibot/software/v0/entry/env/env.sh
RT=config/a3_runtime_config.pingpong.yaml
RUN=/agibot/a3_deploy/logs/agi_capture/$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"

# DRY-RUN/PASSIVE receive-only -> SHADOW probe -> PD_STAND -> low-gain MOTION, all captured.
# keys: h (SHADOW: obs/IO health, ts FROZEN — no swing) -> s (PD_STAND) -> m (MOTION: clock runs)
#       -> 1 (ONE forehand; --single-swing auto-holds stand after, press 1 to repeat) -> q
# FOREHAND ONLY — never press `b` (backhand is not deploy-ready). Keep gain 0.4 for bring-up.
# The SWING is captured only in the MOTION portion; SHADOW/PD_STAND log holding poses.
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --legs-passive --gain-scale 0.4 --single-swing \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"

sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
# copy logs back through the jump host, then analyze on the dev box:
# rsync -azP -e "ssh -J agi@<HDU_WIFI_IP>" agi@<MDU_IP>:"$RUN"/ ./logs/agi_capture/from_mdu/
python scripts/analyze_obs_log.py logs/agi_capture/from_mdu/obs.csv logs/agi_capture/from_mdu/trace.csv
```

> The runner writes only `trace.csv`/`obs.csv`; `status.log` is the `tee`'d stdout+stderr
> (contains the CONFIG banner, 1 Hz status, and the first-tick dump), and `first_tick.txt`
> is sliced out of it. `analyze_obs_log.py` saves a `*.report.txt` next to each CSV.

---

## Hoisted Full-Body Verification

The policy outputs a **31-DOF** action every 20 ms — waist, arms, **and legs**. The upper-body
hoist demo runs with `--legs-passive`, which **overwrites leg q_des to nominal**, so it does
**not** exercise the leg policy at all. This section turns the legs back on (`legs_passive=false`)
to verify the **full-body command path** while still hoisted — and is explicit about what that
does and does **not** prove.

### Three distinct things (do not conflate them)

| # | Test | Flag | Validates | Does **not** validate |
|---|---|---|---|---|
| 1 | **Upper-body hoist demo** | `--legs-passive` (legs_passive=**true**) | arm/waist swing, obs/action/yaw/backend path | **any** leg policy output (legs are held nominal) |
| 2 | **Full-body command verification (hoisted)** | **no** `--legs-passive` (legs_passive=**false**) | 31-DOF commands are **produced, bounded, synchronized, not railing**; legs/waist *can* follow | **real balance** — the robot is suspended, feet bear no load |
| 3 | **True full-body balance test** *(later, not now)* | legs_passive=false, **feet on ground** | actual standing/balance under the policy | — (this is the goal of a future stage) |

> ⚠️ **Read this before you start.**
> - If `legs_passive=true`, **you are not testing full-body** — legs are pinned to nominal.
> - If the robot is **hoisted**, you **cannot claim balance success** — feet bear no load, so
>   measured leg motion will look strange and tracking will be poor. That is **expected**.
> - **True full-body balance requires a later ground-contact test** with a support rope / frame,
>   **low gain**, and **level 0 first**. Do **not** attempt it from this section.
> - The goal **here** is narrow and safe: confirm the **31-DOF command is generated and safe**
>   (finite, bounded, sync'd, clamp not exploding, legs actually commanded), so we know the
>   full-body policy is *active and well-behaved* before we ever put weight on the feet.

**What every stage looks at** (1 Hz status + analyzer):
`legs_passive` · `mode` · `level` · `gain` · `sdir` · `ts` (strike clock) · `|act|`/`maxact` ·
`clamp` · `halts` · `sync_miss`, plus the new **`[fullbody]`** line:
`waist/Lleg/Rleg cmdR (commanded range) vs measR (measured range)`.

**PASS / WARN / FAIL** (from `analyze_obs_log.py`):
- **PASS** — `sync_miss=0`, no NaN/Inf, no halt, `q_des` finite, qd bounded, clamp rate low,
  **and leg q_des is present when `legs_passive=false`**.
- **WARN** — leg **tracking** is poor but commands are bounded (**expected** on a hoist — feet
  are unloaded), or a joint clamps occasionally.
- **FAIL** — `q_des` non-finite, **qd spikes**, **clamp count explodes**, **sync misses**, or
  the robot **halts**. *Stop immediately.*

> ⚠️ **Prerequisite — use a freshly staged ping-pong dist.**
> `build_a3_deploy_pkg.sh --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml`
> now preserves the ping-pong runtime-config basename, stages `model_15200.onnx`,
> and emits `run_a3_pingpong.sh`. If you are using an **older dist** or built with
> `--build-only`, you can still hit `YAML::BadFile` because the ping-pong cfg/model
> never got copied into `dist/`. Symptom:
> ```
> terminate called after throwing an instance of 'YAML::BadFile'
>   what():  bad file: config/a3_runtime_config.pingpong.yaml
> ```
> Fix — rebuild or restage (x86 example; for the MDU stage into `dist/a3_deploy_rockchip/`):
> ```bash
> cd ~/workspace/HOPE/agi/a3_deploy_example
> bash scripts/build_a3_deploy_pkg.sh \
>   --arch x86_64 \
>   --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
> # verify wrapper + cfg alias + model:
> ls dist/a3_deploy_x86_64/run_a3_pingpong.sh \
>    dist/a3_deploy_x86_64/config/a3_runtime_config.pingpong.yaml \
>    dist/a3_deploy_x86_64/models/model_15200.onnx
> ```
> (The `--build-only` flag additionally skips **all** asset staging — only use it for a pure
> compile check, never to produce a runnable dist.)

---

### Stage A — MuJoCo hoist, upper-body baseline (`legs_passive=true`)

Establishes the known-good baseline: arms/waist move, **legs held nominal**.

```bash
# Terminal 1 — AGI MuJoCo hoist sim + iceoryx (same as Stage 4, Terminal 1):
distrobox enter hope
cd ~/workspace/HOPE/agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/install/bin
source /opt/ros/jazzy/setup.bash
for p in ros2_plugin_proto aimrt_msgs joint_msgs mujoco_sim_msgs; do source ../share/$p/local_setup.bash; done
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd):$(pwd)/../lib"
pkill -9 -x aimrt_main 2>/dev/null; pkill -9 -x iox-roudi 2>/dev/null; rm -f /dev/shm/iox*
setsid ./iox-roudi >/tmp/iox-roudi.log 2>&1 </dev/null & sleep 1
MUJOCO_GL=egl ./aimrt_main --cfg_file_path=./cfg/a3_pingpong_hoist_cfg.yaml
```

```bash
# Terminal 2 — runner, UPPER-BODY baseline (legs held). Capture to its own dir:
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}"
RUN="$HOME/workspace/HOPE/agi/a3_deploy_example/logs/agi_capture/mujoco_upperbody_hoist"; mkdir -p "$RUN"
./a3_deploy_onnx_ref_pingpong \
  --runtime-cfg config/a3_runtime_config.pingpong.yaml \
  --start passive --level 1 --legs-passive \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: h (SHADOW) -> s (PD_STAND) -> m (MOTION) -> 0 (windup) -> 1 (swing) -> dwell ~10s -> q
```
**Expect:** banner shows `legs_passive = true`; arms/waist swing; the `[fullbody]` line shows
`Lleg cmdR≈0.000 Rleg cmdR≈0.000` (legs held). This is the reference, **not** a full-body test.

### Stage B — MuJoCo hoist, full-body command check (`legs_passive=false`, low gain)

Same sim. Drop `--legs-passive` so the policy drives the legs; start at **low gain**.

```bash
# Terminal 1 — same MuJoCo hoist sim as Stage A (restart it if you stopped it).
# Terminal 2 — runner, FULL-BODY (legs policy-driven), low gain, own capture dir:
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example/dist/a3_deploy_x86_64
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}"
RUN="$HOME/workspace/HOPE/agi/a3_deploy_example/logs/agi_capture/mujoco_fullbody_hoist"; mkdir -p "$RUN"
./a3_deploy_onnx_ref_pingpong \
  --runtime-cfg config/a3_runtime_config.pingpong.yaml \
  --start passive --level 1 --gain-scale 0.25 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: h (SHADOW) -> s (PD_STAND) -> m (MOTION) -> 0 (windup, hold here, watch legs) ...
#       ... ONLY once level 0 is calm: 1 (swing) -> dwell ~10s -> q
```
**Expect:** banner/status show `legs_passive = false` / `legs_passive=0`; `trace.csv` has 31 joints;
**leg q_des changes over time**; measured leg q follows *somewhat*; `clamp` not exploding; `halts=0`;
`sync_miss=0`; rate ≈ 50 Hz.
> **Leg motion in the hoist may look strange** — the feet aren't supporting the robot, so the
> policy's balance corrections have nothing to push against. Bounded + non-diverging is the bar
> here, **not** realistic leg motion.

**Reference — what a PASSing hoist full-body run actually looks like** (MuJoCo, gain 0.25, ~3200 MOTION ticks):
the legs are commanded through **large** ranges and follow them (`full-body: legs driven` PASS,
`leg tracking ~78%`, knees/ankles 90–101%), but with **hip cmdR ≈ 4 rad** and **leg qd ≈ 24 rad/s**,
and **heavy clamping** (knees pinned at the −0.122 rail ~99%, waist_roll ±0.349 ~65%, ankles ~55%).
That **looks alarming but is the expected out-of-distribution (OOD) hoist artifact**: training assumed
**planted feet bearing weight**, so the hanging-leg observation drives the policy to whip the unloaded
legs through big arcs. **This is exactly why a hoist is not a balance test.** PASS here = `legs driven`
+ `q_des finite` + `qd bounded` (no divergence) + `sync_miss=0` + `OVERALL≠FAIL`; the large amplitudes /
clamping are **WARN-level, expected**, and must **not** be read as ground-ready leg behavior. On the
ground (feet planted, in-distribution) the leg commands should shrink to small stabilizing motions —
that is what the later ground-contact test checks. **Because the hoist legs flail this much, do NOT step
`--gain-scale` up toward 1.0 from this run, and treat the same big leg `cmdR` on hardware (Stage C) as
expected in SHADOW only.**
>
> Sim-side hiccup to ignore: a brief `rate→0Hz` with `halts` climbing while `ticks` are **frozen**
> (then recovering, `sync_miss` stays 0) is a MuJoCo Terminal-1 stall (common with `MUJOCO_GL=egl`),
> **not** a policy fault.

### Stage C — Real robot **hoisted**, **shadow-only** full-body check (no publish)

Robot **stays hoisted**, **physical e-stop in hand**. `--start shadow` computes the full 31-DOF
command but **never publishes** it — the safest possible first look at leg commands on hardware.

```bash
# On the MDU (via the HDU jump host), in the shipped rockchip dist dir.
# Terminal A must already be running EtherCAT/state (Stage 6.1). Then in Terminal B:
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_fullbody_shadow; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start shadow --level 1 --gain-scale 0.25 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# Stays in SHADOW (no publish). Let it run ~20–30 s. Ctrl-C / q to stop.
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
**Confirm from status/first-tick:** `legs_passive=0`; `mode=SHADOW`; 31-DOF `q_des` exists;
**leg q_des nonzero / changing** (the `[fullbody]` `Lleg/Rleg cmdR > 0`); `|act|`/`maxact` bounded;
`q_des` bounded (clamp small); kp/kd sane in the first-tick dump; `sync_miss=0`.

> ⚠️ **SHADOW clock — read this or you'll misread the run.** SHADOW does not publish, and the
> driver's `ticks`/`rate` are **publish-gated**, so in SHADOW **`ticks=0`, `rate=0.0Hz` is normal**
> (the policy *is* running — `infer` happens every cycle — it just isn't counted). The runner now
> **free-runs the swing clock in SHADOW** so the obs evolves through the real swing: watch **`ts`
> advance** (e.g. 0→…→94) and **`|act|` oscillate** — that is your representative preview. The
> `[fullbody]` `Lleg/Rleg cmdR` should then show the legs moving across the swing.
> *If instead `ts` stays pinned (e.g. `ts=0`) and `|act|` converges to a single value (~25–27) with
> `clamp` firing on one joint ~99%*, you're on a **stale binary without the free-clock fix** — the
> policy is frozen on the windup frame and the action saturates there. That's not instability, but
> it's not a useful preview either → rebuild + reship (below). (To force the old behavior:
> `--shadow-frozen-clock`.)
>
> **Stale-binary check (do this first):** gate on markers **unique to the 2026-07-02 binary**
> — `swing complete` and `clip layout from ONNX`. (Do NOT gate on `target_src` / `fullbody` /
> `RUN CONFIG` / `auto_hold`: the stale Jul-01 rockchip binary has all of those too, so they
> give false confidence.) If the unique markers are missing, the MDU is running an **old
> rockchip binary** — cross-build from the HOST (the builder auto-stages the model from the
> runtime cfg; **no manual `cp` of the ONNX**), then reship:
> ```bash
> # HOST shell (NOT distrobox — needs docker, no ROS):
> cd ~/workspace/HOPE/agi/a3_deploy_example
> scripts/build_a3_deploy_pkg.sh --arch rockchip \
>   --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml --jobs $(nproc)
> # verify it's ARM, points at p4, AND has the 07-02 code, then reship:
> file    dist/a3_deploy_rockchip/a3_deploy_onnx_ref_pingpong          # -> aarch64
> grep    model_path dist/a3_deploy_rockchip/config/a3_runtime_config.pingpong.yaml  # -> models/model_p4_deployparity.onnx
> strings dist/a3_deploy_rockchip/a3_deploy_onnx_ref_pingpong | grep -E "swing complete|clip layout from ONNX" | head
> rsync -azP -e "ssh -J agi@<HDU_WIFI_IP>" dist/a3_deploy_rockchip/ agi@<MDU_IP>:/agibot/a3_deploy/
> ```
> (MDU deploy dir = `/agibot/a3_deploy` — the runner's `Executable Path` in the AimRT init report.
> Use your real IPs; the bring-up box has been jump `agi@172.19.81.9`, MDU `agi@10.42.10.12`.)

### Stage D — Real robot **hoisted**, low-gain full-body **MOTION level 0**

Only after Stage C is clean. **Still hoisted, e-stop in hand.** Publishes, but holds the windup
(no swing) so you watch the legs under closed-loop command at low gain.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_fullbody_motion_l0; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --level 0 --gain-scale 0.25 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: h (SHADOW probe) -> s (PD_STAND, settle) -> m (MOTION) -> stay at level 0 (windup).
#   Do NOT press 1 yet. Watch the legs for ~15–20 s, then q.
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
**Watch (stop on any):** leg twitching / violent motion; `qd` spikes (qdpk in the diag); the
**clamp-rate WARN**; `halts` incrementing; tracking %; `sync_miss` rising. Bounded, quiet legs = good.

### Stage E — Real robot **hoisted**, low-gain full-body **MOTION level 1**

Only after level 0 is stable. **Raise gain slowly with `]`** (not straight to 1.0), then try a brief swing.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_fullbody_motion_l1; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --level 0 --gain-scale 0.25 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: s -> m -> (level 0 stable) -> ] ] (step gain up, pause at each) -> 1 (brief swing) -> dwell ~10s -> q
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
> **This still does not prove full-body balance** — only full-body **command safety** while
> hoisted. Balance is the later ground-contact test (rope, low gain, level 0), not this.

### Pull logs + analyze (all stages)

```bash
# On the dev box. MuJoCo (Stage A/B) logs are already local under logs/agi_capture/.
# MDU (Stage C/D/E): pull each RUN dir back through the HDU jump host (use YOUR IPs;
# the bring-up box has been jump=agi@172.19.81.9  MDU=agi@10.42.10.12).
# NOTE: --mkpath creates the nested dest dirs; plain rsync only makes the LAST
# component, so a missing parent fails with: mkdir "...motion_l0" failed: No such file.
# (rsync < 3.2.3 lacks --mkpath -> run `mkdir -p logs/agi_capture/mdu_fullbody/{shadow,motion_l0,motion_l1}` first.)
rsync -azP --mkpath -e "ssh -J agi@<HDU_WIFI_IP>" agi@<MDU_IP>:/tmp/pp_fullbody_shadow/    logs/agi_capture/mdu_fullbody/shadow/
rsync -azP --mkpath -e "ssh -J agi@<HDU_WIFI_IP>" agi@<MDU_IP>:/tmp/pp_fullbody_motion_l0/ logs/agi_capture/mdu_fullbody/motion_l0/
rsync -azP --mkpath -e "ssh -J agi@<HDU_WIFI_IP>" agi@<MDU_IP>:/tmp/pp_fullbody_motion_l1/ logs/agi_capture/mdu_fullbody/motion_l1/

# Analyze (obs.csv first, then trace.csv — trace.csv carries the full-body audit):
python3 scripts/analyze_obs_log.py \
  logs/agi_capture/mujoco_fullbody_hoist/obs.csv logs/agi_capture/mujoco_fullbody_hoist/trace.csv
python3 scripts/analyze_obs_log.py \
  logs/agi_capture/mdu_fullbody/shadow/trace.csv \
  logs/agi_capture/mdu_fullbody/motion_l0/trace.csv \
  logs/agi_capture/mdu_fullbody/motion_l1/trace.csv
```
The trace report now includes a **`===== full-body (31-DOF) verification =====`** block:
per-joint `desR/measR/trk%` for waist+legs, leg `q_des` changing %, L/R leg max `|q_des|` and
max `|qd|`, waist max `|q_des|` + rail-ticks, top-10 command movers, top-10 mis-trackers, and
checks: **`full-body: legs driven`**, **`leg qd bounded`**, **`leg tracking (hoist)`** (WARN if
poor — expected), **`no joint frequently railed`**.

### What to paste back for diagnosis

1. The **`RUN CONFIG`** banner (confirms `legs_passive`, `gain_scale`, `level`, `loc_mode`).
2. A few **`[status]`** lines — especially `legs_passive=`, `sync_miss=`, `halts=`, `clamp=`, `maxact=`, `ts=`.
3. The matching **`[fullbody]`** lines (`waist/Lleg/Rleg cmdR vs measR`).
4. The one-time **`[pp FIRST-TICK DEBUG]`** block (or `first_tick.txt`).
5. Any **`WARN`/`[pp WARN]`** lines (clamp-rate, secondary-IMU, q_des-clamped).
6. The analyzer's **`OVERALL`** + the **full-body block** from the trace `*.report.txt`.

---

## Ground-Contact Full-Body Bring-up

This is the **balance** stage — the first time the robot bears its own weight under the learned
policy. It is categorically riskier than the hoist: the failure mode is now **falling**, not a
twitch in mid-air.

> ⛔ **Read before you start — non-negotiable.**
> - **Hoisted full-body PASS does NOT imply ground-contact balance PASS.** A hoist proves the
>   commands are *generated and bounded*; it proves *nothing* about balance.
> - **`level 0` is NOT a stable hold** — it freezes the policy at an OOD windup frame and twitches.
>   **For a static hold, use `PD_STAND` (`s`), never MOTION level 0.**
> - **The transition into MOTION is the main risk** — engaging MOTION jumps the legs from the
>   stand/default pose to the policy's windup pose (a ~2 rad kick was seen even at gain 0.25 on
>   the hoist). Never step straight from PD_STAND to full-gain MOTION level 1.
> - **Use very low gain and short bursts first.** Start `--gain-scale 0.05–0.10`.
> - **Do NOT connect a live ball planner or hit real balls** until ground-contact swing is stable.
> - **Safety rope / support frame taut, physical e-stop in hand, area clear** at every stage.

**PASS / WARN / FAIL for ground-contact** (combine the analyzer with what you see + the rope):
- **PASS** — `sync_miss=0`, no NaN/Inf, no halt, qd bounded, clamp rate low, robot stays
  **supported/upright** (`upright (gravz~-1)` PASS), feet do not slip badly.
- **WARN** — hip tracking poor but bounded, occasional clamp, mild rope load, the known level-0
  twitch, waist_roll over-limit (clamped).
- **FAIL — STOP / e-stop** — violent leg motion, **foot slip**, qd spike, frequent rail/clamp,
  `sync_miss>0`, `halts` incrementing, **torso IMU missing**, robot **loses balance**, or you had
  to hit the e-stop. (`halts`, foot-slip and rope-load are **not** in the CSV — read `halts` from
  the `[status]` line; foot-slip + rope-load are **visual**.)

All MDU commands run in the shipped deploy dir on the robot (`/agibot/a3_deploy`), Terminal B,
with **Terminal A already running HAL/EtherCAT** (Stage 6.1).

> 🔑 **CRITICAL FINDING (2026-06-30): on the ground the LEGS need their own gain.** A uniform low
> `--gain-scale 0.05` makes the leg kp (trained ~150–250) far too soft (~10) → **knees sag, robot
> falls forward** (only the rope holds it). The hoist hid this (unloaded feet). PD_STAND stood fine
> because `--official-stand` uses stiff knee kp (~2000). **A single global gain can't win** — legs
> need HIGH gain to stand, arms want LOW gain for a gentle swing. So the runner now has:
> - **`--leg-gain-scale F`** — scales ONLY the legs (slots 19–30); `--gain-scale` keeps arms/waist
>   gentle. Default: legs follow `--gain-scale` (hoist/legacy). **On the ground set `--leg-gain-scale`
>   ~0.5–1.0** so the knees can bear weight. (Neck stays at its fixed PD.)
> - **`--ankle-gain-scale F`** — scales ONLY the 4 ankle slots (L/R pitch+roll), independent of
>   `--leg-gain-scale`. **The ankle is the standing-balance joint** (ankle strategy: ankle_pitch
>   torque holds the CoM over the feet) and its *trained* kp is low, so a modest leg-gain still
>   leaves the ankle too soft → the robot **pitches forward about the ankle** (gravX climbs, would
>   fall). Stiffen the ankles here (e.g. 1.0, or **>1.0** if it still tips — stiffer than training)
>   while hips/knees stay gentle. Default: follow `--leg-gain-scale`.
> - **`--motion-blend-sec S`** (default 0.5) — ramps `q_des` from the entry pose into the policy
>   target over S seconds so the now-stiff legs **don't snap** through the ~1.5–2 rad windup kick.
>
> So the ground gain strategy is: **stiff ankles (don't tip) + enough leg gain (bear weight) +
> gentle arms (swing) + blended entry.** If the robot tips FORWARD, the first knob is
> `--ankle-gain-scale`, not leg-gain (cranking all legs makes the swing violent for no balance gain).

> 🛑 **SUPERSEDING VERDICT (2026-06-30, decisive ground MOTION data): gain knobs do NOT make the
> policy-driven legs stand on the ground — it's the contact-OOD wall. Ship legs-held instead.**
> With the cleanest run (`--official-stand --leg-gain-scale 0.5 --ankle-gain-scale 1.0 --level 0`),
> PASSIVE→PD_STAND held perfectly upright (gravX −0.01), but pressing `m` drove the LEGS through
> huge ranges — `left_hip_pitch` cmdR **2.107 rad**, `right_knee` 1.19, **`right_ankle_pitch`
> clamped 81% of ticks, 3.535 rad past limit** — which **lifted a foot** (`baseZ` 0.95→1.057) and
> tipped to gravX 0.40 until it crashed. This **disproves the ankle-gain-starvation theory**: the
> policy is *commanding* the legs/ankle far out of range, not sagging from a soft ankle, so a
> stiffer ankle just follows the bad command harder (that's why `--ankle-gain-scale 2.0/3.0` never
> survived to MOTION). The policy was trained with a reference-driven pelvis + planted feet; on real
> ground its windup leg motions lift a foot. **Ground-stand is purely a GAINS story, not pose:**
> `official_stand_q == stand_q == default_q` (identical) — only `--official-stand` (stiff per-joint
> `a3_pd_stand_kps/kds`) stands free; non-official PD_STAND (uniform kp=60) is too soft and falls.
>
> **FIX (front-end only — never patches MJCF/PD): `--legs-passive --official-stand` together** now
> holds the legs at the (identical) nominal pose with **AGI's official ground-stand gains verbatim**
> (the config proven to stand free), while waist+arms swing via policy. Banner shows
> `leg_hold = official`. `--legs-passive` **alone** keeps the trained leg PD (HOIST only — official
> gains buzz a hoisted robot). Stale-binary gate adds `grep leg_hold`. **Residual risk:** the
> official stand is a STATIC hold (no active rebalance) — expect it to hold a gentle **level-0** arm
> windup, but a full **level-1** swing's dynamic CoM shift may still tip it (keep the rope on).
> True legs-in-the-loop full-body needs a **retrain** (Path B: enforce A3 ankle/hip ranges in the
> action space + a foot-contact / no-lift reward). See Stage G5′ below.

### Stage G0 — Pre-flight checklist (do every item)

- [ ] Robot **on the ground** but still on the **safety rope / support frame**; rope taut.
- [ ] **Physical e-stop in hand**; area around legs + arms **clear**.
- [ ] **HAL/EtherCAT running** (Terminal A) — the 6 state topics are publishing.
- [ ] **Fresh binary** deployed and it has the new strings:

```bash
cd /agibot/a3_deploy
file ./a3_deploy_onnx_ref_pingpong     # -> ELF 64-bit ... aarch64
strings ./a3_deploy_onnx_ref_pingpong | grep -E "swing complete|clip layout from ONNX" | head
#   MUST print BOTH — they are unique to the 2026-07-02 binary. (Do NOT rely on
#   fullbody/auto_hold/target_src: the stale Jul-01 binary has those too.)
#   Missing -> stale binary -> rebuild rockchip with the pingpong --runtime-cfg + reship (Stage 5 / re-stage box).
```

- [ ] **Config points at the intended ONNX** and it exists:

```bash
grep -E '^[[:space:]]*model_path' config/a3_runtime_config.pingpong.yaml   # -> models/model_p4_deployparity.onnx
ls -la models/model_p4_deployparity.onnx
```

- [ ] In the **RUN CONFIG banner** (printed at startup) confirm: `legs_passive = false`,
  `loc_mode = perfect_tracking(B)`, `gain_scale = 0.05` (or `0.10`), `level = 0`,
  and `trace_csv`/`obs_csv` set (logging enabled).

### Stage G1 — Ground-contact SHADOW (no publish)

Feet on ground, **rope bearing the weight** (SHADOW does **not** drive the robot — our runner
computes but publishes nothing). Verifies the obs/command pipeline against **real ground-contact
state** (planted feet → different joint angles + IMU than the hoist) before anything is published.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g1_shadow; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start shadow --level 1 --gain-scale 0.05 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# stays in SHADOW (no publish). run ~20–30 s, then q.
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
**Confirm:** `sync_miss=0`; `sec_imu=1` (torso IMU present — first-tick `SAFETY:` line);
`proj_grav ≈ [0,0,-1]`; `q_des` finite; **leg q_des present/changing** (`[fullbody]` `Lleg/Rleg
cmdR > 0`); `clamp` reasonable; kp/kd `[20,250]/[2,8]`; no NaN/Inf; **`ts` advances** (the free-running
SHADOW clock — judge liveness by `ts`, not `rate`/`ticks`, which stay 0 in SHADOW).

### Stage G2 — PD_STAND on the ground (quiet static hold)

`PD_STAND` is the **only** proper static hold. On the ground you need the **production stand gains**
(`--official-stand`, knee kp ~2000) — the default gentle flat PD (kp=60) is a hoist setting and will
**sag** under weight. Engage it gently with the rope taut.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g2_stand; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --official-stand \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: confirm PASSIVE is limp (rope holds) -> press s (PD_STAND) -> robot holds the nominal pose.
#   Let it settle ~10–20 s. Watch it stand quietly. Then q (or p to go limp on the rope).
```
**Confirm:** robot holds the **default pose quietly** on the ground; `gravZ ≈ -1` (upright), small
`baseZ` drift; measured `q`/`qd`/IMU steady. **Do not judge swing here — PD_STAND is not the policy
swing.** If it leans, you may nudge `--stand-kp/--stand-kd`, but stay conservative — do **not** chase
aggressive gains.

> ⚠️ **`gravZ` in PASSIVE/PD_STAND needs the `observe_imu` fix (binary built 2026-06-30 or later).**
> On older binaries `ComputeCommand` only runs in SHADOW/MOTION, so in PASSIVE/PD_STAND the status
> shows a **frozen `gravZ=-1.00 / grav=[0,0,-1] / baseZ=0.950`** (the code defaults) and **no
> `[diag]` block** — that is NOT a real "upright" reading. Check: if G1 SHADOW showed a real tilt
> (e.g. `[0.10,0.07,-0.99]`) but G2 PD_STAND shows exactly `[0,0,-1]`, you're on a pre-fix binary →
> **judge G2 uprightness VISUALLY + by the rope going slack**, not from `gravZ`. The fixed binary
> reads the real IMU in every mode (so the analyzer's `upright (gravz~-1)` check is then meaningful
> for PD_STAND captures too).

### Stage G3′ — Ground swing with legs HELD at official gains (RECOMMENDED / shippable)

This is the path that **works on the ground today** (see the SUPERSEDING VERDICT above): the legs are
held at the nominal stand pose with **AGI's official ground-stand gains** (the proven-to-stand
config) while **waist + arms swing via the policy**. It sidesteps the contact-OOD leg commands that
make full-body MOTION lift a foot. `--legs-passive --official-stand` together select this mode
(banner: `legs_passive = true`, `leg_hold = official`).

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g3p_legsheld; mkdir -p "$RUN"
# legs HELD @ official ground-stand gains (verbatim; --leg/--ankle-gain-scale are IGNORED for the
# held legs). Arms/waist gentle. Rope ON for the first run.
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --official-stand --legs-passive --level 0 --gain-scale 0.30 --motion-blend-sec 0.8 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: s (PD_STAND official — should stand free; let the rope go slack to confirm)
#    -> m (MOTION level 0: legs HOLD, arms+waist do the windup) -> watch gravX ~10 s
#    -> 1 (ONE short swing burst, rope still on) -> p -> q.
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
**Confirm in the banner:** `legs_passive = true` AND `leg_hold = official` (if it says `trained`, you
forgot `--official-stand` → legs use the trained PD and may not hold). **PASS:** PD_STAND stands with
the rope slack; in MOTION the `[fullbody]` line shows **`Lleg/Rleg cmdR≈0, measR≈0`** (legs held, not
driven), `gravX` stays near 0 through the level-0 windup, `halts=0`, `sync_miss=0`. **The level-1
swing is the risk point** — the static leg hold does not actively rebalance, so a vigorous swing's CoM
shift can still tip it; keep the rope on and watch `gravX`. If level-0 holds but level-1 tips, that is
the expected limit of a static-leg hold → true full-body needs the retrain (Path B).

> ✅ **Verified 2026-06-30:** legs held cleanly (`Lleg/Rleg cmdR=0`), pelvis upright (`gravX≈±0.02`,
> `gravZ −1.00`) through a full **level-1** forehand on the ground, `halts=0`/`sync_miss=0` — the
> upper-body swing runs without lifting a foot. **Observed limit:** the swing still tips the robot
> forward ("腰往下倒") — the policy drives `waist_pitch` to its forward A3 limit (des +0.419, clamped
> ~39%) and the forehand arms reach forward → CoM past the static base of support.

**Escalation if it tips forward (front-end only — the FLOOR before retrain):**
1. **Freeze the waist too** with `--waist-passive` (with `--official-stand` it holds the waist at
   nominal with official gains → banner `waist_hold = official`). This makes the swing **arms-only**
   and keeps the heavy torso rigidly upright over the feet, removing the dominant forward-tip
   contributor (the soft-gain forward-leaning waist):
   ```bash
   taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
     --start passive --official-stand --legs-passive --waist-passive --level 0 --gain-scale 0.30 --motion-blend-sec 0.8 \
     --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
   ```
2. If it **still** tips, lower the arm gain (`--gain-scale 0.15` then `0.10`) so the arms reach less
   aggressively (gentler reach = smaller CoM shift, at the cost of a smaller swing).
>
> ✅✅ **VERIFIED 2026-06-30 — Tier-1 PASS (shippable):** `--legs-passive --waist-passive
> --official-stand --level 1 --gain-scale 0.30` → the robot STANDS FREE with the rope slack and
> executes the arms-only forehand on the ground (waist frozen upright, pelvis `gravX 0.05–0.07`,
> legs held, `halts=0`/`sync_miss=0`). Residual: a slight forward lean during the forward arm reach.
> That lean is **not** "legs too stiff" (a stiffer ankle resists tilt *more*) — it's the CoM moving
> forward with the arm reach and no active rebalance to counter it (frozen legs can't lean back).
> Softer legs would sag, not help. Reduce it with gentler arm reach (`--gain-scale 0.15→0.10`); a
> lean-free *full* swing needs the retrain (Path B).
>
> ⚠️ **This is the floor of front-end freezing.** Each frozen joint pushes the policy further OOD (the
> held joints' obs no longer match what it commanded) — the robot becomes "a stiff stander waving its
> arms." If **arms-only + low gain** still tips, this policy's swing inherently shifts the CoM beyond a
> static stance, and the only real fix is the **retrain (Path B)**: enforce A3 ankle/hip ranges in the
> action space + a foot-contact / balance reward so legs+waist actively balance *during* the swing.

### Stage G3 — Low-gain *full-body* MOTION transition test (legs in the loop — known contact-OOD)

> ⚠️ **This is the full-body-legs path, which hits the contact-OOD wall** (SUPERSEDING VERDICT above):
> at MOTION the policy commands the legs/ankle far out of range and **lifts a foot**, tipping the
> robot forward — the gain knobs do **not** fix it. Run this **only on the rope**, as a *diagnostic*
> to reproduce/measure the wall (not as a route to a standing swing). For an actual ground swing use
> **Stage G3′**. A true legs-in-the-loop full-body swing requires the retrain (Path B / Stage G5′).

The point is to watch the **PD_STAND/default → policy windup** transition at the lowest gain. Enter
MOTION; **stay at level 0 only briefly** (it's the OOD frozen windup that twitches — see the
runbook). Do **not** press `1` yet.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g3_motion_l0; mkdir -p "$RUN"
# STIFF ANKLES so it doesn't tip forward (the #1 knob); moderate legs; GENTLE arms;
# BLENDED entry. If gravX climbs forward, RAISE --ankle-gain-scale (1.0 -> 1.5 -> 2.0).
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --level 0 --gain-scale 0.05 --leg-gain-scale 0.5 --ankle-gain-scale 1.0 --motion-blend-sec 0.5 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: s (PD_STAND, settle) -> m (MOTION, level 0) -> watch the transition ~5–10 s -> p/q.
#   Keep a finger on the e-stop. Do NOT press 1. If it tips FORWARD, raise --ankle-gain-scale
#   (1.0 -> 1.5 -> 2.0) until gravX settles near upright; watch for ankle buzz at high values.
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
**Watch — STOP (p / e-stop) on any:** a sudden **hip/knee jump**, **forward lean** (`gravX` climbing
past ~0.3≈17° and not settling), **foot slip**, **qd spike** (`qdpk` in `[diag]`), the **clamp-rate
WARN**, `halts` incrementing, `sync_miss` rising. **Knees soft / robot sinks forward = leg gain too
low → raise `--leg-gain-scale`.** Bounded + quiet (legs holding, gravX settling) = good.

> The MOTION-engage windup kick (~1.5–2 rad) is now smoothed by `--motion-blend-sec` (default 0.5 s).
> If it's still abrupt, raise it (e.g. `--motion-blend-sec 1.0`). Stiff legs + blended entry is what
> makes the transition safe — NOT a uniformly low gain (which just collapses the knees).

### Stage G4 — Low-gain continuous swing (level 1, short burst)

> 🔑 **CORRECTED 2026-06-30 — the failure is LEVEL 0, not legs-in-the-loop. LEVEL 1 full-body WORKS.**
> The level-0 re-test (no `--legs-passive`, leg 0.5 / ankle 1.0, `--official-stand`) lifted a foot
> *worse* than before (legs measured **2.6 rad**, hip_pitch cmdR **3.08**, **gravX 0.67 / ~42°**,
> right_ankle_pitch clamped **95% / 7.3 rad over**) — but that is the **frozen-windup level-0 hold**
> (clock pinned `ts=0`, `tts=5.000` constant → an OOD fixed point; in training the clock always
> advanced). At **level 1** (clock advancing through the trained motion) the **full-body legs-in-the-
> loop swing works on the ground — complete swing + self-balancing ("自己调整位置")**, operator-
> observed. It recovered cleanly on `s`/PD_STAND, `halts=0`/`sync_miss=0` (policy chaos at level 0,
> not a fault). **So a legs-in-loop ground swing does NOT need a retrain** — enter MOTION **at level 1**
> and never dwell at level 0. The remaining piece is the level-0 "ready/wait" state (between swings):
> don't freeze at the windup — front-end fix is to hold the legs (Tier-1 mode) while waiting and release
> to full-body at level 1, or keep the clock gently looping. Capture a clean `--level 1`-from-start
> full-body trace to document. Tier-1 (Stage G3′) remains a safe fallback demo.

Only after G3 settles (legs holding, gravX bounded). `level 1` is **closer to the training
distribution** than the frozen level 0, so the swing is smoother — keep arm gain low, legs stiff,
burst short.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g4_motion_l1; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --level 0 --gain-scale 0.05 --leg-gain-scale 0.5 --motion-blend-sec 0.5 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: s -> m (level 0, confirm legs hold + gravX bounded) -> 1 (ONE short swing burst, ~3–5 s) -> p -> q.
#   (use the --leg-gain-scale that held in G3)
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
**Watch:** arms, waist, **hips**, feet (slip?), **support-rope load**, **gravX** (forward lean), `qd`,
clamps, `halts`, `sync_miss`. **Do NOT try to hit a real ball.** Short burst, then back to PD_STAND.

### Stage G5 — Gain ramp (one gain per log)

Only after G4 is stable. Raise gain **slowly** with `]`, staying below full gain until repeated short
tests are clean. **Return to PASSIVE/PD_STAND between tests.** **Log each gain level in its own file**
— the analyzer WARNs ("single gain in capture") when one log pools multiple gains, which muddies the
tracking numbers.

```bash
RT=config/a3_runtime_config.pingpong.yaml
for G in 0.10 0.20 0.30; do
  RUN=/tmp/pp_ground_g5_gain_${G}; mkdir -p "$RUN"
  taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
    --start passive --level 0 --gain-scale $G \
    --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
  # keys per run: s -> m (level 0 calm) -> 1 (short burst) -> p -> q.  Do NOT press ] (keep the gain fixed per log).
  sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
done
```

### Stage G6 — Optional clean level-1 capture

If the hardware stays stable, record a **level-1-only, single-gain** capture for a clean tracking
number + the AGI report. Optional, not required for deploy success.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g6_l1_clean; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --level 1 --gain-scale 0.20 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# keys: s -> m -> (stays level 1) -> dwell one clean swing window -> q. Keep gain FIXED.
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```

### Stage G7 — Deployable full-body: `--auto-leg-hold` (ready ↔ swing)

The deployable behavior, built on the level-0/level-1 finding: **`--auto-leg-hold` holds legs+waist at
level 0 (stable ready stand — avoids the frozen-windup foot-lift) and releases them at level 1
(full-body self-balancing swing).** The `0`/`1` key becomes "ready ↔ swing", both stable. The
pose-blend re-arms on the toggle so the stiff official stand gains do **not** snap the legs on the
1→0 re-engage (no jump). Rope **on** for the first `0↔1` toggles.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g7_autohold; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --official-stand --auto-leg-hold --level 0 \
  --gain-scale 0.30 --leg-gain-scale 0.5 --ankle-gain-scale 1.0 --motion-blend-sec 0.8 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# banner: auto_hold = ON.  Start at level 0 (legs+waist HELD = stable ready).
# s (PD_STAND official, rope slack -> confirm it stands)
# m (MOTION, still level 0 -> legs HELD, stable ready stand, status legs_passive=1)
# 1 (RELEASE -> full-body swing, legs_passive flips to 0, gravX bounded, self-balances)
# 0 (re-HOLD -> back to stable ready, NO jump thanks to the blend; legs_passive=1)
#  ... repeat 1/0 a few times, watch the toggle ...   p   q
sed -n '/FIRST-TICK DEBUG/,/^==*$/p' "$RUN/status.log" > "$RUN/first_tick.txt"
```
**Watch at each toggle:** `legs_passive=` flips 1↔0 with the level; the `[fullbody]` `Lleg/Rleg cmdR`
goes ~0 (held) at level 0 and moderate (~0.6 rad) at level 1; `gravX` stays bounded through the
release **and** the re-engage (no jolt). If the 1→0 re-engage still jumps, raise `--motion-blend-sec`
(e.g. 1.0). If the hoist is lowered and the knees still sink, jump straight to Stage G9:
that is a weight-bearing issue (`--leg-stand-gains` / clamp / smoothing), not something mocap or
`--gain-scale` alone will fix. **STOP** on any tip that doesn't settle, hip/knee jump, `halts`/
`sync_miss` rising.

> 🔑 **Swing-clock reset (fix 2026-06-30):** on each `0→1` release the swing clock resets to its
> **windup** so `ts` starts **near 0** and the swing begins from the start (matching the held pose).
> Before this fix, the free-running clock made a release snap into a mid-cycle phase (`ts≈78`) while
> the legs were still at the stand → lurch/fall. **Verify after pressing `1`: `ts` should start low,
> not jump to a mid value.** Use `--leg-gain-scale 0.5 --ankle-gain-scale 1.0` (the gains that
> balanced standalone) — a bare `--gain-scale 0.30` leaves the legs/ankle too soft.
>
> ✅✅ **VERIFIED 2026-07-01:** with the clock fix, pressing `1` gives `ts=0` at release and the
> full-body swing runs **balanced** — gravX bounded −0.01…0.13, gravZ −1.00, legs+waist driven
> (cmdR 0.5–1.3, ankle within range), `halts=0`/`sync_miss=0`, clock cycling 0→50→0. The auto-hold
> **ready (level 0) ↔ full-body swing (level 1)** loop works on the ground (legs-in-the-loop, no
> retrain). Mild windup-phase lean (gravX ~0.13) is the static residual; optional polish = a
> swing-speed ramp on release.

### Stage G8 — Knee-sink under load: a posture *cap* (`--leg-clamp-rad`) — but NOT the weight-bearing fix

> ⚠️ **READ G9 FIRST.** Hardware (2026-07-01) showed the clamp is only a **safety limiter**: when the
> hoist is lowered the knees STILL sink (the clamp adds zero holding torque), then the squat guard
> reverts to level 0. The real fix is **leg stiffness → Stage G9 (`--leg-stand-gains`)**. The clamp is
> still useful *with* G9 (it keeps the now-stiff legs near upright), but on its own it does not
> make the legs weight-bearing. The diag below correctly shows the policy commands a crouch — that's
> real — but capping the crouch is not enough; the legs also need the stiffness to hold the body up.



When the hoist is **lowered** at level 1, the knees **sink** and the robot crouches/leans forward —
even with `--leg-gain-scale 0.7`. The per-joint leg diag (now printed when legs are policy-driven)
root-causes it: **the policy COMMANDS a deep crouch-and-lean, it is NOT a soft-PD collapse.**

```
-- LEGS (policy-driven) --   (level 1, --leg-gain-scale 0.7)
 left_hip_pitch    des=-0.50..-0.60   q=-0.23..-0.30   err -0.27..-0.29  trk 12-22%
 right_hip_pitch   des=-0.58..-0.77   q=-0.24..-0.35   err -0.34..-0.42  trk 19-35%
 left_knee         des=+0.55..+0.60   q=+0.42..+0.56                     trk 31-54%
 right_knee        des=+0.76..+0.67   q=+0.43..+0.58                     trk 36-65%
 left_ankle_pitch  des=-0.91          q=-0.22..-0.30   err -0.69..-0.60  trk 11-20%
 right_ankle_pitch des=-0.41..-0.68   q=-0.21..-0.28   err -0.19..-0.40  trk  6-10%
```

`des` itself is a squat: **hip_pitch −0.6…−0.77** (torso pitched forward), **knee +0.6…+0.67**,
**ankle_pitch −0.7…−0.9** (dorsiflexed). That triad is the trained "plant-and-weight-shift" swing
posture — legitimate in sim where the planted-foot contact dynamics hold it, but on the real robot
it is **not a stable static stand**: the CoM target is well forward of the feet. So the sink is the
robot (partially) *following* a crouch command, not the leg PD failing to hold a good pose.

> ⚠️ **Therefore raising `--leg-gain-scale` to 0.8 is the WRONG lever** — a stiffer leg tracks the
> crouch *more faithfully* → it squats harder and pitches forward → falls. (That is why 0.7 already
> sank and the squat guard tripped — correctly.) **The fix is to cap the commanded crouch**, not to
> chase it harder.

**Fix — `--leg-clamp-rad R`** (front-end only; AGI's MJCF/PD untouched): clamps each **policy-driven**
leg slot to `nominal ± R` rad, keeping the legs near the proven upright stand while leaving room for
small balance moves. No-op when legs are HELD (level 0) or `R=0`. The diag's leg `des` after the clamp
shows the bounded values (e.g. hip_pitch `des` capped near nominal±R instead of −0.77), which is the
verification signal. The `[pp SAFETY]` squat/tilt guard stays armed as the hard fallback.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g8_legclamp; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --official-stand --auto-leg-hold --level 0 \
  --gain-scale 0.30 --leg-gain-scale 0.7 --ankle-gain-scale 1.0 --motion-blend-sec 0.8 \
  --leg-clamp-rad 0.25 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# banner: safety = ... leg_clamp=0.25rad
# s -> m (level 0 held, stands free) -> 1 (release) -> WATCH the leg des in [diag]:
#   hip_pitch des should be ~nominal-0.25 (NOT -0.77); ankle_pitch des ~nominal-0.25 (NOT -0.9).
#   Knees should hold; gravX bounded. Lower the hoist GRADUALLY (rope still on).
# If it still leans -> drop to --leg-clamp-rad 0.15.  If too stiff/robotic, won't shift weight -> 0.35.
```
**HOIST / LIGHT SUPPORT ONLY** until the knee holds under load. The guard auto-reverts to level 0
(held official stand) on knee-sink (knee bend > `--squat-guard-rad`) or tilt (`|gravX/Y|` >
`--tilt-guard`); press `1` to retry once stable.

### Stage G9 — Weight-bearing released legs: `--leg-stand-gains` (the real knee-sink fix)

**Root cause of the knee-sink, quantified.** The leg that HOLDS at level 0 and the leg that SINKS at
level 1 differ in **stiffness**, not posture:

| leg joint | official ground-stand kp (`a3_pd_stand_kps`) | policy kp (raw ONNX) | released kp @ `--leg-gain-scale 0.6` |
|---|---|---|---|
| **knee** | **2000** | ≤250 | ~150 |
| hip_pitch | 1500 | ~150 | ~90 |
| ankle_pitch | 500 | ~60 | ~36 |

At level 0 the held knee runs at **kp 2000** (AGI's proven ground stand) → bears weight, stands free.
At level 1 the released knee drops to the **policy PD ≈ 150** — **~13× softer** — so it sags under real
body load. The clamp (G8) limits q_des but adds no torque, so it can't fix this.

**Fix — `--leg-stand-gains`:** keep AGI's official ground-stand PD on the legs **even when RELEASED**
(level 1). The policy still drives the (clamped) leg q_des, but the gains are the weight-bearing ones,
so the knees hold the body up like the level-0 hold while making small swing-coupled moves. Front-end
only (AGI's MJCF/PD untouched); `--leg/--ankle-gain-scale` are ignored for the legs (official verbatim).

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g9_standgains; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --official-stand --auto-leg-hold --level 0 \
  --leg-stand-gains --leg-clamp-rad 0.15 \
  --gain-scale 0.30 --motion-blend-sec 0.8 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# banner: leg_stand_g = ON   safety = ... leg_clamp=0.15rad
# s -> m (level 0 held, stands free) -> 1 (release) -> LOWER THE HOIST GRADUALLY, rope still on.
```

> 🔑 `--leg-stand-gains` **requires** `--official-stand` and pairs with a **TIGHT** `--leg-clamp-rad`
> (0.15–0.20): stiff gains (kp~2000) drive the leg firmly to whatever the clamp allows, so a loose
> clamp would force a stiff *crouch*. A tight clamp keeps the firm legs near the upright stand.

**Watch:** with the hoist lowered, the knees should **hold** (no sink) — `[diag]` `left/right_knee_joint`
`q` stays near the clamped `des`, `baseZ` no longer dips at the strike, the squat guard does NOT trip.
**Expected residual:** a forward **lean** during the forehand reach may remain — stiff legs hold a
near-fixed pose, they don't *actively* rebalance the swing's CoM shift (same residual as the Tier-1
arms-only deploy). Reduce it with a gentler swing (`--gain-scale 0.20→0.15`). **A fully balanced
dynamic legs-in-the-loop swing — legs actively shifting to counter the CoM — needs the RETRAIN (Path B);
the front end can make the legs bear weight, not learn to balance.** **HOIST / light support only.**

#### G9.1 — Stiff gains TWITCH; smooth the leg reference (`--leg-smooth-alpha`)

Hardware (2026-07-01): `--leg-stand-gains` alone makes the legs **twitch/convulse at level 1**. The
official gains are *static-hold* gains (silent at level 0 holding a constant pose); at level 1 the
policy drives a **time-varying** leg q_des and — because the clamp/hold keeps the leg near nominal —
the policy sees its leg commands aren't followed, escalates/saturates the leg action, and q_des
chatters at the clamp boundary every tick. ×kp 2000 = violent chatter.

> **The soft/stiff bracket:** soft policy gains → knees **sink** (no weight-bearing); stiff official
> gains → legs **twitch** (can't track the moving/OOD policy reference). No single gain wins —
> the trained policy's leg commands assume sim planted-foot contact dynamics that don't exist on the
> real standing robot. The proper fix is the **retrain (Path B)**; the front-end attempt below is
> a long shot.

**`--leg-smooth-alpha A`** EMA-low-passes the released leg q_des (`out = A*in + (1-A)*prev`, seeded
from nominal) so the stiff gains track a SMOOTH reference instead of the jitter. `A=1.0` off;
`A≈0.2` moderate (τ≈4 ticks @50 Hz); smaller = heavier (as `A→0` the legs approach a stiff *held*
stand — safe, bears weight, but barely moves). The experiment is whether some `A` lets the legs move
enough to be "in the loop" without twitching.

```bash
RT=config/a3_runtime_config.pingpong.yaml
RUN=/tmp/pp_ground_g9_smooth; mkdir -p "$RUN"
taskset -c 4-7 ./a3_deploy_onnx_ref_pingpong --runtime-cfg $RT \
  --start passive --official-stand --auto-leg-hold --level 0 \
  --leg-stand-gains --leg-clamp-rad 0.15 --leg-smooth-alpha 0.2 \
  --gain-scale 0.30 --motion-blend-sec 0.8 \
  --trace-csv "$RUN/trace.csv" --obs-csv "$RUN/obs.csv" 2>&1 | tee "$RUN/status.log"
# banner: leg_stand_g = ON,  safety = ... leg_smooth=0.20
# s -> m -> 1 -> watch the LEGS: twitch gone? then lower the hoist GRADUALLY.
# Still twitches -> drop --leg-smooth-alpha to 0.10 (heavier).  Sluggish but stable -> that's the floor.
```

If no `A` gives a twitch-free swing where the legs still contribute, the front-end is exhausted →
ship **Tier-1** (held legs + arms swing) and **retrain (Path B)** for legs-in-the-loop.

### Pull logs + analyze (dev box)

```bash
# --mkpath creates nested dest dirs; use YOUR IPs (jump agi@172.19.81.9  MDU agi@10.42.10.12):
for S in g1_shadow g2_stand g3_motion_l0 g4_motion_l1 g6_l1_clean; do
  rsync -azP --mkpath -e "ssh -J agi@<HDU_WIFI_IP>" \
    agi@<MDU_IP>:/tmp/pp_ground_${S}/ logs/agi_capture/mdu_ground/${S}/
done
# gain-ramp logs (Stage G5):
for G in 0.10 0.20 0.30; do
  rsync -azP --mkpath -e "ssh -J agi@<HDU_WIFI_IP>" \
    agi@<MDU_IP>:/tmp/pp_ground_g5_gain_${G}/ logs/agi_capture/mdu_ground/g5_gain_${G}/
done

# analyze each (obs.csv + trace.csv; trace carries the full-body + ground-contact audit):
for S in g3_motion_l0 g4_motion_l1 g6_l1_clean; do
  python3 scripts/analyze_obs_log.py logs/agi_capture/mdu_ground/${S}/trace.csv logs/agi_capture/mdu_ground/${S}/obs.csv
done
```
The trace report's **full-body block** now also prints **`upright (gravz~-1)`**, **`hip tracking`**,
and **capture hygiene** (`single level in capture` / `single gain in capture` — these WARN if a log
pools level 0+1 or multiple gains, so you know to re-capture clean).

### What would count as "Ground-Contact Bring-up passed"

There are **two tiers**, because the decisive ground data (SUPERSEDING VERDICT) split the goal:

**Tier 1 — SHIPPABLE (legs-held swing, Stage G3′).** Passed when, on the ground with the rope present
but not bearing load: (1) `PD_STAND --official-stand` holds quiet and upright (`gravz≈-1`, rope slack,
no sag); (2) in `--legs-passive --official-stand` MOTION the banner reads `leg_hold = official`, the
`[fullbody]` line shows legs held (`Lleg/Rleg cmdR≈0`), and `gravX` stays near 0 through the level-0
windup; (3) a short level-1 arm/waist swing burst stays bounded and upright — `sync_miss=0`,
`halts=0`, no foot slip. This is the demonstrable "robot stands and swings" deploy.

**Tier 2 — FULL-BODY (legs in the loop) is BLOCKED on a retrain (Path B).** The full-body MOTION
stages (G3/G4/G5/G6) reproduce the **contact-OOD wall**: the policy commands the legs/ankle out of
range and lifts a foot — no front-end gain fixes it (proven). Tier 2 cannot pass with `model_15200`.
The prerequisite is a **retrain** that enforces A3 ankle/hip ranges in the action space and adds a
foot-contact / no-lift (and ideally a balance) reward, so windup no longer lifts a foot on the
ground. Until then, treat G3/G4 as **rope-only diagnostics**, ship Tier 1, and do **not** connect a
live ball planner.

---

## Stage 7 — Checklist for every new checkpoint

After **every** new training checkpoint:

1. **Export** `.pt` → ONNX (Stage 1) — **bake both clips**, verify ~1.24 MB + 180→31.
2. **Copy** the ONNX to `assets/a3_runtime/models/model_15200.onnx` and into `dist/.../models/` (Stage 3.1).
3. **Config** — only edit `onnx.model_path` if you **renamed** the file (Stage 3.2).
4. **Metadata inspect** — dump and confirm 180→31 + joint_names/scale/gains (Stage 3.4).
5. **C++/Python parity** — `run_parity.sh` + the 3 gates (Stage 3.5).
6. **e2e smoke** — included in `run_parity.sh` (q_des finite/bounded, kp/kd ranges).
7. **MuJoCo** — training-side eval (2.2) **and** AGI MuJoCo shadow→motion (Stage 4).
8. **Only then** consider hardware (Stage 6).

**What must repeat vs. what doesn't:**

- **Weights-only change** (obs layout, action layout, `joint_names`, `action_scale`,
  `default_joint_pos`, `kp/kd`, `body_names` all **unchanged**): you still **re-run
  parity + e2e + MuJoCo + AGI MuJoCo** (the weights are different, so the *behavior* is),
  but you do **not** rewrite the C++ runner front-end. Re-build only if you want the new
  weights baked into `dist/`; otherwise just swapping the ONNX file + re-copy is enough.
- **Contract change** (any of obs layout / action layout / `joint_names` / `action_scale`
  / `default_joint_pos` / `kp/kd` / `body_names` changed): the runner contract must be
  **re-checked carefully** — the front-end (`pp_obs_builder`, `pp_joint_map`,
  `pp_onnx_policy`) may need code changes, and the parity golden / jointmap test must be
  re-validated before any sim or hardware run.

---

## Stage 8 — Troubleshooting

| symptom | fix |
|---|---|
| `No such file or directory` building a harness | the parity/e2e sources live in `scripts/pingpong_parity/`, not the repo root. Prefer `bash scripts/pingpong_parity/run_parity.sh`. |
| ONNX path wrong / runner can't find model | build with `--runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml`, then check `ls dist/a3_deploy_x86_64/models/model_15200.onnx` and run through `./run_a3_pingpong.sh` so the packaged cfg/aimrt pairing stays consistent. |
| `model_15200.onnx` missing after build | the durable source is still `assets/a3_runtime/models/model_15200.onnx`; if a package is missing it, rebuild with the ping-pong `--runtime-cfg` instead of doing a plain arch-only build. |
| `YAML::BadFile: bad file: config/a3_runtime_config.pingpong.yaml` | you are likely using an old `dist/` built before the packager started preserving the ping-pong cfg alias, or you used `--build-only`. Re-run Stage 3.3 / Stage 5 with `--runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml`. |
| `libonnxruntime.so not found` | the harnesses need `-Wl,-rpath,$ORT/lib`; the runner needs `export LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}"`. ORT = `thirdparty/onnxruntime/onnxruntime-linux-x64-1.19.2`. |
| `ModuleNotFoundError: onnxruntime` in `run_parity.sh` | system `python3` lacks it. Set `PYBIN=` to a python with `onnxruntime`+`numpy` (e.g. the conda env). |
| ROS commands fail / `aimrt_main: command not found` | you didn't `source /opt/ros/jazzy/setup.bash` (the `hope` box's `.bashrc` is broken — do it in every shell). |
| AimRT plugin / `undefined symbol rclcpp::QOSEventHandlerBase` | Humble-ABI plugin under Jazzy. Use the **iceoryx-only** path (Path A `a3_pingpong` sim + `a3_aimrt_config.pingpong_iceoryx.yaml`); don't use `mujoco_sim_standalone` (needs Humble). |
| `RouDi not found` / `Timeout registering at RouDi` | iceoryx broker not up, or stale shm. `pkill -9 -x iox-roudi; pkill -9 -x aimrt_main; rm -f /dev/shm/iox*`, then start `iox-roudi` **before** the runner. |
| `iox-roudi` won't start / stale `/dev/shm/iox*` | `rm -f /dev/shm/iox*` and re-launch; always clean shm between runs. |
| runner `rate=0` / immediate safe-halt | transport mismatch: the sim's `/body_drive/*` topics are on **ros2** but the runner cfg is **iceoryx**. Use an iceoryx sim cfg (Path A) or an ros2 runner aimrt cfg. |
| `sync_aligned` false / `not all 6 topics ready` | state not arriving: in MuJoCo the sim isn't publishing (check Terminal 1); on hardware EtherCAT isn't really sending (back to MDU Terminal A). |
| MDU: `ros2 topic hz` shows "not published" | **normal** on the MDU — topics are on iceoryx shared memory, invisible to the DDS-only `ros2` CLI. Judge readiness from the runner's dry-run "6 topics ready", not `ros2 topic hz`. |
| `q_des` too large / out of range | check Stage 3.4 metadata + the e2e smoke (q_des should be finite, max ≈ 0.88 rad). A bad `action_scale`/`default_joint_pos` in metadata means a bad export. |
| kp/kd mismatch vs training | the runner publishes `joint_stiffness`/`joint_damping` from ONNX metadata; if they don't match training, re-export (the gains come from the env at export time). Real backend is **implicit PD** — gains must match. |
| neck buzzes | neck must be **passive** (`q=0, kp=40, kd=2`); if it isn't, the jointmap/passive override is wrong — re-run `pp_jointmap_test` / `pp_policy_test`. |
| wrists buzz | low-inertia instability; lower `--gain-scale` (e.g. 0.4) and/or `--swing-speed`. |
| MuJoCo viewer fails on a headless box | prefix the sim with `MUJOCO_GL=egl`. |
| wrong distrobox | export/eval = **`grasping`**; build/parity/AGI-MuJoCo/cross-compile = **`hope`**; hardware = **MDU**. `play.py` is Isaac → never runs in `hope`. |
| `file ./a3_deploy_onnx_ref_pingpong` shows `x86-64` on the MDU | you shipped the wrong package; rsync `dist/a3_deploy_rockchip/` (aarch64), not the x86 one. |
| C++/Python parity FAILS (`max|Δ|` large) | the ONNX, the obs golden, or the decode drifted. Re-dump metadata; regenerate the Python reference; check you didn't change obs layout without updating `pp_obs_builder` + the golden. |

---

## Quick reference — the very first commands for a brand-new checkpoint

```bash
# 1) EXPORT  (distrobox grasping)
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking && source setup_train_env.sh
RUN=logs/rsl_rl/agibot_a3_hope/<YOUR_RUN_DIR>
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo headless=true num_envs=2 \
  checkpoint=$RUN/model_<N>.pt          # task cfg bakes BOTH clips automatically
ls -la $RUN/exported/policy.onnx                                    # expect ~1.24 MB
python scripts/inspect_a3_deploy_contract.py --onnx $RUN/exported/policy.onnx  # 180->31

# 2) INSTALL + BUILD + VERIFY  (distrobox hope)
distrobox enter hope
cd ~/workspace/HOPE/agi/a3_deploy_example && source /opt/ros/jazzy/setup.bash
cp $RUN/exported/policy.onnx assets/a3_runtime/models/model_15200.onnx   # (use the absolute path)
bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --jobs $(nproc)
cp assets/a3_runtime/models/model_15200.onnx dist/a3_deploy_x86_64/models/
bash scripts/pingpong_parity/run_parity.sh                               # PASS @ ~1e-6

# 3) then: AGI MuJoCo (Stage 4) -> Rockchip cross-build (Stage 5) -> hardware (Stage 6)
```

---

### See also
- [PINGPONG_DEPLOY_ALIGNMENT.md](PINGPONG_DEPLOY_ALIGNMENT.md) — the runner↔backend contract, tables, safety checklist, and §6 x86 verification commands.
- [MUJOCO_VALIDATION_RUNBOOK.md](MUJOCO_VALIDATION_RUNBOOK.md) — the detailed AGI-MuJoCo hands-on (Path A / Path B, acceptance table).
- [PINGPONG_RUN.md](PINGPONG_RUN.md) — short runner run/validation guide (modes, flags).
- [HARDWARE_BRINGUP_CHECKLIST.md](HARDWARE_BRINGUP_CHECKLIST.md) — AGI native-runner hardware harness validation (network/EtherCAT/gate).
- [../../reimplement.md](../../reimplement.md) — long-form narrative (Step 14 env, Step 15 export, Step 16 sim-to-sim).
