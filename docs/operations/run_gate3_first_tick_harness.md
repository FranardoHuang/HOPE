# Vendor Gate3 exact-PGID first-tick harness

Status: source-only and unit-tested on 2026-07-12. No vendor simulator, Kit, ROS/AimRT
transport, planner, production runner, Pod/GPU, or robot was started while implementing this
harness. The tracked legacy rally script is now historical evidence, not an approved launcher
for a new formal result.

The final behavioral arbiter is the Agibot-provided vendor MuJoCo Gate3/Gate3B chain. Isaac is
for training and diagnosis only. A safe first tick is a runtime prerequisite, not a Gate3
behavior pass; checkpoint promotion still requires the later vendor behavior paper.

## Why the legacy launcher is quarantined

The content-bound audit is
`configs/gate3_legacy_process_audit_20260712.json` (SHA-256
`9d6e50463ca0b1b7ae29d5d4b9ae32ed748de76c8e84cb89a72db3c2fa8c4eb3`). It binds the
tracked `pp_gate3_rally.sh` and `pp_rally_conductor.py` bytes and records 14 concrete risks:

| risk | old source | consequence |
| --- | --- | --- |
| fuzzy startup/shutdown kill | shell lines 23--27 and 89--94 call eleven `pkill -9` commands | can kill another Kit/sim/planner/runner/session |
| fuzzy dropout signal | conductor lines 229--231 and 311--318 do `pgrep -f` then SIGSTOP/SIGCONT every match | can suspend a planner it did not launch |
| no owned process identity | `set +e` plus unrecorded background `setsid` jobs | no PID/PGID/starttime/cmdline/token proof |
| no trap | no EXIT/INT/TERM cleanup around the launch sequence | interrupt/early failure leaks processes |
| hard-coded workspace | `/home/dongc1/...` GEAR/DIST/WS/SIM_INSTALL | reviewed and executed bytes can differ |
| no artifact binding | sim/config/MJCF/planner YAML/runner/model are not SHA-bound | result has no reproducible engine/policy lineage |
| inherited ROS graph | ROS domain/RMW/overlays are not fixed in a ledger | can join foreign transport/runtime state |
| destructive shared outputs | fixed `/tmp` files and wildcard `/dev/shm/iox1_0_*` removal | clobbers evidence/shared state |
| weak readiness | loose log/topic text with no child identity/rc binding | stale or foreign text can pass readiness |
| loader ordering | no formal `--model-preflight-only` before runtime | backend can start before exact-179 metadata is accepted |
| publish-capable runner | conductor omits `--no-publish` and accepts free-form `PP_EXTRA_ARGS` | not a first-tick safety harness |
| false runner-ready | the 90-second boot loop has no post-loop assertion | prints “runner up” after timeout failure |
| partial cleanup | conductor kills only the direct runner PID on timeout | descendants leak, inviting later broad kill |
| no conflict lock | launch begins by killing instead of rejecting concurrency | destroys rather than isolates another run |

Do not “fix” a failed new preflight by invoking the old cleanup block. Preserve the conflict and
resolve ownership outside this harness.

## New source boundary

`scripts/run_gate3_first_tick_harness.py` (current source SHA-256
`828041d6212f87d947e4574042b5c2d7b91da7bff0b7ca5f0f4fe9d6aad33a12`) exposes two
modes:

- `plan` is the default. It validates the entire contract, read-only checkouts and exact
  conflicts, prints or writes a no-clobber plan, and starts no process.
- `run` additionally requires the literal arming phrase
  `I_UNDERSTAND_VENDOR_SIM_FIRST_TICK_NO_PUBLISH`. It is Linux `/proc`-only, simulation-only,
  starts the runner passive with exactly one `--no-publish`, stops after the structured first
  tick, and never authorizes a robot.

The implementation uses direct `subprocess.Popen(..., shell=False, start_new_session=True)`.
It has no `pgrep`, `pkill`, `killall`, fuzzy command search, or single-PID best-effort cleanup.

## Required immutable contract

There is intentionally no checked-in runnable instance because the private vendor/runtime
artifacts are machine-local and no runtime is authorized in this change. Create a strict JSON
contract only after the harness commit is available on the target host. It must bind:

1. `source_commit` and the exact harness SHA;
2. absolute, non-symlink paths plus SHA-256 for:
   - vendor sim executable;
   - vendor sim config;
   - vendor `a3_pingpong.xml` MJCF;
   - planner executable and planner config;
   - production runner executable, runtime config and formal ONNX;
   - the exact Kit executable used only for conflict detection;
3. clean read-only training/evaluation checkouts at explicit commits; every runtime artifact,
   component working directory, ledger and lock must be outside both checkouts;
4. a vendor sim config containing the exact bound MJCF absolute path—not merely its basename;
5. an explicit `ROS_DOMAIN_ID` in `[0,232]`, RMW implementation, `ROS_LOCALHOST_ONLY=1`,
   `A3_SOURCE_ROBOT_ENV=0`, `A3_HARDWARE_ALLOWED=0`, `A3_TRANSPORT=iceoryx`,
   `MUJOCO_GL=egl`, and fully enumerated PATH/library/Python/AMENT environment;
6. exact no-clobber ledger/lock paths and separate exact Kit/sim/planner/runner conflict lock paths;
7. direct argv arrays. The sim argv must contain the bound sim config token, planner argv the
   bound planner config token, and runner argv the bound runtime config/model plus
   `--planner --no-publish --start passive`;
8. the required future runner output
   `--first-tick-json {HARNESS_FIRST_TICK_JSON}`;
9. the fixed decision/diagnostic policy below. Any attempt to grant Isaac promotion authority,
   promote a causal diagnostic, or enable body-command publishing invalidates the contract.

The runtime config/model are CLI-bound even when their paths also appear inside YAML. Any other
absolute command token must itself be one of the SHA-bound artifacts.

## Default plan

Compute the contract SHA independently, then run:

```bash
python3 scripts/run_gate3_first_tick_harness.py \
  --contract /absolute/external-control/gate3_first_tick_contract.json \
  --expected-contract-sha256 "$CONTRACT_SHA" \
  --mode plan \
  --plan-output /absolute/external-control/plan.json
```

Plan mode performs only reads plus the explicitly requested no-clobber plan write. It verifies
artifact hashes, both clean commits, lock absence and exact existing-process conflicts. On Linux,
the process scan walks `/proc` and matches only a bound executable realpath or exact cmdline token;
it never signals. On a host without `/proc`, plan can document the contract but marks runtime
ineligible.

## Runtime sequence and ownership

Future runtime syntax—**do not execute until the blocker below is closed and a reviewed contract
exists**:

```bash
python3 scripts/run_gate3_first_tick_harness.py \
  --contract /absolute/external-control/gate3_first_tick_contract.json \
  --expected-contract-sha256 "$CONTRACT_SHA" \
  --mode run \
  --arm-vendor-sim-no-publish I_UNDERSTAND_VENDOR_SIM_FIRST_TICK_NO_PUBLISH
```

The fixed order is:

1. rerun all hashes/checkouts/conflict checks and acquire a no-clobber owned lock;
2. run the production binary's exact formal loader first:
   `--planner --no-publish --model-preflight-only`;
3. require rc=0 plus `[pp PREFLIGHT] accepted`, `backend_not_initialized=true`, and
   `obs_dim=179`; reject any backend-config/initialized/started text;
4. start vendor sim, then planner, then production runner, each directly in its own new
   session/PGID and each with the same unpredictable ownership token;
5. require new per-run log readiness—not `/tmp` history—and stop at
   `[pp FIRST-TICK DEBUG]` plus the complete structured first-tick artifact;
6. clean up in reverse order.

At launch, the ledger records each direct PID, exact PGID/session, `/proc` starttime, real
cmdline, executable and token. Before TERM or KILL, the harness enumerates only that recorded
PGID and reads every member twice. Every member must inherit the token, be no older than the
leader, and retain the same PID/starttime/cmdline identity; subsequent KILL can only target a
surviving subset of the first validated group. A new/foreign/changed member causes cleanup to
fail closed rather than broad-kill. The owned lock is removed only after inode, PID and token
content still match.

Every invocation gets a new mode-0700 directory. Separate stdout/stderr logs, return code,
process identities, validated TERM/KILL membership, artifact/checkouts/contract/plan SHAs,
error and final ledger content SHA are never overwritten.

## Mandatory full first-tick state

Debug min/mean/max text is insufficient for cross-engine diagnosis. An accepted first tick must
write schema-v1 JSON containing only finite values:

| field | required full length/content |
| --- | --- |
| `joint_names` | 31 unique names; binds the joint tail of qpos/qvel |
| `qpos` | vendor free base + 31 joints = 38 |
| `qvel` | vendor free base velocity + 31 joints = 37 |
| `base_pose` | xyz + quaternion = 7 |
| `racket_pose` | xyz + quaternion = 7 |
| `target` | position, velocity, normal, rho, time-to-strike, swing type, valid |
| `obs` | exact formal actor row = 179 |

The trace additionally declares free-base qpos/qvel layouts, `wxyz` pose quaternions,
world/table target frame and `deploy_parity_face179` observation contract.
The harness recomputes a file SHA, a canonical whole-trace SHA and separate canonical SHAs for
joint names, qpos, qvel, base pose, racket pose, target and obs. Missing, malformed, wrong-length or non-finite
state fails the run even if the old debug marker appeared.

Current blocker: the production runner does **not yet implement**
`--first-tick-json`; existing `--trace-csv`/`--obs-csv` do not jointly expose full vendor qpos,
qvel, base, racket, target and observation. Therefore the new source can be statically verified
and planned, but a real first-tick run must fail closed until a separately reviewed C++ change
adds this output and its native tests. This branch deliberately does not modify C++ or launch
the runtime.

## Ready-state hypothesis is not a conclusion

The ledger freezes the observed initial-state mismatch so it cannot disappear into an aggregate
score:

- fresh exact training reset: pelvis `(0,0,1.0684)` plus training default q;
- vendor named `stand`: pelvis `(-0.0416378,0.000359,1.06839)`, approximate rpy
  `(-0.030°,0.249°,0.042°)`;
- mapped 31-joint L2 difference `0.171845 rad`, dominated by head-yaw
  `-0.169416 rad`; without head, L2 is still `0.028789 rad`.

Stage-1 bank `contact_pos` is env-origin absolute, while the 175/179 target-position observation
is target minus current racket FK. The `-4.16 cm` root-x difference therefore does not
automatically cancel and may explain part of the Isaac/MuJoCo racket-center gap. That is a
hypothesis only.

Preregister a same-immutable-K100, no-threshold-change four-cell diagnostic:

1. vendor stand unchanged;
2. vendor joints with root only matched to Isaac;
3. vendor root with joints only matched to Isaac;
4. root and joints both matched to Isaac.

All four are `evaluation_contract_exact=false` causal diagnostics. Never change the formal vendor
stand, book these as scores, or infer causality before all four cells are content-bound.

## Engine-gap ladder recorded in every plan/ledger

The harness records these four stages as `not_run`, `inference_allowed=false`:

1. identical kinematic joint/racket replay;
2. identical open-loop actions and initial state;
3. closed loop with identical externally supplied observation rows;
4. each engine's native closed loop.

Each later paper must bind joint order, action scaling/clamp, PD, timestep/decimation, ready
state, signed face, contact/termination and the vendor MJCF. The ready-state four-cell design is
one factor inside this ladder, not a shortcut to a conclusion. Isaac remains diagnostic even if
its score is higher; only vendor Gate3/Gate3B behavior can promote.

## Verification in this change

```bash
python3 -m py_compile scripts/run_gate3_first_tick_harness.py
pytest -q tests/test_run_gate3_first_tick_harness.py
python3 -m json.tool configs/gate3_legacy_process_audit_20260712.json >/dev/null
```

Accepted local result: `25 passed`. Tests cover contract/path/SHA/environment fences,
no-publish/passive/structured-trace argv, exact lock/process conflict behavior, synthetic `/proc`
starttime/cmdline/token validation, exact group signaling, full first-tick vector/SHA validation,
no-clobber outputs, arming, and the 14 legacy risks. No simulator or production runner is invoked
by the test suite.

Remaining gates after source merge:

1. add/test the production `--first-tick-json` output;
2. materialize a machine-local contract from the real vendor/runtime bytes;
3. run formal loader, then one no-publish first tick and review its full-state ledger;
4. close per-clip normal envelope and canonical recovery tuple;
5. run the no-reset vendor Gate3 behavior paper and immutable Gate3B scoring;
6. only after those, consider any deployment/robot workflow under its separate safety gate.
