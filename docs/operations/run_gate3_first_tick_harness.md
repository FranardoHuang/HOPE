# Vendor Gate3 first-tick static plan gate

Status: **plan-only source gate**, corrected after red-team review on 2026-07-12. It cannot launch
the vendor simulator, Kit, ROS/AimRT transport, planner, production runner, Pod/GPU work, or a
robot, and it cannot send a signal. Its only child commands are read-only Git queries with
`GIT_OPTIONAL_LOCKS=0`. A plan is not a runtime preflight, first tick, or Gate3 result.

The final behavioral arbiter remains the Agibot-provided vendor MuJoCo Gate3/Gate3B chain. Isaac
is for training and diagnosis only. A future safe first tick is merely a runtime prerequisite;
checkpoint promotion still requires vendor behavior evidence.

## Why the legacy launcher remains quarantined

The content-bound static audit is
`configs/gate3_legacy_process_audit_20260712.json` (SHA-256
`46f58c951ae0efc5e81198d494b7e223a98e03c6ccbc2d8b79992ba308e272b5`). It binds the tracked
`pp_gate3_rally.sh` and `pp_rally_conductor.py` bytes and records 14 concrete risks:

| risk | old source | consequence |
| --- | --- | --- |
| fuzzy startup/shutdown kill | shell lines 23--27 and 89--94 call eleven `pkill -9` commands | can kill another Kit/sim/planner/runner/session |
| fuzzy dropout signal | conductor lines 229--231 and 311--318 do `pgrep -f` then SIGSTOP/SIGCONT every match | can suspend a planner it did not launch |
| no owned process identity | `set +e` plus unrecorded background `setsid` jobs | no PID/PGID/starttime/cmdline proof |
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

Do not invoke the old cleanup block to make a future check pass. The tracked scripts are historical
result provenance only.

## Red-team correction: there is no runtime mode

The first implementation in feature commit `1fc69d1` included an armed future-runtime supervisor.
Red-team review correctly rejected that shape: core path hashes are not a complete dependency
closure, `/proc` snapshots alone are not a safe startup handshake, and a plan-only gate must not
carry dormant signal/launch code. That commit must not be merged by itself.

The corrected `scripts/run_gate3_first_tick_harness.py` (source SHA-256
`a32d42131dff058fd05fb88696e9005dbf452c2591c9b28611159e646beb9d4c`) has only one
operation: validate a schema-2 static contract and print or atomically create a plan ledger. There
is no `--mode`, arming phrase, `Popen`, signal,
runtime lock, process scan, `run_harness`, or first-tick consumer. Passing `--mode run` or the old
arming option is an argparse error before the contract or Git check is touched.

The validator does start read-only Git helper processes to verify source/training/evaluation
commits and cleanliness. Their environment is rebuilt rather than inherited:

- `GIT_OPTIONAL_LOCKS=0` prevents optional index writes/locks;
- `GIT_DIR`, `GIT_WORK_TREE`, and other ambient Git variables are absent;
- `--no-optional-locks` is also passed on the command line; fsmonitor, untracked-cache and hooks are
  disabled, and global/system config, pagers and terminal prompting are disabled;
- each declared checkout path must equal `git rev-parse --show-toplevel`, not a nested directory.

Accordingly, “starts no process” is not a correct description. The exact statement is: it starts
only read-only Git helpers; it starts no sim/Kit/transport/planner/runner and sends no signal.

## What the static contract can prove

There is intentionally no checked-in machine-local instance. A schema-2 contract can bind:

1. the source commit and exact plan-gate source SHA;
2. absolute core paths plus SHA-256 for vendor sim/config/MJCF, planner binary/config, production
   runner/runtime-config/formal ONNX, and a proposed Kit executable;
3. two clean, exact-commit Git worktree roots for training and evaluation;
4. an explicit local-only environment proposal with hardware disabled;
5. fixed proposed argv arrays: sim/config, planner/config, and passive production runner with
   `--planner --no-publish --start passive --first-tick-json {HARNESS_FIRST_TICK_JSON}`;
6. the immutable future full-state schema, Gate3 decision policy, ready-state diagnostic and
   engine-gap ladder.

Every input path must use its canonical resolved spelling. The validator walks every path
component with `lstat`; a symlink leaf **or ancestor** is rejected. A hash is accepted only when
device/inode/size/mtime/mode are unchanged across the read.

The external contract is read once, SHA-checked and parsed from those same bytes; an identity
change during the read fails. It is not hashed and then reopened for parsing.

Proposed commands are deliberately exact rather than shell-like. `--flag=/absolute/path`, any
unbound absolute token, any relative/unclassified payload, extra flag, changed order, or working
directory inside a train/eval checkout fails. This prevents a core SHA table from being bypassed
by a second path spelling. These are only proposed argv bytes; no command is executed.

## What it explicitly cannot prove

Every accepted contract and plan must preserve these five blockers with `evidence: null`:

1. **runner output:** production C++ still lacks native full
   qpos38/qvel37/base7/racket7/target/obs179 `--first-tick-json`;
2. **process supervision:** no reviewed pidfd plus cgroup/supervisor startup handshake exists;
3. **complete artifact closure:** PATH/LD_LIBRARY_PATH/PYTHONPATH/AMENT directory manifests,
   AimRT shared objects, transitive `.so` files and plugins are all null;
4. **vendor-config semantics:** separate config and MJCF hashes do not prove which MJCF the parser
   resolves. Substring containment is explicitly not accepted as semantic binding;
5. **runtime evidence transaction:** the atomic no-replace runtime ledger and exact-owned runtime
   lock protocol have not been designed or reviewed.

Therefore this source must not say “exact runtime,” “runtime eligible,” or “safe to launch.” Filling
or deleting any blocker makes the schema-2 contract invalid. Closing them requires a separate
reviewed runtime implementation; do not reactivate the deleted code in this plan gate.

## Produce a static plan

Compute the external contract SHA independently, then run:

```bash
python3 scripts/run_gate3_first_tick_harness.py \
  --contract /absolute/external-control/gate3_first_tick_static_contract.json \
  --expected-contract-sha256 "$CONTRACT_SHA" \
  --plan-output /absolute/external-control/gate3_first_tick_static_plan.json
```

Without `--plan-output`, the plan is printed. With it, the parent directory must already exist and
contain no symlink component. The writer:

1. creates a same-directory mode-0600 temporary file with exclusive create;
2. writes and `fsync`s the complete bytes;
3. uses `link(temp, target)` as the atomic no-replace operation;
4. fails if another writer created the target first, preserving that writer's bytes;
5. `fsync`s the parent directory and removes the temporary link.

It never uses `os.replace`, so an existence check cannot turn into an overwrite race.

The schema-2 output is a static-plan ledger. Its runtime block is fixed to `status=not_run`, an
empty component/signal list, no runtime lock, no behavior result and no ownership token. It records
the core artifacts, checkouts, proposed environment/argv, all null blockers and the diagnostic
policy. The content SHA covers the complete ledger content.

## Future first-tick and cross-engine diagnosis

The future evidence bar remains full finite vendor `qpos[38]`, `qvel[37]`, base pose `[7]`, racket
pose `[7]`, target tuple and formal observation `[179]`, with explicit layouts/frames and separate
canonical SHAs. Existing debug summaries, `--trace-csv`, and `--obs-csv` do not jointly satisfy it.

The initial-state facts remain hypotheses, not scores:

- fresh training reset: pelvis `(0,0,1.0684)` plus training default q;
- vendor `stand`: pelvis `(-0.0416378,0.000359,1.06839)`, approximate rpy
  `(-0.030°,0.249°,0.042°)`;
- mapped joint L2 difference `0.171845 rad`, dominated by head-yaw `-0.169416 rad`; without head,
  L2 remains `0.028789 rad`.

The same-immutable-K100 vendor/root-only/joints-only/full-match design stays preregistered,
not-run and `evaluation_contract_exact=false`. The formal vendor stand is unchanged.

The engine-gap ladder also stays not-run with no inference authority:

1. identical kinematic joint/racket replay;
2. identical open-loop actions and initial state;
3. closed loop with identical externally supplied observation rows;
4. each engine's native closed loop.

Isaac remains training/diagnostic-only. Only vendor Gate3/Gate3B behavior can promote.

## Verification in the correction

```bash
python3 -m py_compile scripts/run_gate3_first_tick_harness.py
pytest -q tests/test_run_gate3_first_tick_harness.py
python3 -m json.tool configs/gate3_legacy_process_audit_20260712.json >/dev/null
```

Accepted local result: `30 passed`. Tests cover hard rejection of old runtime/arming CLI before any
mock process/signal, no runtime-supervisor symbols, `GIT_OPTIONAL_LOCKS=0`, exact Git top-level,
symlink ancestors, SHA/path/environment fences, argv equals/absolute/relative bypasses, null
dependency/semantic blockers, atomic link no-clobber including a competing-create race, and the 14
legacy risks. The contract same-byte SHA/parser path and a mid-read mutation are also covered. No
simulator, Kit, transport, planner, runner, Pod/GPU, or robot is invoked.

Remaining work after source merge is exactly the five blockers above, followed by one separately
reviewed no-publish first tick, per-clip normal/recovery gates, and vendor no-reset Gate3/Gate3B.
