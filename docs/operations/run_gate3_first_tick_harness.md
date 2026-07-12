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
`3dce92d777959b18d7fb0c0d38f3193e366f2aa030830d7fa48f4df3422010dc`). It binds the tracked
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
`612f68bfdd7375838f38d4f89a25fcee5db1e2ce19eac7e55b60bdee47b4d680`) has only one
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
- each source/training/evaluation checkout path must equal `git rev-parse --show-toplevel`, not a
  nested directory; its absolute Git dir and common dir are also recorded.

Accordingly, “starts no process” is not a correct description. The exact statement is: it starts
only read-only Git helpers; it starts no sim/Kit/transport/planner/runner and sends no signal.

## What the static contract can prove

There is intentionally no checked-in machine-local instance. A schema-2 contract can bind:

1. the source commit and exact plan-gate source SHA;
2. absolute core paths plus SHA-256 for vendor sim/config/MJCF, planner binary/config, production
   runner/runtime-config/formal ONNX, and a proposed Kit executable;
3. the clean source Git identity plus two clean, exact-commit Git worktree roots for training and
   evaluation, including every worktree's Git dir/common dir;
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

Every accepted contract and plan must preserve these five blockers with `evidence: null` until a
separately reviewed runtime run supplies the evidence. The diagnostic below implements useful
plumbing for the first item, but lacks same-sample/planner/runtime exactness by construction, so
runtime evidence remains null:

1. **runner output:** production C++ must produce and runtime-verify native full
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

## Production first-tick joined-source diagnostic (implemented, not run)

The production ping-pong runner now accepts `--first-tick-json ABS_PATH`, but only with
`--no-publish`/`--dry-run`, a strict exact/publishable 179-D model and `--planner`. It rejects an
empty output value, legacy-model escape, model-only preflight, reference playback, warmup and
PD_STAND/MOTION start before backend initialization. PASSIVE waits; SHADOW observes the first
planner-engaged actor **candidate**. Constructor prewarm, yaw capture, waiting/invalid/recovery and
idle rows do not consume the one-shot. This does not certify that the current planner tuple is a
same-tick atomic snapshot: that planner fix is still NO-MERGE.

The output schema is structurally diagnostic. Both the outer document and payload fix
`evaluation_contract_exact=false`; the payload fixes
`planner_snapshot_exact=false`, `native_sample_alignment_exact=false`,
`source_binary_binding_exact=false`, `source_semantics_closure_exact=false`, and
`runtime_artifact_closure_exact=false`, with non-empty
reasons. Gate3/Gate3B and checkpoint-promotion consumers must reject it. Closing those items later
requires a new schema/version, never flipping these v1 fields to true.

The runner reads the canonical non-symlink ONNX once under stable
device/inode/size/mtime checks, hashes those bytes, and constructs ONNX Runtime from that exact
buffer. This closes model load/hash TOCTOU for the diagnostic, but does not bind the built runner
binary or transitive loader closure. No `source_commit` claim is emitted.

`RobotState` has no root linear velocity, so the runner never zero-fills, integrates or estimates
it. A subscription-only sim sidecar reads the existing vendor pelvis pose/twist and right-racket
pose topics. It has no publisher/reset/command. Python uses kernel `flock(LOCK_EX)` plus one
whole-record `pwrite`; C++ uses `flock(LOCK_SH)` plus one whole-record `pread`. The owned mode-0600
file and JSON paths are canonical, non-symlink and exclusive; JSON uses fsynced temporary bytes plus
hard-link no-replace.

Finite/unit/fresh source records, strictly advancing topic stamps, positive even generations,
<=20 ms native-header skew and <=30 ms RobotState/sidecar receipt join are required. These are
proximity checks only. The tracked vendor publisher stamps each ROS message asynchronously at
publish time and carries no shared MuJoCo sample sequence, so pelvis pose/twist/racket and
RobotState may be adjacent but different simulator samples. The JSON therefore calls qpos/qvel a
`closest_receipt_window_not_common_sim_tick` join and records every source/receipt stamp. A formal
schema needs a publisher-side shared sample sequence and exact consumer match.

The diagnostic records joined qpos38/qvel37/base7/racket7, the observed 179-vector, raw 31-D action,
target candidate, joint/frame/layout metadata and per-payload/whole-payload SHAs. Observation base
(external position + yaw-aligned IMU) is kept separate from vendor-world base; source checks require
<=3 cm position and <=0.02 projected-gravity disagreement. The native `right_racket` position must
agree with the formal wrist control-point FK within 5 mm.

`configs/gate3_first_tick_source_contract_20260712.json` hashes a reviewed source **subset** useful
for auditing this instrumentation. It explicitly fixes `source_semantics_closure_exact=false` and
does not claim parser-backed/transitive closure. Vendor config→MJCF parser resolution, publisher
binary/config/transitive membership, full planner/wire/frame/backend adapters, exact process
ownership, runtime ledger/lock and an actual backend tick all remain OPEN/null. No simulator,
backend, transport, Pod/GPU or hardware was started.

## Produce a static plan

Compute the external contract SHA independently, then run:

```bash
python3 scripts/run_gate3_first_tick_harness.py \
  --contract /absolute/external-control/gate3_first_tick_static_contract.json \
  --expected-contract-sha256 "$CONTRACT_SHA" \
  --plan-output /absolute/external-control/gate3_first_tick_static_plan.json
```

Without `--plan-output`, the plan is printed. With it, the parent directory must already exist,
contain no symlink component, and be outside the source, training and evaluation worktree roots
**and** all recorded Git dirs/common dirs. Location rejection happens before creating a temporary
file. Immediately before an allowed external write, all three Git identities are compared again
and all three worktrees must still be clean at their recorded commits. The writer then:

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

Accepted local result: `32 passed`. Tests cover hard rejection of old runtime/arming CLI before any
mock process/signal, no runtime-supervisor symbols, `GIT_OPTIONAL_LOCKS=0`, exact Git top-level,
symlink ancestors, SHA/path/environment fences, argv equals/absolute/relative bypasses, null
dependency/semantic blockers, atomic link no-clobber including a competing-create race, and the 14
legacy risks. Real Git repos prove source/train/eval/Git-dir outputs are rejected without a file or
dirty status, an external output succeeds, and a late dirty checkout blocks the external write.
The contract same-byte SHA/parser path and a mid-read mutation are also covered. No simulator, Kit,
transport, planner, runner, Pod/GPU, or robot is invoked.

The native-output source gate adds dependency-light ABI/writer tests plus a C++ unit file that is
picked up by the production `run_tests` glob:

```bash
python3 -m py_compile \
  agi/a3_deploy_example/scripts/gate3_first_tick_state_bridge.py
pytest -q \
  tests/test_gate3_first_tick_state_bridge.py \
  tests/test_pp_first_tick_json_cpp.py
```

Accepted host source result: `6 passed`. It covers kernel-locked whole-record transfer, nonzero
native root linear velocity preservation, idle-not-consuming/planner-candidate capture, canonical
exclusive paths, mode 0600, atomic no-replace payload output, per-vector SHA fields, and
fail-closed nonfinite/fabricated-root-velocity/cross-source-skew/policy-base, odd generation,
source-stamp regression and empty-output-flag cases. It also parses the output and proves all five
exactness flags are fixed false and a formal-style consumer rejects it. The full
ROS/Jazzy/AimRT Release build and `PpFirstTickJson.*` GTests have not run on this branch; that is an
explicit open verification item, not an optional pass.

Remaining work after source merge is exactly the five blockers above, followed by one separately
reviewed no-publish first tick, per-clip normal/recovery gates, and vendor no-reset Gate3/Gate3B.
