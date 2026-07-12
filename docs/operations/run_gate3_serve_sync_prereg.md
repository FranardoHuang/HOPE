# Gate3 Serve Synchronization Preregistration

Status: `preregistered_runtime_blocked`; design-check only

This operation validates a fail-closed design for Gate3 serve synchronization.
It does not start or signal the vendor MuJoCo simulator, planner, runner,
fake-ball publisher, Pod, or real robot. The validator's only child commands
are two fixed read-only Git identity queries with optional locks disabled.

## The authorization rule

Human-readable log text has no authorization value. In particular,
`-> MOTION (PUBLISHING)` and
`HOPE planner READY: corrected base pose fresh` remain useful diagnostics, but
no stdout/stderr byte, marker order, marker age, baseline offset, or log inode
may arm a ball publisher. Exact diagnostic logging remains preregistered with:

```text
PYTHONUNBUFFERED=1
RCUTILS_LOGGING_USE_STDOUT=1
RCUTILS_LOGGING_BUFFERED_STREAM=0
```

A future reviewed runtime must instead provide three machine-readable states on a
supervisor-owned channel:

- planner: `READY_NO_BALL`;
- runner: `WAITING_BALL_READY`, acknowledging the exact planner clock-sample
  sequence it consumed;
- vendor MuJoCo backend: `BACKEND_READY_NO_BALL`, with its owned process and
  backend-session identity.

The planner/runner records carry the same run nonce, session nonce,
policy-model SHA, source epoch, and exact readiness `base_sequence_anchor`.
They separately expose the live `base_sequence_current`, source age, lease
validity, and `actor_base_ready`; the runner additionally carries immutable and
current base-revocation generation. The vendor record binds the same run/session
plus a backend-session nonce, PID, exact PGID, `/proc` start ticks, executable,
argv, config, MJCF, plant, AimRT/plugin and transitive runtime closure. All three
records bind same-host `CLOCK_MONOTONIC`, monotonic status sequence, and exact
content identity. The supervisor may accept them only while every value still
matches and all samples are inside the reviewed freshness window.

No planner, runner, or vendor-backend restart can inherit a prior ACK. Before
arm, any changed process/start ticks, bytes, argv/config/model/MJCF/plant/
environment/closure, backend session, run nonce or session nonce requires a new
joint session. After ACK, the same change fails terminally before the next
publish.

The ACK does not freeze a 100 Hz localization sequence for a 300 Hz publisher.
The anchor stays immutable in the ledger; the current sequence may repeat while
fresh or advance strictly within the same source epoch. It may never regress
below the anchor. The runner revocation generation must remain equal to its
anchor, so malformed/implausible invalid→valid recovery cannot hide behind an
unchanged epoch. `actor_base_ready=true` additionally means finite and fresh,
inside the hard workspace/continuity bounds, latest base z above `base_low`, and
any recovery hold still owns its engaged epoch/revocation lease.

After ACK acceptance the fake-ball publisher is still `OWNED_DISARMED`. The
future supervisor must durably write and re-read the accepted joint ACK in a
unique, exclusive, no-follow, locked, fsynced, atomic no-replace ledger. The
only arm edge is `OWNED_DISARMED -> ARM_COMMITTED`, which creates one
content-bound single-use arm token. Immediately before the first publish, the
exact owned publisher must atomically consume that token and revalidate its own
PID/PGID/start ticks, pidfd/cgroup, executable/argv/config/environment/runtime
closure plus every identity, frame, epoch, sequence, model, timeout, one-shot
config and trajectory binding. No log marker can create the ledger or token.

Chronology is content-bound rather than inferred. The ACK ledger pre-binds an
arm-commit record identity; the unique arm edge creates that pwrite-once record
with `arm_committed_monotonic_ns`. After consuming the token, the publisher
creates and fsyncs a separate immutable first-publish record. Its conservative
`first_publish_monotonic_ns` is the fixed deadline origin and can never slide.
The publisher then re-reads that record and revalidates every live guard
immediately before sample zero; drift during record creation fails without a
publish.

That first-frame check is not enough by itself. Before every later 300 Hz
sample, the publisher must revalidate planner, runner, vendor and publisher
pidfd/cgroup liveness; legal machine-state transitions; session/backend session;
source epoch, current sequence, source age, lease, actor readiness and runner
revocation generation; immutable trajectory identity; and the strictly
increasing publisher status sequence. The content-bound trajectory has exactly
N indexed samples; supervisor-visible `next_sample_index`/`last_published_index`
must prove every index exactly once, and terminal success requires `next=N`,
`last=N-1`. A failure must enter
`TERMINAL_DISARMED_FAILED` before the next publish, not after the remainder of
the trajectory.

Pre-serve readiness words are not reused after a ball exists. Machine status
must stay fresh within 40 ms and move within 60 ms onto its declared forward
path: planner `BALL_OBSERVED|COMMANDING`, runner `TRACKING|ACTOR_ACTIVE`, vendor
`BACKEND_ONE_SHOT_ACTIVE`. Unknown/backward/stale states or a missed deadline
fail before the next publish.

Runner readiness is also explicit. Prearm `WAITING_BALL_READY` requires
publish-capable MOTION (`no-publish=false`), healthy owned supervisor/backend,
level0/no active clip, completed yaw capture, finite q/dq/IMU, reviewed upright/
still bounds and exact model/179 observation/action contracts. Postarm
`TRACKING` retains level0 while waiting for a command; `ACTOR_ACTIVE` instead
requires exactly one accepted clip with a usable epoch/revocation lease. A
false/unknown readiness bit, multiple clips or safety abort blocks the next
sample.

The success path is strictly one way:

```text
ACK_ACCEPTED -> OWNED_DISARMED -> ARM_COMMITTED -> ONE_SHOT_ACTIVE
             -> TERMINAL_DISARMED_SUCCESS
```

Every nonterminal state has a failure edge to
`TERMINAL_DISARMED_FAILED`. Both terminal states are absorbing: there is no
ACK/token reuse, retry, reset, re-arm, second serve or post-terminal publish.

## Current hard blockers

Runtime launch remains unauthorized. The preregistration intentionally leaves
all 49 runtime bindings null, including:

- reviewed supervisor source/executable, planner/runner/vendor machine-status schemas, joint-ACK
  schema, atomic-ledger schema, pidfd/cgroup ownership handshake, same-host clock
  contract, and unique run/session nonce binding;
- planner and runner executables, exact argv/config/environment/model/runtime
  closures, plus runner AimRT config, plugin closure and transitive shared
  libraries;
- fake-ball executable, argv, config, environment, machine status, explicit
  frame argument, pidfd/cgroup identity, runtime/transitive closure, one-shot
  config, exact trajectory, arm-token/arm-commit/first-publish schemas and
  terminal evidence;
- vendor PID/cgroup identity, backend readiness session, MJCF, plant, runtime
  executable/config/environment, AimRT plugins and transitive shared libraries;
- exact frame transform/identical-frame binding and accepted-ACK runtime
  evidence.

There is also a concrete frame blocker. The content-addressed fake-ball source
defaults `frame_id` to `world`, while the content-addressed vendor simulator
publishes pelvis pose/twist and racket pose as `odom`. The Gate3 planner sim
profile deliberately declares ball=`world`, base=`odom`, and requires a common
formal frame, so it fails closed. No transform is bound, and no exact
`fake_ball_publisher` frame parameter is bound. Do not paper over this by
assuming `world == odom`; either bind a verified source-target transform or
bind an exact same-frame publisher argument and its coordinate evidence.

The tracked fake-ball publisher has a second independent blocker: it loops
forever and automatically calls `_reset()` after each pause. D0 requires
content-bound `one_shot=true`, `max_serves=1`, exactly one trajectory at 300 Hz,
and `auto_reset=false`. The trajectory binding includes frame, initial state,
dynamics, table/end bounds, sample rate, and exact sample-count or terminal
rule. Until reviewed source implements that contract, this tracked publisher is
evidence only and must never be armed.

The tracked `agi/a3_deploy_example/scripts/pp_gate3_rally.sh` is still forbidden:
it uses broad process-name kills and starts the publisher before a machine ACK.
No launch or cleanup line may be copied from it.

## Bound source closure

The design-check verifies actual bytes, not only claims in JSON. It binds the
plan-only first-tick prerequisite and the reviewed formal tuple plus runner
transport/ownership source subset:

```text
hope_ws/src/hope_planner/hope_planner/node.py
hope_ws/src/hope_planner/hope_planner/node_runtime_contract.py
hope_ws/src/hope_planner/hope_planner/flat_command_wire.py
hope_ws/src/hope_planner/config/hope_planner.yaml
hope_ws/src/hope_planner/config/hope_planner.sim.yaml
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_planner_input.hpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_policy.hpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_frame_math.hpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_pingpong/pp_reference_clock.hpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/a3_pingpong_main.cpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/robot_io/a3_aimrt_backend.hpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/robot_io/a3_aimrt_backend.cpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/config/a3_aimrt_config.pingpong_ros2body.yaml
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/robot_io/robot_io_backend.hpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_deploy/a3_policy_driver.hpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/src/a3_deploy/a3_policy_driver.cpp
agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/CMakeLists.txt
```

It separately binds the fake-ball source and vendor simulator config that prove
the current `world`/`odom` mismatch. Every bound path must resolve to a regular
file contained by the exact Git worktree root. Duplicate JSON keys, non-finite
numbers, copied preregistrations and any actual-byte drift fail closed.

This 18-file list is deliberately **not** called the complete planner runtime
closure. Ball estimation/prediction, strike planning, physics/contact helpers,
ONNX parser/observation/action helpers, Python/ROS packages, shared libraries
and plugins remain outside this reviewed subset. Their complete transitive
content address must later populate
`planner_runtime_dependency_closure_sha256`,
`runner_transitive_shared_library_closure_sha256` and the other null runtime
bindings. Until then launch remains blocked.

## Reproduce the design-only result

The immutable preregistration is
`configs/gate3_serve_sync_prereg_20260712.json`, SHA-256
`7a2da058457c4156bb74b69e071cfb2da83a1fea00ba516d94421feac29b0a77`.

```bash
python3 scripts/validate_gate3_serve_sync_prereg.py \
  --repo-root . \
  --prereg configs/gate3_serve_sync_prereg_20260712.json \
  --expected-prereg-sha256 7a2da058457c4156bb74b69e071cfb2da83a1fea00ba516d94421feac29b0a77 \
  --mode design-check
```

Expected: JSON containing `"status": "pass_design_only"`,
`"launch_authorized": false`, `"frame_contract_ready": false`,
`"active_status_runtime_present": false`, runtime blocker count `49`, and exit 0.
This proves only that the present design is explicit and fail-closed. It does
not prove any vendor behavior.

The following must return exit 1 and print one `MISSING` line for every null
binding, followed by the frame, machine-ACK, and one-shot blockers:

```bash
python3 scripts/validate_gate3_serve_sync_prereg.py \
  --repo-root . \
  --prereg configs/gate3_serve_sync_prereg_20260712.json \
  --expected-prereg-sha256 7a2da058457c4156bb74b69e071cfb2da83a1fea00ba516d94421feac29b0a77 \
  --mode launch-check
```

Focused regression:

```bash
python3 -m pytest -q tests/test_gate3_serve_sync_prereg.py
```

Promotion requires a new reviewed runtime contract and new evidence. Do not
backfill this preregistration and do not treat `pass_design_only` as permission
to launch.
