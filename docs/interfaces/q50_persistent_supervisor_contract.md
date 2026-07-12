# q50 persistent supervisor contract

Status: `Partial` source contract. Host tests pass, but this supervisor has not been deployed or
used to start a Pod process.

This interface defines a [persistent supervisor](../DEFINITIONS.md#persistent-supervisor): a tiny
manual launcher that keeps the already-reviewed model-4000 [q50](../DEFINITIONS.md#q50-and-k100)
consumer alive when the invoking SSH session disappears. It does not change the consumer, its
execution config, the all-four activation, either prepared runtime contract, the K100 paper, any
training checkout, or any policy/evaluation semantics.

## Authority boundary

The public command set is exactly:

- `launch`: perform one content-bound startup handshake in the Pod's fixed, previously absent state
  directory;
- `inspect`: read and validate preserved startup identity or the terminal Pod result.

There is no process-control, remote-login, retry, simulator, trainer, worker, deployment, or robot
authority. A failed or partial state directory is evidence and is never deleted or reused. Recovery
requires diagnosis and a separately reviewed version with a new fixed state directory.

## Immutable inputs

The caller pins the supervisor config by SHA-256. That config separately pins:

- the supervisor source bytes;
- the existing activation-consuming runner and execution-config bytes;
- the exact all-four activation bytes;
- Pod1 and Pod2's distinct prepared runtime-contract bytes;
- one fixed supervisor state directory and one expected terminal-result path per Pod;
- Pod1 arm order `seed1,seed3` and Pod2 arm order `seed2,seed4`.

The child command is constructed rather than accepted from the caller. Each Pod independently
binds the Python path, resolved executable path and binary SHA. The environment is a fixed config
map, not inherited from SSH; interpreter/shell injection variables such as `PYTHONPATH`,
`PYTHONHOME`, `LD_PRELOAD`, `BASH_ENV` and `ENV` are rejected. Exact argv and environment digests
are written to both hello and immutable launch ledger. Both public commands also require their own
invoking environment to equal that fixed map, so terminal inspection cannot import the bound runner
under an SSH-provided `PYTHONPATH` or loader environment.

## Two-phase startup

1. The parent creates the configured state directory and fixed combined stdout/stderr log with
   exclusive no-clobber operations.
2. The child creates a new session, redirects stdin to `/dev/null`, redirects stdout/stderr to the
   fixed log, closes all other inherited file descriptors, and records a hello containing
   `PID=PGID`, Linux boot id plus `/proc/<pid>/stat` start ticks, executable realpath/SHA, exact
   argv/environment digests, and every bound path/SHA.
3. The parent independently verifies the live `PID=PGID`, start ticks and complete hello. Only then
   does it atomically publish an immutable prepared ledger followed by a commit token that hashes
   both hello and ledger. The first possible visibility of the token's final no-clobber link is the
   single irreversible no-retry point. A successful directory fsync supplies durability evidence;
   if that fsync or any later parent-side observation/evidence write fails, the visible token still
   forbids retry and the parent returns committed pending with best-effort error evidence.
4. The child times out and exits by itself only while the token is absent. Once the token exists,
   the startup deadline has no cancellation meaning: the child revalidates identity, result absence
   and all bound bytes, publishes a no-clobber acknowledgment, then rechecks
   identity/token/ledger/result absence immediately before `execve`. A slow rehash or slow atomic
   acknowledgment publication after token commit is pending work, never retry authority. Any
   post-token validation/setup failure writes `child_exit.json` and does not execute the runner.
5. The parent first observes acknowledgment for its own bounded window. Absence yields
   `token_published_pending_ack` with return code zero. After a valid acknowledgment it uses a
   separate bounded exec-observation window: exact executable/argv/environment yields
   `running_exact`; a still-live pre-exec child yields `committed_pending_exec` with return code
   zero. The fixed state directory rejects every second launch in all committed states. Later
   `inspect` converges to exact running, a validated terminal result, or
   `committed_child_failed` (return code 3).

The parent never needs a cleanup operation. A parent crash before the token leaves no execution
authority; the child self-exits after the configured timeout. A crash after the token leaves the
immutable ledger needed to inspect the detached runner.

## Read-only inspection

For a live process, `inspect` requires the ledger boot id, PID, PGID, `/proc` start ticks and
executable realpath/SHA to keep identifying the same child. Before acknowledgment it reports
`token_published_pending_ack` even if the old startup deadline has passed. After acknowledgment, a
non-runner command line is `committed_pending_exec`; exact runner argv plus the wrong environment,
PID reuse or identity drift is `committed_child_failed`, while exact argv/environment is
`running_exact`. This prevents a reused PID or same-argv different executable from being reported
as the q50 runner without misclassifying either pre-ack atomic publication or post-ack pre-exec
stalls.

After process exit, `inspect` freezes the configured `pod_result.json` bytes and SHA before invoking
the original bound runner's complete validator with that exact SHA. It then rereads the file and
requires byte/SHA stability, a valid canonical content hash, and exact equality between document
content and validator-returned content. Exit without that result is reported as
`committed_child_failed`; it is never relabelled successful from a return code or log alone.

## Verification and limitations

The focused host suite covers the no-clobber ledger/token/ack, duplicate launch rejection,
pre-existing result rejection, parent exit and parent stall before commit, child token timeout,
artifact mismatch, reused-PID/executable/environment mismatch, exact live inspection, minimal-result
rejection and delegation to the original runner's full terminal-result validator. It also scans the
source for remote-login and process-control APIs. Config, hello, ledger, token, acknowledgment and
terminal wrapper JSON all reject duplicate object keys and non-finite constants before semantic
validation, so two conforming consumers cannot assign different meanings to one evidence file. A
deterministic pre-token stall crosses the deadline and asserts that no token or fake-runner marker
exists. Two post-token regressions cross that same old deadline: one delays the complete rehash and
one stalls the acknowledgment's atomic publication for 1.15 seconds. Both return
`token_published_pending_ack`, reject a second launch, emit no fatal-before-later-runner sequence,
and converge through `inspect` to exact running. A separate post-ack stall produces
`committed_pending_exec` with the same no-retry/convergence property. An A-to-B result swap during
bound validation is rejected. Two more deterministic regressions fail the token directory fsync
after its final link and fail the parent observation write after its final link. Both return
`token_published_pending_ack` with `retry_authorized=false`, preserve the no-clobber state, start no
duplicate, and later converge through `inspect` without a parent or child fatal. The focused suite
has 23 cases; the queue, consumer and supervisor set has 63.

This contract does not prove that an external Pod manager will preserve processes when it destroys
an entire container or cgroup. It only removes ordinary SSH file-descriptor/session lifetime from
the top-level runner. The verified top-level environment cannot close untracked environment files
that the already-bound `judge.sh` may source later; that pre-existing evaluation-tool closure is a
separate limitation. Hash-check-to-open TOCTOU is also trusted only inside the no-clobber control
directory. This interface does not validate a MuJoCo score; G05/G06 remain `Partial` until the
actual q50 results are complete and independently aggregated.
