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
   does it atomically publish an immutable launch ledger followed by a commit token that hashes both
   hello and ledger.
4. The child times out and exits by itself if the token is absent. When the token is present, it
   revalidates identity, result absence and all bound bytes; after potentially slow rehashes it
   checks deadline/identity/token immediately before the no-clobber commit acknowledgment. A timely
   validated acknowledgment is the irreversible commit point. After it, the child no longer uses
   the commit deadline as a cancellation promise, but it still rechecks identity/token/ledger/result
   absence immediately before `execve`.
5. The parent observes exec for a separate bounded window. Exact executable/argv/environment yields
   `running_exact`; otherwise it records and returns `committed_pending_exec`, never a launch error.
   The fixed state directory still rejects every second launch. Later `inspect` converges pending to
   exact running, a validated terminal result, or terminal failure.

The parent never needs a cleanup operation. A parent crash before the token leaves no execution
authority; the child self-exits after the configured timeout. A crash after the token leaves the
immutable ledger needed to inspect the detached runner.

## Read-only inspection

For a live process, `inspect` requires the ledger boot id, PID, PGID, `/proc` start ticks and
executable realpath/SHA to keep identifying the same child. Before acknowledgment it reports
`token_published_pre_ack` only until the commit deadline. After acknowledgment, a non-runner
command line is `committed_pending_exec`; exact runner argv plus the wrong environment is an error,
while exact argv/environment is `running_exact`. This prevents a reused PID or same-argv different
executable from being reported as the q50 runner without misclassifying a post-ack pre-exec stall.

After process exit, `inspect` freezes the configured `pod_result.json` bytes and SHA before invoking
the original bound runner's complete validator with that exact SHA. It then rereads the file and
requires byte/SHA stability, a valid canonical content hash, and exact equality between document
content and validator-returned content. Exit without that result is reported as
`terminal_without_result`; it is never relabelled successful from a return code or log alone.

## Verification and limitations

The focused host suite covers the no-clobber ledger/token/ack, duplicate launch rejection,
pre-existing result rejection, parent exit and parent stall before commit, child token timeout,
artifact mismatch, reused-PID/executable/environment mismatch, exact live inspection, minimal-result
rejection and delegation to the original runner's full terminal-result validator. It also scans the
source for remote-login and process-control APIs. Config, hello, ledger, token, acknowledgment and
terminal wrapper JSON all reject duplicate object keys and non-finite constants before semantic
validation, so two conforming consumers cannot assign different meanings to one evidence file. A
deterministic delayed-rehash regression crosses the deadline and asserts no acknowledgment, result
or fake-runner marker exists. A separate post-ack chdir stall produces `committed_pending_exec`,
rejects a second launch, then converges through `inspect` to exact running. An A-to-B result swap
during bound validation is rejected.

This contract does not prove that an external Pod manager will preserve processes when it destroys
an entire container or cgroup. It only removes ordinary SSH file-descriptor/session lifetime from
the top-level runner. The verified top-level environment cannot close untracked environment files
that the already-bound `judge.sh` may source later; that pre-existing evaluation-tool closure is a
separate limitation. Hash-check-to-open TOCTOU is also trusted only inside the no-clobber control
directory. This interface does not validate a MuJoCo score; G05/G06 remain `Partial` until the
actual q50 results are complete and independently aggregated.
