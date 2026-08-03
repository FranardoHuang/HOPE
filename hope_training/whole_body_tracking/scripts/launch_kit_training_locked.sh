#!/bin/bash -p
# Start one long Isaac/Kit command while serializing only its boot window.
#
# The child is placed in a new process group and closes the lock fd, so the
# launcher can release the per-host boot lock as soon as a reliable log marker
# appears.  A pre-marker exit, hard timeout, or content-bearing stale log is
# handled by that exact PGID only.

set -euo pipefail

readonly TRUSTED_PATH=/usr/bin:/bin
readonly BASH_BIN=/bin/bash
readonly ENV_BIN=/usr/bin/env
readonly FLOCK_BIN=/usr/bin/flock
readonly SETSID_BIN=/usr/bin/setsid
readonly PS_BIN=/usr/bin/ps
readonly GREP_BIN=/usr/bin/grep
readonly STAT_BIN=/usr/bin/stat
readonly MKFIFO_BIN=/usr/bin/mkfifo
readonly PYTHON_BIN=/usr/bin/python3.10
export PATH=$TRUSTED_PATH

# None of these caller-controlled hooks are inputs to this launcher.  Clear
# them before invoking Python, Bash, Git-adjacent setup, or any native tool.
while IFS='=' read -r variable_name _; do
  case $variable_name in
    PYTHONPATH|PYTHONHOME|BASH_ENV|ENV|GIT*|LD_*|DYLD_*|XDG_*)
      unset "$variable_name"
      ;;
  esac
done < <("$ENV_BIN")

internal_handoff_token=
if [[ ${1:-} == --hope-private-fd-handoff ]]; then
  if [[ $# -lt 4 ]]; then
    echo "invalid private Kit launch handoff" >&2
    exit 2
  fi
  internal_handoff_token=$2
  shift 2
fi
if [[ -n ${HOPE_KIT_BOOT_FDS:-} ]]; then
  echo "caller-supplied HOPE_KIT_BOOT_FDS is forbidden" >&2
  exit 2
fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 LOG_FILE COMMAND [ARG ...]" >&2
  exit 2
fi

log_file=$1
shift

lock_file=/workspace/.kit_boot.lock
if [[ -n ${KIT_BOOT_LOCK:-} && ${KIT_BOOT_LOCK} != "$lock_file" ]]; then
  echo "KIT_BOOT_LOCK must be the pod-wide $lock_file" >&2
  exit 2
fi
marker=${KIT_BOOT_MARKER:-Learning iteration}
timeout_s=${KIT_BOOT_TIMEOUT_S:-900}
stale_timeout_s=${KIT_BOOT_STALE_TIMEOUT_S:-180}
poll_s=${KIT_BOOT_POLL_S:-5}
wait_for_completion=${KIT_WAIT_FOR_COMPLETION:-0}
completion_timeout_s=${KIT_COMPLETION_TIMEOUT_S:-120}
state_file=${KIT_BOOT_STATE_FILE:-${log_file}.launch}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
identity_helper=${script_dir}/exact_process_group.py
leader_identity_file=${state_file}.leader.json
term_identity_file=${state_file}.pre_term.json
kill_identity_file=${state_file}.pre_kill.json
start_gate_file=${state_file}.start_gate

if ! [[ $timeout_s =~ ^[1-9][0-9]*$ && $stale_timeout_s =~ ^[1-9][0-9]*$ && $poll_s =~ ^[1-9][0-9]*$ && $completion_timeout_s =~ ^[1-9][0-9]*$ && $wait_for_completion =~ ^[01]$ ]]; then
  echo "KIT_BOOT_TIMEOUT_S, KIT_BOOT_STALE_TIMEOUT_S, KIT_BOOT_POLL_S, and KIT_COMPLETION_TIMEOUT_S must be positive integers; KIT_WAIT_FOR_COMPLETION must be 0 or 1" >&2
  exit 2
fi

for required_tool in \
  "$BASH_BIN" "$ENV_BIN" "$FLOCK_BIN" "$SETSID_BIN" "$PS_BIN" "$GREP_BIN" \
  "$STAT_BIN" "$MKFIFO_BIN" "$PYTHON_BIN"; do
  if [[ ! -x $required_tool ]]; then
    echo "required trusted launch tool is missing: $required_tool" >&2
    exit 127
  fi
done
if [[ ! -f $identity_helper ]]; then
  echo "exact process-group identity helper is missing: $identity_helper" >&2
  exit 127
fi

mkdir -p "$(dirname "$log_file")" "$(dirname "$state_file")"
if [[ -z $internal_handoff_token ]]; then
  script_path=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
  exec "$PYTHON_BIN" -c '
import fcntl
import os
import secrets
import stat
import sys

(
    lock_path,
    log_path,
    state_path,
    leader_path,
    term_path,
    kill_path,
    gate_path,
    script,
    *arguments
) = sys.argv[1:]
for reserved in (leader_path, term_path, kill_path, gate_path):
    if os.path.lexists(reserved):
        raise SystemExit("no-clobber launch artifact already exists: %s" % reserved)

if not hasattr(os, "O_NOFOLLOW"):
    raise SystemExit("formal Kit launch requires os.O_NOFOLLOW")
nofollow = os.O_NOFOLLOW
cloexec = getattr(os, "O_CLOEXEC", 0)
lock_flags = os.O_RDWR | os.O_APPEND | nofollow | cloexec
new_flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | os.O_APPEND
    | nofollow
    | cloexec
)
opened = []
try:
    lock_fd = os.open(lock_path, lock_flags)
    opened.append(lock_fd)
    log_fd = os.open(log_path, new_flags, 0o600)
    opened.append(log_fd)
    state_fd = os.open(state_path, new_flags, 0o600)
    opened.append(state_fd)
except OSError as exc:
    for descriptor in opened:
        os.close(descriptor)
    raise SystemExit("no-clobber launch bootstrap refused: %s" % exc)

def same_regular_path(descriptor, path):
    descriptor_stat = os.fstat(descriptor)
    path_stat = os.lstat(path)
    return (
        stat.S_ISREG(descriptor_stat.st_mode)
        and stat.S_ISREG(path_stat.st_mode)
        and (descriptor_stat.st_dev, descriptor_stat.st_ino)
        == (path_stat.st_dev, path_stat.st_ino)
    )

if not same_regular_path(lock_fd, lock_path):
    raise SystemExit("pod-wide Kit boot lock must be one stable regular file")
if not same_regular_path(log_fd, log_path):
    raise SystemExit("launch log must be one stable regular file")
if not same_regular_path(state_fd, state_path):
    raise SystemExit("launch state must be one stable regular file")

# Duplicate above the target range first, otherwise dup2 could clobber another
# descriptor that has not yet been moved.
high = [
    fcntl.fcntl(descriptor, fcntl.F_DUPFD, 20)
    for descriptor in (log_fd, state_fd, lock_fd)
]
for descriptor in (log_fd, state_fd, lock_fd):
    os.close(descriptor)
for source, target in zip(high, (7, 8, 9)):
    os.dup2(source, target)
    os.close(source)
    os.set_inheritable(target, True)

token = secrets.token_hex(32)
read_fd, write_fd = os.pipe()
try:
    if os.write(write_fd, token.encode("ascii")) != len(token):
        raise SystemExit("private handoff token write was short")
finally:
    os.close(write_fd)
handoff_source = fcntl.fcntl(read_fd, fcntl.F_DUPFD, 20)
os.close(read_fd)
os.dup2(handoff_source, 10)
os.close(handoff_source)
os.set_inheritable(10, True)

environment = dict(os.environ)
os.execve(
    script,
    [script, "--hope-private-fd-handoff", token, log_path, *arguments],
    environment,
)
' "$lock_file" "$log_file" "$state_file" \
  "$leader_identity_file" "$term_identity_file" "$kill_identity_file" \
  "$start_gate_file" "$script_path" "$@"
fi
"$PYTHON_BIN" -c '
import hashlib
import hmac
import os
import re
import stat
import sys

expected_token = sys.argv[1]
descriptor_paths = sys.argv[2:5]
tool_paths = sys.argv[5:]
if re.fullmatch(r"[0-9a-f]{64}", expected_token) is None:
    raise SystemExit("private handoff token is malformed")
try:
    observed_token = os.read(10, 65).decode("ascii")
finally:
    os.close(10)
if not hmac.compare_digest(observed_token, expected_token):
    raise SystemExit("private handoff token differs from inherited descriptor")
for descriptor_text, path in zip(("7", "8", "9"), descriptor_paths):
    descriptor = int(descriptor_text)
    descriptor_stat = os.fstat(descriptor)
    path_stat = os.lstat(path)
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise SystemExit(
            "inherited launch descriptor does not bind stable path: %s" % path
        )
    if descriptor in (7, 8) and (
        descriptor_stat.st_size != 0
        or path_stat.st_size != 0
        or descriptor_stat.st_nlink != 1
        or path_stat.st_nlink != 1
    ):
        raise SystemExit(
            "new launch log/state must be empty single-link files: %s" % path
        )
print(
    "bootstrap_handoff_token_sha256=%s"
    % hashlib.sha256(expected_token.encode("ascii")).hexdigest()
)
for path in tool_paths:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or not os.access(path, os.X_OK):
        raise SystemExit("trusted launch tool is not regular/executable: %s" % path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or identity != (
        final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns
    ):
        raise SystemExit("trusted launch tool changed while hashing: %s" % path)
    label = os.path.basename(path).replace(".", "_")
    print("trusted_tool_%s_sha256=%s" % (label, digest.hexdigest()))
' "$internal_handoff_token" "$log_file" "$state_file" "$lock_file" \
  "$BASH_BIN" "$ENV_BIN" "$FLOCK_BIN" "$SETSID_BIN" "$PS_BIN" \
  "$GREP_BIN" "$STAT_BIN" "$MKFIFO_BIN" "$PYTHON_BIN" >&8
"$FLOCK_BIN" -x 9
"$PYTHON_BIN" -c '
import os
import stat
import sys

path = sys.argv[1]
descriptor = os.fstat(9)
pathname = os.lstat(path)
if (
    not stat.S_ISREG(descriptor.st_mode)
    or not stat.S_ISREG(pathname.st_mode)
    or descriptor.st_nlink != 1
    or pathname.st_nlink != 1
    or (descriptor.st_dev, descriptor.st_ino)
    != (pathname.st_dev, pathname.st_ino)
):
    raise SystemExit("Kit boot lock pathname identity changed after flock")
' "$lock_file"

# The setsid leader is a tiny inherited-FD gate wrapper.  It cannot execute
# the workload until the parent has durably bound PID=PGID plus starttime.
stop_signal=
trap '[[ -n $stop_signal ]] || stop_signal=INT' INT
trap '[[ -n $stop_signal ]] || stop_signal=TERM' TERM
trap '[[ -n $stop_signal ]] || stop_signal=HUP' HUP

stop_exit_code() {
  case $stop_signal in
    HUP) printf '129\n' ;;
    INT) printf '130\n' ;;
    TERM) printf '143\n' ;;
    *) printf '1\n' ;;
  esac
}

"$MKFIFO_BIN" -m 600 -- "$start_gate_file"
exec 6<>"$start_gate_file"
gate_program='
import os
import stat
import sys

descriptor = int(sys.argv[1])
if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
    raise SystemExit(125)
token = os.read(descriptor, 2)
os.close(descriptor)
if token != b"G":
    raise SystemExit(125)
command = sys.argv[2:]
if not command:
    raise SystemExit(125)
if not os.path.isabs(command[0]):
    raise SystemExit(125)
os.execv(command[0], command)
'
"$SETSID_BIN" "$PYTHON_BIN" -c "$gate_program" 6 "$@" \
  </dev/null 1>&7 2>&1 7>&- 8>&- 9>&- &
pid=$!
pgid=$("$PS_BIN" -o pgid= -p "$pid" | tr -d '[:space:]')

reap_still_gated_child() {
  local reason=$1
  printf 'X' >&6 || true
  exec 6>&-
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      printf 'still_gated_reaped=%s\n' "$reason" >&8
      return 0
    fi
    sleep 1
  done
  printf 'still_gated_reap_refused=%s\n' "$reason" >&8
  echo "still-gated child did not reap; no PID-only signal sent" >&2
  return 121
}

if [[ -z $pgid || $pgid != "$pid" ]]; then
  printf 'pid=%s\n' "$pid" >&8
  printf 'observed_pgid=%s\n' "${pgid:-missing}" >&8
  printf 'identity_bind_refused=pid_pgid_mismatch\n' >&8
  echo "failed to establish isolated still-gated process group" >&2
  set +e
  reap_still_gated_child pid_pgid_mismatch
  reap_rc=$?
  set -e
  (( reap_rc == 0 )) && exit 121
  exit "$reap_rc"
fi
if [[ -n $stop_signal ]]; then
  printf 'startup_stop_signal=%s\n' "$stop_signal" >&8
  set +e
  reap_still_gated_child "signal_${stop_signal}"
  reap_rc=$?
  set -e
  "$FLOCK_BIN" -u 9
  (( reap_rc == 0 )) || exit "$reap_rc"
  exit "$(stop_exit_code)"
fi

{
  printf 'pid=%s\n' "$pid"
  printf 'pgid=%s\n' "$pgid"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'marker=%s\n' "$marker"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
} >&8

set +e
leader_identity=$("$PYTHON_BIN" "$identity_helper" bind \
  --pid "$pid" --pgid "$pgid" --output "$leader_identity_file" 2>&8)
identity_rc=$?
set -e
if (( identity_rc != 0 )); then
  printf 'identity_bind_refused=proc_identity_unverified\n' >&8
  echo "child identity bind refused while workload remained gated" >&2
  set +e
  reap_still_gated_child identity_bind_refused
  reap_rc=$?
  set -e
  (( reap_rc == 0 )) && exit 121
  exit "$reap_rc"
fi
read -r bound_pid bound_pgid bound_starttime extra <<<"$leader_identity"
if [[ $bound_pid != "$pid" || $bound_pgid != "$pgid" || ! $bound_starttime =~ ^[1-9][0-9]*$ || -n ${extra:-} ]]; then
  printf 'identity_bind_refused=helper_output_invalid\n' >&8
  echo "identity helper returned invalid binding while workload remained gated" >&2
  set +e
  reap_still_gated_child helper_output_invalid
  reap_rc=$?
  set -e
  (( reap_rc == 0 )) && exit 121
  exit "$reap_rc"
fi
printf 'leader_starttime_ticks=%s\n' "$bound_starttime" >&8
printf 'leader_identity_evidence=%s\n' "$leader_identity_file" >&8
if [[ -n $stop_signal ]]; then
  printf 'bound_stop_signal=%s\n' "$stop_signal" >&8
  set +e
  reap_still_gated_child "signal_${stop_signal}"
  reap_rc=$?
  set -e
  "$FLOCK_BIN" -u 9
  (( reap_rc == 0 )) || exit "$reap_rc"
  exit "$(stop_exit_code)"
fi
printf 'G' >&6
exec 6>&-
printf 'workload_gate_released_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&8

echo "KIT_BOOT_STARTED pid=$pid pgid=$pgid log=$log_file marker=$marker"
deadline=$((SECONDS + timeout_s))
last_log_size=
last_log_mtime=
last_log_change_s=

log_fingerprint() {
  local fingerprint
  if fingerprint=$("$STAT_BIN" -c '%s %Y' -- "$log_file" 2>/dev/null); then
    :
  elif fingerprint=$("$STAT_BIN" -f '%z %m' "$log_file" 2>/dev/null); then
    :
  else
    return 1
  fi
  if ! [[ $fingerprint =~ ^[0-9]+[[:space:]][0-9]+$ ]]; then
    return 1
  fi
  printf '%s\n' "$fingerprint"
}

terminate_exact_group() {
  local rc residual
  set +e
  "$PYTHON_BIN" "$identity_helper" term \
    --leader-evidence "$leader_identity_file" \
    --output "$term_identity_file" >&8 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf 'cleanup_identity_refused=before_term\n' >&8
    echo "exact identity changed before TERM; no signal sent; manual review required" >&2
    return 121
  fi
  printf 'term_identity_evidence=%s\n' "$term_identity_file" >&8
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
    fi
    set +e
    residual=$("$PYTHON_BIN" "$identity_helper" check --group-evidence "$term_identity_file" 2>&8)
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf 'cleanup_identity_refused=term_wait\n' >&8
      echo "process-group identity changed after TERM; no KILL sent; manual review required" >&2
      return 122
    fi
    [[ $residual == 0 ]] && break
    sleep 1
  done
  if [[ ${residual:-1} != 0 ]]; then
    set +e
    "$PYTHON_BIN" "$identity_helper" kill \
      --term-evidence "$term_identity_file" \
      --output "$kill_identity_file" >&8 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf 'cleanup_identity_refused=before_kill\n' >&8
      echo "residual group is not an exact subset of the TERM snapshot; no KILL sent" >&2
      return 122
    fi
    printf 'kill_identity_evidence=%s\n' "$kill_identity_file" >&8
    for _ in 1 2 3 4 5; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
      fi
      set +e
      residual=$("$PYTHON_BIN" "$identity_helper" check --group-evidence "$term_identity_file" 2>&8)
      rc=$?
      set -e
      if (( rc != 0 )); then
        printf 'cleanup_identity_refused=kill_wait\n' >&8
        return 122
      fi
      [[ $residual == 0 ]] && break
      sleep 1
    done
    if [[ ${residual:-1} != 0 ]]; then
      printf 'cleanup_residual_members=%s\n' "$residual" >&8
      echo "exact residual process-group members remain after KILL; manual review required" >&2
      return 122
    fi
  fi
  wait "$pid" 2>/dev/null || true
  return 0
}

cleanup_completed_group() {
  local rc residual
  completed_initial_members=
  set +e
  completed_initial_members=$("$PYTHON_BIN" "$identity_helper" completed-term \
    --leader-evidence "$leader_identity_file" \
    --output "$term_identity_file" 2>&8)
  rc=$?
  set -e
  if (( rc != 0 )) || ! [[ $completed_initial_members =~ ^[0-9]+$ ]]; then
    printf 'completion_cleanup_identity_refused=before_term\n' >&8
    echo "completed-group identity refused before descendant TERM; quarantined for manual review" >&2
    return 121
  fi
  printf 'completion_term_identity_evidence=%s\n' "$term_identity_file" >&8
  residual=$completed_initial_members
  for _ in 1 2 3 4 5; do
    set +e
    residual=$("$PYTHON_BIN" "$identity_helper" completed-check \
      --group-evidence "$term_identity_file" 2>&8)
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf 'completion_cleanup_identity_refused=term_wait\n' >&8
      echo "completed process-group identity changed after TERM; quarantined for manual review" >&2
      return 122
    fi
    [[ $residual == 0 ]] && break
    sleep 1
  done
  if [[ ${residual:-1} != 0 ]]; then
    set +e
    "$PYTHON_BIN" "$identity_helper" completed-kill \
      --term-evidence "$term_identity_file" \
      --output "$kill_identity_file" >&8 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf 'completion_cleanup_identity_refused=before_kill\n' >&8
      echo "completed residual group is not an exact TERM-snapshot subset; quarantined" >&2
      return 122
    fi
    printf 'completion_kill_identity_evidence=%s\n' "$kill_identity_file" >&8
    for _ in 1 2 3 4 5; do
      set +e
      residual=$("$PYTHON_BIN" "$identity_helper" completed-check \
        --group-evidence "$term_identity_file" 2>&8)
      rc=$?
      set -e
      if (( rc != 0 )); then
        printf 'completion_cleanup_identity_refused=kill_wait\n' >&8
        return 122
      fi
      [[ $residual == 0 ]] && break
      sleep 1
    done
    if [[ ${residual:-1} != 0 ]]; then
      printf 'completion_cleanup_residual_members=%s\n' "$residual" >&8
      echo "exact completed-group residual members remain after KILL; quarantined" >&2
      return 122
    fi
  fi
  printf 'completion_cleanup_completed=true\n' >&8
  return 0
}

finish_ready() {
  printf 'ready_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&8
  "$FLOCK_BIN" -u 9
  echo "KIT_BOOT_READY pid=$pid pgid=$pgid log=$log_file"
  exit 0
}

finish_completed() {
  local completion_deadline process_state rc cleanup_rc
  completion_deadline=$((SECONDS + completion_timeout_s))
  while true; do
    process_state=$($PS_BIN -o stat= -p "$pid" 2>/dev/null || true)
    process_state=${process_state//[[:space:]]/}
    [[ -z $process_state || $process_state == Z* ]] && break
    if (( SECONDS >= completion_deadline )); then
      printf 'completion_timeout_s=%s\n' "$completion_timeout_s" >&8
      set +e
      terminate_exact_group
      rc=$?
      set -e
      "$FLOCK_BIN" -u 9
      (( rc == 0 )) || exit "$rc"
      exit 124
    fi
    sleep 1
  done
  set +e
  wait "$pid"
  rc=$?
  set -e
  printf 'completion_exit_code=%s\n' "$rc" >&8
  set +e
  cleanup_completed_group
  cleanup_rc=$?
  set -e
  if (( cleanup_rc != 0 )); then
    printf 'terminal_kind=completion_cleanup_quarantined\n' >&8
    printf 'terminal_exit_code=%s\n' "$cleanup_rc" >&8
    "$FLOCK_BIN" -u 9
    exit "$cleanup_rc"
  fi
  if (( rc != 0 )); then
    printf 'terminal_kind=completion_nonzero_exit\n' >&8
    printf 'terminal_exit_code=%s\n' "$rc" >&8
    "$FLOCK_BIN" -u 9
    exit "$rc"
  fi
  if (( completed_initial_members != 0 )); then
    printf 'terminal_kind=completion_residual_group\n' >&8
    printf 'terminal_exit_code=123\n' >&8
    "$FLOCK_BIN" -u 9
    exit 123
  fi
  printf 'completion_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&8
  printf 'terminal_kind=clean_completion\n' >&8
  printf 'terminal_exit_code=0\n' >&8
  "$FLOCK_BIN" -u 9
  echo "KIT_COMPLETION_READY pid=$pid pgid=$pgid log=$log_file"
  exit 0
}

while true; do
  if [[ -n $stop_signal ]]; then
    echo "KIT_BOOT_STOP pid=$pid pgid=$pgid signal=$stop_signal" >&2
    printf 'stop_signal=%s\n' "$stop_signal" >&8
    set +e
    terminate_exact_group
    cleanup_rc=$?
    set -e
    "$FLOCK_BIN" -u 9
    (( cleanup_rc == 0 )) || exit "$cleanup_rc"
    printf 'terminal_kind=signal_stop\n' >&8
    printf 'terminal_exit_code=%s\n' "$(stop_exit_code)" >&8
    exit "$(stop_exit_code)"
  fi
  # Marker is authoritative even if it arrives on the same poll that would
  # otherwise cross either watchdog deadline.
  if "$GREP_BIN" -Fq -- "$marker" "$log_file"; then
    if [[ $wait_for_completion == 1 ]]; then
      finish_completed
    fi
    finish_ready
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    set +e
    wait "$pid"
    rc=$?
    set -e
    if (( rc == 0 )); then
      # A clean child exit is still a failed boot when the required marker was
      # never emitted; returning zero here would falsely authorize the arm.
      rc=1
    fi
    printf 'pre_marker_exit_code=%s\n' "$rc" >&8
    printf 'terminal_kind=pre_marker_exit\n' >&8
    printf 'terminal_exit_code=%s\n' "$rc" >&8
    "$FLOCK_BIN" -u 9
    echo "KIT_BOOT_FAILED pid=$pid pgid=$pgid exit=$rc log=$log_file" >&2
    tail -n 80 "$log_file" >&2 || true
    exit "$rc"
  fi

  fingerprint=$(log_fingerprint) || {
    echo "KIT_BOOT_WATCHDOG_ERROR pid=$pid pgid=$pgid unable_to_stat_log=$log_file" >&2
    set +e
    terminate_exact_group
    cleanup_rc=$?
    set -e
    printf 'boot_watchdog_error=log_fingerprint\n' >&8
    "$FLOCK_BIN" -u 9
    (( cleanup_rc == 0 )) || exit "$cleanup_rc"
    printf 'terminal_kind=watchdog_error\n' >&8
    printf 'terminal_exit_code=126\n' >&8
    exit 126
  }
  log_size=${fingerprint%% *}
  log_mtime=${fingerprint#* }

  # An empty redirection target is not evidence of a wedged importer; the hard
  # timeout remains responsible until the child has emitted at least one byte.
  if (( log_size > 0 )); then
    if [[ -z $last_log_change_s || $log_size != "$last_log_size" || $log_mtime != "$last_log_mtime" ]]; then
      last_log_size=$log_size
      last_log_mtime=$log_mtime
      last_log_change_s=$SECONDS
    elif (( SECONDS - last_log_change_s >= stale_timeout_s )); then
      # Close the check-to-signal window once more before classifying a stale
      # importer; a marker that arrived during stat processing still wins.
      if "$GREP_BIN" -Fq -- "$marker" "$log_file"; then
        finish_ready
      fi
      echo "KIT_BOOT_STALE pid=$pid pgid=$pgid after=${stale_timeout_s}s last_size=${last_log_size} log=$log_file" >&2
      printf 'boot_stale_timeout_s=%s\n' "$stale_timeout_s" >&8
      printf 'boot_stale_last_size_bytes=%s\n' "$last_log_size" >&8
      printf 'boot_stale_last_mtime_epoch=%s\n' "$last_log_mtime" >&8
      set +e
      terminate_exact_group
      cleanup_rc=$?
      set -e
      "$FLOCK_BIN" -u 9
      (( cleanup_rc == 0 )) || exit "$cleanup_rc"
      printf 'terminal_kind=stale_timeout\n' >&8
      printf 'terminal_exit_code=125\n' >&8
      exit 125
    fi
  else
    last_log_size=
    last_log_mtime=
    last_log_change_s=
  fi

  if (( SECONDS >= deadline )); then
    if "$GREP_BIN" -Fq -- "$marker" "$log_file"; then
      finish_ready
    fi
    echo "KIT_BOOT_TIMEOUT pid=$pid pgid=$pgid after=${timeout_s}s log=$log_file" >&2
    set +e
    terminate_exact_group
    cleanup_rc=$?
    set -e
    printf 'boot_timeout_s=%s\n' "$timeout_s" >&8
    "$FLOCK_BIN" -u 9
    (( cleanup_rc == 0 )) || exit "$cleanup_rc"
    printf 'terminal_kind=boot_timeout\n' >&8
    printf 'terminal_exit_code=124\n' >&8
    exit 124
  fi

  sleep "$poll_s"
done
