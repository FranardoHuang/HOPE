#!/usr/bin/env bash
# Start one long Isaac/Kit command while serializing only its boot window.
#
# The child is placed in a new process group and closes the lock fd, so the
# launcher can release the per-host boot lock as soon as a reliable log marker
# appears.  A pre-marker exit, hard timeout, or content-bearing stale log is
# handled by that exact PGID only.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 LOG_FILE COMMAND [ARG ...]" >&2
  exit 2
fi

log_file=$1
shift

lock_file=${KIT_BOOT_LOCK:-/workspace/.kit_boot.lock}
marker=${KIT_BOOT_MARKER:-Learning iteration}
timeout_s=${KIT_BOOT_TIMEOUT_S:-900}
stale_timeout_s=${KIT_BOOT_STALE_TIMEOUT_S:-180}
poll_s=${KIT_BOOT_POLL_S:-5}
state_file=${KIT_BOOT_STATE_FILE:-${log_file}.launch}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
identity_helper=${script_dir}/exact_process_group.py
leader_identity_file=${state_file}.leader.json
term_identity_file=${state_file}.pre_term.json
kill_identity_file=${state_file}.pre_kill.json

if ! [[ $timeout_s =~ ^[1-9][0-9]*$ && $stale_timeout_s =~ ^[1-9][0-9]*$ && $poll_s =~ ^[1-9][0-9]*$ ]]; then
  echo "KIT_BOOT_TIMEOUT_S, KIT_BOOT_STALE_TIMEOUT_S, and KIT_BOOT_POLL_S must be positive integers" >&2
  exit 2
fi

for required_tool in flock setsid ps grep stat python3; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    echo "required launch tool is missing: $required_tool" >&2
    exit 127
  fi
done
if [[ ! -f $identity_helper ]]; then
  echo "exact process-group identity helper is missing: $identity_helper" >&2
  exit 127
fi

mkdir -p "$(dirname "$log_file")" "$(dirname "$state_file")"
exec 9>"$lock_file"
flock -x 9

# Closing fd 9 in the child is essential: only this short-lived launcher owns
# the boot lock.  setsid makes pid == pgid for exact cleanup and later audit.
setsid "$@" </dev/null >"$log_file" 2>&1 9>&- &
pid=$!
pgid=$(ps -o pgid= -p "$pid" | tr -d '[:space:]')
if [[ -z $pgid || $pgid != "$pid" ]]; then
  echo "failed to establish an isolated process group: pid=$pid pgid=${pgid:-missing}" >&2
  {
    printf 'pid=%s\n' "$pid"
    printf 'observed_pgid=%s\n' "${pgid:-missing}"
    printf 'identity_bind_refused=pid_pgid_mismatch\n'
  } >"$state_file"
  echo "child identity was never bound; no signal sent; manual review required" >&2
  exit 121
fi

{
  printf 'pid=%s\n' "$pid"
  printf 'pgid=%s\n' "$pgid"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'marker=%s\n' "$marker"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
} >"$state_file"

set +e
leader_identity=$(python3 "$identity_helper" bind \
  --pid "$pid" --pgid "$pgid" --output "$leader_identity_file" 2>>"$state_file")
identity_rc=$?
set -e
if (( identity_rc != 0 )); then
  printf 'identity_bind_refused=proc_identity_unverified\n' >>"$state_file"
  echo "child proc identity was not bound; no signal sent; manual review required" >&2
  exit 121
fi
read -r bound_pid bound_pgid bound_starttime extra <<<"$leader_identity"
if [[ $bound_pid != "$pid" || $bound_pgid != "$pgid" || ! $bound_starttime =~ ^[1-9][0-9]*$ || -n ${extra:-} ]]; then
  printf 'identity_bind_refused=helper_output_invalid\n' >>"$state_file"
  echo "identity helper returned invalid binding; no signal sent; manual review required" >&2
  exit 121
fi
printf 'leader_starttime_ticks=%s\n' "$bound_starttime" >>"$state_file"
printf 'leader_identity_evidence=%s\n' "$leader_identity_file" >>"$state_file"

echo "KIT_BOOT_STARTED pid=$pid pgid=$pgid log=$log_file marker=$marker"
deadline=$((SECONDS + timeout_s))
last_log_size=
last_log_mtime=
last_log_change_s=

log_fingerprint() {
  local fingerprint
  if fingerprint=$(stat -c '%s %Y' -- "$log_file" 2>/dev/null); then
    :
  elif fingerprint=$(stat -f '%z %m' "$log_file" 2>/dev/null); then
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
  python3 "$identity_helper" term \
    --leader-evidence "$leader_identity_file" \
    --output "$term_identity_file" >>"$state_file" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf 'cleanup_identity_refused=before_term\n' >>"$state_file"
    echo "exact identity changed before TERM; no signal sent; manual review required" >&2
    return 121
  fi
  printf 'term_identity_evidence=%s\n' "$term_identity_file" >>"$state_file"
  for _ in 1 2 3 4 5; do
    set +e
    residual=$(python3 "$identity_helper" check --group-evidence "$term_identity_file" 2>>"$state_file")
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf 'cleanup_identity_refused=term_wait\n' >>"$state_file"
      echo "process-group identity changed after TERM; no KILL sent; manual review required" >&2
      return 122
    fi
    [[ $residual == 0 ]] && break
    sleep 1
  done
  if [[ ${residual:-1} != 0 ]]; then
    set +e
    python3 "$identity_helper" kill \
      --term-evidence "$term_identity_file" \
      --output "$kill_identity_file" >>"$state_file" 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf 'cleanup_identity_refused=before_kill\n' >>"$state_file"
      echo "residual group is not an exact subset of the TERM snapshot; no KILL sent" >&2
      return 122
    fi
    printf 'kill_identity_evidence=%s\n' "$kill_identity_file" >>"$state_file"
    for _ in 1 2 3 4 5; do
      set +e
      residual=$(python3 "$identity_helper" check --group-evidence "$term_identity_file" 2>>"$state_file")
      rc=$?
      set -e
      if (( rc != 0 )); then
        printf 'cleanup_identity_refused=kill_wait\n' >>"$state_file"
        return 122
      fi
      [[ $residual == 0 ]] && break
      sleep 1
    done
    if [[ ${residual:-1} != 0 ]]; then
      printf 'cleanup_residual_members=%s\n' "$residual" >>"$state_file"
      echo "exact residual process-group members remain after KILL; manual review required" >&2
      return 122
    fi
  fi
  wait "$pid" 2>/dev/null || true
  return 0
}

finish_ready() {
  printf 'ready_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$state_file"
  flock -u 9
  echo "KIT_BOOT_READY pid=$pid pgid=$pgid log=$log_file"
  exit 0
}

while true; do
  # Marker is authoritative even if it arrives on the same poll that would
  # otherwise cross either watchdog deadline.
  if grep -Fq -- "$marker" "$log_file"; then
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
    printf 'pre_marker_exit_code=%s\n' "$rc" >>"$state_file"
    printf 'terminal_kind=pre_marker_exit\n' >>"$state_file"
    printf 'terminal_exit_code=%s\n' "$rc" >>"$state_file"
    flock -u 9
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
    printf 'boot_watchdog_error=log_fingerprint\n' >>"$state_file"
    flock -u 9
    (( cleanup_rc == 0 )) || exit "$cleanup_rc"
    printf 'terminal_kind=watchdog_error\n' >>"$state_file"
    printf 'terminal_exit_code=126\n' >>"$state_file"
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
      if grep -Fq -- "$marker" "$log_file"; then
        finish_ready
      fi
      echo "KIT_BOOT_STALE pid=$pid pgid=$pgid after=${stale_timeout_s}s last_size=${last_log_size} log=$log_file" >&2
      printf 'boot_stale_timeout_s=%s\n' "$stale_timeout_s" >>"$state_file"
      printf 'boot_stale_last_size_bytes=%s\n' "$last_log_size" >>"$state_file"
      printf 'boot_stale_last_mtime_epoch=%s\n' "$last_log_mtime" >>"$state_file"
      set +e
      terminate_exact_group
      cleanup_rc=$?
      set -e
      flock -u 9
      (( cleanup_rc == 0 )) || exit "$cleanup_rc"
      printf 'terminal_kind=stale_timeout\n' >>"$state_file"
      printf 'terminal_exit_code=125\n' >>"$state_file"
      exit 125
    fi
  else
    last_log_size=
    last_log_mtime=
    last_log_change_s=
  fi

  if (( SECONDS >= deadline )); then
    if grep -Fq -- "$marker" "$log_file"; then
      finish_ready
    fi
    echo "KIT_BOOT_TIMEOUT pid=$pid pgid=$pgid after=${timeout_s}s log=$log_file" >&2
    set +e
    terminate_exact_group
    cleanup_rc=$?
    set -e
    printf 'boot_timeout_s=%s\n' "$timeout_s" >>"$state_file"
    flock -u 9
    (( cleanup_rc == 0 )) || exit "$cleanup_rc"
    printf 'terminal_kind=boot_timeout\n' >>"$state_file"
    printf 'terminal_exit_code=124\n' >>"$state_file"
    exit 124
  fi

  sleep "$poll_s"
done
