#!/usr/bin/env bash
# Start one long Isaac/Kit command while serializing only its boot window.
#
# The child is placed in a new process group and closes the lock fd, so the
# launcher can release the per-host boot lock as soon as a reliable log marker
# appears.  A pre-marker exit or timeout is handled by that exact PGID only.

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
poll_s=${KIT_BOOT_POLL_S:-5}
state_file=${KIT_BOOT_STATE_FILE:-${log_file}.launch}

if ! [[ $timeout_s =~ ^[1-9][0-9]*$ && $poll_s =~ ^[1-9][0-9]*$ ]]; then
  echo "KIT_BOOT_TIMEOUT_S and KIT_BOOT_POLL_S must be positive integers" >&2
  exit 2
fi

for required_tool in flock setsid ps grep; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    echo "required launch tool is missing: $required_tool" >&2
    exit 127
  fi
done

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
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  exit 1
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

echo "KIT_BOOT_STARTED pid=$pid pgid=$pgid log=$log_file marker=$marker"
deadline=$((SECONDS + timeout_s))
while true; do
  if grep -Fq -- "$marker" "$log_file"; then
    printf 'ready_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$state_file"
    flock -u 9
    echo "KIT_BOOT_READY pid=$pid pgid=$pgid log=$log_file"
    exit 0
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
    flock -u 9
    echo "KIT_BOOT_FAILED pid=$pid pgid=$pgid exit=$rc log=$log_file" >&2
    tail -n 80 "$log_file" >&2 || true
    exit "$rc"
  fi

  if (( SECONDS >= deadline )); then
    echo "KIT_BOOT_TIMEOUT pid=$pid pgid=$pgid after=${timeout_s}s log=$log_file" >&2
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    printf 'boot_timeout_s=%s\n' "$timeout_s" >>"$state_file"
    flock -u 9
    exit 124
  fi

  sleep "$poll_s"
done
