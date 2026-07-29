#!/bin/bash
# Read-only hourly patrol for explicitly bound training namespaces.
#
# Sources of truth, in decreasing priority:
#   1. repeated --pod/--root/--namespace/--checkout arguments;
#   2. --spec (or POD_PATROL_SPEC), whose matrix/patrol records bind the same;
#   3. the current N1 wave record, when it exists.
#
# The patrol never sends a signal and never writes on a pod.  Its small progress
# cursor lives on the caller under /tmp (override with --state-dir).  A run is
# stale only when its last observed Learning iteration/checkpoint stops
# advancing; run.log mtime is deliberately ignored because unrelated stdout can
# keep a wedged trainer looking alive.
#
# Backward-compatible:
#   bash scripts/pod_patrol.sh
#   bash scripts/pod_patrol.sh --full
#
# Explicit current-wave form:
#   bash scripts/pod_patrol.sh --full \
#     --spec configs/n1_contact_20260729/n1_upper_reward_wave_20260729.v1.json \
#     --pod pod1=root@HOST:PORT \
#     --root pod1=/workspace/franco/CURRENT_RUN_ROOT
set -u

parse_rsl_iteration_block() {
  # Print the newest complete RSL-RL block as:
  # iteration, update_s, collection_s, learning_s, env_steps_per_s, timesteps.
  # Starting a newer block does not discard the previous complete block.
  awk -v emit_all="${2:-0}" '
    function number_after(text, prefix, suffix, value) {
      value = text
      sub(prefix, "", value)
      sub(suffix, "", value)
      gsub(/[[:space:]]/, "", value)
      return value
    }
    function save_complete() {
      if (active && have_computation && have_total && have_update) {
        latest_iteration = iteration
        latest_update = update_s
        latest_collection = collection_s
        latest_learning = learning_s
        latest_rate = env_steps_per_s
        latest_total = total_timesteps
        found = 1
        if (emit_all && emitted_iteration != iteration) {
          printf "%s\t%s\t%s\t%s\t%s\t%s\n", iteration, update_s,
            collection_s, learning_s, env_steps_per_s, total_timesteps
          emitted_iteration = iteration
        }
      }
    }
    {
      line = $0
      gsub(/\033\[[0-9;]*[A-Za-z]/, "", line)
      if (match(line, /Learning iteration[[:space:]]+[0-9]+\/[0-9]+/)) {
        token = substr(line, RSTART, RLENGTH)
        sub(/^.*Learning iteration[[:space:]]+/, "", token)
        sub(/\/.*$/, "", token)
        iteration = token
        active = 1
        have_computation = have_total = have_update = 0
        next
      }
      if (!active) {
        next
      }
      if (line ~ /Computation:[[:space:]]*[0-9.]+[[:space:]]+steps\/s/ &&
          line ~ /collection:[[:space:]]*[0-9.]+s/ &&
          line ~ /learning[[:space:]:]+[0-9.]+s/) {
        env_steps_per_s = number_after(line, "^.*Computation:[[:space:]]*", "[[:space:]]+steps/.*$")
        collection_s = number_after(line, "^.*collection:[[:space:]]*", "s,.*$")
        learning_s = number_after(line, "^.*learning[[:space:]:]*", "s.*$")
        have_computation = 1
        save_complete()
        next
      }
      if (line ~ /Total timesteps:[[:space:]]*[0-9]+/) {
        total_timesteps = number_after(line, "^.*Total timesteps:[[:space:]]*", "[[:space:]]*$")
        have_total = 1
        save_complete()
        next
      }
      if (line ~ /Iteration time:[[:space:]]*[0-9.]+s/) {
        update_s = number_after(line, "^.*Iteration time:[[:space:]]*", "s[[:space:]]*$")
        have_update = 1
        save_complete()
      }
    }
    END {
      if (!found) {
        exit 1
      }
      if (!emit_all) {
        printf "%s\t%s\t%s\t%s\t%s\t%s\n", latest_iteration,
          latest_update, latest_collection, latest_learning, latest_rate,
          latest_total
      }
    }
  ' "$1"
}

# Direct parser hook for the patrol unit tests.  It performs no writes.
if [ "${1:-}" = "_parse-rsl-log" ]; then
  [ "$#" -eq 2 ] || exit 2
  parse_rsl_iteration_block "$2"
  exit $?
fi
if [ "${1:-}" = "_parse-rsl-log-all" ]; then
  [ "$#" -eq 2 ] || exit 2
  parse_rsl_iteration_block "$2" 1
  exit $?
fi

usage() {
  cat <<'EOF'
Usage: pod_patrol.sh [--full] [--spec FILE]
                     [--pod NAME=USER@HOST:PORT]
                     [--root NAME=/absolute/run/root]
                     [--namespace NAME=/absolute/run/namespace]
                     [--checkout NAME=/absolute/checkout]
                     [--state-dir DIR]
                     [--stale-after SECONDS]
                     [--startup-stale-after SECONDS]

--pod, --root, --namespace, and --checkout may be repeated.  CLI values extend
the spec; a CLI --pod for the same NAME overrides the spec endpoint.
EOF
}

FULL=0
SPEC=${POD_PATROL_SPEC:-}
STATE_DIR=${POD_PATROL_STATE_DIR:-"/tmp/pod_patrol_state_v2_$(id -u)"}
STALE_AFTER=${POD_PATROL_STALE_AFTER:-900}
STARTUP_STALE_AFTER=${POD_PATROL_STARTUP_STALE_AFTER:-1800}
SSH_BIN=${POD_PATROL_SSH_BIN:-ssh}
SSH_KEY=${POD_PATROL_SSH_KEY:-"$HOME/.ssh/id_ed25519_runpod"}
NOW_EPOCH=${POD_PATROL_NOW_EPOCH:-}

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/pod_patrol.XXXXXX") || exit 2
STATE_LOCK=
cleanup() {
  if [ -n "$STATE_LOCK" ] && [ -d "$STATE_LOCK" ]; then
    rm -f "$STATE_LOCK/owner_pid"
    rmdir "$STATE_LOCK" 2>/dev/null || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT HUP INT TERM
POD_ARGS="$TMP_ROOT/pods"
ROOT_ARGS="$TMP_ROOT/roots"
NAMESPACE_ARGS="$TMP_ROOT/namespaces"
CHECKOUT_ARGS="$TMP_ROOT/checkouts"
: >"$POD_ARGS"
: >"$ROOT_ARGS"
: >"$NAMESPACE_ARGS"
: >"$CHECKOUT_ARGS"

require_mapping() {
  case "$2" in
    *=*) ;;
    *)
      echo "ERROR $1 requires NAME=VALUE, got: $2" >&2
      exit 2
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --full)
      FULL=1
      shift
      ;;
    --spec|--pod|--root|--namespace|--checkout|--state-dir|--stale-after|--startup-stale-after)
      if [ "$#" -lt 2 ]; then
        echo "ERROR missing value for $1" >&2
        usage >&2
        exit 2
      fi
      option=$1
      value=$2
      shift 2
      case "$option" in
        --spec) SPEC=$value ;;
        --pod)
          require_mapping "$option" "$value"
          printf '%s\n' "$value" >>"$POD_ARGS"
          ;;
        --root)
          require_mapping "$option" "$value"
          printf '%s\n' "$value" >>"$ROOT_ARGS"
          ;;
        --namespace)
          require_mapping "$option" "$value"
          printf '%s\n' "$value" >>"$NAMESPACE_ARGS"
          ;;
        --checkout)
          require_mapping "$option" "$value"
          printf '%s\n' "$value" >>"$CHECKOUT_ARGS"
          ;;
        --state-dir) STATE_DIR=$value ;;
        --stale-after) STALE_AFTER=$value ;;
        --startup-stale-after) STARTUP_STALE_AFTER=$value ;;
      esac
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$STALE_AFTER:$STARTUP_STALE_AFTER" in
  *[!0-9:]*|:*|*:) echo "ERROR stale thresholds must be positive integers" >&2; exit 2 ;;
esac
if [ "$STALE_AFTER" -le 0 ] || [ "$STARTUP_STALE_AFTER" -le 0 ]; then
  echo "ERROR stale thresholds must be positive integers" >&2
  exit 2
fi
if [ -z "$NOW_EPOCH" ]; then
  NOW_EPOCH=$(date +%s)
fi
case "$NOW_EPOCH" in *[!0-9]*|'') echo "ERROR POD_PATROL_NOW_EPOCH must be an integer" >&2; exit 2 ;; esac

if [ -n "$SPEC" ] && [ ! -f "$SPEC" ]; then
  echo "ERROR patrol spec does not exist: $SPEC" >&2
  exit 2
fi

TARGETS="$TMP_ROOT/targets.tsv"
python3 - "$SPEC" "$POD_ARGS" "$ROOT_ARGS" "$NAMESPACE_ARGS" "$CHECKOUT_ARGS" >"$TARGETS" <<'PY'
import json
import pathlib
import re
import sys

spec_path, pod_path, root_path, namespace_path, checkout_path = sys.argv[1:]
targets = {}


def target(name):
    name = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or ".." in name:
        raise SystemExit(f"invalid pod name: {name!r}")
    return targets.setdefault(
        name,
        {
            "endpoint": "",
            "roots": [],
            "namespaces": [],
            "expected_namespaces": [],
            "checkouts": [],
        },
    )


def append_unique(items, value, label):
    if value is None:
        return
    value = str(value)
    if any(c in value for c in "\t\r\n\x1f*?[]"):
        raise SystemExit(f"invalid {label}: control/glob character")
    path = pathlib.PurePosixPath(value)
    if not value.startswith("/") or ".." in path.parts:
        raise SystemExit(f"{label} must be an absolute normalized path: {value!r}")
    if value not in items:
        items.append(value)


def endpoint_text(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    if isinstance(value.get("ssh_endpoint"), str):
        return value["ssh_endpoint"]
    host = value.get("host")
    port = value.get("port", value.get("ssh_port"))
    user = value.get("user", "root")
    if host and port is not None:
        return f"{user}@{host}:{port}"
    return ""


def add_endpoint_mapping(mapping):
    if not isinstance(mapping, dict):
        return
    for name, value in mapping.items():
        text = endpoint_text(value)
        if text:
            target(name)["endpoint"] = text


def add_run(name, run):
    if not isinstance(run, dict):
        return
    entry = target(name)
    namespace = run.get("namespace")
    if namespace:
        append_unique(entry["namespaces"], namespace, "namespace")
    log_path = run.get("log_path")
    if log_path:
        append_unique(
            entry["namespaces"],
            str(pathlib.PurePosixPath(str(log_path)).parent),
            "log namespace",
        )
    status = str(run.get("status", "")).lower()
    active_status = status in {"running", "booted", "active"} or status.startswith(
        ("running_", "booted_", "active_")
    )
    if namespace and active_status:
        append_unique(
            entry["expected_namespaces"], namespace, "expected namespace"
        )


if spec_path:
    with open(spec_path, "rb") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise SystemExit("patrol spec must be a JSON object")

    patrol = document.get("patrol", {})
    if isinstance(patrol, dict):
        add_endpoint_mapping(patrol.get("endpoints"))
        for row in patrol.get("targets", []):
            if not isinstance(row, dict):
                continue
            name = row.get("pod_id", row.get("name"))
            if not name:
                continue
            entry = target(name)
            text = endpoint_text(row)
            if text:
                entry["endpoint"] = text
            for value in row.get("roots", []):
                append_unique(entry["roots"], value, "root")
            for value in row.get("namespaces", []):
                append_unique(entry["namespaces"], value, "namespace")
                append_unique(
                    entry["expected_namespaces"], value, "expected namespace"
                )
            for value in row.get("checkouts", []):
                append_unique(entry["checkouts"], value, "checkout")

    add_endpoint_mapping(document.get("pod_endpoints"))
    source = document.get("source", {})
    if isinstance(source, dict):
        add_endpoint_mapping(source.get("pod_endpoints"))

    pods = document.get("pods")
    if isinstance(pods, dict):
        add_endpoint_mapping(pods)
    elif isinstance(pods, list):
        for row in pods:
            if isinstance(row, dict) and row.get("name"):
                text = endpoint_text(row)
                if text:
                    target(row["name"])["endpoint"] = text

    matrix = document.get("matrix", [])
    if isinstance(matrix, list):
        for row in matrix:
            if not isinstance(row, dict) or not row.get("pod_id"):
                continue
            name = row["pod_id"]
            entry = target(name)
            text = endpoint_text(row)
            if text:
                entry["endpoint"] = text
            append_unique(entry["roots"], row.get("run_root"), "run root")
            append_unique(entry["checkouts"], row.get("checkout"), "checkout")
            for key, value in row.items():
                if key.endswith("_runs") and isinstance(value, list):
                    for run in value:
                        add_run(name, run)

    # One launcher spec can also be patrolled when --pod supplies its identity.
    launch = document.get("spec", document)
    if isinstance(launch, dict) and launch.get("namespace"):
        launch_name = launch.get("pod_id")
        if not launch_name and len(targets) == 1:
            launch_name = next(iter(targets))
        if launch_name:
            add_run(launch_name, launch)
            append_unique(
                target(launch_name)["expected_namespaces"],
                launch.get("namespace"),
                "expected namespace",
            )
            launch_source = launch.get("source", {})
            if isinstance(launch_source, dict):
                append_unique(
                    target(launch_name)["checkouts"],
                    launch_source.get("checkout"),
                    "checkout",
                )


def read_mappings(path):
    result = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            name, value = raw.split("=", 1)
            result.append((name, value))
    return result


cli_pods = read_mappings(pod_path)
for name, value in cli_pods:
    target(name)["endpoint"] = value
for name, value in read_mappings(root_path):
    append_unique(target(name)["roots"], value, "root")
for name, value in read_mappings(namespace_path):
    append_unique(target(name)["namespaces"], value, "namespace")
    append_unique(target(name)["expected_namespaces"], value, "expected namespace")
for name, value in read_mappings(checkout_path):
    append_unique(target(name)["checkouts"], value, "checkout")

endpoint_re = re.compile(
    r"^([A-Za-z0-9_.-]+@)?(\[[0-9A-Fa-f:]+\]|[A-Za-z0-9][A-Za-z0-9.-]*):([0-9]+)$"
)
selected_names = {name for name, _value in cli_pods}
if selected_names:
    targets = {name: targets[name] for name in sorted(selected_names)}

for name in sorted(targets):
    entry = targets[name]
    endpoint = entry["endpoint"]
    host = ""
    port = ""
    if endpoint:
        match = endpoint_re.fullmatch(endpoint)
        if not match:
            raise SystemExit(
                f"endpoint for {name} must be USER@HOST:PORT: {endpoint!r}"
            )
        host = f"{match.group(1) or ''}{match.group(2)}"
        port = match.group(3)
        if host.startswith("-") or (match.group(1) or "").startswith("-"):
            raise SystemExit(f"unsafe SSH endpoint for {name}: {endpoint!r}")
        if not 1 <= int(port) <= 65535:
            raise SystemExit(f"invalid SSH port for {name}: {port}")
    fields = [
        name,
        host,
        port,
        "\x1f".join(entry["roots"]),
        "\x1f".join(entry["namespaces"]),
        "\x1f".join(entry["checkouts"]),
        "\x1f".join(entry["expected_namespaces"]),
    ]
    print("\x1e".join(fields))
PY
parser_status=$?
if [ "$parser_status" -ne 0 ]; then
  exit "$parser_status"
fi
if [ ! -s "$TARGETS" ]; then
  echo "WARN patrol has no bound pod target; pass --spec or --pod" >&2
  exit 1
fi

umask 077
if [ -L "$STATE_DIR" ]; then
  echo "ERROR local patrol state dir must not be a symlink: $STATE_DIR" >&2
  exit 2
fi
state_dir_was_present=0
if [ -d "$STATE_DIR" ]; then
  state_dir_was_present=1
fi
mkdir -p "$STATE_DIR" || {
  echo "ERROR cannot create local patrol state dir: $STATE_DIR" >&2
  exit 2
}
state_owner=$(stat -f %u "$STATE_DIR" 2>/dev/null || stat -c %u "$STATE_DIR" 2>/dev/null)
if [ -z "$state_owner" ] || [ "$state_owner" != "$(id -u)" ]; then
  echo "ERROR local patrol state dir is not owned by this user: $STATE_DIR" >&2
  exit 2
fi
if [ "$state_dir_was_present" -eq 0 ]; then
  chmod 700 "$STATE_DIR" || exit 2
else
  state_mode=$(stat -f %Lp "$STATE_DIR" 2>/dev/null || stat -c %a "$STATE_DIR" 2>/dev/null)
  if [ "$state_mode" != "700" ]; then
    echo "ERROR existing patrol state dir must already be mode 700: $STATE_DIR" >&2
    exit 2
  fi
fi
STATE_LOCK="$STATE_DIR/.pod_patrol.lock"
if ! mkdir "$STATE_LOCK" 2>/dev/null; then
  lock_owner=$(cat "$STATE_LOCK/owner_pid" 2>/dev/null || echo "")
  case "$lock_owner" in
    *[!0-9]*|'') lock_alive=1 ;;
    *)
      if ps -p "$lock_owner" -o pid= >/dev/null 2>&1; then
        lock_alive=1
      else
        lock_alive=0
      fi
      ;;
  esac
  if [ "$lock_alive" -eq 0 ]; then
    rm -f "$STATE_LOCK/owner_pid"
    rmdir "$STATE_LOCK" 2>/dev/null || true
  fi
  if ! mkdir "$STATE_LOCK" 2>/dev/null; then
    echo "WARN another patrol owns local cursor lock: $STATE_LOCK owner_pid=${lock_owner:-?}" >&2
    exit 1
  fi
fi
printf '%s\n' "$$" >"$STATE_LOCK/owner_pid"

REMOTE_FILE="$TMP_ROOT/remote.sh"
declare -f parse_rsl_iteration_block >"$REMOTE_FILE"
cat >>"$REMOTE_FILE" <<'REMOTE'
set -u
POD=$1
FULL=$2
ROOT_TEXT=$3
NAMESPACE_TEXT=$4
CHECKOUT_TEXT=$5
EXPECTED_TEXT=$6
[ "$ROOT_TEXT" = "-" ] && ROOT_TEXT=
[ "$NAMESPACE_TEXT" = "-" ] && NAMESPACE_TEXT=
[ "$CHECKOUT_TEXT" = "-" ] && CHECKOUT_TEXT=
[ "$EXPECTED_TEXT" = "-" ] && EXPECTED_TEXT=
PROC_ROOT=${POD_PATROL_PROC_ROOT:-/proc}
UNIT_SEP=$(printf '\037')
old_ifs=$IFS
IFS=$UNIT_SEP
set -f
ROOTS=$ROOT_TEXT
NAMESPACES=$NAMESPACE_TEXT
CHECKOUTS=$CHECKOUT_TEXT
EXPECTED_NAMESPACES=$EXPECTED_TEXT
IFS=$old_ifs

emit_gpu_rows() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'REMOTE_WARN\t%s\tnvidia-smi unavailable\n' "$POD"
    return
  fi
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null |
  while IFS=, read -r idx util mem; do
    idx=$(printf '%s' "$idx" | tr -dc '0-9')
    util=$(printf '%s' "$util" | tr -dc '0-9')
    mem=$(printf '%s' "$mem" | tr -dc '0-9')
    printf 'GPU\t%s\t%s\t%s\t%s\n' "$POD" "${idx:-?}" "${util:-0}" "${mem:-0}"
  done
}

namespace_leader_matches_pid() {
  namespace=$1
  wanted_pid=$2
  sidecar="$namespace/run.log.launch.leader.json"
  [ -f "$sidecar" ] || return 1
  leader_pid=$(grep -oE '"pid":[[:space:]]*[0-9]+' "$sidecar" |
    head -1 | grep -oE '[0-9]+$')
  leader_start=$(grep -oE '"starttime_ticks":[[:space:]]*[0-9]+' "$sidecar" |
    head -1 | grep -oE '[0-9]+$')
  [ "$leader_pid" = "$wanted_pid" ] || return 1
  [ -r "$PROC_ROOT/$leader_pid/stat" ] || return 1
  actual_start=$(awk '{print $22}' "$PROC_ROOT/$leader_pid/stat" 2>/dev/null)
  [ -n "$leader_start" ] && [ "$actual_start" = "$leader_start" ]
}

find_namespace() {
  wanted=$1
  wanted_pid=$2
  old_ifs=$IFS
  IFS=$UNIT_SEP
  # Prefer the namespace whose immutable leader sidecar binds this exact PID
  # and Linux starttime.  This prevents a same-run_name process elsewhere from
  # satisfying the wrong expected namespace.
  for candidate in $NAMESPACES; do
    [ -n "$candidate" ] || continue
    if [ "${candidate##*/}" = "$wanted" ] &&
       [ -f "$candidate/run.log" ] &&
       namespace_leader_matches_pid "$candidate" "$wanted_pid"; then
      printf '%s\n' "$candidate"
      IFS=$old_ifs
      return
    fi
  done
  for root in $ROOTS; do
    [ -n "$root" ] || continue
    if [ -f "$root/$wanted/run.log" ] &&
       namespace_leader_matches_pid "$root/$wanted" "$wanted_pid"; then
      printf '%s\n' "$root/$wanted"
      IFS=$old_ifs
      return
    fi
  done
  # Legacy namespaces without a leader sidecar remain searchable, but a
  # present mismatching sidecar is never bypassed.
  for candidate in $NAMESPACES; do
    [ -n "$candidate" ] || continue
    if [ "${candidate##*/}" = "$wanted" ] &&
       [ -f "$candidate/run.log" ] &&
       [ ! -e "$candidate/run.log.launch.leader.json" ]; then
      printf '%s\n' "$candidate"
      IFS=$old_ifs
      return
    fi
  done
  for root in $ROOTS; do
    [ -n "$root" ] || continue
    if [ -f "$root/$wanted/run.log" ] &&
       [ ! -e "$root/$wanted/run.log.launch.leader.json" ]; then
      printf '%s\n' "$root/$wanted"
      IFS=$old_ifs
      return
    fi
  done
  IFS=$old_ifs
}

find_rsl_run() {
  wanted=$1
  old_ifs=$IFS
  IFS=$UNIT_SEP
  for checkout in $CHECKOUTS; do
    [ -n "$checkout" ] || continue
    experiment_root="$checkout/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_action_ball_n1_reward_screen_diagnostic"
    [ -d "$experiment_root" ] || continue
    set +f
    for run_dir in "$experiment_root"/*_"$wanted"-DIAGNOSTIC_UNAUTHORIZED; do
      [ -d "$run_dir" ] || continue
      printf '%s\n' "$run_dir"
      set -f
      IFS=$old_ifs
      return
    done
    set -f
  done
  IFS=$old_ifs
}

checkpoint_progress() {
  run_dir=$1
  best=-1
  best_mtime=0
  if [ -n "$run_dir" ]; then
    set +f
    for checkpoint in "$run_dir"/model_*.pt; do
      [ -f "$checkpoint" ] || continue
      number=${checkpoint##*/model_}
      number=${number%.pt}
      case "$number" in
        *[!0-9]*|"") continue ;;
      esac
      if [ "$number" -gt "$best" ]; then
        best=$number
        best_mtime=$(stat -c %Y "$checkpoint" 2>/dev/null || echo 0)
      fi
    done
    set -f
  fi
  printf '%s\t%s\n' "$best" "$best_mtime"
}

rollout_steps() {
  namespace=$1
  run_dir=$2
  # A namespace receipt/spec is authoritative over generated RSL-RL params.
  for file in "$namespace/launch_claim.json" "$namespace/launch_spec.json" \
    "$namespace/run_binding.json"; do
    [ -f "$file" ] || continue
    value=$(grep -oE '"num_steps_per_env"[[:space:]]*:[[:space:]]*[0-9]+' \
      "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
    if [ -n "$value" ] && [ "$value" -gt 0 ]; then
      printf '%s\n' "$value"
      return
    fi
  done
  if [ -n "$run_dir" ]; then
    file="$run_dir/params/training_contract.json"
    if [ -f "$file" ]; then
      value=$(grep -oE '"num_steps_per_env"[[:space:]]*:[[:space:]]*[0-9]+' \
        "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
      if [ -n "$value" ] && [ "$value" -gt 0 ]; then
        printf '%s\n' "$value"
        return
      fi
    fi
    file="$run_dir/params/agent.yaml"
    if [ -f "$file" ]; then
      value=$(grep -E '^[[:space:]]*num_steps_per_env:[[:space:]]*[0-9]+' \
        "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
      if [ -n "$value" ] && [ "$value" -gt 0 ]; then
        printf '%s\n' "$value"
        return
      fi
    fi
  fi
  printf '?\n'
}

rollout_num_envs() {
  namespace=$1
  run_dir=$2
  process_args=$3
  # Prefer the same namespace-bound JSON receipts/specs used for rollout
  # length.  The process argv is only a final fallback for older diagnostic
  # namespaces that predate those receipts.
  for file in "$namespace/launch_claim.json" "$namespace/launch_spec.json" \
    "$namespace/run_binding.json"; do
    [ -f "$file" ] || continue
    value=$(grep -oE '"num_envs"[[:space:]]*:[[:space:]]*[0-9]+' \
      "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
    if [ -n "$value" ] && [ "$value" -gt 0 ]; then
      printf '%s\n' "$value"
      return
    fi
  done
  if [ -n "$run_dir" ]; then
    file="$run_dir/params/training_contract.json"
    if [ -f "$file" ]; then
      value=$(grep -oE '"num_envs"[[:space:]]*:[[:space:]]*[0-9]+' \
        "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
      if [ -n "$value" ] && [ "$value" -gt 0 ]; then
        printf '%s\n' "$value"
        return
      fi
    fi
    for file in "$run_dir/params/env.yaml" "$run_dir/params/agent.yaml"; do
      [ -f "$file" ] || continue
      value=$(grep -E '^[[:space:]]*num_envs:[[:space:]]*[0-9]+' \
        "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
      if [ -n "$value" ] && [ "$value" -gt 0 ]; then
        printf '%s\n' "$value"
        return
      fi
    done
  fi
  value=$(printf '%s\n' "$process_args" |
    sed -n 's/.*[[:space:]]num_envs=\([0-9][0-9]*\)\([[:space:]].*\)\{0,1\}$/\1/p')
  if [ -n "$value" ] && [ "$value" -gt 0 ]; then
    printf '%s\n' "$value"
    return
  fi
  printf '?\n'
}

emit_gpu_rows
ps -eo pid=,etimes=,args= 2>/dev/null |
while read -r pid elapsed rest; do
  case "$rest" in *scripts/train.py*) ;; *) continue ;; esac
  run_name=$(printf '%s\n' "$rest" |
    sed -n 's/.*[[:space:]]run_name=\([^[:space:]]*\).*/\1/p')
  [ -n "$run_name" ] || {
    printf 'REMOTE_WARN\t%s\ttrainer pid=%s has no run_name argument\n' "$POD" "$pid"
    continue
  }
  case "$run_name" in
    "."|".."|*/*|*[!A-Za-z0-9_.-]*)
      printf 'REMOTE_WARN\t%s\ttrainer pid=%s has unsafe run_name\n' "$POD" "$pid"
      continue
      ;;
  esac
  namespace=$(find_namespace "$run_name" "$pid")
  if [ -z "$namespace" ]; then
    printf 'REMOTE_WARN\t%s\t%s pid=%s is outside bound roots/namespaces\n' \
      "$POD" "$run_name" "$pid"
    continue
  fi
  log="$namespace/run.log"
  iteration=$(grep -oE 'Learning iteration [0-9]+/[0-9]+' "$log" 2>/dev/null |
    tail -1 | sed -E 's/.* ([0-9]+)\/[0-9]+/\1/')
  [ -n "$iteration" ] || iteration=-1
  rsl_run=$(find_rsl_run "$run_name")
  checkpoint_row=$(checkpoint_progress "$rsl_run")
  checkpoint=$(printf '%s\n' "$checkpoint_row" | cut -f1)
  checkpoint_mtime=$(printf '%s\n' "$checkpoint_row" | cut -f2)
  timing_iteration=?
  update_s=?
  collection_s=?
  learning_s=?
  env_steps_per_s=?
  total_timesteps=?
  timing_rows=$(parse_rsl_iteration_block "$log" 1 2>/dev/null || true)
  timing_row=$(printf '%s\n' "$timing_rows" | tail -1)
  if [ -n "$timing_row" ]; then
    timing_iteration=$(printf '%s\n' "$timing_row" | cut -f1)
    update_s=$(printf '%s\n' "$timing_row" | cut -f2)
    collection_s=$(printf '%s\n' "$timing_row" | cut -f3)
    learning_s=$(printf '%s\n' "$timing_row" | cut -f4)
    env_steps_per_s=$(printf '%s\n' "$timing_row" | cut -f5)
    total_timesteps=$(printf '%s\n' "$timing_row" | cut -f6)
  fi
  num_steps_per_env=$(rollout_steps "$namespace" "$rsl_run")
  num_envs=$(rollout_num_envs "$namespace" "$rsl_run" "$rest")
  json_row=$(grep -F 'HOPE_EXACT_BEHAVIOR_UPDATE_JSON' "$log" 2>/dev/null | tail -1)
  legal=$(printf '%s\n' "$json_row" |
    grep -oE '"virtual_legal_return_per_strike":[0-9.null]+' |
    head -1 | cut -d: -f2)
  fall=$(printf '%s\n' "$json_row" |
    grep -oE '"post_strike_physical_fall_rate":[0-9.null]+' |
    head -1 | cut -d: -f2)
  if [ -n "$timing_rows" ]; then
    printf '%s\n' "$timing_rows" |
    while IFS="$(printf '\t')" read -r sample_iteration sample_update \
      sample_collection sample_learning sample_rate sample_total; do
      printf 'TIMING\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$POD" "$run_name" "$sample_iteration" "$sample_update" \
        "$sample_collection" "$sample_learning" "$sample_rate" \
        "$sample_total" "$num_steps_per_env" "$num_envs"
    done
  fi
  printf 'RUN\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$POD" "$pid" "$elapsed" "$run_name" "$namespace" "$iteration" \
    "$checkpoint" "$checkpoint_mtime" "${legal:-?}" "${fall:-?}" "$log" \
    "$timing_iteration" "$update_s" "$collection_s" "$learning_s" \
    "$env_steps_per_s" "$total_timesteps" "$num_steps_per_env" "$num_envs"
done

expected_namespace_alive() {
  namespace=$1
  run_name=${namespace##*/}
  sidecar="$namespace/run.log.launch.leader.json"
  [ -f "$sidecar" ] || return 1
  leader_pid=$(grep -oE '"pid":[[:space:]]*[0-9]+' "$sidecar" |
    head -1 | grep -oE '[0-9]+$')
  namespace_leader_matches_pid "$namespace" "$leader_pid" || return 1
  [ -r "$PROC_ROOT/$leader_pid/cmdline" ] || return 1
  tr '\000' '\n' <"$PROC_ROOT/$leader_pid/cmdline" |
    grep -Fxq "run_name=$run_name"
}

old_ifs=$IFS
IFS=$UNIT_SEP
for namespace in $EXPECTED_NAMESPACES; do
  [ -n "$namespace" ] || continue
  if ! expected_namespace_alive "$namespace"; then
    state=MISSING
    if [ -f "$namespace/run.log" ] &&
       grep -qE 'Learning iteration [0-9]+/[0-9]+' "$namespace/run.log"; then
      state=EXITED
    fi
    printf 'REMOTE_WARN\t%s\texpected namespace %s trainer %s\n' \
      "$POD" "$namespace" "$state"
  fi
done
IFS=$old_ifs
REMOTE

status=0
TAB=$(printf '\t')
FIELD_SEP=$(printf '\036')
while IFS="$FIELD_SEP" read -r pod ssh_host ssh_port roots namespaces checkouts \
  expected_namespaces; do
  [ -n "$pod" ] || continue
  echo "== $pod $(date -u +%H:%MZ) =="
  if [ -z "$ssh_host" ] || [ -z "$ssh_port" ]; then
    echo "WARN $pod has no endpoint in wave spec/--pod; patrol did not connect"
    status=1
    continue
  fi

  ssh_error="$TMP_ROOT/${pod}.ssh_error"
  [ -n "$roots" ] || roots=-
  [ -n "$namespaces" ] || namespaces=-
  [ -n "$checkouts" ] || checkouts=-
  [ -n "$expected_namespaces" ] || expected_namespaces=-
  if ! snapshot=$(
    "$SSH_BIN" -o BatchMode=yes -o ConnectTimeout=10 -i "$SSH_KEY" \
      -p "$ssh_port" "$ssh_host" bash -s -- \
      "$pod" "$FULL" "$roots" "$namespaces" "$checkouts" \
      "$expected_namespaces" \
      2>"$ssh_error" <"$REMOTE_FILE"
  ); then
    detail=$(tail -1 "$ssh_error" 2>/dev/null)
    echo "WARN $pod unreachable endpoint=$ssh_host:$ssh_port${detail:+ ($detail)}"
    status=1
    continue
  fi

  printf '%s\n' "$snapshot" |
  while IFS="$TAB" read -r kind f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 \
    f12 f13 f14 f15 f16 f17 f18 f19; do
    case "$kind" in
      GPU)
        if [ "$FULL" -eq 1 ]; then
          echo "GPU  $f1 gpu$f2 util=${f3}% memory=${f4}MiB"
        fi
        ;;
      REMOTE_WARN)
        echo "WARN $f1 $f2"
        ;;
      TIMING)
        sample_pod=$f1
        sample_run=$f2
        sample_iteration=$f3
        sample_update=$f4
        sample_collection=$f5
        sample_learning=$f6
        sample_rate=$f7
        sample_total=$f8
        sample_num_steps=${f9:-?}
        sample_num_envs=${f10:-?}
        if [ "$sample_pod" != "$pod" ]; then
          echo "WARN $pod rejected timing row for unexpected pod=$sample_pod"
          continue
        fi
        case "$sample_run:$sample_iteration" in
          *[!A-Za-z0-9_.:-]*|*:|*:*[!0-9]*)
            echo "WARN $pod rejected malformed timing identity"
            continue
            ;;
        esac
        sample_vector_step=?
        case "$sample_num_steps:$sample_update" in
          *[!0-9.:]*|\?:*|*:\?) ;;
          *)
            sample_vector_step=$(awk \
              -v update="$sample_update" -v steps="$sample_num_steps" \
              'BEGIN { if (steps > 0) printf "%.4f", update / steps; else printf "?" }')
            ;;
        esac
        sample_collection_vector_step_wall_s=?
        case "$sample_num_steps:$sample_collection" in
          *[!0-9.:]*|\?:*|*:\?) ;;
          *)
            sample_collection_vector_step_wall_s=$(awk \
              -v collection="$sample_collection" -v steps="$sample_num_steps" \
              'BEGIN { if (steps > 0) printf "%.6f", collection / steps; else printf "?" }')
            ;;
        esac
        sample_amortized_e2e_vector_step_wall_s=?
        case "$sample_num_steps:$sample_update" in
          *[!0-9.:]*|\?:*|*:\?) ;;
          *)
            sample_amortized_e2e_vector_step_wall_s=$(awk \
              -v update="$sample_update" -v steps="$sample_num_steps" \
              'BEGIN { if (steps > 0) printf "%.6f", update / steps; else printf "?" }')
            ;;
        esac
        sample_collection_environment_step_us=?
        sample_collection_environment_steps_per_s=?
        case "$sample_num_envs:$sample_num_steps:$sample_collection" in
          *[!0-9.:]*|\?:*|*:\?*|*:*:\?) ;;
          *)
            sample_collection_environment_step_us=$(awk \
              -v collection="$sample_collection" -v envs="$sample_num_envs" \
              -v steps="$sample_num_steps" \
              'BEGIN { count = envs * steps; if (count > 0 && collection > 0) printf "%.3f", collection / count * 1000000; else printf "?" }')
            sample_collection_environment_steps_per_s=$(awk \
              -v collection="$sample_collection" -v envs="$sample_num_envs" \
              -v steps="$sample_num_steps" \
              'BEGIN { count = envs * steps; if (count > 0 && collection > 0) printf "%.3f", count / collection; else printf "?" }')
            ;;
        esac
        sample_key=$(printf '%s' "$sample_pod|$sample_run" | cksum | awk '{print $1}')
        timing_file="$STATE_DIR/v2_$sample_key.timing.tsv"
        timing_identity="# identity pod=$sample_pod run=$sample_run"
        if [ -L "$timing_file" ]; then
          echo "WARN $sample_pod/$sample_run timing cursor is a symlink; refusing"
          continue
        fi
        last_sample=-1
        if [ -f "$timing_file" ]; then
          actual_identity=$(head -1 "$timing_file")
          if [ "$actual_identity" != "$timing_identity" ]; then
            echo "WARN $sample_pod/$sample_run timing cursor identity mismatch"
            continue
          fi
          last_sample=$(tail -1 "$timing_file" | cut -f2)
          case "$last_sample" in *[!0-9]*|'') last_sample=-1 ;; esac
        fi
        if [ "$sample_iteration" -gt "$last_sample" ]; then
          timing_tmp="$timing_file.tmp.$$"
          if [ -f "$timing_file" ]; then
            cp "$timing_file" "$timing_tmp"
          else
            printf '%s\n' "$timing_identity" >"$timing_tmp"
            printf 'observed_epoch\titeration\tupdate_s\tcollection_s\tlearning_s\tenv_steps_per_s\ttotal_timesteps\tnum_steps_per_env\tvector_policy_step_s\tnum_envs\tcollection_vector_step_wall_s\tamortized_e2e_vector_step_wall_s\tcollection_environment_step_us\tcollection_environment_steps_per_s\tenv_steps_per_s_compat\tvector_policy_step_s_compat\n' \
              >>"$timing_tmp"
          fi
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$NOW_EPOCH" "$sample_iteration" "$sample_update" \
            "$sample_collection" "$sample_learning" "$sample_rate" \
            "$sample_total" "$sample_num_steps" "$sample_vector_step" \
            "$sample_num_envs" "$sample_collection_vector_step_wall_s" \
            "$sample_amortized_e2e_vector_step_wall_s" \
            "$sample_collection_environment_step_us" \
            "$sample_collection_environment_steps_per_s" \
            "legacy_rsl_reported_rate" \
            "legacy_alias_of_amortized_e2e_vector_step_wall_s" \
            >>"$timing_tmp"
          mv "$timing_tmp" "$timing_file"
        fi
        ;;
      RUN)
        run_pod=$f1
        pid=$f2
        elapsed=$f3
        run_name=$f4
        namespace=$f5
        iteration=$f6
        checkpoint=$f7
        checkpoint_mtime=$f8
        legal=$f9
        fall=$f10
        log=$f11
        timing_iteration=${f12:-?}
        update_s=${f13:-?}
        collection_s=${f14:-?}
        learning_s=${f15:-?}
        env_steps_per_s=${f16:-?}
        total_timesteps=${f17:-?}
        num_steps_per_env=${f18:-?}
        num_envs=${f19:-?}
        if [ "$run_pod" != "$pod" ]; then
          echo "WARN $pod rejected run row for unexpected pod=$run_pod"
          continue
        fi
        case "$run_name" in
          "."|".."|*/*|*[!A-Za-z0-9_.-]*)
            echo "WARN $pod rejected unsafe run_name from remote"
            continue
            ;;
        esac
        case "$pid:$elapsed:$iteration:$checkpoint:$checkpoint_mtime" in
          *[!0-9:-]*|:*|*::*)
            echo "WARN $pod/$run_name rejected malformed numeric progress row"
            continue
            ;;
        esac
        vector_policy_step_s=?
        case "$num_steps_per_env:$update_s" in
          *[!0-9.:]*|\?:*|*:\?)
            ;;
          *)
            vector_policy_step_s=$(awk \
              -v update="$update_s" -v steps="$num_steps_per_env" \
              'BEGIN { if (steps > 0) printf "%.4f", update / steps; else printf "?" }')
            ;;
        esac
        collection_vector_step_wall_s=?
        case "$num_steps_per_env:$collection_s" in
          *[!0-9.:]*|\?:*|*:\?) ;;
          *)
            collection_vector_step_wall_s=$(awk \
              -v collection="$collection_s" -v steps="$num_steps_per_env" \
              'BEGIN { if (steps > 0) printf "%.6f", collection / steps; else printf "?" }')
            ;;
        esac
        amortized_e2e_vector_step_wall_s=?
        case "$num_steps_per_env:$update_s" in
          *[!0-9.:]*|\?:*|*:\?) ;;
          *)
            amortized_e2e_vector_step_wall_s=$(awk \
              -v update="$update_s" -v steps="$num_steps_per_env" \
              'BEGIN { if (steps > 0) printf "%.6f", update / steps; else printf "?" }')
            ;;
        esac
        collection_environment_step_us=?
        collection_environment_steps_per_s=?
        case "$num_envs:$num_steps_per_env:$collection_s" in
          *[!0-9.:]*|\?:*|*:\?*|*:*:\?) ;;
          *)
            collection_environment_step_us=$(awk \
              -v collection="$collection_s" -v envs="$num_envs" \
              -v steps="$num_steps_per_env" \
              'BEGIN { count = envs * steps; if (count > 0 && collection > 0) printf "%.3f", collection / count * 1000000; else printf "?" }')
            collection_environment_steps_per_s=$(awk \
              -v collection="$collection_s" -v envs="$num_envs" \
              -v steps="$num_steps_per_env" \
              'BEGIN { count = envs * steps; if (count > 0 && collection > 0) printf "%.3f", count / collection; else printf "?" }')
            ;;
        esac
        state_key=$(printf '%s' "$run_pod|$run_name" | cksum | awk '{print $1}')
        state_file="$STATE_DIR/v2_$state_key.state"
        previous_pid=
        previous_iteration=-1
        previous_checkpoint=-1
        progress_epoch=$NOW_EPOCH
        previous_legal=
        previous_timing_iteration=?
        if [ -f "$state_file" ]; then
          if [ -L "$state_file" ]; then
            echo "WARN $run_pod/$run_name progress cursor is a symlink; refusing"
            continue
          fi
          IFS="$TAB" read -r identity_pod identity_run previous_pid \
            previous_iteration previous_checkpoint progress_epoch \
            previous_legal previous_timing_iteration \
            _previous_update _previous_collection _previous_learning \
            _previous_rate _previous_total _previous_num_steps \
            _previous_vector_step _previous_num_envs \
            _previous_collection_vector_step \
            _previous_amortized_e2e_vector_step \
            _previous_collection_environment_step_us \
            _previous_collection_environment_steps_per_s <"$state_file"
          if [ "$identity_pod" != "$run_pod" ] ||
             [ "$identity_run" != "$run_name" ]; then
            echo "WARN $run_pod/$run_name progress cursor identity mismatch"
            continue
          fi
          previous_timing_iteration=${previous_timing_iteration:-?}
        elif [ "$checkpoint" -ge 0 ] && [ "$checkpoint_mtime" -gt 0 ]; then
          progress_epoch=$checkpoint_mtime
        fi

        if [ "$pid" != "$previous_pid" ] ||
           [ "$iteration" -gt "$previous_iteration" ] ||
           [ "$checkpoint" -gt "$previous_checkpoint" ]; then
          progress_epoch=$NOW_EPOCH
        fi
        state_tmp="$state_file.tmp.$$"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$run_pod" "$run_name" "$pid" "$iteration" "$checkpoint" \
          "$progress_epoch" "$legal" \
          "$timing_iteration" "$update_s" "$collection_s" "$learning_s" \
          "$env_steps_per_s" "$total_timesteps" "$num_steps_per_env" \
          "$vector_policy_step_s" "$num_envs" \
          "$collection_vector_step_wall_s" \
          "$amortized_e2e_vector_step_wall_s" \
          "$collection_environment_step_us" \
          "$collection_environment_steps_per_s" >"$state_tmp"
        mv "$state_tmp" "$state_file"

        progress_age=$((NOW_EPOCH - progress_epoch))
        stale=0
        if [ "$iteration" -lt 0 ] && [ "$checkpoint" -lt 0 ]; then
          if [ "$elapsed" -gt "$STARTUP_STALE_AFTER" ]; then
            stale=1
            echo "WARN $run_pod/$run_name pid=$pid has no Learning iteration/checkpoint after ${elapsed}s"
          fi
        elif [ "$progress_age" -gt "$STALE_AFTER" ]; then
          stale=1
          echo "WARN $run_pod/$run_name pid=$pid progress stale ${progress_age}s iteration=$iteration checkpoint=$checkpoint"
        fi

        if [ -n "$previous_legal" ] &&
           [ "$legal" != "?" ] && [ "$legal" != "null" ] &&
           [ "$previous_legal" != "?" ] && [ "$previous_legal" != "null" ]; then
          awk -v p="$previous_legal" -v c="$legal" \
            'BEGIN { exit ! (p > 0.02 && c < p / 2) }' &&
            echo "WARN $run_pod/$run_name legal/strike halved: $previous_legal -> $legal"
        fi

        if [ "$FULL" -eq 1 ] || [ "$stale" -eq 1 ]; then
          echo "OK   $run_pod/$run_name pid=$pid iteration=$iteration checkpoint=$checkpoint update_s=$update_s collection_s=$collection_s learning_s=$learning_s env_steps_per_s=$env_steps_per_s env_steps_per_s_compat=legacy_rsl_reported_rate total_timesteps=$total_timesteps num_steps_per_env=$num_steps_per_env num_envs=$num_envs vector_policy_step_s=$vector_policy_step_s vector_policy_step_s_compat=legacy_alias_of_amortized_e2e_vector_step_wall_s collection_vector_step_wall_s=$collection_vector_step_wall_s amortized_e2e_vector_step_wall_s=$amortized_e2e_vector_step_wall_s collection_environment_step_us=$collection_environment_step_us collection_environment_steps_per_s=$collection_environment_steps_per_s legal/strike=$legal fall=$fall progress_age=${progress_age}s namespace=$namespace log=$log"
        fi
        ;;
      '')
        ;;
      *)
        echo "WARN $pod malformed patrol row: $kind"
        ;;
    esac
  done
done <"$TARGETS"

exit "$status"
