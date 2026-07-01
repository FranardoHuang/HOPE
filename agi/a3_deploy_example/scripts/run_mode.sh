#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${A3_DEPLOY_DIST:-${DEPLOY_ROOT}/dist/a3_deploy_x86_64}"

usage() {
  cat <<'EOF'
Usage:
  ./run_mode.sh A|B|C [warmup_sec] [gain_scale] [runner flags...]
  ./run_mode.sh --print-cmd A|B|C [warmup_sec] [gain_scale] [runner flags...]

Modes:
  A  fabricated localization (reproduce fake world-pose error)
  B  perfect_tracking (current hardware-safe placeholder)
  C  oracle pelvis (sim only; start ./run_oracle.sh first)

Environment:
  A3_DEPLOY_DIST=/abs/path/to/dist/a3_deploy_x86_64
  A3_TRANSPORT=iceoryx|ros2
  A3_RUNNER_START=motion|shadow|passive|pd_stand
  A3_RUN_MODE_AUTO_RESET=0|1

Examples:
  ./run_mode.sh B 10 0.4
  ./run_mode.sh B 10 0.4 --auto-leg-hold --leg-gain-scale 0.5 --ankle-gain-scale 1.0
EOF
}

source_if_exists() {
  local setup_file=$1
  if [[ -f "${setup_file}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup_file}"
    set -u
  fi
}

resolve_runtime_cfg() {
  local runtime_cfg=${A3_RUNTIME_CFG:-}
  local candidate
  if [[ -n "${runtime_cfg}" && -f "${runtime_cfg}" ]]; then
    printf '%s\n' "${runtime_cfg}"
    return 0
  fi
  for candidate in \
    "${DIST_DIR}/config/a3_runtime_config.pingpong.yaml" \
    "${DIST_DIR}/config/a3_runtime_config.yaml"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_aimrt_cfg() {
  local transport=${A3_TRANSPORT:-iceoryx}
  local cfg="${DIST_DIR}/config/a3_aimrt_config.${transport}.yaml"
  if [[ -f "${cfg}" ]]; then
    printf '%s\n' "${cfg}"
    return 0
  fi
  return 1
}

PRINT_CMD=0
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --print-cmd)
    PRINT_CMD=1
    shift
    ;;
esac

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  usage >&2
  exit 2
fi
shift
WARMUP="${1:-10}"
if [[ $# -gt 0 ]]; then
  shift
fi
GAIN="${1:-1.0}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

case "${MODE}" in
  A|a)
    LOC_FLAGS=(--loc-mode fabricated)
    CSV="${A3_OBS_CSV:-/tmp/obs_A_fabricated.csv}"
    ;;
  B|b)
    LOC_FLAGS=(--perfect-tracking)
    CSV="${A3_OBS_CSV:-/tmp/obs_B_perfect.csv}"
    ;;
  C|c)
    LOC_FLAGS=(--oracle-pelvis)
    CSV="${A3_OBS_CSV:-/tmp/obs_C_oracle.csv}"
    if [[ ! -e /dev/shm/pp_oracle_pelvis ]]; then
      echo "[run_mode] mode C requested but /dev/shm/pp_oracle_pelvis is missing; start ./scripts/run_oracle.sh first" >&2
    fi
    ;;
  *)
    echo "mode must be A, B, or C" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DIST_DIR}" ]]; then
  echo "dist dir missing: ${DIST_DIR}" >&2
  echo "build it first with:" >&2
  echo "  bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml" >&2
  exit 66
fi

RUNTIME_CFG="$(resolve_runtime_cfg || true)"
if [[ -z "${RUNTIME_CFG}" ]]; then
  echo "ping-pong runtime config missing under ${DIST_DIR}/config/; rebuild with:" >&2
  echo "  bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml" >&2
  exit 66
fi

RUNNER_WRAPPER="${DIST_DIR}/run_a3_pingpong.sh"
RUNNER_START="${A3_RUNNER_START:-motion}"
AUTO_RESET="${A3_RUN_MODE_AUTO_RESET:-1}"

CMD=()
if [[ -x "${RUNNER_WRAPPER}" ]]; then
  CMD=(
    "${RUNNER_WRAPPER}"
    --start "${RUNNER_START}"
    --warmup-sec "${WARMUP}"
    --gain-scale "${GAIN}"
    --official-stand
    "${LOC_FLAGS[@]}"
    --obs-csv "${CSV}"
    "${EXTRA_ARGS[@]}"
  )
else
  AIMRT_CFG="$(resolve_aimrt_cfg || true)"
  if [[ ! -x "${DIST_DIR}/a3_deploy_onnx_ref_pingpong" ]]; then
    echo "runner missing: ${DIST_DIR}/a3_deploy_onnx_ref_pingpong" >&2
    exit 66
  fi
  if [[ -z "${AIMRT_CFG}" ]]; then
    echo "AimRT config missing under ${DIST_DIR}/config/; rebuild with:" >&2
    echo "  bash scripts/build_a3_deploy_pkg.sh --arch x86_64 --runtime-cfg src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.pingpong.yaml" >&2
    exit 66
  fi
  source_if_exists /opt/ros/jazzy/setup.bash
  if [[ -z "${ROS_DISTRO:-}" ]]; then
    source_if_exists /opt/ros/humble/setup.bash
  fi
  export LD_LIBRARY_PATH="${DIST_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export FASTRTPS_DEFAULT_PROFILES_FILE="${DIST_DIR}/config/fastrtps_profile.xml"
  CMD=(
    "${DIST_DIR}/a3_deploy_onnx_ref_pingpong"
    --runtime-cfg "${RUNTIME_CFG}"
    --aimrt-cfg "${AIMRT_CFG}"
    --start "${RUNNER_START}"
    --warmup-sec "${WARMUP}"
    --gain-scale "${GAIN}"
    --official-stand
    "${LOC_FLAGS[@]}"
    --obs-csv "${CSV}"
    "${EXTRA_ARGS[@]}"
  )
fi

echo "============================================================"
echo " MODE ${MODE} : ${LOC_FLAGS[*]}  gain_scale=${GAIN}  ->  ${CSV}"
echo " shared /body_drive/* rehearsal against ${DIST_DIR}"
echo " warmup=${WARMUP}s start=${RUNNER_START} auto_reset=${AUTO_RESET}"
echo "============================================================"

if [[ "${PRINT_CMD}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}" < /dev/null &
RUNNER=$!
trap 'kill "${RUNNER}" 2>/dev/null || true' INT TERM

if [[ "${AUTO_RESET}" == "1" ]]; then
  (
    sleep 2
    "${SCRIPT_DIR}/reset_sim.sh" >/dev/null 2>&1 || true
    sleep 3
    "${SCRIPT_DIR}/reset_sim.sh" >/dev/null 2>&1 || true
    echo "[run_mode] auto-reset sent (x2) when the source sim exposes /sim/a3/reset"
  ) &
fi

wait "${RUNNER}"
