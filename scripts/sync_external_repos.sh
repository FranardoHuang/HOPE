#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTRL_URL="${TTRL_REPO_URL:-https://github.com/purdue-tracelab/TTRL-ICRA2026.git}"
TTRL_DIR="${TTRL_REPO_DIR:-${ROOT_DIR}/external_repos/TTRL-ICRA2026}"

sync_ttrl() {
  mkdir -p "$(dirname "${TTRL_DIR}")"

  if [ ! -d "${TTRL_DIR}/.git" ]; then
    echo "Cloning TTRL reference into ${TTRL_DIR}"
    git clone "${TTRL_URL}" "${TTRL_DIR}"
  fi

  cd "${TTRL_DIR}"

  local before
  before="$(git rev-parse --short HEAD)"

  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "TTRL has local uncommitted changes; fetching only and leaving the worktree untouched."
    git fetch --prune origin
    echo "TTRL remains at ${before}."
    return 0
  fi

  git fetch --prune origin

  local branch="${TTRL_BRANCH:-}"
  if [ -z "${branch}" ]; then
    local default_ref
    default_ref="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD || true)"
    branch="${default_ref#origin/}"
  fi
  if [ -z "${branch}" ] || [ "${branch}" = "refs/remotes/origin/HEAD" ]; then
    branch="main"
  fi

  if ! git show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
    if git show-ref --verify --quiet "refs/remotes/origin/main"; then
      branch="main"
    elif git show-ref --verify --quiet "refs/remotes/origin/master"; then
      branch="master"
    else
      echo "Could not find an origin branch to sync." >&2
      return 1
    fi
  fi

  if git show-ref --verify --quiet "refs/heads/${branch}"; then
    git checkout "${branch}"
  else
    git checkout -B "${branch}" "origin/${branch}"
  fi

  git pull --ff-only origin "${branch}"

  local after full_after
  after="$(git rev-parse --short HEAD)"
  full_after="$(git rev-parse HEAD)"

  echo "TTRL synced on ${branch}: ${before} -> ${after}"
  echo "TTRL source commit: ${full_after}"
}

sync_ttrl
