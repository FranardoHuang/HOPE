#!/usr/bin/env python3
"""Run the matched, exclusive A211/C211 profiler-off ABBA rate diagnostic.

This tool deliberately does not add another training recipe.  It consumes one
already validated A211 ``scale4096`` plan and one already validated C211
``scale4096`` plan, revalidates each through its pinned launcher, and derives
four fresh workloads in the order A-C-C-A.  The only semantic mutations are:

* ``max_iterations=61`` (10 warm-up + 50 measured + one excluded tail),
* ``algo.runner.save_interval=1000``,
* a fresh rate-only run name, and
* a fresh control namespace (with its derived log path).

The result is diagnostic/rate-only.  It is never promotion, long-run, physics,
or deployment evidence.  The main timing path explicitly disables the optional
ActionBall update profiler; profiler rows in a log make the run invalid.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PLAN_KIND = "action_ball_a211_c211_rate_abba_plan_v1"
BLOCK_KIND = "action_ball_a211_c211_rate_block_claim_v1"
BLOCK_RESULT_KIND = "action_ball_a211_c211_rate_block_result_v1"
RECEIPT_KIND = "action_ball_a211_c211_rate_abba_receipt_v1"

ORDER = ("A211", "C211", "C211", "A211")
MAX_ITERATIONS = 61
SAVE_INTERVAL = 1000
WARMUP_UPDATES = 10
MEASURED_UPDATES = 50
TAIL_UPDATES = 1
NUM_ENVS = 4096
STEPS_PER_ENV = 24
CI_BLOCK_LENGTH = 5
CI_BLOCK_COUNT = MEASURED_UPDATES // CI_BLOCK_LENGTH
T_975_DF9 = 2.2621571628540993
COMPLETION_TIMEOUT_S = 7200

PROFILE_ENV = "HOPE_ACTION_BALL_UPDATE_PROFILE"
PROFILE_PREFIX = "HOPE_ACTION_BALL_UPDATE_PROFILE_JSON="
BEHAVIOR_PREFIX = "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
TRAINING_LEDGER_EVENT = "action_ball_training_ledger"

SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITERATION_TIME_RE = re.compile(
    r"^\s*Iteration time:\s*([0-9]+(?:\.[0-9]+)?)s\s*$"
)


class RateBenchRefused(ValueError):
    """Raised when an input or observed run is not the matched rate task."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RateBenchRefused("value is not canonical finite JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RateBenchRefused("expected one regular file: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    final = path.lstat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or identity != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise RateBenchRefused("file changed while hashing: %s" % path)
    return digest.hexdigest()


def _sha(value: Any, *, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise RateBenchRefused("%s must be 64 lowercase hex" % name)
    return value


def _plain_int(
    value: Any, *, name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum:
        raise RateBenchRefused("%s must be an integer >= %d" % (name, minimum))
    if maximum is not None and value > maximum:
        raise RateBenchRefused("%s exceeds %d" % (name, maximum))
    return value


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise RateBenchRefused("%s must be a finite number" % name)
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        raise RateBenchRefused("%s must be finite%s" % (name, " and positive" if positive else ""))
    return parsed


def _exact_dict(value: Any, keys: Iterable[str], *, name: str) -> dict[str, Any]:
    expected = tuple(keys)
    if type(value) is not dict or set(value) != set(expected):
        raise RateBenchRefused(
            "%s keys differ: expected=%s observed=%s"
            % (name, sorted(expected), sorted(value) if type(value) is dict else type(value).__name__)
        )
    return value


def _absolute_file(value: str, *, name: str) -> Path:
    if type(value) is not str:
        raise RateBenchRefused("%s must be a path string" % name)
    path = Path(value)
    if not path.is_absolute():
        raise RateBenchRefused("%s must be absolute" % name)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RateBenchRefused("%s does not exist" % name) from exc
    if resolved != path or not path.is_file():
        raise RateBenchRefused("%s must be one resolved regular file" % name)
    return path


def _future_absolute_file(value: str, *, name: str) -> Path:
    if type(value) is not str:
        raise RateBenchRefused("%s must be a path string" % name)
    path = Path(value)
    if not path.is_absolute():
        raise RateBenchRefused("%s must be absolute" % name)
    if path.exists() or path.is_symlink():
        raise RateBenchRefused("%s is no-clobber and already exists" % name)
    parent = path.parent.resolve(strict=True)
    if parent != path.parent or not parent.is_dir():
        raise RateBenchRefused("%s parent must be one resolved directory" % name)
    return path


def _load_canonical_file(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != _sha(expected_sha256, name="expected %s SHA-256" % name):
        raise RateBenchRefused("%s file SHA-256 differs" % name)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RateBenchRefused("%s is not strict UTF-8 JSON" % name) from exc
    if type(value) is not dict or raw != canonical_bytes(value) + b"\n":
        raise RateBenchRefused("%s must use canonical JSON bytes plus newline" % name)
    return value


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise RateBenchRefused("no-clobber output already exists: %s" % path)
    raw = canonical_bytes(dict(value)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


@contextlib.contextmanager
def _profile_environment(value: Any):
    if value not in (None, "0"):
        raise RateBenchRefused("base plan must be profiler-off or not-requested")
    marker = object()
    previous = os.environ.get(PROFILE_ENV, marker)
    try:
        if value is None:
            os.environ.pop(PROFILE_ENV, None)
        else:
            os.environ[PROFILE_ENV] = "0"
        yield
    finally:
        if previous is marker:
            os.environ.pop(PROFILE_ENV, None)
        else:
            os.environ[PROFILE_ENV] = str(previous)


def _load_launcher(path: Path, expected_sha256: str, *, family: str):
    digest = sha256_file(path)
    if digest != _sha(expected_sha256, name="expected %s launcher SHA-256" % family):
        raise RateBenchRefused("%s launcher SHA-256 differs" % family)
    module_name = "_hope_rate_%s_%s" % (family.lower(), digest[:16])
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RateBenchRefused("cannot import %s launcher" % family)
    module = importlib.util.module_from_spec(spec)
    scripts = str(path.parent)
    inserted = scripts not in sys.path
    if inserted:
        sys.path.insert(0, scripts)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        raise RateBenchRefused("cannot load pinned %s launcher: %s" % (family, exc)) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(scripts)
            except ValueError:  # pragma: no cover - defensive
                pass
    return module


def _assignment_rows(argv: Any, *, name: str) -> dict[str, str]:
    if type(argv) is not list or len(argv) < 3 or any(type(item) is not str for item in argv):
        raise RateBenchRefused("%s must be a training argv list" % name)
    rows: dict[str, str] = {}
    for item in argv[2:]:
        key, separator, value = item.partition("=")
        if not separator:
            continue
        normalized = key.lstrip("+")
        if normalized in rows:
            raise RateBenchRefused("%s has duplicate assignment %s" % (name, normalized))
        rows[normalized] = value
    return rows


def _runtime_source_pin(
    payload: Mapping[str, Any], *, checkout: Path, path: Path, expected_sha256: str, name: str
) -> dict[str, str]:
    try:
        relative = path.relative_to(checkout).as_posix()
    except ValueError as exc:
        raise RateBenchRefused("%s must be inside source checkout" % name) from exc
    sources = payload.get("runtime_sources")
    if type(sources) is not dict:
        raise RateBenchRefused("plan runtime_sources must be an object")
    matches = []
    for pin in sources.values():
        if type(pin) is dict and pin.get("path") == relative:
            matches.append(pin)
    if len(matches) != 1 or matches[0].get("sha256") != expected_sha256:
        raise RateBenchRefused("%s is not uniquely pinned by the validated plan" % name)
    return {"path": relative, "sha256": expected_sha256}


def _call_launcher_validator(module: Any, payload: dict[str, Any], *, family: str) -> None:
    validator = getattr(module, "_revalidate_claim_payload", None)
    if not callable(validator):
        raise RateBenchRefused("%s launcher has no claim revalidation hook" % family)
    before = canonical_sha256(payload)
    try:
        parameters = inspect.signature(validator).parameters
        if "claimed" in parameters:
            validator(payload, claimed=False)
        else:  # exact repin hook for a launcher that removes the optional keyword
            validator(payload)
    except BaseException as exc:
        raise RateBenchRefused("%s launcher rejected the existing plan: %s" % (family, exc)) from exc
    if canonical_sha256(payload) != before:
        raise RateBenchRefused("%s launcher mutated the plan while validating" % family)


def _validate_base_plan(
    *,
    family: str,
    plan_path: Path,
    plan_sha256: str,
    launcher_path: Path,
    launcher_sha256: str,
) -> dict[str, Any]:
    outer = _load_canonical_file(plan_path, plan_sha256, name="%s plan" % family)
    _exact_dict(
        outer,
        ("schema_version", "kind", "launch_claim_sha256", "canonical_payload"),
        name="%s plan envelope" % family,
    )
    payload = outer["canonical_payload"]
    claim_sha = _sha(outer["launch_claim_sha256"], name="%s launch claim" % family)
    if type(payload) is not dict or canonical_sha256(payload) != claim_sha:
        raise RateBenchRefused("%s launch plan seal differs" % family)
    if payload.get("diagnostic_unauthorized") is not True:
        raise RateBenchRefused("%s plan is not diagnostic_unauthorized" % family)
    if any(
        payload.get(key) is not True
        for key in (
            "formal_evidence_prohibited",
            "promotion_prohibited",
            "resume_prohibited",
            "export_prohibited",
            "deployment_prohibited",
            "hardware_prohibited",
            "fresh_only",
        )
    ):
        raise RateBenchRefused("%s plan does not preserve diagnostic prohibitions" % family)
    spec = payload.get("spec")
    if type(spec) is not dict:
        raise RateBenchRefused("%s plan spec is absent" % family)
    source = spec.get("source")
    if type(source) is not dict:
        raise RateBenchRefused("%s spec source is absent" % family)
    checkout = Path(source.get("checkout", ""))
    if not checkout.is_absolute() or checkout.resolve(strict=True) != checkout:
        raise RateBenchRefused("%s checkout must be exact and absolute" % family)
    if source.get("commit_sha") is None or re.fullmatch(r"[0-9a-f]{40}", str(source["commit_sha"])) is None:
        raise RateBenchRefused("%s source commit is invalid" % family)
    launcher = _load_launcher(launcher_path, launcher_sha256, family=family)
    profile = payload.get("output_contract", {}).get("update_profile", {})
    forwarded = profile.get("forwarded_value")
    if forwarded not in (None, "0") or profile.get("mode") not in (
        "not_requested",
        "explicit_profiler_off",
    ):
        raise RateBenchRefused("%s base plan is profiler-instrumented" % family)
    with _profile_environment(forwarded):
        _call_launcher_validator(launcher, payload, family=family)
    launcher_pin = _runtime_source_pin(
        payload,
        checkout=checkout,
        path=launcher_path,
        expected_sha256=launcher_sha256,
        name="%s launcher" % family,
    )
    kit_relative = getattr(launcher, "KIT_LAUNCHER_SOURCE", None)
    if type(kit_relative) is not str:
        raise RateBenchRefused("%s launcher does not expose its Kit lock source" % family)
    kit_path = checkout / kit_relative
    kit_sha = sha256_file(kit_path)
    kit_pin = _runtime_source_pin(
        payload,
        checkout=checkout,
        path=kit_path,
        expected_sha256=kit_sha,
        name="locked Kit launcher",
    )
    gpu = spec.get("gpu")
    if (
        type(gpu) is not dict
        or gpu.get("require_empty") is not True
        or type(gpu.get("index")) is not int
        or type(gpu.get("uuid")) is not str
        or type(gpu.get("lock_path")) is not str
    ):
        raise RateBenchRefused("%s rate input must be an exclusive GPU plan" % family)
    if any(value is True for key, value in spec.items() if "colocation" in key.lower()):
        raise RateBenchRefused("%s rate input must disable colocation" % family)
    if (
        spec.get("stage") != "scale4096"
        or spec.get("num_envs") != NUM_ENVS
        or spec.get("max_iterations") != 5
        or spec.get("save_interval") != 1
    ):
        raise RateBenchRefused("%s input must be the exact 4096x5 scale plan" % family)
    deferred = payload.get("output_contract", {}).get("deferred_matched_speed_measurement")
    if (
        type(deferred) is not dict
        or deferred.get("num_envs") != NUM_ENVS
        or deferred.get("steps_per_env") != STEPS_PER_ENV
        or deferred.get("warmup_updates") != WARMUP_UPDATES
        or deferred.get("minimum_measured_updates") != MEASURED_UPDATES
        or deferred.get("abba_order") != ["current_A", "current_C", "current_C", "current_A"]
        or deferred.get("main_timing_mode") != "profiler_off"
        or deferred.get("isolation") != "exclusive_single_process_same_gpu"
    ):
        raise RateBenchRefused("%s deferred matched-speed contract differs" % family)
    argv = payload.get("training_argv")
    assignments = _assignment_rows(argv, name="%s training argv" % family)
    required = {
        "num_envs": str(NUM_ENVS),
        "max_iterations": "5",
        "algo.runner.save_interval": "1",
    }
    if any(assignments.get(key) != value for key, value in required.items()):
        raise RateBenchRefused("%s training argv scale budget differs" % family)
    expected_target = "online_solver" if family == "A211" else "direct_ball"
    expected_reuse = "true" if family == "A211" else "false"
    if (
        assignments.get("task.racket.action_ball_target_source") != expected_target
        or assignments.get("task.racket.action_ball_reuse_exact_question_until_semantics_change")
        != expected_reuse
    ):
        raise RateBenchRefused("%s target-source runtime differs" % family)
    bundle = payload.get("bundle")
    family_recipe_key = "arm" if family == "A211" else "recipe"
    if type(bundle) is not dict or type(bundle.get(family_recipe_key)) is not dict:
        raise RateBenchRefused("%s plan recipe is absent" % family)
    return {
        "family": family,
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "launch_claim_sha256": claim_sha,
        "launcher": launcher_pin,
        "launcher_absolute_path": str(launcher_path),
        "kit_launcher": kit_pin,
        "kit_launcher_absolute_path": str(kit_path),
        "source": copy.deepcopy(source),
        "gpu": copy.deepcopy(gpu),
        "spec": copy.deepcopy(spec),
        "training_argv": list(argv),
        "assignments": assignments,
        "runtime_assets": copy.deepcopy(payload.get("runtime_assets")),
        "matched_recipe": copy.deepcopy(bundle[family_recipe_key]),
        "four_grid_manifest": copy.deepcopy(bundle.get("isaac_four_grid_manifest")),
        "update_profile": copy.deepcopy(profile),
    }


def _matched_plan_contract(a: Mapping[str, Any], c: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("source", "gpu", "kit_launcher", "four_grid_manifest"):
        if a[key] != c[key]:
            raise RateBenchRefused("A/C matched plan %s differs" % key)
    a_recipe = a["matched_recipe"]
    c_recipe = c["matched_recipe"]
    for key in ("ppo", "soft_weights", "reference_guard_mode"):
        if a_recipe.get(key) != c_recipe.get(key):
            raise RateBenchRefused("A/C matched recipe %s differs" % key)
    shared_assignments = (
        "seed",
        "num_envs",
        "algo.runner.empirical_normalization",
        "algo.policy.actor_hidden_dims",
        "algo.policy.critic_hidden_dims",
        "algo.policy.init_noise_std",
        "algo.policy.noise_std_type",
        "algo.algorithm.entropy_coef",
        "algo.algorithm.schedule",
        "algo.algorithm.learning_rate",
        "algo.algorithm.desired_kl",
        "algo.algorithm.clip_param",
        "algo.algorithm.num_learning_epochs",
        "algo.algorithm.num_mini_batches",
        "action_ball_dynamic_ready_artifact_sha256",
        "action_ball_dynamic_ready_nominal_receipt_sha256",
        "motion_file",
        "task.racket.clip_names",
        "task.racket.action_ball_manifest_sha256",
        "task.racket.action_ball_seed",
        "task.actions.control_step_action_delay_min",
        "task.actions.control_step_action_delay_max",
        "task.push.enable",
        "task.physical_ball",
        "task.racket.virtual_ball",
        "task.racket.action_ball_target_observation_noise",
    )
    for key in shared_assignments:
        if a["assignments"].get(key) != c["assignments"].get(key):
            raise RateBenchRefused("A/C matched argv assignment %s differs" % key)
    return {
        "source_commit": a["source"]["commit_sha"],
        "checkout": a["source"]["checkout"],
        "gpu_index": a["gpu"]["index"],
        "gpu_uuid": a["gpu"]["uuid"],
        "gpu_lock_path": a["gpu"]["lock_path"],
        "num_envs": NUM_ENVS,
        "steps_per_env": STEPS_PER_ENV,
        "ppo": copy.deepcopy(a_recipe["ppo"]),
        "soft_weights": copy.deepcopy(a_recipe["soft_weights"]),
        "reference_guard_mode": a_recipe["reference_guard_mode"],
    }


def _replace_assignment(argv: Sequence[str], key: str, value: str) -> list[str]:
    output = list(argv)
    matches = []
    for index, item in enumerate(output):
        raw_key, separator, _old = item.partition("=")
        if separator and raw_key.lstrip("+") == key:
            matches.append((index, raw_key))
    if len(matches) != 1:
        raise RateBenchRefused("training argv must contain exactly one %s" % key)
    index, raw_key = matches[0]
    output[index] = "%s=%s" % (raw_key, value)
    return output


def _derive_training_argv(base: Sequence[str], *, run_name: str) -> tuple[list[str], list[dict[str, str]]]:
    original = list(base)
    output = _replace_assignment(original, "max_iterations", str(MAX_ITERATIONS))
    output = _replace_assignment(output, "algo.runner.save_interval", str(SAVE_INTERVAL))
    output = _replace_assignment(output, "run_name", run_name)
    changed = []
    for before, after in zip(original, output):
        if before != after:
            key = before.partition("=")[0].lstrip("+")
            changed.append({"field": key, "before": before, "after": after})
    if [row["field"] for row in changed] != [
        "max_iterations",
        "algo.runner.save_interval",
        "run_name",
    ]:
        raise RateBenchRefused("derived argv changed outside the exact allowlist")
    return output, changed


def _block_claim(
    *, index: int, family: str, base: Mapping[str, Any], run_prefix: str
) -> dict[str, Any]:
    parent = Path(base["spec"]["namespace"]).parent
    component = "%s-b%02d-%s" % (run_prefix, index, family.lower())
    if SAFE_COMPONENT.fullmatch(component) is None:
        raise RateBenchRefused("derived namespace component is unsafe")
    namespace = parent / component
    run_name = component + "-DIAGNOSTIC-RATE-ONLY"
    runtime_dir = parent / run_name
    argv, argv_mutations = _derive_training_argv(base["training_argv"], run_name=run_name)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": BLOCK_KIND,
        "diagnostic_unauthorized": True,
        "rate_only": True,
        "formal_evidence_prohibited": True,
        "family": family,
        "order_index": index,
        "source_plan": copy.deepcopy(base["plan"]),
        "source_launch_claim_sha256": base["launch_claim_sha256"],
        "launcher": copy.deepcopy(base["launcher"]),
        "namespace": str(namespace),
        "run_name": run_name,
        "runtime_dir": str(runtime_dir),
        "run_log": str(namespace / "run.log"),
        "launch_state": str(namespace / "run.log.launch"),
        "budget": {
            "max_iterations": MAX_ITERATIONS,
            "save_interval": SAVE_INTERVAL,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
            "tail_updates": TAIL_UPDATES,
        },
        "allowed_semantic_mutations": [
            "max_iterations",
            "save_interval",
            "run_name",
            "namespace_and_derived_log_paths",
        ],
        "argv_mutations": argv_mutations,
        "training_argv": argv,
        "profiler_environment": {PROFILE_ENV: "0"},
    }
    return {**unsigned, "block_claim_sha256": canonical_sha256(unsigned)}


def build_benchmark_plan(
    *,
    a_plan_path: Path,
    a_plan_sha256: str,
    c_plan_path: Path,
    c_plan_sha256: str,
    a_launcher_path: Path,
    a_launcher_sha256: str,
    c_launcher_path: Path,
    c_launcher_sha256: str,
    run_prefix: str,
    receipt_path: Path,
) -> dict[str, Any]:
    if SAFE_COMPONENT.fullmatch(run_prefix) is None:
        raise RateBenchRefused("run prefix is unsafe")
    a = _validate_base_plan(
        family="A211",
        plan_path=a_plan_path,
        plan_sha256=a_plan_sha256,
        launcher_path=a_launcher_path,
        launcher_sha256=a_launcher_sha256,
    )
    c = _validate_base_plan(
        family="C211",
        plan_path=c_plan_path,
        plan_sha256=c_plan_sha256,
        launcher_path=c_launcher_path,
        launcher_sha256=c_launcher_sha256,
    )
    matched = _matched_plan_contract(a, c)
    blocks = []
    for index, family in enumerate(ORDER):
        blocks.append(_block_claim(index=index, family=family, base=a if family == "A211" else c, run_prefix=run_prefix))
    paths = []
    for block in blocks:
        for key in ("namespace", "runtime_dir", "run_log", "launch_state"):
            path = Path(block[key])
            if path.exists() or path.is_symlink():
                raise RateBenchRefused("derived %s already exists: %s" % (key, path))
            paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise RateBenchRefused("derived ABBA paths are not unique")
    self_path = Path(__file__).resolve(strict=True)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "diagnostic_unauthorized": True,
        "rate_only": True,
        "formal_evidence_prohibited": True,
        "promotion_prohibited": True,
        "resume_prohibited": True,
        "export_prohibited": True,
        "deployment_prohibited": True,
        "hardware_prohibited": True,
        "benchmark_source": {"path": str(self_path), "sha256": sha256_file(self_path)},
        "inputs": {
            "A211": {
                "plan": copy.deepcopy(a["plan"]),
                "launcher": {"path": str(a_launcher_path), "sha256": a_launcher_sha256},
            },
            "C211": {
                "plan": copy.deepcopy(c["plan"]),
                "launcher": {"path": str(c_launcher_path), "sha256": c_launcher_sha256},
            },
        },
        "matched_contract": matched,
        "order": list(ORDER),
        "timing_contract": {
            "mode": "profiler_off_rsl_iteration_wall",
            "profile_environment": {PROFILE_ENV: "0"},
            "profile_rows_required": 0,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
            "tail_updates": TAIL_UPDATES,
            "total_updates": MAX_ITERATIONS,
            "ci_method": "contiguous_5_update_batch_means_student_t",
            "ci_confidence": 0.95,
            "ci_block_length_updates": CI_BLOCK_LENGTH,
            "ci_block_count": CI_BLOCK_COUNT,
            "isolation": "exclusive_single_process_same_gpu_full_abba_lock",
        },
        "kit_launcher": copy.deepcopy(a["kit_launcher"]),
        "kit_launcher_absolute_path": a["kit_launcher_absolute_path"],
        "blocks": blocks,
        "receipt_path": str(receipt_path),
    }
    return {**unsigned, "content_sha256": canonical_sha256(unsigned)}


def _validate_benchmark_plan(value: Any) -> dict[str, Any]:
    keys = (
        "schema_version",
        "kind",
        "diagnostic_unauthorized",
        "rate_only",
        "formal_evidence_prohibited",
        "promotion_prohibited",
        "resume_prohibited",
        "export_prohibited",
        "deployment_prohibited",
        "hardware_prohibited",
        "benchmark_source",
        "inputs",
        "matched_contract",
        "order",
        "timing_contract",
        "kit_launcher",
        "kit_launcher_absolute_path",
        "blocks",
        "receipt_path",
        "content_sha256",
    )
    row = _exact_dict(value, keys, name="benchmark plan")
    unsigned = dict(row)
    seal = _sha(unsigned.pop("content_sha256"), name="benchmark plan content SHA")
    if canonical_sha256(unsigned) != seal:
        raise RateBenchRefused("benchmark plan content seal differs")
    if (
        row["schema_version"] != SCHEMA_VERSION
        or row["kind"] != PLAN_KIND
        or row["diagnostic_unauthorized"] is not True
        or row["rate_only"] is not True
        or row["order"] != list(ORDER)
    ):
        raise RateBenchRefused("benchmark plan identity differs")
    source_pin = row["benchmark_source"]
    if type(source_pin) is not dict or source_pin.get("path") != str(Path(__file__).resolve(strict=True)):
        raise RateBenchRefused("benchmark source path differs")
    if sha256_file(Path(source_pin["path"])) != source_pin.get("sha256"):
        raise RateBenchRefused("benchmark source SHA differs")
    blocks = row["blocks"]
    if type(blocks) is not list or len(blocks) != len(ORDER):
        raise RateBenchRefused("benchmark must contain four ABBA blocks")
    for index, (block, family) in enumerate(zip(blocks, ORDER)):
        if type(block) is not dict or block.get("family") != family or block.get("order_index") != index:
            raise RateBenchRefused("benchmark ABBA block identity differs")
        unsigned_block = dict(block)
        block_sha = _sha(unsigned_block.pop("block_claim_sha256", None), name="block claim SHA")
        if canonical_sha256(unsigned_block) != block_sha:
            raise RateBenchRefused("benchmark block claim seal differs")
        if block.get("allowed_semantic_mutations") != [
            "max_iterations",
            "save_interval",
            "run_name",
            "namespace_and_derived_log_paths",
        ]:
            raise RateBenchRefused("benchmark block mutation allowlist differs")
        assignments = _assignment_rows(block.get("training_argv"), name="benchmark block argv")
        if (
            assignments.get("max_iterations") != str(MAX_ITERATIONS)
            or assignments.get("algo.runner.save_interval") != str(SAVE_INTERVAL)
            or assignments.get("run_name") != block.get("run_name")
        ):
            raise RateBenchRefused("benchmark block argv budget/run name differs")
    return row


def _load_and_revalidate_benchmark_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    row = _validate_benchmark_plan(
        _load_canonical_file(path, expected_sha256, name="benchmark plan")
    )
    inputs = row["inputs"]
    validated = {}
    for family in ("A211", "C211"):
        source = inputs.get(family)
        if type(source) is not dict:
            raise RateBenchRefused("benchmark %s input is absent" % family)
        plan_pin = source.get("plan")
        launcher_pin = source.get("launcher")
        if type(plan_pin) is not dict or type(launcher_pin) is not dict:
            raise RateBenchRefused("benchmark %s pins are invalid" % family)
        validated[family] = _validate_base_plan(
            family=family,
            plan_path=_absolute_file(plan_pin.get("path"), name="%s plan" % family),
            plan_sha256=plan_pin.get("sha256"),
            launcher_path=_absolute_file(launcher_pin.get("path"), name="%s launcher" % family),
            launcher_sha256=launcher_pin.get("sha256"),
        )
    if _matched_plan_contract(validated["A211"], validated["C211"]) != row["matched_contract"]:
        raise RateBenchRefused("benchmark matched contract drifted")
    for index, family in enumerate(ORDER):
        expected = _block_claim(
            index=index,
            family=family,
            base=validated[family],
            run_prefix=Path(row["blocks"][index]["namespace"]).name.rsplit("-b%02d-" % index, 1)[0],
        )
        if expected != row["blocks"][index]:
            raise RateBenchRefused("benchmark derived block drifted")
    return row


def _counter_value(value: Any, *, name: str) -> int:
    return _plain_int(value, name=name, minimum=0)


def _flatten_solver_rejections(row: Mapping[str, Any]) -> int:
    rejections = row.get("solver_rejections")
    if type(rejections) is not dict:
        raise RateBenchRefused("training ledger solver_rejections is absent")
    total = 0
    for action, reasons in rejections.items():
        if type(action) is not str or type(reasons) is not dict:
            raise RateBenchRefused("training ledger solver rejection shape differs")
        for reason, count in reasons.items():
            if type(reason) is not str:
                raise RateBenchRefused("solver rejection reason is invalid")
            total += _counter_value(count, name="solver rejection counter")
    return total


def _ledger_snapshot(row: Mapping[str, Any], *, family: str) -> dict[str, int]:
    cache = row.get("exact_question_answer_cache")
    if family == "A211":
        if type(cache) is not dict or cache.get("policy") != "reuse_exact_question_until_semantics_change":
            raise RateBenchRefused("A211 rate ledger lacks exact-question cache authority")
        novel = _counter_value(cache.get("novel_producer_count"), name="A211 novel producer count")
        hits = _counter_value(cache.get("consumer_hit_count"), name="A211 cache hit count")
    else:
        if cache is not None:
            raise RateBenchRefused("C211 direct-ball ledger must not install the inverse cache")
        novel = 0
        hits = 0
    return {
        "online_inverse_solve_count": novel,
        "exact_question_cache_hit_count": hits,
        "solver_rejection_count": _flatten_solver_rejections(row),
    }


def _snapshot_delta(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    output = {}
    for key in after:
        delta = after[key] - before.get(key, 0)
        if delta < 0:
            raise RateBenchRefused("training ledger counter decreased: %s" % key)
        output[key + "_delta"] = delta
    return output


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RateBenchRefused("cannot summarize an empty sample")
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise RateBenchRefused("sample standard deviation requires two rows")
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _block_ci(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != MEASURED_UPDATES:
        raise RateBenchRefused("block CI requires exactly 50 measured updates")
    block_means = [
        _mean(values[start : start + CI_BLOCK_LENGTH])
        for start in range(0, len(values), CI_BLOCK_LENGTH)
    ]
    if len(block_means) != CI_BLOCK_COUNT:
        raise RateBenchRefused("block CI batch count differs")
    center = _mean(block_means)
    standard_error = _sample_std(block_means) / math.sqrt(len(block_means))
    margin = T_975_DF9 * standard_error
    return {
        "method": "contiguous_5_update_batch_means_student_t",
        "confidence": 0.95,
        "block_length_updates": CI_BLOCK_LENGTH,
        "block_count": CI_BLOCK_COUNT,
        "student_t_critical": T_975_DF9,
        "block_means_s": block_means,
        "mean_s": center,
        "standard_error_s": standard_error,
        "lower_s": center - margin,
        "upper_s": center + margin,
    }


def _parse_prefixed_json(lines: Sequence[str], prefix: str, *, name: str) -> list[dict[str, Any]]:
    output = []
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            row = json.loads(line[len(prefix) :])
        except json.JSONDecodeError as exc:
            raise RateBenchRefused("%s row is malformed JSON" % name) from exc
        if type(row) is not dict:
            raise RateBenchRefused("%s row must be an object" % name)
        output.append(row)
    return output


def _training_ledgers(lines: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if type(row) is dict and row.get("event") == TRAINING_LEDGER_EVENT:
            output.append(row)
    return output


def analyze_run_log(text: str, *, family: str) -> dict[str, Any]:
    if family not in ("A211", "C211"):
        raise RateBenchRefused("rate log family must be A211 or C211")
    clean_lines = [ANSI_RE.sub("", line) for line in text.splitlines()]
    if any(line.startswith(PROFILE_PREFIX) for line in clean_lines):
        raise RateBenchRefused("profiler instrumentation leaked into rate-only timing")
    timings = []
    for line in clean_lines:
        match = ITERATION_TIME_RE.fullmatch(line)
        if match:
            timings.append(_finite(float(match.group(1)), name="iteration time", positive=True))
    if len(timings) != MAX_ITERATIONS:
        raise RateBenchRefused("rate log must contain exactly 61 iteration-time rows")
    behavior = _parse_prefixed_json(clean_lines, BEHAVIOR_PREFIX, name="exact behavior")
    ledgers = _training_ledgers(clean_lines)
    if len(behavior) != MAX_ITERATIONS or len(ledgers) != MAX_ITERATIONS:
        raise RateBenchRefused("rate log must contain one behavior and one ledger row per update")
    behavior_updates = [
        _plain_int(row.get("ppo_update"), name="behavior PPO update") for row in behavior
    ]
    ledger_updates = [
        _plain_int(row.get("step"), name="training ledger step") for row in ledgers
    ]
    expected_updates = list(range(behavior_updates[0], behavior_updates[0] + MAX_ITERATIONS))
    if behavior_updates != expected_updates or ledger_updates != expected_updates:
        raise RateBenchRefused("rate log update identities are not one common contiguous window")
    measured_slice = slice(WARMUP_UPDATES, WARMUP_UPDATES + MEASURED_UPDATES)
    measured_times = timings[measured_slice]
    measured_behavior = behavior[measured_slice]
    snapshots = [_ledger_snapshot(row, family=family) for row in ledgers]
    zero = {key: 0 for key in snapshots[0]}
    warm_end = snapshots[WARMUP_UPDATES - 1]
    measured_end = snapshots[WARMUP_UPDATES + MEASURED_UPDATES - 1]
    tail_end = snapshots[-1]
    deltas = {
        "warmup": _snapshot_delta(warm_end, zero),
        "measured": _snapshot_delta(measured_end, warm_end),
        "tail": _snapshot_delta(tail_end, measured_end),
        "full_observed": _snapshot_delta(tail_end, zero),
    }
    if family == "A211":
        if (
            deltas["warmup"]["online_inverse_solve_count_delta"] != 1
            or deltas["measured"]["online_inverse_solve_count_delta"] != 0
            or deltas["tail"]["online_inverse_solve_count_delta"] != 0
        ):
            raise RateBenchRefused("A211 ABBA rate task is not one cold solve plus steady exact reuse")
    elif any(
        window["online_inverse_solve_count_delta"] != 0 for window in deltas.values()
    ):
        raise RateBenchRefused("C211 direct-ball rate task performed an inverse solve")
    update_rows = []
    exact_strata: dict[str, list[float]] = {}
    for offset, (seconds, row) in enumerate(zip(measured_times, measured_behavior)):
        counters = row.get("counters")
        if type(counters) is not dict:
            raise RateBenchRefused("exact behavior counters are absent")
        terminal = _counter_value(counters.get("terminal_reset_count", 0), name="terminal reset count")
        timeout = _counter_value(counters.get("timeout_reset_count", 0), name="timeout reset count")
        reasons = {
            key: _counter_value(value, name="termination reason counter")
            for key, value in sorted(counters.items())
            if key.startswith("termination_reason_")
        }
        signature = "terminal=%d,timeout=%d" % (terminal, timeout)
        exact_strata.setdefault(signature, []).append(seconds)
        update_rows.append(
            {
                "update": behavior_updates[WARMUP_UPDATES + offset],
                "iteration_time_s": seconds,
                "terminal_reset_count": terminal,
                "timeout_reset_count": timeout,
                "termination_reason_counts": reasons,
            }
        )
    reset_free = [row["iteration_time_s"] for row in update_rows if row["terminal_reset_count"] + row["timeout_reset_count"] == 0]
    reset_present = [row["iteration_time_s"] for row in update_rows if row["terminal_reset_count"] + row["timeout_reset_count"] > 0]
    strata = {
        "exact_reset_count": {
            key: {"update_count": len(values), "mean_iteration_time_s": _mean(values)}
            for key, values in sorted(exact_strata.items())
        },
        "reset_free": {
            "update_count": len(reset_free),
            "mean_iteration_time_s": None if not reset_free else _mean(reset_free),
        },
        "reset_present": {
            "update_count": len(reset_present),
            "mean_iteration_time_s": None if not reset_present else _mean(reset_present),
        },
    }
    ci = _block_ci(measured_times)
    env_steps = NUM_ENVS * STEPS_PER_ENV
    return {
        "family": family,
        "profiler_mode": "explicit_off",
        "profile_row_count": 0,
        "update_ids": {
            "first": expected_updates[0],
            "warmup": expected_updates[:WARMUP_UPDATES],
            "measured": expected_updates[WARMUP_UPDATES : WARMUP_UPDATES + MEASURED_UPDATES],
            "tail": expected_updates[-TAIL_UPDATES:],
        },
        "timing": {
            "warmup_iteration_times_s": timings[:WARMUP_UPDATES],
            "measured_update_rows": update_rows,
            "excluded_tail_iteration_times_s": timings[-TAIL_UPDATES:],
            "measured_mean_iteration_time_s": ci["mean_s"],
            "measured_env_steps_per_second": env_steps / ci["mean_s"],
            "block_confidence_interval": ci,
        },
        "solver_inverse_deltas": deltas,
        "reset_strata": strata,
    }


def _contrast(block_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [row["analysis"]["family"] for row in block_results] != list(ORDER):
        raise RateBenchRefused("ABBA result order differs")
    means = [
        row["analysis"]["timing"]["block_confidence_interval"]["block_means_s"]
        for row in block_results
    ]
    if any(len(value) != CI_BLOCK_COUNT for value in means):
        raise RateBenchRefused("ABBA block means are incomplete")
    a_means = [(means[0][i] + means[3][i]) / 2.0 for i in range(CI_BLOCK_COUNT)]
    c_means = [(means[1][i] + means[2][i]) / 2.0 for i in range(CI_BLOCK_COUNT)]
    differences = [a - c for a, c in zip(a_means, c_means)]
    log_ratios = [math.log(c / a) for a, c in zip(a_means, c_means)]

    def ten_row_interval(values: Sequence[float]) -> dict[str, float]:
        center = _mean(values)
        se = _sample_std(values) / math.sqrt(len(values))
        margin = T_975_DF9 * se
        return {"mean": center, "standard_error": se, "lower": center - margin, "upper": center + margin}

    difference_ci = ten_row_interval(differences)
    log_ratio_ci = ten_row_interval(log_ratios)
    mean_a = _mean(a_means)
    mean_c = _mean(c_means)
    return {
        "method": "symmetric_abba_paired_contiguous_batch_means",
        "paired_block_count": CI_BLOCK_COUNT,
        "a_mean_iteration_time_s": mean_a,
        "c_mean_iteration_time_s": mean_c,
        "a_minus_c_iteration_time_s": difference_ci,
        "c_over_a_iteration_time_ratio": mean_c / mean_a,
        "a_over_c_env_step_rate_ratio": mean_c / mean_a,
        "a_over_c_rate_ratio_95ci": {
            "lower": math.exp(log_ratio_ci["lower"]),
            "center": math.exp(log_ratio_ci["mean"]),
            "upper": math.exp(log_ratio_ci["upper"]),
            "log_standard_error": log_ratio_ci["standard_error"],
        },
    }


def _probe_gpu_empty(gpu_index: int, gpu_uuid: str) -> dict[str, Any]:
    identity = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if identity.returncode != 0:
        raise RateBenchRefused("nvidia-smi GPU identity query failed")
    rows = [tuple(part.strip() for part in line.split(",", 1)) for line in identity.stdout.splitlines() if line.strip()]
    if (str(gpu_index), gpu_uuid) not in rows:
        raise RateBenchRefused("physical GPU index/UUID binding differs")
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if compute.returncode != 0:
        raise RateBenchRefused("nvidia-smi compute-process query failed")
    matching = [line.strip() for line in compute.stdout.splitlines() if line.strip() and line.split(",", 1)[0].strip() == gpu_uuid]
    if matching:
        raise RateBenchRefused("exclusive ABBA GPU is not empty: %s" % matching)
    return {"gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "compute_processes": [], "empty": True}


def _open_exclusive_gpu_lock(path: Path) -> int:
    if not path.is_absolute():
        raise RateBenchRefused("GPU lock path must be absolute")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise RateBenchRefused("GPU lock must be a regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RateBenchRefused("exclusive ABBA GPU lock is already held") from exc
    return descriptor


def _validate_completion_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RateBenchRefused("Kit completion state is absent")
    observed = {}
    required = {"completion_exit_code", "terminal_kind", "terminal_exit_code"}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in required:
            if key in observed:
                raise RateBenchRefused("Kit completion state has duplicate %s" % key)
            observed[key] = value
    expected = {"completion_exit_code": "0", "terminal_kind": "clean_completion", "terminal_exit_code": "0"}
    if observed != expected:
        raise RateBenchRefused("rate workload did not exit cleanly")
    return observed


def _run_block(plan_path: Path, plan_sha256: str, block: Mapping[str, Any]) -> dict[str, Any]:
    namespace = Path(block["namespace"])
    runtime_dir = Path(block["runtime_dir"])
    if namespace.exists() or namespace.is_symlink() or runtime_dir.exists() or runtime_dir.is_symlink():
        raise RateBenchRefused("rate block namespace/runtime is already spent")
    namespace.mkdir(mode=0o700)
    claim_path = namespace / "rate_block_claim.json"
    _write_exclusive_json(claim_path, block)
    log_path = Path(block["run_log"])
    state_path = Path(block["launch_state"])
    plan = _load_and_revalidate_benchmark_plan(plan_path, plan_sha256)
    kit = Path(plan["kit_launcher_absolute_path"])
    source = plan["matched_contract"]
    python = plan["inputs"][block["family"]]["plan"]
    base_outer = _load_canonical_file(Path(python["path"]), python["sha256"], name="block source plan")
    isaac_python = base_outer["canonical_payload"]["spec"]["source"]["isaac_python"]
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "KIT_BOOT_MARKER": "Learning iteration",
        "KIT_BOOT_TIMEOUT_S": "2700",
        "KIT_BOOT_STALE_TIMEOUT_S": "1800",
        "KIT_BOOT_POLL_S": "5",
        "KIT_BOOT_STATE_FILE": str(state_path),
        "KIT_WAIT_FOR_COMPLETION": "1",
        "KIT_COMPLETION_TIMEOUT_S": str(COMPLETION_TIMEOUT_S),
    }
    command = [
        isaac_python,
        str(Path(__file__).resolve(strict=True)),
        "_exec-block",
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan_sha256,
        "--block-index",
        str(block["order_index"]),
    ]
    result = subprocess.run(
        [str(kit), str(log_path), *command],
        cwd=Path(source["checkout"]) / "hope_training/whole_body_tracking",
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise RateBenchRefused("locked Kit rate block returned %d; namespace is spent" % result.returncode)
    completion = _validate_completion_state(state_path)
    analysis = analyze_run_log(log_path.read_text(encoding="utf-8"), family=block["family"])
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": BLOCK_RESULT_KIND,
        "diagnostic_unauthorized": True,
        "rate_only": True,
        "block_claim_sha256": block["block_claim_sha256"],
        "family": block["family"],
        "order_index": block["order_index"],
        "namespace": str(namespace),
        "completion": completion,
        "run_log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        "analysis": analysis,
    }
    output = {**unsigned, "content_sha256": canonical_sha256(unsigned)}
    result_path = namespace / "rate_block_result.json"
    file_sha = _write_exclusive_json(result_path, output)
    return {**output, "artifact": {"path": str(result_path), "sha256": file_sha}}


def _launcher_runtime_exec_environment(
    *,
    launcher: Any,
    payload: Mapping[str, Any],
    block: Mapping[str, Any],
) -> dict[str, str]:
    """Recreate the pinned launcher's non-argv runtime authority for a rate block."""

    spec = copy.deepcopy(payload.get("spec"))
    if type(spec) is not dict:
        raise RateBenchRefused("runtime source plan spec is absent")
    spec["namespace"] = block["namespace"]
    spec["log_path"] = block["run_log"]
    namespace_receipt = getattr(launcher, "_runtime_namespace_receipt", None)
    prelong_environment = getattr(
        launcher, "_prelong_semantics_exec_environment", None
    )
    admission_constants = getattr(launcher, "_A", None)
    if (
        not callable(namespace_receipt)
        or not callable(prelong_environment)
        or admission_constants is None
    ):
        raise RateBenchRefused(
            "pinned launcher lacks namespace/pre-long runtime authority hooks"
        )
    receipt_env = getattr(admission_constants, "GPU_NAMESPACE_RECEIPT_ENV", None)
    receipt_sha_env = getattr(
        admission_constants, "GPU_NAMESPACE_RECEIPT_SHA_ENV", None
    )
    if type(receipt_env) is not str or type(receipt_sha_env) is not str:
        raise RateBenchRefused("pinned launcher lacks GPU namespace receipt constants")
    try:
        receipt_path, receipt_sha = namespace_receipt(
            spec, block["block_claim_sha256"]
        )
        materialization = payload["materialization_inputs"]
        if block["family"] == "A211":
            reward_sha = materialization["arm_materialization"][
                "runtime_effective_reward_sha256"
            ]
        else:
            reward_sha = materialization["reward_materialization"][
                "runtime_effective_reward_sha256"
            ]
        prelong = prelong_environment(spec["stage"], reward_sha)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RateBenchRefused(
            "pinned launcher could not build the rate runtime authority: %s" % exc
        ) from exc
    if type(receipt_path) is not Path:
        receipt_path = Path(receipt_path)
    if (
        receipt_path != Path(block["namespace"]) / receipt_path.name
        or not receipt_path.is_file()
        or sha256_file(receipt_path) != _sha(receipt_sha, name="namespace receipt SHA")
    ):
        raise RateBenchRefused("derived GPU namespace receipt identity differs")
    if (
        type(prelong) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in prelong.items())
    ):
        raise RateBenchRefused("pinned launcher returned an invalid pre-long environment")
    return {
        receipt_env: str(receipt_path),
        receipt_sha_env: receipt_sha,
        **prelong,
    }


def _internal_exec_block(plan_path: Path, plan_sha256: str, block_index: int) -> int:
    plan = _load_and_revalidate_benchmark_plan(plan_path, plan_sha256)
    index = _plain_int(block_index, name="block index", maximum=len(ORDER) - 1)
    block = plan["blocks"][index]
    claim_path = Path(block["namespace"]) / "rate_block_claim.json"
    claim = _load_canonical_file(claim_path, sha256_file(claim_path), name="rate block claim")
    if claim != block:
        raise RateBenchRefused("runtime rate block claim differs from benchmark plan")
    family = block["family"]
    input_pin = plan["inputs"][family]
    base_outer = _load_canonical_file(Path(input_pin["plan"]["path"]), input_pin["plan"]["sha256"], name="runtime source plan")
    payload = base_outer["canonical_payload"]
    launcher_path = Path(input_pin["launcher"]["path"])
    launcher = _load_launcher(launcher_path, input_pin["launcher"]["sha256"], family=family)
    profile = payload["output_contract"]["update_profile"]["forwarded_value"]
    with _profile_environment(profile):
        _call_launcher_validator(launcher, payload, family=family)
    runtime_asset_environment = getattr(getattr(launcher, "_B", None), "_runtime_asset_exec_environment", None)
    if not callable(runtime_asset_environment):
        raise RateBenchRefused("pinned launcher lacks runtime asset environment hook")
    launcher_environment = _launcher_runtime_exec_environment(
        launcher=launcher,
        payload=payload,
        block=block,
    )
    checkout = Path(plan["matched_contract"]["checkout"])
    wbt = checkout / "hope_training/whole_body_tracking"
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(wbt / "source/whole_body_tracking"),
        "CUDA_VISIBLE_DEVICES": str(plan["matched_contract"]["gpu_index"]),
        "HYDRA_FULL_ERROR": "1",
        "WANDB_MODE": "offline",
        "HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256": block["block_claim_sha256"],
        PROFILE_ENV: "0",
        **launcher_environment,
        **runtime_asset_environment(payload["runtime_assets"]),
    }
    os.chdir(wbt)
    argv = block["training_argv"]
    os.execve(argv[0], argv, environment)
    raise AssertionError("execve returned")


def execute_benchmark(
    plan_path: Path,
    plan_sha256: str,
    *,
    block_runner: Callable[[Path, str, Mapping[str, Any]], dict[str, Any]] = _run_block,
    gpu_probe: Callable[[int, str], dict[str, Any]] = _probe_gpu_empty,
) -> dict[str, Any]:
    plan = _load_and_revalidate_benchmark_plan(plan_path, plan_sha256)
    receipt_path = _future_absolute_file(plan["receipt_path"], name="rate receipt")
    gpu = plan["matched_contract"]
    lock_fd = _open_exclusive_gpu_lock(Path(gpu["gpu_lock_path"]))
    results = []
    probes = []
    try:
        probes.append({"boundary": "before_abba", **gpu_probe(gpu["gpu_index"], gpu["gpu_uuid"])})
        for index, block in enumerate(plan["blocks"]):
            result = block_runner(plan_path, plan_sha256, block)
            if result.get("family") != ORDER[index] or result.get("order_index") != index:
                raise RateBenchRefused("rate block runner returned the wrong ABBA cell")
            results.append(result)
            probes.append({"boundary": "after_block_%d" % index, **gpu_probe(gpu["gpu_index"], gpu["gpu_uuid"])})
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "diagnostic_unauthorized": True,
        "rate_only": True,
        "formal_evidence_prohibited": True,
        "promotion_prohibited": True,
        "resume_prohibited": True,
        "export_prohibited": True,
        "deployment_prohibited": True,
        "hardware_prohibited": True,
        "benchmark_plan": {"path": str(plan_path), "sha256": plan_sha256, "content_sha256": plan["content_sha256"]},
        "order": list(ORDER),
        "timing_contract": copy.deepcopy(plan["timing_contract"]),
        "matched_contract": copy.deepcopy(plan["matched_contract"]),
        "exclusive_gpu_probes": probes,
        "blocks": results,
        "abba_contrast": _contrast(results),
        "interpretation": {
            "consumer_rate_only": True,
            "novel_question_producer_rate_excluded": True,
            "profile_attribution_excluded": True,
            "scale4096_or_long_readiness_claimed": False,
        },
    }
    receipt = {**unsigned, "content_sha256": canonical_sha256(unsigned)}
    file_sha = _write_exclusive_json(receipt_path, receipt)
    return {"status": "COMPLETE_DIAGNOSTIC_RATE_ONLY", "receipt": str(receipt_path), "receipt_sha256": file_sha, "content_sha256": receipt["content_sha256"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="validate A/C plans and write one no-clobber ABBA plan")
    plan.add_argument("--a-plan", required=True)
    plan.add_argument("--expected-a-plan-sha256", required=True)
    plan.add_argument("--c-plan", required=True)
    plan.add_argument("--expected-c-plan-sha256", required=True)
    plan.add_argument("--a-launcher", required=True)
    plan.add_argument("--expected-a-launcher-sha256", required=True)
    plan.add_argument("--c-launcher", required=True)
    plan.add_argument("--expected-c-launcher-sha256", required=True)
    plan.add_argument("--run-prefix", required=True)
    plan.add_argument("--receipt", required=True)
    plan.add_argument("--output", required=True)
    execute = sub.add_parser("execute", help="run the exact exclusive A-C-C-A plan")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--confirm-plan-sha256", required=True)
    internal = sub.add_parser("_exec-block", help=argparse.SUPPRESS)
    internal.add_argument("--plan", required=True)
    internal.add_argument("--plan-sha256", required=True)
    internal.add_argument("--block-index", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            output = _future_absolute_file(args.output, name="benchmark plan output")
            receipt = _future_absolute_file(args.receipt, name="future rate receipt")
            document = build_benchmark_plan(
                a_plan_path=_absolute_file(args.a_plan, name="A211 plan"),
                a_plan_sha256=args.expected_a_plan_sha256,
                c_plan_path=_absolute_file(args.c_plan, name="C211 plan"),
                c_plan_sha256=args.expected_c_plan_sha256,
                a_launcher_path=_absolute_file(args.a_launcher, name="A211 launcher"),
                a_launcher_sha256=args.expected_a_launcher_sha256,
                c_launcher_path=_absolute_file(args.c_launcher, name="C211 launcher"),
                c_launcher_sha256=args.expected_c_launcher_sha256,
                run_prefix=args.run_prefix,
                receipt_path=receipt,
            )
            file_sha = _write_exclusive_json(output, document)
            result = {"status": "PLANNED_DIAGNOSTIC_RATE_ONLY", "plan": str(output), "plan_sha256": file_sha, "content_sha256": document["content_sha256"]}
        elif args.command == "execute":
            result = execute_benchmark(
                _absolute_file(args.plan, name="benchmark plan"),
                _sha(args.confirm_plan_sha256, name="confirmed benchmark plan SHA"),
            )
        else:
            return _internal_exec_block(
                _absolute_file(args.plan, name="benchmark plan"),
                _sha(args.plan_sha256, name="benchmark plan SHA"),
                args.block_index,
            )
        print(json.dumps(result, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        return 0
    except (RateBenchRefused, FileNotFoundError, OSError, ValueError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
