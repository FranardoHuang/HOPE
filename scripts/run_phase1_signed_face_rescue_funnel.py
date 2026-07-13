#!/usr/bin/env python3
"""Fail-closed launcher for the Phase-1 signed-face single-seed funnel.

This program launches simulator training only.  It owns exactly the four A/B/C/D
cells in the checked manifest, never searches for a process to signal, and never
contains a robot command.  L1 must finish and produce a no-clobber activation
record before L2 can start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CONTRACT_MARKER = "[train.py] hard training contract:"
READY_MARKER = "Learning iteration"
FAILURE_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemory|Segmentation fault|\bNaN\b|\bInf\b|\bKilled\b",
    re.IGNORECASE,
)

EXPECTED_CELLS = {
    "A": ("hot_parent", 0.0, False, "hot_control"),
    "B": ("hot_parent", -0.4, False, "hot_guidance"),
    "C": ("fresh", 0.0, True, "fresh_control"),
    "D": ("fresh", -0.4, True, "fresh_guidance"),
}
EXPECTED_CURRENT_ONLY_KEYS = {
    "motion_adaptive_alpha",
    "motion_adaptive_kernel_size",
    "motion_adaptive_lambda",
    "motion_adaptive_uniform_ratio",
    "motion_clip_switch_prob",
    "motion_event_timing",
    "motion_post_swing_buffer_size",
    "motion_post_swing_min_fill",
    "motion_post_swing_min_hold",
    "motion_post_swing_start_prob",
    "motion_rsi_skip_settle_frames",
    "motion_stagger_hold_max_steps",
    "motion_stagger_initial_clock",
    "racket_midswing_resample_prob",
    "racket_midswing_resample_tts_floor",
    "racket_strike_phase",
    "racket_strike_window_pos_s",
    "racket_strike_window_s",
    "racket_strike_window_wide_s",
    "racket_target_bias_per_swing",
    "racket_target_delay_steps",
    "racket_target_dropout_prob",
    "racket_target_jitter_pos_per_s",
    "racket_target_jitter_vel_per_s",
    "racket_target_noise_ar1_rho",
    "racket_target_noise_ar1_sigma",
    "racket_target_noise_white",
    "racket_target_post_strike_dropout_s",
}


class ContractError(RuntimeError):
    """A frozen launch invariant was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be one lowercase SHA-256")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys changed: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read manifest: {exc}") from exc
    if data.get("schema_version") != 1:
        raise ContractError("manifest schema_version must be 1")
    if data.get("manifest_id") != "phase1-signed-face-rescue-single-seed-funnel-20260713-v5":
        raise ContractError("unexpected manifest_id")
    if data.get("simulation_only") is not True or data.get("real_robot_commands_forbidden") is not True:
        raise ContractError("manifest must be simulation-only and explicitly forbid robot commands")
    if data.get("seed_replication_before_l2_decision_forbidden") is not True:
        raise ContractError("manifest must forbid first-round seed replication")

    source = data.get("source")
    runtime = data.get("runtime")
    inputs = data.get("inputs")
    shared = data.get("shared_training_contract")
    stages = data.get("stages")
    transition = data.get("hot_start_contract_transition")
    activation = data.get("activation_contract")
    evaluation = data.get("evaluation_contract")
    if not all(isinstance(v, dict) for v in (
        source, runtime, inputs, shared, stages, transition, activation, evaluation
    )):
        raise ContractError("source/runtime/inputs/shared/stages/transition/activation/evaluation must be objects")
    if not COMMIT_RE.fullmatch(str(source.get("expected_training_commit", ""))):
        raise ContractError("expected_training_commit must be a full lowercase Git commit")
    critical = source.get("critical_files")
    if not isinstance(critical, dict) or not critical:
        raise ContractError("critical_files must be a non-empty object")
    for relative, digest in critical.items():
        rel = Path(relative)
        if not isinstance(relative, str) or rel.is_absolute() or ".." in rel.parts:
            raise ContractError(f"unsafe critical source path: {relative!r}")
        require_sha(digest, f"critical source {relative}")
    ignored_asset = source.get("ignored_runtime_asset")
    if not isinstance(ignored_asset, dict):
        raise ContractError("ignored_runtime_asset must be an object")
    require_exact_keys(ignored_asset, {
        "relative_path", "file_count", "total_file_bytes", "tree_content_sha256",
        "restore_source_checkout", "restore_source_commit", "restore_source_relative_path",
        "target_must_be_gitignored", "symlinks_forbidden",
    }, "ignored runtime asset")
    expected_asset = {
        "relative_path": "source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
        "file_count": 46,
        "total_file_bytes": 15378264,
        "tree_content_sha256": "0137f59b1fe45e7d5f8fa731bedca905f5466bc98e8d1354081fe071d60426c6",
        "restore_source_checkout": "/workspace/codexschema/nohope",
        "restore_source_commit": "6d93bcb16c422a2f42748c2dc99432559653480b",
        "restore_source_relative_path": "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
        "target_must_be_gitignored": True,
        "symlinks_forbidden": True,
    }
    if ignored_asset != expected_asset:
        raise ContractError("ignored A3 runtime asset contract changed")

    if runtime.get("pod") != "pod1" or runtime.get("gpu") != 0:
        raise ContractError("v5 reserves exactly Pod1 GPU0")
    if runtime.get("initial_gpu_must_have_zero_compute_processes") is not True:
        raise ContractError("the four-cell pool must start on an empty GPU")
    if runtime.get("maximum_trainers_on_gpu") != 4:
        raise ContractError("one card must contain exactly four causal cells")
    if runtime.get("kit_boot_marker") != CONTRACT_MARKER:
        raise ContractError("Kit boot marker changed")
    if runtime.get("isaaclab_root") != "/workspace/IsaacLab":
        raise ContractError("v3 requires the reviewed IsaacLab root")
    require_sha(runtime.get("training_environment_sha256"), "training environment")
    if runtime.get("setup_train_env_local_override_forbidden") is not True:
        raise ContractError("training-local environment overrides must remain forbidden")

    expected_input_keys = {
        "forehand_motion", "backhand_motion", "schema3_train_bank", "hot_parent_checkpoint"
    }
    require_exact_keys(inputs, expected_input_keys, "inputs")
    for name in ("forehand_motion", "backhand_motion", "schema3_train_bank"):
        item = inputs[name]
        if not isinstance(item, dict):
            raise ContractError(f"{name} must be an object")
        rel = Path(str(item.get("relative_path", "")))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ContractError(f"unsafe {name} path")
        require_sha(item.get("sha256"), f"{name} hash")
    parent = inputs["hot_parent_checkpoint"]
    if not isinstance(parent, dict):
        raise ContractError("hot parent must be an object")
    if parent.get("embedded_iteration") != 13800:
        raise ContractError("hot parent must be exact model_13800")
    if parent.get("embedded_training_contract_schema_version") != 3:
        raise ContractError("hot parent must carry schema-3")
    if parent.get("embedded_training_contract_lineage_exact") is not True:
        raise ContractError("hot parent input must itself be exact-bound")
    parent_sha = require_sha(parent.get("sha256"), "hot parent hash")
    embedded_sha = require_sha(
        parent.get("embedded_training_contract_sha256"), "hot parent embedded contract"
    )
    adjacent_sha = require_sha(
        parent.get("adjacent_training_contract_sha256"), "hot parent adjacent contract"
    )
    if parent_sha != "478efa8d163ec53dbade328c5de18947f6c068df78cbadff8e46a29844bdc9e6":
        raise ContractError("hot parent checkpoint changed")
    if embedded_sha != adjacent_sha:
        raise ContractError("hot parent embedded and adjacent contracts differ")

    exact_shared = {
        "training_seed": 3,
        "face_command_pairing": "shared_plus_y",
        "mount_normal_sign_per_clip": [1.0, -1.0],
        "zero_joint_friction": True,
        "motion_kinematics_exact": True,
        "question_bank_schema_version": 3,
        "question_bank_split": "train",
        "actor_observation_contract": "deploy_parity_face179",
        "actor_observation_dim": 179,
        "action_dim": 31,
        "strike_phase_per_clip": [0.471, 0.338],
        "event_timing_mode": "disabled",
        "save_interval": 100,
        "positional_guidance_weight": 0.0,
        "face_guidance_theta_max": math.pi,
    }
    for key, expected in exact_shared.items():
        if shared.get(key) != expected:
            raise ContractError(f"shared contract field {key} changed")
    recipe = shared.get("base_recipe")
    if not isinstance(recipe, list) or not recipe or not all(isinstance(x, str) for x in recipe):
        raise ContractError("base_recipe must be a non-empty list of strings")
    forbidden_recipe_fragments = (
        "checkpoint_path=", "seed=", "num_envs=", "max_iterations=", "run_name=",
        "racket_face_guidance_weight", "racket_face_guidance_theta_max",
        "ros2 ", "run_deploy", "real_robot", "joint_command", "/dev/",
    )
    for arg in recipe:
        if any(fragment in arg.lower() for fragment in forbidden_recipe_fragments):
            raise ContractError(f"base recipe contains a per-cell or forbidden argument: {arg}")

    cells = data.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ContractError("manifest must contain exactly four cells")
    actual_cells: dict[str, tuple[Any, ...]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ContractError("every cell must be an object")
        cell_id = cell.get("cell_id")
        if cell_id in actual_cells:
            raise ContractError(f"duplicate cell {cell_id}")
        actual_cells[cell_id] = (
            cell.get("initialization"), cell.get("face_guidance_weight"),
            cell.get("expected_lineage_exact"), cell.get("causal_role"),
        )
    if actual_cells != EXPECTED_CELLS:
        raise ContractError(f"A/B/C/D causal grid changed: {actual_cells}")

    if set(stages) != {"l1", "l2"}:
        raise ContractError("stages must be exactly l1 and l2")
    expected_stage = {
        "l1": (512, 25, False),
        "l2": (4096, 1001, True),
    }
    seen_names: set[str] = set()
    for stage_name, (num_envs, iterations, requires_activation) in expected_stage.items():
        stage = stages[stage_name]
        if stage.get("num_envs") != num_envs or stage.get("max_iterations") != iterations:
            raise ContractError(f"{stage_name} training budget changed")
        if stage.get("requires_activation") is not requires_activation:
            raise ContractError(f"{stage_name} activation rule changed")
        names = stage.get("run_names")
        terminal = stage.get("expected_terminal_checkpoint_iteration")
        if not isinstance(names, dict) or set(names) != set(EXPECTED_CELLS):
            raise ContractError(f"{stage_name} must name A/B/C/D exactly once")
        if not isinstance(terminal, dict) or set(terminal) != set(EXPECTED_CELLS):
            raise ContractError(f"{stage_name} terminal map must cover A/B/C/D")
        for cell_id, name in names.items():
            if not isinstance(name, str) or not RUN_NAME_RE.fullmatch(name) or name in seen_names:
                raise ContractError(f"unsafe or duplicate run name: {name!r}")
            if f"_{cell_id}_" not in name or not name.endswith("_seed3"):
                raise ContractError(f"run name does not bind cell and seed: {name}")
            seen_names.add(name)
    if stages["l1"]["expected_terminal_checkpoint_iteration"] != {
        "A": 13824, "B": 13824, "C": 24, "D": 24
    }:
        raise ContractError("L1 terminal off-by-one contract changed")
    if stages["l2"].get("relative_milestones") != [200, 500, 1000]:
        raise ContractError("L2 relative milestones must be +200/+500/+1000")
    if stages["l2"].get("launch_authorized") is not False:
        raise ContractError("L2 must remain blocked until a signed directional paper is frozen")
    if not isinstance(stages["l2"].get("blocked_on"), str) or not stages["l2"]["blocked_on"]:
        raise ContractError("L2 must name its signed-paper blocker")
    if stages["l2"].get("checkpoint_iterations") != {
        "A": [14000, 14300, 14800], "B": [14000, 14300, 14800],
        "C": [200, 500, 1000], "D": [200, 500, 1000],
    }:
        raise ContractError("L2 absolute checkpoint map changed")
    if stages["l2"]["expected_terminal_checkpoint_iteration"] != {
        "A": 14800, "B": 14800, "C": 1000, "D": 1000
    }:
        raise ContractError("L2 terminal off-by-one contract changed")

    if transition.get("classification") != "explicit_inexact_representation_transfer":
        raise ContractError("hot A/B must remain explicit inexact transfers")
    if transition.get("checkpoint_allow_missing_contract") is not False:
        raise ContractError("hot A/B may not allow a missing parent contract")
    if transition.get("checkpoint_allow_contract_mismatch") is not True:
        raise ContractError("hot A/B must expose the known source-contract extension")
    if transition.get("checkpoint_tolerant") is not False:
        raise ContractError("hot A/B must use strict tensor loading")
    if transition.get("all_parent_and_current_common_fields_must_match") is not True:
        raise ContractError("hot transition must compare every common field")
    if set(transition.get("allowed_current_only_top_level_keys", [])) != EXPECTED_CURRENT_ONLY_KEYS:
        raise ContractError("allowed source-contract extension keys changed")

    if activation.get("l2_requires_all_four_l1_terminal_checkpoints") is not True:
        raise ContractError("L2 must require all four L1 terminals")
    if activation.get("all_four_emitted_hard_contract_sha256_must_match") is not True:
        raise ContractError("L1 must converge on one emitted hard-contract SHA")
    if activation.get("hot_lineage_must_be_false") is not True or activation.get("fresh_lineage_must_be_true") is not True:
        raise ContractError("activation must preserve hot-inexact/fresh-exact lineage")
    if activation.get("l1_activation_alone_may_launch_l2") is not False:
        raise ContractError("L1 completion alone may not launch L2")
    if activation.get("l2_requires_separate_signed_directional_paper_activation") is not True:
        raise ContractError("L2 must require a separate signed directional paper activation")
    if evaluation.get("automatic_judge_launch") is not False:
        raise ContractError("no exam SHA is frozen; automatic judge must remain off")
    if evaluation.get("second_seed_automatic") is not False:
        raise ContractError("a second seed may not be automatic")
    if evaluation.get("l2_training_launch_authorized") is not False:
        raise ContractError("L2 training launch must remain blocked")
    if evaluation.get("signed_directional_checkpoint_paper") is not None:
        raise ContractError("v5 must not invent a signed directional checkpoint paper")
    return data


def cell_by_id(manifest: dict[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [cell for cell in manifest["cells"] if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ContractError(f"unknown cell: {cell_id}")
    return matches[0]


def source_input_paths(manifest: dict[str, Any]) -> dict[str, tuple[Path, str]]:
    root = Path(manifest["runtime"]["source_asset_root"]).resolve()
    result: dict[str, tuple[Path, str]] = {}
    for name in ("forehand_motion", "backhand_motion", "schema3_train_bank"):
        item = manifest["inputs"][name]
        path = (root / item["relative_path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"{name} escapes source asset root") from exc
        result[name] = (path, item["sha256"])
    return result


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_training_source(manifest: dict[str, Any]) -> tuple[Path, Path]:
    source = manifest["source"]
    checkout = Path(source["training_checkout"]).resolve()
    try:
        head = git_output(checkout, "rev-parse", "HEAD")
        dirty = git_output(checkout, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"training checkout is missing or unreadable by Git: {checkout}") from exc
    if head != source["expected_training_commit"]:
        raise ContractError("training checkout is at the wrong commit")
    if dirty:
        raise ContractError("training checkout is dirty")
    wbt = checkout / source["wbt_relative_path"]
    for relative, expected in source["critical_files"].items():
        path = wbt / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"critical training source is missing or changed: {path}")
    return checkout, wbt


def asset_tree_content(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"ignored runtime asset root is missing or a symlink: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"ignored runtime asset contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"ignored runtime asset contains a special entry: {path}")
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": canonical_sha256({"files": rows}),
    }


def verify_ignored_runtime_asset(
    manifest: dict[str, Any], checkout: Path, wbt: Path
) -> dict[str, Any]:
    spec = manifest["source"]["ignored_runtime_asset"]
    target_candidate = wbt / spec["relative_path"]
    if target_candidate.is_symlink():
        raise ContractError("ignored runtime asset target root may not be a symlink")
    target = target_candidate.resolve()
    try:
        target.relative_to(wbt.resolve())
    except ValueError as exc:
        raise ContractError("ignored runtime asset target escapes training worktree") from exc
    restore_checkout = Path(spec["restore_source_checkout"]).resolve()
    if git_output(restore_checkout, "rev-parse", "HEAD") != spec["restore_source_commit"]:
        raise ContractError("ignored asset restore checkout commit changed")
    if git_output(restore_checkout, "status", "--porcelain"):
        raise ContractError("ignored asset restore checkout is dirty")
    restore_candidate = restore_checkout / spec["restore_source_relative_path"]
    if restore_candidate.is_symlink():
        raise ContractError("ignored asset restore root may not be a symlink")
    restore = restore_candidate.resolve()
    target_content = asset_tree_content(target)
    restore_content = asset_tree_content(restore)
    expected = {
        "file_count": spec["file_count"],
        "total_file_bytes": spec["total_file_bytes"],
        "tree_content_sha256": spec["tree_content_sha256"],
    }
    if target_content != expected or restore_content != expected:
        raise ContractError("ignored A3 asset target/restore tree does not match preregistration")
    target_relative = target.relative_to(checkout.resolve())
    ignored = subprocess.run(
        ["git", "-C", str(checkout), "check-ignore", "-q", str(target_relative)],
        check=False,
    )
    if ignored.returncode != 0:
        raise ContractError("restored A3 asset is not Git-ignored in the training checkout")
    return {
        "target_path": str(target),
        "restore_source_path": str(restore),
        **target_content,
        "target_gitignored": True,
        "symlinks_present": False,
    }


def build_training_environment(manifest: dict[str, Any], wbt: Path) -> dict[str, str]:
    """Build a deterministic source-first environment for the exact worktree."""

    runtime = manifest["runtime"]
    local_override = wbt / "setup_train_env.local.sh"
    if local_override.exists():
        raise ContractError(f"untracked training environment override is forbidden: {local_override}")
    isaaclab = Path(runtime["isaaclab_root"]).resolve()
    pythonpath_entries = [
        (wbt / "source/whole_body_tracking").resolve(),
        isaaclab / "source/isaaclab",
        isaaclab / "source/isaaclab_tasks",
        isaaclab / "source/isaaclab_assets",
        isaaclab / "source/isaaclab_rl",
    ]
    for path in pythonpath_entries:
        if not path.is_dir():
            raise ContractError(f"training PYTHONPATH entry is missing: {path}")
    pythonpath = ":".join(str(path) for path in pythonpath_entries)
    environment = {
        "HOME": "/root",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "root",
        "USER": "root",
        "SHELL": "/bin/bash",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin",
        "HOPE_ISAAC_PYTHON": runtime["isaac_python"],
        "HOPE_ISAACLAB_ROOT": str(isaaclab),
        "HOPE_WBT_PYTHONPATH": pythonpath,
        "PYTHONPATH": pythonpath,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "TMPDIR": "/workspace/tmp",
        "PIP_CACHE_DIR": "/workspace/.cache/pip",
        "XDG_CACHE_HOME": "/workspace/.cache",
        "WANDB_DIR": "/workspace/codexschema/.wandb",
        "WANDB_ENTITY": "BerkeleyPingPong",
        "WANDB_REGISTRY_ORG": "dongc_1-university-of-california-berkeley-org",
        "WANDB_PROJECT": "hope_wbc",
        "WANDB_MOTION_PROJECT": "csv_to_npz",
        "CUDA_VISIBLE_DEVICES": str(runtime["gpu"]),
        "PYTHONUNBUFFERED": "1",
    }
    for key in ("TMPDIR", "XDG_CACHE_HOME", "WANDB_DIR"):
        if not Path(environment[key]).is_dir():
            raise ContractError(f"training environment directory is missing: {environment[key]}")
    if canonical_sha256(environment) != runtime["training_environment_sha256"]:
        raise ContractError("deterministic training environment digest changed")
    return environment


def verify_training_module_resolution(
    python: Path, wbt: Path, environment: dict[str, str]
) -> str:
    code = (
        "import importlib.util, pathlib; "
        "spec=importlib.util.find_spec('whole_body_tracking'); "
        "print('NONE' if spec is None or spec.origin is None else pathlib.Path(spec.origin).resolve())"
    )
    try:
        raw = subprocess.check_output(
            [str(python), "-c", code], cwd=wbt, env=environment,
            text=True, stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"exact training module resolution failed: {exc.output.strip()}") from exc
    if raw == "NONE":
        raise ContractError("whole_body_tracking is not resolvable in the exact training environment")
    module_path = Path(raw).resolve()
    expected_root = (wbt / "source/whole_body_tracking/whole_body_tracking").resolve()
    try:
        module_path.relative_to(expected_root)
    except ValueError as exc:
        raise ContractError(f"whole_body_tracking resolved outside exact worktree: {module_path}") from exc
    return str(module_path)


CHECKPOINT_AUDIT_CODE = r"""
import json, sys, torch
path = sys.argv[1]
obj = torch.load(path, map_location='cpu', weights_only=False)
floating_tensors = 0
floating_elements = 0
nonfinite = 0
stack = [obj]
seen_containers = set()
while stack:
    value = stack.pop()
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        floating_tensors += 1
        floating_elements += value.numel()
        nonfinite += int((~torch.isfinite(value)).sum().item())
    elif isinstance(value, dict):
        identity = id(value)
        if identity not in seen_containers:
            seen_containers.add(identity)
            stack.extend(value.values())
    elif isinstance(value, (list, tuple)):
        identity = id(value)
        if identity not in seen_containers:
            seen_containers.add(identity)
            stack.extend(value)
infos = obj.get('infos')
if not isinstance(infos, dict):
    infos = {}
print(json.dumps({
    'iter': obj.get('iter'),
    'training_contract_schema_version': infos.get('training_contract_schema_version'),
    'training_contract_sha256': infos.get('training_contract_sha256'),
    'training_contract_lineage_exact': infos.get('training_contract_lineage_exact'),
    'training_contract_provenance_location': 'infos',
    'floating_tensor_count': floating_tensors,
    'floating_elements': floating_elements,
    'nonfinite_floating_elements': nonfinite,
}, sort_keys=True))
"""


def checkpoint_audit(python: Path, path: Path) -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [str(python), "-c", CHECKPOINT_AUDIT_CODE, str(path)],
            text=True, stderr=subprocess.STDOUT,
        )
        result = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot inspect checkpoint {path}: {exc}") from exc
    if result.get("nonfinite_floating_elements") != 0 or result.get("floating_tensor_count", 0) <= 0:
        raise ContractError(f"checkpoint is non-finite or has no floating tensors: {path}")
    return result


def verify_inputs(manifest: dict[str, Any], python: Path) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, (path, expected) in source_input_paths(manifest).items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"{name} is missing or has the wrong SHA: {path}")
        verified[name] = {"path": str(path), "sha256": expected}
    parent = manifest["inputs"]["hot_parent_checkpoint"]
    checkpoint = Path(parent["path"])
    sidecar = Path(parent["adjacent_training_contract_path"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != parent["sha256"]:
        raise ContractError("hot parent checkpoint is missing or changed")
    if not sidecar.is_file() or sha256_file(sidecar) != parent["adjacent_training_contract_sha256"]:
        raise ContractError("hot parent adjacent hard contract is missing or changed")
    audit = checkpoint_audit(python, checkpoint)
    expected_audit = {
        "iter": parent["embedded_iteration"],
        "training_contract_schema_version": parent["embedded_training_contract_schema_version"],
        "training_contract_sha256": parent["embedded_training_contract_sha256"],
        "training_contract_lineage_exact": 1,
        "training_contract_provenance_location": "infos",
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ContractError(f"hot parent {key} mismatch: {audit.get(key)!r} != {expected!r}")
    verified["hot_parent_checkpoint"] = {
        "path": str(checkpoint), "sha256": parent["sha256"], "audit": audit,
        "adjacent_training_contract_path": str(sidecar),
        "adjacent_training_contract_sha256": parent["adjacent_training_contract_sha256"],
    }
    return verified


def verify_production_locations(
    manifest: dict[str, Any], config_path: Path, launcher_path: Path, checkout: Path
) -> None:
    control = Path(manifest["runtime"]["external_control_root"]).resolve()
    for label, path in (("manifest", config_path.resolve()), ("launcher", launcher_path.resolve())):
        try:
            path.relative_to(control)
        except ValueError as exc:
            raise ContractError(f"production {label} must live under {control}") from exc
        try:
            path.relative_to(checkout)
        except ValueError:
            pass
        else:
            raise ContractError(f"production {label} may not live inside the training checkout")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_launch_state(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    free_raw = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(free_raw) != 1 or not free_raw[0].strip().isdigit():
        raise ContractError(f"cannot read GPU {gpu} free memory")
    pids_raw = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    pids: list[int] = []
    for raw in pids_raw:
        if raw.strip().isdigit() and int(raw.strip()) not in pids:
            pids.append(int(raw.strip()))
    trainers: list[int] = []
    for pid in pids:
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.is_file() and b"scripts/train.py" in cmdline.read_bytes():
            trainers.append(pid)
    return {"gpu": gpu, "free_memory_mib": int(free_raw[0]), "compute_pids": pids, "trainer_pids": trainers}


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ContractError("/proc/meminfo lacks MemAvailable")


def build_command(
    manifest: dict[str, Any], stage_name: str, cell_id: str, *, wbt: Path
) -> list[str]:
    stage = manifest["stages"][stage_name]
    cell = cell_by_id(manifest, cell_id)
    runtime = manifest["runtime"]
    shared = manifest["shared_training_contract"]
    paths = source_input_paths(manifest)
    transition = manifest["hot_start_contract_transition"]
    command = [
        "env", f"CUDA_VISIBLE_DEVICES={runtime['gpu']}", "PYTHONUNBUFFERED=1",
        runtime["isaac_python"], "scripts/train.py", *shared["base_recipe"],
        f"seed={shared['training_seed']}", f"num_envs={stage['num_envs']}",
        f"max_iterations={stage['max_iterations']}",
        f"algo.runner.save_interval={shared['save_interval']}",
        f"run_name={stage['run_names'][cell_id]}",
    ]
    if cell["initialization"] == "hot_parent":
        command.extend([
            f"checkpoint_path={manifest['inputs']['hot_parent_checkpoint']['path']}",
            f"checkpoint_tolerant={str(transition['checkpoint_tolerant']).lower()}",
            f"checkpoint_allow_missing_contract={str(transition['checkpoint_allow_missing_contract']).lower()}",
            f"checkpoint_allow_contract_mismatch={str(transition['checkpoint_allow_contract_mismatch']).lower()}",
        ])
    else:
        command.extend([
            "checkpoint_path=null", "checkpoint_tolerant=false",
            "checkpoint_allow_missing_contract=false", "checkpoint_allow_contract_mismatch=false",
        ])
    command.extend([
        "task.plant.zero_joint_friction=true",
        "++task.motion.allow_legacy_link_origin_velocity=false",
        "++task.motion.event_timing_mode=disabled",
        f"motion_file={paths['forehand_motion'][0]}",
        f"motion_file_2={paths['backhand_motion'][0]}",
        "task.racket.strike_phase_per_clip=[0.471,0.338]",
        f"++task.racket.question_bank={paths['schema3_train_bank'][0]}",
        "++task.racket.face_command_pairing=shared_plus_y",
        "++task.rewards.racket_guidance_weight=0.0",
        f"++task.rewards.racket_face_guidance_weight={cell['face_guidance_weight']}",
        f"++task.rewards.racket_face_guidance_theta_max={shared['face_guidance_theta_max']}",
    ])
    forbidden_runtime_tokens = ("ros2 ", "run_deploy", "real_robot", "joint_command", "/dev/")
    if any(any(token in part.lower() for token in forbidden_runtime_tokens) for part in command):
        raise ContractError("constructed command unexpectedly contains a robot/runtime token")
    if wbt != Path(manifest["source"]["training_checkout"]).resolve() / manifest["source"]["wbt_relative_path"]:
        raise ContractError("command was built for a different training worktree")
    return [str(part) for part in command]


def verify_emitted_contract(
    contract_path: Path, manifest: dict[str, Any], *, hot: bool
) -> tuple[str, dict[str, Any]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    shared = manifest["shared_training_contract"]
    inputs = manifest["inputs"]
    if contract.get("schema_version") != 3:
        raise ContractError("emitted training contract is not schema-3")
    expected_scalars = {
        "actor_obs_contract": shared["actor_observation_contract"],
        "actor_obs_total_dim": shared["actor_observation_dim"],
        "face_command_pairing": shared["face_command_pairing"],
        "mount_normal_sign_per_clip": shared["mount_normal_sign_per_clip"],
        "strike_phase_per_clip": shared["strike_phase_per_clip"],
        "motion_kinematics_exact": True,
        "motion_allow_legacy_link_origin_velocity": False,
    }
    for key, expected in expected_scalars.items():
        if contract.get(key) != expected:
            raise ContractError(f"emitted contract {key} mismatch: {contract.get(key)!r}")
    if len(contract.get("joint_names", [])) != 31 or len(contract.get("action_joint_ids", [])) != 31:
        raise ContractError("emitted contract does not bind 31 actions/joints")
    friction = contract.get("joint_friction_coefficients")
    if not isinstance(friction, list) or len(friction) != 31 or any(float(v) != 0.0 for v in friction):
        raise ContractError("emitted contract is not exact 31/31 zero-friction")
    clips = contract.get("motion_clips")
    if not isinstance(clips, list) or [clip.get("sha256") for clip in clips] != [
        inputs["forehand_motion"]["sha256"], inputs["backhand_motion"]["sha256"]
    ]:
        raise ContractError("emitted motion clip order/SHA changed")
    bank = contract.get("question_bank")
    if not isinstance(bank, dict) or bank.get("sha256") != inputs["schema3_train_bank"]["sha256"]:
        raise ContractError("emitted question bank SHA changed")
    if bank.get("schema_version") != 3 or bank.get("split") != "train" or bank.get("exact") is not True:
        raise ContractError("emitted question bank is not exact schema-3 train")
    if contract.get("motion_event_timing") != {"mode": "disabled"}:
        raise ContractError("signed-face funnel must not silently enable T1 timing")

    if hot:
        parent_path = Path(inputs["hot_parent_checkpoint"]["adjacent_training_contract_path"])
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent_keys, current_keys = set(parent), set(contract)
        if parent_keys - current_keys:
            raise ContractError(f"current contract dropped parent keys: {sorted(parent_keys - current_keys)}")
        extras = current_keys - parent_keys
        if extras != EXPECTED_CURRENT_ONLY_KEYS:
            raise ContractError(f"hot transition has an unregistered contract extension: {sorted(extras)}")
        for key in sorted(parent_keys):
            if parent[key] != contract[key]:
                raise ContractError(f"hot transition changed common hard-contract field {key}")
    return sha256_file(contract_path), contract


def emitted_contract_path(log_path: Path, run_name: str) -> Path:
    lines = [
        line.split(CONTRACT_MARKER, 1)[1].strip()
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if CONTRACT_MARKER in line
    ]
    if len(lines) != 1:
        raise ContractError(f"expected one emitted hard-contract marker, got {lines}")
    path = Path(lines[0]).resolve()
    if path.name != "training_contract.json" or path.parent.name != "params":
        raise ContractError(f"invalid emitted hard-contract path: {path}")
    if not path.parent.parent.name.endswith(f"_{run_name}") or not path.is_file():
        raise ContractError("emitted hard-contract path does not bind this run")
    return path


def wait_ready(log_path: Path, state_path: Path, run_name: str, timeout: int, poll: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if FAILURE_RE.search(text):
            raise ContractError(f"{run_name} log contains a hard failure before ready")
        if READY_MARKER in text:
            return
        state = parse_launch_state(state_path)
        pid_raw = state.get("pid", "")
        if not pid_raw.isdigit() or state.get("pgid") != pid_raw:
            raise ContractError(f"{run_name} launch state lost pid==pgid")
        if not process_alive(int(pid_raw)):
            raise ContractError(f"{run_name} exited before first learning iteration")
        time.sleep(poll)
    raise ContractError(f"{run_name} did not reach first learning iteration within {timeout}s")


def verify_guidance_log(log_path: Path, cell: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    weight = str(float(cell["face_guidance_weight"]))
    expected = [
        f"rewards.racket_face_guidance.weight={weight}",
        "rewards.racket_face_guidance.params.theta_max=3.141592653589793",
        "ZERO_FRICTION_RUNTIME_OK",
    ]
    for marker in expected:
        if marker not in text:
            raise ContractError(f"{cell['cell_id']} log does not prove applied value: {marker}")
    if cell["initialization"] == "hot_parent":
        if "explicit hard-contract mismatch override" not in text or "RESUMED from checkpoint" not in text:
            raise ContractError("hot cell did not expose its inexact contract transition and resume")
    elif "RESUMED from checkpoint" in text or "TOLERANT warm-start" in text:
        raise ContractError("fresh cell unexpectedly loaded a checkpoint")


def activation_payload(
    path: Path, manifest: dict[str, Any], config_sha: str, launcher_sha: str
) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) == "":
        raise ContractError("activation file is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("artifact_kind") != "phase1_signed_face_rescue_l1_activation":
        raise ContractError("wrong activation artifact kind")
    content = value.get("content")
    if not isinstance(content, dict):
        raise ContractError("activation content is not an object")
    if value.get("content_sha256") != canonical_sha256(content):
        raise ContractError("activation canonical content SHA mismatch")
    required = {
        "manifest_id": manifest["manifest_id"],
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "training_commit": manifest["source"]["expected_training_commit"],
        "status": "l1_all_four_terminal_l2_blocked_pending_signed_directional_paper",
    }
    for key, expected in required.items():
        if content.get(key) != expected:
            raise ContractError(f"activation {key} mismatch")
    require_sha(content.get("emitted_hard_contract_sha256"), "activation hard contract")
    cells = content.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(EXPECTED_CELLS):
        raise ContractError("activation does not cover A/B/C/D")
    common_sha = content["emitted_hard_contract_sha256"]
    for cell_id, expected_spec in EXPECTED_CELLS.items():
        initialization, guidance, lineage, _role = expected_spec
        cell = cells[cell_id]
        if not isinstance(cell, dict):
            raise ContractError(f"activation cell {cell_id} is not an object")
        expected_terminal = manifest["stages"]["l1"]["expected_terminal_checkpoint_iteration"][cell_id]
        expected_cell = {
            "run_name": manifest["stages"]["l1"]["run_names"][cell_id],
            "initialization": initialization,
            "face_guidance_weight": guidance,
            "expected_lineage_exact": lineage,
            "training_contract_sha256": common_sha,
        }
        for key, expected in expected_cell.items():
            if cell.get(key) != expected:
                raise ContractError(f"activation cell {cell_id} {key} mismatch")
        audit = cell.get("checkpoint_audit")
        if not isinstance(audit, dict):
            raise ContractError(f"activation cell {cell_id} lacks checkpoint audit")
        expected_audit = {
            "iter": expected_terminal,
            "training_contract_schema_version": 3,
            "training_contract_sha256": common_sha,
            "training_contract_lineage_exact": int(lineage),
            "training_contract_provenance_location": "infos",
            "nonfinite_floating_elements": 0,
        }
        for key, expected in expected_audit.items():
            if audit.get(key) != expected:
                raise ContractError(f"activation cell {cell_id} audit {key} mismatch")
        if audit.get("floating_tensor_count", 0) <= 0:
            raise ContractError(f"activation cell {cell_id} has no floating checkpoint tensors")
        checkpoint_path = Path(str(cell.get("checkpoint_path", "")))
        if checkpoint_path.name != f"model_{expected_terminal}.pt":
            raise ContractError(f"activation cell {cell_id} terminal checkpoint path mismatch")
        for key in (
            "checkpoint_sha256", "launch_state_sha256", "training_log_sha256"
        ):
            require_sha(cell.get(key), f"activation cell {cell_id} {key}")
    return value


def runtime_preflight(
    manifest: dict[str, Any], config_path: Path, launcher_path: Path, *,
    config_sha: str, launcher_sha: str, stage_name: str, activation_path: Path | None,
    activation_sha: str | None,
) -> dict[str, Any]:
    if stage_name == "l2":
        raise ContractError(
            "L2 is blocked: freeze a separate immutable signed-face directional checkpoint "
            "paper path/SHA and issue a reviewed v6 activation before launch"
        )
    checkout, wbt = verify_training_source(manifest)
    verify_production_locations(manifest, config_path, launcher_path, checkout)
    runtime = manifest["runtime"]
    python = Path(runtime["isaac_python"])
    locked = wbt / runtime["locked_launcher_relative_path"]
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ContractError("Isaac Python is missing or not executable")
    if not locked.is_file() or not os.access(locked, os.X_OK):
        raise ContractError("locked Kit launcher is missing/not executable")
    training_environment = build_training_environment(manifest, wbt)
    module_path = verify_training_module_resolution(python, wbt, training_environment)
    verified_inputs = verify_inputs(manifest, python)
    verified_inputs["ignored_runtime_asset"] = verify_ignored_runtime_asset(
        manifest, checkout, wbt
    )
    if available_memory_mib() < runtime["minimum_host_available_memory_mib"]:
        raise ContractError("insufficient host memory for the four-cell pool")

    return {
        "checkout": checkout, "wbt": wbt, "python": python, "locked": locked,
        "training_environment": training_environment,
        "training_environment_sha256": canonical_sha256(training_environment),
        "training_module_path": module_path,
        "verified_inputs": verified_inputs, "activation": None,
        "activation_file_sha256": activation_sha,
    }


def stage_run_root(manifest: dict[str, Any], stage_name: str) -> Path:
    return Path(manifest["runtime"]["artifact_root"]) / "runs" / stage_name


def verify_existing_stage_cell(
    manifest: dict[str, Any], preflight: dict[str, Any], *,
    config_sha: str, launcher_sha: str, stage_name: str, cell_id: str,
) -> int | None:
    """Validate one already-claimed cell without writing or signalling.

    A verified live cell may be skipped after an SSH interruption.  A verified
    terminal cell may also be skipped.  Any half-written or failed claim stays
    preserved and blocks automatic retry under the same run name.
    """

    runtime = manifest["runtime"]
    cell = cell_by_id(manifest, cell_id)
    run_name = manifest["stages"][stage_name]["run_names"][cell_id]
    run_dir = stage_run_root(manifest, stage_name) / run_name
    if not run_dir.exists():
        return None
    launch_contract_path = run_dir / runtime["launch_contract_basename"]
    verified_path = run_dir / "runtime_verified.json"
    state_path = run_dir / runtime["launch_state_basename"]
    log_path = run_dir / runtime["training_log_basename"]
    if not all(path.is_file() for path in (
        launch_contract_path, verified_path, state_path, log_path
    )):
        raise ContractError(
            f"{stage_name}/{cell_id} is a partial no-clobber claim; preserve and review it"
        )
    launch_contract = json.loads(launch_contract_path.read_text(encoding="utf-8"))
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    expected_identity = {
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "training_commit": manifest["source"]["expected_training_commit"],
        "stage": stage_name,
        "cell_id": cell_id,
        "run_name": run_name,
        "training_environment_sha256": preflight["training_environment_sha256"],
    }
    for key, expected in expected_identity.items():
        if launch_contract.get(key) != expected:
            raise ContractError(f"existing {stage_name}/{cell_id} launch contract {key} mismatch")
    expected_command = build_command(manifest, stage_name, cell_id, wbt=preflight["wbt"])
    if launch_contract.get("command") != expected_command:
        raise ContractError(f"existing {stage_name}/{cell_id} command differs from preregistration")
    for key, expected in {
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "stage": stage_name,
        "cell_id": cell_id,
        "run_name": run_name,
        "training_environment_sha256": preflight["training_environment_sha256"],
        "training_module_path": preflight["training_module_path"],
    }.items():
        if verified.get(key) != expected:
            raise ContractError(f"existing {stage_name}/{cell_id} runtime {key} mismatch")
    contract_path = Path(verified.get("emitted_hard_contract_path", ""))
    contract_sha, _ = verify_emitted_contract(
        contract_path, manifest, hot=cell["initialization"] == "hot_parent"
    )
    if verified.get("emitted_hard_contract_sha256") != contract_sha:
        raise ContractError(f"existing {stage_name}/{cell_id} hard contract changed")
    if stage_name == "l2":
        activation = preflight["activation"]
        if activation is None or contract_sha != activation["content"]["emitted_hard_contract_sha256"]:
            raise ContractError(f"existing L2/{cell_id} is not bound to the supplied activation")
        if launch_contract.get("l1_activation_sha256") != preflight["activation_file_sha256"]:
            raise ContractError(f"existing L2/{cell_id} launch used a different activation file")
    verify_guidance_log(log_path, cell)
    state = parse_launch_state(state_path)
    pid_raw = state.get("pid", "")
    if not pid_raw.isdigit() or state.get("pgid") != pid_raw:
        raise ContractError(f"existing {stage_name}/{cell_id} state lost pid==pgid")
    pid = int(pid_raw)
    if process_alive(pid):
        cmdline = Path(f"/proc/{pid}/cmdline")
        if not cmdline.is_file() or f"run_name={run_name}".encode() not in cmdline.read_bytes():
            raise ContractError(f"existing {stage_name}/{cell_id} PID identity changed")
        return pid
    terminal = manifest["stages"][stage_name]["expected_terminal_checkpoint_iteration"][cell_id]
    checkpoint = Path(verified.get("training_run_dir", "")) / f"model_{terminal}.pt"
    if not checkpoint.is_file():
        raise ContractError(
            f"existing {stage_name}/{cell_id} exited before terminal; automatic retry is forbidden"
        )
    audit = checkpoint_audit(preflight["python"], checkpoint)
    expected_audit = {
        "iter": terminal,
        "training_contract_schema_version": 3,
        "training_contract_sha256": contract_sha,
        "training_contract_lineage_exact": int(cell["expected_lineage_exact"]),
        "training_contract_provenance_location": "infos",
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise ContractError(f"existing terminal {stage_name}/{cell_id} {key} mismatch")
    return None


def launch_stage(
    manifest: dict[str, Any], config_path: Path, launcher_path: Path, *,
    config_sha: str, launcher_sha: str, stage_name: str, activation_path: Path | None,
    activation_sha: str | None,
) -> None:
    preflight = runtime_preflight(
        manifest, config_path, launcher_path, config_sha=config_sha, launcher_sha=launcher_sha,
        stage_name=stage_name, activation_path=activation_path, activation_sha=activation_sha,
    )
    runtime = manifest["runtime"]
    root = stage_run_root(manifest, stage_name)
    if not root.parent.is_dir():
        raise ContractError(f"artifact runs root must already exist: {root.parent}")
    root.mkdir(exist_ok=True)

    existing_cells: set[str] = set()
    existing_live_pids: set[int] = set()
    for cell_id in ("A", "B", "C", "D"):
        run_dir = root / manifest["stages"][stage_name]["run_names"][cell_id]
        if run_dir.exists():
            existing_cells.add(cell_id)
            pid = verify_existing_stage_cell(
                manifest, preflight, config_sha=config_sha, launcher_sha=launcher_sha,
                stage_name=stage_name, cell_id=cell_id,
            )
            if pid is not None:
                existing_live_pids.add(pid)
    snapshot = gpu_snapshot(runtime["gpu"])
    if not existing_cells and snapshot["compute_pids"]:
        raise ContractError(f"four-cell stage requires an initially empty GPU, found {snapshot['compute_pids']}")
    unrelated = set(snapshot["compute_pids"]) - existing_live_pids
    if existing_cells and unrelated:
        raise ContractError(f"GPU has compute PIDs not owned by verified stage cells: {sorted(unrelated)}")

    for cell_id in ("A", "B", "C", "D"):
        if cell_id in existing_cells:
            print(json.dumps({"status": "existing_verified", "stage": stage_name,
                              "cell": cell_id}, sort_keys=True))
            continue
        cell = cell_by_id(manifest, cell_id)
        run_name = manifest["stages"][stage_name]["run_names"][cell_id]
        run_dir = root / run_name
        run_dir.mkdir(exist_ok=False)
        log_path = run_dir / runtime["training_log_basename"]
        state_path = run_dir / runtime["launch_state_basename"]
        launch_contract_path = run_dir / runtime["launch_contract_basename"]
        command = build_command(manifest, stage_name, cell_id, wbt=preflight["wbt"])
        before = gpu_snapshot(runtime["gpu"])
        if len(before["trainer_pids"]) >= runtime["maximum_trainers_on_gpu"]:
            raise ContractError("GPU already has four trainers before all cells were launched")
        if before["free_memory_mib"] < runtime["minimum_free_gpu_memory_mib_before_each_launch"]:
            raise ContractError("GPU free memory fell below the preregistered launch floor")
        launch_contract = {
            "artifact_kind": "phase1_signed_face_rescue_cell_launch_contract",
            "schema_version": 1,
            "manifest_id": manifest["manifest_id"],
            "manifest_file_sha256": config_sha,
            "launcher_file_sha256": launcher_sha,
            "training_commit": manifest["source"]["expected_training_commit"],
            "stage": stage_name,
            "cell_id": cell_id,
            "causal_role": cell["causal_role"],
            "training_seed": 3,
            "initialization": cell["initialization"],
            "face_guidance_weight": cell["face_guidance_weight"],
            "face_guidance_theta_max": math.pi,
            "expected_lineage_exact": cell["expected_lineage_exact"],
            "run_name": run_name,
            "gpu_snapshot_before": before,
            "verified_inputs": preflight["verified_inputs"],
            "training_environment_sha256": preflight["training_environment_sha256"],
            "training_module_path": preflight["training_module_path"],
            "l1_activation_sha256": activation_sha,
            "command": command,
            "automatic_judge_launch": False,
            "real_robot_commands_forbidden": True,
        }
        write_json_exclusive(launch_contract_path, launch_contract)
        environment = preflight["training_environment"].copy()
        environment.update({
            "KIT_BOOT_MARKER": runtime["kit_boot_marker"],
            "KIT_BOOT_TIMEOUT_S": str(runtime["kit_boot_timeout_seconds"]),
            "KIT_BOOT_POLL_S": str(runtime["poll_seconds"]),
            "KIT_BOOT_STATE_FILE": str(state_path),
        })
        subprocess.run(
            [str(preflight["locked"]), str(log_path), *command],
            cwd=preflight["wbt"], env=environment, check=True,
        )
        contract_path = emitted_contract_path(log_path, run_name)
        contract_sha, _contract = verify_emitted_contract(
            contract_path, manifest, hot=cell["initialization"] == "hot_parent"
        )
        if stage_name == "l2":
            expected_contract_sha = preflight["activation"]["content"]["emitted_hard_contract_sha256"]
            if contract_sha != expected_contract_sha:
                raise ContractError("L2 emitted contract differs from L1 activation")
        wait_ready(
            log_path, state_path, run_name,
            runtime["post_contract_ready_timeout_seconds"], runtime["poll_seconds"],
        )
        verify_guidance_log(log_path, cell)
        state = parse_launch_state(state_path)
        if state.get("pid", "") != state.get("pgid", "") or not state.get("pid", "").isdigit():
            raise ContractError("locked launcher did not record pid==pgid")
        verified = {
            "artifact_kind": "phase1_signed_face_rescue_cell_runtime_verified",
            "schema_version": 1,
            "manifest_file_sha256": config_sha,
            "launcher_file_sha256": launcher_sha,
            "stage": stage_name,
            "cell_id": cell_id,
            "run_name": run_name,
            "pid": int(state["pid"]),
            "pgid": int(state["pgid"]),
            "training_run_dir": str(contract_path.parent.parent),
            "emitted_hard_contract_path": str(contract_path),
            "emitted_hard_contract_sha256": contract_sha,
            "training_environment_sha256": preflight["training_environment_sha256"],
            "training_module_path": preflight["training_module_path"],
            "guidance_applied": True,
            "lineage_expectation": cell["expected_lineage_exact"],
        }
        write_json_exclusive(run_dir / "runtime_verified.json", verified)
        print(json.dumps({"status": "launched_verified", "stage": stage_name, "cell": cell_id,
                          "pid": int(state["pid"]), "contract_sha256": contract_sha}, sort_keys=True))


def finalize_l1(
    manifest: dict[str, Any], config_path: Path, launcher_path: Path, *,
    config_sha: str, launcher_sha: str,
) -> dict[str, Any]:
    preflight = runtime_preflight(
        manifest, config_path, launcher_path, config_sha=config_sha, launcher_sha=launcher_sha,
        stage_name="l1", activation_path=None, activation_sha=None,
    )
    root = stage_run_root(manifest, "l1")
    cell_results: dict[str, Any] = {}
    contract_shas: set[str] = set()
    for cell_id in ("A", "B", "C", "D"):
        cell = cell_by_id(manifest, cell_id)
        run_name = manifest["stages"]["l1"]["run_names"][cell_id]
        run_dir = root / run_name
        verified_path = run_dir / "runtime_verified.json"
        state_path = run_dir / manifest["runtime"]["launch_state_basename"]
        log_path = run_dir / manifest["runtime"]["training_log_basename"]
        if not verified_path.is_file() or not state_path.is_file() or not log_path.is_file():
            raise ContractError(f"L1 {cell_id} runtime evidence is incomplete")
        verified = json.loads(verified_path.read_text(encoding="utf-8"))
        if verified.get("cell_id") != cell_id or verified.get("run_name") != run_name:
            raise ContractError(f"L1 {cell_id} runtime evidence identity mismatch")
        state = parse_launch_state(state_path)
        if state.get("pid") != state.get("pgid") or not state.get("pid", "").isdigit():
            raise ContractError(f"L1 {cell_id} launch state lost pid==pgid")
        if process_alive(int(state["pid"])):
            raise ContractError(f"L1 {cell_id} is still running; finalize is read-only")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if FAILURE_RE.search(text):
            raise ContractError(f"L1 {cell_id} log contains a hard failure signature")
        verify_guidance_log(log_path, cell)
        contract_path = Path(verified["emitted_hard_contract_path"])
        contract_sha, _ = verify_emitted_contract(
            contract_path, manifest, hot=cell["initialization"] == "hot_parent"
        )
        if contract_sha != verified.get("emitted_hard_contract_sha256"):
            raise ContractError(f"L1 {cell_id} emitted contract changed after launch")
        contract_shas.add(contract_sha)
        terminal = manifest["stages"]["l1"]["expected_terminal_checkpoint_iteration"][cell_id]
        checkpoint = Path(verified["training_run_dir"]) / f"model_{terminal}.pt"
        if not checkpoint.is_file():
            raise ContractError(f"L1 {cell_id} terminal checkpoint is missing: {checkpoint}")
        before = checkpoint.stat()
        time.sleep(2.0)
        after = checkpoint.stat()
        if before.st_size <= 0 or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ContractError(f"L1 {cell_id} terminal checkpoint is unstable")
        audit = checkpoint_audit(preflight["python"], checkpoint)
        expected = {
            "iter": terminal,
            "training_contract_schema_version": 3,
            "training_contract_sha256": contract_sha,
            "training_contract_lineage_exact": int(cell["expected_lineage_exact"]),
            "training_contract_provenance_location": "infos",
        }
        for key, value in expected.items():
            if audit.get(key) != value:
                raise ContractError(f"L1 {cell_id} checkpoint {key} mismatch")
        cell_results[cell_id] = {
            "run_name": run_name,
            "initialization": cell["initialization"],
            "face_guidance_weight": cell["face_guidance_weight"],
            "expected_lineage_exact": cell["expected_lineage_exact"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_audit": audit,
            "training_contract_path": str(contract_path),
            "training_contract_sha256": contract_sha,
            "launch_state_sha256": sha256_file(state_path),
            "training_log_sha256": sha256_file(log_path),
        }
    if len(contract_shas) != 1:
        raise ContractError(f"L1 A/B/C/D emitted different hard contracts: {sorted(contract_shas)}")
    content = {
        "manifest_id": manifest["manifest_id"],
        "manifest_file_sha256": config_sha,
        "launcher_file_sha256": launcher_sha,
        "training_commit": manifest["source"]["expected_training_commit"],
        "status": "l1_all_four_terminal_l2_blocked_pending_signed_directional_paper",
        "training_seed": 3,
        "emitted_hard_contract_sha256": next(iter(contract_shas)),
        "cells": cell_results,
        "automatic_judge_launch": False,
        "l2_training_launch_authorized": False,
        "l2_blocked_on": manifest["stages"]["l2"]["blocked_on"],
        "second_seed_authorized": False,
        "stop_or_promote_authorized": False,
        "real_robot_commands_forbidden": True,
    }
    artifact = {
        "artifact_kind": "phase1_signed_face_rescue_l1_activation",
        "schema_version": 1,
        "content": content,
        "content_sha256": canonical_sha256(content),
    }
    output = Path(manifest["stages"]["l1"]["activation_output"])
    write_json_exclusive(output, artifact)
    return {"activation": str(output), "sha256": sha256_file(output), "content": artifact}


def print_plan(manifest: dict[str, Any], stage_name: str, wbt: Path) -> None:
    rows = []
    for cell_id in ("A", "B", "C", "D"):
        cell = cell_by_id(manifest, cell_id)
        rows.append({
            "cell_id": cell_id,
            "causal_role": cell["causal_role"],
            "initialization": cell["initialization"],
            "seed": 3,
            "face_guidance_weight": cell["face_guidance_weight"],
            "run_name": manifest["stages"][stage_name]["run_names"][cell_id],
            "command": build_command(manifest, stage_name, cell_id, wbt=wbt),
        })
    print(json.dumps({"stage": stage_name, "gpu": 0, "four_distinct_cells": True, "cells": rows}, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--stage", choices=("l1", "l2"), default="l1")
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--expected-activation-sha256")
    parser.add_argument("action", choices=("static-validate", "validate", "plan", "launch", "finalize-l1"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    launcher_path = Path(__file__).resolve()
    expected_config_sha = require_sha(args.expected_config_sha256, "expected config hash")
    expected_launcher_sha = require_sha(args.expected_launcher_sha256, "expected launcher hash")
    if not config_path.is_file() or sha256_file(config_path) != expected_config_sha:
        raise ContractError("manifest file SHA mismatch")
    if sha256_file(launcher_path) != expected_launcher_sha:
        raise ContractError("launcher file SHA mismatch")
    manifest = load_manifest(config_path)
    if args.action == "static-validate":
        print(json.dumps({"status": "static_valid", "manifest_id": manifest["manifest_id"],
                          "config_sha256": expected_config_sha,
                          "launcher_sha256": expected_launcher_sha}, sort_keys=True))
        return 0
    preflight = runtime_preflight(
        manifest, config_path, launcher_path,
        config_sha=expected_config_sha, launcher_sha=expected_launcher_sha,
        stage_name=args.stage, activation_path=args.activation,
        activation_sha=args.expected_activation_sha256,
    )
    if args.action == "validate":
        snapshot = gpu_snapshot(manifest["runtime"]["gpu"])
        if snapshot["compute_pids"]:
            raise ContractError(f"reserved GPU is not empty: {snapshot['compute_pids']}")
        print(json.dumps({"status": "runtime_validated_no_writes", "stage": args.stage,
                          "source_commit": manifest["source"]["expected_training_commit"],
                          "gpu": snapshot}, sort_keys=True))
        return 0
    if args.action == "plan":
        print_plan(manifest, args.stage, preflight["wbt"])
        return 0
    if args.action == "launch":
        launch_stage(
            manifest, config_path, launcher_path,
            config_sha=expected_config_sha, launcher_sha=expected_launcher_sha,
            stage_name=args.stage, activation_path=args.activation,
            activation_sha=args.expected_activation_sha256,
        )
        return 0
    if args.action == "finalize-l1":
        if args.stage != "l1":
            raise ContractError("finalize-l1 requires --stage l1")
        result = finalize_l1(
            manifest, config_path, launcher_path,
            config_sha=expected_config_sha, launcher_sha=expected_launcher_sha,
        )
        print(json.dumps({"status": "l1_activation_written", "path": result["activation"],
                          "sha256": result["sha256"]}, sort_keys=True))
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
