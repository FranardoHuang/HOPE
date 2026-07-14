#!/usr/bin/env python3
"""Build one reviewable schema-v2 post-swing capture plan from local exact bytes.

This is an offline snapshot tool.  It never opens SSH, creates a capture/launch
namespace, invokes Hydra, or starts a simulator.  Its output remains inert until
the separately reviewed one-shot controller receives the exact file SHA.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any


SCRIPT = Path(__file__).with_name("run_preregistered_post_swing_capture.py")
SPEC = importlib.util.spec_from_file_location("post_swing_capture_plan_contract", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import infrastructure
    raise RuntimeError("cannot load post-swing capture plan contract")
contract = importlib.util.module_from_spec(SPEC)
prior_dont_write = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    SPEC.loader.exec_module(contract)
finally:
    sys.dont_write_bytecode = prior_dont_write


class PlanBuildError(RuntimeError):
    """The current bytes cannot produce a valid frozen plan."""


def _file_row(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    raw = contract._read_regular_bytes(path, f"plan input {path}")
    value = path.relative_to(relative_to).as_posix() if relative_to is not None else str(path)
    return {"path": value, "bytes": len(raw), "sha256": contract._sha256_bytes(raw)}


def _python_snapshot(requested: Path) -> dict[str, Any]:
    requested = contract._canonical_absolute_path(str(requested), "Python entry")
    current = requested
    visited: set[Path] = set()
    rows: list[dict[str, Any]] = []
    while True:
        if current in visited:
            raise PlanBuildError("Python symlink chain contains a cycle")
        visited.add(current)
        try:
            info = current.lstat()
        except OSError as exc:
            raise PlanBuildError(f"Python chain entry is missing: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(current)
            rows.append({"kind": "symlink", "path": str(current), "target": target})
            current = (
                Path(os.path.normpath(str(current.parent / target)))
                if not Path(target).is_absolute()
                else Path(os.path.normpath(target))
            )
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o111 == 0:
            raise PlanBuildError("Python chain does not terminate in a regular executable")
        rows.append({"kind": "regular", **_file_row(current)})
        break
    if len(rows) < 2:
        raise PlanBuildError("frozen Isaac Python must expose its venv symlink chain")
    pyvenv = requested.parents[1] / "pyvenv.cfg"
    return {
        "requested_path": str(requested),
        "resolved_path": str(current),
        "symlink_chain": rows,
        "pyvenv_cfg": _file_row(pyvenv),
    }


def _parse_pair(value: str, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise PlanBuildError(f"{label} must be KEY=VALUE")
    key, item = value.split("=", 1)
    if not key or not item:
        raise PlanBuildError(f"{label} must have non-empty key and value")
    return key, item


def _runtime_trees(values: list[str]) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        label, remainder = _parse_pair(value, "runtime tree")
        if ":" not in remainder:
            raise PlanBuildError("runtime tree must be LABEL=/absolute/path:on|off")
        path_text, enabled = remainder.rsplit(":", 1)
        if enabled not in {"on", "off"}:
            raise PlanBuildError("runtime tree Python-path flag must be on or off")
        path = contract._canonical_absolute_path(path_text, f"runtime tree {label}")
        rows.append(
            {
                "label": label,
                "path": str(path),
                "on_pythonpath": enabled == "on",
                **contract._inventory(path, skip_git=True),
            }
        )
    if {row["label"] for row in rows} != set(contract.RUNTIME_TREE_LABELS) or len(rows) != len(
        contract.RUNTIME_TREE_LABELS
    ):
        raise PlanBuildError("runtime trees must contain each required label exactly once")
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    template_raw = contract._read_regular_bytes(args.template_plan, "template plan")
    template = contract._strict_json_loads(template_raw, "template plan")
    source = contract._canonical_absolute_path(str(args.capture_source_checkout), "capture source")
    output = contract._canonical_absolute_path(str(args.output), "plan output")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise PlanBuildError("plan output must stay outside the immutable capture source tree")
    git_row = contract._verify_executable_row(args.git, _file_row(args.git), "git executable")
    nvidia_row = contract._verify_executable_row(
        args.nvidia_smi, _file_row(args.nvidia_smi), "nvidia-smi executable"
    )
    git_path = Path(git_row["path"])
    head = contract._git_output(git_path, source, "rev-parse", "HEAD")
    if contract._git_output(git_path, source, "status", "--porcelain=v1", "--untracked-files=no"):
        raise PlanBuildError("capture source has tracked changes")
    files = {
        "controller": _file_row(source / "scripts/run_preregistered_post_swing_capture.py", relative_to=source),
        "inference_runner": _file_row(
            source / "hope_training/whole_body_tracking/scripts/play.py", relative_to=source
        ),
        "lean_queue_runtime": _file_row(
            source / "hope_training/whole_body_tracking/scripts/lean_queue_runtime.py", relative_to=source
        ),
        "producer": _file_row(
            source
            / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
            relative_to=source,
        ),
        "attestor": _file_row(source / "scripts/attest_post_swing_teacher.py", relative_to=source),
    }
    old_source = contract._require_mapping(template["capture_source"], "template capture source")
    old_asset = contract._require_mapping(old_source["ignored_runtime_asset"], "template ignored asset")
    asset_relative = old_asset.get("relative_path") or old_asset.get("target_relative_path")
    asset_relative = contract._canonical_relative_path(asset_relative, "ignored asset relative path")
    asset_path = source / asset_relative
    asset = {
        "relative_path": asset_relative.as_posix(),
        **contract._inventory(asset_path),
        "symlinks_forbidden": True,
    }
    machine_id = contract.MACHINE_ID_PATH
    boot_id = contract.BOOT_ID_PATH
    teacher_source = contract._require_mapping(template["teacher_checkpoint"], "template teacher")
    teacher = {
        key: teacher_source[key]
        for key in (
            "path", "sha256", "embedded_iteration", "floating_elements",
            "nonfinite_floating_elements", "fresh_lineage", "training_source_commit",
            "hard_contract", "launch_claim", "run_binding", "milestone_receipt",
        )
    }
    environment = dict(_parse_pair(value, "environment") for value in args.env)
    if "PATH" not in environment:
        raise PlanBuildError("at least one --env PATH=... row is required")
    plan_id = args.plan_id
    plan = {
        "schema_version": 2,
        "plan_id": plan_id,
        "status": "preregistered_capture_not_started",
        "human_owner": template.get("human_owner", "UNASSIGNED"),
        "executor": template.get("executor", "Codex"),
        "purpose": template.get("purpose", "external natural-wrap teacher capture"),
        "simulation_only": True,
        "capture_source": {
            "checkout": str(source),
            "commit": head,
            "clean_required": True,
            "files": files,
            "ignored_runtime_asset": asset,
            "full_tree": contract._inventory(source, skip_git=True),
        },
        "teacher_checkpoint": teacher,
        "ordered_motion_inputs": template["ordered_motion_inputs"],
        "question_bank": template["question_bank"],
        "capture_contract": {
            "pod": "pod2",
            "gpu": 2,
            "gpu_uuid": args.gpu_uuid,
            "cuda_visible_devices": args.gpu_uuid,
            "runtime_device": "cuda:0",
            "num_envs": 4096,
            "target_count": 4096,
            "max_inference_steps": 20000,
            "seed": args.seed,
            "wrap_teleport": False,
            "post_swing_start_prob": 0.25,
            "root_linear_velocity_limit_mps": 2.0,
            "root_angular_velocity_limit_radps": 4.0,
            "namespace_id": plan_id,
            "output_directory": str(contract.CAPTURE_PARENT / plan_id),
            "launch_root": str(contract.LAUNCH_PARENT / plan_id),
            "output_must_be_absent_before_one_shot": True,
            "capture_is_inference_only": True,
            "ppo_updates": 0,
            "natural_wrap_only": True,
            "timeout_or_failure_reset_states_forbidden": True,
            "launch_handoff": "execve_same_pid_v1",
        },
        "runtime_environment": {
            "node": {
                "hostname": args.hostname,
                "machine_id_path": str(machine_id),
                "machine_id_sha256": contract._sha256_file(machine_id),
                "boot_id_path": str(boot_id),
                "boot_id_sha256": contract._sha256_file(boot_id),
            },
            "gpu": {
                "physical_index": 2,
                "uuid": args.gpu_uuid,
                "lease_path": str(contract.GPU_LEASE_PATH),
            },
            "python": _python_snapshot(args.python),
            "runtime_trees": _runtime_trees(args.runtime_tree),
            "tools": {"git": git_row, "nvidia_smi": nvidia_row},
            "environment": {"exact": environment},
            "compose_timeout_s": args.compose_timeout_s,
        },
        "runtime_recipe_derivation": {
            "source": "teacher_checkpoint.run_binding exact training_argv",
            "keep_all_task_motion_bank_seed_num_env_overrides": True,
            "deduplicate_identical_hydra_keys": True,
            "replace_executable_train_with_play": True,
            "remove_keys": sorted(contract.EXPECTED_REMOVE_KEYS),
            "add_keys": sorted(contract.EXPECTED_ADD_KEYS),
            "runtime_hard_contract_must_equal_teacher_checkpoint_hard_contract_before_first_state": True,
            "seed_must_be_applied_by_play": True,
        },
        "authorization": {
            "capture_authorized": True,
            "attestation_authorized_only_after_complete_capture": True,
            "first_reset_probe_authorized": False,
            "scientific_training_authorized": False,
            "second_seed_authorized": False,
            "judge_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        },
        "failure_policy": {
            "preserve_partial_namespace": True,
            "same_namespace_retry_forbidden": True,
            "automatic_retry_forbidden": True,
            "exact_numeric_process_group_only": True,
            "pod1_and_pod2_gpu0_forbidden": True,
        },
    }
    contract._validate_plan(plan)
    return plan


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--template-plan", type=Path, required=True)
    value.add_argument("--capture-source-checkout", type=Path, required=True)
    value.add_argument("--plan-id", required=True)
    value.add_argument("--gpu-uuid", required=True)
    value.add_argument("--hostname", default=socket.gethostname())
    value.add_argument("--python", type=Path, default=contract.ISAAC_PYTHON)
    value.add_argument("--git", type=Path, required=True)
    value.add_argument("--nvidia-smi", type=Path, required=True)
    value.add_argument("--runtime-tree", action="append", default=[], required=True)
    value.add_argument("--env", action="append", default=[], required=True)
    value.add_argument("--seed", type=int, default=3)
    value.add_argument("--compose-timeout-s", type=int, default=120)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = build(args)
        raw = contract._canonical_bytes(plan) + b"\n"
        parent_fd = contract._ensure_real_directory(args.output.parent)
        os.close(parent_fd)
        contract._exclusive_write(args.output, raw)
    except (PlanBuildError, contract.CaptureContractError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"path": str(args.output), "sha256": contract._sha256_bytes(raw)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
