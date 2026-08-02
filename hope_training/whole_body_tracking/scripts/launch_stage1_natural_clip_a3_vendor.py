#!/usr/bin/env python3
"""Launch one of the three code-owned ball-free Stage-1 V2 natural-clip lanes.

This launcher is intentionally independent of the ActionBall launch stack.  It
has no operator-supplied motion, phase, seed, task, or training-budget override:
all scientific identity comes from
``whole_body_tracking.tasks.tracking.stage1_natural_clip_contract``.  The only
placement controls are the stage, reviewed lane, namespace root, and physical
GPU.

``--dry-run`` prints one canonical JSON document containing the immutable spec
and exact ``train.py`` argv without creating a namespace.  A live launch first
claims ``<root>/<run_name>`` with exclusive-create semantics, stores the same
canonical document there, maps the selected physical GPU to logical
``cuda:0``, and directly execs ``train.py``.  A claimed namespace is permanently
spent even when exec or training later fails.

The launcher binds only the versioned VendorV2 profile (225-D actor, 318-D critic and dense
full-phase official-paddle learning); the historical VendorV1 profile is not launchable from
current source.  Every run is diagnostic-only.  It does not authorize promotion, resume,
export, deployment, judging, or hardware use.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union


SCHEMA_VERSION = 2
SPEC_KIND = "stage1_natural_clip_a3_vendor_launch_spec_v2"
TASK_PROFILE_ID = "HOPEPingPongStage1NaturalClipA3VendorV2"
EXPERIMENT_NAME = "agibot_a3_stage1_natural_clip_v2"
DIAGNOSTIC_SUFFIX = "diagnostic_unauthorized"

_THIS_FILE = Path(__file__).resolve()
_WBT_ROOT = _THIS_FILE.parents[1]
_REPO_ROOT = _WBT_ROOT.parents[1]
_TRAIN_FILE = _WBT_ROOT / "scripts/train.py"
_TASK_FILE = _WBT_ROOT / f"cfg/task/{TASK_PROFILE_ID}.yaml"
_CONTRACT_FILE = (
    _WBT_ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/"
    "stage1_natural_clip_contract.py"
)


class LaunchRefused(RuntimeError):
    """A fail-closed launcher validation error."""


def _load_contract_module():
    module_name = (
        "whole_body_tracking.tasks.tracking.stage1_natural_clip_contract"
    )
    spec = importlib.util.spec_from_file_location(module_name, _CONTRACT_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise LaunchRefused(
            f"cannot load Stage-1 natural-clip contract: {_CONTRACT_FILE}"
        )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules while the
    # module executes.  The dependency-free contract does not import its
    # package, so this remains host-safe without importing Isaac Lab.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_C = _load_contract_module()
STAGE1_NATURAL_CLIP_LANES = _C.STAGE1_NATURAL_CLIP_LANES
STAGE1_NATURAL_CLIP_LANES_BY_ID = _C.STAGE1_NATURAL_CLIP_LANES_BY_ID

EXACT_STAGE_BUDGETS: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    {
        "smoke": (1, 2, 1),
        "probe": (4096, 5, 1),
        "long": (4096, 20_001, 100),
    }
)
LANE_SEEDS: Mapping[str, int] = MappingProxyType(
    {
        lane.lane_id: seed
        for seed, lane in enumerate(STAGE1_NATURAL_CLIP_LANES)
    }
)


def _validate_code_owned_tables() -> None:
    lanes = tuple(STAGE1_NATURAL_CLIP_LANES)
    if len(lanes) != 3:
        raise LaunchRefused(
            "Stage-1 launcher requires exactly three code-owned lanes"
        )
    if set(STAGE1_NATURAL_CLIP_LANES_BY_ID) != {
        lane.lane_id for lane in lanes
    }:
        raise LaunchRefused("Stage-1 lane id index differs from the lane table")
    for field in ("lane_id", "action_id", "motion_path", "motion_sha256"):
        values = tuple(getattr(lane, field) for lane in lanes)
        if len(values) != len(set(values)):
            raise LaunchRefused(
                f"Stage-1 code-owned lanes repeat {field}: {values!r}"
            )
    phases = tuple(lane.strike_phase for lane in lanes)
    if len(phases) != len(set(phases)):
        raise LaunchRefused(
            f"Stage-1 code-owned lanes repeat strike_phase: {phases!r}"
        )
    if set(LANE_SEEDS) != {lane.lane_id for lane in lanes}:
        raise LaunchRefused("Stage-1 lane seed table is incomplete")
    if len(set(LANE_SEEDS.values())) != len(lanes):
        raise LaunchRefused("Stage-1 lanes must have independent seeds")


_validate_code_owned_tables()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise LaunchRefused(f"cannot read reviewed motion: {path}") from exc
    return digest.hexdigest()


def _plain_gpu(value: object) -> int:
    if type(value) is not int or value < 0:
        raise LaunchRefused("gpu must be a non-negative plain integer")
    return value


def _lane_identity(lane_id: str) -> dict[str, Any]:
    try:
        lane = STAGE1_NATURAL_CLIP_LANES_BY_ID[lane_id]
    except (KeyError, TypeError) as exc:
        raise LaunchRefused(
            "lane must be exactly one of: "
            + ", ".join(sorted(STAGE1_NATURAL_CLIP_LANES_BY_ID))
        ) from exc
    motion = (_REPO_ROOT / lane.motion_path).resolve()
    try:
        motion.relative_to(_REPO_ROOT)
    except ValueError as exc:  # pragma: no cover - contract is code-owned
        raise LaunchRefused("reviewed motion escapes the repository root") from exc
    if not motion.is_file():
        raise LaunchRefused(f"reviewed motion is missing: {motion}")
    actual_sha256 = _sha256_file(motion)
    if actual_sha256 != lane.motion_sha256:
        raise LaunchRefused(
            "reviewed motion SHA-256 mismatch: "
            f"lane={lane.lane_id} expected={lane.motion_sha256} "
            f"actual={actual_sha256}"
        )
    return {
        "lane_id": lane.lane_id,
        "action_id": lane.action_id,
        "side": lane.side,
        "motion_path": lane.motion_path,
        "motion_absolute_path": str(motion),
        "motion_sha256": lane.motion_sha256,
        "frame_count": lane.frame_count,
        "strike_frame": lane.strike_frame,
        "strike_phase": lane.strike_phase,
        "cycle_seconds": lane.cycle_seconds,
    }


def _run_name(*, lane_id: str, seed: int, stage: str) -> str:
    return (
        f"stage1_natural_clip_v2_a3_vendor_{lane_id}_seed{seed}_{stage}_"
        f"{DIAGNOSTIC_SUFFIX}"
    )


def build_launch_payload(
    *,
    stage: str,
    lane_id: str,
    root: Union[Path, str],
    gpu: int,
    python_executable: Optional[Union[Path, str]] = None,
) -> dict[str, Any]:
    """Build and validate one stable, side-effect-free spec/argv document."""

    if stage not in EXACT_STAGE_BUDGETS:
        raise LaunchRefused(
            "stage must be exactly one of: "
            + ", ".join(EXACT_STAGE_BUDGETS)
        )
    gpu = _plain_gpu(gpu)
    lane = _lane_identity(lane_id)
    seed = LANE_SEEDS[lane_id]
    num_envs, max_iterations, save_interval = EXACT_STAGE_BUDGETS[stage]

    namespace_root = Path(root).expanduser().resolve()
    if namespace_root == Path(namespace_root.anchor):
        raise LaunchRefused("root may not be the filesystem root")
    run_name = _run_name(lane_id=lane_id, seed=seed, stage=stage)
    namespace = namespace_root / run_name
    # Preserve a venv's executable path.  Resolving the symlink can turn
    # ``/path/to/venv/bin/python`` into the base interpreter and make execvpe
    # silently leave the exact runtime environment.
    python = Path(
        sys.executable if python_executable is None else python_executable
    ).expanduser().absolute()
    if not python.is_file():
        raise LaunchRefused(f"Python executable is missing: {python}")
    for source, label in (
        (_TRAIN_FILE, "train.py"),
        (_TASK_FILE, "Stage-1 task profile"),
        (_CONTRACT_FILE, "Stage-1 lane contract"),
    ):
        if not source.is_file():
            raise LaunchRefused(f"{label} is missing: {source}")

    # Hydra receives real filesystem characters.  The surrounding canonical
    # launch document still escapes them deterministically when serialized.
    motion_list = json.dumps(
        [lane["motion_absolute_path"]],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    clip_names = json.dumps(
        [lane["action_id"]],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    argv = [
        str(python),
        str(_TRAIN_FILE),
        f"task={TASK_PROFILE_ID}",
        "algo=ppo",
        "headless=true",
        "logger=tensorboard",
        "video=false",
        "device=cuda:0",
        "algo.policy.init_noise_std=0.02",
        "algo.policy.noise_std_type=log",
        f"seed={seed}",
        f"num_envs={num_envs}",
        f"max_iterations={max_iterations}",
        f"algo.runner.save_interval={save_interval}",
        f"run_name={run_name}",
        f"task.experiment_name={EXPERIMENT_NAME}",
        f"motion_file={motion_list}",
        "motion_file_2=null",
        "registry_name=null",
        "registry_name_2=null",
        "task.registry_name=null",
        "task.registry_name_2=null",
        f"task.racket.clip_names={clip_names}",
        "task.racket.action_ball_diagnostic_unauthorized=true",
        f"hydra.run.dir={namespace / 'hydra'}",
    ]
    spec = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "task_profile": TASK_PROFILE_ID,
        "experiment_name": EXPERIMENT_NAME,
        "stage": stage,
        "budget": {
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "save_interval": save_interval,
        },
        "lane": lane,
        "seed": seed,
        "physical_gpu": gpu,
        "logical_device": "cuda:0",
        "cuda_visible_devices": str(gpu),
        "namespace_root": str(namespace_root),
        "namespace": str(namespace),
        "run_name": run_name,
        "registry_name": None,
        "registry_name_2": None,
        "diagnostic_unauthorized": True,
        "authorization": {
            "training": False,
            "promotion": False,
            "resume": False,
            "export": False,
            "deployment": False,
            "judge": False,
            "hardware": False,
        },
    }
    return {"argv": argv, "spec": spec}


def _trainer_output_matches(run_name: str) -> tuple[Path, ...]:
    experiment_root = _WBT_ROOT / "logs/rsl_rl" / EXPERIMENT_NAME
    if not experiment_root.is_dir():
        return ()
    suffix = "_" + run_name
    return tuple(
        sorted(
            path
            for path in experiment_root.iterdir()
            if path.is_dir()
            and (path.name == run_name or path.name.endswith(suffix))
        )
    )


def assert_fresh(payload: Mapping[str, Any]) -> None:
    spec = payload["spec"]
    namespace = Path(spec["namespace"])
    if namespace.exists():
        raise LaunchRefused(f"namespace is already spent: {namespace}")
    matches = _trainer_output_matches(spec["run_name"])
    if matches:
        raise LaunchRefused(
            "trainer run_name is already spent: "
            + ", ".join(str(path) for path in matches)
        )


def claim_namespace(payload: Mapping[str, Any]) -> Path:
    """Atomically spend one namespace and persist the exact launch payload."""

    assert_fresh(payload)
    namespace = Path(payload["spec"]["namespace"])
    try:
        namespace.parent.mkdir(parents=True, exist_ok=True)
        namespace.mkdir(mode=0o750, exist_ok=False)
        spec_path = namespace / "launch_spec_and_argv.v2.json"
        with spec_path.open("x", encoding="ascii") as stream:
            stream.write(canonical_json(payload) + "\n")
    except OSError as exc:
        raise LaunchRefused(
            f"cannot exclusively claim fresh namespace: {namespace}"
        ) from exc
    return namespace


def exec_training(payload: Mapping[str, Any]) -> None:
    claim_namespace(payload)
    document = canonical_json(payload)
    print(document, flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = payload["spec"]["cuda_visible_devices"]
    argv = list(payload["argv"])
    try:
        os.chdir(_WBT_ROOT)
        os.execvpe(argv[0], argv, env)
    except OSError as exc:
        # The namespace remains spent by design: reusing a name after a partial
        # launch would make operational provenance ambiguous.
        raise LaunchRefused(f"cannot exec train.py: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=tuple(EXACT_STAGE_BUDGETS)
    )
    parser.add_argument(
        "--lane",
        required=True,
        choices=tuple(lane.lane_id for lane in STAGE1_NATURAL_CLIP_LANES),
    )
    parser.add_argument(
        "--root",
        required=True,
        help="dedicated no-clobber namespace root",
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="physical GPU exposed to the trainer as logical cuda:0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print canonical spec/argv JSON without claiming or execing",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = build_launch_payload(
            stage=args.stage,
            lane_id=args.lane,
            root=args.root,
            gpu=args.gpu,
        )
        assert_fresh(payload)
        if args.dry_run:
            print(canonical_json(payload))
            return 0
        exec_training(payload)
        return 0  # pragma: no cover - os.execvpe replaces the process
    except LaunchRefused as exc:
        print(
            canonical_json(
                {
                    "kind": "stage1_natural_clip_launch_refused_v2",
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
