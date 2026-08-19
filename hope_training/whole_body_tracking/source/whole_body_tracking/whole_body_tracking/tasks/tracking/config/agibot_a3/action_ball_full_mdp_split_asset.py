"""Code-owned verifier for the fresh full-MDP split-rubber robot asset.

The offline producer owns deterministic reconstruction from the enclosed
reviewed source bundle, URDF and STL files.  This consumer verifies the actual
runtime-selected directory with that tracked producer, so a private immutable
run snapshot and the shared source directory have identical semantics when and
only when their bytes reconstruct to the reviewed model.  Neither a producer
receipt nor the value returned by ``check`` authorizes a run.  Live PhysX
contact evidence remains a separate scene/runtime responsibility.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType


ACTION_BALL_FULL_MDP_SPLIT_ASSET_ROOT = Path(
    "/workspace/franco/runtime_assets/a3p0807_split_rubber_diagnostic_v3"
)
ACTION_BALL_FULL_MDP_SPLIT_ASSET_MODEL = (
    ACTION_BALL_FULL_MDP_SPLIT_ASSET_ROOT / "model.usd"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[9]
_PRODUCER_SOURCE = (
    _REPOSITORY_ROOT / "scripts" / "derive_a3p0807_split_rubber_usd.py"
)
_PRODUCER_MODULE_NAME = "_action_ball_full_mdp_split_asset_producer"


class ActionBallFullMdpSplitAssetError(RuntimeError):
    """The selected robot is not the exact reconstructed diagnostic asset."""


@dataclass(frozen=True)
class ActionBallFullMdpExpectedColliderMesh:
    """Clone-only composed-Mesh expectation rebuilt from enclosed sources."""

    name: str
    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    translate_in_wrist_m: tuple[float, float, float]


@dataclass(frozen=True)
class ActionBallFullMdpExpectedColliderGeometry:
    """Exact four-collider geometry expected in every composed live wrist."""

    meshes: tuple[ActionBallFullMdpExpectedColliderMesh, ...]


def _require_real_directory(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise ActionBallFullMdpSplitAssetError(f"missing {label}: {path}") from exc
    if not path.is_dir() or path.is_symlink():
        raise ActionBallFullMdpSplitAssetError(
            f"{label} must be one real non-symlink directory: {path}"
        )


def _require_regular_file(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise ActionBallFullMdpSplitAssetError(f"missing {label}: {path}") from exc
    if not path.is_file() or path.is_symlink():
        raise ActionBallFullMdpSplitAssetError(
            f"{label} must be one regular non-symlink file: {path}"
        )


def _load_current_producer_module() -> ModuleType:
    """Load the tracked producer directly; no caller may nominate a verifier."""

    _require_regular_file(_PRODUCER_SOURCE, "tracked split-asset producer")
    spec = importlib.util.spec_from_file_location(
        _PRODUCER_MODULE_NAME, _PRODUCER_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ActionBallFullMdpSplitAssetError(
            f"cannot load tracked split-asset producer: {_PRODUCER_SOURCE}"
        )
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(_PRODUCER_MODULE_NAME)
    sys.modules[_PRODUCER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ActionBallFullMdpSplitAssetError(
            "tracked split-asset producer cannot be imported"
        ) from exc
    finally:
        if previous is None:
            sys.modules.pop(_PRODUCER_MODULE_NAME, None)
        else:
            sys.modules[_PRODUCER_MODULE_NAME] = previous
    return module


def _selected_split_asset_paths() -> tuple[Path, Path]:
    """Return one canonical, real runtime-selected root and model path."""

    selected = os.environ.get("HOPE_AGIBOT_A3_USD_PATH")
    if not selected:
        raise ActionBallFullMdpSplitAssetError(
            "fresh full-MDP requires an absolute runtime-selected model.usd"
        )
    model = Path(selected)
    if not model.is_absolute() or model.name != "model.usd":
        raise ActionBallFullMdpSplitAssetError(
            "fresh full-MDP runtime selection must be an absolute model.usd"
        )
    root = model.parent
    _require_real_directory(root, "split-asset root")
    _require_regular_file(model, "split-asset model.usd")
    try:
        resolved_model = model.resolve(strict=True)
    except OSError as exc:
        raise ActionBallFullMdpSplitAssetError(
            f"split-asset model.usd cannot be resolved: {model}"
        ) from exc
    if resolved_model != model:
        raise ActionBallFullMdpSplitAssetError(
            "split-asset model selection must be canonical and contain no symlink"
        )
    return root, model


def require_action_ball_full_mdp_split_asset() -> str:
    """Verify and return the actual fresh diagnostic model path.

    This no-argument API deliberately has no expected-path, pin, receipt,
    checker, or verdict injection surface.  Success means only that the
    selected canonical path exists and the current tracked producer
    independently reconstructs its bytes from the actual enclosed sources.
    The producer's returned telemetry is intentionally ignored.
    """

    selected_root, selected_model = _selected_split_asset_paths()

    producer = _load_current_producer_module()
    check = getattr(producer, "check", None)
    if not callable(check):
        raise ActionBallFullMdpSplitAssetError(
            "tracked split-asset producer lacks callable check(output_root)"
        )
    try:
        # Do not inspect or propagate the return value.  It is recomputed
        # telemetry, not a capability, receipt, or launch verdict.
        check(selected_root)
    except Exception as exc:
        raise ActionBallFullMdpSplitAssetError(
            "v3 split-rubber asset failed enclosed-source reconstruction"
        ) from exc
    return str(selected_model)


def action_ball_full_mdp_expected_collider_geometry(
) -> ActionBallFullMdpExpectedColliderGeometry:
    """Rebuild the four live Mesh expectations from the code-owned v3 source.

    This no-argument clone is not a capability or a run verdict.  The scene
    installer compares it against the independently composed live stage before
    subscribing, so a stronger USD layer cannot silently swap equally named
    red/black/handle geometry after the offline asset check.
    """

    selected_root = Path(require_action_ball_full_mdp_split_asset()).parent
    producer = _load_current_producer_module()
    source_geometry = getattr(producer, "_source_geometry", None)
    if not callable(source_geometry):
        raise ActionBallFullMdpSplitAssetError(
            "tracked split-asset producer lacks enclosed-source geometry rebuild"
        )
    try:
        urdf_facts, output_meshes, _evidence = source_geometry(
            source_urdf=(
                selected_root
                / "source"
                / "urdf"
                / "model.urdf"
            ),
            source_mesh_root=(
                selected_root / "source" / "meshes"
            ),
        )
        translations = urdf_facts["translations"]
    except Exception as exc:
        raise ActionBallFullMdpSplitAssetError(
            "v3 enclosed collider geometry reconstruction failed"
        ) from exc
    source_by_output = {
        "wrist_shell_collider": "right_wrist_yaw_link",
        "black_rubber_collider": "pingpang_black_link",
        "red_rubber_collider": "pingpang_red_link",
        "racket_handle_collider": "right_hand_pingpang_link",
    }
    meshes = []
    try:
        for name in (
            "wrist_shell_collider",
            "black_rubber_collider",
            "red_rubber_collider",
            "racket_handle_collider",
        ):
            triangles = output_meshes[name]
            points = tuple(
                tuple(float(value) for value in point)
                for triangle in triangles
                for point in triangle
            )
            meshes.append(
                ActionBallFullMdpExpectedColliderMesh(
                    name=name,
                    points=points,
                    face_vertex_counts=(3,) * len(triangles),
                    face_vertex_indices=tuple(range(len(points))),
                    translate_in_wrist_m=tuple(
                        float(value)
                        for value in translations[source_by_output[name]]
                    ),
                )
            )
    except Exception as exc:
        raise ActionBallFullMdpSplitAssetError(
            "v3 enclosed collider geometry projection differs"
        ) from exc
    return ActionBallFullMdpExpectedColliderGeometry(meshes=tuple(meshes))


__all__ = (
    "ActionBallFullMdpExpectedColliderGeometry",
    "ActionBallFullMdpExpectedColliderMesh",
    "ACTION_BALL_FULL_MDP_SPLIT_ASSET_MODEL",
    "ACTION_BALL_FULL_MDP_SPLIT_ASSET_ROOT",
    "ActionBallFullMdpSplitAssetError",
    "action_ball_full_mdp_expected_collider_geometry",
    "require_action_ball_full_mdp_split_asset",
)
