#!/usr/bin/env python3
"""Isolated pinned-byte launcher for the formal fitted MuJoCo Gate.

Supply the committed bytes through external ``git show`` to
``python -I -S -B -``.  It imports only the standard library until it has
verified one exact clean Git commit, the fixed preregistered trust spec,
NumPy/MuJoCo dependency trees, and the complete repository-local runtime
source/data closure.  Repository modules are then compiled directly from the
already hashed bytes through a custom loader; timestamp/hash pyc files are
never consulted.

The operator's fresh staging directory must live under the fixed ignored
artifact store.  After the core returns, the bootstrap derives a final
content-addressed capsule id from the commit, motion/solver/physics/geometry
pins and every output byte, reserves that final id without clobbering, repairs
the detached worktree metadata, seals the tree read-only, and writes one
retained receipt through a pre-reserved descriptor.

The core Gate refuses formal execution without the attestation produced here.
This launcher grants no deployment or hardware authorization.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CORE_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate.py"
)
BOOTSTRAP_REPO_PATH = (
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_fitted_ball_gate_bootstrap.py"
)
SOURCE_PATHS = {
    "hope_training/whole_body_tracking/scripts/"
    "action_ball_action_set_contract.py": (
        HERE / "action_ball_action_set_contract.py"
    ),
    BOOTSTRAP_REPO_PATH: Path(__file__).resolve(),
    CORE_REPO_PATH: HERE / "mujoco_teacher_motion_fitted_ball_gate.py",
    "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_native_ball_diagnostic.py": (
        HERE / "mujoco_teacher_motion_native_ball_diagnostic.py"
    ),
    "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py": (
        HERE / "mujoco_motion_player.py"
    ),
    "hope_training/whole_body_tracking/scripts/audit_motion_npz.py": (
        HERE / "audit_motion_npz.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "motion_kinematics_contract.py": (
        HERE / "motion_kinematics_contract.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "racket_geometry_contract.py": (
        HERE / "racket_geometry_contract.py"
    ),
    "hope_training/whole_body_tracking/scripts/"
    "canonical_mujoco_identity.py": (
        HERE / "canonical_mujoco_identity.py"
    ),
    "hope_training/ball_physics_fit/contact_model.py": (
        REPO_ROOT / "hope_training/ball_physics_fit/contact_model.py"
    ),
    "scripts/mujoco_table_scene.py": (
        REPO_ROOT / "scripts/mujoco_table_scene.py"
    ),
    "scripts/audit_motion_schema2_table_net_clearance.py": (
        REPO_ROOT
        / "scripts/audit_motion_schema2_table_net_clearance.py"
    ),
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/geometry.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/geometry.py"
    ),
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/table_frame.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/table_frame.py"
    ),
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/"
        "racket_contact_geometry.py"
    ),
}
DATA_PATHS = {
    "configs/a3_runtime_body_order.txt": (
        REPO_ROOT / "configs/a3_runtime_body_order.txt"
    ),
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/hope_commands.py": (
        REPO_ROOT
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
    ),
}
CORE_MODULE_NAME = "_fitted_ball_gate_core"
TRUST_SPEC_REPO_PATH = (
    "configs/mujoco_fitted_ball_pre_registered_launch_v2.json"
)
TRUST_SPEC_SCHEMA_VERSION = 2
TRUST_SPEC_ARTIFACT_TYPE = (
    "mujoco_fitted_ball_pre_registered_dual_manifest_launch_v2"
)
PHYSICAL_GATE_MATERIALIZATION_KIND = (
    "fresh_n5_disposable_physical_gate_manifest_materialization_v1"
)
GENERIC_PHYSICAL_GATE_MATERIALIZATION_KIND = (
    "action_ball_disposable_physical_gate_manifest_materialization_v2"
)
GATE_TOP_LEVEL_FIELDS = (
    "racket_geometry_contract",
    "physical_contact_contract",
)
GATE_ACTION_FIELDS = (
    "physical_ball_launch",
    "physical_task_binding",
    "admission",
)
ACTION_SET_PROFILE = "fresh_upper_nomove_n5_v3"
LEGACY_FRESH_N5_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)
MATERIALIZATION_PROFILE_CENTER_KEYS = (
    "contact_offset_center_b_yaw_m",
    "time_to_contact_center_s",
    "incoming_direction_center_b_yaw",
    "incoming_speed_center_mps",
    "spin_direction_center_b_yaw",
    "spin_magnitude_center_radps",
    "base_spawn_center_w_xy_m",
    "base_travel_center_b_yaw_xy_m",
)
MATERIALIZATION_ACTION_IDENTITY_KEYS = frozenset(
    {
        "action_id",
        "action_uid",
        "family",
        "motion_path",
        "motion_sha256",
        "scope",
        "profile_center",
        "profile_center_sha256",
    }
)
FIVE_SOLID_OBSTACLE_ORDER = (
    "motion_table_top",
    "motion_table_robot_keepout",
    "motion_net",
    "motion_net_post_left",
    "motion_net_post_right",
)
FIVE_SOLID_KEEPOUT_NAME = "motion_table_robot_keepout"
FLOOR_GEOM_NAME = "floor"
LEGAL_FOOT_BODY_NAMES = (
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
)
FOOT_FLOOR_PENETRATION_TOLERANCE_M = 0.002
NONFOOT_FLOOR_PENETRATION_TOLERANCE_M = 0.0001
NONFOOT_GROUND_CLEARANCE_GUARD_M = 0.0005
GROUND_DISTANCE_QUERY_CAP_M = 0.01
GROUND_CONTACT_FORCE_THRESHOLD_N = 1.0e-6
ACTION_SET_CONTRACT_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "expected_n",
        "scope",
        "mobility_mode",
        "ordered_action_ids",
        "ordered_action_uids",
        "order_uid_digest_sha256",
        "manifest_path",
        "manifest_sha256",
        "experiment_name",
        "actor_obs_contract",
        "actor_obs_width",
        "namespace_identity",
        "contract_sha256",
    }
)
CAPSULE_STORE_REPO_PATH = (
    "hope_training/whole_body_tracking/artifacts/"
    "formal_fitted_ball_capsules_v1"
)
CAPSULE_LAYOUT = "formal_fitted_ball_retained_capsule_v1"
CAPSULE_CHECKOUT_DIRNAME = "checkout"
CAPSULE_ARTIFACTS_DIRNAME = "artifacts"
CAPSULE_FORMAL_RECEIPT_RELPATH = (
    "artifacts/fitted_ball_receipt.json"
)
CAPSULE_RETAINED_RECEIPT_BASENAME = "retained_capsule_receipt.json"
CAPSULE_VIDEO_RELPATH = "artifacts/videos"
_STAGING_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
EXTERNAL_CAPSULE_ENV = "HOPE_FITTED_GATE_EXTERNAL_PREEXEC_CAPSULE_V1"
REQUIRED_EXTERNAL_DISTRIBUTIONS = ("mujoco", "numpy")
MODULE_BINDINGS = {
    CORE_MODULE_NAME: CORE_REPO_PATH,
    "contact_model": "hope_training/ball_physics_fit/contact_model.py",
    "audit_motion_npz": (
        "hope_training/whole_body_tracking/scripts/audit_motion_npz.py"
    ),
    "motion_kinematics_contract": (
        "hope_training/whole_body_tracking/scripts/"
        "motion_kinematics_contract.py"
    ),
    "_hope_racket_contact_geometry": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/"
        "racket_contact_geometry.py"
    ),
    "racket_geometry_contract": (
        "hope_training/whole_body_tracking/scripts/"
        "racket_geometry_contract.py"
    ),
    "mujoco_table_scene": "scripts/mujoco_table_scene.py",
    "mujoco_motion_player": (
        "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py"
    ),
    "mujoco_teacher_motion_native_ball_diagnostic": (
        "hope_training/whole_body_tracking/scripts/"
        "mujoco_teacher_motion_native_ball_diagnostic.py"
    ),
    "canonical_mujoco_identity": (
        "hope_training/whole_body_tracking/scripts/"
        "canonical_mujoco_identity.py"
    ),
    "whole_body_tracking.tasks.table_tennis.geometry": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/geometry.py"
    ),
    "whole_body_tracking.tasks.table_tennis.table_frame": (
        "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/table_frame.py"
    ),
    "_mjcf_table_augmenter": (
        "scripts/audit_motion_schema2_table_net_clearance.py"
    ),
}
SECURITY_ENV_KEYS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONPYCACHEPREFIX",
    "LD_PRELOAD",
    "DYLD_INSERT_LIBRARIES",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
)


class BootstrapError(RuntimeError):
    """The exact execution capsule could not be established."""


@dataclass(frozen=True)
class PinnedBytes:
    repo_path: str
    path: Path
    raw: bytes
    expected_sha256: str
    stat_device: int
    stat_inode: int
    stat_size: int
    stat_mtime_ns: int


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise BootstrapError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> Any:
    raise BootstrapError(f"nonfinite JSON constant {value!r}")


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BootstrapError(f"{label} must be lowercase SHA-256")
    return value


def _require_commit(value: str) -> str:
    if (
        len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BootstrapError(
            "--code-commit must be an exact lowercase 40-digit Git SHA"
        )
    return value


def _cli_value(argv: Sequence[str], flag: str) -> str:
    indices = [index for index, value in enumerate(argv) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise BootstrapError(f"formal bootstrap requires exactly one {flag}")
    value = argv[indices[0] + 1]
    if value.startswith("--"):
        raise BootstrapError(f"{flag} has no value")
    return value


def _repo_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BootstrapError(f"{label} must be a non-empty repository path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise BootstrapError(f"{label} must be a canonical repository-relative path")
    return value


def _parse_trust_spec(raw: bytes) -> dict[str, Any]:
    """Parse the fixed, committed authority consumed by the external launcher."""

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("committed launch trust spec must be exact UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapError("committed launch trust spec must be an object")
    exact_keys = {
        "schema_version",
        "artifact_type",
        "authorization",
        "bootstrap",
        "training_manifest",
        "physical_gate_manifest",
        "physical_gate_materialization_receipt",
        "profile_pins",
        "launch_evidence_trust_root",
        "runtime_environment",
        "receipt_payload_sha256",
    }
    sealed = dict(value)
    observed_seal = sealed.pop("receipt_payload_sha256", None)
    if (
        set(value) != exact_keys
        or value.get("schema_version") != TRUST_SPEC_SCHEMA_VERSION
        or value.get("artifact_type")
        != TRUST_SPEC_ARTIFACT_TYPE
        or observed_seal != _sha256(_canonical_json_bytes(sealed))
    ):
        raise BootstrapError("committed launch trust spec schema/seal mismatch")
    authorization = value.get("authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "formal_simulation_authorized",
        "hardware_authorized",
        "registered_before_gate_run",
        "decision_id",
        "human_dri",
    }:
        raise BootstrapError("committed launch authorization key set is not exact")
    if (
        authorization["formal_simulation_authorized"] is not True
        or authorization["hardware_authorized"] is not False
        or authorization["registered_before_gate_run"] is not True
        or not isinstance(authorization["decision_id"], str)
        or not authorization["decision_id"]
        or not isinstance(authorization["human_dri"], str)
        or not authorization["human_dri"]
    ):
        raise BootstrapError("committed launch authorization is not simulation-only preregistration")

    for key in (
        "bootstrap",
        "training_manifest",
        "physical_gate_manifest",
        "physical_gate_materialization_receipt",
        "profile_pins",
        "launch_evidence_trust_root",
    ):
        binding = value.get(key)
        if not isinstance(binding, dict) or set(binding) != {
            "repo_path",
            "sha256",
        }:
            raise BootstrapError(f"trust spec {key} binding key set is not exact")
        binding["repo_path"] = _repo_relative_path(
            binding["repo_path"], f"trust spec {key}.repo_path"
        )
        binding["sha256"] = _require_sha(
            binding["sha256"], f"trust spec {key}.sha256"
        )
    if value["bootstrap"]["repo_path"] != BOOTSTRAP_REPO_PATH:
        raise BootstrapError(
            "trust spec bootstrap binding must name the fixed launcher path"
        )

    environment = value.get("runtime_environment")
    if not isinstance(environment, dict) or set(environment) != {
        "python_executable_sha256",
        "git_executable_sha256",
        "python_version",
        "python_cache_tag",
        "python_import_roots",
        "required_distributions",
    }:
        raise BootstrapError("trust spec runtime_environment key set is not exact")
    environment["python_executable_sha256"] = _require_sha(
        environment["python_executable_sha256"],
        "trust spec Python executable SHA",
    )
    environment["git_executable_sha256"] = _require_sha(
        environment["git_executable_sha256"],
        "trust spec Git executable SHA",
    )
    if (
        not isinstance(environment["python_version"], str)
        or not environment["python_version"]
        or not isinstance(environment["python_cache_tag"], str)
        or not environment["python_cache_tag"]
    ):
        raise BootstrapError("trust spec Python version/cache tag is invalid")
    import_roots = environment.get("python_import_roots")
    if not isinstance(import_roots, list) or not import_roots:
        raise BootstrapError(
            "trust spec must preregister at least one Python import root"
        )
    normalized_roots: list[dict[str, str]] = []
    seen_roots: set[str] = set()
    for index, row in enumerate(import_roots):
        if not isinstance(row, dict) or set(row) != {
            "path",
            "tree_sha256",
        }:
            raise BootstrapError(
                f"trust spec Python import root {index} key set is not exact"
            )
        path_value = row.get("path")
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or Path(path_value).as_posix() != path_value
            or ".." in Path(path_value).parts
            or path_value in seen_roots
        ):
            raise BootstrapError(
                f"trust spec Python import root {index} is not a unique "
                "canonical absolute path"
            )
        seen_roots.add(path_value)
        normalized_roots.append(
            {
                "path": path_value,
                "tree_sha256": _require_sha(
                    row.get("tree_sha256"),
                    f"trust spec Python import root {index} tree SHA",
                ),
            }
        )
    environment["python_import_roots"] = normalized_roots
    distributions = environment.get("required_distributions")
    if (
        not isinstance(distributions, dict)
        or set(distributions) != set(REQUIRED_EXTERNAL_DISTRIBUTIONS)
    ):
        raise BootstrapError(
            "trust spec required distribution set must be exact "
            f"{list(REQUIRED_EXTERNAL_DISTRIBUTIONS)}"
        )
    for name in REQUIRED_EXTERNAL_DISTRIBUTIONS:
        row = distributions.get(name)
        if not isinstance(row, dict) or set(row) != {
            "import_name",
            "version",
            "import_root",
            "package_subpath",
            "tree_sha256",
        }:
            raise BootstrapError(
                f"trust spec distribution {name} key set is not exact"
            )
        import_root = row.get("import_root")
        package_subpath = row.get("package_subpath")
        if (
            row.get("import_name") != name
            or not isinstance(row.get("version"), str)
            or not row["version"]
            or import_root not in seen_roots
            or not isinstance(package_subpath, str)
            or not package_subpath
            or Path(package_subpath).is_absolute()
            or Path(package_subpath).as_posix() != package_subpath
            or ".." in Path(package_subpath).parts
        ):
            raise BootstrapError(
                f"trust spec distribution {name} identity/path is invalid"
            )
        row["tree_sha256"] = _require_sha(
            row.get("tree_sha256"),
            f"trust spec distribution {name} tree SHA",
        )
    return value


def _git_output_from(
    git: Path, repo_root: Path, arguments: Sequence[str]
) -> bytes:
    try:
        return subprocess.run(
            [str(git), "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(
            f"git command failed: {arguments}: "
            f"{exc.stderr.decode('utf-8', errors='replace')}"
        ) from exc


def _validate_checkout_at(
    git: Path, repo_root: Path, expected_commit: str
) -> None:
    head = _git_output_from(git, repo_root, ("rev-parse", "HEAD")).decode().strip()
    if head != expected_commit:
        raise BootstrapError(
            f"checkout commit mismatch: expected {expected_commit}, got {head}"
        )
    status = _git_output_from(
        git, repo_root, ("status", "--porcelain", "--untracked-files=all")
    )
    if status:
        raise BootstrapError("formal bootstrap requires an exact clean checkout")


def _validate_detached_checkout_at(
    git: Path, repo_root: Path, expected_commit: str
) -> None:
    _validate_checkout_at(git, repo_root, expected_commit)
    completed = subprocess.run(
        [
            str(git),
            "-C",
            str(repo_root),
            "symbolic-ref",
            "--quiet",
            "HEAD",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 1 or completed.stdout or completed.stderr:
        raise BootstrapError(
            "formal capsule checkout must be an exact detached HEAD"
        )


def _validate_pinned_tools(
    *,
    trust_spec: Mapping[str, Any],
    python_identity: Mapping[str, Any],
    git: Path,
) -> None:
    expected = trust_spec["runtime_environment"]
    git_raw = _read_unpinned_regular_nofollow(git, "git executable")[1]
    if (
        python_identity["executable_sha256"]
        != expected["python_executable_sha256"]
        or python_identity["version"] != expected["python_version"]
        or python_identity["cache_tag"] != expected["python_cache_tag"]
        or _sha256(git_raw) != expected["git_executable_sha256"]
    ):
        raise BootstrapError(
            "Python/Git runtime differs from the preregistered committed trust spec"
        )


def _materializer_forward_args(arguments: Sequence[str]) -> list[str]:
    allowed_value_flags = {
        "--source-repo",
        "--code-commit",
        "--capsule-dir",
        "--out",
        "--render-dir",
        "--render-fps",
    }
    allowed_switches = {"--preflight-only"}
    consumed: set[int] = set()
    for index, value in enumerate(arguments):
        if index in consumed:
            continue
        if value in allowed_switches:
            if value in arguments[:index]:
                raise BootstrapError(f"duplicate formal launcher switch {value}")
            consumed.add(index)
            continue
        if value not in allowed_value_flags:
            raise BootstrapError(
                f"external launcher rejects unregistered/security-critical argument {value!r}"
            )
        if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
            raise BootstrapError(f"{value} has no value")
        if value in arguments[:index]:
            raise BootstrapError(f"duplicate formal launcher flag {value}")
        consumed.update((index, index + 1))
    for required in (
        "--source-repo",
        "--code-commit",
        "--capsule-dir",
        "--out",
    ):
        _cli_value(arguments, required)
    forwarded = ["--out", _cli_value(arguments, "--out")]
    for optional in ("--render-dir", "--render-fps"):
        if optional in arguments:
            forwarded.extend((optional, _cli_value(arguments, optional)))
    if "--preflight-only" in arguments:
        forwarded.append("--preflight-only")
    return forwarded


def _make_tree_read_only(root: Path) -> None:
    for current_root, directory_names, file_names in os.walk(
        root, topdown=False, followlinks=False
    ):
        current = Path(current_root)
        for name in file_names:
            path = current / name
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(os.lstat(path).st_mode)
            os.chmod(path, 0o555 if mode & 0o111 else 0o444)
        for name in directory_names:
            path = current / name
            if not path.is_symlink():
                os.chmod(path, 0o555)
    os.chmod(root, 0o555)


def _ensure_plain_directory(path: Path, *, mode: int = 0o755) -> Path:
    """Create one directory or re-open an existing non-symlink directory."""

    try:
        os.mkdir(path, mode)
    except FileExistsError:
        pass
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise BootstrapError(f"cannot lstat directory {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BootstrapError(
            f"capsule layout component must be a plain directory: {path}"
        )
    return path.resolve()


def _prepare_capsule_staging_layout(
    *,
    source_repo: Path,
    requested_capsule_dir: str,
    requested_out: str,
    requested_render_dir: Optional[str],
) -> tuple[Path, Path, Path]:
    """Reserve the only legal pre-publication capsule layout.

    The operator chooses only a short staging name.  The final retained
    capsule name is content-addressed after the formal output exists.
    """

    lexical = Path(requested_capsule_dir).expanduser()
    if not lexical.is_absolute():
        raise BootstrapError("--capsule-dir must be an absolute staging path")
    staging_name = lexical.name
    if (
        _STAGING_ID.fullmatch(staging_name) is None
        or (
            len(staging_name) == 64
            and all(character in "0123456789abcdef" for character in staging_name)
        )
    ):
        raise BootstrapError(
            "--capsule-dir basename must be a short non-digest staging id"
        )
    store_parent = source_repo / "hope_training/whole_body_tracking"
    _assert_no_symlink_component(store_parent)
    artifacts_parent = _ensure_plain_directory(
        store_parent / "artifacts"
    )
    store_root = _ensure_plain_directory(
        artifacts_parent / "formal_fitted_ball_capsules_v1"
    )
    expected_staging = store_root / staging_name
    if lexical != expected_staging:
        raise BootstrapError(
            "--capsule-dir must be one direct child of the fixed repository "
            f"capsule store {store_root}"
        )
    if os.path.lexists(expected_staging):
        raise BootstrapError(
            f"external capsule staging directory must be new: {expected_staging}"
        )
    try:
        os.mkdir(expected_staging, 0o700)
    except OSError as exc:
        raise BootstrapError(
            f"cannot reserve fresh capsule staging directory "
            f"{expected_staging}: {exc}"
        ) from exc
    staging_stat = os.lstat(expected_staging)
    if not stat.S_ISDIR(staging_stat.st_mode):
        raise BootstrapError("reserved capsule staging path is not a directory")
    artifacts_root = expected_staging / CAPSULE_ARTIFACTS_DIRNAME
    try:
        os.mkdir(artifacts_root, 0o700)
    except OSError as exc:
        raise BootstrapError(
            f"cannot reserve capsule artifact directory {artifacts_root}: {exc}"
        ) from exc
    checkout_root = expected_staging / CAPSULE_CHECKOUT_DIRNAME
    expected_out = expected_staging / CAPSULE_FORMAL_RECEIPT_RELPATH
    if Path(requested_out).expanduser() != expected_out:
        raise BootstrapError(
            "--out must be exactly the capsule artifact "
            f"{CAPSULE_FORMAL_RECEIPT_RELPATH}: {expected_out}"
        )
    if requested_render_dir is not None:
        expected_render = expected_staging / CAPSULE_VIDEO_RELPATH
        if Path(requested_render_dir).expanduser() != expected_render:
            raise BootstrapError(
                "--render-dir must be exactly the capsule artifact video "
                f"directory: {expected_render}"
            )
    return expected_staging, checkout_root, artifacts_root


def _materialize_external_capsule(arguments: Sequence[str]) -> int:
    """Create a fresh detached exact-commit worktree, then exec its bootstrap.

    This stage is authorized only when its bytes were supplied on standard input
    by an external ``git show <commit>:<bootstrap>`` command.  It never claims
    to attest the bytes that are already executing.
    """

    if sys.argv[0] != "-" or globals().get("__file__") != "<stdin>":
        raise BootstrapError(
            "formal materialization must execute bootstrap bytes from external "
            "`git show <commit>:<bootstrap> | python -I -S -B - ...`"
        )
    forwarded = _materializer_forward_args(arguments)
    source_repo = _assert_no_symlink_component(
        Path(_cli_value(arguments, "--source-repo"))
    )
    code_commit = _require_commit(_cli_value(arguments, "--code-commit"))
    git = _git_binary()
    _validate_checkout_at(git, source_repo, code_commit)
    (
        capsule_staging_root,
        capsule_checkout_root,
        capsule_artifacts_root,
    ) = _prepare_capsule_staging_layout(
        source_repo=source_repo,
        requested_capsule_dir=_cli_value(arguments, "--capsule-dir"),
        requested_out=_cli_value(arguments, "--out"),
        requested_render_dir=(
            _cli_value(arguments, "--render-dir")
            if "--render-dir" in arguments
            else None
        ),
    )
    _validate_checkout_at(git, source_repo, code_commit)
    trust_raw = _git_output_from(
        git, source_repo, ("show", f"{code_commit}:{TRUST_SPEC_REPO_PATH}")
    )
    trust_spec = _parse_trust_spec(trust_raw)
    materialization_raw = _git_output_from(
        git,
        source_repo,
        (
            "show",
            f"{code_commit}:"
            f"{trust_spec['physical_gate_materialization_receipt']['repo_path']}",
        ),
    )
    if (
        _sha256(materialization_raw)
        != trust_spec["physical_gate_materialization_receipt"]["sha256"]
    ):
        raise BootstrapError(
            "external materialization receipt differs from the committed "
            "trust binding"
        )
    action_set_profile = _materialization_profile_from_raw(
        materialization_raw
    )
    expected_bootstrap = _git_output_from(
        git, source_repo, ("show", f"{code_commit}:{BOOTSTRAP_REPO_PATH}")
    )
    if _sha256(expected_bootstrap) != trust_spec["bootstrap"]["sha256"]:
        raise BootstrapError(
            "external git-show bootstrap bytes differ from the committed "
            "bootstrap authority"
        )
    python_identity = _executable_identity()
    _validate_isolated_python(python_identity)
    _validate_pinned_tools(
        trust_spec=trust_spec,
        python_identity=python_identity,
        git=git,
    )
    _validate_external_dependency_roots(trust_spec, install=False)
    try:
        subprocess.run(
            [
                str(git),
                "-C",
                str(source_repo),
                "worktree",
                "add",
                "--detach",
                str(capsule_checkout_root),
                code_commit,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(
            "cannot create fresh detached launch capsule: "
            f"{exc.stderr.decode('utf-8', errors='replace')}"
        ) from exc
    capsule_checkout_root = capsule_checkout_root.resolve()
    _validate_detached_checkout_at(git, capsule_checkout_root, code_commit)
    bootstrap_path = capsule_checkout_root / BOOTSTRAP_REPO_PATH
    bootstrap_raw = _read_unpinned_regular_nofollow(
        bootstrap_path, "materialized bootstrap"
    )[1]
    if bootstrap_raw != expected_bootstrap:
        raise BootstrapError("materialized bootstrap differs from exact git-show bytes")
    _make_tree_read_only(capsule_checkout_root)
    marker = {
        "schema_version": 1,
        "artifact_type": "external_preexec_immutable_launch_capsule_v1",
        "capsule_layout": CAPSULE_LAYOUT,
        "source_repo": str(source_repo),
        "capsule_staging_root": str(capsule_staging_root),
        "checkout_root": str(capsule_checkout_root),
        "artifacts_root": str(capsule_artifacts_root),
        "code_commit": code_commit,
        "trust_spec_repo_path": TRUST_SPEC_REPO_PATH,
        "trust_spec_sha256": _sha256(trust_raw),
        "bootstrap_sha256": _sha256(expected_bootstrap),
        "action_set_profile": action_set_profile,
        "materializer_source": "external_git_show_stdin",
        "fresh_detached_worktree": True,
        "checkout_read_only_before_exec": True,
    }
    environment = os.environ.copy()
    environment[EXTERNAL_CAPSULE_ENV] = _canonical_json_bytes(marker).decode(
        "utf-8"
    )
    runtime_arguments = [
        "--code-commit",
        code_commit,
        "--action-set-profile",
        action_set_profile,
        "--training-manifest",
        str(
            capsule_checkout_root
            / trust_spec["training_manifest"]["repo_path"]
        ),
        "--training-manifest-sha256",
        trust_spec["training_manifest"]["sha256"],
        "--physical-gate-manifest",
        str(
            capsule_checkout_root
            / trust_spec["physical_gate_manifest"]["repo_path"]
        ),
        "--physical-gate-manifest-sha256",
        trust_spec["physical_gate_manifest"]["sha256"],
        "--physical-gate-materialization-receipt",
        str(
            capsule_checkout_root
            / trust_spec["physical_gate_materialization_receipt"][
                "repo_path"
            ]
        ),
        "--physical-gate-materialization-receipt-sha256",
        trust_spec["physical_gate_materialization_receipt"]["sha256"],
        "--profile-pins",
        str(capsule_checkout_root / trust_spec["profile_pins"]["repo_path"]),
        "--profile-pins-sha256",
        trust_spec["profile_pins"]["sha256"],
        "--launch-trust-root",
        str(
            capsule_checkout_root
            / trust_spec["launch_evidence_trust_root"]["repo_path"]
        ),
        "--launch-trust-root-sha256",
        trust_spec["launch_evidence_trust_root"]["sha256"],
        *forwarded,
    ]
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(bootstrap_path),
            *runtime_arguments,
        ],
        environment,
    )
    raise AssertionError("os.execve unexpectedly returned")


def _assert_no_symlink_component(path: Path) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    parts = lexical.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise BootstrapError(f"cannot lstat {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise BootstrapError(f"symlink path component is forbidden: {current}")
    return lexical.resolve()


def _read_unpinned_regular_nofollow(
    path: Path, label: str
) -> tuple[Path, bytes, os.stat_result]:
    resolved = _assert_no_symlink_component(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(resolved), flags)
    except OSError as exc:
        raise BootstrapError(f"cannot open {label}: {resolved}: {exc}") from exc
    chunks: list[bytes] = []
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BootstrapError(f"{label} is not a regular file: {resolved}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    return resolved, raw, metadata


def _read_regular_nofollow(
    path: Path, expected_sha256: str, label: str
) -> tuple[Path, bytes, os.stat_result]:
    expected = _require_sha(expected_sha256, f"{label} expected SHA")
    resolved, raw, metadata = _read_unpinned_regular_nofollow(
        path, label
    )
    actual = _sha256(raw)
    if actual != expected:
        raise BootstrapError(
            f"{label} SHA mismatch: expected {expected}, got {actual}"
        )
    return resolved, raw, metadata


def _hash_regular_tree(path: Path, label: str) -> dict[str, Any]:
    """Hash every regular file under one symlink-free import tree.

    The manifest is intentionally derived only from relative paths, byte
    lengths, and SHA-256 digests.  It is therefore suitable for preregistration
    in the committed trust spec and cannot be satisfied by a post-hoc package
    version string.
    """

    root = _assert_no_symlink_component(path)
    initial_root_stat = os.lstat(root)
    if not stat.S_ISDIR(initial_root_stat.st_mode):
        raise BootstrapError(f"{label} is not a directory: {root}")
    pending = [root]
    rows: list[dict[str, Any]] = []
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda row: row.name)
        except OSError as exc:
            raise BootstrapError(f"cannot scan {label}: {current}: {exc}") from exc
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise BootstrapError(
                    f"cannot stat {label} member {entry_path}: {exc}"
                ) from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise BootstrapError(
                    f"symlink is forbidden in preregistered import tree: "
                    f"{entry_path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                pending.append(entry_path)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise BootstrapError(
                    f"special file is forbidden in preregistered import tree: "
                    f"{entry_path}"
                )
            resolved, raw, descriptor_stat = _read_unpinned_regular_nofollow(
                entry_path, f"{label} member"
            )
            path_stat = os.stat(entry_path, follow_symlinks=False)
            if (
                resolved != entry_path.resolve()
                or descriptor_stat.st_dev != path_stat.st_dev
                or descriptor_stat.st_ino != path_stat.st_ino
                or descriptor_stat.st_size != path_stat.st_size
                or len(raw) != descriptor_stat.st_size
            ):
                raise BootstrapError(
                    f"{label} member identity changed while hashing: "
                    f"{entry_path}"
                )
            rows.append(
                {
                    "path": entry_path.relative_to(root).as_posix(),
                    "size_bytes": len(raw),
                    "sha256": _sha256(raw),
                }
            )
    final_root_stat = os.lstat(root)
    if (
        initial_root_stat.st_dev != final_root_stat.st_dev
        or initial_root_stat.st_ino != final_root_stat.st_ino
        or initial_root_stat.st_mtime_ns != final_root_stat.st_mtime_ns
    ):
        raise BootstrapError(f"{label} root changed while hashing")
    if not rows:
        raise BootstrapError(f"{label} contains no regular files")
    rows.sort(key=lambda row: row["path"])
    return {
        "path": str(root),
        "tree_sha256": _sha256(
            _canonical_json_bytes(
                {
                    "schema_version": 1,
                    "manifest_type": "symlink_free_regular_file_tree_v1",
                    "files": rows,
                }
            )
        ),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "symlink_free": True,
        "root_device": int(final_root_stat.st_dev),
        "root_inode": int(final_root_stat.st_ino),
        "root_mtime_ns": int(final_root_stat.st_mtime_ns),
    }


def _validate_external_dependency_roots(
    trust_spec: Mapping[str, Any],
    *,
    install: bool,
) -> dict[str, Any]:
    """Validate preregistered third-party bytes and optionally expose them.

    No ``site`` helper or ``.pth`` file is executed.  Exact roots are appended
    only after their complete regular-file tree and the NumPy/MuJoCo package
    subtrees match hashes committed before the Gate run.
    """

    runtime = trust_spec["runtime_environment"]
    roots: list[dict[str, Any]] = []
    for index, expected in enumerate(runtime["python_import_roots"]):
        receipt = _hash_regular_tree(
            Path(expected["path"]), f"Python import root {index}"
        )
        if receipt["tree_sha256"] != expected["tree_sha256"]:
            raise BootstrapError(
                f"Python import root {index} differs from the committed "
                "dependency tree"
            )
        roots.append(receipt)
    distributions: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_EXTERNAL_DISTRIBUTIONS:
        expected = runtime["required_distributions"][name]
        package_path = (
            Path(expected["import_root"]) / expected["package_subpath"]
        )
        receipt = _hash_regular_tree(
            package_path, f"required distribution {name}"
        )
        if receipt["tree_sha256"] != expected["tree_sha256"]:
            raise BootstrapError(
                f"required distribution {name} differs from the committed "
                "package tree"
            )
        distributions[name] = {
            **receipt,
            "import_name": expected["import_name"],
            "expected_version": expected["version"],
            "import_root": expected["import_root"],
            "package_subpath": expected["package_subpath"],
        }
    root_paths = [row["path"] for row in roots]
    if install:
        collisions = [path for path in root_paths if path in sys.path]
        if collisions:
            raise BootstrapError(
                f"preregistered import roots were already exposed: {collisions}"
            )
        sys.path.extend(root_paths)
    return {
        "authority": "committed_symlink_free_dependency_tree_v1",
        "site_module_executed": False,
        "pth_files_executed": False,
        "import_roots": roots,
        "required_distributions": distributions,
        "installed_directly_on_sys_path": bool(install),
    }


def _assert_external_dependency_roots_stable(
    trust_spec: Mapping[str, Any],
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    current = _validate_external_dependency_roots(
        trust_spec, install=False
    )
    for key in ("import_roots", "required_distributions"):
        if current[key] != initial[key]:
            raise BootstrapError(
                f"external dependency {key} changed during Gate execution"
            )
    return {
        "authority": current["authority"],
        "import_root_count": len(current["import_roots"]),
        "required_distributions": sorted(
            current["required_distributions"]
        ),
        "post_runtime_stable": True,
    }


def _git_binary() -> Path:
    candidate = shutil.which("git")
    if not candidate:
        raise BootstrapError("git executable is unavailable")
    return _assert_no_symlink_component(Path(candidate).resolve())


def _git_output(git: Path, arguments: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            [str(git), "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(
            f"git command failed: {arguments}: "
            f"{exc.stderr.decode('utf-8', errors='replace')}"
        ) from exc


def _validate_checkout(git: Path, expected_commit: str) -> None:
    head = _git_output(git, ("rev-parse", "HEAD")).decode().strip()
    if head != expected_commit:
        raise BootstrapError(
            f"checkout commit mismatch: expected {expected_commit}, got {head}"
        )
    status = _git_output(
        git, ("status", "--porcelain", "--untracked-files=all")
    )
    if status:
        raise BootstrapError(
            "formal bootstrap requires an exact clean checkout"
        )


def _load_pinned_json_object(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    _resolved, raw, _metadata = _read_regular_nofollow(
        path, expected_sha256, label
    )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"{label} must be exact UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} root must be an object")
    return value, raw


def _receipt_pin(
    value: Any,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise BootstrapError(f"{label} must be one exact path/SHA binding")
    return {
        "path": _repo_relative_path(value["path"], f"{label}.path"),
        "sha256": _require_sha(value["sha256"], f"{label}.sha256"),
    }


def _materialization_action_set_contract(
    receipt: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Return the sealed schema-2 action set, or legacy schema-1 ``None``."""

    schema_version = receipt.get("schema_version")
    kind = receipt.get("kind")
    if (
        schema_version == 1
        and kind == PHYSICAL_GATE_MATERIALIZATION_KIND
    ):
        return None
    if not (
        schema_version == 2
        and kind == GENERIC_PHYSICAL_GATE_MATERIALIZATION_KIND
    ):
        raise BootstrapError(
            "unsupported physical-gate materialization receipt schema/kind"
        )
    value = receipt.get("action_set_contract")
    if (
        not isinstance(value, Mapping)
        or set(value) != ACTION_SET_CONTRACT_IDENTITY_KEYS
    ):
        raise BootstrapError(
            "schema-2 materialization action_set_contract keys are not exact"
        )
    contract = dict(value)
    declared_sha = _require_sha(
        contract.pop("contract_sha256", None),
        "materialization action_set_contract.contract_sha256",
    )
    if _sha256(_canonical_json_bytes(contract)) != declared_sha:
        raise BootstrapError(
            "schema-2 materialization action_set_contract seal is false"
        )
    expected_n = value.get("expected_n")
    action_ids = value.get("ordered_action_ids")
    action_uids = value.get("ordered_action_uids")
    if (
        value.get("schema_version") != 1
        or value.get("kind")
        != "whole_body_tracking.action_ball.action_set_contract"
        or not isinstance(value.get("profile_id"), str)
        or not value["profile_id"]
        or type(expected_n) is not int
        or expected_n <= 0
        or not isinstance(action_ids, list)
        or len(action_ids) != expected_n
        or len(set(action_ids)) != expected_n
        or any(not isinstance(item, str) or not item for item in action_ids)
        or not isinstance(action_uids, list)
        or len(action_uids) != expected_n
        or len(set(action_uids)) != expected_n
        or any(type(item) is not int or item <= 0 for item in action_uids)
    ):
        raise BootstrapError(
            "schema-2 materialization action-set identity is invalid"
        )
    return dict(value)


def _materialization_action_set_profile(
    receipt: Mapping[str, Any],
) -> str:
    contract = _materialization_action_set_contract(receipt)
    return (
        ACTION_SET_PROFILE
        if contract is None
        else str(contract["profile_id"])
    )


def _materialization_profile_from_raw(raw: bytes) -> str:
    try:
        receipt = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(
            "materialization receipt must be exact UTF-8 JSON"
        ) from exc
    if not isinstance(receipt, dict):
        raise BootstrapError("materialization receipt root must be an object")
    return _materialization_action_set_profile(receipt)


def _materialization_action_identity_matrix(
    training_manifest: Mapping[str, Any],
    action_set_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    action_order = training_manifest.get("action_order")
    actions = training_manifest.get("actions")
    expected_ids = list(action_set_contract["ordered_action_ids"])
    expected_uids = list(action_set_contract["ordered_action_uids"])
    if (
        not isinstance(action_order, list)
        or action_order != expected_ids
        or not isinstance(actions, list)
        or len(actions) != len(expected_ids)
    ):
        raise BootstrapError(
            "strict manifest action matrix is disconnected from schema-2 "
            "action set"
        )
    scope = action_set_contract.get("scope")
    if not isinstance(scope, str) or not scope:
        raise BootstrapError("schema-2 action-set scope is invalid")
    output: list[dict[str, Any]] = []
    for index, (action_id, action_uid, row) in enumerate(
        zip(expected_ids, expected_uids, actions)
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("action_id") != action_id
            or row.get("action_uid") != action_uid
            or not isinstance(row.get("family"), str)
            or not row["family"]
        ):
            raise BootstrapError(
                f"strict manifest action[{index}] ID/UID/family is invalid"
            )
        motion_path = _repo_relative_path(
            row.get("motion_path"), f"strict action[{index}].motion_path"
        )
        motion_sha = _require_sha(
            row.get("motion_sha256"),
            f"strict action[{index}].motion_sha256",
        )
        profile = row.get("ball_profile")
        if not isinstance(profile, Mapping):
            raise BootstrapError(
                f"strict action[{index}].ball_profile must be an object"
            )
        missing = [
            key
            for key in MATERIALIZATION_PROFILE_CENTER_KEYS
            if key not in profile
        ]
        if missing:
            raise BootstrapError(
                f"strict action[{index}] lacks profile centers {missing}"
            )
        center = {
            key: profile[key]
            for key in MATERIALIZATION_PROFILE_CENTER_KEYS
        }
        output.append(
            {
                "action_id": action_id,
                "action_uid": action_uid,
                "family": row["family"],
                "motion_path": motion_path,
                "motion_sha256": motion_sha,
                "scope": scope,
                "profile_center": center,
                "profile_center_sha256": _sha256(
                    _canonical_json_bytes(center)
                ),
            }
        )
    return output


def _validate_dual_manifest_closure(
    *,
    training_manifest_path: Path,
    training_manifest_sha256: str,
    training_manifest_repo_path: str,
    physical_gate_manifest_path: Path,
    physical_gate_manifest_sha256: str,
    physical_gate_manifest_repo_path: str,
    materialization_receipt_path: Path,
    materialization_receipt_sha256: str,
    expected_action_set_profile: str,
) -> dict[str, Any]:
    """Prove the disposable manifest is only the receipt-bound Gate overlay."""

    training, _training_raw = _load_pinned_json_object(
        training_manifest_path,
        training_manifest_sha256,
        "strict training manifest",
    )
    physical, _physical_raw = _load_pinned_json_object(
        physical_gate_manifest_path,
        physical_gate_manifest_sha256,
        "disposable physical-gate manifest",
    )
    receipt, _receipt_raw = _load_pinned_json_object(
        materialization_receipt_path,
        materialization_receipt_sha256,
        "physical-gate materialization receipt",
    )
    common_receipt_keys = {
        "schema_version",
        "kind",
        "strict_training_manifest",
        "physical_task_bundle",
        "physical_gate_manifest",
        "candidate_entries",
        "compiler_manifests",
        "bank_gate_reports",
        "action_order",
        "strict_training_manifest_preserved",
        "inline_manifest_gate_only",
        "selector_executed",
        "authorization_granted",
    }
    action_set_contract = _materialization_action_set_contract(receipt)
    schema2 = action_set_contract is not None
    expected_receipt_keys = (
        common_receipt_keys
        | {"action_set_contract", "action_identity_matrix"}
        if schema2
        else common_receipt_keys
    )
    training_pin = _receipt_pin(
        receipt.get("strict_training_manifest"),
        "receipt.strict_training_manifest",
    )
    physical_pin = _receipt_pin(
        receipt.get("physical_gate_manifest"),
        "receipt.physical_gate_manifest",
    )
    _receipt_pin(
        receipt.get("physical_task_bundle"),
        "receipt.physical_task_bundle",
    )
    if (
        set(receipt) != expected_receipt_keys
        or _materialization_action_set_profile(receipt)
        != expected_action_set_profile
        or training_pin
        != {
            "path": training_manifest_repo_path,
            "sha256": training_manifest_sha256,
        }
        or physical_pin
        != {
            "path": physical_gate_manifest_repo_path,
            "sha256": physical_gate_manifest_sha256,
        }
        or receipt.get("strict_training_manifest_preserved") is not True
        or receipt.get("inline_manifest_gate_only") is not True
        or receipt.get("selector_executed") is not False
        or receipt.get("authorization_granted") is not False
    ):
        raise BootstrapError(
            "physical-gate materialization receipt identity/crossbinding "
            "is not exact"
        )
    for group_name in ("compiler_manifests", "bank_gate_reports"):
        group = receipt.get(group_name)
        if schema2:
            if not isinstance(group, list):
                raise BootstrapError(
                    f"schema-2 receipt.{group_name} must be a list"
                )
        else:
            if not isinstance(group, dict):
                raise BootstrapError(
                    f"receipt.{group_name} must be an object"
                )
            if set(group) != {"base", "append"}:
                raise BootstrapError(
                    f"receipt.{group_name} must contain exact base/append pins"
                )
            for role in ("base", "append"):
                _receipt_pin(
                    group[role], f"receipt.{group_name}.{role}"
                )
    candidate_entries = receipt.get("candidate_entries")
    if not isinstance(candidate_entries, list):
        raise BootstrapError("receipt.candidate_entries must be a list")
    candidate_order: list[str] = []
    for index, row in enumerate(candidate_entries):
        if not isinstance(row, dict) or set(row) != {
            "action_id",
            "path",
            "sha256",
        }:
            raise BootstrapError(
                f"receipt.candidate_entries[{index}] keys are not exact"
            )
        action_id = row.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise BootstrapError(
                f"receipt.candidate_entries[{index}].action_id is invalid"
            )
        _receipt_pin(
            {"path": row["path"], "sha256": row["sha256"]},
            f"receipt.candidate_entries[{index}]",
        )
        candidate_order.append(action_id)

    training_order = training.get("action_order")
    training_actions = training.get("actions")
    physical_actions = physical.get("actions")
    if (
        not isinstance(training_order, list)
        or not training_order
        or any(not isinstance(value, str) or not value for value in training_order)
        or len(set(training_order)) != len(training_order)
        or receipt.get("action_order") != training_order
        or candidate_order != training_order
        or physical.get("action_order") != training_order
        or not isinstance(training_actions, list)
        or not isinstance(physical_actions, list)
        or len(training_actions) != len(training_order)
        or len(physical_actions) != len(training_order)
    ):
        raise BootstrapError(
            "strict/disposable/receipt action order is not one exact closure"
        )
    if schema2:
        assert action_set_contract is not None
        if (
            action_set_contract.get("manifest_path")
            != training_manifest_repo_path
            or action_set_contract.get("manifest_sha256")
            != training_manifest_sha256
            or action_set_contract.get("expected_n")
            != len(training_order)
            or action_set_contract.get("ordered_action_ids")
            != training_order
            or [
                row.get("action_uid")
                for row in training_actions
                if isinstance(row, Mapping)
            ]
            != action_set_contract.get("ordered_action_uids")
        ):
            raise BootstrapError(
                "schema-2 action set is disconnected from the strict "
                "manifest identity"
            )
        expected_matrix = _materialization_action_identity_matrix(
            training, action_set_contract
        )
        observed_matrix = receipt.get("action_identity_matrix")
        if (
            not isinstance(observed_matrix, list)
            or any(
                not isinstance(row, Mapping)
                or set(row) != MATERIALIZATION_ACTION_IDENTITY_KEYS
                for row in observed_matrix
            )
            or observed_matrix != expected_matrix
        ):
            raise BootstrapError(
                "schema-2 family/motion/profile-center matrix drifted"
            )
    elif training_order != list(LEGACY_FRESH_N5_ORDER):
        raise BootstrapError(
            "schema-1 materialization is restricted to the exact fresh N5 "
            "action order"
        )
    if any(name in training for name in GATE_TOP_LEVEL_FIELDS):
        raise BootstrapError(
            "strict training manifest contains gate-only top-level fields"
        )
    training_top = dict(training)
    del training_top["actions"]
    physical_top = dict(physical)
    del physical_top["actions"]
    for name in GATE_TOP_LEVEL_FIELDS:
        value = physical_top.pop(name, None)
        if not isinstance(value, dict) or not value:
            raise BootstrapError(
                f"disposable physical manifest lacks exact {name}"
            )
    if physical_top != training_top:
        raise BootstrapError(
            "disposable physical manifest changed strict top-level fields"
        )
    for index, (action_id, strict_row, physical_row) in enumerate(
        zip(training_order, training_actions, physical_actions)
    ):
        if (
            not isinstance(strict_row, dict)
            or not isinstance(physical_row, dict)
            or strict_row.get("action_id") != action_id
            or physical_row.get("action_id") != action_id
            or any(name in strict_row for name in GATE_ACTION_FIELDS)
        ):
            raise BootstrapError(
                f"strict/disposable action[{index}] identity is invalid"
            )
        stripped = dict(physical_row)
        for name in GATE_ACTION_FIELDS:
            value = stripped.pop(name, None)
            if not isinstance(value, dict) or not value:
                raise BootstrapError(
                    f"disposable action[{index}] lacks exact gate field {name}"
                )
        if stripped != strict_row:
            raise BootstrapError(
                f"disposable action[{index}] changed strict training fields"
            )
        if schema2:
            admission = physical_row["admission"]
            if not isinstance(admission, Mapping):
                raise BootstrapError(
                    f"schema-2 action[{index}] admission is malformed"
                )
            for group_name, path_key, sha_key in (
                (
                    "compiler_manifests",
                    "compiler_manifest_path",
                    "compiler_manifest_sha256",
                ),
                (
                    "bank_gate_reports",
                    "bank_gate_report_path",
                    "bank_gate_report_sha256",
                ),
            ):
                group = receipt[group_name]
                if len(group) != len(training_order):
                    raise BootstrapError(
                        f"schema-2 receipt.{group_name} is not exact N"
                    )
                evidence = group[index]
                if (
                    not isinstance(evidence, Mapping)
                    or set(evidence) != {"action_id", "path", "sha256"}
                    or evidence.get("action_id") != action_id
                    or admission.get(path_key) != evidence.get("path")
                    or admission.get(sha_key) != evidence.get("sha256")
                ):
                    raise BootstrapError(
                        f"schema-2 action[{index}] {group_name} "
                        "is disconnected"
                    )
                _receipt_pin(
                    {
                        "path": evidence["path"],
                        "sha256": evidence["sha256"],
                    },
                    f"schema-2 receipt.{group_name}[{index}]",
                )
    return receipt


def _parse_manifest(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, _raw = _load_pinned_json_object(
        path, expected_sha256, "physical-contact manifest"
    )
    contract = manifest.get("physical_contact_contract")
    if not isinstance(contract, dict):
        raise BootstrapError("manifest lacks physical_contact_contract")
    sources = contract.get("runtime_execution_source_sha256")
    data = contract.get("runtime_execution_data_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_PATHS):
        raise BootstrapError(
            "runtime execution source pin closure is not exact"
        )
    if not isinstance(data, dict) or set(data) != set(DATA_PATHS):
        raise BootstrapError(
            "runtime execution data pin closure is not exact"
        )
    return manifest, sources, data


def _capture_closure(
    *,
    paths: Mapping[str, Path],
    pins: Mapping[str, Any],
    commit: str,
    git: Path,
) -> dict[str, PinnedBytes]:
    output: dict[str, PinnedBytes] = {}
    for repo_path, path in paths.items():
        expected = _require_sha(pins[repo_path], f"pin {repo_path}")
        resolved, raw, metadata = _read_regular_nofollow(
            path, expected, repo_path
        )
        git_raw = _git_output(git, ("show", f"{commit}:{repo_path}"))
        git_sha = _sha256(git_raw)
        if git_sha != expected or git_raw != raw:
            raise BootstrapError(
                f"working bytes do not equal pinned commit blob: {repo_path}"
            )
        output[repo_path] = PinnedBytes(
            repo_path=repo_path,
            path=resolved,
            raw=raw,
            expected_sha256=expected,
            stat_device=int(metadata.st_dev),
            stat_inode=int(metadata.st_ino),
            stat_size=int(metadata.st_size),
            stat_mtime_ns=int(metadata.st_mtime_ns),
        )
    return output


def _executable_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    _resolved, raw, _metadata = _read_unpinned_regular_nofollow(
        executable, "Python executable"
    )
    startup_modules: list[dict[str, Any]] = []
    for module_name in ("sitecustomize", "usercustomize"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise BootstrapError(
                f"{module_name} is loaded without a bindable source file"
            )
        module_path = Path(module_file).resolve()
        _module_resolved, module_raw, _module_metadata = (
            _read_unpinned_regular_nofollow(
                module_path, f"{module_name} source"
            )
        )
        startup_modules.append(
            {
                "module": module_name,
                "path": str(module_path),
                "sha256": _sha256(module_raw),
                "size_bytes": len(module_raw),
            }
        )
    return {
        "executable": str(executable),
        "executable_sha256": _sha256(raw),
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "cache_tag": sys.implementation.cache_tag,
        "isolated": bool(sys.flags.isolated),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "no_user_site": bool(sys.flags.no_user_site),
        "no_site": bool(sys.flags.no_site),
        # Python <3.11 has no named ``safe_path`` flag.  Its isolated mode
        # nevertheless omits the script/current directory from sys.path and
        # implies -E/-s, which is the effective guarantee required here.
        "safe_path": bool(
            getattr(sys.flags, "safe_path", sys.flags.isolated)
        ),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "optimize": int(sys.flags.optimize),
        "initial_sys_path": list(sys.path),
        "environment": {
            key: os.environ.get(key) for key in SECURITY_ENV_KEYS
        },
        "startup_modules": startup_modules,
    }


def _validate_isolated_python(identity: Mapping[str, Any]) -> None:
    if not (
        identity["isolated"]
        and identity["ignore_environment"]
        and identity["no_user_site"]
        and identity["no_site"]
        and identity["safe_path"]
        and identity["dont_write_bytecode"]
        and identity["optimize"] == 0
    ):
        raise BootstrapError(
            "formal bootstrap requires `python -I -S -B` with optimization disabled"
        )
    if identity["startup_modules"]:
        raise BootstrapError(
            "sitecustomize/usercustomize must not execute in the formal runtime"
        )
    for key in (
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ):
        if identity["environment"].get(key):
            raise BootstrapError(
                f"security-sensitive environment variable is set: {key}"
            )


def _load_external_capsule_authority(
    *,
    arguments: Sequence[str],
    python_identity: Mapping[str, Any],
    git: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    raw_marker = os.environ.get(EXTERNAL_CAPSULE_ENV)
    if not raw_marker:
        raise BootstrapError(
            "formal runtime requires an externally materialized pre-exec capsule"
        )
    try:
        marker = json.loads(
            raw_marker,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise BootstrapError("external capsule marker is not exact JSON") from exc
    exact_marker_keys = {
        "schema_version",
        "artifact_type",
        "capsule_layout",
        "source_repo",
        "capsule_staging_root",
        "checkout_root",
        "artifacts_root",
        "code_commit",
        "trust_spec_repo_path",
        "trust_spec_sha256",
        "bootstrap_sha256",
        "materializer_source",
        "fresh_detached_worktree",
        "checkout_read_only_before_exec",
    }
    if (
        not isinstance(marker, dict)
        or set(marker) != exact_marker_keys
        or marker.get("schema_version") != 1
        or marker.get("artifact_type")
        != "external_preexec_immutable_launch_capsule_v1"
        or marker.get("capsule_layout") != CAPSULE_LAYOUT
        or marker.get("trust_spec_repo_path") != TRUST_SPEC_REPO_PATH
        or marker.get("materializer_source") != "external_git_show_stdin"
        or marker.get("fresh_detached_worktree") is not True
        or marker.get("checkout_read_only_before_exec") is not True
    ):
        raise BootstrapError("external capsule marker schema/authority is invalid")
    code_commit = _require_commit(str(marker.get("code_commit", "")))
    if code_commit != _require_commit(
        _cli_value(arguments, "--code-commit")
    ):
        raise BootstrapError("external capsule commit differs from runtime arguments")
    source_repo = _assert_no_symlink_component(
        Path(str(marker.get("source_repo", "")))
    )
    capsule_staging_root = _assert_no_symlink_component(
        Path(str(marker.get("capsule_staging_root", "")))
    )
    capsule_checkout_root = _assert_no_symlink_component(
        Path(str(marker.get("checkout_root", "")))
    )
    capsule_artifacts_root = _assert_no_symlink_component(
        Path(str(marker.get("artifacts_root", "")))
    )
    if (
        capsule_checkout_root != REPO_ROOT.resolve()
        or capsule_staging_root.parent
        != source_repo / CAPSULE_STORE_REPO_PATH
        or capsule_checkout_root
        != capsule_staging_root / CAPSULE_CHECKOUT_DIRNAME
        or capsule_artifacts_root
        != capsule_staging_root / CAPSULE_ARTIFACTS_DIRNAME
        or tuple(
            capsule_staging_root.parent.parts[
                -len(Path(CAPSULE_STORE_REPO_PATH).parts) :
            ]
        )
        != Path(CAPSULE_STORE_REPO_PATH).parts
    ):
        raise BootstrapError(
            "bootstrap is not executing from the externally materialized capsule"
        )
    expected_out = (
        capsule_staging_root / CAPSULE_FORMAL_RECEIPT_RELPATH
    )
    if Path(_cli_value(arguments, "--out")).expanduser() != expected_out:
        raise BootstrapError(
            "--out escaped the externally materialized artifact root"
        )
    if "--render-dir" in arguments:
        expected_render = capsule_staging_root / CAPSULE_VIDEO_RELPATH
        if (
            Path(_cli_value(arguments, "--render-dir")).expanduser()
            != expected_render
        ):
            raise BootstrapError(
                "--render-dir escaped the externally materialized artifact root"
            )
    expected_bootstrap = capsule_checkout_root / BOOTSTRAP_REPO_PATH
    if Path(sys.argv[0]).resolve() != expected_bootstrap.resolve():
        raise BootstrapError("formal runtime bootstrap path is outside the capsule")
    root_mode = stat.S_IMODE(os.lstat(capsule_checkout_root).st_mode)
    if root_mode & 0o222:
        raise BootstrapError(
            "external capsule checkout was not made read-only before exec"
        )
    artifact_mode = stat.S_IMODE(os.lstat(capsule_artifacts_root).st_mode)
    if artifact_mode & 0o200 == 0:
        raise BootstrapError(
            "external capsule artifact root is not writable for O_EXCL output"
        )
    _validate_detached_checkout_at(git, capsule_checkout_root, code_commit)
    trust_path = capsule_checkout_root / TRUST_SPEC_REPO_PATH
    resolved, trust_raw, _metadata = _read_unpinned_regular_nofollow(
        trust_path, "committed launch trust spec"
    )
    if resolved != trust_path.resolve():
        raise BootstrapError("committed trust spec path changed")
    trust_sha = _require_sha(
        marker.get("trust_spec_sha256"), "external trust spec SHA"
    )
    git_trust_raw = _git_output_from(
        git,
        capsule_checkout_root,
        ("show", f"{code_commit}:{TRUST_SPEC_REPO_PATH}"),
    )
    if (
        _sha256(trust_raw) != trust_sha
        or trust_raw != git_trust_raw
    ):
        raise BootstrapError(
            "committed trust spec differs from external git-show bytes"
        )
    trust_spec = _parse_trust_spec(trust_raw)
    bootstrap_resolved, bootstrap_raw, _bootstrap_metadata = (
        _read_unpinned_regular_nofollow(
            expected_bootstrap, "executing capsule bootstrap"
        )
    )
    git_bootstrap_raw = _git_output_from(
        git,
        capsule_checkout_root,
        ("show", f"{code_commit}:{BOOTSTRAP_REPO_PATH}"),
    )
    bootstrap_sha = _require_sha(
        marker.get("bootstrap_sha256"), "external bootstrap SHA"
    )
    if (
        bootstrap_resolved != expected_bootstrap.resolve()
        or bootstrap_raw != git_bootstrap_raw
        or _sha256(bootstrap_raw) != bootstrap_sha
        or bootstrap_sha != trust_spec["bootstrap"]["sha256"]
    ):
        raise BootstrapError(
            "executing bootstrap differs from external Git bytes or the "
            "committed bootstrap authority"
        )
    _validate_pinned_tools(
        trust_spec=trust_spec,
        python_identity=python_identity,
        git=git,
    )
    for flag, key in (
        ("--training-manifest", "training_manifest"),
        ("--physical-gate-manifest", "physical_gate_manifest"),
        (
            "--physical-gate-materialization-receipt",
            "physical_gate_materialization_receipt",
        ),
        ("--profile-pins", "profile_pins"),
        ("--launch-trust-root", "launch_evidence_trust_root"),
    ):
        expected_path = (
            capsule_checkout_root / trust_spec[key]["repo_path"]
        ).resolve()
        if Path(_cli_value(arguments, flag)).resolve() != expected_path:
            raise BootstrapError(f"{flag} is not bound by the committed trust spec")
    for flag, key in (
        ("--training-manifest-sha256", "training_manifest"),
        (
            "--physical-gate-manifest-sha256",
            "physical_gate_manifest",
        ),
        (
            "--physical-gate-materialization-receipt-sha256",
            "physical_gate_materialization_receipt",
        ),
        ("--profile-pins-sha256", "profile_pins"),
        ("--launch-trust-root-sha256", "launch_evidence_trust_root"),
    ):
        if _cli_value(arguments, flag) != trust_spec[key]["sha256"]:
            raise BootstrapError(f"{flag} is not bound by the committed trust spec")
    materialization_path = Path(
        _cli_value(
            arguments, "--physical-gate-materialization-receipt"
        )
    )
    materialization_sha = _cli_value(
        arguments,
        "--physical-gate-materialization-receipt-sha256",
    )
    materialization_raw = _read_regular_nofollow(
        materialization_path,
        materialization_sha,
        "physical-gate materialization receipt",
    )[1]
    expected_action_set_profile = _materialization_profile_from_raw(
        materialization_raw
    )
    if (
        _cli_value(arguments, "--action-set-profile")
        != expected_action_set_profile
        or marker.get("action_set_profile")
        != expected_action_set_profile
    ):
        raise BootstrapError(
            "--action-set-profile is not bound by the exact materialization "
            "receipt"
        )
    marker["capsule_sha256"] = _sha256(_canonical_json_bytes(marker))
    marker["trust_spec_authorization"] = trust_spec["authorization"]
    return marker, trust_spec, trust_raw


def _attestation_rows(
    closure: Mapping[str, PinnedBytes],
) -> list[dict[str, Any]]:
    return [
        {
            "repo_path": row.repo_path,
            "path": str(row.path),
            "expected_sha256": row.expected_sha256,
            "executed_sha256": _sha256(row.raw),
            "git_blob_sha256": row.expected_sha256,
            "size_bytes": len(row.raw),
            "symlink_free": True,
            "initial_stat": {
                "device": row.stat_device,
                "inode": row.stat_inode,
                "size": row.stat_size,
                "mtime_ns": row.stat_mtime_ns,
            },
        }
        for row in sorted(closure.values(), key=lambda item: item.repo_path)
    ]


class PinnedBytesLoader(importlib.abc.Loader):
    def __init__(
        self, snapshot: PinnedBytes, capsule_sha256: str
    ) -> None:
        self.snapshot = snapshot
        self.capsule_sha256 = capsule_sha256

    def create_module(self, spec: Any) -> Optional[Any]:
        return None

    def exec_module(self, module: Any) -> None:
        module.__file__ = str(self.snapshot.path)
        module.__cached__ = None
        module.__pinned_capsule_id__ = self.capsule_sha256
        module.__pinned_executed_sha256__ = _sha256(self.snapshot.raw)
        code = compile(
            self.snapshot.raw,
            str(self.snapshot.path),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
        exec(code, module.__dict__)


class PinnedBytesFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        sources: Mapping[str, PinnedBytes],
        capsule_sha256: str,
    ) -> None:
        self.sources = sources
        self.capsule_sha256 = capsule_sha256

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[Any] = None,
    ) -> Optional[Any]:
        repo_path = MODULE_BINDINGS.get(fullname)
        if repo_path is None:
            return None
        snapshot = self.sources[repo_path]
        return importlib.util.spec_from_loader(
            fullname,
            PinnedBytesLoader(snapshot, self.capsule_sha256),
            origin=str(snapshot.path),
        )


def _install_pinned_execution(
    *,
    sources: Mapping[str, PinnedBytes],
    data: Mapping[str, PinnedBytes],
    capsule_sha256: str,
    consumed_data: set[str],
) -> None:
    preloaded = sorted(set(MODULE_BINDINGS) & set(sys.modules))
    if preloaded:
        raise BootstrapError(
            f"repository-local modules were preloaded: {preloaded}"
        )
    finder = PinnedBytesFinder(sources, capsule_sha256)
    sys.meta_path.insert(0, finder)

    original_spec_from_file = importlib.util.spec_from_file_location
    source_by_resolved = {
        str(snapshot.path): snapshot for snapshot in sources.values()
    }

    def exact_spec_from_file_location(
        name: str,
        location: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        resolved = str(Path(location).expanduser().resolve())
        snapshot = source_by_resolved.get(resolved)
        if snapshot is None:
            return original_spec_from_file(
                name, location, *args, **kwargs
            )
        if MODULE_BINDINGS.get(name) != snapshot.repo_path:
            raise BootstrapError(
                f"unregistered dynamic module binding {name!r} -> "
                f"{snapshot.repo_path}"
            )
        return importlib.util.spec_from_loader(
            name,
            PinnedBytesLoader(snapshot, capsule_sha256),
            origin=str(snapshot.path),
        )

    importlib.util.spec_from_file_location = exact_spec_from_file_location

    original_read_text = Path.read_text
    data_by_resolved = {
        str(snapshot.path): snapshot for snapshot in data.values()
    }

    def exact_read_text(
        path: Path,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
    ) -> str:
        resolved = str(path.expanduser().resolve())
        snapshot = data_by_resolved.get(resolved)
        if snapshot is None:
            return original_read_text(path, encoding=encoding, errors=errors)
        consumed_data.add(snapshot.repo_path)
        return snapshot.raw.decode(
            encoding or "utf-8", errors or "strict"
        )

    Path.read_text = exact_read_text


def _post_runtime_stability(
    *,
    sources: Mapping[str, PinnedBytes],
    data: Mapping[str, PinnedBytes],
    consumed_data: set[str],
    git: Path,
    commit: str,
) -> dict[str, Any]:
    if consumed_data != set(data):
        raise BootstrapError(
            "runtime did not consume the exact pinned data closure: "
            f"got={sorted(consumed_data)}"
        )
    replaced: list[str] = []
    for snapshot in (*sources.values(), *data.values()):
        resolved, raw, metadata = _read_regular_nofollow(
            snapshot.path,
            snapshot.expected_sha256,
            f"post-runtime {snapshot.repo_path}",
        )
        if (
            resolved != snapshot.path
            or int(metadata.st_dev) != snapshot.stat_device
            or int(metadata.st_ino) != snapshot.stat_inode
            or int(metadata.st_size) != snapshot.stat_size
            or int(metadata.st_mtime_ns) != snapshot.stat_mtime_ns
            or raw != snapshot.raw
        ):
            replaced.append(snapshot.repo_path)
    if replaced:
        raise BootstrapError(
            f"runtime source/data inode or bytes changed: {replaced}"
        )
    _validate_checkout(git, commit)
    return {
        "source_files_stable": len(sources),
        "data_files_stable": len(data),
        "consumed_data": sorted(consumed_data),
        "checkout_commit": commit,
        "checkout_clean": True,
    }


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"{label} must be exact UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} must be one JSON object")
    return value


def _runtime_input_digest_rows(
    formal_receipt: Mapping[str, Any],
    *,
    role_prefixes: Sequence[str],
) -> list[dict[str, str]]:
    snapshot = formal_receipt.get("runtime_input_snapshot")
    files = snapshot.get("files") if isinstance(snapshot, Mapping) else None
    if not isinstance(files, list):
        return []
    output: list[dict[str, str]] = []
    for raw_row in files:
        if not isinstance(raw_row, Mapping):
            continue
        digest = raw_row.get("sha256")
        roles = raw_row.get("roles")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(roles, list)
        ):
            continue
        selected = sorted(
            role
            for role in roles
            if isinstance(role, str)
            and any(role.startswith(prefix) for prefix in role_prefixes)
        )
        for role in selected:
            output.append({"role": role, "sha256": digest})
    output.sort(key=lambda row: (row["role"], row["sha256"]))
    return output


def _retained_action_set_identity(
    *,
    formal_receipt: Mapping[str, Any],
    strict_manifest_repo_path: str,
    strict_manifest_sha256: str,
) -> tuple[str, list[str], list[int], str]:
    value = formal_receipt.get("action_set_contract")
    if not isinstance(value, Mapping) or set(value) != (
        ACTION_SET_CONTRACT_IDENTITY_KEYS
    ):
        raise BootstrapError(
            "formal receipt action_set_contract key set is not exact"
        )
    contract = dict(value)
    declared_sha = _require_sha(
        contract.pop("contract_sha256", None),
        "formal receipt action_set_contract.contract_sha256",
    )
    if _sha256(_canonical_json_bytes(contract)) != declared_sha:
        raise BootstrapError(
            "formal receipt action_set_contract payload seal is false"
        )
    expected_n = contract.get("expected_n")
    action_set_profile = contract.get("profile_id")
    action_ids = contract.get("ordered_action_ids")
    action_uids = contract.get("ordered_action_uids")
    formal_actions = formal_receipt.get("actions")
    formal_action_ids = (
        [row.get("action_id") for row in formal_actions]
        if isinstance(formal_actions, list)
        and all(isinstance(row, Mapping) for row in formal_actions)
        else None
    )
    formal_action_uids = (
        [row.get("action_uid") for row in formal_actions]
        if isinstance(formal_actions, list)
        and all(isinstance(row, Mapping) for row in formal_actions)
        else None
    )
    if (
        contract.get("schema_version") != 1
        or contract.get("kind")
        != "whole_body_tracking.action_ball.action_set_contract"
        or not isinstance(action_set_profile, str)
        or not action_set_profile
        or type(expected_n) is not int
        or expected_n <= 0
        or not isinstance(action_ids, list)
        or len(action_ids) != expected_n
        or any(not isinstance(item, str) or not item for item in action_ids)
        or len(set(action_ids)) != expected_n
        or not isinstance(action_uids, list)
        or len(action_uids) != expected_n
        or any(type(item) is not int or item <= 0 for item in action_uids)
        or len(set(action_uids)) != expected_n
        or contract.get("manifest_path") != strict_manifest_repo_path
        or contract.get("manifest_sha256") != strict_manifest_sha256
        or formal_receipt.get("expected_actions") != expected_n
        or formal_receipt.get("expected_action_order") != action_ids
        or formal_receipt.get("action_order") != action_ids
        or formal_action_ids != action_ids
        or formal_action_uids != action_uids
    ):
        raise BootstrapError(
            "formal receipt action-set/strict-manifest/action identity "
            "does not close"
        )
    return (
        declared_sha,
        list(action_ids),
        list(action_uids),
        action_set_profile,
    )


def _retained_formal_action_identity_matrix(
    *,
    formal_receipt: Mapping[str, Any],
    ordered_action_ids: Sequence[str],
    ordered_action_uids: Sequence[int],
) -> tuple[list[dict[str, Any]], str]:
    """Reopen the schema-2 family/motion/scope/profile-center closure."""

    matrix = formal_receipt.get("action_identity_matrix")
    declared_sha = _require_sha(
        formal_receipt.get("action_identity_matrix_sha256"),
        "formal action identity matrix SHA",
    )
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(ordered_action_ids)
        or any(
            not isinstance(row, Mapping)
            or set(row) != MATERIALIZATION_ACTION_IDENTITY_KEYS
            for row in matrix
        )
        or _sha256(_canonical_json_bytes(matrix)) != declared_sha
    ):
        raise BootstrapError(
            "formal schema-2 action identity matrix/key set/seal drifted"
        )
    actions = formal_receipt.get("actions")
    if (
        not isinstance(actions, list)
        or len(actions) != len(ordered_action_ids)
        or any(not isinstance(row, Mapping) for row in actions)
    ):
        raise BootstrapError(
            "formal schema-2 action rows are not exact N"
        )
    normalized: list[dict[str, Any]] = []
    for index, (action_id, action_uid, raw_matrix, raw_action) in enumerate(
        zip(ordered_action_ids, ordered_action_uids, matrix, actions)
    ):
        matrix_row = dict(raw_matrix)
        action_row = dict(raw_action)
        center = matrix_row.get("profile_center")
        center_sha = _require_sha(
            matrix_row.get("profile_center_sha256"),
            f"formal action identity matrix[{index}] profile center SHA",
        )
        if (
            matrix_row.get("action_id") != action_id
            or matrix_row.get("action_uid") != action_uid
            or not isinstance(matrix_row.get("family"), str)
            or not matrix_row["family"]
            or not isinstance(matrix_row.get("scope"), str)
            or not matrix_row["scope"]
            or not isinstance(center, Mapping)
            or set(center) != set(MATERIALIZATION_PROFILE_CENTER_KEYS)
            or _sha256(_canonical_json_bytes(center)) != center_sha
        ):
            raise BootstrapError(
                f"formal action identity matrix[{index}] is invalid"
            )
        _repo_relative_path(
            matrix_row.get("motion_path"),
            f"formal action identity matrix[{index}].motion_path",
        )
        motion_sha = _require_sha(
            matrix_row.get("motion_sha256"),
            f"formal action identity matrix[{index}].motion_sha256",
        )
        if (
            action_row.get("action_id") != action_id
            or action_row.get("action_uid") != action_uid
            or action_row.get("family") != matrix_row["family"]
            or action_row.get("scope") != matrix_row["scope"]
            or action_row.get("motion_sha256") != motion_sha
            or action_row.get("profile_center") != center
            or action_row.get("profile_center_sha256") != center_sha
        ):
            raise BootstrapError(
                f"formal action[{index}] is disconnected from the "
                "schema-2 identity matrix"
            )
        normalized.append(matrix_row)
    return normalized, declared_sha


def _retained_five_solid_identity(
    formal_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    scene = formal_receipt.get("five_solid_safety_scene")
    if not isinstance(scene, Mapping):
        raise BootstrapError(
            "formal receipt lacks the five-solid safety scene"
        )
    geometry_sha = _require_sha(
        scene.get("five_solid_geometry_sha256"),
        "formal five-solid geometry SHA",
    )
    geometry_payload = scene.get("geometry_payload")
    if (
        not isinstance(geometry_payload, Mapping)
        or _sha256(_canonical_json_bytes(geometry_payload))
        != geometry_sha
        or scene.get("obstacle_order")
        != list(FIVE_SOLID_OBSTACLE_ORDER)
        or scene.get("under_table_keepout_role") != "robot_only"
        or scene.get("ball_keepout_native_pair_enabled") is not False
        or scene.get("ball_keepout_analytic_surface_enabled") is not False
    ):
        raise BootstrapError(
            "formal five-solid geometry/filter contract is disconnected"
        )
    expected_ground_policy = {
        "floor_geom_name": FLOOR_GEOM_NAME,
        "legal_foot_body_names": list(LEGAL_FOOT_BODY_NAMES),
        "all_collision_enabled_robot_geoms_floor_pair_enabled": True,
        "foot_floor_penetration_tolerance_m": (
            FOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_floor_penetration_tolerance_m": (
            NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
        ),
        "nonfoot_force_threshold_n": GROUND_CONTACT_FORCE_THRESHOLD_N,
        "continuous_nonfoot_clearance_guard_m": (
            NONFOOT_GROUND_CLEARANCE_GUARD_M
        ),
        "continuous_distance_query_cap_m": (
            GROUND_DISTANCE_QUERY_CAP_M
        ),
    }
    if scene.get("ground_contact_policy") != expected_ground_policy:
        raise BootstrapError(
            "formal five-solid ground-contact policy drifted"
        )
    compiled = scene.get("compiled_by_dt")
    if not isinstance(compiled, Mapping) or set(compiled) != {
        "0.0010",
        "0.0005",
    }:
        raise BootstrapError(
            "formal five-solid compiled timestep set is not exact"
        )
    assembled: list[dict[str, str]] = []
    expected_ground_contract_keys = {
        "floor_geom_name",
        "floor_geom_id",
        "floor_geom_type",
        "legal_foot_body_names",
        "legal_foot_body_ids",
        "legal_foot_geom_names",
        "legal_foot_geom_ids",
        "nonfoot_floor_pair_enabled_robot_geom_count",
        "all_collision_enabled_robot_geoms_floor_pair_enabled",
        "foot_floor_penetration_tolerance_m",
        "nonfoot_floor_penetration_tolerance_m",
        "nonfoot_force_threshold_n",
        "continuous_nonfoot_clearance_guard_m",
        "continuous_distance_query_cap_m",
        "policy",
    }
    for timestep in ("0.0010", "0.0005"):
        row = compiled[timestep]
        ground = (
            row.get("ground_contact_safety_contract")
            if isinstance(row, Mapping)
            else None
        )
        if (
            not isinstance(row, Mapping)
            or row.get("five_solid_geometry_sha256") != geometry_sha
            or row.get("ball_keepout_native_pair_enabled") is not False
            or row.get("ball_keepout_analytic_surface_enabled") is not False
            or not isinstance(ground, Mapping)
            or set(ground) != expected_ground_contract_keys
            or ground.get("floor_geom_name") != FLOOR_GEOM_NAME
            or type(ground.get("floor_geom_id")) is not int
            or ground["floor_geom_id"] < 0
            or ground.get("floor_geom_type") != "plane"
            or ground.get("legal_foot_body_names")
            != list(LEGAL_FOOT_BODY_NAMES)
            or not isinstance(ground.get("legal_foot_body_ids"), list)
            or len(ground["legal_foot_body_ids"]) != 2
            or len(set(ground["legal_foot_body_ids"])) != 2
            or any(
                type(value) is not int or value <= 0
                for value in ground["legal_foot_body_ids"]
            )
            or not isinstance(ground.get("legal_foot_geom_names"), list)
            or not ground["legal_foot_geom_names"]
            or any(
                not isinstance(value, str) or not value
                for value in ground["legal_foot_geom_names"]
            )
            or not isinstance(ground.get("legal_foot_geom_ids"), list)
            or len(ground["legal_foot_geom_ids"])
            != len(ground["legal_foot_geom_names"])
            or len(set(ground["legal_foot_geom_ids"]))
            != len(ground["legal_foot_geom_ids"])
            or any(
                type(value) is not int or value < 0
                for value in ground["legal_foot_geom_ids"]
            )
            or type(
                ground.get(
                    "nonfoot_floor_pair_enabled_robot_geom_count"
                )
            )
            is not int
            or ground[
                "nonfoot_floor_pair_enabled_robot_geom_count"
            ]
            < 0
            or ground.get(
                "all_collision_enabled_robot_geoms_floor_pair_enabled"
            )
            is not True
            or ground.get("foot_floor_penetration_tolerance_m")
            != FOOT_FLOOR_PENETRATION_TOLERANCE_M
            or ground.get("nonfoot_floor_penetration_tolerance_m")
            != NONFOOT_FLOOR_PENETRATION_TOLERANCE_M
            or ground.get("nonfoot_force_threshold_n")
            != GROUND_CONTACT_FORCE_THRESHOLD_N
            or ground.get("continuous_nonfoot_clearance_guard_m")
            != NONFOOT_GROUND_CLEARANCE_GUARD_M
            or ground.get("continuous_distance_query_cap_m")
            != GROUND_DISTANCE_QUERY_CAP_M
            or not isinstance(ground.get("policy"), str)
            or not ground["policy"]
        ):
            raise BootstrapError(
                f"formal five-solid/ground compiled row {timestep} drifted"
            )
        assembled.append(
            {
                "timestep_s": timestep,
                "assembled_xml_sha256": _require_sha(
                    row.get("assembled_xml_sha256"),
                    (
                        "formal five-solid assembled XML SHA "
                        f"{timestep}"
                    ),
                ),
                "ground_contact_safety_contract_sha256": _sha256(
                    _canonical_json_bytes(ground)
                ),
            }
        )
    return {
        "geometry_sha256": geometry_sha,
        "obstacle_order": list(FIVE_SOLID_OBSTACLE_ORDER),
        "robot_only_keepout": FIVE_SOLID_KEEPOUT_NAME,
        "ball_keepout_native_pair_enabled": False,
        "ball_keepout_analytic_surface_enabled": False,
        "ground_contact_policy": expected_ground_policy,
        "assembled_xml_by_dt": assembled,
    }


def _retained_capsule_identity(
    *,
    formal_receipt: Mapping[str, Any],
    formal_raw: bytes,
    artifact_tree: Mapping[str, Any],
) -> dict[str, Any]:
    attestation = formal_receipt.get("runtime_code_identity")
    trust = (
        attestation.get("committed_trust_spec")
        if isinstance(attestation, Mapping)
        else None
    )
    bindings = trust.get("bindings") if isinstance(trust, Mapping) else None

    def binding(name: str) -> tuple[str, str]:
        row = bindings.get(name) if isinstance(bindings, Mapping) else None
        if not isinstance(row, Mapping) or set(row) != {
            "repo_path",
            "sha256",
        }:
            raise BootstrapError(
                f"formal receipt trust binding {name} key set is not exact"
            )
        return (
            _repo_relative_path(
                row["repo_path"],
                f"formal receipt trust binding {name}.repo_path",
            ),
            _require_sha(
                row["sha256"],
                f"formal receipt trust binding {name}.sha256",
            ),
        )

    training_path, training_sha = binding("training_manifest")
    physical_path, physical_sha = binding("physical_gate_manifest")
    materialization_path, materialization_sha = binding(
        "physical_gate_materialization_receipt"
    )
    (
        action_set_contract_sha,
        ordered_action_ids,
        ordered_action_uids,
        action_set_profile,
    ) = _retained_action_set_identity(
        formal_receipt=formal_receipt,
        strict_manifest_repo_path=training_path,
        strict_manifest_sha256=training_sha,
    )

    actions = formal_receipt.get("actions")
    motion_rows: list[dict[str, str]] = []
    if isinstance(actions, list):
        for row in actions:
            if not isinstance(row, Mapping):
                continue
            action_id = row.get("action_id")
            digest = row.get("motion_sha256")
            if isinstance(action_id, str) and isinstance(digest, str):
                motion_rows.append(
                    {"action_id": action_id, "sha256": digest}
                )
    geometry_prefixes = (
        "racket_geometry_",
        "scene_source:",
        "selected_face_mesh:",
        "mujoco_identity_manifest",
        "vendor_root_mjcf",
    )
    formal_schema_version = formal_receipt.get("schema_version")
    materialization_schema_version = formal_receipt.get(
        "materialization_receipt_schema_version"
    )
    materialization_kind = formal_receipt.get(
        "materialization_receipt_kind"
    )
    generic_schema2 = bool(
        formal_schema_version == 2
        and materialization_schema_version == 2
        and materialization_kind
        == GENERIC_PHYSICAL_GATE_MATERIALIZATION_KIND
    )
    if formal_schema_version == 2 and not generic_schema2:
        raise BootstrapError(
            "formal schema-2 receipt is not bound to the exact generic "
            "schema-2 materialization kind"
        )
    if formal_schema_version not in (1, 2):
        raise BootstrapError("unsupported formal fitted-ball receipt schema")
    if (
        formal_schema_version == 1
        and (
            materialization_schema_version == 2
            or materialization_kind
            == GENERIC_PHYSICAL_GATE_MATERIALIZATION_KIND
        )
    ):
        raise BootstrapError(
            "formal schema-1 receipt falsely claims schema-2 materialization"
        )
    action_identity_matrix: list[dict[str, Any]] = []
    action_identity_matrix_sha = ""
    if generic_schema2:
        (
            action_identity_matrix,
            action_identity_matrix_sha,
        ) = _retained_formal_action_identity_matrix(
            formal_receipt=formal_receipt,
            ordered_action_ids=ordered_action_ids,
            ordered_action_uids=ordered_action_uids,
        )
    identity = {
        "schema_version": 3 if generic_schema2 else 2,
        "code_commit": (
            attestation.get("code_commit", "")
            if isinstance(attestation, Mapping)
            else ""
        ),
        "strict_training_manifest_repo_path": training_path,
        "strict_training_manifest_sha256": training_sha,
        "physical_gate_manifest_repo_path": physical_path,
        "physical_gate_manifest_sha256": physical_sha,
        "physical_gate_materialization_receipt_repo_path": (
            materialization_path
        ),
        "physical_gate_materialization_receipt_sha256": (
            materialization_sha
        ),
        "action_set_contract_sha256": action_set_contract_sha,
        "ordered_action_ids": ordered_action_ids,
        "ordered_action_uids": ordered_action_uids,
        "profile_pins_sha256": binding("profile_pins")[1],
        "launch_evidence_trust_root_sha256": binding(
            "launch_evidence_trust_root"
        )[1],
        "motion_sha256": motion_rows,
        "solver_source_sha256": _runtime_input_digest_rows(
            formal_receipt, role_prefixes=("solver_source:",)
        ),
        "physics_sha256": _runtime_input_digest_rows(
            formal_receipt,
            role_prefixes=("venue_yaml", "fitted_contact_model"),
        ),
        "geometry_sha256": _runtime_input_digest_rows(
            formal_receipt, role_prefixes=geometry_prefixes
        ),
        "formal_receipt_sha256": _sha256(formal_raw),
        "formal_receipt_payload_sha256": formal_receipt.get(
            "receipt_payload_sha256", ""
        ),
        "artifact_tree_sha256": artifact_tree.get("tree_sha256", ""),
        "artifact_file_count": artifact_tree.get("file_count", 0),
        "artifact_total_size_bytes": artifact_tree.get(
            "total_size_bytes", 0
        ),
    }
    if generic_schema2:
        identity.update(
            {
                "action_set_profile": action_set_profile,
                "action_identity_matrix": action_identity_matrix,
                "action_identity_matrix_sha256": (
                    action_identity_matrix_sha
                ),
                "five_solid_safety": _retained_five_solid_identity(
                    formal_receipt
                ),
            }
        )
    return identity


def _directory_identity(path: Path, label: str) -> dict[str, int]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise BootstrapError(f"cannot lstat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BootstrapError(f"{label} is not a plain directory: {path}")
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(stat.S_IMODE(metadata.st_mode)),
    }


def _reserve_retained_receipt(path: Path) -> int:
    try:
        return os.open(
            str(path),
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise BootstrapError(
            f"cannot reserve retained capsule receipt {path}: {exc}"
        ) from exc


def _write_reserved_retained_receipt(
    path: Path, descriptor: int, payload: Mapping[str, Any]
) -> str:
    data = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor_stat = os.fstat(descriptor)
    path_stat = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
    ):
        raise BootstrapError("retained capsule receipt pathname was replaced")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise BootstrapError("short write to retained capsule receipt")
        offset += written
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    readback = b"".join(chunks)
    final_descriptor_stat = os.fstat(descriptor)
    final_path_stat = os.stat(path, follow_symlinks=False)
    if (
        readback != data
        or final_descriptor_stat.st_dev != final_path_stat.st_dev
        or final_descriptor_stat.st_ino != final_path_stat.st_ino
        or stat.S_IMODE(final_path_stat.st_mode) != 0o444
    ):
        raise BootstrapError(
            "retained capsule receipt durable readback/identity failed"
        )
    os.close(descriptor)
    return _sha256(readback)


def _finalize_retained_capsule(
    *,
    source_repo: Path,
    staging_root: Path,
    checkout_root: Path,
    artifacts_root: Path,
    formal_receipt_path: Path,
    code_commit: str,
    git: Path,
    core_return_code: int,
) -> dict[str, Any]:
    """Publish one completed staging run under its content-addressed id."""

    source_repo = _assert_no_symlink_component(source_repo)
    staging_root = _assert_no_symlink_component(staging_root)
    checkout_root = _assert_no_symlink_component(checkout_root)
    artifacts_root = _assert_no_symlink_component(artifacts_root)
    expected_store = source_repo / CAPSULE_STORE_REPO_PATH
    if (
        staging_root.parent != expected_store
        or checkout_root != staging_root / CAPSULE_CHECKOUT_DIRNAME
        or artifacts_root != staging_root / CAPSULE_ARTIFACTS_DIRNAME
        or formal_receipt_path
        != staging_root / CAPSULE_FORMAL_RECEIPT_RELPATH
    ):
        raise BootstrapError("capsule finalizer received an escaped layout")
    _validate_detached_checkout_at(git, checkout_root, code_commit)
    formal_path, formal_raw, _formal_stat = _read_unpinned_regular_nofollow(
        formal_receipt_path, "formal fitted-ball receipt"
    )
    if formal_path != formal_receipt_path.resolve():
        raise BootstrapError("formal receipt path changed before publication")
    formal_receipt = _strict_json_object(
        formal_raw, "formal fitted-ball receipt"
    )
    artifact_tree = _hash_regular_tree(
        artifacts_root, "formal fitted-ball capsule artifacts"
    )
    identity = _retained_capsule_identity(
        formal_receipt=formal_receipt,
        formal_raw=formal_raw,
        artifact_tree=artifact_tree,
    )
    capsule_id = _sha256(_canonical_json_bytes(identity))
    final_root = expected_store / capsule_id
    try:
        os.mkdir(final_root, 0o700)
    except FileExistsError as exc:
        raise BootstrapError(
            f"content-addressed capsule already exists: {final_root}"
        ) from exc
    except OSError as exc:
        raise BootstrapError(
            f"cannot reserve content-addressed capsule {final_root}: {exc}"
        ) from exc
    try:
        # macOS refuses to rename a directory carrying the provenance xattr
        # while its own write bits are clear.  Runtime is already complete;
        # the full checkout/output closure is revalidated below before the
        # retained receipt is minted and the tree is sealed read-only again.
        os.chmod(staging_root / CAPSULE_CHECKOUT_DIRNAME, 0o755)
        os.rename(
            staging_root / CAPSULE_CHECKOUT_DIRNAME,
            final_root / CAPSULE_CHECKOUT_DIRNAME,
        )
        os.rename(
            staging_root / CAPSULE_ARTIFACTS_DIRNAME,
            final_root / CAPSULE_ARTIFACTS_DIRNAME,
        )
        os.rmdir(staging_root)
    except OSError as exc:
        raise BootstrapError(
            f"cannot publish reserved capsule members {staging_root} -> "
            f"{final_root}: {exc}; reserved final id will not be reused"
        ) from exc
    final_checkout = final_root / CAPSULE_CHECKOUT_DIRNAME
    final_artifacts = final_root / CAPSULE_ARTIFACTS_DIRNAME
    final_formal_receipt = final_root / CAPSULE_FORMAL_RECEIPT_RELPATH
    repair = subprocess.run(
        [str(git), "-C", str(source_repo), "worktree", "repair", str(final_checkout)],
        check=False,
        capture_output=True,
    )
    if repair.returncode != 0:
        raise BootstrapError(
            "cannot repair moved retained worktree metadata: "
            f"{repair.stderr.decode('utf-8', errors='replace')}"
        )
    _validate_detached_checkout_at(git, final_checkout, code_commit)
    repeated_formal = _read_unpinned_regular_nofollow(
        final_formal_receipt, "published formal fitted-ball receipt"
    )[1]
    repeated_tree = _hash_regular_tree(
        final_artifacts, "published fitted-ball capsule artifacts"
    )
    if (
        repeated_formal != formal_raw
        or repeated_tree["tree_sha256"] != artifact_tree["tree_sha256"]
        or repeated_tree["file_count"] != artifact_tree["file_count"]
        or repeated_tree["total_size_bytes"]
        != artifact_tree["total_size_bytes"]
    ):
        raise BootstrapError(
            "formal receipt/artifact bytes changed during capsule publication"
        )
    retained_path = final_root / CAPSULE_RETAINED_RECEIPT_BASENAME
    retained_descriptor = _reserve_retained_receipt(retained_path)
    _make_tree_read_only(final_root)
    filesystem = {
        "capsule_root": _directory_identity(final_root, "capsule root"),
        "checkout": _directory_identity(final_checkout, "capsule checkout"),
        "artifacts": _directory_identity(final_artifacts, "capsule artifacts"),
    }
    if any(row["mode"] & 0o222 for row in filesystem.values()):
        raise BootstrapError("retained capsule directories are still writable")
    retained_receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "retained_formal_fitted_ball_capsule_v1",
        "layout": CAPSULE_LAYOUT,
        "capsule_id": capsule_id,
        "code_commit": code_commit,
        "paths": {
            "checkout": CAPSULE_CHECKOUT_DIRNAME,
            "artifacts": CAPSULE_ARTIFACTS_DIRNAME,
            "formal_receipt": CAPSULE_FORMAL_RECEIPT_RELPATH,
        },
        "identity": identity,
        "artifact_tree": {
            key: artifact_tree[key]
            for key in (
                "tree_sha256",
                "file_count",
                "total_size_bytes",
                "symlink_free",
            )
        },
        "filesystem": filesystem,
        "checkout": {
            "commit": code_commit,
            "clean": True,
            "detached": True,
            "read_only": True,
        },
        "gate_return_code": int(core_return_code),
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
    }
    retained_receipt["receipt_payload_sha256"] = _sha256(
        _canonical_json_bytes(retained_receipt)
    )
    retained_sha = _write_reserved_retained_receipt(
        retained_path, retained_descriptor, retained_receipt
    )
    for directory in (final_root, expected_store):
        descriptor = os.open(
            str(directory),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {
        "capsule_id": capsule_id,
        "capsule_root": str(final_root),
        "formal_receipt": {
            "path": str(final_formal_receipt),
            "sha256": _sha256(formal_raw),
        },
        "retained_capsule_receipt": {
            "path": str(retained_path),
            "sha256": retained_sha,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        python_identity = _executable_identity()
        _validate_isolated_python(python_identity)
        if not os.environ.get(EXTERNAL_CAPSULE_ENV):
            return _materialize_external_capsule(arguments)
        code_commit = _require_commit(
            _cli_value(arguments, "--code-commit")
        )
        training_manifest_path = Path(
            _cli_value(arguments, "--training-manifest")
        )
        training_manifest_sha = _require_sha(
            _cli_value(arguments, "--training-manifest-sha256"),
            "--training-manifest-sha256",
        )
        physical_manifest_path = Path(
            _cli_value(arguments, "--physical-gate-manifest")
        )
        physical_manifest_sha = _require_sha(
            _cli_value(arguments, "--physical-gate-manifest-sha256"),
            "--physical-gate-manifest-sha256",
        )
        materialization_receipt_path = Path(
            _cli_value(
                arguments, "--physical-gate-materialization-receipt"
            )
        )
        materialization_receipt_sha = _require_sha(
            _cli_value(
                arguments,
                "--physical-gate-materialization-receipt-sha256",
            ),
            "--physical-gate-materialization-receipt-sha256",
        )
        git = _git_binary()
        external_capsule, trust_spec, trust_raw = (
            _load_external_capsule_authority(
                arguments=arguments,
                python_identity=python_identity,
                git=git,
            )
        )
        external_dependencies = _validate_external_dependency_roots(
            trust_spec, install=True
        )
        _validate_dual_manifest_closure(
            training_manifest_path=training_manifest_path,
            training_manifest_sha256=training_manifest_sha,
            training_manifest_repo_path=trust_spec["training_manifest"][
                "repo_path"
            ],
            physical_gate_manifest_path=physical_manifest_path,
            physical_gate_manifest_sha256=physical_manifest_sha,
            physical_gate_manifest_repo_path=trust_spec[
                "physical_gate_manifest"
            ]["repo_path"],
            materialization_receipt_path=materialization_receipt_path,
            materialization_receipt_sha256=materialization_receipt_sha,
            expected_action_set_profile=_cli_value(
                arguments, "--action-set-profile"
            ),
        )
        _manifest, source_pins, data_pins = _parse_manifest(
            physical_manifest_path, physical_manifest_sha
        )
        sources = _capture_closure(
            paths=SOURCE_PATHS,
            pins=source_pins,
            commit=code_commit,
            git=git,
        )
        data = _capture_closure(
            paths=DATA_PATHS,
            pins=data_pins,
            commit=code_commit,
            git=git,
        )
        trust_spec_snapshot = _capture_closure(
            paths={
                TRUST_SPEC_REPO_PATH: REPO_ROOT / TRUST_SPEC_REPO_PATH
            },
            pins={TRUST_SPEC_REPO_PATH: _sha256(trust_raw)},
            commit=code_commit,
            git=git,
        )
        git_identity_raw = _read_unpinned_regular_nofollow(
            git, "git executable"
        )[1]
        attestation: dict[str, Any] = {
            "schema_version": 2,
            "loader": "external_preexec_capsule_pinned_bytes_v1",
            "repository_pyc_used": False,
            "code_commit": code_commit,
            "external_preexec": external_capsule,
            "committed_trust_spec": {
                "repo_path": TRUST_SPEC_REPO_PATH,
                "sha256": _sha256(trust_raw),
                "authorization": trust_spec["authorization"],
                "bindings": {
                    key: dict(trust_spec[key])
                    for key in (
                        "bootstrap",
                        "training_manifest",
                        "physical_gate_manifest",
                        "physical_gate_materialization_receipt",
                        "profile_pins",
                        "launch_evidence_trust_root",
                    )
                },
                "runtime_environment": dict(
                    trust_spec["runtime_environment"]
                ),
            },
            "external_python_dependencies": external_dependencies,
            "sources": _attestation_rows(sources),
            "data": _attestation_rows(data),
            "module_bindings": dict(sorted(MODULE_BINDINGS.items())),
            "python": python_identity,
            "git": {
                "executable": str(git),
                "executable_sha256": _sha256(git_identity_raw),
            },
        }
        attestation["capsule_sha256"] = _sha256(
            _canonical_json_bytes(attestation)
        )
        consumed_data: set[str] = set()
        _install_pinned_execution(
            sources=sources,
            data=data,
            capsule_sha256=attestation["capsule_sha256"],
            consumed_data=consumed_data,
        )
        core_spec = importlib.util.spec_from_loader(
            CORE_MODULE_NAME,
            PinnedBytesLoader(
                sources[CORE_REPO_PATH],
                attestation["capsule_sha256"],
            ),
            origin=str(sources[CORE_REPO_PATH].path),
        )
        if core_spec is None or core_spec.loader is None:
            raise BootstrapError("cannot construct pinned core module")
        core = importlib.util.module_from_spec(core_spec)
        sys.modules[CORE_MODULE_NAME] = core
        core_spec.loader.exec_module(core)
        core.RUNTIME_EXECUTION_ATTESTATION = attestation
        core.RUNTIME_EXECUTION_DATA_CONSUMPTION = consumed_data

        def finalize_runtime() -> dict[str, Any]:
            result = _post_runtime_stability(
                sources={**sources, **trust_spec_snapshot},
                data=data,
                consumed_data=consumed_data,
                git=git,
                commit=code_commit,
            )
            result["external_python_dependencies"] = (
                _assert_external_dependency_roots_stable(
                    trust_spec, external_dependencies
                )
            )
            return result

        core.RUNTIME_EXECUTION_FINALIZER = finalize_runtime
        core_return_code = int(core.main(arguments))
        retained = _finalize_retained_capsule(
            source_repo=Path(str(external_capsule["source_repo"])),
            staging_root=Path(
                str(external_capsule["capsule_staging_root"])
            ),
            checkout_root=Path(str(external_capsule["checkout_root"])),
            artifacts_root=Path(str(external_capsule["artifacts_root"])),
            formal_receipt_path=Path(_cli_value(arguments, "--out")),
            code_commit=code_commit,
            git=git,
            core_return_code=core_return_code,
        )
        print(
            "[fitted-ball-bootstrap] retained "
            f"capsule_id={retained['capsule_id']} "
            f"root={retained['capsule_root']} "
            "formal_receipt_sha256="
            f"{retained['formal_receipt']['sha256']} "
            "retained_receipt_sha256="
            f"{retained['retained_capsule_receipt']['sha256']}"
        )
        return core_return_code
    except Exception as exc:
        print(
            f"[fitted-ball-bootstrap][FATAL] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
