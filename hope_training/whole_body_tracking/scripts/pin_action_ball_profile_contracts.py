#!/usr/bin/env python3
"""Host-side re-pin of the action-ball physics/solver profile contract SHAs.

人话:manifest 里的 solver/physics profile SHA 不许手填。这支脚本把 runtime 里
真正执行的两个合同函数(``action_ball_physics_profile_contract`` /
``action_ball_solver_profile_contract``)从 hope_commands.py 的 AST 原样抠出来,
用与发射时相同的 cfg 默认值、场馆物理 YAML 与桌面几何常数在 CPU 上跑一遍,
输出两个 payload 与 SHA-256。build_action_ball_manifest.py 的
``--solver-profile-sha256/--physics-profile-sha256`` 只能喂这里的输出;运行时
boot 会用同一套函数重新计算并拒绝任何不一致。

The cfg values are read from the ``RacketTargetCommandCfg`` dataclass defaults
in the same AST, so a knob change in code changes the pin automatically; a
launch that overrides any of these knobs in YAML must re-run this script with
``--override key=value``.

``--source-rev`` hashes the five solver implementation sources from a git
revision instead of the working tree.  Only that mode emits the formal
external-commit authority required by the launch gates.  Omitting it remains
useful for local diagnostics, but the resulting document carries an explicit
worktree-only authority that every formal consumer must reject.

The commit itself is deliberately not embedded in the JSON.  A profile-pins
file is normally committed after the executable sources it describes, so a
top-level ``source_rev`` would either be stale or create an impossible
self-reference.  Formal launch binds the exact commit externally and reopens
these five blob digests from that commit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import types

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPTS_DIR.parents[2]
MDP_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
GEOMETRY_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/geometry.py"
)
SOLVER_SOURCES = (
    "hope_commands.py",
    "continuous_questions.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
)
PROFILE_PINS_SCHEMA_VERSION = 1
PROFILE_PINS_KIND = "whole_body_tracking.action_ball.profile_pins"
FORMAL_SOURCE_AUTHORITY = "external_exact_commit_subset_blob_map_v1"
FORMAL_COMMIT_BINDING = "external_preexec_immutable_launch_capsule_v1"
WORKTREE_SOURCE_AUTHORITY = "uncommitted_worktree_diagnostic_only_v1"
CFG_FIELDS = (
    "vb_table_surface_z",
    "vb_table_near_x",
    "vb_min_landing_depth",
    "vb_capture_radius",
    "vb_min_approach_speed",
    "vb_rollout_h",
    "vb_rollout_steps",
    "cq_n_iters",
    "cq_tol_m",
    "cq_speed_budget",
    "cq_max_redraw_rounds",
    "cq_overdraw",
)
GEOMETRY_CONSTANTS = (
    "TABLE_LENGTH",
    "TABLE_WIDTH",
    "NET_X",
    "NET_HEIGHT",
    "BALL_RADIUS",
)


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_tree_bytes(raw: bytes, *, filename: str) -> ast.Module:
    return ast.parse(raw.decode("utf-8"), filename=filename)


def _extract_functions(tree: ast.Module, path: Path, names, namespace):
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    found = {node.name for node in nodes}
    if found != wanted:
        raise SystemExit(f"missing contract functions: {sorted(wanted - found)}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _extract_constants(tree: ast.Module, names) -> dict:
    wanted = set(names)
    values: dict = {}
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            ]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            targets = [node.target.id]
            value = node.value
        for name in targets:
            if name in wanted and value is not None:
                values[name] = ast.literal_eval(value)
    missing = wanted - set(values)
    if missing:
        raise SystemExit(f"missing module constants: {sorted(missing)}")
    return values


def _extract_cfg_defaults(tree: ast.Module, class_name: str, fields) -> dict:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict = {}
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target, ast.Name
                ):
                    name = statement.target.id
                    if name in fields and statement.value is not None:
                        values[name] = ast.literal_eval(statement.value)
            missing = set(fields) - set(values)
            if missing:
                raise SystemExit(
                    f"{class_name} lacks literal defaults for: {sorted(missing)}"
                )
            return values
    raise SystemExit(f"class {class_name} not found")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha256(repo_root: Path, rev: str, relative: str) -> str:
    return hashlib.sha256(
        _git_blob_bytes(repo_root, rev, relative)
    ).hexdigest()


def _git_blob_bytes(repo_root: Path, rev: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{rev}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout


def _resolve_git_commit(repo_root: Path, rev: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            f"{rev}^{{commit}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", result) is None:
        raise SystemExit(
            f"--source-rev did not resolve to one full commit OID: {result!r}"
        )
    return result


def _load_contact_geometry_contract(
    repo_root: Path,
    source_rev: str | None,
) -> dict:
    """Execute and verify the stdlib-only exact-face geometry source.

    ``action_ball_solver_profile_contract`` binds both the geometry payload
    SHA and the source bytes.  Loading only the four historical solver files
    made this pinner incapable of reproducing the live v2 contract and, worse,
    allowed a geometry-source edit to leave an apparently valid solver pin.
    """

    relative = f"{MDP_REL}/racket_contact_geometry.py"
    source_path = repo_root / relative
    raw = (
        _git_blob_bytes(repo_root, source_rev, relative)
        if source_rev
        else source_path.read_bytes()
    )
    module_name = "_pin_action_ball_racket_contact_geometry"
    module = types.ModuleType(module_name)
    module.__file__ = (
        f"{source_rev}:{relative}" if source_rev else str(source_path)
    )
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous

    payload = getattr(module, "GEOMETRY_SOURCE_PAYLOAD", None)
    declared_sha256 = getattr(module, "GEOMETRY_SOURCE_SHA256", None)
    if not isinstance(payload, dict):
        raise SystemExit(
            "racket_contact_geometry.GEOMETRY_SOURCE_PAYLOAD must be a dict"
        )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    computed_sha256 = hashlib.sha256(canonical).hexdigest()
    if declared_sha256 != computed_sha256:
        raise SystemExit(
            "racket_contact_geometry.GEOMETRY_SOURCE_SHA256 does not match "
            "its canonical GEOMETRY_SOURCE_PAYLOAD"
        )
    return {
        "payload": payload,
        "sha256": computed_sha256,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_venue_params(path: Path) -> SimpleNamespace:
    """Mirror virtual_ball.load_venue_params without importing torch."""

    with path.open("r") as handle:
        raw = yaml.safe_load(handle)
    pad = raw["contact"]["paddle"]
    return SimpleNamespace(
        k_d=float(raw["flight"]["k_d"]),
        k_m=float(raw["flight"]["k_m"]),
        g=float(raw["flight"]["g"]),
        ball_radius=float(raw["ball"]["radius"]),
        inertia_coeff=float(raw["ball"]["inertia_coeff"]),
        paddle_a_t=float(pad["a_t"]),
        paddle_b_t=float(pad["b_t"]),
        paddle_mu=float(pad["mu_safety"]),
        paddle_e_g1=float(pad["e_exp_g1"]),
        paddle_e_g2=float(pad["e_exp_g2"]),
        source_path=str(path.resolve()),
    )


def _source_authority(source_sha256: dict, *, formal: bool) -> dict:
    source_blob_map_sha256 = hashlib.sha256(
        json.dumps(
            source_sha256,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "authority": (
            FORMAL_SOURCE_AUTHORITY
            if formal
            else WORKTREE_SOURCE_AUTHORITY
        ),
        "commit_binding": FORMAL_COMMIT_BINDING if formal else "none",
        "embedded_commit": False,
        "source_blob_map_sha256": source_blob_map_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument(
        "--venue-yaml",
        default=None,
        help="default: <repo>/configs/ball_physics_venue.yaml",
    )
    parser.add_argument(
        "--source-rev",
        default=None,
        help="hash solver implementation sources from this git revision "
        "instead of the working tree (verification mode)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="cfg knob override matching an explicit launch YAML override",
    )
    parser.add_argument("--out", default=None, help="write the JSON here too")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_rev = (
        _resolve_git_commit(repo_root, args.source_rev)
        if args.source_rev
        else None
    )
    commands_path = repo_root / MDP_REL / "hope_commands.py"
    if source_rev:
        commands_raw = _git_blob_bytes(
            repo_root,
            source_rev,
            f"{MDP_REL}/hope_commands.py",
        )
        commands_label = (
            f"{source_rev}:{MDP_REL}/hope_commands.py"
        )
        tree = _module_tree_bytes(
            commands_raw,
            filename=commands_label,
        )
    else:
        commands_label = str(commands_path)
        tree = _module_tree(commands_path)

    namespace = {
        "hashlib": hashlib,
        "json": json,
        "Path": Path,
    }
    namespace.update(
        _extract_constants(
            tree,
            (
                "_ACTION_BALL_PHYSICS_PROFILE_SCHEMA_VERSION",
                "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION",
            ),
        )
    )
    _extract_functions(
        tree,
        Path(commands_label),
        (
            "_action_ball_canonical_sha256",
            "_action_ball_sha256_file",
            "action_ball_physics_profile_contract",
            "action_ball_solver_profile_contract",
        ),
        namespace,
    )

    cfg_values = _extract_cfg_defaults(
        tree, "RacketTargetCommandCfg", CFG_FIELDS
    )
    for item in args.override:
        key, _, raw_value = item.partition("=")
        if key not in CFG_FIELDS:
            raise SystemExit(f"--override key {key!r} is not a pinned knob")
        cfg_values[key] = ast.literal_eval(raw_value)
    cfg = SimpleNamespace(**cfg_values)

    if source_rev:
        geometry_raw = _git_blob_bytes(
            repo_root,
            source_rev,
            GEOMETRY_REL,
        )
        geometry_tree = _module_tree_bytes(
            geometry_raw,
            filename=f"{source_rev}:{GEOMETRY_REL}",
        )
    else:
        geometry_tree = _module_tree(repo_root / GEOMETRY_REL)
    geometry = _extract_constants(geometry_tree, GEOMETRY_CONSTANTS)

    venue_relative = "configs/ball_physics_venue.yaml"
    temporary_venue = None
    if args.venue_yaml:
        venue_yaml = Path(args.venue_yaml).resolve()
        venue_raw = venue_yaml.read_bytes()
        venue_label = str(venue_yaml)
        physics_repo_root = repo_root
        prm = _load_venue_params(venue_yaml)
        temporary_venue = None
    elif source_rev:
        venue_raw = _git_blob_bytes(
            repo_root,
            source_rev,
            venue_relative,
        )
        temporary_venue = tempfile.TemporaryDirectory(
            prefix="action-ball-historical-venue-"
        )
        physics_repo_root = Path(temporary_venue.name)
        venue_yaml = physics_repo_root / venue_relative
        venue_yaml.parent.mkdir(parents=True)
        venue_yaml.write_bytes(venue_raw)
        venue_label = f"{source_rev}:{venue_relative}"
        prm = _load_venue_params(venue_yaml)
    else:
        venue_yaml = repo_root / venue_relative
        venue_raw = venue_yaml.read_bytes()
        venue_label = str(venue_yaml)
        physics_repo_root = repo_root
        prm = _load_venue_params(venue_yaml)

    # _cq_planes verbatim: solve on the exact surfaces the scorer grades on.
    surface_z = float(cfg.vb_table_surface_z) + float(prm.ball_radius)
    net_x = float(cfg.vb_table_near_x) + float(geometry["NET_X"])
    net_top_z = (
        float(cfg.vb_table_surface_z)
        + float(geometry["NET_HEIGHT"])
        + float(prm.ball_radius)
    )
    opponent_far_x = float(cfg.vb_table_near_x) + float(
        geometry["TABLE_LENGTH"]
    )
    table_half_width = float(geometry["TABLE_WIDTH"]) / 2.0

    physics = namespace["action_ball_physics_profile_contract"](
        cfg,
        prm,
        repo_root=physics_repo_root,
        surface_z=surface_z,
        net_x=net_x,
        net_top_z=net_top_z,
        opponent_near_x=float(cfg.vb_table_near_x),
        opponent_far_x=opponent_far_x,
        table_half_width=table_half_width,
    )
    if temporary_venue is not None:
        temporary_venue.cleanup()

    if source_rev:
        source_sha256 = {
            name: _git_blob_sha256(
                repo_root, source_rev, f"{MDP_REL}/{name}"
            )
            for name in SOLVER_SOURCES
        }
    else:
        source_sha256 = {
            name: _sha256_file(repo_root / MDP_REL / name)
            for name in SOLVER_SOURCES
        }

    contact_geometry = _load_contact_geometry_contract(
        repo_root,
        source_rev,
    )
    if (
        source_sha256["racket_contact_geometry.py"]
        != contact_geometry["source_sha256"]
    ):
        raise SystemExit(
            "racket contact geometry bytes changed while profile was pinned"
        )

    solver = namespace["action_ball_solver_profile_contract"](
        cfg,
        physics_profile_sha256=physics["sha256"],
        source_sha256=source_sha256,
        contact_geometry_contract={
            "payload": contact_geometry["payload"],
            "sha256": contact_geometry["sha256"],
        },
        net_top_z=net_top_z,
    )

    report = {
        "schema_version": PROFILE_PINS_SCHEMA_VERSION,
        "kind": PROFILE_PINS_KIND,
        "source_authority": _source_authority(
            source_sha256,
            formal=source_rev is not None,
        ),
        "cfg": cfg_values,
        "geometry": geometry,
        # Keep the profile portable.  The physics payload already binds the
        # same repository-relative path and exact file bytes.
        "venue_yaml": physics["payload"]["venue_source"]["path"],
        "venue_yaml_sha256": hashlib.sha256(venue_raw).hexdigest(),
        "planes": {
            "surface_z": surface_z,
            "net_x": net_x,
            "net_top_z": net_top_z,
        },
        "solver_implementation_source_sha256": source_sha256,
        "contact_geometry": {
            "payload": contact_geometry["payload"],
            "sha256": contact_geometry["sha256"],
        },
        "physics_profile_sha256": physics["sha256"],
        "solver_profile_sha256": solver["sha256"],
        "physics_payload": physics["payload"],
        "solver_payload": solver["payload"],
    }
    encoded = json.dumps(report, indent=1, sort_keys=True)
    print(encoded)
    if args.out:
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
