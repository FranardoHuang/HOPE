#!/usr/bin/env python3
"""Is the MuJoCo GPU lane asking the same question as the Isaac A211/C211 run?

人话
----
这个文件回答一个问题:**mjlab 这条 GPU 车道,和我们真正在 Isaac 上跑的 A211/C211
训练,问的是不是同一道题?差在哪?每条差异要不要紧?**

它**不是一层假的对齐层**。表里绝大多数行的裁定就是"差着,而且要紧"。这个文件的
价值不是让两边看起来对齐,而是把差异变成**机器会拒绝的东西**:
一条 mjlab 的曲线不能再被当成 Isaac 结论的证据,除非这张表上零条 blocking。

它也**不是一张手写的对照表**。每一行两侧都是**活值**:

* Isaac 侧 —— AST 从源码取值(``hope_env_cfg.py`` / ``tracking_env_cfg.py`` /
  ``agibot_a3.py`` / ``training_contract.py``)、host-load 无依赖的 trainability
  叶子、直接解析智元 MJCF、读 ``cfg/algo/ppo.yaml``;
* mjlab 侧 —— 直接 import 本车道的 ``a3_train_ppo`` / ``a3_plant_env`` /
  ``a3_court_env``,读它们的模块常量与 dataclass 默认值。

所以源码一动这张表就动。**没有一行是"钉个 SHA 就能闭嘴"的**:每一行的裁定
(``declared``)会跟当场量出来的裁定(``observed``)对账,对不上就当场炸,而
``observed`` 是从两边活值算的,不是从任何一份手抄件读的。这条正是 MEMORY 里
「指纹不等于语义一致」和「改软硬门要连证据一起改」两条准绳的落地。

边界(不要过度宣称)
------------------
* 它不证明"补上这些差异之后两边就会学出同一个策略"。
* 它不做逐位对拍,**也不允许别人做**:mujoco-warp 无 CPU 回退且实测非确定性
  (§9.2.0,``pendula`` 无接触也发散),所以任何"逐位一致"的验收都是错的。
  :func:`assert_cross_engine_claim` 见到就拒。
* Isaac 侧的 reward 权重可以在发车时被 reward-pack YAML 覆盖,所以本表读到的是
  **类体里声明的那一份**,这一点在该行的 ``caveat`` 里写明。
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Verdict vocabulary.  Closed on purpose -- a free-text verdict is a verdict
# nobody can machine-check.
# ---------------------------------------------------------------------------

#: Both sides resolve to the same live value.  Re-derived every call.
ALIGNED = "aligned"

#: The two lanes differ, and the difference changes what question is being
#: asked, so an mjlab number cannot stand in for an Isaac number.  Any row in
#: this class blocks :func:`assert_cross_engine_claim`.
DIVERGENT_BLOCKING = "divergent_blocking"

#: The lanes differ on purpose.  Name the accepted scope and any narrower claim
#: restrictions; this verdict does not enter ``blocking_axes``.
DIVERGENT_DECLARED = "divergent_declared"

#: This lane cannot read one of the two sides at all (missing artifact, missing
#: API, no repo checkout next to the deployed copy).  Blocks, because "I could
#: not check" must never read like "I checked and it matched".
UNVERIFIABLE = "unverifiable"

VERDICTS = (ALIGNED, DIVERGENT_BLOCKING, DIVERGENT_DECLARED, UNVERIFIABLE)

#: Verdict classes that forbid a cross-engine comparability claim.
BLOCKING_VERDICTS = (DIVERGENT_BLOCKING, UNVERIFIABLE)


class AlignmentError(RuntimeError):
    """The alignment ledger no longer describes the two lanes it claims to."""


class AlignmentClaimRefused(AlignmentError):
    """A receipt tried to claim more cross-engine authority than it has."""


# ---------------------------------------------------------------------------
# Locating the Isaac side.
#
# The deployed copy of this lane lives at /workspace/mjlab_lane on the pod and
# is NOT inside a repo checkout.  That is exactly the case this must fail
# closed on: without the Isaac sources there is nothing to compare against, and
# a receipt written there must say UNVERIFIABLE rather than nothing at all.
# ---------------------------------------------------------------------------

_ISAAC_REL = Path("hope_training/whole_body_tracking")
_TRACKING_REL = _ISAAC_REL / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
_GEOM_REL = _ISAAC_REL / "source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py"

ISAAC_SOURCE_RELPATHS = {
    "hope_env_cfg": _TRACKING_REL / "config/agibot_a3/hope_env_cfg.py",
    "tracking_env_cfg": _TRACKING_REL / "tracking_env_cfg.py",
    "a211_leaf": _TRACKING_REL / "action_ball_a211_trainability.py",
    "c211_leaf": _TRACKING_REL / "action_ball_c211_trainability.py",
    "robot_cfg": _ISAAC_REL / "source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py",
    "hope_actions": _TRACKING_REL / "mdp/hope_actions.py",
    "training_contract": _ISAAC_REL / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py",
    "ppo_yaml": _ISAAC_REL / "cfg/algo/ppo.yaml",
    "train_entry": _ISAAC_REL / "scripts/train.py",
    "geometry": _GEOM_REL,
    "joint_order_contract": Path("configs/a3_joint_order_bijection_v1.json"),
    "gmr_joint_order": Path("configs/a3_gmr_dof_pos_joint_order.txt"),
    "runtime_joint_order": Path("configs/a3_runtime_articulation_joint_order.txt"),
    "ready_pose": Path(
        "configs/action_ball_n1_measured_20260803/"
        "evidence_holdpass_robust20n_20260803/"
        "take061.measured_teacher.yaw_aligned_full_seed."
        "robust20n.dynamic_ready.v2.json"
    ),
    "vendor_mjcf": Path(
        "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"),
}


def resolve_repo_root() -> Path | None:
    """Walk up from this file for a checkout that actually holds the sources.

    ``HOPE_REPO_ROOT`` overrides, so the pod-deployed copy can be pointed at a
    worktree without being moved into one.
    """

    override = os.environ.get("HOPE_REPO_ROOT")
    candidates = [Path(override)] if override else []
    candidates += list(_HERE.parents)
    for root in candidates:
        if (root / ISAAC_SOURCE_RELPATHS["hope_env_cfg"]).is_file():
            return root
    return None


def isaac_sources(root: Path | None = None) -> dict:
    """``{name: Path}`` for every live Isaac source, plus what is missing."""

    root = root if root is not None else resolve_repo_root()
    if root is None:
        return {"root": None, "paths": {}, "missing": sorted(ISAAC_SOURCE_RELPATHS)}
    paths, missing = {}, []
    for name, rel in ISAAC_SOURCE_RELPATHS.items():
        path = root / rel
        if path.is_file():
            paths[name] = path
        else:
            missing.append(name)
    return {"root": root, "paths": paths, "missing": sorted(missing)}


# ---------------------------------------------------------------------------
# A whitelist literal evaluator over one module's AST.
#
# Same paradigm as ``mujoco_native/isaac_live_constants.py``: read the VALUE,
# not a hash of the bytes.  Re-pinning a digest cannot satisfy any of this.
# We do not import the Isaac modules -- ``hope_env_cfg.py`` pulls the whole of
# Isaac Lab and does not import on a plain host or in the mjlab venv.
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise AlignmentError(f"cannot parse live Isaac source {path}") from exc


def _module_consts(tree: ast.Module) -> dict:
    """Module-level ``name -> value node``; a rebound name is refused."""

    found: dict = {}
    duplicated: set = set()
    for statement in tree.body:
        value = getattr(statement, "value", None)
        if value is None:
            continue
        targets: list = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id in found:
                    duplicated.add(target.id)
                found[target.id] = value
    for name in duplicated:
        found.pop(name, None)
    return found


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal(node: ast.AST, consts: dict, depth: int = 0) -> Any:
    """Evaluate one node as a literal, or raise.  Whitelist only."""

    if depth > 12:
        raise AlignmentError("literal nesting too deep to evaluate safely")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literal(item, consts, depth + 1) for item in node.elts]
    if isinstance(node, ast.Dict):
        out = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise AlignmentError("dict unpacking is not a literal")
            out[_literal(key, consts, depth + 1)] = _literal(value, consts, depth + 1)
        return out
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand, consts, depth + 1)
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal(node.left, consts, depth + 1)
        right = _literal(node.right, consts, depth + 1)
        if isinstance(left, list) and isinstance(right, list):
            return left + right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left + right
        raise AlignmentError("unsupported '+' operands")
    if isinstance(node, ast.Name):
        if node.id not in consts:
            raise AlignmentError(f"name {node.id!r} is not a module-level literal")
        return _literal(consts[node.id], consts, depth + 1)
    if isinstance(node, ast.Call):
        callee = _dotted(node.func)
        if callee in ("list", "tuple") and len(node.args) == 1 and not node.keywords:
            return list(_literal(node.args[0], consts, depth + 1))
    raise AlignmentError(f"node {type(node).__name__} is not an evaluable literal")


def _call_kwargs(node: ast.AST) -> tuple[str, dict]:
    """``(dotted callee, {kwarg name: value node})`` for a call node."""

    if not isinstance(node, ast.Call):
        raise AlignmentError("expected a call expression")
    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
    return _dotted(node.func), kwargs


def _class_body(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for statement in ast.walk(tree):
        if isinstance(statement, ast.ClassDef) and statement.name == class_name:
            return statement
    raise AlignmentError(f"class {class_name!r} not found in the live source")


def _class_assignments(tree: ast.Module, class_name: str) -> dict:
    """``attr -> value node`` for the class body's own assignments, in order."""

    out: dict = {}
    for statement in _class_body(tree, class_name).body:
        value = getattr(statement, "value", None)
        if value is None:
            continue
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = value
    return out


def _function_return_call(tree: ast.Module, func_name: str) -> ast.Call:
    """The single ``return <Call>(...)`` of a module-level function."""

    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef) and statement.name == func_name:
            for node in ast.walk(statement):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                    return node.value
    raise AlignmentError(f"function {func_name!r} has no returned call to read")


def _post_init_self_attrs(tree: ast.Module, class_name: str) -> dict:
    """``"decimation" -> 4``, ``"sim.dt" -> 0.005`` from a ``__post_init__``.

    Isaac writes the two numbers that define the control rate inside
    ``__post_init__``, so a module-level reader would silently find nothing and
    a reader that fell back to a default would invent an answer.
    """

    consts = _module_consts(tree)
    out: dict = {}
    for statement in _class_body(tree, class_name).body:
        if not (isinstance(statement, ast.FunctionDef)
                and statement.name == "__post_init__"):
            continue
        for node in ast.walk(statement):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                dotted = _dotted(target)
                if not dotted.startswith("self."):
                    continue
                try:
                    out[dotted[len("self."):]] = _literal(node.value, consts)
                except AlignmentError:
                    continue
    return out


def _live_nodes(root: ast.AST):
    """Same-scope nodes, pruning nested helpers and constant-dead branches."""
    yield root
    children = list(ast.iter_child_nodes(root))
    if isinstance(root, ast.If) and isinstance(root.test, ast.Constant):
        children = root.body if bool(root.test.value) else root.orelse
    for child in children:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda, ast.ClassDef)):
            yield from _live_nodes(child)


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    rows = [n for n in _class_body(tree, cls).body
            if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(rows) != 1:
        raise AlignmentError(f"{cls}.{name} does not have one live owner")
    return rows[0]


def _has_assign(fn: ast.FunctionDef, target: str, expression: str) -> bool:
    wanted = ast.dump(ast.parse(expression, mode="eval").body,
                      include_attributes=False)
    binds = lambda n: (_dotted(n) == target or (  # noqa: E731
        isinstance(n, (ast.Tuple, ast.List)) and any(binds(x) for x in n.elts)))
    for node in _live_nodes(fn):
        if isinstance(node, ast.Assign) and any(
            binds(t) for t in node.targets
        ) and ast.dump(node.value, include_attributes=False) == wanted:
            return True
    return False


def _one_arg_call(fn: ast.FunctionDef, callee: str, arg: str) -> bool:
    rows = [n for n in _live_nodes(fn) if isinstance(n, ast.Call)
            and _dotted(n.func) == callee]
    return (len(rows) == 1 and not rows[0].keywords and len(rows[0].args) == 1
            and isinstance(rows[0].args[0], ast.Name)
            and rows[0].args[0].id == arg)


# ---------------------------------------------------------------------------
# Isaac readers.
# ---------------------------------------------------------------------------


def load_live_leaf(path: Path):
    """Host-load one Isaac trainability leaf straight off disk.

    The A/C leaves import only ``math``/``typing``, so this works in the mjlab
    venv with no Isaac Lab present.  It is a real import of the real authority,
    which is stronger than any AST read: the layouts below are the ones the
    Isaac run itself asserts on.
    """

    unique = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        f"mjlab_alignment_live_{path.stem}_{unique}", path)
    if spec is None or spec.loader is None:
        raise AlignmentError(f"cannot host-load live Isaac leaf {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - defensive
        sys.modules.pop(spec.name, None)
        raise AlignmentError(f"live Isaac leaf {path} did not import: {exc!r}") from exc
    return module


#: The Isaac termination class chain, base first.  Later entries override
#: earlier ones by attribute name, exactly like the configclass MRO does.
ISAAC_TERMINATION_CHAIN = (
    ("tracking_env_cfg", "TerminationsCfg"),
    ("hope_env_cfg", "HOPEDeployParityTerminationsCfg"),
    ("hope_env_cfg", "HOPEActionBallTerminationsCfg"),
)

#: The reward class chain used by the A211 construction leaf.  C211 adds two
#: more terms on top; both are read so neither can drift unseen.
ISAAC_REWARD_CHAIN = (
    ("tracking_env_cfg", "RewardsCfg"),
    ("hope_env_cfg", "HOPERewardsCfg"),
    ("hope_env_cfg", "HOPEDeployParityRewardsCfg"),
    ("hope_env_cfg", "HOPEVirtualBallRewardsCfg"),
    ("hope_env_cfg", "HOPEActionBallRewardsCfg"),
)
ISAAC_REWARD_CHAIN_C211 = ISAAC_REWARD_CHAIN + (
    ("hope_env_cfg", "HOPEActionBallC211RewardsCfg"),
)


def isaac_terminations(paths: dict) -> dict:
    """Resolved ``robot_hit_table``-inclusive termination union, live.

    Each entry is ``{"func": dotted, "time_out": bool, "params": {...}}``.  A
    term declared as a factory call (``robot_hit_table = table_hit_done_term()``)
    is followed into the factory's ``return DoneTerm(...)``.
    """

    trees = {name: _parse(paths[name]) for name in ("tracking_env_cfg", "hope_env_cfg")}
    resolved: dict = {}
    for module_name, class_name in ISAAC_TERMINATION_CHAIN:
        tree = trees[module_name]
        consts = _module_consts(tree)
        for attr, node in _class_assignments(tree, class_name).items():
            call = node
            if isinstance(call, ast.Call) and not call.keywords and not call.args:
                # A zero-argument factory: follow it into its returned DoneTerm.
                factory = _dotted(call.func)
                if factory and "." not in factory:
                    call = _function_return_call(tree, factory)
            callee, kwargs = _call_kwargs(call)
            entry: dict = {"func": _dotted(kwargs["func"]) if "func" in kwargs else callee,
                           "time_out": False, "params": {}}
            if "time_out" in kwargs:
                entry["time_out"] = bool(_literal(kwargs["time_out"], consts))
            if "params" in kwargs and isinstance(kwargs["params"], ast.Dict):
                for key, value in zip(kwargs["params"].keys, kwargs["params"].values):
                    name = _literal(key, consts)
                    try:
                        entry["params"][name] = _literal(value, consts)
                    except AlignmentError:
                        # SceneEntityCfg(...) and friends are not literals; the
                        # ledger only compares the numeric thresholds.
                        entry["params"][name] = "<non-literal>"
            resolved[attr] = entry
    return resolved


def isaac_reward_terms(paths: dict, chain=ISAAC_REWARD_CHAIN_C211) -> dict:
    """``term -> {"func": dotted, "weight": float|None}`` for the live chain."""

    trees = {name: _parse(paths[name]) for name in ("tracking_env_cfg", "hope_env_cfg")}
    resolved: dict = {}
    for module_name, class_name in chain:
        tree = trees[module_name]
        consts = _module_consts(tree)
        for attr, node in _class_assignments(tree, class_name).items():
            if not isinstance(node, ast.Call):
                continue
            callee, kwargs = _call_kwargs(node)
            if not callee.endswith("RewTerm"):
                continue
            weight = None
            if "weight" in kwargs:
                try:
                    weight = float(_literal(kwargs["weight"], consts))
                except AlignmentError:
                    weight = None
            resolved[attr] = {
                "func": _dotted(kwargs["func"]) if "func" in kwargs else callee,
                "weight": weight,
            }
    return resolved


def isaac_actuator_table(paths: dict) -> dict:
    """``{group: {"joints": [...], "effort": {...}, "kp": {...}, "kd": {...}}}``."""

    tree = _parse(paths["robot_cfg"])
    consts = _module_consts(tree)
    if "AGIBOT_A3_CFG" not in consts:
        raise AlignmentError("AGIBOT_A3_CFG is not a module-level assignment")
    _callee, kwargs = _call_kwargs(consts["AGIBOT_A3_CFG"])
    if "actuators" not in kwargs or not isinstance(kwargs["actuators"], ast.Dict):
        raise AlignmentError("AGIBOT_A3_CFG has no literal actuators mapping")
    out: dict = {}
    node = kwargs["actuators"]
    for key, value in zip(node.keys, node.values):
        group = _literal(key, consts)
        _cls, akw = _call_kwargs(value)
        entry = {"joints": _literal(akw["joint_names_expr"], consts)}
        for field, alias in (("effort_limit_sim", "effort"),
                             ("stiffness", "kp"), ("damping", "kd")):
            entry[alias] = _literal(akw[field], consts)
        out[group] = entry
    return out


def mjcf_actuated_joints(paths: dict) -> list:
    """Ordered joint names driven by a ``<motor>`` in the vendor MJCF."""

    root = ET.parse(paths["vendor_mjcf"]).getroot()
    names = [m.get("joint") for m in root.iter("motor") if m.get("joint")]
    if not names:
        raise AlignmentError("vendor MJCF declares no <motor> actuators")
    return names


def mjcf_ctrlrange(paths: dict) -> dict:
    """``joint -> +|ctrlrange|`` from the vendor MJCF.

    Legitimate to read as text rather than from a compiled model: the plant
    audit's 92 matched field groups include ``actuator_ctrlrange [31 rows]``,
    i.e. the compiled model carries these bytes through unchanged.
    """

    root = ET.parse(paths["vendor_mjcf"]).getroot()
    out = {}
    for motor in root.iter("motor"):
        joint, rng = motor.get("joint"), motor.get("ctrlrange")
        if joint and rng:
            lo, hi = (float(v) for v in rng.split())
            out[joint] = max(abs(lo), abs(hi))
    return out


def _expand_over_joints(table: Any, patterns: list, joints: list, what: str) -> dict:
    """Expand one Isaac actuator field (dict of regex, or a scalar) per joint.

    Fails closed on a joint no pattern matches and on a joint two patterns
    match: both are states in which "the Isaac value for this joint" has no
    single answer, and guessing one is how a comparison silently becomes wrong.
    """

    if not isinstance(table, dict):
        table = {pattern: table for pattern in patterns}
    out: dict = {}
    for joint in joints:
        hits = [value for pattern, value in table.items()
                if re.fullmatch(pattern, joint)]
        if len(hits) != 1:
            raise AlignmentError(
                f"{what}: joint {joint!r} matched {len(hits)} Isaac patterns")
        out[joint] = float(hits[0])
    return out


def isaac_per_joint(paths: dict) -> dict:
    """``{"kp": {...}, "kd": {...}, "effort": {...}, "action_scale": {...}}``.

    ``action_scale`` reproduces ``AGIBOT_A3_ACTION_SCALE`` -- ``0.25 * effort /
    kp`` per joint -- from the same literals the Isaac module builds it from.
    """

    joints = mjcf_actuated_joints(paths)
    groups = isaac_actuator_table(paths)
    out = {"kp": {}, "kd": {}, "effort": {}}
    for group, entry in groups.items():
        mine = [j for j in joints
                if any(re.fullmatch(p, j) for p in entry["joints"])]
        for alias in ("kp", "kd", "effort"):
            out[alias].update(
                _expand_over_joints(entry[alias], entry["joints"], mine,
                                    f"{group}.{alias}"))
    unresolved = [j for j in joints if j not in out["kp"]]
    if unresolved:
        raise AlignmentError(f"no Isaac actuator group covers joints {unresolved}")
    out["action_scale"] = {j: 0.25 * out["effort"][j] / out["kp"][j] for j in joints}
    out["joints"] = joints
    return out


def isaac_ppo_params(paths: dict) -> dict:
    import yaml  # pyyaml ships in both the Isaac and the mjlab venv

    with open(paths["ppo_yaml"], "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# mjlab side.  Plain imports -- this IS the lane.
# ---------------------------------------------------------------------------


def _load_lane_module(name: str):
    path = _HERE / f"{name}.py"
    if not path.is_file():
        raise AlignmentError(f"mjlab lane module {name} is missing at {path}")
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mjlab_side():
    """Everything the ledger needs from this lane, read live."""

    sys.path.insert(0, str(_HERE))
    try:
        train = _load_lane_module("a3_train_ppo")
    finally:
        if sys.path and sys.path[0] == str(_HERE):
            sys.path.pop(0)
    plant = train.court.plant
    return {"train": train, "court": train.court, "plant": plant,
            "geom": train.geom, "task": train.TaskCfg(), "sim": train.SimCfg()}


# ---------------------------------------------------------------------------
# The ledger.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """One alignment question, its declared verdict, and how to re-derive it."""

    key: str
    human: str                       # 一行人话,给不读代码的人
    declared: str                    # the verdict this repo currently asserts
    why: str                         # why the difference does or does not matter
    closable_by: str                 # what would actually close it, or "" if closed
    probe: Callable[[dict, dict], dict]
    caveat: str = ""

    def __post_init__(self) -> None:
        if self.declared not in VERDICTS:
            raise AlignmentError(f"axis {self.key}: verdict {self.declared!r} is not in the vocabulary")
        if self.declared != ALIGNED and not self.why:
            raise AlignmentError(f"axis {self.key}: a divergence must say why it matters")
        if self.declared in BLOCKING_VERDICTS and not self.closable_by:
            raise AlignmentError(f"axis {self.key}: a blocking row must say what would close it")


def _row(isaac: Any, mjlab: Any, observed: str, **extra) -> dict:
    out = {"isaac": isaac, "mjlab": mjlab, "observed": observed}
    out.update(extra)
    return out


# -- probes -----------------------------------------------------------------


def _probe_actor_abi(paths: dict, lane: dict) -> dict:
    leaf = load_live_leaf(paths["a211_leaf"])
    isaac = [[str(n), int(w)] for n, w in leaf.A211_ACTOR_LAYOUT]
    mjlab = [[str(n), int(w)] for n, w in lane["train"].OBS_LAYOUT]
    same = isaac == mjlab
    return _row({"width": leaf.A211_ACTOR_WIDTH, "layout": isaac},
                {"width": lane["train"].OBS_WIDTH, "layout": mjlab},
                ALIGNED if same else DIVERGENT_BLOCKING)


def _probe_critic_abi(paths: dict, lane: dict) -> dict:
    leaf = load_live_leaf(paths["a211_leaf"])
    isaac_width = int(leaf.A211_CRITIC_WIDTH)
    cfg = lane["train"].build_agent_cfg(seed=0, iterations=1, num_steps_per_env=1,
                                        experiment="alignment_probe")
    groups = cfg["obs_groups"]
    symmetric = groups.get("critic") == groups.get("actor")
    return _row({"width": isaac_width, "asymmetric": True,
                 "privileged_rows": ["command(62)", "body_pos(42)", "body_ori(84)",
                                     "motion_anchor_pos_b(3)", "motion_anchor_ori_b(6)"]},
                {"width": lane["train"].OBS_WIDTH, "asymmetric": not symmetric,
                 "obs_groups": groups},
                DIVERGENT_BLOCKING if symmetric else ALIGNED)


def _probe_raw_action_affine(paths: dict, lane: dict) -> dict:
    raw = paths["joint_order_contract"].read_bytes()
    contract, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    train_tree = _parse(_HERE / "a3_train_ppo.py")
    train_consts = _module_consts(train_tree)
    names = lambda p: [r.strip() for r in p.read_text().splitlines()  # noqa: E731
                       if r.strip() and not r.lstrip().startswith("#")]
    source, target = names(paths["gmr_joint_order"]), names(paths["runtime_joint_order"])
    r, g = (contract["target_from_source_indices"],
            contract["source_from_target_indices"])
    if (digest != _literal(train_consts["ACTION_JOINT_ORDER_CONTRACT_SHA256"], train_consts)
            or contract["contract_id"] != _literal(
                train_consts["ACTION_JOINT_ORDER_CONTRACT_ID"], train_consts)
            or hashlib.sha256(paths["gmr_joint_order"].read_bytes()).hexdigest()
            != contract["source_order"]["file_sha256"]
            or hashlib.sha256(paths["runtime_joint_order"].read_bytes()).hexdigest()
            != contract["target_order"]["file_sha256"]
            or source != mjcf_actuated_joints(paths) or len(set(target)) != 31
            or target != [source[i] for i in r] or sorted(r) != list(range(31))
            or any(g[s] != i for i, s in enumerate(r))):
        raise AlignmentError("raw_action_affine tracked joint-order contract differs")
    per_joint = isaac_per_joint(paths)
    scale = [per_joint["action_scale"][j] for j in target]
    ready_raw = paths["ready_pose"].read_bytes()
    ready, runner_tree = json.loads(ready_raw), _parse(
        _HERE / "mujoco_gpu_ac_full_mdp_wait_rsl3.py")
    plant = ready["runtime_plant"]
    offset = plant["default_joint_pos_rad"]
    init, advance = (_method(train_tree, "A3ReadyBallVecEnv", n)
                     for n in ("__init__", "_advance_plant"))
    required = (
        (init, "action_offset_np", "_runtime_action_offset(pose_payload, action_joint_names)"),
        (init, "kp_np", "kp_actuator[runtime_from_actuator]"),
        (init, "kd_np", "kd_actuator[runtime_from_actuator]"),
        (init, "q_adr_act", "q_adr_actuator[runtime_from_actuator]"),
        (init, "v_adr_act", "v_adr_actuator[runtime_from_actuator]"),
        (init, "jnt_of_act", "m.actuator_trnid[:, 0].astype(int)[runtime_from_actuator]"),
        (init, "jrange", "m.jnt_range[jnt_of_act]"),
        (init, "self.kp", "T(kp_np)"),
        (init, "self.tau_hi", "T(m.actuator_ctrlrange[runtime_from_actuator, 1])"),
        (init, "self.act_scale", "0.25 * self.tau_hi / self.kp"),
        (init, "self.action_offset", "T(action_offset_np)"),
        (init, "self.jnt_lo", "T(jrange[:, 0])"),
        (init, "self.jnt_hi", "T(jrange[:, 1])"),
        (advance, "incoming", "actions.to(self.device)"),
        (advance, "pre_clamp_qdes", "self.action_offset.unsqueeze(0) + self.act_scale * incoming"),
    )
    ctrl_rows = [n for n in _live_nodes(advance) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Subscript) and _dotted(t.value) == "d.ctrl"
                         for t in n.targets)]
    ctrl_ok = len(ctrl_rows) == 1 and ast.dump(
        ctrl_rows[0].value, include_attributes=False) == ast.dump(
        ast.parse("tau[:, self.actuator_from_runtime]", mode="eval").body,
        include_attributes=False)
    runner_main = next(n for n in runner_tree.body
                       if isinstance(n, ast.FunctionDef) and n.name == "main")
    vendor = any(isinstance(n, ast.Call) and _dotted(n.func) == "wait.TaskCfg"
                 and any(k.arg == "action_scale_mode"
                         and isinstance(k.value, ast.Constant) and k.value.value == "vendor"
                         for k in n.keywords) for n in _live_nodes(runner_main))
    offset_owner = next(n for n in train_tree.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "_runtime_action_offset")
    build_cfg = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "build_agent_cfg")
    runner_cfg_calls = [n for n in _live_nodes(build_cfg) if isinstance(n, ast.Call)
                        and _dotted(n.func) == "RslRlOnPolicyRunnerCfg"]
    clip_none = (len(runner_cfg_calls) == 1 and any(
        k.arg == "clip_actions" and isinstance(k.value, ast.Constant)
        and k.value.value is None for k in runner_cfg_calls[0].keywords))
    tracking = _parse(paths["tracking_env_cfg"])
    _, action_kw = _call_kwargs(_class_assignments(tracking, "ActionsCfg")["joint_pos"])
    wait = _parse(_HERE / "mujoco_gpu_ac_full_mdp_initial_wait_env.py")
    wrappers_ok = all(_one_arg_call(
        _method(wait, "FullMdpInitialWaitVecEnv", name),
        "self._advance_plant", "actions") for name in ("step", "_step_full_a"))
    isaac_main = next(n for n in _parse(paths["train_entry"]).body if
                      isinstance(n, ast.FunctionDef) and n.name == "_run_with_environment_close_owner")
    if (_literal(_class_assignments(train_tree, "TaskCfg")["action_clip"],
                 train_consts) is not None
            or not clip_none
            or not all(_has_assign(*row) for row in required) or not ctrl_ok or not vendor
            or not wrappers_ok
            or not _one_arg_call(isaac_main, "RslRlVecEnvWrapper", "env")
            or not _has_assign(offset_owner, "values",
                               'runtime_plant["default_joint_pos_rad"]')
            or _literal(action_kw["use_default_offset"], _module_consts(tracking)) is not True
            or plant["joint_names"] != target or plant["action_joint_ids"] != list(range(31))
            or any(abs(a - b) > 2e-7 for a, b in zip(plant["action_scale_rad"], scale))
            or hashlib.sha256(ready_raw).hexdigest()
            != _literal(_module_consts(runner_tree)["READY_POSE_SHA256"],
                        _module_consts(runner_tree))):
        raise AlignmentError("raw_action_affine production callpath differs")
    common = {"raw_policy_clip": None, "joint_order_contract_sha256": digest,
              "joint_order": "runtime_articulation_joint_pos",
              "scale": scale, "offset": offset,
              "offset_source": "runtime_plant.default_joint_pos_rad"}
    return _row(common, dict(common, final_ctrl_order="GMR actuator"), ALIGNED,
                components={k: ALIGNED for k in
                            ("raw_policy_action", "runtime_joint_order", "scale", "offset")})


def _probe_executable_qdes_guard(paths: dict, lane: dict) -> dict:
    attrs = _post_init_self_attrs(_parse(paths["hope_env_cfg"]),
                                  "HOPEPingPongActionBallAgibotA3EnvCfg")
    expected_attrs = {
        "actions.joint_pos.pre_apply_limit_guard": True,
        "actions.joint_pos.pre_apply_guard_policy_dt_s": 0.02,
        "actions.joint_pos.pre_apply_guard_expected_decimation": 4,
        "actions.joint_pos.pre_apply_guard_margin_rad": 0.0,
        "actions.joint_pos.pre_apply_guard_margin_fraction": 0.05,
        "actions.joint_pos.project_finite_preclamp_qdes_without_termination": True,
    }
    if any(attrs.get(name) != value for name, value in expected_attrs.items()):
        raise AlignmentError("executable_qdes_guard active Isaac settings differ")

    actions_tree, train_tree = (_parse(paths["hope_actions"]),
                                _parse(_HERE / "a3_train_ppo.py"))
    cfg, consts = (_class_assignments(actions_tree, "ClampedJointPositionActionCfg"),
                   _module_consts(actions_tree))
    process = _method(actions_tree, "ClampedJointPositionAction", "process_actions")
    required = (
        ("inset", "self._pre_apply_guard_margin_rad + "
         "self._pre_apply_guard_margin_fraction * hard_travel"),
        ("hard_inner_lower", "hard_lower + inset"),
        ("hard_inner_upper", "hard_upper - inset"),
        ("target_lower", "torch.maximum(lower, hard_inner_lower)"),
        ("target_upper", "torch.minimum(upper, hard_inner_upper)"),
        ("projection_inset", "self._finite_projection_soft_envelope_inset_fraction * travel"),
        ("target_lower", "torch.maximum(target_lower, lower + projection_inset)"),
        ("target_upper", "torch.minimum(target_upper, upper - projection_inset)"),
        ("ballistic_next", "safe_joint_pos + safe_joint_vel * self._pre_apply_guard_policy_dt_s"),
        ("crossing_violation", "~state_finite | safe_joint_pos.le(hard_inner_lower) | "
         "safe_joint_pos.ge(hard_inner_upper) | ballistic_next.le(hard_inner_lower) | "
         "ballistic_next.ge(hard_inner_upper)"),
        ("per_joint_guard", "qdes_safety_violation | crossing_violation"),
        ("brake_target", "torch.clamp(safe_joint_pos - safe_joint_vel * "
         "self._pre_apply_guard_policy_dt_s, min=target_lower, max=target_upper)"),
        ("self._processed_actions", "torch.where(per_joint_guard, brake_target, nominal_target)"),
    )
    advance = _method(train_tree, "A3ReadyBallVecEnv", "_advance_plant")
    qdes_owners = [n for n in _live_nodes(advance) if isinstance(n, ast.Assign)
                   and any(_dotted(t) == "q_des" for t in n.targets)]
    qdes = "torch.clamp(self.action_offset.unsqueeze(0) + self.act_scale * "
    qdes += "safe_actions, self.jnt_lo, self.jnt_hi)"
    if (_literal(cfg["clamp"], consts) is not True
            or _literal(cfg["finite_projection_soft_envelope_inset_fraction"], consts) != .05
            or _literal(cfg["pre_apply_guard_brake_mode"], consts) != "velocity_horizon_v1"
            or not all(_has_assign(process, *row) for row in required)
            or len(qdes_owners) != 1 or not _has_assign(advance, "q_des", qdes)):
        raise AlignmentError("executable_qdes_guard live descriptor differs")
    return _row(
        {"finite_proposal": "soft∩hard 5%-inset projection",
         "state_guard": "q/qdot 0.02s prediction and brake"},
        {"finite_proposal": "hard joint-range clamp only", "state_guard": None},
        DIVERGENT_DECLARED,
        claim_restrictions=["policy_transfer", "promotion", "matched_causal_comparison"],
        allowed_scope="MuJoCo-only diagnostic_unauthorized 4096x24x25000")


def _probe_pd_gains(paths: dict, lane: dict) -> dict:
    per_joint = isaac_per_joint(paths)
    joints = per_joint["joints"]
    kp, kd = lane["plant"].vendor_pd_for_joint_names(joints)
    mjlab_kp = {j: float(v) for j, v in zip(joints, kp)}
    mjlab_kd = {j: float(v) for j, v in zip(joints, kd)}
    kp_diff = sorted(j for j in joints if mjlab_kp[j] != per_joint["kp"][j])
    kd_diff = sorted(j for j in joints if mjlab_kd[j] != per_joint["kd"][j])
    return _row({"kp": per_joint["kp"], "kd": per_joint["kd"]},
                {"kp": mjlab_kp, "kd": mjlab_kd},
                ALIGNED if not (kp_diff or kd_diff) else DIVERGENT_BLOCKING,
                joints_differing_kp=kp_diff, joints_differing_kd=kd_diff)


def _probe_effort_limits(paths: dict, lane: dict) -> dict:
    per_joint = isaac_per_joint(paths)
    ctrlrange = mjcf_ctrlrange(paths)
    diff = sorted(j for j in per_joint["joints"]
                  if ctrlrange[j] != per_joint["effort"][j])
    return _row({"source": "effort_limit_sim", "n": len(per_joint["effort"])},
                {"source": "vendor MJCF <motor ctrlrange>", "n": len(ctrlrange)},
                ALIGNED if not diff else DIVERGENT_BLOCKING,
                joints_differing=diff)


#: Every Isaac terminal predicate, mapped onto this lane's terminal predicate
#: or explicitly onto ``None`` with a reason.  Enumeration, not a spot check:
#: a termination added upstream lands here unmapped and the ledger goes red.
ISAAC_TO_MJLAB_TERMINATION = {
    "time_out": ("timeout_truncation", "both truncate on the episode horizon"),
    "anchor_pos": (None, "needs a motion reference; this lane has no teacher"),
    "anchor_ori": (None, "needs a motion reference; this lane has no teacher"),
    "ee_body_pos": (None, "needs a motion reference; this lane has no teacher"),
    "base_fell_tilt": ("fall_tilt", "same quantity, different threshold"),
    "base_too_low": ("fall_height", "same quantity, different threshold"),
    "robot_hit_table": (None, "this lane has a table in the scene and no "
                              "terminal table guard; since 2026-08-06 it "
                              "MEASURES robot-vs-table contact per episode and "
                              "--report refuses a run whose rate is non-zero, "
                              "which is evidence plus a block, not a guard"),
    "joint_qdes_forbidden": (None, "this lane hard-clamps q_des to jnt_range instead"),
    "joint_actual_forbidden": (None, "not a terminal upstream either (terminate=False)"),
}


def _probe_termination_union(paths: dict, lane: dict) -> dict:
    isaac = isaac_terminations(paths)
    unmapped = sorted(set(isaac) - set(ISAAC_TO_MJLAB_TERMINATION))
    if unmapped:
        raise AlignmentError(
            "Isaac grew termination terms this ledger does not classify: "
            f"{unmapped}.  Add them to ISAAC_TO_MJLAB_TERMINATION with a reason.")
    stale = sorted(set(ISAAC_TO_MJLAB_TERMINATION) - set(isaac))
    if stale:
        raise AlignmentError(
            f"ISAAC_TO_MJLAB_TERMINATION names terms Isaac no longer has: {stale}")
    mjlab_terms = list(lane["train"].TERMINATION_TERMS)
    missing = sorted(name for name, (twin, _why) in ISAAC_TO_MJLAB_TERMINATION.items()
                     if twin is None and not isaac[name].get("time_out"))
    isaac_view = {
        name: {"func": entry["func"], "time_out": entry["time_out"],
               "thresholds": {k: v for k, v in entry["params"].items()
                              if isinstance(v, (int, float)) and not isinstance(v, bool)}}
        for name, entry in isaac.items()}
    # The one Isaac terminal this lane can at least SEE.  Asserted live rather
    # than described in prose: the reason string above claims the lane measures
    # robot-vs-table contact and refuses reports on it, and this is what makes
    # that claim checkable instead of decorative.
    train = lane["train"]
    table_channel = {
        "measured_per_episode": hasattr(train, "robot_table_contact_fields"),
        "report_refusal_code": "ROBOT_LEANED_ON_THE_TABLE",
        "refusal_wired": "ROBOT_LEANED_ON_THE_TABLE" in _refusal_codes(train),
        "terminal": False,
    }
    if not (table_channel["measured_per_episode"] and table_channel["refusal_wired"]):
        raise AlignmentError(
            "ISAAC_TO_MJLAB_TERMINATION says this lane measures robot-vs-table "
            "contact and refuses reports on it, but the lane does not: "
            f"{table_channel}")
    return _row(isaac_view,
                {"terminal": mjlab_terms,
                 "truncation": ["timeout_truncation"],
                 "min_pelvis_z": float(lane["task"].min_pelvis_z),
                 "max_tilt_proj_g": float(lane["task"].max_tilt_proj_g),
                 "robot_table_channel": table_channel},
                ALIGNED if not missing else DIVERGENT_BLOCKING,
                isaac_terms_with_no_mjlab_twin=missing)


def _refusal_codes(train_module) -> tuple:
    """Every refusal code ``report_refusals`` can emit, read out of its source.

    Source-derived rather than exercised, so this works with no receipts to
    hand.  A code deleted upstream stops appearing here and the ledger row that
    depends on it fails closed.
    """

    path = Path(train_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    codes = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef)
                and node.name == "report_refusals"):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Tuple) and inner.elts:
                head = inner.elts[0]
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    if head.value.isupper():
                        codes.append(head.value)
    return tuple(codes)


def _probe_fall_thresholds(paths: dict, lane: dict) -> dict:
    """The two predicates that DO exist on both sides -- do the numbers agree?"""

    import math

    isaac = isaac_terminations(paths)
    limit_angle = float(isaac["base_fell_tilt"]["params"]["limit_angle"])
    min_height = float(isaac["base_too_low"]["params"]["minimum_height"])
    # Isaac's bad_orientation fires when the tilt exceeds limit_angle; the same
    # event in projected-gravity terms is proj_g_z > -cos(limit_angle).
    isaac_proj_g = -math.cos(limit_angle)
    mjlab_proj_g = float(lane["task"].max_tilt_proj_g)
    mjlab_height = float(lane["task"].min_pelvis_z)
    same = (abs(isaac_proj_g - mjlab_proj_g) <= 1e-12
            and abs(min_height - mjlab_height) <= 1e-12)
    return _row({"limit_angle_rad": limit_angle,
                 "equivalent_proj_g_z": isaac_proj_g,
                 "tilt_deg": math.degrees(limit_angle),
                 "minimum_height_m": min_height},
                {"max_tilt_proj_g": mjlab_proj_g,
                 "tilt_deg": math.degrees(math.acos(min(1.0, -mjlab_proj_g))),
                 "min_pelvis_z": mjlab_height},
                ALIGNED if same else DIVERGENT_BLOCKING)


#: The Isaac reward terms that DEFINE the ActionBall income hierarchy.  If one
#: of these disappears upstream the meaning of "reward groups differ" changed,
#: and this ledger must be re-read by a human rather than kept quiet.
ISAAC_REWARD_ANCHORS = (
    "base_position",            # target: commanded base goal
    "death_penalty",            # safety union charge
    "qdes_limit_barrier",       # command-side joint barrier
    "joint_limit",              # plant-side joint barrier
    "c225_strike_ball_paddle_center_proximity",   # strike guidance
    "virtual_landing",          # outcome
)


def _probe_reward_surface(paths: dict, lane: dict) -> dict:
    isaac = isaac_reward_terms(paths)
    missing_anchors = sorted(a for a in ISAAC_REWARD_ANCHORS if a not in isaac)
    if missing_anchors:
        raise AlignmentError(
            "the Isaac ActionBall reward anchors this ledger reasons about are "
            f"gone: {missing_anchors}.  The reward-group verdict is now unread.")
    mjlab_groups = dict(lane["train"].REWARD_TERM_GROUP)
    overlap = sorted(set(isaac) & set(mjlab_groups))
    coverage = {group: sorted(t for t, g in mjlab_groups.items() if g == group)
                for group in ("balance", "mimic", "strike", "target", "outcome",
                              "strike_guidance", "regularizer", "safety")}
    empty = sorted(g for g in ("mimic", "strike", "target", "outcome")
                   if not coverage[g])
    return _row({"n_terms": len(isaac), "terms": sorted(isaac),
                 "anchors": {a: isaac[a] for a in ISAAC_REWARD_ANCHORS}},
                {"n_terms": len(mjlab_groups), "terms": sorted(mjlab_groups),
                 "group_coverage": coverage},
                DIVERGENT_BLOCKING if (empty or overlap != []) else ALIGNED,
                shared_term_names=overlap,
                isaac_groups_with_zero_mjlab_terms=empty)


def _probe_episode_structure(paths: dict, lane: dict) -> dict:
    leaf = load_live_leaf(paths["a211_leaf"])
    wait = leaf.action_ball_211_wait_contract_facts()
    tree = _parse(paths["tracking_env_cfg"])
    attrs = _post_init_self_attrs(tree, "TrackingEnvCfg")
    task = lane["task"]
    mjlab_ticks = task.episode_length_s / (lane["plant"].DECIMATION
                                           * lane["plant"].AGIBOT_OPTION["timestep"])
    # Two independent conditions, both required: same horizon in policy ticks,
    # AND a WAIT/reveal schedule at all.  This lane has no WAIT, so `has_wait`
    # is False by construction -- written as a live check anyway so that adding
    # one later actually moves this row instead of leaving it hardcoded shut.
    has_wait = hasattr(lane["train"].A3ReadyBallVecEnv, "wait_ticks_buf")
    same = (abs(mjlab_ticks - wait["episode_horizon_ticks"]) <= 1e-9) and has_wait
    return _row({"episode_length_s": attrs.get("episode_length_s"),
                 "policy_dt_s": wait["policy_dt_s"],
                 "horizon_ticks": wait["episode_horizon_ticks"],
                 "wait_ticks": [wait["min_wait_ticks"], wait["max_wait_ticks"]],
                 "wait_masks_task_rows": wait["wait_task_ball_base_and_clocks_masked"],
                 "required_active_ticks": wait["required_active_ticks"]},
                {"episode_length_s": float(task.episode_length_s),
                 "horizon_ticks": float(mjlab_ticks),
                 "wait_ticks": None, "reveal": None,
                 "ball_reserve_after_s": float(task.ball_reserve_after_s)},
                ALIGNED if same else DIVERGENT_BLOCKING)


def _probe_control_rate(paths: dict, lane: dict) -> dict:
    tree = _parse(paths["tracking_env_cfg"])
    attrs = _post_init_self_attrs(tree, "TrackingEnvCfg")
    isaac_dt = float(attrs["sim.dt"])
    isaac_dec = int(attrs["decimation"])
    mj_dt = float(lane["plant"].AGIBOT_OPTION["timestep"])
    mj_dec = int(lane["plant"].DECIMATION)
    same_policy = abs(isaac_dt * isaac_dec - mj_dt * mj_dec) <= 1e-12
    return _row({"physics_dt": isaac_dt, "decimation": isaac_dec,
                 "policy_dt": isaac_dt * isaac_dec},
                {"physics_dt": mj_dt, "decimation": mj_dec,
                 "policy_dt": mj_dt * mj_dec},
                ALIGNED if same_policy else DIVERGENT_BLOCKING,
                physics_rate_differs=(isaac_dt != mj_dt))


def _probe_obs_noise_dr(paths: dict, lane: dict) -> dict:
    tree = _parse(paths["training_contract"])
    consts = _module_consts(tree)
    channels = _literal(consts["ACTION_BALL_DR_L0N_PROPRIO_NOISE_CHANNELS"], consts)
    identity_l0n = _literal(consts["ACTION_BALL_DR_L0N_IDENTITY"], consts)
    task = lane["task"]
    mjlab_noise = {"policy_observation_corruption": False,
                   "reset_joint_noise_rad": float(task.reset_joint_noise_rad),
                   "reset_root_xy_noise_m": float(task.reset_root_xy_noise_m),
                   "reset_root_yaw_noise_rad": float(task.reset_root_yaw_noise_rad),
                   "plant_randomization": False}
    # The four grid is exactly {corruption off, corruption on} with the plant
    # frozen, so a lane that adds reset randomisation is neither cell.
    reset_noise_on = any(v > 0.0 for k, v in mjlab_noise.items()
                         if k.startswith("reset_"))
    matches_a_cell = (not reset_noise_on) and (
        mjlab_noise["policy_observation_corruption"] in (False, True))
    return _row({"dr_l0n_identity": identity_l0n,
                 "dr_levels": ["DR-L0 (corruption off)", "DR-L0N (corruption on)"],
                 "proprio_noise_channels": channels,
                 "plant_randomization": False,
                 "reset_noise": "none (start_pose_ramp is DR-L1, off in the four grid)"},
                mjlab_noise,
                ALIGNED if matches_a_cell else DIVERGENT_BLOCKING,
                lane_adds_reset_randomisation=reset_noise_on)


def _probe_question_distribution(paths: dict, lane: dict) -> dict:
    leaf = load_live_leaf(paths["a211_leaf"])
    facts = leaf.action_ball_211_question_source_contract_facts(family="A211")
    sampler = facts["question_sampler"]
    serve = lane["geom"].ServeConfig.reachable_returner()
    mjlab = {"source": "uniform box, resampled every serve",
             "pos_x": list(serve.pos_x_range), "pos_y": list(serve.pos_y_range),
             "pos_z": list(serve.pos_z_range), "vel_x": list(serve.vel_x_range),
             "vel_y": list(serve.vel_y_range), "vel_z": list(serve.vel_z_range),
             "spin": list(serve.spin_range)}
    isaac = {k: sampler[k] for k in sorted(sampler)
             if isinstance(sampler[k], (str, bool, int, float))}
    return _row(isaac, mjlab, DIVERGENT_BLOCKING)


def _probe_vendor_plant_inheritance(paths: dict, lane: dict) -> dict:
    """Franco's standing rule: the MuJoCo plant inherits the vendor MJCF."""

    root = ET.parse(paths["vendor_mjcf"]).getroot()
    option = root.find("option")
    vendor = dict(option.attrib) if option is not None else {}
    ours = dict(lane["plant"].AGIBOT_OPTION)
    checked = {}
    for key, cast in (("timestep", float), ("noslip_iterations", int),
                      ("noslip_tolerance", float)):
        if key in vendor:
            checked[key] = {"vendor": cast(vendor[key]), "ours": ours.get(key)}
    ts_ok = ("timestep" in checked
             and float(checked["timestep"]["vendor"]) == float(ours["timestep"]))
    return _row({"option": vendor,
                 "note": "solver/iterations/ls_iterations are MuJoCo defaults, not vendor choices"},
                {"option": ours, "decimation": int(lane["plant"].DECIMATION)},
                ALIGNED if ts_ok else DIVERGENT_BLOCKING,
                registered_deviation="noslip_iterations: mujoco-warp implements no noslip pass",
                fields_checked=checked)


def _probe_ball_contact_model(paths: dict, lane: dict) -> dict:
    court = lane["court"]
    return _row({"physical_ball": False,
                 "path": "analytic flight/contact (physical_ball=false on the live C211 arm)",
                 "solref": None, "solimp": None, "restitution": None},
                {"physical_ball": True,
                 "ball_solref": list(court.BALL_SOLREF),
                 "ball_solimp": list(court.BALL_SOLIMP),
                 "ball_friction": list(court.COURT_FRICTION),
                 "ball_solreffriction": list(court.BALL_SOLREFFRICTION),
                 "net_e_assumed": float(court.NET_E_ASSUMED),
                 "racket_e_constant": float(court.RACKET_E_CONSTANT),
                 "aero_wired_in": False},
                DIVERGENT_BLOCKING)


def _probe_ppo_hyperparams(paths: dict, lane: dict) -> dict:
    yaml_cfg = isaac_ppo_params(paths)
    algo, policy, runner = (yaml_cfg["algorithm"], yaml_cfg["policy"],
                            yaml_cfg["runner"])
    cfg = lane["train"].build_agent_cfg(seed=0, iterations=1,
                                        num_steps_per_env=runner["num_steps_per_env"],
                                        experiment="alignment_probe")
    ours = cfg["algorithm"]
    isaac_view = {k: algo[k] for k in ("learning_rate", "schedule", "desired_kl",
                                       "entropy_coef", "num_learning_epochs",
                                       "num_mini_batches", "gamma", "lam",
                                       "max_grad_norm")}
    ours_view = {k: ours.get(k) for k in isaac_view}
    differing = sorted(k for k in isaac_view if isaac_view[k] != ours_view[k])
    net_same = (list(policy["actor_hidden_dims"]) == list(cfg["actor"]["hidden_dims"])
                and policy["activation"] == cfg["actor"]["activation"]
                and float(policy["init_noise_std"])
                == float(cfg["actor"]["distribution_cfg"]["init_std"]))
    isaac_view["actor_hidden_dims"] = list(policy["actor_hidden_dims"])
    isaac_view["init_noise_std"] = policy["init_noise_std"]
    ours_view["actor_hidden_dims"] = list(cfg["actor"]["hidden_dims"])
    ours_view["init_noise_std"] = cfg["actor"]["distribution_cfg"]["init_std"]
    return _row(isaac_view, ours_view,
                ALIGNED if (not differing and net_same) else DIVERGENT_DECLARED,
                differing_keys=differing, network_and_init_std_match=net_same)


def _probe_geometry_provenance(paths: dict, lane: dict) -> dict:
    """Which ``geometry.py`` did this lane actually load, and is it the repo's?

    ``a3_court_env`` resolves geometry in the order env var -> a copy sitting
    next to itself -> the repo.  The pod-deployed lane HAS such a copy, so on
    the pod it is running a byte copy with no digest check at all -- one rung
    below "a fingerprint that proves the bytes did not move".
    """

    loaded = Path(getattr(lane["geom"], "__file__", "")).resolve()
    live = paths["geometry"].resolve()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    loaded_sha = digest(loaded) if loaded.is_file() else None
    live_sha = digest(live)
    same_file = loaded == live
    same_bytes = loaded_sha == live_sha
    return _row({"path": str(live), "sha256": live_sha},
                {"path": str(loaded), "sha256": loaded_sha,
                 "is_the_repo_file": same_file},
                ALIGNED if same_bytes else DIVERGENT_BLOCKING,
                loaded_a_copy_not_the_repo_file=not same_file)


def _probe_determinism_tier(paths: dict, lane: dict) -> dict:
    """Item (e): the cross-engine comparison can only ever be statistical."""

    return _row({"tier1_exact": ["question", "curriculum", "receipt", "ABI",
                                 "action identity"],
                 "tier2": "statistical equivalence only"},
                {"engine": "mujoco-warp", "cpu_fallback": False,
                 "bitwise_reproducible": False,
                 "measured": "pendula (contact-free) diverged 1007/1024 worlds, "
                             "max abs dqpos 1.4e-05 (EXP 9.2.0)"},
                DIVERGENT_DECLARED)


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------

AXES: tuple = (
    Axis("actor_observation_abi",
         "策略看到的那根向量:Isaac 是 211 维 17 行(含 measured teacher 与任务包),"
         "本车道是 114 维 10 行(本体感 + 球的相对位置)。不是同一道题。",
         DIVERGENT_BLOCKING,
         "两边输入不同,策略学到的映射不可互相解释;本车道没有 teacher 行,"
         "所以它连 mimic 这一层都不存在。",
         "需要 measured teacher artifact 与在线 question solver;不是本车道能"
         "单独补的(见 §9.2.9 (d))。",
         _probe_actor_abi),

    Axis("critic_observation_abi",
         "Isaac 的 critic 是 319 维特权观测(含 62 维 command、42/84 维全身位姿);"
         "本车道 critic 和 actor 吃同一份 114 维。",
         DIVERGENT_BLOCKING,
         "非对称 critic 改变价值估计的方差与可学性,同预算下的学习曲线不可比。",
         "需要 Isaac 的 motion command manager 提供 command/body_pos/body_ori 行。",
         _probe_critic_abi),

    Axis("raw_action_affine",
         "策略的31列先按tracked bijection固定为runtime articulation顺序;"
         "两边同样无raw clip,并用同一份逐关节scale与动态ready offset做"
         "q_des_raw=offset+scale*action。",
         ALIGNED,
         "",
         "",
         _probe_raw_action_affine,
         caveat="这只对齐raw proposal;执行前的q_des保护由下一轴单独裁定。"),

    Axis("executable_qdes_guard",
         "raw proposal之后并不同构:Isaac把有限请求投影到soft∩hard"
         "的5%内缩包络,并按鲜q/qdot和20 ms预测做刹车;MuJoCo"
         "只做hard joint-range clamp。",
         DIVERGENT_DECLARED,
         "执行动作和反馈动力学不同,所以禁止policy transfer和matched causal"
         "声明;但这不阻断明确标为diagnostic_unauthorized的MuJoCo-only 25k工程跑。",
         "",
         _probe_executable_qdes_guard),

    Axis("pd_gains",
         "PD 增益 Kp/Kd:本车道的 VENDOR_KP/KD 是手抄的,这一行把它逐关节"
         "跟 Isaac 活的 stiffness/damping 对上。",
         ALIGNED,
         "",
         "",
         _probe_pd_gains),

    Axis("effort_limits",
         "力矩上限:Isaac 的 effort_limit_sim 与智元 MJCF 的 <motor ctrlrange> "
         "逐关节相同。",
         ALIGNED,
         "",
         "",
         _probe_effort_limits),

    Axis("termination_union",
         "什么算这一局结束。Isaac 有 9 条(含撞桌、q_des 禁区、参考包络);"
         "本车道只有 3 条(摔倒高度、倾角、非有限)+ 超时截断。",
         DIVERGENT_BLOCKING,
         "终止 union 直接决定回报的支撑集(CaT:二值终止 => 该分支回报恒零)。"
         "尤其是 robot_hit_table —— 本车道场景里**有桌子**、机器人能撞上去,"
         "却没有任何护栏,等于允许一条 Isaac 判死的行为。",
         "robot_hit_table 这一条本轮已经**测到了**:接触探针同一趟里数机器人-桌面"
         "接触,收据逐集报,--report 见非零就拒(ROBOT_LEANED_ON_THE_TABLE)。"
         "但那是'证据+阻断',不是护栏 —— 装成硬终止会改训练分布,属发车决定。"
         "参考包络三条需要 teacher,补不动。",
         _probe_termination_union),

    Axis("fall_thresholds",
         "两边都有的那两条(摔倒角、骨盆过低)的阈值到底一不一样。",
         DIVERGENT_BLOCKING,
         "Isaac 40 度 / 0.5 m,本车道 60 度 / 0.70 m:本车道对倾角更宽容、"
         "对下蹲更严格,早期终止率不可比。",
         "把 TaskCfg 的两个数改成从这一行读到的 Isaac 活值即可;"
         "会改变既有收据的可比性,属发车决定。",
         _probe_fall_thresholds),

    Axis("reward_surface",
         "奖励长什么样。Isaac 是完整 ActionBall 层级(balance/mimic/strike/"
         "target/outcome);本车道 10 项,mimic/strike/target/outcome 四组覆盖为 0。",
         DIVERGENT_BLOCKING,
         "本车道的 reach/touch 是**拍球距离的整形**,不是击球质量,更不是上台;"
         "把它的曲线读成'学会回球'是错的。",
         "需要 measured teacher(mimic)、question packet(target)与 analytic "
         "outcome evaluator(outcome);属 §9.2 的 reward 层级工作。",
         _probe_reward_surface,
         caveat="Isaac 侧读的是类体里声明的权重;发车时 reward-pack YAML 仍可覆盖。"),

    Axis("episode_structure",
         "一局的形状。Isaac:500 tick(10 s),开局 5--25 tick 的 WAIT 把任务行"
         "遮住,揭示后至少 200 tick 有效。本车道:150 tick(3 s),没有 WAIT/揭示。",
         DIVERGENT_BLOCKING,
         "WAIT/揭示是四格实验的第二根轴(§5.6.2d);没有它就测不到 reveal bridge"
         "的可学性,而且 episode 长度差 3.3 倍,终止率与集长曲线不可比。",
         "WAIT 结构本身可以照搬(它只是一个每 env 的计数器 + 观测掩码),"
         "但掩掉的那些行本车道根本没有,所以照搬会得到一个空的 WAIT。",
         _probe_episode_structure),

    Axis("control_rate",
         "策略频率两边都是 50 Hz(policy dt = 0.02 s),这一条真的对齐了。",
         ALIGNED,
         "",
         "",
         _probe_control_rate,
         caveat="物理步长不同(Isaac 5 ms / 本车道 1 ms,后者是智元显式值)。"
                "行里的 physics_rate_differs 记着这件事;它改的是积分误差,"
                "不改策略看到的时钟,所以不按 blocking 记。"),

    Axis("observation_noise_and_dr",
         "随机性开关。Isaac 四格 = A/C 族 x {本体感噪声 关/开},plant 随机化"
         "全关(DR-L0/L0N,是有意的归因设计)。本车道观测无噪声,但复位时"
         "给关节/根位姿加了均匀噪声。",
         DIVERGENT_BLOCKING,
         "本车道相当于 DR-L0 的观测轴 + 一份 Isaac 四格里**没有**的复位随机化;"
         "所以它既不是 A0/C0 也不是 A1/C1,不能当任一格的对照。",
         "观测噪声三通道可以照 DR-L0N 的活值加(通道与边界这一行已经读出来了);"
         "复位噪声要么关掉、要么在四格里单独立轴。",
         _probe_obs_noise_dr),

    Axis("question_distribution",
         "问的题不一样。Isaac A211/C211 现在是**一道固定题**(所有 32 个域等级"
         "恰好为零、profile 中心点);本车道每次发球从一个均匀盒子里重新采。",
         DIVERGENT_BLOCKING,
         "固定题 vs 分布题是两种可学性:前者回答'这道题学不学得会',"
         "后者回答'这一类题学不学得会'。混着读会得出相反结论。",
         "本车道可以把 ServeConfig 收成一个点(退化盒子)来对上'固定题',"
         "但 Isaac 的题是 desired-contact / incoming-ball 包,不是发球盒子。",
         _probe_question_distribution),

    Axis("vendor_plant_inheritance",
         "Franco 的规矩:MuJoCo 设置继承智元的 MuJoCo,不是 mjlab 默认。"
         "这一行把智元 MJCF 的 <option> 跟本车道显式写的 OPTION 对上。",
         ALIGNED,
         "",
         "",
         _probe_vendor_plant_inheritance,
         caveat="noslip_iterations=3 带不过去,是已登记的具名偏离(mujoco-warp 无 noslip pass)。"),

    Axis("ball_contact_model",
         "球怎么弹。本车道是真接触(标定过的 solref/solimp/摩擦);"
         "现役 Isaac C211 走解析路径(physical_ball=false)。",
         DIVERGENT_BLOCKING,
         "一边是引擎解出来的接触,一边是解析式给的出射;命中率/上台率不可比,"
         "而且本车道的球拍恢复系数与网都还没有标定(具名缺口)。",
         "需要两边同时上原生接触,或两边同时走解析;当前谁都不是。",
         _probe_ball_contact_model),

    Axis("ppo_hyperparameters",
         "PPO 超参。网络、init_std、lr、KL、epochs、minibatch、gamma、lam 一致;"
         "熵系数不同(Isaac 0.01,本车道量过之后取 0.002)。",
         DIVERGENT_DECLARED,
         "熵系数差异有实测理由(31 维动作下 rsl-rl 的逐维熵奖励把 std 从 1.00 "
         "推到 1.16),写在 build_agent_cfg 的注释里;它改探索强度但不改题目。",
         "",
         _probe_ppo_hyperparams),

    Axis("geometry_provenance",
         "球台/球/发球这些几何常量,本车道到底读的是仓库那一份,还是它自己旁边"
         "的一份拷贝。",
         ALIGNED,
         "",
         "",
         _probe_geometry_provenance,
         caveat="pod 上的 /workspace/mjlab_lane 里有一份同名拷贝,解析顺序优先于仓库;"
                "这一行就是用来在那种部署下当场变红的。"),

    Axis("determinism_tier",
         "跨引擎只能做统计对拍,不能做逐位对拍。",
         DIVERGENT_DECLARED,
         "mujoco-warp 无 CPU 回退且实测非确定性(无接触的 pendula 也发散),"
         "所以任何'逐位一致'的验收都是错的。",
         "",
         _probe_determinism_tier),
)


#: TaskCfg / SimCfg fields that no axis reads, and why that is acceptable.
#: No wildcard: a new knob must be classified by hand or the guard fires.
UNCLASSIFIED_LANE_FIELDS_ALLOWED = {
    "obs_scale_lin_vel": "this lane's own observation scaling; there is no 114-D twin to compare",
    "obs_scale_ang_vel": "same",
    "obs_scale_joint_vel": "same",
    "obs_scale_ball_vel": "same",
    "obs_clip": "same",
    "w_alive": "priced by the reward_surface axis as a group, not per weight",
    "w_pose": "same", "w_upright": "same", "w_height": "same",
    "w_reach": "same", "w_touch": "same", "w_action_rate": "same",
    "w_joint_vel": "same", "w_torque": "same", "r_termination": "same",
    "k_pose": "reward kernel shape; no Isaac twin term exists to compare against",
    "k_upright": "same", "k_height": "same",
    "reach_len_m": "same", "touch_sigma_m": "same",
    "reset_joint_vel_noise": "read by the observation_noise_and_dr axis as part of reset noise",
    "ball_dead_z_hope": "ball housekeeping; Isaac's rally death is an outcome term",
    "ball_dead_x_lo_hope": "same", "ball_dead_x_hi_hope": "same",
    "nworld": "scale knob, not a task semantic",
    "cone": "solver knob; priced by the capacity census, not by task alignment",
    "add_pairs": "same", "njmax": "same", "nconmax": "same",
    "ball_spawn_hope": "model-build spawn; the live serve is read by question_distribution",
}

#: Lane fields each axis DOES read.  Kept explicit so the enumeration guard is
#: a real guard and not a restatement of whatever the code happens to touch.
LANE_FIELDS_READ_BY_AXES = {
    "action_scale": "raw_action_affine",
    "action_scale_mode": "raw_action_affine",
    "action_clip": "raw_action_affine",
    "episode_length_s": "episode_structure",
    "ball_reserve_after_s": "episode_structure",
    "min_pelvis_z": "fall_thresholds",
    "max_tilt_proj_g": "fall_thresholds",
    "reset_joint_noise_rad": "observation_noise_and_dr",
    "reset_root_xy_noise_m": "observation_noise_and_dr",
    "reset_root_yaw_noise_rad": "observation_noise_and_dr",
}


def unclassified_lane_fields(lane: dict | None = None) -> list:
    """TaskCfg/SimCfg knobs that neither an axis nor the allow-list names."""

    lane = lane or mjlab_side()
    fields = set(vars(lane["task"])) | set(vars(lane["sim"]))
    known = set(LANE_FIELDS_READ_BY_AXES) | set(UNCLASSIFIED_LANE_FIELDS_ALLOWED)
    return sorted(fields - known)


def stale_classifications(lane: dict | None = None) -> list:
    """Names the tables claim exist but the live dataclasses no longer have."""

    lane = lane or mjlab_side()
    fields = set(vars(lane["task"])) | set(vars(lane["sim"]))
    named = set(LANE_FIELDS_READ_BY_AXES) | set(UNCLASSIFIED_LANE_FIELDS_ALLOWED)
    return sorted(named - fields)


def build_ledger(root: Path | None = None, lane: dict | None = None) -> dict:
    """Resolve every axis against live values on both sides.

    Raises :class:`AlignmentError` when a declared verdict no longer matches
    what the two lanes actually do -- in EITHER direction.  A row that says
    ALIGNED and is not is the obvious failure; a row that says
    DIVERGENT_BLOCKING and has quietly become aligned is the one that rots,
    because it keeps a closed gap looking open and nobody re-reads it.
    """

    sources = isaac_sources(root)
    lane = lane or mjlab_side()
    unclassified = unclassified_lane_fields(lane)
    stale = stale_classifications(lane)
    if unclassified:
        raise AlignmentError(
            "these mjlab lane knobs are classified by no alignment axis and by "
            f"no allow-list entry: {unclassified}.  Classify them, or the table "
            "silently stops describing the lane it claims to describe.")
    if stale:
        raise AlignmentError(
            f"the alignment tables name lane knobs that no longer exist: {stale}")

    rows: dict = {}
    if sources["missing"] or sources["root"] is None:
        for axis in AXES:
            rows[axis.key] = {
                "human": axis.human, "declared": axis.declared,
                "observed": UNVERIFIABLE, "why": axis.why,
                "closable_by": axis.closable_by, "caveat": axis.caveat,
                "isaac": None, "mjlab": None,
                "error": f"Isaac sources unreachable: missing={sources['missing']}",
            }
        summary_root = None
    else:
        for axis in AXES:
            try:
                row = axis.probe(sources["paths"], lane)
            except AlignmentError:
                raise
            except Exception as exc:  # a probe that cannot read must not pass
                row = _row(None, None, UNVERIFIABLE, error=repr(exc))
            row.update({"human": axis.human, "declared": axis.declared,
                        "why": axis.why, "closable_by": axis.closable_by,
                        "caveat": axis.caveat})
            rows[axis.key] = row
        summary_root = str(sources["root"])

    drifted = sorted(k for k, r in rows.items()
                     if r["observed"] != r["declared"]
                     and not (r["observed"] == UNVERIFIABLE and summary_root is None))
    if drifted:
        detail = {k: {"declared": rows[k]["declared"], "observed": rows[k]["observed"],
                      "error": rows[k].get("error")} for k in drifted}
        raise AlignmentError(
            "the alignment table no longer describes the two lanes; declared vs "
            f"live verdict differs on {drifted}: "
            + json.dumps(detail, ensure_ascii=False, sort_keys=True))

    counts = {verdict: sum(1 for r in rows.values() if r["observed"] == verdict)
              for verdict in VERDICTS}
    blocking = sorted(k for k, r in rows.items()
                      if r["observed"] in BLOCKING_VERDICTS)
    executable_guard_restricted = (
        rows.get("executable_qdes_guard", {}).get("observed")
        == DIVERGENT_DECLARED
    )
    ledger = {
        "kind": "mjlab_lane_isaac_alignment_ledger_v1",
        "isaac_repo_root": summary_root,
        "isaac_sources_missing": sources["missing"],
        "n_axes": len(AXES),
        "verdict_counts": counts,
        "blocking_axes": blocking,
        "cross_engine_comparable": not blocking,
        "policy_transfer_authorized": (
            not blocking and not executable_guard_restricted
        ),
        "matched_causal_comparison_authorized": (
            not blocking and not executable_guard_restricted
        ),
        "mujoco_only_diagnostic_25k_blocked_by_alignment": False,
        "bitwise_parity_is_never_a_valid_acceptance": True,
        "rows": rows,
    }
    ledger["ledger_sha256"] = hashlib.sha256(
        json.dumps(ledger, sort_keys=True, ensure_ascii=False,
                   default=str).encode("utf-8")).hexdigest()
    return ledger


# ---------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------

#: Claims a receipt may ask this module to bless.
CLAIM_CROSS_ENGINE_COMPARABLE = "cross_engine_comparable"
CLAIM_BITWISE_PARITY = "bitwise_parity"
CLAIM_POLICY_TRANSFER = "policy_transfer"
CLAIM_MATCHED_CAUSAL_COMPARISON = "matched_causal_comparison"
CLAIMS = (
    CLAIM_CROSS_ENGINE_COMPARABLE,
    CLAIM_BITWISE_PARITY,
    CLAIM_POLICY_TRANSFER,
    CLAIM_MATCHED_CAUSAL_COMPARISON,
)


def assert_cross_engine_claim(ledger: dict, claim: str) -> None:
    """Refuse a claim this lane has not earned.  Fail-closed, no soft mode.

    ``bitwise_parity`` is refused unconditionally and forever: mujoco-warp has
    no CPU fallback and is measurably non-deterministic even without contact,
    so a bit-exact acceptance is not a strict standard, it is a wrong one.
    """

    if claim not in CLAIMS:
        raise AlignmentClaimRefused(f"unknown cross-engine claim {claim!r}")
    if claim == CLAIM_BITWISE_PARITY:
        raise AlignmentClaimRefused(
            "bitwise cross-engine parity is not a valid acceptance criterion: "
            "mujoco-warp has no CPU fallback and diverges run to run even on a "
            "contact-free model (EXP 9.2.0).  Use an N-seed statistical band.")
    guard = (ledger.get("rows") or {}).get("executable_qdes_guard") or {}
    if (
        claim in (CLAIM_POLICY_TRANSFER, CLAIM_MATCHED_CAUSAL_COMPARISON)
        and guard.get("observed") == DIVERGENT_DECLARED
    ):
        raise AlignmentClaimRefused(
            "executable_qdes_guard differs: Isaac projects into its inset "
            "soft/hard envelope and brakes from q/qdot, while MuJoCo hard-clamps "
            f"only; claim {claim!r} is outside the declared MuJoCo-only scope"
        )
    blocking = ledger.get("blocking_axes") or []
    if blocking:
        raise AlignmentClaimRefused(
            "this lane is not asking the same question as the Isaac A211/C211 "
            f"run; blocking alignment axes: {blocking}")


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="write the ledger as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ledger = build_ledger()
    if args.out:
        Path(args.out).write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True,
                       default=str), encoding="utf-8")
    if not args.quiet:
        print("=" * 78)
        print("mjlab GPU lane  vs  Isaac A211/C211 -- live alignment ledger")
        print("=" * 78)
        print(f"  isaac repo root : {ledger['isaac_repo_root']}")
        print(f"  axes            : {ledger['n_axes']}")
        for verdict in VERDICTS:
            print(f"  {verdict:22s}: {ledger['verdict_counts'][verdict]}")
        print(f"  cross-engine comparable : {ledger['cross_engine_comparable']}")
        print("-" * 78)
        for key, row in ledger["rows"].items():
            print(f"[{row['observed']:19s}] {key}")
            print(f"    {row['human']}")
            if row["observed"] in BLOCKING_VERDICTS and row["why"]:
                print(f"    为什么要紧: {row['why']}")
            if row["closable_by"]:
                print(f"    怎么补    : {row['closable_by']}")
            if row["caveat"]:
                print(f"    注意      : {row['caveat']}")
        print("=" * 78)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
