"""Read the LIVE value of an Isaac constant, so a hand copy can be checked.

人话:这个文件回答一个很窄的问题 —— "Isaac 那边这个数**现在**到底是多少?"

``mujoco_native`` 里有一批常量的真源在 Isaac 侧(``hope_env_cfg.py`` /
``terminations.py``),这里存的只是副本。这些副本过去只被"语义 AST 指纹"罩着,
而指纹只证明**源文件那几个节点的字节没动过**;源文件一动,把指纹重钉成新值是
一行的事,副本有没有跟着动没有任何机制在看。5ed998f1(08-04)就是这么让桌面
终局的复刻停在原地两天:同一个提交把指纹扩到覆盖新函数并重新盖章,语义没移植。

这个模块提供的是另一条路:**直接把 Isaac 源码里那个值读出来**,然后跟副本逐个
比。它不 import isaaclab —— ``hope_env_cfg.py`` 拉的是整棵 Isaac Lab,host 上
根本装不了 —— 而是解析 AST,把目标节点当字面量求值。求值器是白名单式的:
常量、元组/列表、模块级名字、``list()``/``tuple()``、序列相加,别的一律拒绝。
所以它读不出"运行时才知道的值",遇到就 fail closed 报 blocker,不会猜。

和 ``action_ball_211_abi.live_source_parity_blockers`` 是同一个范式;那边的叶子
是 dependency-free 的,可以直接 host-load 比活值,这边不行,退一步用 AST 取值 ——
但取的是**值**,不是哈希:把 ``0.02`` 改成 ``0.03`` 再把指纹重钉一遍,这里照样红。
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


class IsaacLiveConstantError(RuntimeError):
    """The live Isaac value could not be read as an unambiguous literal."""


#: Selector kinds this module can answer.  Anything else fails closed.
SELECTOR_KINDS = (
    # ("assignment", "TABLE_HIT_MARGIN_M")
    #   -> the module-level assignment's value.
    "assignment",
    # ("class_term_param", "HOPEDeployParityTerminationsCfg",
    #  "base_fell_tilt", "limit_angle")
    #   -> ``ClassDef.body`` assignment ``base_fell_tilt = DoneTerm(...,
    #      params={"limit_angle": 0.7})`` -> ``0.7``.
    "class_term_param",
    # ("function_return_param", "table_hit_done_term", "margin")
    #   -> the ``params={...}`` dict of the ``return DoneTerm(...)`` inside
    #      that function -> the value stored under ``"margin"``.
    "function_return_param",
)


@lru_cache(maxsize=16)
def _parse_module(path_text: str) -> ast.Module:
    path = Path(path_text)
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=path_text)
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise IsaacLiveConstantError(f"cannot parse live source {path_text}") from exc


def _assign_targets(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Assign):
        targets: Sequence[ast.AST] = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = (node.target,)
    else:
        return ()
    return tuple(item.id for item in targets if isinstance(item, ast.Name))


@lru_cache(maxsize=16)
def _module_level_assignments(path_text: str) -> dict:
    """``name -> value node`` for module-level assignments only.

    Module level on purpose: a same-named class attribute or local must never
    be able to answer a question about a module constant.
    """

    tree = _parse_module(path_text)
    found: dict = {}
    duplicated: set = set()
    for statement in tree.body:
        value = getattr(statement, "value", None)
        if value is None:
            continue
        for name in _assign_targets(statement):
            if name in found:
                duplicated.add(name)
            found[name] = value
    for name in duplicated:
        # A rebound module constant has no single live value; refuse it rather
        # than silently answering with the last binding.
        found.pop(name, None)
    return found


@lru_cache(maxsize=16)
def _import_from_bindings(path_text: str) -> dict:
    """``imported name -> dotted module`` for module-level ``from X import Y``."""

    tree = _parse_module(path_text)
    bindings: dict = {}
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom) or statement.level:
            continue
        module = statement.module or ""
        for alias in statement.names:
            if alias.name == "*":
                continue
            bindings[alias.asname or alias.name] = module
    return bindings


def _normalize(value: Any) -> Any:
    """Sequences compare as tuples; everything else keeps its exact type."""

    if isinstance(value, (tuple, list)):
        return tuple(_normalize(item) for item in value)
    return value


def _same_value(left: Any, right: Any) -> bool:
    """Exact equality, with ``bool``/``int``/``float`` kept distinguishable."""

    left = _normalize(left)
    right = _normalize(right)
    if isinstance(left, tuple) != isinstance(right, tuple):
        return False
    if isinstance(left, tuple):
        if len(left) != len(right):
            return False
        return all(_same_value(a, b) for a, b in zip(left, right))
    if type(left) is not type(right):
        return False
    return left == right


class _LiteralResolver:
    """Evaluate one Isaac source node as a literal, or refuse to answer.

    The whitelist is deliberately small.  Everything it cannot fold — a call it
    does not know, an attribute, an f-string, a comprehension — raises, and the
    caller turns that into a blocker.  A guard that guesses is worse than one
    that stops.
    """

    def __init__(
        self, path_text: str, companions: Mapping[str, str] | None = None
    ) -> None:
        self.path_text = path_text
        self.companions = dict(companions or {})

    def resolve_name(self, name: str, chain: tuple[str, ...]) -> Any:
        if name in chain:
            raise IsaacLiveConstantError(f"cyclic live constant reference {name!r}")
        local = _module_level_assignments(self.path_text)
        if name in local:
            return self.evaluate(local[name], chain + (name,))
        module = _import_from_bindings(self.path_text).get(name)
        if module is None:
            raise IsaacLiveConstantError(
                f"live name {name!r} is neither assigned nor imported in "
                f"{self.path_text}"
            )
        companion_path = self.companions.get(module)
        if companion_path is None:
            raise IsaacLiveConstantError(
                f"live name {name!r} comes from unregistered module {module!r}"
            )
        companion = _LiteralResolver(companion_path, self.companions)
        assignments = _module_level_assignments(companion_path)
        if name not in assignments:
            raise IsaacLiveConstantError(
                f"live name {name!r} is not a module constant of {module!r}"
            )
        return companion.evaluate(assignments[name], chain + (name,))

    def evaluate(self, node: ast.AST, chain: tuple[str, ...] = ()) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.evaluate(item, chain) for item in node.elts)
        if isinstance(node, ast.Name):
            return self.resolve_name(node.id, chain)
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            operand = self.evaluate(node.operand, chain)
            if isinstance(operand, bool) or not isinstance(operand, (int, float)):
                raise IsaacLiveConstantError("unary operand is not a plain number")
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.evaluate(node.left, chain)
            right = self.evaluate(node.right, chain)
            if isinstance(left, tuple) and isinstance(right, tuple):
                return left + right
            raise IsaacLiveConstantError("only sequence concatenation is folded")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"list", "tuple"}
            and len(node.args) == 1
            and not node.keywords
        ):
            value = self.evaluate(node.args[0], chain)
            if not isinstance(value, tuple):
                raise IsaacLiveConstantError(
                    f"{node.func.id}() argument is not a literal sequence"
                )
            return value
        raise IsaacLiveConstantError(
            f"live value is not a foldable literal ({type(node).__name__})"
        )


def _unique(matches: Sequence[Any], description: str) -> Any:
    if len(matches) != 1:
        raise IsaacLiveConstantError(
            f"{description} is not unique in the live source ({len(matches)} found)"
        )
    return matches[0]


def _params_dict(call: ast.AST, description: str) -> ast.Dict:
    if not isinstance(call, ast.Call):
        raise IsaacLiveConstantError(f"{description} is not a term constructor call")
    params = [
        keyword.value for keyword in call.keywords if keyword.arg == "params"
    ]
    node = _unique(params, f"{description} params=")
    if not isinstance(node, ast.Dict):
        raise IsaacLiveConstantError(f"{description} params= is not a literal dict")
    return node


def _dict_entry(node: ast.Dict, key: str, description: str) -> ast.AST:
    matches = [
        value
        for stored_key, value in zip(node.keys, node.values)
        if isinstance(stored_key, ast.Constant) and stored_key.value == key
    ]
    return _unique(matches, f"{description} key {key!r}")


def _selected_node(path_text: str, selector: Sequence[Any]) -> ast.AST:
    kind = selector[0] if selector else None
    if kind not in SELECTOR_KINDS:
        raise IsaacLiveConstantError(f"unsupported live selector kind {kind!r}")
    tree = _parse_module(path_text)
    if kind == "assignment":
        _kind, name = selector
        assignments = _module_level_assignments(path_text)
        if name not in assignments:
            raise IsaacLiveConstantError(
                f"live module constant {name!r} is absent or rebound"
            )
        return assignments[name]
    if kind == "class_term_param":
        _kind, class_name, attribute, param = selector
        classes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        class_node = _unique(classes, f"class {class_name!r}")
        attributes = [
            node.value
            for node in class_node.body
            if attribute in _assign_targets(node) and getattr(node, "value", None)
        ]
        call = _unique(attributes, f"{class_name}.{attribute}")
        return _dict_entry(
            _params_dict(call, f"{class_name}.{attribute}"),
            param,
            f"{class_name}.{attribute}",
        )
    _kind, function_name, param = selector
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    function_node = _unique(functions, f"function {function_name!r}")
    returns = [
        node.value
        for node in ast.walk(function_node)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    call = _unique(returns, f"{function_name}() return")
    return _dict_entry(
        _params_dict(call, f"{function_name}()"), param, f"{function_name}()"
    )


def live_value(
    source_path: Any,
    selector: Sequence[Any],
    *,
    companions: Mapping[str, Any] | None = None,
) -> Any:
    """Return the value the live Isaac source actually ships for ``selector``."""

    path_text = str(Path(source_path))
    resolver = _LiteralResolver(
        path_text,
        {module: str(Path(path)) for module, path in (companions or {}).items()},
    )
    return _normalize(resolver.evaluate(_selected_node(path_text, selector)))


def parity_blockers(
    prefix: str,
    entries: Sequence[Sequence[Any]],
    *,
    companions: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """List every mirrored constant that no longer equals its live Isaac value.

    ``entries`` are ``(key, source_path, selector, mirrored_value)``.  Empty
    means every listed copy still equals what the live source ships right now —
    which is a strictly stronger statement than "the source bytes still hash to
    the pin", because re-pinning the hash cannot satisfy it.
    """

    blockers: list[str] = []
    for entry in entries:
        key, source_path, selector, mirrored = entry
        try:
            live = live_value(source_path, selector, companions=companions)
        except IsaacLiveConstantError as exc:
            blockers.append(f"{prefix}_live_value_unreadable:{key}:{exc}")
            continue
        if not _same_value(live, mirrored):
            blockers.append(
                f"{prefix}_mirrored_constant_differs:{key}:live={live!r} "
                f"mirror={_normalize(mirrored)!r}"
            )
    return tuple(blockers)


def clear_caches() -> None:
    """Drop the parsed-source caches (tests repoint the sources at tmp files)."""

    _parse_module.cache_clear()
    _module_level_assignments.cache_clear()
    _import_from_bindings.cache_clear()


__all__ = [
    "IsaacLiveConstantError",
    "SELECTOR_KINDS",
    "clear_caches",
    "live_value",
    "parity_blockers",
]
