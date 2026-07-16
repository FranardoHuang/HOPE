"""Dependency-light guards for planner launch-file composition."""

import ast
from pathlib import Path


def _launch_tree() -> ast.Module:
    source = (
        Path(__file__).resolve().parents[1]
        / "launch"
        / "hope_planner.launch.py"
    ).read_text(encoding="utf-8")
    return ast.parse(source)


def _named_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def test_task_revision_launch_is_explicit_opt_in_with_overlay_loaded_last():
    tree = _launch_tree()

    declarations = _named_calls(tree, "DeclareLaunchArgument")
    task_revision = next(
        call
        for call in declarations
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "task_revision"
    )
    assert ast.literal_eval(_keyword(task_revision, "default_value")) == "false"

    nodes = _named_calls(tree, "Node")
    assert len(nodes) == 2
    legacy = next(
        call
        for call in nodes
        if isinstance(_keyword(call, "condition"), ast.Call)
        and ast.unparse(_keyword(call, "condition").func) == "UnlessCondition"
    )
    revision = next(
        call
        for call in nodes
        if isinstance(_keyword(call, "condition"), ast.Call)
        and ast.unparse(_keyword(call, "condition").func) == "IfCondition"
    )
    assert [ast.unparse(value) for value in _keyword(legacy, "parameters").elts] == [
        "str(config)"
    ]
    assert [ast.unparse(value) for value in _keyword(revision, "parameters").elts] == [
        "str(config)",
        "str(task_revision_config)",
    ]
