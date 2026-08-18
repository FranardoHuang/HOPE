from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "a3_train_ppo.py"


def _env_call(function_name: str) -> ast.Call:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "A3ReadyBallVecEnv"
    ]
    assert len(calls) == 1
    return calls[0]


def test_train_and_eval_forward_explicit_scene_inputs() -> None:
    expected = {
        "xml_path": "Path(args.xml_path) if args.xml_path else None",
        "ready_pose_path": "Path(args.ready_pose) if args.ready_pose else None",
    }
    for function_name in ("train", "evaluate"):
        keywords = {item.arg: ast.unparse(item.value) for item in _env_call(function_name).keywords}
        assert {name: keywords[name] for name in expected} == expected


def test_cli_declares_both_explicit_scene_inputs() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    options = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert {"--xml-path", "--ready-pose"} <= options
