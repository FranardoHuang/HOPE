"""Regression checks for the one-path timing-paper CLI."""

from __future__ import annotations

import ast
from pathlib import Path


EVALUATOR = Path(__file__).resolve().parents[1] / "scripts" / "isaac_bank_exam.py"


def _source() -> str:
    return EVALUATOR.read_text(encoding="utf-8")


def test_timing_cli_requires_only_the_paper_path():
    source = _source()
    assert 'timing_paper_raw = str(_cfg(cfg, "timing_paper", "")).strip()' in source
    assert "expected_timing_paper_sha256" not in source
    assert "expected_timing_paper_semantic_sha256" not in source
    assert "_load_timing_paper_from_path(" in source
    assert "timing_paper_path\n        )" in source


def test_simple_timing_loader_keeps_internal_strict_validation():
    tree = ast.parse(_source())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_timing_paper_from_path"
    )
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"timing_sha256_file", "load_timing_paper"} <= calls

    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_timing_paper"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["expected_file_sha256"], ast.Name)
    assert keywords["expected_file_sha256"].id == "file_sha"
    semantic = keywords["expected_semantic_sha256"]
    assert isinstance(semantic, ast.Call)
    assert isinstance(semantic.func, ast.Attribute)
    assert semantic.func.attr == "get"
    assert isinstance(semantic.args[0], ast.Constant)
    assert semantic.args[0].value == "paper_semantic_sha256"
