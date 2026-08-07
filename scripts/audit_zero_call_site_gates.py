#!/usr/bin/env python3
"""扫出"写了一道门,但全仓没人调用它"的函数。

人话:这个仓里出现过好几次同一种病 —— 有人写了一个 ``_validate_x`` /
``assert_x``,把它当成一道护栏写进文档,但它一个调用点都没有。它不拦任何东西,
读代码的人却以为它在守着。

这个脚本把找它们的方法**固定下来**,因为前两轮是靠临时命令行做的,而那个临时
做法有两个具体的漏洞,2026-08-07 各漏掉了一批:

1. **名字形状清单每一条都带前导下划线**,于是所有公开方法(``assert_x`` 而不是
   ``_assert_x``)整类看不见。``ActionBirthBroker.assert_known_generation``
   就是这么漏掉的。
2. **``def`` 只收模块级**,于是类里的方法整类看不见。同一个漏。

另外还有一条:必须排除 ``.claude/worktrees/``。那底下是别的 agent 会话留下的
几千个 ``.py``,不排掉会把**别人树里**的调用点当成本仓的,于是真的死门被算成
"有人用"。

判据用 token 频次而不是 AST 调用图:一个纯 AST 调用图会漏掉
``getattr(obj, "name")``、``monkeypatch.setattr(M, "name", ...)``、
``importlib`` 加载的共享库这些形式。token 频次会把它们都算进去,所以它给出的是
**"没人提过它"的上界**;命中数 <= 1 才报,也就是"全仓只有 def 那一处提到它"。

用法:
    python3 scripts/audit_zero_call_site_gates.py            # 扫全仓
    python3 scripts/audit_zero_call_site_gates.py --self-test  # 只自检方法学
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

# 门形状。**公开与带下划线的都要有** —— 这正是前一轮漏掉第六个的原因。
_STEMS = (
    "validate",
    "require",
    "valid",
    "check",
    "assert",
    "verify",
    "reverify",
    "ensure",
    "reject",
    "forbid",
    "guard",
    "refuse",
    "enforce",
    "must",
)
GATE_NAME_RE = re.compile(r"^_?(?:%s)_" % "|".join(_STEMS))

# 不属于本仓的树。别的 agent 会话的 worktree 会把死门伪装成活门。
EXCLUDED_DIR_PARTS = (".claude", ".git", "node_modules", "__pycache__")

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Finding(NamedTuple):
    name: str
    path: Path
    lineno: int
    code_hits: int
    doc_hits: int


def is_gate_name(name: str) -> bool:
    return bool(GATE_NAME_RE.match(name))


def iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIR_PARTS for part in path.parts):
            continue
        yield path


def iter_doc_files(root: Path) -> Iterator[Path]:
    for pattern in ("*.md", "*.rst", "*.txt"):
        for path in root.rglob(pattern):
            if any(part in EXCLUDED_DIR_PARTS for part in path.parts):
                continue
            yield path


def collect_gate_defs(path: Path) -> list[tuple[str, int]]:
    """收集这个文件里所有门形状的 def —— **含嵌套**(类里的方法也算)。"""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):  # ast.walk 是含嵌套的关键
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_gate_name(node.name):
                found.append((node.name, node.lineno))
    return found


def count_tokens(paths: Iterable[Path], names: set[str]) -> dict[str, int]:
    counts = {name: 0 for name in names}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in TOKEN_RE.findall(text):
            if token in counts:
                counts[token] += 1
    return counts


def scan(root: Path) -> tuple[list[Finding], dict[str, int]]:
    py_files = sorted(iter_python_files(root))
    defs: dict[str, list[tuple[Path, int]]] = {}
    for path in py_files:
        for name, lineno in collect_gate_defs(path):
            defs.setdefault(name, []).append((path, lineno))
    names = set(defs)
    code_hits = count_tokens(py_files, names)
    doc_hits = count_tokens(sorted(iter_doc_files(root)), names)
    findings = [
        Finding(name, sites[0][0], sites[0][1], code_hits[name], doc_hits[name])
        for name, sites in sorted(defs.items())
        if len(sites) == 1 and code_hits[name] <= 1
    ]
    stats = {
        "py_files": len(py_files),
        "gate_defs": sum(len(v) for v in defs.values()),
        "distinct_gate_names": len(defs),
        "zero_call_site": len(findings),
    }
    return findings, stats


# ---------------------------------------------------------------------------
# 自检:证明这个脚本比它取代的那个做法**多看见一档**
# ---------------------------------------------------------------------------

_FIXTURE = '''
class Thing:
    def assert_public_nested_gate(self):
        """A public gate declared as a class method, never called anywhere."""
        raise RuntimeError("nope")

    def _assert_private_nested_gate(self):
        raise RuntimeError("nope")


def _validate_module_level_gate():
    raise RuntimeError("nope")


def not_a_gate_at_all():
    return 1
'''

# 被取代的那个做法:只认带下划线的形状,只收模块级 def。
_OLD_GATE_NAME_RE = re.compile(r"^_(?:%s)_" % "|".join(_STEMS))


def _old_collect_gate_defs(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.name, node.lineno)
        for node in tree.body  # 只收模块级,不 walk
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _OLD_GATE_NAME_RE.match(node.name)
    ]


def self_test() -> int:
    """老做法必须漏掉公开的/嵌套的门,新做法必须都扫到。"""

    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "fixture_gate_shapes.py"
        fixture.write_text(_FIXTURE, encoding="utf-8")
        new_names = {name for name, _ in collect_gate_defs(fixture)}
        old_names = {name for name, _ in _old_collect_gate_defs(fixture)}

    expected_new = {
        "assert_public_nested_gate",
        "_assert_private_nested_gate",
        "_validate_module_level_gate",
    }
    problems = []
    if new_names != expected_new:
        problems.append(
            f"新做法扫到的不是预期集合: {sorted(new_names)} != {sorted(expected_new)}"
        )
    # 老做法必须**漏掉**这两类 —— 否则这条自检没有判别力,
    # 也就说明"前一轮为什么会漏"这件事没被这个脚本钉住。
    if "assert_public_nested_gate" in old_names:
        problems.append("老做法居然扫到了公开方法,自检失去判别力")
    if "_assert_private_nested_gate" in old_names:
        problems.append("老做法居然扫到了嵌套方法,自检失去判别力")
    if old_names != {"_validate_module_level_gate"}:
        problems.append(f"老做法扫到的不是预期集合: {sorted(old_names)}")

    for line in problems:
        print(f"SELF-TEST FAIL: {line}")
    if problems:
        return 1
    print(
        "SELF-TEST OK: 老做法只看见 1 个(模块级带下划线),"
        "新做法看见 3 个(含公开方法与类内嵌套)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录(默认:本脚本的上一级)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只跑方法学自检,不扫全仓",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    findings, stats = scan(args.root)
    print(
        "扫描规模: {py_files} 个 .py、{gate_defs} 个门形状 def"
        "({distinct_gate_names} 个不同名字)".format(**stats)
    )
    print(f"零调用点(全仓 .py token 命中 <= 1): {stats['zero_call_site']}")
    print()
    for finding in findings:
        rel = finding.path.relative_to(args.root)
        print(
            f"  {finding.name}\n"
            f"    {rel}:{finding.lineno}  代码命中={finding.code_hits} "
            f"文档命中={finding.doc_hits}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
