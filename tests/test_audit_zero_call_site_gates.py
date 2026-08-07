"""把"零调用点扫描"的方法学钉住,而不是把扫描结果钉住。

人话:钉结果需要维护一张允许名单,每加一道门就要改一次,这是仪式。
真正会重犯的是**方法学**:前两轮漏掉第六个死门,是因为名字形状清单每条都带
前导下划线、并且 ``def`` 只收模块级。所以这里只测那两条:

- 公开方法(``assert_x``,不带下划线)必须被认成门形状;
- 类里的嵌套 ``def`` 必须被收进来。

全仓扫描本身是手动跑的(``python3 scripts/audit_zero_call_site_gates.py``),
不进套件 —— 它要读一千多个文件,而且它的输出是给人判决用的,不是断言。
"""

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_zero_call_site_gates.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "audit_zero_call_site_gates", _SCRIPT
)
A = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = A
assert _SPEC.loader is not None
_SPEC.loader.exec_module(A)


def test_self_test_passes():
    """脚本自带的方法学自检必须绿 —— 它同时断言老做法会漏。"""

    assert A.self_test() == 0


@pytest.mark.parametrize(
    "name",
    [
        "assert_known_generation",  # 就是 2026-08-07 漏掉的那个形状
        "assert_x",
        "validate_x",
        "verify_x",
        "require_x",
        "check_x",
        "ensure_x",
        "enforce_x",
    ],
)
def test_public_gate_shapes_are_recognized(name):
    assert A.is_gate_name(name), f"公开门形状 {name} 没被认出来"


@pytest.mark.parametrize(
    "name", ["_assert_x", "_validate_x", "_require_x", "_must_x"]
)
def test_underscore_gate_shapes_still_recognized(name):
    assert A.is_gate_name(name)


@pytest.mark.parametrize(
    "name",
    ["assertion_count", "validated", "checkpoint", "requirements", "verify"],
)
def test_non_gate_names_are_not_swept_in(name):
    """形状要求 ``词 + 下划线``,否则 ``checkpoint`` 这种词会被卷进来。"""

    assert not A.is_gate_name(name)


def test_nested_defs_are_collected(tmp_path):
    source = tmp_path / "m.py"
    source.write_text(
        "class C:\n"
        "    def assert_inner(self):\n"
        "        def _validate_deeper():\n"
        "            pass\n"
        "        return _validate_deeper\n",
        encoding="utf-8",
    )
    names = {name for name, _lineno in A.collect_gate_defs(source)}
    assert names == {"assert_inner", "_validate_deeper"}


def test_agent_worktrees_are_excluded(tmp_path):
    """别人 worktree 里的调用点不许算成本仓有人用它。"""

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(
        "def assert_lonely():\n    pass\n", encoding="utf-8"
    )
    foreign = tmp_path / ".claude" / "worktrees" / "other"
    foreign.mkdir(parents=True)
    (foreign / "caller.py").write_text(
        "assert_lonely()\nassert_lonely()\n", encoding="utf-8"
    )
    findings, stats = A.scan(tmp_path)
    assert stats["py_files"] == 1, "别人的 worktree 被算进扫描规模了"
    assert [f.name for f in findings] == ["assert_lonely"], (
        "别人树里的调用点把一道死门伪装成了活门"
    )


def test_a_real_call_site_clears_the_finding(tmp_path):
    """反向判别力:同一份 fixture 加一个本仓调用点,就不该再报。"""

    (tmp_path / "m.py").write_text(
        "def assert_lonely():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "caller.py").write_text(
        "from m import assert_lonely\nassert_lonely()\n", encoding="utf-8"
    )
    findings, _stats = A.scan(tmp_path)
    assert findings == []


def test_script_is_syntactically_loadable():
    ast.parse(_SCRIPT.read_text(encoding="utf-8"))
