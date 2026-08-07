"""跑满 N 个 update 之后,最后一份存档的编号是 N-1 —— 这条约定由 RSL-RL 的活源码定。

人话
====
A211/C211 的 ``scale4096`` 验收门要"验明正身":跑完之后必须存在**那一份**终局
checkpoint,它的文件名编号和里面记的 ``iter`` 都得对得上,而且 ``infos`` 里得写着
这次是哪一份 launch claim 发出来的。

2026-08-07 之前这道门要的是 ``model_5.pt`` / ``iter == 5``(预算就是 5 个 update)。
**RSL-RL 落盘的末位却是 ``model_4.pt`` / ``iter == 4``**:它的迭代变量
``for it in range(start_iter, tot_iter)`` 在**循环体内**做
``self.current_learning_iteration = it``,循环结束后的收尾存盘用的就是那个末值。
于是这道门在任何预算下都不可能被满足 —— 不是"严格",是**瞄错了**。

这个文件做三件事:

1. **不手抄那个 4**。直接读能找到的每一份 RSL-RL ``on_policy_runner.py`` 活源码,
   用 AST 核对上面那条约定还成立(赋值在循环体内、循环外没有第二次赋值、收尾存盘用的
   就是这个属性),然后**真的跑一遍**同形状的
   ``for it in range(0, N)`` 把末值算出来。RSL-RL 哪天把赋值挪到循环外,这里当场红,
   而不是让发射器的门继续要一个不存在的文件。
2. 把这个末值和三个消费方(pre-long gate 常量、两个发射器的
   ``_terminal_checkpoint_iteration``、四格 barrier 的 ``TERMINAL_MODEL_ITERATION``)
   逐一对上,**A/C 两族必须同一个答案**。
3. 变异:把共享常量改回旧的差一格值、或让文件名与编号脱钩、或让发射器自己的预算与
   共享常量不一致 —— 三种情况都必须 ``LaunchRefused``,而且 A、C 一起红。

另外钉一条**不许出现自哈希循环**:launch claim 的 SHA 是对 ``training_argv`` 等内容
算出来的,所以它不能再进 argv。两个发射器都是在 exec 那一刻用环境变量
``HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256`` 把它交给 trainer 的;本文件核对 argv 构造
函数里确实没有任何 launch-claim 字样。
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


TESTS_DIR = Path(__file__).resolve().parent
WBT_DIR = TESTS_DIR.parent
SCRIPTS = WBT_DIR / "scripts"
REPO_ROOT = WBT_DIR.parents[1]
VENDORED_RSL_RL = (
    REPO_ROOT
    / "external_repos/TTRL-ICRA2026/rsl_rl/rsl_rl/runners/on_policy_runner.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load(
    "terminal_index_probe_prelong_gate",
    SCRIPTS / "action_ball_4096x5_prelong_gate.py",
)
A = _load(
    "terminal_index_probe_a211_launcher",
    SCRIPTS / "launch_action_ball_a211_four_arm_diagnostic.py",
)
C = _load(
    "terminal_index_probe_c211_launcher",
    SCRIPTS / "launch_action_ball_c211_diagnostic.py",
)



def _barrier():
    """四格 barrier **故意**在 import 时就对终局编号 fail closed。

    所以它只能懒加载:模块级 import 会把"常量漂了"变成整份测试收集不起来的
    collection error,盖住本文件真正想报的那句话(哪个常量和 RSL-RL 对不上)。
    """

    return _load(
        "terminal_index_probe_four_grid_barrier",
        SCRIPTS / "action_ball_211_four_grid_prelong_barrier.py",
    )


# ---------------------------------------------------------------------------
# RSL-RL 活源码:约定核对 + 末值实算
# ---------------------------------------------------------------------------


def _rsl_rl_sources() -> dict:
    """能找到几份 RSL-RL 就核对几份(pod 上装的那份 + 仓内 vendored 那份)。"""

    found = {}
    try:
        spec = importlib.util.find_spec("rsl_rl")
    except (ImportError, ValueError):  # pragma: no cover - host without rsl_rl
        spec = None
    if spec is not None and spec.origin:
        candidate = (
            Path(spec.origin).resolve().parent / "runners" / "on_policy_runner.py"
        )
        if candidate.is_file():
            found["installed"] = candidate
    if VENDORED_RSL_RL.is_file():
        found["vendored"] = VENDORED_RSL_RL
    return found


RSL_RL_SOURCES = _rsl_rl_sources()


def _self_assignments(node, attribute: str) -> list:
    rows = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Assign):
            continue
        for target in candidate.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == attribute
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                rows.append(candidate)
    return rows


def _learn_and_init(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunner":
            members = {
                item.name: item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "learn" in members and "__init__" in members:
                return members["__init__"], members["learn"]
    raise AssertionError("%s has no OnPolicyRunner.learn to read" % path)


def _model_filename_formatted_value(call: ast.Call):
    """``self.save(os.path.join(..., f"model_{X}.pt"))`` 里的那个 ``X``。"""

    for node in ast.walk(call):
        if not isinstance(node, ast.JoinedStr):
            continue
        literals = [
            piece.value
            for piece in node.values
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
        ]
        formatted = [
            piece.value
            for piece in node.values
            if isinstance(piece, ast.FormattedValue)
        ]
        if (
            len(formatted) == 1
            and any(text.endswith("model_") for text in literals)
            and any(text.startswith(".pt") for text in literals)
        ):
            return formatted[0]
    return None


def _is_self_save(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _terminal_iteration_from_live_source(path: Path, num_updates: int) -> int:
    """核对 RSL-RL 的迭代/存盘约定,然后**跑一遍**同形状的循环取末值。"""

    init, learn = _learn_and_init(path)

    zero_start = [
        node
        for node in _self_assignments(init, "current_learning_iteration")
        if isinstance(node.value, ast.Constant) and node.value.value == 0
    ]
    assert zero_start, "%s: 新建 runner 的 current_learning_iteration 不再从 0 起" % path

    body = learn.body
    loop_index = None
    loop = None
    for index, statement in enumerate(body):
        if not isinstance(statement, ast.For):
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        in_loop = [
            node
            for node in _self_assignments(statement, "current_learning_iteration")
            if isinstance(node.value, ast.Name)
            and node.value.id == statement.target.id
        ]
        if in_loop:
            assert loop is None, "%s: 有不止一个迭代循环写 current_learning_iteration" % path
            loop_index = index
            loop = statement
    assert loop is not None, (
        "%s: 找不到「循环体内把迭代号写进 current_learning_iteration」的那个循环" % path
    )
    loop_variable = loop.target.id

    assert (
        isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "range"
        and len(loop.iter.args) == 2
    ), "%s: 迭代循环不再是 range(start, stop) 形状" % path

    inside = set(id(node) for node in ast.walk(loop))
    outside = [
        node
        for node in _self_assignments(learn, "current_learning_iteration")
        if id(node) not in inside
    ]
    assert not outside, (
        "%s: 循环之外又给 current_learning_iteration 赋值了,末值不再是 N-1" % path
    )

    finals = []
    for statement in body[loop_index + 1 :]:
        for node in ast.walk(statement):
            if _is_self_save(node):
                finals.append(node)
    assert len(finals) == 1, "%s: 循环之后的收尾存盘不是恰好一次 self.save" % path
    formatted = _model_filename_formatted_value(finals[0])
    assert (
        formatted is not None
        and isinstance(formatted, ast.Attribute)
        and formatted.attr == "current_learning_iteration"
        and isinstance(formatted.value, ast.Name)
        and formatted.value.id == "self"
    ), "%s: 收尾存盘的文件名不再是 model_{self.current_learning_iteration}.pt" % path

    in_loop_saves = [
        node
        for node in ast.walk(loop)
        if _is_self_save(node)
        and isinstance(_model_filename_formatted_value(node), ast.Name)
    ]
    assert in_loop_saves, "%s: 循环内不再按迭代号存 model_{it}.pt" % path

    # 到这里"末值就是循环最后一次的 it"已经被核对过了。真的跑一遍同形状的循环取末值,
    # 而不是再手写一个 num_updates - 1。
    observed = None
    for iteration in range(0, num_updates):
        observed = iteration
    assert observed is not None
    return observed


def test_at_least_one_rsl_rl_source_is_readable():
    """一份 RSL-RL 都看不到时必须红:这条 pin 不能在无凭据的情况下静默通过。"""

    assert RSL_RL_SOURCES, "找不到任何 RSL-RL on_policy_runner.py,终局编号无从核对"


@pytest.mark.parametrize("origin", sorted(RSL_RL_SOURCES))
def test_every_visible_rsl_rl_agrees_the_terminal_index_is_n_minus_one(origin):
    path = RSL_RL_SOURCES[origin]
    assert (
        _terminal_iteration_from_live_source(path, GATE.EXPECTED_UPDATES)
        == GATE.EXPECTED_UPDATES - 1
    )


def test_shared_gate_constants_match_the_live_rsl_rl_convention():
    origin, path = sorted(RSL_RL_SOURCES.items())[0]
    terminal = _terminal_iteration_from_live_source(path, GATE.EXPECTED_UPDATES)
    assert GATE.TERMINAL_CHECKPOINT_ITERATION == terminal, origin
    assert GATE.TERMINAL_CHECKPOINT_FILENAME == "model_%d.pt" % terminal
    # 预算本身没变(4096 envs x 5 updates),别顺手把 5 也改了。
    assert GATE.EXPECTED_UPDATES == 5
    assert GATE.TERMINAL_CHECKPOINT_FILENAME == "model_4.pt"


# ---------------------------------------------------------------------------
# 三个消费方必须同一个答案(A/C 两族一起)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("launcher", (A, C), ids=("a211", "c211"))
def test_both_launchers_aim_at_the_last_written_checkpoint(launcher):
    expected_updates = launcher.BUDGETS["scale4096"][1]
    assert expected_updates == GATE.EXPECTED_UPDATES
    assert (
        launcher._terminal_checkpoint_iteration(expected_updates)
        == GATE.TERMINAL_CHECKPOINT_ITERATION
    )
    assert launcher._P.TERMINAL_CHECKPOINT_FILENAME == "model_4.pt"


def test_a_and_c_families_share_one_terminal_checkpoint_answer():
    """Franco 的口径:A 和 C 除了 obs 和 reward 之外应当处处相同。"""

    assert A.BUDGETS["scale4096"] == C.BUDGETS["scale4096"]
    assert A._terminal_checkpoint_iteration(
        A.BUDGETS["scale4096"][1]
    ) == C._terminal_checkpoint_iteration(C.BUDGETS["scale4096"][1])
    assert (
        A._P.TERMINAL_CHECKPOINT_FILENAME == C._P.TERMINAL_CHECKPOINT_FILENAME
    )
    assert _barrier().TERMINAL_MODEL_ITERATION == GATE.TERMINAL_CHECKPOINT_ITERATION


def test_the_file_that_owns_the_terminal_index_is_inside_both_launch_claims():
    """决定终局编号的那份源码必须在两族的 provenance 钉子表里,否则等于放宽 fail-closed。"""

    for launcher in (A, C):
        pinned = {relative for relative, _label in launcher.RUNTIME_SOURCE_PATHS}
        assert any(
            str(relative).endswith("action_ball_4096x5_prelong_gate.py")
            for relative in pinned
        )


# ---------------------------------------------------------------------------
# 变异:该拦的仍拦
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("launcher", (A, C), ids=("a211", "c211"))
def test_launcher_refuses_when_the_terminal_index_drifts_back_to_the_budget(
    launcher, monkeypatch
):
    """把共享常量改回旧的差一格值(= 预算 5),必须当场拒。"""

    monkeypatch.setattr(
        launcher._P,
        "TERMINAL_CHECKPOINT_ITERATION",
        launcher._P.EXPECTED_UPDATES,
    )
    with pytest.raises(launcher.LaunchRefused, match="terminal"):
        launcher._terminal_checkpoint_iteration(launcher.BUDGETS["scale4096"][1])


@pytest.mark.parametrize("launcher", (A, C), ids=("a211", "c211"))
def test_launcher_refuses_when_filename_and_index_are_decoupled(
    launcher, monkeypatch
):
    monkeypatch.setattr(
        launcher._P, "TERMINAL_CHECKPOINT_FILENAME", "model_5.pt"
    )
    with pytest.raises(launcher.LaunchRefused, match="terminal"):
        launcher._terminal_checkpoint_iteration(launcher.BUDGETS["scale4096"][1])


@pytest.mark.parametrize("launcher", (A, C), ids=("a211", "c211"))
def test_launcher_refuses_when_its_own_budget_leaves_the_shared_one(launcher):
    for wrong in (4, 6, 0, -1, True, 5.0, "5", None):
        with pytest.raises(launcher.LaunchRefused, match="terminal"):
            launcher._terminal_checkpoint_iteration(wrong)


@pytest.mark.parametrize("launcher", (A, C), ids=("a211", "c211"))
def test_launcher_refuses_when_the_shared_expected_updates_drifts(
    launcher, monkeypatch
):
    monkeypatch.setattr(launcher._P, "EXPECTED_UPDATES", 6)
    with pytest.raises(launcher.LaunchRefused, match="terminal"):
        launcher._terminal_checkpoint_iteration(launcher.BUDGETS["scale4096"][1])


# ---------------------------------------------------------------------------
# 不许出现自哈希循环
# ---------------------------------------------------------------------------


LAUNCHER_SOURCE_FILES = {
    "a211": SCRIPTS / "launch_action_ball_a211_four_arm_diagnostic.py",
    "c211": SCRIPTS / "launch_action_ball_c211_diagnostic.py",
}
CLAIM_EXEC_ENV = "HOPE_N1_DIAGNOSTIC_LAUNCH_CLAIM_SHA256"


def _function_node(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, "%s: 期望恰好一个 %s" % (path, name)
    return matches[0]


@pytest.mark.parametrize("family", sorted(LAUNCHER_SOURCE_FILES))
def test_launch_claim_never_enters_the_argv_it_hashes(family):
    """claim 的哈希算的就是 argv,所以 claim 不能再出现在 argv 里(自哈希循环)。"""

    path = LAUNCHER_SOURCE_FILES[family]
    argv_builder = _function_node(path, "_training_argv")
    literals = [
        node.value
        for node in ast.walk(argv_builder)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    for text in literals:
        assert "launch_claim" not in text, text
        assert CLAIM_EXEC_ENV not in text
    names = {
        node.id for node in ast.walk(argv_builder) if isinstance(node, ast.Name)
    }
    assert "claim_sha" not in names


@pytest.mark.parametrize("family", sorted(LAUNCHER_SOURCE_FILES))
def test_launch_claim_reaches_the_trainer_through_the_exec_environment(family):
    """它走的是 exec 边界的环境变量,而且只此一处。"""

    source = LAUNCHER_SOURCE_FILES[family].read_text(encoding="utf-8")
    assert source.count('"%s": claim_sha,' % CLAIM_EXEC_ENV) == 1
    assert source.count("os.execve(argv[0], argv, environment)") == 1
