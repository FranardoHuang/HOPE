"""共享 4096x5 门在**导入期**跟生产方对的那张活值表,到底会不会拒。

人话:`action_ball_4096x5_prelong_gate.py` 里有三张表 ——
21 个 WAIT 档位、mimic 项集合、哪些 mimic 项走 Cauchy 核 —— 它们是这道门
判揭示->回放那本账时的**判据本身**。2026-08-07 接线那一批给它们加了一条
import 期检查:必须逐一等于生产方 `action_ball_prelong_semantics.py` 的活值,
对不上就 `RuntimeError`,模块根本导不进来。

那条检查是承重的:
  * 档位错位 -> 逐档守恒式照样成立,但比的是错档号,拒绝面整体走形;
  * Cauchy/exp 归类不一致 -> 核函数字符串检查会拒掉正确收据,或放过错的;
  * mimic 项集合不一致 -> 拒绝理由会指错地方。

但它当时**零测试**:把那整段 `if ... raise RuntimeError` 换成 `pass`,
四个相关测试模块 447 条一条都不红(2026-08-07 独立验收实测,见 exp §9.2.15)。
本模块补上这一条,并且刻意**不放进** `test_action_ball_4096x5_prelong_gate.py`
—— 那个文件同期有别的 workflow 在大改,单独一个模块不会跟他们抢同一片字节。

怎么验的:把源码里的档位常量改窄一档,在一个新命名空间里重新 exec
(`__file__` 仍指向真文件,所以它照样按路径找得到生产方),必须抛 RuntimeError;
再用**没改过**的同一份源码 exec 一次作控制组,证明红的原因是那条检查本身,
不是"这段代码本来就跑不起来"。
"""

from __future__ import annotations

from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "action_ball_4096x5_prelong_gate.py"
)

_COHORT_LINE = "BRIDGE_WAIT_COHORTS = tuple(range(5, 26))"
_REFUSAL = "reveal-bridge contract differs from the semantic producer"


def _exec(text: str) -> dict:
    namespace = {
        "__file__": str(SCRIPT),
        "__name__": "_prelong_gate_producer_contract_probe",
    }
    exec(compile(text, str(SCRIPT), "exec"), namespace)  # noqa: S102
    return namespace


def test_the_unmutated_gate_source_still_executes():
    """控制组先跑:原样那份必须 exec 得通,而且确实是 21 个 WAIT 档。"""

    namespace = _exec(SCRIPT.read_text())
    assert len(namespace["BRIDGE_WAIT_COHORTS"]) == 21
    assert namespace["BRIDGE_CAUCHY_MIMIC_TERMS"]
    assert namespace["BRIDGE_MIMIC_TERMS"]


def test_narrowing_the_wait_cohorts_is_refused_at_import_time():
    """把 21 档改成 20 档 —— 导入就必须炸,而且是那条检查的原话。"""

    source = SCRIPT.read_text()
    narrowed = source.replace(
        _COHORT_LINE, "BRIDGE_WAIT_COHORTS = tuple(range(5, 25))", 1
    )
    assert narrowed != source, (
        f"锚点 {_COHORT_LINE!r} 在源码里找不到了 —— 先更新这条测试,"
        "别让它退化成空跑"
    )
    with pytest.raises(RuntimeError, match=_REFUSAL):
        _exec(narrowed)


def test_reclassifying_a_mimic_kernel_is_refused_at_import_time():
    """把一项 Cauchy mimic 挪出 Cauchy 名单 —— 同样导入就炸。

    这一条盖的是另一半:档位对上了,但核函数归类漂了。
    """

    source = SCRIPT.read_text()
    dropped = source.replace('    "motion_racket_long_axis",\n', "", 1)
    assert dropped != source, (
        "锚点 motion_racket_long_axis 那一行不在了 —— 先更新这条测试"
    )
    with pytest.raises(RuntimeError, match=_REFUSAL):
        _exec(dropped)
