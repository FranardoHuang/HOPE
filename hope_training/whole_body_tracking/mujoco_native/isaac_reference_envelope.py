"""Which reference envelope does the native ActionBall lane actually run?

人话:这条 MuJoCo 复刻车道镜像的是 Isaac 的 ``HOPEActionBallTerminationsCfg``,
不是它的父类。父类 ``HOPEDeployParityTerminationsCfg`` 的 ``ee_body_pos`` 看
"两脚 + 两腕";ActionBall 子类把它**收窄成只有两脚** —— 腕本来就是挥拍时要甩最远
的那一端,0.25 m 的 z 包络套在它身上等于在惩罚我们要教的动作(``build_1`` V9 实测
新策略几乎每次 reset 都在 1.67 步内被腕部 guard 掐掉)。复刻此前抄的是父类那份四个
身体的名单,于是它会在现役 kernel 明确放行的腕部位移上终止。

So this module reads that one term back out of the live Isaac config **as
values** -- the body names, the threshold, and which class in the inheritance
chain actually supplied them -- instead of keeping a fourth hand-written copy of
the list.  It is the same paradigm as
``action_ball_211_abi.live_source_parity_blockers``: a SHA can only say "these
bytes moved", it can never say "the replica runs the same envelope".  ``5ed998f1``
is the worked example -- the AST fingerprint was re-stamped inside the very
commit that changed the Isaac semantics, and the replica sat unchanged for two
days because every test fed all four bodies the same number.

It also answers a second question the AST fingerprint structurally cannot: **which
terms does that class declare at all?**  The phase-fidelity selector names its
terms one by one, so a term nobody listed is invisible to it -- adding a fresh
override to ``HOPEActionBallTerminationsCfg`` would not move the pinned digest by
a single bit.  :func:`live_declared_term_blockers` compares the live class body
against what this lane knows about, so a new or deleted term fails closed.

And a third: **in what ORDER does Isaac evaluate those terms?**

人话:终止原因的**先后**也是一份手抄件,而且抄的是一条横跨两个文件、三个类的继承
链 —— ``tracking_env_cfg.TerminationsCfg``(time_out / anchor_pos / anchor_ori /
ee_body_pos)→ ``HOPEDeployParityTerminationsCfg``(加 base_fell_tilt /
base_too_low / robot_hit_table)→ ``HOPEActionBallTerminationsCfg``(加两条
joint_*)。``configclass`` 是 dataclass 底子,字段顺序 = 先父类按声明序、再子类的新
字段;子类覆写一条**不会**把它挪到队尾,它留在父类给的位置上。同一步里两条终止同时
成立时,谁在前谁就是被记录的那个原因 —— 所以这个顺序就是"实验把锅算在谁头上"。

那份手抄件此前只有 AST 指纹罩着,而且**指纹的覆盖面比它保护的语义面还小**:
``base_config`` 选择器只点了 ``TerminationsCfg|time_out`` 一个名字,于是往那个类里
新加一条终止项、或者把 ``anchor_pos``/``ee_body_pos`` 换个位置,指纹一个 bit 都不会
动 —— 和 ``ee_body_pos`` 那个窟窿是同一个形状,只是高了一层。
:func:`live_termination_reason_order` 把这条链的真实字段顺序算出来,
:func:`live_termination_reason_order_blockers` 拿它跟复刻的四份原因名单逐位比,
并要求这四份名单**恰好划分**现役的终止项集合 —— 多一条少一条都落不进任何一格。

The Isaac config cannot be imported on a plain host (it pulls all of Isaac Lab),
so values are recovered from the AST by
:mod:`~mujoco_native.isaac_live_constants`, whose evaluator is a small
whitelist and refuses to guess.
"""

from __future__ import annotations

import ast
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import isaac_live_constants


REPO_ROOT = Path(__file__).resolve().parents[3]

#: The live Isaac task config that owns every HOPE termination class.
ISAAC_TERMINATION_CONFIG = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
#: The live Isaac robot leaf that owns the A3 body-name lists.
ISAAC_BODY_NAME_LISTS = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
#: The live Isaac file that owns the *grandparent* termination class.  The two
#: HOPE classes live in ``hope_env_cfg.py``; the class they both ultimately
#: derive from does not, and it is the one that supplies the HEAD of the
#: evaluation order (``time_out`` first, then the three reference terms).
ISAAC_BASE_TERMINATION_CONFIG = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
)
#: Isaac module path -> live file, for names ``hope_env_cfg`` imports rather
#: than assigns (``A3_FEET_BODIES`` / ``A3_HAND_BODIES``).
BODY_NAME_MODULE = "whole_body_tracking.robots.agibot_a3"

#: The class this lane mirrors, and the parent it deliberately does NOT.
ACTION_BALL_TERMINATIONS_CLASS = "HOPEActionBallTerminationsCfg"
DEPLOY_PARITY_TERMINATIONS_CLASS = "HOPEDeployParityTerminationsCfg"
#: The grandparent both derive from, declared in another file entirely.
BASE_TERMINATIONS_CLASS = "TerminationsCfg"

#: Termination base classes that live outside the main task config, and the file
#: that declares each one.  A base this map does not name and the config file
#: does not declare ends the walk with a refusal -- an inheritance step this
#: reader cannot see is an evaluation order it must not pretend to know.
EXTERNAL_TERMINATION_BASES = {
    BASE_TERMINATIONS_CLASS: ISAAC_BASE_TERMINATION_CONFIG,
}

#: The reference-envelope term whose ActionBall override this lane must follow.
REFERENCE_ENVELOPE_TERM = "ee_body_pos"

#: Every termination term each class is known to declare, and one plain-language
#: line for each so a reader does not have to open the Isaac file.
#:
#: 人话:这两张表是"这条车道知道这个类里有哪几条终止项"。多一条少一条都必须有人
#: 来看 —— 指纹的选择器按名字点名,新名字它天生看不见。
DECLARED_TERMS = {
    BASE_TERMINATIONS_CLASS: {
        "time_out": "回合走满时长(截断,不算硬终止)",
        "anchor_pos": "躯干高度偏离参考太多就终止",
        "anchor_ori": "躯干朝向偏离参考太多就终止",
        "ee_body_pos": "被点名的末端 body 高度偏离参考太多就终止",
    },
    DEPLOY_PARITY_TERMINATIONS_CLASS: {
        "anchor_pos": "躯干高度偏离参考太多就终止",
        "anchor_ori": "躯干朝向偏离参考太多就终止",
        "ee_body_pos": "被点名的末端 body 高度偏离参考太多就终止",
        "base_fell_tilt": "摔倒(绝对倾角)",
        "base_too_low": "坐下去了(绝对高度)",
        "robot_hit_table": "机器人撞到桌子",
    },
    ACTION_BALL_TERMINATIONS_CLASS: {
        "ee_body_pos": "把参考包络收窄成只有双脚,腕部不再终止",
        "joint_qdes_forbidden": "指令关节角非有限/进入物理禁区",
        "joint_actual_forbidden": "实测关节角撞机械硬限位(现在只记录不终止)",
    },
}

#: Body-name lists in the live A3 robot leaf that a reference envelope may draw
#: from.  A name outside these lists is a typo or an Isaac-only spelling and
#: would not resolve against the MuJoCo model.
BODY_NAME_VOCABULARY_SYMBOLS = ("A3_FEET_BODIES", "A3_HAND_BODIES")


class IsaacReferenceEnvelopeError(RuntimeError):
    """The live Isaac reference envelope cannot be read as values."""


def _companions(body_names_path: Any = None) -> dict:
    return {
        BODY_NAME_MODULE: (
            Path(body_names_path)
            if body_names_path is not None
            else ISAAC_BODY_NAME_LISTS
        )
    }


def _config(config_path: Any = None) -> Path:
    return Path(config_path) if config_path is not None else ISAAC_TERMINATION_CONFIG


def _external_bases(base_config_path: Any = None) -> dict:
    """``class name -> file``, with the one out-of-file base optionally repointed."""

    if base_config_path is None:
        return {name: Path(path) for name, path in EXTERNAL_TERMINATION_BASES.items()}
    return {BASE_TERMINATIONS_CLASS: Path(base_config_path)}


def _class_source(
    class_name: str, config_path: Any = None, base_config_path: Any = None
) -> Path:
    """The file that declares ``class_name``, main config unless declared external."""

    external = _external_bases(base_config_path)
    if class_name in external:
        return external[class_name]
    return _config(config_path)


# ---------------------------------------------------------------------------
# Which terms does the live class declare?
# ---------------------------------------------------------------------------


def _class_defs(source: Path) -> dict:
    try:
        stat = source.stat()
    except OSError as exc:
        raise IsaacReferenceEnvelopeError(
            f"cannot open the live Isaac termination config {source}: {exc}"
        ) from exc
    # Keyed on the file's identity AND its stamp, so a test that rewrites a
    # path in place still gets the new contents.
    return _class_defs_cached(str(source), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _class_defs_cached(source_text: str, _mtime_ns: int, _size: int) -> dict:
    source = Path(source_text)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise IsaacReferenceEnvelopeError(
            f"cannot parse the live Isaac termination config {source}: {exc}"
        ) from exc
    classes: dict = {}
    duplicated = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name in classes:
            duplicated.add(node.name)
        classes[node.name] = node
    for name in duplicated:
        classes.pop(name, None)
    return classes


def _class_body_terms(node: ast.ClassDef) -> tuple:
    """``(name, call node)`` for every term the class body constructs, in order.

    A termination term is always built by a call -- ``DoneTerm(...)`` for the
    ones written out longhand, ``table_hit_done_term()`` for the one built by a
    shared factory.  Plain literals (``obs_mode = "full"``, a WIP marker) are
    not terms and are deliberately not counted: the phase-fidelity pin already
    documents that unrelated assignments inside these classes must not disturb
    it, and widening this to every assignment would contradict that.
    """

    terms = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            targets: Sequence[ast.AST] = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
        else:
            continue
        value = getattr(statement, "value", None)
        if not isinstance(value, ast.Call):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                terms.append((target.id, value))
    return tuple(terms)


def _class_body_names(node: ast.ClassDef) -> tuple:
    """Names the class body binds to a *constructed* value, in source order."""

    return tuple(name for name, _call in _class_body_terms(node))


def live_declared_terms(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    config_path: Any = None,
    *,
    base_config_path: Any = None,
) -> tuple:
    """Every termination term the live class body constructs, in source order."""

    source = _class_source(class_name, config_path, base_config_path)
    classes = _class_defs(source)
    if class_name not in classes:
        raise IsaacReferenceEnvelopeError(
            f"live Isaac class {class_name} is absent or declared twice in {source}"
        )
    return _class_body_names(classes[class_name])


def live_chain_sources(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    config_path: Any = None,
    *,
    base_config_path: Any = None,
) -> tuple:
    """``(class, declaring file)`` most-derived first, across files.

    A base that is neither declared in the file being walked nor listed in
    :data:`EXTERNAL_TERMINATION_BASES` raises: an inheritance step this reader
    cannot see is a field order it must not pretend to know.  (Before the
    grandparent was registered, such a base silently *ended* the walk -- which
    is how the head of the evaluation order stayed unguarded.)
    """

    external = _external_bases(base_config_path)
    chain = []
    seen = set()
    current = class_name
    source = _class_source(class_name, config_path, base_config_path)
    while True:
        classes = _class_defs(source)
        if current not in classes:
            raise IsaacReferenceEnvelopeError(
                f"live Isaac class {current} is absent or declared twice in {source}"
            )
        if current in seen:
            raise IsaacReferenceEnvelopeError(
                f"live Isaac class chain for {class_name} is cyclic"
            )
        seen.add(current)
        chain.append((current, source))
        bases = list(classes[current].bases)
        if not bases:
            return tuple(chain)
        if len(bases) != 1:
            raise IsaacReferenceEnvelopeError(
                f"live Isaac class {current} has {len(bases)} bases; this value "
                "reader only follows single inheritance"
            )
        base = bases[0]
        if not isinstance(base, ast.Name):
            raise IsaacReferenceEnvelopeError(
                f"live Isaac class {current} derives from an expression this "
                "reader cannot resolve to a class name"
            )
        if base.id in classes:
            current = base.id
        elif base.id in external:
            current, source = base.id, external[base.id]
        else:
            raise IsaacReferenceEnvelopeError(
                f"live Isaac class {current} derives from {base.id}, which is "
                f"declared neither in {source} nor in any file this reader was "
                "pointed at (see EXTERNAL_TERMINATION_BASES)"
            )


def live_class_chain(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    config_path: Any = None,
    *,
    base_config_path: Any = None,
) -> tuple:
    """The inheritance chain, most-derived first."""

    return tuple(
        name
        for name, _source in live_chain_sources(
            class_name, config_path, base_config_path=base_config_path
        )
    )


def live_term_owner_class(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    term_name: str = REFERENCE_ENVELOPE_TERM,
    config_path: Any = None,
    *,
    base_config_path: Any = None,
) -> str:
    """Which class in the live chain actually supplies ``term_name``.

    人话:子类覆写了就是子类说了算,没覆写就沿着继承往上找 —— 和 Python 自己
    解析的结果一致。以前复刻抄的就是"没意识到子类覆写过"这一步。
    """

    for name, source in live_chain_sources(
        class_name, config_path, base_config_path=base_config_path
    ):
        if term_name in _class_body_names(_class_defs(source)[name]):
            return name
    raise IsaacReferenceEnvelopeError(
        f"live Isaac class chain for {class_name} declares no {term_name}"
    )


def live_declared_term_blockers(
    config_path: Any = None, *, base_config_path: Any = None
) -> tuple:
    """Every class whose live term list is no longer the one this lane knows.

    人话:这道门专门补 AST 指纹选择器的天生盲区 —— 选择器只认它点过名的项,
    有人往这个类里**新加**一条终止项,指纹一个 bit 都不会动。这里比的是集合。
    """

    blockers = []
    for class_name, expected in DECLARED_TERMS.items():
        try:
            live = live_declared_terms(
                class_name, config_path, base_config_path=base_config_path
            )
        except IsaacReferenceEnvelopeError as exc:
            blockers.append(f"isaac_declared_terms_unreadable:{class_name}:{exc}")
            continue
        if len(set(live)) != len(live):
            blockers.append(
                f"isaac_declared_terms_duplicated:{class_name}:live={live!r}"
            )
            continue
        if set(live) != set(expected):
            added = sorted(set(live) - set(expected))
            removed = sorted(set(expected) - set(live))
            blockers.append(
                f"isaac_declared_terms_differ:{class_name}:"
                f"added={added!r} removed={removed!r}"
            )
    return tuple(blockers)


# ---------------------------------------------------------------------------
# In what ORDER does Isaac evaluate the terms?
# ---------------------------------------------------------------------------


def _live_term_calls(chain: Sequence) -> dict:
    """``term -> the constructor call that WINS`` (most-derived declaration)."""

    calls: dict = {}
    for name, source in chain:  # most-derived first, so the first one wins
        for term, call in _class_body_terms(_class_defs(source)[name]):
            calls.setdefault(term, call)
    return calls


def live_termination_reason_order(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    *,
    config_path: Any = None,
    base_config_path: Any = None,
) -> tuple:
    """The order IsaacLab's TerminationManager will evaluate the live terms in.

    ``configclass`` is dataclass-backed, so the field order is: every base's
    fields first (most-basal base first, in its own source order), then each
    subclass's NEW fields in source order.  **An override keeps the slot its
    base gave it** -- ``HOPEActionBallTerminationsCfg.ee_body_pos`` re-points the
    term but does not move it behind ``base_fell_tilt``.  Getting that wrong is
    not cosmetic: when two terms fire on the same step, the first one is the
    reason the episode is recorded under, i.e. which guard an experiment blames.
    """

    chain = live_chain_sources(
        class_name, config_path, base_config_path=base_config_path
    )
    order: list = []
    for name, source in reversed(chain):  # most-basal first
        for term, _call in _class_body_terms(_class_defs(source)[name]):
            if term not in order:
                order.append(term)
    return tuple(order)


def live_timeout_term_names(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    *,
    config_path: Any = None,
    base_config_path: Any = None,
) -> frozenset:
    """Live terms built with ``time_out=True`` -- truncations, not hard deaths.

    人话:哪几条是"回合走满"而不是"真摔了",也从活的构造调用里读,不再靠手抄。
    """

    chain = live_chain_sources(
        class_name, config_path, base_config_path=base_config_path
    )
    names = set()
    for term, call in _live_term_calls(chain).items():
        for keyword in call.keywords:
            if (
                keyword.arg == "time_out"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                names.add(term)
    return frozenset(names)


def _is_subsequence(needle: Sequence, haystack: Sequence) -> bool:
    iterator = iter(haystack)
    return all(item in iterator for item in needle)


def live_termination_reason_order_blockers(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    *,
    config_path: Any = None,
    base_config_path: Any = None,
    mirrored_active_order: Sequence = (),
    mirrored_hard_order: Sequence = (),
    mirrored_partition: Mapping[str, Sequence] = (),
) -> tuple:
    """Every way the replica's termination reason lists disagree with Isaac's.

    ``mirrored_partition`` maps a label to one of the replica's narrower ordered
    lists (phase-fidelity reasons, base/joint reasons, the table-guard reason).
    Together they must **exactly partition** the live hard terms: every live term
    lands in exactly one bucket, in the live relative order.  That is what makes
    a brand-new Isaac termination term fail closed -- it belongs to no bucket,
    and no amount of re-pinning a digest changes that.
    """

    try:
        live_order = live_termination_reason_order(
            class_name, config_path=config_path, base_config_path=base_config_path
        )
        live_timeouts = live_timeout_term_names(
            class_name, config_path=config_path, base_config_path=base_config_path
        )
    except IsaacReferenceEnvelopeError as exc:
        return (f"isaac_reason_order_unreadable:{class_name}:{exc}",)

    blockers = []
    live_hard = tuple(term for term in live_order if term not in live_timeouts)
    if tuple(mirrored_active_order) != live_order:
        blockers.append(
            f"isaac_active_reason_order_differs:{class_name}:"
            f"live={live_order!r} replica={tuple(mirrored_active_order)!r}"
        )
    if tuple(mirrored_hard_order) != live_hard:
        blockers.append(
            f"isaac_hard_reason_order_differs:{class_name}:"
            f"live={live_hard!r} replica={tuple(mirrored_hard_order)!r}"
        )

    buckets = dict(mirrored_partition)
    covered: list = []
    for label, bucket in buckets.items():
        bucket = tuple(bucket)
        if not _is_subsequence(bucket, live_hard):
            blockers.append(
                f"isaac_reason_bucket_out_of_live_order:{label}:"
                f"live={live_hard!r} replica={bucket!r}"
            )
        covered.extend(bucket)
    duplicated = sorted({name for name in covered if covered.count(name) > 1})
    if duplicated:
        blockers.append(
            f"isaac_reason_bucket_overlap:{class_name}:duplicated={duplicated!r}"
        )
    if set(covered) != set(live_hard):
        unbucketed = sorted(set(live_hard) - set(covered))
        unknown = sorted(set(covered) - set(live_hard))
        blockers.append(
            f"isaac_reason_partition_differs:{class_name}:"
            f"live_terms_in_no_replica_bucket={unbucketed!r} "
            f"replica_reasons_isaac_does_not_declare={unknown!r}"
        )
    return tuple(blockers)


# ---------------------------------------------------------------------------
# The reference envelope itself, as values
# ---------------------------------------------------------------------------


def live_body_name_vocabulary(body_names_path: Any = None) -> tuple:
    """Every A3 body name a reference envelope is allowed to name."""

    source = (
        Path(body_names_path) if body_names_path is not None else ISAAC_BODY_NAME_LISTS
    )
    vocabulary = []
    for symbol in BODY_NAME_VOCABULARY_SYMBOLS:
        try:
            value = isaac_live_constants.live_value(source, ("assignment", symbol))
        except isaac_live_constants.IsaacLiveConstantError as exc:
            raise IsaacReferenceEnvelopeError(
                f"cannot read live A3 body-name list {symbol}: {exc}"
            ) from exc
        if not isinstance(value, tuple) or not all(
            isinstance(name, str) and name for name in value
        ):
            raise IsaacReferenceEnvelopeError(
                f"live A3 body-name list {symbol} is not a list of body names"
            )
        vocabulary.extend(value)
    return tuple(vocabulary)


def live_reference_envelope(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    *,
    config_path: Any = None,
    body_names_path: Any = None,
) -> dict:
    """The ``ee_body_pos`` envelope the live class actually ships.

    Returns the body names in source order and the z threshold, both read as
    values.  Raises rather than guessing: an envelope this lane cannot read is
    an envelope it must not pretend to reproduce.
    """

    companions = _companions(body_names_path)
    owner = live_term_owner_class(class_name, REFERENCE_ENVELOPE_TERM, config_path)
    # The owner may be the grandparent, which lives in a different file; read the
    # value out of the file that actually declares the winning term.
    source = _class_source(owner, config_path)
    values = {}
    for param in ("body_names", "threshold"):
        selector = ("class_term_param", owner, REFERENCE_ENVELOPE_TERM, param)
        try:
            values[param] = isaac_live_constants.live_value(
                source, selector, companions=companions
            )
        except isaac_live_constants.IsaacLiveConstantError as exc:
            raise IsaacReferenceEnvelopeError(
                f"cannot read live {class_name}.{REFERENCE_ENVELOPE_TERM} "
                f"{param}: {exc}"
            ) from exc

    body_names = values["body_names"]
    threshold = values["threshold"]
    vocabulary = live_body_name_vocabulary(body_names_path)
    if (
        not isinstance(body_names, tuple)
        or not body_names
        or len(set(body_names)) != len(body_names)
        or not all(isinstance(name, str) and name for name in body_names)
    ):
        raise IsaacReferenceEnvelopeError(
            f"live {class_name}.{REFERENCE_ENVELOPE_TERM} body_names is not a "
            f"non-empty list of distinct body names: {body_names!r}"
        )
    unknown = [name for name in body_names if name not in vocabulary]
    if unknown:
        raise IsaacReferenceEnvelopeError(
            f"live {class_name}.{REFERENCE_ENVELOPE_TERM} names bodies that are "
            f"not in the A3 body-name lists: {unknown!r}"
        )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise IsaacReferenceEnvelopeError(
            f"live {class_name}.{REFERENCE_ENVELOPE_TERM} threshold is not a "
            f"number: {threshold!r}"
        )
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise IsaacReferenceEnvelopeError(
            f"live {class_name}.{REFERENCE_ENVELOPE_TERM} threshold must be "
            f"finite and positive: {threshold!r}"
        )
    return {
        "class_name": class_name,
        "owner_class": owner,
        "term_name": REFERENCE_ENVELOPE_TERM,
        "body_names": tuple(body_names),
        "threshold_m": threshold,
    }


def live_reference_envelope_blockers(
    class_name: str = ACTION_BALL_TERMINATIONS_CLASS,
    *,
    config_path: Any = None,
    body_names_path: Any = None,
    mirrored_body_names: Sequence = (),
    mirrored_threshold_m: Any = None,
) -> tuple:
    """Every way the replica's envelope disagrees with the live Isaac one."""

    try:
        live = live_reference_envelope(
            class_name, config_path=config_path, body_names_path=body_names_path
        )
    except IsaacReferenceEnvelopeError as exc:
        return (f"isaac_reference_envelope_unreadable:{class_name}:{exc}",)
    blockers = []
    if tuple(mirrored_body_names) != live["body_names"]:
        blockers.append(
            f"isaac_reference_envelope_body_names_differ:{class_name}:"
            f"live={live['body_names']!r} replica={tuple(mirrored_body_names)!r}"
        )
    if mirrored_threshold_m is None or float(mirrored_threshold_m) != (
        live["threshold_m"]
    ):
        blockers.append(
            f"isaac_reference_envelope_threshold_differs:{class_name}:"
            f"live={live['threshold_m']!r} replica={mirrored_threshold_m!r}"
        )
    return tuple(blockers)


def clear_caches() -> None:
    """Drop the parsed-source caches (tests repoint the sources at tmp files)."""

    _class_defs_cached.cache_clear()
    isaac_live_constants.clear_caches()


__all__ = [
    "ACTION_BALL_TERMINATIONS_CLASS",
    "BASE_TERMINATIONS_CLASS",
    "BODY_NAME_MODULE",
    "BODY_NAME_VOCABULARY_SYMBOLS",
    "DECLARED_TERMS",
    "DEPLOY_PARITY_TERMINATIONS_CLASS",
    "EXTERNAL_TERMINATION_BASES",
    "ISAAC_BASE_TERMINATION_CONFIG",
    "ISAAC_BODY_NAME_LISTS",
    "ISAAC_TERMINATION_CONFIG",
    "IsaacReferenceEnvelopeError",
    "REFERENCE_ENVELOPE_TERM",
    "REPO_ROOT",
    "clear_caches",
    "live_body_name_vocabulary",
    "live_chain_sources",
    "live_class_chain",
    "live_declared_term_blockers",
    "live_declared_terms",
    "live_reference_envelope",
    "live_reference_envelope_blockers",
    "live_term_owner_class",
    "live_termination_reason_order",
    "live_termination_reason_order_blockers",
    "live_timeout_term_names",
]
