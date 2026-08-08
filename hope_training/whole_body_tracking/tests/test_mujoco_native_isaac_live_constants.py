"""Hand-copied Isaac constants must equal the LIVE value, not just an old hash.

人话:``mujoco_native`` 里有一批常量的真源在 Isaac 侧,这里存的是副本。副本过去
只被"语义 AST 指纹"罩着 —— 指纹只说"源文件那几个节点的字节没动过"。源文件一动,
把指纹重钉成新值是一行的事,副本跟没跟上没有任何机制在看:5ed998f1(08-04)就是
这么让桌面终局的复刻停在原地两天的。

这个模块把那次事故当成模板反复重放:**先替那个粗心的作者把指纹重钉好**(所以
旧的哈希门此刻完全通过),再要求新的活值门照样拦下来。每条变异都特意做成
"粗一个档次的检查就抓不到":

* 拍面半轴只改厚度那一维 —— 长度、个数、对角结构全不变;
* 五段桌台只换 ``post_left``/``post_right`` 的顺序 —— 集合与长度一字不差;
* 参考包络只把脚和腕的**顺序**对调 —— 集合与长度一字不差;
* ``margin`` 改成指向**另一个模块常量** —— ``TABLE_HIT_MARGIN_M`` 自己没动,
  只比模块常量的检查会放行;
* ``margin_fraction`` 从 0.02 改成 0.05 —— 而 ``0.02`` 在同一个文件里还有别的
  出处,"这个数还在"式的检查会放行。

Scope note: this module owns the extractor (:mod:`mujoco_native.
isaac_live_constants`), the robot/table guard copies and the four native sibling
digests pinned in the N1 reward/event kernel.  The phase-fidelity lane's own
consumer tests live with that lane.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import isaac_live_constants as live  # noqa: E402
from mujoco_native import n1_reward_event_kernel as kernel  # noqa: E402
from mujoco_native import table_termination as term  # noqa: E402


ISAAC_CONFIG = term.ISAAC_TERMINATION_CONFIG
ISAAC_CALLABLES = term.ISAAC_TERMINATION_CALLABLES
ISAAC_BODY_NAMES = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/agibot_a3.py"
)
BODY_NAME_COMPANIONS = {"whole_body_tracking.robots.agibot_a3": ISAAC_BODY_NAMES}
DEPLOY_PARITY_CFG = "HOPEDeployParityTerminationsCfg"
ACTION_BALL_CFG = "HOPEActionBallTerminationsCfg"


@pytest.fixture(autouse=True)
def _clear_live_caches():
    """Parsed-source caches are keyed by path; mutated copies must not stick."""

    live.clear_caches()
    kernel._live_source_digest_blockers_cached.cache_clear()
    yield
    live.clear_caches()
    kernel._live_source_digest_blockers_cached.cache_clear()


def _mutated(tmp_path: Path, source: Path, old: str, new: str, name: str) -> Path:
    """Copy one live Isaac source with a single unique textual substitution."""

    text = source.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"mutation anchor is not unique: {old!r}"
    target = tmp_path / name
    target.write_text(text.replace(old, new), encoding="utf-8")
    return target


def _repin_table_config(monkeypatch, path: Path) -> None:
    """Do the careless author's re-pin for them: the SHA gate now passes."""

    monkeypatch.setattr(term, "ISAAC_TERMINATION_CONFIG", path)
    # Read the live selector list; a fourth hand-copy of it here is exactly the
    # drift this file exists to catch.
    repinned = term._semantic_ast_sha256(
        path,
        term.ISAAC_TERMINATION_CONFIG_SELECTORS,
        "repin",
    )
    monkeypatch.setattr(
        term, "EXPECTED_ISAAC_TERMINATION_CONFIG_SEMANTIC_AST_SHA256", repinned
    )


def _repin_table_callables(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(term, "ISAAC_TERMINATION_CALLABLES", path)
    repinned = term._semantic_ast_sha256(
        path, term.ISAAC_TERMINATION_CALLABLE_SELECTORS, "repin"
    )
    monkeypatch.setattr(
        term, "EXPECTED_ISAAC_TERMINATION_CALLABLES_SEMANTIC_AST_SHA256", repinned
    )


def _envelope_blockers(config: Path, mirrored) -> tuple:
    """Read one class's ``ee_body_pos`` body list straight out of a live cfg."""

    return live.parity_blockers(
        "probe",
        (
            (
                "ee_body_names",
                config,
                ("class_term_param", DEPLOY_PARITY_CFG, "ee_body_pos", "body_names"),
                mirrored,
            ),
        ),
        companions=BODY_NAME_COMPANIONS,
    )


# ---------------------------------------------------------------------------
# Non-vacuity floors
# ---------------------------------------------------------------------------


def test_every_mirrored_constant_currently_equals_its_live_isaac_value():
    assert term.live_isaac_constant_blockers() == ()
    assert kernel.live_source_digest_blockers() == ()


def test_the_registries_are_not_empty_and_name_no_duplicates():
    table_keys = [entry[0] for entry in term.mirrored_isaac_constant_entries()]
    digest_keys = [entry[0] for entry in kernel.mirrored_source_digest_entries()]
    assert len(table_keys) == len(set(table_keys)) == 7
    assert len(digest_keys) == len(set(digest_keys)) == 4


def test_a_verbatim_copy_at_another_path_is_clean(tmp_path, monkeypatch):
    """Floor: the extractor itself must not be what fails in the mutations."""

    copied = tmp_path / "hope_env_cfg.py"
    copied.write_text(ISAAC_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(term, "ISAAC_TERMINATION_CONFIG", copied)
    assert term.live_isaac_constant_blockers() == ()
    assert _envelope_blockers(
        copied,
        (
            "left_ankle_roll_Link",
            "right_ankle_roll_Link",
            "left_wrist_yaw_Link",
            "right_wrist_yaw_Link",
        ),
    ) == ()


# ---------------------------------------------------------------------------
# 5ed998f1 replays: mutate the live source, RE-PIN the fingerprint, still refuse
# ---------------------------------------------------------------------------


def test_repinned_margin_change_is_still_refused(tmp_path, monkeypatch):
    mutated = _mutated(
        tmp_path,
        ISAAC_CONFIG,
        "TABLE_HIT_MARGIN_M = 0.02",
        "TABLE_HIT_MARGIN_M = 0.03",
        "hope_env_cfg.py",
    )
    _repin_table_config(monkeypatch, mutated)
    # The hash gate is satisfied by construction; only the value gate can see it.
    assert term.live_isaac_constant_blockers()
    with pytest.raises(
        term.TableTerminationContractError, match="no longer equal the live"
    ):
        term.verify_isaac_source_authority()

    # And this is exactly the 5ed998f1 shape: switch the value gate off and the
    # re-pinned fingerprint alone waves the drifted source straight through.
    monkeypatch.setattr(term, "live_isaac_constant_blockers", lambda: ())
    assert term.verify_isaac_source_authority()["config_semantic_ast_sha256"] == (
        term.EXPECTED_ISAAC_TERMINATION_CONFIG_SEMANTIC_AST_SHA256
    )


def test_repinned_blade_thickness_change_is_still_refused(tmp_path, monkeypatch):
    """Only the middle half extent moves: length, count and diagonality survive."""

    mutated = _mutated(
        tmp_path,
        ISAAC_CONFIG,
        "TABLE_RACKET_BLADE_HALF_EXTENTS_M = (0.082, 0.008, 0.082)",
        "TABLE_RACKET_BLADE_HALF_EXTENTS_M = (0.082, 0.010, 0.082)",
        "hope_env_cfg.py",
    )
    _repin_table_config(monkeypatch, mutated)
    blockers = term.live_isaac_constant_blockers()
    assert any("racket_blade_half_extents_m" in item for item in blockers), blockers
    assert not any("center_offset" in item for item in blockers), blockers
    with pytest.raises(term.TableTerminationContractError):
        term.verify_isaac_source_authority()


def test_repinned_margin_rebound_to_another_constant_is_still_refused(
    tmp_path, monkeypatch
):
    """``TABLE_HIT_MARGIN_M`` itself is untouched -- only the term stops using it.

    A guard that compared the module constant would pass here.  This one reads
    what ``robot_hit_table`` is actually constructed with.
    """

    mutated = _mutated(
        tmp_path,
        ISAAC_CONFIG,
        '"margin": TABLE_HIT_MARGIN_M,',
        '"margin": TABLE_HIT_FORCE_THRESHOLD_N,',
        "hope_env_cfg.py",
    )
    assert "TABLE_HIT_MARGIN_M = 0.02" in mutated.read_text(encoding="utf-8")
    assert (
        live.live_value(mutated, ("assignment", "TABLE_HIT_MARGIN_M"))
        == term.TABLE_GUARD_MARGIN_M
    )
    _repin_table_config(monkeypatch, mutated)
    blockers = term.live_isaac_constant_blockers()
    assert any("table_guard_margin_m" in item for item in blockers), blockers


def test_repinned_assembly_role_reorder_is_still_refused(tmp_path, monkeypatch):
    """Same five names, same length -- only the order moves."""

    mutated = _mutated(
        tmp_path,
        ISAAC_CALLABLES,
        '    "net",\n    "post_left",\n    "post_right",\n)',
        '    "net",\n    "post_right",\n    "post_left",\n)',
        "terminations.py",
    )
    live_roles = live.live_value(mutated, ("assignment", "_TABLE_GUARD_OBSTACLE_ROLES"))
    assert set(live_roles) == set(term.TABLE_ASSEMBLY_ROLES)
    assert len(live_roles) == len(term.TABLE_ASSEMBLY_ROLES)
    _repin_table_callables(monkeypatch, mutated)
    blockers = term.live_isaac_constant_blockers()
    assert any("table_assembly_roles" in item for item in blockers), blockers
    with pytest.raises(term.TableTerminationContractError):
        term.verify_isaac_source_authority()


def test_repinned_proxy_artifact_repointing_is_still_refused(tmp_path, monkeypatch):
    """The proxy SHA is unchanged; only the path the term reads it from moves."""

    mutated = _mutated(
        tmp_path,
        ISAAC_CONFIG,
        '    "configs/a3_table_collision_proxy_a3p0807_20260808/"\n'
        '    "a3_table_collision_components.v1.json"',
        '    "configs/a3_table_collision_proxy_a3p0807_20260808/"\n'
        '    "a3_table_collision_components.v2.json"',
        "hope_env_cfg.py",
    )
    _repin_table_config(monkeypatch, mutated)
    blockers = term.live_isaac_constant_blockers()
    assert any(
        "collision_proxy_artifact_repo_relative_path" in item for item in blockers
    ), blockers
    assert not any(
        "collision_proxy_artifact_sha256" in item for item in blockers
    ), blockers


# ---------------------------------------------------------------------------
# The extractor folds exactly what it claims, on the real Isaac expressions
# ---------------------------------------------------------------------------


def test_an_imported_sequence_concatenation_is_folded_from_the_companion_leaf():
    """``A3_FEET_BODIES + A3_HAND_BODIES`` lives in another file entirely."""

    names = live.live_value(
        ISAAC_CONFIG,
        ("class_term_param", DEPLOY_PARITY_CFG, "ee_body_pos", "body_names"),
        companions=BODY_NAME_COMPANIONS,
    )
    assert names == (
        "left_ankle_roll_Link",
        "right_ankle_roll_Link",
        "left_wrist_yaw_Link",
        "right_wrist_yaw_Link",
    )
    narrowed = live.live_value(
        ISAAC_CONFIG,
        ("class_term_param", ACTION_BALL_CFG, "ee_body_pos", "body_names"),
        companions=BODY_NAME_COMPANIONS,
    )
    assert narrowed == ("left_ankle_roll_Link", "right_ankle_roll_Link")


def test_a_body_order_swap_is_refused_even_though_the_set_is_identical(tmp_path):
    """Same four bodies, same length -- feet and hands only trade places."""

    mutated = _mutated(
        tmp_path,
        ISAAC_CONFIG,
        '                "body_names": A3_FEET_BODIES + A3_HAND_BODIES, '
        '"ignore_hold": True},',
        '                "body_names": A3_HAND_BODIES + A3_FEET_BODIES, '
        '"ignore_hold": True},',
        "hope_env_cfg.py",
    )
    mirrored = (
        "left_ankle_roll_Link",
        "right_ankle_roll_Link",
        "left_wrist_yaw_Link",
        "right_wrist_yaw_Link",
    )
    live_names = live.live_value(
        mutated,
        ("class_term_param", DEPLOY_PARITY_CFG, "ee_body_pos", "body_names"),
        companions=BODY_NAME_COMPANIONS,
    )
    assert set(live_names) == set(mirrored)
    assert len(live_names) == len(mirrored)
    assert _envelope_blockers(mutated, mirrored)


def test_a_threshold_change_is_refused_even_though_the_number_recurs(tmp_path):
    """``0.02`` still occurs elsewhere in the file, so "the number exists" passes."""

    mutated = _mutated(
        tmp_path,
        ISAAC_CONFIG,
        '            "action_name": "joint_pos",\n'
        '            "limit_source": "joint_pos_limits",\n'
        '            "margin_rad": 0.0,\n            "margin_fraction": 0.02,\n',
        '            "action_name": "joint_pos",\n'
        '            "limit_source": "joint_pos_limits",\n'
        '            "margin_rad": 0.0,\n            "margin_fraction": 0.05,\n',
        "hope_env_cfg.py",
    )
    assert "TABLE_HIT_MARGIN_M = 0.02" in mutated.read_text(encoding="utf-8")
    blockers = live.parity_blockers(
        "probe",
        (
            (
                "margin_fraction",
                mutated,
                (
                    "class_term_param",
                    ACTION_BALL_CFG,
                    "joint_qdes_forbidden",
                    "margin_fraction",
                ),
                0.02,
            ),
            (
                "margin_rad",
                mutated,
                (
                    "class_term_param",
                    ACTION_BALL_CFG,
                    "joint_qdes_forbidden",
                    "margin_rad",
                ),
                0.0,
            ),
        ),
    )
    assert len(blockers) == 1
    assert "margin_fraction" in blockers[0]


# ---------------------------------------------------------------------------
# The extractor refuses rather than guesses
# ---------------------------------------------------------------------------


def test_a_runtime_value_is_reported_as_unreadable_not_as_a_match(
    tmp_path, monkeypatch
):
    mutated = _mutated(
        tmp_path,
        ISAAC_CONFIG,
        "TABLE_HIT_MARGIN_M = 0.02",
        'TABLE_HIT_MARGIN_M = float(os.environ.get("HOPE_MARGIN", "0.02"))',
        "hope_env_cfg.py",
    )
    _repin_table_config(monkeypatch, mutated)
    blockers = term.live_isaac_constant_blockers()
    assert any(
        item.startswith("table_guard_live_value_unreadable:table_guard_margin_m")
        for item in blockers
    ), blockers


def test_a_rebound_module_constant_has_no_single_live_value(tmp_path):
    mutated = _mutated(
        tmp_path,
        ISAAC_CALLABLES,
        '_TABLE_GUARD_OBSTACLE_ROLES = (\n    "top",',
        '_TABLE_GUARD_OBSTACLE_ROLES = ("decoy",)\n_TABLE_GUARD_OBSTACLE_ROLES = (\n'
        '    "top",',
        "terminations.py",
    )
    with pytest.raises(live.IsaacLiveConstantError, match="absent or rebound"):
        live.live_value(mutated, ("assignment", "_TABLE_GUARD_OBSTACLE_ROLES"))


def test_an_unregistered_companion_module_is_refused(tmp_path):
    copied = tmp_path / "hope_env_cfg.py"
    copied.write_text(ISAAC_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(live.IsaacLiveConstantError, match="unregistered module"):
        live.live_value(
            copied,
            ("class_term_param", DEPLOY_PARITY_CFG, "ee_body_pos", "body_names"),
        )


def test_absent_class_attribute_or_param_is_refused(tmp_path):
    copied = tmp_path / "hope_env_cfg.py"
    copied.write_text(ISAAC_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(live.IsaacLiveConstantError, match="not unique"):
        live.live_value(
            copied,
            ("class_term_param", DEPLOY_PARITY_CFG, "anchor_pos", "no_such_param"),
        )
    with pytest.raises(live.IsaacLiveConstantError, match="not unique"):
        live.live_value(
            copied,
            ("class_term_param", DEPLOY_PARITY_CFG, "no_such_term", "threshold"),
        )
    with pytest.raises(live.IsaacLiveConstantError, match="unsupported live selector"):
        live.live_value(copied, ("no_such_kind", "anything"))


def test_bool_and_int_are_not_interchangeable(tmp_path):
    source = tmp_path / "toy.py"
    source.write_text("FLAG = 1\n", encoding="utf-8")
    entry = ("flag", source, ("assignment", "FLAG"))
    assert live.parity_blockers("toy", ((*entry, 1),)) == ()
    assert live.parity_blockers("toy", ((*entry, True),))


# ---------------------------------------------------------------------------
# The pinned native sibling digests
# ---------------------------------------------------------------------------


def test_native_source_digest_drift_is_visible_on_the_host(tmp_path):
    drifted = tmp_path / "n1_ball_core.py"
    drifted.write_text("# not the real core\n", encoding="utf-8")
    blockers = kernel.source_digest_blockers(
        (
            (
                "n1_ball_core_source_sha256",
                drifted,
                kernel.EXPECTED_N1_BALL_CORE_SOURCE_SHA256,
            ),
        )
    )
    assert len(blockers) == 1
    assert blockers[0].startswith(
        "native_source_digest_differs:n1_ball_core_source_sha256"
    )
    missing = kernel.source_digest_blockers(
        (("gone", tmp_path / "absent.py", "0" * 64),)
    )
    assert missing and missing[0].startswith("native_source_unreadable:gone")


def test_the_event_facts_contract_refuses_a_stale_native_pin(monkeypatch):
    monkeypatch.setattr(kernel, "EXPECTED_N1_BALL_CORE_SOURCE_SHA256", "0" * 64)
    kernel._live_source_digest_blockers_cached.cache_clear()
    with pytest.raises(
        kernel.N1RewardEventKernelError, match="no longer match the live files"
    ):
        kernel.native_physical_event_facts_contract()


def test_pinned_native_sibling_digests_are_the_files_on_disk():
    for key, path, expected in kernel.mirrored_source_digest_entries():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, key


# ---------------------------------------------------------------------------
# The receipt says which comparison actually ran
# ---------------------------------------------------------------------------


def test_table_guard_receipt_self_declares_the_live_constant_parity():
    receipt = term.verify_isaac_source_authority()
    assert receipt["live_constant_parity"] == (
        "margin_racket_body_blade_box_proxy_path_sha_assembly_roles"
    )
    assert receipt["live_constant_parity_constants_compared"] == "7"


def test_the_table_guard_gate_fires_on_a_synthetic_blocker(monkeypatch):
    monkeypatch.setattr(
        term, "live_isaac_constant_blockers", lambda: ("synthetic_table_blocker",)
    )
    with pytest.raises(term.TableTerminationContractError, match="synthetic_table"):
        term.verify_isaac_source_authority()


# ---------------------------------------------------------------------------
# C211 奖励权重:镜像 vs Isaac 现役值(2026-08-08 新增)
#
# 这一组的由来是一次真事故,不是假想:2026-08-04 Isaac 把 ``upright_exp`` 从 1.0
# 重定价到 0.25(`_REWARD_PACK_V2_DIRECT`),``action_ball_c211_env`` 那份手抄权重
# 四天没跟上,而所有的门都是绿的 —— 因为当时唯一"看着"这件事的东西是收据里的
# ``reward_pack_resolver_source_sha256 = sha256(train.py)``:整文件指纹,天天变,
# 没有任何消费者读它,也没有任何门会因为它拒收。
#
# 每条变异都做成"粗一个档次的检查照样通过",并且**在同一条测试里先把那个粗检查
# 断言成过得去**,再要求活值门变红。
# ---------------------------------------------------------------------------

from mujoco_native import action_ball_c211_env as c211  # noqa: E402


HOPE_ENV_CFG = c211.HOPE_ENV_CFG_PY
TRAIN_PY = c211.TRAIN_PY


def _repoint_cfg(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(c211, "HOPE_ENV_CFG_PY", path)


def _repoint_train(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(c211, "TRAIN_PY", path)


def test_c211_reward_weight_mirror_is_green_on_the_real_sources():
    assert c211.live_isaac_reward_weight_blockers() == ()


def test_the_pack_is_the_authority_not_the_cfg_class_body():
    """`upright_exp` 的类体是 0.0,现役是 0.25 —— 读错一层就会得出相反结论。

    这条把"为什么要读 reward_pack 那一层"钉死:如果哪天有人把选择器改回
    ``class_term_weight``,它读到的是 0.0,而镜像收的是 0.25,本条立刻红。
    """

    class_body = live.live_value(
        HOPE_ENV_CFG, ("class_term_weight", "HOPEDeployParityRewardsCfg", "upright_exp")
    )
    launch_resolved = live.live_value(
        TRAIN_PY, ("pair_table_value", c211.ISAAC_REWARD_PACK_TABLE, "upright_exp")
    )
    assert class_body == 0.0
    assert launch_resolved == 0.25
    assert c211._MIRRORED_PRIOR_WEIGHTS["upright_exp"] == launch_resolved


def test_isaac_repricing_upright_exp_again_turns_the_mirror_red(tmp_path, monkeypatch):
    """重放 08-04 那次事故:Isaac 动了,镜像没动。

    粗检查(项数不变、项名不变、0.25 这个字面量在 train.py 里还有别的出处)全部
    照样通过 —— 这三条都在下面断言过。
    """

    mutated = _mutated(
        tmp_path,
        TRAIN_PY,
        '("upright_exp", 0.25),',
        '("upright_exp", 1.0),',
        "train_upright.py",
    )
    text = mutated.read_text(encoding="utf-8")
    assert text.count('("action_rate_l2", -0.1),') == 1     # 其余行没动
    assert "0.25" in text                                   # 这个数在文件里还在
    _repoint_train(monkeypatch, mutated)

    blockers = c211.live_isaac_reward_weight_blockers()
    assert any("upright_exp" in item and "live=1.0" in item for item in blockers), (
        blockers
    )


def test_swapping_two_weights_between_terms_turns_the_mirror_red(tmp_path, monkeypatch):
    """把 base_ang_vel_xy 和 base_lin_vel_z 的权重对调。

    和不变(-0.55)、排序后的多重集不变、项数不变 —— 三条粗检查在下面逐条断言过,
    它们全过。只有逐项比值抓得住。
    """

    text = HOPE_ENV_CFG.read_text(encoding="utf-8")
    # 锚点必须带上行尾注释才唯一:同样这两行在 HOPEHitterPureRewardsCfg 里还有一份
    # 逐字相同的拷贝(那个类不在 C211 链上)。这本身就是影子检查存在的理由。
    ang = (
        "base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)"
        "  # roll/pitch rate"
    )
    lin = (
        "base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)"
        "  # vertical bob"
    )
    assert text.count(ang) == 1 and text.count(lin) == 1
    swapped = text.replace(ang, "@@ANG@@").replace(lin, "@@LIN@@")
    swapped = swapped.replace(
        "@@ANG@@", "base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)"
    ).replace(
        "@@LIN@@", "base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)"
    )
    mutated = tmp_path / "hope_env_cfg_swapped.py"
    mutated.write_text(swapped, encoding="utf-8")
    _repoint_cfg(monkeypatch, mutated)

    live_now = {
        term_name: live.live_value(
            mutated, ("class_term_weight", class_name, term_name)
        )
        for term_name, class_name in c211.ISAAC_CLASS_SOURCED_PRIOR_WEIGHTS
    }
    mirrored = {
        term_name: c211._MIRRORED_PRIOR_WEIGHTS[term_name]
        for term_name, _c in c211.ISAAC_CLASS_SOURCED_PRIOR_WEIGHTS
    }
    # 粗检查全过:
    assert len(live_now) == len(mirrored)
    assert sum(live_now.values()) == pytest.approx(sum(mirrored.values()))
    assert sorted(live_now.values()) == pytest.approx(sorted(mirrored.values()))

    blockers = c211.live_isaac_reward_weight_blockers()
    assert any("base_ang_vel_xy" in item for item in blockers), blockers
    assert any("base_lin_vel_z" in item for item in blockers), blockers


def test_a_downstream_class_shadowing_the_term_turns_the_mirror_red(
    tmp_path, monkeypatch
):
    """C211 类里新增一条同名 RewTerm,**值故意写成和现在一模一样**。

    所以逐项比值这一层完全过得去(下面断言过:选择器读到的还是 -1.0e-4)。
    抓住它的只有影子检查 —— 没有它,以后任何人在下游改这个权重,门都会绿着放行。
    """

    anchor = "class HOPEActionBallC211RewardsCfg(HOPEActionBallRewardsCfg):"
    mutated = _mutated(
        tmp_path,
        HOPE_ENV_CFG,
        anchor,
        anchor
        + "\n    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)  # shadow",
        "hope_env_cfg_shadow.py",
    )
    _repoint_cfg(monkeypatch, mutated)

    # 值这一层过得去:选择器指的那个类没动。
    assert live.live_value(
        mutated, ("class_term_weight", "HOPEDeployParityRewardsCfg", "joint_vel")
    ) == pytest.approx(-1.0e-4)
    assert live.parity_blockers(
        "probe", c211.mirrored_isaac_reward_weight_entries()
    ) == ()

    blockers = c211.live_isaac_reward_weight_blockers()
    assert any(
        item.startswith("c211_reward_weight_shadowed:joint_vel") for item in blockers
    ), blockers


def test_a_deleted_pack_row_refuses_instead_of_falling_back(tmp_path, monkeypatch):
    """把 ``action_rate_l2`` 那一行从包里删掉。

    这条是陷阱题:``action_rate_l2`` 的 cfg 类体权重**也是 -0.1**
    (``tracking_env_cfg.RewardsCfg``)。任何"包里没有就退回类体"的读法都会宣布
    "对齐",而事实是这一项已经不再由包定价了。必须拒绝,不许猜。
    """

    mutated = _mutated(
        tmp_path,
        TRAIN_PY,
        '    ("action_rate_l2", -0.1),\n',
        "",
        "train_no_action_rate.py",
    )
    _repoint_train(monkeypatch, mutated)

    blockers = c211.live_isaac_reward_weight_blockers()
    assert any(
        "action_rate_l2" in item and "unreadable" in item for item in blockers
    ), blockers


def test_a_row_the_reader_cannot_parse_refuses_instead_of_skipping(
    tmp_path, monkeypatch
):
    """包里混进一行不是 ``(name, value)`` 的东西 -> 拒绝,不是跳过。

    跳过是最坏的失败模式:被跳过的那一行完全可能就是要找的那一行。
    """

    mutated = _mutated(
        tmp_path,
        TRAIN_PY,
        '    ("upright_exp", 0.25),',
        '    ("upright_exp", 0.25, "note"),',
        "train_bad_row.py",
    )
    _repoint_train(monkeypatch, mutated)

    blockers = c211.live_isaac_reward_weight_blockers()
    assert any("upright_exp" in item and "unreadable" in item for item in blockers), (
        blockers
    )


def test_the_c211_reward_contract_refuses_on_a_synthetic_weight_blocker(monkeypatch):
    """记录与阻断同一批:门必须真的拦,不能只往收据里写一行。"""

    monkeypatch.setattr(
        c211, "live_isaac_reward_weight_blockers", lambda: ("synthetic_weight_drift",)
    )
    env = object.__new__(c211.MujocoC211DiagnosticVecEnv)
    with pytest.raises(c211.C211EnvError, match="synthetic_weight_drift"):
        env._build_reward_contract(None)


# ---------------------------------------------------------------------------
# A211 侧那两条:同一个机制,同样要能变红
# ---------------------------------------------------------------------------

from mujoco_native import action_ball_a211_env as a211  # noqa: E402


def test_a211_reward_weight_mirror_is_green_on_the_real_sources():
    assert a211.live_isaac_reward_weight_blockers() == ()


def test_isaac_repricing_racket_progress_turns_the_a211_mirror_red(
    tmp_path, monkeypatch
):
    """`racket_progress` 是 A 族"够球"那条腿的价钱,改了没人看就是又一次 upright_exp。

    锚点带上行尾的 params 才唯一:HOPEHitterPureRewardsCfg 里有一条同名的 weight=0.0
    (那个类不在 C211/A211 的链上)—— 又一个影子检查存在的理由。
    """

    old = (
        "racket_progress = RewTerm(func=mdp.racket_progress, weight=10.0, "
        'params={"command_name": "racket_target"})'
    )
    mutated = _mutated(
        tmp_path,
        HOPE_ENV_CFG,
        old,
        old.replace("weight=10.0", "weight=2.5"),
        "hope_env_cfg_progress.py",
    )
    monkeypatch.setattr(c211, "HOPE_ENV_CFG_PY", mutated)

    blockers = a211.live_isaac_reward_weight_blockers()
    assert any(
        "racket_progress" in item and "live=2.5" in item for item in blockers
    ), blockers


def test_a211_shadowing_racket_progress_is_caught_even_at_the_same_value(
    tmp_path, monkeypatch
):
    anchor = "class HOPEActionBallC211RewardsCfg(HOPEActionBallRewardsCfg):"
    mutated = _mutated(
        tmp_path,
        HOPE_ENV_CFG,
        anchor,
        anchor
        + "\n    racket_progress = RewTerm(func=mdp.racket_progress, weight=10.0,"
        ' params={"command_name": "racket_target"})  # shadow',
        "hope_env_cfg_progress_shadow.py",
    )
    monkeypatch.setattr(c211, "HOPE_ENV_CFG_PY", mutated)

    # 值这一层完全过得去:
    assert live.parity_blockers(
        "probe", a211.mirrored_isaac_reward_weight_entries()
    ) == ()

    blockers = a211.live_isaac_reward_weight_blockers()
    assert any(
        item.startswith("a211_reward_weight_shadowed:racket_progress")
        for item in blockers
    ), blockers


def test_every_implemented_c211_prior_term_is_either_compared_or_declared():
    """14 条实现项不许有第三种状态:没人比、也没人记。

    把一条从"没比"名单里删掉 —— 名单还是合法的 dict、其余 8 条一字不动,任何
    "名单格式对不对"式的检查都会放行。只有覆盖面自检抓得住。
    """

    assert c211.live_isaac_reward_weight_blockers() == ()
    trimmed = tuple(
        row
        for row in c211.C211_REWARD_WEIGHT_PARITY_NOT_LIVE_COMPARED
        if row["term"] != "motion_racket_normal"
    )
    assert len(trimmed) == len(c211.C211_REWARD_WEIGHT_PARITY_NOT_LIVE_COMPARED) - 1
    assert all(set(row) == {"term", "resolution", "blocked_on"} for row in trimmed)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c211, "C211_REWARD_WEIGHT_PARITY_NOT_LIVE_COMPARED", trimmed)
        blockers = c211.live_isaac_reward_weight_blockers()
    assert any(
        "neither_compared_nor_declared_uncompared" in item
        and "motion_racket_normal" in item
        for item in blockers
    ), blockers
