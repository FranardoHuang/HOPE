"""DR 档位发射入口的验收 —— 选 L1 真的换了东西,选 L0 一个字节都没动。

人话
====
2026-08-08 之前,"这一跑用哪一档随机性"是两个发射器里各写死的一行常量,都指向
DR-L0。DR-L1 那两片 profile(五条 plant 轴 + ``start_pose_ramp``,六条随机性轴)
**选不中**。本文件是这个入口的机器验收,分四组:

1. **DR-L0 逐字不变** —— 四格归因跑的解析结果必须与硬钉常量时代逐字相同。
   下面那些字面量是**改动之前**发射器里的原文,不是从今天的代码抄回来的。
2. **选 DR-L1 真的换值** —— 六条轴各自的实际取值必须变成候选 config 里写明的
   day-1 基线;并且有一条**反退化**用例:把两档喂成同值时,那条"两档必须不同"
   的断言必须炸。没有它,"L1 生效了"这句话会变成恒真。
3. **两个出处对不上就拒** —— 逐轴把候选 manifest 的声明值改一个数,解析必须拒收,
   而且错误信息要点名是哪条轴。这就是"比活值不比手抄"。
4. **收据谎报必拒** —— 包括**重新封印过**的谎报(改完值把 seal 也算对)。
   只查 seal 的检查挡不住这种,所以必须逐字段回比重新解析的结果。
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training/whole_body_tracking/scripts"
TRAINING_CONTRACT = (
    REPO
    / "hope_training/whole_body_tracking/source/whole_body_tracking"
    / "whole_body_tracking/utils/training_contract.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load("dr_launch_levels_probe", SCRIPTS / "action_ball_211_dr_launch_levels.py")
A = _load(
    "a211_dr_level_probe",
    SCRIPTS / "launch_action_ball_a211_four_arm_diagnostic.py",
)
C = _load("c211_dr_level_probe", SCRIPTS / "launch_action_ball_c211_diagnostic.py")
sys.path.insert(
    0, str(REPO / "hope_training/whole_body_tracking/source/whole_body_tracking")
)
TC = _load("dr_level_training_contract_probe", TRAINING_CONTRACT)


# --------------------------------------------------------------------------- #
# 改动之前发射器里的原文。**不要**用 D.task_profile_id(...) 生成这张表 —— 那样
# 这组用例会退化成 "解析结果等于它自己"。
# --------------------------------------------------------------------------- #
FROZEN_DR_L0 = {
    ("A211", "task_profile"): (
        "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability"
    ),
    ("C211", "task_profile"): (
        "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability"
    ),
    ("A211", "task_profile_source"): (
        "hope_training/whole_body_tracking/cfg/task/"
        "HOPEPingPongActionBallA211VendorV2N1DRL0Learnability.yaml"
    ),
    ("C211", "task_profile_source"): (
        "hope_training/whole_body_tracking/cfg/task/"
        "HOPEPingPongActionBallC211VendorV2N1DRL0Learnability.yaml"
    ),
    ("A211", "lineage_kind"): (
        "action_ball_a211_split_ready_online_question_dr_l0_lineage_v5"
    ),
    ("C211", "lineage_kind"): "action_ball_c211_direct_ball_split_ready_lineage_v4",
}
FROZEN_DR_L0_MANIFEST = (
    "configs/action_ball_n1_measured_20260803/"
    "action_ball_211_dr_l0_learnability_candidate.v1.json"
)
FROZEN_DR_L0_IDENTITY = "action_ball_dr_l0_exact_all_off_v1"


def _manifest(level: str) -> dict:
    return json.loads((REPO / D.manifest_source(level)).read_text(encoding="utf-8"))


def _resolved(level: str, family: str = "A211") -> dict:
    ramp = D.default_start_pose_ramp(TC, level)
    payload = D.contract_payload(TC, level, start_pose_ramp=ramp)
    return D.resolve(
        level,
        family=family,
        contract_payload_document=payload,
        manifest_document=_manifest(level),
        contract_sha256=D.contract_sha256(TC, level, start_pose_ramp=ramp),
        manifest_file_sha256="a" * 64,
    )


@pytest.fixture(scope="module")
def resolved_by_level() -> dict:
    return {level: _resolved(level) for level in D.LEVELS}


# --------------------------------------------------------------------------- #
# 1. DR-L0 逐字不变
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ("A211", "C211"))
def test_dr_l0_resolution_is_byte_identical_to_the_pre_entrance_literals(family):
    assert D.task_profile_id(family, "dr_l0") == FROZEN_DR_L0[
        (family, "task_profile")
    ]
    assert D.task_profile_source(family, "dr_l0") == FROZEN_DR_L0[
        (family, "task_profile_source")
    ]
    assert D.lineage_kind(family, "dr_l0") == FROZEN_DR_L0[(family, "lineage_kind")]


def test_dr_l0_manifest_and_identity_are_byte_identical():
    assert D.manifest_source("dr_l0") == FROZEN_DR_L0_MANIFEST
    assert D.hard_contract_identity("dr_l0") == FROZEN_DR_L0_IDENTITY


def test_both_launchers_still_expose_the_same_dr_l0_constants():
    """入口接线不许顺手改掉两个发射器对外的那几个名字的取值。"""

    assert A.TASK_PROFILE_ID == FROZEN_DR_L0[("A211", "task_profile")]
    assert C.TASK_PROFILE_ID == FROZEN_DR_L0[("C211", "task_profile")]
    assert A.TASK_PROFILE_SOURCE == FROZEN_DR_L0[("A211", "task_profile_source")]
    assert C.TASK_PROFILE_SOURCE == FROZEN_DR_L0[("C211", "task_profile_source")]
    assert A.LINEAGE_KIND == FROZEN_DR_L0[("A211", "lineage_kind")]
    assert C.LINEAGE_KIND == FROZEN_DR_L0[("C211", "lineage_kind")]
    assert A.DR_L0_MANIFEST_SOURCE == FROZEN_DR_L0_MANIFEST


def test_the_default_level_is_dr_l0_on_both_launchers():
    """默认值一变就是一次静默的实验改动:四格会在没人按旗标时换档。"""

    assert D.DEFAULT_LEVEL == "dr_l0"
    for module in (A, C):
        parser = module._parser()
        subparsers = next(
            action
            for action in parser._actions
            if action.__class__.__name__ == "_SubParsersAction"
        )
        template = subparsers.choices["template"]
        option = next(
            action for action in template._actions if action.dest == "dr_level"
        )
        assert option.default == "dr_l0"
        assert tuple(option.choices) == ("dr_l0", "dr_l1")


@pytest.mark.parametrize("module", (A, C), ids=("A211", "C211"))
def test_an_unknown_level_is_refused_at_the_parser_and_at_the_lineage_gate(
    module, tmp_path
):
    parser = module._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["template", "--dr-level", "dr_l9"])
    # argparse 只挡命令行;手写一份 spec 绕过它时,发射器自己那道门必须接住,
    # 而且要翻译成发射器的拒收类型,不能把权威模块的异常直接漏出去。
    with pytest.raises(module.LaunchRefused):
        module._validate_lineage(
            tmp_path / "no-such-checkout", "a" * 40, {}, level="dr_l9"
        )


def test_dr_l0_activates_no_axis_at_all(resolved_by_level):
    low = resolved_by_level["dr_l0"]
    assert low["active_axis_count"] == 0
    assert all(low["axes"][name] is None for name in D.AXIS_ORDER)


# --------------------------------------------------------------------------- #
# 2. 选 DR-L1 真的换值
# --------------------------------------------------------------------------- #
def test_dr_l1_resolves_to_its_own_profile_manifest_and_contract(resolved_by_level):
    high = resolved_by_level["dr_l1"]
    assert high["task_profile"] == (
        "HOPEPingPongActionBallA211VendorV2N1DRL1Learnability"
    )
    assert high["task_profile_source"].endswith("DRL1Learnability.yaml")
    assert (REPO / high["task_profile_source"]).is_file()
    assert high["hard_contract_identity"] == (
        "action_ball_dr_l1_measured_plant_restored_v1"
    )
    assert high["candidate_manifest_source"] == (
        "configs/action_ball_n1_measured_20260805/"
        "action_ball_211_dr_l1_restored_plant_candidate.v1.json"
    )
    assert high["lineage_kind"] == (
        "action_ball_a211_split_ready_online_question_dr_l1_lineage_v1"
    )


def test_dr_l1_carries_the_day_one_value_of_every_one_of_the_six_axes(
    resolved_by_level,
):
    """收据里必须是**取值**,不是档名。这里逐条对候选 config 里的数字。"""

    axes = resolved_by_level["dr_l1"]["axes"]
    assert resolved_by_level["dr_l1"]["active_axis_count"] == 6

    friction = axes["physics_material"]
    assert friction["static_friction_range"] == [0.3, 1.6]
    assert friction["dynamic_friction_range"] == [0.3, 1.2]
    assert friction["restitution_range"] == [0.0, 0.5]
    assert friction["num_buckets"] == 64

    offset = axes["add_joint_default_pos"]
    assert offset["pos_distribution_params"] == [-0.01, 0.01]
    assert offset["operation"] == "add"

    com = axes["base_com"]
    assert com["body_names"] == "torso_link"
    assert com["com_range"] == {
        "x": [-0.025, 0.025],
        "y": [-0.05, 0.05],
        "z": [-0.05, 0.05],
    }

    mass = axes["randomize_link_mass"]
    assert mass["mass_distribution_params"] == [0.85, 1.15]
    assert mass["operation"] == "scale"

    gains = axes["randomize_pd_gains"]
    assert gains["stiffness_distribution_params"] == [0.8, 1.2]
    assert gains["damping_distribution_params"] == [0.7, 1.3]
    assert gains["distribution"] == "log_uniform"

    ramp = axes["start_pose_ramp"]
    assert ramp["enabled"] is True
    assert ramp["ramp_steps"] == 96000
    assert ramp["pose_range"]["x"] == [-1.0, 0.0]
    assert ramp["pose_range"]["y"] == [-1.2625, 1.2625]
    assert ramp["pose_range"]["yaw"] == [
        -0.5235987755982988,
        0.5235987755982988,
    ]
    assert ramp["hold_clock_owner"] == "action_ball_task_receipt"


def test_the_two_levels_differ_on_every_axis(resolved_by_level):
    D.assert_levels_differ(resolved_by_level)


def test_feeding_both_levels_the_same_block_makes_the_distinctness_check_fail(
    resolved_by_level,
):
    """反退化:夹具把两档喂成同值时,那条断言必须炸而不是默默通过。

    这条用例的存在理由就是"粗一个档次的检查就过不了" —— 如果 assert_levels_differ
    只比档名,下面第一个 case 会溜过去;如果它只比 seal,第二个会溜过去。
    """

    same = {level: copy.deepcopy(resolved_by_level["dr_l1"]) for level in D.LEVELS}
    with pytest.raises(D.DrLaunchLevelError):
        D.assert_levels_differ(same)

    # 只把档名和身份改回 L0、六条轴仍然是 L1 的值 —— 这正是"档名对了但值没换"。
    faked = {
        "dr_l0": copy.deepcopy(resolved_by_level["dr_l1"]),
        "dr_l1": copy.deepcopy(resolved_by_level["dr_l1"]),
    }
    low = faked["dr_l0"]
    low["dr_launch_level"] = "dr_l0"
    low["task_profile"] = FROZEN_DR_L0[("A211", "task_profile")]
    low["task_profile_source"] = FROZEN_DR_L0[("A211", "task_profile_source")]
    low["lineage_kind"] = FROZEN_DR_L0[("A211", "lineage_kind")]
    low["hard_contract_identity"] = FROZEN_DR_L0_IDENTITY
    low["contract_sha256"] = "0" * 64
    low["candidate_manifest_source"] = FROZEN_DR_L0_MANIFEST
    low["content_sha256"] = D.canonical_sha256(
        {k: v for k, v in low.items() if k != "content_sha256"}
    )
    with pytest.raises(D.DrLaunchLevelError, match="vacuously true"):
        D.assert_levels_differ(faked)


def test_the_authority_does_not_hand_copy_any_axis_number():
    """六条轴的数字一个都不许出现在这份权威里 —— 出现即第四份手抄副本。"""

    source = (SCRIPTS / "action_ball_211_dr_launch_levels.py").read_text(
        encoding="utf-8"
    )
    for literal in (
        "0.85",
        "1.15",
        "0.8,",
        "1.2,",
        "0.7,",
        "1.3,",
        "0.025",
        "96000",
        "1.2625",
        "0.5235987755982988",
        "1.6",
    ):
        assert literal not in source, literal


# --------------------------------------------------------------------------- #
# 3. 两个出处对不上就拒(逐轴)
# --------------------------------------------------------------------------- #
_AXIS_MUTATIONS = {
    "physics_material": ("static_friction_range", [0.2, 1.8]),
    "add_joint_default_pos": ("pos_distribution_params", [-0.02, 0.02]),
    "base_com": ("body", "pelvis_link"),
    "randomize_link_mass": ("mass_distribution_params", [0.8, 1.2]),
    "randomize_pd_gains": ("damping_distribution_params", [0.9, 1.1]),
}


@pytest.mark.parametrize("axis", sorted(_AXIS_MUTATIONS))
def test_one_changed_number_in_the_candidate_manifest_is_refused(axis):
    key, value = _AXIS_MUTATIONS[axis]
    manifest = _manifest("dr_l1")
    assert manifest["restored_axes"][axis][key] != value
    manifest["restored_axes"][axis][key] = value
    ramp = D.default_start_pose_ramp(TC, "dr_l1")
    with pytest.raises(D.DrLaunchLevelError, match=axis):
        D.resolve(
            "dr_l1",
            family="A211",
            contract_payload_document=D.contract_payload(
                TC, "dr_l1", start_pose_ramp=ramp
            ),
            manifest_document=manifest,
            contract_sha256=D.contract_sha256(TC, "dr_l1", start_pose_ramp=ramp),
            manifest_file_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "path,value",
    (
        (("start_pose_ramp", "ramp_steps"), 48000),
        (("start_pose_ramp", "hold_clock_owner"), "motion_hold_steps"),
        (("start_pose_ramp", "payload_sha256"), "b" * 64),
    ),
)
def test_a_drifted_start_pose_ramp_declaration_is_refused(path, value):
    manifest = _manifest("dr_l1")
    node = manifest
    for key in path[:-1]:
        node = node[key]
    assert node[path[-1]] != value
    node[path[-1]] = value
    ramp = D.default_start_pose_ramp(TC, "dr_l1")
    with pytest.raises(D.DrLaunchLevelError, match="start_pose_ramp"):
        D.resolve(
            "dr_l1",
            family="A211",
            contract_payload_document=D.contract_payload(
                TC, "dr_l1", start_pose_ramp=ramp
            ),
            manifest_document=manifest,
            contract_sha256=D.contract_sha256(TC, "dr_l1", start_pose_ramp=ramp),
            manifest_file_sha256="a" * 64,
        )


def test_a_dr_l0_manifest_that_secretly_activates_an_event_is_refused():
    manifest = _manifest("dr_l0")
    manifest["required_post_finalizer_state"]["events.randomize_pd_gains"] = (
        D.L1_ACTIVE_MARKER
    )
    with pytest.raises(D.DrLaunchLevelError, match="randomize_pd_gains"):
        D.resolve(
            "dr_l0",
            family="A211",
            contract_payload_document=D.contract_payload(TC, "dr_l0"),
            manifest_document=manifest,
            contract_sha256=D.contract_sha256(TC, "dr_l0"),
            manifest_file_sha256="a" * 64,
        )


def test_a_manifest_from_the_other_level_is_refused():
    """把 L0 的 manifest 喂给 L1(或反过来)必须拒,不能靠"看起来像"过关。"""

    for level, other in (("dr_l0", "dr_l1"), ("dr_l1", "dr_l0")):
        ramp = D.default_start_pose_ramp(TC, level)
        with pytest.raises(D.DrLaunchLevelError):
            D.resolve(
                level,
                family="A211",
                contract_payload_document=D.contract_payload(
                    TC, level, start_pose_ramp=ramp
                ),
                manifest_document=_manifest(other),
                contract_sha256=D.contract_sha256(
                    TC, level, start_pose_ramp=ramp
                ),
                manifest_file_sha256="a" * 64,
            )


def test_a_payload_from_the_other_level_is_refused():
    ramp = D.default_start_pose_ramp(TC, "dr_l1")
    with pytest.raises(D.DrLaunchLevelError):
        D.resolve(
            "dr_l1",
            family="A211",
            contract_payload_document=D.contract_payload(TC, "dr_l0"),
            manifest_document=_manifest("dr_l1"),
            contract_sha256=D.contract_sha256(TC, "dr_l1", start_pose_ramp=ramp),
            manifest_file_sha256="a" * 64,
        )


# --------------------------------------------------------------------------- #
# 4. 收据谎报必拒
# --------------------------------------------------------------------------- #
def test_an_honest_receipt_block_is_accepted(resolved_by_level):
    for level in D.LEVELS:
        block = resolved_by_level[level]
        assert D.validate_declared(copy.deepcopy(block), resolved=block) == block


def test_a_receipt_that_renames_the_level_is_refused(resolved_by_level):
    lying = copy.deepcopy(resolved_by_level["dr_l1"])
    lying["dr_launch_level"] = "dr_l0"
    with pytest.raises(D.DrLaunchLevelError):
        D.validate_declared(lying, resolved=resolved_by_level["dr_l1"])


def test_a_resealed_lie_is_still_refused(resolved_by_level):
    """改完值把 seal 也算对 —— 只查 seal 的检查在这里会放行,所以必须逐字段回比。"""

    lying = copy.deepcopy(resolved_by_level["dr_l1"])
    lying["dr_launch_level"] = "dr_l0"
    lying["axes"]["randomize_link_mass"]["mass_distribution_params"] = [1.0, 1.0]
    lying["content_sha256"] = D.canonical_sha256(
        {key: value for key, value in lying.items() if key != "content_sha256"}
    )
    with pytest.raises(D.DrLaunchLevelError, match="differs from the re-resolved"):
        D.validate_declared(lying, resolved=resolved_by_level["dr_l1"])


def test_a_receipt_that_reports_only_the_level_name_is_refused(resolved_by_level):
    """只写档名、不写取值的收据无法复算,必须当场拒。"""

    thin = {"dr_launch_level": "dr_l1"}
    thin["content_sha256"] = D.canonical_sha256(dict(thin))
    with pytest.raises(D.DrLaunchLevelError, match="keys differ"):
        D.validate_declared(thin, resolved=resolved_by_level["dr_l1"])


def test_a_receipt_with_blanked_axes_is_refused(resolved_by_level):
    blanked = copy.deepcopy(resolved_by_level["dr_l1"])
    blanked["axes"] = {name: None for name in D.AXIS_ORDER}
    blanked["content_sha256"] = D.canonical_sha256(
        {key: value for key, value in blanked.items() if key != "content_sha256"}
    )
    with pytest.raises(D.DrLaunchLevelError):
        D.validate_declared(blanked, resolved=resolved_by_level["dr_l1"])


# --------------------------------------------------------------------------- #
# 5. "入口通了但那一档还发不出去"必须与"入口没接"可分辨
# --------------------------------------------------------------------------- #
def test_dr_l0_passes_preflight_and_dr_l1_stops_with_a_named_checklist():
    D.preflight_launchable("dr_l0")
    assert D.launch_blockers("dr_l0") == ()
    with pytest.raises(D.DrLaunchLevelError) as excinfo:
        D.preflight_launchable("dr_l1")
    message = str(excinfo.value)
    assert "has a launch entrance but no materialized lineage yet" in message
    blockers = D.launch_blockers("dr_l1")
    assert len(blockers) == 3
    for blocker in blockers:
        assert blocker in message


@pytest.mark.parametrize("module", (A, C), ids=("A211", "C211"))
def test_both_launchers_stop_on_dr_l1_before_touching_any_artifact(module, tmp_path):
    """选 dr_l1 会在 lineage 那一步明确停车,而且拒收信息就是那张清单。"""

    with pytest.raises(module.LaunchRefused) as excinfo:
        module._validate_lineage(
            tmp_path / "no-such-checkout", "a" * 40, {}, level="dr_l1"
        )
    message = str(excinfo.value)
    assert "dr_l1_lineage_artifact_missing" in message
    assert "dr_l1_grid_activity" not in message


@pytest.mark.parametrize("module", (A, C), ids=("A211", "C211"))
def test_the_receipt_lie_gate_has_a_real_call_site_in_the_launcher(module):
    """``validate_declared`` 不许是一道零调用点的门。

    它必须被 claim 复核路径真的调用,并且拒收信息里带自己的名字 —— 否则谎报只会
    撞上那句泛泛的 "output contract drifted",读日志的人无从知道差在哪。
    """

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_DRL.validate_declared(" in source
    assert "claimed DR launch level self-report differs" in source
    assert source.count('"dr_launch_level_contract"') >= 2


@pytest.mark.parametrize("module", (A, C), ids=("A211", "C211"))
def test_the_dr_l1_profile_and_manifest_are_pinned_in_the_provenance_table(module):
    """选得中就必须钉得住:未被 provenance 覆盖的文件不许参与决策。"""

    pinned = {relative for relative, _label in module.RUNTIME_SOURCE_PATHS}
    assert module.DR_LAUNCH_LEVELS_SOURCE in pinned
    assert module.DR_L1_TASK_PROFILE_SOURCE in pinned
    assert module.DR_L1_MANIFEST_SOURCE in pinned
    for relative in (
        module.DR_LAUNCH_LEVELS_SOURCE,
        module.DR_L1_TASK_PROFILE_SOURCE,
        module.DR_L1_MANIFEST_SOURCE,
    ):
        assert (REPO / relative).is_file(), relative


# --------------------------------------------------------------------------- #
# 6. 注册表本身不许退化
# --------------------------------------------------------------------------- #
def test_two_levels_may_not_share_a_profile_a_lineage_or_a_manifest(monkeypatch):
    for field, value in (
        ("profile_infix", "DRL0"),
        ("manifest_source", D.manifest_source("dr_l0")),
        ("hard_contract_identity", D.hard_contract_identity("dr_l0")),
    ):
        levels = copy.deepcopy(D._LEVELS)
        levels["dr_l1"][field] = value
        monkeypatch.setattr(D, "_LEVELS", levels)
        with pytest.raises(RuntimeError):
            D._assert_registry_is_well_formed()
        monkeypatch.undo()

    lineages = dict(D.LINEAGE_KINDS)
    lineages[("A211", "dr_l1")] = lineages[("A211", "dr_l0")]
    monkeypatch.setattr(D, "LINEAGE_KINDS", lineages)
    with pytest.raises(RuntimeError):
        D._assert_registry_is_well_formed()
    monkeypatch.undo()


def test_flipping_the_default_level_is_refused_at_import_time(monkeypatch):
    """默认档换成 dr_l1 = 四格在没人按旗标时静默换档。这条门盯的就是它。"""

    monkeypatch.setattr(D, "DEFAULT_LEVEL", "dr_l1")
    with pytest.raises(RuntimeError, match="default DR launch level"):
        D._assert_registry_is_well_formed()


def test_an_axis_without_a_cross_check_field_map_is_refused(monkeypatch):
    """字段映射表为空 = 那条轴根本没被比过,却照样过关。"""

    trimmed = copy.deepcopy(D._L1_AXIS_FIELD_MAP)
    trimmed["base_com"] = {}
    monkeypatch.setattr(D, "_L1_AXIS_FIELD_MAP", trimmed)
    with pytest.raises(RuntimeError, match="base_com"):
        D._assert_registry_is_well_formed()
