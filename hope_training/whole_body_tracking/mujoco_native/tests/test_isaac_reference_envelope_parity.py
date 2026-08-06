"""The native lane must run the LIVE ActionBall reference envelope, not the parent's.

人话:Isaac 的 ``HOPEActionBallTerminationsCfg`` 把 ``ee_body_pos`` 从"两脚+两腕"
收窄成"只有两脚"(腕是挥拍要甩最远的那一端,0.25 m 的 z 包络套上去等于在惩罚我们
要教的动作)。复刻侧以前抄的是父类那份四个身体的名单,于是它会在现役 kernel 明确
放行的腕部位移上终止 —— 而且**不会响**:相位保真的 AST 指纹选择器只点了
``joint_qdes_forbidden,joint_actual_forbidden`` 两项,覆写改了它一个 bit 都不动;
现有测试也看不见,因为它们给四个身体喂的是同一个数(和轴对齐盒那次同一个错误)。

每条检查都配一个"粗一档就过不了"的变异:

* 只改腕、不改脚的位移 —— 四个格子同值的测试对它完全是瞎的。
* 只改现役覆写、不改复刻 —— 门必须拒绝(值比对 + 指纹双双开火)。
* 把覆写整条删掉(退回父类)—— 指纹必须开火,而复刻的活值必须跟着变成四个身体。
* 往那个类里新加一条终止项 —— 指纹按名字点名,天生看不见,所以由声明项集合兜底。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
WBT_ROOT = REPO_ROOT / "hope_training/whole_body_tracking"
if str(WBT_ROOT) not in sys.path:
    sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import isaac_reference_envelope as envelope  # noqa: E402
from mujoco_native import n1_ball_core as n1  # noqa: E402
from mujoco_native import vec_env  # noqa: E402


ACTION_BALL = envelope.ACTION_BALL_TERMINATIONS_CLASS
DEPLOY_PARITY = envelope.DEPLOY_PARITY_TERMINATIONS_CLASS

#: The ActionBall override exactly as the live config writes it.
LIVE_OVERRIDE_BODY_NAMES = '"body_names": list(A3_FEET_BODIES),'
#: The whole override statement, so a test can delete it and inherit the parent.
LIVE_OVERRIDE_BLOCK = """    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only_hold_aware,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": list(A3_FEET_BODIES),
            "ignore_hold": True,
        },
    )
"""


def _mutated_config(tmp_path: Path, old: str, new: str) -> Path:
    """One textual substitution on a copy of the live Isaac termination config."""

    source = vec_env.TERMINATION_SOURCE_CONFIG
    text = source.read_text("utf-8")
    assert text.count(old) == 1, f"mutation anchor is not unique: {old!r}"
    target = tmp_path / source.name
    target.write_text(text.replace(old, new), "utf-8")
    envelope.clear_caches()
    return target


def _clear_contract_caches() -> None:
    vec_env._phase_fidelity_sample_contract_cached.cache_clear()
    vec_env._termination_blocker_receipt_cached.cache_clear()
    vec_env._termination_contract_receipt_cached.cache_clear()
    envelope.clear_caches()


@pytest.fixture(autouse=True)
def _restore_caches():
    _clear_contract_caches()
    yield
    _clear_contract_caches()


# ---------------------------------------------------------------------------
# The replica runs the ActionBall envelope, read as values
# ---------------------------------------------------------------------------


def test_replica_envelope_is_the_live_action_ball_override_not_the_parent():
    live_child = envelope.live_reference_envelope(ACTION_BALL)
    live_parent = envelope.live_reference_envelope(DEPLOY_PARITY)

    # The two really are different -- if they were not, this whole test would
    # be vacuous and could not tell a correct mirror from a stale one.
    assert live_child["body_names"] != live_parent["body_names"]
    assert set(live_child["body_names"]) < set(live_parent["body_names"])

    # The subclass override -- not the inherited parent -- is what supplies it.
    assert live_child["owner_class"] == ACTION_BALL
    assert envelope.live_class_chain(ACTION_BALL)[:2] == (ACTION_BALL, DEPLOY_PARITY)

    assert vec_env.PHASE_TERMINATIONS_MIRRORED_ISAAC_CLASS == ACTION_BALL
    assert vec_env.PHASE_EE_BODY_NAMES == live_child["body_names"]
    assert vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M == live_child["threshold_m"]

    # The wrists are the bodies the override removed; they must be gone.
    wrists = set(live_parent["body_names"]) - set(live_child["body_names"])
    assert wrists == {"left_wrist_yaw_Link", "right_wrist_yaw_Link"}
    assert not wrists & set(vec_env.PHASE_EE_BODY_NAMES)


def test_wrist_only_displacement_is_no_longer_representable_at_all():
    """只改腕、不改脚:四个格子同值的旧测试对这个 bug 完全是瞎的。

    The old sample carried four slots, feet first and wrists last.  A wrist-only
    excursion is slots 2..3 over threshold with slots 0..1 at zero -- which the
    live kernel allows and the replica terminated on.  With the envelope read
    live there are only two slots, so the same vector is now rejected outright
    instead of being scored as if the wrists were still guarded.
    """

    feet = len(vec_env.PHASE_EE_BODY_NAMES)
    over = np.nextafter(vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M, np.inf)
    parent_bodies = len(envelope.live_reference_envelope(DEPLOY_PARITY)["body_names"])
    assert parent_bodies > feet

    wrist_only = {
        "schema_version": 1,
        "kind": "a3_mujoco_phase_fidelity_sample_v1",
        "motion_phase_context": "non_hold_swing_or_follow_through",
        "in_hold": False,
        "reference_terminations_enabled": True,
        "anchor_pos_z_error_m": 0.0,
        "anchor_projected_gravity_z_error_abs": 0.0,
        "ee_body_pos_z_error_m": [0.0] * feet + [over] * (parent_bodies - feet),
    }
    with pytest.raises(
        vec_env.VecEnvContractError, match="finite non-negative values"
    ):
        vec_env.exact_phase_fidelity_reasons(wrist_only)

    # And within the live envelope, EACH foot alone still terminates -- fed one
    # at a time, never the same number in every slot.
    for index in range(feet):
        errors = [0.0] * feet
        errors[index] = over
        one_foot = {**wrist_only, "ee_body_pos_z_error_m": errors}
        assert vec_env.exact_phase_fidelity_reasons(one_foot) == ("ee_body_pos",)
        below = [0.0] * feet
        below[index] = vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M
        assert (
            vec_env.exact_phase_fidelity_reasons(
                {**wrist_only, "ee_body_pos_z_error_m": below}
            )
            == ()
        )


def test_phase_contract_and_tape_carry_the_live_two_body_order():
    contract = vec_env.phase_fidelity_sample_contract()
    assert contract["ee_body_order"] == list(vec_env.PHASE_EE_BODY_NAMES)
    assert contract["ee_body_order_mirrors_isaac_class"] == ACTION_BALL
    assert contract["ee_body_order_source"] == "live_isaac_class_term_param_value"
    assert contract["live_declared_terms_compared"][ACTION_BALL] == sorted(
        envelope.DECLARED_TERMS[ACTION_BALL]
    )

    # 人话:磁带侧以前写死"必须 4 个身体",那是第四份手抄的宽度。
    stale = dict(contract)
    stale["ee_body_order"] = list(
        envelope.live_reference_envelope(DEPLOY_PARITY)["body_names"]
    )
    stale.pop("content_sha256")
    stale["content_sha256"] = n1._sha256(n1._canonical_json_bytes(stale))
    with pytest.raises(n1.N1BallCoreError, match="body order differs from the live"):
        n1._phase_sample_contract_fields(stale)

    # Same length, wrong order: a count check cannot see this one at all.
    shuffled = dict(contract)
    shuffled["ee_body_order"] = list(reversed(contract["ee_body_order"]))
    shuffled.pop("content_sha256")
    shuffled["content_sha256"] = n1._sha256(n1._canonical_json_bytes(shuffled))
    with pytest.raises(n1.N1BallCoreError, match="body order differs from the live"):
        n1._phase_sample_contract_fields(shuffled)


# ---------------------------------------------------------------------------
# Mutations: the gate must actually fire
# ---------------------------------------------------------------------------


def test_changing_only_the_live_override_is_refused(tmp_path, monkeypatch):
    """改现役覆写但不改复刻 -> 门必须拒绝。"""

    mutated = _mutated_config(
        tmp_path,
        LIVE_OVERRIDE_BODY_NAMES,
        '"body_names": A3_FEET_BODIES + A3_HAND_BODIES,',
    )
    # The live value moved; the replica constant did not.
    assert envelope.live_reference_envelope(ACTION_BALL, config_path=mutated)[
        "body_names"
    ] != vec_env.PHASE_EE_BODY_NAMES
    assert envelope.live_reference_envelope_blockers(
        ACTION_BALL,
        config_path=mutated,
        mirrored_body_names=vec_env.PHASE_EE_BODY_NAMES,
        mirrored_threshold_m=vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M,
    )

    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", mutated)
    _clear_contract_caches()
    with pytest.raises(
        vec_env.VecEnvContractError,
        match="action_ball_config semantic AST SHA-256 drifted",
    ):
        vec_env.phase_fidelity_sample_contract()


def test_deleting_the_override_moves_both_the_fingerprint_and_the_live_value(
    tmp_path, monkeypatch
):
    """把覆写整条删掉 = 退回父类的四个身体。

    这正是 5ed998f1 那个形状的反面:指纹会开火,而且就算有人把指纹重钉过去,
    复刻读的是活值,会跟着变回四个身体,不会像以前那样停在原地。
    """

    mutated = _mutated_config(tmp_path, LIVE_OVERRIDE_BLOCK, "")
    inherited = envelope.live_reference_envelope(ACTION_BALL, config_path=mutated)
    parent = envelope.live_reference_envelope(DEPLOY_PARITY, config_path=mutated)
    assert inherited["owner_class"] == DEPLOY_PARITY
    assert inherited["body_names"] == parent["body_names"]
    assert len(inherited["body_names"]) == 4

    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", mutated)
    _clear_contract_caches()
    # The selector now NAMES ``ee_body_pos``, so a config without it cannot even
    # be selected -- which is why the refusal quotes the selector by name.
    with pytest.raises(
        vec_env.VecEnvContractError,
        match=r"HOPEActionBallTerminationsCfg\|ee_body_pos",
    ):
        vec_env.phase_fidelity_sample_contract()


def test_changing_only_the_override_threshold_is_refused(tmp_path, monkeypatch):
    mutated = _mutated_config(
        tmp_path,
        LIVE_OVERRIDE_BLOCK,
        LIVE_OVERRIDE_BLOCK.replace('"threshold": 0.25', '"threshold": 0.35'),
    )
    assert envelope.live_reference_envelope(ACTION_BALL, config_path=mutated)[
        "threshold_m"
    ] == 0.35
    blockers = envelope.live_reference_envelope_blockers(
        ACTION_BALL,
        config_path=mutated,
        mirrored_body_names=vec_env.PHASE_EE_BODY_NAMES,
        mirrored_threshold_m=vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M,
    )
    assert any("threshold_differs" in item for item in blockers)

    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", mutated)
    _clear_contract_caches()
    with pytest.raises(
        vec_env.VecEnvContractError,
        match="action_ball_config semantic AST SHA-256 drifted",
    ):
        vec_env.phase_fidelity_sample_contract()


def test_a_new_term_in_the_action_ball_class_is_caught_by_the_declared_set(
    tmp_path, monkeypatch
):
    """指纹按名字点名,新加的名字它天生看不见 —— 所以集合检查必须兜住。"""

    mutated = _mutated_config(
        tmp_path,
        LIVE_OVERRIDE_BLOCK,
        LIVE_OVERRIDE_BLOCK
        + "\n    base_fell_tilt = DoneTerm("
        "func=mdp.bad_orientation, params={\"limit_angle\": 1.4})\n",
    )
    # The named-selector digest is blind to it: the new name is not in the
    # selector, so the pinned SHA does not move by a single bit.
    selectors = (
        ("class_header", DEPLOY_PARITY),
        (
            "class_assignments",
            f"{DEPLOY_PARITY}|anchor_pos,anchor_ori,ee_body_pos,"
            "base_fell_tilt,base_too_low,robot_hit_table",
        ),
        ("class_header", ACTION_BALL),
        (
            "class_assignments",
            f"{ACTION_BALL}|ee_body_pos,joint_qdes_forbidden,"
            "joint_actual_forbidden",
        ),
    )
    assert vec_env._semantic_ast_sha256(mutated, selectors) == (
        vec_env.EXPECTED_PHASE_CONFIG_SEMANTIC_AST_SHA256
    )

    blockers = envelope.live_declared_term_blockers(mutated)
    assert any("isaac_declared_terms_differ" in item for item in blockers)
    assert any("base_fell_tilt" in item for item in blockers)

    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", mutated)
    _clear_contract_caches()
    with pytest.raises(
        vec_env.VecEnvContractError,
        match="no longer equal the live Isaac cfg",
    ):
        vec_env.phase_fidelity_sample_contract()


def test_a_removed_term_in_the_parent_class_is_caught_by_the_declared_set(
    tmp_path, monkeypatch
):
    mutated = _mutated_config(
        tmp_path,
        "    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, "
        'params={"minimum_height": 0.5})\n',
        "",
    )
    blockers = envelope.live_declared_term_blockers(mutated)
    assert any("base_too_low" in item for item in blockers)

    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", mutated)
    _clear_contract_caches()
    with pytest.raises(vec_env.VecEnvContractError):
        vec_env.phase_fidelity_sample_contract()


def test_unrelated_edits_elsewhere_in_the_config_do_not_fire(tmp_path, monkeypatch):
    """负对照:选择器是语义的,别的类怎么改都不该动这道门。"""

    expected = vec_env.phase_fidelity_sample_contract()
    source = vec_env.TERMINATION_SOURCE_CONFIG
    unrelated = tmp_path / source.name
    unrelated.write_text(
        source.read_text("utf-8")
        + "\n\ndef unrelated_reference_envelope_probe():\n    return 17\n",
        "utf-8",
    )
    monkeypatch.setattr(vec_env, "TERMINATION_SOURCE_CONFIG", unrelated)
    monkeypatch.setattr(
        vec_env.table_termination, "ISAAC_TERMINATION_CONFIG", unrelated
    )
    _clear_contract_caches()
    assert vec_env.phase_fidelity_sample_contract() == expected


# ---------------------------------------------------------------------------
# The value reader refuses to guess
# ---------------------------------------------------------------------------


def test_an_unreadable_body_list_fails_closed(tmp_path):
    mutated = _mutated_config(
        tmp_path,
        LIVE_OVERRIDE_BODY_NAMES,
        '"body_names": _pick_bodies_at_runtime(),',
    )
    with pytest.raises(envelope.IsaacReferenceEnvelopeError, match="cannot read live"):
        envelope.live_reference_envelope(ACTION_BALL, config_path=mutated)
    assert envelope.live_reference_envelope_blockers(
        ACTION_BALL,
        config_path=mutated,
        mirrored_body_names=vec_env.PHASE_EE_BODY_NAMES,
        mirrored_threshold_m=vec_env.PHASE_EE_BODY_POS_Z_THRESHOLD_M,
    )


def test_a_body_name_outside_the_a3_lists_fails_closed(tmp_path):
    mutated = _mutated_config(
        tmp_path,
        LIVE_OVERRIDE_BODY_NAMES,
        '"body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],',
    )
    # Same length, same spelling apart from the case IsaacLab matches on -- a
    # count or a set-size check would pass this happily.
    with pytest.raises(
        envelope.IsaacReferenceEnvelopeError, match="not in the A3 body-name lists"
    ):
        envelope.live_reference_envelope(ACTION_BALL, config_path=mutated)


def test_the_body_name_vocabulary_comes_from_the_live_robot_leaf():
    vocabulary = envelope.live_body_name_vocabulary()
    assert set(vec_env.PHASE_EE_BODY_NAMES) <= set(vocabulary)
    assert "left_wrist_yaw_Link" in vocabulary
