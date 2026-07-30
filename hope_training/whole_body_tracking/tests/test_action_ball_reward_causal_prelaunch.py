import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts/action_ball_reward_causal_prelaunch.py"
SPEC = importlib.util.spec_from_file_location("reward_causal_prelaunch", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class Tensor:
    """Tiny tensor protocol is intentionally not faked here.

    Transactional tensor math is covered on Pod with the real Torch/Isaac
    runtime.  Host tests exercise the no-clobber/content/coverage contracts and
    prove that raw values cannot be supplied through the report API.
    """


def _term(name, group, weight=1.0, axis=None):
    return {
        "name": name,
        "callable": f"pkg.{name}",
        "weight": weight,
        "params": {},
        "role": "objective",
        "recipe_term_sha256": "a" * 64,
        "group": group,
        "source": "test",
        "expected_weight_sign": "positive" if weight > 0 else "negative",
        "expected_contribution": "positive" if weight > 0 else "negative",
        "adjustability": "preregistered_scientific",
        "causal_axis": axis or f"{name}_axis",
    }


def test_no_clobber_receipt_is_content_bound(tmp_path):
    report = {
        "schema_version": 1,
        "all_active_objectives_causal": True,
        "sha256": "1" * 64,
    }
    output = tmp_path / "new"
    target = MOD.write_no_clobber_receipt(
        output, report, bindings={"commit": "2" * 40}
    )
    payload = json.loads(target.read_text())
    expected = payload.pop("sha256")
    assert expected == MOD._sha256_bytes(MOD._canonical_bytes(payload))
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        MOD.write_no_clobber_receipt(
            output, report, bindings={"commit": "2" * 40}
        )


def test_intervention_contracts_are_executable_requirements_not_pass_claims():
    contracts = MOD.build_action_ball_reward_intervention_contracts()
    assert contracts["status"] == "REQUIRED_LIVE_CANARY_NOT_EXECUTED"
    rows = {
        row["intervention_id"]: row for row in contracts["rows"]
    }
    assert set(rows) == {
        "delayed_hard_death",
        "reference_reset",
        "soft_retreat_vs_cross",
        "progress_closed_loop",
    }
    assert rows["delayed_hard_death"][
        "intervention_lag_control_steps"
    ] == [0, 12, 40, 78, 100]
    assert rows["reference_reset"]["required_termination_terms"] == [
        "anchor_pos",
        "anchor_ori",
        "ee_body_pos",
    ]
    assert rows["soft_retreat_vs_cross"][
        "horizon_control_steps"
    ] == [2, 45, 46, 180, 181]
    assert rows["progress_closed_loop"]["acceptance"][
        "undiscounted_abs_max"
    ] == pytest.approx(1.0e-6)
    assert all(
        row["status"] == "REQUIRED_LIVE_CANARY_NOT_EXECUTED"
        and row["result"] is None
        for row in rows.values()
    )
    assert "PASS" not in json.dumps(contracts, sort_keys=True)


def test_git_binding_requires_clean_untracked_free_head_blob(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "causal-audit@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Causal Audit Test"],
        cwd=repo,
        check=True,
    )
    producer = repo / "producer.py"
    producer.write_text("print('tracked')\n")
    subprocess.run(["git", "add", "producer.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "producer"], cwd=repo, check=True)

    binding = MOD._git_binding(repo, producer)
    assert binding["checkout_clean_including_untracked"] is True
    assert binding["producer_head_blob_sha256"] == MOD._sha256_file(producer)

    untracked = repo / "untracked.txt"
    untracked.write_text("not allowed\n")
    with pytest.raises(MOD.CausalAuditError, match="including untracked"):
        MOD._git_binding(repo, producer)
    untracked.unlink()

    # Even if Git status is deliberately told to ignore the path, the producer
    # bytes must still equal the blob in HEAD.
    subprocess.run(
        ["git", "update-index", "--assume-unchanged", "producer.py"],
        cwd=repo,
        check=True,
    )
    producer.write_text("print('different but status-hidden')\n")
    with pytest.raises(MOD.CausalAuditError, match="bytes differ"):
        MOD._git_binding(repo, producer)


@pytest.mark.skipif(torch is None, reason="real tensor semantic regression requires Torch")
def test_root_velocity_probe_verifies_production_getter_readback():
    class Data:
        def __init__(self):
            self.root_state_w = torch.zeros((1, 13), dtype=torch.float32)

        @property
        def root_ang_vel_b(self):
            # This fake production getter intentionally derives a fresh tensor
            # from the authoritative root state on every access.
            return self.root_state_w[:, 10:13].clone()

    data = Data()
    env = types.SimpleNamespace(
        scene={"robot": types.SimpleNamespace(data=data)}
    )
    term_cfg = types.SimpleNamespace(params={"asset_cfg": None})
    setup = MOD._setup_root_velocity_l2(
        env,
        term_cfg,
        term_name="base_ang_vel_xy",
        axis="base_roll_pitch_rate",
        velocity_slice=slice(10, 13),
        component=0,
    )
    setup.prepare_baseline()
    setup.verify_production_readback("baseline")
    setup.prepare_worsened()
    setup.verify_production_readback("worsened")

    class BrokenData(Data):
        @property
        def root_ang_vel_b(self):
            return torch.zeros((1, 3), dtype=torch.float32)

    broken = types.SimpleNamespace(
        scene={"robot": types.SimpleNamespace(data=BrokenData())}
    )
    broken_setup = MOD._setup_root_velocity_l2(
        broken,
        term_cfg,
        term_name="base_ang_vel_xy",
        axis="base_roll_pitch_rate",
        velocity_slice=slice(10, 13),
        component=0,
    )
    broken_setup.prepare_worsened()
    with pytest.raises(MOD.CausalAuditError, match="production readback"):
        broken_setup.verify_production_readback("worsened")


@pytest.mark.skipif(torch is None, reason="real tensor semantic regression requires Torch")
def test_foot_landing_advanced_index_baseline_changes_authoritative_tensor():
    history = torch.ones((1, 2, 3, 3), dtype=torch.float32)
    contact_time = torch.zeros((1, 3), dtype=torch.float32)
    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(
            net_forces_w_history=history,
            current_contact_time=contact_time,
        )
    )
    env = types.SimpleNamespace(
        scene=types.SimpleNamespace(sensors={"feet": sensor}),
        step_dt=0.02,
    )
    sensor_cfg = types.SimpleNamespace(name="feet", body_ids=[0, 2])
    term_cfg = types.SimpleNamespace(
        params={"sensor_cfg": sensor_cfg, "force_threshold_n": 300.0}
    )
    setup = MOD._setup_foot_soft_landing(env, term_cfg)
    setup.prepare_baseline()
    assert torch.count_nonzero(history[0, :, [0, 2], :]) == 0
    assert torch.all(history[0, :, 1, :] == 1.0)
    setup.prepare_worsened()
    assert history[0, 0, 0, 2] == 600.0


def test_candidate_b_only_multiplies_three_tracking_terms():
    active = [
        _term("racket_position", "hope_hit_landing_task", 4.0),
        _term("racket_velocity", "hope_hit_landing_task", 0.5),
        _term("racket_normal", "hope_hit_landing_task", 0.5),
        _term("virtual_landing", "hope_hit_landing_task", 1648.8),
        _term("upright_exp", "mjlab_balance_stability", 1.0),
        _term("death_penalty", "immutable_safety", -3600.0),
    ]
    candidates = MOD._candidate_recipes(active, 0.02)
    a = {row["name"]: row["weight"] for row in candidates[0]["terms"]}
    b = {row["name"]: row["weight"] for row in candidates[1]["terms"]}
    assert b["racket_position"] == 4 * a["racket_position"]
    assert b["racket_velocity"] == 4 * a["racket_velocity"]
    assert b["racket_normal"] == 4 * a["racket_normal"]
    assert b["virtual_landing"] == a["virtual_landing"]
    assert b["upright_exp"] == a["upright_exp"]
    assert b["death_penalty"] == a["death_penalty"]
    assert candidates[0]["applied_to_training"] is False
    assert candidates[1]["applied_to_training"] is False


def test_racket_progress_budget_separates_unit_raw_from_callable_cap():
    active = [
        _term("racket_progress", "hope_hit_landing_task", 10.0),
        _term("racket_position", "hope_hit_landing_task", 4.0),
    ]
    candidates = MOD._candidate_recipes(active, 0.02)
    a = candidates[0]
    rows = {row["name"]: row for row in a["terms"]}
    assert rows["racket_progress"]["unit_raw_weighted_budget"] == pytest.approx(0.2)
    assert rows["racket_progress"]["bounded_weighted_budget"] == pytest.approx(0.03)
    assert rows["racket_position"]["unit_raw_weighted_budget"] == pytest.approx(0.08)
    assert rows["racket_position"]["bounded_weighted_budget"] == pytest.approx(0.08)
    group = "hope_hit_landing_task"
    assert a["unit_raw_dimensioned_budget_by_group"][group][
        "dense_positive_per_control_step"
    ] == pytest.approx(0.28)
    assert a["callable_bounded_dimensioned_budget_by_group"][group][
        "dense_positive_per_control_step"
    ] == pytest.approx(0.11)


def test_expected_adopted_action_ball_objectives_have_reviewed_mutations():
    # Source-level expectation only.  Formal closure comes from exact equality
    # between the real post-compose receipt and live RewardManager at runtime;
    # a future active objective without a recipe then fails closed.
    adopted = {
        "upright_exp",
        "hit_unstable_support",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "racket_position",
        "racket_velocity",
        "racket_normal",
        "base_position",
        "racket_progress",
        "virtual_landing",
        "joint_torques",
        "arm_torque_saturation",
        "action_rate_clamped",
        "action_acc_l2",
        "undesired_contacts",
        "foot_slip_sq",
        "foot_velocity",
        "foot_soft_landing",
        "base_ang_vel_xy",
        "base_lin_vel_z",
        "joint_vel",
        "qdes_limit_barrier",
        "joint_limit",
        "death_penalty",
    }
    assert adopted - set(MOD._SETUP_BY_TERM) == set()


def test_unsupported_objective_is_explicit_fail_closed(monkeypatch):
    recipe = {
        "sha256": "3" * 64,
        "terms": [
            {
                "name": "brand_new_objective",
                "callable": "pkg.brand_new_objective",
                "weight": 1.0,
                "params": {},
            }
        ],
    }
    taxonomy = {
        "sha256": "4" * 64,
        "active_terms": [
            _term(
                "brand_new_objective",
                "mjlab_balance_stability",
                axis="new_axis",
            )
        ],
    }
    fake_recipe_module = types.SimpleNamespace(
        REWARD_TERM_ROLE_OBJECTIVE="objective",
        build_action_ball_reward_group_taxonomy=lambda terms: taxonomy,
        build_effective_reward_receipt=lambda cfg, expected_sha256=None: recipe,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        fake_recipe_module,
    )
    env = types.SimpleNamespace(
        num_envs=1,
        reward_manager=types.SimpleNamespace(
            active_terms=["brand_new_objective"],
            get_term_cfg=lambda name: types.SimpleNamespace(weight=1.0),
        ),
    )
    report = MOD.build_live_causal_report(env, recipe, step_dt_s=0.02)
    assert report["all_active_objectives_causal"] is False
    assert report["causal_pass_count"] == 0
    assert report["coverage"] == [
        {
            "term_name": "brand_new_objective",
            "group": "mjlab_balance_stability",
            "causal_axis": "new_axis",
            "accounting_scope": "per_control_step",
            "status": "unsupported_fail_closed",
            "reason": "no reviewed transactional mutation recipe",
        }
    ]
    assert (
        report["groups"]["mjlab_balance_stability"][
            "all_active_objectives_causal"
        ]
        is False
    )


def test_live_manager_recipe_must_equal_post_compose_receipt(monkeypatch):
    recipe = {
        "sha256": "3" * 64,
        "terms": [
            {
                "name": "upright_exp",
                "callable": "pkg.upright_exp",
                "weight": 1.0,
                "params": {},
            }
        ],
    }
    mismatched = {
        **recipe,
        "terms": [{**recipe["terms"][0], "weight": 2.0}],
    }
    fake_recipe_module = types.SimpleNamespace(
        REWARD_TERM_ROLE_OBJECTIVE="objective",
        build_action_ball_reward_group_taxonomy=lambda terms: {
            "sha256": "4" * 64,
            "active_terms": [],
        },
        build_effective_reward_receipt=lambda cfg, expected_sha256=None: mismatched,
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        fake_recipe_module,
    )
    env = types.SimpleNamespace(
        num_envs=1,
        reward_manager=types.SimpleNamespace(
            active_terms=["upright_exp"],
            get_term_cfg=lambda name: types.SimpleNamespace(
                weight=1.0, func=lambda env: None, params={}
            ),
        ),
    )
    with pytest.raises(
        MOD.CausalAuditError,
        match="callable/weight/params differ",
    ):
        MOD.build_live_causal_report(env, recipe, step_dt_s=0.02)


def test_audit_failure_uses_nonzero_hard_exit(monkeypatch):
    seen = []

    def fake_exit(code):
        seen.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(MOD.os, "_exit", fake_exit)
    with pytest.raises(SystemExit) as raised:
        MOD._hard_exit_after_audit_failure()
    assert raised.value.code == 1
    assert seen == [1]


def test_live_term_raw_cannot_be_injected():
    calls = []

    def func(env, scale):
        calls.append((env, scale))
        return types.SimpleNamespace(
            ndim=1,
            shape=(1,),
            detach=lambda: types.SimpleNamespace(
                cpu=lambda: types.SimpleNamespace(item=lambda: 0.75)
            ),
        )

    cfg = types.SimpleNamespace(func=func, params={"scale": 2.0})
    env = object()
    assert MOD._call_live_term(env, cfg, name="x") == pytest.approx(0.75)
    assert calls == [(env, 2.0)]
    assert "raw" not in MOD._call_live_term.__code__.co_varnames


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "x"])
def test_finite_fails_closed(bad):
    with pytest.raises(MOD.CausalAuditError):
        MOD._finite(bad, label="bad")
