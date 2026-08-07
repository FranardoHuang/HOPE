"""Host-only contract tests for the ActionBall projection-penalty Hydra knob."""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import pytest

# The checked-in host Python intentionally has no Hydra/OmegaConf runtime.  This
# test only imports train.py's pure translation helpers, so provide the same
# decorator/import-time shims used by the neighboring host-only suites.
try:  # pragma: no cover - the Pod/Isaac environment takes the real import path
    import hydra as _hydra  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised by the host lane
    hydra_stub = types.ModuleType("hydra")
    hydra_stub.main = lambda **_kwargs: (lambda fn: fn)
    sys.modules["hydra"] = hydra_stub

try:  # pragma: no cover - the Pod/Isaac environment takes the real import path
    import omegaconf as _omegaconf  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised by the host lane
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.ListConfig = list
    omegaconf_stub.OmegaConf = SimpleNamespace()
    sys.modules["omegaconf"] = omegaconf_stub

from test_effective_reward_recipe import RECIPE
from test_reward_flags_overrides import (
    _Term,
    _apply_legacy_v1,
    _make_env_cfg,
    train_mod,
)


def qdes_projection_penalty():
    """Stable callable identity for the dependency-light reward receipt."""


def _projection_cfg(weight=-1.0):
    cfg = _make_env_cfg()
    cfg.rewards.qdes_projection_penalty = _Term(
        weight=weight,
        func=qdes_projection_penalty,
        params={"action_name": "joint_pos", "knee_frac": 0.05},
    )
    return cfg


def _apply_projection(task, cfg=None):
    cfg = _projection_cfg() if cfg is None else cfg
    applied = _apply_legacy_v1(cfg, task)
    return cfg, applied


def _projection_only_receipt(cfg):
    return RECIPE.build_effective_reward_receipt(
        {"rewards": {"qdes_projection_penalty": cfg.rewards.qdes_projection_penalty}}
    )


def test_absent_override_preserves_source_default_and_emits_no_marker():
    cfg, applied = _apply_projection({})
    assert cfg.rewards.qdes_projection_penalty.weight == -1.0
    assert not any("qdes_projection_penalty" in marker for marker in applied)


@pytest.mark.parametrize("weight", [-5, -2.5, 0])
def test_exact_numeric_weight_in_reviewed_range_applies(weight):
    cfg, applied = _apply_projection(
        {"rewards": {"qdes_projection_penalty_weight": weight}}
    )
    expected = float(weight)
    assert cfg.rewards.qdes_projection_penalty.weight == -1.0
    assert cfg.rewards.qdes_projection_penalty.params["objective_weight"] == expected
    assert any(
        f"objective_weight={expected},manager_weight=-1.0" in marker
        for marker in applied
    )


@pytest.mark.parametrize(
    "weight",
    [
        None,
        True,
        False,
        "-2.5",
        [],
        float("nan"),
        float("inf"),
        float("-inf"),
        0.01,
        -5.01,
    ],
)
def test_wrong_type_nonfinite_and_out_of_range_weights_fail_without_mutation(weight):
    cfg = _projection_cfg()
    with pytest.raises(train_mod._OverrideError, match=r"finite range \[-5\.0, 0\.0\]"):
        _apply_projection(
            {"rewards": {"qdes_projection_penalty_weight": weight}},
            cfg=cfg,
        )
    assert cfg.rewards.qdes_projection_penalty.weight == -1.0


def test_missing_term_and_unknown_keys_fail_loudly():
    with pytest.raises(train_mod._OverrideError, match="qdes_projection_penalty"):
        _apply_legacy_v1(
            _make_env_cfg(),
            {"rewards": {"qdes_projection_penalty_weight": -2.5}},
        )
    with pytest.raises(train_mod._OverrideError, match="qdes_projection_penalty_wieght"):
        _apply_projection(
            {"rewards": {"qdes_projection_penalty_wieght": -2.5}}
        )
    with pytest.raises(train_mod._OverrideError, match="qdes_projection_penalty_func"):
        _apply_projection(
            {"rewards": {"qdes_projection_penalty_func": "arbitrary"}}
        )


def test_whitelist_contains_only_the_exact_projection_weight_knob_once():
    keys = [key for key in train_mod._REWARD_KEYS if key.startswith("qdes_projection")]
    assert keys == ["qdes_projection_penalty_weight"]


def test_effective_receipt_tracks_nonzero_dose_and_zero_control_changes_sha():
    baseline = _projection_cfg()
    baseline_receipt = _projection_only_receipt(baseline)
    assert baseline_receipt["terms"][0]["name"] == "qdes_projection_penalty"
    assert baseline_receipt["terms"][0]["weight"] == -1.0

    treatment, _ = _apply_projection(
        {"rewards": {"qdes_projection_penalty_weight": -2.5}}
    )
    treatment_receipt = _projection_only_receipt(treatment)
    assert treatment_receipt["terms"][0]["weight"] == -1.0
    assert treatment_receipt["terms"][0]["params"]["objective_weight"] == -2.5
    assert treatment_receipt["sha256"] != baseline_receipt["sha256"]
    taxonomy = RECIPE.build_action_ball_reward_group_taxonomy(
        treatment_receipt["terms"]
    )
    assert taxonomy["active_terms"][0]["expected_weight_sign"] == "negative"

    control, _ = _apply_projection(
        {"rewards": {"qdes_projection_penalty_weight": 0}}
    )
    control_receipt = _projection_only_receipt(control)
    assert control_receipt["terms"][0]["weight"] == -1.0
    assert control_receipt["terms"][0]["params"]["objective_weight"] == 0.0
    assert control_receipt["sha256"] != baseline_receipt["sha256"]


def test_zero_control_has_explicit_action_ball_contract_pin():
    cfg, _ = _apply_projection(
        {"rewards": {"qdes_projection_penalty_weight": 0}}
    )
    assert train_mod._qdes_projection_penalty_contract_weight(cfg) == 0.0
    assert train_mod._qdes_projection_penalty_contract(cfg) == {
        "schema_version": 2,
        "objective_weight": 0.0,
        "reward_manager_weight": -1.0,
        "weight_independent_exposure": True,
        "exposure_denominator": "control_step_observed_sample_count",
        "hypothetical_unweighted_penalty": "projection_penalty_value_sum",
        "per_joint_exposure": True,
        "action_name": "joint_pos",
        # 2026-08-07 裁定二:核换成开源那条线性尾巴,合同的语义面必须跟着换 ——
        # 旧公式串/旧参数名的 sidecar 不能静默续到新数学上。
        "knee_frac": 0.05,
        "kernel_unit": "radian",
        "tail": "linear_unbounded_slope_one_per_radian",
        "per_joint_cap": None,
    }

    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert '"effective_reward_recipe": effective_reward_receipt' in source
    assert '"qdes_projection_penalty_weight": (' in source
    assert "_qdes_projection_penalty_contract_weight(env_cfg)" in source
    assert "_qdes_projection_penalty_contract(" in source


@pytest.mark.parametrize("weight", [True, "-2.5", float("nan"), 0.01, -5.01])
def test_contract_pin_revalidates_tampered_runtime_weight(weight):
    cfg = _projection_cfg(weight=weight)
    with pytest.raises(RuntimeError, match="projection_penalty"):
        train_mod._qdes_projection_penalty_contract_weight(cfg)


def test_contract_rejects_manager_weight_bypass_after_explicit_override():
    cfg, _ = _apply_projection(
        {"rewards": {"qdes_projection_penalty_weight": 0}}
    )
    cfg.rewards.qdes_projection_penalty.weight = 0.0
    with pytest.raises(RuntimeError, match="exposure contract"):
        train_mod._qdes_projection_penalty_contract(cfg)
