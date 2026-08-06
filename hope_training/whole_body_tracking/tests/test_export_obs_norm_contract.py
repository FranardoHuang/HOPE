"""Static regression guards for ONNX observation-normalization provenance (no Isaac/Torch)."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"


def _source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    ast.parse(text)
    return text


def test_native_export_metadata_requires_explicit_obs_norm_truth():
    source = _source(EXPORTER)
    tree = ast.parse(source)
    attach = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "attach_onnx_metadata"
    )
    kwonly = {arg.arg: default for arg, default in zip(attach.args.kwonlyargs, attach.args.kw_defaults)}
    assert "obs_norm_baked" in kwonly
    assert kwonly["obs_norm_baked"] is None, "obs_norm_baked must be a required keyword"
    assert "trained_with_obs_norm" in kwonly
    assert kwonly["trained_with_obs_norm"] is None
    assert "source_checkpoint_path" in kwonly
    assert kwonly["source_checkpoint_path"] is None
    assert '"obs_norm_baked": "1" if obs_norm_baked else "0"' in source
    assert '"trained_with_obs_norm": "1" if trained_with_obs_norm else "0"' in source
    assert '"empirical_normalization": "1" if trained_with_obs_norm else "0"' in source
    assert "obs_norm_baked = is_empirical_normalizer(normalizer)" in source
    assert "not isinstance(normalizer, torch.nn.Identity)" in source
    assert "return obs_norm_baked" in source


def test_every_native_attach_call_passes_obs_norm_baked():
    paths = [
        ROOT / "scripts/play.py",
        ROOT / "scripts/rsl_rl/play.py",
        ROOT / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py",
    ]
    calls = 0
    for path in paths:
        tree = ast.parse(_source(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            if name != "attach_onnx_metadata":
                continue
            calls += 1
            assert any(keyword.arg == "obs_norm_baked" for keyword in node.keywords), path
            assert any(keyword.arg == "trained_with_obs_norm" for keyword in node.keywords), path
            assert any(keyword.arg == "source_checkpoint_path" for keyword in node.keywords), path
    # 2026-08-06: 4 -> 3。删掉的那一处在 my_on_policy_runner.MyOnPolicyRunner.save 里,
    # 那个 runner 从 8a9d329c 引入起就零实例化(全仓零引用),它的 save() 与现役
    # MotionOnPolicyRunner.save() 的导出段逐行相同。计数本身仍然是硬的:它防的是
    # "某个 attach 调用点被 AST 走漏"。
    assert calls == 3


def test_native_callers_use_runner_normalizer_not_policy_attribute():
    for path in (
        ROOT / "scripts/play.py",
        ROOT / "scripts/rsl_rl/play.py",
        ROOT / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py",
    ):
        source = _source(path)
        assert 'getattr(ppo_runner.alg.policy, "actor_obs_normalizer"' not in source
        assert 'getattr(self.alg.policy, "actor_obs_normalizer"' not in source


def test_standalone_export_always_overwrites_donor_flag():
    source = _source(ROOT / "scripts/standalone_onnx_export.py")
    assert 'donor_meta["obs_norm_baked"] = "1" if args.bake_obs_norm else "0"' in source
    assert 'donor_meta["empirical_normalization"]' in source


def test_mask_provenance_is_checkpoint_bound_before_any_exact_promotion():
    exporter = _source(EXPORTER)
    standalone = _source(ROOT / "scripts/standalone_onnx_export.py")
    assert "facts cannot establish provenance for actor bytes" in exporter
    assert "bind_actor_leg_ref_mask_metadata(metadata, runtime_facts)" not in exporter
    assert exporter.index('metadata["source_checkpoint_sha256"]') < exporter.index(
        "bind_actor_leg_ref_mask_metadata(\n            metadata,\n            training_contract"
    )
    assert 'training_contract.get("actor_leg_ref_mask") is not True' in exporter
    assert standalone.rindex('donor_meta["source_checkpoint_sha256"]') < standalone.rindex(
        "bind_actor_leg_ref_mask_metadata("
    )


def test_action_ball_diagnostic_brand_precedes_every_exact_promotion():
    native = _source(EXPORTER).split(
        "def attach_onnx_metadata(", 1
    )[1]
    standalone = _source(
        ROOT / "scripts/standalone_onnx_export.py"
    ).split("def main() -> int:", 1)[1]

    native_lineage = native.index(
        "training_contract_lineage_exact = "
        "checkpoint_contract_lineage_exact(checkpoint)"
    )
    native_brand = native.index(
        "training_contract_diagnostic = "
        "bind_action_ball_diagnostic_metadata("
    )
    native_formal = native.index(
        "validate_schema3_contract(training_contract)",
        native_brand,
    )
    native_exact = native.rindex('metadata["training_contract_exact"] = "1"')
    assert native_lineage < native_brand < native_formal < native_exact
    assert (
        "lineage_exact=training_contract_lineage_exact,"
        in native[native_brand:native_formal]
    )

    standalone_lineage = standalone.index(
        "training_contract_lineage_exact = "
        "checkpoint_contract_lineage_exact(ckpt)"
    )
    standalone_brand = standalone.index(
        "training_contract_diagnostic = "
        "bind_action_ball_diagnostic_metadata("
    )
    standalone_exact = standalone.index(
        '"training_contract_exact": "1" '
        'if training_contract_lineage_exact else "0"',
        standalone_brand,
    )
    assert standalone_lineage < standalone_brand < standalone_exact
    assert (
        "lineage_exact=training_contract_lineage_exact,"
        in standalone[standalone_brand:standalone_exact]
    )


def test_action_ball_export_identity_is_checkpoint_bound_and_live_env_checked():
    native = _source(EXPORTER)
    standalone = _source(ROOT / "scripts/standalone_onnx_export.py")
    assert "bind_action_ball_action_set_metadata(" in native
    assert "checkpoint_action_set_identity = action_ball_action_set_identity(" in native
    assert "runtime_identity_getter = getattr(" in native
    assert '"action_ball_hard_contract", None' in native
    assert "validate_action_ball_action_set_runtime_identity(" in native
    assert (
        native.index("checkpoint_action_set_identity = action_ball_action_set_identity(")
        < native.index("validate_action_ball_action_set_runtime_identity(")
    )
    # Standalone export has no live Isaac env, so it may only replace donor
    # labels from the adjacent checkpoint-bound contract.
    assert "bind_action_ball_action_set_metadata(" in standalone
    assert standalone.index("bind_action_ball_action_set_metadata(") > standalone.index(
        "training_contract_lineage_exact = "
        "checkpoint_contract_lineage_exact(ckpt)"
    )


def test_mujoco_consumer_implements_normalization_truth_table():
    source = _source(ROOT / "scripts/mujoco_eval_onnx.py")
    assert "self.obs_norm_baked and self.empirical_normalization is False" in source
    assert "stale normalization sidecar ignored" in source
    assert "this ONNX declares empirical_normalization=1" in source
    assert "obs normalization: ON (baked into ONNX graph)" in source


def test_train_retiming_is_distinct_from_native_export_clock():
    exporter = _source(EXPORTER)
    play = _source(ROOT / "scripts/play.py")
    assert 'metadata["export_reference_clock_speed"] = "1"' in exporter
    assert 'metadata["training_motion_speed_scale_range_json"]' in exporter
    assert 'metadata["training_motion_speed_scale_per_clip_json"]' in exporter
    # Export verification must not compare a deliberately native replay env's
    # clock override against the immutable training-time sampling distribution.
    assert '"motion_speed_scale_range": _contract_value(motion_cfg.speed_scale_range)' not in exporter
    assert '"motion_speed_scale_per_clip": _contract_value(' not in exporter
    # The main replay path clears both random and deterministic per-clip
    # retiming; forgetting the latter silently exports/replays the slow side.
    assert "env_cfg.commands.motion.speed_scale_range = (1.0, 1.0)" in play
    assert "env_cfg.commands.motion.speed_scale_per_clip = None" in play
