"""Exact Isaac robot/table termination port contracts for native MuJoCo."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import single_env  # noqa: E402
from mujoco_native import table_termination as term  # noqa: E402


def _synthetic_components() -> term.CollisionComponents:
    owners = np.zeros(43, dtype=np.int64)
    centers = np.zeros((43, 3), dtype=np.float64)
    axes = np.tile(np.eye(3, dtype=np.float64)[None, :, :] * 0.01, (43, 1, 1))
    return term.CollisionComponents(
        owner_indices=owners,
        local_centers_m=centers,
        local_half_axes_m=axes,
        artifact_sha256="a" * 64,
        content_sha256="b" * 64,
    )


def _five_boxes():
    lo = np.full((5, 3), 100.0, dtype=np.float64)
    hi = lo + 1.0
    lo[0] = (-0.1, -0.1, -0.1)
    hi[0] = (0.1, 0.1, 0.1)
    return lo, hi


def test_numpy_guard_matches_inclusive_exact_and_nonfinite_semantics():
    positions = np.full((32, 3), 10.0, dtype=np.float64)
    rotations = np.tile(np.eye(3, dtype=np.float64), (32, 1, 1))
    lo, hi = _five_boxes()
    components = _synthetic_components()
    assert not term.geometric_robot_table_hit(
        positions, rotations, components, lo, hi, racket_body_index=31
    )

    positions[0] = (0.0, 0.0, 0.0)
    assert term.geometric_robot_table_hit(
        positions, rotations, components, lo, hi, racket_body_index=31
    )

    # Axis-aligned: the exact test is inclusive, so a touching face still ends
    # the episode, and one float past touching does not.  The retired broad
    # phase inflated every component by COMPONENT_WORLD_AABB_GUARD_M and so
    # answered True a micrometre beyond contact; Isaac's exact terminal does
    # not, and neither may this port.
    positions[0] = (0.11, 0.0, 0.0)
    assert term.geometric_robot_table_hit(
        positions, rotations, components, lo, hi, racket_body_index=31
    )
    positions[0] = (0.110001, 0.0, 0.0)
    assert not term.geometric_robot_table_hit(
        positions, rotations, components, lo, hi, racket_body_index=31
    )

    positions[7, 0] = np.nan
    assert term.geometric_robot_table_hit(
        positions, rotations, components, lo, hi, racket_body_index=31
    )


def test_terminal_is_exact_sat_not_the_retired_world_aabb_broad_phase():
    """A rotated box whose EMPTY CORNER straddles the table is not a hit.

    人话:斜着的碰撞盒,把它胀成正方盒子会把空角落也算进去。旧版 MuJoCo 复刻就是
    这么判的,于是比 Isaac 更容易误报撞桌。这条用例在旧实现下必然失败。

    45-degree yaw puts the component's world AABB corner inside box 0 while
    every SAT axis still separates the two solids.  ``geometric_robot_table_hit``
    must answer False, exactly as the live Isaac terminal does.
    """

    lo, hi = _five_boxes()
    components = _synthetic_components()
    positions = np.full((32, 3), 10.0, dtype=np.float64)
    angle = np.pi / 4.0
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    rotations = np.tile(np.eye(3, dtype=np.float64), (32, 1, 1))
    rotations[0] = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Half-extent 0.01 cubes: rotated world AABB half-extent is 0.01*sqrt(2)
    # along x and y, so a centre at 0.1 + 0.012 is inside the inflated box in
    # every axis, while the true corner-to-face distance stays positive.
    positions[0] = (0.1 + 0.012, 0.1 + 0.012, 0.0)

    owner_rotation = rotations[components.owner_indices]
    centre = positions[components.owner_indices] + np.einsum(
        "cij,cj->ci", owner_rotation, components.local_centers_m
    )
    rotated_axes = np.einsum(
        "cij,ckj->cki", owner_rotation, components.local_half_axes_m
    )
    world_half = np.abs(rotated_axes).sum(axis=1) + term.COMPONENT_WORLD_AABB_GUARD_M
    broad = np.ones((43, 5), dtype=bool)
    for axis in range(3):
        broad &= (
            ((centre + world_half)[:, axis, None] >= lo[None, :, axis])
            & ((centre - world_half)[:, axis, None] <= hi[None, :, axis])
        )
    assert bool(broad[0, 0]), "the discriminating case needs the prefilter to pass"

    assert not term.geometric_robot_table_hit(
        positions, rotations, components, lo, hi, racket_body_index=31
    )


def test_numpy_guard_verdict_equals_the_live_isaac_terminal_kernel():
    """Randomised cross-implementation parity against the shipped Isaac kernel.

    This is the check that would have caught the port standing still while
    ``terminations`` moved from the broad phase to exact SAT in 5ed998f1.
    """

    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(WBT_ROOT / "tests"))
    from test_reward_flags_mdp import _PKG, _load  # noqa: E402  (installs the isaaclab stub)

    mdp_dir = (
        WBT_ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    )
    sys.modules[_PKG].__path__ = [str(mdp_dir)]
    isaac = _load(f"{_PKG}.terminations", "terminations.py")

    components = term.load_collision_components()
    lo = np.array(
        [
            (0.48, -0.7825, 0.69),
            (0.48, -0.7825, -0.02),
            (1.845, -0.9325, 0.74),
            (1.84, 0.8825, 0.74),
            (1.84, -0.9425, 0.74),
        ],
        dtype=np.float64,
    )
    hi = np.array(
        [
            (3.26, 0.7825, 0.78),
            (3.26, 0.7825, 0.73),
            (1.895, 0.9325, 0.9325),
            (1.90, 0.9425, 0.9525),
            (1.90, -0.8825, 0.9525),
        ],
        dtype=np.float64,
    )
    def retired_broad_phase_verdict(positions, rotations):
        """The pre-5ed998f1 rule, kept here ONLY so the sample can prove it differs."""

        owner_rotation = rotations[components.owner_indices]
        centre = positions[components.owner_indices] + np.einsum(
            "cij,cj->ci", owner_rotation, components.local_centers_m
        )
        axes = np.einsum(
            "cij,ckj->cki", owner_rotation, components.local_half_axes_m
        )
        half = np.abs(axes).sum(axis=1) + term.COMPONENT_WORLD_AABB_GUARD_M
        broad = np.ones((43, 5), dtype=bool)
        for axis in range(3):
            broad &= (
                ((centre + half)[:, axis, None] >= lo[None, :, axis])
                & ((centre - half)[:, axis, None] <= hi[None, :, axis])
            )
        if bool(np.any(broad)):
            return True
        racket_rotation = rotations[31]
        blade_centre = positions[31] + (
            racket_rotation @ term.RACKET_BLADE_CENTER_OFFSET_WRIST_M
        )
        blade_half = np.abs(
            np.einsum("ij,kj->ki", racket_rotation, term.RACKET_BLADE_LOCAL_HALF_AXES_M)
        ).sum(axis=0)
        return bool(
            np.any(
                np.all(
                    ((blade_centre + blade_half)[None, :] >= lo)
                    & ((blade_centre - blade_half)[None, :] <= hi),
                    axis=1,
                )
            )
        )

    rng = np.random.default_rng(20260806)
    hit_samples = 0
    retired_rule_disagreements = 0
    for trial in range(400):
        # Park the whole robot far away and graze ONE owner body across the
        # near-top corner of the table slab.  Poses drawn uniformly over the
        # court all overlap something, which never separates the two rules.
        positions = np.full((32, 3), 50.0, dtype=np.float64)
        owner = int(components.owner_indices[trial % 43])
        positions[owner] = rng.uniform((0.36, -0.30, 0.60), (0.62, 0.30, 0.92))
        quats = rng.normal(size=(32, 4))
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        rotations = np.stack(
            [
                np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
            ],
            axis=1,
        )
        port = term.geometric_robot_table_hit(
            positions, rotations, components, lo, hi, racket_body_index=31
        )
        live = bool(
            isaac._geometric_table_contact_hit_mask_unchecked(
                torch.as_tensor(positions, dtype=torch.float64)[None],
                torch.as_tensor(quats, dtype=torch.float64)[None],
                torch.zeros((1, 3), dtype=torch.float64),
                torch.as_tensor(components.owner_indices, dtype=torch.long),
                torch.as_tensor(components.local_centers_m, dtype=torch.float64),
                torch.as_tensor(components.local_half_axes_m, dtype=torch.float64),
                torch.as_tensor(lo, dtype=torch.float64),
                torch.as_tensor(hi, dtype=torch.float64),
                racket_body_index=31,
                racket_blade_center_offset_wrist_m=torch.as_tensor(
                    term.RACKET_BLADE_CENTER_OFFSET_WRIST_M, dtype=torch.float64
                ),
                racket_blade_local_half_axes_m=torch.as_tensor(
                    term.RACKET_BLADE_LOCAL_HALF_AXES_M, dtype=torch.float64
                ),
            )[0].item()
        )
        assert port == live, (
            "MuJoCo port and live Isaac terminal disagree on the same pose"
        )
        if live:
            hit_samples += 1
        if retired_broad_phase_verdict(positions, rotations) != live:
            retired_rule_disagreements += 1
    assert hit_samples > 0, "parity sample never exercised a hit"
    # Without this the parity assertion is toothless: a sample on which the two
    # rules agree everywhere would also pass against the retired broad phase.
    assert retired_rule_disagreements > 0, (
        "parity sample never separated the exact terminal from the retired "
        "world-AABB broad phase, so it cannot detect the port standing still"
    )


def test_exact_sources_artifact_and_table_geometry_are_reopened_and_pinned():
    source = term.verify_isaac_source_authority()
    assert source == {
        "config_semantic_ast_sha256": (
            term.EXPECTED_ISAAC_TERMINATION_CONFIG_SEMANTIC_AST_SHA256
        ),
        "callables_semantic_ast_sha256": (
            term.EXPECTED_ISAAC_TERMINATION_CALLABLES_SEMANTIC_AST_SHA256
        ),
        "action_latch_semantic_ast_sha256": (
            term.EXPECTED_ISAAC_ACTION_LATCH_SEMANTIC_AST_SHA256
        ),
    }
    components = term.load_collision_components()
    assert components.owner_indices.shape == (43,)
    assert components.local_centers_m.shape == (43, 3)
    assert components.local_half_axes_m.shape == (43, 3, 3)
    assert components.artifact_sha256 == term.EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256

    scene_module = single_env._load_table_scene_module()
    rows = scene_module.action_ball_policy_obstacle_geometry()
    contract = scene_module.action_ball_policy_geometry_contract(rows)
    lo, hi = term._validated_table_aabbs(contract)
    assert contract["sha256"] == term.EXPECTED_ACTION_BALL_TABLE_GEOMETRY_SHA256
    assert lo.shape == hi.shape == (5, 3)
    np.testing.assert_allclose(
        hi - lo,
        np.asarray(
            [row["full_extents_m"] for row in contract["payload"]["obstacles"]]
        )
        + 2.0 * term.TABLE_GUARD_MARGIN_M,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_source_config_drift_fails_closed(tmp_path, monkeypatch):
    drifted = tmp_path / "hope_env_cfg.py"
    source = term.ISAAC_TERMINATION_CONFIG.read_text(encoding="utf-8")
    assert "TABLE_HIT_MARGIN_M = 0.02" in source
    drifted.write_text(
        source.replace("TABLE_HIT_MARGIN_M = 0.02", "TABLE_HIT_MARGIN_M = 0.03", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(term, "ISAAC_TERMINATION_CONFIG", drifted)
    with pytest.raises(
        term.TableTerminationContractError,
        match="config semantic AST SHA-256 drifted",
    ):
        term.verify_isaac_source_authority()


def test_source_action_latch_drift_fails_closed(tmp_path, monkeypatch):
    drifted = tmp_path / "hope_actions.py"
    source = term.ISAAC_ACTION_LATCH.read_text(encoding="utf-8")
    old = "return latch.finalize(self._sample_table_contact_current())"
    assert old in source
    drifted.write_text(
        source.replace(old, "return latch.hit", 1), encoding="utf-8"
    )
    monkeypatch.setattr(term, "ISAAC_ACTION_LATCH", drifted)
    with pytest.raises(
        term.TableTerminationContractError,
        match="action-latch semantic AST SHA-256 drifted",
    ):
        term.verify_isaac_source_authority()


def test_source_table_callable_drift_fails_closed(tmp_path, monkeypatch):
    drifted = tmp_path / "terminations.py"
    source = term.ISAAC_TERMINATION_CALLABLES.read_text(encoding="utf-8")
    old = "    if require_substep_latch:\n"
    assert old in source
    drifted.write_text(
        source.replace(old, "    if False and require_substep_latch:\n", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(term, "ISAAC_TERMINATION_CALLABLES", drifted)
    with pytest.raises(
        term.TableTerminationContractError,
        match="callables semantic AST SHA-256 drifted",
    ):
        term.verify_isaac_source_authority()


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "overlap.logical_and_(separation <= obb_radius + box_radius)",
            "overlap.logical_and_(separation < obb_radius + box_radius)",
        ),
        (
            "component_exact &= valid[:, None, None]",
            "component_exact |= ~valid[:, None, None]",
        ),
        (
            "return body_hit | racket_hit | invalid_runtime",
            "return body_hit | racket_hit",
        ),
    ),
    ids=("sat", "attribution", "terminal-mask"),
)
def test_helper_only_drift_with_synchronized_self_report_fails_closed(
    tmp_path, monkeypatch, old, new
):
    source = term.ISAAC_TERMINATION_CALLABLES.read_text(encoding="utf-8")
    assert source.count(old) == 1
    drifted = tmp_path / "terminations.py"
    drifted.write_text(source.replace(old, new, 1), encoding="utf-8")

    wrapper_selector = (("function", "robot_hit_table"),)
    label = "Isaac robot/table termination callables"
    assert term._semantic_ast_sha256(
        drifted, wrapper_selector, label
    ) == term._semantic_ast_sha256(
        term.ISAAC_TERMINATION_CALLABLES, wrapper_selector, label
    )
    attacker_sha = term._semantic_ast_sha256(
        drifted, term.ISAAC_TERMINATION_CALLABLE_SELECTORS, label
    )
    with drifted.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n_TABLE_GUARD_CALLABLES_SELF_REPORTED_SHA256 = "
            f"{attacker_sha!r}\n"
        )
    assert term._semantic_ast_sha256(
        drifted, term.ISAAC_TERMINATION_CALLABLE_SELECTORS, label
    ) == attacker_sha

    monkeypatch.setattr(term, "ISAAC_TERMINATION_CALLABLES", drifted)
    with pytest.raises(
        term.TableTerminationContractError,
        match="callables semantic AST SHA-256 drifted",
    ):
        term.verify_isaac_source_authority()


def test_source_semantic_pin_ignores_unrelated_config_class_body(
    tmp_path, monkeypatch
):
    expected = term.verify_isaac_source_authority()
    source = term.ISAAC_TERMINATION_CONFIG.read_text(encoding="utf-8")
    marker = (
        'class HOPEDeployParityTerminationsCfg(TerminationsCfg):\n'
        '    """Swing-only reference envelopes plus always-on absolute fall/sink guards."""\n'
    )
    assert marker in source
    unrelated = tmp_path / "hope_env_cfg.py"
    unrelated.write_text(
        source.replace(
            marker,
            marker + "\n    unrelated_a225_leaf_marker = 1\n",
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(term, "ISAAC_TERMINATION_CONFIG", unrelated)
    assert term.verify_isaac_source_authority() == expected


def test_collision_artifact_drift_fails_closed(tmp_path, monkeypatch):
    drifted = tmp_path / "components.json"
    drifted.write_bytes(term.COLLISION_PROXY_ARTIFACT.read_bytes() + b"\n")
    monkeypatch.setattr(term, "COLLISION_PROXY_ARTIFACT", drifted)
    term._load_collision_components_cached.cache_clear()
    with pytest.raises(term.TableTerminationContractError, match="artifact SHA-256 drifted"):
        term.load_collision_components()
    term._load_collision_components_cached.cache_clear()


class _FakeMujoco:
    class mjtObj:
        mjOBJ_BODY = 1

    @staticmethod
    def mj_name2id(model, kind, name):
        assert kind == _FakeMujoco.mjtObj.mjOBJ_BODY
        return model.body_name_to_id.get(name, -1)


class _FakeOwnerFrameModel:
    def __init__(self):
        self.body_name_to_id = {
            name: index + 1 for index, name in enumerate(term.TABLE_CONTACT_BODY_NAMES)
        }
        self.body_parentid = np.zeros(33, dtype=np.int64)
        self.body_parentid[2:] = 1
        self.body_pos = np.zeros((33, 3), dtype=np.float64)
        self.body_quat = np.zeros((33, 4), dtype=np.float64)
        self.body_quat[:, 0] = 1.0


@pytest.mark.parametrize("field", ["body_pos", "body_quat"])
def test_same_named_model_with_changed_owner_local_frame_fails_closed(field):
    expected_model = _FakeOwnerFrameModel()
    observed_model = _FakeOwnerFrameModel()
    if field == "body_pos":
        observed_model.body_pos[7] = (0.001, 0.0, 0.0)
    else:
        observed_model.body_quat[7] = (0.0, 1.0, 0.0, 0.0)
    expected = term._owner_frame_contract(_FakeMujoco, expected_model)
    observed = term._owner_frame_contract(_FakeMujoco, observed_model)
    with pytest.raises(term.TableTerminationContractError, match="owner-local"):
        term._assert_owner_frame_contract_equal(expected, observed)


def test_arbitrary_root_mjcf_path_is_rejected_before_live_model_binding(tmp_path):
    arbitrary = tmp_path / "a3_pingpong.xml"
    arbitrary.write_text("<mujoco/>", encoding="utf-8")
    with pytest.raises(
        term.TableTerminationContractError, match="pre-registered root MJCF path"
    ):
        term.bind_pre_registered_owner_frames(
            _FakeMujoco, _FakeOwnerFrameModel(), arbitrary
        )


def test_plant_binding_rejects_nonfour_control_decimation():
    contract = (
        WBT_ROOT.parents[1]
        / "configs/a3_vendor_runtime_authority_20260802_r8"
        / "bh_loop_c.shared_ready.training_contract.json"
    )
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["control_decimation"] = 3
    payload["policy_step_dt_s"] = payload["physics_step_dt_s"] * 3
    with pytest.raises(single_env.ContractError, match="control_decimation=4"):
        single_env.PlantBinding.from_mapping(
            payload, source_path=str(contract), source_sha256="0" * 64
        )


def test_real_mujoco_guard_constructs_and_samples_current_pose():
    pytest.importorskip("mujoco")
    contract = (
        WBT_ROOT.parents[1]
        / "configs/a3_vendor_runtime_authority_20260802_r8"
        / "bh_loop_c.shared_ready.training_contract.json"
    )
    binding = single_env.load_plant_binding(contract)
    env = single_env.MujocoSingleEnv(binding)
    tape_payload = single_env.build_probe_tape(binding, delay_steps=0)
    tape = single_env.FixedTape.from_mapping(
        tape_payload,
        source_path="synthetic",
        source_sha256=hashlib.sha256(b"synthetic").hexdigest(),
        binding=binding,
    )
    env.reset(
        reset_state=tape.reset_state,
        delay_steps=tape.delay_steps,
        history_fill_action=tape.history_fill_action,
    )
    assert type(env._robot_table_guard.sample(env.data)) is bool
    row = env.step(np.zeros(31, dtype=np.float64))
    assert type(row["robot_hit_table_substep"]) is bool
    if row["robot_hit_table_substep"]:
        assert 0 <= row["robot_hit_table_first_substep"] < binding.control_decimation
    else:
        assert row["robot_hit_table_first_substep"] is None
