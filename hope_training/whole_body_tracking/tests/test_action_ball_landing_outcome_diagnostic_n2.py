from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import inspect
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
_MDP_ROOT = (
    _SOURCE_ROOT
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
_R06_PATH = _MDP_ROOT / "action_ball_landing_outcome_device.py"
for _path in (_SOURCE_ROOT, _MDP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


if sys.version_info < (3, 10) and "action_ball_full_mdp_reset_genesis" not in sys.modules:
    _genesis_path = _SOURCE_ROOT / "action_ball_full_mdp_reset_genesis.py"
    _genesis_source = _genesis_path.read_text(encoding="utf-8").replace(
        ", slots=True", ""
    ).replace("@dataclass(slots=True)", "@dataclass")
    _genesis_module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(
            "action_ball_full_mdp_reset_genesis", loader=None
        )
    )
    _genesis_module.__file__ = str(_genesis_path)
    sys.modules["action_ball_full_mdp_reset_genesis"] = _genesis_module
    exec(
        compile(_genesis_source, str(_genesis_path), "exec"),
        _genesis_module.__dict__,
    )

import action_ball_full_mdp_diagnostic_capacity as CAP


def _install_namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


for _name, _path in (
    ("whole_body_tracking", _SOURCE_ROOT / "whole_body_tracking"),
    ("whole_body_tracking.tasks", _SOURCE_ROOT / "whole_body_tracking" / "tasks"),
    (
        "whole_body_tracking.tasks.tracking",
        _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking",
    ),
    (
        "whole_body_tracking.tasks.tracking.config",
        _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking" / "config",
    ),
    (
        "whole_body_tracking.tasks.tracking.config.agibot_a3",
        _SOURCE_ROOT
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3",
    ),
    ("whole_body_tracking.tasks.tracking.mdp", _MDP_ROOT),
):
    _install_namespace(_name, _path)


def _load_r06():
    name = "_focused_action_ball_landing_outcome_diagnostic_n2"
    retained = sys.modules.get(name)
    if retained is not None and Path(retained.__file__).resolve() == _R06_PATH:
        return retained
    spec = importlib.util.spec_from_file_location(name, _R06_PATH)
    assert spec is not None and spec.loader is not None
    dependency_path = _MDP_ROOT / "action_ball_landing_placement_torch.py"
    dependency_name = (
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_landing_placement_torch"
    )
    dependency_spec = importlib.util.spec_from_file_location(
        dependency_name, dependency_path
    )
    assert dependency_spec is not None and dependency_spec.loader is not None
    dependency = importlib.util.module_from_spec(dependency_spec)
    sys.modules[dependency_name] = dependency
    dependency_spec.loader.exec_module(dependency)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R06 = _load_r06()
_CUDA_DEVICE = (
    "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") == "2" else "cuda:2"
)


@pytest.fixture
def real_env_cfg_module():
    if os.environ.get("ACTIONBALL_R06_REAL_ENV_CFG_TEST") != "1":
        pytest.skip("requires the Isaac SimulationApp test wrapper")
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg

    return hope_env_cfg


def _env(H, family: str, device: str):
    cfg_type = (
        H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg
        if family == "A"
        else H.HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg
    )
    cfg = cfg_type()
    return SimpleNamespace(
        cfg=cfg,
        num_envs=2,
        device=device,
        full_mdp_cold_restore_dormant=False,
        _action_ball_r10_cold_restore_capsule=None,
    )


class _Asset:
    def __init__(self, root: torch.Tensor):
        self.data = SimpleNamespace(root_state_w=root.clone())

    def write_root_pose_to_sim(self, value, env_ids=None):
        assert env_ids is not None
        self.data.root_state_w[env_ids, :7] = value

    def write_root_velocity_to_sim(self, value, env_ids=None):
        assert env_ids is not None
        self.data.root_state_w[env_ids, 7:] = value


class _ReplicatedScene(dict):
    """Minimal InteractiveScene facts required by the concrete scene port."""

    def __init__(self, *, num_envs: int):
        super().__init__()
        self.cfg = SimpleNamespace(replicate_physics=True)
        self.env_prim_paths = [
            f"/World/envs/env_{index}" for index in range(num_envs)
        ]


def _physical_owner(scene_module, physical_module, spec, binding, device):
    exact_device = torch.device(device)
    origins = torch.zeros((2, 3), dtype=torch.float32, device=exact_device)
    scene = _ReplicatedScene(num_envs=2)
    for name in spec.scene_entity_names:
        root = torch.zeros((2, 13), dtype=torch.float32, device=exact_device)
        root[:, 2] = scene_module.PARK_POSITION_ENV_M[2]
        root[:, 3] = 1.0
        scene[name] = _Asset(root)
    port = scene_module.IsaacLabPhysicalFlightScenePort(
        scene=scene,
        spec=spec,
        env_origins=origins,
    )
    genesis_module = physical_module._reset_genesis
    issue = genesis_module.issue_action_ball_full_mdp_reset_genesis(
        num_envs=2,
        device=exact_device,
    )
    owner = physical_module.ActionBallPhysicalFlightDeviceOwner(
        num_envs=2,
        scene_body_names=spec.scene_entity_names,
        scene_port=port,
        diagnostic_n2_capacity_binding=binding,
        reset_genesis_authority=issue.authority,
        reset_genesis_receipt=issue.receipt,
    )
    return owner


def _canonical_cold_cardinality_graph():
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
        action_ball_full_mdp_ball_scene as scene_module,
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_physical_flight_device as physical_module,
    )

    capacity = CAP.DIAGNOSTIC_FLIGHT_CAPACITY
    spec = scene_module.ActionBallFullMdpDiagnosticBallSceneSpec(
        schema_version=scene_module.SCHEMA_VERSION,
        kind=scene_module.DIAGNOSTIC_SCENE_SPEC_KIND,
        capacity_authority_kind=CAP.DIAGNOSTIC_CAPACITY_KIND,
        formal_capacity_receipt_sha256=None,
        flight_capacity=capacity,
        scene_entity_names=tuple(
            f"{scene_module.SCENE_ENTITY_PREFIX}{index:03d}"
            for index in range(capacity)
        ),
        prim_paths=tuple(
            f"{{ENV_REGEX_NS}}/{scene_module.SCENE_PRIM_PREFIX}{index:03d}"
            for index in range(capacity)
        ),
        ball_radius_m=0.02,
        ball_mass_kg=0.0034,
        park_position_env_m=scene_module.PARK_POSITION_ENV_M,
        collision_enabled=True,
        gravity_enabled=True,
    )
    binding = CAP.construct_diagnostic_n2_capacity_binding(spec)
    physical = _physical_owner(
        scene_module, physical_module, spec, binding, "cpu"
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=SimpleNamespace(env_origins=physical.scene_port.env_origins),
    )
    return env, physical


@pytest.mark.parametrize("family,gain", [("A", 1.0), ("C", 0.0)])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            _CUDA_DEVICE,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or torch.cuda.device_count()
                < (1 if _CUDA_DEVICE == "cuda:0" else 3),
                reason="physical CUDA:2 unavailable",
            ),
        ),
    ],
)
def test_real_r06_diagnostic_n2_allocates_only_and_holds_formal_surfaces(
    family, gain, device, real_env_cfg_module
):
    env = _env(real_env_cfg_module, family, device)
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
        action_ball_full_mdp_ball_scene as scene_module,
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_physical_flight_device as physical_module,
    )

    spec = env.cfg.action_ball_full_mdp_ball_scene_spec
    binding = CAP.construct_diagnostic_n2_capacity_binding(spec)
    physical = _physical_owner(
        scene_module, physical_module, spec, binding, device
    )
    owner = R06.construct_diagnostic_n2_no_save_r06(
        env=env,
        physical_owner=physical,
        diagnostic_n2_capacity_binding=binding,
    )

    assert type(owner) is R06.ActionBallLandingOutcomeDeviceCoordinator
    assert owner.num_envs == 2
    assert owner.flight_slot_capacity == 2
    assert owner.mailbox_capacity == 2
    assert owner.device == torch.device(device)
    assert owner.dtype is torch.float32
    assert owner.flight_state.shape == (2, 2)
    assert owner.mailbox_state.shape == (2, 2)
    assert owner._placement_treatment_gain == gain
    assert owner._diagnostic_family_gain_identity_sha256 != "0" * 64
    assert torch.count_nonzero(owner._c10_projection_token).item() == 0
    assert owner.diagnostic_n2_no_save is True
    assert owner.diagnostic_unauthorized is True
    for formal_name in ("runtime_binding", "payment_authority", "capacity_authority"):
        assert not hasattr(owner, formal_name)

    namespace = owner.diagnostic_n2_no_save_namespace_projection()
    namespace_view = owner.require_owned_diagnostic_n2_no_save_namespace(namespace)
    assert namespace_view.family == family
    assert namespace_view.run_id == R06.DIAGNOSTIC_N2_NO_SAVE_RUN_IDS[family]
    assert (
        namespace_view.carry_chain_id
        == R06.DIAGNOSTIC_N2_NO_SAVE_CARRY_CHAIN_IDS[family]
    )
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(namespace)
    with pytest.raises(TypeError, match="cannot be serialized"):
        namespace.__reduce__()

    for call, match in (
        (
            lambda: owner.prepare_from_reveal_final_preview(
                object(), expected_reveal_final_preview_sha256="0" * 64
            ),
            "missing formal H/C authority",
        ),
        (lambda: owner.project_checkpoint_live_mutation(), "R10"),
        (lambda: owner.state_dict(object()), "export a checkpoint"),
        (
            lambda: owner.load_state_dict(
                {}, expected_checkpoint_content_sha256="0" * 64
            ),
            "restore a checkpoint",
        ),
        (lambda: owner.__getstate__(), "cannot be exported"),
    ):
        with pytest.raises((R06.LandingOutcomeDeviceError, TypeError), match=match):
            call()


def test_constructor_has_no_caller_shape_dtype_or_family_numeric_authority():
    parameters = inspect.signature(
        R06.construct_diagnostic_n2_no_save_r06
    ).parameters
    assert tuple(parameters) == (
        "env",
        "physical_owner",
        "diagnostic_n2_capacity_binding",
    )
    for forbidden in ("num_envs", "device", "dtype", "family", "gain"):
        assert forbidden not in parameters


def test_redteam_rejects_foreign_scene_shape_and_field_only_family(
    real_env_cfg_module,
):
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
        action_ball_full_mdp_ball_scene as scene_module,
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_physical_flight_device as physical_module,
    )

    env = _env(real_env_cfg_module, "A", "cpu")
    spec = env.cfg.action_ball_full_mdp_ball_scene_spec
    binding = CAP.construct_diagnostic_n2_capacity_binding(spec)
    physical = _physical_owner(
        scene_module, physical_module, spec, binding, "cpu"
    )
    env.num_envs = 3
    with pytest.raises(R06.LandingOutcomeDeviceError, match="exact N=2"):
        R06.construct_diagnostic_n2_no_save_r06(
            env=env,
            physical_owner=physical,
            diagnostic_n2_capacity_binding=binding,
        )

    fake_cfg = SimpleNamespace(
        action_ball_full_mdp_family_role="A",
        action_ball_full_mdp_ball_scene_spec=spec,
    )
    fake_env = SimpleNamespace(cfg=fake_cfg, num_envs=2, device="cpu")
    with pytest.raises(R06.LandingOutcomeDeviceError, match="exact registered"):
        R06.construct_diagnostic_n2_no_save_r06(
            env=fake_env,
            physical_owner=physical,
            diagnostic_n2_capacity_binding=binding,
        )

    foreign_env = _env(real_env_cfg_module, "A", "cpu")
    foreign_spec = foreign_env.cfg.action_ball_full_mdp_ball_scene_spec
    foreign_binding = CAP.construct_diagnostic_n2_capacity_binding(foreign_spec)
    env = _env(real_env_cfg_module, "A", "cpu")
    with pytest.raises(R06.LandingOutcomeDeviceError, match="stale or foreign"):
        R06.construct_diagnostic_n2_no_save_r06(
            env=env,
            physical_owner=physical,
            diagnostic_n2_capacity_binding=foreign_binding,
        )


def test_live_cardinality_rejects_env_device_mismatch_before_registry_or_install():
    env, physical = _canonical_cold_cardinality_graph()
    env.device = "meta"
    pending_before = dict(R06._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS)

    with pytest.raises(
        R06.LandingOutcomeDeviceError,
        match="Env/Physical/scene device differs",
    ):
        R06._diagnostic_live_cardinality_anchors(
            env=env,
            physical_owner=physical,
        )

    assert dict(R06._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS) == pending_before
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")


def test_live_cardinality_rejects_same_valued_foreign_scene_capability():
    env, physical = _canonical_cold_cardinality_graph()
    scene_port = physical.scene_port
    original = scene_port._scene_port_capability
    foreign = copy.copy(original)
    assert foreign is not original
    for name in (
        "num_envs",
        "flight_capacity",
        "device_type",
        "device_index",
        "scene_spec_sha256",
        "_port_identity",
        "_token",
    ):
        assert getattr(foreign, name) == getattr(original, name)
    scene_port._scene_port_capability = foreign
    assert physical._scene_port_capability is original
    pending_before = dict(R06._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS)

    with pytest.raises(
        R06.LandingOutcomeDeviceError,
        match="Physical retained scene capability differs",
    ):
        R06._diagnostic_live_cardinality_anchors(
            env=env,
            physical_owner=physical,
        )

    assert dict(R06._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS) == pending_before
    assert not hasattr(env, "_action_ball_full_mdp_reset_genesis_install")
    assert not hasattr(env, "_action_ball_full_mdp_components")


def test_parallel_diagnostic_construction_is_registry_isolated(
    real_env_cfg_module,
):
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
        action_ball_full_mdp_ball_scene as scene_module,
    )
    from whole_body_tracking.tasks.tracking.mdp import (
        action_ball_physical_flight_device as physical_module,
    )

    inputs = []
    for family in ("A", "C", "A", "C"):
        env = _env(real_env_cfg_module, family, "cpu")
        spec = env.cfg.action_ball_full_mdp_ball_scene_spec
        binding = CAP.construct_diagnostic_n2_capacity_binding(spec)
        physical = _physical_owner(
            scene_module, physical_module, spec, binding, "cpu"
        )
        inputs.append((env, physical, binding, family))

    def construct(values):
        env, physical, binding, family = values
        owner = R06.construct_diagnostic_n2_no_save_r06(
            env=env,
            physical_owner=physical,
            diagnostic_n2_capacity_binding=binding,
        )
        projection = owner.diagnostic_n2_no_save_namespace_projection()
        return owner.require_owned_diagnostic_n2_no_save_namespace(
            projection
        ).family

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert tuple(pool.map(construct, inputs)) == ("A", "C", "A", "C")
    assert not R06._DIAGNOSTIC_N2_PENDING_CONSTRUCTIONS
