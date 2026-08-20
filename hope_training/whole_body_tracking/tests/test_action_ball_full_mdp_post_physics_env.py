"""Fresh Isaac post-physics seam regression.

The production module imports IsaacLab, but these behavioral tests substitute a
minimal base class so they can exercise the copied step without constructing a
Kit scene.  The source-pin test separately reads the exact Pod1 IsaacLab tree.
Run this file only with the Pod1 training interpreter and CUDA hidden.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "full_mdp_env.py"
)


def _load_subject():
    subject_name = "_action_ball_full_mdp_env_test_subject"
    if subject_name in sys.modules:
        return sys.modules[subject_name]

    class FakeManagerBasedRLEnv:
        def __init__(self, *args, **kwargs):
            raise AssertionError("focused tests must not construct a Kit env")

        def close(self):
            pass

    class FakeManagerBasedRLEnvCfg:
        pass

    stubs = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab.envs.common": types.ModuleType("isaaclab.envs.common"),
        "isaaclab.envs.manager_based_rl_env": types.ModuleType(
            "isaaclab.envs.manager_based_rl_env"
        ),
        "isaaclab.envs.manager_based_rl_env_cfg": types.ModuleType(
            "isaaclab.envs.manager_based_rl_env_cfg"
        ),
    }
    stubs["isaaclab"].__path__ = []
    stubs["isaaclab.envs"].__path__ = []
    stubs["isaaclab.envs.common"].VecEnvStepReturn = tuple
    stubs["isaaclab.envs.manager_based_rl_env"].ManagerBasedRLEnv = (
        FakeManagerBasedRLEnv
    )
    stubs["isaaclab.envs.manager_based_rl_env_cfg"].ManagerBasedRLEnvCfg = (
        FakeManagerBasedRLEnvCfg
    )

    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(subject_name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[subject_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


M = _load_subject()


def _pinned_isaaclab_source() -> Path:
    candidates: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        candidate = (
            Path(entry)
            / "isaaclab"
            / "envs"
            / "manager_based_rl_env.py"
        )
        if candidate.is_file():
            candidates.append(candidate.resolve())
    build2 = Path(
        "/workspace/IsaacLab-8320e0be/source/isaaclab/isaaclab/envs/"
        "manager_based_rl_env.py"
    )
    if build2.is_file():
        candidates.append(build2.resolve())
    unique = tuple(dict.fromkeys(candidates))
    if not unique:
        pytest.skip("pinned build_2 IsaacLab source is unavailable")
    assert len(unique) == 1, f"ambiguous IsaacLab sources: {unique!r}"
    return unique[0]


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_name(node: ast.AST) -> str:
    return _dotted_name(node.func) if isinstance(node, ast.Call) else ""


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    assert len(methods) == 1
    return methods[0]


def _statement_call_index(statements: list[ast.stmt], name: str) -> int:
    matches = [
        index
        for index, statement in enumerate(statements)
        if any(
            isinstance(node, ast.Call) and _call_name(node) == name
            for node in ast.walk(statement)
        )
    ]
    assert len(matches) == 1, (name, matches)
    return matches[0]


def test_exact_build2_upstream_source_pin_and_reorder_counterexample():
    source_path = _pinned_isaaclab_source()
    source_bytes = source_path.read_bytes()
    M._validate_pinned_upstream_source_bytes(source_bytes)

    tree = ast.parse(source_bytes.decode("utf-8"))
    step = _class_method(tree, "ManagerBasedRLEnv", "step")
    loop = next(statement for statement in step.body if isinstance(statement, ast.For))
    sim_index = _statement_call_index(loop.body, "self.sim.step")
    update_index = _statement_call_index(loop.body, "self.scene.update")
    loop.body[sim_index], loop.body[update_index] = (
        loop.body[update_index],
        loop.body[sim_index],
    )
    ast.fix_missing_locations(tree)
    with pytest.raises(
        M.FullMdpUpstreamSourceDriftError,
        match="physics order|final upstream",
    ):
        M._assert_pinned_upstream_step_order(ast.unparse(tree))


def test_local_step_has_the_only_legal_post_update_pre_manager_hook():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    step = _class_method(
        tree, "ActionBallFullMdpManagerBasedRLEnv", "step"
    )
    loop = next(
        statement
        for statement in ast.walk(step)
        if isinstance(statement, ast.For)
        and any(
            isinstance(node, ast.Call) and _call_name(node) == "range"
            for node in ast.walk(statement.iter)
        )
    )
    apply_index = _statement_call_index(
        loop.body, "self.action_manager.apply_action"
    )
    write_index = _statement_call_index(loop.body, "self.scene.write_data_to_sim")
    sim_index = _statement_call_index(loop.body, "self.sim.step")
    recorder_index = _statement_call_index(
        loop.body, "self.recorder_manager.record_post_physics_decimation_step"
    )
    render_index = _statement_call_index(loop.body, "self.sim.render")
    update_index = _statement_call_index(loop.body, "self.scene.update")
    hook_index = _statement_call_index(
        loop.body, "self._publish_post_physics_substep"
    )
    assert (
        apply_index
        < write_index
        < sim_index
        < recorder_index
        < render_index
        < update_index
    )
    assert hook_index == update_index + 1 == len(loop.body) - 1

    common = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.AugAssign)
        and _dotted_name(node.target) == "self.common_step_counter"
    )
    termination = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and _call_name(node) == "self.termination_manager.compute"
    )
    reward = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and _call_name(node) == "self.reward_manager.compute"
    )
    update = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call) and _call_name(node) == "self.scene.update"
    )
    hook = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and _call_name(node) == "self._publish_post_physics_substep"
    )
    assert update.lineno < hook.lineno < common.lineno
    assert common.lineno < termination.lineno < reward.lineno

    calls = {
        _call_name(node)
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
    }
    assert not any(name.endswith("add_physics_callback") for name in calls)

    seam_methods = (
        "_protected_manager_state",
        "_assert_protected_manager_state_unchanged",
        "_publish_post_physics_substep",
        "step",
    )
    forbidden_suffixes = (
        ".item",
        ".cpu",
        ".numpy",
        ".tolist",
        ".synchronize",
        ".get_rng_state",
        ".manual_seed",
        ".seed",
        ".rand",
        ".randn",
        ".randint",
    )
    for method_name in seam_methods:
        method = _class_method(
            tree, "ActionBallFullMdpManagerBasedRLEnv", method_name
        )
        method_calls = {
            _call_name(node)
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
        }
        assert not any(
            name.endswith(suffix)
            for name in method_calls
            for suffix in forbidden_suffixes
        )


def test_cold_local_binding_rejects_same_valued_method_replacement(monkeypatch):
    owner_type = M.ActionBallFullMdpManagerBasedRLEnv
    original = owner_type.step
    foreign = types.FunctionType(
        original.__code__,
        original.__globals__,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    foreign.__module__ = original.__module__
    foreign.__qualname__ = original.__qualname__
    foreign.__annotations__ = dict(original.__annotations__)
    monkeypatch.setattr(owner_type, "step", foreign)
    with pytest.raises(
        M.FullMdpUpstreamSourceDriftError,
        match="method 'step' was replaced",
    ):
        M._assert_runtime_uses_pinned_local_step()


@pytest.mark.parametrize("method_name", ("step", "__init__"))
def test_cold_local_binding_rejects_in_place_code_replacement(method_name):
    owner_type = M.ActionBallFullMdpManagerBasedRLEnv
    method = vars(owner_type)[method_name]
    original_code = method.__code__
    constants = list(original_code.co_consts)
    string_index = next(
        index
        for index, value in enumerate(constants)
        if type(value) is str
    )
    constants[string_index] = constants[string_index] + " [foreign]"
    foreign_code = original_code.replace(co_consts=tuple(constants))
    assert foreign_code.co_filename == original_code.co_filename
    method.__code__ = foreign_code
    try:
        with pytest.raises(
            M.FullMdpUpstreamSourceDriftError,
            match=rf"method '{method_name}' was replaced",
        ):
            M._assert_runtime_uses_pinned_local_step()
    finally:
        method.__code__ = original_code
    M._assert_runtime_uses_pinned_local_step()


def test_cold_local_binding_rejects_in_place_init_kwdefault_mutation():
    constructor = M.ActionBallFullMdpManagerBasedRLEnv.__init__
    keyword_defaults = constructor.__kwdefaults__
    assert keyword_defaults is not None
    original = dict(keyword_defaults)
    keyword_defaults["full_mdp_cold_restore_dormant"] = True
    try:
        with pytest.raises(
            M.FullMdpUpstreamSourceDriftError,
            match="method '__init__' was replaced",
        ):
            M._assert_runtime_uses_pinned_local_step()
    finally:
        keyword_defaults.clear()
        keyword_defaults.update(original)
    M._assert_runtime_uses_pinned_local_step()


def test_cold_local_binding_rejects_instance_executable_shadow():
    owner_type = M.ActionBallFullMdpManagerBasedRLEnv
    env = object.__new__(owner_type)
    env.step = types.MethodType(owner_type.step, env)
    with pytest.raises(
        M.FullMdpUpstreamSourceDriftError,
        match="instance overrides cold-bound executable names: step",
    ):
        M._assert_runtime_uses_pinned_local_step(env)


def test_cold_local_binding_rejects_same_valued_foreign_export(monkeypatch):
    owner_type = M.ActionBallFullMdpManagerBasedRLEnv
    namespace = {
        name: value
        for name, value in vars(owner_type).items()
        if name not in {"__dict__", "__weakref__"}
    }
    foreign_type = type(owner_type.__name__, owner_type.__bases__, namespace)
    assert foreign_type.step is owner_type.step
    assert foreign_type.__module__ == owner_type.__module__
    monkeypatch.setattr(
        M, "ActionBallFullMdpManagerBasedRLEnv", foreign_type
    )
    with pytest.raises(
        M.FullMdpUpstreamSourceDriftError,
        match="not its cold-bound module export",
    ):
        M._assert_runtime_uses_pinned_local_step()


def test_cold_local_binding_rejects_module_and_property_replacement(monkeypatch):
    replacement_module = types.ModuleType(M.__name__)
    replacement_module.__dict__.update(vars(M))
    monkeypatch.setitem(sys.modules, M.__name__, replacement_module)
    with pytest.raises(
        M.FullMdpUpstreamSourceDriftError,
        match="not its cold-bound module export",
    ):
        M._assert_runtime_uses_pinned_local_step()
    monkeypatch.setitem(sys.modules, M.__name__, M)

    owner_type = M.ActionBallFullMdpManagerBasedRLEnv
    original = owner_type.full_mdp_runtime_owner
    monkeypatch.setattr(
        owner_type,
        "full_mdp_runtime_owner",
        property(original.fget),
    )
    with pytest.raises(
        M.FullMdpUpstreamSourceDriftError,
        match="getter 'full_mdp_runtime_owner' was replaced",
    ):
        M._assert_runtime_uses_pinned_local_step()


def test_missing_owner_is_rejected_before_base_construction():
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError,
        match="requires one post-physics owner",
    ):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg=None,
            full_mdp_post_physics_owner_factory=None,
        )


def test_unfrozen_concrete_owner_and_extension_mode_fail_before_base_construction(
    monkeypatch,
):
    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError, match="remains HOLD"
    ):
        M._frozen_concrete_owner_pins()
    config_module = types.ModuleType(M.FULL_MDP_CONFIG_MODULE)
    for name, _role, _task_id in M.FULL_MDP_DIAGNOSTIC_CONFIG_TYPES:
        setattr(config_module, name, type(name, (), {}))
    for name in (
        "gymnasium",
        "action_ball_full_mdp_canary_target_profile",
        M.FULL_MDP_RUNTIME_FACTORY_MODULE,
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, M.FULL_MDP_CONFIG_MODULE, config_module
    )
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    factory_calls = []

    def factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return None

    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError,
        match="exact registered A/C full-MDP EnvCfg type",
    ):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg=None,
            full_mdp_post_physics_owner_factory=factory,
            full_mdp_post_physics_expected_dependency_dag_sha256="a" * 64,
        )
    assert factory_calls == []
    assert "_action_ball_full_mdp_runtime_lease" not in vars(env)
    assert not any(
        name.startswith("_action_ball_full_mdp_manager_construction")
        or name.startswith("_action_ball_full_mdp_base_construction")
        for name in vars(env)
    )
    monkeypatch.setattr(
        M,
        "PINNED_FULL_MDP_OWNER_MODULE",
        "fake",
    )
    with pytest.raises(
        M.FullMdpPostPhysicsOwnerMissingError, match="partially frozen"
    ):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            env,
            cfg=None,
            full_mdp_post_physics_owner_factory=factory,
            full_mdp_post_physics_expected_dependency_dag_sha256="c" * 64,
        )
    assert factory_calls == []
    assert vars(env) == {}
    monkeypatch.setattr(
        M.builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", True, raising=False
    )
    with pytest.raises(M.FullMdpUnsupportedRuntimeError, match="extension-mode"):
        M._require_standalone_simulation_app()
    monkeypatch.setattr(M.builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", False)
    M._require_standalone_simulation_app()


def _concrete_owner_fixture_source(*, stale_variant: bool = False) -> str:
    publish_body = (
        "        marker = stamp\n        del marker\n        return None"
        if stale_variant
        else "        return None"
    )
    return f'''class ConcreteOwner:
    def __init__(self, env, lease, dependency_dag):
        self._env = env
        self._lease = lease
        self._dependency_dag = dependency_dag

    @property
    def full_mdp_post_physics_dependency_dag_sha256(self):
        return self._dependency_dag

    @property
    def full_mdp_post_physics_env(self):
        return self._env

    @property
    def full_mdp_post_physics_lease(self):
        return self._lease

    def publish_post_physics_substep(self, stamp):
{{publish_body}}

    def alternate_post_physics_substep(self, stamp):
        return stamp
'''.format(publish_body=publish_body)


def _load_concrete_owner_fixture(tmp_path, monkeypatch):
    module_name = f"_full_mdp_concrete_owner_{tmp_path.name.replace('-', '_')}"
    source_path = tmp_path / "concrete_owner.py"
    source_path.write_text(
        _concrete_owner_fixture_source(), encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, source_path


def _concrete_owner_fixture_pins(module, source_path, dependency_dag):
    source_bytes = source_path.read_bytes()
    return (
        module.__name__,
        "ConcreteOwner",
        M.hashlib.sha256(source_bytes).hexdigest(),
        M._concrete_owner_class_ast_sha256(source_bytes, "ConcreteOwner"),
        dependency_dag,
    )


def test_construction_binds_loaded_module_class_executable_and_dependency_dag(
    tmp_path, monkeypatch
):
    module, source_path = _load_concrete_owner_fixture(tmp_path, monkeypatch)
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    lease = object()
    dependency_dag = "d" * 64
    owner = module.ConcreteOwner(env, lease, dependency_dag)
    pins = _concrete_owner_fixture_pins(
        module, source_path, dependency_dag
    )

    binding = env._validate_concrete_owner_install(
        owner,
        concrete_pins=pins,
        expected_dependency_dag=dependency_dag,
        expected_lease=lease,
    )

    assert binding.module_object is module
    assert binding.owner_type is module.ConcreteOwner
    assert binding.publish_function is vars(module.ConcreteOwner)[
        "publish_post_physics_substep"
    ]
    assert binding.direct_executable_sha256 == (
        M._live_owner_direct_executable_sha256(module.ConcreteOwner)
    )


def test_disk_repin_cannot_authorize_a_different_already_loaded_executable(
    tmp_path, monkeypatch
):
    module, source_path = _load_concrete_owner_fixture(tmp_path, monkeypatch)
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    lease = object()
    dependency_dag = "d" * 64
    owner = module.ConcreteOwner(env, lease, dependency_dag)

    source_path.write_text(
        _concrete_owner_fixture_source(stale_variant=True), encoding="utf-8"
    )
    pins = _concrete_owner_fixture_pins(
        module, source_path, dependency_dag
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError,
        match="loaded concrete owner executable differs",
    ):
        env._validate_concrete_owner_install(
            owner,
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )


def test_module_export_or_instance_publish_shadow_cannot_spoof_concrete_type(
    tmp_path, monkeypatch
):
    module, source_path = _load_concrete_owner_fixture(tmp_path, monkeypatch)
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    lease = object()
    dependency_dag = "d" * 64
    owner_type = module.ConcreteOwner
    owner = owner_type(env, lease, dependency_dag)
    pins = _concrete_owner_fixture_pins(
        module, source_path, dependency_dag
    )

    module.ConcreteOwner = object
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="pinned module export"
    ):
        env._validate_concrete_owner_install(
            owner,
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )

    module.ConcreteOwner = owner_type
    owner.publish_post_physics_substep = lambda stamp: None
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="shadowed or not bound"
    ):
        env._validate_concrete_owner_install(
            owner,
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )


def test_keyed_manifest_rejects_method_swap_foreign_globals_and_defaults(
    tmp_path, monkeypatch
):
    module, source_path = _load_concrete_owner_fixture(tmp_path, monkeypatch)
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    lease = object()
    dependency_dag = "d" * 64
    owner_type = module.ConcreteOwner
    pins = _concrete_owner_fixture_pins(
        module, source_path, dependency_dag
    )

    publish = owner_type.publish_post_physics_substep
    alternate = owner_type.alternate_post_physics_substep
    owner_type.publish_post_physics_substep = alternate
    owner_type.alternate_post_physics_substep = publish
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="foreign identity"
    ):
        env._validate_concrete_owner_install(
            owner_type(env, lease, dependency_dag),
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )

    owner_type.publish_post_physics_substep = publish
    owner_type.alternate_post_physics_substep = alternate
    foreign = types.FunctionType(
        publish.__code__,
        dict(publish.__globals__),
        name=publish.__name__,
    )
    foreign.__module__ = publish.__module__
    foreign.__qualname__ = publish.__qualname__
    owner_type.publish_post_physics_substep = foreign
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="foreign globals"
    ):
        env._validate_concrete_owner_install(
            owner_type(env, lease, dependency_dag),
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )

    owner_type.publish_post_physics_substep = publish
    publish.__defaults__ = (None,)
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="unpinned callable state"
    ):
        env._validate_concrete_owner_install(
            owner_type(env, lease, dependency_dag),
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )
    publish.__defaults__ = None


def test_subclass_and_loaded_local_helper_replacement_are_rejected():
    class Subclass(M.ActionBallFullMdpManagerBasedRLEnv):
        pass

    subclass = object.__new__(Subclass)
    with pytest.raises(
        M.FullMdpUnsupportedRuntimeError, match="rejects subclasses"
    ):
        M.ActionBallFullMdpManagerBasedRLEnv.__init__(
            subclass,
            cfg=None,
            full_mdp_post_physics_owner_factory=None,
        )

    owner_type = M.ActionBallFullMdpManagerBasedRLEnv
    original = owner_type._poison

    def replacement(self, *, reason, exact_stamp):
        return None

    owner_type._poison = replacement
    try:
        with pytest.raises(
            M.FullMdpUpstreamSourceDriftError,
            match="method '_poison' was replaced",
        ):
            M._assert_runtime_uses_pinned_local_step()
    finally:
        owner_type._poison = original


def test_stamp_is_exact_immutable_and_integer_only():
    stamp = M.FullMdpPhysicsSubstepStamp(
        control_step=7,
        physics_substep=2,
        physics_substeps_per_control=4,
        sim_step=43,
        event_phase=M.FullMdpPhysicsEventPhase.POST_SCENE_UPDATE,
    )
    assert stamp.exact_tuple() == (7, 2, 4, 43, 1)
    with pytest.raises(AttributeError):
        object.__setattr__(stamp, "sim_step", 44)
    dispatch = M._ControlStepDispatch(
        control_step=1, decimation=1, sim_step_before=0
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="plain integers"
    ):
        dispatch.prepare(physics_substep=True, sim_step=1)


def test_dispatch_rejects_pre_step_skipped_and_duplicated_final_substeps():
    pre_step = M._ControlStepDispatch(
        control_step=1, decimation=2, sim_step_before=10
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="expected sim step"
    ):
        pre_step.prepare(physics_substep=0, sim_step=10)

    skipped = M._ControlStepDispatch(
        control_step=1, decimation=2, sim_step_before=10
    )
    first = skipped.prepare(physics_substep=0, sim_step=11)
    skipped.commit(first)
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="final physics substep"
    ):
        skipped.finish()

    complete = M._ControlStepDispatch(
        control_step=1, decimation=2, sim_step_before=10
    )
    first = complete.prepare(physics_substep=0, sim_step=11)
    complete.commit(first)
    final = complete.prepare(physics_substep=1, sim_step=12)
    complete.commit(final)
    complete.finish()
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="skipped, duplicated or reordered"
    ):
        complete.prepare(physics_substep=1, sim_step=12)


class _ActionManager:
    def __init__(self, trace):
        self.trace = trace

    def process_action(self, action):
        self.trace.append(("process", tuple(action.shape)))

    def apply_action(self):
        self.trace.append(("apply",))


class _Scene:
    def __init__(self, trace):
        self.trace = trace

    def write_data_to_sim(self):
        self.trace.append(("write",))

    def update(self, *, dt):
        self.trace.append(("update", dt))


class _Simulation:
    def __init__(self, trace, *, rendering):
        self.trace = trace
        self.rendering = rendering

    def has_gui(self):
        return self.rendering

    def has_rtx_sensors(self):
        return False

    def step(self, *, render):
        assert render is False
        self.trace.append(("sim",))

    def render(self):
        self.trace.append(("render",))

    def forward(self):
        self.trace.append(("forward",))


class _RecorderManager:
    active_terms = ()

    def __init__(self, trace):
        self.trace = trace

    def record_pre_step(self):
        self.trace.append(("pre_record",))

    def record_post_physics_decimation_step(self):
        self.trace.append(("post_physics_record",))

    def record_post_step(self):
        self.trace.append(("post_record",))

    def record_pre_reset(self, env_ids):
        self.trace.append(("pre_reset", tuple(env_ids.shape)))

    def record_post_reset(self, env_ids):
        self.trace.append(("post_reset", tuple(env_ids.shape)))


class _TerminationManager:
    def __init__(self, trace, count):
        self.trace = trace
        self._terminated = torch.zeros(count, dtype=torch.bool)
        self._time_outs = torch.zeros(count, dtype=torch.bool)

    @property
    def terminated(self):
        return self._terminated

    @property
    def time_outs(self):
        return self._time_outs

    def compute(self):
        self.trace.append(("termination",))
        return self._terminated | self._time_outs


class _RewardManager:
    def __init__(self, trace, count):
        self.trace = trace
        self.count = count
        self.fail = False

    def compute(self, *, dt):
        self.trace.append(("reward", dt))
        if self.fail:
            raise ValueError("reward failure counterexample")
        return torch.zeros(self.count)


class _ObservationManager:
    def __init__(self, trace, count):
        self.trace = trace
        self.count = count

    def compute(self, **kwargs):
        self.trace.append(("observation", kwargs))
        return {"policy": torch.zeros(self.count, 1)}


class _CommandTerm:
    def __init__(self, count):
        self.time_left = torch.ones(count)
        self.command_counter = torch.zeros(count, dtype=torch.long)


class _CommandManager:
    def __init__(self, trace, count):
        self.trace = trace
        self._term = _CommandTerm(count)

    @property
    def active_terms(self):
        return ["motion"]

    def get_term(self, name):
        assert name == "motion"
        return self._term

    def compute(self, *, dt):
        self.trace.append(("command", dt))


class _EventManager:
    available_modes = ()

    def apply(self, **kwargs):
        raise AssertionError("no interval event is configured")


class _Owner:
    def __init__(self, env, trace, options):
        self.env = env
        self.trace = trace
        self.mutate_clock = bool(options.get("mutate_clock", False))
        self.mutate_reset = bool(options.get("mutate_reset", False))
        self.mutate_reset_buf = bool(options.get("mutate_reset_buf", False))
        self.mutate_command_clock = bool(
            options.get("mutate_command_clock", False)
        )
        self.reentrant_caught = bool(options.get("reentrant_caught", False))
        self.fail = bool(options.get("fail", False))
        self.return_value = options.get("return_value")
        self.stamps = []
        self._full_mdp_post_physics_dependency_dag_sha256 = None
        self._full_mdp_post_physics_env = env
        self._full_mdp_post_physics_lease = None

    @property
    def full_mdp_post_physics_dependency_dag_sha256(self):
        return self._full_mdp_post_physics_dependency_dag_sha256

    @full_mdp_post_physics_dependency_dag_sha256.setter
    def full_mdp_post_physics_dependency_dag_sha256(self, value):
        self._full_mdp_post_physics_dependency_dag_sha256 = value

    @property
    def full_mdp_post_physics_env(self):
        return self._full_mdp_post_physics_env

    @full_mdp_post_physics_env.setter
    def full_mdp_post_physics_env(self, value):
        self._full_mdp_post_physics_env = value

    @property
    def full_mdp_post_physics_lease(self):
        return self._full_mdp_post_physics_lease

    @full_mdp_post_physics_lease.setter
    def full_mdp_post_physics_lease(self, value):
        self._full_mdp_post_physics_lease = value

    def publish_post_physics_substep(self, stamp):
        assert self.trace[-1][0] == "update"
        self.trace.append(("hook", stamp.exact_tuple()))
        self.stamps.append(stamp)
        if self.mutate_clock:
            self.env.common_step_counter += 1
        if self.mutate_reset:
            self.env.termination_manager._terminated.logical_not_()
        if self.mutate_reset_buf:
            self.env.reset_buf = self.env.reset_buf.clone()
        if self.mutate_command_clock:
            self.env.command_manager._term.time_left.add_(1.0)
        if self.reentrant_caught:
            try:
                self.env.step(torch.zeros(2, 4))
            except M.FullMdpPostPhysicsProtocolError:
                pass
        if self.fail:
            raise ValueError("owner failure counterexample")
        return self.return_value


def _fake_env(*, rendering=False, decimation=3, owner_kwargs=None):
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    trace = []
    count = 2
    env.device = "cpu"
    env.cfg = types.SimpleNamespace(
        decimation=decimation,
        sim=types.SimpleNamespace(render_interval=1),
        num_rerenders_on_reset=0,
    )
    env.physics_dt = 0.005
    env.step_dt = 0.02
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.episode_length_buf = torch.zeros(count, dtype=torch.long)
    env.action_manager = _ActionManager(trace)
    env.scene = _Scene(trace)
    env.sim = _Simulation(trace, rendering=rendering)
    env.recorder_manager = _RecorderManager(trace)
    env.termination_manager = _TerminationManager(trace, count)
    env.reward_manager = _RewardManager(trace, count)
    env.observation_manager = _ObservationManager(trace, count)
    env.command_manager = _CommandManager(trace, count)
    env.event_manager = _EventManager()
    env.extras = {}
    env._action_ball_full_mdp_components = object()
    env._full_mdp_active_dispatch = None
    env._full_mdp_post_physics_poison = None
    env._full_mdp_post_physics_expected_dependency_dag = "a" * 64
    env._full_mdp_post_physics_lease = object()
    owner = _Owner(env, trace, owner_kwargs or {})
    owner.full_mdp_post_physics_dependency_dag_sha256 = "a" * 64
    owner.full_mdp_post_physics_env = env
    owner.full_mdp_post_physics_lease = env._full_mdp_post_physics_lease
    env._full_mdp_post_physics_owner = owner
    env._full_mdp_post_physics_publish = owner.publish_post_physics_substep
    return env, owner, trace


def _without_render(trace):
    return [row for row in trace if row[0] != "render"]


def test_render_path_preserves_physics_and_exact_hook_parity_without_rng_use():
    rng_before = torch.random.get_rng_state().clone()
    plain, plain_owner, plain_trace = _fake_env(rendering=False)
    rendered, rendered_owner, rendered_trace = _fake_env(rendering=True)
    action = torch.zeros(2, 4)
    plain_result = plain.step(action)
    rendered_result = rendered.step(action)
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert _without_render(rendered_trace) == plain_trace
    assert [stamp.exact_tuple() for stamp in plain_owner.stamps] == [
        (1, 0, 3, 1, 1),
        (1, 1, 3, 2, 1),
        (1, 2, 3, 3, 1),
    ]
    assert [stamp.exact_tuple() for stamp in rendered_owner.stamps] == [
        stamp.exact_tuple() for stamp in plain_owner.stamps
    ]
    assert [row[0] for row in rendered_trace].count("render") == 3
    assert plain.common_step_counter == rendered.common_step_counter == 1
    assert plain._sim_step_counter == rendered._sim_step_counter == 3
    assert plain.episode_length_buf.tolist() == [1, 1]
    assert rendered.episode_length_buf.tolist() == [1, 1]
    assert len(plain_result) == len(rendered_result) == 5


@pytest.mark.parametrize(
    "owner_kwargs, expected",
    [
        ({"mutate_clock": True}, "mutated a protected manager"),
        ({"mutate_reset": True}, "changed version"),
        ({"mutate_command_clock": True}, "changed version"),
        ({"reentrant_caught": True}, "reentrant or poisoned"),
        ({"fail": True}, "owner failed"),
        ({"return_value": object()}, "must return None"),
    ],
)
def test_owner_mutation_or_failure_is_fail_stop_before_termination_reward(
    owner_kwargs, expected
):
    env, owner, trace = _fake_env(owner_kwargs=owner_kwargs)
    action = torch.zeros(2, 4)
    with pytest.raises(M.FullMdpPostPhysicsProtocolError, match=expected):
        env.step(action)
    assert owner.stamps
    assert not any(row[0] in {"termination", "reward"} for row in trace)
    trace_before_retry = list(trace)
    with pytest.raises(M.FullMdpPostPhysicsPoisonedError, match="cold reconstruction"):
        env.step(action)
    assert trace == trace_before_retry


def test_two_inference_mode_steps_and_inference_reset_mutation_counterexample():
    env, owner, trace = _fake_env(decimation=1)
    action = torch.zeros(2, 4)
    with torch.inference_mode():
        env.step(action)
        assert env.reset_buf.is_inference()
        env.step(action)
    assert len(owner.stamps) == 2
    assert env.common_step_counter == 2

    mutated, mutated_owner, mutated_trace = _fake_env(decimation=1)
    with torch.inference_mode():
        mutated.step(action)
        assert mutated.reset_buf.is_inference()
        mutated_owner.mutate_reset_buf = True
        prior_manager_rows = sum(
            row[0] in {"termination", "reward"} for row in mutated_trace
        )
        with pytest.raises(
            M.FullMdpPostPhysicsProtocolError,
            match="changed identity",
        ):
            mutated.step(action)
    assert sum(
        row[0] in {"termination", "reward"} for row in mutated_trace
    ) == prior_manager_rows


def test_post_publication_reward_failure_poisons_before_any_retry():
    env, owner, trace = _fake_env(decimation=2)
    env.reward_manager.fail = True
    action = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="reward failure counterexample"):
        env.step(action)
    assert len(owner.stamps) == 2
    assert any(row[0] == "termination" for row in trace)
    assert any(row[0] == "reward" for row in trace)
    trace_before_retry = list(trace)
    with pytest.raises(M.FullMdpPostPhysicsPoisonedError, match="cold reconstruction"):
        env.step(action)
    assert trace == trace_before_retry


def test_poison_record_cannot_be_disabled_by_replacing_a_module_global(
    monkeypatch,
):
    monkeypatch.setattr(
        M, "_PoisonRecord", lambda **kwargs: None, raising=False
    )
    env, owner, trace = _fake_env(decimation=1)
    env.reward_manager.fail = True
    action = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="reward failure counterexample"):
        env.step(action)
    assert owner.stamps
    assert isinstance(env._full_mdp_post_physics_poison, tuple)
    trace_before_retry = list(trace)
    with pytest.raises(M.FullMdpPostPhysicsPoisonedError, match="cold reconstruction"):
        env.step(action)
    assert trace == trace_before_retry


def test_owner_lease_and_external_authority_are_rejected_at_install(
    tmp_path, monkeypatch
):
    module, source_path = _load_concrete_owner_fixture(tmp_path, monkeypatch)
    env = object.__new__(M.ActionBallFullMdpManagerBasedRLEnv)
    lease = object()
    dependency_dag = "d" * 64
    pins = _concrete_owner_fixture_pins(
        module, source_path, dependency_dag
    )

    wrong_lease_owner = module.ConcreteOwner(
        env, object(), dependency_dag
    )
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="lease differs"
    ):
        env._validate_concrete_owner_install(
            wrong_lease_owner,
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )

    wrong_dag_owner = module.ConcreteOwner(env, lease, "e" * 64)
    with pytest.raises(
        M.FullMdpPostPhysicsProtocolError, match="dependency DAG differs"
    ):
        env._validate_concrete_owner_install(
            wrong_dag_owner,
            concrete_pins=pins,
            expected_dependency_dag=dependency_dag,
            expected_lease=lease,
        )


def test_hot_dispatch_does_not_rescan_owner_python_bindings():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    owner_type = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpManagerBasedRLEnv"
    )
    assert all(
        not (
            isinstance(node, ast.FunctionDef)
            and node.name == "_assert_owner_binding_current"
        )
        for node in owner_type.body
    )
    hot_methods = {
        "step",
        "_before_policy_step",
        "_publish_post_physics_substep",
        "_after_reward_close",
        "_reset_idx",
    }
    for method in owner_type.body:
        if (
            not isinstance(method, ast.FunctionDef)
            or method.name not in hot_methods
        ):
            continue
        assert not any(
            isinstance(node, ast.Call)
            and _call_name(node) == "self._assert_owner_binding_current"
            for node in ast.walk(method)
        )
