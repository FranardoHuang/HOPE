from __future__ import annotations

from dataclasses import replace
import inspect
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from test_action_ball_full_mdp_ball_scene import S, _Asset, _spec

_PHYSICAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_physical_flight_device.py"
)
_PHYSICAL_SPEC = importlib.util.spec_from_file_location(
    "_test_scene_physical_device", _PHYSICAL_PATH
)
assert _PHYSICAL_SPEC is not None and _PHYSICAL_SPEC.loader is not None
D = importlib.util.module_from_spec(_PHYSICAL_SPEC)
sys.modules[_PHYSICAL_SPEC.name] = D
_PHYSICAL_SPEC.loader.exec_module(D)


def _port(device: torch.device):
    _, spec = _spec(cadence=5, horizon=5)
    origins = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
        dtype=torch.float32,
        device=device,
    )
    scene = {}
    for name in spec.scene_entity_names:
        root = torch.zeros((2, 13), dtype=torch.float32, device=device)
        root[:, :3] = origins + torch.tensor(
            S.PARK_POSITION_ENV_M, dtype=torch.float32, device=device
        )
        root[:, 3] = 1.0
        scene[name] = _Asset(root)
    return S.IsaacLabPhysicalFlightScenePort(
        scene=scene,
        spec=spec,
        env_origins=origins,
    )


def _request(port, *, stamp=(7, 0, 2, 13, 1)):
    shape = (port.num_envs, port.flight_capacity)
    observe = torch.tensor(
        [[True, False], [False, True]], dtype=torch.bool, device=port.device
    )
    key = torch.zeros(shape + (32,), dtype=torch.uint8, device=port.device)
    key[observe] = 17
    generation = torch.full(shape, -1, dtype=torch.int64, device=port.device)
    generation[observe] = 5
    ordinal = torch.full(shape, -1, dtype=torch.int64, device=port.device)
    ordinal[observe] = 0
    state = port.read_state_env()
    return D.PhysicalPostPhysicsCaptureRequest(
        exact_stamp=stamp,
        observe_mask=observe,
        flight_slot=torch.arange(
            port.flight_capacity, dtype=torch.int64, device=port.device
        ).unsqueeze(0).expand(shape).clone(),
        full_key_sha256=key,
        ball_generation=generation,
        observation_ordinal=ordinal,
        previous_ball_center_m=state[..., :3].clone(),
        current_state_env_f32=state,
        _owner_identity=object(),
        _token=object(),
    )


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_capture_proves_scene_observation_but_faults_missing_event_producers_without_host_sync(
    device_name,
):
    port = _port(torch.device(device_name))
    request = _request(port)
    facts = port.capture_post_physics_facts(request)

    assert type(facts) is D.IsaacPostPhysicsFacts
    assert facts._owner_identity is request
    assert facts._capture_token is request._token
    assert torch.equal(
        facts.observation_stamp.control_step[request.observe_mask],
        torch.full((2,), 7, dtype=torch.int64, device=port.device),
    )
    assert torch.equal(
        facts.observation_stamp.physics_substep[request.observe_mask],
        torch.zeros((2,), dtype=torch.int32, device=port.device),
    )
    assert torch.equal(
        facts.observation_stamp.event_phase[request.observe_mask],
        torch.full((2,), 2, dtype=torch.int8, device=port.device),
    )
    assert not torch.any(facts.selected_contact_event)
    assert not torch.any(facts.net_crossing_event)
    assert not torch.any(facts.first_descending_crossing_event)
    assert not torch.any(facts.crossing_report_delivered)
    assert torch.equal(facts.producer_contract_fault, request.observe_mask)
    assert not torch.any(facts.nonfinite_observation)
    assert not torch.any(facts.engine_overflow)
    assert S.POSTPHYSICS_FACT_PRODUCERS_BOUND is False

    source = inspect.getsource(
        S.IsaacLabPhysicalFlightScenePort.capture_post_physics_facts
    )
    for forbidden in (".item(", ".cpu(", ".tolist(", ".numpy("):
        assert forbidden not in source


def test_merged_wrist_contact_is_not_mislabelled_selected_rubber_authority():
    """A ball-wrist actor is not used; only exact rubber collider paths are."""

    source = inspect.getsource(S.IsaacPhysxBallFactOwner.on_contact_report)
    assert "explicitly inventoried handle/wrist/venue collider" in source
    assert "rubber_path = collider" in source
    assert "other_actor_path != expected_wrist_actor" in source
    assert S.POSTPHYSICS_FACT_PRODUCERS_BOUND is False
    hold = " ".join(S.POSTPHYSICS_CAPTURE_HOLD_REASONS)
    assert "fresh N=2 SimulationApp counterexample probe" in hold


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    [
        ("exact_stamp", lambda value: (0, *value[1:]), "stamp range"),
        (
            "flight_slot",
            lambda value: value.to(torch.int32),
            "flight_slot tensor ABI",
        ),
        (
            "full_key_sha256",
            lambda value: value[..., :-1],
            "full_key_sha256 tensor ABI",
        ),
        (
            "ball_generation",
            lambda value: value.to(torch.int32),
            "ball_generation tensor ABI",
        ),
        (
            "observation_ordinal",
            lambda value: value.to(torch.int32),
            "observation_ordinal tensor ABI",
        ),
    ],
)
def test_capture_rejects_wrong_stamp_slot_key_generation_or_ordinal_abi(
    field, mutation, message
):
    port = _port(torch.device("cpu"))
    request = _request(port)
    object.__setattr__(request, field, mutation(getattr(request, field)))
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match=message):
        port.capture_post_physics_facts(request)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda request: request.flight_slot.__setitem__((0, 0), 1),
        lambda request: request.full_key_sha256[0, 0].zero_(),
        lambda request: request.ball_generation.__setitem__((0, 0), -1),
        lambda request: request.observation_ordinal.__setitem__((0, 0), -1),
        lambda request: request.current_state_env_f32.__setitem__(
            (0, 0, 0), request.current_state_env_f32[0, 0, 0] + 1.0
        ),
    ],
)
def test_capture_marks_mutated_live_slot_key_generation_ordinal_or_state_as_fault(
    mutate,
):
    port = _port(torch.device("cpu"))
    request = _request(port)
    mutate(request)
    with pytest.raises(
        S.ActionBallFullMdpBallSceneError,
        match="slot/key/generation/ordinal/state self-consistency",
    ):
        port.capture_post_physics_facts(request)


def test_cuda_mutated_request_sets_fault_without_host_synchronization(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    port = _port(torch.device("cuda"))
    request = _request(port)
    request.flight_slot[0, 0] = 1
    original_item = torch.Tensor.item

    def forbidden_item(value):
        if value.is_cuda:
            raise AssertionError("scene postphysics synchronized CUDA to host")
        return original_item(value)

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    facts = port.capture_post_physics_facts(request)
    assert facts.producer_contract_fault.device.type == "cuda"
    assert torch.equal(
        facts.producer_contract_fault,
        torch.tensor(
            [[True, False], [False, True]],
            dtype=torch.bool,
            device=port.device,
        ),
    )


def test_capture_reports_nonfinite_scene_observation_without_turning_it_into_a_miss():
    port = _port(torch.device("cpu"))
    request = _request(port)
    request.current_state_env_f32[0, 0, 0] = float("nan")
    port.assets[0].data.root_state_w[0, 0] = float("nan")
    facts = port.capture_post_physics_facts(request)
    assert facts.nonfinite_observation[0, 0]
    assert facts.producer_contract_fault[0, 0]
    assert not facts.selected_contact_event[0, 0]


class _Header:
    def __init__(self, *, ball: str, collider: str):
        self.type = SimpleNamespace(name="CONTACT_FOUND")
        self.actor0 = ball
        self.actor1 = collider.rsplit("/", 1)[0]
        self.collider0 = ball
        self.collider1 = collider


class _Settings:
    def __init__(self, *, readback=False, raises=False):
        self.value = True
        self.readback = readback
        self.raises = raises
        self.writes = []

    def set_bool(self, path, value):
        if self.raises:
            raise RuntimeError("write failed")
        self.writes.append((path, value))
        self.value = value

    def get(self, path):
        assert path == "/physics/disableContactProcessing"
        return self.readback if self.readback is not None else self.value


class _PhysxState:
    def __init__(self, running=False):
        self.running = running

    def is_running(self):
        return self.running


def _fact_owner(*, device: torch.device, n: int = 2, k: int = 2):
    ball = tuple(
        tuple(f"/World/envs/env_{env}/Ball_{slot}" for slot in range(k))
        for env in range(n)
    )
    red = tuple(f"/World/envs/env_{env}/Robot/red/collider" for env in range(n))
    black = tuple(f"/World/envs/env_{env}/Robot/black/collider" for env in range(n))
    centres = torch.zeros((n, k, 3), dtype=torch.float32, device=device)
    owner = S.IsaacPhysxBallFactOwner._diagnostic_unauthorized_for_test(
        _test_token=S._PHYSX_FACT_CHECKPOINT_TOKEN,
        num_envs=n,
        flight_capacity=k,
        device=device,
        scene_identity_sha256="1" * 64,
        concrete_ball_prim_paths=ball,
        red_rubber_collider_paths=red,
        black_rubber_collider_paths=black,
        venue=S.CanonicalVenuePlanes(
            near_table_x_m=0.5,
            far_table_x_m=3.24,
            table_half_width_m=0.7625,
            table_surface_z_m=0.76,
            landing_ball_center_z_m=0.78,
            net_x_m=1.87,
            net_clear_ball_center_z_m=0.9325,
        ),
        center_sampler=lambda env, slot: centres[env, slot].clone(),
        expected_authority_validator=lambda value: value,
        path_decoder=lambda value: value,
        callback_order=S.CALLBACK_ORDER_CONTACT_BEFORE_HEARTBEAT,
    )
    return owner, centres, ball, red, black


def _authority(owner, *, rubber=None):
    shape = (owner.num_envs, owner.flight_capacity)
    active = torch.ones(shape, dtype=torch.bool, device=owner.device)
    selected = torch.full(
        shape,
        S.RUBBER_RED if rubber is None else rubber,
        dtype=torch.int8,
        device=owner.device,
    )
    key = torch.ones(shape + (32,), dtype=torch.uint8, device=owner.device)
    generation = torch.zeros(shape, dtype=torch.int64, device=owner.device)
    return S.ExpectedRubberAuthorityView(
        active_mask=active,
        expected_rubber=selected,
        full_key_sha256=key,
        ball_generation=generation,
        projection_sha256="3" * 64,
        _owner_identity=object(),
        _token=object(),
    )


def _owner_request(owner, *, stamp=(1, 0, 1, 1, 1)):
    shape = (owner.num_envs, owner.flight_capacity)
    return D.PhysicalPostPhysicsCaptureRequest(
        exact_stamp=stamp,
        observe_mask=torch.ones(shape, dtype=torch.bool, device=owner.device),
        flight_slot=torch.arange(owner.flight_capacity, device=owner.device).unsqueeze(0).expand(shape),
        full_key_sha256=torch.ones(shape + (32,), dtype=torch.uint8, device=owner.device),
        ball_generation=torch.zeros(shape, dtype=torch.int64, device=owner.device),
        observation_ordinal=torch.zeros(shape, dtype=torch.int64, device=owner.device),
        previous_ball_center_m=torch.zeros(shape + (3,), dtype=torch.float32, device=owner.device),
        current_state_env_f32=torch.zeros(shape + (13,), dtype=torch.float32, device=owner.device),
        _owner_identity=object(),
        _token=object(),
    )


def _capture_owner(owner, centres, *, stamp=(1, 0, 1, 1, 1)):
    request = _owner_request(owner, stamp=stamp)
    live = torch.zeros(
        (owner.num_envs, owner.flight_capacity, 13),
        dtype=torch.float32,
        device=owner.device,
    )
    live[..., :3] = centres
    return owner.capture(
        request=request, live_state=live, facts_type=D.IsaacPostPhysicsFacts, stamp_type=D.PhysicsStampGrid
    )


def _direct_arm(owner, *, active=None, rubber=None, key=None, generation=None):
    shape = (owner.num_envs, owner.flight_capacity)
    active = (
        torch.ones(shape, dtype=torch.bool, device=owner.device)
        if active is None
        else active
    )
    rubber = (
        torch.full(shape, S.RUBBER_RED, dtype=torch.int8, device=owner.device)
        if rubber is None
        else rubber
    )
    rubber = torch.where(
        active, rubber, torch.full_like(rubber, S.RUBBER_INACTIVE)
    )
    key = (
        torch.ones(shape + (32,), dtype=torch.uint8, device=owner.device)
        if key is None
        else key
    )
    generation = (
        torch.zeros(shape, dtype=torch.int64, device=owner.device)
        if generation is None
        else generation
    )
    owner._bind_action_epoch_expected_rubber(
        active_mask=active,
        expected_rubber=rubber,
        ball_generation=generation,
        full_key_sha256=key,
        _installer_token=S._PHYSX_FACT_OWNER_TOKEN,
    )


def _mark_live_subscriptions(owner):
    epoch = object()
    owner._bind_live_subscriptions(
        applied_ball_prim_paths=tuple(
            path for row in owner.concrete_ball_prim_paths for path in row
        ),
        contact_subscription=object(),
        heartbeat_subscription=object(),
        error_subscription=object(),
        _installer_token=S._PHYSX_FACT_OWNER_TOKEN,
        _subscription_epoch=epoch,
    )
    return epoch


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_action_epoch_direct_capture_joins_full_key_and_allows_unarmed_empty(
    device_name,
):
    owner, centres, *_ = _fact_owner(device=torch.device(device_name))
    _mark_live_subscriptions(owner)
    request = _owner_request(owner)
    _direct_arm(owner)
    request.full_key_sha256[0, 0, 0] = 9
    owner.on_post_step_heartbeat(0.005)
    facts = owner.capture(
        request=request,
        live_state=request.current_state_env_f32,
        facts_type=D.IsaacPostPhysicsFacts,
        stamp_type=D.PhysicsStampGrid,
    )
    assert facts.producer_contract_fault[0, 0]
    assert not facts.producer_contract_fault[0, 1]

    empty = _owner_request(owner, stamp=(2, 0, 1, 2, 1))
    empty.observe_mask.zero_()
    empty.full_key_sha256.zero_()
    empty.ball_generation.fill_(-1)
    owner.on_post_step_heartbeat(0.005)
    no_facts = owner.capture(
        request=empty,
        live_state=empty.current_state_env_f32,
        facts_type=D.IsaacPostPhysicsFacts,
        stamp_type=D.PhysicsStampGrid,
    )
    assert not torch.any(no_facts.producer_contract_fault)
    assert not torch.any(no_facts.selected_contact_event)


def test_direct_selected_contact_wrong_face_known_none_and_unknown_alias_are_distinct():
    owner, centres, ball, red, black = _fact_owner(device=torch.device("cpu"))
    owner._diagnostic_unauthorized = False
    _mark_live_subscriptions(owner)
    _direct_arm(owner)
    centres[0, 0] = torch.tensor((1.0, 0.0, 1.0))
    owner.on_contact_report(
        [
            _Header(ball=ball[0][0], collider=red[0]),
            _Header(ball=ball[0][1], collider=black[0]),
        ],
        [],
    )
    owner.on_post_step_heartbeat(0.005)
    facts = _capture_owner(owner, centres)
    assert facts.selected_contact_event[0, 0]
    assert not facts.selected_contact_event[0, 1]
    assert not facts.selected_contact_event[1, 0]
    assert not torch.any(facts.producer_contract_fault)
    assert owner.diagnostic_telemetry()["wrong_face_event_count_by_ball"][0, 1] == 1

    _direct_arm(owner)
    owner.on_contact_report(
        [_Header(ball=ball[0][0], collider="/World/prototype/red_alias")], []
    )
    owner.on_post_step_heartbeat(0.005)
    fault = _capture_owner(owner, centres, stamp=(2, 0, 1, 2, 1))
    assert torch.all(fault.producer_contract_fault)
    assert not torch.any(fault.selected_contact_event)


def test_same_ball_selected_and_opposite_face_in_one_callback_is_ambiguous_fault():
    owner, centres, ball, red, black = _fact_owner(device=torch.device("cpu"))
    owner._diagnostic_unauthorized = False
    _mark_live_subscriptions(owner)
    _direct_arm(owner)
    centres[0, 0] = torch.tensor((1.0, 0.0, 1.0))
    owner.on_contact_report(
        [
            _Header(ball=ball[0][0], collider=red[0]),
            _Header(ball=ball[0][0], collider=black[0]),
        ],
        [],
    )
    owner.on_post_step_heartbeat(0.005)
    facts = _capture_owner(owner, centres)
    assert facts.producer_contract_fault[0, 0]
    assert not facts.selected_contact_event[0, 0]


def test_selected_plus_known_handle_is_ambiguous_but_handle_only_is_none():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    handle = "/World/envs/env_0/Robot/handle"
    other_handle = "/World/envs/env_1/Robot/handle"
    owner._known_non_rubber_path_to_binding = {
        handle: (0, "handle", "/World/envs/env_0/Robot"),
        other_handle: (1, "handle", "/World/envs/env_1/Robot"),
    }
    owner._diagnostic_unauthorized = False
    _mark_live_subscriptions(owner)
    _direct_arm(owner)
    owner.on_contact_report(
        [
            _Header(ball=ball[0][0], collider=red[0]),
            _Header(ball=ball[0][0], collider=handle),
            _Header(ball=ball[1][1], collider=other_handle),
        ],
        [],
    )
    owner.on_post_step_heartbeat(0.005)
    facts = _capture_owner(owner, centres)
    assert facts.producer_contract_fault[0, 0]
    assert not facts.selected_contact_event[0, 0]
    assert not facts.producer_contract_fault[1, 1]
    assert not facts.selected_contact_event[1, 1]


@pytest.mark.parametrize("ball_on_left", (True, False))
def test_cross_env_known_handle_faults_in_both_header_orders(ball_on_left):
    owner, centres, ball, _red, _black = _fact_owner(device=torch.device("cpu"))
    handle = "/World/envs/env_1/Robot/handle"
    actor = "/World/envs/env_1/Robot"
    owner._known_non_rubber_path_to_binding = {
        handle: (1, "handle", actor)
    }
    owner._diagnostic_unauthorized = False
    _mark_live_subscriptions(owner)
    _direct_arm(owner)
    ball_path = ball[0][0]
    header = SimpleNamespace(type=SimpleNamespace(name="CONTACT_FOUND"))
    if ball_on_left:
        header.actor0, header.collider0 = ball_path, ball_path
        header.actor1, header.collider1 = actor, handle
    else:
        header.actor0, header.collider0 = actor, handle
        header.actor1, header.collider1 = ball_path, ball_path
    owner.on_contact_report([header], [])
    owner.on_post_step_heartbeat(0.005)
    facts = _capture_owner(owner, centres)
    assert torch.all(facts.producer_contract_fault)
    assert not torch.any(facts.selected_contact_event)


def test_live_table_bounds_join_canonical_venue_and_reject_move_or_scale():
    venue = S.CanonicalVenuePlanes(
        near_table_x_m=0.5,
        far_table_x_m=3.24,
        table_half_width_m=0.7625,
        table_surface_z_m=0.76,
        landing_ball_center_z_m=0.78,
        net_x_m=1.87,
        net_clear_ball_center_z_m=0.9325,
    )
    expected_min = (0.5, -0.7625, 0.73)
    expected_max = (3.24, 0.7625, 0.76)
    S._require_canonical_table_bounds(
        minimum_env_m=expected_min,
        maximum_env_m=expected_max,
        venue=venue,
        table_thickness_m=0.03,
    )
    for minimum, maximum in (
        ((0.51, -0.7625, 0.73), expected_max),
        (expected_min, (3.25, 0.7625, 0.76)),
    ):
        with pytest.raises(S.ActionBallFullMdpBallSceneError, match="pose or size"):
            S._require_canonical_table_bounds(
                minimum_env_m=minimum,
                maximum_env_m=maximum,
                venue=venue,
                table_thickness_m=0.03,
            )


def test_table_collider_inventory_is_one_exact_enabled_static_cube_child():
    root = "/World/envs/env_0/TableObstacle"
    expected = f"{root}/geometry/mesh"
    assert (
        S._require_exact_table_collider_inventory(
            table_root=root,
            colliders=((expected, "Cube", True, False),),
        )
        == expected
    )
    mutations = (
        (),
        ((expected, "Cube", False, False),),
        ((expected, "Cube", True, True),),
        ((f"{root}/mesh", "Cube", True, False),),
        (
            (expected, "Cube", True, False),
            (f"{root}/geometry/other", "Cube", True, False),
        ),
        (
            (
                "/World/envs/env_1/TableObstacle/geometry/mesh",
                "Cube",
                True,
                False,
            ),
        ),
    )
    for colliders in mutations:
        with pytest.raises(
            S.ActionBallFullMdpBallSceneError,
            match="table collider inventory differs",
        ):
            S._require_exact_table_collider_inventory(
                table_root=root,
                colliders=colliders,
            )


def test_composed_mesh_arrays_reject_swapped_points_translation_and_duplicate():
    expected = SimpleNamespace(
        name="red_rubber_collider",
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        face_vertex_counts=(3,),
        face_vertex_indices=(0, 1, 2),
        translate_in_wrist_m=(0.1, 0.2, 0.3),
    )
    common = dict(
        name=expected.name,
        actual_points=expected.points,
        actual_face_vertex_counts=expected.face_vertex_counts,
        actual_face_vertex_indices=expected.face_vertex_indices,
        actual_translate_in_wrist_m=expected.translate_in_wrist_m,
        expected=expected,
    )
    S._require_composed_collider_mesh_arrays(**common)
    mutations = (
        {"actual_points": tuple(reversed(expected.points))},
        {"actual_translate_in_wrist_m": (0.2, 0.2, 0.3)},
        {"actual_face_vertex_indices": (0, 1, 1)},
    )
    for mutation in mutations:
        with pytest.raises(S.ActionBallFullMdpBallSceneError, match="geometry differs"):
            S._require_composed_collider_mesh_arrays(**(common | mutation))


@pytest.mark.parametrize(
    ("has_api", "approximation"),
    ((False, "convexHull"), (True, "none")),
)
def test_named_mesh_requires_mesh_collision_api_and_convex_hull(
    has_api, approximation
):
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="MeshCollisionAPI"):
        S._require_mesh_collision_approximation(
            name="red_rubber_collider",
            has_mesh_collision_api=has_api,
            approximation=approximation,
        )
    S._require_mesh_collision_approximation(
        name="red_rubber_collider",
        has_mesh_collision_api=True,
        approximation="convexHull",
    )


def test_full_key_change_resets_same_generation_same_face_latches():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner._diagnostic_unauthorized = False
    _mark_live_subscriptions(owner)
    first_key = torch.ones((2, 2, 32), dtype=torch.uint8)
    _direct_arm(owner, key=first_key)
    owner.on_contact_report([_Header(ball=ball[0][0], collider=red[0])], [])
    owner.on_post_step_heartbeat(0.005)
    first = _capture_owner(owner, centres)
    assert first.selected_contact_event[0, 0]
    assert owner._contact_latch[0, 0]

    next_key = first_key.clone()
    next_key[0, 0, 0] = 2
    _direct_arm(owner, key=next_key)
    assert not owner._contact_latch[0, 0]


def test_live_identity_rearms_next_substep_and_admits_later_selected_contact():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner._diagnostic_unauthorized = False
    _mark_live_subscriptions(owner)

    _direct_arm(owner)
    owner.on_post_step_heartbeat(0.005)
    first = _capture_owner(owner, centres)
    assert not torch.any(first.selected_contact_event)
    assert not torch.any(first.producer_contract_fault)
    assert owner._action_epoch_direct_binding is False

    _direct_arm(owner)
    centres[1, 1] = torch.tensor((1.2, 0.1, 1.0))
    owner.on_post_step_heartbeat(0.005)
    owner.on_contact_report(
        [_Header(ball=ball[1][1], collider=red[1])], []
    )
    second = _capture_owner(owner, centres, stamp=(1, 1, 2, 2, 1))
    expected = torch.zeros((2, 2), dtype=torch.bool)
    expected[1, 1] = True
    assert torch.equal(second.selected_contact_event, expected)
    assert not torch.any(second.producer_contract_fault)


def test_contact_injected_after_capture_before_next_heartbeat_is_not_a_next_step_fact():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner.callback_order = S.CALLBACK_ORDER_HEARTBEAT_BEFORE_CONTACT
    owner._diagnostic_unauthorized = False
    _mark_live_subscriptions(owner)
    _direct_arm(owner)
    owner.on_post_step_heartbeat(0.005)
    first = _capture_owner(owner, centres)
    assert not torch.any(first.selected_contact_event)

    # This contact arrives outside any new PhysX step: capture(t) has already
    # completed and heartbeat(t+1) has not begun.  It must never be re-labelled
    # as a selected contact belonging to t+1.
    _direct_arm(owner)
    owner.on_contact_report([_Header(ball=ball[0][0], collider=red[0])], [])
    owner.on_post_step_heartbeat(0.005)
    delayed = _capture_owner(owner, centres, stamp=(1, 1, 2, 2, 1))
    assert delayed.producer_contract_fault[0, 0]
    assert not delayed.selected_contact_event[0, 0]


def test_physx_owner_selected_face_wrong_face_and_normal_miss_are_distinct():
    owner, centres, ball, red, black = _fact_owner(device=torch.device("cpu"))
    owner.bind_expected_rubber_authority(_authority(owner))
    centres[0, 0] = torch.tensor((1.0, 0.0, 1.0))
    centres[0, 1] = torch.tensor((1.1, 0.0, 1.0))
    owner.on_contact_report(
        [
            _Header(ball=ball[0][0], collider=red[0]),
            _Header(ball=ball[0][1], collider=black[0]),
        ],
        [],
    )
    owner.on_post_step_heartbeat(0.005)
    facts = _capture_owner(owner, centres)
    # The algorithmic core sees the exact selected face, but missing real
    # subscriptions must keep the production fact fail-closed.
    assert not facts.selected_contact_event[0, 0]
    assert not facts.selected_contact_event[0, 1]  # wrong face
    assert torch.all(facts.producer_contract_fault)
    assert torch.count_nonzero(facts.selected_contact_ball_center_m) == 0
    assert torch.equal(
        facts.selected_contact_ball_center_m,
        facts.selected_contact_outgoing_segment_anchor_m,
    )
    telemetry = owner.diagnostic_telemetry()
    assert telemetry["wrong_face_event_count_by_ball"][0, 1] == 1
    assert telemetry["engine_overflow_attribution"] == (
        "scene_global_broadcast_to_live_rows"
    )


def test_physx_owner_uses_continuous_centres_and_single_latches_not_net_collision():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner.bind_expected_rubber_authority(_authority(owner))
    centres[0, 0] = torch.tensor((1.0, 0.1, 1.10))
    owner.on_contact_report([_Header(ball=ball[0][0], collider=red[0])], [])
    owner.on_post_step_heartbeat(0.005)
    first = _capture_owner(owner, centres)
    # Inject only the already-decoded diagnostic latch; no callback/geometry
    # test may claim this is a live selected-contact fact.
    owner._contact_latch[0, 0] = True
    assert not first.net_crossing_event[0, 0]

    owner.bind_expected_rubber_authority(_authority(owner))
    centres[0, 0] = torch.tensor((2.0, 0.3, 1.00))
    owner.on_post_step_heartbeat(0.005)
    second = _capture_owner(owner, centres, stamp=(2, 0, 1, 2, 1))
    assert second.net_crossing_event[0, 0]
    assert second.net_clear_at_crossing[0, 0]
    assert not second.first_descending_crossing_event[0, 0]

    owner.bind_expected_rubber_authority(_authority(owner))
    centres[0, 0] = torch.tensor((2.2, 0.4, 0.60))
    owner.on_post_step_heartbeat(0.005)
    third = _capture_owner(owner, centres, stamp=(3, 0, 1, 3, 1))
    assert not third.net_crossing_event[0, 0]
    assert third.first_descending_crossing_event[0, 0]
    assert torch.allclose(
        third.first_descending_crossing_xy_m[0, 0],
        torch.tensor((2.11, 0.355)),
    )

    owner.bind_expected_rubber_authority(_authority(owner))
    centres[0, 0] = torch.tensor((2.3, 0.5, 0.50))
    owner.on_post_step_heartbeat(0.005)
    fourth = _capture_owner(owner, centres, stamp=(4, 0, 1, 4, 1))
    assert not fourth.net_crossing_event[0, 0]
    assert not fourth.first_descending_crossing_event[0, 0]


def test_physx_owner_missing_heartbeat_and_global_engine_error_fault_live_rows():
    owner, centres, _ball, _red, _black = _fact_owner(device=torch.device("cpu"))
    owner.bind_expected_rubber_authority(_authority(owner))
    missing = _capture_owner(owner, centres)
    assert torch.all(missing.producer_contract_fault)

    owner.bind_expected_rubber_authority(_authority(owner))
    owner.on_post_step_heartbeat(0.005)
    owner.on_error_event(SimpleNamespace(type=SimpleNamespace(name="PHYSX_CUDA_ERROR")))
    fault = _capture_owner(owner, centres, stamp=(2, 0, 1, 2, 1))
    assert torch.all(fault.producer_contract_fault)
    assert not torch.any(fault.engine_overflow)
    telemetry = owner.diagnostic_telemetry()
    assert dict(telemetry["scene_global_error_counts"])["PHYSX_CUDA_ERROR"] == 1


def test_physx_owner_checkpoint_has_latches_centres_stamps_but_no_subscriptions():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner.bind_expected_rubber_authority(_authority(owner))
    centres[0, 0] = torch.tensor((1.0, 0.0, 1.0))
    owner.on_contact_report([_Header(ball=ball[0][0], collider=red[0])], [])
    owner.on_post_step_heartbeat(0.005)
    _capture_owner(owner, centres)
    owner._contact_latch[0, 0] = True
    checkpoint = owner.checkpoint_projection()
    assert not hasattr(checkpoint, "contact_subscription")
    assert checkpoint.last_heartbeat == 1
    assert checkpoint.last_capture_heartbeat == 1
    restored, *_ = _fact_owner(device=torch.device("cpu"))
    restored.restore_checkpoint_projection(checkpoint)
    restored_checkpoint = restored.checkpoint_projection()
    assert torch.equal(
        restored_checkpoint.previous_center_m, checkpoint.previous_center_m
    )
    assert torch.equal(
        restored_checkpoint.selected_contact_latch,
        checkpoint.selected_contact_latch,
    )
    assert restored_checkpoint.callback_sequence == checkpoint.callback_sequence


def test_physx_owner_checkpoint_rejects_mutation_and_duplicate_stamp():
    owner, centres, _ball, _red, _black = _fact_owner(device=torch.device("cpu"))
    owner.bind_expected_rubber_authority(_authority(owner))
    owner.on_post_step_heartbeat(0.005)
    _capture_owner(owner, centres)
    checkpoint = owner.checkpoint_projection()
    checkpoint.previous_center_m[0, 0, 0] = float("nan")
    restored, *_ = _fact_owner(device=torch.device("cpu"))
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="identity/header"):
        restored.restore_checkpoint_projection(checkpoint)

    clean = owner.checkpoint_projection()
    forged = replace(clean, last_heartbeat=-1, content_sha256="0" * 64)
    forged = replace(
        forged,
        content_sha256=owner._checkpoint_content_sha256(forged),
    )
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="identity/header"):
        restored.restore_checkpoint_projection(forged)

    owner.bind_expected_rubber_authority(_authority(owner))
    owner.on_post_step_heartbeat(0.005)
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="non-monotonic"):
        _capture_owner(owner, centres)


def test_physx_owner_stale_subscription_epoch_faults_without_latching_callback():
    owner, _centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner.bind_expected_rubber_authority(_authority(owner))
    owner._live_subscription_epoch = object()
    owner._on_epoch_contact_report(
        object(), [_Header(ball=ball[0][0], collider=red[0])], []
    )
    assert owner._producer_fault_sticky is True
    assert not torch.any(owner._contact_candidate_event)


def test_contact_processing_preenabled_setting_acquires_lease_without_hot_write():
    S._CONTACT_PROCESSING_LEASE_OWNER = None
    owner, *_ = _fact_owner(device=torch.device("cpu"))
    settings = _Settings(readback=False)
    owner._acquire_process_global_contact_processing(
        settings_iface=settings, physx_iface=_PhysxState(True)
    )
    assert settings.writes == []
    assert owner._contact_processing_lease_owned is True
    owner.shutdown()
    # Process-global state is deliberately not restored to True.
    assert settings.writes == []
    assert S._CONTACT_PROCESSING_LEASE_OWNER is None


@pytest.mark.parametrize(
    ("running", "readback"),
    [
        (False, False),
        (True, True),
        (True, None),
    ],
)
def test_contact_processing_rejects_unattached_or_not_preenabled(
    running, readback
):
    S._CONTACT_PROCESSING_LEASE_OWNER = None
    owner, *_ = _fact_owner(device=torch.device("cpu"))
    with pytest.raises(
        S.ActionBallFullMdpBallSceneError,
        match="not enabled before simulation attachment",
    ):
        owner._acquire_process_global_contact_processing(
            settings_iface=_Settings(readback=readback),
            physx_iface=_PhysxState(running),
        )
    assert owner._contact_processing_lease_owned is False


def test_contact_processing_rejects_an_observed_callback_even_if_physx_stopped():
    S._CONTACT_PROCESSING_LEASE_OWNER = None
    owner, *_ = _fact_owner(device=torch.device("cpu"))
    owner.on_post_step_heartbeat(0.005)
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="after a physics callback"):
        owner._acquire_process_global_contact_processing(
            settings_iface=_Settings(readback=False),
            physx_iface=_PhysxState(True),
        )


def test_contact_processing_lease_rejects_second_scene_owner():
    S._CONTACT_PROCESSING_LEASE_OWNER = None
    first, *_ = _fact_owner(device=torch.device("cpu"))
    second, *_ = _fact_owner(device=torch.device("cpu"))
    first._acquire_process_global_contact_processing(
        settings_iface=_Settings(readback=False), physx_iface=_PhysxState(True)
    )
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="another scene owner"):
        second._acquire_process_global_contact_processing(
            settings_iface=_Settings(readback=False), physx_iface=_PhysxState(True)
        )
    first.shutdown()


def test_shutdown_invalidates_epoch_before_callbacks_drains_all_and_is_idempotent():
    S._CONTACT_PROCESSING_LEASE_OWNER = None
    owner, _centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    _direct_arm(owner)
    owner._acquire_process_global_contact_processing(
        settings_iface=_Settings(readback=False), physx_iface=_PhysxState(True)
    )
    epoch = object()
    calls = []

    class _Handle:
        def __init__(self, index, callback=None, raises=False):
            self.index = index
            self.callback = callback
            self.raises = raises

        def unsubscribe(self):
            calls.append(self.index)
            if self.callback is not None:
                self.callback()
            if self.raises:
                raise RuntimeError("unsubscribe failed")

    owner._contact_subscription = _Handle(
        0,
        callback=lambda: owner._on_epoch_contact_report(
            epoch, [_Header(ball=ball[0][0], collider=red[0])], []
        ),
        raises=True,
    )
    owner._heartbeat_subscription = _Handle(
        1, callback=lambda: owner._on_epoch_post_step_heartbeat(epoch, 0.005)
    )
    owner._error_subscription = _Handle(2)
    owner._live_subscription_epoch = epoch
    owner._applied_contact_report_paths = owner._ordered_ball_paths

    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="teardown"):
        owner.shutdown()
    assert calls == [0, 1, 2]
    assert not torch.any(owner._contact_candidate_event)
    assert owner._last_heartbeat == 0
    assert S._CONTACT_PROCESSING_LEASE_OWNER is None
    owner.shutdown()
    assert calls == [0, 1, 2]


def test_shutdown_releases_process_lease_for_next_scene_owner():
    S._CONTACT_PROCESSING_LEASE_OWNER = None
    first, *_ = _fact_owner(device=torch.device("cpu"))
    second, *_ = _fact_owner(device=torch.device("cpu"))
    first._acquire_process_global_contact_processing(
        settings_iface=_Settings(readback=False), physx_iface=_PhysxState(True)
    )
    first.shutdown()
    second._acquire_process_global_contact_processing(
        settings_iface=_Settings(readback=False), physx_iface=_PhysxState(True)
    )
    second.shutdown()
    assert S._CONTACT_PROCESSING_LEASE_OWNER is None


def test_physx_owner_heartbeat_before_contact_and_cross_env_rubber_fail_closed():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner.callback_order = S.CALLBACK_ORDER_HEARTBEAT_BEFORE_CONTACT
    owner.bind_expected_rubber_authority(_authority(owner))
    owner.on_post_step_heartbeat(0.005)
    centres[0, 0] = torch.tensor((1.0, 0.0, 1.0))
    owner.on_contact_report([_Header(ball=ball[0][0], collider=red[1])], [])
    facts = _capture_owner(owner, centres)
    assert torch.all(facts.producer_contract_fault)
    assert not torch.any(facts.selected_contact_event)


def test_physx_owner_rejects_dirty_checkpoint_and_fake_bindings():
    owner, centres, ball, red, _black = _fact_owner(device=torch.device("cpu"))
    owner.bind_expected_rubber_authority(_authority(owner))
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="drained"):
        owner.checkpoint_projection()
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="direct subscription"):
        owner.bind_subscriptions(
            applied_ball_prim_paths=tuple(path for row in ball for path in row),
            contact_subscription=object(),
            heartbeat_subscription=object(),
            error_subscription=object(),
        )
    # Even a same-process caller that reaches Python-private tokens cannot turn
    # the diagnostic core into a successful fact producer.
    epoch = object()
    owner._bind_live_subscriptions(
        applied_ball_prim_paths=tuple(path for row in ball for path in row),
        contact_subscription=object(),
        heartbeat_subscription=object(),
        error_subscription=object(),
        _installer_token=S._PHYSX_FACT_OWNER_TOKEN,
        _subscription_epoch=epoch,
    )
    owner.abort_expected_rubber_authority(owner._bound_authority)
    owner.bind_expected_rubber_authority(_authority(owner))
    centres[0, 0] = torch.tensor((1.0, 0.0, 1.0))
    owner.on_contact_report([_Header(ball=ball[0][0], collider=red[0])], [])
    owner.on_post_step_heartbeat(0.005)
    facts = _capture_owner(owner, centres)
    assert torch.all(facts.producer_contract_fault)
    assert not torch.any(facts.selected_contact_event)


def test_production_installer_exposes_no_caller_path_validator_or_order_surface():
    assert not hasattr(S, "install_isaac_physx_ball_fact_owner")
    signature = inspect.signature(
        S.IsaacLabPhysicalFlightScenePort.install_action_epoch_live_physx_fact_owner
    )
    assert tuple(signature.parameters) == ("self", "stage")


class _ReadOnlyContactReportPrim:
    def __init__(self, *, armed):
        self.armed = armed
        self.queries = []
        self.inventory = ("root", "geometry", "geometry/mesh")

    def HasAPI(self, api_type):
        self.queries.append(api_type)
        return self.armed


def test_live_contact_report_gate_is_read_only_on_success_and_failure():
    api_type = object()
    ready = _ReadOnlyContactReportPrim(armed=True)
    missing = _ReadOnlyContactReportPrim(armed=False)
    ready_before = (ready.armed, ready.inventory)
    missing_before = (missing.armed, missing.inventory)

    assert S._require_pre_attached_contact_report_prims(
        ball_prims=(("/World/envs/env_0/Ball", ready),),
        contact_report_api_type=api_type,
    ) == ("/World/envs/env_0/Ball",)
    with pytest.raises(
        S.ActionBallFullMdpBallSceneError,
        match="lacks its pre-attach PhysxContactReportAPI",
    ):
        S._require_pre_attached_contact_report_prims(
            ball_prims=(("/World/envs/env_1/Ball", missing),),
            contact_report_api_type=api_type,
        )

    assert (ready.armed, ready.inventory) == ready_before
    assert (missing.armed, missing.inventory) == missing_before
    assert ready.queries == [api_type]
    assert missing.queries == [api_type]


def test_live_installer_has_no_post_attach_stage_or_schema_writes():
    sources = "\n".join(
        (
            inspect.getsource(S.IsaacPhysxBallFactOwner.install_live_physx_subscriptions),
            inspect.getsource(S._require_pre_attached_contact_report_prims),
            inspect.getsource(S._install_isaac_physx_ball_fact_owner),
            inspect.getsource(S._live_probe_isaac_physx_ball_fact_owner),
        )
    )
    for forbidden in (
        ".Apply(",
        ".Set(",
        "DefinePrim",
        "RemovePrim",
        "remove_prim",
        "delete_prim",
    ):
        assert forbidden not in sources


def test_scene_port_cannot_install_a_diagnostic_core():
    port = _port(torch.device("cpu"))
    owner, *_ = _fact_owner(device=torch.device("cpu"))
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="binding is HOLD"):
        port.install_physx_fact_owner(owner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_physx_owner_n64_mixed_plane_path_has_no_hot_host_sync():
    owner, centres, _ball, _red, _black = _fact_owner(
        device=torch.device("cuda"), n=64, k=3
    )
    owner.bind_expected_rubber_authority(_authority(owner))
    env = torch.arange(64, device=owner.device).view(64, 1)
    slot = torch.arange(3, device=owner.device).view(1, 3)
    centres[..., 0] = 1.0 + 0.01 * env + 0.1 * slot
    centres[..., 1] = 0.001 * env
    centres[..., 2] = 0.9 + 0.02 * slot
    owner.on_post_step_heartbeat(0.005)
    facts = _capture_owner(owner, centres)
    assert facts.producer_contract_fault.device.type == "cuda"
    assert torch.all(facts.producer_contract_fault)
    assert not torch.any(facts.selected_contact_event)
    source = inspect.getsource(S.IsaacPhysxBallFactOwner.capture)
    for forbidden in (".item(", ".cpu(", ".tolist(", ".numpy("):
        assert forbidden not in source
