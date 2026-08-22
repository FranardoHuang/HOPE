"""Fresh multi-flight ActionBall scene for the constructed full MDP.

This module is intentionally separate from ``attach_physical_ball_scene`` and
``PhysicalBallManager``.  The legacy path owns one ``pb_ball`` per environment
and cannot represent delayed settlement while a later shot is already in
flight.  The production path materializes exactly ``K`` independently
addressable rigid bodies per environment, where ``K`` is read from a fully
verified ``FrozenFlightCapacityReceipt``.  A separate exact type permits only
the code-owned K=2 plant used by the first disposable no-save diagnostic; it
has no formal capacity-receipt field.  Both shapes use this one port, while the
Physical owner separately refuses every portable/launch path for diagnostic K.

The Isaac imports are lazy so the scene contract and its tests stay runnable
with CUDA hidden.  Structural tests do not authorize runtime integration: the
real Isaac API binding and a full-scene Pod1 test are still explicit HOLDs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ast
import hashlib
import inspect
import math
from pathlib import Path
import sys
from typing import Callable, Mapping, Protocol, Sequence

import action_ball_physical_flight_contract as _flight


SCHEMA_VERSION = 1
CONTRACT_SOURCE_SHA256 = (
    "beff093949dda4fb8aab963217cd31d37c1d59875440a7043526864d224f2675"
)
SCENE_SPEC_KIND = "action_ball_full_mdp_ball_scene_spec_v1"
DIAGNOSTIC_SCENE_SPEC_KIND = (
    "action_ball_full_mdp_code_owned_diagnostic_ball_scene_spec_v1"
)
SCENE_ENTITY_PREFIX = "action_ball_flight_ball_"
SCENE_PRIM_PREFIX = "ActionBallFlightBall_"
LEGACY_SCENE_ENTITY_NAME = "pb_ball"
PARK_POSITION_ENV_M = (0.0, 0.0, -20.0)

RUNTIME_INTEGRATED = False
POD_FULL_SCENE_VALIDATED = False
LAUNCH_AUTHORIZED = False
INTEGRATION_RESIDUALS = (
    "bind the scene collection to the constructed Isaac env config",
    "exercise real RigidObject write/read APIs on Pod1 with K from the launch receipt",
    "prove post-physics ordering before R06 and no-fail armed child publication",
    "include the scene adapter in the whole-owner checkpoint and resume root",
    "bind real ball-racket, ball-net and ball-table event producers plus engine-overflow truth",
)

# The original 0807 preconverted A3 USD did not expose red/black as independent
# collider identities.  The diagnostic split-rubber derivative disables that
# merged collision subtree and adds named red/black/handle colliders below the
# same wrist actor.  Runtime authority is the live stage plus the PhysX contact
# header, never this source comment, a build digest, or collider geometry.
POSTPHYSICS_FACT_PRODUCERS_BOUND = False
POSTPHYSICS_CAPTURE_HOLD_REASONS = (
    "the split-rubber live stage and its exact callback collider tokens still "
    "require a fresh N=2 SimulationApp counterexample probe",
    "this code-owned diagnostic lane remains no-save/no-checkpoint and does "
    "not authorize formal launch or deployment",
)

RUBBER_INACTIVE = -1
RUBBER_RED = 0
RUBBER_BLACK = 1
CALLBACK_ORDER_CONTACT_BEFORE_HEARTBEAT = "contact_before_post_step_heartbeat"
CALLBACK_ORDER_HEARTBEAT_BEFORE_CONTACT = "post_step_heartbeat_before_contact"
CALLBACK_ORDER_SAME_STEP_CHRONOLOGY = "same_step_event_chronology"
_PINNED_ERROR_CLASSIFICATIONS = (
    "USD_LOAD_ERROR",
    "PHYSX_ERROR",
    "PHYSX_CUDA_ERROR",
    "PHYSX_TOO_MANY_ERRORS",
)
# The current error stream does not expose a documented, exact queue-overflow
# event.  Generic CUDA errors and "too many errors" are producer failures, not
# proof that contact reports were dropped.  Keep engine_overflow false until a
# real engine classification can support that narrower claim.
_OVERFLOW_ERROR_CLASSIFICATIONS: tuple[str, ...] = ()

_ISAAC_SCENE_PORT_CAPABILITY_TOKEN = object()
_ISAAC_SCENE_WRITE_TOKEN = object()
_PHYSX_FACT_OWNER_TOKEN = object()
_PHYSX_FACT_CHECKPOINT_TOKEN = object()
# ``/physics/disableContactProcessing`` is process-global in Kit.  At most one
# ActionBall scene owner may acquire it.  Shutdown releases this module's
# lease, but deliberately does not write ``True``: doing so could disable
# contact reporting underneath another engine consumer in the same process.
_CONTACT_PROCESSING_LEASE_OWNER: object | None = None


class ActionBallFullMdpBallSceneError(RuntimeError):
    """The fresh scene cannot be constructed or safely bound."""


@dataclass(frozen=True)
class CanonicalVenuePlanes:
    """Construction-pinned tracking-frame planes used by the fact owner."""

    near_table_x_m: float
    far_table_x_m: float
    table_half_width_m: float
    table_surface_z_m: float
    landing_ball_center_z_m: float
    net_x_m: float
    net_clear_ball_center_z_m: float

    def __post_init__(self) -> None:
        values = (
            self.near_table_x_m,
            self.far_table_x_m,
            self.table_half_width_m,
            self.table_surface_z_m,
            self.landing_ball_center_z_m,
            self.net_x_m,
            self.net_clear_ball_center_z_m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ActionBallFullMdpBallSceneError(
                "canonical venue planes must contain finite real numbers"
            )
        if not (
            self.near_table_x_m < self.net_x_m < self.far_table_x_m
            and self.table_half_width_m > 0.0
            and self.landing_ball_center_z_m > self.table_surface_z_m
            and self.net_clear_ball_center_z_m
            > self.landing_ball_center_z_m
        ):
            raise ActionBallFullMdpBallSceneError(
                "canonical venue plane order differs"
            )


def build_canonical_venue_planes() -> CanonicalVenuePlanes:
    """Build the scene-owned tracking-frame planes from canonical geometry.

    These are code construction constants, not caller evidence.  The live
    TableObstacle/TableNet prim inventory is checked separately by the scene
    installer before any callback subscription is accepted.
    """

    from whole_body_tracking.tasks.table_tennis import geometry

    near_x = 0.5
    surface_z = float(geometry.TABLE_HEIGHT)
    radius = float(geometry.BALL_RADIUS)
    return CanonicalVenuePlanes(
        near_table_x_m=near_x,
        far_table_x_m=near_x + float(geometry.TABLE_LENGTH),
        table_half_width_m=0.5 * float(geometry.TABLE_WIDTH),
        table_surface_z_m=surface_z,
        landing_ball_center_z_m=surface_z + radius,
        net_x_m=near_x + float(geometry.NET_X),
        net_clear_ball_center_z_m=(
            surface_z + float(geometry.NET_HEIGHT) + radius
        ),
    )


def _require_canonical_table_bounds(
    *,
    minimum_env_m: Sequence[float],
    maximum_env_m: Sequence[float],
    venue: CanonicalVenuePlanes,
    table_thickness_m: float,
) -> None:
    """Join the analytic planes to the composed live table collider bounds."""

    minimum = tuple(float(value) for value in minimum_env_m)
    maximum = tuple(float(value) for value in maximum_env_m)
    expected_minimum = (
        venue.near_table_x_m,
        -venue.table_half_width_m,
        venue.table_surface_z_m - float(table_thickness_m),
    )
    expected_maximum = (
        venue.far_table_x_m,
        venue.table_half_width_m,
        venue.table_surface_z_m,
    )
    if (
        len(minimum) != 3
        or len(maximum) != 3
        or not math.isfinite(float(table_thickness_m))
        or float(table_thickness_m) <= 0.0
        or any(not math.isfinite(value) for value in minimum + maximum)
        or any(
            abs(actual - expected) > 1.0e-5
            for actual, expected in zip(
                minimum + maximum, expected_minimum + expected_maximum
            )
        )
    ):
        raise ActionBallFullMdpBallSceneError(
            "live TableObstacle pose or size differs from canonical venue planes"
        )


def _require_composed_collider_mesh_arrays(
    *,
    name: str,
    actual_points: Sequence[Sequence[float]],
    actual_face_vertex_counts: Sequence[int],
    actual_face_vertex_indices: Sequence[int],
    actual_translate_in_wrist_m: Sequence[float],
    expected: object,
) -> None:
    """Reject a named live Mesh whose composed geometry differs from v3."""

    expected_name = getattr(expected, "name", None)
    expected_points = getattr(expected, "points", None)
    expected_counts = getattr(expected, "face_vertex_counts", None)
    expected_indices = getattr(expected, "face_vertex_indices", None)
    expected_translate = getattr(expected, "translate_in_wrist_m", None)
    try:
        points = tuple(tuple(float(value) for value in point) for point in actual_points)
        counts = tuple(int(value) for value in actual_face_vertex_counts)
        indices = tuple(int(value) for value in actual_face_vertex_indices)
        translate = tuple(float(value) for value in actual_translate_in_wrist_m)
    except (TypeError, ValueError) as exc:
        raise ActionBallFullMdpBallSceneError(
            f"live split-rubber Mesh arrays are malformed: {name}"
        ) from exc
    if (
        expected_name != name
        or points != expected_points
        or counts != expected_counts
        or indices != expected_indices
        or not isinstance(expected_translate, tuple)
        or len(translate) != 3
        or len(expected_translate) != 3
        or any(
            abs(actual - float(reference)) > 1.0e-9
            for actual, reference in zip(translate, expected_translate)
        )
    ):
        raise ActionBallFullMdpBallSceneError(
            f"live split-rubber composed Mesh geometry differs: {name}"
        )


def _require_mesh_collision_approximation(
    *, name: str, has_mesh_collision_api: bool, approximation: object
) -> None:
    if has_mesh_collision_api is not True or approximation != "convexHull":
        raise ActionBallFullMdpBallSceneError(
            f"live split-rubber MeshCollisionAPI differs: {name}"
        )


def _require_exact_table_collider_inventory(
    *, table_root: str, colliders: Sequence[tuple[str, str, bool, bool]]
) -> str:
    """Require IsaacLab's one measured static Cuboid collision child."""

    expected = f"{table_root}/geometry/mesh"
    if type(table_root) is not str or not table_root.startswith("/"):
        raise ActionBallFullMdpBallSceneError("live table root path differs")
    if tuple(colliders) != ((expected, "Cube", True, False),):
        raise ActionBallFullMdpBallSceneError(
            "live table collider inventory differs"
        )
    return expected


def _require_pre_attached_contact_report_prims(
    *, ball_prims: Sequence[tuple[str, object]], contact_report_api_type: object
) -> tuple[str, ...]:
    """Read only: prove every factory-owned ball was armed before attach."""

    applied: list[str] = []
    for path, prim in ball_prims:
        has_api = getattr(prim, "HasAPI", None)
        if not callable(has_api) or not bool(has_api(contact_report_api_type)):
            raise ActionBallFullMdpBallSceneError(
                "fresh ball lacks its pre-attach PhysxContactReportAPI: "
                f"{path}"
            )
        applied.append(path)
    return tuple(applied)


@dataclass(frozen=True)
class ExpectedRubberAuthorityView:
    """Diagnostic projection shape; it is not a production authority receipt.

    The class remains useful for exercising the fail-closed core, but is
    deliberately caller-constructible and therefore cannot authorize the live
    installer.  Production must replace it with Racket's exact owned view.
    """

    active_mask: object
    expected_rubber: object
    full_key_sha256: object
    ball_generation: object
    projection_sha256: str
    _owner_identity: object
    _token: object


@dataclass(frozen=True)
class PhysxFactOwnerCheckpoint:
    """Process-local diagnostic state; live handles are intentionally absent."""

    schema_version: int
    scene_identity_sha256: str
    callback_order: str
    expected_active: object
    expected_rubber: object
    expected_full_key_sha256: object
    expected_ball_generation: object
    previous_center_m: object
    previous_center_valid: object
    selected_contact_latch: object
    net_crossed_latch: object
    first_descending_crossing_latch: object
    last_heartbeat: int
    last_capture_heartbeat: int
    callback_sequence: int
    last_exact_stamp: tuple[int, int, int, int, int] | None
    overflow_sticky: bool
    producer_fault_sticky: bool
    wrong_face_event_count_by_ball: object
    scene_global_error_counts: tuple[tuple[str, int], ...]
    unknown_scene_global_error_count: int
    content_sha256: str
    _token: object


@dataclass(frozen=True)
class PhysxCallbackContact:
    """One decoded PhysX contact header plus same-callback ball-root sample."""

    ball_prim_path: str
    other_actor_path: str
    other_collider_path: str
    ball_center_env_m: tuple[float, float, float]
    callback_sequence: int
    physics_heartbeat: int


class _Subscription(Protocol):
    def unsubscribe(self) -> object: ...


def _plain_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ActionBallFullMdpBallSceneError(f"{label} must be a positive exact int")
    return value


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionBallFullMdpBallSceneError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ActionBallFullMdpBallSceneError(f"{label} must be finite and positive")
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionBallFullMdpBallSceneError(f"{label} must be one lowercase SHA-256")
    return value


def verify_frozen_physical_flight_contract_source() -> str:
    """Fail if the imported portable contract is not the reviewed source bytes."""

    path = Path(_flight.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != CONTRACT_SOURCE_SHA256:
        raise ActionBallFullMdpBallSceneError(
            "physical-flight contract source differs from the frozen review pin"
        )
    return actual


@dataclass(frozen=True)
class ActionBallFullMdpBallSceneSpec:
    """Exact scene materialization derived from one explicit C/H/K receipt."""

    schema_version: int
    kind: str
    contract_source_sha256: str
    capacity_receipt_sha256: str
    flight_capacity: int
    scene_entity_names: tuple[str, ...]
    prim_paths: tuple[str, ...]
    ball_radius_m: float
    ball_mass_kg: float
    park_position_env_m: tuple[float, float, float]
    collision_enabled: bool
    gravity_enabled: bool

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.kind != SCENE_SPEC_KIND:
            raise ActionBallFullMdpBallSceneError("scene spec schema/kind differs")
        if self.contract_source_sha256 != CONTRACT_SOURCE_SHA256:
            raise ActionBallFullMdpBallSceneError("scene spec contract source pin differs")
        _sha256(self.capacity_receipt_sha256, label="capacity_receipt_sha256")
        capacity = _plain_positive_int(self.flight_capacity, label="flight_capacity")
        expected_names = tuple(
            f"{SCENE_ENTITY_PREFIX}{index:03d}" for index in range(capacity)
        )
        expected_paths = tuple(
            f"{{ENV_REGEX_NS}}/{SCENE_PRIM_PREFIX}{index:03d}"
            for index in range(capacity)
        )
        if self.scene_entity_names != expected_names or self.prim_paths != expected_paths:
            raise ActionBallFullMdpBallSceneError("scene body identity/order differs")
        if LEGACY_SCENE_ENTITY_NAME in self.scene_entity_names:
            raise ActionBallFullMdpBallSceneError("legacy pb_ball is forbidden")
        _finite_positive(self.ball_radius_m, label="ball_radius_m")
        _finite_positive(self.ball_mass_kg, label="ball_mass_kg")
        if (
            len(self.park_position_env_m) != 3
            or any(not math.isfinite(float(value)) for value in self.park_position_env_m)
        ):
            raise ActionBallFullMdpBallSceneError("park_position_env_m differs")
        if type(self.collision_enabled) is not bool or not self.collision_enabled:
            raise ActionBallFullMdpBallSceneError(
                "fresh physical balls must retain collision-enabled plant physics"
            )
        if type(self.gravity_enabled) is not bool or not self.gravity_enabled:
            raise ActionBallFullMdpBallSceneError(
                "fresh physical balls must retain gravity-enabled plant physics"
            )

    @property
    def canonical_sha256(self) -> str:
        return _flight.canonical_sha256(self.to_mapping(include_digest=False))

    def to_mapping(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "contract_source_sha256": self.contract_source_sha256,
            "capacity_receipt_sha256": self.capacity_receipt_sha256,
            "flight_capacity": self.flight_capacity,
            "scene_entity_names": list(self.scene_entity_names),
            "prim_paths": list(self.prim_paths),
            "ball_radius_m": self.ball_radius_m,
            "ball_mass_kg": self.ball_mass_kg,
            "park_position_env_m": list(self.park_position_env_m),
            "collision_enabled": self.collision_enabled,
            "gravity_enabled": self.gravity_enabled,
        }
        if include_digest:
            payload["canonical_sha256"] = _flight.canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class ActionBallFullMdpDiagnosticBallSceneSpec:
    """Code-owned N=2 plant shape for a disposable no-save diagnostic.

    This is intentionally a separate type from the production receipt-backed
    spec.  ``formal_capacity_receipt_sha256`` is structurally absent rather
    than filled with a marker, so callers cannot mistake this diagnostic
    capacity for launch authority.
    """

    schema_version: int
    kind: str
    capacity_authority_kind: str
    formal_capacity_receipt_sha256: None
    flight_capacity: int
    scene_entity_names: tuple[str, ...]
    prim_paths: tuple[str, ...]
    ball_radius_m: float
    ball_mass_kg: float
    park_position_env_m: tuple[float, float, float]
    collision_enabled: bool
    gravity_enabled: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.kind != DIAGNOSTIC_SCENE_SPEC_KIND
        ):
            raise ActionBallFullMdpBallSceneError(
                "diagnostic scene spec schema/kind differs"
            )
        if type(self.capacity_authority_kind) is not str or not (
            self.capacity_authority_kind
        ):
            raise ActionBallFullMdpBallSceneError(
                "diagnostic capacity authority kind is missing"
            )
        if self.formal_capacity_receipt_sha256 is not None:
            raise ActionBallFullMdpBallSceneError(
                "diagnostic scene must not claim a formal capacity receipt"
            )
        capacity = _plain_positive_int(
            self.flight_capacity,
            label="flight_capacity",
        )
        if capacity != 2:
            raise ActionBallFullMdpBallSceneError(
                "diagnostic scene capacity must be exactly K=2"
            )
        expected_names = tuple(
            f"{SCENE_ENTITY_PREFIX}{index:03d}" for index in range(capacity)
        )
        expected_paths = tuple(
            f"{{ENV_REGEX_NS}}/{SCENE_PRIM_PREFIX}{index:03d}"
            for index in range(capacity)
        )
        if self.scene_entity_names != expected_names or self.prim_paths != expected_paths:
            raise ActionBallFullMdpBallSceneError(
                "diagnostic scene body identity/order differs"
            )
        _finite_positive(self.ball_radius_m, label="ball_radius_m")
        _finite_positive(self.ball_mass_kg, label="ball_mass_kg")
        if (
            len(self.park_position_env_m) != 3
            or any(
                not math.isfinite(float(value))
                for value in self.park_position_env_m
            )
        ):
            raise ActionBallFullMdpBallSceneError(
                "park_position_env_m differs"
            )
        if type(self.collision_enabled) is not bool or not self.collision_enabled:
            raise ActionBallFullMdpBallSceneError(
                "fresh physical balls must retain collision-enabled plant physics"
            )
        if type(self.gravity_enabled) is not bool or not self.gravity_enabled:
            raise ActionBallFullMdpBallSceneError(
                "fresh physical balls must retain gravity-enabled plant physics"
            )

    @property
    def canonical_sha256(self) -> str:
        return _flight.canonical_sha256(self.to_mapping(include_digest=False))

    def to_mapping(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "capacity_authority_kind": self.capacity_authority_kind,
            "formal_capacity_receipt_sha256": None,
            "flight_capacity": self.flight_capacity,
            "scene_entity_names": list(self.scene_entity_names),
            "prim_paths": list(self.prim_paths),
            "ball_radius_m": self.ball_radius_m,
            "ball_mass_kg": self.ball_mass_kg,
            "park_position_env_m": list(self.park_position_env_m),
            "collision_enabled": self.collision_enabled,
            "gravity_enabled": self.gravity_enabled,
        }
        if include_digest:
            payload["canonical_sha256"] = _flight.canonical_sha256(payload)
        return payload


def build_action_ball_full_mdp_ball_scene_spec(
    *,
    capacity_receipt: _flight.FrozenFlightCapacityReceipt,
    expected_capacity_receipt_sha256: str,
    ball_radius_m: float,
    ball_mass_kg: float,
) -> ActionBallFullMdpBallSceneSpec:
    """Build the scene spec without accepting a caller-supplied/default ``K``."""

    verify_frozen_physical_flight_contract_source()
    if type(capacity_receipt) is not _flight.FrozenFlightCapacityReceipt:
        raise ActionBallFullMdpBallSceneError(
            "capacity must be a frozen physical-flight capacity receipt"
        )
    expected = _sha256(
        expected_capacity_receipt_sha256,
        label="expected_capacity_receipt_sha256",
    )
    if capacity_receipt.canonical_sha256 != expected:
        raise ActionBallFullMdpBallSceneError("capacity receipt external pin differs")
    if capacity_receipt.integration_status != _flight.INTEGRATION_STATUS:
        raise ActionBallFullMdpBallSceneError("capacity receipt integration status differs")
    capacity = capacity_receipt.configured_flight_capacity
    if capacity != capacity_receipt.required_inclusive_flight_capacity:
        raise ActionBallFullMdpBallSceneError("configured K differs from frozen required K")
    return ActionBallFullMdpBallSceneSpec(
        schema_version=SCHEMA_VERSION,
        kind=SCENE_SPEC_KIND,
        contract_source_sha256=CONTRACT_SOURCE_SHA256,
        capacity_receipt_sha256=expected,
        flight_capacity=capacity,
        scene_entity_names=tuple(
            f"{SCENE_ENTITY_PREFIX}{index:03d}" for index in range(capacity)
        ),
        prim_paths=tuple(
            f"{{ENV_REGEX_NS}}/{SCENE_PRIM_PREFIX}{index:03d}"
            for index in range(capacity)
        ),
        ball_radius_m=_finite_positive(ball_radius_m, label="ball_radius_m"),
        ball_mass_kg=_finite_positive(ball_mass_kg, label="ball_mass_kg"),
        park_position_env_m=PARK_POSITION_ENV_M,
        collision_enabled=True,
        gravity_enabled=True,
    )


def _scene_has(scene: object, name: str) -> bool:
    if getattr(scene, name, None) is not None:
        return True
    if isinstance(scene, Mapping):
        return name in scene
    try:
        scene[name]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError):
        return False
    return True


def _require_replicated_source_scene_paths(
    scene: object,
    *,
    num_envs: int,
) -> tuple[str, ...]:
    """Bind the port to InteractiveScene's homogeneous source inheritance.

    With ``replicate_physics=True`` IsaacLab constructs env_1..N by inheriting
    env_0 (``copy_from_source=False``).  Composed collider content therefore
    has one source truth; concrete rigid/contact paths remain per-environment.
    """

    cfg = getattr(scene, "cfg", None)
    env_prim_paths = getattr(scene, "env_prim_paths", None)
    if getattr(cfg, "replicate_physics", None) is not True:
        raise ActionBallFullMdpBallSceneError(
            "fresh full-MDP scene requires homogeneous replicated physics"
        )
    if type(env_prim_paths) not in (list, tuple):
        raise ActionBallFullMdpBallSceneError(
            "fresh full-MDP scene does not expose concrete env prim paths"
        )
    paths = tuple(env_prim_paths)
    expected = tuple(f"/World/envs/env_{index}" for index in range(num_envs))
    if paths != expected:
        raise ActionBallFullMdpBallSceneError(
            "fresh full-MDP replicated env prim paths differ"
        )
    return paths


def _replicated_source_stage_row(
    stage_inventory: list[tuple[object, ...]],
    *,
    num_envs: int,
) -> tuple[object, ...]:
    """Return the sole composed-content authority row for a replicated scene."""

    if len(stage_inventory) != num_envs or num_envs <= 0:
        raise ActionBallFullMdpBallSceneError(
            "fresh full-MDP replicated stage inventory width differs"
        )
    row = stage_inventory[0]
    if len(row) != 7:
        raise ActionBallFullMdpBallSceneError(
            "fresh full-MDP replicated source inventory ABI differs"
        )
    return row


def attach_action_ball_full_mdp_ball_scene(
    env_cfg: object,
    *,
    spec: ActionBallFullMdpBallSceneSpec | ActionBallFullMdpDiagnosticBallSceneSpec,
) -> tuple[str, ...]:
    """Attach exactly K fresh rigid objects to an Isaac scene config.

    This function does not attach a table or racket and does not reuse any
    legacy physical-ball entity.  Those plant objects must already be owned by
    the constructed full scene.
    """

    if type(spec) not in (
        ActionBallFullMdpBallSceneSpec,
        ActionBallFullMdpDiagnosticBallSceneSpec,
    ):
        raise ActionBallFullMdpBallSceneError("scene spec must be builder-owned")
    verify_frozen_physical_flight_contract_source()
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        raise ActionBallFullMdpBallSceneError("env_cfg.scene is missing")
    if _scene_has(scene, LEGACY_SCENE_ENTITY_NAME):
        raise ActionBallFullMdpBallSceneError(
            "fresh full-MDP scene refuses legacy scene entity 'pb_ball'"
        )
    occupied = tuple(name for name in spec.scene_entity_names if _scene_has(scene, name))
    if occupied:
        raise ActionBallFullMdpBallSceneError(
            f"fresh physical scene entities already exist: {occupied!r}"
        )

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
    except Exception as exc:  # pragma: no cover - real Isaac import is a Pod gate.
        raise ActionBallFullMdpBallSceneError(
            "Isaac Lab scene APIs are unavailable; runtime remains HOLD"
        ) from exc

    for name, prim_path in zip(spec.scene_entity_names, spec.prim_paths):
        cfg = RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(pos=spec.park_position_env_m),
            spawn=sim_utils.SphereCfg(
                radius=spec.ball_radius_m,
                # Contact-report schema authoring must happen while Isaac Lab
                # spawns the rigid body, before SimulationContext.reset()
                # attaches Tensor API views.  Applying the schema later makes
                # PhysX rebuild ``geometry/mesh`` and invalidates every live
                # simulation view in the process.
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1.0e5,
                    max_depenetration_velocity=10.0,
                    enable_gyroscopic_forces=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=spec.ball_mass_kg),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.95, 0.95), roughness=0.4
                ),
            ),
        )
        setattr(scene, name, cfg)
    return spec.scene_entity_names


@dataclass(frozen=True)
class PrevalidatedIsaacSceneWrite:
    """Complete scene after-image.  Callers cannot add a row after preflight."""

    root_state_world_by_slot: tuple[object, ...]
    selected_mask_by_slot: tuple[object, ...]
    scratch_root_state_world_by_slot: tuple[object, ...]
    expected_num_envs: int
    expected_capacity: int
    _port_identity: object
    _write_nonce: int
    _token: object


@dataclass(frozen=True, eq=False)
class IsaacPhysicalFlightScenePortCapability:
    num_envs: int
    flight_capacity: int
    device_type: str
    device_index: int | None
    scene_spec_sha256: str
    _port_identity: object
    _token: object


@dataclass(frozen=True, eq=False)
class IsaacPhysicalFlightSceneApplyReceipt:
    write_nonce: int
    full_grid_write: bool
    readback_verified: bool
    _port_identity: object
    _handle_identity: object
    _token: object


@dataclass(frozen=True, eq=False)
class IsaacPhysicalFlightSceneAbortReceipt:
    write_nonce: int
    _port_identity: object
    _handle_identity: object
    _token: object


class PhysicalFlightScenePort(Protocol):
    """Minimal runtime port consumed by the physical-flight owner."""

    num_envs: int
    flight_capacity: int
    device: object

    def read_state_env(self) -> object: ...

    def preflight_write(
        self,
        state_env: object,
        selected_mask: object,
        *,
        reveal_boundary_receipt: object,
    ) -> object: ...

    def apply_prevalidated_write(self, handle: object) -> None: ...

    def bind_action_epoch_scene_writer(
        self, physical_owner: object, epoch_owner: object
    ) -> None: ...

    def preflight_action_epoch_write(self) -> object: ...


class IsaacPhysxBallFactOwner:
    """Low-level PhysX callback owner for selected contact and plane facts.

    Contact attribution is exact-path based: one fresh ball path plus the
    externally selected red/black collider path.  The common wrist actor and
    contact position are deliberately ignored.  Ball centre and the outgoing
    segment anchor are sampled from the ball's PhysX root in that same contact
    callback; under the current R06 contract they are the same ordered sample.

    The contact report is not used to infer net or table crossing.  Those facts
    come only from consecutive engine ball centres with half-open endpoint
    rules and one-shot latches.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        flight_capacity: int,
        device: object,
        scene_identity_sha256: str,
        concrete_ball_prim_paths: Sequence[Sequence[str]],
        red_rubber_collider_paths: Sequence[str],
        black_rubber_collider_paths: Sequence[str],
        venue: CanonicalVenuePlanes,
        center_sampler: Callable[[int, int], object],
        expected_authority_validator: Callable[[object], object],
        path_decoder: Callable[[object], str],
        callback_order: str,
        known_non_rubber_collider_bindings: Sequence[
            tuple[str, int, str, str]
        ] = (),
        wrist_actor_paths: Sequence[str] = (),
        _installer_token: object = None,
    ) -> None:
        import torch

        if _installer_token is not _PHYSX_FACT_OWNER_TOKEN:
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact owner must be constructed by the live installer"
            )
        self.num_envs = _plain_positive_int(num_envs, label="num_envs")
        self.flight_capacity = _plain_positive_int(
            flight_capacity, label="flight_capacity"
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.scene_identity_sha256 = _sha256(
            scene_identity_sha256, label="scene_identity_sha256"
        )
        if type(venue) is not CanonicalVenuePlanes:
            raise ActionBallFullMdpBallSceneError("venue must be canonical")
        self.venue = venue
        if callback_order not in (
            CALLBACK_ORDER_CONTACT_BEFORE_HEARTBEAT,
            CALLBACK_ORDER_HEARTBEAT_BEFORE_CONTACT,
            CALLBACK_ORDER_SAME_STEP_CHRONOLOGY,
        ):
            raise ActionBallFullMdpBallSceneError("callback_order is not pinned")
        self.callback_order = callback_order
        if not all(
            callable(value)
            for value in (
                center_sampler,
                expected_authority_validator,
                path_decoder,
            )
        ):
            raise ActionBallFullMdpBallSceneError(
                "fact owner callpoint is not callable"
            )
        self._center_sampler = center_sampler
        self._expected_authority_validator = expected_authority_validator
        self._path_decoder = path_decoder
        if (
            len(concrete_ball_prim_paths) != self.num_envs
            or any(
                len(row) != self.flight_capacity
                for row in concrete_ball_prim_paths
            )
            or len(red_rubber_collider_paths) != self.num_envs
            or len(black_rubber_collider_paths) != self.num_envs
            or len(wrist_actor_paths) not in (0, self.num_envs)
        ):
            raise ActionBallFullMdpBallSceneError(
                "fact owner path grid dimensions differ"
            )
        ball_map: dict[str, tuple[int, int]] = {}
        rubber_map: dict[str, tuple[int, int]] = {}
        for env_index, row in enumerate(concrete_ball_prim_paths):
            for slot_index, raw in enumerate(row):
                path = self._exact_prim_path(raw, label="ball prim path")
                if path in ball_map:
                    raise ActionBallFullMdpBallSceneError(
                        "ball prim paths are not unique"
                    )
                ball_map[path] = (env_index, slot_index)
            red = self._exact_prim_path(
                red_rubber_collider_paths[env_index], label="red rubber collider"
            )
            black = self._exact_prim_path(
                black_rubber_collider_paths[env_index], label="black rubber collider"
            )
            if red == black or red in rubber_map or black in rubber_map:
                raise ActionBallFullMdpBallSceneError(
                    "rubber collider paths are not distinct"
                )
            rubber_map[red] = (env_index, RUBBER_RED)
            rubber_map[black] = (env_index, RUBBER_BLACK)
        if set(ball_map).intersection(rubber_map):
            raise ActionBallFullMdpBallSceneError(
                "ball and rubber collider identities overlap"
            )
        self._ball_path_to_cell = ball_map
        self._rubber_path_to_env_and_side = rubber_map
        self.concrete_ball_prim_paths = tuple(
            tuple(row) for row in concrete_ball_prim_paths
        )
        self._ordered_ball_paths = tuple(
            path
            for row in self.concrete_ball_prim_paths
            for path in row
        )
        self.red_rubber_collider_paths = tuple(red_rubber_collider_paths)
        self.black_rubber_collider_paths = tuple(black_rubber_collider_paths)
        self._wrist_actor_paths = tuple(
            self._exact_prim_path(value, label="wrist rigid actor")
            for value in wrist_actor_paths
        )
        known_non_rubber: dict[str, tuple[int, str, str]] = {}
        for binding in known_non_rubber_collider_bindings:
            if (
                type(binding) is not tuple
                or len(binding) != 4
                or type(binding[1]) is not int
                or binding[1] < 0
                or binding[1] >= self.num_envs
                or binding[2]
                not in ("handle", "wrist_shell", "old_merged", "table")
            ):
                raise ActionBallFullMdpBallSceneError(
                    "known non-rubber collider binding ABI differs"
                )
            path = self._exact_prim_path(
                binding[0], label="known non-rubber collider"
            )
            actor = self._exact_prim_path(
                binding[3], label="known non-rubber rigid actor"
            )
            if path in known_non_rubber:
                raise ActionBallFullMdpBallSceneError(
                    "known non-rubber collider paths are not unique"
                )
            known_non_rubber[path] = (binding[1], binding[2], actor)
        self._known_non_rubber_path_to_binding = known_non_rubber
        if (
            set(known_non_rubber).intersection(ball_map)
            or set(known_non_rubber).intersection(rubber_map)
        ):
            raise ActionBallFullMdpBallSceneError(
                "known non-rubber collider identities overlap ball/rubber paths"
            )

        shape = (self.num_envs, self.flight_capacity)
        self._expected_active = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._expected_rubber = torch.full(
            shape, RUBBER_INACTIVE, dtype=torch.int8, device=self.device
        )
        self._expected_key = torch.zeros(
            shape + (32,), dtype=torch.uint8, device=self.device
        )
        self._expected_generation = torch.full(
            shape, -1, dtype=torch.int64, device=self.device
        )
        self._bound_authority: ExpectedRubberAuthorityView | None = None
        self._bound_projection_sha256: str | None = None
        self._previous_center = torch.zeros(
            shape + (3,), dtype=torch.float32, device=self.device
        )
        self._previous_valid = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._contact_latch = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._contact_candidate_event = torch.zeros(
            shape + (2,), dtype=torch.bool, device=self.device
        )
        self._contact_candidate_center = torch.zeros(
            shape + (2, 3), dtype=torch.float32, device=self.device
        )
        self._contact_candidate_heartbeat = torch.full(
            shape + (2,), -1, dtype=torch.int64, device=self.device
        )
        self._known_non_rubber_candidate_event = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._binding_fault = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._net_latch = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._landing_latch = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._last_heartbeat = 0
        self._last_capture_heartbeat = 0
        self._callback_sequence = 0
        self._last_exact_stamp: tuple[int, int, int, int, int] | None = None
        self._overflow_sticky = False
        self._producer_fault_sticky = False
        self._wrong_face_event_count = torch.zeros(
            shape, dtype=torch.int64, device=self.device
        )
        self._scene_global_error_counts = {
            name: 0 for name in _PINNED_ERROR_CLASSIFICATIONS
        }
        self._unknown_scene_global_error_count = 0
        self._contact_subscription: object | None = None
        self._heartbeat_subscription: object | None = None
        self._error_subscription: object | None = None
        self._applied_contact_report_paths: tuple[str, ...] = ()
        self._live_subscription_epoch: object | None = None
        self._closed = False
        self._contact_processing_lease_owned = False
        self._identity = object()
        # No Python-private token is authority.  This core is permanently
        # diagnostic until a separately typed production owner exists.
        self._diagnostic_unauthorized = True
        self._action_epoch_direct_binding = False
        self._action_epoch_idle_binding = False

    @classmethod
    def _diagnostic_unauthorized_for_test(
        cls, *, _test_token: object, **kwargs
    ) -> IsaacPhysxBallFactOwner:
        """Build the algorithmic core without claiming live production facts."""

        if _test_token is not _PHYSX_FACT_CHECKPOINT_TOKEN:
            raise ActionBallFullMdpBallSceneError(
                "diagnostic PhysX core token differs"
            )
        return cls(_installer_token=_PHYSX_FACT_OWNER_TOKEN, **kwargs)

    @staticmethod
    def _exact_prim_path(value: object, *, label: str) -> str:
        if type(value) is not str or not value.startswith("/") or "//" in value:
            raise ActionBallFullMdpBallSceneError(
                f"{label} must be an absolute concrete prim path"
            )
        return value

    def _tensor(self, value: object, *, shape: tuple[int, ...], dtype, label: str):
        import torch

        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != self.device
        ):
            raise ActionBallFullMdpBallSceneError(f"{label} tensor ABI differs")
        return value

    @property
    def subscriptions_bound(self) -> bool:
        return (
            self._contact_subscription is not None
            and self._heartbeat_subscription is not None
            and self._error_subscription is not None
            and self._applied_contact_report_paths
            == self._ordered_ball_paths
            and self._live_subscription_epoch is not None
        )

    def bind_expected_rubber_authority(
        self, value: object
    ) -> ExpectedRubberAuthorityView:
        """Retain a diagnostic projection before stepping.

        This method cannot authorize production; the production installer and
        scene-port binding both remain fail-closed until the external Racket
        owner exposes its non-constructible exact view.
        """

        import torch

        if self._bound_authority is not None:
            raise ActionBallFullMdpBallSceneError(
                "expected-rubber authority already awaits capture"
            )
        validated = self._expected_authority_validator(value)
        if validated is not value or type(value) is not ExpectedRubberAuthorityView:
            raise ActionBallFullMdpBallSceneError(
                "expected-rubber authority is foreign or self-asserted"
            )
        shape = (self.num_envs, self.flight_capacity)
        active = self._tensor(
            value.active_mask, shape=shape, dtype=torch.bool, label="active_mask"
        )
        rubber = self._tensor(
            value.expected_rubber,
            shape=shape,
            dtype=torch.int8,
            label="expected_rubber",
        )
        key = self._tensor(
            value.full_key_sha256,
            shape=shape + (32,),
            dtype=torch.uint8,
            label="full_key_sha256",
        )
        generation = self._tensor(
            value.ball_generation,
            shape=shape,
            dtype=torch.int64,
            label="ball_generation",
        )
        projection = _sha256(value.projection_sha256, label="projection_sha256")
        valid_rubber = (rubber == RUBBER_RED) | (rubber == RUBBER_BLACK)
        bad = (
            (active & ~valid_rubber)
            | (~active & (rubber != RUBBER_INACTIVE))
            | (active & torch.eq(key, 0).all(dim=-1))
            | (active & (generation < 0))
        )
        if self.device.type == "cpu" and bool(bad.any()):
            raise ActionBallFullMdpBallSceneError(
                "expected-rubber projection contains an invalid live row"
            )
        if self.device.type != "cpu":
            # The exact row fault is transferred by Physical's one existing
            # packet.  No callback-path device-to-host synchronization occurs.
            self._producer_fault_sticky = self._producer_fault_sticky or False
        new_identity = active & (
            ~self._expected_active
            | ~torch.eq(key, self._expected_key).all(dim=-1)
            | (generation != self._expected_generation)
        )
        inactive = ~active
        reset = new_identity | inactive
        self._previous_valid.logical_and_(~reset)
        self._contact_latch.logical_and_(~reset)
        self._net_latch.logical_and_(~reset)
        self._landing_latch.logical_and_(~reset)
        self._contact_candidate_event.zero_()
        self._known_non_rubber_candidate_event.zero_()
        self._binding_fault.copy_(bad)
        self._expected_active.copy_(active)
        self._expected_rubber.copy_(rubber)
        self._expected_key.copy_(key)
        self._expected_generation.copy_(generation)
        self._bound_authority = value
        self._bound_projection_sha256 = projection
        return value

    def _bind_action_epoch_expected_rubber(
        self,
        *,
        active_mask: object,
        expected_rubber: object,
        ball_generation: object,
        full_key_sha256: object,
        _installer_token: object,
    ) -> None:
        """Bind exact owner-joined face truth without legacy digest carriers."""

        import torch

        if _installer_token is not _PHYSX_FACT_OWNER_TOKEN:
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch selected-rubber binding is scene-installer-only"
            )
        if self._bound_authority is not None or self._action_epoch_direct_binding:
            raise ActionBallFullMdpBallSceneError(
                "one selected-rubber binding already awaits capture"
            )
        shape = (self.num_envs, self.flight_capacity)
        active = self._tensor(
            active_mask, shape=shape, dtype=torch.bool, label="active_mask"
        )
        rubber = self._tensor(
            expected_rubber,
            shape=shape,
            dtype=torch.int8,
            label="expected_rubber",
        )
        generation = self._tensor(
            ball_generation,
            shape=shape,
            dtype=torch.int64,
            label="ball_generation",
        )
        full_key = self._tensor(
            full_key_sha256,
            shape=shape + (32,),
            dtype=torch.uint8,
            label="full_key_sha256",
        )
        bad = (
            active
            & (rubber != RUBBER_RED)
            & (rubber != RUBBER_BLACK)
        ) | (~active & (rubber != RUBBER_INACTIVE)) | (active & (generation < 0)) | (
            active & torch.eq(full_key, 0).all(dim=-1)
        )
        new_identity = active & (
            ~self._expected_active
            | ~torch.eq(full_key, self._expected_key).all(dim=-1)
            | (generation != self._expected_generation)
            | (rubber != self._expected_rubber)
        )
        reset = new_identity | ~active
        self._previous_valid.logical_and_(~reset)
        self._contact_latch.logical_and_(~reset)
        self._net_latch.logical_and_(~reset)
        self._landing_latch.logical_and_(~reset)
        self._contact_candidate_event.zero_()
        self._known_non_rubber_candidate_event.zero_()
        self._binding_fault.copy_(bad)
        self._expected_active.copy_(active)
        self._expected_rubber.copy_(rubber)
        self._expected_generation.copy_(generation)
        self._expected_key.copy_(full_key)
        self._action_epoch_idle_binding = False
        self._action_epoch_direct_binding = True

    def _action_epoch_activity_mask(self):
        import torch

        if (
            self._bound_authority is not None
            or self._bound_projection_sha256 is not None
            or self._action_epoch_direct_binding
            or self._action_epoch_idle_binding
        ):
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch activity census crossed an armed callback epoch"
            )
        # Sticky producer faults belong to the next live fact row.  They do
        # not make an otherwise empty scene transaction active: dense-empty
        # capture cannot consume or expose them, so using them as activity
        # would permanently re-enable the full K-grid hot path after one
        # global engine error.
        return (
            self._expected_active
            | self._contact_candidate_event.any(dim=-1)
            | self._known_non_rubber_candidate_event
            | self._binding_fault
        ).detach().clone()

    def _begin_action_epoch_idle_binding(self) -> None:
        """Open one callback epoch after all writers proved the grid idle."""

        if (
            self._bound_authority is not None
            or self._bound_projection_sha256 is not None
            or self._action_epoch_direct_binding
            or self._action_epoch_idle_binding
        ):
            raise ActionBallFullMdpBallSceneError(
                "one callback binding already awaits capture"
            )
        self._action_epoch_idle_binding = True
        self._action_epoch_direct_binding = True

    def _complete_action_epoch_idle_binding(
        self, *, exact_stamp: object
    ) -> None:
        """Seal one idle callback epoch without reading any scene tensor."""

        exact = exact_stamp
        if (
            type(exact) is not tuple
            or len(exact) != 5
            or any(type(value) is not int for value in exact)
        ):
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch idle binding exact stamp ABI differs"
            )
        if (
            not self._action_epoch_idle_binding
            or not self._action_epoch_direct_binding
            or self._bound_authority is not None
            or self._bound_projection_sha256 is not None
        ):
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch idle binding ACK is missing, stale, or foreign"
            )
        if self._last_exact_stamp is not None and exact <= self._last_exact_stamp:
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch idle binding stamp is duplicate or non-monotonic"
            )
        self._contact_candidate_event.zero_()
        self._known_non_rubber_candidate_event.zero_()
        self._binding_fault.zero_()
        self._action_epoch_idle_binding = False
        self._action_epoch_direct_binding = False
        self._last_capture_heartbeat = self._last_heartbeat
        self._last_exact_stamp = exact

    def abort_expected_rubber_authority(self, value: object) -> None:
        """Cancel one unstepped projection without manufacturing fact rows."""

        if value is not self._bound_authority:
            raise ActionBallFullMdpBallSceneError(
                "expected-rubber abort authority is stale or foreign"
            )
        self._bound_authority = None
        self._bound_projection_sha256 = None
        self._contact_candidate_event.zero_()
        self._known_non_rubber_candidate_event.zero_()
        self._binding_fault.zero_()

    def bind_subscriptions(
        self,
        *,
        applied_ball_prim_paths: Sequence[str],
        contact_subscription: object,
        heartbeat_subscription: object,
        error_subscription: object,
    ) -> None:
        """Reject caller-provided handles; they are not engine evidence."""

        raise ActionBallFullMdpBallSceneError(
            "direct subscription binding cannot authorize PhysX facts"
        )

    def _bind_live_subscriptions(
        self,
        *,
        applied_ball_prim_paths: Sequence[str],
        contact_subscription: object,
        heartbeat_subscription: object,
        error_subscription: object,
        _installer_token: object,
        _subscription_epoch: object,
    ) -> None:
        """Bind only handles minted by this module's live PhysX installer."""

        expected = self._ordered_ball_paths
        actual = tuple(applied_ball_prim_paths)
        if _installer_token is not _PHYSX_FACT_OWNER_TOKEN or actual != expected or any(
            value is None
            for value in (
                contact_subscription,
                heartbeat_subscription,
                error_subscription,
                _subscription_epoch,
            )
        ):
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact subscriptions/contact-report paths differ"
            )
        if self.subscriptions_bound:
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact subscriptions were rebound"
            )
        self._applied_contact_report_paths = actual
        self._contact_subscription = contact_subscription
        self._heartbeat_subscription = heartbeat_subscription
        self._error_subscription = error_subscription
        self._live_subscription_epoch = _subscription_epoch

    def _acquire_process_global_contact_processing(
        self, *, settings_iface: object, physx_iface: object
    ) -> None:
        """Verify the pre-attach Kit setting and acquire this scene's lease.

        This setting is a necessary process-global precondition, not contact
        authority.  Only an observed contact callback can prove a contact.
        """

        global _CONTACT_PROCESSING_LEASE_OWNER

        if self._last_heartbeat != 0 or self._callback_sequence != 0:
            raise ActionBallFullMdpBallSceneError(
                "contact processing was enabled after a physics callback"
            )
        is_running = getattr(physx_iface, "is_running", None)
        if (
            _CONTACT_PROCESSING_LEASE_OWNER is not None
            and _CONTACT_PROCESSING_LEASE_OWNER is not self._identity
        ):
            raise ActionBallFullMdpBallSceneError(
                "process-global contact processing already has another scene owner"
            )
        getter = getattr(settings_iface, "get", None)
        if (
            not callable(is_running)
            or is_running() is not True
            or not callable(getter)
            or getter("/physics/disableContactProcessing") is not False
        ):
            raise ActionBallFullMdpBallSceneError(
                "contact processing was not enabled before simulation attachment"
            )
        _CONTACT_PROCESSING_LEASE_OWNER = self._identity
        self._contact_processing_lease_owned = True

    def install_live_physx_subscriptions(
        self, *, stage: object, post_step_order: int = 100
    ) -> None:
        """Validate named colliders, arm fresh balls and subscribe once."""

        global _CONTACT_PROCESSING_LEASE_OWNER

        try:
            import carb
            from pxr import PhysxSchema
            from omni.physx import (
                get_physx_interface,
                get_physx_simulation_interface,
            )
        except Exception as exc:  # pragma: no cover - Pod Kit gate.
            raise ActionBallFullMdpBallSceneError(
                "live PhysX fact APIs are unavailable"
            ) from exc
        physx = get_physx_interface()
        ball_prims: list[tuple[str, object]] = []
        for path in self._ordered_ball_paths:
            prim = stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                raise ActionBallFullMdpBallSceneError(
                    f"fresh ball prim is missing: {path}"
                )
            ball_prims.append((path, prim))
        from pxr import Usd, UsdGeom, UsdPhysics
        from whole_body_tracking.tasks.table_tennis import geometry

        def require_collision(path: str, *, enabled: bool, label: str) -> object:
            prim = stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                raise ActionBallFullMdpBallSceneError(
                    f"named {label} collider prim is missing: {path}"
                )
            has_collision = getattr(prim, "HasAPI", None)
            if not callable(has_collision) or not bool(
                has_collision(UsdPhysics.CollisionAPI)
            ):
                raise ActionBallFullMdpBallSceneError(
                    f"named {label} prim is not a collision source: {path}"
                )
            enabled_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            if not enabled_attr.IsValid() or bool(enabled_attr.Get()) is not enabled:
                raise ActionBallFullMdpBallSceneError(
                    f"named {label} collisionEnabled differs: {path}"
                )
            return prim

        for label, paths in (
            ("red rubber", self.red_rubber_collider_paths),
            ("black rubber", self.black_rubber_collider_paths),
        ):
            for path in paths:
                try:
                    require_collision(path, enabled=True, label=label)
                except BaseException as exc:
                    raise ActionBallFullMdpBallSceneError(
                        f"named {label} collider API cannot be queried: {path}"
                    ) from exc
        self._acquire_process_global_contact_processing(
            settings_iface=carb.settings.get_settings(), physx_iface=physx
        )
        try:
            applied = _require_pre_attached_contact_report_prims(
                ball_prims=ball_prims,
                contact_report_api_type=PhysxSchema.PhysxContactReportAPI,
            )
        except BaseException:
            if _CONTACT_PROCESSING_LEASE_OWNER is self._identity:
                _CONTACT_PROCESSING_LEASE_OWNER = None
            self._contact_processing_lease_owned = False
            raise
        epoch = object()

        def contact_callback(headers: object, contact_data: object) -> None:
            self._on_epoch_contact_report(epoch, headers, contact_data)

        def heartbeat_callback(dt: float) -> None:
            self._on_epoch_post_step_heartbeat(epoch, dt)

        def error_callback(event: object) -> None:
            self._on_epoch_error_event(epoch, event)

        handles: list[object] = []
        try:
            simulation = get_physx_simulation_interface()
            contact = simulation.subscribe_contact_report_events(contact_callback)
            handles.append(contact)
            heartbeat = physx.subscribe_physics_on_step_events(
                heartbeat_callback, False, post_step_order
            )
            handles.append(heartbeat)
            stream = physx.get_error_event_stream()
            error = stream.create_subscription_to_pop(
                error_callback, name="ActionBall fresh PhysX fact owner"
            )
            handles.append(error)
            self._bind_live_subscriptions(
                applied_ball_prim_paths=applied,
                contact_subscription=contact,
                heartbeat_subscription=heartbeat,
                error_subscription=error,
                _installer_token=_PHYSX_FACT_OWNER_TOKEN,
                _subscription_epoch=epoch,
            )
        except BaseException:
            for handle in reversed(handles):
                unsubscribe = getattr(handle, "unsubscribe", None)
                if callable(unsubscribe):
                    try:
                        unsubscribe()
                    except BaseException:
                        self._producer_fault_sticky = True
            if _CONTACT_PROCESSING_LEASE_OWNER is self._identity:
                _CONTACT_PROCESSING_LEASE_OWNER = None
            self._contact_processing_lease_owned = False
            raise

    def _require_current_subscription_epoch(self, epoch: object) -> bool:
        if epoch is not self._live_subscription_epoch:
            self._producer_fault_sticky = True
            return False
        return True

    def _on_epoch_post_step_heartbeat(self, epoch: object, dt: float) -> None:
        if self._require_current_subscription_epoch(epoch):
            self.on_post_step_heartbeat(dt)

    def _on_epoch_contact_report(
        self, epoch: object, headers: object, contact_data: object
    ) -> None:
        if self._require_current_subscription_epoch(epoch):
            self.on_contact_report(headers, contact_data)

    def _on_epoch_error_event(self, epoch: object, event: object) -> None:
        if self._require_current_subscription_epoch(epoch):
            self.on_error_event(event)

    def on_post_step_heartbeat(self, _dt: float) -> None:
        self._callback_sequence += 1
        self._last_heartbeat += 1

    def _decode(self, value: object) -> str:
        result = self._path_decoder(value)
        if type(result) is not str or not result.startswith("/"):
            raise ActionBallFullMdpBallSceneError(
                "PhysX callback path decoder returned no concrete prim path"
            )
        return result

    def on_contact_report(self, headers: object, _contact_data: object) -> None:
        """Latch only CONTACT_FOUND for exact ball + expected-rubber collider."""

        import torch

        self._callback_sequence += 1
        if self._bound_authority is None and not self._action_epoch_direct_binding:
            self._producer_fault_sticky = True
            return
        try:
            iterator = iter(headers)
        except TypeError:
            self._producer_fault_sticky = True
            return
        for header in iterator:
            event_name = getattr(getattr(header, "type", None), "name", "")
            if event_name not in ("CONTACT_FOUND", "CONTACT_FOUND_EVENT"):
                continue
            try:
                collider0 = self._decode(getattr(header, "collider0"))
                collider1 = self._decode(getattr(header, "collider1"))
                actor0 = self._decode(getattr(header, "actor0"))
                actor1 = self._decode(getattr(header, "actor1"))
            except BaseException:
                self._producer_fault_sticky = True
                continue
            left_actor = self._ball_path_to_cell.get(actor0)
            left_collider = self._ball_path_to_cell.get(collider0)
            right_actor = self._ball_path_to_cell.get(actor1)
            right_collider = self._ball_path_to_cell.get(collider1)
            if (
                left_actor is not None
                and left_collider is not None
                and left_actor != left_collider
            ) or (
                right_actor is not None
                and right_collider is not None
                and right_actor != right_collider
            ):
                self._producer_fault_sticky = True
                continue
            left = left_actor if left_actor is not None else left_collider
            right = right_actor if right_actor is not None else right_collider
            if left is not None and right is not None:
                self._producer_fault_sticky = True
                continue
            if left is not None:
                cell = left
                rubber_path = collider1
                other_actor_path = actor1
            elif right is not None:
                cell = right
                rubber_path = collider0
                other_actor_path = actor0
            else:
                # Contact reports can contain unrelated pairs when another API
                # is installed elsewhere.  They are outside this owner's scope.
                continue
            env_index, slot_index = cell
            if (
                self.callback_order
                in (
                    CALLBACK_ORDER_HEARTBEAT_BEFORE_CONTACT,
                    CALLBACK_ORDER_SAME_STEP_CHRONOLOGY,
                )
                and self._last_heartbeat <= self._last_capture_heartbeat
            ):
                # Production's measured callback order is heartbeat then
                # contact.  A contact injected after capture(t) but before
                # heartbeat(t+1) is stale and must not be relabelled as t+1.
                self._binding_fault[env_index, slot_index] = True
                continue
            expected_wrist_actor = (
                self._wrist_actor_paths[env_index]
                if self._wrist_actor_paths
                else None
            )
            rubber = self._rubber_path_to_env_and_side.get(rubber_path)
            if rubber is None:
                # A live, explicitly inventoried handle/wrist/venue collider
                # is a normal non-selected contact.  Any other spelling could
                # be an instance/prototype alias for a selected rubber and
                # therefore must not be silently converted into a miss.
                known = self._known_non_rubber_path_to_binding.get(rubber_path)
                if known is None:
                    self._producer_fault_sticky = True
                else:
                    known_env, known_role, expected_actor = known
                    if (
                        known_env != env_index
                        or known_role == "old_merged"
                        or other_actor_path != expected_actor
                    ):
                        self._producer_fault_sticky = True
                    else:
                        self._known_non_rubber_candidate_event[
                            env_index, slot_index
                        ] = True
                continue
            rubber_env, rubber_side = rubber
            if rubber_env != env_index or (
                expected_wrist_actor is not None
                and other_actor_path != expected_wrist_actor
            ):
                self._producer_fault_sticky = True
                continue
            try:
                center = self._tensor(
                    self._center_sampler(env_index, slot_index),
                    shape=(3,),
                    dtype=torch.float32,
                    label="callback ball centre",
                )
            except BaseException:
                self._producer_fault_sticky = True
                continue
            # Store callback sequence, not a caller-pinned callback order.
            # Capture accepts a contact if it occurred since the preceding
            # capture and the post-step heartbeat also advanced.  This covers
            # both PhysX callback orders without mistaking a stale callback
            # from an older step for the current step.
            heartbeat = self._last_heartbeat
            self._contact_candidate_center[
                env_index, slot_index, rubber_side
            ].copy_(center)
            self._contact_candidate_heartbeat[
                env_index, slot_index, rubber_side
            ] = heartbeat
            self._contact_candidate_event[
                env_index, slot_index, rubber_side
            ] = True

    def on_error_event(self, event: object) -> None:
        raw = getattr(event, "type", event)
        name = getattr(raw, "name", str(raw).split(".")[-1])
        if name not in _PINNED_ERROR_CLASSIFICATIONS:
            self._unknown_scene_global_error_count += 1
            self._producer_fault_sticky = True
            return
        self._scene_global_error_counts[name] += 1
        self._producer_fault_sticky = True
        if name in _OVERFLOW_ERROR_CLASSIFICATIONS:
            self._overflow_sticky = True

    def capture(self, *, request: object, live_state: object, facts_type: type, stamp_type: type):
        """Materialize one exact Physical packet after callbacks and scene update."""

        import torch

        shape = (self.num_envs, self.flight_capacity)
        observe = self._tensor(
            getattr(request, "observe_mask", None),
            shape=shape,
            dtype=torch.bool,
            label="request observe_mask",
        )
        current = self._tensor(
            live_state,
            shape=shape + (13,),
            dtype=torch.float32,
            label="live state",
        )[..., :3]
        key = self._tensor(
            getattr(request, "full_key_sha256", None),
            shape=shape + (32,),
            dtype=torch.uint8,
            label="request full_key_sha256",
        )
        generation = self._tensor(
            getattr(request, "ball_generation", None),
            shape=shape,
            dtype=torch.int64,
            label="request ball_generation",
        )
        exact = getattr(request, "exact_stamp", None)
        if self._action_epoch_idle_binding:
            raise ActionBallFullMdpBallSceneError(
                "dense capture cannot consume an idle callback binding"
            )
        legacy_bound = (
            self._bound_authority is not None
            and self._bound_projection_sha256 is not None
        )
        authority_bound = legacy_bound or self._action_epoch_direct_binding
        if (
            type(exact) is not tuple
            or len(exact) != 5
            or any(type(value) is not int for value in exact)
        ):
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact owner lacks one exact expected-rubber capture authority"
            )
        if self._last_exact_stamp is not None and exact <= self._last_exact_stamp:
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact capture stamp is duplicate or non-monotonic"
            )
        control, substep, _decimation, _sim_step, _phase = exact
        identity_fault = observe & (
            ~self._expected_active
            | ~torch.eq(key, self._expected_key).all(dim=-1)
            | (generation != self._expected_generation)
        )
        unbound_fault = observe & ~torch.full(
            shape,
            authority_bound,
            dtype=torch.bool,
            device=self.device,
        )
        rubber_index = self._expected_rubber.to(dtype=torch.int64).clamp(0, 1)
        selected_candidate = torch.gather(
            self._contact_candidate_event, -1, rubber_index.unsqueeze(-1)
        ).squeeze(-1)
        selected_candidate_heartbeat = torch.gather(
            self._contact_candidate_heartbeat, -1, rubber_index.unsqueeze(-1)
        ).squeeze(-1)
        selected_candidate_center = torch.gather(
            self._contact_candidate_center,
            -2,
            rubber_index.unsqueeze(-1).unsqueeze(-1).expand(shape + (1, 3)),
        ).squeeze(-2)
        other_rubber_index = 1 - rubber_index
        wrong_face_candidate = torch.gather(
            self._contact_candidate_event,
            -1,
            other_rubber_index.unsqueeze(-1),
        ).squeeze(-1)
        wrong_face = observe & wrong_face_candidate
        self._wrong_face_event_count.add_(wrong_face.to(dtype=torch.int64))
        ambiguous_face = observe & selected_candidate & (
            wrong_face_candidate | self._known_non_rubber_candidate_event
        )
        heartbeat_fault = observe & (
            (self._last_heartbeat <= self._last_capture_heartbeat)
            | (
                selected_candidate
                & (
                    (selected_candidate_heartbeat < self._last_capture_heartbeat)
                    | (selected_candidate_heartbeat > self._last_heartbeat)
                )
            )
        )
        callback_fault = observe & ~torch.full(
            shape,
            self.subscriptions_bound,
            dtype=torch.bool,
            device=self.device,
        )
        nonfinite = observe & ~torch.isfinite(current).all(dim=-1)
        producer_fault = (
            identity_fault
            | unbound_fault
            | heartbeat_fault
            | callback_fault
            | self._binding_fault
            | ambiguous_face
        )
        if self._diagnostic_unauthorized and not self._action_epoch_direct_binding:
            producer_fault = producer_fault | observe
        if self._producer_fault_sticky:
            producer_fault = producer_fault | observe
        engine_overflow = torch.full(
            shape,
            self._overflow_sticky,
            dtype=torch.bool,
            device=self.device,
        ) & observe

        contact = (
            observe
            & selected_candidate
            & ~self._contact_latch
            & ~producer_fault
        )
        contact_center = torch.where(
            contact.unsqueeze(-1),
            selected_candidate_center,
            torch.zeros_like(current),
        )
        # Contact centre and the R06 outgoing-segment *start* are the same
        # ball-root sample in CONTACT_FOUND.  This is not a post-impact sample
        # and must never be described as one.  The contact point is ignored.
        outgoing_anchor = contact_center.clone()
        self._contact_latch.logical_or_(contact)
        segment_start = torch.where(
            contact.unsqueeze(-1), outgoing_anchor, self._previous_center
        )
        has_segment = observe & (self._previous_valid | contact)
        finite_segment = (
            torch.isfinite(segment_start).all(dim=-1)
            & torch.isfinite(current).all(dim=-1)
        )
        net_cross = (
            has_segment
            & self._contact_latch
            & ~self._net_latch
            & finite_segment
            & (segment_start[..., 0] < self.venue.net_x_m)
            & (current[..., 0] >= self.venue.net_x_m)
        )
        net_den = current[..., 0] - segment_start[..., 0]
        net_alpha = (self.venue.net_x_m - segment_start[..., 0]) / torch.where(
            net_cross, net_den, torch.ones_like(net_den)
        )
        net_z = segment_start[..., 2] + net_alpha * (
            current[..., 2] - segment_start[..., 2]
        )
        net_clear = net_cross & (net_z >= self.venue.net_clear_ball_center_z_m)
        landing_cross = (
            has_segment
            & self._contact_latch
            & ~self._landing_latch
            & finite_segment
            & (segment_start[..., 2] > self.venue.landing_ball_center_z_m)
            & (current[..., 2] <= self.venue.landing_ball_center_z_m)
            & (current[..., 2] < segment_start[..., 2])
        )
        landing_den = segment_start[..., 2] - current[..., 2]
        landing_alpha = (
            segment_start[..., 2] - self.venue.landing_ball_center_z_m
        ) / torch.where(landing_cross, landing_den, torch.ones_like(landing_den))
        landing_xy = segment_start[..., :2] + landing_alpha.unsqueeze(-1) * (
            current[..., :2] - segment_start[..., :2]
        )
        self._net_latch.logical_or_(net_cross)
        self._landing_latch.logical_or_(landing_cross)
        self._previous_center.copy_(
            torch.where(observe.unsqueeze(-1), current, self._previous_center)
        )
        self._previous_valid.logical_or_(observe)

        def stamp(active, event_phase: int):
            return stamp_type(
                control_step=torch.where(
                    active,
                    torch.full(shape, control, dtype=torch.int64, device=self.device),
                    torch.full(shape, -1, dtype=torch.int64, device=self.device),
                ),
                physics_substep=torch.where(
                    active,
                    torch.full(shape, substep, dtype=torch.int32, device=self.device),
                    torch.full(shape, -1, dtype=torch.int32, device=self.device),
                ),
                event_phase=torch.where(
                    active,
                    torch.full(shape, event_phase, dtype=torch.int8, device=self.device),
                    torch.full(shape, -1, dtype=torch.int8, device=self.device),
                ),
            )

        result = facts_type(
            observation_stamp=stamp(observe, 2),
            current_state_env_f32=live_state,
            selected_contact_event=contact.clone(),
            selected_contact_ball_center_m=contact_center.clone(),
            selected_contact_outgoing_segment_anchor_m=outgoing_anchor,
            selected_contact_stamp=stamp(contact, 0),
            net_crossing_event=net_cross.clone(),
            net_clear_at_crossing=net_clear.clone(),
            net_crossing_stamp=stamp(net_cross, 1),
            crossing_report_delivered=self._contact_latch.clone() & observe,
            first_descending_crossing_event=landing_cross.clone(),
            first_descending_crossing_xy_m=torch.where(
                landing_cross.unsqueeze(-1), landing_xy, torch.zeros_like(landing_xy)
            ),
            first_descending_crossing_stamp=stamp(landing_cross, 2),
            nonfinite_observation=nonfinite.clone(),
            producer_contract_fault=producer_fault.clone(),
            engine_overflow=engine_overflow.clone(),
            _owner_identity=request,
            _capture_token=request._token,
        )
        self._contact_candidate_event.zero_()
        self._known_non_rubber_candidate_event.zero_()
        self._binding_fault.zero_()
        self._bound_authority = None
        self._bound_projection_sha256 = None
        self._action_epoch_idle_binding = False
        self._action_epoch_direct_binding = False
        self._last_capture_heartbeat = self._last_heartbeat
        self._last_exact_stamp = exact
        return result

    def shutdown(self) -> None:
        """Drop live handles.  Checkpoint restore must subscribe afresh."""

        global _CONTACT_PROCESSING_LEASE_OWNER

        if self._closed:
            return
        # Invalidate first.  A callback queued inside ``unsubscribe`` can no
        # longer be admitted as current even if the engine invokes it while
        # the remaining handles are being drained.
        self._closed = True
        self._live_subscription_epoch = None
        handles = (
            self._contact_subscription,
            self._heartbeat_subscription,
            self._error_subscription,
        )
        self._contact_subscription = None
        self._heartbeat_subscription = None
        self._error_subscription = None
        self._applied_contact_report_paths = ()
        unsubscribe_failed = False
        for handle in handles:
            unsubscribe = getattr(handle, "unsubscribe", None)
            if handle is not None and not callable(unsubscribe):
                unsubscribe_failed = True
                continue
            if callable(unsubscribe):
                try:
                    unsubscribe()
                except BaseException:
                    unsubscribe_failed = True
        if (
            self._contact_processing_lease_owned
            and _CONTACT_PROCESSING_LEASE_OWNER is self._identity
        ):
            _CONTACT_PROCESSING_LEASE_OWNER = None
        self._contact_processing_lease_owned = False
        if unsubscribe_failed:
            self._producer_fault_sticky = True
            raise ActionBallFullMdpBallSceneError(
                "PhysX subscription teardown was not acknowledged"
            )

    def diagnostic_telemetry(self) -> Mapping[str, object]:
        """Return non-authorizing attribution diagnostics.

        Error counts are scene-global.  The per-row ``engine_overflow`` bit in
        a capture is only a conservative broadcast to live rows and must not
        be summed as independent per-ball engine failures.
        """

        return {
            "scene_global_error_counts": tuple(
                (name, self._scene_global_error_counts[name])
                for name in _PINNED_ERROR_CLASSIFICATIONS
            ),
            "wrong_face_event_count_by_ball": self._wrong_face_event_count.detach().clone(),
            "wrong_face_count_semantics": "capture_batches_with_known_opposite_rubber",
            "engine_overflow_attribution": "scene_global_broadcast_to_live_rows",
            "unknown_scene_global_error_count": self._unknown_scene_global_error_count,
        }

    @staticmethod
    def _checkpoint_content_sha256(value: PhysxFactOwnerCheckpoint) -> str:
        import torch

        digest = hashlib.sha256()
        for field in (
            value.schema_version,
            value.scene_identity_sha256,
            value.callback_order,
            value.last_heartbeat,
            value.last_capture_heartbeat,
            value.callback_sequence,
            value.last_exact_stamp,
            value.overflow_sticky,
            value.producer_fault_sticky,
            value.scene_global_error_counts,
            value.unknown_scene_global_error_count,
        ):
            digest.update(repr(field).encode("utf-8"))
            digest.update(b"\0")
        for tensor in (
            value.expected_active,
            value.expected_rubber,
            value.expected_full_key_sha256,
            value.expected_ball_generation,
            value.previous_center_m,
            value.previous_center_valid,
            value.selected_contact_latch,
            value.net_crossed_latch,
            value.first_descending_crossing_latch,
            value.wrong_face_event_count_by_ball,
        ):
            if not isinstance(tensor, torch.Tensor):
                return ""
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()

    def checkpoint_projection(self) -> PhysxFactOwnerCheckpoint:
        import torch

        if (
            self._bound_authority is not None
            or self._bound_projection_sha256 is not None
            or self._action_epoch_direct_binding
            or self._action_epoch_idle_binding
            or bool(torch.any(self._contact_candidate_event))
            or bool(torch.any(self._binding_fault))
        ):
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact checkpoint requires a drained capture boundary"
            )
        checkpoint = PhysxFactOwnerCheckpoint(
            schema_version=2,
            scene_identity_sha256=self.scene_identity_sha256,
            callback_order=self.callback_order,
            expected_active=self._expected_active.detach().clone(),
            expected_rubber=self._expected_rubber.detach().clone(),
            expected_full_key_sha256=self._expected_key.detach().clone(),
            expected_ball_generation=self._expected_generation.detach().clone(),
            previous_center_m=self._previous_center.detach().clone(),
            previous_center_valid=self._previous_valid.detach().clone(),
            selected_contact_latch=self._contact_latch.detach().clone(),
            net_crossed_latch=self._net_latch.detach().clone(),
            first_descending_crossing_latch=self._landing_latch.detach().clone(),
            last_heartbeat=self._last_heartbeat,
            last_capture_heartbeat=self._last_capture_heartbeat,
            callback_sequence=self._callback_sequence,
            last_exact_stamp=self._last_exact_stamp,
            overflow_sticky=self._overflow_sticky,
            producer_fault_sticky=self._producer_fault_sticky,
            wrong_face_event_count_by_ball=(
                self._wrong_face_event_count.detach().clone()
            ),
            scene_global_error_counts=tuple(
                (name, self._scene_global_error_counts[name])
                for name in _PINNED_ERROR_CLASSIFICATIONS
            ),
            unknown_scene_global_error_count=self._unknown_scene_global_error_count,
            content_sha256="0" * 64,
            _token=_PHYSX_FACT_CHECKPOINT_TOKEN,
        )
        return replace(
            checkpoint,
            content_sha256=self._checkpoint_content_sha256(checkpoint),
        )

    def restore_checkpoint_projection(self, value: object) -> None:
        import torch

        shape = (self.num_envs, self.flight_capacity)
        error_counts_valid = False
        if type(value) is PhysxFactOwnerCheckpoint:
            counts = value.scene_global_error_counts
            error_counts_valid = (
                type(counts) is tuple
                and len(counts) == len(_PINNED_ERROR_CLASSIFICATIONS)
                and all(
                    type(pair) is tuple
                    and len(pair) == 2
                    and pair[0] == expected_name
                    and type(pair[1]) is int
                    and pair[1] >= 0
                    for pair, expected_name in zip(
                        counts, _PINNED_ERROR_CLASSIFICATIONS
                    )
                )
            )
        if (
            type(value) is not PhysxFactOwnerCheckpoint
            or value._token is not _PHYSX_FACT_CHECKPOINT_TOKEN
            or value.schema_version != 2
            or value.scene_identity_sha256 != self.scene_identity_sha256
            or value.callback_order != self.callback_order
            or type(value.last_heartbeat) is not int
            or type(value.last_capture_heartbeat) is not int
            or value.last_heartbeat < 0
            or value.last_capture_heartbeat < 0
            or value.last_heartbeat < value.last_capture_heartbeat
            or type(value.callback_sequence) is not int
            or value.callback_sequence < value.last_heartbeat
            or (
                value.last_exact_stamp is not None
                and (
                    type(value.last_exact_stamp) is not tuple
                    or len(value.last_exact_stamp) != 5
                    or any(type(part) is not int for part in value.last_exact_stamp)
                )
            )
            or type(value.overflow_sticky) is not bool
            or type(value.producer_fault_sticky) is not bool
            or type(value.unknown_scene_global_error_count) is not int
            or value.unknown_scene_global_error_count < 0
            or _sha256(value.content_sha256, label="checkpoint content_sha256")
            != self._checkpoint_content_sha256(
                replace(value, content_sha256="0" * 64)
            )
            or not error_counts_valid
            or self._bound_authority is not None
            or self._bound_projection_sha256 is not None
            or self._action_epoch_direct_binding
            or self._action_epoch_idle_binding
            or any(
                handle is not None
                for handle in (
                    self._contact_subscription,
                    self._heartbeat_subscription,
                    self._error_subscription,
                    self._live_subscription_epoch,
                )
            )
            or bool(torch.any(self._contact_candidate_event))
            or bool(torch.any(self._binding_fault))
        ):
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact checkpoint identity/header differs"
            )
        expected_active = self._tensor(
            value.expected_active,
            shape=shape,
            dtype=torch.bool,
            label="checkpoint expected_active",
        )
        expected_rubber = self._tensor(
            value.expected_rubber,
            shape=shape,
            dtype=torch.int8,
            label="checkpoint expected_rubber",
        )
        expected_key = self._tensor(
            value.expected_full_key_sha256,
            shape=shape + (32,),
            dtype=torch.uint8,
            label="checkpoint expected_full_key_sha256",
        )
        expected_generation = self._tensor(
            value.expected_ball_generation,
            shape=shape,
            dtype=torch.int64,
            label="checkpoint expected_ball_generation",
        )
        previous = self._tensor(
            value.previous_center_m,
            shape=shape + (3,),
            dtype=torch.float32,
            label="checkpoint previous_center_m",
        )
        valid = self._tensor(
            value.previous_center_valid,
            shape=shape,
            dtype=torch.bool,
            label="checkpoint previous_center_valid",
        )
        contact = self._tensor(
            value.selected_contact_latch,
            shape=shape,
            dtype=torch.bool,
            label="checkpoint selected_contact_latch",
        )
        net = self._tensor(
            value.net_crossed_latch,
            shape=shape,
            dtype=torch.bool,
            label="checkpoint net_crossed_latch",
        )
        landing = self._tensor(
            value.first_descending_crossing_latch,
            shape=shape,
            dtype=torch.bool,
            label="checkpoint first_descending_crossing_latch",
        )
        wrong_face_count = self._tensor(
            value.wrong_face_event_count_by_ball,
            shape=shape,
            dtype=torch.int64,
            label="checkpoint wrong_face_event_count_by_ball",
        )
        active_semantics_bad = (
            expected_active
            & (
                ((expected_rubber != RUBBER_RED) & (expected_rubber != RUBBER_BLACK))
                | torch.eq(expected_key, 0).all(dim=-1)
                | (expected_generation < 0)
            )
        ) | (
            ~expected_active
            & (expected_rubber != RUBBER_INACTIVE)
        )
        state_semantics_bad = (
            (contact & ~expected_active)
            | (net & ~contact)
            | (landing & ~contact)
            | (valid & ~torch.isfinite(previous).all(dim=-1))
            | (wrong_face_count < 0)
        )
        if bool(torch.any(active_semantics_bad | state_semantics_bad)):
            raise ActionBallFullMdpBallSceneError(
                "PhysX fact checkpoint semantic relations differ"
            )
        self._expected_active.copy_(expected_active)
        self._expected_rubber.copy_(expected_rubber)
        self._expected_key.copy_(expected_key)
        self._expected_generation.copy_(expected_generation)
        self._previous_center.copy_(previous)
        self._previous_valid.copy_(valid)
        self._contact_latch.copy_(contact)
        self._net_latch.copy_(net)
        self._landing_latch.copy_(landing)
        self._last_heartbeat = value.last_heartbeat
        self._last_capture_heartbeat = value.last_capture_heartbeat
        self._callback_sequence = value.callback_sequence
        self._last_exact_stamp = value.last_exact_stamp
        self._overflow_sticky = value.overflow_sticky
        self._producer_fault_sticky = value.producer_fault_sticky
        self._wrong_face_event_count.copy_(wrong_face_count)
        self._scene_global_error_counts = dict(value.scene_global_error_counts)
        self._unknown_scene_global_error_count = value.unknown_scene_global_error_count
        self._action_epoch_idle_binding = False
        self._action_epoch_direct_binding = False


def _install_isaac_physx_ball_fact_owner(
    *,
    port: object,
    stage: object,
    concrete_ball_prim_paths: Sequence[Sequence[str]],
    red_rubber_collider_paths: Sequence[str],
    black_rubber_collider_paths: Sequence[str],
    venue: CanonicalVenuePlanes,
    expected_authority_validator: Callable[[object], object],
    path_decoder: Callable[[object], str],
    callback_order: str,
    known_non_rubber_collider_bindings: Sequence[
        tuple[str, int, str, str]
    ] = (),
    wrist_actor_paths: Sequence[str] = (),
    post_step_order: int = 100,
) -> IsaacPhysxBallFactOwner:
    """Install the exact live core after concrete stage/path validation.

    This low-level constructor remains intentionally private to the scene
    port's owner-bound installer.  Its path/validator arguments are not an
    authority surface; the public port resolves all paths from the live stage
    and supplies the exact cold-bound Racket method itself.
    """

    return _live_probe_isaac_physx_ball_fact_owner(
        port=port,
        stage=stage,
        concrete_ball_prim_paths=concrete_ball_prim_paths,
        red_rubber_collider_paths=red_rubber_collider_paths,
        black_rubber_collider_paths=black_rubber_collider_paths,
        venue=venue,
        expected_authority_validator=expected_authority_validator,
        path_decoder=path_decoder,
        callback_order=callback_order,
        known_non_rubber_collider_bindings=known_non_rubber_collider_bindings,
        wrist_actor_paths=wrist_actor_paths,
        post_step_order=post_step_order,
    )


def _live_probe_isaac_physx_ball_fact_owner(
    *,
    port: object,
    stage: object,
    concrete_ball_prim_paths: Sequence[Sequence[str]],
    red_rubber_collider_paths: Sequence[str],
    black_rubber_collider_paths: Sequence[str],
    venue: CanonicalVenuePlanes,
    expected_authority_validator: Callable[[object], object],
    path_decoder: Callable[[object], str],
    callback_order: str,
    known_non_rubber_collider_bindings: Sequence[
        tuple[str, int, str, str]
    ] = (),
    wrist_actor_paths: Sequence[str] = (),
    post_step_order: int = 100,
) -> IsaacPhysxBallFactOwner:
    """Diagnostic-only live probe; never installs into the production port."""

    import torch

    if type(port) is not IsaacLabPhysicalFlightScenePort:
        raise ActionBallFullMdpBallSceneError(
            "fact owner requires the exact Isaac scene port"
        )
    if len(concrete_ball_prim_paths) != port.num_envs:
        raise ActionBallFullMdpBallSceneError(
            "concrete ball path grid env width differs"
        )

    def sample_center(env_index: int, slot_index: int):
        asset = port.assets[slot_index]
        view = getattr(asset, "root_physx_view", None)
        getter = getattr(view, "get_transforms", None)
        if not callable(getter):
            raise ActionBallFullMdpBallSceneError(
                "ball PhysX root transform getter is unavailable"
            )
        transforms = getter()
        if (
            not isinstance(transforms, torch.Tensor)
            or tuple(transforms.shape) != (port.num_envs, 7)
            or transforms.dtype != torch.float32
            or transforms.device != port.device
        ):
            raise ActionBallFullMdpBallSceneError(
                "ball PhysX root transform tensor ABI differs"
            )
        return transforms[env_index, :3] - port.env_origins[env_index]

    owner = IsaacPhysxBallFactOwner(
        num_envs=port.num_envs,
        flight_capacity=port.flight_capacity,
        device=port.device,
        scene_identity_sha256=port.spec.canonical_sha256,
        concrete_ball_prim_paths=concrete_ball_prim_paths,
        red_rubber_collider_paths=red_rubber_collider_paths,
        black_rubber_collider_paths=black_rubber_collider_paths,
        venue=venue,
        center_sampler=sample_center,
        expected_authority_validator=expected_authority_validator,
        path_decoder=path_decoder,
        callback_order=callback_order,
        known_non_rubber_collider_bindings=known_non_rubber_collider_bindings,
        wrist_actor_paths=wrist_actor_paths,
        _installer_token=_PHYSX_FACT_OWNER_TOKEN,
    )
    owner.install_live_physx_subscriptions(
        stage=stage, post_step_order=post_step_order
    )
    return owner


class IsaacLabPhysicalFlightScenePort:
    """Thin adapter around K Isaac ``RigidObject`` instances.

    Construction validates every field/method that can be checked without a
    physics step.  Isaac calls can still fail asynchronously; the owner treats
    any such exception as poison and never fabricates rollback.
    """

    def __init__(
        self,
        *,
        scene: object,
        spec: ActionBallFullMdpBallSceneSpec | ActionBallFullMdpDiagnosticBallSceneSpec,
        env_origins: object,
    ) -> None:
        import torch

        if type(spec) not in (
            ActionBallFullMdpBallSceneSpec,
            ActionBallFullMdpDiagnosticBallSceneSpec,
        ):
            raise ActionBallFullMdpBallSceneError("scene spec must be builder-owned")
        if _scene_has(scene, LEGACY_SCENE_ENTITY_NAME):
            raise ActionBallFullMdpBallSceneError("legacy pb_ball cannot enter fresh port")
        assets: list[object] = []
        for name in spec.scene_entity_names:
            try:
                asset = scene[name]  # type: ignore[index]
            except (KeyError, TypeError) as exc:
                raise ActionBallFullMdpBallSceneError(
                    f"fresh scene body {name!r} is missing"
                ) from exc
            data = getattr(asset, "data", None)
            root = getattr(data, "root_state_w", None)
            if (
                not isinstance(root, torch.Tensor)
                or root.ndim != 2
                or root.shape[1] != 13
                or root.dtype != torch.float32
                or not callable(getattr(asset, "write_root_pose_to_sim", None))
                or not callable(getattr(asset, "write_root_velocity_to_sim", None))
            ):
                raise ActionBallFullMdpBallSceneError(
                    f"fresh scene body {name!r} does not expose the pinned root-state API"
                )
            assets.append(asset)
        if not isinstance(env_origins, torch.Tensor) or env_origins.shape != (
            assets[0].data.root_state_w.shape[0],
            3,
        ):
            raise ActionBallFullMdpBallSceneError("env_origins shape differs")
        if env_origins.dtype != torch.float32:
            raise ActionBallFullMdpBallSceneError("env_origins dtype must be float32")
        env_prim_paths = _require_replicated_source_scene_paths(
            scene,
            num_envs=int(env_origins.shape[0]),
        )
        device = assets[0].data.root_state_w.device
        if env_origins.device != device or any(
            asset.data.root_state_w.device != device
            or asset.data.root_state_w.shape[0] != env_origins.shape[0]
            for asset in assets
        ):
            raise ActionBallFullMdpBallSceneError("scene body device/env width differs")
        self.spec = spec
        self.assets = tuple(assets)
        self.env_origins = env_origins
        self.num_envs = int(env_origins.shape[0])
        self._env_prim_paths = env_prim_paths
        self.flight_capacity = spec.flight_capacity
        self.device = device
        self._identity = object()
        # Fixed at construction.  Runtime writes never derive a dynamic CUDA
        # index set from the selected mask.
        self._all_env_ids = torch.arange(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._next_write_nonce = 1
        self._active_write_handles: dict[int, PrevalidatedIsaacSceneWrite] = {}
        self._scene_port_capability = IsaacPhysicalFlightScenePortCapability(
            num_envs=self.num_envs,
            flight_capacity=self.flight_capacity,
            device_type=self.device.type,
            device_index=self.device.index,
            scene_spec_sha256=self.spec.canonical_sha256,
            _port_identity=self._identity,
            _token=_ISAAC_SCENE_PORT_CAPABILITY_TOKEN,
        )
        self._physx_fact_owner: IsaacPhysxBallFactOwner | None = None
        # The lean launch path has one cold-bound writer.  Hot launch receives
        # no tensors, verdicts, digests, or receipts from its caller: it pulls
        # the complete after-image through Physical's exact retained epoch
        # projection.  This is causal ownership, not same-writer comparison.
        self._action_epoch_physical_owner: object | None = None
        self._action_epoch_owner: object | None = None
        self._action_epoch_racket_owner: object | None = None

    def install_action_epoch_live_physx_fact_owner(
        self,
        *,
        stage: object,
    ) -> None:
        """Resolve canonical live prims and install one exact subscriber set."""

        if (
            self._action_epoch_physical_owner is None
            or self._action_epoch_owner is None
            or self._action_epoch_racket_owner is None
            or self._physx_fact_owner is not None
        ):
            raise ActionBallFullMdpBallSceneError(
                "live PhysX fact owner requires the three cold-bound owners"
            )
        get_prim = getattr(stage, "GetPrimAtPath", None)
        if not callable(get_prim):
            raise ActionBallFullMdpBallSceneError("live USD stage API differs")
        from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
            action_ball_full_mdp_split_asset as split_asset,
        )

        expected_geometry = (
            split_asset.action_ball_full_mdp_expected_collider_geometry()
        )
        if type(expected_geometry) is not split_asset.ActionBallFullMdpExpectedColliderGeometry:
            raise ActionBallFullMdpBallSceneError(
                "v3 enclosed collider geometry projection type differs"
            )
        expected_meshes = {
            mesh.name: mesh for mesh in expected_geometry.meshes
        }
        if (
            len(expected_geometry.meshes) != 4
            or len(expected_meshes) != 4
            or any(
                type(mesh) is not split_asset.ActionBallFullMdpExpectedColliderMesh
                for mesh in expected_geometry.meshes
            )
        ):
            raise ActionBallFullMdpBallSceneError(
                "v3 enclosed collider geometry inventory differs"
            )
        concrete_ball_paths: list[tuple[str, ...]] = []
        red_paths: list[str] = []
        black_paths: list[str] = []
        wrist_actor_paths: list[str] = []
        non_rubber_bindings: list[tuple[str, int, str, str]] = []
        stage_inventory: list[tuple[object, ...]] = []
        for env_index, env_root in enumerate(self._env_prim_paths):
            robot = f"{env_root}/Robot"
            wrist = f"{robot}/right_wrist_yaw_Link/action_ball_named_colliders"
            wrist_actor = f"{robot}/right_wrist_yaw_Link"
            red = f"{wrist}/red_rubber_collider"
            black = f"{wrist}/black_rubber_collider"
            handle = f"{wrist}/racket_handle_collider"
            wrist_shell = f"{wrist}/wrist_shell_collider"
            old_merged = f"{robot}/right_wrist_yaw_Link/collisions"
            table_root = f"{env_root}/TableObstacle"
            # Full-MDP's current top-only table cfg intentionally has no
            # TableNet collision prim.  Net crossing is derived from ordered
            # ball centres, never inferred from collision callbacks.
            for label, path in (
                ("environment", env_root),
                ("robot", robot),
                ("wrist actor", wrist_actor),
                ("red rubber", red),
                ("black rubber", black),
                ("racket handle", handle),
                ("wrist shell", wrist_shell),
                ("old merged wrist collision", old_merged),
                ("table wrapper", table_root),
            ):
                prim = get_prim(path)
                if prim is None or not prim.IsValid():
                    raise ActionBallFullMdpBallSceneError(
                        f"live split-rubber {label} prim is missing: {path}"
                    )
            row = tuple(
                f"{env_root}/{SCENE_PRIM_PREFIX}{slot:03d}"
                for slot in range(self.flight_capacity)
            )
            if any(
                get_prim(path) is None or not get_prim(path).IsValid()
                for path in row
            ):
                raise ActionBallFullMdpBallSceneError(
                    f"live ActionBall K-body prim grid is missing for env {env_index}"
                )
            concrete_ball_paths.append(row)
            red_paths.append(red)
            black_paths.append(black)
            wrist_actor_paths.append(wrist_actor)
            non_rubber_bindings.extend(
                (
                    (handle, env_index, "handle", wrist_actor),
                    (wrist_shell, env_index, "wrist_shell", wrist_actor),
                    (old_merged, env_index, "old_merged", wrist_actor),
                )
            )
            stage_inventory.append(
                (red, black, handle, wrist_shell, old_merged, table_root, row)
            )

        # Validate the collision semantics of the exact live stage, not an
        # offline asset receipt.  Named face/handle/table sources must be
        # enabled; the retired merged wrist subtree must remain disabled.
        from pxr import Usd, UsdGeom, UsdPhysics
        from whole_body_tracking.tasks.table_tennis import geometry

        def live_collision_enabled(path: str, *, label: str) -> bool:
            prim = get_prim(path)
            api = UsdPhysics.CollisionAPI(prim)
            attr = api.GetCollisionEnabledAttr()
            if not api or not attr.IsValid():
                raise ActionBallFullMdpBallSceneError(
                    f"live split-rubber {label} has no CollisionAPI: {path}"
                )
            value = attr.Get()
            if type(value) is not bool:
                raise ActionBallFullMdpBallSceneError(
                    f"live split-rubber {label} collisionEnabled is absent: {path}"
                )
            return value

        # IsaacLab's exact homogeneous scene contract above says env_1..N
        # inherit composed content from env_0.  Validate the complete Mesh,
        # collision and table-bounds truth once at that source.  Concrete
        # paths and every ball rigid actor remain checked for every env.
        source_stage_row = _replicated_source_stage_row(
            stage_inventory,
            num_envs=self.num_envs,
        )
        for (
            red,
            black,
            handle,
            wrist_shell,
            old_merged,
            table_root,
            row,
        ) in (source_stage_row,):
            parent = get_prim(str(red).rsplit("/", 1)[0])
            wrist_actor = parent.GetParent()
            if not bool(wrist_actor.HasAPI(UsdPhysics.RigidBodyAPI)):
                raise ActionBallFullMdpBallSceneError(
                    "live split-rubber wrist owner is not a rigid actor"
                )
            children = tuple(parent.GetChildren())
            child_names = frozenset(child.GetName() for child in children)
            if len(children) != 4 or child_names != frozenset(
                {
                    "racket_handle_collider",
                    "wrist_shell_collider",
                    "black_rubber_collider",
                    "red_rubber_collider",
                }
            ):
                raise ActionBallFullMdpBallSceneError(
                    "live split-rubber named collider inventory differs"
                )
            if any(
                bool(child.HasAPI(UsdPhysics.RigidBodyAPI))
                for child in parent.GetChildren()
            ):
                raise ActionBallFullMdpBallSceneError(
                    "named split-rubber collider became an independent rigid actor"
                )
            for child in children:
                name = child.GetName()
                mesh = UsdGeom.Mesh(child)
                points = mesh.GetPointsAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                xform_order = child.GetAttribute("xformOpOrder").Get()
                translate = child.GetAttribute("xformOp:translate").Get()
                if (
                    not mesh
                    or points is None
                    or counts is None
                    or indices is None
                    or tuple(xform_order or ()) != ("xformOp:translate",)
                    or translate is None
                ):
                    raise ActionBallFullMdpBallSceneError(
                        f"live split-rubber composed Mesh ABI differs: {name}"
                    )
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(child)
                approximation = mesh_collision_api.GetApproximationAttr().Get()
                _require_mesh_collision_approximation(
                    name=name,
                    has_mesh_collision_api=bool(
                        child.HasAPI(UsdPhysics.MeshCollisionAPI)
                    ),
                    approximation=approximation,
                )
                _require_composed_collider_mesh_arrays(
                    name=name,
                    actual_points=points,
                    actual_face_vertex_counts=counts,
                    actual_face_vertex_indices=indices,
                    actual_translate_in_wrist_m=translate,
                    expected=expected_meshes.get(name),
                )
            for label, path in (
                ("red rubber", red),
                ("black rubber", black),
                ("racket handle", handle),
                ("wrist shell", wrist_shell),
            ):
                if live_collision_enabled(path, label=label) is not True:
                    raise ActionBallFullMdpBallSceneError(
                        f"live split-rubber {label} is disabled: {path}"
                    )
            if live_collision_enabled(
                old_merged, label="old merged wrist collision"
            ) is not False:
                raise ActionBallFullMdpBallSceneError(
                    "retired merged wrist collision remains enabled"
                )
            table_prim = get_prim(table_root)
            table_colliders = tuple(
                descendant
                for descendant in Usd.PrimRange(table_prim)
                if bool(descendant.HasAPI(UsdPhysics.CollisionAPI))
            )
            table_collider_path = _require_exact_table_collider_inventory(
                table_root=table_root,
                colliders=tuple(
                    (
                        str(collider.GetPath()),
                        str(collider.GetTypeName()),
                        live_collision_enabled(
                            str(collider.GetPath()), label="table collider"
                        ),
                        bool(collider.HasAPI(UsdPhysics.RigidBodyAPI)),
                    )
                    for collider in table_colliders
                ),
            )
            # The table's static actor path is not inferred from its stage
            # hierarchy.  Raw PhysX headers must measure it before table-only
            # contact can be classified as ordinary no-contact.
            relative_bounds = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=False,
            ).ComputeRelativeBound(table_prim, table_prim.GetParent())
            aligned = relative_bounds.ComputeAlignedRange()
            _require_canonical_table_bounds(
                minimum_env_m=aligned.GetMin(),
                maximum_env_m=aligned.GetMax(),
                venue=build_canonical_venue_planes(),
                table_thickness_m=float(geometry.TABLE_THICKNESS),
            )
        for env_row in stage_inventory:
            for path in env_row[-1]:
                # Every concrete fresh RigidObject root remains a separate
                # PhysX actor/contact-report binding even though its composed
                # source content is inherited from env_0.
                prim = get_prim(path)
                if not bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                    raise ActionBallFullMdpBallSceneError(
                        f"fresh ActionBall prim is not a rigid actor: {path}"
                    )

        def exact_authority(value: object) -> object:
            if type(value) is not ExpectedRubberAuthorityView:
                raise ActionBallFullMdpBallSceneError(
                    "legacy expected-rubber authority entered the ActionEpoch lane"
                )
            return value

        def decode_path(value: object) -> str:
            # Isaac Sim 4.5 contact headers expose encoded integer paths.
            # PhysicsSchemaTools is the engine-owned inverse; strings/Sdf.Path
            # remain accepted for focused tests and version-compatible probes.
            if type(value) is int:
                from pxr import PhysicsSchemaTools

                text = str(PhysicsSchemaTools.intToSdfPath(value))
            else:
                text = str(value)
            if not text.startswith("/"):
                raise ActionBallFullMdpBallSceneError(
                    "PhysX callback did not expose a concrete absolute path"
                )
            return text

        owner = _install_isaac_physx_ball_fact_owner(
            port=self,
            stage=stage,
            concrete_ball_prim_paths=tuple(concrete_ball_paths),
            red_rubber_collider_paths=tuple(red_paths),
            black_rubber_collider_paths=tuple(black_paths),
            venue=build_canonical_venue_planes(),
            expected_authority_validator=exact_authority,
            path_decoder=decode_path,
            callback_order=CALLBACK_ORDER_HEARTBEAT_BEFORE_CONTACT,
            known_non_rubber_collider_bindings=tuple(non_rubber_bindings),
            wrist_actor_paths=tuple(wrist_actor_paths),
        )
        if type(owner) is not IsaacPhysxBallFactOwner or not owner.subscriptions_bound:
            raise ActionBallFullMdpBallSceneError(
                "live PhysX fact owner returned without exact subscriptions"
            )
        # The code-owned N=2 lane stays diagnostic/no-save, but real callback
        # facts are permitted.  This flag never authorizes checkpoint/export.
        owner._diagnostic_unauthorized = True
        self._physx_fact_owner = owner

    def shutdown_action_epoch_live_physx_fact_owner(self) -> None:
        """Idempotently invalidate and drain the live callback owner."""

        owner = self._physx_fact_owner
        if owner is None:
            return
        # Clear the public port reference before unsubscribe begins.  A
        # construction-failure callback cannot reacquire this closing owner.
        self._physx_fact_owner = None
        owner.shutdown()

    def bind_action_epoch_scene_writer(
        self, physical_owner: object, epoch_owner: object
    ) -> None:
        """Cold-bind the sole Physical writer and its exact ActionEpoch.

        Construction identity is intentionally the only admission mechanism.
        A source hash, caller boolean, or reveal receipt cannot grant write
        authority.  Rebinding after any scene write has been prepared is
        rejected even when the caller supplies the same shaped objects.
        """

        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_physical_flight_device as physical,
            )
        except ImportError:  # pragma: no cover - direct focused module loading.
            import action_ball_full_mdp_epoch as epoch
            import action_ball_physical_flight_device as physical

        if (
            self._action_epoch_physical_owner is physical_owner
            and self._action_epoch_owner is epoch_owner
        ):
            return
        projector = getattr(
            physical_owner, "action_epoch_scene_write_projection", None
        )
        validator = getattr(
            physical_owner,
            "require_owned_action_epoch_scene_write_projection",
            None,
        )
        physical_type = physical.ActionBallPhysicalFlightDeviceOwner
        epoch_type = epoch.ActionEpochOwner
        if (
            self._action_epoch_physical_owner is not None
            or self._action_epoch_owner is not None
            or self._active_write_handles
            or self._next_write_nonce != 1
            or type(physical_owner) is not physical_type
            or type(epoch_owner) is not epoch_type
            or getattr(physical_owner, "scene_port", None) is not self
            or getattr(physical_owner, "_action_epoch_owner", None)
            is not epoch_owner
            or getattr(physical_owner, "num_envs", None) != self.num_envs
            or getattr(physical_owner, "flight_capacity", None)
            != self.flight_capacity
            or getattr(epoch_owner, "num_envs", None) != self.num_envs
            or getattr(epoch_owner, "device", None) != self.device
            or getattr(physical_owner, "device", None) != self.device
            or not callable(projector)
            or getattr(projector, "__self__", None) is not physical_owner
            or getattr(projector, "__func__", None)
            is not physical_type.action_epoch_scene_write_projection
            or not callable(validator)
            or getattr(validator, "__self__", None) is not physical_owner
            or getattr(validator, "__func__", None)
            is not physical_type.require_owned_action_epoch_scene_write_projection
        ):
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch scene writer identity or direct producer API differs"
            )
        self._action_epoch_physical_owner = physical_owner
        self._action_epoch_owner = epoch_owner

    def preflight_action_epoch_write(self) -> PrevalidatedIsaacSceneWrite:
        """Prepare exactly one Physical-owned launch without caller payloads."""

        import torch

        physical_owner = self._action_epoch_physical_owner
        epoch_owner = self._action_epoch_owner
        if physical_owner is None or epoch_owner is None:
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch scene writer is not construction-bound"
            )
        if getattr(physical_owner, "_action_epoch_owner", None) is not epoch_owner:
            raise ActionBallFullMdpBallSceneError(
                "Physical no longer retains the bound ActionEpoch"
            )
        if self._active_write_handles:
            raise ActionBallFullMdpBallSceneError(
                "one ActionEpoch scene write already awaits apply or abort"
            )
        try:
            projection = physical_owner.action_epoch_scene_write_projection()
            owned = physical_owner.require_owned_action_epoch_scene_write_projection(
                projection
            )
        except BaseException as exc:
            raise ActionBallFullMdpBallSceneError(
                "Physical has no exact active ActionEpoch scene launch"
            ) from exc
        state = getattr(owned, "state_env_f32", None)
        selected = getattr(owned, "selected_mask", None)
        write_kind = getattr(owned, "kind", None)
        if (
            owned is not projection
            or write_kind not in ("launch", "retire")
            or getattr(owned, "physical_owner", None) is not physical_owner
            or not isinstance(state, torch.Tensor)
            or tuple(state.shape)
            != (self.num_envs, self.flight_capacity, 13)
            or state.dtype != torch.float32
            or state.device != self.device
            or not state.is_contiguous()
            or not isinstance(selected, torch.Tensor)
            or tuple(selected.shape)
            != (self.num_envs, self.flight_capacity)
            or selected.dtype != torch.bool
            or selected.device != self.device
            or not selected.is_contiguous()
        ):
            raise ActionBallFullMdpBallSceneError(
                "Physical ActionEpoch launch projection ABI differs"
            )
        invalid = (
            ((write_kind == "launch") & selected.to(torch.int64).sum(dim=1).gt(1))
            | (selected & ~torch.isfinite(state).all(dim=-1)).any(dim=1)
        )
        if self.device.type == "cpu":
            if bool(invalid.any()):
                raise ActionBallFullMdpBallSceneError(
                    "Physical ActionEpoch launch selects multiple slots or a "
                    "selected write is nonfinite"
                )
        # The exact Physical owner already validates and masks the selected
        # rows before constructing this private projection.  Reasserting its
        # own clone on CUDA is a same-writer echo: it cannot make the write
        # safer and used to destroy the whole CUDA context without attribution.

        root_state_world_by_slot: list[object] = []
        selected_mask_by_slot: list[object] = []
        scratch_root_state_world_by_slot: list[object] = []
        for slot in range(self.flight_capacity):
            candidate = state[:, slot].detach().clone()
            candidate[:, :3].add_(self.env_origins)
            root_state_world_by_slot.append(candidate)
            selected_mask_by_slot.append(selected[:, slot].detach().clone())
            scratch_root_state_world_by_slot.append(torch.empty_like(candidate))
        nonce = self._next_write_nonce
        self._next_write_nonce += 1
        handle = PrevalidatedIsaacSceneWrite(
            root_state_world_by_slot=tuple(root_state_world_by_slot),
            selected_mask_by_slot=tuple(selected_mask_by_slot),
            scratch_root_state_world_by_slot=tuple(
                scratch_root_state_world_by_slot
            ),
            expected_num_envs=self.num_envs,
            expected_capacity=self.flight_capacity,
            _port_identity=self._identity,
            _write_nonce=nonce,
            _token=_ISAAC_SCENE_WRITE_TOKEN,
        )
        self._active_write_handles[nonce] = handle
        return handle

    def bind_action_epoch_physics_fact_source(
        self,
        *,
        physical_owner: object,
        epoch_owner: object,
        racket_owner: object,
    ) -> None:
        """Cold-bind the exact live selected-rubber producer identities.

        This method does not install callbacks by accepting paths or a caller
        validator.  It retains only the three causal owners; the concrete
        stage installer below resolves and validates named collider prims.
        """

        import torch

        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_epoch as epoch,
            )
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_physical_flight_device as physical,
            )
            from whole_body_tracking.tasks.tracking.mdp import hope_commands
        except ImportError:  # pragma: no cover - focused direct imports.
            import action_ball_full_mdp_epoch as epoch
            import action_ball_physical_flight_device as physical
            import hope_commands

        racket_type = hope_commands.RacketTargetCommand
        face_method = getattr(
            racket_owner,
            "action_ball_full_mdp_action_epoch_selected_rubber_view",
            None,
        )
        allocation = getattr(
            physical_owner, "action_epoch_physics_fact_allocation", None
        )
        allocation_validator = getattr(
            physical_owner,
            "require_owned_action_epoch_physics_fact_allocation",
            None,
        )
        racket_device_raw = getattr(racket_owner, "device", None)
        if type(racket_device_raw) not in (str, torch.device):
            racket_device = None
        else:
            try:
                racket_device = torch.device(racket_device_raw)
            except (RuntimeError, TypeError):
                racket_device = None
        if (
            self._action_epoch_physical_owner is not physical_owner
            or self._action_epoch_owner is not epoch_owner
            or self._action_epoch_racket_owner is not None
            or type(physical_owner)
            is not physical.ActionBallPhysicalFlightDeviceOwner
            or type(epoch_owner) is not epoch.ActionEpochOwner
            or type(racket_owner) is not racket_type
            or getattr(racket_owner, "num_envs", None) != self.num_envs
            or racket_device != self.device
            or not callable(face_method)
            or getattr(face_method, "__self__", None) is not racket_owner
            or getattr(face_method, "__func__", None)
            is not racket_type.action_ball_full_mdp_action_epoch_selected_rubber_view
            or not callable(allocation)
            or getattr(allocation, "__self__", None) is not physical_owner
            or getattr(allocation, "__func__", None)
            is not physical.ActionBallPhysicalFlightDeviceOwner.action_epoch_physics_fact_allocation
            or not callable(allocation_validator)
            or getattr(allocation_validator, "__self__", None)
            is not physical_owner
            or getattr(allocation_validator, "__func__", None)
            is not physical.ActionBallPhysicalFlightDeviceOwner.require_owned_action_epoch_physics_fact_allocation
        ):
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch PhysX fact-source owner identity/API differs"
            )
        self._action_epoch_racket_owner = racket_owner

    def arm_action_epoch_physics_fact_source(self) -> None:
        """Arm one callback epoch from Physical allocation + Racket face."""

        import torch

        physical_owner = self._action_epoch_physical_owner
        epoch_owner = self._action_epoch_owner
        racket_owner = self._action_epoch_racket_owner
        fact_owner = self._physx_fact_owner
        if (
            physical_owner is None
            or epoch_owner is None
            or racket_owner is None
            or fact_owner is None
        ):
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch live PhysX fact source is not construction-bound"
            )
        try:
            physical = physical_owner.action_epoch_physics_fact_allocation()
            physical = physical_owner.require_owned_action_epoch_physics_fact_allocation(
                physical
            )
            racket = (
                racket_owner.
                action_ball_full_mdp_action_epoch_selected_rubber_view()
            )
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_physical_flight_device as physical_module,
            )
            from whole_body_tracking.tasks.tracking.mdp import hope_commands
        except ImportError:  # pragma: no cover
            import action_ball_physical_flight_device as physical_module
            import hope_commands
        except BaseException as exc:
            raise ActionBallFullMdpBallSceneError(
                "exact Physical/Racket face projection is unavailable"
            ) from exc
        shape = (self.num_envs, self.flight_capacity)
        if (
            type(physical)
            is not physical_module.ActionEpochPhysicsFactAllocationProjection
            or type(racket)
            is not hope_commands.ActionBallFullMdpActionEpochSelectedRubberView
            or physical.physical_owner is not physical_owner
            or physical.epoch_owner is not epoch_owner
            or racket.racket_owner is not racket_owner
            or racket.physical_owner is not physical_owner
            or racket.epoch_owner is not epoch_owner
            or type(physical.active_mask) is not torch.Tensor
            or type(racket.active_mask) is not torch.Tensor
            or tuple(physical.active_mask.shape) != shape
            or tuple(racket.active_mask.shape) != shape
            or physical.active_mask.dtype != torch.bool
            or racket.active_mask.dtype != torch.bool
            or physical.active_mask.device != self.device
            or racket.active_mask.device != self.device
            or type(racket.expected_rubber) is not torch.Tensor
            or tuple(racket.expected_rubber.shape) != shape
            or racket.expected_rubber.dtype != torch.int8
            or racket.expected_rubber.device != self.device
            or type(physical.ball_generation) is not torch.Tensor
            or tuple(physical.ball_generation.shape) != shape
            or physical.ball_generation.dtype != torch.int64
            or physical.ball_generation.device != self.device
            or type(physical.full_key_sha256) is not torch.Tensor
            or tuple(physical.full_key_sha256.shape) != shape + (32,)
            or physical.full_key_sha256.dtype != torch.uint8
            or physical.full_key_sha256.device != self.device
        ):
            raise ActionBallFullMdpBallSceneError(
                "Physical allocation and Racket selected face do not join"
            )
        expected = torch.where(
            physical.active_mask,
            racket.expected_rubber,
            torch.full_like(racket.expected_rubber, RUBBER_INACTIVE),
        )
        # Racket re-reads Physical's allocation before deriving the face.  A
        # second host equality check would only validate a same-writer echo
        # and synchronize CUDA.  Keep all ABI/semantic work device-local.
        invalid = physical.active_mask & (
            (expected != RUBBER_RED) & (expected != RUBBER_BLACK)
        )
        fact_owner._bind_action_epoch_expected_rubber(
            active_mask=physical.active_mask,
            expected_rubber=expected,
            ball_generation=physical.ball_generation,
            full_key_sha256=physical.full_key_sha256,
            _installer_token=_PHYSX_FACT_OWNER_TOKEN,
        )

    def _action_epoch_live_fact_owner(self):
        fact_owner = self._physx_fact_owner
        if fact_owner is None:
            raise ActionBallFullMdpBallSceneError(
                "ActionEpoch callback operation lacks the live PhysX fact owner"
            )
        return fact_owner

    def action_epoch_physics_fact_activity_mask(self):
        return self._action_epoch_live_fact_owner()._action_epoch_activity_mask()

    def begin_action_epoch_idle_physics_fact_source(self) -> None:
        """Open one empty callback epoch without constructing a K-grid."""

        self._action_epoch_live_fact_owner()._begin_action_epoch_idle_binding()

    def complete_action_epoch_idle_physics_fact_source(
        self, exact_stamp: object
    ) -> None:
        """Seal the exact empty callback epoch without a scene state read."""

        self._action_epoch_live_fact_owner()._complete_action_epoch_idle_binding(
            exact_stamp=exact_stamp
        )

    def install_physx_fact_owner(self, value: object) -> None:
        raise ActionBallFullMdpBallSceneError(
            "production PhysX fact binding is HOLD until the exact external "
            "Racket authority and live callback order are owned"
        )

    @property
    def scene_port_capability(self) -> IsaacPhysicalFlightScenePortCapability:
        return self._scene_port_capability

    def require_owned_scene_port_capability(
        self, value: object
    ) -> IsaacPhysicalFlightScenePortCapability:
        if (
            type(value) is not IsaacPhysicalFlightScenePortCapability
            or value is not self._scene_port_capability
            or value._port_identity is not self._identity
            or value._token is not _ISAAC_SCENE_PORT_CAPABILITY_TOKEN
        ):
            raise ActionBallFullMdpBallSceneError(
                "Isaac scene-port capability is stale or foreign"
            )
        return value

    def read_state_env(self):
        import torch

        state = torch.stack(
            tuple(asset.data.root_state_w for asset in self.assets), dim=1
        ).to(dtype=torch.float32)
        # ``stack`` already owns fresh storage; translating that after-image
        # in place avoids a second full-grid clone without aliasing Isaac.
        state[..., :3].sub_(self.env_origins.unsqueeze(1))
        return state

    def capture_post_physics_facts(self, request: object):
        """Capture the provable part of the exact Physical postphysics ABI.

        The caller-owned request already binds the Physical slot/key/generation/
        ordinal image.  This concrete port joins its current post-scene-update
        K-body root state to that retained image without a device-to-host
        transfer; key/generation/ordinal remain Physical-owner facts, not an
        independent scene proof.  The current scene has no exact contact/net/table/overflow fact
        producers, so every observed live row is explicitly marked as a producer
        contract fault.  The pinned robot merges both rubbers and the handle
        into one wrist body, so a ball-wrist report is not selected-rubber
        authority.  Event ``False``
        values are therefore failure sentinels, never evidence of an ordinary
        miss.
        """

        import torch

        physical_module = sys.modules.get(type(request).__module__)
        request_type = getattr(
            physical_module, "PhysicalPostPhysicsCaptureRequest", None
        )
        facts_type = getattr(physical_module, "IsaacPostPhysicsFacts", None)
        stamp_type = getattr(physical_module, "PhysicsStampGrid", None)
        if (
            request_type is None
            or facts_type is None
            or stamp_type is None
            or type(request) is not request_type
        ):
            raise ActionBallFullMdpBallSceneError(
                "postphysics request is not the exact Physical owner ABI"
            )

        shape = (self.num_envs, self.flight_capacity)

        def exact_tensor(name: str, *, suffix: tuple[int, ...], dtype):
            value = getattr(request, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape + suffix
                or value.dtype != dtype
                or value.device != self.device
            ):
                raise ActionBallFullMdpBallSceneError(
                    f"postphysics request {name} tensor ABI differs"
                )
            return value

        exact = getattr(request, "exact_stamp", None)
        if (
            type(exact) is not tuple
            or len(exact) != 5
            or any(type(value) is not int for value in exact)
        ):
            raise ActionBallFullMdpBallSceneError(
                "postphysics request exact stamp ABI differs"
            )
        control, substep, decimation, sim_step, phase = exact
        if (
            control < 1
            or substep < 0
            or decimation < 1
            or substep >= decimation
            or sim_step < 1
            or phase != 1
        ):
            raise ActionBallFullMdpBallSceneError(
                "postphysics request exact stamp range differs"
            )

        observe = exact_tensor("observe_mask", suffix=(), dtype=torch.bool)
        flight_slot = exact_tensor("flight_slot", suffix=(), dtype=torch.int64)
        full_key = exact_tensor(
            "full_key_sha256", suffix=(32,), dtype=torch.uint8
        )
        generation = exact_tensor(
            "ball_generation", suffix=(), dtype=torch.int64
        )
        ordinal = exact_tensor(
            "observation_ordinal", suffix=(), dtype=torch.int64
        )
        previous_center = exact_tensor(
            "previous_ball_center_m", suffix=(3,), dtype=torch.float32
        )
        request_state_raw = getattr(request, "current_state_env_f32", None)
        physical_owner = self._action_epoch_physical_owner
        owner_issued_state_read = (
            request_state_raw is None
            and physical_owner is not None
            and request._owner_identity
            is getattr(physical_owner, "_owner_identity", None)
            and request._token
            is getattr(physical_module, "_POSTPHYSICS_CAPTURE_REQUEST_TOKEN", None)
        )
        if request_state_raw is None:
            if not owner_issued_state_read:
                raise ActionBallFullMdpBallSceneError(
                    "postphysics request current_state_env_f32 tensor ABI differs"
                )
            request_state = None
        else:
            request_state = exact_tensor(
                "current_state_env_f32", suffix=(13,), dtype=torch.float32
            )

        live_state = self.read_state_env()
        expected_slot = torch.arange(
            self.flight_capacity, dtype=torch.int64, device=self.device
        ).unsqueeze(0).expand(shape)
        # These are request self-consistency checks, not independent scene
        # authority for key/generation/ordinal identity.  The exact Physical
        # owner retains and validates that image.  Until the missing causal
        # event producers are bound, ``observe`` below faults every live row.
        request_self_consistency_fault = (
            (flight_slot != expected_slot)
            | (observe & torch.eq(full_key, 0).all(dim=-1))
            | (observe & (generation < 0))
            | (observe & (ordinal < 0))
            | (
                torch.zeros(shape, dtype=torch.bool, device=self.device)
                if request_state is None
                else ~(
                    torch.eq(request_state, live_state)
                    | (torch.isnan(request_state) & torch.isnan(live_state))
                ).all(dim=-1)
            )
        )
        nonfinite = observe & (
            ~torch.isfinite(live_state).all(dim=-1)
            | ~torch.isfinite(previous_center).all(dim=-1)
        )
        if (
            self._physx_fact_owner is None
            and self.device.type == "cpu"
            and bool(request_self_consistency_fault.any())
        ):
            raise ActionBallFullMdpBallSceneError(
                "postphysics request slot/key/generation/ordinal/state self-consistency differs"
            )

        if self._physx_fact_owner is not None:
            facts = self._physx_fact_owner.capture(
                request=request,
                live_state=live_state,
                facts_type=facts_type,
                stamp_type=stamp_type,
            )
            # Physical owns request identity; this comparison only detects a
            # corrupted scene call boundary.  Fold it into the existing typed
            # device fault instead of synchronizing CPU or raising before the
            # causal producer can publish its packet.
            facts.producer_contract_fault.logical_or_(
                request_self_consistency_fault
            )
            facts.nonfinite_observation.logical_or_(nonfinite)
            return facts

        def stamp(active, *, event_phase: int):
            return stamp_type(
                control_step=torch.where(
                    active,
                    torch.full(
                        shape, control, dtype=torch.int64, device=self.device
                    ),
                    torch.full(
                        shape, -1, dtype=torch.int64, device=self.device
                    ),
                ),
                physics_substep=torch.where(
                    active,
                    torch.full(
                        shape, substep, dtype=torch.int32, device=self.device
                    ),
                    torch.full(
                        shape, -1, dtype=torch.int32, device=self.device
                    ),
                ),
                event_phase=torch.where(
                    active,
                    torch.full(
                        shape, event_phase, dtype=torch.int8, device=self.device
                    ),
                    torch.full(
                        shape, -1, dtype=torch.int8, device=self.device
                    ),
                ),
            )

        no_event = torch.zeros(shape, dtype=torch.bool, device=self.device)
        # Failure sentinels only.  POSTPHYSICS_FACT_PRODUCERS_BOUND remains
        # false until these fields have causal engine producers.
        no_event_stamp = stamp(no_event, event_phase=2)
        return facts_type(
            observation_stamp=stamp(observe, event_phase=2),
            current_state_env_f32=live_state,
            selected_contact_event=no_event.clone(),
            selected_contact_ball_center_m=torch.zeros(
                shape + (3,), dtype=torch.float32, device=self.device
            ),
            selected_contact_outgoing_segment_anchor_m=torch.zeros(
                shape + (3,), dtype=torch.float32, device=self.device
            ),
            selected_contact_stamp=no_event_stamp,
            net_crossing_event=no_event.clone(),
            net_clear_at_crossing=no_event.clone(),
            net_crossing_stamp=stamp(no_event, event_phase=1),
            crossing_report_delivered=no_event.clone(),
            first_descending_crossing_event=no_event.clone(),
            first_descending_crossing_xy_m=torch.zeros(
                shape + (2,), dtype=torch.float32, device=self.device
            ),
            first_descending_crossing_stamp=stamp(no_event, event_phase=2),
            nonfinite_observation=nonfinite,
            producer_contract_fault=(observe | request_self_consistency_fault),
            engine_overflow=no_event.clone(),
            _owner_identity=request,
            _capture_token=request._token,
        )

    def preflight_write(
        self,
        state_env,
        selected_mask,
        *,
        reveal_boundary_receipt: object,
    ):
        import torch

        expected_state = (self.num_envs, self.flight_capacity, 13)
        expected_mask = (self.num_envs, self.flight_capacity)
        if (
            not isinstance(state_env, torch.Tensor)
            or state_env.shape != expected_state
            or state_env.dtype != torch.float32
            or state_env.device != self.device
            or not isinstance(selected_mask, torch.Tensor)
            or selected_mask.shape != expected_mask
            or selected_mask.dtype != torch.bool
            or selected_mask.device != self.device
        ):
            raise ActionBallFullMdpBallSceneError("scene preflight tensor ABI differs")
        raise ActionBallFullMdpBallSceneError(
            "caller value cannot authorize a scene write; the exact owned "
            "reveal-boundary receipt is not wired"
        )

    def apply_prevalidated_write(
        self, handle: object
    ) -> IsaacPhysicalFlightSceneApplyReceipt:
        import torch

        if (
            type(handle) is not PrevalidatedIsaacSceneWrite
            or handle._port_identity is not self._identity
            or handle._token is not _ISAAC_SCENE_WRITE_TOKEN
            or self._active_write_handles.get(handle._write_nonce) is not handle
        ):
            raise ActionBallFullMdpBallSceneError("scene write handle is not prevalidated")
        if (
            handle.expected_num_envs != self.num_envs
            or handle.expected_capacity != self.flight_capacity
            or len(handle.root_state_world_by_slot) != self.flight_capacity
            or len(handle.selected_mask_by_slot) != self.flight_capacity
            or len(handle.scratch_root_state_world_by_slot)
            != self.flight_capacity
        ):
            raise ActionBallFullMdpBallSceneError("scene write handle binding differs")
        # No validation or allocation is allowed after this point.  The owner
        # poisons itself if Isaac raises; it never claims rollback.
        for asset, candidate, selected_mask, root_state in zip(
            self.assets,
            handle.root_state_world_by_slot,
            handle.selected_mask_by_slot,
            handle.scratch_root_state_world_by_slot,
        ):
            torch.where(
                selected_mask.unsqueeze(1),
                candidate,
                asset.data.root_state_w,
                out=root_state,
            )
            asset.write_root_pose_to_sim(
                root_state[:, :7], env_ids=self._all_env_ids
            )
            asset.write_root_velocity_to_sim(
                root_state[:, 7:], env_ids=self._all_env_ids
            )
        readback_verified = False
        if self.device.type == "cpu":
            readback_verified = all(
                torch.equal(
                    asset.data.root_state_w,
                    root_state,
                )
                for asset, root_state in zip(
                    self.assets,
                    handle.scratch_root_state_world_by_slot,
                )
            )
            if not readback_verified:
                raise ActionBallFullMdpBallSceneError(
                    "Isaac scene write returned without exact selected-row readback"
                )
        self._active_write_handles.pop(handle._write_nonce, None)
        return IsaacPhysicalFlightSceneApplyReceipt(
            write_nonce=handle._write_nonce,
            full_grid_write=True,
            readback_verified=readback_verified,
            _port_identity=self._identity,
            _handle_identity=handle,
            _token=_ISAAC_SCENE_WRITE_TOKEN,
        )

    def require_owned_apply_receipt(
        self,
        handle: object,
        receipt: object,
    ) -> IsaacPhysicalFlightSceneApplyReceipt:
        if (
            type(handle) is not PrevalidatedIsaacSceneWrite
            or type(receipt) is not IsaacPhysicalFlightSceneApplyReceipt
            or receipt._port_identity is not self._identity
            or receipt._handle_identity is not handle
            or receipt._token is not _ISAAC_SCENE_WRITE_TOKEN
            or receipt.write_nonce != handle._write_nonce
            or receipt.full_grid_write is not True
            or (self.device.type == "cpu" and not receipt.readback_verified)
            or self._active_write_handles.get(handle._write_nonce) is not None
        ):
            raise ActionBallFullMdpBallSceneError(
                "Isaac scene apply receipt is stale or foreign"
            )
        return receipt

    def abort_prevalidated_write(
        self, handle: object
    ) -> IsaacPhysicalFlightSceneAbortReceipt:
        if (
            type(handle) is not PrevalidatedIsaacSceneWrite
            or handle._port_identity is not self._identity
            or handle._token is not _ISAAC_SCENE_WRITE_TOKEN
            or self._active_write_handles.get(handle._write_nonce) is not handle
        ):
            raise ActionBallFullMdpBallSceneError(
                "Isaac scene abort handle is stale or foreign"
            )
        self._active_write_handles.pop(handle._write_nonce, None)
        return IsaacPhysicalFlightSceneAbortReceipt(
            write_nonce=handle._write_nonce,
            _port_identity=self._identity,
            _handle_identity=handle,
            _token=_ISAAC_SCENE_WRITE_TOKEN,
        )

    def require_owned_abort_receipt(
        self,
        handle: object,
        receipt: object,
    ) -> IsaacPhysicalFlightSceneAbortReceipt:
        if (
            type(handle) is not PrevalidatedIsaacSceneWrite
            or type(receipt) is not IsaacPhysicalFlightSceneAbortReceipt
            or receipt._port_identity is not self._identity
            or receipt._handle_identity is not handle
            or receipt._token is not _ISAAC_SCENE_WRITE_TOKEN
            or receipt.write_nonce != handle._write_nonce
            or self._active_write_handles.get(handle._write_nonce) is not None
        ):
            raise ActionBallFullMdpBallSceneError(
                "Isaac scene abort receipt is stale or foreign"
            )
        return receipt


def _class_ast_sha256(value: type[object]) -> str:
    tree = ast.parse(inspect.getsource(value))
    return hashlib.sha256(
        ast.dump(tree, include_attributes=False).encode("utf-8")
    ).hexdigest()


ISAAC_SCENE_PORT_AST_SHA256 = _class_ast_sha256(
    IsaacLabPhysicalFlightScenePort
)


__all__ = [
    "ActionBallFullMdpBallSceneError",
    "ActionBallFullMdpBallSceneSpec",
    "ActionBallFullMdpDiagnosticBallSceneSpec",
    "CALLBACK_ORDER_CONTACT_BEFORE_HEARTBEAT",
    "CALLBACK_ORDER_HEARTBEAT_BEFORE_CONTACT",
    "CanonicalVenuePlanes",
    "CONTRACT_SOURCE_SHA256",
    "DIAGNOSTIC_SCENE_SPEC_KIND",
    "ExpectedRubberAuthorityView",
    "INTEGRATION_RESIDUALS",
    "ISAAC_SCENE_PORT_AST_SHA256",
    "IsaacPhysicalFlightSceneAbortReceipt",
    "IsaacPhysicalFlightSceneApplyReceipt",
    "IsaacPhysicalFlightScenePortCapability",
    "IsaacPhysxBallFactOwner",
    "IsaacLabPhysicalFlightScenePort",
    "LAUNCH_AUTHORIZED",
    "LEGACY_SCENE_ENTITY_NAME",
    "PARK_POSITION_ENV_M",
    "POD_FULL_SCENE_VALIDATED",
    "POSTPHYSICS_CAPTURE_HOLD_REASONS",
    "POSTPHYSICS_FACT_PRODUCERS_BOUND",
    "PhysxCallbackContact",
    "PhysxFactOwnerCheckpoint",
    "PhysicalFlightScenePort",
    "PrevalidatedIsaacSceneWrite",
    "RUNTIME_INTEGRATED",
    "RUBBER_BLACK",
    "RUBBER_INACTIVE",
    "RUBBER_RED",
    "SCENE_ENTITY_PREFIX",
    "attach_action_ball_full_mdp_ball_scene",
    "build_action_ball_full_mdp_ball_scene_spec",
    "build_canonical_venue_planes",
    "verify_frozen_physical_flight_contract_source",
]
