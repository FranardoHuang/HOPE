"""Canonical HOPE table-tennis geometry, expressed in the HOPE world frame.

The HOPE world frame (ROS 2 REP-103, identical to the planner / mocap reference docs):

* **Origin** — the near-side **left** corner of the table *surface*, from Player One's (P1) perspective.
* **X** — forward, toward Player Two (P2), along the table length:  ``x in [0, 2.74] m``.
* **Y** — left, from P1's perspective, along the table width:        ``y in [-1.525, 0] m``.
* **Z** — up; **z = 0 is the table surface** (the floor is therefore at ``z = -0.76 m``).

Key landmarks (see ``mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md`` §2.1
and ``hope_ws/.../hope_world_frame.yaml``)::

    Origin (P1 near-side left corner)   ( 0.0  ,  0.0   ,  0.0 )
    Net center line                     ( 1.37 , -0.7625,  0.0 )
    P1 half center                      ( 0.685, -0.7625,  0.0 )
    P2 half center                      ( 2.055, -0.7625,  0.0 )
    Floor directly below origin         ( 0.0  ,  0.0   , -0.76)

This module is the **single source of truth** for the geometry used to build the Isaac Lab scene.
It is pure Python (no Isaac / torch imports) so it can be imported anywhere and unit-tested without a
simulator. The numbers mirror ``hope_ws/src/hope_planner/hope_planner/constants.py`` (ITTF regulation
table) so the simulator and the model-based planner share one consistent world.

The simulation world frame is set **equal to** this HOPE frame: each environment's local origin sits
at the table-surface height (HOPE z = 0), so an asset's environment-local position *is* its HOPE-frame
position. With multiple environments, every environment is an independent court with its own HOPE
origin anchored at that environment's origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

##
# ITTF regulation table (meters). Mirrors hope_planner.constants.TableParams.
##
TABLE_LENGTH: float = 2.74          # along +X
TABLE_WIDTH: float = 1.525          # along -Y
TABLE_HEIGHT: float = 0.76          # table surface above the floor
TABLE_THICKNESS: float = 0.05       # visual/collision slab thickness of the table top

NET_X: float = 1.37                 # net plane position along X (== TABLE_LENGTH / 2)
NET_HEIGHT: float = 0.1525          # net height above the table surface
NET_OVERHANG: float = 0.15          # net extends this far past each table edge in Y
NET_THICKNESS: float = 0.01         # net slab thickness along X

LINE_WIDTH: float = 0.02            # painted boundary / center line width
LINE_THICKNESS: float = 0.001       # painted line slab thickness (sits just above the surface)

##
# Ball (40 mm). These MUST match ``configs/ball_physics_venue.yaml`` (the single source of truth shared
# with the MuJoCo C++ sim and the Record reference); a regression test asserts they stay in sync.
##
BALL_RADIUS: float = 0.02           # 40 mm diameter (20 mm radius)
BALL_MASS: float = 0.0034           # 3.4 g — the retro-reflective-coated MATCH ball (venue fit 2026-07-03;
                                    # competition uses this same coated ball, NOT the clean 2.7 g ITTF ball)
BALL_INERTIA_COEFF: float = 2.0 / 3.0  # hollow sphere: I = c * m * R^2; used by the spin-coupled contact model

##
# Coordinate landmarks in the HOPE frame.
##
ORIGIN: tuple[float, float, float] = (0.0, 0.0, 0.0)
TABLE_CENTER: tuple[float, float, float] = (TABLE_LENGTH / 2.0, -TABLE_WIDTH / 2.0, 0.0)
NET_CENTER: tuple[float, float, float] = (NET_X, -TABLE_WIDTH / 2.0, 0.0)
P1_HALF_CENTER: tuple[float, float, float] = (TABLE_LENGTH / 4.0, -TABLE_WIDTH / 2.0, 0.0)
P2_HALF_CENTER: tuple[float, float, float] = (3.0 * TABLE_LENGTH / 4.0, -TABLE_WIDTH / 2.0, 0.0)
FLOOR_Z: float = -TABLE_HEIGHT      # the floor surface, in HOPE z

##
# Robot nominal standing pose, P1 side (HOPE frame).
# From HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md: "Robot nominal standing position
# ~ (-0.5, -0.7625, floor)". X < 0 puts the robot behind the near table end; Y centers it on the
# table width; the pelvis height is added on top of FLOOR_Z by the robot-specific config.
##
P1_STAND_X: float = -0.5
P1_STAND_Y: float = -TABLE_WIDTH / 2.0   # -0.7625, centered on table width


def table_top_center() -> tuple[float, float, float]:
    """Center of the table-top collision/visual slab (its top face is at HOPE z = 0)."""
    return (TABLE_CENTER[0], TABLE_CENTER[1], -TABLE_THICKNESS / 2.0)


def net_center() -> tuple[float, float, float]:
    """Center of the net slab (spans z in [0, NET_HEIGHT])."""
    return (NET_X, -TABLE_WIDTH / 2.0, NET_HEIGHT / 2.0)


def net_size() -> tuple[float, float, float]:
    """Full extents (x, y, z) of the net slab, including the lateral overhang past the table edges."""
    return (NET_THICKNESS, TABLE_WIDTH + 2.0 * NET_OVERHANG, NET_HEIGHT)


def table_top_size() -> tuple[float, float, float]:
    """Full extents (x, y, z) of the table-top slab."""
    return (TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS)


@dataclass
class ServeConfig:
    """Parameters for the scripted ball serve used to reset / visualize ball flight.

    The ball is spawned over one half of the table and launched toward the other side with a slight
    downward component, so a reset shows a full flight -> table bounce -> rising arc. All positions are
    in the HOPE frame; velocities are in m/s in the HOPE frame. Ranges are sampled uniformly per reset.
    """

    # Spawn box over the P2 half (so the ball travels toward the P1-side robot). Spawn height is well
    # above the net top (0.1525 m) so the served arc clears the net at x = 1.37 for ~all resets.
    pos_x_range: tuple[float, float] = (2.0, 2.4)
    pos_y_range: tuple[float, float] = (-1.1, -0.4)
    pos_z_range: tuple[float, float] = (0.55, 0.80)
    # Launch velocity toward P1 (-X), roughly flat / slightly up so it clears the net then arcs down
    # under gravity onto the P1 half, with small lateral spread.
    vel_x_range: tuple[float, float] = (-5.0, -3.5)
    vel_y_range: tuple[float, float] = (-0.4, 0.4)
    vel_z_range: tuple[float, float] = (-0.2, 0.5)
    # Initial spin (rad/s) sampled uniformly per HOPE axis. Tens of rad/s is the physical regime the
    # mocap fit implies (~25-75 rad/s); this exercises the Magnus flight term and the spin-coupled
    # bounce. Set both ends to 0.0 to serve a spin-free ball.
    spin_range: tuple[float, float] = (-100.0, 100.0)

    @classmethod
    def reachable_returner(cls) -> "ServeConfig":
        """Narrow, zero-spin serve for fixed-base / arm-only **returner** training (and the first smoke run).

        Tuned against the fitted flight + spin-equation bounce model (``physics/flight.py`` +
        ``spin_contact.py``): every sample clears the net, bounces once on the **P1 half** (x ~ 0.6-0.8 m),
        and rises into a paddle-reachable band (z ~ 0.2-0.44 m above the table) near the table front edge
        (x ~ 0), centred on the robot's stance line (y = -0.7625). Spin is zero so the trajectory is
        predictable. This gives the per-hit ``pass_net``/``landing`` rewards a consistent ball to hit
        instead of the wide, ±100 rad/s default serve. Validated by sweep: 16/16 box corners + 40/40
        random interior samples clear the net and land a hittable window. Confirm reachability against the
        A3 racket workspace visually (``play_table_tennis.py --serve returner``) before a long run."""
        return cls(
            pos_x_range=(1.95, 2.05),
            pos_y_range=(-0.85, -0.70),
            pos_z_range=(0.64, 0.72),
            vel_x_range=(-3.5, -3.1),
            vel_y_range=(-0.05, 0.05),
            vel_z_range=(0.35, 0.55),
            spin_range=(0.0, 0.0),
        )


@dataclass
class BounceMaterials:
    """PhysX contact-material parameters for the ball and the static surfaces.

    IMPORTANT — the ball<->table and ball<->racket bounces are NOT resolved by PhysX restitution any
    more. They are owned by the code-driven, spin-coupled **spin-equation** contact model (see
    ``physics/spin_contact.py``, applied per substep by ``TableTennisEnv``), because PhysX's scalar
    Newton/Coulomb restitution cannot reproduce the fitted angle-dependent tangential impulse or the
    spin update ``dw = -(3/2R)(n x dv_t)``.

    To get out of PhysX's way, ``ball_restitution = 0`` — with the multiplicative combine mode (see
    ``table_tennis_env_cfg._surface_material``) the *effective* restitution of every ball<->surface
    contact is the **product**, so a zero ball restitution makes PhysX add **no** bounce on any surface:
    it only resolves penetration (anti-tunnelling, helped by CCD), while the code sets the true
    post-contact velocity + spin. The friction values are kept small/benign for the rare resting/rolling
    case (after a point is effectively over); they do not affect the code-driven bounce.
    """

    # Ball (dynamic) — restitution 0 hands ALL bounces to the code-driven spin-equation model.
    ball_restitution: float = 0.0
    ball_static_friction: float = 0.1
    ball_dynamic_friction: float = 0.1
    # Table top.
    table_restitution: float = 0.95   # product with ball (0) is 0; kept for clarity / future tuning
    table_static_friction: float = 0.4
    table_dynamic_friction: float = 0.4
    # Floor.
    floor_restitution: float = 0.4
    floor_static_friction: float = 0.8
    floor_dynamic_friction: float = 0.8
    # Net.
    net_restitution: float = 0.1
    net_static_friction: float = 0.5
    net_dynamic_friction: float = 0.5


@dataclass
class OutOfBoundsBox:
    """Axis-aligned region (HOPE frame) outside which the ball is considered dead / out of play.

    Generous margins around the table so the ball can fly well past either player before the episode
    is reset. Used by the termination / serve-reset logic.
    """

    x: tuple[float, float] = (-2.0, 4.74)
    y: tuple[float, float] = (-3.0, 1.5)
    # A ball resting on the floor has center z = FLOOR_Z + BALL_RADIUS = -0.74; trigger just above that
    # (-0.73) so a grounded/missed ball ends the episode promptly instead of idling until timeout.
    z: tuple[float, float] = (FLOOR_Z + BALL_RADIUS + 0.01, 3.0)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {"x": self.x, "y": self.y, "z": self.z}
