"""Per-env correlated ground patch, robot side only (task.plant.terrain_rough_height_range).

人话:让机器人学抬脚的"凹凸地垫"。每个 env 自带一块静态的随机凹凸地面,只铺在机器人这一侧
(球台近沿之前);整个球台足迹连同前向余量都强制平在 z=0。机器人出生点是平地,其余区域
由低频平滑场形成连续坡/波,而不是每10 cm顶点独立抽白噪声。谷底之下再垫一块全局兜底地板接住
"走出垫子/穿模"的极端情况。

WHY this shape (and not ``TerrainImporterCfg(terrain_type="generator")``):

* A generator terrain swaps ``scene.env_origins`` onto terrain-TILE origins, while every cloned
  static asset (table obstacle / shadow table slabs) stays on the GridCloner grid — robots would
  be teleported away from their own tables.  The 2026-07-22 generator wiring had exactly this
  fault, which is why no rough arm ever learned anything on it.
* With ``env_spacing=2.5`` m and a 2.74 m table, one env's table footprint overlaps its
  neighbour's robot zone, so a single shared ground mesh cannot be simultaneously "flat under my
  table" and "rough under their feet".  Per-env patches under ``{ENV_REGEX_NS}`` follow the
  shadow-table precedent: cross-env pairs are collision-filtered, each robot only ever touches
  its own pad.
* ``replicate_physics=True`` clones the pad by reference, so every env shares ONE cooked collision
  mesh (identical bump pattern across envs; the pattern itself is seeded per run — the global
  numpy RNG is already seeded from ``env_cfg.seed`` before ``InteractiveScene`` is built).

Zero-centred semantics of the authored band: ``terrain_rough_height_range=[lo, hi]`` keeps its
contract spelling, but correlated control values are symmetric about zero and the final mesh is
quantized to 5 mm levels inside ``[-(hi-lo)/2, +(hi-lo)/2]``. ``[0.0, 0.04]`` therefore means
"at most ±2 cm about the calibrated floor", not "0..4 cm on top of it".

Module layout keeps the host-test contract of this repo: everything above
``attach_rough_ground_patch`` is pure numpy (py3.8 host pytest imports it); Isaac Lab / pxr
imports only happen inside the attach/spawn functions (Kit 起来后才 import 得动).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

# --- fixed pad geometry (metres). One place; the contract identifies the plant by the noise band,
# these constants are part of the "robot_side_correlated_spawn_flat_v2" terrain type. ---
HORIZONTAL_SCALE_M = 0.1   # cell size; matches the isaaclab rough-terrain precedent
VERTICAL_SCALE_M = 0.005   # 5 mm height quantization (= noise step)
SLOPE_THRESHOLD = 0.75     # vertical-wall correction, isaaclab precedent
# Exact Pod1 CPU FK over the four ready-pose ankle collision meshes reached
# radius 0.470509 m from the env origin (MJCF 70c4fd65..., ready pose
# ab6b7e41...).  Keep every collider plus one 10 cm terrain cell on exact
# flat ground; the old 0.20 m core put both feet in the rough/blend shoulder.
SPAWN_FLAT_RADIUS_M = 0.60
SPAWN_BLEND_RADIUS_M = 0.80
TABLE_BLEND_WIDTH_M = 0.30
SMOOTHING_PASSES = 4
X_BACK_M = 3.0             # rough zone reaches this far behind the table near edge
X_FORWARD_MARGIN_M = 0.5   # flat zone reaches this far beyond the table far edge
Y_HALF_M = 3.0             # pad half-width
SAFETY_FLOOR_MARGIN_M = 0.05  # backstop plane sits this far below the deepest possible valley
MIN_BAND_M = 0.01          # (hi - lo) below this quantizes to a dead-flat pad -> refused
# Above this band, convert_height_field_to_mesh's slope wall correction (threshold scaled by
# horizontal/vertical = 15 levels) starts pulling below-zero rough vertices onto the flat table
# boundary column -> the "every vertex at x>=near_x is exactly 0" invariant would break.
MAX_BAND_M = 2 * int(SLOPE_THRESHOLD * HORIZONTAL_SCALE_M / VERTICAL_SCALE_M) * VERTICAL_SCALE_M

ROUGH_PATCH_SCENE_ATTR = "rough_ground_patch"
SAFETY_FLOOR_SCENE_ATTR = "rough_safety_floor"


def zero_mean_half_band_m(height_range) -> float:
    """Authored ``[lo, hi]`` -> the ± half-band about z=0 actually built."""
    lo, hi = float(height_range[0]), float(height_range[1])
    return (hi - lo) / 2.0


def _table_length_m() -> float:
    """TABLE_LENGTH from table_tennis.geometry (dimension source of truth), host-test safe.

    Normal package import on the pod.  On a bare host neither the package (its ``__init__``
    registers Isaac tasks) nor a file-path exec of geometry.py works (py3.8 chokes on its
    ``tuple[float, float]`` dataclass annotations), so fall back to READING the pinned constant
    out of the source text without executing it — geometry.py stays the single written-down
    source of the number either way.
    """
    value = getattr(_table_length_m, "_value", None)
    if value is None:
        try:
            from whole_body_tracking.tasks.table_tennis import geometry

            value = float(geometry.TABLE_LENGTH)
        except Exception:
            import pathlib
            import re

            path = (
                pathlib.Path(__file__).resolve().parents[1] / "table_tennis" / "geometry.py"
            )
            match = re.search(
                r"^TABLE_LENGTH:\s*float\s*=\s*([0-9.]+)",
                path.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            if match is None:
                raise RuntimeError(f"cannot read TABLE_LENGTH from {path}")
            value = float(match.group(1))
        _table_length_m._value = value
    return value


def patch_extents_m(near_x_m: Optional[float]) -> Tuple[float, float, float]:
    """(x_min, x_max, y_half) of the pad in the env frame.

    With a table (``near_x_m`` = env-frame x of the near edge): rough behind the edge, flat from
    the edge across the whole table plus a margin.  Without one (no racket_target command, e.g. a
    flat-tracking lineage): a symmetric all-rough pad around the robot.
    """
    if near_x_m is None:
        return (-X_BACK_M, X_BACK_M, Y_HALF_M)
    x_min = float(near_x_m) - X_BACK_M
    x_max = float(near_x_m) + _table_length_m() + X_FORWARD_MARGIN_M
    return (x_min, x_max, Y_HALF_M)


def _box_smooth(value: np.ndarray) -> np.ndarray:
    padded = np.pad(value, ((1, 1), (1, 1)), mode="edge")
    result = np.zeros_like(value, dtype=np.float64)
    for row in range(3):
        for col in range(3):
            result += padded[row : row + value.shape[0], col : col + value.shape[1]]
    return result / 9.0


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def build_patch_height_field(
    height_range,
    flat_from_x_m: Optional[float],
    x_min_m: float,
    x_max_m: float,
    y_half_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flat-heavy correlated height field in units of ``VERTICAL_SCALE_M``.

    Rows walk +x from ``x_min_m``, columns walk +y from ``-y_half_m`` (the
    ``convert_height_field_to_mesh`` convention). A seeded normal field is smoothed, centered,
    peak-normalized and quantized within the symmetric integer levels ``{-K, .., +K}``, where
    ``K = floor(((hi-lo)/2) / VERTICAL_SCALE_M)``. Every vertex with
    ``x >= flat_from_x_m`` (the table side) is forced to exactly 0. Four fixed smoothing passes
    correlate the sampled field before
    quantization, so adjacent 10 cm cells form a continuous slope/wave instead of independent white
    noise. A circular spawn platform is also exact flat; both flat boundaries have a smooth shoulder.
    """
    lo, hi = float(height_range[0]), float(height_range[1])
    if (hi - lo) < MIN_BAND_M - 1e-12:
        raise ValueError(
            f"rough patch band (hi - lo) must be >= {MIN_BAND_M} m — heights are "
            f"re-centred to ±(hi-lo)/2 about z=0, a narrower band quantizes to a "
            f"dead-flat pad (got [{lo}, {hi}])"
        )
    if (hi - lo) > MAX_BAND_M + 1e-12:
        raise ValueError(
            f"rough patch band (hi - lo) must be <= {MAX_BAND_M:g} m — beyond that the "
            f"slope wall correction pulls below-zero vertices onto the flat table "
            f"boundary column (got [{lo}, {hi}])"
        )
    half_band = zero_mean_half_band_m(height_range)
    ratio = half_band / VERTICAL_SCALE_M
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(
            f"rough patch band (hi - lo) must be a multiple of {2 * VERTICAL_SCALE_M:g} m "
            f"(heights quantize to {VERTICAL_SCALE_M * 1000:g} mm levels; a non-multiple "
            f"band would silently build a different amplitude than authored; got [{lo}, {hi}])"
        )
    # floor (never round up): the built band must never EXPAND beyond the authored one.
    levels = int(math.floor(ratio + 1e-9))
    if levels < 1:
        raise ValueError(
            "rough patch band (hi - lo) quantizes to zero levels; require "
            f"(hi - lo) >= {MIN_BAND_M} m (got half-band {half_band} m)"
        )
    # ceil (never truncate): the built span must never fall SHORT of the declared extents;
    # at most one extra cell of overshoot past x_max/y_half.
    num_rows = int(math.ceil((x_max_m - x_min_m) / HORIZONTAL_SCALE_M - 1e-9)) + 1
    num_cols = int(math.ceil(2.0 * y_half_m / HORIZONTAL_SCALE_M - 1e-9)) + 1
    if num_rows < 2 or num_cols < 2:
        raise ValueError("rough patch extents are degenerate")
    surface = rng.standard_normal((num_rows, num_cols))
    for _ in range(SMOOTHING_PASSES):
        surface = _box_smooth(surface)
    surface -= float(surface.mean())
    peak = float(np.max(np.abs(surface)))
    if peak <= 1.0e-12:
        raise RuntimeError("rough patch random field unexpectedly collapsed")
    surface /= peak
    x_coords = x_min_m + np.arange(num_rows) * HORIZONTAL_SCALE_M
    y_coords = -y_half_m + np.arange(num_cols) * HORIZONTAL_SCALE_M
    radius = np.sqrt(
        np.square(x_coords[:, None]) + np.square(y_coords[None, :])
    )
    surface *= _smoothstep01(
        (radius - SPAWN_FLAT_RADIUS_M)
        / (SPAWN_BLEND_RADIUS_M - SPAWN_FLAT_RADIUS_M)
    )
    if flat_from_x_m is not None:
        surface *= _smoothstep01(
            (float(flat_from_x_m) - x_coords[:, None]) / TABLE_BLEND_WIDTH_M
        )
    hf = np.rint(surface * levels).astype(np.int16)
    np.clip(hf, -levels, levels, out=hf)
    hf[radius <= SPAWN_FLAT_RADIUS_M + 1.0e-9] = 0
    if flat_from_x_m is not None:
        hf[x_coords >= float(flat_from_x_m) - 1.0e-9, :] = 0
    return hf


def _spawn_rough_ground_patch(prim_path, cfg, translation=None, orientation=None):
    """Spawner body (wrapped with isaaclab's ``@clone`` at attach time).

    Runs inside Kit at scene build.  The global numpy RNG has already been seeded from
    ``env_cfg.seed`` by ``ManagerBasedEnv.__init__`` (seed is set BEFORE the scene is built), so
    deriving the pad seed from it keeps the bump pattern reproducible per run seed.
    """
    import isaacsim.core.utils.prims as prim_utils
    import trimesh

    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
    from isaaclab.terrains.utils import create_prim_from_mesh

    seed = cfg.seed
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    hf = build_patch_height_field(
        cfg.height_range_m,
        cfg.flat_from_x_m,
        cfg.x_min_m,
        cfg.x_max_m,
        cfg.y_half_m,
        rng,
    )
    vertices, triangles = convert_height_field_to_mesh(
        hf, HORIZONTAL_SCALE_M, VERTICAL_SCALE_M, SLOPE_THRESHOLD
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # convert_* builds the surface from (0, 0); shift so the pad sits at its env-frame extents.
    mesh.apply_translation((cfg.x_min_m, -cfg.y_half_m, 0.0))
    create_prim_from_mesh(
        prim_path,
        mesh,
        translation=translation,
        orientation=orientation,
        physics_material=cfg.physics_material,
        visual_material=cfg.visual_material,
    )
    return prim_utils.get_prim_at_path(prim_path)


def _isaac_spawner_bindings():
    """Build (once) the Isaac-dependent spawner callable + cfg class as MODULE attributes.

    人话:train.py 会把 env cfg 原样 pickle 进 launch 记录(params/env.pkl)。pickle 按
    ``module.qualname`` 找类和函数——函数体内现定义的类/闭包直接炸。所以这两个对象必须挂成
    模块属性;又因为它们 import isaaclab,只能懒构建(host-only 测试 import 本模块不动 Isaac)。
    配套的模块级 ``__getattr__``(PEP 562)让"新进程先 unpickle"这种顺序也能解析到它们。
    """
    cls = globals().get("RoughGroundPatchSpawnerCfg")
    if cls is not None:
        return cls

    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone as _clone
    from isaaclab.utils import configclass

    wrapper = _clone(_spawn_rough_ground_patch)
    wrapper.__name__ = "spawn_rough_ground_patch"
    wrapper.__qualname__ = "spawn_rough_ground_patch"
    globals()["spawn_rough_ground_patch"] = wrapper

    @configclass
    class RoughGroundPatchSpawnerCfg(SpawnerCfg):
        """Per-env zero-mean rough pad (fields are read back by the ground_plant contract)."""

        func: Callable = wrapper
        height_range_m: Tuple[float, float] = (0.0, 0.0)  # AUTHORED band; built as ±(hi-lo)/2
        flat_from_x_m: Optional[float] = None
        x_min_m: float = 0.0
        x_max_m: float = 0.0
        y_half_m: float = 0.0
        seed: Optional[int] = None  # None -> derive from the run-seeded global numpy RNG
        physics_material: object = None
        visual_material: object = None

    RoughGroundPatchSpawnerCfg.__module__ = __name__
    RoughGroundPatchSpawnerCfg.__qualname__ = "RoughGroundPatchSpawnerCfg"
    globals()["RoughGroundPatchSpawnerCfg"] = RoughGroundPatchSpawnerCfg
    return RoughGroundPatchSpawnerCfg


def __getattr__(name):
    # PEP 562: resolve the lazily-built Isaac bindings for pickle/e.g. env.pkl round-trips.
    if name in ("RoughGroundPatchSpawnerCfg", "spawn_rough_ground_patch"):
        _isaac_spawner_bindings()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def attach_rough_ground_patch(env_cfg, height_range):
    """Mutate ``env_cfg`` from the plane recipe to the per-env zero-mean rough pad.

    Called by train.py's ``task.plant.terrain_rough_height_range`` branch AFTER the ground/robot
    friction overrides (so the pad inherits the post-override ground material).  Steps:

    1. add ``scene.rough_ground_patch`` — the per-env static height-field collider;
    2. add ``scene.rough_safety_floor`` — a global backstop plane below the deepest valley;
    3. remove ``scene.terrain`` — env origins fall back to the GridCloner grid, which is exactly
       where every cloned static asset (tables included) already lives.

    Returns the human-readable ``applied`` lines for the launch summary.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    scene = getattr(env_cfg, "scene", None)
    terrain = None if scene is None else getattr(scene, "terrain", None)
    if terrain is None or getattr(terrain, "terrain_type", None) != "plane":
        raise RuntimeError(
            "attach_rough_ground_patch requires the plane TerrainImporter starting recipe"
        )
    if getattr(scene, ROUGH_PATCH_SCENE_ATTR, None) is not None:
        raise RuntimeError("rough ground patch is already attached")
    material_src = getattr(terrain, "physics_material", None)
    if material_src is None:
        raise RuntimeError("attach_rough_ground_patch requires scene.terrain.physics_material")

    lo, hi = float(height_range[0]), float(height_range[1])
    half_band = zero_mean_half_band_m((lo, hi))
    if half_band * 2.0 < MIN_BAND_M:
        raise RuntimeError(
            f"rough patch band (hi - lo) must be >= {MIN_BAND_M} m; got [{lo}, {hi}]"
        )

    def _ground_material():
        return sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode=getattr(material_src, "friction_combine_mode", "multiply"),
            restitution_combine_mode=getattr(material_src, "restitution_combine_mode", "multiply"),
            static_friction=float(material_src.static_friction),
            dynamic_friction=float(material_src.dynamic_friction),
            restitution=float(getattr(material_src, "restitution", 0.0)),
        )

    rt = getattr(getattr(env_cfg, "commands", None), "racket_target", None)
    near_x = float(rt.vb_table_near_x) if rt is not None and hasattr(rt, "vb_table_near_x") else None
    x_min, x_max, y_half = patch_extents_m(near_x)

    RoughGroundPatchSpawnerCfg = _isaac_spawner_bindings()

    setattr(
        scene,
        ROUGH_PATCH_SCENE_ATTR,
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RoughGroundPatch",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            spawn=RoughGroundPatchSpawnerCfg(
                height_range_m=(lo, hi),
                flat_from_x_m=near_x,
                x_min_m=x_min,
                x_max_m=x_max,
                y_half_m=y_half,
                seed=None,
                physics_material=_ground_material(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.31, 0.28, 0.25), roughness=0.9
                ),
            ),
        ),
    )

    # Backstop plane below the deepest valley: catches anything that leaves the pad. Visual grid
    # sized to cover the cloner grid.
    drop = half_band + SAFETY_FLOOR_MARGIN_M
    num_envs = getattr(scene, "num_envs", None)
    spacing = getattr(scene, "env_spacing", None)
    if isinstance(num_envs, int) and num_envs > 0 and isinstance(spacing, (int, float)):
        extent = math.ceil(math.sqrt(num_envs)) * float(spacing) + 2.0 * X_BACK_M + 20.0
    else:
        extent = 400.0
    setattr(
        scene,
        SAFETY_FLOOR_SCENE_ATTR,
        AssetBaseCfg(
            prim_path="/World/roughSafetyFloor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -drop)),
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=_ground_material(), size=(extent, extent)
            ),
        ),
    )

    # Remove the plane importer LAST: scene.env_origins falls back to the GridCloner grid — the
    # same grid every cloned static asset (tables) is placed on. Keep the sim default material
    # aligned with the ground values (post_init aliased it to the plane's material object).
    scene.terrain = None
    sim = getattr(env_cfg, "sim", None)
    if sim is not None and hasattr(sim, "physics_material"):
        sim.physics_material = _ground_material()

    flat_txt = (
        f"flat table zone x>={near_x:g} m (near edge; table+{X_FORWARD_MARGIN_M:g} m margin)"
        if near_x is not None
        else "no table command -> all-rough pad"
    )
    return [
        (
            "scene.rough_ground_patch: per-env flat-heavy correlated pad "
            f"within ±{half_band:.3f} m of z=0 (authored [{lo:g}, {hi:g}]), rough x<"
            f"{near_x if near_x is not None else x_max:g} (robot side), {flat_txt}, "
            f"extents x=[{x_min:g}, {x_max:g}] y=±{y_half:g}, cell {HORIZONTAL_SCALE_M:g} m, "
            f"smoothing passes {SMOOTHING_PASSES}, spawn-flat radius "
            f"{SPAWN_FLAT_RADIUS_M:g} m, "
            f"quantized {VERTICAL_SCALE_M * 1000:g} mm"
        ),
        f"scene.rough_safety_floor: global backstop plane at z=-{drop:.3f} m",
        (
            "scene.terrain=None: plane importer removed; env origins fall back to the cloner "
            "grid where the cloned tables live (fresh-from-random only; ground_plant 合同块会"
            "拒绝平地谱系 resume)"
        ),
    ]
