# HOPE Table-Tennis Match Environment (Isaac Lab)

A clean, modular Isaac Lab task that simulates a **standard table-tennis competition scene** — floor,
table, net (+ posts), a dynamic ITTF ball, and the **Agibot A3** humanoid — with realistic ball flight,
bounce, and racket/table contact. The table/net are invisible cuboid colliders overlaid with a
**realistic USD mesh** for visuals (see [Table visuals](#table-visuals-usd-overlay)). Built as a
manager-based `ManagerBasedRLEnv` so RL training can be layered on later; the first goal here is a
**correct physics + visualization scene**.

Gym id: **`HOPE-TableTennis-AgibotA3-v0`**

## Coordinate frame (canonical HOPE frame, used everywhere)

The simulation world frame **is** the HOPE frame (ROS 2 REP-103), identical to the planner / mocap docs
and `hope_ws/.../hope_world_frame.yaml`:

| Axis | Direction | Range on the table |
|------|-----------|--------------------|
| **X** | toward Player Two (P2), along the table length | `0 → +2.74 m` |
| **Y** | left, from P1's perspective, along the table width | `0 → −1.525 m` |
| **Z** | up; **z = 0 is the table surface** | floor at `z = −0.76 m` |

Origin = the **near-side left corner of the table surface** (P1 perspective). Each parallel environment
is an independent court whose local origin coincides with this HOPE origin, so an asset's
environment-local position *is* its HOPE-frame position. Landmarks (net center `(1.37, −0.7625, 0)`,
P1/P2 half centers, floor at `−0.76`) all match the reference docs — see [`geometry.py`](geometry.py),
which is the single source of truth and is regression-tested against the ITTF/HOPE constants in
`tests/test_table_tennis_geometry.py`.

## How to run (visualization, no policy)

This needs Isaac Sim / Isaac Lab, which live in the **`grasping` distrobox**. The launcher
`hope_isaac_py` (defined by `setup_train_env.sh`) runs Isaac's bundled Python with the working-tree
PYTHONPATH, so your local edits to this task win:

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh                          # defines hope_isaac_py + sets PYTHONPATH

hope_isaac_py scripts/play_table_tennis.py                 # 1 court, robot free-standing, spin-aware physics ON
hope_isaac_py scripts/play_table_tennis.py --num_envs 9    # 9 courts
hope_isaac_py scripts/play_table_tennis.py --fix_base      # pin the pelvis (stable view; no balance policy yet)
hope_isaac_py scripts/play_table_tennis.py --disable_aero  # fall back to PhysX gravity only (no drag/spin contacts)
hope_isaac_py scripts/play_table_tennis.py --magnus 0.1    # override the Magnus coefficient + heavier serve spin
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300   # no-GUI smoke run (loads + steps)
```

Each reset serves a ball from over the P2 half toward the P1-side robot; you should see it arc (with drag +
Magnus curvature), bounce on the table at the mocap-fitted normal restitution `e_n = 0.908`, and continue
toward the robot. The robot holds its default standing pose (zero action). The console prints `ball
aerodynamics active: True/False` so you can confirm the spin-aware physics callback registered (on unless
`--disable_aero`).

Notes:
* **First launch is slow** — `UrdfFileCfg` converts the A3 URDF → USD once, then caches it.
* **Without `--fix_base` the robot may drift/topple** after a few seconds (there is no balance/return
  controller yet — that is the RL follow-up). Use `--fix_base` for a stable view of the ball physics.

## Physics model

The ball flight + bounce now use the **experimentally-calibrated, spin-aware** model fitted from mocap
(see the repo-root `docs/ball_physics.md` and the shared `configs/ball_physics.yaml`). Ball mass is
**2.7 g**, radius 20 mm, hollow-sphere inertia (c = 2/3).

* **Flight** (drag + Magnus) is added every physics substep (400 Hz) by `TableTennisEnv` via a
  physics-step callback (`a = −k_d|v|v + k_m(ω×v)`, `k_d = 0.1222` 1/m, `k_m = 0.0042`). See
  [`ball.py`](ball.py). On by default; `--disable_aero` reverts to PhysX gravity only.
* **Contacts are code-driven, not PhysX restitution.** PhysX cannot reproduce the fitted angle-dependent,
  spin-coupled impulse, so `ball_restitution = 0` neutralizes its bounce (it only prevents tunnelling, via
  CCD) and the **spin-equation** model ([`physics/spin_contact.py`](physics/spin_contact.py)) sets the
  post-contact velocity AND spin directly: table normal restitution `e_n = 0.908` (GRIPPY tangential —
  near no-slip, converts speed↔spin; OptiTrack recal, was slippery 0.87);
  racket hit `e_eff = 0.463` with the fitted angle-dependent friction. The net keeps a low PhysX
  restitution so the ball still dies on a net touch.
* **Outgoing-shot landing** is predicted at each racket hit by forward-integrating the same flight model
  ([`physics/landing.py`](physics/landing.py)); exposed as the `landing_in_opponent_half` reward and the
  `ball_predicted_landing` critic observation.
* Physics runs at **400 Hz** (`sim.dt = 0.0025`), control at **100 Hz** (`decimation = 4`). The high
  physics rate keeps the small, fast ball from tunnelling and bounds the code-driven-contact snap error.

## Modularity / extension points

| Concern | Where |
|---|---|
| Table / net / ball dimensions, landmarks, materials, serve, bounds | [`geometry.py`](geometry.py) |
| Ball aerodynamics (drag / Magnus) math + config | [`ball.py`](ball.py) |
| Scene assets (one `build_*` helper per prim) + MDP managers | [`table_tennis_env_cfg.py`](table_tennis_env_cfg.py) |
| Per-substep force application / env class | [`table_tennis_env.py`](table_tennis_env.py) |
| Realistic table+net USD mesh (visual overlay) | [`table_usd/`](table_usd/) |
| Ball/robot observations, serve event, rewards, terminations | [`mdp/`](mdp/) |
| Robot choice, stand pose, action scale, Gym registration | [`config/agibot_a3/`](config/agibot_a3/) |

### Table visuals (USD overlay)

The table / net / posts / center line are kept as **invisible cuboid colliders** (`visible=False`) that
own all bounce physics, and a realistic **USD mesh** (Purdue PACE, ICRA 2026, MIT — see
[`table_usd/LICENSE-PACE-ICRA2026-MIT.txt`](table_usd/LICENSE-PACE-ICRA2026-MIT.txt)) is overlaid for
looks via `build_table_usd_visual_cfg` (`scene.table_visual`). Only the USD's
*base* geometry layer is referenced, which carries **no PhysX colliders**, so physics is unchanged. The
USD frame is aligned to the HOPE frame by translating its local origin to
`(TABLE_LENGTH/2, −TABLE_WIDTH/2, FLOOR_Z)` (floor at its local z = 0, surface at 0.76, net top 0.9125).
The USD is ~1.76 m wide vs the ITTF/cuboid 1.525 m, so the visual table is slightly wider than its
collider — cosmetic only. To go back to plain boxes, drop `table_visual` and flip the cuboids'
`visible=False` back to `True`; for memory-tight headless training set `scene.table_visual = None`.

To add a second robot (P2), add an articulation to the scene cfg. To add real match rewards
(return-over-net, landing in the opponent half, racket-to-ball tracking), add terms in `RewardsCfg` /
`mdp/rewards.py`.

## Known limitations / calibration TODOs

These are first-pass defaults chosen for a visibly correct scene; they need an in-sim calibration pass
(the PhysX analogue of `hope_planner.calibrate_ball_physics`). All knobs live in `geometry.py` /
`ball.py` so calibration touches one place:

1. **In-sim validation of the code-driven contacts** (the offline torch↔Record parity is already a green
   test): on an Isaac/GPU box, confirm the realized bounce equals `e_n = 0.908` (and a glancing no-spin
   drop picks up spin — grippy table) and that the velocity
   write in the physics callback is applied before PhysX integrates the substep (the load-bearing
   assumption). Also verify which racket-body / face-normal axis is correct for paddle-hit detection.
2. **Robot facing**: the A3 is spawned with identity orientation (assumed +X-forward). If the URDF
   forward axis is −X, flip `init_state.rot` in `config/agibot_a3/table_tennis_env_cfg.py`.
3. **Balance**: there is no balance/return policy yet, so the free-standing robot may drift/topple over
   several seconds — use `--fix_base` for a stable visualization. A returner policy is the RL follow-up.
4. **Very fast smashes** (≫ 8 m/s) may still need PhysX CCD on the ball to avoid tunnelling through the
   2.9 mm racket blade; the 400 Hz physics rate covers normal serve/rally speeds.

## Verification status

* `tests/test_ball_physics_vs_record.py` — the torch port (contact / flight / landing) matches the Record
  numpy oracle to <1e-6 (contact ~1e-10, flight bit-exact, landing 0 mm). **Passing**, CPU-only.
* `tests/test_table_tennis_geometry.py` (frame/geometry + aero defaults + YAML consistency) — **passing**
  on a plain Python host (torch/YAML-dependent checks auto-skip if missing).
* All modules pass `py_compile`.
* **In-sim runtime** (asset spawn, code-driven contacts applied via the physics callback, landing reward,
  robot stand) must be verified inside Isaac Lab with `scripts/play_table_tennis.py` — not runnable on a
  host without Isaac Sim.
