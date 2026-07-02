# Frames And Coordinates

Status: Draft

## Canonical World Frame

Use the HOPE `world` frame unless a gate explicitly says otherwise.

- Origin: near-side left corner of the table surface.
- X: toward opponent along table length.
- Y: left along table width. The table extends along `-Y` (its center is at `y = -0.7625`).
- Z: up. **World `z = 0` IS the table surface.** The floor is at `z = -0.76` (`floor_origin`). The `0.76 m` value is the surface-to-floor height, NOT the surface world Z.
- Table length: `2.74 m`.
- Table width: `1.525 m`.
- Table surface-to-floor height: `0.76 m` (the surface itself is at world `z = 0`).
- Net height: `0.1525 m`.

Landmark coordinates (from `hope_world_frame.yaml`):

- `table_center`: `[1.37, -0.7625, 0.0]`
- `net_center`: `[1.37, -0.7625, 0.0]`
- `p1_half_center`: `[0.685, -0.7625, 0.0]`
- `p2_half_center`: `[2.055, -0.7625, 0.0]`
- `floor_origin`: `[0, 0, -0.76]`

Current config source:

- [`hope_ws/src/hope_bringup/config/hope_world_frame.yaml`](../../hope_ws/src/hope_bringup/config/hope_world_frame.yaml)
- [`hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py) for the Isaac table-tennis scene.

For `HOPE-TableTennis-AgibotA3-v0`, the Isaac environment-local world frame is intentionally the HOPE
frame: the table surface is `z = 0`, the floor is `z = -0.76`, and table/net landmarks match the values
above. Do not introduce a table-center origin in that task without updating this file and G04.

## Mocap Frames

Current expected mocap object names:

- Table rigid body: `PPT`
- Player 1 robot rigid body: current config default `ppp2` (publishes `/P1/pose`)
- Player 2 robot rigid body: current config default `ppp3` (publishes `/P2/pose`)
- Ball: auto-detected marker unless pinned by live topic name

Current relay config:

- `hope_ws/src/hope_bringup/config/avatar_pro_vrpn.yaml`

## Base Link

`base_link` must come from the robot model and SDK. Do not assume the mocap marker cluster equals `base_link` unless the transform has been measured.

Pending measurements:

- `P1 mocap frame -> P1_base_link`
- `P2 mocap frame -> P2_base_link`
- A3 standing `base_link` height

## Racket Frame

The racket is not tracked by motion capture. It must be inferred from:

1. `world -> base_link`
2. robot joint state
3. robot model FK
4. fixed racket mount transform

Current (non-final) racket mount, from `robots/agibot_a3.py` / `HOPEPingPong.yaml`:

- `A3_MOUNT_OFFSET = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)` in the `right_wrist_yaw_Link` frame.
- Racket-face normal = racket-local `+Y` (`mount_normal_axis = 1`, `mount_normal_sign = +1`).

These values are current but NOT final and are expected to change after calibration.

## Update Rule

Any frame, axis, origin, static transform, or measured calibration change must update this file and the relevant gate doc.
