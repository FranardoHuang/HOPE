# Frames And Coordinates

Status: Draft

## Canonical World Frame

Use the HOPE `world` frame unless a gate explicitly says otherwise.

- Origin: near-side left corner of the table surface.
- X: toward opponent along table length.
- Y: left along table width.
- Z: up.
- Table length: `2.74 m`.
- Table width: `1.525 m`.
- Table surface height: `0.76 m`.
- Net height: `0.1525 m`.

Current config source:

- `hope_ws/src/hope_bringup/config/hope_world_frame.yaml`

## Mocap Frames

Current expected mocap object names:

- Table rigid body: `PPT`
- Player 1 robot rigid body: `P1`
- Player 2 robot rigid body: `P2`
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

The racket mount transform is not yet canonical.

## Update Rule

Any frame, axis, origin, static transform, or measured calibration change must update this file and the relevant gate doc.
