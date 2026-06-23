# Policy Observation And Action

Status: Draft

## HITTER-Compatible Contract

The baseline policy should be compatible with the HITTER-style separation:

- Planner provides target racket position, target racket velocity, and time to strike.
- WBC policy combines planner target, robot state, and previous action.
- Policy outputs desired joint positions.

## Actor Observation Candidates

Based on HITTER and current HOPE docs:

- base angular velocity
- projected gravity
- base forward vector
- target base position relative to current base
- target racket position
- target racket velocity
- time to strike
- joint positions
- joint velocities
- previous action

## Critic-Only Candidates

For asymmetric actor-critic training:

- base linear velocity
- body poses
- time left in episode
- reference joint positions and velocities

## Action

Initial baseline action:

- desired joint position targets in canonical joint order

The exact dimension is pending the accepted A3 joint contract.

## Update Rule

Any observation, action, scaling, normalization, latency, command frequency, or joint-order change must update this file and the relevant gate docs.
