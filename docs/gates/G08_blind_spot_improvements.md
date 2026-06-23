# G08 Blind-Spot Improvements

Status: Research track

## Goal

Improve beyond the HITTER-compatible baseline by targeting weaknesses that can decide matches.

This gate should not block the first reproduction loop. It becomes important once the baseline can hit and return reliably.

## Inputs

- Baseline planner and WBC results.
- Real and simulated evaluation data.
- Failure cases from G03-G07.
- References such as TTRL, table-tennis papers, and opponent-modeling work.

## Candidate Tracks

1. Spin perception and spin-aware ball dynamics.
2. Double-bounce and short-ball handling.
3. Deep-ball and non-fixed hitting-plane handling.
4. Serve generation.
5. Topspin, backspin, sidespin, block, chop, and loop stroke repertoire.
6. Opponent intent modeling and tactical adaptation.
7. Multi-agent training.
8. Vision-based perception to reduce mocap dependency.

## Related Directories

- `hope_ws/src/hope_planner`
- `hope_training/whole_body_tracking`
- `agi/A3_MuJoCo_Sim/`
- `external_repos/TTRL-ICRA2026`
- Future experiment folders to be defined by project management

## Operation Docs

Pick the operation doc based on the selected mini-spec. Common starting points:

- [../operations/run_planner.md](../operations/run_planner.md)
- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)

## Acceptance Criteria

Each blind-spot track needs its own mini-spec before implementation:

- Failure case being targeted.
- Measurable improvement metric.
- Required data.
- Required simulator changes.
- Deployment risk.
- Owner and expected demo.

## Current State

Done:

- HITTER paper limitations have been identified: fixed hitting plane, external mocap dependency, ignored spin, limited stroke repertoire, no autonomous serving, no explicit multi-agent/opponent adaptation.
- TTRL is locally available as a reference.

Not done:

- No blind-spot track has been selected for implementation.
- No spin/double-bounce/serve experiments are implemented.

## Risks

- Starting this gate before the baseline works can scatter effort.
- The highest-impact blind spot may depend on first real/RL results.

## Next Steps

1. Wait for first RL loop and baseline failure modes.
2. Pick one blind spot with high match impact and low integration risk.
3. Write a mini-spec before code changes.
