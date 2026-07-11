# Phase-1 q10 collector fixture

`manifest.json` is a two-arm, one-barrier causal face pair. The unit test copies these
immutable bytes and materializes tiny local worker-state, judge-report, MuJoCo-summary,
checkpoint-audit, and archive-index artifacts around it. This keeps the fixture readable
while exercising every SHA edge with real bytes instead of checked-in placeholder hashes.

The fixture is synthetic and dependency-light. It does not represent a simulator result,
authorize a decision, connect to a Pod, or invoke a real robot.
