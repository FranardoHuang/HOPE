# Yikang branch selective-integration audit (2026-07-12)

Scope: compare yikang's small branch-only changes with `origin/main@b2067ba`, port only current,
verifiable pieces, and never merge either old-base branch wholesale. No Pod, simulator runtime,
GPU, training or hardware was used.

## A. Fit-lineage ball-physics reference: accepted with hardening

Source evidence: `stage1-fixed-point` commits `bc86995` and `f0ac2fb`. The reconstructed
`hope_training/ball_physics_fit/reference_oracle.py` is now the default reference for
`test_ball_physics_vs_record.py`; an explicitly set but incomplete `RECORD_DIR` fails before a
missing dependency can turn the run into a skip. The reference delegates to the committed
fit-lineage NumPy contact model, not the Torch implementation under test.

The branch adds two protections not present in the source commit:

1. contact and table normals are normalized with overflow/underflow-safe scaling; zero, NaN, Inf,
   wrong-shape and multi-row table normals raise instead of producing NaNs;
2. every result can record SHA-256 for `reference_oracle.py`, `contact_model.py` and
   `ball_physics_venue.yaml`, plus their combined content identity.

At this branch revision the combined identity is `4a08b9f9...4fa23` (individual full hashes are
printed by the test). Host-only contract tests pass `7` cases inside
`test_reference_oracle_contract.py`. The full script then ran in the local `qai` Torch CPU
environment and reported `ALL PASSED`: table contact max `|dv|=1.69e-10`, max `|dw|=9.35e-10`;
paddle `1.38e-10` / `4.63e-9`; RK4 max position error `0`; landing error `0.000 mm`.

The absent `test_ball_channel_vs_record.py` from yikang's old base was not imported wholesale. Its
large C++/channel surface is not required to restore the existing main parity test and should be
reviewed independently against the current C++ stack if revived.

## B. Plain-MuJoCo stand diagnostic: accepted with contract correction

Source evidence: `yikang-linux-port-0711@6b10998`. The tool remains diagnostic-only but no longer
retypes gains by joint-name pattern. It parses the production `a3_default_angles`,
`a3_pd_stand_kps` and `a3_pd_stand_kds` arrays from the tracked production header at runtime and
fails on missing, duplicate, wrong-length or non-finite arrays.

The original script supplied head gains `40/2` while calling them production PD_STAND. The current
header explicitly defines a 29-DOF policy view and says the neck slots remain passive. The port now
keeps `head_yaw_joint` and `head_pitch_joint` passive. It does not alter the vendor MJCF or its
integrator.

`--identity-only` is dependency-light and binds:

- vendor `a3_pingpong.xml`: `2ab1cd31...3feb97`;
- production parameter header: `df73e3f6...c5c8d8`.

`--check` additionally records finite state, pelvis-z min/max/drift, maximum pelvis tilt, left/right/
both-foot floor-contact fractions, timestep and actual MuJoCo integrator. Its default thresholds are
diagnostic thresholds, not Gate3 acceptance criteria. `--snapshot` is visual evidence only. Source/
identity tests pass `4` cases and pycompile passes; the 10-second numerical run was not executed on
this Mac because the MuJoCo Python binding is absent. AimRT, planner, policy, Gate3/Gate3B and
hardware remain `not_run` by construction.

## C. `head_discipline`: diagnosis retained, old recipe patch excluded

Source evidence: `407a443`, based on the old `hitter@5c346ea` surface. The diagnosis is credible:
head yaw/pitch are not covered by the tracked upper-body set, and a static offset can evade velocity/
action-rate terms. But the patch enables `-0.5` in `HOPEPingPongHitterPureRallyFinalV2.yaml` and adds
a FinalV2-only whitelist. Neither FinalV2 nor FinalV2Plus exists on `origin/main@b2067ba`; importing
those pieces would create a stale recipe surface rather than integrate with current main.

There are also two unresolved design interactions:

- `origin/hitter` derives FinalV2Plus's exact reward key set from FinalV2 minus a fixed exclusion
  set, so adding the key requires a deliberate V2Plus semantic decision rather than accidental
  inheritance;
- `origin/hitter@0fccc3c` uses a passive-head action contract for FinalV3, an alternative mechanism
  for the same symptom. Reward discipline and passive action masking must not be silently stacked.

Therefore this branch ports no head reward term, whitelist or YAML weight and launches no training.
Dependency-light guards verify that current main has no FinalV2/FinalV2Plus recipe and no silently
exposed `head_discipline_weight`. If the current training line later adopts a reward-side solution,
first define its named recipe and explicit zero default, then run a paired validation against the
unchanged control and the passive-head alternative. Because head shaping overlaps balance,
recovery and ready-state behavior, any combined rollout must treat it as a reward interaction and
use the same constant-total-budget/paired discipline as the recovery mixture work; `-0.5` is only a
hypothesis, not a validated default.

## Verification

```bash
python3 -m py_compile \
  scripts/view_a3_stand.py \
  hope_training/ball_physics_fit/reference_oracle.py \
  hope_training/whole_body_tracking/tests/test_ball_physics_vs_record.py

pytest -q \
  tests/test_view_a3_stand.py \
  hope_training/whole_body_tracking/tests/test_reference_oracle_contract.py \
  tests/test_yikang_head_discipline_integration_decision.py

python3 scripts/view_a3_stand.py --identity-only
/Users/Franco/opt/anaconda3/envs/qai/bin/python \
  hope_training/whole_body_tracking/tests/test_ball_physics_vs_record.py

/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py \
  -k 'not motion_loader_rejects_wrong_body_order_and_fps_contracts and not motion_loader_rejects_non_scalar_or_nonfinite_fps'
```

Current results: selective host suite `13 passed`; identity-only exits zero; Torch parity is
`ALL PASSED`; current reward/config coverage is `47 + 41 = 88 passed`. Running the latter MDP file
without deselection gives `88 passed, 2 failed`; both failures are the pre-existing current-main
`MotionLoader` treating a single `pathlib.Path` as iterable before reaching the intended validation,
not this selective diff. The MuJoCo 10-second stand remains unrun because no local environment has
the binding. Since the head reward code was excluded, current reward/config bytes and behavior are
unchanged.
