# v12, fifth-action and displacement-conditioned lower-body teacher design

Status: Design
Human owner: Franco
Executor: Codex (design); runtime executor unassigned
Branch: Franco_codex/new-motion-batch-20260713

## Question and decision scope

This record keeps three related but separately decidable questions:

1. Does the v12 forehand/backhand block pair beat the best earlier block candidate on a
   block-specific paper?
2. Does a backhand high-press action add safe return coverage for high balls that the four-action
   library misses?
3. Can one displacement-conditioned lateral-step teacher be composed with different upper-body
   strokes without losing racket contact geometry, support-foot validity or balance?

The exact raw-video intake is complete, but no processed motion or immutable behavioral paper
exists. This record is therefore `Design`, not a preregistration or queue authorization.

## Inputs and immutable bindings

- raw intake: `configs/motion_video_intake_20260713.json`, SHA-256
  `44b00b3c46c837d797990bc6f6255055c0ff83c1bb8643ca81f9707033ca304c`;
- historical Franco/v6/v7 intake and safety/counterfactual ledgers remain separate generations;
- final physical arbiter: exact Agibot vendor MuJoCo Gate3 for one shot and Gate3B for randomized
  arrival; Isaac is training/diagnostic only.

Missing bindings before any run include canonical-beta GVHMR/GMR outputs, runtime-order schema-2
NPZs, manually reviewed phase events, compatible question banks, selector train/exam split,
vendor MJCF/runtime SHA and a fixed transition/seed budget.

## Design and controls

### v12 and the fifth high-press action

Process only the v12 pair as the new v-series primary candidate; "expected best" is a hypothesis.
Compare it against the strongest safe earlier block candidate under the same block-specific incoming
ball distribution, station envelope, per-side denominator, seeds and training budget. The forehand
cell cannot promote until the signed-face honesty gate is closed.

The high-press action gets its own question family: high reachable contact points, a forward and
downward-facing racket path, and legal net/table landing. It is not allowed to inherit a loop paper
or to gain credit only from a kinematic counterfactual. Four-versus-five actions is tested only
after all five candidates independently pass the same safety/dynamics gates. The stable selector
is fit on a train split from incoming-ball features and chooses the action with the best conservative
return estimate; the immutable exam split never supplies selector labels. Ball quality is future
work and is not part of the first selector.

### Lateral teacher composition

Every upper- and lower-body source is segmented into three event intervals: preparation/step,
nominal strike support, and follow-through/recovery. Because the recordings are air motions, a
human-reviewed nominal strike frame is only an alignment anchor until a task-specific contact
manifold confirms it.

For each pre/post interval, use monotone event-anchored time maps. The composed interval duration is
the longest **dynamically feasible** upper/lower duration, not merely the video with most frames.
TOPP or another time-scaling solver may lengthen a fixed path under velocity/acceleration limits;
it cannot repair a wrong foot-contact path, change stride distance or prove balance.

Ownership at the seam is explicit:

- the lower-body teacher owns world root translation/yaw, feet and leg joints/contact phases;
- the upper-body stroke is represented pelvis-relative and owns the racket/contact objective;
- pelvis height/roll/pitch and torso are coupled seam variables, resolved by constrained whole-body
  IK/trajectory optimization rather than copied from both sources;
- the strike anchor must preserve racket position/velocity/signed face while support-foot pose,
  foot clearance/no-slip and a non-collapsed stance remain valid.

The closing step does not target two absolute foot poses. For each candidate, estimate the initial
heading-aligned horizontal left-to-right foot-separation vector from the first stable ready window
(a robust median, not one noisy video frame). The terminal ready set restores that relative vector
after removing common root translation/yaw. This preserves both lateral width and any intended
fore-aft stagger even when the two feet do not start symmetrically. It applies only after recovery:
the step and strike-support phases may change the separation. A hard minimum width/no-crossing
guard remains active throughout, so a source whose initial stance is itself unsafe is rejected
rather than canonized.

Lateral distance is a signed parameter of foot placements/root displacement, not a uniform scale
of every joint or z coordinate. First measure the safe interval from each source candidate. Test
independently recorded left/right paths before allowing reflection; a mirrored lower body must pass
joint-map, asymmetry, self-collision and vendor dynamics checks and cannot mirror the right-hand
racket stroke.

The minimum composition ablation is:

- C0: current stroke teacher with no displacement-conditioned lower-body composition;
- C1: event-aligned composed teacher at the recorded displacement;
- C2: the same teacher conditioned on a frozen displacement grid including zero;
- C3: only after C2 passes, compare independently recorded two-direction teachers with one
  mirror-derived family.

Naive "pad both to the longest clip and concatenate joints" is retained only as a falsification
control. It cannot enter RL if root ownership, foot contacts or whole-body safety fail. A TOPP
on/off comparison holds the geometric path and contact anchors fixed and repeats every downstream
gate after retiming.

## Acceptance and failure rules

Before any RL slot, every exact generated motion must pass runtime-order schema 2, L0 finite/limit/
endpoint checks, vendor-MJCF L1 self/racket-handle clearance, full-trajectory table/net swept
clearance of at least 5 mm, and vendor dynamics/foot-contact replay. Safety is non-compensable.

A lateral interface is accepted only if all preregistered displacements preserve the signed strike
geometry and support constraints; isolated success at the recorded distance is not generalization.
Every terminal recovery must also return to the candidate-bound initial foot-separation ready set;
ending with a narrower closed stance is a failure even if the robot has not yet fallen.
The composed teacher must improve lateral task coverage or balance without degrading common-station
return beyond a frozen non-inferiority margin. The later four-versus-five decision requires unique
held-out coverage, not merely a higher training reward.

## Reproduction

Only the intake audit is executable now:

```bash
python3 scripts/audit_motion_video_intake.py \
  --manifest configs/motion_video_intake_20260713.json \
  --source-root /Users/Franco/Downloads
```

GVHMR/GMR, event annotation, composition, papers and RL commands remain intentionally absent until
their input/output hashes and no-clobber artifact roots are frozen.

## Results

Design only. Seven raw videos passed intake; no efficacy, composition or simulator result exists.

## Limitations and claims not made

Upper/lower motion independence is an engineering hypothesis. Pelvis/torso dynamics, angular
momentum and foot contact couple them. TOPP provides path timing, not contact stability. One
successful displacement does not prove continuous lateral generalization, and left/right symmetry
is not assumed.

## Decision and next action

After current P0 closures, run the cheap offline chain in priority order: v12 pair, high-press,
then the four lateral candidates. Freeze task-specific papers only after their safe contact
manifolds are measured. Do not allocate RL GPUs to any item that has not passed the full offline
certificate chain.
