# Experiment Records

This directory is the decision ledger for individual hypotheses and runs. It complements, but does
not replace, immutable machine contracts in `../../configs/`, gate acceptance documents, operation
commands or the main-only current queue in [NOW.md](../NOW.md).

## Required Record

Create or update one Markdown record before an experiment is launched. Use a stable descriptive
name such as `phase1_forehand_signed_face_forensic_20260713.md`. At minimum record:

- question and decision scope;
- human owner, executor and branch;
- current status: `Design`, `Preregistered`, `Running`, `Complete`, `Failed` or `Blocked`;
- exact source/config/checkpoint/asset/schedule inputs and hashes, or explicit missing bindings;
- independent and dependent variables, fixed controls and seed/blocking plan;
- denominator, censoring/reset policy, thresholds and decision rule frozen before results;
- reproducible validation/run commands and output locations;
- result, limitations, rejected claims and next action.

If an experiment has no immutable config yet, keep it `Design` or `Blocked`. A record never grants
permission to run a simulator, signal a process, deploy or command hardware. Those permissions must
come from the relevant operation and gate contracts.

## Template

```markdown
# <experiment name>

Status: Design
Human owner: <person>
Executor: <person or agent>
Branch: <branch>

## Question and decision scope

## Inputs and immutable bindings

## Design and controls

## Acceptance and failure rules

## Reproduction

## Results

## Limitations and claims not made

## Decision and next action
```

Keep raw heavy outputs under the documented ignored artifact root. Commit only small contracts,
summaries and evidence needed to reproduce the decision.
