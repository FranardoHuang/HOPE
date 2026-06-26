# G06 Isaac-To-MuJoCo Parity

Status: Not started

## Goal

Test whether a policy or trajectory behavior learned in Isaac can be replayed or approximated in MuJoCo.

This gate is the sim-to-sim bridge before real deployment.

## Inputs

- Isaac-trained policy from G05.
- MuJoCo A3 model from G04.
- Shared joint order and observation/action contract.

## Outputs

- Replay/evaluation script or procedure.
- Cross-sim metrics.
- Known mismatch list.
- Decision on whether MuJoCo is only a deployment dry-run or also a training backend.

## Related Directories

- `agi/A3_MuJoCo_Sim/`
- `agi/code_deployment/a3_deploy_example/`
- `hope_training/whole_body_tracking`
- `docs/interfaces/policy_observation_action.md`

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)

## Acceptance Criteria

- The same action ordering is verified in both simulators.
- A simple standing or tracking behavior can be compared.
- Divergence sources are documented: contact, latency, actuator, timestep, observation delay, model mismatch.

## Current State

Done:

- MuJoCo and deploy support materials exist.

Not done:

- No Isaac policy exists yet.
- No parity procedure exists yet.

## Risks

- A policy can appear valid in Isaac but fail in MuJoCo because of actuator/contact mismatch.
- Debugging sim-to-real without sim-to-sim parity can waste hardware time.

## Next Steps

1. Wait for G05 first loop.
2. Define the minimal replay target: standing, swing reference, or racket target tracking.
3. Add parity metrics before tuning.
