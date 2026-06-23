# Agent Rules For This Repo

Read `docs/START_HERE.md` before making changes.

## Documentation Is Part Of The Work

Whenever an agent changes code, moves files, adds assets, changes goals, discovers a blocker, or verifies a gate, update the documentation in the same branch.

Required updates:

- Update the affected `docs/gates/G*.md`.
- Update `docs/PROGRESS.md` with a short dated entry.
- Update `docs/PROJECT_MAP.md` if folder roles change.
- Update `docs/ASSET_POLICY.md` if git, ignored assets, LFS, or external repo policy changes.
- Update files under `docs/interfaces/` when frames, ROS topics, messages, joint order, observations, actions, or runtime contracts change.
- Update files under `docs/operations/` when setup, build, test, training, or deployment commands change.

Do not rely on chat history as the only place where project state is recorded.

## Gate Discipline

Do not mark a gate `Done` unless the gate document has reproducible commands, verification results, known limitations, and current inputs/outputs.

Use `Partial` when materials or code exist but verification is incomplete.

## Asset Discipline

Keep source and small configs in git. Keep heavy runtime artifacts under ignored local roots such as `vendor_assets/` unless the team explicitly adopts Git LFS or another artifact system.

Do not delete local-only assets unless they have been moved, backed up, or explicitly declared obsolete.

If a task requires an ignored file or folder, update `docs/operations/setup_local_sync.md` with the manual restore path and update the relevant gate doc with the dependency. Do not assume ignored files exist on another machine.

## Safety

Do not run real robot command tests unless the relevant gate and operation docs say the dry-run, joint order, command scaling, and safe halt checks have passed.
