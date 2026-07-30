# 三反手 upper baseline：已停用

这条 [`upper N3 safe warm-start`](../DEFINITIONS.md#upper-n3-safe) 已于 2026-07-28
作废并由 fresh exact N=5 action-conditioned ball-first 训练取代。

不要运行旧的 UpperSafe task、N3 launcher，不要 resume `model_10809.pt`，也不要把历史
`f5_upper_seed0` 成绩当作真机授权。旧探针在 Hydra/Gym 构造前即失败，没有训练 update、
checkpoint 或有效 Isaac safety smoke。

根因、三反手历史成绩和保留证据见
[EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728](../experiments/2026-07/EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md)。
当前可执行训练流程只从 [INDEX](../INDEX.md) 路由到 fresh N=5 ActionBall 实验与 G05。
