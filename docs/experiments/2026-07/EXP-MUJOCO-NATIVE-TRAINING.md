# EXP-MUJOCO-NATIVE-TRAINING — 原生微调能否减少迁移损失？

- 状态：blocked
- 实现状态：preflight 代码已在 off-main 分支，trainer 未实现；全局优先级只看 NOW
- 阶段/轴：课程阶段 1 及后续 / 候选训练引擎
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：E1

共享缩写见 [术语与人话对照](../../DEFINITIONS.md)。

假设：用同一个 exact checkpoint 初始化的批量原生 MuJoCo `rsl_rl VecEnv`，可以改善留出 MuJoCo 的击球状态/平衡，
同时不损害厂商 Gate3 稳定性。

预检收窄了 `Trainer-v0` 的范围（旧记录曾写为 D0）：当前厂商 `main` 的仿真回路没有球/球桌/球网，
因此 `Trainer-v0` 不能声称物理回球或连续对打。对每种有效 plant，它必须独立复现 reset observation 和固定 action tape，
使用独立的 reward-replay oracle，并且只把 actor/distribution/冻结 actor normalizer 加载到全新的 critic 和 optimizer。
训练器与评估器不得共用正在被检验的实现。

## 当前实现边界（2026-07-13）

- off-main 候选 `origin/codex/mujoco-training-preflight@6e5fce3` 已实现 MJCF closure、2 s
  parity trace 合同，以及 source/objective/warm-start/physics trust boundary；63 项
  focused test 和顶层 `468 passed, 9 skipped` 通过。
- 这些测试的作用是防假绿，不是授权训练。该候选保留 11 个 blocker，
  `physics/objective/loader/isolated-producer` 四个 trusted 位和 `smoke/formal/long`
  三个授权位全部为 false。
- binary contact/net/landing 只能写 diagnostic，不得冒充 Isaac dense
  net-height/landing-error/spin reward。候选当前明确为 `NO-MERGE`，不只是“等待普通 review”。
- 四个 red-team P1 正确性缺口仍开：action tape/trace 未覆盖 clamp 与 runtime adapter；静态
  source closure 可被 alias/`exec` 绕过；JSON 接受 duplicate key/NaN；MJCF `compiler strippath`
  解析错误。必须先补负测并修正，不能用“授权位全 false”掩盖源码错误。

`Trainer-v0` 首卷仍是：冻结 actor 对照与等预算 warm-start 微调，至少两个 seed，
并使用不可变的留出考卷。目前不存在 trusted isolated runner、single-env core、
VecEnv、PPO 冒烟、微调或结果。

下一门：先关闭四个 red-team P1 并重新 review/merge；再实现禁止 child-process escape、记录
runtime module closure 的 isolated runner 与 `vendor_gate3_v1` single-env core，然后做 evaluator
action tape 和独立 same-shape reward oracle。single-env core 必须用 N=1/8/32/64 报告 sim-only
及完整 rollout+一次 PPO update 的吞吐、内存、CPU 和扩展效率；只有两臂×两 seed 预计能在
48 小时内完成且留 30% 余量，才继续 CPU-Python 长训，否则转 C++/OpenMP 或另行过 parity 门的
MJX/MJWarp。该路线不阻塞 `Gate3-D0`；吞吐继续门内只允许预注册、限预算的单次 PPO update
smoke，除此之外在这些门通过前不得启动 PPO 长训。

权威资料：[v0 预检](../../research/mujoco_training_v0_preflight_2026-07-12.md) 和
[G06](../../gates/G06_isaac_to_mujoco.md)。
