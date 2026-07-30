# ActionBall 发射前 Reward 因果审计

这一步不是看一眼 Reward 曲线，而是在与正式训练相同的 Hydra compose 后启动 **1 个真实 Isaac
环境**，逐个调用当前 `RewardManager` 的 active objective。工具见
`hope_training/whole_body_tracking/scripts/action_ball_reward_causal_prelaunch.py`。

## 它证明什么

- 四组账本固定为：MJLab 来源的平衡/稳定、BeyondMimic 来源的模仿、HOPE 自己的击球/上台、
  不可调安全项。
- 每个 active objective 必须有受控 baseline 和只恶化权威 causal axis 的 paired state；
  raw 值只能由 live Reward callable 返回，不能从 JSON 注入。
- `weight × raw × policy_dt` 的 worsening delta 必须严格小于零。只在自然 rollout 里见到非零
  不算因果证明。
- 某个新 term 没有经过复核的 transactional mutation 时，会留在 coverage 表里并令总结果
  `FAIL_CLOSED`；不能把没测到写成“无影响”。
- receipt 同时报 A（当前 composed baseline）和 B（只把拍位/拍速/拍面三项乘 4）的单位 raw
  预算。这只是候选剂量表，脚本不改训练配置，也不自动调权重。

baseline/worsened 会把门控 nuisance tensor 在两次调用中设成完全相同的受控值。两次之间唯一变化
是 taxonomy 写明的 causal axis。每个**显式受控的 Reward 输入 tensor**都在 probe 后逐字节恢复；
Reward callable 自己的诊断计数器可能前进，但它们不作为 raw term 的输入，且 receipt 写完即销毁
这个 1-env 审计环境。probe 过程中不执行 optimizer，也不推进 simulator。

## 前置

1. exact clean commit/checkout；不可从 dirty 工作树出正式 receipt。
2. 使用正式发射的全部 ActionBall identity、动作顺序、manifest、solver、policy contract、
   motion 文件和 expected effective Reward SHA。
3. `num_envs=1`，一个从未存在过的 `reward_causal_audit.output_dir`。工具用 no-clobber 创建目录，
   已存在就拒绝。
4. 与正式 run 相同的 editable source overlay 和 Isaac/PhysX 环境。

## 命令形状

把正式 launch 命令里的 `scripts/train.py` 换成下面脚本，保留 task/motion/manifest/receipt/SHA
等全部 Hydra override，并额外设置：

```bash
python3 scripts/action_ball_reward_causal_prelaunch.py \
  task=HOPEPingPongActionBall \
  num_envs=1 \
  device=cuda:0 \
  ...与正式训练完全相同的 ActionBall override... \
  ++reward_causal_audit.output_dir=/workspace/runs/<new-no-clobber-name>/reward_causal
```

成功时写：

```text
<output_dir>/receipt.json
```

`report.all_active_objectives_causal`、每组
`groups.<group>.all_active_objectives_causal` 和所有 `coverage[*].status` 都必须通过。脚本即使
失败也尽量保留 fail-closed receipt，便于知道缺的是 mutation 覆盖、runtime shape、错误 callable，
还是方向相反。

## 剂量怎么读

当前 adopted A 仍是：

- 拍位/拍速/拍面 `4 / 0.5 / 0.5`；
- 上台 `1648.8`，每拍理论满额 `1648.8 × 0.02 = 32.976`；
- generic unsafe death `-3600`，一次非 timeout 终止 `-72`；
- q_des 与 actual-q 软限位各 `-40`。

B 只把三项 tracking 变为 `16 / 2 / 2`。它只能在 A 主跑稳定后占用不影响主吞吐的独立卡做
小预算 canary；receipt 中出现 B 不代表已采用。判断比例时同时看：

1. controlled causal delta 是否方向正确；
2. 正式 activation ledger 的逐组 signed contribution 分位数；
3. 模仿误差、长期 physical return/landing 与 unsafe/table/fall；
4. 不能因为 landing 是 one-shot，就拿单步均值直接和 dense 模仿项比较；上台项按每拍，dense
   项按每控制步/每 cycle 积分后再比。

`racket_progress` 另行分账。它是有符号的
`prev_racket_distance - current_racket_distance`，每控制步 raw 夹在 `[-0.15, 0.15] m`，并在
reset/resample 步严格清零。当前 weight 10、`policy_dt=0.02 s` 时：

- “raw=1”的单位表是 `+0.2/step`，只用于权重维度对照，**不是可实现的持续收入**；
- callable 的绝对单步上界是 `10 × 0.02 × 0.15 = 0.03`；
- 正式运行必须按 swing 报 signed telescoping 累计，并同时给
  `eligible_prestrike_control_step_count`、`eligible_swing_count`、
  `reset_or_resample_zeroed_step_count`、正/负 raw sum 和每-swing p5/p50/p95。prelaunch 没有
  rollout，receipt 对这些 empirical 字段诚实写 `null/required_from_training_activation_ledger`，
  不伪造测量值。

## 测试

Host 只验证 receipt/no-clobber/候选剂量/coverage fail-closed。普通 Python 3 无 Torch 时仍运行
dependency-light 合同测试，并跳过两条真实 tensor mutation 语义回归；全量 focused 需安装 Torch：

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_action_ball_reward_causal_prelaunch.py
```

正式证据必须在 Pod 的真实 Isaac 环境运行本页 1-env 命令；host fake 或自然 nonzero 都不得冒充。
