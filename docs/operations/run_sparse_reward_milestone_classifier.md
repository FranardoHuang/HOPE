# 运行稀疏 Reward milestone classifier

本操作只消费已经物化的累计计数 JSON，并新增一个 no-clobber receipt。它没有 SSH、process signal、
queue mutation、checkpoint execution、judge、simulator 或真机入口。接口字段见
[稀疏 Reward 资格账本](../interfaces/sparse_reward_eligibility_ledger.md)。

## 前置条件

1. trainer source 必须包含 per-update sparse ledger；旧 source 的 EMA `virtual_*_rate` 不能反推整数分母，
   也不得补写。2026-07-15 已在跑的 `2c2d70d` 三格不具备本 emitter。
2. sidecar 从同一 TensorBoard event stream 累加 `0..milestone` 的所有 ledger tag；重复 step、缺 tag、
   窗口不完整或动作族不全必须拒绝物化 measurement。
3. measurement 必须绑定 checkpoint bytes、training claim、source commit 与冻结 run contract。

## 命令

一个 milestone 只能得到方向资格：

```bash
python3 scripts/classify_sparse_reward_milestones.py \
  --contract configs/phase1_sparse_reward_eligibility_contract_20260715.yaml \
  --measurement /path/to/model_200.sparse-ledger.json \
  --output /path/to/model_200.sparse-classification.receipt.json
```

连续两个 milestone 一起读取；只有两窗都完整才会写 `DECISION_ELIGIBLE`：

```bash
python3 scripts/classify_sparse_reward_milestones.py \
  --contract configs/phase1_sparse_reward_eligibility_contract_20260715.yaml \
  --measurement /path/to/model_200.sparse-ledger.json \
  --measurement /path/to/model_500.sparse-ledger.json \
  --output /path/to/model_500.sparse-classification.receipt.json
```

目标 receipt 已存在时命令必须失败，绝不覆盖。所有状态的
`automatic_trainer_action` 都是 `CONTINUE_UNCHANGED`；收到 `MEASUREMENT_INVALID` 时修复未来仪表/新
namespace，不重写原计数，也不把它当成 setting 负结果。

## 本地源码验证

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  tests/test_classify_sparse_reward_milestones.py

/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py \
  -k 'qdot_limit_hinge or sparse_virtual_reward_ledger'

/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py \
  -k qdot_limit_hinge
```

这些测试是 E1 source evidence；没有证明真实 Isaac logger、Pod milestone sidecar 或 Phase-B physical
contact 已运行。
