# 运行 A2/B2 跨 Pod 热启动 signed-face L1

状态：**runtime source ready；NOT LAUNCHED。** `A2/B2` 是同一个父 checkpoint 的两条探索性
L1（`512 env × 25 update`）：A2 为 guidance `0.0` 对照，B2 只把 signed-face guidance 改为
`-0.4`。术语见[定义](../DEFINITIONS.md)。本入口没有 judge、L2、第二 seed、晋级、部署或真机模式。

冻结文件：

- manifest：[`phase1_signed_face_a2b2_l1_prereg_20260714.json`](../../configs/phase1_signed_face_a2b2_l1_prereg_20260714.json)，SHA `890cb8f7c1176c0e8a5e3102eb40c80d02ffcb4f6ed355465a006d875806c659`；
- launcher：[`run_phase1_signed_face_a2b2_l1.py`](../../scripts/run_phase1_signed_face_a2b2_l1.py)，SHA `35d730facd6d5b2387a923232e62b47589b33b420e5d51d9489700c93f49b63e`；
- training source：commit `4467d79f1ed425a4263f0caaad2f661e1ec737ad`，tree
  `497db1d8f2d7fb1b554337928f098a2951d4cf0d`；
- external control root：`/workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2`；
- run root：`/workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/runs/l1`。

## Source gate

```bash
python3 scripts/run_phase1_signed_face_a2b2_l1.py --mode static-validate
python3 scripts/run_phase1_signed_face_a2b2_l1.py --mode plan
pytest -q tests/test_run_phase1_signed_face_a2b2_l1.py
```

当前结果：`21 passed`，`py_compile`、`static-validate`、plan 均 rc0。运行消费者只重复检查会改变本轮
科学/安全结论的事实：clean commit/tree、父 checkpoint bytes/finite/embedded contract、父→子核心
hard-contract diff、输入 SHA、零摩擦 argv/runtime marker/31 项 hard contract、全新 run namespace 和空 GPU。

## 一次性运行

两台 Pod 都必须先把 manifest/launcher 以以上 SHA 安装到 exact external control root 并设为只读；exact
training checkout 必须位于
`/workspace/codexschema/nohope_signed_face_a2b2_hot_l1_v2_4467d79`，恢复被忽略 A3 asset 目录。父
`model_13800.pt` 与相邻 hard contract 必须在 manifest 的 exact path/SHA 上存在。claim 前目标 GPU0
必须没有任何 compute process，artifact arm 与 exact trainer run name 必须都不存在。

Pod1 GPU0 只运行 A2：

```bash
python3 /workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2/run_phase1_signed_face_a2b2_l1.py --mode validate-runtime --cell A2 --host pod1
python3 /workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2/run_phase1_signed_face_a2b2_l1.py --mode launch-one --cell A2 --host pod1 --root-confirm ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_A2_B2_HOT_EXPLICIT_ZERO_FRICTION_L1_CROSS_POD_V2
```

Pod2 GPU0 只运行 B2：

```bash
python3 /workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2/run_phase1_signed_face_a2b2_l1.py --mode validate-runtime --cell B2 --host pod2
python3 /workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2/run_phase1_signed_face_a2b2_l1.py --mode launch-one --cell B2 --host pod2 --root-confirm ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_A2_B2_HOT_EXPLICIT_ZERO_FRICTION_L1_CROSS_POD_V2
```

每次 `validate-runtime` 后必须重新确认本条命令 rc0，再执行紧邻的唯一 `launch-one`。launcher 创建 atomic
claim 后等待 locked wrapper 看见 hard-contract marker 即返回；trainer 自行继续。任何失败都保留
claim/log/failure，不得删目录、换名重放或自动 retry。不得用 `pkill/killall` 或 broad signal。

自然退出且 GPU0 为空后，各 Pod 只 finalise 自己的格：

```bash
python3 /workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2/run_phase1_signed_face_a2b2_l1.py --mode finalize-cell --cell A2 --host pod1
python3 /workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2/run_phase1_signed_face_a2b2_l1.py --mode finalize-cell --cell B2 --host pod2
```

终档必须是 finite `model_13824.pt`、embedded iter `13824`、lineage `0`，并绑定相邻 hard-contract SHA
和 outer claim。两条结果仍只证明探索 L1 provenance；后续判卷/第二 seed 必须另过门。

跨 Pod pair 只在 Pod1 汇总：把两格各自只读的 `terminal_result.json` 和它绑定的
`params/training_contract.json` 复制到 exact `pair_inputs/v1/{A2,B2}/`，hard contract 统一命名为
`training_contract.json`，四文件均设为只读。随后运行：

```bash
python3 /workspace/codexschema/phase1_signed_face_a2b2_hot_l1_v2_20260714/control/v2/run_phase1_signed_face_a2b2_l1.py --mode finalize-pair --host pod1
```

finalizer 会用 Pod1 GPU0 UUID 验明实际机器，重放两份 terminal claim/checkpoint audit，并完整比较两份 hard
contract；包括所有 current-only 值在内只能有 `racket_guidance_reward.signed_face.weight` 一处差异。
