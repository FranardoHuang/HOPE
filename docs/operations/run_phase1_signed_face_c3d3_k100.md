# 运行 C3/D3 同卷 signed-face K100

本操作把已经终档的显式零摩擦对照 `C3` 和有符号拍面引导 `D3` 各自在同一张
[`K100`](../DEFINITIONS.md#q50-and-k100)（正手/反手各 50 次）上判一次。两条 checkpoint 必须先分别通过
generic [`checkpoint attestor`](../DEFINITIONS.md)，随后
[`paired execution consumer`](../DEFINITIONS.md) 才能启动现有 `judge.sh`。它只发布 paired 行为计数和
`D3-C3` 差值；不授权 L2、第二 seed、stop/promote、采用 setting、Gate3、部署或真机。

当前只完成 source/static gate，**本分支没有 SSH、attest 或 judge**。

## 1. 冻结输入

| 输入 | Exact binding |
| --- | --- |
| C3 checkpoint / hard / producer claim / terminal | `6b3e2cb1...70e7` / `d76dc944...ef2c` / `aa240e2f...a6d8` / `8c579386...e8ef` |
| D3 checkpoint / hard / producer claim / terminal | `44c6117c...85b8` / `98f6468f...34f4` / `7a1970d2...56d2` / `ccb9933c...7f0e` |
| paired L1 receipt | `bb3cd749477861b1cd55f059ed3b23307784030dcad758db3a819c3c8a37bbde` |
| schedule file / semantic / order | `f2777dcd...1ca` / `3ca4bdba...3365` / `09f778f2...bd0` |
| activation file / content | `e0125b0e...bb4` / `533beb03...3d8`；face signs 必须是 JSON floats `[1.0,-1.0]` |
| consumer manifest / runner | `cd1dfedc...f66088` / `f8515cf6...0a674` |

完整路径在
[`phase1_signed_face_c3d3_k100_execution_20260714.json`](../../configs/phase1_signed_face_c3d3_k100_execution_20260714.json)。
任何一项缺失、SHA/bytes/type 漂移或已有 partial output 都 fail closed；不得删目录后重判。

## 2. Source gate

```bash
SOURCE=/path/to/clean/independent/eval-worktree
cd "$SOURCE"
python3 scripts/run_phase1_signed_face_c3d3_k100.py static-validate
python3 scripts/run_phase1_signed_face_c3d3_k100.py source-plan
python3 -m pytest -q tests/test_run_phase1_signed_face_c3d3_k100.py
```

`source-plan` 明确输出 `runtime_request_bound=false`、`writes_or_launches_performed=false`，不是运行授权。

## 3. 两份 generic attestor exact request

在同一个 clean、独立 eval worktree 中分别建立 `C3_ATTEST_REQUEST` 与 `D3_ATTEST_REQUEST`。每份都必须按
[checkpoint attestor 操作](run_phase1_signed_face_k100_checkpoint_attestor.md)完整填写，尤其是：

- exact checkpoint path/bytes/SHA、iter `24`、schema `3`、lineage exact integer `1`；
- 相邻 `params/training_contract.json` path/bytes/SHA 与 plant canonical SHA；
- 从对应 terminal receipt 的 `training_launch_claim` 逐字段复制出的 standalone producer-claim JSON，file
  SHA 与 canonical SHA；不得把整个 `terminal_result.json` 冒充 claim；
- 此独立 eval worktree 的 exact commit/tree/clean；Isaac 与 MuJoCo Python resolved executable、SHA、完整
  fingerprint；exact MJCF path/bytes/SHA；
- output 必须分别为 generic attestor 的 checkpoint-SHA namespace。

先做无写入 plan：

```bash
python3 scripts/attest_phase1_signed_face_k100_checkpoint.py --repo-root "$SOURCE" \
  --request "$C3_ATTEST_REQUEST" plan
python3 scripts/attest_phase1_signed_face_k100_checkpoint.py --repo-root "$SOURCE" \
  --request "$D3_ATTEST_REQUEST" plan
```

两条一次性 runtime attestation 命令是：

```bash
python3 scripts/attest_phase1_signed_face_k100_checkpoint.py --repo-root "$SOURCE" \
  --request "$C3_ATTEST_REQUEST" attest
python3 scripts/attest_phase1_signed_face_k100_checkpoint.py --repo-root "$SOURCE" \
  --request "$D3_ATTEST_REQUEST" attest
```

任一失败即保留现场并停止；不得 attest 另一 checkpoint 后绕过缺失的一侧。

## 4. Paired execution request 与运行

建立一份绝对路径、非 symlink 的 `PAIR_REQUEST`，完整填写 consumer `load_request` schema：manifest
file bytes/SHA、两份 attestor request file/canonical SHA、两份 attestation claim/evidence file/content SHA、
两份 checkpoint-adjacent `env.yaml` bytes/SHA、同一 Isaac/MuJoCo venv activation bytes/SHA、distinct GPU
lane、Kit lock 和唯一 output root。任一 runtime binding 缺失都拒绝。

```bash
python3 scripts/run_phase1_signed_face_c3d3_k100.py --request "$PAIR_REQUEST" plan

python3 scripts/run_phase1_signed_face_c3d3_k100.py \
  --request "$PAIR_REQUEST" \
  --root-confirm ROOT_APPROVES_SIM_ONLY_C3_D3_SIGNED_K100_PAIRED_EXECUTION_V1 \
  execute
```

`execute` 先重放 generic attestor 的 checkpoint/hard/producer-claim/runtime/MJCF/paper 验证，再要求指定 GPU
没有 compute PID。它把 `env.yaml` 只读复制到独立 evaluation root，顺序运行 C3、D3；不写训练 run，
不发送任何 signal。每侧仍使用同一 schedule、seed `0`、noise `0.0`、`qdes-clamp`、one-question-reset 和
全 100 次无条件分母。最后的 `paired_behavior_result.json` 保留两侧 raw counts、SHA 和 `D3-C3`，但所有
训练/晋级授权仍为 false；L2 需要人类复核后另发 decision contract。
