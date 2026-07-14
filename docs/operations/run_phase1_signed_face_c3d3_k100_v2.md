# 运行 C3/D3 同卷 signed-face K100 v2

状态：**v2 asset source gate 已通过；asset packaging blocker 已关闭，但 exact K100 仍被
velocity-limit parity blocker 阻断。** 本操作只修复 v1 的 ignored Isaac 资产打包缺口。`C3` 是显式
零摩擦对照，`D3` 是只增加拍面引导的匹配臂；两者仍只允许在同一张
[`K100`](../DEFINITIONS.md#q50-and-k100)（正手/反手各 50 次、失败不删）上各判一次。

v1 已在 C3 导出 ONNX 前自然失败并永久冻结：

- output root：
  `/workspace/codexschema/phase1_signed_face_c3d3_l1_20260714/evaluations/signed_face_k100_c3d3_v1`；
- `C3/exit.json` 记录 `rc=1`、无 signal，`judge.runner.log` SHA-256
  `a27b9538fa7ce887ea21ca756f67ee4d54652d1e1cdbc452b9c9b69161c31d0b`；
- 致命异常是 v1 eval checkout 缺少 ignored
  `assets/agibot_a3/urdf/model.urdf`，尚未导出 ONNX、进入 MuJoCo 或产生 K100 行为成绩。

不得删除、改名、补写或重放 v1 output/attestation，也不得向 v1 eval checkout 补资产后绕过。

后续保留的 diagnostic 已用同一训练 A3 asset closure 和 exact checkpoint/bank/plant binding 分别运行
C3/D3：两侧都成功导出 ONNX 并进入 MuJoCo，证明缺资产不再是当前 blocker。该运行显式传入
`--allow-inexact-contract`，所以日志中的 `evaluation_contract_exact=false` 是预期且必须保留；但旧接口
仍让 `formal_execution_contract_ok=true`，两侧因而都在第 0 题开始前同样 fail closed：

```text
formal BankExam reached bound PhysX joint-velocity limit on articulation indices [8]; MuJoCo lacks same braking constraint
```

因此两边都是 `scheduled=50/side`、`asked=0`，没有任何 K100 attempt/score，方向分不存在。原诊断
PID `1873348/1873349` 已退出；只读证据保存在
`.../evaluations/diagnostic_asset_hydrated_inexact_v1/{C3,D3}/judge.runner.log`。当前 blocker 是
articulation index `8` 的 PhysX velocity-limit
braking 与 MuJoCo 执行语义不等价；不是 C3/D3 行为差异。该 parity 合同修复并经新 source binding
复核前，只允许本页 source/asset/plan 检查，**不得执行第 4 节 paired exact judge**。明确
`allow-inexact` 的方向筛只可作诊断，不能变成 formal K100、L2、第二 seed 或 promote 证据。若要让
这种方向筛越过 velocity guard，必须另外显式传入 `--allow-velocity-limit-proxy`；该 flag 只有在
`--target-source bank --allow-inexact-contract` 同时存在时才合法，并继续把
`formal_execution_contract_ok/evaluation_contract_exact` 固定为 false；单独
`--allow-inexact-contract` 不得静默改变 velocity-limit 语义。

## 1. v2 修复合同

v2 使用新的 [`checkpoint attestor`](../DEFINITIONS.md) namespace
`.../executions/signed_face_k100_v2/<checkpoint-sha>/` 和新的 paired output
`.../evaluations/signed_face_k100_c3d3_v2/`。它在任何 evidence/claim/judge 前执行：

1. 递归枚举 C3/D3 训练时实际 checkout 内的 ignored `agibot_a3` 目录；canonical inventory（按相对路径
   排序的 path/bytes/SHA-256 清单摘要）必须与 request 完全相等，且
   `urdf/model.urdf` 必须是 43,240 bytes、SHA-256
   `79655f05d204c24f028778425aa971410773d1f8bbbd214de6fdb8f8ae75d1cc`。
2. `libGLU.so.1` 只做 `ctypes.CDLL` 可加载性预检；本合同不扩展成系统依赖审计。
3. C3 request 的 `hydration_mode=hydrate_absent`：先在 eval checkout 的 ignored assets 父目录完成并复核
   staged copy，再原子独占 destination root；每个子目录用 `mkdir(exist_ok=false)`，每个已 `fsync` 的
   regular file 用同文件系统 `link(2)` 原子 no-replace 发布。禁止用 `rename(2)` 覆盖/合并并发同名目标；
   任何 sentinel、partial destination 或 stage 都原样保留并阻断重试。
4. D3 request 的 `hydration_mode=verify_existing`：只能复核 C3 已发布的同一 destination/inventory。
5. paired consumer 完整重放两份 v2 evidence/claim，并要求两侧共享同一 asset inventory、destination、
   MJCF、plant 和可加载的 `libGLU.so.1`。

资产来源固定为：

```text
/workspace/codexschema/nohope_signed_face_c3d3_l1_4467d79/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3
```

这就是两条 checkpoint 的训练 checkout；不得换成相似仓库、symlink 或后来重建的未绑定目录。

当前 source gate 的五份精确字节为：

| 文件 | SHA-256 |
| --- | --- |
| `hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py` | `343fce91c1358d34764c754832261798e1e94490cb479003cfe2fff1523cd714` |
| `scripts/attest_phase1_signed_face_k100_checkpoint_v2.py` | `4792d7e279042abfdc0f130ab9bca3006cb300f32f8c76e4cc069c5ec4c0cb5a` |
| `configs/phase1_signed_face_k100_checkpoint_attestor_v2_20260714.json` | `9bc2be01224ffd75c1fdaf46a5c9945a8291bf94398104be1c6480eb16e40097` |
| `scripts/run_phase1_signed_face_c3d3_k100_v2.py` | `6cb8c6936f7aa19b6e6553e5709bfb61564a7a4ed7d5385b48960db3347c1744` |
| `configs/phase1_signed_face_c3d3_k100_execution_v2_20260714.json` | `28c6b8cfa207c24e210013d6473a570f8b8038157b02b14f285b09af882c1654` |

级联 source binding 与并发 sentinel 攻击回归已通过：focused `57 passed`，attestor/pair
`static-validate` 及 pair `source-plan` 均 rc0。这里只证明 source/发布合同，不代表新 diagnostic 已运行。

## 2. Source gate 与新 eval checkout

`SOURCE` 必须是合入 v2 后建立的全新 clean detached checkout；目标
`$SOURCE/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3`
在 C3 attest 前必须 absent。不要复用 v1 checkout。

```bash
SOURCE=/workspace/codexschema/nohope_eval_c3d3_k100_v2_<main-short-sha>
cd "$SOURCE"

python3 scripts/attest_phase1_signed_face_k100_checkpoint_v2.py static-validate
python3 scripts/run_phase1_signed_face_c3d3_k100_v2.py static-validate
python3 scripts/run_phase1_signed_face_c3d3_k100_v2.py source-plan
python3 -m pytest -q \
  tests/test_attest_phase1_signed_face_k100_checkpoint_v2.py \
  tests/test_run_phase1_signed_face_c3d3_k100_v2.py

python3 scripts/attest_phase1_signed_face_k100_checkpoint_v2.py asset-plan
```

`asset-plan` 必须输出 `training_asset_inventory_valid_no_writes`、精确 inventory、required URDF 和
`libglu.loadable=true`。若 `libGLU.so.1` 不可加载，先由 Pod 管理者补齐系统运行库，再重新执行纯只读
`asset-plan`；不得用 attestation 或 judge 试探依赖。

## 3. 两份 v2 attestor request

沿用 v1 request 的 checkpoint/hard/producer-claim/Python/MJCF/paper 精确绑定，但必须改用 v2 manifest、
v2 output root 和下列完全相同的 `isaac_asset_bundle`：

```json
{
  "source_root": "/workspace/codexschema/nohope_signed_face_c3d3_l1_4467d79/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
  "destination_root": "<SOURCE>/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3",
  "inventory": "<逐字段复制 asset-plan 输出>",
  "required_urdf": {
    "relative_path": "urdf/model.urdf",
    "bytes": 43240,
    "sha256": "79655f05d204c24f028778425aa971410773d1f8bbbd214de6fdb8f8ae75d1cc"
  },
  "hydration_mode": "<C3=hydrate_absent; D3=verify_existing>"
}
```

先只读 plan：

```bash
C3_ATTEST_REQUEST=/absolute/no-clobber/request-root/C3.v2.attestor_request.json
D3_ATTEST_REQUEST=/absolute/no-clobber/request-root/D3.v2.attestor_request.json

python3 scripts/attest_phase1_signed_face_k100_checkpoint_v2.py \
  --repo-root "$SOURCE" --request "$C3_ATTEST_REQUEST" plan
python3 scripts/attest_phase1_signed_face_k100_checkpoint_v2.py \
  --repo-root "$SOURCE" --request "$D3_ATTEST_REQUEST" plan
```

确认 C3 destination 与两个 v2 checkpoint-SHA output 均 absent 后，严格按 C3→D3 顺序各消费一次：

```bash
python3 scripts/attest_phase1_signed_face_k100_checkpoint_v2.py \
  --repo-root "$SOURCE" --request "$C3_ATTEST_REQUEST" attest
python3 scripts/attest_phase1_signed_face_k100_checkpoint_v2.py \
  --repo-root "$SOURCE" --request "$D3_ATTEST_REQUEST" attest
```

任一 SSH timeout 都是 `UNKNOWN`：先只读检查固定 evidence/claim/destination，绝不重放写命令。

## 4. Paired v2 request 与执行

当前本节 **BLOCKED / NO-LAUNCH**：articulation `[8]` velocity-limit parity 尚未闭合。以下命令保留为
阻塞解除后的合同真源，不是当前运行授权。

`PAIR_REQUEST` 必须绑定 v2 manifest file bytes/SHA、上述两份 request、两份实际 v2
evidence/claim file/content SHA、两份 checkpoint-adjacent `env.yaml`、两个 runtime activation、distinct
GPU lane 和 v2 output root。

```bash
PAIR_REQUEST=/absolute/no-clobber/request-root/PAIR_REQUEST.v2.json

python3 scripts/run_phase1_signed_face_c3d3_k100_v2.py \
  --repo-root "$SOURCE" --request "$PAIR_REQUEST" plan

python3 scripts/run_phase1_signed_face_c3d3_k100_v2.py \
  --repo-root "$SOURCE" --request "$PAIR_REQUEST" \
  --root-confirm ROOT_APPROVES_SIM_ONLY_C3_D3_SIGNED_K100_PAIRED_EXECUTION_V2 \
  execute
```

`execute` 不训练、不写原训练 run、不发 signal，也不授权 L2、第二 seed、stop/promote、采用 setting、
Gate3、部署或真机。只有 `paired_behavior_result.json` 完成并通过 consumer 复核后，才存在可供人类
后续决策的行为计数。
