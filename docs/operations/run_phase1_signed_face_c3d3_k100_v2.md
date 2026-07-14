# 运行 C3/D3 同卷 signed-face K100 v2

状态：**v2 source gate 已通过；尚未在 Pod 上 attest 或 judge。** 本操作只修复 v1 的 ignored Isaac
资产打包缺口。`C3` 是显式零摩擦对照，`D3` 是只增加拍面引导的匹配臂；两者仍只允许在同一张
[`K100`](../DEFINITIONS.md#q50-and-k100)（正手/反手各 50 次、失败不删）上各判一次。

v1 已在 C3 导出 ONNX 前自然失败并永久冻结：

- output root：
  `/workspace/codexschema/phase1_signed_face_c3d3_l1_20260714/evaluations/signed_face_k100_c3d3_v1`；
- `C3/exit.json` 记录 `rc=1`、无 signal，`judge.runner.log` SHA-256
  `a27b9538fa7ce887ea21ca756f67ee4d54652d1e1cdbc452b9c9b69161c31d0b`；
- 致命异常是 v1 eval checkout 缺少 ignored
  `assets/agibot_a3/urdf/model.urdf`，尚未导出 ONNX、进入 MuJoCo 或产生 K100 行为成绩。

不得删除、改名、补写或重放 v1 output/attestation，也不得向 v1 eval checkout 补资产后绕过。

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
   staged copy，再用 exclusive destination directory 发布；任何 partial destination 都保留并阻断重试。
4. D3 request 的 `hydration_mode=verify_existing`：只能复核 C3 已发布的同一 destination/inventory。
5. paired consumer 完整重放两份 v2 evidence/claim，并要求两侧共享同一 asset inventory、destination、
   MJCF、plant 和可加载的 `libGLU.so.1`。

资产来源固定为：

```text
/workspace/codexschema/nohope_signed_face_c3d3_l1_4467d79/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3
```

这就是两条 checkpoint 的训练 checkout；不得换成相似仓库、symlink 或后来重建的未绑定目录。

合入前 source gate 的四份精确字节为：

| 文件 | SHA-256 |
| --- | --- |
| `scripts/attest_phase1_signed_face_k100_checkpoint_v2.py` | `45d35083d8a02b53e30b875839d0b004306511f6cafcbc6c36753c488f4d024a` |
| `configs/phase1_signed_face_k100_checkpoint_attestor_v2_20260714.json` | `7bfa3b843de15b2c107c8254b04a585d78ac2c029c478d7294dc16a13e3c3a93` |
| `scripts/run_phase1_signed_face_c3d3_k100_v2.py` | `61338a382418e941413af58128a38c0d6530853b4bac5a406836136dd9a9befa` |
| `configs/phase1_signed_face_c3d3_k100_execution_v2_20260714.json` | `c6c8db751b60015b946889e41966e637daaa935c7e8753f498365d4fd67263e8` |

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
