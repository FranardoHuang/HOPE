# 为一个 checkpoint 冻结 signed-face K100 执行证据（不启动判卷）

本操作运行 [`signed-face K100 checkpoint attestor`](../DEFINITIONS.md)：它把一份明确指定的
checkpoint 与相邻 hard contract、fresh lineage、producer claim、评测源码与 Python runtime、MJCF/plant、
以及已经物化的 exact [`K100`](../DEFINITIONS.md#q50-and-k100) schedule/activation 一次性对齐。它只写
`evidence + claim`，不导出 ONNX、不启动 judge/Isaac/MuJoCo/trainer，不给 checkpoint 做停止或晋级决定，
也没有部署或真机入口。

当前状态是 source/static gate 通过，**没有任何 runtime request 被 attest**。每个候选必须另写一份经过
review 的 exact request；本仓库不提供带通配符或猜“最新 checkpoint”的可执行模板。

## 1. 冻结源码

```bash
SOURCE=/path/to/clean/detached/nohope
MANIFEST="$SOURCE/configs/phase1_signed_face_k100_checkpoint_attestor_20260714.json"
CONSUMER="$SOURCE/scripts/attest_phase1_signed_face_k100_checkpoint.py"
CORRECTION="$SOURCE/configs/phase1_signed_face_exam_k100_runtime_receipt_correction_20260714.json"

MANIFEST_SHA=0c9fa56ef44b7a2e4d3e7b3c661df195ed3f6cbfd5aa0e03b79e344f604bd4f5
CONSUMER_SHA=b42ecd08ea1516ab50cda0d47ee957b82e7c1e5b0c19fc3f7f588862ed7c5ec3
CORRECTION_SHA=4ef8b8d868e1e17733710e525e1bffb3ed4ed8c6013b7b834bfd63571e0c30ca

test -z "$(git -C "$SOURCE" status --porcelain)"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA"
test "$(sha256sum "$CONSUMER" | awk '{print $1}')" = "$CONSUMER_SHA"
test "$(sha256sum "$CORRECTION" | awk '{print $1}')" = "$CORRECTION_SHA"
python3 "$CONSUMER" --repo-root "$SOURCE" --manifest "$MANIFEST" static-validate
```

manifest 同时绑定 `judge.sh`、`mujoco_eval_onnx.py`、signed-face scorer 和 schedule module 的源码 SHA；
`attest` 时还要求 request 指定的 checkout commit/tree 与 clean 状态逐项匹配。换 evaluator 或 runner 必须发
新 manifest/version，不能沿用旧 request。

## 2. 旧 runtime receipt 的显式 correction

原文件
[`phase1_signed_face_exam_k100_runtime_receipt_20260714.json`](../../configs/phase1_signed_face_exam_k100_runtime_receipt_20260714.json)
保持 bytes/SHA `c0eca638...2048` 不变。它的摘要把
`signed_face_contract.mount_normal_sign_per_clip` 写成 JSON integers `[1,-1]`；actual source manifest、
materializer 和已写 activation 的合同都是 exact floats `[1.0,-1.0]`。这不是可以用“数值相等”吞掉的差别。

versioned
[`correction pointer`](../../configs/phase1_signed_face_exam_k100_runtime_receipt_correction_20260714.json)
记录旧值、正确值和 actual activation file/content SHA。consumer 不读取旧 receipt 的 sign 数值作为权威；
它必须直接读 actual activation bytes，重算 `content_sha256`，并严格拒绝 integer/bool 替代 float。旧 receipt
仍可用于其余已明确绑定的 file hash 取证，但不能再给 sign 字段定类型。

## 3. 每个 checkpoint 的 exact request

request 顶层必须完整给出以下独立事实，不能省略、glob、猜目录或自动挑最大迭代：

- checkpoint 绝对路径、bytes、SHA、`model_<N>.pt` filename iteration 和预期 embedded iteration；两者必须
  是同一个 plain integer；
- 相邻且唯一的 `<checkpoint-dir>/params/training_contract.json` path/bytes/SHA；checkpoint 内嵌 contract
  SHA 必须一致；
- `training_contract_schema_version=3`、fresh lineage exact integer `1`；bool `true` 不作等价值；
- producer claim 的绝对路径、file SHA 和 canonical JSON SHA；checkpoint 内嵌
  `training_launch_claim_sha256` 必须等于后者；
- source checkout 的绝对路径、exact commit/tree、`clean_required=true`；
- checkpoint Python 与 evaluator Python 的 launcher path、resolved executable path/SHA、exact
  `sys.version` 和固定 package closure/version；
- MJCF path/bytes/SHA；hard contract 中固定 plant 字段的 canonical SHA；
- 唯一 output root：
  `/workspace/codexschema/phase1_signed_face_rescue_20260713/executions/signed_face_k100_v1/<checkpoint_sha256>`；
- 所有授权字段保持 attestation-only：trainer/judge/checkpoint execution/L2/第二 seed/stop-promote/formal
  score/Gate3/deploy/real robot 全 false。

hard contract 还必须逐类型满足：`deploy_parity_face179`/179D、`shared_plus_y`、exact float
`[1.0,-1.0]`、motion kinematics exact 且禁止 legacy link-origin velocity。plant SHA 覆盖 31-joint order、
action ids、PD、armature、effort/velocity limits、friction semantics、qdes clamp 和 dt/decimation；不能只绑
MJCF 文件名。

## 4. no-write plan

```bash
REQUEST=/absolute/path/to/reviewed_exact_request.json
python3 "$CONSUMER" \
  --repo-root "$SOURCE" \
  --manifest "$MANIFEST" \
  --request "$REQUEST" \
  plan
```

`plan` 只检查 request 的封闭 schema、类型、filename/iteration、adjacency 和唯一 namespace，不读取
checkpoint/private paper/MJCF，也不写文件。唯一接受状态为
`exact_request_valid_runtime_not_read_no_writes`。

## 5. 单次 attest

仅在 source、checkpoint、hard contract、producer claim、两个 Python runtime、MJCF 和 materialized paper
都已恢复，且 checkpoint-SHA output root 从未存在时运行：

```bash
test ! -e \
  "/workspace/codexschema/phase1_signed_face_rescue_20260713/executions/signed_face_k100_v1/<checkpoint_sha256>"

python3 "$CONSUMER" \
  --repo-root "$SOURCE" \
  --manifest "$MANIFEST" \
  --request "$REQUEST" \
  attest
```

consumer 先完成所有 no-write 验证：checkpoint SHA/filename/embed/finite、lineage、producer claim、相邻 hard
contract、runtime fingerprint、MJCF/plant、actual schedule schema/semantic/order，以及 actual activation 的
file/content SHA、全 100 次分母和 exact float signed-face contract。全部通过后才原子创建 checkpoint-SHA
namespace，以 `O_EXCL` 写 `execution_evidence.json`，最后写 `execution_claim.json`。

成功状态只能是 `checkpoint_execution_inputs_attested_judge_not_started`。claim 状态为
`attested_not_executed_no_decision`；它仍写明 `judge_started=false`、`stop_or_promote_authorized=false`。
未来 judge 需要另一份 reviewed runner 显式消费该 exact claim；本 consumer 没有该入口。

## 6. fail-closed 与失败保全

- 同一 checkpoint SHA 只能映射到一个全局 namespace；换 `request_id` 或路径不能重判/覆盖。
- output root 已存在就拒绝，包括只有 evidence、没有 claim 的 partial failure；不得删掉后重跑。修复必须
  另发 versioned contract，并保留旧现场。
- actual activation 缺失、SHA 不符、content SHA 不符或 sign 变成 integer/bool，全部拒绝；旧 summary 不能救。
- checkpoint 非 finite、filename/embed iteration 不同、lineage 非 exact、producer claim 不匹配、hard contract
  非相邻、source dirty、runtime/MJCF/plant drift，全部在写 claim 前拒绝。
- request 中所有 path 拒绝 glob/wildcard、`..`、非规范写法与 runtime symlink ancestry；checkpoint、hard
  contract、producer claim、MJCF、paper、request 与 manifest 在使用后再次核对 inode/metadata/SHA，中途替换
  同样拒绝。
- attestation 不是成绩，不得据此停止 trainer、晋级 checkpoint、购买第二 seed 或更新 Gate3/部署结论。
