# 物化 signed-face exact K100 paper（不启动判卷）

本操作从 E2 rebound exam bank 生成一份新的 [`K100`](../DEFINITIONS.md#q50-and-k100)：正手/反手各
50 题，全部已排定 attempt 都留在分母。它只发布 schedule 和 paper-only activation，不启动 trainer、
judge、Isaac、MuJoCo、vendor stack 或真机，也不授权 L2、第二 seed、checkpoint 晋级或停止。

预注册与决定规则见
[`EXP-P1-SIGNED-FACE-EXAM-PAPER`](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)。
私有 bank 恢复路径见 [`setup_local_sync.md`](setup_local_sync.md)。

## 1. 冻结输入与源码

```bash
SOURCE=/path/to/clean/detached/nohope
CONFIG="$SOURCE/configs/phase1_signed_face_exam_k100_activation_prereg_20260714.json"
CONSUMER="$SOURCE/scripts/materialize_phase1_signed_face_exam_k100.py"
CONFIG_SHA=e401305d4564def80677e6d881ef4afabde01d96ea7ea6aa08224d86835de556
CONSUMER_SHA=4e094bbebe525fb9cd756c3fa6eebe7436c72f94aba2a12ecd136f612761ac6e
PY=/workspace/hope_isaac_venv/bin/python

test -z "$(git -C "$SOURCE" status --porcelain)"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA"
test "$(sha256sum "$CONSUMER" | awk '{print $1}')" = "$CONSUMER_SHA"
```

clean detached source 必须包含 manifest 绑定的原样 consumer、schedule module、schema-3 bank loader、
signed-face scorer 和 tracked E2 result ledger；content SHA 不符即拒绝。consumer 本身没有 subprocess、SSH、
signal、trainer、judge 或 simulator 调用面。

exact bank 必须恢复在：

```text
/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/
  schema3_exam_bank_rebind_v1/s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz
```

它必须是普通非 symlink 文件：`63,643` bytes、SHA
`60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca`，physics SHA
`09dfe899...afb95`、source-family SHA `9603a178...a9db`、schema-3 `exam`、正/反手 `183/188`。
缺文件就停止；不能复制旧 bank 改名，不能重生成猜测版。

## 2. no-write static validate

先运行不读 private bank、也不创建输出的源码门：

```bash
cd "$SOURCE"
"$PY" "$CONSUMER" \
  --config "$CONFIG" \
  --repo-root "$SOURCE" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-consumer-sha256 "$CONSUMER_SHA" \
  static-validate
```

唯一可接受状态是 `source_reviewed_runtime_consume_not_run`，并同时报告 `output_created=false`、
`trainer_started=false`、`judge_started=false`。这只是 E1，不是 materialized paper。

本分支已在 macOS host 运行这一步，rc0；由于本机没有 exact private bank，未运行下一节。

## 3. 单次 consume

只在 exact bank 已恢复、output root 从未存在时运行：

```bash
test ! -e \
  /workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1

cd "$SOURCE"
"$PY" "$CONSUMER" \
  --config "$CONFIG" \
  --repo-root "$SOURCE" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-consumer-sha256 "$CONSUMER_SHA" \
  consume
```

consumer 会以 `allow_legacy=false` 加载 bank、验证 runtime physics/family、从 bank bytes 重新导出所有原子
question ID，并检查每行 raw-A demanded normal 为单位向量且按 `[+1,-1]` 转成 physical-B 后严格
`x>1e-6`。随后沿用现有 schema-v3 schedule 算法，以 seed `0`、hold `[0,100]`、每侧无放回 `50`
生成严格轮转纸。

输出顺序固定为：

1. 独占创建 `.../papers/signed_face_exam_k100_v1/`；
2. `O_EXCL` 写 `signed_face_exam_k100.schedule.json`；
3. 从落盘 bytes 对 exact bank 重验 file/semantic/question-order SHA、100 个唯一题和 50/侧；
4. 最后 `O_EXCL` 写 `signed_face_exam_k100.activation.json`。

成功状态只能是 `paper_materialized_not_started`。立即归档两文件的 path、bytes、file SHA、schedule
semantic SHA、ordered question-ID SHA 和 activation content SHA。activation 仍必须显示 trainer/judge/L2/
第二 seed/stop/promote/formal score/Gate3/deploy/real robot 全 false。

## 4. fail-closed 与 partial 保全

- 旧 paper file/semantic/order receipt `66e89986...71cb3` / `7dc6af82...dff3e` /
  `b87e81a3...21f91` 明确禁用；consumer 不提供 `--schedule` 输入。
- 任一侧不足 50、重复 question ID、unsigned/oriented-plane fallback、bank/source mutation 都必须停止。
- 如果 output root 或任一文件已存在，不能覆盖、删除后重跑，也不能续写 partial。失败现场原样保留；修复
  后必须新发 v2 路径和 manifest。
- schedule 存在但 activation 不存在，表示 partial failure，绝不是可判卷状态。

## 5. consume 成功后仍然阻断

本 activation 只是 immutable paper receipt。它不会消费 checkpoint，也不绑定某个 L2 arm/ONNX/judge
runtime。需要另立后续 reviewed execution contract，逐项绑定 checkpoint↔hard-contract lineage、exact
evaluator source/runtime/MJCF/plant 和本 schedule/activation receipt；在该合同进入 main 并独立激活前，
不得启动 judge、trainer、第二 seed，不能产生 formal/G06/Gate3 成绩。
