# 严格重绑定 Stage-1 signed-face exam bank

本操作只处理仿真题库 metadata，不启动 trainer、judge、Isaac、MuJoCo、vendor stack 或真机。目标是把
旧 [`schema-3 bank`](../DEFINITIONS.md) 的 `exam` split 严格迁移到 signed-face target commit，同时保持
371 道题的非 metadata 数组与旧/新物理行为一致。输出使用独立
[`no-clobber`](../DEFINITIONS.md) 目录；存在即拒绝，completion report 最后写入。

实验合同见
[`EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND`](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)。

## 1. 固定输入与禁令

- source repo 必须是 clean exact
  `882fea4285f0cf9a97ba79d79ae8af31d26ea1ed`，路径
  `/workspace/codexschema/nohope_signed_face_rescue_882fea4`；不得在运行中 pull、切换或修改。
- base commit 固定为 `6d93bcb16c422a2f42748c2dc99432559653480b`。
- runtime 固定为 `/workspace/hope_isaac_venv/bin/python`。
- 旧 exam 必须是普通非 symlink 文件：`63,968` bytes、SHA
  `d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096`、split `exam`、正手
  `183` 题、反手 `188` 题。缺失时按
  [`setup_local_sync.md`](setup_local_sync.md#phase-1-fresh-and-causal-bundle-2026-07-11)恢复并先验 SHA；
  不能用相邻或重生成文件代替。
- 不得设置 `allow_legacy=true`，不得改 manifest，不能复用 train-v2 输出目录，不能删除失败目录后原名重跑。
- 只允许仿真数据操作；本页没有真实机器人命令。

## 2. 固化 control 副本

```bash
SOURCE=/workspace/codexschema/nohope_signed_face_rescue_882fea4
CONTROL=/workspace/codexschema/phase1_signed_face_rescue_20260713/control/exam_bank_rebind_v1
CONFIG="$CONTROL/phase1_signed_face_exam_bank_rebind_prereg_20260713.json"
TOOL="$CONTROL/rebind_stage1_question_bank_physics_contract.py"
CONFIG_SHA=2153553abe105ace0ae8a90c174198e57b141379d9b3bfc76bdee8d52af7616a
TOOL_SHA=cf8f6353a6b2a8d90aa7cbb960d5bdb9e681fb174458e088a9341ccda5b8e968

test "$(git -C "$SOURCE" rev-parse HEAD)" = \
  882fea4285f0cf9a97ba79d79ae8af31d26ea1ed
test -z "$(git -C "$SOURCE" status --porcelain)"

mkdir -p "$CONTROL"
install -m 0444 configs/phase1_signed_face_exam_bank_rebind_prereg_20260713.json "$CONFIG"
install -m 0555 scripts/rebind_stage1_question_bank_physics_contract.py "$TOOL"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA"
test "$(sha256sum "$TOOL" | awk '{print $1}')" = "$TOOL_SHA"
```

若 control 里同名文件已经存在，先逐字节核对；不要覆盖已经参与过 run 的 control。内容不同就发布新的
manifest ID、control 路径和输出版本，不能现场改 v1。

## 3. 只读 validate

```bash
/workspace/hope_isaac_venv/bin/python "$TOOL" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-script-sha256 "$TOOL_SHA" \
  validate
```

成功只允许输出 `status=validated_no_writes`，且不得创建
`/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_exam_bank_rebind_v1`。它证明：

1. source HEAD/clean、七文件物理合同、Git diff、唯一新增 helper 和旧 executable AST 门通过；
2. loader/generator 在 base/target 间不变；
3. exam path/bytes/SHA/split/题数/旧 family 及两份动作 receipt 精确匹配；
4. 全部非 metadata 数组 finite，只有四个预注册 metadata leaf 将发生变化。

`validated_no_writes` 不是完成证据，不能授权 L2 或 judge。

## 4. 独立发布

只有第 3 节在同一机器、同一 control、同一 clean source 上通过后才运行：

```bash
/workspace/hope_isaac_venv/bin/python "$TOOL" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-script-sha256 "$TOOL_SHA" \
  run
```

consumer 独占创建 `.../assets/schema3_exam_bank_rebind_v1/`，先写
`s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz`，以 target runtime
`allow_legacy=false` 复核 schema/split/motion contract，并对正手 183 + 反手 188 道题比较 base/target
contact 与 flight tensor raw bytes、重跑 landing/net。只有这些全过后才最后写 `rebind_report.json`。

成功输出必须是 `status=published`。立即记录 bank/report 完整 path、bytes、SHA、report
`content_sha256`，并复核：

- `question_arrays_changed=false`、`legacy_load_used=false`；
- target physics SHA 为 `09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95`；
- target source-family SHA 为 `9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db`；
- 两侧 question count 精确为 `183/188`，每侧 `old_new_all_output_bytes_equal=true`；
- manifest/tool SHA、source proof、input receipt、motion receipts、四-leaf delta 和 runtime versions 都齐全。

失败目录和 partial 必须原样保留供诊断。不能覆盖、删除或继续写；修复后只能新发 v2。

## 5. 通过后仍然阻断的步骤

新 metadata 会改变 bank 文件 SHA，而 question ID 把 bank SHA 作为输入，因此所有旧 schedule 都失效。
通过本页 E2 后还要从**新 bank**另行物化并冻结新的 immutable schedule，绑定 schedule path/file SHA、
semantic SHA、题序、每侧分母和独立 activation。该 paper contract reviewed 前：

- 不得启动 L2 judge；
- 不得把 train 与旧 exam 组合成 exact same-family 成绩；
- 不得用旧 K100 schedule 的 bank row“近似复用”；
- 不得声称 Isaac/MuJoCo/G06/Gate3 行为通过。

## 历史 train-v2 兼容性

原 train manifest 文件保持 byte-exact SHA
`5b22a6dd3c41ba1abd44e631e408ed73ada2ac66fc7ff86dc62d48f69ff2ad29`，generalized consumer 仍接受它的
原 profile；本变化没有改写已发布 train bank/report。历史生产 report 绑定的是 commit
`62dfbbfd7500db51437c4115f6fc2e9a5d86d9e4` 中
consumer SHA `c9296d17...0824e`，不能用本页新 tool SHA 回填或改写历史证据。
