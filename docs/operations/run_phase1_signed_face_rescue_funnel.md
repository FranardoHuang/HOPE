# 运行 Phase-1 有符号拍面单-seed 机制漏斗

状态：机器预注册已通过源码测试；尚未在 Pod 运行，G05/G06 仍为 `Partial`。

本页运行 [`EXP-P1-SIGNED-FACE-RESCUE-FUNNEL`](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)
冻结的四个因果格。`run_name` 是每条训练不可复用的运行名；`seed` 是随机初始化/采样种子；其他缩写见
[术语与人话对照](../DEFINITIONS.md)。本操作只允许仿真训练，不运行 judge、部署或任何真实机器人命令。

## 已冻结的边界

- 源码必须是 clean detached commit
  `882fea4285f0cf9a97ba79d79ae8af31d26ea1ed`。它包含 signed-face 源码修复；不得改用仍在训练的旧
  `6d93bcb...` 工作树，也不得在 launcher 内 pull/switch/checkout。
- 一张卡只放四个**不同**因果格，全部使用 seed 3：A/B 是同父 checkpoint 的引导关/开，C/D 是同一
  fresh seed 的引导关/开。首轮禁止复制 seed。
- A/B 的父模型是 `model_13800.pt`，checkpoint SHA
  `478efa8d...d9e6`，相邻/嵌入旧 hard-contract SHA `3a3b3d95...b9972`。当前源码给 hard contract
  增加了 event timing/target cadence 字段，所以 A/B 必须显式写成
  `checkpoint_allow_contract_mismatch=true` 的 **inexact representation transfer**；它们永远不是 fresh
  exact 证据。validator 要求父/新合同全部共同字段逐值相同，且只允许预注册的新增 key。
- C/D 不读取 checkpoint，必须保存 `training_contract_lineage_exact=1`；A/B 必须保存 `0`。四格运行后
  emitted hard-contract SHA 必须完全相同。
- L1（小机制冒烟）是 `512 env × 25 update`；只有四个终档均 finite、iteration/lineage/合同正确后，
  `finalize-l1` 才以 no-clobber 方式生成 completion/activation 证据。它**不单独授权 L2**。L2 设计为
  `4096 env × 1001 update`，保存热启动 `14000/14300/14800` 与 fresh `200/500/1000`，但当前还缺
  独立冻结的 signed directional checkpoint paper，manifest 固定 `launch_authorized=false`。
- L2 只训练/产 checkpoint。immutable signed-face 小卷尚未冻结 path/SHA，因此 launcher 明确不启动
  judge，不停臂、不晋级，也不自动购买第二个 seed。

## 1. 准备独立源码与外部控制目录

先确认没有正在使用目标 Git 工作树的 trainer。只在该检查通过后，由现有 repo 建 detached worktree；
不要修改历史训练目录：

```bash
SOURCE_COMMIT=882fea4285f0cf9a97ba79d79ae8af31d26ea1ed
SOURCE=/workspace/codexschema/nohope_signed_face_rescue_882fea4
CONTROL=/workspace/codexschema/phase1_signed_face_rescue_20260713/control/v1
ARTIFACT=/workspace/codexschema/phase1_signed_face_rescue_20260713

git -C /workspace/codexschema/nohope worktree add --detach "$SOURCE" "$SOURCE_COMMIT"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
mkdir -p "$CONTROL" "$ARTIFACT/runs"
```

从已合入本功能的 clean checkout 复制下面两个 Git 文件到外部控制目录；生产 launcher 会拒绝在训练
checkout 内运行，并要求调用方给出两个文件的最终 SHA：

```bash
install -m 0444 configs/phase1_signed_face_rescue_funnel_prereg_20260713.json \
  "$CONTROL/phase1_signed_face_rescue_funnel_prereg_20260713.json"
install -m 0555 scripts/run_phase1_signed_face_rescue_funnel.py \
  "$CONTROL/run_phase1_signed_face_rescue_funnel.py"

CONFIG="$CONTROL/phase1_signed_face_rescue_funnel_prereg_20260713.json"
LAUNCHER="$CONTROL/run_phase1_signed_face_rescue_funnel.py"
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
LAUNCHER_SHA=$(sha256sum "$LAUNCHER" | awk '{print $1}')
```

不要手改生产副本。commit 合入后以本页记录的最终 SHA 对账；如果不同，停止并回到源码审查，不能现场
“更新期望值”。

本版冻结值：manifest `2fed82058342555eec8adbd890d87dc1a9e3120e4d7ee00690b02df77851b0aa`；
launcher `ea3f6b84621c326b247819214ba9fdc5f78e40c6a9719a09ff1fc441f2dfbb1d`。

## 2. 只读校验与四格命令复核

`static-validate` 只读 manifest；`validate` 再核对 source commit/clean、关键源码 SHA、三个训练输入、
父 checkpoint finite/iteration/contract/lineage、主机 RAM 与 GPU0 空闲状态。`plan` 打印四条完整 argv，
不写 run 目录：

```bash
python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  static-validate

python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l1 validate

python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l1 plan >"$CONTROL/l1.plan.json"
```

人工复核 `l1.plan.json`：四条都必须 `seed=3`、同 `CUDA_VISIBLE_DEVICES=0`、同源码/资产/零摩擦/179-D
合同；A/B 仅 guidance 与 run name 不同，C/D 也仅这两项不同；A/B 有同一 `model_13800.pt`，C/D 必须
是 `checkpoint_path=null`。

## 3. 启动 L1

Pod 级 Kit boot 仍由已有 locked launcher 串行；四个 trainer 通过第一学习 iteration 后并发运行。
本命令对每格先原子 claim run directory，再写 launch contract；已有半写 claim 时 fail closed，不删除、
不覆盖：

```bash
python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l1 launch
```

SSH 中断后，trainer 已由 locked launcher 以 `pid=pgid` 的独立进程组启动。重复**同一条**命令时，只有
带完整 `launch_contract.json`、`run.log.launch`、`runtime_verified.json` 且进程身份/命令/合同仍一致的
格会被只读复核后跳过；完整终档格也会跳过。若某格只有部分文件、进程提前退出或 terminal 缺失，自动
重试被禁止：保留目录/日志，先诊断，再另做经过审查的新 run name。不得删除 claim 来伪装首次启动。

只按每格 `run.log.launch` 内的数值 PGID 检查。禁止 `pkill`、`killall`、`pgrep -f` 批量信号；本 launcher
本身不会发送信号。真实机器人不在本操作范围。

## 4. L1 终档与 completion/activation 证据

等待四格自然终档。`finalize-l1` 是只读审计加一次 no-clobber activation 写入：它要求 A/B 的
`model_13824.pt` 与 C/D 的 `model_24.pt` 稳定、finite、文件名 iteration 等于嵌入 iteration，四个
checkpoint 都绑定同一个 emitted contract，且 lineage 分别为 `0/0/1/1`。

```bash
python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l1 finalize-l1

ACTIVATION="$CONTROL/l1_activation.json"
ACTIVATION_SHA=$(sha256sum "$ACTIVATION" | awk '{print $1}')
```

该文件已存在时命令拒绝覆盖。不要手改或删除后重做；若需更正，保留旧证据并物化新版本合同。它只证明
L1 完整，不代表 L2 已授权。

## 5. L2 当前 fail-closed

当前 v1 对所有 `--stage l2 validate|plan|launch` 都在任何 runtime 写入前返回：
`L2 is blocked`。原因不是缺 GPU，而是 immutable signed-face directional checkpoint paper 的
schedule/path/SHA 尚未冻结。不得绕过 validator，也不得把 L1 completion 文件改名为 L2 授权。

```bash
! python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l2 validate
```

后续必须提交 reviewed v2，把同一 immutable paper 的 path/SHA、判读合同和 activation closure 一起冻结，
才能讨论 L2。到那时每个预注册 checkpoint 仍须检查进程、GPU、RAM、完整日志中的
NaN/Inf/Traceback/OOM/Killed、文件名与嵌入 iteration、tensor finite、checkpoint ↔ 相邻 hard-contract
SHA/lineage；不要为填空闲卡复制第二 seed。

## 已知限制

- 本地测试只证明 manifest/argv/lineage/no-clobber/activation 逻辑，不能代替 Isaac runtime 行为。
- A/B 因源码 hard-contract 扩展而必为 inexact；它们只能回答“旧策略是否可被救回”，不能成为 fresh
  baseline。
- L2 的 signed-face directional exam 仍需独立冻结 schedule/path/SHA 与 judge source 后再运行；本工具
  故意没有自动 judge 路径。
- 任何一次训练成功都不替代 vendor MuJoCo Gate3/Gate3B，也不授权部署或真实机器人。
