# 运行 Phase-1 有符号拍面单-seed 机制漏斗

状态：v1–v5 的学习前失败证据均已保留；v6 的 A/B/C 已到终档，D 在 runtime verified/checkpoint 前
Kit boot timeout；只允许新的 [v6r1](../DEFINITIONS.md) D 单格补跑，尚未启动，G05/G06 仍为 `Partial`。

本页运行 [`EXP-P1-SIGNED-FACE-RESCUE-FUNNEL`](../experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)
冻结的四个因果格。`run_name` 是每条训练不可复用的运行名；`seed` 是随机初始化/采样种子；其他缩写见
[术语与人话对照](../DEFINITIONS.md)。本操作只允许仿真训练，不运行 judge、部署或任何真实机器人命令。

> 当前入口是下文“v6r1 D-only 补跑”，不是重新执行旧四格命令。原 v6 D 的 no-clobber claim 必须
> 原样保留；不得删除、覆盖或用原 `run_name` 重试。

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
  exact 证据。validator 只允许父/新合同中的 `question_bank` 按冻结 old→new 值改变；其他共同字段仍
  逐值相同，且只允许预注册的新增 key。
- C/D 不读取 checkpoint，必须保存 `training_contract_lineage_exact=1`；A/B 必须保存 `0`。四格运行后
  emitted hard-contract SHA 必须完全相同。
- L1（小机制冒烟）是 `512 env × 25 update`；只有四个终档均 finite、iteration/lineage/合同正确后，
  `finalize-l1` 才以 no-clobber 方式生成 completion/activation 证据。它**不单独授权 L2**。L2 设计为
  `4096 env × 1001 update`，保存热启动 `14000/14300/14800` 与 fresh `200/500/1000`，但当前还缺
  独立冻结的 signed directional checkpoint paper，manifest 固定 `launch_authorized=false`。
- L2 只训练/产 checkpoint。immutable signed-face 小卷尚未冻结 path/SHA，因此 launcher 明确不启动
  judge，不停臂、不晋级，也不自动购买第二个 seed。

## 1. 历史 v6：准备独立源码与外部控制目录

先确认没有正在使用目标 Git 工作树的 trainer。只在该检查通过后，由现有 repo 建 detached worktree；
不要修改历史训练目录：

```bash
SOURCE_COMMIT=882fea4285f0cf9a97ba79d79ae8af31d26ea1ed
SOURCE=/workspace/codexschema/nohope_signed_face_rescue_882fea4
CONTROL=/workspace/codexschema/phase1_signed_face_rescue_20260713/control/v6
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
CONFIG_SHA=95f7cc9584feaf267f63652888e29f424881a7425088363083132d35a91ece63
LAUNCHER_SHA=e990b933af47e2de371ba88623e7493fe5e66b8937f7ec2881f27808060f1db0
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$LAUNCHER_SHA"
```

不要手改生产副本。commit 合入后以本页记录的最终 SHA 对账；如果不同，停止并回到源码审查，不能现场
“更新期望值”。

本版冻结值：manifest `95f7cc9584feaf267f63652888e29f424881a7425088363083132d35a91ece63`；
launcher `e990b933af47e2de371ba88623e7493fe5e66b8937f7ec2881f27808060f1db0`。

v1 文件保留在 `control/v1`，其 runtime `validate` 在创建任何 run claim 前 fail closed。根因不是父模型
非有限：旧审计只遍历 checkpoint 顶层，而 RSL-RL 的浮点权重位于嵌套 state dict，合同 provenance
位于 `checkpoint["infos"]`。只读复核得到 `74` 个浮点 tensor、`1,762,715` 个浮点元素、非有限元素
`0`，并从 `infos` 读回 schema `3`、合同 SHA `3a3b3d95...b9972`、lineage `1`。v2 改为递归扫描嵌套
容器并只从 `infos` 读取 runner 写入的三项合同字段；`control/v1` 不覆盖、不复用。

v2 通过上述 checkpoint preflight，但首格在创建 claim 后、第一次学习迭代前因
`ModuleNotFoundError: whole_body_tracking` 退出。根因是 launcher 只检查了机器 env 文件存在，却没有
把 exact detached worktree 的 source-first `PYTHONPATH` 传入 child；直接 source 机器 env 又会错误指向
历史 `6d93bcb` checkout。该失败目录和日志永久保留，v2 不自动重试。v3 换新 run name/control 路径，
逐字节绑定 tracked `setup_train_env.sh`，拒绝任何 `setup_train_env.local.sh`，显式构造确定性环境 SHA
`ddaa0eff...d743`，并在创建 claim 前要求 `whole_body_tracking.__file__` 位于 exact `882fea4` worktree。

v3 的环境 SHA 正确，但 preflight 把路径检查写成了真正 import；IsaacLab 在 `SimulationApp` 启动前
导入 `omni.kit` 本来就会失败，所以 v3 也在 claim 前被拒绝。v4 保留该 control 证据，改用
`importlib.util.find_spec` 只解析 `whole_body_tracking` 的 source path、不执行包；正式 import 仍由
locked Kit boot 在 `SimulationApp` 之后完成。

v4 随后越过 Python/Kit 环境，但在创建 Isaac scene 时发现 detached worktree 没有 Git-ignored 的 A3
URDF/mesh/config 目录；A 格仍在第一次 learning iteration 前退出，失败 claim/log 原样保留。v5 按
[本地资产恢复手册](setup_local_sync.md)从 clean exact `6d93bcb` checkout 复制到 ignored 目标，绑定
`46` 文件、`15,378,264` bytes、canonical tree SHA `0137f59b...26c6`，拒绝 symlink/特殊文件/extra
file，并在 claim 前同时复核 restore source 与 target tree；Git 工作树仍必须 clean。

v5 随后成功创建 scene，但 schema-3 loader 发现旧 train bank 的 physics contract 绑定
`virtual_ball.py=3dc52373...5ed4`，而 target commit 是 `14113de4...3c8`，因此在 hard-contract marker、
第一次 learning iteration 和任何 checkpoint 前退出。A 的 claim/log 必须保留，B/C/D 没有创建；禁止
把 `question_bank_allow_legacy` 打开后重试。若失败 child 卡在 Kit 清理，只能先保存完整日志，再按
`run.log.launch` 记录的精确 PGID 处理；不得使用 broad process match。

## 1b. 生成不覆盖旧文件的目标合同 train bank

这一步是 CPU/Torch 数据门，不启动 trainer、judge 或机器人。它只允许“已有物理函数不变、唯一新增
helper 未被旧题生成/回放路径消费”这一种 metadata-only 重绑定；否则必须重新生成题库。

```bash
REBIND_CONTROL=/workspace/codexschema/phase1_signed_face_rescue_20260713/control/bank_rebind_v2
mkdir -p "$REBIND_CONTROL"

install -m 0444 configs/phase1_signed_face_bank_rebind_prereg_20260713.json \
  "$REBIND_CONTROL/phase1_signed_face_bank_rebind_prereg_20260713.json"
# 历史 train-v2 report 绑定 62dfbbfd7500db51437c4115f6fc2e9a5d86d9e4 中的旧 consumer。
git show 62dfbbfd7500db51437c4115f6fc2e9a5d86d9e4:scripts/rebind_stage1_question_bank_physics_contract.py \
  > "$REBIND_CONTROL/rebind_stage1_question_bank_physics_contract.py"
chmod 0555 "$REBIND_CONTROL/rebind_stage1_question_bank_physics_contract.py"

REBIND_CONFIG="$REBIND_CONTROL/phase1_signed_face_bank_rebind_prereg_20260713.json"
REBIND_TOOL="$REBIND_CONTROL/rebind_stage1_question_bank_physics_contract.py"
REBIND_CONFIG_SHA=5b22a6dd3c41ba1abd44e631e408ed73ada2ac66fc7ff86dc62d48f69ff2ad29
REBIND_TOOL_SHA=c9296d1770cf589296ebcb0216c8bf510f62f5ebfe958fd52e373a75ecb0824e
test "$(sha256sum "$REBIND_CONFIG" | awk '{print $1}')" = "$REBIND_CONFIG_SHA"
test "$(sha256sum "$REBIND_TOOL" | awk '{print $1}')" = "$REBIND_TOOL_SHA"

/workspace/hope_isaac_venv/bin/python "$REBIND_TOOL" \
  --config "$REBIND_CONFIG" \
  --expected-config-sha256 "$REBIND_CONFIG_SHA" \
  --expected-script-sha256 "$REBIND_TOOL_SHA" \
  validate

/workspace/hope_isaac_venv/bin/python "$REBIND_TOOL" \
  --config "$REBIND_CONFIG" \
  --expected-config-sha256 "$REBIND_CONFIG_SHA" \
  --expected-script-sha256 "$REBIND_TOOL_SHA" \
  run
```

本版冻结 SHA：rebind manifest
`5b22a6dd3c41ba1abd44e631e408ed73ada2ac66fc7ff86dc62d48f69ff2ad29`；consumer
`c9296d1770cf589296ebcb0216c8bf510f62f5ebfe958fd52e373a75ecb0824e`。`validate` 必须
no-write；`run` 会独占创建
`.../assets/schema3_bank_rebind_v2/`，先写 bank、以目标 runtime 运行 exact motion contract 和 1481 题
old/new bitwise contact/flight replay，再把 completion report 写在最后。目录已存在或只有 partial 时都
拒绝覆盖；调查后必须发布新版本，不能删除后原名重跑。

以上命令只用于复核历史 train-v2 control。当前 tracked consumer 已增加封闭的 exam-v1 profile，但原
train manifest 仍 byte-exact 且兼容；不得用新 consumer SHA 改写已发布 train report。exam 独立迁移只按
[exam-bank 运行手册](run_phase1_signed_face_exam_bank_rebind.md)执行。

`validated_no_writes` 只说明输入和源码前置门通过，**不是**可消费的完成证据；只有 `run` 返回
`published` 且 report-last 文件的内容/SHA 全部复核通过，v6 才能绑定该 bank。v6 launcher 还必须解析
report 中的 target commit、physics SHA `09dfe899...afb95`、family SHA `9603a178...a9db`、exact motion
门和正反手两份 replay 结论，不能只看输出文件存在。

成功后把输出 bank 和 report 的完整 SHA 冻结到新的 v6 funnel manifest/control/run names，再运行
本页后续步骤。v5 manifest 不得现场改写。train bank 重绑定不能授权 L2/judge：对应 exam bank 尚未有
相同 target family 证据，且 signed directional checkpoint paper 仍未冻结。

本次生产 `run` 已返回 `published`。冻结输出：bank path
`/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_bank_rebind_v2/s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz`，
SHA `3a9d8851c1c0b13ef82f58228ea1cf83213157c70d72daa514f1bed3a3885b71`；report path
`.../schema3_bank_rebind_v2/rebind_report.json`，SHA
`9fffed0308eb0102e3575c3a255e9466c04f45e6c0c303cefb5541a19decbb37`，content SHA
`3ea60706f48dc2af911d733869c9023ac9dd25d6aa4db4a26de8868359b5a32d`。v6 launcher 必须解析并绑定
这些值，不能用现场重新计算的新值替代。

历史 `control/bank_rebind_v1` 的 no-write preflight 因 `ast.dump` 跨 Python 小版本字段不稳定而拒绝；它
没有创建 output root。v1 文件/输出原样保留，禁止覆盖。v2 仅把冻结证据改成 helper 原始源码片段 SHA，
并继续在同一运行 Python 内要求移除 helper 后旧/新 AST 完全相同；其余数组、metadata、runtime replay
和 no-clobber 门都不变。

## 2. 只读校验与四格命令复核

`static-validate` 只读 manifest；`validate` 再核对 source commit/clean、关键源码 SHA、两个动作、rebound
train bank 及其 report closure、父 checkpoint finite/iteration/contract/lineage、主机 RAM 与 GPU0
空闲状态。`plan` 打印四条完整 argv，
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

## 4. 原 v6 L1 结果：不能直接 finalize

原设计要求等待四格自然终档，再由 `finalize-l1` 检查 A/B 的 `model_13824.pt` 与 C/D 的
`model_24.pt`。实际 epoch-1 v6 source 为 clean `50c49e58a9413ec6ac1c3ed2565d9a78acdb5e64`；生产
manifest/launcher SHA 为 `97779cee...eebf2` / `9463f228...85052`。A/B/C 已到终档，三格
hard-contract 均为 `dfc583d4...888a5`，lineage 为 `0/0/1`。B 的 terminal checkpoint 稳定后 child
未自然退出，已按记录的 exact PGID `1758211` 单独终止并保留 SHA `cf619541...dcafe` 的 action 证据。

生产 control 与当前 tracked control 使用相同 v6 `manifest_id`，但 source/config/launcher SHA 不同：
生产是 `50c49e5` + `97779cee...eebf2` / `9463f228...85052`，tracked v6 是 `882fea4` 且 bytes 不同。
这是名字碰撞，不是可替换版本；所有恢复与消费必须按 exact SHA，禁止只按 manifest ID 选文件。

D 的 launch contract/state/log SHA 为 `f6dd2fd2...e0b63` / `4e1ab699...f350` /
`baa02f52...3610`，PID `1759428` 已死，但没有 `runtime_verified.json`、hard-contract marker、learning
iteration 或 checkpoint。原 v6 `finalize-l1` 因此必须失败；下面的历史命令只说明原接口，不得执行：

```bash
! python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l1 finalize-l1

```

不要手改或删除旧 D claim 后重做；这次 failure 只能通过下面的新版本单格合同处理。

## 4a. v6r1 D-only 补跑

`v6r1` 是“v6 的第一次版本化补跑”，不是第二 seed，也不是新配方。它只拥有唯一运行名
`phase1_signed_face_l1_v6r1_D_fresh_guidance_seed3`（seed3 的 fresh 拍面引导 D 单格）；exact foreign
v6 launcher 会先重建原 D argv，consumer 再证明新 argv 仅把一个 `run_name` 改为该新名字。

### 恢复外部 exact 控制与 epoch-1 源证据

`50c49e5` 尚未进入当前仓库对象库；先按[本地同步手册](setup_local_sync.md)恢复以下 ignored/private
依赖，不能用当前 tracked v6 文件代替：

```bash
ARTIFACT=/workspace/codexschema/phase1_signed_face_rescue_20260713
FOREIGN_CONTROL="$ARTIFACT/control/v6"
RETRY_CONTROL="$ARTIFACT/control/v6r1"

test "$(sha256sum "$FOREIGN_CONTROL/phase1_signed_face_rescue_funnel_prereg_v6_20260713.json" | awk '{print $1}')" = \
  97779cee50819ae6ff34d62f6f3c2aed6b13c360b1bf7f0d075aec1f07feebf2
test "$(sha256sum "$FOREIGN_CONTROL/run_phase1_signed_face_rescue_funnel.py" | awk '{print $1}')" = \
  9463f228b26e0a2af548dc749b42428cc3dd1a6379c9d11448e854cfa9d85052
test "$(sha256sum "$ARTIFACT/source_50c49e5.bundle" | awk '{print $1}')" = \
  2a794e2c0f9c4adefd5194d94c404bbdf137cf5368f9c2c2aedf2bc50cc0a39e
test "$(sha256sum "$ARTIFACT/source_50c49e5_git_evidence.txt" | awk '{print $1}')" = \
  12dc839fc76217cd714cfd8ef8f61c42c7e8231cce2b218f34fd42da4a008c99
test "$(sha256sum "$ARTIFACT/l1_checkpoint_audit.jsonl" | awk '{print $1}')" = \
  620767581cb47dda23843822129b09c66507b0cdc887e283d619d4b51fb0d354
test "$(git -C /workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5 rev-parse HEAD)" = \
  50c49e58a9413ec6ac1c3ed2565d9a78acdb5e64
test -z "$(git -C /workspace/codexschema/nohope_signed_face_rescue_epoch1_50c49e5 status --porcelain)"
```

只有在 `control/v6r1` 不存在时创建它；已存在的 partial/claim 不能删除后复用。复制 tracked v6r1 文件，
并逐项核对最终 SHA：

```bash
test ! -e "$RETRY_CONTROL"
mkdir "$RETRY_CONTROL"
install -m 0444 configs/phase1_signed_face_d_retry_prereg_20260713.json \
  "$RETRY_CONTROL/phase1_signed_face_d_retry_prereg_v6r1_20260713.json"
install -m 0555 scripts/run_phase1_signed_face_d_retry.py \
  "$RETRY_CONTROL/run_phase1_signed_face_d_retry.py"

RETRY_CONFIG="$RETRY_CONTROL/phase1_signed_face_d_retry_prereg_v6r1_20260713.json"
RETRY_LAUNCHER="$RETRY_CONTROL/run_phase1_signed_face_d_retry.py"
RETRY_CONFIG_SHA=4bbe496aaf5486e3d82f1c689cafa13ac584c11f5ee780a7a2328a6b701fd8db
RETRY_LAUNCHER_SHA=db03ff6f99fa71ed24dd0aa509f79f68a5ffc52dfa222437bed81cc3da412747
test "$(sha256sum "$RETRY_CONFIG" | awk '{print $1}')" = "$RETRY_CONFIG_SHA"
test "$(sha256sum "$RETRY_LAUNCHER" | awk '{print $1}')" = "$RETRY_LAUNCHER_SHA"
```

### validate、plan 与唯一一次 launch

`static-validate` 只看 tracked v6r1 合同；`validate`/`plan` 还会加载 exact foreign v6，重新验证 clean
source、Isaac/Python runtime closure、动作、bank、rebind report、原 D 失败三件套/dead PID/零
checkpoint、source bundle、GPU0 空和 `/workspace/.kit_boot.lock` 空。三个动作都不写 claim：

```bash
python3 "$RETRY_LAUNCHER" --config "$RETRY_CONFIG" \
  --expected-config-sha256 "$RETRY_CONFIG_SHA" \
  --expected-launcher-sha256 "$RETRY_LAUNCHER_SHA" static-validate
python3 "$RETRY_LAUNCHER" --config "$RETRY_CONFIG" \
  --expected-config-sha256 "$RETRY_CONFIG_SHA" \
  --expected-launcher-sha256 "$RETRY_LAUNCHER_SHA" validate
python3 "$RETRY_LAUNCHER" --config "$RETRY_CONFIG" \
  --expected-config-sha256 "$RETRY_CONFIG_SHA" \
  --expected-launcher-sha256 "$RETRY_LAUNCHER_SHA" plan >"$RETRY_CONTROL/d_retry.plan.json"
```

人工确认 plan 的 old/new argv 只有一个 `run_name` token 不同后，才执行唯一一次 `launch`：

```bash
python3 "$RETRY_LAUNCHER" --config "$RETRY_CONFIG" \
  --expected-config-sha256 "$RETRY_CONFIG_SHA" \
  --expected-launcher-sha256 "$RETRY_LAUNCHER_SHA" launch
```

launcher 在启动前以 O_EXCL 写 `d_retry_launch_contract.json`；任何后续失败都会保留 claim，且 Python
consumer 没有直接 signal API，也没有自动第二次重试路径。它调用的 exact-SHA frozen Kit wrapper
在 pre-marker boot timeout 时会按既有逻辑只对 state 中该隔离 PGID 做 TERM→KILL；这是唯一预注册的
自动 cleanup，不是“零 signal”，也不授权 broad signal。若 wrapper 已看到 hard-contract marker 而
后续 first-iteration 等待超时，wrapper 不会再管 trainer，arm 可能仍活着：不要重复 `launch`、不要删除
`control/v6r1`，只读核对 `run.log.launch` 的精确 `pid=pgid`、`run.log`、GPU 与
`runtime_verified.json`，再由人工决定如何处理该 **单一 state PGID**。本操作不授权真机。

### 混合 L1 finalizer

只在 D 进程自然退出后运行。finalizer 要求日志含终档 `Learning iteration 24/25`，`model_24.pt`
稳定/finite、lineage `1`、合同 `dfc583d4...888a5`；同时重新 hash/审计原 A/B/C 四类外层证据、终档
checkpoint 和 B exact-PGID action。通过后才 no-clobber 写
`control/v6r1/l1_mixed_activation.json`：

```bash
python3 "$RETRY_LAUNCHER" --config "$RETRY_CONFIG" \
  --expected-config-sha256 "$RETRY_CONFIG_SHA" \
  --expected-launcher-sha256 "$RETRY_LAUNCHER_SHA" finalize-mixed-l1
```

该 activation 明确记录 A/B/C 来自 original v6、D 来自 v6r1，并绑定两套 config/launcher 与 retry
lineage；原始 checkpoint audit SHA `62076758...d354` 还要精确证明 A/B/C 的 checkpoint/finite/lineage
与 D 当时无 run dir。activation 同时记录 Python consumer 未直接发 signal、frozen wrapper 允许的
exact-PGID boot-timeout cleanup，以及成功 D 路径未执行该 cleanup。它只关闭 L1 完整性，不授权
judge、L2、第二 seed、stop/promote、部署或真机。

## 5. L2 当前 fail-closed

当前 v6 对所有 `--stage l2 validate|plan|launch` 都在任何 runtime 写入前返回：
`L2 is blocked`。原因不是缺 GPU，而是 immutable signed-face directional checkpoint paper 的
schedule/path/SHA 尚未冻结。不得绕过 validator，也不得把 L1 completion 文件改名为 L2 授权。

```bash
! python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l2 validate
```

后续必须提交 reviewed v7，把同一 immutable paper 的 path/SHA、判读合同和 activation closure 一起冻结，
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
