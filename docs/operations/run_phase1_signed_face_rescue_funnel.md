# 运行 Phase-1 有符号拍面单-seed 机制漏斗

状态：v1–v5 的学习前失败证据均已保留；v5 被旧 train-bank 物理合同正确拒绝，严格新制品重绑定已预注册、尚未运行，G05/G06 仍为 `Partial`。

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
CONTROL=/workspace/codexschema/phase1_signed_face_rescue_20260713/control/v5
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

本版冻结值：manifest `362e8237179b0de15522d522c675155db7b8a884b8b9bd01f83061973571f6a5`；
launcher `cae16fddfee2eec073f0fb81099d07607f11d687f830ff28b7aa005c52fa32fd`。

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
install -m 0555 scripts/rebind_stage1_question_bank_physics_contract.py \
  "$REBIND_CONTROL/rebind_stage1_question_bank_physics_contract.py"

REBIND_CONFIG="$REBIND_CONTROL/phase1_signed_face_bank_rebind_prereg_20260713.json"
REBIND_TOOL="$REBIND_CONTROL/rebind_stage1_question_bank_physics_contract.py"
REBIND_CONFIG_SHA=$(sha256sum "$REBIND_CONFIG" | awk '{print $1}')
REBIND_TOOL_SHA=$(sha256sum "$REBIND_TOOL" | awk '{print $1}')

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

`validated_no_writes` 只说明输入和源码前置门通过，**不是**可消费的完成证据；只有 `run` 返回
`published` 且 report-last 文件的内容/SHA 全部复核通过，v6 才能绑定该 bank。v6 launcher 还必须解析
report 中的 target commit、physics SHA `09dfe899...afb95`、family SHA `9603a178...a9db`、exact motion
门和正反手两份 replay 结论，不能只看输出文件存在。

成功后把输出 bank 和 report 的完整 SHA 冻结到新的 v6 funnel manifest/control/run names，再运行
本页后续步骤。v5 manifest 不得现场改写。train bank 重绑定不能授权 L2/judge：对应 exam bank 尚未有
相同 target family 证据，且 signed directional checkpoint paper 仍未冻结。

历史 `control/bank_rebind_v1` 的 no-write preflight 因 `ast.dump` 跨 Python 小版本字段不稳定而拒绝；它
没有创建 output root。v1 文件/输出原样保留，禁止覆盖。v2 仅把冻结证据改成 helper 原始源码片段 SHA，
并继续在同一运行 Python 内要求移除 helper 后旧/新 AST 完全相同；其余数组、metadata、runtime replay
和 no-clobber 门都不变。

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

当前 v5 对所有 `--stage l2 validate|plan|launch` 都在任何 runtime 写入前返回：
`L2 is blocked`。原因不是缺 GPU，而是 immutable signed-face directional checkpoint paper 的
schedule/path/SHA 尚未冻结。不得绕过 validator，也不得把 L1 completion 文件改名为 L2 授权。

```bash
! python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --stage l2 validate
```

后续必须提交 reviewed v6，把同一 immutable paper 的 path/SHA、判读合同和 activation closure 一起冻结，
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
