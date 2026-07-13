# 运行 signed-face C2/D2 provenance-complete L1

状态：**C2 已自然到达 terminal，但 v1 outer verifier 因 float/int 类型误判没有发布
`runtime_verified`；D2 尚未 claim，必须使用本页 v1r1 续接。** 本操作只运行
[`signed-face C2/D2`](../DEFINITIONS.md) 两条 fresh L1 小臂：C2 关闭 signed-face guidance，D2 只把
同一 Reward 权重改为 `-0.4`。`L1` 是 `512 env × 25 update` 的发射/合同冒烟，不是判卷或晋级。
`run_name` 是不可复用的训练运行名；`claim` 是外层原子运行占位；其余术语见
[人话定义](../DEFINITIONS.md)。

本工具没有 activation、judge、L2、第二 seed、自动 retry、部署或真机入口。它也不复用 v9 的 C/D
claim/checkpoint；旧 `5f691b3` source 与 `466f8ea` control 只作为只读根因证据。

`v1r1` 是 [one-shot continuation](../DEFINITIONS.md)：只验证已经完成的 C2，并只允许原子 claim
仍不存在的 D2。它不是 C2 retry，也不改变 seed、配方、训练 source 或 L1 边界。

## 冻结输入

- 训练 source：commit `4467d79f1ed425a4263f0caaad2f661e1ec737ad`，tree
  `497db1d8f2d7fb1b554337928f098a2951d4cf0d`；这是从最新 main 形成的新 source gate，不修改旧
  `5f691b3` worktree。
- manifest：[`phase1_signed_face_cd_l1_prereg_20260714.json`](../../configs/phase1_signed_face_cd_l1_prereg_20260714.json)，
  SHA-256 `785ad96dd53e1809ddcf86d1ecd80572b02e3c96ffd6d6599cab20a73b559895`。
- launcher/finalizer：[`run_phase1_signed_face_cd_l1.py`](../../scripts/run_phase1_signed_face_cd_l1.py)，
  SHA-256 `0fa250207246e8bf69b6475125882b45e817f9e777d13039614c82dad9a803ba`。
- v1r1 manifest：[`phase1_signed_face_cd_l1_v1r1_continuation_20260714.json`](../../configs/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json)，
  SHA-256 `8d893009d91bbaa395abaa9474f7048e80e3d2f50c054d3e5a93a74bda56e232`；v1r1 launcher：
  [`continue_phase1_signed_face_cd_l1_v1r1.py`](../../scripts/continue_phase1_signed_face_cd_l1_v1r1.py)，
  SHA-256 `b991fed75ef1bdd4d8ac9e0c6057e57a6e36c257597a163074249ca005691a6c`。
- 动作、train bank、physics、A3 ignored asset、Python/pip 与 IsaacLab 的 SHA/commit 全在 manifest；任一
  漂移均在 claim 前拒绝。
- Pod1 上 C2 固定 physical GPU1，D2 固定 physical GPU2；每条 claim 前只要求它自己的 GPU 没有任何
  compute PID。两次 Kit boot 仍由 host-wide lock 串行，但 C2 写出 `runtime_verified` 后继续在 GPU1
  训练，D2 可立即在 GPU2 boot 并与它并发。两个 command 使用完全相同、source-first 的 `PYTHONPATH`
  与 local `device=cuda:0`；outer claim/环境分别绑定 physical GPU lane。Kit 的 carb/TBB thread cap 都是
  `16/16`，并要求日志出现 exact runtime marker。

## 1. 安装只读 control 和独立 source

先确认目标 GPU 和目标 source path 没有 trainer。不要 pull、切换或修改历史训练 checkout。只从已经
拥有上述 commit 的 clean Git clone 创建新 detached worktree；如果 commit 不可取，停止，不用现场补丁：

```bash
SOURCE_COMMIT=4467d79f1ed425a4263f0caaad2f661e1ec737ad
SOURCE=/workspace/codexschema/nohope_signed_face_cd_l1_4467d79
CONTROL=/workspace/codexschema/phase1_signed_face_cd_l1_20260714/control/v1

git -C /path/to/clean-control-clone worktree add --detach "$SOURCE" "$SOURCE_COMMIT"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain --untracked-files=all)"
```

detached worktree 不含被 Git ignore 的 A3 URDF/mesh/config。按
[`setup_local_sync`](setup_local_sync.md) 从冻结的 `6d93bcb...480b` restore tree **复制**到新 source；
不使用 symlink。launcher 会对 source/restore 两边重算 `46` files、`15,378,264` bytes 和 tree SHA
`0137f59b...26c6`，并确认 target 仍被 Git ignore。

从包含本功能最终提交的 clean checkout 把两个 tracked 文件安装到全新 control；路径或权限不符会拒绝：

```bash
mkdir -p "$CONTROL"
install -m 0444 configs/phase1_signed_face_cd_l1_prereg_20260714.json \
  "$CONTROL/phase1_signed_face_cd_l1_prereg_20260714.json"
install -m 0555 scripts/run_phase1_signed_face_cd_l1.py \
  "$CONTROL/run_phase1_signed_face_cd_l1.py"

CONFIG="$CONTROL/phase1_signed_face_cd_l1_prereg_20260714.json"
LAUNCHER="$CONTROL/run_phase1_signed_face_cd_l1.py"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = \
  785ad96dd53e1809ddcf86d1ecd80572b02e3c96ffd6d6599cab20a73b559895
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = \
  0fa250207246e8bf69b6475125882b45e817f9e777d13039614c82dad9a803ba
```

已存在的 control、run、claim 或同名 training log 不覆盖、不删除；停止并审计。

## 2. 只读检查

`static-validate` 和 `plan` 不创建产物、不启动进程；`validate-runtime` 还分别检查 Pod1 GPU1/GPU2 空闲、RAM、
source/assets/Python/IsaacLab/PYTHONPATH，但仍不 claim：

```bash
python3 "$LAUNCHER" --manifest "$CONFIG" --mode static-validate
python3 "$LAUNCHER" --manifest "$CONFIG" --mode plan
python3 "$LAUNCHER" --manifest "$CONFIG" --mode validate-runtime
```

必须看到 source commit/tree 精确匹配、`writes_or_launches_performed=false`，以及 activation/judge/L2/
second-seed/stop-or-promote 全为 false。任一失败都不启动。

## 3. 串行 boot、跨卡并发训练和独立终档

以下 v1 原始命令只保留为冻结设计说明，**当前现场不得再次执行 `launch-next`**：C2 claim 已存在，且
旧 verifier 的假拒绝没有产生 retry 权限。当前只执行下一节的 v1r1 命令。

每次 `launch-next` 只原子创建一条臂。首次只能是 GPU1 上的 C2；C2 写出 hard contract 和
`runtime_verified`、shared Kit lock 释放后，第二次即可在空闲 GPU2 claim D2，**不等待 C2 终档**：

```bash
python3 "$LAUNCHER" --manifest "$CONFIG" --mode launch-next \
  --root-confirm ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_C2_D2_L1_V1

# 上一命令返回即表示 C2 runtime_verified；它继续训练。GPU2 仍须为空。
python3 "$LAUNCHER" --manifest "$CONFIG" --mode launch-next \
  --root-confirm ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_C2_D2_L1_V1

# 只读监控两份 sidecar 的 exact PID/PGID；各自自然退出后分别 finalize，顺序不限。
python3 "$LAUNCHER" --manifest "$CONFIG" --mode finalize-cell --cell C2
python3 "$LAUNCHER" --manifest "$CONFIG" --mode finalize-cell --cell D2
python3 "$LAUNCHER" --manifest "$CONFIG" --mode finalize-pair
```

不要用 `pkill`、`killall` 或命令行匹配信号。reviewed shell wrapper 只在 hard-contract marker 前超时/退出时
管理它刚创建且记录的 exact `PID=PGID`；Python launcher 自身没有 signal 路径。

每条 terminal 必须同时满足：

1. exact trainer 已自然退出，**本臂所绑定的 GPU**重新为空；日志没有
   NaN/Inf/Traceback/OOM/malloc/Killed，并恰有一次
   `KIT_THREAD_CAP_OK ... 16 ... 16`；
2. 文件名为 `model_24.pt`、内嵌 `iter=24`、所有浮点 tensor finite、schema `3`、fresh lineage `1`；
3. checkpoint `infos` 的 hard-contract SHA 等于相邻 `params/training_contract.json`；该合同明确包含
   positional guidance `0.0` 和本臂 signed-face weight；
4. checkpoint `infos` 的 launch-claim SHA 等于由 manifest/launcher、exact source、优化配方、cell/run、
   host/physical-GPU lane、seed、terminal iteration 和原子 claim directory inode/device 重建的 SHA；
5. C2/D2 两份 hard contract 去掉 `racket_guidance_reward.signed_face.weight` 后逐值相同，原 SHA 必须不同。

`finalize-pair` 只写 provenance-complete 的 L1 成对结果，仍固定 activation/judge/L2/第二 seed/晋级为
false。K100 paper 已另行物化，但其 paper-only activation 也固定 trainer/L2/judge=false；两份 false
不能拼成授权。

若 C2/D2 L1 全通过，最短的下一版本是一个独立 C2/D2-only L2 consumer：同时内容绑定本页 paired
result 与 `phase1_signed_face_exam_k100_runtime_receipt_20260714.json`，再冻结同 source/runtime/GPU lane
的 `4096 env × 1001 update`、`+200/+500/+1000` checkpoint/claim/finalizer。它仍不得自动启动 judge，
也不得买第二 seed。当前 manifest/launcher 没有这些 mode，所以不能直接把 L1 输出传给 L2。

## 4. C2 float/int 假拒绝与 v1r1 D2-only 续接

v1 C2 已自然退出并产生 `model_24.pt`。现场只读证据被冻结为：launch contract
`26bf204d...0e96`、canonical outer claim `37fe2443...86e5`、launch state
`2bcc5656...beb8`、final log `abffd457...6dc3`、hard contract `83f47ae6...2772`、terminal
checkpoint `dbbc7a28...6f6`。旧 `runtime_verified.json`、`launch_failure.json` 和
`terminal_result.json` 都必须继续不存在。

根因不是训练合同漂移：Hydra 参数明确是
`++task.racket.mount_normal_sign_per_clip=[1.0,-1.0]`，训练端也把该项转换为 float 后写入 hard
contract；v1 verifier 却用 `[1,-1]` 整数期望做 exact-type 深比较。Python 中 bool/int/float 的相等
语义容易掩盖这种错位，所以 v1r1 明确只接受两个 exact float，并对 `[True,-1.0]`、`[1,-1]` fail
closed。

新 evidence 绝不写入 preserved C2 arm，而是使用独立 no-clobber 路径
`/workspace/codexschema/phase1_signed_face_cd_l1_20260714/continuations/v1r1/`。安装新的只读 control；
同时复制原 v1 helper 供 v1r1 import，并逐字节绑定其旧 SHA：

```bash
CONTROL_V1R1=/workspace/codexschema/phase1_signed_face_cd_l1_20260714/control/v1r1
mkdir "$CONTROL_V1R1"
install -m 0444 configs/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json \
  "$CONTROL_V1R1/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json"
install -m 0555 scripts/continue_phase1_signed_face_cd_l1_v1r1.py \
  "$CONTROL_V1R1/continue_phase1_signed_face_cd_l1_v1r1.py"
install -m 0444 scripts/run_phase1_signed_face_cd_l1.py \
  "$CONTROL_V1R1/run_phase1_signed_face_cd_l1.py"

CONFIG_V1R1="$CONTROL_V1R1/phase1_signed_face_cd_l1_v1r1_continuation_20260714.json"
LAUNCHER_V1R1="$CONTROL_V1R1/continue_phase1_signed_face_cd_l1_v1r1.py"
test "$(sha256sum "$CONFIG_V1R1" | awk '{print $1}')" = \
  8d893009d91bbaa395abaa9474f7048e80e3d2f50c054d3e5a93a74bda56e232
test "$(sha256sum "$LAUNCHER_V1R1" | awk '{print $1}')" = \
  b991fed75ef1bdd4d8ac9e0c6057e57a6e36c257597a163074249ca005691a6c
test "$(sha256sum "$CONTROL_V1R1/run_phase1_signed_face_cd_l1.py" | awk '{print $1}')" = \
  0fa250207246e8bf69b6475125882b45e817f9e777d13039614c82dad9a803ba
```

先只读验证，再一次性发布 C2 attestation；只有 attestation 可逐值 replay、D2 arm 与 exact run name
仍不存在、Pod1 GPU2 为空时，才运行唯一发射 mode：

`attest-c2` 首次发布时要求 GPU1 为空以记录 terminal barrier；后续 replay 不继续占用或保留 GPU1，
所以另一条合法训练后来使用 GPU1 不会阻断 D2，D2 自身仍只以 GPU2 空闲为发射条件。

```bash
python3 "$LAUNCHER_V1R1" --manifest "$CONFIG_V1R1" --mode static-validate
python3 "$LAUNCHER_V1R1" --manifest "$CONFIG_V1R1" --mode plan
python3 "$LAUNCHER_V1R1" --manifest "$CONFIG_V1R1" --mode validate-runtime
python3 "$LAUNCHER_V1R1" --manifest "$CONFIG_V1R1" --mode attest-c2
python3 "$LAUNCHER_V1R1" --manifest "$CONFIG_V1R1" --mode launch-d2 \
  --root-confirm ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_D2_ONLY_V1R1

# D2 exact PID=PGID 自然退出后：
python3 "$LAUNCHER_V1R1" --manifest "$CONFIG_V1R1" --mode finalize-d2
python3 "$LAUNCHER_V1R1" --manifest "$CONFIG_V1R1" --mode finalize-pair
```

parser 没有 `--cell`、`launch-next`、`launch-c2` 或 retry mode。D2 claim/checkpoint 同时绑定 v1r1
manifest、v1r1 launcher、原 v1 source/recipe 和 C2 attestation；pair receipt 显式声明 mixed outer
control（C2=v1，D2=v1r1），并要求规范化 trainer recipe 与 hard contract 都只能在
`racket_guidance_reward.signed_face.weight` 上不同。该 receipt 仍不授权判卷或 L2。

## 失败处置

任一失败都保留完整 control、claim、state、log、adjacent contract 和 checkpoint。wrapper rc 非零会写
no-clobber `launch_failure.json`；后续合同/终档验证失败即使没有该 marker，既存原子 claim 也会阻断
再次发射。禁止删除目录或换名自动重试。先归档和定位根因；只有配方/合同不变且另有审阅过的新 namespace
时，才可能形成新的尝试。本页不授权真机。
