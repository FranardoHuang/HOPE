# 从第 0 帧静止准备态生成短击球路径

本操作只生成 host-only [schema-2 motion](../DEFINITIONS.md) 候选。它不启动训练、simulator、部署或真机，
也不把 quintic 路径冒充 [`TOPP`](../DEFINITIONS.md) / 动力学证书。完整实验边界见
[EXP-MOTION-READY-TO-STRIKE-0P5](../experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

## 单候选生成

`--ready-source` 是提供共同准备姿态的 schema-2 动作；工具只取其第 0 帧姿态并把速度显式置零。
`--contact-frame` 和 `--join-frame` 都是源动作行号；`--blend-intervals` 是从最后一个静止准备行到
源 join 行的 50 Hz 区间数。下面命令只是形状示例，真实路径和帧号必须来自预注册动作记录：

```bash
python3 scripts/build_ready_to_strike_motion.py \
  --source /ABS/SOURCE.schema2.npz \
  --ready-source /ABS/SHARED_READY.schema2.npz \
  --ready-frame 0 \
  --contact-frame 66 \
  --join-frame 60 \
  --hold-frames 4 \
  --blend-intervals 16 \
  --output-npz /ABS/EMPTY/candidate.npz \
  --output-contract /ABS/EMPTY/candidate.contract.json
```

对于 50 Hz、`hold_frames=4` 的 0.5 秒候选，必须满足：

```text
blend_intervals + (contact_frame - join_frame) = 22
output_contact_frame = 25
```

join 还必须至少早于受保护的触球前 `0.1 s` 一行。不要看到结果后随意调一个 join；应先冻结一组
`(contact_frame - join_frame, blend_intervals)` 配对，再全量报告。每个输出 JSON 必须确认：

- `frame0_shared_ready_pose_bitwise_equal=true`；
- `initial_zero_velocity_frames>=3`；
- `protected_window_bitwise_equal=true`；
- `ready_source_velocity_channels_ignored=true`；
- `training_authorized=false`；
- NPZ SHA 与 JSON 内绑定相同。

生成器使用同目录双 hard-link 的 no-clobber 发布，但不是断电/crash 原子事务。若只出现 NPZ 或 JSON
之一，或进程被强杀，保留目录作为失败证据；不得删除后用同一个 attempt 自动重放。输出目录只能是
可信私有 real directory，不能含 symlink。

输入字段集必须精确。原生 schema-2 可以只有六个时序通道、`fps` 和四项 active kinematics metadata；
若存在历史迁移溯源，则以下三项必须同时存在且由 canonical v2 writer 产生：

```text
kinematics_migration_source_sha256 = lowercase 64-hex unicode scalar
kinematics_migration_source_point  = link_origin | center_of_mass unicode scalar
kinematics_migration_tool          = migrate_motion_kinematics.py/v2 unicode scalar
```

不要删除正式资产的三元组来绕过检查。生成器把击球 source 三项逐位复制到输出，并只在 JSON 中另行记录
ready-source；它不会重读旧 legacy ancestor bytes，所以这不是 ancestor SHA 的重新认证。三项残缺、数组而非
scalar、bytes/object/integer、未知额外字段或非法值都会在发布前拒绝。

2026-07-17 的首次 Pod2 namespace `attempt_2137b82b` 使用旧 v1 source gate，正反手均因错误拒绝上述
完整三元组而停止；没有候选、TOPP 或 GPU 行为。该目录只作证据，不得删除或在同 namespace 重发。

第二次 namespace `attempt_2_66f93559` 已证明生成合同可消费真实资产，但 production-FK TOPP 只找到
正手 `0.64 s`、反手 `0.94 s` 可行上界。后续不得手改 join 猜点；使用版本化
[`configs/ready_to_strike_join_ladder_20260717.yaml`](../../configs/ready_to_strike_join_ladder_20260717.yaml)
派生 `join_frame=contact_frame-delta`、`blend_intervals=22-delta`。先跑 `delta=6/17` 的 ready×side
端点因子阵，再按冻结规则跑中点 `12` 与按需细化 `9/14`；已跑的 shared-ready `delta=6` 不重放。
新格逐个报告，不因某一格失败自动 retry。

## Stage-1 历史结果的只读认证

端点 Stage-1 的原始 runner 没有把自身源码写进 summary，所以原始数值不能直接解锁下一层。不要重跑
六格；用 tracked historical attestor 对既有树做一次只读重建。默认命令只 dry-run，不写 receipt：

```bash
python3 scripts/attest_ready_to_strike_ladder_stage1.py \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage1_8d74025e \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml
```

dry-run 成功后，才允许在同一组未变输入上发布唯一的 no-clobber receipt：

```bash
python3 scripts/attest_ready_to_strike_ladder_stage1.py \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage1_8d74025e \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml \
  --execute \
  --confirm ATTEST_READY_TO_STRIKE_STAGE1_ONCE
```

执行源码、queue 和 Stage-1 树必须位于同一台 host 的绝对路径；如果 Pod 上没有该 tracked source，先以
O_EXCL 写入 Stage-1 根目录并核 source SHA，再从该副本运行。receipt 固定为根目录下
`stage1_historical_attestation.json`，存在即拒绝再次执行。认证会重验候选 schema-2、generator contract、
TOPP input/output、生产 FK body order、直接工具依赖、预算、触球行、拍速、拍面、首帧零速及可行时间上界。
queue 中 `checkout_commit` 与 `generator_source_commit` 是两个独立 source root：前者必须提供绑定的
TOPP/MJCF/URDF/body-order，后者由 Stage-1 根目录中实际执行的 immutable generator copy 对 SHA；不要
错误要求旧训练 checkout 包含后置生成器，也不要拿当前 main 文件替换历史副本。
schema-2 source/candidate 的 `joint_vel` 必须按 generator 的 float32 输入梯度逐位重算；TOPP FK output
必须按其 float64 工作区梯度再转 float32。两条 producer 合同不同，审计时不可混用或放宽为任选其一。
本 Stage-1 历史族的 TOPP budget envelope scale 固定为冻结 v3 工具默认 `1.5`；receipt 必须精确核该值，
不得误写 `1.0`，也不得只检查为正数。
它只把旧结果升级为 screening evidence；因为历史证书没有完整 argv、transitive source 和 MJCF closure，
`physics_replay_exact/source_closure_exact/mjcf_closure_exact` 仍必须是 `false`，不能冒充动力学重放或部署通过。

## Stage-2 四个中点的一次性执行（当前 v8，尚未远端执行）

Stage-1 receipt `7cf1c7c9…c377f` 已成功发布后，四个 `delta=12` 中点只能由 tracked runner 消费一次。
activation 精确绑定 runner SHA、Stage-1 receipt SHA 和唯一结果目录；换目录重复执行会在创建任何 namespace
前 fail closed。先从包含 runner 的 clean main source 做 dry-run：

```bash
python3 scripts/run_ready_to_strike_join_ladder_stage2.py \
  --activation configs/ready_to_strike_join_ladder_stage2_activation_v8_20260717.json \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage2_d12_v8_canonical_interpreter
```

dry-run 必须报告四格且不创建结果目录。确认 receipt、runner、queue、旧 runtime 与两份动作资产 SHA 全部
一致后，才可用同一组输入执行：

```bash
python3 scripts/run_ready_to_strike_join_ladder_stage2.py \
  --activation configs/ready_to_strike_join_ladder_stage2_activation_v8_20260717.json \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage2_d12_v8_canonical_interpreter \
  --execute \
  --confirm RUN_READY_TO_STRIKE_STAGE2_ONCE
```

v8 不再重跑 generator：它只从 v2 的 terminal summary 读取正式科学事实，并逐字节复验四份
candidate/contract 后复制进新 namespace。旧 V1 summary、generator 副本和 `run.log` 都不是本轮科学输入；
日志只留作诊断，不能因为文本或手抄 SHA 漂移改变实验是否获准。v2 的正式结论只有“四份
candidate/contract 已生成、四次 TOPP 返回 rc1、没有 timing”，不能据诊断日志宣称已定位 rc1 根因。

本轮新执行环境从冻结 Git objects 提供完整 MJCF 闭包。runner 解析原 XML，拒绝 DTD/entity/include、路径逃逸、
重复引用和非 regular/symlink 文件；然后从冻结 commit 的 Git object 读取 `1 XML + 74 mesh`，核
model-tree OID、每个 blob OID、SHA 和大小，以原相对目录 O_EXCL 快照。闭包固定为 `75 files / 14,127,373
bytes / e0381752…b962de`，worktree fallback 不构成授权。TOPP/URDF/body-order 与两份预算动作也只读冻结。
v8 还必须把解释器入口绑定为 `/workspace/hope_mjeval_venv/bin/python`，并以 canonical realpath、目标
binary SHA、Python version 和 venv prefix 作为身份；`readlink()` 字面 target 只保留为前后不变的 TOCTOU
证据，不再把语义等价的 symlink 拼写当失败。它还核解释器在去除外部 `PYTHONPATH`
后可 import 实际所需的 `numpy+mujoco`，并在启动四个 child 前用 exact MJCF 做一次 parser preflight。
项目的 TOPP 脚本并不 import `scipy`，因此不得再用 `scipy` 是否存在作为启动门。四个 TOPP 可并行，
但每个 CPU child 的 reviewed timeout 为 3600 秒；任一格
失败都发布 terminal summary、全批不重试。`runup_s` 必须同时等于 output contact-frame/fps 与 timing bound，
budget scale 必须为冻结默认 `1.5`。成功只说明 Stage-2 screening 执行完整；只有 `<=0.5 s` 且 hard gate
全过的格才能进入 L0/L1，仍无训练、部署或真机权限。

旧 v1 namespace `join_ladder_stage2_d12_8d74025e` 已永久消费且不得再运行：四格 generator 均 rc0，
但 validator 把 generator 的 float32 producer-gradient 错按 TOPP 的 float64 workspace-gradient 重算，故
全部在 TOPP 前 fail closed。summary SHA=`f92e6b8b…63c0e`。v2 不改变动作、join、预算或 acceptance；只把
candidate/TOPP 两条已冻结的 producer 合同分开验证，并在新 activation 中精确绑定旧 failure summary、
旧 runner/activation SHA 和 `automatic_retry=false`。旧 summary 缺失或改变时，v2 在创建新 namespace 前拒绝。
v2 namespace `join_ladder_stage2_d12_v2_float32_producer` 也已永久消费：四份 candidate 与 v1 逐字节一致，
四个 TOPP 均 rc1，summary SHA=`6910db28…f1476`，没有 timing。旧日志只作诊断；正式合同不把日志文本
或 SHA 当作候选、时间律或结果证据。v3 dry-run 在创建结果 namespace 前又正确拒绝：consumer 错把 v1
contract SHA 配给 v2 candidate；
stderr SHA=`c58baf2d…2e16`，execute/TOPP 均未启动，固定 v3 结果目录仍不存在且不得用新源码复用该 activation。
v4 使用新 source、activation 和 namespace，绑定 v2 summary 中实际四份 contract SHA；除此之外只修
source-closure，不改 candidate、join、hold、预算、acceptance 或 TOPP 算法。
v4 dry-run 随后又在结果 root 前暴露重复量尺：四份 log 已按完整 SHA 绑定，consumer 仍额外猜测日志必须
同时含 `.stl` 与英文 `no such file or directory`；真实不可变 log 格式不同，execute/TOPP 同样未启动。
v5 使用新 activation/namespace，删除这条脆弱文本解释，却仍保留四份 exact log SHA；其中一个 SHA 又因
手抄一字符错误而在结果 root 前 fail closed，execute/TOPP 仍未启动。这个反例说明诊断日志不应进入科学
合同。v6 因此删除全部旧日志、V1 summary 和 generator 副本前置，只保留 v2 summary 中的四份
candidate/contract 与 `TOPP rc1/timing unavailable` 事实。main `8b371eb7` 的唯一 v6 dry-run 全绿，随后
execute natural terminal；summary=`b5209bc7…`，四格 generator=`0`、TOPP rc=`1`、无 timing，
`75 files/74 mesh` closure 与无残留均正确。只读 forensics 发现四份 log 同 SHA `f1d5088e…`，共同首错为
`/usr/bin/python3` 缺 `mujoco`，故分类为 runtime dependency closure 而非动作失败。targeted probe 已在
清空 `PYTHONPATH` 后用 `/workspace/hope_mjeval_venv/bin/python` 成功 import `numpy 2.5`、`mujoco 3.10`
并加载 exact MJCF（`nq=38,nv=37,nbody=33,ngeom=79,nmesh=74`）。v7 绑定 package/ELF closure 后的唯一
远端 dry-run 因 `readlink()` 字面 target 过绑定而在 root/child 前 fail closed，execute=`0`。v8 只改为
canonical interpreter identity，继续绑定两份 package 完整 RECORD、实际加载 ELF、每条 `DT_NEEDED`
解析边、canonical `ldd/readelf` 与 MJCF pre/post snapshot；科学四格和 acceptance 不变。TOPP 非零或
snapshot 漂移时 exact 标志必须为 false。本地 Stage-2 专项 `91 passed`，runner/activation SHA 为
`40e89c6a…ae09` / `e878de11…0447`，上面的**唯一一次**远端 dry-run/execute 已获 source 授权但尚未执行。
这仍不是 behavior、动力学或 0.5 秒通过；若 dry-run 或 execute 失败，保全新 namespace 且不重放。

## 下一步不是直接训练

每条候选依次执行：

1. 使用 exact vendor MJCF、URDF 和 runtime body order 做 production FK 重建；
2. `topp_mintime.py --objective runup --body-mode fk`，要求 run-up `<=0.5 s`，并保持触球行、拍速和拍面；
3. schema-2 L0、vendor L1 自碰/自打、整轨桌网余隙 `>=5 mm`；
4. CoP、摩擦锥、力矩和连续平衡动力学；
5. 用该候选 motion/certificate SHA 新物化 0.5 秒 K100；Isaac 只作 inexact 诊断，最终跑 vendor MuJoCo。

任一上游门失败都停止该候选。不能用 Reward 抵消自碰、桌网、非 finite 或动力学失败。
