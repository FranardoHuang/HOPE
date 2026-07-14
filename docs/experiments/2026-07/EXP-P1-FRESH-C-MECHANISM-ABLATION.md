# Fresh C 单机制首轮消融（2026-07-14）

状态：`Partial`。五个全新 `retry-v2` 已越过 `+500`；第六个 qdot-limit tail 的首次发射在第 0 update
超时，完整 namespace 已保留。相同配方的 `retry-v2` 已在 Pod2 GPU2 正常进入训练并到 iter `79`；尚无
任何机制的行为晋级结论。

## 问题与固定对照

用同一个 fresh C（从零初始化的共享拍面、guidance 权重为零）配方、seed 3、4096 environments、
1001 updates 和 `200/500/1000` checkpoints，只改变一个机制；组合格只改变 V1+V2 两项：

- V1：释放手腕线速度模仿；
- V2：击球窗内动作模仿缩放为 0.25；
- V1+V2：测两者交互；
- 击球前底座减速 reward 权重 1.0；
- 随挥后 replay 起点概率 0.5。

每格都显式写全四轴控制向量，公共 base 不暗含这些轴。第六格直接惩罚 31 个实际关节速度接近 runtime
limit 的尾部，仍显式写全前四轴控制值。权威机器清单是
[`phase1_fresh_c_mechanism_queue_20260714.yaml`](../../../configs/phase1_fresh_c_mechanism_queue_20260714.yaml)。

## 首次发射的基础设施负结果

五条 attempt-1 依次在 Pod1 GPU0 创建 claim 并启动子进程，但全部在第 0 update、训练 marker 前 rc=1
退出；进程已死，五目录都只有 `queue_claim.json`、`run.log`、`run.log.launch`，没有 model。根因是
`setup_train_env.sh` 只导出 `HOPE_WBT_PYTHONPATH`，旧队列却直接调用 Python，子进程没有
`PYTHONPATH`，最终为 `ModuleNotFoundError: whole_body_tracking`。这不是五个机制的训练失败。

五个旧 job 已标为 `rejected` 并永久保全；不能删除 claim，也不能复用 run directory。唯一允许的
infra-only retry 是完全继承 motion、bank、exam、source、recipe、seed 和预算的五个 `retry-v2` job。

## 修复与继续门

doctor 和 trainer 现在由同一个 child-environment builder 生成 CUDA 与
`PYTHONPATH=${HOPE_WBT_PYTHONPATH}`。source commit/clean、三类资产和
`find_spec('whole_body_tracking')` 的 exact source origin 在创建 run directory/claim 前完成；失败不留
新 claim。launcher 等到第一个 `Learning iteration`，不是只等早期 Kit marker。该次修复时
`doctor --live` 仍只验 source/assets/module；当时没有可信的 no-Kit Hydra compose 接口，因此没有把
配置解析冒充已通过。

推荐只用单进程 `fill --count N --execute`：每条依次 doctor、claim、等 first iteration、重采六卡再选
下一条。任何失败立即停止且不自动 retry。所有 `retry-v2` 真正到 checkpoint 前，本实验没有科学结论，
[G05](../../gates/G05_isaac_training_first_loop.md) 保持 `Partial`。

2026-07-14T09:53:56Z 的只读现场验收中，`plan --live` 看到两 Pod 六 GPU 的 compute occupancy 都为 0，
五个旧 claim 均为 Pod1 GPU0 `launched` 且因 queue status=`rejected` 不占新槽；五个 retry-v2 依次分到
Pod1 GPU0/1/2、Pod2 GPU0/1。随后五个 `doctor --live` 全部返回 `DOCTOR_OK`，并实际解析到
`/workspace/hope_isaac_venv/bin/python` 与 exact source module。没有创建 retry-v2 claim 或 trainer；这只
关闭了刚才的 import 基础设施 blocker。

## Retry-v2 live 与 checkpoint-100 证据

2026-07-14T10:09:52Z，五条 retry-v2 的 exact PID=PGID 均存活，最新 update 为
`160/146/132/117/103`，日志无 NaN、Inf、Traceback、OOM、malloc 或 Killed。Pod1 GPU0/1/2 各一条，
Pod2 GPU0/1 各一条，Pod2 GPU2 空；两 Pod swap=0。五份 `model_100.pt` 后续均验证 filename=embedded
iteration `100`、76 tensor 全 finite、相邻 schema-3 hard contract SHA 匹配、fresh lineage `1`。
这些只证明训练已正常进入和 checkpoint 可审计；`+200` 前不作机制判断。

## 第六格 qdot-limit tail 的冻结配方

qdot 格使用 clean source `a6ccdc7a1c696ff37878039f1e1d83dea28a2bfa`，仍为 fresh seed3、
4096 environments × 1001 updates、guidance `0`、同一 v4rg 动作/bank/exam/零摩擦 plant。四个既有机制轴
固定为 `free_wrist_vel_mimic=false`、`motion_scale_in_window=1.0`、`base_decel_weight=0.0`、
`post_swing_start_prob=0.25`；唯一新增 treatment 是
`joint_velocity_limit_hinge_weight=-5.0`、margin `0.85`。`-5` 下单个关节恰到 limit 的惩罚约
`0.00363`，31 关节全到 limit 约 `0.1125`，小于 14/10/5 主击球 shaping，且只作用于最后 15% 速度
余量。首格不同时扫描 margin，也不买第二 seed。

`+200` 只作方向筛：要求 checkpoint finite/lineage/contract、qdot reward 实际非零且 finite，并比较
末 21 updates 与前 21 updates 的 limit-tail/关节最大速度趋势；同时 completion/composite 相对既有 fresh-C
方向对照不得下降超过 5 个百分点，pre-strike fall 不得恶化超过 2 个百分点。由于旧 control source 没有
qdot contract 字段，这一格只能 screen；若有正向信号，下一空槽必须在同一 `a6ccdc7` source、同 seed 跑
weight `0` 的 exact matched control 后才允许作因果采用判断。

## `+500` 方向早判

五条 `model_500.pt` 均 filename=embedded iteration、finite、schema-3 hard-contract SHA 与 fresh lineage
通过，trainer 无 fatal。所有数字均为 TensorBoard updates `480–500` 的 21 点均值，不使用单点：

- V1 free-wrist：completion `0.692`，pre/post fall `0.203/0.065`，但 position pass 只有 `0.113`；是平衡
  改善与击球位置退化的混合结果，保到 1000，不复制 seed。
- V2 quarter-window：completion `0.176`、pre-fall `0.751`、composite `0`；单独机制判为
  `eligible-to-replace`，不复制 seed。
- V1+V2：composite `0.0893`、normal pass `0.268`、击球误差约 `1.05 cm / 0.0946 m/s / 6.96°`，是当前
  唯一强击球质量信号；但 completion `0.391`、pre-fall `0.616`，必须保到 1000 看平衡债是否收敛。
- base-decel：base speed 比 matched control 低 `13.6%`，completion `0.648`，但 height/upright/position
  退化；保到 1000，不复制 seed。
- post-swing `0.5`：completion `0.698`、pre-fall `0.192`、base speed `0.0827 m/s`，但 normal/composite
  仍为零；缺 realized-start numerator/denominator，暂不能证明 treatment 的实际覆盖率。

matched control 同窗 completion `0.607`、pre/post fall `0.293/0.140`、base speed `0.1133 m/s`。因此本轮
不是“全部失败”，而是显出击球精度、覆盖和恢复平衡之间的交互；下一版 harness 必须把每个机制的
activation numerator/denominator 作为必填合同。

### Pod1 资源移交边界

按 Franco 2026-07-14 的冲刺安排，Pod1 从 19:00 CST 起完全留给 Yikang。V1、V2、V1+V2 只对其已记录
PGID `1895946/1896608/1897260` 发出 `TERM`，自然收口于 iter `792/782/743`；三条最新均为
`model_700.pt`，claim、launch log 和真实 RSL log directory 均保留，未发 `KILL`。这不是行为淘汰：其
`+500` 结论仍有效，终档不足 1000 的边界必须保留。后续本卷只在 Pod2 调度，机器合同为
`dispatch_pods: [pod2]`。

## qdot attempt-1 的基础设施超时

qdot attempt-1 在 Pod2 GPU2 创建唯一 schema-1 claim，PID=PGID `323083`，但日志停在 A3 URDF importer
的已知 axis warning 后，900 秒内没有第一条 `Learning iteration`、没有 hard contract 或 checkpoint。
launcher 按合同只终止该 PGID并返回 rc124；其余五臂未受影响。新旧 source 的 ignored A3 tree 均为同一
46 files / 15,378,264 bytes，逐相对路径与大小一致；因此这不能算 qdot reward 的科学失败，也不能复用
旧 namespace。完全相同 recipe/source/seed/budget 的 `fresh_c_qdot_limit_hinge_w5_retry_v2` 是唯一允许的
基础设施重试，并将首次使用 schema-2 canonical claim 与 no-Kit Hydra compose P0 harness。

retry-v2 的运行复核：PID=PGID `326576`，queue claim digest
`3910e3e20f44bad3871cd9c9fb4d0e024cbe5d9f48a8d1fc336d8082c4be8fb6`；外层 claim 的 96 项 argv 与
`/proc` 逐项一致，真实 RSL directory 为
`2026-07-14_11-05-03_phase1_fresh_c_qdot_hinge_w5_seed3_retry_v2_20260714`。到 iter `79` 时 fatal `0`；
`model_0.pt` 的 76 个 tensor / 1,762,715 个浮点元素全部 finite，内嵌 iter `0`，hard contract、fresh
lineage 与 claim SHA 全匹配。这里只证明 harness 与训练启动闭合；weight `0` 的同-source 匹配对照和
`+200/+500/+1000` 早判仍缺。

同-source weight `0` 对照现已在结果前 machine-preregister：source
`a6ccdc7a1c696ff37878039f1e1d83dea28a2bfa`、seed `3`、4096 environments、1001 updates、同
motion/bank/exam/plant 与四个既有机制轴；唯一 qdot 差异为 treatment `-5.0`、control `0.0`，margin
同为 `0.85`。control run name 是
`phase1_fresh_c_qdot_hinge_control_seed3_20260714`，只允许 Pod2 dispatch；它不是第二 seed。

control attempt-1 随后也在动态 URDF import 返回前停住：PID=PGID `327651`，iter `0`，无 hard contract/
checkpoint，日志在最后一条 malformed-axis parser error 后不再增长。成功 treatment 含完全相同的
`libGLU.so.1` warning 与 12 条 parser error，但下一行完成 scene creation，因此这些 warning 不是差异根因，
qdot weight 在 reward 构造前也不可能致因。身份复核后 exact TERM 无响应，最终只对同 PGID KILL；claim/
log 全保留。完全不改 recipe/source/seed/GPU 的
`phase1_fresh_c_qdot_hinge_control_seed3_retry_v2_20260714` 是唯一允许的 fresh namespace retry；若再次停在
同 phase，则停止自动 retry并转预转换 USD/boot harness 根因线。

### qdot 同源配对 `+500` 早判

unchanged control retry-v2 随后成功启动。两边 `model_500.pt` 均为 filename=embedded iter `500`、76 tensors/
1,762,717 elements 全 finite、fresh lineage `1`；相邻 schema-3 contract SHA 与 schema-2 queue claim 均匹配，
hard contract 只有预期的 enabled/weight `false,0` 对 `true,-5` 两处差异。TensorBoard updates `480–500`
的 21 点均值显示 treatment 相对 control：raw/arm/leg qdot max 分别下降 `16.4%/10.9%/18.0%`，near-limit
fraction 下降 `20.1%`，torque saturation 下降 `35.5%`；pre/post fall 从 `0.294/0.140` 降到
`0.209/0.0616`，completion 从 `0.607` 升到 `0.688`。机制 Reward 在 treatment `21/21` 非零且 finite，
control `21/21` 为零。

代价同样明确：position pass 从 `0.418` 降到 `0.107`，position mean error 从 `0.219 m` 升到 `0.311 m`；
两边 exact composite 均为零。日志没有 activation numerator/denominator、normalized exceedance 或 per-joint
tail，因此结论只能是“qdot/平衡方向改善但击球位置明显退化”的 mixed signal；不采用、不买第二 seed，
保留 terminal checkpoint 后再用 immutable judge 判断，且未过主效应前不启动 `V1+V2 × qdot` 交互。

control 随后自然完成到 `model_1000.pt` 并退出；模型 SHA-256 为
`b667286972bed9400f26f9b6ce1fe5c6dc093ee4b2305bdbd7ec08dc75612cb9`，filename/embedded iteration
均为 `1000`，76 tensors / 1,762,717 elements 全 finite，fresh lineage `1`。内嵌/相邻 schema-3
contract SHA 均为 `25faa6f59d8d9a5eb2e4b57f2fc827422529d0264ff93b43639abd889fb9da12`，claim 为
`c73ac441ad6ea8ff64d4caa86376af3976b97d844c11a8278349ba788198a959`，正式 failure regex 为 `0`。
对应 treatment `model_1000.pt` 也已复核：SHA-256
`8814debb38b31cc1f311567945eaef7d993170332c89241831811e1fcd0e556e`，iter `1000`、76 tensors /
1,762,717 elements finite、fresh lineage `1`、schema-3 contract
`3f6a532adeb1e3cefce5fa16745ab879fa204d8df5d47909d4c62839abe79091` 与 claim
`3910e3e20f44bad3871cd9c9fb4d0e024cbe5d9f48a8d1fc336d8082c4be8fb6` 均 exact，fatal `0`。

更重要的是，updates `980–1000` 的 21 点均值推翻了 `+500` 时“精度代价持续存在”的外推。terminal
treatment/control 分别为：position pass `0.878/0.593`，position error `0.0474/0.0962 m`，signed
composite `0.310/0.146`，virtual return `0.454/0.265`；pre-fall `0.0721/0.0668`、post-fall
`0.0235/0.0243`、completion `0.798/0.804`，平衡/完成率基本持平。arm torque saturation 的惩罚幅度也从
`0.001257` 降到 `0.001054`。因此 `-5` 从“考虑低剂量补救”改判为 **晚熟候选**：不启动低剂量扫描、
不买第二 seed、不做 `V1+V2 × qdot` 交互，先用两份 terminal checkpoint 跑同题 immutable
MuJoCo/vendor judge。训练内曲线本身仍不构成采用或部署证据。

## 发射 harness P0 收紧（尚未产生新 run）

在 qdot 格发射前，队列入口补上四个反复出错的执行合同，但没有连接 Pod 或改动现有五条 trainer：

- recipe 先编译为单义 Hydra key 集；`key/+key/++key` 视为同一个 key，重复 key、harness 自己生成的
  seed/预算/run/motion/bank/device/claim key、Hydra flag、删除和 interpolation 都在 SSH 前拒绝；
- 整份 YAML 的 `run_dir` 全局唯一，ready run 不能放进 ready source checkout；远端只允许原子创建一个
  从未存在的 run directory，不能覆盖旧 log/state；
- standalone doctor 与 launch 内置 doctor 都用真实最终 override 向量运行
  `train.py --cfg job --resolve`，且位于 claim 前；该路径只做 Hydra compose，不启动 Kit；
- canonical claim content 绑定 source、caller argv、run name、seed/预算/milestones、motion/bank/exam identity
  和 Pod/GPU；其 digest 自动成为真实 argv 的 `training_launch_claim_sha256`，claim 同时保存完整执行 argv。

focused 回归为 `19 passed`，并继续覆盖 active queue。此处只关闭发射前 source-contract 缺口：没有新增
claim、checkpoint、机制成绩或 qdot runtime 证据，G05 继续 `Partial`。

## Pod2-only pre-probe 机器处置

2026-07-14 的 pre-probe 清单把历史可运行行全部终态化：qdot `-5/0` 两行按已验证的自然
`model_1000` 终档标为 `complete`；V1/V2/V1+V2 按 Pod1 资源移交的 `792/782/743` 收口标为
`rejected`，明确不是行为失败；base-decel/post-swing 只保留已验证的 `model_500` screen，也以
“activation closed、非行为失败、不得重发该 namespace”终结。新的 conditional P1 pair 仍 blocked，source
改绑 `main@c7e1a90`，control/treatment 分别优先 Pod2 GPU1/GPU2。队列顶层发射闩为 false；本次没有连接
Pod、没有 probe 或科学训练结果。
