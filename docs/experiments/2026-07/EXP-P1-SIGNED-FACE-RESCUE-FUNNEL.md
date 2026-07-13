# EXP-P1-SIGNED-FACE-RESCUE-FUNNEL：有符号拍面修复后的单-seed 机制漏斗

状态：`v8_abc_terminal_d_second_precontract_boot_timeout_retry_stopped`
证据等级：E3 partial（v8 的 A/B/C 串行前序已终档；第四格 D 在 hard contract 前再次 boot timeout）
人类负责人：franco  
执行者：Codex  
全局优先级：只继承 [`NOW` 队列第 1 项](../../NOW.md#统一工作队列唯一优先级账本)，本页不另建队列。

术语统一见[术语与人话对照](../../DEFINITIONS.md)。这里的实验臂是一个训练配置，不是机器人的手臂；
`seed` 是随机初始化/采样种子；`checkpoint` 是训练中间保存的模型。

## 问题与第一性原理

旧零摩擦候选在正手 signed-face（保留拍面正反号的判分）上长期为零。训练中的指数拍面 Reward 在
约 `170°` 误差处几乎没有梯度，因此只修 evaluator 不会自动把 policy 从反面局部最优拉回来。
需要区分两个问题：

1. 旧 checkpoint 在诚实 telemetry 下继续训练，是否会自行恢复；
2. 默认关闭的线性拍面角度引导是否同时能救热启动，并避免从零训练再次进入反面局部最优。

首轮不需要四个 seed。机制主效应尚未成立前，seed 复制只会更精确地估计一个失败配方。最小高信息
设计是同一 seed 的“初始化方式 × 线性引导”`2×2`。

## 假设

- H1：只修有符号量尺、不开线性引导，热启动臂不会在 1000 次更新内明显脱离正手约 `170°` 的死区。
- H2：线性引导权重 `-0.4`、角度截断上限 `pi` 能给反面状态非零梯度；相对匹配对照，它会降低
  正手 signed-face 误差并提高正手有符号拍面通过率。
- H3：若 H2 只在热启动成立而 fresh 不成立，最快演示路径可继续热启动，但它不能成为 fresh baseline；
  若 fresh 也成立，才值得购买第二个 seed。

## 首轮四个机制单元

L2 四格固定：seed 3、`v4rg_runtime_order_v3` 动作对、同一 schema-3 train bank、零关节摩擦 plant、
179 维观测、31 维动作、PPO 配置、4096 environments、checkpoint cadence，并只跑到相对 checkpoint
`+1000`。L1 只把环境数/预算缩成 `512 × 25 update` 来验证同一机制的发射、合同和终档完整性。

| 单元 | 人话名称 | 初始化 | 唯一机制变化 |
| --- | --- | --- | --- |
| A | 热启动诚实对照 | 历史 `SZ` seed3 的已审计强 checkpoint | 线性拍面引导关闭 |
| B | 热启动拍面救援 | 与 A 完全相同的父 checkpoint | 线性拍面引导 `-0.4`，角度上限 `pi` |
| C | 从零诚实对照 | seed 3 从零初始化 | 线性拍面引导关闭 |
| D | 从零拍面预防 | 与 C 相同 seed 从零初始化 | 线性拍面引导 `-0.4`，角度上限 `pi` |

A/B 的父 checkpoint 路径、SHA、嵌入迭代、相邻 training-contract SHA 必须在 machine prereg 中逐项
冻结。C/D 的随机源和完整 launch argv 必须逐字节相同；除预注册的引导字段和必须唯一的 `run_name`
外不得漂移。

## 执行漏斗与 GPU 预算

1. **源码门：** signed-face 的 `n/-n` 负控、NumPy/MuJoCo/Torch 一致性和全回归进入 `main`；source
   commit 未知或工作树不 clean 时禁止启动。
2. **L1 机制冒烟：** 四单元各跑 `512 env × 25 iter`。必须在日志中看到引导 on/off 的实际 applied
   值、`theta_max=pi`、有符号正反手指标、finite checkpoint 与相邻合同。任何一项缺失都不上 L2。
3. **L2 单-seed canary：** 一张 RTX 5090 四槽并发，各跑到相对 checkpoint `+1000`；按 Pod 级
   Kit lock 错峰启动。fresh 用 `max_iterations=1001`，保证 0 起数的 runner 真正写出
   `model_1000.pt`；热启动也用 1001 次调用预算，并把偏移加到父 checkpoint 迭代号。保存相对
   `+200/+500/+1000`，不跑旧 17000-iteration terminal，不增发 seed。
4. **判读：** `+200/+500` 只看训练内 paired 曲线和 hard failure；`+1000` 才在同一 immutable 小卷上
   做每侧固定题量的方向筛。所有单元到 1000 自然结束，不把小卷当正式晋级分。
5. **L3 解锁：** 只有 B 相对 A 或 D 相对 C 在正手 signed-face、较差侧综合命中和安全三者都不退化，
   才给该胜者**连同匹配对照**补第二个 seed。没有第二 seed 复现，不买第三、第四 seed。

其余 GPU 优先给已过各自离线安全门的 Franco 五动作与横移老师、同卷导出/评估或 `NOW` 队首的其他
独立机制；v12 只保留为 Jiayi 路线的后排代表对照。不得为了显示利用率而复制 A–D；没有合法输入时
允许空闲。

## 指标与决策规则

每个 checkpoint 全量记录：

- 正手/反手 signed normal error、normal-pass、position-pass、velocity-pass 与 composite；
- 修正后 virtual-return telemetry；它只作诊断，不能替代 signed composite；
- root fall、guard reset、异常接触、KL、value loss、policy std、NaN/Inf/OOM/Traceback；
- checkpoint 文件名迭代 = 嵌入迭代、所有浮点 tensor finite、checkpoint ↔ 相邻 hard-contract SHA、
  source commit、launch argv、父 checkpoint lineage。

硬失败（合同/SHA 漂移、非有限值、开关未 applied、错误 exactness、crash）立即保留日志并停止本臂；
只允许在配方和合同不变、根因明确且安全时精确 PGID 重试。

L2 的“值得买第二 seed”必须同时满足：

1. 引导臂相对匹配对照在相对 `+500` 和 `+1000` 两个点的正手 signed normal error 都更低，且 `+1000` 的
   改善不少于 `20°`；
2. `+1000` 小卷正手 signed composite 至少从零变为非零，反手和 position/velocity 较差侧不出现
   超过 `10` 个百分点的绝对退化；
3. root fall 不增加，且没有新的 guard-reset/异常接触集中失败。

这些阈值只决定是否购买复现，不是 accepted-baseline 阈值。L3 后仍须 q50、fresh stability、厂商
MuJoCo Gate3/Gate3B 和连续卷；A/B 的热启动结果永远不能洗成 fresh 证据。

## 2026-07-13 machine prereg

signed-face scorer 修复已进入训练源码 commit
`882fea4285f0cf9a97ba79d79ae8af31d26ea1ed`。机器配置
[`phase1_signed_face_rescue_funnel_prereg_20260713.json`](../../../configs/phase1_signed_face_rescue_funnel_prereg_20260713.json)
和 fail-closed launcher
[`run_phase1_signed_face_rescue_funnel.py`](../../../scripts/run_phase1_signed_face_rescue_funnel.py)
已物化；操作真源见
[运行手册](../../operations/run_phase1_signed_face_rescue_funnel.md)。最终文件 SHA 在运行手册中逐项冻结。

静态合同把 L1 固定为同卡四格 `512 env × 25 update`，L2 设计固定为
`4096 env × 1001 update`；四格全部 seed 3，热启动里程碑为 `14000/14300/14800`，fresh 为
`200/500/1000`。v6 launcher + rebind focused 回归为 `32 passed`，覆盖重复/错误 seed、配方漂移、hot/fresh lineage
洗白、未注册 hard-contract key、非零 friction、旧 face pairing、伪造/缺格 activation、半写
no-clobber claim、缺失 Git checkout、未冻结 paper 时的 L2 启动和自动 judge 等拒绝路径。这是 E1
源码证据，不是 Isaac 启动或学习结果。

### v1 runtime preflight 拒绝与 v2 修正

Pod1 的 v1 `validate` 在创建任何 run claim 前拒绝父 checkpoint，报“non-finite or no floating
tensors”。只读诊断证明父文件 SHA 未变，递归扫描为 `74` 个浮点 tensor、`1,762,715` 个浮点元素、
nonfinite `0`；真正根因是 v1 只扫描 checkpoint 顶层，并把 runner 实际写在 `infos` 字典里的
schema/合同 SHA/lineage 当成顶层字段读取。v1 生产副本保留在 `control/v1`，没有训练或半写 run。

v2 把 manifest/control/activation 路径升级到 `...-v2`/`control/v2`，递归遍历嵌套 dict/list/tuple 的
浮点 tensor，并明确要求 provenance 来自 `checkpoint["infos"]`。缺 `infos`、字段不符、非有限或零
浮点 tensor 仍 fail closed。配方、四格、seed、预算与 L2 blocker 均未改变。

### v2 pre-learning 环境失败与 v3 修正

v2 checkpoint/runtime `validate` 通过后，A 格 Kit 已启动但在第一次 learning iteration 和 hard-contract
marker 前以 `ModuleNotFoundError: whole_body_tracking` 退出。launcher 已先原子创建 A 的 run claim；该
目录、日志、PID=PGID state 和 launch contract 原样保留，v2 按 no-clobber 规则不重试，也没有 B/C/D
claim。当时 GPU 随进程退出恢复为空。

根因是 v2 只检查 `/workspace/codexschema/env.sh` 存在，却没有把 exact detached worktree 的
source-first Python 环境传给 child。直接 source 该机器文件也不合法，因为它把 `HOPE_WBT` 固定到旧
`6d93bcb` checkout。v3 使用新 control 与新四格 run name，绑定 tracked `setup_train_env.sh` SHA，拒绝
untracked `setup_train_env.local.sh`，从 exact `882fea4` 和 reviewed IsaacLab root 构造确定性环境 SHA
`ddaa0eff...d743`；runtime `validate` 必须在 claim 前解析 `whole_body_tracking` 到该 exact worktree。
训练配方、seed、预算、输入和 L2 blocker 仍不变。

### v3 Kit 前 import 假拒绝与 v4 修正

v3 的确定性环境 SHA 和 source-first 路径已经正确，但 preflight 为证明来源而真正 import 了
`whole_body_tracking`。该包会继续导入 IsaacLab/`omni.kit`，而 `omni.kit` 只有在 `SimulationApp`
启动后才合法，因此 v3 在 claim 前得到预期的 Kit 前 import 错误。这不是 child 训练失败，也没有产生
新 run。v4 保留 `control/v3`，用 `importlib.util.find_spec` 只解析 module origin、不执行包；正式 import
仍由 locked Kit boot 在创建 `SimulationApp` 后完成。

### v4 ignored A3 资产缺失与 v5 修正

v4 已进入 `SimulationApp` 并成功加载项目包，但创建 scene 时发现 detached worktree 缺
`assets/agibot_a3/urdf/model.urdf`。该目录被仓库 `.gitignore` 排除，正常 `git worktree add` 不会复制；
因此 v4 A 仍在第一次 learning iteration/hard-contract marker 前退出，B/C/D 未创建，失败证据保留。

v5 从 clean exact `6d93bcb` runtime checkout 恢复完整 A3 ignored tree 到 exact `882fea4` worktree，
同时冻结 restore checkout/path 与 target path：`46` files、`15,378,264` file bytes、canonical tree SHA
`0137f59b...26c6`。preflight 同时重算两棵树，拒绝 symlink、特殊/额外/缺失文件，并要求 target 确实被
Git ignore、两个 checkout 均 clean。资产恢复不改变训练配方、源码 commit、seed、预算或 exactness。

### v5 旧题库物理合同拒绝与严格重绑定

v5 已越过环境、Kit、A3 资产和 scene 构建，但在第一次 learning iteration、hard-contract marker 和任何
checkpoint 之前被 schema-3 loader 正确拒绝。旧 train bank 绑定的 `virtual_ball.py` SHA 是
`3dc52373...5ed4`，目标 `882fea4` 源码是 `14113de4...3c8`；不能用
`question_bank_allow_legacy=true` 绕过。A 的 claim/log 保留，B/C/D 没有创建；当时 trainer 在异常清理中
仍占 Pod1 GPU0，但没有发生学习，后续只允许按记录的精确 PGID 清理。

重绑定不是重算题目，也不是“声明兼容”。machine prereg
[`phase1_signed_face_bank_rebind_prereg_20260713.json`](../../../configs/phase1_signed_face_bank_rebind_prereg_20260713.json)
和 consumer
[`rebind_stage1_question_bank_physics_contract.py`](../../../scripts/rebind_stage1_question_bank_physics_contract.py)
要求：七个物理合同文件只允许 `virtual_ball.py` 改动；冻结 Git diff 与新增
`signed_face_hemisphere` 的源码片段 SHA，移除该唯一新函数后旧/新 executable AST 必须完全相同；题库生成器和
loader 在两个 commit 间逐字节相同；全部非 metadata 数组保持 key/order、dtype、shape 和 C-order raw
bytes SHA；metadata 只能改四个 leaf。发布前还要在同一目标 Torch runtime 对 1481 道 train 题逐 tensor
比较旧/新 contact 与 flight 输出原始 bytes，重跑 landing/net 门，并以 `allow_legacy=false` 同时验证
schema、split、source family 和 exact motion/frame/phase 合同。

输出使用新 no-clobber 目录，旧 bank 不覆盖；先生成 bank，再写 completion report。新 train family SHA
预注册为 `9603a178...9db`。该 train bank 即使通过也只解除 L1 发射阻塞；旧 exam 的 family SHA 会与它
不同，所以在对应 exam bank 用同样证据完成 runtime 重绑定或重新生成之前，不能授权 L2 exact judge，
也不能把这次重绑定冒充 signed-directional paper。exam 严格 rebind 已另行完成 E2 runtime 数据门，
但新 bank 绑定的 schedule/paper activation 尚未物化，
见 [`EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND`](EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)。

v1 的 no-write Pod preflight 又抓到一个跨 Python 版本的证据编码问题：`ast.dump` 在本机与 Pod Python
小版本间字段不同，导致相同 helper 源码的冻结 AST SHA 不同。v1 未创建输出 root，control 和错误原样
保留。v2 改为冻结跨版本稳定的 helper 原始源码片段 SHA；“在同一执行 Python 中移除 helper 后旧/新
AST 相等”仍保留，因此没有放宽源码门，也没有改变题库、输出语义或训练配方。

v2 随后在 Pod1 目标 Python `3.10.18`、NumPy `1.26.4`、Torch `2.7.0+cu128` 正式发布：新 bank
`s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz` 为 `215,715` bytes，SHA
`3a9d8851...5b71`；report-last SHA `9fffed03...bb37`，canonical content SHA
`3ea60706...a32d`。24 个非 metadata 数组未变；正手 `757/757`、反手 `724/724` 的旧/新 contact/flight
所有 tensor raw bytes 相同，landing/net 全过。最大落点误差正/反手为 `0.004372/0.004531 m`，最小过网
余量为 `0.273830/0.341401 m`。目标 physics SHA 为 `09dfe899...afb95`，新 train family 为
`9603a178...a9db`；source worktree 复核后仍 clean exact `882fea4`。

v6 manifest 不只绑定 bank 文件：launcher 解析 report 的 content SHA、target commit、rebind
manifest/consumer SHA、24 数组、四-leaf metadata、exact motion gate 和两侧 replay，任一缺失都在 claim
前 fail closed。旧 parent 的 `question_bank` 是唯一允许变化的共同 hard-contract 字段，精确从旧
`2da2bd.../b21c16...` 过渡到新 `3a9d88.../9603a1...`；其他全部共同字段仍逐值相同。A/B 因该显式
变化继续是 lineage `0`，C/D fresh 才是 `1`。

### epoch-1 v6 实际 L1 与 D-only v6r1

实际 Pod1 L1 使用后续集成出的 clean source commit
`50c49e58a9413ec6ac1c3ed2565d9a78acdb5e64`，它把 unmasked command observation 的 provenance
epoch 与上述 bank rebind 合到同一源码。该 commit 尚不在当前仓库对象库；exact 恢复 bundle SHA 为
`2a794e2c...0a39e`，运行时外部 v6 manifest/launcher SHA 分别为
`97779cee...eebf2` / `9463f228...85052`。这些是忽略资产/外部控制依赖，不得用仓库中旧
`882fea4` v6 文件假冒，恢复方法见[本地同步手册](../../operations/setup_local_sync.md)。这是一个必须
显式处理的 **同 manifest ID、不同 source/control SHA collision**：两套都叫
`phase1-signed-face-rescue-single-seed-funnel-20260713-v6`，但实际 epoch-1 绑定
`50c49e5` + `97779cee...eebf2` / `9463f228...85052`，当前 tracked 文件绑定 `882fea4` 且 bytes 不同；
consumer 必须按 SHA 选择前者，不能只按名字选控制。

v6 的 A/B/C 已到预注册终档：A/B 为 `model_13824.pt`、lineage `0/0`，C 为 `model_24.pt`、lineage
`1`；三格 checkpoint SHA 分别为 `a1fbb766...68d2e`、`c73f59dc...20f3d`、
`5ce4de67...66b11`，三格 emitted hard-contract SHA 均为 `dfc583d4...888a5`。原始 A/B/C finite/lineage
审计与 D 空 run-dir 行共同冻结在 `l1_checkpoint_audit.jsonl`，SHA `62076758...d354`。B 在终档
checkpoint 已稳定后仍卡住，
只对其记录的 PGID `1758211` 做过一次精确终止；action 证据 SHA `cf619541...dcafe`，不是 broad kill。
D 的 outer launch contract/state/log SHA 分别为 `f6dd2fd2...e0b63`、`4e1ab699...f350`、
`baa02f52...3610`；PID `1759428` 已不存在，`runtime_verified.json` 与任何 `model_*.pt` 都未产生。
timeout 诊断 SHA `ae7de7a3...b5a0c` 表明它只走到 Kit/scene boot，尚未出现 hard-contract marker 或
learning iteration。因此不能删除旧 D claim 后原名重跑，也不能把 A/B/C 三格写成完整 L1。

新 machine prereg
[`phase1_signed_face_d_retry_prereg_20260713.json`](../../../configs/phase1_signed_face_d_retry_prereg_20260713.json)
与 consumer
[`run_phase1_signed_face_d_retry.py`](../../../scripts/run_phase1_signed_face_d_retry.py)把这次例外收窄为
[v6r1](../../DEFINITIONS.md) D 单格：加载并校验 exact foreign v6 控制，调用它重建原 D argv，再证明
新 argv **只有一个** `run_name` 项从 `phase1_signed_face_l1_v6_D_fresh_guidance_seed3` 变为
`phase1_signed_face_l1_v6r1_D_fresh_guidance_seed3`。启动前还须复核旧失败三件套、dead PID、无旧
checkpoint、clean exact source、runtime closure、bank/report、GPU0 空和 Kit lock 空；新
`control/v6r1` 全部 no-clobber。Python consumer 没有直接 signal API；但它调用的 frozen
`launch_kit_training_locked.sh` 在 pre-marker boot timeout 时会只对 state 中该隔离 PGID 做
TERM→KILL，这是合同允许的精确 cleanup，不能写成“零 signal”。旧 D claim 不改不删，失败后也不自动
给第二次重试权限。若 wrapper 已返回 ready、但后续 hard-contract/first-iteration 等待超时，trainer
可能仍存活；只允许按 `run.log.launch` 的精确 `pid=pgid` 人工审计，不得自动再发 D。

第二层 no-clobber 直接检查 `50c49e5` checkout 的 frozen RSL-RL log root：任何名字以后缀
`_phase1_signed_face_l1_v6r1_D_fresh_guidance_seed3` 结尾的现存目录、file、symlink 或异常 entry 都在
写 control claim 前 fail closed。这样即使 control 目录被漏拷或从未产生，也不能对同一个 run name
重复发射。终档不 glob “最新目录”，只消费 `runtime_verified.json` 绑定的 exact run dir。

混合 finalizer 只有在新 D 自然记录 `Learning iteration 24/25`、进程退出、`model_24.pt` finite、
lineage `1` 且 hard-contract SHA 仍为 `dfc583d4...888a5` 时，才重新逐项审计旧 A/B/C 终档和 B 的
exact-PGID action，并写一份 A/B/C=`original_v6`、D=`v6r1_single_cell_retry` 的 no-clobber activation。
activation 同时绑定两套 config/launcher SHA 与 old-D→new-D retry lineage；它明确保持
judge/L2/第二 seed/stop-promote 全 false，并记录 consumer 无直接 signal、冻结 wrapper 的 exact-PGID
timeout cleanup policy 以及成功 D 路径未执行该 cleanup。当前只有 E1 工具/测试，v6r1 尚未实际运行。

### v8 独立串行四格与第二次 D pre-contract timeout

后续 foreign v8 不是把 v6/v6r1 只改运行名后继续。它使用 clean source
`72418fff817d2d9beb9f764562b5a28e82a13044`（tree `8f99fe95...7709`）、全新 manifest/launcher
`f786da9f...8029` / `58e798fc...6afa`，并明确 `v6_artifacts_adopted=false`。A/B/C/D 按 terminal barrier
串行发射；A、B、C 前序实际运行并终档，D 是 zero-based ordinal `3` 的第四格。当前小账只完整归档 D
失败和其直接 C 前序 receipt，不冒充对 A/B/C 的新一轮完整重审：C 的 `model_24.pt` SHA 为
`f7b0decb...4f51`，finite/lineage1，hard-contract SHA `dfc583d4...888a5`，且自然退出。

D 于 `2026-07-13T17:22:19Z` 以精确 `PID=PGID=1782834` 启动，900 秒内未出现 hard-contract marker、
`runtime_verified.json`、learning iteration、checkpoint 或训练日志目录；日志也没有
NaN/Inf/Traceback/OOM/malloc/Killed。frozen locked wrapper 只对该 PGID 做合同内 cleanup 并返回 124。
failure/state/launch-contract/run-log SHA 分别为 `0e5bb13b...f98a9`、`80939e6d...c90e`、
`5649884d...de5`、`5b2c91ac...d43e`。机器真源是
[`phase1_signed_face_v8_d_boot_failure_20260714.json`](../../../configs/phase1_signed_face_v8_d_boot_failure_20260714.json)。

这已是继旧 v6 D 之后第二次独立的 D pre-contract Kit boot timeout。没有新的根因证据前，自动换名或
配方不变重试被拒绝；下一步是 boot root-cause，不是继续消耗 GPU。v8 没有四格 activation，L2、judge、
第二 seed、部署和真机仍全部 false。最终只读审计为 Pod1 `0` trainer/worker/judge、三张 GPU 无 compute。

### 父合同扩展边界

父 `model_13800.pt` SHA 冻结为 `478efa8d...d9e6`，嵌入/相邻 hard-contract SHA 均为
`3a3b3d95...b9972`。当前源码已在该旧合同上增加 event timing、target cadence 等不可变字段；因此
A/B **不能** strict exact resume。它们固定为 `checkpoint_allow_contract_mismatch=true` 的显式
inexact representation transfer；launcher 只允许 `question_bank` 按上述 old→new 精确值改变，其余
共同字段逐值相同，且只允许 manifest 列出的 current-only key；后代 lineage 必须为 `0`。C/D 不读
checkpoint，lineage 必须为 `1`。
四格 emitted hard-contract SHA 必须一致。这一处理没有把旧 checkpoint 洗成 fresh 证据。

L1 只是一份 25-update launch-integrity smoke。原 v6 已因 D 的 pre-runtime timeout 不能再用
`finalize-l1`；只有上述 D-only v6r1 自然终档并通过 mixed finalizer，才可能写完整 L1 completion。
**该文件本身仍不能启动 L2**。半写/提前退出格保留证据并阻断自动重试；v6r1 没有信号、broad kill、
judge、部署或真机命令路径。

## 仍未关闭的发射/判卷缺口

- clean detached `50c49e5` epoch-1 训练 worktree和 exact ignored A3 资产已在 Pod1 建立；v1 audit 假拒绝、
  v2/v4/v5 pre-learning 失败与 v3 Kit 前假拒绝均已归档且未覆盖。v5 是旧题库 physics-contract 拒绝，
  没有 learning iteration/checkpoint；v2 rebound train bank/report 已正式发布。v6 A/B/C 已有终档，D
  在 runtime verified 前 boot timeout；当时只预注册了尚未启动的 v6r1 D-only 路径，现已由下述 v8
  独立尝试取代并停止自动重试。
- v6r1 计划已被独立 v8 串行尝试取代；v8 D 再次在合同前 timeout，因此仍无四格
  completion/activation。禁止自动 retry，先做 boot root-cause。
- 相对 `+1000` 的 immutable signed-face directional checkpoint paper 的 exact schedule/path/SHA 尚未
  冻结。exam-v1 rebind 已完成 E2 真实 371 题 replay/output，但基于新 bank 的 schedule/paper activation
  仍没有；
  manifest 明确 `l2.launch_authorized=false`。必须另发 reviewed v7 paper activation 后才能启动 L2。
  当前 launcher 也不启动 judge、不能自动晋级或购买第二 seed。

当前不授权再次启动原 v6 D、v6r1 或 v8 D。只有 boot 根因闭环并形成新的内容绑定合同后，才可评审新的
versioned attempt；也不授权 L2、judge、第二 seed、部署或真机。
