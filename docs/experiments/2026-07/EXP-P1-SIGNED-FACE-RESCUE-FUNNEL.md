# EXP-P1-SIGNED-FACE-RESCUE-FUNNEL：有符号拍面修复后的单-seed 机制漏斗

状态：`v6_v8_d_same_table_to_physx_boundary_root_cause_open_retry_stopped`
证据等级：E3 partial（两次 D 都停在同一 table USD→PhysX 边界；根因尚未证明、retry 仍停止）
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

### epoch-1 v6 实际 L1 与 v6r1 validator 失效

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

当日新增的 [v6r1](../../DEFINITIONS.md) D-only config/consumer 最终只安装到 `control/v6r1`，没有
claim、runtime、training 或 signal。首次真实 `validate` 在任何写入前发现其合同不可能成立：冻结的
checkpoint audit 对 D 明确记录 `run_dirs=[]`，真实 would-be training path 也不存在，但 validator 却
要求该 path 必须是 directory。团队没有伪造目录，也没有启动 v6r1；旧文件按 exact SHA 留作
superseded evidence。

新
[`phase1_signed_face_d_retry_prereg_v6r2_20260714.json`](../../../configs/phase1_signed_face_d_retry_prereg_v6r2_20260714.json)
与
[`validate_phase1_signed_face_d_retry_v6r2.py`](../../../scripts/validate_phase1_signed_face_d_retry_v6r2.py)
只修正这条历史 source contract：旧 expected path 必须 absent，任何 directory、regular file、symlink、
special 或 unstatable entry 都是冲突。v6r2 只支持 `static-validate`，没有 runtime preflight、命令重建、
进程检查、launch、signal 或 mixed finalizer；`validate/plan/launch/finalize` 均 fail closed。它绑定 v6r1
exact config/consumer SHA、D audit `run_dirs=[]` 与 foreign-v8 terminal receipt，但没有生成 retry claim，
也不能作为补跑入口。后续尝试必须先闭环 boot 根因，再另建 v6r3-or-later preregistration。

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

### 只读根因审计：边界已收窄，根因仍开放

三次低频只读 SSH 的报告、完整八个 run/Kit 日志归档、inventory 和 system snapshot 已由
[`phase1_signed_face_boot_root_cause_results_20260714.json`](../../../configs/phase1_signed_face_boot_root_cause_results_20260714.json)
逐字节绑定；四份外层证据 SHA 分别为 `b54cb06a...76af`、`b1893512...c1d`、
`29dabc9e...3a6`、`02b78e2d...e25`。审计没有远端写入、signal、进程启动或真机动作。

事实层结论比“卡在 Starting simulation”更窄：v6/v8 D 的 Kit log 都以加载
`ping_pong_table_urdf_base.usd` 为最后一行，elapsed 分别为 `11.649/10.505 s`，均未看到随后
`Physics using context`。两 worktree 的 table USD 都是 regular file，均为 `683,433` bytes、SHA
`c6fc99a8...996`。相邻 C 用同一份各自 table bytes 从该行到 PhysX 只需 `2.339/3.031 s`，并都训练到
`24/25`；v8 D 还在 C clean shutdown 后 `44 s` 才启动，因此并发的 A/B/C 不是复现所必需。两份 D
argv 去掉 v8 新增的 launch-claim provenance 并规范化 versioned `run_name` 后逐 token 相同；这不是
PPO、signed-face Reward 或已进入学习后的失败。

审计时三张 GPU 均为 `0%/0 MiB`、host available memory `976 GiB`、`/workspace` free `362 GiB`、
`/dev/shm` free `201 GiB`，所以**持续容量饱和**不符合事后快照；但这不是故障当时的采样，不能排除
瞬时 driver/filesystem stall。v6 B、v6 D、v8 D 的 PID 对应 Carbonite shared-memory 残留只证明异常或
精确 PGID cleanup 后存在相关性，不证明其导致卡住。`dmesg` 因权限被拒，journal 无匹配也不能证明
历史 kernel 事件不存在。

当前最强但仍属推断的类别是“第四个 Kit process 或累积 Pod state 在 table composition→PhysX 交界
触发 hang”。新的
[`phase1_signed_face_boot_diagnostic_prereg_20260714.json`](../../../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)
只登记一个 design-only `2×2`：D-first 对 ordinal-4，以及 host IPC 对每进程 private IPC namespace；
后者只能靠 namespace teardown 回收本进程 Carbonite 对象，禁止 unlink host shared memory。probe 必须
scene-boot-only，在第一条 PhysX marker 后退出，不得进入 learning/checkpoint。该设计没有 launcher，
所有 execution/Pod/process/signal/training/retry/judge/deploy/hardware 权限均为 false；须另行审阅源码、
静态测试和明确授权后才可运行。

本次账本验证命令为
`pytest -q tests/test_phase1_signed_face_boot_root_cause_ledger.py`（`8 passed`，含本机恢复证据的实际
size/SHA 与 tar 47-entry 复算）以及最新 main 基线上的 `pytest -q tests`（`722 passed, 9 skipped`）。裸 `pytest -q`
会收集 vendor/Isaac 子树并因本机缺 `zmq`、`torch`、`hydra` 在 collection 阶段失败；这不是本次账本
回归失败，也没有以安装依赖或修改环境掩盖。

### 父合同扩展边界

父 `model_13800.pt` SHA 冻结为 `478efa8d...d9e6`，嵌入/相邻 hard-contract SHA 均为
`3a3b3d95...b9972`。当前源码已在该旧合同上增加 event timing、target cadence 等不可变字段；因此
A/B **不能** strict exact resume。它们固定为 `checkpoint_allow_contract_mismatch=true` 的显式
inexact representation transfer；launcher 只允许 `question_bank` 按上述 old→new 精确值改变，其余
共同字段逐值相同，且只允许 manifest 列出的 current-only key；后代 lineage 必须为 `0`。C/D 不读
checkpoint，lineage 必须为 `1`。
四格 emitted hard-contract SHA 必须一致。这一处理没有把旧 checkpoint 洗成 fresh 证据。

L1 只是一份 25-update launch-integrity smoke。原 v6 与 v8 都因第 4 格 D 的 pre-contract timeout 没有
完整 activation；v6r1 已被 validator 矛盾否决，v6r2 又明确没有 runtime/finalizer，所以当前不存在可写
L1 completion 的入口。半写/提前退出格原样保留并阻断自动重试；v6r2 没有 signal、broad kill、judge、
部署或真机命令路径。

## 仍未关闭的发射/判卷缺口

- clean detached `50c49e5` epoch-1 训练 worktree和 exact ignored A3 资产已在 Pod1 建立；v1 audit 假拒绝、
  v2/v4/v5 pre-learning 失败与 v3 Kit 前假拒绝均已归档且未覆盖。v5 是旧题库 physics-contract 拒绝，
  没有 learning iteration/checkpoint；v2 rebound train bank/report 已正式发布。v6 A/B/C 已有终档，D
  在 runtime verified 前 boot timeout；当时只预注册了尚未启动的 v6r1 D-only 路径，现已由下述 v8
  独立尝试取代并停止自动重试。
- v6r1 的 expected-absent validator 错误已由 source-only v6r2 修正；v6r2 明确 NOT LAUNCHED，不能
  validate runtime、plan、launch 或 finalize。v8 D 又在合同前 timeout，因此仍无四格
  completion/activation。禁止自动 retry，先做 boot root-cause。
- 相对 `+1000` 的 immutable signed-face directional checkpoint paper 的 exact schedule/path/SHA 尚未
  冻结。exam-v1 rebind 已完成 E2 真实 371 题 replay/output，但基于新 bank 的 schedule/paper activation
  仍没有；
  manifest 明确 `l2.launch_authorized=false`。必须另发 reviewed v7 paper activation 后才能启动 L2。
  当前 launcher 也不启动 judge、不能自动晋级或购买第二 seed。

当前不授权再次启动原 v6 D、v6r1、v6r2 或 v8 D。只有 boot 根因闭环并形成新的内容绑定合同后，才可
评审新的 versioned attempt；也不授权 L2、judge、第二 seed、部署或真机。

## 2026-07-14 C2/D2 provenance 修正版（C2 terminal；D2 待 v1r2）

旧 v9 `5f691b3400fe3feda1a690675912a97f09e906bb` source 和
`466f8ea935310407f73b95e812bbd5f0a18705b4` control 的只读复核发现一个比 boot ordinal 更基础的
证据缺口：它的 emitted hard contract SHA `dfc583d4...888a5` **没有**包含 positional/signed-face
guidance 的 post-Hydra 权重。v9 虽然把 outer launch claim 写进 checkpoint，但仅靠相邻合同仍不能证明
复制出去的 fresh checkpoint 属于 weight `0.0` 还是 `-0.4`。因此旧 C/D、一次 one-update 诊断和旧
namespace 均不采用；这也不是给原 v6/v8 D 换名自动 retry。

新 source gate `4467d79f1ed425a4263f0caaad2f661e1ec737ad` 在 schema-3 hard contract 中加入
`racket_guidance_reward`：位置与有符号拍面两项都绑定 post-override `weight`、`command_name` 和截断
边界；非法正权重、NaN/bool、错误 command 或角度越界 fail closed。checkpoint `infos` 另写外层原子
launch-claim SHA，该 claim 绑定 manifest/launcher、exact source/critical files、C2/D2 优化配方、seed、
终档迭代和 claim directory inode/device，不进入 scientific hard-contract SHA。训练入口同时移植并
运行时核对 Kit carb/TBB `16/16` 与 `useOmniJob=false`；两条 child 使用完全相同的 source-first
`PYTHONPATH`。

[`signed-face C2/D2`](../../DEFINITIONS.md) 是全新 fresh-only 对照，不是旧四格 completion：C2 的位置/
拍面 guidance 都是 `0`，D2 只把拍面权重改为 `-0.4`；二者 seed3、512 env、25 update、动作、train
bank、zero-friction plant、face179/action31 和 event-timing-off 全相同。按团队“先六卡各一条”的硬调度，
C2 固定 Pod1 GPU1、D2 固定 Pod1 GPU2；各自 GPU 必须在 claim 前为空。一次 invocation 只创建一格，
shared Kit boot lock 仍串行；但 C2 写出 `runtime_verified` 后继续训练，D2 即可在另一卡 boot 并发，无需
等 C2 终档。physical GPU 是绑定进 outer claim 的运营 lane；两条训练 command 都看到 local
`device=cuda:0`，科学配方除 guidance weight 外不变。terminal 必须 finite、`iter=24`、lineage `1`，
同时匹配相邻含权重 hard contract 与外层 claim/source/GPU lane；
成对 finalizer 还要求两份 hard contract 去掉唯一 nested weight 后逐值相同。失败 claim 永不覆盖，
没有自动 retry。

机器 prereg/launcher 的最终 SHA 见运行手册；专项静态/攻击测试
`28 passed`，source launch-claim/thread-cap 测试 `28 passed`，reward/hard-contract override 测试
`58 passed`，合入 `main@da9ba58` 后仓内 `tests/` 为 `822 passed, 10 skipped`。`static-validate` 与 plan rc0。
后续 C2 已实际启动并自然形成 terminal hard contract/checkpoint；D2 尚未 claim。当前裁决更新为：

- **GO（待 root 审阅）：** 只允许按
  [C2/D2 运行手册](../../operations/run_phase1_signed_face_cd_l1.md)用 v1r2 replay 已完成 C2 的冻结证据，
  再购买尚未 claim 的 D2 L1 provenance smoke；
- **NO-GO：** 旧 v9 artifact 采用、同 namespace retry、activation、judge、L2、第二 seed、stop/promote、
  部署或真机。

这个 source gate 不推翻旧四格 activation 缺失，也不证明 signed-face reward 有效；只有两条新 L1 真正
终档后才能说明 provenance 闭合。K100 paper 已在并行 main 工作中物化，但其 paper-only activation
固定 trainer/L2/judge=false；下一版本仍须以 no-clobber consumer 同时绑定 C2/D2 pair result 与该 exact
paper receipt，当前文件不能直接启动 L2。行为结论仍须后续同卷 execution。实验继续 `running/partial`。

### C2 float/int outer false rejection 与不可重跑续接

C2 v1 的 launch contract/state/canonical claim SHA 为 `26bf204d...0e96` / `2bcc5656...beb8` /
`37fe2443...86e5`；natural terminal log/hard-contract/model24 SHA 为 `abffd457...6dc3` /
`83f47ae6...2772` / `dbbc7a28...6f6`。trainer 写出的
`mount_normal_sign_per_clip=[1.0,-1.0]` 与实际 Hydra command 一致，但 v1 verifier 用整数
`[1,-1]` 做 exact-type 深比较，因而在 post-boot 假拒绝。旧 runtime/failure/result 都 absent；这三个
absence 是事实边界，不得通过事后补写把它描述成 v1 runtime verified。

冻结 [`v1r1`](../../DEFINITIONS.md) 虽修复 exact float，却把 source `4467d79` 实际发出的五键 compact
`question_bank` 错当成还应直含第六个 `physics_contract_sha256`，因此对合法 C2 合同精确假拒绝。该
physics SHA 实际只存在于 exact NPZ metadata/source-family contract，不能伪造进 trainer record。
最后一次成功只读快照 `2026-07-13T22:32:07Z` 证明 v1r1 control/evidence/pair、D2 arm/exact run
全部 absent 且没有 write/claim/launch；后续 SSH 状态 unknown。v1/v1r1 bytes 均冻结并禁止运行，历史
absence 也不授权 launch。

新的 [`v1r2`](../../DEFINITIONS.md) manifest/launcher SHA 为 `4e202589...8c638` /
`2b53c865...45a12`。它精确复现 v1r1 错误，只接受实际五键 compact shape，并独立解析 exact NPZ
`meta_json` 绑定 bank file/schema/split/source-family/physics SHA；source-family 内嵌 physics 也必须
一致。伪造第六键、metadata drift，或 v1r1 evidence root/C2 receipt/pair receipt、D2 arm/exact run
任一出现都 fail closed。attestation 进入独立 no-clobber `continuations/v1r2/`，先写并立即 replay
content-bound v1r1 absence receipt，不向 preserved C2 arm 增加文件。

mixed outer-control pair（C2=v1、D2=v1r2）只有在两条 normalized trainer recipe 与 hard contract
都只差 signed-face weight 时才可发布。外部 control 固定为 `scripts/ + configs/` 六文件 mini-tree，
同时绑定 v1r2 与冻结 v1/v1r1 helper/manifest；临时 mini-tree subprocess `static-validate/plan` 通过，
缺文件/扁平/symlink/重复 JSON key 均失败。v1r2 专项攻击测试 `52 passed`；本分支没有连接 Pod、安装 control、写
attestation 或启动 D2。三代聚焦回归为 `111 passed`，受支持的完整仓内 `tests/` 为
`934 passed, 10 skipped`；因此还没有 paired L1 result，更没有 activation/judge/L2/第二 seed/行为结论。

### 2026-07-14 A2/B2 热启动探索 L1 runtime source gate

plan-only commit `c3c60f0` 已由全新 v2 namespace 的 one-shot consumer 闭合，不复用旧 v6 A/B
artifact。A2 固定 Pod1 GPU0、guidance `0.0`；B2 固定 Pod2 GPU0、guidance `-0.4`。两格共用 exact
`model_13800.pt` 父模型、seed3、`512 env × 25 update`、显式零摩擦与同一 train bank；child 预期
`model_13824.pt`、lineage `0`。claim 前必须重验 clean commit/tree、父 checkpoint+hard contract、核心
父→子 diff、输入 SHA、空 GPU 和 exact run absence；失败 namespace 永不重放。source/static/attack
`27 passed`，static/plan rc0；尚未 SSH、安装 control 或启动 trainer，judge/L2/第二 seed/晋级/真机全
false。跨 Pod pair finalizer 还会用实际 GPU0 UUID 验明 Pod，并要求两份 hard contract 的所有
current-only 值都匹配预注册值，完整比较后只允许 signed-face weight 不同。运行真源见
[操作](../../operations/run_phase1_signed_face_a2b2_l1.md)。

## 2026-07-14 第二圈第六机制：真实 qdot 上限尾部惩罚（source only）

这一机制回答的是“policy 是否因为把真实关节速度推到 plant 上限附近而破坏击球/平衡”，不是再做一条
`action_rate` 平滑。新增的 [`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge) 对每个 runtime
关节计算 `abs(qd)/limit`，只惩罚超过 margin 的尾部并对 31 关节取均值。默认 weight `0.0`、margin
`0.85`；启用时 weight 必须非正。关节速度和分母都来自同一个 articulation runtime order，错误顺序、
非 31 关节、零/非有限 limit 或跨 environment limit 漂移全部 fail closed。

若后续物化这一臂，最小 paired 语义是：以同一冻结 `model_13800.pt`、seed3、signed-face guidance
`-0.4`、动作/题库/plant/优化器为共同父项，control 保持 hinge weight `0.0`，treatment 只改变预注册的
一个负 weight；两条都直接从共同父 checkpoint 出发，禁止从 B 的 `+200` child 再续训。weight、margin、
公式、joint-order/limit source 必须同时出现在 hard contract；outer launch claim 还须逐字节绑定 exact
Hydra argv 和 manifest。沿用 `+200/+500/+1000` 早判，先看 limit-tail 下降是否同时保住 signed
composite、root fall 和击球侧速度；单一机制未胜过匹配对照前，不与 recovery reward 混合，也不买第二
seed。

当前只完成 E1 source gate：VirtualBall 默认关闭、Hydra fail-loud override/applied marker、hard-contract
binding 与 qdot-focused `30 passed`。没有 machine prereg、采用的负 weight、Pod runtime、checkpoint、judge、
L2 或晋级授权；本节不能当作排队即发的训练配置。
