# G06 Isaac-To-MuJoCo Parity

Status: Partial (parity procedure operational and used to gate the 2026-07-02 sim-to-real; formal per-checkpoint acceptance thresholds still to be recorded)

## 2026-08-29 当前双端阶段与环境裁决（仍`Partial`）

新机可复现边界现已具体化：Mu的Python3.12.3基础venv有133项tracked lock，Pod fresh resolver精确闭合；
实际run继续由EPA48/RSL3 run-local site掌权。Isaac launcher只接受machine provisioning给出的EULA=`Y`与
privacy=`Y|N`，不替人选择。合法Isaac二进制、split USD、Mu的92个ignored mesh和private凭据仍须外部恢复，
所以这是可执行安装路径而非“纯clone自包含”，G06仍`Partial`。

`255df4a1`的Physical/R06收敛候选已在Pod clean exact CPU套件通过，但真实GPU fixed tape、journal/WAL
逐字对拍与matched profile尚未运行；三卡当前均由只读训练占用，不能同卡混跑或停止旧run换结果。该候选仍
保留每个backend的Scene/plant事实和per-substep事件，只减少同一R06 owner的重复finalize，故通过后也只会
改善实现成本，不会把任务成功变成sim2sim或安全证据。

`observed_at=2026-08-29T11:52:28Z`时，现役Mu已产生`663/294,108` selected contact和597次net crossing
但0 legal landing；Isaac为`0 R03 / 78,072 physical launch`。二者处在不同学习阶段，不能把曲线差异直接
当physics parity。
fixed tape仍把shared initial state、joint order、action decoder与qdes对齐，current Pod runtime身份也通过
独立检查；所以没有“Pod整体装坏”的证据。剩余first-divergence仍属于implicit PhysX drive与显式Mu PD、
friction/contact/integrator响应，需同tape plant字段对签，不能用policy或额外Gate掩盖。

`12:19:19Z`的同分母刷新最终把4× Mu判负：ACK0..623新/旧selected contact=`5/334`，最近10轮四项mimic
也全部更差；但这是同一Mu
backend内的Reward经济反例，不是sim2sim差异。后继只在playback-active行增强paddle kernel，先分别在两端
做exact fixed-action与fresh learning；三卡未自然空闲前保持`未测`。现有snapshot无resume authority，不能
用旧checkpoint跳过fresh对照或把Build4 warm-start混进跨引擎判定。
静态cross-backend消费审计发现`8a57a522`只让Isaac读取playback scale、Mu没有读取，故此前双dry-run不能让它
成为训练源。final exact source=`b0d7d562`已在Mu reward/ledger镜像同一Motion playback事实，且没有扩大reward
ABI；Pod exact Mu=`354 passed,7 skipped`、共享十文件同进程=`367 passed,25 skipped`、launcher/setup=
`98 passed`。final exact目录恢复Mu ignored资产后，两端完整launcher dry-run仍RC0；这关闭代码语义和启动输入
组合，不是双引擎CUDA fixed-action或physics parity。真实双端fixed-action与fresh matched仍等自然空闲GPU，
继续写`未测`。

当前repo部署合同也不等于跨机parity：一次性EULA/隐私setup事实不再被错做成per-run Gate；setup已删除
环境path discovery，GL由caller显式绑定、核SONAME并记录观察SHA。最新`bb0be1a2`从全新exact checkout完成
真实Kit fixed-action：0 done/timeout，action/state SHA与既有基线逐字一致，checkout clean且GPU/lock释放；
但空白机private资产可获得性仍`PARTIAL`。G06不从历史fresh-clone receipt推导跨机器逐位曲线一致。

## 2026-08-29 R8 matched first-divergence（仍`Partial`）

controller修复后的Mu有限学习窗把hard-edge降到`19,816/1,499,136=1.322%`，说明旧约87%主要是实现错误；
但该窗仍为`0/6,594 launch` raw/selected contact、0 legal landing，且后段hard-edge回升，不能称parity或学习
成功。同源码Isaac fixed tape继续0 done/timeout且摘要一致，只闭合共享输入与有限执行。实际Mu
`runtime.mjb`独立读取为`.001 s`、Euler/Newton/iterations100/noslip0，policy clock为`.001×20=.020 s`；
attach阶段的`.002/.001`warning不是最终compiled clock。两端controller/integrator/contact response仍未对签，
G06保持`Partial / diagnostic_unauthorized=true`。

fresh R07→Motion ready传递经消费者审计为无下游读取的self-echo，已删除而保留R07事实/critic与legacy API；
这减少同写者自证，不删除跨backend独立plant事实。exact Pod复核和新source双端replacement仍是下一验收项。

clean `75373daa`的exact Pod复核现已完成：Isaac fixed-action action/state SHA与父版逐字一致、0 done/timeout；
Mu 61-update窗finite且hard/guard降至`.789%/5.727%`，但`0/6,001 launch` contact、0 landing。它证明
shared fresh路径删减没有破坏固定输入执行，并支持替换旧negative；跨backend response、hit与landing仍未闭合，
所以不改变G06状态。

最终`d8fd8423` exact checkout重复通过相同三组测试、signed-authority与fixed tape后，双端fresh长期
replacement已取得连续ACK；旧`954200d5`双run按精确进程身份停止，root/checkpoint/log可恢复。到
`2026-08-29T08:07:38Z`，新Mu为`0/16,501 launch` selected contact，新Isaac为`0/377 launch` raw
actor-pair contact；两端课程入口都已打开而hit尚未形成。Mu balance边界改善、mimic三坏一好；Isaac
mimic三好一坏。该差异仍是学习证据而非physics parity，G06保持`Partial / diagnostic_unauthorized=true`。

50-update Isaac profile又证明响应差异之外存在明确的数据流成本：无active flight时Physical postphysics
均值`.014 s/update`，全H48 active时约`5.010 s/update`，超过同轮D05的`2.650 s`。它指向每substep的
Physical捕获→Epoch full-key join/journal→R06 settle/retire事务，不指向另一个“安全Gate”。后续融合必须让
两backend固定tape的contact/outcome/reason/fault/counter与scene-retire结果保持一致；该性能归因不提高
parity证据等级。

旧V9不是matched 0807实验：Mu实际载入legacy root。R8固定带现从同一clean commit、同一0807 plant身份、
同一artifact actor center、同一joint order与`512×H48×31` tape运行；两端initial q/dq exact、0 done/timeout，
逐tick qdes最大差`5.96e-8 rad`。Isaac clock=`.005 s×4`，Mu clock=`.001 s×20`，共同policy step为`.020 s`。

首个20 ms后q/dq已差`.00973 rad/.89084 rad/s`，tick47 q/root/racket差
`.13028 rad/.07872 m/.05263 m`。因此shared decoder/order与Pod通用安装损坏均无证据，剩余是backend
plant/controller/integrator/contact response；若继续归因，下一最小实验应记录contact-free与standing-contact
下的substep effort/clamp、constraint impulse及同torque tape，而不是增加task Gate。fixed tape没有physics
PASS阈值，行为接近也不等于数值parity；G06仍`Partial / diagnostic_unauthorized=true`。

旧V9 launcher的wrong-object根因已经闭合：它显式传入legacy `a3_pingpong.xml`，不是Jiayi讨论的A3P0807。
这使旧run的sim2sim差异不可比，但不推出整个Pod安装损坏。correct-0807、同clean source `954200d5`的fresh
双端现持续训练；首个Mu root因fresh checkout缺ignored 0807 `meshes/`在首ACK前fail-closed，恢复
[setup记录](../operations/setup_local_sync.md#restore-the-ignored-a3p0807-mujoco-mesh-closure)后以新root启动，
没有复用失败namespace。`observed_at=2026-08-29T05:18:37Z`时Isaac/Mu到ACK1172/3421；Isaac recent50
launch/contact=`4,882/0`、累计=`131,132/0`。Mu recent50 launch/raw/selected/legal-landing=
`5,096/3/0/0`、累计selected/legal=`48/0`，累计launch=`430,287`。这是真实学习证据，不是physics parity
分母；约`.0112%`累计selected/launch远未达到hit基本成功，recent50又为0。Mu recent50 p50/p90=
`6.264/6.392 s/H48`，两端finite与durable边界clean；Mu recent50 actual-hard-edge/qdes-guard已到
`93.139%/93.380%`，因此episode/return增长主要是边界污染，不能代签两端行为接近或课程晋级。

Mu `model_2000.pt`的GPU1隔离`512×240`诊断把aggregate hard edge收敛到三关节：
`waist_pitch/waist_roll/left_ankle_roll`，且`77.47%` hard rows发生在outcome-settled/recovery。
`waist_pitch`全部撞上限时mean action=`-.551`、nominal qdes约`-.325 rad`，实际目标方向远离上限；这进一步
反驳“相同qdes应产生相同响应”或“policy直接顶边”的简化解释。下一parity输入是这三关节的同tape
`q/dq/qdes/tau-before-clamp/tau-after-clamp`，不是再加task Gate。receipt见
[`model2000 joint diagnostic`](../../configs/action_ball_fullmdp_mu_model2000_jointdiag_20260829.json)。

该三关节probe现已接到真实Mu plant owner：默认训练路径不构造trace；显式隔离模式固定`512×240`，在原
`.001 s×20`循环内记录q/dq、raw/executable qdes、raw/clamped tau和hard-edge，并保存可供后续Mu反事实与
Isaac复用的action tape。它避免复制PD公式作same-writer自证；首次exact调用以诊断Tensor归并API错误在
首step fail-closed，修正只采用公开functional API并让trace绕开训练ledger，不改plant；该失败root不计
trajectory证据。

修正后的fresh exact trace已完成：三关节hard=`5,596/3,076/43 rows`，torque clamp三项均为0。
`waist_pitch`首hard时raw qdes向内约`-.325 rad`，旧shared executable却为同侧`+.33266 rad`；这是
velocity-horizon target被envelope夹在风险同侧、只给约`-6.98 Nm`制动的实现错误。候选v2只在共享纯tensor
owner内把单侧crossing映射到反侧maximum-inward endpoint，双侧/非有限仍保留旧bounded fallback；它与Isaac
已启用的policy-boundary选择对齐，但reason审计与Isaac substep响应尚未闭合，G06继续`Partial`。

最终fresh reason replay继续复用同一NPZ action tape，使三关节hard总量`8,715→171`（`-98.04%`），
waist pitch`5,596→171`，waist roll与left ankle roll都归零，且三项torque clamp仍为0。这支持旧弱制动
是主要因果。done `498→510`，但510个done全部有生产termination bit解释：`base_fell_tilt=390`、
`base_too_low=9`、`robot_hit_table=120`（可重叠），`joint_qdes_forbidden=0`、unknown bits=0、
done-without-reason=0；NPZ SHA-256=`d09b650d…9ead9`。reason替换故障疑点已经闭合，finite学习与Isaac
同字段尚未闭合，所以不能把98.04%写成parity或训练PASS，G06仍`Partial`。

与physics parity无关但会阻断evidence closure的UID漂移也已定位：portable writer是当前0807 action UID
`2552478955674699`，offline consumer仍钉旧UID `5527597793770800`。consumer expected值已更新，producer与
consumer仍是两个独立实现，组合测试保留单侧漂移反例；这只修证据可消费性，不提升G06状态。

当前Isaac run的live identity又直接记录Python `3.11.13`、Torch `2.7.0+cu128`、RSL-RL `3.1.2`、
TensorDict `0.10.0`，AppLauncher来自`IsaacLab-8320e0be`、Isaac Sim路径为`5.1.0`；这些与Jiayi的
environment reproduction合同一致。因此已证wrong-object是旧Mu plant选择，不是当前受控FullMDP软件栈装错。
但Build4 branch本身没有锁这套环境：其path-autodiscovery在当前Pod1默认命中Python3.10、Isaac Sim4.5、
`IsaacLab@21f71363…`、RSL2.3.1和旧PPO接口。若Jiayi本机Build4用5.1/RSL3而Pod用默认脚本，两条曲线确实
不是同一运行环境。历史actual override/argv/input SHA仍缺，故不能把差异的全部比例归因给这项漂移。

仓库部署能力不再依赖补齐这份历史证据：Pod1远端clean clone `e3ef4e98…`已经用显式exact runtime输入完成
dry-run、`52 passed`和真实Kit/PhysX fixed-action probe，0 done/time-out且退出后GPU释放。receipt见
[`fresh-clone deployment`](../../configs/action_ball_isaac51_fresh_clone_deployment_20260829.json)。它回答
“repo能否接入一台已合法准备好Isaac 5.1及private assets的新机器”，不回答跨机器逐位一致或G06 parity。

当前sim2sim裁决缺的不是另一个“响应差异必须小”的安全Gate，而是匹配对象：Jiayi本机exact asset SHA、
implicit/explicit actuator backend、clamp/effort、friction/contact参数、substep clock以及同一31-D tape。
在该receipt缺失前，G06只保留“首差已定位到controller/plant response”的事实，不调policy来补偿未知sim差异。

## 2026-08-28 V9同源课程双端复核（仍`Partial`）

clean exact source=`eb57233b4522d527455a0cbd7c547eb2ec49a68c`的双fresh长期replacement已在Pod1运行。
`observed_at=2026-08-28T09:48:33Z`时Mu ACK0..83=`2,064,384` transitions，近10 episode mean/wall=
`141.96 tick/6.57 s`、launch/R03/contact=`1,210/855/0`；Isaac ACK0..18=`466,944` transitions，
近10 episode mean/wall=`73.55 tick/16.76 s`、physical launch/R03/contact=`0/0/0`。两端finite与durable
边界clean，但阶段分母不同，不能把学习曲线当physics parity；旧V8已精确停止并只读保留。

V9已在Isaac真实Kit证明tick48自然打开Take061 task/playback；旧V8混源和tick295外推不再是跨引擎合同。
MuJoCo同一source有限窗已自然完成：p50/p90=`6.644/6.854 s/H48`，reveal=`10,860`、launch=`6,658`、
R03 valid=`5,107`、selected contact=`0/6,658`。Isaac对应窗却约82 tick全base tilt且launch=0，Mu约140 tick
并为table/tilt混合；这是真实behavior divergence，不是physics parity，也尚不能归因“Pod physics错”。
同physical-ready、joint order、q/dq和固定31-D action tape的first-divergence已完成：genesis修后initial
q/dq exact，live centered tape的executable qdes在tick0--34最大只差`5.96e-8 rad`，而q在首个20 ms已
分叉；shared decoder/order已排除，差异属于backend plant/controller response。G06保持`Partial`、
`diagnostic_unauthorized=true`；
task success不是parity/safety Gate，独立plant identity、fixed tape、finite与durable边界继续保留。

## 2026-08-27 FullMDP V8双后端replacement（仍`Partial`）

V7共享候选合同是PPO V5、Reward28、Observation V3 `215/231`、四个真实due/耗尽clock、Mu wire
`10/5/6`与Isaac milestone schema8。两个backend共用action/joint与paddle纯tensor数学，仍由各自plant
producer给事实；这关闭配方漂移，不等于physics parity。

V6 exact source=`caddecb76727ea55b0ce089453eea91cb5a9f8ea`已停止并冻结为negative只读谱系：
Mu/Isaac分别在`1,439,028/409,414`次launch后selected contact仍为0，且hard-edge反升。V7本地聚焦=
`606 passed, 34 skipped`、Pod1隔离矩阵=`706 passed, 32 skipped`。source `1d33130b`的fresh Mu/Isaac
均已有28项finite ACK；启动窗recent10 mean/p50分别为`5.360/5.366`与`8.423/8.380 s/H48`，但两窗都
没有task/contact分母且不是matched physics。contact、landing和matched physics继续为`未测`。学习分母只认
[课程实验§10.7](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md#107-v6结论翻转生存形成但mimichit桥与joint经济失败)，
性能证据只认[热路实验§16](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md#fullmdp-v6-rate-current)，
ABI只认[Observation接口](../interfaces/policy_observation_action.md#current-portable-fullmdp-semantic-observation-v3-actor-215--critic-231)。

V7后续固定窗已把两端合计约218万launch后的selected contact保持为0，且两端optimizer LR均卡`1e-5`；
两端共享失败支持learner schedule根因，但不构成physics parity。V8保持共同plant-facing语义，只共享替换为
[`PPO V6`](../DEFINITIONS.md#fullmdp-ppo-v6)。source `0ad85ae1…`两端exact rate已完成：Mu
p50/p90=`3.796/3.999 s`，Isaac=`6.835/7.612 s`；fresh双端已有连续ACK与exact LR`1e-4`。当前尚无
due/contact/landing或matched physics分母，G06继续`Partial`。

跨引擎Gate保留真正独立的plant/source、finite、joint/table/contact、full-key/generation、WAL/artifact
boundary与可重放fixed tape；same-writer echo、task成功和R07 ready不是parity或安全Gate。当前没有contact/
landing、formal checkpoint、completion或matched parity，所以G06保持`Partial`，不授权resume、promotion、
export、部署或真机；`diagnostic_unauthorized=true`保持。

**HISTORICAL / FROZEN — 2026-08-23 FullMDP V5双后端摘要（Gate仍`Partial`）：**两条live run的clean且已push source为
`39f9481950a660e198dedac7fd402806d648906b`。exact Pod broad CPU/ABI为
`792 passed, 57 skipped, 0 failed`；额外clean runtime focused重跑为`77 passed, 0 failed`，不与前者
相加伪装unique总数。MuJoCo GPU direct为`5/0`并完成RSL H48 one-update `1/0`；Isaac GPU为projection
`4/0`、selected-rubber `27/0`、runner drain `1/0`，合计`32/0`，但runner drain不是Isaac integration，
真实Kit fresh run提供了更强路径。

旧V4 MuJoCo已停止且未完成12500：最终ACK `0..4798`（`943,521,792` transitions），自然打开
due/reveal与launch/R03分母但contact/selected/landing仍为0。corrected V5的两个fresh namespace分别在
GPU1取得Isaac ACK `0..63`（`12,582,912` transitions）、在GPU0取得MuJoCo ACK `0..8`
（`1,769,472` transitions）；两端finite/fault/conservation均clean，但启动验收前缀的episode分别全部base tilt与全部
`robot_hit_table`，所有task链event仍为0。学习轨迹现冻结到Isaac ACK450 / Mu ACK385：Isaac从早期显著
回退恢复并创新高，累计due/public=`20/20`、ACCEPT/reject=`17/3`、playback=`2`；Mu最新20窗mean
length/return=`186.693/17.771`，累计scheduled/public/terminal-overlap=`307/297/10`、natural
launch/missed=`6/0`、R03 present/physically-valid=`1/1`。两端contact/outcome/landing/recovery仍为0。
这证明task→playback与reveal→launch实现链可达；Mu contact=`0/6 launch`是diagnostic negative，不能判
mimic成功或launch→contact已闭合。Isaac因launch=0，contact仍`未测`；两端landing仍`未测`。
actor/critic保持`203/219`、`history_length=0`，不增加
offset或stage；stdout不是证据
authority。H48速度是方向，fresh结果尚未做matched-strata，不能正式归因。自然mimic/hit/landing分母、
12500 completion、formal independent playback、keyed business-chain replay、physics/transfer parity以及promotion/deploy均未闭合，
两条fresh run都保持`diagnostic_unauthorized=true`，本Gate不晋级。详细冻结值见本页末尾当前块。

**2026-08-22 `aa42418b`双fresh现状（Gate仍`Partial`）：**Isaac与portable MuJoCo现均从clean source
`aa42418b187e8f3edf49d5757868fe0215e62d42`以H48 fresh运行。Isaac no-key critic `[216:219]`是N/A zero，
不读取ContactSensor；keyed R07仍读取真实contact/support/slip/recovery，actor 203-D不变。exact Pod=
`304 passed, 6 skipped`；Isaac匹配profiler-off中位`6.346 s/H48`。只读刷新到Isaac ACK176时，recent10
collection中位`8.238 s/H48`、episode均长`148.93 tick`；MuJoCo update118时对应为`9.585 s/H48`和
`151.17 tick`。两进程均仍active。
MuJoCo首两个一次性namespace因ignored EPA48 receipt/RSL3 wheel缺失在trainer/PPO和首个ACK前自然失败；按
三文件固定SHA manifest恢复后，r3 H48 update0--19中位`9.391 s`且连续finite ACK。两端fault/CENSOR/
nonfinite/conservation均0。

课程分母也已自然打开：Isaac累计`due/selected/construction/key=2/2/2/2`；MuJoCo累计
`due/reveal=15/15`、deferred=0且R03 present/physically-valid=`174/174`。这只证明row活过tick295时
task/action-mimic入口立即开放；两端仍无launch/contact/landing，故击球与上台为`未测`。两条均为
`diagnostic_unauthorized=true`且[仅全新训练、禁止续跑](../DEFINITIONS.md#fresh-only-no-resume)，不授权
numerical/physics parity、transfer、promotion、export、deploy或真机。详细见
[课程解阻实验§6.6](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md)。

**HISTORICAL / SUPERSEDED — 2026-08-22 fresh课程双后端对齐候选：**Isaac/MuJoCo现共享首次due tick295与
`295+293k`节奏；R07均不再授权task曝光或actor phase。MuJoCo修正了hidden joint=reset而body=frame0的
矛盾，hidden joint/body同取reset-ready，reveal后同切action frame0；Isaac本来没有该dense teacher错误，
但同样移除了pre-exposure R07依赖。两端body orientation采用同一fine+coarse语义。该对齐尚未取得exact
Pod fixed-tape或fresh runtime结果，不代签数值parity/transfer。详细见
[课程解阻实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md)。

**HISTORICAL / SUPERSEDED — 2026-08-22 dual-backend live prefix：**Isaac H48 successor `99405266…`已在GPU0
fresh启动并有durable ACK `0..10`，前10个完整wall median=`18.445 s`、Reward finite且fault0；portable
MuJoCo H48 successor同时在GPU2运行，最近只读为update `1110`、last-10 pre-ACK median=`9.771 s`且
Reward nonfinite/conservation fault为0。两条现在都真实运行，但不同backend、不同live state且不是matched
strata，不能据此签数值parity、transfer、physics promotion或本Gate晋级。

**HISTORICAL / SUPERSEDED — 2026-08-21 correction：**portable
MuJoCo r3已经停止在durable ACK `10249`；终段last-100 wall mean/median=`4.890/4.886 s/update`。
直接停止原因是真实MuJoCo-Warp `EPA_HORIZON` overflow fail-stop；旧229-D observation还存在IDLE clock
随global step漂移，所以即使扩大EPA容量也不得resume r3。fresh successor现已从clean detached `96f0ca69`
在GPU2启动H48 `4096×12500`，最近只读前缀为update `0..4`五个连续durable ACK且child仍在运行；这不是
completion或本Gate晋级。

successor同时切到semantic A203/219，并由Isaac与MuJoCo各自native producer生成同一语义布局与static
scale；R06缩为现有owner内key8+publication live selection与四tensor输出，不新建owner/receipt/Gate。
host实现不代签exact GPU ObservationManager/snapshot parity或deploy producer，所以仍无transfer authority。
详细合同与依赖见[当前长跑TODO](../operations/action_ball_dual_backend_longrun_todo_20260819.md)。

**2026-08-21 FullMDP PPO V2双后端配方身份（Gate仍`Partial`）：**Isaac与portable MuJoCo现在消费同一
typed learning recipe `H48/U12500/save500/E5/MB8/gamma=.99/lambda=.98`。MuJoCo不再接受H override，
snapshot/completion绑定包含预算与保存节奏的execution SHA；旧evidence schema 2/completion schema 3
又由下述runtime binding原子升级为ACK schema 3、completion schema 4、summary schema 3。
当前runtime/runner/ledger/consumer host union=`125 passed, 2 skipped`。这只关闭配方与wire漂移，不证明两后端数值、吞吐或学习
等价；exact GPU H48与fixed-snapshot parity仍`未测`，旧H24产物不得resume或冒充V2证据。

branch又新增一个不改变typed recipe的有限墙钟面：portable Full-A显式rate probe与Isaac默认false的task flag
都只允许`10 warm-up + 50 measured + 1 tail`、H48/4096、profiler-off，且保持
`diagnostic_unauthorized`；Isaac CLI仍不能把root `max_iterations`改成61。另有tracked N64×H48 fixed-tape
保存Reward20、actor203/critic219、plant与全部离散生命周期raw arrays，compare无tolerance/verdict。host分别为
rate runner `23 passed,1 skipped`、tape `6 passed`、Isaac focused `147 passed,26 skipped`。首次Pod尝试
在GPU/lock前因ignored EPA48资产未恢复而fail closed；只恢复三份
exact SHA资产后，同一GPU2/lock实际两次tape的离散/reason/events与初态全exact，连续repeat max envelope=
actor/critic `0.005192`、qpos `0.000634`、qvel `0.103833`、Reward20 `6.10e-6`，五个自然strata仍`未测`。
同一SSH的H48/4096 rate measured50 p50/p90=`9.448/9.661 s`、throughput=`20,779.64/s`、H24-equivalent
p50=`4.724 s`，source clean、前后apps empty、lock free。该结果关闭MuJoCo有限构造/吞吐HOLD，不关闭
ASan physics promotion、Isaac matched wall或本Gate `Partial`。

随后one-shot launcher从clean detached `96f0ca69887aba44c71983529d05e759e1a4cd2f`在GPU2 UUID
`GPU-473a79f3-8736-6c7f-c3db-290c6be385b8`启动fresh namespace
`fullmdp-a-h48-v2-96f0ca69-20260821`：Full-A `4096×48×12500/save500`、fresh runtime site、无resume/
retry/signal/`ACCEPT`门。update0的collection/learning/pre-ACK=`9.354775/0.284285/9.639704 s`，Reward20/
storage finite且fault为0，model0 SHA=`50ebc7c9…7b26`；最近已见update `0..4`五个连续ACK且child为`R`。
这关闭real-launch HOLD，只提供运行中engineering prefix；12500 completion、业务阶段分母与transfer仍未闭合。

**2026-08-21 portable Full-A fresh EPA48 runtime binding（Gate仍`Partial`）：**当前branch candidate在
base `074e2a0d`之上的runtime diff `df4d5ea6…f2d3`新增
[`--mujoco-warp-runtime-site`](../DEFINITIONS.md#mujoco-fullmdp-longrun-flags)（fresh双wheel隔离导入目录）；
legacy WAIT不绑定，binder不自报`source_commit`，future launcher须从clean Git truth传入。三份SHA、
import顺序/fail-close、EPA-only durable identity、RSL进程内门与schema `3/4/3`的完整合同只在
[portable Full-A实验§0](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md#epa48-fresh-runtime-binding-20260821)
维护；恢复与caller工序见
[`setup_local_sync`](../operations/setup_local_sync.md#bind-the-exact-epa48--rsl-rl-312-site-for-portable-full-a)。
Pod1无CUDA actual dual-wheel import=`19 passed in 4.33s`。tracked fixed fixture/replay又在GPU2
`GPU-473a79f3-8736-6c7f-c3db-290c6be385b8`、PCI `00000001:BE:00`各重复10次：stock24均为exact
`EPA_HORIZON` mask `256`/contact0，fork48均为mask `0`/contact1且dist/pos/frame finite，verdict=
`PASS_EPA48_FIXED_FIXTURE_REPLAY`；标准lock覆盖全程，结束apps empty/lock free。fixed-tape、MuJoCo H48 wall
与fresh longrun发射前缀已由上段闭合；ASan/instrumented独立oracle、Isaac matched wall、longrun completion
仍未闭合。故非完整physics/training GO，保持
`diagnostic_unauthorized=true / checkpoint_authority=false / resume_authority=false`，且尚未合`main`。
one-shot launcher又在Pod1 clean detached `2e4279ba`以真实venv完成无副作用dry-run：固定H48 argv与source
commit正确，run root未创建、lock元数据未变、GPU未查询；这只关闭发射入口构造，不代签real run或本Gate。

**2026-08-21 project-owned [MuJoCo-Warp EPA48](../DEFINITIONS.md#mujoco-warp-epa48-fork) build chain（Gate仍`Partial`）：**portable r3已经停止在
durable ACK `10249`；真实`EPA_HORIZON` fail-stop证明stock `MJ_MAX_EPAHORIZON=24`可达，但当时没有把
exact稀有geom pair/pose另存为可复现fixture，所以不能事后宣称已有24-fail样本。当前分支新增
[`PROVENANCE.json`](../../configs/mujoco_warp_epa48_20260821/PROVENANCE.json)、tracked two-hunk patch和
[`build_mujoco_warp_epa48.py`](../../scripts/build_mujoco_warp_epa48.py)：base只认PyPI
`mujoco-warp==3.10.0.3` sdist SHA-256 `f2219646…3070`、tag `v3.10.0.3` / commit
`710c34ca…5728`；source只准`pyproject.toml` local version改为
`3.10.0.3+hope.epa48.1`及`types.py`的horizon `24->48`。builder不下载、不安装，以
`--no-index --no-deps --no-build-isolation`产出ignored wheel，核全树two-file allowlist、wheel filename/
METADATA、patch前后各自281-file count+manifest digest、ZIP member/RECORD与wheel SHA；复验时重建完整
source并逐文件绑定wheel，严格复核schema-4 receipt/fork/source/wheel evidence。任一
missing/extra/duplicate/unsafe member都拒绝。builder环境
字段只作自报复现telemetry，不当成可独立认证的authority。tracked
[`HOST_BUILD_RECEIPT_SUMMARY.json`](../../configs/mujoco_warp_epa48_20260821/HOST_BUILD_RECEIPT_SUMMARY.json)
绑定本轮GNU patch zero-fuzz真实build的full receipt SHA `336f6454…041`与wheel SHA
`58f47b1c…61a`；wheel构建后另从pinned输入重建fresh source逐字节验wheel，避免build backend改写
首棵source后自证；host builder suite=`12 passed`。它仍只是`PASS_BUILD_CHAIN_ONLY`，不会代替
ignored full receipt重验。
Fixture/GPU/oracle是本Gate会变化的科学进度，只在本节、实验记录和compact summary更新，不进入
immutable build receipt；raw provenance SHA只留summary作telemetry。因此推进本Gate或改说明文字不会
迫使相同source/patch/wheel重新构建。

本轮明确拒绝提交一个1545行、仅由526行host fake-worker测试覆盖的通用search/capture WIP：没有CUDA和
两套可恢复隔离runtime时，它只能自验schema，不能证明Warp API、物理卡身份或EPA physics；当前算法还会
在合法最坏配置启动4356个subprocess。等GPU恢复后，应先用约200--300行临时批搜找到真实candidate，
随后只提交固定XML/pose/probe contract与singleton replay-only回归，不保留durable generator、搜索预算或
worker history attestation。这是删除同源自证和结构臃肿，不是降低物理门槛。

临时搜索已找到一对CUDA-qualified synthetic singleton；branch候选只保留固定fixture与singleton replay，
不提交搜索器。tracked replay自身已完成上述exact Pod 10+10次差分。当前仍缺
ActionBall fixed-tape数值与reason/counter/safety一致性，以及instrumented/ASan独立oracle。stock CPU MuJoCo同样硬编码24且该边界
可能越界，不能直接当oracle。runtime `d.overflow`与warning gate一位不降级，旧r3不resume；稳定
run-local site的actual dual-wheel import与tracked fixture replay已经通过，但其余physics gates仍未闭合。
因此这里只关闭可审计build chain、import identity和一对固定差分证据，没有关闭完整科学证据，
`diagnostic_unauthorized=true / training_authorized=false`，完整恢复/离线构建命令见
[`setup_local_sync`](../operations/setup_local_sync.md#restore-and-build-the-project-owned-mujoco-warp-epa48-fork)。

**2026-08-20 R07 payment-window branch candidate（Gate仍`Partial`）：**`1f9ed762…`将Isaac
R07 reward eligibility收窄为与MuJoCo一致的唯一谓词：
`phase == PHASE_OUTCOME_SETTLED && 10 <= deadline_relative_age <= 77`。readiness plant计算仍每个
control tick执行，不被payment window截断；REVEAL/LAUNCH不付R07，RETIRED保留的immutable
fact也不得重放age 77付款。selected true reset只清被选env的R07行，peer行与付款不变；
没有新增owner、receipt或gate。exact host联合回归=`190 passed, 6 skipped`，独立red-team为
`P0=0 / P1=0`。该commit尚未合入`main`且未live GPU验证，所以只是branch-scoped
candidate，不改变本Gate状态或当前adopted runtime。

**2026-08-20 current correction（Gate仍`Partial`）：**本段supersede下文把`take061/q_ready`
当physical reset、joint teacher在reveal立即切measured frame0、自然shot完成当nonterminal selected reset
并增加generation、ledger只有22 events，以及把zero-action table/fall表现当发车门的历史实现/判词。

portable FullMDP true Gym reset现在写`runtime_plant.default_joint_pos_rad`、配置default root加env origin、
零joint/root velocity和零current/previous action。30 s / 1500-tick内的due固定为
`2,295,588,881,1174,1467`；每次只作state-dependent `ACCEPT/DEFER`，DEFER零写且不在下一tick补试。
pre-swing HOLD的joint teacher为runtime default/zero velocity，body与R07 target为measured frame0；
`active_motion_s > 0`后才公开measured sampler。phase只允许`0/2/5/6/8`。

自然recovery success/window-timeout发`shot_retired`并停在phase8，不发Gym done、不改robot/action/episode/
`reset_generation`；后续真实ACCEPT才开启下一shot。只有真实Gym done发`selected_reset`并使
`reset_generation`恰增1。thin ledger累计26个event、5个terminal bit；其中
`completed_action_epoch`只能由同一env行闭合launch/selected contact/fault-free physically-valid
R03/fault-free eligible+source-valid R06/exact 68-cell R07/natural RETIRE后原子发布，
跨env边际不能代签。optimizer前prepare，成功后才snapshot与append+fsync ACK。

raw proposal/order/scale/default-offset host已闭合；MuJoCo hard-range与Isaac soft-inset/finite projection/
state-dependent brake仍为`DIVERGENT_DECLARED`，因此本run只有MuJoCo-only
`diagnostic_unauthorized` authority，没有transfer/promotion/matched cross-backend authority。
zero-action的table/fall/contact/recovery率只是telemetry；exact keepout termination保留，错误额外witness
不恢复。consumer把`engineering_run_complete`与slot0 `business_chain_complete`分开；25k工程seal
不能代签业务链，而本代因73动作、双侧与科学窗口报告未闭合，`full_a_complete`
固定为`false`。

**HISTORICAL LIVE PREFIX（已由顶部停止状态取代）：**当时host精确回归为runner/ledger/consumer=`96 passed, 1 skipped`，env/action/outcome=
`59 passed, 7 skipped`，alignment=`14 passed, 21 deselected`。Pod1 GPU2 fresh r2 exact
`9e7c1c614b1e22eeec4de243f55d58293da155ce`已保持production cadence不变并通过真实GPU focused
`8 passed in 23.00s`，随后进入4096-env trainer且首个optimizer update返回。stock RSL-RL 3.1.2
save写出7,882,391-byte `model_0.pt`后，因`log_dir=None`/`disable_logs`路径未初始化
`runner.logger_type`而触发`AttributeError`；one-shot `rc=99`、evidence 0 bytes、ACK=0。该未ACK文件
保持`diagnostic_unauthorized/checkpoint_authority=false/resume_authority=false`，不能算snapshot；r2
namespace封存且不可重用。fresh r3 exact source
`dc62684c41e70e40dedaf191a32921b6cd98b344`只显式安装upstream默认
`runner.logger_type='tensorboard'`字段，不启日志或上传；同一production cadence的真实GPU focused
再次`8/8`，worker/trainer PID=`864055/865285`，单一trainer进程的`4096×24×25000`当时进入active。
当时durable ACK=`0..7`，update 1/5只读consumer均通过；已ACK `model_0.pt`为7,882,391 bytes、
SHA-256 `06883851e67ccaaa921cfeeb8bf5c983ee6b3443d67465d8cde1d08ed63f528f`。前8个pre-ACK core
iteration范围`4.889..5.640 s`、median约`5.025 s`；每update Reward20/actual均有98,304行finite，
conservation fault=0且policy std finite。update0 due/defer/ACCEPT=`4096/4096/0`，到update5累计
`8192/8192/0`，update4 exact-table/Gym reset=4,096；这些只作行为telemetry。该段只是当时的live prefix；
最终进程止于ACK10249且没有engineering/business/Full-A终局，
Gate保持`Partial`。详细当前真源见
[`EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819`](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md)。

本次host结果的可复现命令（均从repo root执行；skip只允许标明为GPU-only）为：

```bash
PY=/Users/Franco/opt/anaconda3/bin/python3
$PY -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_full_mdp_update_ledger.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_gpu_ac_full_mdp_wait_rsl3.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_full_mdp_longrun_consumer.py
$PY -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_a3_train_ppo_action_transform.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_gpu_ac_full_mdp_initial_wait_env.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_gpu_ac_full_mdp_wait_transition.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_full_mdp_portable_outcome.py
$PY -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_isaac_alignment.py \
  -k 'raw_clip or affine_offset or action_axes or final_ctrl or vendor_scale or physical_ready or active_isaac or deleted_brake or extra_mujoco or declaring_the_live_guard or declared_guard_divergence'
```

**HISTORICAL / SUPERSEDED — 2026-08-19 slot0 question/teacher纵切片：**portable Full-A已删除midpoint serve、
`x+vt-0.5gt²`和`normal=-incoming`临时代码。fresh action仍严格为slot0/UID `6907688916670928`
（`take_058_unit02_fh`）；manifest center经live base yaw与shared Physical reverse-integration生成integer-tick
launch state，world origin只在真实qpos写入时恢复。measured motion NPZ、31-joint order、body order、mount sign、
strike frame52和canonical teacher rate在cold load一次绑定；public teacher与Isaac一致，在reveal原子切到
frame0，prepare阶段position保持frame0且reference velocity为0，随后才按rounded measured clock推进。
take061 physical-ready reset未改写。ready到frame0的`3.2918 rad`控制缺口没有伪装成teacher：仓内只有
逐字镜像Isaac diagnostic oracle的纯`q_des` recurrence helper，production尚无consumer。host focused=
`22 passed,7 skipped`；真实GPU shared reverse kernel、certified qdes/hold、R06/R07、Reward11--13和
portable `4096×25000`仍缺，因此不能宣称Full-A或长跑就绪。详细边界见
[portable纵切片实验](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md)。

**HISTORICAL / SUPERSEDED — 2026-08-19 first Full-A slice：**portable 229/399-D lane在既有WAIT VecEnv上增加
一条显式`full_a_mode`纵切片：逐行reveal、把launch state真实写入MuJoCo ball、20个physics substep、
live contact array、bounded flight terminal与selected reset。runner只累计真实extras，receipt诚实写
`task_lifecycle=full_a_slice_attempted`和`full_a_complete=false`。当前host纵切片还从同一postphysics
racket site发布真实R03 achieved position/velocity/normal，并由engine-neutral kernel计算Reward项0--9；
当前host路径又在真实generic contact edge的同一physics substep完成selected-rubber分类与一次性Reward10，
但fresh GPU节点仍未执行，R06 landing outcome、R07 recovery与Reward11--13仍为`not_produced`，因此它不是
portable MuJoCo A完成证据。当前MuJoCo host组合为`38 passed,9 skipped`；新增opt-in节点不伪造contact bool，而是从production racket geom与live
MuJoCo contact rows证明ball-racket pair，再穿真实`env.step`验证contact latch。首次Pod节点使用mesh
geom frame原点作为碰撞体内部点而失败；诊断证明该原点不在mesh体积内，而production measured-racket
site会产生exact pair。测试改用这个独立live site后，clean Git `2c8ef444…`在Pod1 GPU1/NUMA3得到
`1 passed`。这条历史收据只关闭generic racket-contact调用点；新selected-rubber/Reward10路径仍需另一个fresh GPU收据，
R03/Reward0--10也未由host代签GPU，R06/R07/Reward11--13仍缺失。更不能用host或native
114/114-D A1000代签本Gate。

**2026-08-18 dual MuJoCo lanes（Gate 仍 `Partial`）：**portable lane已完成真实MuJoCo-Warp WAIT
`learn(1)`（`N=2 × 24`、upstream RSL-RL3.1.2），但仍是`idle_wait_only`，不是full MuJoCo A。
同时启用已有native MJLab court trainer作为真实学习基线，因为它已有ball launch、physical contact、
episode termination和Reward telemetry。其XML与split-ready pose现为显式CLI输入，不再依赖隐式
`/workspace`路径。Pod1 GPU2已完成`1024 × 2` finite/capacity canary并启动fresh 1000-update长跑；
它的native observation是114-D、动作31-D。native结果不能promotion 229/399-D portable合同。
native A1000的完整JSONL update20/50显示episode length约145.45/145.56、min racket-ball distance约
0.454/0.448 m、binary contact仍0；action-rate income约`-3.65e-5/-3.78e-5`，capacity/nonfinite正常。
按预注册继续，不把短窗0 contact当停止门。
50--100窗口随后得到`6/8452=0.071%`真实binary contact，100--200为`32/17025=0.188%`；
mean min distance仍约0.462 m，故只记为稀少接触率上升。capacity/nonfinite继续为0，运行保留到500。
该native A1000现已自然完成1000 update。后500-update窗口binary contact=`4078/87546=4.658%`，
mean min distance约`0.325 m`，相对早期有明显方向性；但robot-table episode fraction仍约`97%`，而且
该lane只有114/114-D observation、10-term简化Reward，没有WAIT/reveal/ActionEpoch/outcome/recovery。
所以它是“MuJoCo能快速学到更靠近球并偶发接触”的E1趋势，不是portable FullMDP A，也不关闭本Gate。

**2026-08-18 producer-first更正（Gate 仍 `Partial`）：**一次M05 root骨架尝试只把真实
Plant/MotionBank/R05/M04构造到一起，但`step/reset`仍fail-closed，属于HANDOFF禁止的zero-callpoint
债务，已完整撤回。进一步审计发现该WIP是73个未跟踪Python文件、约12.4万行，并继续面向历史
A211/C211 `211/319`；不能整包入Git或作为当前portable MDP。live Isaac现役合同是FullMDP ActionEpoch
`229/399`，因此MuJoCo 2.0只认该宽度。第一纵切片从tracked `a3_train_ppo.py`真实plant的all-world
`reset -> sim.forward` callpoint直接构造initial-WAIT 229/399 TensorDict，production硬上限500 LOC；
不新增root/owner/receipt/schema。该reset切片通过也不授权`learn(1)`；真实step、Reward、termination、
masked reset闭合后才可GPU训练。见[唯一 TODO](../operations/action_ball_single_action_dual_backend_todo_20260817.md)。

第一纵切片现已按该边界落成：一份engine-neutral module只拥有229/399列名、宽度和拼接顺序；Isaac
observation source改为消费它；MuJoCo subclass复用tracked `A3ReadyBallVecEnv`的真实all-world
`_reset_idx -> sim.forward`，在reveal前把球park于HOPE-local `(0,0,+10)`，并把live contact array
中任何ball-involving row作为硬失败，再把live robot state与
canonical zero task/clock/fact/reward rows与ActionEpoch `IDLE` one-hot组成RSL TensorDict。新增production净增253 LOC，没有root、
owner、receipt、SHA或registry；`step()`继续硬拒绝缺失的Reward/termination/lifecycle。Host focused=
`9 passed,1 skipped`后，Pod1 GPU2 fresh Git `495a0870`的唯一live test已自然RC0：N=1 real reset、
finite 229/399、IDLE one-hot、+10m park与raw no-ball-contact均通过，result SHA=`6fe2e70c…0ba6030`。
因此reset/readback子门为`PASS-live`。后续新commit/new namespace直接从compiled model断言
`physics_dt=0.001`、20 substeps、control dt=`0.02`，再次RC0；attach warning没有改变最终timestep。
`noslip=0`是MuJoCo-Warp不实现vendor noslip pass的已登记backend deviation。真实step、Reward、
termination和masked reset闭合前，本Gate仍为`Partial`且禁止`learn(1)`。

最小WAIT transition已在host闭合真实20-substep plant调用、Reward20、四项shared termination和逐env reset。
实现没有迁移旧M04：MuJoCo内部`cvel`先从kinematic-root
`subtree_com`参考点平移到每个body的inertial COM，再与Isaac `body_lin_vel_w/body_ang_vel_w`同义；
rotating offset-body的native `mj_jacBodyCom @ qvel` oracle最大误差为`5.6e-17`，而raw `cvel`线速度
误差范数约`0.50`。Reward使用Isaac同式的上一拍live-anchor x/y+yaw teacher pose cache；本拍Reward后、
最终forward后才刷新下一拍。每个policy step在最后一次integration后显式`forward`，再读取derived body
tensors并锁存同tickresolved robot-table contact。该contact单独导出具名backend bool，不复用Isaac
`robot_hit_table` bit16；mixed-nonfinite qdes按joint回退而不清掉同行finite action，并保留raw终止证据；
portable component-[OBB](../DEFINITIONS.md#obb) [SAT](../DEFINITIONS.md#sat-collision-test) keepout仍是learn前HOLD。host组合=`8 passed, 4 skipped`；
production净增396 LOC。真实GPU结果如下，不能反向把host skip写成live evidence。

Pod1 GPU2随后从fresh Git commit `e71ee1a350d…` 运行同两文件，`12 passed`、RC0。该门真实覆盖
N=1 reset/非零step/timeout/Reward20/park和N=2 selected-reset peer preservation；result SHA256=
`f4d41aa983522243b657a437dc22065ad82895e4c1965ea777a384845c2591eb`，log SHA256=
`5fa33b26af3fea72007f6d48915cc96239a22c2729fd7140e637edddfa7286c1`。测试结束后GPU2无compute process、
queue lock释放、checkout仍为exact commit。Gate据此提升为`PASS-live-step / HOLD-learn`；portable SAT
keepout与RSL3.1.2 trainer未闭前仍禁止`learn(1)`，不能把resolved contact提升为Isaac同义bit16。

下一条SAT纵切片保持同一plant/root且production净增249 LOC：construction复用既有native authority绑定
32-body顺序、62个collision-component OBB、racket blade与五个table AABB；policy loop只运行fixed-shape
Torch 15-axis SAT。MuJoCo-Warp每次step留下pre-integration derived pose，所以substep hook跳过state0、
读取state1--19，最终显式forward读取state20，正好覆盖一个control step的20个post-state。resolved-contact
仍是独立backend telemetry；shared bit16只由几何keepout写。host float32/float64、nonfinite与45°空角
`broad=true/exact=false`判别卷均通过，组合=`12 passed, 5 skipped`；current 8320 branch重钉的
config/callable语义AST（含fixed-dense SAT与component/blade fusion）同时通过live constant parity与
随机Isaac-vs-native SAT对照。真实GPU test仍
是skip，因此状态为`PASS-host-SAT / HOLD-live-SAT`，不能据此启动PPO。

trainer边界也已缩成一份薄launcher：直接构造现有WAIT env，再调用upstream RSL-RL 3.1.2
`OnPolicyRunner.learn(1)`；没有复制runner、PPO或storage，也没有新增owner、receipt、WAL或checkpoint。
构造前核module binding、构造后核实际PPO、ActorCritic、RolloutStorage对象均来自同一
`rsl-rl-lib==3.1.2` distribution，optimizer为exact `torch.optim.Adam`，拒绝有副作用的同名预载模块。
host focused=`7 passed, 1 skipped`，WAIT/SAT组合=`21 passed, 6 skipped`；唯一RSL3真实GPU用例仍是skip。因此它只关闭production callpoint，
科学状态为`PASS-host-callpoint / HOLD-live-RSL3`。薄launcher必须单次读取并SHA绑定运行器冻结的
ready-pose，再让真实WAIT env直接解析同一份bytes，不能让direct test与production update读取不同文件。必须在fresh空卡先通过上段live SAT，再用隔离安装的
RSL3.1.2运行`N=2 × 24`一次update；当前不得写成MuJoCo A已运行。

第一次真实整合one-shot改用Pod1 GPU0共卡：物理卡原有1个peer、free显存大于20GiB，本进程树固定在
GPU0本地NUMA的CPU `32-47`。direct完成`16/19`后，3个真实env构造均被同一错误path gate拒绝：运行器
已将77个A3文件逐项快照到fresh namespace，内容与clean Git canonical A3树完全相同，但native table
authority仍要求原绝对路径。该条件不是plant identity。fresh修复保留root SHA、portable source-closure
receipt与live owner-local frame三重校验，只删除绝对路径相等；不同字节拒绝与同字节异路径正例=`2 passed`。
该run的`direct_rc=1/rsl3_rc=99`保留且不重试；fresh live 19-test仍待验证。

fresh identity修复后的下一次one-shot已越过绝对路径门，三个live构造共同停在attached-scene body命名：
portable A3注册名为`pelvis_link`，MJLab attach后的live名为`robot/pelvis_link`。这不是几何或plant漂移。
authority现要求adapter显式传`robot/`，用它只做live ID解析；写入比较合同的仍是canonical 32-body名、
父子关系与owner-local frame，所以不能自动猜prefix，也没有放宽portable identity。native bare-model默认行为不变。
host owner-frame namespace卷=`7 passed`，device SAT卷=`4 passed,1 skipped`；新fresh GPU 19-test通过前
仍保持`HOLD-live-RSL3`。

新commit `61887b43…` 已在Pod1 GPU0与一个既有peer共卡完成fresh one-shot：process tree绑定GPU0本地
CPU `32-47`，运行前后该卡均只保留原peer且约25GiB free。19个direct GPU测试全部通过、0 skip；随后
同一clean checkout直接调用upstream RSL-RL3.1.2，完成`N=2 × 24`一次PPO update和48 transitions，
policy/critic=`229/399`，`diagnostic_unauthorized=true`。result/log SHA=`322592ce…f07a`/
`7d4fbee7…2a3b`，queue lock自然释放。因此SAT与真实trainer调用点提升为`PASS-live`。receipt明确
`task_lifecycle=idle_wait_only`：它只证明WAIT engineering `learn(1)`，没有A question、launch、contact、
outcome或long-run证据，G06仍为`Partial`。

**2026-08-02 successor 提案（Gate 仍 `Partial`）：**下一版将 MuJoCo 设为 N73 主训练引擎；Isaac
只提供 N1 最小可学证据和冻结 handoff。G06 未来应拆成 portable contract/plant/reward/reset parity、
可选 Isaac checkpoint replay diagnostic、MuJoCo native VecEnv/PPO 三个子门；本页下方的 mandatory
Isaac-trained ONNX 与 reset-first 179-D 条款是旧版接受条件，尚未由 successor 实现取代。新依赖、
真球 matched benchmark 和验收草案见
[MuJoCo 原生下一版准备账](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)。
在代码/合同和 `main` 主板切换前，Gate 状态不晋级。

**2026-08-04 19:53 current correction（Gate 仍 `Partial`）：**exact Pod WIP r6 已使
A211/C211 各完成 `1 env x 2 PPO update + reset-boundary save + fresh-process cold-load`；
211/319 observation有限，fresh WAIT canary通过，cold-load/update2 exact均为true。A/C result
SHA分别为`d58cb750…83bb2`/`440a1f2e…23733`，native+legacy组合回归为
`219 passed, 2 skipped, 0 failed`。这关闭下文r3/r4所描述的“尚未跑通当前A/C执行链”阻塞，
但不关闭Gate：当前仍是CPU、cap64、partial-reward、diagnostic-only路径，不是4096或formal
training。每个update还有7个无selected-contact的TASK_ACTIVE hard-terminal row，而当前receipt
没有具体termination reason；补齐reason telemetry、完整reward/safety、mid-episode resume、
4096与cross-engine parity前不得晋级。随后对exact checkpoint做确定性replay，已把每个update的
7个hard-terminal全部定位为`joint_actual_forbidden`：A在episode tick `70..84`，C在`69..88`，
全部早于nominal strike；timeout/base/table/contact/strike/landing均0。因此当前MuJoCo 4096 scale
明确阻塞，须先用sealed current mean-only/std.02与未sealed4σ-inset候选做同条件100+ tick诊断；
若采用inset必须产生新lineage，不能拿r6代签。下方同日r3/tick74段落保留为历史根因记录，其
运行态由本段覆盖。

**2026-08-04 A211/C211 移植 successor（Gate 仍 `Partial`）：**当前 branch 已有相互独立的
211/319-D A/C ABI、`task_valid`、各自 task/reward 和 reset-boundary checkpoint。A 消费 desired
contact，formal question source 是 `online_solver + complete-semantic exact-answer cache`：cold Q/Q'
各真实解一次，sampler/curriculum/RNG 每次 reset 正常推进。C 消费 incoming-ball-at-contact，formal
source 是 `direct_ball`、总 inverse call=`0`，且不使用 `immutable_tape` 或 answer cache；C 的最小 task
reward 只有 nominal-strike 拍心-球心距离与 actual-selected-rubber-contact-gated 的一次落点，没有
desired-contact、独立 hit bonus 或额外 dense outcome。

两族 physical reset 都使用 tracked split-ready，而 exact measured frame 0 只作 teacher authority：
direct physical-birth 同门槛扫描为 `0/73`。Host 已实现 seed=`20260804` 的 per-env 5--25 tick
RESET_WAIT；WAIT 中 plant/teacher 都停在 split-ready，最大25 tick 已被 `60/240/1.2 s` hold receipt
覆盖；球停在无接触 parked state，不能用反向弹道在 WAIT 内先穿过桌/地。task reveal 同 tick
原子安装 sealed incoming-ball launch state和tuple、把 teacher 切到 measured frame 0并公开原始约
`.712376 s` teacher-start clock，让 dense mimic 学 bridge。checkpoint v3 保存 reset-boundary WAIT
continuation。Pod WIP r3 已真实进入physics/update并验证parked-ball→reveal，但尚未完成验收：native
fresh actor原先没有Isaac采用的safe-ready output bias；补成末层weight=0、bias=normalized hold、
std=`.02`后能穿过16-tick WAIT，却仍在tick74因`waist_roll_joint` actual hard-limit失败，早于nominal
strike tick108，且qdes恒为`-0.0816`。这证明bootstrap必要但不足，也证明旧Isaac split-ready/PD不能
直接代签MuJoCo稳定hold。当前须先关闭per-joint 4σ executed-qdes margin和MuJoCo-native
`>=500` control-tick safe hold；随后A/C各自重跑`1 env x 2 PPO update + save/fresh-process cold-load`，
rollout可继续采集到reset boundary但不得丢弃或覆盖transition。当前还没有clean S1 exact receipt，
也没有跨引擎 WAIT parity receipt。当前 scalar reward 已消费与 Isaac 数值/集合同义的 partial subset：
upright、base angular/vertical velocity、joint velocity、action rate、13-body non-wrist pose/velocity mimic
与 measured-paddle position/velocity/signed-face/long-axis；WAIT 中 prior 继续工作而 task 严格 mask。
脚接触/滑动/落地、undesired-contact、Isaac applied-torque、完整 safety/projection、termination/
mid-episode resume/export 与 GPU 4096 VecEnv 仍未闭合。因此只能称 `A211/C211 code-path partial`，
不能称 MuJoCo 移植已完成或 trainer ready。
当前 five-update scale 的 implementation strict-zero 只包括 qdes-hard、actual-hard、nonfinite；
fall/base-too-low/robot-hit-table 仍终止，但属于待按 hidden-WAIT、revealed-pre-strike、post-strike
分相位守恒报告的行为账，不能改写成“初始 policy 必须零次”的实现门。

**2026-08-03 C-lite trainer 历史（Gate 仍 `Partial`）：**native 路径已新增连续
physics-substep transcript 的 observed selected-rubber outcome resolver，可分 racket/table/net/floor 事件、
valid achieved flight 与 observed landing。scalar reward 没有独立 contact bonus：只有
`1/(1+(distance/.15m)^2)` 拍心-球心距离、legal landing=`1`和 opponent-side-out=`.5`；
actual contact 只是 outcome eligibility。diagnostic PPO shell 已保存 actor/critic、Adam、normalizer、
Python/NumPy/Torch RNG 与 contract/observation/action/reward SHA，并证明 reset-boundary save/cold-load
next-update exact parity；显式禁止 mid-episode resume/formal authorization。

`MujocoN1DiagnosticVecEnv` 现已接 C-lite normal step，但只接受 immediate `TASK_ACTIVE` portable parent，
显式拒绝尚未移植的 RESET_WAIT/task-valid authority。为了不伪造 full mimic，MuJoCo C-lite 当前
motion/balance scalar 明确为0，所以只是 plumbing/learnability smoke，不是和 Isaac 配方等价的
canonical N1。host 历史 portable VecEnv/Torch/trainer 聚焦数保留；latest exact clean Pod component
suite 已是 `108 passed, 0 skipped, 0 failed`，因此“MuJoCo+Torch Pod 组件未测”这一旧口径关闭。
branch 另有
`run_mujoco_c_lite_pod_diagnostic.py`：默认只做 SHA-validating plan，执行必须显式确认
`diagnostic_unauthorized`；它在 reset boundary 保存，再用 fresh Python 进程重建真 core/
trainer，比较 next-update transition/reason/safety 以及 model/Adam/normalizer/RNG 摘要。
host 聚焦用例已过。exact Pod commit `42500ade` 随后物化了 SHA 冻结的 robot tape/
question，用真 MuJoCo 3.10.0 + Torch CPU 完成 `1 env x 2 step x 2 PPO update`；reset-boundary
checkpoint 保存/加载 SHA 同为 `e623d214…0026`，fresh child 自然退出，下一 update 的
receipt/model/optimizer/normalizer/RNG/reason+safety transition 全部 exact。tracked result 为
`configs/mujoco_c_lite_20260803/42500ade_pod1_reset_boundary_cold_load.receipt.v1.json`，file SHA
`ad62b45d…377a`。commit `934b7c03` 又先按 `action_specific_hold` producer 生成正确 robot tape，
再由 executable runner 重跑同一 `1x2x2`：checkpoint SHA=`1d72324e…d8a3`、fresh-process
cold-load exact=`true`。历史 local/untracked result（本轮显式不纳入 S0 source commit）位于
`configs/mujoco_fixed_center_20260803/934b7c03_pod1_action_specific_hold_1x2x2.result.v1.json`
file SHA=`9987e723…aa3b`。这关闭 executable plumbing gate，不关闭收据中列出的 formal 211/319
ABI、mimic/phase/physics parity、mid-episode resume、throughput/export/deploy blockers。

这也是 [PRE-LONG 基础闭包](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md#122-pre-long-基础闭包2026-08-03)
的第五项：任何 current A/C long 前，A211 与 C211 必须分别由 exact clean Pod 真实 runner 跑完
`1 env x 2 PPO update`，保存后在 fresh process cold-load，并验证下一 update/state 与源
SHA/normalizer/checkpoint lineage；component test、host fake core、历史76-D C-lite 或只构造 CLI 都不够。

当前 MuJoCo 实现状态已不再只是 single-env：
`deec4a52c758b1f173436d4522e3e13e7ccb7bfd` 已在 native physical-ball core 外增加一条
CPU sequential diagnostic `VecEnv`，具有 deterministic batched reset、76-D purpose-group observation、
finite no-reward rollout、strict physics-substep contact-event ledger 和 exact tape-timeout latch。
`41411c3b6a6ef3ad03c2cba41370e84709066d8d` 又从
`HOPEDeployParityTerminationsCfg` 绑定了两个 exact base termination subset：
`base_fell_tilt := pelvis_up_world_z < cos(0.7)` 与
`base_too_low := pelvis_link_origin_height_w_m < 0.5`，都是严格小于、control step 后取样、
sticky latch，子集内的 reason order 为 tilt 优先于 height。其 Isaac 源配置字节 SHA 被固定；
源语义漂移时 fail closed。termination blocker receipt 只在首次校验源后缓存，
4096 次 cache-hit 调用合计 `.446 ms`，receipt SHA-256=`353382b4…3789`。

2026-08-03 当前 successor 又加入 `joint_actual_forbidden` exact diagnostic predicate：每个
control step 后，以 runtime joint order 将实际 `q` 对照 MuJoCo `model.jnt_range`，固定
exact-zero bounds tolerance，并 sticky 保留同一 tick 内任一 physics substep 触边；非有限/无效区间或到达任一 raw hard edge 即触发。它与 tilt/height
共享 sticky latch，reason order 为 tilt→height→joint actual。Isaac 配置及 termination callable
源码分别 SHA pin，漂移即拒绝。Host 三组聚焦回归为 `45 passed, 8 skipped`；skip 是缺少
MuJoCo/SciPy 的真引擎用例。该增量仍是 `diagnostic_unauthorized`，robot/table、qdes、phase/recovery、
compact reset、Reward/PPO/save/resume/export 继续 fail closed，G06 保持 `Partial`。

同日 current-worktree successor 再关闭 `joint_qdes_forbidden` 这一具名子项：绑定 Isaac
`pre_clamp_qdes_forbidden_zone`、`joint_pos_limits`、`margin_rad=0`、`margin_fraction=0.02`
与 `finite_preclamp_qdes_projection_enabled=true`。因此 finite pre-clamp 越界仍由 projection+
penalty 保留 transition，只有有效 affine qdes 含 NaN/Inf 才触发 hard termination；reason order
扩为 tilt→height→joint qdes→joint actual，sticky latch 不变。receipt 继续双源码 SHA pin，
host 三组聚焦回归 `45 passed, 10 skipped`（新增 skip 仍来自 host 缺 MuJoCo/SciPy）。PPO
`step()` 继续在 physics 前 fail closed；robot/table、phase/recovery、compact reset、Reward、
save/resume/export 仍未闭合，Gate 不晋级。
同一 exact `0d1d641e` 随后已传入 Pod1 clean checkout
`/workspace/franco/actionball_mujoco_0d1d641e_20260803`，用 Pod 现有
`/workspace/hope_isaac_venv/bin/python` 执行完整 native suite=`55 passed in 13.59 s`，
所有 host optional skip 在 Pod 均实际执行。这只证实 current diagnostic
scene/VecEnv/termination subset 能在 Pod runtime 运行，不改变上述 PPO 阻塞。

current successor 继续移植 `robot_hit_table`：新 exact guard 重开 Isaac
config/callable/`hope_actions.py` latch、43-component collision artifact 与五实体 table
geometry 的 SHA，并只接受 canonical root MJCF。它还将 verified base model 与实际
augmented/precompiled live model 的 32 个 owner body 按 name、selected parent、local
position/local quaternion 逐项比较，同名但 owner-frame 漂移会 fail closed。随后按 component
world-AABB + live racket OBB broadening 对加 `0.02 m` margin 的桌体 AABB 做 inclusive
overlap。每个 physics substep 取样，control step 内 sticky 并保留首次 substep；只接受
decimation=4，reason order 为 tilt→height→table→qdes→actual。Host 完整 native
suite=`60 passed, 12 skipped`，skip 是 host 无 MuJoCo 的真引擎路径；该增量尚待 exact Pod
重验。随后 current-worktree successor 已实现 diagnostic lane 的 per-env episode done latch 与
terminated-batch compact reset：`episode_dones=exact_hard_terminations OR time_outs`，只 reset
命中的 core/episode length/hard latch/ledger，未命中行连续推进。返回 observation 是 reset 后
next state；另以 mask 绑定 reset 前 terminal observation，terminal ledger 也是 reset 前 caller-owned
deep copy。`robot_hit_table` 的首次 physics substep 保留在 terminal snapshot，新 episode latch 清空。
rollout v4 receipt 按 env 冻结完整 question source SHA 列表，并公开 semantic 与 digest-only terminal
trace descriptor，使 receipt 加返回 trace 可独立重算总 digest。Host 四组聚焦回归=`62 passed,
13 skipped`；skip 是 host 缺 torch/MuJoCo 的集成路径，不是 PASS。

后续 current-worktree 最小切片已从 Isaac authority 精确绑定 reference-envelope 三项：
`anchor_pos`、`anchor_ori`、`ee_body_pos`。阈值分别为 anchor z 误差严格 `>0.25 m`、
reference/robot projected-gravity body-z 绝对差严格 `>0.8`、四个 feet/hands body z 误差任一
严格 `>0.25 m`；等于阈值不触发。`recovery_hold` 的 `in_hold=true` 会屏蔽三项，且仍受
ActionBall episode-frozen `reference_terminations_enabled` gate 约束。hard reason order 现固定为
anchor pos→anchor ori→end-effector body pos→tilt→height→table→qdes→actual。
所用 termination class inheritance 与 direct term assignments（含源码顺序）、raw predicates、
hold/gate helpers、command gate 与 A3 feet/hands assignment 采用 selected-AST semantic SHA pin；
相关语义改变会 fail closed，无关 A225/C225 或
reward WIP 不会伪造漂移。既有 table/base guard 也已收窄到实际消费的 term factory、termination
classes、`robot_hit_table` callable、physics-substep latch class/method 与 qdes/actual callable AST；
阈值、term/order 或 latch 语义变化会拒绝，无关 config class append 不漂移，不再依赖易碎的整文件 SHA。

production `MujocoN1BallCore` 现已提供显式、不能猜 phase 的安装缝：只有加载外部 SHA-bound
`a3_mujoco_phase_fidelity_reference_tape_v1` 后才广告 sample contract。reference tape 绑定
plant/scene/robot-tape/sample-contract SHA、`pelvis_link`、四个 feet/hands body order、逐 tick
post-control-step MotionCommand reference、hold context 和 episode-frozen gate；row 数必须与 robot
tape 完全相等。core 从 live MuJoCo pelvis link-origin z、pelvis rotation 推导 projected-gravity-z，
并从四个真实 body `xpos.z` 计算误差，未读取或推断 `time_to_contact`。文件/content seal、authority
source SHA、binding、gate 恒定性、hold/context、finite/range 任一不符均 fail closed。

VecEnv 仍要求全部 core 同时广告相同 contract SHA 且每 tick 返回完整 sample；mixed advertisement、
SHA 不同、漏样本或未广告却返回样本都会使整批失效。默认未安装 external tape 的 core 保持不广告，
receipt 写 `exact_phase_fidelity_runtime_sample_available=false`，当前 formal blocker 是
`native_core_phase_fidelity_reference_tape_not_installed`；安装合法 tape 后 termination receipt 才变为
`FORMAL_TERMINATION_AVAILABLE_DIAGNOSTIC_ONLY`，training/promotion 权限仍全 false。rollout receipt 现为
v4，并把 runtime availability、contract SHA、reference-tape SHA lineage、每 env canonical phase sample
与 native physical-event facts transcript 纳入 digest。当前 phase sample contract
SHA=`e33568f5…f1d2596`；host 五组扩展回归=`89 passed, 18 skipped`，其中指定的
core/termination/vec/reward 四组=`72 passed, 13 skipped`。这是重验前的 host 口径；其中
真实 MuJoCo core emission 与部分 torch VecEnv runtime 当时因缺依赖而 skip。
完整 Reward/PPO/save/resume/export 仍 fail closed，G06 保持 `Partial`。

当前 successor 又修复了两个跨 runtime 问题：selected-AST pin 不再受 Python 3.12+
新增空 `type_params` 影响，并对 `Ellipsis/bytes/complex` 做显式可移植编码；
runner exact-resume tensor digest 先把 scalar reshape 成 1-D 再 view bytes，不改任何 tensor
内容。host native+plant 联合回归=`115 passed, 18 skipped`。exact Pod detached clean
`299145e9` 又分别通过 native=`110`、plant=`26`、runner guards=`25`，合计
`161 passed, 0 skipped, 0 failed`。这些只关闭 diagnostic core/runtime guard 的
Python 3.10/MuJoCo 3.10/Torch 2.7 复核，不是 Reward/PPO 或 normal-step 授权。
同一 exact `7135d5ce` 随后已传入 Pod1 clean checkout
`/workspace/franco/actionball_7135d5ce_20260803`，上述四组完整 native 回归=
`72 passed in 17.44 s`；这关闭当时 table-guard successor 的 host optional skip，但早于上述
compact-reset/lineage successor；该 successor 的最新 exact Pod 复核是上述 `299145e9/161 passed`，
两者都不改变 Reward/PPO blocker 或授权状态。

以下 single-env 记录是 current A211/C211 split-ready contract 之前的历史 predecessor，只保留底层
plant/action 证据；其中的 reset composition 不再是 current A/C 发车输入。该 predecessor 绑定
schema-3 31-D action、implicit total-PD、episode-fixed delay、
immutable teacher reference + 独立 sealed physical reset/hold 和 100-tick fixed tape。
首轮 tick9 hand↔hip/wrist↔table 失败的根因是把动态 v5 teacher frame0 当成静态出生状态；
teacher reference 没有被改写，physical reset 现使用在当前 exact MJCF 重审的 shared
root/leg + v5 非腿关节，并由 LP 求 envelope 内 hold qdes/history。修复后 d0/d1/d2 各跑满
`100 ticks / 400 substeps`，qdes clamp、velocity、自碰和桌碰事件全为0，因此状态更新为
`IN_PROGRESS / BIRTH-HOLD-SAFETY-PASS`，仍不是 trainer ready。三条 effort clip 分别为
`1108/1098/1084`，不得外推为机械准入或 learnability。clean Pod
`/workspace/franco/actionball_mujoco_41411c3b_20260803` 上三个聚焦测试集为
`48 passed in 15.71 s`。但正常 `step()` 仍在 physics 前 fail closed：剩余的
Isaac-equivalent robot/table collision termination、phase fidelity、
terminated-batch compact reset、teacher/official-racket-site p/v/face/long-axis、完整 reward 与
PPO/save/resume/export 仍未闭合。formal canonical N1 authorization 也仍因最终
ABI/reward/scheduler/measured authority 未冻结而 `BLOCKED`。
详见 [MuJoCo native single-env 运行账](../operations/run_mujoco_native_single_env.md)。

2026-08-03 的并行增量已将 single-env 推进为**native physical-ball plumbing probe**：新 scene
绑定 table/ball/racket contact pair、portable/backend asset closure、immutable-question/external expected SHA，
并在每个physics substep上 latch首次接触、recontact/同时接触invalid及contact-end outgoing state。
Host为`30 passed, 7 skipped`，Pod MuJoCo 3.10.0为`37 passed`；一次真immutable authority演练
运行400 substeps、仅触发一次table edge，跨reset/fresh-core trace确定。但explicit launch还没重现
immutable tape的aero/table-bounce轨迹，收据正确写`incoming_question_parity=false`；没有racket hit、
reward、PPO、checkpoint或trainer授权；现在已有的是上述 no-reward diagnostic VecEnv，
不能把它写成 trainer。因此 Gate 不晋级。

MuJoCo 拍面几何使用 2026-08-03 v2 exact identity：`right_racket` site/FK 不变，只修正
collision proxy 的 Y 厚度，新 root MJCF SHA-256=`70c4fd65…36c0a`。旧 v1 identity 仍
保存历史收据；formal lane 须新建 v2 的 L0/vendor-L1/table-net successor 链，禁止
把旧证书原地 repin 到新 MJCF。

portable parity 还必须绑定两项新权威：同一份实测 racket teacher，以及同一批 fixed
swings 上的 reward landscape/实际收入收据。旧 schema-v3 长轴错了45°，已 revoked；本地
schema-v4 已用 URDF/MJCF 正确轴完成 exact `73/73` full-phase 运动学重定向、50 Hz 物化与
独立 FK 反算，并生成 receipt-bound 73-action manifest。完整机械口径仍是
`0/73 admitted`：`57/73` 有已观测 hard failure，另 `16/73` 仍为 `UNKNOWN`；
37/73 超速、58/73 近限位是较早窄口径机理反例。prototype/source capsule/final ABI 也未闭合，
因此尚不是 formal N73 teacher。Isaac 与
MuJoCo 必须分别从自己的 achieved FK 对同一 measured teacher 计误差，不得用 retargeted q 自生
teacher 再宣称 parity。

V2 reward 已在实际 profile 上改动；冻结历史误差上收紧/初始 adaptive sigma 的
window 收入为 `2.664360/2.872667`，且对实际73 catalog 的静态会计是73/73满足
max motion `3.6575` < target `4.0296/4.3104` < landing `6`。迁移前两端仍须逐项匹配 `e/sigma`、有限差分
改善、eligible denominator 与 discounted per-swing 训练收入，并验证
`动作模仿 < 目标击球 < 上台结果`，不能用 static/counterfactual 结果代签已学会。

## Goal

Test whether a policy learned in Isaac can be replayed or approximated in MuJoCo.

This gate is the sim-to-sim bridge before real deployment.

## Inputs

- Isaac-trained policy ONNX from G05 (exported with the full metadata contract).
- MuJoCo A3 model from G04 (`a3_pingpong.xml`).
- Shared joint order and observation/action contract (`docs/interfaces/policy_observation_action.md`).

## Outputs

- Replay/evaluation procedure with Isaac-exact metrics.
- Cross-sim metrics and known mismatch list.
- Decision on which MuJoCo configuration is deploy-faithful.

## Related Directories

- `hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py` — the parity evaluator.
- `agi/a3_deploy_example/` — active deploy tree: `MUJOCO_VALIDATION_RUNBOOK.md`, `SIM_DEPLOY_REHEARSAL.md`, `SIM_FIDELITY_NOTE_FOR_AGI.md`.
- `agi/A3_MuJoCo_Sim/` — vendor AimRT MuJoCo sim (the explicit-PD subscriber lives here).
- `agi/code_deployment/a3_deploy_example/` — older vendor reference subset.

## Operation Docs

- [../operations/run_training.md](../operations/run_training.md)
- [../operations/run_deploy_dryrun.md](../operations/run_deploy_dryrun.md)
- [../operations/run_phase1_task_revision_0p5_exam.md](../operations/run_phase1_task_revision_0p5_exam.md)
- [../operations/run_phase1_signed_face_exam_k100.md](../operations/run_phase1_signed_face_exam_k100.md)
- [../operations/run_shared_interface_rehearsal.md](../operations/run_shared_interface_rehearsal.md)
- [../operations/run_gate3_first_tick_harness.md](../operations/run_gate3_first_tick_harness.md)

## Acceptance Criteria

- The same action ordering is verified in both simulators.
- The exported deploy ONNX (not a re-export) runs in MuJoCo with the training observation rebuilt exactly.
- Divergence sources are documented: contact, latency, actuator, timestep, observation delay, model mismatch.
- Exact-strike metrics from Isaac are reproduced in MuJoCo and recorded per accepted checkpoint.
- Before MuJoCo training starts, each explicit effective-plant profile reproduces a byte-frozen
  Python MuJoCo evaluator on a reset-first 179-D observation and a fixed short 31-D action tape.
  Per-term rewards must separately match an independent reward-replay oracle; the current evaluator
  has no training-reward API.
- The MuJoCo `VecEnv` completes deterministic reset, finite rollout, at least one PPO update,
  checkpoint resume and deploy export while recording measured throughput and a complete engine-bound
  training contract.
- Formal return training/scoring uses physical ball-racket-table/net contact and landing state;
  analytic virtual return remains diagnostic and cannot promote a policy.
- The first fine-tune paper is preregistered before launch: same source checkpoint, frozen control
  versus warm-start fine-tune, equal budget, multiple seeds and an immutable held-out
  [K100](../DEFINITIONS.md#q50-and-k100) (100 fixed questions, 50 per side) with
  per-side fall/hit/return. Final promotion still requires independent vendor Gate3/Gate3B.
- The selected 0.5-second vendor paper preserves all 100 question rows and order, starts every question
  from zero-velocity motion frame 0, and reaches contact at tick 25 of 50 Hz control. Each row must
  traverse the same production planner, C++ runner, vendor MJCF and effective plant, then emit
  attempt/completion/hit/return/fall/deadline fields without deleting failures.

## Current State

### 2026-07-20 W/Y export is diagnostic-only; lineage precedes the vendor adapter

Both real W/Y zero-write plans passed on exact `origin/main@a0c1284`, and fresh `179→31` ONNX artifacts passed
an independent structural check plus finite CPU ONNX Runtime inference. W's ONNX SHA-256 is
`ee0e2e83c8f3dc8302fcef609fe13b2feaf69e247e39f405d1ea6c30b652d970`; Y's is
`72da43d96ab9dd95e1da6aba2ed548ad26e61863b70cf8120c120132b7b8f995`.

The promotion gate nevertheless fails closed before vendor behavior: both checkpoints record
`training_contract_lineage_exact=0`, and both ONNX exports record `training_contract_exact=0`. They are useful
diagnostic artifacts, not deployable policies. The local prerequisite chain is:

1. remediate or retrain W/Y with exact checkpoint lineage and re-export an exact-contract ONNX;
2. implement the frozen K100-to-serve/`task_revision`/production-planner/C++-runner/vendor-MJCF adapter;
3. run and preserve all 100 vendor rows before comparing or promoting either candidate.

This is a dependency description, not a competing priority list; global execution order is owned only by
[`docs/NOW.md`](../NOW.md).

The native MuJoCo trainer remains valuable, but it follows this lineage-and-adapter chain rather than replacing
it. A separate processed-qdes action-slew matrix may run as a single-swing diagnostic; it cannot satisfy this
Gate or bypass the continuous-recovery order `T0 → T1 → T2`. G06 remains `Partial`, and `Gate3-D0` remains
`Open`.

### 2026-07-19 demo-priority vendor same-paper is preparation-only

A local read-only source audit found no runnable production path from the 0.5-second timing paper to
the vendor chain. The two demo-priority candidates are `W` (racket-position priority with the
non-striking arm free) and `Y` (racket-position priority with imitation muted in the strike window);
`U` (racket-position priority with a stronger ready pose) remains the stable fallback.

The completed Isaac K100 drives the policy directly and bypasses the production planner. The Python
`mujoco_eval_onnx.py` path supports 179-D observations and a fixed bank, but it does not consume the
per-question timing paper or retime every row to 25 control ticks. Gate3's fake-ball input accepts a
flat `N × 6` serve list (initial position plus velocity), and no adapter currently maps timing-paper
rows through serve generation, same-ball
[`task_revision`](../DEFINITIONS.md#planner-task-revision), the production planner and the vendor
runner. The old `pp_gate3_rally.sh` / `pp_rally_conductor.py` path remains quarantined and forbidden.

One exact, read-only filesystem-wide Pod1 search has now located one W and one Y `model_6700.pt`.
Both checkpoints load with embedded iteration `6700`, `74` floating tensors / `1,762,715` floating
elements / zero non-finite elements, and actor dimensions `179→31`. Each run also contains
`params/training_contract.json`, `env.pkl`, `agent.pkl`, and `env.yaml`. This closes only the static
training/export-input check; it is not vendor behavior or parity evidence.

The standalone exporter now has a genuinely zero-write `--plan`. It uses a weights-only checkpoint
load, requires a non-negative integer `checkpoint_iteration`, validates finite checkpoint materials,
the donor, motions, harvest, train bank, contract and formal face-179 envelope, and exits before the
first directory/temp/graph/artifact write. Its JSON reports `artifact_written=false`,
`graph_export_not_executed=true`, dimensions and formal-material status. The five-file focused suite
passes `97` tests in `0.38s`, including the unchanged normal-export fake smoke. Neither real W/Y plan
has run on a Pod and no ONNX artifact has been created.

The next runtime capability is the adapter described in
the acceptance criteria above, with the exact same 100 questions (50 per side), frame-0 zero
velocity, 25 ticks, forehand time scale `2.64`, backhand time scale `1.8`, and the same planner,
MuJoCo XML model (MJCF) and effective plant. Until that adapter and its per-question output exist,
W/Y are candidates rather than a successful demo. G05 and G06 remain `Partial`; `Gate3-D0` remains
`Open`. Detailed evidence and the frozen output contract are in the
[half-second sprint record](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md).

### 2026-07-17 exact-0.5 Isaac K100 remains upstream of MuJoCo parity

The first checkpoint-bound [0.5-second timing exam](../DEFINITIONS.md#timing-exam-0p5) launch failed
closed before evaluator creation because its immutable bank was absent; v1 is permanently consumed.
The bank/report were then restored with exact size/SHA and no-clobber permissions. The asset-restored
v2 supervisor binds harness `be17289c…cc59`, activation `2b91248b…0626`, the v1 failure receipt,
`taskrev_p2_equal_reward@model_5700`, all 100 exact-25-tick attempts and a fresh state/output namespace.
Its focused source suite is `41 passed, 1 skipped`; the skipped delegated-cgroup probe and the v2
RunPod behavior exam are still open.

G06 therefore remains `Partial`. Even a completed Isaac result is explicitly inexact; the same
checkpoint and immutable questions must still run in vendor MuJoCo before any parity or deployment claim. See the
[experiment](../experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md) and
[operation](../operations/run_phase1_task_revision_0p5_exam.md).

### 2026-07-15 analytic Reward is not the physical referee

The current VirtualBall task does use achieved racket FK state to analytically predict contact, net
crossing and landing, but those outcome terms remain a training model with dense partial credit. The
separate Isaac Phase-A engine-integrated ball diagnostic is metrics-only, and the live recipe leaves
racket impulse off; there is therefore no current physical-return reward or policy result. Before comparing analytic versus
physical outcome reward, Phase-B hit/net/landing events and all-serves denominators must close; Phase-B's
paddle impulse still reuses the analytic contact law. The same
actor/racket trajectory must be replayed against the Agibot vendor MuJoCo referee. Detailed exact-source
semantics are in [the Reward truth audit](../experiments/2026-07/EXP-P1-REWARD-PHYSICAL-TRUTH-AUDIT-20260715.md).
No Gate3/Gate3B score changes, and G06 remains `Partial`.

Done (2026-06-27 → 2026-07-02, recorded 2026-07-03):

- The parity procedure exists and is battle-tested: `scripts/mujoco_eval_onnx.py` loads the exact
  exported deploy ONNX, reads the whole actuator contract from ONNX metadata (joint_names,
  default_joint_pos, action_scale, kp/kd, body_names — fails loudly if missing), auto-detects the
  175-D deploy-parity vs 180-D legacy obs contract, rebuilds the Isaac actor observation in MuJoCo
  (same frame math; the deploy-honest racket-target reframe is verified by
  `scripts/realsensor_obs_reference.py`), and reproduces Isaac's exact-strike metrics
  (pos/vel/normal pass, composite, hit-speed error, velocity attainment) with per-clip
  forehand/backhand breakdowns and per-step CSVs.
- Historical diagnostics implicated actuator PD integration: with the same ONNX and
  byte-identical `a3_pingpong.xml`, MuJoCo with `implicitfast` + kd in `dof_damping` was stable with
  clean swings, while the AGI deploy sim's
  explicit-Euler PD path (`joint_actuator_subscriber.cc`, MJCF without an integrator attribute,
  passive damping not zeroed) diverges within ~0.1 s. Switching only the PD integration moved
  hit-speed error 0.61 → 0.31 m/s and velocity attainment 0.35 → 0.88. This comparison did **not**
  prove Isaac equivalence because passive kd bypassed the total effort clip; the 2026-07-14 audit
  below revokes that exactness interpretation while preserving the numbers as diagnostics.
  Historical one-flag reproduction:
  `--pd-mode implicit` vs `--pd-mode explicit --keep-passive`. See
  `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md`.
- Historical verdict stance (2026-07-02): implicit PD was treated as the Isaac-faithful cross-check,
  but that label is superseded by the 2026-07-14 total-effort correction below. The
  binding pre-hardware gate is the AGI explicit clipped-PD MuJoCo run ("falls in MuJoCo = falls on
  the real robot"). The deployed policy was fine-tuned to survive it
  (`launch_explicitpd_ft.sh`, exported via `export_onnx_explicitpd.sh`).
- A deploy-faithful episode protocol exists: `--deploy-faithful` mirrors the C++ runner
  (nominal-stand start, windup hold with pinned time_to_strike, one full clip per swing, rest
  between swings, no teleports, absolute fall terminations only), reporting swing completion rates
  and time-to-fall.
- Eval mode B exists (2026-07-04): `--target-source venue-balls` (`mujoco_eval_onnx.py` +
  `scripts/venue_ball_sampler.py`) samples fitted venue incoming balls (with spin), StrikeSpec-
  inverts the demanded racket state (pos/vel/normal, sign-matched to the swing side's reference
  face), drives the unchanged target pipeline, and scores a virtual return at the exact-strike
  frame (capture gate → venue contact model → drag+Magnus flight → bounds + net clearance).
  Headline reported as `return_success_rate` per strike; mode-A (`boxes`) output stays
  byte-identical. First run: pos/vel tracking survives the OOD venue distribution (3.7 cm /
  0.18 m/s) but the face normal is clip-locked (36-76° err, 0% legal returns) — the 175-D
  contract has no normal channel (`docs/motion_and_contract_v3.md`). v1 caveats: uncorrelated
  box sampling, human-receiver contact heights (0.98-1.26 m vs trained 0.72-1.13 m —
  intentional realism, expect pos_pass to drop), incompatible with `--deploy-faithful`.
- The normal counterfactual is a committed output (2026-07-05; was an ad-hoc uncommitted
  analysis on 07-04): every venue strike is auto-rescored with the DEMANDED face normal swapped
  into the achieved kinematics — `cf_*` columns after the 14 venue columns + a CF summary
  block. Committed record (P2 product line, 9600 steps seed 0, 44 strikes): actual 0/44 vs
  counterfactual 44/44, CF median landing error 0.10 m; the 07-04 2400-step run reproduces
  byte-identically (first 43 CSV columns). The face-orientation channel alone fails the return.
- Fixed-normal inversion exists and delivered a verdict (2026-07-05): `--venue-fixed-normal`
  pins the StrikeSpec normal at the clip reference face (`solve_fixed_normal`, velocity-only
  LM; free `solve()` untouched; 16/16 planner tests). Result: the path-A ceiling is ~0% — a
  brute-force reachability scan (face pinned, all |v_r| ≤ 6 m/s, ~7k landings/ball) shows the
  forehand face ([0.41,0.90,-0.17], near-sideways) lands x ≤ 1.4 m at ANY racket velocity
  (never clears the net at 1.87 m) and the backhand face only reaches a net-hugging cross-court
  sliver (x≈1.9-2.0, |y|≈0.3-0.67) outside the legal landing box (≥0.3 m depth guard =
  training's own dink rule). Premise verified: mode-A achieved normal is within 1.9° of the
  clip reference, so the pinned face IS the policy's face. Planner adaptation cannot rescue the
  clip-locked face; the normal-channel contract change (175→179) is the only path.
  Evidence: pod `/workspace/franco/cf_eval/` (scan_reachability.py, modeB_*.log).
- A deploy-parity mid-swing switch stress protocol exists (2026-07-05): `--switch-stress P`
  (multiswing only; default off = byte-identical) aborts the swing each step with probability P
  exactly like the deploy runner's planner re-decides (training `clip_switch` semantics:
  uniform new clip, windup frame, fresh hold + target, robot untouched; tracking guards off —
  balance falls + timeout only). Reports switches, falls, 2 s post-switch survival, post-switch
  vs clean-swing hit rates. First matrix ({P2, R11} × {implicit, explicit+keep-passive} ×
  {~0, 0.002, 0.01}/step, 24000 steps each): zero falls in all 12 runs, 100% post-switch
  survival, post-switch hit rate ≈ clean — the switch discontinuity alone does not topple even
  the non-switch-trained P2 in MuJoCo; R11's in-distribution hit-rate tax remains visible on
  the explicit gate (0.98-0.99 vs P2's 0.99-1.00). Logs: pod `/workspace/franco/cf_eval/sw_*`.
- A documented validation flow with an acceptance-criteria table exists:
  `agi/a3_deploy_example/MUJOCO_VALIDATION_RUNBOOK.md` (rate ~50 Hz, sync stable, infer < 20 ms,
  projected gravity sanity, bounded actions, neck passive).

Not done:

- Formal per-checkpoint acceptance: the metric thresholds and the numbers for the currently shipped
  checkpoint (`model_p4_deployparity` / explicitpd_ft `model_25700`) are not yet pasted into this
  gate as an accepted record.
- (Fixed 2026-07-03, branch `audit-leftover-fixes`.) `eval_realsensor_hopex.sh` /
  `export_onnx_explicitpd.sh` now resolve their own location and take `HOPE_EVAL_*` /
  `HOPE_EXPORT_*` env overrides, and `mujoco_eval_onnx.py` resolves strike phases as CLI >
  ONNX `clip_strike_phases` metadata > built-in legacy `(0.36, 0.50)` (plus a
  `clip_seg_lengths`-vs-npz mismatch warning). The `--onnx`/`--motion-files` defaults still point
  at a legacy run — pass current artifacts explicitly.
- **Decision recorded 2026-07-12:** native MuJoCo training/fine-tuning is now a P0 implementation
  track; start with native CPU MuJoCo and measure the A3 workload before choosing an accelerated
  backend. The current code remains validation/dry-run only, and vendor Gate3/Gate3B remains an
  independent final arbiter.

## Risks

- A policy can appear valid in Isaac but fail in MuJoCo because of actuator/contact mismatch — this
  happened (explicit-PD divergence) and cost significant time before the root cause was isolated.
- Evaluating with the script's stale defaults silently tests the wrong contract; always pass the
  checkpoint's own clips/phases.

## Next Steps

1. Remediate W/Y checkpoint lineage and produce a fresh ONNX with both checkpoint lineage and exported
   training contract exact; keep the current inexact artifacts diagnostic-only.
2. Implement the frozen 100-row vendor adapter, then run the exact same paper through serve generation,
   same-ball `task_revision`, production planner, C++ runner, vendor MJCF and effective plant.
3. Implement the frozen evaluator semantics independently in a native MuJoCo `rsl_rl VecEnv`; keep
   trainer and evaluator imports separate, pass reset/action-tape parity plus an independent reward
   replay canary, then require one finite PPO smoke before any long run.
4. Preregister and run the same-checkpoint frozen-control versus warm-start-fine-tune multi-seed
   held-out K100 paper; do not let the training environment grade itself.
5. Record the accepted sim2sim numbers for the shipped checkpoint (implicit cross-check + explicit
   clipped-PD gate + `--deploy-faithful` protocol) in this gate.
6. When the mocap→planner bridge lands, extend the MuJoCo rehearsal to consume live
   `/racket/command` targets instead of sampled planner-equivalents
   (`docs/operations/run_shared_interface_rehearsal.md`).

## Audit update 2026-07-10: formal BankExam ruler

The old headline scores are not a trustworthy promotion ruler. The evaluator
had an exact-strike one-step offset, omitted pre-strike failures from its
denominator, compared different question slices across noise columns and did
not enforce the held-out split. These are now closed:

- one immutable schedule with stable question IDs and per-attempt seeds;
- all scheduled attempts remain in the denominator;
- every noise/model column receives the same ordered questions;
- train/exam split, motion SHA/order/frame and physics-source lineage are
  fail-closed;
- every formal attempt starts from the MJCF named `stand` keyframe with all
  hidden state and last action reset; teacher-reference reset is diagnostic;
- schedule, ready-state, MJCF and resolved execution-contract SHA are emitted
  in summaries and attempt CSVs;
- actuator integration, armature, ctrl/velocity limits and q-des contract come
  from schema-v3 rather than observation width guesses.

Non-zero PhysX joint friction has no exact MuJoCo `frictionloss` equivalent.
Formal BankExam therefore refuses it. `--allow-inexact-contract` may run a
direct-number proxy, but the result is stamped
`evaluation_contract_exact=false` and cannot be booked. Here `exact` means the
listed execution protocol is bound; it does not claim complete cross-engine
dynamics equivalence.

All key historical scores must be rerun after fresh export; retain old values
only with an explicit `old scorer` label.

The 2026-07-11 local Phase-1 snapshot also contained a NumPy
`virtual_return_scorer.py` and a saved-run `termination_contract.py`.  They
were initially retained as simulator-independent specifications.  The current
schema-v3 adapter branch now closes both production seams without modifying
the physics-hash-bound `venue_ball_sampler.py`:

- `mujoco_eval_onnx.py` delegates actual and counterfactual returns to the
  NumPy 10 ms RK4/ball-centre-plane scorer and binds scorer source, venue YAML,
  parameters and score spec into the execution contract;
- `bank_exam_schedule.py` materializes a balanced, canonical JSON paper with
  an exact per-clip quota, immutable content IDs, deterministic hold values and
  per-attempt noise seeds. Its hashed release rule defines `H` ready-stand
  actions followed by raw clip frame 0. MuJoCo accepts it with
  `--exam-schedule-json`; Isaac consumes the same artifact;
- `isaac_bank_exam.py` keeps the saved train bank untouched, installs one
  evaluator-owned exam row per environment after a nominal-stand reset, emits
  raw all-attempt JSON/CSV, and invalidates the whole cell on truncation.
  Exact cells additionally verify the runtime train-bank schema/family/SHA;
  historical legacy banks are allowed only in the explicit inexact canary lane
  and are recorded as an inexact reason.

Dependency-light verification on 2026-07-11 passed `67` adapter/audit tests
with one optional Torch parity skip, `85` formal CPU contract tests and `141`
unique tests in the combined contract run with the same optional skip. This is
implementation evidence, not a gate pass:
the shared-paper Pod canary and question-order/hash equality across both
simulators are still pending.  M3f/M2/G1 predate exact schema-3 checkpoint
binding, so their canary cells must say `evaluation_contract_exact=false`; only
a fresh exact-lineage model can produce a bookable score.

The M2 Isaac quota-10 leg has now passed runtime artifact validation: all 20
scheduled rows are present and uncensored, its bank/schedule SHA and ordered
IDs match the supplied paper, and its diagnostic return rate is 16/20. The
matching MuJoCo q1 leg initially stopped before rollout because the historical
`obs_norm.npz` has four zero std dimensions. They are valid constant features
under the saved `(obs-mean)/(std+eps)` implementation with `eps=1e-2`.
MuJoCo now accepts finite non-negative std only when every `std+eps` divisor is
strictly positive; negative/non-finite scales and unprotected zeros remain
fatal. A rerun is required, and cross-simulator canary status remains pending.

The next MuJoCo pre-rollout attempt exposed a main/rollout scope error in
`training_hold_protocol` and an avoidable dependency on the shell variable
`HOPE_STAGE1_QB`. One pure helper now derives hold-aware guard semantics in
both scopes. BankExam also resolves the current checkout's dependency-light
`stage1_question_bank.py` directly and records its SHA in the execution
contract, rather than importing the Isaac task package or trusting ambient
shell state. Both failures occurred before rollout and produced no score; the
same-paper rerun remains required.

That rerun and the full single-question diagnostic matrix are now complete.
At quota 10, M3f/M2/G1 MuJoCo return was 17/20, 10/20 and 9/20; G1 backhand was
0/10 in both engines. At quota 50, M3f returned 91/100 in MuJoCo versus 99/100
in Isaac, while M2 returned 51/100 versus 86/100. Both survivors also completed
the same-paper 5% action-noise and second evaluation-seed cells. Every ledger
was complete and uncensored, and every cross-engine bank/schedule SHA and
ordered ID check passed. All MuJoCo `fell` rows were tracking guards rather
than absolute physical falls. The detailed per-side table and result hashes
are in `docs/PHASE1_SCHEMA3_RESULTS_2026-07-11.md`.

MuJoCo carry-state BankExam remains a separate inexact diagnostic. Its summary
now includes `return_and_recover_rate`: among paper rows that have a scheduled
next opportunity, a row counts only if it legally returns and naturally
completes its swing. A post-strike guard preserves the return result but fails
recovery. The final paper row is excluded from this product denominator.

The completed q50 carry-state cells produced return-and-recover rates of
70/99 for M3f and 30/99 for M2. Overall returns were 82/100 and 40/100; no
absolute physical fall occurred, while tracking guards/timeouts remained
failed opportunities. Summary SHAs are `091bd045...0e6ea` and
`5658b7cc...b8774`. This is useful candidate ranking but remains an inexact
continuity diagnostic; Isaac continuous and a fresh exact-lineage policy are
still required for gate completion.

The historical main-matrix extension is also complete. At clean q50, R1b
seed 1/2 returned only 15/100 and 17/100 in MuJoCo (both 3/50 forehand),
despite 95/100 and 90/100 in Isaac, and stopped before robustness. C1 returned
50/100 in MuJoCo versus 96/100 in Isaac and advanced. Its MuJoCo noise and
second-schedule cells returned 48/100 and 55/100; its carry-state cell returned
42/100 and both returned+recovered on 26/99 next-opportunity rows. No C1 cell
had an absolute physical fall. M3f therefore remains the historical diagnostic
leader (`91/100` clean and `70/99` continuity product), while all of these
cells remain `evaluation_contract_exact=false`.

The formal friction gap now has a training-side, fail-loud control rather than
an undocumented source edit. Fresh runs may set
`task.plant.zero_joint_friction=true`; `train.py` then zeros every actuator
friction field before environment construction, and the existing schema-v3
runtime fact collector records the expanded zero vector. The checked-in
non-zero plant remains unchanged by default and is still diagnostic-only in
this gate. Override/contract unit tests passed `60` tests in an isolated Pod
worktree; the training entry also refuses to continue unless the instantiated
contract contains exactly 31 aligned zero coefficients. This does not complete
G06: a from-scratch schema-v3 checkpoint on
migrated schema-2 motion, a bound train bank, export, and exact BankExam are
still pending.

The export/judge replay path now preserves the two new runtime controls instead
of composing the default plant/layout after training. For a schema-3
checkpoint, `judge.sh` reads the adjacent hard contract: exactly 31 zero
friction coefficients restore `task.plant.zero_joint_friction=true`; the
declared non-zero default remains false; partial-zero, malformed, negative or
non-finite vectors fail closed. The same sidecar supplies the validated
175/179/181 actor contract and is cross-checked against saved face/station
flags. Face-command enabled state and pairing, legacy-motion permission and
motion exactness flow into ONNX metadata. Thus a legacy causal export remains
explicitly inexact while a future fresh zero-friction export can reach the
formal MuJoCo plant check without a compose mismatch. The dependency-light
contract/judge regression now passes `38` tests. No terminal fresh checkpoint
or exact BankExam result exists yet, so this gate remains `Partial`.

The 179-D exact-construction smoke now proves the export inputs can coexist in
one live contract: schema-2 runtime-order motion, schema-v3 bank, shared face
pairing and a 31-zero plant. Both fresh seeds wrote `model_0.pt` with schema-3
contract SHA `3a3b3d95...b9972` and embedded lineage exact `1`; the four causal
`model_17000.pt` files bind their own sidecars with lineage `0`. `judge.sh`
dry-run resolves the canonical adjacent exam banks and now adds
`--allow-inexact-contract` only for diagnostic motion/pairing contracts, while
fresh exact candidates receive no escape. It also resets `PYTHONPATH` from the
current checkout's `setup_train_env.sh`, preventing another user's Pod checkout
from supplying export code. Terminal export and same-paper Isaac/MuJoCo cells
are still pending, so G06 remains `Partial`.

The evaluation cadence is no longer terminal-only. Two checkpoint-curve
workers attempted the missing causal `17000/18000/19000` and fresh
`0/1000/2000` immutable BankExams. The first two attempts are preserved as
evaluator preflight failures (missing ignored A3 asset link, then buffered
export-success handshake despite an ONNX file), not booked model results. The
links now resolve only to the frozen training assets and the retry uses
unbuffered export output. A third preflight correctly reached sidecar creation
and exposed the known four constant observation dimensions. The sidecar writer
now preserves finite zero std only under its bound `eps=0.01` and still rejects
negative/non-finite or non-positive divisors; both Pods reproduced the same
four zeros with valid SHA-bound output. Each Pod serializes the Isaac export phase;
after an export reaches MuJoCo, CPU exams may overlap with
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Every result directory
is checkpoint-specific, so no old ONNX or normalizer is reused. The workers
run from the detached evaluator while both training checkouts remain clean at
`6d93bcb`. Diagnostic pairing/motion still receives the explicit inexact
escape; the `SZ` target cell may not. No curve result is booked yet, and G06
therefore remains `Partial`.

The inexact escape is now a one-way result downgrade in both simulators. An
exact-provenance fresh checkpoint evaluated with legacy face pairing (`LZ/LP`)
is allowed only as a diagnostic and must emit
`evaluation_contract_exact=false`; MuJoCo applies this when assembling the
bank contract, and Isaac records the pairing as an inexact reason before its
scorecard. `SP` remains a non-target plant ablation even when its bytes are
fully reproducible. This prevents the 2x2 diagnostic grid from laundering a
formal target label.

The next checkpoint preflights closed two more evaluator-only blockers. The Pod CPU venv contained
`onnxruntime` but not the `onnx` graph package required by formal inspection; both Pods now pin
`onnx==1.22.0`, and the generated 179-D graphs pass checker and runtime. Fresh exact models then
stopped before rollout because Isaac's float32 metadata representation of the same MJCF armature
decimals differed by at most `2.71e-9`, while the comparison threshold was `1e-10`. Passing that
field exposed the same representation issue at the `118.2` ankle effort limit: float32 metadata is
`118.199996948...` (`3.0517578e-6`, about 0.4 ULP). Formal plant comparison now requires exact
float32-grid identity rather than a field-specific tolerance and tests both sides of the 0.5-ULP
boundary plus next-grid rejection. A separate report fix propagates final artifact/escape exactness into the denominator
section, so a legacy causal report can no longer display `true` while its summary JSON says `false`.
These preserved attempts are not model scores. A corrected exact fresh BankExam is still required,
so G06 remains `Partial`.

Formal retry then proved why report code must not live in a physics-hashed module: changing only
`BankExamSampler.denominator_report()` changed the complete `venue_ball_sampler.py` SHA, and the
schema-v3 bank refused export before rollout. The sampler is restored byte-identically
(`00e28e85...30cc`), while final artifact exactness is now substituted by the outer MuJoCo
evaluator. Only the recorded judge PGIDs were terminated after Isaac's failed shutdown hung; no
training process was signalled. This retained attempt is not a score.

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_bank_exam_schedule.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py
```

The corrected float32-grid plant gate has now produced the first exact fresh
MuJoCo checkpoint curve. On the same clean q10 paper, `SZ` seed 1 scored
`0.00/0.50/0.90` and seed 2 `0.00/0.50/1.00` at `model_0/1000/2000`;
all six reports say `evaluation_contract_exact=true`. At 2000 the side splits
were FH/BH `0.80/1.00` and `1.00/1.00`. These are successful formal direction
screens, not q50 acceptance cells. The 20000 causal rows stayed explicitly
inexact: M3 old/S1 `0.45/1.00`, M2 old/S1 `0.50/0.50`.

The current single-question BankExam also does not certify real continuous
timing. Live training preserves state across natural clip wraps, but the
complete-clip schedule is materially slower than the conservative venue A-B-A
sample and installs the next target after the observed opponent-hit event.
The future continuity gate must use the same immutable question **and interval**
schedule in Isaac and MuJoCo, report per-opportunity carry-state failures, and
retain zero resets/teleports. The reproducible timing audit and required metrics
are in `docs/research/phase1_continuous_rally_timing_2026-07-11.md`. G06 remains
`Partial` pending q50, Isaac same-paper companion results, terminal lineage
verification, and event-driven continuity evaluation.

The causal terminal cadence no longer waits for an impossible filename. The
first normal M2-S1 completion proved that a continuation resumed at 16999 for
4000 updates finishes/saves at iteration 20998. Its terminal checkpoint is
finite and SHA-bound. The later paired terminal q10 judged M2-old/S1 at
`0.40/0.35` aggregate (both forehands zero), but remains an inexact,
non-decisive direction screen. Cadence and scale-out causal manifests now
target `model_20998.pt`; the exact waiting-worker PGIDs were replaced without
signalling trainers or fresh workers. This changes only checkpoint discovery,
not the immutable exam or causal `evaluation_contract_exact=false` rule.

Cross-engine exactness for `SZ` is deliberately narrow. All-zero friction is
byte/semantics reproducible, but prior frozen-plant evidence says it is not a
safe proxy for the deployment plant. Conversely, current `SP/LP` non-zero
coefficients cannot be made exact by feeding the same numbers into MuJoCo
`frictionloss`, because the physical meanings differ. G06 therefore has no
deployment-qualified plant cell yet. Closure requires a measured, versioned
friction model with engine-specific adapters, a fresh `SC` training cell and
the full train-plant x eval-plant transfer matrix; until then, `SZ` scores can
validate the evaluation contract but cannot clear sim-to-real parity.
`SP` is consequently an inexact diagnostic, with an explicit evaluator escape;
it cannot be booked and cannot block the later formal SZ jobs in the same
milestone-major queue.

The offline plant-contract v1 boundary now implements the fail-closed half of
that closure plan. It refuses non-zero cross-unit numeric conversion, requires
one content-addressed latent model plus independent PhysX/MuJoCo fit and probe
reports, checks the canonical 31-joint order and rejects requested runtime
envelopes outside calibrated load/speed/temperature/pose support. Crucially,
the final MuJoCo leg is not a generic standalone evaluator: it must bind the
Agibot vendor `a3_pingpong.xml`, the Gate3/Gate3B runtime source and a raw
31-joint adapter-instantiation report. Current BankExam remains useful
development/selection evidence but cannot substitute for that vendor-runtime
cell. No calibration bytes, passed runtime probe, vendor instantiation report
or fresh `SC` checkpoint exists; the compiler is not wired to either engine,
so G06 remains `Partial`.

Original causal terminal and original fresh exams now run in separate
workers/state directories, so neither checkpoint-availability order can block
the other. Q10 manifests declare the screen-only/no-promotion policy at both
manifest and job level, and the checked-in `phase1_checkpoint_curve_worker.py`
rejects omissions or contradictions, checks the schedule, and requires the
same canonical screen-policy-plus-job contract SHA before reusing a completed
state. This is an
operational guard, not permission to
book q10; q50 and the same-paper Isaac/MuJoCo pair remain the decision gate.
Pod1 fresh starts at 4000 because that checkpoint had not existed when the old
combined worker was replaced; Pod2 4000 was already handled and starts at 6000.

The first corrected terminal MuJoCo q10 pair is preserved at M2-old/S1
`0.40/0.35` aggregate (FH both `0/10`; BH `8/10`/`7/10`). Both are inexact
diagnostics and the prefix is too small to decide. Full result/checkpoint/report
hashes are tracked in `configs/phase1_M2_terminal_q10_pair_20260711.json`;
neither cell advances or stops without q50 and its Isaac companion.

The matching Pod1 M3 terminal pair is now also complete and finite. M3-old's
`model_20998.pt` has SHA `320b77c9...417a`, matches adjacent contract
`7542c59b...d941b`, and carries causal/inexact lineage. On immutable schedule
`7a908142...d614`, M3-old returned FH/BH/aggregate
`0.50/0.40/0.45`, while M3-S1 returned `1.00/1.00/1.00`; paired aggregate
delta is `+0.55`. This triggered the separately frozen K=100 q50 paper. On
that shared 50-per-side schedule, M3-old returned FH/BH/aggregate
`0.62/0.22/0.42`; the raw ledger has one physical fall plus eight guard resets
(the legacy summary's `fell=9` is their union), while M3-S1 returned
`1.00/1.00/1.00` with zero such terminations. Aggregate delta is `+0.58`, so M3-S1 wins
the MuJoCo terminal selection inside this same legacy swing-family causal
paper. Both results remain `evaluation_contract_exact=false`. The same-paper
Isaac companion then scored both cells `0.98/1.00/0.99` FH/BH/aggregate,
delta zero, on the identical question order. It does not reproduce the MuJoCo
ranking, so cross-engine selection, continuity and calibrated plant remain
open. Full terminal and paired bindings are in
`configs/phase1_M3_old_terminal_audit_20260711.json` and
`configs/phase1_M3_terminal_q10_pair_20260711.json`; q50 execution/result
hashes are in `configs/phase1_M3_terminal_q50_result_20260711.json`; the Isaac
ledger is `configs/phase1_M3_terminal_q50_isaac_result_20260711.json`.

The four newly refilled causal workers have also been corrected from eval
`46a0ce2`'s legacy state schema without changing their judge paper. Only the
four exact, childless legacy worker PGIDs were TERM-signalled; hardened PGIDs
are Pod1 `1416771/1416784` and Pod2 `198759/198771`. Each rejudged 17k state
returned zero and now binds manifest, job spec, job contract, checkpoint,
judge and both clean commits. Old state/log bytes remain immutable beside a
content-addressed correction sidecar. This closes provenance for future
milestones but does not change their causal `evaluation_contract_exact=false`
status.

The six older original/scale-out workers were independently hardened as well.
Current PGIDs are Pod1 `1432280/1432292/1432304` and Pod2
`200706/200718/200730`; no trainer or judge was signalled. Five available old
states were rejudged rc=0 and now bind manifest, job and job-contract SHA.
`configs/phase1_global_curve_worker_hardening_result_20260711.json` preserves
the exact signal scope and all transaction hashes.

Fresh SZ seed1 also closed its first exact checkpoint-selection q50. On one
K=100, 50-per-side paper, the analytic virtual-return scorer gave model 2000 FH/BH/aggregate
`0.66/1.00/0.83`, while model 4000 returned `0.00/1.00/0.50`; model 2000 is
retained. The whole arm continued at that paper's decision time and was only later stopped by the
separate 2026-07-13 operational resource decision. Both evaluations are exact/fresh, but
all attempts finalized through a non-physical post-strike guard, so this is
not a continuous or deploy-stability gate. The result is bound in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`. Its fresh
same-paper Isaac companion gave both checkpoints `0.98/1.00/0.99`
FH/BH/aggregate, delta zero. The MuJoCo ranking is therefore not reproduced;
the cross-engine checkpoint gate stays open. Companion hashes are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

Question-level forensics localize reproducible state differences but do not yet establish their
causes. Fresh model 4000's mean FH racket-center error is `13.15 cm` in
MuJoCo, beyond the frozen `9.5 cm` analytic contact margin on all 50 questions, versus
`2.48 cm` in Isaac; model 2000 is `9.03/3.03 cm`. M3-old BH has mean signed
normal error `168.15 deg`, and the later face-sign audit shows that the analytic
`VirtualReturnScorer` path may erase `n/-n` through `orient_normal`.
The earlier wording “MuJoCo physical outcome” was wrong: this evaluator has no simulated ball
contact and its incoming ball is visual-only. Both reported return cells are analytic outcomes
derived from racket state. Thus same question bytes/order are necessary but not sufficient, while
the exact engine/trajectory/scorer contribution remains unresolved. The forensic result is bound in
`configs/phase1_cross_engine_saturation_forensic_result_20260711.json`.

The next gate is preregistered as a strict 2x2: Isaac/MuJoCo x physical
truth/analytic counterfactual, with the original K100 order and capture/speed
thresholds frozen. Missing/duplicate/non-finite cells, changed order, or a
virtual-only physical cell all fail closed. Numeric Isaac ready/base/racket,
signed-face-before-orient and analytic state instrumentation is implemented.
Isaac PhysicalBall Phase-B source implementation exists at `612f54d`, but it has no accepted Pod
runtime, post-contact K100 ledger or content-addressed four-cell evidence manifest. Until those
runtime cells exist, G06 remains `Partial`.

Run `python3 scripts/validate_phase1_queue_governance.py` before any curve
manifest is copied or launched. The validator enforces the 142-job/24-slot
q10 screen contract and refuses q50 through the generic worker. Plant parity
remains separate: SZ is only zero-friction protocol exact, while SP/LP are
historical direct-number proxies. The repair contract is
`docs/research/phase1_plant_semantics_repair_2026-07-11.md`, status
`blocked_on_calibration_evidence`.

The v1 plant preregistration's source snapshot is now explicitly historical at
`d4ca566`; current strict-face179 `training_contract.py` bytes differ and the
current-checkout verifier fails closed. No current Gate3/SC result may consume
that stale snapshot. Re-preregistration of the complete current training,
adapter, judge and vendor-runtime closure is required before this plant leg can
advance; G06 remains `Partial`.

### 2026-07-12 final-engine priority

The final behavioral arbiter is the Agibot-provided A3 MuJoCo deploy chain called Gate 3 in
`docs/operations/run_pingpong_end_to_end.md`: fake ball -> real planner -> production-equivalent
C++ runner -> vendor MuJoCo. Isaac remains a fast training/diagnostic engine and native MuJoCo
training/fine-tuning is now P0; a win inside either training engine cannot promote a checkpoint that
fails Gate 3 balance, completion or recovery.
Continuous candidates must run without between-serve simulation reset and eventually satisfy
zero falls, zero operator rescues and complete recovery after every engaged swing. Gate 3B adds
the immutable stage distribution and hit/return scoring, but it does not weaken Gate 3 stability.

The current Isaac/MuJoCo gap is an open causal problem, not evaluator noise to average away.
The preregistered engine x physical/analytic 2x2 now has its Isaac PhysicalBall Phase-B source
mechanism, but the runtime gate remains closed until one clean-detached K100 ledger and
moving-blade substep audit exist. Plant semantics, ready-state, termination, observation/action
runtime and signed racket-face measurements must remain separately bound so a score difference
can be localized rather than hidden in one aggregate return rate.

The exact model-2000 SZ paper now adds a separate transfer warning: MuJoCo K100 return across
seeds 1/2/3/4 is `.83/1.00/1.00/.20`. This is not a plant fall signature (all physical-fall
counts are zero); it is checkpoint/learning seed instability on the current single-strike paper.
It blocks a stable Phase-1 checkpoint baseline before Gate 3. Do not average away seed 4, and do
not attribute the variance to Isaac/MuJoCo until the same checkpoints have the registered physical
instrument cells.

### 2026-07-12 MuJoCo training/fine-tuning P0

The project has promoted native MuJoCo training/fine-tuning from undecided/evaluation-only to P0.
This responds to repeated evidence that Isaac training metrics can stay high while held-out MuJoCo
strike execution and analytic return degrade. Matched physical-fall counts are zero, so balance
degradation is not established. The decision does not make the training environment the final judge.

The first backend must independently implement the frozen meanings from
`scripts/mujoco_eval_onnx.py`, must wrap batched native MuJoCo state as an `rsl_rl VecEnv`, and must load the
vendor A3 MJCF while bypassing the single-world real-time AimRT/ROS/GUI loop. It must not import a
shared observation/action/reward implementation from the evaluator, because shared mistakes would
create a common-mode false green. Its training contract binds engine/version, MJCF plus mesh
closure, resolved plant/PD/integrator/dt, runtime action post-processing, observation/action,
reward/termination/reset, question bank and source checkpoint hashes.

The preflight identifies two non-equivalent profiles. `isaac_bank_parity_v1` reproduces the current
schema-3 BankExam/Isaac profile; `vendor_gate3_v1` preserves the resolved 1 ms vendor plant, explicit
per-step PD, hard joint limits, neck override and frozen runtime flags. Loading the same source MJCF
does not make them equal because BankExam mutates the in-memory model. Reset/observation/action-tape
parity is judged against the frozen Python evaluator for the named profile. Per-term reward is
judged by a separate replay oracle: the evaluator records metrics and analytic virtual return but
does not implement PPO training rewards, and no independent C reward evaluator was found.

The first causal paper uses one exact source checkpoint: frozen control versus actor warm-start
fine-tune, fresh critic/optimizer, equal budget, at least two training seeds and an immutable held-out
K100. Formal return learning/scoring requires physical ball-racket-table/net contact and landing;
analytic virtual return remains diagnostic. A future MJX/MJWarp path is throughput work with its own
parity burden, not an exact-vendor label. Final promotion remains the unchanged vendor Gate3/Gate3B.

The tracked vendor MJCF currently has no ball, table or net and the existing analytic `BallPhysics`
driver is not wired into `MujocoSimModule::SimLoop()`. The 2--3 day `Trainer-v0` is therefore explicitly a
one-shot balance/strike-state fine-tune, not a physical-return or continuous-rally claim. Actor warm
start must load only actor/distribution/actor-normalizer state into a newly initialized critic and
optimizer; current `load_actor_tolerant()` does not enforce that boundary.
The separately tracked `mujoco-ball-wiring@4607410` handoff is not merged into this audited main and
has not passed its vendor build/runtime acceptance; formal physical-return work cannot consume it
until that independent gate closes.

No MuJoCo `VecEnv`, PPO smoke, training run or result exists yet. This decision adds an implementation
and acceptance path but does not close the engine gap; G06 remains `Partial`. The audited file
boundary, canary tapes, evaluator isolation and `Trainer-v0` sequence are recorded in
[the MuJoCo training-v0 preflight](../research/mujoco_training_v0_preflight_2026-07-12.md).

### Gate 3 face-command wire and engine-gap localization

The 179-D Phase-1 policies cannot be tested by adding `179` to a shape whitelist. Their last four
columns require the actor's raw mount-A world-frame normal and a zero rho placeholder atomically
paired with position/velocity. The external planner wire deliberately carries the physical,
opponent-facing striking face B instead. A versioned flat schema-2 publisher/receiver and exact
`deploy_parity_face179` ONNX metadata path are now implemented in source. The loader additionally
requires `face_command_enabled=1`, `shared_plus_y`, `mount_plusY_A`, an exact schema-3 train bank,
train split and lowercase content/source-family SHA-256 bindings; width and term names alone are
not enough. Schema-2 rows require a world-frame opponent-facing unit B normal (`B.x>1e-6`) and zero
rho. After clip selection the runner applies exact `[+1,-1]` to the normal only to recover raw A;
position and velocity are unchanged. Any malformed/unknown row after an active face tuple records
`invalid_after`; the publisher turns a bad solve or payload into an explicit finite `valid=0` row
on both wires, so silence cannot
keep an old swing eligible for the longer command timeout. Schema 1 remains the default for
existing models and cannot engage a 179 actor. This is not yet a gate result: the
vendor-source offline x86 build is recorded below, while a ROS/AimRT-enabled build, no-publish
first-tick parity trace, and full Gate 3 MuJoCo run are pending.
Active-swing fields are atomic. Post-swing recovery is not yet exact: the current Gate 3 runner
combines a synthesized base-anchored hold position with the previous swing's velocity/normal,
and no Phase-1 contract proves that hybrid tuple is on-distribution. A canonical recovery tuple
or separately accepted vendor-MuJoCo recovery paper is required before continuous promotion.
The physical-B positive-X invariant is also only a minimum sign/frame guard. Source now exports
and enforces a content-bound per-clip normal envelope from the exact training bank, as described below. A new
envelope-bearing formal ONNX, self-hit evidence and vendor behavior gate are still absent, so this
source guard must not be promoted into a Gate 3 result.

The same-policy Isaac/MuJoCo gap is localized in stages rather than one aggregate score:

1. replay identical joint/racket trajectories kinematically to isolate geometry, frames and scorer;
2. replay identical open-loop actions from a bound initial state to expose actuator/plant/integrator drift;
3. run closed loop with identical externally supplied observation rows to isolate policy/runtime timing;
4. only then compare each engine's native observation and physical contact in the full closed loop.

Each stage binds joint order, action scale/clamp, PD, dt/decimation, initial/ready state, signed face,
contact/termination and vendor MJCF SHA. Gate 3/Gate 3B is the final behavioral leg; Isaac remains a
training/diagnostic leg even if its score is higher.

#### 2026-07-11 isolated vendor-source build evidence

Source commit `8d56ea86f6450c198836969360bc133146934617` was archived into the isolated
Pod1 path `/workspace/codexschema/gate3_face179_8d56ea8`; neither the live training checkout nor
the eval checkout was changed. The local ONNX Runtime 1.19.2 archive used by the build has SHA-256
`eb00c64e0041f719913c4080e0fed7d9963dc3aa9b54664df6036d8308dbcd33`. A Release configure with
ROS messages and AimRT disabled built both `run_tests` and the actual
`a3_deploy_onnx_ref_pingpong` executable. Focused `PpPlannerInput.*:PpFace179Wire.*` was 10/10;
the full native suite was 195 passed / 4 skipped (only absent optional fixture/asset tests).
The test binary SHA-256 was
`1349038f5a3bd057026630f1fdcc9636cf68d5acef1041712911e2808140a1fe`; all 78 compile commands
contained the finite-safety flags and none contained `-ffast-math` or `-ffinite-math-only`.

This closes the offline vendor-source compile/test leg only. It does not exercise ROS/AimRT,
load a formal 179 ONNX, tick the production backend, instantiate the vendor MuJoCo, or score a
ball. Therefore G06 remains Partial and Gate 3/Gate 3B remains open.

The next matched fresh checkpoint paper is also preregistered at model 4000,
but it does not weaken this cross-engine gate. It reuses the **same K100 file
bytes**, semantic schedule, question order, exact-family bank and 2k stability
thresholds for all four `SZ` seeds. The offline queue cannot invoke a judge;
it can only combine two read-only Pod checkpoint audits after all four
`model_4000.pt` files are finite, embed iter 4000, bind the same adjacent
schema-3 hard-contract SHA and retain exact fresh lineage. A future runner must
consume the content-addressed activation artifact and still bind the current
MuJoCo evaluator. Source verification is `20 passed`; no Pod/runtime action has
occurred.

This is seed/checkpoint evidence, not an engine-parity result. Known seed1 4k
already returns only `.50` on this MuJoCo paper and scores `.99` in the analytic
Isaac companion, so the four-seed stability gate cannot pass and the existing
instrument disagreement remains. Seed4 at 4k can support “delayed learning”
only against the unchanged `.65` aggregate/`.50` each-side thresholds; it
cannot close family stability, physical Isaac truth, calibrated plant, or the
Agibot vendor MuJoCo Gate3/Gate3B final gate. The frozen paper and barrier are
documented in
`docs/operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md`; G06
remains `Partial`.

### Production-root-bound stand viewer is diagnostic, not Gate3 (2026-07-12)

`yikang-linux-port-0711@6b10998` supplied a useful plain-MuJoCo stand viewer, but the original
version retyped gains and assigned head gains `40/2` while describing them as production PD_STAND.
The tracked production header instead defines a 29-DOF policy view and explicitly leaves the two
neck slots passive. The selective port now parses the production pose/Kp/Kd arrays from
`a3_policy_parameters.hpp` at runtime, requires every 29-DOF joint exactly once, leaves head yaw/
pitch passive, and never modifies the vendor MJCF or integrator.

Dependency-light `--identity-only` binds vendor `a3_pingpong.xml=2ab1cd31...3feb97` and the
production parameter header `df73e3f6...c5c8d8`; four host-only parser/identity tests and pycompile
pass. Those are root-source identities only; the MJCF's 74 transitive mesh assets are not yet an
asset-closure hash and remain part of the formal runtime binding. With MuJoCo installed, `--check`
records actual timestep/integrator; finite status for
qpos/qvel/qacc/ctrl/actuator-force arrays; pelvis-z range/drift, maximum pelvis tilt and per-foot/
both-foot floor-contact fractions. Its default
thresholds are diagnostic tripwires, not Gate3 acceptance. The current Mac lacks the MuJoCo binding,
so no 10-second numerical result, snapshot or plant claim is recorded.

This tool starts no planner, policy, AimRT/backend, first tick, Gate3/Gate3B or hardware. It can
localize a base MJCF/PD/integrator problem before policy evaluation, but it cannot promote a policy
or explain the same-policy Isaac/MuJoCo gap alone. Instructions are in
`docs/operations/run_deploy_dryrun.md`; selective-port evidence is in
`docs/research/yikang_selective_integration_20260712.md`. G06 remains `Partial`.

The production runner now also has a fail-closed `--model-preflight-only` path. It requires
`--no-publish` or `--dry-run`, constructs `PpPolicy` before any backend object is created, and on
success emits parsed `publishable_model_contract=true`, `training_contract_exact=1`, the accepted
observation width, and training-contract/source-checkpoint SHA-256 before exiting. No-publish is a
transport/runtime diagnostic state, not permission to relax model metadata. This safely separates
“the formal 179 export is loadable under the production
metadata contract” from “the vendor backend has started.” The former still requires an isolated
binary run with the formal candidate; the latter, first actor tick, normal-envelope/recovery
contracts and Gate 3/Gate 3B behavior all remain open.

The first full-dependency probe found a common loader defect before any score was produced:
`PpOnnxPolicy` chained `GetInputTypeInfo(...).GetTensorTypeAndShapeInfo()` through a temporary
owner. The borrowed tensor-info handle was already dangling when its shape was read and real 175-
and 179-D models could throw `length_error`/`bad_alloc`. The source now retains both input
`TypeInfo` owners through all shape/type reads and adds an optional real-ONNX regression. This
finding invalidates the failed loader attempt, not the model; isolated Release rebuild plus the
formal-ONNX test and production preflight are required before marking the repair verified.

#### 2026-07-11 formal 179 production-loader gate

The repair and model-only preflight are now verified in a second isolated archive,
`/workspace/codexschema/gate3_face179_a82eba6`, from exact source
`a82eba6c7dbfad0c6750b2ca5684f3f2f7b6ea6e` (tree `7d0452ea...354a`, archive SHA
`7553dde0...c58`). The configure enabled both ROS messages and the AimRT backend; Release built
`run_tests`, `a3_deploy_onnx_ref_pingpong` and `a3_policy_runtime_probe`. Their binary SHAs were
`0aef44d2...3440c`, `1f0e13de...20cc` and `8cf9b300...36e0`. The formal SZ seed2 model-2000 ONNX
was copied read-only into the archive and retained SHA `350b51cc...34cc2`.

With `A3_PP_ONNX_PATH` bound to that model, the lifetime regression passed 1/1; the full suite was
205 pass, 9 optional-asset skips and 0 failures (214 total). Without no-publish,
`--model-preflight-only` exited 2 before model/backend initialization. With
`--planner --no-publish --model-preflight-only`, it exited 0 and printed
`backend_not_initialized=true`, `obs_dim=179`, training contract `3a3b3d95...b9972` and source
checkpoint `d920...5e22`. Both stdout/stderr searches found no `backend cfg`, backend initialized or
backend started line. Accepted/preflight and full-suite logs have SHAs `2962d653...b5f4` and
`eb15d603...f64e`.

The direct-CMake executable needed its build-tree TBB directory on `LD_LIBRARY_PATH`; the packaged
runner stages TBB. `PpPolicy` construction performs one intended zero-observation ONNX prewarm
inference, but no policy driver, backend tick, transport, simulator, Kit or command path started.
The live training/eval checkouts remained clean at `6d93bcb...`/`46a0ce2...`, and no isolated
process remained. This closes the pre-envelope formal-model production loading proof only. The
same ONNX is intentionally rejected by the stricter source below because it lacks the new envelope
metadata. Re-export/rebuild, first backend tick, canonical recovery tuple and full vendor MuJoCo
Gate 3/Gate 3B behavior remain open, so G06 stays Partial.

Red-team follow-up downgrades the 2026-07-11 loader proof to lifetime/backend-order evidence only:
at that source revision `diagnostic_no_publish` was also passed as the loader's legacy-contract
escape, and the optional real-model test explicitly enabled it. The inspected model happened to
declare exact lineage, but the run did not prove that no-publish and live-publish enforced the same
parsed contract. The stricter source below removes that coupling; a new envelope-bearing model and
rebuilt production binary must rerun the proof before publishable-model loading is closed again.

### Recovery tuple and named-ready mismatch are now explicit Gate3 blockers (2026-07-12)

The recovery A/B/C preregistration binds the read-only Gate3 policy blob at commit
`1d46ef2cbb915efc135251f9b32f4ec25d0342ab`, SHA `8c9814c...0eba4`, and rejects its current idle
179-D tuple as a formal train/deploy match: idle position is newly anchored to the live base while
velocity and face normal/rho remain from the previous strike. Training produces only an all-old
tuple before reveal or an atomically installed all-new tuple. The same runner also zeroes the
actor's last-action observation during static-stand handoff; that intervention is not T1's
carry-state contract and must be replaced or explicitly isolated before a no-reset score is valid.

A second static audit prevents the word `stand` from hiding a different initial state. The bound
Isaac reset pelvis is `(0,0,1.0684) m`; the vendor MJCF stand key is
`(-0.0416378,0.000359049,1.06839) m` with approximate roll/pitch/yaw
`(-0.030,0.249,0.042) deg`. The full 31-joint vectors differ by `0.171845 rad` L2, dominated by
head-yaw `-0.169416 rad`; excluding the head still leaves `0.028789 rad`. Stage-1 contact positions
are environment-origin absolute and the 179-D actor observes target minus current racket FK, so the
`4.16 cm` root-x offset does not automatically cancel. These numbers define a causal hypothesis,
not a proven root cause of the Isaac/vendor discrepancy. Formal A/B/C evaluation therefore blocks
until one content-addressed numeric contract binds the exact ready base, joint vector, racket FK,
target position/velocity/normal/rho and observation result in both engines.

All arms must consume the same immutable random-arrival rows, question order and deadlines, with no
physical reset, teleport, last-action/history/noise reset, or replacement of infeasible rows. q10
remains directional only; q50 is the decision paper. The final MuJoCo path has two distinct gates:

1. Gate3 is a hard runtime prerequisite. It binds the exact C++ runner, vendor MJCF, calibrated
   plant and model, and must pass first-tick parity plus continuous stability.
2. Gate3B may run only after Gate3 and must reuse the same runtime contract. It consumes the
   immutable random-arrival q50 schedule and is the final behavior arbiter for first-strike
   non-regression and return quality.

Isaac remains the development/cross-engine precheck, not the final behavior vote. A discrepancy is
blocked and root-caused, never averaged. The design-only validator is green (`50 passed`, prereg SHA
`ca7806df...d810616`), while launch remains intentionally blocked on separate Gate3 runtime/
stability and Gate3B scoring judges, their shared runtime contract, exact A policy-ownership/PPO
accounting, calibrated plant and safety bindings. See
`docs/operations/run_phase1_recovery_tuple_prereg.md`; G06 remains `Partial`.

The 2026-07-13 primary-source audit adds no runtime credit. ACE's near-time-optimal reset MPC is
evidence for an interruptible bridge/prepare architecture, not for free-standing humanoid balance;
HITTER samples the next task after swing completion; SMASH uses strike-centred recovery clips and
cyclic phase but does not publish a mid-followthrough random-reveal comparison; PACE's five-serve
episode is likewise not that treatment. Consequently G06 defines `T0` as cycle-bound install,
`T1` as event-driven structure with frozen rewards, and `T2` as a later learned-shaping increment.
Random arrival is first an immutable environment axis. Balance/ready shaping starts with paired
`2^2`; a third readiness potential and `2^3` require separate critic train/calibration splits and a
one-shot preregistered critic-gate q50 disjoint from sealed formal Gate3B q50, without hidden-future
leakage. Any self-hit, reset/teleport/history clear, deadline shift/censoring, per-transition-cell
collapse, fifth-and-later opportunity decay, one-shot regression or Isaac/vendor direction reversal
fails promotion. Full sources and DOE boundaries are in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`. No paper substitutes for the exact
Gate3/Gate3B runtime result, so G06 remains `Partial`.

上述 2026-07-13 reward 顺序只完成文档设计收紧。当前 machine prereg/validator 仍固定旧三项
reward 和 full `2^3`；因此它们不能作为新顺序的机制证据，也不能用于 launch。需要新的内容寻址
config、validator、测试和 operation 对账后才可解除这一阻塞，G06 仍为 `Partial`。

### 2026-07-12 Gate3 first-tick static plan gate (red-team corrected)

The historical `pp_gate3_rally.sh` launch command is no longer an approved formal launcher.
Content-bound audit `configs/gate3_legacy_process_audit_20260712.json` records 14 concrete risks:
eleven fuzzy `pkill -9` calls, conductor `pgrep -f` SIGSTOP/SIGCONT, no PID/PGID/starttime/token
ledger or trap, hard-coded unbound workspaces, inherited ROS graph, destructive fixed `/tmp` and
shared-memory cleanup, no formal-loader-first gate, publish-capable free-form runner args, a boot
loop that proceeds after timeout, partial direct-PID cleanup, and no concurrency lock. The old
scripts remain historical result provenance; do not invoke their cleanup to make a new run pass.

Red-team review rejected feature commit `1fc69d1` as mergeable runtime shape: it carried an armed
future supervisor before dependency closure or a safe startup handshake existed. The corrected
`scripts/run_gate3_first_tick_harness.py` is **plan-only**. It has no runtime option/arming phrase,
direct process launcher, signal path, process scan, runtime lock or trace consumer. Old
`--mode run`/arming arguments fail in argparse before any contract/Git work. Its only child commands
are read-only Git queries with `GIT_OPTIONAL_LOCKS=0`; therefore “starts no process” is too broad,
but it starts no sim/Kit/transport/planner/runner and sends no signal.

Schema-2 validation binds core absolute path+SHA pairs, but does not call that set an exact runtime
closure. Every path must equal its resolved spelling and every component is checked with `lstat`,
so symlink ancestors fail. Training/eval paths must be clean exact-commit Git top-levels. Proposed
argv arrays are fixed and passive/no-publish; `--flag=/abs`, unbound absolute paths, relative
payloads and extra flags fail. The optional plan output uses fsynced temporary bytes plus atomic
hard-link create and directory fsync; it never uses overwrite-capable `os.replace`. It is rejected
under the recorded source/train/eval worktrees or any Git dir/common dir, then all three clean Git
identities are revalidated before an external write. The ledger's runtime block is permanently
`not_run`, with no components, signals, lock, behavior result or ownership token. Source tests pass
`32` cases; no runtime was launched.

The plan explicitly keeps five runtime blockers null: the production C++ full
`--first-tick-json` needs verified same-sample runtime output (the source diagnostic below is
structurally inexact and has not run); exact process ownership still needs pidfd plus a cgroup/reviewed supervisor
startup handshake; PATH/LD/Python/AMENT directory manifests and AimRT/transitive `.so`/plugin
closure are absent; separate vendor config/MJCF hashes do not prove parser-resolved semantics; and
the atomic runtime ledger/exact lock transaction is undesigned. String containment in a config is
not accepted as MJCF binding. Filling or deleting a blocker invalidates the static contract. A
separate reviewed runtime implementation must close all five; this source never becomes runtime
eligible by changing a flag.

The ledger also freezes a ready-state hypothesis without turning it into a result. Fresh training
starts at pelvis `(0,0,1.0684)` plus default q; vendor `stand` is
`(-0.0416378,0.000359,1.06839)` with about `(-0.030,0.249,0.042) deg` rpy. Mapped joint L2 is
`0.171845 rad`, dominated by head-yaw `-0.169416 rad`; excluding the head still leaves
`0.028789 rad`. Because Stage-1 bank contact positions are env-origin absolute while 175/179 target
position is relative to current racket FK, the `-4.16 cm` root-x shift need not cancel. It may
contribute to the engine gap, but is not yet causal evidence. The preregistered same-K100
vendor/root-only/joints-only/full-match four-cell diagnostic remains inexact and unrun; the formal
vendor stand is unchanged.

Every plan records the four-stage engine-gap ladder as not run with no inference authority:
kinematic replay, open-loop action replay, external-observation closed loop, then native closed
loop. Isaac stays training/diagnostic-only. A future first tick would close only a runtime
prerequisite; only Agibot vendor MuJoCo Gate3/Gate3B behavior can promote a checkpoint. Full static
operation and remaining blockers are in `docs/operations/run_gate3_first_tick_harness.md`. G06
remains `Partial`.

#### 2026-07-12 content-bound per-clip demanded-normal source gate

Formal 179 export and loading now bind the raw-A normal distribution rather than accepting every
opponent-facing physical-B unit vector. The train NPZ must be exact schema 3, split `train`, ordered
`forehand,backhand`, `shared_plus_y` and `mount_plusY_A`, with its bytes and source family matching
the checkpoint contract. The contract and both exporters must carry the exact
`mount_normal_sign_per_clip=[+1,-1]`. Each clip is processed alone: normalize raw-A rows only after
the bank's `2e-4` unit check; require `sign[clip] * raw_A.x > 1e-6` for physical-B wire
representability and `raw_A_row · reference_A > 1e-6`, the same open-hemisphere margin enforced at
runtime; normalize the row-vector sum; and save
the minimum row-to-center dot. This
`per_clip_sign_preserving_spherical_mean_cap_v1` construction never averages forehand with
backhand or opposite face signs.

The ONNX carries envelope schema/frame/convention/pairing/algorithm, bank/runtime tolerances,
clip order, the exact sign table, two centers, references, dot thresholds, row counts, and
duplicated bank/family SHA
bindings. A dependency-free C++ SHA-256 implementation recomputes the canonical metadata payload;
the loader then rejects missing keys, a stale payload hash, bank/family mismatch, malformed or
non-unit vectors, flipped centers, invalid thresholds/counts and wrong clip order. `PpPolicy`
converts only the physical-B wire normal to A after selecting the clip, then checks both the raw-A
reference hemisphere and selected cap before its engage transaction commits clock, position,
velocity, side or normal. A positive-X physical-B unit vector whose converted raw A is outside the
selected support sets `face_command_out_of_train_envelope` and cannot start a swing. Older 179
ONNX files lack these
mandatory keys and therefore fail closed even under model-only/no-publish loading. Other registered
110/175/177/180 models retain their prior loader behavior.

Host verification currently covers the Python derivation suite, Python exporter/contract suite,
an Isaac-free standalone-import subprocess, locale-independent standard SHA-256/numeric parsing
and a compiled dependency-light C++ parse/accept/reject smoke. The Python results are `34 passed`
for contract/export and `11 passed` for the planner wire. The prospective real-bank fixture
binds bank `2da2bd12...a0700`, source family `b21c161a...28ad5`, `757/724` rows, raw-A/B sign
ranges and cap minima `0.974278/0.972078`; it is a read-only source-contract expectation, not a
behavior result. The fixture does not contain the ignored NPZ: restore and SHA-check it using
[setup_local_sync.md, Phase-1 fresh and causal bundle](../operations/setup_local_sync.md#phase-1-fresh-and-causal-bundle-2026-07-11).
A full vendor-dependency Release build and freshly re-exported real 179 model now pass the strict
model gate described below. A spherical cap is still only an on-distribution envelope, not a
collision or behavior proof:
self-hit instrumentation, the recovery tuple and no-reset vendor MuJoCo Gate 3/Gate 3B remain
open. G06 stays `Partial`.

#### 2026-07-12 publishable-model and atomic-export hardening

`PpPolicyConfig::diagnostic_no_publish` no longer reaches the ONNX contract escape. Plain
`--no-publish`, `--dry-run` and `--model-preflight-only` now load with the same publishable
schema-2 packaging, exact/complete schema-3 execution contract, normalization, effort envelope,
registered layout and (for 179) full content-bound normal envelope required by a live publisher.
The only legacy escape is `--allow-legacy-model-diagnostic`; CLI validation requires no-publish
and rejects its combination with model-preflight before loading a model or constructing a backend.
The optional real-model GTest now uses the strict constructor and requires parsed exact/schema-3/
publishable flags plus the 179 envelope when applicable.

The production preflight certificate is also derived rather than asserted by mode: it checks the
parsed booleans, prints `publishable_model_contract=true training_contract_exact=1`, and prints the
parsed envelope sign table. `verify_face179_preflight_failclosed.py` runs one valid model then
creates temporary metadata-stripped, missing-envelope and `training_contract_exact=0` variants;
each must exit nonzero with no backend marker, while legacy-diagnostic plus preflight must exit 2.
The helper/unit source gate passes locally; the real runner integration result is recorded below.

The standalone exporter now completes every checkpoint/donor/motion/harvest/bank/envelope check
before creating a graph, writes only an owned same-directory temp, checks the ONNX and metadata
round trip, fsyncs, and atomically replaces the destination. Failure tests prove an existing
`policy.onnx` remains byte-identical and no temp remains. Focused host results are `41 passed` plus
one optional real runner/model integration skip; planner wire remains `11 passed`.

The missing integration was then run on Pod1 from an isolated tree whose 19 changed files match
exact source `2fa35340b63f98c04c67c8b29c80939610fd86e9` (tree
`299a2907229c1aaa4b581007c0ebe46cd914a011`). ROS/Jazzy + AimRT 1.6 Release rebuilt all three
targets. Fresh SZ seed3 model-2000 was re-exported from checkpoint `11f3a288...e77a5`, motions
`f2cb2d9f...1687`/`17225533...7534` and train bank `2da2bd12...a0700`; the new ONNX is
`0c428ddf...b7b155`, with envelope payload `df3fd8ae...08502e`. The native suite reports 219
tests, 210 pass, 9 optional-asset skips and 0 failures. Strict production preflight exits 0 and
prints parsed publishable/exact/179/sign/bank/family values with no backend marker. Graph-identical
metadata-stripped, missing-envelope and exact=0 variants each exit 3 before backend; legacy plus
preflight exits 2. An exam-bank-as-train failure preserves the existing ONNX SHA and leaves zero
temp files. All 824 compile commands are free of fast/finite-only math flags.

The content-addressed ledger is
`configs/gate3_face179_strict_preflight_evidence_20260712.json`. This closes the corrected export,
full Release build and strict model-only preflight gates. It did not start the vendor simulator,
transport or backend, so first tick, planner-policy closed loop, self-hit, continuous stability and
Gate3/Gate3B behavior remain open; G06 remains `Partial`.

### Planner proposal held out by coordinate and clock pairing counterexamples (2026-07-12)

The selective planner proposal at `69418a9` is **not merged** despite its green host suite. Its
explicit schema-2 side was selected from `intercept_y_world - base_y_world`, whereas the C++ policy
forms `tgt_b = R_yaw^-1 * (target_world - base_world)`. This changes decisions inside the allowed
heading range: with base yaw 10 degrees and target delta `(0.67, 0.02) m`, the proposed selector
chooses BH while policy-frame Y is `-0.09665 m` and requires FH. A wrong side also chooses the wrong
clip before the formal face179 `[+1,-1]` physical-B-to-raw-A conversion. The correction must use
the same normalized base-yaw frame and fail closed if orientation is missing or invalid.

The proposal's `2.6 s` prediction horizon exposed a second policy-clock mismatch. Current formal
179 clips have approximately 1.30 s FH and 0.88 s BH maximum in-training windup. Unlike 110, 179
does not wait while a valid command is earlier than that clip window; it clamps and engages, so an
approximately 1.89 s Gate3 arrival would put the FH/BH strike clocks about 0.59/1.01 s early. A
formal fix needs selected-clip, metadata-bound waiting with continued freshness/revocation checks,
or a preregistered serve schedule already inside the relevant windup window. Merely widening the
planner horizon is not a demo fix.

Finally, future serve release must bind both exact-owned runner fresh MOTION and exact-owned
planner fresh readiness, including executable/argv/config and PID/PGID/start-ticks/log inode. A
runner marker alone cannot prove the first planner command exists. The reviewed source remains
outside main until these pairings and C++ tests close. No vendor runtime ran; G06 remains `Partial`.

#### Planner correction still held by revoke and yaw-ownership counterexamples (2026-07-12)

The follow-up candidate `71b0b23` fixes the original pure helper geometry and adds selected-clip
windup waiting plus a dual runner/planner readiness preregistration, but it is still **not merged**.
Independent state-machine review found that a base sample ageing past `0.2 s`, or a newly malformed
base sample, clears only the planner's in-process corrected-base tuple. Neither path immediately
publishes a schema-2 invalid flat command. A previously valid racket tuple can consequently remain
inside the C++ subscriber's `0.5 s` command timeout and become eligible again if base freshness
returns before another admitted ball solve.

Publishing an invalid row alone is insufficient for formal 179: the current runner applies the
legacy `planner_invalid_grace_s=0.25` to every actor. A tuple explicitly revoked while waiting can
therefore still be classified fresh during the grace interval. Formal schema 2 must revoke
immediately; the historical grace may remain only for explicitly registered legacy contracts.

The proposed Python selector also consumes the corrected mocap/pelvis quaternion, while
`LocMode::kExternalBase` deliberately keeps the runner's boot-yaw-aligned IMU orientation and
ignores the quaternion carried by `/a3/base_pose_flat` for the policy target frame. Passing the same
quaternion to two pure helpers does not prove those runtime authorities coincide. Before merge, the
runner must fail closed whenever the planner's proposed side disagrees with the runner's actual
policy-frame geometry (with an explicit boundary rule), and dynamic tests must cover stale revoke,
malformed-base revoke, revoke during windup waiting, recovery before timeout, and mismatched yaw
authorities. No simulator, backend, Pod process or robot ran; G06 remains `Partial`.

#### Third planner correction held by same-tick and cross-topic causality failures (2026-07-12)

Candidate `6aae7ac` added dual flat revocation, immediate formal invalid, a base-receive epoch,
runner-frame side consistency, sample-driven READY and stronger source/environment bindings. Its
host suite is genuinely green at `198 passed, 2 skipped`, but it is still **not merged**.

`ComputeCommand` invokes `PlannerEngageStep_` before it samples the current IMU/base mailbox. A
previous tick with base age `0.199 s` can therefore leave `base_fresh=true`; on the next tick the
sample may be stale or explicitly invalid, yet engage can latch level 1 before current localization
is read. The same ordering makes the side check use `last_base_quat_w_` while the observation later
uses the current IMU. A small yaw change can move `tgt_b.y` from the ambiguous band to the exclusive
side region after the wrong clip has already latched.

The receive-time epoch is also insufficient across two DDS topics. Per-topic ordering permits
base invalid then base valid, followed by a delayed pre-revoke racket valid and only later the
racket invalid. Because the old valid is received after the base invalid, a local timestamp
comparison misclassifies it as post-epoch. Formal closure requires a single source epoch/sequence
carried in both base and racket payloads (or one atomic combined topic), exact equality at engage,
and a single same-tick localization snapshot shared by engage, side/face gates, windup wait and
the policy observation. The immutable prereg must also bind and parse the mailbox/wire/frame helper
bytes it currently omits. No runtime, simulator, Pod mutation or robot ran; G06 remains `Partial`.

#### Shared-epoch candidate still held by source-age and active-revoke counterexamples (2026-07-12)

The next unmerged worktree adds schema-3 racket rows, schema-2 base rows, a shared source epoch and
sequence, source-header-to-monotonic mapping, a common mailbox transaction mutex and a single
localization snapshot reused by engage and observation. Its host suite reaches `155 passed, 2
skipped`; an isolated ROS/Jazzy Release build reaches `220 passed, 5 optional skips`. These are
useful source results, not merge authority.

A second fresh review still reproduces five P1 groups. Once a formal stream is established, a
recognized legacy-schema packet can downgrade state without poisoning pre-barrier recovery.
Formal invalid state still has one receive-wall-time `>` dependency, so valid and invalid events
sharing a clock tick need not revoke deterministically. The Python base lease is keyed to receive
time rather than mapped source time, and expiry occurs after current-sample admission; an already
old base can therefore calculate a command or print READY, then be rejected by the runner.

Active formal 179 swings also fail to latch both the engage epoch and a base-revocation generation.
An epoch change followed by fast recovery, or a local malformed base followed by same-epoch valid,
can be hidden between two policy ticks while an E1 frozen target continues on newer localization.
Base epoch/revocation changes are localization safety revokes and must abort/rearm even if ordinary
racket flutter remains frozen in flight. The latter behavior must be stated once: current comments
conflict over whether malformed racket input aborts an active swing.

Finally, the planner advertises world/table frame codes but does not compare the incoming ball/base
ROS `frame_id` to a configured formal authority. A fresh finite sample in a different frame can be
relabeled as formal world. Exact schema-3 must bind and enforce both frame ids, or remain inexact
behind a runtime publisher/header gate. The serve preregistration still omits wire/mailbox/frame
helpers, merged-YAML parser semantics, same-host monotonic authority, unique publisher/domain and
hot-restart session closure. The candidate remains **NO-MERGE** and G06 remains `Partial`.

#### 2026-07-12 joined-source first-tick diagnostic

The production runner now implements no-publish-only `--first-tick-json` instrumentation for a
strict 179-D model. PASSIVE waits; SHADOW records the first observed planner-engaged actor candidate;
idle/wait/invalid/recovery rows do not consume it. The output is canonical mode 0600, fsynced atomic
hard-link no-replace and contains joined qpos38/qvel37/base7/racket7, target candidate, obs179,
action31, layouts, clocks and content SHAs. It does not emit a source-commit claim.

`RobotState` lacks root linear velocity, so a subscription-only sim sidecar reads the vendor pelvis
pose/twist and right-racket pose topics without publisher/reset/command or estimation. Kernel
`flock` plus whole-record `pwrite/pread`, freshness, finite/unit checks, strictly advancing stamps,
positive even generations, 20 ms native-header skew and a 30 ms RobotState/sidecar receipt join are
enforced. The observation base is recorded separately from joined vendor-world base and the native
racket point must agree with formal FK within 5 mm.

This is deliberately not a native same-tick snapshot. The tracked vendor publishers stamp messages
asynchronously at publish time and expose no common MuJoCo sample sequence. The current planner also
has known same-tick snapshot/shared payload epoch blockers. Both outer document and payload fix
`evaluation_contract_exact=false`; planner/native/source-binary/source-semantics/runtime-closure
exactness are fixed false with non-empty reasons. Gate3/Gate3B and promotion consumers must reject
this v1 schema.

The model path has no load/hash TOCTOU: stable canonical ONNX bytes are hashed and passed directly
to ONNX Runtime. The checked-in ledger hashes only a reviewed source subset and fixes
`source_semantics_closure_exact=false`; it is not parser-backed closure. Vendor config→MJCF parser
resolution, publisher binary/config/transitive membership, planner/wire/frame/backend closure,
owned supervisor/timeout, runtime ledger and actual backend first tick remain OPEN/null.

Host source checks are `6 passed`; the combined static-plan+diagnostic tests are `38 passed`. No
simulator, transport, backend, Kit, Pod/GPU or hardware ran. Full ROS/Jazzy/AimRT Release build and
native GTest are also unrun. G06 remains `Partial`.

### 2026-07-12 model-4000 matched-q50 execution source gate

The fresh `SZ` model-4000 matched MuJoCo paper now has a source-reviewed consumer for the existing
all-four activation barrier. It strictly pins queue, preregistration, queue validator, fresh exact
result helper, itself and the four evaluation-tool files. It accepts no queue-only or one-Pod
authorization: every command revalidates the activation content hash, exact barrier id, both Pod
audit hashes, four checkpoint audit records and the immutable K100 file/semantic/order hashes.
At runtime each Pod additionally rehashes its two local model-4000 checkpoints and reruns the
finite/embedded iteration/contract/lineage audit.

The execution contract does not call the schedule materializer and rejects a different path even
when its bytes happen to match the paper audited into the activation. `prepare` creates an
activation-bound no-clobber contract. `run` executes two serial pinned `judge.sh` children per Pod,
sets the common Kit boot lock, verifies each new-session PID equals its PGID, waits for completion
and preserves state/log on every failure without exposing any signal API. Seed1 is rerun, not
reused. A result must reproduce `evaluation_contract_exact=true`, K100/50-per-side, schedule/order,
vendor-development MJCF, execution/ready-state and checkpoint/hard-contract bindings; report,
summary and raw attempt ledger are rehashed before the Pod result is written.

Aggregation retains the unchanged model-2000 stability thresholds but fixes
`family_stable_claim_allowed=false` because seed1 `.50` was known before preregistration. Seed4's
only permitted conclusion is delayed learning versus persistent weakness through 4k. This paper
still does not answer the Isaac/MuJoCo physical-instrument gap, calibrated plant, recovery,
continuous stability or Agibot vendor Gate3/Gate3B behavior. Focused queue+consumer tests pass
`40`; at source merge no Pod audit, activation, MuJoCo judge or simulator had run.

The all-four barrier was then materialized outside train/eval on 2026-07-13 local time. Pod1's
seed1/3 audit is `3fc325e1...247b8`; Pod2's seed2/4 audit is `4f25786b...565f7`. Their exact union
created activation file SHA `9dea76c2...ce704` with content SHA `eaa92ca2...aa4fb`, covering all
four seeds and retaining `judges_started=0`. Source, K100, both audits and activation are present
at the same absolute paths on both Pods. Both runner `contract-check` calls passed; immediate
pre-run snapshots found no child judge, MuJoCo evaluator, play/Kit process or shared-lock holder.

Both Pod runtime contracts were subsequently created by no-clobber `prepare`. Pod1's file/content
SHAs are `2b76a5a...8201e` / `36e878f0...5ba73`; Pod2's are
`dbecc102...d1c9b` / `91a0070a...30794`. Direct binding validation rehashed both local checkpoints
per Pod and confirmed iteration 4000, finite tensors, exact lineage, the shared hard-contract SHA,
clean exact train/eval checkouts and an empty post-prepare process/lock snapshot. Both contracts
remain `prepared_not_started`, `jobs_started=0`, `auto_start=false`; no `run`, judge, aggregate,
score, signal or hardware action occurred. The remaining launch blocker is a reviewed persistent
parent supervisor that retains serial two-seed ownership and final-result materialization after an
SSH disconnect. This is execution-paper preparation rather than behavior evidence, so G06 remains
`Partial`.

### 2026-07-14 signed-face v8 再次在合同前阻断，不授权判卷

Pod1 epoch-1 v6 的 A/B/C 已到终档，但 D 在产生 runtime verified/checkpoint 前 Kit boot timeout；因此
不存在完整四格 L1 activation。后续 [v6r1](../DEFINITIONS.md) 首次真实 `validate` 在任何 claim/训练前
发现 expected-absent 合同错误：checkpoint audit 明确 D `run_dirs=[]`，但 validator 却要求旧 would-be
training path 是 directory。团队没有伪造目录；v6r1 从未启动。新 [v6r2](../DEFINITIONS.md) 只发布
source-only 静态修正，要求旧 path absent 且任何 entry kind fail closed；它没有 runtime、命令重建、
launch、signal 或 mixed finalizer，明确 NOT LAUNCHED，不能补出 L1 activation。

后续 foreign v8 不采用 v6 artifact，而以新 source/manifest/launcher 按 A/B/C/D terminal barrier 串行
运行。A/B/C 前序已终档；D 作为第四格又在 900 秒内未产生 hard contract/runtime verified/checkpoint，
exact-PGID wrapper cleanup 后 rc=124。它是继 v6 D 后第二次独立 pre-contract Kit boot timeout；自动
retry 已停止，必须先做 boot root-cause。v8 没有四格 activation，且 L2/judge/第二 seed 均为 false，
所以不能进入 Isaac/MuJoCo 同卷，更不能成为 Gate3/Gate3B 或部署证据。G06 保持 `Partial`。

只读 postmortem 现已证明两个 D 的最后 Kit 语义操作都是加载 byte-identical table USD，且都未进入
PhysX context；相邻 C 则分别在 `2.339/3.031 s` 越过同一边界。它只把 failure boundary 从泛称的
“boot timeout”收窄，未证明第四进程、Carbonite cleanup、driver 或 filesystem 中的任一项是根因。
结果 ledger 明确把 fact/inference/unknown 分开；`dmesg` 不可读，共享内存残留只记 correlation。
计划中的 D-first/ordinal-4 与 fresh private IPC 对照仍是 [design-only prereg](../../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)，
不能运行训练、不能生成 L1 activation，也不能授权 Isaac/MuJoCo 判卷。G06 保持 `Partial`。

### 2026-07-13 MuJoCo trainer preflight 红队：授权安全，源码门暂缓

独立复核确认 `codex/mujoco-training-preflight@6e5fce3` 的 focused `63 passed`、顶层
`468 passed, 9 skipped`、`valid_but_blocked`，以及 `--require-ready`/certificate 输出均 rc=2。
七个授权布尔值全为 false，也没有 consumer/launcher，所以当前分支不能启动或批准训练。
但它仍因四个 red-team P1 正确性缺口被标为 `NO-MERGE`：trace 缺 action clamp/runtime adapter
状态且 action tape 太小；静态 source independence 可经 alias/exec 绕过；JSON 未拒 duplicate key/NaN；
MJCF `compiler strippath` 语义建错。vendor scene 仍无可碰撞球/台/网，因此 v0 只允许无球
balance/strike-state diagnostic，不能记 physical return。

修复这四个源码门后，第一个 single-env core 还要通过预注册的 N=1/8/32/64 吞吐继续门，并证明
两臂×两 seed 在 48 小时内完成且留 30% 余量。这个门不替代独立 exact vendor Gate3/Gate3B，
也不阻塞几天内 `Gate3-D0`。G06 仍为 `Partial`。

### 2026-07-12 文档路由

引擎/迁移现状只在 [`docs/NOW.md`](../NOW.md) 汇总一次，详细实验记录放在
[`docs/experiments/`](../experiments/README.md)。经过筛选的
[`docs/TIMELINE.md`](../TIMELINE.md) 只记录已经进入 `main` 的重要能力和根因修复，并明确说明
Isaac–MuJoCo gap 只有部分可复现差异和候选来源已定位，整体因果归因与修复均未闭合。
Legacy Gate3 diagnostic 与 current exact-179 Gate3
属于两份独立实验记录，不能互相填充结果单元。本次纯文档迁移没有运行 evaluator、模拟器、backend、
Pod 或真机，也不改变 G06 的 `Partial` 状态。

### 2026-07-15 legacy V9 `7/7` proxy correction

The branch-local V9/v12fix `7/7` value is not a physical-return result. Its success predicate combines
planner engage, ready, at least one completed swing and recovery, while the result explicitly records
`physical_contact_measured=false` and `landing_measured=false`; the fake-ball publisher supplies planner
poses rather than a MuJoCo contact/flight event chain. Repetition covered a fixed forehand region, so it
also cannot establish unseen-ball, backhand or multi-action generalization. It may be retained only as an
exact planner-policy cycle-stability proxy. Physical selection still requires a vendor-MuJoCo receiver
with all-serves denominators and a disjoint held-out paper. The source audit and candidate mechanisms are
recorded in [the Jiayi/Yikang cross-learning experiment](../experiments/2026-07/EXP-V9-YIKANG-CROSS-LEARNING-20260715.md);
no runtime, score or Gate status changes here, and G06 remains `Partial`.

#### 2026-07-13 exact formal tuple and portable Release integrated

The earlier planner red-team candidates above are superseded by exact source `c0a8e46`. Formal
racket schema 3 carries shared epoch, command sequence and exact `base_sequence_ref`; a bounded
base history proves that causal reference, while side/target/yaw, base-low, observation, active
abort and recovery use one latest tick-start base. Fixed-latency barriers no longer chase receive
time, and Python/C++ share the same workspace and source-time continuity limits. Every formal actor
path checks latest finite/fresh/plausible/base-low state and preserves the latched engage epoch plus
base revocation generation across recovery.

The 23 effective source/config/test paths were transplanted byte-for-byte onto latest main with
manifest SHA-256 `8af1a2fc37dc912f41cb5609a687b481fbadbddc531ff4f430d6294796665fd3`;
no source conflict was resolved semantically. Local planner/source is `180 passed, 2 optional
skipped`; serve preregistration is `39 passed`; full root tests are `521 passed, 9 skipped`.
Serve design-check passes only with all 49 runtime bindings blocked, and launch-check fails with 49
`MISSING` lines.

The same exact source passed isolated Pod2 Ubuntu 24.04/GCC 13 portable Release: focused
`PpPlannerInput.*:PpFirstTickJson.*` `40/40`, complete native `233 passed + 5 optional skips + 0
failed`, both test and production runner binaries linked, and all 80 compile commands retained
strict finite math. This closes the exact source/binary merge blocker only. ROS/Jazzy/AimRT were
disabled, the production runner was not executed, and formal ONNX runtime, backend first tick,
vendor MuJoCo behavior, continuous stability and hardware remain unrun.

The v2 joined-source ledger remains deliberately inexact: it has no common native MuJoCo sample
sequence, executed binary/runtime closure or owned supervisor. The separate serve v4 design keeps
49 runtime bindings null and cannot arm a publisher. Detailed hashes and reproduction are in
[the exact build experiment](../experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md) and
[serve operation](../operations/run_gate3_serve_sync_prereg.md). G06 remains `Partial`.

### 2026-07-13 persistent model-4000 q50 startup source gate

A detached two-phase [persistent supervisor](../DEFINITIONS.md#persistent-supervisor) now closes
the activation consumer's top-level SSH-lifetime gap without changing any evaluated bytes. Its
caller-SHA-pinned config binds the existing consumer/config/activation and a distinct prepared
runtime contract plus Python realpath/binary SHA for each Pod. A fixed no-clobber state directory
contains child hello, immutable launch ledger, commit token, commit acknowledgment and combined log.
The token is withheld until the parent verifies `PID=PGID`, Linux boot id/procfs start ticks,
executable SHA, exact argv/fixed-environment digest and every artifact SHA; parent loss or stall
before that token makes the child self-exit rather than start an unowned judge. Deadline, process
identity and token/ledger binding are checked before token publication. Atomic token final-link
visibility is irreversible even if the following directory fsync fails; after it, slow rehash,
acknowledgment publication and exec are pending committed work rather than deadline failure. The
child still rechecks all bytes and
identity/token/ledger/result before acknowledgment and before `execve`. Separate acknowledgment and
exec-observation windows return `token_published_pending_ack` or `committed_pending_exec` with return
code zero instead of creating retry authority when progress is not yet visible.

Read-only `inspect` rehashes the complete closure. A live result requires the preserved PID, PGID,
start ticks, executable, command line and environment digest to match; terminal acceptance is
delegated to the unchanged runner's complete schedule/arms/lineage/count/report validator. A
pre-existing result prevents launch. No retry, remote login, process-control, trainer/worker,
simulator, deployment or robot surface exists. The detailed contract and commands are in
[the model-4000 q50 operation](../operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md#persistent-top-level-launch-source-gate).

Supervisor tests pass `24`; combined queue/consumer/supervisor tests pass `64`. Tokenless deadline
expiry cannot execute; post-token delayed rehash, a 1.15-second acknowledgment atomic-publication
stall and post-ack delayed exec all reject restart and later converge without a
fatal-before-later-runner sequence. Terminal validation also freezes bytes/SHA and rejects an A-to-B
replacement. Post-link token-directory-fsync plus evidence-stat failure, token temporary-cleanup
failure and parent-observation-write failure are separately covered: all return committed pending,
reject restart and later inspect as exact running without a fatal. This is host source evidence only: Linux procfs has not yet been smoke-tested, the wrapper is
not deployed, and no MuJoCo judge or score ran. It therefore closes neither the matched
[q50/K100](../DEFINITIONS.md#q50-and-k100) result nor vendor Gate3/Gate3B. G06 remains `Partial`.

### 2026-07-13 Phase-1 trainer 裁剪不改变 MuJoCo 证据门

16 条 fresh 广度臂中的前 8 条已按负责人后续运营决定精确停止并保留最新 finite、schema-3、
fresh-lineage checkpoint；其余 8 条在后续 signed-face 取证后也已停止。formal `SZ` seed1/2/4 trainer
虽在首波停止，但 model-4000
四 seed checkpoint 在此之前已经通过 all-four readiness 并进入两 Pod 的
`prepared_not_started` K100 runtime contract，因此后续 matched q50 输入没有变化。

这次动作没有运行 MuJoCo judge、没有新 q50/physical/Gate3 分，也没有修改 q10 screen-only 或
q50 `whole_arm_stop_allowed=false` 合同。停止运行不能替代 signed-face 诚实门、同一 checkpoint
跨引擎归因、厂商 runtime `Gate3/Gate3B` 或标定 plant。运行与证据边界详见
[拍面×plant 广度实验](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)；G06 保持 `Partial`。

model-4000 取证后，剩余 8 臂的最近 24 个 K20 格又全部出现正手 signed composite=0，
法向误差 `164.4°–175.2°`，而 parsed return 可达 1.0。负责人因此批准第二波精确停臂；
现在 16 条 fresh 广度 trainer 均已保留证据并停止，无残留 judge/Kit 且两 Pod GPU 为空。
这只使得“修 signed-face 后再训”成为明确顺序，不会关闭跨引擎或 Gate3/Gate3B；G06 保持
`Partial`。

### 2026-07-13 model-4000 matched q50 结果与拍面仪器失真

内容绑定的 Linux supervisor 冒烟、两 Pod 判卷和单次 aggregate 已完成。Pod1/Pod2 result
file SHA 为 `02d0e58d...645d` / `d31323a6...4e6f`；aggregate file/content SHA 为
`1ba88e39...d195` / `226e6050...648d`。四 seed parsed rate `.50/.88/.98/.00` 使预注册稳定门
的 median/worst/spread/worst-side 四项全失败，seed4 为 21 次物理 root fall 且 `0/100`，
因此不支持 4k 晚熟。

这份卷更重要的 G06 证据是同 checkpoint 内部的仪器矛盾：seed2/3 解析正手 return
`38/50` / `48/50`，但 raw-A 有符号法向差 `172.33°/174.35°`，位置+速度+法向复合
命中均为 `0/50`。因此旧 `orient_normal` 解析回台分已被实证为正手符号盲区，
不得作为跨引擎或部署晋级证据。需先用 `n/-n` 负控修表、重跑同卷，再以同一
checkpoint 进行 kinematic replay → open-loop action → external-observation closed-loop → native
closed-loop 归因。该 Python BankExam 仍不是 Agibot vendor Gate3/Gate3B runtime，所以 G06 保持
`Partial`。详见 [稳定性实验](../experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md) 和
[拍面符号取证](../experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。

### 2026-07-13 signed-face analytic ruler source gate

feature source 已把 analytic scorer 升为 schema 2：formal 路径必须从 ONNX metadata 读取完整
`mount_normal_sign_per_clip`，把 achieved/target raw-A 通过每 clip sign 绑定到 opponent-facing
physical-B，并在 `orient_normal` 前要求 strict signed hemisphere 与 achieved/target B.x `>1e-6`。
结果与 strike CSV 同时写 signed-face exactness、dot/error 和 physical-B-facing 字段；scorer source、
venue config、参数、sign table 与门限进入 execution-contract SHA。缺 metadata、非法 sign 或长度不符
均 fail closed。

只有显式 `--allow-inexact-contract` 可保留旧 unsigned-plane 诊断；它必须写
`signed_face_exact=false` 与 `evaluation_contract_exact=false`，不能晋级。phase/spatial 动作筛卷调用点
也已迁移到 raw-A achieved/target + clip-id；历史 v5 和 q50 仍绑定旧 scorer，只能作为 paired legacy
列，不能被新源码追认。`n/-n` 负控锁定了根因：冲量与落点逐值相同，但新门只让正确 physical face
contact/return。

本地 focused source/unit 回归为 `38 passed, 1 skipped`，顶层 broad 为 `546 passed, 9 skipped`；
没有运行 MuJoCo/Isaac/Pod/judge/vendor
Gate3/Gate3B。下一步是在同一 immutable K100/同 checkpoint 下生成新 execution contract 和不覆盖的
paired result，再按 kinematic replay → open-loop action → external-observation closed-loop → native
closed-loop 分层归因。analytic scorer 即便通过也仍是 diagnostic，不替代 physical return 或 vendor
runtime；G06 继续 `Partial`。

训练侧随后为同一问题物化了单-seed A/B/C/D 漏斗，但它没有改变 G06 的裁判边界。A/B 从旧
`model_13800.pt` 进入当前源码时，因为 hard-contract 新增 event-timing/target-cadence 字段，只能是
`training_contract_lineage_exact=0` 的显式表示迁移；C/D 才允许 fresh lineage `1`。L1 是
25-update launch-integrity smoke，其四格 completion 文件不能授权 L2 或 judge。immutable signed-face
directional checkpoint paper path/SHA 尚未冻结，manifest 明确 `l2.launch_authorized=false` 且
`automatic_judge_launch=false`。源码/攻击回归 `23 passed` 不构成 Isaac/MuJoCo 行为或 Gate3 结果。
Pod 首次 v1 preflight 的 checkpoint 假拒绝已根因到“顶层扫描/顶层 provenance”错误；v2 递归审计
嵌套 tensor，并只接受 runner 写在 `checkpoint["infos"]` 的合同字段。v2 首格又在学习前暴露 detached
worktree 的 source-first 环境没有传入 child；v3 又因在 `SimulationApp` 前真正 import IsaacLab 而
假拒绝。v4 用确定性环境 SHA 与不执行包的 exact-worktree `find_spec` 在 claim 前关闭发射缝，随后
又在 scene 构建时发现 ignored A3 资产不会随 worktree 出现。v5 绑定并验证 exact restore/target 资产树。
上述修复都没有缩小 Isaac–MuJoCo 行为差，也没有启动 judge；
操作见 [signed-face 漏斗运行手册](../operations/run_phase1_signed_face_rescue_funnel.md)，G06 保持
`Partial`。

v5 又在第一次 learning iteration 前揭示旧 train bank 的 physics-contract SHA 与 `882fea4` 不同。
main 的严格 rebind consumer 不放宽 schema-3 loader：它只在 source/AST、全部问题数组 raw bytes、四个
metadata leaf、exact motion contract，以及全部 1481 道题 old/new contact/flight bitwise replay 同时
通过时发布新路径 train bank。Pod1 已发布 bank/report `3a9d8851...5b71` / `9fffed03...bb37`，两侧
landing/net 全过；这只是 E2 runtime data gate。train rebind 后 source-family SHA 已变为
`9603a178...a9db`；旧 immutable exam
不能与它组成 exact 同 family 证据。对应 exam bank 还未同法重绑定/重生成，L2 directional paper 与
judge 仍阻断，所以这项能力不构成 G06 或 Gate3 结果。

### 2026-07-14 signed-face exam bank E2 数据门完成

exam 对应的严格数据门已完成 E2 runtime replay 与 no-clobber 发布。generalized consumer 仍 byte-exact
接受历史 train-v2 manifest，同时新增一个封闭 exam-v1 profile：旧 exam path、`63,968` bytes、SHA
`d7db2568...f5096`、split `exam`、正/反手 `183/188`、旧 family `b21c161a...8ad5` 和独立 no-clobber
输出都不可替换；目标 physics/family 与已发布 train-v2 同为 `09dfe899...afb95` / `9603a178...a9db`。
mutation/source-receipt/profile 回归为 `18 passed`。Pod1 目标 runtime 的 24 个非 metadata 数组未变，
正/反手 `183/188` 道题 old/new output bytes 全相同并通过 landing/net；发布 bank/report SHA 为
`60e1a7ad...d1ca` / `dd4332ed...ad0`。

新 bank SHA 改变 question ID，旧 schedule 不得复用；必须从新 bank 重新冻结独立 schedule/paper
activation 后才能启动 judge。详见
[实验卷宗](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)与
[运行手册](../operations/run_phase1_signed_face_exam_bank_rebind.md)。当前 L2、signed-face paper、G06 与
Gate3 状态均不变，G06 保持 `Partial`。

### 2026-07-14 signed-face K100 materializer/activation source gate

E2 rebound bank 的下一层 paper source gate 已预注册并通过 host 静态/攻击回归。consumer 不接受旧
schedule 输入；它只从 exact `63,643`-byte bank SHA `60e1a7ad...d1ca` 重新生成包含新 bank SHA 的原子
question ID，并复用现有 schema-v3 deterministic schedule 算法冻结 seed `0`、hold `[0,100]`、每侧
无放回 50 的 K100。所有 100 个 scheduled attempt 保持在分母；missing/invalid/reset 不能删除。

paper 同时冻结 raw-A/physical-B signed 身份：clip order `forehand,backhand`、sign `[+1,-1]`，每个 target
raw-A normal 必须 finite/unit，映射后的 opponent-facing physical-B 必须严格 `x>1e-6`；unsigned 或先
`orient_normal` 再判身份的路径拒绝。旧 paper file/semantic/question-order receipt 均列入禁用表。
output root 必须不存在，schedule 与 activation 都 no-replace，activation 最后写；partial root 不能续写。

manifest/consumer SHA 为 `e401305d...e556` / `4e094bbe...ac6e`；mutation、旧 schedule、unsigned、
重复题、单侧不足和 partial no-reuse 回归共 `14 passed`，latest-main root `747 passed, 10 skipped`，
`static-validate` rc0。随后 Pod1 用 clean detached `748b6d5` source 成功执行单次 exact-bank consume：
schedule 为 100 个唯一题、正反手各 50，file/semantic/question-order SHA 为 `f2777dcd...1ca` /
`3ca4bdba...3365` / `09f778f2...bd0`；activation file/content SHA 为 `e0125b0e...bb4` /
`533beb03...3d8`，并在 schedule 落盘复核后最后写入。runtime receipt 见
[`phase1_signed_face_exam_k100_runtime_receipt_20260714.json`](../../configs/phase1_signed_face_exam_k100_runtime_receipt_20260714.json)。
activation 固定 trainer/judge/L2/第二 seed/晋级/部署/真机全 false；后续还需独立 reviewed execution
contract。详见
[实验](../experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)与
[操作](../operations/run_phase1_signed_face_exam_k100.md)。paper 已物化但没有 checkpoint/judge 行为，G06
继续 `Partial`。

### 2026-07-13 pelvis point/axis frame correction

A focused frame audit found two concrete MuJoCo evaluator errors without finding a gross
`xyzw/wxyz`, Z-up, gravity-sign, joint-axis or joint-name permutation error.

- Diagnostic `teacher-reference` reset copied the motion schema's pelvis-COM world linear velocity
  directly into the freejoint translation, which is the pelvis link-origin world velocity. The A3
  pelvis origin-to-COM offset is about `0.1273 m`; the corrected path applies
  `v_origin = v_com - omega_world x (R_world_body * body_ipos)` only to a clip explicitly bound as
  `center_of_mass`. Checkpoint-bound schema-3 native and standalone exports now carry
  `motion_body_lin_vel_points` for every clip; explicitly bound `link_origin` clips retain direct
  assignment and remain exact-ineligible. Schema 1/2 or contractless standalone re-exports strip
  the field rather than inheriting an unproved donor claim.
  Old exact schema-2 exports have one narrow all-COM compatibility rule. Missing/inexact aggregate
  metadata cannot identify a point and now fails loudly before teacher-reference reset instead of
  being guessed.
- `base_ang_vel` used `mjOBJ_BODY` with local output. In MuJoCo that is expressed in the compiled
  inertia-principal axes, not the pelvis link/IMU axes required by the actor and used for projected
  gravity. The corrected read uses `mjOBJ_XBODY` with local output. The vendor A3 pelvis inertia
  axes differ from the link axes by about `0.3315 deg`, so this is a real every-step observation
  mismatch but not, by magnitude alone, evidence for the observed cross-engine strike gap.
- The evaluator requires exactly one freejoint owned by `pelvis_link`, at qpos/dof address zero.
  Other free bodies, such as a dynamic ball, remain permitted.

A separate read-only audit found the analogous latent bug in the vendor ROS `SimReset` nonzero
base-twist subscriber: its published twist is world/odom link-origin twist, but world angular
velocity is copied directly into body-local freejoint qvel. Existing keyframe scripts and formal
K100 send zero velocity, so this does not explain their behavior and is not changed in this Python
evaluator ticket. The open G04/G07 interface contract is recorded in
[`frames_and_coordinates.md`](../interfaces/frames_and_coordinates.md).

The real `a3_pingpong.xml` regression uses nonzero orientation, three-axis angular velocity and COM
velocity; it checks freejoint origin velocity, COM world velocity and the actor gyro frame. The
focused reproduction is:

```bash
/Users/yyk956614/anaconda3/envs/backend/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_mujoco_reference_reset_com_frame.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py
```

This focused command passes `51` tests on the local CPU environment with the real MJCF test
executed and zero skips; the formal CPU contract group passes `115` with zero skips, the complete
contract union passes `183` with zero skips, and the repository's supported root `tests/` suite
passes `554`.
No policy rollout, Pod job, vendor backend or hardware ran. The reset correction does not change
formal `stand-keyframe` K100 because that path starts with zero qvel; the gyro correction does
affect its actor observation. The separately preregistered
vendor/root-only/joints-only/full-match ready-state four-cell remains unrun, so these source fixes
do not close the causal Isaac-to-MuJoCo gap. Full forensic scope and limitations are in
[`EXP-MUJOCO-PELVIS-FRAME-PARITY`](../experiments/2026-07/EXP-MUJOCO-PELVIS-FRAME-PARITY.md);
G06 remains `Partial`.

### 2026-08-28 FullMDP physical-birth first-divergence correction

clean `981327de` fixed tape先证明两端在首个action前已经不同：Isaac写asset default而MuJoCo写Take061
dynamic-ready，initial joint/root/racket-position最大差为`1.5199 rad/.1778 m/.5376 m`。该Isaac reset Event
实现已在`179148e3`修复；修后双端同commit initial q/dq exact，root position最大差`4.1e-7 m`、拍心position
最大差`.467 mm`，48 tick done/time-out无mismatch。这关闭的是出生实现错误，不是physics parity。

首版tape又被证明围绕错误的raw-zero问问题：Take061 live fresh actor mean最大绝对值`16.3001`，所以v1实际把
两端拉向asset-default q_des，并每8 tick全量终止/reset。当前tape改为同一tracked live actor mean中心的
`±.02`扰动。clean `3343fe90` centered exact compare中两端48 tick均无Done，但Isaac相对初态joint/root/
racket最大漂移`.349 rad/.092 m/.170 m`，Mu仅`.012 rad/.008 m/.010 m`；跨端joint max差从tick0
`.0059 rad`增至tick47 `.3480 rad`。v3 record已把shared decoder排除：tick0--34 actual q_des最大只差
`5.96e-8 rad`，而q在首个20 ms步已经分叉；tick35后Isaac腰滚先接近hard-inner才触发guard q_des分叉。
剩余是backend plant/controller response，不能偷换成Pod安装损坏；也不得用birth修复、同步Done或该分叉
包络宣布cross-engine parity、训练成功、transfer或部署。G06保持`Partial`。

### 2026-07-14 evaluator parity red-team closure (source gate only)

Independent review found that the first implicit-effort guard could miss saturation-equivalence
errors when P and D cancel, and that the first self-contact counter treated every non-world dynamic
body as robot. The corrected evaluator executes Isaac's total `clip(P-D)` law for bound zero-passive
implicit joints, rejects passive/unbound proxies from formal status, classifies only pelvis-subtree
robot geom pairs and fails formal BankExam on any such pair. A dynamic-ball negative control is
explicitly excluded.

The same review closed three evidence-publication bypasses: command-mask provenance now accepts only
canonical callables or strict empty built-in partials; the revoked model-2000 Phase-B rider is denied directly
by content SHA even when a caller bypasses the 2x2 validator; and cumulative scoreboard CSVs refuse
an old header rather than appending misaligned wider rows. The historical Phase-B launch command is
now documented as forbidden and a replacement requires a post-epoch checkpoint/new rider/all four
cells rerun. No new policy rollout, immutable K100, vendor backend, Gate3/Gate3B or robot result was
produced, so G06 remains `Partial`. Full scope is in
[the integration experiment](../experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md).

A second independent review found two residual fail-open paths before merge. First, `isinstance`
also accepted a `functools.partial` subclass whose overridden `__call__` executed different command
semantics while `.func` still named the canonical function; provenance now unwraps only exact
built-in partial objects at every layer. Second, self-contact was classified only after all physics
substeps in a control step, so an earlier transient collision could affect dynamics and disappear
before grading. Classification and formal refusal now happen after every `mj_step`, with diagnostic
substep aggregates retained. Dependency-free negative tests reproduce both attacks; the optional
real MuJoCo contact/frame modules remain separate runtime evidence rather than being inferred from
those tests. G06 remains `Partial` pending an immutable behavior paper.

The two optional MuJoCo modules were then collected on isolated Pod2 CPU. Invalid synthetic
controls failed first and were preserved; correcting only their execution-equivalence and
articulation assumptions produced `10/10` with the same production evaluator bytes. This is E2
source/runtime evidence for the helpers, not an immutable policy paper, vendor MJCF run, Gate3 or
Isaac-to-MuJoCo gap closure. Exact SHA are in
[`mujoco_eval_optional_runtime_test_results_20260714.json`](../../configs/mujoco_eval_optional_runtime_test_results_20260714.json),
and G06 remains `Partial`.

### 2026-07-14 C2/D2 L1 不授权跨引擎判卷

新的 [C2/D2 L1 source gate](G05_isaac_training_first_loop.md)修复的是训练 checkpoint 的证据身份：
guidance weight 进入相邻 hard contract，outer claim/source/GPU lane 进入 checkpoint `infos`。它只准备在
Pod1 GPU1/GPU2 买两条 fresh 25-update provenance smoke；当前没有 runtime contract、`model_24.pt`、
ONNX export、immutable signed-face **policy** paper execution 或 MuJoCo 结果。K100 schedule/paper-only
activation 已另行物化，但该 activation 明确把 trainer/L2/judge 全设为 false。

launcher/finalizer 源码中不存在 activation、judge、L2、第二 seed 或 stop/promote mode；pair result 也把
这些授权固定为 false。即使未来两个 L1 terminal 通过，也只能说明 source/claim/contract/finite/lineage
闭合，不能说明 guidance 有效，更不能作为 Isaac/MuJoCo、vendor Gate3/Gate3B 或真机成绩。进入 L2
之前还需独立、no-clobber 的 L1 result consumer，并把已物化 paper 的 exact receipt/SHA 纳入一个新的
L2-only activation；当前 C2/D2 pair result 自己不能翻转该 paper activation 的 false 位。启动 judge
又需另一个 reviewed execution contract。G06 保持 `Partial`。

### 2026-07-14 C3/D3 K100 v1 asset-packaging failure 与 v2 source gate

C3/D3 paired K100 v1 在 C3 ONNX 导出前自然失败：独立 eval checkout 缺少 Git-ignored
`agibot_a3/urdf/model.urdf` 及同闭包 meshes/config。tracked-clean gate 因上游
`assets/.gitignore:*` 没有覆盖这些 runtime bytes。v1 output/attestation 已消费并永久冻结；没有 ONNX、
MuJoCo attempt 或 K100 成绩，因此不能判 C3/D3 行为。

v2 在全新 attestor/pair namespace 中把训练 checkout 的 ignored A3 递归 canonical inventory、required
URDF、C3 一次 hydrate/D3 exact verify 角色与 `libGLU.so.1` 可加载性放到 claim/judge 前。focused
`56 passed`，static/source-plan rc0。用同一训练 asset closure 和 exact checkpoint/bank/plant binding
做的 C3/D3 inexact diagnostic 均成功导出并进入 MuJoCo，故 asset packaging blocker 已关闭；日志明确
为 `evaluation_contract_exact=false`，且两侧均在第 0 题前因
`formal BankExam reached bound PhysX joint-velocity limit on articulation indices [8]; MuJoCo lacks same
braking constraint` fail closed。两侧均 `asked=0`，没有 attempt/score/方向分；新的 open blocker 是
articulation `[8]` 的
velocity-limit braking parity。可复现合同见
[v2 操作](../operations/run_phase1_signed_face_c3d3_k100_v2.md)。G06 仍为 `Partial`：parity 修复前不得
执行 formal paired K100；明确 allow-inexact 的方向筛不能授权 L2、第二 seed、stop/promote、部署或真机。
evaluator→attestor→paired manifest 的 SHA 级联与 hydrate 并发 sentinel no-replace 攻击已在 source
层通过（focused `57 passed`、static/source-plan rc0）；没有新的 Pod runtime 或行为结论。

### 2026-07-15 frame-0 等待 v2 的跨引擎边界

新的等待设计合同冻结了同一条连续序列在两个引擎都必须满足的状态边界：揭题前未来 action/clip/frame0/
target/deadline 全部不可见；揭题时一次原子切换；切换仅改 reference，不能写 root/joints、teleport、
reset 或清 history/action/delay/noise；selected public action 的 frame-0 pose 使用 phase-entry station XY，
所有 root/joint/body reference velocity 为零。Ready 是全部数值安全/可达 tolerance 的 fail-closed 合取。

当前 v2 只过 exact-byte CPU design check。现役 `commands.py` 的 default-stand hold、未清零 anchor velocity
与 live per-tick XY reanchor 和该设计冲突；source adapter、numeric tolerance、carry-state runtime receipt、
Isaac full-scene probe 与 vendor MuJoCo continuous gate 均未绑定。因而不能把 v2 写成 Isaac/MuJoCo parity、
Gate3/Gate3B 或部署证据；`launch-check` 必须失败，G06 保持 `Partial`。复现入口见
[恢复操作](../operations/run_phase1_recovery_tuple_prereg.md)。

### 2026-08-03 N1 reward/event kernel 与 native physical-facts integration（历史 predecessor）

新增 `mujoco_native/n1_reward_event_kernel.py`，仅把调用方已 source-bound 的事实映射为 motion-mimic、A
target-window、closed-swing/hit、achieved outgoing flight、predicted outcome 与 observed net/legal-landing 的
独立布尔分母和付款资格。它不读取 MuJoCo state、不从 target/window 推断 contact、不预测落点，
也不分配 reward value。predicted outcome 严格要求 selected-rubber actual contact 后的 finite
outgoing flight；未命中 timeout 只保留 closed-swing 分母，不能付款。

production core 现在每 tick 返回 source-bound `a3_mujoco_n1_physical_event_facts_v1`：累计
racket contact edge count、首个 contact stamp、simultaneous/recontact invalid reasons，以及带
`(policy_tick, physics_substep)` 的首个 contact-free outgoing position/linear velocity/spin。VecEnv 要求
全部 core all-or-none 广告同一 contract，逐行重验 source/contract SHA、finite vector和严格事件顺序；
缺行或坏行会使整批失效。validated facts 进入 `DiagnosticBatchStep` 和 rollout v4 digest，并在
compact reset 前冻结 terminal tick。generic racket geom 命中继续只证明 blade contact；新增的
versioned selected-rubber classifier 只有在该 contact edge 已发生后，才用 official site frame、URDF
red/black outer planes 与 exact STL 派生的 strict inscribed safe disk 分类。safe disk 外为
`edge_or_rim_ambiguous`，球心在两 outer planes 之间为 `between_outer_planes_ambiguous`，两类都 fail
closed。classifier/question lineage 精确绑定 `action_id/action_uid/mount_normal_sign`、manifest/motion/
geometry/physics SHA、scene/assembled XML/backend/classifier SHA；legacy/manual question 仍无 authority。
source/dependency-light host focused tests=`73 passed, 28 skipped`。exact Pod detached clean
`4b43ac52` 再对 selected-rubber classifier、native ball core、reward event kernel 与 VecEnv 跑
`81 passed, 0 skipped, 0 failed`；这证明 current source 在目标依赖环境可 import/执行所覆盖路径，
但该 suite 没有发起实际球拍击球 rollout，所以 contact emission 仍为 `未测`；
`reward_authorized=false` 不变。

因此 normal `step()` 仍在 physics 前 fail closed。selected-rubber 的 source/classifier 子门已闭合，
但还需 exact Pod 真 MuJoCo ball-racket contact rollout；其余接口是 desired-contact/window、
outgoing-flight predictor、observed net/legal-landing resolver、swing-closure 和
per-term reward magnitude/weights receipt；全部齐备且能独立 replay sum-closure 后才能打开 normal reward。
在这个历史快照中 PPO/save/cold-load 尚未实现；后续76-D C-lite 已完成 reset-boundary cold-load，
current A211/C211 host code path 也已实现，但两族各自的 exact-Pod `1 env x 2 PPO update +
save/fresh-process cold-load` 仍为`未测`。mid-episode exact resume 继续缺
MuJoCo/core/ledger/delay/RNG state hooks，G06 保持 `Partial`。

action-specific hold 的候选身份也从 path-bound v1 升级为 portable v2。generator 只把 repo-relative
POSIX logical path 和 source SHA 写入 canonical payload；consumer 固定从 repo root 解析，拒绝旧 v1、
绝对路径、`.`/`..`、repo 外来源及 symlink escape，run-tape 的 root-MJCF identity 使用同一 resolver。
host focused=`18 passed,6 skipped`；exact Pod detached clean current source 的真 MuJoCo d0/d1/d2 focused=
`24 passed,0 skipped,0 failed`。这关闭了换 checkout 路径就改变 hold SHA 的工程 blocker，但还没有
进入 ball-racket contact、Reward 或 PPO。

portable hold 通过后，exact Pod `592835dc` 用同一 immutable question 得到一次真实 generic racket
edge、零 table edge、valid actual contact/outgoing flight；runtime selected-rubber sidecar 报正号红面，
`policy_tick=1/physics_substep=3`，球心切向距拍心 `0.007168732 m` 小于 strict safe radius
`0.044263876 m`，无 ambiguity/invalid reason。旧 CLI receipt 没有序列化这份分类，所以 successor
把 ball-core receipt 升到 v2，加入完整 classification seal、classifier/question/scene/backend SHA、
stamp、face sign 和半径；无 generic contact 或 generic contact 无 classifier 时均写 explicit unknown、
`fail_closed=true`。host Python3.14 focused=`15 passed,3 skipped`；exact Pod detached clean
`95382a53` receipt-v2 replay=`18 passed`，红面 face sign `+1`、tick/substep=`1/3`、tangential
distance=`0.007168732 m < 0.044263876 m`、invalid reasons=`[]`，且 receipt/backend identity seals
独立重算一致。selected-rubber contact receipt 子门至此关闭；Reward/PPO 仍未授权，G06 保持 `Partial`。

### 2026-08-19 portable FullMDP A纵切片边界

portable MuJoCo已真实穿过row-wise reveal、ball state launch、20-substep plant、live generic racket
contact、bounded terminal与selected reset；exact Pod节点用真实contact rows证明production latch，而非
runner手造extras。host路径又将真实postphysics racket FK写入R03 row，并消费Reward项0--9。runner固定
发布`task_lifecycle=full_a_slice_attempted`、`full_a_complete=false`。

当前确定缺口是共享73-action lineage和mount sign、selected-rubber contact、R06 landing outcome、R07
recovery及Reward项10--13；R03/Reward0--9还缺fresh GPU调用证据。因此portable MuJoCo Full-A长跑继续
`HOLD`。native 114/114-D A1000的吞吐与contact提升只能作为工程/Reward经济参考，不能代签
229/399-D portable语义。该缺口不阻塞Isaac 4096 A长跑；Isaac运行期间继续接producer，但不得热补
活跃Isaac源码或把partial receipt改名为Full-A成功。G06保持`Partial`。

### 2026-08-19 action0 identity与observed selected-rubber Reward10

fresh Isaac FullMDP运行态已经核为固定action slot0：73行manifest/motion bank只作冷身份与reference，
fresh cadence、genesis、Device-R05和selected reset不消费legacy balanced sampler。为避免MuJoCo另造动作，
新增dependency-light portable catalog作为唯一manifest pin、center-column与冷FK reference source；既有9字段catalog dataclass
ABI保持不变，commands和timing owner只re-export同一真源。MuJoCo Full-A state固定绑定同一个slot0、UID
`6907688916670928`和manifest mount sign；没有round-robin或`env_id % 73`。

generic contact仍由MuJoCo live contact rows产生。第一次contact edge出现的同一physics substep立即读取
ball center、racket site position和site rotation，再用engine-neutral Torch kernel区分selected、opposite、
edge/rim、between-planes和invalid。分类kernel绝不由距离制造contact；strict safe-radius与outer-plane
边界沿用shared racket geometry。只有selected分类发布一个control-step事件脉冲并写
`PHYSICAL_SELECTED_CONTACT`，所以Reward10只支付一次；held contact不会持续付钱，invalid只写owner fault。

host复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/tests/test_action_ball_full_mdp_portable_catalog.py \
  hope_training/whole_body_tracking/tests/test_mujoco_selected_rubber_classifier.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_gpu_ac_full_mdp_initial_wait_env.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_gpu_ac_full_mdp_wait_transition.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_gpu_ac_full_mdp_wait_rsl3.py \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_gpu_ac_table_keepout.py
```

结果=`38 passed, 9 skipped`。Isaac catalog/cadence/timing回归为`47 passed, 24 skipped`；另外真实
`MotionLoader + RacketTargetCommand`冷builder对73条motion逐列对拍的整文件回归为`12 passed, 3 skipped`。
skip均是
需要Isaac或MuJoCo-Warp GPU的节点，不能算live证据。动作条件化question/selected teacher、R06/R07、
Reward11--13、per-action/per-side分母和fresh GPU真实contact/reset仍未闭，因此runner继续固定
`full_a_complete=false`，G06保持`Partial`，portable长跑继续HOLD。

首次fresh GPU1组合尝试固定commit `e6cefaa0…`与namespace
`mujoco-fullmdp-contact-e6cefaa0.Pod1GPU1.2nfiiysp`，自然`rc=1`且未重试。首个N1节点在真实launch后
先遇到独立姿态/桌面terminal，旧测试却把任意`done`都要求为Full-A flight outcome，因而在
`outcome_code=NONE`假失败，selected-contact与N2 reset两节点尚未执行。tests-only窄修把该节点的shot
deadline钉到launch transition：仍经过一次真实physics并验证ball移动、outcome与selected reset，但不再让
无关terminal抢先。host文件=`11 passed,6 skipped`；新commit/fresh namespace GPU复核待执行。

随后准备的fresh namespace `mujoco-fullmdp-contact-cdfc5ad1.Pod1GPU1._p88w9df`没有进入GPU或pytest：
一次性脚本手抄了错误的完整commit SHA，checkout自然`rc=128`。该namespace保留且不复用，本轮不做第三次
尝试；下一执行件必须从本地`git rev-parse HEAD`机器读取完整SHA，而不是人工扩写短SHA。

2026-08-19下一次空闲GPU1件确已从`git rev-parse HEAD`绑定完整`0790504c…`，fresh namespace为
`mujoco-fullmdp-gpu-gate.0790504c.Pod1GPU1.E6oHC4xX`，且checkout/test SHA门通过；但CPU隔离命令写成Pod
不存在的`/usr/bin/numactl`，在Python/pytest/MuJoCo/GPU零调用前自然`rc=127`。该namespace与result/log封存、
本轮不重试；下一件只能使用已只读确认存在的`/usr/bin/taskset`，并在创建namespace前把tool origin作为
precondition。selected contact与masked reset live门因此仍`未测`，G06保持`Partial`。

### HISTORICAL / SUPERSEDED — 2026-08-20 portable R06/R07与非终止shot reset

portable Full-A host纵切片现已把selected-rubber后的首次net证据、landing crossing、R06 denominator与
Reward11/12接到同一生命周期。普通no-contact是有PRESENT/POLICY/SOURCE的零奖励样本；invalid evidence只
PRESENT并写fault。placement只允许selected contact、首次landing、过网且清网、飞向对侧，off-table的0.5
只适用于已经到达对侧的球。Physical与R06 critic fact会保留，Reward10--12仍只消费当步event pulse。

R07以shared deadline而非早落点为年龄原点，使用真实MuJoCo contact-force法向分量和双脚`>=10 N`、
0.9 soft joint-limit、finite normalized quaternion与同一frame0 recovery teacher。每row还维护age `10..77`
连续性、expected/eligible=`68/68`与sticky fault；跳过格或NaN后恢复都从故障格起停止Reward13，并在允许
success/timeout或selected reset前fail closed。shot完成只清ActionEpoch/ball row并增加generation，不再发
Gym done、重置机器人、动作或episode；真实safety/time-limit才进入全环境reset。

四份direct suite为`49 passed, 8 skipped`。两个GPU selected-reset节点不再伪改recovery clock，而是要求
真实`env.step`逐格走完age 1--77并验68个R07 present/eligible、末格才reset。这些skip仍包含真实
MuJoCo-Warp selected contact、contact force与nonterminal reset；没有fresh GPU PASS、4096容量/吞吐、
terminal consumer和25k学习趋势前，不得把该slice改名为portable Full-A成功或长跑完成。

同日fresh GPU0 slot-2 gate以commit `ed11390e…`和独立CPU `16--31`与Isaac共卡；发车前目标卡只有一个
compute peer且free memory超过20 GiB。WAIT N1/N2两节点真实PASS，第三个Full-A节点在构造期自然失败：
base `super().__init__()`内部先调用`reset -> _compute_obs`，而Full-A motion/teacher buffers必须在base构造返回后
才能原子安装，旧代码因此访问不存在的`_full_a_motion_phase_code`。结果=`2 passed, 1 failed`、RC=1；
namespace `mujoco-fullmdp-gpu-gate.ed11390e.Pod1GPU0Slot2Taskset.50yWOSvf`封存且不重试。

窄修让构造期 observation在`_fullmdp_initialized=false`时保持原deterministic WAIT surface；只有buffer整体安装后
才发布Full-A phase/teacher/task/facts。host新增删除三份Full-A buffer仍能完成构造observation的反例，四份suite=
`50 passed, 8 skipped`。该修复只关闭初始化顺序，不代签剩余三个GPU节点；fresh successor仍须新commit和
新namespace，G06继续`Partial`。

constructor fix的fresh commit `3534eeb0…`随后在同一slot新namespace
`mujoco-fullmdp-gpu-gate.3534eeb0.Pod1GPU0Slot2Taskset.GqVOyqcQ`越过构造并完成第一步真实Full-A；
GPU-only test随后在生产physics断言前被自己的extras exact-key集合拒绝，因为该集合漏列生产已经发布的
`full_a_landing_opponent_bound`。结果=`1 failed`、RC=1，该namespace封存不重试。tests-only窄修把该字段
加入exact集合；production 0行变化，仍须fresh commit/namespace执行三个节点。

tests-only successor `66bc4c87…`再次越过构造、exact extras、launch与outcome，然后在R07循环前被fixture的
`initial_age==0`断言拒绝；真实deadline锚点为tick 1，而该step结束后common counter已到2，所以正确年龄为1。
结果=`1 failed`、RC=1，namespace `mujoco-fullmdp-gpu-gate.66bc4c87.Pod1GPU0Slot2Taskset.DCj7RXfQ`
封存不重试。修正仅让GPU test从真实`initial_age+1`推进至77，仍硬验age 10--77共68个present/eligible格、
末格非终止selected reset及精确episode增量；production仍0行变化。

fresh tests-only commit `c2d8c536…`继续越过上述三层，并在真实recovery循环中暴露首个生产行为阻塞：
age 77之前MuJoCo触发Gym safety reset，测试对shot完成不得调用`_reset_idx`的硬门因此失败。结果=`1 failed`、
RC=1；namespace `mujoco-fullmdp-gpu-gate.c2d8c536.Pod1GPU0Slot2Taskset.d4aWD12U`封存不重试。
这与已记录的take061 physical-ready / take058 action-frame0控制消费缺口一致，但本次trace尚未打印具体
termination bit，不能把原因进一步写死为tilt/low/table。当前不能启动portable 25k：否则会把真实safety
episode reset混成R07 nonterminal shot reset。下一件先在fresh诊断中记录首次reset age、terminal bits和plant state，
再修production control/hold边界；G06保持`Partial`。

fresh diagnostic commit `c0aa32e9…`现把首个reset钉死：recovery age=`16`、terminal bits=`16`即
`robot_hit_table`；base position=`[0.1073775,-0.1811592,1.0640811]`、projected gravity约
`[-0.020965,0.001150,-0.999780]`，resolved robot-table contact=`0`，但exact geometric
table-keepout=`true`。因此不是tilt或base-too-low，也不能简单归因split-ready角度gap；真实阻塞是物理hold过程中
机器人OBB/拍面在约0.32秒后进入table guard。namespace
`mujoco-fullmdp-gpu-diag.c0aa32e9.Pod1GPU0Slot2Taskset.EdCf3ic7`封存不重试。下一步须定位首个命中的
component/blade、与Isaac同ready/table frame对拍，再决定修ready控制还是frame/guard；不得关闭guard绕过。

### 2026-08-22 课程解阻后的fresh portable MuJoCo（Gate仍`Partial`）

旧H48 run在ACK3023累计`due=901 / reveal=0 / deferred=901`，继续等待不会自然产生任务分母。课程与
reset-ready reference的同一最小修复已经exact Pod聚焦回归；launcher另把child cwd从source checkout改为
唯一run root，防止`MUJOCO_LOG.TXT`污染immutable source。该路径回归=`11 passed`，没有用`.gitignore`
掩盖错误输出边界。

fresh commit `23c0f6c…`、namespace `fullmdp-a-h48-v2-mujoco-unblock-23c0f6c8-20260822`现于GPU0运行。
22:55 UTC快照已连续到至少update47，Reward/storage全finite，nonfinite/conservation fault均为0。最近20轮
wall中位约`9.634 s/H48`、约`20,408 transitions/s`，H24-equivalent约`4.82 s`，进入旧`6 s/H24`要求的
量级。early update0--9到recent update38--47的episode均长从`105.61`升到`149.75 tick`；update45首次
出现`due=1 / reveal=1 / deferred=0`，同一rollout随后有phase-2 task row `27`个、R03 physically-valid
`26`个并产生Reward0--9非零梯度。这是自然课程首穿的runtime证据。selected contact与landing仍为0，
所以击球和上台继续记`未测`。近期`motion_body_ori`约`2,526--3,831`，说明错误reference/单窄核造成的
近零梯度已经解除，但不能代签mimic或击球成功。

Isaac与MuJoCo现在是两个独立clean detached checkout和两个fresh namespace，不共享可变source。旧run均按
精确PID/start-ticks停止、root/checkpoint/admin pre/post证据只读保留，completion没有伪造。双后端同时live
只关闭工程发车与早期finite证据；学习阶段对拍、selected contact、landing、formal completion、transfer与
部署仍未成立，`diagnostic_unauthorized=true`，所以G06保持`Partial`。详细证据见
[`EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822`](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md)。

### 2026-08-22 q_des与mimic→contact闭环候选（Gate仍`Partial`）

portable FullMDP现与Isaac调用同一纯tensor executable-q_des guard，包含finite fallback、共同soft/hard
inner projection、鲜q/qdot 20 ms crossing brake与终止位；alignment ledger以live AST call/参数反例将该轴
从`DIVERGENT_DECLARED`改为`ALIGNED`。这只关闭该轴，203/219 Observation以外的critic/termination/plant
物理差异仍各自保留。

slot0确定性构造还发现旧portable teacher与contact约错开13 cm；现由measured strike site和共享exact-face
ball-centre offset唯一构造contact，且共同Isaac D05补回此前漏掉的球半径。host证明perfect mimic在strike
tick的ball/site/selected-face几何闭合；真实MuJoCo-Warp/PhysX contact、landing与多seed学习曲线仍需fresh
GPU证据，不能用该代数测试授权transfer或promotion，G06保持`Partial`。

### 2026-08-22 MuJoCo ready-pose显式绑定候选（Gate仍`Partial`）

`dd82bb7b`的首次fresh launcher已取得GPU0 lock并创建独立run root，但child在PPO前因
`ACTIONBALL_READY_POSE`未绑定自然RC1。此前dry-run只验证argv/GPU env，外层shell偶然存在的变量成了隐藏
运行输入。候选新增required `--ready-pose`，核canonical regular file与固定SHA后再显式传给child；missing、
wrong SHA和symlink均在建root前拒绝。该失败root不resume、不复用。新的exact Pod test与连续ACK仍待完成，
不把launcher host PASS写成GPU或学习证据，G06保持`Partial`。

### 2026-08-23 portable V3阶段复核与evidence V4候选（Gate仍`Partial`）

只读到portable MuJoCo V3 update1256时，recent100 episode均长约`230.73 tick`，累计
`due/reveal/deferred=10648/10648/0`，但只有一次launch且没有R03/contact/R06/landing；timeout仍为0。
这不是reference重新偏移的证据：reset-ready→measured frame0 bridge、official racket site与ball-centre
几何已经由既有TODO/readiness实验和确定性反例闭合。当前判断仍是balance改善、mimic未基本形成、hit/landing
`未测`。recent100 raw H48 mean/median=`11.269/11.262 s`，未达约6秒量级目标。

branch候选让portable launch只在exact tick发生、missed tick在PPO前具名失败；invalid contact退休shot而不
伪造Gym reset。evidence schema 4在原有一次host reduction中检查完整rollout finite、done二值与sigma为正，
consumer拒绝旧schema冒充兼容；q-des Done语义与Isaac一致，actual hard-edge与projection intervention保留
telemetry。Mu launcher的Warp/CUDA/TMP/pycache均落fresh run root。

host聚焦与launcher证据同G05；exact EPA48/RSL3 Pod runtime、clean detached source、fresh V4 ACK、真实contact/
landing、双后端physics parity、12500 completion与transfer均未测。G06继续`Partial`，不授权resume、promotion、
export、部署或真机。

### 2026-08-23 V4最终冻结与clean V5双fresh当前Gate（仍`Partial`）

两条live run使用的clean且已push source为`39f9481950a660e198dedac7fd402806d648906b`。exact Pod broad CPU/ABI
回归为`792 passed, 57 skipped, 0 failed`；同一clean source另做runtime focused重跑为
`77 passed, 0 failed`，后者不与broad suite相加伪装unique总数。MuJoCo GPU direct为`5/0`，RSL H48
one-update为`1/0`。Isaac GPU projection `4/0`、selected-rubber `27/0`、runner pre-optimizer drain `1/0`，
合计`32/0`；runner drain只关闭runner拒绝边界，不是Isaac integration，下面真实Kit fresh run是更强证据。

旧V4 MuJoCo最终冻结在durable ACK `0..4798`（`943,521,792` transitions），进程已停止且没有completion。
recent20 wall mean/median=`14.146/13.994 s/H48`；累计public `due/reveal/defer=1,637,789/1,637,789/0`、
launch=`527,957`、R03 physically-valid=`145,814`，racket contact、selected contact和landing均为0。
累计episode=`3,376,589`；终止原因marginal计数为base tilt=`1,521,819`、
`robot_hit_table=1,860,583`、`base_low=383`；
finite、fault与conservation均为0。它只证明旧实现能自然到达due/launch/R03，不能把0 contact裁成课程定论，
也不代签V5；整条旧run仅为`diagnostic_unauthorized`证据。

fresh Isaac namespace `fullmdp-a-h48-v5-isaac-chronology-39f94819-20260823T144237Z`在GPU1取得验收前缀
ACK `0..63`（`12,582,912` transitions）。最后5个console iteration为
`7.78/7.47/8.17/6.79/8.19 s`；累计`139,264`个episode全部base tilt，D05、motion、launch、contact、
outcome和recovery event全部为0，finite、fault与conservation均为0。后续只读到ACK97呈方向性balance学习：
10-update窗口episode length mean约`87.606→97.776`，return约`7.052→8.705`；但仍无due，不能称balance
完成。该真实Kit路径保持`diagnostic_unauthorized=true`。

fresh MuJoCo namespace `fullmdp-a-h48-v5-mujoco-chronology-39f94819-20260823T144237Z`在GPU0取得验收前缀
ACK `0..8`（`1,769,472` transitions）。recent5 wall mean/median=`8.727/8.777 s/H48`，吞吐
`22,528.7 transitions/s`；累计`16,378`个episode、mean length=`105.213 tick`，全部因
`robot_hit_table`结束。due、launch、R03、contact、outcome、landing和recovery全部为0；finite、storage、
fact-integrity、lifecycle fault与conservation均clean，policy std从`.02000`到`.019945`。该前缀同样只是
`diagnostic_unauthorized`证据。

学习轨迹现原位冻结到Isaac ACK450 / MuJoCo ACK385；窗口与分母详见
[双后端TODO §0.4](../operations/action_ball_dual_backend_longrun_todo_20260819.md#04-2026-08-23-v4最终冻结与v5第一性原理自查)。
Isaac最新20窗episode mean length/return=`170.249/15.625`，从旧低谷恢复并超过旧峰值；累计
due/public=`20/20`、ACCEPT/reject=`17/3`、playback=2。Mu最新20窗=`186.693/17.771`，累计
scheduled/public/overlap=`307/297/10`、launch/missed=`6/0`、R03 present/physically-valid=`1/1`。
两端contact、outcome、landing与recovery仍为0；Mu contact=`0/6 launch`是小样本diagnostic negative，
Isaac contact因launch=0为`未测`，两端landing因selected contact=0为`未测`。独立mimic成功仍未闭合。最新冻结wall为
Isaac stdout辅助mean/median=`9.488/9.465 s`、Mu durable mean/median=`9.235/9.223 s`，约6秒方向未达。

V5保持actor/critic `203/219`、`history_length=0`，不增加offset、Observation、Stage或balance Reward；
自然课程仍是balance→mimic→hit→landing。H48只表示缩短迭代墙钟的工程方向；上述fresh速度没有
profiler-off matched-strata对照，不能正式归因或声称性能Gate通过。stdout marker不是authority：durable
证据只认optimizer后WAL/fsync、owner ACK与EPOCH_ACK/fsync链。live 39f的regular-file stdout未失败；下一
source `a3c528f1…`已把两处post-durable裸print改为best-effort并增加closed-pipe反例，不重启live。

Mu summary的`business_chain_complete`当前只代表producer逐row attestation加consumer边际聚合一致性，不是
独立same-env/same-epoch重放；晋级前仍需keyed carry-state或可重放trace。

当前仍未闭合自然mimic/hit/landing分母、12500 completion、formal independent playback、keyed chain replay、physics/transfer
parity以及promotion/deploy。两条fresh run均不授权resume、export、部署或真机，G06保持`Partial`。
