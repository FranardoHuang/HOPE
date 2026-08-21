# EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819

> 问题：portable MuJoCo怎样在不复制Isaac owner graph的前提下，逐纵切片消费同一slot0题目、measured teacher、真实事件与Reward20，直到可以做H48的`4096×12500`长跑？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`successor-tools-branch-candidate / Pod dual-wheel+fixture+tape+H48 rate PASS / fresh-longrun RUNNING`
> 证据等级：E2 exact Pod H48 wall/fixed tape与fresh ACK前缀 + E1源码/host反例；无ASan physics-promotion或完成证据

<a id="epa48-fresh-runtime-binding-20260821"></a>
## 0. 2026-08-21 successor的EPA48 fresh runtime binding

本节是runtime binding详细真源，并supersede下文的r3 active及旧wire叙述。r3已止于ACK `10249`且
不得resume；当前实现仍是branch candidate，尚未合入`main`。fresh训练namespace已从clean detached
`96f0ca69887aba44c71983529d05e759e1a4cd2f`真实发射，身份和前缀证据见本节下文。

Full-A新增[`--mujoco-warp-runtime-site`](../../DEFINITIONS.md#mujoco-fullmdp-longrun-flags)（本次run的
MuJoCo-Warp/RSL-RL隔离导入目录）：路径必须绝对、父目录已存在且canonical，site本身不存在、非symlink且
不在`sys.path`。正常fresh CLI中binder先于runner自身的Torch/RSL-RL/WAIT/MJLab/MuJoCo-Warp import；
legacy WAIT不绑定，不带`--full-a`时传该flag也拒绝。binder以`O_NOFOLLOW`读取并在前后核regular file、
单hard-link、device/inode/size/mtime/ctime及canonical path，只接受三份ignored输入：

| 输入 | 固定SHA-256 |
| --- | --- |
| `vendor_assets/mujoco_warp_epa48_1/build_receipt.json` | `336f6454296d3c062e26fb0c330d6dbca4b2fd0ad4e50f386f8a647db013e041` |
| `vendor_assets/mujoco_warp_epa48_1/wheelhouse/mujoco_warp-3.10.0.3+hope.epa48.1-py3-none-any.whl` | `58f47b1c3b4249d82666f25d3a302ff5a215043a3d7a3b9445a5ca7ef15b561a` |
| `vendor_assets/rsl_rl_3_1_2/rsl_rl_lib-3.1.2-py3-none-any.whl` | `406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d` |

三份stable-read命中SHA后才创建mode `0700`的site，把两枚wheel解到**同一个**`sys.path[0]`；失败site
视为spent。预载`mujoco_warp/rsl_rl/mjlab`，输入缺失/变化/symlink，site已存在/别名化，或
distribution/spec/module origin落到foreign site，均fail closed；import失败还恢复`sys.path`并清partial prefix。

成功绑定要求fresh site中的`mujoco-warp==3.10.0.3+hope.epa48.1`、
`MJ_MAX_EPAHORIZON=48`及loaded `types.py` SHA-256=
`391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696`。
RSL-RL 3.1.2也须在同site赢得distribution/spec，随后仍由既有process-local gate核版本、六份source SHA、
module与live class/callable origin。持久化的`mujoco_warp_runtime`仅含EPA schema-1 mapping，并进入ACK、
snapshot infos、completion、summary；RSL不复制进去。wire升级为ACK `2→3`、completion `3→4`、summary
`2→3`，无旧wire fallback。binder不读Git或自报`source_commit`；future launcher必须从clean Git truth
传入，不能把当前WIP或binder常量当clean source identity。

scoped runtime diff SHA-256=`df4d5ea686f017206da3ad7cee5ef328cae1079da19d8da92b061fafedaef2d3`；
host union=`125 passed, 2 skipped`。Pod1 current-branch WIP命令为：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /workspace/mjlab_venv/bin/python -m pytest -q \
  hope_training/whole_body_tracking/mjlab_lane/tests/test_mujoco_full_mdp_epa48_runtime.py -rs
```

结果`19 passed in 4.33s`，actual import覆盖EPA package/types与RSL package/runner，并核version/horizon/
distribution/module roots。因命令隐藏CUDA，它本身不是GPU physics或训练证据；后续同页已经补齐ActionBall
fixed-tape、H48 wall和fresh Full-A运行前缀。instrumented/ASan physics-promotion oracle仍`未测`。
因此仍为`diagnostic_unauthorized=true / checkpoint_authority=false / resume_authority=false`、
`full_a_complete=false`，不是training GO。

### 固定EPA差分fixture与replay-only候选

一次性临时搜索已找到tracked
[`mujoco_warp_epa48_ellipsoid_cylinder_cross_v1.json`](../../../configs/fixtures/mujoco_warp_epa48_ellipsoid_cylinder_cross_v1.json)
及同名MJCF。tracked replay在Pod root
`/workspace/franco/mktemp/epa48-tracked-replay-20260821-v1`、GPU2
`GPU-473a79f3-8736-6c7f-c3db-290c6be385b8` / PCI `00000001:BE:00`、两套Python环境和两份Warp cache上
各重复10次：stock24 masks=`[256]×10`/contact0；fork48 masks=`[0]×10`/contact1且raw dist/pos/frame
finite。verdict=`PASS_EPA48_FIXED_FIXTURE_REPLAY`；script/JSON/XML SHA-256分别为
`d8d055…24b7f7` / `5bd5fd…8d6e6` / `f611bb…c58ce`，raw stock/fork/summary结果SHA-256分别为
`92bebd…bc1a` / `831a28…3ce` / `af6694…dee`。标准GPU2 lock覆盖全程，结束apps empty/lock free。

branch候选只提交fixed pair与[`replay_mujoco_warp_epa48_fixture.py`](../../../scripts/replay_mujoco_warp_epa48_fixture.py)：
它要求stock24/fork48独立interpreter、独立cache、同一physical GPU UUID，no-clobber输出raw contact和summary；
host=`17 passed, 1 skipped`，上述exact Pod replay已PASS。该证据只证明“一对24-overflow/48-finite
CUDA差分”，不证明通用碰撞正确性；ASan/instrumented independent oracle、ActionBall fixed-tape、
matched H48和training authorization均保持HOLD。
该fixture/tool代码commit为`a331832a`。

### clean-Git H48 one-shot launcher候选

[`launch_mujoco_full_mdp_successor.py`](../../../scripts/launch_mujoco_full_mdp_successor.py)从clean Git truth
读取40-hex source commit，核canonical Python、fresh `/workspace/.../<namespace>`、GPU index→UUID、空compute
apps与显式flock；然后固定启动`4096×12500`、H48 recipe、save500及run-local EPA48/RSL3 site。lock覆盖child
lifetime，launcher前台等待自然rc；没有monitor、retry、signal、resume或`ACCEPT` gate。host=`11 passed`，
Pod1 clean detached `2e4279ba`已用真实`/workspace/mjlab_venv/bin/python`完成dry-run：输出精确绑定
source commit、Full-A `4096×12500`、H48、save500与run-local site；run root前后均不存在，lock inode/size/
mtime/ctime不变，且dry-run在GPU查询前返回。随后同一窄入口已从clean detached `96f0ca69`真实发射；这仍只
是branch-scoped、`diagnostic_unauthorized`工程长跑，不是training/promotion授权。launcher初版/venv入口
修复commit为`4468b681` / `2e4279ba`。

<a id="h48-fixed-tape-and-rate-probes-20260821"></a>
### H48 fixed-tape与有限rate probes

tracked [`mujoco_full_mdp_h48_tape_v1.json`](../../../configs/fixtures/mujoco_full_mdp_h48_tape_v1.json)
固定`N64×H48×31`、seed0、zero reset noise和SplitMix64 `[-.02,.02]` action tape。
[`probe_mujoco_full_mdp_h48_tape.py`](../../../scripts/probe_mujoco_full_mdp_h48_tape.py)先通过真实Full-A
binder建立run-local EPA48/RSL3 site，再构造`FullMdpInitialWaitVecEnv(full_a_mode=True)`；raw NPZ包含initial
qpos/qvel/actor203/critic219，以及逐tick Reward20、actor203/critic219、qpos/qvel、done/reason/timeout/table
contact/reset generation/phase/outcome/action key和全部公开Full-A event。record loader不信summary：它独立重算
tracked config与完整action bytes，核NPZ SHA、字段集合、exact shape/dtype/finite；compare拒绝NumPy广播，
从raw event重算缺失strata，只报告离散mismatch cell和数值difference envelope，不提供tolerance、PASS verdict、
promotion或training authority。自然H48未覆盖的launch/contact/outcome/recovery必须写`未测`，不得注入生命周期。
host=`6 passed`，独立终审`P0=0/P1=0`；代码commit=`404031d2`。

portable trainer的`--full-a --diagnostic-rate-probe`复用production H48/4096 PPO、Reward ledger、durable ACK/
fsync调用面，但固定`10 warm-up + 50 measured + 1 tail`，拒绝
`HOPE_ACTION_BALL_UPDATE_PROFILE`和`HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES`为非0；不写snapshot/completion，
只报告61项raw update seconds、measured p50/p90、transitions/s和peak CUDA allocation。`--num-updates`不能
覆盖61，正式12500/save500路径逐字不变；host=`23 passed,1 skipped`，代码commit=`3f730100`。

Isaac训练入口也只新增默认false的`task.action_ball_full_mdp_rate_probe`：true时在runner边界安装同一61预算，
但root `max_iterations`仍须缺席或为typed 12500，不能另开第二配方；profiler-on在Kit前拒绝。focused=
`147 passed,26 skipped`。第一轮Pod执行把clean detached checkout切到`404031d2`后，在GPU/lock前因ignored
EPA48 wheel/build receipt缺失而fail closed；只恢复本机按文档SHA复核的三份内容寻址资产后，同一GPU2 UUID
与外层lock自然完成两次tape、compare和rate，SSH rc0，source最终clean、前后apps empty、unlock后可重新获取。

两次tape各3072 rows；离散/reason/events全exact，initial qpos/qvel/actor/critic exact。连续repeat max/mean为：

| 字段 | max abs | mean abs |
| --- | ---: | ---: |
| actor203 / critic219 | `0.005191662` | `1.63e-5 / 1.51e-5` |
| qpos | `0.000633504` | `9.64e-6` |
| qvel | `0.103833243` | `0.001050465` |
| Reward20 | `6.0955e-6` | `3.25e-8` |

每次reveal due/deferred=`64/64`，所以reveal/launch/contact/outcome/recovery都诚实为`未测`；compare不据此
造PASS。A/B summary+arrays SHA分别为`195023ad…d4c43`/`0c0499d0…077a`与
`19bd8753…ae99`/`21444b7b…b6db`，compare SHA=`b6d5c6d…886c0`。

rate为`4096×48×61`，50个measured update共`9,830,400` transitions、wall=`473.078529 s`，
throughput=`20,779.63677/s`，p50/p90=`9.44830058/9.66072240 s/update`，全过程
`575.269436 s`，Torch peak allocated=`1,046,172,672 B`；evidence SHA=`5e7ba562…d763`。
H24-equivalent p50=`4.72415 s/update`，所以按当前“约6秒是旧H24量级、H48约12秒不是硬Gate”的取舍，
MuJoCo不再继续堆微优化。把6秒直接要求在H48并宣称仍差1.57倍的判读明确不采用。

instrumented/ASan仍是physics promotion/transfer的独立证据缺口，不是未训练policy表现门，也不阻塞本来就
`diagnostic_unauthorized`且保留overflow/nonfinite fail-stop的fresh工程长跑。

### Fresh H48 longrun运行前缀

clean detached source=`96f0ca69887aba44c71983529d05e759e1a4cd2f`；namespace=
`fullmdp-a-h48-v2-96f0ca69-20260821`；run root=
`/workspace/franco/runs/fullmdp-a-h48-v2-96f0ca69-20260821`；GPU2 UUID=
`GPU-473a79f3-8736-6c7f-c3db-290c6be385b8`。发射前empty-app与nonblocking lock通过，launcher PID
`2030437`持lock等待唯一child PID `2030453`；child exact argv为Full-A `4096×48×12500/save500`、fresh
runtime site且无resume/retry/signal/`ACCEPT`门。

update0 durable ACK：prepared SHA=`1d186373…614e53`，`196,608` transitions，collection/learning/pre-ACK=
`9.354775/0.284285/9.639704 s`；storage与Reward20 finite，conservation/nonfinite fault为0；
`model_0.pt`大小`6,617,367 B`、SHA=`50ebc7c9…7b26`。最近一次只读检查已见update `0..4`共5个连续ACK，
child仍为`R`。这证明真实production路径开始持续推进，不证明12500 completion或任何business stage已经出现。

## 1. 采用、延后、拒绝

采用：

- fixed action严格沿用现役Isaac cadence的slot0/UID `6907688916670928`，不把已cold-load的73行bank冒充73动作训练。
- cold load一次校验manifest/motion SHA、31-joint order、tracked-body order、mount sign与measured timing；hot step只消费device tensor。
- question由slot0 manifest center、live base yaw、shared Physical reverse-integration和integer control tick生成；world origin只在qpos launch写入时恢复。
- fixed cadence沿用shared 30 s / 1500-control-tick schedule：due opportunity固定为
  `2,295,588,881,1174,1467`（`2 + 293k`）；每行在due时按live readiness得到
  `ACCEPT`或`DEFER`。`DEFER`是zero-write，不在tick 3补试，下一机会仍是下一个冻结due。
- true Gym reset写`runtime_plant.default_joint_pos_rad`、配置default root加env origin、零joint/root
  velocity和零current/previous action history；`take061/q_ready`只保留输入provenance，不是physical birth authority。
- pre-swing HOLD期间public joint teacher是runtime default、joint velocity为0；body reference与R07 target
  使用measured frame0。只有`active_motion_s > 0`后才公开measured sampler，即使rounded frame ordinal仍为0。
- raw action affine的proposal、runtime/schema-2关节顺序、scale与
  `runtime_plant.default_joint_pos_rad` offset已与active Isaac闭合；`take061/physical-ready`只保留
  provenance，不是reset或affine-offset authority。
  executable q-des guard仍明确`DIVERGENT_DECLARED`：MuJoCo机械hard clamp没有复制Isaac的
  soft-inset与state-dependent brake，因此只授权MuJoCo-only诊断，不授权transfer/matched结论。
- successor直接目标是同一upstream RSL-RL 3.1.2进程`4096×48×12500 = 2,457,600,000`
  transitions；`1000`只是只读早期节点，不停车。
- thin update ledger只在optimizer boundary事务性记一条ACK；rate/window/per-side表全部
  由独立offline consumer计算。

延后：

- contact census合并、scratch/reward预分配和20-substep CUDA graph；先闭正确语义，再做4096
  profiler-off同源吞吐优化。
- C family与per-side非零分母；当前这一代只训slot0 forehand，backhand必须诚实为0，
  不为了表格对称额外发一条run。

拒绝：

- midpoint serve、`x+vt-0.5gt²`、`normal=-incoming`与generic-contact冒充selected-rubber。
- 把true reset改成take061或take058 frame0；fresh FullMDP birth authority是runtime default plant。
- 把diagnostic qdes bridge写入Motion observation或Reward teacher。它只表达历史命令consumer递推，
  不是fresh Motion public teacher或动作执行authority。
- 用WAIT `N=2×learn(1)`或native 114/114-D速度冒充portable Full-A长跑。
- 要求未训练zero-action policy在真实rollout中活过R07 age `10..77`才允许开训。
  68格连续性仍是确定性host lifecycle反例，但不是随机策略表现门。
- 保留一套额外keepout witness生产代码。现有exact keepout terminal不删；table/fall/bit16
  在训练中记telemetry，只有独立确定性证据证明frame/guard错误时才升为环境阻塞。
- 在hot runner内计算窗口表、成功率或每update向stdout打大JSON；这些派生项留给
  独立consumer。

## 2. 本轮纵切片

新增dependency-light `mujoco_full_mdp_portable_question.py`：

- 读取fixed slot0的sealed measured motion，验证`96×31` joint、tracked body、50 Hz、strike frame52、joint-order contract和mount sign；
- 从live env-local base pose构造action center，调用与Isaac同源的`physical_ball.back_integrate_incoming`做discover-then-integer-tick reverse flight；
- 生成45-D task与13-D launch state，并让contact tick严格命中measured strike frame；
- 提供无production consumer的`step_diagnostic_split_ready_qdes_bridge`，仅逐字表达Isaac oracle命令递推，不导出body/teacher接口。

MuJoCo Full-A env现在：

- reveal只安装env-local task/launch；prelaunch球保持park，launch才加对应env origin；
- phase只允许`IDLE=0 / REVEAL=2 / LAUNCH=5 / OUTCOME=6 / RETIRED=8`；
- true Gym reset使用default plant/zero action history；HOLD joint teacher保持default/zero velocity，body/R07
  独立使用measured frame0；
- natural recovery success或window-timeout发布`shot_retired`并进入phase8；它不发Gym done，不改robot、
  current/previous action、episode length或`reset_generation`，直到后续due真正ACCEPT才替换shot；
- 只有真实Gym done才发布`selected_reset`并使`reset_generation`恰增1；
- R03 expected source延到真实contact tick，不再在reveal后第一拍伪造strike fact。
- R06/R07、Reward11--13、shot-retire/Gym-reset和20-term守恒已进production extras，不再
  以`not_produced`应付runner；它们的真实次数可为0，但schema、事件和reset generation
  必须一致。
- thin ledger在device上累计26个lifecycle/event、5个terminal bit、classification、Reward20和
  reset/identity/integrity计数；每update只用一次`torch.cat(...).cpu()`把该固定向量送到host。
- 第26个`completed_action_epoch`不是边际推导：只有同一env行保留的launch、selected contact、R03、
  R06、无fault的68格R07与自然RETIRE同时闭合才发布；跨env拼接这些里程碑必须保持业务未完成。
- optimizer前要求48步、storage reward/return/advantage finite、done只能为`0/1`、timeout与bit1
  一致、done与terminal-or-resolved-table一致、`selected_reset == done`且generation delta只等于done，并验
  Reward20守恒和独立钉住的slot0/UID/sign/family。optimizer前只prepare冻结payload；只有upstream
  optimizer成功返回后才写snapshot并append+fsync ACK；optimizer/write/fsync失败不记假ACK。
- stock RSL serializer在update `0,500,...,12000,12499`留26份no-clobber快照；它们均
  `diagnostic_unauthorized=true`、`checkpoint_authority=false`、`resume_authority=false`。

## 3. 反例与结果

question/teacher历史focused host=`22 passed, 7 skipped`。当前MuJoCo env/action/outcome卷为
`59 passed, 7 skipped`，alignment当前两轴反例=`14 passed, 21 deselected`；runner/ledger/consumer集成为
`96 passed, 1 skipped`，其中production writer生成的真实prefix直接由独立consumer读取。另有
`py_compile`与`git diff --check`通过。反例覆盖：

Pod1 GPU2的前两条live one-shot已经产生两条封存反例；两条result均为
`status=failed_no_retry / final_rc=99 / ACK=0`，且trainer均未启动：

- exact commit `4aadd698e44f2e03a916ea1fd8c1daa1b2c2466c`、fresh namespace
  `mujoco-fullmdp-a4096-u25000-4aadd698-20260820t071549cst`、wrapper SHA-256
  `2d87de4d1c8a752d58266cfa1cebc092ac484a2235869c879f625cb21ccc8251`停在
  `first_error_phase=rsl3_source_gate`，首错为
  `ModuleNotFoundError: No module named 'tensordict'`。worker虽由
  `/workspace/mjlab_venv/bin/python`进入，child却直接调用`/usr/bin/python3.12`，丢失venv依赖；
  该namespace不重试。
- 同commit的fresh r1 namespace
  `mujoco-fullmdp-a4096-u25000-4aadd698-20260820t072045cst-r1`、wrapper SHA-256
  `6512a8c4de4c4627289caa6703d022f70ac130ed1e4fd4e03f3bacfb83b7090f`
  改为用venv入口执行child，同时独立核system Python identity；source gate越过，真实GPU focused
  为`5 passed, 3 failed in 23.11s`。三个失败均观测tick 2
  `due=true / deferred=true / reveal=false`，而旧测试错误要求zero action必然ACCEPT；该namespace
  同样封存且不重试。

r1的production观测符合冻结语义：due只是机会，live readiness为false时必须DEFER。r2没有改
production，只由独立GPU用例验证自然`ACCEPT XOR DEFER`、DEFER zero-write和tick 3不补试；下游
contact/outcome用例显式安装tests-only readiness。该注入只隔离真实调用点，不能计入policy或business
evidence。

fresh r2使用exact commit `9e7c1c614b1e22eeec4de243f55d58293da155ce`、namespace
`mujoco-fullmdp-a4096-u25000-9e7c1c61-20260820t073755cst-r2`和wrapper SHA-256
`36cc7a6166e1249061a45f7a3f7f1145a014a2b5a1f6f8417b84e0f58fefce5b`。真实GPU focused为
`8 passed in 23.00s`，随后进入实际4096-env trainer，首个optimizer update已经返回。紧接着stock
RSL-RL 3.1.2 save先写出7,882,391-byte `model_0.pt`，再因`log_dir=None`/`disable_logs`路径没有初始化
`runner.logger_type`而抛出
`AttributeError: 'OnPolicyRunner' object has no attribute 'logger_type'`；one-shot result为
`status=failed_no_retry / final_rc=99`。
此时evidence文件仍为0 bytes、ACK=0，`model_0.pt`没有对应ACK，只是封存且
`diagnostic_unauthorized/checkpoint_authority=false/resume_authority=false`的故障文件，不能计入26份
snapshot。该namespace封存且不可重用。最窄fresh r3修复只是显式安装upstream默认字段
`runner.logger_type='tensorboard'`以满足stock save，不启用logger、TensorBoard写入或上传；必须使用
新commit、新namespace，不能重试r2。

**HISTORICAL / SUPERSEDED —** fresh r3当时使用exact source `dc62684c41e70e40dedaf191a32921b6cd98b344`、namespace
`mujoco-fullmdp-a4096-u25000-dc62684c-20260820t074950cst-r3`和wrapper SHA-256
`0f5adc6024f01ffee7e761ab7b620d70855e541dbea298216ab9093e30695fd6`；worker PID=`864055`、单一
trainer PID=`865285`。真实GPU focused再次为`8/8`，随后同一trainer进程的
`4096 env × 24 step × 25000 update`已经启动且仍alive；result文件在active期间仍为0 bytes，不是失败seal。
当前durable ACK为`0..7`，`update_1`/`update_5`只读consumer均`passed`。首份已ACK
`model_0.pt`为7,882,391 bytes，SHA-256
`06883851e67ccaaa921cfeeb8bf5c983ee6b3443d67465d8cde1d08ed63f528f`，仍只有
`diagnostic_unauthorized/checkpoint_authority=false/resume_authority=false`。

前8个pre-ACK core iteration为`4.889..5.640 s`，median约`5.025 s`；这是active前缀，不代签25k
终局吞吐。每update的Reward20与actual reward均有98,304行finite，conservation fault=0且policy std
finite。行为telemetry中，update0为due/defer/ACCEPT=`4096/4096/0`；到update5累计为
`8192/8192/0`，update4出现4,096行exact-table/Gym reset。这些未训练policy表现不阻断engineering
run；当前`engineering_run_complete=false`、`business_chain_complete=false`、`full_a_complete=false`，
终点consumer与最终趋势仍为`未测`。

- 改action center/incoming center会改变task与launch，旧midpoint shortcut不能假绿；
- 两个env origin只影响world qpos写入，不污染env-local task；
- true reset的joint/root birth与action history逐项来自default/zero authority，不再从take061推导；
  HOLD joint teacher仍为default/zero velocity，而body/R07 measured-frame0通道保持独立；
- 任意中间bridge teacher会同时改变actor teacher字段和dense body reward，测试直接拒绝；
- contact tick97精确采到strike frame52；true-reset clear、phase8 retain/defer、后续ACCEPT
  与peer preservation逐行验证。
- 历史raw `[-4,+4]` clip使default-offset到slot0 frame0的五个关节需求
  `-6.473237/10.688715/-5.803502/6.699031/-15.155990`结构上不可达，而对应decoded
  q-des均仍在现有joint envelope内；去掉raw clip后该确定性不可学阻塞关闭。
- timeout错位、`done=2`、shot-retire误增generation、Gym done不增generation、common step卡住、Reward20不守恒、
  同源UID自证、optimizer/fsync失败与JSONL重复/缺口均会拒绝；event次数为0、
  table/fall多或recovery failure多不会被误拒。
- successor独立consumer要求完整文件有12500行、index `0..12499`无缺口/重复、每行
  `4096/48/196608`与最终累计精确，再另验26份有限model/optimizer快照。零分母的rate输出
  `null/未测`，不输出假0%。
- consumer分别输出`engineering_run_complete`与slot0 `business_chain_complete`。12500个ACK可关闭
  工程长跑，但业务事件全零时仍不得称slot0链出现；又73动作、双侧与科学窗口
  报告未闭合，本代`full_a_complete`固定为`false`。

## 4. 仍未闭合

1. historical r3已止于durable ACK `10249`；EPA overflow与旧IDLE clock泄漏共同禁止resume。下文
   `0..7`与active PID只保留为当时前缀，不是当前运行态。
2. successor的fresh dual-wheel import、tracked 10+10次fixture replay、ActionBall H48 fixed-tape与MuJoCo
   H48 wall已经实跑；instrumented/ASan physics-promotion oracle和Isaac matched wall尚未闭合。
3. 使用本runtime binder的fresh longrun已经发射并有schema-3 durable ACK前缀，但12500个ACK与schema-4
   completion尚未产生。C family、backhand分母、第二seed、promotion/export/deploy也仍`未测`，
   `business_chain_complete/full_a_complete=false`。

这些是真实剩余的证据边界。zero-action table/fall、零contact或低recovery率仍只是未训练策略telemetry，
不作为发车阻塞；现有overflow/keepout termination与证据完整性检查继续保留。当前是branch-scoped
runtime import closure与运行中engineering prefix，不是physics promotion、完整engineering longrun或MuJoCo Full-A完成。
