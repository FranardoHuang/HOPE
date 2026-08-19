# EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819

> 问题：portable MuJoCo怎样在不复制Isaac owner graph的前提下，逐纵切片消费同一slot0题目、measured teacher、真实事件与Reward20，直到可以做`4096×25000`长跑？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`partial-host / GPU-and-longrun-HOLD`
> 证据等级：E1源码与host反例

## 1. 采用、延后、拒绝

采用：

- fixed action严格沿用现役Isaac cadence的slot0/UID `6907688916670928`，不把已cold-load的73行bank冒充73动作训练。
- cold load一次校验manifest/motion SHA、31-joint order、tracked-body order、mount sign与measured timing；hot step只消费device tensor。
- question由slot0 manifest center、live base yaw、shared Physical reverse-integration和integer control tick生成；world origin只在qpos launch写入时恢复。
- Motion teacher按Isaac真语义：public reveal原子切measured frame0，prepare阶段reference velocity为0，rounded clock离开frame0后才进入swing。

延后：

- R06 legal landing/outcome、R07 recovery、Reward11--13和per-side denominator；它们必须消费真实事件，不能由host fixture自签。
- contact census合并、scratch/reward预分配和20-substep CUDA graph；先闭正确语义，再做4096 matched优化。

拒绝：

- midpoint serve、`x+vt-0.5gt²`、`normal=-incoming`与generic-contact冒充selected-rubber。
- 把take061 reset直接改成take058 frame0；physical-ready是独立admitted reset事实。
- 把diagnostic qdes bridge写入Motion observation或Reward teacher。Isaac只有命令consumer做递推，teacher本身立即是frame0。
- 用WAIT `N=2×learn(1)`或native 114/114-D速度冒充portable Full-A长跑。

## 2. 本轮纵切片

新增dependency-light `mujoco_full_mdp_portable_question.py`：

- 读取fixed slot0的sealed measured motion，验证`96×31` joint、tracked body、50 Hz、strike frame52、joint-order contract和mount sign；
- 从live env-local base pose构造action center，调用与Isaac同源的`physical_ball.back_integrate_incoming`做discover-then-integer-tick reverse flight；
- 生成45-D task与13-D launch state，并让contact tick严格命中measured strike frame；
- 提供无production consumer的`step_diagnostic_split_ready_qdes_bridge`，仅逐字表达Isaac oracle命令递推，不导出body/teacher接口。

MuJoCo Full-A env现在：

- reveal只安装env-local task/launch；prelaunch球保持park，launch才加对应env origin；
- actor teacher joint列和Reward20 body teacher在reveal立即读取frame0，prepare速度为0；
- selected reset恢复原take061 ready，下一reveal重新发布frame0，peer row不漂移；
- R03 expected source延到真实contact tick，不再在reveal后第一拍伪造strike fact。

## 3. 反例与结果

focused host=`22 passed, 7 skipped`，另有`py_compile`与`git diff --check`通过。反例覆盖：

- 改action center/incoming center会改变task与launch，旧midpoint shortcut不能假绿；
- 两个env origin只影响world qpos写入，不污染env-local task；
- take061到take058最大raw joint gap约`3.29183197 rad`时，reveal不改physical reset，但teacher立即exact frame0且速度为0；
- 任意中间bridge teacher会同时改变actor teacher字段和dense body reward，测试直接拒绝；
- contact tick97精确采到strike frame52；selected clear/reveal与peer preservation逐行验证。

## 4. 仍未闭合

1. production没有certified split-ready hold-qdes、ready到frame0 command bridge、逆动力学或前馈消费链；纯helper不算调用点。
2. shared reverse kernel、selected-rubber、masked reset与teacher全链尚未在fresh MuJoCo GPU同一次真实env执行。
3. R06/R07、Reward11--13、legal outcome/recovery与一次性payment/retire仍缺。
4. 完整生命周期、per-action/per-side denominator与terminal consumer闭合前，不生成portable `4096×25000` wrapper。

因此本轮是严格的host语义前进，不是Full-A完成，也不是长跑授权。
