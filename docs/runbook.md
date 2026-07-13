# 跑批作战手册(常驻;发射/判卷/运维的全部操作知识,人话)

一句话:**这本手册让任何操作者能安全地发射训练臂、判卷、不踩已知的坑**。
当前 setting/阶段看 [NOW.md](NOW.md),具体实验看 [experiments/](experiments/README.md),
值得记住的 main 变化看 [TIMELINE.md](TIMELINE.md),动作管线看
[motion_pipeline.md](motion_pipeline.md)。本手册只负责操作,不承担项目状态或实验结论。

## 统一队列、排序与算力纪律

### 唯一账本

- 当前优先级只看 [NOW 的统一工作队列](NOW.md#统一工作队列唯一优先级账本)。不得再建
  “pod1 队列”“pod2 队列”或 agent 自己的并行账本。
- 任何空闲资源都取队列中最靠前且依赖已满足的项；被阻塞时先把阻塞写回该行，
  再取下一项。卡和 pod 没有永久角色。
- 具体 `run_name`、命令、PID/PGID、checkpoint、失败重试和产物路径只写进对应
  `docs/experiments/` 记录的 run table。`NOW` 只写优先级、责任人、阻塞和下一份
  会改变决策的证据。

### 排序法则

每项工作的顺序 = **结论保质期 × 解锁量**：后续改动会不会让它作废，以及有多少
其他工作在等它。因此默认顺序是：

1. 先固定尺子、数据合同、plant 语义和会改变所有实验的结构。
2. 再做出题/课程、动作源、观测和 reward 结构。
3. 结构定了才扫权重、锚点和小超参数。
4. 抗噪、部署加固和长跑放在结构证据之后。

新想法先过“机制检查 → 短信号档 → 留出卷”，不能结构未定就盲扫权重。

Seed 不是填满 GPU 的默认单位，而是晋级后才购买的稳定性证据：

1. 首轮机制筛选固定一个阻断 seed；同卡四槽放四个不同的单变量/小因子单元，不放同一失败配方的
   四个 seed。
2. 机制单元先跑 `512 env × 25 iter`，再以 `4096 env` 跑到相对 checkpoint `+1000`；使用现有
   cadence 的 `+200/+500/+1000` 做曲线。fresh runner 要写出 `model_1000.pt` 必须传
   `max_iterations=1001`（0 起数）；热启动把相对偏移加到父迭代号。canary 到预算自然结束，只决定
   是否值得延长，不把早期低分写成整个 family 永久失败。
3. 只有一个机制单元和匹配对照在同卷上显示较差侧改善，才给**这对**补第二个 seed。第二 seed
   不复现就停止，不用第三、第四 seed 投票救配方。
4. `3–4` seed、terminal 和整套双引擎/连续门只给可能成为 accepted baseline 的候选。

所有已经运行的 seed 仍必须全量报告；本规则限制的是下一份算力是否解锁，不允许挑 seed。
详细证据层级和 2026-07-11 过量复现的回看见
[Phase-1 消融加速制度](research/phase1_ablation_acceleration_2026-07-11.md#seed-是晋级税不是首轮并发单位)。

### RTX 5090 实测算力手册

以下是 2026-07-03 至 07-11 在 32.6 GiB RTX 5090 上的历史实测，是排程起点，
不是永久性能保证。更换代码、env 数、驱动或资产后先做短测。

| 任务 | 实测成本/纪律 |
| --- | --- |
| 4096 env 训练，单卡独占 | 约 `2.0–2.2 s/iter`；2k 约 1.2 h，4k 约 2.4 h，8k 约 4.7 h，12k 约 7 h，20k 约 12 h。 |
| 同卡 2 条 | 每条约慢 20–25%，总吞吐约增 37%；显存通常足够。 |
| 同卡 3–4 条广度消融 | 每条约慢 25–45%；07-11 实测 4 条/卡总显存约 `22.9–23.2 GiB`、GPU util `87–97%`。只用于“多测配方”优先的广度波。 |
| 关键路径/长跑/时间敏感验证 | 默认独占卡，不为填满空位拖慢决策。 |
| Kit 启动 + env build | 约 2 min。 |
| 512 env × 25 iter 机制检查 | 约 3 min。 |
| `play.py` 原生 ONNX export | 占一个 GPU 槽，约 4 min。 |
| MuJoCo 判卷 | 纯 CPU 约 25 min/checkpoint，可与训练并行。 |

调度硬规则：

1. 所有 Isaac 发射必须经过 pod 级 `/workspace/bin/kit_boot_lock.sh`，同时只启动一个 Kit；
   两次发射仍至少错开 60 s，避免 CUDA 枚举和冷启动缓存相撞。
2. **新增任务按全场 GPU 轮转，不先塞满一张卡。** 保留已经启动且合同绑定的实验原位运行；之后先给
   Pod1/GPU0→GPU1→GPU2、Pod2/GPU0→GPU1→GPU2 的每张可用卡各放第一条，再按相同顺序放第二条、
   第三条，只有 Pod1 才允许第四轮。若某张卡被团队成员占用、前置门未过或新任务会破坏严格配对，
   记录原因并跳到下一张可用卡；不得为了“整圈”迁移、重启或复制既有实验。每一槽都必须对应不同的
   可辨识问题和预注册早判，不能用失败配方的额外 seed 补位。
3. 3–4 条/卡不是默认权利；只有广度消融、短测确认显存/利用率后才可用。四槽是四个可辨识机制
   单元的容量，不是“同配方四 seed”的配额。
4. 判断空槽要同时看 `nvidia-smi --query-compute-apps`、利用率和日志活性；不只看显存或 PID。
5. CPU 重任务也属于同一队列；发射前比较各节点 `loadavg`，并设 `OMP_NUM_THREADS=1`。
6. 实际用时、显存或吞吐与上表明显不同时，在实验 run table 记录，再更新本手册。

## 环境地图(pod:162.43.172.171)

| 用途 | 路径/环境 | 说明 |
| --- | --- | --- |
| 训练(Isaac) | `source /workspace/franco/env.sh`(内含 isaac venv + HOPE_URDF_IMPORTER_NO_UI=1) | 跑 train.py / play.py 用它 |
| **判卷(MuJoCo 评估器)** | `source /workspace/hope_mjeval_venv/bin/activate` | **isaac venv 没有 onnxruntime,评估器在里面必死**(2026-07-06 实错) |
| 主检出 | /workspace/franco/nohope(main) | 评估器/工具从这里跑 |
| S1 分支 worktree | /workspace/franco/nohope_s1(stage1-fixed-point) | 训练臂从这里跑;**PYTHONPATH 必须把 worktree 的 source 放最前**,否则 import 到主检出旧代码 |
| 动作资产 | /workspace/shared/motions/*.npz | worktree 里 assets/agibot_a3 被 gitignore 全量忽略 → **新 worktree 要从主检出软链**(07-06) |
| 观测扩列脚本 | /workspace/shared/pad_obs_cols.py(末尾追加)/ pad_obs_cols_insert.py(按位插列);**末尾追加版 07-09 已收编进 repo**=`hope_training/whole_body_tracking/scripts/pad_obs_cols.py`(同源+CPU 单测) | 存档跨观测维热启用;jiayi 的 177 是**插在第 167 列**不是末尾;R10c 手术须从 repo 路径调用:`python hope_training/whole_body_tracking/scripts/pad_obs_cols.py src.pt dst.pt 179 181` |

## 发射核对单(每臂过一遍,过不了不点火;2026-07-06 立)

1. **卡上现有进程先 ps 认领**:每个占 GPU 的 pid 是谁的、活的还是死的(僵尸评估占槽让发射器
   死等 3 小时,07-06 实错;卡死两天的 eval 进程该杀就杀)。
2. 动作对非 hopex → `task.racket.strike_phase_per_clip` **显式传**(默认值是 hopex 的;
   题库只管目标不管时机——第一波正手全废的根因)。
3. 登记表五件套齐(缺烤入标记出题器 fail-closed 拦,v4 实锤)。
4. 卷与动作同源同锚(_cal 卷配 _cal 动作;锚=卷 meta 的 anchor_phase)。
5. 观测契约与存档维度匹配(179 臂 = 扩列存档 + `task.actor_obs_contract=null` 放行)。
6. yaml 未声明的新键用 `++` 前缀(question_bank/face_command/腕踢除都是)。
7. **两个 Isaac 不许同秒启动**(撞 CUDA 枚举,报"no suitable CUDA GPU";错峰 ≥60s,
   机制检查也一样)。
8. 重定向的日志**目录要先 mkdir**(目录不存在 → 发射壳当场死,连报错都看不到;两犯)。
9. 冒烟/机制检查的摘要 **grep 必须含 WARN|Error|Traceback**(只抓预期信号=确认偏误;
   落地警告就是这么漏的);**并确认含 `q_des CLAMP ACTIVE` 行**——限位剪切 2026-07-06 起
   默认开(jiayi 发现:不剪切的产品线在 MuJoCo 门禁里根本站不起来),缺这行=有人显式关了,
   只允许出现在"老配方复现"对照臂上。
10. run_name 当场进入对应的 `docs/experiments/` run table；若责任/优先级变化再更新 NOW。

## 判卷链(北极星数字怎么产;2026-07-06 全链踩通)

**标准入口=`hope_training/whole_body_tracking/scripts/judge.sh <run_dir> [checkpoint.pt]`
(2026-07-09 收编:自动解析 env.yaml 的动作对/相位/题库→原生导出+sidecar→双侧×双噪声档考卷→
md 报告落 run_dir/judge/;解析不到 fail-loud 要求手传;`--dry-run` 打印命令链,`--help` 看旗标),
以下手动步骤仅供排障。**

```
model_13599.pt ──play.py 原生导出(isaac venv,占一个 GPU 槽 ~4 分钟)──> exported/policy.onnx
    ──mujoco_eval_onnx.py(mjeval venv,纯 CPU)──> 按侧考卷分数
```

- **终版存档名是 model_13599 不是 13600**(rsl_rl 末迭代 0 起数;监视器等 13600 会永远等)。
- **导出后先补两个 sidecar 再考**(07-07 实错):原生导出不产 obs 归一化(obs_norm.npz)和
  噪声 std(learned_std.npy),缺前者模型吃未归一化观测必得 ~0 分,缺后者评估器直接 FATAL。
  一条命令齐活:`make_std_sidecar.py --checkpoint <model_N.pt>`(两个文件都落到 exported/)。
- **bank 题源考卷(阶段 1 正式卷)**:`--target-source bank --exam-bank <schema-v3 exam npz>` +
  显式 `--strike-phase-per-clip`;评估器会自动绑定当前 checkout 的
  `stage1_question_bank.py`,mjeval venv 不需要 Isaac 包链,也不应手工设置
  `HOPE_STAGE1_QB`;分母报表(kept/asked/锥内比例/难度中位)开头自动打印,入账连它一起抄。
- **`--qdes-clamp` 建议一律开**(人话:考卷把动作剪到关节限位,跟训练和真机部署一个规矩;不开的话
  考卷是三方里唯一不剪的,会错放"骑剪切板"策略/错杀靠剪切的健康策略——fixE 复核 07-08 定的);
  开没开都会打在报告头(`qdes_clamp=ON/OFF`),入账时连状态一起抄。默认关=老读数可比性不破。
- **判 07-05 后代际的臂建议 `--hold-ref stand`**(人话:等球段参考喂"站姿+零速",跟 07-05 以后
  训练的等球语义一致;默认 `clip`=冻结起手帧,是 07-05 前的老语义——拿老语义考新代际就是
  07-07 事故的形状)。两开关对健康臂无扰动已在 fixE 六格复核里裁过(fixC composite 不变)。
- **已知常量:考卷 CSV 里 torso_z 恒比 ref_torso_z 高 ~+0.11,不是 bug、不是 sim2sim 税,
  下次别再当谜查**(2026-07-09 取证定案)。人话:题库 clip 的参考躯干全程是蹲姿(z≈0.95,
  hold 帧和挥拍段都是),而策略学的是"站直打球"(站姿躯干 z≈1.07,XML stand 骨盆 1.068),
  差就是这 0.11-0.12。这是训练自己教出来的:07-05 后的等球语义只把**关节**参考换成站姿,
  anchor(躯干)参考仍指着 clip 蹲姿帧,关节奖励赢了锚位置奖励——Isaac 训练 metric
  (robot_anchor_pos_z − reference_anchor_pos_z)M2/R9a 末期同样 +0.127,MuJoCo 考卷
  +0.11~0.12 是忠实复现(两边差 ≤0.02;列错位已排除,ONNX body_names 有 assert,两边都是
  torso_Link)。开局 ~8 步的抬升 = 每集从 clip 蹲姿帧 RSI 初始化后 PD 拉回站姿(0.16 s);
  anchor_pos 终止门 0.25 容得下它,不会误杀。(v5 臂如 R9e 的训练均值差只有 +0.06,那是
  高摔率把大量"刚重置在低 z"的步数掺进了均值,不是站姿真差。)**要警惕的是它的行为学后果**:
  带 anchor_pos 缰绳训的健康臂会用手臂在世界系补拍面高度,照样接触;删缰绳臂(anchor_pos_off,
  R9a)拍子跟着躯干整体上浮 ~0.10,球从拍下穿过(R9a 反手接触率 0.15/0.00 的机理)——
  这笔账记在训练配方头上,不记在考卷头上。
- 导出**必须走 play.py 原生路**:快速导出器(standalone_onnx_export)的 donor/harvest 是
  "动作对锁定"的,现存工件都是 hopex 对——非 hopex 臂用它必死;且评估器要消费 ONNX 里的
  clip 元数据,原生导出才正确。导出后 play.py 进死循环,要 kill 进程组。
- 考卷命令骨架(阶段 1,双侧各跑一趟,锚点用**该臂自己的题库锚**):
  `--target-source venue-balls --venue-contact-fixed <锚xyz> --venue-spin-max 0
  --venue-vel-box -2.5 -1.0 -0.3 0.3 -1.0 0.3 --strike-phase-per-clip <该对相位> --steps 3000`
- 评估器的"抖动"协议需要策略噪声文件(learned_std.npy)**,原生导出不产它**——一行生成
  并 `--std` 传入:`np.save(.../exported/learned_std.npy,
  torch.load(model.pt)['model_state_dict']['std'])`(07-06 实错,FATAL 信息自带此配方)。
- **判卷链也要机制检查**:臂还在训的时候就拿任意旧 ONNX 把"导出→考卷"端到端冒烟一遍
  (07-06 判卷链**五**连坑烧掉 3 小时:导出器动作对锁定 / pod main 被挡三层 / 等 GPU 槽 /
  评估器跑错 venv / std 文件缺失——全部可以提前消掉)。

## 判卷铁律(尺子)

- **北极星 = MuJoCo 考卷回球率**,连分母一起记(可解率/锥内率)。
- **报数全表(franco 2026-07-08,当晚扩维)**:正手/反手 × 击球率/上台率 × 训练内/考卷 × 单球/连续击球,十六格齐;连续击球四格在反手修复前标"未入账";
  只报总数=漏病(反手全零就是分侧抓出来的)。
- **击球率与上台率:正式入账必须算 MuJoCo 版**(franco 2026-07-06);Isaac 训练内的
  虚拟球版只作过程监控,可以并列报但必须标注"训练内虚拟球"。两边可都算,裁决以 MuJoCo 为准。
- 跟踪三合格是诊断尺,不用它判死臂(07-06 拍面 25° 误差照样 79% 上台的教训)。

## 运维杂项(都付过学费)

- 管理实验进程禁止 `pkill/killall` 或模式匹配批量信号：它会命中 ssh 远端 shell 或相似 run。
  必须从经核对的 launch sidecar 读取 exact PID/PGID，先保全 checkpoint/contract/log，再按
  [RunPod 精确停止流程](operations/run_on_runpod.md#已登记-phase-1-实验臂的算力释放)执行。
- Isaac 退出码不可信(异常后仍 exit=0),判活/判死只看日志签名。
- pod 没装 git-lfs:commit/checkout/push 钩子会"假失败"——`-c core.hookspath=/dev/null`
  + `GIT_LFS_SKIP_SMUDGE=1`;pull 被挡的三层:本地脏文件(stash 或备份后 checkout)、
  LFS 钩子、**散落的未跟踪文件与新增文件同名**(挪走再拉)。
- **活 pod 上禁用 rsync `--delete-excluded`(07-09 付费,大额学费)**:给 pod2 增量同步 repo 时
  `--exclude logs/ --delete-excluded` 把**目的端的 logs/ 整树删了**——热启存档和六个在跑臂的
  run 目录(存档/env.yaml/tfevents)全灭,六臂随后全部僵死(R 态空转/GPU 0%/stdout 冻结),
  只能杀掉从 13000 重发(损失 1-4.5h×6)。规矩:向运行中的 pod 同步**永不带任何 delete 类旗标**;
  --exclude 的语义是"不传输",加 --delete-excluded 就成了"帮你把目的端也删了"。
- **共享检出上禁用盲 `git stash pop`(07-09 付费)**:pod 检出里躺着陈年 stash(07-06 两条);
  "stash → pull → pop"套路在"本次无可 stash 内容"时,pop 会弹出**陈年 stash**,在 train.py/
  评估器等活体文件上留冲突标记——watchdog 下一巡的复活/判卷就跑挂。规矩:pop 前先
  `git stash list` 核对条目时间;要么用显式引用(`git stash pop stash@{0}` 且确认是自己刚存的),
  要么树干净时干脆不 stash。修复=对 UU 文件 `git restore --staged --worktree`(pop 失败时
  stash 自动保留,零丢失)。
- CPU 池任务(出题/求解)记得 `OMP_NUM_THREADS=1`,不然 16 进程 × 默认线程把机器拖死。
- 长任务一律 `setsid nohup ... &` + 日志文件;监视器轮询周期 ≥8 分钟,事件去重。
- **后台任务卫生(franco 2026-07-06)**:一个目的一个监视器,目的消失立刻停;
  **后台任务的超时参数不生效,必须显式停**(07-06 实测:5 个"以为到点自灭"的等待器
  全部还活着,攒到 9 个被 franco 抓包);**每次状态汇报时清点一遍**,报"几个活着、各干什么"。
- GPU 并发数不再用“永久≤2 条/卡”的相互冲突口径；关键路径独占、广度波
  可在短测后用 3–4 条/卡，以本文上方「RTX 5090 实测算力手册」为唯一口径。

## pod 原生 Gate 3 底座:ROS2 Jazzy + 厂商 sim 直编(yikang 2026-07-10 实测跑通)

vendor AimRT+MuJoCo 全链路(Gate 3/3B 的底座)在 pod1 原生编译并冒烟通过,
distrobox 不再是唯一路径。三个脚本在 pod1 `/workspace/yikang/gate3/`:

| 脚本 | 干什么 | 时长 | 何时跑 |
| --- | --- | --- | --- |
| `bootstrap_ros2_jazzy.sh` | apt 层:ROS2 Jazzy + 全部编译/运行依赖 | ~4 min | **每次 pod 重启后必须重跑**(apt 层随重启蒸发;/workspace 持久) |
| `build_gate3.sh` | origin/main 干净 worktree(`gate3/nohope`)→ 编 vendor sim(iceoryx ON)→ 编 deploy runner → dist 补装 | 全量 ~20 min(128 核);增量分钟级 | 代码/依赖变了才需要 |
| `smoke_gate3.sh` | headless 冒烟:xvfb 下起 sim(自拉 iox-roudi)→ runner `--dry-run` 40s | ~70 s | 每次重建后 |

**判绿标准(smoke_runner.log)**:`rate=50.0Hz`(六路 /body_drive state 组帧满速)+
`sync_miss=0` + `halts=0` + ticks 持续累计 + 结尾 `[pingpong] done` 干净退出;
sim 侧 `AimRT startup completed`。冒烟验证的是 **I/O 契约**(obs→jointmap→decode→sync→
transport),不是策略稳定性——vendor sim 是显式 Euler PD,与真机(隐式 PD-in-backend)
不同侧,详见 `agi/a3_deploy_example/SIM_FIDELITY_NOTE_FOR_AGI.md` 与 MUJOCO_VALIDATION_RUNBOOK
第 0 节。

**学费清单(每条都付过)**:

1. **libacl1-dev 缺失 = AimRT 静默关 iceoryx 插件**——sim 编"成功"但没有实时通道,
   runner 端到生成器表达式才报错。装齐后必须**全清重编**(rm -rf build,缓存有鬼状态)。
2. ROS 的 setup.bash / setup_a3_env.sh 与 `set -u` 不兼容(unbound variable)。
3. vendor sim `build.sh` 默认 EXAMPLES=ON,`examples/hardware` 要拷 repo 里没有的
   vendor 预编译目录 → 加 `-DAIMRT_MUJOCO_SIM_BUILD_EXAMPLES=OFF`(a3_pingpong 在
   src/models 恒编,不受影响)。
4. dev 包依赖(bootstrap 已含):libeigen3-dev、libzmq3-dev、libmsgpack-dev、
   nlohmann-json3-dev、libacl1-dev、libncurses-dev。
5. `thirdparty/onnxruntime` 与 `thirdparty/unitree_sdk2` 是 gitignored vendor payload:
   前者 setup_a3_env.sh 自动下载(目录要先 mkdir,git 不追踪空目录);后者用本地空
   INTERFACE stub 即可链过(A3 链不用 Unitree 符号;要做 G1 需找 jiayi 拿真包)。
6. sim GUI 无条件 GLFW,pod 无显示栈 → `xvfb-run -a`(xvfb + libglx-mesa0)。
7. **runner 只认 obs 175/177/180 维**(pp_onnx_policy.hpp 硬校验)——S1 的 179 维导出
   进不了现役 C++ runner,这就是契约日 181 要解的;冒烟模型用 pod 现货 175D
   deployparity 导出(dist 内如实命名 deployparity_175_20260703.onnx,别冒充 hitter177)。
8. dist 补装两件(build_gate3.sh 第 5 步已自动化):wrapper 首选
   `config/a3_runtime_config.pingpong.yaml`(软链到实际 cfg);`libonnxruntime.so.1*`
   拷到 dist 根(wrapper 只把 SCRIPT_DIR 加进 LD_LIBRARY_PATH)。
9. ⚠ 知情项:repo 里 vendor MJCF `<option>` 无 integrator 属性 = 显式 Euler
   (SIM_FIDELITY_NOTE 建议的 implicitfast 修复未落在 tracked 拷贝);跑行为级判读前
   先与 jiayi distrobox 实跑版对配置。
10. pod2 的 IP/端口随重启变更,SSH 连不上先查 RunPod 控制台,别按旧地址重试浪费时间。

**下一步接线(Gate 3B 方向)**:hope_ws planner(colcon 编译)→ `--planner` 模式
(aimrt cfg 自动切 pingpong_ros2body 双插件)→ pp_planner_conductor.py 假球闭环;
发球生成器按场馆物理采样 + 判分器(1l 行规格)。

### 冒烟二级:推理路+发布路+驱动闭环(yikang 2026-07-10 晚,pod1;脚本 smoke2/3/5/7 同目录)

一级冒烟(dry-run)只验了状态流方向;二级把剩下三条路全部实证,**冒烟目标全部达成**:

| 路 | 证据 | 判定 |
| --- | --- | --- |
| 推理路(shadow) | ONNX 元数据全读对(clip 139/132、相位 0.470/0.333);动作有限无 NaN;ts 自由钟推进 | ✅ |
| 发布路+驱动+状态回流 | 播放模式摆腰+右臂:moving=1、fault=0、跟踪误差 0.02-0.08 rad(限 0.3)持续 >12s,倒地/瞬移后两种姿态都过 | ✅ |
| 安全逻辑 | FALL GUARD(gravZ>-0.5 持续 0.5s → PASSIVE)与 tracking fault(>0.3 rad 熔断)均真实触发且行为正确 | ✅(白送的保护逻辑验证) |
| sim reset 链路(ROS2 debug 路) | scripts/reset_sim.sh → /sim/a3/reset 关键帧瞬移生效(gravZ 瞬间 -0.97) | ✅ |
| PD_STAND 直立保持 | 接住直立后 1s 内塌掉(gravZ -0.97→-0.10) | ⚠ **不入冒烟判据**:与学费 9(显式 Euler 执行器)机制吻合,行为级留待与 jiayi 对配置后裁决;真机 backend=隐式 PD,note 预期真机反而稳 |

**学费清单增补(11-15)**:

11. 状态行语义(源码定案):`ticks/rate` 只数**已发布**命令(SHADOW 冻 0 是 by-design,ts 走的是
    shadow 自由钟);`baseZ` 在 perfect-tracking 下是**参考值(假的)**,判摔只能看 `grav`(真 IMU);
    `|act|/maxact` 是 ONNX 原始输出,`clamp` 数的是关节限位命中。
12. **⚠ 安全级发现:pingpong C++ 路径没有 ±20 raw action clip**(只有非 pingpong 的 29-DOF
    decoder 有,a3_action_decoder.cpp:12);出分布策略 raw action 实测飙到 48-52,只被
    action_scale+关节限位兜住。训练侧 07-07 已加 ±20 clip,**部署侧缺口待补**(契约日议题)。
13. 交互按键需要**真 TTY**(isatty 短路,管道被静默忽略)→ `printf 序列 | script -qec "…" /dev/null`;
    按键 `0/1` 与挥拍档位键冲突,**不带 --reference-playback 旗标时数字键不是分组键**;
    分组摆动测试用键 `5`(腰+右臂)最顺。
14. 无人值守起步用 `--warmup-sec N`(先 PD_STAND N 秒再切目标模式);但 sim 先于 runner 起 ≈10s,
    零命令期机器人已倒 → 热身从倒地开始必被 FALL GUARD 熔断。要直立起步:reset_sim.sh 瞬移
    + 连按 's'(ros2 pub 时序不可控,靠 0.5s 保护宽限窗内的那一发接住)。
15. 播放摆动的跟踪熔断阈 0.3 rad 是对的:倒地承重关节 25% 增益跟不住会正确熔断;
    满增益(--ref-gain-scale 1.0)下腰+臂不承重段 0.02-0.08 rad 干净通过。

### Gate 3 闭环底座 pod 复活(yikang 2026-07-10 深夜;jiayi 编排移植版)

**结论:全链首跑打通**——假球发布器 → 真规划器(**快解生产路径实弹**:replan 50 Hz/iter 6,
落点解 land≈2.052m 正中)→ C++ runner `--planner` → 厂商 sim(iceoryx body-drive),
`[pp engage] forehand/backhand locked … tts 传递正确`,conductor 状态机(验证站立后才按键)
在 pod 上工作,PD 站立 z=1.05 站得住(**修正 smoke7 的初判:显式 Euler 下 PD_STAND 可站,
之前塌是落地时序;fidelity note 的发散是 15200 老模型时代的事**)。

**判卷级现状 = 1/3 FAIL,失败签名=档案级已知病**:跑的是 repo 仅有的 175D 冒烟模型
(deployparity_175_20260703,07-03 导出 = **防摔修复栈之前的代际**),挥完摔(min_z 0.15)
正是 Gate 2.5 点名的"post-swing hold = 175-era killers";`--no-imu-yaw` 复跑无改变。
**要复现 jiayi 的 10/10 需要 model_17400_hitter177.onnx——repo/pod 都没有,只在 jiayi 本地,
已列交接项**。

**跑法(pod)**:`bash /workspace/yikang/gate3/pp_closedloop_pod.sh`(jiayi 原版
scripts/pp_planner_closedloop.sh + pp_planner_conductor.py 的路径移植版,sed 换前缀 + xvfb 化)。
**学费(16-18)**:16. hope_ws colcon 编译要先 `touch hope_ws/src/vrpn_mocap/COLCON_IGNORE`
(hope_bringup 声明 vrpn_mocap 运行期依赖,闭环用不到但 colcon 要它的环境文件);
17. smoke175 runtime cfg 是**工作树未跟踪文件**(src config 里),重打包必须显式
`--runtime-cfg …smoke175.yaml`,否则打包器找 17400 模型报错并**清掉 dist**;
18. conductor 依赖 sim 源编 install 的 `mujoco_sim_msgs` overlay + 持久 reset publisher
(逐次 `ros2 pub --once` 的 discovery 不可靠,jiayi 已踩过并写进 conductor 注释)。

### wandb 取模型→pod 自导 ONNX→110-D 正卷全链(yikang 2026-07-11 凌晨)

**模型交接件不用等人:jiayi 本地训练一直同步 wandb**——entity `BerkeleyPingPong`、项目
`hope_wbc`(注意不是 dongc_1 entity 下的同名项目;用 `wandb.Api().viewer.teams` 才看得到)。
run 文件里有全套 `model_*.pt`(footfix=4trve8lg、rallyfinal=63o1e21x、v2_fresh=njfc21an)。

**pod 自导 ONNX 配方(110-D 世代,已验证)**:下载 .pt → 重建
`logs/rsl_rl/<experiment>/<run>/` 目录结构 → hitter 分支代码 + `play.py
task=HOPEPingPongHitterPure algo=ppo num_envs=2 checkpoint=… motion_file=/workspace/shared/motions/hope_{forehand,backhand}_hopex.npz
algo.runner.empirical_normalization=false headless=true`。学费(19-22):

19. **`empirical_normalization` 是死旗标**(ppo.yaml 注释白纸黑字):jiayi 环境 rsl_rl 3.1.2
    的 shim 拦掉它,全谱系 checkpoint 无归一化状态、ONNX 全部 obs→Gemm 直连;pod 老版
    rsl_rl 该旗标还活着,不带 `algo.runner.empirical_normalization=false` 加载必炸
    KeyError obs_norm_state_dict。
20. motion_file 错路径不会 fail-loud,会**静默掉进 wandb registry 兜底**(拉到错的动作对=
    烤错元数据);正确路径=/workspace/shared/motions/(jiayi pod 克隆的 preprocessed 目录是空的)。
21. 导出后回放循环崩溃(get_observations 返回 tuple,老版 IsaacLab API 差异)**不伤产物**——
    "[INFO] Exported ONNX" 打印且元数据齐全即有效;验收 = onnx.load 查
    `hitter_pure_pos_range_per_clip`/`clip_seg_lengths`/`clip_strike_phases`。
22. **考卷必须配世代**:legacy pp_planner_closedloop(177 时代)对 110-D 模型 engage 恒 0
    ——规划器 x_hit 偏移 0.67 vs 110-D 元数据拍面 0.51(x 带宽为零);110-D 正卷 =
    `pp_gate3_rally.sh`(x_hit:=1.03 等参数内置,含 rcl 参数顺序防坑注释)+
    `hope_planner.hitter_pure.yaml`(实机)。

**13200_footfix08 的 Gate 3 首读数(此前文档标 PENDING)**:pod 移植版 pp_gate3_rally,
13 PASS / 7 FAIL,3 发接回 1、摔 1、站位漂移 1.39m;PASS 侧=基建+站位语义全干净
(baked reach-x +0.510 与元数据严丝合缝、station 误差 0.04/0.032m、0/483 tick 倾倒、
sync_miss=0);FAIL 侧=|action|max 28.3、**击球朝向 26.2°/挥后朝向漂移/基速 p90 超标**
——正是该世代文档已知的 heading-recovery 开放问题(RallyFinalV2 立项原因),非基建问题。
工件:pod /workspace/yikang/gate3/rally_run.log、/tmp/pp_rally_report.json;
跑法=`bash /workspace/yikang/gate3/pp_gate3_rally_pod.sh`。
