# 跑批作战手册(常驻;发射/判卷/运维的全部操作知识,人话)

一句话:**这本手册让任何人(或下一个 claude)能安全地发射训练臂、判卷、不踩已知的坑**。
规划看 [NOW.md](NOW.md),历史与制度看 [TIMELINE.md](TIMELINE.md),动作管线看
[motion_pipeline.md](motion_pipeline.md)。每条坑都带日期,来源可查。

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
10. run_name 当场进 NOW 队列表,发射才算完成。

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
  显式 `--strike-phase-per-clip`;mjeval venv 下设 `HOPE_STAGE1_QB=<stage1_question_bank.py 路径>`
  (绕过 isaaclab 包链);分母报表(kept/asked/锥内比例/难度中位)开头自动打印,入账连它一起抄。
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

- pkill 的模式会匹配到 ssh 远端 shell 自己的命令行 → 用方括号断字:`pkill -f "run_name=s1[_]"`。
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
- GPU 槽位规则:≤2 真任务/卡;判断"空槽"用 `nvidia-smi --query-compute-apps`
  按卡计数(--id 过滤是好用的;07-06 误判其实是僵尸进程,见核对单第 1 条)。
