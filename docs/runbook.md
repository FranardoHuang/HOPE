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
| 观测扩列脚本 | /workspace/shared/pad_obs_cols.py(末尾追加)/ pad_obs_cols_insert.py(按位插列) | 存档跨观测维热启用;jiayi 的 177 是**插在第 167 列**不是末尾 |

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

```
model_13599.pt ──play.py 原生导出(isaac venv,占一个 GPU 槽 ~4 分钟)──> exported/policy.onnx
    ──mujoco_eval_onnx.py(mjeval venv,纯 CPU)──> 按侧考卷分数
```

- **终版存档名是 model_13599 不是 13600**(rsl_rl 末迭代 0 起数;监视器等 13600 会永远等)。
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
- **击球率与上台率:正式入账必须算 MuJoCo 版**(franco 2026-07-06);Isaac 训练内的
  虚拟球版只作过程监控,可以并列报但必须标注"训练内虚拟球"。两边可都算,裁决以 MuJoCo 为准。
- 跟踪三合格是诊断尺,不用它判死臂(07-06 拍面 25° 误差照样 79% 上台的教训)。

## 运维杂项(都付过学费)

- pkill 的模式会匹配到 ssh 远端 shell 自己的命令行 → 用方括号断字:`pkill -f "run_name=s1[_]"`。
- Isaac 退出码不可信(异常后仍 exit=0),判活/判死只看日志签名。
- pod 没装 git-lfs:commit/checkout/push 钩子会"假失败"——`-c core.hookspath=/dev/null`
  + `GIT_LFS_SKIP_SMUDGE=1`;pull 被挡的三层:本地脏文件(stash 或备份后 checkout)、
  LFS 钩子、**散落的未跟踪文件与新增文件同名**(挪走再拉)。
- CPU 池任务(出题/求解)记得 `OMP_NUM_THREADS=1`,不然 16 进程 × 默认线程把机器拖死。
- 长任务一律 `setsid nohup ... &` + 日志文件;监视器轮询周期 ≥8 分钟,事件去重。
- **后台任务卫生(franco 2026-07-06)**:一个目的一个监视器,目的消失立刻停;
  **后台任务的超时参数不生效,必须显式停**(07-06 实测:5 个"以为到点自灭"的等待器
  全部还活着,攒到 9 个被 franco 抓包);**每次状态汇报时清点一遍**,报"几个活着、各干什么"。
- GPU 槽位规则:≤2 真任务/卡;判断"空槽"用 `nvidia-smi --query-compute-apps`
  按卡计数(--id 过滤是好用的;07-06 误判其实是僵尸进程,见核对单第 1 条)。
