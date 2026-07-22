# 分支修复审计 2026-07-22 — jiayi/yikang 分支上的修复都在哪、搬不搬、为什么

- 责任人：Franco(拍板);执行者:Claude
- 背景:Franco 抱怨"jiayi 分支很多修复长期没 track 到"。本文是 2026-07-22 全面审计的
  裁决账本:每个候选修复给出【立即搬 / 需消融 / 只搬思想 / 不搬】四选一 + 原因 + 去向。
  以后每次审计都续写同名新日期文件,配合[三人分支追踪看板](#三人分支追踪看板)防再失踪。
- 站立异响外部复核 PDF 已入库:
  [agibot_a3_standing_chatter_evidence_ranked_review_2026-07-22.pdf](agibot_a3_standing_chatter_evidence_ranked_review_2026-07-22.pdf)
  (E1 级:站立异响首查部署侧高增益静态站立/两套站立分支/现场二进制;训练侧 Reward 修正
  不是第一反应,但 action-rate 偏弱与 plant 覆盖缺口成立——由
  [抖动-地面-脚部消融波](../experiments/2026-07/EXP-P1-CHATTER-GROUND-FOOT-WAVE.md)消融)。

## 裁决表:jiayi(dongc1,分支 hitterobs / HitterV11)

| 候选 | 出处 | 裁决 | 状态/去向 | 原因(人话) |
| --- | --- | --- | --- | --- |
| YAML 写 null 删不掉继承参数 | 8ee2e82a | 立即搬 | ✅ 07-22 已落盘 main 血统 train.py(`_REWARD_NULL_REMOVABLE_PARAMS` 表 + 幂等删参;test_reward_flags_overrides.py 7 项新测) | 子任务想删父层 reward 参数只有 `key: null` 一种写法,旧覆盖层把 None 当"没写"跳过 → 永远删不掉,换函数签名的后继任务直接炸 |
| 精确续训包(课程计数器不归零 + Ctrl-C 安全存档) | 9f684ae5 | 立即搬 | ✅ 07-22 已落盘 `my_on_policy_runner.py`(`hope_exact_resume_state` schema2;test_exact_resume_state.py) | 跑 2万-2.5万 iter 长训的刚需:续训时 `common_step_counter` 归零会把课程表拉回起点 |
| checkpoint 归一化 2x2 真值表预检 | yikang 635be7cf(cherry 自 yikang;jiayi 分支也带) | 立即搬 | ✅ 07-22 已落盘新模块 `checkpoint_normalization_preflight.py` + train.py Kit 前预检(test_checkpoint_normalization_preflight.py) | 老路径等 Kit 起完才炸费解 KeyError,白烧几分钟 GPU 启动费;现在 CPU 上先炸+给单条修复命令 |
| 被动阻尼折进训练 kd | jiayi 07-09 起(A/B 收据:挥拍后基座漂移均值 0.066→0.084、峰值 0.121→0.464,复现~90% AGI 台架跟随放大) | 需消融 | 本波 kdpassive 臂;`task.plant.passive_damping_fold` 键名已冻结,**源码未接线**,kdpassive_contract 闸门锁死渲染 | 有 A/B 收据的真实信号,但改的是 plant 语义,必须走消融+合同块指纹,不能直接抄 |
| AdamW + 噪声 std 上下限 | 0d0c777b(HUGWBC 出处) | 需消融(暂缓) | 不进本波;记档待将来单独消融 | 优化器换代影响全谱系可比性,当前优先级低于速度泛化与本波 |
| tanh 有界 qdes 合同(gain loss fix) | 1da21c90 | 只搬思想 | 已写成设计规范(见下"tanh 规范");不搬实现 | Franco 拍板:动作超限用 penalty 不上 tanh 壳,维持 qdes_clamp+qbar barrier(memory: feedback-tanh-vs-penalty) |
| V15 把 action_rate 降到 -0.01 + 投影约束管平滑 | hitterobs V15 | 反向情报,记录不搬 | 已写进本波 ar 臂判读 caveat 与预注册 | 与我们"加大 action_rate"方向相反;正反两说都记录,让剂量曲线自己说话 |
| V15 模仿权重翻倍到 2.0("挥拍老师只占稠密收入 1/5 拍速塌了") | hitterobs V15 | 反向情报,记录不搬 | 记档 | 与 Franco 澄清的第 8 条方向不同(我们先做惩罚减负 penlight,不加大模仿/击球权重) |

**tanh 规范(只搬思想的落地文字)**:若将来任何人给动作通道上非线性壳(tanh/软饱和),
规范写死:**默认姿态处的局部斜率必须 = 原 action scale**,否则等效增益静默变化,
qdes 限位、判分器与部署解码全部错位。exact 语义以合入 main 的实现+单测为机器真源。

## 裁决表:yikang(Catrunaround,分支 yikang-stationary-v2-0722 / yikang-v14-legfreeze)

| 候选/判决 | 裁决 | 状态/去向 | 原因(人话) |
| --- | --- | --- | --- |
| stationary-v2 正手线(s16 约 300 迭代 pass 15%+;慢球臂摔倒率 10.3%→2.1%) | 继续,别停 | 建议设固定考卷里程碑判断点,防"感觉不错"无限续 | 正手起飞是真信号 |
| stationary-v2 反手判"结构性死亡"(拍面差 44-57°、门限 15°、134 帧全 0%) | **伪影坐实,死刑撤销**(07-22 晚复扫执行完毕;终审=投影修复后 stationary 重扫,进行中) | 复扫结果见下"全库扫描矩阵执行结果" | 钉根删 yaw 15.7° + 剖面吃掉 rally_yaw 40° ≈ 55.7°,正是 44-57° 拍面差;钉根复现件换 generic 剖面即回 100% |
| v14-legfreeze(锁腿热启动) | 按其纪律收线 | 首轮 reject(击球 1-3%);ft4 今早刚发未归档;连 reject 两轮即收 | 与 main 的锁腿消融(R9)证据方向一致 |
| 归一化预检(635be7cf 家族) | 立即搬 | ✅ 已随 jiayi 表第 3 行落盘 | 同上 |

### 反手拍面死刑复核(2026-07-22 三路对抗复核裁决:应复扫)

三路只读侦查(拍面符号/来球速度锥/题库专属性)+ 逐路对抗复核 + 汇总,结论:

1. **两条嫌疑通路被证据排除**:
   - 符号/自动定向不是凶手:yikang 的扫描工具(分支 `gen_stage1_questions.py`)在面门前
     做半球对齐(`:1284`,报告误差 = min(θ,180°−θ) ≤90°,对参考法线翻号严格不变),
     所以"漏乘 mount_normal_sign=-1"(工具确实全文不读该参数)对误差数值零影响;纯翻号
     错误的签名是 ~0° 而不是 44-57°;phase-scan 合法性经 `orient_normal` 内部翻法线,
     同样符号不敏感。
   - 题库复用不是凶手:stationary 剖面强制加载被扫反手投影 NPZ 自己的投影合同
     (`:1118-1123`),锚与逐帧拍点/法线都来自该 NPZ 自身 FK——v1"题库触点必须按当前
     动作生成"的教训在 v2 已落实。
2. **找到具体伪影机制(投影/剖面约定)**:
   - `project_stationary_upperbody_motion.py:709-724` 把 pelvis 根位姿**无条件替换为
     reset 单位姿态**,把反手源动作里 ~40-60° 的转体整个删掉——被删的骨盆旋转 1:1
     旋转世界拍面;判死锚帧参考法线 [-0.375,0.764,-0.526](指向后-左-下=拍背方向)正是
     删大转体后的典型签名(raw 角 ~123-136° 折叠成 44-57°)。
   - stationary 剖面强制 rally_yaw=0(`:461-470`),取消了反手原生 -40° 对角线归一化,
     而来球锥是正反手共用的直线锥。
   - 仓库内有现成 100%→0% A/B 反证:同一 clip(hope_backhand_v4rg_cal)投影前在 generic
     剖面两档速度全 100%(strike_annotations.yaml:326-338),投影后 stationary 扫描
     134 帧全 0%;分支自己对姊妹动作族也自认过同病根("投影到 reset-root 约定后拍面
     系统性错向",run_stationary_upperbody_v2.md:64-71)。
   - 次要保守偏置:扫描大概率用默认速度档 2-5 m/s 而非场馆档 ~1.0-2.5(前科:96% vs
     17%);phase-scan 把拍面钉死为该帧 FK 法线(0° 自由度),严于训练 15° 门。
3. **三步复扫命令(零改码,pod 侧)**:
   - 步 1 判定性诊断:量源 NPZ f45 帧 pelvis 四元数与单位姿态的夹角
     (`np.degrees(2*arccos(|w|))`);≈40-60° ⇒ 钉根删转体坐实。
   - 步 2 几何 A/B:同一投影 NPZ 换 generic 剖面重扫
     `python scripts/gen_stage1_questions.py --phase-scan --clip hope_backhand_v4rg_cal:<投影BH>.npz --profile generic --seed 0`
     (generic 会把 -40° 归一化施加回拍面/挥速向量;分数从 0% 显著回升 ⇒ 伪影坐实)。
   - 步 3 速度档对照:stationary 剖面 `--speed-range 1.0 2.5` 重扫。
   - 判读:步 1 大角度且步 2 回升 ⇒ 死刑撤销,改投影/剖面约定后重评;步 1 小角度且
     步 2/3 仍全 0 ⇒ 死刑升级为"成立"(该资产在锁腿合同下真不可行,需换反手参考动作)。
   - 注意:即便复扫仍低分,死刑的准确表述也是"**此投影约定+此 clip** 不可行",不是
     "反手结构性死亡"。

### 全动作库扫描与修复方案(Franco 07-22 拍板:不止反手,全库都扫、都要修好)

反手三步复扫只是判定性第一格;同一个投影钉根伪影可能污染**所有**走过投影/剖面约定的
判决。方案(pod 已恢复,排位在判卷欠账之后,pod2 空卡可跑——phase-scan 是 CPU torch
物理,不占训练 GPU):

1. **先修根因,再全量重扫**(否则扫了也是废数据):
   - 投影工具(yikang 分支 `project_stationary_upperbody_motion.py:709-724`)钉根时
     **保留源骨盆 yaw**(或把 rally_yaw 语义带进 station anchor);投影合同必须记录
     "被删根旋转角",超过阈值(建议 10°)fail-loud 拒绝静默投影。
   - stationary 剖面的 rally_yaw 处理与 clip 注册表归一化对齐(现在强制 =0,把反手
     -40° 对角线几何整个吃掉)。
   - 修好后所有投影动作**重投影**(照 r2/motions_r2 流程,投影工具 SHA 会变,合同
     对账自动强制重做)。
2. **扫描矩阵**(每 clip 一行落账,登记到本文档系列的下一期):
   - 范围:main 注册表(cfg/strike_annotations.yaml,37 条)里全部现役 `_cal` 族
     (v4rg/v5rg/swing/oblique/v5syn/v4rgsyn/swingsyn/v5topp 正反手)+ yikang
     stationary-v2 的全部投影变体(BH 134 帧、v12fix_comv 族 107/109 帧等)。
   - 每 clip 跑:源几何 × generic 剖面(rally_yaw 归一化生效)、投影后 × stationary
     剖面(若有投影版),各 × 两个速度档(场馆档 `--speed-range 1.0 2.5` +
     默认档 2.0-5.0),记录 train_phase_candidates 与 pass band。
   - 工具零改码:`gen_stage1_questions.py --phase-scan --clip <注册表键>:<npz> --profile
     generic|stationary-upperbody-v1 --seed 0 [--speed-range 1.0 2.5]`。
3. **判读纪律**(防"感觉不行"式死刑再次发生):
   - 源几何高分、投影后 0% ⇒ 投影伪影,修约定重投影重扫,原判决作废;
   - 两边都低 ⇒ 动作真不可行,换参考动作,死刑成立且写明"在此合同下";
   - 任何"投影后 0%"判决,没有源几何 A/B 对照的一律只算"存疑",不得进决策。
4. 责任归属(Franco 07-22 改判:谁在看着解决就谁做):修复执行 Claude——基于
   origin/yikang-stationary-v2-0722 开分支改投影工具+重投影+重扫,推新分支不动他的分支,
   投影约定变更知会 yikang;修复合入前,stationary-v2 家族所有"结构性排除"判决全部降级为存疑。

### 全库扫描矩阵执行结果(07-22 晚,已执行;日志 pod2 /workspace/codexschema/motion_scan_20260722/)

**反手裁决:伪影坐实,死刑撤销。** 证据:①源 NPZ f45 骨盆相对单位姿态总转角 **24.7°**
(yaw 15.7°)——机制成立,幅度修正(原预测 40-60° 偏高);②判定性 A/B:按投影工具同法做的
钉根复现件换 generic 剖面(把 -40° 归一化还给它),f40-43 两档速度**全 100%**,对照死刑扫描
134 帧全 0%;③数值自洽:被删 yaw 15.7° + stationary 剖面强制 rally_yaw=0 吃掉的 40° ≈ 55.7°,
正落在死刑报告 44-57° 拍面差带内——两刀合起来就是那个拍面差。诚实边界:yikang 原投影 NPZ 与
stationary 合同不在 pod2,步 2 用逐行同法复现件、步 3 blocked;终审以投影修复(进行中,执行
Claude)后的 stationary 重扫为准。**新发现:投影几何最优锚窗从源件 f44-46 前移到 f40-43,
投影资产锚帧必须重选,不能沿用源件锚。**

**全库源几何:16 条现役 clip × 2 速度档零失败扫完,没有第二个死刑候选**(每条都有 100% pass 帧)。
观察名单:bh_v5rg_cal 注册帧只有 38-42%(已知支撑脚滑移腿病,最佳帧在 f20-22);
fh_v5rg/fh_v5syn/fh_v5topp band 仅 3-4 帧(锚帧容错小);bh_swing/fh_oblique 默认速度档注册帧
50-71%。建议下次登记时把这几条的 train_phase_candidates 对齐扫描表(全表见 pod2 日志 32 条)。

## PDF 五缺陷对账(部署侧四欠账 2026-07-22 已修)

| PDF 缺陷 | 排名 | 状态 |
| --- | --- | --- |
| 两套站立路径增益接口不一致(--stand-kp 调不到 planner static 官方高增益分支) | 2 | ✅ 07-22 修(只改 agi/):启动打印两条路径 Kp/Kd 来源与数值(STAND GAIN SOURCES 横幅);static handoff ENTER/EXIT 带时间戳进日志+CSV `planner_static` 列;新旗标 `--planner-static-gain-scale`(默认 1.0=逐字节不变,memcmp 验证过) |
| 部署二进制身份未知 | 3 | ✅ 07-22 修:CMake configure 期注入 git SHA+脏标+UTC,每次运行第一行打印 build fingerprint |
| 遥测缺力矩/电流 | 6 | ✅ 部分修:vendor SDK(joint_msgs State.msg)只有 effort(实测力矩),**无电流无温度**;obs/trace CSV 尾部纯增 31 列 `tau_*` + `planner_static` 列,老列不动。电流/温度记"SDK 不暴露" |
| action-rate 偏弱 | — | → 本波 ar02/ar05/ar10 消融(mjlab 外部先验:平滑全押 action_rate -0.1,与现役同值;"比 -0.1 更重未必有益"由三臂裁决) |
| effort 随机化缺失 | — | 未修,记欠账(plant 覆盖轴之一,本波先买摩擦/地形两轴) |
| plant 地面覆盖缺口(摩擦/不平整) | — | → 07-22 已接线(train.py task.plant 五新键 + schema-3 ground_plant 合同块指纹)+ 本波 grip/rough 消融 |

**部署侧修复验证的如实声明**:三项修复默认行为逐字节不变(scale=1.0 走原赋值分支,
memcmp 断言一致;CSV 只在尾部增列;指纹只加打印)。但 portable Release **未能在本机
整体编译验证**(macOS 无 Linux vendor 栈):已做 CMake 指纹块独立实跑 + 改动块逐字
摘录 `-Wall -Wextra -Werror` 编译零告警 + 运行断言;**下次在 Linux 构建机必须跑一遍
docs/operations/build_and_test.md 的 portable Release + run_tests 收尾**。

## 三人分支追踪看板

自动生成:`python scripts/branch_dashboard.py --min-ahead 1`(只读;先 `git fetch --prune`)。
纪律:领先 main 的每个提交,要么搬进 main、要么在本系列文档记"不搬 + 原因",不允许失踪。

2026-07-22 快照(基准 origin/main @ 66583b9b)——只列头部,全量看板跑脚本:

| 人 | 有未合并提交的分支 | 头部分支(领先 main) |
| --- | --- | --- |
| Franco | 10 条 | planner-policy-main-integration(6)、first-reset-probe-telemetry(5)、planner-policy-demo-fixes(4) |
| jiayi(dongc1) | 6 条 | hitterobs(40,07-22 还在动)、HitterV11(36)、hitter(28) |
| yikang(Catrunaround) | 20 条 | cat_stable(167!,07-21)、rally-v11-topp-prestrike(51)、standhit-0714(50)、v14-legfreeze(41,07-22)、stationary-v2-0722(30,07-22) |
| 其他 | 1 条 | jeremy/pingpong-virtualball-onestep-20260722(1,07-22 新出现——下次审计要看) |

优先追踪队列(下次审计先看):① jiayi hitterobs(40 条领先,V15 家族还在长);
② yikang cat_stable(167 条领先,从没审过);③ yikang stationary-v2-0722(反手复扫
结果);④ jeremy 新分支(07-22 首次出现)。jiayi 分支禁止整体 merge(NOW 已判),
只选择性重做。
