# judge_results_20260722 — 已收口终档首批判卷(11 臂,诊断档,成绩暂不可解读)

> **2026-07-22 终审更新:Round 6(governor 协议重判)已完成并取代本文件前几轮的全部
> MuJoCo 数字 —— 见文末"Round 6"一节。** 上面的 round3 表保留仅作伪影档案。

- 日期:2026-07-22;执行者:Claude(后台判卷 agent);裁决待 Franco
- 性质:**诊断档,不是正式 BankExam 分**。两个原因:①这批 checkpoint 的
  `training_contract_lineage_exact` 全是 0(res1 续训/warm-start 出身),考卷走
  `--allow-inexact-contract` 诊断通道,报告头如实印 `evaluation_contract_exact=false`;
  ②发现疑似评估器对 task-revision 代际的语义欠账(见"必须先裁决的疑点"),裁决前
  **全部数字不得用于臂间排名或机制结论**。
- 判卷链修通记录:judge.sh 两个炸点都是旗标级修复,未改源码——
  ①题库名推不出同源考卷 → `--exam-bank .../schema3_exam_bank_rebind_v1/s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz`;
  ②导出合同 mismatch `{'episode_length_s': (16.0, 10.0)}` → `--export-extra "task.env.episode_length_s=16.0"`(训练真值 16.0,逐 run 对过 env.yaml 与 training_contract.json)。
  成功拼法与串行队列脚本:pod 上 `/workspace/codexschema/judge_fix_round3_pod{1,2}.sh`,
  日志 `/workspace/codexschema/judge_fix_20260722_pod{1,2}.log`。
- 报告原件:各 run 目录 `judge/judge_report_model_*_20260722_*.md`(两 pod)。

## 成绩表(11/11 rc=0;全臂 22 格接触率=回球率=0)

有区分度的只有"活到击球帧数"(每侧 各188 题左右)与反手拍面/拍位误差:

| 目标(checkpoint) | 正手活到击球 (ns=0/0.05) | 反手活到击球 (ns=0/0.05) | 反手拍面误差 | 反手拍位误差 |
|---|---|---|---|---|
| v_qbar(10700,唯一自然终档) | 0, 0 | 30, 27 | ~15° | ~0.33 m |
| v_c_s1(13100) | 0, 0 | 150, 151 | ~32° | ~0.30 m |
| v_c_s2(13000) | 0, 0 | 178, 179 | ~33° | ~0.29 m |
| v_c_s3(13100) | 0, 0 | 131, 139 | ~35° | ~0.28 m |
| v_yaw_r2(13900) | 0, 0 | 159, 162 | ~37° | ~0.27 m |
| v_p05(13000) | 0, 0 | 188, 188 | ~35° | ~0.27 m |
| w_c_s0_res1_r5(13900) | 0, 0 | 0, 0 | – | – |
| w_c_s1_res1_r3(11200) | 0, 0 | 86, 67 | ~40° | ~0.24 m |
| w_p08_res1_r2(12900) | 0, 0 | 1, 0 | ~37° | ~0.22 m |
| w_yaw_res1_r2(12000) | 0, 0 | 1, 1 | ~53-58° | ~0.28 m |
| w_p05_res1_r4(11800) | 0, 0 | 142, 140 | ~45° | ~0.28 m |

## 裁决结果(07-22 晚取证闭环:评估器伪影坐实,本批 MuJoCo 分数作废)

大假设成立,但方向与初判相反:**不是策略等揭题,是策略比参考快**。证据链:
- 742 次尝试 **100% 终止于 `ee_body_pos`**(腕/踝参考相对 z 差>0.25m);anchor_pos/anchor_ori/物理摔倒全为 0——**没有一次摔倒**。
- 正手 366/366 全部死在第 5 步(0.1s,一步不差):策略开题即全速挥拍(0.1s 内拍速 1.3→6.0 m/s,
  这是 task-revision 训练态 governor 把参考奴役到"中位 0.5s 打完"教出来的爆发节奏),而评估器参考
  按 1x 原生慢放(正手 1.32s 才到击球帧)——机器人冲到参考前面,守卫收题。
- 反手死在 38-50 步、集中在原生击球帧 ±2 步:反手原生 0.90s 恰在训练混合 fast_deploy 上沿,协议误差小,
  故能活到击球帧(但判分帧错位,打不准)。
- 代际根因:planner_revision 07-17 才进库(bdc57ef1);评估器逐字实现的是老代际协议
  (tts=剪辑相位倒数、参考 1x 播放),7-13 老代际判卷正常正是协议匹配的自然结果。
  评估器全文 0 处 planner_revision;该代际 actor 的 tts 语义已是任务 deadline 倒数(hope_commands.py:3078)。

**处置(建议已采纳)**:①本批全部 MuJoCo 分数作废(0 接触=评估器伪影,不是策略能力);
②诊断档 B 立即跑:用评估器现成 CLI 阈值旋钮把三个 reference 相对阈值调大、只留物理摔倒+超时,
回答"物理摔不摔+原生击球帧附近的二次挥拍会不会命中"(仍非正式分);③正式修复 A 立案:
给 mujoco_eval_onnx.py 按 ONNX metadata 门控补 phase-governor 揭题协议(~150-200 行,三份实现
逐位对拍:torch commands.py:1239 / C++ pp_phase_governor.hpp / 新 numpy 版;老代际判卷逐字节不变),
落地后全量重判。选项 C(isaac_bank_exam 原生支持该代际)作兜底,不替代 A。

## Round 5(诊断档 B:抬训练守卫)——伪影之下还有真问题(07-22 夜)

拼法:round3 基础上 `--exam-extra` 追加 `--ee-term-z 10.0 --anchor-term-z 10.0 --anchor-term-ori 10.0`
(旗标名自 mujoco_eval_onnx.py:4295-4306;报告头带 guard-override 警告,仍诊断档)。11/11 rc=0,
报告与 round4 并存于各 run judge/ 目录(时间戳 1212xx-1217xx)。

1. **守卫伪影坐实**:抬掉 reference 相对守卫后,反手活到击球帧 0→**188/188**(全臂)。
2. **但真相更深:100% 物理摔倒**(11 臂 × 双噪声档 × 371 题全部 fall_root_z/fall_tilt,0 次
   活到 16 秒超时)。修正 round4 的"零摔倒"解读:守卫只是把"即将摔倒"提前判了 0.4-0.7 秒,
   收题时还站着 ≠ 能站住。
3. **零接触维持**:反手到帧但拍位差 0.27-0.42 m,不存在被守卫掩盖的二次挥拍命中;正手
   摔倒前依旧到不了击球帧(除 v_c_s1 零星 1-4 题,拍面翻 ~110-120°)。
4. **分层信息(方向参考)**:V 臂 0.86-0.95 s 摔、以 fall_tilt(倾倒)为主;W 臂 1.10-1.19 s
   摔、以 fall_root_z(高度塌)为主——"W 比 V 稳"的排序在 MuJoCo 也成立,但只多撑 0.25 s。
5. **解读边界(为什么仍不是终审)**:本轮考卷仍按老代际协议出题(参考 1x 慢放、tts=剪辑
   相位倒数)——策略的 tts obs 语义整场错位,在错误协议下冲刺、失衡、miss,不能直接判
   "这批臂不会打";**终审 = 选项 A(governor 协议)重判,进行中**。另 exam.log 警告
   explicit-PD/零阻尼/零摩擦 plant 与训练合同差异在诊断豁免清单里,绝对数值解读须记入。
6. 若 A 协议重判后仍全摔/零接触,则升级为正式的 sim-to-sim gap 证据(带 lineage-inexact 与
   plant-差异双 caveat);Isaac 训练内 virtual outcome(W 完成率 ~95%)与 MuJoCo 的落差将是
   下一个 P0 调查对象。

## 必须先裁决的疑点(不是链路机械故障)

1. **机器人没摔,但每题只活约 26 控制步(~0.5 s)就被收题**:exam.log base_pitch 均值
   ~2°,371/371 early、0 超时。收题来源已定位到评估器的 **reference 相对终止**
   (`mujoco_eval_onnx.py::check_terminations`:|参考躯干z−实际躯干z|>0.25、
   proj-grav 差>0.8、任一 ee 相对z差>0.25)——即"没跟上参考剪辑"就收题,与摔倒无关、
   也不是 0.5s deadline guard。
2. **对照事实**:7 月 13 日老代际臂用同一条判卷链跑出正常卷(每题 ~140 步、接触率 1.0、
   回球率 0.5-1.0)。今天这批全部是 task-revision 代际(planner_revision,initial_tts
   混合 ~0.5s 为主),judge.sh 07-21 起会把 planner_revision 从 env.yaml 回搬进导出。
3. **正反手不对称恰与相位算术吻合**:反手 strike_phase=0.338 的击球帧在 ~26 步收题窗内
   (大多数臂反手能"活到击球"但打不准),正手 0.471 的击球帧在收题之后(11 臂正手全部
   0 个活到击球)。疑似评估器的参考时钟/hold 协议与该代际训练时的 task-revision 揭题
   协议对不上:策略在等揭题、参考剪辑已开挥 → 0.5 s 内 reference 相对偏差超限 → 收题。
4. 反手拍位 0.22-0.33 m、拍面 32-58° 的误差可能同源(半秒内根本挥不到位),
   同样裁决前不可解读。

## 裁决选项(建议顺序)

1. 先取证:从任一报告的 exam.log 抽 term reasons 分布(anchor_pos/anchor_ori/ee_body_pos
   哪个先触发)+ 逐题第一帧 obs 的 TTS/hold 语义 dump,和 Isaac 训练态逐项对。
2. 若坐实"评估器未实现该代际揭题协议":给 mujoco_eval_onnx.py 补 task-revision 代际的
   hold/揭题语义(或判卷时显式禁用 reference 相对终止、只留物理摔倒终止的诊断档),
   全部 11 臂重判;这是判分器欠账,不是臂的行为结论。
3. 若排除评估器问题:结论转向"该代际在 MuJoCo 下确实无法迁移",升级为最高优先的
   sim-to-sim gap 证据(需带 lineage-inexact caveat)。

在裁决完成前,本文件的数字只证明"判卷链已修通、可以出报告",不证明任何臂的行为水平。

## Round 6(正式修复 A:governor 揭题协议重判)—— 终审真分(07-22 晚)

**一句话:评估器学会了 task-revision 代际的揭题协议后,11 臂全部出真分 —— 反手 11/11
臂接触率 100%,V 臂反手回球率最高 100%;正手分裂成两种真实失败模式(V 爆发离包络、
W 到帧但拍面反 102-144°),与 Isaac 训练侧"V摔但回球好 / W稳但不回球"完全吻合。**

### 协议修复(选项 A)落地内容

- main 提交:`9d22dc38`(协议主体,merge `3bacbfa1`)+ `a5dbfdfb`(参考 jv 缩放补丁);
  分支 `Franco_codex/mujoco-eval-governor-protocol-20260722`。
- 门控:ONNX metadata `planner_task_revision`(exporter 早已烘入,评估器现在消费)。
  有 = 强制 governor 协议(双钥匙,同 C++ RequirePpPlannerTaskRevisionDoubleKey 语义);
  无 = 老代际路径逐字节不变;畸形 = fail-closed 拒判,绝不静默回退老时钟。
- 新代际行为(全部镜像训练态,host 三方对拍全绿):
  - 参考时钟:开题 phase=0/rate=0,numpy 双精度 governor(镜像 commands.py
    `_advance_planner_phase`)推帧,恰在确定性 initial_tts deadline 到点走到击球帧;
    击球后按训练同款公式平滑接回原生跟随播放直到 clip 结束。
  - actor 的 time_to_strike obs = 任务 deadline 倒数、跟随段钉 0(hope_commands.py
    schema-4 语义);击球判分帧 = governor 到达帧 + 训练同款一次性 latch。
  - 参考关节速度 obs 按 governed 帧速缩放(begin=0 静止揭题,冲刺 ~2.6x)——
    第一轮 smoke 抓出的欠账:漏这一条时反手拍位差 0.157m,补上后 0.033m。
  - legacy hold 时钟由 initial_tts 取代(题库 hold_steps 留账不睡);
    `--planner-initial-tts` 旗标,缺省 0.5s = 训练混合的部署基线点质量,报告头如实印。
- 对拍:C++ golden trace 向量(逐字校验自 test_pp_phase_governor.cpp,1e-12)、python
  参考 planner_revision.advance_phase 逐 tick(1e-12)、torch 版 commands.py **真源码抽出**
  float64 帧域逐位 ==;metadata 拒收矩阵;老代际不变结构锁
  (tests/test_eval_phase_governor_parity.py,40 用例全绿)。

### 重判拼法(11/11 rc=0)

- 专用判卷 checkout `/workspace/codexschema/nohope_judge_20260722`(两 pod,本地 clone
  提速 + fetch main detached;assets/agibot_a3 为 gitignored 工件,从钉死 checkout 拷入;
  三个钉死 checkout 未动)。队列脚本 `/workspace/codexschema/judge_governor_pod{1,2}.sh`,
  日志 `judge_governor_20260722_pod{1,2}.log`;round3 同款旗标(--exam-bank 同源考卷 +
  --export-extra episode_length_s=16.0 + --allow-inexact-contract),串行 + nice +
  JUDGE_LOCK_WAIT_S=10800。checkpoint 全部与 round3 目标一致(这些 run 已无更新档)。
- 报告原件:各 run `judge/judge_report_model_*_20260722_13*.md`(报告头带
  `planner governor` 行;governor 协议参数落 summary JSON `planner_governor` 块)。

### 成绩表(接触率/回球率 = 全尝试分母;双噪声档 ns=0 / ns=0.05)

| 臂(ckpt) | 正手到帧 | 反手到帧 | 反手接触/尝试 | 反手回球/尝试 | 反手拍位 | 反手拍面 | 正手拍面 | 存活步数 |
|---|---|---|---|---|---|---|---|---|
| v_qbar(10700) | 0/183 | 188/188 | 1.00, 1.00 | **1.00, 1.00** | 0.03m | 11° | –(0.1s被收) | ~26 |
| v_c_s1(13100) | 0/183 | 188/188 | 1.00, 1.00 | **1.00, 1.00** | 0.04m | 14° | – | ~26 |
| v_c_s2(13000) | 0/183 | 188/188 | 1.00, 1.00 | 0.30, 0.41 | 0.06m | 17° | – | ~26 |
| v_c_s3(13100) | 0/183 | 188/188 | 1.00, 1.00 | 0.95, 0.95 | 0.06m | 15° | – | ~26 |
| v_yaw_r2(13900) | 0/183 | 188/188 | 1.00, 1.00 | **1.00, 1.00** | 0.04m | 12° | – | ~25 |
| v_p05(13000) | 0/183 | 188/188 | 1.00, 1.00 | **1.00, 1.00** | 0.02m | 11° | – | ~26 |
| w_c_s0_res1_r5(13900) | **183/183** | 188/188 | 1.00, 1.00 | 0.86, 0.89 | 0.03m | 11° | 144° | ~45 |
| w_c_s1_res1_r3(11200) | **183/183** | 188/188 | 1.00, 1.00 | 0.60, 0.61 | 0.02m | 16° | 136° | ~43 |
| w_p08_res1_r2(12900) | **183/183** | 188/188 | 1.00, 0.99 | 0.93, 0.93 | 0.05m | 14° | 121° | ~40 |
| w_yaw_res1_r2(12000) | **183/183** | 188/188 | 1.00, 1.00 | 0.95, 0.90 | 0.04m | 14° | 133° | ~43 |
| w_p05_res1_r4(11800) | **183/183** | 188/188 | 1.00, 1.00 | 0.68, 0.68 | 0.04m | 12° | 102° | ~40 |

- 正手接触率全臂 0(V:到不了帧;W:到帧但拍面反,contact 的 signed-face 门不放行)。
- 物理摔倒率全臂 0、超时 0:收题 100% 是 ee 参考包络守卫(round5 已证明抬掉守卫放到底
  就是物理摔——守卫只是把塌倒提前 ~0.3-0.7s 判掉)。存活步数 = 平均每题控制步
  (V ~26 步 = 0.5s 击球 + 挥后塌;W ~40-45 步 = 击球 + 更长的跟随段跟得住)。

### 与作废批(round3,老协议)对比

| 量 | round3(协议错位) | round6(governor 协议) |
|---|---|---|
| 反手到帧(11 臂) | 0-188 参差(中位 ~131) | **11/11 全 188** |
| 反手接触率 | 全 0 | **11/11 全 ~1.00** |
| 反手回球率 | 全 0 | 0.30-1.00(V 四臂 1.00) |
| 反手拍位 | 0.22-0.33 m | 0.02-0.06 m |
| 反手拍面 | 15-58° | 11-17° |
| 正手到帧 | 全 0 | V 全 0;**W 全 183** |

### 解读(仍带 lineage-inexact + plant 差异双 caveat)

1. **协议伪影结论终审坐实**:同一批 checkpoint、同一考卷,只换评估器协议,反手从全 0
   到全 100% 接触——round3/4/5 的 0 分确系判分器欠账。
2. **正手是真问题,且 V/W 失败模式不同**:V 臂开题 0.1s 内拍速冲到 6 m/s、腕部冲出参考
   包络被收(挥后失衡同款爆发节奏);W 臂能全程贴着 governed 参考走到击球帧(拍位
   0.06-0.10m)但拍面反 102-144° ≈ 180°−x,即用背面迎球。与 Isaac 训练侧
   "V摔但回球好 / W稳但不回球"的臂间已知格局一致——这是行为/迁移问题,不再是协议问题。
3. **机制候选(留给下一步)**:正手 W 拍面反 ≈ 单翻病家族(mount sign / face 约定)在
   MuJoCo 侧的重现,值得从 exam CSV 的 signed_face_dot 分布 + Isaac 侧同题对照入手;
   V 正手爆发离包络可用 --ready-state teacher-reference(RSI 到 windup 帧,训练 75% 开题
   分布)做一格诊断,分离"冷 stand 起步转移"与"根本挥不了 2.6x 正手"。
4. 成绩性质:仍为谱系诊断档(evaluation_contract_exact=false,lineage_exact=0 +
   explicit-PD/零阻尼/零摩擦 plant 豁免),但**协议正确**,臂间排名与机制结论可用。
