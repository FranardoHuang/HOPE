# judge_results_20260722 — 已收口终档首批判卷(11 臂,诊断档,成绩暂不可解读)

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
