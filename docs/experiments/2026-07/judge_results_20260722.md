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
