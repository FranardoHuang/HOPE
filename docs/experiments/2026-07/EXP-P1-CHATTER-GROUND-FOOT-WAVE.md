# EXP-P1-CHATTER-GROUND-FOOT-WAVE — 抖动-地面-脚部消融波（8 臂，2026-07-22）

- 状态：`preregistered`（两道 wiring 闸门未开 + source.commit 占位，未渲染任何命令）
- 阶段/轴：Phase 1 / 抖动惩罚剂量 × 地面物理覆盖 × 脚部塑形 × 惩罚减负 各买单变量臂
- 集成小目标：把"站立异响 E1 复核 PDF + Franco 2026-07-22 八条"里训练侧成立的欠账
  （action-rate 偏弱、plant 地面覆盖缺口、脚部塑形缺失、软惩罚压制模仿）变成可拒绝的
  单变量证据
- 人类负责人：Franco（情报来源与拍板）
- 执行者：Claude
- 复核/决策负责人：Franco
- 最高证据等级：`E1`（prereg + 队列/渲染器/单测已落盘：47 项三件套测试 + 40 项脚部
  reward 测试 + 38 项地面接线测试全绿）。发射后升级路径同 intel 波：逐臂 probe →
  `E3` runtime mechanics；终档 K100 同卷 → 单 seed 机制诊断（谱系 inexact，永为
  diagnostic）；正名须 exact-lineage 重跑。
- 创建日期/最后复核日期：2026-07-22 / 2026-07-22

人话导言：部署侧站立异响的外部 E1 级复核（[PDF 已入库](../../research/agibot_a3_standing_chatter_evidence_ranked_review_2026-07-22.pdf)）
把首要嫌疑指向部署侧两套站立增益接口（修复走部署侧四欠账，另立
[分支修复审计](../../research/branch_fix_audit_20260722.md)），但同时确认两条训练侧欠账成立：
**action-rate 平滑罚偏弱**（现役 -0.1，别人经验 0.1 一路测到 1）与 **plant 地面覆盖缺口**
（地面摩擦从没随机过、机器人材质摩擦下界 0.3 很滑、地面永远是平的——别的队加了摩擦+
不平整后机器人脚就不搓地了）。加上 mjlab reward 采纳审计
（[取舍文档](../../research/mjlab_reward_adoption_20260722.md)）与 Franco 第 8 条
（新加软限制压制模仿→学不会击球），共 8 条单变量臂。父本只用 `W`（拍心优先×自由非击球
臂，`model_6700`），基础配方与 24 格矩阵 [`w_c_s0`](EXP-P1-BALANCE-TEMPORAL-MATRIX-20260720.md)
（C+S0）逐字相同——那一格就是对照，不再买新对照。续训臂预算 `13301` updates
（6700+13301≈20001，对齐 Jiayi 新目标 2万-2.5万 iter 的下沿）；rough 一臂 fresh 独立
`20001` updates。

三件套：[configs/phase1_chatter_ground_foot_wave_20260722.yaml](../../../configs/phase1_chatter_ground_foot_wave_20260722.yaml)
+ [scripts/run_phase1_chatter_ground_foot_wave_queue.py](../../../scripts/run_phase1_chatter_ground_foot_wave_queue.py)
+ [tests/test_run_phase1_chatter_ground_foot_wave_queue.py](../../../tests/test_run_phase1_chatter_ground_foot_wave_queue.py)。

## 8 臂各自的假设与判读标准

| 臂 | 改动（人话） | 假设 | 判读标准（对 w_c_s0 同里程碑） |
| --- | --- | --- | --- |
| ar02 | action_rate -0.1→-0.2 | 平滑罚加倍能压 q_des 抖动而不伤击球 | 抖动代理（action_rate 台账、slew 探针尾部）下降且击球组（pass/composite/virtual return）不掉超过噪声带 |
| ar05 | action_rate -0.1→-0.5 | 剂量中点 | 同上；三点连成剂量曲线 |
| ar10 | action_rate -0.1→-1.0 | 别人经验的上限档 | 若 ar10 击球塌、ar02/ar05 不塌，最优剂量在中间；反向情报（jiayi V15 降到 -0.01 靠投影约束）同时记录 |
| grip | 机器人材质摩擦随机化 静[0.3,1.6]→[0.6,1.6]、动[0.3,1.2]→[0.5,1.2] | 摩擦下界太滑是脚搓地/漂移的一部分 | 脚滑探针（settle foot-slip 台账）、挥拍后基座漂移下降；击球不掉 |
| rough | 随机凹凸地形 2-6 cm，fresh 从零 20001 updates | 平地过拟合是部署失稳的一部分；粗糙地能学出更稳的支撑 | 只看绝对水平：能否在粗糙地上学会站立击球（pass>0）+ fall 率；与 w_c_s0 对比仅方向参考（谱系不同，必须带 caveat） |
| footrw | foot_soft_landing -3e-3 @300 N | 落地冲击塑形能减少砸脚/抖动 | first-contact 峰值力分布左移、抖动代理下降;击球不掉。量纲：实现输出是无量纲超阈倍数（阈值归一、单脚封顶 3），mjlab -1e-5/N 等效剂量 = -3e-3（别抄 -0.1=33 倍剂量） |
| penlight | 六个软惩罚降 ~1/3（face -0.4→-0.13、foot_ori -0.3→-0.1、upright -1.0→-0.33、pre-strike slip -0.4→-0.13、foot_slip_sq -1.0→-0.33、foot_drag -0.5→-0.17；后两键 07-22 新接 CLI） | Franco 第 8 条：软限制压制模仿→学不会击球；减负后击球恢复 | 击球组回升（尤其模仿类臂的 hit/pass）；失稳率不恶化超噪声带。硬保护（限位/包络/qdes_clamp）不动 |
| kdpassive | passive_damping_fold=true（源码未接线） | 训练 kd 与部署有效 kd 有缝（jiayi 07-09 A/B：漂移均值 0.066→0.084、峰值 0.121→0.464，~90% 复现 AGI 台架跟随放大） | 实现合入后再定；判读须带"jiayi 侧训练分布自述还没调好"caveat |

judge/K100 同卷判读用现役 signed-face 判分器；单臂结论只升到"方向证据"，
胜者档位须另在 exact-lineage 重跑才能进正式配方。

## Run table（发射后逐臂回填）

| job | run_name | stage | pod/gpu | 发射时刻 | 终档 | 里程碑成绩 | 处置 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grip | p1cgf_grip_seed3_20260722 | — | — | — | — | — | 未发射 |
| footrw | p1cgf_footrw_seed3_20260722 | — | — | — | — | — | 未发射 |
| ar05 | p1cgf_ar05_seed3_20260722 | — | — | — | — | — | 未发射 |
| penlight | p1cgf_penlight_seed3_20260722 | — | — | — | — | — | 未发射 |
| rough | p1cgf_rough_seed3_20260722（fresh） | — | — | — | — | — | 未发射 |
| ar02 | p1cgf_ar02_seed3_20260722 | — | — | — | — | — | 未发射 |
| ar10 | p1cgf_ar10_seed3_20260722 | — | — | — | — | — | 未发射 |
| kdpassive | p1cgf_kdpassive_seed3_20260722 | — | — | — | — | — | 未发射（源码未接线） |

## 依赖（全部满足前渲染器 fail-closed，不出任何 SSH 命令）

1. **source.commit 占位符**：`PENDING_40HEX_AFTER_WIRING_MERGE`。07-22 工作分支
   （脚部 reward + 地面/地形键 + YAML-null 删参 + 归一化预检 + 精确续训包三搬运）合并、
   在 pod 重新 checkout `/workspace/codexschema/nohope_cgf_20260722` 后钉 40-hex。
2. **groundfoot_contract.wiring_confirmed 翻真**（锁 grip/rough/footrw/penlight 四臂）：主控
   grep 远端 exact commit 的 train.py，确认 7 个新键逐字在白名单（wiring 本地已落盘+测试
   全绿，等合并核对；penlight 因蹭滑/拖脚两键 07-22 新接 CLI 而入锁）。
3. **kdpassive_contract.wiring_confirmed 翻真**（锁 kdpassive 一臂）：
   `task.plant.passive_damping_fold` 源码完全未接线，实现+单测+ground_plant 式合同块
   合入 main 后才翻。
4. **pod 排位**：pod 已于 07-22 下午以原端口恢复（pod1=yikang 四条训练占满，pod2 空）；
   第一优先是判卷欠账（已收口终档 model_10700-13300 一份成绩都没有，可即刻在 pod2 跑），
   本波 smoke/发射排其后，用 pod2 空槽按 launch_order 填。
5. **发射工序**：先 probe（2 updates smoke），一格通过即按 launch_order 发全矩阵
   （精简治理：不设多层仪式）；同 pod boot 串行、错峰 ≥60 s。

## 谱系与合同声明

- 全波 `evidence_class=diagnostic_only_intentional_parent_contract_mismatch`：W 后代
  hard contract 有意 mismatch;grip 两键还会让新 checkpoint 长出 `ground_plant`
  schema-3 合同块（这是指纹在工作，不是事故）。
- **rough fresh 铁律**：平地 checkpoint 不得静默上粗糙地——渲染器断言 rough 命令无任何
  checkpoint 键；`ground_plant` 合同块 + resume `_contract_diff` 双保险。
- **连续能力次序不被绕过**：本波是站立单拍配方消融，不改变连续能力的固定次序
  `T0`（按周期换题）→ `T1`（事件驱动结构、reward 冻结）→ `T2`（learned shaping）；
  任何臂的胜出都不授权跳过 T0/T1 直接上 T2 或连续演练。
- 本队列只做仿真（simulation_only），永不驱动真实机器人。
