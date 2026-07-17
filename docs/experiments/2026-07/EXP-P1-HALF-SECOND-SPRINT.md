# EXP-P1-HALF-SECOND-SPRINT — 用二十三个单 seed 问题冲刺半秒击球

- 状态：`running`
- 阶段/轴：Phase 1 / 准备时间、动作速度、模仿强度与平衡 Reward
- 集成小目标：从共同 `model_5700` 找到至少一条能在 0.5 秒题中触球的候选
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：`E1`（Isaac 训练诊断）
- 创建日期：2026-07-18

这里的 `run_name` 是每条训练的唯一可读名字；参数真源在
[`phase1_half_second_sprint_20260718.yaml`](../../../configs/phase1_half_second_sprint_20260718.yaml)。
[`initial TTS mixture`](../../DEFINITIONS.md#initial-tts-mixture) 是“新球揭题时还剩多少准备时间”的
混合分布；[`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge) 是接近关节速度上限后才收费的惩罚，
不是随机推力。

## 为什么现在直接跑组合

现役代表 `model_5700` 在严格 [0.5 秒 K100](EXP-P1-TASK-REVISION-0P5-K100.md) 中是
`0/100` 触球、`0/100` 上台、`0/100` 摔倒。它首先暴露的是“来不及完成击球”，不是单拍平衡崩溃。
因此本轮不再复制同一失败设置的 seed，而是从同一 parent 同时测试二十三个不同解释：课程是否太慢、
速度惩罚是否太强、老师动作是否在触球窗和球拍目标打架、拍心梯度是否不够直接，以及加强准备姿态与
释放非击球臂能否守住加速后的平衡。

## 简化启动流程

本轮训练不使用 SHA、receipt、claim 或 activation。每次启动只做三件事：

1. 训练配置能被 Hydra 正常解析；
2. 日志出现第一个真实训练 iteration；
3. 日志没有 `Fatal`、`Traceback`、`OOM` 或 importer 崩溃。

通过这三项就接受为运行中的训练；checkpoint 的行为读数负责判断科学结果。这样保留最基本的
“确实开始训练”检查，但不让发布式账本挡住 GPU。

## 所有格共同不变

| 项目 | 固定值 |
| --- | --- |
| parent | `taskrev_p2_equal_reward@model_5700`，完整恢复 policy/value/optimizer/normalizer |
| source | `/workspace/codexschema/nohope_task_revision_b1f5a38` |
| 动作 | `hope_forehand_v4rg_cal.npz` + `hope_backhand_v4rg_cal.npz` |
| 训练题库 | `s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz` |
| 训练量 | `4096 env`，先续训 `1001` update，每 `100` 保存；只把胜者延长到 `4001` update |
| seed | 每格只跑 `seed3`；弱格不复制 seed |
| 普通短准备课程 | 更短压力/严格半秒/快速部署/长来球 = `0.10/0.45/0.40/0.05` |

只有 `ultra_half` 和 `feasible` 改课程分布：前者为
`0.05/0.70/0.20/0.05`，后者为 `0.05/0.10/0.45/0.40`。

## 二十三个不同科学格

| 格 | 人话问题 | 相对普通对照的唯一主要变化 |
| --- | --- | --- |
| `control` | 普通短准备课程本身是否足够？ | 无 |
| `focus_qdot0` | 关节速度铰链惩罚是不是主要瓶颈？ | qdot 权重 `-5 → 0` |
| `ultra_half` | 七成严格半秒样本会不会学得更快？ | 使用 ultra-half 课程 |
| `feasible` | 先学更多可完成样本能否搭出迁移台阶？ | 使用 feasible 课程 |
| `actionrate0` | 动作变化率平滑是否阻止快速起拍？ | action-rate 权重 `-0.1 → 0` |
| `window_mimic0` | 临近触球时老师是否在和实时目标打架？ | 触球窗内模仿比例 `0.25 → 0` |
| `global_mimic_half` | 老师全程降成软先验是否更好？ | 全局模仿比例 `1.0 → 0.5` |
| `position_heavy` | 先把拍心赶到球附近是否更容易起步？ | 拍心/拍速/拍面权重从约 `9.67/9.67/9.66` 改为 `17/7/5`，总和仍为 `29` |
| `strong_ready` | 加速时更强准备姿态能否守住平衡？ | 脚姿态 `-0.3 → -0.6`；击球前直立 `-1 → -2` |
| `free_arm` | 非击球臂不模仿老师后能否专注平衡？ | 开启自由非击球臂 |
| `ready_free` | 强准备姿态与自由非击球臂会不会相互增益？ | 合并上面两个变化 |
| `full_combo` | 把速度、目标主导和准备平衡一起放开能否最快出 demo？ | 合并 qdot0、actionrate0、两种 mimic 降权、position-heavy、strong-ready、free-arm |
| `full_combo_ultra` | 完整组合与七成严格半秒样本会互相增强还是冲突？ | `full_combo` + ultra-half 课程 |
| `full_combo_feasible` | 完整组合是否需要更多可完成样本先搭出台阶？ | `full_combo` + feasible 课程 |
| `mimic075_window0` | 触球窗外保留更多老师约束是否更稳？ | 全局模仿 `0.75`，触球窗模仿 `0` |
| `velocity_heavy` | 半秒动作更缺拍速还是拍心到位？ | 拍心/拍速/拍面权重 `7/17/5` |
| `position_ready` | 拍心优先与强准备姿态是否相互增益？ | position-heavy + strong-ready |
| `actionrate_half` | 平滑惩罚完全关闭是否过猛？ | action-rate 权重 `-0.1 → -0.05` |

`full_combo` 是演示候选搜索，不是单变量因果结论；单变量格负责解释它为什么有效或无效。

## 当前接受运行的 run 映射

| 实际 `run_name` | 对应问题 | 实际课程 | 说明 |
| --- | --- | --- | --- |
| `hs_a_control_seed3` | 旧课程对照 | 旧 balanced `0.15/0.20/0.30/0.35` | Pod2 accepted；不等同于新表的 short-focus control |
| `hs_b2_deadline_focus_seed3` | focus 下 qdot `-5` 对照 | `legacy_focus_1 = 0.25/0.40/0.25/0.10` | Pod2 accepted；与下一条组成旧课程内的 qdot 配对 |
| `hs_c2_deadline_qdot0_seed3` | focus 下 qdot `0` | `legacy_focus_1 = 0.25/0.40/0.25/0.10` | Pod2 accepted；只与上一条作直接 qdot 比较 |
| `hs_d2_ultra_half_qdot0_seed3` | ultra-half 课程 + qdot 0 | `0.05/0.70/0.20/0.05` | Pod2，accepted |
| `hs_e_feasible_qdot0_seed3` | feasible 课程 + qdot 0 | `0.05/0.10/0.45/0.40` | Pod2，accepted |
| `hs_f_focus_window_mimic0_seed3` | 触球窗老师静音 + qdot 0 | short-focus `0.10/0.45/0.40/0.05` | Pod2，accepted |
| `hs_g2_focus_actionrate0_seed3` | 关闭 action-rate 惩罚 | short-focus | Pod2，accepted |
| `hs_h4_focus_qdot0_globalmimic05_seed3` | 全局模仿减半 + qdot 0 | short-focus | Pod2，accepted |
| `hs_i3_focus_qdot0_pos17_7_5_seed3` | 拍心/拍速/拍面 `17/7/5` + qdot 0 | short-focus | Pod2，accepted |
| `hs_o2_feasible_fullcombo_seed3` | feasible 课程 × 完整组合 | feasible | Pod2，accepted |
| `hs_q2_focus_qdot0_mimic075_window0_seed3` | 全局模仿 `0.75`、触球窗静音 + qdot 0 | short-focus | Pod2，accepted |
| `hs_r2_focus_qdot0_vel7_17_5_seed3` | 拍心/拍速/拍面 `7/17/5` + qdot 0 | short-focus | Pod2，accepted |
| `hs_p1_j2_focus_qdot0_ready2x_seed3` | 强准备姿态 + qdot 0 | short-focus | Pod1，accepted |
| `hs_p1_k2_focus_qdot0_freearm_seed3` | 自由非击球臂 + qdot 0 | short-focus | Pod1，accepted |
| `hs_p1_l2_fullcombo_seed3` | 完整组合 | short-focus | Pod1，accepted |
| `hs_p1_m2_focus_qdot0_ready_free_seed3` | 强准备姿态 × 自由非击球臂 + qdot 0 | short-focus | Pod1，accepted |
| `hs_p1_n2_ultra_fullcombo_seed3` | ultra-half 课程 × 完整组合 | ultra-half | Pod1，accepted |
| `hs_p1_p2_focus_qdot0_actionrate_half_seed3` | action-rate 惩罚减半 + qdot 0 | short-focus | Pod1，accepted |
| `hs_p1_u_cached_pos_ready_seed3` | 拍心优先 × 强准备姿态 | short-focus | Pod1，cached USD，accepted |
| `hs_p1_v_cached_vel_ready_seed3` | 拍速优先 × 强准备姿态 | short-focus | Pod1，cached USD，accepted |
| `hs_p1_w_cached_pos_free_seed3` | 拍心优先 × 自由非击球臂 | short-focus | Pod1，cached USD，accepted |
| `hs_p1_x_cached_vel_free_seed3` | 拍速优先 × 自由非击球臂 | short-focus | Pod1，cached USD，accepted |
| `hs_p1_y_cached_pos_window0_seed3` | 拍心优先 × 触球窗老师静音 | short-focus | Pod1，cached USD，accepted |

前三条是已经启动的旧课程映射，不能被改写成新 short-focus 的严格配对。其余 accepted run 均已通过
“配置解析、首 iteration、fatal 扫描”这三个简化启动检查；这里的 accepted 只表示训练确实在跑，
不表示这个格已经胜出。

## 启动失败与精确处置

| 尝试 | 失败位置 | 原因与处置 |
| --- | --- | --- |
| B、C | 首 iteration 前 | 参数前缀替换误伤相邻权重，Hydra 配置拒绝；修正为 B2/C2 后启动 |
| D | importer | 同时启动多个 Kit 时 importer `rc134`；错峰后 D2 启动 |
| G | shell | 参数被错误追加到 shebang；重新生成 G2 脚本后启动 |
| H、H3 | 首 iteration 前 | H 使用无效配置键，H3 使用无效 CUDA device；修正为 H4 后启动 |
| I2 | importer | importer 挂起；核对精确进程组后只停止 I2，I3 已启动 |
| O3 | 首 iteration 前 | 重复发射；只停止尚未训练的重复实例，保留 O2 |
| S2 | importer | importer malloc 失败，保留日志，不把它算成科学失败 |
| S3 | importer | importer 挂起；核对精确进程组后只停止 S3 |
| S4 | importer | importer 挂起；核对精确进程组后只停止 S4 |
| Z、Z2 | Kit 启动 | GPU2 第四路两次在 env/reward 前的 USD shader discovery 处 allocator abort；保留日志，不判配方失败，不再盲试 |

Pod1 当前是 11 条 accepted run 和 1 个空槽。预转换 USD 的 5 条新作业均已越过
首 iteration，并开始直接输出四个初始 TTS 桶的整数机会、完成、触球和合法回台计数。
GPU2 空槽保留到现有三路自然降到两路，然后才作为第三路补 `velocity_window0`；
不在同一第四路条件下继续碰运气。

## `+100`（5701–5800）训练内早判

预转换 USD 的五条新臂已经完成共同 parent 后的前 100 个 update。以下四列固定按初始准备时间
`<0.5 秒`、`=0.5 秒`、`(0.5, 0.9] 秒`、`>0.9 秒` 排列；所有比例都用 5701–5800
逐 update 的整数计数聚合后再计算，不使用跨历史 EMA。

| 候选 | 四桶完成率（%） | 四桶训练内合法回台率（%） | 读法 |
| --- | --- | --- | --- |
| `V`：拍速优先 × 强准备姿态 | `62.29 / 61.14 / 59.40 / 55.10` | **`21.96 / 36.03 / 38.21 / 27.01`** | 五臂中四桶回台率最高，但完成率最低 |
| `W`：拍心优先 × 自由非击球臂 | **`87.43 / 87.16 / 85.92 / 82.41`** | `19.41 / 29.90 / 30.75 / 18.97` | 五臂中四桶完成率最高，但回台率低于 `V` |

四桶的 `strike opportunity`（真实进入可判触球机会的整数分母）依次约为
`9.6k–9.9k / 42.9k–44.5k / 36.9k–38.9k / 4.3k–4.6k`。`U`（拍心优先 × 强准备）
与 `Y`（拍心优先 × 触球窗老师静音）落在完成和回台的折中区域，`X`（拍速优先 × 自由非击球臂）
也介于 `V` 与 `W` 的两端取舍之间。

这些是 Isaac 训练内的 virtual capture / legal return（虚拟触球/合法回台）读数，不是 vendor MuJoCo
部署成绩。当前没有一条臂同时在完成率、触球机会、回台率和安全性上全维支配其他臂，因此 `+100`
不淘汰任何一条；继续到 `+200`，再按同 parent 的非支配关系早判。

## `+200`（5801–5900）第二个独立训练窗

第二个不重叠的 100-update 窗继续用逐 update 整数计数。下表的完成率与训练内合法回台率为四桶合计；
摔倒率同时列出击球前与击球后的分解，单位均为百分比。

| 候选 | 完成率 | 训练内合法回台率 | 摔倒率合计（击球前 / 击球后） |
| --- | ---: | ---: | ---: |
| `U`：拍心优先 × 强准备 | `93.80` | `31.29` | `0.956`（`0.143 / 0.814`） |
| `V`：拍速优先 × 强准备 | `44.19` | **`62.59`** | `21.80`（`20.17 / 1.62`） |
| `W`：拍心优先 × 自由非击球臂 | **`94.37`** | `30.81` | `0.901`（`0.150 / 0.751`） |
| `X`：拍速优先 × 自由非击球臂 | `46.38` | `59.55` | `21.39`（`19.61 / 1.77`） |
| `Y`：拍心优先 × 触球窗老师静音 | `93.58` | `31.52` | **`0.855`**（`0.126 / 0.729`） |

四桶仍按 `<0.5 / =0.5 / (0.5, 0.9] / >0.9 秒` 排列：

| 候选 | 四桶完成率（%） | 四桶训练内合法回台率（%） |
| --- | --- | --- |
| `U` | `94.94 / 94.62 / 93.18 / 89.10` | `25.45 / 32.87 / 32.80 / 16.15` |
| `V` | `45.58 / 44.81 / 43.45 / 41.75` | `41.66 / 64.80 / 67.32 / 47.56` |
| `W` | `95.54 / 95.12 / 93.79 / 89.77` | `23.55 / 32.50 / 32.53 / 15.55` |
| `X` | `47.92 / 46.98 / 45.73 / 42.90` | `40.62 / 60.74 / 64.35 / 49.40` |
| `Y` | `94.48 / 94.28 / 93.14 / 89.09` | `24.14 / 32.96 / 33.19 / 19.85` |

各候选每桶的 `strike opportunity` 分母分别约为
`3.8k–10.0k / 37.5k–44.4k / 32.6k–38.9k / 3.8k–4.7k`，本窗已经不是因机会稀疏而产生的
零值。相对第一个窗，`U` 的完成/回台为 `+11.53 / +1.44` 个百分点、摔倒 `-3.10` 个百分点；
`W` 为 `+7.91 / +2.13 / -2.58`；`Y` 为 `+10.00 / +1.87 / -2.94`。三条拍心优先组都向
“高完成、约三成训练内回台、低摔倒”稳定收敛。

相反，`V` 的完成率 `-16.07` 个百分点、回台率 `+27.56` 个百分点、摔倒率 `+9.94` 个百分点；
`X` 分别为 `-24.01 / +27.56 / +13.40`。两条拍速优先组出现了“高回台、高击球前摔倒”的相变：
它们说明速度路径可以显著提高训练内回台，却不是可直接采用的胜者。五条候选仍无全维支配关系，因此
本窗不停任何一条，继续观察这种相变是否稳定。这里的回台仍是 Isaac training virtual 指标，不是
vendor MuJoCo 部署成绩。

## 判读与停止

- `+100/+200`：只检查自然续训、明显崩坏和是否开始出现击球机会。
- `+500/+1000`：比较半秒题触球、合法回台、摔倒、准备姿态和平衡；没有真实击球机会时，零回台不判失败。
- 同 parent 上保留非支配候选；明显弱格停止后把算力给胜者续训，不给弱格复制 seed。
- 最终候选必须再跑同一 0.5 秒题；Isaac 结果仍只是诊断，vendor MuJoCo 才是最终部署裁判。

## 当前决定

- 决定：`inconclusive / running`。
- 已采用的训练策略：同 parent、单 seed、二十三个不同科学问题并行；优先产生可击球候选，再用单变量格解释。
- 当前不能声称：任何格已经解决半秒击球、连续对打或跨引擎部署。
