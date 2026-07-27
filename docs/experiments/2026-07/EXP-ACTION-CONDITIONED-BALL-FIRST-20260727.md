# EXP-ACTION-CONDITIONED-BALL-FIRST-20260727

Status: Prelaunch / no formal run started  
Date: 2026-07-27 CST  
Human owner: Franco  
Executor: Codex assisted; Franco supplied the design decision and compute intent

## 问题

训练要同时满足：

1. 球与下发 task 物理自洽；
2. 每个动作都学会自己附近的来球与落点泛化；
3. 训练前不依赖一个尚无能力数据的 selector；
4. 训练后能产出 planner 可校准、可拒绝域外请求的逐动作能力证据。

历史 task-first 先采拍速/拍面，无法保证存在匹配来球。现有 free ball-first 虽自洽，却可能让
solver 总选容易方向，学习曲线不能代表指定动作的能力。

## 假设

主假设：按

```text
冻结 action -> 采该动作的到球时间/ball/base/aim
            -> fixed-action 解 task + teacher rate -> 执行
```

训练，可在不预先实现 selector 的情况下得到逐动作条件能力。每动作先找各轴 marginal frontier，
再用 joint `rho` 把联合域的
[`safe-policy failure`](../../DEFINITIONS.md#safe-policy-failure)控制在 `10% ± 2.5pp`。

竞争解释：

- 当前学习改善主要来自球/task 自洽；
- free solver 在暗中降低难度；
- 实际低 Reward 权重减少了 imitation 与 task 的梯度冲突；
- 同一 physics oracle 同时解题和判题产生循环验证。

因此“现在效果不错”不能直接归因于任一 setting。

## 冻结合同

详细真源是[按动作条件化 Ball-first 合同](../../interfaces/action_conditioned_ball_first_contract.md)。
本实验只列裁决：

- 训练期 selector 关闭；action 由 balanced schedule 选择并在 episode 内冻结；
- 每动作独立 profile、RNG tape、lazy pool、curriculum 和 evidence window；
- schema v3 使用 exact 32-arm catalog；`no_move` 禁四个 base-travel arm 后有效 28 个；
- time-to-contact 两侧、所有连续维度 lower/upper 和两个方向的 tangent 正负侧都独立扩张；
- `teacher_rate=required site speed/reference site speed`，不 clip；ready wait 最多 1 秒；
- manifest 是 metadata，不自报授权；formal motion admission 单独验证；
- base spawn 只在 true reset 写 root；base goal 属于 per-swing task receipt，WRAP 不写 root；
- 首轮只跑 `no_move`，`move` 等 base timing/collision/generation ledger；
- aim 是能力轴；只训固定 aim 时其他 aim 必须 OOD/abstain；
- 10% 只看 safe closed policy failure，solver reject、table hit、fall、infrastructure invalid 分账；
- 256 attempt 只作 canary；正式窗口默认至少 768；
- 新正手 table hit 零容忍。

## 最小因果矩阵

先冻结 level-0 proposal tape、`no_move`、action schedule 和落点域，不启动态课程。

### A. Solver 配对

| 臂 | Solver | Reward | 回答 |
| --- | --- | --- | --- |
| `S-free` | free direction | 当前实际低权重 | 曲线是否受“总挑容易方向”推动 |
| `S-fixed` | selected-action fixed direction | 当前实际低权重 | 指定动作绑定的真实代价 |

训练比较只使用双方都可解的 intersection tape；同时报告全 proposal coverage，避免 redraw 换题造成
幸存者偏差。

### B. Reward 配对

冻结 `S-fixed` 的 solved tape：

| 臂 | Solver | Racket quality 权重 | 回答 |
| --- | --- | --- | --- |
| `R-effective` | fixed | runtime 实际 `4/0.5/0.5` | 当前现象的真实基线 |
| `R-nominal-high` | fixed | 名义 `393.4/295.1/229.5` | 高权重是否改善服从，还是放大冲突 |

每臂至少两个 paired fresh seeds，并保存 exact effective-Reward receipt。若需估计 solver×Reward
交互，再补完整 `2×2`，不在首轮同时扩矩阵。

## 课程测试

因果配对裁决后才启用：

1. per-arm marginal：time-to-contact、contact position、speed magnitude/direction、spin
   magnitude/direction、base spawn、aim；`move` 以后开放 base travel；
2. joint `rho`：20% anchor / 60% interior / 20% frontier-probe；
3. difficulty target：10% 对 20%，两臂独立 run，不在一个 controller 中切换；
4. frozen heldout：固定 policy contract、独立 checkpoint SHA、全局单调 generation、互斥 seed
   block、不可重放 window receipt；训练 caller 不能自报 evaluator authority。

live 训练每动作、每候选 arm 保留最近 100 个同 cell 的安全闭环结果，优先排“对成功率伤害最小”
的方向；不足样本、固定轮次和最大 starvation age 强制探索。rolling-100 只选候选，256 个 frozen
attempt 作 canary，至少 768 个互斥 heldout 才能正式改变 frontier。

## 动作与 scope 消融

- `no_move` 对 `move`：只在 move 的到达时序/桌碰/recovery ledger 通过后；
- upper scope 对 full scope；
- full scope、但无腿 pose Reward，对 full scope + 12 腿 pose Reward；
- 单挥拍 episode 对延长 episode/多 cycle；
- N5 固定后再做 N93 full-body inventory/canary，不能从 N5 checkpoint 假装同一 actor contract。

物理桌 collider 和 table-hit truth 不做“关闭安全门”的科学消融；可比较 shaping penalty，但硬安全
事件始终保留。

## 新正手 Gate

旧 `fh_loop` 从候选 N5 view 排除但保留 bytes。`fh_loop_high` 比较 station X shift：

```text
0 cm / -5 cm / -10 cm
```

upper/full 共同取最近全过档。每档独立报告：

- action-specific post-retime behavior/contact `t_hit`；
- `t_cycle` 与 shared-ready recovery；
- vendor MuJoCo physical `right_racket` site strike speed；
- ready→recovery 全轨 table/body/ground clearance；
- fixed-action center-ball solver coverage 与独立 forward return；
- Pod Isaac filtered table-contact positive/negative smoke。

source frame-54 wrist-COM 速度与 1.08 s anchor 仅是 diagnostic，不能替代上述四项。

## 发射顺序

1. host contracts；
2. 新正手/N5 motion admission；
3. Pod1 CPU exact checkout；
4. Pod1 table scene；
5. 单动作 `no_move` level 0，1 env × 2 PPO updates；
6. N5 center-only canary；
7. marginal → joint curriculum；
8. Solver A/B、Reward A/B、10%/20%；
9. move；
10. Pod2 N93 inventory/admission → CPU → 1-env canary。

GPU 仅在现场确认空闲后使用；不得发送信号给现役训练、清未知 lock 或覆盖旧 namespace。

## 当前证据与阻塞

截至本记录（下列历史 pass count 在 schema v3 集成完成后必须以新 union 重写）：

- host 已有 strict manifest、per-action deterministic sampler、fixed-action proposal solver、
  async marginal→joint curriculum 和 exact-resume/runtime receipt contract。2026-07-27 的 v3
  迁移中间检查为 sampler+manifest+adapter `184 passed`，加 curriculum/evaluation 后 domain-core
  `210 passed in 8.35s`；旧 `254 passed` 属于已废弃的对称 7-axis schema，不能作为本轮最终成绩。
  最终数字须等 runtime/Motion/train union 全绿后更新；
- curriculum 的 checkpoint generation 已与 policy contract 分离；load 从 genesis receipt 重放
  reducer，拒绝 generation 回退、同代多 checkpoint、跨 mobility 重复采样区间和伪造
  `certified`；
- fixed-action replay 已补 global speed budget、mount face sign 和 venue normal closing-speed
  `[1.4,7.2] m/s` 门；对应本机 Torch 回归已进入上述核心 pass count，仍待 Pod 环境复跑；
- 红队复现 sampler 可伪造 birth、broker 批量失败不回滚 provider tape、课程证据无 frozen
  evaluator capability 三个 launch blocker；当前正在收口 exact birth transcript、stateful provider
  checkpoint/rollback 和 evaluator authority，完成前课程只能 hold，不能凭调用方自报证据扩域；
- action-ball 首 episode 固定 `init_at_random_ep_len=false`，确保首批 attempt 从完整 true reset
  开始；该位进入 preflight/policy recipe SHA。旧 CQ producer 仍关闭，只保留并哈希 fixed-action
  solver 的 overdraw/iterations/tolerance/speed-budget/max-external-rounds 五个 knob；
- 新正手 source SHA 是
  `7d045fcb036ffa668dede4607cfcc82e789a0db7ab86fd8df9dd52cfd5ac4153`；
- source 只有 wrist-COM diagnostic；正式 upper/full、grounded trace、behavior `t_hit` 和 trusted
  promotion certificate 缺失；
- Pod1/Pod2 预检时全部 GPU 被现役 `train.py` 占用，未打扰；
- sampler 的 N93/E4096 单轮 state 为 `160,906 B`，但真实 `4096 birth + 4096 sample` 已到
  `6,070,936 B`；100 轮线性外推约 `607 MB`。这暴露的是退休历史没有跨 sampler/broker/pool/
  provider/Racket 原子 compact 的长跑阻塞，不可用压 JSON 掩盖；
- 2026-07-27 只读复核 Pod3：用户给出的
  `/workspace/yikang/chingmu_retarget/out_refined/` 当前不存在；现有
  `chingmu_a3_units_v2` 与 `ball_ext` 有 74 组 action/ball sidecar 精确同名配对，
  `units_meta` 和球报告则有 108 个击球单元。样例 metadata 已含 strike frame、incoming
  position/fit velocity、outgoing velocity、face、station 与 retime，可作为 center 来源；
  但不能把 74 或 108 猜成用户声明的 93。N93 仍缺 exact ordered 93 件输出、compiler/admission
  receipt 与逐件 action-center manifest；
- 生产 planner/C++ 仍把动作折叠为正/反手两个 clip，selector core 尚未进入生产 wire。

所以当前状态是 **source/CPU/prelaunch 可推进，正式 GPU 长跑不可启动**。解除阻塞须新增可复现证据，
不能靠修改 manifest 布尔值。

## 结果表

尚无正式 run。任何后续条目至少记录：

```text
commit / manifest / motion admission / prototype / solver / physics / reward SHA
Pod + GPU + exact run namespace
action UID + profile + curriculum cell/window
P/A/I/S/C/L/F/U/X conservation
per-action solver coverage, legal return, table/fall/collision
checkpoint and heldout capability artifact SHA
```
