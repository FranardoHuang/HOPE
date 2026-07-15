# EXP-P1-DEMO-HOTSTART-PORTFOLIO — 今夜七个组合方案的严格续训

- 状态：`running；七条接受后代均已通过 first_iter 并 live，两条原始基础设施失败保持 rejected/absent`
- 阶段/轴：阶段 1，面向次日演示的组合方案
- 集成小目标：从三个已经学到约 3500 次更新的母本出发，尽快得到多个能兼顾挥拍、拍面和平衡的候选
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：`E2`（母本 provenance、七条后代 runtime/first_iter、两份 model-3700 合同证据；尚无行为排序）
- 创建日期/最后复核日期：2026-07-16

`v4rg`、[`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge) 等共享术语按
[术语与人话对照](../../DEFINITIONS.md)解释；本文的“母本”就是本轮开始前保存的 model-3500 checkpoint。

## 问题与诚实边界

问题不是“单独哪个 Reward 有因果作用”，而是：在剩余一晚内，能否把已经出现方向信号的机制组合成多个
可测试候选。七条都使用严格全状态续训：policy、value function 和 optimizer 一起从母本继续；不是从零再等
学习起步。

这批故意允许母本训练合同与新组合不同，因此 `checkpoint_allow_contract_mismatch=true`。训练器会把所有
后代永久标为 formal-ineligible：它们只能进入演示候选排序，不能冒充正式因果消融、fresh lineage 或最终
vendor MuJoCo 通过证据。

## 冻结 setting

| 字段 | 值 |
| --- | --- |
| source | `2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e` |
| 动作/题库 | `v4rg` 正反手与同一 schema-3 signed-face train bank |
| 母本 | Pod2 上 qdot、V1+V2、普通对照三个 `model_3500.pt` 的只读 v2 snapshot |
| 续训语义 | `checkpoint_tolerant=false`、不允许缺 hard contract、允许显式合同变化、完整 optimizer 保留 |
| 共同训练项 | seed 3；4096 environments；episode 10 秒；击球位置/速度/拍面 Reward=`14/10/5` |
| 追加预算 | 5001 updates；每 100 保存 |
| 绝对 checkpoint | `3700/4000/4500/5500/7500`，即母本后 `+200/+500/+1000/+2000/+4000` |
| 资源 | 前六条只允许 Pod2 GPU0/GPU1，按 0→1 逐圈；第七条只允许 GPU2 第四槽；今晚实测上限每卡四条 |

## 七个候选

| 候选 | 母本 | 组合 |
| --- | --- | --- |
| 强拍面版 | qdot | 两项模仿放松 + qdot `-5` + 拍面 `-0.4` |
| 中拍面版 | qdot | 同上，但拍面 `-0.2` |
| 不同 basin 强拍面版 | V1+V2 | qdot `-5` + 拍面 `-0.4` |
| 自由非击球臂版 | V1+V2 | qdot `-2.5` + 拍面 `-0.4` + 非击球臂不模仿 |
| 保守模仿版 | 普通对照 | qdot `-5` + 拍面 `-0.4`，不放松两项模仿 |
| 全栈版 | 普通对照 | 两项模仿放松 + qdot `-5` + 拍面 `-0.4` + 脚朝向 `-0.6` + 自由非击球臂 |
| 16 秒长回合版 | qdot | 两项模仿放松 + qdot `-5` + 拍面 `-0.4` + 脚朝向 `-0.3` + 自由非击球臂；同 episode 连续 3–4 拍观察平衡债 |

这里比较的是“明天哪个组合更可能可用”，不是把不同母本之间的差异解释成单一机制效果。

## 启动门与停止规则

1. 先用只读 `parent-inspect` 验证三个原始 `queue_claim.json`、`run_binding.json`、checkpoint 与 hard
   contract 的完整关系：claim canonical SHA/完整 argv、binding 的 source/run/process 关系、embedded
   iter=`3500`、非空 actor/critic、非空 optimizer `state/param_groups`、finite、schema-3、checkpoint↔hard
   SHA、checkpoint↔原始 claim SHA。inspect 不创建任何文件。
2. inspect 通过后，唯一一次 `parent-attest` 用 `O_EXCL` 把 checkpoint、hard contract、原始 claim 和
   binding 复制到固定 `parent_snapshots_v2/<parent>/`，设为只读，再只从 snapshot bytes 重做同一审计并
   no-clobber 写 v2 receipt。旧 v1 receipt 路径不能解锁本队列；运行期也不再读取可变 live 母本。
3. 2026-07-16 02:10 CST 只读 inspect 与唯一 attest 均通过；v2 receipt file SHA 为
   `fd200bd65ee00d33fb50a73f5de8d011cd810498ef626a3ca9d3a63b5bff2f34`。三个母本的
   checkpoint/hard/claim/binding SHA 已回填，机器队列显式切到 `launch_authorized=true`、六行
   `ready`；该切换不自动点火或重试。
4. 六条 scaleout 的 `model_500` 全部保全后，若 Pod2 GPU0/GPU1 当前各不超过三条，可先用第 4 槽发
   job1/job2，不必为此停现役。其余四组合只在按独立证据精确停止四条弱臂后逐圈补入；保留 GPU0 的
   V1-only 和 GPU1 的 foot-`-0.6`，最终仍为每卡四条。第七条仅在 GPU2 已有三条且第 4 槽可用时发射。
5. launch 成功前必须在日志同时看到显式 hard-contract mismatch、从 snapshot model-3500 恢复到
   iteration 3500 和 `optimizer=resumed`；在同一 GPU launch lock 内、调用 trainer 前，还要把该 job 的
   snapshot checkpoint/hard/claim/binding 四个 file SHA 与 activated queue 再对拍，关闭 verify SSH 到 load
   之间的漂移窗口。新 hard contract 必须逐值包含该行 qdot/conditional-face 权重。
6. `phase=first_iter` 还要求 run binding 的 PID=PGID/starttime/cmdline 在 `/proc` 仍是同一活进程，并且日志
   已出现第一条真实 `Learning iteration > 3500`；只有 RESUMED 行、3500 起始打印、自然退出或 PID reuse
   都失败。失败只写 exact identity 人工处置记录，不自动发信号或重试。
7. `+200/+500/+1000` 看是否启动、finite、机制是否真的激活和是否明显崩坏；只在真实击球后才有意义的
   稀疏回台指标，样本不足时继续。`+2000/+4000` 才用于次日候选排序。
   对第七条，`+200` 只判结构/激活，`+500` 才判安全和平衡债，`+1000` 才进入候选排序；稀疏命中为零
   不能在这些早期节点杀臂。
8. 任一 namespace 失败都保留，不自动 replay；本批不授权第二 seed、正式晋级、真机或 broad process signal。

## 两条基础设施失败与唯一人工重试

原自由非击球臂行在首迭代前以 `pre_marker_exit/rc134` 退出，日志包含 `malloc invalid size`；绑定进程
PID/PGID `429116`、starttime `557505718` 已确认不存在。原普通母本保守模仿行在首迭代前触发
`stale_timeout/rc125`；PID/PGID `429974`、starttime `557535387` 已确认不存在。两条都没有产生行为证据，
原 run directory 与 claim 永久保留并标为 `rejected`；claim、binding、run log、launch state、leader identity，
以及 stale 行的 pre-TERM/pre-KILL evidence SHA 全部绑定在机器队列的 `terminal_contract`。

仅为这两个基础设施失败新增一次 `retry_v2`。新行使用新 id、run name 和 run directory，loader 逐字段比较
parent、完整 recipe、seed、budget 与 milestones 必须和 predecessor 相等；claim 绑定 `retry_of`、完整终态证据、
`manual_retry_limit=1`、`automatic_retry=false`、`recipe_equal=true`。先在 GPU1 发自由臂 retry，消费新 claim 后
才允许在 GPU0 发保守模仿 retry。点火前还会在同一 GPU 锁内、创建新 run directory 之前重算旧
claim/binding/log/launch/identity 的 5 或 7 个 SHA，双次扫描旧 PGID 全部 `/proc` 成员，解析已绑定的
leader/pre-TERM/pre-KILL 身份并逐个要求 PID absent，同时拒绝这些成员仍持有 NVML context。两条 retry-v2
现已各消费一次新 namespace 并 live；原失败 `429116/429974` 仍为 `/proc` 与 NVML 双重 absent。

历史母本的完整 recipe 仍以其 canonical self-bound queue claim + run binding 账本为信任边界；本轮会反向验证
账本、argv 与 checkpoint lineage，但没有另造一份独立的旧 recipe 真源。这是已知边界，不影响本轮 v3 的
snapshot/load 时序修复。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | 证据 | 有效性说明 |
| --- | --- | --- | --- |
| qdot 母本强拍面 `phase1_demo_qdot_v1v2_face_w0p4_seed3_20260716` | live | PID=PGID `426506`；iter `4192` / `model_4100`；first proof `fedbe84a...05af` | demo-only |
| qdot 母本中拍面 `phase1_demo_qdot_v1v2_face_w0p2_seed3_20260716` | live | PID=PGID `427190`；iter `4193` / `model_4100`；first proof `e397c918...899e` | demo-only |
| V1+V2 母本强拍面 `phase1_demo_v1v2_qdot_w5_face_w0p4_seed3_20260716` | live | PID=PGID `428347`；iter `4146` / `model_4100`；first proof `0890be8b...b0b9` | demo-only |
| V1+V2 母本自由臂 `phase1_demo_v1v2_qdot_w2p5_face_w0p4_free_arm_seed3_20260716` | rejected | 首迭代前 malloc invalid size；rc134；旧 namespace 永不复用 | infrastructure-only |
| 普通母本保守模仿 `phase1_demo_control_qdot_w5_face_w0p4_seed3_20260716` | rejected | 首迭代前 content-bearing stale timeout；rc125；旧 namespace 永不复用 | infrastructure-only |
| 普通母本全栈 `phase1_demo_control_full_stack_free_arm_foot_w0p6_seed3_20260716` | live | PID=PGID `431061`；iter `4091` / `model_4000`；first proof `3e623c0a...8d47` | demo-only |
| qdot 母本 16 秒长回合 `phase1_demo_qdot_long_carry_free_arm_16s_seed3_20260716` | live | PID=PGID `431910`；iter `4067` / `model_4000`；first proof `1e7abe7e...21e` | demo-only |
| 自由臂基础设施重试 `phase1_demo_v1v2_qdot_w2p5_face_w0p4_free_arm_seed3_20260716_retry_v2` | live | PID=PGID `432838`；iter `3979` / `model_3900`；first proof `bb00993c...5455` | demo-only |
| 保守模仿基础设施重试 `phase1_demo_control_qdot_w5_face_w0p4_seed3_20260716_retry_v2` | live | PID=PGID `433601`；iter `3975` / `model_3900`；first proof `46afdddc...bcaa` | demo-only |

## 复现

队列与专用 runner 分别为
[`phase1_pod2_demo_hotstart_portfolio_20260716.yaml`](../../../configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml)
和 [`run_phase1_demo_hotstart_queue.py`](../../../scripts/run_phase1_demo_hotstart_queue.py)。普通 lean queue 继续
保持 fresh-only，本实验没有放宽它。

```bash
python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml plan

python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml parent-inspect

python3 scripts/run_phase1_demo_hotstart_queue.py \
  --queue configs/phase1_pod2_demo_hotstart_portfolio_20260716.yaml parent-attest
```

两条 parent 命令默认都只是 dry-run；正式执行 inspect 使用独立确认词。正式 attest 会先再跑一遍只读 inspect，
它通过后才消费 v2 snapshot namespace。receipt 与七类 SHA 已由唯一运行回填；当前 activated 配置及其
pending 反事实 fixture、终态/recipe/残留进程攻击与 assignment 测试共 `32` 个专项测试通过。`fill` 仍会按
现场容量和已有 claim fail closed。七条后代 first observed iteration 均为 `3501`。2026-07-16 04:32 CST
只读复核时七条均 live、fatal0，且七个 `model_3700` 全部通过 stable load、filename=embedded `3700`、
74 个浮点 tensor / 1,762,715 个元素全 finite（nonfinite `0`）、schema-3 hard contract、
checkpoint↔hard/claim/binding 与 `lineage_exact=0`。PID `426506/427190` 已出现 `model_4000`，但本轮未审其
内容。这些仍只是启动/checkpoint 证据，不是击球或回台行为结果，也没有产生行为赢家。

2026-07-16 05:02 CST（UTC 21:02）只读复核时七条仍 live、fatal0。PID
`426506/427190/428347/431061` 的 `model_4000` 均通过 stable load、filename=embedded `4000`、
74 个浮点 tensor / 1,762,715 个元素全 finite（nonfinite `0`）、schema-3 hard contract、
checkpoint↔hard/claim/binding 与 `lineage_exact=0`；PID `431910/432838/433601` 尚未产生该文件，记为
`UNKNOWN` 而不是失败，七条均未出现 `model_4500`。日志末窗没有明确 activation count 或 eligible
sparse-hit count，故 instrumentation 保持 `UNKNOWN`，不据零值排名或停臂；可见 fall-rate 仅作诊断，
不是正式安全结论。

2026-07-16 05:29 CST 的全 Pod2 审计又覆盖了全部 12 条而不只七个 demo：12/12 live、fatal0，latest
checkpoint 的 embedded iteration、finite、schema-3 hard、claim/binding 与 lineage 全部通过。七个 demo
中，16 秒长回合也已产生并通过 `model_4000`，所以当前是五条通过 `+500`、两条尚在 `model_3900`。
同一末窗只给出方向性信号：中拍面版的 strike/10 cm 比强拍面版高；普通母本全栈的 10 cm 指标最高但
fall 也更高；16 秒长回合因 episode 不匹配不能直接横比。所有 demo 仍缺 activation 与 eligible-hit 整数
分母，因此不得据这些窗口排名、停臂或宣布行为赢家。

## 决定

- 决定：`inconclusive`
- 是否已纳入当前 setting：`no`
- 下一个 gate：剩余两条继续到 `+500`，七条再到 `+1000`，按安全、平衡与资格充分后的同一演示卷排序。
