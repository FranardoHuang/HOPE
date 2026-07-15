# EXP-P1-DEMO-HOTSTART-PORTFOLIO — 今夜六个组合方案的严格续训

- 状态：`blocked`
- 阶段/轴：阶段 1，面向次日演示的组合方案
- 集成小目标：从三个已经学到约 3500 次更新的母本出发，尽快得到多个能兼顾挥拍、拍面和平衡的候选
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：`E1`
- 创建日期/最后复核日期：2026-07-16

`v4rg`、[`qdot-limit hinge`](../../DEFINITIONS.md#qdot-limit-hinge) 等共享术语按
[术语与人话对照](../../DEFINITIONS.md)解释；本文的“母本”就是本轮开始前保存的 model-3500 checkpoint。

## 问题与诚实边界

问题不是“单独哪个 Reward 有因果作用”，而是：在剩余一晚内，能否把已经出现方向信号的机制组合成多个
可测试候选。六条都使用严格全状态续训：policy、value function 和 optimizer 一起从母本继续；不是从零再等
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
| 资源 | 只允许 Pod2 GPU0/GPU1，按 0→1 逐圈；今晚实测上限每卡四条 |

## 六个候选

| 候选 | 母本 | 组合 |
| --- | --- | --- |
| 强拍面版 | qdot | 两项模仿放松 + qdot `-5` + 拍面 `-0.4` |
| 中拍面版 | qdot | 同上，但拍面 `-0.2` |
| 不同 basin 强拍面版 | V1+V2 | qdot `-5` + 拍面 `-0.4` |
| 自由非击球臂版 | V1+V2 | qdot `-2.5` + 拍面 `-0.4` + 非击球臂不模仿 |
| 保守模仿版 | 普通对照 | qdot `-5` + 拍面 `-0.4`，不放松两项模仿 |
| 全栈版 | 普通对照 | 两项模仿放松 + qdot `-5` + 拍面 `-0.4` + 脚朝向 `-0.6` + 自由非击球臂 |

这里比较的是“明天哪个组合更可能可用”，不是把不同母本之间的差异解释成单一机制效果。

## 启动门与停止规则

1. 先用只读 `parent-inspect` 验证三个原始 `queue_claim.json`、`run_binding.json`、checkpoint 与 hard
   contract 的完整关系：claim canonical SHA/完整 argv、binding 的 source/run/process 关系、embedded
   iter=`3500`、非空 actor/critic、非空 optimizer `state/param_groups`、finite、schema-3、checkpoint↔hard
   SHA、checkpoint↔原始 claim SHA。inspect 不创建任何文件。
2. inspect 通过后，唯一一次 `parent-attest` 用 `O_EXCL` 把 checkpoint、hard contract、原始 claim 和
   binding 复制到固定 `parent_snapshots_v2/<parent>/`，设为只读，再只从 snapshot bytes 重做同一审计并
   no-clobber 写 v2 receipt。旧 v1 receipt 路径不能解锁本队列；运行期也不再读取可变 live 母本。
3. receipt SHA 和三个 checkpoint/hard/claim/binding SHA 未回填前，机器队列保持
   `launch_authorized=false`、六行 `blocked`；不会自动解锁或重试。
4. 六条 scaleout 的 `model_500` 全部保全后，若 Pod2 GPU0/GPU1 当前各不超过三条，可先用第 4 槽发
   job1/job2，不必为此停现役。其余四组合只在按独立证据精确停止四条弱臂后逐圈补入；保留 GPU0 的
   V1-only 和 GPU1 的 foot-`-0.6`，最终仍为每卡四条。GPU2 的后续候选不塞进本 v2。
5. launch 成功前必须在日志同时看到显式 hard-contract mismatch、从 snapshot model-3500 恢复到
   iteration 3500 和 `optimizer=resumed`；在同一 GPU launch lock 内、调用 trainer 前，还要把该 job 的
   snapshot checkpoint/hard/claim/binding 四个 file SHA 与 activated queue 再对拍，关闭 verify SSH 到 load
   之间的漂移窗口。新 hard contract 必须逐值包含该行 qdot/conditional-face 权重。
6. `phase=first_iter` 还要求 run binding 的 PID=PGID/starttime/cmdline 在 `/proc` 仍是同一活进程，并且日志
   已出现第一条真实 `Learning iteration > 3500`；只有 RESUMED 行、3500 起始打印、自然退出或 PID reuse
   都失败。失败只写 exact identity 人工处置记录，不自动发信号或重试。
7. `+200/+500/+1000` 看是否启动、finite、机制是否真的激活和是否明显崩坏；只在真实击球后才有意义的
   稀疏回台指标，样本不足时继续。`+2000/+4000` 才用于次日候选排序。
8. 任一 namespace 失败都保留，不自动 replay；本批不授权第二 seed、正式晋级、真机或 broad process signal。

历史母本的完整 recipe 仍以其 canonical self-bound queue claim + run binding 账本为信任边界；本轮会反向验证
账本、argv 与 checkpoint lineage，但没有另造一份独立的旧 recipe 真源。这是已知边界，不影响本轮 v3 的
snapshot/load 时序修复。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | 证据 | 有效性说明 |
| --- | --- | --- | --- |
| qdot 母本强拍面 `phase1_demo_qdot_v1v2_face_w0p4_seed3_20260716` | blocked | 等 parent receipt/GPU release | demo-only |
| qdot 母本中拍面 `phase1_demo_qdot_v1v2_face_w0p2_seed3_20260716` | blocked | 同上 | demo-only |
| V1+V2 母本强拍面 `phase1_demo_v1v2_qdot_w5_face_w0p4_seed3_20260716` | blocked | 同上 | demo-only |
| V1+V2 母本自由臂 `phase1_demo_v1v2_qdot_w2p5_face_w0p4_free_arm_seed3_20260716` | blocked | 同上 | demo-only |
| 普通母本保守模仿 `phase1_demo_control_qdot_w5_face_w0p4_seed3_20260716` | blocked | 同上 | demo-only |
| 普通母本全栈 `phase1_demo_control_full_stack_free_arm_foot_w0p6_seed3_20260716` | blocked | 同上 | demo-only |

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
它通过后才消费 v2 snapshot namespace。parent receipt、七类 SHA 的显式回填激活和 GPU release 完成前，
`fill` 会 fail closed。专用与相邻 generic queue host 回归为 `68 passed`；本页仍没有 Pod 行为结果。

## 决定

- 决定：`inconclusive`
- 是否已纳入当前 setting：`no`
- 下一个 gate：parent receipt → 显式激活 → Pod2 首迭代 → 绝对 checkpoint receipt → 次日同一演示卷排序。
