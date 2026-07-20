# Franco 优先、static 与 motion 的 GVHMR 预注册

- 状态：`completed`（本卷的 S0/M0 GVHMR 结构批已完成；后续 exact GMR 诊断已回收，
  schema-2 仍未授权）
- 人类负责人：Franco
- 执行者：Codex
- 工作分支：`Franco_codex/motion-gvhmr-prereg-20260713`
- 创建日期/最后复核日期：2026-07-13 / 2026-07-20

## 决策与最小目标

动作主线按用途而不是录制版本号排序：先消费 Franco 自录的正/反手 × 拉/挡四类动作，再处理反手高点拍压
第五动作和横移下肢老师；v12 只是 Jiayi 路线的后续代表对照，本轮不授权执行。

2026-07-11 的六段 Franco 源视频早已完成内容寻址的 [GVHMR](../DEFINITIONS.md)（单目视频人体动作
恢复）、GMR、落地和 240 Hz 安全诊断，所以 F0 不重跑昂贵结构提取。反手拉 B/C 的人工名义视觉锚点
补录为 frame 49/50；空挥没有球，`contact_truth`（真实触球时刻）仍为空。F1/F2 的动作专属题族、
合法整体 `SE(2)` 站位、schema-2、L0/L1、桌网、动力学和平衡证书另行闭环。

新视频的最小可并行工作只包含两个互不阻塞的离线结构批次：

- [`S0`](../DEFINITIONS.md)（static high-press batch）：仅 `static_backhand_high_press`（反手高点拍压）；
- [`M0`](../DEFINITIONS.md)（motion lateral-teacher batch）：仅四条横移老师，顺序为 left-1、left-2、right-1、right-2。

两批各自拥有不相交的 execution record、state root 和 output namespace；一批失败不授权重试，也不阻塞另一
批。本 GVHMR 结构批通过本身不自动授权 GMR、schema-2、仿真、TOPP、RL、部署或真机；后续
exact-GMR 已按独立合同完成结构诊断，授权边界仍以对应卷宗为准。

## 内容绑定

- Franco 旧 intake：`configs/motion_video_intake_20260711.json`，SHA-256
  `6661f9ac3dba930835a8c241ecb4a779d62be0cbb9aa4d1712d5470db6c9b289`；
- Franco GVHMR results：`configs/motion_video_gvhmr_results_20260711.json`，SHA-256
  `0aaee40d689033518f24e9667478142398ae370f3bbde73e1638c0bfb321fd86`；
- Franco B/C 名义视觉锚点：`configs/motion_video_franco_backhand_loop_visual_review_20260713.json`，
  SHA-256 `9e2a7a51c443d53d7b8ed5c39d02ca0a523f59eb195b1cfc2a335041126498f1`；
- 新视频 intake：`configs/motion_video_intake_20260713.json`，SHA-256
  `44b00b3c46c837d797990bc6f6255055c0ff83c1bb8643ca81f9707033ca304c`；
- 新视频人工事件窗口：`configs/motion_video_manual_event_review_20260713.json`，SHA-256
  `6a79dcc4528293728226192513f1d9ce5f266b35291a5504dea4559d7fc8049d`；
- S0 预注册：`configs/motion_video_gvhmr_prereg_20260713.json`，SHA-256
  `c610366e7e382b20f9b64b01a9c57b2722b72be501ada2aa16c24e350207f1ba`；
- M0 预注册：`configs/motion_video_gvhmr_motion_prereg_20260713.json`，SHA-256
  `19794d62446335c2d125564d9ea7ee77e59e1aed39f7aeef4b4039843dce0f08`。

Franco 反手拉旧反事实结果只用于排序后续筛选：B 为 frame 49、`32/32`、距最近旧题 `0.164936 m`；
C 为 frame 50、`27/32`、`0.236505 m`；A 为 `1/32`、`0.775391 m`。其他三类 Franco 动作在不匹配的
共用旧题上为零，不能据此淘汰；它们需要各自动作题族，只有 A/B/C 互为同一动作候选。

## 运行与安全闭包

S0/M0 共用 exact clean GVHMR commit `6ec3ca39336c50492c0fae65fba2fb831fc7d866`、21 文件
checkpoint/body-model 树、固定 motion Python、`/usr/bin/nvidia-smi` 和结构审计器。secure queue 只接受
这两个 committed prereg 与各自一次性 execution record；运行参数不能从 CLI 覆盖。

每批只审计和绑定自己的 source，因而未复制 v12 不会挡住 S0/M0。队列从 `O_NOFOLLOW` source fd
逐字节建立 batch 私有、只读、no-clobber 快照，先后复核原 source 与快照的 inode、mtime、ctime、bytes
和 SHA；GVHMR child 只读该快照路径，不再读可变 staging 路径。state/output 使用原子 no-clobber claim
与跨 state lock。2026-07-11 旧 launcher 只保留 gzip 历史源码证据，不再是 `scripts/` 下的可执行入口。
两批可在分别通过 exact attestation 后使用同一或不同 Pod 的空闲卡。2026-07-13 已在 Pod1 使用
GPU 1/2 分别执行 S0/M0；每批只消费自己的私有快照，没有复用或覆盖旧 namespace。

M0 的 GVHMR 输出只证明人体结构重建。未来机器人坐标合同必须去除公共 root 平移、对齐朝向，再要求末端
左右脚水平分离向量回到该候选初始 ready window 的鲁棒向量，包含前后脚错位；更窄的“合脚”不能替代。

## Host 验收

```bash
python3 scripts/validate_motion_video_gvhmr_prereg.py static \
  --prereg configs/motion_video_gvhmr_prereg_20260713.json
python3 scripts/validate_motion_video_gvhmr_prereg.py static \
  --prereg configs/motion_video_gvhmr_motion_prereg_20260713.json
python3 -m pytest -q \
  tests/test_validate_motion_video_gvhmr_prereg.py \
  tests/test_run_motion_video_gvhmr_preregistered_queue.py \
  tests/test_run_motion_video_gvhmr_queue.py \
  tests/test_audit_motion_video_intake.py \
  tests/test_audit_gvhmr_result.py
```

2026-07-13：两份 static contract 均通过；聚焦套件 `50 passed`，仓库 `tests/` 为
`573 passed, 9 skipped`。随后两份 exact execution record 和 queue 都在 Pod1 完成：

| 批次 | 输入 | 结构结果 | 结论 |
| --- | --- | --- | --- |
| S0：反手高点拍压 | `static_backhand_high_press` | `88/88` 帧，`6,952` 个所需 tensor 元素 finite | GVHMR structural pass |
| M0：横移老师 | left-1 / left-2 / right-1 / right-2 | `105/105`、`97/97`、`82/82`、`96/96` 帧，合计 `30,020` 个所需 tensor 元素 finite | 4/4 GVHMR structural pass |

结果总账为 `configs/motion_video_gvhmr_s0_m0_results_20260713.json`，SHA-256
`08b5e8338ac07a20f18034811167c941fed7168703cad4308bc8b2f1e0569726`；它绑定 source、execution record、
queue state、output、binding 与 structural audit 的逐文件 SHA。该通过只说明 SMPL-X 结构和有限数完整，
在本结构批形成时还没有 GMR、schema-2、脚接触/末态站距、桌网、自碰、动力学、simulator、RL 或真机
结果。后来回收的 exact GMR 只关闭了其中的结构诊断：S0 仍缺高球效果题族，M0 末态 stance
gate 为 `0/4`，schema-2 及其后各门仍未开。当前结论见
[exact GMR 卷宗](motion_exact_gmr_s0_m0_20260713.md)。

## 当时下一步与当前决议

1. post-GVHMR handoff、canonical-beta 与原 exact-GMR v2 namespace 都已完成；不得重跑或覆盖。
2. S0 先建独立高球拍压题族；结构结果不能提前当成击球有效性。
3. M0 保留所需左右位移，同时修复末态回到该动作自己的初始 stance；新候选必须使用新版本 no-clobber 合同。
4. F1/F2 继续消费既有 Franco 结果；v12 只在上述主线有可比证据后作为 Jiayi 对照进入新版本合同。

GVHMR 命令见[结构批操作文档](../operations/run_motion_video_gvhmr_prereg.md)；结果消费见
[post-GVHMR 操作文档](../operations/run_motion_post_gvhmr_exact.md)。
