# v12、高点拍压与横移动作视频登记

- 状态：`completed`
- 人类负责人：Franco
- 执行者：Codex
- 工作分支：`Franco_codex/new-motion-batch-20260713`

## 问题与决策范围

验证 `${HOME}/Downloads/{v12,static,motion}` 下新录制的七段私有视频是否存在，字节和媒体属性是否精确匹配，
并记录它们的预期用途；不因此宣称任何动作安全、有效或可用于训练。

## 输入与不可变绑定

只含元数据的 manifest 为 `configs/motion_video_intake_20260713.json`（SHA-256：
`44b00b3c46c837d797990bc6f6255055c0ff83c1bb8643ca81f9707033ca304c`）。它绑定七个 HEVC、1920x1080、30 Hz 的文件，
每个文件为 82–105 帧：

- v12 正手/反手挡球：Jiayi 主线要求的 v 系列首要候选；
- `static/pai.mp4`：针对高击球点提出的右手机器人反手高点拍压第五动作；
- 两个左移和两个右移候选：作为下肢移动老师的输入。

原始视频仍为私有且只存在于本地。各文件的字节数和 SHA-256 值均记在 manifest 中；本记录有意不复制完整表格。
验收用审计器的 SHA-256 为
`ffdae64ac3437a3d962eb006eadc9d4d429c4a14e41484c6ef9a594b596fc299`。

## 设计与对照

素材登记清单的 `schema_version: 2` 把挥拍视频与横移下肢老师分开；这不是机器人动作 NPZ 的
`schema-2 motion`。移动老师不带正手/反手或挥拍标签，不得被静默地当作挡球片段消费。
现有 `schema_version: 1` Franco/v6/v7 manifest 仍然有效，且保持字节兼容。

这些标签编码的是用户提供的假设，不是已测得的性能。特别是，在 v12 通过与任务匹配的考卷前，不将它称为最好的 v 系列动作；
高点拍压也不在拉球/挡球考卷上评分。

## 验收与失败规则

只有当每个相对路径都位于给定根目录下，且字节数、SHA-256 和 ffprobe 字段均匹配时，素材登记才通过。
重复 JSON 键、NaN/Infinity、不安全路径、角色/动作不匹配，以及候选排名不完整都必须关闭失败。素材登记不授予算力、仿真器、部署或硬件授权。

## 复现

```bash
python3 scripts/audit_motion_video_intake.py \
  --manifest configs/motion_video_intake_20260713.json \
  --source-root /Users/Franco/Downloads
python3 -m pytest -q tests/test_audit_motion_video_intake.py
```

## 结果

2026-07-13，七个本地文件全部通过精确字节/hash/媒体验证。聚焦测试套件通过 `11` 项测试；仓库测试套件通过 `472` 项，
跳过 `9` 项。没有视频被复制到 Pod，也没有运行 GVHMR、GMR、仿真器、RL 或硬件流程。

## 局限与未宣称事项

尚无球接触真值、击球帧、镜像/坐标系证明、schema-2 机器人动作、桌/网间隙、动力学、平衡或厂商 MuJoCo 结果。
空挥视频的语义不能证明某个挥拍能打回哪些来球。

## 决定与下一步

在未来预处理队列中，先处理 v12，然后是高点拍压和横移候选；但在当前更高优先级的 q50/planner/拍面符号收口完成，
且离线动作门拥有经复核的下游消费者之前，不启动任何一项。详细实验设计见
[`motion_v12_high_press_lateral_teacher_20260713.md`](motion_v12_high_press_lateral_teacher_20260713.md)。
