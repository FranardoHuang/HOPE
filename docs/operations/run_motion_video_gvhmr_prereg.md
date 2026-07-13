# Franco 优先后的 static / motion 离线 GVHMR 小批

本操作只运行两个 exact、互不阻塞的 [GVHMR](../DEFINITIONS.md)（单目视频人体动作恢复）结构批：

- [`S0`](../DEFINITIONS.md)（static high-press batch）：`static/pai.mp4` 反手高点拍压一条；
- [`M0`](../DEFINITIONS.md)（motion lateral-teacher batch）：`motion/left_dang1.mp4`、`left_dang2.mp4`、`right_dang1.mp4`、`right_dang2.mp4`
  四条横移老师，固定顺序执行。

2026-07-11 的 Franco 六段已经有 exact GVHMR/GMR 结果，不得重跑。反手拉 B/C 的 frame 49/50 仅是
空挥名义视觉锚点，不是真实触球。v12 是 Jiayi 路线后续对照，本操作不授权。详细边界见
[实验卷宗](../experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)。

本操作不授权 GMR、schema-2、仿真、TOPP、RL、部署或真机。

## 1. Host 静态门

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

这只证明源码合同；不证明 Pod 上有素材、权重或空闲显存。

## 2. 分别建立不覆盖的 staging

S0 与 M0 使用完全不同的 staging、record 和 state root，可以选同一或不同 Pod。`<POD>` 是 SSH 别名；
`<POD_ID>` 只能是 `pod1` 或 `pod2`。目标根已存在就停止，不得删证据后复用。

S0：

```bash
ssh <POD> 'test ! -e /workspace/codexschema/motion_video_intake_20260713_s0 && \
  mkdir -p /workspace/codexschema/motion_video_intake_20260713_s0/raw/static'
scp "$HOME/Downloads/static/pai.mp4" \
  <POD>:/workspace/codexschema/motion_video_intake_20260713_s0/raw/static/pai.mp4
```

M0：

```bash
ssh <POD> 'test ! -e /workspace/codexschema/motion_video_intake_20260713_m0 && \
  mkdir -p /workspace/codexschema/motion_video_intake_20260713_m0/raw/motion'
scp "$HOME/Downloads/motion/left_dang1.mp4" \
    "$HOME/Downloads/motion/left_dang2.mp4" \
    "$HOME/Downloads/motion/right_dang1.mp4" \
    "$HOME/Downloads/motion/right_dang2.mp4" \
  <POD>:/workspace/codexschema/motion_video_intake_20260713_m0/raw/motion/
```

不要复制 v12，也不要把私有 MP4 加入 Git。两份 attestation 只审自己的 exact source，因此一批没准备好
不会挡住另一批。

## 3. 分别生成一次性 execution record

在包含本功能的 clean checkout 上运行。`<GPU>` 只能是 `0/1/2`；19,000 MiB 是启动前采样门，不是显存
预留。attestation 会逐字节/媒体审计本批 source，并绑定 clean GVHMR commit、完整权重树、motion Python、
`/usr/bin/nvidia-smi`、validator/argv、空 output namespace 和全新 state root。

S0：

```bash
python3 scripts/validate_motion_video_gvhmr_prereg.py attest \
  --prereg configs/motion_video_gvhmr_prereg_20260713.json \
  --pod-id <POD_ID> \
  --source-root /workspace/codexschema/motion_video_intake_20260713_s0/raw \
  --gvhmr-root /workspace/franco/motion_work/GVHMR \
  --python /workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10 \
  --record /workspace/codexschema/motion_video_intake_20260713_s0/control/gvhmr_execution_record_static_s0_v1.json \
  --gpu <GPU> --max-used-mib 19000
```

M0：

```bash
python3 scripts/validate_motion_video_gvhmr_prereg.py attest \
  --prereg configs/motion_video_gvhmr_motion_prereg_20260713.json \
  --pod-id <POD_ID> \
  --source-root /workspace/codexschema/motion_video_intake_20260713_m0/raw \
  --gvhmr-root /workspace/franco/motion_work/GVHMR \
  --python /workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10 \
  --record /workspace/codexschema/motion_video_intake_20260713_m0/control/gvhmr_execution_record_lateral_m0_v1.json \
  --gpu <GPU> --max-used-mib 19000
```

record、state 或任一本批 output namespace 已存在都会 fail closed。确需同配方重跑时必须另建版本化合同，
不能删除旧证据。

## 4. 独立启动两个 exact queue

每批只有两个 CLI 参数，没有 GPU、source、output、poll 或 timeout 覆盖面。下面的 `python3` 必须解析到
attestation 记录的相同 host Python executable/SHA。

S0：

```bash
python3 scripts/run_motion_video_gvhmr_preregistered_queue.py \
  --prereg configs/motion_video_gvhmr_prereg_20260713.json \
  --execution-record \
    /workspace/codexschema/motion_video_intake_20260713_s0/control/gvhmr_execution_record_static_s0_v1.json
```

M0：

```bash
python3 scripts/run_motion_video_gvhmr_preregistered_queue.py \
  --prereg configs/motion_video_gvhmr_motion_prereg_20260713.json \
  --execution-record \
    /workspace/codexschema/motion_video_intake_20260713_m0/control/gvhmr_execution_record_lateral_m0_v1.json
```

两批可在各自 attestation 后占不同空闲卡；任何一批失败只停本批并保留 log/binding/partial output，不改变另
一批权限。M0 内部仍按四条固定顺序，某条失败后不继续下一条。

启动后，queue 先从已核验的 staging source fd 逐字节创建
`<state_root>/bound_sources/<source_relpath>` 私有快照；文件为 `0400`、目录为 `0500`，命名保持原 stem，
GVHMR child 的 `--video` 只指向该快照。原 source 在复制前后、快照在 child 前后都核对
inode/mtime/ctime/bytes/SHA，所以 staging 路径即使被等长改写后恢复原字节和 mtime，也不会改变 child
实际消费的输入。旧 `run_motion_video_gvhmr_queue_20260711.py` 已从 `scripts/` 移除；gzip 文件只用于核对
07-11 历史结果的工具 SHA，不能作为本批入口。

## 5. 只读检查与结果边界

S0 inspect：

```bash
python3 scripts/validate_motion_video_gvhmr_prereg.py inspect \
  --prereg configs/motion_video_gvhmr_prereg_20260713.json \
  --record /workspace/codexschema/motion_video_intake_20260713_s0/control/gvhmr_execution_record_static_s0_v1.json
```

M0 inspect：

```bash
python3 scripts/validate_motion_video_gvhmr_prereg.py inspect \
  --prereg configs/motion_video_gvhmr_motion_prereg_20260713.json \
  --record /workspace/codexschema/motion_video_intake_20260713_m0/control/gvhmr_execution_record_lateral_m0_v1.json
```

完成时每个本批 output 必须有 `status=complete` binding、finite tensor、正确帧数和 output↔structural-audit
SHA；queue state 必须为 complete。通过仍只代表人体重建结构完整。

特别是 M0 不验证机器人脚距。后续 GMR/schema-2 合同必须在机器人坐标中去掉公共 root 平移、对齐朝向，
再要求末端双脚水平分离向量回到每条候选的初始 ready-window 向量（含前后错位）；更窄的合脚为失败。
