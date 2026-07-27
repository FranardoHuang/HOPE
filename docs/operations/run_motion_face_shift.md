# Canonical 五动作编译与换面检查

本页给出当前五动作候选的只读诊断、编译检查和 Agibot MuJoCo 回放方式。接口语义以
[动作预处理合同](../interfaces/motion_preprocessing_contract.md)为准，实验结论见
[五动作库正式化实验](../experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)。

这里的动作归档（NPZ）、MuJoCo 模型描述（MJCF）、机器人描述（URDF）、正向运动学（FK）和
时间最优路径参数化（TOPP）首次缩写均见[动作库术语表](../DEFINITIONS.md#动作库术语)。
诊断核心与完整候选的区别见
[`motion artifact class`](../DEFINITIONS.md#motion-artifact-class)，整桥见
[`canonical-ready contact bridge receipt`](../DEFINITIONS.md#canonical-ready-bridge-receipt)，
source/output 窗绑定见
[`protected-window digest`](../DEFINITIONS.md#protected-window-digest)。

## 1. 当前入口

正式候选链由以下仓库模块组成：

| 模块 | 职责 |
| --- | --- |
| `canonical_motion_recipe.py` | 严格读取内容绑定 recipe |
| `canonical_body_scope.py` | 构造 upper/full 身体参考 |
| `canonical_face_manifold.py` | 求解正手挡七关节拍面流形 |
| `canonical_motion_geometry.py` | 建 direct ready→core/window→ready 几何 |
| `canonical_path_topp.py` | 运动学时间初始解 |
| `canonical_torque_path_topp.py` | 固定基座力矩核；站地但根节点自由的模型仍失败封闭 |
| `canonical_schema2_builder.py` | 重建 exact schema-2 NPZ |
| `canonical_motion_compiler.py` | 搜索并原子发布五动作 × 两 scope 候选 |
| `canonical_motion_bank_gate.py` | 独立重读十件 bank 并逐门失败封闭；producer manifest 不能自证 |
| `mujoco_motion_player.py` | Agibot MuJoCo 逐帧 FK 播放 |
| `canonical_mujoco_dynamics_gate.py` | 厂商模型几何/动力学筛选与明确 non-claim |
| `canonical_motion_registry.py` | 五行 registry、授权状态机和 runtime table |

recipe 真源是：

```text
configs/canonical_motion_library_v2_20260724.json
```

旧 `syn_face_shift.py` 的单轴、固定偏移、固定切片流程不属于这条链，不能用来生成新候选。
`canonical_face_manifold.py` 输出也只属于 `diagnostic_face_core`；必须经过 canonical compiler 的
direct ready→contact→ready 整桥，才可能成为 `canonical_compiler_output`。

## 2. Pod 纪律

pod 上所有命令必须只用中央处理器并使用 `nice -n 19`。不得请求图形处理器，不得停止、重启或改变
任何训练进程。
输出目录必须事先不存在；失败结果不能覆盖或冒充成功件。

先只读核对 recipe、source、ready 和模型：

```bash
nice -n 19 test -f configs/canonical_motion_library_v2_20260724.json
nice -n 19 test -f "$CANONICAL_SOURCE_NPZ"
nice -n 19 test -f "$CANONICAL_READY_NPZ"
nice -n 19 test -f "$CANONICAL_MJCF"
nice -n 19 test -f "$CANONICAL_URDF"
nice -n 19 test ! -e "$CANONICAL_OUTPUT_DIR"
```

变量名要使用任务专用前缀，不得复用 `HOME`、`PATH` 等系统变量。

### 2.1 Ready identity 预检

当前 `canonical_ready_v1.npz` 只是一份已登记、只读保留的 donor baseline：

```text
ready NPZ SHA =
cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0
donor = bh_loop_c source frame 0
donor source NPZ SHA =
d5338168e692c8a2c19fbfac8aeb56653fa79a1f45cebc6803a460835fbc1fba
```

它不能继续被默认解释成最终共同中立位。exact FK 的
[`face-neutral ready audit`](../DEFINITIONS.md#face-neutral-ready-audit)显示，upper 从该 ready 到 scoped
反手 f44 / flipped 正手目标的拍面夹角为 `33.4/146.6 deg`，full 为 `32.9/147.1 deg`；因此 v1
明显偏反手，已失败。

当前钉住 v1 的 recipe 只能复现 probe，不能发布最终十件。下一轮正式编译前必须先：

1. 保留现有站姿、root 和拍心安全设计目标，重新联合求解右肩、肘、腕七关节的
   face-neutral ready；
2. 给新 ready 写新的 sidecar 路径、文件 SHA、donor/求解 provenance、runtime pose/
   exact-zero-velocity digest 和 ready-FK identity；不得覆盖、改写或追认 v1；
3. 由独立入口核对正反手拍面角、拍心、关节限位、碰撞和站立条件，再把 exact 新 identity 写进新版
   recipe；
4. ready identity 一旦改变，五动作 × upper/full 十件全部重编并重跑 bridge、window、time law、
   FK、动力学和行为门；旧 probe2 证书不得继承。

新 ready 的求解器、容差或独立报告尚未落地时，流程必须停在这里并失败封闭，不能让 compiler
静默回退到 v1。

## 3. Host 回归

先运行与新合同直接相关的测试：

```bash
python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_canonical_body_scope.py \
  hope_training/whole_body_tracking/tests/test_canonical_face_manifold.py \
  hope_training/whole_body_tracking/tests/test_canonical_motion_geometry.py \
  hope_training/whole_body_tracking/tests/test_canonical_motion_recipe.py \
  hope_training/whole_body_tracking/tests/test_canonical_motion_library_recipe.py \
  hope_training/whole_body_tracking/tests/test_canonical_path_topp.py \
  hope_training/whole_body_tracking/tests/test_canonical_torque_path_topp.py \
  hope_training/whole_body_tracking/tests/test_canonical_root_pose_codec.py \
  hope_training/whole_body_tracking/tests/test_canonical_schema2_builder.py \
  hope_training/whole_body_tracking/tests/test_canonical_motion_compiler.py \
  hope_training/whole_body_tracking/tests/test_canonical_motion_bank_gate.py \
  hope_training/whole_body_tracking/tests/test_canonical_protected_window.py \
  hope_training/whole_body_tracking/tests/test_mujoco_motion_player.py \
  hope_training/whole_body_tracking/tests/test_canonical_mujoco_dynamics_gate.py \
  hope_training/whole_body_tracking/tests/test_canonical_motion_registry.py \
  tests/test_canonical_motion_markers.py \
  tests/test_canonical_weighted_arc_path.py
```

marker 语义真源是 `configs/canonical_motion_marker_semantics_v2_20260724.json`（由 recipe 的
`marker_authority` 字段内容绑定）：重定时窗口 seed 取各动作 legacy ge80 种子，排名 anchor 取
nominal event（合成正手挡取 construction annotation）。torque 层与 grounded ready 需要
`scipy`（pod2 mjeval venv 已装）。

consumer 还必须单独跑：

```bash
python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py
```

测试通过只证明实现与反例合同闭合，不生成资产，也不授权训练。

## 4. 七关节换面诊断

`canonical_face_manifold.py` 的命令行只打印一份内存求解 receipt，不写 NPZ。要复现 scope-specific
结果，`$SCOPED_BH_BLOCK_NPZ` 必须先由 `canonical_body_scope.py` 的同一规则材料化；直接传 raw
source 只得到 raw-source diagnostic，不能代替 upper/full 两份证据。

这里的 `$SCOPED_BH_BLOCK_NPZ`、解出的窗口和 receipt 都是
`artifact_class=diagnostic_face_core`：不是正式 motion source/output，不含 canonical-ready→contact
bridge，也没有 `publication_class`。它们只能作为 compiler 的内容绑定输入证据，不能改名、复制到
候选目录或直接写入 registry。

```bash
nice -n 19 "$CANONICAL_PYTHON" \
  hope_training/whole_body_tracking/scripts/canonical_face_manifold.py \
  --mjcf "$CANONICAL_MJCF" \
  --urdf "$CANONICAL_URDF" \
  --source "$SCOPED_BH_BLOCK_NPZ" \
  --canonical-ready "$CANONICAL_READY_NPZ" \
  --joint-order configs/a3_runtime_articulation_joint_order.txt \
  --window 34:48 \
  --mode normal \
  --contact-frame 44 \
  > "$FACE_MANIFOLD_RECEIPT_JSON"
```

必须核对：

- active joints 恰为右肩三轴、右肘和右腕三轴；
- 15 帧每帧均有可行候选，完整窗口回溯成功；
- site position、signed raw `+Y` normal 和离散 site velocity 残差过门；
- 关节范围、逐帧 step、URDF 速度利用率过门；
- upper/full 分开求解，不能复制其中一套角度；
- receipt 的 non-claims 仍包含碰撞、逆动力学、平衡、训练、部署和硬件。

不得从输出中抽取 `right_wrist_roll_joint` 后重新包装成单轴配方；七关节向量是一个整体的流形分支。
也不得把 face receipt 的 `maximum_direct_ready_time_lower_bound_s` 当成已经生成了 ready bridge；
它只是乐观下界，完整桥必须由 compiler 和独立 verifier 从 exact output 重算。

## 5. 十件编译

正式入口是 `canonical_motion_compile_cli.py`。它只调用编译器，不复制算法；以下输入全部必填并
写入最终 receipt：

- recipe 中内容寻址的 exact schema-2 source motion，以及
  [canonical-ready 单状态旁车](../DEFINITIONS.md#canonical-ready-sidecar)；exact 9-key 表见
  [动作预处理合同](../interfaces/motion_preprocessing_contract.md)第 7 节；
  ready sidecar 不是 source motion，face-core 临时 NPZ 也不是；
- 31 关节 acceleration limits；
- full root 六坐标的位置、速度、加速度范围；
- S0 full 的显式 grounding policy；
- 搜索密度、worker 数和 thread/process backend；
- recipe、repo root 和先前不存在的输出目录。

当前 CPU probe 的完整形式是：

```bash
nice -n 19 "$CANONICAL_PYTHON" \
  hope_training/whole_body_tracking/scripts/canonical_motion_compile_cli.py \
  --recipe configs/canonical_motion_library_v2_20260724.json \
  --repo-root "$CANONICAL_REPO_ROOT" \
  --output "$CANONICAL_OUTPUT_DIR" \
  --joint-acceleration-receipt "$CANONICAL_ACCELERATION_RECEIPT" \
  --joint-acceleration-receipt-sha256 "$CANONICAL_ACCELERATION_RECEIPT_SHA256" \
  --full-root-position-lower 0 -0.4 0.85 -1 -1 -1 \
  --full-root-position-upper 0.4 0.1 1.05 1 1 1 \
  --full-root-velocity 1 1 0.5 2 2 2 \
  --full-root-acceleration 10 10 5 20 20 20 \
  --s0-full-grounding-offset-m 0.075 \
  --samples-per-scaled-unit 6 \
  --min-connector-intervals 5 \
  --min-core-intervals 5 \
  --grid-subdivisions 4 \
  --search-workers 32 \
  --search-parallel-backend process
```

这些数值是 receipt-bound candidate probe 参数，不是训练配置或完整动力学极限。先加 `--dry-run`
可验证全部输入/哈希而不创建输出；正式运行仍要求输出路径不存在。

编译器只可在五个动作的 upper/full 共十格 producer build 全部成功后原子落盘；任意一格失败时不得
留下一个看似完整的 library。原子落盘仍不等于独立 verifier 已接受 `compiler_candidate`。
`adv2c3` 只出现在搜索报告的 comparator 字段，不是默认入口或 tie-break。

producer manifest 对每格还必须出现且内容绑定：

1. `artifact_class=canonical_compiler_output` 与
   `publication_class=compiler_candidate`，两者不能用一个字符串代替；
2. canonical-ready donor source SHA/frame、ready pose digest、三组 runtime velocity
   exact-zero digest（start/end 合计六类 checks）和
   [整桥收据](../DEFINITIONS.md#canonical-ready-bridge-receipt)；
3. source/output 两个[受保护窗口摘要](../DEFINITIONS.md#protected-window-digest)，以及同时绑定
   两者、marker 映射和允许变换的 transformation receipt；
4. output 首末共同 ready、三组 runtime velocity 在 start/end 的六类 exact-zero checks 和整文件 SHA；
5. 所有 post-build gate 仍为 pending、三类 authorization 均为 false。

`canonical_motion_bank_gate.py` 或其后继独立 verifier 必须从 source/output/ready/model 的 exact
snapshot 重算以上内容，并把 no-clobber report 内容寻址。只复用 compiler 内部对象、相信 producer
JSON 或只比较文件名都不算独立验收。当前没有十件均通过并被接受的 verifier report，因此仍为
`0/10 training-authorized`。

当前时间层仍只产生运动学初始解。即使输出 manifest 写出
`PASS_COMPILER_CANDIDATE_ONLY`，也必须同时看到：

```text
publication_class=compiler_candidate
training_authorized=false
hardware_authorized=false
```

候选进入 bank registry 时还必须显式 `deployment_authorized=false`。
时间律统一称
[`no-brake time law`](../DEFINITIONS.md#no-brake-time-law)；不得把 bang-bang profile 写成数学
`C3`，也不得从运动学 envelope 外推出恒执行器扭矩。

## 6. Agibot MuJoCo 播放

每份候选先做 headless FK smoke：

```bash
nice -n 19 "$CANONICAL_PYTHON" \
  hope_training/whole_body_tracking/scripts/mujoco_motion_player.py \
  --motion "$CANONICAL_MOTION_NPZ" \
  --mjcf "$CANONICAL_MJCF" \
  --report "$KINEMATIC_PLAYER_REPORT_JSON"
```

需要人工观看时，在不占用训练图形会话的独立环境加 `--viewer --loop`。播放器只调用 `mj_forward`，
不会推进控制器；`PASS` 仅表示 schema-2 中 32 个 body 与厂商模型 FK 对齐。

候选 recipe 的 frame 是 `a3_robot_origin_ground_z0`，不是 ROS/球台消息里的 `world`。纯 FK 播放不
需要球台变换；一旦做桌网或 planner 检查，必须使用
[坐标合同中的显式双向桥](../interfaces/frames_and_coordinates.md#canonical-motion-hope-world-bridge)，
不能因为两者轴向相同就把 frame ID 当成同一个。

随后运行厂商模型 screen：

```bash
nice -n 19 "$CANONICAL_PYTHON" \
  hope_training/whole_body_tracking/scripts/canonical_mujoco_dynamics_gate.py \
  --motion "$CANONICAL_MOTION_NPZ" \
  --mjcf "$CANONICAL_MJCF" \
  --urdf "$CANONICAL_URDF" \
  --expected-mjcf-sha256 "$CANONICAL_MJCF_SHA256" \
  --out "$DYNAMICS_GATE_REPORT_JSON"
```

对站地但根节点自由的动作，站地接触力分配尚未实现时，力矩结论必须是
`INCOMPLETE_FAIL_CLOSED`。这是正确的停止信号，不得用 shell 忽略返回码后把报告记成通过。

## 7. Registry 与 consumer 验收

候选 registry 可以做
[`identity-only registry audit`](../DEFINITIONS.md#identity-only-registry-audit)，但 audit 结果禁止
导出动作 loader table。正式 runtime adapter 默认请求 `authorization_purpose="training"`，因而必须
拒绝当前 producer outputs；即使以后独立 verifier 接受为 `compiler_candidate`，仍因未 adopted 而
拒绝。本节没有把任何候选写成 adopted。

进入训练前需要同时证明：

1. 配置完成
   [`canonical runtime four-pin`](../DEFINITIONS.md#canonical-runtime-four-pin)：registry JSON SHA、
   [`motion alignment digest`](../DEFINITIONS.md#motion-alignment-digest)、canonical-ready SHA、
   canonical-ready FK 真值 SHA 四项全部精确钉住；
2. registry 从 ready donor source SHA/frame、runtime pose、三组 runtime velocity exact-zero digest、
   ready-FK 和五件
   有序 endpoint 导出同一个
   [`registry shared-ready digest`](../DEFINITIONS.md#registry-shared-ready-digest)，且该值进入
   alignment；不能只看五行复制了同一个 ready path/SHA；
3. 五个 NPZ 顺序、family、phase、face sign 和 contact opportunity 全由同一 registry 导出；
4. 每件都由独立 verifier 证明是 `canonical_compiler_output`，并绑定 source/output 两个
   protected-window digest、transformation receipt 和完整 ready bridge；diagnostic face core 不能
   进入 registry；
5. source/build/model/applicability/evidence/adoption 的 path/SHA 全部进入 alignment；重读任一字节或
   有序 row 漂移都会改变 digest；
6. [`motion evidence certificate chain`](../DEFINITIONS.md#motion-evidence-certificate-chain)从 E1 到
   row 声明等级逐级完整，且没有跨动作、scope、variant 或 NPZ 复用；
7. [`strict motion artifact parser`](../DEFINITIONS.md#strict-motion-artifact-parser)实际解析 question
   bank 和 training config；deployment/hardware 还必须解析 ONNX metadata，并以官方 ONNX parser +
   full checker 验证其绑定的 model。缺 parser、坏 bytes 或 lineage 不符都失败封闭；
8. 五个 clip 的首末姿态都是同一 ready，`joint_vel/body_lin_vel_w/body_ang_vel_w` 三组 runtime
   速度在 start/end 的六类 checks 全为零；ready 的 32-body FK 必须与内容寻址 FK 真值逐位一致；
9. `canonical_ready_mode` 的 hold 和 true reset 都从该 ready 取值；
10. 中途换 clip、随机状态初始化、post-swing replay、reset noise、yaw 扰动和 wrap teleport 均被拒绝；
   episode 内只允许在
   [`shared zero-speed ready boundary`](../DEFINITIONS.md#shared-zero-speed-boundary)换件；
11. [`motion publication state`](../DEFINITIONS.md#motion-publication-state)由独立审查从已验收
   `compiler_candidate` 单向、逐级新建 `training_adopted` record，且三类 authorization、最低 E-level
   与必需工件完全匹配；禁止改名、跳级、原地改布尔值或倒退覆盖旧 record。

在第 11 项发生前，不提供训练启动命令。代码测试、identity-only audit、compiler 输出或一份
`PASS` 报告都不能代替该晋级。

## 8. 失败时怎么处理

- face manifold 失败：增加合法候选分支或改几何，不退回单腕翻转；
- face core 通过但 ready bridge/window digest 缺失：保持 diagnostic，补 compiler/独立 verifier，
  不改名晋级；
- source/output window digest 或 transformation receipt 不匹配：按 cross-splice 硬失败处理，不从
  相邻动作、另一个 scope 或另一次 build 借窗口；
- source-anchor diagnostic 偏晚或窗口前有可避免的零加速度平台：先区分 source marker 与正式
  behavior/contact 击球真值，再修 strike-first 排名/分段 time law。正式
  [`t_hit`](../DEFINITIONS.md#canonical-t-hit) 只按动作特定范围判，不再使用通用 `0.5 s` 硬门；
  也不能用更短 [`t_cycle`](../DEFINITIONS.md#canonical-t-cycle) 掩盖错误触球锚；
- window 末前必须刹车，或跨 exact `window_end` 的 50 Hz segment 为负加速度：延长随挥、改 core
  或修离散时间律，不把窗口改成静止段；
- kinematic retimer 失败：检查路径和显式 limits，不静默裁剪；
- grounded torque/contact 不完整：实现接触力分配和连续验证，不把 fixed-base 结果外推；
- S0 地穿：用有上限的常量 Z 修正形成新资产，再重跑所有门；
- registry 或 consumer 失败：修正内容绑定和状态机，不手改 authorization；
- 任一输出失败：整套十件保持未验收、authorization=false，不先训练“看起来没问题”的子集。
