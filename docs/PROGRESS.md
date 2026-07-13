# 简短进度记录

本文件只保留短日期摘要，不再做第三份实验真源。更新时只写几句话并链接到权威位置：

- 当前 setting/采用状态：[NOW](NOW.md)
- 实验设计与证据：[experiments/](experiments/README.md)
- `main` 上的重要变化：[TIMELINE](TIMELINE.md)
- 可复现验收：[gates/](gates/)
- 缩写与人话释义：[DEFINITIONS](DEFINITIONS.md)

旧 1700 行记录完整保存在
[历史 PROGRESS](experiments/archive/PROGRESS_legacy_through_2026-07-12.md)。

## 2026-07-14

- B/C 独立 schema-2/FK prereg 的 source gate 已闭合：两份计划绑定 exact 私有 SE(2) PKL/report、
  不重叠 no-clobber 输出与 `91/98@30 Hz -> 151/163@50 Hz`；共享合同绑定 restricted pickle、formal
  donor SHA/三行 metadata 期望、vendor `1 XML + 74 mesh` closure、31-joint/32-body order，以及
  link-origin pose/COM velocity。consumer 只接受 `--hope_frame off`。两份 `static` 与专项
  `17 passed`，基于最新 `origin/main@7679b30` 的仓内回归 `782 passed, 10 skipped`；没有读取私有 PKL/ONNX、没有
  FK/schema-2/L0/L1/simulator/RL/真机。下一步仅为逐资产
  no-write runtime `inspect`。见[实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)和
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- Pod2 CPU 补跑了 MuJoCo evaluator 两个此前因本机缺依赖而 skip 的 optional 模块。首次真实收集
  `2 failed, 8 passed`，定位为 synthetic fixture 把非等价执行路径当对照、把 welded child 当可碰
  articulation；只修夹具后，同一 production evaluator bytes 在 Python `3.12.3` / MuJoCo `3.10.0`
  得到 `10 passed`。失败与通过日志/source SHA 已冻结在
  [runtime result](../configs/mujoco_eval_optional_runtime_test_results_20260714.json)；该结果不包含 policy、
  vendor MJCF、Gate3、GPU 训练或真机，G04/G06 仍为 `Partial`。

- signed-face E2 rebound exam bank 的下一层 immutable K100 source gate 已冻结：严格复用现有 schema-v3
  schedule 算法，从 exact bank SHA 重建 question ID，seed0/hold0–100/每侧无放回50/全100次分母；raw-A
  `[+1,-1]` physical-B 身份、旧纸拒绝、no-replace 和 activation-last 均 fail closed。专项攻击回归
  `14 passed`、latest-main root `747 passed, 10 skipped`、`static-validate` rc0；本任务未访问 Pod，本机缺
  exact private bank，所以 runtime `consume` 未跑，schedule/activation SHA 仍不存在，L2/judge/第二
  seed/晋级全阻断。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-PAPER.md)与
  [操作](operations/run_phase1_signed_face_exam_k100.md)。

- 非击球臂 A0/A1 直接 mask 已从设计升级为 E1 machine prereg：训练 override 同时只从位置/姿态/
  线速度/角速度四条模仿 Reward 删除左 shoulder/elbow/wrist，并用负测证明右击球臂/躯干、reward
  参数、关节/动作/力矩/接触/自碰/终止安全均不变；四项 post-override body list 已进入 checkpoint
  hard contract，A0/A1 各绑不同 SHA，去掉该唯一字段后必须完全相同。两条 fresh seed17 长臂绑定同 motion/bank/
  `4096 env × 1001 update`，默认 plan-only、root token 点火、no-clobber runtime/finalizer、
  `+200/+500/+1000` 早判；A2 固定预算继续 blocked。Pod1 A0 已以 exact PID=PGID `1811464` 运行，
  `model_200.pt` 的 iter/finite/fresh lineage/hard-contract SHA 绑定通过；旧 outer verifier 因错误要求 compact
  bank record 直含 metadata physics SHA 而假拒绝，A1 从未 claim。一次性 v1r1 continuation 已补
  `12 passed` 的 source gate：绑定 old+new control、复现旧错误、独立解析 bank metadata、先 attest 既有 A0，
  再且仅再 claim A1；禁止重跑 A0，A0/A1 漂移或预存在均 fail closed。尚无 A1 trainer、配对终档、同卷
  判读或真机。见[实验](experiments/non_striking_arm_imitation_ablation_20260713.md)与
  [操作](operations/run_phase1_non_striking_arm_imitation_a01.md)。

- MuJoCo frame/evaluator integration 的独立红队 `NO-MERGE` 阻塞已逐项关闭并合入 main：bound implicit
  改为每 substep 执行 Isaac `clip(P-D)`；被动/无 effort-limit 代理 formal fail closed；自碰只认 pelvis
  机器人子树且 formal 首次即拒绝，动态球不误报；mask 供证只接受 canonical/严格空 partial；旧
  Phase-B rider direct loader 按内容 SHA 撤销；旧 scoreboard header 不再错列追加。合入后 focused
  `147 passed, 2 skipped`、当前 main 仓内 `tests/` 为 `714 passed, 9 skipped`；两项 focused skip 都因
  本机无 `mujoco`，不是 physics 通过。本机也无 `torch`，Phase-B Torch 套件未收集。重要合同修复已记入
  [TIMELINE](TIMELINE.md)；没有运行 Pod、Isaac、vendor backend、Gate3/Gate3B 或真机。测试和剩余
  optional-runtime 边界见
  [集成卷宗](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md)；G04/G06 仍为 `Partial`。

- 第二轮独立红队又抓到两个残余假绿并在候选分支修正：可覆写 `__call__` 的 partial subclass 曾能以
  canonical `.func` 洗出 epoch 1，现逐层仅接受 exact built-in partial；自碰曾只看 control step 末态，
  现每个 MuJoCo physics substep 后 formal 首碰即拒绝、diagnostic 完整累计。两项均有 dependency-free
  攻击复现与负测；未运行 MuJoCo/Isaac/vendor/Gate3/真机，G04/G06 继续 `Partial`。见
  [集成卷宗](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md)。

- v6/v8 D 两次 pre-contract timeout 的三次低频只读审计已机器入账：两份 D 都以加载 byte-identical
  table USD（`683,433` bytes，SHA `c6fc99a8...996`）为 Kit 最后一行且未到 PhysX；相邻 C 在
  `2.339/3.031 s` 越过同一边界，v8 D 在 C clean shutdown 后 `44 s` 才启动。事后 GPU/RAM/disk/shm
  非饱和只排弱持续容量耗尽；Carbonite 残留只记相关，`dmesg` 未获权限，根因仍未证明。已冻结
  [结果 ledger](../configs/phase1_signed_face_boot_root_cause_results_20260714.json)与 design-only
  `D-first/ordinal-4 × host/private IPC` [诊断 prereg](../configs/phase1_signed_face_boot_diagnostic_prereg_20260714.json)；
  无 Pod/process/signal/training/retry/judge/部署/真机权限。专项 `8 passed`，最新 main 基线 host
  `tests/` 回归 `722 passed, 9 skipped`。

- B/C schema-2 前置审计纠正了关节列序合同：GMR `dof_pos` 与 Isaac/runtime `joint_pos` 的 31 个
  名字相同但顺序不同。新增两份内容绑定的 order 真源、双向 permutation、旧 mirror 与完整 ONNX metadata
  fail-closed validator；converter 改读合同，历史 L0 auditor 保持已被运行账本绑定的 byte-exact 源码、
  由 validator AST 复核其 target mirror。重复/缺失/额外/错序/错误长度/partial
  metadata/duplicate JSON key/NaN 负测专项 `12 passed`，基于 `origin/main@5734dc8` 的 repo 回归
  `733 passed, 10 skipped`。未读私有 B/C 资产、未跑
  FK/schema-2/simulator/RL/真机，证书仍
  为 0；见[空间重定向实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)、
  [关节接口](interfaces/joint_order_and_robot_state.md)与
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- 反手拉 B/C 的 rank-0 主选已各有独立 no-clobber 整轨站位实体化 prereg（SHA
  `e016ca74...51aee` / `27f938cd...9d454`）和 restricted-pickle consumer
  `21ebbe68...87375`。consumer 只做冻结的 proper [SE(2)](DEFINITIONS.md)，验证 xyzw 左乘、
  Z/fps/dof/non-spatial exact、可选 world velocity 同转、save/reload 逆变换、刚体距离和 report-last；
  专项 `10 passed`、全仓 host tests `656 passed, 9 skipped`；两份 exact 私有源先 inspect，后在 Pod1
  CPU-only runtime `consume`。B motion/report SHA 为 `27827912...ad6` / `a238c077...df3`，C 为
  `0dd981a6...f48b` / `b3b93d2c...f67`，最大逆误差 `<2.23e-16`。没有 simulator/RL/真机，
  schema-2/L0/vendor L1/桌网/动力学仍未跑、证书仍为 0，只解锁 schema-2 prereg。见
  [实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)与
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- signed-face exam bank 已在 Pod1 目标 runtime 完成 no-write validate 与独立 E2 发布：新 bank/report SHA
  为 `60e1a7ad...d1ca` / `dd4332ed...ad0`，24 个非 metadata 数组未变，正/反手 `183/188` 题 old/new
  output bytes 一致且 landing/net 全过。它只通过数据门；新 bank 绑定的 immutable schedule、paper
  activation、L2/judge/formal score 仍阻断。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)。

- signed-face foreign v8 使用新 source/manifest/launcher 串行跑过 A/B/C 前序，D 作为第四格又在
  900 秒内未到 hard contract/runtime verified；exact-PGID wrapper cleanup 后 rc=124，没有学习、checkpoint
  或 NaN/Inf/Traceback/OOM。继旧 v6 D 后这是第二次独立 pre-contract timeout，自动重试已停止，转入
  boot 根因；四格 activation/L2/judge/第二 seed 全 false。最终 Pod1 审计为 0 trainer/worker/judge、
  三张 GPU 空。见[机制漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)。

- v6r1 的首次真实 `validate` 在写 claim/训练前发现合同自相矛盾：immutable audit 明确 D 的
  `run_dirs=[]`，但 validator 却要求旧 would-be training path 必须存在。团队没有伪造目录；v6r1
  从未 claim、launch、signal 或训练。新 [v6r2](DEFINITIONS.md) 只发布静态源码修正：旧 path 必须
  absent，任何目录/file/symlink/special entry 都 fail closed；它只支持 `static-validate`，没有 runtime
  preflight、命令重建、launch 或 finalizer。专项 `14 passed`，合入当前 main 后仓内 `tests/` 为
  `713 passed, 9 skipped`；v6r2 明确未启动，下一步仍是第 4 格 Kit boot 根因与独立新 prereg。

- S0/M0 的下一层 exact GMR 已形成两份独立 no-clobber plan 与共享 consumer：五条 canonical-beta PT、
  converter argv、Python/pip、A3 model tree、两套 joint/body order 和 31-joint bijection 都是 required；M0
  预冻结 exact 30 Hz ready sample、足点 FK、前后/横向二维脚距、3 cm component band 与独立 5 mm 防收窄门。
  07-14 只读回执补齐 clean tree、model/mapping、关键 import 与 Python/pip SHA，但 direct retarget XML
  order/site 段被传输截断；共享 runtime 以 16 项机器清单继续 blocked，两份 batch 已预注册且真实
  `static` 均 rc=2。专项 `12 passed`、全仓 `645 passed, 9 skipped`；未运行 GMR/仿真/RL/真机。见
  [exact GMR 卷宗](experiments/motion_exact_gmr_s0_m0_20260713.md)。

## 2026-07-13

- 从现场 `50c49e5` 选择性移植 evaluator parity guard、pelvis COM→link-origin、XBODY gyro 与
  `actor_leg_ref_mask` epoch 供证到最新 main 基线；没有吞入旧分支的 `NOW`/实验状态。combined focused
  `115 passed, 2 skipped`，root suite `647 passed, 9 skipped`。这是 E1 source integration；没有新
  K100、vendor backend、Gate3 或真机结果，跨引擎 gap 仍 inconclusive。见
  [集成卷宗](experiments/2026-07/EXP-MUJOCO-EVAL-FRAME-INTEGRATION.md)。

- 反手拉 B/C 的 22 条 signed 整轨 proposal 已收敛为 exactly one primary per asset：只把 3 组
  `yaw=0` 的 R0/R1 逐字段同义项合并，随后按平移范数、偏航、回球余量、身体余隙、frame 和 ID
  冻结完整备选顺序。主选 B=`98e7b883...f3c14`、C=`aa0c86fd...f299`；只有桌/网外部几何失败可换
  下一位，schema-2/L0/vendor L1/内部动力学失败必须停止资产。专项 `13 passed`；没有物化、GMR、
  simulator、训练或真机，证书仍为 0；全仓回归 `646 passed, 9 skipped`。见
  [实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)与
  [操作](operations/run_motion_spatial_retarget_screen.md)。

- S0/M0 canonical-beta 已从 E1 计划升为 E2 runtime 结果：Pod1 的 clean detached `c3f58be` 用冻结
  Python `3.10.20` 在 CPU 上依次完成两批 `static/inspect/consume`。S0/M0 completion manifest SHA 为
  `964a7333...f1be3` / `5cef05f7...71a65`，共 `1+4` 条，五条 non-beta 内容 bit-exact，donor copy SHA
  均为 `f405ba45...4cbf2`；formal/training/hardware 仍全 false，M0 脚位/初末脚距/容差/pass 仍全 null。
  未运行 GMR、GPU trainer 或真机；下一步仅解锁独立 exact GMR prereg。见
  [canonical-beta 卷宗](experiments/motion_canonical_beta_s0_m0_20260713.md)。

- signed-face exam bank 的独立严格重绑定已完成 E1 预注册：原 train-v2 manifest 保持 byte-exact，
  generalized consumer 以封闭 profile 另行冻结旧 exam path/`63,968` bytes/SHA、split、`183/188` 题、
  旧/目标 family 与独立 no-clobber output；mutation、source-byte receipt 和双 profile synthetic rebind 为
  `18 passed`。本分支未访问 Pod 或目标 runtime，未生成 bank/report；真实 371 题 replay、从新 bank
  重建 schedule 与 judge 仍阻断，G06 保持 `Partial`。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND.md)与
  [操作](operations/run_phase1_signed_face_exam_bank_rebind.md)。

- epoch-1 signed-face v6 的 A/B/C 已到终档，D 在 `runtime_verified`/checkpoint 前 Kit boot timeout；
  旧 D launch/state/log SHA 与 dead PID/零 checkpoint 诊断、B 终档后 exact-PGID cleanup、`50c49e5`
  source bundle 与 A/B/C checkpoint audit `62076758...d354` 都已冻结。当日新增的
  [v6r1](DEFINITIONS.md) D-only validator 后续被真实 `validate` 证明错误要求一个本应不存在的旧
  training dir；它从未 claim、launch、signal 或训练，现只作 superseded evidence，修正见 07-14 条目。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- signed-face v5 在 scene 构建后、第一次学习前被旧 schema-3 train-bank physics contract 正确拒绝；
  A claim/log 保留，B/C/D 未创建，没有 checkpoint。新增严格 no-clobber 重绑定 consumer：只允许一个
  冻结 helper 的加法式源码变化，要求所有问题数组 raw bytes 不变、metadata 精确四 leaf，并在目标
  runtime 重跑 exact motion contract 与 1481 题 old/new bitwise physics replay。v1 no-write Pod
  preflight 又抓到 Python 小版本相关的 `ast.dump` SHA 假拒绝；v2 改用 helper 原始源码片段 SHA、仍
  保留同 runtime AST 等价门。v2 已发布 bank/report SHA `3a9d8851...5b71` / `9fffed03...bb37`，24 数组
  未变，两侧 landing/net 全过；v6 launcher 绑定完整 report 及父旧 bank→当前新 bank 的唯一精确
  common-field transition。专项 `32 passed`；v6 L1 尚未启动，旧 exam family 也未重绑定，故 L2/judge
  继续阻断。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- signed-face 单-seed 漏斗的首次 Pod v1 preflight 在创建 run 前抓到 checkpoint 审计假拒绝：旧代码
  只看顶层 tensor/合同键，实际 RSL-RL 权重嵌套且 provenance 在 `infos`。父 `model_13800.pt` 递归
  `74` 个浮点 tensor、`1,762,715` 元素、nonfinite `0`；v2 改为递归扫描并绑定 `infos`，保留 v1
  证据且不改四格/seed/预算/L2 blocker。v2 随后在首格学习前因 exact worktree `PYTHONPATH` 未传给
  child 退出；失败 claim/log 保留，其他三格未创建。v3 绑定 tracked setup、拒绝 local override，并
  在 claim 前解析模块来源。v3 因在 `SimulationApp` 前真正 import IsaacLab 而假拒绝；v4 改用
  `find_spec` 只验 exact module origin。v4 再在 scene 构建时发现 ignored A3 资产缺失；失败 claim
  保留，v5 从 clean `6d93bcb` 恢复并绑定 source/target `46` files、`15,378,264` bytes、tree SHA
  `0137f59b...26c6`。专项 `23 passed`；v5 Pod launch 尚未记为完成。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- 有符号拍面修复后的首轮消融已从 E0 设计升级为 machine prereg：同卡只跑 seed3 的
  hot/fresh × face-guidance-off/on 四个因果格；热启动明确保持 lineage0，fresh 必须 lineage1，半写
  claim/no-clobber/缺失 Git checkout 均 fail closed。focused `23 passed`；L1 尚无 Pod 行为结果，L2
  在 signed directional checkpoint paper 的 path/SHA 冻结前硬阻断，也没有 judge/真机授权。见
  [实验](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)与
  [操作](operations/run_phase1_signed_face_rescue_funnel.md)。

- S0/M0 exact post-GVHMR handoff 已在证据机完成，分别为 4,970/9,242 bytes、SHA-256
  `d57a93e0...a1054` / `60c55150...088ef`。下一层 canonical-beta 已做成两份独立 no-clobber prereg：
  复用旧 materializer 的 PT/save-reload 审计，只注入旧 Franco exact donor，不重算新 cohort。host static
  与新旧专项为 `15 passed, 1 skipped`，最新 main 重放回归 `620 passed, 9 skipped`；真实 PT 的后续
  consume 已按本节首条完成，GMR/schema-2/安全/效果/训练仍未授权。
  M0 的 foot sites、初末二维脚距、容差和 pass 全保持 null，必须由未来 exact GMR 产生。详见
  [canonical-beta 卷宗](experiments/motion_canonical_beta_s0_m0_20260713.md)。

- S0/M0 的五条 exact GVHMR 结果已增加 post-GVHMR no-clobber consumer：两份 prereg 同时绑定 tracked
  summary、execution record、queue state、每条 binding/audit/PT 和 canonical-beta donor，host static
  两批通过，专项 `8 passed`；后续 runtime handoff 与 canonical-beta consume 已按本节其他条目完成，
  GMR/schema-2 仍未运行；S0
  禁止借用拉球题，M0 后续必须恢复含前后错位的初始二维脚间向量，双脚并拢不算成功。详见
  [实验卷宗](experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)与
  [操作文档](operations/run_motion_post_gvhmr_exact.md)。

- Franco 动作主线第一次从“排队”进入 runtime：Pod1 上 S0 高点拍压 `88/88` 帧、M0 四条横移
  `105/105、97/97、82/82、96/96` 帧全部通过 exact GVHMR finite structural audit；输入、execution
  record、queue、output、binding 和 audit SHA 已进入
  [`motion_video_gvhmr_s0_m0_results_20260713.json`](../configs/motion_video_gvhmr_s0_m0_results_20260713.json)。
  同时 signed spatial-retarget 对真实 v5 输入完成 640-cell screen，反手拉 B/C 分别产生 `19/3` 个
  bounded proposal，但 certificate 仍是 `0`，所以只解锁物化/安全门，不解锁 TOPP、RL、Gate3 或真机。
  详见[GVHMR 小批](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)和
  [空间重定位](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)。

- 反手拉 B/C 的 signed spatial-retarget 首次对真实 v5 输入点火，在生成 proposal 前抓到验证器
  schema 假拒绝：`capture_table_pose_observed=false` 位于 `frame_contract`，而旧代码误从只含
  path/bytes/SHA 的 `frame_contract_evidence` 读取。修复后仍同时绑定 evidence SHA，且缺失/true
  fail closed；新 prereg/tool SHA 为 `0f757c8c...af66a` / `d053dd50...5259b`。这只解除输入验证阻塞，
  尚不是动作晋级。详见[动作空间重定位实验](experiments/2026-07/EXP-MOTION-SPATIAL-RETARGET.md)。

- 复盘 Phase-1 的 GPU 证据购买方式：`SZ` 在 2k 已失去稳定性资格后，把四 seed 都继续买到 4k
  对拒绝 baseline 属于过量复现。新制度改为一个阻断 seed 先跑四个不同机制单元，固定
  相对 `+200/+500/+1000` checkpoint；只有胜者和匹配对照补第二 seed，`3–4` seed/terminal 只给正式候选。
  第一张新纸是“热启动/从零 × 线性拍面引导关/开”的四格；当时只有 E0 设计，尚未启动 Pod、训练、
  judge 或真机。详见[机制漏斗](experiments/2026-07/EXP-P1-SIGNED-FACE-RESCUE-FUNNEL.md)和
  [算力制度](research/phase1_ablation_acceleration_2026-07-11.md#seed-是晋级税不是首轮并发单位)。

- signed-face 诚实门已在 feature source 闭合：Isaac virtual reward 与 NumPy/MuJoCo analytic
  scorer 都在 `orient_normal` 前绑定 raw-A、每 clip `[+1,-1]` physical-B 和严格 +X/hemisphere
  门；`n/-n` 负控证明同一冲量/落点下错面不再记分，旧 unsigned 路径只能显式 inexact。seed3
  TensorBoard 九个 milestone 的 content-bound 摘要又显示正手误差 `174.02°`/normal pass `0` 时
  训练回台仍 `.965`；实际 `env.yaml` 绑定启用的 face-blind reward 及 `20/30/5/5` 权重。step13800
  的 `2.961×` 只是跨环境/正反手的全局 reward-tag 比值，不能量化正手错面支付份额；准确结论收紧为
  “wrong-face FH states were treated as reward-eligible by the active face-blind reward path”，而非
  “已量化错面支付”或单因素因果。focused 为 `38 passed, 1 skipped`，
  顶层 broad 为 `546 passed, 9 skipped`；
  没有 simulator/Pod/真机行为结果，fresh canary 与同卷复判仍待做，G05/G06 保持 `Partial`。详见
  [拍面符号卷宗](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- Fresh `SZ model_4000` 四 seed 同一 K100 已通过 Linux fake-runner 冒烟、两 Pod 一次性持久
  启动和正式 aggregate 完成：`50/88/98/0`，median `.69`、worst `.00`、spread `.98`、
  worst-side `.00`，四项稳定门全失败；seed4 有 21 次 root fall，判为持续弱而非晚熟。
  seed2/3 正手 parsed `38/50,48/50` 但 signed composite 均 `0/50`，法向误差
  `172.33°/174.35°`，所以旧高分不晋级。aggregate file/content SHA 为
  `1ba88e39...d195` / `226e6050...648d`；详见
  [稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。
- 动作离线顺序纠正为 Franco 主线优先：六段旧素材复用 exact GVHMR/GMR，不再重跑；反手拉 B/C 的
  frame 49/50 只登记为空挥名义视觉锚点。新视频拆成互不阻塞的 [S0/M0](DEFINITIONS.md) 离线结构批：高点拍压单条与四条横移候选，
  v12 本轮不授权。两批绑定 GVHMR/权重/Python/`nvidia-smi`/validator/argv，并用 batch-only source
  fd 生成私有只读快照供 child 消费，再以 inode/mtime/ctime/SHA 复核；同时拒绝 symlink、原子 claim
  不相交 state/output。07-11 旧 launcher 已压成仅证据 gzip，不再提供通用入口；M0 未来终点必须恢复初始、
  朝向对齐且含前后错位的双脚分离向量。Host 聚焦套件 `50 passed`，仓库 `tests/` 为
  `573 passed, 9 skipped`；本分支未复制 Pod、未启动
  GVHMR/GMR/simulator/RL/真机。见
  [实验卷宗](experiments/motion_video_gvhmr_prereg_franco_static_motion_20260713.md)与
  [操作文档](operations/run_motion_video_gvhmr_prereg.md)。
- Phase-1 fresh 广度池分两波完成负责人批准的运营收口：16 臂已全部在保留日志并验证最后
  checkpoint 的迭代、`1,762,715` 个浮点元素 finite、schema-3、fresh lineage 与相邻合同 SHA 后，
  只按各自登记 PGID 停止。第二波前又确认 24/24 最近 K20 格的正手 signed composite 都为 0；
  TERM 未退出时仅在确认无 live child/Kit-lock holder 后对同一
  exact PGID 使用 KILL，没有 broad kill、worker/judge 信号或真机命令。这不是预注册 q10/q50
  阈值停止结论，旧 `screen_only`/`whole_arm_stop_allowed=false` 语义不变；完整曲线、PGID 和 checkpoint
  SHA 见[拍面×plant 广度实验](experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md)。
- MuJoCo pelvis 点/轴 frame 审计在 `codex/mujoco-com-reset-frame` 修正两处源码合同：每个
  motion clip 显式声明 COM/link-origin 线速度点，teacher-reference 只对 COM 做 rigid-point
  转换，含糊的旧 inexact 包拒绝 reset；actor `base_ang_vel` 从 MuJoCo inertia-principal axes
  改为 pelvis link/IMU axes，并对 pelvis 自身恰好一个、零地址 freejoint fail-loud（不禁止球等
  其他 free body）。真实 A3 MJCF 的 formal CPU group 为 `115 passed, 0 skipped`，完整合同 union
  为 `183 passed, 0 skipped`，支持的根目录 `tests/` 为 `554 passed`；10 秒 plain-MuJoCo PD stand
  为 `1.816 mm` z 漂移、`0.311 deg` 最大倾角、
  双脚接触 `100%`。没有在 Pod/vendor backend/真机上运行 policy rollout；ready-state 四格仍未运行。
  两轮独立 review 复核公式、MuJoCo BODY/XBODY/freejoint 语义、mixed/count 负控和 standalone
  old-donor 兼容后均无 P0/P1/P2。
  另登记 vendor ROS 非零 `SimReset` world-angular→body-qvel 的潜伏接口 bug，当前全零 keyframe
  路径不触发。详见 [G06](gates/G06_isaac_to_mujoco.md) 和
  [frame 合同](interfaces/frames_and_coordinates.md)。
  同日只读复核用户给的两个 Pod：一台 SSH 握手连续 reset；另一台 3 张 RTX 5090 全空闲、无
  train/eval 进程，`/workspace/franco/nohope` 停在 `16a94b1`，其未刷新的 `origin/main` 也仅到
  `7b85546`。所以这两台当前都没有运行或验证本 ticket，不能把本地源码通过当成云上训练结果。
- exact planner-policy tuple 源码已在 latest-main 集成候选中闭合：23 项有效源码/配置逐字节匹配
  `c0a8e46`，portable Release 为 focused `40/40`、native `233 passed + 5 optional skips`，主线本地
  回归为 planner `180 passed, 2 skipped`、serve `39 passed`、root `521 passed, 9 skipped`。这只关闭
  source/binary merge blocker；ROS/Jazzy/AimRT、formal ONNX runtime、backend first tick、vendor
  MuJoCo 和真机都未运行。详见
  [实验卷宗](experiments/2026-07/EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD.md)。
- 最新 main 登记并在本地逐字节核验了 7 段私有新视频：v12 正反手挡球、高点拍压第五动作，以及
  左右横移各两段下肢老师。新版 intake 合同能区分挥拍与横移动作，拒绝重复 JSON 键、非有限数和
  角色/动作错配；7/7 文件与 11 项专项测试通过，仓库测试为 `472 passed, 9 skipped`。同时建立了
  [动作组合设计](experiments/motion_v12_high_press_lateral_teacher_20260713.md)、
  [视频 intake 记录](experiments/motion_video_intake_v12_static_motion_20260713.md)和
  [非击球臂消融](experiments/non_striking_arm_imitation_ablation_20260713.md)。这只证明素材登记和
  设计已落地；没有复制到 Pod，也没有 GVHMR/GMR、仿真、RL 或真机行为结果。
- 按负责人阅读路径重写 [NOW](NOW.md)：先解释题目、参考动作、179 维输入、31 维关节目标、
  Reward、PPO 和独立判卷怎样组成一套完整训练，再按现行课程逐段写问题、解法、效果与差距。
  同时纠正阶段编号：阶段 2 是虚拟球变到达状态，站位/脚步是其中的解法；阶段 3 才是物理球
  进场；连续恢复和 `Gate3/Gate3B` 分别是横向能力线和部署验证线。成绩卡明确为 Python
  BankExam 单拍解析诊断，不是 Gate3。本次只改文档，没有新增训练、仿真行为或真机结果。
- Fresh `SZ model_4000` 四 seed 同卷的 Pod1/Pod2 readiness audit 与 all-four activation
  已物化（activation file `9dea76c2...ce704`，content `eaa92ca2...aa4fb`），两 Pod
  `contract-check` 通过。随后两份 no-clobber runtime contract 已完成 `prepare`；Pod1/Pod2
  file SHA 分别为 `2b76a5a...8201e`、`dbecc102...d1c9b`。当前仍是
  `prepared_not_started/jobs_started=0/auto_start=false`，没有 run、judge、新分或真机动作；该
  readiness/prepare 事务当时未发 trainer signal，后续 8 臂运营停止是本节首条记录的独立决定。
  持久监督器 source gate 后续已审绿，仍缺 Linux fake-runner smoke 与正式 job。详见
  [Fresh SZ 稳定性实验](experiments/2026-07/EXP-P1-FRESH-SZ-STABILITY.md)。
- `model_4000` 同卷启动新增一次性、无覆盖的持久监督器：父进程只在核对 PID=PGID、procfs 身份、固定环境和完整 SHA 闭包后发布不可逆 token；token 可见后的超时、证据 `stat` 或临时清理异常都只能报告 committed-pending，不能产生重试权限。supervisor+queue+consumer 为 `64 passed`；这仍是 host 源码门，Linux/Pod 与 MuJoCo judge 尚未运行。详见[执行卷宗](experiments/phase1_fresh_sz_model4000_q50_20260713.md)。
- Native MuJoCo feasibility/implementation 已确认为 P0，但不阻塞几天内 `Gate3-D0`。off-main
  preflight `6e5fce3` 的 63 项 focused test、顶层 `468 passed, 9 skipped` 和七个 false 授权位
  证明 fail-closed；red team 同时抓出 action trace、source alias/exec、strict JSON、MJCF
  `strippath` 四个高优先级正确性缺口，所以当前 `NO-MERGE`。single-env core 未来还必须过
  N=1/8/32/64 与 48 小时留 30% 余量的吞吐继续门。它不是 trainer、`VecEnv`、PPO smoke 或训练结果，详见
  [实验卷宗](experiments/2026-07/EXP-MUJOCO-NATIVE-TRAINING.md)。
- 正手拍面复核纠正了“所有 seed 都约 170°”的旧说法：model-2000 seed1/2/3 raw-A 误差为
  `171.10/172.94/173.39°`，seed4 没有正手 exact strike；解析回球器的 `orient_normal`
  可能抹掉正负号。signed-face 诚实门通过前，旧解析高分不用于晋级；详见
  [拍面符号卷宗](experiments/2026-07/EXP-P1-FACE-SIGN-FORENSIC.md)。
- 连续拍等待/恢复设计经原文和现役代码复核后收紧：T0 按周期换题，T1 只改事件驱动结构并冻结
  reward，T2 才允许 learned shaping；随机到球先作为环境轴。若 T1 失败，先做平衡债/ready
  potential 的配对 `2^2`，第三 critic 只有独立校准并通过隔离 q50 后才能进入 `2^3`。
  这次只收紧文档设计边界；现有 machine prereg/validator/operation 仍固定旧三 reward/full `2^3`，
  必须另做内容寻址同步后才能点火。没有训练、simulator、Pod 或真机行为结果；详见
  [连续时序审计](research/phase1_continuous_rally_timing_2026-07-11.md)。
- 新动作的冻结站位 `0/64` 不再解释为“动作无效”：正式问题是动作自身安全触球流形 × 适配来球/动作题族
  × 合法整轨 `SE(2)` 站位。反手拉 B/C 仍只到重定位候选，挡球需另出题；没有重跑 screen。
- `main@3c7e507` 先补回了缺失的 INDEX 和实验账骨架；本分支把它升级为中文一站式路由、
  术语人话表、逐实验卷宗、精简 NOW/TIMELINE/PROGRESS、唯一队列和算力纪律。合入 `main`
  前，新版 NOW 仍只是一份提案。本次文档迁移没有运行训练、simulator、Pod 进程或真机。

## 2026-07-12

- 完成 native MuJoCo `Trainer-v0` 只读 preflight：现役 vendor main 的 sim loop 没有球/球台/网，
  所以首卷只做单拍 balance/strike-state fine-tune；reward 用独立 replay oracle；warm start 只载 actor，
  critic/optimizer 全新。没有启动 backend、sim、Pod 或真机，详见
  [preflight](research/mujoco_training_v0_preflight_2026-07-12.md)。
