# 独立任意 N 动作 canonical bank

这条工序把一个固定顺序的源动作胶囊编译成每动作 `upper` 和 `full`
两件 canonical 候选。它是独立 bank，不是 canonical-five 的追加模式，也不会修改
canonical-five 的 recipe、默认动作顺序或 gate 行为。术语见
[`DEFINITIONS.md`](../DEFINITIONS.md)。

这条链只解决“怎样完整、可复查地生产任意 N 动作候选”。**编译成功和 bank gate
通过都不等于动作能把对应来球打上台，更不授权训练、部署或真机。**

## 输入和输出

输入必须是：

1. `materialize_chingmu73_source_capsule.py` 生成的
   `SOURCE_CAPSULE_RECEIPT.json`，其中动作顺序、每件源 NPZ、metadata、击球帧、
   `t_hit`、`t_cycle`、站位与 base spawn 都逐字节绑定；
2. 一份既有 canonical compiler template；
3. template 自己绑定的 canonical-ready，以及独立的 source-hold 动作；
4. 逐关节加速度包络 receipt；
5. 明确的 marker、准备段、恢复段与 full-root 编译参数。

`materialize_arbitrary_motion_bank_recipe.py` 从 capsule 派生固定动作顺序，写出
`canonical_arbitrary_n_recipe_v1`。输出必须事先不存在；发布用原子 hard-link，
并发时后到者失败且不会删除先到者的文件。

`canonical_motion_arbitrary_bank.py` 复用现有 compiler primitives，要求精确生成
`N × 2` 件：

```text
motion_0 upper
motion_0 full
motion_1 upper
motion_1 full
...
```

每件输出、sidecar、顺序、SHA-256、编译后 `t_hit/t_cycle` 和最短恢复时间都写进
同目录 `BUILD_MANIFEST.json`。缺任一动作、scope、sidecar 或内容漂移都失败。

## 坐标与站位合同

73 件源动作的采集站位不同，绝不能把这些世界坐标直接作为 policy task。producer
逐动作重开 metadata 并检查：

```text
base_spawn_center_w_xy_m = station_xy_hope_m + [0.5, 0.7625]
```

运行时合同固定为：

- source 动作是 action-local，不携带可供 policy 使用的绝对世界站位；
- ball、contact、racket task 和 base goal 都用该回合**实际 base spawn/yaw**的局部坐标；
- `no_move` 的 goal 等于该回合实际 spawn，不等于另一个动作的站位；
- `move` 的 goal 是相对实际 spawn 的局部位移；
- 对 base、ball、contact 与 task 同施平面平移/偏航时，base-local task 必须不变；
- 只交换两个动作的 station/base，而没有对 ball/contact/task 施同一个平面刚体变换，
  必须拒绝。

host 测试包含平面刚体变换等变性和跨动作 station swap 负例。

## 三步运行

先生成 recipe。下面只展示参数形状；正式 N73 必须使用合入当前分支后的
`assets/motions/chingmu73_20260728/CLIP_ORDER.json` 与 tracked bytes 重新生成 source
capsule，并在命令中填入现场重算的 SHA-256。

```bash
python3 hope_training/whole_body_tracking/scripts/materialize_arbitrary_motion_bank_recipe.py \
  --repo-root . \
  --bank-id <independent-bank-id> \
  --source-capsule <SOURCE_CAPSULE_RECEIPT.json> \
  --expected-source-capsule-sha256 <sha256> \
  --compiler-template <compiler-recipe.json> \
  --expected-compiler-template-sha256 <sha256> \
  --source-hold-motion <source-hold.npz> \
  --expected-source-hold-motion-sha256 <sha256> \
  --acceleration-receipt <joint-acceleration-receipt.json> \
  --expected-acceleration-receipt-sha256 <sha256> \
  --output <new-recipe.json>
```

在动用 MuJoCo/编译预算前先做 source-only dry validation：

```bash
python3 hope_training/whole_body_tracking/scripts/canonical_motion_arbitrary_bank.py \
  --repo-root . \
  --recipe <new-recipe.json> \
  --output <must-not-exist-output-directory> \
  --dry-run
```

只有 dry receipt 明确显示固定顺序、`candidate_count=2*N`、所有授权为 `false`，
才允许在一个全新目录实际编译：

```bash
python3 hope_training/whole_body_tracking/scripts/canonical_motion_arbitrary_bank.py \
  --repo-root . \
  --recipe <new-recipe.json> \
  --output <new-bank-directory>
```

## 独立 generic-v2 gate

实际编译后，先生成 schema-v2 generic registry、逐件连续 swept-clearance receipt
以及模型身份，再用独立进程运行：

```bash
python3 hope_training/whole_body_tracking/scripts/canonical_motion_generic_bank_gate.py \
  --manifest <new-bank-directory>/BUILD_MANIFEST.json \
  --bank-dir <new-bank-directory> \
  --recipe <new-recipe.json> \
  --repo-root . \
  --registry <generic-registry.json> \
  --expected-registry-sha256 <sha256> \
  --mjcf <model.xml> \
  --urdf <model.urdf> \
  --body-order <body-order.txt> \
  --expected-compiled-signature <sha256> \
  --swept-clearance-receipt <swept-clearance.json> \
  --expected-swept-clearance-receipt-sha256 <sha256> \
  --output <new-bank-gate-report.json>
```

wrapper 在一个进程局部临时注入任意 N matrix，再调用既有独立 gate 的 FK、动力学、
持久时间律、grounded left/midpoint/right 和连续 swept-clearance 检查，最后恢复旧
canonical-five 常量。它应作为独立 CLI 进程运行，不应与另一个 in-process legacy
gate 并发共享 Python 模块。

report 会绑定 wrapper 自己的路径/SHA。当前 motion admission 的工具信任集尚未采用
这个新 producer，因此 report 即使完整通过，admission 也应继续 fail-closed。

## 2026-07-29 本地旧 capsule dry 证据

ignored 本地 capsule
`chingmu73_f10_exact_v1_consumer_v1_local` 已通过 source/compiler-input dry：

- 动作数 `73`，候选矩阵 `146`；
- recipe SHA-256
  `69e1b6e0ca8be6630c9659751f070f700067148edf1e32bc0557a0c16483a486`；
- producer SHA-256
  `f831f361b1f9ef0ccd643292f30f6bc6859a9d56084d524ae360ca78e6ac7c73`；
- placement contract 已写入 receipt；
- `compiler_outputs_present=false`、`bank_gate_pass=false`；
- training/deployment/hardware authorization 全为 `false`。

这份本地 capsule 绑定旧 action config 与旧 solver pin，只证明 schema、顺序和 dry
consumer 可以工作。正式 N73 必须在 main 的 tracked 73 bytes/order 合入本分支后重建，
不得把这份 ignored receipt 当正式材料，也不得续成 N93。

## 仍需真实通过的门

完整 N73 admission 至少还缺：

1. 由 tracked `CLIP_ORDER.json` 与 73 件 tracked NPZ 重建的 source capsule；
2. 146 件真实编译输出及完整 content-bound `BUILD_MANIFEST`；
3. grounded shared-ready 与逐件 ready FK；
4. 编译后逐件 `t_hit/t_cycle`、准备/恢复、teacher phase-rate 复验；
5. MuJoCo FK、plant-specific dynamics、grounded L/M/R；
6. 连续扫掠的桌面、桌边、球网、地面、自碰撞安全；
7. **逐动作、逐 scope 的 post-retime teacher fitted-ball 物理回台门**，同时检查触球帧、
   实际拍位/拍速/signed face、球拍接触、过网、落台、桌碰和跌倒；这是防止再次训练出
   “动作学会了但指定击球帧根本打不上台”的必需门；
8. schema-v2 registry、alignment/evidence/adoption 与 admission 工具信任集；
9. Isaac filtered-contact smoke；
10. 之后才是 action-ball policy canary 和训练授权。

任一件、任一 scope、任一 identity/SHA、时序、shared-ready、FK、安全或物理回台证据
不全，整 bank 失败；不得用平均数掩盖单件失败，不得伪造证书或授权。
