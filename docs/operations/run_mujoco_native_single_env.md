# MuJoCo native single-env fixed-tape diagnostic

这条工序只验证一件事：真实 vendor A3 MJCF 加五实体桌/网场景，能否按 schema-3 的 31-D
`action -> episode-fixed delay -> affine qdes -> 5% soft-envelope interior 与 hard-inner 交集投影
-> total-PD` 合同确定性走完100个 control tick。

它没有球、Reward、Observation、VecEnv 或 PPO。所有输出都带
`diagnostic_unauthorized=true`；成功不能授权 canonical training、promotion、deployment 或真机命令。
teacher-reset smoke 使用所选动作 NPZ 的 teacher frame 0：root/q/dq 与 root velocity point semantics 都写进
fixed tape；history fill 与 probe action 中心由同一 q0 反解，避免 reset 是 teacher、第一步却被默认站立
qdes 拉走。省略 `--teacher-motion` 时仍可生成 vendor `stand` root + executed-zero-action-q
的负对照 tape，receipt 会用
不同 `reset_mode`，禁止混报。
工具还会重验 NPZ 内的 joint-order contract ID/SHA 与
`configs/a3_joint_order_bijection_v1.json` 的当前字节一致；不允许只靠 31 列宽度猜测列语义。
每次读 tape 时还会重读 teacher NPZ、重算 SHA 和指定 frame，并验证 delay history fill
确实解码到同一 q0；tape 中的自声明 lineage 不能单独写入 receipt。

## 1. 生成不可变 fixed tape

在仓库根目录执行：

```bash
python -m hope_training.whole_body_tracking.mujoco_native.single_env make-tape \
  --contract configs/a3_vendor_runtime_authority_20260802_r8/bh_loop_c.shared_ready.training_contract.json \
  --teacher-motion assets/motions/chingmu_n1_take061u04_mechanical_candidate_v5_20260803/hope_Take_061_unit04_BH.measured_v5.npz \
  --teacher-frame 0 \
  --delay 0 \
  --tape /tmp/a3_mujoco_delay0_tape.json
```

该 v5 资产 SHA 必须是
`5899706b32cd60fa7b1e08094cd39d6aae59f4ed9b95c493026fb3b3dbb98101`。它的 NPZ 内部明确写有
`measured_racket_mechanical_admission=0` 和 `diagnostic_unauthorized=1`；即使本 runner 完成，也不得称为
canonical motion safety 或 N1 放行。

工具拒绝覆盖已有文件。标准输出记录 tape SHA、plant-binding SHA、teacher motion SHA、`100x31`
形状和 delay。
分别生成 delay `0/1/2` 时要使用三个不同输出路径。

## 2. 在带 MuJoCo Python 包的隔离环境运行

```bash
python -m hope_training.whole_body_tracking.mujoco_native.single_env run \
  --contract configs/a3_vendor_runtime_authority_20260802_r8/bh_loop_c.shared_ready.training_contract.json \
  --tape /tmp/a3_mujoco_delay0_tape.json \
  --trace /tmp/a3_mujoco_delay0_trace.npz \
  --receipt /tmp/a3_mujoco_delay0_receipt.json
```

receipt 必须同时满足：

- `status=DIAGNOSTIC_FIXED_TAPE_COMPLETE`；
- `counters.policy_ticks=100`、`counters.physics_substeps=400`；
- `reasons.fixed_tape_complete=1`；
- contract、binding、tape、vendor MJCF、augmented scene、table geometry 和 trace SHA 均非空；
- reset lineage 精确绑定 `Take_061_unit04_BH`、motion SHA 和 frame 0；
- delay histogram 只有本 episode 的一个固定 lag；
- safety 区保留 joint-velocity、table/self-contact、首次触碰 tick/pair、pelvis 高度/朝上分量和
  penetration 计数；
- 四个 authorization 位全部为 false。

两次相同输入的 `trace_content_sha256` 必须一致。NPZ 文件 SHA 另行记录；科学 parity 使用不依赖
zip 容器元数据的 content SHA。

`DIAGNOSTIC_FIXED_TAPE_COMPLETE` 只表示100 tick 执行链走完。joint velocity 在每个 `mj_step` 后
采样，包含最后一个 physics substep；self-contact 必须保留 maximum penetration 与 worst pair。
只要出现 table/self-contact 或
joint-velocity 越界，`safety.diagnostic_no_contact_gate_passed` 就是 false；不能把 runner complete
误报成 plant safety pass。

## 3. 测试

无 MuJoCo 的 host 仍可验证 JSON、31-D order、delay、decoder、SHA 和 no-clobber：

```bash
pytest -q hope_training/whole_body_tracking/tests/test_mujoco_native_single_env.py
```

在安装 `mujoco` 的环境，同一命令会额外编译 vendor A3 + 五实体桌网并执行完整100-tick smoke；
该用例在无依赖 host 明确显示 `skipped`，不能把 skip 报成真实 runner 已通过。

## 4. 2026-08-03 v5 exact diagnostic 结果

exact motion/audit/receipt SHA 分别为 `5899706b…b98101`、`c968ea8b…c2ff6`、
`756218ed…05ddde`。当时实际消费的 root MJCF SHA 为 `70c4fd65…36c0a`。delay 0/1/2 都走完
100 policy ticks / 400 physics substeps，但安全门全部 **FAIL**：

| delay | velocity events | self pairs / substeps | table pairs / substeps | max self / table penetration | first contacts |
|---:|---:|---:|---:|---:|---|
| 0 | 18 | 411 / 271 | 159 / 136 | 9.12 / 12.81 mm | tick 9: left hand–left hip; right wrist–table |
| 1 | 20 | 421 / 279 | 149 / 134 | 26.90 / 5.22 mm | tick 9: same pairs |
| 2 | 26 | 435 / 275 | 154 / 139 | 18.52 / 5.13 mm | tick 9: same pairs |

delay 0 在 reset 时是无接触状态；pelvis 从 `z=0.89184 m, up_z=0.86947` 在 0.20 s 内移到
`z=0.80308 m, up_z=0.8830`，随后两类接触同时出现。右腕–桌最短距离从 reset 的 `203.81 mm`
收缩到 tick 8 的 `32.18 mm`，tick 9 穿入 `11.63 mm`；左手–左髋从 `98.34 mm` 收缩到
tick 9 穿入 `2.15 mm`。将 probe 改成完全恒定的 teacher-q0 hold 仍在 tick 8/9 碰自身/球桌，
将 root 向下试探移动 10.6–30 mm 也不能消除失败。因此当前证据指向开放环姿态在该 plant/PD
下快速失稳，不是 delay 分支、小幅 probe 或单一 root-ground 高度偏差；不得通过关闭接触门
或放宽限制把该诊断改写成 PASS。运行产物位于 `/tmp/a3_mj_take061_v5.xeL7rC`。
