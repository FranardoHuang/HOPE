# EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET — 反手拉 B 整轨桌网余隙门

- 状态：`source_static_pass_runtime_not_run`
- 阶段/轴：新动作库 / 整轨桌板、网与网柱几何余隙
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：E1（dependency-light source/static gate；尚无 exact MuJoCo runtime 结果）

## 问题

只问已经通过 [动作 L0 静态审计](../../DEFINITIONS.md#motion-l0-static)和
[厂商 L1 安全审计](../../DEFINITIONS.md#motion-vendor-l1-safety)的 Franco 反手拉 B：把同一条 exact
[`schema-2 motion`](../../DEFINITIONS.md) 放到现役 tracking task 的固定桌位后，整条轨迹中球拍/拍柄和
机器人其余 enabled collision geom 是否始终与桌板、网和两根网柱保持至少 `5 mm` 余隙。

本门不问动作能否击球上台，也不推进动力学。B 只有在本门 runtime certificate 通过后，才有资格进入
独立的 vendor 动力学/平衡门；C 仍不消费、不自动 fallback。

## 冻结输入与坐标系

预注册
[`motion_backhand_loop_b_table_net_clearance_prereg_20260715.json`](../../../configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json)
SHA-256 为 `9d7126bc09166bc428d2c79327417e250384ff45190c09d7ee86d90469eeb1e6`。它逐字绑定：

- B schema-2 NPZ SHA-256 `e2eb99e6...d28cc`；
- vendor L1 certificate SHA-256 `6840df34...db60`，且运行消费前必须读到
  `vendor_l1_complete=true`、`table_net_authorized=true`；
- vendor L1 plan `fd47a398...f78a770`、validator `6368bda7...fac2e`、canonical MJCF
  `2ab1cd31...feb97`、75-file closure `e0381752...962de` 和 compiled robot collision contract
  `18e7f6ff...386e5`；
- canonical HOPE 桌几何、`table_tennis_env_cfg.py::build_net_post_cfg` 的网柱尺寸/左右放置、tracking
  command 默认桌位和 tracking scene adapter 的 exact source bytes。

两套坐标不混写：schema-2/MJCF 世界以机器人环境地面原点为原点，`+X` 向前、`+Y` 是机器人解剖左、
`+Z` 向上且地面 `z=0`；canonical HOPE 桌坐标以 P1 近端左上角的桌面为原点，桌面 `z=0`、
`x∈[0,2.74]`、`y∈[-1.525,0]`。唯一允许的变换是无旋转纯平移：

```text
p_schema2_mjcf = p_HOPE + [0.5, 1.525/2, 0.76]
```

它得到 MJCF 世界中的桌板中心 `(1.87, 0, 0.735)`、网中心 `(1.87, 0, 0.83625)`，以及
`y=±0.9125`、中心高 `0.84625` 的两根 `20×20×172.5 mm` 网柱。这个桌位是冻结的现役训练反事实桌位，
不是从录制视频估出的 capture extrinsic；`capture_table_pose_observed=false`。

## 审计合同

validator
[`audit_motion_schema2_table_net_clearance.py`](../../../scripts/audit_motion_schema2_table_net_clearance.py)
SHA-256 为 `1ef347ba...39a7a`。它继承 vendor L1 已验证的 root 线性、四元数 shortest-arc slerp 和关节线性
插值，将 `151 @ 50 Hz` 扫成 `1201 @ 400 Hz` 有限样本。运行时把四个静态 box 追加到 canonical
`worldbody` 的末尾，通过 in-memory XML + exact 74-file mesh map 编译；canonical robot geom ID、qpos0、
拓扑和 compiled collision SHA 必须保持不变。

每个样本对 37 个 enabled robot collision geom 与 4 个障碍做 `37×4=148` 对检查。球拍与拍柄另做汇总，
但不从全机器人门中排除。hard 判定直接使用 exact MuJoCo saturation predicate：距离 `<5 mm` 就否决整条
动作；`5–20 mm` 只登记 warning。任一 robot-obstacle hard event 都不可由 reward、其他安全分或其他帧
补偿，并保守标记相邻 source frame。二分报告分别写 midpoint estimate 与真正的 certified lower
bracket（已扣除 saturation predicate 的 `1e-12 m` 数值裕量），不能把 midpoint 冒充下界。运行只调用
`mj_forward`，`mj_step_calls=0`。

400 Hz 仍是有限密扫，不是数学连续时间 swept-volume 证明。桌腿/厂商 visual mesh 不是本门几何；本门只
绑定现役 collision slab、net 和保守加入的两根网柱。动力学、平衡、TOPP、击球/回台、RL、Gate3 和真机
全部在范围外。

## Source/static 结果

2026-07-15 本地 dependency-light 验证：

```bash
python3 -m pytest -q tests/test_motion_backhand_loop_b_table_net_clearance.py
# 29 passed

python3 scripts/audit_motion_schema2_table_net_clearance.py \
  --prereg configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json \
  --expected-prereg-sha256 9d7126bc09166bc428d2c79327417e250384ff45190c09d7ee86d90469eeb1e6 \
  static
# PASS ... source_exact=true runtime_audit=false no_write=true continuous_time_claim=false
```

首次红队发现旧实现把 SHA 检查和 JSON/NPZ/MJCF 再打开分成两次 path read，且输出只在写前按 path
重查，存在 TOCTOU（检查时与使用时不一致）窗口；因此旧 source commit 明确 **NO-MERGE**。修复后所有
runtime 输入都经 `O_NOFOLLOW` read-only fd、`fstat` identity/size/time、单次 bytes snapshot，再从同一
bytes 做 hash + JSON/NPZ/XML parse；NPZ 只从 `BytesIO` 加载并拒绝 duplicate ZIP member、错误 dtype 与
NaN/Inf；canonical XML 与 74 mesh 从同一个 pinned model-root dirfd 读取并仅用内存 bytes 编译。输出则
绑定 parent dirfd 的 device/inode，使用 `openat(O_EXCL|O_NOFOLLOW)`、file+dir `fsync`，并从同一 dirfd
复核新建 inode/bytes；父目录换名或替换会 fail closed 并不发布。

第二次红队发现执行完整 phase/self-collision 模块会触发未绑定的普通 project import；预置伪造
`virtual_return_scorer` 可进入模块命名空间，因此该版本仍 NO-MERGE。当前实现不再 exec 这两个大模块：
densify/slerp/unsafe-source/qpos 仅保留四个 dependency-free 纯函数，source gate 强制它们与 exact upstream
四个函数 AST 逐字义等价；距离门是本地最小 MuJoCo saturation+bisection kernel，并对 exact upstream
`_far/geom_clearance` 做数值 parity。`ground_gmr_pkl`、`virtual_return_scorer`、`audit_motion_npz` 的伪造
`sys.modules` 注入负测证明本门不会消费它们。同期把误标为 lower bound 的 midpoint 拆成 midpoint estimate
与 certified lower bracket，并把网柱几何的真实 `table_tennis_env_cfg.py` source 纳入 plan。冻结 L1
validator 的验证调用结束后还会逐项恢复 `sys.path`/`sys.modules`，exact-bytes module loader 也不留下
private module entry；运行环境检查已改为本地 snapshot 版本，不再执行会改 `sys.path` 的 legacy L0 module。

反例覆盖 transitive-module injection、纯函数 upstream parity、midpoint/lower-bracket 诚实性、网柱 source
漂移、certificate path swap、model-root replacement、output-parent swap、duplicate/malformed NPZ、
`4.99/5.00/5.01 mm` 边界、非球拍 robot geom 撞网、任一 hard failure 不可补偿、HOPE→MJCF
旋转/平移漂移、漏网柱、桌板厚度漂移、duplicate obstacle name、vendor L1 未授权、continuous-time 假声明、
阈值放宽、dry-run 写文件和 certificate overwrite。

## 当前决定与下一步

源码门通过，只授权在 code review 后使用 exact `/workspace/hope_mjeval_venv` 做一次无写 `dry-run`。
目前没有 runtime 结果、没有 table/net certificate，也没有动作晋级；G08 保持 Partial。`dry-run` 必须先
验证现存 L1 certificate 的 exact SHA/authorization、输出父目录真实存在且 target absent。通过后才能执行
唯一一次 `O_EXCL` audit；完整命令和失败处理见
[操作文档](../../operations/run_motion_backhand_loop_b_table_net_clearance.md)。
