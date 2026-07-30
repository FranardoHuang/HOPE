# EXP-ROUGH-GROUND-FRICTION-FIX-20260729 — 抬脚地形 + 地面摩擦对齐 MuJoCo

日期:2026-07-29。人类负责人:Franco。执行者:Claude、Codex。状态:host 实现已落，
Pod Isaac smoke 尚未运行。

## 人话总结

机器人在 Isaac 里学不会抬脚移动,查出三个病因,这次一起修:

1. **旧凹凸地形会把机器人传送到没有桌子的地方。**`terrain_type="generator"` 的全局地形会把
   `scene.env_origins` 换成地形 tile 的原点,而克隆出来的静态桌子(TableObstacle/ShadowTable)
   还钉在原来的克隆网格上——机器人 reset 到 tile 原点,桌子却不在那里。另外 env 间距 2.5 m、
   桌长 2.74 m,邻居的桌子足迹和本 env 的机器人活动区在空间上重叠,一张共享地面 mesh 根本
   做不到"我脚下凹凸、你桌下平整"。
2. **旧噪声全部往上凸(0 到 hi),平均地面被抬高。**桌面 0.76、动作库 clip、虚拟球标定都假设
   地面平均在 z=0;地面整体抬高 = 全套标定错位。
3. **Isaac 的脚地摩擦远小于 MuJoCo 评估端。**MuJoCo(a3_pingpong.xml)地面和机器人碰撞体
   都是 μ=1.5,同优先级合成取逐元素最大 → 接触 μ=1.5,还开了 noslip;Isaac 旧配方是地面
   1.0 × 机器人材质随机 (0.3~1.6) 相乘 → 有效 μ 均值 ≈0.95,最低 0.3(等于在冰面上学走路)。

## 修复(代码,已合入工作区)

- 新模块 `tasks/tracking/terrain_patch.py`:**per-env 零均值凹凸垫**。
  - 每个 env 自带一块静态高度场 collider(`{ENV_REGEX_NS}/RoughGroundPatch`,shadow-table
    同款挂载+跨 env 碰撞过滤先例),`replicate_physics` 下全 env 共享一份烹好的 mesh。
  - **凹凸只铺机器人一侧**(球台近沿 near_x 之前);**桌子整个足迹+0.5 m 余量强制平在 z=0**。
  - **高度以 0 为平均**:写 `[lo, hi]`,实际铺 ±(hi-lo)/2 绕 0(5 mm 量化)。例
    `[0.0, 0.04]` = ±2 cm。桌面/动作库/虚拟球标定全不动。
  - 谷底之下 5 cm 再垫一块全局兜底地板,接住走出垫子的极端情况。
  - 摘掉 plane TerrainImporter 后 env origins 自动回落克隆网格 = 桌子所在的网格,机器人和
    自己的桌子永远在一起。
  - 凹凸 pattern 由 run seed 决定(可复现);已知限制:所有 env 同一 pattern(replicate
    physics 的代价),reset 时脚最多插入半带宽(±2 cm 档 ≤2 cm),由 PhysX depenetration
    正常解决。
- `train.py`:`task.plant.terrain_rough_height_range` 分支改为挂垫子;带宽合法域
  `0.01 m <= (hi-lo) <= 0.15 m` 且必须是 0.01 的倍数(更窄=死平垫;更宽=mesh 的斜坡竖墙
  修正会把负高度顶到桌侧平区边界列;非倍数=5 mm 量化会悄悄铺出与所写不同的幅度)。
- 合同(training_contract.py):rough 的 `terrain_type` 串改名
  `random_rough_heightfield` → `robot_side_zero_mean_patch`。**故意改名**:旧 generator 形态
  若产出过 checkpoint,不许与新垫子谱系静默互认 resume;同时旧串也过不了 schema-3 结构校验,
  所以 `standalone_onnx_export.py` 对旧 rough-generator checkpoint 会 `[FATAL] ground_plant is
  invalid` 拒绝导出——这是**有意的谱系切断**,不是 checkpoint 损坏。平地默认整块缺席不变,
  历史 checkpoint 逐字节兼容。

## 发射用配置(下一条腿部/移动臂建议)

YAML（`task.plant` 下；CLI 只逐键传下面四项，**不得**额外传
`ground_static_friction`/`ground_dynamic_friction`）：

```yaml
task:
  plant:
    # 地面材质保持默认 1.0/1.0(ground 两键不写):乘法合成下,脚地有效摩擦 = 下面这两个
    # 区间本身,写多少就是多少,一个旋钮讲完。
    robot_material_static_friction_range: [0.5, 1.7]
    robot_material_dynamic_friction_range: [0.4, 1.5]
    # 物理一致性:逐桶 dynamic=min(static, dynamic),不再采出"动>静"的非物理组合
    robot_material_make_consistent: true
    # 零均值凹凸垫 ±2 cm(只铺机器人一侧,桌下强制平)
    terrain_rough_height_range: [0.0, 0.04]
```

每行人话:
- `robot_material_static_friction_range=[0.5, 1.7]` — 静摩擦从"有点滑"到"比 MuJoCo 更黏",
  评估端的 1.5 稳稳在分布内。
- `robot_material_dynamic_friction_range=[0.4, 1.5]` — 动摩擦上限**顶到 1.5**:旧配方动摩擦
  最高只有 1.2,评估端真正管打滑的 1.5 训练里从来没见过,这是最大的一处缺口;下限 0.4 保留
  "偶尔滑"的压力(抑制蹭脚 wiggle),但不再有 0.3 的冰面。
- `terrain_rough_height_range=[0.0, 0.04]` — 脚下 ±2 cm 随机起伏逼机器人抬脚,桌子那侧纹丝不动。

选型依据(对照两头):
- **评估锚**:厂商 MJCF(sha256 钉死)地面+脚 μ=1.5、同优先级取 max、还开 noslip——1.5 必须
  在训练分布内,且要有像样的质量(新区间静/动上侧 ≥1.5 的质量约 17%/9%)。
- **文献锚**:G1 [0.1,1.25]、Berkeley Humanoid [0.2,1.25] 等——他们的评估/实机地面在 0.6~1.0,
  上限 1.25 罩得住他们的 gate,罩不住我们的 1.5,不能照抄;低尾巴压 wiggle 的机理照收。
- 对比旧默认 [0.3,1.6]/[0.3,1.2]:均值 0.95/0.75 → 1.1/0.95,低尾 0.3→0.5/0.4,动摩擦上限
  1.2→1.5。方向就是"范围保留、整体抬高、动摩擦顶到评估真值"。

注意:
- **任何一键显式出现都会让 schema-3 合同长出 `ground_plant` 块** → 平地谱系 checkpoint 不能
  resume 到这套 plant 上,rough/高摩擦臂必须 fresh-from-random(合同会 fail-loud,这是设计)。
- MuJoCo 端不用改:1.5 是厂商 MJCF 写死、被 eval 合同 sha256 钉住的评估真值,本次是把 Isaac
  的训练分布拉过去罩住它。
- 带宽想更狠再加大 hi:合法域 `0.01 ≤ hi-lo ≤ 0.15` 且是 0.01 的倍数;首发建议 ±2 cm。
- `robot_material_make_consistent`(2026-07-29 新 plant 键,本次已接线):isaaclab 的材质随机化
  默认静/动**独立采样**,约 1/3 的桶会采到 动>静 的非物理组合;true = 逐桶
  dynamic=min(static, dynamic)。**只做 opt-in、不翻默认**:翻 EventCfg 默认会让所有现存谱系
  resume 时物理悄悄变化而合同看不见;false 的唯一拼写是缺席(历史 6 键 ground_plant 块逐字节
  兼容),true 会让合同块长出第 7 键(= 新 plant,新臂 fresh-from-random 正好从出生带上)。
  注意开了它之后动摩擦有效分布整体下移(min 钳制),所以动摩擦上限给到 1.5 更有必要——
  "保留旧区间 [0.3,1.6]/[0.3,1.2] + make_consistent"的组合会把动摩擦均值压到 ~0.66 且冰面低尾
  还在,不采纳。
- 另一个参考里的好点子先记账不实施:critic 侧把采样到的 μ 当 privileged obs(HITTER/FALCON
  式非对称 AC)——本仓库 critic 目前不看材质,加了会改 obs 合同,单独立项。

## 验证

- host（Python 3.8）：`test_terrain_patch.py` 11 项、`test_ground_plant_wiring.py` 49 项；
  加上 `test_training_contract_schema3 / test_judge_plant_contract /
  test_reward_flags_overrides / test_exact_resume_state` 的联合回归为 **379 passed**。
- 当前证据等级仅 E1。fresh N5 首轮保持平地/no-move，不把 rough patch 混进同一因果轴。
- rough/move 正式发射前必须补：2-env clone/origin/contact isolation、兜底地板 drop、
  桌 footprint raycast、多个 seed 的前 100 physics substeps、pickle/mesh 内容绑定，以及
  4096-env VRAM/吞吐。同一门还要拒绝 ready 脚初始插进凸起所制造的假 fall/硬限位样本。
