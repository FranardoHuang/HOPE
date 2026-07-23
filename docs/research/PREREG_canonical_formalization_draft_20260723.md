# PREREG 草案:动作库 canonical 正式化 + 三变体资产进 main(franco_pipeline_20260722 → 新战役)

- 状态:`draft-for-franco-review`(本文档为草案文本,未动任何工具/仓库)
- 人类负责人:Franco;起草:Claude(franco_pipeline_20260722 战役收口)
- 依赖证据:FINAL_11_TABLE v3(URDF 口径)、facescan/*、geom/*、shared_ready_prefix_thit.json、
  boundary_cert_20260722 C3 终审、fh_block_syn 单轴合成验证(SYNR:门全净,sp2 100/sp4 96/sp5 92)

## 一、工具改动面(解除窄绑定;每项独立 code-review + 版本化合同,禁止顺手改物理)

| 工具 | 现状 | 改动面 | 不变量 |
|---|---|---|---|
| materialize_motion_spatial_se2.py | ALLOWED_ASSETS 硬编码 B/C | 白名单改为"资产表驱动"(新 configs/motion_asset_table_v1.json,含 11+1 段:统一名、源 pkl sha、SE2(平移+yaw)、世代) | 变换语义/校验/no-clobber 逐字不动;站位 yaw 修正(fh_loop -147°/bh_block -80°/s0 -72°)作为表中 SE2 合同项,附锚窗速度方向审计证据 |
| materialize_motion_schema2_fk.py | ASSET_IDS=(B,C) | 同上表驱动;S0/v12 世代 pkl 的 restricted-loader 允许清单扩展(numpy _reconstruct/ndarray/scalar 三条,仍禁任意 global) | 30→50Hz/FK/velocity 数学逐字节不动(shadow 已证 bit-exact);донor/闭包绑定机制不动 |
| audit_motion_schema2_l0_static_v2.py | ASSET_ID=B | 表驱动 + 每段独立证书路径 | V1/V2 双层合同结构不动;C 右肩 roll 贴限(超额 0.0)须人工裁定条款 |
| audit_motion_schema2_vendor_l1_safety.py | B 专用 | 表驱动 | 31 关节双射、0.005/0.020 阈值、no-clobber 不动 |
| audit_motion_schema2_table_net_clearance.py | B 专用 | 表驱动 | v4 语义(saturated lower bound 等)不动 |
| csv_to_npz_mujoco.apply_grip_rotation | 现为 grip 标定用 | 仅新增"单轴常量平移"入口(挡族翻面);grip 标定语义按 v3 裁决整体弃用 | URDF 安装口径唯一真源 |

## 二、资产表(11 段 + 合成 1 段)与每段验收门

统一验收门(全段):SE2 材料化 exact → schema-2 三证(L0/L1/桌网,阈值同 B 模板)→
URDF 口径六轴扫描(解拍面 ±25°/F13,慢档能量锥 0.6-4×)default 档最佳帧 ≥50% 且有非空 ≥50% 窗
→ C3 t_hit(击球链+硬件包络)可行 → 进注册表。
段特例:
- fh_v4rg/bh_v4rg:在册资产不重产;**裁决项:bank *_cal 的 bake 拍面偏 URDF 40.3°,raw 件全库不存在**
  ——二选一由 Franco 裁:(a) 承认 bake 为 v4rg 历史口径、注册表注差异;(b) 从 v4 视频重走 GVHMR→GMR 产 raw 件对齐 URDF 口径。
- fh_block(原版):URDF 面全轴 0%(GMR 腕轨迹拍面病)——不入册;由 **fh_block_syn** 顶位(见下),原版降级为"重定向修复案"排队。
- fh_block_syn:= bh_block 轨迹 + 站位 SE2(yaw -80°)+ **right_wrist_roll 单轴 -180° 常量平移**(绝对轨迹 [-154,-136]° ∈ 限位 ±160°,零裁剪);验收附加:合成件谱系必须绑定 bh_block 源 sha + 平移参数。
- s0_highpress:入册用高压专属卷(见四);泛用卷数字只入档。
- v12×2:GVHMR/GMR 影子链正式化(补 v12 GVHMR prereg——07-13 文档"v12 本轮不授权"需 Franco 显式解除);native betas 口径入谱系。
- M0×4:不入本战役(stance 死门维持)。

## 三、三变体烤制与命名规范(每段三件套,进 main 供直接训练)

| 变体后缀 | 定义 | 验收 |
|---|---|---|
| (无后缀,原版) | SE2+schema-2 全轨 | 三证 + 六轴扫描门 |
| `*_adv2c3` | 前置切片:从 2c/3 前置帧起(帧号=C3 终审 ready_frame,逐段入表),首帧即就绪零速 | 三证重跑(切片件独立证书);t_hit ≤0.5 逐段核(前置共享口径:分侧全绿已证,SET_ALL 差 B/fh_loop 0.035-0.039s) |
| `*_loop` | 闭环版:尾部恒扭矩(bang-bang,τ×1.0)回位接回起始 ready,**同帧始末、连拍不瞬移** | 三证重跑;回位段独立动力学重放(CoP/滑移只报数);t_cycle 逐段入表(t_recover 参考列已备:0.093-0.267s) |

命名:`<统一名>[_syn][_adv2c3|_loop]`;注册表每件绑定:源世代 sha、SE2 合同、变体参数、扫描锚窗、per-clip sign。

**注册表规范新条款(防错赋号)**:`mount_normal_sign` 为逐 clip 显式字段,**禁止按正/反手侧别自动赋号**。
挡族(fh_block_syn/bh_block/v12×2 挡)遵循**同面约定**:正反两翼用同一物理拍面(180° 腕转保同面朝球),
fh_block_syn 的 sign 与 bh_block 相同;拉族维持每侧异面惯例。两族约定并存,注册表逐条写明。

## 四、题库(两张新卷 + 既有卷)

1. **挡球快球卷**:速度 4-8 m/s 主带(挡族边界档窗 300-460ms 实证),方向 vy≤1.2;慢档格保留"非设计用途"注记。
2. **S0 高压卷**:高点带 1.04-1.09m、陡降 vz(按档 [-2,-0.8]/[-3.5,-1.5]/[-5,-2]/[-6,-2.5])、能量锥 0.6-4×;
   实证:S0 在该卷 1-8 m/s 全档 100%(仅 5-8 次帧 92)——"够高就能扣"成立。
3. 拉族沿用泛用卷(URDF 口径重出 train/exam split)。

## 五、执行顺序与回滚

工具泛化(五件,review 后合入)→ 资产表 v1 冻结 → 11+1 段原版三证(pod 串行,产物 no-clobber)→
三变体烤制+各自三证 → 注册表+两张新卷 → main 合入(单 PR,Franco 终审)。
任何一段任何门 FAIL:该段停在该门入账,不阻他段;禁止改阈值/物理救分。
影子产物(franco_pipeline_20260722)全程只作预期值参照,不得直接改名入册。
