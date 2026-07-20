# 脚步幅度 β 缩放合同（泛化轴 D，footwork scale）

- 状态：`parameterized contract + runnable skeleton`（2026-07-20）。本合同定死 β 的参数化、
  分桶与验收量；**本轮不产出任何可训练资产**——动作库里唯一的横移素材（catalog 角色
  `shared_lateral_footwork_module` 的 motion/{left,right}_dang{1,2}）M0 stance 门是 `0/4`
  reject，且没有任何过动力学门的移动参考。
- 工具入口：[`hope_training/whole_body_tracking/scripts/scale_footwork.py`](../../hope_training/whole_body_tracking/scripts/scale_footwork.py)，
  单测 [`tests/test_scale_footwork.py`](../../hope_training/whole_body_tracking/tests/test_scale_footwork.py)。
- 上游语义合同：[`stroke_footwork_composition.md`](stroke_footwork_composition.md) 与
  [`configs/motion_role_catalog.json`](../../configs/motion_role_catalog.json)（冻结语义，本合同不改）。
- 人话：β 是"这一步迈多大"。工具吃一条脚步模块参考 + 一个 β + 击球动作元数据，吐一份
  **组合规格 JSON**——告诉未来的全身求解器"root 该移多远、脚该落哪、收步该回到什么站姿、
  验收查什么"。它不是动作、不是 npz、更不是训练资格。

## β 的定义与作用面

β 是脚步模块**位移量**的缩放因子，作用面严格限定为：

1. **root 有符号横向位移**：`d_target = β · d_source`（movement frame 中的 signed lateral，
   左移为正）；
2. **落脚点横向偏移**：每只脚第 k 个支撑锚点 `y'_k = y_0 + β · (y_k − y_0)`，`y_0` 是该脚
   第一个支撑锚点；前后（fore-aft）分量与高度（z）不变；
3. **支撑脚支撑期位置钉死**：锚点即约束，支撑期内不参与任何缩放或插值；
4. **收步目标不缩放**：末态必须恢复该动作源自己的初始左右脚分离向量。

明令禁止的实现方式（写死进每份规格的 `forbidden_implementations`）：只放大 hip-roll、
对全身关节统一缩放、缩放 z 轴。方向翻转（β<0）是镜像不是缩放，镜像有自己的门链
（v12 C3），本工具直接拒收。

## 分桶（写死在每份输出 manifest 里）

各泛化轴同一比例约定；轴 D 的 β 亦同：

| 桶 | β 值 |
|---|---|
| train | 0.80 / 1.00 / 1.20 |
| interpolation | 0.90 / 1.10 |
| OOD | 0.65 / 1.35 |

β 不在这张冻结网格上 → 直接拒收（分桶是预注册的，不接受任意 β）。β=0 拒收：d=0 的
"原地"必须直接引用 catalog 里真实的 `stationary_strike` 资产，不允许把移动动作压扁冒充。

## 参数化（从脚步 npz 提取，movement frame = 初始朝向对齐的水平坐标系）

1. 左右脚分离向量（left − right 的 fore-aft/lateral 分量；初/末双支撑窗内取稳健中位数，
   不用单个含噪视频帧）；
2. 双脚中心（同窗中位数）；
3. root signed lateral displacement（末窗与初窗 root 横向中位数之差）；
4. 左右脚 contact phase 表（逐脚支撑区间 `[start, end]` 帧；**硬输入**，npz 内嵌
   `left/right_foot_contact` 0/1 数组或 `--contact-json` 二选一，缺失即拒收，不做无标注猜测）；
5. 落脚位置（每脚第 2 个及以后支撑区间的触地锚点）。

## 真正的验收量清单（不是 β 本身）

β 只是刻度；一个派生组合的验收对象是：

1. **achieved signed root displacement**（`β·d_source`，规格里的实际数）；
2. **落脚点**（缩放后的 footfall targets，前后/高度未动）；
3. **初始/最小/终端 stance**：冻结 stance 门（阈值抄自
   [exact-GMR 卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)的预注册冻结判定）——
   前后分量 |Δ| ≤ 0.03 m；横向 signed 分量 |Δ| ≤ 0.03 m；初始横向绝对分离 ≥ 0.05 m 且
   左右符号不翻；末态横向分离最多变窄 0.005 m（**独立硬门**：变窄 2.4 cm 即使落在 3 cm
   带内也必须失败——M0 right2 就是这么死的）；
4. **foot crossing / 最小站宽**：源轨迹逐帧 + β 缩放后的落脚排程（支撑期锚点、摆动期线性
   插值近似，仅用于检查）都要过 sign-no-flip + 最小横向净距（工具默认 0.03 m，非冻结，
   使用它的实验须预注册数值）；
5. **contact phase 一致性**：标注支撑帧的高度带（z ≤ 各脚最低点 + 0.03 m）与支撑期水平
   漂移预算（默认 0.02 m）——标注与轨迹矛盾即拒收；
6. **recovery-ready 误差预算**：末态恢复初始分离向量，分量带 0.03 m + 变窄硬上限 0.005 m
   （冻结），支撑滑移预算 0.02 m（工具默认）；
7. **完整 Gate 链**：任何 `<strike, footwork, signed distance, phase alignment, retiming>`
   派生动作必须重过 runtime-order schema、L0 有限数/限位、厂商 MJCF L1 自碰/间隙、
   桌网扫掠间隙、厂商动力学/足接触重放；组合参数变化即重置资格。规格 JSON 永远写
   `training_authorized: false`。

## fail-closed 规则（全部有单测，M0 已知 0/4 数值作负例）

拒收即退出码 2、不写任何输出。触发条件：β=0 / β<0 / 网格外 β / 非有限值 / 缺 contact
phase 标注 / body 顺序无法解析 / 朝向漂移超 ~15°（横移模块不该转身）/ stance 门失败
（负例：left1 变窄 0.095425 m、left2 横向变化 0.200557 m、right1 变窄 0.076532 m、
right2 变窄 0.024300 m 只死在 0.005 m 硬门）/ 双脚交叉或站宽违例（源与缩放排程分别查，
β=1.35 可以把 β=0.8 合法的排程放大成交叉）/ 支撑期滑移超预算 / root 位移小于工具级下限
0.02 m（"不动的脚步模块不是脚步模块"；这不是任务 reach/deadband，那属于后续实验预注册）/
catalog 里 `input_gate_status=rejected_*` 的源资产（修好的素材必须换新 asset_id 重登记
重过门，不许借旧名）/ 击球元数据非法（strike_phase 必须是 (0,1) 内有限数；给 catalog 时
击球资产必须是 `stationary_strike`、脚步资产必须是 `shared_lateral_footwork_module`）。

## 输出与内容寻址

组合规格 JSON 单文件，一层内容 SHA（Franco 拍板的精简治理，无多层审批链）：
`inputs.footwork_npz_sha256`（+ 可选 contact-json/catalog SHA）绑定输入，`spec_sha256` =
除自身外全部字段的 canonical JSON SHA-256。`beta_buckets` 全表逐字写进每份规格。

## 与 catalog / composition 合同的关系

- 本合同只管"β 怎么缩放、怎么验收"；**谁能当脚步模块、谁能当击球、横移何时触发**由
  [`stroke_footwork_composition.md`](stroke_footwork_composition.md) 冻结语义定，本合同不越权；
- 组合交界所有权沿用 v12：脚步模块负责 root 平移/偏航、双足、腿关节与接触阶段；上肢挥拍
  以骨盆相对坐标负责球拍/接触目标；骨盆/躯干是耦合变量由受约束全身求解给出——规格里的
  摆动期线性插值只是 crossing 检查的近似，不是轨迹；
- 语义修正不覆盖安全事实：M0 四条 stance `0/4` reject 不因本合同存在而改变；本合同的
  骨架只有在未来出现"保留位移、末态回自身初始 stance、过完输入门"的新脚步资产后才可能
  产出可执行组合规格，而那条规格仍要重过完整 Gate 链才谈训练。

## 校验入口

```bash
/path/to/python -m pytest hope_training/whole_body_tracking/tests/test_scale_footwork.py -q
python hope_training/whole_body_tracking/scripts/scale_footwork.py --help
```

## 边界与未宣称事项

单一 movement frame 假设横移全程朝向近似不变（超限即拒收），不支持转身脚步；最小站宽
0.03 m、滑移预算 0.02 m、root 位移下限 0.02 m 是工具默认而非冻结阈值；本合同不宣称任何
平衡性、接触稳定性或训练收益——那些只能由 Gate 链和后续预注册实验回答。
