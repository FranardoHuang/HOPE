# EXP-A3P-P1-0807-COLLISION-PROXY-20260808

Status: v1 代理已重做并落库；两个引擎的门已同强度；变异测试 23 条全绿；
Isaac 实跑 C0/A0 `materialize → recipe → oracle32` 见 §7。2026-08-30 的 v2
multi-OBB 是功能分支候选：已绑定双端实际 collider 并集与不变的 20 mm
no-touch；合入/现役运行替换仍未授权，详见 §9。

Evidence: 纯 CPU 几何重算（可复现）＋ pod 上对真 bundle 的派生证明 ＋ 变异测试 ＋ Isaac 实跑
解释器：`/workspace/hope_isaac_venv/bin/python`（Python 3.10.18）

前情：[EXP-A3P-P1-0807-DUAL-ENGINE-MODEL-SET-20260807](EXP-A3P-P1-0807-DUAL-ENGINE-MODEL-SET-20260807.md)
（`5d6924d7` 把 Isaac 的 plant 身份门从"字节完整性"升成"派生证明"，ActionBall
第一次真的在 0807 上开机，然后在第一个走步阶段 `oracle32` 被桌面守卫拒收）。

## Question

`5d6924d7` 之后 `oracle32` 撞的那道门——`terminations.py` 里退役 0409 那六个摘要的
第三份拷贝——**拒收是对的**。它后面那件物理事是：把桌面碰撞代理重做到 0807，
并回答"左夹爪第一次进入 `robot_hit_table` 视野"会怎样。

## 一句话结论

1. 代理分量 **43 → 62**，多出来的 20 个全是左手 OmniPicker3 夹爪。
2. **出生姿态反而更安全**：离加余量的代理盒从 `11.98 mm` 变成 `36.77 mm`。
   因为旧的粗占位手 `left_hand_link`（一个 83×42×56 mm 半轴的胖盒子）退场了，
   而真夹爪虽然伸得更远，却是 20 个细条，最近的那条离桌子更远。
3. **但在动起来之后夹爪是真的会咬人**：现役单臂 clip `Take_061_unit04_BH`
   全程最小净空从 `17.51 mm` 掉到 `7.22 mm`，最紧的那个分量从右手拍柄
   变成了左夹爪 `left_link6`。全库 73 条上，**20 个夹爪分量让 542 帧从"合法"
   变成"违规"**，并且在 5107 帧里的 **2669 帧（52%）它就是最近的那个分量**。
4. 这不是靠放宽余量或排除夹爪能"解决"的东西，见 §4.c 的处置建议。
5. **A0 的 `oracle32` 仍然不过**，但它在退役 0409 盘上就是这个样子（32/32 击球前
   `robot_hit_table`），**不是本轮造成的**，见 §7。

## 1. 代理重做了什么

生产器 `hope_training/whole_body_tracking/scripts/materialize_a3_table_collision_proxy.py`
的源 URDF 从 `agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf`
换成 `agi/URDF/A3P-P1-32dof-0807-OP3-pingpang/urdf/model.urdf`；产物落在新目录
`configs/a3_table_collision_proxy_a3p0807_20260808/a3_table_collision_components.v1.json`。

**退役的 0409 产物留在 `configs/a3_table_collision_proxy_20260731/` 不删**——
它是一份完整、自洽、`content_sha256` 封好章的"错机器人"代理，正好当反面样本，
变异测试第一条就用它。

| | 数量 | 说明 |
| --- | --- | --- |
| 归一化大小写后两副共有 | 42 | 局部 OBB 中心与半轴最大差 **`1.117e-12 m`**，mesh 字节完全相同 |
| 只有 0409 有 | 1 | `left_wrist_yaw_Link ← left_hand_link`（旧的粗占位手） |
| 只有 0807 有 | 20 | `left_wrist_yaw_Link ← left_base_link / left_link1…left_link18`，即左手 OmniPicker3 夹爪 |

两个容易被误读的点，先点清：

- **62 个 `component_id` 全部是新字符串**。0807 包把 mesh 文件名改成了小写
  （`head_pitch_Link.STL` → `head_pitch_link.stl`），所以逐字符比 id 会得到
  "0 个共有"。这纯是命名，不是几何。上表的 42 是按 `(owner, link, index, mesh)`
  全部转小写之后比出来的。
- **右手拍子的碰撞几何逐位没动**。`pingpang_black_link` / `pingpang_red_link` /
  `pingbang_ball_link` / `right_hand_pingpang_link` 四个分量的局部中心在两副盘上
  完全相同，所以 `TABLE_RACKET_BLADE_CENTER_OFFSET_WRIST_M` 和
  `TABLE_RACKET_BLADE_HALF_EXTENTS_M` **不需要重切**，本轮没动它们。

## 2. 门：从"没被改过"升成"是从这台机器人量出来的"

`5d6924d7` 的教训一字不改地适用：六个 SHA-256 只证明"这份缓存没被人编辑过"，
不证明"这份缓存是这台机器人的缓存"——任何机器人的 USD 都同样能过。
本轮把同一套派生证明装到三个地方。

**派生证明是什么**：照抄 IsaacLab 的
`isaaclab/sim/converters/asset_converter_base.py::_config_to_hash`，
用 bundle 自己的 `config.yaml`（去掉 `asset_path` / `usd_dir` / `usd_file_name`
三个路径键）加上**现场 URDF 的字节**离线重算 MD5，要求与 bundle 里存的
`.asset_hash` 逐位相同。不起 Kit。名字可以改，这一步改不了。

| 装在哪 | 做了什么 |
| --- | --- |
| 生产器 `_plant_identity` | 读 `configs/a3p_p1_0807_model_set_v1.json` 收据 → 核对 URDF 摘要 → **逐个核对本次用到的 collision mesh 与收据 closure 一致**（`.asset_hash` 只盖 URDF 文本、不盖 mesh，缺这一步 mesh 是散的）→ 核对 `config.yaml` 的 `asset_path` → 派生证明 |
| Isaac 运行时 `_verify_live_bundle_is_a_cache_of_this_plant` | 在 `_verify_loaded_runtime_usd_bundle` 里对**活体 bundle** 重做同一次派生，用仓里那份已跟踪的 plant URDF |
| 产物自陈 `plant_identity` | 把 converter 的 `config.yaml` **原文**带进产物，所以没有 pod bundle 的人（MuJoCo 那条、笔记本上审阅的人）也能自己重算一遍 |

在真数据上核过：0807 bundle 的 `config.yaml` ＋ 仓里 `agi/URDF/A3P-P1-32dof-0807-OP3-pingpang/urdf/model.urdf`
重算 = `676efde5febed3c0fde0f2ad59650cdf` = bundle 里 `.asset_hash` 存的值；
换成 0409 URDF 立刻变成别的值。顺带也证明了收据里那句
`identical_bytes_to_isaac_asset` 是真的——**已跟踪的那份包和 git-ignored 的
Isaac 资产包逐字节相同**，63 个文件对收据 closure 零失配。

新的钉子（六份 0409 拷贝里的第二、三份就此归位）：

| 项 | 值 |
| --- | --- |
| 产物 sha256 | `896a5c96f5e16f266067841d72c1009e058eccf42850fff2f1c22ee46bda8b96` |
| 产物 `content_sha256` | `6a759c006b5f94872398962e7858a8c926f8f84fc67f08bff70f299e050373ce` |
| plant URDF sha256 | `15c83f5f3beea71350583143aef4d622d5219df65a0bed9a660a0edb7d388d09` |
| USD bundle tree sha256 | `365ba37edd5e5e1d4fac22f2cbb3ec871ead7bb49aeadb50161ef523a9ae6747` |
| USD bundle 总字节 | `60519988`（0409 是 `21897893`——夹爪网格进来了） |
| IsaacLab `.asset_hash` | `676efde5febed3c0fde0f2ad59650cdf` |

## 3. 两个引擎同强度（Franco「A 和 C 应该大多数设置都一样」）

`mujoco_native/table_termination.py` 以前查文件 sha、schema、`body_order`、
自封 `content_sha256`、分量数、有限性、覆盖率，**唯独不读产物自己文档里就写着的
`source_urdf` 和 `runtime_usd_bundle`**，而 Isaac 侧是读的。这个不对称的实际含义是：
一份封得很好的"别的机器人"的代理，这边会收、那边会拒。

现在这条也读了，并且**用产物自带的 `config.yaml` 加仓里的 URDF 重做同一次派生**
——MuJoCo 那条没有 pod bundle，也不需要有。另外把 `runtime_usd_bundle.files`
从"只比 `bundle_tree_sha256`"改成逐行相等（以前 `files` 里的行可以被改而
`bundle_tree_sha256` 不动，就不被发现）。

**污染源也堵了**：生产器原来只拿六个硬编码摘要验 `--runtime-usd-bundle-root`，
等于把一份"没人证明过是谁"的 USD 洗成一份两个引擎都信的**已跟踪**产物。
现在生产阶段就要过派生证明。

**选择器覆盖面**（"指纹不等于语义一致"那条）：
mujoco 侧钉 Isaac 常量 AST 的选择器新增 8 条常量 + 2 个函数；
`hope_env_cfg` 那一侧把 `TABLE_COLLISION_PROXY_ARTIFACT_PATH` / `..._SHA256`
纳入切片（换产物指针就是换机器人）。config 选择器元组从函数体里提出来成
模块常量 `ISAAC_TERMINATION_CONFIG_SELECTORS`，测试里那份**手抄的第四份拷贝
删掉，改读活的**。

## 4. 左夹爪那 20 个新分量到底怎么样

### 4.0 方法与自检

纯 CPU：从**同一副 URDF** 做正运动学（所以 OBB 所在的 owner 坐标系和 FK 给出的
owner 位姿是同一个坐标系，不是两套约定对拍），再对五个桌面 AABB 做 15 轴分离轴
测试，取投影间隙的最大值。对"很小的盒子 vs 很大的盒子"这个最大值取在面法线上、
就是真实欧氏距离；无论如何它是下界，**符号是精确的**。

三条自检，都过：

| 自检 | 结果 |
| --- | --- |
| FK 对 clip 自带的 `body_pos_w`（0409 盘） | 最大 `4.97e-7 m` |
| FK 对 clip 自带的 `body_pos_w`（0807 盘对 0409 采的 clip） | 最大 `1.555e-3 m`，正是厂商修的 ±1.5 mm 踝关节不对称 |
| 复现历史记录的三个数（`32 / 12 / 122 mm`，见 EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802 §5.6.5） | `31.98 / 11.98 / 122.29 mm` |

最后一条值得单独说：那三个数原来只存在于文档正文和提交信息里，
测它们的 pod 探针没进仓（该文 §5.6.12 把"逐 body 间隙扫一遍"列为待办）。
**本轮第一次用仓内可复现的输入把它们算了出来**，数字对上。

### 4.a 出生姿态：合法，而且比以前宽

出生姿态取自
`configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json`
的 `physical_ready`。桌子 `near_x=0.5, surface_z=0.76`，余量 `0.02`。

| | 离真台面板 | 离加 20 mm 余量的代理盒 | 最紧的分量 |
| --- | --- | --- | --- |
| 0409（43 分量） | `31.98 mm` | **`11.98 mm`** | `left_hand_link` vs `top` |
| 0807（62 分量） | `56.77 mm` | **`36.77 mm`** | `left_link13` vs `keepout` |

**夹爪比手更靠前，但更细。** 旧占位手中心在腕系 `x=0.131 m`、半轴
`(0.083, 0.042, 0.056)`，最远伸到 `0.214 m`，是一个把整只手包起来的胖盒子；
真夹爪最远的分量中心在 `x=0.2387 m`（`left_link14-1` / `left_link7-1`），
确实更靠前，但每一条都是细杆。出生姿态下左手是垂在身侧的，胖盒子的"侧面"
才是逼近桌面的那一面，细杆没有那个侧面。

**所以 (a) 的答案是：现役出生姿态仍然合法，而且余量从 12 mm 变成 37 mm，宽了三倍。**
20 个新分量在出生姿态上不但无害，还把出生姿态从"上身一动就碰线"里救了出来。

### 4.b 教师 frame 0 与全库 73 条

**take061 的教师 frame 0**（就是历史上那个 122 mm）：

| | 离真台面板 | 离代理盒 | 最紧的分量 |
| --- | --- | --- | --- |
| 0409 | `143.60 mm` | `122.29 mm` | `left_hand_link` vs `top` |
| 0807 | `121.79 mm` | `101.79 mm` | `left_link13` vs `top` |

窄了 `20.5 mm`，但 102 mm 仍然很宽。

**整条 take061_unit04_BH（57 帧，就是现役单臂在跑的那条）**：

| | 全程最小净空（对代理盒） | 出现在 | 最紧的分量 | 违规帧 |
| --- | --- | --- | --- | --- |
| 0409 | `17.51 mm` | frame 43 | `right_hand_pingpang_link` | 0 |
| 0807 | **`7.22 mm`** | frame 23 | **`left_link6`** | 0 |
| 0807 去掉 20 个夹爪分量 | `17.87 mm` | — | — | 0 |

**没有违规帧，所以现役单臂能跑**（§7 的 `oracle32` 也证实了）。但最紧处从
17.5 mm 掉到 7.2 mm，主导它的从右手拍柄换成了左夹爪——挥拍中段左夹爪离真台面板
只剩 27.2 mm。

**全库 73 条**（`assets/motions/chingmu73_measured_v4_20260803`，5107 帧，
逐帧对代理盒）：

| | 违规帧 | 逐 clip 最小净空 min / p05 / 中位 / max（mm） | 各 clip frame 0 min / p05 / 中位（mm） |
| --- | --- | --- | --- |
| 0409（43 分量） | 1025 | −179.29 / −128.13 / −10.43 / 39.56 | −8.01 / 21.10 / 78.01 |
| 0807 去掉夹爪（42 分量） | 663 | — | 4.94 / 32.40 / — |
| 0807（62 分量） | 1205 | −179.80 / −116.24 / −14.86 / 25.62 | −32.07 / 6.81 / 62.43 |

把这三行拆开读：

- **1025 → 663**：这是**旧占位手退场**的功劳，它一个人就占了约 362 帧。
- **663 → 1205**：这是**20 个夹爪分量新增的 542 帧**。
- 净效果 1025 → 1205，**+180 帧 / +17.6%**。
- 夹爪在 **2669 / 5107 帧（52%）里是最近的那个分量**；
  单帧净空最大损失 **176.84 mm**（`hope_Take_058_unit03_FH` frame 28）。
- frame 0：去掉夹爪 73 条全部为正（最小 4.94 mm）；加上夹爪有 **1 条**转负
  （`hope_Take_058_unit08_FH`，−32.07 mm），p05 从 32.40 mm 掉到 6.81 mm。

**这里必须诚实说清一件先于本轮就存在的事**：上表的绝对违规帧数**主要不是夹爪造成的**。
0409 盘上本来就有 1025 帧违规，最深的是 `right_knee_Link` 在正手弓步时进到
桌下 `keepout` 体（`hope_Take_058_unit07_FH` frame 61，−179 mm）。
原始 73 条库里就有会撞桌子的 clip，这也正是
`scripts/certify_action_ball_swept_clearance.py`（5 mm 硬净空）存在的原因，
ActionBall 并没有把 73 条全部录用。**所以 (b) 的答案要分两半报**：
绝对数字是旧账，**增量 542 帧 / 52% 主导权是本轮新增的**。

坐标系的前提也写明：上面把 clip 自带的 root 位姿当成 env-local 位姿。
这个前提在 take061 上被 122 mm 的复现验证过；如果哪条 clip 在回放时另有
重定位，那条的绝对数字要重算，但**增量结论不受影响**（同一位姿下比有无夹爪）。

### 4.c / 4.d 判定

- **(d) 出生姿态这一格：夹爪无害，而且是净改善。** 12 mm → 37 mm。
  任务描述里"夹爪比手更靠前所以更危险"这个担心，在出生姿态上被数据推翻了。
- **(c) 运动中这一格：是真发现，要报。** 20 个新分量让全库多 542 帧违规、
  在一半以上的帧里成为最近分量、把现役单臂的最紧处砍掉 10 mm。
  **本轮没有用放宽余量或排除夹爪来"解决"它**——`20 mm` 是 fail-closed 门，
  它自陈"会在真实接触之前就终止"（`hope_env_cfg.py:713-715`），
  把它调小等于把安全边界让给未验证的几何。

**建议的处置（都不在本轮范围，需要 Franco 拍板）**：

1. **先确认几何是不是过保守**。0807 的夹爪碰撞网格是**按字节复制自视觉网格**的
   （厂商口头确认"碰撞=视觉"，收据 `written_evidence_on_file=false`）。
   视觉网格通常比真实碰撞体胖。如果厂商能给真碰撞体，62 个分量里最紧的那批会松。
   这是**唯一一条"改几何"而不是"改门"的路**，应该先走。
2. **左臂的姿态本身值得改**。真正的问题不是夹爪多了 20 个盒子，而是
   **非持拍的左臂在挥拍时离桌面只有几厘米**。teacher 动作是人打的，人没有
   240 mm 的夹爪。这属于重定向/动作库的事，与"换动作库"那一摊放在一起看。
3. **录用门要重跑**。`certify_action_ball_swept_clearance.py` 的 5 mm 硬净空
   要在 0807 盘上对全库重跑一遍，重新划录用集。本轮的 542 帧就是它的输入。

## 5. 变异测试

`hope_training/whole_body_tracking/tests/test_table_proxy_plant_identity.py`，
23 条，`/workspace/hope_isaac_venv/bin/python` 3.10.18。四个必答场景：

| 场景 | 结果 | 开火的那句话 |
| --- | --- | --- |
| (i) 0409 产物，`source_urdf` / `runtime_usd_bundle` / `plant_identity` / 夹爪清单**全部重新盖章**成 0807 的值 | **REFUSED** | `component count is malformed`（62 ≠ 43） |
| (i′) 同上，再把分量数**补齐到 62** | **REFUSED** | `omits left OmniPicker3 gripper collision links` |
| (i″) MuJoCo 侧同一份伪造产物 | **REFUSED** | `must contain 62 components` |
| (ii) 正确的 0807 产物，Isaac 侧 | **PASS** | 62 分量，覆盖 32 个 body |
| (ii′) 正确的 0807 产物，MuJoCo 侧 | **PASS** | 同一份文件、同一个 `.asset_hash` |
| (iii) 产物字节追加一个换行 | **REFUSED** | `artifact SHA mismatch` |
| (iii′) 改一个 OBB 中心 5 cm 后**刷新文件 sha** | **REFUSED** | `content SHA mismatch` |
| (iv) **只删掉 20 个夹爪分量**、其余不动 | **REFUSED** | `component count is malformed` |
| (iv′) 同上，再把分量数**补齐回 62** | **REFUSED** | `omits left OmniPicker3 gripper collision links` |

派生证明本身也逐条削弱过，每次只削一条腿：

| 削弱 | 结果 |
| --- | --- |
| 去掉整个 `plant_identity` | CAUGHT `no derivation proof` |
| 换成别人的 `isaaclab_asset_hash` | CAUGHT |
| 改产物自带的 `config.yaml` 原文并重新算它的 sha | CAUGHT `converter configuration` |
| MuJoCo：改 config 原文并把六文件映射里的 `config.yaml` 行也改齐 | CAUGHT `six-file pin` |
| MuJoCo：把盘上的 plant URDF **连同它的摘要 pin 一起**换成 0409（每一个字符串比较都对上了） | CAUGHT `not derived from the reviewed plant` ——只剩派生这一条能看出来 |
| 活体 bundle：`config.yaml` 指 0807 但 `.asset_hash` 是从 0409 URDF 算的（**今天骗过我们的那个情形**） | CAUGHT `not a cache of the proxied plant` |
| 活体 bundle：`config.yaml` 指别的资产包 | CAUGHT `different asset package` |
| 活体 bundle：换了个 converter 设置、`.asset_hash` **算得对**但不是评审过的值 | CAUGHT `differs from the reviewed pin` |
| 活体 bundle：真正 0807 的缓存 | **PASS** |

粗一档就过不了的证据：
`_rederive_isaaclab_asset_hash` 换成 0409 URDF 立刻变值；
只挪三个路径键（`asset_path` / `usd_dir` / `usd_file_name`）不变值；
翻一个 `merge_fixed_joints` 就变值。

## 6. 回归

同一棵 worktree、同一个基线 commit（`5d6924d7`），
`/workspace/hope_isaac_venv/bin/python`（3.10.18），
`pytest hope_training/whole_body_tracking/tests -q -rs -p no:randomly -p no:cacheprovider -rf`：

| | failed | passed | skipped | errors | 用时 |
| --- | --- | --- | --- | --- | --- |
| 改前（`5d6924d7`，worktree `proxy0807_20260808`） | 122 | 7750 | 53 | 19 | 2068 s |
| 改后（worktree `proxy0807_test_20260808`） | 122 | **7773** | 53 | 19 | 2029 s |

**失败集合逐条相同**（`diff` 两份 `FAILED` 清单为空）；`+23` 全是本轮新写的测试。

中途出过三条真的新失败，都修掉了，不是靠调门:

| 新失败 | 原因 | 处置 |
| --- | --- | --- |
| `test_reward_flags_mdp::test_table_guard_first_hit_ledger_conserves_cells_categories_and_phases` | 测试自己 `range(43)` 造分量，还手算了 `(4,3,45,5)` 的台账形状 | 改成读 `_A3_COLLISION_PROXY_COMPONENT_COUNT`，形状写成 `N+2` |
| `test_reward_flags_mdp::test_table_guard_oracle_first_hit_export_is_sidecar_with_honest_gaps` | 同上 | 同上 |
| `test_table_obstacle_termination::test_full_assembly_rejects_invalid_cached_table_boxes` | 新测试文件叫 `test_a3_...`，**排序排到了它所依赖的 `test_table_obstacle_termination` 前面**，把那个模块的 isaaclab stub / table_tennis 包安装提前到了全量跑的最前段，别的模块跟着受影响，`monkeypatch` 打到了另一个 `table_frame` 对象上 | 文件改名 `test_table_proxy_plant_identity.py`，排到宿主模块之后；这是既有的全局状态脆弱性，不是本轮语义问题，但不该由本轮引进 |


## 7. 实跑

Pod1 GPU1，worktree `/workspace/franco/proxy0807_isaac_20260808`，
commit `d724faaa`，USD bundle
`/workspace/franco/runtime_assets/a3p0807_preconverted_usd_13e5ecfe`。

| 格 | 阶段 | 结果 |
| --- | --- | --- |
| C0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off | materialize | `EXIT=0`，48 s |
| 同上 | recipe | `EXIT=0`，58 s |
| 同上 | **oracle32**（第一个真正走 env step 的阶段） | **`EXIT=0`，263 s** |
| A0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off | materialize | `EXIT=0`，48 s |
| 同上 | recipe | `EXIT=0`，57 s |
| 同上 | oracle32 | `EXIT=2`，**不是被这次改动挡的**，见下 |

**C0 的 `oracle32` 过了，这是本轮的验收线。** 上一轮它在第一步就被
`robot_hit_table live USD bundle file differs from pin: .asset_hash` 拒收；
现在它跑完 263 秒、`robot_hit_table` 在终止表里是 `False`。也就是说
新的六个摘要、派生证明、62 分量产物三样都在活体上过了。

**A0 的 `oracle32` 停在验收指标，不是停在门。** Kit 起来了、跑完了 995 个控制步、
`teacher_qdes_oracle_32ep.json` 也写出来了；拒收的是 A211 的验收判据：
32 个 episode **全部在击球前以 `robot_hit_table` 终止**，`exact_strike` 分母为 0。

这一条**不是本轮造成的**。同一个格、同一个族、在退役 0409 盘上的历史读数：

| 跑 | 盘 | control_steps | 终止 | 击球 |
| --- | --- | --- | --- | --- |
| `s8_bridge_20260806/a0_oracle32_20260806_s8r4` | 0409 | 1005 | `robot_hit_table` 32/32 击球前 | 0 |
| `s10_cells_a_20260806/a0_oracle32_s10r5` | 0409 | 1005 | `robot_hit_table` 32/32 击球前 | 0 |
| **本轮 `a0_oracle32_pxy8`** | **0807** | **995** | `robot_hit_table` 32/32 击球前 | 0 |

形状一模一样，episode 长度只差 1%。**A211 的 teacher-qdes oracle 在加了桌子之后
本来就打不完一拍**，这是先于本轮就开着的问题，和 §4.b 那 1025 帧是同一件事的
两个面。本轮既没造成它、也没修好它，**更没有为了让它过而动任何一道门**。

真资产层的补充收据（在 pod 上对真 bundle 跑，不是夹具）：

| 场景 | 运行时门 | 生产器 |
| --- | --- | --- |
| 0807 bundle | **PASS**（tree `365ba37e…`） | `--check` 逐字节复现，62 分量，`896a5c96…` |
| 0409 bundle，摘要不动 | REFUSED：`file differs from pin: .asset_hash` | REFUSED |
| 0409 bundle，**六个摘要全部重新盖章** | REFUSED：`converted from a different asset package: config.yaml asset_path=…/agibot_a3/urdf/model.urdf` | REFUSED（同一句） |
| 0807 bundle ＋ **0409 URDF** | — | REFUSED：`left OmniPicker3 gripper collision links are not all materialized`（20 个全列出来） |


## 8. 这份读数不能和什么比

- **今天是"新机器人 + 旧目标"。** 换动作库本轮不做（另一整套阻塞：in-service
  闭包 28 文件 / 62 指针、motion 身份在 5 个 leaf 名下 22 处、`ACTION_FACTS` 里
  四个从动作测出来的数必须重测、`action_uid` 两处推导冲突）。
- **拍点 site 已经移了 `0.502 mm`。** 本轮独立量过：同一组关节角下，
  右腕拍面中心在两副盘上差 `0.5014 mm`（出生姿态）/ `0.5028 mm`（教师 frame 0），
  全身 link 原点最大位移 `1.83 mm`（`right_ankle_pitch_Link`，就是厂商修的
  ±1.5 mm 踝关节不对称）。**所以现在的读数不能直接和换库之后的比。**
- **`configs/a3_motor_tn/*`（torque-speed authority）没有接线**，`source/` 下零消费方，
  本轮没动。
- 退役 0409 那六个摘要在仓里的六份拷贝，本轮动了第二、三、四份
  （生产器、`terminations.py`、产物自陈）；第一份 `launch_n1_reward_screen_diagnostic.py`
  在 `5d6924d7` 已动；剩下两份
  （`configs/phase1_balance_action_slew_launch_manifest_20260720.json`、
  `configs/phase1_lower_body_stability_launch_manifest_20260720.json`）
  是历史作业记录，不动。

## 9. 2026-08-30 v2：从 single OBB 空角改为双端 actual-collider union

### 9.1 根因与不可改的边界

frozen-teacher replay 的 component 55
`right_hand_pingpang_link.stl` 在现有 20 mm 桌面守卫上给出
`-1.018 mm` SAT overlap，但原 mesh / convex hull 对同一桌面的垂直净空是
`72.239 mm`。这是非凸手掌+球拍被一个大盒子包住后产生的空角误报，
不是真实触桌。处置边界因此是：

- `20 mm` no-touch 原样保留；
- 不删 component，不缩 margin，不用 raw surface 冒充 backend collision volume；
- 本轮合同只声称 component55 target refinement 保守覆盖 Isaac 与
  MuJoCo 的实际 live collider 并集；其他 61 个 source component 仍沿用
  原 shared conservative source proxy，不宣称已证明全机器人 actual-collider union。

这也纠正一个旧表述：两端并不是都用整个
`right_hand_pingpang_link.stl` 的 whole-mesh convex hull。Isaac FullMDP split live stage
关掉旧的 merged wrist collision，打开 red / black / handle / wrist-shell 四个
`convexHull` mesh；其中 handle 是原 hand STL 的完整三角形子集。MuJoCo
canonical MJCF 则用 optimized wrist/racket mesh，再加 palm ellipsoid、finger/thumb
capsule 和 handle capsule。所以 v2 同时封存 Isaac source-hull cover 与 exact
MuJoCo collision inventory，不再用一句“双端同 STL”代签。

### 9.2 最小保守分解

`right_hand_pingpang_link.stl` 共 `10,258` 三角形、`4,584` 唯一顶点。在钉死的
NumPy `1.26.4` / SciPy `1.11.4` / Qhull `Qt` 上，全局 convex hull 为
`307` 顶点、`610` facet。生产器用一个严格内部点将每个完整 facet
连成 `610` 个 tetrahedron，每个 tetrahedron 只归一个 leaf；后验钉死：

- tetra 体积和与 hull 体积 `0.0016095072911877638 m³` 的绝对误差为
  `6.505e-19 m³`；
- 两个 leaf 各 `305` tetra，每个 tetra 的四顶点均在所属 OBB 内；
- PCA 基的 eigenvalue 降序、`1e-10` 近简并 tie group、轴符号和右手系
  都显式 canonicalize，不接受平台任意 eigenvector 符号；
- mesh frame 内序列化为 float32 后做全顶点覆盖后验，并有
  `1 µm` 外扩。这不再被写成“所有 row owner-frame 序列化后全局已验证”；
  后者必须等 final-owner transform 的逐 row containment 补齐后才能声称。

只有这一个 source component 从 1 leaf 切为 2 leaves；其余 61 个仍为
1 leaf。因此 source component 仍是 `62`，runtime proxy row 是 `63`，相对 v1
只增 `1/62 ≈ 1.6%` SAT row，不是 64-leaf 或成倍 guard。两个 row
共用同一 `source_component_id`，但各有独立 `component_id=#obb0000/0001`；
报表不得把 `63` 误写成 `63` 个 source mesh。

MuJoCo 实际 wrist 形状不全在 source hull 内，所以生产器又将每个完整
mesh hull / ellipsoid / capsule 分配给其中一个 leaf，用解析 support interval
扩展该 OBB，再做 float32 后验。最坏 projection interval 仍有
`1.001 µm` 内含 reserve；没有靠表面采样声称覆盖整个几何体。

### 9.3 冻结 witness 与身份

在 frozen component-55 owner pose 上，v2 两个 expanded leaf 对原 20 mm table-top
guard 的最小归一化 SAT reserve 为 `+24.884 mm`；将同一 table AABB 六面
再扩 `5 mm` 后仍为 `+19.884 mm`。这比“把 -1.018 mm 翻成一个很小正数”
更强，且不改 20 mm 语义。回归用同一 owner pose 与 production 15-axis SAT
直接钉住这两个数。

v2 工件与身份（分支候选，不代表现役 run 已替换）：

| 项 | 值 |
| --- | --- |
| schema / artifact type | `2` / `a3_table_collision_component_multi_obb_v2` |
| artifact file SHA-256 | `c15b132c86de9b06dfc8b6e8838ddc2848b8b9246d1f4d34ebc0ca070ea2f033` |
| artifact `content_sha256` | `1a5e6f22f1933d2f2d8bdf87b77397b8f006b815e421bea9dfa3ed6710458a78` |
| plant identity | `a3_collision_proxy_plant_identity_v2` |
| target hull geometry SHA-256 | `96b05900b79150be2546423adf8a4f2e9db700ed9870e1c9148a57a72ee288a0` |
| MuJoCo MJCF SHA-256 | `70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a` |
| MuJoCo collision binding content SHA-256 | `e81780c50f3c92da4ea6561c26c7da3faf0be24ecd8cfd7025cde159b9bdad3e` |

loader 只消费已物化字节，不在训练热路重算 hull/PCA。Isaac 与 MuJoCo
都校验 v2 schema、file/content SHA、plant identity、source/proxy count，并校验
exact MuJoCo MJCF / collider inventory / binding SHA。任一漂移都 fail closed。

2026-08-30 的 consumer 补洞将 diagnostic witness 升为 v2：`0..62` 是
proxy row，`63` 是 runtime racket blade，同时携带 leaf-specific
`component_id` 和 parent `source_component_id`。双 loader 额外拒绝 source ID
数量漂移、`proxy_box_index/count` 不连续或有洞的 split mapping。Isaac
attribution 二次读取也重验 exact file SHA，不再可用替换工件制造
geometry/label TOCTOU。

Pod exact `213c8622` 上的 materializer `--check` 通过，扩展 CPU union
包含 teacher replay / initial-wait / MJLab keepout，结果 `200 passed, 2 skipped`
(`20.93 s`)。两个 skip 均是显式 CUDA opt-in；GPU1 虽无显存进程，但仍有
Isaac/MuJoCo owner lock 且与现役 GPU2 共享 NUMA，本轮没有越权运行 CUDA。

### 9.4 证据边界

此 v2 只修 shared geometric table guard 的 conservative cover，不改 Reward、teacher、
policy observation、PPO、table margin 或 active run。生产器与聚焦回归只允许在
Pod exact checkout 执行，命令见
[Build And Test](../../operations/build_and_test.md#a3-table-collision-proxy-v2)。通过回归只能将
v2 称为 merge candidate；未在 clean adopted source 上重放真实双端的同一
teacher tape 前，不能把它写成新的训练/物理结果。

Pod1 final exact checkout
`/workspace/franco/mktemp/multiobb-44011beb.exact` 已给出：

- 钉死系统 Python 的 `--check` 逐字节通过：`63 components`，file SHA
  `7f26e55b…07cc`；
- materializer + Isaac/Mu loader + plant identity + table SAT + live-constant 联合回归
  `152 passed in 13.64 s`；其中新回归又将 Isaac split producer 的
  `2806/178/178/9654` 个完整三角形逐项证明为已覆盖 source mesh 的子集；
- MJLab keepout 依赖回归 `16 passed, 1 skipped in 2.93 s`；唯一 skip 是显式
  opt-in 的 CUDA direct 用例。GPU0/2 当时有 active run，GPU1 与 GPU2 共享
  NUMA，故本轮按算力纪律不为了一条 device test 污染现役训练窗口。

在首次联合回归中另发现一个本分支造成的旧 fixture 假红：tensor shape 仍写死
`62`，已改为读 production count。live-constant 的另一个文本 mutation anchor
在基线 `3f0b80d3` 上也因两处相同配置而失败；后继只将变异限制在具名
`HOPEActionBallTerminationsCfg` 内，不改 production code，因此 final 152 条全绿。
