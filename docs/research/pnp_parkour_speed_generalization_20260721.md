# "PNP 跑酷论文"考证与速度泛化情报（2026-07-21）

结论先说：**没有找到叫 "PNP/PnP" 的跑酷论文**。最接近的是 **PHP — Perceptive Humanoid Parkour**
（arXiv 2602.15827，项目页 php-parkour.github.io，"php-parkour" 口头转述极易变成 "PNP 跑酷"）。
但经全文核对，**PHP 里没有参考动作速度增广/重定时**。目前公开的 2025-26 跑酷论文里，
没有一篇把 "reference 速度泛化" 当主贡献并给消融。"速度泛化是最高价值轴" 这条情报
更可能来自口头交流本身，而不是某篇跑酷论文的实验结论——**spdmix 臂就是在替社区补这个消融**。

## 1. 搜索过程（fail-closed，供复核）

- arXiv 元数据 API：`ti:parkour` 全量（2022-2026 共 23 篇，逐条看过标题）；
  `all:parkour AND all:PnP` = 0 篇；`all:PnP AND all:humanoid` = 0 篇；
  `abs:"plug-and-play" AND abs:parkour` = 0 篇；`ti:PnP AND cat:cs.RO` 只有位姿估计类（PnP 算法）。
- Web 搜索："PnP parkour"（只命中游戏店/健身房）、"PNP 跑酷 人形机器人"（命中 PNP Robotics 公司，
  是卖 Franka 机械臂的，无跑酷论文）、RSS/CoRL 2026 + parkour + retiming 等组合，均无。
- 全文抓取并 grep 了 13 篇候选（关键词 speed/retim/time-scal/playback/phase rate/TOPP/warp）：
  PHP 2602.15827、Deep Whole-body Parkour 2601.07701、Hiking in the Wild 2601.07718、
  TTT-Parkour 2602.02331、ParkourFormer 2605.25782、PUMA 2601.15995、ETH 运动生成跑酷 2604.17335、
  HIL 2505.12619、Acrobotics 2509.02727、Parkour in the Wild 2505.11164、SWAP 2606.19928、
  OmniRetarget 2509.26633、SONIC 2511.07820。**没有一篇做 reference 速度增广**。
  （2604.17335 只做地形几何增广 §III-A2；HIL 的 DTW 只用于评测；PHP 见下。）

## 2. 最接近名字的候选：PHP（已下载 papers/2602.15827v2.pdf，6 页）

- 做什么（人话）：把人类跑酷动作剪成原子技能库，用 motion matching（特征空间最近邻）
  在线拼成长程参考轨迹，逐段训跟踪 expert，再 DAgger+RL 蒸馏成单个深度视觉学生策略；
  G1 实机翻越 1.25 m 高墙、约 3 m/s 冲刺跨栏（p1 图 1）。
- 与 BeyondMimic 的关系（p5，§III-C1）：**expert 跟踪完全沿用 BeyondMimic**——
  DeepMimic 式跟踪奖励 + action rate/关节限位/碰撞罚 + 跟踪超差提前终止 + 轻量域随机化；
  重定向用 OmniRetarget。就是我们 HOPE 同款技术栈的跑酷版，配方可迁移。
- 它的"泛化"来自哪里：不是速度缩放，而是**入场状态多样性**——motion matching 生成
  不同接近距离、不同步态相位的技能入场（p4，"从跑步的不同相位起跳/跨越"），
  加 pre-skill 行走时长随机 [0.1, 3] s；速度控制靠 2D 速度指令挑选参考（p4-5），
  即靠**数据覆盖**而非 retiming。另有 Adaptive Sampling（p5-6）：失败多的动作段加采样，
  高墙攀爬没它不收敛——这条对我们 C+S0 的难段（如快摆速段）也值得记一笔。
- 明确核对：全文无 speed scale / retime / time-warp / playback / TOPP 字样（速度只出现在
  "3 m/s 实测速度"和相机模糊讨论里）。**不要引用 PHP 来背书 spdmix。**

## 3. 真正做了参考速度增广的两篇（内容最接近，非跑酷）

- **AdaMimic**（arXiv 2510.14454，G1 实机）：相位推进 φ_k = φ_{k-1} + Δφ_k，
  Δφ_k = Δt / T_motion（式 (1)）；第二阶段训一个 **phase adapter** 在线输出 δ 相位增量
  叠加到基准相位率上（= 在线 phase-rate 缩放），tracking adapter 按 δ 缩放动作补偿（式 (9)）。
  **有消融**：AdaMimic 全量 > AdaMimic-Stage1（固定速度跟踪）> DeepMimic-Adapt（规则增广），
  远跳任务上自动"拉长滞空段、着陆段多补偿"（图 6）。证明**让时间轴能伸缩确实降低跟踪误差**，
  但它是"策略自己选速度"，不是"命令策略换速度"。
- **Humanoid-GPT**（arXiv 2606.03985，§3.1）：离线 time-warping 增广——每条序列统一加速、
  减速各若干档，数据集约 ×5，目的写明"提高对动作速度的鲁棒性"；没公布具体倍率档位，
  也没有单独消融这一项。当"业界默认做法"的旁证用，不能当定量依据。

## 4. 给 HOPE 的三条落地建议

1. **在线 speed_scale_range：spdmix 用 [0.8, 1.2] 起步是合理的，但要当成假设检验来跑。**
   没有任何公开论文给出"最优范围"消融，[0.8,1.2] 相当于 Humanoid-GPT 式温和 warp。
   实现上必须是**一致的时间伸缩**：时长 ×1/s 的同时参考关节速度、根速度 ×s，
   不能只跳帧不改速度项，否则跟踪奖励里的速度项自相矛盾。spdmix 若稳，下一波再试 [0.7,1.3]。
2. **TOPP 烤入资产管评测，在线缩放管训练。** 在线均匀缩放（尤其 >1.0×）会产生动力学
   不可行的参考，直接在缩放参考上评测会把"参考本身做不到"记成"策略不会"。建议评测用
   TOPP 重定时资产摆一个固定速度阶梯（0.8/0.9/1.0/1.1/1.2×），每档参考都是可行的，
   这样速度泛化曲线才干净；训练侧继续用廉价的在线缩放拿覆盖。
3. **相位观测：现阶段不需要加。** 我们和 PHP 一样是 BeyondMimic 式"参考相对量"观测
   （当前帧参考位姿/速度 + 骨盆位姿误差），缩放后的速度信息已经在观测里，策略能感知节奏；
   注意不要暴露绝对动作时长/绝对时间类特征。只有当固定倍率跟踪在快档（1.2×）系统性失败时，
   才值得学 AdaMimic 加"策略自选相位率"的通道（他家消融显示这在难段收益最大）——记为后手。

## 5. 遗留 caveat

- "速度泛化是最高价值泛化轴"目前**只有口头情报支撑**，spdmix 结果出来前不要写进结论性文档。
- 若 Franco 能回忆起 "PNP" 的原始出处（讲者/项目页），值得再核一次；本次检索截至 2026-07-21，
  覆盖 arXiv 元数据 + 全文 grep + 中英文 web 搜索，未发现同名论文。
