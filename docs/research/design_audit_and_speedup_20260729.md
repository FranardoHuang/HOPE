# N1 设计背书审计 + 训练加速尽调(2026-07-29)

> 用途:交给 Codex 的自足工单。每条带 file:line、具体改法、验收标准和文献引用,不依赖产生本文的会话上下文。
> 证据来源:①16-agent 设计背书审计(4 面 × 盘点/上游比对/文献检索/内部背书四段);②3 路吞吐基准检索(官方 benchmark、tracking 栈、reset 实现);③终止处理文献尽调;④pod1/pod2 现场实测(2026-07-29,V100×3,4096 env)。
> 行号注意:工作树在持续被编辑,hope_actions.py / hope_rewards.py / terminations.py / hope_env_cfg.py 的行号以当日快照为准,漂移 ±90 行内;函数名/字段名为稳定锚点。

---

## 1. 设计背书矩阵(57 项)

四个设计面(reward / 环境 / curriculum / 算法)共审 57 项承重设计,每项四段流水:盘点(file:line)→ 与上游 BeyondMimic 层(仓库内 `tracking_env_cfg.py` 即上游快照)/已引用 paper 比对 → 偏差逐条外部文献检索 → 仓库内消融/EXP/裁定核查。判定分布:

| 判定 | 数量 | 含义 |
|---|---|---|
| backed_external | 17 | 有直接文献/成熟开源栈背书 |
| backed_internal | 7 | 有自家 wave A/B 或 ablation 记录 |
| ruling_only | 29 | 只有裁定,无任何证据记录 |
| **contradicted** | **2** | **被证据(含自家数据)反向** |
| **UNBACKED** | **1** | **无任何背书且高承重** |

### 1.1 必须处理的三项

**① tracking 权重 4.0/0.5/0.5 —— UNBACKED(头条)。**
事实链:名义配方 393.4/295.1/229.5 因 CLI"后写后赢"被事故性覆盖为 4.0/0.5/0.5(差约 113×,证据:EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md E1 composed env.yaml 审计);随后 EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md §3 把事故值**转正**("393.4/295.1/229.5 只作后续消融");当前 live N=1 波即跑在 4.0/0.5/0.5 上。预登记的配对 A/B 从未执行。引用先例 HITTER(高权重窗口项)方向相反。
工单:执行预登记 A/B(4.0/0.5/0.5 vs 393.4/295.1/229.5,同 seed 同 clip),在结果落地前,所有 reward 结论标注"低权重前提"。

**② 收入分层 1:3:7.5 冻结阶梯 —— contradicted。**
它的 landing 档在自己的单变量 wave 里输给 1.2×(1648.8 vs 791.9,单 seed、~4k 截断,已进 banked 默认);且质量档因①从未真正生效——"banked 波 ~50% legal 率归功于 1:3:7.5"的叙事建立在未生效配方上。
工单:与①同一 wave 补 ladder_flat 对照(预登记 #9 从未发射);文档里更正归因。

**③ 参考偏离终止相位门 —— contradicted,现状为全相位裸奔。**
b00a936e 设计(center 保留终止、marginal 起关闭)被自家 4096-env 数据推翻(center 相位 reset 风暴卡死早期训练,G05 2026-07-29 记录);应对是正式 N5 argv 钉死 `+task.racket.reference_guard_mode=metrics_only`——**所有相位都没有追踪保真终止,且无替代门**。这是 feasibility-vs-fidelity 的文档化风险:policy 可任意偏离老师,只有 metrics 在看,直接威胁 sim2real 动作质量。
工单:见第 2 节裁定(替代门方案)。

### 1.2 高承重 ruling_only(按危险排序)

1. **rolling-30 环门(new-band direct-count 2/3/4)**:marginal 晋级唯一大门;零外部先例("无发表系统单独审计新增带")、零经验验证;判定统计量 24h 内纯纸面改版一次(81d598d7→b00a936e);n=30 直数下 1–2 次偶然失败翻转 expand/lock;设计上无任何机制复查已错误晋级的带(heldout 只测全域)。工单:heldout 加 new-band 专项切片。
2. **σ 静态 + 死区**:>~3σ 零梯度已咬过产线(正手 face 漂到 93–150°,EXP-V2-REWARD-FREEZE §0.11;rescue 臂从未发射);排队的 sigma_static 消融(§0.9 #5)从未跑。工单:补 rescue 或 σ 课程消融;监控各质量通道的 kernel 饱和率。
3. **-72 死亡定价**:内部已标记敏感轴(姊妹臂 4× 摔率差);-36→-72 翻倍无推导记录;death09 消融"待点头"后无下文。
4. **entropy 0.01 全局钉死**:上游谱系 0.005/0.004;J1 消融承诺后降级"记录性核查"、从未跑;过高熵把 σ 顶向 2% 硬限带,与 qdes/actual 越限风暴可能同源(run_pingpong_end_to_end.md:289 已记录晚期 std 膨胀判读风险)。
5. **max_iterations=3e11 无界哨兵**(e8ffe8e8 引入,无理由记录):"看 strike_success 平台期手动停"对"聚合曲线掩盖分层失败"零防御——此病仓库已吃过四次。工单:long 档钉 20k–25k(与目标一致),平台期判读改按 per-clip/per-family 切片。
6. **balanced round-robin 放弃 hard-negative mining**:落后 clip 恒 1/5 数据、curriculum 停 level 0;PHC 在万 clip 规模用失败加权采样(反例);交叉点未知。记录在案即可,N=5 规模暂不动。

### 1.3 外部矛盾提示

**seg_start 定点出生 vs DeepMimic RSI 消融**:内部证据真实(mid-swing 出生被薅:ep len 18→7.3、摔 ×10;随机相位仅 38.7% 到击球帧),判 backed_internal;但正面对撞 DeepMimic 发表的 RSI 消融(随机相位对难动作必要)。建议排一个 pinned-vs-RSI 对照,给"过拟合单一 ready 姿态→实机初始姿态偏差退化"风险定界。ready 姿态本身 DEFINITIONS.md 标注 uncertified。

### 1.4 放心清单(有背书,不动)

非对称 actor-critic(强)、解析虚拟球不走 PhysX(强)、counter-RNG 确定性发球(强)、摩擦 DR 对齐 MuJoCo(强)、2% inset pre-apply guard(wave 证据)、question-bank/exam 拆分(自家 ablation + Procgen 先例)、strict exact-resume(强)、obs 归一化(强)、boundary-deferred save(强)、击球窗三核族(HITTER 直接先例)、上肢解耦 hold 静默(ExBody/HITTER/EMP)、32 臂 frontier 采样与单均匀抽样(OpenAI ADR arXiv:1910.07113)、qdes clamp 训练=部署(自家 ablation)。

### 1.5 欠账消融队列(承诺过、从未跑)

| 消融 | 预登记处 | 状态 |
|---|---|---|
| 4.0/0.5/0.5 vs 393/295/229 | EXP-EFFECTIVE-REWARD-CAUSALITY | 未跑(头条) |
| ladder_flat(#9) | EXP-V2-REWARD-FREEZE §0.9 | 未发射 |
| sigma_static(#5) | 同上 | 未发射 |
| death09(-1800 vs -900) | 同上 | "待点头"无下文 |
| J1(entropy) | ppo.yaml 行内史 | 降级后消失 |
| base03(base_frac 0.3) | EXP-V2-REWARD-FREEZE | 未发射 |
| qdes 罚 -5 vs -20 | EXP-EFFECTIVE 07-29 | 预登记,待跑 |
| pinned-vs-RSI 出生 | 本文 §1.3 | 新增建议 |

---

## 2. 参考偏离终止裁定(BeyondMimic 家族证据)

### 2.1 家族各栈的原设(全部已核实)

| 栈 | 参考偏离终止 | 阈值 | 自述定位 |
|---|---|---|---|
| DeepMimic (1804.02717) | 摔倒 ET(躯干触地) | — | ET 与 RSI 并列为论文两大训练贡献 |
| PHC (2305.06456) | enableEarlyTermination=True | 全局 0.25 m | 数据分布/课程机制 + 失败序列 hard-negative mining |
| H2O (2403.04436) | 三条 ET:低高度/姿态/参考距离 | **全身平均 link 距离 0.5 m** | 原文明确"sample-efficiency 装置" |
| MaskedMimic (2409.14393) | 关节偏离终止 | 0.25 m(平地)/0.5 m(地形) | 同上 |
| BeyondMimic (2508.08241) | anchor_pos(z)/anchor_ori/ee_body_pos | 0.25 m / 0.8 rad / 0.25 m | 与本仓 tracking_env_cfg.py:263–283 逐字一致(上游 repo 已核) |
| AMP (2104.02180) | 无参考终止(风格判别器替代) | — | 反例:代价是弱保真、可游走 |

**结论一:没有任何 tracking 栈用"纯罚分"替代参考偏离终止。** DeepMimic 的机制论述(原文引句):无 ET 时早期数据被"在地上徒劳挣扎"的样本淹没,是 class imbalance 问题;ET 同时是"策展机制,偏置数据分布"。这就是 §1.1③ 说的机制差异:**qdes 越界是单步指令事件(clamp 后无害),参考偏离是复利型状态发散**——一旦脱流形,指数核模仿奖励饱和归零、梯度消失、rollout 数据脱分布,终止承担的是数据质量职能。全相位 `metrics_only` 等于走进 AMP 角落却没有 AMP 的判别器。

**结论二:但"他们的 ET 免费,我们的不免费"。** 参考栈 reset 是 O(kernel)(§3.2),ET 只花样本效率;BeyondMimic 官方 issues #49/#56 里"episode 长度恒 1"的极端风暴,用户抱怨的全是学习、无人提迭代时间。我们的 reset 带 per-env Python 仪式,才把 ET 变成墙钟灾难——§3 修复落地后,ET 的成本模型回到正常,"为了速度砍终止"的理由就消失了,只剩"为了早期学习"的理由,而那个理由文献里有更好的答案(见 2.2)。

### 2.2 早期大规模终止的发表解法(不是砍终止)

1. **阈值宽→紧课程**(Babadi et al. arXiv:1907.11842,locomotion imitation 直接先例):起步宽(参照 H2O 的 0.5 m 全身平均——比我们 0.25 m 逐体 z-only 宽一倍还换了统计量),随训练收紧到目标阈值。
2. **CaT 式连续截断**(arXiv:2403.18765):把 tracking 偏离当软约束,`rewards *= (1−δ); dones = δ`,δ 随偏离深度连续、**仿真不重置**;p_max 0.05→0.25 课程。三行实现,保留"偏离降低回报"的教育信号,免掉 mass-reset;在我们 reset 仍偏贵的过渡期额外省墙钟。对 tracking 偏离用 CaT 是外推(其论文软约束为 torque/velocity/action-rate 类),机制上同构。
3. **Reset 后 grace window**:出生后 K 步免参考终止,让新生策略离开出生态再计偏离(工程惯例,多栈 reset 后有 push/稳定期同型)。
4. **PHC 式 recovery 训练**(2305.06456):学会回到参考而非重置——最接近"不终止"的已发表方案,但属于后期能力项,不解早期问题。

### 2.3 分场景推荐矩阵(本仓)

| 场景 | 推荐 | 理由 |
|---|---|---|
| 诊断 reward screen(现状) | `metrics_only` 可暂留,但必须:①每迭代记录偏离分布(anchor/逐 body 偏离直方图);②加逐 body 触发归因遥测(当前 115/迭代不知是腕还是踝在烧——腕=探索被罚,踝=在摔,处置完全不同);③禁止进正式波 | 短期止血,读数完整性优先 |
| 正式早期 | 二选一:阈值课程(ee_body_pos 起步 0.5 m 或换 H2O 的全身平均统计量,随 curriculum 收紧回 0.25)或 CaT 式连续截断(tracking 偏离 p_max 0.05 起) + grace window | 有直接文献先例;保留数据策展职能 |
| 正式后期 | 恢复相位门语义:任务相位接管后参考终止退场(方向本身有 Termination Curriculum 家族背书;§1.1③ 被推翻的是"center 相位硬保留"的实操,不是"晚期退场"的方向) | 模仿是脚手架,收入分层如此设计 |
| 永远保留 | 摔倒 / 桌撞 / joint_actual 硬越限终止 | 全家族一致,不可恢复态 |

工单:①先加逐 body 归因遥测(便宜,一天内);②正式 N5 argv 把 `reference_guard_mode=metrics_only` 替换为阈值课程或 CaT 实现之一,过 G05 前不得裸奔;③ee_body_pos 的 0.25 m z-only-per-body 统计量重审(vs H2O 全身平均 0.5 m)。

---

## 3. 采集/Reset 加速(带背书)

### 3.1 "大家都这么慢吗?"——不是

同几何(4096 env × 24 步 rollout,rsl_rl PPO)公开基准与内部实测对照:

| 栈 / 任务 | 硬件 | steps/s(整管线) | collection/迭代 |
|---|---|---|---|
| Isaac Lab 官方 G1 rough(29DoF 类人) | RTX 4090 | 82,000 | ~1.2s |
| 同上 | L40 | 62,000 | ~1.6s |
| H1 rough(19DoF) | GB10 DGX Spark | 65,955 | 1.256s(官方日志原文) |
| ANYmal rough(Rudin 2021, arXiv:2109.11978) | RTX A6000 | ~123,000 | ~0.8s/迭代整段 |
| MaskedMimic(transformer 策略) | 4×A100 | ~24.8k 聚合(~6.2k/卡) | — |
| **本仓 legacy(s1_wave1/3)** | **1×V100** | **24–25k** | **3.8–3.9s** |
| 本仓 chingmu | 1×V100 | 15–17k | 6.4s |
| **本仓 action_ball(现役)** | **1×V100** | **1–4k** | **25–116s** |

V100 外推健康带(基于 4090/L40/GB10 锚点 + Isaac Lab 论文"开销主体是 CPU 编排与 kernel launch 而非物理"的结论):**20k–40k steps/s**。legacy 24–25k 恰在带内——**机器与场景无罪,超额时间全部来自 action_ball 层**。公开日志的 collection:learning 比在 5:1–25:1(收集 1–4s),没有任何公开记录接近 25–116s。Isaac Lab 论文实测 manager 抽象本身只贵 ~3.5%(arXiv:2511.04831)——"每步 Python 应占个位数百分比,不是数倍"。

### 3.2 参考栈的 reset 成本:O(kernel 数),不是 O(env 数)

- legged_gym `reset_idx`:全批张量化,~15–25 个小 kernel + 2 次 indexed sim 写,**0.1–1ms/批,与批大小无关**;唯一 Python 循环遍历 ~10 个 reward 名。
- Isaac Lab manager reset:O(term 数) 的 Python 调度(5–20 个函数调用)+ 批张量;零 per-env Python。
- BeyondMimic 上游 `_resample_command`:一次 multinomial + conv1d + ~10 elementwise + 2 次 indexed 写;零 per-env Python(本仓继承层同形,见 commands.py ~4040–4180、5934–5935——**慢不在继承的 BeyondMimic 机器,在叠加的 action_ball 治理层**)。
- DeepMimic RSI:一次 RNG + 一帧插值 + 两个向量拷贝,论文视为免费操作。
- **全生态检索结论:没有任何公开 RL 栈在训练热路径做 per-reset per-env 的 Python 记账**(收据/哈希链/台账)。审计粒度一律是 run 或 checkpoint(确定性靠 seed+config 离线重建);唯一的哈希链审计文献在 LLM 训练侧,预算也是每 optimizer 步 ~3.4ms 且建议移出热路径。
- 直接旁证:BeyondMimic 官方 repo issues #49/#56——用户遇到"Mean Episode Length 恒为 1"(每步全 env 重置的极端风暴),抱怨的全是数据/学习问题,**无人提到迭代时间**;reset 在参考栈里墙钟免费。
- 量化:100-env reset 批,参考实现 ~1ms;叠加 per-env Python 仪式(收据 dict + canonical JSON + SHA-256 + 台账)~0.1–1ms/env → 10–100ms+/批,高 1–2 个数量级。按现役 2,800–16,500 reset/迭代 → **每迭代 0.3–16+ 秒纯 Python 只花在 reset 上**,与实测 25–116s collection 和单线程钉死完全自洽。

### 3.3 我们的时间去哪了(已核实清单)

**当前 #1(live 数据):`joint_actual_forbidden` 风暴。** qdes 投影上线后(runs 7a14b0b9/5dbb4e58),finite-qdes reset 墙消除,但**投影罚采样为零、吞吐未恢复:实际关节动态过冲终止 ~4,700–5,100 次/update**。处理见 §4。

**每步(与 reset 无关的税,≈6.4s 基线相对 legacy 4.7s 的残差主体):**
- 60–120 次隐式 GPU→CPU 同步/policy step(`float(reduce)`/`bool(x.any())` 形态):`_update_metrics` 全局 EMA 簇 10 次(hope_commands.py ~17858–17898)+ 每 clip 循环 7×N(~18035–18045);`_vb_evaluate` 6+3N 次,含每步同步的 `bool(exact_strike.any())` 早退守卫(~16405);`_compute_strike_timing` 每步被调两遍(~15185 与 ~17746);MotionCommand `_update_command` ~8 次(commands.py ~6056–6134)。每次同步排空整条 GPU 流水线——GPU 4–12% + 单 Python 线程钉死即此签名。
- joint-safety ledger:每步 5 次 × 7 个 (4096,31) clone(~11.5MB/步,hope_actions.py ~272–292);4096 元素收据字符串每步 Python 走三遍(~1437、1098–1122,只在 reset 变化,可 hoist)。
- 每步身份重验(31 关节名单重建比对,terminations.py ~152–196 等 ~5 处)与 counter-rally per-env dict 往返(hope_commands.py ~16740–16778)。

**每 reset(风暴期主导):**
- `_retire_many_diagnostic`/`_request_many_diagnostic` 逐收据 Python 循环,每张做 assert_contract + **未缓存的 `canonical_sha256`**(8 处 @property,action_ball_runtime.py 619/727/1543/1620/2201/2427/2727/4401,每次访问全量 JSON+SHA);
- 终止证明档案逐 env 转录(~22 键 dict、~13 张量切片、~9 次 .item()/env,hope_actions.py ~1990–2110;近期修复只封了内存上限,CPU 构造仍逐 env);
- 快路径残留小 O(N²):draw 区间 any() 全表扫描(action_ball_runtime.py ~7836–7841 + 8296–8309,可用 per-uid running max 归 O(1));`_resolve_pending` 每 env 3 次 `.item()`(commands.py ~3091–3131)。

**已排除:** Triton 虚拟球快核正常(两 run 日志零回退警告);PPO 学习段只占 0.9–1.3%,不是杠杆。

### 3.4 修复清单(排序 + 验收)

| # | 修复 | 预期收益 | 背书 |
|---|---|---|---|
| 1 | joint_actual 风暴处理(§4) | 25–116s → ~6.4s 量级 | CaT/DeepMimic 家族,见 §2/§4 |
| 2 | EMA/metrics 改 device 累计、每迭代同步一次;去掉 bool 早退;`_compute_strike_timing` 去重 | 6.4s → 5s 量级 | Isaac Lab 论文:每步 Python 应为个位数 %;官方"避免 per-step .cpu()/.item()" |
| 3 | `canonical_sha256` 加缓存槽(frozen dataclass,post-init 后不变) | reset 批成本大降 | 全生态:审计按 run/checkpoint 粒度,不进热路径 |
| 4 | 证明档案转录改惰性(存 GPU 视图,导出时物化);收据元组 hoist 出步循环 | 每步 ~2–4ms + 风暴期更多 | 同上 |
| 5 | draw 区间扫描 O(1) 化;`_resolve_pending` 批量化 | 秒级/初始批 | 常规向量化 |
| 6 | 队列填满三卡(128 核负载 6) | 波吞吐 ×3,零代码 | 单一队列准绳 |
| 7 | (风暴清除后)8192-env A/B 看 steps/s | 可能再 ×1.5–2 | ProtoMotions 维护者:"最好结果 8192 env/卡";每步固定开销摊薄 |

验收线:steps/s ≥ 15k(V100 健康带下沿),采集期 GPU util > 60%,20k 迭代 ≤ 9h/run。

### 3.5 sim 侧旋钮与一个正确性警告

- 桌面/网柱等 collider 用 box/convex 原语,勿用 trimesh(官方 how-to:SDF/trimesh 显著更贵,高长宽比 convex 有 GPU→CPU 回退风险);solver position iterations 按稳定性下探(一个超配的 actor 拖累全场);`replicate_physics=True` + cloner collision filtering 保 broadphase 线性。
- **正确性警告(与速度无关但必须查):** 启动时刷屏的 `GPU contact filter for collider ... is not supported` 不是性能回退,是**该 filtered pair 的 force_matrix/contact 数据返回零或 NaN**(IsaacLab #1995/#4108/forum 290590)。本仓 4+ 个桌面 collider × 4096 env 都在警告名单上。工单:核对哪些消费方在读这些 filtered 矩阵(racket/table 接触判定、`robot_hit_table` 终止——当前恒 0.0000,需要证明是真零而非坏传感器);修法:filtered prim 换原语碰撞近似,或改净接触力+几何归因,避免 per-pair filtered reporting。

---

## 4. qdes / joint_actual 终止方案现状与验收

**已落地(本日工作树,方向经文献背书):** `project_finite_preclamp_qdes_without_termination` + `qdes_projection_penalty`(首发 -5,-20 仅预登记消融;罚投影前超出量 `1−exp(−4·excess/span)`;界内恒等保斜率;device 端 per-joint 遥测零同步;qdes 终止收窄为 nonfinite+物理硬 latch;joint_actual 保留)。业界一致背书:无主流栈对指令级越界 reset(legged_gym clip 默认 100=不裁、Isaac Lab clip 默认 None、DeepMimic 只在摔倒终止);clipped-action problem(Chou ICML'17;Fujita&Maeda CAPG ICML'18)要求罚 pre-clamp 超出或 rl_games bound loss(|μ|>1.1,系数 10,AMP 全家默认)。

**验收(冻结策略、零罚分复测):** per-joint 投影前超出量趋零;饱和占比(qdes 恰贴 clamp 的步比例,抓骑线)下降;违规驻留 <0.5%(CaT 可部署线,arXiv:2403.18765)。

**新头号问题:`joint_actual_forbidden` 风暴(~4.7–5.1k/update)。** 这是**状态级**越限,安全语义比指令级重,不能直接照搬"不终止"。候选(按证据强度):
1. **CaT 式连续截断**:`rewards *= (1−δ); dones=δ`,δ 按越限深度连续给、仿真不重置;catastrophic 约束 p_max=1.0 从头启用、软约束 0.05→0.25 课程(CaT 消融:二值终止 baseline 回报恒零 = 现状的精确画像)。
2. **soft-limit 罚分带 + 只在深越限终止**:legged_gym 惯例(soft_dof_pos_limit ~0.95 罚,真硬限才终止);等价于把现有 -40 barrier 的 margin 当第一道防线、终止阈值外推。
3. **根因检查先行**:与 bh_block qdes 风暴同一嫌疑——参考轨迹逐关节限位余量直方图(bh_block vs bh_loop_c);若参考本身贴限,任何终止/罚分语义都在惩罚模仿,应先修参考或该关节豁免(V12 r8 腕部先例)。
工单顺序:先 3(一次直方图定性),再按结果选 1 或 2;当前诊断波可先用 2 的宽松版止血。

**Codex 两项前置修复的定性(已核):** reference guard metrics-only 解 loop_c 风暴、proof archive 有界化——方向对;要求:有界丢弃打 WARN 进摘要;顺手全仓清点 append-only 容器(已三处:provider_history、sampler 台账、proof archive——模式性问题)。

---

## 5. 工单排序汇总(给 Codex)

1. joint_actual 风暴:参考贴限直方图 → CaT 式或 soft-band 方案(§4)。
2. 每步同步风暴四件套 + SHA 缓存 + 档案惰性化(§3.4 #2–4)。
3. 欠账 A/B 一波双做:权重 4/0.5/0.5 vs 393/295/229 + ladder_flat(§1.1①②)。
4. contact filter 正确性核查(§3.5)。
5. 参考偏离替代门(§2 落地后)。
6. rolling-30 环门加 new-band heldout 切片(§1.2#1)。
7. max_iterations 钉 20k–25k;平台期判读改分层切片(§1.2#5)。
8. 队列填满三卡(即刻,零代码)。

---

## 附录:主要引用

Isaac Lab 官方基准 isaac-sim.github.io/IsaacLab/.../performance_benchmarks.html;Isaac Lab 论文 arXiv:2511.04831;Rudin et al. arXiv:2109.11978;Isaac Gym arXiv:2108.10470;BeyondMimic arXiv:2508.08241 + HybridRobotics/whole_body_tracking(issues #49 #56);PHC arXiv:2305.06456;H2O arXiv:2403.04436(issue #66 V100 数据点);MaskedMimic arXiv:2409.14393;ProtoMotions(issues #163 #178);DeepMimic arXiv:1804.02717;AMP arXiv:2104.02180;CaT arXiv:2403.18765;CAPG arXiv:1802.07564;Chou et al. PMLR v70;Beta/bang-bang arXiv:2111.02552;ADR arXiv:1910.07113;Termination Curriculum arXiv:1907.11842;Not Only Rewards But Also Constraints arXiv:2308.12517;IsaacLab issues #1995 #4108 #364 + NVIDIA forum 290590(contact filter);内部:EXP-V2-REWARD-FREEZE-20260726.md、EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md、EXP-UPPER-N3-BACKHAND-SAFE-WARMSTART-20260728.md、reward_audit_20260725.md、G05、DEFINITIONS.md。

---

## 6. Codex 独立裁定与执行顺序（2026-07-29）

本节不是对上文工单的照单全收，而是结合 `eaf55fba` Pod1 新证据后的采用决定。原则是：
**改变学习目标、样本分布或终止语义的改动要 canary；由身份、数学等价或现场反例直接确定的
正确性/性能修复不做科学 A/B，只做回归、profile 和行为验收。**

本项目训练与部署本体是 **AgiBot A3（31 关节）**。上文公开 G1/H1 benchmark 只能提供栈级
吞吐或算法模式的方向性参照，不能提供 A3 的 joint/limit/plant/ready 数值；本地
`candidate_id=G1` 也只是 A3 grounded-ready 构造候选编号，不是机器人型号。

### 6.1 直接执行，不做学习 A/B

1. **先修 A3 upper q/qd 自相矛盾，而不是先放宽 actual hard edge。**
   `eaf55fba` 的 4096-env updates 0--4 在 PPO 学习前已有
   `2,549/3,986/4,225/4,188/4,162` 次 actual raw-hard terminal，而 q_des projection、
   projection penalty 和 nonfinite 都为零。第一轮把 `canonical_ready_v1` donor 的旧零接触
   结果错误外推到了现役 upper；实际两条 upper 的 12 腿 qpos 已全片恒定并等于 exact AgiBot A3
   grounded-ready candidate：
   candidate SHA-256
   `585bbd7d643857abd08108eac7b4dd997b228d0df1a9921334ca845cd931d71e`，
   receipt file SHA-256
   `ee7dea1aec81169e1d002bbe0b2cfa75c793a97a3f89e1e740d0064dc8be7c46`；
   exact model 为 `A3T2.5_pingpong_0519`；`candidate_id=G1` 只是 A3 构造候选代号，不是
   Unitree G1。actual upper frame 0 已双脚 `3+3` 接触，真实缺陷是恒定腿 qpos 仍带非零腿
   qvel。Pod 原型把 12 列腿 qvel 归零并重建 schema-2 后，right-racket 全帧位姿、线角速度
   bitwise 不变。因而首选是发布 qvel-only A3 资产修复，而不是改 physical hard-edge 语义。
   旧/新学习 A/B 没有意义；需要的是 A3 exact FK/接地不变量、1-env 构造和 4096-env
   reset/strike 回归。
2. **immutable receipt SHA 缓存。** 只给深不可变 frozen receipt 的 canonical digest 做外部
   weak-identity cache，不把 cache 写进 dataclass state；`vars/repr/equality/pickle/deepcopy`、
   wire payload 和 exact-resume 身份必须不变。验收为 Pod 单测和 reset-batch timing，不做训练 A/B。
3. **同一 policy step 的 strike timing 去重。** 只允许正常
   `_update_metrics → _update_command` 顺序消费一次 host step-token handoff；direct call、
   reset/resample、token 变化仍重算。验收为逐步 tensor parity，不做训练 A/B。
4. **合并 metrics [device-to-host transfer（D2H，设备到主机传输）](../DEFINITIONS.md#device-to-host-transfer)。**
   global + per-action error reduction、顺序和逐 step Python EMA 公式保持不变；布尔 count
   在支持的 `N<=8192` 下直接按 float32 求和（整数远低于 `2**24`，因此精确），再把
   `10+8×N` 个 scalar 合成一次有序 D2H。N=1/5/73、
   empty/nonempty strike、adaptive sigma、reference perturb 和 exact resume 必须逐步相同；
   profiler 确认同步数下降即可，不做学习 A/B。
5. **只删除可保持状态的 device→host 分支。** `fired_valid.any()` 可用 device mask/reduction
   替代并在空集保持旧 metric；`exact_strike.any()` 的 no-strike 分支还承担 cache/EMA/lazy-load
   语义，当前不得直接删除。
6. **table contact sensor 正/负控和 deployment qdes 包络对齐是正确性门，不是实验臂。**
   filtered-contact warning 下必须注入一次应触发桌碰和一次不应触发的轨迹；训练额外 5% qdes
   inset 若采用，ONNX/C++/MuJoCo 必须导出同一包络，不能继续称“训练=部署”。
7. **curriculum heldout new-band consumer 是统计完整性修复。** 上文把 rolling-30 写成唯一
   晋级门是错误的；recent-100/rolling-30 只有调度权，formal heldout 才有发布权。schema-4
   已经生成动作×轴×侧 frozen `NB/NB_F`，缺的是 marginal consumer：它仍用全域
   `F/(L+F)`。直接改为 `NB_F/NB` Wilson 区间并保留全域 admission/unsafe blocker，加“全域
   与新增带结论相反”的反例测试；不用新建第二套切片，也不做学习 A/B。

### 6.2 不能视为直接证明，保留单变量 canary

- `4/0.5/0.5` 的实际 tracking 权重虽然来自覆盖事故，但这不能反推出
  `393.4/295.1/229.5` 就是金标准。不同 kernel、dt 和激活率下 raw weight 不可直接比较。
  健康 baseline 跑通后，以 realized income、激活分母、梯度/饱和率和动作保真做小筛选。
- 上文把现有 landing ladder 判成 `contradicted` 也过强：现有证据只是单 seed、约 3.8k、
  未到预登记里程碑，而且高 tracking 权重当时没有实际生效。应立即更正文档归因，但胜负仍需
  单变量 canary。
- landing ladder、negative Reward 剂量、`-72` death、entropy、sigma/rescue、RSI、
  reference guard、CaT、8192 env 和 actual hard-edge 终止放宽都会改变优化问题或数据分布，
  必须小 canary；不得与 grounded-ready/热路径修复混在同一首发。
- CaT 原论文采用基于违规概率的随机终止/未来回报调制；上文
  `rewards *= (1-delta); dones=delta; simulation 不重置` 不是可直接照搬的精确实现，而且把
  tracking/actual mechanical edge 映射到 CaT 属于本项目外推。
- “证明档案存 GPU view、导出时再物化”会让 view 随底层 tensor 继续变化，不是 immutable
  evidence。可做预分配聚合或 terminal 时 clone，但不能把可变 view 当收据。

### 6.3 当前执行顺序

1. 先在 Pod 物化 A3 upper qvel-only 修复；所有 qpos/root/timing 不变，只把恒定腿 qpos
   对应的 stale qvel 归零并重建 schema-2，必须证明 right-racket 全帧位姿/线角速度、strike
   frame 和球题逐位/数值不变。full 仍走完整 ready→core→ready compiler。
2. 同时合入上面的等价热路径修复，在 Pod 跑 focused regression 与 profiler。
3. qvel-fixed upper 的 `1 env × 2 update` 自然完成后，立即跑 fresh 4096-env 五轮：
   episode 必须越过 `t_hit`、strike opportunity 非零、raw-hard/table/fall/nonfinite 不升，
   并报告 collection seconds、environment steps/s 和 exact checkpoint。
4. 若 A3 q/qd 修复后仍有 mass raw-hard，才启用“fresh birth ready 内、substep 瞬态且
   policy-step 末已恢复”的窄 diagnostic canary；task phase/current-post edge/nonfinite
   仍 Done。不得直接上全相位 CaT。
5. 只有健康 baseline 达到至少 `15k environment-steps/s` 且能看到击球数据，才按
   `origin/main` 因果顺序先比较 ball/task pairing、再比较 free/fixed solver，之后才启动
   Reward、reference termination 和 curriculum 剂量比较；否则比较的是 reset/基础设施，不是
   学习设置。

### 6.4 决策账本与执行进度（2026-07-30）

| 分类 | 决策 | 当前证据/进度 |
| --- | --- | --- |
| 直接修 | A3 upper 恒定腿 qpos 对应 stale qvel 归零；qpos/root/timing 不变 | Pod1 已物化两条资产；exact A3 双脚接触、joint/collision 检查通过，right-racket 全帧位姿与线/角速度 bitwise 不变。反手拉/挡新 motion SHA 分别为 `3b7cabde…` / `a228e569…` |
| 直接修 | immutable receipt SHA cache、同 step strike timing 去重、metrics D2H 合批、`fired_valid` device mask | source 已合入；focused Pod 回归 `162 passed`，另有 11 个旧 metric fixture 因未构造两个 runtime flag 失败，真实构造器会初始化，暂不阻塞首发 |
| 直接修 | curriculum marginal formal gate 从全域 `F/(L+F)` 改读已有新增带 `NB_F/NB` Wilson 区间 | source 与反例测试已合入；保留全域 admission/unsafe blocker |
| 先不修 | 用 neutral-arm `candidate_id=G1` 重新求当前击球上肢；用零速度 static-contact LP 阻断动态挥拍诊断 | 前者回答了错误问题；后者在当前双脚接触且几何安全姿态上返回 infeasible，只说明不能被动静止保持，不证明动态 policy 不可训练。两项都保留原始证据，不作为首发门 |
| 先不修 | CaT、Beta/tanh、有界 policy 分布、8192 env、full-body 完整 ready 链 | 都不在第一条可迭代 upper policy 的关键路径；不得与本次 replacement 混改 |
| 需要 canary | Reward 各组权重/负项剂量、reference guard、课程失败率、entropy/sigma/RSI、full-body | 会改变优化目标或样本分布；只有 qvel-fixed baseline 出现持续 strike 后才按单变量小预算比较 |
| 运行验收 | exact `4096 env × 5 update × save1` probe | launcher 增加唯一固定 `probe` budget；这是把同一 setting 放大到真实并行规模，不是学习 A/B |

两条 qvel-fixed 资产和 N=1 bundle 已纳入 Git。Pod1 的反手拉
`n1_qvelfix_smoke_5ecf0e06_loop_gpu1_r3` 与反手挡
`n1_qvelfix_smoke_5ecf0e06_block_gpu1_r4` 均自然完成 `1 env × 2 update`，iteration 分别为
`4.65/3.18 s` 与 `4.67/2.92 s`；四个 checkpoint 均已在 Pod 载入并逐 tensor 验证 finite。
两条都没有 q_des、table 或 fall terminal，但 N=1 小样本已经看到 actual raw-hard：
loop 第二轮 `2` 次，block 每轮 `1` 次，主要为踝关节。该比例不能用来调 Reward，也不能伪装成
健康；下一唯一收口门是 fixed `4096 env × 5 update` probe，确认 episode 能跨过各自
`t_hit`、strike 非零，且 raw-hard/table/fall/nonfinite 没有爆炸。达到后立即发 long，不等待
上表“先不修”项目。

### 6.5 4096 probe 反证与 A3 stable-upper successor（2026-07-30）

上节的 4096 门已经完成，而且**否决**了 qvel-fixed long：

- 反手拉五轮 iteration 为 `30.51/40.39/40.78/35.47/35.82 s`，mean episode
  约 `23` 步；反手挡为 `38.35/50.39/43.55/43.14/54.73 s`，mean episode
  约 `12` 步。两者都短于 `t_hit≈31/24` 的有效击球窗口，strike opportunity 恒零；
- q_des projection/nonfinite、table hit 基本为零，finite checkpoint 连续产出；失败主因是
  actual raw mechanical edge。反手拉每轮约 `2.5k--4.2k` 次，反手挡末轮约 `7.7k` 次，
  集中在左 ankle pitch 下侧、waist pitch 上侧与右 ankle roll 上侧；
- `4096×24=98,304` environment steps/update，当前约 `2--3k steps/s`。因此不能靠
  Reward、更多 env 或更多 GPU 掩盖 plant birth 问题，也没有必要跑更长才裁决。

进一步对 exact AgiBot A3 合同的复核推翻了“grounded candidate 等于闭环稳定 ready”的假设。
两条 upper 的 frame 0 共享深蹲下肢（knee 约 `1.17--1.20 rad`）、root
`z=0.920683 m` 且 pitch `-11.19°`。该姿态虽然几何双脚接触，却不是现役 implicit-PD plant
的稳定 hold：fresh actor qdes 已正确约等于该 motion ready，age<=1 hard 为零，但重力/接触后
在 `0.24--0.48 s` 漂到真实硬边界。tracked receipt 的 static-ground 字段此前实际上是
`feasible=null / missing scipy`，不能写成 PASS；后来得到的 infeasible 结果也不应再被忽略成
与出生稳定性无关的纯 telemetry。

直接 successor 是 `A3 stable-upper` 合同修复，不做学习 A/B：

1. head/arms 的 q/qd、三腰相对 frame-0 的轨迹增量与 qd、frame count、timing 与 strike
   frame 保持；三腰 frame-0 常量重基准到 A3 runtime ready 零位；
2. 12 个腿关节改成 `AGIBOT_A3_CFG.init_state` 的官方 runtime stand，腿 qd 为零；
3. root X/Y 与 source yaw 保持，root 改为 upright、`z=1.0684 m`；
4. exact A3 重建全部 schema-2 FK/velocity；由于 racket world pose 会改变，旧 ball/task
   binding 必须重新物化，禁止跨 bytes 复用；
5. 先做 deterministic closed-loop hold，再串行跑 `1 env×2` 与 `4096×5`。只有 episode
   越过 `t_hit`、strike 非零、raw-hard/table/fall/nonfinite 不爆炸且 checkpoint finite，才发
   finite long。

历史 v1 由 `configs/a3_upper_stable_stand_v1.json` 保留；腰部补全后的
`configs/a3_upper_stable_stand_v2.json` 与
`scripts/materialize_a3_stable_upper_motion.py` 已实现并在 Pod focused regression
`12 passed`。反手拉/挡 stable-upper motion SHA-256 分别为
`4343a85e227de02f634d99d27499df2a4fa63b93df069ea2edb44524dca075ff` 与
`08aeafaff2a14b62c4d9d37c77855c2ca5a9f9cb2ffde7f97b748676b681df01`；exact A3 均为
双脚 `3+3` 接触且 static-ground LP `feasible=true`。击球帧 site speed 分别保持
`1.8243512604` 与 `1.6183056627 m/s`，但世界拍位最多变化 `0.138/0.064 m`，故 N1
materializer 已改为保留来球 profile 宽度、把完整 contact box 平移到新 selected face
center。两动作新 N1 bundle 均 materialize PASS，SHA-256 为 `054be7f2…` / `6973f1a3…`。
CaT、真实 hard-edge 放宽、Beta/tanh、Reward 剂量、8192 env、full-body 与 N5/N73 均不混入
该 successor。

v1 的两动作 `1 env×2` smoke 随后给出同一反例：每轮都只在 age `16--17` 触发
`waist_pitch_joint` 上侧 raw mechanical edge，腿/踝、q_des、table、fall 与 nonfinite 均为零，
4 个 checkpoint finite。原因是 v1 虽替换 lower/root，却保留了旧深蹲动作 frame-0
`waist_pitch=+0.103 rad`；A3 runtime stand 的三腰 ready 实为零。v2 因而按上述合同把三腰
整条 q 轨迹做常量平移、保持动作增量与 qd，再重算 FK/contact。这是 v1 漏项的直接修复，不需要
更长线或学习 A/B。Pod focused test 为 `2 passed`；v2 loop/block motion SHA-256 为
`0fa46ad66d57edd006b0a70a7de0542d8d53945ee3ae9802fdbd937555a0c85b` /
`cc9bbccd1b5b6207a0ce9677944ba27fa4a062a1eaa61886d802c9d21830caa0`，均为双脚
`3+3` 接触且 static LP `feasible=true`。击球帧 site speed 为 `1.8181/1.6422 m/s`，
因此仍须重绑 task，不能复用 v1 bundle。

v2 N1 producer 的 Pod focused regression 为 `11 passed, 2 deselected`；loop/block bundle
SHA-256 为 `85c7a27607b6afa74276653a76cc3fdfd843b5a18b91bebd9e041fe5f3f93627` /
`09d0dea31f5e220cdd917f5f0863da5c9d5b5974d1863d6bbd0da1d91c88afdd`，均
materialize PASS。
真实 Isaac scene 物化的 policy contract SHA-256 为
`03f833e11e75a4c2583fadd9fabee20e3fa418e07cfaf573d6e4c4956fb14227` /
`3442881f50a6e1b38ae094b5d6078ebe025372bd88ee63f56ed79c801f7f4627`；
lower 与 waist 的 normalized bias 均为零。

历史审计同时收窄了五轮 probe 的解释：保留的 `s1w4_M2_v4rg` 是从 `model_13000` 连 optimizer
warm-resume 的旧两动作 run，恢复后约第 2--12 update 才跨击球窗；没有 fresh 0--300 update
历史日志。因此五轮足以发现当前 raw-hard/sample-starvation 并拒绝直接 long，却不能证明 fresh
policy 永远无法自救。stable-ready 的 4096×5 健康后，下一判别应是 fresh 100--300 update
recovery；strike 出现且 raw-hard 下降才续 20k。

### 6.6 stable-upper v2 真实 probe 与 recovery（2026-07-30）

stable-upper v2 的真实 Pod 结果否决了“静态 LP 可保持即可直接 long”，但没有把五轮结果外推成
“fresh policy 永远学不会”：

- loop `1 env × 2 update` 自然完成，iteration 为 `4.72/3.07 s`；block 为
  `4.63/3.04 s`。两者 update 1 的 mean episode length 均为 `29`，q_des/table/fall/nonfinite
  为零，但各出现一次 actual raw-hard，主要仍在腰部；
- loop exact `4096 env × 5 update` 自然完成，iteration 为
  `26.73/42.63/33.41/34.01/33.96 s`，mean episode length 为
  `24.00/24.20/22.17/21.01/21.43`，每轮 actual raw-hard 为
  `3741/3979/4167/4654/4635`，strike opportunity 恒零。以
  `98,304` environment steps/update 计算，吞吐约 `2.3--3.7k steps/s`；
- block 的 exact 4096 probe 在 URDF/scene 构造阶段持续约 900 秒、约 40 CPU cores 满载而
  GPU 约 1%，尚未出现 `Learning iteration` 即由 launcher 自己的
  `KIT_BOOT_STALE_TIMEOUT_S=900` fail-closed 停止。它没有 PPO 结果，不能与 loop probe
  混写；但证明当前 4096 构造路径存在非线性 CPU/receipt 初始化成本；
- 为区分“五轮样本饥饿”与“fresh policy 可在较长线自救”，同一 block setting 按预注册
  启动了 `1024 env × 100 update` recovery
  `n1_stable_v2_canary_e8f1b8e5_block_gpu1_r1`。它运行到 update 77；iteration 通常约
  `10--18 s`，mean episode 始终约 `21--22`，strike 始终为零，actual raw-hard 始终约
  `47--49` episode events/rollout，q_des/table/nonfinite 为零、fall 极少。
  `model_20/40/60.pt` 均已写出，`model_20.pt` 在 Pod 逐 tensor 验证 finite。该证据已经否定
  “当前出生合同可在 100 updates 内自然恢复到击球窗”，但不外推成任何稳定 ready 都不可学；
- update 77 后新题首次命中 producer 已允许的 float32 rate 边界，Motion consumer 却用严格
  `min <= rate <= max` 再验一次，触发
  `ValueError: action-ball teacher_rate is outside its certified range`。这是 producer/consumer
  数值接缝不一致，不是物理失败；修复必须复用同一 canonical `5e-7` 边界容差且继续禁止 clipping，
  不做学习 A/B。`194e9786` 已完成该修复；Pod1 focused test 同时证明合法边界通过、旧的越界
  篡改仍 fail closed（`2 passed`）。

另在 Pod 用 exact A3 MuJoCo 动态 replay 复核 stable-v2 teacher。block/loop 分别在约
`1.24/1.26 s` 后因 tilt 超门失败，COM support margin 约 `-0.375/-0.391 m`；两脚仍接地且
slip 很小。这说明 static contact LP 只证明“存在一组瞬时平衡力”，不证明 `qdes=q` 或整段 teacher
在 official low-gain plant 下动态稳定。当前 receipt 又只保存 residual/effort ratio，没有保存
LP 的 31 维 actuator solution；因此不能直接从旧 receipt 派生 hold qdes。

静态 motion 审计排除了“老师动作自己贴腰限位”的解释。stable v2 teacher 的 runtime
`waist_roll` 与 `waist_pitch` 分别位于约 `[-0.046,0.014]`、`[0,0.141] rad`（loop）以及
`[0,0.077]`、`[0,0.098] rad`（block），与 A3 hard limits 有显著余量；q_des projection
也持续为零。结合 official A3 waist PD（yaw `85/3`、roll/pitch `50/2`），当前首因更像
official low-gain plant 在重力、接触和全身耦合下不能从该出生/qdes 合同闭环保持到击球窗，
而不是 Reward、solver、teacher limit 或有限 q_des。

因此下一直接修仍是统一的 **dynamic-ready contract**，不做 Reward A/B：physical spawn、
initial qdes bias、actor observation、teacher reference 与 motion frame 0 必须描述同一姿态。
但执行顺序必须是先求并绑定 action-specific nominal hold qdes，证明 nominal plant 能稳定保持，
再增加 preparation window；没有 hold compensation 时单纯延长等待反而会增加击球前死亡。
incoming-ball TTC 下界随后绑定 `preparation + validated t_hit + margin`，再跑 `1 env×2` 与规模
probe。不能以提高 Reward、放宽 actual hard edge、CaT 或增加 env 数掩盖动态不可保持的出生状态。

性能工单同时收窄为测量优先：现有 ball→task 已是 reset-time GPU batch，且 episode 内缓存，
在 update 分段 profiler 证明 solver 占主导前不重写算法。优先量化 physics rollout、
metrics/D2H、termination、safety archive、reset birth/retire、sampling/solve、state write 与
PPO；随后先移除 reset 风暴下的逐环境 Python/完整 archive 税。PPO 约 `0.1 s`，不在关键路径。

### 6.7 reset 语义复核与 dynamic-ready 实现状态（2026-07-30）

Jiayi 补充的设计意图是“先从站姿学会击球，达到约 35% 后再把歪掉/失败状态混入 reset”。
这不是当前 stable-v2 失败的解释：现役 ActionBall 启动时强制
`canonical_ready_mode=true`、`stand_start_prob=1.0`、`post_swing_start_prob=0.0`，
并在 canonical 分支返回后才会到 legacy stand/post-swing/RSI 三选一。每次 true reset
实际写入所选动作 motion frame 0（零速度），不是 post-swing buffer。block stable-v2 的
12 个下肢关节与 A3 default stand 逐位相等、root z=`1.0684 m`；19 个上身关节仍是动作专属
ready。当前代码也没有“strike 35% 后启用失败 reset”的门；这是未来可实现的 curriculum，
不是已经提前触发的机制。若以后接入，它会改变出生分布，必须等健康 strike baseline 后做
小 canary，不能混入本次修复。

当前直接修正在两层收口：

1. ground-contact LP 保留历史 feasibility 默认字节/目标不变，新增显式 opt-in 的
   `hold_minimax_normalized_available_torque`，按每关节正负可执行力矩分别最小化最大利用率；
2. 新的 `materialize_a3_dynamic_ready_contract.py` 从 exact stable-v2 frame 0、运行时
   training contract 和 exact A3 MJCF 生成动作专属 hold-qdes 候选。它明确相交
   projected-soft 与 hard-inner qdes 包络，并修正了一个独立审查发现的关键顺序问题：
   MuJoCo LP 的 post-root actuator row 与 A3 runtime joint order 不同，边界必须
   runtime→MuJoCo scatter，求解力矩再 MuJoCo→runtime gather，之后才能做
   `qdes=q+tau/Kp`。

这两层只是候选生成，不授权训练。下一证据仍是 Pod 的 exact LP regression、两动作候选物化，
以及 Isaac `env.reset()` 后第 `0/1/10/final` 帧截图与无 PPO closed-loop hold。只有姿态在
`t_hit+margin` 前不触发 actual-hard/table/fall，才把 hold qdes 接入 actor bias/last-action/
observation/reference，并重跑 `1 env×2`。旧 `4ff48b21` 两条 long 到 update 169 仍为
strike=`0`，且其旧 reference reset 语义不代表 dynamic-ready successor，继续保留但不作为
新配方证据。
