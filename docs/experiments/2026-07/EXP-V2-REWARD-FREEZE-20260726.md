# v2.2 冻结权重表(v4rg 谱系)— prereg 草稿,2026-07-26

**状态**:待 Franco 签发。签发后 = v4rg 谱系上任何 v2.2 science 臂的权重合同;发射前仍需一格 2-iter smoke。

## 1. 冻结表(scripts/v2_weight_calibration.py 输出,逐字)

| 键 | 冻结值 |
|---|---|
| racket_position / velocity / normal | **393.4 / 295.1 / 229.5**(内部比例 60:45:35 保持) |
| virtual_landing(legal_base,base_frac 0.6,σ 1.0) | **1648.8** |
| virtual_pass_net / strike_success / capture_bonus / spin | 0(v2.1/v2.2 删除项,冻结防回流) |
| 模仿六项 / upright_exp / hit_unstable_support / qbar / 各安全平滑项 | v2.2 默认包不变 |
| action_rate / action_acc 值 clamp | 9.0 / 36.0(**机制未实现——fresh 臂发射前置,见 §4**) |

**阶梯核对**(每步等效收入,weight-units):模仿 **2.462** : 质量 **7.385** : 上台 **18.461** = 1 : 3 : 7.5(Franco 07-26 终裁比例,锚在实测模仿收入);单拍上台奖 ~1425;罚项预算 0.27/步(f=0.15×早期地板 1.8)。

## 2. 依据(probe 收据)

- run:`v2probe_a_resume_seed3_20260725_r3` @ pod1,checkout `fb4c5baa`(分支 Franco_codex/v2-reward-20260725),resume model_6700(hs W 谱系),迭代 6700→6899,4096 env;三次尝试事由见 namespace 的 ATTEMPTS.md(r1 admission 空信任集/r2 并行 boot 死锁,皆基础设施性,r3 干净跑完)。
- 实测(last-50 均值):T_c=46.3 步;ρ_I=0.547;exact-strike 误差 位置 9mm / 速度 0.15m/s / **拍面 1.93°**(σ 三通道全部收至 floor 0.075/0.886/0.262——sigma-normal 活体验证);p_capture≈1.0;p_legal≈0.6(net 0.794×inbounds 0.615);落点距台心 0.643m → legal_base 门内值 0.864;`strike_window_hit_rate`=0.216=13/46.3 诚实窗(**C1 修复活体验证**;旧 bug 会读 0.5+)。
- **probe 关键发现(定权口径修正)**:质量核为"触点尖峰"非窗内平铺(swing-through 基准与拍速目标在窗前后段天然远)——实测每拍有效满值步数 k_eff = pos 0.73 / vel 0.057 / normal 0.165;计算器据此从 duty×ρ_Q 口径改为 k_eff 口径。
- measured JSON(逐字节):`{"motion_lineage":"v4rg_runtime_order_v3","I_weight_sum":4.5,"rho_I":0.547,"k_eff_pos":0.73,"k_eff_vel":0.057,"k_eff_normal":0.165,"T_c_steps":46.3,"window_steps":13,"p_legal_target":0.6,"E_land_value_per_legal":0.864,"action_rate_sq_p95":282.0,"action_acc_sq_p95":40.0}`

## 3. 谱系约束

本表**只对 v4rg_runtime_order_v3 谱系有效**。换 canonical(Franco)动作库:①走 canonical 正门 admission 并收回 v4rg 两条 legacy 信任集条目;②按新谱系重跑 200-iter probe → 重出冻结表(k_eff/T_c/p/E 全是动作依赖量;公式与 1:3:7.5 比例共享)。

## 4. 发射前置与已知残项

1. 任意臂:一格 2-iter smoke(渲染层默认即 v2.2+冻结值)。
2. **fresh-from-random 臂**:action_rate/acc 值 clamp 机制必须先实现(冻结表已给档位 9.0/36.0;无 clamp 的 fresh 臂存在早期自杀区间)。resume 谱系臂不受此限。
3. probe 用旧混合(capture 850/success 30/landing climb 30)运行——所有测量为权重无关仪器量,不受影响;但该 run 的 reward 总量曲线不代表冻结表行为,勿用于对照。
4. ‖Δa‖² 实测均值 ~94 偏高(200-iter 换奖励混合后的 KL 扰动期),p95 用 3×均值保守替代;首个 science 臂读实测分布后可回调 clamp。
