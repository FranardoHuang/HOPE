# EXP-BANKED-CGROUP-BHLOOPC-20260727 — 题库化 bh_loop_c 单变量消融(C-group)

Status: running → 部分收口(2026-07-28 晨)
Human owner: Franco;执行者:Codex 发射(07-27 00:09),Claude/Fable 07-28 接力收口
Namespace: pod1+pod2 `/workspace/codexschema/nohope_pin_20260727/`(wave1 对照在 pod2 `wave1_20260727/`)

## 问题与设计

人话:这一批第一次把"每一拍的拍速/拍面是从来球逆解出来的答案(题库)"当基线,老的
uniform 盒子臂留作对照。每臂与基线只差一个开关。单 clip `bh_loop_c_upper_wave1.npz`
(SHA `97f5e847...54c8`),题库 `banks/bh_loop_c_train.npz`(n=8192, --grip off),
4096 env,25k iter,发射器 `launch_banked.sh`(六个强制 ++ 键;桌障=新默认)。

| 臂 | 单变量 | 卡 |
|---|---|---|
| c_base_bank_seed0 | 基线(题库+桌障) | pod1 GPU0 |
| c_notable_seed0 | 关桌子 | pod1 GPU1 |
| c_ep20s_seed0 | 回合 10s→20s | pod1 GPU2 |
| c_speedwide_seed0 | 老师速度带 0.6-1.0→0.5-1.2 | pod2 |
| c_base_bank_seed1 | 换种子 seed=1 | pod2 |
| wave1_bhloopc_r2_seed0 | uniform 盒子对照(20k) | pod2 |
| c_ballwide_seed0 | 来球带 2-5→1.5-7 m/s | **未发**(宽题库未生成) |

## 读数(07-28 晨,最后 3×2000-update 窗合并,分母在实验数据报告)

| 臂 | 进度 | legal/strike | 摔/回合 | 判定 |
|---|---|---|---|---|
| base | ~22k/25k | **0.488**(升) | 0.023 | 学成,旗舰 |
| speedwide | ~21k/25k | 0.463(趋平) | 0.022 | 学成;完成率降 3.5pp 为代价换速度覆盖 |
| notable | 7275 中断 | 0.103(末窗) | 0.016 | **同期领先 base**(6-8k:0.103 vs 0.087);桌障有可见学习税;无终局 |
| ep20 | ~22k | =base 逐位 | — | **无效臂**:`++env.episode_length_s=20.0` 未生效,与 base 逐位相同(每个 2000-桶计数器全等)。07-28 TERM 释放 |
| seed1 | ~20k | **0.000**(峰 0.004@4-8k 后崩) | 0.018 | 崩塌:不打球策略胜出 |
| wave1(uniform) | ~19k/20k | **0.000**(峰 0.005 后崩) | 0.019 | 崩塌;题库 vs uniform 同种子对照成立 |

## 结论(截至 4k-22k 证据)

1. **题库(球反解目标)是当前配方能学会回球的必要条件**:同 seed0 下 bank 0.488 vs uniform 0.000。
2. **但不是充分条件**:bank+seed1 照样崩。崩塌臂 Mean reward(−11~−14)反而高于学成臂
   (−43~−51)——**当前收入结构下"不打球"仍是更优策略**,击球靠脆弱路径维持;种子能翻车。
   反崩塌手段(上台收入抬升 / death 定价 / 站立起步)需进下一代配方,旗舰臂必须双 seed 冗余。
3. 桌障是新默认且有学习税(notable 同期领先),幅度待 notable 续跑或重发后定。
4. ep20 键传递链断了:`++env.episode_length_s` 没进 env,修好前 20s 消融不算测过。
5. v2 reward-scale 侧(独立谱系,4k 截面,详见 [freeze 文档](EXP-V2-REWARD-FREEZE-20260726.md)):
   table_r12(上台 1.2×)legal/strike 0.050 全面领先 defer0 0.019 / r3(延付)0.008,摔率也最低
   ——**上台奖 1.2× 方向 + 不延付**是下一代配方的当前最佳档。

## 07-28 处置

- ep20(白跑)与 notable 僵尸进程(07-27 05:00 interrupt 后 wedge 10h,model_7275.pt 已存)
  按 PID 精确 TERM 释放(两者同 PGID,不按组杀);收据在 pod1 namespace `STOP_RECEIPTS_20260728.md`。
- seed1(已崩塌、证据已入账)同批释放腾卡。
- base/speedwide 跑到 25k 自然收口;wave1 跑完 20k 作对照终档。
