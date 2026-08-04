# EXP-A3-P1-0803-31ACTION-NORMALIZATION-20260803

Status: partial；future-primary candidate 已 materialize，现役 runtime pointer 未切换

Evidence: E2 exact Pod IsaacLab import + 20-step finite diagnostic；successor safe-ready、全动作拍心
FK/重定向与 MuJoCo parity 仍未测

## Question

能否不覆盖现役 `agibot_a3/`，把 2026-08-03 A3-P1 交付确定性地投影成与现有
31-action policy ABI 兼容的独立 successor，同时不伪造左夹爪资产，并确保右拍心合同没有被
归一化过程移动？

## 2026-08-04 decision

- **Adopt:** 用户裁决 raw URDF 是该交付的几何 ground truth；workbook 中冲突的
  `left_OP3_joint` mount 不再拥有否决权。左夹爪不进入 policy action，因此项目明确拥有一个
  versioned training projection：九个 gripper movable coordinates 全锁在 raw-URDF `q=0`。
  这里的 `q=0` 是**项目锁定位**，不是 vendor neutral、硬件 home 或耦合模型声明。
- **Adopt:** 保留 21 个夹爪 link 的全部 delivered inertial、质量与 joint origin；夹爪子树质量
  `0.76626209416 kg`，normalized 63 unique-link 总质量 `57.60001015416 kg`。九个零位均在 raw
  limit 内。
- **Adopt:** 交付缺 20 个 gripper collision mesh。不得用 visual mesh 或自造 primitive 冒充；
  归一化只删除这 20 个确实无法解析的左夹爪 collision element，并把它登记成 project-owned
  collision-disabled contract。它允许右手训练 candidate bring-up，但不证明左手碰撞或真机 parity。
- **Adopt:** 删除 fixed joint 上无运动学意义、会阻塞 Isaac importer 的五个非法 axis；对四个
  含 `-` 的 mesh basename 产生原字节 `_` alias；删除唯一无 geometry 且颜色为 NaN 的 visual。
- **Defer:** 现役 pointer 切换、successor safe-ready、桌/自碰、同动作全相位拍心 FK/重定向、
  dynamics/fixed-tape parity 与 MuJoCo identity v3。它们通过前，`training_authorized=false`。
- **Reject:** 剪掉整只夹爪及其质量；伪造缺失 collision；把 project lock 写成 vendor neutral；
  因为右拍 local 合同相同就声称 successor world-FK 与现役相同；在 receipt 闭合前覆盖
  `agibot_a3/`。

## Raw and generated authority

- Raw source: `112 files / 57,803,270 bytes`，closure SHA-256
  `b1da6430fb20901ffd4fedbf60ee1cda452b12d25bd02f3816f359c24a47818f`；主 URDF
  SHA-256 `7dc98e48602036a93d1e7492f7632d30d71477743e76559730637d9f95151704`。
- Tracked producer: [`prepare_a3_p1_0803_31d_asset.py`](../../../scripts/prepare_a3_p1_0803_31d_asset.py)，
  SHA-256 `058f7d50…a0eb0`。
- Tracked exact diff/closure receipt:
  [`a3_p1_0803_31d_v1.json`](../../../configs/a3_p1_0803_31d_v1.json)，schema 2。
- Ignored generated output:
  `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3_p1_0803_31d_v1/`。
- Output: 63 links / 62 joints / 31 movable joints / 104 mesh references / 100 unique meshes；
  URDF SHA-256 `2f15df8a…2535`；101-file closure SHA-256 `73a47e85…8f08`。

`scripts/prepare_a3_p1_0803_31d_asset.py` 默认 no-clobber：版本化 output 已存在就拒绝；输出路径若是
现役 `agibot_a3/` 或其子目录也在写入前拒绝。`--check`（只核验，不写文件）同时重算 raw closure、
normalization diff、31-joint order、右拍合同、output closure 与 manifest。

## Racket-centre result

拍心的 local contract 没有漂移：normalized/现役的 `pingpang_red_joint` 是

```text
parent = right_hand_pingpang_Link
xyz    = [0.21021, 0.032078, 0.032036] m
rpy    = [0, 0, 0]
```

raw 中同一 parent/child 只采用交付的小写 `_link` 拼法；归一化的名称映射不改变 transform。
这里的“拍心”沿用现役 `official_racket_site` 工程控制点，不冒充红面面积几何质心；二者的
面内距离约 `1.264 mm`，详细定义仍只认
[`racket_contact_geometry.md`](../../interfaces/racket_contact_geometry.md)。

rigid hand、red、black 与 centre-marker 四个 mesh 的 SHA 也逐一相同。Normalized 对 raw 的共同
`q=0` 拍心 world transform 是 position/orientation exact `0/0`；更强的依据是归一化保留了整个
右臂 chain 的 joint origin/axis/limit 与 link inertial，所以对所有共同 31-joint `q`，normalized
就是 raw successor 的拍心 FK。

但 successor **不是**旧 plant 的同一个 world-FK：0803 raw 将 `right_elbow_joint` origin 从现役
约 `[0.010, 0, -0.133]` 改成 `[0.001, 0, -0.1325]`。在共同 `q=0`，successor 与现役拍心位置差
`9.013878 mm`，orientation 相同。因此现有 measured-v4 动作只证明旧 URDF；切 successor 前必须
在新 URDF 上重跑 full-clip measured-site→FK/retarget audit。不能用 local site exact 代签该 gate。

## Pod evidence boundary

项目 lock/collision 裁决没有改变 generated output bytes；closure 仍是此前 Pod1 实测的
`73a47e85…8f08`，所以原 exact importer receipt 仍只证明同一字节：

- IsaacLab converter 正常退出；31 articulation joints 与 runtime order exact；
- merge-fixed 后 32 个 required runtime bodies，missing/extra=`0/0`；
- 20 physics steps 全 finite，初始 q/qdes 在 imported hard limit 内；
- 现役 `agibot_a3/` pointer 未改。

它不证明 safe-ready、table/self collision、拍心 world-FK/Jacobian、训练可学性或真机。尤其缺失
夹爪 collision 的 project contract 是明确的模型简化，不是物理完整性证书。

## Reproducible host checks

```bash
python3 scripts/prepare_a3_p1_0803_31d_asset.py --check
python3 -m pytest -q tests/test_prepare_a3_p1_0803_31d_asset.py
python3 -m py_compile scripts/prepare_a3_p1_0803_31d_asset.py
```

Observed 2026-08-04: `PASS`；`6 passed`；producer compiles。

## Promotion boundary

这项改动不阻塞当前右手训练：现役训练仍消费原来的 `agibot_a3/`，没有被热切。若要把 0803
candidate 升为 future primary/pre-long plant，必须在 exact closure 上补：

1. 当前 threshold-first birth/safe-ready 与后续 durability receipt；
2. 为 successor exact URDF/USD 重生目前硬钉旧 SHA/43-OBB geometry 的 table-collision proxy，
   并明确 left-gripper termination coverage；同时确认 collisionless gripper simplification 风险；
3. 新 URDF 上 Take061 及候选动作的 full-phase official-site position/orientation/velocity/FK audit；
4. same-state old/new dynamics 差异解释与 fixed-tape RNG/reason/counter/safety parity；
5. MuJoCo identity v3 与 cross-engine parity。

全部闭合前，manifest 保持 `canonical_runtime=false / training_authorized=false /
deployment_authorized=false / hardware_authorized=false`。
