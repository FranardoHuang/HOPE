# S0/M0 exact donor canonical-beta materialization

- 状态：`preregistered`
- 人类负责人：Franco
- 执行者：Codex
- 证据等级：E1（consumer、静态合同与合成 save/reload）；上游 handoff 为 E2

共享缩写见[术语与人话对照](../DEFINITIONS.md)。本记录只回答：已经完整绑定的
[`S0`](../DEFINITIONS.md) 单条高点拍压和 [`M0`](../DEFINITIONS.md) 四条横移老师，能否逐条注入旧 Franco
动作库的同一 exact body-shape donor，同时保证 beta 之外的 GVHMR 内容不变。它不回答动作是否安全、能否
击球、横移是否稳定或能否进入训练。

## 为什么不用旧 cohort materializer 直接重跑

旧 `scripts/materialize_canonical_gvhmr_betas.py` 会对输入 cohort 重新聚合，并且至少要求两条视频。这里
需要的是复用同一位表演者已经验过的 exact 10 维 donor；S0 又只有一条。因此新 consumer 只复用旧工具
已审过的 PT load、beta shape/dtype、replacement、non-beta semantic digest、save/reload 和 no-clobber
primitive，不修改旧工具或历史结果，也不从新五条重新估计体型。

冻结 donor：

- body-shape contract：`diagnostic_same_performer_coordinatewise_median_betas_v1`；
- vector SHA-256：`a03f1642151453316f0c99f81a743a604e29c656c9fffd4bac89353f7c4d9cc6`；
- `canonical_betas.json`：915 bytes，SHA-256
  `f405ba45d7f2233d3735c2e6b59409203cc98b1e97deef00224e16f2c7c4cbf2`；
- completion manifest：26,857 bytes，SHA-256
  `4043b34210b30be4151457623c3152ebd09650fb2caf587eb78359887a06bc2b`。

输出目录里的 `canonical_betas.json` 必须是这份 donor 原件的逐字节 copy，不是 S0/M0 新 cohort 的聚合
结果。每个 PT 只允许修改 `smpl_params_global.betas`；保存并重新加载后，其他所有支持的 leaf digest 和
leaf count 都必须一致。source dtype 量化后不能复现同一个 vector SHA 时 fail closed，禁止静默 cast。

## exact upstream 与两份独立计划

post-GVHMR handoff 已分别完成并冻结：

| 批次 | handoff bytes | handoff SHA-256 | canonical-beta prereg SHA-256 |
| --- | ---: | --- | --- |
| S0 高点拍压单条 | 4,970 | `d57a93e0...a1054` | `236cace8...94f19` |
| M0 四条横移候选 | 9,242 | `60c55150...088ef` | `c70d1fdb...fcb69` |

机器真源是：

- `configs/motion_canonical_betas_s0_prereg_20260713.json`；
- `configs/motion_canonical_betas_m0_prereg_20260713.json`；
- `scripts/materialize_motion_handoff_canonical_betas.py`，SHA-256
  `37db5bf38f2759b356647008c95bda879c46ce3fbc406968d3be93999cac7f79`。

两批 output root 完全分离；任一批失败不阻塞另一批，也不能覆盖后重跑。consumer 同时绑定自己的 SHA、
被复用的旧 materializer SHA，以及 imported post-handoff helper SHA。handoff、donor、completion 与所有
source PT 都要求 regular non-symlink file、exact bytes/SHA。

## M0 脚间距在这一层为什么必须为空

canonical-beta PT 仍在人体 SMPL-X/GVHMR 坐标中，没有 A3 左右足 site，因而不能产生机器人脚距数字。
这层只冻结四条 clip 的 exact `ready_before`/`ready_after` 窗口和未来测量规则：去掉公共 root XY、heading
对齐初始准备朝向，测 `d_xy = right_foot_xy - left_foot_xy` 的二维稳健中位数，并同时保留横向分离与
前后错位。

`foot_site_mapping`、initial/terminal `d_xy`、component tolerance 和 `stance_passed` 当前全部为 `null`；
任何在 GMR 前填入数字或 `passed=true` 的计划都会被拒绝。下一份 exact GMR prereg 必须先绑定 A3 model、
body/joint order、foot sites、窗口到 GMR sample 的映射及两个分量的容差，才能看结果。双脚并拢或更窄
不能冒充恢复成功。

S0 同样保持 `observed_ball_contact=null`、`strike_effectiveness=null`、`safety_result=null`，并明确禁止
借用拉球题纸。它后续需要高球拍压自己的题族。

## 验证与结果

```bash
for PLAN in \
  configs/motion_canonical_betas_s0_prereg_20260713.json \
  configs/motion_canonical_betas_m0_prereg_20260713.json
do
  PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  python3 scripts/materialize_motion_handoff_canonical_betas.py \
    --prereg "$PLAN" --expected-prereg-sha256 "$PLAN_SHA" static
done

python3 -m pytest -q \
  tests/test_materialize_motion_handoff_canonical_betas.py \
  tests/test_materialize_canonical_gvhmr_betas.py
```

2026-07-13 host 结果：S0/M0 两份 tracked static contract 均 PASS；新旧 materializer 专项合计
`15 passed, 1 skipped`，skip 原因是 host 无 Torch。合成 runtime 覆盖了 S0 singleton 与 M0 四条 donor injection、
non-beta bit-exact save/reload、donor byte copy、no-clobber、handoff/window 篡改和 M0 假脚距/假通过拒绝。
重放到最新 main 后，完整 `tests/` 回归为 `616 passed, 9 skipped`。

真实 `inspect/consume` 尚未在绑定的 motion Python 环境执行，所以本实验仍是 `preregistered`，没有新五条
canonical-beta PT。运行入口见[操作文档](../operations/run_motion_handoff_canonical_betas.md)。

## 决定边界

采用这两份计划作为 S0/M0 唯一 canonical-beta 入口。成功 consume 只会把下一步改成“可另建 exact GMR
prereg”；它不会直接授权 GMR、schema-2、L0/L1、TOPP、RL、Gate3 或真机。
