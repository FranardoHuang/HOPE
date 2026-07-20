# 击球与共享横移脚步的组合语义合同

- 状态：`frozen semantics`（2026-07-20 用户拍板）；本合同只锁语义，不授权任何训练、仿真或真机动作。
- 语义唯一真源：[`configs/motion_role_catalog.json`](../../configs/motion_role_catalog.json)，
  由 [`scripts/validate_motion_role_catalog.py`](../../scripts/validate_motion_role_catalog.py) fail-closed 校验。
- 人话：动作库里只有两种角色——"原地击球"和"共享横移脚步模块"。机器人想横移，唯一合法理由是
  "为了打到一个真实来球"；没有独立的走路动作，也没有独立的刹车老师。

## 角色定义

1. **原地击球（`stationary_strike`）**：所有非 `motion/` 的击球动作素材。"原地"只指没有有意的
   base 平移或迈步；它不冻结腿，允许重心转移和腿部姿态变化（catalog 中
   `lower_body_semantics=legs_free_weight_shift_allowed`）。每条只服务自己的动作位
   （`activation_scope=own_action_slot`）。
2. **共享横移脚步模块（`shared_lateral_footwork_module`）**：`motion/{left,right}_dang{1,2}.mp4`
   四条素材。它们是跨所有击球动作复用的下半身/根节点参考（`shared_across_action_slots=true`），
   不存在独立 locomotion 动作，也不存在独立 stop teacher；停下与收步是脚步模块 recover 段的一部分。
   历史 intake 里的 `lateral_locomotion_teacher`/`lateral_step_teacher` 字段是被本合同取代的旧标签，
   intake 文件字节保持冻结。

## 触发条件：横移只能由有效击球意图触发

- 策略/规划器判定当前来球需要的横向位移在 stationary reach/deadband 之内时：使用击球动作自身的
  下半身参考，不组合脚步模块。
- 所需横向位移超出 reach/deadband 时：把共享脚步模块与该击球动作组合，按
  `prepare -> strike -> recover` 三段事件对齐（对齐锚点为人工复核的标称击球帧，在任务接触流形确认
  之前不得当作真实击球帧）。
- 没有击球意图就不得移动：不存在"为了移动而移动"的合法输入。
- reach/deadband 的具体数值属于后续实验合同，必须在使用它的实验里预注册并内容绑定；本合同只冻结
  "带内用自身下肢、带外才组合"这一结构。

## 组合与重过门规则

- 组合交界所有权沿用 [v12 实验设计](../experiments/motion_v12_high_press_lateral_teacher_20260713.md)：
  脚步模块负责 world root 平移/偏航、双足、腿部关节与接触阶段；上肢挥拍以骨盆相对坐标负责球拍/
  接触目标；骨盆高度/倾角与躯干是耦合变量，由受约束的全身求解给出，不得从两个动作源直接复制。
- 每一个派生动作 `<strike, footwork, signed distance, phase alignment, retiming>` 都是新动作，
  必须重新通过完整 Gate 链（runtime-order schema、L0 有限数/限位、厂商 MJCF L1 自碰/间隙、
  桌网扫掠间隙、厂商动力学/足接触重放，以及其后的任何行为门）。任何组合参数变化都会重置资格。
- TOPP/重定时只在几何与接触先过门后使用，重定时后必须重复所有下游门。

## 语义修正不覆盖安全事实

- M0 四条脚步素材的机器人 stance gate 仍是 `0/4` reject（catalog 中
  `input_gate_status=rejected_stance_0_of_4`）；改名不改这个事实，修复并重走 exact 链之前它们不是
  可训练资产。
- Franco 反手拉 B 目前只有静态/`mj_forward` 证据（`mj_step_calls=0`），不得称"不倒"，也不得标
  `training_authorized`。
- v4rg 正反手 runtime NPZ 对是现役 formal 训练绑定；其合同早于厂商动力学门，因此在 catalog 中带
  唯一的 `grandfathered_formal_runtime_pair=true` 豁免。除这一对外，`vendor_mujoco_dynamic_pass=false`
  的条目一律 `training_authorized=false`。

## 校验入口

```bash
python3 scripts/validate_motion_role_catalog.py
```

校验内容：catalog 与两份 intake 清单 asset_id/SHA-256 精确互覆盖、三份来源清单文件重算 SHA 一致、
角色/方向/触发范围/训练授权规则全部满足、dang 文件名与 movement_direction 一致、祖父豁免唯一。
