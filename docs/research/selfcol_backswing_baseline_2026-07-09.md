# 反手引拍余隙基线表(自碰撞检查器附带产出,2026-07-09)

立项来源:`docs/TIMELINE.md` 07-09(晚五)franco hint——**"正手加引拍很方便,反手可能和躯干
碰撞"**;07-09(晚六)franco 纠错把借行程自由度收窄到**手臂链(肩内旋+肘抬高)+ 腰偏航**,
并给这张表定了第二个主项:**"管'肘抬高后小臂-躯干余隙'"**。

工具:`hope_training/whole_body_tracking/scripts/audit_self_collision.py`(L1,厂商 MJCF
逐帧 `mj_forward` + 接触检测)。相位取 `cfg/strike_annotations.yaml`。

## 口径

- **引拍最深帧 t\***:从触球帧 c 沿时间回溯,只要拍体(躯干系)还在远离它的触球位置就继续退,
  第一个局部极大即转向帧 = 引拍最深处。**不是** `argmax over [0,c]`——那会退到准备姿态去。
- **拍**=`right_racket_collision`(拍面)+`right_racket_handle_collision`(拍柄);
  **小臂**=`right_elbow_collision`+`right_wrist_roll_collision`(腕滚=前臂旋前,算小臂);
  **躯干**=`torso_collision`;**骨盆**=`pelvis_collision`。
- 距离=MuJoCo 几何精确距离,精度 0.1 mm。**不采信 `mj_geomDistance` 的返回值**(MuJoCo 3.10
  mesh-mesh 缺陷:`distmax` 超过真距时会偶发返回 0.0),只用它的截断谓词二分——见工具 docstring。
- 扫描窗 = `[0, 触球帧]`。

## 读表须知(一条,重要)

**验收数字看 `@t*` 两列。**`窗内最小` 两列是整个击球前窗口的下确界,它的 argmin **经常落在
f0-f5 的准备姿态**(手臂自然垂在身侧,离躯干本来就近),那不是引拍造成的余隙。
例:`hope_backhand_v4rg_cal` 小臂-躯干 窗内最小 182.4 mm @f0,而引拍最深帧 @t*=30 处是 184.5 mm。

## 结论(六对现役 + 全体 14 对,全 PASS,零穿透接触)

- **28 条 `*_cal` 全部 PASS,自撞接触对数 = 0**(退出码 0)。franco hint 描述的碰撞在**现有
  资产上尚未发生**——它是**加深引拍之后的风险**,不是当前病灶。
- **反手引拍最深帧余隙(12 条有相位登记的反手)**:
  - 拍-躯干 **189.7 ~ 236.5 mm**
  - 小臂-躯干 **127.3 ~ 212.5 mm**
- **哪个面先撞,取决于资产族——两个面都必须约束**(数据不支持"小臂一定更紧"):
  | 族 | 拍-躯干 @t\* | 小臂-躯干 @t\* | 先撞的面 |
  |---|---|---|---|
  | v5 全族(v5rg/v5hA/v5hL/v5hLs/v5hLt/v5syn/v5syn35/v5topp) | **189.7–193.8** | 211.0–212.5 | **拍** |
  | v4rg / v4rgsyn | 236.5 | **184.5** | **小臂** |
  | swing | 196.3 | 198.7 | 基本持平 |
  | swingsyn | 210.1 | **127.3**(全体最薄) | **小臂** |
  → `extend_stroke v2` 的自碰撞约束**必须同时含拍-躯干与小臂-躯干两个面**,不能只挑一个。
  晚六关切的"肘抬高后小臂余隙"在 v4rg/swingsyn 上确是主约束面(184.5 / 127.3 mm);
  但在整个 v5 族上,**先撞的是拍**(189.7 mm)。本检查器两面都判。
- 换算:反手引拍最深处,离撞上躯干还剩 **13–21 cm**(取两面较小者,逐族)。`extend_stroke`
  的 ΔL 预算一旦把该面的法向位移压过这个数,就会从"判炸器全绿"直接掉进自撞——
  **这正是本检查器补的洞**。

## 表(反手;`-` = 该 clip 未登记触球相位)

| clip | verdict | T | 触球帧 c | 引拍最深帧 t* | 拍-躯干 @t* [mm] | 小臂-躯干 @t* [mm] | 拍-躯干 窗内最小 [mm] | @f | 小臂-躯干 窗内最小 [mm] | @f | 拍-骨盆 窗内最小 [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hope_backhand_swing_cal | PASS | 108 | 55 | 40 | 196.3 | 198.7 | 186.7 | 43 | 113.5 | 5 | 234.9 |
| hope_backhand_v4rg_cal | PASS | 134 | 45 | 30 | 236.5 | 184.5 | 232.7 | 27 | 182.4 | 0 | 291.2 |
| hope_backhand_v5rg_cal | PASS | 59 | 23 | 14 | 193.8 | 211.7 | 193.8 | 14 | 210.4 | 13 | 332.6 |
| hope_backhand_swingsyn_cal | PASS | 212 | 104 | 69 | 210.1 | 127.3 | 187.0 | 94 | 113.5 | 32 | 234.9 |
| hope_backhand_v4rgsyn_cal | PASS | 222 | 46 | 31 | 236.5 | 184.5 | 232.7 | 28 | 182.4 | 0 | 291.2 |
| hope_backhand_v5hA_cal | PASS | 59 | 23 | 14 | 193.0 | 212.5 | 192.7 | 13 | 210.8 | 13 | 332.2 |
| hope_backhand_v5hAs_cal | PASS | 59 | - | - | - | - | 189.7 | 13 | 210.1 | 13 | 334.5 |
| hope_backhand_v5hB_cal | PASS | 59 | - | - | - | - | 193.8 | 13 | 210.6 | 12 | 332.4 |
| hope_backhand_v5hL_cal | PASS | 59 | 23 | 14 | 193.0 | 212.5 | 192.7 | 13 | 210.8 | 13 | 332.2 |
| hope_backhand_v5hLs_cal | PASS | 59 | 23 | 14 | 189.7 | 211.4 | 189.7 | 13 | 210.1 | 13 | 334.5 |
| hope_backhand_v5hLt_cal | PASS | 71 | 26 | 16 | 189.7 | 211.0 | 189.7 | 16 | 210.2 | 15 | 334.5 |
| hope_backhand_v5syn35_cal | PASS | 97 | 27 | 18 | 189.7 | 211.4 | 189.7 | 17 | 210.1 | 17 | 334.5 |
| hope_backhand_v5syn_cal | PASS | 114 | 44 | 34 | 189.7 | 211.1 | 189.7 | 34 | 210.1 | 33 | 334.5 |
| hope_backhand_v5topp_cal | PASS | 201 | 63 | 48 | 189.7 | 211.3 | 189.7 | 46 | 210.2 | 45 | 334.6 |

**关于"现役六对"**:工单未列举是哪六对,登记表里有 15 对、磁盘上 14 对。我按"在训臂底片"
解读(`s1_wave4/expected_arms.txt` + `docs/NOW.md` 在训 12 臂表),得六族:
`v4rg`(M2/R1b-R8b/R9a)、`v5rg`(M1/M5)、`swing`(M3/M3b)、`v5hLs`(R9c)、`v5hLt`(R9d)、
`v5syn`(R9e/R9g)。**边界情形**:`v5syn35`(R9f)与 `v5hA`(R9b,已判死)也各有过臂,
按更宽的口径会变成七/八族。
**该歧义不影响结论**:本表把磁盘上 14 对 28 条 `*_cal` **全跑**了(六对现役全在其中),
28/28 PASS、0 接触对。无论"六对"怎么划,结论不变。

## 表(正手,对照)

| clip | verdict | T | 触球帧 c | t* | 拍-躯干 @t* [mm] | 小臂-躯干 @t* [mm] |
|---|---|---|---|---|---|---|
| hope_forehand_swing_cal | PASS | 98 | 36 | 0 ⚠ | 224.7 | 128.4 |
| hope_forehand_v4rg_cal | PASS | 141 | 66 | 57 | 423.2 | 181.2 |
| hope_forehand_v5rg_cal | PASS | 58 | 39 | 25 | 341.6 | 169.4 |
| hope_forehand_swingsyn_cal | PASS | 192 | 70 | 3 ⚠ | 224.7 | 128.4 |
| hope_forehand_v4rgsyn_cal | PASS | 215 | 67 | 58 | 423.2 | 181.2 |
| hope_forehand_v5hA_cal | PASS | 58 | 39 | 25 | 347.2 | 173.5 |
| hope_forehand_v5hL_cal | PASS | 58 | 39 | 25 | 347.2 | 173.5 |
| hope_forehand_v5hLs_cal | PASS | 58 | 39 | 25 | 347.3 | 173.5 |
| hope_forehand_v5hLt_cal | PASS | 58 | 39 | 25 | 347.3 | 173.5 |
| hope_forehand_v5syn35_cal | PASS | 77 | 41 | 27 | 347.3 | 173.5 |
| hope_forehand_v5syn_cal | PASS | 77 | 41 | 27 | 347.3 | 173.5 |
| hope_forehand_v5topp_cal | PASS | 93 | 53 | 37 | 347.7 | 173.5 |

**正反手不对称,定量(逐族同底片对比,拍-躯干 @t\*)**:

| 族 | 正手 [mm] | 反手 [mm] | 反手/正手 |
|---|---|---|---|
| v4rg | 423.2 | 236.5 | 56% |
| v4rgsyn | 423.2 | 236.5 | 56% |
| v5rg | 341.6 | 193.8 | 57% |
| v5syn | 347.3 | 189.7 | 55% |
| v5topp | 347.7 | 189.7 | 55% |

反手引拍最深处,拍离躯干的余隙**只有正手的 55–57%**(v4/v5 族;swing 族 t\* 退化,见下,
不计入)。这与 07-09(晚四)行程账本"反手行程只有正手 60–70%"同向且互相独立:
反手行程短,不只是路程本身短,**躯干还把可用空间挡掉了一半**。franco hint 得到定量证实。

⚠ 两条异常如实报(同一条底片):`hope_forehand_swing_cal` t\*=0、`hope_forehand_swingsyn_cal`
t\*=3——回溯规则一路退到片头。取证:这两条是**同一段空挥**(swingsyn = swing 的合成重定时版,
二者窗内最小值逐位相同 128.1 mm),拍体从片头起就单调远离触球点直到触球,**没有引拍转向点**
(空挥,准备姿态即最深)。**不是工具 bug,是 clip 本身没有引拍**;其 `@t*` 应按"片头姿态"读,
不可当引拍最深帧用,故上文正反手比值表把 swing 族排除。反手侧同底片的
`hope_backhand_swing_cal` t\*=40 / `swingsyn` t\*=69 都正常,有清晰转向。

## 复现

```bash
/workspace/hope_mjeval_venv/bin/python \
  hope_training/whole_body_tracking/scripts/audit_self_collision.py \
  /workspace/franco/motion_work/motions/regen_0708_candidates/hope_*hand_*_cal.npz \
  /workspace/franco/motion_work/motions/v5_height_fix/hope_*hand_*_cal.npz \
  --body-order /workspace/franco/body_order_isaac.txt \
  --md selfcol_report.md --json selfcol.json --baseline-md backswing_baseline.md
# -> exit 0, 28/28 PASS, 0 colliding pairs
```
