# 动作体系:从视频到标准老师动作到泛化接口(v1,2026-07-06)

一句话:**拍一段打球视频 → 产出一条"机器人可执行、拍面已标定、打直线归一"的标准老师动作 →
训练时按每道题的需求(打向/快慢/拍面)改造老师**。本文是全链路的人话说明书;活的进度看
[NOW.md](NOW.md),历史看 [TIMELINE.md](TIMELINE.md)。

## 第一段:视频 → 标准老师动作(已跑通,六条 clip 全部产出)

```
手机视频 ──GVHMR(视频估人体)──GMR(重定向到 A3;帧0 warm-up 到收敛,2026-07-08 修复)──落地/去漂──CSV
      ──csv_to_npz_mujoco --grip-rot α β(标定烤入,MuJoCo FK 全身一致)──> hope_*_cal.npz
      ──audit_motion_npz.py(L0 判炸器审计:URDF 限位/首帧健康,超了才掐)──> PASS 才入库
```

**素材采集守则(2026-07-09 改版;franco:动作素材=网上职业选手视频做可行性验证,
暂时做不到完美——所以管线鲁棒化是主路,守则分两档)**:
- **网络素材选材清单**(不可控,只能挑):腿可见优先(腿被挡时 GVHMR 的腿是先验想象的,
  v5 实测 vitpose 腿部置信度 0.07-0.33,连带滑脚/深蹲两处病)>侧面机位>慢动作/高帧率来源
  (30fps 拍 12 rad/s 鞭打每帧转 23°,节奏形状保真受限)>有静止段。挑不到的缺陷由管线兜:
  warm-up(头)、标准人体型(尺度)、腿任务驱动/移植(想象腿)、时间律合成(时序)、判炸器(守门)。
- **自有拍摄守则**(可控时全守):腿不被挡、开拍前静止 1-2s、收尾静止 1s、优先 60fps。
- **标准人体型(canonical betas,2026-07-09 定)**:网络素材逐视频估计体型在遮挡下是噪声源
  (v5 被估矮 9% 的病根)——管线统一用一副标定过的标准 SMPL 体型重定向,只取视频的姿态;
  v5hA 的 betas 替换即此原则的首例,应制度化为 GMR 前置步。

关键环节与已踩过的坑(全部有实锤,详见 TIMELINE 07-05/06):

1. **管线看不见拍子**:视频只重建人体,所有"拍面"原本是腕系 +Y 的虚构量。
   解法=**握拍角标定**:以"标准对拉必然合法回球"为先验,在触球真值帧联合反解
   "共享握拍角 + 每条 clip 的斜线方向"。当前值 Rz(+5°)Rx(+40°)(终值联合复核在跑);
   烤入 = 逐帧重解腕三关节使**机器人拍面(⊥腕+Y,硬件事实)对齐人的真实拍面**。
2. **登记表(cfg/strike_annotations.yaml)= 唯一可信源**,每条 clip 五件套:
   `phase`(视频真值触球帧,人工逐帧核)、`train_phase_candidates`(训练最优相位,
   可回球性扫描给出,**与视频真值可以合法不同**)、`rally_yaw_deg`(对拉轴向;
   **登记约定=打直线为正**,用时按 -rally_yaw 转正再按需旋转)、会话级 `grip`、
   `face_normal_reliable`(标定后 true-calibrated)。
3. **质检关卡(每条新 clip 必过)**:⓪**L0 判炸器审计**(audit_motion_npz.py,分支
   `motion-feasibility-audit`,2026-07-08 上线):按厂商 URDF 逐关节审位置/速度双口径/加速度/
   首帧健康/重采样一致性,FAIL 不入库;修复建议语义=**超了限位才掐、掐只对头尾、中段超速给
   慢放系数、都不适用整段拒收**(存量六 clip 首帧幽灵走登记表豁免旗,待接线);①**+X 重落地**
   (reground_hope_frame.py,必须在标定**之前**——2026-07-06 发现现六条 _cal 漏了这步,倾斜被
   rally_yaw 吸收侥幸自洽,新 clip 不许再欠这个债);②烤入面残差 <0.1°;③**腕限位零裁剪**
   (超限=硬件够不到标定拍面,是发现不是瑕疵,要上报);④真值帧可回球性(轻/重两种球型);
   ⑤(SMASH 采纳)tracker 可执行性过滤——变体/生成动作先让跟踪策略仿真跑一遍,跟不稳的丢;
   ⑥登记表五件套齐全(缺烤入标记会被出题器 fail-closed 拦下——2026-07-06 v4 实锤)。**训练用
   非默认动作对必须显式传 `strike_phase_per_clip`**(默认值是 hopex 的,题库只管目标不管时机)。
4. **坐标/旋转类是本项目最大坑源**(已连抓五个真 bug:z 偏移、骨盆系混用、腕点≠拍面点、
   旋向符号、mount-offset≠腕链)。铁律:任何涉及旋转/坐标系的交付必须带独立数值对抗检查。
5. **GMR IK 冷启动毛刺(2026-07-08 归因+修复;TIMELINE 07-08 全证)**:mink 差分 IK 从
   全零直立位起解,人首帧已蹲好 → 每条 clip 前 3-4 帧是"求解器爬进解"的收敛轨迹(首帧峰
   7.4-15.9 rad/s 全在腿),v5 反手因蹲位最深冲破 URDF 速度硬限位=反手 93% 摔的主因。
   修复(pod GMR 分支 `hope-frame0-warmup`):①帧 0 warm-up 到真收敛(整轮 pass max|Δq|<1e-4)
   再录制;②修"人帧 0 被丢"bug;③--velocity-limit 旗标(逐关节 URDF 表)——**裁决默认关**:
   mink 限位是逐 solve 步信赖域,开着削合法挥拍(实测最狠 -26%),限位检查一律放事后判炸器。
   **工序坑两个**:重跑管线必须 `PYTHONPATH=/workspace/franco/motion_work/GMR`(库是 editable
   install 指向 yikang 树);滤波不是修法(瞬态是台阶不是抖动:Butterworth/SavGol 无效,重平滑
   削挥拍峰 23-25%)。**修复只救新生成资产**——现役 _cal 全量重生成(L6)攒着一次做:顺带修
   现役 backhand_v5_cal 被瞬态污染的落地(悬空 ~5.8cm)+ 还 reground 债;重生成后触球相位全部
   重标(如反手 v5 0.362→~0.391)、登记表/题库连锁重出,时机门=修C 收卷后 franco 拍。

### canonical GMR 中间安全屏（2026-07-11）

`screen_motion_gmr_phase_safety.py` 是 schema-2 之前的一道 CPU 快门，不是终审。它只接受
内容寻址的显式清单，不扫目录；每条必须绑定 canonical-beta grounded GMR PKL、
grounding report、MJCF 和所有工具/venue 配置 SHA。它在 vendor MJCF 里用官方
`right_racket` site 取拍心和拍面，用 `mj_differentiatePos + mj_objectVelocity` 取拍心速度；
对源轨迹作 8 子步/区间插值，地面穿透、任何 robot self-contact 或拍面/拍柄对头颈/
躯干/对侧臂/下肢的余隙低于 5 mm 都会硬排除相邻源帧。20 mm 内另报 warning。

本次十条 canonical GMR 的 654 个源帧/5162 个 240 Hz 样本全部没有上述危险，
最薄拍-身余隙是 `40.2466 mm`。但这只是有限采样，不是连续时间证书；MJCF 也没有
table/net geom。更关键的是，root-z grounding 并未证明 GMR world→HOPE +X/虚拟球桌变换，
mirror status 仍未验证。所以工具的当前 v4 合同虽预先冻结 64 题，却强制
`consumed_for_returnability=false`，所有击球相位、question coverage 和 2-vs-4 selector 均为
`null/blocked`。一次过早评分的 v2 已保留但撤销全部回球列，只接受其与 v3/v4 相同的安全子树。
详见 `configs/motion_video_gmr_phase_safety_results_20260711.json`。

可复现命令在 Pod1 的独立 control bundle 内运行，不改 training checkout：

```bash
CUDA_VISIBLE_DEVICES= /workspace/hope_isaac_venv/bin/python \
  screen_motion_gmr_phase_safety.py \
  --manifest motion_video_gmr_phase_safety_prereg_20260711.json \
  --expected-manifest-sha256 232cd9ef1a72381895b54c75cc87c82e991d9c605ea169e86605b3afb9e64e15 \
  validate
```

只有 schema-2 + HOPE +X reground（或独立验证的显式 proper-rigid 4x4 transform）和 mirror 语义到齐后，
才可将 `frame_contract.returnability_enabled` 改为 true 并用同一题纸重跑；禁止把速度峰帧当击球帧。

这个条件随后以**canonical counterfactual**方式满足，而不是伪造录制现场桌位：十条 final MP4 的
正常方向中文背景字 + GMR 右臂主导共同验证 no-mirror/no-side-swap；每条矩阵只由 frame-0 pelvis
heading/XY 与 ground z 生成，映到 HOPE robot origin/+X，题目结果不能反调矩阵。v5 实际消费
64 题后，所有 zero-retarget exact coverage 都是 `0/64`，所以 2-vs-4 不可判；Franco 反手拉
B/C 只有 intrinsic `32/32`、`27/32` 的候选证据。`TOPP` 暂停到显式 spatial retarget、schema-2、
L0/L1、桌网和动力学完成；最终由智元 vendor MuJoCo Gate3/Gate3B（无 reset）主判。详见
`docs/interfaces/motion_gmr_hope_frame_contract.md` 和
`configs/motion_video_gmr_phase_counterfactual_results_20260711.json`。

v5 后的 spatial-retarget 不是“把动作整体拖到球上就算过”。预注册 v1 只能对整条
motion 原子施加保地 SE(2)（R0 平移；R1 冻结小角度 yaw+平移），且全十动作都必须
做同卷。它是 planner 站位要求，不是录制桌/相机外参。当前只能产 proposal；每个精确
candidate 必须物化 schema-2，再重跑 L0、vendor-MJCF L1 和整轨迹桌网 `>=5mm` 门，才能
继续动力学/TOPP。见 `docs/operations/run_motion_spatial_retarget_screen.md`。

schema-2 的 L0 replay 不能假设已存 pelvis body pose 就是 producer 原始 free-joint qpos。当前格式只存
MuJoCo FK 后归一化并投影到 float32 的 body pose；把它再次当 qpos 注入后要求所有 pose/velocity byte
equal 是非幂等合同。反手拉 B 的 V1 dry-run 已以 1 个 float32 格量级 fail closed，未生成证书。V2
只把这个不可重构比较改为预注册的 field-specific 数值门：link pose 两个
[`ULP`](DEFINITIONS.md#float32-ulp) 格并带物理 cap，COM velocity 从已存 link pose + exact `body_ipos`
按 50 Hz 差分误差传播，angular/joint velocity 仍 byte exact；joint range、ground、support-foot 和
safety 门完全不变。长期若要恢复 bit replay，应升级 motion schema 显式保存 pre-normalization
free-joint qpos，而不是继续放宽 schema-2。该 V2 已用 Pod2 exact runtime 发布 L0 certificate
`60c08185...afc6`，仅解锁 vendor L1 自碰/球拍自打；桌网、动力学、训练和真机仍 blocked。详见
[`EXP-MOTION-BACKHAND-LOOP-B-L0`](experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)。

当前资产:`hope_{forehand,backhand}_{v5,oblique,v4}_cal.npz`(v4=hopex 视频重跑;**hopex 资产
与 v4_cal 同底片**——真源都是 raw_video_hopex/*_v4.mp4,动作组消融里两者不构成独立对照)。
**swing 对试产件**(2026-07-08,修复版管线全链,判炸器双 PASS):
`regen_test_0708/npz/hope_{forehand,backhand}_swing_warmup_cal.npz`(98/108 帧)——正式入库
差:触球帧登记、会话握拍复核(试产暂借 Rz5Rx40)、题库生成;动作组消融成员=swing/v4/v5。

## 第二段:标准老师动作 → 题目与训练

- **题库生成器** `gen_stage1_questions.py`(yikang):从 _cal 的锚点帧出题(来球 → 反解应有
  拍面+拍速=答案),难度标签、train/exam 切分、`--phase-scan` 相位×可回球性扫描、`--grip`。
  **_cal 是唯一合法输入源**(raw+登记表旋转会把锚点放错 ~11cm,fail-closed 守卫已加)。
- **训练接线**(yikang,分支 stage1-fixed-point):拍面指令观测(175→179,法线 3+ρ 占位 1)、
  题库目标、腕从模仿踢除(R16,SMASH 论文同款且背书)、curriculum v1、loader 强制校验。
- **评估器** `mujoco_eval_onnx.py`(franco):179 契约支持已交付(175/180 逐字节回归过);
  阶段考卷开关(固定点/无旋/速度档)排本轮。

## 第三段:泛化老师接口(适配器;franco 2026-07-06 修正版)

三个轴,按人打球的真实机构设计:

| 轴 | 机构 | 状态 |
| --- | --- | --- |
| **整身朝向旋转** | 整套动作(全身)绕锚点旋 φ——打向适配;rally_yaw 机制复用,同一条 clip 天然服务任何方向 | 机制已验证(标定/扫描全程在用) |
| **加减速** | 变速重定时(R14 机制:参考变速+速度需求同步缩放) | 已实现,消融待变速考卷 |
| **拍面/侧向速度** | **小臂带动为主**(肩+肘+腕全链、从引拍起全程渐变),不是触球窗拧腕——v1(腕局部+0.24s 窗)已被包络实验否决:限位裁剪 96-100%、腕速需 8-10 倍、拍位漂移 33cm。v2 两条:全程链级 morph(每题的拍面差当"迷你握拍角"烤整条)或 **预烤变体库+SMASH 最近邻检索**(特征=锚相对击球点+拍面,加 ε 扰动模拟部署预测误差) | v1 否决;v2 设计定,变体库路线优先 |

SMASH 采纳清单(全部进消融,franco 2026-07-06):击球窗分通道奖励(位置 0.02s 紧/朝向速度
0.1s 宽)、腕踢除(已有)、相位依赖任务噪声(叠 A1 成 v3)、tracker 过滤、区域自适应采样。
分歧记录:SMASH 不给 actor 拍面指令(拍面由速度隐含)——轻旋场景成立;我们补旋时面≠f(v)
→ 消融臂「显式拍面指令 vs 速度隐含」裁决,若隐含够用契约日省 3 维。

## 归属与状态

| 段 | 负责人 | 状态 |
| --- | --- | --- |
| 视频→_cal 管线 | franco(标定/烤入)+ 管线脚本 | ✅ 六条 clip 产出;握拍终值联合复核在跑 |
| 题库/训练接线 | yikang | ✅ 分支就绪,反手 bank 重出中 |
| 179 评估器/考卷开关 | franco | 评估器 ✅;阶段考卷开关本轮 |
| 适配器 v2(变体库+检索) | franco 设计 / 实现待认领 | 设计定稿,阶段 1 第二波消融 |
| S1 反手臂(新基线候选) | 合流:yikang 接线 + franco 评估器 | 机制检查就绪,今晚点火 |
