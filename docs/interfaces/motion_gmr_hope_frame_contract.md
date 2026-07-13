# GMR 空挥到 HOPE +X/虚拟桌坐标合同

日期：2026-07-11
状态：canonical counterfactual frame/mirror runtime 已验；实拍桌外参、触球真值仍为空

## 这条合同允许什么

它只允许把 Franco/v6/v7 十条空挥放进**固定在机器人原点的 HOPE 标准虚拟桌**里做
64 题反事实相位/覆盖诊断。它不声称洗衣房里存在球桌，也不把空挥视频解释成实拍击球。

- source：已内容寻址且通过离散 grounding/240 Hz 安全屏的 A3 GMR MuJoCo world，地面 `z=0`；
- target：env-local robot origin，`+X=向对手`、`+Y=机器人左侧`、`+Z=向上`；
- table：near edge `x=0.5 m`、surface `z=0.76 m`、center `y=0`、长 `2.74 m`、
  宽 `1.525 m`、网高 `0.1525 m`；这是 HOPE canonical counterfactual table；
- racket：vendor MJCF `right_racket` site，拍面轴沿 site local `+Y`，速度来自
  `mj_differentiatePos + mj_objectVelocity`；
- contact truth：永远是 `null`，直到另有真实球/拍接触观测。

## 变换是怎么锁死的

每条 clip 独立生成一个 proper-rigid `4x4`：

1. 只读 frame-0 pelvis position/quaternion；
2. 将 pelvis local `+X` 投影到地面，绕 `+Z` 旋转到 target `+X`；
3. 将 frame-0 pelvis `XY` 平移到 `(0,0)`；
4. `Z` 不平移，保持已经审过的 ground plane；
5. 要求 `RᵀR=I`、`det(R)=+1`、mapped heading `0`、mapped root XY `0`。

这一矩阵在看任何 64 题结果前已经冻结；严禁为了提高覆盖率再调 yaw、XY 或桌位。
十矩阵 semantic SHA 为
`5eb91bba6ec13d760b8084a82a43aaa53ab5b1427e9143ccde6e8d328e78015a`。

## 镜像证据

最终 MP4 的 midpoint crop 逐条绑定。十条画面右侧洗衣机中文标签和控制面板均为正常、
未反射方向；crop 像素不进 git，只保存 source/frame/crop SHA。独立的 GMR 关节证据要求
右臂七关节相邻帧平方增量至少是左臂的 `5x`，十条实测最小约 `9.98x`。两层同时通过才写：

```text
mirror_status = verified_not_mirrored
side_swap_required = false
```

证据账本：

- `configs/motion_video_mirror_witness_review_20260711.json`
- `configs/motion_video_gmr_frame_contract_results_20260711.json`

## v5 诊断结果与边界

同一张 64 题纸已实际消费，但十条 motion 的 exact zero-retarget counterfactual coverage
全部是 `0`。这表示“拍位、拍速/拍面与固定题在同一帧没有同时满足”，不表示动作本身无效：

- Franco 反手拉 B：intrinsic `32/32`，phase `0.5444`；
- Franco 反手拉 C：intrinsic `27/32`，phase `0.5155`；
- Franco 反手拉 A：intrinsic `1/32`；其余为 `0`。

intrinsic 会把球假想移到该帧拍心，因此 B/C 只能作为“显式空间重定向”候选。它们在该
intrinsic 峰距最近 immutable question position 仍约 `0.165/0.237 m`。所有 library 都是
exact `0/64`、common support `0`，所以 2-vs-4 不能据此判优。

## 下游强制顺序

`TOPP = paused_until_spatial_retarget`。候选必须先完成显式、预注册且不反调 frame 的空间重定向，
再做 schema-2、L0/L1、桌网 swept-volume、动力学/平衡；TOPP 后重复全套审计。动作与 2-vs-4
最终以智元 vendor MuJoCo Gate3/Gate3B 为主门，**不允许 reset 掩盖动作之间的连续状态**。

spatial-retarget v1 不修改本页已冻结的 GMR-world→HOPE 坐标变换，而是在该目标坐标内
增加一个“站哪里用哪一帧”的规划变量。只允许一个原子应用到整条 motion 的保地
SE(2)：绕 HOPE robot origin 旋转，再平移 XY；Z/scale/镜像/关节/逐帧修改一律禁止。它的
transform 不是 capture extrinsic，也不能回填本页的 per-asset frame matrix。完整合同在
`configs/motion_video_spatial_retarget_prereg_20260712.json`。没有 candidate-specific schema-2/L0/L1/
桌网整轨迹证书的输出只能叫 proposal。

## 若要回答“录制现场真实桌位”

现有空挥不足，必须另采标定段；不得从本轮 `0/64` 反推桌外参。最低采集规格：

1. 同一未裁剪视频先拍左右不对称、可解码的 AprilTag/ArUco 板，明确 `L/R` 和像素镜像；
2. 同时拍全桌四角、两网柱、地面机器人 root 点和一条可量尺长度，保存 tag 尺寸/ID；
3. 红/黑胶皮分别正对镜头静止一帧，登记右手、胶皮面和握拍；
4. 相机内参、畸变、原始文件 SHA、时间戳和 table-corner PnP residual 一并入账；
5. 至少两视角或 mocap/深度独立复核 chirality 与尺度；误差门预注册后才能生成 capture extrinsic。

在这些材料到齐前，`real_capture_returnability=null`。

## S0/M0 post-GVHMR 与横移末态站距合同（2026-07-13）

`configs/motion_post_gvhmr_{s0,m0}_prereg_20260713.json` 把新五条 exact GVHMR output 与各自 execution
record、final queue state、binding、structural audit 和 donor canonical-beta artifact 收成下一阶段的唯一
输入。handoff 只允许另建 canonical-beta materialization；GMR 和 schema-2 仍必须按各自 exact source/body
order/output 另做 prereg。schema-2 的位置点是 link origin，线速度点是 center of mass，且必须绑定
runtime articulation body names；缺任一字段不能消费。

M0 的“回到准备姿态”在 robot-coordinate GMR 后定义。每条 clip 去掉公共 root XY，并把 heading 对齐到
该 clip 初始准备朝向；随后分别在人工绑定的 `ready_before` 与 `ready_after` 窗口对
`d_xy = right_foot_xy - left_foot_xy` 求稳健中位数。acceptance 必须同时保留横向站距与前后脚错位，不能
用双脚并拢、更窄站姿或绝对足位姿相等替代。foot-site mapping 与数值容差尚未预注册，因此当前没有
M0 站距通过结果。S0 仍为无球空挥，不能消费拉球题或声称高点拍压有效。复现命令见
[`run_motion_post_gvhmr_exact.md`](../operations/run_motion_post_gvhmr_exact.md)。

post-GVHMR handoff 后的 canonical-beta 层进一步冻结在
`configs/motion_canonical_betas_{s0,m0}_prereg_20260713.json`。它只将 exact donor 写进人体
`smpl_params_global.betas`，不会生成 A3 足点。M0 的 ready windows、`d_xy` 定义、横向分离与前后错位两个
必保留分量已经固定，但 `foot_site_mapping`、initial/terminal `d_xy`、component tolerance 和
`stance_passed` 必须全部保持 `null`，producer 固定为未来另行预注册的 exact GMR。任一提前填值、额外
`passed` 字段或更窄/合脚替代都 fail closed。canonical-beta 复现入口见
[`run_motion_handoff_canonical_betas.md`](../operations/run_motion_handoff_canonical_betas.md)。
