# 随挥结束教师状态制品合同

状态：**源码候选已闭合；4096-environment 运行门未过，保持 `Partial` 与
`launch_authorized=false`。** 这里的“教师状态”是从一个既有策略自然完成整段动作时采到的机器人状态，
用来让两个消融臂从同一外生恢复分布起步；它不是动作模仿 teacher，也不是任意 episode timeout 快照。

相关术语见 [定义表](../DEFINITIONS.md)。正式运行步骤见
[producer operation](../operations/run_post_swing_teacher_capture.md)。

## 两段式信任边界

1. capture 状态只由 `MotionCommand` 的 natural-wrap 分支直接从 live articulation tensors 读取；没有公开 writer、
   module-global capability 或任何“传任意 arrays 即签发”的接口。配置时先用 `O_EXCL` 建立固定
   `natural_wrap_capture.claim.json` namespace，已存在即永久 fail closed。
2. producer 只能 no-clobber 发布 claim、`natural_wrap_states.npz` 和 `natural_wrap_capture.json`。artifact
   只证明 exact reviewed producer source、exclusive claim 与 runtime hard-contract bytes 相互绑定；普通 Python
   runtime 无法提供 callback 的密码学证明，所以 callback 名称/自报标签不作为安全证据。
3. `scripts/attest_post_swing_teacher.py` 是独立的一次性 consumer。它重新核对实际 checkpoint bytes、
   checkpoint 内嵌 schema-3/fresh-lineage/launch-claim、相邻 `params/training_contract.json`、checkpoint source、
   capture source、按序 motion bytes、runtime articulation joint order 和 plant joint-velocity limits，才可
   no-clobber 发布 `teacher_receipt.json`。
4. trainer 同时重读 receipt、exclusive claim、raw capture result 和 NPZ；任何一个 SHA、字段、source hash、motion/joint order、
   velocity bound 或 no-clobber provenance 不一致都 fail closed。

这四份输入均用一次 `O_NOFOLLOW` open、同一 descriptor 的前后 `fstat` 与单个 immutable byte buffer；SHA、
JSON 解析和 `np.load(BytesIO)` 不得重新开路径。NPZ ZIP member 必须逐名唯一，JSON boolean 不得冒充 integer，
integer/float 也不得靠隐式 coercion 过门。

## 状态数值合同

NPZ 只能有三个 float32 array：

- `root_state_origin_relative[count,13]`：环境原点相对位置、`wxyz` 四元数、COM 线速度、角速度；
- `joint_pos[count,J]`；
- `joint_vel[count,J]`。

所有值必须 finite；四元数范数绝对误差不超过 `1e-4`。joint position 在 trainer adoption 时再对当前
soft joint limit 逐元素检查。joint velocity 必须小于 schema-3 runtime contract 的逐关节 plant limit。
floating base 没有 PhysX actuator velocity limit，因此 producer 必须预注册一个正的 root linear norm 上限和
root angular norm 上限；attestor 与 trainer 都重查同一数值，不能由 receipt 临时放宽。

## 首个 reset 验收

正式消融必须启用 `require_ready_at_init` 与 `fail_fast_first_reset`，并在 hard contract 中绑定：

- 初始 cohort 最少实际采用的状态数；
- 最少采用比例；
- 实际比例与 `post_swing_start_prob` 的最大绝对偏差；
- runtime write 后 root/joint position/joint velocity readback 是否与目标一致。

只有 Bernoulli `selected` 不够；`started` 必须在两次 simulator write 返回后计数，要求 readback 的队列还必须
看到相同状态。4096-environment Pod probe 未证明这些运行时行为前，本能力不得解锁训练。

## 默认关闭兼容性

receipt/capture 路径为空、首 reset gate 关闭时，不采样、不写文件，也不向 schema-3 training contract 添加
null/default 字段。`training_contract_extension()` 的回归要求 default mapping 展开前后 canonical bytes 相同；
因此旧 checkpoint 的 hard-contract SHA 不因这项未启用能力漂移。
