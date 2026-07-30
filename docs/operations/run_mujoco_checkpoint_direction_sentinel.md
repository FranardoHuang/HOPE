# 运行 fresh N5 MuJoCo 检查点方向哨兵

本工序只运行
[MuJoCo 检查点方向哨兵](../DEFINITIONS.md#mujoco-checkpoint-direction-sentinel)：用 CPU 对 fresh
N5 检查点做方向、plant 与桌碰安全诊断。它复用同一张不可变 Python
[`BankExam`](../DEFINITIONS.md#python-bankexam) 题纸，但当前**不产生可入账的成功率、上台率或
动作优先级**。现阶段正常完成也应因 MuJoCo↔PhysX 桌碰传感器尚未获证而返回 `3`。

每个命令行参数的人话含义列在命令后的参数表中；共享术语见
[术语表](../DEFINITIONS.md)。任何真实机器人命令都不属于本工序。

## 1. 发射前冻结输入

只允许 clean、exact checkout。先从 fresh N5 训练产物和它自己的合同中确定以下绝对路径，禁止沿用
旧 N4 的 ONNX、动作顺序、题库或 schedule：

```bash
SOURCE=/workspace/codexschema/nohope_n5_eval_checkout
PYTHON=/workspace/hope_isaac_venv/bin/python
MJCF=/workspace/codexschema/vendor/a3_pingpong.xml
EXAM_BANK=/workspace/codexschema/fresh_n5/exam_bank.npz
EXAM_SCHEDULE=/workspace/codexschema/fresh_n5/exam_schedule.json
M0=/workspace/codexschema/fresh_n5/model_00000/policy.onnx
M1=/workspace/codexschema/fresh_n5/model_01000/policy.onnx
MOTIONS=(
  /workspace/codexschema/fresh_n5/motions/action_00.npz
  /workspace/codexschema/fresh_n5/motions/action_01.npz
  /workspace/codexschema/fresh_n5/motions/action_02.npz
  /workspace/codexschema/fresh_n5/motions/action_03.npz
  /workspace/codexschema/fresh_n5/motions/action_04.npz
)

test -z "$(git -C "$SOURCE" status --porcelain)"
test -f "$MJCF"
test -f "$EXAM_BANK"
test -f "$EXAM_SCHEDULE"
test -f "$M0"
test -f "$M1"
test "${#MOTIONS[@]}" -eq 5
for motion in "${MOTIONS[@]}"; do test -f "$motion"; done
```

这里 `M0/M1` 只是“两个显式检查点”的人话占位，不是固定 run name。正式执行前要把路径换成
fresh N5 的实际绝对路径，并把 checkpoint、ONNX、MJCF、题库、schedule 和五件 ordered motion 的
SHA-256 另存到发射 ledger。ONNX 内 metadata 必须与这五件动作的数量、顺序和 bytes 一致。

## 2. 创建全新 CPU-only namespace

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/workspace/codexschema/mujoco_direction_sentinel_n5_${STAMP}"
test ! -e "$OUT"

CUDA_VISIBLE_DEVICES='' "$PYTHON" \
  "$SOURCE/hope_training/whole_body_tracking/scripts/mujoco_checkpoint_direction_sentinel.py" \
  --milestone "n5_initial=$M0" \
  --milestone "n5_candidate=$M1" \
  --output-dir "$OUT" -- \
  --mjcf "$MJCF" \
  --motion-files "${MOTIONS[@]}" \
  --target-source bank \
  --exam-bank "$EXAM_BANK" \
  --exam-schedule-json "$EXAM_SCHEDULE" \
  --steps 0 \
  --qdes-clamp \
  --require-obs-norm-provenance
RC=$?
test "$RC" -eq 3
```

参数解释：

| 参数 | 人话 |
| --- | --- |
| `--milestone LABEL=/abs/policy.onnx` | 给一个 exact ONNX 检查点贴稳定标签；可重复多次，但标签不得重复。 |
| `--output-dir` | 本次哨兵的全新输出根；已有路径即拒绝，符合 [`no-clobber`](../DEFINITIONS.md#no-clobber)。 |
| `--mjcf` | exact vendor MuJoCo XML 模型绝对路径；其 bytes 会进入 execution receipt。 |
| `--motion-files` | 五件训练动作，必须按 ONNX/题库的 frozen action order 逐件列出。 |
| `--target-source bank` | 强制使用不可变 BankExam，不使用随机 box 或 venue-ball 分布。 |
| `--exam-bank` / `--exam-schedule-json` | 同一张题库和固定题序；所有 checkpoint 共卷、失败不删分母。 |
| `--steps 0` | 由完整 schedule 和 episode 上限计算保守安全 cap，不截断题纸。 |
| `--qdes-clamp` | 使用训练/部署同口径的软关节位置限位；clamp 依赖仍单独报告。 |
| `--require-obs-norm-provenance` | normalizer 来源不完整就停止，不把 identity normalization 当有效策略输入。 |

封装器自己拥有 ONNX、输出目录、零 action noise、inexact 标记、post-step qvel proxy、桌面障碍和
无渲染参数；passthrough 中重复、缩写或改写这些参数会 fail closed。它强制
`CUDA_VISIBLE_DEVICES=''`，不占 GPU、不读取或修改训练锁，也不发送 signal。

## 3. 返回码与机器收据

| 返回码 | 含义 | 后续 |
| ---: | --- | --- |
| `0` | 所有 machine gate 通过 | 仅在完整 plant/桌碰等价证书将来进入 reviewed source 后才可能；仍不自动授权 Gate3/真机。 |
| `3` | evaluator 完成并写出证据，但至少一个 stop gate 触发 | 当前预期。保留目录；不得改名为正式分数。 |
| `2` | 输入、evaluator 或 summary 无效 | 保留失败现场，换新 namespace 修复；禁止覆盖续写。 |

顶层固定产物：

- `direction_sentinel.json`：逐 milestone gates、ONNX SHA、执行命令、方向与安全摘要；
- `direction_sentinel.csv`：逐 mode 的 actor/q_des/raw-qvel/table/fall 索引；
- 每个 milestone 子目录中的 evaluator summary、三份 CSV 及 stdout/stderr。

封装器会重算每份 CSV 的 SHA，并要求它们都位于自己的 milestone 子目录。plant parity 未通过时，
JSON 里的 pass/return/composite 字段必须为 `null`，CSV 对应格必须为空；任何 numeric 泄漏都会触发
stop。

## 4. fresh N5 晋级前读数

每个动作至少逐项报告：

1. 31 维首次 actor output、逐关节最大绝对 actor output；
2. 首次 raw/applied q_des、首次 clamp 数和全程 clamp 比例；
3. physics substep 前 qvel 比率、`mj_step` 后且 proxy 前的 raw qvel 比率、proxy 后比率；
4. exact strike tick 的世界系球拍线速度、目标速度、带符号拍面、来球速度，以及
   `dot(v_racket - v_ball, n_target)`；
5. 同 tick 是否撞桌、是否 physical fall；撞桌、摔倒和 tracking guard 必须分开；
6. table-contact binding、ready state、live policy/plant facts、MJCF load bytes 与完整 execution SHA。

promotion 只看 proxy 前 raw qvel；post-step qvel clamp 只允许延续方向诊断，不能修复 plant parity。
任一同 tick 桌碰/摔倒都会清零阈值式 pass/return 字段，但保留连续误差与接触诊断。

## 5. 当前硬阻断

- accepted MuJoCo↔PhysX table/contact-sensor parity certificate 仍为空；所以 success-score authority
  必须为 false。
- 本机没有 MuJoCo Python runtime，真实 table/physics 的 6 个集成测试只在此机 skip；需在 Pod 的
  CPU 环境补跑，不能用 dependency-light 单测替代。
- fresh N5 exact ONNX、ordered N5 motion、exam bank 与 schedule 尚未在本记录绑定，故本页当前只是
  可复现工序，不是已执行结果。
- in-memory 障碍只复刻当前 Isaac table-top slab；桌腿等场景闭包仍不在这条诊断中。
- Python BankExam 使用解析回球，不是真实球拍—球碰撞，不替代 vendor Gate3/Gate3B，更不授权真机。

旧 N4 的 `0/52`、视频和 ledger 已判为 **INVALID diagnostic**：策略首帧 actor output 极大、
31 关节中 26–27 个 q_des 依赖 clamp，且 `mj_step` 后 raw qvel 在首个约 5 ms 已超出 PhysX bound。
这些产物只能复现“当前 MuJoCo plant 合同不成立”，不得进入 fresh N5 分动作成绩表，也不再追加 N4
正式视频。

## 6. 源码验证

```bash
cd "$SOURCE"
python3 -m py_compile \
  hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
  hope_training/whole_body_tracking/scripts/mujoco_checkpoint_direction_sentinel.py

python3 -m pytest -q \
  tests/test_mujoco_table_scene.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_align_flags.py \
  hope_training/whole_body_tracking/tests/test_mujoco_eval_p0_contracts.py \
  hope_training/whole_body_tracking/tests/test_mujoco_ready_state_contract.py \
  hope_training/whole_body_tracking/tests/test_mujoco_direction_sentinel.py
```

2026-07-28 当前分支结果为 `109 passed, 6 skipped`；6 个 skip 都是当前 host 缺 MuJoCo runtime 的
可选 table/physics 集成测试。详见
[实验记录](../experiments/2026-07/EXP-MUJOCO-CHECKPOINT-DIRECTION-SENTINEL-20260728.md)与
[G06](../gates/G06_isaac_to_mujoco.md)。
