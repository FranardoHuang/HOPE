# Phase 1 Wave-B 下肢稳定六格队列

本文只操作 [Wave-B B0/B1/B2](../DEFINITIONS.md) 的仿真训练队列：

- B0：两种新下肢 reward 均为零，但两套 measurement probe 都启用；
- B1：静态 `v4rg` 十二腿关节软模仿，weight `+0.5`；
- B2：不读取 motion reference 的有符号脚距下界与实际腿速 bundle，weight `-0.25`。

实验真源是
[EXP-P1-LOWER-BODY-STABILITY-20260720](../experiments/2026-07/EXP-P1-LOWER-BODY-STABILITY-20260720.md)。
本页不授权真机、judge、第二 seed 或 M0 横移老师。

## 入口与不变量

- config：`configs/phase1_lower_body_stability_20260720.yaml`
- renderer：`scripts/run_phase1_lower_body_stability_queue.py`
- reviewed manifest：`configs/phase1_lower_body_stability_launch_manifest_20260720.json`
- exact source：`5db7366aaa1562d592093dc0d512ec212f14e39e`
- remote checkout：`/workspace/codexschema/nohope_lowerbody_wave_20260720`
- probe：每格 `4096 environments × 24 control steps × 2 PPO updates`，自然退出终档
  `model_6701.pt`
- long：每格继续 `1001 updates`，absolute milestones `6900/7200/7700`
- held control：`action_rate_l2=-0.1`；processed-qdes slew 必须缺席
- evidence：旧 parent 的 reward contract 有意改变，因此所有后代
  `training_contract_lineage_exact=0`，只属于 causal continuation

六个科学 `run_name`（唯一运行名，定义见[术语表](../DEFINITIONS.md)）已写在 config 中。不要手改
rendered SSH argv、run root、GPU 或权重；需要变化时修改 config、tests、实验记录和 manifest 后重新复核。

## 0. 先核对 main 权威与工作树

在仓库根执行：

```bash
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$BALANCE_REPO_ROOT"

git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test -z "$(git status --porcelain)"
git cat-file -e origin/main:configs/phase1_lower_body_stability_20260720.yaml
git cat-file -e origin/main:scripts/run_phase1_lower_body_stability_queue.py
git cat-file -e origin/main:configs/phase1_lower_body_stability_launch_manifest_20260720.json
python3 - <<'PY'
import subprocess

text = subprocess.check_output(
    ["git", "show", "origin/main:docs/NOW.md"], text=True
)
title = "- **[11｜P1] 稳定机制 Wave A/B。**"
start = text.find(title)
if start < 0:
    raise SystemExit("Wave A/B claim missing from origin/main NOW")
end = text.find("\n- **[", start + len(title))
entry = text[start:] if end < 0 else text[start:end]
required = (
    "责任人 franco；执行者 Codex；执行分支",
    "Franco_codex/balance-ablation-round-20260720",
    "Franco_codex/lowerbody-wave-queue-20260720",
    "phase1_lower_body_stability_20260720",
)
if any(value not in entry for value in required):
    raise SystemExit("Wave A/B owner/executor/both branches/queue id not bound in one NOW entry")
PY
```

任一检查失败时，只允许运行本页的本地 validate/plan；不得渲染或执行 SSH。只要
`origin/main` 尚未同时登记 Wave-B queue id 与两个分支，检查就必须失败；只有合入并完成同一
NOW 行认领后才能渲染。
功能分支中的 `NOW` 不是授权。全局顺序和算力归属只看最新 `origin/main:docs/NOW.md`，本实验只定义局部依赖
`validate -> manifest preflight -> six probe receipts -> long`。

## 1. 本地验证与默认 NO-LAUNCH plan

```bash
LOWER_BODY_TEST_PYTHON=/Users/Franco/opt/anaconda3/envs/fast/bin/python
test -x "$LOWER_BODY_TEST_PYTHON"

"$LOWER_BODY_TEST_PYTHON" -m pytest -q -p no:cacheprovider \
  hope_training/whole_body_tracking/tests/test_lower_body_stability_wave.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py \
  tests/test_run_phase1_lower_body_stability_queue.py

"$LOWER_BODY_TEST_PYTHON" -m py_compile \
  scripts/run_phase1_lower_body_stability_queue.py \
  tests/test_run_phase1_lower_body_stability_queue.py

python3 scripts/run_phase1_lower_body_stability_queue.py \
  --queue configs/phase1_lower_body_stability_20260720.yaml
```

四套 focused tests 必须共同返回 `269 passed`（`35 + 124 + 68 + 42`）。最后一条命令必须
返回 `commands_emitted=false`，且 JSON 中没有 `ssh_argv`。默认调用不是 dry-run
SSH；它根本不生成远端命令。

## 2. 核对 reviewed manifest

```bash
MANIFEST=configs/phase1_lower_body_stability_launch_manifest_20260720.json
MANIFEST_SHA="$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"

shasum -a 256 \
  configs/phase1_lower_body_stability_20260720.yaml \
  scripts/run_phase1_lower_body_stability_queue.py \
  "$MANIFEST"
```

本次冻结值依次是 `7193d789...3146`、`abc8d34d...8556`、`bae461b0...5565`；manifest 内部
canonical content SHA 是 `feed3ed1...d576`。任一 bytes 变化都要生成新 manifest 并更新实验记录。

manifest 必须逐项绑定：

1. clean detached source commit 与十份 required source-file SHA，包括冻结
   `algo=ppo` 的 `num_steps_per_env=24` 源文件；
2. 当前 config/renderer bytes；
3. A3 ignored runtime tree 的 SHA、文件数和总字节；
4. `model.usd` 及其完整六文件 bundle；
5. 正/反手静态 `v4rg_runtime_order_v3`、schema-3 train bank；
6. W/V parent checkpoint 与相邻 hard contract。

checked-in manifest 不是 ambient authority。每次 command render 都要显式给 `--launch-manifest` 和
`--expected-launch-manifest-sha256`；placeholder、旧 config SHA 或不同 source commit 都会拒绝。

## 3. 只渲染 probe 命令

```bash
python3 scripts/run_phase1_lower_body_stability_queue.py \
  --queue configs/phase1_lower_body_stability_20260720.yaml \
  --authorize-launch --stage probe \
  --launch-manifest "$MANIFEST" \
  --expected-launch-manifest-sha256 "$MANIFEST_SHA" \
  > /tmp/phase1_lower_body_stability_probe_commands.json
```

renderer 只使用本地只读 Git subprocess 复核 `origin/main` authority；没有 SSH、signal、trainer 或
remote-write 执行 surface，它只输出 argv 数组。执行前逐格复核
`job_id/resource/run_dir/queue_claim_content_sha256`。同一 Pod 一次只能 boot 一个 Kit；Pod1 与 Pod2
可以各 boot 一格，看到首个 `Learning iteration` 后才轮到同 Pod 下一格。不得并发冷启动三格。

每个远端 body 都会先做 source/manifest/tree/checkpoint/GPU/Hydra compose 检查，再以 no-clobber 写
manifest、claim 和 metadata。专用 supervisor 在 trainer 前绑定 PID/PGID/starttime/argv，等待
`model_6701.pt` 自然退出并记录 terminal status。不要对 probe 手工 TERM/KILL 来伪造 receipt。

## 4. 自然退出后生成并收回 receipt

对每个 JSON job 只执行同一行给出的 `probe_verifier_ssh_argv`。verifier 要求：

- bound PID、整个 exact PGID 都已消失，assigned GPU 无 compute PID；
- 日志没有 traceback、OOM、NaN/Inf 或 semantic fatal；
- terminal checkpoint 含 finite policy/value/optimizer/obs normalizer/privileged normalizer；
- hard contract 同时含 pose/bundle 两块，B2 明写 `uses_motion_reference=false`；
- motion/bank SHA 与 manifest 相同，processed-qdes block 缺席；
- update `6700/6701` 的 pose/bundle ledger 分母一致，B0/B1/B2 enabled counter 对应正确；
- 每个 update 的 pose/bundle observed 必须严格等于 `4096 env × 24 control steps = 98304`；`4096`
  或 `98304 ± 4096` 都拒绝，不能把缺 rollout 的日志当完整 update；
- reference-motion magnitude 非零，三个 bundle 有界和满足 component identity；
- signed stance-width sum 只要求 finite，允许为负。

把六份 canonical `probe_receipt.json` 收到一个本地、非 Git 目录：

```text
/tmp/phase1_lower_body_stability_probe_receipts_20260720/
  w_b0/probe_receipt.json
  w_b1/probe_receipt.json
  w_b2/probe_receipt.json
  v_b0/probe_receipt.json
  v_b1/probe_receipt.json
  v_b2/probe_receipt.json
```

probe 不是科学结果，不得进入行为成绩表或据此采用 reward。
receipt 的 SHA/claim/verifier 绑定用于 trusted-operator 模型下防误配、误复用和手工布尔解锁，不是带密钥
签名；能够恶意改本地代码和 receipt 文件的操作者不在本门威胁模型内。禁止手写 receipt，并保留远端
claim/binding/status/checkpoint/log 供复核；若未来需要抵抗恶意制品写入者，必须另加远端重读或签名证明。

## 5. 六份 receipt 才能渲染 long

```bash
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test -z "$(git status --porcelain)"
git cat-file -e origin/main:configs/phase1_lower_body_stability_20260720.yaml
git cat-file -e origin/main:scripts/run_phase1_lower_body_stability_queue.py
git cat-file -e origin/main:configs/phase1_lower_body_stability_launch_manifest_20260720.json
python3 - <<'PY'
import subprocess

text = subprocess.check_output(
    ["git", "show", "origin/main:docs/NOW.md"], text=True
)
title = "- **[11｜P1] 稳定机制 Wave A/B。**"
start = text.find(title)
if start < 0:
    raise SystemExit("Wave A/B claim missing from origin/main NOW")
end = text.find("\n- **[", start + len(title))
entry = text[start:] if end < 0 else text[start:end]
required = (
    "责任人 franco；执行者 Codex；执行分支",
    "Franco_codex/balance-ablation-round-20260720",
    "Franco_codex/lowerbody-wave-queue-20260720",
    "phase1_lower_body_stability_20260720",
)
if any(value not in entry for value in required):
    raise SystemExit("Wave A/B owner/executor/both branches/queue id not bound in one NOW entry")
PY

python3 scripts/run_phase1_lower_body_stability_queue.py \
  --queue configs/phase1_lower_body_stability_20260720.yaml \
  --authorize-launch --stage train \
  --launch-manifest "$MANIFEST" \
  --expected-launch-manifest-sha256 "$MANIFEST_SHA" \
  --probe-receipts-dir /tmp/phase1_lower_body_stability_probe_receipts_20260720 \
  > /tmp/phase1_lower_body_stability_long_commands.json
```

缺任一格、digest/claim/runtime/counter 不一致时，renderer 必须在生成任何 long SSH argv 前失败。
六条 long claim 都绑定同一个 receipt-set SHA。启动后只在 `6900/7200/7700` 三个绝对 checkpoint
做 matched-parent 判读；不能因某格早期好看而改变其他格预算。

## 6. 停止纪律

队列不自动 stop、retry 或 promote。若后续裁决要求停止，只能读取该格 launcher sidecar，并重新验证
numeric PID/PGID、leader starttime 与完整 argv；随后按
[RunPod 操作纪律](run_on_runpod.md#launch-rules-hard-won-2026-07-03--read-before-starting-any-job-over-ssh)
对 exact PGID 执行 TERM，再按同一身份条件决定是否 KILL。禁止 `pkill -f`、`killall` 或相似名字匹配。

M0 是独立输入资产门的 0/4 negative evidence，不是第七格。本队列没有 M0 asset key、motion path、
job 或 claim；未来左右移动老师必须另过 schema-2、L0/L1、桌网和动力学门。

## 7. 输出与文档更新

每次状态变化同步更新：

- 本实验记录：manifest/receipt/checkpoint/指标与采用/拒绝结论；
- G05：可复现命令、结果、输入/输出与限制；
- `PROGRESS.md`：一条带日期摘要；
- 只有 long 真正启动、setting/成绩/owner 变化时，才在最新 main 的 `NOW` 统一队列更新；
- 只有重要能力/结论进入 main 时才更新 `TIMELINE`。

本文没有任何真机命令。G07 安全门闭合前不得把训练 argv 改成部署或真实机器人控制。
