# 轻量训练队列的运行绑定与 checkpoint 里程碑取证合同

状态：`Partial / E1 source gate`。源码与依赖轻量的攻击测试已通过；尚未用合入该能力的 exact source
启动 Pod trainer，也没有远端 [`run_binding.json`](../DEFINITIONS.md#trainer-run-binding) 或里程碑证据。

## 解决什么问题

训练器会自行生成 `experiment/timestamp_run_name` 形式的 RSL-RL 日志目录。外部控制器若按最新 mtime、
目录 glob 或 stdout 猜目录，就可能把一个 job 的 checkpoint 归给另一个 job。当前合同让**训练器本身**在
选定日志目录后、创建 scene 前发布不可覆盖的运行绑定；只有 exact source 已包含本接口时才在 YAML 显式设
[`runtime_binding: true`](../DEFINITIONS.md#runtime-binding-flag)。后续
[`milestone attestor`](../DEFINITIONS.md#milestone-attestor)只允许沿该绑定读取预注册 checkpoint。

状态机只有下面一条正向路径：

1. queue harness 原子创建全新 `run_dir`、`milestones/` 和 schema-2 `queue_claim.json`；
2. launcher 证明 claim 实际位于 claimed Pod；trainer 核对 digest、真实 argv、PID=PGID、双读一致的
   `/proc` starttime、物理 GPU，以及 claimed source 的 actual HEAD/clean；
3. trainer 确定 exact RSL log directory 后，以 no-clobber 方式发布 `run_binding.json`；
4. 到 `+200/+500/+1000` 等 YAML 预注册里程碑时，取证器只拼出
   `<bound_log_dir>/model_<iteration>.pt`，不扫描目录；
5. checkpoint、相邻 hard contract 和 launch lineage 全部通过后，原子发布
   `run_dir/milestones/model_<iteration>.json`。

任何一步失败都不删除、覆盖或自动重试科学 namespace，也不产生 receipt。

## Source ignored-runtime-asset 绑定

P1 queue row 的 `source` 可包含 `ignored_runtime_asset`：`target_relative_path`、donor 的
`checkout/commit/relative_path`、`file_count/total_file_bytes/tree_content_sha256`，以及固定为 true 的
`symlinks_forbidden/target_must_be_gitignored`。schema-2 queue claim 的 `content.source` 必须逐字段保留整个
mapping；只保留 checkout/commit 会丢失训练实际加载的 URDF/mesh 身份，属于无效 claim。

水合 receipt 位于 source checkout 之外：
`/workspace/codexschema/lean_training_source_asset_receipts/<source-commit>/<canonical-asset-contract-sha>/<pod>/receipt.json`。
其 content 绑定 Pod、source、完整资产合同、target path、46-file inventory、Git-ignore 结论和 URDF
43/43 唯一 mesh 引用 closure，并有 canonical content SHA。science doctor 必须重算 target 与 donor 后消费
已有 exact receipt；不得在 doctor/binding/terminal 路径补造。这样 source clean 不会掩盖 ignored tree
缺失或漂移。该 receipt 只证明 source runtime asset 身份，不证明 scene boot、Reward、checkpoint 或行为。

`fill --execute` 的控制面也只有一条权威远端路径：每臂一次 `_launch_script` SSH，在同一个 per-GPU 短锁
中按 doctor checks → 容量 → namespace/claim → Kit 顺序执行。standalone `doctor` 仍是无状态诊断入口，
但不会在 `fill` 中先跑一遍再被 launch 重复；第一遍既未占槽也未写 claim，不能关闭 TOCTOU，只会增加一次
SSH、Git/module/Hydra compose 的暂态失败面。execute 结果 schema 2 用
`preflight_mode=embedded_in_atomic_launch` 明示该语义，不再返回独立 `doctor_output`。

## `run_binding.json` schema 1

外层必须包含 `schema_version=1`、`content` 与 canonical JSON 的 `content_sha256`。`content` 至少绑定：

- `job_id`、schema-2 queue claim 的 absolute path 与 canonical content SHA；
- exact normalized `run_dir`、`binding_path`、`rsl_log_dir` 和 `run_name`；
- `process.pid`、`process.pgid`、Linux `/proc/<pid>/stat` 的 `starttime_ticks` 与完整 argv；
- claim-declared `pod`、物理 `gpu`、source checkout/commit 与 binding 时实际 clean HEAD；
- queue claim 的完整 training argv 与排序去重后的正整数 milestones。

claim 必须位于 `run_dir/queue_claim.json`，binding 必须位于 `run_dir/run_binding.json`。RSL 目录必须严格
位于 exact source 的 `hope_training/whole_body_tracking/logs/rsl_rl/` 下两层，叶名必须是 canonical
timestamp 加 claimed run name。trainer 的 `/proc` argv 必须与 claim 逐项相同，且
`CUDA_VISIBLE_DEVICES` 必须等于 claimed physical GPU；callback 再执行 read-only Git HEAD/clean 复核，
不能只信启动前 doctor。半套 claim/binding override 直接失败；`runtime_binding` 默认 `false` 的旧 source
和非科学 warmup 不注入二者，行为不变。P1 当前只接 fresh run：`checkpoint_path` 必须为空，且 terminal
milestone 必须严格小于 `max_iterations`，避免注册永远不会出现的 `model_N.pt`。

发布使用同目录 `O_CREAT|O_EXCL` 临时文件、`fsync`、no-replace hard link 与 directory `fsync`。目标已经
存在、是 symlink，或并发 writer 已占用临时路径时一律失败，不提供 overwrite/update 模式。

## 里程碑取证

取证命令只接受 binding path 与一个预注册 iteration，不接受 checkpoint path、glob 或“latest”。它先重放
binding↔claim，然后检查 live PID 的 PGID/starttime/argv；PID 已复用时即使 checkpoint 存在也拒绝。原进程
已经自然退出可以继续取证，因为启动时身份已冻结，但 receipt 会诚实写 `process_state_at_attestation=exited`。

checkpoint 必须是 exact `model_<iteration>.pt` regular non-symlink file，filename iteration 与内嵌 plain-int
`iter` 必须相等。递归检查 checkpoint 中所有 floating/complex tensors，至少存在一个且 NaN/Inf 数为零。相邻
`params/training_contract.json` 必须是 schema 3，其逐字节 SHA 必须等于 checkpoint infos 的
`training_contract_sha256`；`training_contract_lineage_exact` 只接受 plain `0/1`，并原样写入 receipt，不能把
causal checkpoint 洗成 fresh。checkpoint infos 的 `training_launch_claim_sha256` 必须等于 bound claim SHA。
checkpoint 与 hard contract 在装载/哈希期间发生变化时拒绝。

receipt 同样使用 no-clobber 原子发布；第二次取证同一路径会失败并保留首次 bytes。这份 receipt 只证明
checkpoint 身份、finite 与训练合同谱系，不是 q10/q50 判卷、行为晋级或自动 stop/promote。

rolling continuation 不能直接把 binding path 交给通用 runtime。专用 runner 只接受冻结 YAML 的 `job-id`
与 parent+offset 物化后的 absolute milestone；它从 job 派生 Pod/run directory，并用发射时冻结的 runner SHA
重建 schema-2 launch claim。目标 Pod 的唯一 SSH 内先 O_NOFOLLOW 稳定读取现存 claim，重验 canonical digest、
job/source/Pod/GPU/run directory 和 claim/binding argv override。它不修改活 trainer checkout；同一 SSH 把
reviewed `lean_queue_runtime.py` 以 SHA 命名、O_EXCL/no-replace 地物化为只读 content-addressed snapshot，
已有 snapshot 只接受逐字节相同，symlink/race/mismatch 全拒绝。runtime 进程内再对拍 expected claim/job/
自身 SHA，最后才调用上面的 no-clobber attestor，并把 runtime SHA 写进 receipt。run directory、source、
recipe 或 slot 漂移都会在 checkpoint 打开前拒绝。这个 wrapper 没有 PID、checkpoint path、stop、signal
或 retry 输入面。

## 启动 phase telemetry 与冷启动边界

仅当 queue claim/binding 两个 override 同时存在时，trainer 才输出机器可读
`LEAN_QUEUE_PHASE` JSON：`hydra_resolved`、`app_started`、`log_dir_bound`、`scene_import_start`、
`scene_import_done`、`hard_contract_written`（含 adjacent contract SHA）。既有 exact-PGID launcher 看到真实
`Learning iteration` 并成功返回后，queue harness
才把 `phase=first_iter` 追加到 `.launch` sidecar；冻结 launcher bytes 不变。普通训练不增加这些 marker。

watchdog 的 signal identity 也不能只信启动时 PGID 数字。launcher 在 spawn 后由 source-pinned helper
双读 `/proc` starttime 并核对 `getpgid`，sidecar 引用 no-clobber leader/pre-TERM/pre-KILL evidence。TERM
等待枚举整组而非只轮询 leader；KILL 只接受 pre-TERM exact member set 的子集。leader 在 TERM 前消失、
PID reuse、读中漂移或新成员加入时不得 signal，并以 rc121/122 要求人工复核。leader 在 TERM 后退出但
已绑定 child 仍活时，可以只在 child 的 PID/starttime/PGID 仍与 pre-TERM evidence 一致时收口。

这组 marker 可以区分 Hydra、Kit app、A3 scene/URDF import 和首个 update。main 已另有独立的
per-source+Pod+physical-GPU [`boot-warmup`](../DEFINITIONS.md#boot-warmup) namespace，以及 content-bearing
日志默认 180 秒 stale watchdog；warmup 明确不继承科学 run 的 P1 binding path，也不冒充 checkpoint
证据。P1 保持 generic launcher/watchdog 语义，不增加 retry 或 broad signal。

`boot-warmup` 的 1 environment 只探最小 importer/cache 路径，不能授权正式规模。P1.1 的
[`full-scene-probe`](../DEFINITIONS.md#full-scene-probe)另沿同一 source/Pod/GPU、完整 scene recipe 与原
`num_envs` 派生两次 update 的隔离运行；它只在 launcher 看到首个 `Learning iteration` 后报告
`first_iteration_observed=true`。probe claim 明写 `not_science=true / attestable=false / promotable=false`，
并将唯一内部 milestone 固定为 `[1]`。它不生成 science 的 `run_binding.json` 或 milestone receipt，而使用
独立 `full_scene_probe_binding.json`：trainer 仍逐项绑定 claim argv、RSL log、source、GPU 与自身 PID/
starttime；额外记录同 PGID 的 supervisor leader，并要求其 `/proc` argv 精确等于 claim-bound prefix 加完整
trainer argv。普通 attestor 看到 `attestable=false` 必须拒绝。

supervisor 是 `setsid` 建立的 leader，trainer 是同组 child；supervisor 只 `wait`，不发 `TERM/KILL`，自然
结束后 no-clobber 写 `full_scene_probe_exit.json`，明确区分 `normal_exit+exit_code` 与 `signal+signal`。独立
`finalize-full-scene-probe` 只访问调用者选择的 Pod，并要求 trainer/supervisor 及整个原 PGID 均自然消失；
仍 live/orphan 只返回 not-ready。它还比较 current-YAML expected claim SHA，并由 **finalizer runtime 本身**
从 immutable claim 导出 receipt、target 与 donor，重算两棵当前文件树库存和 URDF mesh 引用闭包，再与
claim-bound source-asset receipt 逐项比较。shell 中的 source-asset doctor 只是更早的诊断门，它的成功输出不是
终档授权依据；直接调用 runtime `finalize()` 也不能绕过当前资产重验。PID reuse、signal/nonzero rc、
fatal（含 NaN/Inf/Killed）、phase 缺失、`model_1` 缺失/iteration 错/nonfinite/causal-lineage、
contract/claim/source/asset 漂移均形成不可解锁的终档失败。通过结果的当前闭包证据
(`source_asset_receipt.current_closure`) 分别记录 target/donor 的观测路径、tree-content SHA、文件数/字节数、
URDF 闭包和 donor clean commit 状态，使 immutable result 能直接回答终档当下读到了什么。
通过或失败都写 immutable
`probe_result.json`；重复调用只接受 byte-identical receipt，且从不自动重试、改 queue status 或晋级。
probe execute 的容量快照只读取 selected dispatch Pod/GPU；普通 science fill 仍保留 all-Pod claim 防重复。

P1.5 后，probe row 必须显式 `runtime_binding=true`，并在 immutable argv 中各出现一次
`task.actor_obs_contract=deploy_parity_face179`、`task.plant.zero_joint_friction=true`、
`task.physical_ball=true` 与原 `num_envs`。trainer 的 `scene_import_done` payload 是运行事实而非配置复述：记录
实际 environment 数、物理球开关及 `pb_ball/pb_table/pb_table_visual` 三个实体。finalizer 还从 exact clean
source checkout 直接 file-load dependency-light schema-3 validator，复核 hard contract 的 actor observation
合同与 31/31 零摩擦；不允许 package import 间接启动 Kit/Omni。

若 trainer 在 supervisor 写 exit receipt 前由 launcher 判定 pre-marker/boot-stale/boot-timeout 终止，launcher
只有在精确组收口成功后才追加 bound `terminal_kind/terminal_exit_code`。finalizer 消费该 sidecar 与原 leader
identity evidence 时只能生成 `failed/unlock_authorized=false`，缺字段、identity 仍活或 PGID 未空则保持
not-ready。PID 数字被新 starttime 复用不再当 run failure；它只证明原 identity 已消失，随后仍需两次稳定
PGID 扫描。调用者不再选择 source-asset receipt 路径，而从 immutable claim 唯一导出，避免错误 CLI 把结果
namespace 烧死；并发相同 finalization 只接受原子胜者的 byte-identical bytes。

`main@caeb9ad` 的历史 strict probe 在 shell wrapper 中确实重算并通过了当时的 target/donor
doctor，但当时的 runtime/result 还没有上述 in-process `current_closure`。因此该 receipt 仍按其原语义作为
4096-environment 启动/终档证据，不追认为已经运行了新授权逻辑；只有绑定新 exact source 且结果实际
带 `current_closure` 的新 attempt 才能声明这项能力。checkpoint 文件名与内嵌 iteration、fresh lineage 也都只
接受 plain integer；JSON boolean 不得借 Python 中 `True == 1` 混过终档。

## 源码复现

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_exact_process_group.py \
  hope_training/whole_body_tracking/tests/test_launch_kit_training_locked.py \
  hope_training/whole_body_tracking/tests/test_lean_queue_runtime.py \
  hope_training/whole_body_tracking/tests/test_full_scene_probe_runtime.py \
  tests/test_run_lean_training_queue.py \
  tests/test_run_phase1_rolling_timing_supercombo_queue.py \
  hope_training/whole_body_tracking/tests/test_training_launch_claim.py \
  hope_training/whole_body_tracking/tests/test_training_thread_caps.py \
  tests/test_lean_training_source_asset.py

python3 -m py_compile \
  scripts/run_lean_training_queue.py \
  hope_training/whole_body_tracking/scripts/exact_process_group.py \
  hope_training/whole_body_tracking/scripts/train.py \
  hope_training/whole_body_tracking/scripts/lean_queue_runtime.py \
  hope_training/whole_body_tracking/scripts/full_scene_probe_runtime.py

bash -n hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh
```

当前整合 harness/source-asset 回归结果为 `146 passed`，其中 full-scene terminal 专项 `39 passed`。
watchdog identity 专项覆盖 PID reuse、双读漂移、leader 先退但已绑定
child 残留、新成员加入、空组与正常 stale/hard timeout；其余负测覆盖每臂单一原子 SSH、内置 doctor 门在
容量/claim 前执行、多臂每臂
各一次 transaction，以及 fake log dir、binding/receipt overwrite、静态及读中 PID reuse、
source dirty/YAML verifier 漂移、不可达 terminal milestone、warmup/legacy capability 串线、filename/embed
iteration 错位、nested float/complex NaN、hard-contract SHA 错绑、launch-claim lineage 错绑，以及 full-scene
probe 的环境数漂移、科学 namespace 串线、错确认词、reserved Pod、placeholder/reuse，以及 terminal still-live、
PID reuse、nonzero/signal exit、fatal、missing/NaN/wrong-iteration model、contract/claim/source drift 和结果替换。
