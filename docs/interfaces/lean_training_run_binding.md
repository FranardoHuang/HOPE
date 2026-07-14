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

## 启动 phase telemetry 与冷启动边界

仅当 queue claim/binding 两个 override 同时存在时，trainer 才输出机器可读
`LEAN_QUEUE_PHASE` JSON：`hydra_resolved`、`app_started`、`log_dir_bound`、`scene_import_start`、
`scene_import_done`。既有 exact-PGID launcher 看到真实 `Learning iteration` 并成功返回后，queue harness
才把 `phase=first_iter` 追加到 `.launch` sidecar；冻结 launcher bytes 不变。普通训练不增加这些 marker。

这组 marker 可以区分 Hydra、Kit app、A3 scene/URDF import 和首个 update。main 已另有独立的
per-source+Pod+physical-GPU [`boot-warmup`](../DEFINITIONS.md#boot-warmup) namespace，以及 content-bearing
日志默认 180 秒 stale watchdog；warmup 明确不继承科学 run 的 P1 binding path，也不冒充 checkpoint
证据。P1 保持 generic launcher/watchdog 语义，不增加 retry 或 broad signal。

`boot-warmup` 的 1 environment 只探最小 importer/cache 路径，不能授权正式规模。P1.1 的
[`full-scene-probe`](../DEFINITIONS.md#full-scene-probe)另沿同一 source/Pod/GPU、完整 scene recipe 与原
`num_envs` 派生两次 update 的隔离运行；它只在 launcher 看到首个 `Learning iteration` 后报告
`first_iteration_observed=true`。probe claim 明写 `not_science=true / attestable=false / promotable=false`，
不生成本接口的 `run_binding.json` 或 milestone receipt，因此不能通过本接口被追认为科学 checkpoint。
probe execute 的容量快照只读取 selected dispatch Pod/GPU；普通 science fill 仍保留 all-Pod claim 防重复。

## 源码复现

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_lean_queue_runtime.py \
  tests/test_run_lean_training_queue.py \
  hope_training/whole_body_tracking/tests/test_training_launch_claim.py \
  hope_training/whole_body_tracking/tests/test_training_thread_caps.py

python3 -m py_compile \
  scripts/run_lean_training_queue.py \
  hope_training/whole_body_tracking/scripts/train.py \
  hope_training/whole_body_tracking/scripts/lean_queue_runtime.py

bash -n hope_training/whole_body_tracking/scripts/launch_kit_training_locked.sh
```

当前 focused 结果为 `76 passed`。负测覆盖每臂单一原子 SSH、内置 doctor 门在容量/claim 前执行、多臂每臂
各一次 transaction，以及 fake log dir、binding/receipt overwrite、静态及读中 PID reuse、
source dirty/YAML verifier 漂移、不可达 terminal milestone、warmup/legacy capability 串线、filename/embed
iteration 错位、nested float/complex NaN、hard-contract SHA 错绑、launch-claim lineage 错绑，以及 full-scene
probe 的环境数漂移、科学 namespace 串线、错确认词、reserved Pod、placeholder/reuse。
