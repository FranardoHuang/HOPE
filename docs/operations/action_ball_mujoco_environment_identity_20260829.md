# ActionBall MuJoCo FullMDP 基础环境重建

Status: Partial

这份操作只回答一个问题：在一台已有合适NVIDIA驱动、可访问Python包源的Linux机器上，怎样重建
Pod1当前MuJoCo FullMDP基础venv。它不提供ignored/private机器人资产，不接受任何第三方许可，也不证明
学习、跨仿真一致性、部署或真机安全。

## 唯一环境输入

- Python：CPython `3.12.3`；
- 完整非editable解析闭包：
  [`configs/action_ball_mujoco_base_venv_constraints_20260829.txt`](../../configs/action_ball_mujoco_base_venv_constraints_20260829.txt)，
  `133`行，SHA-256=`6e26d1e0d8befbe7199d016751b13c66bc81aa36c76a059c15095e1a51526bb1`；
- 直接请求项只有`mjlab==1.5.3`与验证工具`pytest==9.1.1`；先单独bootstrap
  `pip==26.2.1`和`setuptools==83.0.0`。

以上来自Pod1现役`/workspace/mjlab_venv`的
`python -m pip freeze --all | LC_ALL=C sort`。Pod1观察到`/usr/bin/python3.12` SHA-256为
`1d3cf64f…74fd5`，但它是发行版构建产物，不作跨机器硬门；Python语义版本和解析闭包才是这里需要独立核对的
事实。把解释器文件SHA强行相等当“安全”会拒绝合法重建，却不会增加学习或physics信息。

## Fresh venv步骤

以下命令只允许指向尚不存在的新目录；不要覆盖现役venv，也不要在训练checkout内创建环境：

```bash
MUJOCO_FULLMDP_VENV=/workspace/mjlab_venv_fresh
test ! -e "$MUJOCO_FULLMDP_VENV"
/usr/bin/python3.12 --version
/usr/bin/python3.12 -m venv "$MUJOCO_FULLMDP_VENV"
"$MUJOCO_FULLMDP_VENV/bin/python" -m pip install --upgrade \
  pip==26.2.1 setuptools==83.0.0
"$MUJOCO_FULLMDP_VENV/bin/python" -m pip install \
  -c configs/action_ball_mujoco_base_venv_constraints_20260829.txt \
  mjlab==1.5.3 pytest==9.1.1
"$MUJOCO_FULLMDP_VENV/bin/python" -m pip check
diff -u configs/action_ball_mujoco_base_venv_constraints_20260829.txt \
  <("$MUJOCO_FULLMDP_VENV/bin/python" -m pip freeze --all | LC_ALL=C sort)
```

如果package index已经发布不同候选或缺少锁内wheel，安装必须fail closed；不要删约束、换版本或从另一个
机器复制`site-packages`来让命令通过。CUDA 13相关wheel仍要求host NVIDIA driver兼容；driver身份应作为
机器事实另行记录，不能从Python lock推出。

## Production runtime的两层身份

基础lock中有ambient `mujoco-warp==3.10.0.3`和`rsl-rl-lib==5.4.0`，因为它们属于MJLab解析闭包；它们
不是portable FullMDP实际执行真源。one-shot launcher会在fresh run-owned `runtime_site`中绑定并独立核验：

- EPA48 `mujoco-warp==3.10.0.3+hope.epa48.1` exact wheel；
- `rsl-rl-lib==3.1.2` exact wheel；
- base venv中的`mjlab==1.5.3` selected tree。

这个分层不维护第二套训练配方：base lock负责“环境能被重建”，
[`runtime_stack`](../DEFINITIONS.md#mujoco-fullmdp-runtime-stack)负责“这一条run实际导入了谁”。
只在环境创建时比较一次完整lock；每个optimizer update重复比较同一写入方产物既不独立，也不携带新信息。

## Repo到可运行机器的剩余步骤

1. 按[`setup_local_sync.md`](setup_local_sync.md)恢复A3P0807的92个ignored mesh，以及exact EPA48、
   RSL-RL 3.1.2 wheel/receipt；
2. 在fresh exact checkout分别运行host测试；
3. 用`launch_mujoco_full_mdp_successor.py --dry-run`核显式Python、plant、ready pose、GPU UUID和run root；
4. 确认机器有writable `/workspace`，并为目标GPU一次性执行
   `LOCK_PATH=/tmp/hope_lean_queue_gpuN.lock; test ! -e "$LOCK_PATH";`
   `(umask 077; set -o noclobber; : > "$LOCK_PATH")`，再核它是owner-only regular file且不是symlink；
5. 只在空闲GPU和fresh namespace启动fixed-action或训练。

第4步必须只对不存在的pathname执行；若文件已存在就停下核进程，而不是用`install`重写inode。`/tmp`在重启
后通常清空，只能在确认该GPU没有trainer后重新provision。

因此“repo + 本文package来源 + setup_local_sync列出的外部字节”具备可复现步骤；“纯Git clone立即训练”仍为
`FALSE`。缺失private/ignored输入必须报告为`PARTIAL`，不能用更多success Gate掩盖。

`2026-08-29`在Pod1以validated training checkout=`8a57a522`建立fresh exact目录，恢复92个mesh并复核
root XML SHA=`7bbda723…bcae1`、payload=`92 files / 25,331,878 bytes`、RSL3 wheel SHA=
`40686735…e06d`；完整Mu successor launcher dry-run RC0。真实CUDA fixed-action因三卡占用仍`未测`。
