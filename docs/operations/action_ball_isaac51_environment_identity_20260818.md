# ActionBall Isaac 5.1 环境身份合同

> 状态：`PASS-current-Pod-runtime / PARTIAL-fresh-machine-availability / branch-scoped`
> 更新：2026-08-29
> 来源：Jiayi 的 `ENVIRONMENT_REPRODUCTION.md`、Pod2 import/ABI/AppLauncher验证，以及
> `origin/build_4@324e60d1`与Pod1实际路径/import复核。原文仍在Pod1 Jiayi checkout，SHA-256为
> `b461ed8d98fad352680abf16833f6c706f1c9fde8af8074ec5d27e016ecf9346`、`17,075` bytes；仓库已把其中
> FullMDP共用的软件身份、安装顺序与83项非editable约束吸收到本页和tracked constraints。旧Hitter资产与
> checkpoint只用于Build4历史复现，不混入FullMDP运行合同。
> 本页不改变 `origin/main:docs/NOW.md` 的项目优先级。

## 为什么必须换环境

旧 Pod 环境与 Jiayi/build_2 不是“同一环境的小版本差异”。旧环境使用 Isaac Sim 4.5、IsaacLab 2.1、
Python 3.10、RSL-RL 2.3；build_2 使用 Isaac Sim 5.1、IsaacLab `8320e0be…`、Python 3.11、
RSL-RL 3.1.2/TensorDict。前者的 PPO rollout 接口是 `act(obs, critic_obs)`，后者是
`act(obs: TensorDict)`；episode/reset/recorder/observation-history lifecycle 也不同。兼容分支只能让旧栈
启动，不能证明 Reward、rollout、normalization、W&B 字段或学习曲线等价。

## 精确身份

| 项 | 精确值 |
| --- | --- |
| Jiayi 训练提交 | `35e65eb7f3e1bf21fa5719aa0c0a7a90b830b836` |
| 历史 FullMDP 环境验证提交 | `758e88eefe6e9ce625ae57f5a732bc2024b7c74a` |
| 下一条4096执行提交 | 由最终一次性wrapper钉定exact clean commit；本页不维护会自引用漂移的SHA |
| Isaac Sim | `5.1.0-rc.19+release.26219.9c81211b.gl` |
| IsaacLab | `8320e0be5c0f2def58d5b19d308c6d2539d47cb2` |
| Python | `3.11.13` |
| PyTorch | `2.7.0+cu128` |
| RSL-RL | `3.1.2` |
| TensorDict / TorchRL | `0.10.0 / 0.10.1` |
| gymnasium / wandb / numpy | `1.2.1 / 0.25.1 / 1.26.4` |
| 已验证OS/GPU | Ubuntu 22.04.5 + RTX 5070 Ti Laptop；Ubuntu 24.04.3 + RTX 5090 |
| Isaac Sim下载件 | `isaac-sim-standalone-5.1.0-linux-x86_64.zip`，`8,768,419,777` bytes |
| 外部venv constraints | `configs/action_ball_isaac51_external_venv_constraints_20260829.txt`，SHA-256 `6b1c9a34cd3ac40a970c5bf53330069b420f5f993d298b34ec72ff55ccea81e9` |

必须在实际进程中看到：

```text
PPO.act(self, obs: TensorDict)
PPO.process_env_step(self, obs: TensorDict, rewards, dones, extras)
```

如果看到 `act(obs, critic_obs)`，立即停止；不能用代码兼容把它升级成等价证据。

## Build4 本机与 Pod 曲线差异：先关闭运行身份漂移

`origin/build_4`的exact commit是`324e60d120856556875d65eb26931c6f89d7f5de`。该branch自己的README目标仍是
Isaac Sim `4.5.0`、IsaacLab `2.1.0`、Python `3.10`，且没有提交Jiayi的后续环境说明或package lock。
它的`setup_train_env.sh`只按路径存在性自动选择执行面，不核对版本、commit、ABI或PPO签名。因此同一个
Build4 commit在两台机器上可以静默成为两个训练系统。当前branch已删除该fallback authority：
`setup_train_env.sh`只接受caller或ignored `setup_train_env.local.sh`显式给出的Python/IsaacLab/site路径，
缺失或路径无效直接返回失败；tracked `setup_train_env.local.example.sh`给出5.1模板。

2026-08-29在当前Pod1按该脚本默认选择顺序做live import，结果与当前受控FullMDP进程如下：

| 项 | Build4脚本在Pod1默认命中 | 当前受控FullMDP进程 |
| --- | --- | --- |
| Python入口 | `/workspace/hope_isaac_venv/bin/python` → `/usr/bin/python3.10` | `/workspace/isaacsim-5.1.0/python.sh` |
| IsaacLab | `/workspace/IsaacLab@21f71363…` | `/opt/IsaacLab-8320e0be@8320e0be…` |
| Python / Isaac Sim / IsaacLab | `3.10.18 / 4.5.0.0 / 0.36.21` | `3.11.13 / 5.1路径 / 0.54.2` |
| RSL-RL / TensorDict / TorchRL | `2.3.1 / 0.9.1 / missing` | `3.1.2 / 0.10.0 / 0.10.1` |
| gymnasium / wandb | `1.3.0 / 0.28.0` | `1.2.1 / 0.25.1` |
| PPO接口 | `act(obs, critic_obs)` | `act(obs: TensorDict)` |

所以，若Jiayi本机Build4跑在5.1/RSL3而Pod直接`source setup_train_env.sh`，两条曲线不是同环境对照；这一层
漂移已经坐实。尚不能声称它解释了历史曲线差异的全部，因为缺少Jiayi本机和历史Pod run的actual
`setup_train_env.local.sh`、argv、asset/motion/checkpoint SHA、seed和runtime receipt。下一次Build4复现不再
使用path autodiscovery作权威：显式指定Python与IsaacLab exact路径，在进程内记录上述版本、两个PPO签名和
全部输入SHA，不匹配就不启动。这里需要的是一个简单的唯一运行身份，不是更多事后success/safety Gate。

## Pod2 路径

```text
Isaac Sim: /workspace/isaacsim-5.1.0
IsaacLab:  /workspace/IsaacLab-8320e0be
venv:      /workspace/hope_drone_venv
Git root:  /workspace/franco/mktemp/fullmdp-isaac51-rsl3-git.20260818T091500CST
```

`PYTHONPATH` 顺序必须为 working-tree source、Python3.11 venv site-packages、IsaacLab 的
`isaaclab/isaaclab_tasks/isaaclab_assets/isaaclab_rl`。训练由
`/workspace/isaacsim-5.1.0/python.sh` 启动，不能调用旧 `/workspace/hope_isaac_venv/bin/python`。

## 新机器部署边界、重建配方与实测

`git clone`本身不是完整Isaac安装包，也不应成为一个：Isaac Sim二进制受EULA约束，split-rubber USD与
A3P0807 Mu meshes是ignored/private资产。仓库负责的第一性合同是另一层：所有外部输入用
显式路径、版本和SHA绑定；代码只从一个clean commit启动；缺任何一项都在scene/training前失败，不能靠
path autodiscovery悄悄换成另一套ABI。

仓库现在闭合了可由公开软件重建的部分：IsaacLab固定到`8320e0be…`；Python环境使用
[`action_ball_isaac51_external_venv_constraints_20260829.txt`](../../configs/action_ball_isaac51_external_venv_constraints_20260829.txt)
的83项exact约束，并保持Jiayi已验证的分阶段安装顺序；Ubuntu Noble的headless GL直接来自系统
`libopengl0=1.7.0-1build1`和`libglu1-mesa=9.0.2-1.1build1`。一个最小重建序列是：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$STACK_ISAACLAB"
git -C "$STACK_ISAACLAB" checkout --detach 8320e0be5c0f2def58d5b19d308c6d2539d47cb2
/usr/bin/python3.11 -m venv "$STACK_VENV"
CONSTRAINTS="$STACK_REPO/configs/action_ball_isaac51_external_venv_constraints_20260829.txt"
"$STACK_VENV/bin/python" -m pip install -c "$CONSTRAINTS" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128
"$STACK_VENV/bin/python" -m pip install -c "$CONSTRAINTS" \
  gymnasium==1.2.1 wandb==0.25.1 hydra-core==1.3.2 pillow==11.3.0 \
  packaging==23.2 GitPython==3.1.59 onnx==1.22.0 cloudpickle==3.1.2 \
  orjson==3.12.0 importlib-metadata==9.0.0 toml==0.10.2 prettytable==3.3.0 \
  flatdict==4.1.0 h5py==3.16.0 moviepy==2.2.1 tensorboard==2.21.0
"$STACK_VENV/bin/python" -m pip install -c "$CONSTRAINTS" --no-deps \
  pyvers==0.1.0 tensordict==0.10.0 torchrl==0.10.1 rsl-rl-lib==3.1.2
"$STACK_VENV/bin/python" -m pip install --no-deps \
  -e "$STACK_ISAACLAB/source/isaaclab" \
  -e "$STACK_ISAACLAB/source/isaaclab_assets" \
  -e "$STACK_ISAACLAB/source/isaaclab_contrib" \
  -e "$STACK_ISAACLAB/source/isaaclab_tasks" \
  -e "$STACK_ISAACLAB/source/isaaclab_rl" \
  -e "$STACK_ISAACLAB/source/isaaclab_mimic"
"$STACK_VENV/bin/python" -m pip install -c "$CONSTRAINTS" --no-deps numpy==1.26.4
```

新机仍需人类合法下载上述Isaac Sim zip并在安装期建立一次EULA/隐私authority；仓库不能替人同意许可。
Ubuntu Noble安装两项系统GL包后，launcher检查规范regular file与direct SONAME并把观察到的路径/SHA写进
dry-run及完成JSON，不再把Noble字节SHA误当跨Ubuntu版本不变的学习身份。不同受支持发行版可以有不同GL
bytes，但必须通过真实Kit probe；这比复制Franco私有GL目录或继续堆一个无信息Gate更直接。

2026-08-29在Pod1从远端重新克隆
`Franco_codex/actionball-fullmdp-v2-20260821@e3ef4e98e903adc006775c6b23b97005f1473df0`，不复用原开发
checkout，完成了以下闭环：

- clean checkout上用`launch_isaac_full_mdp_successor.py`显式传入Isaac Python、Kit Python、
  IsaacLab root、Python 3.11 site、USD、RSL wheel、GPU UUID和CPU affinity；`--dry-run`通过；
- focused launcher/runtime-inventory测试为`52 passed in 10.96 s`；
- GPU1真实启动Kit/PhysX并自然完成`512×48×31` fixed-action probe，`done/time-out=0/0`，日志中的
  traceback/CUDA/OOM/runtime/fatal/segfault扫描为0，退出后GPU释放且checkout仍clean。

完整路径与字节receipt见
[`action_ball_isaac51_fresh_clone_deployment_20260829.json`](../../configs/action_ball_isaac51_fresh_clone_deployment_20260829.json)。
这把当时源码“能否接入该Pod已经合法准备好的精确外部runtime”裁决为历史`PASS`；它**不**宣称纯Git
自包含安装，也不授权学习、physics parity、promotion、部署或真机。后续复核确认G05已记录EULA接受、
当前Pod长期运行同一隐私配置，故删除`e9823e90…`错误新增的两个per-run确认Gate；OpenGL/GLU目录和真实库
路径、观察SHA与direct SONAME链接仍纳入pre-root身份。本地36项launcher测试通过。当前source
`4cd30d63…`真实Kit fixed-action probe已在GPU0自然完成：`512×48×31`、`done/time-out=0/0`，action/state
SHA与历史受控probe逐字相同，退出后GPU释放；这仍只是工程/运行身份证据。

新空白机器的**可获得性**仍为`PARTIAL`，但范围已缩小：split USD、A3P0807 Mu meshes、合法Isaac下载和
private repo/artifact凭据仍需团队持久locator；RSL wheel有公共安装来源与现存exact SHA，Python环境已有
tracked constraints，OpenGL/GLU有系统包来源。历史Jiayi本机和旧Pod曲线不保证逐位可比，也不阻塞当前
Pod受控运行；真正要比较曲线时必须同时保存source、runtime、全部输入、seed和实际argv receipt。

## Git 与非 Git 资产

代码已经通过 Git branch `Franco_codex/actionball-isaac51-rsl3-20260818` 传输；`758e88e…`只保留为
历史环境验证receipt，不再充当下一条4096的源码身份。下一条运行只能消费最终wrapper内钉定、独立复核
且能在remote clean clone复现的唯一commit；不能把浮动branch HEAD或本页历史SHA代签。此前不能只用
Git 的原因不是 Git 本身，而是 FullMDP 长期处于未提交 WIP，并依赖
`.gitignore` 排除的专有机器人资产和大型 checkpoint。现在把两者分开：代码由 clean Git tree
证明，外部资产由独立路径、字节和派生关系证明。

本次 fresh FullMDP 不加载 legacy `model_21800.pt`；它需要 split-rubber USD：

```text
/workspace/franco/runtime_assets/a3p0807_split_rubber_diagnostic_v3/model.usd
SHA256=a3cd382943ff9f70beecf88c729a6cc1c052a3c0a0cbffe91003ec319ab78140
```

73条 measured motion 与 manifest 在 Git tree 中。legacy Hitter/build_2 checkpoint 和原始 A3
URDF/meshes仍是非 Git 资产；如复现 Jiayi 的 Hitter baseline，必须按其附件清单单独复制和校验，
不能把 FullMDP 的 split USD 冒充那份资产。

## 已验证与未验证

- `PASS`：package版本、import origin、两个 PPO 签名、Isaac Sim build、clean build_2 `6144×1`。
- `PASS`：FullMDP IsaacLab8320 lifecycle、exact Kit cfg/train wiring、N=2 canonical reset和forced selected reset。
- `PASS-direct`：FullMDP RSL3 optimizer boundary/WAL adapter；v11 compact joint-safety结构顺序已通过host反例。
- `PASS-historical-N2`：FullMDP RSL3真实 `N=2×2`曾闭合optimizer/WAL，只作工程证据。
- `PASS-Pod-CUDA`：clean Git `2c8ef444…`在Jiayi Python3.11/Torch2.7-cu128完成LM info/NaN/finite-overflow三参数，CUDA context存活。
- `PASS-environment / FAIL-first-4096-entry`：commit `5ee1ffa6…` 的first one-shot通过GPU preexec、sealed RSL和真实Kit Python身份，但身份代码在`AppLauncher`前导入Torch/RSL，Kit startup后约0.34秒segfault；Hydra解析成功，scene/PPO/WAL零调用。successor必须把class/source attestation移到AppLauncher成功后的同一Kit进程，不能用pre-App import代签。
- `PASS-repo-to-runtime-historical`：remote clean clone `e3ef4e98…`完成dry-run、52项focused test和真实Kit/PhysX fixed-action probe；边界只覆盖“该Pod已有精确外部runtime时仓库能启动”。
- `PASS-current-source-contract / PASS-real-probe`：一次性runtime authority不再错误地逐run确认；current-source launcher测试与真实Kit fixed-action probe通过，GL观察身份写入输出。
- `PASS-public-software-recipe / PARTIAL-private-assets`：IsaacLab、83项Python约束、RSL版本与系统GL来源已tracked；合法Isaac下载、split USD、92项Mu meshes和private访问仍需外部提供。
- `未测`：可信4096 A1000趋势、C、完整checkpoint/restore、跨机器逐位曲线一致性。

环境 `PASS` 只回答“代码在同一软件栈上执行”，不回答 Reward 是否合理、是否可学或跨机逐位相同。
后续 FullMDP 运行继续标记 `diagnostic_unauthorized=true`。
