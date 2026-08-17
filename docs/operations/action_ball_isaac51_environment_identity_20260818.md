# ActionBall Isaac 5.1 环境身份合同

> 状态：`PASS-environment / branch-scoped`
> 更新：2026-08-18
> 来源：Jiayi 的 `ENVIRONMENT_REPRODUCTION.md` 与 Pod2 实际 import/ABI/AppLauncher 验证。
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
| 本分支 FullMDP 执行提交 | `758e88eefe6e9ce625ae57f5a732bc2024b7c74a` |
| Isaac Sim | `5.1.0-rc.19+release.26219.9c81211b.gl` |
| IsaacLab | `8320e0be5c0f2def58d5b19d308c6d2539d47cb2` |
| Python | `3.11.13` |
| PyTorch | `2.7.0+cu128` |
| RSL-RL | `3.1.2` |
| TensorDict / TorchRL | `0.10.0 / 0.10.1` |
| gymnasium / wandb / numpy | `1.2.1 / 0.25.1 / 1.26.4` |

必须在实际进程中看到：

```text
PPO.act(self, obs: TensorDict)
PPO.process_env_step(self, obs: TensorDict, rewards, dones, extras)
```

如果看到 `act(obs, critic_obs)`，立即停止；不能用代码兼容把它升级成等价证据。

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

## Git 与非 Git 资产

代码已经通过 Git branch `Franco_codex/actionball-isaac51-rsl3-20260818` 传输，remote clean clone HEAD
为 `758e88e…`。此前不能只用 Git 的原因不是 Git 本身，而是 FullMDP 长期处于未提交 WIP，并依赖
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
- `PASS-direct`：FullMDP RSL3 optimizer boundary/WAL adapter。
- `WAIT-resource`：FullMDP RSL3真实 `N=2×2`，只等Pod2 GPU0自然空闲。
- `未测`：A1000学习趋势、C、完整checkpoint/restore、MuJoCo GPU semantic runtime。

环境 `PASS` 只回答“代码在同一软件栈上执行”，不回答 Reward 是否合理、是否可学或跨机逐位相同。
后续 FullMDP 运行继续标记 `diagnostic_unauthorized=true`。
